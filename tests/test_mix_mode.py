# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Tests for Long-Form Mix Mode analysis, planning, API/CLI branching, and stills-only runs."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import yaml
from typer.testing import CliRunner

from mvdirector import api
from mvdirector.cli import app
from mvdirector.comfy.client import ComfyClient
from mvdirector.config import ProjectConfig, load_config
from mvdirector.errors import ConfigError, MVDError
from mvdirector.models import AssetKind, AssetStatus, ProjectMode, ShotType
from mvdirector.planning import prompts
from mvdirector.planning.mix_planner import build_mix_manifest, quantize_loop_clip_seconds
from mvdirector.project import Project
from mvdirector.review import policy as review_policy
from tests.conftest import SR, FakeComfy, click_track, write_workflows


class _FakeProvider:
    def __init__(self, fake: FakeComfy) -> None:
        self.fake = fake

    @contextmanager
    def lease(self, project: Project):
        yield ComfyClient(
            project.config.comfy, transport=self.fake.transport(), sleep=lambda _: None
        )


class _ScriptedMixLLM:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.prompts: list[str] = []

    def complete_json(self, *, system: str, content, schema):  # noqa: ANN001
        self.calls.append(system)
        self.prompts.append(" ".join(getattr(block, "text", str(block)) for block in content))
        return schema.model_validate(
            {
                "style": "Anamorphic 35mm atmospheric architectural cinema",
                "negative_prompt": "people, faces, text",
                "chapters": [
                    {
                        "id": "ch01",
                        "environment": "Flooded obsidian monolith hall",
                        "palette": "deep cyan and silver",
                        "lighting": "cool overhead shafts",
                        "motion_style": "Slow steady forward drift over ripples",
                        "clip_prompts": [
                            "Wide archway over dark mirror water",
                            "Medium reflection of cyan light shafts on wet stone",
                        ],
                    },
                    {
                        "id": "ch02",
                        "environment": "Sun-drenched desert mirror canyon",
                        "palette": "burnt amber and copper",
                        "lighting": "warm low-angle sunset",
                        "motion_style": "Brisk light sweep across sandstone",
                        "clip_prompts": [
                            "Wide red sandstone canyon with rotating brass ring",
                            "Medium copper light caustics across desert rock",
                        ],
                    },
                ],
            }
        )


def _make_mix_project(
    tmp_path: Path,
    *,
    visual_source: str = "clips",
    with_tracklist: bool = True,
) -> Project:
    root = tmp_path / "mix_proj"
    (root / "inputs").mkdir(parents=True)
    # Chapter 1 (0..16s): quieter 100 BPM; Chapter 2 (16..32s): louder 140 BPM. NO vocals or lyrics!
    part1 = 0.35 * click_track(seconds=16.0, bpm=100.0)
    part2 = 1.00 * click_track(seconds=16.0, bpm=140.0)
    sf.write(root / "inputs" / "mix.wav", np.concatenate([part1, part2]), SR)

    tracklist_rel: str | None = None
    if with_tracklist:
        tracklist_rel = "inputs/tracklist.json"
        (root / tracklist_rel).write_text(
            json.dumps(
                {
                    "title": "Warehouse Set",
                    "source": "weaveshift",
                    "tracks": [
                        {
                            "start": "00:00",
                            "title": "Deep Opening",
                            "artist": "Klang",
                            "mood_tags": ["deep", "misty"],
                            "lyrics": "Drifting through obsidian water under silver skylights",
                        },
                        {
                            "start": "00:16",
                            "title": "Peak Igniter",
                            "artist": "Nova",
                            "bpm": 174.0,  # Explicit DnB BPM must not be octave-folded
                            "mood_tags": ["kinetic", "industrial"],
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )

    # Mix mode only needs t2i (and i2v when visual_source == 'clips') — omit still & lipsync!
    wf_names = {"t2i": ".png"} if visual_source == "stills" else {"t2i": ".png", "i2v": ".mp4"}
    write_workflows(root / "workflows", wf_names)

    cfg = ProjectConfig(title="Warehouse Mix", mode=ProjectMode.MIX)
    cfg.inputs.tracklist = tracklist_rel
    cfg.llm.enabled = False
    cfg.qc.vision_enabled = False
    cfg.qc.on_missing_dependency = "skip"
    cfg.mix.visual_source = visual_source  # type: ignore[assignment]
    cfg.mix.clips_per_segment = 2
    cfg.mix.clip_seconds = 5.0
    cfg.mix.xfade_seconds = 1.0
    cfg.mix.default_segment_seconds = 30.0
    cfg.comfy.base_url = "http://comfy.test"
    cfg.comfy.poll_interval = 0.0
    cfg.comfy.workflows = {name: f"{name}.bindings.json" for name in wf_names}
    (root / "project.yaml").write_text(yaml.safe_dump(cfg.model_dump(mode="json")))
    return Project.open(root)


def test_mix_analysis_runs_without_vocals_or_lyrics_and_preserves_per_chapter_bpm(
    tmp_path: Path,
) -> None:
    project = _make_mix_project(tmp_path, with_tracklist=True)
    _, analysis = api.analyze(project)

    assert len(analysis.chapters) == 2
    ch1, ch2 = analysis.chapters
    assert ch1.id == "ch01" and ch1.start == 0.0 and ch1.end == 16.0
    assert ch2.id == "ch02" and ch2.start == 16.0 and ch2.end == 32.0
    # Chapter 1 detected ~100 BPM; Chapter 2 preserved explicit 174 BPM without octave folding.
    assert 92.0 <= ch1.bpm <= 108.0
    assert ch2.bpm == 174.0
    assert ch2.energy > ch1.energy
    assert ch1.start <= ch1.downbeat_offset < ch1.end
    assert ch2.start <= ch2.downbeat_offset < ch2.end


def test_quantize_loop_clip_seconds_snaps_to_bar_multiples() -> None:
    # At 120 BPM (4/4), 1 bar = 2.0s. With xfade=1.0s and target=5.2s, 2 bars + 1s = 5.0s.
    dur = quantize_loop_clip_seconds(
        bpm=120.0,
        beats_per_bar=4,
        target_clip_seconds=5.2,
        xfade_seconds=1.0,
        max_clip_seconds=8.0,
    )
    assert dur == 5.0


def test_mix_planner_llm_and_compose_still_prompt_apply_distinct_chapter_palettes(
    tmp_path: Path,
) -> None:
    project = _make_mix_project(tmp_path)
    _, analysis = api.analyze(project)
    llm = _ScriptedMixLLM()
    manifest = build_mix_manifest(project.config, analysis, "Dark architectural techno", llm)  # type: ignore[arg-type]

    assert len(llm.calls) == 1  # Batched into a single LLM request
    assert "lyrics_excerpt=" in llm.prompts[0]
    assert "obsidian water under silver skylights" in llm.prompts[0]
    assert manifest.mode is ProjectMode.MIX
    assert len(manifest.chapters) == 2
    assert len(manifest.shots) == 4  # 2 chapters x 2 loop units
    assert all(s.type is ShotType.BROLL for s in manifest.shots)
    assert manifest.characters == {}

    prompt_ch1 = prompts.compose_still_prompt(manifest, manifest.shot("ch01_c01"))
    prompt_ch2 = prompts.compose_still_prompt(manifest, manifest.shot("ch02_c01"))
    assert "deep cyan and silver" in prompt_ch1
    assert "burnt amber and copper" in prompt_ch2
    assert manifest.shots_for_chapters(["ch02"]) == ["ch02_c01", "ch02_c02"]


def test_validate_and_generate_chapter_filtering_without_preflight_references(
    tmp_path: Path, sample_png: bytes, sample_mp4: bytes
) -> None:
    project = _make_mix_project(tmp_path, visual_source="clips")
    fake = FakeComfy(png=sample_png, mp4=sample_mp4)
    provider = _FakeProvider(fake)

    # Validation succeeds even though 'still' and 'lipsync' workflows do not exist!
    checks = api.validate_project(project, comfy_provider=provider)
    assert all(c.ok for c in checks), [c for c in checks if not c.ok]

    api.analyze(project)
    api.plan(project)

    # Generate ONLY chapter ch02 (2 stills + 2 videos = 4 ComfyUI jobs). No reference sheets needed!
    res = api.generate(
        project,
        [AssetKind.STILL, AssetKind.VIDEO],
        chapters=["ch02"],
        comfy_provider=provider,
    )
    assert len(fake.queued) == 4
    assert res.manifest.shot("ch01_c01").still.status is AssetStatus.PENDING
    assert res.manifest.shot("ch02_c01").video.status is AssetStatus.PASSED
    assert res.manifest.shot("ch02_c02").video.status is AssetStatus.PASSED

    with pytest.raises(MVDError, match="No chapter"):
        api.generate(project, [AssetKind.STILL], chapters=["ch99"], comfy_provider=provider)


def test_stills_only_visual_source_skips_video_diffusion_and_marks_ready(
    tmp_path: Path, sample_png: bytes
) -> None:
    project = _make_mix_project(tmp_path, visual_source="stills")
    project.config.review.mode = api.ReviewMode.STRICT
    fake = FakeComfy(png=sample_png)
    provider = _FakeProvider(fake)

    checks = api.validate_project(project, comfy_provider=provider)
    assert all(c.ok for c in checks)

    api.analyze(project)
    api.plan(project)

    # Passing [STILL, VIDEO] (what `mvd run` sends) automatically runs ONLY stills!
    res = api.generate(
        project,
        [AssetKind.STILL, AssetKind.VIDEO],
        comfy_provider=provider,
    )
    assert len(fake.queued) == 4  # 4 stills, 0 videos
    assert all(s.still.status is AssetStatus.PASSED for s in res.manifest.shots)
    assert all(s.video.status is AssetStatus.PENDING for s in res.manifest.shots)
    assert all(review_policy.is_ready(s, stills_only=True) for s in res.manifest.shots)

    # Explicitly asking for videos in stills-only mode raises a clear error.
    with pytest.raises(MVDError, match="visual_source is 'stills'"):
        api.generate(project, [AssetKind.VIDEO], comfy_provider=provider)


def test_cli_init_analyze_plan_and_status_in_mix_mode(tmp_path: Path) -> None:
    runner = CliRunner()
    proj_dir = tmp_path / "cli_mix"

    res_init = runner.invoke(app, ["init", str(proj_dir), "--mode", "mix"])
    assert res_init.exit_code == 0, res_init.output
    cfg = load_config(proj_dir / "project.yaml")
    assert cfg.mode is ProjectMode.MIX
    assert (proj_dir / "inputs" / "tracklist.json").is_file()

    # Replace starter tracklist timestamps to fit a 20s synthetic audio file.
    sf.write(proj_dir / "inputs" / "mix.wav", click_track(seconds=20.0, bpm=124.0), SR)
    (proj_dir / "inputs" / "tracklist.json").write_text(
        json.dumps(
            {
                "tracks": [
                    {"start": "00:00", "title": "Opening", "bpm": 124},
                    {"start": "00:10", "title": "Closing", "bpm": 126},
                ]
            }
        ),
        encoding="utf-8",
    )
    # Switch off LLM so CLI plan runs deterministically offline.
    cfg_raw = yaml.safe_load((proj_dir / "project.yaml").read_text())
    cfg_raw["llm"]["enabled"] = False
    (proj_dir / "project.yaml").write_text(yaml.safe_dump(cfg_raw))

    res_an = runner.invoke(app, ["analyze", "-p", str(proj_dir)])
    assert res_an.exit_code == 0, res_an.output
    assert "2 chapters" in res_an.output

    res_plan = runner.invoke(app, ["plan", "-p", str(proj_dir)])
    assert res_plan.exit_code == 0, res_plan.output
    assert "2 chapters, 6 loop units" in res_plan.output

    res_status = runner.invoke(app, ["status", "-p", str(proj_dir)])
    assert res_status.exit_code == 0, res_status.output
    assert "Mode: mix   Chapters: 2   Cached segments: 0/2" in res_status.output


def test_standard_mode_rejects_mix_only_flags(project: Project, tmp_path: Path) -> None:
    cue = tmp_path / "tracks.txt"
    cue.write_text("00:00 Track A\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="mix mode"):
        api.analyze(project, tracklist=cue)


def test_mix_security_rejects_escaping_tracklist_and_unsafe_chapter_ids(tmp_path: Path) -> None:
    from pydantic import ValidationError

    from mvdirector.models import MixChapter

    project = _make_mix_project(tmp_path)
    project.config.inputs.tracklist = "../../outside.json"
    with pytest.raises(ConfigError, match="project folder"):
        api.analyze(project)

    for bad_id in ("../escape", "ch/01", "-flag", ""):
        with pytest.raises(ValidationError):
            MixChapter(id=bad_id, index=0, start=0.0, end=10.0, title="Bad", bpm=120.0)
        with pytest.raises(ConfigError, match="chapter id"):
            project.segment_video_path(bad_id)


def test_project_relative_tracklist_override_and_empty_approve_guard(tmp_path: Path) -> None:
    project = _make_mix_project(tmp_path)
    # Write a secondary tracklist inside the project and pass it as a project-relative Path
    # while cwd is outside the project root.
    alt_rel = Path("inputs/alt_tracks.txt")
    project.write_text(alt_rel, "00:00 Alt Opening\n00:12 Alt Closing\n")

    _, analysis = api.analyze(project, tracklist=alt_rel)
    assert [c.title for c in analysis.chapters] == ["Alt Opening", "Alt Closing"]

    _, manifest = api.plan(project, overwrite=True, tracklist=alt_rel)
    assert [c.title for c in manifest.chapters] == ["Alt Opening", "Alt Closing"]

    # Calling api.approve with neither shot_ids nor chapters raises MVDError.
    with pytest.raises(MVDError, match="shot id or --chapter"):
        api.approve(project)
