# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Tests for non-lipsync video styles ('narrative' and 'visual') and vocal noise floor."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf
from typer.testing import CliRunner

from mvdirector import api, tuning
from mvdirector.audio import analysis as audio
from mvdirector.cli import app
from mvdirector.config import PlanningConfig, ProjectConfig
from mvdirector.errors import ConfigError
from mvdirector.generation.preflight import check_assets
from mvdirector.generation.runner import load_templates
from mvdirector.llm.base import LLMClient
from mvdirector.models import (
    Analysis,
    AssetKind,
    LyricLine,
    ProjectMode,
    ShotType,
    TimeSpan,
    VideoStyle,
)
from mvdirector.planning import prompts
from mvdirector.planning.planner import (
    BibleReply,
    CharacterReply,
    SetReply,
    ShotBatchReply,
    ShotCreative,
    WardrobeReply,
    build_manifest,
)
from mvdirector.project import Project


def _vocal_analysis(duration: float = 16.0) -> Analysis:
    return Analysis(
        duration=duration,
        tempo_bpm=120.0,
        beats=[0.5 * i for i in range(int(duration * 2))],
        bars=[2.0 * i for i in range(int(duration // 2))],
        bar_intensity=[0.4, 0.5, 0.7, 0.9, 0.6, 0.5, 0.8, 0.4][: int(duration // 2)],
        vocal_windows=[TimeSpan(start=0.0, end=duration)],
        lyric_lines=[
            LyricLine(start=0.0, end=8.0, text="walking through the neon rain", section="Verse 1"),
            LyricLine(start=8.0, end=16.0, text="lights flicker in the dark", section="Chorus"),
        ],
    )


class _RecordingLLM(LLMClient):
    """Fake LLM that records system prompts and returns valid bible + shot batches."""

    def __init__(self, *, shot_type: ShotType = ShotType.PERFORMANCE) -> None:
        self.systems: list[str] = []
        self.contents: list[str] = []
        self._shot_type = shot_type

    def complete_json(
        self,
        system: str,
        content: list[dict[str, Any]],
        schema: type[Any],
    ) -> Any:
        self.systems.append(system)
        self.contents.append("\n".join(str(block.get("text", "")) for block in content))
        if schema is BibleReply:
            return BibleReply(
                concept="A nocturnal journey through rain-slicked architecture.",
                appearance="Tall dancer in her late twenties with sharp features.",
                characters=[
                    CharacterReply(
                        id="protagonist",
                        name="Maya",
                        appearance="Tall dancer in her late twenties with sharp features.",
                        wardrobe=[WardrobeReply(id="coat", description="Dark wool trench coat.")],
                    )
                ],
                wardrobe=[WardrobeReply(id="coat", description="Dark wool trench coat.")],
                style="35mm anamorphic cinema",
                palette="deep indigo, amber streetlights",
                lighting="wet reflections and practical sodium lamps",
                negative_prompt="text, watermark, blurry",
                settings=[
                    SetReply(
                        name="Rainy Overpass",
                        sections=["Verse 1"],
                        description="Concrete pedestrian bridge above wet highway lanes.",
                        movement="Slow lateral dolly tracking past steel railings.",
                    ),
                    SetReply(
                        name="Glass Atrium",
                        sections=["Chorus"],
                        description="Empty botanical conservatory glowing in mist.",
                        movement="Sweeping wide arc beneath condensation-beaded glass.",
                    ),
                    SetReply(
                        name="Subway Tunnel",
                        sections=["Chorus"],
                        description="Tiled underground platform with flickering fluorescent tubes.",
                        movement="Forward glide down empty tracks.",
                    ),
                ],
                rules=["Maintain wet reflective surfaces.", "No subtitles."],
            )
        if schema is ShotBatchReply:
            text = self.contents[-1]
            shot_ids = [
                tok.split("=")[1]
                for tok in text.split()
                if tok.startswith("id=s") and tok[4:].isdigit()
            ]
            return ShotBatchReply(
                shots=[
                    ShotCreative(
                        id=sid,
                        type=self._shot_type,
                        framing="wide",
                        setting="Rainy Overpass",
                        character="protagonist" if self._shot_type.shows_singer else None,
                        characters=["protagonist"] if self._shot_type.shows_singer else [],
                        wardrobe="coat" if self._shot_type.shows_singer else None,
                        still_prompt=(
                            "Wide shot of Maya standing on the rainy overpass, mouth closed."
                            if self._shot_type.shows_singer
                            else "Wide shot of rain streaking down the empty glass atrium."
                        ),
                        motion_prompt="Slow dolly forward as rain mist drifts past sodium lights.",
                    )
                    for sid in shot_ids
                ]
            )
        raise AssertionError(f"Unexpected schema {schema}")


def test_narrative_style_removes_lipsync_and_keeps_characters():
    llm = _RecordingLLM(shot_type=ShotType.PERFORMANCE)
    cfg = ProjectConfig(planning=PlanningConfig(video_style=VideoStyle.NARRATIVE))
    manifest = build_manifest(
        cfg,
        _vocal_analysis(),
        "[Verse 1]\nwalking through the neon rain\n[Chorus]\nlights flicker in the dark\n",
        "Moody synthpop",
        llm,
    )

    assert manifest.shots
    assert all(shot.type is not ShotType.LIPSYNC for shot in manifest.shots)
    assert any(shot.type is ShotType.PERFORMANCE for shot in manifest.shots)
    assert manifest.character.appearance.strip()
    assert prompts.NARRATIVE_BIBLE_ADDENDUM in llm.systems[0]
    assert prompts.NARRATIVE_SHOTS_ADDENDUM in llm.systems[1]
    assert "vocal='solo'" not in llm.contents[1]


def test_visual_style_forces_all_broll_strips_characters_and_passes_preflight(project: Project):
    # Even if a misbehaving LLM returns a character and PERFORMANCE shot types, VISUAL enforces
    # 100% BROLL and empty characters.
    llm = _RecordingLLM(shot_type=ShotType.PERFORMANCE)
    cfg = ProjectConfig(planning=PlanningConfig(video_style=VideoStyle.VISUAL))
    manifest = build_manifest(
        cfg,
        _vocal_analysis(),
        "[Verse 1]\nwalking through the neon rain\n",
        "Ambient electronic",
        llm,
    )

    assert manifest.shots
    assert all(shot.type is ShotType.BROLL for shot in manifest.shots)
    assert all(
        shot.character is None and shot.characters == [] and shot.wardrobe is None
        for shot in manifest.shots
    )
    assert manifest.characters == {}
    assert manifest.character.appearance == ""
    assert llm.systems[0] == prompts.VISUAL_BIBLE_SYSTEM
    assert llm.systems[1] == prompts.VISUAL_SHOTS_SYSTEM

    templates = load_templates(project, ["t2i"])
    preflight = check_assets(project, manifest, templates)
    assert preflight.problems == []


def test_analyze_and_plan_allow_missing_vocals_and_lyrics_for_non_lipsync(tmp_path: Path):
    proj_dir = api.init_project(tmp_path / "inst-proj")
    sr = 22050
    t = np.linspace(0.0, 6.0, sr * 6, endpoint=False)
    mix = (0.3 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
    sf.write(proj_dir / "inputs" / "mix.wav", mix, sr)
    # Remove starter lyrics.txt and do not create vocals.wav.
    (proj_dir / "inputs" / "lyrics.txt").unlink()
    (proj_dir / "inputs" / "style.md").write_text("Cinematic instrumental synthwave\n")

    # Default LIPSYNC mode still requires vocals.wav.
    with pytest.raises(ConfigError, match="vocals"):
        api.analyze(proj_dir)

    # Switching to visual mode allows analysis and planning without vocals.wav or lyrics.txt.
    yaml_path = proj_dir / "project.yaml"
    yaml_text = yaml_path.read_text()
    yaml_text = yaml_text.replace("video_style: lipsync", "video_style: visual").replace(
        "  enabled: true                  # false = rule-based",
        "  enabled: false                 # false = rule-based",
    )
    yaml_path.write_text(yaml_text)

    _, analyzed = api.analyze(proj_dir)
    assert analyzed.vocal_windows == []
    assert analyzed.lyric_lines == []

    _, manifest = api.plan(proj_dir)
    assert manifest.shots
    assert all(s.type is ShotType.BROLL for s in manifest.shots)


def test_cli_plan_style_override_and_mix_mode_rejection(tmp_path: Path):
    proj_dir = api.init_project(tmp_path / "cli-style")
    yaml_path = proj_dir / "project.yaml"
    yaml_path.write_text(
        yaml_path.read_text().replace(
            "  enabled: true                  # false = rule-based",
            "  enabled: false                 # false = rule-based",
        )
    )
    project = Project.open(proj_dir)
    project.save_analysis(_vocal_analysis())
    (proj_dir / "inputs" / "lyrics.txt").write_text("[Verse 1]\nhello world\n")
    (proj_dir / "inputs" / "style.md").write_text("Indie pop\n")

    runner = CliRunner()
    res_visual = runner.invoke(app, ["plan", "-p", str(proj_dir), "--style", "visual"])
    assert res_visual.exit_code == 0, res_visual.output
    manifest_v = Project.open(proj_dir).load_manifest()
    assert all(s.type is ShotType.BROLL for s in manifest_v.shots)

    res_narr = runner.invoke(
        app, ["plan", "-p", str(proj_dir), "--overwrite", "--style", "narrative"]
    )
    assert res_narr.exit_code == 0, res_narr.output
    manifest_n = Project.open(proj_dir).load_manifest()
    assert all(s.type is ShotType.PERFORMANCE for s in manifest_n.shots)

    with pytest.raises(ConfigError, match="standard mode"):
        api.plan(
            proj_dir,
            overwrite=True,
            mode=ProjectMode.MIX,
            video_style=VideoStyle.VISUAL,
        )


def test_generate_visual_does_not_require_lipsync_or_still_workflows(
    project: Project, monkeypatch: pytest.MonkeyPatch
):
    cfg = project.config.model_copy(deep=True)
    cfg.planning.video_style = VideoStyle.VISUAL
    cfg.llm.enabled = False
    # Remove 'still' and 'lipsync' workflows from config to prove visual mode doesn't need them.
    cfg.comfy.workflows = {"t2i": cfg.comfy.workflows["t2i"], "i2v": cfg.comfy.workflows["i2v"]}
    manifest = build_manifest(cfg, _vocal_analysis(8.0), "", "Ambient", llm=None)
    project.save_manifest(manifest)

    loaded_names: list[list[str]] = []
    real_load = load_templates

    def spy_load(proj: Project, names: list[str]):
        loaded_names.append(list(names))
        return real_load(proj, names)

    class _StopAfterLoad(Exception):
        pass

    monkeypatch.setattr(api, "load_templates", spy_load)
    monkeypatch.setattr(
        api,
        "build_quality_gates",
        lambda *_a, **_kw: (_ for _ in ()).throw(_StopAfterLoad()),
    )

    project_no_lipsync = Project(project.root, cfg)
    with pytest.raises(_StopAfterLoad):
        api.generate(project_no_lipsync, [AssetKind.STILL, AssetKind.VIDEO], skip_preflight=True)

    assert loaded_names == [["t2i", "i2v"]]


def test_vocal_windows_ignore_near_silent_stem_bleed():
    sr = audio.ANALYSIS_SR
    t = np.linspace(0.0, 4.0, sr * 4, endpoint=False)
    bleed = (1e-4 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)  # ~ -80 dBFS
    rms, times = audio.rms_envelope(bleed, sr)
    assert audio.detect_vocal_windows(rms, times) == []


def test_auto_tune_skips_lipsync_knobs_when_style_disables_lipsync():
    profile = tuning.profile_track(
        lyrics_text="[Instrumental]\n",
        style_text="Purely instrumental ambient synthwave",
        analysis=Analysis(
            duration=16.0,
            tempo_bpm=120.0,
            beats=[0.5 * i for i in range(32)],
            bars=[2.0 * i for i in range(8)],
            bar_intensity=[0.5] * 8,
            vocal_windows=[],
            lyric_lines=[],
        ),
    )
    assert profile.vocal_mode == "instrumental"

    for style in (VideoStyle.NARRATIVE, VideoStyle.VISUAL):
        cfg = ProjectConfig(planning=PlanningConfig(video_style=style))
        adjustments = tuning.recommend_adjustments(cfg, profile)
        keys = {adj.key for adj in adjustments}
        assert "planning.max_lipsync_share" not in keys
        assert "video.lipsync_min_vocal_fraction" not in keys


def test_validate_project_only_checks_workflows_needed_for_video_style(
    project: Project, monkeypatch: pytest.MonkeyPatch
):
    from mvdirector.llm.base import ChatBackend

    checked_workflows: list[str] = []

    def fake_load(proj: Project, names: list[str]):
        checked_workflows.extend(names)
        return {}

    class _FakeComfy:
        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

        def ping(self):
            return None

    class _FakeProvider:
        def lease(self, _proj: Project):
            return _FakeComfy()

    monkeypatch.setattr(api, "load_templates", fake_load)
    monkeypatch.setattr(api.media, "require_tool", lambda _name: "/usr/bin/fake")

    for style, expected in (
        (VideoStyle.VISUAL, ["t2i", "i2v"]),
        (VideoStyle.NARRATIVE, ["t2i", "still", "i2v"]),
    ):
        checked_workflows.clear()
        cfg = project.config.model_copy(deep=True)
        cfg.planning.video_style = style
        cfg.llm.enabled = False
        api.validate_project(Project(project.root, cfg), comfy_provider=_FakeProvider())  # type: ignore[arg-type]
        assert checked_workflows == expected

    # Verify ChatBackend accepts literal newlines inside JSON string values (open-weights LLMs).
    from pydantic import BaseModel

    from mvdirector.llm.base import Reply

    class _Simple(BaseModel):
        lyrics: str

    class _StubBackend(ChatBackend):
        def __init__(self, text: str) -> None:
            super().__init__(model="stub", max_tokens=1000)
            self._text = text

        def _send(self, system, messages, schema):
            return Reply(text=self._text)

    backend = _StubBackend('{"lyrics": "line one\nline two"}')
    parsed = backend.complete_json(
        system="sys", content=[{"type": "text", "text": "go"}], schema=_Simple
    )
    assert parsed.lyrics == "line one\nline two"
