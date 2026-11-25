# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Tests for F12: per-section storyboard contact sheets (draft_grid Path A & Path B)."""

from __future__ import annotations

from contextlib import contextmanager

import pytest
from PIL import Image
from typer.testing import CliRunner

from mvdirector import api, cleanup
from mvdirector.cli import app
from mvdirector.comfy.client import ComfyClient
from mvdirector.comfy.provider import StaticComfyProvider
from mvdirector.errors import ConfigError
from mvdirector.generation import draft_grid
from mvdirector.generation.runner import RunBudget, ShotRunner, load_templates
from mvdirector.models import AssetStatus, CheckStatus, ReferenceSet, SetDesign, Shot
from mvdirector.planning.planner import build_manifest
from mvdirector.qc.common import dimensions_check
from mvdirector.retry import RetryPolicy
from tests.conftest import FakeComfy, needs_ffmpeg, offline_gates, write_workflows


class _FakeProvider(StaticComfyProvider):
    def __init__(self, fake: FakeComfy, project) -> None:
        super().__init__()
        self._fake = fake
        self._project = project

    def lease(self, project):
        @contextmanager
        def _cm():
            yield ComfyClient(
                self._project.config.comfy,
                transport=self._fake.transport(),
                sleep=lambda _: None,
            )

        return _cm()


def _make_shots_for_grouping(base_shot: Shot, sections: list[str]) -> list[Shot]:
    shots: list[Shot] = []
    for idx, sec in enumerate(sections):
        s = base_shot.model_copy(deep=True)
        s.id = f"s{idx + 1:03d}"
        s.index = idx
        s.start = float(idx * 2)
        s.end = float((idx + 1) * 2)
        s.audio_slice.start = s.start
        s.audio_slice.end = s.end
        s.audio_slice.head_handle = 0.0
        s.section = sec
        shots.append(s)
    return shots


def test_draft_grid_contiguous_grouping_and_3x2_aspect_math(project, analysis) -> None:
    manifest = build_manifest(project.config, analysis, "lyrics", "style", llm=None)
    # Repeated Chorus blocks must stay separated, and 7-shot Verse must chunk (6 + 1).
    manifest.shots = _make_shots_for_grouping(
        manifest.shots[0],
        [
            "Chorus",
            "Chorus",
            *["Verse"] * 7,
            "Chorus",
            "Chorus",
        ],
    )
    chunks = draft_grid.group_section_shots(manifest, max_panels=6)
    assert [(sec, len(grp)) for sec, grp in chunks] == [
        ("Chorus", 2),
        ("Verse", 6),
        ("Verse", 1),
        ("Chorus", 2),
    ]

    # 3x2 canvas sizing keeps sliced panels strictly 16:9 passing ImageQC.dimensions_check.
    w3, h3 = draft_grid.grid_render_size(project, "3x2", draft=True)
    assert (w3, h3) == (1920, 720)
    sheet_3x2 = Image.new("RGB", (w3, h3), (30, 40, 50))
    panels = draft_grid.slice_contact_sheet(
        sheet_3x2,
        "3x2",
        6,
        inset=0.012,
        target_size=project.config.video.generation_size(draft=True),
    )
    assert len(panels) == 6
    for panel in panels:
        check = dimensions_check(
            panel.width,
            panel.height,
            expected_width=640,
            expected_height=360,
            aspect_tolerance=0.03,
            min_scale=0.5,
        )
        assert check.status is CheckStatus.PASSED, check.detail

    with pytest.raises(ConfigError, match="No matching shots"):
        draft_grid.group_section_shots(manifest, section="Nonexistent")


def test_draft_grid_path_a_stills_and_cleanup(project, analysis, sample_png) -> None:
    write_workflows(
        project.path("workflows"),
        {"t2i": ".png", "still": ".png", "lipsync": ".mp4", "i2v": ".mp4"},
        still_extra={"set_image": "set"},
    )
    manifest = build_manifest(project.config, analysis, "lyrics", "style", llm=None)
    manifest.shots = _make_shots_for_grouping(manifest.shots[0], ["Chorus"] * 4)
    for shot in manifest.shots:
        shot.setting = "Alley"
    manifest.visual_bible.settings = [SetDesign(name="Alley", description="Neon brick alley.")]
    ref_rel = "assets/reference/base_front.png"
    project.path(ref_rel).parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), "white").save(project.path(ref_rel))
    manifest.character.references["base"] = ReferenceSet(sheet=ref_rel, panels={"front": ref_rel})
    plate_rel = "assets/plates/alley.png"
    project.path(plate_rel).parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (128, 72), "navy").save(project.path(plate_rel))
    manifest.visual_bible.settings[0].plate = plate_rel
    project.save_manifest(manifest)

    fake = FakeComfy(png=sample_png)
    results = api.generate_draft_grid(
        project.root,
        section="Chorus",
        mode="stills",
        draft=True,
        comfy_provider=_FakeProvider(fake, project),
    )
    assert len(results) == 1
    res = results[0]
    assert res.layout == "2x2"
    assert res.sheet_path.is_file()
    assert len(res.panel_paths) == 4
    # Single ComfyUI call rendered all 4 shot stills!
    assert len(fake.queued) == 1
    assert set(fake.upload_files) == {"base_front.png", "alley.png"}

    reloaded = project.load_manifest()
    assert len(reloaded.draft_grids) == 1
    for shot in reloaded.shots:
        assert shot.still.status is AssetStatus.PASSED
        assert shot.still.selected_attempt == 1
        assert shot.draft_grid_sheet == project.rel(res.sheet_path)
        assert shot.still.selected is not None and project.path(shot.still.selected.path).is_file()

    # Cleanup leaves selected draft-grid stills and assets/draft_grids/ intact.
    plan = cleanup.plan_cleanup(project, reloaded)
    assert not plan.files


@needs_ffmpeg
def test_draft_grid_path_b_r2v_multicut_and_runner_attachment(
    project, analysis, sample_png, sample_mp4
) -> None:
    write_workflows(
        project.path("workflows"),
        {"t2i": ".png", "still": ".png", "lipsync": ".mp4", "i2v": ".mp4"},
        still_extra={"set_image": "set"},
    )
    # Add a dedicated r2v workflow accepting draft_grid_image + reference_image + set_image.
    import json

    graph = json.loads((project.path("workflows/graph.api.json")).read_text())
    graph["3"]["inputs"]["draft_grid"] = ""
    graph["3"]["inputs"]["ref2"] = ""
    (project.path("workflows/graph.api.json")).write_text(json.dumps(graph))
    r2v_bindings = {
        "workflow": "graph.api.json",
        "inputs": {
            "positive_prompt": [{"node": "1", "field": "text"}],
            "negative_prompt": [{"node": "1", "field": "negative"}],
            "seed": [{"node": "2", "field": "seed"}],
            "width": [{"node": "2", "field": "width"}],
            "height": [{"node": "2", "field": "height"}],
            "num_frames": [{"node": "2", "field": "length"}],
            "fps": [{"node": "2", "field": "fps"}],
            "image": [{"node": "3", "field": "image"}],
            "reference_image": [{"node": "3", "field": "face"}],
            "draft_grid_image": [{"node": "3", "field": "draft_grid"}],
            "set_image": [{"node": "3", "field": "set"}],
            "audio": [{"node": "3", "field": "audio"}],
        },
        "required": ["positive_prompt", "image", "seed"],
        "output": {"node": "9", "extensions": [".mp4"]},
    }
    (project.path("workflows/r2v.bindings.json")).write_text(json.dumps(r2v_bindings))
    project.config.comfy.workflows["r2v"] = "r2v.bindings.json"
    (project.root / "project.yaml").write_text(
        __import__("yaml").safe_dump(project.config.model_dump(mode="json"))
    )

    manifest = build_manifest(project.config, analysis, "lyrics", "style", llm=None)
    manifest.shots = _make_shots_for_grouping(manifest.shots[0], ["Verse"] * 3)
    project.save_manifest(manifest)

    fake = FakeComfy(png=sample_png, mp4=sample_mp4)
    results = api.generate_draft_grid(
        project.root,
        section="Verse",
        mode="both",
        draft=True,
        comfy_provider=_FakeProvider(fake, project),
    )
    assert len(results) == 1
    res = results[0]
    assert res.video_path is not None and res.video_path.is_file()
    # Queued once for contact sheet + once for R2V video.
    assert len(fake.queued) == 2
    r2v_prompt = fake.queued[1]["1"]["inputs"]["text"]
    assert "[Shot 1] At 00:00.000, Opens on:" in r2v_prompt
    assert "[Shot 2] At " in r2v_prompt and "Hard cut to:" in r2v_prompt
    assert "Hard cuts between distinct camera angles" in r2v_prompt

    reloaded = project.load_manifest()
    for shot in reloaded.shots:
        assert shot.allow_cuts is True
        assert shot.draft_grid_sheet == project.rel(res.sheet_path)
        assert shot.video.status is AssetStatus.PASSED
        assert shot.video.selected is not None and project.path(shot.video.selected.path).is_file()

    # ShotRunner._video_reference_paths forwards draft_grid_sheet to R2V templates.
    templates = load_templates(project, ["still", "t2i", "lipsync", "i2v", "r2v"])
    comfy = ComfyClient(project.config.comfy, transport=fake.transport(), sleep=lambda _: None)
    runner = ShotRunner(
        project,
        reloaded,
        comfy,
        offline_gates(project.config),
        RetryPolicy(project.config.retry, project.config.seed, llm=None),
        templates,
        RunBudget(5),
    )
    vrefs = runner._video_reference_paths(reloaded.shots[0], templates["r2v"])
    assert any(
        param == "draft_grid_image" and role == "draft_grid_image" for param, role, _ in vrefs
    )

    # CLI `mvd draft-grid` happy path (sheet only).
    cli_res = CliRunner().invoke(
        app, ["draft-grid", "--section", "Verse", "--mode", "sheet", "-p", str(project.root)]
    )
    assert cli_res.exit_code == 0
    assert "sec_01_verse" in cli_res.output


@needs_ffmpeg
def test_draft_grid_section_filter_preserves_noncontiguous_runs_and_r2v_durations(
    project, analysis, sample_png, sample_mp4
) -> None:
    write_workflows(
        project.path("workflows"),
        {"t2i": ".png", "still": ".png", "lipsync": ".mp4", "i2v": ".mp4"},
        still_extra={"set_image": "set"},
    )
    manifest = build_manifest(project.config, analysis, "lyrics", "style", llm=None)
    # 2 Chorus (0..4s) -> 14 Verse (4..32s) -> 2 Chorus (32..36s)
    manifest.shots = _make_shots_for_grouping(
        manifest.shots[0],
        ["Chorus"] * 2 + ["Verse"] * 14 + ["Chorus"] * 2,
    )
    project.save_manifest(manifest)

    # F12-BUG-1: Filtering by section='Chorus' must NOT merge Chorus 1 and Chorus 2 across Verse.
    chunks = draft_grid.group_section_shots(manifest, section="Chorus", max_panels=6)
    assert [(sec, [s.id for s in grp]) for sec, grp in chunks] == [
        ("Chorus", ["s001", "s002"]),
        ("Chorus", ["s017", "s018"]),
    ]

    # Non-adjacent shot_ids with an intervening unselected shot in the same section must split.
    gapped = draft_grid.group_section_shots(
        manifest, shot_ids=["s003", "s005", "s006"], max_panels=6
    )
    assert [(sec, [s.id for s in grp]) for sec, grp in gapped] == [
        ("Verse", ["s003"]),
        ("Verse", ["s005", "s006"]),
    ]

    fake = FakeComfy(png=sample_png, mp4=sample_mp4)
    results = api.generate_draft_grid(
        project.root,
        section="Chorus",
        mode="r2v",
        draft=True,
        comfy_provider=_FakeProvider(fake, project),
    )
    assert [r.sheet_id for r in results] == ["sec_01_chorus", "sec_05_chorus"]
    assert [r.shot_ids for r in results] == [["s001", "s002"], ["s017", "s018"]]
    # 2 sheets + 2 separate R2V renders (~4.0s each, NOT one ~36s span across the Verse).
    assert len(fake.queued) == 4
    r2v_frames = [
        fake.queued[1]["2"]["inputs"]["length"],
        fake.queued[3]["2"]["inputs"]["length"],
    ]
    assert all(f < 150 for f in r2v_frames), f"Unexpected merged R2V frame count: {r2v_frames}"


def test_draft_grid_subset_calls_use_distinct_sheet_ids_without_collision(
    project, analysis, sample_png
) -> None:
    manifest = build_manifest(project.config, analysis, "lyrics", "style", llm=None)
    manifest.shots = _make_shots_for_grouping(manifest.shots[0], ["Chorus"] * 4)
    project.save_manifest(manifest)

    fake = FakeComfy(png=sample_png)
    provider = _FakeProvider(fake, project)
    r1 = api.generate_draft_grid(
        project.root,
        shots=["s001", "s002"],
        mode="sheet",
        overwrite=False,
        comfy_provider=provider,
    )
    r2 = api.generate_draft_grid(
        project.root,
        shots=["s003", "s004"],
        mode="sheet",
        overwrite=False,
        comfy_provider=provider,
    )
    assert len(r1) == 1 and len(r2) == 1
    assert r1[0].sheet_id != r2[0].sheet_id
    assert r1[0].sheet_id == "sec_01_chorus_s001_s002"
    assert r2[0].sheet_id == "sec_01_chorus_s003_s004"
    assert len(fake.queued) == 2

    reloaded = project.load_manifest()
    assert len(reloaded.draft_grids) == 2
    assert {g.id for g in reloaded.draft_grids} == {
        "sec_01_chorus_s001_s002",
        "sec_01_chorus_s003_s004",
    }

    # Long shot IDs exceeding 64 chars fall back to a deterministic <= 64-char hashed ID.
    for idx, shot in enumerate(reloaded.shots):
        shot.id = f"very_long_custom_shot_identifier_prefix_000000000000_{idx + 1}"
    long_id_a = draft_grid._build_sheet_id(0, "Chorus", reloaded.shots[:2], is_full_chunk=False)
    long_id_b = draft_grid._build_sheet_id(0, "Chorus", reloaded.shots[2:4], is_full_chunk=False)
    assert len(long_id_a) <= 64 and len(long_id_b) <= 64
    assert long_id_a != long_id_b
