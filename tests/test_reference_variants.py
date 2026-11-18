# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Tests for columns sheet layout, neutral grey studio background, invariants, and ref videos."""

from __future__ import annotations

import json

import pytest
from PIL import Image
from typer.testing import CliRunner

from mvdirector.cli import app
from mvdirector.comfy.client import ComfyClient
from mvdirector.config import ProjectConfig, ReferenceConfig
from mvdirector.errors import ConfigError
from mvdirector.generation import reference
from mvdirector.generation.runner import RunBudget, ShotRunner, load_templates
from mvdirector.models import (
    AssetKind,
    AssetStatus,
    Attempt,
    Character,
    RetryStrategy,
    WardrobeState,
)
from mvdirector.planning import prompts
from mvdirector.planning.planner import (
    BibleReply,
    CharacterReply,
    SetReply,
    ShotBatchReply,
    WardrobeReply,
    build_manifest,
)
from mvdirector.qc.factory import QualityGates
from mvdirector.retry import RetryPolicy
from tests.conftest import FakeComfy, needs_ffmpeg, write_workflows
from tests.test_pipeline import ScriptedGate

COL_COLORS = {
    "face": (210, 40, 40),
    "profile": (40, 210, 40),
    "front": (40, 40, 210),
    "back": (210, 210, 40),
}


def _make_columns_sheet(path, width=1536, height=1024):
    image = Image.new("RGB", (width, height), "grey")
    c1, c2, mid = width // 3, 2 * width // 3, height // 2
    boxes = {
        "face": (0, 0, c1, mid),
        "profile": (0, mid, c1, height),
        "front": (c1, 0, c2, height),
        "back": (c2, 0, width, height),
    }
    for name, box in boxes.items():
        image.paste(Image.new("RGB", (box[2] - box[0], box[3] - box[1]), COL_COLORS[name]), box[:2])
    image.save(path)
    return path


def test_columns_layout_splits_and_locks_back_panel(project, analysis, tmp_path) -> None:
    assert ReferenceConfig(layout="columns").size() == (1536, 1024)
    project.config.reference.layout = "columns"
    project.config.reference.background = "grey"
    manifest = build_manifest(project.config, analysis, "lyrics", "style", llm=None)

    sheet_path = _make_columns_sheet(tmp_path / "columns.png")
    locked = reference.lock_reference(project, manifest, sheet_path)
    assert set(locked.panels) == {"face", "profile", "front", "back"}
    for name, expected in COL_COLORS.items():
        panel = Image.open(project.path(locked.panels[name]))
        assert panel.getpixel((panel.width // 2, panel.height // 2)) == expected


def test_grey_background_and_invariants_flow_from_bible_to_state_sheets(analysis) -> None:
    class InvariantDirector:
        def complete_json(self, *, system, content, schema):
            if schema is BibleReply:
                return BibleReply(
                    style="35mm neon",
                    palette="cyan and black",
                    lighting="diffuse",
                    negative_prompt="blurry",
                    characters=[
                        CharacterReply(
                            id="kira",
                            name="Kira",
                            appearance="22-year-old Korean woman with silver ponytail.",
                            invariants="glowing cyan hoop earrings in both ears",
                            wardrobe=[
                                WardrobeReply(id="scifi", description="Black techwear jacket."),
                                WardrobeReply(id="arcade", description="Cream windbreaker."),
                            ],
                        )
                    ],
                    settings=[SetReply(name="Stage", description="neon stage")],
                )
            return ShotBatchReply(shots=[])

    cfg = ProjectConfig()
    cfg.reference.layout = "columns"
    cfg.reference.background = "grey"
    manifest = build_manifest(cfg, analysis, "lyrics", "style", InvariantDirector())
    assert manifest.character.invariants == "glowing cyan hoop earrings in both ears"

    base_prompt = prompts.sheet_prompt(manifest, "columns", background="grey")
    assert "neutral grey studio background" in base_prompt
    assert "FULL-BODY BACK" in base_prompt

    edit_prompt = prompts.state_sheet_prompt(manifest, "arcade", "columns", background="grey")
    assert "Keep glowing cyan hoop earrings in both ears." in edit_prompt
    assert "Cream windbreaker." in edit_prompt
    assert "Korean woman" not in edit_prompt  # face is never repeated on edit sheets


def test_reference_video_generation_requires_locked_sheet(project, analysis, sample_mp4) -> None:
    manifest = build_manifest(project.config, analysis, "lyrics", "style", llm=None)
    templates = load_templates(project, ["i2v"])
    comfy = ComfyClient(
        project.config.comfy, transport=FakeComfy(mp4=sample_mp4).transport(), sleep=lambda _: None
    )
    with pytest.raises(ConfigError, match="Lock the reference sheet"):
        reference.generate_video_candidates(project, manifest, comfy, templates, None, count=1)


@needs_ffmpeg
def test_reference_video_chaining_locking_and_runner_upload(
    project, analysis, tmp_path, sample_png, sample_mp4
) -> None:
    project.config.video.max_clip_seconds = 4.0
    write_workflows(
        project.path("workflows"),
        {"t2i": ".png", "still": ".png", "lipsync": ".mp4", "i2v": ".mp4"},
    )
    # Bind optional reference_video on node 3's spare field for i2v and lipsync.
    for wf_name in ("i2v", "lipsync"):
        p = project.path(f"workflows/{wf_name}.bindings.json")
        b = json.loads(p.read_text())
        b["inputs"]["reference_video"] = [{"node": "3", "field": "profile"}]
        p.write_text(json.dumps(b))

    manifest = build_manifest(project.config, analysis, "lyrics", "style", llm=None)
    manifest.character = Character(
        appearance="Kira, silver ponytail.",
        invariants="glowing cyan hoop earrings",
        wardrobe=[
            WardrobeState(id="scifi", description="Black techwear."),
            WardrobeState(id="arcade", description="Cream windbreaker."),
        ],
    )
    manifest.shot("s002").wardrobe = "scifi"
    manifest.shot("s004").wardrobe = "arcade"

    sheet_img = tmp_path / "sheet.png"
    Image.new("RGB", (2048, 896), "grey").save(sheet_img)
    reference.lock_reference(project, manifest, sheet_img, state_id="scifi")
    reference.lock_reference(project, manifest, sheet_img, state_id="arcade")

    fake = FakeComfy(png=sample_png, mp4=sample_mp4)
    comfy = ComfyClient(project.config.comfy, transport=fake.transport(), sleep=lambda _: None)
    templates = load_templates(project, ["t2i", "still", "lipsync", "i2v"])

    base_videos = reference.generate_video_candidates(
        project, manifest, comfy, templates, "scifi", count=1
    )
    assert len(base_videos) == 1 and base_videos[0].name == "scifi_video_candidate_01.mp4"
    assert fake.upload_files == ["scifi_sheet.png"]
    assert "Keep the exact face" not in fake.queued[0]["1"]["inputs"]["text"]

    locked_base = reference.lock_reference_video(
        project, manifest, base_videos[0], state_id="scifi"
    )
    assert locked_base == "assets/reference/scifi_video.mp4"
    assert manifest.character.references["scifi"].video == locked_base

    # Second outfit chains BOTH the arcade sheet and the base turnaround video.
    fake_chain = FakeComfy(png=sample_png, mp4=sample_mp4)
    comfy_chain = ComfyClient(
        project.config.comfy, transport=fake_chain.transport(), sleep=lambda _: None
    )
    arcade_videos = reference.generate_video_candidates(
        project, manifest, comfy_chain, templates, "arcade", count=1
    )
    assert len(arcade_videos) == 1
    assert fake_chain.upload_files == ["arcade_sheet.png", "scifi_video.mp4"]
    assert (
        "Preserve the exact face, body proportions" in fake_chain.queued[0]["1"]["inputs"]["text"]
    )

    # Runner forwards locked reference_video into downstream lipsync video jobs.
    shot = manifest.shot("s002")
    shot.still.status = AssetStatus.PASSED
    still_path = project.asset_path(shot, AssetKind.STILL, 1)
    still_path.parent.mkdir(parents=True, exist_ok=True)
    still_path.write_bytes(sample_png)
    shot.still.attempts = [
        Attempt(
            number=1,
            strategy=RetryStrategy.INITIAL,
            seed=1,
            prompt="still",
            path=project.rel(still_path),
        )
    ]
    shot.still.selected_attempt = 1

    fake_run = FakeComfy(png=sample_png, mp4=sample_mp4)
    runner = ShotRunner(
        project,
        manifest,
        ComfyClient(project.config.comfy, transport=fake_run.transport(), sleep=lambda _: None),
        QualityGates(image=ScriptedGate([True]), video=ScriptedGate([True]), face_analyzer=None),
        RetryPolicy(project.config.retry, project.config.seed, llm=None),
        templates,
        RunBudget(10),
    )
    runner.run([AssetKind.VIDEO], shot_ids=["s002"])
    assert "scifi_video.mp4" in fake_run.upload_files
    assert fake_run.queued[0]["3"]["inputs"]["profile"].startswith("mvdirector/upload_")

    # CLI lock-video and assets status display
    project.save_manifest(manifest)
    cli_runner = CliRunner()
    lock_res = cli_runner.invoke(
        app,
        [
            "reference",
            "lock-video",
            str(arcade_videos[0]),
            "--state",
            "arcade",
            "-p",
            str(project.root),
        ],
    )
    assert lock_res.exit_code == 0 and "Locked reference video" in lock_res.output
    assets_res = cli_runner.invoke(app, ["assets", "-p", str(project.root)])
    assert "+ video" in assets_res.output

    bad_vid = tmp_path / "bad.txt"
    bad_vid.write_text("not a video", encoding="utf-8")
    with pytest.raises(ConfigError, match="Unsupported reference video extension"):
        reference.lock_reference_video(project, manifest, bad_vid, state_id="scifi")
