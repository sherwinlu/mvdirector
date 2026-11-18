# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Tests for 3D blocking previz planning, rendering, CLI, and workflow integration."""

from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError
from typer.testing import CliRunner

from mvdirector import media
from mvdirector.cli import app
from mvdirector.comfy.client import ComfyClient
from mvdirector.errors import ConfigError, LLMError
from mvdirector.generation import plates, previz
from mvdirector.generation.runner import RunBudget, ShotRunner, load_templates
from mvdirector.models import (
    AssetKind,
    AssetStatus,
    Attempt,
    Character,
    RetryStrategy,
    SetDesign,
)
from mvdirector.planning import prompts
from mvdirector.planning.planner import build_manifest
from mvdirector.planning.previz import (
    BlockingActor,
    BlockingKeyframe,
    CameraKeyframe,
    PrevizPlan,
    default_previz_plan,
    plan_previz,
)
from mvdirector.qc.factory import QualityGates
from mvdirector.retry import RetryPolicy
from tests.conftest import FakeComfy, needs_ffmpeg, write_workflows
from tests.test_pipeline import ScriptedGate


def test_default_previz_plan_includes_cast_and_stand_in_mapping(project, analysis) -> None:
    project.config.video.max_clip_seconds = 4.0
    manifest = build_manifest(project.config, analysis, "lyrics", "style", llm=None)
    manifest.characters = {
        "kira": Character(name="Kira", appearance="Kira."),
        "clone": Character(name="Clone", appearance="Masked clone."),
    }
    manifest.character = manifest.characters["kira"]
    shot = manifest.shot("s002")
    shot.character = "kira"
    shot.characters = ["kira", "clone"]

    plan = default_previz_plan(shot, manifest)
    assert len(plan.camera) == 2
    assert any(a.kind == "figure" and a.role == "primary" for a in plan.actors)
    assert any(a.kind == "figure" and a.role == "extra" for a in plan.actors)
    assert "Kira" in plan.stand_in_mapping and "Clone" in plan.stand_in_mapping


def test_plan_previz_sends_set_plate_to_llm_and_falls_back_on_error(
    project, analysis, tmp_path
) -> None:
    manifest = build_manifest(project.config, analysis, "lyrics", "style", llm=None)
    manifest.visual_bible.settings = [SetDesign(name="Neon City", description="glowing stage")]
    shot = manifest.shot("s001")
    shot.setting = "Neon City"
    plate_img = tmp_path / "plate.png"
    Image.new("RGB", (64, 36), "navy").save(plate_img)
    plates.lock_plate(project, manifest, plate_img, "Neon City")

    class FakeLLM:
        def __init__(self, fail: bool = False) -> None:
            self.fail = fail
            self.seen_content = None

        def complete_json(self, *, system, content, schema):
            if self.fail:
                raise LLMError("timeout")
            self.seen_content = content
            return PrevizPlan(
                stand_in_mapping="Dark figure becomes Kira; cyan ring becomes the light disc.",
                actors=[
                    BlockingActor(
                        id="kira",
                        kind="figure",
                        role="primary",
                        color=(38, 42, 52),
                        size=(0.55, 1.72, 0.35),
                        keyframes=[BlockingKeyframe(time=0.0, position=(0.0, 0.0, 0.0))],
                    ),
                    BlockingActor(
                        id="disc",
                        kind="ring",
                        role="prop",
                        color=(0, 220, 240),
                        size=(0.6, 0.08, 0.6),
                        keyframes=[BlockingKeyframe(time=0.0, position=(0.8, 1.2, 0.0))],
                    ),
                ],
                camera=[
                    CameraKeyframe(time=0.0, eye=(0.0, 2.0, 4.0), look_at=(0.0, 1.0, 0.0)),
                    CameraKeyframe(time=4.0, eye=(0.0, 5.5, 0.01), look_at=(0.0, 0.0, 0.0)),
                ],
            )

    llm = FakeLLM()
    plan = plan_previz(
        shot, manifest, llm, set_plate_path=project.path("assets/plates/neon_city.png")
    )
    assert any(block.get("type") == "image" for block in llm.seen_content)
    assert any(a.kind == "ring" for a in plan.actors)
    assert "light disc" in plan.stand_in_mapping

    # Frame rendering handles overhead straight-down view and rings cleanly without NaN.
    frame = previz.render_blocking_frame(plan, 4.0, 160, 90)
    arr = np.asarray(frame)
    assert arr.shape == (90, 160, 3) and arr.std() > 1.0

    # LLM errors fall back cleanly to deterministic blocking.
    fallback = plan_previz(shot, manifest, FakeLLM(fail=True))
    assert len(fallback.actors) >= 3 and len(fallback.camera) == 2


@needs_ffmpeg
def test_previz_generate_lock_cli_and_runner_upload(
    project, analysis, tmp_path, sample_png, sample_mp4
) -> None:
    project.config.video.max_clip_seconds = 4.0
    project.config.llm.enabled = False
    write_workflows(
        project.path("workflows"),
        {"t2i": ".png", "still": ".png", "lipsync": ".mp4", "i2v": ".mp4"},
    )
    # Bind optional previz_video on i2v / lipsync
    for wf_name in ("i2v", "lipsync"):
        p = project.path(f"workflows/{wf_name}.bindings.json")
        b = json.loads(p.read_text())
        b["inputs"]["previz_video"] = [{"node": "3", "field": "profile"}]
        p.write_text(json.dumps(b))

    manifest = build_manifest(project.config, analysis, "lyrics", "style", llm=None)
    project.save_manifest(manifest)

    # Generate previz via CLI for shot s001
    cli = CliRunner()
    res = cli.invoke(app, ["previz", "generate", "--shot", "s001", "-p", str(project.root)])
    assert res.exit_code == 0 and "Generated previz" in res.output

    updated = project.load_manifest()
    s1 = updated.shot("s001")
    assert s1.previz_video == "assets/previz/s001.mp4"
    assert s1.previz_mapping
    info = media.probe(project.path(s1.previz_video))
    assert info.width is not None and info.height is not None and info.duration > 1.0

    # Lock an external blocking clip for s002
    ext_clip = tmp_path / "external_blocking.mp4"
    ext_clip.write_bytes(sample_mp4)
    res_lock = cli.invoke(
        app,
        [
            "previz",
            "lock",
            str(ext_clip),
            "--shot",
            "s002",
            "--mapping",
            "Dark figure becomes Kira.",
            "-p",
            str(project.root),
        ],
    )
    assert res_lock.exit_code == 0 and "Locked previz for s002" in res_lock.output
    updated = project.load_manifest()
    s1 = updated.shot("s001")
    assert updated.shot("s002").previz_mapping == "Dark figure becomes Kira."

    # Seed passed still for s001 and verify ShotRunner uploads previz_video & updates prompt
    still_path = project.asset_path(s1, AssetKind.STILL, 1)
    still_path.parent.mkdir(parents=True, exist_ok=True)
    still_path.write_bytes(sample_png)
    s1.still.status = AssetStatus.PASSED
    s1.still.attempts = [
        Attempt(
            number=1,
            strategy=RetryStrategy.INITIAL,
            seed=1,
            prompt="still",
            path=project.rel(still_path),
        )
    ]
    s1.still.selected_attempt = 1

    fake = FakeComfy(png=sample_png, mp4=sample_mp4)
    runner = ShotRunner(
        project,
        updated,
        ComfyClient(project.config.comfy, transport=fake.transport(), sleep=lambda _: None),
        QualityGates(image=ScriptedGate([True]), video=ScriptedGate([True]), face_analyzer=None),
        RetryPolicy(project.config.retry, project.config.seed, llm=None),
        load_templates(project, ["t2i", "still", "lipsync", "i2v"]),
        RunBudget(10),
    )
    runner.run([AssetKind.VIDEO], shot_ids=["s001"])
    assert "s001.mp4" in fake.upload_files
    assert fake.queued[0]["3"]["inputs"]["profile"].startswith("mvdirector/upload_")
    prompt = s1.video.attempts[0].prompt
    assert "Use the previz reference video to define the pacing" in prompt
    assert "Stand-in mapping:" in prompt
    assert prompts.compose_motion_prompt(updated, s1, previz_attached=False).startswith(
        "The singer"
    )


def test_previz_path_rejects_unsafe_shot_id(project) -> None:
    with pytest.raises(ConfigError, match="shot id"):
        project.previz_path("../escape")


def test_previz_validates_colors_sizes_and_extensions(project, analysis, tmp_path) -> None:
    with pytest.raises(ValidationError):
        BlockingActor(id="bad_rgb", color=(300, -1, 10))
    with pytest.raises(ValidationError):
        BlockingActor(id="bad_size", size=(0.5, 0.0, 0.5))
    with pytest.raises(ValidationError):
        BlockingKeyframe(time=float("nan"), position=(0.0, 0.0, 0.0))

    # Empty camera list renders safely via fallback camera rather than crashing.
    frame = previz.render_blocking_frame(PrevizPlan(), 0.0, 64, 64)
    assert frame.size == (64, 64)

    manifest = build_manifest(project.config, analysis, "lyrics", "style", llm=None)
    bad_ext = tmp_path / "notes.txt"
    bad_ext.write_text("not a video", encoding="utf-8")
    with pytest.raises(ConfigError, match="Unsupported previz video extension"):
        previz.lock_previz(project, manifest, bad_ext, "s001")


@needs_ffmpeg
def test_previz_refine_updates_saved_plan_and_rerenders(
    project, analysis, monkeypatch: pytest.MonkeyPatch
) -> None:
    project.config.video.max_clip_seconds = 4.0
    project.config.llm.enabled = False
    manifest = build_manifest(project.config, analysis, "lyrics", "style", llm=None)
    project.save_manifest(manifest)

    cli = CliRunner()
    gen_res = cli.invoke(app, ["previz", "generate", "--shot", "s001", "-p", str(project.root)])
    assert gen_res.exit_code == 0
    plan_file = project.previz_plan_path("s001")
    assert plan_file.is_file()

    # Rejects blank feedback, missing llm, and missing plan.
    with pytest.raises(ConfigError, match="non-empty --feedback"):
        previz.refine_previz_shot(project, manifest, None, "s001", "   ")
    with pytest.raises(ConfigError, match="llm.enabled: true"):
        previz.refine_previz_shot(project, manifest, None, "s001", "Zoom in closer")

    class RefineLLM:
        def __init__(self) -> None:
            self.seen_prompt = ""

        def complete_json(self, *, system, content, schema):
            self.seen_prompt = content[-1]["text"]
            return PrevizPlan(
                stand_in_mapping="Tighter close-up on the singer.",
                actors=[
                    BlockingActor(
                        id="singer",
                        kind="figure",
                        role="primary",
                        color=(38, 42, 52),
                        size=(0.55, 1.72, 0.35),
                        keyframes=[BlockingKeyframe(time=0.0, position=(0.0, 0.0, 0.0))],
                    )
                ],
                camera=[
                    CameraKeyframe(time=0.0, eye=(0.0, 1.5, 1.4), look_at=(0.0, 1.48, 0.0)),
                    CameraKeyframe(time=4.0, eye=(0.0, 1.5, 1.0), look_at=(0.0, 1.48, 0.0)),
                ],
            )

    refine_llm = RefineLLM()
    with pytest.raises(ConfigError, match="No editable 3D blocking plan"):
        previz.refine_previz_shot(project, manifest, refine_llm, "s002", "Orbit faster")

    from mvdirector import api as api_mod

    monkeypatch.setattr(api_mod, "build_llm", lambda _cfg: refine_llm)
    res = cli.invoke(
        app,
        [
            "previz",
            "refine",
            "--shot",
            "s001",
            "--feedback",
            "Zoom tighter on face",
            "-p",
            str(project.root),
        ],
    )
    assert res.exit_code == 0 and "Refined previz for s001" in res.output
    assert "Zoom tighter on face" in refine_llm.seen_prompt
    assert "CURRENT 3D BLOCKING PLAN:" in refine_llm.seen_prompt
    assert project.load_manifest().shot("s001").previz_mapping == "Tighter close-up on the singer."
    saved = PrevizPlan.model_validate_json(plan_file.read_text(encoding="utf-8"))
    assert saved.actors[0].id == "singer"
