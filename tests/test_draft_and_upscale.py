# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Tests for draft-resolution generation and selective post-review upscaling."""

from __future__ import annotations

import re

import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError
from typer.testing import CliRunner

from mvdirector.assemble import plan_clips
from mvdirector.cli import app
from mvdirector.comfy.client import ComfyClient
from mvdirector.config import VideoConfig
from mvdirector.generation import reference, upscale
from mvdirector.generation.runner import RunBudget, ShotRunner, load_templates
from mvdirector.models import (
    AssetKind,
    AssetState,
    AssetStatus,
    Attempt,
    QCReport,
    RetryStrategy,
    ReviewDecision,
)
from mvdirector.planning.planner import build_manifest
from mvdirector.qc.factory import QualityGates
from mvdirector.qc.image import ImageQC
from mvdirector.qc.video import VideoQC
from mvdirector.retry import RetryPolicy
from tests.conftest import FakeComfy, needs_ffmpeg, write_upscale_workflow


def test_video_config_draft_size_validation_and_defaults() -> None:
    cfg = VideoConfig(width=1280, height=720)
    assert cfg.generation_size(draft=False) == (1280, 720)
    assert cfg.generation_size(draft=True) == (640, 360)

    custom = VideoConfig(width=1280, height=720, draft_width=480, draft_height=270)
    assert custom.generation_size(draft=True) == (480, 270)

    with pytest.raises(ValidationError, match="draft_width and video.draft_height"):
        VideoConfig(draft_width=640)


def test_size_multiple_moves_sizes_onto_the_workflow_grid_keeping_aspect() -> None:
    # LTX halves the size and snaps it to 32, so 640x360 comes back as 640x320.
    ltx = VideoConfig(width=1280, height=720, size_multiple=64)
    assert ltx.generation_size(draft=True) == (576, 320)
    assert ltx.generation_size(draft=False) == (1280, 704)
    for width, height in (ltx.generation_size(draft=True), ltx.generation_size(draft=False)):
        assert width % 64 == 0 and height % 64 == 0
        assert abs(width / height / (1280 / 720) - 1) <= 0.03

    # Sizes written out in the config are used as written; the default leaves sizes alone.
    pinned = VideoConfig(size_multiple=64, draft_width=480, draft_height=270)
    assert pinned.generation_size(draft=True) == (480, 270)
    assert VideoConfig().generation_size(draft=True) == (640, 360)

    portrait = VideoConfig(width=1080, height=1920, size_multiple=64)
    assert portrait.generation_size(draft=False) == (1088, 1920)
    with pytest.raises(ValidationError):
        VideoConfig(size_multiple=0)


def test_draft_runner_uses_draft_resolution_and_passes_dimension_qc(
    project, analysis, tmp_path, sample_png
) -> None:
    project.config.video.max_clip_seconds = 4.0
    project.config.video.draft_width = 480
    project.config.video.draft_height = 270
    project.config.qc.vision_enabled = False
    project.config.qc.on_missing_dependency = "skip"

    manifest = build_manifest(project.config, analysis, "lyrics", "style", llm=None)
    portrait = tmp_path / "portrait.png"
    portrait.write_bytes(sample_png)
    reference.lock_reference(project, manifest, portrait)

    # Render a sharp 480x270 still (< 0.5 * 720, so non-draft QC would reject it).
    draft_img = tmp_path / "draft.png"
    noise = np.random.default_rng(7).integers(0, 255, (270, 480, 3), dtype=np.uint8)
    Image.fromarray(noise).save(draft_img)

    fake = FakeComfy(png=draft_img.read_bytes())
    comfy = ComfyClient(project.config.comfy, transport=fake.transport(), sleep=lambda _: None)
    image_qc = ImageQC(project.config.qc, project.config.video, face_analyzer=None, vision=None)
    video_qc = VideoQC(
        project.config.qc,
        project.config.video,
        face_analyzer=None,
        mouth_tracker=None,
        vision=None,
    )
    runner = ShotRunner(
        project,
        manifest,
        comfy,
        QualityGates(image=image_qc, video=video_qc, face_analyzer=None),
        RetryPolicy(project.config.retry, project.config.seed, llm=None),
        load_templates(project, ["t2i", "still", "lipsync", "i2v"]),
        RunBudget(10),
        draft=True,
    )
    runner.run([AssetKind.STILL], shot_ids=["s001"])
    assert fake.queued[0]["2"]["inputs"]["width"] == 480
    assert fake.queued[0]["2"]["inputs"]["height"] == 270
    assert manifest.shot("s001").still.status is AssetStatus.PASSED


@needs_ffmpeg
def test_selective_upscale_assemble_and_reset(project, analysis, tmp_path, sample_png, sample_mp4):
    project.config.video.max_clip_seconds = 4.0
    project.config.review.mode = "strict"

    upscale_inputs = {
        "width": [{"node": "2", "field": "width"}],
        "height": [{"node": "2", "field": "height"}],
        "fps": [{"node": "2", "field": "fps"}],
        "seed": [{"node": "2", "field": "seed"}],
        "positive_prompt": [{"node": "1", "field": "text"}],
    }
    write_upscale_workflow(project, extra_inputs=upscale_inputs)

    manifest = build_manifest(project.config, analysis, "lyrics", "style", llm=None)
    for shot in manifest.shots:
        still_p = project.asset_path(shot, AssetKind.STILL, 1)
        still_p.parent.mkdir(parents=True, exist_ok=True)
        still_p.write_bytes(sample_png)
        shot.still = AssetState(
            status=AssetStatus.PASSED,
            selected_attempt=1,
            attempts_allowed=4,
            attempts=[
                Attempt(
                    number=1,
                    strategy=RetryStrategy.INITIAL,
                    seed=1,
                    prompt="still",
                    path=project.rel(still_p),
                    qc=QCReport(passed=True, score=0.9),
                )
            ],
        )

    # Mark s001 passed (ready), leave s002 exhausted (needs review).
    s1, s2 = manifest.shot("s001"), manifest.shot("s002")
    for idx, (shot, status) in enumerate(
        [(s1, AssetStatus.PASSED), (s2, AssetStatus.EXHAUSTED)], start=1
    ):
        vid_p = project.asset_path(shot, AssetKind.VIDEO, 1)
        vid_p.parent.mkdir(parents=True, exist_ok=True)
        vid_p.write_bytes(sample_mp4)
        shot.video = AssetState(
            status=status,
            selected_attempt=1,
            attempts_allowed=4,
            attempts=[
                Attempt(
                    number=1,
                    strategy=RetryStrategy.INITIAL,
                    seed=idx,
                    prompt=f"motion {shot.id}",
                    path=project.rel(vid_p),
                    qc=QCReport(passed=status is AssetStatus.PASSED, score=0.9),
                )
            ],
        )
    s1.review.decision = ReviewDecision.APPROVED

    # Error cleanly when upscale workflow isn't configured yet
    project.save_manifest(manifest)
    cli = CliRunner()
    err_res = cli.invoke(app, ["upscale", "-p", str(project.root)])
    assert err_res.exit_code == 1 and "No 'upscale' workflow configured" in err_res.output

    # Configure upscale workflow and run selective upscaling
    project.config.comfy.workflows["upscale"] = "upscale.bindings.json"
    fake = FakeComfy(mp4=sample_mp4)
    write_upscale_workflow(
        project, extra_inputs=upscale_inputs, length=25
    )  # FakeComfy returns an .mp4

    comfy = ComfyClient(project.config.comfy, transport=fake.transport(), sleep=lambda _: None)
    tpl = load_templates(project, ["upscale"])["upscale"]
    out = upscale.upscale_videos(project, manifest, comfy, tpl, shot_ids=["s001", "s002"])
    assert len(out) == 1 and out[0].name == "s001_a01_upscaled.mp4"
    assert s1.video.upscaled_path == "assets/videos/s001_a01_upscaled.mp4"
    assert s1.video.upscaled_from_attempt == 1
    assert s2.video.upscaled_path is None  # unapproved exhausted shot was skipped

    # Second run is idempotent (0 new Comfy jobs)
    out2 = upscale.upscale_videos(project, manifest, comfy, tpl, shot_ids=["s001", "s002"])
    assert out2 == [] and len(fake.queued) == 1

    # Assembly prefers upscaled_path over draft clip
    plans = plan_clips(project, manifest, force=True)
    assert plans[0].source == project.path("assets/videos/s001_a01_upscaled.mp4")
    assert plans[1].source == project.path("assets/videos/s002_a01.mp4")

    # Resetting the asset clears stale upscaled_path
    s1.video.reset(2, note="redo")
    assert s1.video.upscaled_path is None and s1.video.upscaled_from_attempt is None


@needs_ffmpeg
def test_draft_assemble_and_target_resolution_upscale(
    project, analysis, tmp_path, sample_png, sample_mp4
):
    from mvdirector import api as mvd_api
    from mvdirector import media
    from mvdirector.errors import ConfigError

    project.config.video.max_clip_seconds = 4.0
    project.config.review.mode = "strict"
    upscale_inputs = {
        "width": [{"node": "2", "field": "width"}],
        "height": [{"node": "2", "field": "height"}],
        "fps": [{"node": "2", "field": "fps"}],
        "seed": [{"node": "2", "field": "seed"}],
        "positive_prompt": [{"node": "1", "field": "text"}],
    }
    write_upscale_workflow(project, extra_inputs=upscale_inputs, length=25)
    project.config.comfy.workflows["upscale"] = "upscale.bindings.json"

    manifest = build_manifest(project.config, analysis, "lyrics", "style", llm=None)
    for idx, shot in enumerate(manifest.shots, start=1):
        still_p = project.asset_path(shot, AssetKind.STILL, 1)
        still_p.parent.mkdir(parents=True, exist_ok=True)
        still_p.write_bytes(sample_png)
        vid_p = project.asset_path(shot, AssetKind.VIDEO, 1)
        vid_p.parent.mkdir(parents=True, exist_ok=True)
        vid_p.write_bytes(sample_mp4)
        shot.still = AssetState(
            status=AssetStatus.PASSED,
            selected_attempt=1,
            attempts_allowed=4,
            attempts=[
                Attempt(
                    number=1,
                    strategy=RetryStrategy.INITIAL,
                    seed=idx,
                    prompt="still",
                    path=project.rel(still_p),
                    qc=QCReport(passed=True, score=0.9),
                )
            ],
        )
        shot.video = AssetState(
            status=AssetStatus.PASSED,
            selected_attempt=1,
            attempts_allowed=4,
            attempts=[
                Attempt(
                    number=1,
                    strategy=RetryStrategy.INITIAL,
                    seed=idx,
                    prompt="motion",
                    path=project.rel(vid_p),
                    qc=QCReport(passed=True, score=0.9),
                )
            ],
        )
        shot.review.decision = ReviewDecision.APPROVED
    project.save_manifest(manifest)

    # Upscale with explicit 1080p dimensions and progress callback
    fake = FakeComfy(mp4=sample_mp4)
    comfy = ComfyClient(project.config.comfy, transport=fake.transport(), sleep=lambda _: None)
    tpl = load_templates(project, ["upscale"])["upscale"]
    progress: list[tuple[str, int, int]] = []
    upscaled = upscale.upscale_videos(
        project,
        manifest,
        comfy,
        tpl,
        width=1920,
        height=1080,
        on_shot_done=lambda sid, done, total: progress.append((sid, done, total)),
    )
    assert len(upscaled) == len(manifest.shots)
    assert fake.queued[0]["2"]["inputs"]["width"] == 1920
    assert fake.queued[0]["2"]["inputs"]["height"] == 1080
    assert len(progress) == len(manifest.shots)
    assert progress[-1][1:] == (len(manifest.shots), len(manifest.shots))

    # plan_clips(draft=True) uses original draft clip even when upscaled_path exists
    draft_plans = plan_clips(project, manifest, force=False, draft=True)
    assert all("_upscaled" not in str(p.source) for p in draft_plans)

    # Draft assembly renders at draft resolution (640x360) and writes *_draft.mp4
    draft_out = mvd_api.assemble(project, draft=True)
    assert draft_out.name.endswith("_draft.mp4")
    draft_info = media.probe(draft_out)
    assert (draft_info.width, draft_info.height) == (640, 360)

    # Explicit target resolution assembly renders at 1920x1080
    hd_out = mvd_api.assemble(
        project, output=project.output_dir / "master_1080p.mp4", width=1920, height=1080
    )
    hd_info = media.probe(hd_out)
    assert (hd_info.width, hd_info.height) == (1920, 1080)

    # Dimension validation rejects partial, tiny, or odd sizes
    with pytest.raises(ConfigError):
        mvd_api.assemble(project, width=1920)
    with pytest.raises(ConfigError):
        mvd_api.assemble(project, width=32, height=32)
    with pytest.raises(ConfigError, match="even integers >= 64"):
        mvd_api.assemble(project, width=641, height=361)
    with pytest.raises(ConfigError):
        upscale.upscale_videos(project, manifest, comfy, tpl, width=1920)
    with pytest.raises(ConfigError):
        upscale.upscale_videos(project, manifest, comfy, tpl, width=32, height=32)
    with pytest.raises(ConfigError, match="even integers >= 64"):
        upscale.upscale_videos(project, manifest, comfy, tpl, width=641, height=361)
    with pytest.raises(ValidationError):
        VideoConfig(width=641, height=360)

    # Empty manifests fail cleanly instead of passing an empty concat list to ffmpeg.
    from mvdirector.errors import MVDError
    from mvdirector.models import Manifest

    project.save_manifest(Manifest(title="Empty", duration=1.0, fps=24, shots=[]))
    with pytest.raises(MVDError, match="No shots with frames to assemble"):
        mvd_api.assemble(project, force=True, draft=True)


_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")


def test_cli_exposes_draft_and_size_flags_on_assemble_and_upscale():
    # Typer styles help with Rich when it detects CI (GITHUB_ACTIONS), so compare plain text.
    runner = CliRunner()
    assemble_help = runner.invoke(app, ["assemble", "--help"])
    assert assemble_help.exit_code == 0
    assemble_text = _ANSI_ESCAPE.sub("", assemble_help.output)
    for flag in ("--draft", "--width", "--height"):
        assert flag in assemble_text

    upscale_help = runner.invoke(app, ["upscale", "--help"])
    assert upscale_help.exit_code == 0
    upscale_text = _ANSI_ESCAPE.sub("", upscale_help.output)
    for flag in ("--width", "--height"):
        assert flag in upscale_text
