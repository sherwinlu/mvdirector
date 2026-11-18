# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Tests for opt-in multi-angle montage mode (allow_cuts)."""

from __future__ import annotations

import numpy as np

from mvdirector.config import ProjectConfig
from mvdirector.models import CheckStatus, ShotType
from mvdirector.planning import prompts
from mvdirector.planning.planner import (
    BibleReply,
    SetReply,
    ShotBatchReply,
    ShotCreative,
    build_manifest,
)
from mvdirector.qc import vision
from mvdirector.qc.video import VideoQC


def _synthetic_montage_frames(n_per_angle: int = 12) -> list[np.ndarray]:
    """Three distinct camera angles with gentle smooth motion inside each angle."""
    rng = np.random.default_rng(42)
    frames: list[np.ndarray] = []
    for base_level in (30, 125, 220):
        base = rng.integers(base_level - 15, base_level + 16, (64, 64, 3), dtype=np.int16)
        for step in range(n_per_angle):
            shifted = np.roll(base, shift=step, axis=1)
            frames.append(np.clip(shifted, 0, 255).astype(np.uint8))
    return frames


def test_planner_guards_allow_cuts_behind_config_flag(analysis) -> None:
    class MontageDirector:
        def complete_json(self, *, system, content, schema):
            if schema is BibleReply:
                return BibleReply(
                    appearance="Kira.",
                    style="neon",
                    palette="cyan",
                    lighting="hard",
                    negative_prompt="blurry",
                    settings=[SetReply(name="Stage", description="neon stage")],
                )
            return ShotBatchReply(
                shots=[
                    ShotCreative(
                        id="s001",
                        type=ShotType.PERFORMANCE,
                        framing="medium",
                        setting="Stage",
                        allow_cuts=True,
                        still_prompt="On stage.",
                        motion_prompt="0-4s: multi-angle cuts.",
                    )
                ]
            )

    default_cfg = ProjectConfig()
    default_manifest = build_manifest(default_cfg, analysis, "lyrics", "style", MontageDirector())
    assert default_manifest.shot("s001").allow_cuts is False

    enabled_cfg = ProjectConfig()
    enabled_cfg.planning.allow_multicut_shots = True
    montage_manifest = build_manifest(enabled_cfg, analysis, "lyrics", "style", MontageDirector())
    s1 = montage_manifest.shot("s001")
    assert s1.allow_cuts is True
    assert "Hard cuts between distinct camera angles" in prompts.compose_motion_prompt(
        montage_manifest, s1
    )
    assert "Multi-angle montage (allow_cuts=true)" in vision.VisionJudge.specification(
        s1, montage_manifest
    )


def test_video_qc_allows_hard_cuts_only_when_allow_cuts_is_true() -> None:
    cfg = ProjectConfig()
    video_qc = VideoQC(cfg.qc, cfg.video, face_analyzer=None, mouth_tracker=None, vision=None)
    frames = _synthetic_montage_frames()

    single_take_result = video_qc._temporal_checks(frames, allow_cuts=False)[0]
    assert single_take_result.status is CheckStatus.FAILED
    assert single_take_result.hard is True
    assert "abrupt scene change" in single_take_result.detail

    montage_result = video_qc._temporal_checks(frames, allow_cuts=True)[0]
    assert montage_result.status is CheckStatus.PASSED
    assert montage_result.metrics["cut_count"] == 2.0
    assert "multi-cut montage (2 cut(s))" in montage_result.detail

    # Pure random noise on every frame must still fail even when allow_cuts=True.
    rng = np.random.default_rng(99)
    strobe_frames = [rng.integers(0, 256, (64, 64, 3), dtype=np.uint8) for _ in range(16)]
    strobe_result = video_qc._temporal_checks(strobe_frames, allow_cuts=True)[0]
    assert strobe_result.status is CheckStatus.FAILED

    # Alternating hard cuts after a calm pair (non-empty non_cut) must fail cut-density guard.
    black = np.full((64, 64, 3), 20, dtype=np.uint8)
    white = np.full((64, 64, 3), 235, dtype=np.uint8)
    partial_strobe = [black, black] + [white if i % 2 == 0 else black for i in range(14)]
    partial_result = video_qc._temporal_checks(partial_strobe, allow_cuts=True)[0]
    assert partial_result.status is CheckStatus.FAILED
    assert partial_result.hard is True
    assert "too many abrupt scene changes" in partial_result.detail
