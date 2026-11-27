# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Fewer avoidable retries: close-up size anchor and near-static lip-sync clips."""

from __future__ import annotations

import pytest

from mvdirector.models import (
    AudioSlice,
    Character,
    CheckResult,
    CheckStatus,
    Framing,
    Manifest,
    SetDesign,
    Shot,
    ShotType,
    VisualBible,
    WardrobeState,
)
from mvdirector.planning import prompts
from mvdirector.qc.video import _waive_static_for_passing_lipsync


def _manifest(shot_type: ShotType, framing: Framing, still_prompt: str) -> Manifest:
    return Manifest(
        title="t",
        duration=4.0,
        fps=24,
        character=Character(
            name="The Witch",
            appearance="Dark braid, amber eyes.",
            wardrobe=[WardrobeState(id="wool", description="Dark-green wool dress.")],
        ),
        visual_bible=VisualBible(
            style="folk horror",
            settings=[SetDesign(name="Cabin", description="A dark timber cabin. Candles.")],
        ),
        shots=[
            Shot(
                id="s017",
                index=0,
                start=0.0,
                end=3.0,
                num_frames=73,
                type=shot_type,
                framing=framing,
                audio_slice=AudioSlice(path="a.wav", start=0.0, end=3.25, head_handle=0.0),
                wardrobe="wool" if shot_type.shows_singer else None,
                setting="Cabin",
                still_prompt=still_prompt,
            )
        ],
    )


def test_close_up_gets_the_size_anchor_even_when_the_prompt_says_close_up():
    """Regression: 17 of 26 close-ups failed face_framing (face ~26% < 30%) because the anchor
    was skipped whenever the planner's prompt already contained "close-up"."""
    manifest = _manifest(
        ShotType.LIPSYNC,
        Framing.CLOSE_UP,
        "Close-up portrait of The Witch, framed tightly from the upper chest up.",
    )
    composed = prompts.compose_still_prompt(manifest, manifest.shots[0])
    assert composed.startswith(prompts.CLOSE_UP_SIZE_ANCHOR)
    assert "at least a third of the frame height" in composed


@pytest.mark.parametrize(
    ("shot_type", "framing", "text"),
    [
        (ShotType.BROLL, Framing.CLOSE_UP, "Close-up of a candle flame on the table."),
        (ShotType.LIPSYNC, Framing.MEDIUM, "Medium shot of The Witch by the hearth."),
    ],
)
def test_size_anchor_only_for_close_ups_of_the_singer(shot_type, framing, text):
    manifest = _manifest(shot_type, framing, text)
    composed = prompts.compose_still_prompt(manifest, manifest.shots[0])
    assert prompts.CLOSE_UP_SIZE_ANCHOR not in composed


def test_medium_close_anchor_still_defers_to_explicit_framing():
    manifest = _manifest(
        ShotType.LIPSYNC, Framing.MEDIUM_CLOSE, "Medium close-up of The Witch singing."
    )
    composed = prompts.compose_still_prompt(manifest, manifest.shots[0])
    assert not composed.startswith("Chest-up medium close-up shot")


def test_planner_instructions_define_close_up_by_size():
    assert "just above the head to the collarbone" in prompts.SHOTS_SYSTEM
    assert "chin to the collarbone" not in prompts.SHOTS_SYSTEM
    assert "upper chest up" in prompts.SHOTS_SYSTEM  # named as the thing not to write


@pytest.mark.parametrize(
    "raw_prompt",
    [
        "Close-up from behind The Witch's head as she faces the hearth.",
        "Close-up of The Witch's hands gripping a silver key.",
        "Tight close-up backlit silhouette of The Witch in the doorway.",
    ],
)
def test_close_up_size_anchor_skipped_for_non_portrait_performance_close_ups(raw_prompt: str):
    manifest = _manifest(ShotType.PERFORMANCE, Framing.CLOSE_UP, raw_prompt)
    composed = prompts.compose_still_prompt(manifest, manifest.shots[0])
    assert prompts.CLOSE_UP_SIZE_ANCHOR not in composed
    assert "tight head-and-shoulders close-up" not in composed
    assert prompts.CAMERA_PLACEMENT_NOTE in composed


def test_near_static_lipsync_clip_with_jump_cut_or_strobe_flicker_is_not_waived(
    tmp_path, monkeypatch
):
    import numpy as np
    import soundfile as sf

    from mvdirector import media
    from mvdirector.config import ProjectConfig
    from mvdirector.qc.lipsync import SyncMeasurement
    from mvdirector.qc.video import VideoQC

    wav_path = tmp_path / "slice.wav"
    sr = 16000
    t = np.linspace(0.0, 4.0, sr * 4, endpoint=False)
    sf.write(str(wav_path), 0.3 * np.sin(2 * np.pi * 440.0 * t), sr)
    clip_path = tmp_path / "clip.mp4"
    clip_path.write_bytes(b"fake")

    manifest = _manifest(ShotType.LIPSYNC, Framing.CLOSE_UP, "Close-up of The Witch singing.")
    shot = manifest.shots[0]
    shot.end = 4.0
    shot.num_frames = 96

    class GoodSync:
        def score(self, frames, fps, samples, sample_rate, max_lag_seconds, max_faces=1):
            return SyncMeasurement(correlation=0.70, lag_frames=0, face_presence=1.0)

    monkeypatch.setattr(
        media,
        "probe",
        lambda _: media.MediaInfo(duration=4.0, width=1280, height=704, fps=24.0, has_audio=False),
    )

    cfg = ProjectConfig()
    gate = VideoQC(cfg.qc, cfg.video, None, None, None, lipsync_scorer=GoodSync())

    # 1. Clean near-static singing clip -> static failure is waived when lip-sync passes.
    base = np.full((64, 64, 3), 40, dtype=np.uint8)
    clean_frames = [base.copy() for _ in range(96)]
    monkeypatch.setattr(media, "extract_frames", lambda *_, **__: clean_frames)
    clean_report = gate.evaluate(clip_path, shot, manifest, wav_path, None)
    clean_temporal = next(c for c in clean_report.checks if c.name == "temporal")
    assert clean_report.passed
    assert clean_temporal.status is CheckStatus.PASSED
    assert clean_temporal.metrics.get("static_waived") == 1.0

    # 2. Near-static clip with a single mid-clip jump cut -> NOT waived, hard fails.
    jump_frames = [base.copy() for _ in range(48)] + [
        np.full((64, 64, 3), 240, dtype=np.uint8) for _ in range(48)
    ]
    monkeypatch.setattr(media, "extract_frames", lambda *_, **__: jump_frames)
    jump_report = gate.evaluate(clip_path, shot, manifest, wav_path, None)
    jump_temporal = next(c for c in jump_report.checks if c.name == "temporal")
    assert not jump_report.passed and jump_report.has_hard_failure
    assert jump_temporal.status is CheckStatus.FAILED and jump_temporal.hard
    assert "abrupt scene change" in jump_temporal.detail
    assert "static_waived" not in jump_temporal.metrics

    # 3. Near-static clip with 7 intermittent strobe frames (p95 > flicker_bad) -> NOT waived.
    strobe_frames = [base.copy() for _ in range(96)]
    for idx in (10, 20, 30, 40, 50, 60, 70):
        strobe_frames[idx] = np.full((64, 64, 3), 95, dtype=np.uint8)
    monkeypatch.setattr(media, "extract_frames", lambda *_, **__: strobe_frames)
    strobe_report = gate.evaluate(clip_path, shot, manifest, wav_path, None)
    strobe_temporal = next(c for c in strobe_report.checks if c.name == "temporal")
    assert strobe_temporal.status is CheckStatus.FAILED and strobe_temporal.score == 0.0
    assert "flicker" in strobe_temporal.detail
    assert "static_waived" not in strobe_temporal.metrics


def _static() -> CheckResult:
    return CheckResult(
        name="temporal",
        status=CheckStatus.FAILED,
        hard=True,
        score=0.0,
        detail="clip is almost static",
        metrics={"median_motion": 0.00078},
    )


def _sync(status: CheckStatus) -> CheckResult:
    return CheckResult(name="lipsync", status=status, hard=True, score=0.9)


def test_near_static_singing_clip_with_good_lipsync_is_accepted():
    checks = _waive_static_for_passing_lipsync([_static(), _sync(CheckStatus.PASSED)])
    temporal = checks[0]
    assert temporal.status is CheckStatus.PASSED and not temporal.hard
    assert temporal.score is None  # does not move the aggregate score
    assert temporal.metrics["static_waived"] == 1.0
    assert temporal.metrics["median_motion"] == pytest.approx(0.00078)


def test_static_still_fails_when_lipsync_fails_or_is_missing():
    failed = _waive_static_for_passing_lipsync([_static(), _sync(CheckStatus.FAILED)])
    assert failed[0].status is CheckStatus.FAILED
    missing = _waive_static_for_passing_lipsync([_static()])
    assert missing[0].status is CheckStatus.FAILED


def test_other_temporal_failures_are_not_waived():
    cut = CheckResult(
        name="temporal",
        status=CheckStatus.FAILED,
        hard=True,
        score=0.0,
        detail="clip contains an abrupt scene change",
    )
    out = _waive_static_for_passing_lipsync([cut, _sync(CheckStatus.PASSED)])
    assert out[0].status is CheckStatus.FAILED
