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
    assert "chin to the collarbone" in prompts.SHOTS_SYSTEM
    assert "upper chest up" in prompts.SHOTS_SYSTEM  # named as the thing not to write


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
