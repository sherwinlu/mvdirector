# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""The per-check tallies behind ``mvd qc-report``."""

from __future__ import annotations

from mvdirector.models import (
    AssetStatus,
    Attempt,
    AudioSlice,
    CheckResult,
    CheckStatus,
    Manifest,
    QCReport,
    RetryStrategy,
    Shot,
    ShotType,
)
from mvdirector.report import render, summarize


def _shot(index: int) -> Shot:
    return Shot(
        id=f"s{index:03d}",
        index=index,
        start=0.0,
        end=4.0,
        type=ShotType.LIPSYNC,
        audio_slice=AudioSlice(path="a.wav", start=0.0, end=4.0, head_handle=0.0),
        num_frames=97,
    )


def _attempt(number: int, checks: list[CheckResult], passed: bool, error: str | None = None):
    return Attempt(
        number=number,
        strategy=RetryStrategy.INITIAL if number == 1 else RetryStrategy.NEW_SEED,
        seed=number,
        prompt="p",
        path=None if error else f"assets/stills/s001_a{number:02d}.png",
        qc=None if error else QCReport(passed=passed, score=0.8 if passed else 0.2, checks=checks),
        error=error,
    )


def _check(name: str, status: CheckStatus, detail: str = "") -> CheckResult:
    return CheckResult(name=name, status=status, detail=detail)


def test_summary_counts_checks_attempts_and_asset_outcomes():
    manifest = Manifest(title="t", duration=8.0, fps=24, shots=[_shot(1), _shot(2)])
    first, second = manifest.shots

    # Shot 1: still failed sharpness once, then passed; vision was skipped both times.
    skipped = _check("vision_spec", CheckStatus.SKIPPED, "vision check disabled")
    first.still.attempts = [
        _attempt(1, [_check("sharpness", CheckStatus.FAILED, "blurry"), skipped], passed=False),
        _attempt(2, [_check("sharpness", CheckStatus.PASSED), skipped], passed=True),
    ]
    first.still.status, first.still.selected_attempt = AssetStatus.PASSED, 2
    # Shot 2: one ComfyUI error, then an exhausted still a reviewer accepted.
    second.still.attempts = [
        _attempt(1, [], passed=False, error="out of memory"),
        _attempt(2, [_check("face_present", CheckStatus.FAILED, "no face")], passed=False),
    ]
    second.still.status, second.still.selected_attempt = AssetStatus.PASSED, 2
    second.still.accepted_by_reviewer = True

    summary = summarize(manifest)
    stills = summary.stills
    assert (stills.shots, stills.attempts, stills.errored_attempts) == (2, 4, 1)
    assert (stills.passed, stills.passed_first_try, stills.accepted_by_reviewer) == (2, 0, 1)
    assert (stills.exhausted, stills.pending) == (0, 0)
    assert (stills.checks["sharpness"].passed, stills.checks["sharpness"].failed) == (1, 1)
    assert stills.checks["vision_spec"].skipped == 2
    assert stills.checks["vision_spec"].skip_reasons == {"vision check disabled": 2}
    assert stills.checks["face_present"].failed == 1
    # Nothing has been generated for the videos yet.
    assert summary.videos.pending == 2 and summary.videos.checks == {}


def test_render_lists_every_check_with_skip_reasons_in_text_and_markdown():
    manifest = Manifest(title="t", duration=4.0, fps=24, shots=[_shot(1)])
    shot = manifest.shots[0]
    shot.video.attempts = [
        _attempt(
            1,
            [
                _check("lipsync", CheckStatus.PASSED),
                _check("identity", CheckStatus.SKIPPED, "face analysis unavailable"),
            ],
            passed=True,
        )
    ]
    shot.video.status, shot.video.selected_attempt = AssetStatus.PASSED, 1

    text = render(summarize(manifest))
    assert "Videos" in text and "lipsync" in text and "face analysis unavailable (1)" in text
    assert "1 shots, 1 attempts (0 errored). Passed: 1 (1 on the first attempt" in text

    markdown = render(summarize(manifest), markdown=True)
    assert "## Videos" in markdown
    assert "| `lipsync` | 1 | 1 | 0 | 0 |  |" in markdown
    assert "| `identity` | 0 | 0 | 0 | 1 | face analysis unavailable (1) |" in markdown
    assert "No QC reports recorded." in markdown  # the stills section
