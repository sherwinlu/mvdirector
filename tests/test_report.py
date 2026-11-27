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
    ContinuityMode,
    Manifest,
    QCReport,
    RetryStrategy,
    Review,
    ReviewDecision,
    Shot,
    ShotType,
)
from mvdirector.report import attempt_stats, render, summarize


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
    assert type(stills.checks) is dict and type(summary.videos.checks) is dict
    serialized = summary.to_dict()
    assert serialized["stills"]["checks"]["vision_spec"]["skip_reasons"] == {
        "vision check disabled": 2
    }


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


def test_attempt_stats_empty_and_zero_passed_manifests() -> None:
    empty = Manifest(title="t", duration=0.0, fps=24, shots=[])
    stats_empty = attempt_stats(empty)
    assert stats_empty.total_shots == 0
    assert stats_empty.settled_count == 0
    assert stats_empty.mean_attempts_per_passed == 0.0
    assert stats_empty.p90_attempts == 0.0
    assert stats_empty.human_reject_rate == 0.0
    assert set(stats_empty.by_kind) == {"still", "video"}
    assert set(stats_empty.by_shot_type) == {"singing", "broll", "chained", "performance"}

    pending_shot = _shot(1)
    pending_shot.still.attempts = [_attempt(1, [], passed=False, error="oom")]
    pending_shot.still.status = AssetStatus.EXHAUSTED
    no_pass = Manifest(title="t", duration=4.0, fps=24, shots=[pending_shot])
    stats_no_pass = attempt_stats(no_pass)
    assert stats_no_pass.settled_count == 0
    assert stats_no_pass.exhausted_count == 1
    assert stats_no_pass.pending_count == 1
    assert stats_no_pass.errored_attempts == 1
    assert stats_no_pass.mean_attempts_per_passed == 0.0


def test_attempt_stats_splits_by_kind_shot_type_and_human_reject_rate(project) -> None:
    import json

    from typer.testing import CliRunner

    from mvdirector import api
    from mvdirector.cli import app

    s1 = _shot(1)  # singing (LIPSYNC)
    s1.still.attempts = [_attempt(1, [], passed=True)]
    s1.still.status, s1.still.selected_attempt = AssetStatus.PASSED, 1
    s1.video.attempts = [
        _attempt(1, [], passed=False),
        _attempt(2, [], passed=False),
        _attempt(3, [], passed=True),
    ]
    s1.video.status, s1.video.selected_attempt = AssetStatus.PASSED, 3
    s1.review = Review(required=True, decision=ReviewDecision.APPROVED)

    s2 = _shot(2)  # broll
    s2.type = ShotType.BROLL
    s2.still.attempts = [_attempt(1, [], passed=True)]
    s2.still.status, s2.still.selected_attempt = AssetStatus.PASSED, 1
    s2.video.attempts = [_attempt(1, [], passed=False)]
    s2.video.status = AssetStatus.PENDING
    s2.review = Review(required=True, decision=ReviewDecision.REJECTED, note="bad")

    s3 = _shot(3)  # chained (ContinuityMode.CONTINUE)
    s3.type = ShotType.PERFORMANCE
    s3.continuity = ContinuityMode.CONTINUE
    s3.still.attempts = [_attempt(1, [], passed=True)]
    s3.still.status, s3.still.selected_attempt = AssetStatus.PASSED, 1
    s3.video.attempts = [_attempt(1, [], passed=False), _attempt(2, [], passed=True)]
    s3.video.status, s3.video.selected_attempt = AssetStatus.PASSED, 2

    manifest = Manifest(title="t", duration=12.0, fps=24, shots=[s1, s2, s3])
    project.save_manifest(manifest)

    stats = api.attempt_stats(manifest)
    assert stats.total_shots == 3
    assert stats.settled_count == 5  # s1(2) + s2(still=1) + s3(2)
    assert stats.pending_count == 1
    assert stats.total_attempts == 9
    assert stats.reviewed_count == 2
    assert stats.approved_count == 1
    assert stats.rejected_count == 1
    assert stats.human_reject_rate == 0.5
    assert stats.by_kind["still"].mean_attempts_per_passed == 1.0
    assert stats.by_kind["video"].mean_attempts_per_passed == 2.5
    assert stats.by_kind["video"].p90_attempts == 2.9
    assert stats.by_kind["video"].p90_multiplier == 2.9
    assert stats.by_shot_type["singing"].mean_attempts_per_passed == 2.0
    assert stats.by_shot_type["broll"].mean_attempts_per_passed == 1.0
    assert stats.by_shot_type["chained"].mean_attempts_per_passed == 1.5
    assert api.attempt_stats(project.root).to_dict() == stats.to_dict()

    runner = CliRunner()
    txt = runner.invoke(app, ["qc-report", "-p", str(project.root)])
    assert txt.exit_code == 0
    assert "Attempt & Rejection Stats" in txt.output
    assert "50.0%" in txt.output

    md = runner.invoke(app, ["qc-report", "--markdown", "-p", str(project.root)])
    assert md.exit_code == 0
    assert "## Stills" in md.output
    assert "## Attempt & Rejection Stats" in md.output

    js = runner.invoke(app, ["qc-report", "--json", "-p", str(project.root)])
    assert js.exit_code == 0
    parsed = json.loads(js.output)
    assert parsed["attempt_stats"]["human_reject_rate"] == 0.5
    assert parsed["attempt_stats"]["by_kind"]["video"]["mean_attempts_per_passed"] == 2.5
