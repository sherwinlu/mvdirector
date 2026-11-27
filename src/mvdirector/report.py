# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Per-check pass, fail and skip counts over every attempt in a manifest.

Used by ``mvd qc-report`` so the result of a run can be published as numbers rather than as
"it worked". Skipped checks are counted with their reasons, which is how a run that never
exercised a check shows up as such.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any

from mvdirector.models import (
    AssetKind,
    AssetStatus,
    CheckStatus,
    ContinuityMode,
    Manifest,
    ReviewDecision,
    Shot,
    ShotType,
)

# Order the checks are listed in, so the tables read the same way every time.
_CHECK_ORDER = [
    "dimensions",
    "sharpness",
    "face_present",
    "face_framing",
    "identity",
    "duration",
    "temporal",
    "face_presence",
    "lipsync",
    "vision_spec",
    "vision_artifacts",
    "vision_character",
]


@dataclass
class CheckTally:
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    skip_reasons: Counter[str] = field(default_factory=Counter)

    @property
    def ran(self) -> int:
        return self.passed + self.failed

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "failed": self.failed,
            "skipped": self.skipped,
            "skip_reasons": dict(self.skip_reasons),
        }


@dataclass
class AssetTally:
    shots: int = 0
    attempts: int = 0
    errored_attempts: int = 0
    passed: int = 0  # shots whose asset passed QC
    passed_first_try: int = 0
    exhausted: int = 0
    accepted_by_reviewer: int = 0
    pending: int = 0
    checks: dict[str, CheckTally] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "shots": self.shots,
            "attempts": self.attempts,
            "errored_attempts": self.errored_attempts,
            "passed": self.passed,
            "passed_first_try": self.passed_first_try,
            "exhausted": self.exhausted,
            "accepted_by_reviewer": self.accepted_by_reviewer,
            "pending": self.pending,
            "checks": {name: entry.to_dict() for name, entry in self.checks.items()},
        }


@dataclass
class QCReportSummary:
    stills: AssetTally
    videos: AssetTally

    def tally(self, kind: AssetKind) -> AssetTally:
        return self.stills if kind is AssetKind.STILL else self.videos

    def to_dict(self) -> dict[str, Any]:
        return {"stills": self.stills.to_dict(), "videos": self.videos.to_dict()}


def summarize(manifest: Manifest) -> QCReportSummary:
    summary = QCReportSummary(stills=AssetTally(), videos=AssetTally())
    for shot in manifest.shots:
        for kind in (AssetKind.STILL, AssetKind.VIDEO):
            state = shot.asset(kind)
            tally = summary.tally(kind)
            tally.shots += 1
            if state.status is AssetStatus.PASSED:
                tally.passed += 1
                if state.accepted_by_reviewer:
                    tally.accepted_by_reviewer += 1
                elif len(state.attempts) == 1:
                    tally.passed_first_try += 1
            elif state.status is AssetStatus.EXHAUSTED:
                tally.exhausted += 1
            else:
                tally.pending += 1
            for attempt in state.attempts:
                tally.attempts += 1
                if attempt.error:
                    tally.errored_attempts += 1
                if attempt.qc is None:
                    continue
                for check in attempt.qc.checks:
                    entry = tally.checks.setdefault(check.name, CheckTally())
                    if check.status is CheckStatus.PASSED:
                        entry.passed += 1
                    elif check.status is CheckStatus.FAILED:
                        entry.failed += 1
                    else:
                        entry.skipped += 1
                        entry.skip_reasons[check.detail or "no reason recorded"] += 1
    return summary


def _ordered(checks: dict[str, CheckTally]) -> list[tuple[str, CheckTally]]:
    known = [(name, checks[name]) for name in _CHECK_ORDER if name in checks]
    extra = sorted((name, tally) for name, tally in checks.items() if name not in _CHECK_ORDER)
    return known + extra


def _escape_md_cell(value: str) -> str:
    """Escape text for a Markdown table cell so untrusted strings cannot break or inject markup."""
    cleaned = value.replace("\r\n", " ").replace("\r", " ").replace("\n", " ")
    return cleaned.replace("|", "\\|").replace("`", "\\`").replace("<", "&lt;").replace(">", "&gt;")


def render(summary: QCReportSummary, markdown: bool = False) -> str:
    """A text or Markdown report. Markdown is meant to be pasted into a README or an issue."""
    lines: list[str] = []
    for label, tally in (("Stills", summary.stills), ("Videos", summary.videos)):
        lines.append(f"{'## ' if markdown else ''}{label}")
        lines.append(
            f"{tally.shots} shots, {tally.attempts} attempts ({tally.errored_attempts} errored). "
            f"Passed: {tally.passed} ({tally.passed_first_try} on the first attempt, "
            f"{tally.accepted_by_reviewer} accepted by a reviewer). "
            f"Exhausted: {tally.exhausted}. Pending: {tally.pending}."
        )
        if not tally.checks:
            lines.append("No QC reports recorded." + ("\n" if markdown else ""))
            continue
        if markdown:
            lines.append("")
            lines.append("| Check | Ran | Passed | Failed | Skipped | Skip reasons |")
            lines.append("|---|---|---|---|---|---|")
        else:
            lines.append(
                f"  {'check':<18}{'ran':>6}{'pass':>6}{'fail':>6}{'skip':>6}  skip reasons"
            )
        for name, entry in _ordered(tally.checks):
            reasons = "; ".join(
                f"{reason} ({count})" for reason, count in entry.skip_reasons.most_common(3)
            )
            if markdown:
                cell_name = _escape_md_cell(name)
                cell_reasons = _escape_md_cell(reasons)
                lines.append(
                    f"| `{cell_name}` | {entry.ran} | {entry.passed} | {entry.failed} | "
                    f"{entry.skipped} | {cell_reasons} |"
                )
            else:
                lines.append(
                    f"  {name:<18}{entry.ran:>6}{entry.passed:>6}{entry.failed:>6}"
                    f"{entry.skipped:>6}  {reasons}"
                )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


# --------------------------------------------------------------------------- attempt & reject stats

_SHOT_TYPE_BUCKETS = ("singing", "broll", "chained", "performance")


@dataclass
class AttemptBucketStats:
    assets: int = 0
    settled_count: int = 0
    exhausted_count: int = 0
    pending_count: int = 0
    attempts: int = 0
    errored_attempts: int = 0
    passed_attempts: int = 0
    mean_attempts_per_passed: float = 0.0
    p90_attempts: float = 0.0
    p90_multiplier: float = 0.0  # p90_attempts / mean_attempts_per_passed


@dataclass
class AttemptStatsSummary:
    total_shots: int = 0
    settled_count: int = 0
    exhausted_count: int = 0
    pending_count: int = 0
    total_attempts: int = 0
    errored_attempts: int = 0
    mean_attempts_per_passed: float = 0.0
    p90_attempts: float = 0.0
    p90_multiplier: float = 0.0
    reviewed_count: int = 0  # shots a reviewer approved or rejected at least once
    approved_count: int = 0  # shots whose current decision is an approval
    rejected_count: int = 0  # shots a reviewer rejected at least once
    total_rejections: int = 0  # every rejection, counting repeat rejections of a shot
    human_reject_rate: float = 0.0  # rejected_count / reviewed_count
    by_kind: dict[str, AttemptBucketStats] = field(default_factory=dict)
    by_shot_type: dict[str, AttemptBucketStats] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = (pct / 100.0) * (len(sorted_values) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = rank - lo
    return float(sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * frac)


def _shot_type_bucket(shot: Shot) -> str:
    if shot.continuity is ContinuityMode.CONTINUE:
        return "chained"
    if shot.type is ShotType.LIPSYNC:
        return "singing"
    if shot.type is ShotType.BROLL:
        return "broll"
    return "performance"


@dataclass
class _BucketAcc:
    stats: AttemptBucketStats = field(default_factory=AttemptBucketStats)
    passed_attempt_counts: list[float] = field(default_factory=list)

    def record(self, status: AssetStatus, attempts_count: int, errored: int) -> None:
        self.stats.assets += 1
        self.stats.attempts += attempts_count
        self.stats.errored_attempts += errored
        if status is AssetStatus.PASSED:
            self.stats.settled_count += 1
            self.stats.passed_attempts += attempts_count
            self.passed_attempt_counts.append(float(attempts_count))
        elif status is AssetStatus.EXHAUSTED:
            self.stats.exhausted_count += 1
        else:
            self.stats.pending_count += 1

    def finalize(self) -> AttemptBucketStats:
        if self.stats.settled_count > 0:
            self.stats.mean_attempts_per_passed = round(
                self.stats.passed_attempts / self.stats.settled_count, 4
            )
            ordered = sorted(self.passed_attempt_counts)
            p90 = _percentile(ordered, 90.0)
            self.stats.p90_attempts = round(p90, 4)
            # How much a slow (p90) asset costs relative to the average passed one: 1.0 means
            # attempts are uniform, 2.0 means the slowest tenth take twice the mean.
            mean = self.stats.passed_attempts / self.stats.settled_count
            self.stats.p90_multiplier = round(p90 / mean, 4)
        return self.stats


def _rejections(shot: Shot) -> int:
    """Times a reviewer rejected ``shot``, surviving the review reset on regeneration.

    A manifest written before ``rejection_count`` existed only has the current decision, so a
    shot rejected but not yet regenerated counts once.
    """
    pending_legacy = int(shot.review.decision is ReviewDecision.REJECTED)
    return max(shot.rejection_count, pending_legacy)


def attempt_stats(manifest: Manifest) -> AttemptStatsSummary:
    """Compute retry, attempt-multiplier, and human-rejection statistics for ``manifest``."""
    overall = _BucketAcc()
    kind_acc = {kind.value: _BucketAcc() for kind in (AssetKind.STILL, AssetKind.VIDEO)}
    shot_acc = {name: _BucketAcc() for name in _SHOT_TYPE_BUCKETS}

    approved_count = 0
    rejected_count = 0
    reviewed_count = 0
    total_rejections = 0

    for shot in manifest.shots:
        rejections = _rejections(shot)
        approved = shot.review.decision is ReviewDecision.APPROVED
        approved_count += approved
        rejected_count += rejections > 0
        reviewed_count += approved or rejections > 0
        total_rejections += rejections
        shot_bucket = _shot_type_bucket(shot)
        for kind in (AssetKind.STILL, AssetKind.VIDEO):
            state = shot.asset(kind)
            attempts_count = len(state.attempts)
            errored = sum(1 for a in state.attempts if a.error)
            overall.record(state.status, attempts_count, errored)
            kind_acc[kind.value].record(state.status, attempts_count, errored)
            shot_acc[shot_bucket].record(state.status, attempts_count, errored)

    overall_stats = overall.finalize()
    reject_rate = round(rejected_count / reviewed_count, 4) if reviewed_count else 0.0

    return AttemptStatsSummary(
        total_shots=len(manifest.shots),
        settled_count=overall_stats.settled_count,
        exhausted_count=overall_stats.exhausted_count,
        pending_count=overall_stats.pending_count,
        total_attempts=overall_stats.attempts,
        errored_attempts=overall_stats.errored_attempts,
        mean_attempts_per_passed=overall_stats.mean_attempts_per_passed,
        p90_attempts=overall_stats.p90_attempts,
        p90_multiplier=overall_stats.p90_multiplier,
        reviewed_count=reviewed_count,
        approved_count=approved_count,
        rejected_count=rejected_count,
        total_rejections=total_rejections,
        human_reject_rate=reject_rate,
        by_kind={k: acc.finalize() for k, acc in kind_acc.items()},
        by_shot_type={k: acc.finalize() for k, acc in shot_acc.items()},
    )


def render_attempt_stats(stats: AttemptStatsSummary, markdown: bool = False) -> str:
    """Format ``AttemptStatsSummary`` as a plain-text or Markdown summary block."""
    lines: list[str] = []
    lines.append(f"{'## ' if markdown else ''}Attempt & Rejection Stats")
    lines.append(
        f"Settled assets: {stats.settled_count} "
        f"(exhausted: {stats.exhausted_count}, pending: {stats.pending_count}). "
        f"Mean attempts/passed: {stats.mean_attempts_per_passed:.2f}, "
        f"p90: {stats.p90_attempts:.2f} ({stats.p90_multiplier:.2f}x the mean). "
        f"Human reject rate: {stats.human_reject_rate:.1%} "
        f"({stats.rejected_count}/{stats.reviewed_count} reviewed shots rejected, "
        f"{stats.total_rejections} rejection(s) in total)."
    )
    rows: list[tuple[str, AttemptBucketStats]] = [
        (f"kind:{k}", v) for k, v in stats.by_kind.items()
    ] + [(f"shot:{k}", v) for k, v in stats.by_shot_type.items()]
    if markdown:
        lines.append("")
        lines.append("| Bucket | Assets | Settled | Attempts | Mean/passed | p90 |")
        lines.append("|---|---|---|---|---|---|")
        for label, bucket in rows:
            lines.append(
                f"| `{_escape_md_cell(label)}` | {bucket.assets} | {bucket.settled_count} | "
                f"{bucket.attempts} | {bucket.mean_attempts_per_passed:.2f} | "
                f"{bucket.p90_attempts:.2f} |"
            )
    else:
        lines.append(
            f"  {'bucket':<18}{'assets':>8}{'settled':>9}{'attempts':>10}"
            f"{'mean/pass':>11}{'p90':>8}"
        )
        for label, bucket in rows:
            lines.append(
                f"  {label:<18}{bucket.assets:>8}{bucket.settled_count:>9}"
                f"{bucket.attempts:>10}{bucket.mean_attempts_per_passed:>11.2f}"
                f"{bucket.p90_attempts:>8.2f}"
            )
    return "\n".join(lines).rstrip() + "\n"
