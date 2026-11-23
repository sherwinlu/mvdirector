# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Per-check pass, fail and skip counts over every attempt in a manifest.

Used by ``mvd qc-report`` so the result of a run can be published as numbers rather than as
"it worked". Skipped checks are counted with their reasons, which is how a run that never
exercised a check shows up as such.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

from mvdirector.models import AssetKind, AssetStatus, CheckStatus, Manifest

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
    checks: dict[str, CheckTally] = field(default_factory=lambda: defaultdict(CheckTally))


@dataclass
class QCReportSummary:
    stills: AssetTally
    videos: AssetTally

    def tally(self, kind: AssetKind) -> AssetTally:
        return self.stills if kind is AssetKind.STILL else self.videos


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
                    entry = tally.checks[check.name]
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
