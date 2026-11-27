# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Turn individual check results into one pass/fail decision."""

from __future__ import annotations

import math

from mvdirector.models import CheckResult, CheckStatus, QCReport


def linear_score(value: float, bad: float, good: float) -> float:
    """Map ``value`` onto 0-1, where ``bad`` scores 0 and ``good`` scores 1.

    Works for metrics where lower is better too: pass ``bad`` greater than ``good``.
    """
    if not (math.isfinite(value) and math.isfinite(bad) and math.isfinite(good)):
        return 0.0
    if bad == good:
        return 1.0 if value >= good else 0.0
    return float(min(1.0, max(0.0, (value - bad) / (good - bad))))


def floor_passing_score(value: float, minimum: float, good: float, pass_floor: float) -> float:
    """Map a passing metric in ``[minimum, good]`` onto ``[pass_floor, 1.0]``.

    Prevents a heavily weighted passing check right above its threshold from pulling an
    otherwise-passing asset below ``pass_score`` and causing avoidable retry renders.
    """
    if not all(math.isfinite(x) for x in (value, minimum, good, pass_floor)):
        return 0.0
    clamped_floor = min(1.0, max(0.0, pass_floor))
    progress = 1.0 if good <= minimum else linear_score(value, minimum, good)
    return float(clamped_floor + (1.0 - clamped_floor) * progress)


def skipped(name: str, reason: str) -> CheckResult:
    return CheckResult(name=name, status=CheckStatus.SKIPPED, detail=reason)


def aggregate(checks: list[CheckResult], weights: dict[str, float], pass_score: float) -> QCReport:
    """An asset passes when no hard check failed and its weighted score clears the bar.

    Skipped checks are recorded for the audit trail but do not affect the score.
    """
    active = [c for c in checks if c.status is not CheckStatus.SKIPPED]
    for check in active:
        check.weight = weights.get(check.name, check.weight)
    scored = [
        c
        for c in active
        if c.score is not None
        and math.isfinite(c.score)
        and math.isfinite(c.weight)
        and c.weight > 0.0
    ]
    total_weight = sum(c.weight for c in scored)
    if total_weight > 0:
        score = sum((c.score or 0.0) * c.weight for c in scored) / total_weight
    else:
        # Nothing is scored, so only a hard failure sinks the asset; an advisory failure is
        # reported without affecting it.
        score = 0.0 if any(c.hard and c.status is CheckStatus.FAILED for c in active) else 1.0
    hard_failure = any(c.hard and c.status is CheckStatus.FAILED for c in active)
    return QCReport(
        passed=not hard_failure and score >= pass_score, score=round(score, 4), checks=checks
    )
