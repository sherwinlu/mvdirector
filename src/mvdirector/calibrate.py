# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Suggest QC thresholds from clips a person has labelled good or bad.

Every check records the number it measured under ``metrics`` in the manifest. Given labels for
some attempts, this module finds, for each metric, the cut that best separates good from bad
and names the config key it belongs to. It is a starting point for ``project.yaml``, not a
verdict: twenty to thirty labelled clips per shot type is the minimum worth acting on.

Labels come from a CSV or JSON file, or from review decisions already in the manifest.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mvdirector.errors import ConfigError
from mvdirector.models import AssetKind, Manifest, ReviewDecision

# Metric name -> (config key, True when a higher value is better).
METRIC_CONFIG: dict[str, tuple[str, bool]] = {
    "correlation": ("qc.video.lipsync_min", True),
    "similarity": ("qc.image.identity_hard_min / qc.video.identity_hard_min", True),
    "laplacian_variance": ("qc.image.sharpness_bad", True),
    "face_height_fraction": ("qc.image.min_face_fraction[<framing>]", True),
    "face_presence": ("qc.video.face_presence_min", True),
    "median_motion": ("qc.video.static_max_motion", True),
    "peak_motion": ("qc.video.jump_cut_threshold", False),
    "p95_motion": ("qc.video.flicker_bad", False),
}
_IGNORED_METRICS = {"width", "height", "duration", "lag_seconds"}


@dataclass(frozen=True)
class Label:
    shot_id: str
    kind: AssetKind
    attempt: int | None  # None means the selected attempt
    good: bool


@dataclass(frozen=True)
class Suggestion:
    metric: str
    config_key: str
    higher_is_better: bool
    good_count: int
    bad_count: int
    threshold: float
    balanced_accuracy: float
    good_median: float
    bad_median: float


def load_labels(path: Path) -> list[Label]:
    """Read ``shot,kind,attempt,label`` rows from a CSV, or a JSON list of the same fields.

    ``kind`` is ``still`` or ``video``. ``attempt`` may be empty for the selected attempt.
    ``label`` is ``good`` or ``bad`` (``1``/``0`` and ``pass``/``fail`` are accepted).
    """
    if not path.is_file():
        raise ConfigError(f"Labels file not found: {path}")
    text = path.read_text(encoding="utf-8")
    rows: list[dict[str, object]]
    if path.suffix.lower() == ".json":
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ConfigError(f"{path}: invalid JSON ({exc})") from exc
        if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
            raise ConfigError(f"{path} must hold a JSON list of label objects")
        rows = [dict(item) for item in data]
    else:
        rows = [dict(row) for row in csv.DictReader(text.splitlines())]
    labels = []
    for index, row in enumerate(rows, start=1):
        try:
            shot_id = str(row["shot"]).strip()
            if not shot_id:
                raise ValueError("shot must not be empty")
            kind = AssetKind(str(row.get("kind", "video")).strip().lower())
            raw_attempt = row.get("attempt")
            attempt = int(str(raw_attempt)) if raw_attempt not in (None, "") else None
            good = _parse_label(str(row["label"]))
        except (KeyError, ValueError, TypeError) as exc:
            raise ConfigError(f"{path} row {index}: {exc}") from exc
        labels.append(Label(shot_id, kind, attempt, good))
    if not labels:
        raise ConfigError(f"{path} holds no labels")
    return labels


def _parse_label(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in ("good", "pass", "passed", "ok", "1", "true", "yes", "approved"):
        return True
    if lowered in ("bad", "fail", "failed", "0", "false", "no", "rejected"):
        return False
    raise ValueError(f"label must be good or bad, not {value!r}")


def labels_from_reviews(manifest: Manifest) -> list[Label]:
    """Approved shots count as good videos; rejected shots count as bad."""
    labels = []
    for shot in manifest.shots:
        decision = shot.review.decision
        if decision is None or shot.video.selected_attempt is None:
            continue
        labels.append(
            Label(
                shot.id,
                AssetKind.VIDEO,
                shot.video.selected_attempt,
                decision is ReviewDecision.APPROVED,
            )
        )
    return labels


def collect_samples(manifest: Manifest, labels: list[Label]) -> dict[str, list[tuple[float, bool]]]:
    """Metric name -> (value, is_good) for every labelled attempt that recorded it."""
    samples: dict[str, list[tuple[float, bool]]] = {}
    for label in labels:
        try:
            shot = manifest.shot(label.shot_id)
        except KeyError as exc:
            raise ConfigError(f"Label names unknown shot {label.shot_id!r}") from exc
        state = shot.asset(label.kind)
        number = label.attempt if label.attempt is not None else state.selected_attempt
        if number is None:
            raise ConfigError(
                f"{label.shot_id} {label.kind.value} has no selected attempt (no QC report to "
                "learn from)"
            )
        attempt = next((a for a in state.attempts if a.number == number), None)
        if attempt is None or attempt.qc is None:
            raise ConfigError(
                f"{label.shot_id} {label.kind.value} attempt {number} has no QC report to learn "
                "from"
            )
        for check in attempt.qc.checks:
            for metric, value in check.metrics.items():
                if metric in _IGNORED_METRICS:
                    continue
                val = float(value)
                if math.isfinite(val):
                    samples.setdefault(metric, []).append((val, label.good))
    return samples


def suggest(samples: dict[str, list[tuple[float, bool]]]) -> list[Suggestion]:
    suggestions = []
    for metric, pairs in sorted(samples.items()):
        good = np.array([v for v, ok in pairs if ok], dtype=np.float64)
        bad = np.array([v for v, ok in pairs if not ok], dtype=np.float64)
        if good.size == 0 or bad.size == 0:
            continue
        key, higher = METRIC_CONFIG.get(metric, (f"(no config key for {metric})", True))
        threshold, accuracy = _best_cut(good, bad, higher)
        suggestions.append(
            Suggestion(
                metric=metric,
                config_key=key,
                higher_is_better=higher,
                good_count=int(good.size),
                bad_count=int(bad.size),
                threshold=threshold,
                balanced_accuracy=accuracy,
                good_median=float(np.median(good)),
                bad_median=float(np.median(bad)),
            )
        )
    return suggestions


def _best_cut(good: np.ndarray, bad: np.ndarray, higher_is_better: bool) -> tuple[float, float]:
    """The cut between consecutive sorted values that maximises balanced accuracy."""
    values = np.unique(np.concatenate([good, bad]))
    if values.size == 1:
        return float(values[0]), 0.5
    candidates = (values[:-1] + values[1:]) / 2.0
    best_cut, best_score = float(candidates[0]), -1.0
    for cut in candidates:
        if higher_is_better:
            tpr = float(np.mean(good >= cut))
            tnr = float(np.mean(bad < cut))
        else:
            tpr = float(np.mean(good <= cut))
            tnr = float(np.mean(bad > cut))
        score = (tpr + tnr) / 2.0
        if score > best_score:
            best_cut, best_score = float(cut), score
    return round(best_cut, 4), round(best_score, 3)


def render(suggestions: list[Suggestion], label_count: int) -> str:
    if not suggestions:
        return (
            f"{label_count} label(s) read, but no metric had both good and bad examples. "
            "Label some clips of each kind, or check that the checks ran (mvd qc-report).\n"
        )
    lines = [
        f"Thresholds suggested from {label_count} labelled attempt(s). Treat fewer than 20 "
        "labels per metric as a hint, not a setting.",
        "",
        f"  {'metric':<22}{'good':>5}{'bad':>5}{'good med':>10}{'bad med':>9}{'cut':>9}"
        f"{'bal.acc':>9}  config key",
    ]
    for s in suggestions:
        direction = ">=" if s.higher_is_better else "<="
        lines.append(
            f"  {s.metric:<22}{s.good_count:>5}{s.bad_count:>5}{s.good_median:>10.4f}"
            f"{s.bad_median:>9.4f}{direction + format(s.threshold, '.4f'):>9}"
            f"{s.balanced_accuracy:>9.3f}  {s.config_key}"
        )
    lines.append("")
    lines.append(
        "A balanced accuracy near 0.5 means the metric does not separate your labels; do not "
        "tighten that threshold."
    )
    return "\n".join(lines) + "\n"
