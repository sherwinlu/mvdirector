# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Structural audio cue detection: first energy lift, peaks/drops, and breakdown valleys.

Pure NumPy/Librosa signal-processing helpers that locate musical landmarks directly from an
audio mix without requiring lyrics or stem separation.

Credits & Attribution:
    Composite energy/onset/brightness landmark detection ported from
    ``sherwinlu/weaveshift`` (``weaveshift/cues.py``).
"""

from __future__ import annotations

import math
import warnings
from typing import Literal

import numpy as np

from mvdirector.models import AudioCues

_ANALYSIS_SR = 22050
_HOP_LENGTH = 512
_MIN_LIFT_SPREAD_DB = 1.5


def _median_step_sec(times: np.ndarray, default: float = 0.5) -> float:
    """Median frame spacing in seconds; falls back for degenerate short tracks."""
    if len(times) > 1:
        step = float(np.median(np.diff(times)))
        if math.isfinite(step) and step > 0.0:
            return step
    return default


def _moving_average(x: np.ndarray, n: int) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if len(arr) <= 1:
        return arr
    width = min(len(arr), max(1, int(n)))
    if width == 1:
        return arr
    kernel = np.ones(width, dtype=np.float64) / width
    return np.convolve(arr, kernel, mode="same")


def _robust_z(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.size == 0:
        return arr
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    if mad < 1e-9:
        std = float(np.std(arr))
        if std < 1e-9:
            return np.zeros_like(arr, dtype=np.float64)
        return (arr - med) / std
    return 0.6745 * (arr - med) / mad


def _pick_local_peaks(
    times: np.ndarray,
    score: np.ndarray,
    min_time: float,
    max_time: float,
    min_distance_sec: float,
    count: int,
) -> list[float]:
    """Select well-separated local maxima within ``[min_time, max_time]``."""
    if len(times) < 3:
        return []
    mask = (times >= min_time) & (times <= max_time)
    idxs = np.where(mask)[0]
    if len(idxs) == 0:
        return []

    candidates: list[tuple[float, float]] = []
    for i in idxs:
        if i <= 0 or i >= len(score) - 1:
            continue
        if (
            score[i] >= score[i - 1]
            and score[i] >= score[i + 1]
            and (score[i] > score[i - 1] or score[i] > score[i + 1])
        ):
            candidates.append((float(score[i]), float(times[i])))

    candidates.sort(reverse=True, key=lambda pair: pair[0])

    chosen: list[float] = []
    for _, t in candidates:
        if all(abs(t - existing) >= min_distance_sec for existing in chosen):
            chosen.append(t)
        if len(chosen) >= count:
            break
    return sorted(chosen)


def _first_sustained_energy_lift(
    times: np.ndarray,
    energy: np.ndarray,
    duration_sec: float,
) -> float | None:
    """Find the first point where smoothed energy meaningfully exceeds intro baseline."""
    if len(times) == 0 or duration_sec <= 0.0:
        return None

    intro_floor = min(20.0, duration_sec * 0.25)
    intro_end = min(max(intro_floor, duration_sec * 0.12), 45.0)
    search_start = min(8.0, duration_sec * 0.05)
    search_end = min(duration_sec * 0.85, max(search_start + 3.0, duration_sec * 0.55))

    intro_mask = times <= intro_end
    baseline = (
        float(np.percentile(energy[intro_mask], 60))
        if np.any(intro_mask)
        else float(np.percentile(energy, 35))
    )
    high = float(np.percentile(energy, 72))
    if not (math.isfinite(baseline) and math.isfinite(high)):
        return None
    if high - baseline < _MIN_LIFT_SPREAD_DB:
        return None
    threshold = baseline + 0.45 * (high - baseline)

    frame_step = _median_step_sec(times)
    sustain_sec = min(4.0, max(1.0, duration_sec * 0.12))
    sustain_frames = max(3, int(round(sustain_sec / max(frame_step, 0.1))))

    mask = (times >= search_start) & (times <= search_end)
    for i in np.where(mask)[0]:
        j = min(len(energy), int(i) + sustain_frames)
        if j <= i:
            continue
        if float(np.mean(energy[i:j] >= threshold)) >= 0.75:
            return float(times[i])
    return None


def _find_breakdown_candidate(
    times: np.ndarray,
    energy: np.ndarray,
    duration_sec: float,
) -> float | None:
    """Find a low-energy valley after the track is underway (requires >= 90 s track)."""
    if len(times) == 0 or duration_sec < 90.0:
        return None

    min_time = max(45.0, duration_sec * 0.25)
    max_time = duration_sec * 0.82
    mask = (times >= min_time) & (times <= max_time)
    if not np.any(mask):
        return None
    idxs = np.where(mask)[0]
    i = int(idxs[int(np.argmin(energy[idxs]))])
    return float(times[i])


def _shift_cue(value: float | None, offset: float, duration: float) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    clamped = max(0.0, min(duration, value))
    return round(offset + clamped, 3)


def detect_audio_cues(
    samples: np.ndarray,
    sr: int = _ANALYSIS_SR,
    *,
    offset: float = 0.0,
) -> AudioCues:
    """Detect ``first_lift``, ``main_peak``, ``second_peak``, and ``breakdown`` from mono audio."""
    if sr <= 0:
        return AudioCues(confidence="low")
    arr = np.asarray(samples, dtype=np.float32)
    if arr.ndim > 2:
        return AudioCues(confidence="low")
    if arr.ndim == 2:
        axis = 0 if arr.shape[0] <= arr.shape[-1] else -1
        arr = np.mean(arr, axis=axis, dtype=np.float32)
    clean = np.nan_to_num(arr.reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
    if clean.size < sr:
        return AudioCues(confidence="low")
    peak_amp = float(np.max(np.abs(clean)))
    if not math.isfinite(peak_amp) or peak_amp < 1e-6:
        return AudioCues(confidence="low")

    base_offset = max(0.0, float(offset)) if math.isfinite(offset) else 0.0
    duration_sec = clean.size / float(sr)

    import librosa

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        rms = librosa.feature.rms(y=clean, frame_length=2048, hop_length=_HOP_LENGTH)[0]
        times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=_HOP_LENGTH)
        frame_step = _median_step_sec(times)
        rms_db = librosa.power_to_db(np.maximum(rms, 1e-10) ** 2, ref=np.max)
        energy = _moving_average(rms_db, max(3, int(round(2.5 / frame_step))))

        onset_env = librosa.onset.onset_strength(y=clean, sr=sr, hop_length=_HOP_LENGTH)
        onset_times = librosa.frames_to_time(
            np.arange(len(onset_env)), sr=sr, hop_length=_HOP_LENGTH
        )
        onset_smooth = _moving_average(
            onset_env, max(3, int(round(2.0 / _median_step_sec(onset_times))))
        )
        onset_interp = (
            np.interp(times, onset_times, onset_smooth)
            if len(onset_times) > 0 and len(times) > 0
            else np.zeros_like(times)
        )

        centroid = librosa.feature.spectral_centroid(y=clean, sr=sr, hop_length=_HOP_LENGTH)[0]
        centroid_times = librosa.frames_to_time(
            np.arange(len(centroid)), sr=sr, hop_length=_HOP_LENGTH
        )
        centroid_interp = (
            np.interp(times, centroid_times, centroid)
            if len(centroid_times) > 0 and len(times) > 0
            else np.zeros_like(times)
        )
        centroid_z = (
            _robust_z(_moving_average(centroid_interp, max(3, int(round(3.0 / frame_step)))))
            if len(centroid_interp) > 0
            else np.zeros_like(times)
        )

    energy_z = _robust_z(energy)
    onset_z = _robust_z(onset_interp)
    score = 0.55 * energy_z + 0.35 * onset_z + 0.10 * centroid_z
    score = _moving_average(score, max(3, int(round(1.5 / frame_step))))

    if duration_sec >= 40.0:
        peak_min = max(20.0, duration_sec * 0.12)
        peak_max = max(peak_min + 10.0, duration_sec * 0.90)
        min_distance = 35.0
    else:
        peak_min = duration_sec * 0.12
        peak_max = duration_sec * 0.90
        min_distance = max(2.0, duration_sec * 0.25)

    peak_candidates = _pick_local_peaks(
        times=times,
        score=score,
        min_time=peak_min,
        max_time=peak_max,
        min_distance_sec=min_distance,
        count=4,
    )

    first_lift = _first_sustained_energy_lift(times, energy, duration_sec)
    main_peak = peak_candidates[0] if peak_candidates else None
    second_peak = peak_candidates[1] if len(peak_candidates) > 1 else None

    breakdown = _find_breakdown_candidate(times, energy, duration_sec)

    confidence: Literal["low", "medium", "high"] = "low"
    if main_peak is not None and len(score) > 0:
        pctl = float(np.mean(score <= float(np.interp(main_peak, times, score))))
        if pctl > 0.92:
            confidence = "high"
        elif pctl > 0.80:
            confidence = "medium"

    return AudioCues(
        first_lift=_shift_cue(first_lift, base_offset, duration_sec),
        main_peak=_shift_cue(main_peak, base_offset, duration_sec),
        second_peak=_shift_cue(second_peak, base_offset, duration_sec),
        breakdown=_shift_cue(breakdown, base_offset, duration_sec),
        confidence=confidence,
    )


def _fmt_cue_time(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    minutes, secs = divmod(total, 60)
    return f"{minutes}:{secs:02d}"


def format_cues_summary(cues: AudioCues | None) -> str:
    """Compact human-readable landmark summary for planning and prepare prompts."""
    if cues is None:
        return ""
    parts: list[str] = []
    for label, value in (
        ("first_lift", cues.first_lift),
        ("main_peak", cues.main_peak),
        ("second_peak", cues.second_peak),
        ("breakdown", cues.breakdown),
    ):
        if value is not None:
            parts.append(f"{label}={_fmt_cue_time(value)} ({value:.1f}s)")
    if not parts:
        return ""
    return ", ".join(parts) + f" [confidence={cues.confidence}]"
