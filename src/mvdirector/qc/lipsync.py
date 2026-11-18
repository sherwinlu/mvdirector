# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Lip-sync measurement: does the mouth open when the voice is loud?

Singing has a strong, simple cue: mouth opening tracks vocal loudness. Correlating
the two series gives an objective score with no trained sync model. The maths here
is pure numpy; face tracking is supplied by a MouthTracker.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SyncMeasurement:
    correlation: float
    lag_frames: int  # positive: the mouth moves after the audio
    face_presence: float  # share of frames where a face was tracked


def envelope_at_fps(samples: np.ndarray, sample_rate: int, fps: float, n_frames: int) -> np.ndarray:
    """Vocal loudness (RMS) over each video frame's time window."""
    edges = np.round(np.arange(n_frames + 1) * sample_rate / fps).astype(int)
    envelope = np.zeros(n_frames)
    for index in range(n_frames):
        window = samples[edges[index] : edges[index + 1]]
        if window.size:
            envelope[index] = float(np.sqrt(np.mean(window.astype(np.float64) ** 2)))
    return envelope


def fill_gaps(series: np.ndarray) -> np.ndarray:
    """Linearly interpolate NaN values; an all-NaN series becomes zeros."""
    valid = ~np.isnan(series)
    if not valid.any():
        return np.zeros_like(series)
    positions = np.arange(len(series))
    return np.interp(positions, positions[valid], series[valid])


def smooth(series: np.ndarray, width: int = 3) -> np.ndarray:
    """Moving average. Edges are padded with their own value so no false dip appears."""
    if len(series) < width:
        return series
    padded = np.pad(series, width // 2, mode="edge")
    return np.convolve(padded, np.ones(width) / width, mode="valid")


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or a.std() < 1e-9 or b.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def measure_sync(mouth: np.ndarray, envelope: np.ndarray, max_lag_frames: int) -> SyncMeasurement:
    """Best correlation between mouth opening and loudness within ``±max_lag_frames``."""
    length = min(len(mouth), len(envelope))
    mouth, envelope = mouth[:length], envelope[:length]
    presence = float(np.mean(~np.isnan(mouth))) if length else 0.0
    mouth_series, envelope_series = smooth(fill_gaps(mouth)), smooth(envelope)

    best_correlation, best_lag = -1.0, 0
    for lag in range(-max_lag_frames, max_lag_frames + 1):
        if lag >= 0:
            a, b = mouth_series[lag:], envelope_series[: length - lag]
        else:
            a, b = mouth_series[: length + lag], envelope_series[-lag:]
        correlation = _pearson(a, b)
        if correlation > best_correlation:
            best_correlation, best_lag = correlation, lag
    return SyncMeasurement(
        correlation=max(0.0, best_correlation), lag_frames=best_lag, face_presence=presence
    )


def is_silent(envelope: np.ndarray, floor: float = 1e-4) -> bool:
    """True when there is no vocal to sync against."""
    return envelope.size == 0 or float(envelope.max()) < floor
