# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Motion continuity and beat-to-visual-motion analysis in pure NumPy.

Measures:
1. Camera-motion continuity across shot joins (duplicate frames, skipped frames, abrupt camera
   velocity jumps, and speed dips) via phase-correlation translation + scale velocity estimation.
2. Beat-to-visual-motion cross-correlation against Gaussian beat pulses, paired with a
   circular-shift permutation baseline for chance-corrected significance scoring.

Credits & Attribution:
    - Camera-motion join continuity checking inspired by ``atomtanstudio/sound-and-vision``
      (https://github.com/atomtanstudio/sound-and-vision — ``server/sv/video/jointool.py``).
    - Beat-vs-motion Gaussian pulse cross-correlation inspired by
      ``multimodal-art-projection/AutoMV``
      (https://github.com/multimodal-art-projection/AutoMV — ``evaluate/objective/beat.py``).
    - Circular-shift permutation significance baseline inspired by
      ``tysonfarrington-cmyk/xenoform``
      (https://github.com/tysonfarrington-cmyk/xenoform — ``src/xenoform/performer.py``).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

_FLOW_SIDE = 96


@dataclass(frozen=True)
class JoinContinuityReport:
    """Continuity diagnostics across the boundary between two consecutive clips."""

    duplicate_boundary_frame: bool
    velocity_jump: float
    speed_dip_ratio: float
    pre_speed: float
    join_speed: float
    post_speed: float
    issues: list[str] = field(default_factory=list)

    @property
    def smooth(self) -> bool:
        return not self.issues


@dataclass(frozen=True)
class BeatMotionSyncReport:
    """Cross-correlation between visual motion energy and musical beat pulses."""

    correlation: float
    best_lag_seconds: float
    chance_baseline: float
    z_score: float


def _to_luma(frame: np.ndarray, side: int = _FLOW_SIDE) -> np.ndarray:
    img = Image.fromarray(frame).convert("L").resize((side, side), Image.Resampling.BILINEAR)
    return np.asarray(img, dtype=np.float64) / 255.0


def estimate_frame_shift(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """Estimate ``(dx, dy, mad)`` between two greyscale frames via FFT phase correlation.

    ``dx`` and ``dy`` are expressed in normalized units of the downscaled grid side.
    """
    if a.ndim != 2 or b.ndim != 2 or a.shape != b.shape or a.size == 0:
        return 0.0, 0.0, 0.0
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        return 0.0, 0.0, 0.0
    h, w = a.shape
    mad = float(np.mean(np.abs(b - a)))
    if mad < 1e-6:
        return 0.0, 0.0, mad

    win = np.outer(np.hanning(h), np.hanning(w))
    fa = np.fft.fft2((a - float(np.mean(a))) * win)
    fb = np.fft.fft2((b - float(np.mean(b))) * win)
    cross = fa * np.conj(fb)
    denom = np.abs(cross)
    cross = np.where(denom > 1e-12, cross / np.maximum(denom, 1e-12), 0.0)
    corr = np.real(np.fft.ifft2(cross))
    peak_y, peak_x = np.unravel_index(int(np.argmax(corr)), corr.shape)
    dy = float(peak_y - h if peak_y > h // 2 else peak_y)
    dx = float(peak_x - w if peak_x > w // 2 else peak_x)
    return dx / max(w, 1), dy / max(h, 1), mad


def measure_join_continuity(
    prev_tail_frames: Sequence[np.ndarray],
    next_head_frames: Sequence[np.ndarray],
    *,
    duplicate_mad_max: float = 0.002,
    velocity_jump_max: float = 0.08,
    speed_dip_min_ratio: float = 0.55,
) -> JoinContinuityReport:
    """Audit camera motion smoothness across a boundary between two chained clips.

    Detects duplicated seam frames, abrupt velocity spikes (skipped frames / camera lurches), and
    mid-join speed dips (diffusion clips starting from rest).
    """
    if not prev_tail_frames or not next_head_frames:
        return JoinContinuityReport(
            duplicate_boundary_frame=False,
            velocity_jump=0.0,
            speed_dip_ratio=1.0,
            pre_speed=0.0,
            join_speed=0.0,
            post_speed=0.0,
            issues=[],
        )

    pre = [_to_luma(f) for f in prev_tail_frames[-4:]]
    post = [_to_luma(f) for f in next_head_frames[:4]]

    pre_vecs = [estimate_frame_shift(a, b) for a, b in zip(pre[:-1], pre[1:])]
    join_dx, join_dy, join_mad = estimate_frame_shift(pre[-1], post[0])
    post_vecs = [estimate_frame_shift(a, b) for a, b in zip(post[:-1], post[1:])]

    pre_speeds = [math.hypot(dx, dy) + mad * 0.25 for dx, dy, mad in pre_vecs]
    post_speeds = [math.hypot(dx, dy) + mad * 0.25 for dx, dy, mad in post_vecs]
    pre_speed = float(np.median(pre_speeds)) if pre_speeds else 0.0
    post_speed = float(np.median(post_speeds)) if post_speeds else 0.0
    join_speed = math.hypot(join_dx, join_dy) + join_mad * 0.25
    surround_speed = (
        0.5 * (pre_speed + post_speed)
        if (pre_speeds and post_speeds)
        else max(pre_speed, post_speed)
    )

    dup = join_mad <= duplicate_mad_max and surround_speed > duplicate_mad_max * 2.0
    if pre_vecs and post_vecs:
        mean_pre_dx = float(np.mean([v[0] for v in pre_vecs]))
        mean_pre_dy = float(np.mean([v[1] for v in pre_vecs]))
        mean_post_dx = float(np.mean([v[0] for v in post_vecs]))
        mean_post_dy = float(np.mean([v[1] for v in post_vecs]))
        vel_jump = max(
            math.hypot(join_dx - mean_pre_dx, join_dy - mean_pre_dy),
            math.hypot(mean_post_dx - mean_pre_dx, mean_post_dy - mean_pre_dy),
        )
    else:
        vel_jump = abs(join_speed - surround_speed)

    dip_ratio = join_speed / surround_speed if surround_speed > 1e-4 else 1.0

    issues: list[str] = []
    if dup:
        issues.append("duplicate seam frame at clip join")
    if vel_jump >= velocity_jump_max:
        issues.append(f"camera velocity jump across join ({vel_jump:.3f})")
    if surround_speed > 0.01 and dip_ratio < speed_dip_min_ratio:
        issues.append(f"camera speed dip at join ({dip_ratio:.0%} of surrounding speed)")

    return JoinContinuityReport(
        duplicate_boundary_frame=dup,
        velocity_jump=round(vel_jump, 4),
        speed_dip_ratio=round(dip_ratio, 4),
        pre_speed=round(pre_speed, 4),
        join_speed=round(join_speed, 4),
        post_speed=round(post_speed, 4),
        issues=issues,
    )


def measure_beat_motion_sync(
    frames: Sequence[np.ndarray],
    fps: float,
    beat_times: Sequence[float],
    *,
    sigma_seconds: float = 0.10,
    max_lag_seconds: float = 0.50,
    permutations: int = 100,
    seed: int = 0,
) -> BeatMotionSyncReport:
    """Measure normalized cross-correlation between frame motion energy and Gaussian beat pulses.

    Evaluates a circular-shift permutation baseline so periodic/noisy motion does not masquerade as
    true beat lock.
    """
    if len(frames) < 4 or not math.isfinite(fps) or fps <= 0.0 or not beat_times:
        return BeatMotionSyncReport(
            correlation=0.0, best_lag_seconds=0.0, chance_baseline=0.0, z_score=0.0
        )

    sigma_safe = sigma_seconds if math.isfinite(sigma_seconds) else 0.10
    lag_safe = (
        max_lag_seconds if (math.isfinite(max_lag_seconds) and max_lag_seconds >= 0.0) else 0.0
    )

    luma = [_to_luma(f, side=64) for f in frames]
    motion = np.array([float(np.mean(np.abs(b - a))) for a, b in zip(luma[:-1], luma[1:])])
    n = motion.size
    if n < 3 or float(np.std(motion)) < 1e-9:
        return BeatMotionSyncReport(
            correlation=0.0, best_lag_seconds=0.0, chance_baseline=0.0, z_score=0.0
        )

    times = (np.arange(n, dtype=np.float64) + 0.5) / fps
    pulses = np.zeros(n, dtype=np.float64)
    inv_2s2 = 1.0 / (2.0 * max(sigma_safe, 0.01) ** 2)
    for beat in beat_times:
        if math.isfinite(beat):
            pulses += np.exp(-((times - beat) ** 2) * inv_2s2)
    if float(np.std(pulses)) < 1e-9:
        return BeatMotionSyncReport(
            correlation=0.0, best_lag_seconds=0.0, chance_baseline=0.0, z_score=0.0
        )

    max_lag = min(n // 3, max(1, int(round(lag_safe * fps))))

    def _best_xcorr(series: np.ndarray, ref: np.ndarray) -> tuple[float, int]:
        best_r, best_l = -1.0, 0
        for lag in range(-max_lag, max_lag + 1):
            if lag >= 0:
                a, b = series[lag:], ref[: n - lag]
            else:
                a, b = series[: n + lag], ref[-lag:]
            if a.size < 3 or float(np.std(a)) < 1e-9 or float(np.std(b)) < 1e-9:
                continue
            r = float(np.corrcoef(a, b)[0, 1])
            if math.isfinite(r) and r > best_r:
                best_r, best_l = r, lag
        return max(0.0, best_r), best_l

    obs_corr, best_lag = _best_xcorr(motion, pulses)
    if permutations <= 0 or n <= 4:
        return BeatMotionSyncReport(
            correlation=round(obs_corr, 4),
            best_lag_seconds=round(best_lag / fps, 3),
            chance_baseline=0.0,
            z_score=0.0,
        )

    rng = np.random.default_rng(seed)
    shifts = rng.integers(1, n, size=permutations)
    null_scores = np.array([_best_xcorr(np.roll(motion, int(s)), pulses)[0] for s in shifts])
    baseline = float(np.mean(null_scores))
    std = float(np.std(null_scores))
    z = (obs_corr - baseline) / std if std > 1e-6 else 0.0
    return BeatMotionSyncReport(
        correlation=round(obs_corr, 4),
        best_lag_seconds=round(best_lag / fps, 3),
        chance_baseline=round(baseline, 4),
        z_score=round(z, 2),
    )
