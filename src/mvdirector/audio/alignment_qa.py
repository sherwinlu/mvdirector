# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Pre-spend multi-witness lyric alignment verification.

Cross-checks forced/aligned lyric timestamps against independent acoustic cues before GPU or API
spend begins:
1. Pure-NumPy normalized cross-correlation (NCCF) autocorrelation voicing periodicity.
2. RMS-gated high-frequency (>3.4 kHz) consonant-band energy ratio.
3. Cut-boundary word-split witness flagging boundaries that slice through active sung words.

Credits & Attribution:
    Techniques inspired by ``JavaDerek/music-video-pipeline``
    (https://github.com/JavaDerek/music-video-pipeline — ``alignment_quality.py``,
    ``voicing.py``, and ``join_witness.py``).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from mvdirector.models import AlignmentWarning, LyricLine, TimeSpan

_VOICING_FRAME_SECONDS = 0.032
_VOICING_HOP_SECONDS = 0.016
_VOICING_MIN_HZ = 75.0
_VOICING_MAX_HZ = 600.0
_VOICING_NCCF_THRESHOLD = 0.45
_CONSONANT_CUTOFF_HZ = 3400.0
_SILENCE_GATE_DB = -15.0  # relative to median active RMS (prevents silent-tail ratio blowup)
_MIN_RMS_FLOOR = 1e-4


def _mono_finite(samples: np.ndarray) -> np.ndarray:
    wav = np.asarray(samples, dtype=np.float64)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if wav.size and not np.all(np.isfinite(wav)):
        wav = np.nan_to_num(wav, nan=0.0, posinf=0.0, neginf=0.0)
    return wav


def voicing_ratio(
    samples: np.ndarray,
    sr: int,
    *,
    min_hz: float = _VOICING_MIN_HZ,
    max_hz: float = _VOICING_MAX_HZ,
    nccf_threshold: float = _VOICING_NCCF_THRESHOLD,
) -> float:
    """Share of non-silent frames exhibiting periodic vocal pitch via NCCF autocorrelation.

    Returns a value in ``[0.0, 1.0]`` (0.0 for silence or unvoiced noise/percussion).
    """
    wav = _mono_finite(samples)
    if wav.size == 0 or sr <= 0:
        return 0.0

    frame_len = max(32, int(round(_VOICING_FRAME_SECONDS * sr)))
    hop_len = max(16, int(round(_VOICING_HOP_SECONDS * sr)))
    if wav.size < frame_len:
        return 0.0

    min_lag = max(1, int(math.floor(sr / max(max_hz, 1.0))))
    max_lag = min(frame_len // 2, int(math.ceil(sr / max(min_hz, 1.0))))
    if max_lag <= min_lag:
        return 0.0

    n_frames = 1 + (wav.size - frame_len) // hop_len
    if n_frames <= 0:
        return 0.0

    rms_vals = np.empty(n_frames, dtype=np.float64)
    for i in range(n_frames):
        seg = wav[i * hop_len : i * hop_len + frame_len]
        rms_vals[i] = float(np.sqrt(np.mean(seg * seg)))

    active_mask = rms_vals >= _MIN_RMS_FLOOR
    if not np.any(active_mask):
        return 0.0
    median_rms = float(np.median(rms_vals[active_mask]))
    gate = max(_MIN_RMS_FLOOR, median_rms * (10.0 ** (_SILENCE_GATE_DB / 20.0)))

    active_count = 0
    voiced_count = 0
    w = frame_len - max_lag
    if w <= 1:
        return 0.0
    n_fft = 1 << (frame_len + w - 1).bit_length()
    for i in range(n_frames):
        if rms_vals[i] < gate:
            continue
        active_count += 1
        seg = wav[i * hop_len : i * hop_len + frame_len]
        seg = seg - float(np.mean(seg))
        x0 = seg[:w]
        e0 = float(np.dot(x0, x0))
        if e0 <= 1e-12:
            continue
        # FFT cross-correlation of x0 against seg for lags [min_lag, max_lag].
        corr = np.fft.irfft(
            np.fft.rfft(seg, n=n_fft) * np.conj(np.fft.rfft(x0, n=n_fft)),
            n=n_fft,
        )[min_lag : max_lag + 1]
        csum = np.concatenate(([0.0], np.cumsum(seg * seg)))
        el = csum[min_lag + w : max_lag + 1 + w] - csum[min_lag : max_lag + 1]
        denom = np.sqrt(np.maximum(0.0, e0 * el))
        valid = denom > 1e-12
        best_nccf = float(np.max(corr[valid] / denom[valid])) if np.any(valid) else 0.0
        if best_nccf >= nccf_threshold:
            voiced_count += 1

    return voiced_count / active_count if active_count > 0 else 0.0


def consonant_band_ratio(
    samples: np.ndarray,
    sr: int,
    *,
    cutoff_hz: float = _CONSONANT_CUTOFF_HZ,
) -> float:
    """Fraction of spectral energy above ``cutoff_hz`` over active (non-silent) frames.

    Gated by an absolute + relative RMS floor so silent tails do not inflate the ratio.
    """
    wav = _mono_finite(samples)
    if wav.size == 0 or sr <= 0:
        return 0.0

    frame_len = max(64, int(round(_VOICING_FRAME_SECONDS * sr)))
    hop_len = max(32, int(round(_VOICING_HOP_SECONDS * sr)))
    if wav.size < frame_len:
        frame_len = wav.size
        hop_len = frame_len
    n_frames = 1 + (wav.size - frame_len) // hop_len
    if n_frames <= 0:
        return 0.0

    rms_vals = np.empty(n_frames, dtype=np.float64)
    for i in range(n_frames):
        seg = wav[i * hop_len : i * hop_len + frame_len]
        rms_vals[i] = float(np.sqrt(np.mean(seg * seg)))
    active_mask = rms_vals >= _MIN_RMS_FLOOR
    if not np.any(active_mask):
        return 0.0
    median_rms = float(np.median(rms_vals[active_mask]))
    gate = max(_MIN_RMS_FLOOR, median_rms * (10.0 ** (_SILENCE_GATE_DB / 20.0)))

    window = np.hanning(frame_len)
    freqs = np.fft.rfftfreq(frame_len, d=1.0 / sr)
    high_mask = freqs >= cutoff_hz

    total_energy = 0.0
    high_energy = 0.0
    for i in range(n_frames):
        if rms_vals[i] < gate:
            continue
        seg = wav[i * hop_len : i * hop_len + frame_len] * window
        power = np.abs(np.fft.rfft(seg)) ** 2
        total_energy += float(np.sum(power))
        high_energy += float(np.sum(power[high_mask]))
    if total_energy <= 1e-12:
        return 0.0
    return max(0.0, min(1.0, high_energy / total_energy))


def audit_lyric_alignment(
    vocal_samples: np.ndarray,
    sr: int,
    lyric_lines: Sequence[LyricLine],
    vocal_windows: Sequence[TimeSpan] = (),
    *,
    min_voicing_ratio: float = 0.15,
    max_consonant_ratio: float = 0.72,
) -> list[AlignmentWarning]:
    """Audit aligned/estimated lyric spans against independent vocal acoustics.

    Flags three pre-spend anomalies:
    - ``unvoiced_lyric``: lyric span shows negligible periodic voicing (likely hallucinated timing
      or placed onto instrumental bleed).
    - ``high_consonant_bleed``: lyric span is dominated by >3.4 kHz unvoiced sibilance/cymbal bleed.
    - ``unaligned_vocal_window``: substantial vocal activity window (>= 1.5s) has no lyric lines.
    """
    wav = _mono_finite(vocal_samples)
    if wav.size == 0 or sr <= 0:
        return []

    duration = wav.size / sr
    warnings: list[AlignmentWarning] = []

    for idx, line in enumerate(lyric_lines):
        if (
            line.start is None
            or line.end is None
            or not (math.isfinite(line.start) and math.isfinite(line.end))
            or line.end <= line.start
            or line.end <= 0.0
        ):
            continue
        line_start = max(0.0, line.start)
        line_end = max(line_start, line.end)
        s = max(0, min(wav.size, int(round(line_start * sr))))
        e = max(s, min(wav.size, int(round(line_end * sr))))
        if e - s < max(32, int(0.08 * sr)):
            continue
        seg = wav[s:e]
        vr = voicing_ratio(seg, sr)
        cr = consonant_band_ratio(seg, sr)
        if vr < min_voicing_ratio:
            warnings.append(
                AlignmentWarning(
                    kind="unvoiced_lyric",
                    start=round(line_start, 3),
                    end=round(line_end, 3),
                    detail=(
                        f"Line {idx + 1} ({line.text[:40]!r}) has low vocal periodicity "
                        f"(voicing={vr:.2f} < {min_voicing_ratio:.2f})."
                    ),
                )
            )
        elif cr > max_consonant_ratio:
            warnings.append(
                AlignmentWarning(
                    kind="high_consonant_bleed",
                    start=round(line_start, 3),
                    end=round(line_end, 3),
                    detail=(
                        f"Line {idx + 1} ({line.text[:40]!r}) is dominated by >3.4 kHz energy "
                        f"(ratio={cr:.2f} > {max_consonant_ratio:.2f}); possible cymbal/bleed."
                    ),
                )
            )

    timed_spans = [
        (max(0.0, ln.start), max(0.0, ln.end))
        for ln in lyric_lines
        if ln.start is not None
        and ln.end is not None
        and math.isfinite(ln.start)
        and math.isfinite(ln.end)
        and ln.end > ln.start
    ]
    for win in vocal_windows:
        if not (math.isfinite(win.start) and math.isfinite(win.end)):
            continue
        win_start = max(0.0, win.start)
        win_end = max(win_start, win.end)
        win_dur = win_end - win_start
        if win_dur < 1.5 or win_start >= duration:
            continue
        covered = sum(
            (
                max(0.0, min(win_end, ln_end) - max(win_start, ln_start))
                for ln_start, ln_end in timed_spans
            ),
            0.0,
        )
        if covered < 0.15 * win_dur:
            warnings.append(
                AlignmentWarning(
                    kind="unaligned_vocal_window",
                    start=round(win_start, 3),
                    end=round(win_end, 3),
                    detail=(
                        f"Vocal window {win_start:.2f}s-{win_end:.2f}s has active singing "
                        "without aligned lyrics (possible omitted verse or adlib)."
                    ),
                )
            )
    return warnings


def detect_boundary_word_cuts(
    cut_times: Sequence[float],
    word_spans: Sequence[tuple[str, float, float]],
    *,
    margin_seconds: float = 0.04,
) -> list[AlignmentWarning]:
    """Flag cut boundaries that slice strictly inside an aligned sung word.

    Inspired by ``JavaDerek/music-video-pipeline``'s ``join_witness.py``.
    """
    warnings: list[AlignmentWarning] = []
    for cut in cut_times:
        if not math.isfinite(cut) or cut <= 0.0:
            continue
        for word, w_start, w_end in word_spans:
            if not (math.isfinite(w_start) and math.isfinite(w_end)) or w_end <= 0.0:
                continue
            if w_end - w_start < 2.0 * margin_seconds:
                continue
            if (w_start + margin_seconds) < cut < (w_end - margin_seconds):
                warnings.append(
                    AlignmentWarning(
                        kind="boundary_word_cut",
                        start=round(max(0.0, w_start), 3),
                        end=round(max(0.0, w_end), 3),
                        detail=(
                            f"Cut at {cut:.3f}s splits word {word!r} ({w_start:.3f}s-{w_end:.3f}s)."
                        ),
                    )
                )
                break
    return warnings
