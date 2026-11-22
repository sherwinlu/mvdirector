# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Broadcast-grade pure-NumPy audio QA: ITU-R BS.1770-4 LUFS, 4x True Peak, and splice clicks.

Provides deterministic loudness, clipping, and splice-click diagnostics with zero extra binary
dependencies beyond NumPy and SoundFile.

Credits & Attribution:
    Techniques inspired by ``tysonfarrington-cmyk/xenoform``
    (https://github.com/tysonfarrington-cmyk/xenoform — ``src/xenoform/media_qa.py``).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf


@dataclass(frozen=True)
class BroadcastAudioReport:
    """Measured loudness, true-peak headroom, and splice-click anomalies for a mix."""

    integrated_lufs: float
    true_peak_dbtp: float
    splice_clicks: list[float] = field(default_factory=list)


def _biquad_stateful(
    signal: np.ndarray,
    b0: float,
    b1: float,
    b2: float,
    a1: float,
    a2: float,
    state: tuple[float, float] = (0.0, 0.0),
) -> tuple[np.ndarray, tuple[float, float]]:
    """Direct Form II Transposed IIR biquad filter carrying state across chunks."""
    out = np.empty_like(signal, dtype=np.float64)
    z1, z2 = state
    for i in range(signal.size):
        x = float(signal[i])
        y = b0 * x + z1
        z1 = b1 * x - a1 * y + z2
        z2 = b2 * x - a2 * y
        out[i] = y
    return out, (z1, z2)


def _biquad(
    signal: np.ndarray, b0: float, b1: float, b2: float, a1: float, a2: float
) -> np.ndarray:
    """Direct Form II Transposed IIR biquad filter over 1-D float64 signal."""
    out, _ = _biquad_stateful(signal, b0, b1, b2, a1, a2)
    return out


_BiquadCoeffs = tuple[float, float, float, float, float]


def _k_weight_coeffs(sr: int) -> tuple[_BiquadCoeffs, _BiquadCoeffs]:
    """Compute ITU-R BS.1770-4 K-weighting stage-1 high-shelf and stage-2 HPF biquads."""
    # Stage 1: pre-filter high-shelf (+3.9998 dB at 1681.97 Hz, Q=0.7072).
    db = 3.999843853973347
    f0 = 1681.974450955533
    q = 0.7071752369554196
    k = math.tan(math.pi * f0 / sr)
    vh = 10.0 ** (db / 20.0)
    vb = vh**0.4996667741545416
    a0 = 1.0 + k / q + k * k
    s1 = (
        (vh + vb * k / q + k * k) / a0,
        2.0 * (k * k - vh) / a0,
        (vh - vb * k / q + k * k) / a0,
        2.0 * (k * k - 1.0) / a0,
        (1.0 - k / q + k * k) / a0,
    )

    # Stage 2: RLB high-pass filter (f0=38.13547087602444 Hz, Q=0.5003270373238773)
    f0_hp = 38.13547087602444
    q_hp = 0.5003270373238773
    k_hp = math.tan(math.pi * f0_hp / sr)
    a0_hp = 1.0 + k_hp / q_hp + k_hp * k_hp
    s2 = (
        1.0 / a0_hp,
        -2.0 / a0_hp,
        1.0 / a0_hp,
        2.0 * (k_hp * k_hp - 1.0) / a0_hp,
        (1.0 - k_hp / q_hp + k_hp * k_hp) / a0_hp,
    )
    return s1, s2


def _k_weight(samples: np.ndarray, sr: int) -> np.ndarray:
    """Apply BS.1770-4 K-weighting to ``(N, C)`` float64 audio."""
    if sr <= 3400:
        # K-weighting stage-1 shelf (1682 Hz) requires Nyquist > 1682 Hz; skip at ultra-low SR.
        return np.asarray(samples, dtype=np.float64)
    try:
        from scipy.signal import lfilter

        s1, s2 = _k_weight_coeffs(sr)
        out = np.empty_like(samples, dtype=np.float64)
        for ch in range(samples.shape[1]):
            y = lfilter([s1[0], s1[1], s1[2]], [1.0, s1[3], s1[4]], samples[:, ch])
            out[:, ch] = lfilter([s2[0], s2[1], s2[2]], [1.0, s2[3], s2[4]], y)
        return out
    except ImportError:
        s1, s2 = _k_weight_coeffs(sr)
        out = np.empty_like(samples, dtype=np.float64)
        for ch in range(samples.shape[1]):
            y = _biquad(samples[:, ch], *s1)
            out[:, ch] = _biquad(y, *s2)
        return out


def _channel_weights(n_channels: int) -> np.ndarray:
    """ITU-R BS.1770-4 channel weights (L/R/C=1.0, LFE=0.0, Ls/Rs=1.41 for 5.1)."""
    if n_channels >= 6:
        weights = [1.0, 1.0, 1.0, 0.0, 1.41, 1.41] + [1.0] * (n_channels - 6)
    elif n_channels == 5:
        weights = [1.0, 1.0, 1.0, 1.41, 1.41]
    else:
        weights = [1.0] * n_channels
    return np.array(weights, dtype=np.float64)


def integrated_lufs(samples: np.ndarray, sr: int) -> float:
    """Compute ITU-R BS.1770-4 gated integrated loudness in LUFS.

    Returns ``-70.0`` for silent or ultra-short (<400 ms) signals.
    """
    arr = np.asarray(samples, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.size == 0 or sr <= 0:
        return -70.0

    ch_weights = _channel_weights(arr.shape[1])
    block_len = max(1, int(round(0.400 * sr)))
    hop_len = max(1, int(round(0.100 * sr)))
    if arr.shape[0] < block_len:
        power = float(np.sum(np.mean(arr * arr, axis=0) * ch_weights))
        return -70.0 if power <= 1e-12 else max(-70.0, -0.691 + 10.0 * math.log10(power))

    weighted = _k_weight(arr, sr)
    n_blocks = 1 + (weighted.shape[0] - block_len) // hop_len
    block_powers = np.empty(n_blocks, dtype=np.float64)
    for idx in range(n_blocks):
        seg = weighted[idx * hop_len : idx * hop_len + block_len]
        block_powers[idx] = float(np.sum(np.mean(seg * seg, axis=0) * ch_weights))

    # Absolute gate at -70 LUFS + relative gate at -10 LU.
    return _gated_lufs_from_block_powers(block_powers)


def true_peak_dbtp(samples: np.ndarray, *, oversample: int = 4) -> float:
    """Compute 4x-oversampled True Peak in dBTP using windowed sinc interpolation."""
    arr = np.asarray(samples, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.size == 0:
        return -120.0

    kernels = _true_peak_kernels(oversample)
    peak = _max_oversampled_peak(arr, kernels)
    return -120.0 if peak <= 1e-12 else float(20.0 * math.log10(peak))


def scan_splice_clicks(
    samples: np.ndarray,
    sr: int,
    *,
    boundary_times: Sequence[float] | None = None,
    drum_onset_times: Sequence[float] = (),
    spike_ratio: float = 8.0,
    min_hpf_peak: float = 0.02,
    onset_exclude_seconds: float = 0.035,
) -> list[float]:
    """Detect sharp audio splice clicks while excluding legitimate musical drum attacks.

    Applies a simple high-frequency emphasis (first-order/second-order difference emphasizing >2 kHz
    discontinuities), compares impulse spikes against a 10 ms local median floor, verifies the
    spike lacks a sustained 10 ms musical tail, and ignores spikes near known drum onsets.
    """
    wav = np.asarray(samples, dtype=np.float64)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if wav.size < 64 or sr <= 0:
        return []

    # High-frequency discontinuity detector (second difference strongly emphasizes sample steps).
    hpf = np.abs(np.diff(wav, n=2, prepend=wav[:2]))
    win_ms = max(8, int(round(0.010 * sr)))  # 10 ms neighborhood
    clicks: list[float] = []

    if boundary_times is not None:
        search_windows = [
            (
                max(win_ms, int(round((t - 0.025) * sr))),
                min(wav.size - win_ms, int(round((t + 0.025) * sr))),
            )
            for t in boundary_times
            if t > 0.0
        ]
    else:
        step = max(win_ms, int(round(0.020 * sr)))
        search_windows = [
            (i, min(wav.size - win_ms, i + step)) for i in range(win_ms, wav.size - win_ms, step)
        ]

    for lo, hi in search_windows:
        if hi <= lo:
            continue
        local = hpf[lo:hi]
        rel_idx = int(np.argmax(local))
        peak = float(local[rel_idx])
        if peak < min_hpf_peak:
            continue
        idx = lo + rel_idx
        t_sec = idx / sr
        if any(abs(t_sec - onset) <= onset_exclude_seconds for onset in drum_onset_times):
            continue
        pre = hpf[max(0, idx - win_ms) : max(1, idx - 2)]
        post = hpf[min(wav.size - 1, idx + 3) : min(wav.size, idx + win_ms)]
        if pre.size == 0 or post.size == 0:
            continue
        floor = max(1e-5, float(np.median(np.concatenate([pre, post]))))
        tail_rms = float(np.sqrt(np.mean(post * post)))
        # True splice clicks are impulse-like (high peak-to-floor and low sustained tail).
        if (
            peak / floor >= spike_ratio
            and tail_rms < 0.35 * peak
            and (not clicks or t_sec - clicks[-1] > 0.05)
        ):
            clicks.append(round(t_sec, 4))
    return clicks


def _gated_lufs_from_block_powers(block_powers: np.ndarray) -> float:
    """Apply ITU-R BS.1770-4 absolute (-70 LUFS) and relative (-10 LU) gates."""
    if block_powers.size == 0:
        return -70.0
    abs_thresh = 10.0 ** ((-70.0 + 0.691) / 10.0)
    above_abs = block_powers[block_powers > abs_thresh]
    if above_abs.size == 0:
        return -70.0
    ungated_mean = float(np.mean(above_abs))
    rel_thresh = ungated_mean * (10.0 ** (-10.0 / 10.0))
    gated = above_abs[above_abs > rel_thresh]
    if gated.size == 0:
        return -70.0
    return float(-0.691 + 10.0 * math.log10(float(np.mean(gated))))


def _true_peak_kernels(oversample: int) -> list[np.ndarray]:
    factor = max(1, oversample)
    if factor == 1:
        return []
    half_taps = 6
    taps = np.arange(-half_taps, half_taps, dtype=np.float64)
    win = 0.5 * (1.0 + np.cos(math.pi * taps / half_taps))
    kernels: list[np.ndarray] = []
    for phase in range(1, factor):
        offset = phase / factor
        k = np.sinc(taps - offset) * win
        k /= np.sum(k)
        kernels.append(k)
    return kernels


def _max_oversampled_peak(arr: np.ndarray, kernels: Sequence[np.ndarray]) -> float:
    peak = float(np.max(np.abs(arr))) if arr.size else 0.0
    if not kernels or arr.shape[0] < 8:
        return peak
    for ch in range(arr.shape[1]):
        col = arr[:, ch]
        for kernel in kernels:
            interp = np.convolve(col, kernel, mode="same")
            peak = max(peak, float(np.max(np.abs(interp))))
    return peak


def analyze_broadcast_audio(
    audio_path: Path,
    *,
    boundary_times: Sequence[float] | None = None,
    drum_onset_times: Sequence[float] = (),
    chunk_seconds: float = 30.0,
) -> BroadcastAudioReport:
    """Measure LUFS, 4x True Peak, and splice clicks on an audio file in bounded-memory chunks.

    Streaming keeps RAM usage ~25 MB even on multi-hour 48 kHz WAV/FLAC mixes while preserving
    filter state and overlap across chunk boundaries for exact BS.1770-4 results.
    """
    try:
        return _analyze_broadcast_soundfile(
            audio_path,
            boundary_times=boundary_times,
            drum_onset_times=drum_onset_times,
            chunk_seconds=chunk_seconds,
        )
    except sf.LibsndfileError:
        import tempfile

        from mvdirector import media

        with tempfile.TemporaryDirectory(prefix="mvd_bqa_") as tmp:
            wav_path = media.transcode_to_wav(audio_path, Path(tmp) / "audio.wav")
            return _analyze_broadcast_soundfile(
                wav_path,
                boundary_times=boundary_times,
                drum_onset_times=drum_onset_times,
                chunk_seconds=chunk_seconds,
            )


def _analyze_broadcast_soundfile(
    audio_path: Path,
    *,
    boundary_times: Sequence[float] | None = None,
    drum_onset_times: Sequence[float] = (),
    chunk_seconds: float = 30.0,
) -> BroadcastAudioReport:
    with sf.SoundFile(str(audio_path)) as handle:
        sr = int(handle.samplerate)
        n_channels = int(handle.channels)
        if sr <= 0 or n_channels <= 0:
            return BroadcastAudioReport(integrated_lufs=-70.0, true_peak_dbtp=-120.0)

        ch_weights = _channel_weights(n_channels)
        block_len = max(1, int(round(0.400 * sr)))
        hop_len = max(1, int(round(0.100 * sr)))
        chunk_frames = max(block_len * 4, int(round(max(1.0, chunk_seconds) * sr)))
        kernels = _true_peak_kernels(4)
        tp_overlap = 6
        win_ms = max(8, int(round(0.010 * sr)))
        click_overlap = max(win_ms + 4, int(round(0.050 * sr)))

        try:
            from scipy.signal import lfilter as scipy_lfilter
        except ImportError:
            scipy_lfilter = None

        use_k = sr > 3400
        s1, s2 = _k_weight_coeffs(sr) if use_k else ((1.0, 0.0, 0.0, 0.0, 0.0),) * 2
        if scipy_lfilter is not None:
            zi1: list[np.ndarray] = [np.zeros(2, dtype=np.float64) for _ in range(n_channels)]
            zi2: list[np.ndarray] = [np.zeros(2, dtype=np.float64) for _ in range(n_channels)]
        else:
            st1: list[tuple[float, float]] = [(0.0, 0.0) for _ in range(n_channels)]
            st2: list[tuple[float, float]] = [(0.0, 0.0) for _ in range(n_channels)]

        k_carry = np.empty((0, n_channels), dtype=np.float64)
        tp_carry = np.empty((0, n_channels), dtype=np.float64)
        click_carry = np.empty(0, dtype=np.float64)
        block_powers: list[float] = []
        short_sq_sum = np.zeros(n_channels, dtype=np.float64)
        total_frames = 0
        max_peak = 0.0
        clicks: list[float] = []

        while True:
            chunk = handle.read(chunk_frames, dtype="float64", always_2d=True)
            if chunk.size == 0:
                break
            chunk_start_sample = total_frames
            total_frames += chunk.shape[0]
            short_sq_sum += np.sum(chunk * chunk, axis=0)

            # 1. Exact streaming 4x True Peak with 6-sample overlap.
            tp_buf = np.concatenate([tp_carry, chunk], axis=0) if tp_carry.size else chunk
            max_peak = max(max_peak, _max_oversampled_peak(tp_buf, kernels))
            tp_carry = chunk[-tp_overlap:] if chunk.shape[0] >= tp_overlap else tp_buf[-tp_overlap:]

            # 2. Exact streaming BS.1770-4 K-weighting + 400 ms / 100 ms blocks.
            if use_k:
                weighted = np.empty_like(chunk, dtype=np.float64)
                for ch in range(n_channels):
                    if scipy_lfilter is not None:
                        y, zi1[ch] = scipy_lfilter(
                            [s1[0], s1[1], s1[2]], [1.0, s1[3], s1[4]], chunk[:, ch], zi=zi1[ch]
                        )
                        weighted[:, ch], zi2[ch] = scipy_lfilter(
                            [s2[0], s2[1], s2[2]], [1.0, s2[3], s2[4]], y, zi=zi2[ch]
                        )
                    else:
                        y, st1[ch] = _biquad_stateful(chunk[:, ch], *s1, state=st1[ch])
                        weighted[:, ch], st2[ch] = _biquad_stateful(y, *s2, state=st2[ch])
            else:
                weighted = chunk

            k_buf = np.concatenate([k_carry, weighted], axis=0) if k_carry.size else weighted
            if k_buf.shape[0] >= block_len:
                n_blk = 1 + (k_buf.shape[0] - block_len) // hop_len
                for idx in range(n_blk):
                    seg = k_buf[idx * hop_len : idx * hop_len + block_len]
                    block_powers.append(float(np.sum(np.mean(seg * seg, axis=0) * ch_weights)))
                k_carry = k_buf[n_blk * hop_len :]
            else:
                k_carry = k_buf

            # 3. Streaming splice-click scan with 50 ms overlap.
            mono = chunk.mean(axis=1)
            overlap_len = click_carry.size
            c_buf = np.concatenate([click_carry, mono]) if overlap_len else mono
            buf_start_sec = (chunk_start_sample - overlap_len) / sr
            buf_end_sec = (chunk_start_sample + chunk.shape[0]) / sr
            if boundary_times is not None:
                local_bounds: list[float] | None = [
                    t - buf_start_sec for t in boundary_times if buf_start_sec < t <= buf_end_sec
                ]
            else:
                local_bounds = None
            local_onsets = [
                t - buf_start_sec
                for t in drum_onset_times
                if buf_start_sec - 0.05 <= t <= buf_end_sec + 0.05
            ]
            if local_bounds is None or local_bounds:
                local_clicks = scan_splice_clicks(
                    c_buf,
                    sr,
                    boundary_times=local_bounds,
                    drum_onset_times=local_onsets,
                )
                for rel_t in local_clicks:
                    abs_t = round(buf_start_sec + rel_t, 4)
                    if abs_t > 0.0 and (not clicks or abs_t - clicks[-1] > 0.05):
                        clicks.append(abs_t)
            click_carry = (
                mono[-click_overlap:] if mono.size >= click_overlap else c_buf[-click_overlap:]
            )

    if total_frames == 0:
        lufs = -70.0
    elif total_frames < block_len:
        power = float(np.sum((short_sq_sum / total_frames) * ch_weights))
        lufs = -70.0 if power <= 1e-12 else max(-70.0, -0.691 + 10.0 * math.log10(power))
    else:
        lufs = _gated_lufs_from_block_powers(np.array(block_powers, dtype=np.float64))

    tp_db = -120.0 if max_peak <= 1e-12 else float(20.0 * math.log10(max_peak))
    return BroadcastAudioReport(
        integrated_lufs=round(lufs, 2),
        true_peak_dbtp=round(tp_db, 2),
        splice_clicks=clicks,
    )
