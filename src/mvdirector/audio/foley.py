# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Diegetic SFX / foley leveling, onset hit re-timing, and per-scene song ducking.

Supports blending non-musical sound effects or generated diegetic audio (for example from LTX-2.5
audio-video clips or standalone foley renders) cleanly over the master song mix without clipping or
clashing with vocals.

Credits & Attribution:
    - Active-RMS loudness leveling and onset hit re-timing inspired by
      ``thefullnacho/on-the-one`` (https://github.com/thefullnacho/on-the-one —
      ``on_the_one/foley.py``, ``on_the_one/foleymix.py``).
    - Single-pass FFmpeg linear per-scene song ducking inspired by
      ``tsondo/musicvision`` (https://github.com/tsondo/musicvision —
      ``src/musicvision/utils/audio.py:build_mixed_audio``).
"""

from __future__ import annotations

import math
import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from mvdirector import media


@dataclass(frozen=True)
class DiegeticCue:
    """A diegetic audio clip placed onto the song timeline with optional song ducking."""

    audio_path: Path
    start: float
    end: float
    gain_db: float = 0.0
    duck_db: float = -6.0
    fade_seconds: float = 0.15


def level_active_rms(
    samples: np.ndarray,
    sr: int,
    *,
    target_db: float = -26.0,
    peak_ceiling_db: float = -6.0,
    active_window_db: float = -30.0,
) -> np.ndarray:
    """Normalize audio by its active (non-silent) 10 ms RMS while respecting a peak ceiling.

    Sharp clicks/impulses and soft ambience beds reach consistent perceived loudness without
    over-boosting quiet noise floors.
    """
    arr = np.asarray(samples, dtype=np.float64)
    if arr.size == 0 or sr <= 0:
        return arr.copy()

    mono = arr.mean(axis=1) if arr.ndim > 1 else arr
    peak = float(np.max(np.abs(arr)))
    if not math.isfinite(peak) or peak <= 1e-9:
        return arr.copy()

    frame_len = max(1, int(round(0.010 * sr)))
    n_frames = max(1, mono.size // frame_len)
    trimmed = mono[: n_frames * frame_len].reshape(n_frames, frame_len)
    frame_rms = np.sqrt(np.mean(trimmed * trimmed, axis=1))
    peak_rms = float(np.max(frame_rms))
    if peak_rms <= 1e-9:
        return arr.copy()

    threshold = peak_rms * (10.0 ** (active_window_db / 20.0))
    active = frame_rms[frame_rms >= threshold]
    if active.size == 0:
        return arr.copy()

    active_rms = float(np.sqrt(np.mean(active * active)))
    target_rms = 10.0 ** (target_db / 20.0)
    ceiling = 10.0 ** (peak_ceiling_db / 20.0)
    gain = min(target_rms / max(active_rms, 1e-9), ceiling / peak)
    return arr * gain


def detect_foley_onsets(
    samples: np.ndarray,
    sr: int,
    *,
    min_gap_seconds: float = 0.12,
    threshold_ratio: float = 0.25,
) -> list[float]:
    """Detect envelope-attack onset times (seconds) in a foley clip."""
    arr = np.asarray(samples, dtype=np.float64)
    mono = arr.mean(axis=1) if arr.ndim > 1 else arr
    if mono.size == 0 or sr <= 0:
        return []

    hop = max(1, int(round(0.005 * sr)))
    n_hops = mono.size // hop
    if n_hops < 3:
        return []
    env = np.sqrt(np.mean(mono[: n_hops * hop].reshape(n_hops, hop) ** 2, axis=1))
    flux = np.maximum(0.0, np.diff(env, prepend=env[0]))
    peak_flux = float(np.max(flux))
    if peak_flux <= 1e-9:
        return []

    thresh = peak_flux * threshold_ratio
    min_gap_hops = max(1, int(round(min_gap_seconds * sr / hop)))
    onsets: list[float] = []
    last_hop = -min_gap_hops
    for idx in range(1, n_hops - 1):
        if (
            flux[idx] >= thresh
            and flux[idx] >= flux[idx - 1]
            and flux[idx] >= flux[idx + 1]
            and idx - last_hop >= min_gap_hops
        ):
            onsets.append(round(idx * hop / sr, 4))
            last_hop = idx
    return onsets


def retime_foley_hits(
    samples: np.ndarray,
    sr: int,
    target_times: Sequence[float],
    total_duration: float,
    *,
    pre_attack_seconds: float = 0.020,
    fade_seconds: float = 0.005,
) -> np.ndarray:
    """Slice detected attacks from ``samples`` and place them onto exact ``target_times``.

    Avoids re-rendering SFX when moving hits onto beat or visual-action timestamps.
    """
    arr = np.asarray(samples, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    if not math.isfinite(total_duration) or total_duration <= 0.0 or sr <= 0:
        return np.zeros((1, arr.shape[1]), dtype=np.float64)
    n_out = max(1, int(round(total_duration * sr)))
    out = np.zeros((n_out, arr.shape[1]), dtype=np.float64)
    if arr.shape[0] == 0 or not target_times:
        return out

    onsets = detect_foley_onsets(arr, sr)
    if not onsets:
        onsets = [0.0]

    pre_samples = max(0, int(round(pre_attack_seconds * sr)))
    fade_samples = max(1, int(round(fade_seconds * sr)))
    hit_slices: list[np.ndarray] = []
    for idx, onset_s in enumerate(onsets):
        s_idx = max(0, int(round(onset_s * sr)) - pre_samples)
        next_s = onsets[idx + 1] if idx + 1 < len(onsets) else arr.shape[0] / sr
        e_idx = min(arr.shape[0], max(s_idx + 1, int(round(next_s * sr)) - pre_samples))
        chunk = arr[s_idx:e_idx].copy()
        f_len = min(fade_samples, chunk.shape[0] // 2)
        if f_len > 0:
            ramp = np.linspace(0.0, 1.0, f_len)[:, None]
            chunk[:f_len] *= ramp
            chunk[-f_len:] *= ramp[::-1]
        hit_slices.append(chunk)

    for idx, target_s in enumerate(target_times):
        if not math.isfinite(target_s) or target_s < 0.0:
            continue
        chunk = hit_slices[idx % len(hit_slices)]
        dst_s = max(0, int(round(target_s * sr)) - pre_samples)
        if dst_s >= n_out:
            continue
        copy_len = min(chunk.shape[0], n_out - dst_s)
        out[dst_s : dst_s + copy_len] += chunk[:copy_len]
    return np.clip(out, -1.0, 1.0)


def ducking_volume_expression(cues: Sequence[DiegeticCue]) -> str:
    """Build a single-pass FFmpeg per-frame linear ducking ``volume`` expression."""
    factors: list[str] = []
    for cue in cues:
        if (
            not (
                math.isfinite(cue.start)
                and math.isfinite(cue.end)
                and math.isfinite(cue.duck_db)
                and math.isfinite(cue.fade_seconds)
            )
            or cue.end <= cue.start
            or cue.start < 0.0
            or cue.duck_db >= 0.0
        ):
            continue
        safe_duck = max(-40.0, min(-0.1, cue.duck_db))
        duck_gain = 10.0 ** (safe_duck / 20.0)
        fade = max(0.01, min(cue.fade_seconds, (cue.end - cue.start) * 0.45))
        s = cue.start
        e = cue.end
        s_in = s + fade
        e_out = e - fade
        expr = (
            f"if(between(t,{s:.4f},{s_in:.4f}),"
            f"1-(1-{duck_gain:.4f})*(t-{s:.4f})/{fade:.4f},"
            f"if(between(t,{s_in:.4f},{e_out:.4f}),{duck_gain:.4f},"
            f"if(between(t,{e_out:.4f},{e:.4f}),"
            f"{duck_gain:.4f}+(1-{duck_gain:.4f})*(t-{e_out:.4f})/{fade:.4f},1)))"
        )
        factors.append(f"({expr})")
    return "*".join(factors) if factors else "1"


def build_ducked_audio_mix(
    music_path: Path,
    cues: Sequence[DiegeticCue],
    output_path: Path,
    *,
    limiter_limit: float = 0.98,
) -> Path:
    """Blend diegetic SFX cues over ``music_path`` with smooth ducking and true-peak limiting."""
    valid_cues = [
        c
        for c in cues
        if c.audio_path.is_file()
        and math.isfinite(c.start)
        and math.isfinite(c.end)
        and math.isfinite(c.gain_db)
        and math.isfinite(c.duck_db)
        and math.isfinite(c.fade_seconds)
        and c.start >= 0.0
        and c.end > c.start
    ]
    if not valid_cues:
        return media.transcode_to_wav(music_path, output_path)
    safe_limit = max(0.1, min(1.0, float(limiter_limit))) if math.isfinite(limiter_limit) else 0.98

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(music_path.resolve())]
    for cue in valid_cues:
        cmd.extend(["-i", str(cue.audio_path.resolve())])

    vol_expr = ducking_volume_expression(valid_cues)
    filters = [f"[0:a]volume='{vol_expr}':eval=frame[music]"]
    mix_labels = ["[music]"]
    for idx, cue in enumerate(valid_cues, start=1):
        delay_ms = max(0, int(round(cue.start * 1000.0)))
        dur = max(0.05, cue.end - cue.start)
        fade = max(0.01, min(cue.fade_seconds, dur * 0.45))
        safe_gain_db = max(-40.0, min(24.0, cue.gain_db))
        gain = 10.0 ** (safe_gain_db / 20.0)
        fade_out_start = max(0.0, dur - fade)
        filters.append(
            f"[{idx}:a]atrim=0:{dur:.4f},asetpts=PTS-STARTPTS,"
            f"afade=t=in:st=0:d={fade:.4f},afade=t=out:st={fade_out_start:.4f}:d={fade:.4f},"
            f"volume={gain:.4f},adelay={delay_ms}:all=1[cue{idx}]"
        )
        mix_labels.append(f"[cue{idx}]")

    filters.append(
        f"{''.join(mix_labels)}amix=inputs={len(mix_labels)}:duration=first:normalize=0,"
        f"alimiter=limit={safe_limit:.3f}[out]"
    )

    fd, tmp_name = tempfile.mkstemp(prefix=".mvd_foley_", suffix=".wav", dir=output_path.parent)
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        cmd.extend(
            [
                "-filter_complex",
                ";".join(filters),
                "-map",
                "[out]",
                "-c:a",
                "pcm_s16le",
                str(tmp_path.resolve()),
            ]
        )
        media.run(cmd)
        os.replace(tmp_path, output_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    return output_path


def _read_audio(src: Path) -> tuple[np.ndarray, int]:
    """Decode ``src`` to float64 ``(N, C)`` samples, via ffmpeg for formats libsndfile lacks."""
    try:
        samples, sr = sf.read(str(src), dtype="float64", always_2d=True)
        return samples, int(sr)
    except sf.LibsndfileError:
        with tempfile.TemporaryDirectory(prefix="mvd_sfx_") as tmp:
            wav_path = media.transcode_to_wav(src, Path(tmp) / "sfx.wav")
            samples, sr = sf.read(str(wav_path), dtype="float64", always_2d=True)
            return samples, int(sr)


def write_leveled_foley(src: Path, dst: Path, *, target_db: float = -26.0) -> Path:
    """Level ``src`` by active RMS and write it atomically to ``dst`` as 24-bit PCM WAV.

    Accepts any format ffmpeg can decode (MP3, Opus, AAC/M4A, ...), not only WAV/FLAC: the
    output is always PCM because compressed source subtypes are not valid inside a WAV file.
    """
    samples, sr = _read_audio(src)
    leveled = level_active_rms(samples, sr, target_db=target_db)
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".mvd_lvl_", suffix=".wav", dir=dst.parent)
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        sf.write(str(tmp_path), leveled, sr, subtype="PCM_24")
        os.replace(tmp_path, dst)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    return dst
