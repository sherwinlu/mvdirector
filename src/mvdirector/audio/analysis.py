# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Audio analysis: tempo, beat and bar grid, per-bar intensity and vocal activity.

The signal-processing helpers are pure functions over numpy arrays so they can be
unit-tested without audio files. ``analyze_project`` is the only part that does I/O.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Literal

import numpy as np

from mvdirector import tuning as tuning_mod
from mvdirector.audio import alignment_qa as alignment_qa_mod
from mvdirector.audio import broadcast_qa as broadcast_qa_mod
from mvdirector.audio import lyrics as lyrics_mod
from mvdirector.audio import stem_pack as stem_pack_mod
from mvdirector.models import (
    BOTH_VOICES,
    Analysis,
    BroadcastAudioQA,
    MusicalFeatures,
    TimeSpan,
    VideoStyle,
    VocalTurn,
)
from mvdirector.project import Project

log = logging.getLogger(__name__)

ANALYSIS_SR = 22050
HOP_LENGTH = 512
# Tempos outside this range are folded by whole octaves: beat trackers often lock onto half or
# double time (a slow blues reads as 185 BPM), and cuts follow the beat grid.
TEMPO_RANGE_BPM = (70.0, 160.0)

_PITCH_CLASSES = ("C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")
_KRUMHANSL_MAJOR = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88], dtype=np.float64
)
_KRUMHANSL_MINOR = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17], dtype=np.float64
)
_MIN_KEY_CORRELATION = 0.25


def load_mono(path: Path, sr: int = ANALYSIS_SR) -> np.ndarray:
    import librosa  # imported lazily: slow to import and unused by most commands

    samples, _ = librosa.load(str(path), sr=sr, mono=True)
    return samples


def fold_tempo(
    tempo: float,
    beats: list[float],
    low: float = TEMPO_RANGE_BPM[0],
    high: float = TEMPO_RANGE_BPM[1],
) -> tuple[float, list[float]]:
    """Halve or double ``tempo`` until it is within [low, high], adjusting ``beats`` to match.

    Halving keeps every other beat (starting with the first); doubling adds a beat midway
    between each pair.
    """
    if not math.isfinite(tempo) or tempo <= 0 or low <= 0 or high < low:
        return tempo, beats
    while tempo > high and len(beats) > 1:
        tempo, beats = tempo / 2, beats[::2]
    # Cap octave doublings so a near-zero degenerate tempo cannot blow up memory.
    for _ in range(4):
        if tempo >= low or len(beats) <= 1:
            break
        doubled = [t for a, b in zip(beats, beats[1:]) for t in (a, (a + b) / 2)]
        tempo, beats = tempo * 2, [*doubled, beats[-1]]
    return tempo, beats


def detect_beats(samples: np.ndarray, sr: int = ANALYSIS_SR) -> tuple[float, list[float]]:
    """Return (tempo in BPM, beat times in seconds), folded into ``TEMPO_RANGE_BPM``."""
    import librosa

    tempo, frames = librosa.beat.beat_track(y=samples, sr=sr, hop_length=HOP_LENGTH)
    times = librosa.frames_to_time(frames, sr=sr, hop_length=HOP_LENGTH)
    return fold_tempo(float(np.atleast_1d(tempo)[0]), [float(t) for t in times])


def bars_from_beats(beats: list[float], beats_per_bar: int) -> list[float]:
    """Bar start times, assuming the first detected beat is a downbeat.

    librosa has no downbeat detector, so the bar phase can be off by a beat or two.
    Cuts still land on beats; edit ``bars`` in analysis.json to fix the phase.
    """
    return beats[::beats_per_bar]


def rms_envelope(samples: np.ndarray, sr: int = ANALYSIS_SR) -> tuple[np.ndarray, np.ndarray]:
    """Return (rms per hop, time of each hop in seconds)."""
    count = len(samples) // HOP_LENGTH
    if count == 0:
        return np.zeros(1), np.zeros(1)
    trimmed = samples[: count * HOP_LENGTH].reshape(count, HOP_LENGTH)
    rms = np.sqrt(np.mean(trimmed.astype(np.float64) ** 2, axis=1))
    times = (np.arange(count) + 0.5) * HOP_LENGTH / sr
    return rms, times


def detect_vocal_windows(
    rms: np.ndarray,
    times: np.ndarray,
    threshold_db: float = -30.0,
    min_duration: float = 0.3,
    merge_gap: float = 0.6,
    min_peak_rms: float = 1e-3,
) -> list[TimeSpan]:
    """Spans where the vocal stem is louder than ``threshold_db`` relative to its peak.

    Gaps shorter than ``merge_gap`` (breaths) are bridged; blips shorter than
    ``min_duration`` (stem bleed) are dropped. Stems whose peak RMS is below
    ``min_peak_rms`` (-60 dBFS by default) are treated as silent so near-zero stem bleed
    does not register as singing.
    """
    peak = float(rms.max()) if rms.size else 0.0
    if not math.isfinite(peak) or peak < min_peak_rms or peak <= 0.0:
        return []
    safe_rms = np.where(np.isfinite(rms), rms, 0.0)
    active = 20.0 * np.log10(np.maximum(safe_rms, 1e-10) / peak) > threshold_db
    hop = float(times[1] - times[0]) if len(times) > 1 else 0.0

    spans: list[list[float]] = []
    start: float | None = None
    for is_active, time in zip(active, times):
        if is_active and start is None:
            start = float(time) - hop / 2
        elif not is_active and start is not None:
            spans.append([start, float(time) - hop / 2])
            start = None
    if start is not None:
        spans.append([start, float(times[-1]) + hop / 2])

    merged: list[list[float]] = []
    for span in spans:
        if merged and span[0] - merged[-1][1] <= merge_gap:
            merged[-1][1] = span[1]
        else:
            merged.append(span)
    return [
        TimeSpan(start=round(max(0.0, s), 3), end=round(e, 3))
        for s, e in merged
        if e - s >= min_duration
    ]


def _interp_matrix(target_freqs: np.ndarray, bin_freqs: np.ndarray) -> np.ndarray:
    """Linear-interpolation matrix mapping ``bin_freqs`` columns onto ``target_freqs`` rows."""
    mat = np.zeros((len(target_freqs), len(bin_freqs)), dtype=np.float64)
    valid = (target_freqs >= bin_freqs[0]) & (target_freqs <= bin_freqs[-1])
    rows = np.flatnonzero(valid)
    tf = target_freqs[rows]
    hi = np.clip(np.searchsorted(bin_freqs, tf, side="right"), 1, len(bin_freqs) - 1)
    lo = hi - 1
    denom = np.maximum(bin_freqs[hi] - bin_freqs[lo], 1e-12)
    alpha = (tf - bin_freqs[lo]) / denom
    mat[rows, lo] = 1.0 - alpha
    mat[rows, hi] = alpha
    return mat


def _comb_pitch_and_polyphony(
    stft_mag: np.ndarray, freqs: np.ndarray, voiced: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(primary_f0_hz, second_voice_ratio)`` per frame using a gated harmonic comb."""
    n_frames = stft_mag.shape[1]
    cand_f = np.geomspace(80.0, 540.0, 96)
    m_f0 = _interp_matrix(cand_f, freqs).astype(np.float32)
    m_harm = sum(
        ((h**-0.6) * _interp_matrix(cand_f * h, freqs) for h in range(1, 5)),
        np.zeros_like(m_f0, dtype=np.float64),
    ).astype(np.float32)
    mag32 = np.asarray(stft_mag, dtype=np.float32)
    freqs32 = np.asarray(freqs, dtype=np.float32)

    sal = np.sqrt(np.maximum(m_f0 @ mag32, 0.0) * np.maximum(m_harm @ mag32, 0.0))
    best_idx = np.argmax(sal, axis=0)
    f0 = cand_f[best_idx]
    m1 = sal[best_idx, np.arange(n_frames)].astype(np.float64)

    r2 = np.zeros(n_frames, dtype=np.float64)
    v_idx = np.flatnonzero(voiced & (m1 > 1e-6))
    # Process in frame chunks so long songs stay bounded in peak RAM.
    chunk_size = 2048
    for offset in range(0, v_idx.size, chunk_size):
        chunk = v_idx[offset : offset + chunk_size]
        res = mag32[:, chunk].copy()
        f_primary = f0[chunk].astype(np.float32)[None, :]
        for h in range(1, 9):
            center = h * f_primary
            bw = np.maximum(center * np.float32(0.09), np.float32(27.0))
            res[np.abs(freqs32[:, None] - center) <= bw] = 0.0
        sal2 = np.sqrt(np.maximum(m_f0 @ res, 0.0) * np.maximum(m_harm @ res, 0.0))
        r2[chunk] = np.max(sal2, axis=0) / m1[chunk]
    return f0, r2


def _split_phrase_voice_changes(
    s_idx: int,
    e_idx: int,
    rms: np.ndarray,
    voiced: np.ndarray,
    log_f0: np.ndarray,
    r2: np.ndarray,
    mfcc: np.ndarray,
    mel_norm: np.ndarray,
    sr: int,
) -> list[tuple[int, int]]:
    """Split a vocal phrase at solo<->solo or solo<->together pitch/timbre/polyphony transitions."""
    win = max(8, int(round(0.45 * sr / HOP_LENGTH)))
    if e_idx - s_idx < 2 * win + 2:
        return [(s_idx, e_idx)]

    scores = np.zeros(e_idx - s_idx, dtype=np.float64)
    pitches = np.zeros(e_idx - s_idx, dtype=np.float64)
    polys = np.zeros(e_idx - s_idx, dtype=np.float64)
    stride = 2 if (e_idx - s_idx) > 4 * win else 1
    for k in range(s_idx + win, e_idx - win + 1, stride):
        lv = voiced[k - win : k]
        rv = voiced[k : k + win]
        if int(lv.sum()) < 5 or int(rv.sum()) < 5:
            continue
        pitch_jump = abs(
            float(np.median(log_f0[k - win : k][lv])) - float(np.median(log_f0[k : k + win][rv]))
        )
        poly_jump = abs(
            float(np.median(r2[k - win : k][lv])) - float(np.median(r2[k : k + win][rv]))
        )
        mfcc_jump = (
            float(
                np.linalg.norm(
                    np.median(mfcc[:, k - win : k][:, lv], axis=1)
                    - np.median(mfcc[:, k : k + win][:, rv], axis=1)
                )
            )
            / 22.0
        )
        mel_l = np.median(mel_norm[:, k - win : k][:, lv], axis=1)
        mel_r = np.median(mel_norm[:, k : k + win][:, rv], axis=1)
        mel_jump = float(np.linalg.norm(mel_l - mel_r))
        pitches[k - s_idx] = pitch_jump
        polys[k - s_idx] = poly_jump
        scores[k - s_idx] = pitch_jump + 1.2 * poly_jump + 0.30 * mfcc_jump + 1.4 * mel_jump

    chosen: list[int] = []
    order = np.argsort(scores)[::-1]
    for rel in order:
        if pitches[rel] < 0.42 and polys[rel] < 0.24 and scores[rel] < 0.60:
            break
        k = s_idx + int(rel)
        lo = max(s_idx + 4, k - 4)
        hi = min(e_idx - 4, k + 5)
        dip = lo + int(np.argmin(rms[lo:hi])) if hi > lo else k
        if any(abs(dip - prev) < win for prev in chosen):
            continue
        if dip - s_idx < win // 2 or e_idx - dip < win // 2:
            continue
        chosen.append(dip)

    cuts = [s_idx, *sorted(chosen), e_idx]
    return [(a, b) for a, b in zip(cuts[:-1], cuts[1:]) if b > a]


def _cluster_two_solo_voices(
    oct_vals: np.ndarray,
    mfcc_mat: np.ndarray,
    durs: np.ndarray,
    *,
    force_two_voices: bool,
) -> dict[int, str] | None:
    """Cluster solo phrases into ``voice_1`` (low) and ``voice_2`` (high), or None if 1 singer."""
    if len(oct_vals) < 2:
        return None
    feat_pitch = ((oct_vals - oct_vals.mean()) * 2.0)[:, None]
    mfcc_scale = np.maximum(mfcc_mat.std(axis=0), 6.0)
    feat_mfcc = ((mfcc_mat - mfcc_mat.mean(axis=0)) / mfcc_scale) * 0.35
    features = np.hstack([feat_pitch, feat_mfcc])

    order = np.argsort(oct_vals)
    c0 = features[order[len(order) // 6]].copy()
    c1 = features[order[max(0, len(order) - 1 - len(order) // 6)]].copy()
    if np.allclose(c0, c1):
        dists = np.linalg.norm(features[:, None, :] - features[None, :, :], axis=2)
        i0, i1 = np.unravel_index(int(np.argmax(dists)), dists.shape)
        c0, c1 = features[i0].copy(), features[i1].copy()
        if np.allclose(c0, c1):
            return None

    labels = np.zeros(len(oct_vals), dtype=int)
    for _ in range(15):
        d0 = np.linalg.norm(features - c0, axis=1)
        d1 = np.linalg.norm(features - c1, axis=1)
        new_labels = (d1 < d0).astype(int)
        if np.array_equal(new_labels, labels) and int(new_labels.sum()) not in (0, len(oct_vals)):
            break
        labels = new_labels
        if int((labels == 0).sum()) == 0 or int((labels == 1).sum()) == 0:
            return None
        c0 = features[labels == 0].mean(axis=0)
        c1 = features[labels == 1].mean(axis=0)

    total_dur = float(durs.sum())
    min_cluster_dur = min(float(durs[labels == 0].sum()), float(durs[labels == 1].sum()))
    if min_cluster_dur < min(0.8, 0.12 * total_dur):
        return None

    oct_0 = float(np.median(oct_vals[labels == 0]))
    oct_1 = float(np.median(oct_vals[labels == 1]))
    delta_semitones = 12.0 * abs(oct_0 - oct_1)
    inter_dist = float(np.linalg.norm(c0 - c1))
    intra_0 = float(np.mean(np.linalg.norm(features[labels == 0] - c0, axis=1)))
    intra_1 = float(np.mean(np.linalg.norm(features[labels == 1] - c1, axis=1)))
    fisher = inter_dist / max(0.5 * (intra_0 + intra_1), 0.25)

    distinct = (
        (delta_semitones >= 2.0 or inter_dist >= 0.75)
        if force_two_voices
        else ((delta_semitones >= 4.5 and fisher >= 1.5) or (inter_dist >= 1.6 and fisher >= 2.4))
    )
    if not distinct:
        return None
    low_label = 0 if oct_0 <= oct_1 else 1
    voice_for = {low_label: "voice_1", 1 - low_label: "voice_2"}
    return {idx: voice_for[int(lbl)] for idx, lbl in enumerate(labels)}


def _relabel_octave_unison_phrases(
    solo_indices: np.ndarray,
    voice_labels: list[str],
    mel_profiles: np.ndarray,
) -> None:
    """Detect octave-unison phrases whose spectrum is a convex mixture of both solo voices."""
    if len(solo_indices) < 3:
        return
    v1_idx = [int(i) for i in solo_indices if voice_labels[int(i)] == "voice_1"]
    v2_idx = [int(i) for i in solo_indices if voice_labels[int(i)] == "voice_2"]
    if not v1_idx or not v2_idx:
        return

    mean_v1 = mel_profiles[v1_idx].mean(axis=0)
    mean_v2 = mel_profiles[v2_idx].mean(axis=0)
    # Build pure solo prototypes from the half of each cluster farthest from the other voice.
    v1_pure = sorted(
        v1_idx, key=lambda i: float(np.linalg.norm(mel_profiles[i] - mean_v2)), reverse=True
    )
    v2_pure = sorted(
        v2_idx, key=lambda i: float(np.linalg.norm(mel_profiles[i] - mean_v1)), reverse=True
    )
    p1 = mel_profiles[v1_pure[: max(1, (len(v1_pure) + 1) // 2)]].mean(axis=0)
    p2 = mel_profiles[v2_pure[: max(1, (len(v2_pure) + 1) // 2)]].mean(axis=0)
    p1 = p1 / max(float(np.linalg.norm(p1)), 1e-12)
    p2 = p2 / max(float(np.linalg.norm(p2)), 1e-12)
    if float(np.linalg.norm(p1 - p2)) < 0.20:
        return

    basis = np.column_stack([p1, p2])
    candidates: list[int] = []
    for idx in solo_indices.tolist():
        prof = mel_profiles[idx]
        weights, *_ = np.linalg.lstsq(basis, prof, rcond=None)
        weights = np.maximum(0.0, weights)
        w_sum = float(weights.sum())
        if w_sum <= 1e-9:
            continue
        shares = weights / w_sum
        err_both = float(np.linalg.norm(prof - basis @ weights))
        err_1 = float(np.linalg.norm(prof - p1 * float(p1 @ prof)))
        err_2 = float(np.linalg.norm(prof - p2 * float(p2 @ prof)))
        err_single = min(err_1, err_2)
        if min(float(shares[0]), float(shares[1])) >= 0.25 and err_both < 0.78 * err_single:
            candidates.append(idx)

    # Keep at least one solo anchor overall so two pure solo voices never collapse completely.
    if candidates and len(candidates) < len(solo_indices):
        for idx in candidates:
            voice_labels[idx] = BOTH_VOICES


def detect_vocal_turns(
    samples: np.ndarray,
    sr: int = ANALYSIS_SR,
    *,
    force_two_voices: bool = False,
) -> list[VocalTurn]:
    """Diarize singing into ``voice_1`` (solo low), ``voice_2`` (solo high), and ``both``.

    Combines a gated harmonic-comb pitch & polyphony detector (catching simultaneous harmony and
    counter-melody) with vocal-tract MFCC clustering and non-negative spectral mixture fitting
    (catching octave unison). Returns ``[]`` on single-singer or silent tracks.
    """
    if samples.size < 2048 or sr < 4000:
        return []
    clean_samples = np.nan_to_num(
        np.asarray(samples, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
    )
    rms, times = rms_envelope(clean_samples, sr)
    peak = float(rms.max()) if rms.size else 0.0
    if not math.isfinite(peak) or peak <= 0.0:
        return []

    raw_phrases = detect_vocal_windows(
        rms, times, threshold_db=-32.0, min_duration=0.25, merge_gap=0.12
    )
    if not raw_phrases:
        return []
    if len(raw_phrases) == 1 and raw_phrases[0].duration < 1.2 and not force_two_voices:
        return []

    import librosa

    stft_mag = np.abs(librosa.stft(clean_samples, n_fft=2048, hop_length=HOP_LENGTH))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    flatness = librosa.feature.spectral_flatness(S=stft_mag)[0]
    mel_fmax = min(5000.0, sr * 0.45)
    mel = np.sqrt(
        librosa.feature.melspectrogram(S=stft_mag**2, sr=sr, n_mels=40, fmin=70.0, fmax=mel_fmax)
    )
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(stft_mag**2 + 1e-12), n_mfcc=13)[1:9]

    n = min(len(rms), len(times), stft_mag.shape[1], len(flatness), mel.shape[1], mfcc.shape[1])
    if n < 8:
        return []
    rms_n, times_n = rms[:n], times[:n]
    stft_n, mfcc_n, mel_n = stft_mag[:, :n], mfcc[:, :n], mel[:, :n]
    mel_norms = np.maximum(np.linalg.norm(mel_n, axis=0, keepdims=True), 1e-12)
    mel_unit = mel_n / mel_norms

    voiced = (rms_n > peak * (10.0 ** (-32.0 / 20.0))) & (flatness[:n] < 0.35)
    f0_n, r2_n = _comb_pitch_and_polyphony(stft_n, freqs, voiced)
    voiced &= (f0_n >= 75.0) & (f0_n <= 550.0)
    log_f0 = np.log2(np.clip(f0_n, 75.0, 550.0))
    hop = float(times_n[1] - times_n[0]) if n > 1 else HOP_LENGTH / sr

    phrases: list[tuple[float, float, float, np.ndarray, np.ndarray, float, float]] = []
    for span in raw_phrases:
        s_idx = max(0, int(np.searchsorted(times_n, span.start, side="left")))
        e_idx = min(n, int(np.searchsorted(times_n, span.end, side="right")))
        for sub_s, sub_e in _split_phrase_voice_changes(
            s_idx, e_idx, rms_n, voiced, log_f0, r2_n, mfcc_n, mel_unit, sr
        ):
            v_mask = voiced[sub_s:sub_e]
            if int(v_mask.sum()) < 4:
                continue
            sub_start = max(0.0, float(times_n[sub_s]) - hop / 2)
            sub_end = float(times_n[sub_e - 1]) + hop / 2
            if sub_end - sub_start < 0.25:
                continue
            oct_med = float(np.median(log_f0[sub_s:sub_e][v_mask]))
            hz_med = float(np.median(f0_n[sub_s:sub_e][v_mask]))
            r2_med = float(np.median(r2_n[sub_s:sub_e][v_mask]))
            mfcc_med = np.median(mfcc_n[:, sub_s:sub_e][:, v_mask], axis=1)
            prof_med = np.median(mel_unit[:, sub_s:sub_e][:, v_mask], axis=1)
            prof_med = prof_med / max(float(np.linalg.norm(prof_med)), 1e-12)
            phrases.append((sub_start, sub_end, oct_med, mfcc_med, prof_med, r2_med, hz_med))

    if not phrases:
        return []

    durs = np.array([p[1] - p[0] for p in phrases], dtype=np.float64)
    total_dur = float(durs.sum())
    poly_mask = np.array([p[5] >= 0.25 for p in phrases], dtype=bool)
    solo_indices = np.flatnonzero(~poly_mask)

    voice_labels = [""] * len(phrases)
    for idx in np.flatnonzero(poly_mask).tolist():
        voice_labels[idx] = BOTH_VOICES

    solo_clusters = (
        _cluster_two_solo_voices(
            np.array([phrases[i][2] for i in solo_indices], dtype=np.float64),
            np.array([phrases[i][3] for i in solo_indices], dtype=np.float64),
            durs[solo_indices],
            force_two_voices=force_two_voices,
        )
        if len(solo_indices) >= 2
        else None
    )

    if solo_clusters is not None:
        for local_i, phrase_idx in enumerate(solo_indices.tolist()):
            voice_labels[phrase_idx] = solo_clusters[local_i]
        mel_profiles = np.array([p[4] for p in phrases], dtype=np.float64)
        _relabel_octave_unison_phrases(solo_indices, voice_labels, mel_profiles)
    elif poly_mask.all():
        if float(np.median([p[5] for p in phrases])) < 0.30:
            return []
    elif poly_mask.any() and (~poly_mask).any():
        poly_dur = float(durs[poly_mask].sum())
        solo_dur = float(durs[~poly_mask].sum())
        if min(poly_dur, solo_dur) < min(0.8, 0.12 * total_dur):
            return []
        for idx in solo_indices.tolist():
            voice_labels[idx] = "voice_1"
    else:
        return []

    merged: list[tuple[float, float, str, list[tuple[float, float]]]] = []
    for (p_start, p_end, _oct, _mfcc, _prof, _r2, hz_med), voice in zip(phrases, voice_labels):
        dur = p_end - p_start
        if merged and merged[-1][2] == voice and p_start - merged[-1][1] <= 0.6:
            prev_s, _prev_e, prev_v, hz_weights = merged[-1]
            hz_weights.append((hz_med, dur))
            merged[-1] = (prev_s, p_end, prev_v, hz_weights)
        else:
            merged.append((p_start, p_end, voice, [(hz_med, dur)]))

    return [
        VocalTurn(
            start=round(s, 3),
            end=round(e, 3),
            voice=v,
            median_f0_hz=(
                None
                if v == BOTH_VOICES
                else round(sum(h * w for h, w in hw) / max(sum(w for _, w in hw), 1e-9), 1)
            ),
        )
        for s, e, v, hw in merged
    ]


def bar_intensity(
    rms: np.ndarray, times: np.ndarray, bars: list[float], duration: float
) -> list[float]:
    """Mean loudness of each bar, scaled so the loudest bar is 1.0."""
    if not bars:
        return []
    edges = [*bars, duration]
    means = []
    for start, end in zip(edges[:-1], edges[1:]):
        mask = (times >= start) & (times < end)
        means.append(float(rms[mask].mean()) if mask.any() else 0.0)
    peak = max(means)
    return [round(m / peak, 3) if peak > 0 else 0.0 for m in means]


def _estimate_key_mode(
    chroma_mean: np.ndarray,
) -> tuple[str | None, Literal["major", "minor"] | None]:
    """Estimate musical key and mode via Krumhansl-Schmuckler profile correlation."""
    centered = chroma_mean.astype(np.float64) - float(np.mean(chroma_mean))
    norm = float(np.linalg.norm(centered))
    if not math.isfinite(norm) or norm <= 1e-9:
        return None, None
    vec = centered / norm

    best_corr = -1.0
    best_key: str | None = None
    best_mode: Literal["major", "minor"] | None = None
    mode_profiles: tuple[tuple[Literal["major", "minor"], np.ndarray], ...] = (
        ("major", _KRUMHANSL_MAJOR),
        ("minor", _KRUMHANSL_MINOR),
    )
    for mode_name, raw_profile in mode_profiles:
        prof = raw_profile - float(np.mean(raw_profile))
        prof = prof / float(np.linalg.norm(prof))
        for tonic, pitch_name in enumerate(_PITCH_CLASSES):
            corr = float(np.dot(vec, np.roll(prof, tonic)))
            if corr > best_corr:
                best_corr = corr
                best_key = f"{pitch_name} {mode_name}"
                best_mode = mode_name
    if best_corr < _MIN_KEY_CORRELATION:
        return None, None
    return best_key, best_mode


def _brightness_label(centroid_hz: float) -> Literal["dark", "warm", "balanced", "bright"]:
    if centroid_hz < 1200.0:
        return "dark"
    if centroid_hz < 2000.0:
        return "warm"
    if centroid_hz < 3200.0:
        return "balanced"
    return "bright"


def _percussiveness_label(
    ratio: float,
) -> Literal["ambient", "melodic", "balanced", "rhythmic", "percussive"]:
    if ratio < 0.12:
        return "ambient"
    if ratio < 0.25:
        return "melodic"
    if ratio < 0.45:
        return "balanced"
    if ratio < 0.65:
        return "rhythmic"
    return "percussive"


def _dynamic_range_label(dr_db: float) -> Literal["flat", "moderate", "dynamic", "cinematic"]:
    if dr_db < 6.0:
        return "flat"
    if dr_db < 12.0:
        return "moderate"
    if dr_db < 18.0:
        return "dynamic"
    return "cinematic"


_MAX_FEATURE_SECONDS = 180.0


def _bounded_mono(mix: np.ndarray, sr: int) -> np.ndarray:
    """Return a finite 1D float32 array bounded to at most ``_MAX_FEATURE_SECONDS``."""
    arr = np.asarray(mix, dtype=np.float32)
    if arr.ndim > 1:
        # Downmix (channels, samples) or (samples, channels) to mono.
        axis = 0 if arr.shape[0] <= arr.shape[-1] else -1
        arr = np.mean(arr, axis=axis, dtype=np.float32)
    arr = np.nan_to_num(arr.reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
    max_samples = int(_MAX_FEATURE_SECONDS * sr)
    if arr.size <= max_samples or max_samples < 768:
        return arr
    # Sample three equal windows (opening, middle, closing) for O(1) memory on long tracks.
    seg = max_samples // 3
    mid = (arr.size - seg) // 2
    return np.concatenate([arr[:seg], arr[mid : mid + seg], arr[-seg:]])


def analyze_musical_features(
    mix: np.ndarray,
    sr: int = ANALYSIS_SR,
    *,
    tempo_bpm: float | None = None,
    harmonic_mix: np.ndarray | None = None,
) -> MusicalFeatures:
    """Measure key/mode, timbral brightness, percussiveness, dynamics, and onset density."""
    if mix.size == 0 or sr <= 0:
        return MusicalFeatures()
    clean = _bounded_mono(mix, sr)
    if clean.size < 256:
        return MusicalFeatures()
    peak = float(np.max(np.abs(clean)))
    if not math.isfinite(peak) or peak < 1e-6:
        return MusicalFeatures()
    clean = clean / np.float32(peak)

    import librosa

    n_fft = min(2048, max(256, 1 << int(math.floor(math.log2(clean.size)))))
    mag = np.abs(librosa.stft(clean, n_fft=n_fft, hop_length=HOP_LENGTH))
    if mag.size == 0:
        return MusicalFeatures()

    frame_rms = librosa.feature.rms(S=mag, frame_length=n_fft)[0]
    peak_rms = float(np.max(frame_rms)) if frame_rms.size else 0.0
    if not math.isfinite(peak_rms) or peak_rms <= 1e-9:
        return MusicalFeatures()
    active = frame_rms > (peak_rms * 0.01)
    if not active.any():
        active = np.ones_like(frame_rms, dtype=bool)

    chroma_source_power = mag**2
    chroma_active = active
    if harmonic_mix is not None and harmonic_mix.size >= 256:
        h_clean = _bounded_mono(harmonic_mix, sr)
        h_peak = float(np.max(np.abs(h_clean))) if h_clean.size else 0.0
        if math.isfinite(h_peak) and h_peak >= 1e-6:
            h_clean = h_clean / np.float32(h_peak)
            h_mag = np.abs(librosa.stft(h_clean, n_fft=n_fft, hop_length=HOP_LENGTH))
            if h_mag.size > 0:
                h_rms = librosa.feature.rms(S=h_mag, frame_length=n_fft)[0]
                h_peak_rms = float(np.max(h_rms)) if h_rms.size else 0.0
                if math.isfinite(h_peak_rms) and h_peak_rms > 1e-9:
                    h_active = h_rms > (h_peak_rms * 0.01)
                    chroma_source_power = h_mag**2
                    chroma_active = h_active if h_active.any() else np.ones_like(h_rms, dtype=bool)

    chroma = librosa.feature.chroma_stft(
        S=chroma_source_power[:, chroma_active], sr=sr, n_fft=n_fft, tuning=0.0
    )
    key, mode = _estimate_key_mode(np.mean(chroma, axis=1))

    centroids = librosa.feature.spectral_centroid(S=mag, sr=sr)[0]
    centroid_hz = float(np.median(centroids[active]))
    if not math.isfinite(centroid_hz) or centroid_hz < 0.0:
        centroid_hz = 0.0
    brightness = _brightness_label(centroid_hz)

    harm, perc = librosa.decompose.hpss(mag)
    e_harm = float(np.sum(harm.astype(np.float64) ** 2))
    e_perc = float(np.sum(perc.astype(np.float64) ** 2))
    e_total = e_harm + e_perc
    perc_ratio = max(0.0, min(1.0, e_perc / e_total)) if e_total > 1e-12 else 0.0
    percussiveness = _percussiveness_label(perc_ratio)

    active_rms = np.maximum(frame_rms[active].astype(np.float64), 1e-10)
    rms_db = 20.0 * np.log10(active_rms / peak_rms)
    dr_db = max(0.0, float(np.percentile(rms_db, 95) - np.percentile(rms_db, 15)))
    if not math.isfinite(dr_db):
        dr_db = 0.0
    dynamic_range = _dynamic_range_label(dr_db)

    duration = clean.size / float(sr)
    onset_env = librosa.onset.onset_strength(y=clean, sr=sr, hop_length=HOP_LENGTH)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=HOP_LENGTH)
    onset_density = (len(onsets) / duration) if duration > 0 else 0.0
    if not math.isfinite(onset_density) or onset_density < 0.0:
        onset_density = 0.0

    parts: list[str] = []
    if key:
        parts.append(key)
    parts.append(f"{brightness} timbre")
    parts.append(f"{percussiveness} texture")
    parts.append(f"{dynamic_range} range ({dr_db:.1f} dB)")
    if onset_density > 0:
        parts.append(f"{onset_density:.1f} onsets/s")
    if tempo_bpm is not None and math.isfinite(tempo_bpm) and tempo_bpm > 0:
        parts.append(f"~{tempo_bpm:.0f} BPM")

    return MusicalFeatures(
        key=key,
        mode=mode,
        brightness=brightness,
        spectral_centroid_hz=round(centroid_hz, 1),
        percussiveness=percussiveness,
        percussive_ratio=round(perc_ratio, 3),
        dynamic_range=dynamic_range,
        dynamic_range_db=round(dr_db, 1),
        onset_density=round(onset_density, 2),
        summary=", ".join(parts),
    )


def analyze_project(project: Project) -> Analysis:
    """Analyze the project's mix and stems and return the result (does not save it)."""
    inputs, video = project.config.inputs, project.config.video
    require_vocals = project.config.planning.video_style is VideoStyle.LIPSYNC
    mix_path = project.input_path(inputs.mix, "mix")
    mix = load_mono(mix_path)
    vocals_path = project.path(inputs.vocals) if inputs.vocals else None
    if require_vocals:
        vocals: np.ndarray | None = load_mono(project.input_path(inputs.vocals, "vocals"))
    elif vocals_path is not None and vocals_path.is_file():
        vocals = load_mono(vocals_path)
    else:
        vocals = None
    duration = len(mix) / ANALYSIS_SR

    lyrics_candidate = project.path(inputs.lyrics) if inputs.lyrics else None
    if require_vocals:
        lyrics_path: Path | None = project.input_path(inputs.lyrics, "lyrics")
    elif lyrics_candidate is not None and lyrics_candidate.is_file():
        lyrics_path = lyrics_candidate
    else:
        lyrics_path = None

    lyrics_raw = lyrics_path.read_text("utf-8-sig") if lyrics_path is not None else ""
    lyrics_for_tuning = (
        ""
        if lyrics_path is None or tuning_mod.is_starter_input(lyrics_path, "lyrics.txt")
        else lyrics_raw
    )
    _, style_for_tuning = tuning_mod.read_optional_project_inputs(project)
    pre_profile = tuning_mod.profile_track(
        lyrics_text=lyrics_for_tuning,
        style_text=style_for_tuning,
        title_hint=project.config.title,
    )
    effective_beats_per_bar = video.beats_per_bar
    if (
        project.config.auto_tune
        and video.beats_per_bar == 4
        and pre_profile.beats_per_bar_hint not in (None, 4)
    ):
        effective_beats_per_bar = pre_profile.beats_per_bar_hint
        log.info(
            "Auto-tuned video.beats_per_bar: %d -> %d (meter cue in style/lyrics)",
            video.beats_per_bar,
            effective_beats_per_bar,
        )

    stems_dir = project.path(inputs.stems_dir) if inputs.stems_dir.strip() else None
    discovered_stems = (
        stem_pack_mod.discover_stems(stems_dir)
        if stems_dir is not None and stems_dir != project.root
        else []
    )
    if discovered_stems:
        log.info(
            "Discovered %d stem(s) in %s: %s",
            len(discovered_stems),
            inputs.stems_dir,
            ", ".join(f"{s.name} ({s.role})" for s in discovered_stems),
        )

    # Drums/percussion stems give the cleanest beat grid; fall back to the full mix.
    if inputs.drums:
        beat_source = load_mono(project.input_path(inputs.drums, "drums"))
    else:
        rhythm_stems = stem_pack_mod.select_rhythm_stems(discovered_stems)
        rhythm_audio = (
            stem_pack_mod.load_stems_mono_sum(rhythm_stems, ANALYSIS_SR, loader=load_mono)
            if rhythm_stems
            else None
        )
        beat_source = rhythm_audio if rhythm_audio is not None else mix
    tempo, beats = detect_beats(beat_source)
    bars = bars_from_beats(beats, effective_beats_per_bar)
    log.info("Tempo %.1f BPM, %d beats, %d bars, %.1fs", tempo, len(beats), len(bars), duration)
    if len(bars) < 2:
        log.warning("No usable beat grid was found; shots will be cut at equal intervals.")

    mix_rms, mix_times = rms_envelope(mix)
    if vocals is not None:
        vocal_rms, vocal_times = rms_envelope(vocals)
        windows = detect_vocal_windows(vocal_rms, vocal_times)
    else:
        windows = []
    log.info(
        "Found %d vocal windows covering %.1fs", len(windows), sum(w.duration for w in windows)
    )

    lines = lyrics_mod.parse_lyrics(lyrics_raw)
    words = (
        lyrics_mod.load_or_transcribe_words(project)
        if lines and (vocals is not None or inputs.word_timestamps)
        else []
    )
    if words:
        lines = lyrics_mod.align_lines_to_words(lines, words)
    lines = lyrics_mod.estimate_missing_timings(lines, windows)

    alignment_warnings = (
        alignment_qa_mod.audit_lyric_alignment(vocals, ANALYSIS_SR, lines, windows)
        if vocals is not None and lines
        else []
    )
    if words and bars:
        word_spans = [(w.word, w.start, w.end) for w in words]
        alignment_warnings.extend(alignment_qa_mod.detect_boundary_word_cuts(bars, word_spans))
    if alignment_warnings:
        log.info(
            "Pre-spend alignment QA flagged %d warning(s) (see analysis.json alignment_warnings)",
            len(alignment_warnings),
        )

    multi_singer_hint = lyrics_mod.labels_have_multiple_singers(line.section for line in lines)
    vocal_turns = (
        detect_vocal_turns(vocals, ANALYSIS_SR, force_two_voices=multi_singer_hint)
        if vocals is not None
        else []
    )
    if vocal_turns:
        log.info(
            "Detected %d vocal turns across %d distinct voice(s)",
            len(vocal_turns),
            len({t.voice for t in vocal_turns}),
        )

    harmonic_stems = stem_pack_mod.select_harmonic_stems(discovered_stems)
    harmonic_audio = (
        stem_pack_mod.load_stems_mono_sum(harmonic_stems, ANALYSIS_SR, loader=load_mono)
        if harmonic_stems
        else None
    )
    musical_features = analyze_musical_features(
        mix, ANALYSIS_SR, tempo_bpm=tempo, harmonic_mix=harmonic_audio
    )
    if musical_features.summary:
        log.info("Musical character: %s", musical_features.summary)

    stem_arrangement = (
        stem_pack_mod.analyze_stems(
            discovered_stems,
            project_root=project.root,
            bars=bars,
            duration=duration,
            sr=ANALYSIS_SR,
            loader=load_mono,
        )
        if discovered_stems
        else None
    )
    if stem_arrangement and stem_arrangement.summary:
        log.info("Stem arrangement: %s", stem_arrangement.summary)

    bqa_report = broadcast_qa_mod.analyze_broadcast_audio(
        mix_path,
        drum_onset_times=beats,
    )
    broadcast_qa_model = BroadcastAudioQA(
        integrated_lufs=bqa_report.integrated_lufs,
        true_peak_dbtp=bqa_report.true_peak_dbtp,
        splice_clicks=bqa_report.splice_clicks,
    )
    if bqa_report.true_peak_dbtp > -0.5 or bqa_report.splice_clicks:
        log.warning(
            "Broadcast audio QA: %.1f LUFS, %.2f dBTP, %d splice click(s)",
            bqa_report.integrated_lufs,
            bqa_report.true_peak_dbtp,
            len(bqa_report.splice_clicks),
        )
    else:
        log.info(
            "Broadcast audio QA: %.1f LUFS, %.2f dBTP (clean)",
            bqa_report.integrated_lufs,
            bqa_report.true_peak_dbtp,
        )

    analysis = Analysis(
        duration=round(duration, 3),
        tempo_bpm=round(tempo, 2),
        beats=[round(b, 3) for b in beats],
        bars=[round(b, 3) for b in bars],
        bar_intensity=bar_intensity(mix_rms, mix_times, bars, duration),
        vocal_windows=windows,
        vocal_turns=vocal_turns,
        lyric_lines=lines,
        alignment_warnings=alignment_warnings,
        broadcast_qa=broadcast_qa_model,
        musical_features=musical_features,
        stems=stem_arrangement,
    )
    profile = tuning_mod.profile_track(
        lyrics_text=lyrics_for_tuning,
        style_text=style_for_tuning,
        title_hint=project.config.title,
        analysis=analysis,
    )
    return analysis.model_copy(update={"track_profile": profile})
