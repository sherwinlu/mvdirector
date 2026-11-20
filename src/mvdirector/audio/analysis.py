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

import numpy as np

from mvdirector.audio import lyrics as lyrics_mod
from mvdirector.models import Analysis, TimeSpan
from mvdirector.project import Project

log = logging.getLogger(__name__)

ANALYSIS_SR = 22050
HOP_LENGTH = 512
# Tempos outside this range are folded by whole octaves: beat trackers often lock onto half or
# double time (a slow blues reads as 185 BPM), and cuts follow the beat grid.
TEMPO_RANGE_BPM = (70.0, 160.0)


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
) -> list[TimeSpan]:
    """Spans where the vocal stem is louder than ``threshold_db`` relative to its peak.

    Gaps shorter than ``merge_gap`` (breaths) are bridged; blips shorter than
    ``min_duration`` (stem bleed) are dropped.
    """
    peak = float(rms.max()) if rms.size else 0.0
    if peak <= 0.0:
        return []
    active = 20.0 * np.log10(np.maximum(rms, 1e-10) / peak) > threshold_db
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


def analyze_project(project: Project) -> Analysis:
    """Analyze the project's mix and stems and return the result (does not save it)."""
    inputs, video = project.config.inputs, project.config.video
    mix = load_mono(project.input_path(inputs.mix, "mix"))
    vocals = load_mono(project.input_path(inputs.vocals, "vocals"))
    duration = len(mix) / ANALYSIS_SR

    # Drums give the cleanest beat grid; fall back to the full mix.
    beat_source = load_mono(project.input_path(inputs.drums, "drums")) if inputs.drums else mix
    tempo, beats = detect_beats(beat_source)
    bars = bars_from_beats(beats, video.beats_per_bar)
    log.info("Tempo %.1f BPM, %d beats, %d bars, %.1fs", tempo, len(beats), len(bars), duration)
    if len(bars) < 2:
        log.warning("No usable beat grid was found; shots will be cut at equal intervals.")

    mix_rms, mix_times = rms_envelope(mix)
    vocal_rms, vocal_times = rms_envelope(vocals)
    windows = detect_vocal_windows(vocal_rms, vocal_times)
    log.info(
        "Found %d vocal windows covering %.1fs", len(windows), sum(w.duration for w in windows)
    )

    lines = lyrics_mod.parse_lyrics(project.input_path(inputs.lyrics, "lyrics").read_text("utf-8"))
    words = lyrics_mod.load_or_transcribe_words(project)
    if words:
        lines = lyrics_mod.align_lines_to_words(lines, words)
    lines = lyrics_mod.estimate_missing_timings(lines, windows)

    return Analysis(
        duration=round(duration, 3),
        tempo_bpm=round(tempo, 2),
        beats=[round(b, 3) for b in beats],
        bars=[round(b, 3) for b in bars],
        bar_intensity=bar_intensity(mix_rms, mix_times, bars, duration),
        vocal_windows=windows,
        lyric_lines=lines,
    )
