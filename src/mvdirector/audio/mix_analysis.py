# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Windowed, bounded-memory audio analysis for long-form continuous mixes.

Analyzes each track segment independently so multi-BPM DJ mixes get per-chapter tempo,
downbeat phase, and energy measurements while keeping peak RAM small on 30-to-90+ minute files.
"""

from __future__ import annotations

import logging
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

from mvdirector import media
from mvdirector.audio.analysis import (
    ANALYSIS_SR,
    HOP_LENGTH,
    bar_intensity,
    detect_beats,
    rms_envelope,
)
from mvdirector.audio.tracklist import (
    ResolvedTrack,
    TracklistSpec,
    load_tracklist,
    resolve_chapters,
)
from mvdirector.errors import ConfigError, MediaToolError
from mvdirector.models import Analysis, AnalyzedChapter
from mvdirector.project import Project

log = logging.getLogger(__name__)

_RMS_CHUNK_SECONDS = 180.0
_BEAT_WINDOW_SECONDS = 120.0
_DEFAULT_BPM = 120.0


def probe_audio_duration(path: Path) -> float:
    """Return audio duration in seconds without decoding the full file into memory."""
    if not path.is_file():
        raise ConfigError(f"Audio file not found: {path}")
    try:
        duration = float(sf.info(str(path)).duration)
    except (OSError, RuntimeError):
        duration = media.probe(path).duration
    if not math.isfinite(duration) or duration <= 0.0:
        raise ConfigError(f"Could not determine a positive duration for audio file: {path}")
    return round(duration, 3)


def load_mono_window(
    path: Path,
    start: float,
    duration: float,
    sr: int = ANALYSIS_SR,
) -> np.ndarray:
    """Decode ``[start, start + duration]`` seconds of ``path`` as mono float32 samples."""
    if duration <= 0.0:
        return np.zeros(0, dtype=np.float32)
    safe_start = max(0.0, start)
    if shutil.which("ffmpeg") is not None:
        cmd = [
            "ffmpeg",
            "-v",
            "error",
            "-ss",
            f"{safe_start:.4f}",
            "-t",
            f"{duration:.4f}",
            "-i",
            str(path.resolve()),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(sr),
            "-f",
            "f32le",
            "pipe:1",
        ]
        proc = subprocess.run(cmd, capture_output=True, check=False)
        if proc.returncode != 0:
            tail = "\n".join(
                proc.stderr.decode("utf-8", errors="replace").strip().splitlines()[-8:]
            )
            raise MediaToolError(f"ffmpeg audio decode failed (exit {proc.returncode}):\n{tail}")
        return np.frombuffer(proc.stdout, dtype=np.float32).copy()

    import librosa

    samples, _ = librosa.load(str(path), sr=sr, mono=True, offset=safe_start, duration=duration)
    return np.asarray(samples, dtype=np.float32)


def _stream_mix_rms(
    path: Path, duration: float, sr: int = ANALYSIS_SR
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the full-mix RMS envelope in bounded-size chunks."""
    rms_parts: list[np.ndarray] = []
    time_parts: list[np.ndarray] = []
    cursor = 0.0
    while cursor < duration - 1e-3:
        chunk_dur = min(_RMS_CHUNK_SECONDS, duration - cursor)
        samples = load_mono_window(path, cursor, chunk_dur, sr=sr)
        if samples.size >= HOP_LENGTH:
            chunk_rms, chunk_times = rms_envelope(samples, sr=sr)
            rms_parts.append(chunk_rms)
            time_parts.append(chunk_times + cursor)
        cursor += chunk_dur
    if not rms_parts:
        return np.zeros(1, dtype=np.float64), np.zeros(1, dtype=np.float64)
    return np.concatenate(rms_parts), np.concatenate(time_parts)


def _low_energy_dip_candidates(rms: np.ndarray, times: np.ndarray) -> list[float]:
    """Local loudness minima suitable for snapping automatic chapter boundaries."""
    if rms.size < 16:
        return []
    hop_dt = float(times[1] - times[0]) if times.size > 1 else HOP_LENGTH / ANALYSIS_SR
    smooth_win = max(3, int(round(4.0 / max(hop_dt, 1e-4))))
    kernel = np.ones(smooth_win, dtype=np.float64) / smooth_win
    smoothed = np.convolve(rms, kernel, mode="same")
    radius = max(2, smooth_win)
    dips: list[float] = []
    for idx in range(radius, len(smoothed) - radius):
        local = smoothed[idx - radius : idx + radius + 1]
        if smoothed[idx] <= float(local.min()) + 1e-12 and (
            not dips or float(times[idx]) - dips[-1] >= 4.0
        ):
            dips.append(round(float(times[idx]), 3))
    return dips


def _align_beat_phase(rel_beats: list[float], beat_sec: float) -> float:
    """Circular-mean beat phase in ``[0, beat_sec)`` matching detected onset beats."""
    if not rel_beats or beat_sec <= 0.0:
        return 0.0
    angles = np.array(
        [2.0 * math.pi * (b % beat_sec) / beat_sec for b in rel_beats], dtype=np.float64
    )
    mean_angle = math.atan2(float(np.sin(angles).mean()), float(np.cos(angles).mean()))
    if mean_angle < 0.0:
        mean_angle += 2.0 * math.pi
    return (mean_angle / (2.0 * math.pi)) * beat_sec


def _chapter_beats_and_bpm(
    mix_path: Path,
    track: ResolvedTrack,
    beats_per_bar: int,
) -> tuple[float, float, list[float], list[float]]:
    """Measure or phase-lock ``(bpm, downbeat_offset, beats, bars)`` for one chapter."""
    win_dur = min(track.end - track.start, _BEAT_WINDOW_SECONDS)
    samples = load_mono_window(mix_path, track.start, win_dur, sr=ANALYSIS_SR)
    detected_bpm = _DEFAULT_BPM
    rel_beats: list[float] = []
    if samples.size >= ANALYSIS_SR:
        raw_bpm, rel_beats = detect_beats(samples, sr=ANALYSIS_SR)
        if math.isfinite(raw_bpm) and raw_bpm > 0.0:
            detected_bpm = raw_bpm

    if track.bpm is not None:
        bpm = round(track.bpm, 2)
        beat_sec = 60.0 / bpm
        phase = _align_beat_phase(rel_beats, beat_sec)
    else:
        bpm = round(detected_bpm, 2)
        beat_sec = 60.0 / bpm
        phase = (rel_beats[0] % beat_sec) if rel_beats else 0.0

    downbeat = min(track.end - 1e-3, round(track.start + phase, 3))
    downbeat = max(track.start, downbeat)

    beats: list[float] = []
    count = int(math.floor((track.end - downbeat) / beat_sec)) + 1
    for k in range(max(0, count)):
        t = round(downbeat + k * beat_sec, 3)
        if track.start <= t < track.end:
            beats.append(t)
    bars = beats[:: max(1, beats_per_bar)]
    return bpm, downbeat, beats, bars


def analyze_mix_project(
    project: Project,
    *,
    tracklist_override: Path | None = None,
) -> Analysis:
    """Analyze a long-form mix in windowed chunks and return an ``Analysis`` with ``chapters``."""
    inputs, video, mix_cfg = project.config.inputs, project.config.video, project.config.mix
    mix_path = project.input_path(inputs.mix, "mix")
    duration = probe_audio_duration(mix_path)

    spec: TracklistSpec | None = None
    if tracklist_override is not None:
        spec = load_tracklist(tracklist_override)
    elif inputs.tracklist:
        spec = load_tracklist(project.input_path(inputs.tracklist, "tracklist"))

    mix_rms, mix_times = _stream_mix_rms(mix_path, duration)
    dip_candidates = _low_energy_dip_candidates(mix_rms, mix_times) if spec is None else None
    resolved = resolve_chapters(
        spec,
        duration,
        default_segment_seconds=mix_cfg.default_segment_seconds,
        snap_candidates=dip_candidates,
    )

    raw_means: list[float] = []
    for item in resolved:
        mask = (mix_times >= item.start) & (mix_times < item.end)
        raw_means.append(float(mix_rms[mask].mean()) if mask.any() else 0.0)
    peak_rms = max([float(mix_rms.max()), *raw_means, 0.0])

    chapters: list[AnalyzedChapter] = []
    all_beats: list[float] = []
    all_bars: list[float] = []
    weighted_bpm = 0.0

    for item, mean_rms in zip(resolved, raw_means):
        bpm, downbeat, ch_beats, ch_bars = _chapter_beats_and_bpm(
            mix_path, item, video.beats_per_bar
        )
        if item.energy is not None:
            energy = round(item.energy, 3)
        elif peak_rms > 0.0:
            # Normalize chapter mean RMS against the loudest chapter while clamping [0, 1].
            chapter_peak = max(raw_means) or peak_rms
            energy = round(min(1.0, max(0.05, mean_rms / chapter_peak)), 3)
        else:
            energy = 0.5

        chapters.append(
            AnalyzedChapter(
                id=item.id,
                index=item.index,
                start=item.start,
                end=item.end,
                title=item.title,
                artist=item.artist,
                bpm=bpm,
                downbeat_offset=downbeat,
                energy=energy,
                mood_tags=item.mood_tags,
                genre_tags=item.genre_tags,
                theme_prompt=item.theme_prompt,
                clip_prompts=item.clip_prompts,
                lyrics=item.lyrics,
            )
        )
        all_beats.extend(ch_beats)
        all_bars.extend(ch_bars)
        weighted_bpm += bpm * (item.end - item.start)

    avg_bpm = round(weighted_bpm / duration, 2) if duration > 0 else _DEFAULT_BPM
    intensities = bar_intensity(mix_rms, mix_times, all_bars, duration)
    log.info(
        "Mix analysis complete: %.1fs across %d chapter(s), mean tempo %.1f BPM",
        duration,
        len(chapters),
        avg_bpm,
    )
    return Analysis(
        duration=duration,
        tempo_bpm=avg_bpm,
        beats=all_beats,
        bars=all_bars,
        bar_intensity=intensities,
        vocal_windows=[],
        vocal_turns=[],
        lyric_lines=[],
        chapters=chapters,
    )
