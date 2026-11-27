# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Deterministic timing: turn the beat grid into a gap-free list of shot time ranges.

Timing is computed in code, not by the language model, so cuts always land on bar
lines and the shots always cover the whole song.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from mvdirector import plugins
from mvdirector.audio.lyrics import section_singers
from mvdirector.config import VideoConfig
from mvdirector.models import BOTH_VOICES, Analysis, ShotType, TimeSpan, VocalTurn

log = logging.getLogger(__name__)

_VALID_BAR_STEPS = (16, 8, 4, 2, 1)
_HIGH_INTENSITY_THRESHOLD = 0.80
_MEDIUM_INTENSITY_THRESHOLD = 0.55
_MIN_INTENSITY_SPREAD = 0.15


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    type: ShotType
    lyrics: str
    intensity: float
    section: str | None = None
    voice: str | None = None

    @property
    def duration(self) -> float:
        return self.end - self.start


def frames_for(duration: float, fps: int, multiple: int, offset: int) -> int:
    """Smallest frame count of the form ``multiple * n + offset`` covering ``duration``."""
    needed = math.ceil(duration * fps - 1e-6)
    steps = max(1, math.ceil((needed - offset) / multiple))
    return multiple * steps + offset


def split_long_span_at_silences(
    start: float,
    end: float,
    vocal_windows: list[TimeSpan],
    fps: int,
    *,
    min_seconds: float,
    max_seconds: float,
) -> list[int]:
    """Split ``[start, end]`` into exact-integer frame counts preferring inter-vocal silence gaps.

    Guarantees ``sum(counts) == round(end * fps) - round(start * fps)`` so chained sub-clips
    never accumulate frame drift.

    Credits & Attribution:
        Silence-midpoint exact-integer sub-clip splitting inspired by ``tsondo/musicvision``
        (https://github.com/tsondo/musicvision — ``src/musicvision/engine_registry.py``).
    """
    total_frames = max(1, round(end * fps) - round(start * fps))
    max_frames = max(1, int(math.floor(max_seconds * fps)))
    min_frames = max(1, min(max_frames, int(math.ceil(min_seconds * fps))))
    if total_frames <= max_frames:
        return [total_frames]

    # Collect silence gap midpoints inside (start, end) between sorted vocal windows.
    clipped = sorted(
        (
            (max(start, w.start), min(end, w.end))
            for w in vocal_windows
            if min(end, w.end) > max(start, w.start)
        ),
        key=lambda pair: pair[0],
    )
    silence_frames: list[int] = []
    for prev_w, next_w in zip(clipped[:-1], clipped[1:]):
        if next_w[0] - prev_w[1] >= 0.15:
            mid = 0.5 * (prev_w[1] + next_w[0])
            f_idx = round((mid - start) * fps)
            if 0 < f_idx < total_frames:
                silence_frames.append(f_idx)

    counts: list[int] = []
    cursor = 0
    while total_frames - cursor > max_frames:
        remaining = total_frames - cursor
        lo = cursor + min_frames
        hi = min(cursor + max_frames, total_frames - min_frames)
        if hi < lo:
            half = remaining // 2
            counts.append(half)
            cursor += half
            continue
        ideal = cursor + remaining / max(2, math.ceil(remaining / max_frames))
        candidates = [f for f in silence_frames if lo <= f <= hi]
        cut = min(candidates, key=lambda f: abs(f - ideal)) if candidates else int(round(ideal))
        cut = max(lo, min(hi, cut))
        counts.append(cut - cursor)
        cursor = cut
    counts.append(total_frames - cursor)
    return counts


def dp_phrase_cuts(
    analysis: Analysis,
    video: VideoConfig,
    candidate_points: list[float],
) -> list[float]:
    """Shortest-path DP over ``candidate_points`` minimizing mid-phrase/mid-lyric splits.

    Selects a subset of ``candidate_points`` (keeping ``0.0`` and ``analysis.duration``) such
    that every segment stays within ``[min_clip_seconds, max_clip_seconds]`` when possible and
    avoids slicing through active lyric lines or vocal windows.

    Credits & Attribution:
        Global shortest-path phrase-aware slicing inspired by ``JavaDerek/music-video-pipeline``
        (https://github.com/JavaDerek/music-video-pipeline — ``slicing.py:phrase_aware_slicing``).
    """
    if analysis.duration <= 0.0:
        return [0.0]
    interior = sorted(
        {round(p, 3) for p in candidate_points if 1e-6 < p < analysis.duration - 1e-6}
    )
    pts = [0.0, *[p for p in interior if p < analysis.duration - 1e-6], analysis.duration]
    n = len(pts)
    if n <= 2:
        return pts

    def _split_penalty(cut_time: float) -> float:
        if cut_time <= 0.0 or cut_time >= analysis.duration:
            return 0.0
        penalty = 0.0
        for line in analysis.lyric_lines:
            if (
                line.start is not None
                and line.end is not None
                and line.start + 0.08 < cut_time < line.end - 0.08
            ):
                penalty += 10.0
        for win in analysis.vocal_windows:
            if win.start + 0.12 < cut_time < win.end - 0.12:
                penalty += 3.0
        return penalty

    penalties = [_split_penalty(p) for p in pts]
    lyric_spans = _aligned_lyric_spans(analysis) if video.broll_min_clip_seconds is not None else []
    intensities = analysis.bar_intensity
    has_contrast = (
        video.dynamic_pacing
        and len(intensities) >= len(analysis.bars)
        and len(analysis.bars) >= 2
        and (max(intensities[: len(analysis.bars)]) - min(intensities[: len(analysis.bars)]))
        >= _MIN_INTENSITY_SPREAD
    )
    inf = float("inf")
    dp = [inf] * n
    prev = [-1] * n
    dp[0] = 0.0

    for j in range(1, n):
        for i in range(j - 1, -1, -1):
            span = pts[j] - pts[i]
            if span > video.max_clip_seconds + 1e-6:
                break
            min_span = _min_clip_for_span(analysis, video, pts[i], pts[j], lyric_spans)
            if span < min_span - 1e-6:
                continue
            if has_contrast:
                span_intensity = _intensity_between(analysis, pts[i], pts[j])
                if span_intensity >= _HIGH_INTENSITY_THRESHOLD:
                    target_dur = min_span
                elif span_intensity < _MEDIUM_INTENSITY_THRESHOLD:
                    target_dur = video.max_clip_seconds
                else:
                    target_dur = 0.5 * (min_span + video.max_clip_seconds)
            else:
                target_dur = 0.5 * (min_span + video.max_clip_seconds)
            dur_cost = 0.15 * abs(span - target_dur)
            cost = dp[i] + penalties[j] + dur_cost
            if cost < dp[j]:
                dp[j] = cost
                prev[j] = i

    if prev[n - 1] == -1:
        return []  # no feasible constrained tiling; caller falls back cleanly

    chosen: list[float] = []
    cur = n - 1
    while cur >= 0:
        chosen.append(pts[cur])
        cur = prev[cur]
    chosen.reverse()
    return chosen


def shift_to_vocal_onset(
    points: list[float],
    vocal_windows: list[TimeSpan],
    video: VideoConfig,
    *,
    max_shift_seconds: float = 0.45,
) -> list[float]:
    """Locally nudge cut boundaries (zero cascade) to trim leading silence before vocal onsets.

    When a vocal phrase begins shortly after a cut (within ``max_shift_seconds``), shifts only
    that single boundary forward to the vocal onset if both neighboring shots remain within
    ``[min_clip_seconds, max_clip_seconds]`` and the previous shot is not actively sung there.

    Credits & Attribution:
        Zero-cascade local vocal-onset shifting inspired by ``JavaDerek/music-video-pipeline``
        (https://github.com/JavaDerek/music-video-pipeline — ``slicing.py:_prefer_vocal_onset``).
    """
    if len(points) < 3 or not vocal_windows:
        return list(points)
    result = list(points)
    for idx in range(1, len(result) - 1):
        cut = result[idx]
        prev_cut = result[idx - 1]
        next_cut = result[idx + 1]
        # Skip if vocals are already active across this boundary.
        if any(w.start <= cut <= w.end for w in vocal_windows):
            continue
        upcoming = [w.start for w in vocal_windows if cut < w.start <= cut + max_shift_seconds]
        if not upcoming:
            continue
        onset = round(min(upcoming), 3)
        if (
            onset - prev_cut <= video.max_clip_seconds + 1e-6
            and next_cut - onset >= video.min_clip_seconds - 1e-6
        ):
            result[idx] = onset
    return result


def _allowed_bar_steps(
    bar_length: float, video: VideoConfig, *, min_clip_override: float | None = None
) -> list[int]:
    """Bar step sizes (from 16, 8, 4, 2, 1) that fit within [min_clip, max_clip_seconds]."""
    min_clip = video.min_clip_seconds if min_clip_override is None else min_clip_override
    max_bars = next((n for n in _VALID_BAR_STEPS if n * bar_length <= video.max_clip_seconds), 1)
    min_bars = next(
        (n for n in reversed(_VALID_BAR_STEPS) if n * bar_length >= min_clip),
        1,
    )
    steps = [n for n in _VALID_BAR_STEPS if min_bars <= n <= max_bars]
    return steps or [max_bars]


def _merge_time_spans(spans: Sequence[TimeSpan]) -> list[TimeSpan]:
    """Merge overlapping or adjacent ``TimeSpan`` intervals into disjoint sorted spans."""
    ordered = sorted((s for s in spans if s.end > s.start), key=lambda s: (s.start, s.end))
    if not ordered:
        return []
    merged: list[TimeSpan] = [ordered[0]]
    for span in ordered[1:]:
        prev = merged[-1]
        if span.start <= prev.end:
            merged[-1] = TimeSpan(start=prev.start, end=max(prev.end, span.end))
        else:
            merged.append(span)
    return merged


def _span_overlap_total(spans: Sequence[TimeSpan], start: float, end: float) -> float:
    """Total seconds of ``[start, end]`` covered by the union of ``spans`` (no double-counting)."""
    if end <= start or not spans:
        return 0.0
    clipped = [
        TimeSpan(start=max(start, s.start), end=min(end, s.end))
        for s in spans
        if min(end, s.end) > max(start, s.start)
    ]
    return sum(s.duration for s in _merge_time_spans(clipped))


def _aligned_lyric_spans(analysis: Analysis) -> list[TimeSpan]:
    """Disjoint timed spans for lines aligned by Whisper/forced alignment."""
    raw = [
        TimeSpan(start=ln.start, end=ln.end)
        for ln in analysis.lyric_lines
        if (
            ln.timing_source == "aligned"
            and ln.start is not None
            and ln.end is not None
            and ln.end > ln.start
        )
    ]
    return _merge_time_spans(raw)


def _is_sung_span(
    analysis: Analysis,
    start: float,
    end: float,
    min_vocal_fraction: float,
    lyric_spans: list[TimeSpan],
) -> bool:
    """True when ``[start, end]`` carries genuine singing (guarding against vocal-stem bleed)."""
    duration = end - start
    if duration <= 0.0:
        return False
    sung = _span_overlap_total(analysis.vocal_windows, start, end)
    if sung / duration < min_vocal_fraction:
        return False
    if lyric_spans:
        lyric_overlap = _span_overlap_total(lyric_spans, start, end)
        if lyric_overlap < min(0.25, duration * 0.25):
            return False
    return True


def _min_clip_for_span(
    analysis: Analysis,
    video: VideoConfig,
    start: float,
    end: float,
    lyric_spans: list[TimeSpan],
) -> float:
    """Minimum clip duration for ``[start, end]``: full floor if sung, else B-roll floor."""
    if video.broll_min_clip_seconds is None:
        return video.min_clip_seconds
    if _is_sung_span(analysis, start, end, video.lipsync_min_vocal_fraction, lyric_spans):
        return video.min_clip_seconds
    return video.broll_min_clip_seconds


def _step_for_intensity(intensity: float, allowed_steps: list[int]) -> int:
    """Map bar intensity (0-1) to a bar count: calmer bars hold longer, peaks cut faster."""
    max_bars = allowed_steps[0]
    min_bars = allowed_steps[-1]
    if intensity >= _HIGH_INTENSITY_THRESHOLD:
        target = max(min_bars, max_bars // 4)
    elif intensity >= _MEDIUM_INTENSITY_THRESHOLD:
        target = max(min_bars, max_bars // 2)
    else:
        target = max_bars
    return min(allowed_steps, key=lambda n: abs(n - target))


def _singer_handoff_windows(analysis: Analysis) -> list[tuple[float, float]]:
    """Non-bleeding (gap_start, gap_end) handoff windows whenever the active singer changes."""
    duration = analysis.duration
    windows: list[tuple[float, float]] = []
    for prev_t, next_t in zip(analysis.vocal_turns[:-1], analysis.vocal_turns[1:]):
        if prev_t.voice == next_t.voice:
            continue
        if next_t.start < prev_t.end - 0.15:
            if 0.0 < next_t.start < duration:
                windows.append((next_t.start, next_t.start))
            if abs(prev_t.end - next_t.end) > 0.15 and 0.0 < prev_t.end < duration:
                windows.append((prev_t.end, prev_t.end))
            continue
        gap_s = min(prev_t.end, next_t.start)
        gap_e = max(prev_t.end, next_t.start)
        if 0.0 < 0.5 * (gap_s + gap_e) < duration:
            windows.append((gap_s, gap_e))

    prev_singers: set[str] | None | object = False
    prev_end = 0.0
    timed_lines = sorted(
        (ln for ln in analysis.lyric_lines if ln.start is not None and ln.end is not None),
        key=lambda ln: ln.start,  # type: ignore[arg-type, return-value]
    )
    for line in timed_lines:
        curr = section_singers(line.section)
        has_explicit = curr is None or bool(curr)
        if not has_explicit:
            continue
        if prev_singers is not False and curr != prev_singers:
            l_start = float(line.start)  # type: ignore[arg-type]
            gap_s = min(prev_end, l_start)
            gap_e = max(prev_end, l_start)
            mid = 0.5 * (gap_s + gap_e)
            if 0.0 < mid < duration and not any(
                abs(0.5 * (a + b) - mid) <= 1.0 for a, b in windows
            ):
                windows.append((gap_s, gap_e))
        prev_singers = curr
        prev_end = float(line.end)  # type: ignore[arg-type]

    windows.sort(key=lambda pair: 0.5 * (pair[0] + pair[1]))
    return windows


def _pick_handoff_cut(
    gap_start: float, gap_end: float, bars: list[float], beats: list[float]
) -> float:
    """Cut inside a singer-handoff gap, snapping to a bar or beat only if it avoids vocal bleed."""
    mid = 0.5 * (gap_start + gap_end)
    bars_in_gap = [b for b in bars if gap_start - 0.05 <= b <= gap_end + 0.05]
    if bars_in_gap:
        best_bar = min(bars_in_gap, key=lambda b: abs(b - mid))
        return round(best_bar, 3)
    beats_in_gap = [bt for bt in beats if gap_start - 0.03 <= bt <= gap_end + 0.03]
    if beats_in_gap:
        best_beat = min(beats_in_gap, key=lambda bt: abs(bt - mid))
        return round(best_beat, 3)
    return round(mid, 3)


def _section_bar_indices(
    analysis: Analysis, bars: list[float], *, exclude_near: list[float] | None = None
) -> set[int]:
    """Bar indices where a new song section begins, snapped to the nearest bar line."""
    if not bars:
        return set()
    bar_arr = np.asarray(bars, dtype=np.float64)
    indices: set[int] = set()
    for start_time, _ in section_timeline(analysis):
        if exclude_near and any(abs(start_time - h) <= 1.2 for h in exclude_near):
            continue
        idx = int(np.argmin(np.abs(bar_arr - start_time)))
        if 0 < idx < len(bars):
            indices.add(idx)
    return indices


def _phrase_points(analysis: Analysis, bars: list[float]) -> list[float]:
    """Candidate cut points at vocal-window and lyric-section boundaries, snapped to near bars."""
    duration = analysis.duration
    raw: set[float] = {0.0, duration}
    for window in analysis.vocal_windows:
        if 0.0 < window.start < duration:
            raw.add(window.start)
        if 0.0 < window.end < duration:
            raw.add(window.end)
    for start_time, _ in section_timeline(analysis):
        if 0.0 < start_time < duration:
            raw.add(start_time)
    if len(bars) < 2:
        return sorted(raw)
    bar_arr = np.asarray(bars, dtype=np.float64)
    snap_tolerance = float(np.median(np.diff(bar_arr))) * 0.5
    snapped: set[float] = {0.0, duration}
    for point in raw:
        if point in (0.0, duration):
            continue
        nearest = float(bar_arr[int(np.argmin(np.abs(bar_arr - point)))])
        chosen = nearest if abs(nearest - point) <= snap_tolerance else point
        if 0.0 < chosen < duration:
            snapped.add(round(chosen, 3))
    return sorted(snapped)


def _validate_pacing_cuts(
    candidate: object,
    duration: float,
    video: VideoConfig,
    analysis: Analysis | None = None,
) -> list[float] | None:
    """Validate plugin cut points are finite, increasing, cover [0, duration], and obey bounds."""
    if not isinstance(candidate, list) or len(candidate) < 2:
        return None
    pts: list[float] = []
    for raw in candidate:
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            return None
        val = float(raw)
        if not math.isfinite(val):
            return None
        pts.append(round(val, 3))
    if abs(pts[0] - 0.0) > 1e-3 or abs(pts[-1] - duration) > 1e-2:
        return None
    pts[0] = 0.0
    pts[-1] = round(duration, 3)
    max_clip = max(0.1, video.max_clip_seconds)
    lyric_spans = (
        _aligned_lyric_spans(analysis)
        if analysis is not None and video.broll_min_clip_seconds is not None
        else []
    )
    fallback_min = (
        video.broll_min_clip_seconds
        if video.broll_min_clip_seconds is not None
        else video.min_clip_seconds
    )
    for prev, nxt in zip(pts[:-1], pts[1:]):
        span = nxt - prev
        raw_min = (
            _min_clip_for_span(analysis, video, prev, nxt, lyric_spans)
            if analysis is not None
            else fallback_min
        )
        min_clip = min(raw_min, duration)
        if span <= 0.0 or span < min_clip - 1e-2 or span > max_clip + 1e-2:
            return None
    return pts


def cut_points(
    analysis: Analysis,
    video: VideoConfig,
    *,
    snap_sections: bool = False,
    pacing_plugin: str | None = None,
) -> list[float]:
    """Shot boundaries from 0 to the song's end, on bar lines or phrase boundaries.

    When ``video.cut_mode`` is ``"phrases"``, cuts follow vocal windows and lyric sections
    (snapping to nearby bar lines). Otherwise cuts step along the bar grid—either at a uniform
    bar interval (``video.dynamic_pacing: false``) or scaled by bar intensity and snapped to
    section boundaries (``video.dynamic_pacing: true``, or when ``snap_sections`` is enabled for
    multi-character songs).
    """
    duration, bars = analysis.duration, analysis.bars
    if not math.isfinite(duration) or duration <= 0.0:
        return [0.0]
    max_clip_seconds = max(0.1, video.max_clip_seconds)
    solo_voices = {t.voice for t in analysis.vocal_turns if t.voice != BOTH_VOICES}
    use_handoffs = snap_sections or video.cut_mode == "phrases" or len(solo_voices) >= 2
    handoff_cuts = (
        [
            cut
            for gs, ge in _singer_handoff_windows(analysis)
            if 0.0 < (cut := _pick_handoff_cut(gs, ge, bars, analysis.beats)) < duration
        ]
        if use_handoffs
        else []
    )
    points = [0.0]
    if video.cut_mode == "phrases":
        points = _phrase_points(analysis, bars)
    elif len(bars) >= 2:
        bar_length = float(np.median(np.diff(bars)))
        allowed_steps = _allowed_bar_steps(bar_length, video)
        broll_steps = (
            _allowed_bar_steps(bar_length, video, min_clip_override=video.broll_min_clip_seconds)
            if video.broll_min_clip_seconds is not None
            else allowed_steps
        )
        lyric_spans = _aligned_lyric_spans(analysis) if broll_steps != allowed_steps else []
        bars_per_shot = allowed_steps[0]

        def _steps_for_bar(bar_idx: int) -> list[int]:
            if broll_steps == allowed_steps:
                return allowed_steps
            b_start = bars[bar_idx]
            b_end = bars[bar_idx + 1] if bar_idx + 1 < len(bars) else duration
            return (
                allowed_steps
                if _is_sung_span(
                    analysis, b_start, b_end, video.lipsync_min_vocal_fraction, lyric_spans
                )
                else broll_steps
            )

        if not video.dynamic_pacing and not snap_sections and not handoff_cuts:
            points += [b for b in bars[::bars_per_shot] if 0.0 < b < duration]
        else:
            intensities = analysis.bar_intensity
            has_contrast = (
                video.dynamic_pacing
                and len(intensities) >= len(bars)
                and (max(intensities[: len(bars)]) - min(intensities[: len(bars)]))
                >= _MIN_INTENSITY_SPREAD
            )
            section_bars = _section_bar_indices(analysis, bars, exclude_near=handoff_cuts)
            idx = 0
            num_bars = len(bars)
            while idx < num_bars:
                bar_time = bars[idx]
                if 0.0 < bar_time < duration and bar_time != points[-1]:
                    points.append(bar_time)

                bar_allowed = _steps_for_bar(idx)
                step = (
                    _step_for_intensity(float(intensities[idx]), bar_allowed)
                    if has_contrast
                    else bar_allowed[0]
                )
                min_for_bar = (
                    video.broll_min_clip_seconds
                    if (
                        video.broll_min_clip_seconds is not None
                        and bar_allowed == broll_steps
                        and broll_steps != allowed_steps
                    )
                    else video.min_clip_seconds
                )

                # Snap early if a new section starts or intensity surges to a faster tier.
                next_idx = idx + step
                for candidate in range(idx + 1, min(num_bars, next_idx)):
                    is_section_start = candidate in section_bars
                    cand_allowed = _steps_for_bar(candidate)
                    is_intensity_surge = (
                        has_contrast
                        and _step_for_intensity(float(intensities[candidate]), cand_allowed) < step
                    )
                    if (is_section_start or is_intensity_surge) and (
                        bars[candidate] - bars[idx] >= min_for_bar
                    ):
                        next_idx = candidate
                        break
                idx = next_idx
    if video.phrase_dp_slicing:
        candidates = sorted(
            {
                round(p, 3)
                for p in [*analysis.bars, *points, *_phrase_points(analysis, bars)]
                if 0.0 <= p <= duration
            }
        )
        dp_points = dp_phrase_cuts(analysis, video, candidates)
        if dp_points:
            points = dp_points

    if handoff_cuts:
        points = sorted({round(p, 3) for p in [*points, *handoff_cuts, 0.0, duration]})
    elif points[-1] != duration:
        points.append(duration)

    protected = {round(c, 3) for c in handoff_cuts}
    merge_lyric_spans = (
        _aligned_lyric_spans(analysis) if video.broll_min_clip_seconds is not None else []
    )

    def _span_min_clip(span_start: float, span_end: float) -> float:
        return _min_clip_for_span(analysis, video, span_start, span_end, merge_lyric_spans)

    # Drop boundaries that would leave a sliver; prefer keeping singer-handoff cuts over bar cuts.
    merged = [points[0]]
    for point in points[1:-1]:
        r_point = round(point, 3)
        if point - merged[-1] >= _span_min_clip(merged[-1], point):
            merged.append(point)
        elif (
            r_point in protected
            and len(merged) > 1
            and round(merged[-1], 3) not in protected
            and point - merged[-2] >= _span_min_clip(merged[-2], point)
        ):
            merged[-1] = point
    if len(merged) > 1 and duration - merged[-1] < _span_min_clip(merged[-1], duration):
        merged.pop()
    merged.append(duration)

    # Split anything too long (merged slivers, very slow songs, no beat grid) into equal parts.
    final = [merged[0]]
    for point in merged[1:]:
        span = point - final[-1]
        parts = max(1, math.ceil(span / max_clip_seconds - 1e-9))
        origin = final[-1]
        final += [origin + span * k / parts for k in range(1, parts + 1)]
    cuts = [round(p, 3) for p in final]
    if video.prefer_vocal_onset and analysis.vocal_windows and not handoff_cuts:
        cuts = shift_to_vocal_onset(cuts, analysis.vocal_windows, video)
    if pacing_plugin:
        try:
            fn = plugins.load_plugin(plugins.PACING_GROUP, pacing_plugin)
            proposed = fn(list(cuts), analysis, video)
            validated = _validate_pacing_cuts(proposed, duration, video, analysis=analysis)
            if validated is not None:
                return validated
            log.warning(
                "Pacing plugin %r returned invalid cut points; falling back to default cuts.",
                pacing_plugin,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "Pacing plugin %r failed (%s); falling back to default cuts.", pacing_plugin, exc
            )
    return cuts


def section_timeline(analysis: Analysis) -> list[tuple[float, str]]:
    """(start time, section label) each time the lyrics move into a new section."""
    timeline: list[tuple[float, str]] = []
    for line in sorted(
        (ln for ln in analysis.lyric_lines if ln.start is not None and ln.section),
        key=lambda ln: ln.start,  # type: ignore[arg-type, return-value]
    ):
        if not timeline or timeline[-1][1] != line.section:
            timeline.append((line.start, line.section))  # type: ignore[arg-type]
    return timeline


def section_at(timeline: list[tuple[float, str]], time: float) -> str | None:
    """The section in progress at ``time``; None before the first sung section."""
    current = None
    for start, label in timeline:
        if start <= time:
            current = label
        else:
            break
    return current


def voice_activity_in_span(turns: list[VocalTurn], start: float, end: float) -> dict[str, float]:
    """Seconds of solo ('voice_1', 'voice_2') and duet ('both') singing in ``[start, end]``."""
    clipped = [
        (max(start, t.start), min(end, t.end), t.voice)
        for t in turns
        if min(end, t.end) > max(start, t.start)
    ]
    if not clipped:
        return {}
    edges = sorted({s for s, _, _ in clipped} | {e for _, e, _ in clipped})
    totals: dict[str, float] = {}
    for a, b in zip(edges[:-1], edges[1:]):
        if b - a <= 1e-9:
            continue
        mid = 0.5 * (a + b)
        active = {v for s, e, v in clipped if s <= mid <= e}
        if not active:
            continue
        voice = BOTH_VOICES if (BOTH_VOICES in active or len(active) >= 2) else next(iter(active))
        totals[voice] = totals.get(voice, 0.0) + (b - a)
    return totals


def dominant_voice(turns: list[VocalTurn], start: float, end: float) -> str | None:
    """Dominant vocal state ('voice_1', 'voice_2', or 'both') inside ``[start, end]``."""
    totals = voice_activity_in_span(turns, start, end)
    if not totals:
        return None
    best_voice, best_seconds = max(totals.items(), key=lambda pair: pair[1])
    return best_voice if best_seconds >= 0.05 else None


def _section_between(
    analysis: Analysis, timeline: list[tuple[float, str]], start: float, end: float
) -> str | None:
    """Section of lyric lines sung in ``[start, end]``, else the timeline section at midpoint."""
    totals: dict[str, float] = {}
    for line in analysis.lyric_lines:
        if line.section and line.start is not None and line.end is not None:
            shared = TimeSpan(start=line.start, end=line.end).overlap(start, end)
            if shared > 0:
                totals[line.section] = totals.get(line.section, 0.0) + shared
    if totals:
        return max(totals.items(), key=lambda pair: pair[1])[0]
    return section_at(timeline, (start + end) / 2)


def segment(
    analysis: Analysis,
    video: VideoConfig,
    *,
    snap_sections: bool = False,
    pacing_plugin: str | None = None,
) -> list[Segment]:
    points = cut_points(analysis, video, snap_sections=snap_sections, pacing_plugin=pacing_plugin)
    timeline = section_timeline(analysis)
    lyric_spans = _aligned_lyric_spans(analysis)
    segments = []
    for start, end in zip(points[:-1], points[1:]):
        if end <= start:
            continue
        is_lipsync = _is_sung_span(
            analysis, start, end, video.lipsync_min_vocal_fraction, lyric_spans
        )
        segments.append(
            Segment(
                start=start,
                end=end,
                type=ShotType.LIPSYNC if is_lipsync else ShotType.PERFORMANCE,
                lyrics=_lyrics_between(analysis, start, end),
                intensity=_intensity_between(analysis, start, end),
                section=_section_between(analysis, timeline, start, end),
                voice=dominant_voice(analysis.vocal_turns, start, end),
            )
        )
    return segments


def _lyrics_between(analysis: Analysis, start: float, end: float) -> str:
    """Lines whose midpoint falls inside the shot, so each line belongs to one shot."""
    lines = [
        line.text
        for line in analysis.lyric_lines
        if line.start is not None
        and line.end is not None
        and start <= (line.start + line.end) / 2 < end
    ]
    return " / ".join(lines)


def _intensity_between(analysis: Analysis, start: float, end: float) -> float:
    edges = [*analysis.bars, analysis.duration]
    weighted, total = 0.0, 0.0
    for bar_start, bar_end, value in zip(edges[:-1], edges[1:], analysis.bar_intensity):
        shared = max(0.0, min(bar_end, end) - max(bar_start, start))
        weighted += shared * value
        total += shared
    return round(weighted / total, 3) if total > 0 else 0.5
