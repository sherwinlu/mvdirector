# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Lyric parsing and timing.

Timing comes from the best available source, in this order:

1. Word timestamps (a JSON file you supply, or faster-whisper if installed) matched
   against the known lyrics, so transcription mistakes do not change the text.
2. An estimate that spreads lines across detected vocal activity by length.
"""

from __future__ import annotations

import json
import logging
import re
from difflib import SequenceMatcher

import librosa
from pydantic import BaseModel

from mvdirector.models import LyricLine, TimeSpan
from mvdirector.project import Project

log = logging.getLogger(__name__)

# Only square brackets mark a section. Parenthesised lines are sung adlibs and stay lyrics.
_SECTION_TAG = re.compile(r"^\s*\[([^\]]+)\]\s*$")
_TOKEN = re.compile(r"[a-z0-9']+")
WHISPER_SAMPLE_RATE = 16000


class Word(BaseModel):
    word: str
    start: float
    end: float


def parse_lyrics(text: str) -> list[LyricLine]:
    """Split lyrics into lines, turning ``[Chorus]``-style tags into section labels.

    Lines in parentheses are kept as lyrics (adlibs are sung and need timing too).
    """
    lines: list[LyricLine] = []
    section: str | None = None
    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped:
            continue
        tag = _SECTION_TAG.match(stripped)
        if tag:
            section = tag.group(1).strip()
            continue
        lines.append(LyricLine(text=stripped, section=section))
    return lines


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def load_or_transcribe_words(project: Project) -> list[Word]:
    """Word timestamps from the configured JSON file, else faster-whisper, else none."""
    inputs = project.config.inputs
    if inputs.word_timestamps:
        raw = json.loads(project.input_path(inputs.word_timestamps, "word_timestamps").read_text())
        return [Word.model_validate(item) for item in raw]
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        log.warning(
            "faster-whisper is not installed; lyric timings will be estimated from vocal "
            "activity. Install the 'align' extra for word-level alignment."
        )
        return []
    log.info("Transcribing vocals for word timestamps (first run downloads the model)")
    try:
        # Decode the audio ourselves and pass samples in: faster-whisper's own decoder
        # depends on PyAV, whose version often disagrees with faster-whisper's.
        samples, _ = librosa.load(
            str(project.input_path(inputs.vocals, "vocals")), sr=WHISPER_SAMPLE_RATE, mono=True
        )
        model = WhisperModel("small", compute_type="int8")
        segments, _ = model.transcribe(samples, word_timestamps=True)
        return [
            Word(word=w.word, start=w.start, end=w.end)
            for seg in segments
            for w in (seg.words or [])
        ]
    except Exception as exc:  # noqa: BLE001 - alignment is an enhancement; never block analysis
        log.warning(
            "Transcription failed (%s: %s); lyric timings will be estimated from vocal activity.",
            type(exc).__name__,
            exc,
        )
        return []


def align_lines_to_words(lines: list[LyricLine], words: list[Word]) -> list[LyricLine]:
    """Give each lyric line the time span of the transcribed words it matches.

    Lines with no matching words are left untimed for ``estimate_missing_timings``.
    """
    lyric_tokens: list[str] = []
    token_line: list[int] = []
    for index, line in enumerate(lines):
        for token in tokenize(line.text):
            lyric_tokens.append(token)
            token_line.append(index)

    word_tokens: list[str] = []
    token_word: list[int] = []
    for index, word in enumerate(words):
        for token in tokenize(word.word):
            word_tokens.append(token)
            token_word.append(index)

    spans: dict[int, list[float]] = {}
    matcher = SequenceMatcher(a=lyric_tokens, b=word_tokens, autojunk=False)
    for block in matcher.get_matching_blocks():
        for offset in range(block.size):
            line_index = token_line[block.a + offset]
            word = words[token_word[block.b + offset]]
            span = spans.setdefault(line_index, [word.start, word.end])
            span[0], span[1] = min(span[0], word.start), max(span[1], word.end)

    aligned = []
    for index, line in enumerate(lines):
        if index in spans:
            start, end = spans[index]
            aligned.append(
                line.model_copy(
                    update={
                        "start": round(start, 3),
                        "end": round(end, 3),
                        "timing_source": "aligned",
                    }
                )
            )
        else:
            aligned.append(line)
    log.info("Aligned %d of %d lyric lines to word timestamps", len(spans), len(lines))
    return aligned


def estimate_missing_timings(lines: list[LyricLine], windows: list[TimeSpan]) -> list[LyricLine]:
    """Fill in untimed lines.

    With no aligned lines at all, lines are spread across vocal activity in proportion
    to their length. Otherwise each run of untimed lines shares the gap between its
    timed neighbours.
    """
    if not lines or all(line.timing_source == "aligned" for line in lines):
        return lines
    if not any(line.timing_source == "aligned" for line in lines):
        return _spread_over_windows(lines, windows)

    result = list(lines)
    song_end = max([w.end for w in windows] + [ln.end for ln in lines if ln.end is not None])
    index = 0
    while index < len(result):
        if result[index].start is not None:
            index += 1
            continue
        run_end = index
        while run_end < len(result) and result[run_end].start is None:
            run_end += 1
        gap_start = result[index - 1].end if index > 0 else 0.0
        gap_end = result[run_end].start if run_end < len(result) else song_end
        gap_start = gap_start or 0.0
        gap_end = max(gap_end if gap_end is not None else song_end, gap_start)
        weights = [max(1, len(result[i].text)) for i in range(index, run_end)]
        cursor = gap_start
        for i, weight in zip(range(index, run_end), weights):
            share = (gap_end - gap_start) * weight / sum(weights)
            result[i] = result[i].model_copy(
                update={
                    "start": round(cursor, 3),
                    "end": round(cursor + share, 3),
                    "timing_source": "estimated",
                }
            )
            cursor += share
        index = run_end
    return result


def _spread_over_windows(lines: list[LyricLine], windows: list[TimeSpan]) -> list[LyricLine]:
    total = sum(w.duration for w in windows)
    if total <= 0:
        return lines

    def to_absolute(vocal_time: float, at_end: bool) -> float:
        """Map seconds of cumulative singing back to a position in the song."""
        elapsed = 0.0
        for window in windows:
            reached = elapsed + window.duration
            if vocal_time < reached or (at_end and vocal_time <= reached + 1e-9):
                return window.start + (vocal_time - elapsed)
            elapsed = reached
        return windows[-1].end

    weights = [max(1, len(line.text)) for line in lines]
    result, cursor = [], 0.0
    for line, weight in zip(lines, weights):
        share = total * weight / sum(weights)
        result.append(
            line.model_copy(
                update={
                    "start": round(to_absolute(cursor, at_end=False), 3),
                    "end": round(to_absolute(cursor + share, at_end=True), 3),
                    "timing_source": "estimated",
                }
            )
        )
        cursor += share
    return result
