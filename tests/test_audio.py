# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np

from mvdirector.audio import analysis as audio
from mvdirector.audio import lyrics
from mvdirector.models import LyricLine, TimeSpan
from tests.conftest import BPM, SR, click_track, vocal_track


def test_beat_tracking_finds_tempo_of_click_track():
    tempo, beats = audio.detect_beats(click_track(), SR)
    assert abs(tempo - BPM) < 4
    assert np.allclose(np.diff(beats), 0.5, atol=0.05)


def test_bars_take_every_nth_beat():
    assert audio.bars_from_beats([0.0, 0.5, 1.0, 1.5, 2.0, 2.5], 4) == [0.0, 2.0]


def test_vocal_windows_bridge_breaths_and_drop_blips():
    # Singing 1-3 s and 3.3-5 s (a breath between), plus a 0.1 s blip of bleed at 8 s.
    samples = vocal_track([(1.0, 3.0), (3.3, 5.0), (8.0, 8.1)], seconds=10.0)
    rms, times = audio.rms_envelope(samples, SR)
    windows = audio.detect_vocal_windows(rms, times)
    assert len(windows) == 1
    assert abs(windows[0].start - 1.0) < 0.05 and abs(windows[0].end - 5.0) < 0.05


def test_vocal_windows_empty_for_silence():
    rms, times = audio.rms_envelope(np.zeros(SR * 2, dtype=np.float32), SR)
    assert audio.detect_vocal_windows(rms, times) == []


def test_bar_intensity_is_scaled_to_loudest_bar():
    times = np.arange(0.05, 4.0, 0.1)
    rms = np.where(times < 2.0, 0.1, 0.4)
    assert audio.bar_intensity(rms, times, [0.0, 2.0], 4.0) == [0.25, 1.0]


def test_parse_lyrics_reads_section_tags():
    lines = lyrics.parse_lyrics("[Verse 1]\nhello there\n\n[Chorus]\nsing it loud\n")
    assert [(ln.text, ln.section) for ln in lines] == [
        ("hello there", "Verse 1"),
        ("sing it loud", "Chorus"),
    ]


def test_alignment_tolerates_transcription_errors():
    lines = lyrics.parse_lyrics("walking through the neon rain\nI can hear you call my name")
    heard = [
        "walking",
        "through",
        "the",
        "neon",
        "reign",
        "I",
        "can",
        "here",
        "you",
        "call",
        "my",
        "name",
    ]
    words = [lyrics.Word(word=w, start=float(i), end=i + 0.8) for i, w in enumerate(heard)]
    aligned = lyrics.align_lines_to_words(lines, words)
    assert (aligned[0].start, aligned[0].end) == (0.0, 3.8)
    assert (aligned[1].start, aligned[1].end) == (5.0, 11.8)
    assert all(ln.timing_source == "aligned" for ln in aligned)


def test_estimate_spreads_lines_over_vocal_windows_only():
    lines = [LyricLine(text="aaaa"), LyricLine(text="bbbb")]
    windows = [TimeSpan(start=10.0, end=12.0), TimeSpan(start=20.0, end=22.0)]
    timed = lyrics.estimate_missing_timings(lines, windows)
    assert (timed[0].start, timed[0].end) == (10.0, 12.0)
    assert (timed[1].start, timed[1].end) == (20.0, 22.0)
    assert timed[0].timing_source == "estimated"


def test_estimate_fills_gap_between_aligned_neighbours():
    lines = [
        LyricLine(text="one", start=0.0, end=2.0, timing_source="aligned"),
        LyricLine(text="two"),
        LyricLine(text="three", start=6.0, end=8.0, timing_source="aligned"),
    ]
    timed = lyrics.estimate_missing_timings(lines, [TimeSpan(start=0.0, end=8.0)])
    assert (timed[1].start, timed[1].end, timed[1].timing_source) == (2.0, 6.0, "estimated")


def test_parenthesised_adlibs_are_lyrics_not_section_tags():
    text = (
        "[Intro \u2013 whispered adlib over filtered synths]\n"
        "(Knight moves~)\n"
        "[Post-Chorus \u2013 clipped chant, finger snaps]\n"
        "Step, step, knight moves,\n"
        "(Step, step, hey!)\n"
    )
    lines = lyrics.parse_lyrics(text)
    assert [(ln.text, ln.section) for ln in lines] == [
        ("(Knight moves~)", "Intro \u2013 whispered adlib over filtered synths"),
        ("Step, step, knight moves,", "Post-Chorus \u2013 clipped chant, finger snaps"),
        ("(Step, step, hey!)", "Post-Chorus \u2013 clipped chant, finger snaps"),
    ]


def test_adlib_lines_align_to_transcribed_words():
    lines = lyrics.parse_lyrics("(Knight moves~)\nStep, step, knight moves,")
    heard = ["Knight", "moves", "Step", "step", "knight", "moves"]
    words = [lyrics.Word(word=w, start=float(i), end=i + 0.8) for i, w in enumerate(heard)]
    aligned = lyrics.align_lines_to_words(lines, words)
    assert (aligned[0].start, aligned[0].end) == (0.0, 1.8)
    assert aligned[1].start == 2.0
