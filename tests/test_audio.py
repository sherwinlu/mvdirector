# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np
import pytest

from mvdirector.audio import analysis as audio
from mvdirector.audio import lyrics
from mvdirector.models import LyricLine, TimeSpan
from tests.conftest import BPM, SR, click_track, vocal_track


def test_beat_tracking_finds_tempo_of_click_track():
    tempo, beats = audio.detect_beats(click_track(), SR)
    assert abs(tempo - BPM) < 4
    assert np.allclose(np.diff(beats), 0.5, atol=0.05)


def test_double_time_tempo_is_folded_to_half_time():
    beats = [i * 0.325 for i in range(10)]  # ~185 BPM
    tempo, folded = audio.fold_tempo(185.0, beats)
    assert tempo == 92.5 and folded == beats[::2]


def test_half_time_tempo_is_folded_to_double_time():
    tempo, folded = audio.fold_tempo(50.0, [0.0, 1.2, 2.4])
    assert tempo == 100.0 and folded == pytest.approx([0.0, 0.6, 1.2, 1.8, 2.4])


def test_tempo_in_range_and_degenerate_beats_are_untouched():
    assert audio.fold_tempo(120.0, [0.0, 0.5, 1.0]) == (120.0, [0.0, 0.5, 1.0])
    assert audio.fold_tempo(200.0, [0.0]) == (200.0, [0.0])
    assert audio.fold_tempo(0.0, []) == (0.0, [])
    assert audio.fold_tempo(float("inf"), [0.0, 0.5]) == (float("inf"), [0.0, 0.5])
    # Near-zero tempo must not blow up memory via unbounded doubling.
    tempo, folded = audio.fold_tempo(1e-6, [0.0, 1.0])
    assert tempo == pytest.approx(1.6e-5) and len(folded) == 17


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


def _harmonic_voice(t: np.ndarray, f0_hz: float, rolloff: float) -> np.ndarray:
    """Synthetic singing voice with fundamental + harmonics and distinct spectral tilt."""
    sig = np.zeros_like(t, dtype=np.float64)
    for k in range(1, 7):
        sig += (k ** (-rolloff)) * np.sin(2.0 * np.pi * (f0_hz * k) * t)
    peak = max(float(np.max(np.abs(sig))), 1e-9)
    return (0.45 * sig / peak).astype(np.float32)


def test_detect_vocal_turns_separates_male_and_female_including_zero_gap_handoff():
    seconds = 16.0
    t = np.arange(int(seconds * SR), dtype=np.float64) / SR
    samples = np.zeros_like(t, dtype=np.float32)
    # Male (~130 Hz): 1.0-4.0s and 8.0-11.5s; Female (~280 Hz): 4.5-8.0s (zero gap into 8.0s!)
    # and 12.0-15.0s.
    m1 = (t >= 1.0) & (t < 4.0)
    f1 = (t >= 4.5) & (t < 8.0)
    m2 = (t >= 8.0) & (t < 11.5)
    f2 = (t >= 12.0) & (t < 15.0)
    samples[m1] = _harmonic_voice(t[m1], 130.0, rolloff=1.6)
    samples[f1] = _harmonic_voice(t[f1], 280.0, rolloff=0.9)
    samples[m2] = _harmonic_voice(t[m2], 135.0, rolloff=1.6)
    samples[f2] = _harmonic_voice(t[f2], 275.0, rolloff=0.9)

    turns = audio.detect_vocal_turns(samples, SR)
    assert [turn.voice for turn in turns] == ["voice_1", "voice_2", "voice_1", "voice_2"]
    assert turns[0].median_f0_hz == pytest.approx(130.0, abs=10.0)
    assert turns[1].median_f0_hz == pytest.approx(280.0, abs=12.0)
    # Zero-gap handoff at 8.0s is split cleanly rather than merged into one turn.
    assert turns[1].end == pytest.approx(8.0, abs=0.25)
    assert turns[2].start == pytest.approx(8.0, abs=0.25)

    # Solo voice with small natural pitch movement is never falsely split into two singers.
    solo = np.zeros_like(t, dtype=np.float32)
    solo[(t >= 1.0) & (t < 5.0)] = _harmonic_voice(t[(t >= 1.0) & (t < 5.0)], 210.0, rolloff=1.3)
    solo[(t >= 6.0) & (t < 11.0)] = _harmonic_voice(t[(t >= 6.0) & (t < 11.0)], 225.0, rolloff=1.3)
    assert audio.detect_vocal_turns(solo, SR) == []


def test_alignment_drops_stray_single_stopword_across_singer_gap():
    lines = lyrics.parse_lyrics("forgive the falling snow\nand the night goes cold")
    # Whisper missed "snow" on line 1, and line 2 starts 6s later with "the".
    words = [
        lyrics.Word(word="forgive", start=1.0, end=1.5),
        lyrics.Word(word="the", start=1.5, end=1.7),
        lyrics.Word(word="falling", start=1.7, end=2.3),
        lyrics.Word(word="and", start=8.0, end=8.3),
        lyrics.Word(word="the", start=8.3, end=8.5),
        lyrics.Word(word="night", start=8.5, end=9.0),
        lyrics.Word(word="goes", start=9.0, end=9.4),
        lyrics.Word(word="cold", start=9.4, end=10.0),
    ]
    aligned = lyrics.align_lines_to_words(lines, words)
    assert aligned[0].end == pytest.approx(2.3)
    assert aligned[1].start == pytest.approx(8.0)


def test_format_transcript_tags_duet_sections_from_vocal_turns():
    from mvdirector.models import VocalTurn

    segs = [
        lyrics.TranscriptSegment(text="Forgive the snow", start=1.0, end=3.5),
        lyrics.TranscriptSegment(text="You were late", start=4.0, end=6.5),
        lyrics.TranscriptSegment(text="We fall together", start=7.0, end=10.0),
    ]
    turns = [
        VocalTurn(start=1.0, end=3.6, voice="voice_1", median_f0_hz=130.0),
        VocalTurn(start=3.9, end=6.6, voice="voice_2", median_f0_hz=275.0),
        VocalTurn(start=6.9, end=10.1, voice="both", median_f0_hz=None),
    ]
    formatted = lyrics.format_transcript_as_lyrics(segs, pause_threshold=3.0, vocal_turns=turns)
    assert "[Section 1: Singer 1]\nForgive the snow" in formatted
    assert "[Section 2: Singer 2]\nYou were late" in formatted
    assert "[Section 3: Both]\nWe fall together" in formatted


def test_detect_vocal_turns_identifies_harmony_and_octave_unison_as_both():
    seconds = 20.5
    t = np.arange(int(seconds * SR), dtype=np.float64) / SR
    samples = np.zeros_like(t, dtype=np.float32)

    m_solo_1 = (t >= 0.5) & (t < 4.0)
    f_solo = (t >= 4.5) & (t < 8.0)
    harmony = (t >= 8.5) & (t < 12.5)
    m_solo_2 = (t >= 12.5) & (t < 16.0)  # zero-gap transition out of harmony into solo!
    unison = (t >= 16.5) & (t < 19.8)

    samples[m_solo_1] = _harmonic_voice(t[m_solo_1], 130.0, rolloff=1.7)
    samples[f_solo] = _harmonic_voice(t[f_solo], 280.0, rolloff=0.8)
    samples[harmony] = 0.65 * (
        _harmonic_voice(t[harmony], 130.0, rolloff=1.7)
        + _harmonic_voice(t[harmony], 330.0, rolloff=0.8)
    )
    samples[m_solo_2] = _harmonic_voice(t[m_solo_2], 135.0, rolloff=1.7)
    samples[unison] = 0.65 * (
        _harmonic_voice(t[unison], 130.0, rolloff=1.7)
        + _harmonic_voice(t[unison], 260.0, rolloff=0.8)
    )

    turns = audio.detect_vocal_turns(samples, SR)
    assert [turn.voice for turn in turns] == ["voice_1", "voice_2", "both", "voice_1", "both"]
    assert turns[2].median_f0_hz is None
    assert turns[4].median_f0_hz is None
    assert turns[2].end == pytest.approx(12.5, abs=0.3)
    assert turns[3].start == pytest.approx(12.5, abs=0.3)


def test_detect_vocal_turns_handles_non_finite_and_low_sample_rate_cleanly():
    poisoned_nan = np.array([np.nan] * 4096 + [0.4] * 4096, dtype=np.float32)
    poisoned_inf = np.array([np.inf] * 4096 + [0.4] * 4096, dtype=np.float32)
    low_sr = np.sin(2.0 * np.pi * 200.0 * np.arange(32000) / 8000.0).astype(np.float32)
    assert audio.detect_vocal_turns(poisoned_nan, SR) == []
    assert audio.detect_vocal_turns(poisoned_inf, SR) == []
    assert audio.detect_vocal_turns(low_sr, 8000) == []
