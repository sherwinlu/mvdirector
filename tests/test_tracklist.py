# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Tests for tracklist/cue ingestion, WeaveShift normalization, and YouTube chapter export."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mvdirector.audio.tracklist import (
    format_ffmetadata_chapters,
    format_youtube_chapters,
    load_tracklist,
    parse_timestamp,
    resolve_chapters,
)
from mvdirector.errors import ConfigError
from mvdirector.models import MixChapter


def test_parse_timestamp_formats_and_validation() -> None:
    assert parse_timestamp(0) == 0.0
    assert parse_timestamp(185.25) == 185.25
    assert parse_timestamp("03:05.5") == 185.5
    assert parse_timestamp("75:30") == 4530.0  # DJ mixes often write MM >= 60
    assert parse_timestamp("01:02:03.5") == 3723.5

    for invalid in (True, -1.0, float("nan"), float("inf"), "", "03:60", "01:65:00", "abc"):
        with pytest.raises(ConfigError):
            parse_timestamp(invalid)  # type: ignore[arg-type]


def test_load_weaveshift_json_and_yaml_aliases(tmp_path: Path) -> None:
    payload = {
        "mix_title": "Late Night Drive",
        "dj": "DJ Sol",
        "source": "weaveshift",
        "segments": [
            {
                "cue_in": "00:00",
                "track_title": " Subterranean Echo ",
                "performer": "Klang",
                "tempo_bpm": 124.0,
                "camelot": "8A",
                "intensity": 0.42,
                "mood": "hypnotic, deep",
                "genre": ["dub techno"],
                "visual_prompt": "Misty concrete vault",
                "lyric_excerpt": "Neon rain falls across the glass cathedral",
                "lufs": -9.4,
                "transition": {"type": "eq_swap", "bars": 8},
            },
            {
                "start_time": 252.5,
                "name": "Solar Flare",
                "tempo": 128,
                "vibe": ["kinetic", "bright"],
            },
        ],
    }
    json_path = tmp_path / "setlist.json"
    json_path.write_text(json.dumps(payload), encoding="utf-8")
    spec = load_tracklist(json_path)

    assert spec.title == "Late Night Drive"
    assert spec.artist == "DJ Sol"
    assert len(spec.tracks) == 2
    t0, t1 = spec.tracks
    assert t0.start == 0.0 and t0.title == "Subterranean Echo" and t0.artist == "Klang"
    assert t0.bpm == 124.0 and t0.key == "8A" and t0.energy == 0.42
    assert t0.mood_tags == ["hypnotic", "deep"]
    assert t0.genre_tags == ["dub techno"]
    assert t0.theme_prompt == "Misty concrete vault"
    assert t0.lyrics == "Neon rain falls across the glass cathedral"
    assert t1.start == 252.5 and t1.bpm == 128.0 and t1.mood_tags == ["kinetic", "bright"]


def test_load_cue_sheet_supports_60plus_minutes_and_rem_metadata(tmp_path: Path) -> None:
    cue_text = """REM GENRE "Progressive House"
PERFORMER "Channel One"
TITLE "90 Minute Workout Mix"
FILE "mix.mp3" MP3
  TRACK 01 AUDIO
    TITLE "Warmup Horizon"
    PERFORMER "Artist A"
    REM BPM 122
    REM MOOD "warm, sunrise"
    INDEX 01 00:00:00
  TRACK 02 AUDIO
    TITLE "Peak Engine"
    PERFORMER "Artist B"
    REM BPM 130
    INDEX 01 42:10:30
  TRACK 03 AUDIO
    TITLE "Cooldown Mist"
    INDEX 01 75:00:00
"""
    cue_path = tmp_path / "mix.cue"
    cue_path.write_text(cue_text, encoding="utf-8")
    spec = load_tracklist(cue_path)

    assert spec.title == "90 Minute Workout Mix"
    assert spec.artist == "Channel One"
    assert [t.start for t in spec.tracks] == [0.0, 2530.4, 4500.0]
    assert spec.tracks[0].bpm == 122.0
    assert spec.tracks[0].mood_tags == ["warm", "sunrise"]
    assert spec.tracks[0].genre_tags == ["Progressive House"]
    assert spec.tracks[1].artist == "Artist B"


def test_load_txt_tracklist_parses_bpm_hashtags_and_artists(tmp_path: Path) -> None:
    txt = """# Tracklist export
00:00 Aether - First Light (120 BPM) #ambient #calm
04:30 [126 bpm] Kinetic Tide #driving
1:02:15 Final Drift
"""
    txt_path = tmp_path / "tracks.txt"
    txt_path.write_text(txt, encoding="utf-8")
    spec = load_tracklist(txt_path)

    assert len(spec.tracks) == 3
    assert spec.tracks[0].artist == "Aether"
    assert spec.tracks[0].title == "First Light"
    assert spec.tracks[0].bpm == 120.0
    assert spec.tracks[0].mood_tags == ["ambient", "calm"]
    assert spec.tracks[1].start == 270.0 and spec.tracks[1].bpm == 126.0
    assert spec.tracks[2].start == 3735.0 and spec.tracks[2].title == "Final Drift"


def test_resolve_chapters_handles_fixed_intervals_lead_ins_and_errors(tmp_path: Path) -> None:
    # 1. Fixed intervals (60 minutes -> 15 x 4-minute chapters, snapped to nearby candidates).
    auto = resolve_chapters(
        None,
        3600.0,
        default_segment_seconds=240.0,
        snap_candidates=[235.5, 482.0],
    )
    assert len(auto) == 15
    assert auto[0].start == 0.0 and auto[0].end == 235.5
    assert auto[1].start == 235.5 and auto[1].end == 482.0
    assert auto[-1].end == 3600.0
    assert auto[0].id == "ch01" and auto[-1].id == "ch15"

    # 2. Lead-in >= 10s inserts an automatic Intro chapter so [0, duration] has no gap.
    late_path = tmp_path / "late.txt"
    late_path.write_text("00:18 Track One\n03:00 Track Two\n", encoding="utf-8")
    resolved_late = resolve_chapters(load_tracklist(late_path), 360.0)
    assert [(c.title, c.start, c.end) for c in resolved_late] == [
        ("Intro", 0.0, 18.0),
        ("Track One", 18.0, 180.0),
        ("Track Two", 180.0, 360.0),
    ]

    # 3. Short lead-in (< 10s) clamps Track 1 to 0.0s instead of creating a micro-chapter.
    tiny_lead = tmp_path / "tiny.txt"
    tiny_lead.write_text("00:02 Track One\n03:00 Track Two\n", encoding="utf-8")
    resolved_tiny = resolve_chapters(load_tracklist(tiny_lead), 360.0)
    assert resolved_tiny[0].start == 0.0 and resolved_tiny[0].title == "Track One"

    # 4. Out-of-order or past-duration timestamps fail strictly.
    bad_order = tmp_path / "bad_order.txt"
    bad_order.write_text("02:00 First\n01:30 Second\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="strictly increase"):
        resolve_chapters(load_tracklist(bad_order), 300.0)

    past_end = tmp_path / "past_end.txt"
    past_end.write_text("00:00 First\n06:00 Past End\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="mix duration"):
        resolve_chapters(load_tracklist(past_end), 300.0)

    # 5. Overlapping explicit end time (> next track start) fails strictly.
    overlap_json = tmp_path / "overlap.json"
    overlap_json.write_text(
        json.dumps(
            {
                "tracks": [
                    {"start": 0.0, "end": 180.0, "title": "First"},
                    {"start": 120.0, "end": 240.0, "title": "Overlapping Second"},
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="overlaps next track"):
        resolve_chapters(load_tracklist(overlap_json), 300.0)


def test_stress_120_chapter_tracklist_and_adversarial_ffmetadata(tmp_path: Path) -> None:
    # 120-track 8-hour setlist stress test -> 3-digit zero-padded chapter IDs (ch001..ch120).
    stress_txt = tmp_path / "festival_120.txt"
    lines = [
        f"{(idx * 240) // 3600}:{((idx * 240) % 3600) // 60:02d}:00 "
        f"Artist {idx + 1}; = # \\ [CHAPTER] - Track {idx + 1} (128 BPM) #techno"
        for idx in range(120)
    ]
    stress_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    spec = load_tracklist(stress_txt)
    resolved = resolve_chapters(spec, duration=120 * 240.0)
    assert len(resolved) == 120
    assert resolved[0].id == "ch001"
    assert resolved[99].id == "ch100"
    assert resolved[-1].id == "ch120"
    assert resolved[-1].end == 28800.0

    chapters = [
        MixChapter(
            id=r.id,
            index=r.index,
            start=r.start,
            end=r.end,
            title=r.title,
            artist=r.artist,
            bpm=r.bpm or 128.0,
        )
        for r in resolved
    ]
    yt_text, warnings = format_youtube_chapters(chapters, total_duration=28800.0)
    assert warnings == []
    assert len(yt_text.splitlines()) == 120

    # Adversarial metadata with newlines/delimiters must never inject extra [CHAPTER] headers.
    ffmeta = format_ffmetadata_chapters(chapters, title="Evil\n[CHAPTER]\nSTART=0\n;=#\\Mix")
    assert ffmeta.splitlines().count("[CHAPTER]") == 120


def test_youtube_and_ffmetadata_chapter_formatting() -> None:
    chapters = [
        MixChapter(
            id="ch01", index=0, start=0.0, end=1200.0, title="First", artist="DJ A", bpm=120
        ),
        MixChapter(
            id="ch02",
            index=1,
            start=1200.0,
            end=2600.0,
            title="DJ B - Second; Cut=1",
            artist="DJ B",
            bpm=125,
        ),
        MixChapter(id="ch03", index=2, start=2600.0, end=3900.0, title="Third #Peak", bpm=128),
    ]
    yt_text, warnings = format_youtube_chapters(chapters, total_duration=3900.0)
    assert warnings == []
    assert yt_text.splitlines() == [
        "0:00:00 DJ A - First",
        "0:20:00 DJ B - Second; Cut=1",
        "0:43:20 Third #Peak",
    ]

    # Under 1 hour uses MM:SS and warns if < 3 chapters.
    short_yt, short_warn = format_youtube_chapters(chapters[:2], total_duration=2600.0)
    assert len(short_warn) == 1 and "at least 3 chapters" in short_warn[0]
    assert short_yt.splitlines()[0] == "00:00 DJ A - First"
    assert short_yt.splitlines()[1] == "20:00 DJ B - Second; Cut=1"

    ffmeta = format_ffmetadata_chapters(chapters, title="Set=1; #Live")
    assert ";FFMETADATA1" in ffmeta
    assert "title=Set\\=1\\; \\#Live" in ffmeta
    assert "START=1200000\nEND=2600000\ntitle=DJ B - Second\\; Cut\\=1" in ffmeta
