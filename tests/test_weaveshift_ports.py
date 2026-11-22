# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Tests for features ported from WeaveShift: cues, report/CSV tracklists, DSP, and hardening."""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import yaml

from mvdirector import api, media
from mvdirector.audio import analysis as audio
from mvdirector.audio.cues import detect_audio_cues, format_cues_summary
from mvdirector.audio.stems import DemucsSeparator
from mvdirector.audio.tracklist import load_tracklist, resolve_chapters
from mvdirector.config import ProjectConfig
from mvdirector.errors import ConfigError, MediaToolError
from mvdirector.models import (
    Analysis,
    AnalyzedChapter,
    AudioCues,
    ProjectMode,
)
from mvdirector.planning.mix_planner import _chapter_sample_starts, build_mix_manifest
from mvdirector.project import Project
from tests.conftest import SR, click_track

# --------------------------------------------------------------------------- structural cues


def _synthetic_structured_track(sr: int = SR) -> np.ndarray:
    """120 s synthetic track: quiet intro -> peak 1 -> breakdown valley -> peak 2 -> outro."""
    duration = 120.0
    n = int(duration * sr)
    t = np.arange(n, dtype=np.float64) / sr
    rng = np.random.default_rng(7)

    # Low ambient pad baseline across the whole track.
    sig = 0.04 * np.sin(2.0 * np.pi * 110.0 * t)

    def _add_peak(start_s: float, end_s: float, amp: float) -> None:
        mask = (t >= start_s) & (t < end_s)
        tt = t[mask]
        # Brighter multi-harmonic synth + rhythmic percussive pulses every 0.5 s.
        pulse = (np.mod(tt, 0.5) < 0.035).astype(np.float64)
        burst = amp * (
            0.5 * np.sin(2.0 * np.pi * 440.0 * tt) + 0.5 * np.sin(2.0 * np.pi * 1760.0 * tt)
        ) + amp * 0.8 * pulse * rng.standard_normal(tt.size)
        sig[mask] += burst

    # First sustained lift + peak 1 (24..54s), breakdown valley (56..82s), peak 2 (84..108s).
    _add_peak(24.0, 54.0, amp=0.55)
    sig[(t >= 56.0) & (t < 82.0)] *= 0.25
    _add_peak(84.0, 108.0, amp=0.45)

    return np.clip(sig, -1.0, 1.0).astype(np.float32)


def test_detect_audio_cues_finds_lift_peaks_and_breakdown() -> None:
    sig = _synthetic_structured_track()
    cues = detect_audio_cues(sig, SR)

    assert cues.first_lift is not None and 18.0 <= cues.first_lift <= 32.0
    assert cues.main_peak is not None and 22.0 <= cues.main_peak <= 56.0
    assert cues.second_peak is not None and 80.0 <= cues.second_peak <= 112.0
    assert abs(cues.second_peak - cues.main_peak) >= 35.0
    assert cues.breakdown is not None and 54.0 <= cues.breakdown <= 84.0
    assert cues.confidence in ("medium", "high")

    summary = format_cues_summary(cues)
    assert "first_lift=" in summary and "main_peak=" in summary and "breakdown=" in summary

    shifted = detect_audio_cues(sig, SR, offset=100.0)
    assert shifted.main_peak == pytest.approx(cues.main_peak + 100.0, abs=0.01)
    assert shifted.breakdown == pytest.approx(cues.breakdown + 100.0, abs=0.01)


def test_detect_audio_cues_handles_flat_silent_and_degenerate_inputs() -> None:
    # Constant-amplitude sine has no meaningful energy lift.
    t = np.arange(int(20.0 * SR), dtype=np.float32) / SR
    flat = (0.3 * np.sin(2.0 * np.pi * 220.0 * t)).astype(np.float32)
    flat_cues = detect_audio_cues(flat, SR)
    assert flat_cues.first_lift is None

    for bad in (
        np.zeros(0, dtype=np.float32),
        np.zeros(SR * 5, dtype=np.float32),
        np.ones(SR // 4, dtype=np.float32),
        np.array([np.nan, np.inf, -np.inf] * (SR // 2), dtype=np.float32),
    ):
        empty = detect_audio_cues(bad, SR)
        assert empty == AudioCues(confidence="low")
    assert detect_audio_cues(flat, sr=0) == AudioCues(confidence="low")
    assert detect_audio_cues(np.ones((2, 2, SR), dtype=np.float32), SR) == AudioCues(
        confidence="low"
    )
    assert format_cues_summary(None) == ""
    assert format_cues_summary(AudioCues(confidence="low")) == ""


# ----------------------------------------------------------------------- WeaveShift report & CSV


def test_load_weaveshift_setlist_report_json(tmp_path: Path) -> None:
    report = {
        "app": "WeaveShift 1.1.0 Setlist Builder",
        "schema_version": "1.0.0",
        "output_file": "/tmp/out/Midnight Drive Vol 2.wav",
        "output_duration_sec": 540.0,
        "tracks": [
            {"path": "01_Opening.wav", "title": "Opening Pulse", "bpm": 124.0, "key": "A minor"},
            {
                "path": "/nonexistent/../../02_Neon_Horizon.wav",
                "title": "",
                "bpm": 128.0,
                "key": "C major",
            },
            {"path": "03_Final_Light.mp3", "title": "Final Light", "bpm": 132.0},
        ],
        "track_source_zero_in_mix_sec": [0.0, 178.0, 360.0],
        "transitions": [
            {
                "index": 1,
                "mix_transition_start_sec": 180.0,
                "source_b_cue_sec": 4.0,
                "next_track_source_zero_in_mix_sec": 176.0,
            },
            {
                "index": 2,
                "mix_transition_start_sec": 362.5,
            },
        ],
    }
    path = tmp_path / "mix.weaveshift_setlist_report.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    spec = load_tracklist(path)
    assert spec.source == "weaveshift_report"
    assert spec.title == "Midnight Drive Vol 2"
    assert len(spec.tracks) == 3
    assert spec.tracks[0].start == 0.0 and spec.tracks[0].title == "Opening Pulse"
    assert spec.tracks[0].key == "A minor" and spec.tracks[0].bpm == 124.0
    # Track 2 uses next_zero (176.0) + b_cue (4.0) = 180.0 and safely extracts basename stem.
    assert spec.tracks[1].start == 180.0 and spec.tracks[1].title == "02_Neon_Horizon"
    # Track 3 falls back to mix_transition_start_sec (362.5).
    assert spec.tracks[2].start == 362.5 and spec.tracks[2].title == "Final Light"

    resolved = resolve_chapters(spec, duration=540.0)
    assert [(r.start, r.end, r.key) for r in resolved] == [
        (0.0, 180.0, "A minor"),
        (180.0, 362.5, "C major"),
        (362.5, 540.0, None),
    ]


def test_weaveshift_report_missing_timeline_raises_config_error(tmp_path: Path) -> None:
    bad = tmp_path / "broken.weaveshift_setlist_report.json"
    bad.write_text(
        json.dumps({"tracks": [{"title": "A"}, {"title": "B"}], "transitions": [{}]}),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="missing timeline start"):
        load_tracklist(bad)

    neg = tmp_path / "neg.weaveshift_setlist_report.json"
    neg.write_text(
        json.dumps(
            {
                "tracks": [{"title": "A"}, {"title": "B"}],
                "transitions": [{"mix_transition_start_sec": -5.0}],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="negative start"):
        load_tracklist(neg)

    bad_artist = tmp_path / "bad_artist.weaveshift_setlist_report.json"
    bad_artist.write_text(
        json.dumps(
            {
                "tracks": [{"title": "A", "artist": 123}, {"title": "B"}],
                "transitions": [{"mix_transition_start_sec": 60.0}],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="artist must be a string"):
        load_tracklist(bad_artist)

    dot_path = tmp_path / "dot.weaveshift_setlist_report.json"
    dot_path.write_text(
        json.dumps(
            {
                "output_file": r"C:\mixes\Final Set.wav",
                "tracks": [{"path": "..", "title": ""}, {"path": r"C:\tracks\02_Wave.wav"}],
                "transitions": [{"mix_transition_start_sec": 60.0}],
            }
        ),
        encoding="utf-8",
    )
    spec = load_tracklist(dot_path)
    assert spec.title == "Final Set"
    assert spec.tracks[0].title == "Track 01"
    assert spec.tracks[1].title == "02_Wave"


def test_load_weaveshift_cues_csv_unescapes_and_preserves_cues(tmp_path: Path) -> None:
    csv_text = (
        "title,path,bpm_manifest,bpm_detected,key,duration_sec,final_start_sec,"
        "first_lift_source_sec,first_lift_final_sec,main_peak_source_sec,main_peak_final_sec,"
        "second_peak_source_sec,second_peak_final_sec,breakdown_source_sec,breakdown_final_sec,"
        "transition_out_sec,visual_mood,confidence,notes\n"
        "'=Opening Anthem,a.wav,124.00,123.80,8A,180.0,0.000,"
        "24.0,24.000,60.0,60.000,120.0,120.000,90.0,90.000,182.000,"
        '"neon, misty",high,\n'
        "'Til Dawn,b.wav,,130.50,9A,180.0,178.000,"
        "20.0,,55.0,,110.0,,80.0,,355.000,"
        "electric,medium,\n"
    )
    csv_path = tmp_path / "weaveshift_cues.csv"
    csv_path.write_text(csv_text, encoding="utf-8")

    spec = load_tracklist(csv_path)
    assert spec.source == "csv"
    assert len(spec.tracks) == 2

    t0, t1 = spec.tracks
    # Leading formula guard stripped on '=Opening Anthem, preserved on 'Til Dawn.
    assert t0.title == "=Opening Anthem"
    assert t0.bpm == 124.0 and t0.key == "8A" and t0.mood_tags == ["neon", "misty"]
    assert t0.cues == AudioCues(
        first_lift=24.0, main_peak=60.0, second_peak=120.0, breakdown=90.0, confidence="high"
    )

    assert t1.title == "'Til Dawn"
    assert t1.bpm == 130.5  # Fell back to bpm_detected when bpm_manifest was empty.
    # Source-relative cues shifted by final_start_sec (178.0).
    assert t1.cues == AudioCues(
        first_lift=198.0, main_peak=233.0, second_peak=288.0, breakdown=258.0, confidence="medium"
    )

    # Note: transition_out_sec (182.0) > next final_start_sec (178.0) must NOT fail overlap check.
    resolved = resolve_chapters(spec, duration=360.0)
    assert [(r.start, r.end) for r in resolved] == [(0.0, 178.0), (178.0, 360.0)]
    assert resolved[0].cues == t0.cues


def test_csv_without_timeline_start_raises_actionable_error(tmp_path: Path) -> None:
    csv_path = tmp_path / "no_timeline.csv"
    csv_path.write_text("title,path,final_start_sec\nSong A,a.wav,\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="--timeline"):
        load_tracklist(csv_path)


# --------------------------------------------------------------------------- DSP ports


def test_fold_tempo_to_hint_and_halving_cap() -> None:
    beats_87 = [i * (60.0 / 87.0) for i in range(5)]
    tempo, beats = audio.fold_tempo_to_hint(87.0, 174.0, beats_87)
    assert tempo == pytest.approx(174.0)
    assert len(beats) == 9

    beats_128 = [i * (60.0 / 128.0) for i in range(8)]
    tempo_low, beats_low = audio.fold_tempo_to_hint(128.0, 64.0, beats_128)
    assert tempo_low == pytest.approx(64.0)
    assert beats_low == beats_128[::2]

    # Invalid hint falls back to standard range.
    t_fb, _ = audio.fold_tempo_to_hint(180.0, float("nan"), beats_128)
    assert t_fb == pytest.approx(90.0)

    # Extreme tempo halving is bounded at 4 octaves.
    t_ext, b_ext = audio.fold_tempo(1e6, [float(i) for i in range(32)])
    assert t_ext == pytest.approx(1e6 / 16.0)
    assert len(b_ext) == 2


def test_camelot_key_and_chroma_contrast_guard() -> None:
    assert audio.camelot_key("C major") == "8B"
    assert audio.camelot_key("A minor") == "8A"
    assert audio.camelot_key("F# minor") == "11A"
    assert audio.camelot_key("D-flat major") == "3B"
    assert audio.camelot_key("Eb minor") == "2A"
    assert audio.camelot_key("B♭ maj") == "6B"
    assert audio.camelot_key(None) is None
    assert audio.camelot_key("not a key") is None
    assert audio.camelot_key("C major minor") is None

    # Tonal chord gets key + Camelot notation.
    t = np.arange(int(3.0 * SR), dtype=np.float32) / SR
    a_minor = (
        np.sin(2 * np.pi * 220.0 * t)
        + np.sin(2 * np.pi * 261.63 * t)
        + np.sin(2 * np.pi * 329.63 * t)
    ) / 3.0
    mf_tonal = audio.analyze_musical_features(a_minor.astype(np.float32), SR)
    assert mf_tonal.key == "A minor"
    assert mf_tonal.camelot == "8A"

    # White noise has flat chroma spectrum -> contrast guard rejects spurious key.
    rng = np.random.default_rng(123)
    noise = (0.3 * rng.standard_normal(SR * 4)).astype(np.float32)
    mf_noise = audio.analyze_musical_features(noise, SR)
    assert mf_noise.key is None
    assert mf_noise.mode is None
    assert mf_noise.camelot is None


def test_detect_vocal_windows_heuristic_and_non_lipsync_fallback(tmp_path: Path) -> None:
    seconds = 14.0
    t = np.arange(int(seconds * SR), dtype=np.float64) / SR
    # Percussive background + two midrange harmonic vocal-like phrases at 2..5s and 9..12s.
    mix = 0.08 * click_track(seconds=seconds, bpm=120.0).astype(np.float64)
    for start_s, end_s in ((2.0, 5.0), (9.0, 12.0)):
        mask = (t >= start_s) & (t < end_s)
        tt = t[mask]
        mix[mask] += 0.45 * (
            0.5 * np.sin(2.0 * np.pi * 440.0 * tt)
            + 0.35 * np.sin(2.0 * np.pi * 880.0 * tt)
            + 0.15 * np.sin(2.0 * np.pi * 1320.0 * tt)
        )
    mix32 = np.clip(mix, -1.0, 1.0).astype(np.float32)

    windows = audio.detect_vocal_windows_heuristic(mix32, SR)
    assert len(windows) >= 2
    assert any(w.overlap(2.0, 5.0) >= 2.0 for w in windows)
    assert any(w.overlap(9.0, 12.0) >= 2.0 for w in windows)

    # Degenerate inputs return [].
    assert audio.detect_vocal_windows_heuristic(np.zeros(SR * 2, dtype=np.float32), SR) == []
    assert audio.detect_vocal_windows_heuristic(np.ones(100, dtype=np.float32), SR) == []
    assert audio.detect_vocal_windows_heuristic(np.ones((2, 2, SR), dtype=np.float32), SR) == []

    # In narrative style with lyrics.txt but no vocals.wav, analyze_project uses heuristic fallback.
    proj_dir = api.init_project(tmp_path / "narr-no-stem")
    sf.write(proj_dir / "inputs" / "mix.wav", mix32, SR)
    (proj_dir / "inputs" / "lyrics.txt").write_text("[Verse 1]\nline one\nline two\n")
    (proj_dir / "inputs" / "style.md").write_text("Indie electronic\n")
    yaml_path = proj_dir / "project.yaml"
    yaml_path.write_text(
        yaml_path.read_text().replace("video_style: lipsync", "video_style: narrative")
    )

    _, analyzed = api.analyze(proj_dir)
    assert analyzed.vocal_windows
    assert all(ln.timing_source == "estimated" for ln in analyzed.lyric_lines)
    assert analyzed.cues is not None


def test_mix_planner_anchors_loop_samples_to_chapter_cues() -> None:
    chapter = AnalyzedChapter(
        id="ch01",
        index=0,
        start=0.0,
        end=240.0,
        title="Peak Driver",
        bpm=128.0,
        key="8A",
        energy=0.8,
        cues=AudioCues(first_lift=30.0, main_peak=95.0, second_peak=175.0, breakdown=135.0),
    )
    starts = _chapter_sample_starts(chapter, clip_dur=6.0, clips_per_seg=3, total_duration=240.0)
    # Main peak (95), first_lift (30), second_peak (175) -> centered (-3s) and sorted.
    assert starts == [27.0, 92.0, 172.0]

    analysis = Analysis(
        duration=240.0,
        tempo_bpm=128.0,
        beats=[0.0],
        bars=[0.0],
        bar_intensity=[0.8],
        vocal_windows=[],
        chapters=[chapter],
    )
    cfg = ProjectConfig(title="Cue Mix", mode=ProjectMode.MIX)
    cfg.mix.clips_per_segment = 3
    cfg.mix.clip_seconds = 6.0
    cfg.mix.xfade_seconds = 1.0
    cfg.llm.enabled = False

    manifest = build_mix_manifest(cfg, analysis, "Dark ambient", llm=None)
    assert [s.start for s in manifest.shots] == pytest.approx([27.0, 92.0, 172.0], abs=1.0)


def test_mix_analysis_preserves_csv_cues_and_key_end_to_end(tmp_path: Path) -> None:
    root = tmp_path / "csv_mix"
    (root / "inputs").mkdir(parents=True)
    sf.write(root / "inputs" / "mix.wav", click_track(seconds=24.0, bpm=124.0), SR)
    (root / "inputs" / "cues.csv").write_text(
        "title,final_start_sec,bpm_manifest,key,main_peak_final_sec,confidence\n"
        "Track One,0.0,124.0,8A,6.0,high\n"
        "Track Two,12.0,174.0,11A,18.0,medium\n",
        encoding="utf-8",
    )
    cfg = ProjectConfig(title="CSV Mix", mode=ProjectMode.MIX)
    cfg.inputs.tracklist = "inputs/cues.csv"
    cfg.llm.enabled = False
    (root / "project.yaml").write_text(yaml.safe_dump(cfg.model_dump(mode="json")))

    _, analyzed = api.analyze(Project.open(root))
    assert len(analyzed.chapters) == 2
    assert analyzed.chapters[0].key == "8A"
    assert analyzed.chapters[0].cues is not None and analyzed.chapters[0].cues.main_peak == 6.0
    assert analyzed.chapters[1].bpm == 174.0 and analyzed.chapters[1].key == "11A"
    assert analyzed.chapters[1].cues is not None and analyzed.chapters[1].cues.main_peak == 18.0


# --------------------------------------------------------------------------- subprocess hardening


def test_media_probe_and_transcode_restrict_protocol_whitelist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wav = tmp_path / "sample.wav"
    sf.write(wav, np.zeros(SR, dtype=np.float32), SR)

    seen_cmds: list[list[str]] = []
    real_run = media.run

    def _spy_run(cmd: list[str], *, timeout: float | None = None):
        seen_cmds.append(list(cmd))
        return real_run(cmd, timeout=timeout)

    monkeypatch.setattr(media, "run", _spy_run)

    info = media.probe(wav)
    assert info.duration == pytest.approx(1.0, abs=0.05)
    assert "-protocol_whitelist" in seen_cmds[0] and "file" in seen_cmds[0]
    assert "--" in seen_cmds[0]

    dst = tmp_path / "out.wav"
    media.transcode_to_wav(wav, dst)
    assert dst.is_file()
    assert "-protocol_whitelist" in seen_cmds[1] and "file" in seen_cmds[1]

    sliced = tmp_path / "slice.wav"
    media.slice_audio(wav, sliced, 0.0, 0.5)
    assert sliced.is_file()
    assert "-protocol_whitelist" in seen_cmds[2] and "file" in seen_cmds[2]


def test_demucs_separator_uses_double_dash_and_retries_mps_on_cpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mix = tmp_path / "song.wav"
    mix.write_bytes(b"fake")

    sep_mps = DemucsSeparator(model="htdemucs", device="mps")
    cmd = sep_mps._build_command(mix, tmp_path / "out", include_drums=False)
    assert cmd[-2:] == ["--", str(mix.resolve())]

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    calls: list[list[str]] = []

    def _fail_mps_then_succeed_cpu(cmd: list[str], **_kwargs) -> subprocess.CompletedProcess[str]:
        calls.append(list(cmd))
        if "-d" in cmd and cmd[cmd.index("-d") + 1] == "mps":
            return subprocess.CompletedProcess(cmd, 1, "", "RuntimeError: MPS backend error")
        out_dir = Path(cmd[cmd.index("-o") + 1]) / "htdemucs" / "song"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "vocals.wav").write_bytes(b"VOCALS")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(subprocess, "run", _fail_mps_then_succeed_cpu)
    vocals_out = tmp_path / "inputs" / "vocals.wav"
    res = sep_mps.separate(mix, vocals_out)
    assert res.vocals == vocals_out and vocals_out.read_bytes() == b"VOCALS"
    assert len(calls) == 2
    assert calls[0][calls[0].index("-d") + 1] == "mps"
    assert calls[1][calls[1].index("-d") + 1] == "cpu"

    # CPU device failures (even with stderr=None) must not retry.
    calls.clear()
    sep_cpu = DemucsSeparator(model="htdemucs", device="cpu")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda cmd, **_kw: (
            calls.append(list(cmd)),
            subprocess.CompletedProcess(cmd, 1, "", None),  # type: ignore[arg-type]
        )[1],
    )
    with pytest.raises(MediaToolError, match="demucs failed"):
        sep_cpu.separate(mix, vocals_out)
    assert len(calls) == 1
