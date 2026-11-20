# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Tests for audio-only ingestion (`mvd prepare`): stem split, transcription, and inference."""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import yaml
from pydantic import ValidationError
from typer.testing import CliRunner

from mvdirector import api as api_mod
from mvdirector import cli as cli_mod
from mvdirector import media
from mvdirector.audio.lyrics import (
    Transcript,
    TranscriptSegment,
    WhisperTranscriber,
    Word,
    format_transcript_as_lyrics,
)
from mvdirector.audio.prepare import (
    AudioSummary,
    InferredInputsReply,
    infer_inputs_from_transcript,
    prepare_project,
    resolve_source_audio,
)
from mvdirector.audio.stems import DemucsSeparator, StemResult
from mvdirector.config import PrepareConfig
from mvdirector.errors import ConfigError, MediaToolError
from mvdirector.models import ShotType, TimeSpan
from mvdirector.project import Project


def _write_synthetic_song_wav(path: Path, duration: float = 12.0, sr: int = 22050) -> Path:
    """Create a 120 BPM synthetic song with strong beats and clear vocal-band tone bursts."""
    n = int(duration * sr)
    t = np.arange(n, dtype=np.float32) / sr
    audio = np.zeros(n, dtype=np.float32)
    # 120 BPM kick pulses every 0.5s
    for beat in np.arange(0.0, duration, 0.5):
        start = int(beat * sr)
        length = min(int(0.08 * sr), n - start)
        if length > 0:
            env = np.exp(-np.linspace(0, 6, length, dtype=np.float32))
            audio[start : start + length] += 0.6 * env * np.sin(2 * np.pi * 80 * t[:length])
    # Singing-like harmonic burst between 2s and 10s
    v_start, v_end = int(2.0 * sr), int(10.0 * sr)
    audio[v_start:v_end] += 0.35 * np.sin(2 * np.pi * 440 * t[v_start:v_end])
    sf.write(str(path), audio, sr, subtype="PCM_16")
    return path


def _write_synthetic_vocal_wav(path: Path, duration: float = 12.0, sr: int = 22050) -> Path:
    """Create a vocal stem active from 2s to 10s (silent elsewhere)."""
    n = int(duration * sr)
    t = np.arange(n, dtype=np.float32) / sr
    audio = np.zeros(n, dtype=np.float32)
    v_start, v_end = int(2.0 * sr), int(10.0 * sr)
    audio[v_start:v_end] = 0.5 * np.sin(2 * np.pi * 440 * t[v_start:v_end])
    sf.write(str(path), audio, sr, subtype="PCM_16")
    return path


def _encode_mp3(wav_path: Path, mp3_path: Path) -> Path:
    media.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(wav_path.resolve()),
            "-c:a",
            "libmp3lame",
            "-b:a",
            "128k",
            str(mp3_path.resolve()),
        ]
    )
    return mp3_path


class _FakeSeparator:
    def __init__(self, duration: float = 12.0) -> None:
        self.calls: list[tuple[Path, Path, Path | None]] = []
        self.duration = duration

    def separate(
        self, mix_path: Path, vocals_out: Path, *, drums_out: Path | None = None
    ) -> StemResult:
        self.calls.append((mix_path, vocals_out, drums_out))
        vocals_out.parent.mkdir(parents=True, exist_ok=True)
        _write_synthetic_vocal_wav(vocals_out, duration=self.duration)
        drums_written: Path | None = None
        if drums_out is not None:
            drums_out.parent.mkdir(parents=True, exist_ok=True)
            _write_synthetic_song_wav(drums_out, duration=self.duration)
            drums_written = drums_out
        return StemResult(vocals=vocals_out, drums=drums_written)


class _FakeTranscriber:
    def __init__(self, transcript: Transcript | None = None) -> None:
        self.calls: list[Path] = []
        self.transcript = transcript or Transcript(
            segments=[
                TranscriptSegment(text="walking through the midnight rain", start=2.1, end=5.4),
                TranscriptSegment(text="calling out your neon name", start=5.6, end=9.4),
            ],
            words=[
                Word(word="walking", start=2.1, end=2.6),
                Word(word="through", start=2.6, end=3.0),
                Word(word="the", start=3.0, end=3.2),
                Word(word="midnight", start=3.2, end=4.2),
                Word(word="rain", start=4.2, end=5.4),
                Word(word="calling", start=5.6, end=6.2),
                Word(word="out", start=6.2, end=6.6),
                Word(word="your", start=6.6, end=7.0),
                Word(word="neon", start=7.0, end=8.2),
                Word(word="name", start=8.2, end=9.4),
            ],
            language="en",
        )

    def transcribe(self, audio_path: Path) -> Transcript:
        self.calls.append(audio_path)
        return self.transcript


class _FakeInferLLM:
    def __init__(self, reply: InferredInputsReply | None = None) -> None:
        self.seen_system = ""
        self.seen_request = ""
        self.reply = reply or InferredInputsReply(
            lyrics=(
                "[Verse 1]\nWalking through the midnight rain\n\n"
                "[Chorus]\nCalling out your neon name"
            ),
            music_style="Synthwave dark pop, 120 BPM, yearning emotional vocal delivery",
            visual_direction=(
                "Solo singer in a rain-slicked neon alley at night, wearing a dark trench coat. "
                "Unobstructed close-up and medium-close lip-sync performance shots under warm "
                "neon signage intercut with wet-asphalt reflection cutaways."
            ),
        )

    def complete_json(self, *, system, content, schema):
        assert schema is InferredInputsReply
        self.seen_system = system
        self.seen_request = content[0]["text"]
        return self.reply


def test_prepare_config_validation() -> None:
    cfg = PrepareConfig()
    assert cfg.stem_model == "htdemucs"
    assert cfg.stem_device == "auto"
    assert cfg.whisper_model == "small"
    assert cfg.whisper_language is None

    with pytest.raises(ValidationError):
        PrepareConfig(stem_model="--evil-flag")
    with pytest.raises(ValidationError):
        PrepareConfig(stem_model="../escape")
    with pytest.raises(ValidationError):
        PrepareConfig(whisper_language="english")
    assert PrepareConfig(whisper_language="en", stem_device="mps").stem_device == "mps"


def test_demucs_separator_command_and_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ConfigError, match="Invalid Demucs model name"):
        DemucsSeparator(model="../bad")
    with pytest.raises(ConfigError, match="Invalid stem_device"):
        DemucsSeparator(device="tpu")

    sep = DemucsSeparator(model="htdemucs", device="mps")
    mix = tmp_path / "mix.wav"
    mix.write_bytes(b"fake")

    cmd_two = sep._build_command(mix, tmp_path / "out", include_drums=False)
    assert "--two-stems" in cmd_two and "vocals" in cmd_two
    assert "-d" in cmd_two and "mps" in cmd_two

    cmd_four = DemucsSeparator(device="auto")._build_command(
        mix, tmp_path / "out", include_drums=True
    )
    assert "--two-stems" not in cmd_four
    assert "-d" not in cmd_four

    # Missing demucs raises an actionable ConfigError.
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    with pytest.raises(ConfigError, match=r"\.\[stems\]"):
        sep.separate(mix, tmp_path / "vocals.wav")

    # Successful run copies vocals.wav and drums.wav atomically.
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())

    def _fake_run(cmd: list[str], **_kwargs) -> subprocess.CompletedProcess[str]:
        out_idx = cmd.index("-o") + 1
        out_dir = Path(cmd[out_idx]) / "htdemucs" / "mix"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "vocals.wav").write_bytes(b"VOCALS")
        (out_dir / "drums.wav").write_bytes(b"DRUMS")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(subprocess, "run", _fake_run)
    vocals_out = tmp_path / "inputs" / "vocals.wav"
    drums_out = tmp_path / "inputs" / "drums.wav"
    res = sep.separate(mix, vocals_out, drums_out=drums_out)
    assert res.vocals == vocals_out and vocals_out.read_bytes() == b"VOCALS"
    assert res.drums == drums_out and drums_out.read_bytes() == b"DRUMS"

    # Non-zero exit raises MediaToolError.
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda cmd, **_kw: subprocess.CompletedProcess(cmd, 2, "", "boom"),
    )
    with pytest.raises(MediaToolError, match="demucs failed"):
        sep.separate(mix, vocals_out)


def test_whisper_transcriber_missing_dep_and_lyrics_formatting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(__import__("sys").modules, "faster_whisper", None)
    with pytest.raises(ConfigError, match=r"\.\[align\]"):
        WhisperTranscriber().transcribe(tmp_path / "vocals.wav")

    assert format_transcript_as_lyrics([]) == "[Instrumental]\n"
    segs = [
        TranscriptSegment(text="Line one", start=1.0, end=2.5),
        TranscriptSegment(text="Line two", start=2.8, end=4.0),
        TranscriptSegment(text="Chorus hook", start=8.5, end=10.5),
    ]
    formatted = format_transcript_as_lyrics(segs, pause_threshold=3.0)
    assert "[Section 1]\nLine one\nLine two" in formatted
    assert "[Section 2]\nChorus hook" in formatted


def test_infer_inputs_from_transcript_llm_and_fallback() -> None:
    summary = AudioSummary(
        duration=12.0,
        tempo_bpm=120.0,
        vocal_windows=[TimeSpan(start=2.0, end=10.0)],
        vocal_fraction=0.667,
        energy_arc=["0:00-0:06 moderate (0.45)", "0:06-0:12 peak (0.90)"],
    )
    transcript = _FakeTranscriber().transcript

    llm = _FakeInferLLM()
    reply, used_llm = infer_inputs_from_transcript(transcript, summary, llm=llm)
    assert used_llm is True
    assert "TEMPO: ~120 BPM" in llm.seen_request
    assert "walking through the midnight rain" in llm.seen_request
    assert "ENERGY ARC:" in llm.seen_request
    assert "[Chorus]" in reply.lyrics
    assert "lip-sync" in reply.visual_direction.lower()

    # Fallback when llm is None still produces valid lyrics + lip-sync visual direction.
    fallback_reply, used_llm_fb = infer_inputs_from_transcript(transcript, summary, llm=None)
    assert used_llm_fb is False
    assert "[Section 1]" in fallback_reply.lyrics
    assert "lip-sync" in fallback_reply.visual_direction.lower()

    # Instrumental fallback when no segments were transcribed.
    inst_reply, _ = infer_inputs_from_transcript(
        Transcript(segments=[], words=[]), summary, llm=None
    )
    assert inst_reply.lyrics.strip() == "[Instrumental]"
    assert "instrumental" in inst_reply.music_style.lower()


def test_resolve_source_audio_validation_and_mp3_discovery(tmp_path: Path) -> None:
    proj_dir = tmp_path / "song"
    api_mod.init_project(proj_dir)
    proj = Project.open(proj_dir)

    with pytest.raises(ConfigError, match="No source audio found"):
        resolve_source_audio(proj, None)

    bad_ext = tmp_path / "song.flac"
    bad_ext.write_bytes(b"fake")
    with pytest.raises(ConfigError, match="Unsupported audio format"):
        resolve_source_audio(proj, bad_ext)

    sibling_mp3 = proj_dir / "inputs" / "mix.mp3"
    sibling_mp3.write_bytes(b"fake mp3")
    assert resolve_source_audio(proj, None) == sibling_mp3


def test_prepare_end_to_end_from_mp3_to_lipsync_manifest(tmp_path: Path) -> None:
    """User supplies ONLY an .mp3 file; prepare -> analyze -> plan produces lip-sync shots."""
    proj_dir = tmp_path / "audio_only_song"
    api_mod.init_project(proj_dir)

    # Disable LLM for `plan` so rule-based planner runs deterministically, and enable drums +
    # word_timestamps to verify all optional outputs are written cleanly.
    cfg_path = proj_dir / "project.yaml"
    cfg_data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg_data["llm"]["enabled"] = False
    cfg_data["inputs"]["drums"] = "inputs/drums.wav"
    cfg_data["inputs"]["word_timestamps"] = "inputs/word_timestamps.json"
    cfg_path.write_text(yaml.safe_dump(cfg_data), encoding="utf-8")

    # Create a real .mp3 outside the project directory.
    raw_wav = _write_synthetic_song_wav(tmp_path / "source.wav", duration=12.0)
    mp3_path = _encode_mp3(raw_wav, tmp_path / "my_track.mp3")

    separator = _FakeSeparator(duration=12.0)
    transcriber = _FakeTranscriber()
    fake_llm = _FakeInferLLM()

    result = api_mod.prepare(
        proj_dir,
        audio=mp3_path,
        separator=separator,
        transcriber=transcriber,
        llm=fake_llm,
    )

    assert result.inferred_with_llm is True
    assert result.mix_path == (proj_dir / "inputs" / "mix.wav").resolve()
    assert result.mix_path.is_file()
    assert media.probe(result.mix_path).has_audio
    assert result.vocals_path.is_file()
    assert result.drums_path is not None and result.drums_path.is_file()
    assert result.word_timestamps_path is not None and result.word_timestamps_path.is_file()
    assert len(json.loads(result.word_timestamps_path.read_text(encoding="utf-8"))) == 10
    assert "[Chorus]" in result.lyrics_path.read_text(encoding="utf-8")
    assert "Visual direction:" in result.style_path.read_text(encoding="utf-8")
    assert result.vocal_windows >= 1
    assert result.transcribed_words == 10

    # Downstream `analyze` and `plan` work immediately and schedule lip-sync shots!
    _, analysis = api_mod.analyze(proj_dir)
    assert len(analysis.vocal_windows) >= 1
    assert any(ln.timing_source == "aligned" for ln in analysis.lyric_lines)

    _, manifest = api_mod.plan(proj_dir)
    assert any(shot.type == ShotType.LIPSYNC for shot in manifest.shots)

    # Re-running without --overwrite fails fast BEFORE invoking separator or transcriber.
    sep_calls_before = len(separator.calls)
    trans_calls_before = len(transcriber.calls)
    with pytest.raises(ConfigError, match="--overwrite"):
        prepare_project(
            Project.open(proj_dir),
            audio=mp3_path,
            overwrite=False,
            separator=separator,
            transcriber=transcriber,
            llm=fake_llm,
        )
    assert len(separator.calls) == sep_calls_before
    assert len(transcriber.calls) == trans_calls_before


def test_prepare_cli_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    proj_dir = tmp_path / "cli_song"
    runner = CliRunner()
    assert runner.invoke(cli_mod.app, ["init", str(proj_dir)]).exit_code == 0

    src_wav = _write_synthetic_song_wav(tmp_path / "input.wav", duration=8.0)
    fake_llm = _FakeInferLLM()
    monkeypatch.setattr(api_mod, "build_llm", lambda _cfg: fake_llm)
    monkeypatch.setattr(
        "mvdirector.audio.prepare.DemucsSeparator",
        lambda **_kw: _FakeSeparator(duration=8.0),
    )
    monkeypatch.setattr(
        "mvdirector.audio.prepare.WhisperTranscriber",
        lambda **_kw: _FakeTranscriber(),
    )

    res = runner.invoke(cli_mod.app, ["prepare", str(src_wav), "-p", str(proj_dir)])
    assert res.exit_code == 0, res.output
    assert "Prepared inputs from" in res.output
    assert "inferred via LLM" in res.output
    assert (proj_dir / "inputs" / "vocals.wav").is_file()
    assert (proj_dir / "inputs" / "lyrics.txt").is_file()
    assert (proj_dir / "inputs" / "style.md").is_file()
