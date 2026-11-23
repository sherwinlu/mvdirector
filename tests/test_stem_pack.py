# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Tests for Suno/DAW multi-stem pack ingestion, security, analysis, and planning integration."""

from __future__ import annotations

import stat
import zipfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from typer.testing import CliRunner

from mvdirector import api as api_mod
from mvdirector import cli as cli_mod
from mvdirector.audio import stem_pack
from mvdirector.audio.analysis import analyze_project
from mvdirector.audio.lyrics import Transcript, TranscriptSegment, Word
from mvdirector.audio.prepare import prepare_project
from mvdirector.errors import ConfigError
from mvdirector.models import (
    Analysis,
    StemArrangement,
    StemTrackInfo,
    TimeSpan,
)
from mvdirector.planning.planner import build_manifest
from mvdirector.project import Project


def _write_tone_wav(
    path: Path,
    *,
    duration: float = 8.0,
    sr: int = 22050,
    freq: float = 440.0,
    active_span: tuple[float, float] | None = None,
    amplitude: float = 0.4,
) -> Path:
    n = int(duration * sr)
    t = np.arange(n, dtype=np.float32) / sr
    audio = np.zeros(n, dtype=np.float32)
    start_s, end_s = active_span or (0.0, duration)
    s_idx = max(0, int(start_s * sr))
    e_idx = min(n, int(end_s * sr))
    if e_idx > s_idx:
        audio[s_idx:e_idx] = amplitude * np.sin(2 * np.pi * freq * t[s_idx:e_idx])
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sr, subtype="PCM_16")
    return path


def _write_kick_wav(
    path: Path,
    *,
    duration: float = 8.0,
    sr: int = 22050,
    bpm: float = 120.0,
    active_span: tuple[float, float] | None = None,
) -> Path:
    n = int(duration * sr)
    t = np.arange(n, dtype=np.float32) / sr
    audio = np.zeros(n, dtype=np.float32)
    start_s, end_s = active_span or (0.0, duration)
    beat_step = 60.0 / bpm
    beat = start_s
    while beat < end_s:
        idx = int(beat * sr)
        length = min(int(0.08 * sr), n - idx)
        if length > 0:
            env = np.exp(-np.linspace(0.0, 6.0, length, dtype=np.float32))
            audio[idx : idx + length] += 0.7 * env * np.sin(2 * np.pi * 80.0 * t[:length])
        beat += beat_step
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sr, subtype="PCM_16")
    return path


def _make_suno_zip(zip_path: Path, tmp_dir: Path, *, include_vocals: bool = True) -> Path:
    """Build a realistic Suno Auto Split .zip with numbered stem WAVs and __MACOSX junk."""
    stems_src = tmp_dir / "src_stems"
    files: dict[str, Path] = {}
    if include_vocals:
        files["0 Lead Vocals.wav"] = _write_tone_wav(
            stems_src / "0 Lead Vocals.wav", freq=440.0, active_span=(1.0, 5.0), amplitude=0.5
        )
        files["1 Backing Vocals.wav"] = _write_tone_wav(
            stems_src / "1 Backing Vocals.wav", freq=554.37, active_span=(4.0, 7.5), amplitude=0.35
        )
    files["2 Drums.wav"] = _write_kick_wav(
        stems_src / "2 Drums.wav", bpm=120.0, active_span=(0.0, 8.0)
    )
    files["3 Bass.wav"] = _write_tone_wav(
        stems_src / "3 Bass.wav", freq=110.0, active_span=(0.0, 8.0), amplitude=0.35
    )
    files["7 Synth.wav"] = _write_tone_wav(
        stems_src / "7 Synth.wav", freq=220.0, active_span=(2.0, 8.0), amplitude=0.3
    )
    files["9 Brass.wav"] = _write_tone_wav(
        stems_src / "9 Brass.wav", freq=329.63, active_span=(4.0, 8.0), amplitude=0.3
    )

    with zipfile.ZipFile(zip_path, "w") as zf:
        for arc_name, file_path in files.items():
            zf.write(file_path, arcname=f"My Song Stems/{arc_name}")
        zf.writestr("__MACOSX/My Song Stems/._0 Lead Vocals.wav", b"mac-junk")
        zf.writestr("My Song Stems/.DS_Store", b"ds-store")
        zf.writestr("My Song Stems/README.txt", b"exported from suno")
    return zip_path


class _FailIfCalledSeparator:
    def separate(self, mix_path: Path, vocals_out: Path, *, drums_out: Path | None = None):
        raise AssertionError("Demucs should not be called when vocal stems are provided")


class _FakeTranscriber:
    def __init__(self) -> None:
        self.calls: list[Path] = []

    def transcribe(self, vocals_path: Path) -> Transcript:
        self.calls.append(vocals_path)
        return Transcript(
            segments=[TranscriptSegment(text="midnight lights on the avenue", start=1.1, end=4.8)],
            words=[
                Word(word="midnight", start=1.1, end=1.8),
                Word(word="lights", start=1.9, end=2.5),
                Word(word="on", start=2.6, end=3.0),
                Word(word="the", start=3.1, end=3.5),
                Word(word="avenue", start=3.6, end=4.8),
            ],
            language="en",
        )


class _RecordingLLM:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def complete_json(self, system: str, content, schema, **kwargs):
        text = content if isinstance(content, str) else str(content)
        self.prompts.append(text)
        name = schema.__name__
        if name == "InferredInputsReply":
            return schema.model_validate(
                {
                    "lyrics": "[Verse 1]\nMidnight lights on the avenue\n",
                    "music_style": "Driving synth-pop with brass swells and crisp drums.",
                    "visual_direction": "Moody neon performance with clean lip-sync close-ups.",
                }
            )
        if name == "BibleReply":
            return schema.model_validate(
                {
                    "appearance": "Singer in dark coat.",
                    "style": "Cinematic neon realism.",
                    "palette": "Deep indigo and amber.",
                    "lighting": "Rim lighting.",
                    "negative_prompt": "blurry, watermark",
                    "settings": [
                        {
                            "name": "Neon Street",
                            "sections": ["Verse 1"],
                            "description": "Rain-slicked avenue.",
                            "movement": "Slow push-in.",
                        }
                    ],
                }
            )
        if name == "ShotBatchReply":
            # Return minimal valid direction for whatever shot IDs are in the prompt.
            import re

            ids = re.findall(r"id=(s\d+)", text)
            return schema.model_validate(
                {
                    "shots": [
                        {
                            "id": sid,
                            "type": "lipsync",
                            "setting": "Neon Street",
                            "framing": "medium_close",
                            "still_prompt": "Singer performing under neon rain.",
                            "motion_prompt": "Gentle push-in.",
                        }
                        for sid in ids
                    ]
                }
            )
        raise AssertionError(f"Unexpected schema {name}")


# --------------------------------------------------------------------------- classification


@pytest.mark.parametrize(
    ("filename", "expected_role"),
    [
        # Suno Auto Split (11-12 stems)
        ("0 Lead Vocals.wav", "lead_vocals"),
        ("1 Backing Vocals.wav", "backing_vocals"),
        ("2 Drums.wav", "drums"),
        ("3 Bass.wav", "bass"),
        ("4 Keyboard.wav", "keys"),
        ("5 Percussion.wav", "percussion"),
        ("6 Strings.wav", "strings"),
        ("7 Synth.wav", "synth"),
        ("8 Other.wav", "other"),
        ("9 Brass.wav", "brass"),
        ("10 Woodwinds.wav", "woodwinds"),
        ("11 Guitar.wav", "guitar"),
        # Suno Split from Mix
        ("Vocals.wav", "vocals"),
        ("Instrumental.wav", "instrumental"),
        ("No Vocals.wav", "instrumental"),
        ("Song Title - Without Vocals.mp3", "instrumental"),
        # Suno Premier Advanced Split (granular instruments + edge cases)
        ("Kick.wav", "drums"),
        ("Bass Drum.wav", "drums"),
        ("Snare.wav", "drums"),
        ("Hi-Hat.wav", "drums"),
        ("Toms.wav", "drums"),
        ("Shaker.wav", "percussion"),
        ("Synth Bass.wav", "bass"),
        ("Bass Guitar.wav", "bass"),
        ("Electric Guitar.wav", "guitar"),
        ("Grand Piano.wav", "keys"),
        ("Violin.wav", "strings"),
        ("Lead Synth.wav", "synth"),
        ("Trumpet.wav", "brass"),
        ("Alto Sax.wav", "woodwinds"),
        ("Flute.wav", "woodwinds"),
        ("Harmonies.wav", "backing_vocals"),
        ("Choir.wav", "backing_vocals"),
        ("SFX Risers.wav", "fx"),
    ],
)
def test_classify_stem_name(filename: str, expected_role: str) -> None:
    assert stem_pack.classify_stem_name(filename) == expected_role


# --------------------------------------------------------------------------- security


@pytest.mark.parametrize(
    "evil_name",
    ["../escape.wav", "sub/../../escape.wav", "/etc/passwd.wav", "C:\\windows\\bad.wav"],
)
def test_extract_zip_rejects_path_traversal(tmp_path: Path, evil_name: str) -> None:
    bad_zip = tmp_path / "evil.zip"
    with zipfile.ZipFile(bad_zip, "w") as zf:
        zf.writestr(evil_name, b"RIFFfake")
    with pytest.raises(ConfigError, match="unsafe path"):
        stem_pack.extract_zip_stems(bad_zip, tmp_path / "out")


def test_extract_zip_rejects_symlink_entries(tmp_path: Path) -> None:
    bad_zip = tmp_path / "symlink.zip"
    with zipfile.ZipFile(bad_zip, "w") as zf:
        info = zipfile.ZipInfo("0 Lead Vocals.wav")
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        zf.writestr(info, "/etc/passwd")
    with pytest.raises(ConfigError, match="symlink"):
        stem_pack.extract_zip_stems(bad_zip, tmp_path / "out")


def test_extract_zip_enforces_size_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(stem_pack, "MAX_STEM_FILE_BYTES", 1024)
    big_zip = tmp_path / "big.zip"
    with zipfile.ZipFile(big_zip, "w") as zf:
        zf.writestr("0 Lead Vocals.wav", b"0" * 2048)
    with pytest.raises(ConfigError, match="limit"):
        stem_pack.extract_zip_stems(big_zip, tmp_path / "out")


def test_extract_zip_enforces_max_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(stem_pack, "MAX_STEM_FILES", 2)
    many_zip = tmp_path / "many.zip"
    with zipfile.ZipFile(many_zip, "w") as zf:
        for i in range(3):
            zf.writestr(f"{i} Synth.wav", b"RIFF")
    with pytest.raises(ConfigError, match="Too many stem files"):
        stem_pack.extract_zip_stems(many_zip, tmp_path / "out")


def test_extract_zip_rejects_corrupt_archive(tmp_path: Path) -> None:
    corrupt = tmp_path / "corrupt.zip"
    corrupt.write_bytes(b"not a zip")
    with pytest.raises(ConfigError, match="Invalid zip"):
        stem_pack.extract_zip_stems(corrupt, tmp_path / "out")


def test_extract_zip_enforces_max_entries_and_rejects_null_byte(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(stem_pack, "MAX_ZIP_ENTRIES", 2)
    flood_zip = tmp_path / "flood.zip"
    with zipfile.ZipFile(flood_zip, "w") as zf:
        for i in range(3):
            zf.writestr(f"note_{i}.txt", b"x")
    with pytest.raises(ConfigError, match="too many entries"):
        stem_pack.extract_zip_stems(flood_zip, tmp_path / "out")

    bad_info = zipfile.ZipInfo("placeholder.wav")
    bad_info.filename = "0 Lead\x00Vocals.wav"
    with pytest.raises(ConfigError, match="unsafe path"):
        stem_pack._validate_zip_member(bad_info)


def test_prepare_overwrite_with_zip_in_stems_dir_does_not_duplicate_stems(
    fresh_project: Project, tmp_path: Path
) -> None:
    stems_dir = fresh_project.path(fresh_project.config.inputs.stems_dir)
    stems_dir.mkdir(parents=True, exist_ok=True)
    _make_suno_zip(stems_dir / "suno_export.zip", tmp_path)

    first = prepare_project(
        fresh_project,
        audio=None,
        separator=_FailIfCalledSeparator(),  # type: ignore[arg-type]
        transcriber=_FakeTranscriber(),  # type: ignore[arg-type]
        llm=None,
    )
    second = prepare_project(
        fresh_project,
        audio=None,
        overwrite=True,
        separator=_FailIfCalledSeparator(),  # type: ignore[arg-type]
        transcriber=_FakeTranscriber(),  # type: ignore[arg-type]
        llm=None,
    )
    assert first.stems_ingested == 6
    assert second.stems_ingested == 6
    wav_names = sorted(p.name for p in stems_dir.glob("*.wav"))
    assert len(wav_names) == 6
    assert not any(name.endswith("_2.wav") for name in wav_names)


@pytest.mark.parametrize("bad_dir", ["", "."])
def test_prepare_rejects_empty_or_root_stems_dir(
    fresh_project: Project, tmp_path: Path, bad_dir: str
) -> None:
    fresh_project.config.inputs.stems_dir = bad_dir
    zip_path = _make_suno_zip(tmp_path / "suno.zip", tmp_path)
    with pytest.raises(ConfigError, match="stems_dir"):
        prepare_project(
            fresh_project,
            audio=zip_path,
            separator=_FailIfCalledSeparator(),  # type: ignore[arg-type]
            transcriber=_FakeTranscriber(),  # type: ignore[arg-type]
            llm=None,
        )


def test_mix_stems_sanitizes_non_finite_samples(tmp_path: Path) -> None:
    bad = tmp_path / "bad.wav"
    good = tmp_path / "good.wav"
    arr = np.full(2048, 0.25, dtype=np.float32)
    arr[10] = np.nan
    arr[20] = np.inf
    sf.write(str(bad), arr, 22050, subtype="FLOAT")
    _write_tone_wav(good, duration=0.1, freq=440.0, amplitude=0.25)

    out = stem_pack.mix_stems_to_wav([bad, good], tmp_path / "mixed.wav")
    samples, _ = sf.read(str(out), dtype="float64")
    assert np.all(np.isfinite(samples))


# --------------------------------------------------------------------------- mixing & selection


def test_mix_stems_prevents_clipping_and_excludes_redundant_instrumental(tmp_path: Path) -> None:
    lead = _write_tone_wav(tmp_path / "0 Lead Vocals.wav", duration=2.0, freq=440.0, amplitude=0.8)
    drums = _write_tone_wav(tmp_path / "2 Drums.wav", duration=2.0, freq=100.0, amplitude=0.8)
    inst = _write_tone_wav(tmp_path / "Instrumental.wav", duration=2.0, freq=220.0, amplitude=0.8)

    ingested = stem_pack.ingest_stem_sources([lead, drums, inst], tmp_path / "stems")
    selected = stem_pack.select_mix_stems(ingested)
    assert {s.role for s in selected} == {"lead_vocals", "drums"}

    out = stem_pack.mix_stems_to_wav([s.path for s in selected], tmp_path / "mix.wav")
    samples, _ = sf.read(str(out), dtype="float64")
    assert float(np.max(np.abs(samples))) <= 0.99


# --------------------------------------------------------------------------- prepare integration


@pytest.fixture
def fresh_project(tmp_path: Path) -> Project:
    root = api_mod.init_project(tmp_path / "song")
    return Project.open(root)


def test_prepare_from_suno_zip_alone(fresh_project: Project, tmp_path: Path) -> None:
    zip_path = _make_suno_zip(tmp_path / "suno_stems.zip", tmp_path)
    transcriber = _FakeTranscriber()
    llm = _RecordingLLM()

    result = prepare_project(
        fresh_project,
        audio=zip_path,
        separator=_FailIfCalledSeparator(),  # type: ignore[arg-type]
        transcriber=transcriber,  # type: ignore[arg-type]
        llm=llm,  # type: ignore[arg-type]
    )

    assert result.stems_ingested == 6
    assert result.mix_path.is_file()
    assert result.vocals_path.is_file()
    assert transcriber.calls == [result.vocals_path]
    # Lead vocals end at 5.0s while backing vocals run to 7.5s; vocals.wav must be lead-only.
    vocals, sr = sf.read(str(result.vocals_path), dtype="float32")
    tail_rms = float(np.sqrt(np.mean(vocals[int(5.5 * sr) : int(7.0 * sr)] ** 2)))
    assert tail_rms < 1e-3, "Backing vocals leaked into lead vocals.wav"
    assert any("INSTRUMENTATION & STEMS:" in p for p in llm.prompts)


def test_prepare_from_master_mix_plus_suno_zip(fresh_project: Project, tmp_path: Path) -> None:
    zip_path = _make_suno_zip(tmp_path / "suno_stems.zip", tmp_path)
    master = _write_tone_wav(tmp_path / "my_song.wav", duration=8.0, freq=330.0, amplitude=0.25)

    result = prepare_project(
        fresh_project,
        audio=[master, zip_path],
        separator=_FailIfCalledSeparator(),  # type: ignore[arg-type]
        transcriber=_FakeTranscriber(),  # type: ignore[arg-type]
        llm=None,
    )

    assert result.stems_ingested == 6
    # Verify inputs/mix.wav came from the master (not synthesized from the zip).
    master_samples, _ = sf.read(str(master), dtype="float32")
    mix_samples, _ = sf.read(str(result.mix_path), dtype="float32")
    assert np.allclose(master_samples, mix_samples, atol=1e-4)


def test_prepare_discovers_zip_dropped_in_stems_dir_with_no_args(
    fresh_project: Project, tmp_path: Path
) -> None:
    stems_dir = fresh_project.path(fresh_project.config.inputs.stems_dir)
    stems_dir.mkdir(parents=True, exist_ok=True)
    _make_suno_zip(stems_dir / "suno_export.zip", tmp_path)

    result = prepare_project(
        fresh_project,
        audio=None,
        separator=_FailIfCalledSeparator(),  # type: ignore[arg-type]
        transcriber=_FakeTranscriber(),  # type: ignore[arg-type]
        llm=None,
    )
    assert result.stems_ingested == 6
    assert result.vocals_path.is_file()
    assert result.mix_path.is_file()


def test_prepare_from_multiple_premier_stems_and_instrumental_only(
    fresh_project: Project, tmp_path: Path
) -> None:
    zip_path = _make_suno_zip(tmp_path / "inst_only.zip", tmp_path, include_vocals=False)
    extra_kick = _write_kick_wav(tmp_path / "Kick.wav", duration=8.0)

    result = prepare_project(
        fresh_project,
        audio=[zip_path, extra_kick],
        separator=_FailIfCalledSeparator(),  # type: ignore[arg-type]
        transcriber=_FakeTranscriber(),  # type: ignore[arg-type]
        llm=None,
    )
    assert result.stems_ingested == 5
    assert result.mix_path.is_file()
    assert result.vocals_path.is_file()
    vocals, _ = sf.read(str(result.vocals_path), dtype="float32")
    assert float(np.max(np.abs(vocals))) < 1e-4


def test_cli_prepare_accepts_multiple_paths(
    fresh_project: Project, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    zip_path = _make_suno_zip(tmp_path / "suno.zip", tmp_path)
    master = _write_tone_wav(tmp_path / "song.wav", duration=8.0, freq=220.0)
    (fresh_project.root / "project.yaml").write_text(
        "title: Stem CLI\nllm:\n  enabled: false\n", encoding="utf-8"
    )
    monkeypatch.setattr(
        "mvdirector.audio.prepare.WhisperTranscriber",
        lambda **kwargs: _FakeTranscriber(),
    )

    runner = CliRunner()
    res = runner.invoke(
        cli_mod.app,
        ["prepare", str(master), str(zip_path), "--project", str(fresh_project.root)],
    )
    assert res.exit_code == 0, res.output
    assert "6 stems" in res.output


# --------------------------------------------------------------------------- analysis & planning


def test_analyze_project_populates_stem_arrangement_and_planner_uses_it(
    fresh_project: Project, tmp_path: Path
) -> None:
    zip_path = _make_suno_zip(tmp_path / "suno.zip", tmp_path)
    prepare_project(
        fresh_project,
        audio=zip_path,
        separator=_FailIfCalledSeparator(),  # type: ignore[arg-type]
        transcriber=_FakeTranscriber(),  # type: ignore[arg-type]
        llm=None,
    )

    analysis = analyze_project(fresh_project)
    assert isinstance(analysis, Analysis)
    assert analysis.stems is not None
    assert len(analysis.stems.stems) == 6
    roles = {s.role for s in analysis.stems.stems}
    assert {"lead_vocals", "backing_vocals", "drums", "bass", "synth", "brass"} <= roles
    assert len(analysis.stems.backing_vocal_windows) >= 1
    assert "Brass" in analysis.stems.summary
    assert "Drums" in analysis.stems.summary

    # Verify the planner feeds both the arrangement summary and per-shot stems to the LLM.
    llm = _RecordingLLM()
    lyrics_text = fresh_project.path(fresh_project.config.inputs.lyrics).read_text(encoding="utf-8")
    style_text = fresh_project.path(fresh_project.config.inputs.style).read_text(encoding="utf-8")
    build_manifest(fresh_project.config, analysis, lyrics_text, style_text, llm=llm)  # type: ignore[arg-type]

    assert any("ARRANGEMENT & INSTRUMENTATION:" in p for p in llm.prompts)
    assert any("stems=[" in p for p in llm.prompts)


def test_active_stems_in_span_filters_lead_vocals_and_orders_by_energy() -> None:
    arrangement = StemArrangement(
        stems=[
            StemTrackInfo(
                id="lead_vocals",
                name="Lead Vocals",
                role="lead_vocals",
                path="inputs/stems/lead_vocals.wav",
                active_fraction=0.8,
                energy_share=0.4,
                active_spans=[TimeSpan(start=0.0, end=10.0)],
            ),
            StemTrackInfo(
                id="drums",
                name="Drums",
                role="drums",
                path="inputs/stems/drums.wav",
                active_fraction=0.9,
                energy_share=0.35,
                active_spans=[TimeSpan(start=0.0, end=10.0)],
            ),
            StemTrackInfo(
                id="brass",
                name="Brass",
                role="brass",
                path="inputs/stems/brass.wav",
                active_fraction=0.4,
                energy_share=0.25,
                active_spans=[TimeSpan(start=5.0, end=10.0)],
            ),
        ]
    )
    assert stem_pack.active_stems_in_span(arrangement, 0.0, 4.0) == ["Drums"]
    assert stem_pack.active_stems_in_span(arrangement, 5.0, 9.0) == ["Drums", "Brass"]


class _RecordingSeparator:
    """Stands in for Demucs, recording whether a drum stem was requested of it."""

    def __init__(self) -> None:
        self.drums_requested: list[Path | None] = []

    def separate(self, mix_path: Path, vocals_out: Path, *, drums_out: Path | None = None):
        from mvdirector.audio.stems import StemResult

        self.drums_requested.append(drums_out)
        _write_tone_wav(vocals_out, freq=440.0, active_span=(1.0, 5.0), amplitude=0.5)
        if drums_out is not None:
            _write_kick_wav(drums_out, bpm=120.0, active_span=(0.0, 8.0))
        return StemResult(vocals=vocals_out, drums=drums_out)


def test_prepare_overwrite_keeps_the_stem_packs_own_drum_stem(
    fresh_project: Project, tmp_path: Path
) -> None:
    """`--overwrite` made Demucs clobber a drum stem the stem pack had just supplied.

    `need_drums_out` was set whenever `overwrite` was true, and `overwrite` short-circuited the
    `not drums_target.is_file()` guard — so with an external master mix plus a drums-but-no-vocals
    pack, Demucs overwrote the user's own drum stem written ten lines earlier and redid the work.
    """
    # A drum stem only exists when inputs.drums names one.
    fresh_project.config.inputs.drums = "inputs/drums.wav"
    stems_dir = fresh_project.path(fresh_project.config.inputs.stems_dir)
    stems_dir.mkdir(parents=True, exist_ok=True)
    _make_suno_zip(stems_dir / "pack.zip", tmp_path, include_vocals=False)
    master = _write_tone_wav(
        tmp_path / "master.wav", freq=220.0, active_span=(0.0, 8.0), amplitude=0.4
    )

    separator = _RecordingSeparator()
    result = prepare_project(
        fresh_project,
        audio=master,
        overwrite=True,
        separator=separator,  # type: ignore[arg-type]
        transcriber=_FakeTranscriber(),  # type: ignore[arg-type]
        llm=None,
    )

    # Demucs is still needed for the vocals the pack lacks, but must not be asked for drums.
    assert separator.drums_requested == [None], separator.drums_requested
    assert result.drums_path is not None and result.drums_path.is_file()

    # The surviving drum stem is the pack's kick, not a Demucs rewrite of the master tone.
    pack_drums, _ = sf.read(str(result.drums_path))
    assert float(np.abs(pack_drums).max()) > 0.0
