# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Audio-only project preparation: build stems, lyrics, and style brief from a single mix file.

Given only an ``.mp3`` or ``.wav`` file, this stage:
1. Ingests/normalizes the audio into ``inputs.mix`` (PCM WAV by default).
2. Separates ``inputs.vocals`` (and optional ``inputs.drums``) so lip-sync and vocal activity work.
3. Transcribes the isolated vocals with word-level timestamps.
4. Infers structured ``[Section]``-tagged ``inputs/lyrics.txt`` and a lip-sync-ready
   ``inputs/style.md`` from the transcribed lyrics and audio dynamics.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import soundfile as sf
from pydantic import BaseModel, ConfigDict

from mvdirector import media
from mvdirector import tuning as tuning_mod
from mvdirector.audio import analysis as analysis_mod
from mvdirector.audio import stem_pack as stem_pack_mod
from mvdirector.audio.lyrics import (
    Transcriber,
    Transcript,
    WhisperTranscriber,
    format_transcript_as_lyrics,
    parse_lyrics,
    segment_speaker_label,
    tokenize,
)
from mvdirector.audio.stems import (
    SUPPORTED_AUDIO_SUFFIXES,
    DemucsSeparator,
    StemSeparator,
    atomic_copy,
)
from mvdirector.errors import ConfigError
from mvdirector.llm import LLMClient, complete_json_with_optional_audio
from mvdirector.models import BOTH_VOICES, MusicalFeatures, TimeSpan, VocalTurn
from mvdirector.planning import prompts
from mvdirector.planning.brief import ensure_writable
from mvdirector.project import Project

log = logging.getLogger(__name__)

# Minimum token similarity between the model's lyrics and the transcript. Below it the model
# rewrote the song, which would break word-level lyric alignment, so the transcript is used.
MIN_LYRIC_FIDELITY = 0.85


class InferredInputsReply(BaseModel):
    model_config = ConfigDict(extra="ignore")
    lyrics: str
    music_style: str
    visual_direction: str


@dataclass(frozen=True)
class AudioSummary:
    duration: float
    tempo_bpm: float
    vocal_windows: list[TimeSpan]
    vocal_fraction: float
    energy_arc: list[str]
    vocal_turns: list[VocalTurn] = field(default_factory=list)
    musical_features: MusicalFeatures | None = None
    stem_summary: str = ""


@dataclass(frozen=True)
class PrepareResult:
    mix_path: Path
    vocals_path: Path
    drums_path: Path | None
    lyrics_path: Path
    style_path: Path
    word_timestamps_path: Path | None
    tempo_bpm: float
    duration: float
    vocal_windows: int
    transcribed_words: int
    inferred_with_llm: bool
    stems_ingested: int = 0


def _format_mmss(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    minutes, secs = divmod(total, 60)
    return f"{minutes}:{secs:02d}"


def _energy_label(value: float) -> str:
    if value >= 0.80:
        return "peak"
    if value >= 0.55:
        return "high"
    if value >= 0.30:
        return "moderate"
    return "calm"


def format_energy_arc(
    bars: list[float], intensities: list[float], duration: float, max_buckets: int = 8
) -> list[str]:
    """Compress per-bar intensities into a small list of timestamped energy labels."""
    if not bars or not intensities or duration <= 0:
        return []
    count = min(len(bars), len(intensities))
    edges = [*bars[:count], duration]
    bucket_size = max(1, (count + max_buckets - 1) // max_buckets)
    arc: list[str] = []
    for start_idx in range(0, count, bucket_size):
        end_idx = min(count, start_idx + bucket_size)
        mean_val = sum(intensities[start_idx:end_idx]) / (end_idx - start_idx)
        span_start = edges[start_idx]
        span_end = edges[end_idx]
        arc.append(
            f"{_format_mmss(span_start)}-{_format_mmss(span_end)} "
            f"{_energy_label(mean_val)} ({mean_val:.2f})"
        )
    return arc


def summarize_audio(
    mix: np.ndarray,
    vocals: np.ndarray,
    beats_per_bar: int,
    drums: np.ndarray | None = None,
    *,
    harmonic_mix: np.ndarray | None = None,
    stems: Sequence[stem_pack_mod.IngestedStem] = (),
    project_root: Path | None = None,
) -> AudioSummary:
    """Compute compact tempo, vocal coverage, and energy-arc cues for lyric/style inference."""
    duration = len(mix) / analysis_mod.ANALYSIS_SR
    beat_source = drums if drums is not None else mix
    tempo, beats = analysis_mod.detect_beats(beat_source)
    bars = analysis_mod.bars_from_beats(beats, beats_per_bar)
    mix_rms, mix_times = analysis_mod.rms_envelope(mix)
    vocal_rms, vocal_times = analysis_mod.rms_envelope(vocals)
    windows = analysis_mod.detect_vocal_windows(vocal_rms, vocal_times)
    vocal_turns = analysis_mod.detect_vocal_turns(vocals)
    intensities = analysis_mod.bar_intensity(mix_rms, mix_times, bars, duration)
    musical_features = analysis_mod.analyze_musical_features(
        mix, analysis_mod.ANALYSIS_SR, tempo_bpm=tempo, harmonic_mix=harmonic_mix
    )
    stem_summary = ""
    if stems and project_root is not None:
        arrangement = stem_pack_mod.analyze_stems(
            stems,
            project_root=project_root,
            bars=bars,
            duration=duration,
            sr=analysis_mod.ANALYSIS_SR,
            loader=analysis_mod.load_mono,
        )
        stem_summary = arrangement.summary if arrangement else ""
    vocal_seconds = sum(w.duration for w in windows)
    vocal_fraction = min(1.0, vocal_seconds / duration) if duration > 0 else 0.0
    return AudioSummary(
        duration=round(duration, 3),
        tempo_bpm=round(tempo, 2),
        vocal_windows=windows,
        vocal_fraction=round(vocal_fraction, 3),
        energy_arc=format_energy_arc(bars, intensities, duration),
        vocal_turns=vocal_turns,
        musical_features=musical_features,
        stem_summary=stem_summary,
    )


def fallback_style_brief(summary: AudioSummary, *, has_lyrics: bool) -> tuple[str, str]:
    """Deterministic style brief used when ``llm.enabled`` is false."""
    tempo_desc = (
        "upbeat, energetic"
        if summary.tempo_bpm >= 120
        else "mid-tempo, driving"
        if summary.tempo_bpm >= 90
        else "slow, atmospheric"
    )
    vocal_desc = (
        f"lead vocal track ({int(round(summary.vocal_fraction * 100))}% vocal coverage)"
        if has_lyrics
        else "instrumental track"
    )
    mf = summary.musical_features
    musical_bits: list[str] = []
    if mf is not None:
        if mf.key:
            musical_bits.append(f"in {mf.key}")
        musical_bits.append(f"{mf.brightness} timbre")
        musical_bits.append(f"{mf.percussiveness} texture")
    musical_clause = f", {', '.join(musical_bits)}" if musical_bits else ""
    music_style = (
        f"{tempo_desc.capitalize()} contemporary track at ~{summary.tempo_bpm:.0f} BPM "
        f"({summary.duration:.0f}s){musical_clause}, {vocal_desc}."
    )
    if has_lyrics:
        visual_direction = (
            "Solo vocalist performing on camera in cinematic, moody lighting with clear, "
            "unobstructed close-up and medium-close lip-sync shots during sung hooks and verses "
            "(no masks or foreground obstructions over the mouth), intercut with atmospheric "
            "cutaways that match the song's energy arc."
        )
    else:
        visual_direction = (
            "Cinematic instrumental visual piece driven by lighting, architecture, movement, and "
            "atmosphere synced to the track's tempo and dynamics."
        )
    return music_style, visual_direction


def _build_inference_prompt(transcript: Transcript, summary: AudioSummary) -> str:
    header = [
        f"DURATION: {summary.duration:.1f}s ({_format_mmss(summary.duration)})",
        f"TEMPO: ~{summary.tempo_bpm:.0f} BPM",
        f"VOCAL COVERAGE: {int(round(summary.vocal_fraction * 100))}% "
        f"({len(summary.vocal_windows)} vocal windows)",
    ]
    if transcript.multilingual and transcript.language:
        header.append(f"DETECTED LANGUAGE: multilingual (primary: {transcript.language})")
    elif transcript.multilingual:
        header.append("DETECTED LANGUAGE: multilingual")
    elif transcript.language:
        header.append(f"DETECTED LANGUAGE: {transcript.language}")
    if summary.energy_arc:
        header.append("ENERGY ARC: " + ", ".join(summary.energy_arc))
    if summary.musical_features and summary.musical_features.summary:
        header.append(f"MUSICAL CHARACTER: {summary.musical_features.summary}")
    if summary.stem_summary:
        header.append(f"INSTRUMENTATION & STEMS: {summary.stem_summary}")
    voices = {t.voice for t in summary.vocal_turns}
    if len(voices) >= 2 or BOTH_VOICES in voices:
        both_note = (
            ", and simultaneous duet sections like [Chorus: Both]" if BOTH_VOICES in voices else ""
        )
        header.append(
            "DETECTED VOICES: 2 distinct vocalists (Singer 1 lower register, Singer 2 higher "
            f"register — tag alternating sections like [Verse 1: Singer 1]{both_note})"
        )

    if transcript.segments:
        lines = []
        for seg in transcript.segments:
            if not seg.text.strip():
                continue
            speaker = segment_speaker_label(seg.start, seg.end, summary.vocal_turns)
            tag = f" ({speaker})" if speaker else ""
            lines.append(
                f"[{_format_mmss(seg.start)} - {_format_mmss(seg.end)}]{tag} {seg.text.strip()}"
            )
        body = "TRANSCRIBED VOCALS:\n" + "\n".join(lines)
    else:
        body = "TRANSCRIBED VOCALS:\n(No intelligible vocals detected; treat as instrumental.)"

    return "\n".join(header) + "\n\n" + body


def lyric_fidelity(lyrics: str, transcript: Transcript) -> float:
    """Share (0-1) of words that the lyrics and the transcript have in common, in order."""
    raw_sung = " ".join(seg.text for seg in transcript.segments).strip()
    raw_written = " ".join(line.text for line in parse_lyrics(lyrics)).strip()
    sung = tokenize(raw_sung)
    written = tokenize(raw_written)
    if not sung and not written:
        if not raw_sung and not raw_written:
            return 1.0
        return SequenceMatcher(
            a="".join(raw_sung.split()), b="".join(raw_written.split()), autojunk=False
        ).ratio()
    return SequenceMatcher(a=sung, b=written, autojunk=False).ratio()


def infer_inputs_from_transcript(
    transcript: Transcript,
    summary: AudioSummary,
    *,
    llm: LLMClient | None,
    audio_path: Path | None = None,
) -> tuple[InferredInputsReply, bool]:
    """Infer structured ``lyrics.txt`` and ``style.md`` from transcribed vocals + audio summary."""
    has_lyrics = any(seg.text.strip() for seg in transcript.segments)
    turns = summary.vocal_turns or None
    if llm is None:
        music_style, visual_direction = fallback_style_brief(summary, has_lyrics=has_lyrics)
        return (
            InferredInputsReply(
                lyrics=format_transcript_as_lyrics(transcript.segments, vocal_turns=turns),
                music_style=music_style,
                visual_direction=visual_direction,
            ),
            False,
        )

    reply = complete_json_with_optional_audio(
        llm,
        system=prompts.INFER_FROM_LYRICS_SYSTEM,
        text=_build_inference_prompt(transcript, summary),
        schema=InferredInputsReply,
        audio_path=audio_path,
    )
    lyrics_text = reply.lyrics.strip()
    if not lyrics_text:
        lyrics_text = format_transcript_as_lyrics(transcript.segments, vocal_turns=turns).strip()
    else:
        fidelity = lyric_fidelity(lyrics_text, transcript)
        if fidelity < MIN_LYRIC_FIDELITY:
            log.warning(
                "The model's lyrics match only %.0f%% of the transcript; using the transcript.",
                fidelity * 100,
            )
            lyrics_text = format_transcript_as_lyrics(
                transcript.segments, vocal_turns=turns
            ).strip()
    if not reply.music_style.strip() or not reply.visual_direction.strip():
        raise ConfigError("The model returned an empty music_style or visual_direction.")
    return (
        InferredInputsReply(
            lyrics=lyrics_text,
            music_style=reply.music_style.strip(),
            visual_direction=reply.visual_direction.strip(),
        ),
        True,
    )


def _target_file(project: Project, relative: str | None, label: str) -> tuple[str, Path]:
    if not relative or not relative.strip():
        raise ConfigError(f"inputs.{label} is empty in project.yaml.")
    cleaned = relative.strip()
    target = project.path(cleaned)
    if target == project.root or target.is_dir():
        raise ConfigError(f"inputs.{label} must name a file inside {project.root}: {relative!r}")
    return cleaned, target


def _validate_audio_file(path: Path) -> Path:
    if not path.is_file():
        raise ConfigError(f"Audio file not found: {path}")
    if path.suffix.lower() not in SUPPORTED_AUDIO_SUFFIXES:
        supported = ", ".join(sorted(SUPPORTED_AUDIO_SUFFIXES))
        raise ConfigError(
            f"Unsupported audio format {path.suffix!r} for {path}. Supported formats: {supported}."
        )
    return path


def resolve_source_audio(project: Project, audio: Path | None) -> Path:
    """Locate and validate the source .mp3 or .wav file for preparation."""
    if audio is not None:
        return _validate_audio_file(audio)

    _, mix_target = _target_file(project, project.config.inputs.mix, "mix")
    if mix_target.is_file():
        return _validate_audio_file(mix_target)

    if mix_target.suffix.lower() == ".wav":
        sibling_mp3 = mix_target.with_suffix(".mp3")
        if sibling_mp3.is_file():
            return _validate_audio_file(sibling_mp3)

    raise ConfigError(
        f"No source audio found at {mix_target} (or {mix_target.with_suffix('.mp3')}). "
        "Pass an .mp3 or .wav file: `mvd prepare /path/to/song.mp3`."
    )


def _ingest_mix(src: Path, dst: Path, *, overwrite: bool) -> Path:
    if src.resolve() == dst.resolve():
        return dst
    if dst.is_file() and not overwrite:
        raise ConfigError(
            f"Existing mix at {dst} would be overwritten. Pass --overwrite to replace."
        )
    if dst.suffix.lower() == ".wav":
        return media.transcode_to_wav(src, dst)
    if src.suffix.lower() == dst.suffix.lower():
        return atomic_copy(src, dst)
    raise ConfigError(
        f"Cannot convert {src.suffix!r} into inputs.mix target {dst.name!r}; "
        "set inputs.mix to a .wav path (e.g. inputs/mix.wav)."
    )


_MIX_STEM_NAMES = frozenset({"mix", "master", "full mix", "full_mix", "song", "track"})


def _is_explicit_mix_name(path: Path) -> bool:
    return stem_pack_mod.clean_stem_label(path.name).lower() in _MIX_STEM_NAMES


def _discover_dropped_stem_sources(project: Project, stems_dir: Path) -> list[Path]:
    """Find .zip archives or stem files dropped into ``inputs/stems/`` or ``inputs/*.zip``."""
    found: list[Path] = []
    if stems_dir.is_dir():
        for entry in sorted(stems_dir.iterdir()):
            if entry.is_symlink() or not entry.is_file():
                continue
            if entry.suffix.lower() in (
                stem_pack_mod.ARCHIVE_SUFFIXES | stem_pack_mod.STEM_AUDIO_SUFFIXES
            ):
                found.append(entry)
    inputs_dir = project.root / "inputs"
    if inputs_dir.is_dir():
        for entry in sorted(inputs_dir.iterdir()):
            if (
                not entry.is_symlink()
                and entry.is_file()
                and entry.suffix.lower() in stem_pack_mod.ARCHIVE_SUFFIXES
            ):
                found.append(entry)
    return found


def _partition_prepare_inputs(
    project: Project,
    audio: Path | Sequence[Path] | None,
    stems_dir: Path,
) -> tuple[Path | None, list[Path]]:
    """Split prepare inputs into ``(master_mix_path, stem_sources)``.

    Preserves classic single-file behavior when one audio file is passed with no stem packs.
    When any .zip archive, directory, or multiple audio files are given (or dropped into
    ``inputs/stems/`` / ``inputs/*.zip``), classifies mix vs. stem sources automatically.
    """
    if audio is None:
        raw_paths: list[Path] = []
    elif isinstance(audio, Path):
        raw_paths = [audio]
    else:
        raw_paths = list(audio)

    for p in raw_paths:
        if not p.exists():
            raise ConfigError(f"Audio or stem file not found: {p}")

    # Single explicit audio file with no zip/dir: classic single-mix mode unless stems_dir has zips.
    if (
        len(raw_paths) == 1
        and raw_paths[0].is_file()
        and raw_paths[0].suffix.lower() not in stem_pack_mod.ARCHIVE_SUFFIXES
    ):
        dropped = _discover_dropped_stem_sources(project, stems_dir)
        return _validate_audio_file(raw_paths[0]), dropped

    # No explicit args: check for dropped stem packs + optional existing mix.
    if not raw_paths:
        dropped = _discover_dropped_stem_sources(project, stems_dir)
        if not dropped:
            return resolve_source_audio(project, None), []
        _, mix_target = _target_file(project, project.config.inputs.mix, "mix")
        existing_mix: Path | None = None
        if mix_target.is_file():
            existing_mix = mix_target
        elif mix_target.suffix.lower() == ".wav" and mix_target.with_suffix(".mp3").is_file():
            existing_mix = mix_target.with_suffix(".mp3")
        return existing_mix, dropped

    # Multiple paths, or at least one .zip / directory.
    pack_sources: list[Path] = []
    loose_audio: list[Path] = []
    for p in raw_paths:
        if p.is_dir() or p.suffix.lower() in stem_pack_mod.ARCHIVE_SUFFIXES:
            pack_sources.append(p)
        elif p.suffix.lower() in stem_pack_mod.STEM_AUDIO_SUFFIXES:
            loose_audio.append(p)
        else:
            supported = ", ".join(
                sorted(stem_pack_mod.STEM_AUDIO_SUFFIXES | stem_pack_mod.ARCHIVE_SUFFIXES)
            )
            raise ConfigError(
                f"Unsupported input {p.name!r}. Supported formats: {supported} or a folder."
            )

    master_mix: Path | None = None
    remaining_loose: list[Path] = []
    if len(loose_audio) == 1 and pack_sources:
        candidate = loose_audio[0]
        if (
            _is_explicit_mix_name(candidate)
            or stem_pack_mod.classify_stem_name(candidate.name) == "other"
        ):
            master_mix = candidate
        else:
            remaining_loose.append(candidate)
    else:
        for candidate in loose_audio:
            if master_mix is None and _is_explicit_mix_name(candidate):
                master_mix = candidate
            else:
                remaining_loose.append(candidate)

    return master_mix, [*pack_sources, *remaining_loose]


def _write_silent_wav(
    dst: Path, duration_seconds: float, sr: int = analysis_mod.ANALYSIS_SR
) -> Path:
    """Write a silent mono PCM16 WAV of ``duration_seconds`` atomically (instrumental fallback)."""
    n_samples = max(1, int(round(max(duration_seconds, 0.1) * sr)))
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".mvd_sil_", suffix=".wav", dir=dst.parent)
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        sf.write(str(tmp_path), np.zeros(n_samples, dtype=np.float32), sr, subtype="PCM_16")
        os.replace(tmp_path, dst)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    return dst


def _target_dir(project: Project, relative: str | None, label: str) -> Path:
    if not relative or not relative.strip():
        raise ConfigError(f"inputs.{label} is empty in project.yaml.")
    cleaned = relative.strip()
    target = project.path(cleaned)
    if target == project.root or target.is_file():
        raise ConfigError(
            f"inputs.{label} must name a subfolder inside {project.root}: {relative!r}"
        )
    return target


def prepare_project(
    project: Project,
    audio: Path | Sequence[Path] | None = None,
    *,
    overwrite: bool = False,
    separator: StemSeparator | None = None,
    transcriber: Transcriber | None = None,
    llm: LLMClient | None = None,
) -> PrepareResult:
    """Build mix, vocal stem, lyrics.txt, and style.md from audio or Suno/DAW stem packs."""
    inputs = project.config.inputs
    prepare_cfg = project.config.prepare

    # Fail fast before running stem extraction, separation, transcription, or LLM calls.
    lyrics_rel, style_rel = ensure_writable(project, overwrite=overwrite)
    _, mix_target = _target_file(project, inputs.mix, "mix")
    _, vocals_target = _target_file(project, inputs.vocals, "vocals")
    drums_target = _target_file(project, inputs.drums, "drums")[1] if inputs.drums else None
    stems_dir = _target_dir(project, inputs.stems_dir, "stems_dir")
    timestamps_rel = (
        _target_file(project, inputs.word_timestamps, "word_timestamps")[0]
        if inputs.word_timestamps
        else None
    )

    master_mix_src, stem_sources = _partition_prepare_inputs(project, audio, stems_dir)
    if (
        master_mix_src is not None
        and master_mix_src.resolve() != mix_target.resolve()
        and mix_target.is_file()
        and not overwrite
    ):
        raise ConfigError(
            f"Existing mix at {mix_target} would be overwritten. Pass --overwrite to replace."
        )

    # 1. Extract/ingest any stem packs (.zip, folder, or loose stems) into inputs/stems/.
    ingested_stems: list[stem_pack_mod.IngestedStem] = []
    if stem_sources:
        ingested_stems = stem_pack_mod.ingest_stem_sources(
            stem_sources, stems_dir, overwrite=overwrite
        )
    if not ingested_stems:
        ingested_stems = stem_pack_mod.discover_stems(stems_dir)

    # 2. Ingest/normalize mix into inputs.mix (or synthesize from stems if no master mix given).
    synthesized_mix = False
    if master_mix_src is not None:
        mix_path = _ingest_mix(master_mix_src, mix_target, overwrite=overwrite)
    elif mix_target.is_file() and not overwrite:
        mix_path = mix_target
    elif ingested_stems:
        mix_stems = stem_pack_mod.select_mix_stems(ingested_stems)
        log.info(
            "No master mix provided; synthesizing %s from %d stem(s).",
            mix_target.name,
            len(mix_stems),
        )
        mix_path = stem_pack_mod.mix_stems_to_wav([s.path for s in mix_stems], mix_target)
        synthesized_mix = True
    else:
        raise ConfigError(
            f"No source audio or stem pack found at {mix_target} or {stems_dir}. "
            "Pass an .mp3/.wav file or a Suno stem .zip: `mvd prepare /path/to/stems.zip`."
        )

    # 3. Route vocal stem (preferring lead vocals from stem pack) and optional drum stem.
    pack_vocals = stem_pack_mod.select_vocal_stems(ingested_stems)
    pack_drums = stem_pack_mod.select_drum_stems(ingested_stems)

    vocals_path: Path
    drums_path: Path | None = drums_target if (drums_target and drums_target.is_file()) else None

    if pack_vocals and (overwrite or not vocals_target.is_file()):
        log.info(
            "Using %s from stem pack for %s (skipping Demucs).",
            ", ".join(s.name for s in pack_vocals),
            vocals_target.name,
        )
        vocals_path = stem_pack_mod.mix_stems_to_wav([s.path for s in pack_vocals], vocals_target)
        if drums_target is not None and pack_drums and (overwrite or not drums_target.is_file()):
            drums_path = stem_pack_mod.mix_stems_to_wav([s.path for s in pack_drums], drums_target)
    elif ingested_stems and not pack_vocals and synthesized_mix:
        # Instrumental-only stem pack (no vocals stem and no external master mix).
        mix_info = sf.info(str(mix_path))
        log.info(
            "Stem pack has no vocal stem; writing silent %s for instrumental track.",
            vocals_target.name,
        )
        vocals_path = _write_silent_wav(vocals_target, mix_info.duration)
        if drums_target is not None and pack_drums and (overwrite or not drums_target.is_file()):
            drums_path = stem_pack_mod.mix_stems_to_wav([s.path for s in pack_drums], drums_target)
    else:
        if drums_target is not None and pack_drums and (overwrite or not drums_target.is_file()):
            drums_path = stem_pack_mod.mix_stems_to_wav([s.path for s in pack_drums], drums_target)
        need_stems = (
            overwrite
            or not vocals_target.is_file()
            or (drums_target is not None and not drums_target.is_file())
        )
        if need_stems:
            sep = separator or DemucsSeparator(
                model=prepare_cfg.stem_model, device=prepare_cfg.stem_device
            )
            need_drums_out = drums_target if (drums_target and not drums_target.is_file()) else None
            stem_result = sep.separate(mix_path, vocals_target, drums_out=need_drums_out)
            vocals_path = stem_result.vocals
            if stem_result.drums is not None:
                drums_path = stem_result.drums
        else:
            log.info("Using existing vocal stem at %s", vocals_target)
            vocals_path = vocals_target

    # 4. Extract tempo/energy/vocal summary and transcribe the isolated vocals.
    mix_samples = analysis_mod.load_mono(mix_path)
    vocal_samples = analysis_mod.load_mono(vocals_path)
    if drums_path is not None and drums_path.is_file():
        drum_samples: np.ndarray | None = analysis_mod.load_mono(drums_path)
    else:
        rhythm_stems = stem_pack_mod.select_rhythm_stems(ingested_stems)
        drum_samples = (
            stem_pack_mod.load_stems_mono_sum(
                rhythm_stems, analysis_mod.ANALYSIS_SR, loader=analysis_mod.load_mono
            )
            if rhythm_stems
            else None
        )
    harmonic_stems = stem_pack_mod.select_harmonic_stems(ingested_stems)
    harmonic_samples = (
        stem_pack_mod.load_stems_mono_sum(
            harmonic_stems, analysis_mod.ANALYSIS_SR, loader=analysis_mod.load_mono
        )
        if harmonic_stems
        else None
    )
    summary = summarize_audio(
        mix_samples,
        vocal_samples,
        project.config.video.beats_per_bar,
        drums=drum_samples,
        harmonic_mix=harmonic_samples,
        stems=ingested_stems,
        project_root=project.root,
    )

    if transcriber is not None:
        trans = transcriber
    else:
        lyrics_hint, style_hint = tuning_mod.read_optional_project_inputs(project)
        hint_src = master_mix_src or (stem_sources[0] if stem_sources else mix_path)
        stem_hint = "" if hint_src.stem.lower() == "mix" else hint_src.stem
        title_hint = (
            f"{project.config.title} {stem_hint}".strip()
            if project.config.title.strip().lower() not in ("", "my song", "untitled")
            else stem_hint
        )
        effective_prepare, whisper_adjustments = tuning_mod.tune_prepare_config(
            project.config,
            lyrics_text=lyrics_hint,
            style_text=style_hint,
            title_hint=title_hint,
        )
        for adj in whisper_adjustments:
            log.info(
                "Auto-tuned %s: %r -> %r (%s)",
                adj.key,
                adj.current,
                adj.recommended,
                adj.reason,
            )
        trans = WhisperTranscriber(
            model_size=effective_prepare.whisper_model,
            language=effective_prepare.whisper_language,
            multilingual=effective_prepare.whisper_multilingual,
        )
    transcript = trans.transcribe(vocals_path)
    if not transcript.words:
        log.warning(
            "No words were transcribed from %s; lip-sync shots will rely on vocal activity.",
            vocals_path,
        )

    # 5. Infer structured lyrics.txt and lip-sync-ready style.md from the transcript.
    inferred, used_llm = infer_inputs_from_transcript(
        transcript, summary, llm=llm, audio_path=mix_path
    )

    # 6. Persist generated text inputs atomically under the project lock.
    lyrics_text = inferred.lyrics.strip() + "\n"
    style_text = (
        f"Music style: {inferred.music_style.strip()}\n\n"
        f"Visual direction: {inferred.visual_direction.strip()}\n"
    )
    with project.run_lock():
        ensure_writable(project, overwrite=overwrite)
        lyrics_path = project.write_text(lyrics_rel, lyrics_text)
        style_path = project.write_text(style_rel, style_text)
        word_timestamps_path: Path | None = None
        if timestamps_rel is not None:
            payload = json.dumps(
                [w.model_dump() for w in transcript.words], indent=2, ensure_ascii=False
            )
            word_timestamps_path = project.write_text(timestamps_rel, payload + "\n")
        project.storage.push()

    return PrepareResult(
        mix_path=mix_path,
        vocals_path=vocals_path,
        drums_path=drums_path,
        lyrics_path=lyrics_path,
        style_path=style_path,
        word_timestamps_path=word_timestamps_path,
        tempo_bpm=summary.tempo_bpm,
        duration=summary.duration,
        vocal_windows=len(summary.vocal_windows),
        transcribed_words=len(transcript.words),
        inferred_with_llm=used_llm,
        stems_ingested=len(ingested_stems),
    )
