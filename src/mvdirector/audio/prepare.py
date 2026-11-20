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
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict

from mvdirector import media
from mvdirector.audio import analysis as analysis_mod
from mvdirector.audio.lyrics import (
    Transcriber,
    Transcript,
    WhisperTranscriber,
    format_transcript_as_lyrics,
    parse_lyrics,
    tokenize,
)
from mvdirector.audio.stems import (
    SUPPORTED_AUDIO_SUFFIXES,
    DemucsSeparator,
    StemSeparator,
    atomic_copy,
)
from mvdirector.errors import ConfigError
from mvdirector.llm import LLMClient, text_block
from mvdirector.models import TimeSpan
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
) -> AudioSummary:
    """Compute compact tempo, vocal coverage, and energy-arc cues for lyric/style inference."""
    duration = len(mix) / analysis_mod.ANALYSIS_SR
    beat_source = drums if drums is not None else mix
    tempo, beats = analysis_mod.detect_beats(beat_source)
    bars = analysis_mod.bars_from_beats(beats, beats_per_bar)
    mix_rms, mix_times = analysis_mod.rms_envelope(mix)
    vocal_rms, vocal_times = analysis_mod.rms_envelope(vocals)
    windows = analysis_mod.detect_vocal_windows(vocal_rms, vocal_times)
    intensities = analysis_mod.bar_intensity(mix_rms, mix_times, bars, duration)
    vocal_seconds = sum(w.duration for w in windows)
    vocal_fraction = min(1.0, vocal_seconds / duration) if duration > 0 else 0.0
    return AudioSummary(
        duration=round(duration, 3),
        tempo_bpm=round(tempo, 2),
        vocal_windows=windows,
        vocal_fraction=round(vocal_fraction, 3),
        energy_arc=format_energy_arc(bars, intensities, duration),
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
    music_style = (
        f"{tempo_desc.capitalize()} contemporary track at ~{summary.tempo_bpm:.0f} BPM "
        f"({summary.duration:.0f}s), {vocal_desc}."
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
    if transcript.language:
        header.append(f"DETECTED LANGUAGE: {transcript.language}")
    if summary.energy_arc:
        header.append("ENERGY ARC: " + ", ".join(summary.energy_arc))

    if transcript.segments:
        lines = [
            f"[{_format_mmss(seg.start)} - {_format_mmss(seg.end)}] {seg.text.strip()}"
            for seg in transcript.segments
            if seg.text.strip()
        ]
        body = "TRANSCRIBED VOCALS:\n" + "\n".join(lines)
    else:
        body = "TRANSCRIBED VOCALS:\n(No intelligible vocals detected; treat as instrumental.)"

    return "\n".join(header) + "\n\n" + body


def lyric_fidelity(lyrics: str, transcript: Transcript) -> float:
    """Share (0-1) of words that the lyrics and the transcript have in common, in order."""
    sung = tokenize(" ".join(seg.text for seg in transcript.segments))
    written = tokenize(" ".join(line.text for line in parse_lyrics(lyrics)))
    if not sung and not written:
        return 1.0
    return SequenceMatcher(a=sung, b=written, autojunk=False).ratio()


def infer_inputs_from_transcript(
    transcript: Transcript,
    summary: AudioSummary,
    *,
    llm: LLMClient | None,
) -> tuple[InferredInputsReply, bool]:
    """Infer structured ``lyrics.txt`` and ``style.md`` from transcribed vocals + audio summary."""
    has_lyrics = any(seg.text.strip() for seg in transcript.segments)
    if llm is None:
        music_style, visual_direction = fallback_style_brief(summary, has_lyrics=has_lyrics)
        return (
            InferredInputsReply(
                lyrics=format_transcript_as_lyrics(transcript.segments),
                music_style=music_style,
                visual_direction=visual_direction,
            ),
            False,
        )

    reply = llm.complete_json(
        system=prompts.INFER_FROM_LYRICS_SYSTEM,
        content=[text_block(_build_inference_prompt(transcript, summary))],
        schema=InferredInputsReply,
    )
    lyrics_text = reply.lyrics.strip()
    if not lyrics_text:
        lyrics_text = format_transcript_as_lyrics(transcript.segments).strip()
    else:
        fidelity = lyric_fidelity(lyrics_text, transcript)
        if fidelity < MIN_LYRIC_FIDELITY:
            log.warning(
                "The model's lyrics match only %.0f%% of the transcript; using the transcript.",
                fidelity * 100,
            )
            lyrics_text = format_transcript_as_lyrics(transcript.segments).strip()
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


def prepare_project(
    project: Project,
    audio: Path | None = None,
    *,
    overwrite: bool = False,
    separator: StemSeparator | None = None,
    transcriber: Transcriber | None = None,
    llm: LLMClient | None = None,
) -> PrepareResult:
    """Build mix, vocal stem, lyrics.txt, and style.md from a single .mp3 or .wav file."""
    inputs = project.config.inputs
    prepare_cfg = project.config.prepare

    # Fail fast before running stem separation, transcription, or LLM calls.
    lyrics_rel, style_rel = ensure_writable(project, overwrite=overwrite)
    _, mix_target = _target_file(project, inputs.mix, "mix")
    _, vocals_target = _target_file(project, inputs.vocals, "vocals")
    drums_target = _target_file(project, inputs.drums, "drums")[1] if inputs.drums else None
    timestamps_rel = (
        _target_file(project, inputs.word_timestamps, "word_timestamps")[0]
        if inputs.word_timestamps
        else None
    )

    source_audio = resolve_source_audio(project, audio)
    if source_audio.resolve() != mix_target.resolve() and mix_target.is_file() and not overwrite:
        raise ConfigError(
            f"Existing mix at {mix_target} would be overwritten. Pass --overwrite to replace."
        )

    # 1. Ingest/normalize mix into inputs.mix.
    mix_path = _ingest_mix(source_audio, mix_target, overwrite=overwrite)

    # 2. Separate vocal stem (and optional drum stem) for lip-sync and vocal activity.
    need_stems = (
        overwrite
        or not vocals_target.is_file()
        or (drums_target is not None and not drums_target.is_file())
    )
    if need_stems:
        sep = separator or DemucsSeparator(
            model=prepare_cfg.stem_model, device=prepare_cfg.stem_device
        )
        stem_result = sep.separate(mix_path, vocals_target, drums_out=drums_target)
        vocals_path, drums_path = stem_result.vocals, stem_result.drums
    else:
        log.info("Using existing vocal stem at %s", vocals_target)
        vocals_path, drums_path = vocals_target, drums_target

    # 3. Extract tempo/energy/vocal summary and transcribe the isolated vocals.
    mix_samples = analysis_mod.load_mono(mix_path)
    vocal_samples = analysis_mod.load_mono(vocals_path)
    drum_samples = analysis_mod.load_mono(drums_path) if drums_path else None
    summary = summarize_audio(
        mix_samples, vocal_samples, project.config.video.beats_per_bar, drums=drum_samples
    )

    trans = transcriber or WhisperTranscriber(
        model_size=prepare_cfg.whisper_model, language=prepare_cfg.whisper_language
    )
    transcript = trans.transcribe(vocals_path)
    if not transcript.words:
        log.warning(
            "No words were transcribed from %s; lip-sync shots will rely on vocal activity.",
            vocals_path,
        )

    # 4. Infer structured lyrics.txt and lip-sync-ready style.md from the transcript.
    inferred, used_llm = infer_inputs_from_transcript(transcript, summary, llm=llm)

    # 5. Persist generated text inputs atomically under the project lock.
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
    )
