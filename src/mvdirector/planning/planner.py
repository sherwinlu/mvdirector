# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Build the shot manifest: deterministic timing plus creative direction.

Creative direction comes from the language model when one is available. Without it,
a rule-based director fills the same fields so the pipeline still runs end to end.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
import statistics
import unicodedata
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

from mvdirector import tuning as tuning_mod
from mvdirector.audio.analysis import describe_vocal_register
from mvdirector.audio.cues import format_cues_summary
from mvdirector.audio.lyrics import (
    is_hook_section,
    is_syllabic_char,
    labels_have_multiple_singers,
    parse_lyrics,
    section_singers,
)
from mvdirector.audio.stem_pack import active_stems_in_span
from mvdirector.config import ProjectConfig
from mvdirector.errors import LLMError, LLMUnavailableError
from mvdirector.llm import LLMClient, complete_json_with_optional_audio, text_block
from mvdirector.models import (
    BOTH_VOICES,
    Analysis,
    AssetState,
    AudioSlice,
    Character,
    Framing,
    Manifest,
    PropDesign,
    SectionBeat,
    SetDesign,
    Shot,
    ShotType,
    SongInterpretation,
    TimeSpan,
    VideoStyle,
    VisualBible,
    WardrobeState,
    ascii_fold,
    normalize_exclusions,
    validate_safe_id,
)
from mvdirector.planning import prompts
from mvdirector.planning.segmenter import (
    dominant_voice,
    frames_for,
    segment,
    voice_activity_in_span,
)

log = logging.getLogger(__name__)

_LIPSYNC_FRAMINGS = [Framing.CLOSE_UP, Framing.MEDIUM_CLOSE]
# Most cutaways in a row before the singer has to come back.
_MAX_CUTAWAY_RUN = 2
_SHARE_EPS = 1e-9
# Shot scale for shots with no singing to frame, by pace: the closer the lens, the faster it feels.
_PACE_FRAMING = {"calm": Framing.WIDE, "moderate": Framing.MEDIUM, "high": Framing.MEDIUM_CLOSE}
_PACE_MOTION = {
    "calm": "Slow, steady camera drift. Little movement in the frame.",
    "moderate": "Steady camera with a gentle push-in. Natural movement in the frame.",
    "high": "Brisk movement in the frame and a firm, smooth push-in. Camera stays stable.",
}
_BROLL_PACE_MOTION = {
    "calm": "Light, dust and shadow drift slowly through the uninhabited space. Steady camera.",
    "moderate": (
        "Light and shadow shift steadily across the empty, uninhabited space with natural "
        "movement. Gentle push-in."
    ),
    "high": (
        "Brisk movement of light and environmental elements across the empty, uninhabited space. "
        "Firm, smooth push-in. Camera stays stable."
    ),
}
_SCALE = {Framing.WIDE: 0, Framing.MEDIUM: 1, Framing.MEDIUM_CLOSE: 2, Framing.CLOSE_UP: 3}
_TIGHT_TO_WIDE = [Framing.CLOSE_UP, Framing.MEDIUM_CLOSE, Framing.MEDIUM, Framing.WIDE]
_FRAMING_PHRASE = {
    Framing.CLOSE_UP: "Close-up",
    Framing.MEDIUM_CLOSE: "Medium close-up",
    Framing.MEDIUM: "Medium shot",
    Framing.WIDE: "Wide shot",
}
_CONTEXT_SHOTS = 4  # already-directed shots shown to the model for continuity


class _Reply(BaseModel):
    model_config = ConfigDict(extra="ignore")


def _slug_id(value: object, label: str = "id", prefix: str = "id") -> str:
    if isinstance(value, str):
        cleaned = unicodedata.normalize("NFKC", value).strip()
        if (
            cleaned
            and cleaned[0].isalnum()
            and ".." not in cleaned
            and "/" not in cleaned
            and "\\" not in cleaned
        ):
            folded = ascii_fold(cleaned).lower()
            slug = re.sub(r"[^a-z0-9_-]+", "_", folded).strip("_-")
            if slug and slug[0].isalnum():
                if len(slug) <= 64:
                    return slug
                digest = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()[:6]
                return f"{slug[:57].rstrip('_-') or prefix}_{digest}"
            digest = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()[:6]
            return f"{prefix}_{digest}"
    return validate_safe_id(str(value), label)


def _unique_id(base: str, used: set[str]) -> str:
    candidate = base
    suffix = 2
    while candidate in used:
        tail = f"_{suffix}"
        candidate = f"{base[: 64 - len(tail)].rstrip('_-') or 'id'}{tail}"
        suffix += 1
    used.add(candidate)
    return candidate


class WardrobeReply(_Reply):
    id: str
    description: str

    @field_validator("id", mode="before")
    @classmethod
    def _check_id(cls, value: object) -> str:
        return _slug_id(value, "wardrobe state id", "state")


class CharacterReply(_Reply):
    id: str
    name: str = ""
    appearance: str
    invariants: str = ""
    wardrobe: list[WardrobeReply] = Field(default_factory=list)

    @field_validator("id", mode="before")
    @classmethod
    def _check_id(cls, value: object) -> str:
        return _slug_id(value, "character id", "char")

    @field_validator("invariants", "name", mode="before")
    @classmethod
    def _coerce_char_text(cls, value: object) -> object:
        return _coerce_text(value)


class SetReply(_Reply):
    name: str
    sections: list[str] = Field(default_factory=list)
    description: str = ""
    movement: str = ""
    parent_set: str | None = None


class PropReply(_Reply):
    id: str
    name: str
    description: str = ""
    sections: list[str] = Field(default_factory=list)

    @field_validator("id", mode="before")
    @classmethod
    def _check_id(cls, value: object) -> str:
        return _slug_id(value, "prop id", "prop")


def _none_to_str(value: object) -> object:
    return "" if value is None else value


def _coerce_text(value: object, sep: str = ", ") -> object:
    """Accept null or a list of strings where a prose/comma-separated string is expected."""
    if value is None:
        return ""
    if isinstance(value, list) and all(item is None or isinstance(item, str) for item in value):
        return sep.join(item.strip() for item in value if isinstance(item, str) and item.strip())
    return value


def _none_to_list(value: object) -> object:
    if value is None:
        return []
    if isinstance(value, list):
        return [item for item in value if item is not None]
    return value


class AudioListeningNotes(_Reply):
    """Structured musical-feel observations heard directly from the track's MP3 audio."""

    meter_and_groove: str = ""
    instrumentation_and_texture: str = ""
    vocal_character: str = ""
    atmosphere_and_subtext: str = ""
    key_musical_turns: list[str] = Field(default_factory=list)

    @field_validator(
        "meter_and_groove",
        "instrumentation_and_texture",
        "vocal_character",
        "atmosphere_and_subtext",
        mode="before",
    )
    @classmethod
    def _coerce_listen_text(cls, value: object) -> object:
        return _coerce_text(value)

    @field_validator("key_musical_turns", mode="before")
    @classmethod
    def _coerce_listen_turns(cls, value: object) -> object:
        return _none_to_list(value)

    def format_for_prompt(self) -> str:
        lines: list[str] = []
        if self.meter_and_groove.strip():
            lines.append(f"Meter & groove: {self.meter_and_groove.strip()}")
        if self.instrumentation_and_texture.strip():
            lines.append(f"Instrumentation & texture: {self.instrumentation_and_texture.strip()}")
        if self.vocal_character.strip():
            lines.append(f"Vocal character: {self.vocal_character.strip()}")
        if self.atmosphere_and_subtext.strip():
            lines.append(f"Sonic atmosphere & subtext: {self.atmosphere_and_subtext.strip()}")
        turns = [t.strip() for t in self.key_musical_turns if t.strip()]
        if turns:
            lines.append("Key musical turns:\n" + "\n".join(f"- {t}" for t in turns))
        return "\n".join(lines)


class SectionBeatReply(_Reply):
    section: str = ""
    emotion: str = ""
    visual_beat: str = ""
    camera_energy: str = ""

    @field_validator("section", "emotion", "visual_beat", "camera_energy", mode="before")
    @classmethod
    def _coerce_null_str(cls, value: object) -> object:
        return _none_to_str(value)


class SongInterpretationReply(_Reply):
    meaning: str = ""
    tone: str = ""
    music_feel: str = ""
    visual_motifs: list[str] = Field(default_factory=list)
    section_beats: list[SectionBeatReply] = Field(default_factory=list)

    @field_validator("meaning", "tone", "music_feel", mode="before")
    @classmethod
    def _coerce_null_str(cls, value: object) -> object:
        return _none_to_str(value)

    @field_validator("visual_motifs", "section_beats", mode="before")
    @classmethod
    def _coerce_null_list(cls, value: object) -> object:
        return _none_to_list(value)


class BibleReply(_Reply):
    interpretation: SongInterpretationReply | None = None
    appearance: str = ""
    invariants: str = ""
    style: str
    palette: str
    lighting: str
    negative_prompt: str
    settings: list[SetReply]
    props: list[PropReply] = Field(default_factory=list)
    concept: str = ""
    wardrobe: list[WardrobeReply] = Field(default_factory=list)
    characters: list[CharacterReply] = Field(default_factory=list)
    rules: list[str] = Field(default_factory=list)
    exclusions: list[str] = Field(default_factory=list)

    @field_validator("exclusions", mode="before")
    @classmethod
    def _normalize_exclusions(cls, value: object) -> list[str]:
        return normalize_exclusions(value)

    @field_validator("interpretation", mode="before")
    @classmethod
    def _coerce_interpretation(cls, value: object) -> object:
        if isinstance(value, str):
            text = value.strip()
            return {"meaning": text} if text else None
        return value

    @field_validator(
        "appearance",
        "invariants",
        "style",
        "palette",
        "lighting",
        "negative_prompt",
        "concept",
        mode="before",
    )
    @classmethod
    def _coerce_bible_text(cls, value: object) -> object:
        return _coerce_text(value)


# Spellings a model is likely to use for the allowed values, after lower-casing and turning
# spaces and hyphens into underscores.
_ENUM_ALIASES = {
    "lip_sync": "lipsync",
    "b_roll": "broll",
    "closeup": "close_up",
    "close_up_shot": "close_up",
    "medium_close_up": "medium_close",
    "medium_closeup": "medium_close",
    "medium_shot": "medium",
    "wide_shot": "wide",
}


def _normalize_choice(value: object) -> object:
    """Accept "Close-up" for ``close_up`` and similar. Anything else is left for validation."""
    if not isinstance(value, str):
        return value
    key = "_".join(value.strip().lower().replace("-", " ").split())
    return _ENUM_ALIASES.get(key, key)


class ShotCreative(_Reply):
    id: str
    type: ShotType
    framing: Framing | None = None  # small models return null for a cutaway with nothing to frame
    setting: str
    still_prompt: str
    motion_prompt: str
    wardrobe: str | None = None
    character: str | None = None
    characters: list[str] = Field(default_factory=list)
    allow_cuts: bool = False
    props: list[str] = Field(default_factory=list)

    @field_validator("characters", "props", mode="before")
    @classmethod
    def _coerce_null_list(cls, value: object) -> object:
        return _none_to_list(value)

    @field_validator("id")
    @classmethod
    def _check_id(cls, value: str) -> str:
        return validate_safe_id(value, "shot id")

    @field_validator("type", mode="before")
    @classmethod
    def _tolerate_type_spelling(cls, value: object) -> object:
        return _normalize_choice(value)

    @field_validator("framing", mode="before")
    @classmethod
    def _tolerate_framing_spelling(cls, value: object) -> object:
        normalized = _normalize_choice(value)
        return None if normalized in ("", "none", "null", "n/a") else normalized


class ShotBatchReply(_Reply):
    shots: list[ShotCreative]


def build_manifest(
    config: ProjectConfig,
    analysis: Analysis,
    lyrics_text: str,
    style_text: str,
    llm: LLMClient | None,
    *,
    mix_path: Path | None = None,
    audio_llm: LLMClient | None = None,
    exclusions: Sequence[str] | None = None,
) -> Manifest:
    config, tuned_adjustments = tuning_mod.tune_config_for_planning(
        config, analysis, lyrics_text=lyrics_text, style_text=style_text
    )
    for adj in tuned_adjustments:
        log.info(
            "Auto-tuned %s: %r -> %r (%s)",
            adj.key,
            adj.current,
            adj.recommended,
            adj.reason,
        )
    manifest = Manifest(
        title=config.title,
        duration=analysis.duration,
        fps=config.video.fps,
    )
    # Derive multi-singer section snapping deterministically from lyric section tags and
    # acoustic voice turns BEFORE calling the LLM, so cut points never vary with model output.
    if config.video.snap_sections == "auto":
        solo_voices = {t.voice for t in analysis.vocal_turns if t.voice != BOTH_VOICES}
        snap_sections = has_multiple_section_singers(analysis, lyrics_text) or len(solo_voices) >= 2
    else:
        snap_sections = config.video.snap_sections == "always"
    manifest.shots = _timed_shots(config, analysis, snap_sections=snap_sections)
    video_style = config.planning.video_style
    merged_exclusions = normalize_exclusions(
        [
            *config.inputs.exclusions,
            *prompts.parse_exclusions_from_style(style_text),
            *(exclusions or ()),
        ]
    )
    if llm is None:
        log.warning("LLM disabled: using rule-based direction. Edit manifest.json to refine it.")
        manifest.visual_bible = prompts.sync_bible_exclusions(
            VisualBible(style=" ".join(style_text.split()), exclusions=merged_exclusions)
        )
    else:
        listening_notes: AudioListeningNotes | None = None
        wants_listen = config.planning.audio_listen != "never" and (
            config.planning.audio_listen == "always" or not getattr(llm, "supports_audio", False)
        )
        effective_audio_llm = (
            audio_llm
            if (audio_llm is not None and getattr(audio_llm, "supports_audio", False))
            else (
                llm
                if (
                    config.planning.audio_listen == "always"
                    and getattr(llm, "supports_audio", False)
                )
                else None
            )
        )
        if (
            wants_listen
            and mix_path is not None
            and mix_path.is_file()
            and effective_audio_llm is not None
        ):
            listening_notes = _listen_to_audio(
                effective_audio_llm,
                mix_path,
                lyrics_text=lyrics_text,
                analysis=analysis,
            )
        _write_bible(
            manifest,
            lyrics_text,
            style_text,
            llm,
            analysis=analysis,
            mix_path=mix_path,
            audio_notes=listening_notes,
            video_style=video_style,
            exclusions=merged_exclusions,
        )
    solo_vocals = config.planning.vocals == "solo"
    voice_map = (
        _map_voices_to_characters(analysis, manifest)
        if video_style is VideoStyle.LIPSYNC and not solo_vocals
        else {}
    )
    if video_style is VideoStyle.LIPSYNC and config.planning.max_lipsync_share is not None:
        # After the bible, so each shot can be matched to its section or acoustic singer.
        _thin_lipsync(
            manifest.shots,
            config.planning.max_lipsync_share,
            lambda shot: _shot_singer_hint(
                shot, analysis, manifest.characters, voice_map, solo_vocals=solo_vocals
            ),
        )
    _direct_shots(
        manifest,
        config,
        llm,
        analysis=analysis,
        voice_map=voice_map,
        style_text=style_text,
    )
    _populate_slice_mutes(manifest, analysis, voice_map)
    _report_mix(manifest.shots, has_bible=llm is not None)
    return manifest


def _build_interpretation(raw: SongInterpretationReply | None) -> SongInterpretation | None:
    if raw is None:
        return None
    motifs = [m.strip() for m in raw.visual_motifs if m.strip()]
    beats = [
        SectionBeat(
            section=b.section.strip(),
            emotion=b.emotion.strip(),
            visual_beat=b.visual_beat.strip(),
            camera_energy=b.camera_energy.strip(),
        )
        for b in raw.section_beats
        if b.section.strip()
    ]
    meaning = raw.meaning.strip()
    tone = raw.tone.strip()
    music_feel = raw.music_feel.strip()
    if not (meaning or tone or music_feel or motifs or beats):
        return None
    return SongInterpretation(
        meaning=meaning,
        tone=tone,
        music_feel=music_feel,
        visual_motifs=motifs,
        section_beats=beats,
    )


def _listen_to_audio(
    audio_llm: LLMClient,
    mix_path: Path,
    *,
    lyrics_text: str = "",
    analysis: Analysis | None = None,
) -> AudioListeningNotes | None:
    """Run one native MP3 listening pass to supply meter, timbre, and mood to a text planner."""
    parts = [
        "Listen carefully to the attached audio track and describe its exact meter/groove, "
        "instrumentation, vocal delivery, sonic atmosphere, and structural turning points."
    ]
    if analysis is not None and analysis.tempo_bpm > 0:
        parts.append(
            f"Measured tempo/duration: ~{analysis.tempo_bpm:.0f} BPM, {analysis.duration:.0f}s."
        )
    section_list = "\n".join(f"- {name}" for name in _section_order(lyrics_text))
    if section_list:
        parts.append(f"Song sections:\n{section_list}")
    try:
        notes = complete_json_with_optional_audio(
            audio_llm,
            system=prompts.AUDIO_LISTEN_SYSTEM,
            text="\n\n".join(parts),
            schema=AudioListeningNotes,
            audio_path=mix_path,
        )
        if notes.format_for_prompt():
            log.info("Attached native audio listening notes to planning brief.")
            return notes
    except LLMError as exc:
        log.warning("Native audio listening pass failed (%s); continuing with text cues.", exc)
    return None


def _write_bible(
    manifest: Manifest,
    lyrics_text: str,
    style_text: str,
    llm: LLMClient,
    *,
    analysis: Analysis | None = None,
    mix_path: Path | None = None,
    audio_notes: AudioListeningNotes | None = None,
    video_style: VideoStyle = VideoStyle.LIPSYNC,
    exclusions: Sequence[str] = (),
) -> None:
    section_list = "\n".join(f"- {name}" for name in _section_order(lyrics_text)) or "(none)"
    parts = [f"MUSIC STYLE AND BRIEF:\n{style_text}"]
    if analysis is not None:
        if analysis.musical_features and analysis.musical_features.summary:
            parts.append(f"MUSICAL ANALYSIS:\n{analysis.musical_features.summary}")
        elif analysis.tempo_bpm > 0:
            parts.append(
                f"MUSICAL ANALYSIS:\n~{analysis.tempo_bpm:.0f} BPM, "
                f"duration {analysis.duration:.0f}s"
            )
        if analysis.stems and analysis.stems.summary:
            parts.append(f"ARRANGEMENT & INSTRUMENTATION:\n{analysis.stems.summary}")
        cue_summary = format_cues_summary(analysis.cues)
        if cue_summary:
            parts.append(f"STRUCTURAL CUES:\n{cue_summary}")
        reg_summary = describe_vocal_register(analysis.vocal_turns)
        if reg_summary:
            parts.append(f"VOCAL REGISTER:\n{reg_summary}")
    if audio_notes is not None and (notes_text := audio_notes.format_for_prompt()):
        parts.append(f"NATIVE AUDIO LISTENING NOTES (HEARD DIRECTLY FROM THE TRACK):\n{notes_text}")
    if exclusions:
        parts.append(
            "GLOBAL EXCLUSIONS (NEVER SHOW IN ANY SHOT):\n"
            + "\n".join(f"- {item}" for item in exclusions)
        )
    parts.append(f"SONG SECTIONS, IN ORDER:\n{section_list}")
    parts.append(f"LYRICS:\n{lyrics_text}")
    if video_style is VideoStyle.VISUAL:
        bible_system = prompts.VISUAL_BIBLE_SYSTEM
    elif video_style is VideoStyle.NARRATIVE:
        bible_system = f"{prompts.BIBLE_SYSTEM}\n\n{prompts.NARRATIVE_BIBLE_ADDENDUM}"
    else:
        bible_system = prompts.BIBLE_SYSTEM
    try:
        bible = complete_json_with_optional_audio(
            llm,
            system=bible_system,
            text="\n\n".join(parts),
            schema=BibleReply,
            audio_path=mix_path,
        )
    except LLMError as exc:
        raise LLMError(
            f"Could not get the visual bible from the model: {exc} Run `mvd llm-check` to test the "
            "model, or try a more capable one."
        ) from exc
    if video_style is VideoStyle.VISUAL:
        manifest.characters = {}
        manifest.character = Character()
    else:

        def _dedup_wardrobe(items: list[WardrobeReply]) -> list[WardrobeState]:
            seen_w: set[str] = set()
            return [
                WardrobeState(id=_unique_id(w.id, seen_w), description=w.description) for w in items
            ]

        characters: dict[str, Character] = {}
        seen_char_ids: set[str] = set()
        for item in bible.characters:
            if not item.appearance.strip():
                continue
            char_id = _unique_id(item.id, seen_char_ids)
            characters[char_id] = Character(
                name=item.name or char_id,
                appearance=item.appearance,
                invariants=item.invariants,
                wardrobe=_dedup_wardrobe(item.wardrobe),
            )
        if len(characters) > 1:
            primary = next(iter(characters.values()))
            if not primary.wardrobe and bible.wardrobe:
                primary.wardrobe = _dedup_wardrobe(bible.wardrobe)
            if not primary.invariants and bible.invariants:
                primary.invariants = bible.invariants
            manifest.characters = characters
            manifest.character = primary
        elif len(characters) == 1:
            single = next(iter(characters.values()))
            if not single.wardrobe and bible.wardrobe:
                single.wardrobe = _dedup_wardrobe(bible.wardrobe)
            if not single.invariants and bible.invariants:
                single.invariants = bible.invariants
            manifest.character = single
        else:
            manifest.character = Character(
                appearance=bible.appearance,
                invariants=bible.invariants,
                wardrobe=_dedup_wardrobe(bible.wardrobe),
            )
    seen_prop_ids: set[str] = set()
    props = [
        PropDesign(
            id=_unique_id(p.id, seen_prop_ids),
            name=p.name,
            description=p.description,
            sections=p.sections,
        )
        for p in bible.props
    ]
    manifest.visual_bible = prompts.sync_bible_exclusions(
        VisualBible(
            interpretation=_build_interpretation(bible.interpretation),
            concept=bible.concept,
            style=bible.style,
            palette=bible.palette,
            lighting=bible.lighting,
            negative_prompt=bible.negative_prompt,
            settings=[SetDesign(**s.model_dump()) for s in bible.settings if s.name.strip()],
            props=props,
            rules=bible.rules,
            exclusions=normalize_exclusions([*bible.exclusions, *exclusions]),
        )
    )


def has_multiple_section_singers(analysis: Analysis, lyrics_text: str = "") -> bool:
    """Detect deterministically from lyric section tags whether more than one singer performs.

    Only performer cues count: ``[Verse 1: Elias]``, ``[Chorus (Odalie)]``, ``[Female Verse]``,
    ``[Singer 2]`` or a bare capitalised name such as ``[Elias]``. Structural descriptions
    (``[Chorus - big hook]``, ``[Guitar Solo]``, ``[Chorus x2]``) and lower-case moods
    (``[Verse 1 - angry]``) never count, so a solo song is not mistaken for a duet. Ensemble
    cues (``[Chorus: Both]``) always count. ``video.snap_sections`` overrides the result.
    """
    labels: list[str] = [line.section for line in analysis.lyric_lines if line.section]
    labels.extend(_section_order(lyrics_text))
    return labels_have_multiple_singers(labels)


def _section_order(lyrics_text: str) -> list[str]:
    """Section labels in song order, each repeat of the same label kept as it occurs."""
    order: list[str] = []
    for line in parse_lyrics(lyrics_text):
        if line.section and (not order or order[-1] != line.section):
            order.append(line.section)
    return order


def _timed_shots(
    config: ProjectConfig, analysis: Analysis, *, snap_sections: bool = False
) -> list[Shot]:
    video, video_style, shots = config.video, config.planning.video_style, []
    for index, seg in enumerate(
        segment(
            analysis,
            video,
            snap_sections=snap_sections,
            pacing_plugin=config.plugins.pacing,
        )
    ):
        shot_id = f"s{index + 1:03d}"
        slice_start = max(0.0, seg.start - video.handle_seconds)
        slice_end = min(analysis.duration, seg.end + video.handle_seconds)
        if video_style is VideoStyle.VISUAL:
            shot_type = ShotType.BROLL
        elif video_style is VideoStyle.NARRATIVE:
            shot_type = ShotType.PERFORMANCE
        else:
            shot_type = seg.type
        num_frames = frames_for(
            slice_end - slice_start, video.fps, video.frame_multiple, video.frame_offset
        )
        shots.append(
            Shot(
                id=shot_id,
                index=index,
                start=seg.start,
                end=seg.end,
                type=shot_type,
                section=seg.section,
                lyrics=seg.lyrics,
                intensity=seg.intensity,
                audio_slice=AudioSlice(
                    path=f"assets/audio_slices/{shot_id}.wav",
                    start=round(slice_start, 3),
                    end=round(slice_end, 3),
                    head_handle=round(seg.start - slice_start, 3),
                ),
                num_frames=num_frames,
                still=AssetState(attempts_allowed=config.retry.max_attempts),
                video=AssetState(attempts_allowed=config.retry.max_attempts),
            )
        )
    return shots


_is_hook_section = is_hook_section


def _cutaway_run(shots: list[Shot], position: int) -> int:
    """Length of the cutaway run that making ``shots[position]`` a cutaway would create."""
    run = 1
    for step in (-1, 1):
        at = position + step
        while 0 <= at < len(shots) and shots[at].type is ShotType.BROLL:
            run += 1
            at += step
    return run


def _thin_lipsync(
    shots: list[Shot],
    max_share: float,
    singer_of: Callable[[Shot], str | None] | None = None,
) -> None:
    """Turn sung shots into cutaways until singing to camera fits ``max_share`` of the song.

    A cutaway keeps the vocal and drops the singer, so there is no face to keep consistent and
    nothing to lip-sync. The quietest non-hook shots go first. Chorus and hook shots and the last
    sung shot stay on camera, and the singer is never off screen for more than
    ``_MAX_CUTAWAY_RUN`` shots.
    """
    total = sum(s.duration for s in shots)
    indexed_sung = [(i, s) for i, s in enumerate(shots) if s.type is ShotType.LIPSYNC]
    excess = sum(s.duration for _, s in indexed_sung) - max_share * total
    if not indexed_sung or excess <= _SHARE_EPS:
        return

    last_sung = indexed_sung[-1][1]
    # In a duet every singer keeps at least half of their sung shots on camera, so one singer's
    # verse is never turned into cutaways wholesale. A solo song has no such floor, and untagged
    # or shared sections (singer is None) are never treated as a singer.
    singers = {i: singer_of(s) if singer_of else None for i, s in indexed_sung}
    on_camera = Counter(s for s in singers.values() if s is not None)
    floor = (
        {singer: -(-count // 2) for singer, count in on_camera.items()}
        if len(on_camera) > 1
        else {}
    )
    candidates = [
        (i, s) for i, s in indexed_sung if s is not last_sung and not _is_hook_section(s.section)
    ]
    for position, shot in sorted(candidates, key=lambda item: item[1].intensity):
        if excess <= _SHARE_EPS:
            break
        if _cutaway_run(shots, position) > _MAX_CUTAWAY_RUN:
            continue
        singer = singers[position]
        if singer is not None and singer in floor:
            if on_camera[singer] - 1 < floor[singer]:
                continue
            on_camera[singer] -= 1
        shot.type = ShotType.BROLL
        excess -= shot.duration

    if excess > _SHARE_EPS:
        floor_note = ", the half of each singer's shots that stays on camera," if floor else ""
        log.info(
            "Singing share stayed at %.0f%% (cap %.0f%%): remaining sung shots are protected "
            "chorus/hook/final shots%s or hit the %d-cutaway run limit.",
            singing_share(shots) * 100,
            max_share * 100,
            floor_note,
            _MAX_CUTAWAY_RUN,
        )


def pace_label(intensity: float) -> str:
    """How fast a shot should feel, from the song's intensity at that point."""
    return "high" if intensity >= 0.66 else "calm" if intensity < 0.4 else "moderate"


def singing_share(shots: list[Shot]) -> float:
    """Share of the song, by time, that is the singer singing to camera."""
    total = sum(s.duration for s in shots)
    sung = sum(s.duration for s in shots if s.type is ShotType.LIPSYNC)
    return sung / total if total else 0.0


def _report_mix(shots: list[Shot], *, has_bible: bool = True) -> None:
    """Log how the plan splits screen time, and warn when it is all one look."""
    share = singing_share(shots)
    looks = {s.setting.strip().lower() for s in shots if s.setting.strip()}
    log.info(
        "Plan: %d shots, %.0f%% of the song is the singer singing to camera, %d distinct set(s).",
        len(shots),
        share * 100,
        len(looks),
    )
    _report_pace(shots)
    if has_bible and len(shots) >= 3 and len(looks) < 3:
        log.warning(
            "Only %d distinct set(s) in the plan. Three or more looks keep a video from "
            "feeling like one setup; add sets to the visual bible.",
            len(looks),
        )


def _report_pace(shots: list[Shot]) -> None:
    """Warn when shot scale does not follow the song's intensity.

    Pace is built from cut speed, movement in frame, shot scale and camera movement together. Scale
    is the one a plan can be checked on: it should tighten as the song builds and loosen as it
    falls away. A plan that does the opposite works against the music.
    """
    if len(shots) < 8:
        return
    if len({s.framing for s in shots}) < 2:
        log.warning(
            "Every shot has the same framing (%s): the pace will feel flat.",
            shots[0].framing.value,
        )
        return
    if len({s.intensity for s in shots}) < 2:
        return
    fit = statistics.correlation(
        [s.intensity for s in shots], [float(_SCALE[s.framing]) for s in shots]
    )
    if fit < -0.25:
        log.warning(
            "Shot scale runs against the song's intensity (r=%.2f): close-ups land on the quiet "
            "parts and wide shots on the loud ones. Check the framing against the section arc.",
            fit,
        )


# ------------------------------------------------------------------- direction


class _Wardrobe:
    """Tracks the costume state through the song and enforces one-way changes."""

    def __init__(self, ids: list[str], monotonic: bool) -> None:
        self._ids, self._monotonic, self._current = ids, monotonic, 0

    @property
    def current(self) -> str | None:
        return self._ids[self._current] if self._ids else None

    def assign(self, requested: str | None, shot_id: str) -> str | None:
        if not self._ids:
            return None
        index = self._ids.index(requested) if requested in self._ids else self._current
        if requested is not None and requested not in self._ids:
            log.warning(
                "%s: unknown wardrobe state %r; keeping %s", shot_id, requested, self.current
            )
        if self._monotonic and index < self._current:
            log.warning(
                "%s: wardrobe %r would undo an earlier change; keeping %s",
                shot_id,
                requested,
                self.current,
            )
            index = self._current
        self._current = index
        return self._ids[index]


_GENERIC_NAME_PREFIXES = frozenset(
    {
        "the",
        "a",
        "an",
        "singer",
        "vocalist",
        "lead",
        "voice",
        "performer",
        "protagonist",
        "character",
        "narrator",
        "speaker",
        "soloist",
        "young",
        "old",
        "mr",
        "mrs",
        "ms",
        "dr",
        "sir",
        "lady",
        "lord",
        "king",
        "queen",
        "captain",
    }
)


def _character_tokens(cid: str, char: Character) -> list[str]:
    tokens = [cid.strip()]
    if char.name and char.name.strip():
        clean_name = unicodedata.normalize("NFC", char.name.strip())
        candidates = [clean_name]
        # Also recognize the base name before parenthetical/comma epithets (e.g. "Eun-ji (The
        # Kumiho)", "Margot, the Choir Leader").
        base_name = re.split(r"\s*[(\[,]", clean_name, maxsplit=1)[0].strip()
        if base_name and base_name != clean_name:
            candidates.append(base_name)
        # Only an epithet ("Corbin the Host") marks a leading given name as the character's own;
        # a plain first name ("Rose Martin", "Kim Ji-woo") is often a common word or shared.
        epithet = re.match(r"^(.+?)\s+the\s+\S", base_name, flags=re.IGNORECASE)
        given = epithet.group(1).strip(".,:;'\"-") if epithet else ""
        if len(given) >= 3 and given.casefold() not in _GENERIC_NAME_PREFIXES:
            candidates.append(given)
        for cand in candidates:
            tokens.append(cand)
            folded = ascii_fold(cand).strip()
            if folded and folded != cand:
                tokens.append(folded)
    return list(dict.fromkeys(t for t in tokens if len(t) >= 2 or not char.name))


def _section_singer(shot: Shot, characters: dict[str, Character]) -> str | None:
    """The one character a shot's section names, or None when it names none or several."""
    named = _characters_in_text(shot.section, characters)
    return named[0] if len(named) == 1 else None


def _character_token_pattern(token: str) -> tuple[re.Pattern[str], str]:
    """Return ``(regex, kind)`` with ``kind`` in ``('cased', 'unicameral', 'numeric')``."""
    has_alpha = any(ch.isalpha() for ch in token)
    if not has_alpha:
        return re.compile(rf"\b{re.escape(token)}\b"), "numeric"
    if token.upper() != token.lower():
        return re.compile(rf"\b{re.escape(token)}\b", flags=re.IGNORECASE), "cased"
    if any(is_syllabic_char(ch) for ch in token):
        return re.compile(rf"(?<![A-Za-z0-9_]){re.escape(token)}(?![A-Za-z0-9_])"), "unicameral"
    return re.compile(rf"\b{re.escape(token)}\b"), "unicameral"


def _is_proper_token_match(match_text: str, kind: str) -> bool:
    if kind == "numeric":
        return False
    if kind == "cased":
        return match_text[0].isupper()
    return True


def _characters_in_text(
    text: str | None, characters: dict[str, Character], *, proper_noun_only: bool = False
) -> list[str]:
    """Character ids mentioned by id or name in ``text``, in order of appearance.

    When ``proper_noun_only`` is true, cased scripts require a capital initial so common words
    matching a character name (such as "grace" or "dawn") are left alone in prose prompts, numeric
    ids are ignored in prose, and unicameral script names (such as CJK or Hangul) match directly.
    """
    if not text or not characters:
        return []
    folded_text: str | None = None
    found: list[tuple[int, str]] = []
    for cid, char in characters.items():
        positions: list[int] = []
        for token in _character_tokens(cid, char):
            pattern, kind = _character_token_pattern(token)
            matched_token = False
            for match in pattern.finditer(text):
                if proper_noun_only and not _is_proper_token_match(match.group(0), kind):
                    continue
                positions.append(match.start())
                matched_token = True
            if not matched_token and kind == "cased" and token.isascii():
                if folded_text is None:
                    folded_text = ascii_fold(text)
                if folded_text != text:
                    for match in pattern.finditer(folded_text):
                        if proper_noun_only and not _is_proper_token_match(match.group(0), kind):
                            continue
                        positions.append(match.start())
        if positions:
            found.append((min(positions), cid))
    found.sort(key=lambda item: item[0])
    return [cid for _, cid in found]


def _fix_mismatched_singer_prompt(
    shot: Shot, primary_id: str, characters: dict[str, Character]
) -> None:
    """Replace an accidentally swapped proper-noun name when a section names a single singer."""
    mentioned = _characters_in_text(shot.still_prompt, characters, proper_noun_only=True)
    if not mentioned or primary_id in mentioned:
        return
    target = characters[primary_id].name or primary_id

    for wrong_id in mentioned:
        wrong_char = characters[wrong_id]
        for token in _character_tokens(wrong_id, wrong_char):
            pattern, kind = _character_token_pattern(token)

            def _replace_if_proper(match: re.Match[str], token_kind: str = kind) -> str:
                return (
                    target if _is_proper_token_match(match.group(0), token_kind) else match.group(0)
                )

            shot.still_prompt = pattern.sub(_replace_if_proper, shot.still_prompt)
            shot.motion_prompt = pattern.sub(_replace_if_proper, shot.motion_prompt)


_LOW_REGISTER_RE = re.compile(
    r"\b(?:male|man|boy|guy|gentleman|baritone|bass|tenor|he|him|his)\b", flags=re.IGNORECASE
)
_HIGH_REGISTER_RE = re.compile(
    r"\b(?:female|woman|girl|lady|soprano|mezzo|alto|she|her|hers)\b", flags=re.IGNORECASE
)
_EMPTY_SCENE_EXPLICIT_RE = re.compile(
    r"\b(?:no\s+(?:people|person|performers?|singers?|figures?|humans?)|"
    r"empty\s+of\s+people|completely\s+empty|uninhabited|devoid\s+of\s+people)\b",
    flags=re.IGNORECASE,
)
_PERFORMER_CUE_RE = re.compile(
    r"\b(?:singer|vocalist|performer|artist|musician|band|duet|couple|pair|"
    r"man|woman|boy|girl|person|people|figure|figures|crowd|dancer|"
    r"he|him|his|she|her|hers|they|them|their|"
    r"face|eyes|hands|arms|shoulders|"
    r"standing|sitting|walking|running|leaning|turning|looking|gazing|watching|"
    r"holding|wearing|dressed|kneeling|stepping|reaching|dancing|moving)\b",
    flags=re.IGNORECASE,
)


_NEGATED_PEOPLE_RE = re.compile(
    r"\b(?:no|zero|without|empty\s+of|free\s+of|devoid\s+of|uninhabited\s+by)\s+"
    r"(?:(?:human|living|visible|other)\s+)?"
    r"(?:people|persons?|faces?|figures?|performers?|singers?|dancers?|crowds?|humans?|"
    r"silhouettes?|presence)\b",
    re.IGNORECASE,
)
_INANIMATE_MATERIAL = (
    r"(?:carved|painted|porcelain|ceramic|wooden|brass|bronze|stone|marble|glass|"
    r"sugar-glass|sugar|clockwork|miniature|tin|wax|plaster|clay|sculpted|etched|"
    r"engraved|inanimate|toy|music-box|stained-glass|woodcut|gilded)"
)
_LIVE_ACTION_VERB_LOOKAHEAD = (
    r"(?!\s+(?:stands?|sits?|walks?|runs?|looks?|gazes?|stares?|dances?|sings?|"
    r"speaks?|smiles?|weeps?|cries?|turns?)\b)"
)
_INANIMATE_FIGURE_RE = re.compile(
    rf"\b(?:"
    rf"(?:(?:two|three|small|tiny)\s+)?{_INANIMATE_MATERIAL}\s+(?:{_INANIMATE_MATERIAL}\s+)?"
    rf"(?:figures?|figurines?|statues?|statuettes?|angels?|silhouettes?|dancers?|couple|"
    rf"man|woman|boy|girl)"
    rf"|"
    rf"(?:(?:two|three|small|tiny)\s+)?(?:angel|man|woman|boy|girl|couple|dancer|human|sugar)\s+"
    rf"(?:figurines?|statues?|statuettes?|carvings?|busts?|sculptures?|mannequins?|dolls?|"
    rf"ornaments?|reliefs?|portraits?|engravings?|etchings?)"
    rf")\b{_LIVE_ACTION_VERB_LOOKAHEAD}",
    re.IGNORECASE,
)
_ANONYMOUS_POSSESSIVE_DETAIL_RE = re.compile(
    r"\b(?:(?:a|the)\s+)?(?:man|woman|girl|boy|person|human)['\u2019]s\s+"
    r"(?:[a-z]+(?:-[a-z]+)?\s+){0,2}"
    r"(?:hands?|fingers?|fingertips?|wrists?|palms?|knuckles?|gloves?|coat|hat|fedora|shawl|"
    r"dress|jacket|scarf|boots?|shoes?|watch|wristwatch|ring|cuff|sleeve|pocket|cloak|"
    r"umbrella|footprints?|bootprints?|shadow)\b",
    re.IGNORECASE,
)
_BROLL_PEOPLE_RE = re.compile(
    r"\b(?:performers?|singers?|bartenders?|security\s+(?:guards?|staff|officers?|personnel)|"
    r"two\s+figures?|"
    r"(?:a\s+|the\s+)?(?:man|woman|girl|boy|person|couple|he|she)\s+"
    r"(?:stands?|sits?|walks?|looks?|gazes?|stares?|dances?|sings?))\b",
    re.IGNORECASE,
)
_BROLL_TIGHT_FIGURE_RE = re.compile(
    r"\b(?:figures?|silhouettes?|humans?|persons?|angels?|man|woman|girl|boy|couple)\b",
    re.IGNORECASE,
)
_PLURAL_CAST_CUE_RE = re.compile(
    r"\b(?:they|their|them|both|together|unison|harmony|duet|trio|group|band|everyone|all\s+of\s+them|two-shot)\b",
    re.IGNORECASE,
)


def _broll_mentions_people(
    prompt_text: str,
    characters: dict[str, Character] | None = None,
    *,
    framing: Framing | None = None,
) -> bool:
    """True when a b-roll still prompt puts people, dancers, or named characters in frame."""
    cleaned = _NEGATED_PEOPLE_RE.sub("", prompt_text.strip())
    if not cleaned:
        return False
    if characters and _characters_in_text(cleaned, characters, proper_noun_only=True):
        return True
    subject_text = _ANONYMOUS_POSSESSIVE_DETAIL_RE.sub("", _INANIMATE_FIGURE_RE.sub("", cleaned))
    if _BROLL_PEOPLE_RE.search(subject_text):
        return True
    return bool(
        framing in (Framing.CLOSE_UP, Framing.MEDIUM_CLOSE)
        and _BROLL_TIGHT_FIGURE_RE.search(subject_text)
    )


def _is_empty_scene_prompt(
    prompt_text: str, characters: dict[str, Character] | None = None
) -> bool:
    """True when ``prompt_text`` describes an empty setting/object with no performer in frame."""
    cleaned = prompt_text.strip()
    if not cleaned:
        return False
    if _EMPTY_SCENE_EXPLICIT_RE.search(cleaned):
        return True
    if len(cleaned.split()) < 6:
        return False
    if characters and _characters_in_text(cleaned, characters):
        return False
    return not bool(_PERFORMER_CUE_RE.search(cleaned))


def _register_rank(char: Character) -> int:
    text = f"{char.name} {char.appearance}"
    low = len(_LOW_REGISTER_RE.findall(text))
    high = len(_HIGH_REGISTER_RE.findall(text))
    return -1 if low > high else (1 if high > low else 0)


def _map_voices_to_characters(analysis: Analysis, manifest: Manifest) -> dict[str, str]:
    """Map diarized solo voice ids ('voice_1', 'voice_2') to ``manifest.characters`` ids."""
    if not analysis.vocal_turns or len(manifest.characters) < 2:
        return {}
    voice_f0: dict[str, list[float]] = {}
    for turn in analysis.vocal_turns:
        if turn.voice == BOTH_VOICES:
            continue
        voice_f0.setdefault(turn.voice, []).append(turn.median_f0_hz or 0.0)
    if not voice_f0:
        return {}
    ordered_voices = sorted(voice_f0, key=lambda v: statistics.median(voice_f0[v]))

    overlaps: dict[tuple[str, str], float] = {}
    for line in analysis.lyric_lines:
        if line.start is None or line.end is None or not line.section:
            continue
        named = _characters_in_text(line.section, manifest.characters)
        if len(named) != 1:
            continue
        cid = named[0]
        for turn in analysis.vocal_turns:
            if turn.voice == BOTH_VOICES:
                continue
            shared = turn.overlap(line.start, line.end)
            if shared > 0:
                overlaps[(turn.voice, cid)] = overlaps.get((turn.voice, cid), 0.0) + shared

    mapping: dict[str, str] = {}
    used_chars: set[str] = set()
    for (voice, cid), seconds in sorted(overlaps.items(), key=lambda item: item[1], reverse=True):
        if seconds >= 0.1 and voice not in mapping and cid not in used_chars:
            mapping[voice] = cid
            used_chars.add(cid)

    if len(voice_f0) < 2:
        return mapping

    remaining_voices = [v for v in ordered_voices if v not in mapping]
    remaining_chars = sorted(
        (cid for cid in manifest.characters if cid not in used_chars),
        key=lambda cid: _register_rank(manifest.characters[cid]),
    )
    for voice, cid in zip(remaining_voices, remaining_chars):
        mapping[voice] = cid
    return mapping


_TOGETHER_PROMPT_RE = re.compile(
    r"\b(?:sing(?:ing)?\s+together|both\s+sing(?:ing)?|in\s+unison|in\s+harmony)\b",
    flags=re.IGNORECASE,
)


def _duet_character_ids(
    characters: dict[str, Character], voice_map: dict[str, str] | None = None
) -> list[str]:
    """Primary pair/group of singing character ids for ensemble ('both') sections."""
    if voice_map:
        mapped = [cid for cid in characters if cid in set(voice_map.values())]
        if len(mapped) >= 2:
            return mapped
    return list(characters.keys())


def _resolve_designator_cue(
    cue_key: str,
    characters: dict[str, Character],
    voice_map: dict[str, str] | None = None,
) -> str | None:
    """Resolve generic performer cues ('singer 1', 'male', 'female') to a character id."""
    parts = cue_key.split()
    if len(parts) == 2 and parts[1].isdigit():
        idx = int(parts[1])
        voice_key = f"voice_{idx}"
        if voice_map and voice_key in voice_map:
            return voice_map[voice_key]
        ordered = sorted(characters, key=lambda cid: _register_rank(characters[cid]))
        if 1 <= idx <= len(ordered):
            return ordered[idx - 1]
    want_rank = (
        -1 if _LOW_REGISTER_RE.search(cue_key) else (1 if _HIGH_REGISTER_RE.search(cue_key) else 0)
    )
    if want_rank != 0:
        matches = [cid for cid, char in characters.items() if _register_rank(char) == want_rank]
        if len(matches) == 1:
            return matches[0]
    return None


def _section_singers_for_shot(
    shot: Shot,
    characters: dict[str, Character],
    voice_map: dict[str, str] | None = None,
) -> list[str]:
    """Character ids indicated by ``shot.section`` (1 for solo, 2+ for ensemble/both, 0 if none)."""
    if not shot.section or not characters:
        return []
    cues = section_singers(shot.section)
    if cues is None:
        return _duet_character_ids(characters, voice_map)
    named = _characters_in_text(shot.section, characters)
    if named:
        return named
    resolved: list[str] = []
    for cue in sorted(cues):
        cid = _resolve_designator_cue(cue, characters, voice_map)
        if cid and cid not in resolved:
            resolved.append(cid)
    return resolved


def _shot_voice_dominance(shot: Shot, analysis: Analysis) -> tuple[str | None, float, float]:
    """Return ``(dominant_voice, share_of_voiced_time, dominant_seconds)`` inside ``shot``."""
    totals = voice_activity_in_span(analysis.vocal_turns, shot.start, shot.end)
    if not totals:
        return None, 0.0, 0.0
    best_voice, best_sec = max(totals.items(), key=lambda pair: pair[1])
    total_sec = sum(totals.values())
    return best_voice, (best_sec / total_sec if total_sec > 0 else 0.0), best_sec


def _resolve_shot_singers(
    shot: Shot,
    analysis: Analysis | None,
    characters: dict[str, Character],
    voice_map: dict[str, str] | None = None,
    *,
    solo_vocals: bool = False,
) -> list[str]:
    """Resolve active singing character ids for ``shot`` (1 = solo, 2+ = singing together)."""
    if not characters:
        return []
    if solo_vocals:
        return [next(iter(characters))] if shot.type is ShotType.LIPSYNC else []
    sec_singers = _section_singers_for_shot(shot, characters, voice_map)
    if analysis is None or not analysis.vocal_turns:
        return sec_singers
    best_voice, share, seconds = _shot_voice_dominance(shot, analysis)
    if best_voice is None or seconds < 0.15:
        return sec_singers
    if best_voice == BOTH_VOICES and len(characters) >= 2:
        together = (
            sec_singers if len(sec_singers) >= 2 else _duet_character_ids(characters, voice_map)
        )
        if not sec_singers and share >= 0.55 and seconds >= 0.20:
            return together
        if len(sec_singers) >= 2:
            return together
        if len(sec_singers) == 1 and share >= 0.75 and seconds >= 0.30:
            return together
    if voice_map and best_voice in voice_map:
        acoustic_char = voice_map[best_voice]
        if not sec_singers and share >= 0.70:
            return [acoustic_char]
        if len(sec_singers) >= 2 and share >= 0.85 and seconds >= 0.25:
            return [acoustic_char]
        if len(sec_singers) == 1 and share >= 0.85:
            return [acoustic_char]
    return sec_singers


def _shot_singer_hint(
    shot: Shot,
    analysis: Analysis | None,
    characters: dict[str, Character],
    voice_map: dict[str, str] | None = None,
    *,
    solo_vocals: bool = False,
) -> str | None:
    """Return the solo singer id for ``shot``, or ``None`` when shared ('both') or untagged."""
    if characters:
        resolved = _resolve_shot_singers(
            shot, analysis, characters, voice_map, solo_vocals=solo_vocals
        )
        return resolved[0] if len(resolved) == 1 else None
    if analysis is None or not analysis.vocal_turns:
        return None
    best_voice, share, seconds = _shot_voice_dominance(shot, analysis)
    if best_voice is None or best_voice == BOTH_VOICES or seconds < 0.15:
        return None
    return best_voice if share >= 0.70 else None


def _merge_spans(spans: list[tuple[float, float]], gap: float = 0.02) -> list[tuple[float, float]]:
    if not spans:
        return []
    ordered = sorted((a, b) for a, b in spans if b > a)
    if not ordered:
        return []
    merged: list[list[float]] = [[ordered[0][0], ordered[0][1]]]
    for start, end in ordered[1:]:
        if start <= merged[-1][1] + gap:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(a, b) for a, b in merged]


def _neighbor_singer_ids(manifest: Manifest, shot: Shot) -> set[str]:
    active = manifest.active_singer_ids(shot)
    if active:
        return active
    if shot.character and (resolved := manifest.resolve_character_id(shot.character)):
        return {resolved}
    return set()


def _non_singing_gaps(
    slice_start: float,
    slice_end: float,
    covered_spans: list[tuple[float, float]],
    *,
    pad: float = 0.20,
    min_gap: float = 0.25,
) -> list[tuple[float, float]]:
    """Spans inside ``[slice_start, slice_end]`` outside every padded singing span."""
    safe_min_gap = (
        max(1e-6, float(min_gap))
        if (not isinstance(min_gap, bool) and math.isfinite(min_gap))
        else 0.25
    )
    if (
        not covered_spans
        or not (math.isfinite(slice_start) and math.isfinite(slice_end))
        or slice_end - slice_start < safe_min_gap
    ):
        return []
    safe_pad = max(0.0, float(pad)) if (not isinstance(pad, bool) and math.isfinite(pad)) else 0.0
    padded = _merge_spans(
        [
            (max(0.0, a - safe_pad), b + safe_pad)
            for a, b in covered_spans
            if math.isfinite(a) and math.isfinite(b) and b > a
        ]
    )
    if not padded:
        return []
    gaps: list[tuple[float, float]] = []
    cursor = slice_start
    for a, b in padded:
        if b <= cursor:
            continue
        if a >= slice_end:
            break
        if a - cursor >= safe_min_gap:
            gaps.append((cursor, a))
        cursor = max(cursor, b)
    if slice_end - cursor >= safe_min_gap:
        gaps.append((cursor, slice_end))
    return gaps


def _populate_slice_mutes(
    manifest: Manifest,
    analysis: Analysis,
    voice_map: dict[str, str],
) -> None:
    """Record foreign-singer and non-singing lead-in/tail mute spans on lip-sync slices."""
    shots = manifest.shots
    multi_char = len(manifest.characters) > 1
    vocal_spans = [(w.start, w.end) for w in analysis.vocal_windows if w.end > w.start]
    aligned_lyric_spans = [
        (ln.start, ln.end)
        for ln in analysis.lyric_lines
        if (
            ln.timing_source == "aligned"
            and ln.start is not None
            and ln.end is not None
            and ln.end > ln.start
        )
    ]
    if aligned_lyric_spans and vocal_spans:
        aligned_lyric_spans = [
            (a, b)
            for a, b in aligned_lyric_spans
            if any(min(b, ve) > max(a, vs) for vs, ve in vocal_spans)
        ]
    singing_spans: list[tuple[float, float]] = (
        aligned_lyric_spans if aligned_lyric_spans else vocal_spans
    )
    for idx, shot in enumerate(shots):
        if shot.type is not ShotType.LIPSYNC:
            shot.audio_slice.mute_spans = []
            continue
        s_start, s_end = shot.audio_slice.start, shot.audio_slice.end
        active = manifest.active_singer_ids(shot)
        is_together = len(active) >= 2
        handle_mutes: list[tuple[float, float]] = []
        body_mutes: list[tuple[float, float]] = []

        # Silence instrumental lead-in / tail / gaps inside the slice so stem bleed never drives
        # mouth movement before singing begins or after a lyric ends.
        for g_start, g_end in _non_singing_gaps(s_start, s_end, singing_spans):
            if g_start < shot.start:
                handle_mutes.append((g_start, min(g_end, shot.start)))
            if g_end > shot.end:
                handle_mutes.append((max(g_start, shot.end), g_end))
            in_s, in_e = max(g_start, shot.start), min(g_end, shot.end)
            if in_e - in_s >= 0.01:
                body_mutes.append((in_s, in_e))

        if multi_char and active:
            if idx > 0 and s_start < shot.start:
                prev_singers = _neighbor_singer_ids(manifest, shots[idx - 1])
                if prev_singers and prev_singers != active:
                    handle_mutes.append((s_start, shot.start))
            if idx + 1 < len(shots) and s_end > shot.end:
                next_singers = _neighbor_singer_ids(manifest, shots[idx + 1])
                if next_singers and next_singers != active:
                    handle_mutes.append((shot.end, s_end))

        own_voice = dominant_voice(analysis.vocal_turns, shot.start, shot.end)
        is_together_voice = is_together or own_voice == BOTH_VOICES
        for turn in analysis.vocal_turns:
            ov_s = max(s_start, turn.start)
            ov_e = min(s_end, turn.end)
            if ov_e - ov_s < 0.01:
                continue
            turn_char = voice_map.get(turn.voice)
            if is_together_voice:
                # On simultaneous duet shots, both mapped vocalists belong inside the shot body,
                # while solo turns bleeding into handles from adjacent solo bars are muted (unless
                # overlapping solo turns make the handle itself simultaneous 'both').
                foreign_in_body = bool(active and turn_char is not None and turn_char not in active)
                handle_is_both = turn.voice == BOTH_VOICES or (
                    (
                        ov_s < shot.start
                        and dominant_voice(analysis.vocal_turns, ov_s, min(ov_e, shot.start))
                        == BOTH_VOICES
                    )
                    or (
                        ov_e > shot.end
                        and dominant_voice(analysis.vocal_turns, max(ov_s, shot.end), ov_e)
                        == BOTH_VOICES
                    )
                )
                foreign_in_handle = foreign_in_body or not handle_is_both
            elif turn.voice == BOTH_VOICES:
                # A solo singer is part of 'both' inside their own shot body, so only mute handles.
                foreign_in_body = False
                foreign_in_handle = bool(active or own_voice is not None)
            else:
                foreign_solo = (
                    turn_char is not None and bool(active) and turn_char not in active
                ) or (not voice_map and own_voice is not None and turn.voice != own_voice)
                foreign_in_body = foreign_solo
                foreign_in_handle = foreign_solo

            if foreign_in_handle:
                if ov_s < shot.start:
                    handle_mutes.append((ov_s, min(ov_e, shot.start)))
                if ov_e > shot.end:
                    handle_mutes.append((max(ov_s, shot.end), ov_e))
            if foreign_in_body:
                in_s, in_e = max(ov_s, shot.start), min(ov_e, shot.end)
                if in_e - in_s >= 0.01:
                    body_mutes.append((in_s, in_e))

        merged_body = _merge_spans(body_mutes)
        muted_body_dur = sum(b - a for a, b in merged_body)
        # Never mute the entire shot body if diarization disagreed with every frame of the shot.
        keep_body = shot.duration - muted_body_dur >= 0.20
        combined = _merge_spans(handle_mutes + (merged_body if keep_body else []))
        shot.audio_slice.mute_spans = [
            TimeSpan(start=round(a, 3), end=round(b, 3)) for a, b in combined if b - a >= 0.01
        ]


def _direct_shots(
    manifest: Manifest,
    config: ProjectConfig,
    llm: LLMClient | None,
    *,
    analysis: Analysis | None = None,
    voice_map: dict[str, str] | None = None,
    style_text: str = "",
) -> None:
    monotonic = config.planning.wardrobe_monotonic
    wardrobe = _Wardrobe(manifest.character.wardrobe_ids, monotonic)
    cast_wardrobes = {
        cid: _Wardrobe(char.wardrobe_ids, monotonic) for cid, char in manifest.characters.items()
    }
    size = config.llm.shots_per_request if llm else max(1, len(manifest.shots))
    for offset in range(0, len(manifest.shots), size):
        batch = manifest.shots[offset : offset + size]
        creative: dict[str, ShotCreative] = {}
        if llm:
            try:
                creative = _ask_for_batch(
                    manifest,
                    batch,
                    offset,
                    wardrobe.current,
                    cast_wardrobes,
                    llm,
                    allow_multicut=config.planning.allow_multicut_shots,
                    analysis=analysis,
                    voice_map=voice_map,
                    style_text=style_text,
                    video_style=config.planning.video_style,
                    solo_vocals=config.planning.vocals == "solo",
                )
            except LLMUnavailableError as exc:
                log.warning(
                    "The model became unavailable at shots %s to %s (%s); using rule-based "
                    "direction for remaining batches.",
                    batch[0].id,
                    batch[-1].id,
                    exc,
                )
                llm = None
            except LLMError as exc:
                # One bad batch must not lose the whole plan: these shots get simple rule-based
                # direction and can be edited in manifest.json.
                log.warning(
                    "The model could not direct shots %s to %s (%s). They get rule-based "
                    "direction instead. Try a lower llm.shots_per_request or a more capable model.",
                    batch[0].id,
                    batch[-1].id,
                    exc,
                )
        cast_for_guards = manifest.characters or (
            {"_solo": manifest.character} if manifest.character.name else {}
        )
        for shot_pos, shot in enumerate(batch, start=offset):
            item = creative.get(shot.id)
            can_cutaway = _cutaway_run(manifest.shots, shot_pos) <= _MAX_CUTAWAY_RUN
            _apply_creative(
                shot,
                item,
                manifest.visual_bible,
                allow_multicut=config.planning.allow_multicut_shots,
                lipsync_widest=config.planning.lipsync_widest_framing,
                allow_broll_conversion=can_cutaway,
                characters=cast_for_guards,
            )
            if not shot.type.shows_singer:
                shot.character = None
                shot.characters = []
                shot.singers = []
                shot.wardrobe = None
                shot.character_wardrobes = {}
                continue
            if not manifest.characters:
                shot.singers = []
                shot.wardrobe = wardrobe.assign(item.wardrobe if item else None, shot.id)
                continue

            enforced_singers = _resolve_shot_singers(
                shot,
                analysis,
                manifest.characters,
                voice_map,
                solo_vocals=config.planning.vocals == "solo",
            )
            section_chars = _characters_in_text(shot.section, manifest.characters)
            requested_char: str | None = None
            if item and item.character:
                requested_char = manifest.resolve_character_id(item.character)
                if requested_char is None:
                    log.warning(
                        "%s: unknown character %r; defaulting to section/primary singer",
                        shot.id,
                        item.character,
                    )

            is_together = shot.type is ShotType.LIPSYNC and len(enforced_singers) >= 2
            if is_together:
                resolved_primary: str | None = (
                    requested_char if requested_char in enforced_singers else enforced_singers[0]
                )
            elif shot.type is ShotType.LIPSYNC and len(enforced_singers) == 1:
                resolved_primary = enforced_singers[0]
            else:
                resolved_primary = (
                    requested_char
                    or (enforced_singers[0] if enforced_singers else None)
                    or (section_chars[0] if section_chars else None)
                    or manifest.primary_character_id()
                )
            if resolved_primary is None:
                continue
            primary_id = resolved_primary

            if shot.type is ShotType.LIPSYNC and len(enforced_singers) == 1:
                _fix_mismatched_singer_prompt(shot, primary_id, manifest.characters)

            visible_ids = [primary_id]
            if is_together:
                for cid in enforced_singers:
                    if cid not in visible_ids:
                        visible_ids.append(cid)
            if item and item.characters:
                mentioned_in_prompt = set(
                    _characters_in_text(shot.still_prompt, manifest.characters)
                )
                has_plural_cue = bool(_PLURAL_CAST_CUE_RE.search(shot.still_prompt))
                for raw_cid in item.characters:
                    resolved = manifest.resolve_character_id(raw_cid)
                    if resolved is None:
                        log.warning(
                            "%s: ignoring unknown character %r in characters list", shot.id, raw_cid
                        )
                    elif resolved not in visible_ids:
                        if is_together or resolved in mentioned_in_prompt or has_plural_cue:
                            visible_ids.append(resolved)
                        else:
                            log.info(
                                "%s: dropping unmentioned secondary character %r from solo shot",
                                shot.id,
                                resolved,
                            )
            for mentioned_id in _characters_in_text(
                shot.still_prompt, manifest.characters, proper_noun_only=True
            ):
                if mentioned_id not in visible_ids:
                    visible_ids.append(mentioned_id)

            if (
                shot.type is ShotType.LIPSYNC
                and not enforced_singers
                and len(visible_ids) >= 2
                and _TOGETHER_PROMPT_RE.search(f"{shot.still_prompt} {shot.motion_prompt}")
            ):
                is_together = True
                enforced_singers = list(visible_ids)

            if shot.type is ShotType.LIPSYNC:
                if is_together:
                    shot.singers = [
                        primary_id,
                        *(cid for cid in enforced_singers if cid != primary_id),
                    ]
                    if shot.framing is Framing.CLOSE_UP:
                        shot.framing = Framing.MEDIUM_CLOSE
                    mentioned_singers = set(
                        _characters_in_text(shot.still_prompt, manifest.characters)
                    )
                    missing_co_singer = (
                        len(shot.singers) >= 2
                        and len(mentioned_singers & set(shot.singers)) < 2
                        and not _PLURAL_CAST_CUE_RE.search(shot.still_prompt)
                    )
                    if item is None or missing_co_singer:
                        if missing_co_singer:
                            log.warning(
                                "%s: together-singing still_prompt omitted co-singer; "
                                "replacing with two-shot direction",
                                shot.id,
                            )
                        names = prompts.join_names(
                            [manifest.characters[cid].name or cid for cid in shot.singers]
                        )
                        energy = (
                            "high-energy" if pace_label(shot.intensity) == "high" else "restrained"
                        )
                        shot.still_prompt = (
                            f"Medium close-up two-shot of {names} singing together directly to "
                            f"camera, {energy} mood, {shot.setting}."
                        )
                        shot.motion_prompt = (
                            f"{names} sing the line together in harmony with natural head "
                            "movement. Steady camera."
                        )
                else:
                    shot.singers = [primary_id]
                    if shot.framing is Framing.CLOSE_UP:
                        visible_ids = [primary_id]
            else:
                shot.singers = []

            shot.character = primary_id
            shot.characters = visible_ids
            shot.wardrobe = cast_wardrobes[primary_id].assign(
                item.wardrobe if item else None, shot.id
            )
            if primary_id == manifest.primary_character_id():
                wardrobe.assign(shot.wardrobe, shot.id)
            char_wardrobes: dict[str, str] = {}
            for cid in visible_ids:
                w = shot.wardrobe if cid == primary_id else cast_wardrobes[cid].current
                if w is not None:
                    char_wardrobes[cid] = w
            shot.character_wardrobes = char_wardrobes


def _clock(seconds: float) -> str:
    return f"{int(seconds // 60)}:{int(seconds % 60):02d}"


def _ask_for_batch(
    manifest: Manifest,
    batch: list[Shot],
    offset: int,
    current_wardrobe: str | None,
    cast_wardrobes: dict[str, _Wardrobe] | None,
    llm: LLMClient,
    *,
    allow_multicut: bool = False,
    analysis: Analysis | None = None,
    voice_map: dict[str, str] | None = None,
    style_text: str = "",
    video_style: VideoStyle = VideoStyle.LIPSYNC,
    solo_vocals: bool = False,
) -> dict[str, ShotCreative]:
    bible, character = manifest.visual_bible, manifest.character
    lines = []
    interp = bible.interpretation
    if interp is not None:
        interp_lines = ["SONG INTERPRETATION:"]
        if interp.meaning:
            interp_lines.append(f"Meaning: {interp.meaning}")
        if interp.tone:
            interp_lines.append(f"Tone: {interp.tone}")
        if interp.music_feel:
            interp_lines.append(f"Music feel: {interp.music_feel}")
        if interp.visual_motifs:
            interp_lines.append("Visual motifs: " + "; ".join(interp.visual_motifs))
        if len(interp_lines) > 1:
            lines.append("\n".join(interp_lines))
    if style_text.strip() and not (interp and interp.music_feel):
        compact_style = " ".join(style_text.split())
        lines.append(f"MUSIC STYLE: {compact_style[:500]}")
    if bible.concept:
        lines.append(f"CONCEPT: {bible.concept}")
    if interp is not None and interp.section_beats:
        seen_beats: set[str] = set()
        beat_lines: list[str] = []
        for s in batch:
            beat = bible.find_section_beat(s.section)
            if beat is None:
                continue
            key = beat.section.casefold()
            if key in seen_beats:
                continue
            seen_beats.add(key)
            details = [
                f"{label}: {val}"
                for label, val in (
                    ("emotion", beat.emotion),
                    ("visual beat", beat.visual_beat),
                    ("camera/energy", beat.camera_energy),
                )
                if val
            ]
            if details:
                beat_lines.append(f"- {beat.section} | " + " | ".join(details))
        if beat_lines:
            lines.append("SECTION ARCS FOR THIS BATCH:\n" + "\n".join(beat_lines))
    if bible.rules:
        lines.append("PRODUCTION RULES:\n" + "\n".join(f"- {rule}" for rule in bible.rules))
    lines.append(
        f"VISUAL BIBLE\nStyle: {bible.style}\nPalette: {bible.palette}\nLighting: {bible.lighting}"
    )
    if len(manifest.characters) > 1:
        char_lines = ["CHARACTERS (use these exact character ids):"]
        for cid, char in manifest.characters.items():
            w_summary = "; ".join(f"{w.id}: {w.description}" for w in char.wardrobe) or "base"
            always = f" | always: {pinned}" if (pinned := char.pinned()) else ""
            char_lines.append(
                f"- id={cid} ({char.name or cid}) | appearance: {char.appearance} "
                f"| wardrobe: {w_summary}{always}"
            )
        lines.append("\n".join(char_lines))
    elif character.name or character.appearance:
        always = f" | always: {pinned}" if (pinned := character.pinned()) else ""
        lines.append(
            f"CHARACTER (always refer to them by name, never as 'lead vocal' or a snake_case id): "
            f"{character.name or 'the lead singer'} | appearance: {character.appearance}{always}"
        )
    if character.wardrobe:
        lines.append(
            "WARDROBE STATES (in order of appearance):\n"
            + "\n".join(f"- {w.id}: {w.description}" for w in character.wardrobe)
        )
    if bible.settings:
        lines.append(
            "SETS (use these exact names):\n"
            + "\n".join(
                f"- {s.name} | sections: {', '.join(s.sections) or 'any'} | look: {s.description} "
                f"| movement: {s.movement}"
                for s in bible.settings
            )
        )
    if bible.props:
        lines.append(
            "PROPS (use these exact ids when visible):\n"
            + "\n".join(
                f"- id={p.id} ({p.name}) | sections: {', '.join(p.sections) or 'any'} "
                f"| look: {p.description}"
                for p in bible.props
            )
        )
    previous = manifest.shots[max(0, offset - _CONTEXT_SHOTS) : offset]
    if previous:
        lines.append(
            "ALREADY DIRECTED (for continuity):\n"
            + "\n".join(
                f"- {s.id} section={s.section!r} setting={s.setting!r} "
                f"framing={s.framing.value} wardrobe={s.wardrobe}"
                + (f" character={s.character}" if s.character else "")
                for s in previous
            )
        )
    if current_wardrobe:
        lines.append(f"CURRENT WARDROBE STATE: {current_wardrobe}")
    if cast_wardrobes and len(manifest.characters) > 1:
        active_states = ", ".join(
            f"{cid}={tracker.current}" for cid, tracker in cast_wardrobes.items() if tracker.current
        )
        if active_states:
            lines.append(f"CURRENT WARDROBE BY CHARACTER: {active_states}")
    total = manifest.duration or 1.0
    shot_lines = []
    last_pace = pace_label(previous[-1].intensity) if previous else None
    for s in batch:
        pace = pace_label(s.intensity)
        pace_note = f"pace={pace}" + (
            f" (was {last_pace})" if last_pace is not None and pace != last_pace else ""
        )
        last_pace = pace
        singer_hint = ""
        if video_style is VideoStyle.LIPSYNC and len(manifest.characters) > 1:
            resolved_singers = _resolve_shot_singers(
                s,
                analysis,
                manifest.characters,
                voice_map,
                solo_vocals=solo_vocals,
            )
            if len(resolved_singers) >= 2:
                joined = ", ".join(repr(cid) for cid in resolved_singers)
                singer_hint = f"vocal='together' singers=[{joined}] "
            elif len(resolved_singers) == 1:
                singer_hint = f"vocal='solo' singer={resolved_singers[0]!r} "
            elif not solo_vocals:
                sec_chars = _characters_in_text(s.section, manifest.characters)
                if sec_chars:
                    singer_hint = f"vocal='solo' singer={sec_chars[0]!r} "
        stem_hint = ""
        if analysis is not None and analysis.stems is not None:
            active_stems = active_stems_in_span(analysis.stems, s.start, s.end)
            if active_stems:
                stem_hint = f" stems=[{', '.join(active_stems)}]"
        shot_lines.append(
            f"- id={s.id} section={s.section or '(instrumental lead-in)'!r} {singer_hint}"
            f"at={_clock(s.start)} ({s.start / total:.0%} through the song) type={s.type.value} "
            f"duration={s.duration:.1f}s intensity={s.intensity:.2f} {pace_note}{stem_hint} "
            f"lyrics={s.lyrics or '(instrumental)'!r}"
        )
    lines.append("SHOTS\n" + "\n".join(shot_lines))
    if allow_multicut:
        lines.append(
            "MONTAGE MODE ENABLED: For high-energy spans (>= 4s), you may set allow_cuts=true and "
            "write timestamped multi-angle beats (e.g. '0-2s: overhead... 2-4s: low angle...') "
            "with hard cuts between distinct angles."
        )
    if video_style is VideoStyle.VISUAL:
        shots_system = prompts.VISUAL_SHOTS_SYSTEM
    elif video_style is VideoStyle.NARRATIVE:
        shots_system = f"{prompts.SHOTS_SYSTEM}\n\n{prompts.NARRATIVE_SHOTS_ADDENDUM}"
    else:
        shots_system = prompts.SHOTS_SYSTEM
    reply = llm.complete_json(
        system=shots_system,
        content=[text_block("\n\n".join(lines))],
        schema=ShotBatchReply,
    )
    return {item.id: item for item in reply.shots}


def _apply_rule_based_with_set(
    shot: Shot,
    bible: VisualBible,
    raw_setting: str,
    *,
    raw_props: Sequence[str] = (),
    raw_framing: Framing | None = None,
) -> None:
    """Apply rule-based direction while keeping a valid set (and b-roll props) from the bible."""
    shot.setting = _resolve_set(bible, raw_setting, shot.id)
    shot.allow_cuts = False
    shot.props = (
        _resolve_props(bible, list(raw_props), shot.id) if shot.type is ShotType.BROLL else []
    )
    _apply_rule_based(shot, bible=bible, preferred_framing=raw_framing)


def _broll_mentions_cast(text: str, characters: dict[str, Character] | None) -> list[str]:
    """Cast ids explicitly placed in a b-roll prompt by proper name or snake_case id."""
    if not text or not characters:
        return []
    found = _characters_in_text(text, characters, proper_noun_only=True)
    for cid in characters:
        if "_" in cid and cid not in found and re.search(rf"\b{re.escape(cid)}\b", text):
            found.append(cid)
    return found


_WIDER_SHOT_LABEL_RE = re.compile(r"\b(?:wide|full[\s-]body|medium)\s+shot\b", re.IGNORECASE)
_LOWER_CROP_PHRASE_RE = re.compile(
    r"\bfrom\s+(?:the|his|her|their)\s+(?:waist|hips?|knees?)\s+up\b", re.IGNORECASE
)
_TIGHT_FEET_SENTENCE_RE = re.compile(
    r"\b(?:(?:around|at)\s+(?:his|her|their)\s+feet|bare\s+feet|is\s+barefoot)\b",
    re.IGNORECASE,
)
_UPPER_BODY_KEEP_RE = re.compile(
    r"\b(?:face|eyes?|mouth|lips?|jaw|cheek|hair|neck|shoulders?|chest|blazer|jacket|coat|shirt|hands?)\b",
    re.IGNORECASE,
)


def _reconcile_tight_framing_prompt(text: str, framing: Framing) -> str:
    """Align raw wide/waist-up prose with clamped ``close_up``/``medium_close`` framing."""
    if framing not in (Framing.CLOSE_UP, Framing.MEDIUM_CLOSE) or not text:
        return text
    shot_label = "close-up" if framing is Framing.CLOSE_UP else "medium close-up"
    crop_label = "head-and-shoulders" if framing is Framing.CLOSE_UP else "from the chest up"

    def _sub_label(match: re.Match[str]) -> str:
        raw = match.group(0)
        return shot_label.capitalize() if raw[0].isupper() else shot_label

    updated = _WIDER_SHOT_LABEL_RE.sub(_sub_label, text)
    updated = _LOWER_CROP_PHRASE_RE.sub(crop_label, updated)
    kept_sentences = [
        s
        for s in re.split(r"(?<=[.!?])\s+", updated.strip())
        if s and not (_TIGHT_FEET_SENTENCE_RE.search(s) and not _UPPER_BODY_KEEP_RE.search(s))
    ]
    return " ".join(kept_sentences) if kept_sentences else updated


def _apply_creative(
    shot: Shot,
    creative: ShotCreative | None,
    bible: VisualBible,
    *,
    allow_multicut: bool = False,
    lipsync_widest: Framing = Framing.MEDIUM_CLOSE,
    allow_broll_conversion: bool = True,
    characters: dict[str, Character] | None = None,
) -> None:
    """Copy the model's choices onto a shot, enforcing the rules it must not break."""
    if creative is None:
        if not shot.setting and bible.settings:
            shot.setting = bible.settings[0].name
        _apply_rule_based(shot)
        return
    if shot.type is ShotType.BROLL and creative.type is not ShotType.BROLL:
        log.warning(
            "%s: model tried to put the singer back in a cutaway (%s); using rule-based cutaway",
            shot.id,
            creative.type.value,
        )
        _apply_rule_based_with_set(
            shot, bible, creative.setting, raw_props=creative.props, raw_framing=creative.framing
        )
        return
    if shot.type is ShotType.BROLL and _broll_mentions_people(
        creative.still_prompt, characters, framing=creative.framing or shot.framing
    ):
        log.warning(
            "%s: broll still_prompt puts people/characters on camera; using rule-based cutaway",
            shot.id,
        )
        _apply_rule_based_with_set(
            shot, bible, creative.setting, raw_props=creative.props, raw_framing=creative.framing
        )
        return
    if shot.type is ShotType.BROLL:
        leaked_cast = _broll_mentions_cast(creative.still_prompt, characters)
        if leaked_cast:
            log.warning(
                "%s: broll still_prompt puts cast member(s) %s on camera; using rule-based cutaway",
                shot.id,
                ", ".join(leaked_cast),
            )
            _apply_rule_based_with_set(
                shot,
                bible,
                creative.setting,
                raw_props=creative.props,
                raw_framing=creative.framing,
            )
            return
    if shot.type is ShotType.PERFORMANCE and creative.type is not ShotType.LIPSYNC:
        effective_type = creative.type
        if effective_type is ShotType.PERFORMANCE and _is_empty_scene_prompt(
            creative.still_prompt, characters
        ):
            if allow_broll_conversion:
                log.info(
                    "%s: performance still_prompt describes an empty scene with no performer; "
                    "converting to broll cutaway",
                    shot.id,
                )
                effective_type = ShotType.BROLL
            else:
                log.warning(
                    "%s: performance still_prompt omitted the performer after %d cutaways; "
                    "using rule-based performance direction",
                    shot.id,
                    _MAX_CUTAWAY_RUN,
                )
                _apply_rule_based_with_set(shot, bible, creative.setting)
                return
        if effective_type is ShotType.BROLL and _broll_mentions_cast(
            creative.still_prompt, characters
        ):
            log.info(
                "%s: keeping performance type because still_prompt names a cast member",
                shot.id,
            )
            effective_type = ShotType.PERFORMANCE
        if effective_type is ShotType.BROLL and not allow_broll_conversion:
            log.warning(
                "%s: keeping performance shot with rule-based direction so the singer is not off "
                "screen for more than %d shots in a row",
                shot.id,
                _MAX_CUTAWAY_RUN,
            )
            _apply_rule_based_with_set(shot, bible, creative.setting)
            return
        shot.type = effective_type
    shot.framing = creative.framing or _default_framing(shot)
    if shot.type is ShotType.LIPSYNC and _TIGHT_TO_WIDE.index(shot.framing) > _TIGHT_TO_WIDE.index(
        lipsync_widest
    ):
        shot.framing = lipsync_widest  # a wider shot hides the mouth
    shot.setting = _resolve_set(bible, creative.setting, shot.id)
    shot.allow_cuts = bool(allow_multicut and creative.allow_cuts)
    shot.props = _resolve_props(bible, creative.props, shot.id)
    shot.still_prompt = (
        _reconcile_tight_framing_prompt(creative.still_prompt, shot.framing)
        if shot.type.shows_singer
        else creative.still_prompt
    )
    if (
        shot.type is ShotType.BROLL
        and creative.type is ShotType.PERFORMANCE
        and (
            _PERFORMER_CUE_RE.search(creative.motion_prompt)
            or bool(characters and _characters_in_text(creative.motion_prompt, characters))
        )
    ):
        shot.motion_prompt = _BROLL_PACE_MOTION[pace_label(shot.intensity)]
    else:
        shot.motion_prompt = creative.motion_prompt


def _resolve_props(bible: VisualBible, raw_props: list[str], shot_id: str) -> list[str]:
    if not raw_props or not bible.props:
        return []
    resolved: list[str] = []
    for raw in raw_props:
        found = bible.find_prop(raw)
        if found is None:
            log.warning("%s: prop %r is not in the visual bible; ignoring", shot_id, raw)
            continue
        if found.id not in resolved:
            resolved.append(found.id)
    return resolved


def _resolve_set(bible: VisualBible, name: str, shot_id: str) -> str:
    """Map the model's setting onto a set in the bible, so its description is applied."""
    if not bible.settings:
        return name
    exact = bible.find_set(name)
    if exact is not None:
        return exact.name
    wanted = unicodedata.normalize("NFKC", name).strip().casefold()
    for candidate in bible.settings:
        cand_norm = unicodedata.normalize("NFKC", candidate.name).strip().casefold()
        if wanted and cand_norm and (wanted in cand_norm or cand_norm in wanted):
            return candidate.name
    log.warning(
        "%s: setting %r is not a set in the bible; using %r", shot_id, name, bible.settings[0].name
    )
    return bible.settings[0].name


def _default_framing(shot: Shot) -> Framing:
    """The framing the rule-based planner gives this shot."""
    pace = pace_label(shot.intensity)
    if shot.type is ShotType.LIPSYNC:
        if pace == "high":
            return Framing.CLOSE_UP
        if pace == "calm":
            return Framing.MEDIUM_CLOSE
        return _LIPSYNC_FRAMINGS[shot.index % len(_LIPSYNC_FRAMINGS)]
    return _PACE_FRAMING[pace]


def _apply_rule_based(
    shot: Shot,
    *,
    bible: VisualBible | None = None,
    preferred_framing: Framing | None = None,
) -> None:
    pace = pace_label(shot.intensity)
    energy = "high-energy" if pace == "high" else "restrained"
    shot.setting = shot.setting or "performance stage"
    shot.framing = _default_framing(shot)
    if shot.type is ShotType.LIPSYNC:
        shot.still_prompt = (
            f"{_FRAMING_PHRASE[shot.framing]} of the singer singing directly to camera, "
            f"{energy} mood, {shot.setting}."
        )
        shot.motion_prompt = (
            "The singer performs the line to camera with natural head movement. Slow push-in."
        )
        return
    if shot.type is ShotType.BROLL:
        setting_clean = shot.setting.strip()
        setting_bare = (
            setting_clean[4:].strip() if setting_clean.lower().startswith("the ") else setting_clean
        ) or "room"
        prop = bible.find_prop(shot.props[0]) if (bible is not None and shot.props) else None
        if prop is not None:
            shot.framing = (
                preferred_framing
                if preferred_framing in (Framing.CLOSE_UP, Framing.MEDIUM_CLOSE)
                else Framing.CLOSE_UP
            )
            prop_clean = prop.name.strip()
            prop_ref = prop_clean if prop_clean.lower().startswith("the ") else f"the {prop_clean}"
            shot.still_prompt = (
                f"{_FRAMING_PHRASE[shot.framing]} of {prop_ref} resting inside the uninhabited "
                f"{setting_bare}, empty architectural space, {energy} mood."
            )
        else:
            shot.still_prompt = (
                f"{_FRAMING_PHRASE[shot.framing]} of an atmospheric architectural detail inside "
                f"the uninhabited {setting_bare}, empty space, {energy} mood."
            )
        shot.motion_prompt = _BROLL_PACE_MOTION[pace]
        return
    shot.still_prompt = (
        f"{_FRAMING_PHRASE[shot.framing]} of the singer moving to the music, not singing, "
        f"{energy} mood, {shot.setting}."
    )
    shot.motion_prompt = (
        f"The singer moves with the music, mouth closed, not singing. {_PACE_MOTION[pace]}"
    )
