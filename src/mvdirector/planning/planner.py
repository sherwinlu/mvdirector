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
import re
import statistics
import unicodedata
from collections import Counter
from collections.abc import Callable

from pydantic import BaseModel, ConfigDict, Field, field_validator

from mvdirector import tuning as tuning_mod
from mvdirector.audio.lyrics import (
    is_hook_section,
    is_syllabic_char,
    labels_have_multiple_singers,
    parse_lyrics,
    section_singers,
)
from mvdirector.config import ProjectConfig
from mvdirector.errors import LLMError
from mvdirector.llm import LLMClient, text_block
from mvdirector.models import (
    BOTH_VOICES,
    Analysis,
    AssetState,
    AudioSlice,
    Character,
    Framing,
    Manifest,
    PropDesign,
    SetDesign,
    Shot,
    ShotType,
    TimeSpan,
    VisualBible,
    WardrobeState,
    ascii_fold,
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
    "calm": "Light, dust and shadow drift slowly through the frame. No people. Steady camera.",
    "moderate": (
        "Light and shadow shift steadily across the frame with natural movement. "
        "No people. Gentle push-in."
    ),
    "high": (
        "Brisk movement of light and environmental elements across the frame. "
        "No people. Firm, smooth push-in. Camera stays stable."
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
                return slug
            digest = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()[:6]
            return f"{prefix}_{digest}"
    return validate_safe_id(str(value), label)


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


class BibleReply(_Reply):
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
        voices = {t.voice for t in analysis.vocal_turns}
        snap_sections = (
            has_multiple_section_singers(analysis, lyrics_text)
            or len(voices) >= 2
            or BOTH_VOICES in voices
        )
    else:
        snap_sections = config.video.snap_sections == "always"
    manifest.shots = _timed_shots(config, analysis, snap_sections=snap_sections)
    if llm is None:
        log.warning("LLM disabled: using rule-based direction. Edit manifest.json to refine it.")
        manifest.visual_bible = VisualBible(style=" ".join(style_text.split()))
    else:
        _write_bible(manifest, lyrics_text, style_text, llm)
    voice_map = _map_voices_to_characters(analysis, manifest)
    if config.planning.max_lipsync_share is not None:
        # After the bible, so each shot can be matched to its section or acoustic singer.
        _thin_lipsync(
            manifest.shots,
            config.planning.max_lipsync_share,
            lambda shot: _shot_singer_hint(shot, analysis, manifest.characters, voice_map),
        )
    _direct_shots(manifest, config, llm, analysis=analysis, voice_map=voice_map)
    _populate_slice_mutes(manifest, analysis, voice_map)
    _report_mix(manifest.shots, has_bible=llm is not None)
    return manifest


def _write_bible(manifest: Manifest, lyrics_text: str, style_text: str, llm: LLMClient) -> None:
    section_list = "\n".join(f"- {name}" for name in _section_order(lyrics_text)) or "(none)"
    request = (
        f"MUSIC STYLE AND BRIEF:\n{style_text}\n\n"
        f"SONG SECTIONS, IN ORDER:\n{section_list}\n\n"
        f"LYRICS:\n{lyrics_text}"
    )
    try:
        bible = llm.complete_json(
            system=prompts.BIBLE_SYSTEM, content=[text_block(request)], schema=BibleReply
        )
    except LLMError as exc:
        raise LLMError(
            f"Could not get the visual bible from the model: {exc} Run `mvd llm-check` to test the "
            "model, or try a more capable one."
        ) from exc
    characters: dict[str, Character] = {}
    for item in bible.characters:
        if not item.appearance.strip():
            continue
        characters[item.id] = Character(
            name=item.name or item.id,
            appearance=item.appearance,
            invariants=item.invariants,
            wardrobe=[WardrobeState(id=w.id, description=w.description) for w in item.wardrobe],
        )
    if len(characters) > 1:
        primary = next(iter(characters.values()))
        if not primary.wardrobe and bible.wardrobe:
            primary.wardrobe = [
                WardrobeState(id=w.id, description=w.description) for w in bible.wardrobe
            ]
        if not primary.invariants and bible.invariants:
            primary.invariants = bible.invariants
        manifest.characters = characters
        manifest.character = primary
    elif len(characters) == 1:
        single = next(iter(characters.values()))
        if not single.wardrobe and bible.wardrobe:
            single.wardrobe = [
                WardrobeState(id=w.id, description=w.description) for w in bible.wardrobe
            ]
        if not single.invariants and bible.invariants:
            single.invariants = bible.invariants
        manifest.character = single
    else:
        manifest.character = Character(
            appearance=bible.appearance,
            invariants=bible.invariants,
            wardrobe=[WardrobeState(id=w.id, description=w.description) for w in bible.wardrobe],
        )
    manifest.visual_bible = VisualBible(
        concept=bible.concept,
        style=bible.style,
        palette=bible.palette,
        lighting=bible.lighting,
        negative_prompt=bible.negative_prompt,
        settings=[SetDesign(**s.model_dump()) for s in bible.settings],
        props=[PropDesign(**p.model_dump()) for p in bible.props],
        rules=bible.rules,
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
    video, shots = config.video, []
    for index, seg in enumerate(segment(analysis, video, snap_sections=snap_sections)):
        shot_id = f"s{index + 1:03d}"
        slice_start = max(0.0, seg.start - video.handle_seconds)
        slice_end = min(analysis.duration, seg.end + video.handle_seconds)
        shots.append(
            Shot(
                id=shot_id,
                index=index,
                start=seg.start,
                end=seg.end,
                type=seg.type,
                section=seg.section,
                lyrics=seg.lyrics,
                intensity=seg.intensity,
                audio_slice=AudioSlice(
                    path=f"assets/audio_slices/{shot_id}.wav",
                    start=round(slice_start, 3),
                    end=round(slice_end, 3),
                    head_handle=round(seg.start - slice_start, 3),
                ),
                num_frames=frames_for(
                    slice_end - slice_start, video.fps, video.frame_multiple, video.frame_offset
                ),
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


def _character_tokens(cid: str, char: Character) -> list[str]:
    tokens = [cid.strip()]
    if char.name and char.name.strip():
        clean_name = unicodedata.normalize("NFC", char.name.strip())
        tokens.append(clean_name)
        folded_name = ascii_fold(clean_name).strip()
        if folded_name and folded_name != clean_name:
            tokens.append(folded_name)
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
) -> list[str]:
    """Resolve active singing character ids for ``shot`` (1 = solo, 2+ = singing together)."""
    if not characters:
        return []
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
) -> str | None:
    """Return the solo singer id for ``shot``, or ``None`` when shared ('both') or untagged."""
    if characters:
        resolved = _resolve_shot_singers(shot, analysis, characters, voice_map)
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


def _populate_slice_mutes(
    manifest: Manifest,
    analysis: Analysis,
    voice_map: dict[str, str],
) -> None:
    """Record foreign-singer mute spans on lip-sync slices so voices never bleed across cuts."""
    shots = manifest.shots
    multi_char = len(manifest.characters) > 1
    for idx, shot in enumerate(shots):
        if shot.type is not ShotType.LIPSYNC:
            shot.audio_slice.mute_spans = []
            continue
        s_start, s_end = shot.audio_slice.start, shot.audio_slice.end
        active = manifest.active_singer_ids(shot)
        is_together = len(active) >= 2
        handle_mutes: list[tuple[float, float]] = []
        body_mutes: list[tuple[float, float]] = []

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
                )
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
                characters=manifest.characters,
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

            enforced_singers = _resolve_shot_singers(shot, analysis, manifest.characters, voice_map)
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

            visible_ids = [primary_id]
            if is_together:
                for cid in enforced_singers:
                    if cid not in visible_ids:
                        visible_ids.append(cid)
            if item and item.characters:
                for raw_cid in item.characters:
                    resolved = manifest.resolve_character_id(raw_cid)
                    if resolved is None:
                        log.warning(
                            "%s: ignoring unknown character %r in characters list", shot.id, raw_cid
                        )
                    elif resolved not in visible_ids:
                        visible_ids.append(resolved)
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
                    if item is None:
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
                    if len(enforced_singers) == 1:
                        _fix_mismatched_singer_prompt(shot, primary_id, manifest.characters)
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
) -> dict[str, ShotCreative]:
    bible, character = manifest.visual_bible, manifest.character
    lines = []
    if bible.concept:
        lines.append(f"CONCEPT: {bible.concept}")
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
        if len(manifest.characters) > 1:
            resolved_singers = _resolve_shot_singers(s, analysis, manifest.characters, voice_map)
            if len(resolved_singers) >= 2:
                joined = ", ".join(repr(cid) for cid in resolved_singers)
                singer_hint = f"vocal='together' singers=[{joined}] "
            elif len(resolved_singers) == 1:
                singer_hint = f"vocal='solo' singer={resolved_singers[0]!r} "
            else:
                sec_chars = _characters_in_text(s.section, manifest.characters)
                if sec_chars:
                    singer_hint = f"vocal='solo' singer={sec_chars[0]!r} "
        shot_lines.append(
            f"- id={s.id} section={s.section or '(instrumental lead-in)'!r} {singer_hint}"
            f"at={_clock(s.start)} ({s.start / total:.0%} through the song) type={s.type.value} "
            f"duration={s.duration:.1f}s intensity={s.intensity:.2f} {pace_note} "
            f"lyrics={s.lyrics or '(instrumental)'!r}"
        )
    lines.append("SHOTS\n" + "\n".join(shot_lines))
    if allow_multicut:
        lines.append(
            "MONTAGE MODE ENABLED: For high-energy spans (>= 4s), you may set allow_cuts=true and "
            "write timestamped multi-angle beats (e.g. '0-2s: overhead... 2-4s: low angle...') "
            "with hard cuts between distinct angles."
        )
    reply = llm.complete_json(
        system=prompts.SHOTS_SYSTEM,
        content=[text_block("\n\n".join(lines))],
        schema=ShotBatchReply,
    )
    return {item.id: item for item in reply.shots}


def _apply_rule_based_with_set(shot: Shot, bible: VisualBible, raw_setting: str) -> None:
    """Apply rule-based direction while keeping a valid set from the visual bible."""
    shot.setting = _resolve_set(bible, raw_setting, shot.id)
    shot.allow_cuts = False
    shot.props = []
    _apply_rule_based(shot)


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
        _apply_rule_based(shot)
        return
    if shot.type is ShotType.BROLL and creative.type is not ShotType.BROLL:
        log.warning(
            "%s: model tried to put the singer back in a cutaway (%s); using rule-based cutaway",
            shot.id,
            creative.type.value,
        )
        _apply_rule_based_with_set(shot, bible, creative.setting)
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
    shot.still_prompt = creative.still_prompt
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
        cand_norm = unicodedata.normalize("NFKC", candidate.name).casefold()
        if wanted and (wanted in cand_norm or cand_norm in wanted):
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


def _apply_rule_based(shot: Shot) -> None:
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
        shot.still_prompt = (
            f"{_FRAMING_PHRASE[shot.framing]} of an object or corner of the {shot.setting} that "
            f"echoes the song, no people, {energy} mood."
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
