# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Build the shot manifest: deterministic timing plus creative direction.

Creative direction comes from the language model when one is available. Without it,
a rule-based director fills the same fields so the pipeline still runs end to end.
"""

from __future__ import annotations

import logging
import re
import statistics
from collections import Counter
from collections.abc import Callable

from pydantic import BaseModel, ConfigDict, Field, field_validator

from mvdirector.audio.lyrics import parse_lyrics
from mvdirector.config import ProjectConfig
from mvdirector.errors import LLMError
from mvdirector.llm import LLMClient, text_block
from mvdirector.models import (
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
    VisualBible,
    WardrobeState,
    validate_safe_id,
)
from mvdirector.planning import prompts
from mvdirector.planning.segmenter import frames_for, segment

log = logging.getLogger(__name__)

_LIPSYNC_FRAMINGS = [Framing.CLOSE_UP, Framing.MEDIUM_CLOSE]
# Sections whose sung shots stay on camera when the singing share is capped.
# Pre-chorus and post-chorus are excluded so only actual chorus/hook sections stay protected.
_NON_HOOK_CHORUS_RE = re.compile(r"\b(?:pre|post)[\s-]?chorus\b", re.IGNORECASE)
_HOOK_SECTION_RE = re.compile(r"\b(?:chorus|hook)\b", re.IGNORECASE)
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


def _slug_id(value: object, label: str = "id") -> str:
    if isinstance(value, str):
        slug = re.sub(r"[^a-z0-9_-]+", "_", value.strip().lower()).strip("_-")
        if slug and slug[0].isalnum():
            return slug
    return validate_safe_id(str(value), label)


class WardrobeReply(_Reply):
    id: str
    description: str

    @field_validator("id", mode="before")
    @classmethod
    def _check_id(cls, value: object) -> str:
        return validate_safe_id(str(value), "wardrobe state id")


class CharacterReply(_Reply):
    id: str
    name: str = ""
    appearance: str
    invariants: str = ""
    wardrobe: list[WardrobeReply] = Field(default_factory=list)

    @field_validator("id", mode="before")
    @classmethod
    def _check_id(cls, value: object) -> str:
        return _slug_id(value, "character id")


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
        return _slug_id(value, "prop id")


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
    manifest = Manifest(
        title=config.title,
        duration=analysis.duration,
        fps=config.video.fps,
    )
    # Derive multi-singer section snapping deterministically from the lyric section tags
    # BEFORE calling the LLM, so cut points never vary with model output.
    if config.video.snap_sections == "auto":
        snap_sections = has_multiple_section_singers(analysis, lyrics_text)
    else:
        snap_sections = config.video.snap_sections == "always"
    manifest.shots = _timed_shots(config, analysis, snap_sections=snap_sections)
    if llm is None:
        log.warning("LLM disabled: using rule-based direction. Edit manifest.json to refine it.")
        manifest.visual_bible = VisualBible(style=" ".join(style_text.split()))
    else:
        _write_bible(manifest, lyrics_text, style_text, llm)
    if config.planning.max_lipsync_share is not None:
        # After the bible, so each shot's section can be matched to the singer it names.
        _thin_lipsync(
            manifest.shots,
            config.planning.max_lipsync_share,
            lambda shot: _section_singer(shot, manifest.characters),
        )
    _direct_shots(manifest, config, llm)
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


_STRUCTURAL_SECTION_WORDS = frozenset(
    {
        "intro",
        "verse",
        "pre-chorus",
        "prechorus",
        "pre",
        "chorus",
        "post-chorus",
        "postchorus",
        "post",
        "hook",
        "big",
        "refrain",
        "bridge",
        "outro",
        "interlude",
        "instrumental",
        "solo",
        "break",
        "breakdown",
        "drop",
        "build",
        "buildup",
        "build-up",
        "tag",
        "coda",
        "spoken",
        "adlib",
        "ad-lib",
        "adlibs",
        "ad-libs",
        "rap",
        "fade",
        "ending",
        "end",
        "part",
        "section",
        "final",
        "reprise",
        "repeat",
        "repeated",
        "again",
        "vamp",
        "turnaround",
        "soft",
        "loud",
        "quiet",
        "whisper",
        "whispered",
        "acoustic",
        "heavy",
        "stripped",
        "building",
        "calm",
        "intense",
        "acapella",
        "capella",
        "fast",
        "slow",
        "high",
        "low",
        "emotional",
        "softly",
        "gently",
        "out",
        "in",
        "up",
        "down",
        "lead",
        "backing",
        "backup",
        "vocal",
        "vocals",
        "vox",
    }
)
_ENSEMBLE_SECTION_WORDS = frozenset(
    {"duet", "both", "together", "all", "everyone", "ensemble", "choir", "group", "harmony"}
)
# Generic performer designators that name a singer without a proper name ("[Female Verse]").
_DESIGNATOR_WORDS = frozenset(
    {"male", "female", "man", "woman", "boy", "girl", "guy", "lady", "he", "she", "him", "her"}
)
# Designators that only identify a singer together with a number ("[Singer 2]", "Voice 1").
_NUMBERED_DESIGNATORS = frozenset(
    {"singer", "voice", "vocalist", "vocal", "vocals", "rapper", "character", "performer"}
)
_SINGER_SPLIT = re.compile(r"\s*(?:&|\+|/|\band\b)\s*", flags=re.IGNORECASE)
_REPEAT_MARKER = re.compile(r"\s*[(\[]?\s*(?:x|×)\s*\d+\s*[)\]]?\s*$", flags=re.IGNORECASE)
_BRACKETED_CUE = re.compile(r"^(.*?)\s*[(\[]([^()\[\]]+)[)\]]\s*$")
_DASH_SPLIT = re.compile(r"\s+[-—–]\s+")


def _is_structural(words: list[str]) -> bool:
    core = [w for w in (re.sub(r"^\d+|\d+$", "", w) for w in words) if w]
    return bool(core) and all(w in _STRUCTURAL_SECTION_WORDS for w in core)


def _looks_like_name(text: str) -> bool:
    return all(w[:1].isupper() for w in text.split())


def _singers_in_cue(cue: str, *, explicit: bool, cased: bool) -> set[str] | None:
    """Performer keys named by one section cue, or ``None`` when the cue names an ensemble.

    ``explicit`` cues come from a performer form (``Label: Cue``, ``Label (Cue)``,
    ``Label - Cue``) and are trusted more than a bare label, which only counts when it is a
    capitalised proper name or a generic designator such as ``Female``.
    """
    singers: set[str] = set()
    head = cue.split(",", 1)[0].strip()
    for raw in _SINGER_SPLIT.split(head):
        part = _REPEAT_MARKER.sub("", raw.strip())
        if not part:
            continue
        words = part.lower().split()
        if " ".join(words) in _ENSEMBLE_SECTION_WORDS:
            return None
        if len(words) == 2 and words[0] in _NUMBERED_DESIGNATORS and words[1].isdigit():
            singers.add(f"{words[0]} {words[1]}")
            continue
        core = [re.sub(r"^\d+|\d+$", "", w) for w in words]
        core = [w for w in core if w]
        if not core:
            continue
        if any(w in _STRUCTURAL_SECTION_WORDS for w in core):
            # "Female Verse", "Male Chorus": a designator attached to a structural label.
            rest = [w for w in core if w not in _STRUCTURAL_SECTION_WORDS]
            if rest and all(w in _DESIGNATOR_WORDS for w in rest):
                singers.add(" ".join(rest))
            continue
        if all(w in _DESIGNATOR_WORDS for w in core):
            singers.add(" ".join(core))
            continue
        if len(core) > 3 or any(w.isdigit() for w in words):
            continue
        if explicit or (cased and _looks_like_name(part)):
            singers.add(" ".join(core))
    return singers


def _section_cue(label: str) -> tuple[str, bool]:
    """Split a section label into its performer cue and whether that cue was explicit."""
    label = _REPEAT_MARKER.sub("", label.strip())
    bracketed = _BRACKETED_CUE.match(label)
    if bracketed:
        return bracketed.group(2).strip(), True
    if ":" in label:
        return label.split(":", 1)[1].strip(), True
    if _DASH_SPLIT.search(label):
        lhs, rhs = _DASH_SPLIT.split(label, maxsplit=1)
        # A dash often introduces a mood ("Verse 1 - angry"), so only a capitalised name counts.
        return (rhs if _is_structural(lhs.lower().split()) else lhs).strip(), False
    return label, False


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
    singers: set[str] = set()
    for raw in labels:
        label = raw.strip()
        if not label:
            continue
        cue, explicit = _section_cue(label)
        if not cue:
            continue
        found = _singers_in_cue(cue, explicit=explicit, cased=label != label.lower())
        if found is None:
            return True
        singers |= found
    return len(singers) >= 2


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


def _is_hook_section(section: str | None) -> bool:
    """True for chorus/hook sections, excluding pre-chorus and post-chorus."""
    cleaned = _NON_HOOK_CHORUS_RE.sub("", section or "")
    return bool(_HOOK_SECTION_RE.search(cleaned))


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
        tokens.append(char.name.strip())
    return list(dict.fromkeys(t for t in tokens if len(t) >= 2 or not char.name))


def _section_singer(shot: Shot, characters: dict[str, Character]) -> str | None:
    """The one character a shot's section names, or None when it names none or several."""
    named = _characters_in_text(shot.section, characters)
    return named[0] if len(named) == 1 else None


def _characters_in_text(
    text: str | None, characters: dict[str, Character], *, proper_noun_only: bool = False
) -> list[str]:
    """Character ids mentioned by id or name in ``text``, in order of appearance.

    When ``proper_noun_only`` is true, only capitalized occurrences match so common words
    matching a character name (such as "grace" or "dawn") are left alone in prose prompts.
    """
    if not text or not characters:
        return []
    found: list[tuple[int, str]] = []
    for cid, char in characters.items():
        positions: list[int] = []
        for token in _character_tokens(cid, char):
            pattern = re.compile(rf"\b{re.escape(token)}\b", flags=re.IGNORECASE)
            for match in pattern.finditer(text):
                if proper_noun_only and not match.group(0)[0].isupper():
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

    def _replace_if_proper(match: re.Match[str]) -> str:
        return target if match.group(0)[0].isupper() else match.group(0)

    for wrong_id in mentioned:
        wrong_char = characters[wrong_id]
        for token in _character_tokens(wrong_id, wrong_char):
            pattern = re.compile(rf"\b{re.escape(token)}\b", flags=re.IGNORECASE)
            shot.still_prompt = pattern.sub(_replace_if_proper, shot.still_prompt)
            shot.motion_prompt = pattern.sub(_replace_if_proper, shot.motion_prompt)


def _direct_shots(manifest: Manifest, config: ProjectConfig, llm: LLMClient | None) -> None:
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
            )
            if not shot.type.shows_singer:
                shot.character = None
                shot.characters = []
                shot.wardrobe = None
                shot.character_wardrobes = {}
                continue
            if not manifest.characters:
                shot.wardrobe = wardrobe.assign(item.wardrobe if item else None, shot.id)
                continue

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
            if shot.type is ShotType.LIPSYNC and len(section_chars) == 1:
                resolved_primary: str | None = section_chars[0]
            else:
                resolved_primary = (
                    requested_char
                    or (section_chars[0] if section_chars else None)
                    or manifest.primary_character_id()
                )
            if resolved_primary is None:
                continue
            primary_id = resolved_primary
            if shot.type is ShotType.LIPSYNC and len(section_chars) == 1:
                _fix_mismatched_singer_prompt(shot, primary_id, manifest.characters)

            visible_ids = [primary_id]
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
            if shot.type is ShotType.LIPSYNC and shot.framing is Framing.CLOSE_UP:
                visible_ids = [primary_id]

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
            sec_chars = _characters_in_text(s.section, manifest.characters)
            if sec_chars:
                singer_hint = f"singer={sec_chars[0]!r} "
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
        if creative.type is ShotType.BROLL and not allow_broll_conversion:
            log.warning(
                "%s: keeping performance shot with rule-based direction so the singer is not off "
                "screen for more than %d shots in a row",
                shot.id,
                _MAX_CUTAWAY_RUN,
            )
            _apply_rule_based_with_set(shot, bible, creative.setting)
            return
        shot.type = creative.type
    shot.framing = creative.framing or _default_framing(shot)
    if shot.type is ShotType.LIPSYNC and _TIGHT_TO_WIDE.index(shot.framing) > _TIGHT_TO_WIDE.index(
        lipsync_widest
    ):
        shot.framing = lipsync_widest  # a wider shot hides the mouth
    shot.setting = _resolve_set(bible, creative.setting, shot.id)
    shot.allow_cuts = bool(allow_multicut and creative.allow_cuts)
    shot.props = _resolve_props(bible, creative.props, shot.id)
    shot.still_prompt = creative.still_prompt
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
    wanted = name.strip().lower()
    for candidate in bible.settings:
        if wanted and (wanted in candidate.name.lower() or candidate.name.lower() in wanted):
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
