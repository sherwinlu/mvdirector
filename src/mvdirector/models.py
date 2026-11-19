# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Data models persisted to disk: the audio analysis and the shot manifest.

The manifest is the single source of truth for a project. Every stage reads it,
does its work and writes it back, so each stage can be re-run on its own.
All paths stored here are relative to the project root so a project folder can
be moved between machines.
"""

from __future__ import annotations

import re
from collections.abc import Collection
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    field_validator,
    model_serializer,
    model_validator,
)

ANALYSIS_SCHEMA_VERSION = 1
MANIFEST_SCHEMA_VERSION = 1
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


def validate_safe_id(value: str, label: str = "id") -> str:
    """Reject identifiers that could traverse paths or inject delimiters."""
    if not SAFE_ID.match(value):
        raise ValueError(
            f"Invalid {label} {value!r}: must contain only letters, digits, '_' or '-'."
        )
    return value


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class StrictModel(BaseModel):
    """Base for persisted models: unknown fields are errors, not silently dropped."""

    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------- analysis


class TimeSpan(StrictModel):
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def overlap(self, start: float, end: float) -> float:
        """Seconds shared between this span and [start, end]."""
        return max(0.0, min(self.end, end) - max(self.start, start))


class LyricLine(StrictModel):
    text: str
    section: str | None = None
    start: float | None = None
    end: float | None = None
    # "aligned" = from word timestamps, "estimated" = spread over vocal activity.
    timing_source: Literal["aligned", "estimated", "none"] = "none"


class Analysis(StrictModel):
    schema_version: int = ANALYSIS_SCHEMA_VERSION
    duration: float
    tempo_bpm: float
    beats: list[float]
    bars: list[float] = Field(description="Start time of each bar, in seconds.")
    bar_intensity: list[float] = Field(description="Loudness of each bar, scaled 0-1.")
    vocal_windows: list[TimeSpan]
    lyric_lines: list[LyricLine] = Field(default_factory=list)

    @field_validator("schema_version")
    @classmethod
    def _check_schema_version(cls, version: int) -> int:
        if version != ANALYSIS_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported analysis schema_version {version}; "
                f"this build reads version {ANALYSIS_SCHEMA_VERSION}."
            )
        return version


# --------------------------------------------------------------------------------- qc


class CheckStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


class CheckResult(StrictModel):
    name: str
    status: CheckStatus
    hard: bool = Field(
        default=False, description="A failed hard check rejects the asset whatever its score."
    )
    score: float | None = Field(default=None, description="0-1, higher is better.")
    weight: float = 1.0
    detail: str = ""
    fix_hint: str | None = Field(
        default=None, description="Advice fed into the next attempt's prompt."
    )
    metrics: dict[str, float] = Field(default_factory=dict)


class QCReport(StrictModel):
    passed: bool
    score: float
    checks: list[CheckResult] = Field(default_factory=list)

    @property
    def failed_checks(self) -> list[CheckResult]:
        return [c for c in self.checks if c.status is CheckStatus.FAILED]

    @property
    def has_hard_failure(self) -> bool:
        return any(c.hard for c in self.failed_checks)

    @property
    def issues(self) -> list[str]:
        return [f"{c.name}: {c.detail}" if c.detail else c.name for c in self.failed_checks]

    @property
    def fix_hints(self) -> list[str]:
        return [c.fix_hint for c in self.failed_checks if c.fix_hint]


# --------------------------------------------------------------------------- manifest


class ShotType(str, Enum):
    LIPSYNC = "lipsync"  # singer on camera, singing
    PERFORMANCE = "performance"  # singer on camera, not singing
    BROLL = "broll"  # no singer

    @property
    def shows_singer(self) -> bool:
        return self is not ShotType.BROLL


class Framing(str, Enum):
    CLOSE_UP = "close_up"
    MEDIUM_CLOSE = "medium_close"
    MEDIUM = "medium"
    WIDE = "wide"


class AssetKind(str, Enum):
    STILL = "still"
    VIDEO = "video"


class AssetStatus(str, Enum):
    PENDING = "pending"
    PASSED = "passed"
    EXHAUSTED = "exhausted"  # attempts used up without a pass; best attempt is kept


class RetryStrategy(str, Enum):
    INITIAL = "initial"
    NEW_SEED = "new_seed"
    REWRITE_PROMPT = "rewrite_prompt"
    ADJUST_PARAMS = "adjust_params"


class Attempt(StrictModel):
    number: int
    strategy: RetryStrategy
    seed: int
    prompt: str
    params: dict[str, Any] = Field(default_factory=dict)
    path: str | None = None
    qc: QCReport | None = None
    error: str | None = None
    created_at: str = Field(default_factory=utc_now)


class AssetState(StrictModel):
    status: AssetStatus = AssetStatus.PENDING
    attempts: list[Attempt] = Field(default_factory=list)
    attempts_allowed: int = 0
    selected_attempt: int | None = None
    prompt_override: str | None = Field(
        default=None, description="Rewritten shot prompt, set by the retry policy."
    )
    param_overrides: dict[str, Any] = Field(default_factory=dict)
    director_note: str | None = Field(
        default=None, description="Reviewer feedback to apply on the next attempt."
    )
    accepted_by_reviewer: bool = Field(
        default=False, description="A person accepted an attempt that did not pass QC."
    )
    upscaled_path: str | None = Field(
        default=None, description="Optional upscaled output derived from the selected attempt."
    )
    upscaled_from_attempt: int | None = Field(
        default=None, description="Attempt number that upscaled_path was generated from."
    )

    @property
    def selected(self) -> Attempt | None:
        for attempt in self.attempts:
            if attempt.number == self.selected_attempt:
                return attempt
        return None

    def pass_attempts(self, *, after_fallback: bool = False) -> list[Attempt]:
        """Attempts belonging to the active pass (after a fallback conversion, if any)."""
        if not after_fallback:
            return self.attempts
        starts = [i for i, a in enumerate(self.attempts) if a.strategy is RetryStrategy.INITIAL]
        return self.attempts[starts[-1] :] if len(starts) >= 2 else []

    def best_attempt(self) -> Attempt | None:
        """Highest-scoring attempt that produced a file; hard failures rank last."""
        starts = [i for i, a in enumerate(self.attempts) if a.strategy is RetryStrategy.INITIAL]
        current_pass = self.attempts[starts[-1] :] if starts else self.attempts
        usable = [a for a in current_pass if a.path and a.qc]
        if not usable:
            return None
        return max(usable, key=lambda a: (not a.qc.has_hard_failure, a.qc.score))  # type: ignore[union-attr]

    def reset(self, extra_attempts: int, note: str | None = None) -> None:
        """Send the asset back for regeneration, keeping history as an audit trail."""
        self.status = AssetStatus.PENDING
        self.selected_attempt = None
        self.accepted_by_reviewer = False
        self.upscaled_path = None
        self.upscaled_from_attempt = None
        self.attempts_allowed = len(self.attempts) + extra_attempts
        self.director_note = note


class ReviewReason(str, Enum):
    EXHAUSTED = "exhausted"
    RETRIED = "retried"
    NEAR_THRESHOLD = "near_threshold"
    SAMPLED = "sampled"
    ESCALATED = "escalated"
    LIPSYNC = "lipsync"
    FALLBACK = "fallback"


class ReviewDecision(str, Enum):
    APPROVED = "approved"
    REJECTED = "rejected"


class Review(StrictModel):
    required: bool = False
    reasons: list[ReviewReason] = Field(default_factory=list)
    decision: ReviewDecision | None = None
    note: str | None = None
    decided_at: str | None = None


class AudioSlice(StrictModel):
    """Audio handed to the video model. Includes handles trimmed off at assembly."""

    path: str
    start: float
    end: float
    head_handle: float = Field(description="Seconds between slice start and shot start.")


class Shot(StrictModel):
    id: str
    index: int
    start: float
    end: float
    type: ShotType
    framing: Framing = Framing.MEDIUM
    section: str | None = Field(
        default=None, description="Song section this shot falls in, from the lyric tags."
    )
    setting: str = Field(default="", description="Name of a set in the visual bible.")
    character: str | None = Field(
        default=None, description="Primary character id on camera / singing in this shot."
    )
    characters: list[str] = Field(
        default_factory=list,
        description="All character ids visible in this shot (for duets, two-shots, or bands).",
    )
    wardrobe: str | None = Field(
        default=None, description="Wardrobe state id worn in the shot's first frame."
    )
    character_wardrobes: dict[str, str] = Field(
        default_factory=dict,
        description="Wardrobe state id per visible character in multi-character shots.",
    )
    lyrics: str = ""
    fallback_from: ShotType | None = Field(
        default=None,
        description="Set when the shot was converted from this type after it could not be made.",
    )
    intensity: float = 0.5
    still_prompt: str = ""
    motion_prompt: str = ""
    allow_cuts: bool = Field(
        default=False,
        description="Opt-in multi-angle montage mode: allows internal cuts inside this clip.",
    )
    props: list[str] = Field(
        default_factory=list,
        description="Recurring hero prop ids or names visible in this shot.",
    )
    previz_video: str | None = Field(
        default=None,
        description="Optional 3D blocking / previz video path guiding camera motion and staging.",
    )
    previz_mapping: str = Field(
        default="",
        description="Stand-in mapping from blocking shapes in previz_video to real subjects/props.",
    )
    audio_slice: AudioSlice
    num_frames: int
    still: AssetState = Field(default_factory=AssetState)
    video: AssetState = Field(default_factory=AssetState)
    review: Review = Field(default_factory=Review)

    @field_validator("id")
    @classmethod
    def _check_id(cls, value: str) -> str:
        return validate_safe_id(value, "shot id")

    @field_validator("character")
    @classmethod
    def _check_character(cls, value: str | None) -> str | None:
        return validate_safe_id(value, "character id") if value is not None else None

    @field_validator("characters")
    @classmethod
    def _check_characters(cls, value: list[str]) -> list[str]:
        return [validate_safe_id(item, "character id") for item in value]

    @field_validator("wardrobe")
    @classmethod
    def _check_wardrobe(cls, value: str | None) -> str | None:
        return validate_safe_id(value, "wardrobe state id") if value is not None else None

    @field_validator("character_wardrobes")
    @classmethod
    def _check_character_wardrobes(cls, value: dict[str, str]) -> dict[str, str]:
        for char_id, state_id in value.items():
            validate_safe_id(char_id, "character id")
            validate_safe_id(state_id, "wardrobe state id")
        return value

    @property
    def duration(self) -> float:
        return self.end - self.start

    def asset(self, kind: AssetKind) -> AssetState:
        return self.still if kind is AssetKind.STILL else self.video


class WardrobeState(StrictModel):
    id: str
    description: str = Field(
        description="Everything worn in this state, including what is absent (e.g. 'no cape')."
    )

    @field_validator("id")
    @classmethod
    def _check_id(cls, value: str) -> str:
        return validate_safe_id(value, "wardrobe state id")


PANEL_NAMES = ("front", "profile", "three_quarter", "face")
COLUMNS_PANEL_NAMES = ("face", "profile", "front", "back")
BASE_STATE = "base"  # reference key used when a character has no wardrobe states


class ReferenceSet(StrictModel):
    """A locked character reference sheet and the panels cut from it."""

    sheet: str | None = Field(default=None, description="The locked reference sheet image.")
    panels: dict[str, str] = Field(
        default_factory=dict,
        description="Panel crops cut from the sheet (front, profile, three_quarter, face, back).",
    )
    video: str | None = Field(
        default=None,
        description="Optional locked silent turnaround and expression reference video.",
    )

    def panel(self, name: str) -> str | None:
        """A panel path. The whole sheet stands in for the front panel when none was cut."""
        return self.panels.get(name) or (self.sheet if name == "front" else None)

    def identity_images(self) -> list[str]:
        """Panels whose faces are reliable identity anchors (profile faces are not)."""
        paths = [self.panels[n] for n in ("face", "front", "three_quarter") if n in self.panels]
        return paths or ([self.sheet] if self.sheet else [])


class Character(StrictModel):
    name: str = Field(default="", description="Display name or role of the character.")
    appearance: str = Field(
        default="", description="Permanent look: face, hair, makeup, build, marks. No clothing."
    )
    invariants: str = Field(
        default="",
        description=(
            "Details written into every still of this character and kept across wardrobe changes "
            "(accessories, hair). Nothing a costume change would alter."
        ),
    )
    parent_character: str | None = Field(
        default=None,
        description="Optional character id whose locked base sheet this base sheet edits.",
    )
    wardrobe: list[WardrobeState] = Field(
        default_factory=list, description="Costume states in the order they appear in the song."
    )
    references: dict[str, ReferenceSet] = Field(
        default_factory=dict,
        description="Locked reference sheets, keyed by wardrobe state id ('base' if none).",
    )

    @field_validator("parent_character")
    @classmethod
    def _check_parent_character(cls, value: str | None) -> str | None:
        return validate_safe_id(value, "parent character id") if value is not None else None

    @field_validator("references")
    @classmethod
    def _check_reference_keys(cls, value: dict[str, ReferenceSet]) -> dict[str, ReferenceSet]:
        for key in value:
            validate_safe_id(key, "reference key")
        return value

    @property
    def wardrobe_ids(self) -> list[str]:
        return [state.id for state in self.wardrobe]

    def state(self, state_id: str | None) -> WardrobeState | None:
        return next((w for w in self.wardrobe if w.id == state_id), None)

    def describe(self, state_id: str | None) -> str:
        """Appearance plus the costume for ``state_id``, as sent in generation prompts."""
        state = self.state(state_id)
        if state is None:
            return self.appearance
        return f"{self.appearance} Wearing: {state.description}".strip()

    def pinned(self) -> str:
        """``invariants`` trimmed of surrounding space and trailing sentence punctuation."""
        return self.invariants.strip().rstrip(".;:, ").strip()

    def portrait_text(self) -> str:
        """Description for the base reference sheet: appearance in the first costume."""
        return self.describe(self.wardrobe[0].id if self.wardrobe else None)

    # ---- reference sheets

    def reference_key(self, state_id: str | None) -> str:
        """The key a sheet for ``state_id`` is stored under."""
        if state_id:
            return state_id
        return self.wardrobe[0].id if self.wardrobe else BASE_STATE

    @property
    def default_reference(self) -> ReferenceSet | None:
        """The base sheet: the first costume's, or the single one for a one-costume character."""
        if self.wardrobe and self.wardrobe[0].id in self.references:
            return self.references[self.wardrobe[0].id]
        return self.references.get(BASE_STATE)

    def reference_for(self, state_id: str | None) -> ReferenceSet | None:
        """The sheet for a wardrobe state, falling back to the base sheet."""
        if state_id and state_id in self.references:
            return self.references[state_id]
        return self.default_reference

    def states_without_reference(self, used: Collection[str | None]) -> list[str]:
        """Wardrobe states in ``used`` that have no sheet of their own."""
        missing = []
        for index, state in enumerate(self.wardrobe):
            if state.id not in used or state.id in self.references:
                continue
            if index == 0 and self.default_reference is not None:
                continue  # a base sheet (including a migrated portrait) covers the first costume
            missing.append(state.id)
        return missing


class SetDesign(StrictModel):
    name: str
    sections: list[str] = Field(
        default_factory=list, description="Parts of the song this set is used for."
    )
    description: str = Field(default="", description="What the set looks like.")
    movement: str = Field(default="", description="How the singer moves and the camera works here.")
    parent_set: str | None = Field(
        default=None,
        description="Optional set name whose locked plate this set's plate is edited from.",
    )
    plate: str | None = Field(
        default=None, description="Locked image of the empty location, used as a second reference."
    )


class PropDesign(StrictModel):
    id: str
    name: str
    description: str = Field(
        default="",
        description="Physical design, materials, scale in cm, and negative exclusions.",
    )
    sections: list[str] = Field(
        default_factory=list, description="Parts of the song where this prop appears."
    )
    plate: str | None = Field(
        default=None, description="Locked two-angle reference image of the prop."
    )

    @field_validator("id")
    @classmethod
    def _check_id(cls, value: str) -> str:
        return validate_safe_id(value, "prop id")


class VisualBible(StrictModel):
    concept: str = ""
    style: str = ""
    palette: str = ""
    lighting: str = ""
    negative_prompt: str = ""
    settings: list[SetDesign] = Field(default_factory=list)
    props: list[PropDesign] = Field(default_factory=list)
    rules: list[str] = Field(
        default_factory=list, description="Rules every shot follows, shown to the director model."
    )

    def find_set(self, name: str) -> SetDesign | None:
        """Look a set up by name, ignoring case and surrounding space."""
        wanted = name.strip().lower()
        return next((s for s in self.settings if s.name.strip().lower() == wanted), None)

    def find_prop(self, query: str) -> PropDesign | None:
        """Look a prop up by id or name, ignoring case and surrounding space."""
        wanted = query.strip().lower()
        return next(
            (
                p
                for p in self.props
                if p.id.strip().lower() == wanted or p.name.strip().lower() == wanted
            ),
            None,
        )

    def resolve_props(self, queries: Collection[str]) -> list[PropDesign]:
        """Resolve prop ids or names to unique bible props, preserving order."""
        resolved: list[PropDesign] = []
        seen: set[str] = set()
        for query in queries:
            prop = self.find_prop(query)
            if prop is not None and prop.id not in seen:
                seen.add(prop.id)
                resolved.append(prop)
        return resolved


class Manifest(StrictModel):
    schema_version: int = MANIFEST_SCHEMA_VERSION
    title: str
    duration: float
    fps: int
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)
    character: Character = Field(default_factory=Character)
    characters: dict[str, Character] = Field(
        default_factory=dict,
        description="Named characters/performers (for duets, bands, or ensembles), keyed by id.",
    )
    visual_bible: VisualBible = Field(default_factory=VisualBible)
    shots: list[Shot] = Field(default_factory=list)
    escalated_types: list[ShotType] = Field(
        default_factory=list,
        description="Shot types a reviewer has rejected; all of them get spot-checked.",
    )
    total_generations: int = 0

    @field_validator("schema_version")
    @classmethod
    def _check_schema_version(cls, version: int) -> int:
        if version != MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported manifest schema_version {version}; "
                f"this build reads version {MANIFEST_SCHEMA_VERSION}."
            )
        return version

    @field_validator("characters")
    @classmethod
    def _check_character_keys(cls, value: dict[str, Character]) -> dict[str, Character]:
        for key in value:
            validate_safe_id(key, "character id")
        return value

    @model_validator(mode="after")
    def _sync_primary_character(self) -> Manifest:
        if self.characters:
            # Keep `character` an in-memory alias to the canonical primary entry in `characters`
            # so legacy callers always see updates without maintaining two diverging copies.
            self.character = next(iter(self.characters.values()))
        return self

    @model_serializer(mode="wrap")
    def _serialize(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        data: dict[str, Any] = handler(self)
        if self.characters:
            # Persist multi-character casts in `characters` only; `character` is an in-memory alias.
            data.pop("character", None)
        return data

    def primary_character_id(self) -> str | None:
        return next(iter(self.characters)) if self.characters else None

    @staticmethod
    def _norm_slug(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")

    def resolve_character_id(self, char_id: str | None) -> str | None:
        """Resolve a character key or display name to its canonical id in ``characters``."""
        if not self.characters:
            return None
        if char_id is None:
            return self.primary_character_id()
        cleaned = char_id.strip()
        if not cleaned:
            return None
        if cleaned in self.characters:
            return cleaned
        wanted = cleaned.lower()
        for key, char in self.characters.items():
            if key.lower() == wanted or (char.name and char.name.strip().lower() == wanted):
                return key
        wanted_slug = self._norm_slug(cleaned)
        if not wanted_slug:
            return None
        for key, char in self.characters.items():
            if self._norm_slug(key) == wanted_slug or (
                char.name and self._norm_slug(char.name) == wanted_slug
            ):
                return key
        if len(wanted_slug) >= 2:
            token_matches = [
                key
                for key, char in self.characters.items()
                if char.name and wanted_slug in re.findall(r"[a-z0-9]+", char.name.lower())
            ]
            if len(token_matches) == 1:
                return token_matches[0]
        return None

    def character_for(self, char_id: str | None = None) -> Character:
        """Look up a character by id, falling back to the primary character."""
        if self.characters:
            resolved = self.resolve_character_id(char_id)
            if resolved and resolved in self.characters:
                return self.characters[resolved]
            return next(iter(self.characters.values()))
        return self.character

    def shot_character(self, shot: Shot) -> Character:
        return self.character_for(shot.character)

    def shot_wardrobe(self, shot: Shot, char_id: str | None = None) -> str | None:
        if char_id and char_id in shot.character_wardrobes:
            return shot.character_wardrobes[char_id]
        resolved_shot_char = self.resolve_character_id(shot.character)
        if char_id is None or char_id == shot.character or char_id == resolved_shot_char:
            return shot.wardrobe
        char = self.character_for(char_id)
        return char.wardrobe[0].id if char.wardrobe else None

    def visible_characters(self, shot: Shot) -> list[tuple[str | None, Character, str | None]]:
        """Characters visible in ``shot``: ``(char_id, Character, wardrobe_state_id)``."""
        if not self.characters:
            return [(None, self.character, shot.wardrobe)]
        primary_id = self.resolve_character_id(shot.character) or self.primary_character_id()
        ids: list[str] = []
        if primary_id:
            ids.append(primary_id)
        for cid in shot.characters:
            resolved = self.resolve_character_id(cid)
            if resolved and resolved not in ids:
                ids.append(resolved)
        return [(cid, self.characters[cid], self.shot_wardrobe(shot, cid)) for cid in ids]

    def shot(self, shot_id: str) -> Shot:
        for shot in self.shots:
            if shot.id == shot_id:
                return shot
        raise KeyError(f"No shot with id {shot_id!r}")
