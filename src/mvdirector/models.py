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
import unicodedata
from collections.abc import Collection, Sequence
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


def ascii_fold(value: str) -> str:
    """Fold Latin diacritics to ASCII (e.g. 'José' -> 'Jose') via NFKD decomposition."""
    return "".join(
        ch
        for ch in unicodedata.normalize("NFKD", value)
        if not unicodedata.category(ch).startswith("M")
    )


def validate_safe_id(value: str, label: str = "id") -> str:
    """Reject identifiers that could traverse paths or inject delimiters."""
    if not SAFE_ID.match(value):
        raise ValueError(
            f"Invalid {label} {value!r}: must contain only letters, digits, '_' or '-'."
        )
    return value


def normalize_exclusions(raw: Any) -> list[str]:
    """Normalize a string or sequence of exclusion items into deduplicated phrases."""
    if raw is None:
        return []
    if isinstance(raw, str):
        items: Sequence[str] = re.split(r"[,;\n]+", raw)
    elif isinstance(raw, (list, tuple, set, frozenset)):
        items = list(raw)
    else:
        raise ValueError("exclusions must be a string or list of strings.")
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, str):
            raise ValueError("exclusions entries must be strings.")
        norm = " ".join(item.split())
        if not norm:
            continue
        key = norm.casefold()
        if key not in seen:
            seen.add(key)
            cleaned.append(norm)
    return cleaned


def accept_do_not_show_alias(data: Any) -> Any:
    """Fold ``do_not_show`` alias into ``exclusions`` before strict extra='forbid' validation."""
    if not isinstance(data, dict) or "do_not_show" not in data:
        return data
    updated = dict(data)
    alias_val = updated.pop("do_not_show")
    base_val = updated.get("exclusions")
    updated["exclusions"] = normalize_exclusions(
        [*normalize_exclusions(base_val), *normalize_exclusions(alias_val)]
    )
    return updated


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class StrictModel(BaseModel):
    """Base for persisted models: unknown fields are errors, not silently dropped.

    ``allow_inf_nan=False`` rejects ``NaN`` / ``Infinity`` in *every* float field. A hand-edited
    (or LLM-written) ``manifest.json`` carrying ``1e400`` used to load fine and then blow up deep
    in assembly with a bare ``OverflowError: cannot convert float infinity to integer``; a
    non-finite number now fails validation at the edge, with the field named.
    """

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class ProjectMode(str, Enum):
    STANDARD = "standard"
    MIX = "mix"


class VideoStyle(str, Enum):
    """Direction style for standard-mode music videos."""

    LIPSYNC = "lipsync"
    NARRATIVE = "narrative"
    VISUAL = "visual"


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


BOTH_VOICES = "both"


class VocalTurn(TimeSpan):
    """A contiguous vocal phrase attributed to one acoustic voice cluster or simultaneous duet."""

    voice: str = Field(
        description=(
            "Acoustic voice id ('voice_1', 'voice_2', ordered low-to-high pitch, or 'both' "
            "when multiple voices sing simultaneously)."
        )
    )
    median_f0_hz: float | None = Field(
        default=None, description="Median fundamental frequency in Hz across voiced frames."
    )

    @field_validator("voice")
    @classmethod
    def _check_voice(cls, value: str) -> str:
        return validate_safe_id(value, "voice id")


class LyricLine(StrictModel):
    text: str
    section: str | None = None
    start: float | None = None
    end: float | None = None
    # "aligned" = from word timestamps, "estimated" = spread over vocal activity.
    timing_source: Literal["aligned", "estimated", "none"] = "none"


class AudioCues(StrictModel):
    """Structural energy landmarks detected from audio (no lyrics required).

    Credits & Attribution:
        Composite energy/onset/brightness landmark detection ported from
        ``sherwinlu/weaveshift`` (``weaveshift/cues.py``).
    """

    first_lift: float | None = Field(
        default=None,
        ge=0.0,
        allow_inf_nan=False,
        description="First sustained energy lift above intro baseline, in seconds.",
    )
    main_peak: float | None = Field(
        default=None,
        ge=0.0,
        allow_inf_nan=False,
        description="Primary energy/onset/brightness peak (drop or main chorus), in seconds.",
    )
    second_peak: float | None = Field(
        default=None,
        ge=0.0,
        allow_inf_nan=False,
        description="Secondary structural peak separated from main_peak, in seconds.",
    )
    breakdown: float | None = Field(
        default=None,
        ge=0.0,
        allow_inf_nan=False,
        description="Lowest-energy mid-track valley/breakdown candidate, in seconds.",
    )
    confidence: Literal["low", "medium", "high"] = Field(
        default="low",
        description="Peak prominence confidence ('low', 'medium', or 'high').",
    )


class AnalyzedChapter(StrictModel):
    """Per-track or fixed-interval segment measured during long-form mix audio analysis."""

    id: str
    index: int = Field(ge=0)
    start: float = Field(ge=0.0, allow_inf_nan=False)
    end: float = Field(gt=0.0, allow_inf_nan=False)
    title: str
    artist: str | None = None
    bpm: float = Field(gt=0.0, allow_inf_nan=False)
    key: str | None = None
    downbeat_offset: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)
    energy: float = Field(default=0.5, ge=0.0, le=1.0, allow_inf_nan=False)
    mood_tags: list[str] = Field(default_factory=list)
    genre_tags: list[str] = Field(default_factory=list)
    theme_prompt: str | None = None
    clip_prompts: list[str] = Field(default_factory=list)
    lyrics: str | None = None
    cues: AudioCues | None = Field(
        default=None,
        description="Structural energy landmarks (lift, peaks, breakdown) within this chapter.",
    )

    @field_validator("id")
    @classmethod
    def _check_id(cls, value: str) -> str:
        return validate_safe_id(value, "chapter id")

    @model_validator(mode="after")
    def _check_span(self) -> AnalyzedChapter:
        if self.end <= self.start:
            raise ValueError(
                f"Chapter '{self.id}' end ({self.end}) must exceed start ({self.start})"
            )
        return self

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


VocalMode = Literal["solo", "duet", "group", "instrumental"]
_ISO_LANG_RE = re.compile(r"^[a-z]{2,3}$")


class TrackProfile(StrictModel):
    """Detected musical, vocal, and linguistic characteristics used to auto-tune parameters."""

    languages: list[str] = Field(
        default_factory=list,
        description="Detected ISO-639-1/639-2 language codes (e.g. ['en'], ['fr', 'en'], ['zh']).",
    )
    multilingual: bool = Field(
        default=False,
        description="True when lyrics or style cues span more than one language.",
    )
    vocal_mode: VocalMode = Field(
        default="solo",
        description="Primary vocal arrangement ('solo', 'duet', 'group', or 'instrumental').",
    )
    rap_or_fast_vocal: bool = Field(
        default=False,
        description=(
            "True when style cues or measured syllable rate indicate rap or rapid phrasing."
        ),
    )
    drop_or_chop_heavy: bool = Field(
        default=False,
        description=(
            "True when style/section cues or peak non-lyric bars indicate drops/vocal chops."
        ),
    )
    extreme_vocals: bool = Field(
        default=False,
        description="True when cues indicate screamed, whispered, or heavily processed vocals.",
    )
    non_4_4_or_free_time: bool = Field(
        default=False,
        description="True when cues indicate 3/4, 6/8, odd meter, or rubato/beatless timing.",
    )
    beats_per_bar_hint: int | None = Field(
        default=None,
        ge=2,
        le=12,
        description="Meter grouping inferred from style cues when explicitly non-4/4 (e.g. 3).",
    )
    syllables_per_second: float | None = Field(
        default=None,
        ge=0.0,
        allow_inf_nan=False,
        description="Measured sung syllable/token rate across aligned or active vocal spans.",
    )
    intensity_contrast: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
        description="90th-minus-20th percentile of bar_intensity (0-1).",
    )
    reasons: list[str] = Field(
        default_factory=list,
        description="Human-readable explanations for detected track characteristics.",
    )

    @field_validator("languages")
    @classmethod
    def _check_languages(cls, values: list[str]) -> list[str]:
        cleaned: list[str] = []
        for raw in values:
            code = raw.strip().lower()
            if not _ISO_LANG_RE.match(code):
                raise ValueError(f"Invalid ISO language code in track_profile: {raw!r}")
            if code not in cleaned:
                cleaned.append(code)
        return cleaned


class MusicalFeatures(StrictModel):
    """Acoustic and musical characteristics measured directly from the audio mix."""

    key: str | None = Field(
        default=None,
        description=(
            "Estimated musical key and mode (e.g. 'A minor', 'C major'), or null if atonal."
        ),
    )
    mode: Literal["major", "minor"] | None = Field(
        default=None,
        description="Estimated harmonic mode ('major' or 'minor'), or null if unclear.",
    )
    camelot: str | None = Field(
        default=None,
        description="Camelot wheel key notation (e.g. '8A', '8B'), or null if unclear.",
    )
    brightness: Literal["dark", "warm", "balanced", "bright"] = Field(
        default="balanced",
        description="Timbral brightness inferred from median spectral centroid.",
    )
    spectral_centroid_hz: float = Field(
        default=0.0,
        ge=0.0,
        allow_inf_nan=False,
        description="Median spectral centroid in Hz across active frames.",
    )
    percussiveness: Literal["ambient", "melodic", "balanced", "rhythmic", "percussive"] = Field(
        default="balanced",
        description="Harmonic-vs-percussive texture balance.",
    )
    percussive_ratio: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
        description="Share (0-1) of signal energy in the percussive component.",
    )
    dynamic_range: Literal["flat", "moderate", "dynamic", "cinematic"] = Field(
        default="moderate",
        description="Loudness contrast across the track.",
    )
    dynamic_range_db: float = Field(
        default=0.0,
        ge=0.0,
        allow_inf_nan=False,
        description="95th-minus-15th percentile RMS loudness spread in dB across active frames.",
    )
    onset_density: float = Field(
        default=0.0,
        ge=0.0,
        allow_inf_nan=False,
        description="Average musical onsets per second.",
    )
    summary: str = Field(
        default="",
        description="Compact human-readable musical character summary for planning prompts.",
    )


StemRole = Literal[
    "lead_vocals",
    "backing_vocals",
    "vocals",
    "drums",
    "percussion",
    "bass",
    "guitar",
    "keys",
    "strings",
    "synth",
    "brass",
    "woodwinds",
    "instrumental",
    "fx",
    "other",
]


class StemTrackInfo(StrictModel):
    """Measured metadata and activity spans for one ingested instrument or vocal stem."""

    id: str
    name: str = Field(description="Human-readable stem label (e.g. 'Lead Vocals', 'Woodwinds').")
    role: StemRole = Field(description="Canonical musical role inferred from the stem filename.")
    path: str = Field(description="Project-relative path to the normalized stem WAV file.")
    active_fraction: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
        description="Share (0-1) of the song duration where this stem is audibly active.",
    )
    energy_share: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
        description="Relative RMS energy share (0-1) compared to all analyzed stems.",
    )
    active_spans: list[TimeSpan] = Field(
        default_factory=list,
        description="Time spans in seconds where this stem is audibly active.",
    )

    @field_validator("id")
    @classmethod
    def _check_id(cls, value: str) -> str:
        return validate_safe_id(value, "stem id")


class StemArrangement(StrictModel):
    """Multi-stem instrumentation and arrangement analysis derived from extracted song stems."""

    stems: list[StemTrackInfo] = Field(
        default_factory=list,
        description="Catalog of analyzed stems with roles and active time spans.",
    )
    backing_vocal_windows: list[TimeSpan] = Field(
        default_factory=list,
        description="Time spans where backing vocals or harmonies are active.",
    )
    rhythm_drops: list[float] = Field(
        default_factory=list,
        description="Bar timestamps (seconds) where drums/bass drop out or slam back in.",
    )
    summary: str = Field(
        default="",
        description="Compact human-readable instrumentation and arrangement summary for planning.",
    )


class AlignmentWarning(StrictModel):
    """Pre-spend alignment anomaly surfaced during audio/lyric verification.

    Inspired by ``JavaDerek/music-video-pipeline`` (``alignment_quality.py``, ``join_witness.py``).
    """

    kind: Literal[
        "unvoiced_lyric",
        "high_consonant_bleed",
        "unaligned_vocal_window",
        "boundary_word_cut",
    ]
    start: float = Field(ge=0.0, allow_inf_nan=False)
    end: float = Field(ge=0.0, allow_inf_nan=False)
    detail: str = ""


class BroadcastAudioQA(StrictModel):
    """Broadcast loudness (ITU-R BS.1770-4 LUFS), 4x True Peak, and splice-click diagnostics."""

    integrated_lufs: float = Field(allow_inf_nan=False)
    true_peak_dbtp: float = Field(allow_inf_nan=False)
    splice_clicks: list[float] = Field(default_factory=list)


class Analysis(StrictModel):
    schema_version: int = ANALYSIS_SCHEMA_VERSION
    duration: float
    tempo_bpm: float
    beats: list[float]
    bars: list[float] = Field(description="Start time of each bar, in seconds.")
    bar_intensity: list[float] = Field(description="Loudness of each bar, scaled 0-1.")
    vocal_windows: list[TimeSpan]
    vocal_turns: list[VocalTurn] = Field(
        default_factory=list,
        description="Diarized vocal turns grouped by acoustic voice pitch and timbre.",
    )
    lyric_lines: list[LyricLine] = Field(default_factory=list)
    alignment_warnings: list[AlignmentWarning] = Field(
        default_factory=list,
        description="Pre-spend acoustic/boundary lyric alignment diagnostics.",
    )
    broadcast_qa: BroadcastAudioQA | None = Field(
        default=None,
        description="Broadcast loudness, 4x true-peak headroom, and splice-click diagnostics.",
    )
    track_profile: TrackProfile | None = Field(
        default=None,
        description="Detected musical/vocal/language profile used for automatic parameter tuning.",
    )
    musical_features: MusicalFeatures | None = Field(
        default=None,
        description="Acoustic musical qualities (key, mode, brightness, percussiveness, dynamics).",
    )
    cues: AudioCues | None = Field(
        default=None,
        description="Structural energy landmarks (first lift, peaks, breakdown) in seconds.",
    )
    stems: StemArrangement | None = Field(
        default=None,
        description="Optional multi-stem instrumentation and arrangement analysis.",
    )
    chapters: list[AnalyzedChapter] = Field(
        default_factory=list,
        description="Per-track segments measured in long-form mix mode.",
    )

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


class ContinuityMode(str, Enum):
    """How a shot connects visually to the preceding shot.

    Inspired by ``afj176/open-cinema-studio`` (``cut`` / ``reframe`` / ``continue``).
    """

    CUT = "cut"  # independent hard cut (default)
    REFRAME = "reframe"  # edit previous shot's end frame into a new camera angle
    CONTINUE = "continue"  # one-take continuation starting from previous shot's post-trim frame


class RetryStrategy(str, Enum):
    INITIAL = "initial"
    HANDOFF = "handoff"
    NEW_SEED = "new_seed"
    REWRITE_PROMPT = "rewrite_prompt"
    REPAIR_STILL = "repair_still"
    ADJUST_PARAMS = "adjust_params"


class Attempt(StrictModel):
    number: int
    strategy: RetryStrategy
    seed: int
    prompt: str
    params: dict[str, Any] = Field(default_factory=dict)
    path: str | None = None
    qc: QCReport | None = None
    sync_delay_seconds: float = Field(
        default=0.0,
        allow_inf_nan=False,
        description=(
            "Measured audio-video lip-sync offset applied to clip in-point during assembly "
            "(inspired by VisualFrisson/ComfyUI-OneShotMV)."
        ),
    )
    optimal_in_offset: float = Field(
        default=0.0,
        allow_inf_nan=False,
        description=(
            "Optional in-point offset in seconds selected for non-lipsync clips within available "
            "handles (ignored on lip-sync shots)."
        ),
    )
    regressed: bool = Field(
        default=False,
        description=(
            "True when an in-place repair did not strictly improve on the previous attempt and "
            "should be excluded from best_attempt selection."
        ),
    )
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
    pending_strategy: RetryStrategy | None = Field(
        default=None,
        description=(
            "Explicit strategy requested for the next pending attempt (repair or recreate)."
        ),
    )
    repair_instruction: str | None = Field(
        default=None,
        description="Explicit user edit instruction for an in-place still repair.",
    )
    repair_source_attempt: int | None = Field(
        default=None,
        description="Attempt number whose rendered image serves as the source for in-place repair.",
    )

    @property
    def selected(self) -> Attempt | None:
        for attempt in self.attempts:
            if attempt.number == self.selected_attempt:
                return attempt
        return None

    def active_attempt(self) -> Attempt | None:
        """Current usable attempt: selected attempt, else best_attempt(), else latest with path."""
        if self.selected is not None and self.selected.path:
            return self.selected
        best = self.best_attempt()
        if best is not None:
            return best
        for attempt in reversed(self.attempts):
            if attempt.path:
                return attempt
        return None

    def clear_pending_overrides(self) -> None:
        """Clear one-shot explicit repair/recreate intent fields after consumption."""
        self.pending_strategy = None
        self.repair_instruction = None
        self.repair_source_attempt = None

    def pass_attempts(self, *, after_fallback: bool = False) -> list[Attempt]:
        """Attempts belonging to the active pass (after a fallback conversion, if any)."""
        if not after_fallback:
            return self.attempts
        starts = [
            i
            for i, a in enumerate(self.attempts)
            if a.strategy in (RetryStrategy.INITIAL, RetryStrategy.HANDOFF)
        ]
        return self.attempts[starts[-1] :] if len(starts) >= 2 else []

    def best_attempt(self) -> Attempt | None:
        """Highest-scoring non-regressed attempt with a file; hard failures rank last."""
        starts = [
            i
            for i, a in enumerate(self.attempts)
            if a.strategy in (RetryStrategy.INITIAL, RetryStrategy.HANDOFF)
        ]
        current_pass = self.attempts[starts[-1] :] if starts else self.attempts
        candidates = [a for a in current_pass if a.path and a.qc]
        usable = [a for a in candidates if not a.regressed] or candidates
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
        self.clear_pending_overrides()


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
    # `start` may be negative: the first shot's slice can reach for a head handle before t=0.
    start: float
    end: float
    head_handle: float = Field(ge=0.0, description="Seconds between slice start and shot start.")
    mute_spans: list[TimeSpan] = Field(
        default_factory=list,
        description=(
            "Song-timeline spans silenced inside this slice so another singer's voice does not "
            "bleed into lip-sync."
        ),
    )

    @model_validator(mode="after")
    def _check_span(self) -> AudioSlice:
        if self.end < self.start:
            raise ValueError(
                f"Audio slice end ({self.end}) must not be before start ({self.start})."
            )
        return self


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
    chapter_id: str | None = Field(
        default=None, description="Parent chapter id when planned in long-form mix mode."
    )
    setting: str = Field(default="", description="Name of a set in the visual bible.")
    character: str | None = Field(
        default=None, description="Primary character id on camera / singing in this shot."
    )
    characters: list[str] = Field(
        default_factory=list,
        description="All character ids visible in this shot (for duets, two-shots, or bands).",
    )
    singers: list[str] = Field(
        default_factory=list,
        description=(
            "Character ids actively singing in this shot (1 for solo A or B, 2+ when singing "
            "together in unison/harmony; empty when not singing)."
        ),
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
    previz_depth_video: str | None = Field(
        default=None,
        description=(
            "Optional linear view-space depth pass rendered alongside previz_video "
            "(white=near, black=far; inspired by heroesjouney/aimoviestudiov2)."
        ),
    )
    previz_mapping: str = Field(
        default="",
        description="Stand-in mapping from blocking shapes in previz_video to real subjects/props.",
    )
    sketch_image: str | None = Field(
        default=None,
        description=(
            "Optional project-relative rough sketch or framing guide image "
            "(lightweight alternative to 3D blocking previz) locking still framing and blocking."
        ),
    )
    sketch_guidance: str = Field(
        default="",
        description=(
            "Optional spatial framing/blocking summary extracted from sketch_image for prompt "
            "conditioning when the workflow has no sketch_image binding."
        ),
    )
    plate_angle: PlateAngle | None = Field(
        default=None,
        description=(
            "Optional set plate turnaround angle ('wide', 'reverse', 'side', 'detail') "
            "overriding automatic framing/prompt selection."
        ),
    )
    draft_grid_sheet: str | None = Field(
        default=None,
        description=(
            "Optional project-relative section storyboard contact sheet containing this shot."
        ),
    )
    continuity: ContinuityMode = Field(
        default=ContinuityMode.CUT,
        description="Shot continuity link to the preceding shot ('cut', 'reframe', or 'continue').",
    )
    end_frame_path: str | None = Field(
        default=None,
        description="Optional target end-frame image path for first/last-frame (FLF2V) workflows.",
    )
    camera_path_zoom: float | None = Field(
        default=None,
        gt=1.0,
        le=3.0,
        allow_inf_nan=False,
        description=(
            "Optional Lanczos zoom factor for an auto-generated end-frame camera-path guide on "
            "empty wide shots (inspired by afj176/open-cinema-studio)."
        ),
    )
    sfx_path: str | None = Field(
        default=None,
        description=(
            "Optional project-relative diegetic SFX/foley audio file leveled and blended over "
            "the music track with smooth ducking during assembly."
        ),
    )
    sfx_gain_db: float = Field(
        default=0.0,
        ge=-40.0,
        le=24.0,
        allow_inf_nan=False,
        description="Playback gain in dB for shot.sfx_path after active-RMS leveling.",
    )
    sfx_duck_db: float = Field(
        default=-6.0,
        ge=-40.0,
        le=0.0,
        allow_inf_nan=False,
        description="Music ducking attenuation in dB while shot.sfx_path plays.",
    )
    audio_slice: AudioSlice
    num_frames: int = Field(ge=0)
    still: AssetState = Field(default_factory=AssetState)
    video: AssetState = Field(default_factory=AssetState)
    review: Review = Field(default_factory=Review)

    @field_validator("id")
    @classmethod
    def _check_id(cls, value: str) -> str:
        return validate_safe_id(value, "shot id")

    @model_validator(mode="after")
    def _check_timeline(self) -> Shot:
        """Reject a backwards shot span.

        ``frame_counts`` turns ``end < start`` into a negative frame count, assembly silently
        drops the shot, and every later shot then plays early — the finished video drifts out of
        sync with the music with no error anywhere. Catch it where it is still fixable.
        """
        if self.start < 0.0:
            raise ValueError(f"Shot {self.id!r} start ({self.start}) must not be negative.")
        if self.end < self.start:
            raise ValueError(
                f"Shot {self.id!r} end ({self.end}) must not be before start ({self.start})."
            )
        return self

    @field_validator("chapter_id")
    @classmethod
    def _check_chapter_id(cls, value: str | None) -> str | None:
        return validate_safe_id(value, "chapter id") if value is not None else None

    @field_validator("character")
    @classmethod
    def _check_character(cls, value: str | None) -> str | None:
        return validate_safe_id(value, "character id") if value is not None else None

    @field_validator("characters", "singers")
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
PLATE_ANGLE_NAMES = ("wide", "reverse", "side", "detail")
PlateAngle = Literal["wide", "reverse", "side", "detail"]
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
    sheet: str | None = Field(
        default=None,
        description="Optional locked 2x2 multi-angle turnaround sheet of the empty location.",
    )
    panels: dict[str, str] = Field(
        default_factory=dict,
        description="Cropped set plate panels keyed by angle (wide, reverse, side, detail).",
    )

    def panel(self, angle: str | None = None) -> str | None:
        """Return the requested angle panel, falling back to primary plate or 'wide'."""
        if angle and angle in self.panels:
            return self.panels[angle]
        if self.plate:
            return self.plate
        if "wide" in self.panels:
            return self.panels["wide"]
        return next(iter(self.panels.values()), None)


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


class SectionBeat(StrictModel):
    """Emotional and narrative direction for one song section."""

    section: str = Field(description="Song section name matching the lyrics tags (e.g. 'Verse 1').")
    emotion: str = Field(
        default="",
        description="Emotional state or tension in this section.",
    )
    visual_beat: str = Field(
        default="",
        description="What happens visually or narratively here to express the lyrics and music.",
    )
    camera_energy: str = Field(
        default="",
        description=(
            "How camera language, pacing, and performance energy should feel in this section."
        ),
    )


class SongInterpretation(StrictModel):
    """Director's reading of the song's meaning, musical style, and emotional arc."""

    meaning: str = Field(
        default="",
        description="What the song is really about: core theme, subtext, and emotional thesis.",
    )
    tone: str = Field(
        default="",
        description="Emotional register and perspective (e.g. bittersweet, defiant, haunted).",
    )
    music_feel: str = Field(
        default="",
        description=(
            "How the musical style, rhythm, instrumentation, and vocal delivery shape visuals."
        ),
    )
    visual_motifs: list[str] = Field(
        default_factory=list,
        description="2 to 5 recurring visual metaphors or motifs grounded in the song's meaning.",
    )
    section_beats: list[SectionBeat] = Field(
        default_factory=list,
        description="Section-by-section emotional and narrative arc across the song.",
    )


class WorldFact(StrictModel):
    """A mutable or irreversible physical world-state fact tracked across shots.

    Inspired by ``afj176/open-cinema-studio`` and ``JavaDerek/music-video-pipeline``.
    """

    entity: str = Field(min_length=1)
    attribute: str = Field(min_length=1)
    value: str = Field(min_length=1)
    start_shot: int = Field(default=0, ge=0)
    irreversible: bool = False


class VisualBible(StrictModel):
    interpretation: SongInterpretation | None = Field(
        default=None,
        description=(
            "Director's interpretation of the song's meaning, musical feel, and section arc."
        ),
    )
    concept: str = ""
    style: str = ""
    palette: str = ""
    lighting: str = ""
    negative_prompt: str = ""
    settings: list[SetDesign] = Field(default_factory=list)
    props: list[PropDesign] = Field(default_factory=list)
    world_facts: list[WorldFact] = Field(
        default_factory=list,
        description="Tracked physical world/prop/lighting states across shot indices.",
    )
    rules: list[str] = Field(
        default_factory=list, description="Rules every shot follows, shown to the director model."
    )
    exclusions: list[str] = Field(
        default_factory=list,
        description=(
            "Global story exclusions ('never show' list) synced into rules and negative_prompt."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _accept_do_not_show_alias(cls, data: Any) -> Any:
        return accept_do_not_show_alias(data)

    @field_validator("exclusions", mode="before")
    @classmethod
    def _normalize_exclusions(cls, value: Any) -> list[str]:
        return normalize_exclusions(value)

    def find_section_beat(self, section: str | None) -> SectionBeat | None:
        """Look up the section beat for ``section`` (exact match, then base without singer tag)."""
        if not section or not self.interpretation or not self.interpretation.section_beats:
            return None
        wanted = unicodedata.normalize("NFKC", section).strip().casefold()
        if not wanted:
            return None
        for beat in self.interpretation.section_beats:
            if unicodedata.normalize("NFKC", beat.section).strip().casefold() == wanted:
                return beat
        wanted_base = wanted.split(":", 1)[0].strip()
        for beat in self.interpretation.section_beats:
            norm = unicodedata.normalize("NFKC", beat.section).strip().casefold()
            if norm.split(":", 1)[0].strip() == wanted_base:
                return beat
        return None

    def find_set(self, name: str) -> SetDesign | None:
        """Look a set up by name, ignoring case, Unicode normalization and surrounding space."""
        wanted = unicodedata.normalize("NFKC", name).strip().casefold()
        return next(
            (
                s
                for s in self.settings
                if unicodedata.normalize("NFKC", s.name).strip().casefold() == wanted
            ),
            None,
        )

    def find_prop(self, query: str) -> PropDesign | None:
        """Look a prop up by id or name, ignoring case, Unicode normalization and space."""
        wanted = unicodedata.normalize("NFKC", query).strip().casefold()
        return next(
            (
                p
                for p in self.props
                if unicodedata.normalize("NFKC", p.id).strip().casefold() == wanted
                or unicodedata.normalize("NFKC", p.name).strip().casefold() == wanted
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


class MixChapter(StrictModel):
    """One track segment in a long-form mix, holding its visual theme and loop shot ids."""

    id: str
    index: int = Field(ge=0)
    start: float = Field(ge=0.0, allow_inf_nan=False)
    end: float = Field(gt=0.0, allow_inf_nan=False)
    title: str
    artist: str | None = None
    bpm: float = Field(gt=0.0, allow_inf_nan=False)
    downbeat_offset: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)
    energy: float = Field(default=0.5, ge=0.0, le=1.0, allow_inf_nan=False)
    mood_tags: list[str] = Field(default_factory=list)
    genre_tags: list[str] = Field(default_factory=list)
    environment: str = ""
    palette: str = ""
    lighting: str = ""
    motion_style: str = ""
    shot_ids: list[str] = Field(default_factory=list)

    @field_validator("id")
    @classmethod
    def _check_id(cls, value: str) -> str:
        return validate_safe_id(value, "chapter id")

    @field_validator("shot_ids")
    @classmethod
    def _check_shot_ids(cls, value: list[str]) -> list[str]:
        return [validate_safe_id(item, "shot id") for item in value]

    @model_validator(mode="after")
    def _check_span(self) -> MixChapter:
        if self.end <= self.start:
            raise ValueError(
                f"Chapter '{self.id}' end ({self.end}) must exceed start ({self.start})"
            )
        return self

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


class DraftGridSheet(StrictModel):
    """Metadata for a generated per-section storyboard contact sheet."""

    id: str
    section: str | None = None
    layout: Literal["2x2", "3x2"] = "2x2"
    sheet_path: str
    shot_ids: list[str] = Field(default_factory=list)
    panel_paths: dict[str, str] = Field(default_factory=dict)
    video_path: str | None = None
    created_at: str = Field(default_factory=utc_now)

    @field_validator("id")
    @classmethod
    def _check_id(cls, value: str) -> str:
        return validate_safe_id(value, "draft grid id")

    @field_validator("shot_ids")
    @classmethod
    def _check_shot_ids(cls, value: list[str]) -> list[str]:
        return [validate_safe_id(item, "shot id") for item in value]


class Manifest(StrictModel):
    schema_version: int = MANIFEST_SCHEMA_VERSION
    mode: ProjectMode = ProjectMode.STANDARD
    title: str
    duration: float = Field(ge=0.0)
    # fps divides into frame durations and crossfade lengths; 0 or a negative value yields an
    # empty timeline or a ZeroDivisionError deep inside assembly.
    fps: int = Field(ge=1, le=1000)
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)
    character: Character = Field(default_factory=Character)
    characters: dict[str, Character] = Field(
        default_factory=dict,
        description="Named characters/performers (for duets, bands, or ensembles), keyed by id.",
    )
    visual_bible: VisualBible = Field(default_factory=VisualBible)
    chapters: list[MixChapter] = Field(
        default_factory=list,
        description="Track segments and visual themes when mode is 'mix'.",
    )
    shots: list[Shot] = Field(default_factory=list)
    draft_grids: list[DraftGridSheet] = Field(
        default_factory=list,
        description="Optional per-section storyboard contact sheets generated in draft_grid mode.",
    )
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
        return re.sub(r"[^a-z0-9]+", "_", ascii_fold(value).strip().lower()).strip("_")

    def resolve_character_id(self, char_id: str | None) -> str | None:
        """Resolve a character key or display name to its canonical id in ``characters``."""
        if not self.characters:
            return None
        if char_id is None:
            return self.primary_character_id()
        cleaned = unicodedata.normalize("NFKC", char_id).strip()
        if not cleaned:
            return None
        if cleaned in self.characters:
            return cleaned
        wanted = cleaned.casefold()
        for key, char in self.characters.items():
            char_name = unicodedata.normalize("NFKC", char.name).strip().casefold()
            if key.casefold() == wanted or (char_name and char_name == wanted):
                return key
        wanted_slug = self._norm_slug(cleaned)
        if wanted_slug:
            for key, char in self.characters.items():
                if self._norm_slug(key) == wanted_slug or (
                    char.name and self._norm_slug(char.name) == wanted_slug
                ):
                    return key
            if len(wanted_slug) >= 2:
                token_matches = [
                    key
                    for key, char in self.characters.items()
                    if char.name
                    and wanted_slug in re.findall(r"[a-z0-9]+", ascii_fold(char.name).lower())
                ]
                if len(token_matches) == 1:
                    return token_matches[0]
        if len(wanted) >= 2:
            unicode_matches = [
                key
                for key, char in self.characters.items()
                if char.name
                and wanted in unicodedata.normalize("NFKC", char.name).casefold().split()
            ]
            if len(unicode_matches) == 1:
                return unicode_matches[0]
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

    def singing_characters(self, shot: Shot) -> list[tuple[str | None, Character, str | None]]:
        """Characters actively singing in ``shot`` (1 for solo, 2+ when singing together)."""
        if shot.type is not ShotType.LIPSYNC:
            return []
        if not self.characters:
            return [(None, self.character, shot.wardrobe)]
        ids: list[str] = []
        for cid in shot.singers:
            resolved = self.resolve_character_id(cid)
            if resolved and resolved not in ids:
                ids.append(resolved)
        if not ids:
            primary = self.resolve_character_id(shot.character) or self.primary_character_id()
            if primary:
                ids.append(primary)
        return [(cid, self.characters[cid], self.shot_wardrobe(shot, cid)) for cid in ids]

    def active_singer_ids(self, shot: Shot) -> set[str]:
        """Canonical character ids actively singing in ``shot``."""
        return {cid for cid, _, _ in self.singing_characters(shot) if cid is not None}

    def shot(self, shot_id: str) -> Shot:
        for shot in self.shots:
            if shot.id == shot_id:
                return shot
        raise KeyError(f"No shot with id {shot_id!r}")

    def chapter(self, chapter_id: str) -> MixChapter:
        for ch in self.chapters:
            if ch.id == chapter_id:
                return ch
        raise KeyError(f"No chapter with id {chapter_id!r}")

    def find_chapter(self, chapter_id: str | None) -> MixChapter | None:
        if not chapter_id:
            return None
        return next((ch for ch in self.chapters if ch.id == chapter_id), None)

    def shots_for_chapters(self, chapter_ids: Collection[str]) -> list[str]:
        """Resolve chapter ids to ordered unique shot ids, raising KeyError on unknown ids."""
        selected: list[str] = []
        seen: set[str] = set()
        for cid in chapter_ids:
            chapter = self.chapter(cid)
            for sid in chapter.shot_ids:
                if sid not in seen:
                    seen.add(sid)
                    selected.append(sid)
        return selected
