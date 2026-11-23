# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Tracklist and cue-sheet ingestion plus YouTube/FFmpeg chapter export for long-form mixes.

Supports structured JSON/YAML (including WeaveShift setlist exports), standard ``.cue`` sheets,
and plain-text timestamp lists (``00:00 Artist - Title``), or falls back to fixed-interval
segmentation when no cue file is supplied.
"""

from __future__ import annotations

import csv
import io
import json
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from mvdirector.errors import ConfigError
from mvdirector.models import AnalyzedChapter, AudioCues, MixChapter

SUPPORTED_TRACKLIST_SUFFIXES = {".json", ".yaml", ".yml", ".cue", ".txt", ".csv"}

_TIMECODE_RE = re.compile(r"^(?:(\d+):)?(\d{1,4}):(\d{2}(?:\.\d+)?)$")
_CUE_INDEX_RE = re.compile(r"^INDEX\s+0?1\s+(\d+):(\d{2}):(\d{2})\s*$", flags=re.IGNORECASE)
_CUE_TRACK_RE = re.compile(r"^TRACK\s+\d+\s+AUDIO\b", flags=re.IGNORECASE)
_CUE_KV_RE = re.compile(r'^([A-Z_]+)\s+(?:"([^"]*)"|(.+?))\s*$', flags=re.IGNORECASE)
_CUE_REM_RE = re.compile(r'^REM\s+([A-Z_]+)\s+(?:"([^"]*)"|(.+?))\s*$', flags=re.IGNORECASE)
_TXT_LINE_RE = re.compile(
    r"^\s*\[?(?P<ts>(?:\d+:)?\d{1,3}:\d{2}(?:\.\d+)?)\]?\s*(?:[-–—|]\s*)?(?P<rest>.+?)\s*$"
)
_TXT_BPM_RE = re.compile(r"[\[(]\s*(\d{2,3}(?:\.\d+)?)\s*bpm\s*[\])]", flags=re.IGNORECASE)
_TXT_TAG_RE = re.compile(r"(?:^|\s)#([A-Za-z0-9][A-Za-z0-9_-]{0,31})\b")

_MIN_AUTO_LEAD_IN_SECONDS = 10.0


def _clean_line(text: str) -> str:
    """Collapse internal whitespace and strip control characters."""
    cleaned = "".join(ch if ch.isprintable() else " " for ch in text)
    return " ".join(cleaned.split())


def _coerce_tags(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        parts = [p.strip().lstrip("#") for p in re.split(r"[,;/]", raw)]
        return [_clean_line(p) for p in parts if _clean_line(p)]
    if isinstance(raw, list | tuple):
        tags: list[str] = []
        for item in raw:
            if not isinstance(item, str):
                raise ValueError(f"Expected string tag, got {type(item).__name__}")
            cleaned = _clean_line(item.lstrip("#"))
            if cleaned and cleaned not in tags:
                tags.append(cleaned)
        return tags
    raise ValueError(f"Expected tag string or list of strings, got {type(raw).__name__}")


def parse_timestamp(value: float | int | str) -> float:
    """Parse seconds (float/int/numeric string) or ``MM:SS(.ms)`` / ``HH:MM:SS(.ms)`` timecodes."""
    if isinstance(value, bool):
        raise ConfigError(f"Invalid timestamp {value!r}: expected seconds or MM:SS / HH:MM:SS.")
    if isinstance(value, int | float):
        sec = float(value)
        if not math.isfinite(sec) or sec < 0.0:
            raise ConfigError(f"Invalid timestamp {value!r}: must be a finite number >= 0.")
        return round(sec, 3)
    if not isinstance(value, str):
        raise ConfigError(f"Invalid timestamp {value!r}: expected number or timecode string.")

    raw = value.strip()
    if not raw:
        raise ConfigError("Timestamp string must not be empty.")

    match = _TIMECODE_RE.match(raw)
    if match:
        hours_s, minutes_s, seconds_s = match.groups()
        minutes = int(minutes_s)
        seconds = float(seconds_s)
        if seconds >= 60.0:
            raise ConfigError(f"Invalid seconds in timestamp {value!r}: must be < 60.")
        if hours_s is not None:
            hours = int(hours_s)
            if minutes >= 60:
                raise ConfigError(f"Invalid minutes in timestamp {value!r}: must be < 60.")
            return round(hours * 3600.0 + minutes * 60.0 + seconds, 3)
        return round(minutes * 60.0 + seconds, 3)

    try:
        numeric = float(raw)
    except ValueError as exc:
        raise ConfigError(
            f"Invalid timestamp {value!r}: expected seconds or MM:SS / HH:MM:SS."
        ) from exc
    if not math.isfinite(numeric) or numeric < 0.0:
        raise ConfigError(f"Invalid timestamp {value!r}: must be a finite number >= 0.")
    return round(numeric, 3)


def _pick_alias(data: dict[str, Any], canonical: str, aliases: tuple[str, ...]) -> None:
    if canonical in data and data[canonical] is not None:
        return
    for alias in aliases:
        if alias in data and data[alias] is not None:
            data[canonical] = data[alias]
            return


class TrackEntryInput(BaseModel):
    """Normalized representation of a single cue/track entry from an external tracklist."""

    model_config = ConfigDict(extra="ignore")

    start: float
    end: float | None = None
    title: str
    artist: str | None = None
    bpm: float | None = Field(default=None, gt=20.0, lt=300.0, allow_inf_nan=False)
    key: str | None = None
    energy: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)
    mood_tags: list[str] = Field(default_factory=list)
    genre_tags: list[str] = Field(default_factory=list)
    theme_prompt: str | None = None
    clip_prompts: list[str] = Field(default_factory=list)
    lyrics: str | None = None
    cues: AudioCues | None = None

    @model_validator(mode="before")
    @classmethod
    def _normalize_aliases(cls, raw: Any) -> Any:
        if not isinstance(raw, dict):
            raise ValueError("Each track entry must be an object/mapping.")
        data = dict(raw)
        _pick_alias(
            data,
            "start",
            ("start_time", "start_sec", "start_seconds", "timestamp", "cue_in", "mix_in", "offset"),
        )
        _pick_alias(data, "end", ("end_time", "end_sec", "end_seconds", "cue_out", "mix_out"))
        _pick_alias(data, "title", ("name", "track_title", "track"))
        _pick_alias(data, "artist", ("performer", "creator"))
        _pick_alias(data, "bpm", ("tempo", "tempo_bpm"))
        _pick_alias(data, "key", ("camelot", "musical_key"))
        _pick_alias(data, "energy", ("intensity", "energy_level"))
        _pick_alias(data, "mood_tags", ("mood", "moods", "vibe", "tags"))
        _pick_alias(data, "genre_tags", ("genre", "genres", "style"))
        _pick_alias(data, "theme_prompt", ("prompt", "visual_prompt", "environment", "theme"))
        _pick_alias(data, "clip_prompts", ("prompts", "shot_prompts"))
        _pick_alias(data, "lyrics", ("lyric_excerpt", "lyrics_excerpt"))
        return data

    @field_validator("start", mode="before")
    @classmethod
    def _parse_start(cls, value: Any) -> float:
        return parse_timestamp(value)

    @field_validator("end", mode="before")
    @classmethod
    def _parse_end(cls, value: Any) -> float | None:
        return None if value is None else parse_timestamp(value)

    @field_validator("title")
    @classmethod
    def _clean_title(cls, value: str) -> str:
        cleaned = _clean_line(value)
        if not cleaned:
            raise ValueError("Track title must not be empty.")
        return cleaned

    @field_validator("artist", "key", "theme_prompt", "lyrics")
    @classmethod
    def _clean_optional_str(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = _clean_line(value)
        return cleaned or None

    @field_validator("mood_tags", "genre_tags", mode="before")
    @classmethod
    def _validate_tags(cls, value: Any) -> list[str]:
        return _coerce_tags(value)

    @field_validator("clip_prompts", mode="before")
    @classmethod
    def _validate_clip_prompts(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, list | tuple):
            raise ValueError("clip_prompts must be a list of strings.")
        cleaned: list[str] = []
        for item in value:
            if not isinstance(item, str):
                raise ValueError("clip_prompts items must be strings.")
            line = _clean_line(item)
            if line:
                cleaned.append(line)
        return cleaned

    @model_validator(mode="after")
    def _check_end_after_start(self) -> TrackEntryInput:
        if self.end is not None and self.end <= self.start:
            raise ValueError(
                f"Track '{self.title}' end ({self.end}) must be greater than start ({self.start})."
            )
        return self


_START_KEYS = (
    "start",
    "start_time",
    "start_sec",
    "start_seconds",
    "timestamp",
    "cue_in",
    "mix_in",
    "offset",
)


def _finite_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    return num if math.isfinite(num) else None


def _stem_from_path_str(raw: Any) -> str:
    """Extract a safe display stem from an untrusted path string without touching disk."""
    if not isinstance(raw, str) or not raw.strip():
        return ""
    stem = Path(raw.replace("\\", "/")).stem.strip()
    return "" if stem in ("", ".", "..") else stem


def _normalize_weaveshift_setlist_report(raw: dict[str, Any]) -> dict[str, Any] | None:
    """Detect and normalize a ``.weaveshift_setlist_report.json`` payload into a TracklistSpec dict.

    Matches the timeline derivation in ``sherwinlu/weaveshift``
    (``weaveshift/chapters.py:chapters_from_setlist_report``) without accessing any file paths
    referenced inside the report.
    """
    raw_tracks = raw.get("tracks")
    if not isinstance(raw_tracks, list) or not raw_tracks:
        return None
    raw_transitions = raw.get("transitions")
    raw_zeros = raw.get("track_source_zero_in_mix_sec")
    transitions: list[Any] = raw_transitions if isinstance(raw_transitions, list) else []
    zeros: list[Any] = raw_zeros if isinstance(raw_zeros, list) else []
    if not (isinstance(raw_transitions, list) or isinstance(raw_zeros, list)):
        return None
    # Regular tracklists already supply per-track start timestamps.
    if any(
        isinstance(item, dict) and any(item.get(k) is not None for k in _START_KEYS)
        for item in raw_tracks
    ):
        return None

    total_duration = _finite_float(raw.get("output_duration_sec"))

    normalized: list[dict[str, Any]] = []
    for idx, item in enumerate(raw_tracks):
        if idx == 0:
            start = 0.0
        else:
            start_opt: float | None = None
            if idx - 1 < len(transitions) and isinstance(transitions[idx - 1], dict):
                tr = transitions[idx - 1]
                next_zero = _finite_float(tr.get("next_track_source_zero_in_mix_sec"))
                b_cue = _finite_float(tr.get("source_b_cue_sec"))
                if next_zero is not None and b_cue is not None:
                    start_opt = next_zero + b_cue
                else:
                    start_opt = _finite_float(tr.get("mix_transition_start_sec"))
            if start_opt is None and idx < len(zeros):
                start_opt = _finite_float(zeros[idx])
            if start_opt is None:
                raise ValueError(
                    f"WeaveShift setlist report is missing timeline start for track #{idx + 1}."
                )
            if start_opt < -1e-3:
                raise ValueError(
                    f"WeaveShift setlist report has negative start ({start_opt}) "
                    f"for track #{idx + 1}."
                )
            start = max(0.0, start_opt)
            if total_duration is not None and total_duration > 0.0:
                start = min(start, total_duration)

        if not isinstance(item, dict):
            raise ValueError(f"Track #{idx + 1} in WeaveShift report must be an object/mapping.")
        fallback_title = _stem_from_path_str(item.get("path"))
        raw_title = item.get("title")
        if raw_title is not None and not isinstance(raw_title, str):
            raise ValueError(f"Track #{idx + 1} title must be a string.")
        title = (raw_title or "").strip() or fallback_title or f"Track {idx + 1:02d}"
        artist_val = item.get("artist")
        if artist_val is not None and not isinstance(artist_val, str):
            raise ValueError(f"Track #{idx + 1} artist must be a string.")
        artist = (artist_val or "").strip() or None
        bpm_raw = _finite_float(item.get("bpm"))
        bpm = round(bpm_raw, 2) if bpm_raw is not None and 20.0 < bpm_raw < 300.0 else None
        key_val = item.get("key")
        if key_val is not None and not isinstance(key_val, str):
            raise ValueError(f"Track #{idx + 1} key must be a string.")
        key = (key_val or "").strip() or None

        entry: dict[str, Any] = {"start": round(start, 3), "title": title}
        if artist:
            entry["artist"] = artist
        if bpm is not None:
            entry["bpm"] = bpm
        if key:
            entry["key"] = key
        normalized.append(entry)

    doc_title = raw.get("title") or (_stem_from_path_str(raw.get("output_file")) or None)
    return {
        "title": doc_title,
        "source": "weaveshift_report",
        "tracks": normalized,
    }


class TracklistSpec(BaseModel):
    """Parsed and validated tracklist document."""

    model_config = ConfigDict(extra="ignore")

    title: str | None = None
    artist: str | None = None
    source: str | None = None
    tracks: list[TrackEntryInput] = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def _normalize_document(cls, raw: Any) -> Any:
        if isinstance(raw, list):
            return {"tracks": raw}
        if not isinstance(raw, dict):
            raise ValueError("Tracklist must be a JSON/YAML object or array of track entries.")
        ws_report = _normalize_weaveshift_setlist_report(raw)
        if ws_report is not None:
            return ws_report
        data = dict(raw)
        _pick_alias(data, "tracks", ("segments", "chapters", "cues", "setlist"))
        _pick_alias(data, "title", ("mix_title", "name"))
        _pick_alias(data, "artist", ("performer", "dj", "channel", "creator"))
        return data

    @field_validator("title", "artist", "source")
    @classmethod
    def _clean_header_str(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = _clean_line(value)
        return cleaned or None


@dataclass(frozen=True)
class ResolvedTrack:
    """Gap-free track segment resolved against the master audio duration."""

    id: str
    index: int
    start: float
    end: float
    title: str
    artist: str | None = None
    bpm: float | None = None
    key: str | None = None
    energy: float | None = None
    mood_tags: list[str] = field(default_factory=list)
    genre_tags: list[str] = field(default_factory=list)
    theme_prompt: str | None = None
    clip_prompts: list[str] = field(default_factory=list)
    lyrics: str | None = None
    cues: AudioCues | None = None


def _parse_cue_sheet(text: str) -> TracklistSpec:
    header_title: str | None = None
    header_artist: str | None = None
    header_genre: str | None = None
    tracks: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if _CUE_TRACK_RE.match(line):
            if current is not None:
                tracks.append(current)
            current = {"mood_tags": [], "genre_tags": []}
            continue

        rem_match = _CUE_REM_RE.match(line)
        if rem_match:
            key = rem_match.group(1).upper()
            val = (
                rem_match.group(2) if rem_match.group(2) is not None else rem_match.group(3)
            ).strip()
            if current is None:
                if key == "GENRE" and val:
                    header_genre = val
            else:
                if key == "BPM" and val:
                    try:
                        current["bpm"] = float(val)
                    except ValueError as exc:
                        raise ConfigError(f"Invalid REM BPM value {val!r} in CUE sheet.") from exc
                elif key == "GENRE" and val:
                    current["genre_tags"].extend(_coerce_tags(val))
                elif key in ("MOOD", "VIBE") and val:
                    current["mood_tags"].extend(_coerce_tags(val))
                elif key == "KEY" and val:
                    current["key"] = val
            continue

        idx_match = _CUE_INDEX_RE.match(line)
        if idx_match and current is not None:
            mm, ss, ff = (int(idx_match.group(i)) for i in (1, 2, 3))
            if ss >= 60 or ff >= 75:
                raise ConfigError(f"Invalid CUE INDEX timestamp {line!r} (expected MM:SS:FF).")
            current["start"] = round(mm * 60.0 + ss + ff / 75.0, 3)
            continue

        kv_match = _CUE_KV_RE.match(line)
        if kv_match:
            key = kv_match.group(1).upper()
            val = (
                kv_match.group(2) if kv_match.group(2) is not None else kv_match.group(3)
            ).strip()
            if key == "TITLE":
                if current is None:
                    header_title = val
                else:
                    current["title"] = val
            elif key == "PERFORMER":
                if current is None:
                    header_artist = val
                else:
                    current["artist"] = val

    if current is not None:
        tracks.append(current)
    if not tracks:
        raise ConfigError("CUE sheet contains no TRACK AUDIO entries.")

    for idx, entry in enumerate(tracks, start=1):
        if "start" not in entry:
            raise ConfigError(f"CUE TRACK {idx:02d} is missing an INDEX 01 timestamp.")
        entry.setdefault("title", f"Track {idx:02d}")
        if not entry["genre_tags"] and header_genre:
            entry["genre_tags"] = _coerce_tags(header_genre)

    try:
        return TracklistSpec.model_validate(
            {"title": header_title, "artist": header_artist, "source": "cue", "tracks": tracks}
        )
    except ValidationError as exc:
        raise ConfigError(f"Invalid CUE sheet:\n{exc}") from exc


def _parse_txt_tracklist(text: str) -> TracklistSpec:
    tracks: list[dict[str, Any]] = []
    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = _TXT_LINE_RE.match(stripped)
        if not match:
            raise ConfigError(
                f"Tracklist line {line_no} must start with a timestamp (e.g. '00:00 Track Name'): "
                f"{raw_line!r}"
            )
        start = parse_timestamp(match.group("ts"))
        rest = match.group("rest")

        bpm: float | None = None
        bpm_match = _TXT_BPM_RE.search(rest)
        if bpm_match:
            bpm = float(bpm_match.group(1))
            rest = (rest[: bpm_match.start()] + " " + rest[bpm_match.end() :]).strip()

        tags = _TXT_TAG_RE.findall(rest)
        if tags:
            rest = _TXT_TAG_RE.sub("", rest).strip()

        artist: str | None = None
        title = _clean_line(rest)
        for sep in (" - ", " — ", " – "):
            if sep in title:
                left, right = title.split(sep, 1)
                if left.strip() and right.strip():
                    artist, title = _clean_line(left), _clean_line(right)
                break
        if not title:
            raise ConfigError(f"Tracklist line {line_no} is missing a track title: {raw_line!r}")

        tracks.append(
            {
                "start": start,
                "title": title,
                "artist": artist,
                "bpm": bpm,
                "mood_tags": tags,
            }
        )

    if not tracks:
        raise ConfigError("Tracklist text file contains no timestamped tracks.")
    try:
        return TracklistSpec.model_validate({"source": "txt", "tracks": tracks})
    except ValidationError as exc:
        raise ConfigError(f"Invalid text tracklist:\n{exc}") from exc


_CSV_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def _unescape_csv_cell(value: str | None) -> str:
    """Strip WeaveShift's leading-apostrophe formula guard while preserving normal apostrophes."""
    if not value:
        return ""
    text = value.strip()
    if len(text) >= 2 and text[0] == "'" and text[1] in _CSV_FORMULA_PREFIXES:
        text = text[1:]
    return text.strip()


def _csv_first(row: dict[str, str], keys: Sequence[str]) -> str:
    for key in keys:
        val = row.get(key)
        if val:
            return val
    return ""


def _parse_csv_cues(row: dict[str, str], track_start: float) -> AudioCues | None:
    found: dict[str, float | None] = {}
    for cue in ("first_lift", "main_peak", "second_peak", "breakdown"):
        final_raw = _csv_first(row, (f"{cue}_final_sec", f"{cue}_final_timestamp"))
        if final_raw:
            found[cue] = parse_timestamp(final_raw)
            continue
        source_raw = _csv_first(row, (f"{cue}_source_sec", f"{cue}_source_timestamp"))
        if source_raw:
            shifted = track_start + parse_timestamp(source_raw)
            if not math.isfinite(shifted):
                raise ConfigError(f"Overflow computing {cue} timestamp from {source_raw!r}.")
            found[cue] = round(shifted, 3)
            continue
        bare_raw = _csv_first(row, (cue,))
        found[cue] = parse_timestamp(bare_raw) if bare_raw else None

    if all(v is None for v in found.values()):
        return None
    conf_raw = row.get("confidence", "").lower()
    confidence: Literal["low", "medium", "high"] = (
        conf_raw if conf_raw in ("low", "medium", "high") else "low"  # type: ignore[assignment]
    )
    return AudioCues(
        first_lift=found["first_lift"],
        main_peak=found["main_peak"],
        second_peak=found["second_peak"],
        breakdown=found["breakdown"],
        confidence=confidence,
    )


def _parse_csv_tracklist(text: str) -> TracklistSpec:
    """Parse ``weaveshift_cues.csv`` or any CSV tracklist with timestamp columns."""
    try:
        reader = csv.DictReader(io.StringIO(text))
        if not reader.fieldnames:
            raise ConfigError("CSV tracklist is empty or missing a header row.")
        raw_rows = list(enumerate(reader, start=2))
    except csv.Error as exc:
        raise ConfigError(f"Could not parse CSV tracklist: {exc}") from exc

    tracks: list[dict[str, Any]] = []
    for row_no, raw_row in raw_rows:
        row = {
            k.strip().lower(): _unescape_csv_cell(v)
            for k, v in raw_row.items()
            if k is not None and k.strip()
        }
        if not any(row.values()):
            continue

        start_raw = _csv_first(
            row,
            (
                "final_start_sec",
                "final_start_timestamp",
                "track_source_zero_in_mix_sec",
                "start_sec",
                "start_timestamp",
                "start",
                "start_time",
                "mix_start_sec",
            ),
        )
        if not start_raw:
            raise ConfigError(
                f"CSV tracklist row {row_no} is missing a start timestamp "
                "(e.g. final_start_sec / start_sec; for weaveshift-cues, re-run with --timeline)."
            )
        start = parse_timestamp(start_raw)

        end_raw = _csv_first(row, ("end_sec", "end_timestamp", "end", "end_time"))
        end = parse_timestamp(end_raw) if end_raw else None

        path_raw = _csv_first(row, ("path", "file", "filepath", "audio_path"))
        fallback_title = _stem_from_path_str(path_raw)
        title = (
            _csv_first(row, ("title", "track", "name", "track_title"))
            or fallback_title
            or f"Track {len(tracks) + 1:02d}"
        )
        artist = _csv_first(row, ("artist", "performer")) or None

        bpm: float | None = None
        for bpm_col in ("bpm_manifest", "bpm", "tempo", "bpm_detected"):
            bpm_raw = row.get(bpm_col, "")
            if not bpm_raw:
                continue
            cand = _finite_float(bpm_raw)
            if cand is not None and 20.0 < cand < 300.0:
                bpm = round(cand, 2)
                break

        key = _csv_first(row, ("key", "camelot_key", "musical_key")) or None
        mood = _csv_first(row, ("visual_mood", "mood", "mood_tags", "vibe"))
        genre = _csv_first(row, ("genre", "genres", "genre_tags"))
        cues = _parse_csv_cues(row, start)

        entry: dict[str, Any] = {"start": start, "title": title}
        if end is not None:
            entry["end"] = end
        if artist:
            entry["artist"] = artist
        if bpm is not None:
            entry["bpm"] = bpm
        if key:
            entry["key"] = key
        if mood:
            entry["mood_tags"] = mood
        if genre:
            entry["genre_tags"] = genre
        if cues is not None:
            entry["cues"] = cues
        tracks.append(entry)

    if not tracks:
        raise ConfigError("CSV tracklist contains no track rows.")
    try:
        return TracklistSpec.model_validate({"source": "csv", "tracks": tracks})
    except ValidationError as exc:
        raise ConfigError(f"Invalid CSV tracklist:\n{exc}") from exc


def load_tracklist(path: Path) -> TracklistSpec:
    """Load and validate a tracklist (``.json``, ``.yaml``, ``.cue``, ``.txt``, or ``.csv``)."""
    if not path.is_file():
        raise ConfigError(f"Tracklist file not found: {path}")
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_TRACKLIST_SUFFIXES:
        supported = ", ".join(sorted(SUPPORTED_TRACKLIST_SUFFIXES))
        raise ConfigError(
            f"Unsupported tracklist format {path.suffix!r} for {path}. Supported: {supported}."
        )
    try:
        text = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError) as exc:
        raise ConfigError(f"Could not read tracklist {path}: {exc}") from exc
    if suffix == ".cue":
        return _parse_cue_sheet(text)
    if suffix == ".txt":
        return _parse_txt_tracklist(text)
    if suffix == ".csv":
        return _parse_csv_tracklist(text)

    try:
        raw = json.loads(text) if suffix == ".json" else yaml.safe_load(text)
    except (json.JSONDecodeError, yaml.YAMLError) as exc:
        raise ConfigError(f"Could not parse tracklist {path}: {exc}") from exc
    try:
        return TracklistSpec.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"Invalid tracklist in {path}:\n{exc}") from exc


def _chapter_id(index: int, total: int) -> str:
    width = max(2, len(str(max(1, total))))
    return f"ch{index + 1:0{width}d}"


def resolve_chapters(
    spec: TracklistSpec | None,
    duration: float,
    *,
    default_segment_seconds: float = 240.0,
    snap_candidates: Sequence[float] | None = None,
) -> list[ResolvedTrack]:
    """Partition ``[0, duration]`` into gap-free chapters from a tracklist or fixed intervals."""
    if not math.isfinite(duration) or duration <= 0.0:
        raise ConfigError(f"Mix audio duration must be positive, got {duration!r}.")

    if spec is None:
        count = max(1, round(duration / default_segment_seconds))
        span = duration / count
        tolerance = min(20.0, span * 0.2)
        min_gap = min(15.0, span * 0.5)
        cuts = [0.0]
        candidates = sorted(c for c in (snap_candidates or ()) if 0.0 < c < duration)
        for k in range(1, count):
            target = duration * k / count
            chosen = target
            if candidates:
                near = [
                    c
                    for c in candidates
                    if abs(c - target) <= tolerance
                    and c - cuts[-1] >= min_gap
                    and duration - c >= min_gap
                ]
                if near:
                    chosen = min(near, key=lambda c: abs(c - target))
            cuts.append(round(chosen, 3))
        cuts.append(round(duration, 3))
        return [
            ResolvedTrack(
                id=_chapter_id(idx, count),
                index=idx,
                start=cuts[idx],
                end=cuts[idx + 1],
                title=f"Part {idx + 1:02d}",
            )
            for idx in range(count)
        ]

    entries = list(spec.tracks)
    for idx, entry in enumerate(entries):
        if entry.start >= duration:
            raise ConfigError(
                f"Track '{entry.title}' starts at {entry.start:.2f}s, at or beyond mix duration "
                f"({duration:.2f}s)."
            )
        if idx > 0 and entry.start <= entries[idx - 1].start:
            raise ConfigError(
                f"Track timestamps must strictly increase: '{entry.title}' ({entry.start:.2f}s) "
                f"does not follow '{entries[idx - 1].title}' ({entries[idx - 1].start:.2f}s)."
            )
        if (
            idx + 1 < len(entries)
            and entry.end is not None
            and entry.end > entries[idx + 1].start + 1e-6
        ):
            raise ConfigError(
                f"Track '{entry.title}' end ({entry.end:.2f}s) overlaps next track "
                f"'{entries[idx + 1].title}' start ({entries[idx + 1].start:.2f}s)."
            )

    raw_items: list[TrackEntryInput] = []
    first_start = entries[0].start
    if first_start >= _MIN_AUTO_LEAD_IN_SECONDS:
        raw_items.append(
            TrackEntryInput(
                start=0.0,
                end=first_start,
                title="Intro",
                artist=spec.artist,
            )
        )
        raw_items.extend(entries)
    else:
        raw_items.append(entries[0].model_copy(update={"start": 0.0}))
        raw_items.extend(entries[1:])

    total = len(raw_items)
    resolved: list[ResolvedTrack] = []
    for idx, item in enumerate(raw_items):
        start = round(0.0 if idx == 0 else item.start, 3)
        end = round(duration if idx == total - 1 else raw_items[idx + 1].start, 3)
        if end <= start:
            raise ConfigError(
                f"Resolved chapter '{item.title}' has non-positive length "
                f"({start:.3f}s..{end:.3f}s)."
            )
        resolved.append(
            ResolvedTrack(
                id=_chapter_id(idx, total),
                index=idx,
                start=start,
                end=end,
                title=item.title,
                artist=item.artist or spec.artist,
                bpm=item.bpm,
                key=item.key,
                energy=item.energy,
                mood_tags=list(item.mood_tags),
                genre_tags=list(item.genre_tags),
                theme_prompt=item.theme_prompt,
                clip_prompts=list(item.clip_prompts),
                lyrics=item.lyrics,
                cues=item.cues,
            )
        )
    return resolved


def format_chapter_label(title: str, artist: str | None = None) -> str:
    clean_title = _clean_line(title) or "Untitled"
    clean_artist = _clean_line(artist) if artist else ""
    if clean_artist and not clean_title.casefold().startswith(clean_artist.casefold()):
        return f"{clean_artist} - {clean_title}"
    return clean_title


def _format_yt_timestamp(seconds: float, *, use_hours: bool) -> str:
    total = max(0, int(round(seconds)))
    if use_hours:
        hours, rem = divmod(total, 3600)
        minutes, secs = divmod(rem, 60)
        return f"{hours}:{minutes:02d}:{secs:02d}"
    minutes, secs = divmod(total, 60)
    return f"{minutes:02d}:{secs:02d}"


def format_youtube_chapters(
    chapters: Sequence[MixChapter | AnalyzedChapter],
    total_duration: float,
) -> tuple[str, list[str]]:
    """Render a YouTube-compatible ``00:00 Track Name`` chapter list and validation warnings."""
    if not chapters:
        return "", ["No chapters available to export."]

    warnings: list[str] = []
    if len(chapters) < 3:
        warnings.append(
            "YouTube requires at least 3 chapters to activate timeline scrubbing "
            f"(got {len(chapters)})."
        )
    short = [ch.id for ch in chapters if (ch.end - ch.start) < 10.0 - 1e-6]
    if short:
        warnings.append(
            f"YouTube requires every chapter to be >= 10s; short chapter(s): {', '.join(short)}."
        )

    use_hours = total_duration >= 3600.0
    lines: list[str] = []
    prev_sec = -1
    for idx, ch in enumerate(chapters):
        sec = 0 if idx == 0 else max(prev_sec + 1, int(round(ch.start)))
        prev_sec = sec
        stamp = _format_yt_timestamp(float(sec), use_hours=use_hours)
        label = format_chapter_label(ch.title, ch.artist)
        lines.append(f"{stamp} {label}")
    return "\n".join(lines) + "\n", warnings


def _escape_ffmetadata(value: str) -> str:
    cleaned = _clean_line(value)
    for char in ("\\", "=", ";", "#"):
        cleaned = cleaned.replace(char, f"\\{char}")
    return cleaned


def format_ffmetadata_chapters(chapters: Sequence[MixChapter], title: str) -> str:
    """Render an FFmpeg ``;FFMETADATA1`` document with millisecond chapter markers."""
    lines = [";FFMETADATA1", f"title={_escape_ffmetadata(title)}"]
    for ch in chapters:
        start_ms = max(0, int(round(ch.start * 1000.0)))
        end_ms = max(start_ms + 1, int(round(ch.end * 1000.0)))
        label = format_chapter_label(ch.title, ch.artist)
        lines.extend(
            [
                "",
                "[CHAPTER]",
                "TIMEBASE=1/1000",
                f"START={start_ms}",
                f"END={end_ms}",
                f"title={_escape_ffmetadata(label)}",
            ]
        )
    return "\n".join(lines) + "\n"
