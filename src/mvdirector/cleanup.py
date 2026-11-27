# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Find and delete generations nobody needs any more.

Every retry leaves a still or clip behind. Once an asset has passed, only its selected attempt
is used; the rest are bad takes. They are removed from disk and cleared from the manifest, whose
QC scores and prompts stay as the audit trail. An asset that is still pending or that ran out of
attempts keeps everything, since a person may need to choose between its takes.
"""

from __future__ import annotations

import contextlib
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

from mvdirector.errors import ConfigError
from mvdirector.models import AssetKind, AssetState, AssetStatus, Manifest
from mvdirector.project import Project

# The names a generation is saved under: ``s012_a03.mp4``, ``s012_a03_upscaled.mp4``.
_GENERATION_NAME = re.compile(r"[\w-]+_a\d{2,}(_upscaled)?\.(png|mp4)")
_SEGMENT_CACHE_NAME = re.compile(r"[\w-]+(_draft)?\.(mp4|mp4\.json|loop\.mp4|hud\.png)")
_GENERATION_FOLDERS = ("assets/stills", "assets/videos")


@dataclass
class CleanupPlan:
    superseded: list[Path] = field(default_factory=list)  # takes beaten by a selected attempt
    orphaned: list[Path] = field(default_factory=list)  # generations the manifest never mentions
    cache_files: list[Path] = field(default_factory=list)  # local .comfy_cache render entries
    size_bytes: int = 0  # measured when planned, so it can still be reported after deleting

    @property
    def files(self) -> list[Path]:
        return [*self.superseded, *self.orphaned, *self.cache_files]


def _has_symlink_component(root: Path, relative: str | Path) -> bool:
    """Return True if any path component under ``root`` for ``relative`` is a symlink."""
    current = root
    for part in Path(relative).parts:
        if part in ("", "."):
            continue
        current = current / part
        if current.is_symlink():
            return True
    return False


def _resolve_in_project(project: Project, relative: str) -> Path | None:
    """Resolved path for a manifest entry, or None when it (or a link on it) leaves the project."""
    try:
        return project.path(relative)
    except ConfigError:
        return None


def _is_generation_file(project: Project, path: Path) -> bool:
    """True for a file the runner writes: ``assets/{stills,videos}/<shot>_aNN.{png,mp4}``.

    Attempt paths are read back from manifest.json, which people edit by hand, so a stale or
    mistyped entry can name any file in the project. Only generation-shaped files in the
    generation folders are ever deleted; anything else (inputs, project.yaml, locked
    references) is left alone even when an unselected attempt points at it.
    """
    return path.parent in {project.path(folder) for folder in _GENERATION_FOLDERS} and bool(
        _GENERATION_NAME.fullmatch(path.name)
    )


def _parse_utc(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _file_mtime_utc(path: Path) -> datetime:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)


def _is_expired(ts: datetime, cutoff: datetime | None) -> bool:
    return cutoff is None or ts <= cutoff


def plan_cleanup(
    project: Project,
    manifest: Manifest,
    *,
    retention_days: int | None = None,
    now: datetime | None = None,
) -> CleanupPlan:
    """What would be deleted; nothing is changed."""
    cutoff: datetime | None = None
    if retention_days is not None:
        if isinstance(retention_days, bool) or retention_days < 0:
            raise ConfigError(
                f"retention_days must be a non-negative integer, got {retention_days!r}."
            )
        ref_now = now if now is not None else datetime.now(timezone.utc)
        if ref_now.tzinfo is None:
            ref_now = ref_now.replace(tzinfo=timezone.utc)
        else:
            ref_now = ref_now.astimezone(timezone.utc)
        try:
            cutoff = ref_now - timedelta(days=retention_days)
        except OverflowError:
            # A window reaching past year 1 keeps everything rather than crashing.
            cutoff = datetime.min.replace(tzinfo=timezone.utc)

    plan = CleanupPlan()
    kept: set[Path] = set()
    superseded_candidates: list[Path] = []
    superseded_timestamps: dict[Path, list[datetime]] = {}
    for shot in manifest.shots:
        for kind in AssetKind:
            state = shot.asset(kind)
            if state.upscaled_path:
                upscaled = _resolve_in_project(project, state.upscaled_path)
                if upscaled is not None:
                    kept.add(upscaled)
            settled = _is_settled(state)
            for attempt in state.attempts:
                if not attempt.path:
                    continue
                path = _resolve_in_project(project, attempt.path)
                if path is None:
                    continue
                if not settled or attempt.number == state.selected_attempt:
                    # A kept take reached through a symlink keeps the file the link points at.
                    kept.add(path)
                elif (
                    not _has_symlink_component(project.root, attempt.path)
                    and path.is_file()
                    and _is_generation_file(project, path)
                ):
                    superseded_candidates.append(path)
                    if cutoff is not None:
                        ts = _parse_utc(attempt.created_at) or _file_mtime_utc(path)
                        superseded_timestamps.setdefault(path, []).append(ts)
    seen_superseded: set[Path] = set()
    known_superseded: set[Path] = set()
    for path in superseded_candidates:
        if path in kept:
            continue
        known_superseded.add(path)
        if path in seen_superseded:
            continue
        if cutoff is not None:
            timestamps = superseded_timestamps.get(path)
            newest = max(timestamps) if timestamps else _file_mtime_utc(path)
            if not _is_expired(newest, cutoff):
                continue
        seen_superseded.add(path)
        plan.superseded.append(path)
    for folder in _GENERATION_FOLDERS:
        if _has_symlink_component(project.root, folder):
            continue
        root = project.path(folder)
        if root.is_dir():
            plan.orphaned += sorted(
                p
                for p in root.iterdir()
                if not p.is_symlink()
                and p.is_file()
                and _GENERATION_NAME.fullmatch(p.name)
                and p not in kept
                and p not in known_superseded
                and _is_expired(_file_mtime_utc(p), cutoff)
            )
    if not _has_symlink_component(project.root, "assets/segments"):
        seg_root = project.path("assets/segments")
        if seg_root.is_dir():
            active_segments = {
                path
                for ch in manifest.chapters
                for draft in (False, True)
                for path in (
                    project.segment_video_path(ch.id, draft=draft),
                    project.segment_stamp_path(ch.id, draft=draft),
                )
            }
            plan.orphaned += sorted(
                p
                for p in seg_root.iterdir()
                if not p.is_symlink()
                and p.is_file()
                and _SEGMENT_CACHE_NAME.fullmatch(p.name)
                and p not in active_segments
                and _is_expired(_file_mtime_utc(p), cutoff)
            )
    cache_root = project.root / ".comfy_cache"
    if cache_root.is_dir() and not cache_root.is_symlink():
        cache_files: list[Path] = []
        for dirpath, dirnames, filenames in os.walk(cache_root, followlinks=False):
            current = Path(dirpath)
            dirnames[:] = [d for d in dirnames if not (current / d).is_symlink()]
            for filename in filenames:
                candidate = current / filename
                if (
                    not candidate.is_symlink()
                    and candidate.is_file()
                    and _is_expired(_file_mtime_utc(candidate), cutoff)
                ):
                    cache_files.append(candidate)
        plan.cache_files = sorted(cache_files)
    plan.size_bytes = sum(path.stat().st_size for path in plan.files)
    return plan


def apply_cleanup(project: Project, manifest: Manifest, plan: CleanupPlan) -> None:
    """Delete the planned files and clear the superseded attempts' paths in the manifest."""
    gone = set(plan.superseded)
    for shot in manifest.shots:
        for kind in AssetKind:
            for attempt in shot.asset(kind).attempts:
                if (
                    attempt.path
                    and not _has_symlink_component(project.root, attempt.path)
                    and _resolve_in_project(project, attempt.path) in gone
                ):
                    attempt.path = None
    for path in plan.files:
        if not path.is_symlink():
            path.unlink(missing_ok=True)
    cache_root = project.root / ".comfy_cache"
    if plan.cache_files and cache_root.is_dir() and not cache_root.is_symlink():
        for dirpath, dirnames, _ in os.walk(cache_root, topdown=False, followlinks=False):
            current = Path(dirpath)
            for d in dirnames:
                subdir = current / d
                if subdir.is_dir() and not subdir.is_symlink():
                    with contextlib.suppress(OSError):
                        subdir.rmdir()
        with contextlib.suppress(OSError):
            cache_root.rmdir()


def _is_settled(state: AssetState) -> bool:
    """A passed asset with a selected attempt: its other takes are no longer needed."""
    return state.status is AssetStatus.PASSED and state.selected_attempt is not None
