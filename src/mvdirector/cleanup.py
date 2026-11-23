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
from pathlib import Path

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


def plan_cleanup(project: Project, manifest: Manifest) -> CleanupPlan:
    """What would be deleted; nothing is changed."""
    plan = CleanupPlan()
    kept: set[Path] = set()
    superseded_candidates: list[Path] = []
    for shot in manifest.shots:
        for kind in AssetKind:
            state = shot.asset(kind)
            if state.upscaled_path:
                kept.add(project.path(state.upscaled_path))
            settled = _is_settled(state)
            for attempt in state.attempts:
                if not attempt.path:
                    continue
                path = project.path(attempt.path)
                if not settled or attempt.number == state.selected_attempt:
                    kept.add(path)
                elif path.is_file() and not path.is_symlink():
                    superseded_candidates.append(path)
    seen_superseded: set[Path] = set()
    for path in superseded_candidates:
        if path not in kept and path not in seen_superseded:
            seen_superseded.add(path)
            plan.superseded.append(path)
    for folder in _GENERATION_FOLDERS:
        root = project.path(folder)
        if root.is_dir() and not root.is_symlink():
            plan.orphaned += sorted(
                p
                for p in root.iterdir()
                if not p.is_symlink()
                and p.is_file()
                and _GENERATION_NAME.fullmatch(p.name)
                and p not in kept
                and p not in seen_superseded
            )
    seg_root = project.path("assets/segments")
    if seg_root.is_dir() and not seg_root.is_symlink():
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
        )
    cache_root = project.root / ".comfy_cache"
    if cache_root.is_dir() and not cache_root.is_symlink():
        cache_files: list[Path] = []
        for dirpath, dirnames, filenames in os.walk(cache_root, followlinks=False):
            current = Path(dirpath)
            dirnames[:] = [d for d in dirnames if not (current / d).is_symlink()]
            for filename in filenames:
                candidate = current / filename
                if not candidate.is_symlink() and candidate.is_file():
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
                if attempt.path and project.path(attempt.path) in gone:
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
