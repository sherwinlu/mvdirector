# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""A project is one song: a folder holding config, inputs, the manifest and all assets."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path

from mvdirector.config import CONFIG_FILENAME, ProjectConfig, load_config
from mvdirector.errors import ConfigError
from mvdirector.models import Analysis, AssetKind, Manifest, Shot, utc_now, validate_safe_id
from mvdirector.store import (
    AssetStorage,
    LocalAssetStorage,
    LocalStateStore,
    StateStore,
    atomic_write,
)

_ASSET_EXTENSION = {AssetKind.STILL: ".png", AssetKind.VIDEO: ".mp4"}


def require_safe_id(value: str, label: str) -> str:
    try:
        return validate_safe_id(value, label)
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc


_require_safe_id = require_safe_id


class Project:
    """Resolves paths inside a project folder and loads/saves its state files."""

    def __init__(
        self,
        root: Path,
        config: ProjectConfig,
        state: StateStore | None = None,
        storage: AssetStorage | None = None,
    ) -> None:
        self.root = root.resolve()
        self.config = config
        self.state: StateStore = state or LocalStateStore(self.root)
        self.storage: AssetStorage = storage or LocalAssetStorage()

    @classmethod
    def open(
        cls,
        root: Path,
        state: StateStore | None = None,
        storage: AssetStorage | None = None,
    ) -> Project:
        """Open a workspace folder. `state` and `storage` default to the local implementations."""
        return cls(root, load_config(Path(root) / CONFIG_FILENAME), state, storage)

    # ------------------------------------------------------------------ paths

    def path(self, relative: str | Path) -> Path:
        """Absolute path for a project-relative path, rejecting any escape outside root."""
        raw = Path(relative)
        if raw.is_absolute():
            raise ConfigError(
                f"Path '{relative}' must be relative to the project folder ({self.root})."
            )
        resolved = (self.root / raw).resolve()
        if not resolved.is_relative_to(self.root):
            raise ConfigError(f"Path '{relative}' escapes the project folder ({self.root}).")
        return resolved

    def write_text(self, relative: str | Path, text: str) -> Path:
        """Atomically write UTF-8 text to a project-relative path."""
        target = self.path(relative)
        atomic_write(target, text)
        return target

    def rel(self, absolute: Path) -> str:
        """Project-relative POSIX string, the form stored in the manifest."""
        resolved = Path(absolute).resolve()
        if not resolved.is_relative_to(self.root):
            raise ConfigError(f"Path '{absolute}' is outside the project folder ({self.root}).")
        return resolved.relative_to(self.root).as_posix()

    @property
    def analysis_path(self) -> Path:
        """Workspace location of analysis.json (what the local state store reads and writes)."""
        return self.root / "analysis.json"

    @property
    def manifest_path(self) -> Path:
        """Workspace location of manifest.json (what the local state store reads and writes)."""
        return self.root / "manifest.json"

    @property
    def reference_dir(self) -> Path:
        return self.root / "assets" / "reference"

    @property
    def plates_dir(self) -> Path:
        return self.root / "assets" / "plates"

    @property
    def props_dir(self) -> Path:
        return self.root / "assets" / "props"

    @property
    def previz_dir(self) -> Path:
        return self.root / "assets" / "previz"

    @property
    def segments_dir(self) -> Path:
        return self.root / "assets" / "segments"

    @property
    def output_dir(self) -> Path:
        return self.root / "output"

    @property
    def comfy_subfolder(self) -> str:
        """Portable ComfyUI input subfolder so projects stay isolated without host-path coupling."""
        slug = re.sub(r"[^a-z0-9_-]+", "_", self.root.name.lower()).strip("_-") or "project"
        token = self.state.project_id
        identity = token or f"{self.root.name}\0{self.config.title}\0{self.config.seed}"
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:8]
        return f"mvdirector/{slug[:32]}_{digest}"

    def asset_path(self, shot: Shot, kind: AssetKind, attempt: int) -> Path:
        shot_id = _require_safe_id(shot.id, "shot id")
        filename = f"{shot_id}_a{attempt:02d}{_ASSET_EXTENSION[kind]}"
        return self.path(Path("assets") / f"{kind.value}s" / filename)

    def upscaled_video_path(self, shot_id: str, attempt: int) -> Path:
        safe_id = _require_safe_id(shot_id, "shot id")
        return self.path(Path("assets") / "videos" / f"{safe_id}_a{attempt:02d}_upscaled.mp4")

    def audio_slice_path(self, shot_id: str) -> Path:
        safe_id = _require_safe_id(shot_id, "shot id")
        return self.path(Path("assets") / "audio_slices" / f"{safe_id}.wav")

    def audio_video_slice_path(self, shot_id: str) -> Path:
        safe_id = _require_safe_id(shot_id, "shot id")
        return self.path(Path("assets") / "audio_slices" / f"{safe_id}.mp4")

    def previz_path(self, shot_id: str, suffix: str = ".mp4") -> Path:
        safe_id = _require_safe_id(shot_id, "shot id")
        if not re.fullmatch(r"\.[a-z0-9]{1,8}", suffix):
            raise ConfigError(f"Unsupported previz file extension: {suffix!r}")
        return self.path(Path("assets") / "previz" / f"{safe_id}{suffix}")

    def previz_plan_path(self, shot_id: str) -> Path:
        safe_id = _require_safe_id(shot_id, "shot id")
        return self.path(Path("assets") / "previz" / f"{safe_id}.plan.json")

    def segment_video_path(self, chapter_id: str, *, draft: bool = False) -> Path:
        safe_id = _require_safe_id(chapter_id, "chapter id")
        suffix = "_draft" if draft else ""
        return self.path(Path("assets") / "segments" / f"{safe_id}{suffix}.mp4")

    def segment_stamp_path(self, chapter_id: str, *, draft: bool = False) -> Path:
        video_path = self.segment_video_path(chapter_id, draft=draft)
        return video_path.with_name(f"{video_path.name}.json")

    def input_path(self, relative: str | None, label: str) -> Path:
        if not relative:
            raise ConfigError(f"inputs.{label} is not set in {CONFIG_FILENAME}")
        path = self.path(relative)
        if not path.is_file():
            raise ConfigError(f"Input file for '{label}' not found: {path}")
        return path

    # ------------------------------------------------------------------ state

    def load_analysis(self) -> Analysis:
        return self.state.load_analysis()

    def save_analysis(self, analysis: Analysis) -> None:
        self.storage.push()
        self.state.save_analysis(analysis)

    def load_manifest(self) -> Manifest:
        return self.state.load_manifest()

    def save_manifest(self, manifest: Manifest) -> None:
        # Publish assets first so a stored manifest never points at a file that is not durable.
        manifest.updated_at = utc_now()
        self.storage.push()
        self.state.save_manifest(manifest)

    # ------------------------------------------------------------------- lock

    @property
    def is_locked(self) -> bool:
        return self.state.is_locked

    def run_lock(self) -> AbstractContextManager[None]:
        """Guard long-running writers so two workers never overwrite the manifest."""
        return self.state.lock()

    @contextmanager
    def edit_manifest(self) -> Iterator[Manifest]:
        """Acquire the run lock, load the manifest, and atomically save it on clean exit."""
        with self.run_lock():
            manifest = self.load_manifest()
            yield manifest
            self.save_manifest(manifest)
