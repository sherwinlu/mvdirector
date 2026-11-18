# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Persistence seams: where project state and finished assets live.

The pipeline always works on a local *workspace* folder, because ffmpeg, OpenCV and ComfyUI
uploads need real files. Two small interfaces decide what happens around that folder:

- `StateStore` owns the two JSON documents (analysis, manifest) and the single-writer lock.
  The local implementation uses files and a PID lock; a service can back it with a database
  and per-job lock ownership.
- `AssetStorage` moves the workspace to and from durable storage. The local implementation does
  nothing because the workspace already is the storage; a service can pull inputs from, and push
  finished assets to, an object store.

`Project.save_manifest` calls `AssetStorage.push` before the state is written, so a stored
manifest never points at an asset that was not published. `Project` also stamps
`manifest.updated_at`, so a `StateStore` only has to persist what it is given.
"""

from __future__ import annotations

import json
import os
import socket
import tempfile
import uuid
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from typing import Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel, ValidationError

from mvdirector.config import CONFIG_FILENAME
from mvdirector.errors import ConfigError, ProjectLocked
from mvdirector.models import Analysis, Manifest, utc_now

_ModelT = TypeVar("_ModelT", bound=BaseModel)


@runtime_checkable
class StateStore(Protocol):
    """Analysis and manifest persistence plus the single-writer lock."""

    def load_analysis(self) -> Analysis:
        """Return the stored analysis, or raise `ConfigError` if there is none."""
        ...

    def save_analysis(self, analysis: Analysis) -> None: ...

    def load_manifest(self) -> Manifest:
        """Return the stored manifest, or raise `ConfigError` if there is none."""
        ...

    def save_manifest(self, manifest: Manifest) -> None: ...

    def has_manifest(self) -> bool: ...

    @property
    def project_id(self) -> str:
        """Stable identity that namespaces ComfyUI uploads, or "" to derive one from the config.

        It must survive across workers and workspace copies. A store whose workspace is
        recreated for every job should return its own id (a database key), otherwise each job
        re-uploads its inputs into a new, never-cleaned subfolder.
        """
        ...

    @property
    def is_locked(self) -> bool:
        """True while another writer holds the lock."""
        ...

    def lock(self) -> AbstractContextManager[None]:
        """Context manager granting exclusive write access, or raising `ProjectLocked`.

        Implementations shared between workers must expire or reclaim a lock whose holder died
        (a lease with a heartbeat or TTL). Otherwise one crashed worker blocks the project
        forever. The local store does this by probing the holder's PID.
        """
        ...


@runtime_checkable
class AssetStorage(Protocol):
    """Moves the local workspace to and from durable storage."""

    def pull(self) -> None:
        """Make inputs and earlier assets available in the workspace before a run."""
        ...

    def push(self) -> None:
        """Publish new or changed workspace files. Must be safe to call repeatedly."""
        ...


class LocalAssetStorage:
    """The workspace is the storage, so there is nothing to move."""

    def pull(self) -> None:
        return None

    def push(self) -> None:
        return None


class LocalStateStore:
    """Files in the project folder, guarded by a PID lock that only sees this machine."""

    def __init__(self, root: Path) -> None:
        self.root = root

    @property
    def analysis_path(self) -> Path:
        return self.root / "analysis.json"

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    @property
    def lock_path(self) -> Path:
        return self.root / ".run.lock"

    def load_analysis(self) -> Analysis:
        return _load_model(self.analysis_path, Analysis, "Run `mvd analyze` first.")

    def save_analysis(self, analysis: Analysis) -> None:
        atomic_write(self.analysis_path, analysis.model_dump_json(indent=2))

    def load_manifest(self) -> Manifest:
        return _load_model(self.manifest_path, Manifest, "Run `mvd plan` first.")

    def save_manifest(self, manifest: Manifest) -> None:
        atomic_write(self.manifest_path, manifest.model_dump_json(indent=2))

    def has_manifest(self) -> bool:
        return self.manifest_path.exists()

    @property
    def project_id(self) -> str:
        """A random id persisted in `.mvd_id`, so copying the folder keeps the identity."""
        id_file = self.root / ".mvd_id"
        try:
            if id_file.is_file():
                return id_file.read_text(encoding="utf-8").strip()
            if (self.root / CONFIG_FILENAME).is_file():
                token = uuid.uuid4().hex[:16]
                atomic_write(id_file, token + "\n")
                return token
        except OSError:
            pass
        return ""

    @property
    def is_locked(self) -> bool:
        self._clear_stale_lock()
        return self.lock_path.exists()

    def _clear_stale_lock(self) -> bool:
        """Remove a lock file left behind by a dead local process. Returns True if cleared."""
        if not self.lock_path.exists():
            return False
        try:
            info = json.loads(self.lock_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        if not isinstance(info, dict):
            return False
        pid = info.get("pid")
        host = info.get("host")
        if not isinstance(pid, int) or pid <= 0:
            return False
        # Only probe PID liveness when the lock was written on this machine.
        if host is not None and host != socket.gethostname():
            return False
        if _pid_is_alive(pid):
            return False
        self.lock_path.unlink(missing_ok=True)
        return True

    @contextmanager
    def lock(self) -> Iterator[None]:
        """Guard long-running writers so two processes never overwrite the manifest."""
        self._clear_stale_lock()
        try:
            fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise ProjectLocked(
                f"Another run holds {self.lock_path}. If no run is active, delete the file."
            ) from exc
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {"pid": os.getpid(), "host": socket.gethostname(), "started": utc_now()}
                    )
                )
            yield
        finally:
            self.lock_path.unlink(missing_ok=True)


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _load_model(path: Path, model: type[_ModelT], hint: str) -> _ModelT:
    if not path.is_file():
        raise ConfigError(f"{path.name} not found in {path.parent}. {hint}")
    try:
        return model.model_validate_json(path.read_text(encoding="utf-8"))
    except ValidationError as exc:
        raise ConfigError(f"{path} does not match the expected schema:\n{exc}") from exc


def atomic_write(path: Path, text: str) -> None:
    """Write via a temp file and rename so a crash never leaves a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise
