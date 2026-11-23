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

import errno
import json
import os
import socket
import tempfile
import time
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

    @property
    def _reclaim_path(self) -> Path:
        return self.root / ".run.lock.reclaim"

    @staticmethod
    def _is_stale_lock_info(info: object) -> bool:
        if not isinstance(info, dict):
            return False
        pid = info.get("pid")
        host = info.get("host")
        if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
            return False
        # Only probe PID liveness when the lock was written on this machine.
        if host is not None and host != socket.gethostname():
            return False
        return not _pid_is_alive(pid)

    def _clear_stale_lock(self) -> bool:
        """Remove a lock file left behind by a dead local process without racing live holders."""
        try:
            st_before = self.lock_path.stat()
            info = json.loads(self.lock_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        if not self._is_stale_lock_info(info):
            return False

        # Serialize concurrent stale-lock reclaimers so a second reclaimer cannot unlink a live
        # lock newly created by the winner.
        reclaim_fd: int | None = None
        try:
            reclaim_fd = os.open(self._reclaim_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                if time.time() - self._reclaim_path.stat().st_mtime > 30.0:
                    self._reclaim_path.unlink(missing_ok=True)
            except OSError:
                pass
            return False
        except OSError:
            return False
        try:
            os.close(reclaim_fd)
            reclaim_fd = None
            try:
                st_now = self.lock_path.stat()
                if (st_now.st_dev, st_now.st_ino) != (st_before.st_dev, st_before.st_ino):
                    return False
                now_info = json.loads(self.lock_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return False
            if not self._is_stale_lock_info(now_info):
                return False
            self.lock_path.unlink(missing_ok=True)
            return True
        finally:
            if reclaim_fd is not None:
                os.close(reclaim_fd)
            self._reclaim_path.unlink(missing_ok=True)

    def _acquire_lock_file(self, payload: str) -> tuple[int, int] | None:
        """Create ``.run.lock`` atomically with ``payload`` already written (no 0-byte window)."""
        self.root.mkdir(parents=True, exist_ok=True)
        tmp_fd, tmp_name = tempfile.mkstemp(dir=self.root, prefix=".run.lock.", suffix=".tmp")
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
            tmp_stat = tmp_path.stat()
            try:
                os.link(tmp_path, self.lock_path)
                return (tmp_stat.st_dev, tmp_stat.st_ino)
            except OSError as exc:
                if exc.errno == errno.EEXIST:
                    raise FileExistsError from exc
                if exc.errno not in (errno.EPERM, errno.EXDEV, errno.ENOTSUP, errno.EOPNOTSUPP):
                    raise
        finally:
            tmp_path.unlink(missing_ok=True)

        # Fallback for filesystems that disallow hard links.
        fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
            st = self.lock_path.stat()
            return (st.st_dev, st.st_ino)
        except BaseException:
            self.lock_path.unlink(missing_ok=True)
            raise

    @contextmanager
    def lock(self) -> Iterator[None]:
        """Guard long-running writers so two processes never overwrite the manifest."""
        payload = json.dumps(
            {
                "pid": os.getpid(),
                "host": socket.gethostname(),
                "started": utc_now(),
                "token": uuid.uuid4().hex[:12],
            }
        )
        owner_id: tuple[int, int] | None = None
        for attempt in range(2):
            try:
                owner_id = self._acquire_lock_file(payload)
                break
            except FileExistsError as exc:
                if attempt == 0 and self._clear_stale_lock():
                    continue
                raise ProjectLocked(
                    f"Another run holds {self.lock_path}. If no run is active, delete the file."
                ) from exc
        try:
            yield
        finally:
            try:
                if owner_id is None:
                    self.lock_path.unlink(missing_ok=True)
                else:
                    st = self.lock_path.stat()
                    if (st.st_dev, st.st_ino) == owner_id:
                        self.lock_path.unlink(missing_ok=True)
            except OSError:
                pass


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
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ConfigError(f"Could not read {path}: {exc}") from exc
    try:
        return model.model_validate_json(raw)
    except ValidationError as exc:
        raise ConfigError(f"{path} does not match the expected schema:\n{exc}") from exc


def path_is_file(path: Path) -> bool:
    """``Path.is_file()`` that answers False for an unusable path instead of raising.

    ``is_file()`` swallows only a short list of errnos (ENOENT, ENOTDIR, EBADF, ELOOP), so a path
    the OS rejects outright still propagates ``OSError`` — most easily an over-long component,
    which raises ENAMETOOLONG. Callers testing a path that came from a user or an MCP client
    want "no such file" there, not a traceback.
    """
    try:
        return path.is_file()
    except OSError:
        return False


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
