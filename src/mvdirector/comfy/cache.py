# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Content-addressed ComfyUI render cache.

Canonicalizes a rendered ComfyUI API graph together with the SHA-256 digests of every referenced
local input file (reference portraits, set plates, props, audio slices, previz clips) into a single
deterministic key. Re-running an unchanged shot or retrying an identical configuration hits the
local cache instead of spending redundant GPU/API time.

Credits & Attribution:
    Content-addressed graph canonicalization and input-byte hashing inspired by
    ``tysonfarrington-cmyk/xenoform``
    (https://github.com/tysonfarrington-cmyk/xenoform — ``src/xenoform/morph_cache.py``).
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_EPHEMERAL_INPUT_KEYS = frozenset({"filename_prefix"})
_CHUNK_SIZE = 1024 * 1024
_SAFE_SUFFIX_RE = re.compile(r"\A\.[a-z0-9]{1,8}\Z")
_SHA256_HEX_RE = re.compile(r"\A[0-9a-f]{64}\Z")


def file_sha256(path: Path) -> str:
    """Stream-hash a local file to a lowercase hex SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_value(value: Any, uploaded_digests: Mapping[str, str]) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"Non-finite float in ComfyUI graph: {value!r}")
        return int(value) if value.is_integer() else round(value, 6)
    if isinstance(value, str):
        if value in uploaded_digests:
            return f"sha256:{uploaded_digests[value]}"
        return value
    if isinstance(value, list):
        return [_normalize_value(item, uploaded_digests) for item in value]
    if isinstance(value, dict):
        return {
            str(k): _normalize_value(v, uploaded_digests)
            for k, v in sorted(value.items(), key=lambda pair: str(pair[0]))
            if k not in _EPHEMERAL_INPUT_KEYS
        }
    return value


def _collect_strings(value: Any, out: set[str]) -> None:
    if isinstance(value, str):
        out.add(value)
    elif isinstance(value, list):
        for item in value:
            _collect_strings(item, out)
    elif isinstance(value, dict):
        for k, v in value.items():
            if k not in _EPHEMERAL_INPUT_KEYS:
                _collect_strings(v, out)


def canonical_graph_key(
    graph: Mapping[str, Any],
    uploaded_files: Mapping[str, Path] | None = None,
    *,
    uploaded_digests: Mapping[str, str] | None = None,
    output_node: str | None = None,
    extensions: tuple[str, ...] = (),
) -> str:
    """Compute a deterministic SHA-256 key for a rendered ComfyUI API graph and its input files.

    Only hashes files whose uploaded remote names actually appear inside ``graph`` (and reuses
    any precomputed ``uploaded_digests``) to avoid quadratic re-hashing across large runs.
    """
    referenced: set[str] = set()
    _collect_strings(dict(graph), referenced)

    digests: dict[str, str] = {}
    for remote_name, digest in (uploaded_digests or {}).items():
        if remote_name in referenced:
            digests[remote_name] = digest
    for remote_name, local_path in (uploaded_files or {}).items():
        if remote_name in referenced and remote_name not in digests and local_path.is_file():
            digests[remote_name] = file_sha256(local_path)

    payload = {
        "graph": _normalize_value(dict(graph), digests),
        "output_node": output_node,
        "extensions": sorted(ext.lower() for ext in extensions),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


class ComfyRenderCache:
    """Local content-addressed file cache stored under ``<project>/.comfy_cache``."""

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir

    def _entry_path(self, key: str, suffix: str) -> Path:
        low_suffix = suffix.lower()
        safe_suffix = low_suffix if _SAFE_SUFFIX_RE.match(low_suffix) else ".bin"
        if not _SHA256_HEX_RE.match(key):
            raise ValueError(f"Invalid cache key: {key!r}")
        return self.cache_dir / key[:2] / f"{key}{safe_suffix}"

    def _is_confined(self, path: Path) -> bool:
        try:
            cache_root = self.cache_dir.resolve()
            return path.resolve().is_relative_to(cache_root)
        except OSError:
            return False

    def lookup(self, key: str, destination: Path) -> bool:
        """Copy a cached render into ``destination`` atomically if present. Returns True on hit."""
        entry = self._entry_path(key, destination.suffix or ".bin")
        if (
            self.cache_dir.is_symlink()
            or entry.parent.is_symlink()
            or entry.is_symlink()
            or not self._is_confined(entry)
            or not entry.is_file()
            or entry.stat().st_size == 0
        ):
            return False
        destination.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".cache", dir=destination.parent
        )
        os.close(fd)
        tmp_path = Path(tmp_name)
        try:
            shutil.copy2(entry, tmp_path)
            os.replace(tmp_path, destination)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise
        log.debug("ComfyUI cache hit (%s) -> %s", key[:12], destination.name)
        return True

    def store(self, key: str, source: Path) -> Path:
        """Persist ``source`` into the content-addressed cache atomically."""
        if source.is_symlink() or not source.is_file() or source.stat().st_size == 0:
            raise ValueError(f"Cannot cache missing, empty, or symlinked source: {source}")
        if self.cache_dir.is_symlink():
            raise ValueError(f"Cache directory must not be a symlink: {self.cache_dir}")
        entry = self._entry_path(key, source.suffix or ".bin")
        if entry.parent.is_symlink():
            raise ValueError(f"Cache shard directory must not be a symlink: {entry.parent}")
        entry.parent.mkdir(parents=True, exist_ok=True)
        if not self._is_confined(entry.parent):
            raise ValueError(f"Cache entry parent escapes cache directory: {entry.parent}")
        fd, tmp_name = tempfile.mkstemp(prefix=f".{key[:12]}.", suffix=".part", dir=entry.parent)
        os.close(fd)
        tmp_path = Path(tmp_name)
        try:
            shutil.copy2(source, tmp_path)
            os.replace(tmp_path, entry)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise
        return entry
