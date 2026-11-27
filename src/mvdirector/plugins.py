# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Entry-point plugin registry for optional QC, audio, pacing, and clip-selection extensions.

Configuration files reference plugins strictly by registered entry-point name (never by import
path), preventing arbitrary module loading from untrusted ``project.yaml`` or MCP inputs.
"""

from __future__ import annotations

import logging
import re
import threading
from importlib import metadata
from typing import Any

from mvdirector.errors import ConfigError

log = logging.getLogger(__name__)

QC_CHECKS_GROUP = "mvdirector.qc_checks"
AUDIO_CONDITIONERS_GROUP = "mvdirector.audio_conditioners"
PACING_GROUP = "mvdirector.pacing"
CLIP_SELECTORS_GROUP = "mvdirector.clip_selectors"

ALLOWED_GROUPS: frozenset[str] = frozenset(
    {
        QC_CHECKS_GROUP,
        AUDIO_CONDITIONERS_GROUP,
        PACING_GROUP,
        CLIP_SELECTORS_GROUP,
    }
)

PLUGIN_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}\Z")

_CACHE_LOCK = threading.Lock()
_LOADED_CACHE: dict[tuple[str, str], Any] = {}
_INFLIGHT_LOCKS: dict[tuple[str, str], threading.RLock] = {}


def validate_plugin_group(group: str) -> str:
    """Validate that ``group`` is one of the allowed mvdirector entry-point groups."""
    if not isinstance(group, str) or group not in ALLOWED_GROUPS:
        allowed = ", ".join(sorted(ALLOWED_GROUPS))
        raise ConfigError(f"Unsupported plugin group {group!r}: must be one of {allowed}.")
    return group


def validate_plugin_name(name: str, *, label: str = "plugin") -> str:
    """Validate a plugin name against the strict identifier allowlist."""
    if not isinstance(name, str) or not PLUGIN_NAME_RE.match(name):
        raise ConfigError(
            f"Invalid {label} name {name!r}: must match {PLUGIN_NAME_RE.pattern} "
            "(registered entry-point name only; module paths are not allowed)."
        )
    return name


def list_plugins(group: str) -> dict[str, metadata.EntryPoint]:
    """Return ``{name: EntryPoint}`` for all installed plugins in ``group``."""
    validate_plugin_group(group)
    eps: dict[str, metadata.EntryPoint] = {}
    for ep in metadata.entry_points(group=group):
        if ep.name in eps:
            log.warning(
                "Duplicate entry point %r in plugin group %r; keeping first registration.",
                ep.name,
                group,
            )
            continue
        eps[ep.name] = ep
    return eps


def load_plugin(group: str, name: str) -> Any:
    """Load and cache the registered plugin ``name`` from entry-point ``group``."""
    validate_plugin_group(group)
    clean = validate_plugin_name(name)
    key = (group, clean)
    with _CACHE_LOCK:
        if key in _LOADED_CACHE:
            return _LOADED_CACHE[key]
        key_lock = _INFLIGHT_LOCKS.setdefault(key, threading.RLock())
    with key_lock:
        with _CACHE_LOCK:
            if key in _LOADED_CACHE:
                return _LOADED_CACHE[key]
        eps = list_plugins(group)
        ep = eps.get(clean)
        if ep is None:
            available = ", ".join(sorted(eps)) or "none"
            raise ConfigError(
                f"Unknown plugin {clean!r} in group {group!r} (installed: {available})."
            )
        loaded = ep.load()
        with _CACHE_LOCK:
            _LOADED_CACHE[key] = loaded
            _INFLIGHT_LOCKS.pop(key, None)
        return loaded


def plugin_version(plugin: Any) -> str:
    """Best-effort version string for cache-stamping plugin outputs."""
    ver = getattr(plugin, "__version__", None)
    return str(ver) if ver not in (None, "") else "0"


def clear_cache() -> None:
    """Clear the loaded plugin cache (intended for tests)."""
    with _CACHE_LOCK:
        _LOADED_CACHE.clear()
        _INFLIGHT_LOCKS.clear()
