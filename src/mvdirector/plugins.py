# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Entry-point plugin registry for optional QC, audio, pacing, and clip-selection extensions.

Configuration files reference plugins strictly by registered entry-point name (never by import
path), preventing arbitrary module loading from untrusted ``project.yaml`` or MCP inputs.
"""

from __future__ import annotations

import re
import threading
from importlib import metadata
from typing import Any

from mvdirector.errors import ConfigError

QC_CHECKS_GROUP = "mvdirector.qc_checks"
AUDIO_CONDITIONERS_GROUP = "mvdirector.audio_conditioners"
PACING_GROUP = "mvdirector.pacing"
CLIP_SELECTORS_GROUP = "mvdirector.clip_selectors"

PLUGIN_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}\Z")

_CACHE_LOCK = threading.Lock()
_LOADED_CACHE: dict[tuple[str, str], Any] = {}


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
    return {ep.name: ep for ep in metadata.entry_points(group=group)}


def load_plugin(group: str, name: str) -> Any:
    """Load and cache the registered plugin ``name`` from entry-point ``group``."""
    clean = validate_plugin_name(name)
    key = (group, clean)
    with _CACHE_LOCK:
        if key in _LOADED_CACHE:
            return _LOADED_CACHE[key]
    eps = list_plugins(group)
    ep = eps.get(clean)
    if ep is None:
        available = ", ".join(sorted(eps)) or "none"
        raise ConfigError(f"Unknown plugin {clean!r} in group {group!r} (installed: {available}).")
    loaded = ep.load()
    with _CACHE_LOCK:
        _LOADED_CACHE[key] = loaded
    return loaded


def plugin_version(plugin: Any) -> str:
    """Best-effort version string for cache-stamping plugin outputs."""
    ver = getattr(plugin, "__version__", None)
    return str(ver) if ver not in (None, "") else "0"


def clear_cache() -> None:
    """Clear the loaded plugin cache (intended for tests)."""
    with _CACHE_LOCK:
        _LOADED_CACHE.clear()
