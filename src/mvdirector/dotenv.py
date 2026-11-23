# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Project-scoped ``.env`` helpers with least-privilege allowlisting and 0600 storage.

Reading a project's ``.env`` never mutates ``os.environ`` (so multi-project processes such as
``mvd mcp`` cannot leak secrets between projects) and shell environment variables always take
precedence over values stored on disk. Only explicit LLM provider key names, ``COMFY_*``, and
safe ``MVD_*`` names may be loaded from or written to ``.env``; security-control variables such
as ``MVD_ALLOWED_ENV_VARS`` and ``MVD_MCP_TOKEN`` are never accepted from a project file.
"""

from __future__ import annotations

import logging
import os
import re
import stat
import tempfile
from collections.abc import Mapping
from pathlib import Path

from mvdirector.errors import ConfigError

log = logging.getLogger(__name__)

DOTENV_FILENAME = ".env"
_MAX_DOTENV_BYTES = 64 * 1024  # 64 KiB cap

ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
ALLOWED_ENV_PREFIXES = ("MVD_", "COMFY_")
ALLOWED_LLM_KEY_NAMES = frozenset(
    {
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
    }
)
BLOCKED_ENV_PREFIXES = (
    "AWS_",
    "AZURE_",
    "DOCKER_",
    "GCP_",
    "GH_",
    "GITHUB_",
    "GITLAB_",
    "GPG_",
    "KUBE",
    "NPM_",
    "PYPI_",
    "SSH_",
)
# Security-control and host-binding knobs must come from the operator's environment, never from
# an untrusted project directory's .env file.
_BLOCKED_DOTENV_NAMES = frozenset(
    {
        "MVD_ALLOWED_ENV_VARS",
        "MVD_MCP_TOKEN",
        "MVD_BIND_ADDR",
        "MVD_MEDIAPIPE_CACHE_DIR",
        "MVD_SYNCNET_CACHE_DIR",
    }
)
_EXPORT_PREFIX_RE = re.compile(r"^export\s+")


def is_allowed_dotenv_var(name: str) -> bool:
    """True when ``name`` is permitted in a project-local ``.env`` file."""
    if not ENV_NAME_RE.match(name):
        return False
    upper = name.upper()
    if upper in _BLOCKED_DOTENV_NAMES or upper.startswith(BLOCKED_ENV_PREFIXES):
        return False
    return upper in ALLOWED_LLM_KEY_NAMES or upper.startswith(ALLOWED_ENV_PREFIXES)


def _has_control_chars(value: str) -> bool:
    return any(ord(ch) < 32 or ord(ch) == 127 for ch in value)


def _unescape_double_quoted(raw: str) -> str:
    out: list[str] = []
    idx = 0
    while idx < len(raw):
        ch = raw[idx]
        if ch == "\\" and idx + 1 < len(raw) and raw[idx + 1] in ('"', "\\"):
            out.append(raw[idx + 1])
            idx += 2
            continue
        out.append(ch)
        idx += 1
    return "".join(out)


def _parse_value(raw: str) -> str:
    stripped = raw.strip()
    if not stripped:
        return ""
    if len(stripped) >= 2 and stripped[0] == '"' and stripped[-1] == '"':
        return _unescape_double_quoted(stripped[1:-1])
    if len(stripped) >= 2 and stripped[0] == "'" and stripped[-1] == "'":
        return stripped[1:-1]
    return stripped


def parse_dotenv(text: str) -> dict[str, str]:
    """Parse ``KEY=VALUE`` pairs from ``text``, retaining only allowlisted variable names."""
    values: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        line = _EXPORT_PREFIX_RE.sub("", line, count=1)
        if "=" not in line:
            continue
        key, raw_val = line.split("=", 1)
        key = key.strip()
        if not is_allowed_dotenv_var(key):
            log.debug("Ignoring disallowed variable '%s' in project .env", key)
            continue
        val = _parse_value(raw_val)
        if not val or _has_control_chars(val):
            continue
        values[key] = val
    return values


def _read_regular_text(path: Path, *, warn_permissions: bool = False) -> str | None:
    """Read a regular, non-symlinked, size-bounded UTF-8 file using O_NOFOLLOW when available."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError:
        return None
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            log.warning("Ignoring non-regular file: %s", path)
            return None
        if st.st_size > _MAX_DOTENV_BYTES:
            log.warning("Ignoring oversized %s (>%d bytes)", path, _MAX_DOTENV_BYTES)
            return None
        if warn_permissions and os.name == "posix" and (st.st_mode & 0o077):
            log.warning(
                "%s has overly broad permissions (%o); run `chmod 600 %s` to restrict access.",
                path,
                stat.S_IMODE(st.st_mode),
                path,
            )
        with os.fdopen(fd, "r", encoding="utf-8") as handle:
            fd = -1  # ownership transferred to handle
            return handle.read()
    except (OSError, UnicodeDecodeError) as exc:
        log.warning("Could not read %s: %s", path, exc)
        return None
    finally:
        if fd >= 0:
            os.close(fd)


def read_project_dotenv(project_dir: Path) -> dict[str, str]:
    """Read allowlisted variables from ``<project_dir>/.env``, returning ``{}`` if absent/unsafe."""
    dotenv_path = project_dir / DOTENV_FILENAME
    try:
        st = os.lstat(dotenv_path)
    except OSError:
        return {}
    if stat.S_ISLNK(st.st_mode) or not stat.S_ISREG(st.st_mode):
        log.warning("Ignoring symlink or non-regular .env at %s", dotenv_path)
        return {}
    text = _read_regular_text(dotenv_path, warn_permissions=True)
    return parse_dotenv(text) if text is not None else {}


def merged_project_environ(
    project_dir: Path, base: Mapping[str, str] | None = None
) -> dict[str, str]:
    """Return project ``.env`` variables overlaid with ``base`` (shell env always wins)."""
    env_base = os.environ if base is None else base
    dotenv_vals = read_project_dotenv(project_dir)
    if not dotenv_vals:
        return dict(env_base)
    return {**dotenv_vals, **env_base}


def ensure_dotenv_gitignored(project_dir: Path) -> Path:
    """Ensure ``<project_dir>/.gitignore`` ignores ``.env`` and ``.env.*``."""
    gitignore_path = project_dir / ".gitignore"
    if gitignore_path.is_symlink():
        raise ConfigError(f"Refusing to modify symlinked .gitignore at {gitignore_path}")
    existing = ""
    if gitignore_path.exists():
        if not gitignore_path.is_file():
            raise ConfigError(f"{gitignore_path} exists and is not a regular file")
        existing = gitignore_path.read_text(encoding="utf-8")
    lines = {line.strip() for line in existing.splitlines()}
    needed: list[str] = []
    if ".env" not in lines and "/.env" not in lines:
        needed.append(".env")
    if ".env.*" not in lines and "/.env.*" not in lines:
        needed.append(".env.*")
    if not needed:
        return gitignore_path
    prefix = existing
    if prefix and not prefix.endswith("\n"):
        prefix += "\n"
    updated = prefix + "\n".join(needed) + "\n"
    fd, tmp_name = tempfile.mkstemp(prefix=".gitignore.", suffix=".tmp", dir=project_dir)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(updated)
        os.replace(tmp_path, gitignore_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    return gitignore_path


def write_project_dotenv_var(project_dir: Path, name: str, value: str) -> Path:
    """Atomically save one allowlisted secret into ``<project_dir>/.env`` with mode ``0600``."""
    if not is_allowed_dotenv_var(name):
        raise ConfigError(
            f"Environment variable {name!r} cannot be stored in {DOTENV_FILENAME}. "
            "Only standard LLM key names (OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY, "
            "GOOGLE_API_KEY), COMFY_*, and safe MVD_* variables are permitted."
        )
    cleaned = value.strip()
    if not cleaned or _has_control_chars(cleaned):
        raise ConfigError(
            f"Value for {name} must be a non-empty single-line string without control characters."
        )
    if not project_dir.is_dir():
        raise ConfigError(f"Project folder does not exist: {project_dir}")

    dotenv_path = project_dir / DOTENV_FILENAME
    if dotenv_path.is_symlink():
        raise ConfigError(f"Refusing to overwrite symlinked {dotenv_path}")

    existing_lines: list[str] = []
    if dotenv_path.exists():
        raw = _read_regular_text(dotenv_path)
        if raw is None:
            raise ConfigError(f"Could not safely read existing {dotenv_path}")
        existing_lines = raw.splitlines(keepends=True)
    else:
        existing_lines = [
            "# Local project secrets (owner-only 0600, gitignored). Never commit this file.\n"
        ]

    escaped = cleaned.replace("\\", "\\\\").replace('"', '\\"')
    new_line = f'{name}="{escaped}"\n'
    key_re = re.compile(rf"^[ \t]*(?:export\s+)?{re.escape(name)}[ \t]*=")
    replaced = False
    out_lines: list[str] = []
    for line in existing_lines:
        if key_re.match(line):
            if not replaced:
                out_lines.append(new_line)
                replaced = True
        else:
            out_lines.append(line)
    if not replaced:
        if out_lines and not out_lines[-1].endswith("\n"):
            out_lines[-1] += "\n"
        out_lines.append(new_line)

    fd, tmp_name = tempfile.mkstemp(prefix=".env.", suffix=".tmp", dir=project_dir)
    tmp_path = Path(tmp_name)
    try:
        if os.name == "posix":
            os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            fd = -1
            handle.write("".join(out_lines))
        os.replace(tmp_path, dotenv_path)
        if os.name == "posix":
            os.chmod(dotenv_path, 0o600)
    except BaseException:
        if fd >= 0:
            os.close(fd)
        tmp_path.unlink(missing_ok=True)
        raise

    ensure_dotenv_gitignored(project_dir)
    return dotenv_path
