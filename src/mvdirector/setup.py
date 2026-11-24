# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Local environment diagnostics and interactive project setup helpers."""

from __future__ import annotations

import importlib.util
import ipaddress
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import urllib.parse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import yaml
from pydantic import ValidationError

from mvdirector.comfy.client import ComfyClient
from mvdirector.comfy.workflow import WorkflowTemplate
from mvdirector.config import (
    CONFIG_FILENAME,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_PROVIDER,
    ComfyConfig,
    LLMConfig,
    ProjectConfig,
    is_allowed_api_key_env,
    is_allowed_env_var,
    mask_unset_allowed_env_refs,
)
from mvdirector.dotenv import merged_project_environ, write_project_dotenv_var
from mvdirector.errors import (
    ComfyConnectionError,
    ComfyJobError,
    ConfigError,
    DependencyMissing,
    MVDError,
)
from mvdirector.llm.check import ErrorKind, LLMCheckResult, run_llm_check
from mvdirector.llm.factory import PROVIDERS, build_llm
from mvdirector.llm.http import redact_url
from mvdirector.models import ProjectMode, VideoStyle
from mvdirector.project import Project
from mvdirector.store import atomic_write
from mvdirector.tuning import (
    format_yaml_scalar,
    update_top_level_yaml_scalar,
    update_yaml_scalar,
)

DEFAULT_COMFY_URL = "http://127.0.0.1:8188"
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_PROBE_TIMEOUT = 5.0
_SAFE_OLLAMA_MODEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}\Z")

PROVIDER_ALIASES: dict[str, str] = {
    "chatgpt": "openai",
    "gpt": "openai",
    "claude": "anthropic",
    "google": "gemini",
    "local": "ollama",
}

PROVIDER_MODEL_EXAMPLES: dict[str, str] = {
    "openai": "gpt-4o",
    "anthropic": "claude-sonnet-4-5",
    "gemini": "gemini-2.5-flash",
}


def normalize_llm_provider(raw: str) -> str:
    """Normalize user input (including friendly aliases like ``chatgpt`` or ``claude``)."""
    cleaned = raw.strip().lower()
    return PROVIDER_ALIASES.get(cleaned, cleaned)


def provider_key_env(provider: str) -> str | None:
    """Return the default API key env var for ``provider``, or None if none is required."""
    info = PROVIDERS.get(provider)
    return info.key_env if info is not None else None


_VISION_MODEL_PREFIXES = (
    "gemma3",
    "qwen2.5vl",
    "qwen3-vl",
    "llama3.2-vision",
    "llava",
    "minicpm-v",
    "pixtral",
    "mistral-small3.1",
)

_EXTRAS_SPEC: tuple[tuple[str, tuple[str, ...], str], ...] = (
    ("qc", ("mediapipe",), "MediaPipe face & lip-sync quality checks"),
    ("align", ("faster_whisper",), "Whisper word-level lyric alignment"),
    ("stems", ("demucs",), "Demucs vocal/drum stem separation (`mvd prepare`)"),
    ("ui", ("streamlit",), "Streamlit browser review UI (`mvd review`)"),
    ("anthropic", ("anthropic",), "Anthropic Claude LLM backend"),
    ("syncnet", ("torch", "mediapipe"), "Neural Stable-SyncNet lip-sync check"),
)


@dataclass(frozen=True)
class VideoPreset:
    """Model-family video timing & grid defaults matching docs/workflows.md."""

    key: str
    label: str
    frame_multiple: int | None = None
    frame_offset: int | None = None
    size_multiple: int | None = None
    max_clip_seconds: float | None = None
    workflow_note: str | None = None


VIDEO_PRESETS: dict[str, VideoPreset] = {
    "ltx": VideoPreset(
        key="ltx",
        label="LTX-2 / LTX-2.3 / LTX-2.5 (8n+1 frames, 64px grid)",
        frame_multiple=8,
        frame_offset=1,
        size_multiple=64,
        max_clip_seconds=8.0,
    ),
    "wan": VideoPreset(
        key="wan",
        label="Wan 2.1 + InfiniteTalk (4n+1 frames, 64px grid)",
        frame_multiple=4,
        frame_offset=1,
        size_multiple=64,
        max_clip_seconds=8.0,
        workflow_note=(
            "Set `comfy.workflows.lipsync: infinitetalk_lipsync.bindings.json` in project.yaml "
            "and export its matching `.api.json` workflow from ComfyUI."
        ),
    ),
    "wan-i2v": VideoPreset(
        key="wan-i2v",
        label="Wan 2.1 / 2.2 / HunyuanVideo I2V (4n+1 frames, 5s clips)",
        frame_multiple=4,
        frame_offset=1,
        size_multiple=1,
        max_clip_seconds=5.0,
    ),
    "exact": VideoPreset(
        key="exact",
        label="Exact-duration / cloud API or any other model (1n+0 frames, any resolution)",
        frame_multiple=1,
        frame_offset=0,
        size_multiple=1,
        max_clip_seconds=10.0,
    ),
    "custom": VideoPreset(
        key="custom",
        label="Keep current video settings in project.yaml",
    ),
}


@dataclass(frozen=True)
class ExtraStatus:
    name: str
    installed: bool
    purpose: str


@dataclass(frozen=True)
class OllamaProbe:
    url: str
    reachable: bool
    models: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass(frozen=True)
class ComfyProbe:
    url: str
    reachable: bool
    devices: list[str] = field(default_factory=list)
    vram_total_gb: float | None = None
    vram_free_gb: float | None = None
    comfy_version: str | None = None
    node_count: int | None = None
    error: str | None = None


@dataclass(frozen=True)
class ComfyWiringReport:
    """Summary of ComfyUI reachability, node registry, and local workflow wiring checks."""

    url: str
    reachable: bool
    node_count: int | None = None
    comfy_error: str | None = None
    workflows_ok: list[str] = field(default_factory=list)
    missing_exports: list[str] = field(default_factory=list)
    issues: list[tuple[str, str]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.reachable and not self.issues


@dataclass(frozen=True)
class EnvironmentReport:
    python_version: str
    platform_label: str
    ffmpeg_ok: bool
    ffprobe_ok: bool
    extras: list[ExtraStatus]
    ollama: OllamaProbe
    comfy: ComfyProbe

    @property
    def tools_ok(self) -> bool:
        return self.ffmpeg_ok and self.ffprobe_ok

    def extra_installed(self, name: str) -> bool:
        return any(e.name == name and e.installed for e in self.extras)


@dataclass(frozen=True)
class SetupOptions:
    """Configuration choices to write into a project's ``project.yaml``."""

    title: str | None = None
    mode: ProjectMode | None = None
    comfy_url: str | None = None
    comfy_token_env: str | None = None
    llm_enabled: bool | None = None
    llm_provider: str | None = None
    llm_model: str | None = None
    llm_base_url: str | None = None
    llm_api_key_env: str | None = None
    qc_on_missing_dependency: str | None = None
    qc_vision_enabled: bool | None = None
    video_preset: str | None = None


_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "host.docker.internal"})


def is_loopback_url(url: str) -> bool:
    """Return True when ``url`` points to the local host (not a separate remote server)."""
    parsed = urllib.parse.urlparse(url.strip())
    host = (parsed.hostname or "").lower()
    if host in _LOOPBACK_HOSTS:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def validate_comfy_token_env(name: str) -> str:
    """Validate an environment variable name for ``comfy.headers`` Bearer auth."""
    cleaned = name.strip()
    if not cleaned:
        raise ConfigError("ComfyUI auth token env var name must not be empty.")
    if not is_allowed_env_var(cleaned):
        raise ConfigError(
            f"Environment variable {cleaned!r} is not allowed in {CONFIG_FILENAME}. "
            "Use a COMFY_* or MVD_* name (for example COMFY_TOKEN), or list it in "
            "MVD_ALLOWED_ENV_VARS."
        )
    return cleaned


def _comfy_probe_headers(
    token_env: str | None, *, environ: Mapping[str, str] | None = None
) -> dict[str, str]:
    if not token_env:
        return {}
    validated = validate_comfy_token_env(token_env)
    env = os.environ if environ is None else environ
    if not env.get(validated, "").strip():
        return {}
    return {"Authorization": f"Bearer ${{{validated}}}"}


def validate_http_url(url: str, label: str = "URL") -> str:
    """Ensure a service URL uses http/https with a valid host and no control characters."""
    cleaned = url.strip()
    if not cleaned:
        raise ConfigError(f"{label} must not be empty.")
    if any(ord(ch) < 32 or ord(ch) == 127 or ch.isspace() for ch in cleaned):
        raise ConfigError(f"{label} contains invalid whitespace or control characters.")
    parsed = urllib.parse.urlparse(cleaned)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ConfigError(f"{label} must be an http:// or https:// URL with a host (got {url!r}).")
    return cleaned.rstrip("/")


def check_extras() -> list[ExtraStatus]:
    """Report which optional mvdirector extras are importable in the active Python environment."""
    statuses: list[ExtraStatus] = []
    for name, modules, purpose in _EXTRAS_SPEC:
        installed = all(importlib.util.find_spec(mod) is not None for mod in modules)
        statuses.append(ExtraStatus(name=name, installed=installed, purpose=purpose))
    return statuses


def probe_ollama(
    base_url: str,
    *,
    timeout: float = DEFAULT_PROBE_TIMEOUT,
    transport: httpx.BaseTransport | None = None,
) -> OllamaProbe:
    """Check whether Ollama is reachable and list locally pulled models via ``GET /api/tags``."""
    try:
        normalized = validate_http_url(base_url, "Ollama URL")
    except ConfigError as exc:
        return OllamaProbe(url=base_url, reachable=False, error=str(exc))

    try:
        with httpx.Client(timeout=timeout, transport=transport) as client:
            resp = client.get(f"{normalized}/api/tags")
    except httpx.HTTPError as exc:
        return OllamaProbe(url=normalized, reachable=False, error=f"Unreachable ({exc})")

    if resp.status_code >= 400:
        return OllamaProbe(
            url=normalized,
            reachable=False,
            error=f"HTTP {resp.status_code} from {redact_url(normalized)}/api/tags",
        )

    try:
        data = resp.json()
    except ValueError:
        return OllamaProbe(
            url=normalized,
            reachable=False,
            error="Endpoint responded with non-JSON body",
        )

    raw_models = data.get("models") if isinstance(data, dict) else None
    models: list[str] = []
    if isinstance(raw_models, list):
        for entry in raw_models:
            if isinstance(entry, dict):
                name = entry.get("name") or entry.get("model")
                if isinstance(name, str) and name.strip():
                    models.append(name.strip())
    return OllamaProbe(url=normalized, reachable=True, models=models)


def probe_comfy(
    base_url: str,
    *,
    headers: dict[str, str] | None = None,
    timeout: float = DEFAULT_PROBE_TIMEOUT,
    transport: httpx.BaseTransport | None = None,
    environ: Mapping[str, str] | None = None,
    check_nodes: bool = False,
) -> ComfyProbe:
    """Check whether ComfyUI is reachable and read GPU/VRAM summary from ``/system_stats``."""
    try:
        normalized = validate_http_url(base_url, "ComfyUI URL")
    except ConfigError as exc:
        return ComfyProbe(url=base_url, reachable=False, error=str(exc))

    env = os.environ if environ is None else environ
    try:
        config = ComfyConfig.model_validate(
            {"base_url": normalized, "headers": headers or {}, "request_timeout": timeout},
            context={"environ": env},
        )
    except (ConfigError, ValidationError) as exc:
        return ComfyProbe(url=normalized, reachable=False, error=str(exc))
    node_count: int | None = None
    try:
        with ComfyClient(config, transport=transport) as client:
            stats = client.ping()
            if check_nodes:
                try:
                    info = client.object_info()
                except (ComfyConnectionError, ComfyJobError, ValueError):
                    info = None
                if isinstance(info, dict) and info:
                    node_count = len(info)
    except (ComfyConnectionError, ComfyJobError, ValueError) as exc:
        return ComfyProbe(url=normalized, reachable=False, error=str(exc))

    devices: list[str] = []
    max_total: float | None = None
    max_free: float | None = None
    comfy_version: str | None = None

    if isinstance(stats, dict):
        system = stats.get("system")
        if isinstance(system, dict) and isinstance(system.get("comfyui_version"), str):
            comfy_version = system["comfyui_version"]
        raw_devices = stats.get("devices")
        if isinstance(raw_devices, list):
            for dev in raw_devices:
                if not isinstance(dev, dict):
                    continue
                name = dev.get("name")
                if isinstance(name, str) and name.strip():
                    devices.append(name.strip())
                total = dev.get("vram_total")
                if (
                    isinstance(total, (int, float))
                    and not isinstance(total, bool)
                    and math.isfinite(total)
                    and total > 0
                ):
                    total_gb = total / (1024.0**3)
                    max_total = total_gb if max_total is None else max(max_total, total_gb)
                free = dev.get("vram_free")
                if (
                    isinstance(free, (int, float))
                    and not isinstance(free, bool)
                    and math.isfinite(free)
                    and free >= 0
                ):
                    free_gb = free / (1024.0**3)
                    max_free = free_gb if max_free is None else max(max_free, free_gb)

    return ComfyProbe(
        url=normalized,
        reachable=True,
        devices=devices,
        vram_total_gb=max_total,
        vram_free_gb=max_free,
        comfy_version=comfy_version,
        node_count=node_count,
    )


def inspect_environment(
    *,
    comfy_url: str | None = None,
    comfy_token_env: str | None = None,
    ollama_url: str | None = None,
    comfy_transport: httpx.BaseTransport | None = None,
    ollama_transport: httpx.BaseTransport | None = None,
    environ: Mapping[str, str] | None = None,
    project_dir: Path | None = None,
    check_nodes: bool = False,
) -> EnvironmentReport:
    """Inspect local tools, Python extras, Ollama, and ComfyUI readiness."""
    if environ is not None:
        env: Mapping[str, str] = environ
    elif project_dir is not None and project_dir.is_dir():
        env = merged_project_environ(project_dir)
    else:
        env = os.environ
    resolved_comfy = comfy_url or env.get("COMFY_URL", "").strip() or DEFAULT_COMFY_URL
    resolved_ollama = ollama_url or env.get("MVD_LLM_URL", "").strip() or DEFAULT_OLLAMA_URL
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    plat = f"{platform.system()} {platform.machine()}".strip()
    comfy_headers = _comfy_probe_headers(comfy_token_env, environ=env)
    return EnvironmentReport(
        python_version=py_ver,
        platform_label=plat,
        ffmpeg_ok=shutil.which("ffmpeg") is not None,
        ffprobe_ok=shutil.which("ffprobe") is not None,
        extras=check_extras(),
        ollama=probe_ollama(resolved_ollama, transport=ollama_transport),
        comfy=probe_comfy(
            resolved_comfy,
            headers=comfy_headers,
            transport=comfy_transport,
            environ=env,
            check_nodes=check_nodes,
        ),
    )


def is_ollama_installed() -> bool:
    """Return True when the ``ollama`` CLI is on ``PATH``."""
    return shutil.which("ollama") is not None


def ollama_install_command(system: str | None = None) -> list[str] | None:
    """Return a safe argv-list install command for the host OS package manager, or None."""
    sys_name = system or platform.system()
    if sys_name == "Darwin" and shutil.which("brew"):
        return ["brew", "install", "ollama"]
    if sys_name == "Windows" and shutil.which("winget"):
        return [
            "winget",
            "install",
            "--id",
            "Ollama.Ollama",
            "-e",
            "--accept-package-agreements",
            "--accept-source-agreements",
        ]
    return None


def install_ollama(*, command: Sequence[str] | None = None) -> tuple[bool, str]:
    """Run the OS package manager to install Ollama (no shell execution)."""
    cmd = list(command) if command is not None else ollama_install_command()
    if not cmd:
        return (
            False,
            "No supported automatic installer found for this OS. "
            "Install Ollama from https://ollama.com/download and re-run `mvd setup`.",
        )
    try:
        proc = subprocess.run(cmd, check=False)
    except OSError as exc:
        return False, f"Failed to run {' '.join(cmd)}: {exc}"
    if proc.returncode != 0:
        return False, f"`{' '.join(cmd)}` exited with status {proc.returncode}."
    return True, f"Installed Ollama via `{' '.join(cmd)}`."


def start_ollama_server(
    base_url: str = DEFAULT_OLLAMA_URL,
    *,
    wait_seconds: float = 10.0,
    transport: httpx.BaseTransport | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> OllamaProbe:
    """Start local ``ollama serve`` in the background and wait until ``/api/tags`` responds."""
    try:
        normalized = validate_http_url(base_url, "Ollama URL")
    except ConfigError as exc:
        return OllamaProbe(url=base_url, reachable=False, error=str(exc))
    if not is_loopback_url(normalized):
        return OllamaProbe(
            url=normalized,
            reachable=False,
            error="Cannot auto-start a remote Ollama server; start it on that host.",
        )
    ollama_bin = shutil.which("ollama")
    if not ollama_bin:
        return OllamaProbe(
            url=normalized, reachable=False, error="Ollama binary not found on PATH."
        )
    try:
        subprocess.Popen(
            [ollama_bin, "serve"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError as exc:
        return OllamaProbe(url=normalized, reachable=False, error=f"Failed to start Ollama: {exc}")

    deadline = time.monotonic() + max(0.5, wait_seconds)
    last_probe = OllamaProbe(url=normalized, reachable=False, error="Timed out starting Ollama")
    while True:
        last_probe = probe_ollama(normalized, timeout=2.0, transport=transport)
        if last_probe.reachable or time.monotonic() >= deadline:
            return last_probe
        sleep(0.5)


def pull_ollama_model(
    model: str,
    *,
    base_url: str = DEFAULT_OLLAMA_URL,
    transport: httpx.BaseTransport | None = None,
) -> tuple[bool, str]:
    """Pull an Ollama model tag safely via the local CLI or remote ``/api/pull`` endpoint."""
    cleaned = model.strip()
    if not _SAFE_OLLAMA_MODEL_RE.match(cleaned):
        return False, f"Invalid Ollama model name: {model!r}"
    try:
        normalized = validate_http_url(base_url, "Ollama URL")
    except ConfigError as exc:
        return False, str(exc)

    ollama_bin = shutil.which("ollama")
    if is_loopback_url(normalized) and ollama_bin and transport is None:
        try:
            proc = subprocess.run([ollama_bin, "pull", cleaned], check=False)
        except OSError as exc:
            return False, f"Failed to run `ollama pull {cleaned}`: {exc}"
        if proc.returncode != 0:
            return False, f"`ollama pull {cleaned}` exited with status {proc.returncode}."
        return True, f"Pulled Ollama model '{cleaned}'."

    safe_url = redact_url(normalized)
    try:
        with httpx.Client(timeout=600.0, transport=transport) as client:
            resp = client.post(
                f"{normalized}/api/pull",
                json={"name": cleaned, "stream": False},
            )
    except httpx.HTTPError as exc:
        return False, f"Could not pull '{cleaned}' from {safe_url}: {exc}"
    if resp.status_code >= 400:
        return False, f"Ollama /api/pull returned HTTP {resp.status_code}: {resp.text[:200]}"
    try:
        body = resp.json()
    except ValueError:
        body = None
    if isinstance(body, dict) and body.get("error"):
        return False, f"Ollama /api/pull failed for '{cleaned}': {str(body['error'])[:200]}"
    return True, f"Pulled Ollama model '{cleaned}' on {safe_url}."


def verify_llm_config(
    config: LLMConfig,
    *,
    check_vision: bool = True,
    environ: Mapping[str, str] | None = None,
    transport: httpx.BaseTransport | None = None,
    sleep: Callable[[float], None] | None = None,
) -> LLMCheckResult:
    """Run a live JSON (+ optional vision) sanity check against ``config``."""
    if not config.enabled:
        return LLMCheckResult(json_ok=True, vision="skipped")
    try:
        client = build_llm(config, environ=environ, transport=transport, sleep=sleep)
    except (ConfigError, DependencyMissing) as exc:
        msg = str(exc)
        kind: ErrorKind = "auth" if "api key" in msg.lower() else "error"
        return LLMCheckResult(json_ok=False, json_detail=msg, json_error_kind=kind)
    if client is None:
        return LLMCheckResult(json_ok=True, vision="skipped")
    try:
        return run_llm_check(client, check_vision=check_vision)
    finally:
        closer = getattr(client, "close", None)
        if callable(closer):
            closer()


def _active_workflow_names(project: Project) -> list[str]:
    if project.config.mode is ProjectMode.MIX:
        return ["t2i"] if project.config.mix.visual_source == "stills" else ["t2i", "i2v"]
    if project.config.planning.video_style is VideoStyle.VISUAL:
        return ["t2i", "i2v"]
    if project.config.planning.video_style is VideoStyle.NARRATIVE:
        return ["t2i", "still", "i2v"]
    return list(project.config.comfy.workflows)


def check_comfy_wiring(
    project_dir: Path,
    *,
    comfy_transport: httpx.BaseTransport | None = None,
) -> ComfyWiringReport:
    """Verify ComfyUI reachability, ``/object_info``, and any exported workflow bindings."""
    project = Project.open(project_dir)
    comfy_cfg = project.config.comfy
    url = redact_url(comfy_cfg.base_url)
    env = project.environ()

    workflows_ok: list[str] = []
    missing_exports: list[str] = []
    issues: list[tuple[str, str]] = []
    loaded_templates: dict[str, WorkflowTemplate] = {}

    for name in _active_workflow_names(project):
        if name not in comfy_cfg.workflows:
            issues.append((f"workflow '{name}'", f"comfy.workflows has no entry for '{name}'"))
            continue
        bindings_path = project.path(Path(comfy_cfg.workflows_dir) / comfy_cfg.workflows[name])
        if not bindings_path.is_file():
            issues.append((f"workflow '{name}'", f"Bindings file not found: {bindings_path}"))
            continue
        try:
            tpl = WorkflowTemplate.load(name, bindings_path, root=project.root, environ=env)
        except MVDError as exc:
            msg = str(exc)
            if "which does not exist" in msg:
                missing_exports.append(name)
            else:
                issues.append((f"workflow '{name}'", msg))
        else:
            loaded_templates[name] = tpl
            workflows_ok.append(name)

    reachable = False
    node_count: int | None = None
    comfy_error: str | None = None
    try:
        with ComfyClient(
            comfy_cfg, transport=comfy_transport, upload_subfolder=project.comfy_subfolder
        ) as comfy:
            comfy.ping()
            reachable = True
            try:
                info = comfy.object_info()
            except MVDError:
                info = {}
            if isinstance(info, dict) and info:
                node_count = len(info)
                for wf_name, tpl in loaded_templates.items():
                    bound_targets = {
                        (str(t.node), t.field)
                        for targets in tpl.bindings.inputs.values()
                        for t in targets
                    }
                    live_issues = comfy.validate_graph_live(
                        tpl.graph, object_info=info, skip_inputs=bound_targets
                    )
                    if live_issues:
                        if wf_name in workflows_ok:
                            workflows_ok.remove(wf_name)
                        issues.append(
                            (f"ComfyUI dependencies for '{wf_name}'", "; ".join(live_issues[:5]))
                        )
    except MVDError as exc:
        comfy_error = str(exc)

    return ComfyWiringReport(
        url=url,
        reachable=reachable,
        node_count=node_count,
        comfy_error=comfy_error,
        workflows_ok=workflows_ok,
        missing_exports=missing_exports,
        issues=issues,
    )


def pick_default_ollama_model(
    installed_models: Sequence[str], preferred: str = DEFAULT_LLM_MODEL
) -> str:
    """Choose the best default Ollama model from locally installed tags."""
    if preferred in installed_models:
        return preferred
    for model in installed_models:
        lowered = model.lower()
        if any(lowered.startswith(prefix) for prefix in _VISION_MODEL_PREFIXES):
            return model
    return preferred


def validate_setup_options(options: SetupOptions) -> None:
    """Validate ``SetupOptions`` before creating or modifying any files on disk."""
    if options.title is not None and not options.title.strip():
        raise ConfigError("Project title must not be empty.")
    if options.comfy_url is not None:
        validate_http_url(options.comfy_url, "ComfyUI URL")
    if options.comfy_token_env:
        validate_comfy_token_env(options.comfy_token_env)
    if options.llm_model is not None and not options.llm_model.strip():
        raise ConfigError("llm.model must not be empty.")
    if options.llm_base_url:
        validate_http_url(options.llm_base_url, "LLM base URL")
    if options.llm_api_key_env and not is_allowed_api_key_env(options.llm_api_key_env):
        raise ConfigError(
            f"llm.api_key_env {options.llm_api_key_env!r} is not allowed. "
            "Use a standard provider key name, an MVD_* name, or list it in MVD_ALLOWED_ENV_VARS."
        )
    if options.video_preset is not None and options.video_preset not in VIDEO_PRESETS:
        valid = ", ".join(VIDEO_PRESETS)
        raise ConfigError(f"Unknown video preset {options.video_preset!r}. Choose from: {valid}")


def _replace_section_field_block(
    text: str, section: str, field_name: str, formatted_val: str
) -> str:
    """Replace a field under ``section`` even if it previously held a multi-line YAML block."""
    lines = text.splitlines(keepends=True)
    sec_start: int | None = None
    sec_end = len(lines)
    top_header_re = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*):[ \t]*(?:#.*)?$")
    top_key_re = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*):")
    for idx, raw in enumerate(lines):
        stripped_line = raw.rstrip("\r\n")
        if sec_start is None:
            m = top_header_re.match(stripped_line)
            if m and m.group(1) == section:
                sec_start = idx
        elif top_key_re.match(stripped_line):
            sec_end = idx
            break

    if sec_start is not None:
        field_re = re.compile(
            rf"^([ \t]+){re.escape(field_name)}:[ \t]*([^#\r\n]*?)([ \t]*#[^\r\n]*)?(\r?\n?)$"
        )
        for idx in range(sec_start + 1, sec_end):
            m = field_re.match(lines[idx])
            if m:
                indent, _old_val, comment, newline = m.groups()
                newline = newline or "\n"
                comment_part = ""
                if comment:
                    gap = comment[: len(comment) - len(comment.lstrip())] or "  "
                    comment_part = f"{gap}{comment.lstrip()}"
                # Remove any nested child lines belonging to a prior multi-line mapping/list.
                drop_end = idx + 1
                while drop_end < sec_end:
                    nxt = lines[drop_end]
                    if not nxt.strip():
                        break
                    nxt_indent = nxt[: len(nxt) - len(nxt.lstrip(" \t"))]
                    if len(nxt_indent) > len(indent):
                        drop_end += 1
                    else:
                        break
                replacement = f"{indent}{field_name}: {formatted_val}{comment_part}{newline}"
                lines[idx:drop_end] = [replacement]
                return "".join(lines)

    return update_yaml_scalar(text, section, field_name, formatted_val)


def apply_setup_config(project_dir: Path, options: SetupOptions) -> Path:
    """Apply setup selections to ``project.yaml`` in place, preserving comments and validating."""
    validate_setup_options(options)
    yaml_path = project_dir / CONFIG_FILENAME
    if not yaml_path.is_file():
        raise ConfigError(f"Config file not found: {yaml_path}")

    text = yaml_path.read_text(encoding="utf-8")
    try:
        parsed_before = yaml.safe_load(text) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"{yaml_path} is not valid YAML: {exc}") from exc
    if not isinstance(parsed_before, dict):
        raise ConfigError(f"{yaml_path} must hold a YAML mapping at top level.")

    updated_text = text

    # Top-level scalars
    if options.title is not None:
        updated_text = update_top_level_yaml_scalar(
            updated_text, "title", format_yaml_scalar(options.title.strip())
        )
    if options.mode is not None:
        updated_text = update_top_level_yaml_scalar(
            updated_text, "mode", format_yaml_scalar(options.mode.value)
        )

    # Section updates: (section, field_name, formatted_yaml, replace_block)
    formatted_updates: list[tuple[str, str, str, bool]] = []
    if options.comfy_url is not None:
        normalized_comfy = validate_http_url(options.comfy_url, "ComfyUI URL")
        formatted_updates.append(("comfy", "base_url", format_yaml_scalar(normalized_comfy), False))
    if options.comfy_token_env is not None:
        if options.comfy_token_env == "":
            formatted_updates.append(("comfy", "headers", "{}", True))
        else:
            token_env = validate_comfy_token_env(options.comfy_token_env)
            formatted_updates.append(
                ("comfy", "headers", f'{{Authorization: "Bearer ${{{token_env}}}"}}', True)
            )

    if options.llm_enabled is not None:
        formatted_updates.append(("llm", "enabled", format_yaml_scalar(options.llm_enabled), False))
    if options.llm_provider is not None:
        formatted_updates.append(
            ("llm", "provider", format_yaml_scalar(options.llm_provider), False)
        )
    if options.llm_model is not None:
        formatted_updates.append(
            ("llm", "model", format_yaml_scalar(options.llm_model.strip()), False)
        )
    if options.llm_base_url is not None:
        if options.llm_base_url == "":
            formatted_updates.append(("llm", "base_url", format_yaml_scalar(None), False))
        else:
            normalized_llm = validate_http_url(options.llm_base_url, "LLM base URL")
            formatted_updates.append(("llm", "base_url", format_yaml_scalar(normalized_llm), False))
    if options.llm_api_key_env is not None:
        if options.llm_api_key_env == "":
            formatted_updates.append(("llm", "api_key_env", format_yaml_scalar(None), False))
        else:
            formatted_updates.append(
                ("llm", "api_key_env", format_yaml_scalar(options.llm_api_key_env.strip()), False)
            )

    if options.qc_on_missing_dependency is not None:
        formatted_updates.append(
            (
                "qc",
                "on_missing_dependency",
                format_yaml_scalar(options.qc_on_missing_dependency),
                False,
            )
        )
    if options.qc_vision_enabled is not None:
        formatted_updates.append(
            ("qc", "vision_enabled", format_yaml_scalar(options.qc_vision_enabled), False)
        )

    if options.video_preset is not None:
        preset = VIDEO_PRESETS[options.video_preset]
        if preset.frame_multiple is not None:
            formatted_updates.append(
                ("video", "frame_multiple", format_yaml_scalar(preset.frame_multiple), False)
            )
        if preset.frame_offset is not None:
            formatted_updates.append(
                ("video", "frame_offset", format_yaml_scalar(preset.frame_offset), False)
            )
        if preset.size_multiple is not None:
            formatted_updates.append(
                ("video", "size_multiple", format_yaml_scalar(preset.size_multiple), False)
            )
        if preset.max_clip_seconds is not None:
            formatted_updates.append(
                ("video", "max_clip_seconds", format_yaml_scalar(preset.max_clip_seconds), False)
            )

    for section, field_name, formatted_val, replace_block in formatted_updates:
        if (
            isinstance(parsed_before.get(section), dict) is False
            and parsed_before.get(section) is not None
        ):
            raise ConfigError(f"Section '{section}' in {yaml_path} must be a mapping or null.")
        sec_empty_re = re.compile(
            rf"^{re.escape(section)}:[ \t]*(?:null|~|\{{[ \t]*\}})([ \t]*#[^\r\n]*)?(\r?)$",
            re.MULTILINE,
        )
        updated_text = sec_empty_re.sub(rf"{section}:\1\2", updated_text)
        if replace_block:
            updated_text = _replace_section_field_block(
                updated_text, section, field_name, formatted_val
            )
        else:
            updated_text = update_yaml_scalar(updated_text, section, field_name, formatted_val)

    try:
        parsed_after = yaml.safe_load(updated_text) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"Updated {yaml_path} failed YAML validation: {exc}") from exc
    if not isinstance(parsed_after, dict):
        raise ConfigError(f"Updated {yaml_path} must hold a YAML mapping at top level.")

    effective_env = merged_project_environ(project_dir)
    to_validate = mask_unset_allowed_env_refs(parsed_after, environ=effective_env)
    try:
        ProjectConfig.model_validate(to_validate, context={"environ": effective_env})
    except ValidationError as exc:
        raise ConfigError(f"Invalid configuration after setup in {yaml_path}:\n{exc}") from exc

    atomic_write(yaml_path, updated_text)
    return yaml_path


def render_environment_report(report: EnvironmentReport) -> str:
    """Format a concise human-readable summary of local environment checks."""
    lines = [
        f"Environment: Python {report.python_version} on {report.platform_label}",
        "",
        "System tools:",
        f"  {'OK  ' if report.ffmpeg_ok else 'MISS'} ffmpeg",
        f"  {'OK  ' if report.ffprobe_ok else 'MISS'} ffprobe",
    ]
    if not report.tools_ok:
        lines.append(
            "  -> Install FFmpeg: `brew install ffmpeg` (macOS), "
            "`sudo apt install ffmpeg` (Ubuntu/Debian), or "
            "`winget install Gyan.FFmpeg` (Windows)."
        )

    lines.append("")
    lines.append("Python extras:")
    for extra in report.extras:
        tag = "OK  " if extra.installed else "----"
        lines.append(f"  {tag} {extra.name:<10} {extra.purpose}")
    if not report.extra_installed("qc") or not report.extra_installed("align"):
        lines.append(
            '  -> Recommended local extras: `pip install -c constraints.txt -e ".[qc,align,ui]"`'
        )

    lines.append("")
    lines.append("Services:")
    ollama_display_url = redact_url(report.ollama.url)
    comfy_display_url = redact_url(report.comfy.url)
    if report.ollama.reachable:
        if report.ollama.models:
            shown = ", ".join(report.ollama.models[:6])
            more_count = len(report.ollama.models) - 6
            extra_n = f" (+{more_count} more)" if more_count > 0 else ""
            lines.append(f"  OK   Ollama at {ollama_display_url} (models: {shown}{extra_n})")
        else:
            lines.append(
                f"  WARN Ollama at {ollama_display_url} is running, but no models are pulled."
            )
            lines.append(f"       -> Run: `ollama pull {DEFAULT_LLM_MODEL}`")
    else:
        lines.append(f"  MISS Ollama at {ollama_display_url}: {report.ollama.error}")
        lines.append(
            "       -> Install Ollama (https://ollama.com), start `ollama serve`, "
            f"then `ollama pull {DEFAULT_LLM_MODEL}` — or point at a remote/cloud LLM "
            "with `--ollama-url` or `--llm-provider`."
        )

    if report.comfy.reachable:
        details: list[str] = []
        if report.comfy.comfy_version:
            cleaned_ver = report.comfy.comfy_version.strip().lstrip("v")
            if cleaned_ver:
                details.append(f"v{cleaned_ver}")
        if report.comfy.devices:
            details.append(report.comfy.devices[0])
        if report.comfy.vram_total_gb is not None:
            if report.comfy.vram_free_gb is not None:
                vram_free = report.comfy.vram_free_gb
                vram_total = report.comfy.vram_total_gb
                details.append(f"VRAM {vram_free:.1f}/{vram_total:.1f} GiB free")
            else:
                details.append(f"VRAM {report.comfy.vram_total_gb:.1f} GiB")
        if report.comfy.node_count is not None:
            details.append(f"{report.comfy.node_count} nodes")
        suffix = f" ({', '.join(details)})" if details else ""
        lines.append(f"  OK   ComfyUI at {comfy_display_url}{suffix}")
    else:
        lines.append(f"  MISS ComfyUI at {comfy_display_url}: {report.comfy.error}")
        lines.append(
            "       -> Start ComfyUI locally on port 8188, or point at a home-LAN/cloud server "
            "with `--comfy-url <url>` (and `--comfy-token-env COMFY_TOKEN` if authenticated)."
        )

    return "\n".join(lines)


__all__ = [
    "DEFAULT_COMFY_URL",
    "DEFAULT_LLM_MODEL",
    "DEFAULT_LLM_PROVIDER",
    "DEFAULT_OLLAMA_URL",
    "PROVIDER_ALIASES",
    "PROVIDER_MODEL_EXAMPLES",
    "VIDEO_PRESETS",
    "ComfyProbe",
    "ComfyWiringReport",
    "EnvironmentReport",
    "ExtraStatus",
    "OllamaProbe",
    "SetupOptions",
    "VideoPreset",
    "apply_setup_config",
    "check_comfy_wiring",
    "check_extras",
    "inspect_environment",
    "install_ollama",
    "is_loopback_url",
    "is_ollama_installed",
    "normalize_llm_provider",
    "ollama_install_command",
    "pick_default_ollama_model",
    "probe_comfy",
    "probe_ollama",
    "provider_key_env",
    "pull_ollama_model",
    "render_environment_report",
    "start_ollama_server",
    "validate_comfy_token_env",
    "validate_http_url",
    "validate_setup_options",
    "verify_llm_config",
    "write_project_dotenv_var",
]
