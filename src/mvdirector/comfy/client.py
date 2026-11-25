# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Minimal ComfyUI HTTP client: upload inputs, queue a workflow, wait, download outputs.

Polling ``/history`` is used instead of the websocket because it survives proxies
(such as hosted GPU proxies) and reconnects without extra state.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import math
import os
import re
import socket
import ssl
import struct
import threading
import time
import urllib.parse
import uuid
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from mvdirector import __version__
from mvdirector.comfy.cache import file_sha256
from mvdirector.comfy.workflow import OutputSpec
from mvdirector.config import ComfyConfig
from mvdirector.errors import ComfyConnectionError, ComfyJobError, ComfyResourceError, ConfigError
from mvdirector.llm.http import redact_url, retry_delay, validate_http_base_url

log = logging.getLogger(__name__)

_TRANSIENT_RETRIES = 3
_MAX_DOWNLOAD_BYTES = 512 * 1024 * 1024  # 512 MiB cap per generated asset
#: How many consecutive polls to allow a history entry that reports neither a status nor any
#: outputs before handing it back, so the caller raises the accurate "produced no output" error.
_STATUSLESS_GRACE_POLLS = 3
# /free only flags ComfyUI's worker thread to unload; give it this long to release VRAM.
_VRAM_SETTLE_POLLS = 30
_VRAM_POLL_SECONDS = 1.0
_ALLOWED_OUTPUT_TYPES = frozenset({"output", "temp", "input"})
_MAX_WS_FRAME_BYTES = 4 * 1024 * 1024  # 4 MiB safety cap for ComfyUI WebSocket frames
_WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
_WS_HANDSHAKE_TIMEOUT_SECONDS = 5.0
_WS_MAX_HANDSHAKE_BYTES = 16384
_WS_FORBIDDEN_HEADERS = frozenset(
    {
        "host",
        "upgrade",
        "connection",
        "sec-websocket-key",
        "sec-websocket-version",
        "sec-websocket-extensions",
        "sec-websocket-accept",
    }
)
_WS_HEADER_NAME_RE = re.compile(r"^[!#$%&'*+\-.^_`|~0-9A-Za-z]+\Z")
_WS_MAX_CONTROL_FRAME_BYTES = 125


def _finite_number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return float(value)
    return None


def _reject_json_constant(token: str) -> Any:
    raise ValueError(f"Non-standard JSON constant {token!r}")


@dataclass(frozen=True, slots=True)
class ComfyProgress:
    """Snapshot of a single ComfyUI prompt's live execution progress."""

    prompt_id: str
    state: str  # "queued" | "running" | "downloading" | "done"
    step: int | None = None
    total_steps: int | None = None
    percent: float | None = None
    node: str | None = None
    queue_position: int | None = None
    elapsed_seconds: float = 0.0


def parse_comfy_ws_event(
    payload: dict[str, Any],
    prompt_id: str,
    *,
    elapsed_seconds: float = 0.0,
) -> ComfyProgress | None:
    """Convert a ComfyUI WebSocket JSON message into a ``ComfyProgress`` if it matches."""
    if not isinstance(payload, dict):
        return None
    event_type = payload.get("type")
    data = payload.get("data")
    if not isinstance(data, dict):
        return None
    msg_prompt = data.get("prompt_id")
    if msg_prompt is not None and msg_prompt != prompt_id:
        return None
    safe_elapsed_raw = _finite_number(elapsed_seconds)
    safe_elapsed = round(max(0.0, safe_elapsed_raw), 1) if safe_elapsed_raw is not None else 0.0

    if event_type == "progress":
        val = _finite_number(data.get("value"))
        mx = _finite_number(data.get("max"))
        if val is not None and mx is not None and val >= 0.0 and mx > 0.0:
            pct = round(min(100.0, max(0.0, (val / mx) * 100.0)), 1)
            raw_node = data.get("node")
            return ComfyProgress(
                prompt_id=prompt_id,
                state="running",
                step=int(val),
                total_steps=int(mx),
                percent=pct,
                node=str(raw_node) if raw_node is not None else None,
                elapsed_seconds=safe_elapsed,
            )
        return None

    if event_type == "progress_state":
        nodes = data.get("nodes")
        if not isinstance(nodes, dict):
            return None
        total_val = 0.0
        total_max = 0.0
        active_node: str | None = None
        for node_id, info in nodes.items():
            if not isinstance(info, dict):
                continue
            v = _finite_number(info.get("value"))
            m = _finite_number(info.get("max"))
            if v is not None and m is not None and v >= 0.0 and m > 0.0:
                total_val += min(m, v)
                total_max += m
                if info.get("state") == "running" or active_node is None:
                    active_node = str(node_id)
        if math.isfinite(total_val) and math.isfinite(total_max) and total_max > 0.0:
            pct = round(min(100.0, max(0.0, (total_val / total_max) * 100.0)), 1)
            return ComfyProgress(
                prompt_id=prompt_id,
                state="running",
                step=int(total_val),
                total_steps=int(total_max),
                percent=pct,
                node=active_node,
                elapsed_seconds=safe_elapsed,
            )
        return None

    if event_type == "executing":
        raw_node = data.get("node")
        if raw_node is None:
            # Require explicit prompt_id so idle/connect broadcasts cannot mark queued jobs done.
            if msg_prompt != prompt_id:
                return None
            return ComfyProgress(
                prompt_id=prompt_id,
                state="downloading",
                percent=99.0,
                elapsed_seconds=safe_elapsed,
            )
        return ComfyProgress(
            prompt_id=prompt_id,
            state="running",
            node=str(raw_node),
            elapsed_seconds=safe_elapsed,
        )

    if event_type == "execution_start":
        if msg_prompt != prompt_id:
            return None
        return ComfyProgress(
            prompt_id=prompt_id,
            state="running",
            elapsed_seconds=safe_elapsed,
        )

    return None


class _ComfyWsProgressListener:
    """Best-effort RFC 6455 WebSocket listener for live ComfyUI step progress.

    Uses only the Python standard library so ``mvdirector`` adds zero runtime dependencies.
    Any connection or protocol error is silently swallowed so HTTP ``/history`` polling
    remains the authoritative completion path.
    """

    def __init__(
        self,
        base_url: str,
        client_id: str,
        headers: dict[str, str],
        prompt_id: str,
        started: float,
        on_progress: Callable[[ComfyProgress], None],
    ) -> None:
        self._base_url = base_url
        self._client_id = client_id
        self._headers = headers
        self._prompt_id = prompt_id
        self._started = started
        self._on_progress = on_progress
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._sock: socket.socket | None = None
        self._connected = False
        self._last_ws_event_at: float | None = None
        self._thread: threading.Thread | None = None

    @property
    def connected(self) -> bool:
        with self._lock:
            return self._connected

    @property
    def last_event_at(self) -> float | None:
        with self._lock:
            return self._last_ws_event_at

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run,
            name=f"comfy-ws-{self._prompt_id[:8]}",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        with self._lock:
            self._stop.set()
            sock = self._sock
            self._sock = None
        if sock is not None:
            with suppress(OSError):
                sock.shutdown(socket.SHUT_RDWR)
            with suppress(OSError):
                sock.close()
        if self._thread is not None:
            self._thread.join(timeout=1.5)

    def _register_socket(self, sock: socket.socket) -> bool:
        with self._lock:
            if self._stop.is_set():
                return False
            self._sock = sock
            return True

    def _run(self) -> None:
        sock: socket.socket | None = None
        try:
            parsed = urllib.parse.urlparse(self._base_url)
            if parsed.scheme not in ("http", "https") or not parsed.hostname:
                return
            use_tls = parsed.scheme == "https"
            port = parsed.port or (443 if use_tls else 80)
            if self._stop.is_set():
                return
            raw_sock = socket.create_connection((parsed.hostname, port), timeout=3.0)
            sock = raw_sock
            if use_tls:
                ctx = ssl.create_default_context()
                ctx.minimum_version = ssl.TLSVersion.TLSv1_2
                sock = ctx.wrap_socket(raw_sock, server_hostname=parsed.hostname)
            if not self._register_socket(sock):
                return
            sock.settimeout(_WS_HANDSHAKE_TIMEOUT_SECONDS)

            base_path = parsed.path.rstrip("/")
            ws_path = f"{base_path}/ws?clientId={urllib.parse.quote(self._client_id, safe='')}"
            host_part = f"[{parsed.hostname}]" if ":" in parsed.hostname else parsed.hostname
            host_header = f"{host_part}:{parsed.port}" if parsed.port else host_part
            ws_key = base64.b64encode(os.urandom(16)).decode("ascii")
            req_lines = [
                f"GET {ws_path} HTTP/1.1",
                f"Host: {host_header}",
                "Upgrade: websocket",
                "Connection: Upgrade",
                f"Sec-WebSocket-Key: {ws_key}",
                "Sec-WebSocket-Version: 13",
            ]
            for k, v in self._headers.items():
                if (
                    not _WS_HEADER_NAME_RE.match(k)
                    or _has_control_chars(v)
                    or k.lower() in _WS_FORBIDDEN_HEADERS
                ):
                    continue
                req_lines.append(f"{k}: {v}")
            req_lines.extend(["", ""])
            sock.sendall("\r\n".join(req_lines).encode("utf-8"))
            sock.settimeout(0.1)

            # Read HTTP 101 upgrade response headers (capped at 16 KiB and bounded in time).
            resp_buf = bytearray()
            handshake_deadline = time.monotonic() + _WS_HANDSHAKE_TIMEOUT_SECONDS
            while b"\r\n\r\n" not in resp_buf:
                if self._stop.is_set() or time.monotonic() > handshake_deadline:
                    return
                try:
                    chunk = sock.recv(1024)
                except TimeoutError:
                    continue
                if not chunk:
                    return
                resp_buf.extend(chunk)
                if len(resp_buf) > _WS_MAX_HANDSHAKE_BYTES:
                    return
            header_end = resp_buf.index(b"\r\n\r\n")
            header_lines = resp_buf[:header_end].split(b"\r\n")
            status_parts = header_lines[0].split(b" ", 2)
            if (
                len(status_parts) < 2
                or not status_parts[0].startswith(b"HTTP/1.")
                or status_parts[1] != b"101"
            ):
                return
            expected_accept = base64.b64encode(
                hashlib.sha1((ws_key + _WS_GUID).encode("ascii")).digest()
            ).decode("ascii")
            resp_headers: dict[str, str] = {}
            for raw_line in header_lines[1:]:
                if b":" in raw_line:
                    hk, hv = raw_line.split(b":", 1)
                    key_lower = hk.decode("latin1").strip().lower()
                    val = hv.decode("latin1").strip()
                    if key_lower == "connection" and key_lower in resp_headers:
                        resp_headers[key_lower] = f"{resp_headers[key_lower]}, {val}"
                    else:
                        resp_headers[key_lower] = val
            conn_tokens = {
                tok.strip().lower() for tok in resp_headers.get("connection", "").split(",")
            }
            if (
                resp_headers.get("sec-websocket-accept") != expected_accept
                or resp_headers.get("upgrade", "").lower() != "websocket"
                or "upgrade" not in conn_tokens
            ):
                return

            with self._lock:
                self._connected = True
            leftover = bytes(resp_buf[header_end + 4 :])
            self._read_loop(sock, leftover)
        except Exception as exc:  # noqa: BLE001 - best-effort progress listener
            log.debug("ComfyUI WebSocket progress listener unavailable: %s", exc)
        finally:
            with self._lock:
                self._connected = False
                active = self._sock
                self._sock = None
            for handle in (active, sock):
                if handle is not None:
                    with suppress(OSError):
                        handle.close()

    def _emit_text_payload(self, payload: bytes) -> None:
        if self._stop.is_set():
            return
        try:
            msg = json.loads(payload.decode("utf-8"), parse_constant=_reject_json_constant)
        except (UnicodeDecodeError, ValueError, RecursionError):
            return
        if not isinstance(msg, dict):
            return
        progress = parse_comfy_ws_event(
            msg,
            self._prompt_id,
            elapsed_seconds=time.monotonic() - self._started,
        )
        if progress is not None and not self._stop.is_set():
            with self._lock:
                self._last_ws_event_at = time.monotonic()
            try:
                self._on_progress(progress)
            except Exception:  # noqa: BLE001 - observer errors must not break polling
                log.debug("Progress callback error ignored", exc_info=True)

    def _read_loop(self, sock: socket.socket, initial: bytes) -> None:
        buf = bytearray(initial)
        frag_buf: bytearray | None = None

        def read_bytes(n: int) -> bytes | None:
            while len(buf) < n:
                if self._stop.is_set():
                    return None
                try:
                    chunk = sock.recv(max(4096, n - len(buf)))
                except TimeoutError:
                    continue
                if not chunk:
                    return None
                buf.extend(chunk)
            out = bytes(buf[:n])
            del buf[:n]
            return out

        while not self._stop.is_set():
            hdr = read_bytes(2)
            if hdr is None:
                return
            fin = bool(hdr[0] & 0x80)
            opcode = hdr[0] & 0x0F
            masked = bool(hdr[1] & 0x80)
            length = hdr[1] & 0x7F
            if length == 126:
                ext = read_bytes(2)
                if ext is None:
                    return
                length = struct.unpack("!H", ext)[0]
            elif length == 127:
                ext = read_bytes(8)
                if ext is None:
                    return
                length = struct.unpack("!Q", ext)[0]
            if length > _MAX_WS_FRAME_BYTES:
                return
            # RFC 6455 §5.5: control frames must not be fragmented and have payload <= 125 bytes.
            if opcode >= 0x8 and (not fin or length > _WS_MAX_CONTROL_FRAME_BYTES):
                return
            mask_key = read_bytes(4) if masked else None
            if masked and mask_key is None:
                return
            payload = read_bytes(length) if length > 0 else b""
            if payload is None:
                return

            if opcode == 0x8:  # close
                return
            if opcode == 0x9:  # ping -> send masked pong (allowed mid-fragment per RFC 6455)
                if mask_key is not None and payload:
                    payload = bytes(b ^ mask_key[i % 4] for i, b in enumerate(payload[:125]))
                else:
                    payload = payload[:125]
                pong_mask = os.urandom(4)
                masked_payload = bytes(b ^ pong_mask[i % 4] for i, b in enumerate(payload))
                pong = bytes([0x8A, 0x80 | len(masked_payload)]) + pong_mask + masked_payload
                try:
                    sock.sendall(pong)
                except OSError:
                    return
                continue
            if opcode == 0xA:  # pong
                continue
            if opcode == 0x1:  # text frame (complete or start of fragmented message)
                if mask_key is not None and payload:
                    payload = bytes(b ^ mask_key[i % 4] for i, b in enumerate(payload))
                if fin:
                    frag_buf = None
                    self._emit_text_payload(payload)
                else:
                    frag_buf = bytearray(payload)
                continue
            if opcode == 0x0:  # continuation frame
                if frag_buf is None:
                    continue
                if len(frag_buf) + len(payload) > _MAX_WS_FRAME_BYTES:
                    return
                if mask_key is not None and payload:
                    payload = bytes(b ^ mask_key[i % 4] for i, b in enumerate(payload))
                frag_buf.extend(payload)
                if fin:
                    assembled = bytes(frag_buf)
                    frag_buf = None
                    self._emit_text_payload(assembled)
                continue
            # Ignore binary preview frames (0x2) and unknown data opcodes without unmasking.
            if opcode == 0x2:
                frag_buf = None


def _has_control_chars(value: str) -> bool:
    return any(ord(ch) < 32 or ord(ch) == 127 for ch in value)


def _validate_prompt_id(prompt_id: object, *, label: str) -> str:
    if (
        not isinstance(prompt_id, str)
        or not prompt_id
        or "/" in prompt_id
        or "\\" in prompt_id
        or ".." in prompt_id
        or "?" in prompt_id
        or "#" in prompt_id
        or _has_control_chars(prompt_id)
    ):
        raise ComfyJobError(f"{label}: {prompt_id!r}")
    return prompt_id


def _is_safe_comfy_path(filename: object, subfolder: object) -> bool:
    return (
        isinstance(filename, str)
        and bool(filename)
        and "/" not in filename
        and "\\" not in filename
        and ".." not in filename
        and not _has_control_chars(filename)
        and isinstance(subfolder, str)
        and not subfolder.startswith("/")
        and "\\" not in subfolder
        and ".." not in subfolder.split("/")
        and not _has_control_chars(subfolder)
    )


def _json_dict(response: httpx.Response, label: str) -> dict[str, Any]:
    try:
        data = response.json()
    except (ValueError, RecursionError) as exc:
        raise ComfyJobError(
            f"ComfyUI {label} returned non-JSON response: {response.text[:200]!r}"
        ) from exc
    if not isinstance(data, dict):
        raise ComfyJobError(f"ComfyUI {label} returned unexpected JSON type: {type(data).__name__}")
    return data


class ComfyClient:
    def __init__(
        self,
        config: ComfyConfig,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        upload_subfolder: str = "mvdirector",
        on_progress: Callable[[ComfyProgress], None] | None = None,
    ) -> None:
        self._config = config
        self._sleep = sleep
        self._upload_subfolder = upload_subfolder
        self._client_id = uuid.uuid4().hex
        self._on_progress = on_progress
        self._uses_network = transport is None
        # The VRAM floor is checked once per phase (the first job after a /free), not per job:
        # within a phase ComfyUI keeps the phase's model loaded on purpose.
        self._vram_floor_checked = False
        self._vram_lock = threading.RLock()
        headers = dict(config.headers)
        if not any(key.lower() == "user-agent" for key in headers):
            headers["User-Agent"] = f"mvdirector/{__version__}"
        self._headers = headers
        # Opt-in only (``comfy.api_key_comfy_org: ${COMFY_ORG_API_KEY}``): never forward a
        # paid Comfy.org credential from the ambient environment to whatever host is configured.
        raw_api_key = (config.api_key_comfy_org or "").strip()
        if raw_api_key and _has_control_chars(raw_api_key):
            raise ConfigError("comfy.api_key_comfy_org must not contain control characters.")
        self._api_key_comfy_org = raw_api_key or None
        base_url = validate_http_base_url(config.base_url.strip(), field_name="comfy.base_url")
        self._http = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers=headers,
            timeout=config.request_timeout,
            transport=transport,
        )

    def set_progress_callback(self, callback: Callable[[ComfyProgress], None] | None) -> None:
        """Attach or replace the live progress callback for subsequent jobs on this client."""
        self._on_progress = callback

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> ComfyClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # ---------------------------------------------------------------- public

    def ping(self) -> dict[str, Any]:
        """Return ComfyUI's system stats; raises ComfyConnectionError if unreachable."""
        return _json_dict(self._request("GET", "/system_stats"), "GET /system_stats")

    def free_vram_gb(self) -> float | None:
        """Probe current free GPU VRAM (GiB) from ComfyUI ``/system_stats``, or None if absent.

        Credits & Attribution:
            Pre-shot VRAM floor custody inspired by ``JavaDerek/music-video-pipeline``
            (https://github.com/JavaDerek/music-video-pipeline — ``custody.py``).
        """
        stats = self.ping()
        devices = stats.get("devices")
        if not isinstance(devices, list) or not devices:
            return None
        free_bytes: list[int] = []
        for dev in devices:
            if not isinstance(dev, dict):
                continue
            vram = dev.get("vram_free")
            if (
                isinstance(vram, (int, float))
                and not isinstance(vram, bool)
                and math.isfinite(vram)
                and vram >= 0
            ):
                free_bytes.append(int(vram))
        if not free_bytes:
            return None
        return max(free_bytes) / (1024.0**3)

    def check_vram_floor(
        self,
        min_free_gb: float | None = None,
        *,
        settle_polls: int = _VRAM_SETTLE_POLLS,
        poll_seconds: float = _VRAM_POLL_SECONDS,
    ) -> float | None:
        """Verify free GPU VRAM meets ``min_free_gb``, asking ComfyUI to ``/free`` it first.

        Prevents silent OS GPU wedges when queueing heavy video models onto contended cards.
        ``/free`` only flags ComfyUI's worker to unload models, so VRAM is re-probed for up to
        ``settle_polls`` intervals before giving up with ``ComfyResourceError``, which stops the
        run instead of counting as a failed attempt.
        """
        floor = self._config.min_free_vram_gb if min_free_gb is None else min_free_gb
        if floor is None:
            return None
        avail = self.free_vram_gb()
        if avail is None or avail >= floor:
            return avail
        log.info(
            "Free VRAM %.2f GiB is below floor %.2f GiB; calling /free before queueing",
            avail,
            floor,
        )
        self.free_memory()
        for _ in range(max(1, settle_polls)):
            self._sleep(poll_seconds)
            avail = self.free_vram_gb()
            if avail is None or avail >= floor:
                return avail
        assert avail is not None
        raise ComfyResourceError(
            f"Free GPU VRAM ({avail:.2f} GiB) stayed below comfy.min_free_vram_gb "
            f"({floor:.2f} GiB) after asking ComfyUI to free memory. Another process may be "
            "using the GPU; free it or lower comfy.min_free_vram_gb, then run again."
        )

    def object_info(self, class_type: str | None = None) -> dict[str, Any]:
        """Fetch ComfyUI ``/object_info`` metadata (all nodes or a single ``class_type``)."""
        if class_type is not None:
            if (
                not class_type
                or "/" in class_type
                or "\\" in class_type
                or ".." in class_type
                or _has_control_chars(class_type)
            ):
                raise ComfyJobError(f"Invalid ComfyUI node class_type: {class_type!r}")
            endpoint = f"/object_info/{urllib.parse.quote(class_type, safe='')}"
        else:
            endpoint = "/object_info"
        return _json_dict(self._request("GET", endpoint), f"GET {endpoint}")

    def validate_graph_live(
        self,
        graph: dict[str, Any],
        *,
        object_info: dict[str, Any] | None = None,
        skip_inputs: set[tuple[str, str]] | None = None,
    ) -> list[str]:
        """Validate ``graph`` node classes and checkpoint/LoRA inputs against ``/object_info``.

        Credits & Attribution:
            Live ComfyUI ``/object_info`` pre-flight probing inspired by
            ``VisualFrisson/ComfyUI-OneShotMV`` (``fv/setup_check.py``) and
            ``heroesjouney/aimoviestudiov2`` (``routes_settings.py``).
        """
        info = self.object_info() if object_info is None else object_info
        if not info:
            return []
        problems: list[str] = []
        skip = skip_inputs or set()
        for node_id, node in graph.items():
            if not isinstance(node, dict):
                continue
            ctype = str(node.get("class_type", ""))
            if not ctype:
                continue
            meta = info.get(ctype)
            if not isinstance(meta, dict):
                problems.append(f"node {node_id}: unknown ComfyUI node class '{ctype}'")
                continue
            spec_groups = meta.get("input")
            if not isinstance(spec_groups, dict):
                continue
            all_specs: dict[str, Any] = {}
            for grp in ("required", "optional"):
                if isinstance(spec_groups.get(grp), dict):
                    all_specs.update(spec_groups[grp])
            raw_inputs = node.get("inputs")
            if not isinstance(raw_inputs, dict):
                continue
            for field, val in raw_inputs.items():
                if (str(node_id), field) in skip:
                    continue
                if isinstance(val, list) or not isinstance(val, str) or not val:
                    continue
                spec = all_specs.get(field)
                allowed: list[Any] | None = None
                if isinstance(spec, list) and spec:
                    if isinstance(spec[0], list):
                        allowed = spec[0]
                    elif spec[0] == "COMBO" and len(spec) > 1 and isinstance(spec[1], dict):
                        opts = spec[1].get("options")
                        if not isinstance(opts, list):
                            opts = spec[1].get("values")
                        if isinstance(opts, list):
                            allowed = opts
                if allowed and val not in allowed and field.endswith(("_name", "model", "ckpt")):
                    problems.append(
                        f"node {node_id} ({ctype}).{field}: '{val}' not installed on ComfyUI"
                    )
        return problems

    def free_memory(self, *, unload_models: bool = True, free_memory: bool = True) -> None:
        """Ask ComfyUI to unload models and free VRAM between workflow phases."""
        with self._vram_lock:
            self._vram_floor_checked = False
        try:
            response = self._http.post(
                "/free",
                json={"unload_models": unload_models, "free_memory": free_memory},
            )
            if response.status_code >= 400:
                log.debug("ComfyUI /free returned HTTP %d; ignoring", response.status_code)
        except httpx.TransportError as exc:
            log.debug("ComfyUI /free request failed (%s); ignoring", exc)

    def upload(
        self,
        path: Path,
        subfolder: str | None = None,
        *,
        content_addressed: bool | None = None,
    ) -> str:
        """Upload a file to ComfyUI's input folder and return the name loaders expect.

        ``/upload/image`` stores any file type, so it is used for audio as well. When
        ``content_addressed`` (or ``comfy.dedup_uploads``) is enabled, uploads use a SHA-256
        suffixed name and probe ``HEAD /view`` first to avoid redundant transfers of unchanged
        reference assets (inspired by ``afj176/open-cinema-studio``).
        """
        target_subfolder = self._upload_subfolder if subfolder is None else subfolder
        dedup = self._config.dedup_uploads if content_addressed is None else content_addressed
        upload_name = path.name
        if dedup and path.is_file():
            digest = file_sha256(path)[:12]
            upload_name = f"{path.stem}_{digest}{path.suffix}"
            try:
                probe_resp = self._http.head(
                    "/view",
                    params={
                        "filename": upload_name,
                        "subfolder": target_subfolder,
                        "type": "input",
                    },
                )
                if probe_resp.status_code == 200:
                    log.debug("ComfyUI upload dedup hit: %s/%s", target_subfolder, upload_name)
                    return f"{target_subfolder}/{upload_name}" if target_subfolder else upload_name
            except httpx.TransportError:
                pass
        with path.open("rb") as handle:
            response = self._request(
                "POST",
                "/upload/image",
                files={"image": (upload_name, handle, "application/octet-stream")},
                data={"subfolder": target_subfolder, "overwrite": "true"},
            )
        body = _json_dict(response, "POST /upload/image")
        name = body.get("name")
        raw_sub = body.get("subfolder")
        subfolder_str = "" if raw_sub is None else raw_sub
        if not _is_safe_comfy_path(name, subfolder_str):
            raise ComfyJobError(f"ComfyUI /upload/image returned an unsafe name: {body!r}")
        assert isinstance(name, str) and isinstance(subfolder_str, str)
        return f"{subfolder_str}/{name}" if subfolder_str else name

    def run(
        self,
        graph: dict[str, Any],
        output: OutputSpec,
        destination: Path,
        *,
        partial_execution_targets: list[str] | None = None,
    ) -> Path:
        """Queue ``graph``, wait for it and download its output to ``destination``."""
        with self._vram_lock:
            if not self._vram_floor_checked:
                self.check_vram_floor()
                self._vram_floor_checked = True
        started = time.monotonic()
        prompt_id = self.queue(graph, partial_execution_targets=partial_execution_targets)
        self._emit_progress(ComfyProgress(prompt_id=prompt_id, state="queued", elapsed_seconds=0.0))
        entry = self.wait(prompt_id)
        self._emit_progress(
            ComfyProgress(
                prompt_id=prompt_id,
                state="downloading",
                percent=99.0,
                elapsed_seconds=round(time.monotonic() - started, 1),
            )
        )
        path = self.download_output(entry, output, destination)
        self._emit_progress(
            ComfyProgress(
                prompt_id=prompt_id,
                state="done",
                percent=100.0,
                elapsed_seconds=round(time.monotonic() - started, 1),
            )
        )
        return path

    def queue(
        self,
        graph: dict[str, Any],
        *,
        partial_execution_targets: list[str] | None = None,
    ) -> str:
        payload: dict[str, Any] = {"prompt": graph, "client_id": self._client_id}
        if self._api_key_comfy_org:
            payload["extra_data"] = {"api_key_comfy_org": self._api_key_comfy_org}
        if partial_execution_targets:
            # Partial graph execution targeting specific output nodes (clean-room concept
            # inspired by Edakuban/VocaVid).
            payload["partial_execution_targets"] = list(partial_execution_targets)
        response = self._request(
            "POST",
            "/prompt",
            json=payload,
            ok_statuses=(200, 400),
            # Never re-POST /prompt on a read timeout or 500; ComfyUI may already have queued it.
            retry_on_read_error=False,
        )
        body = _json_dict(response, "POST /prompt")
        if response.status_code == 400 or body.get("node_errors"):
            raise ComfyJobError(f"ComfyUI rejected the workflow: {_summarize_rejection(body)}")
        prompt_id = _validate_prompt_id(
            body.get("prompt_id"), label="ComfyUI returned an invalid prompt_id"
        )
        log.debug("Queued prompt %s", prompt_id)
        return prompt_id

    def wait(self, prompt_id: str) -> dict[str, Any]:
        """Block until the prompt finishes; return its history entry."""
        _validate_prompt_id(prompt_id, label="Invalid ComfyUI prompt_id")
        encoded_id = urllib.parse.quote(prompt_id, safe="")
        started = time.monotonic()
        waited = 0.0
        min_step = (
            self._config.poll_interval
            if self._config.poll_interval > 0.0
            else max(0.05, self._config.job_timeout / 1000.0)
        )
        statusless_polls = 0
        ws_listener: _ComfyWsProgressListener | None = None
        if self._on_progress is not None and self._uses_network:
            ws_listener = _ComfyWsProgressListener(
                self._config.base_url,
                self._client_id,
                self._headers,
                prompt_id,
                started,
                self._on_progress,
            )
            ws_listener.start()
        try:
            while waited <= self._config.job_timeout:
                history = _json_dict(self._request("GET", f"/history/{encoded_id}"), "GET /history")
                entry = history.get(prompt_id)
                if isinstance(entry, dict):
                    status = entry.get("status")
                    if isinstance(status, dict):
                        if status.get("status_str") == "error":
                            raise ComfyJobError(f"ComfyUI job failed: {_summarize_failure(status)}")
                        if status.get("completed", True):
                            return entry
                    elif status is None:
                        # Older ComfyUI builds report no status at all. Outputs mean it is done;
                        # otherwise allow a few more polls in case this build fills them a beat
                        # later, then hand the entry back so download_output can say "produced no
                        # output". Waiting out the full job_timeout here (30 minutes by default)
                        # only to raise a misleading timeout stalls the run and hides the real
                        # problem. The budget counts polls, not seconds, so it holds whatever
                        # poll_interval is set to — including 0.0, where the loop never sleeps.
                        if isinstance(entry.get("outputs"), dict) and entry["outputs"]:
                            return entry
                        statusless_polls += 1
                        if statusless_polls >= _STATUSLESS_GRACE_POLLS:
                            return entry
                if self._on_progress is not None:
                    now = time.monotonic()
                    ws_window = max(5.0, self._config.poll_interval * 2.0)
                    ws_recent = (
                        ws_listener is not None
                        and ws_listener.connected
                        and ws_listener.last_event_at is not None
                        and (now - ws_listener.last_event_at) < ws_window
                    )
                    if not ws_recent:
                        self._poll_queue_progress(prompt_id, elapsed=now - started)
                self._sleep(self._config.poll_interval)
                waited = max(waited + min_step, time.monotonic() - started)
            raise ComfyJobError(
                f"ComfyUI job {prompt_id} timed out after {self._config.job_timeout}s"
            )
        finally:
            if ws_listener is not None:
                ws_listener.stop()

    def _emit_progress(self, progress: ComfyProgress) -> None:
        if self._on_progress is None:
            return
        try:
            self._on_progress(progress)
        except Exception:  # noqa: BLE001 - progress observer must never abort generation
            log.debug("ComfyUI progress callback failed", exc_info=True)

    def _poll_queue_progress(self, prompt_id: str, *, elapsed: float) -> None:
        """Best-effort HTTP ``/queue`` check when WebSocket step events are unavailable."""
        state = "running"
        queue_pos: int | None = None
        try:
            resp = self._http.get("/queue", timeout=min(self._config.request_timeout, 5.0))
            if resp.status_code == 200:
                body = resp.json()
                if isinstance(body, dict):
                    running = body.get("queue_running")
                    pending = body.get("queue_pending")
                    in_running = isinstance(running, list) and any(
                        isinstance(item, list) and len(item) >= 2 and item[1] == prompt_id
                        for item in running
                    )
                    if in_running:
                        state = "running"
                    elif isinstance(pending, list):
                        for idx, item in enumerate(pending, start=1):
                            if isinstance(item, list) and len(item) >= 2 and item[1] == prompt_id:
                                state = "queued"
                                queue_pos = idx
                                break
        except Exception:  # noqa: BLE001 - /queue is optional
            pass
        self._emit_progress(
            ComfyProgress(
                prompt_id=prompt_id,
                state=state,
                queue_position=queue_pos,
                elapsed_seconds=round(elapsed, 1),
            )
        )

    def download_output(self, entry: dict[str, Any], output: OutputSpec, destination: Path) -> Path:
        raw_outputs = entry.get("outputs")
        file_info = _pick_output(raw_outputs if isinstance(raw_outputs, dict) else {}, output)
        raw_filename = file_info.get("filename")
        raw_subfolder = file_info.get("subfolder")
        subfolder = "" if raw_subfolder is None else raw_subfolder
        raw_type = file_info.get("type")
        file_type = "output" if raw_type is None else raw_type
        if (
            not _is_safe_comfy_path(raw_filename, subfolder)
            or not isinstance(file_type, str)
            or file_type not in _ALLOWED_OUTPUT_TYPES
        ):
            raise ComfyJobError(f"ComfyUI returned an unsafe output path: {file_info!r}")
        assert isinstance(raw_filename, str) and isinstance(subfolder, str)
        filename = raw_filename
        params = {
            "filename": filename,
            "subfolder": subfolder,
            "type": file_type,
        }
        destination.parent.mkdir(parents=True, exist_ok=True)
        temp_path = destination.with_name(f"{destination.name}.{uuid.uuid4().hex[:8]}.part")
        last_error: Exception | None = None
        for attempt in range(_TRANSIENT_RETRIES):
            response: httpx.Response | None = None
            try:
                with self._http.stream("GET", "/view", params=params) as resp:
                    response = resp
                    if resp.status_code != 200:
                        if resp.status_code < 500 and resp.status_code != 429:
                            resp.read()
                            raise ComfyJobError(
                                f"GET /view returned {resp.status_code}: {resp.text[:300]}"
                            )
                        last_error = ComfyJobError(f"GET /view returned {resp.status_code}")
                    else:
                        limit_mib = _MAX_DOWNLOAD_BYTES // (1024 * 1024)
                        raw_length = resp.headers.get("content-length")
                        if (
                            raw_length
                            and raw_length.isdigit()
                            and int(raw_length) > _MAX_DOWNLOAD_BYTES
                        ):
                            raise ComfyJobError(
                                f"ComfyUI output '{filename}' exceeds the {limit_mib} MiB "
                                "download limit."
                            )
                        written = 0
                        with temp_path.open("wb") as handle:
                            for chunk in resp.iter_bytes(chunk_size=65536):
                                written += len(chunk)
                                if written > _MAX_DOWNLOAD_BYTES:
                                    raise ComfyJobError(
                                        f"ComfyUI output '{filename}' exceeds the {limit_mib} "
                                        "MiB download limit."
                                    )
                                handle.write(chunk)
                        temp_path.replace(destination)
                        return destination
            except ComfyJobError:
                temp_path.unlink(missing_ok=True)
                raise
            except httpx.TransportError as exc:
                last_error = exc
                temp_path.unlink(missing_ok=True)
            if attempt < _TRANSIENT_RETRIES - 1:
                self._sleep(retry_delay(response, attempt))
        temp_path.unlink(missing_ok=True)
        raise ComfyConnectionError(
            f"Could not reach ComfyUI at {redact_url(self._config.base_url)} ({last_error}). "
            "Check that ComfyUI is running and COMFY_URL is correct."
        )

    # --------------------------------------------------------------- private

    def _request(
        self,
        method: str,
        url: str,
        ok_statuses: tuple[int, ...] = (200,),
        *,
        retry_on_read_error: bool = True,
        **kwargs: Any,
    ) -> httpx.Response:
        last_error: Exception | None = None
        for attempt in range(_TRANSIENT_RETRIES):
            response: httpx.Response | None = None
            try:
                response = self._http.request(method, url, **kwargs)
            except httpx.TransportError as exc:
                if not retry_on_read_error and not isinstance(
                    exc, (httpx.ConnectError, httpx.ConnectTimeout)
                ):
                    raise ComfyConnectionError(
                        f"ComfyUI request {method} {url} interrupted ({exc}). Check ComfyUI queue "
                        "before retrying."
                    ) from exc
                last_error = exc
            else:
                if response.status_code in ok_statuses:
                    return response
                if response.status_code < 500 and response.status_code != 429:
                    raise ComfyJobError(
                        f"{method} {url} returned {response.status_code}: {response.text[:300]}"
                    )
                if not retry_on_read_error and response.status_code not in (429, 503):
                    raise ComfyJobError(
                        f"{method} {url} returned {response.status_code}: {response.text[:300]}"
                    )
                last_error = ComfyJobError(f"{method} {url} returned {response.status_code}")
            if attempt < _TRANSIENT_RETRIES - 1:
                self._sleep(retry_delay(response, attempt))
        raise ComfyConnectionError(
            f"Could not reach ComfyUI at {redact_url(self._config.base_url)} ({last_error}). "
            "Check that ComfyUI is running and COMFY_URL is correct."
        )


def _pick_output(outputs: dict[str, Any], spec: OutputSpec) -> dict[str, Any]:
    """Find the first output file matching the spec's node and extensions."""
    nodes = [spec.node] if spec.node is not None else sorted(outputs)
    extensions = tuple(ext.lower() for ext in spec.extensions)
    for node in nodes:
        # Different nodes and ComfyUI versions report files under different keys
        # (images, gifs, videos, audio).
        node_outputs = outputs.get(node)
        if not isinstance(node_outputs, dict):
            continue
        for files in node_outputs.values():
            if not isinstance(files, list):
                continue
            for item in files:
                if isinstance(item, dict) and str(item.get("filename", "")).lower().endswith(
                    extensions
                ):
                    return item
    raise ComfyJobError(
        f"The workflow finished but produced no {'/'.join(spec.extensions)} output"
        + (f" on node {spec.node}" if spec.node else "")
    )


def _summarize_rejection(body: dict[str, Any]) -> str:
    parts = []
    error = body.get("error")
    if isinstance(error, dict):
        parts.append(str(error.get("message", error)))
    elif error:
        parts.append(str(error))
    raw_node_errors = body.get("node_errors")
    if isinstance(raw_node_errors, dict):
        for node_id, info in raw_node_errors.items():
            if not isinstance(info, dict):
                parts.append(f"node {node_id}: {info}")
                continue
            raw_errors = info.get("errors")
            error_list = raw_errors if isinstance(raw_errors, list) else []
            messages = "; ".join(
                (
                    f"{e.get('message', '')} {e.get('details', '')}".strip()
                    if isinstance(e, dict)
                    else str(e).strip()
                )
                for e in error_list
            )
            parts.append(f"node {node_id} ({info.get('class_type', '?')}): {messages}")
    return " | ".join(parts) or str(body)[:300]


def _summarize_failure(status: dict[str, Any]) -> str:
    raw_messages = status.get("messages")
    messages = raw_messages if isinstance(raw_messages, list) else []
    for message in messages:
        if isinstance(message, list) and len(message) == 2 and message[0] == "execution_error":
            detail = message[1]
            if isinstance(detail, dict):
                exc_msg = str(detail.get("exception_message") or "").strip()
                return f"node {detail.get('node_id')} ({detail.get('node_type')}): {exc_msg}"
            return str(detail).strip() or "execution_error"
    return "unknown error (see the ComfyUI log)"
