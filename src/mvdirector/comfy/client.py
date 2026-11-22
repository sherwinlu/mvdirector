# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Minimal ComfyUI HTTP client: upload inputs, queue a workflow, wait, download outputs.

Polling ``/history`` is used instead of the websocket because it survives proxies
(such as hosted GPU proxies) and reconnects without extra state.
"""

from __future__ import annotations

import logging
import time
import urllib.parse
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx

from mvdirector.comfy.cache import file_sha256
from mvdirector.comfy.workflow import OutputSpec
from mvdirector.config import ComfyConfig
from mvdirector.errors import ComfyConnectionError, ComfyJobError
from mvdirector.llm.http import retry_delay

log = logging.getLogger(__name__)

_TRANSIENT_RETRIES = 3
_MAX_DOWNLOAD_BYTES = 512 * 1024 * 1024  # 512 MiB cap per generated asset


class ComfyClient:
    def __init__(
        self,
        config: ComfyConfig,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        upload_subfolder: str = "mvdirector",
    ) -> None:
        self._config = config
        self._sleep = sleep
        self._upload_subfolder = upload_subfolder
        self._client_id = uuid.uuid4().hex
        self._http = httpx.Client(
            base_url=config.base_url.rstrip("/"),
            headers=config.headers,
            timeout=config.request_timeout,
            transport=transport,
        )

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> ComfyClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # ---------------------------------------------------------------- public

    def ping(self) -> dict[str, Any]:
        """Return ComfyUI's system stats; raises ComfyConnectionError if unreachable."""
        return self._request("GET", "/system_stats").json()

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
            if isinstance(vram, (int, float)) and vram >= 0:
                free_bytes.append(int(vram))
        if not free_bytes:
            return None
        return max(free_bytes) / (1024.0**3)

    def check_vram_floor(self, min_free_gb: float | None = None) -> float | None:
        """Verify free GPU VRAM meets ``min_free_gb`` (calls ``/free`` once before failing).

        Prevents silent OS GPU wedges when queueing heavy video models onto contended cards.
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
        avail = self.free_vram_gb()
        if avail is not None and avail < floor:
            raise ComfyJobError(
                f"Free GPU VRAM ({avail:.2f} GiB) is below "
                f"comfy.min_free_vram_gb ({floor:.2f} GiB)."
            )
        return avail

    def object_info(self, class_type: str | None = None) -> dict[str, Any]:
        """Fetch ComfyUI ``/object_info`` metadata (all nodes or a single ``class_type``)."""
        if class_type is not None:
            if (
                not class_type
                or "/" in class_type
                or "\\" in class_type
                or ".." in class_type
                or any(ord(ch) < 32 or ord(ch) == 127 for ch in class_type)
            ):
                raise ComfyJobError(f"Invalid ComfyUI node class_type: {class_type!r}")
            endpoint = f"/object_info/{urllib.parse.quote(class_type, safe='')}"
        else:
            endpoint = "/object_info"
        data = self._request("GET", endpoint).json()
        return data if isinstance(data, dict) else {}

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
            for field, val in (node.get("inputs") or {}).items():
                if (str(node_id), field) in skip:
                    continue
                if isinstance(val, list) or not isinstance(val, str) or not val:
                    continue
                spec = all_specs.get(field)
                allowed: list[Any] | None = None
                if isinstance(spec, list) and spec:
                    if isinstance(spec[0], list):
                        allowed = spec[0]
                    elif (
                        spec[0] == "COMBO"
                        and len(spec) > 1
                        and isinstance(spec[1], dict)
                        and isinstance(spec[1].get("options"), list)
                    ):
                        allowed = spec[1]["options"]
                if allowed and val not in allowed and field.endswith(("_name", "model", "ckpt")):
                    problems.append(
                        f"node {node_id} ({ctype}).{field}: '{val}' not installed on ComfyUI"
                    )
        return problems

    def free_memory(self, *, unload_models: bool = True, free_memory: bool = True) -> None:
        """Ask ComfyUI to unload models and free VRAM between workflow phases."""
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
        body = response.json()
        name = body["name"]
        return f"{body['subfolder']}/{name}" if body.get("subfolder") else name

    def run(
        self,
        graph: dict[str, Any],
        output: OutputSpec,
        destination: Path,
        *,
        partial_execution_targets: list[str] | None = None,
    ) -> Path:
        """Queue ``graph``, wait for it and download its output to ``destination``."""
        self.check_vram_floor()
        prompt_id = self.queue(graph, partial_execution_targets=partial_execution_targets)
        entry = self.wait(prompt_id)
        return self.download_output(entry, output, destination)

    def queue(
        self,
        graph: dict[str, Any],
        *,
        partial_execution_targets: list[str] | None = None,
    ) -> str:
        payload: dict[str, Any] = {"prompt": graph, "client_id": self._client_id}
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
        body = response.json()
        if response.status_code == 400 or body.get("node_errors"):
            raise ComfyJobError(f"ComfyUI rejected the workflow: {_summarize_rejection(body)}")
        log.debug("Queued prompt %s", body["prompt_id"])
        return body["prompt_id"]

    def wait(self, prompt_id: str) -> dict[str, Any]:
        """Block until the prompt finishes; return its history entry."""
        waited = 0.0
        while waited <= self._config.job_timeout:
            history = self._request("GET", f"/history/{prompt_id}").json()
            entry = history.get(prompt_id)
            if entry is not None:
                status = entry.get("status", {})
                if status.get("status_str") == "error":
                    raise ComfyJobError(f"ComfyUI job failed: {_summarize_failure(status)}")
                if status.get("completed", True):
                    return entry
            self._sleep(self._config.poll_interval)
            waited += self._config.poll_interval
        raise ComfyJobError(f"ComfyUI job {prompt_id} timed out after {self._config.job_timeout}s")

    def download_output(self, entry: dict[str, Any], output: OutputSpec, destination: Path) -> Path:
        file_info = _pick_output(entry.get("outputs", {}), output)
        filename = str(file_info.get("filename", ""))
        subfolder = str(file_info.get("subfolder", ""))
        if (
            not filename
            or "/" in filename
            or "\\" in filename
            or ".." in filename
            or subfolder.startswith("/")
            or "\\" in subfolder
            or ".." in subfolder.split("/")
        ):
            raise ComfyJobError(f"ComfyUI returned an unsafe output path: {file_info!r}")
        params = {
            "filename": filename,
            "subfolder": subfolder,
            "type": file_info.get("type", "output"),
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
            self._sleep(retry_delay(response, attempt))
        temp_path.unlink(missing_ok=True)
        raise ComfyConnectionError(
            f"Could not reach ComfyUI at {self._config.base_url} ({last_error}). "
            "Check that the pod is running and COMFY_URL is correct."
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
            self._sleep(retry_delay(response, attempt))
        raise ComfyConnectionError(
            f"Could not reach ComfyUI at {self._config.base_url} ({last_error}). "
            "Check that the pod is running and COMFY_URL is correct."
        )


def _pick_output(outputs: dict[str, Any], spec: OutputSpec) -> dict[str, Any]:
    """Find the first output file matching the spec's node and extensions."""
    nodes = [spec.node] if spec.node is not None else sorted(outputs)
    extensions = tuple(ext.lower() for ext in spec.extensions)
    for node in nodes:
        # Different nodes report files under different keys (images, gifs, videos, audio).
        for files in outputs.get(node, {}).values():
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
    for node_id, info in (body.get("node_errors") or {}).items():
        messages = "; ".join(
            f"{e.get('message', '')} {e.get('details', '')}".strip() for e in info.get("errors", [])
        )
        parts.append(f"node {node_id} ({info.get('class_type', '?')}): {messages}")
    return " | ".join(parts) or str(body)[:300]


def _summarize_failure(status: dict[str, Any]) -> str:
    for message in status.get("messages", []):
        if isinstance(message, list) and len(message) == 2 and message[0] == "execution_error":
            detail = message[1]
            return (
                f"node {detail.get('node_id')} ({detail.get('node_type')}): "
                f"{detail.get('exception_message', '').strip()}"
            )
    return "unknown error (see the ComfyUI log)"
