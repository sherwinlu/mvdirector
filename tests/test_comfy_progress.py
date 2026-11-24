# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from __future__ import annotations

import base64
import hashlib
import json
import socket
import struct
import threading
import time
from contextlib import suppress
from pathlib import Path
from typing import Any
from unittest.mock import patch

import httpx
import pytest

from mvdirector.comfy.client import (
    ComfyClient,
    ComfyProgress,
    _ComfyWsProgressListener,
    parse_comfy_ws_event,
)
from mvdirector.comfy.workflow import OutputSpec
from mvdirector.config import ComfyConfig


def test_parse_comfy_ws_progress_event() -> None:
    ev = parse_comfy_ws_event(
        {
            "type": "progress",
            "data": {"prompt_id": "p-1", "value": 7, "max": 20, "node": "12"},
        },
        "p-1",
        elapsed_seconds=4.26,
    )
    assert ev == ComfyProgress(
        prompt_id="p-1",
        state="running",
        step=7,
        total_steps=20,
        percent=35.0,
        node="12",
        elapsed_seconds=4.3,
    )


def test_parse_comfy_ws_ignores_other_prompt() -> None:
    ev = parse_comfy_ws_event(
        {
            "type": "progress",
            "data": {"prompt_id": "other-job", "value": 5, "max": 10},
        },
        "p-1",
    )
    assert ev is None


def test_parse_comfy_ws_progress_state_aggregates_nodes() -> None:
    ev = parse_comfy_ws_event(
        {
            "type": "progress_state",
            "data": {
                "prompt_id": "p-2",
                "nodes": {
                    "3": {"value": 10, "max": 10, "state": "finished"},
                    "8": {"value": 5, "max": 10, "state": "running"},
                },
            },
        },
        "p-2",
        elapsed_seconds=12.0,
    )
    assert ev is not None
    assert ev.step == 15
    assert ev.total_steps == 20
    assert ev.percent == 75.0
    assert ev.node == "8"


def test_parse_comfy_ws_executing_none_marks_downloading() -> None:
    ev = parse_comfy_ws_event(
        {"type": "executing", "data": {"prompt_id": "p-1", "node": None}},
        "p-1",
        elapsed_seconds=9.1,
    )
    assert ev is not None
    assert ev.state == "downloading"
    assert ev.percent == 99.0


def test_comfy_client_emits_queue_and_completion_progress(tmp_path: Path) -> None:
    polls = {"history": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/prompt":
            return httpx.Response(200, json={"prompt_id": "job-42"})
        if request.url.path == "/history/job-42":
            polls["history"] += 1
            if polls["history"] < 3:
                return httpx.Response(200, json={})
            return httpx.Response(
                200,
                json={
                    "job-42": {
                        "status": {"completed": True, "status_str": "success"},
                        "outputs": {
                            "9": {
                                "images": [
                                    {"filename": "out.png", "subfolder": "", "type": "output"}
                                ]
                            }
                        },
                    }
                },
            )
        if request.url.path == "/queue":
            if polls["history"] == 1:
                return httpx.Response(
                    200,
                    json={"queue_running": [], "queue_pending": [[1, "other"], [2, "job-42"]]},
                )
            return httpx.Response(
                200,
                json={"queue_running": [[2, "job-42"]], "queue_pending": []},
            )
        if request.url.path == "/view":
            return httpx.Response(200, content=b"fake-png-bytes")
        return httpx.Response(404)

    seen: list[ComfyProgress] = []
    cfg = ComfyConfig(base_url="http://127.0.0.1:8188", poll_interval=0.0, job_timeout=5.0)
    with ComfyClient(
        cfg,
        transport=httpx.MockTransport(handler),
        sleep=lambda _: None,
        on_progress=seen.append,
    ) as client:
        out = client.run({"9": {}}, OutputSpec(node="9"), tmp_path / "shot.png")

    assert out.read_bytes() == b"fake-png-bytes"
    states = [p.state for p in seen]
    assert states == ["queued", "queued", "running", "downloading", "done"]
    assert seen[1].queue_position == 2
    assert seen[-1].percent == 100.0


def test_parse_comfy_ws_event_rejects_nonfinite_negative_and_unprompted_completion() -> None:
    bad_payloads = [
        {"type": "progress", "data": {"prompt_id": "p1", "value": float("nan"), "max": 10}},
        {"type": "progress", "data": {"prompt_id": "p1", "value": float("inf"), "max": 10}},
        {"type": "progress", "data": {"prompt_id": "p1", "value": 5, "max": float("inf")}},
        {"type": "progress", "data": {"prompt_id": "p1", "value": -1, "max": 10}},
        {"type": "progress", "data": {"prompt_id": "p1", "value": True, "max": 10}},
        {
            "type": "progress_state",
            "data": {"prompt_id": "p1", "nodes": {"1": {"value": float("nan"), "max": 10}}},
        },
        {
            "type": "progress_state",
            "data": {"prompt_id": "p1", "nodes": {"1": {"value": 5, "max": float("inf")}}},
        },
        {"type": "executing", "data": {"node": None}},
        {"type": "execution_start", "data": {}},
    ]
    for payload in bad_payloads:
        assert parse_comfy_ws_event(payload, "p1") is None, payload

    assert parse_comfy_ws_event("not-a-dict", "p1") is None  # type: ignore[arg-type]
    ev = parse_comfy_ws_event(
        {"type": "progress", "data": {"prompt_id": "p1", "value": 3, "max": 6}},
        "p1",
        elapsed_seconds=float("nan"),
    )
    assert ev is not None and ev.elapsed_seconds == 0.0


def _ws_frame(
    payload: bytes, *, fin: bool = True, opcode: int = 0x1, mask: bytes | None = None
) -> bytes:
    b0 = (0x80 if fin else 0x00) | (opcode & 0x0F)
    n = len(payload)
    mask_bit = 0x80 if mask is not None else 0x00
    if n < 126:
        hdr = bytes([b0, mask_bit | n])
    elif n < 65536:
        hdr = bytes([b0, mask_bit | 126]) + struct.pack("!H", n)
    else:
        hdr = bytes([b0, mask_bit | 127]) + struct.pack("!Q", n)
    if mask is not None:
        payload = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
        return hdr + mask + payload
    return hdr + payload


def _ws_handshake_reply(
    req_bytes: bytes,
    *,
    status: str = "101 Switching Protocols",
    bad_accept: bool = False,
) -> bytes:
    key = ""
    for line in req_bytes.decode("latin1").split("\r\n"):
        if line.lower().startswith("sec-websocket-key:"):
            key = line.split(":", 1)[1].strip()
    accept = (
        "INVALID=="
        if bad_accept
        else base64.b64encode(
            hashlib.sha1((key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode("ascii")).digest()
        ).decode("ascii")
    )
    return (
        f"HTTP/1.1 {status}\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        f"Sec-WebSocket-Accept: {accept}\r\n\r\n"
    ).encode("ascii")


def test_ws_listener_reassembles_fragments_answers_ping_and_formats_ipv6_host() -> None:
    seen: list[ComfyProgress] = []
    req_captured: list[str] = []
    pong_captured: list[bytes] = []

    client_sock, server_sock = socket.socketpair()

    def serve() -> None:
        try:
            req = server_sock.recv(4096)
            req_captured.append(req.decode("latin1"))
            server_sock.sendall(_ws_handshake_reply(req, status="101"))
            msg = json.dumps(
                {"type": "progress", "data": {"prompt_id": "p-frag", "value": 4, "max": 8}}
            ).encode("utf-8")
            half = len(msg) // 2
            server_sock.sendall(_ws_frame(msg[:half], fin=False, opcode=0x1))
            server_sock.sendall(_ws_frame(b"ping", fin=True, opcode=0x9))
            pong_captured.append(server_sock.recv(64))
            server_sock.sendall(_ws_frame(msg[half:], fin=True, opcode=0x0))
            server_sock.sendall(_ws_frame(b"", fin=True, opcode=0x8))
        finally:
            server_sock.close()

    srv = threading.Thread(target=serve, daemon=True)
    srv.start()
    with patch("mvdirector.comfy.client.socket.create_connection", return_value=client_sock):
        listener = _ComfyWsProgressListener(
            "http://[::1]:8188",
            "cid",
            {"Sec-WebSocket-Version": "99", "X-Bad:Name": "v", "X-Null": "a\x00b", "X-Ok": "1"},
            "p-frag",
            time.monotonic(),
            seen.append,
        )
        listener.start()
        srv.join(timeout=2.0)
        listener.stop()

    assert req_captured
    assert "Host: [::1]:8188\r\n" in req_captured[0]
    assert req_captured[0].lower().count("sec-websocket-version:") == 1
    assert "X-Bad:Name" not in req_captured[0]
    assert "\x00" not in req_captured[0]
    assert "X-Ok: 1\r\n" in req_captured[0]
    assert pong_captured and (pong_captured[0][0] & 0x0F) == 0xA
    assert len(seen) == 1 and seen[0].step == 4 and seen[0].percent == 50.0


def test_ws_listener_rejects_bad_handshake_and_stops_promptly() -> None:
    seen: list[ComfyProgress] = []
    client_sock, server_sock = socket.socketpair()

    def serve_bad() -> None:
        try:
            req = server_sock.recv(4096)
            server_sock.sendall(_ws_handshake_reply(req, status="200 OK 101 ", bad_accept=True))
            with suppress(OSError):
                server_sock.sendall(
                    _ws_frame(
                        json.dumps(
                            {"type": "progress", "data": {"prompt_id": "p", "value": 1, "max": 2}}
                        ).encode("utf-8")
                    )
                )
        finally:
            server_sock.close()

    t = threading.Thread(target=serve_bad, daemon=True)
    t.start()
    with patch("mvdirector.comfy.client.socket.create_connection", return_value=client_sock):
        listener = _ComfyWsProgressListener(
            "http://127.0.0.1:8188", "cid", {}, "p", time.monotonic(), seen.append
        )
        listener.start()
        t.join(timeout=2.0)
        listener.stop()
    assert seen == []

    # Idle open socket stops quickly instead of stalling for 1s+.
    client_idle, server_idle = socket.socketpair()
    connected = threading.Event()
    release = threading.Event()

    def serve_idle() -> None:
        try:
            req = server_idle.recv(4096)
            server_idle.sendall(_ws_handshake_reply(req))
            connected.set()
            release.wait(timeout=3.0)
        finally:
            server_idle.close()

    t2 = threading.Thread(target=serve_idle, daemon=True)
    t2.start()
    with patch("mvdirector.comfy.client.socket.create_connection", return_value=client_idle):
        l2 = _ComfyWsProgressListener(
            "http://127.0.0.1:8188", "cid", {}, "p2", time.monotonic(), lambda _: None
        )
        l2.start()
        assert connected.wait(timeout=2.0)
        t0 = time.monotonic()
        l2.stop()
        elapsed_ms = (time.monotonic() - t0) * 1000
    release.set()
    t2.join(timeout=2.0)
    assert elapsed_ms < 500


def test_comfy_client_rejects_control_chars_in_env_api_key_comfy_org(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("COMFY_ORG_API_KEY", "bad\r\nkey")
    with pytest.raises(ValueError, match="control characters"):
        ComfyConfig(base_url="http://127.0.0.1:8188", api_key_comfy_org="${COMFY_ORG_API_KEY}")


def test_comfy_client_never_forwards_ambient_comfy_org_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An exported COMFY_ORG_API_KEY must not leak to a host the project never opted in for."""
    monkeypatch.setenv("COMFY_ORG_API_KEY", "ambient-secret")
    captured: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(200, json={"prompt_id": "p1"})

    cfg = ComfyConfig(base_url="https://shared-gpu.example.com")
    with ComfyClient(cfg, transport=httpx.MockTransport(handler)) as client:
        client.queue({"1": {"class_type": "X", "inputs": {}}})
    assert "extra_data" not in captured[0]

    opted_in = ComfyConfig(
        base_url="https://shared-gpu.example.com", api_key_comfy_org="${COMFY_ORG_API_KEY}"
    )
    with ComfyClient(opted_in, transport=httpx.MockTransport(handler)) as client:
        client.queue({"1": {"class_type": "X", "inputs": {}}})
    assert captured[1]["extra_data"] == {"api_key_comfy_org": "ambient-secret"}


def test_ws_listener_closes_raw_socket_on_tls_error_and_caps_fragment_size() -> None:
    import ssl

    closed = {"raw": False}
    seen_min_version: list[ssl.TLSVersion] = []

    class DummySock:
        def close(self) -> None:
            closed["raw"] = True

        def shutdown(self, _how: int) -> None:
            pass

    def fake_wrap(self: ssl.SSLContext, _sock: object, **_kw: object) -> object:
        seen_min_version.append(self.minimum_version)
        raise ssl.SSLError("tls failure")

    with (
        patch("mvdirector.comfy.client.socket.create_connection", return_value=DummySock()),
        patch.object(ssl.SSLContext, "wrap_socket", autospec=True, side_effect=fake_wrap),
    ):
        listener = _ComfyWsProgressListener(
            "https://gpu.example.com:8188", "cid", {}, "p-tls", time.monotonic(), lambda _: None
        )
        listener.start()
        listener.stop()
    assert closed["raw"] is True
    assert seen_min_version == [ssl.TLSVersion.TLSv1_2]

    # Fragmented messages exceeding _MAX_WS_FRAME_BYTES abort cleanly.
    seen: list[ComfyProgress] = []
    c_sock, s_sock = socket.socketpair()

    def serve_oversized() -> None:
        try:
            req = s_sock.recv(4096)
            s_sock.sendall(_ws_handshake_reply(req))
            chunk = b"x" * (1024 * 1024)
            with suppress(OSError):
                s_sock.sendall(_ws_frame(chunk, fin=False, opcode=0x1))
                for _ in range(5):
                    s_sock.sendall(_ws_frame(chunk, fin=False, opcode=0x0))
                s_sock.sendall(_ws_frame(b"}", fin=True, opcode=0x0))
        finally:
            s_sock.close()

    t = threading.Thread(target=serve_oversized, daemon=True)
    t.start()
    with patch("mvdirector.comfy.client.socket.create_connection", return_value=c_sock):
        l_big = _ComfyWsProgressListener(
            "http://127.0.0.1:8188", "cid", {}, "p-big", time.monotonic(), seen.append
        )
        l_big.start()
        t.join(timeout=3.0)
        l_big.stop()
    assert seen == []


def test_parse_comfy_ws_event_handles_progress_state_float_overflow() -> None:
    ev = parse_comfy_ws_event(
        {
            "type": "progress_state",
            "data": {
                "prompt_id": "p-ovf",
                "nodes": {
                    "1": {"value": 1e308, "max": 1e308, "state": "running"},
                    "2": {"value": 1e308, "max": 1e308, "state": "running"},
                },
            },
        },
        "p-ovf",
    )
    assert ev is None


def test_ws_listener_blocks_whitespace_headers_and_oversized_control_frames() -> None:
    req_captured: list[str] = []
    pong_captured: list[bytes] = []
    c_sock, s_sock = socket.socketpair()

    def serve() -> None:
        try:
            req = s_sock.recv(4096)
            req_captured.append(req.decode("latin1"))
            s_sock.sendall(_ws_handshake_reply(req))
            # RFC 6455 §5.5 forbids control frames with payload > 125 bytes.
            s_sock.sendall(_ws_frame(b"A" * 200, fin=True, opcode=0x9))
            s_sock.settimeout(0.4)
            with suppress(OSError, TimeoutError):
                data = s_sock.recv(256)
                if data:
                    pong_captured.append(data)
        finally:
            s_sock.close()

    t = threading.Thread(target=serve, daemon=True)
    t.start()
    with patch("mvdirector.comfy.client.socket.create_connection", return_value=c_sock):
        listener = _ComfyWsProgressListener(
            "http://127.0.0.1:8188",
            "cid",
            {" Host": "evil.example", "X-Bad Name": "1", "X-Good-Header": "ok"},
            "p-ctrl",
            time.monotonic(),
            lambda _: None,
        )
        listener.start()
        t.join(timeout=2.0)
        listener.stop()

    assert req_captured
    assert " Host:" not in req_captured[0]
    assert "X-Bad Name:" not in req_captured[0]
    assert "X-Good-Header: ok\r\n" in req_captured[0]
    assert pong_captured == []
