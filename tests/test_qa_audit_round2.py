# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Regressions from the second white/black-box QA, stress and security audit."""

from __future__ import annotations

import contextlib
import socket
import threading
import time
from pathlib import Path

import httpx
import pytest

from mvdirector import api
from mvdirector.errors import ConfigError, MVDError
from mvdirector.mcp import MCPServer
from mvdirector.mcp.server import _MCPHTTPServer, precheck_http_request
from mvdirector.models import AssetKind
from mvdirector.planning.planner import build_manifest
from mvdirector.review import policy


def _planned(project, analysis):
    project.save_analysis(analysis)
    manifest = build_manifest(project.config, analysis, "lyrics", "style", llm=None)
    project.save_manifest(manifest)
    return manifest


# ------------------------------------------------------------------ assemble output target


@pytest.mark.parametrize(
    "target",
    ["manifest.json", "project.yaml", "analysis.json", "inputs/mix.wav", "inputs/lyrics.txt"],
)
def test_assemble_refuses_to_overwrite_project_state_or_inputs(project, analysis, target):
    _planned(project, analysis)
    before = project.path(target).read_bytes()
    with pytest.raises(ConfigError, match="must be a video file"):
        api.assemble(project.root, output=project.root / target, force=True)
    assert project.path(target).read_bytes() == before


def test_assemble_rejects_directory_target_before_rendering(project, analysis, monkeypatch):
    _planned(project, analysis)
    rendered: list[Path] = []
    monkeypatch.setattr(
        "mvdirector.assemble.assemble", lambda *a, **k: rendered.append(a[2]) or a[2]
    )
    # A directory whose name looks like a video, and the project root itself.
    fake_dir = project.root / "output" / "cut.mp4"
    fake_dir.mkdir(parents=True)
    with pytest.raises(ConfigError, match="is a directory"):
        api.assemble(project.root, output=fake_dir, force=True)
    with pytest.raises(ConfigError, match="must be a video file"):
        api.assemble(project.root, output=project.root, force=True)
    assert rendered == []


def test_assemble_refuses_to_overwrite_generated_clips_but_allows_output_dir(
    project, analysis, monkeypatch, tmp_path
):
    manifest = _planned(project, analysis)
    rendered: list[Path] = []
    monkeypatch.setattr(
        "mvdirector.assemble.assemble", lambda *a, **k: rendered.append(a[2]) or a[2]
    )
    clip = project.asset_path(manifest.shots[0], AssetKind.VIDEO, 1)
    clip.parent.mkdir(parents=True, exist_ok=True)
    clip.write_bytes(b"clip")
    with pytest.raises(ConfigError, match="Refusing to overwrite existing project file"):
        api.assemble(project.root, output=clip, force=True)
    assert clip.read_bytes() == b"clip"

    # A sibling project's generated clip is protected too.
    sibling = tmp_path / "other"
    (sibling / "shots").mkdir(parents=True)
    (sibling / "project.yaml").write_text("title: other\n")
    (sibling / "shots" / "s001.mp4").write_bytes(b"theirs")
    with pytest.raises(ConfigError, match="Refusing to overwrite"):
        api.assemble(project.root, output=sibling / "shots" / "s001.mp4", force=True)

    # Re-rendering over an earlier cut in output/, a new file, or a non-project path is fine.
    previous = project.output_dir / "final.mp4"
    previous.parent.mkdir(parents=True, exist_ok=True)
    previous.write_bytes(b"old cut")
    loose = tmp_path / "loose" / "cut.MOV"
    loose.parent.mkdir()
    loose.write_bytes(b"x")
    for ok in (previous, project.root / "exports" / "new.mp4", loose):
        api.assemble(project.root, output=ok, force=True)
    assert rendered == [previous.resolve(), (project.root / "exports" / "new.mp4"), loose]


def test_mcp_assemble_output_cannot_clobber_manifest(project, analysis):
    _planned(project, analysis)
    server = MCPServer(default_project=project.root, allowed_roots=[project.root.parent])
    resp = server.handle_message(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "mvd_assemble",
                "arguments": {"output": "manifest.json", "force": True, "background": False},
            },
        }
    )
    assert resp is not None and resp["result"]["isError"] is True
    assert "must be a video file" in resp["result"]["content"][0]["text"]
    project.load_manifest()  # still valid JSON


# ------------------------------------------------------------------ review notes


def test_review_notes_are_bounded_and_trimmed(project, analysis):
    manifest = _planned(project, analysis)
    sid = manifest.shots[0].id
    with pytest.raises(ValueError, match="keep it under"):
        policy.reject(
            manifest, sid, "x" * (policy.MAX_REVIEW_NOTE_CHARS + 1), AssetKind.VIDEO, project.config
        )
    assert manifest.shots[0].video.director_note is None

    policy.reject(manifest, sid, "  warmer light \n", AssetKind.STILL, project.config)
    assert manifest.shots[0].still.director_note == "warmer light"
    assert manifest.shots[0].review.note == "warmer light"

    with pytest.raises(MVDError, match="keep it under"):
        api.reject(project.root, sid, "y" * 5000)


# ------------------------------------------------------------------ MCP HTTP slow clients


@pytest.fixture
def slow_read_server(tmp_path: Path):
    server = MCPServer(allowed_roots=[tmp_path])
    httpd = _MCPHTTPServer(
        ("127.0.0.1", 0),
        server,
        token="tok",
        loopback_only=True,
        max_concurrent_requests=4,
        read_deadline=0.4,
    )
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield httpd
    finally:
        httpd.shutdown()
        httpd.server_close()


def _ping(port: int) -> int:
    return httpx.post(
        f"http://127.0.0.1:{port}/mcp",
        json={"jsonrpc": "2.0", "id": 1, "method": "ping"},
        headers={"Authorization": "Bearer tok"},
        timeout=5.0,
    ).status_code


def test_mcp_http_slow_clients_cannot_hold_request_slots(slow_read_server):
    port = slow_read_server.server_address[1]
    holders = []
    for i in range(4):
        sock = socket.create_connection(("127.0.0.1", port))
        if i % 2:
            # Trickled body after complete headers.
            sock.sendall(
                b"POST /mcp HTTP/1.1\r\nHost: 127.0.0.1\r\nAuthorization: Bearer tok\r\n"
                b"Content-Length: 500\r\n\r\n{"
            )
        else:
            sock.sendall(b"POST /mcp HTTP/1.1\r\nHost: 127.0.0.1\r\n")  # trickled headers
        holders.append(sock)
    time.sleep(0.1)
    assert _ping(port) == 503  # every slot is held

    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        for sock in holders:
            with contextlib.suppress(OSError):
                sock.sendall(b" ")  # keep each recv() well inside the socket timeout
        time.sleep(0.1)
        if _ping(port) == 200:
            break
    else:
        pytest.fail("slow clients kept every request slot past the read deadline")
    for sock in holders:
        sock.close()


def test_mcp_http_rejects_unauthenticated_post_before_reading_body(slow_read_server):
    port = slow_read_server.server_address[1]
    with socket.create_connection(("127.0.0.1", port)) as sock:
        sock.settimeout(2.0)
        # Announces a body it never sends; the 401 must not wait for it (or the deadline).
        sock.sendall(b"POST /mcp HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Length: 900000\r\n\r\n")
        started = time.monotonic()
        reply = sock.recv(200)
    assert reply.startswith(b"HTTP/1.0 401")
    assert time.monotonic() - started < 0.3


def test_mcp_http_fast_requests_are_unaffected_by_read_deadline(slow_read_server):
    port = slow_read_server.server_address[1]
    assert all(_ping(port) == 200 for _ in range(20))
    assert slow_read_server._pending_reads == {}


def test_precheck_http_request_matches_handler_guards():
    ok = precheck_http_request(
        "POST", "/mcp", {"Host": "127.0.0.1", "Authorization": "Bearer t"}, token="t"
    )
    assert ok is None
    assert precheck_http_request("GET", "/health", {}, token="t") is None
    assert precheck_http_request("POST", "/x", {}, token=None)[0] == 404
    assert precheck_http_request("POST", "/mcp", {"Host": "evil.test"}, token=None)[0] == 403
    assert precheck_http_request("POST", "/mcp", {"Origin": "https://evil.test"})[0] == 403
    assert precheck_http_request("POST", "/mcp", {"Authorization": "Bearer x"}, token="t")[0] == 401


def test_mcp_http_deadline_closed_request_is_dropped_quietly(slow_read_server, capfd):
    port = slow_read_server.server_address[1]
    with socket.create_connection(("127.0.0.1", port)) as sock:
        sock.settimeout(3.0)
        sock.sendall(b"POST /mcp HTTP/1.1\r\nHost: 127.0.0.1\r\n")  # headers never finish
        assert sock.recv(200) == b""  # closed by the deadline, no truncated-request reply
    time.sleep(0.2)
    assert _ping(port) == 200
    assert "Traceback" not in capfd.readouterr().err


# ------------------------------------------------------------------ cleanup


def test_clean_never_deletes_non_generation_files_named_by_stale_attempts(project, analysis):
    from mvdirector.models import AssetStatus, Attempt, QCReport, RetryStrategy

    manifest = _planned(project, analysis)
    shot = manifest.shots[0]
    winner = project.asset_path(shot, AssetKind.STILL, 3)
    loser = project.asset_path(shot, AssetKind.STILL, 2)
    winner.parent.mkdir(parents=True, exist_ok=True)
    winner.write_bytes(b"win")
    loser.write_bytes(b"lose")
    locked_ref = project.reference_dir / "sheet.png"
    locked_ref.parent.mkdir(parents=True, exist_ok=True)
    locked_ref.write_bytes(b"ref")

    def _att(number: int, path: str) -> Attempt:
        return Attempt(
            number=number,
            strategy=RetryStrategy.NEW_SEED,
            seed=number,
            prompt="p",
            path=path,
            qc=QCReport(passed=number == 3, score=0.5),
        )

    # A hand-edited manifest whose stale, unselected attempts name the user's own files.
    shot.still.attempts = [
        _att(1, "inputs/mix.wav"),
        _att(2, project.rel(loser)),
        _att(4, "project.yaml"),
        _att(5, project.rel(locked_ref)),
        _att(3, project.rel(winner)),
    ]
    shot.still.status = AssetStatus.PASSED
    shot.still.selected_attempt = 3
    project.save_manifest(manifest)

    plan = api.clean(project.root, delete=True)
    assert plan.superseded == [loser]
    assert not loser.exists()
    for kept in (winner, locked_ref, project.path("inputs/mix.wav"), project.path("project.yaml")):
        assert kept.is_file(), kept


# ------------------------------------------------------------------ init


def test_init_keeps_files_the_user_already_put_in_the_folder(tmp_path):
    from mvdirector.models import ProjectMode

    target = tmp_path / "mysong"
    (target / "inputs").mkdir(parents=True)
    (target / "workflows").mkdir()
    (target / "inputs" / "lyrics.txt").write_text("[Verse]\nmy real lyrics\n")
    (target / "inputs" / "style.md").write_text("Dark trip-hop")
    (target / "inputs" / "tracklist.json").write_text('{"tracks": "mine"}')
    (target / "workflows" / "ltx_i2v.bindings.json").write_text('{"tuned": true}')

    api.init_project(target, mode=ProjectMode.MIX)

    assert (target / "inputs" / "lyrics.txt").read_text() == "[Verse]\nmy real lyrics\n"
    assert (target / "inputs" / "style.md").read_text() == "Dark trip-hop"
    assert (target / "inputs" / "tracklist.json").read_text() == '{"tracks": "mine"}'
    assert (target / "workflows" / "ltx_i2v.bindings.json").read_text() == '{"tuned": true}'
    # Everything that was missing still comes from the template.
    assert "mode: mix" in (target / "project.yaml").read_text()
    assert (target / "workflows" / "qwen_t2i.bindings.json").is_file()
    assert (target / "workflows" / "README.md").is_file()


# ------------------------------------------------------------------ size overrides


def test_size_overrides_are_capped_before_any_render(project, analysis):
    from mvdirector.assemble import MAX_OUTPUT_DIMENSION, _resolve_output_size

    _planned(project, analysis)
    huge = MAX_OUTPUT_DIMENSION + 2
    with pytest.raises(ConfigError, match="between 64 and 8192"):
        _resolve_output_size(project, draft=False, width=huge, height=720)
    from mvdirector.generation.upscale import upscale_videos

    with pytest.raises(ConfigError, match="between 64 and 8192"):
        upscale_videos(project, project.load_manifest(), None, None, width=1280, height=100_000)
    assert _resolve_output_size(project, draft=False, width=7680, height=4320) == (7680, 4320)

    server = MCPServer(default_project=project.root, allowed_roots=[project.root])
    resp = server.handle_message(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "mvd_assemble",
                "arguments": {"width": 100_000, "height": 100_000, "background": False},
            },
        }
    )
    assert resp is not None and resp["result"]["isError"] is True
    assert "must be <= 8192" in resp["result"]["content"][0]["text"]


# ------------------------------------------------------------------ run budget on cancel


class _StopRun(Exception):
    pass


@pytest.mark.parametrize("max_concurrency", [1, 3])
def test_cancel_at_attempt_started_refunds_the_unspent_generation(
    project, analysis, sample_png, max_concurrency
):
    import yaml

    from mvdirector.comfy.client import ComfyClient
    from tests.conftest import FakeComfy

    cfg_path = project.root / "project.yaml"
    raw = yaml.safe_load(cfg_path.read_text())
    raw["qc"] = {**raw.get("qc", {}), "vision_enabled": False, "on_missing_dependency": "skip"}
    cfg_path.write_text(yaml.safe_dump(raw))
    _planned(project, analysis)
    sheet = project.reference_dir / "candidate.png"
    sheet.parent.mkdir(parents=True, exist_ok=True)
    sheet.write_bytes(sample_png)
    api.lock_reference(project.root, sheet, single=True)

    fake = FakeComfy(png=sample_png)

    class _Provider:
        from contextlib import contextmanager

        @contextmanager
        def lease(self, proj):
            client = ComfyClient(proj.config.comfy, transport=fake.transport(), sleep=lambda _: 0)
            try:
                yield client
            finally:
                client.close()

    started = 0
    finished: list[int] = []

    def observer(event):
        nonlocal started
        if event.kind == "attempt_started":
            started += 1
            if started >= 3:
                raise _StopRun("stop")  # what the MCP cancel does
        if event.kind == "run_finished":
            finished.append(event.generations_used)

    with pytest.raises(_StopRun):
        api.generate(
            project.root,
            [AssetKind.STILL],
            skip_preflight=True,
            observer=observer,
            comfy_provider=_Provider(),
            max_concurrency=max_concurrency,
        )
    manifest = project.load_manifest()
    recorded = sum(len(s.still.attempts) for s in manifest.shots)
    assert len(fake.queued) == recorded == manifest.total_generations
    assert finished == [recorded]  # the run's count equals what reached the GPU


# ------------------------------------------------------------------ unknown shot ids


def test_unknown_shot_ids_fail_instead_of_silently_doing_nothing(project, analysis):
    manifest = _planned(project, analysis)
    real = manifest.shots[0].id
    with pytest.raises(MVDError, match=r"No shot with id 's01'"):
        api.generate(project.root, [AssetKind.STILL], shots=["s01"], skip_preflight=True)
    with pytest.raises(MVDError, match=r"'\.\./\.\./x'"):
        api.generate(project.root, [AssetKind.STILL], shots=[real, "../../x"])
    with pytest.raises(MVDError, match=r"No shot with id 'nope'"):
        api.approve(project.root, ["nope"])
    assert project.load_manifest().total_generations == 0


def test_bearer_scheme_is_case_insensitive_but_token_is_exact():
    def status(auth: str | None) -> int | None:
        headers = {"Host": "127.0.0.1"}
        if auth is not None:
            headers["Authorization"] = auth
        rejection = precheck_http_request("POST", "/mcp", headers, token="tok")
        return None if rejection is None else rejection[0]

    for ok in ("Bearer tok", "bearer tok", "BEARER tok", "  Bearer   tok  "):
        assert status(ok) is None, ok
    for bad in (None, "", "Bearer", "Bearer ", "Bearer TOK", "Bearer tok2", "Basic tok", "tok"):
        assert status(bad) == 401, bad


def test_mcp_http_wrong_token_with_full_body_still_gets_clean_401(slow_read_server):
    port = slow_read_server.server_address[1]
    body = b'{"jsonrpc":"2.0","id":1,"method":"ping","pad":"' + b"a" * 500_000 + b'"}'
    with socket.create_connection(("127.0.0.1", port)) as sock:
        sock.settimeout(3.0)
        sock.sendall(
            b"POST /mcp HTTP/1.1\r\nHost: 127.0.0.1\r\nAuthorization: Bearer wrong\r\n"
            + f"Content-Length: {len(body)}\r\n\r\n".encode()
            + body
        )
        reply = b""
        while chunk := sock.recv(4096):
            reply += chunk
    assert reply.startswith(b"HTTP/1.0 401")
    assert b"Missing or invalid Bearer token" in reply
    assert slow_read_server._pending_reads == {}
    assert _ping(port) == 200
