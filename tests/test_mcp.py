# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Tests for the mvdirector MCP server (protocol, stdio, HTTP security, jobs, and tools)."""

from __future__ import annotations

import io
import json
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from mvdirector import cli
from mvdirector.comfy.client import ComfyClient
from mvdirector.errors import ConfigError
from mvdirector.mcp import (
    SKILL_RESOURCE_URI,
    JobManager,
    MCPServer,
    WorkspaceGuard,
    build_http_server,
    handle_http_request,
    is_loopback_host,
    serve_stdio,
)
from mvdirector.models import (
    AssetKind,
    AssetState,
    AssetStatus,
    Attempt,
    QCReport,
    RetryStrategy,
)
from mvdirector.planning.planner import build_manifest
from tests.conftest import FakeComfy


class _FakeProvider:
    def __init__(self, fake: FakeComfy) -> None:
        self._fake = fake

    @contextmanager
    def lease(self, project):
        client = ComfyClient(
            project.config.comfy, transport=self._fake.transport(), sleep=lambda _: None
        )
        try:
            yield client
        finally:
            client.close()


def _call_tool(
    server: MCPServer, name: str, arguments: dict[str, Any] | None = None
) -> tuple[bool, dict[str, Any]]:
    resp = server.handle_message(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments or {}},
        }
    )
    assert resp is not None and "result" in resp, resp
    result = resp["result"]
    text = result["content"][0]["text"]
    return bool(result["isError"]), json.loads(text)


# ------------------------------------------------------------------ protocol & stdio


def test_initialize_ping_tools_and_resources(tmp_path: Path):
    server = MCPServer(allowed_roots=[tmp_path])

    init = server.handle_message(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "2025-06-18", "capabilities": {}},
        }
    )
    assert init is not None
    assert init["result"]["protocolVersion"] == "2025-06-18"
    assert init["result"]["serverInfo"]["name"] == "mvdirector"
    assert "tools" in init["result"]["capabilities"]

    # Notifications must return None (no response emitted).
    assert server.handle_message({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None

    # Ping
    ping = server.handle_message({"jsonrpc": "2.0", "id": 2, "method": "ping"})
    assert ping == {"jsonrpc": "2.0", "id": 2, "result": {}}

    # Tools list
    tools_resp = server.handle_message({"jsonrpc": "2.0", "id": 3, "method": "tools/list"})
    assert tools_resp is not None
    tools = tools_resp["result"]["tools"]
    names = {t["name"] for t in tools}
    expected = {
        "mvd_init_project",
        "mvd_status",
        "mvd_generate",
        "mvd_task_status",
        "mvd_task_stop",
    }
    assert expected <= names
    for tool in tools:
        assert tool["inputSchema"]["type"] == "object"
        assert "annotations" in tool

    # Resources list & read (bundled Agent Skill)
    res_list = server.handle_message({"jsonrpc": "2.0", "id": 4, "method": "resources/list"})
    assert res_list is not None
    assert res_list["result"]["resources"][0]["uri"] == SKILL_RESOURCE_URI

    res_read = server.handle_message(
        {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "resources/read",
            "params": {"uri": SKILL_RESOURCE_URI},
        }
    )
    assert res_read is not None
    skill_text = res_read["result"]["contents"][0]["text"]
    assert "mvd_generate" in skill_text and "mvd_task_status" in skill_text

    # Unknown method / tool / resource
    bad_method = server.handle_message({"jsonrpc": "2.0", "id": 6, "method": "nope/method"})
    assert bad_method is not None and bad_method["error"]["code"] == -32601

    bad_tool = server.handle_message(
        {"jsonrpc": "2.0", "id": 7, "method": "tools/call", "params": {"name": "mvd_nonexistent"}}
    )
    assert bad_tool is not None and bad_tool["error"]["code"] == -32602

    bad_res = server.handle_message(
        {"jsonrpc": "2.0", "id": 8, "method": "resources/read", "params": {"uri": "mvdirector://x"}}
    )
    assert bad_res is not None and bad_res["error"]["code"] == -32602


def test_serve_stdio_roundtrip_and_parse_error(tmp_path: Path):
    server = MCPServer(allowed_roots=[tmp_path])
    lines = [
        "",
        json.dumps({"jsonrpc": "2.0", "id": 1, "method": "ping"}),
        json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
        "{not valid json",
    ]
    stdin = io.StringIO("\n".join(lines) + "\n")
    stdout = io.StringIO()

    serve_stdio(server, stdin=stdin, stdout=stdout)

    out_lines = [line for line in stdout.getvalue().splitlines() if line.strip()]
    assert len(out_lines) == 2
    assert json.loads(out_lines[0]) == {"jsonrpc": "2.0", "id": 1, "result": {}}
    assert json.loads(out_lines[1])["error"]["code"] == -32700


# ------------------------------------------------------------------ workspace security


def test_workspace_guard_blocks_escapes_and_symlinks(tmp_path: Path):
    allowed = tmp_path / "workspace"
    outside = tmp_path / "outside"
    allowed.mkdir()
    outside.mkdir()
    secret = outside / "secret.png"
    secret.write_bytes(b"secret")
    proj = allowed / "song"
    proj.mkdir()
    link = proj / "escape.png"
    link.symlink_to(secret)

    guard = WorkspaceGuard(default_project=proj, allowed_roots=[allowed])
    assert guard.resolve_project(None) == proj.resolve()
    resolved_mix = guard.resolve_path(proj, "inputs/mix.wav", label="audio")
    assert resolved_mix == (proj / "inputs/mix.wav").resolve()

    with pytest.raises(ConfigError, match="outside the allowed workspace"):
        guard.resolve_project(str(outside))

    with pytest.raises(ConfigError, match="outside the allowed workspace"):
        guard.resolve_path(proj, "../../outside/secret.png", label="image")

    with pytest.raises(ConfigError, match="outside the allowed workspace"):
        guard.resolve_path(proj, "escape.png", label="image")

    with pytest.raises(ConfigError, match="NUL"):
        guard.resolve_path(proj, "bad\x00path.png", label="image")


# ------------------------------------------------------------------ Streamable HTTP & security


def test_http_server_refuses_non_loopback_without_token(tmp_path: Path):
    server = MCPServer(allowed_roots=[tmp_path])
    assert is_loopback_host("127.0.0.1")
    assert is_loopback_host("localhost")
    assert is_loopback_host("::1")
    assert not is_loopback_host("0.0.0.0")

    with pytest.raises(ConfigError, match="bearer token"):
        build_http_server(server, host="0.0.0.0", port=0)

    httpd = build_http_server(server, host="127.0.0.1", port=0, token="secret-token")
    httpd.server_close()


def test_http_transport_protocol_and_security_guards(tmp_path: Path):
    server = MCPServer(allowed_roots=[tmp_path])
    token = "top-secret-token"
    init_body = json.dumps(
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
    ).encode()

    # /health works without auth
    status, _, raw = handle_http_request(server, "GET", "/health", {}, token=token)
    assert status == 200 and json.loads(raw) == {"status": "ok"}

    # Missing / wrong bearer token -> 401
    status, hdrs, _ = handle_http_request(server, "POST", "/mcp", {}, init_body, token=token)
    assert status == 401 and hdrs.get("WWW-Authenticate") == "Bearer"

    status, _, _ = handle_http_request(
        server,
        "POST",
        "/mcp",
        {"Authorization": "Bearer wrong-token"},
        init_body,
        token=token,
    )
    assert status == 401

    # Cross-origin browser request -> 403
    status, _, _ = handle_http_request(
        server,
        "POST",
        "/mcp",
        {"Authorization": f"Bearer {token}", "Origin": "https://evil.example"},
        init_body,
        token=token,
    )
    assert status == 403

    # DNS rebinding Host header on loopback -> 403
    status, _, _ = handle_http_request(
        server,
        "POST",
        "/mcp",
        {"Authorization": f"Bearer {token}", "Host": "evil.example:8765"},
        init_body,
        token=token,
        loopback_only=True,
    )
    assert status == 403

    # Valid initialize (including loopback Origin) -> 200 + Mcp-Session-Id
    auth = {
        "Authorization": f"Bearer {token}",
        "Host": "127.0.0.1:8765",
        "Origin": "http://localhost:3000",
    }
    status, hdrs, raw = handle_http_request(server, "POST", "/mcp", auth, init_body, token=token)
    assert status == 200
    assert "Mcp-Session-Id" in hdrs
    assert json.loads(raw)["result"]["serverInfo"]["name"] == "mvdirector"

    # Notification -> 202 Accepted
    notif = json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}).encode()
    status, _, raw = handle_http_request(server, "POST", "/mcp", auth, notif, token=token)
    assert status == 202 and raw == b""

    # GET /mcp -> 405, DELETE /mcp -> 200, unknown path -> 404
    status, hdrs, _ = handle_http_request(server, "GET", "/mcp", auth, token=token)
    assert status == 405 and "POST" in hdrs.get("Allow", "")

    status, _, raw = handle_http_request(server, "DELETE", "/mcp", auth, token=token)
    assert status == 200 and json.loads(raw) == {"closed": True}

    status, _, _ = handle_http_request(server, "POST", "/unknown", auth, init_body, token=token)
    assert status == 404

    # Oversized body -> 413
    status, _, _ = handle_http_request(
        server, "POST", "/mcp", auth, b"x" * 1_100_000, token=token
    )
    assert status == 413


# ------------------------------------------------------------------ tools & async jobs


def test_project_status_review_tune_and_clean_tools(project, analysis, sample_png):
    manifest = build_manifest(project.config, analysis, "lyrics", "style", llm=None)
    shot = manifest.shots[0]
    still_path = project.asset_path(shot, AssetKind.STILL, 1)
    still_path.parent.mkdir(parents=True, exist_ok=True)
    still_path.write_bytes(sample_png)
    orphan = still_path.parent / "s999_a01.png"
    orphan.write_bytes(sample_png)
    shot.still = AssetState(
        status=AssetStatus.PASSED,
        selected_attempt=1,
        attempts_allowed=4,
        attempts=[
            Attempt(
                number=1,
                strategy=RetryStrategy.INITIAL,
                seed=42,
                prompt="p",
                path=project.rel(still_path),
                qc=QCReport(passed=True, score=0.92),
            )
        ],
    )
    project.save_analysis(analysis)
    project.save_manifest(manifest)

    server = MCPServer(default_project=project.root, allowed_roots=[project.root.parent])

    # Status
    is_err, status = _call_tool(server, "mvd_status")
    assert not is_err, status
    assert status["total_shots"] == len(manifest.shots)
    assert status["still_counts"]["passed"] == 1

    # Single shot inspection
    is_err, detail = _call_tool(server, "mvd_get_shot", {"shot_id": shot.id})
    assert not is_err, detail
    assert detail["id"] == shot.id
    assert detail["still"]["selected_attempt"] == 1
    assert detail["review"]["decision"] in ("none", "pending")

    is_err, missing = _call_tool(server, "mvd_get_shot", {"shot_id": "s999"})
    assert is_err and "not found" in missing["error"]

    # Manifest summary
    is_err, summary = _call_tool(server, "mvd_get_manifest_summary")
    assert not is_err, summary
    assert summary["analysis"]["tempo_bpm"] > 0

    # QC report & tune
    is_err, qc = _call_tool(server, "mvd_qc_report", {"markdown": True})
    assert not is_err and isinstance(qc["report"], str)

    is_err, tuned = _call_tool(server, "mvd_tune", {"apply": False})
    assert not is_err and tuned["applied_to_yaml"] is False

    # Reject
    is_err, rej = _call_tool(
        server, "mvd_reject", {"shot_id": shot.id, "note": "too dark", "target": "still"}
    )
    assert not is_err and rej["rejected"] is True

    # Clean dry-run vs delete
    is_err, plan_only = _call_tool(server, "mvd_clean", {"delete": False})
    assert not is_err and plan_only["deleted"] is False
    assert str(orphan) in plan_only["orphaned"]
    assert orphan.exists()

    is_err, deleted = _call_tool(server, "mvd_clean", {"delete": True})
    assert not is_err and deleted["deleted"] is True
    assert not orphan.exists()


def test_init_project_via_mcp_tool(tmp_path: Path):
    server = MCPServer(allowed_roots=[tmp_path])
    target = tmp_path / "new-mix"
    is_err, res = _call_tool(server, "mvd_init_project", {"directory": str(target), "mode": "mix"})
    assert not is_err
    assert res["mode"] == "mix"
    assert (target / "project.yaml").is_file()
    assert (target / "inputs" / "tracklist.json").is_file()


def test_lock_reference_and_async_generate(project, analysis, sample_png):
    project.config.llm.enabled = False
    manifest = build_manifest(project.config, analysis, "lyrics", "style", llm=None)
    project.save_manifest(manifest)

    ref_candidate = project.reference_dir / "candidate_01.png"
    ref_candidate.parent.mkdir(parents=True, exist_ok=True)
    ref_candidate.write_bytes(sample_png)

    fake = FakeComfy(png=sample_png)
    server = MCPServer(
        default_project=project.root,
        allowed_roots=[project.root],
        comfy_provider=_FakeProvider(fake),
    )

    # Lock a single-image reference sheet first via MCP tool
    is_err, locked = _call_tool(
        server,
        "mvd_lock_reference",
        {"image": "assets/reference/candidate_01.png", "single": True},
    )
    assert not is_err, locked
    assert locked["sheet"]

    first_shot = manifest.shots[0].id
    is_err, submitted = _call_tool(
        server,
        "mvd_generate",
        {"kinds": ["still"], "shots": [first_shot], "skip_preflight": True, "background": True},
    )
    assert not is_err, submitted
    job_id = submitted["job_id"]
    assert submitted["stage"] == "generate"

    final = server.jobs.wait(job_id, timeout=10)
    assert final["status"] == "succeeded", final
    assert final["generations_used"] >= 1
    assert final["event_count"] >= 2

    is_err, polled = _call_tool(server, "mvd_task_status", {"job_id": job_id, "event_limit": 5})
    assert not is_err and polled["status"] == "succeeded"

    is_err, listed = _call_tool(server, "mvd_task_status")
    assert not is_err and any(j["job_id"] == job_id for j in listed["jobs"])


def test_job_manager_cooperative_cancel():
    jobs = JobManager()
    started = threading.Event()
    proceed = threading.Event()

    def _work(ctx):
        started.set()
        proceed.wait(timeout=5)
        ctx.check_cancelled()
        return {"done": True}

    initial = jobs.submit("generate", "/tmp/proj", _work)
    job_id = initial["job_id"]
    assert started.wait(timeout=2)

    stopping = jobs.stop(job_id)
    assert stopping["cancel_requested"] is True
    proceed.set()

    final = jobs.wait(job_id, timeout=5)
    assert final["status"] == "cancelled"
    assert "cancelled" in (final.get("error") or "").lower()


def test_cli_mcp_help_and_non_loopback_guard():
    runner = CliRunner()
    help_res = runner.invoke(cli.app, ["mcp", "--help"])
    assert help_res.exit_code == 0
    assert "--http" in help_res.output
    assert "--workspace" in help_res.output

    # Refuses non-loopback HTTP bind without --token
    bad = runner.invoke(cli.app, ["mcp", "--http", "--host", "0.0.0.0"])
    assert bad.exit_code == 1
    assert "bearer token" in bad.output.lower()
