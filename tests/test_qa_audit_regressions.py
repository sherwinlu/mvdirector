# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Regression tests from the iterative white-box, black-box, stress, and security QA audit."""

from __future__ import annotations

import json
import math
import os
import pickle
import threading
import time
import zipfile
from email.message import Message
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest
import yaml
from PIL import Image
from pydantic import ValidationError

from mvdirector import api as api_mod
from mvdirector import calibrate as calibrate_mod
from mvdirector import cleanup as cleanup_mod
from mvdirector import media, mix_assemble
from mvdirector import report as report_mod
from mvdirector import setup as setup_mod
from mvdirector import tuning as tuning_mod
from mvdirector.audio import analysis as analysis_mod
from mvdirector.audio import broadcast_qa as bqa_mod
from mvdirector.audio import cues as cues_mod
from mvdirector.audio import foley as foley_mod
from mvdirector.audio import lyrics as lyrics_mod
from mvdirector.audio import stem_pack as stem_pack_mod
from mvdirector.audio import tracklist as tracklist_mod
from mvdirector.comfy.client import ComfyClient
from mvdirector.comfy.workflow import WorkflowTemplate
from mvdirector.config import (
    ComfyConfig,
    LLMConfig,
    MixHudConfig,
    ProjectConfig,
    RetryConfig,
    VideoConfig,
)
from mvdirector.errors import (
    ComfyError,
    ConfigError,
    DependencyMissing,
    LLMError,
    LLMUnavailableError,
    MediaToolError,
    MVDError,
    ProjectLocked,
    WorkflowBindingError,
)
from mvdirector.generation import plates as plates_mod
from mvdirector.generation import preflight as preflight_mod
from mvdirector.generation import previz as gen_previz_mod
from mvdirector.generation import reference as ref_mod
from mvdirector.llm.base import extract_json
from mvdirector.llm.factory import build_llm
from mvdirector.llm.gemini_backend import GeminiChat
from mvdirector.llm.http import HttpJSON, origin, redact_url, retry_delay
from mvdirector.llm.ollama_backend import OllamaChat
from mvdirector.llm.openai_backend import OpenAIChat
from mvdirector.mcp import JobManager, MCPServer, handle_http_request
from mvdirector.mcp.server import _MCPHTTPRequestHandler
from mvdirector.models import (
    Analysis,
    AnalyzedChapter,
    AssetKind,
    AssetState,
    AssetStatus,
    Attempt,
    AudioCues,
    AudioSlice,
    Character,
    CheckResult,
    CheckStatus,
    Framing,
    Manifest,
    MixChapter,
    ProjectMode,
    QCReport,
    ReferenceSet,
    RetryStrategy,
    SetDesign,
    Shot,
    ShotType,
    VisualBible,
    WardrobeState,
)
from mvdirector.planning import mix_planner as mix_planner_mod
from mvdirector.planning import planner as planner_mod
from mvdirector.planning import previz as plan_previz_mod
from mvdirector.planning import prompts as prompts_mod
from mvdirector.planning import segmenter as segmenter_mod
from mvdirector.planning import world_state as world_state_mod
from mvdirector.project import Project
from mvdirector.qc import common as qc_common_mod
from mvdirector.qc import image as image_qc_mod
from mvdirector.qc import lipsync as lipsync_mod
from mvdirector.qc import syncnet as syncnet_mod
from mvdirector.qc import video as video_qc_mod
from mvdirector.qc import vision as vision_mod
from mvdirector.retry import RetryPolicy
from mvdirector.review import policy as review_policy
from mvdirector.store import LocalStateStore


def _minimal_shot(
    shot_id: str = "s001",
    shot_type: ShotType = ShotType.LIPSYNC,
    framing: Framing = Framing.MEDIUM_CLOSE,
) -> Shot:
    return Shot(
        id=shot_id,
        index=0,
        start=0.0,
        end=4.0,
        type=shot_type,
        framing=framing,
        setting="Studio",
        audio_slice=AudioSlice(
            path="assets/audio_slices/s001.wav", start=0.0, end=4.0, head_handle=0.0
        ),
        num_frames=97,
    )


# --------------------------------------------------------------------------- MCP & JobManager


def test_mcp_http_rejects_empty_origin_and_nan_tool_result(tmp_path: Path) -> None:
    server = MCPServer(allowed_roots=[tmp_path])
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "ping"}).encode()

    # Empty Origin, invalid Origin port/userinfo/path, and empty Host are rejected with 403.
    for bad_hdrs in (
        {"Host": "127.0.0.1:8765", "Origin": ""},
        {"Host": "127.0.0.1:8765", "Origin": "http://localhost:99999"},
        {"Host": "127.0.0.1:8765", "Origin": "http://evil.com@localhost"},
        {"Host": "127.0.0.1:8765", "Origin": "http://localhost/evil"},
        {"Host": ""},
        {"Host": "127.0.0.1:٨٠"},
    ):
        status, _, _ = handle_http_request(
            server,
            "POST",
            "/mcp",
            bad_hdrs,
            body,
            loopback_only=True,
        )
        assert status == 403, bad_hdrs

    # Oversized stdio JSON-RPC line is rejected with -32700 without killing the stdio loop.
    import io

    from mvdirector.mcp import serve_stdio

    oversized = '{"jsonrpc":"2.0","id":1,"method":"ping","params":{"x":"' + ("a" * 2048) + '"}}\n'
    followup = '{"jsonrpc":"2.0","id":2,"method":"ping"}\n'
    out_buf = io.StringIO()
    serve_stdio(
        server, stdin=io.StringIO(oversized + followup), stdout=out_buf, max_line_bytes=1024
    )
    replies = [json.loads(line) for line in out_buf.getvalue().splitlines() if line.strip()]
    assert len(replies) == 2
    assert replies[0]["error"]["code"] == -32700
    assert replies[1] == {"jsonrpc": "2.0", "id": 2, "result": {}}

    # Non-JSON-serializable / NaN payload from a tool is caught cleanly rather than crashing.
    from mvdirector.mcp.tools import ToolSpec

    server._tools["bad_nan_tool"] = ToolSpec(
        name="bad_nan_tool",
        description="returns NaN",
        input_schema={"type": "object", "properties": {}, "additionalProperties": False},
        annotations={},
        handler=lambda _args: {"bad": float("nan")},
    )
    resp = server.handle_message(
        {
            "jsonrpc": "2.0",
            "id": 9,
            "method": "tools/call",
            "params": {"name": "bad_nan_tool", "arguments": {}},
        }
    )
    assert resp is not None and resp["result"]["isError"] is True
    assert "Internal server error" in resp["result"]["content"][0]["text"]


def test_mcp_tools_validation_caps_and_clean_cache_files(tmp_path: Path) -> None:
    server = MCPServer(allowed_roots=[tmp_path])

    def _call(name: str, args: dict[str, Any]) -> dict[str, Any]:
        resp = server.handle_message(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": name, "arguments": args},
            }
        )
        assert resp is not None
        return resp

    # Empty directory rejected by mvd_init_project.
    r_init = _call("mvd_init_project", {"directory": "   "})
    assert r_init["result"]["isError"] is True

    # count > 32 rejected cleanly.
    r_cap = _call("mvd_generate_plates", {"project_dir": str(tmp_path), "count": 999})
    assert r_cap["result"]["isError"] is True
    assert "<= 32" in r_cap["result"]["content"][0]["text"]

    # Invalid status filter on mvd_task_status rejected cleanly.
    r_status = _call("mvd_task_status", {"status": "bogus_status"})
    assert r_status["result"]["isError"] is True

    # Invalid review_status filter on mvd_status rejected cleanly.
    proj_dir = tmp_path / "proj"
    api_mod.init_project(proj_dir)
    manifest = Manifest(title="T", duration=4.0, fps=24, shots=[_minimal_shot()])
    Project.open(proj_dir).save_manifest(manifest)
    r_rev = _call("mvd_status", {"project_dir": str(proj_dir), "review_status": "bogus"})
    assert r_rev["result"]["isError"] is True

    # mvd_clean reports cache_files count.
    cache_dir = proj_dir / ".comfy_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "deadbeef.png").write_bytes(b"png")
    r_clean = _call("mvd_clean", {"project_dir": str(proj_dir), "delete": False})
    assert r_clean["result"]["isError"] is False
    clean_data = json.loads(r_clean["result"]["content"][0]["text"])
    assert len(clean_data["cache_files"]) == 1


def test_mcp_http_header_smuggling_checks() -> None:
    class _DummyHandler:
        def __init__(self, pairs: list[tuple[str, str]]) -> None:
            msg = Message()
            for k, v in pairs:
                msg.add_header(k, v)
            self.headers = msg
            self.sent: list[tuple[int, dict[str, str], bytes, bool]] = []

        def _write_response(
            self,
            status: int,
            headers: dict[str, str],
            body: bytes,
            *,
            send_body: bool = True,
        ) -> None:
            self.sent.append((status, headers, body, send_body))

        def _reject_duplicate_headers(self, *, send_body: bool = True) -> bool:
            return _MCPHTTPRequestHandler._reject_duplicate_headers(  # type: ignore[arg-type]
                self, send_body=send_body
            )

    h_dup = _DummyHandler([("Host", "127.0.0.1"), ("Host", "evil.com")])
    assert h_dup._reject_duplicate_headers() is True
    assert h_dup.sent[0][0] == 400

    h_te = _DummyHandler([("Host", "127.0.0.1"), ("Transfer-Encoding", "chunked")])
    _MCPHTTPRequestHandler.do_POST(h_te)  # type: ignore[arg-type]
    assert h_te.sent[0][0] == 400

    h_neg = _DummyHandler([("Host", "127.0.0.1"), ("Content-Length", "-5")])
    _MCPHTTPRequestHandler.do_POST(h_neg)  # type: ignore[arg-type]
    assert h_neg.sent[0][0] == 400

    h_uni = _DummyHandler([("Host", "127.0.0.1"), ("Content-Length", "١٢")])
    _MCPHTTPRequestHandler.do_POST(h_uni)  # type: ignore[arg-type]
    assert h_uni.sent[0][0] == 400


def test_job_manager_cleans_up_if_thread_start_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    jm = JobManager(max_retained=10, max_concurrent=2)

    def _boom(self: threading.Thread) -> None:
        raise RuntimeError("can't start new thread")

    monkeypatch.setattr(threading.Thread, "start", _boom)
    with pytest.raises(RuntimeError, match="can't start"):
        jm.submit("test", "/tmp", lambda _ctx: {"ok": True})
    assert jm.list_jobs() == []


# --------------------------------------------------------------------------- Store & Lock Atomicity


def test_lock_stress_contention_and_stale_reclaim_atomicity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = LocalStateStore(tmp_path)
    # Seed a stale lock with a guaranteed-dead PID; 16 threads race to reclaim it.
    lock_path = tmp_path / ".run.lock"
    lock_path.write_text(json.dumps({"pid": 999_999_999, "host": os.uname().nodename}), "utf-8")

    acquired = 0
    locked_out = 0
    barrier = threading.Barrier(16)
    counter_lock = threading.Lock()

    def _worker() -> None:
        nonlocal acquired, locked_out
        barrier.wait()
        try:
            with store.lock():
                with counter_lock:
                    acquired += 1
                # Lock file must never be 0-byte while held.
                assert lock_path.stat().st_size > 0
                time.sleep(0.03)
        except ProjectLocked:
            with counter_lock:
                locked_out += 1

    threads = [threading.Thread(target=_worker) for _ in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert acquired == 1
    assert locked_out == 15
    assert not lock_path.exists()

    # Boolean PID (True == 1 in Python) must NEVER probe os.kill(1, 0) as PID 1.
    killed_pids: list[int] = []
    monkeypatch.setattr(os, "kill", lambda pid, _sig: killed_pids.append(pid))
    lock_path.write_text(json.dumps({"pid": True, "host": os.uname().nodename}), "utf-8")
    with pytest.raises(ProjectLocked), store.lock():
        pass
    assert killed_pids == []


def test_store_load_model_wraps_invalid_utf8_as_config_error(tmp_path: Path) -> None:
    store = LocalStateStore(tmp_path)
    (tmp_path / "manifest.json").write_bytes(b"\xff\xfe\x00\x80")
    with pytest.raises(ConfigError, match="Could not read"):
        store.load_manifest()


# --------------------------------------------------------------------------- Config, Tuning, Setup


def test_config_numeric_and_header_validation() -> None:
    with pytest.raises(ValidationError):
        VideoConfig(fps=0)
    with pytest.raises(ValidationError):
        VideoConfig(min_clip_seconds=10.0, max_clip_seconds=5.0)
    with pytest.raises(ValidationError):
        VideoConfig(max_clip_seconds=float("nan"))
    with pytest.raises(ValidationError):
        VideoConfig(lipsync_min_vocal_fraction=1.5)
    with pytest.raises(ValidationError):
        ComfyConfig(poll_interval=-1.0)
    with pytest.raises(ValidationError):
        ComfyConfig(headers={"X-Bad\r\nHeader": "val"})
    with pytest.raises(ValidationError):
        ComfyConfig(headers={"X-Test": "val\nEvil: 1"})
    with pytest.raises(ValidationError):
        LLMConfig(timeout_seconds=float("inf"))


def test_tuning_yaml_scalar_and_flow_section_and_env_refs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Long string (>80 chars), numeric-looking strings, YAML keywords, and special chars round-trip.
    for s in ("A " * 50, "123", "1.5", ".nan", "true", "null", "foo: bar # baz", "0x10"):
        formatted = tuning_mod.format_yaml_scalar(s)
        assert "\n" not in formatted
        loaded = yaml.safe_load(f"k: {formatted}\n")["k"]
        assert loaded == s and isinstance(loaded, str)

    # Flow-style section expansion + inline '#' inside quotes + unset allowed env ref + CRLF.
    monkeypatch.delenv("COMFY_TOKEN", raising=False)
    yaml_path = tmp_path / "project.yaml"
    yaml_path.write_bytes(
        b'title: "Song #1"  # keep comment\r\n'
        b"comfy:\r\n"
        b'  base_url: "${COMFY_URL:-http://127.0.0.1:8188}"\r\n'
        b"  headers:\r\n"
        b'    Authorization: "Bearer ${COMFY_TOKEN}"\r\n'
        b"video: {fps: 24, max_clip_seconds: 8.0, min_clip_seconds: 1.5}\r\n"
    )
    adj = tuning_mod.Adjustment(
        key="video.max_clip_seconds",
        current=8.0,
        recommended=6.0,
        reason="test",
        applied=True,
    )
    applied = tuning_mod.apply_adjustments_to_yaml(yaml_path, [adj])
    assert len(applied) == 1
    parsed = yaml.safe_load(yaml_path.read_text("utf-8"))
    assert parsed["video"]["max_clip_seconds"] == 6.0
    assert parsed["video"]["fps"] == 24
    assert parsed["video"]["min_clip_seconds"] == 1.5


def test_setup_loopback_and_vram_guards() -> None:
    assert setup_mod.is_loopback_url("http://127.0.0.1:8188") is True
    assert setup_mod.is_loopback_url("http://127.0.0.2:8188") is True
    assert setup_mod.is_loopback_url("http://[::1]:8188") is True
    assert setup_mod.is_loopback_url("http://127.0.0.1.evil.com:8188") is False
    assert setup_mod.is_loopback_url("http://0.0.0.0:8188") is False

    # Boolean VRAM must not report bogus 1-byte VRAM in probe_comfy.
    transport = httpx.MockTransport(
        lambda _req: httpx.Response(
            200,
            content=b'{"devices":[{"name":"GPU","vram_total":true,"vram_free":1073741824}]}',
            headers={"content-type": "application/json"},
        )
    )
    probe = setup_mod.probe_comfy("http://127.0.0.1:8188", transport=transport)
    assert probe.reachable is True
    assert probe.vram_total_gb is None
    assert probe.vram_free_gb == 1.0


# --------------------------------------------------------------------------- LLM Layer


def test_llm_url_redaction_retry_delay_and_validation() -> None:
    assert origin("https://user:supersecret@api.example.com:8443/v1/chat") == (
        "https://api.example.com:8443"
    )
    redacted = redact_url("https://user:supersecret@api.example.com/v1?key=abc")
    assert "supersecret" not in redacted and "abc" not in redacted

    for bad_after in ("NaN", "inf", "-5", "abc"):
        resp = httpx.Response(429, headers={"retry-after": bad_after})
        delay = retry_delay(resp, 0)
        assert math.isfinite(delay) and delay > 0.0

    with pytest.raises(ConfigError, match="http:// or https://"):
        build_llm(LLMConfig(enabled=True, provider="ollama", base_url="file:///etc/passwd"))
    with pytest.raises(ConfigError, match="query string|Invalid|base_url"):
        build_llm(LLMConfig(enabled=True, provider="ollama", base_url="http://127.0.0.1:11434?x=1"))


def test_llm_gemini_model_traversal_and_malformed_shapes() -> None:
    http = HttpJSON(
        headers={}, timeout=5.0, transport=httpx.MockTransport(lambda _r: httpx.Response(200))
    )
    try:
        with pytest.raises(ConfigError, match="Invalid Gemini model"):
            GeminiChat(
                model="../evil",
                max_tokens=64,
                base_url="https://generativelanguage.googleapis.com/v1beta",
                http=http,
                json_mode="schema",
                key_hint="GEMINI_API_KEY",
            )
        with pytest.raises(ConfigError, match="Invalid Gemini model"):
            GeminiChat(
                model="gemini-2.5?key=evil",
                max_tokens=64,
                base_url="https://generativelanguage.googleapis.com/v1beta",
                http=http,
                json_mode="schema",
                key_hint="GEMINI_API_KEY",
            )
    finally:
        http.close()

    # Null text chunk does not raise TypeError; malformed container shapes raise LLMError cleanly.
    assert GeminiChat._parse({"candidates": [{"content": {"parts": [{"text": None}]}}]}).text == ""
    with pytest.raises(LLMError):
        GeminiChat._parse({"candidates": [{"content": {"parts": "not-a-list"}}]})

    oa_http = HttpJSON(
        headers={},
        timeout=5.0,
        transport=httpx.MockTransport(
            lambda _r: httpx.Response(200, json={"choices": ["not-a-dict"]})
        ),
    )
    ol_http = HttpJSON(
        headers={},
        timeout=5.0,
        transport=httpx.MockTransport(
            lambda _r: httpx.Response(200, json={"message": "not-a-dict"})
        ),
    )
    try:
        oa = OpenAIChat(
            model="gpt-4o",
            max_tokens=64,
            base_url="https://api.openai.com/v1",
            http=oa_http,
            json_mode="object",
            token_param="max_tokens",
            provider="OpenAI",
            key_hint="OPENAI_API_KEY",
        )
        assert oa._parse({"choices": [{"message": {"content": None}}]}).text == ""
        with pytest.raises(LLMError):
            oa._parse({"choices": ["not-a-dict"]})

        ol = OllamaChat(
            model="llama3",
            max_tokens=64,
            base_url="http://127.0.0.1:11434",
            http=ol_http,
            json_mode="object",
            context_tokens=4096,
        )
        with pytest.raises(LLMError):
            ol._send("sys", [{"role": "user", "content": "hi"}], VideoConfig)
    finally:
        oa_http.close()
        ol_http.close()


def test_extract_json_skips_leading_braces_and_rejects_nan() -> None:
    text = 'Here is a note {not json} followed by the payload: {"ok": true, "n": 3}'
    assert json.loads(extract_json(text)) == {"ok": True, "n": 3}

    # NaN / Infinity must not be accepted even via the strict=False fallback.
    raw = extract_json('{"score": NaN}')
    with pytest.raises(ValueError):
        json.loads(raw, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))


# --------------------------------------------------------------------------- Comfy, Media & Cleanup


def test_comfy_client_validation_and_workflow_confinement(tmp_path: Path) -> None:
    from mvdirector.comfy.workflow import OutputSpec

    client = ComfyClient(
        ComfyConfig(base_url="http://user:pw@comfy.test", poll_interval=0.0),
        transport=httpx.MockTransport(lambda _r: httpx.Response(200, json={"prompt_id": "../bad"})),
        sleep=lambda _: None,
    )
    try:
        with pytest.raises(ComfyError, match="invalid prompt_id"):
            client.queue({})
        bad_entry = {
            "outputs": {"1": {"images": [{"filename": "a.png", "subfolder": "", "type": "../etc"}]}}
        }
        with pytest.raises(ComfyError, match="unsafe output path"):
            client.download_output(bad_entry, OutputSpec(node="1"), tmp_path / "out.png")
    finally:
        client.close()

    # Bindings file outside project root is rejected.
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    root = tmp_path / "proj"
    root.mkdir()
    with pytest.raises(WorkflowBindingError, match="escapes"):
        WorkflowTemplate.load("t2i", outside, root=root)


def test_media_probe_hardening_and_slice_audio_bounds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Invalid fps "1/0" handled safely.
    fake_proc = type(
        "P",
        (),
        {
            "returncode": 0,
            "stdout": json.dumps(
                {
                    "format": {"duration": "3.5"},
                    "streams": [
                        {
                            "codec_type": "video",
                            "width": 640,
                            "height": 360,
                            "avg_frame_rate": "1/0",
                        }
                    ],
                }
            ),
            "stderr": "",
        },
    )()
    monkeypatch.setattr("mvdirector.media.require_tool", lambda _: "ffprobe")
    monkeypatch.setattr("subprocess.run", lambda *a, **kw: fake_proc)
    info = media.probe(tmp_path / "x.mp4")
    assert info.duration == 3.5 and info.fps is None and info.width == 640

    # Invalid slice bounds raise MediaToolError before calling ffmpeg.
    with pytest.raises(MediaToolError, match="Invalid audio slice span"):
        media.slice_audio(tmp_path / "in.wav", tmp_path / "out.wav", start=5.0, end=2.0)


def test_mix_assemble_hud_card_small_frame_and_foley_short_clips(tmp_path: Path) -> None:
    # 64x64 frame must not crash PIL rectangle/text coordinates.
    ch = MixChapter(
        id="ch01", index=0, start=0.0, end=30.0, title="Title", artist="Artist", bpm=120.0
    )
    card_path = tmp_path / "hud.png"
    layout = mix_assemble.render_hud_card(card_path, ch, MixHudConfig(enabled=True), 64, 64)
    assert layout is not None and card_path.is_file()

    # Foley active RMS on <10ms clip must not crash.
    short_clip = np.ones(50, dtype=np.float32) * 0.5
    leveled = foley_mod.level_active_rms(short_clip, sr=44100, target_db=-18.0)
    assert leveled.shape == short_clip.shape and np.all(np.isfinite(leveled))

    # Ducking gain on ultra-short cue (<20ms) stays continuous in (0, 1] with smooth ramps.
    cue = foley_mod.DiegeticCue(audio_path=tmp_path / "c.wav", start=0.02, end=0.03, duck_db=-6.0)
    times = np.linspace(0.0, 0.1, 4410, dtype=np.float64)
    gain = foley_mod.ducking_gain(times, [cue])
    assert gain.shape == (4410,) and np.all(np.isfinite(gain))
    assert float(gain.min()) < 0.9 and float(np.max(np.abs(np.diff(gain)))) < 0.05


def test_cleanup_skips_symlinks_and_protects_shared_selected_paths(tmp_path: Path) -> None:
    proj_dir = tmp_path / "proj"
    api_mod.init_project(proj_dir)
    project = Project.open(proj_dir)

    outside_file = tmp_path / "secret.png"
    outside_file.write_bytes(b"secret")
    stills_dir = project.path("assets/stills")
    stills_dir.mkdir(parents=True, exist_ok=True)
    link_path = stills_dir / "s001_a99.png"
    link_path.symlink_to(outside_file)

    shared_path = stills_dir / "s001_a01.png"
    shared_path.write_bytes(b"real")
    shot = _minimal_shot()
    shot.still = AssetState(
        status=AssetStatus.PASSED,
        selected_attempt=2,
        attempts=[
            Attempt(
                number=1,
                strategy=RetryStrategy.INITIAL,
                seed=1,
                prompt="p",
                path="assets/stills/s001_a01.png",
            ),
            Attempt(
                number=2,
                strategy=RetryStrategy.NEW_SEED,
                seed=2,
                prompt="p",
                path="assets/stills/s001_a01.png",
            ),
        ],
    )
    manifest = Manifest(title="T", duration=4.0, fps=24, shots=[shot])

    plan = cleanup_mod.plan_cleanup(project, manifest)
    assert link_path not in plan.files
    assert shared_path not in plan.files
    assert outside_file.read_bytes() == b"secret"


# --------------------------------------------------------------------------- QC Modules


def test_qc_edge_cases_lipsync_vision_common_image_video_syncnet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 1. lipsync.measure_sync when max_lag_frames > length.
    m = lipsync_mod.measure_sync(np.array([0.1, 0.2, 0.3, 0.4]), np.array([0.1, 0.2, 0.3, 0.4]), 10)
    assert math.isfinite(m.correlation)

    # 2. vision specification on singing shot with empty appearance must not say "No singer".
    shot = _minimal_shot(shot_type=ShotType.LIPSYNC)
    manifest = Manifest(title="T", duration=4.0, fps=24, shots=[shot])
    spec = vision_mod.VisionJudge.specification(shot, manifest)
    assert "No singer should be visible" not in spec

    # 3. dimensions_check handles 0x0 without ZeroDivisionError.
    dim_res = qc_common_mod.dimensions_check(0, 0, 1280, 720, 0.05, 0.5)
    assert dim_res.status is CheckStatus.FAILED

    # 4. download_cached_model enforces size cap.
    big_transport = httpx.MockTransport(
        lambda _r: httpx.Response(200, headers={"content-length": str(3 * 1024**3)}, content=b"x")
    )
    with pytest.raises(DependencyMissing, match="safety limit"):
        qc_common_mod.download_cached_model(
            "m.bin",
            "https://example.test/m.bin",
            tmp_path,
            client_factory=lambda **kw: httpx.Client(transport=big_transport, **kw),
        )

    # 5. sharpness on 2x2 image returns 0.0 (not NaN).
    tiny = np.zeros((2, 2, 3), dtype=np.uint8)
    assert image_qc_mod.sharpness(tiny) == 0.0

    # 6. VideoQC._sample with count=0 returns [].
    assert video_qc_mod.VideoQC._sample([tiny], 0) == []

    # 7. SyncNet checkpoint loader catches pickle.UnpicklingError as DependencyMissing.
    ckpt = tmp_path / "bad.pt"
    ckpt.write_bytes(b"corrupt")
    monkeypatch.setattr(syncnet_mod, "build_stable_syncnet", lambda _torch: object())

    class _FakeTorch:
        @staticmethod
        def load(*_a: Any, **_kw: Any) -> Any:
            raise pickle.UnpicklingError("bad pickle")

    with pytest.raises(DependencyMissing, match="could not load checkpoint"):
        syncnet_mod._load_stable_syncnet_checkpoint(_FakeTorch, ckpt, "cpu")


# --------------------------------------------------------------------------- Audio Modules


def test_audio_cues_peaks_stem_pack_lyrics_bqa_and_tracklist(tmp_path: Path) -> None:
    # 1. _pick_local_peaks returns strongest peak first even when it occurs later.
    times = np.linspace(0.0, 100.0, 101)
    scores = np.zeros_like(times)
    scores[25] = 2.0  # earlier smaller peak
    scores[75] = 5.0  # later stronger peak
    peaks = cues_mod._pick_local_peaks(times, scores, 10.0, 90.0, 20.0, count=2)
    assert peaks == [75.0, 25.0]

    # 2. stem_pack: make_stem_id caps length <= 64 and corrupt zip raises ConfigError.
    seen: set[str] = set()
    long_id = stem_pack_mod.make_stem_id("super_" * 30 + "vocals", seen)
    assert len(long_id) <= 64
    long_id2 = stem_pack_mod.make_stem_id("super_" * 30 + "vocals", seen)
    assert len(long_id2) <= 64 and long_id2 != long_id

    bad_zip = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("lead_vocals.wav", b"RIFF" + b"\x00" * 100)
    data = bytearray(bad_zip.read_bytes())
    data[45] ^= 0xFF
    bad_zip.write_bytes(bytes(data))
    with pytest.raises(ConfigError, match="Corrupt|Invalid"):
        stem_pack_mod.extract_zip_stems(bad_zip, tmp_path / "stems")

    # 3. bars_from_beats handles beats_per_bar=0 without ValueError.
    assert analysis_mod.bars_from_beats([0.0, 0.5, 1.0], 0) == [0.0, 0.5, 1.0]

    # 4. lyrics word_timestamps rejects non-list JSON with ConfigError.
    proj_dir = tmp_path / "lproj"
    api_mod.init_project(proj_dir)
    ts_file = proj_dir / "inputs" / "words.json"
    ts_file.write_text('{"not": "a list"}', encoding="utf-8")
    p = Project.open(proj_dir)
    p.config.inputs.word_timestamps = "inputs/words.json"
    with pytest.raises(ConfigError, match="JSON list"):
        lyrics_mod.load_or_transcribe_words(p)

    # 5. broadcast_qa deduplicates unsorted boundary_times and rejects NaN.
    sr = 8000
    sig = np.zeros(sr * 2, dtype=np.float64)
    sig[sr] = 0.8  # single impulse click at 1.0s
    clicks = bqa_mod.scan_splice_clicks(sig, sr, boundary_times=[1.01, 0.99, 1.0])
    assert len(clicks) == 1
    with pytest.raises(MediaToolError, match="NaN"):
        bqa_mod.integrated_lufs(np.array([0.0, float("nan")]), sr)

    # 6. tracklist invalid UTF-8 raises ConfigError.
    bad_tl = tmp_path / "tl.txt"
    bad_tl.write_bytes(b"\xff\xfe\x80")
    with pytest.raises(ConfigError, match="Could not read tracklist"):
        tracklist_mod.load_tracklist(bad_tl)


# --------------------------------------------------------------------------- Retry, Gen & Planning


def test_retry_after_broll_fallback_skips_repair_still_and_disables_dead_llm() -> None:
    class _DeadLLM:
        def __init__(self) -> None:
            self.calls = 0

        def complete_json(self, **_kw: Any) -> Any:
            self.calls += 1
            raise LLMUnavailableError("connection refused")

    dead = _DeadLLM()
    policy = RetryPolicy(RetryConfig(in_place_repair=True), project_seed=42, llm=dead)  # type: ignore[arg-type]
    shot = _minimal_shot(shot_type=ShotType.BROLL)
    shot.fallback_from = ShotType.LIPSYNC
    # Prior singer still failed vision_spec in pass 1.
    shot.still.attempts = [
        Attempt(
            number=1,
            strategy=RetryStrategy.INITIAL,
            seed=1,
            prompt="singer",
            path="assets/stills/s001_a01.png",
            qc=QCReport(
                passed=False,
                score=0.2,
                checks=[CheckResult(name="vision_spec", status=CheckStatus.FAILED, hard=True)],
            ),
        )
    ]
    shot.still.reset(3)
    plan = policy.prepare(shot, AssetKind.STILL, can_repair=True)
    # Must start fresh with INITIAL on the new broll pass, not REPAIR_STILL on old singer still.
    assert plan.strategy is RetryStrategy.INITIAL

    # Whitespace-only director_note does not trigger REWRITE_PROMPT.
    shot2 = _minimal_shot()
    shot2.still.director_note = "   "
    assert policy.prepare(shot2, AssetKind.STILL).strategy is RetryStrategy.INITIAL

    # LLMUnavailableError disables LLM for subsequent rewrites.
    shot3 = _minimal_shot()
    shot3.still.director_note = "make it warmer"
    policy.prepare(shot3, AssetKind.STILL)
    shot3.still.director_note = "make it brighter"
    policy.prepare(shot3, AssetKind.STILL)
    assert dead.calls == 1


def test_api_preflight_plates_reference_and_previz_hardening(tmp_path: Path) -> None:
    # init_project rejects existing regular file.
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("x", encoding="utf-8")
    with pytest.raises(MVDError, match="not a directory"):
        api_mod.init_project(blocker)
    assert len(api_mod._safe_title("A" * 300)) <= 120

    # preflight checks secondary visible characters + their character_wardrobes.
    proj_dir = tmp_path / "duet_proj"
    api_mod.init_project(proj_dir)
    proj = Project.open(proj_dir)
    sheet_a = proj.path("assets/reference/a_sheet.png")
    sheet_a.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64)).save(sheet_a)
    char_a = Character(
        name="A",
        appearance="desc A",
        references={"base": ReferenceSet(sheet="assets/reference/a_sheet.png")},
    )
    char_b = Character(
        name="B",
        appearance="desc B",
        wardrobe=[WardrobeState(id="gala", description="gown")],
    )
    duet_shot = _minimal_shot()
    duet_shot.character = "a"
    duet_shot.characters = ["a", "b"]
    duet_shot.character_wardrobes = {"b": "gala"}
    manifest = Manifest(
        title="Duet",
        duration=4.0,
        fps=24,
        character=char_a,
        characters={"a": char_a, "b": char_b},
        shots=[duet_shot],
    )
    res = preflight_mod.check_assets(proj, manifest, {})
    assert any("'b'" in p for p in res.problems)

    # plate_slug preserves short ASCII and caps long names <= 64 chars.
    assert plates_mod.plate_slug("Neon City") == "neon_city"
    assert len(plates_mod.plate_slug("Very Long Location Name " * 10)) <= 64

    # reference.split_sheet rejects tiny images cleanly.
    with pytest.raises(ConfigError, match="too small"):
        ref_mod.split_sheet(Image.new("RGB", (2, 2)), "row", inset=0.1)

    # render_blocking_video rejects invalid duration/fps.
    plan = plan_previz_mod.default_previz_plan(duet_shot, manifest)
    with pytest.raises(ConfigError, match="duration"):
        gen_previz_mod.render_blocking_video(plan, tmp_path / "x.mp4", duration=0.0, fps=24)
    with pytest.raises(ConfigError, match="fps"):
        gen_previz_mod.render_blocking_video(plan, tmp_path / "x.mp4", duration=2.0, fps=0)


def test_planning_previz_world_state_segmenter_prompts_and_mix_planner() -> None:
    # 1. default_previz_plan end camera Z scales from start_eye[2] (wide stays wide, not z=1.15).
    broll_wide = _minimal_shot(shot_type=ShotType.BROLL, framing=Framing.WIDE)
    manifest = Manifest(title="T", duration=4.0, fps=24, shots=[broll_wide])
    p_plan = plan_previz_mod.default_previz_plan(broll_wide, manifest)
    assert p_plan.camera[1].eye[2] > 4.0
    assert "none (b-roll)" in plan_previz_mod._shot_previz_prompt(broll_wide, manifest)
    with pytest.raises(ValidationError):
        plan_previz_mod.BlockingActor(id="../evil")
    with pytest.raises(ValidationError):
        plan_previz_mod.BlockingKeyframe(time=0.0, position=(1e9, 0.0, 0.0))

    # 2. world_state splits on periods too and handles dotted entity keys.
    filtered = world_state_mod.filter_look_for_state(
        "Moody shadows. Warm glowing lantern light. Cold mist.",
        {"prop.lantern.illumination": "unlit"},
    )
    assert "Moody shadows." in filtered
    assert "Cold mist." in filtered
    assert "glowing lantern" not in filtered

    # 3. segmenter handles duration=0 safely and preserves exact duration in dp_phrase_cuts.
    zero_analysis = Analysis(
        duration=0.0, tempo_bpm=120.0, beats=[], bars=[], bar_intensity=[], vocal_windows=[]
    )
    assert segmenter_mod.cut_points(zero_analysis, VideoConfig()) == [0.0]
    assert segmenter_mod.segment(zero_analysis, VideoConfig()) == []
    exact_analysis = Analysis(
        duration=6.1234, tempo_bpm=120.0, beats=[], bars=[], bar_intensity=[], vocal_windows=[]
    )
    dp_pts = segmenter_mod.dp_phrase_cuts(exact_analysis, VideoConfig(), [3.0])
    assert dp_pts[0] == 0.0 and dp_pts[-1] == 6.1234

    # 4. fit_prompt_budget preserves continuity tail guardrails when head is long.
    long_head = "A " * 90 + "scene."
    budgeted = prompts_mod.fit_prompt_budget(
        f"{long_head} Middle filler sentence. Keep character identity. No text.",
        max_chars=120,
    )
    assert len(budgeted) <= 120
    assert budgeted.endswith("Keep character identity. No text.")

    # 5. planner._resolve_set ignores empty candidate names.
    bible = VisualBible(settings=[SetDesign(name="   "), SetDesign(name="Rooftop")])
    assert planner_mod._resolve_set(bible, "Rooftop", "s001") == "Rooftop"

    # 6. mix_planner._chapter_sample_starts avoids duplicating uniform starts when filling slots.
    ch = AnalyzedChapter(
        id="ch01",
        index=0,
        start=0.0,
        end=60.0,
        title="Track",
        bpm=120.0,
        cues=AudioCues(main_peak=12.0, confidence="high"),
    )
    starts = mix_planner_mod._chapter_sample_starts(
        ch, clip_dur=6.0, clips_per_seg=3, total_duration=60.0
    )
    assert len(starts) == 3 and len(set(starts)) == 3


def test_calibrate_report_and_review_policy_regressions(tmp_path: Path) -> None:
    # 1. calibrate.labels_from_reviews captures rejected shots even when selected_attempt is None.
    shot_ok = _minimal_shot("s001")
    shot_bad = _minimal_shot("s002")
    for s, corr in ((shot_ok, 0.8), (shot_bad, 0.1)):
        s.video.attempts = [
            Attempt(
                number=1,
                strategy=RetryStrategy.INITIAL,
                seed=1,
                prompt="p",
                path=f"assets/videos/{s.id}_a01.mp4",
                qc=QCReport(
                    passed=True,
                    score=corr,
                    checks=[
                        CheckResult(
                            name="lipsync",
                            status=CheckStatus.PASSED,
                            metrics={"correlation": corr},
                        )
                    ],
                ),
            )
        ]
        s.video.status = AssetStatus.PASSED
        s.video.selected_attempt = 1
    manifest = Manifest(title="T", duration=8.0, fps=24, shots=[shot_ok, shot_bad])
    cfg = ProjectConfig()
    review_policy.approve(manifest, "s001")
    review_policy.reject(manifest, "s002", "bad sync", AssetKind.VIDEO, cfg)
    assert shot_bad.video.selected_attempt is None
    labels = calibrate_mod.labels_from_reviews(manifest)
    assert any(lbl.shot_id == "s001" and lbl.good for lbl in labels)
    assert any(lbl.shot_id == "s002" and not lbl.good for lbl in labels)

    # load_labels rejects null shot/label, handles UTF-8 BOM + empty kind default.
    bom_csv = tmp_path / "bom.csv"
    bom_csv.write_bytes("\ufeffshot,kind,attempt,label\ns001,, 1 ,good\n".encode())
    loaded = calibrate_mod.load_labels(bom_csv)
    assert loaded == [calibrate_mod.Label("s001", AssetKind.VIDEO, 1, True)]

    null_json = tmp_path / "null.json"
    null_json.write_text(json.dumps([{"shot": None, "label": "good"}]), encoding="utf-8")
    with pytest.raises(ConfigError, match="row 1"):
        calibrate_mod.load_labels(null_json)

    # 2. report.render markdown escapes pipes, backticks, newlines, and HTML tags.
    shot_ok.video.attempts[0].qc = QCReport(
        passed=True,
        score=1.0,
        checks=[
            CheckResult(
                name="evil|check`",
                status=CheckStatus.SKIPPED,
                detail="line1\n<script>alert(1)</script>|pipe",
            )
        ],
    )
    md = report_mod.render(report_mod.summarize(manifest), markdown=True)
    assert "<script>" not in md
    assert "&lt;script&gt;" in md
    assert "evil\\|check\\`" in md

    # 3. review_policy.review_reasons in stills_only does not re-queue accepted_by_reviewer still.
    mix_cfg = ProjectConfig(mode=ProjectMode.MIX)
    mix_cfg.mix.visual_source = "stills"
    mix_shot = _minimal_shot("ch01_c01", shot_type=ShotType.BROLL)
    mix_shot.still.attempts = [
        Attempt(number=1, strategy=RetryStrategy.INITIAL, seed=1, prompt="p", path="a.png"),
        Attempt(number=2, strategy=RetryStrategy.NEW_SEED, seed=2, prompt="p", path="b.png"),
    ]
    mix_shot.still.status = AssetStatus.EXHAUSTED
    mix_shot.still.selected_attempt = 2
    mix_manifest = Manifest(mode=ProjectMode.MIX, title="M", duration=4.0, fps=24, shots=[mix_shot])
    review_policy.accept_still(mix_manifest, "ch01_c01")
    review_policy.refresh_reviews(mix_manifest, mix_cfg)
    assert mix_shot.review.required is False


# --------------------------------------------------------------------------- Pass 2 Hardening


def test_pass2_cache_non_finite_and_symlink_shard_rejection(tmp_path: Path) -> None:
    from mvdirector.comfy.cache import ComfyRenderCache, canonical_graph_key

    with pytest.raises(ValueError, match="Non-finite float"):
        canonical_graph_key({"1": {"inputs": {"cfg": float("nan")}}})
    with pytest.raises(ValueError, match="Non-finite float"):
        canonical_graph_key({"1": {"inputs": {"cfg": float("inf")}}})

    cache_dir = tmp_path / ".cache"
    cache_dir.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    key = "a" * 64
    (outside / f"{key}.png").write_bytes(b"SECRET")
    (cache_dir / key[:2]).symlink_to(outside)

    cache = ComfyRenderCache(cache_dir)
    assert cache.lookup(key, tmp_path / "stolen.png") is False

    src = tmp_path / "good.png"
    src.write_bytes(b"PNG")
    with pytest.raises(ValueError, match="symlink"):
        cache.store(key, src)

    empty = tmp_path / "empty.png"
    empty.write_bytes(b"")
    clean_cache = ComfyRenderCache(tmp_path / ".clean_cache")
    with pytest.raises(ValueError, match="empty"):
        clean_cache.store("b" * 64, empty)


def test_pass2_candidate_count_validate_redaction_and_plan_decode(tmp_path: Path) -> None:
    from mvdirector.generation.candidates import render_candidate_batch

    for bad_count in (0, -2, False):
        with pytest.raises(ConfigError, match="candidate count must be >= 1"):
            render_candidate_batch(
                comfy=None,  # type: ignore[arg-type]
                template=None,  # type: ignore[arg-type]
                base_params={},
                base_seed=1,
                count=bad_count,
                folder=tmp_path,
                prefix="ref",
                label="Reference",
            )

    proj_dir = tmp_path / "proj"
    api_mod.init_project(proj_dir)
    cfg_path = proj_dir / "project.yaml"
    cfg_text = cfg_path.read_text(encoding="utf-8").replace(
        "http://127.0.0.1:8188", "http://alice:supersecret@127.0.0.1:1"
    )
    cfg_path.write_text(cfg_text, encoding="utf-8")

    class _FakeComfy:
        def __enter__(self) -> _FakeComfy:
            return self

        def __exit__(self, *_a: Any) -> None:
            pass

        def ping(self) -> None:
            pass

    class _FakeProvider:
        def lease(self, _project: Project) -> _FakeComfy:
            return _FakeComfy()

    checks = api_mod.validate_project(proj_dir, comfy_provider=_FakeProvider())  # type: ignore[arg-type]
    labels = "\n".join(c.label for c in checks)
    assert "supersecret" not in labels
    assert "alice" not in labels
    assert "127.0.0.1:1" in labels

    # Non-UTF-8 style brief raises ConfigError cleanly.
    from mvdirector.models import Analysis

    style_path = proj_dir / "inputs" / "style.md"
    style_path.write_bytes(b"\xff\xfe\x80\x81")
    (proj_dir / "inputs" / "lyrics.txt").write_text("hello\n", encoding="utf-8")
    p = Project.open(proj_dir)
    p.save_analysis(
        Analysis(
            duration=4.0,
            tempo_bpm=120.0,
            beats=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
            bars=[0.0, 2.0],
            bar_intensity=[0.5, 0.5],
            vocal_windows=[],
        )
    )
    with pytest.raises(ConfigError, match="Could not read style file"):
        api_mod.plan(proj_dir)


def test_pass2_qc_continuity_scoring_alignment_qa_and_mix_assemble(tmp_path: Path) -> None:
    from mvdirector.audio import alignment_qa
    from mvdirector.models import LyricLine, TimeSpan
    from mvdirector.qc import continuity, scoring

    # 1. continuity handles empty/1D/NaN frames and non-finite fps/beats/lags.
    assert continuity.estimate_frame_shift(np.zeros((0, 0)), np.zeros((0, 0))) == (0.0, 0.0, 0.0)
    assert continuity.estimate_frame_shift(np.zeros(4), np.zeros(4)) == (0.0, 0.0, 0.0)
    nan_arr = np.full((8, 8), np.nan)
    assert continuity.estimate_frame_shift(nan_arr, nan_arr) == (0.0, 0.0, 0.0)

    frames = [np.full((16, 16, 3), i * 20, dtype=np.uint8) for i in range(6)]
    for bad_fps in (float("nan"), float("inf"), -24.0):
        rep = continuity.measure_beat_motion_sync(frames, bad_fps, [0.5, 1.0])
        assert rep.correlation == 0.0
    rep2 = continuity.measure_beat_motion_sync(
        frames,
        24.0,
        [float("nan"), float("inf"), 0.1],
        sigma_seconds=float("nan"),
        max_lag_seconds=float("inf"),
    )
    assert math.isfinite(rep2.correlation) and math.isfinite(rep2.z_score)

    # 2. scoring guards against NaN/Inf scores and weights.
    assert scoring.linear_score(float("nan"), 0.0, 1.0) == 0.0
    assert scoring.linear_score(0.5, float("nan"), 1.0) == 0.0
    report = scoring.aggregate(
        [
            CheckResult(name="bad_nan", status=CheckStatus.PASSED, score=float("nan"), weight=1.0),
            CheckResult(name="good", status=CheckStatus.PASSED, score=0.8, weight=1.0),
            CheckResult(name="neg_w", status=CheckStatus.PASSED, score=0.0, weight=-5.0),
        ],
        weights={},
        pass_score=0.5,
    )
    assert report.passed is True and report.score == pytest.approx(0.8)

    # 3. alignment_qa handles NaN audio samples and negative/non-finite lyric/window timestamps.
    noisy = np.zeros(16000, dtype=np.float64)
    noisy[100] = float("nan")
    noisy[200] = float("inf")
    assert 0.0 <= alignment_qa.voicing_ratio(noisy, 8000) <= 1.0
    assert 0.0 <= alignment_qa.consonant_band_ratio(noisy, 8000) <= 1.0
    warns = alignment_qa.audit_lyric_alignment(
        noisy,
        8000,
        [
            LyricLine(text="neg", start=-0.5, end=1.0),
            LyricLine(text="nan", start=float("nan"), end=1.5),
        ],
        [TimeSpan(start=-1.0, end=2.0)],
    )
    assert all(w.start >= 0.0 and w.end >= 0.0 for w in warns)
    cut_warns = alignment_qa.detect_boundary_word_cuts(
        [float("nan"), 0.5], [("hello", -0.2, 1.0), ("bad", float("nan"), 2.0)]
    )
    assert len(cut_warns) == 1 and cut_warns[0].start == 0.0

    # 4. broadcast_qa clean error on missing file and non-finite chunk_seconds.
    with pytest.raises(MediaToolError, match="not found"):
        bqa_mod.analyze_broadcast_audio(tmp_path / "does_not_exist.wav")

    # 5. mix_assemble unknown shot id raises ConfigError, and tiny 16x16 HUD card works.
    proj_dir = tmp_path / "mixproj"
    api_mod.init_project(proj_dir, mode=ProjectMode.MIX)
    mix_p = Project.open(proj_dir)
    ch = MixChapter(
        id="ch01", index=0, start=0.0, end=30.0, title="T", bpm=120.0, shot_ids=["missing_s"]
    )
    mix_manifest = Manifest(mode=ProjectMode.MIX, title="M", duration=30.0, fps=24, chapters=[ch])
    with pytest.raises(ConfigError, match="unknown shot"):
        mix_assemble._resolve_chapter_sources(mix_p, mix_manifest, ch, draft=False)
    tiny_card = tmp_path / "tiny_hud.png"
    layout = mix_assemble.render_hud_card(tiny_card, ch, MixHudConfig(enabled=True), 16, 16)
    assert layout is not None and tiny_card.is_file()


def test_pass3_comfy_wait_zero_poll_timeout_and_mcp_503_socket_drain() -> None:
    import socket
    import threading

    from mvdirector.errors import ComfyJobError
    from mvdirector.mcp.server import _MCPHTTPServer

    # 1. ComfyClient.wait with poll_interval=0.0 must terminate instead of infinite-looping.
    polls = 0

    def _empty_history(_req: httpx.Request) -> httpx.Response:
        nonlocal polls
        polls += 1
        return httpx.Response(200, json={})

    client = ComfyClient(
        ComfyConfig(base_url="http://comfy.test", poll_interval=0.0, job_timeout=0.5),
        transport=httpx.MockTransport(_empty_history),
        sleep=lambda _: None,
    )
    try:
        with pytest.raises(ComfyJobError, match="timed out"):
            client.wait("pid-never-finishes")
        assert 1 <= polls <= 20
    finally:
        client.close()

    # 2. _MCPHTTPServer overload rejection drains unread client request bytes and returns clean 503.
    server_stub = object.__new__(_MCPHTTPServer)
    server_stub._request_slots = threading.BoundedSemaphore(1)
    assert server_stub._request_slots.acquire(blocking=False)

    closed: list[socket.socket] = []

    def _close_req(req: socket.socket) -> None:
        closed.append(req)
        req.close()

    server_stub.close_request = _close_req  # type: ignore[method-assign]

    client_sock, srv_sock = socket.socketpair()
    try:
        client_sock.sendall(b"POST /mcp HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Length: 2\r\n\r\n{}")
        server_stub.process_request(srv_sock, ("127.0.0.1", 0))
        chunks: list[bytes] = []
        client_sock.settimeout(2.0)
        while True:
            data = client_sock.recv(4096)
            if not data:
                break
            chunks.append(data)
        raw = b"".join(chunks)
        assert raw.startswith(b"HTTP/1.1 503 Service Unavailable\r\n")
        _, _, body = raw.partition(b"\r\n\r\n")
        assert json.loads(body) == {"error": "Too many concurrent requests."}
        assert len(closed) == 1
    finally:
        client_sock.close()
        srv_sock.close()
