# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Regressions from the third white/black-box QA, stress, and security audit."""

from __future__ import annotations

import json
import time
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from mvdirector import api, cli, config, dotenv, setup, tuning
from mvdirector.comfy.client import _ComfyWsProgressListener, _json_dict
from mvdirector.comfy.workflow import WorkflowBindings, WorkflowTemplate
from mvdirector.errors import ComfyJobError, ConfigError, LLMError, MVDError
from mvdirector.generation import preflight
from mvdirector.llm.factory import build_llm
from mvdirector.llm.http import (
    _MAX_ERROR_BYTES,
    HttpJSON,
    error_message,
    json_body,
    origin,
    redact_url,
)
from mvdirector.llm.openai_backend import OpenAIChat
from mvdirector.models import Analysis, AssetKind, TimeSpan
from mvdirector.planning.planner import build_manifest
from mvdirector.review import policy as review_policy
from mvdirector.store import LocalStateStore

# ------------------------------------------------------------------ SSE DoS & bounded error read


def test_post_sse_linear_line_splitting_handles_high_line_count() -> None:
    """250k SSE lines in one chunk must parse in linear O(N) time, not quadratic O(N^2)."""
    n_lines = 250_000
    payload = b"data: x\n" * n_lines
    http = HttpJSON(
        headers={},
        timeout=10.0,
        transport=httpx.MockTransport(
            lambda _r: httpx.Response(
                200, headers={"content-type": "text/event-stream"}, content=payload
            )
        ),
    )
    started = time.perf_counter()
    _, events = http.post_sse("http://127.0.0.1:9999/v1/chat/completions", {})
    elapsed = time.perf_counter() - started
    assert events is not None and len(events) == n_lines
    assert elapsed < 1.5


def test_post_sse_bounds_non_200_error_body() -> None:
    """An oversized HTTP 4xx/5xx body on an SSE request must be capped, never read unbounded."""
    oversized = b"E" * (_MAX_ERROR_BYTES * 4)
    http = HttpJSON(
        headers={},
        timeout=10.0,
        retries=1,
        transport=httpx.MockTransport(
            lambda _r: httpx.Response(
                400, headers={"content-type": "text/plain"}, content=oversized
            )
        ),
    )
    resp, events = http.post_sse("http://127.0.0.1:9999/v1/chat/completions", {})
    assert resp.status_code == 400
    assert events == []
    assert len(resp.content) == _MAX_ERROR_BYTES


# ------------------------------------------------------------------ Deep JSON / YAML RecursionError


def test_deeply_nested_json_does_not_crash_llm_comfy_or_setup() -> None:
    deep_arr = "[" * 12_000 + "]" * 12_000
    deep_obj = '{"a":' * 12_000 + "1" + "}" * 12_000

    # OpenAI SSE parser
    http = HttpJSON(
        headers={},
        timeout=5.0,
        retries=1,
        transport=httpx.MockTransport(lambda _r: httpx.Response(200, content=b"{}")),
    )
    chat = OpenAIChat(
        provider="Test",
        model="m",
        max_tokens=100,
        base_url="http://127.0.0.1:9999/v1",
        http=http,
        json_mode="off",
        token_param="max_tokens",
        key_hint="",
        stream=True,
    )
    with pytest.raises(LLMError, match="no choices"):
        chat._parse_sse([deep_arr])

    # llm.http helpers
    resp_deep = httpx.Response(200, content=deep_arr.encode("utf-8"))
    with pytest.raises(LLMError, match="not JSON"):
        json_body(resp_deep, "Test")
    msg = error_message(httpx.Response(500, content=deep_arr.encode("utf-8")))
    assert msg.startswith("[")

    # ComfyUI JSON & WebSocket helpers
    with pytest.raises(ComfyJobError, match="non-JSON"):
        _json_dict(resp_deep, "GET /history")
    ws = _ComfyWsProgressListener("http://127.0.0.1:8188", "cid", {}, "pid", 0.0, lambda _p: None)
    ws._emit_text_payload(deep_obj.encode("utf-8"))  # must silently ignore, never crash

    # Ollama setup probes
    probe = setup.probe_ollama(
        "http://127.0.0.1:11434",
        transport=httpx.MockTransport(
            lambda _r: httpx.Response(200, content=deep_arr.encode("utf-8"))
        ),
    )
    assert probe.reachable is False
    ok, detail = setup.pull_ollama_model(
        "gemma3:12b",
        base_url="http://127.0.0.1:11434",
        transport=httpx.MockTransport(
            lambda _r: httpx.Response(200, content=deep_arr.encode("utf-8"))
        ),
    )
    assert ok is True and "Pulled" in detail


def test_deeply_nested_yaml_rejected_cleanly_by_setup_and_tuning(tmp_path: Path) -> None:
    proj = tmp_path / "deep"
    proj.mkdir()
    (proj / "project.yaml").write_text("a: " + ("[" * 2500) + ("]" * 2500) + "\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="too deeply"):
        setup.apply_setup_config(proj, setup.SetupOptions(title="Hi"))

    adj = [tuning.Adjustment("prepare.whisper_model", "small", "medium", "r", True)]
    with pytest.raises(ConfigError, match="too deeply"):
        tuning.apply_adjustments_to_yaml(proj / "project.yaml", adj)


def test_deeply_nested_lock_file_does_not_crash_store(tmp_path: Path) -> None:
    store = LocalStateStore(tmp_path)
    store.lock_path.write_text("{" * 12_000 + "}" * 12_000, encoding="utf-8")
    assert store.is_locked is True


# ------------------------------------------------------------------ Env isolation & header safety


def test_llm_config_empty_environ_context_does_not_leak_os_environ(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MVD_LLM_URL", "http://host-leak:1234/v1")
    with pytest.raises(ValidationError, match="llm.base_url is required"):
        config.LLMConfig.model_validate(
            {"provider": "openai_compatible", "model": "local-model"},
            context={"environ": {}},
        )


@pytest.mark.parametrize(
    "bad_key",
    [
        "sk-abc\r\nX-Injected: 1",
        "sk-abc\n",
        "sk-abc\x00def",
        "sk-abc\x7f",
    ],
)
def test_build_llm_rejects_control_characters_in_api_key(bad_key: str) -> None:
    with pytest.raises(ConfigError, match="control characters"):
        build_llm(
            config.LLMConfig(provider="openai", model="gpt-4o"),
            environ={"OPENAI_API_KEY": bad_key},
        )


def test_build_llm_strips_whitespace_around_api_key() -> None:
    client = build_llm(
        config.LLMConfig(provider="openai", model="gpt-4o"),
        environ={"OPENAI_API_KEY": "  sk-clean-key  "},
    )
    assert client is not None
    client.close()


# ------------------------------------------------------------------ Setup URLs, symlinks, wiring


@pytest.mark.parametrize(
    "bad_url",
    [
        "http://127.0.0.1:11434?x=1",
        "http://127.0.0.1:8188#frag",
        "https://user:secret@example.com/path?token=1",
    ],
)
def test_setup_validate_http_url_rejects_query_and_fragment(bad_url: str) -> None:
    with pytest.raises(ConfigError, match="query parameters or fragments") as exc_info:
        setup.validate_http_url(bad_url, "Service URL")
    assert "secret" not in str(exc_info.value)


def test_redact_url_and_origin_handle_malformed_ipv6_urls() -> None:
    malformed = "http://user:supersecret@[invalid:8188/api"
    assert redact_url(malformed) == "<invalid-url>"
    assert origin(malformed) == "<invalid-url>"
    probe = setup.probe_ollama(malformed)
    assert probe.reachable is False
    report = setup.EnvironmentReport(
        python_version="3.12.0",
        platform_label="Test",
        ffmpeg_ok=True,
        ffprobe_ok=True,
        extras=[],
        ollama=probe,
        comfy=setup.probe_comfy(malformed),
    )
    rendered = setup.render_environment_report(report)
    assert "supersecret" not in rendered


def test_cli_setup_rejects_query_string_in_comfy_url(tmp_path: Path) -> None:
    runner = CliRunner()
    res = runner.invoke(
        cli.app,
        [
            "setup",
            str(tmp_path / "p"),
            "--non-interactive",
            "--comfy-url",
            "http://127.0.0.1:8188?foo=bar",
            "--llm-provider",
            "none",
        ],
    )
    assert res.exit_code != 0


def test_setup_and_tuning_refuse_symlinked_project_yaml(tmp_path: Path) -> None:
    outside = tmp_path / "outside.txt"
    outside.write_text("SECRET: [not yaml\n", encoding="utf-8")
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "project.yaml").symlink_to(outside)

    with pytest.raises(ConfigError, match="symlinked") as exc1:
        setup.apply_setup_config(proj, setup.SetupOptions(title="Hi"))
    assert "SECRET" not in str(exc1.value)

    adj = [tuning.Adjustment("prepare.whisper_model", "small", "medium", "r", True)]
    with pytest.raises(ConfigError, match="symlinked") as exc2:
        tuning.apply_adjustments_to_yaml(proj / "project.yaml", adj)
    assert "SECRET" not in str(exc2.value)


def test_init_project_refuses_broken_symlink_project_yaml(tmp_path: Path) -> None:
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "project.yaml").symlink_to(tmp_path / "missing.yaml")
    with pytest.raises(MVDError, match="already contains project.yaml"):
        api.init_project(proj)


def test_setup_check_comfy_wiring_reports_traversal_path_as_issue(tmp_path: Path) -> None:
    proj = tmp_path / "proj"
    api.init_project(proj)
    cfg_path = proj / "project.yaml"
    cfg_path.write_text(
        cfg_path.read_text(encoding="utf-8").replace(
            "qwen_t2i.bindings.json", "../../escape.bindings.json"
        ),
        encoding="utf-8",
    )
    report = setup.check_comfy_wiring(
        proj,
        comfy_transport=httpx.MockTransport(lambda _r: httpx.Response(200, json={})),
    )
    assert report.ok is False
    assert any("escapes" in msg for _, msg in report.issues)


def test_tuning_apply_uses_project_dotenv_during_validation(tmp_path: Path) -> None:
    proj = tmp_path / "proj"
    api.init_project(proj)
    dotenv.write_project_dotenv_var(proj, "MVD_LLM_URL", "http://127.0.0.1:1234/v1")
    setup.apply_setup_config(
        proj,
        setup.SetupOptions(
            llm_provider="openai_compatible", llm_model="local-model", llm_base_url=""
        ),
    )
    adj = [tuning.Adjustment("prepare.whisper_model", "small", "medium", "r", True)]
    applied = tuning.apply_adjustments_to_yaml(proj / "project.yaml", adj)
    assert len(applied) == 1


def test_dotenv_write_rejects_oversized_file_before_writing(tmp_path: Path) -> None:
    dotenv.write_project_dotenv_var(tmp_path, "OPENAI_API_KEY", "sk-keep-me")
    with pytest.raises(ConfigError, match="byte limit"):
        dotenv.write_project_dotenv_var(tmp_path, "MVD_HUGE", "x" * 70_000)
    # Existing secret remains intact and readable.
    assert dotenv.read_project_dotenv(tmp_path) == {"OPENAI_API_KEY": "sk-keep-me"}


# ------------------------------------------------------------------ Loader pruning & preflight


def test_workflow_render_prunes_optional_audio_and_image_on_upscale_and_i2v() -> None:
    upscale_graph = {
        "1": {"class_type": "VHS_LoadVideo", "inputs": {"video": "placeholder.mp4"}},
        "2": {"class_type": "LoadAudio", "inputs": {"audio": "stale.wav"}},
        "3": {"class_type": "LoadImage", "inputs": {"image": "stale.png"}},
        "4": {
            "class_type": "LatentRefine",
            "inputs": {"video": ["1", 0], "audio": ["2", 0], "image": ["3", 0]},
        },
    }
    upscale_bindings = WorkflowBindings(
        workflow="up.api.json",
        inputs={
            "video": [{"node": "1", "field": "video"}],
            "audio": [{"node": "2", "field": "audio"}],
            "image": [{"node": "3", "field": "image"}],
        },
        required=["video"],
    )
    up_rendered = WorkflowTemplate("upscale", upscale_bindings, upscale_graph).render(
        {"video": "clip.mp4"}
    )
    assert "1" in up_rendered
    assert "2" not in up_rendered and "3" not in up_rendered
    assert "audio" not in up_rendered["4"]["inputs"]
    assert "image" not in up_rendered["4"]["inputs"]

    # On i2v ("image" in values), optional unbound "audio" is also pruned.
    i2v_graph = {
        "1": {"class_type": "LoadImage", "inputs": {"image": "still.png"}},
        "2": {"class_type": "LoadAudio", "inputs": {"audio": "stale.wav"}},
        "3": {
            "class_type": "VideoModel",
            "inputs": {"image": ["1", 0], "audio": ["2", 0]},
        },
    }
    i2v_bindings = WorkflowBindings(
        workflow="i2v.api.json",
        inputs={
            "image": [{"node": "1", "field": "image"}],
            "audio": [{"node": "2", "field": "audio"}],
        },
        required=["image"],
    )
    i2v_rendered = WorkflowTemplate("i2v", i2v_bindings, i2v_graph).render({"image": "real.png"})
    assert "1" in i2v_rendered and "2" not in i2v_rendered
    assert "audio" not in i2v_rendered["3"]["inputs"]


def test_preflight_video_only_templates_only_require_ref_when_declared_required(
    tmp_path: Path,
) -> None:
    proj_dir = tmp_path / "proj"
    api.init_project(proj_dir)
    proj = api.open_project(proj_dir)
    analysis = Analysis(
        duration=4.0,
        tempo_bpm=120.0,
        beats=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
        bars=[0.0, 2.0],
        bar_intensity=[0.5, 0.5],
        vocal_windows=[],
    )
    manifest = build_manifest(proj.config, analysis, "[Verse]\nHello world", "style", llm=None)
    g = {
        "1": {"class_type": "LoadImage", "inputs": {"image": "x.png"}},
        "2": {"class_type": "SaveVideo", "inputs": {"video": ["1", 0]}},
    }
    t_opt = WorkflowTemplate(
        "lipsync",
        WorkflowBindings(
            workflow="a.json",
            inputs={"reference_image": [{"node": "1", "field": "image"}]},
            required=[],
        ),
        g,
    )
    t_req = WorkflowTemplate(
        "lipsync",
        WorkflowBindings(
            workflow="a.json",
            inputs={"reference_image": [{"node": "1", "field": "image"}]},
            required=["reference_image"],
        ),
        g,
    )
    t_i2v = WorkflowTemplate("i2v", WorkflowBindings(workflow="b.json", inputs={}, required=[]), g)

    # Video-only with optional reference_image should NOT demand a reference sheet.
    res_opt = preflight.check_assets(proj, manifest, {"lipsync": t_opt, "i2v": t_i2v})
    assert res_opt.problems == []

    # Video-only with REQUIRED reference_image MUST report missing reference sheet.
    res_req = preflight.check_assets(proj, manifest, {"lipsync": t_req, "i2v": t_i2v})
    assert any("No reference sheet is locked" in p for p in res_req.problems)


def test_api_generate_video_only_runs_preflight_for_required_r2v_reference(tmp_path: Path) -> None:
    proj_dir = tmp_path / "proj"
    api.init_project(proj_dir)
    wf_dir = proj_dir / "workflows"
    graph = {
        "1": {"class_type": "LoadImage", "inputs": {"image": "still.png"}},
        "2": {"class_type": "LoadImage", "inputs": {"image": "ref.png"}},
        "3": {"class_type": "SaveVideo", "inputs": {"images": ["1", 0], "ref": ["2", 0]}},
    }
    (wf_dir / "vid.api.json").write_text(json.dumps(graph), encoding="utf-8")
    lip_bindings = {
        "workflow": "vid.api.json",
        "inputs": {
            "image": [{"node": "1", "field": "image"}],
            "reference_image": [{"node": "2", "field": "image"}],
        },
        "required": ["image", "reference_image"],
        "output": {"node": "3", "extensions": [".mp4"]},
    }
    i2v_bindings = {
        "workflow": "vid.api.json",
        "inputs": {"image": [{"node": "1", "field": "image"}]},
        "required": ["image"],
        "output": {"node": "3", "extensions": [".mp4"]},
    }
    (wf_dir / "ltx_lipsync.bindings.json").write_text(json.dumps(lip_bindings), encoding="utf-8")
    (wf_dir / "ltx_i2v.bindings.json").write_text(json.dumps(i2v_bindings), encoding="utf-8")

    proj = api.open_project(proj_dir)
    analysis = Analysis(
        duration=4.0,
        tempo_bpm=120.0,
        beats=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
        bars=[0.0, 2.0],
        bar_intensity=[0.5, 0.5],
        vocal_windows=[TimeSpan(start=0.0, end=4.0)],
    )
    manifest = build_manifest(proj.config, analysis, "[Verse]\nHello world", "style", llm=None)
    proj.save_manifest(manifest)

    with pytest.raises(ConfigError, match="No reference sheet is locked"):
        api.generate(proj_dir, [AssetKind.VIDEO])


def test_review_policy_rejects_nul_bytes_in_note(tmp_path: Path) -> None:
    proj_dir = tmp_path / "proj"
    api.init_project(proj_dir)
    proj = api.open_project(proj_dir)
    analysis = Analysis(
        duration=4.0,
        tempo_bpm=120.0,
        beats=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
        bars=[0.0, 2.0],
        bar_intensity=[0.5, 0.5],
        vocal_windows=[],
    )
    manifest = build_manifest(proj.config, analysis, "[Verse]\nHi", "style", llm=None)
    sid = manifest.shots[0].id
    with pytest.raises(ValueError, match="NUL"):
        review_policy.reject(manifest, sid, "bad\x00note", AssetKind.VIDEO, proj.config)
    with pytest.raises(ValueError, match="NUL"):
        review_policy.approve(manifest, sid, note="bad\x00note")
