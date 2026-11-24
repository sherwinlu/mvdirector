# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Tests for local environment diagnostics and `mvd setup`."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
import yaml
from typer.testing import CliRunner

from mvdirector import api
from mvdirector.cli import app
from mvdirector.config import CONFIG_FILENAME, load_config
from mvdirector.errors import ConfigError
from mvdirector.models import ProjectMode
from mvdirector.setup import (
    ComfyProbe,
    EnvironmentReport,
    ExtraStatus,
    OllamaProbe,
    SetupOptions,
    apply_setup_config,
    is_loopback_url,
    optimize_setup_options,
    pick_default_ollama_model,
    probe_comfy,
    probe_ollama,
    render_environment_report,
    validate_comfy_token_env,
    validate_http_url,
)

runner = CliRunner()


# ------------------------------------------------------------------ URL validation


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("http://127.0.0.1:8188", "http://127.0.0.1:8188"),
        ("http://127.0.0.1:8188/", "http://127.0.0.1:8188"),
        ("  https://gpu.example.com:8443/api/  ", "https://gpu.example.com:8443/api"),
    ],
)
def test_validate_http_url_accepts_valid(raw: str, expected: str) -> None:
    assert validate_http_url(raw) == expected


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "   ",
        "file:///etc/passwd",
        "ftp://127.0.0.1:8188",
        "http://",
        "127.0.0.1:8188",
        "http://127.0.0.1:8188\nX-Injected: 1",
        "http://127.0.0.1:8188 /bad",
    ],
)
def test_validate_http_url_rejects_invalid(bad: str) -> None:
    with pytest.raises(ConfigError):
        validate_http_url(bad)


# ------------------------------------------------------------------ service probes


def test_probe_ollama_lists_models() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/tags"
        return httpx.Response(
            200,
            json={
                "models": [
                    {"name": "gemma3:12b"},
                    {"model": "qwen2.5vl:7b"},
                    {"name": ""},
                ]
            },
        )

    probe = probe_ollama("http://127.0.0.1:11434/", transport=httpx.MockTransport(handler))
    assert probe.reachable is True
    assert probe.url == "http://127.0.0.1:11434"
    assert probe.models == ["gemma3:12b", "qwen2.5vl:7b"]
    assert probe.error is None


def test_probe_ollama_handles_http_and_transport_errors() -> None:
    err_transport = httpx.MockTransport(lambda _req: httpx.Response(503, text="unavailable"))
    probe_503 = probe_ollama("http://127.0.0.1:11434", transport=err_transport)
    assert probe_503.reachable is False
    assert "503" in (probe_503.error or "")

    bad_json = httpx.MockTransport(lambda _req: httpx.Response(200, text="not-json"))
    probe_bad = probe_ollama("http://127.0.0.1:11434", transport=bad_json)
    assert probe_bad.reachable is False
    assert "non-JSON" in (probe_bad.error or "")

    def raise_conn(_req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    probe_down = probe_ollama("http://127.0.0.1:11434", transport=httpx.MockTransport(raise_conn))
    assert probe_down.reachable is False
    assert "Unreachable" in (probe_down.error or "")

    probe_invalid = probe_ollama("file:///tmp/ollama")
    assert probe_invalid.reachable is False
    assert probe_invalid.error is not None


def test_probe_comfy_extracts_device_and_vram() -> None:
    gib = 1024**3

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/system_stats"
        return httpx.Response(
            200,
            json={
                "system": {"comfyui_version": "0.45.4"},
                "devices": [
                    {
                        "name": "cuda:0 NVIDIA RTX 4090",
                        "vram_total": 24 * gib,
                        "vram_free": 20 * gib,
                    }
                ],
            },
        )

    probe = probe_comfy("http://127.0.0.1:8188/", transport=httpx.MockTransport(handler))
    assert probe.reachable is True
    assert probe.comfy_version == "0.45.4"
    assert probe.devices == ["cuda:0 NVIDIA RTX 4090"]
    assert probe.vram_total_gb == pytest.approx(24.0)
    assert probe.vram_free_gb == pytest.approx(20.0)


def test_probe_comfy_handles_unreachable_and_invalid_url() -> None:
    def raise_conn(_req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    probe_down = probe_comfy("http://127.0.0.1:8188", transport=httpx.MockTransport(raise_conn))
    assert probe_down.reachable is False
    assert probe_down.error is not None

    probe_bad = probe_comfy("ftp://127.0.0.1:8188")
    assert probe_bad.reachable is False
    assert probe_bad.error is not None


# ------------------------------------------------------------------ model selection & report


def test_pick_default_ollama_model() -> None:
    assert pick_default_ollama_model(["llama3.1:8b", "gemma3:12b"]) == "gemma3:12b"
    assert pick_default_ollama_model(["llama3.1:8b", "qwen2.5vl:7b"]) == "qwen2.5vl:7b"
    assert pick_default_ollama_model(["llama3.1:8b"]) == "gemma3:12b"
    assert pick_default_ollama_model([]) == "gemma3:12b"


def test_render_environment_report_healthy_and_missing() -> None:
    healthy = EnvironmentReport(
        python_version="3.12.4",
        platform_label="Darwin arm64",
        ffmpeg_ok=True,
        ffprobe_ok=True,
        extras=[
            ExtraStatus("qc", True, "MediaPipe QC"),
            ExtraStatus("align", True, "Whisper alignment"),
        ],
        ollama=OllamaProbe("http://127.0.0.1:11434", True, ["gemma3:12b"]),
        comfy=ComfyProbe(
            "http://127.0.0.1:8188",
            True,
            devices=["mps"],
            vram_total_gb=32.0,
            vram_free_gb=24.0,
            comfy_version="0.45.4",
        ),
    )
    text_ok = render_environment_report(healthy)
    assert "OK   ffmpeg" in text_ok
    assert "OK   Ollama at http://127.0.0.1:11434 (models: gemma3:12b)" in text_ok
    assert (
        "OK   ComfyUI at http://127.0.0.1:8188 (v0.45.4, mps, VRAM 24.0/32.0 GiB free)" in text_ok
    )

    missing = EnvironmentReport(
        python_version="3.12.4",
        platform_label="Linux x86_64",
        ffmpeg_ok=False,
        ffprobe_ok=False,
        extras=[ExtraStatus("qc", False, "MediaPipe QC")],
        ollama=OllamaProbe("http://127.0.0.1:11434", False, error="Unreachable"),
        comfy=ComfyProbe("http://127.0.0.1:8188", False, error="Unreachable"),
    )
    text_miss = render_environment_report(missing)
    assert "MISS ffmpeg" in text_miss
    assert "brew install ffmpeg" in text_miss
    assert "ollama pull gemma3:12b" in text_miss
    assert "Start ComfyUI locally" in text_miss


# ------------------------------------------------------------------ YAML updater


def test_apply_setup_config_preserves_comments_and_validates(tmp_path: Path) -> None:
    project_dir = tmp_path / "demo-song"
    api.init_project(project_dir)

    apply_setup_config(
        project_dir,
        SetupOptions(
            title="Midnight Signal",
            comfy_url="http://127.0.0.1:8199/",
            llm_enabled=True,
            llm_provider="ollama",
            llm_model="qwen2.5vl:7b",
            qc_on_missing_dependency="skip",
            video_preset="wan",
        ),
    )

    raw_text = (project_dir / CONFIG_FILENAME).read_text(encoding="utf-8")
    assert "# mvdirector project configuration." in raw_text
    assert "# base seed; every attempt's seed is derived from it" in raw_text

    cfg = load_config(project_dir / CONFIG_FILENAME)
    assert cfg.title == "Midnight Signal"
    assert cfg.comfy.base_url == "http://127.0.0.1:8199"
    assert cfg.llm.provider == "ollama"
    assert cfg.llm.model == "qwen2.5vl:7b"
    assert cfg.qc.on_missing_dependency == "skip"
    assert cfg.video.frame_multiple == 4
    assert cfg.video.frame_offset == 1
    assert cfg.video.size_multiple == 64
    assert cfg.video.max_clip_seconds == 8.0


def test_apply_setup_config_rejects_bad_inputs(tmp_path: Path) -> None:
    project_dir = tmp_path / "bad-song"
    api.init_project(project_dir)

    with pytest.raises(ConfigError, match="Unknown video preset"):
        apply_setup_config(project_dir, SetupOptions(video_preset="nonexistent"))

    with pytest.raises(ConfigError, match="must not be empty"):
        apply_setup_config(project_dir, SetupOptions(title="   "))

    with pytest.raises(ConfigError, match="http:// or https://"):
        apply_setup_config(project_dir, SetupOptions(comfy_url="file:///etc/passwd"))

    with pytest.raises(ConfigError, match="Invalid configuration"):
        apply_setup_config(project_dir, SetupOptions(llm_provider="bogus_provider"))


# ------------------------------------------------------------------ CLI tests


def _fake_report(*, healthy: bool = True) -> EnvironmentReport:
    return EnvironmentReport(
        python_version="3.12.4",
        platform_label="Darwin arm64",
        ffmpeg_ok=healthy,
        ffprobe_ok=healthy,
        extras=[
            ExtraStatus("qc", healthy, "MediaPipe QC"),
            ExtraStatus("align", healthy, "Whisper alignment"),
        ],
        ollama=OllamaProbe(
            "http://127.0.0.1:11434",
            healthy,
            ["gemma3:12b"] if healthy else [],
            None if healthy else "Unreachable",
        ),
        comfy=ComfyProbe(
            "http://127.0.0.1:8188",
            healthy,
            devices=["mps"] if healthy else [],
            vram_total_gb=32.0 if healthy else None,
            vram_free_gb=24.0 if healthy else None,
            error=None if healthy else "Unreachable",
        ),
    )


def test_cli_setup_check_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api, "inspect_environment", lambda **_kw: _fake_report(healthy=True))
    res_ok = runner.invoke(app, ["setup", "--check-only"])
    assert res_ok.exit_code == 0
    assert "OK   ComfyUI" in res_ok.stdout

    monkeypatch.setattr(api, "inspect_environment", lambda **_kw: _fake_report(healthy=False))
    res_fail = runner.invoke(app, ["setup", "--check-only"])
    assert res_fail.exit_code == 1
    assert "MISS ffmpeg" in res_fail.stdout


def test_cli_setup_non_interactive_creates_and_configures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(api, "inspect_environment", lambda **_kw: _fake_report(healthy=True))
    target = tmp_path / "neon-drive"

    res = runner.invoke(
        app,
        [
            "setup",
            str(target),
            "--non-interactive",
            "--title",
            "Neon Drive",
            "--comfy-url",
            "http://127.0.0.1:8188",
            "--video-preset",
            "wan",
        ],
    )
    assert res.exit_code == 0, res.output
    assert "Created and configured" in res.stdout
    assert "infinitetalk_lipsync.bindings.json" in res.stdout

    cfg = load_config(target / CONFIG_FILENAME)
    assert cfg.title == "Neon Drive"
    assert cfg.mode is ProjectMode.STANDARD
    assert cfg.llm.provider == "ollama"
    assert cfg.llm.model == "gemma3:12b"
    assert cfg.video.frame_multiple == 4
    assert cfg.video.size_multiple == 64


def test_cli_setup_interactive_flow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from mvdirector.llm.check import LLMCheckResult
    from mvdirector.setup import ComfyWiringReport

    monkeypatch.setattr(api, "inspect_environment", lambda **_kw: _fake_report(healthy=True))
    monkeypatch.setattr(
        api,
        "verify_llm_config",
        lambda *_a, **_kw: LLMCheckResult(json_ok=True, vision="ok"),
    )
    monkeypatch.setattr(
        api,
        "check_comfy_wiring",
        lambda _dir: ComfyWiringReport(
            url="http://127.0.0.1:8188",
            reachable=True,
            node_count=42,
            missing_exports=["t2i", "still", "lipsync", "i2v"],
        ),
    )
    target = tmp_path / "starlight"

    # Prompts: confirm create, title, comfy url, llm provider, ollama model, video preset
    answers = "y\nStarlight Echo\nhttp://127.0.0.1:8188\nollama\ngemma3:12b\nltx\n"
    res = runner.invoke(app, ["setup", str(target)], input=answers)
    assert res.exit_code == 0, res.output
    assert "LLM JSON check passed" in res.stdout
    assert "42 node classes available" in res.stdout

    cfg = load_config(target / CONFIG_FILENAME)
    assert cfg.title == "Starlight Echo"
    assert cfg.video.frame_multiple == 8
    assert cfg.video.size_multiple == 64
    assert cfg.llm.model == "gemma3:12b"


def test_cli_setup_rejects_invalid_provider_and_missing_cloud_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(api, "inspect_environment", lambda **_kw: _fake_report(healthy=True))
    target = tmp_path / "cloud-test"

    bad_provider = runner.invoke(
        app, ["setup", str(target), "--non-interactive", "--llm-provider", "invalid"]
    )
    assert bad_provider.exit_code == 1

    missing_model = runner.invoke(
        app, ["setup", str(target), "--non-interactive", "--llm-provider", "anthropic"]
    )
    assert missing_model.exit_code == 1
    assert "--llm-model is required" in missing_model.output


# ------------------------------------------------------------------ remote ComfyUI server


def test_is_loopback_url_and_token_env_validation() -> None:
    assert is_loopback_url("http://127.0.0.1:8188") is True
    assert is_loopback_url("http://localhost:8188") is True
    assert is_loopback_url("http://[::1]:8188") is True
    assert is_loopback_url("http://host.docker.internal:8188") is True
    assert is_loopback_url("http://192.168.1.50:8188") is False
    assert is_loopback_url("https://gpu.example.com:8188") is False

    assert validate_comfy_token_env("COMFY_TOKEN") == "COMFY_TOKEN"
    assert validate_comfy_token_env("MVD_COMFY_KEY") == "MVD_COMFY_KEY"
    for bad in ("", "   ", "AWS_SECRET_ACCESS_KEY", "GITHUB_TOKEN", "BAD VAR"):
        with pytest.raises(ConfigError):
            validate_comfy_token_env(bad)


def test_probe_comfy_sends_bearer_auth_header(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COMFY_TOKEN", "secret-remote-tok")
    seen_auth: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_auth.append(request.headers.get("authorization"))
        return httpx.Response(
            200,
            json={"devices": [{"name": "cuda:0 A100", "vram_total": 80 * 1024**3}]},
        )

    probe = probe_comfy(
        "https://gpu.example.com:8188",
        headers={"Authorization": "Bearer ${COMFY_TOKEN}"},
        transport=httpx.MockTransport(handler),
    )
    assert probe.reachable is True
    assert seen_auth == ["Bearer secret-remote-tok"]
    assert probe.devices == ["cuda:0 A100"]


def test_apply_setup_config_remote_comfy_with_unset_token_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("COMFY_TOKEN", raising=False)
    project_dir = tmp_path / "remote-song"
    api.init_project(project_dir)

    apply_setup_config(
        project_dir,
        SetupOptions(
            comfy_url="https://gpu.example.com:8188/",
            comfy_token_env="COMFY_TOKEN",
        ),
    )

    raw_text = (project_dir / CONFIG_FILENAME).read_text(encoding="utf-8")
    assert 'headers: {Authorization: "Bearer ${COMFY_TOKEN}"}' in raw_text

    # Once exported, load_config resolves the Bearer header cleanly.
    monkeypatch.setenv("COMFY_TOKEN", "live-token")
    cfg = load_config(project_dir / CONFIG_FILENAME)
    assert cfg.comfy.base_url == "https://gpu.example.com:8188"
    assert cfg.comfy.headers == {"Authorization": "Bearer live-token"}

    # Passing empty string clears comfy.headers back to {}.
    apply_setup_config(project_dir, SetupOptions(comfy_token_env=""))
    cfg_cleared = load_config(project_dir / CONFIG_FILENAME)
    assert cfg_cleared.comfy.headers == {}


def test_cli_setup_remote_comfy_interactive_and_non_interactive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("COMFY_TOKEN", raising=False)
    probed_urls: list[str | None] = []

    def fake_inspect(**kw: object) -> EnvironmentReport:
        url = kw.get("comfy_url")
        probed_urls.append(url if isinstance(url, str) else None)
        is_remote = url == "https://gpu.example.com:8188"
        return EnvironmentReport(
            python_version="3.12.4",
            platform_label="Darwin arm64",
            ffmpeg_ok=True,
            ffprobe_ok=True,
            extras=[
                ExtraStatus("qc", True, "MediaPipe QC"),
                ExtraStatus("align", True, "Whisper alignment"),
            ],
            ollama=OllamaProbe("http://127.0.0.1:11434", True, ["gemma3:12b"]),
            comfy=ComfyProbe(
                url if isinstance(url, str) else "http://127.0.0.1:8188",
                reachable=is_remote,
                devices=["cuda:0 RTX 6000 Ada"] if is_remote else [],
                vram_total_gb=48.0 if is_remote else None,
                error=None if is_remote else "Connection refused",
            ),
        )

    from mvdirector.llm.check import LLMCheckResult
    from mvdirector.setup import ComfyWiringReport

    monkeypatch.setattr(api, "inspect_environment", fake_inspect)
    monkeypatch.setattr(
        api,
        "verify_llm_config",
        lambda *_a, **_kw: LLMCheckResult(json_ok=True, vision="ok"),
    )
    monkeypatch.setattr(
        api,
        "check_comfy_wiring",
        lambda _dir: ComfyWiringReport(url="https://gpu.example.com:8188", reachable=True),
    )

    # Interactive: enter remote ComfyUI URL + COMFY_TOKEN (leave token secret blank) -> note shown.
    target = tmp_path / "remote-interactive"
    answers = (
        "y\nRemote Track\nhttps://gpu.example.com:8188\nCOMFY_TOKEN\n\nollama\ngemma3:12b\nltx\n"
    )
    res = runner.invoke(app, ["setup", str(target)], input=answers)
    assert res.exit_code == 0, res.output
    assert "Connected to ComfyUI at" in res.stdout
    assert "export COMFY_TOKEN=<your-token>" in res.stdout
    assert probed_urls == [None, "https://gpu.example.com:8188"]

    # Non-interactive with --comfy-url and --comfy-token-env
    target2 = tmp_path / "remote-cli"
    res2 = runner.invoke(
        app,
        [
            "setup",
            str(target2),
            "--non-interactive",
            "--comfy-url",
            "https://gpu.example.com:8188",
            "--comfy-token-env",
            "COMFY_TOKEN",
        ],
    )
    assert res2.exit_code == 0, res2.output
    assert "export COMFY_TOKEN=<your-token>" in res2.stdout
    parsed_yaml = yaml.safe_load((target2 / CONFIG_FILENAME).read_text(encoding="utf-8"))
    assert parsed_yaml["comfy"]["base_url"] == "https://gpu.example.com:8188"
    assert parsed_yaml["comfy"]["headers"] == {"Authorization": "Bearer ${COMFY_TOKEN}"}


def test_setup_regression_multiline_headers_unset_env_and_atomic_init(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("COMFY_TOKEN", raising=False)

    # 1. Invalid options must fail BEFORE creating the project folder.
    not_created = tmp_path / "should-not-exist"
    with pytest.raises(ConfigError):
        api.setup_project(not_created, SetupOptions(comfy_url="file:///etc/passwd"))
    assert not not_created.exists()

    # 2. Replacing a multi-line comfy.headers block must not leave orphaned child lines.
    proj = tmp_path / "multiline"
    api.init_project(proj)
    cfg_path = proj / CONFIG_FILENAME
    raw = cfg_path.read_text(encoding="utf-8")
    raw = raw.replace(
        'headers: {}                    # e.g. {Authorization: "Bearer ${COMFY_TOKEN}"}',
        'headers:\n    Authorization: "Bearer ${COMFY_OLD}"\n    X-Custom: "1"',
        1,
    )
    cfg_path.write_text(raw, encoding="utf-8")

    apply_setup_config(proj, SetupOptions(comfy_token_env="COMFY_TOKEN"))
    updated = cfg_path.read_text(encoding="utf-8")
    assert "COMFY_OLD" not in updated
    assert "X-Custom" not in updated
    assert 'headers: {Authorization: "Bearer ${COMFY_TOKEN}"}' in updated

    # 3. Re-running setup on a project with an unset ${COMFY_TOKEN} succeeds for other fields.
    apply_setup_config(proj, SetupOptions(title="Updated Title"))
    monkeypatch.setenv("COMFY_TOKEN", "tok-after")
    cfg = load_config(cfg_path)
    assert cfg.title == "Updated Title"
    assert cfg.comfy.headers == {"Authorization": "Bearer tok-after"}


def test_probe_comfy_handles_http_auth_error_and_prefixed_version() -> None:
    unauth = httpx.MockTransport(lambda _req: httpx.Response(401, text="Unauthorized"))
    probe = probe_comfy("https://gpu.example.com:8188", transport=unauth)
    assert probe.reachable is False
    assert "401" in (probe.error or "")

    report = EnvironmentReport(
        python_version="3.12.4",
        platform_label="Darwin arm64",
        ffmpeg_ok=True,
        ffprobe_ok=True,
        extras=[],
        ollama=OllamaProbe("http://127.0.0.1:11434", True, ["gemma3:12b"]),
        comfy=ComfyProbe(
            "http://192.168.1.50:8188",
            True,
            devices=["cuda:0 RTX 4090"],
            comfy_version="v0.3.26",
        ),
    )
    rendered = render_environment_report(report)
    assert "(v0.3.26, cuda:0 RTX 4090)" in rendered
    assert "vv0.3.26" not in rendered


def test_setup_exact_preset_and_check_only_non_ollama(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report_no_ollama = EnvironmentReport(
        python_version="3.12.4",
        platform_label="Darwin arm64",
        ffmpeg_ok=True,
        ffprobe_ok=True,
        extras=[ExtraStatus("qc", True, "MediaPipe QC")],
        ollama=OllamaProbe("http://127.0.0.1:11434", False, error="Unreachable"),
        comfy=ComfyProbe("http://192.168.1.50:8188", True, devices=["cuda:0 RTX 4090"]),
    )
    monkeypatch.setattr(api, "inspect_environment", lambda **_kw: report_no_ollama)

    # Default --check-only expects Ollama and fails when unreachable.
    res_ollama = runner.invoke(app, ["setup", "--check-only"])
    assert res_ollama.exit_code == 1

    # Non-Ollama provider does not require local Ollama to pass --check-only.
    res_cloud = runner.invoke(app, ["setup", "--check-only", "--llm-provider", "anthropic"])
    assert res_cloud.exit_code == 0

    # Model-neutral 'exact' preset configures 1n+0 frames and size_multiple=1.
    target = tmp_path / "exact-model"
    res_exact = runner.invoke(
        app,
        [
            "setup",
            str(target),
            "--non-interactive",
            "--comfy-url",
            "http://192.168.1.50:8188",
            "--llm-provider",
            "anthropic",
            "--llm-model",
            "claude-sonnet-4-5",
            "--video-preset",
            "exact",
        ],
    )
    assert res_exact.exit_code == 0, res_exact.output
    cfg = load_config(target / CONFIG_FILENAME)
    assert cfg.video.frame_multiple == 1
    assert cfg.video.frame_offset == 0
    assert cfg.video.size_multiple == 1
    assert cfg.video.max_clip_seconds == 10.0
    assert cfg.comfy.base_url == "http://192.168.1.50:8188"
    assert cfg.comfy.dedup_uploads is True
    assert cfg.comfy.cache_enabled is True
    assert cfg.comfy.sort_batch_for_cache is True


def test_cli_setup_cloud_llm_skips_ollama_verifies_key_and_saves_dotenv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mvdirector import setup as setup_mod
    from mvdirector.dotenv import read_project_dotenv
    from mvdirector.llm.check import LLMCheckResult
    from mvdirector.setup import ComfyWiringReport

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    report_no_ollama = EnvironmentReport(
        python_version="3.12.4",
        platform_label="Darwin arm64",
        ffmpeg_ok=True,
        ffprobe_ok=True,
        extras=[ExtraStatus("qc", True, "MediaPipe QC")],
        ollama=OllamaProbe("http://127.0.0.1:11434", False, error="Unreachable"),
        comfy=ComfyProbe("http://127.0.0.1:8188", True, devices=["mps"]),
    )
    monkeypatch.setattr(api, "inspect_environment", lambda **_kw: report_no_ollama)
    install_called = False

    def fail_if_install(**_kw: object) -> tuple[bool, str]:
        nonlocal install_called
        install_called = True
        return True, "installed"

    monkeypatch.setattr(setup_mod, "install_ollama", fail_if_install)
    seen_keys: list[str | None] = []

    def fake_verify(cfg: object, *, check_vision: bool = True, environ=None) -> LLMCheckResult:
        seen_keys.append((environ or {}).get("OPENAI_API_KEY"))
        return LLMCheckResult(json_ok=True, vision="ok")

    monkeypatch.setattr(api, "verify_llm_config", fake_verify)
    monkeypatch.setattr(
        api,
        "check_comfy_wiring",
        lambda _dir: ComfyWiringReport(url="http://127.0.0.1:8188", reachable=True, node_count=12),
    )

    target = tmp_path / "cloud-openai"
    # Prompts: confirm, title, comfy url, provider (alias), model, api key, preset, save .env
    answers = "y\nCloud Song\nhttp://127.0.0.1:8188\nchatgpt\ngpt-4o\nsk-valid-openai\nexact\ny\n"
    res = runner.invoke(app, ["setup", str(target)], input=answers)
    assert res.exit_code == 0, res.output
    assert install_called is False
    assert seen_keys == ["sk-valid-openai"]
    assert "sk-valid-openai" not in (target / CONFIG_FILENAME).read_text(encoding="utf-8")
    assert read_project_dotenv(target) == {"OPENAI_API_KEY": "sk-valid-openai"}
    cfg = load_config(target / CONFIG_FILENAME)
    assert cfg.llm.provider == "openai"
    assert cfg.llm.model == "gpt-4o"


def test_cli_setup_bad_key_retry_and_quota_switch_to_ollama(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mvdirector import setup as setup_mod
    from mvdirector.dotenv import read_project_dotenv
    from mvdirector.llm.check import LLMCheckResult
    from mvdirector.setup import ComfyWiringReport

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(api, "inspect_environment", lambda **_kw: _fake_report(healthy=True))
    monkeypatch.setattr(
        api,
        "check_comfy_wiring",
        lambda _dir: ComfyWiringReport(url="http://127.0.0.1:8188", reachable=True),
    )

    # Case 1: Wrong key -> retry -> valid key -> saved to .env.
    outcomes = iter(
        [
            LLMCheckResult(
                json_ok=False,
                json_detail="OpenAI rejected the credentials (invalid_api_key).",
                json_error_kind="auth",
            ),
            LLMCheckResult(json_ok=True, vision="ok"),
        ]
    )
    monkeypatch.setattr(api, "verify_llm_config", lambda *_a, **_kw: next(outcomes))
    target1 = tmp_path / "retry-key"
    answers1 = (
        "y\nRetry Key\nhttp://127.0.0.1:8188\nopenai\ngpt-4o\nsk-wrong\n"
        "retry\ngpt-4o\nsk-good\nltx\ny\n"
    )
    res1 = runner.invoke(app, ["setup", str(target1)], input=answers1)
    assert res1.exit_code == 0, res1.output
    assert "Invalid or rejected API key" in res1.stdout
    assert read_project_dotenv(target1) == {"OPENAI_API_KEY": "sk-good"}

    # Case 2: Quota limit reached -> switch to local Ollama (no exhausted key saved).
    outcomes2 = iter(
        [
            LLMCheckResult(
                json_ok=False,
                json_detail="OpenAI usage or quota limit reached (insufficient_quota).",
                json_error_kind="quota",
            ),
            LLMCheckResult(json_ok=True, vision="ok"),
        ]
    )
    monkeypatch.setattr(api, "verify_llm_config", lambda *_a, **_kw: next(outcomes2))
    target2 = tmp_path / "quota-switch"
    answers2 = (
        "y\nQuota Switch\nhttp://127.0.0.1:8188\nopenai\ngpt-4o\nsk-empty-quota\n"
        "switch\nollama\ngemma3:12b\nltx\n"
    )
    res2 = runner.invoke(app, ["setup", str(target2)], input=answers2)
    assert res2.exit_code == 0, res2.output
    assert "Usage or billing quota limit reached" in res2.stdout
    assert read_project_dotenv(target2) == {}
    assert load_config(target2 / CONFIG_FILENAME).llm.provider == "ollama"

    # Case 3: Ollama missing -> installs, starts, pulls model, and verifies.
    report_missing_ollama = EnvironmentReport(
        python_version="3.12.4",
        platform_label="Darwin arm64",
        ffmpeg_ok=True,
        ffprobe_ok=True,
        extras=[ExtraStatus("qc", True, "MediaPipe QC")],
        ollama=OllamaProbe("http://127.0.0.1:11434", False, error="Unreachable"),
        comfy=ComfyProbe("http://127.0.0.1:8188", True, devices=["mps"]),
    )
    monkeypatch.setattr(api, "inspect_environment", lambda **_kw: report_missing_ollama)
    monkeypatch.setattr(setup_mod, "is_ollama_installed", lambda: False)
    monkeypatch.setattr(setup_mod, "ollama_install_command", lambda: ["brew", "install", "ollama"])
    events: list[str] = []
    monkeypatch.setattr(
        setup_mod,
        "install_ollama",
        lambda **_kw: (events.append("install") or True, "Installed Ollama"),
    )
    monkeypatch.setattr(
        setup_mod,
        "start_ollama_server",
        lambda url, **_kw: events.append("start") or OllamaProbe(url, True, []),
    )
    monkeypatch.setattr(
        setup_mod,
        "pull_ollama_model",
        lambda model, **_kw: (events.append(f"pull:{model}") or True, f"Pulled {model}"),
    )
    monkeypatch.setattr(
        setup_mod,
        "probe_ollama",
        lambda url, **_kw: OllamaProbe(url, True, ["gemma3:12b"]),
    )
    monkeypatch.setattr(
        api,
        "verify_llm_config",
        lambda *_a, **_kw: LLMCheckResult(json_ok=True, vision="ok"),
    )
    target3 = tmp_path / "auto-ollama"
    # confirm, title, comfy url, provider=ollama, install=y, model, pull=y, preset
    answers3 = "y\nAuto Ollama\nhttp://127.0.0.1:8188\nollama\ny\ngemma3:12b\ny\nltx\n"
    res3 = runner.invoke(app, ["setup", str(target3)], input=answers3)
    assert res3.exit_code == 0, res3.output
    assert events == ["install", "start", "pull:gemma3:12b"]


def test_check_comfy_wiring_validates_missing_ok_and_broken_workflows(tmp_path: Path) -> None:
    import json

    from mvdirector.setup import check_comfy_wiring
    from tests.conftest import WORKFLOW_GRAPH, write_workflows

    proj = tmp_path / "wiring-song"
    api.init_project(proj)

    # 1. Fresh project: ComfyUI reachable with /object_info, workflows not exported yet.
    def handler_ok(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/system_stats":
            return httpx.Response(
                200, json={"system": {"comfyui_version": "0.45.4"}, "devices": []}
            )
        if request.url.path == "/object_info":
            return httpx.Response(
                200,
                json={
                    "Prompt": {"input": {"required": {"text": ["STRING"], "negative": ["STRING"]}}},
                    "Sampler": {"input": {"required": {"seed": ["INT"], "steps": ["INT"]}}},
                    "Load": {"input": {"required": {"image": ["STRING"], "audio": ["STRING"]}}},
                    "Save": {"input": {"required": {"images": ["IMAGE"]}}},
                },
            )
        raise AssertionError(f"Unexpected {request.url.path}")

    fresh = check_comfy_wiring(proj, comfy_transport=httpx.MockTransport(handler_ok))
    assert fresh.reachable is True
    assert fresh.node_count == 4
    assert fresh.ok is True
    assert set(fresh.missing_exports) == {"t2i", "still", "lipsync", "i2v"}

    # 2. Exported workflows matching /object_info pass wiring validation cleanly.
    wf_dir = proj / "workflows"
    write_workflows(
        wf_dir,
        {
            "qwen_t2i": ".png",
            "qwen_edit_still": ".png",
            "ltx_lipsync": ".mp4",
            "ltx_i2v": ".mp4",
        },
    )
    wired = check_comfy_wiring(proj, comfy_transport=httpx.MockTransport(handler_ok))
    assert wired.ok is True
    assert wired.missing_exports == []
    assert set(wired.workflows_ok) == {"t2i", "still", "lipsync", "i2v"}

    # 3. Missing custom node in /object_info is caught as a wiring issue.
    broken_graph = json.loads(json.dumps(WORKFLOW_GRAPH))
    broken_graph["1"]["class_type"] = "MissingCustomNode"
    (wf_dir / "graph.api.json").write_text(json.dumps(broken_graph), encoding="utf-8")
    broken = check_comfy_wiring(proj, comfy_transport=httpx.MockTransport(handler_ok))
    assert broken.ok is False
    assert any("MissingCustomNode" in prob for _, prob in broken.issues)


def test_cli_setup_non_interactive_verify_exits_nonzero_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mvdirector.llm.check import LLMCheckResult
    from mvdirector.setup import ComfyWiringReport

    monkeypatch.setattr(api, "inspect_environment", lambda **_kw: _fake_report(healthy=True))
    monkeypatch.setattr(
        api,
        "verify_llm_config",
        lambda *_a, **_kw: LLMCheckResult(
            json_ok=False, json_detail="quota exhausted", json_error_kind="quota"
        ),
    )
    monkeypatch.setattr(
        api,
        "check_comfy_wiring",
        lambda _dir: ComfyWiringReport(url="http://127.0.0.1:8188", reachable=True),
    )

    target = tmp_path / "verify-fail"
    res = runner.invoke(
        app,
        ["setup", str(target), "--non-interactive", "--verify"],
    )
    assert res.exit_code == 1
    assert "Usage or billing quota limit reached" in res.stdout


def test_pull_ollama_model_handles_http_200_error_and_redacts_url() -> None:
    from mvdirector.setup import pull_ollama_model

    ok, msg = pull_ollama_model(
        "missing:latest",
        base_url="http://user:supersecret@127.0.0.1:11434",
        transport=httpx.MockTransport(
            lambda _r: httpx.Response(
                200, json={"error": "pull model manifest: file does not exist"}
            )
        ),
    )
    assert ok is False
    assert "file does not exist" in msg
    assert "supersecret" not in msg

    ok2, msg2 = pull_ollama_model(
        "qwen3:8b",
        base_url="http://user:supersecret@127.0.0.1:11434",
        transport=httpx.MockTransport(lambda _r: httpx.Response(200, json={"status": "success"})),
    )
    assert ok2 is True
    assert "supersecret" not in msg2


def test_check_comfy_wiring_redacts_userinfo_in_report_url(tmp_path: Path) -> None:
    from mvdirector.setup import check_comfy_wiring

    proj = tmp_path / "wired-secret"
    api.setup_project(
        proj,
        SetupOptions(
            comfy_url="http://alice:topsecret@127.0.0.1:8188",
            llm_enabled=False,
        ),
    )
    report = check_comfy_wiring(
        proj,
        comfy_transport=httpx.MockTransport(
            lambda r: (
                httpx.Response(200, json={"devices": []})
                if r.url.path == "/system_stats"
                else httpx.Response(200, json={})
            )
        ),
    )
    assert report.reachable is True
    assert "topsecret" not in report.url
    assert report.url == "http://127.0.0.1:8188"


def test_probe_ollama_and_render_environment_report_redact_userinfo() -> None:
    from mvdirector.setup import render_environment_report

    ollama_probe = probe_ollama(
        "http://bob:hunter2@127.0.0.1:11434",
        transport=httpx.MockTransport(lambda _r: httpx.Response(503)),
    )
    assert ollama_probe.reachable is False
    assert "hunter2" not in ollama_probe.error

    report = EnvironmentReport(
        python_version="3.12.0",
        platform_label="Darwin arm64",
        ffmpeg_ok=True,
        ffprobe_ok=True,
        extras=[],
        ollama=ollama_probe,
        comfy=ComfyProbe(
            url="http://carol:s3cr3t@127.0.0.1:8188",
            reachable=True,
            devices=["RTX 4090"],
        ),
    )
    rendered = render_environment_report(report)
    assert "hunter2" not in rendered
    assert "s3cr3t" not in rendered
    assert "http://127.0.0.1:11434" in rendered
    assert "http://127.0.0.1:8188" in rendered


def test_cli_setup_drops_abandoned_provider_secret_on_switch_and_respects_existing_dotenv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mvdirector.dotenv import write_project_dotenv_var
    from mvdirector.llm.check import LLMCheckResult
    from mvdirector.setup import ComfyWiringReport

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(api, "inspect_environment", lambda **_kw: _fake_report(healthy=True))
    monkeypatch.setattr(
        api,
        "check_comfy_wiring",
        lambda _dir: ComfyWiringReport(url="http://127.0.0.1:8188", reachable=True),
    )

    calls = {"n": 0}

    def fake_verify(cfg: object, **_kw: object) -> LLMCheckResult:
        calls["n"] += 1
        if getattr(cfg, "provider", None) == "openai":
            return LLMCheckResult(
                json_ok=True,
                vision="error",
                vision_detail="quota exceeded on vision",
                vision_error_kind="quota",
            )
        return LLMCheckResult(json_ok=True, vision="ok")

    monkeypatch.setattr(api, "verify_llm_config", fake_verify)

    # Interactive: enter OpenAI key, vision fails with quota, user switches to ollama.
    # The abandoned OPENAI_API_KEY must NOT be prompted to save or written to .env.
    switched_dir = tmp_path / "switched"
    answers = "\n".join(
        [
            "y",  # confirm creating new project
            "Switched Song",  # title
            "http://127.0.0.1:8188",  # comfy url
            "openai",  # provider
            "gpt-4o",  # model
            "sk-abandoned-secret",  # key
            "switch",  # switch provider after vision quota failure
            "ollama",  # new provider
            "gemma3:12b",  # model
            "ltx",  # video preset
            "",
        ]
    )
    res = runner.invoke(app, ["setup", str(switched_dir)], input=answers)
    assert res.exit_code == 0, res.output
    assert "Save OPENAI_API_KEY" not in res.stdout
    assert not (switched_dir / ".env").exists()

    # Non-interactive re-run on project that already stores OPENAI_API_KEY in .env
    # should not emit a misleading "Note: export OPENAI_API_KEY=..." warning.
    existing_dir = tmp_path / "with-dotenv"
    api.init_project(existing_dir)
    write_project_dotenv_var(existing_dir, "OPENAI_API_KEY", "sk-already-saved")
    res2 = runner.invoke(
        app,
        [
            "setup",
            str(existing_dir),
            "--non-interactive",
            "--no-verify",
            "--llm-provider",
            "openai",
            "--llm-model",
            "gpt-4o",
        ],
    )
    assert res2.exit_code == 0, res2.output
    assert "Note: export OPENAI_API_KEY" not in res2.stdout


def test_optimize_setup_options_local_vs_remote_and_vram() -> None:
    local_probe = ComfyProbe("http://127.0.0.1:8188", True, vram_total_gb=16.0)
    local_opts = optimize_setup_options(
        SetupOptions(comfy_url="http://127.0.0.1:8188", llm_enabled=True),
        comfy_probe=local_probe,
    )
    assert local_opts.comfy_dedup_uploads is False
    assert local_opts.comfy_cache_enabled is True
    assert local_opts.comfy_sort_batch_for_cache is True
    assert local_opts.comfy_min_free_vram_gb == 4.0
    assert local_opts.qc_vision_blind_observer is False
    assert local_opts.retry_in_place_repair is True

    remote_probe = ComfyProbe("https://gpu.example.com:8188", True, vram_total_gb=48.0)
    remote_opts = optimize_setup_options(
        SetupOptions(
            comfy_url="https://gpu.example.com:8188",
            mode=ProjectMode.MIX,
            llm_enabled=False,
        ),
        comfy_probe=remote_probe,
    )
    assert remote_opts.comfy_dedup_uploads is True
    assert remote_opts.comfy_min_free_vram_gb == 6.0
    assert remote_opts.qc_vision_enabled is False
    assert remote_opts.qc_vision_blind_observer is False
    assert remote_opts.retry_in_place_repair is False

    # Small GPU (< 8 GiB) or unreachable probe leaves VRAM floor unset.
    small_gpu = ComfyProbe("http://127.0.0.1:8188", True, vram_total_gb=6.0)
    assert (
        optimize_setup_options(
            SetupOptions(comfy_url="http://127.0.0.1:8188"), comfy_probe=small_gpu
        ).comfy_min_free_vram_gb
        is None
    )

    # Explicit overrides are always respected.
    explicit = optimize_setup_options(
        SetupOptions(
            comfy_url="https://gpu.example.com:8188",
            comfy_dedup_uploads=False,
            comfy_cache_enabled=False,
            comfy_min_free_vram_gb=8.0,
        ),
        comfy_probe=remote_probe,
    )
    assert explicit.comfy_dedup_uploads is False
    assert explicit.comfy_cache_enabled is False
    assert explicit.comfy_min_free_vram_gb == 8.0


def test_apply_setup_config_validates_and_writes_optimized_fields(tmp_path: Path) -> None:
    project_dir = tmp_path / "opt-song"
    api.init_project(project_dir)

    for bad_vram in (0.1, 300.0, float("nan"), float("inf")):
        with pytest.raises(ConfigError, match="min_free_vram_gb"):
            apply_setup_config(project_dir, SetupOptions(comfy_min_free_vram_gb=bad_vram))

    opts = optimize_setup_options(
        SetupOptions(comfy_url="http://localhost:8188", llm_enabled=True),
        comfy_probe=ComfyProbe("http://localhost:8188", True, vram_total_gb=24.0),
    )
    apply_setup_config(project_dir, opts)
    cfg = load_config(project_dir / CONFIG_FILENAME)
    assert cfg.comfy.dedup_uploads is False
    assert cfg.comfy.cache_enabled is True
    assert cfg.comfy.sort_batch_for_cache is True
    assert cfg.comfy.min_free_vram_gb == 6.0
    assert cfg.qc.vision_blind_observer is False
    assert cfg.retry.in_place_repair is True


def test_cli_setup_local_disables_dedup_and_remote_enables_dedup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(api, "inspect_environment", lambda **_kw: _fake_report(healthy=True))
    local_dir = tmp_path / "local-opt"
    res = runner.invoke(
        app,
        ["setup", str(local_dir), "--non-interactive", "--comfy-url", "http://127.0.0.1:8188"],
    )
    assert res.exit_code == 0, res.output
    assert "Optimized for local ComfyUI: dedup_uploads=false" in res.stdout
    cfg = load_config(local_dir / CONFIG_FILENAME)
    assert cfg.comfy.dedup_uploads is False
    assert cfg.comfy.cache_enabled is True
    assert cfg.comfy.sort_batch_for_cache is True
    assert cfg.comfy.min_free_vram_gb == 6.0
    assert cfg.retry.in_place_repair is True


def test_inspect_environment_forwards_explicit_comfy_headers() -> None:
    from mvdirector.setup import inspect_environment

    seen_auth: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_auth.append(request.headers.get("authorization"))
        return httpx.Response(200, json={"system": {"comfyui_version": "0.45.4"}, "devices": []})

    report = inspect_environment(
        comfy_url="http://127.0.0.1:8188",
        comfy_headers={"Authorization": "Basic dXNlcjpwYXNz"},
        comfy_transport=httpx.MockTransport(handler),
        ollama_transport=httpx.MockTransport(lambda _r: httpx.Response(200, json={"models": []})),
    )
    assert report.comfy.reachable is True
    assert seen_auth == ["Basic dXNlcjpwYXNz"]


def test_comfy_client_sends_api_key_comfy_org_in_extra_data() -> None:
    import json

    from mvdirector.comfy.client import ComfyClient
    from mvdirector.config import ComfyConfig

    captured: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(200, json={"prompt_id": "p-123"})

    cfg = ComfyConfig(base_url="http://127.0.0.1:8188", api_key_comfy_org="comfyui-test-key")
    with ComfyClient(cfg, transport=httpx.MockTransport(handler)) as client:
        pid = client.queue({"1": {"class_type": "Test", "inputs": {}}})

    assert pid == "p-123"
    assert captured[0]["extra_data"] == {"api_key_comfy_org": "comfyui-test-key"}


def test_url_validation_and_loopback_edge_cases() -> None:
    from mvdirector.setup import validate_http_url

    for bad in (
        "http://[invalid",
        "http://:8188",
        "http://127.0.0.1:abc",
        "http://127.0.0.1:99999",
    ):
        with pytest.raises(ConfigError):
            validate_http_url(bad, "ComfyUI URL")

    assert is_loopback_url("http://[invalid") is False
    assert is_loopback_url("http://localhost.:8188") is True
    assert is_loopback_url("http://[::ffff:127.0.0.1]:8188") is True


def test_inspect_environment_merges_comfy_headers_case_insensitively(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mvdirector.setup import inspect_environment

    monkeypatch.setenv("COMFY_TOKEN", "from-env")
    seen_auth_headers: list[list[tuple[str, str]]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_auth_headers.append(
            [
                (k.decode(), v.decode())
                for k, v in request.headers.raw
                if k.lower() == b"authorization"
            ]
        )
        return httpx.Response(200, json={"devices": []})

    inspect_environment(
        comfy_url="http://127.0.0.1:8188",
        comfy_token_env="COMFY_TOKEN",
        comfy_headers={"authorization": "Basic explicit"},
        comfy_transport=httpx.MockTransport(handler),
        ollama_transport=httpx.MockTransport(lambda _r: httpx.Response(200, json={"models": []})),
    )
    assert seen_auth_headers == [[("authorization", "Basic explicit")]]


def test_cli_setup_preserves_mix_mode_in_place_repair_false_on_existing_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(api, "inspect_environment", lambda **_kw: _fake_report(healthy=True))
    mix_dir = tmp_path / "existing-mix"
    api.init_project(mix_dir, mode=ProjectMode.MIX)

    res = runner.invoke(
        app,
        ["setup", str(mix_dir), "--non-interactive", "--comfy-url", "http://127.0.0.1:8188"],
    )
    assert res.exit_code == 0, res.output
    cfg = load_config(mix_dir / CONFIG_FILENAME)
    assert cfg.mode is ProjectMode.MIX
    assert cfg.retry.in_place_repair is False


def test_setup_project_fails_atomically_on_invalid_provider_or_policy(tmp_path: Path) -> None:
    bad_dir = tmp_path / "never-created"
    with pytest.raises(ConfigError, match="Invalid configuration"):
        api.setup_project(bad_dir, SetupOptions(llm_provider="bogus"))
    assert not bad_dir.exists()

    with pytest.raises(ConfigError, match="Invalid configuration"):
        api.setup_project(bad_dir, SetupOptions(qc_on_missing_dependency="ignore"))
    assert not bad_dir.exists()


def test_probe_comfy_ignores_ambient_env_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("COMFY_ORG_API_KEY", "bad\r\nkey")
    probe = probe_comfy(
        "http://127.0.0.1:8188",
        transport=httpx.MockTransport(lambda _r: httpx.Response(200, json={"devices": []})),
    )
    assert probe.reachable is True


def test_rerunning_setup_keeps_existing_tuning_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(api, "inspect_environment", lambda **_kw: _fake_report(healthy=True))
    project_dir = tmp_path / "tuned"
    args = ["setup", str(project_dir), "--non-interactive", "--comfy-url", "http://127.0.0.1:8188"]
    assert runner.invoke(app, args).exit_code == 0

    yaml_path = project_dir / CONFIG_FILENAME
    text = yaml_path.read_text(encoding="utf-8")
    text = text.replace("cache_enabled: true", "cache_enabled: false")
    text = text.replace("in_place_repair: true", "in_place_repair: false")
    text = text.replace("vision_blind_observer: false", "vision_blind_observer: true")
    yaml_path.write_text(text, encoding="utf-8")
    before = load_config(yaml_path)
    assert before.comfy.cache_enabled is False
    assert before.retry.in_place_repair is False
    assert before.qc.vision_blind_observer is True

    res = runner.invoke(app, args)
    assert res.exit_code == 0, res.output
    after = load_config(yaml_path)
    assert after.comfy.cache_enabled is False
    assert after.retry.in_place_repair is False
    assert after.qc.vision_blind_observer is True
    assert "cache_enabled=" not in res.stdout


def test_detect_project_mode_ignores_symlinked_project_yaml(tmp_path: Path) -> None:
    from mvdirector.setup import detect_project_mode

    outside = tmp_path / "outside.yaml"
    outside.write_text("mode: mix\n", encoding="utf-8")
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / CONFIG_FILENAME).symlink_to(outside)
    assert detect_project_mode(proj) is None
