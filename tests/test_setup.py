# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Tests for local environment diagnostics and `mvd setup`."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
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
        "OK   ComfyUI at http://127.0.0.1:8188 (v0.45.4, mps, VRAM 24.0/32.0 GiB free)"
        in text_ok
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
    monkeypatch.setattr(api, "inspect_environment", lambda **_kw: _fake_report(healthy=True))
    target = tmp_path / "starlight"

    # Prompts: confirm create, title, comfy url, llm provider, ollama model, video preset
    answers = "y\nStarlight Echo\nhttp://127.0.0.1:8188\nollama\ngemma3:12b\nltx\n"
    res = runner.invoke(app, ["setup", str(target)], input=answers)
    assert res.exit_code == 0, res.output

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
        is_remote = isinstance(url, str) and "gpu.example.com" in url
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

    monkeypatch.setattr(api, "inspect_environment", fake_inspect)

    # Interactive: enter remote ComfyUI URL + COMFY_TOKEN -> triggers re-probe and note.
    target = tmp_path / "remote-interactive"
    answers = (
        "y\nRemote Track\nhttps://gpu.example.com:8188\nCOMFY_TOKEN\nollama\ngemma3:12b\nltx\n"
    )
    res = runner.invoke(app, ["setup", str(target)], input=answers)
    assert res.exit_code == 0, res.output
    assert "Connected to ComfyUI at https://gpu.example.com:8188" in res.stdout
    assert "export COMFY_TOKEN=<your-token>" in res.stdout
    assert "https://gpu.example.com:8188" in probed_urls

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
    raw_yaml = (target2 / CONFIG_FILENAME).read_text(encoding="utf-8")
    assert "https://gpu.example.com:8188" in raw_yaml
    assert "${COMFY_TOKEN}" in raw_yaml

