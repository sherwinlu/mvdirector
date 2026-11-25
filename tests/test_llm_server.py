# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""End to end: the real CLI, planner and vision judge talking HTTP to a local fake server."""

from __future__ import annotations

import socket

import numpy as np
import pytest
import yaml
from typer.testing import CliRunner

from mvdirector.audio.analysis import analyze_project
from mvdirector.cli import app
from mvdirector.config import LLMConfig
from mvdirector.llm import build_llm, image_block
from mvdirector.models import AssetKind, Framing, Manifest, ShotType
from mvdirector.qc.common import vision_checks
from mvdirector.qc.vision import VisionJudge
from tests.llm_server import FakeLLMServer
from tests.test_vision import _shot


@pytest.fixture
def server():
    fake = FakeLLMServer().start()
    yield fake
    fake.stop()


@pytest.fixture
def text_only_server():
    fake = FakeLLMServer(vision=False).start()
    yield fake
    fake.stop()


def provider_settings(provider: str, url: str) -> dict:
    """The llm: section that points ``provider`` at the fake server."""
    return {
        "openai": {"provider": "openai", "model": "m", "base_url": f"{url}/v1"},
        "openai_compatible": {
            "provider": "openai_compatible",
            "model": "m",
            "base_url": f"{url}/v1",
        },
        "ollama": {"provider": "ollama", "model": "m", "base_url": url},
        "gemini": {"provider": "gemini", "model": "m", "base_url": f"{url}/v1beta"},
    }[provider]


PROVIDERS = ["openai", "openai_compatible", "ollama", "gemini"]


@pytest.fixture(autouse=True)
def keys(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("GEMINI_API_KEY", "g-test")
    monkeypatch.setattr("time.sleep", lambda _: None)


def use_llm(project, settings: dict) -> None:
    path = project.root / "project.yaml"
    data = yaml.safe_load(path.read_text())
    data["llm"].update({"enabled": True, **settings})
    path.write_text(yaml.safe_dump(data))


# ------------------------------------------------------------------ llm-check


@pytest.mark.parametrize("provider", PROVIDERS)
def test_llm_check_works_against_each_protocol(project, server, provider):
    use_llm(project, provider_settings(provider, server.url))
    result = CliRunner().invoke(app, ["llm-check", "-p", str(project.root)])
    assert result.exit_code == 0, result.output
    assert "[  ok] JSON output" in result.output and "[  ok] Image input" in result.output
    assert len(server.requests) == 2 and server.requests[1].images == 1


def test_credentials_travel_in_the_right_header(project, server):
    use_llm(project, provider_settings("openai", server.url))
    CliRunner().invoke(app, ["llm-check", "-p", str(project.root), "--skip-image"])
    assert server.requests[0].headers["authorization"] == "Bearer sk-test"
    use_llm(project, provider_settings("gemini", server.url))
    CliRunner().invoke(app, ["llm-check", "-p", str(project.root), "--skip-image"])
    assert server.requests[1].headers["x-goog-api-key"] == "g-test"


@pytest.mark.parametrize("provider", PROVIDERS)
def test_a_text_only_model_is_reported_clearly(project, text_only_server, provider):
    use_llm(project, provider_settings(provider, text_only_server.url))
    result = CliRunner().invoke(app, ["llm-check", "-p", str(project.root)])
    # Vision checks are on by default, so a model that cannot read images does not fit the config.
    assert result.exit_code == 1, result.output
    assert "[  ok] JSON output" in result.output
    assert "[FAIL] Image input" in result.output and "qc.vision_enabled: false" in result.output


def test_an_unreachable_server_fails_the_check_with_a_clear_message(project):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]  # nothing listens here once the socket closes
    use_llm(project, {"provider": "ollama", "model": "m", "base_url": f"http://127.0.0.1:{port}"})
    result = CliRunner().invoke(app, ["llm-check", "-p", str(project.root)])
    assert result.exit_code == 1
    assert "Could not reach" in result.output and "ollama serve" in result.output


# -------------------------------------------------------------------- planning


@pytest.mark.parametrize("provider", PROVIDERS)
def test_planning_runs_end_to_end_over_http(project, server, provider):
    project.config.video.max_clip_seconds = 4.0
    path = project.root / "project.yaml"
    data = yaml.safe_load(path.read_text())
    data["video"]["max_clip_seconds"] = 4.0
    path.write_text(yaml.safe_dump(data))
    project.save_analysis(analyze_project(project))
    use_llm(project, provider_settings(provider, server.url))

    result = CliRunner().invoke(app, ["plan", "-p", str(project.root)])
    assert result.exit_code == 0, result.output

    manifest = project.load_manifest()
    assert manifest.visual_bible.concept == "A knight turns a cage into a stage."
    assert manifest.character.appearance.startswith("A singer with a small beauty mark")
    assert manifest.character.wardrobe_ids == ["cape", "bare"]
    assert [s.name for s in manifest.visual_bible.settings] == ["Corridor"]
    sung = [s for s in manifest.shots if s.type is ShotType.LIPSYNC]
    assert sung and all(s.still_prompt == f"still prompt for {s.id}" for s in manifest.shots)
    assert all(s.framing is Framing.CLOSE_UP for s in sung)
    assert all(s.wardrobe == "cape" and s.setting == "Corridor" for s in manifest.shots)
    # One bible call, and the shot list fits in one batch.
    assert len(server.of_kind("bible")) == 1 and len(server.of_kind("shots")) == 1


def test_a_reply_that_is_not_json_is_corrected_over_http(project, server):
    server.invalid_first = True
    project.save_analysis(analyze_project(project))
    use_llm(project, provider_settings("ollama", server.url))
    result = CliRunner().invoke(app, ["plan", "-p", str(project.root)])
    assert result.exit_code == 0, result.output
    assert len(server.of_kind("bible")) == 2  # the first reply was prose, then a correction
    assert project.load_manifest().visual_bible.concept


def test_planning_fails_cleanly_when_the_key_is_missing(project, server, monkeypatch):
    project.save_analysis(analyze_project(project))
    use_llm(project, {"provider": "openai", "model": "m", "base_url": f"{server.url}/v1"})
    monkeypatch.delenv("OPENAI_API_KEY")
    result = CliRunner().invoke(app, ["plan", "-p", str(project.root)])
    assert result.exit_code == 1 and "OPENAI_API_KEY" in result.output
    assert not server.requests and not project.manifest_path.exists()


# ---------------------------------------------------------------------- vision


def vision_manifest() -> Manifest:
    return Manifest(title="t", duration=4.0, fps=24)


@pytest.mark.parametrize("provider", PROVIDERS)
def test_the_vision_judge_sends_the_reference_and_the_frame(server, provider):
    llm = build_llm(LLMConfig(**provider_settings(provider, server.url)))
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    judge = VisionJudge(llm)
    verdict = judge.judge([image], _shot(), vision_manifest(), AssetKind.STILL)
    assert verdict.spec_match == 0.9 and verdict.character_match == 0.9
    assert server.of_kind("vision")[0].images == 1  # no reference image was given here


def test_a_text_only_model_turns_vision_off_after_one_request(text_only_server, caplog):
    llm = build_llm(LLMConfig(**provider_settings("ollama", text_only_server.url)))
    judge = VisionJudge(llm)
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    for _ in range(3):
        results = vision_checks(
            judge, [image], _shot(), vision_manifest(), AssetKind.STILL, 0.3, 0.4
        )
        assert all(r.status.value == "skipped" for r in results)
    assert len(text_only_server.requests) == 1
    assert caplog.text.count("Vision checks are off") == 1


def test_image_blocks_are_valid_jpeg_data():
    block = image_block(np.zeros((8, 8, 3), dtype=np.uint8))
    assert block["source"]["media_type"] == "image/jpeg"


# ------------------------------------------- the whole generate-check-retry loop over HTTP


def test_vision_failures_drive_retries_and_a_rewrite_over_http(
    project, server, tmp_path, sample_png, analysis
):
    """Runner, image gate, vision judge, retry policy and the model, all through real HTTP."""
    from mvdirector.comfy.client import ComfyClient
    from mvdirector.generation import reference
    from mvdirector.generation.runner import RunBudget, ShotRunner, load_templates
    from mvdirector.models import AssetStatus, RetryStrategy
    from mvdirector.planning.planner import build_manifest
    from mvdirector.qc.factory import QualityGates
    from mvdirector.qc.image import ImageQC
    from mvdirector.retry import RetryPolicy
    from tests.conftest import FakeComfy
    from tests.llm_server import VERDICT

    project.config.video.max_clip_seconds = 4.0
    manifest = build_manifest(project.config, analysis, "lyrics", "style", llm=None)
    manifest.character.appearance = "A singer with a beauty mark under her left eye."
    sheet = tmp_path / "sheet.png"
    sheet.write_bytes(sample_png)
    reference.lock_reference(project, manifest, sheet)

    wrong_person = {
        **VERDICT,
        "character_match": 0.1,
        "issues": ["wrong person"],
        "fix": "Match the sheet.",
    }
    server.verdicts = [wrong_person, wrong_person, VERDICT]
    llm = build_llm(LLMConfig(**provider_settings("openai_compatible", server.url)))
    gates = QualityGates(
        image=ImageQC(project.config.qc, project.config.video, None, VisionJudge(llm)),
        video=None,  # stills only in this test
        face_analyzer=None,
    )
    fake_comfy = FakeComfy(png=sample_png)
    comfy = ComfyClient(
        project.config.comfy, transport=fake_comfy.transport(), sleep=lambda _: None
    )
    templates = load_templates(project, ["t2i", "still", "lipsync", "i2v"])
    policy = RetryPolicy(project.config.retry, project.config.seed, llm)
    runner = ShotRunner(project, manifest, comfy, gates, policy, templates, RunBudget(20))

    runner.run([AssetKind.STILL], shot_ids=["s002"])

    state = manifest.shot("s002").still
    assert state.status is AssetStatus.PASSED and state.selected_attempt == 3
    assert [a.strategy for a in state.attempts] == [
        RetryStrategy.INITIAL, RetryStrategy.NEW_SEED, RetryStrategy.REWRITE_PROMPT,
    ]  # fmt: skip
    first = state.attempts[0].qc
    assert not first.passed and first.has_hard_failure and "wrong person" in first.issues[0]
    # Three vision calls, each with the reference sheet and the still. One rewrite, then done.
    assert len(server.of_kind("vision")) == 3
    assert all(call.images == 2 for call in server.of_kind("vision"))
    assert len(server.of_kind("rewrite")) == 1
    rewrite_text = str(server.of_kind("rewrite")[0].body)
    assert "wrong person" in rewrite_text and "Match the sheet." in rewrite_text
    assert "A rewritten prompt." in state.attempts[2].prompt
