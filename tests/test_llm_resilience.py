# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Failure behavior of the language-model layer: what counts as unavailable, what is retried, and
when the application stops asking.

These cover the cases a run hits when a model is misconfigured, a server is down, or a local
model cannot read images, which are the cases where the cost of a wrong response is a long wait.
"""

from __future__ import annotations

import logging

import anthropic
import httpx
import numpy as np
import pytest
import yaml
from typer.testing import CliRunner

from mvdirector.cli import app
from mvdirector.config import LLMConfig, ProjectConfig, load_config
from mvdirector.errors import ConfigError, ImagesNotSupportedError, LLMError, LLMUnavailableError
from mvdirector.llm.check import ping_llm
from mvdirector.models import AssetKind, Manifest
from mvdirector.planning.planner import build_manifest
from mvdirector.qc.common import vision_checks
from mvdirector.qc.vision import MAX_UNAVAILABLE_IN_A_ROW, VisionJudge, VisionVerdict
from tests.test_llm import (
    FakeAnthropic,
    FakeServer,
    a_png_block,
    anthropic_backend,
    ask,
    build,
    error_reply,
    openai_reply,
)
from tests.test_vision import _shot

REQUEST = httpx.Request("POST", "http://x")
IMAGE_ERROR = "this model is missing data required for image input"

# ------------------------------------------------------- the model must be named explicitly


@pytest.mark.parametrize("provider", ["openai", "gemini", "anthropic"])
def test_other_providers_need_an_explicit_model(provider):
    """The default model name is an Ollama model, which no other provider has."""
    with pytest.raises(
        ValueError, match=f"llm.model must be set when llm.provider is '{provider}'"
    ):
        LLMConfig(provider=provider)
    assert LLMConfig(provider=provider, model="some-model").model == "some-model"


def test_the_default_is_a_local_ollama_model():
    config = LLMConfig()
    assert (config.provider, config.model) == ("ollama", "gemma3:12b")


def test_a_yaml_file_that_forgets_the_model_gets_a_clear_message(tmp_path):
    path = tmp_path / "project.yaml"
    path.write_text("title: t\nllm: {provider: gemini}\n")
    with pytest.raises(ConfigError, match="llm.model must be set when llm.provider is 'gemini'"):
        load_config(path)
    path.write_text("title: t\nllm: {provider: gemini, model: some-model}\n")
    assert load_config(path).llm.model == "some-model"


def test_a_config_written_back_to_yaml_still_loads(tmp_path):
    """A full dump names every field, including the model, so it never trips the rule."""
    config = ProjectConfig()
    config.llm.provider = "ollama"
    config.llm.model = "llama"
    path = tmp_path / "project.yaml"
    path.write_text(yaml.safe_dump(config.model_dump(mode="json")))
    assert load_config(path).llm.provider == "ollama"


# --------------------------------------- image rejection reported as a server error (not retried)


@pytest.mark.parametrize("provider", ["ollama", "openai_compatible", "gemini"])
def test_a_500_that_says_the_model_cannot_read_images_is_recognised_at_once(provider):
    sleeps: list[float] = []
    server = FakeServer(httpx.Response(500, json={"error": IMAGE_ERROR}))
    client = build(server, sleeps, provider=provider)
    with pytest.raises(ImagesNotSupportedError, match="missing data required for image input"):
        ask(client, [a_png_block()])
    assert len(server.requests) == 1 and sleeps == []


def test_other_server_errors_with_images_are_still_retried():
    sleeps: list[float] = []
    server = FakeServer(httpx.Response(500, json={"error": "out of memory"}))
    with pytest.raises(LLMError) as raised:
        ask(build(server, sleeps, provider="ollama"), [a_png_block()])
    assert not isinstance(raised.value, ImagesNotSupportedError)
    assert len(server.requests) == 3 and sleeps == [2.0, 4.0]


def test_a_text_only_request_is_never_blamed_on_images():
    server = FakeServer(error_reply(400, "image buffer was empty"))
    with pytest.raises(LLMError) as raised:
        ask(build(server, provider="openai_compatible"))
    assert not isinstance(raised.value, ImagesNotSupportedError)


# ------------------------------------------------- what counts as "unavailable"


@pytest.mark.parametrize(
    "outcome",
    [
        httpx.ConnectError("refused"),
        httpx.ReadTimeout("too slow"),
        error_reply(401, "bad key"),
        error_reply(403, "forbidden"),
        error_reply(404, "model not found"),
    ],
    ids=["unreachable", "timeout", "401", "403", "404"],
)
def test_failures_that_retrying_cannot_fix_are_unavailable(outcome):
    with pytest.raises(LLMUnavailableError):
        ask(build(FakeServer(outcome), provider="openai"))


def test_gemini_reports_a_bad_key_as_a_400_and_it_is_unavailable():
    server = FakeServer(error_reply(400, "API key not valid. Please pass a valid API key."))
    with pytest.raises(LLMUnavailableError, match="credentials"):
        ask(build(server, provider="gemini"))


def test_a_bad_reply_from_a_working_model_is_not_unavailable():
    server = FakeServer(openai_reply("not json at all"))
    with pytest.raises(LLMError) as raised:
        ask(build(server, provider="openai_compatible"))
    assert not isinstance(raised.value, LLMUnavailableError)


def test_rate_limiting_is_not_unavailable():
    server = FakeServer(httpx.Response(429, json={"error": {"message": "slow down"}}))
    with pytest.raises(LLMError, match="rate limiting") as raised:
        ask(build(server, provider="openai"))
    assert not isinstance(raised.value, LLMUnavailableError)


def test_anthropic_errors_map_onto_the_same_taxonomy():
    def status_error(cls, status):
        return cls("nope", response=httpx.Response(status, request=REQUEST), body=None)

    unavailable = [
        anthropic.APIConnectionError(request=REQUEST),
        status_error(anthropic.AuthenticationError, 401),
        status_error(anthropic.PermissionDeniedError, 403),
        status_error(anthropic.NotFoundError, 404),
    ]
    for failure in unavailable:
        with pytest.raises(LLMUnavailableError):
            ask(anthropic_backend(FakeAnthropic(failure)))
    with pytest.raises(LLMError) as raised:
        ask(anthropic_backend(FakeAnthropic(status_error(anthropic.BadRequestError, 400))))
    assert not isinstance(raised.value, LLMUnavailableError)


# ---------------------------------------------------- the judge stops asking when it should


class ScriptedLLM:
    """Plays back outcomes in order: an exception is raised, anything else is the verdict."""

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def complete_json(self, *, system, content, schema):
        self.calls += 1
        outcome = self.outcomes.pop(0) if len(self.outcomes) > 1 else self.outcomes[0]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


VERDICT = VisionVerdict(spec_match=0.9, artifact_free=0.9, character_match=0.9)
DOWN = LLMUnavailableError("connection refused")
MANIFEST = Manifest(title="t", duration=4.0, fps=24)


def judge_once(judge):
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    return vision_checks(judge, [image], _shot(), MANIFEST, AssetKind.STILL, 0.3, 0.4)


def test_the_judge_gives_up_after_repeated_unavailability_with_one_warning(caplog):
    llm = ScriptedLLM(DOWN)
    judge = VisionJudge(llm)
    with caplog.at_level(logging.WARNING):
        results = [judge_once(judge) for _ in range(MAX_UNAVAILABLE_IN_A_ROW + 2)]
    assert llm.calls == MAX_UNAVAILABLE_IN_A_ROW  # the later attempts never reached the model
    assert caplog.text.count("Vision checks are off for the rest of this run") == 1
    last = results[-1]
    assert all(c.status.value == "skipped" for c in last)
    assert f"unavailable {MAX_UNAVAILABLE_IN_A_ROW} times in a row" in last[0].detail
    assert "connection refused" in last[0].detail


def test_bad_replies_never_make_the_judge_give_up():
    llm = ScriptedLLM(LLMError("the reply was not valid JSON"))
    judge = VisionJudge(llm)
    for _ in range(10):
        judge_once(judge)
    assert llm.calls == 10 and judge.disabled_reason is None


def test_a_success_resets_the_count():
    llm = ScriptedLLM(DOWN, DOWN, VERDICT, DOWN, DOWN, DOWN)
    judge = VisionJudge(llm)
    for _ in range(5):
        judge_once(judge)
    assert judge.disabled_reason is None  # two, a success, then two: never three in a row
    judge_once(judge)
    assert judge.disabled_reason is not None


# -------------------------------------------------------------------- the early check


class Answers:
    def __init__(self, ok=True, error=None):
        self.ok, self.error = ok, error

    def complete_json(self, *, system, content, schema):
        if self.error:
            raise self.error
        return schema(ok=self.ok, word="ready")


def test_ping_passes_when_the_model_answers():
    ping_llm(Answers())


def test_ping_fails_when_the_model_answers_wrongly():
    with pytest.raises(LLMError, match="not as instructed"):
        ping_llm(Answers(ok=False))


def test_ping_lets_unavailability_through():
    with pytest.raises(LLMUnavailableError):
        ping_llm(Answers(error=DOWN))


def prepare_generation(project, analysis, enabled):
    config = project.config
    config.llm.enabled, config.llm.provider, config.llm.model = enabled, "ollama", "some-model"
    config.qc.on_missing_dependency = "skip"
    (project.root / "project.yaml").write_text(yaml.safe_dump(config.model_dump(mode="json")))
    project.save_manifest(build_manifest(config, analysis, "lyrics", "style", llm=None))


def test_generation_stops_before_any_work_when_the_model_is_unusable(
    project, analysis, monkeypatch
):
    prepare_generation(project, analysis, enabled=True)
    monkeypatch.setattr("mvdirector.api.ping_llm", lambda client: (_ for _ in ()).throw(DOWN))
    result = CliRunner().invoke(app, ["stills", "--skip-preflight", "-p", str(project.root)])
    assert result.exit_code == 1
    assert "The language model is not usable: connection refused" in result.output
    assert "mvd llm-check" in result.output and "llm.enabled: false" in result.output
    assert "ComfyUI" not in result.output  # it never got as far as the GPU server
    assert not project.is_locked


def test_generation_does_not_ping_a_model_that_is_switched_off(project, analysis, monkeypatch):
    prepare_generation(project, analysis, enabled=False)
    pinged = []
    monkeypatch.setattr("mvdirector.api.ping_llm", lambda client: pinged.append(client))

    class StopHere:
        def __init__(self, *args, **kwargs):
            raise LLMError("stop before the GPU server")

    monkeypatch.setattr("mvdirector.comfy.provider.ComfyClient", StopHere)
    result = CliRunner().invoke(app, ["stills", "--skip-preflight", "-p", str(project.root)])
    assert pinged == [] and "stop before the GPU server" in result.output
