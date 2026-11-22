# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""The language-model layer: request shapes, parsing, retries, errors and provider setup."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

import anthropic
import httpx
import numpy as np
import pytest
import yaml
from pydantic import BaseModel
from typer.testing import CliRunner

from mvdirector.cli import app
from mvdirector.config import LLMConfig
from mvdirector.errors import ConfigError, ImagesNotSupportedError, LLMError
from mvdirector.llm import build_llm, extract_json, image_block, text_block
from mvdirector.llm.anthropic_backend import AnthropicChat
from mvdirector.llm.base import estimate_tokens, schema_prompt
from mvdirector.llm.check import make_test_image, run_llm_check
from mvdirector.llm.gemini_backend import GeminiChat
from mvdirector.llm.ollama_backend import OllamaChat
from mvdirector.llm.openai_backend import OpenAIChat
from mvdirector.models import AssetKind
from mvdirector.qc.common import vision_checks
from mvdirector.qc.vision import VisionJudge


class Answer(BaseModel):
    color: str
    score: float = 0.5


GOOD = '{"color": "teal", "score": 0.9}'


@dataclass
class Recorded:
    url: str
    headers: dict[str, str]
    body: dict


class FakeServer:
    """An httpx transport that records requests and plays back queued responses."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests: list[Recorded] = []

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self._handle)

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(
            Recorded(str(request.url), dict(request.headers), json.loads(request.content))
        )
        item = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if isinstance(item, Exception):
            raise item
        return item


def openai_reply(text, finish="stop"):
    message = {"role": "assistant", "content": text}
    return httpx.Response(200, json={"choices": [{"message": message, "finish_reason": finish}]})


def ollama_reply(text, done_reason="stop", prompt_eval_count=100):
    return httpx.Response(
        200,
        json={
            "message": {"role": "assistant", "content": text},
            "done": True,
            "done_reason": done_reason,
            "prompt_eval_count": prompt_eval_count,
        },
    )


def gemini_reply(text, finish="STOP"):
    parts = [{"text": text}]
    return httpx.Response(
        200, json={"candidates": [{"content": {"parts": parts}, "finishReason": finish}]}
    )


def error_reply(status, message, shape="openai"):
    body = {"error": message if shape == "ollama" else {"message": message}}
    return httpx.Response(status, json=body)


def build(server, sleeps=None, environ=None, **fields):
    fields.setdefault("provider", "openai_compatible")
    fields.setdefault("model", "test-model")
    if fields["provider"] == "openai_compatible":
        fields.setdefault("base_url", "http://localhost:1234/v1")
    config = LLMConfig(**fields)
    return build_llm(
        config,
        environ={
            "OPENAI_API_KEY": "sk-test",
            "GEMINI_API_KEY": "g-test",
            "ANTHROPIC_API_KEY": "a-test",
        }
        if environ is None
        else environ,
        transport=server.transport(),
        sleep=(sleeps.append if sleeps is not None else (lambda _: None)),
    )


def ask(client, content=None):
    return client.complete_json(
        system="Judge the color.", content=content or [text_block("What color?")], schema=Answer
    )


def a_png_block():
    return image_block(np.zeros((16, 16, 3), dtype=np.uint8))


# ============================================================ shared behaviour


def test_extract_json_handles_fences_and_prose():
    assert extract_json('Sure!\n```json\n{"a": 1}\n```') == '{"a": 1}'
    with pytest.raises(LLMError, match="no JSON object"):
        extract_json("no braces here")


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ('<think>I should use {"x": 1} maybe</think>{"a": 1}', '{"a": 1}'),
        ('<THINK>\nmulti\nline\n</THINK>\n```json\n{"a": 1}\n```', '{"a": 1}'),
        ('{"a": {"b": 2}} Hope that helps {not json}', '{"a": {"b": 2}}'),
        ('{"a": 1} {"a": 2}', '{"a": 1}'),
        ('{"a": "}"}', '{"a": "}"}'),
    ],
)
def test_extract_json_returns_the_first_complete_object(reply, expected):
    assert extract_json(reply) == expected


def test_extract_json_keeps_malformed_json_so_the_real_error_is_reported():
    assert extract_json('Here: {"a": 1,}') == '{"a": 1,}'
    with pytest.raises(LLMError):
        extract_json('<think>never finished {"a": 1}')


def test_schema_prompt_embeds_valid_json_schema():
    prompt = schema_prompt("Do the thing.", Answer)
    assert prompt.startswith("Do the thing.")
    schema_text = prompt.split("JSON Schema:\n", 1)[1]
    assert json.loads(schema_text)["properties"].keys() == {"color", "score"}


def test_token_estimate_counts_text_and_images():
    messages = [{"role": "user", "content": [text_block("x" * 400), a_png_block()]}]
    assert estimate_tokens("y" * 400, messages) == 200 + 1000


def test_one_correction_is_requested_when_the_reply_is_not_valid(caplog):
    server = FakeServer(openai_reply("not json at all"), openai_reply(GOOD))
    with caplog.at_level(logging.WARNING, logger="mvdirector.llm"):
        answer = ask(build(server))
    assert answer == Answer(color="teal", score=0.9)
    assert len(server.requests) == 2
    retry_messages = server.requests[1].body["messages"]
    assert retry_messages[-2] == {"role": "assistant", "content": "not json at all"}
    assert "That did not validate" in retry_messages[-1]["content"]
    assert "asking for a correction" in caplog.text


def test_two_invalid_replies_raise(caplog):
    server = FakeServer(openai_reply("nope"), openai_reply('{"color": 5}'))
    with pytest.raises(LLMError, match="could not be parsed as Answer"):
        ask(build(server))
    assert len(server.requests) == 2


def test_a_reply_cut_off_by_the_token_limit_says_so_and_is_not_retried():
    server = FakeServer(openai_reply('{"color": "te', finish="length"))
    with pytest.raises(LLMError, match=r"cut off at max_tokens \(8000\).*llm.max_tokens"):
        ask(build(server))
    assert len(server.requests) == 1


def test_a_large_request_warns_once_about_the_context_window(caplog):
    server = FakeServer(openai_reply(GOOD))
    client = build(server, context_tokens=1024)
    with caplog.at_level(logging.WARNING, logger="mvdirector.llm"):
        ask(client, [text_block("word " * 1000)])
        ask(client, [text_block("word " * 1000)])
    assert caplog.text.count("context window is 1024") == 1


# ============================================================ OpenAI protocol


def test_openai_request_shape():
    server = FakeServer(openai_reply(GOOD))
    ask(
        build(server, provider="openai", model="m1", max_tokens=1234),
        [text_block("hi"), a_png_block()],
    )
    request = server.requests[0]
    assert request.url == "https://api.openai.com/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer sk-test"
    body = request.body
    assert body["model"] == "m1" and body["max_completion_tokens"] == 1234
    assert "max_tokens" not in body and body["response_format"] == {"type": "json_object"}
    system, user = body["messages"]
    assert system["role"] == "system" and "Reply with a single JSON object" in system["content"]
    assert user["content"][0] == {"type": "text", "text": "hi"}
    image = user["content"][1]
    assert image["type"] == "image_url" and image["image_url"]["url"].startswith(
        "data:image/jpeg;base64,"
    )


def test_compatible_servers_get_max_tokens_and_no_response_format_by_default():
    server = FakeServer(openai_reply(GOOD))
    ask(build(server, max_tokens=900))
    body = server.requests[0].body
    assert body["max_tokens"] == 900 and "max_completion_tokens" not in body
    assert "response_format" not in body
    assert "authorization" not in server.requests[0].headers  # local servers need no key


@pytest.mark.parametrize(
    ("base_url", "expected"),
    [
        ("http://localhost:1234/v1/", "http://localhost:1234/v1/chat/completions"),
        ("http://localhost:1234/v1/chat/completions", "http://localhost:1234/v1/chat/completions"),
    ],
)
def test_compatible_base_urls_are_tidied(base_url, expected):
    server = FakeServer(openai_reply(GOOD))
    ask(build(server, base_url=base_url))
    assert server.requests[0].url == expected


def test_json_mode_can_be_set_explicitly():
    server = FakeServer(openai_reply(GOOD))
    ask(build(server, json_mode="schema"))
    response_format = server.requests[0].body["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["name"] == "Answer"
    server = FakeServer(openai_reply(GOOD))
    ask(build(server, provider="openai", json_mode="off"))
    assert "response_format" not in server.requests[0].body


def test_a_server_that_rejects_response_format_is_asked_again_without_it(caplog):
    rejection = error_reply(400, "'response_format' is not supported by this server")
    server = FakeServer(rejection, openai_reply(GOOD), openai_reply(GOOD))
    client = build(server, json_mode="object")
    with caplog.at_level(logging.WARNING, logger="mvdirector.llm"):
        assert ask(client).color == "teal"
    assert (
        "response_format" in server.requests[0].body
        and "response_format" not in server.requests[1].body
    )
    ask(client)  # remembered: not sent again
    assert "response_format" not in server.requests[2].body
    assert "does not accept response_format" in caplog.text


def test_json_schema_rejections_are_also_retried_without_json_mode():
    rejection = error_reply(400, "'json_schema' response type is unsupported")
    server = FakeServer(rejection, openai_reply(GOOD))
    assert ask(build(server, json_mode="schema")).color == "teal"
    assert "response_format" not in server.requests[1].body


def test_a_reasoning_models_think_block_is_ignored():
    reply = openai_reply('<think>The color might be {red}.</think>\n{"color": "teal"}')
    assert ask(build(FakeServer(reply))).color == "teal"


def test_content_given_as_a_list_of_parts_is_joined():
    reply = httpx.Response(
        200,
        json={
            "choices": [
                {"message": {"content": [{"type": "text", "text": GOOD}]}, "finish_reason": "stop"}
            ]
        },
    )
    assert ask(build(FakeServer(reply))).color == "teal"


def test_refusals_and_empty_answers_are_reported():
    refused = httpx.Response(
        200,
        json={
            "choices": [
                {"message": {"content": None, "refusal": "I can't help"}, "finish_reason": "stop"}
            ]
        },
    )
    with pytest.raises(LLMError, match="refused.*I can't help"):
        ask(build(FakeServer(refused)))
    with pytest.raises(LLMError, match="no choices"):
        ask(build(FakeServer(httpx.Response(200, json={"choices": []}))))


@pytest.mark.parametrize(
    ("status", "message", "needle"),
    [
        (401, "Incorrect API key", "rejected the credentials.*OPENAI_API_KEY"),
        (404, "model not found", "could not find the model"),
        (500, "boom", "HTTP 500: boom"),
    ],
)
def test_http_errors_become_actionable_messages(status, message, needle):
    server = FakeServer(error_reply(status, message))
    with pytest.raises(LLMError, match=needle):
        ask(build(server, provider="openai"))


def test_a_rejected_image_is_reported_as_unsupported_only_when_images_were_sent():
    rejection = error_reply(400, "This model does not support image input")
    with pytest.raises(ImagesNotSupportedError, match="may not accept images"):
        ask(build(FakeServer(rejection)), [text_block("look"), a_png_block()])
    with pytest.raises(LLMError, match="HTTP 400") as info:
        ask(build(FakeServer(rejection)), [text_block("no image")])
    assert not isinstance(info.value, ImagesNotSupportedError)


def test_an_unreachable_server_is_retried_then_explained():
    sleeps: list[float] = []
    server = FakeServer(httpx.ConnectError("refused"))
    with pytest.raises(
        LLMError, match=r"Could not reach http://localhost:1234 \(refused\).*running"
    ):
        ask(build(server, sleeps))
    assert len(server.requests) == 3 and sleeps == [2.0, 4.0]


def test_rate_limits_are_retried_and_honor_retry_after():
    sleeps: list[float] = []
    limited = httpx.Response(
        429, headers={"retry-after": "7"}, json={"error": {"message": "slow down"}}
    )
    server = FakeServer(limited, openai_reply(GOOD))
    assert ask(build(server, sleeps)).color == "teal"
    assert sleeps == [7.0] and len(server.requests) == 2


def test_persistent_server_errors_give_up_with_the_servers_message():
    server = FakeServer(error_reply(503, "overloaded"))
    with pytest.raises(LLMError, match="HTTP 503: overloaded"):
        ask(build(server))
    assert len(server.requests) == 3


def test_a_timeout_is_not_retried_and_names_the_setting():
    server = FakeServer(httpx.ReadTimeout("slow"))
    with pytest.raises(LLMError, match="timed out after 600 s.*llm.timeout_seconds"):
        ask(build(server))
    assert len(server.requests) == 1


# ================================================================== Ollama


def ollama(server, **fields):
    return build(server, **{"provider": "ollama", "model": "llava", **fields})


def test_ollama_request_shape():
    server = FakeServer(ollama_reply(GOOD))
    ask(ollama(server, max_tokens=777), [text_block("first"), a_png_block(), text_block("then")])
    request = server.requests[0]
    assert request.url == "http://localhost:11434/api/chat"
    body = request.body
    assert body["stream"] is False and body["format"] == "json" and body["model"] == "llava"
    assert body["options"] == {"num_ctx": 16384, "num_predict": 777}
    system, user = body["messages"]
    assert system["role"] == "system"
    assert user["content"] == "first\n\n[Image 1]\n\nthen"
    assert len(user["images"]) == 1 and not user["images"][0].startswith("data:")


def test_ollama_context_size_is_configurable_and_schema_mode_sends_the_schema():
    server = FakeServer(ollama_reply(GOOD))
    ask(ollama(server, context_tokens=32768, json_mode="schema"))
    body = server.requests[0].body
    assert body["options"]["num_ctx"] == 32768
    assert body["format"]["properties"].keys() == {"color", "score"}


def test_ollama_reports_truncated_prompts_and_replies(caplog):
    server = FakeServer(ollama_reply(GOOD, prompt_eval_count=16384))
    with caplog.at_level(logging.WARNING, logger="mvdirector.llm"):
        ask(ollama(server))
    assert "used 16384 of 16384 context tokens" in caplog.text
    with pytest.raises(LLMError, match="cut off at max_tokens"):
        ask(ollama(FakeServer(ollama_reply('{"color": "te', done_reason="length"))))


def test_ollama_errors_say_what_to_do():
    with pytest.raises(LLMError, match="ollama pull llava"):
        ask(ollama(FakeServer(error_reply(404, "model 'llava' not found", "ollama"))))
    unreachable = FakeServer(httpx.ConnectError("refused"))
    with pytest.raises(LLMError, match="ollama serve"):
        ask(ollama(unreachable))
    rejection = error_reply(400, "this model does not support images", "ollama")
    with pytest.raises(ImagesNotSupportedError):
        ask(ollama(FakeServer(rejection)), [a_png_block()])


def test_ollama_retries_without_format_when_it_is_rejected():
    server = FakeServer(error_reply(400, "invalid format", "ollama"), ollama_reply(GOOD))
    assert ask(ollama(server)).color == "teal"
    assert "format" not in server.requests[1].body


def test_ollama_retries_without_format_on_501_structured_output_unavailable():
    server = FakeServer(
        error_reply(501, "structured output is unavailable", "ollama"),
        ollama_reply(GOOD),
        ollama_reply(GOOD),
    )
    client = ollama(server)
    assert ask(client).color == "teal"
    assert "format" in server.requests[0].body
    assert "format" not in server.requests[1].body
    # Subsequent requests skip format immediately without another 501 round-trip
    assert ask(client).color == "teal"
    assert "format" not in server.requests[2].body


def test_an_image_format_error_does_not_switch_off_ollamas_json_mode():
    server = FakeServer(error_reply(400, "unsupported image format: webp", "ollama"))
    with pytest.raises(ImagesNotSupportedError):
        ask(ollama(server), [a_png_block()])
    assert len(server.requests) == 1  # not retried without format


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("http://localhost:11434/api", "http://localhost:11434/api/chat"),
        ("http://box:11434/", "http://box:11434/api/chat"),
    ],
)
def test_ollama_base_urls_are_tidied(given, expected):
    server = FakeServer(ollama_reply(GOOD))
    ask(ollama(server, base_url=given))
    assert server.requests[0].url == expected


def test_ollama_v1_address_is_refused_with_advice():
    with pytest.raises(ConfigError, match="openai_compatible"):
        ollama(FakeServer(ollama_reply(GOOD)), base_url="http://localhost:11434/v1")


# ================================================================== Gemini


def gemini(server, **fields):
    return build(server, **{"provider": "gemini", "model": "gemini-test", **fields})


def test_gemini_request_shape():
    server = FakeServer(gemini_reply(GOOD))
    ask(gemini(server, max_tokens=555), [text_block("look"), a_png_block()])
    request = server.requests[0]
    assert request.url == (
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-test:generateContent"
    )
    assert request.headers["x-goog-api-key"] == "g-test"
    body = request.body
    assert "Reply with a single JSON object" in body["systemInstruction"]["parts"][0]["text"]
    assert body["generationConfig"] == {
        "maxOutputTokens": 555,
        "responseMimeType": "application/json",
    }
    content = body["contents"][0]
    assert content["role"] == "user" and content["parts"][0] == {"text": "look"}
    assert content["parts"][1]["inlineData"]["mimeType"] == "image/jpeg"


def test_gemini_model_prefix_is_accepted_and_json_mode_can_be_off():
    server = FakeServer(gemini_reply(GOOD))
    ask(gemini(server, model="models/gemini-test", json_mode="off"))
    assert server.requests[0].url.endswith("/models/gemini-test:generateContent")
    assert "responseMimeType" not in server.requests[0].body["generationConfig"]


def test_gemini_correction_turn_uses_the_model_role():
    server = FakeServer(gemini_reply("bad"), gemini_reply(GOOD))
    ask(gemini(server))
    roles = [c["role"] for c in server.requests[1].body["contents"]]
    assert roles == ["user", "model", "user"]


def test_gemini_ignores_thought_parts_and_reports_blocks_and_truncation():
    thoughts = httpx.Response(
        200,
        json={
            "candidates": [
                {"content": {"parts": [{"text": "thinking...", "thought": True}, {"text": GOOD}]}}
            ]
        },
    )
    assert ask(gemini(FakeServer(thoughts))).color == "teal"
    blocked_prompt = httpx.Response(200, json={"promptFeedback": {"blockReason": "SAFETY"}})
    with pytest.raises(LLMError, match="no answer.*SAFETY"):
        ask(gemini(FakeServer(blocked_prompt)))
    blocked_reply = httpx.Response(
        200, json={"candidates": [{"content": {"parts": []}, "finishReason": "SAFETY"}]}
    )
    with pytest.raises(LLMError, match="blocked the reply"):
        ask(gemini(FakeServer(blocked_reply)))
    with pytest.raises(LLMError, match="cut off at max_tokens"):
        ask(gemini(FakeServer(gemini_reply('{"color": "te', finish="MAX_TOKENS"))))


def test_gemini_errors_name_the_key_variable_and_the_image_problem():
    with pytest.raises(LLMError, match="rejected the credentials.*GEMINI_API_KEY"):
        ask(gemini(FakeServer(error_reply(403, "API key not valid"))))
    with pytest.raises(ImagesNotSupportedError):
        ask(gemini(FakeServer(error_reply(400, "Image input is not supported"))), [a_png_block()])


# ================================================================ Anthropic


class FakeAnthropic:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls: list[dict] = []
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


@dataclass
class _Block:
    type: str
    text: str


@dataclass
class _Response:
    content: list
    stop_reason: str = "end_turn"


def anthropic_backend(client, **fields):
    return AnthropicChat(model="claude-x", max_tokens=321, client=client, **fields)


def test_anthropic_passes_neutral_blocks_straight_through():
    client = FakeAnthropic(_Response([_Block("text", GOOD)]))
    assert ask(anthropic_backend(client), [text_block("hi"), a_png_block()]).color == "teal"
    call = client.calls[0]
    assert call["model"] == "claude-x" and call["max_tokens"] == 321
    assert "Reply with a single JSON object" in call["system"]
    assert call["messages"][0]["content"][1]["type"] == "image"


def test_anthropic_truncation_and_api_errors():
    client = FakeAnthropic(_Response([_Block("text", '{"color": "te')], stop_reason="max_tokens"))
    with pytest.raises(LLMError, match="cut off at max_tokens"):
        ask(anthropic_backend(client))
    failure = anthropic.APIError("boom", httpx.Request("POST", "http://x"), body=None)
    with pytest.raises(LLMError, match="Anthropic API call failed: boom"):
        ask(anthropic_backend(FakeAnthropic(failure)))


# ============================================================ building clients


def test_disabled_llm_builds_nothing():
    assert build_llm(LLMConfig(enabled=False)) is None


@pytest.mark.parametrize(
    ("fields", "cls"),
    [
        ({"provider": "openai", "model": "m"}, OpenAIChat),
        ({"provider": "openai_compatible", "model": "m", "base_url": "http://h/v1"}, OpenAIChat),
        ({"provider": "ollama", "model": "m"}, OllamaChat),
        ({"provider": "gemini", "model": "m"}, GeminiChat),
    ],
)
def test_each_provider_builds_its_backend(fields, cls):
    assert isinstance(build(FakeServer(openai_reply(GOOD)), **fields), cls)


def test_ollama_is_the_default_provider_and_needs_no_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert isinstance(build_llm(LLMConfig()), OllamaChat)


@pytest.mark.parametrize(
    ("fields", "variable"),
    [
        ({"provider": "anthropic"}, "ANTHROPIC_API_KEY"),
        ({"provider": "openai"}, "OPENAI_API_KEY"),
        ({"provider": "gemini"}, "GEMINI_API_KEY"),
        (
            {
                "provider": "openai_compatible",
                "base_url": "http://h/v1",
                "api_key_env": "MVD_LLM_KEY",
            },
            "MVD_LLM_KEY",
        ),
    ],
)
def test_a_missing_key_names_the_variable_to_set(fields, variable):
    with pytest.raises(ConfigError, match=variable):
        build_llm(LLMConfig(model="m", **fields), environ={})


def test_local_providers_need_no_key():
    server = FakeServer(ollama_reply(GOOD))
    assert ask(ollama(server, environ={})).color == "teal"
    assert "authorization" not in server.requests[0].headers


def test_the_key_variable_is_configurable_and_sent_as_a_bearer_token():
    server = FakeServer(openai_reply(GOOD))
    client = build(
        server, provider="openai", api_key_env="MVD_WORK_KEY", environ={"MVD_WORK_KEY": "sk-work"}
    )
    ask(client)
    assert server.requests[0].headers["authorization"] == "Bearer sk-work"


def test_gemini_falls_back_to_the_google_api_key_variable():
    server = FakeServer(gemini_reply(GOOD))
    ask(gemini(server, environ={"GOOGLE_API_KEY": "google-key"}))
    assert server.requests[0].headers["x-goog-api-key"] == "google-key"


@pytest.mark.parametrize(
    ("provider", "extra", "has_format"),
    [("openai", {}, True), ("openai_compatible", {"base_url": "http://h/v1"}, False)],
)
def test_json_mode_auto_depends_on_the_provider(provider, extra, has_format):
    server = FakeServer(openai_reply(GOOD))
    ask(build(server, provider=provider, **extra))
    assert ("response_format" in server.requests[0].body) is has_format


def test_mvd_llm_url_overrides_default_address_and_satisfies_openai_compatible(monkeypatch):
    server = FakeServer(ollama_reply(GOOD))
    client = build_llm(
        LLMConfig(provider="ollama", model="qwen2.5vl:7b"),
        environ={"MVD_LLM_URL": "http://ollama:11434/"},
        transport=server.transport(),
    )
    ask(client)
    assert server.requests[0].url == "http://ollama:11434/api/chat"

    monkeypatch.setenv("MVD_LLM_URL", "http://vllm:8000/v1")
    compat_config = LLMConfig(provider="openai_compatible", model="Qwen/Qwen2.5-VL-7B-Instruct")
    assert compat_config.base_url == "http://vllm:8000/v1"


def test_config_validation(monkeypatch):
    monkeypatch.delenv("MVD_LLM_URL", raising=False)
    with pytest.raises(ValueError, match="base_url is required"):
        LLMConfig(provider="openai_compatible", model="m")
    with pytest.raises(ValueError):
        LLMConfig(provider="mystery")
    with pytest.raises(ValueError, match="model must not be empty"):
        LLMConfig(model="  ")
    defaults = LLMConfig()
    assert (defaults.provider, defaults.json_mode, defaults.timeout_seconds) == (
        "ollama",
        "auto",
        600.0,
    )


# ============================================================== vision fallback


class ImagelessLLM:
    """A model that rejects images, counting how often it is asked."""

    def __init__(self):
        self.calls = 0

    def complete_json(self, *, system, content, schema):
        self.calls += 1
        raise ImagesNotSupportedError("server said: no image input")


def test_vision_checks_switch_themselves_off_with_one_warning(caplog):
    from mvdirector.models import Manifest
    from tests.test_vision import _shot  # reuse the shot builder

    manifest = Manifest(title="t", duration=4.0, fps=24)
    llm = ImagelessLLM()
    judge = VisionJudge(llm)
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    with caplog.at_level(logging.WARNING):
        first = vision_checks(judge, [image], _shot(), manifest, AssetKind.STILL, 0.3, 0.4)
        second = vision_checks(judge, [image], _shot(), manifest, AssetKind.STILL, 0.3, 0.4)
    assert all(c.status.value == "skipped" and "does not accept images" in c.detail for c in first)
    assert [c.detail for c in second] == [c.detail for c in first]
    assert llm.calls == 1  # the second batch never reached the model
    assert caplog.text.count("Vision checks are off for the rest of this run") == 1
    assert judge.disabled_reason


# ================================================================== llm-check


class CheckLLM:
    def __init__(self, *, json_ok=True, vision="ok", color="blue"):
        self.json_ok, self.vision, self.color = json_ok, vision, color

    def complete_json(self, *, system, content, schema):
        has_image = any(b["type"] == "image" for b in content)
        if not has_image:
            if not self.json_ok:
                raise LLMError("401 from server")
            return schema(ok=True, word="ready")
        if self.vision == "unsupported":
            raise ImagesNotSupportedError("no images")
        if self.vision == "error":
            raise LLMError("server error")
        return schema(color=self.color)


def test_the_test_image_has_one_large_blue_square():
    image = make_test_image()
    assert image.shape == (128, 128, 3) and tuple(image[64, 64]) == (20, 60, 220)
    assert tuple(image[0, 0]) == (255, 255, 255)


@pytest.mark.parametrize(
    ("client", "json_ok", "vision"),
    [
        (CheckLLM(), True, "ok"),
        (CheckLLM(vision="unsupported"), True, "unsupported"),
        (CheckLLM(vision="error"), True, "error"),
        (CheckLLM(color="green"), True, "wrong"),
        (CheckLLM(json_ok=False), False, "skipped"),
    ],
)
def test_llm_check_outcomes(client, json_ok, vision):
    result = run_llm_check(client)
    assert (result.json_ok, result.vision) == (json_ok, vision)


def test_llm_check_can_skip_the_image_test():
    assert run_llm_check(CheckLLM(vision="unsupported"), check_vision=False).vision == "skipped"


def llm_project(project, monkeypatch, client, **llm_fields):
    """Point the project's config at an LLM and make the CLI use ``client``."""
    path = project.root / "project.yaml"
    data = yaml.safe_load(path.read_text())
    data["llm"].update({"enabled": True, **llm_fields})
    path.write_text(yaml.safe_dump(data))
    monkeypatch.setattr("mvdirector.api.build_llm", lambda config: client)
    return ["llm-check", "-p", str(project.root)]


def test_llm_check_command_passes(project, monkeypatch):
    result = CliRunner().invoke(
        app, llm_project(project, monkeypatch, CheckLLM(), provider="ollama", model="llava")
    )
    assert result.exit_code == 0
    assert "Provider: ollama   Model: llava" in result.output
    assert "JSON output" in result.output and "Image input" in result.output


def test_llm_check_command_explains_a_text_only_model(project, monkeypatch):
    args = llm_project(project, monkeypatch, CheckLLM(vision="unsupported"))
    result = CliRunner().invoke(app, args)
    # Vision checks are on in this config, so a model that cannot read images does not fit it.
    assert result.exit_code == 1
    assert "[FAIL] Image input" in result.output and "qc.vision_enabled: false" in result.output


def test_llm_check_skips_the_image_test_when_vision_checks_are_off(project, monkeypatch):
    args = llm_project(project, monkeypatch, CheckLLM(vision="unsupported"))
    path = project.root / "project.yaml"
    data = yaml.safe_load(path.read_text())
    data["qc"]["vision_enabled"] = False  # the documented setup for a text-only model
    path.write_text(yaml.safe_dump(data))
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0
    assert "FAIL" not in result.output
    assert "Image input not tested: qc.vision_enabled is false" in result.output


def test_llm_check_fails_when_the_image_test_errors(project, monkeypatch):
    result = CliRunner().invoke(app, llm_project(project, monkeypatch, CheckLLM(vision="error")))
    assert result.exit_code == 1 and "[FAIL] Image input" in result.output


def test_llm_check_command_fails_when_json_fails(project, monkeypatch):
    args = llm_project(project, monkeypatch, CheckLLM(json_ok=False))
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 1 and "401 from server" in result.output


def test_llm_check_command_with_skip_image_and_when_disabled(project, monkeypatch):
    args = llm_project(project, monkeypatch, CheckLLM(vision="unsupported"))
    assert "Image input" not in CliRunner().invoke(app, [*args, "--skip-image"]).output
    off = llm_project(project, monkeypatch, CheckLLM(), enabled=False)
    assert "llm.enabled is false" in CliRunner().invoke(app, off).output


def test_llm_check_command_reports_a_missing_key(project, monkeypatch):
    path = project.root / "project.yaml"
    data = yaml.safe_load(path.read_text())
    data["llm"].update({"enabled": True, "provider": "openai", "model": "m"})
    path.write_text(yaml.safe_dump(data))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = CliRunner().invoke(app, ["llm-check", "-p", str(project.root)])
    assert result.exit_code == 1 and "OPENAI_API_KEY" in result.output


# ===================================================== robustness found in review


def test_gemini_reports_a_bad_key_as_a_credentials_problem_even_though_it_is_a_400():
    rejection = error_reply(400, "API key not valid. Please pass a valid API key.")
    with pytest.raises(LLMError, match="rejected the credentials.*GEMINI_API_KEY"):
        ask(gemini(FakeServer(rejection)))


@pytest.mark.parametrize("backend", ["openai_compatible", "ollama", "gemini"])
def test_a_reply_that_is_not_json_is_explained_not_a_traceback(backend):
    html = httpx.Response(200, text="<html>Bad gateway from the proxy</html>")
    with pytest.raises(LLMError, match="not JSON.*Bad gateway"):
        ask(build(FakeServer(html), **({"provider": backend, "model": "m"})))
    listing = httpx.Response(200, json=["unexpected"])
    with pytest.raises(LLMError, match="unexpected reply"):
        ask(build(FakeServer(listing), **({"provider": backend, "model": "m"})))


def test_a_failed_parse_shows_what_the_model_said(caplog):
    server = FakeServer(
        openai_reply("Sorry, I cannot do that."), openai_reply("Still no JSON, sorry.")
    )
    with (
        caplog.at_level(logging.DEBUG, logger="mvdirector.llm"),
        pytest.raises(LLMError, match=r"The reply began: 'Still no JSON, sorry\.'.*-v"),
    ):
        ask(build(server))
    assert "Sorry, I cannot do that." in caplog.text  # visible with -v


def test_old_project_files_with_only_the_original_llm_keys_still_load(tmp_path):
    from mvdirector.config import load_config

    path = tmp_path / "project.yaml"
    path.write_text(
        "llm:\n  enabled: true\n  provider: anthropic\n  model: claude-sonnet-5-5\n"
        "  max_tokens: 8000\n  shots_per_request: 20\n"
    )
    llm = load_config(path).llm
    assert (llm.provider, llm.base_url, llm.api_key_env, llm.json_mode) == (
        "anthropic",
        None,
        None,
        "auto",
    )


def test_an_unknown_llm_key_is_still_an_error(tmp_path):
    from mvdirector.config import load_config

    path = tmp_path / "project.yaml"
    path.write_text("llm:\n  provder: openai\n")  # misspelled
    with pytest.raises(ConfigError, match="provder"):
        load_config(path)


def test_a_connect_timeout_is_retried_like_an_unreachable_server():
    sleeps: list[float] = []
    server = FakeServer(httpx.ConnectTimeout("no route"))
    with pytest.raises(LLMError, match="Could not reach.*connection timed out"):
        ask(build(server, sleeps))
    assert len(server.requests) == 3 and sleeps == [2.0, 4.0]


def test_a_malformed_retry_after_falls_back_to_the_normal_pause():
    sleeps: list[float] = []
    limited = httpx.Response(503, headers={"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"})
    ask(build(FakeServer(limited, openai_reply(GOOD)), sleeps))
    assert sleeps == [2.0]


def test_plain_text_error_bodies_are_shown():
    server = FakeServer(httpx.Response(502, text="Bad Gateway from upstream"))
    with pytest.raises(LLMError, match="HTTP 502: Bad Gateway from upstream"):
        ask(build(server))


def test_persistent_rate_limiting_is_reported():
    server = FakeServer(error_reply(429, "quota exceeded"))
    with pytest.raises(LLMError, match="rate limiting requests \\(quota exceeded\\)"):
        ask(build(server))
    assert len(server.requests) == 3


def test_a_content_filtered_reply_is_reported():
    blocked = httpx.Response(
        200,
        json={"choices": [{"message": {"content": ""}, "finish_reason": "content_filter"}]},
    )
    with pytest.raises(LLMError, match="content filter"):
        ask(build(FakeServer(blocked), provider="openai"))


def test_anthropic_accepts_a_custom_address_and_key():
    backend = AnthropicChat(
        model="claude-x", max_tokens=300, api_key="k-test", base_url="http://proxy.local:8080"
    )
    url = backend._client.base_url
    assert (url.scheme, url.host, url.port) == ("http", "proxy.local", 8080)
    assert backend._client.api_key == "k-test"


def test_build_llm_rejects_custom_transport_for_anthropic():
    transport = httpx.MockTransport(lambda req: httpx.Response(200, json={}))
    with pytest.raises(ConfigError, match="transport is not supported"):
        build_llm(
            LLMConfig(provider="anthropic", model="claude-x"),
            environ={"ANTHROPIC_API_KEY": "k-test"},
            transport=transport,
        )


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        (
            {"object": "error", "message": "model 'x' is not loaded"},
            "model 'x' is not loaded",
        ),  # vLLM style
        ({"detail": "Not Found"}, "{'detail': 'Not Found'}"),
        ({"error": {"code": 7}}, "{'code': 7}"),
    ],
)
def test_error_messages_come_from_whichever_json_shape_the_server_uses(body, expected):
    from mvdirector.llm.http import error_message

    assert error_message(httpx.Response(400, json=body)) == expected
