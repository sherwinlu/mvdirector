# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Google Gemini, through the Generative Language REST API."""

from __future__ import annotations

import urllib.parse
from typing import Any

from pydantic import BaseModel

from mvdirector.errors import ConfigError, LLMError
from mvdirector.llm.base import ChatBackend, Message, Reply, blocks_of, has_images
from mvdirector.llm.http import IMAGE_WORDS, HttpJSON, check_response, json_body

_BLOCKED = {"SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "IMAGE_SAFETY"}
_THINKING_RETRY_BUDGET = 1024


def _is_thinking_config_error(text: str) -> bool:
    lowered = text.lower()
    return (
        "thinking" in lowered or "thinkingconfig" in lowered or "thinkingbudget" in lowered
    ) and (
        "unknown" in lowered
        or "invalid" in lowered
        or "unsupported" in lowered
        or "not support" in lowered
    )


def _thoughts_token_count(data: dict[str, Any]) -> int:
    usage = data.get("usageMetadata")
    if not isinstance(usage, dict):
        return 0
    raw = usage.get("thoughtsTokenCount")
    return int(raw) if isinstance(raw, int) and raw > 0 else 0


class GeminiChat(ChatBackend):
    supports_audio: bool = True

    def __init__(
        self,
        *,
        model: str,
        max_tokens: int,
        base_url: str,
        http: HttpJSON,
        json_mode: str,
        key_hint: str,
        context_tokens: int | None = None,
    ) -> None:
        super().__init__(model=model, max_tokens=max_tokens, context_tokens=context_tokens)
        name = model.strip().removeprefix("models/")
        if (
            not name
            or "/" in name
            or "\\" in name
            or ".." in name
            or "?" in name
            or "#" in name
            or any(ord(ch) < 32 or ord(ch) == 127 or ch.isspace() for ch in name)
        ):
            raise ConfigError(f"Invalid Gemini model name: {model!r}")
        encoded = urllib.parse.quote(name, safe="")
        self._url = f"{base_url}/models/{encoded}:generateContent"
        self._http = http
        self._json_mode = json_mode  # "off" or anything else for JSON output
        self._key_hint = key_hint

    @staticmethod
    def _wire_contents(messages: list[Message]) -> list[dict[str, Any]]:
        contents = []
        for message in messages:
            parts: list[dict[str, Any]] = []
            for block in blocks_of(message["content"]):
                if block["type"] in ("image", "audio"):
                    source = block["source"]
                    parts.append(
                        {"inlineData": {"mimeType": source["media_type"], "data": source["data"]}}
                    )
                else:
                    parts.append({"text": block["text"]})
            role = "model" if message["role"] == "assistant" else "user"
            contents.append({"role": role, "parts": parts})
        return contents

    def _send(self, system: str, messages: list[Message], schema: type[BaseModel]) -> Reply:
        config: dict[str, Any] = {"maxOutputTokens": self._max_tokens}
        if self._json_mode != "off":
            config["responseMimeType"] = "application/json"
        payload = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": self._wire_contents(messages),
            "generationConfig": config,
        }
        images = has_images(messages)
        response = self._http.post(
            self._url,
            payload,
            unreachable_hint="Check your network connection.",
            no_retry_words=IMAGE_WORDS if images else (),
        )
        check_response(
            response,
            provider="Gemini",
            key_hint=self._key_hint,
            had_images=images,
            not_found_hint="Check llm.model, for example gemini-... as listed in Google AI Studio.",
        )
        data = json_body(response, "Gemini")
        reply = self._parse(data)
        if reply.truncated and _thoughts_token_count(data) > 0:
            retry_config = {**config, "thinkingConfig": {"thinkingBudget": _THINKING_RETRY_BUDGET}}
            retry_payload = {**payload, "generationConfig": retry_config}
            retry_response = self._http.post(
                self._url,
                retry_payload,
                unreachable_hint="Check your network connection.",
                no_retry_words=IMAGE_WORDS if images else (),
            )
            if retry_response.status_code == 400 and _is_thinking_config_error(retry_response.text):
                return reply
            check_response(
                retry_response,
                provider="Gemini",
                key_hint=self._key_hint,
                had_images=images,
                not_found_hint=(
                    "Check llm.model, for example gemini-... as listed in Google AI Studio."
                ),
            )
            return self._parse(json_body(retry_response, "Gemini"))
        return reply

    @staticmethod
    def _parse(data: dict[str, Any]) -> Reply:
        candidates = data.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            feedback = data.get("promptFeedback")
            reason = (
                feedback.get("blockReason", "no reason given")
                if isinstance(feedback, dict)
                else "no reason given"
            )
            raise LLMError(f"Gemini returned no answer ({reason}).")
        candidate = candidates[0]
        if not isinstance(candidate, dict):
            raise LLMError(f"Gemini returned an unexpected candidate shape: {candidate!r}")
        content = candidate.get("content")
        parts = content.get("parts") if isinstance(content, dict) else None
        if parts is not None and not isinstance(parts, list):
            raise LLMError(f"Gemini returned an unexpected parts field: {parts!r}")
        # Parts flagged as thoughts are the model's reasoning, not its answer.
        text_chunks: list[str] = []
        for part in parts or []:
            if not isinstance(part, dict) or part.get("thought"):
                continue
            chunk = part.get("text")
            if isinstance(chunk, str):
                text_chunks.append(chunk)
        text = "".join(text_chunks)
        finish = candidate.get("finishReason")
        if not text and finish in _BLOCKED:
            raise LLMError(f"Gemini blocked the reply ({finish}).")
        return Reply(text, truncated=finish == "MAX_TOKENS")
