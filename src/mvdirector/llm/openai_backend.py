# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""OpenAI Chat Completions, and any server that speaks the same protocol.

That covers OpenAI itself and local or hosted servers such as LM Studio, llama.cpp, vLLM and
OpenRouter. Images are sent as base64 data URLs.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel

from mvdirector.errors import LLMError
from mvdirector.llm.base import ChatBackend, Message, Reply, blocks_of, has_images
from mvdirector.llm.http import (
    IMAGE_WORDS,
    HttpJSON,
    check_response,
    error_message,
    json_body,
    redact_url,
)

log = logging.getLogger("mvdirector.llm")


_JSON_MODE_WORDS = ("response_format", "json_schema", "json_object")


def _mentions_json_mode(text: str) -> bool:
    lowered = text.lower()
    return any(word in lowered for word in _JSON_MODE_WORDS)


class OpenAIChat(ChatBackend):
    def __init__(
        self,
        *,
        provider: str,
        model: str,
        max_tokens: int,
        base_url: str,
        http: HttpJSON,
        json_mode: str,
        token_param: str,
        key_hint: str,
        context_tokens: int | None = None,
    ) -> None:
        super().__init__(model=model, max_tokens=max_tokens, context_tokens=context_tokens)
        self._provider = provider
        self._url = f"{base_url}/chat/completions"
        self._http = http
        self._json_mode = json_mode  # "off", "object" or "schema"; never "auto" here
        self._token_param = token_param
        self._key_hint = key_hint

    # ------------------------------------------------------------------ request

    @staticmethod
    def _wire_messages(system: str, messages: list[Message]) -> list[dict[str, Any]]:
        wire: list[dict[str, Any]] = [{"role": "system", "content": system}]
        for message in messages:
            content = message["content"]
            if isinstance(content, str):
                wire.append({"role": message["role"], "content": content})
                continue
            parts: list[dict[str, Any]] = []
            for block in blocks_of(content):
                if block["type"] == "image":
                    source = block["source"]
                    url = f"data:{source['media_type']};base64,{source['data']}"
                    parts.append({"type": "image_url", "image_url": {"url": url}})
                else:
                    parts.append({"type": "text", "text": block["text"]})
            wire.append({"role": message["role"], "content": parts})
        return wire

    def _response_format(self, schema: type[BaseModel]) -> dict[str, Any] | None:
        if self._json_mode == "object":
            return {"type": "json_object"}
        if self._json_mode == "schema":
            return {
                "type": "json_schema",
                "json_schema": {"name": schema.__name__, "schema": schema.model_json_schema()},
            }
        return None

    def _send(self, system: str, messages: list[Message], schema: type[BaseModel]) -> Reply:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": self._wire_messages(system, messages),
            self._token_param: self._max_tokens,
        }
        response_format = self._response_format(schema)
        if response_format:
            payload["response_format"] = response_format
        safe_base = redact_url(self._url.rsplit("/chat/completions", 1)[0])
        hint = f"Is the server running at {safe_base}?"
        images = has_images(messages)
        words = IMAGE_WORDS if images else ()
        response = self._http.post(self._url, payload, unreachable_hint=hint, no_retry_words=words)
        if response_format and response.status_code == 400 and _mentions_json_mode(response.text):
            # Some servers do not support it. Remember that and ask in the prompt alone.
            log.warning(
                "%s does not accept response_format (%s); continuing with the prompt alone. "
                "Set llm.json_mode: off to silence this.",
                self._provider,
                error_message(response),
            )
            self._json_mode = "off"
            payload.pop("response_format")
            response = self._http.post(
                self._url, payload, unreachable_hint=hint, no_retry_words=words
            )
        check_response(
            response, provider=self._provider, key_hint=self._key_hint, had_images=images
        )
        return self._parse(json_body(response, self._provider))

    # ----------------------------------------------------------------- response

    def _parse(self, data: dict[str, Any]) -> Reply:
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise LLMError(f"{self._provider} returned no choices: {str(data)[:200]}")
        choice = choices[0]
        if not isinstance(choice, dict):
            raise LLMError(f"{self._provider} returned an unexpected choice: {choice!r}")
        raw_message = choice.get("message")
        message = raw_message if isinstance(raw_message, dict) else {}
        content = message.get("content")
        if isinstance(content, list):  # some servers return a list of parts
            text = "".join(
                part["text"]
                for part in content
                if isinstance(part, dict) and isinstance(part.get("text"), str)
            )
        elif isinstance(content, str):
            text = content
        elif content is None:
            text = ""
        else:
            raise LLMError(
                f"{self._provider} returned non-text message content: {type(content).__name__}"
            )
        finish = choice.get("finish_reason")
        if not text and message.get("refusal"):
            raise LLMError(f"The model refused the request: {message['refusal']}")
        if not text and finish == "content_filter":
            raise LLMError(f"{self._provider} blocked the reply with its content filter.")
        return Reply(text, truncated=finish == "length")
