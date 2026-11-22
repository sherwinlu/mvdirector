# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Google Gemini, through the Generative Language REST API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from mvdirector.errors import LLMError
from mvdirector.llm.base import ChatBackend, Message, Reply, blocks_of, has_images
from mvdirector.llm.http import IMAGE_WORDS, HttpJSON, check_response, json_body

_BLOCKED = {"SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "IMAGE_SAFETY"}


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
        name = model.removeprefix("models/")
        self._url = f"{base_url}/models/{name}:generateContent"
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
        return self._parse(json_body(response, "Gemini"))

    @staticmethod
    def _parse(data: dict[str, Any]) -> Reply:
        candidates = data.get("candidates") or []
        if not candidates:
            reason = (data.get("promptFeedback") or {}).get("blockReason", "no reason given")
            raise LLMError(f"Gemini returned no answer ({reason}).")
        candidate = candidates[0]
        parts = (candidate.get("content") or {}).get("parts") or []
        # Parts flagged as thoughts are the model's reasoning, not its answer.
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        finish = candidate.get("finishReason")
        if not text and finish in _BLOCKED:
            raise LLMError(f"Gemini blocked the reply ({finish}).")
        return Reply(text, truncated=finish == "MAX_TOKENS")
