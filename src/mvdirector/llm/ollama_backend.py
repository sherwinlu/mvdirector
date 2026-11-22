# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Ollama's native chat API.

The OpenAI-compatible endpoint Ollama also offers cannot set the context window per request, and
Ollama's small default window silently drops the start of a long prompt. The native API can, and
it reports how many prompt tokens were actually used, so truncation can be detected.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel

from mvdirector.llm.base import ChatBackend, Message, Reply, blocks_of, has_images
from mvdirector.llm.http import IMAGE_WORDS, HttpJSON, check_response, error_message, json_body

log = logging.getLogger("mvdirector.llm")

DEFAULT_CONTEXT_TOKENS = 16384
_TRUNCATION_MARGIN = 16


def _is_format_error(text: str) -> bool:
    """True for a rejection of the JSON ``format`` option, not of an image's file format."""
    lowered = text.lower()
    return ("format" in lowered or "structured output" in lowered) and "image" not in lowered


class OllamaChat(ChatBackend):
    def __init__(
        self,
        *,
        model: str,
        max_tokens: int,
        base_url: str,
        http: HttpJSON,
        json_mode: str,
        context_tokens: int | None = None,
    ) -> None:
        self._num_ctx = context_tokens or DEFAULT_CONTEXT_TOKENS
        super().__init__(model=model, max_tokens=max_tokens, context_tokens=self._num_ctx)
        self._url = f"{base_url}/api/chat"
        self._base_url = base_url
        self._http = http
        self._json_mode = json_mode  # "off", "object" or "schema"

    @staticmethod
    def _wire_messages(system: str, messages: list[Message]) -> list[dict[str, Any]]:
        wire: list[dict[str, Any]] = [{"role": "system", "content": system}]
        for message in messages:
            content = message["content"]
            if isinstance(content, str):
                wire.append({"role": message["role"], "content": content})
                continue
            # Ollama takes one text field and a separate list of images, so mark where each
            # image sat in the original order.
            texts: list[str] = []
            images: list[str] = []
            for block in blocks_of(content):
                if block["type"] == "image":
                    images.append(block["source"]["data"])
                    texts.append(f"[Image {len(images)}]")
                else:
                    texts.append(block["text"])
            entry: dict[str, Any] = {"role": message["role"], "content": "\n\n".join(texts)}
            if images:
                entry["images"] = images
            wire.append(entry)
        return wire

    def _send(self, system: str, messages: list[Message], schema: type[BaseModel]) -> Reply:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": self._wire_messages(system, messages),
            "stream": False,
            "options": {"num_ctx": self._num_ctx, "num_predict": self._max_tokens},
        }
        if self._json_mode == "object":
            payload["format"] = "json"
        elif self._json_mode == "schema":
            payload["format"] = schema.model_json_schema()
        hint = f"Is Ollama running at {self._base_url}? Start it with `ollama serve`."
        images = has_images(messages)
        words = IMAGE_WORDS if images else ()
        response = self._http.post(self._url, payload, unreachable_hint=hint, no_retry_words=words)
        if (
            "format" in payload
            and response.status_code in (400, 501)
            and _is_format_error(response.text)
        ):
            log.warning(
                "Ollama does not accept this JSON format option (%s); continuing with the prompt "
                "alone. Set llm.json_mode: off to silence this.",
                error_message(response),
            )
            self._json_mode = "off"
            payload.pop("format")
            response = self._http.post(
                self._url, payload, unreachable_hint=hint, no_retry_words=words
            )
        check_response(
            response,
            provider="Ollama",
            key_hint="",
            had_images=images,
            not_found_hint=f"Pull the model first with `ollama pull {self._model}`.",
        )
        data = json_body(response, "Ollama")
        used = data.get("prompt_eval_count")
        if isinstance(used, int) and used >= self._num_ctx - _TRUNCATION_MARGIN:
            log.warning(
                "The prompt used %d of %d context tokens, so Ollama probably dropped part of it. "
                "Raise llm.context_tokens (and check the model supports it) or lower "
                "llm.shots_per_request.",
                used,
                self._num_ctx,
            )
        text = (data.get("message") or {}).get("content") or ""
        return Reply(text, truncated=data.get("done_reason") == "length")
