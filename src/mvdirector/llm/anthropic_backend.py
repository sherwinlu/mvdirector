# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Anthropic Messages API, through the official SDK."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from mvdirector.errors import DependencyMissing, LLMError, LLMQuotaError, LLMUnavailableError
from mvdirector.llm.base import ChatBackend, Message, Reply, TokenUsage, positive_int
from mvdirector.llm.http import is_quota_error


def _anthropic_usage(response: Any) -> TokenUsage | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    inp = positive_int(getattr(usage, "input_tokens", None))
    out = positive_int(getattr(usage, "output_tokens", None))
    cached = positive_int(getattr(usage, "cache_read_input_tokens", None))
    if not (inp or out or cached):
        return None
    return TokenUsage(input_tokens=inp, output_tokens=out, cached_tokens=cached)


class AnthropicChat(ChatBackend):
    def __init__(
        self,
        *,
        model: str,
        max_tokens: int,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 180.0,
        context_tokens: int | None = None,
        client: Any = None,
    ) -> None:
        super().__init__(model=model, max_tokens=max_tokens, context_tokens=context_tokens)
        try:
            import anthropic
        except ImportError as exc:
            raise DependencyMissing(
                "The 'anthropic' package is not installed. Install the 'anthropic' extra "
                "(e.g. `pip install 'mvdirector[anthropic]'`) or switch llm.provider in "
                "project.yaml."
            ) from exc

        self._api_error = anthropic.APIError
        # Failures where asking again cannot help: no connection, bad key, unknown model.
        self._unavailable = (
            anthropic.APIConnectionError,
            anthropic.AuthenticationError,
            anthropic.PermissionDeniedError,
            anthropic.NotFoundError,
        )
        self._owned_client = False
        if client is None:
            options: dict[str, Any] = {"timeout": timeout}
            http_client = None
            default_http_cls = getattr(anthropic, "DefaultHttpxClient", None)
            if callable(default_http_cls):
                http_client = default_http_cls(follow_redirects=False, timeout=timeout)
                options["http_client"] = http_client
            if api_key:
                options["api_key"] = api_key
            if base_url:
                options["base_url"] = base_url
            try:
                client = anthropic.Anthropic(**options)
            except BaseException:
                if http_client is not None:
                    http_client.close()
                raise
            self._owned_client = True
        self._client = client

    def close(self) -> None:
        if self._owned_client:
            closer = getattr(self._client, "close", None)
            if callable(closer):
                closer()
            self._owned_client = False

    def _send(self, system: str, messages: list[Message], schema: type[BaseModel]) -> Reply:
        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system=system,
                messages=messages,
            )
        except self._unavailable as exc:
            raise LLMUnavailableError(f"Anthropic API call failed: {exc}") from exc
        except self._api_error as exc:
            if is_quota_error(str(exc)):
                raise LLMQuotaError(
                    f"Anthropic usage or quota limit reached ({exc}). "
                    "Check your Anthropic billing/usage limits, or switch to another provider "
                    "or local Ollama."
                ) from exc
            raise LLMError(f"Anthropic API call failed: {exc}") from exc
        text = "".join(block.text for block in response.content if block.type == "text")
        stop_reason = getattr(response, "stop_reason", None)
        if not text and stop_reason == "refusal":
            raise LLMError("Anthropic refused the request.")
        return Reply(
            text,
            truncated=stop_reason == "max_tokens",
            usage=_anthropic_usage(response),
            refused=stop_reason == "refusal",
            finish_reason=stop_reason if isinstance(stop_reason, str) else None,
        )
