# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Build the configured backend, with one clear error for each thing that can be misconfigured."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from urllib.parse import urlsplit

import httpx

from mvdirector.config import LLMConfig
from mvdirector.errors import ConfigError
from mvdirector.llm.anthropic_backend import AnthropicChat
from mvdirector.llm.base import LLMClient
from mvdirector.llm.gemini_backend import GeminiChat
from mvdirector.llm.http import HttpJSON, redact_url
from mvdirector.llm.ollama_backend import OllamaChat
from mvdirector.llm.openai_backend import OpenAIChat

log = logging.getLogger("mvdirector.llm")


@dataclass(frozen=True)
class ProviderInfo:
    key_env: str | None  # default environment variable for the API key
    key_required: bool
    base_url: str | None  # default server address
    json_mode: str  # what json_mode "auto" means for this provider


PROVIDERS: dict[str, ProviderInfo] = {
    "anthropic": ProviderInfo("ANTHROPIC_API_KEY", True, None, "off"),
    "openai": ProviderInfo("OPENAI_API_KEY", True, "https://api.openai.com/v1", "object"),
    "gemini": ProviderInfo(
        "GEMINI_API_KEY", True, "https://generativelanguage.googleapis.com/v1beta", "object"
    ),
    "ollama": ProviderInfo(None, False, "http://localhost:11434", "object"),
    # Servers that speak the OpenAI protocol differ in what they accept, so ask in the prompt.
    "openai_compatible": ProviderInfo(None, False, None, "off"),
}
_GEMINI_FALLBACK_KEY_ENV = "GOOGLE_API_KEY"


def _api_key(
    config: LLMConfig, info: ProviderInfo, environ: Mapping[str, str]
) -> tuple[str | None, str]:
    """The key and a hint telling the user where it should come from."""
    name = config.api_key_env or info.key_env
    value = environ.get(name, "") if name else ""
    if not value and config.provider == "gemini" and not config.api_key_env:
        value = environ.get(_GEMINI_FALLBACK_KEY_ENV, "")
    hint = f"Check the {name} environment variable." if name else ""
    wanted = info.key_required or config.api_key_env is not None
    if wanted and not value:
        raise ConfigError(
            f"llm.provider '{config.provider}' needs an API key, but the {name} environment "
            f"variable is not set. Export it, or set llm.api_key_env to the variable you use."
        )
    return (value or None), hint


def _validate_base_url(url: str) -> str:
    if any(ord(ch) < 32 or ord(ch) == 127 or ch.isspace() for ch in url):
        raise ConfigError("llm.base_url must not contain whitespace or control characters.")
    try:
        parts = urlsplit(url)
        _ = parts.port
    except ValueError as exc:
        raise ConfigError(f"Invalid llm.base_url: {exc}") from exc
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ConfigError(
            "llm.base_url must be an http:// or https:// URL with a valid host "
            f"(got {redact_url(url)!r})."
        )
    if parts.query or parts.fragment:
        raise ConfigError("llm.base_url must not include query parameters or fragments.")
    return url


def _normalize_base_url(
    config: LLMConfig, info: ProviderInfo, environ: Mapping[str, str]
) -> str | None:
    raw = (config.base_url or environ.get("MVD_LLM_URL", "").strip() or info.base_url or "").strip()
    if not raw:
        return None
    url = raw.rstrip("/")
    if config.provider == "openai_compatible" and url.endswith("/chat/completions"):
        url = url.removesuffix("/chat/completions")
    if config.provider == "ollama":
        if url.endswith("/v1"):
            raise ConfigError(
                "llm.base_url ends in /v1, which is Ollama's OpenAI-compatible endpoint. For "
                "provider 'ollama' use the plain address (http://localhost:11434), or switch "
                "to provider 'openai_compatible'."
            )
        url = url.removesuffix("/api")
    if not url:
        raise ConfigError("llm.base_url must be an http:// or https:// URL with a valid host.")
    return _validate_base_url(url)


def build_llm(
    config: LLMConfig,
    *,
    environ: Mapping[str, str] | None = None,
    transport: httpx.BaseTransport | None = None,
    sleep: Callable[[float], None] | None = None,
) -> LLMClient | None:
    """Return a client for the configured provider, or None when the LLM is disabled."""
    if not config.enabled:
        return None
    environ = os.environ if environ is None else environ
    info = PROVIDERS[config.provider]
    key, key_hint = _api_key(config, info, environ)
    base_url = _normalize_base_url(config, info, environ)
    json_mode = info.json_mode if config.json_mode == "auto" else config.json_mode
    log.debug(
        "LLM: provider=%s model=%s base_url=%s json_mode=%s",
        config.provider,
        config.model,
        redact_url(base_url) if base_url else None,
        json_mode,
    )

    if config.provider == "anthropic":
        if transport is not None:
            raise ConfigError(
                "Custom httpx transport is not supported for the anthropic provider; "
                "pass a client to AnthropicChat directly."
            )
        return AnthropicChat(
            model=config.model,
            max_tokens=config.max_tokens,
            api_key=key,
            base_url=base_url,
            timeout=config.timeout_seconds,
            context_tokens=config.context_tokens,
        )

    if base_url is None:
        raise ConfigError(f"llm.base_url is required for provider '{config.provider}'.")
    headers = {"Content-Type": "application/json"}
    if config.provider == "gemini" and key:
        headers["x-goog-api-key"] = key
    elif key:
        headers["Authorization"] = f"Bearer {key}"
    http = HttpJSON(
        headers=headers, timeout=config.timeout_seconds, transport=transport, sleep=sleep
    )

    if config.provider == "gemini":
        return GeminiChat(
            model=config.model,
            max_tokens=config.max_tokens,
            base_url=base_url,
            http=http,
            json_mode=json_mode,
            key_hint=key_hint,
            context_tokens=config.context_tokens,
        )
    if config.provider == "ollama":
        return OllamaChat(
            model=config.model,
            max_tokens=config.max_tokens,
            base_url=base_url,
            http=http,
            json_mode=json_mode,
            context_tokens=config.context_tokens,
        )
    return OpenAIChat(
        provider="OpenAI" if config.provider == "openai" else "The server",
        model=config.model,
        max_tokens=config.max_tokens,
        base_url=base_url,
        http=http,
        json_mode=json_mode,
        # OpenAI's newer models require max_completion_tokens; other servers know max_tokens.
        token_param="max_completion_tokens" if config.provider == "openai" else "max_tokens",
        key_hint=key_hint,
        context_tokens=config.context_tokens,
    )
