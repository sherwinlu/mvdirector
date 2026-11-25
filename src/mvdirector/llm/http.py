# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""A small HTTP helper for the backends that talk JSON over HTTP.

Connection problems and rate limits are retried with a pause. Everything else becomes a clear
``LLMError`` that says what to check, because a bare status code does not help someone who is
pointing the app at a server of their own.
"""

from __future__ import annotations

import codecs
import logging
import math
import time
from collections.abc import Callable
from typing import Any
from urllib.parse import urlsplit

import httpx

from mvdirector.errors import (
    ConfigError,
    ImagesNotSupportedError,
    LLMError,
    LLMQuotaError,
    LLMUnavailableError,
)

log = logging.getLogger("mvdirector.llm")

_RETRYABLE = {429, 500, 502, 503, 504}
IMAGE_WORDS = ("image", "vision", "multimodal", "multi-modal")
_KEY_WORDS = ("api key", "api_key", "apikey", "authentication", "unauthorized")
QUOTA_WORDS = (
    "insufficient_quota",
    "exceeded your current quota",
    "quota exceeded",
    "credit balance is too low",
    "billing hard limit",
    "out of credits",
    "usage limit",
    "spending limit",
    "monthly limit",
    "plan and billing",
)
_MAX_RETRY_AFTER = 30.0
_MAX_SSE_BYTES = 16 * 1024 * 1024  # 16 MiB safety cap per streamed LLM response


def is_quota_error(text: str) -> bool:
    """True when an error body indicates account billing, credit, or usage-quota exhaustion."""
    lowered = text.lower()
    return any(word in lowered for word in QUOTA_WORDS)


def _host_port(parts: Any) -> str:
    host = parts.hostname or ""
    if ":" in host:
        host = f"[{host}]"
    try:
        port = parts.port
    except ValueError:
        port = None
    return f"{host}:{port}" if port is not None else host


def origin(url: str) -> str:
    """Return ``scheme://host[:port]`` with any embedded credentials stripped."""
    parts = urlsplit(url)
    host_port = _host_port(parts)
    return f"{parts.scheme}://{host_port}" if host_port else f"{parts.scheme}://"


def redact_url(url: str) -> str:
    """Return ``url`` without embedded userinfo, query string, or fragment."""
    parts = urlsplit(url)
    host_port = _host_port(parts)
    if not parts.scheme or not host_port:
        return origin(url)
    return f"{parts.scheme}://{host_port}{parts.path}"


def validate_http_base_url(url: str, *, field_name: str) -> str:
    """Validate that ``url`` is a clean ``http://`` or ``https://`` base URL with no query/frag."""
    if any(ord(ch) < 32 or ord(ch) == 127 or ch.isspace() for ch in url):
        raise ConfigError(f"{field_name} must not contain whitespace or control characters.")
    try:
        parts = urlsplit(url)
        _ = parts.port
    except ValueError as exc:
        raise ConfigError(f"Invalid {field_name}: {exc}") from exc
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ConfigError(
            f"{field_name} must be an http:// or https:// URL with a valid host "
            f"(got {redact_url(url)!r})."
        )
    if parts.query or parts.fragment:
        raise ConfigError(f"{field_name} must not include query parameters or fragments.")
    return url


class HttpJSON:
    def __init__(
        self,
        *,
        headers: dict[str, str],
        timeout: float,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] | None = None,
        retries: int = 3,
    ) -> None:
        self._client = httpx.Client(headers=headers, timeout=timeout, transport=transport)
        self._timeout = timeout
        self._sleep = sleep or time.sleep  # looked up here so tests can patch time.sleep
        self._retries = retries

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> HttpJSON:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def post(
        self,
        url: str,
        payload: dict[str, Any],
        *,
        unreachable_hint: str = "",
        no_retry_words: tuple[str, ...] = (),
    ) -> httpx.Response:
        """POST JSON, retrying connection problems, rate limits and server errors.

        A server error whose text contains one of ``no_retry_words`` is returned at once: it is a
        description of the request (for example "this model cannot read images"), and sending the
        same request again cannot change the answer.
        """
        last = ""
        for attempt in range(self._retries):
            final = attempt == self._retries - 1
            try:
                response = self._client.post(url, json=payload)
            except httpx.InvalidURL as exc:
                raise ConfigError(f"Invalid LLM URL: {exc}") from exc
            except httpx.ConnectTimeout:
                last = "the connection timed out"
            except httpx.TimeoutException as exc:
                raise LLMUnavailableError(
                    f"The request to {origin(url)} timed out after {self._timeout:.0f} s. "
                    "Raise llm.timeout_seconds for a slow model."
                ) from exc
            except httpx.TransportError as exc:
                last = str(exc) or type(exc).__name__
            else:
                described = _mentions(response, no_retry_words)
                quota_exhausted = response.status_code == 429 and is_quota_error(response.text)
                if (
                    response.status_code in _RETRYABLE
                    and not final
                    and not described
                    and not quota_exhausted
                ):
                    delay = retry_delay(response, attempt)
                    log.warning(
                        "%s returned %d; retrying in %.0f s",
                        origin(url),
                        response.status_code,
                        delay,
                    )
                    self._sleep(delay)
                    continue
                return response
            if not final:
                self._sleep(retry_delay(None, attempt))
        raise LLMUnavailableError(
            f"Could not reach {origin(url)} ({last}). {unreachable_hint}".strip()
        )

    def post_sse(
        self,
        url: str,
        payload: dict[str, Any],
        *,
        unreachable_hint: str = "",
        no_retry_words: tuple[str, ...] = (),
    ) -> tuple[httpx.Response, list[str] | None]:
        """POST JSON and collect SSE ``data:`` payloads while keeping reverse proxies alive.

        On non-2xx responses the full error body is read onto the returned ``httpx.Response`` so
        callers can inspect it with ``check_response`` just like :meth:`post`. If the server
        ignores ``stream: true`` and returns HTTP 200 ``application/json``, the body is read onto
        ``response`` and ``None`` is returned for ``events`` so the caller can parse standard JSON.
        """
        last = ""
        for attempt in range(self._retries):
            final = attempt == self._retries - 1
            try:
                deadline = time.monotonic() + self._timeout
                with self._client.stream("POST", url, json=payload) as response:
                    if response.status_code != 200:
                        response.read()
                        described = _mentions(response, no_retry_words)
                        quota_exhausted = response.status_code == 429 and is_quota_error(
                            response.text
                        )
                        if (
                            response.status_code in _RETRYABLE
                            and not final
                            and not described
                            and not quota_exhausted
                        ):
                            delay = retry_delay(response, attempt)
                            log.warning(
                                "%s returned %d; retrying in %.0f s",
                                origin(url),
                                response.status_code,
                                delay,
                            )
                            self._sleep(delay)
                            continue
                        return response, []
                    content_type = (
                        response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
                    )
                    if content_type == "application/json":
                        raw_json = bytearray()
                        for chunk in response.iter_bytes():
                            if time.monotonic() > deadline:
                                raise LLMUnavailableError(
                                    f"The request to {origin(url)} timed out after "
                                    f"{self._timeout:.0f} s. Raise llm.timeout_seconds for a "
                                    "slow model."
                                )
                            if len(raw_json) + len(chunk) > _MAX_SSE_BYTES:
                                raise LLMError(
                                    f"Response from {origin(url)} exceeded "
                                    f"{_MAX_SSE_BYTES} byte limit."
                                )
                            raw_json.extend(chunk)
                        clean_headers = [
                            (k, v)
                            for k, v in response.headers.multi_items()
                            if k.lower() not in ("content-encoding", "content-length")
                        ]
                        buffered = httpx.Response(
                            status_code=response.status_code,
                            headers=clean_headers,
                            content=bytes(raw_json),
                            request=response.request,
                        )
                        return buffered, None
                    events: list[str] = []
                    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
                    buf = ""
                    total_bytes = 0
                    done = False
                    for chunk in response.iter_bytes():
                        if time.monotonic() > deadline:
                            raise LLMUnavailableError(
                                f"The request to {origin(url)} timed out after "
                                f"{self._timeout:.0f} s. Raise llm.timeout_seconds for a slow "
                                "model."
                            )
                        total_bytes += len(chunk)
                        if total_bytes > _MAX_SSE_BYTES:
                            raise LLMError(
                                f"Streamed response from {origin(url)} exceeded "
                                f"{_MAX_SSE_BYTES} byte limit."
                            )
                        buf += decoder.decode(chunk)
                        while "\n" in buf:
                            raw_line, buf = buf.split("\n", 1)
                            line = raw_line.rstrip("\r")
                            if not line or not line.startswith("data:"):
                                continue
                            data = line[5:].strip()
                            if data == "[DONE]":
                                done = True
                                break
                            if data:
                                events.append(data)
                        if done:
                            break
                    if not done:
                        tail = (buf + decoder.decode(b"", final=True)).rstrip("\r")
                        if tail.startswith("data:"):
                            data = tail[5:].strip()
                            if data and data != "[DONE]":
                                events.append(data)
                    return response, events
            except httpx.InvalidURL as exc:
                raise ConfigError(f"Invalid LLM URL: {exc}") from exc
            except httpx.ConnectTimeout:
                last = "the connection timed out"
            except httpx.TimeoutException as exc:
                raise LLMUnavailableError(
                    f"The request to {origin(url)} timed out after {self._timeout:.0f} s. "
                    "Raise llm.timeout_seconds for a slow model."
                ) from exc
            except httpx.TransportError as exc:
                last = str(exc) or type(exc).__name__
            if not final:
                self._sleep(retry_delay(None, attempt))
        raise LLMUnavailableError(
            f"Could not reach {origin(url)} ({last}). {unreachable_hint}".strip()
        )


def _mentions(response: httpx.Response, words: tuple[str, ...]) -> bool:
    if not words:
        return False
    lowered = response.text.lower()
    return any(word in lowered for word in words)


def retry_delay(response: httpx.Response | None, attempt: int) -> float:
    if response is not None:
        header = response.headers.get("retry-after", "")
        try:
            parsed = float(header)
        except ValueError:
            parsed = math.nan
        if math.isfinite(parsed) and parsed >= 0.0:
            return min(parsed, _MAX_RETRY_AFTER)
    return 2.0 * (attempt + 1)


_retry_delay = retry_delay


def error_message(response: httpx.Response) -> str:
    """The server's own explanation, from whichever JSON shape it used."""
    try:
        data = response.json()
    except ValueError:
        return response.text[:300].strip() or f"HTTP {response.status_code}"
    if isinstance(data, dict):
        error = data.get("error")
        if isinstance(error, dict):
            return str(error.get("message") or error)[:500]
        if isinstance(error, str):
            return error[:500]
        if "message" in data:
            return str(data["message"])[:500]
    return str(data)[:300]


def json_body(response: httpx.Response, provider: str) -> dict[str, Any]:
    """The reply as a JSON object, or a clear error for a proxy page or other non-JSON reply."""
    try:
        data = response.json()
    except ValueError as exc:
        raise LLMError(
            f"{provider} returned a reply that is not JSON: {response.text[:200]!r}"
        ) from exc
    if not isinstance(data, dict):
        raise LLMError(f"{provider} returned an unexpected reply: {str(data)[:200]}")
    return data


def check_response(
    response: httpx.Response,
    *,
    provider: str,
    key_hint: str,
    had_images: bool,
    not_found_hint: str = "Check llm.model and llm.base_url.",
) -> None:
    """Raise a descriptive error for a non-success response."""
    if response.is_success:
        return
    status, message = response.status_code, error_message(response)
    # Gemini reports a bad key as a 400, so the wording is checked as well as the status.
    if status in (401, 403) or (status == 400 and any(w in message.lower() for w in _KEY_WORDS)):
        raise LLMUnavailableError(f"{provider} rejected the credentials ({message}). {key_hint}")
    if status == 404:
        raise LLMUnavailableError(
            f"{provider} could not find the model or endpoint ({message}). {not_found_hint}"
        )
    if status == 402 or (status in (400, 429) and is_quota_error(message)):
        raise LLMQuotaError(
            f"{provider} usage or quota limit reached ({message}). "
            "Check your billing/usage limits, or switch to another provider or local Ollama."
        )
    if status == 429:
        raise LLMError(f"{provider} is rate limiting requests ({message}). Try again shortly.")
    # Servers report "this model cannot read images" with different statuses, some as a 500.
    if had_images and any(w in message.lower() for w in IMAGE_WORDS):
        raise ImagesNotSupportedError(
            f"{provider} rejected the image input ({message}). This model may not accept images."
        )
    raise LLMError(f"{provider} returned HTTP {status}: {message}")
