# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""A small HTTP helper for the backends that talk JSON over HTTP.

Connection problems and rate limits are retried with a pause. Everything else becomes a clear
``LLMError`` that says what to check, because a bare status code does not help someone who is
pointing the app at a server of their own.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable
from typing import Any
from urllib.parse import urlsplit

import httpx

from mvdirector.errors import ConfigError, ImagesNotSupportedError, LLMError, LLMUnavailableError

log = logging.getLogger("mvdirector.llm")

_RETRYABLE = {429, 500, 502, 503, 504}
IMAGE_WORDS = ("image", "vision", "multimodal", "multi-modal")
_KEY_WORDS = ("api key", "api_key", "apikey", "authentication", "unauthorized")
_MAX_RETRY_AFTER = 30.0


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
                if response.status_code in _RETRYABLE and not final and not described:
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
    if status == 429:
        raise LLMError(f"{provider} is rate limiting requests ({message}). Try again shortly.")
    # Servers report "this model cannot read images" with different statuses, some as a 500.
    if had_images and any(w in message.lower() for w in IMAGE_WORDS):
        raise ImagesNotSupportedError(
            f"{provider} rejected the image input ({message}). This model may not accept images."
        )
    raise LLMError(f"{provider} returned HTTP {status}: {message}")
