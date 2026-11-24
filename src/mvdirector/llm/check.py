# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""A quick test of the configured model: does it answer in JSON, and does it see images?

Used by ``mvd llm-check`` so a wrong key, model name, server address or text-only model shows up
in seconds, before a long run.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

import numpy as np
from pydantic import BaseModel

from mvdirector.errors import ImagesNotSupportedError, LLMError, LLMQuotaError, LLMUnavailableError
from mvdirector.llm.base import LLMClient, image_block, text_block

VisionStatus = Literal["ok", "unsupported", "wrong", "skipped", "error"]
ErrorKind = Literal["", "auth", "quota", "rate_limit", "not_found", "unreachable", "error"]


def classify_llm_error(exc: LLMError) -> ErrorKind:
    """Classify an ``LLMError`` for actionable setup/diagnostic reporting."""
    if isinstance(exc, LLMQuotaError):
        return "quota"
    text = str(exc).lower()
    if "rate limiting" in text or "rate limit" in text:
        return "rate_limit"
    if isinstance(exc, LLMUnavailableError):
        if any(
            word in text
            for word in (
                "credentials",
                "authentication",
                "unauthorized",
                "permission",
                "api key",
                "invalid_api_key",
                "401",
                "403",
            )
        ):
            return "auth"
        if any(word in text for word in ("could not find", "not found", "404", "does not exist")):
            return "not_found"
        return "unreachable"
    return "error"


class _JsonReply(BaseModel):
    ok: bool
    word: str


class _ColorReply(BaseModel):
    color: str


@dataclass
class LLMCheckResult:
    json_ok: bool = False
    json_detail: str = ""
    json_error_kind: ErrorKind = ""
    json_seconds: float = 0.0
    vision: VisionStatus = "skipped"
    vision_detail: str = ""
    vision_error_kind: ErrorKind = ""
    vision_seconds: float = 0.0


def make_test_image() -> np.ndarray:
    """A white image with one large blue square, which any model that can see should name."""
    image = np.full((128, 128, 3), 255, dtype=np.uint8)
    image[32:96, 32:96] = (20, 60, 220)
    return image


def ping_llm(client: LLMClient) -> None:
    """Raise ``LLMError`` unless the model answers one trivial JSON request.

    Run before a generation run, so a wrong key or a stopped server is reported before any GPU
    time is spent, not as a stream of skipped checks afterwards.
    """
    reply = client.complete_json(
        system="You are a test endpoint. Follow the instruction exactly.",
        content=[text_block("Reply with ok set to true and word set to 'ready'.")],
        schema=_JsonReply,
    )
    if not reply.ok:
        raise LLMError("The model answered, but not as instructed (ok was not true).")


def run_llm_check(client: LLMClient, check_vision: bool = True) -> LLMCheckResult:
    result = LLMCheckResult()
    started = time.perf_counter()
    try:
        reply = client.complete_json(
            system="You are a test endpoint. Follow the instruction exactly.",
            content=[text_block("Reply with ok set to true and word set to 'ready'.")],
            schema=_JsonReply,
        )
    except LLMError as exc:
        result.json_detail = str(exc)
        result.json_error_kind = classify_llm_error(exc)
    else:
        result.json_ok = reply.ok
        result.json_detail = "" if reply.ok else f"unexpected reply: ok={reply.ok}"
        result.json_error_kind = "" if reply.ok else "error"
    result.json_seconds = time.perf_counter() - started
    if not check_vision:
        return result
    if not result.json_ok:
        result.vision_detail = "skipped because the JSON check failed"
        return result

    started = time.perf_counter()
    try:
        answer = client.complete_json(
            system="You are a test endpoint. Follow the instruction exactly.",
            content=[
                image_block(make_test_image()),
                text_block(
                    "The image shows one large colored square on a white background. What color "
                    "is the square? Put one lowercase color word in the 'color' field."
                ),
            ],
            schema=_ColorReply,
        )
    except ImagesNotSupportedError as exc:
        result.vision, result.vision_detail = "unsupported", str(exc)
    except LLMError as exc:
        result.vision, result.vision_detail = "error", str(exc)
        result.vision_error_kind = classify_llm_error(exc)
    else:
        if "blue" in answer.color.lower():
            result.vision = "ok"
        else:
            result.vision = "wrong"
            result.vision_detail = (
                f"it accepted the image but called the blue square '{answer.color}', so it may "
                "be ignoring images"
            )
    result.vision_seconds = time.perf_counter() - started
    return result
