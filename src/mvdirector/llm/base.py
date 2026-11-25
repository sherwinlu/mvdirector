# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Pieces shared by every language-model backend.

All calls ask for one JSON object and validate it against a pydantic model. Requests are built
from neutral content blocks (``text_block`` and ``image_block``); each backend translates them
into its provider's wire format.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import math
import re
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TypeVar

import numpy as np
from PIL import Image
from pydantic import BaseModel, ValidationError

from mvdirector import media
from mvdirector.errors import LLMError, LLMUnavailableError, MediaToolError

log = logging.getLogger("mvdirector.llm")

T = TypeVar("T", bound=BaseModel)
ContentBlock = dict[str, Any]
Message = dict[str, Any]  # {"role": "user" | "assistant", "content": str | list[ContentBlock]}

MAX_IMAGE_SIDE = 1024
IMAGE_TOKEN_ESTIMATE = 1000  # rough cost of one image, for context warnings only
AUDIO_TOKEN_ESTIMATE = 4000  # rough cost of one attached song clip, for context warnings only
MAX_AUDIO_SECONDS = 360.0
MAX_AUDIO_BYTES = 6 * 1024 * 1024
_MAX_CORRECTION_CHARS = 1500


class LLMClient(Protocol):
    def complete_json(self, *, system: str, content: list[ContentBlock], schema: type[T]) -> T:
        """Send one request and return the reply parsed into ``schema``."""
        ...


def text_block(text: str) -> ContentBlock:
    return {"type": "text", "text": text}


def image_block(image: Path | np.ndarray) -> ContentBlock:
    """Encode an image file or RGB array as a JPEG block, downscaled to save tokens."""
    if isinstance(image, Path):
        with Image.open(image) as opened:
            pil = opened.convert("RGB")
    else:
        pil = Image.fromarray(image).convert("RGB")
    pil.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
    buffer = io.BytesIO()
    pil.save(buffer, format="JPEG", quality=85)
    data = base64.standard_b64encode(buffer.getvalue()).decode("ascii")
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data}}


def audio_block(
    audio: Path,
    *,
    max_seconds: float = MAX_AUDIO_SECONDS,
    max_bytes: int = MAX_AUDIO_BYTES,
) -> ContentBlock:
    """Encode an audio file as a compact mono MP3 block for audio-capable multimodal models."""
    if not audio.is_file():
        raise LLMError(f"Audio file not found for multimodal attachment: {audio}")
    if not math.isfinite(max_seconds) or max_seconds <= 0 or max_bytes <= 0:
        raise LLMError("max_seconds must be a positive finite number and max_bytes must be > 0.")
    with tempfile.TemporaryDirectory(prefix="mvd-audio-") as tmpdir:
        out_mp3 = Path(tmpdir) / "clip.mp3"
        media.run(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-protocol_whitelist",
                "file",
                "-i",
                str(audio.resolve()),
                "-t",
                f"{max_seconds:.1f}",
                "-vn",
                "-ac",
                "1",
                "-ar",
                "24000",
                "-c:a",
                "libmp3lame",
                "-b:a",
                "48k",
                str(out_mp3),
            ],
            timeout=60.0,
        )
        raw = out_mp3.read_bytes()
    if not raw:
        raise LLMError(f"Transcoded audio clip was empty for {audio}")
    if len(raw) > max_bytes:
        raise LLMError(
            f"Transcoded audio clip ({len(raw)} bytes) exceeds limit ({max_bytes} bytes)"
        )
    data = base64.standard_b64encode(raw).decode("ascii")
    return {"type": "audio", "source": {"type": "base64", "media_type": "audio/mpeg", "data": data}}


def maybe_audio_block(llm: LLMClient, audio_path: Path | None) -> ContentBlock | None:
    """Return an ``audio_block`` when ``llm`` supports native audio, or None on fallback."""
    if audio_path is None or not getattr(llm, "supports_audio", False):
        return None
    if not audio_path.is_file():
        return None
    try:
        return audio_block(audio_path)
    except (LLMError, MediaToolError, OSError) as exc:
        log.warning(
            "Could not attach audio %s to LLM request (%s); using text cues.", audio_path, exc
        )
        return None


def complete_json_with_optional_audio(
    llm: LLMClient,
    *,
    system: str,
    text: str,
    schema: type[T],
    audio_path: Path | None = None,
) -> T:
    """Send a JSON completion with optional audio, falling back to text on rejection."""
    text_only = [text_block(text)]
    attached = maybe_audio_block(llm, audio_path)
    if attached is None:
        return llm.complete_json(system=system, content=text_only, schema=schema)
    try:
        return llm.complete_json(system=system, content=[*text_only, attached], schema=schema)
    except LLMUnavailableError:
        raise
    except LLMError as exc:
        log.warning(
            "Model rejected request with attached audio (%s); retrying with text cues only.", exc
        )
        return llm.complete_json(system=system, content=text_only, schema=schema)


_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _reject_json_constant(token: str) -> Any:
    raise ValueError(f"Non-standard JSON constant not allowed: {token!r}")


def _strip_think(text: str) -> str:
    cleaned = _THINK_BLOCK.sub("", text)
    unfinished = cleaned.lower().find("<think>")  # reasoning that never ended: nothing follows
    return cleaned[:unfinished] if unfinished != -1 else cleaned


_MAX_BARE_LIST_PROBES = 8


def _try_bare_list_for_schema(text: str, schema: type[T]) -> T | None:
    """Validate a bare JSON array ``[...]`` when ``schema`` is a single-field list container.

    Advances past short bracketed prose like ``Batch [1]:`` without hijacking a valid JSON object
    that follows it, and returns ``None`` whenever no bare array validates against ``schema``.
    """
    if len(schema.model_fields) != 1:
        return None
    cleaned = _strip_think(text)
    obj_start = cleaned.find("{")
    field_name = next(iter(schema.model_fields))
    pos = cleaned.find("[")
    probes = 0
    while pos != -1 and (obj_start == -1 or pos < obj_start) and probes < _MAX_BARE_LIST_PROBES:
        probes += 1
        next_search = pos + 1
        for strict in (True, False):
            decoder = json.JSONDecoder(strict=strict, parse_constant=_reject_json_constant)
            try:
                value, end = decoder.raw_decode(cleaned, pos)
            except ValueError:
                continue
            except RecursionError:
                raise LLMError("The model reply nested JSON too deeply to parse") from None
            next_search = end
            if isinstance(value, list):
                try:
                    return schema.model_validate({field_name: value})
                except ValidationError:
                    break
        pos = cleaned.find("[", next_search)
    return None


def extract_json(text: str) -> str:
    """Pull the JSON object out of a reply that may include fences, prose or reasoning.

    Reasoning models (often run locally) put their thinking in ``<think>`` tags before the answer,
    and any model may add prose after it. The first complete object is returned. If none decodes,
    the outermost braces are returned instead, so validation reports the real syntax error and
    the correction request can quote it.
    """
    cleaned = _strip_think(text)
    first_start = cleaned.find("{")
    if first_start == -1:
        raise LLMError("The model reply contained no JSON object")
    outer_looks_like_json = bool(re.match(r'^\{\s*"', cleaned[first_start:]))
    for strict in (True, False):
        decoder = json.JSONDecoder(strict=strict, parse_constant=_reject_json_constant)
        pos = first_start
        while pos != -1:
            try:
                value, end = decoder.raw_decode(cleaned, pos)
            except ValueError:
                value = end = None
            except RecursionError:
                # A reply nested thousands of levels deep exhausts the stack rather than
                # raising ValueError; treat it as unusable JSON, not as a crash.
                raise LLMError("The model reply nested JSON too deeply to parse") from None
            if isinstance(value, dict) and end is not None:
                return cleaned[pos:end]
            if outer_looks_like_json:
                break
            pos = cleaned.find("{", pos + 1)
    last = cleaned.rfind("}")
    if last <= first_start:
        raise LLMError("The model reply contained no JSON object")
    return cleaned[first_start : last + 1]


def blocks_of(content: str | list[ContentBlock]) -> list[ContentBlock]:
    return [text_block(content)] if isinstance(content, str) else content


def has_images(messages: list[Message]) -> bool:
    return any(block.get("type") == "image" for m in messages for block in blocks_of(m["content"]))


def estimate_tokens(system: str, messages: list[Message]) -> int:
    """A rough token count (four characters per token, fixed costs for images and audio)."""
    chars, images, audios = len(system), 0, 0
    for message in messages:
        for block in blocks_of(message["content"]):
            block_type = block.get("type")
            if block_type == "image":
                images += 1
            elif block_type == "audio":
                audios += 1
            else:
                chars += len(block.get("text", ""))
    return chars // 4 + images * IMAGE_TOKEN_ESTIMATE + audios * AUDIO_TOKEN_ESTIMATE


def schema_prompt(system: str, schema: type[BaseModel]) -> str:
    """The system prompt plus the instruction to reply with JSON matching ``schema``."""
    return (
        f"{system}\n\nReply with a single JSON object and nothing else. "
        f"It must match this JSON Schema:\n{json.dumps(schema.model_json_schema())}"
    )


@dataclass(frozen=True)
class Reply:
    text: str
    truncated: bool = False  # the model stopped because it hit the token limit


class ChatBackend(ABC):
    """The request loop common to every provider: ask for JSON, validate, correct once."""

    supports_audio: bool = False

    def __init__(self, *, model: str, max_tokens: int, context_tokens: int | None = None) -> None:
        self._model = model
        self._max_tokens = max_tokens
        self._context_tokens = context_tokens
        self._warned_context = False

    def close(self) -> None:
        http = getattr(self, "_http", None)
        closer = getattr(http, "close", None)
        if callable(closer):
            closer()

    def __enter__(self) -> ChatBackend:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def complete_json(self, *, system: str, content: list[ContentBlock], schema: type[T]) -> T:
        system_prompt = schema_prompt(system, schema)
        messages: list[Message] = [{"role": "user", "content": content}]
        self._warn_if_large(system_prompt, messages)
        last_error: Exception | None = None
        for _ in range(2):
            reply = self._send(system_prompt, messages, schema)
            log.debug("Model reply (%d characters): %s", len(reply.text), reply.text[:500])
            try:
                if (bare := _try_bare_list_for_schema(reply.text, schema)) is not None:
                    return bare
                raw_json = extract_json(reply.text)
                try:
                    return schema.model_validate_json(raw_json)
                except ValidationError as first_exc:
                    # Open-weights models (e.g. Qwen2.5-Omni) sometimes emit literal newlines
                    # inside multiline JSON strings like lyrics; retry with strict=False.
                    try:
                        parsed = json.loads(
                            raw_json, strict=False, parse_constant=_reject_json_constant
                        )
                    except (ValueError, RecursionError):
                        raise first_exc from None
                    return schema.model_validate(parsed)
            except (ValidationError, LLMError) as exc:
                last_error = exc
                if reply.truncated:
                    # A correction cannot help: the reply needs more room, not different words.
                    raise LLMError(
                        f"The model's reply was cut off at max_tokens ({self._max_tokens}) before "
                        "the JSON was complete. Raise llm.max_tokens in project.yaml, or lower "
                        "llm.shots_per_request."
                    ) from exc
                log.warning("Model reply failed validation; asking for a correction")
                detail = str(exc)[:_MAX_CORRECTION_CHARS]
                assistant_reply = reply.text if reply.text.strip() else "(empty response)"
                messages += [
                    {"role": "assistant", "content": assistant_reply},
                    {
                        "role": "user",
                        "content": f"That did not validate: {detail}\nSend corrected JSON.",
                    },
                ]
        raise LLMError(
            f"Model reply could not be parsed as {schema.__name__}: {last_error} "
            f"The reply began: {reply.text[:200]!r}. Run with -v to see more."
        )

    def _warn_if_large(self, system: str, messages: list[Message]) -> None:
        if self._context_tokens is None or self._warned_context:
            return
        estimate = estimate_tokens(system, messages)
        if estimate > 0.8 * self._context_tokens:
            self._warned_context = True
            log.warning(
                "This request is about %d tokens and the context window is %d. A server may "
                "silently drop the start of an oversized prompt. Raise llm.context_tokens or "
                "lower llm.shots_per_request.",
                estimate,
                self._context_tokens,
            )

    @abstractmethod
    def _send(self, system: str, messages: list[Message], schema: type[BaseModel]) -> Reply:
        """Send one request and return the model's text."""
