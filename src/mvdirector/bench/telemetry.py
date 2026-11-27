# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Instrumented LLM wrapper capturing latency, tokens, cost, retries, and refusals."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TypeVar

from pydantic import BaseModel, ConfigDict

from mvdirector.bench.config import ModelArm, ModelPricing
from mvdirector.llm.base import (
    CallTelemetry,
    ContentBlock,
    LLMClient,
    TokenUsage,
    is_refusal_error,
)

T = TypeVar("T", bound=BaseModel)


class CallRecord(BaseModel):
    """Serializable per-request telemetry entry."""

    model_config = ConfigDict(allow_inf_nan=False)

    stage: str = ""
    schema_name: str
    model: str
    latency_seconds: float
    attempts: int
    schema_retries: int
    first_pass_ok: bool
    succeeded: bool
    truncated: bool
    refused: bool
    has_audio: bool
    has_images: bool
    input_tokens: int
    output_tokens: int
    thoughts_tokens: int
    cached_tokens: int
    estimated_usd: float
    error: str | None = None


class TelemetrySummary(BaseModel):
    """Aggregated LLM telemetry across a planning run or Vision QC sweep."""

    model_config = ConfigDict(allow_inf_nan=False)

    calls: int = 0
    succeeded: int = 0
    failed: int = 0
    first_pass_ok: int = 0
    first_pass_rate: float = 0.0
    schema_retries: int = 0
    truncations: int = 0
    refusals: int = 0
    audio_calls: int = 0
    image_calls: int = 0
    latency_seconds: float = 0.0
    mean_latency_seconds: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    thoughts_tokens: int = 0
    cached_tokens: int = 0
    estimated_usd: float = 0.0


def estimate_cost_usd(
    usage: TokenUsage,
    pricing: ModelPricing,
    *,
    provider: str,
) -> float:
    """Estimate USD cost for ``usage`` accounting for provider-specific token reporting.

    - Gemini reports ``thoughtsTokenCount`` separately from ``candidatesTokenCount`` and
      includes ``cachedContentTokenCount`` inside ``promptTokenCount``.
    - OpenAI includes ``reasoning_tokens`` inside ``completion_tokens`` and ``cached_tokens``
      inside ``prompt_tokens``.
    - Anthropic reports ``cache_read_input_tokens`` separately from ``input_tokens``.
    """
    cached_rate = (
        pricing.cached_input_per_million
        if pricing.cached_input_per_million is not None
        else pricing.input_per_million * 0.25
    )
    thoughts_rate = (
        pricing.thoughts_per_million
        if pricing.thoughts_per_million is not None
        else pricing.output_per_million
    )
    if provider == "anthropic":
        non_cached_input = max(0, usage.input_tokens)
        cached_input = max(0, usage.cached_tokens)
        billable_output = max(0, usage.output_tokens)
        extra_thoughts = max(0, usage.thoughts_tokens)
    elif provider in ("openai", "openai_compatible"):
        cached_input = min(max(0, usage.cached_tokens), max(0, usage.input_tokens))
        non_cached_input = max(0, usage.input_tokens - cached_input)
        # OpenAI completion_tokens already includes reasoning_tokens.
        billable_output = max(0, usage.output_tokens)
        extra_thoughts = 0
    else:
        # Gemini / Ollama: promptTokenCount includes cachedContentTokenCount;
        # thoughtsTokenCount is billed alongside candidatesTokenCount.
        cached_input = min(max(0, usage.cached_tokens), max(0, usage.input_tokens))
        non_cached_input = max(0, usage.input_tokens - cached_input)
        billable_output = max(0, usage.output_tokens)
        extra_thoughts = max(0, usage.thoughts_tokens)

    usd = (
        (non_cached_input * pricing.input_per_million)
        + (cached_input * cached_rate)
        + (billable_output * pricing.output_per_million)
        + (extra_thoughts * thoughts_rate)
    ) / 1_000_000.0
    return round(usd, 6)


def summarize_records(records: list[CallRecord]) -> TelemetrySummary:
    """Roll up a list of ``CallRecord`` entries into a ``TelemetrySummary``."""
    total = len(records)
    if total == 0:
        return TelemetrySummary()
    succeeded = sum(1 for r in records if r.succeeded)
    first_pass = sum(1 for r in records if r.first_pass_ok)
    total_latency = sum(r.latency_seconds for r in records)
    return TelemetrySummary(
        calls=total,
        succeeded=succeeded,
        failed=total - succeeded,
        first_pass_ok=first_pass,
        first_pass_rate=round(first_pass / total, 4),
        schema_retries=sum(r.schema_retries for r in records),
        truncations=sum(1 for r in records if r.truncated),
        refusals=sum(1 for r in records if r.refused),
        audio_calls=sum(1 for r in records if r.has_audio),
        image_calls=sum(1 for r in records if r.has_images),
        latency_seconds=round(total_latency, 3),
        mean_latency_seconds=round(total_latency / total, 3),
        input_tokens=sum(r.input_tokens for r in records),
        output_tokens=sum(r.output_tokens for r in records),
        thoughts_tokens=sum(r.thoughts_tokens for r in records),
        cached_tokens=sum(r.cached_tokens for r in records),
        estimated_usd=round(sum(r.estimated_usd for r in records), 6),
    )


class InstrumentedLLM:
    """Wraps any real ``LLMClient`` to record telemetry and gate audio attachment per arm."""

    def __init__(self, inner: LLMClient, arm: ModelArm) -> None:
        self._inner = inner
        self._arm = arm
        self._pricing = arm.effective_pricing()
        self._lock = threading.Lock()
        self._records: list[CallRecord] = []
        self._stage_local = threading.local()
        self._hook_installed = False
        # Respect arm.attach_audio so control Arm B2 stays strictly text-only on Gemini.
        self.supports_audio: bool = bool(
            arm.attach_audio and getattr(inner, "supports_audio", False)
        )
        setter = getattr(inner, "set_telemetry_hook", None)
        if callable(setter):
            setter(self._on_backend_call)
            self._hook_installed = True

    @property
    def records(self) -> list[CallRecord]:
        with self._lock:
            return list(self._records)

    def summary(self) -> TelemetrySummary:
        return summarize_records(self.records)

    def reset(self) -> None:
        with self._lock:
            self._records.clear()

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        prev = getattr(self._stage_local, "stage", "")
        self._stage_local.stage = name
        try:
            yield
        finally:
            self._stage_local.stage = prev

    def _current_stage(self) -> str:
        return str(getattr(self._stage_local, "stage", ""))

    def _on_backend_call(self, event: CallTelemetry) -> None:
        usd = estimate_cost_usd(event.usage, self._pricing, provider=self._arm.provider)
        record = CallRecord(
            stage=self._current_stage(),
            schema_name=event.schema,
            model=event.model,
            latency_seconds=event.latency_seconds,
            attempts=event.attempts,
            schema_retries=event.schema_retries,
            first_pass_ok=event.first_pass_ok,
            succeeded=event.succeeded,
            truncated=event.truncated,
            refused=event.refused,
            has_audio=event.has_audio,
            has_images=event.has_images,
            input_tokens=event.usage.input_tokens,
            output_tokens=event.usage.output_tokens,
            thoughts_tokens=event.usage.thoughts_tokens,
            cached_tokens=event.usage.cached_tokens,
            estimated_usd=usd,
            error=event.error,
        )
        with self._lock:
            self._records.append(record)

    def complete_json(self, *, system: str, content: list[ContentBlock], schema: type[T]) -> T:
        if self._hook_installed:
            return self._inner.complete_json(system=system, content=content, schema=schema)
        # Fallback for custom/mock LLMClient implementations that do not subclass ChatBackend.
        started = time.monotonic()
        has_aud = any(b.get("type") == "audio" for b in content)
        has_img = any(b.get("type") == "image" for b in content)
        succeeded = False
        refused = False
        truncated = False
        err_text: str | None = None
        try:
            result = self._inner.complete_json(system=system, content=content, schema=schema)
            succeeded = True
            return result
        except BaseException as exc:
            err_text = str(exc)
            refused = is_refusal_error(exc)
            truncated = "cut off at max_tokens" in err_text.lower()
            raise
        finally:
            record = CallRecord(
                stage=self._current_stage(),
                schema_name=schema.__name__,
                model=self._arm.model,
                latency_seconds=round(max(0.0, time.monotonic() - started), 4),
                attempts=1,
                schema_retries=0,
                first_pass_ok=succeeded,
                succeeded=succeeded,
                truncated=truncated,
                refused=refused,
                has_audio=has_aud,
                has_images=has_img,
                input_tokens=0,
                output_tokens=0,
                thoughts_tokens=0,
                cached_tokens=0,
                estimated_usd=0.0,
                error=err_text,
            )
            with self._lock:
                self._records.append(record)

    def close(self) -> None:
        setter = getattr(self._inner, "set_telemetry_hook", None)
        if callable(setter):
            setter(None)
        closer = getattr(self._inner, "close", None)
        if callable(closer):
            closer()

    def __enter__(self) -> InstrumentedLLM:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
