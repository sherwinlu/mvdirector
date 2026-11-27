# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Benchmark matrix configuration, default model arms, and token pricing."""

from __future__ import annotations

import math
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, ValidationInfo, field_validator

from mvdirector.config import (
    CONFIG_FILENAME,
    LLMConfig,
    LLMProvider,
    ProjectConfig,
    is_allowed_api_key_env,
    mask_unset_allowed_env_refs,
)
from mvdirector.dotenv import read_project_dotenv
from mvdirector.errors import ConfigError
from mvdirector.project import Project
from mvdirector.store import atomic_write

_SAFE_ARM_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,31}\Z")
_SNAPSHOT_SUFFIX_RE = re.compile(
    r"^-(?:latest|\d{4}(?:-?\d{2}(?:-?\d{2})?)?|v\d+(?:\.\d+)*|preview(?:-[\w.-]+)?|exp(?:-[\w.-]+)?)\Z"
)


def _reject_bool_number(value: object, info: ValidationInfo) -> object:
    if isinstance(value, bool):
        raise ValueError(f"{info.field_name} must be a number, not a boolean.")
    return value


def validate_budget_usd(max_budget_usd: float | None) -> float | None:
    """Validate an optional USD spend cap, rejecting booleans, negatives, and NaN/Inf."""
    if max_budget_usd is None:
        return None
    if isinstance(max_budget_usd, bool) or not isinstance(max_budget_usd, (int, float)):
        raise ConfigError(f"max_budget_usd must be a non-negative number, got {max_budget_usd!r}.")
    val = float(max_budget_usd)
    if not math.isfinite(val) or val < 0.0:
        raise ConfigError(
            f"max_budget_usd must be finite and non-negative, got {max_budget_usd!r}."
        )
    return val


class ModelPricing(BaseModel):
    """Estimated USD per 1M tokens for cost comparison across benchmark arms."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    input_per_million: float = Field(ge=0.0)
    output_per_million: float = Field(ge=0.0)
    thoughts_per_million: float | None = Field(default=None, ge=0.0)
    cached_input_per_million: float | None = Field(default=None, ge=0.0)

    @field_validator(
        "input_per_million",
        "output_per_million",
        "thoughts_per_million",
        "cached_input_per_million",
        mode="before",
    )
    @classmethod
    def _no_bool(cls, value: object, info: ValidationInfo) -> object:
        return _reject_bool_number(value, info)


# Default USD / 1M token pricing estimates; override per arm in bench.yaml as rates change.
DEFAULT_PRICING: dict[str, ModelPricing] = {
    # Google Gemini (3.8 Flash baseline + Gemini 4 family)
    "gemini-3.8-flash": ModelPricing(input_per_million=0.30, output_per_million=2.50),
    "gemini-2.5-flash": ModelPricing(input_per_million=0.30, output_per_million=2.50),
    "gemini-4-neon": ModelPricing(input_per_million=1.25, output_per_million=10.00),
    "gemini-4-carbon": ModelPricing(input_per_million=0.50, output_per_million=3.00),
    "gemini-4-hydrogen": ModelPricing(input_per_million=0.15, output_per_million=0.60),
    "gemini-4-argon": ModelPricing(input_per_million=2.50, output_per_million=15.00),
    # Anthropic Claude (official base token rates)
    "claude-fable-5-2": ModelPricing(input_per_million=8.00, output_per_million=40.00),
    "claude-mythos-5-2": ModelPricing(input_per_million=8.00, output_per_million=40.00),
    "claude-fable-5-1": ModelPricing(input_per_million=10.00, output_per_million=50.00),
    "claude-mythos-5-1": ModelPricing(input_per_million=10.00, output_per_million=50.00),
    "claude-fable-5": ModelPricing(input_per_million=10.00, output_per_million=50.00),
    "claude-mythos-5": ModelPricing(input_per_million=10.00, output_per_million=50.00),
    "claude-opus-5-5": ModelPricing(input_per_million=4.00, output_per_million=20.00),
    "claude-sonnet-5-5": ModelPricing(input_per_million=2.00, output_per_million=10.00),
    "claude-haiku-5-5": ModelPricing(input_per_million=0.10, output_per_million=0.50),
    "claude-opus-5": ModelPricing(input_per_million=5.00, output_per_million=25.00),
    "claude-sonnet-5": ModelPricing(input_per_million=2.00, output_per_million=10.00),
    "claude-opus-4-8": ModelPricing(input_per_million=5.00, output_per_million=25.00),
    "claude-opus-4-7": ModelPricing(input_per_million=5.00, output_per_million=25.00),
    "claude-opus-4-6": ModelPricing(input_per_million=5.00, output_per_million=25.00),
    "claude-opus-4-5": ModelPricing(input_per_million=5.00, output_per_million=25.00),
    "claude-opus-4-1": ModelPricing(input_per_million=15.00, output_per_million=75.00),
    "claude-opus-4": ModelPricing(input_per_million=15.00, output_per_million=75.00),
    "claude-sonnet-4-6": ModelPricing(input_per_million=3.00, output_per_million=15.00),
    "claude-sonnet-4-5": ModelPricing(input_per_million=3.00, output_per_million=15.00),
    "claude-sonnet-4": ModelPricing(input_per_million=3.00, output_per_million=15.00),
    "claude-haiku-4-5": ModelPricing(input_per_million=1.00, output_per_million=5.00),
    # OpenAI GPT-6.x, GPT-5.x, and GPT-4.x (estimates; override in bench.yaml)
    "gpt-6.2-astra": ModelPricing(input_per_million=5.00, output_per_million=25.00),
    "gpt-6.2-sol": ModelPricing(input_per_million=2.00, output_per_million=10.00),
    "gpt-6.1-terra": ModelPricing(input_per_million=1.00, output_per_million=4.00),
    "gpt-6.1-sol": ModelPricing(input_per_million=1.75, output_per_million=8.75),
    "gpt-6-astra": ModelPricing(input_per_million=5.00, output_per_million=25.00),
    "gpt-6-sol": ModelPricing(input_per_million=1.75, output_per_million=8.75),
    "gpt-6-luna": ModelPricing(input_per_million=0.25, output_per_million=1.50),
    "gpt-5.6-sol": ModelPricing(input_per_million=1.75, output_per_million=10.00),
    "gpt-5.6-terra": ModelPricing(input_per_million=0.80, output_per_million=3.20),
    "gpt-5.6-luna": ModelPricing(input_per_million=0.25, output_per_million=1.50),
    "gpt-5.5": ModelPricing(input_per_million=1.75, output_per_million=14.00),
    "gpt-5.4": ModelPricing(input_per_million=1.25, output_per_million=10.00),
    "gpt-5.4-mini": ModelPricing(input_per_million=0.25, output_per_million=2.00),
    "gpt-5.2": ModelPricing(input_per_million=1.75, output_per_million=14.00),
    "gpt-5.1": ModelPricing(input_per_million=1.25, output_per_million=10.00),
    "gpt-5": ModelPricing(input_per_million=1.25, output_per_million=10.00),
    "gpt-4.1": ModelPricing(input_per_million=2.00, output_per_million=8.00),
    "gpt-4o": ModelPricing(input_per_million=2.50, output_per_million=10.00),
}


def lookup_pricing(model: str) -> ModelPricing:
    """Return exact or snapshot-suffix matched default pricing, or zeroed fallback."""
    cleaned = model.strip().lower().removeprefix("models/")
    if cleaned in DEFAULT_PRICING:
        return DEFAULT_PRICING[cleaned]
    for prefix in sorted(DEFAULT_PRICING, key=len, reverse=True):
        if cleaned.startswith(prefix):
            suffix = cleaned[len(prefix) :]
            if _SNAPSHOT_SUFFIX_RE.match(suffix):
                return DEFAULT_PRICING[prefix]
    return ModelPricing(input_per_million=0.0, output_per_million=0.0)


class ModelArm(BaseModel):
    """One model arm in the A/B/C benchmark matrix."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    id: str = Field(description="Short arm identifier (e.g. A, B, B2, C, D, E, F).")
    label: str = Field(description="Human-readable name shown in reports after unblinding.")
    provider: LLMProvider
    model: str
    attach_audio: bool = Field(
        default=False,
        description="Attach song MP3 when supported by provider (True for A/B, False for B2).",
    )
    optional: bool = Field(
        default=False,
        description="Control arms skipped by default unless --include-controls or --arm is passed.",
    )
    max_tokens: int = Field(default=32768, ge=256)
    shots_per_request: int = Field(default=10, ge=1)
    timeout_seconds: float = Field(default=600.0, gt=0.0)
    json_mode: Literal["auto", "off", "object", "schema"] = "auto"
    base_url: str | None = None
    api_key_env: str | None = None
    stream: bool = False
    pricing: ModelPricing | None = None
    notes: str = ""

    @field_validator("max_tokens", "shots_per_request", "timeout_seconds", mode="before")
    @classmethod
    def _no_bool_numbers(cls, value: object, info: ValidationInfo) -> object:
        return _reject_bool_number(value, info)

    @field_validator("id")
    @classmethod
    def _validate_arm_id(cls, value: str) -> str:
        cleaned = value.strip()
        if not _SAFE_ARM_ID_RE.match(cleaned):
            raise ValueError(
                f"Invalid arm id {value!r}: use letters, digits, '.', '_' or '-' (1-32 chars)."
            )
        return cleaned

    @field_validator("model", "label")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("must not be empty")
        return cleaned

    @field_validator("api_key_env")
    @classmethod
    def _validate_api_key_env(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        if not cleaned:
            return None
        if not is_allowed_api_key_env(cleaned):
            raise ValueError(
                f"arm api_key_env {value!r} is not allowed. Use a provider key name "
                "(ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY, GOOGLE_API_KEY), "
                "an MVD_* name, or a name listed in MVD_ALLOWED_ENV_VARS."
            )
        return cleaned

    def effective_pricing(self) -> ModelPricing:
        return self.pricing if self.pricing is not None else lookup_pricing(self.model)

    def to_llm_config(self, *, environ: Mapping[str, str] | None = None) -> LLMConfig:
        payload = {
            "enabled": True,
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
            "api_key_env": self.api_key_env,
            "max_tokens": self.max_tokens,
            "shots_per_request": self.shots_per_request,
            "timeout_seconds": self.timeout_seconds,
            "json_mode": self.json_mode,
            "stream": self.stream,
        }
        ctx = {"environ": environ} if environ is not None else None
        try:
            return LLMConfig.model_validate(payload, context=ctx)
        except ValidationError as exc:
            raise ConfigError(
                f"Invalid LLM settings for benchmark arm {self.id!r}:\n{exc}"
            ) from exc


class GPUCostConfig(BaseModel):
    """GPU render costs used to compute Net Effective Cost / 90-shot Video in Track 2."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    still_usd: float = Field(
        default=0.02,
        ge=0.0,
        description="Estimated GPU cost of one rejected still retry (FP penalty).",
    )
    video_usd: float = Field(
        default=0.35,
        ge=0.0,
        description="Estimated GPU cost of one wasted video render from an escaped bad still (FN).",
    )
    shots_per_video: int = Field(
        default=90,
        ge=1,
        description="Reference shot count per full music video for net effective cost scaling.",
    )

    @field_validator("still_usd", "video_usd", "shots_per_video", mode="before")
    @classmethod
    def _no_bool(cls, value: object, info: ValidationInfo) -> object:
        return _reject_bool_number(value, info)


def default_arms() -> list[ModelArm]:
    """Default 6-model + 5 optional control/specialist arms benchmark matrix."""
    return [
        ModelArm(
            id="A",
            label="Gemini 3.8 Flash (audio)",
            provider="gemini",
            model="gemini-3.8-flash",
            attach_audio=True,
            notes="Current baseline (Sugar and Ash setup)",
        ),
        ModelArm(
            id="B",
            label="Gemini 4 Neon (audio)",
            provider="gemini",
            model="gemini-4-neon",
            attach_audio=True,
            notes="Flagship Gemini 4 upgrade with native MP3 listening",
        ),
        ModelArm(
            id="B2",
            label="Gemini 4 Neon (text-only)",
            provider="gemini",
            model="gemini-4-neon",
            attach_audio=False,
            optional=True,
            notes="Optional control isolating native audio vs. reasoning lift",
        ),
        ModelArm(
            id="C",
            label="Claude Sonnet 5.5",
            provider="anthropic",
            model="claude-sonnet-5-5",
            attach_audio=False,
            max_tokens=32768,
            notes="Mid-tier Anthropic benchmark (text-only planning, vision QC)",
        ),
        ModelArm(
            id="D",
            label="Claude Opus 5.5",
            provider="anthropic",
            model="claude-opus-5-5",
            attach_audio=False,
            max_tokens=32768,
            notes="Top-tier Anthropic benchmark",
        ),
        ModelArm(
            id="E",
            label="GPT-6.2 Sol",
            provider="openai",
            model="gpt-6.2-sol",
            attach_audio=False,
            max_tokens=32768,
            notes="Go-to everyday OpenAI benchmark (also consider gpt-6.1-terra for vision QC)",
        ),
        ModelArm(
            id="F",
            label="GPT-6.2 Astra",
            provider="openai",
            model="gpt-6.2-astra",
            attach_audio=False,
            max_tokens=32768,
            notes="Top-tier flagship OpenAI benchmark",
        ),
        ModelArm(
            id="G",
            label="Gemini 4 Carbon (audio)",
            provider="gemini",
            model="gemini-4-carbon",
            attach_audio=True,
            optional=True,
            notes="Optional mid-tier Gemini 4 with native MP3 listening (cost/quality sweet spot)",
        ),
        ModelArm(
            id="H",
            label="Claude Fable 5.2",
            provider="anthropic",
            model="claude-fable-5-2",
            attach_audio=False,
            optional=True,
            max_tokens=32768,
            notes="Optional Track 1 creative ceiling ($8/$40 per MTok; skip for bulk QC)",
        ),
        ModelArm(
            id="I",
            label="Claude Haiku 5.5",
            provider="anthropic",
            model="claude-haiku-5-5",
            attach_audio=False,
            optional=True,
            max_tokens=16384,
            notes="Optional ultra-low-cost Track 2 Vision QC floor ($0.10/$0.50 per MTok)",
        ),
        ModelArm(
            id="J",
            label="GPT-6.1 Terra",
            provider="openai",
            model="gpt-6.1-terra",
            attach_audio=False,
            optional=True,
            max_tokens=16384,
            notes="Optional fast visual-reasoning dark horse for Track 2 Vision QC",
        ),
    ]


class BenchmarkConfig(BaseModel):
    """Top-level configuration for ``mvd bench`` runs."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    arms: list[ModelArm] = Field(default_factory=default_arms, min_length=1)
    gpu_costs: GPUCostConfig = Field(default_factory=GPUCostConfig)

    @field_validator("arms")
    @classmethod
    def _unique_arm_ids(cls, value: list[ModelArm]) -> list[ModelArm]:
        seen: set[str] = set()
        for arm in value:
            key = arm.id.casefold()
            if key in seen:
                raise ValueError(f"Duplicate arm id {arm.id!r} in benchmark config.")
            seen.add(key)
        return value

    def select_arms(
        self,
        arm_ids: Sequence[str] | None = None,
        *,
        include_controls: bool = False,
    ) -> list[ModelArm]:
        """Return matching arms in config order, skipping optional controls unless requested."""
        if arm_ids:
            by_key = {arm.id.casefold(): arm for arm in self.arms}
            selected: list[ModelArm] = []
            seen: set[str] = set()
            for raw in arm_ids:
                key = raw.strip().casefold()
                if key not in by_key:
                    available = ", ".join(a.id for a in self.arms)
                    raise ConfigError(f"Unknown benchmark arm {raw!r}. Available: {available}")
                if key not in seen:
                    selected.append(by_key[key])
                    seen.add(key)
            return selected
        return [arm for arm in self.arms if include_controls or not arm.optional]


def load_benchmark_config(path: Path | None) -> BenchmarkConfig:
    """Load ``bench.yaml`` if provided, else return the built-in default benchmark matrix."""
    if path is None:
        return BenchmarkConfig()
    if not path.is_file():
        raise ConfigError(f"Benchmark config not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, UnicodeDecodeError, yaml.YAMLError, RecursionError) as exc:
        raise ConfigError(f"Could not read benchmark config {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"Benchmark config {path} must contain a YAML mapping.")
    try:
        return BenchmarkConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"Invalid benchmark config in {path}:\n{exc}") from exc


def open_bench_project(root: Path) -> Project:
    """Open a project for LLM-only benchmarking without requiring ComfyUI env vars."""
    root_path = Path(root)
    cfg_path = root_path / CONFIG_FILENAME
    if not cfg_path.is_file():
        raise ConfigError(f"Config file not found: {cfg_path}")
    cwd_dotenv = read_project_dotenv(Path.cwd())
    proj_dotenv = read_project_dotenv(root_path)
    effective_env = {**cwd_dotenv, **proj_dotenv, **os.environ}
    try:
        raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    except (OSError, UnicodeDecodeError, yaml.YAMLError, RecursionError) as exc:
        raise ConfigError(f"Could not read {cfg_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"Config file {cfg_path} must contain a YAML mapping.")
    masked = mask_unset_allowed_env_refs(raw, environ=effective_env)
    try:
        config = ProjectConfig.model_validate(masked, context={"environ": effective_env})
    except ValidationError as exc:
        raise ConfigError(f"Invalid configuration in {cfg_path}:\n{exc}") from exc
    return Project(root_path, config, dotenv={**cwd_dotenv, **proj_dotenv})


def resolve_bench_environ(
    project_environ: Mapping[str, str],
    explicit: Mapping[str, str] | None = None,
) -> Mapping[str, str]:
    """Merge cwd ``.env`` under ``project_environ`` (shell env > project .env > cwd .env)."""
    if explicit is not None:
        return explicit
    cwd_dotenv = read_project_dotenv(Path.cwd())
    if not cwd_dotenv:
        return project_environ
    return {**cwd_dotenv, **project_environ}


def write_default_benchmark_config(path: Path, *, overwrite: bool = False) -> Path:
    """Write an editable starter ``bench.yaml`` containing the default benchmark matrix."""
    if path.exists() and not overwrite:
        raise ConfigError(f"{path} already exists. Pass --overwrite to replace it.")
    cfg = BenchmarkConfig()
    payload = cfg.model_dump(mode="json")
    for arm_dict, arm in zip(payload["arms"], cfg.arms):
        arm_dict["pricing"] = arm.effective_pricing().model_dump(mode="json")
    header = (
        "# mvdirector multi-model benchmark matrix.\n"
        "# Edit model IDs, pricing (USD per 1M tokens), or GPU retry costs before running.\n"
        "# Gemini 4 endpoints: gemini-4-carbon, gemini-4-neon, gemini-4-hydrogen, gemini-4-argon.\n"
        "# Claude 5.5/5.2 aliases: claude-sonnet-5-5, claude-opus-5-5, claude-fable-5-2, "
        "claude-haiku-5-5.\n"
        "# OpenAI GPT-6 endpoints: gpt-6.2-astra, gpt-6.2-sol, gpt-6.1-terra, gpt-6-luna.\n"
        "# Optional control arms (optional: true) run with --include-controls or --arm B2.\n\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(path, header + yaml.safe_dump(payload, sort_keys=False))
    return path
