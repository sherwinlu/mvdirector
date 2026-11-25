# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Pre-production brief generator: draft Suno-ready lyrics and style.md from an idea."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from mvdirector import tuning as tuning_mod
from mvdirector.errors import ConfigError
from mvdirector.llm import LLMClient, text_block
from mvdirector.models import accept_do_not_show_alias, normalize_exclusions
from mvdirector.planning import prompts
from mvdirector.project import Project


class BriefReply(BaseModel):
    model_config = ConfigDict(extra="ignore")
    lyrics: str
    suno_style: str
    visual_direction: str
    exclusions: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _accept_do_not_show_alias(cls, data: Any) -> Any:
        return accept_do_not_show_alias(data)

    @field_validator("exclusions", mode="before")
    @classmethod
    def _normalize_exclusions(cls, value: Any) -> list[str]:
        return normalize_exclusions(value)


def generate_brief(
    idea: str,
    *,
    llm: LLMClient | None,
    duration_seconds: int | None = None,
    vibe: str | None = None,
    exclusions: Sequence[str] | None = None,
    do_not_show: Sequence[str] | None = None,
) -> BriefReply:
    """Ask the LLM for Suno-ready lyrics, a Suno style prompt, and visual direction."""
    cleaned = idea.strip()
    if not cleaned:
        raise ConfigError("Provide a non-empty --idea for the song and video concept.")
    if llm is None:
        raise ConfigError("`mvd brief` requires llm.enabled: true in project.yaml.")
    if duration_seconds is not None and not (5 <= duration_seconds <= 600):
        raise ConfigError("duration_seconds must be between 5 and 600 seconds.")

    merged_exclusions = normalize_exclusions([*(exclusions or ()), *(do_not_show or ())])
    details = [f"CONCEPT / STORY:\n{cleaned}"]
    if duration_seconds is not None:
        details.append(f"TARGET SONG DURATION: ~{duration_seconds} seconds")
    if vibe and vibe.strip():
        details.append(f"MUSICAL & VISUAL VIBE:\n{vibe.strip()}")
    if merged_exclusions:
        details.append(
            "GLOBAL EXCLUSIONS (NEVER SHOW):\n"
            + "\n".join(f"- {item}" for item in merged_exclusions)
        )

    reply = llm.complete_json(
        system=prompts.BRIEF_SYSTEM,
        content=[text_block("\n\n".join(details))],
        schema=BriefReply,
    )
    if not reply.lyrics.strip() or not reply.suno_style.strip():
        raise ConfigError("The model returned empty lyrics or Suno style prompt.")
    reply.exclusions = normalize_exclusions([*reply.exclusions, *merged_exclusions])
    return reply


def _input_target(project: Project, relative: str | None, label: str) -> tuple[str, Path]:
    cleaned = (relative or "").strip()
    if not cleaned:
        raise ConfigError(f"inputs.{label} is empty in project.yaml.")
    target = project.path(cleaned)
    if target == project.root or target.is_dir():
        raise ConfigError(f"inputs.{label} must name a file inside {project.root}: {relative!r}")
    return cleaned, target


def ensure_writable(project: Project, *, overwrite: bool = False) -> tuple[str, str]:
    """Validate target input paths and refuse overwriting non-starter files unless allowed."""
    inputs = project.config.inputs
    lyrics_rel, lyrics_target = _input_target(project, inputs.lyrics, "lyrics")
    style_rel, style_target = _input_target(project, inputs.style, "style")

    if not overwrite:
        for target, template_name, label in (
            (lyrics_target, "lyrics.txt", "lyrics"),
            (style_target, "style.md", "style brief"),
        ):
            if target.is_file() and not tuning_mod.is_starter_input(target, template_name):
                raise ConfigError(
                    f"Existing {label} at {target} would be overwritten. "
                    "Pass --overwrite to replace."
                )
    return lyrics_rel, style_rel


def write_brief(
    project: Project,
    brief: BriefReply,
    *,
    overwrite: bool = False,
    exclusions: Sequence[str] | None = None,
    do_not_show: Sequence[str] | None = None,
) -> tuple[Path, Path]:
    """Write ``inputs/lyrics.txt`` and ``inputs/style.md`` atomically inside the project."""
    lyrics_rel, style_rel = ensure_writable(project, overwrite=overwrite)

    lyrics_text = brief.lyrics.strip() + "\n"
    style_parts = [f"Music style: {brief.suno_style.strip()}"]
    if brief.visual_direction.strip():
        style_parts.append(f"Visual direction: {brief.visual_direction.strip()}")
    merged_exclusions = normalize_exclusions(
        [*brief.exclusions, *(exclusions or ()), *(do_not_show or ())]
    )
    if merged_exclusions:
        style_parts.append(f"Exclusions: {', '.join(merged_exclusions)}")
    style_text = "\n\n".join(style_parts) + "\n"
    return (
        project.write_text(lyrics_rel, lyrics_text),
        project.write_text(style_rel, style_text),
    )
