# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Pre-production brief generator: draft Suno-ready lyrics and style.md from an idea."""

from __future__ import annotations

from importlib import resources
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from mvdirector.errors import ConfigError
from mvdirector.llm import LLMClient, text_block
from mvdirector.planning import prompts
from mvdirector.project import Project


class BriefReply(BaseModel):
    model_config = ConfigDict(extra="ignore")
    lyrics: str
    suno_style: str
    visual_direction: str


def generate_brief(
    idea: str,
    *,
    llm: LLMClient | None,
    duration_seconds: int | None = None,
    vibe: str | None = None,
) -> BriefReply:
    """Ask the LLM for Suno-ready lyrics, a Suno style prompt, and visual direction."""
    cleaned = idea.strip()
    if not cleaned:
        raise ConfigError("Provide a non-empty --idea for the song and video concept.")
    if llm is None:
        raise ConfigError("`mvd brief` requires llm.enabled: true in project.yaml.")
    if duration_seconds is not None and not (5 <= duration_seconds <= 600):
        raise ConfigError("duration_seconds must be between 5 and 600 seconds.")

    details = [f"CONCEPT / STORY:\n{cleaned}"]
    if duration_seconds is not None:
        details.append(f"TARGET SONG DURATION: ~{duration_seconds} seconds")
    if vibe and vibe.strip():
        details.append(f"MUSICAL & VISUAL VIBE:\n{vibe.strip()}")

    reply = llm.complete_json(
        system=prompts.BRIEF_SYSTEM,
        content=[text_block("\n\n".join(details))],
        schema=BriefReply,
    )
    if not reply.lyrics.strip() or not reply.suno_style.strip():
        raise ConfigError("The model returned empty lyrics or Suno style prompt.")
    return reply


def _is_starter_file(path: Path, template_name: str) -> bool:
    if not path.is_file():
        return True
    try:
        starter = (
            resources.files("mvdirector") / "templates" / "inputs" / template_name
        ).read_text(encoding="utf-8")
        current = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError):
        return False
    return current.strip() == starter.strip()


def _input_target(project: Project, relative: str, label: str) -> tuple[str, Path]:
    cleaned = relative.strip()
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
            if target.is_file() and not _is_starter_file(target, template_name):
                raise ConfigError(
                    f"Existing {label} at {target} would be overwritten. "
                    "Pass --overwrite to replace."
                )
    return lyrics_rel, style_rel


def write_brief(
    project: Project, brief: BriefReply, *, overwrite: bool = False
) -> tuple[Path, Path]:
    """Write ``inputs/lyrics.txt`` and ``inputs/style.md`` atomically inside the project."""
    lyrics_rel, style_rel = ensure_writable(project, overwrite=overwrite)

    lyrics_text = brief.lyrics.strip() + "\n"
    style_parts = [f"Music style: {brief.suno_style.strip()}"]
    if brief.visual_direction.strip():
        style_parts.append(f"Visual direction: {brief.visual_direction.strip()}")
    style_text = "\n\n".join(style_parts) + "\n"
    return (
        project.write_text(lyrics_rel, lyrics_text),
        project.write_text(style_rel, style_text),
    )
