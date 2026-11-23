# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Shared helpers for generating and locking preflight candidate images (sheets and plates)."""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from mvdirector import media
from mvdirector.comfy.client import ComfyClient
from mvdirector.comfy.workflow import WorkflowTemplate
from mvdirector.errors import ConfigError, MediaToolError
from mvdirector.project import Project, require_safe_id
from mvdirector.store import path_is_file

log = logging.getLogger(__name__)


def render_candidate_batch(
    comfy: ComfyClient,
    template: WorkflowTemplate,
    base_params: dict[str, Any],
    base_seed: int,
    count: int,
    folder: Path,
    prefix: str,
    label: str,
    extension: str = ".png",
) -> list[Path]:
    """Render ``count`` candidate files into ``folder`` with deterministic sequential seeds."""
    if isinstance(count, bool) or count < 1:
        raise ConfigError(f"{label} candidate count must be >= 1, got {count}.")
    safe_prefix = require_safe_id(prefix, "candidate prefix")
    resolved_folder = folder.resolve()
    paths: list[Path] = []
    for index in range(count):
        destination = (
            resolved_folder / f"{safe_prefix}_candidate_{index + 1:02d}{extension}"
        ).resolve()
        if not destination.is_relative_to(resolved_folder):
            raise ConfigError(f"Candidate path '{destination}' escapes {resolved_folder}")
        graph = template.render({**base_params, "seed": base_seed + index})
        paths.append(comfy.run(graph, template.bindings.output, destination))
        log.info("%s candidate %d saved to %s", label, index + 1, destination)
    return paths


def load_rgb_candidate(image: Path, label: str) -> Image.Image:
    """Open and convert a candidate image to RGB, raising ConfigError if missing or invalid."""
    if not path_is_file(image):
        raise ConfigError(f"{label} image not found: {image}")
    try:
        with Image.open(image) as opened:
            return opened.convert("RGB")
    except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
        raise ConfigError(f"{label} image could not be opened: {image} ({exc})") from exc


def atomic_save_png(image: Image.Image, dst: Path) -> Path:
    """Save ``image`` as PNG atomically via a temporary sibling file and ``os.replace``."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{dst.stem}.", suffix=".png", dir=dst.parent)
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        image.save(tmp_path, format="PNG")
        os.replace(tmp_path, dst)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    return dst


def ensure_upload_audio_track(project: Project, video_path: Path) -> Path:
    """Return ``video_path`` or a cached copy with a silent audio track for ComfyUI loaders.

    ComfyUI video loaders such as ``VHS_LoadVideoPath`` fail on files without an audio stream.
    The original file is never rewritten; failures fall back to uploading ``video_path`` unchanged.
    """
    try:
        return media.with_audio_track(video_path, project.path("assets/cache/audio"))
    except MediaToolError as exc:
        log.warning(
            "Could not add a silent audio track to %s (%s); uploading it unchanged",
            video_path.name,
            exc,
        )
        return video_path
