# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Shared helpers for generating and locking preflight candidate images (sheets and plates)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from PIL import Image

from mvdirector.comfy.client import ComfyClient
from mvdirector.comfy.workflow import WorkflowTemplate
from mvdirector.errors import ConfigError
from mvdirector.project import require_safe_id

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
    """Open and convert a candidate image to RGB, raising ConfigError if missing."""
    if not image.is_file():
        raise ConfigError(f"{label} image not found: {image}")
    with Image.open(image) as opened:
        return opened.convert("RGB")
