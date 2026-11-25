# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Hero prop reference sheets: two-angle sheets of recurring objects used across shots."""

from __future__ import annotations

from pathlib import Path

from mvdirector.comfy.client import ComfyClient
from mvdirector.comfy.workflow import WorkflowTemplate
from mvdirector.errors import ConfigError
from mvdirector.generation.candidates import (
    atomic_save_png,
    load_rgb_candidate,
    render_candidate_batch,
)
from mvdirector.models import Manifest, PropDesign
from mvdirector.planning import prompts
from mvdirector.project import Project, require_safe_id


def _find_prop(manifest: Manifest, query: str) -> PropDesign:
    prop = manifest.visual_bible.find_prop(query)
    if prop is None:
        names = ", ".join(p.name for p in manifest.visual_bible.props) or "(none)"
        raise ConfigError(f"No prop matching '{query}'. Props in the bible: {names}")
    return prop


def generate_candidates(
    project: Project,
    manifest: Manifest,
    comfy: ComfyClient,
    template: WorkflowTemplate,
    prop_name: str | None,
    count: int,
) -> dict[str, list[Path]]:
    """Render ``count`` two-angle candidate sheets for one prop, or every prop."""
    bible, video = manifest.visual_bible, project.config.video
    if not bible.props:
        raise ConfigError(
            "The visual bible has no props. Add entries under visual_bible.props in manifest.json."
        )
    targets = [_find_prop(manifest, prop_name)] if prop_name else bible.props
    base_negative = prompts.merge_negative_prompt(bible.negative_prompt, bible.exclusions)
    negative = f"{base_negative}, {prompts.PROP_NEGATIVE}".strip(", ")
    results: dict[str, list[Path]] = {}
    for index, prop in enumerate(targets):
        safe_id = require_safe_id(prop.id, "prop id")
        base_params = {
            "positive_prompt": prompts.prop_prompt(manifest, prop.id),
            "negative_prompt": negative,
            "width": video.width,
            "height": video.height,
        }
        results[prop.name] = render_candidate_batch(
            comfy=comfy,
            template=template,
            base_params=base_params,
            base_seed=project.config.seed + 2000 + index * 100,
            count=count,
            folder=project.props_dir,
            prefix=safe_id,
            label=f"Prop '{prop.name}'",
        )
    return results


def lock_prop(project: Project, manifest: Manifest, image: Path, prop_name: str) -> Path:
    """Copy the chosen two-angle prop sheet into the project and record it on the prop."""
    loaded = load_rgb_candidate(image, "Prop")
    prop = _find_prop(manifest, prop_name)
    safe_id = require_safe_id(prop.id, "prop id")
    project.props_dir.mkdir(parents=True, exist_ok=True)
    locked = project.path(Path("assets") / "props" / f"{safe_id}.png")
    atomic_save_png(loaded, locked)
    prop.plate = project.rel(locked)
    return locked
