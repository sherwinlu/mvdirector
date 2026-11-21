# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Set plates: one approved image of each empty location, used as a second reference for stills.

A text description of a set is read fresh every time, so the same room comes out differently in
each shot. A plate is generated once, approved by a person, and then supplied to the still
workflow as an image, so the layout, lighting and color grade stay fixed.
"""

from __future__ import annotations

import hashlib
import logging
import re
import unicodedata
from pathlib import Path

from PIL import UnidentifiedImageError

from mvdirector.comfy.client import ComfyClient
from mvdirector.comfy.workflow import WorkflowTemplate
from mvdirector.errors import ConfigError
from mvdirector.generation.candidates import load_rgb_candidate, render_candidate_batch
from mvdirector.models import Manifest, SetDesign, ascii_fold
from mvdirector.planning import prompts
from mvdirector.project import Project

log = logging.getLogger(__name__)


def plate_slug(name: str) -> str:
    """A collision-safe ASCII filename stem for a set name across any script."""
    cleaned = unicodedata.normalize("NFKC", name).strip()
    stem = re.sub(r"[^a-z0-9]+", "_", ascii_fold(cleaned).lower()).strip("_") or "set"
    if all(ord(ch) < 128 for ch in cleaned):
        return stem
    digest = hashlib.sha256(cleaned.casefold().encode("utf-8")).hexdigest()[:6]
    return f"{stem}_{digest}"


def _find_set(manifest: Manifest, name: str) -> SetDesign:
    scene = manifest.visual_bible.find_set(name)
    if scene is None:
        names = ", ".join(s.name for s in manifest.visual_bible.settings) or "(none)"
        raise ConfigError(f"No set named '{name}'. Sets in the bible: {names}")
    return scene


def _resolve_ref_workflow(
    templates: dict[str, WorkflowTemplate], label: str
) -> tuple[WorkflowTemplate, str]:
    edit_template = templates.get("still")
    if edit_template is None:
        raise ConfigError(f"A 'still' workflow is required to {label}.")
    ref_param = (
        "reference_image"
        if edit_template.accepts("reference_image")
        else ("set_image" if edit_template.accepts("set_image") else None)
    )
    if ref_param is None:
        raise ConfigError(
            f"Workflow '{edit_template.name}' must bind 'reference_image' or 'set_image' "
            f"to {label}."
        )
    return edit_template, ref_param


def generate_candidates(
    project: Project,
    manifest: Manifest,
    comfy: ComfyClient,
    template: WorkflowTemplate | dict[str, WorkflowTemplate],
    set_name: str | None,
    count: int,
    *,
    style_ref: Path | None = None,
    embed_set: str | None = None,
    from_set: str | None = None,
) -> dict[str, list[Path]]:
    """Render ``count`` candidate plates for one set, or for every set, keyed by set name.

    Pass ``style_ref`` to seed the visual style from an existing image, ``embed_set`` (with
    ``set_name``) to compose another set's locked plate inside a screen/diorama in the new set,
    or ``from_set`` (or ``scene.parent_set``) to edit that parent set's locked plate.
    """
    bible, video = manifest.visual_bible, project.config.video
    if not bible.settings:
        raise ConfigError("The visual bible has no sets. Run `mvd plan` with the LLM enabled.")
    if sum(opt is not None for opt in (style_ref, embed_set, from_set)) > 1:
        raise ConfigError("Pass either --style-ref, --embed-set, or --from-set, not multiple.")
    if embed_set is not None:
        if not embed_set.strip():
            raise ConfigError("Provide a non-empty set name for --embed-set.")
        if not set_name:
            raise ConfigError("Pass --set when using --embed-set so only the target set uses it.")
    if from_set is not None:
        if not from_set.strip():
            raise ConfigError("Provide a non-empty set name for --from-set.")
        if not set_name:
            raise ConfigError("Pass --set when using --from-set so only the target set is edited.")
    templates = template if isinstance(template, dict) else {"t2i": template, "still": template}
    t2i = templates.get("t2i") or templates.get("still")
    if t2i is None:
        raise ConfigError("A 't2i' workflow is required to generate set plates.")
    scenes = [_find_set(manifest, set_name)] if set_name else bible.settings
    negative = f"{bible.negative_prompt}, {prompts.PLATE_NEGATIVE}".strip(", ")

    uploaded_style_ref: str | None = None
    style_template: WorkflowTemplate | None = None
    style_param: str | None = None
    if style_ref is not None:
        try:
            load_rgb_candidate(style_ref, "Style reference")
        except (OSError, ValueError, UnidentifiedImageError) as exc:
            raise ConfigError(f"Style reference image could not be opened: {style_ref}") from exc
        style_template, style_param = _resolve_ref_workflow(templates, "use --style-ref")
        uploaded_style_ref = comfy.upload(style_ref)

    results: dict[str, list[Path]] = {}
    for scene_index, scene in enumerate(scenes):
        if embed_set is not None:
            if embed_set.strip().lower() == scene.name.strip().lower():
                raise ConfigError(f"Set '{scene.name}' cannot embed itself.")
            embedded = _find_set(manifest, embed_set)
            if not embedded.plate:
                raise ConfigError(
                    f"Lock a plate for '{embedded.name}' before embedding it in '{scene.name}': "
                    f'`mvd plates lock <image> --set "{embedded.name}"`.'
                )
            embedded_path = project.path(embedded.plate)
            if not embedded_path.is_file():
                raise ConfigError(f"Plate file for '{embedded.name}' is missing: {embedded.plate}")
            chosen_template, ref_param = _resolve_ref_workflow(
                templates, f"embed '{embedded.name}' into '{scene.name}'"
            )
            base_params = {
                "positive_prompt": prompts.plate_embed_prompt(manifest, scene.name, embedded.name),
                "negative_prompt": negative,
                "width": video.width,
                "height": video.height,
                ref_param: comfy.upload(embedded_path),
            }
        elif style_template is not None and style_param is not None and uploaded_style_ref:
            chosen_template = style_template
            base_params = {
                "positive_prompt": prompts.plate_style_prompt(manifest, scene.name),
                "negative_prompt": negative,
                "width": video.width,
                "height": video.height,
                style_param: uploaded_style_ref,
            }
        elif source_name := (from_set or scene.parent_set):
            if source_name.strip().lower() == scene.name.strip().lower():
                raise ConfigError(f"Set '{scene.name}' cannot derive from itself.")
            parent = _find_set(manifest, source_name)
            if not parent.plate:
                raise ConfigError(
                    f"Lock a plate for '{parent.name}' before deriving '{scene.name}': "
                    f'`mvd plates lock <image> --set "{parent.name}"`.'
                )
            parent_plate_path = project.path(parent.plate)
            if not parent_plate_path.is_file():
                raise ConfigError(f"Plate file for '{parent.name}' is missing: {parent.plate}")
            chosen_template, ref_param = _resolve_ref_workflow(
                templates, f"derive plate '{scene.name}' from '{parent.name}'"
            )
            base_params = {
                "positive_prompt": prompts.plate_edit_prompt(manifest, scene.name, parent.name),
                "negative_prompt": negative,
                "width": video.width,
                "height": video.height,
                ref_param: comfy.upload(parent_plate_path),
            }
        else:
            chosen_template = t2i
            base_params = {
                "positive_prompt": prompts.plate_prompt(manifest, scene.name),
                "negative_prompt": negative,
                "width": video.width,
                "height": video.height,
            }
        results[scene.name] = render_candidate_batch(
            comfy=comfy,
            template=chosen_template,
            base_params=base_params,
            base_seed=project.config.seed + 1000 + scene_index * 100,
            count=count,
            folder=project.plates_dir,
            prefix=plate_slug(scene.name),
            label=f"Plate for '{scene.name}'",
        )
    return results


def lock_plate(project: Project, manifest: Manifest, image: Path, set_name: str) -> Path:
    """Copy the chosen image into the project and record it on the set."""
    plate = load_rgb_candidate(image, "Plate")
    scene = _find_set(manifest, set_name)
    project.plates_dir.mkdir(parents=True, exist_ok=True)
    locked = project.path(Path("assets") / "plates" / f"{plate_slug(scene.name)}.png")
    plate.save(locked)
    scene.plate = project.rel(locked)
    return locked
