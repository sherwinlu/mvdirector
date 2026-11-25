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
from dataclasses import dataclass, field
from pathlib import Path

from mvdirector.comfy.client import ComfyClient
from mvdirector.comfy.workflow import WorkflowTemplate
from mvdirector.errors import ConfigError
from mvdirector.generation.candidates import (
    atomic_save_png,
    load_rgb_candidate,
    render_candidate_batch,
)
from mvdirector.generation.reference import split_sheet
from mvdirector.models import (
    PLATE_ANGLE_NAMES,
    Framing,
    Manifest,
    SetDesign,
    Shot,
    ShotType,
    ascii_fold,
)
from mvdirector.planning import prompts
from mvdirector.project import Project

log = logging.getLogger(__name__)

_TURNAROUND_ASPECT_RANGE = (0.45, 2.5)
_REVERSE_ANGLE_RE = re.compile(
    r"\b(?:reverse(?:\s+(?:angle|shot|view))?|from\s+behind|"
    r"over[\s-]+(?:the[\s-]+)?(?:[a-z]+'s\s+)?shoulder|looking\s+back|"
    r"opposite\s+(?:side|wall|direction)|behind\s+(?:the\s+)?(?:singer|performer|subject))\b",
    re.IGNORECASE,
)
_SIDE_ANGLE_RE = re.compile(
    r"\b(?:side\s+(?:angle|view|profile)|profile\s+(?:shot|view|angle)|"
    r"lateral|90[\s-]*degree|from\s+the\s+side)\b",
    re.IGNORECASE,
)
_DETAIL_ANGLE_RE = re.compile(
    r"\b(?:detail(?:\s+shot)?|macro|insert\s+shot|architectural\s+detail|"
    r"close[\s-]*up\s+(?:of|on)\s+(?:the\s+)?(?:wall|door|window|floor|texture|surface|"
    r"fixture|table|object))\b",
    re.IGNORECASE,
)
_WIDE_ANGLE_RE = re.compile(
    r"\b(?:wide\s+(?:shot|angle|establishing)|establishing\s+shot|full\s+room)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PlateTurnaroundResult:
    set_name: str
    candidates: list[Path]
    sheet: str | None = None
    panels: dict[str, str] = field(default_factory=dict)


def plate_slug(name: str) -> str:
    """A collision-safe ASCII filename stem (<= 64 chars) for a set name across any script."""
    cleaned = unicodedata.normalize("NFKC", name).strip()
    stem = re.sub(r"[^a-z0-9]+", "_", ascii_fold(cleaned).lower()).strip("_") or "set"
    if all(ord(ch) < 128 for ch in cleaned) and len(stem) <= 64:
        return stem
    digest = hashlib.sha256(cleaned.casefold().encode("utf-8")).hexdigest()[:6]
    return f"{stem[:57].rstrip('_') or 'set'}_{digest}"


def find_set(manifest: Manifest, name: str) -> SetDesign:
    scene = manifest.visual_bible.find_set(name)
    if scene is None:
        names = ", ".join(s.name for s in manifest.visual_bible.settings) or "(none)"
        raise ConfigError(f"No set named '{name}'. Sets in the bible: {names}")
    return scene


def resolve_ref_workflow(
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
    scenes = [find_set(manifest, set_name)] if set_name else bible.settings
    base_negative = prompts.merge_negative_prompt(bible.negative_prompt, bible.exclusions)
    negative = f"{base_negative}, {prompts.PLATE_NEGATIVE}".strip(", ")

    uploaded_style_ref: str | None = None
    style_template: WorkflowTemplate | None = None
    style_param: str | None = None
    if style_ref is not None:
        load_rgb_candidate(style_ref, "Style reference")
        style_template, style_param = resolve_ref_workflow(templates, "use --style-ref")
        uploaded_style_ref = comfy.upload(style_ref)

    results: dict[str, list[Path]] = {}
    for scene_index, scene in enumerate(scenes):
        if embed_set is not None:
            embedded = find_set(manifest, embed_set)
            if embedded is scene:
                raise ConfigError(f"Set '{scene.name}' cannot embed itself.")
            if not embedded.plate:
                raise ConfigError(
                    f"Lock a plate for '{embedded.name}' before embedding it in '{scene.name}': "
                    f'`mvd plates lock <image> --set "{embedded.name}"`.'
                )
            embedded_path = project.path(embedded.plate)
            if not embedded_path.is_file():
                raise ConfigError(f"Plate file for '{embedded.name}' is missing: {embedded.plate}")
            chosen_template, ref_param = resolve_ref_workflow(
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
            parent = find_set(manifest, source_name)
            if parent is scene:
                raise ConfigError(f"Set '{scene.name}' cannot derive from itself.")
            if not parent.plate:
                raise ConfigError(
                    f"Lock a plate for '{parent.name}' before deriving '{scene.name}': "
                    f'`mvd plates lock <image> --set "{parent.name}"`.'
                )
            parent_plate_path = project.path(parent.plate)
            if not parent_plate_path.is_file():
                raise ConfigError(f"Plate file for '{parent.name}' is missing: {parent.plate}")
            chosen_template, ref_param = resolve_ref_workflow(
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
    scene = find_set(manifest, set_name)
    project.plates_dir.mkdir(parents=True, exist_ok=True)
    locked = project.path(Path("assets") / "plates" / f"{plate_slug(scene.name)}.png")
    atomic_save_png(plate, locked)
    scene.plate = project.rel(locked)
    return locked


def select_plate_angle(shot: Shot, scene: SetDesign | None = None) -> str | None:
    """Pick the turnaround panel angle matching ``shot``'s override, prompt cues, or framing."""
    del scene  # reserved for future per-set angle metadata
    if shot.plate_angle is not None:
        return shot.plate_angle
    raw = (shot.still.prompt_override or shot.still_prompt).strip()
    if _REVERSE_ANGLE_RE.search(raw):
        return "reverse"
    if _SIDE_ANGLE_RE.search(raw):
        return "side"
    if _DETAIL_ANGLE_RE.search(raw) or (
        shot.type is ShotType.BROLL and shot.framing is Framing.CLOSE_UP
    ):
        return "detail"
    if _WIDE_ANGLE_RE.search(raw) or shot.framing is Framing.WIDE:
        return "wide"
    return None


def lock_turnaround(
    project: Project,
    manifest: Manifest,
    image: Path,
    set_name: str,
    *,
    inset: float = 0.012,
) -> SetDesign:
    """Save a 2x2 multi-angle set turnaround sheet and crop its 4 angle panels."""
    if isinstance(inset, bool) or not 0.0 <= inset < 0.2:
        raise ConfigError(f"Turnaround panel inset must be in [0.0, 0.2), got {inset!r}.")
    sheet = load_rgb_candidate(image, "Plate turnaround")
    scene = find_set(manifest, set_name)
    low, high = _TURNAROUND_ASPECT_RANGE
    ratio = sheet.width / sheet.height
    if not low <= ratio <= high:
        raise ConfigError(
            f"{image.name} is {sheet.width}x{sheet.height}, whose aspect ratio ({ratio:.2f}) "
            f"is outside the supported 2x2 plate turnaround range ({low}..{high})."
        )
    slug = plate_slug(scene.name)
    project.plates_dir.mkdir(parents=True, exist_ok=True)
    sheet_path = project.path(Path("assets") / "plates" / f"{slug}_sheet.png")
    atomic_save_png(sheet, sheet_path)
    scene.sheet = project.rel(sheet_path)

    panel_dir = project.path(Path("assets") / "plates" / slug)
    panel_dir.mkdir(parents=True, exist_ok=True)
    cropped = split_sheet(sheet, "grid", inset, panel_names=PLATE_ANGLE_NAMES)
    panels: dict[str, str] = {}
    for angle, crop in cropped.items():
        panel_path = project.path(Path("assets") / "plates" / slug / f"{slug}_{angle}.png")
        atomic_save_png(crop, panel_path)
        panels[angle] = project.rel(panel_path)
    scene.panels = panels
    if scene.plate is None and "wide" in panels:
        scene.plate = panels["wide"]
    return scene


def generate_turnaround(
    project: Project,
    manifest: Manifest,
    comfy: ComfyClient,
    templates: dict[str, WorkflowTemplate] | WorkflowTemplate,
    set_name: str,
    count: int = 1,
    *,
    lock: bool = True,
    inset: float = 0.012,
) -> PlateTurnaroundResult:
    """Generate a 2x2 multi-angle turnaround sheet from one approved set plate."""
    if isinstance(inset, bool) or not 0.0 <= inset < 0.2:
        raise ConfigError(f"Turnaround panel inset must be in [0.0, 0.2), got {inset!r}.")
    bible, video = manifest.visual_bible, project.config.video
    scene = find_set(manifest, set_name)
    if not scene.plate:
        raise ConfigError(
            f"Lock a plate for '{scene.name}' before generating a turnaround: "
            f'`mvd plates lock <image> --set "{scene.name}"`.'
        )
    plate_path = project.path(scene.plate)
    if not plate_path.is_file():
        raise ConfigError(f"Plate file for '{scene.name}' is missing: {scene.plate}")
    tpl_map = templates if isinstance(templates, dict) else {"still": templates}
    still_tpl, ref_param = resolve_ref_workflow(tpl_map, f"generate turnaround for '{scene.name}'")
    base_negative = prompts.merge_negative_prompt(bible.negative_prompt, bible.exclusions)
    negative = f"{base_negative}, {prompts.PLATE_TURNAROUND_NEGATIVE}".strip(", ")
    scene_index = next((i for i, item in enumerate(bible.settings) if item.name == scene.name), 0)
    base_params = {
        "positive_prompt": prompts.plate_turnaround_prompt(manifest, scene.name),
        "negative_prompt": negative,
        "width": video.width,
        "height": video.height,
        ref_param: comfy.upload(plate_path),
    }
    candidates = render_candidate_batch(
        comfy=comfy,
        template=still_tpl,
        base_params=base_params,
        base_seed=project.config.seed + 1500 + scene_index * 100,
        count=count,
        folder=project.plates_dir,
        prefix=f"{plate_slug(scene.name)}_turnaround",
        label=f"Plate turnaround for '{scene.name}'",
    )
    if lock:
        lock_turnaround(project, manifest, candidates[0], scene.name, inset=inset)
    return PlateTurnaroundResult(
        set_name=scene.name,
        candidates=candidates,
        sheet=scene.sheet,
        panels=dict(scene.panels),
    )
