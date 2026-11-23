# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Character reference sheets: the identity anchors that stills are built from and checked against.

A sheet is one image with four panels (front, profile, three-quarter and face detail) on a plain
background. It is cut into panels so each can be used on its own: the front panel is the reference
image for stills, and the face and angled panels give the identity check several angles.

The first sheet is generated from the written appearance. Sheets for other wardrobe states are
edits of the first, changing only the clothes, so it is the same person in every costume.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from PIL import Image

from mvdirector import media
from mvdirector.audio.stems import atomic_copy
from mvdirector.comfy.client import ComfyClient
from mvdirector.comfy.workflow import WorkflowTemplate
from mvdirector.errors import ConfigError
from mvdirector.generation.candidates import (
    atomic_save_png,
    ensure_upload_audio_track,
    load_rgb_candidate,
    render_candidate_batch,
)
from mvdirector.models import COLUMNS_PANEL_NAMES, PANEL_NAMES, Manifest, ReferenceSet
from mvdirector.planning import prompts
from mvdirector.planning.segmenter import frames_for
from mvdirector.project import Project, require_safe_id

log = logging.getLogger(__name__)

# Allowed width/height ratios, to catch an image that is not a sheet in the configured layout.
_ASPECT_RANGE = {"row": (1.6, 10.0), "grid": (0.75, 1.35), "columns": (1.1, 2.2)}
_ALLOWED_VIDEO_EXTENSIONS = frozenset({".mp4", ".mov", ".webm", ".mkv"})


def split_sheet(image: Image.Image, layout: str, inset: float) -> dict[str, Image.Image]:
    """Cut a four-panel sheet into panels.

    ``row`` is four panels left to right (front, profile, three_quarter, face). ``grid`` is a 2x2
    grid read left to right, top to bottom. ``columns`` has two stacked headshots on the left
    (face, profile) and full-height front and back columns across the center and right.
    ``inset`` trims that share of each panel on every side to drop dividers and labels.
    """
    width, height = image.size
    if layout == "row":
        names: tuple[str, ...] = PANEL_NAMES
        cells = [(i * width // 4, 0, (i + 1) * width // 4, height) for i in range(4)]
    elif layout == "grid":
        names = PANEL_NAMES
        cells = [
            (x * width // 2, y * height // 2, (x + 1) * width // 2, (y + 1) * height // 2)
            for y in range(2)
            for x in range(2)
        ]
    elif layout == "columns":
        names = COLUMNS_PANEL_NAMES
        c1, c2, mid = width // 3, 2 * width // 3, height // 2
        cells = [
            (0, 0, c1, mid),
            (0, mid, c1, height),
            (c1, 0, c2, height),
            (c2, 0, width, height),
        ]
    else:
        raise ConfigError(f"Unknown sheet layout '{layout}'. Use 'row', 'grid' or 'columns'.")
    panels = {}
    for name, (left, top, right, bottom) in zip(names, cells):
        dx, dy = int((right - left) * inset), int((bottom - top) * inset)
        if right - dx <= left + dx or bottom - dy <= top + dy:
            raise ConfigError(
                f"Reference sheet ({width}x{height}) is too small for layout '{layout}' "
                f"with panel_inset={inset}."
            )
        panels[name] = image.crop((left + dx, top + dy, right - dx, bottom - dy))
    return panels


def _resolve_character(manifest: Manifest, character_id: str | None) -> tuple[str | None, Any]:
    if character_id is not None:
        require_safe_id(character_id, "character id")
        if not manifest.characters:
            raise ConfigError("This manifest has a single character; omit --character.")
        resolved = manifest.resolve_character_id(character_id)
        if resolved is None:
            choices = ", ".join(manifest.characters)
            raise ConfigError(f"Unknown character '{character_id}'. Choices: {choices}.")
        return resolved, manifest.characters[resolved]
    if manifest.characters:
        primary_id = manifest.primary_character_id()
        assert primary_id is not None
        return primary_id, manifest.characters[primary_id]
    return None, manifest.character


def is_base_state(
    manifest: Manifest, state_id: str | None, character_id: str | None = None
) -> bool:
    _, character = _resolve_character(manifest, character_id)
    wardrobe = character.wardrobe_ids
    return state_id is None or not wardrobe or state_id == wardrobe[0]


_is_base_state = is_base_state


def _check_state(manifest: Manifest, state_id: str | None, character_id: str | None = None) -> None:
    if state_id is not None:
        require_safe_id(state_id, "wardrobe state")
    _, character = _resolve_character(manifest, character_id)
    wardrobe = character.wardrobe_ids
    if state_id is not None and wardrobe and state_id not in wardrobe:
        raise ConfigError(f"Unknown wardrobe state '{state_id}'. Choices: {', '.join(wardrobe)}.")
    if state_id is not None and not wardrobe:
        raise ConfigError("This character has no wardrobe states; omit --state.")


def _target_state(
    manifest: Manifest, state_id: str | None, character_id: str | None
) -> tuple[str | None, Any, str, str]:
    char_id, character = _resolve_character(manifest, character_id)
    _check_state(manifest, state_id, character_id)
    key = require_safe_id(character.reference_key(state_id), "wardrobe state")
    prefix = (
        f"{char_id}_{key}"
        if (char_id and (len(manifest.characters) > 1 or character_id is not None))
        else key
    )
    return char_id, character, key, prefix


def _seed_offset(manifest: Manifest, char_id: str | None, character: Any, key: str) -> int:
    char_offset = (
        list(manifest.characters).index(char_id) * 1000
        if char_id and char_id in manifest.characters
        else 0
    )
    return (
        char_offset
        + (character.wardrobe_ids.index(key) if key in character.wardrobe_ids else 0) * 100
    )


def uses_still_workflow(
    manifest: Manifest,
    state_id: str | None,
    character_id: str | None = None,
    *,
    from_character: str | None = None,
) -> bool:
    """Return True when generating this sheet edits an existing reference via ``still``."""
    if not is_base_state(manifest, state_id, character_id):
        return True
    _, character = _resolve_character(manifest, character_id)
    return from_character is not None or character.parent_character is not None


def generate_candidates(
    project: Project,
    manifest: Manifest,
    comfy: ComfyClient,
    templates: dict[str, WorkflowTemplate],
    state_id: str | None,
    count: int,
    character_id: str | None = None,
    *,
    from_character: str | None = None,
) -> list[Path]:
    """Render ``count`` candidate sheets for a person to choose from.

    The base sheet comes from the ``t2i`` workflow, or edits another character's locked base sheet
    via ``still`` when ``from_character`` (or ``character.parent_character``) is set. Sheets for
    later wardrobe states are edits of this character's locked base sheet via ``still``.
    """
    char_id, character, key, prefix = _target_state(manifest, state_id, character_id)
    config = project.config.reference
    if not character.appearance.strip():
        raise ConfigError(
            "The manifest has no character appearance. Add one under 'character' in "
            "manifest.json, or run `mvd plan` with the LLM enabled."
        )
    width, height = config.size()
    negative = f"{manifest.visual_bible.negative_prompt}, {prompts.SHEET_NEGATIVE}".strip(", ")
    base_params: dict[str, Any] = {
        "negative_prompt": negative,
        "width": width,
        "height": height,
    }

    if is_base_state(manifest, state_id, character_id):
        source_char_id = (
            from_character if from_character is not None else character.parent_character
        )
        if source_char_id is not None:
            if not source_char_id.strip():
                raise ConfigError("Provide a non-empty character id for --from-character.")
            src_id, src_char = _resolve_character(manifest, source_char_id)
            if src_id == char_id:
                raise ConfigError(f"Character '{char_id}' cannot derive from itself.")
            template = templates.get("still")
            if template is None:
                raise ConfigError("A 'still' workflow is required to derive a character sheet.")
            if not template.accepts("reference_image"):
                raise ConfigError(
                    f"Workflow '{template.name}' must bind 'reference_image' to derive a "
                    f"character sheet from '{src_id}'."
                )
            src_ref = src_char.default_reference
            if src_ref is None or src_ref.sheet is None:
                raise ConfigError(
                    f"Lock the base sheet for '{src_id}' before deriving '{char_id}': "
                    f"`mvd reference generate --character {src_id}`, then "
                    f"`mvd reference lock <image> --character {src_id}`."
                )
            src_sheet_path = project.path(src_ref.sheet)
            if not src_sheet_path.is_file():
                raise ConfigError(
                    f"Locked reference sheet for '{src_id}' is missing: {src_ref.sheet}"
                )
            params = {
                **base_params,
                "positive_prompt": prompts.derived_sheet_prompt(
                    manifest, config.layout, character_id=char_id, background=config.background
                ),
                "reference_image": comfy.upload(src_sheet_path),
            }
        else:
            template = templates.get("t2i")
            if template is None:
                raise ConfigError(
                    "A 't2i' workflow is required to generate a base reference sheet."
                )
            params = {
                **base_params,
                "positive_prompt": prompts.sheet_prompt(
                    manifest, config.layout, character_id=char_id, background=config.background
                ),
            }
    else:
        if from_character is not None:
            raise ConfigError(
                "Omit --from-character when generating a later wardrobe state "
                "(it edits this character's own base sheet)."
            )
        template = templates.get("still")
        if template is None:
            raise ConfigError(
                "A 'still' workflow is required to generate later wardrobe reference sheets."
            )
        if template.accepts("reference_image"):
            base = character.default_reference
            if base is None or base.sheet is None:
                char_flag = (
                    f" --character {char_id}" if char_id and len(manifest.characters) > 1 else ""
                )
                raise ConfigError(
                    f"Lock the base sheet before '{state_id}': "
                    f"`mvd reference generate{char_flag}`, then "
                    f"`mvd reference lock <image>{char_flag}`. Later costumes are edits of it."
                )
            base_sheet_path = project.path(base.sheet)
            if not base_sheet_path.is_file():
                raise ConfigError(f"Locked base reference sheet is missing: {base.sheet}")
            params = {
                **base_params,
                "positive_prompt": prompts.state_sheet_prompt(
                    manifest, key, config.layout, character_id=char_id, background=config.background
                ),
                "reference_image": comfy.upload(base_sheet_path),
            }
        else:
            params = {
                **base_params,
                "positive_prompt": prompts.sheet_prompt(
                    manifest,
                    config.layout,
                    state_id=key,
                    character_id=char_id,
                    background=config.background,
                ),
            }

    offset = _seed_offset(manifest, char_id, character, key)
    return render_candidate_batch(
        comfy=comfy,
        template=template,
        base_params=params,
        base_seed=project.config.seed + offset,
        count=count,
        folder=project.reference_dir,
        prefix=prefix,
        label=f"Reference for '{prefix}'",
    )


def _existing_reference(
    manifest: Manifest,
    character: Any,
    key: str,
    state_id: str | None,
    character_id: str | None,
) -> ReferenceSet | None:
    return character.references.get(key) or (
        character.default_reference if is_base_state(manifest, state_id, character_id) else None
    )


def generate_video_candidates(
    project: Project,
    manifest: Manifest,
    comfy: ComfyClient,
    templates: dict[str, WorkflowTemplate],
    state_id: str | None,
    count: int,
    character_id: str | None = None,
) -> list[Path]:
    """Render ``count`` silent character turnaround reference videos from a locked sheet."""
    char_id, character, key, prefix = _target_state(manifest, state_id, character_id)
    ref_set = _existing_reference(manifest, character, key, state_id, character_id)
    if ref_set is None or not ref_set.sheet:
        char_flag = f" --character {char_id}" if char_id and len(manifest.characters) > 1 else ""
        state_flag = f" --state {state_id}" if state_id else ""
        raise ConfigError(
            f"Lock the reference sheet for '{prefix}' first: "
            f"`mvd reference lock <image>{char_flag}{state_flag}`."
        )
    sheet_path = project.path(ref_set.sheet)
    raw_w, raw_h = load_rgb_candidate(sheet_path, "Reference").size
    width = max(2, raw_w - (raw_w % 2))
    height = max(2, raw_h - (raw_h % 2))

    template = templates.get("ref_video") or templates.get("i2v")
    if template is None:
        raise ConfigError("Configure 'ref_video' or 'i2v' under comfy.workflows in project.yaml.")
    video_cfg = project.config.video
    clip_seconds = min(4.0, video_cfg.max_clip_seconds)
    num_frames = frames_for(
        clip_seconds, video_cfg.fps, video_cfg.frame_multiple, video_cfg.frame_offset
    )
    chained = False
    negative = f"{manifest.visual_bible.negative_prompt}, {prompts.SHEET_NEGATIVE}".strip(", ")
    params: dict[str, Any] = {
        "negative_prompt": negative,
        "image": comfy.upload(sheet_path),
        "width": width,
        "height": height,
        "num_frames": num_frames,
        "duration": round(clip_seconds, 3),
        "fps": video_cfg.fps,
    }
    if not is_base_state(manifest, state_id, character_id):
        base = character.default_reference
        if base is not None and base.video and template.accepts("reference_video"):
            base_video_path = project.path(base.video)
            if base_video_path.is_file():
                params["reference_video"] = comfy.upload(
                    ensure_upload_audio_track(project, base_video_path)
                )
                chained = True
    params["positive_prompt"] = prompts.reference_video_prompt(
        manifest,
        state_id=key,
        character_id=char_id,
        layout=project.config.reference.layout,
        background=project.config.reference.background,
        chained_from_base=chained,
    )
    offset = _seed_offset(manifest, char_id, character, key) + 50
    return render_candidate_batch(
        comfy=comfy,
        template=template,
        base_params=params,
        base_seed=project.config.seed + offset,
        count=count,
        folder=project.reference_dir,
        prefix=f"{prefix}_video",
        label=f"Reference video for '{prefix}'",
        extension=".mp4",
    )


def lock_reference_video(
    project: Project,
    manifest: Manifest,
    video_path: Path,
    state_id: str | None = None,
    character_id: str | None = None,
) -> str:
    """Copy a chosen turnaround reference video into the project and record it on the state."""
    if not video_path.is_file():
        raise ConfigError(f"Reference video not found: {video_path}")
    suffix = video_path.suffix.lower()
    if suffix not in _ALLOWED_VIDEO_EXTENSIONS:
        allowed = ", ".join(sorted(_ALLOWED_VIDEO_EXTENSIONS))
        raise ConfigError(
            f"Unsupported reference video extension {suffix!r}; expected one of: {allowed}."
        )
    info = media.probe(video_path)
    if info.width is None or info.height is None:
        raise ConfigError(f"Reference video has no video stream: {video_path}")
    char_id, character, key, prefix = _target_state(manifest, state_id, character_id)
    ref_set = _existing_reference(manifest, character, key, state_id, character_id)
    if ref_set is None:
        raise ConfigError(
            f"Lock the reference sheet for '{prefix}' before locking a reference video."
        )
    project.reference_dir.mkdir(parents=True, exist_ok=True)
    dest = project.path(Path("assets") / "reference" / f"{prefix}_video{suffix}")
    if video_path.resolve() != dest.resolve():
        atomic_copy(video_path, dest)
    rel = project.rel(dest)
    ref_set.video = rel
    if (char_id is None or char_id == manifest.primary_character_id()) and (
        manifest.character is not character
    ):
        mirror = manifest.character.references.get(key)
        if mirror is not None:
            mirror.video = rel
    return rel


def lock_reference(
    project: Project,
    manifest: Manifest,
    image: Path,
    state_id: str | None = None,
    split: bool = True,
    character_id: str | None = None,
) -> ReferenceSet:
    """Copy the chosen sheet into the project, cut it into panels and record it.

    With ``split`` false the image is kept whole and used as the front reference, for a single
    portrait or photo that is not a four-panel sheet.
    """
    sheet = load_rgb_candidate(image, "Reference")
    char_id, character, key, prefix = _target_state(manifest, state_id, character_id)
    config = project.config.reference
    if split:
        low, high = _ASPECT_RANGE[config.layout]
        ratio = sheet.width / sheet.height
        if not low <= ratio <= high:
            raise ConfigError(
                f"{image.name} is {sheet.width}x{sheet.height}, which does not look like a "
                f"'{config.layout}' sheet. Check reference.layout in project.yaml, or pass "
                "--single if this is one portrait and not a four-panel sheet."
            )

    project.reference_dir.mkdir(parents=True, exist_ok=True)
    sheet_path = project.path(Path("assets") / "reference" / f"{prefix}_sheet.png")
    atomic_save_png(sheet, sheet_path)
    panels: dict[str, str] = {}
    if split:
        panel_dir = project.path(Path("assets") / "reference" / prefix)
        panel_dir.mkdir(parents=True, exist_ok=True)
        for name, crop in split_sheet(sheet, config.layout, config.panel_inset).items():
            path = project.path(Path("assets") / "reference" / prefix / f"{prefix}_{name}.png")
            atomic_save_png(crop, path)
            panels[name] = project.rel(path)
    existing = character.references.get(key)
    reference = ReferenceSet(
        sheet=project.rel(sheet_path),
        panels=panels,
        video=existing.video if existing else None,
    )
    character.references[key] = reference
    if (char_id is None or char_id == manifest.primary_character_id()) and (
        manifest.character is not character
    ):
        manifest.character.references[key] = reference
    return reference
