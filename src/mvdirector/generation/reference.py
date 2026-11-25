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
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError

from mvdirector import media
from mvdirector.audio.stems import atomic_copy
from mvdirector.comfy.client import ComfyClient
from mvdirector.comfy.workflow import WorkflowTemplate
from mvdirector.config import ReferenceConfig
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
from mvdirector.store import path_is_file

log = logging.getLogger(__name__)

# Allowed width/height ratios, to catch an image that is not a sheet in the configured layout.
_ASPECT_RANGE = {"row": (1.6, 10.0), "grid": (0.75, 1.35), "columns": (1.1, 2.2)}
_ALLOWED_VIDEO_EXTENSIONS = frozenset({".mp4", ".mov", ".webm", ".mkv"})
SEED_ANGLE_NAMES = ("front", "profile", "three_quarter", "face", "back")
SEED_GRID_PANEL_NAMES = ("front", "profile", "back", "face")
MAX_SEED_PHOTOS = 10


@dataclass(frozen=True)
class SeedReferenceResult:
    reference: ReferenceSet | None
    sheet: Path | None
    panels: dict[str, Path]
    candidates: list[Path]
    mode: Literal["direct", "generate"]
    layout: str


def split_sheet(
    image: Image.Image,
    layout: str,
    inset: float,
    *,
    panel_names: tuple[str, ...] | None = None,
) -> dict[str, Image.Image]:
    """Cut a four-panel sheet into panels.

    ``row`` is four panels left to right (front, profile, three_quarter, face). ``grid`` is a 2x2
    grid read left to right, top to bottom. ``columns`` has two stacked headshots on the left
    (face, profile) and full-height front and back columns across the center and right.
    ``inset`` trims that share of each panel on every side to drop dividers and labels.
    """
    if panel_names is not None and len(panel_names) != 4:
        raise ConfigError(f"Expected 4 panel names for '{layout}' sheet, got {len(panel_names)}.")
    width, height = image.size
    if layout == "row":
        names: tuple[str, ...] = panel_names or PANEL_NAMES
        cells = [(i * width // 4, 0, (i + 1) * width // 4, height) for i in range(4)]
    elif layout == "grid":
        names = panel_names or PANEL_NAMES
        cells = [
            (x * width // 2, y * height // 2, (x + 1) * width // 2, (y + 1) * height // 2)
            for y in range(2)
            for x in range(2)
        ]
    elif layout == "columns":
        names = panel_names or COLUMNS_PANEL_NAMES
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


def target_state(
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
    char_id, character, key, prefix = target_state(manifest, state_id, character_id)
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
    char_id, character, key, prefix = target_state(manifest, state_id, character_id)
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
    if not path_is_file(video_path):
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
    char_id, character, key, prefix = target_state(manifest, state_id, character_id)
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
    char_id, character, key, prefix = target_state(manifest, state_id, character_id)
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


def _mirror_primary_reference(
    manifest: Manifest, char_id: str | None, character: Any, key: str, reference: ReferenceSet
) -> None:
    if (char_id is None or char_id == manifest.primary_character_id()) and (
        manifest.character is not character
    ):
        manifest.character.references[key] = reference


def _load_seed_photo(photo: Path, label: str) -> Image.Image:
    """Load a seed photo upright (EXIF orientation applied) as metadata-free RGB pixels.

    Phone photos store their rotation in EXIF and often carry GPS/camera tags; the pixels are
    rotated upright and copied onto a fresh image so no metadata reaches disk or ComfyUI.
    """
    image = Path(photo)
    if not path_is_file(image):
        raise ConfigError(f"{label} image not found: {image}")
    try:
        with Image.open(image) as opened:
            upright = ImageOps.exif_transpose(opened).convert("RGB")
    except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
        raise ConfigError(f"{label} image could not be opened: {image} ({exc})") from exc
    clean = Image.new("RGB", upright.size)
    clean.paste(upright)
    return clean


def _normalize_seed_angle(raw: str) -> str:
    cleaned = (
        raw.strip()
        .lower()
        .replace("3/4", "three_quarter")
        .replace("-", "_")
        .replace(" ", "_")
        .replace("/", "_")
    )
    aliases = {"rear": "back", "3_4": "three_quarter"}
    normalized = aliases.get(cleaned, cleaned)
    if normalized not in SEED_ANGLE_NAMES:
        allowed = ", ".join(SEED_ANGLE_NAMES)
        raise ConfigError(f"Unknown seed photo angle {raw!r}; expected one of: {allowed}.")
    return normalized


def normalize_seed_angles(angles: Sequence[str] | None, photo_count: int) -> list[str] | None:
    """Validate one angle per seed photo and return them normalized (``None`` when not given)."""
    if angles is None:
        return None
    if len(angles) != photo_count:
        raise ConfigError(
            f"Expected {photo_count} angle(s) matching the provided photos, got {len(angles)}."
        )
    normalized: list[str] = []
    for raw_angle in angles:
        norm = _normalize_seed_angle(raw_angle)
        if norm in normalized:
            raise ConfigError(f"Duplicate seed photo angle: {norm!r}.")
        normalized.append(norm)
    return normalized


def _compose_photo_grid(
    images: Sequence[Image.Image],
    size: tuple[int, int],
    *,
    background: str = "white",
    grid: tuple[int, int] | None = None,
) -> Image.Image:
    """Arrange 1–10 photos into an aspect-preserving studio grid with neutral dividers."""
    if grid is not None:
        cols, rows = grid
    elif len(images) <= 4:
        cols, rows = 2, 2
    elif len(images) <= 6:
        cols, rows = 3, 2
    elif len(images) <= 8:
        cols, rows = 4, 2
    else:
        cols, rows = 5, 2

    width, height = size
    fill = (128, 128, 128) if background == "grey" else (255, 255, 255)
    canvas = Image.new("RGB", (width, height), fill)
    draw = ImageDraw.Draw(canvas)

    for idx, img in enumerate(images[: cols * rows]):
        col = idx % cols
        row = idx // cols
        x0 = col * width // cols
        y0 = row * height // rows
        x1 = (col + 1) * width // cols
        y1 = (row + 1) * height // rows
        cell_w, cell_h = x1 - x0, y1 - y0
        pad = max(4, min(cell_w, cell_h) // 40)
        fitted = ImageOps.contain(
            img,
            (max(1, cell_w - 2 * pad), max(1, cell_h - 2 * pad)),
            method=Image.Resampling.LANCZOS,
        )
        px = x0 + (cell_w - fitted.width) // 2
        py = y0 + (cell_h - fitted.height) // 2
        canvas.paste(fitted, (px, py))

    divider = (180, 180, 180)
    line_w = max(2, min(width, height) // 300)
    for col in range(1, cols):
        x = col * width // cols
        draw.line([(x, 0), (x, height)], fill=divider, width=line_w)
    for row in range(1, rows):
        y = row * height // rows
        draw.line([(0, y), (width, y)], fill=divider, width=line_w)
    return canvas


def _available_ref_slots(template: WorkflowTemplate) -> list[str]:
    slots: list[str] = []
    seen_targets: set[tuple[str, str]] = set()
    candidates = [
        "reference_image",
        *(f"reference_image_{n}" for n in range(2, MAX_SEED_PHOTOS + 1)),
    ]
    for param in candidates:
        if not template.accepts(param):
            continue
        targets = {(t.node, t.field) for t in template.bindings.inputs.get(param, ())}
        if not targets or not targets.isdisjoint(seen_targets):
            continue
        seen_targets.update(targets)
        slots.append(param)
    return slots


def seed_from_photos(
    project: Project,
    manifest: Manifest,
    photos: Sequence[Path],
    *,
    angles: Sequence[str] | None = None,
    state_id: str | None = None,
    character_id: str | None = None,
    mode: Literal["direct", "generate"] = "direct",
    layout: Literal["row", "grid", "columns"] | None = None,
    include_back: bool = True,
    count: int = 1,
    lock: bool = True,
    comfy: ComfyClient | None = None,
    templates: dict[str, WorkflowTemplate] | None = None,
) -> SeedReferenceResult:
    """Seed a character reference sheet from 1–10 real photos (direct or ComfyUI synthesis)."""
    if mode not in ("direct", "generate"):
        raise ConfigError(f"Unknown seed mode {mode!r}; expected 'direct' or 'generate'.")
    if isinstance(count, bool) or count < 1:
        raise ConfigError(f"Candidate count must be >= 1, got {count}.")
    if not photos:
        raise ConfigError("Provide at least 1 reference photo (up to 10).")
    if len(photos) > MAX_SEED_PHOTOS:
        raise ConfigError(
            f"At most {MAX_SEED_PHOTOS} seed photos are supported, got {len(photos)}."
        )

    normalized_angles = normalize_seed_angles(angles, len(photos))

    char_id, character, key, prefix = target_state(manifest, state_id, character_id)
    config = project.config.reference
    loaded_images = [
        _load_seed_photo(Path(photo), f"Seed photo #{idx + 1}") for idx, photo in enumerate(photos)
    ]

    if mode == "direct":
        effective_layout = layout or "grid"
        if effective_layout not in ("row", "grid", "columns"):
            raise ConfigError(
                f"Unknown sheet layout '{effective_layout}'. Use 'row', 'grid' or 'columns'."
            )
        default_order = (
            ("front", "profile", "face", "back", "three_quarter")
            if effective_layout == "columns"
            else ("front", "profile", "three_quarter", "face", "back")
        )
        project.reference_dir.mkdir(parents=True, exist_ok=True)
        panel_dir = project.path(Path("assets") / "reference" / prefix)
        panel_dir.mkdir(parents=True, exist_ok=True)

        panel_rel_map: dict[str, str] = {}
        panel_paths: dict[str, Path] = {}
        for idx, img in enumerate(loaded_images):
            angle_name = (
                normalized_angles[idx]
                if normalized_angles is not None
                else (default_order[idx] if idx < len(default_order) else None)
            )
            if angle_name is None:
                continue
            dst = project.path(Path("assets") / "reference" / prefix / f"{prefix}_{angle_name}.png")
            atomic_save_png(img, dst)
            panel_rel_map[angle_name] = project.rel(dst)
            panel_paths[angle_name] = dst

        if len(loaded_images) <= 4:
            sheet_size = (
                config.size() if config.layout == "grid" else ReferenceConfig(layout="grid").size()
            )
        else:
            cols = (len(loaded_images) + 1) // 2
            sheet_size = (cols * 512, 1024)
        sheet_img = _compose_photo_grid(loaded_images, sheet_size, background=config.background)
        sheet_path = project.path(Path("assets") / "reference" / f"{prefix}_sheet.png")
        atomic_save_png(sheet_img, sheet_path)

        reference: ReferenceSet | None = None
        if lock:
            existing = character.references.get(key)
            reference = ReferenceSet(
                sheet=project.rel(sheet_path),
                panels=panel_rel_map,
                video=existing.video if existing else None,
            )
            character.references[key] = reference
            _mirror_primary_reference(manifest, char_id, character, key, reference)
        return SeedReferenceResult(
            reference=reference,
            sheet=sheet_path,
            panels=panel_paths,
            candidates=[],
            mode="direct",
            layout=effective_layout,
        )

    # mode == "generate": synthesize locked 4-panel turnaround via ComfyUI `still` edit
    still_tpl = (templates or {}).get("still")
    if still_tpl is None:
        raise ConfigError("A 'still' workflow is required to generate a seeded reference sheet.")
    if not still_tpl.accepts("reference_image"):
        raise ConfigError(
            f"Workflow '{still_tpl.name}' must bind 'reference_image' to seed from photos."
        )
    if comfy is None:
        raise ConfigError("A ComfyUI client is required when mode='generate'.")

    effective_layout = layout or ("columns" if include_back else config.layout)
    if effective_layout not in ("row", "grid", "columns"):
        raise ConfigError(
            f"Unknown sheet layout '{effective_layout}'. Use 'row', 'grid' or 'columns'."
        )
    width, height = (
        config.size()
        if effective_layout == config.layout
        else ReferenceConfig(layout=effective_layout).size()
    )
    negative = prompts.merge_negative_prompt(
        f"{manifest.visual_bible.negative_prompt}, {prompts.SHEET_NEGATIVE}".strip(", "),
        manifest.visual_bible.exclusions,
    )
    params: dict[str, Any] = {
        "positive_prompt": prompts.seed_sheet_prompt(
            manifest,
            effective_layout,
            state_id=key,
            character_id=char_id,
            background=config.background,
            photo_count=len(loaded_images),
            include_back=include_back,
            include_grid_back=(effective_layout == "grid" and include_back),
        ),
        "negative_prompt": negative,
        "width": width,
        "height": height,
    }
    slots = _available_ref_slots(still_tpl)
    project.reference_dir.mkdir(parents=True, exist_ok=True)
    if len(loaded_images) <= len(slots):
        # Upload re-encoded, metadata-free copies: never the user's original files (EXIF/GPS).
        for number, (slot, img) in enumerate(zip(slots, loaded_images), start=1):
            clean_path = project.path(
                Path("assets") / "reference" / f"{prefix}_seed_photo_{number}.png"
            )
            atomic_save_png(img, clean_path)
            params[slot] = comfy.upload(clean_path)
    else:
        collage = _compose_photo_grid(loaded_images, (1536, 1536), background=config.background)
        collage_path = project.path(Path("assets") / "reference" / f"{prefix}_seed_input.png")
        atomic_save_png(collage, collage_path)
        params["reference_image"] = comfy.upload(collage_path)

    offset = _seed_offset(manifest, char_id, character, key) + 25
    candidates = render_candidate_batch(
        comfy=comfy,
        template=still_tpl,
        base_params=params,
        base_seed=project.config.seed + offset,
        count=count,
        folder=project.reference_dir,
        prefix=f"{prefix}_seed",
        label=f"Seeded reference for '{prefix}'",
    )

    if not lock:
        return SeedReferenceResult(
            reference=None,
            sheet=None,
            panels={},
            candidates=candidates,
            mode="generate",
            layout=effective_layout,
        )

    chosen = load_rgb_candidate(candidates[0], "Seeded reference")
    low, high = _ASPECT_RANGE[effective_layout]
    ratio = chosen.width / chosen.height
    if not low <= ratio <= high:
        raise ConfigError(
            f"{candidates[0].name} is {chosen.width}x{chosen.height}, which does not match "
            f"'{effective_layout}' layout aspect ratio ({low}..{high})."
        )
    project.reference_dir.mkdir(parents=True, exist_ok=True)
    sheet_path = project.path(Path("assets") / "reference" / f"{prefix}_sheet.png")
    atomic_save_png(chosen, sheet_path)

    panel_names = SEED_GRID_PANEL_NAMES if (effective_layout == "grid" and include_back) else None
    panel_dir = project.path(Path("assets") / "reference" / prefix)
    panel_dir.mkdir(parents=True, exist_ok=True)
    panel_rel_map = {}
    panel_paths = {}
    for name, crop in split_sheet(
        chosen, effective_layout, config.panel_inset, panel_names=panel_names
    ).items():
        path = project.path(Path("assets") / "reference" / prefix / f"{prefix}_{name}.png")
        atomic_save_png(crop, path)
        panel_rel_map[name] = project.rel(path)
        panel_paths[name] = path

    existing = character.references.get(key)
    reference = ReferenceSet(
        sheet=project.rel(sheet_path),
        panels=panel_rel_map,
        video=existing.video if existing else None,
    )
    character.references[key] = reference
    _mirror_primary_reference(manifest, char_id, character, key, reference)
    return SeedReferenceResult(
        reference=reference,
        sheet=sheet_path,
        panels=panel_paths,
        candidates=candidates,
        mode="generate",
        layout=effective_layout,
    )
