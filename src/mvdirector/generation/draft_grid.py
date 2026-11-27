# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Per-section storyboard contact sheets (2x2 / 3x2) for rapid draft continuity & R2V montages."""

from __future__ import annotations

import hashlib
import logging
import os
import re
import tempfile
import unicodedata
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from PIL import Image, ImageOps

from mvdirector import media
from mvdirector.comfy.client import ComfyClient
from mvdirector.comfy.workflow import WorkflowTemplate
from mvdirector.errors import ConfigError, MediaToolError
from mvdirector.generation.candidates import (
    atomic_save_png,
    load_rgb_candidate,
)
from mvdirector.generation.plates import plate_slug
from mvdirector.models import (
    AssetKind,
    AssetStatus,
    Attempt,
    DraftGridSheet,
    Manifest,
    QCReport,
    RetryStrategy,
    Review,
    Shot,
    ShotType,
    ascii_fold,
)
from mvdirector.planning import prompts
from mvdirector.planning.segmenter import frames_for
from mvdirector.project import Project
from mvdirector.qc.factory import QualityGates

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class DraftGridResult:
    sheet_id: str
    section: str | None
    layout: Literal["2x2", "3x2"]
    sheet_path: Path
    shot_ids: list[str]
    panel_paths: dict[str, Path]
    video_path: Path | None = None


@dataclass(frozen=True)
class _PlannedChunk:
    section: str | None
    shots: list[Shot]
    global_chunk_idx: int
    is_full_chunk: bool


def _norm_section(name: str | None) -> str:
    return unicodedata.normalize("NFKC", name or "").strip().casefold()


def _sanitize_id_fragment(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", ascii_fold(value)).strip("_-")
    return cleaned or "x"


def _build_sheet_id(
    global_chunk_idx: int,
    sec_label: str | None,
    chunk_shots: Sequence[Shot],
    *,
    is_full_chunk: bool,
) -> str:
    slug = plate_slug(sec_label or f"section_{global_chunk_idx + 1}")
    base = f"sec_{global_chunk_idx + 1:02d}_{slug}"
    if is_full_chunk:
        return base[:64].rstrip("_-")
    first_id = _sanitize_id_fragment(chunk_shots[0].id)
    last_id = _sanitize_id_fragment(chunk_shots[-1].id)
    span = first_id if len(chunk_shots) == 1 else f"{first_id}_{last_id}"
    candidate = f"{base}_{span}"
    if len(candidate) <= 64:
        return candidate
    digest = hashlib.sha256(",".join(s.id for s in chunk_shots).encode("utf-8")).hexdigest()[:8]
    max_base = 64 - 1 - len(digest)
    return f"{base[:max_base].rstrip('_-')}_{digest}"


def _contiguous_runs(
    indexed_shots: Sequence[tuple[int, Shot]],
) -> list[tuple[str | None, list[tuple[int, Shot]]]]:
    runs: list[tuple[str | None, list[tuple[int, Shot]]]] = []
    current_key: str | None = None
    current_label: str | None = None
    current_items: list[tuple[int, Shot]] = []
    prev_idx: int | None = None

    for idx, shot in indexed_shots:
        key = _norm_section(shot.section) if shot.section else f"__shot_{shot.id}"
        if current_items and (key != current_key or prev_idx is None or idx != prev_idx + 1):
            runs.append((current_label, current_items))
            current_items = []
        if not current_items:
            current_key = key
            current_label = shot.section
        current_items.append((idx, shot))
        prev_idx = idx
    if current_items:
        runs.append((current_label, current_items))
    return runs


def _plan_section_chunks(
    manifest: Manifest,
    *,
    section: str | None = None,
    shot_ids: Sequence[str] | None = None,
    max_panels: int = 6,
) -> list[_PlannedChunk]:
    """Compute contiguous timeline chunks preserving global run indices and gap boundaries."""
    if max_panels < 1:
        raise ConfigError(f"max_panels must be >= 1, got {max_panels}.")

    timeline = list(enumerate(manifest.shots))
    timeline_to_global: dict[int, tuple[int, list[int]]] = {}
    global_chunk_idx = 0
    for _label, group in _contiguous_runs(timeline):
        for offset in range(0, len(group), max_panels):
            chunk_items = group[offset : offset + max_panels]
            chunk_indices = [idx for idx, _ in chunk_items]
            for idx in chunk_indices:
                timeline_to_global[idx] = (global_chunk_idx, chunk_indices)
            global_chunk_idx += 1

    filtered = timeline
    if shot_ids is not None:
        wanted_ids = set(shot_ids)
        filtered = [(idx, s) for idx, s in filtered if s.id in wanted_ids]
    if section is not None:
        wanted_sec = _norm_section(section)
        exact = [(idx, s) for idx, s in filtered if _norm_section(s.section) == wanted_sec]
        if exact:
            filtered = exact
        else:
            wanted_base = wanted_sec.split(":", 1)[0].strip()
            filtered = [
                (idx, s)
                for idx, s in filtered
                if _norm_section(s.section).split(":", 1)[0].strip() == wanted_base
            ]
    if not filtered:
        raise ConfigError("No matching shots found for draft_grid contact sheet generation.")

    planned: list[_PlannedChunk] = []
    for label, group in _contiguous_runs(filtered):
        for offset in range(0, len(group), max_panels):
            chunk_items = group[offset : offset + max_panels]
            chunk_indices = [idx for idx, _ in chunk_items]
            g_idx, g_indices = timeline_to_global[chunk_indices[0]]
            planned.append(
                _PlannedChunk(
                    section=label,
                    shots=[shot for _, shot in chunk_items],
                    global_chunk_idx=g_idx,
                    is_full_chunk=(chunk_indices == g_indices),
                )
            )
    return planned


def group_section_shots(
    manifest: Manifest,
    *,
    section: str | None = None,
    shot_ids: Sequence[str] | None = None,
    max_panels: int = 6,
) -> list[tuple[str | None, list[Shot]]]:
    """Group contiguous timeline shots by section and chunk into contact-sheet pages."""
    return [
        (chunk.section, chunk.shots)
        for chunk in _plan_section_chunks(
            manifest, section=section, shot_ids=shot_ids, max_panels=max_panels
        )
    ]


def grid_render_size(
    project: Project, layout: Literal["2x2", "3x2"], *, draft: bool = True
) -> tuple[int, int]:
    """Compute aspect-safe contact-sheet dimensions so sliced cells pass ImageQC."""
    video = project.config.video
    multiple = max(2, video.size_multiple)
    if layout == "2x2":
        return video.generation_size(draft=False)
    cell_w, cell_h = video.generation_size(draft=draft)
    raw_w, raw_h = 3 * cell_w, 2 * cell_h
    if raw_w > 2304:
        # Halving full-res 3x2 (e.g. 3840x1440 -> 1920x720) preserves exact 16:9 cell aspect
        # and keeps each panel >= 0.5x target height (within ImageQC.min_resolution_scale).
        raw_w = max(multiple, (raw_w // 2 // multiple) * multiple)
        raw_h = max(multiple, (raw_h // 2 // multiple) * multiple)
    return raw_w, raw_h


def slice_contact_sheet(
    sheet: Image.Image,
    layout: Literal["2x2", "3x2"],
    shot_count: int,
    inset: float,
    *,
    target_size: tuple[int, int] | None = None,
) -> list[Image.Image]:
    """Slice ``sheet`` row-major into ``shot_count`` panels with divider-inset trimming."""
    cols, rows = (2, 2) if layout == "2x2" else (3, 2)
    capacity = cols * rows
    if shot_count < 1 or shot_count > capacity:
        raise ConfigError(
            f"Layout '{layout}' holds 1..{capacity} panels, got shot_count={shot_count}."
        )
    if isinstance(inset, bool) or not 0.0 <= inset < 0.2:
        raise ConfigError(f"Panel inset must be in [0.0, 0.2), got {inset!r}.")

    width, height = sheet.size
    panels: list[Image.Image] = []
    for idx in range(shot_count):
        col = idx % cols
        row = idx // cols
        left = col * width // cols
        right = (col + 1) * width // cols
        top = row * height // rows
        bottom = (row + 1) * height // rows
        dx, dy = int((right - left) * inset), int((bottom - top) * inset)
        if right - dx <= left + dx or bottom - dy <= top + dy:
            raise ConfigError(
                f"Contact sheet ({width}x{height}) is too small for layout '{layout}' "
                f"with inset={inset}."
            )
        crop = sheet.crop((left + dx, top + dy, right - dx, bottom - dy))
        if target_size is not None:
            tw, th = target_size
            crop_aspect = crop.width / crop.height
            target_aspect = tw / th
            if abs(crop_aspect - target_aspect) / target_aspect > 0.03:
                crop = ImageOps.fit(crop, (tw, th), method=Image.Resampling.LANCZOS)
            elif crop.size != (tw, th):
                crop = crop.resize((tw, th), Image.Resampling.LANCZOS)
        panels.append(crop)
    return panels


def _has_distinct_target(
    template: WorkflowTemplate, param: str, *, excluded_targets: set[tuple[str, str]]
) -> set[tuple[str, str]] | None:
    if not template.accepts(param):
        return None
    targets = {(t.node, t.field) for t in template.bindings.inputs.get(param, ())}
    if not targets or not targets.isdisjoint(excluded_targets):
        return None
    return targets


def _majority_set_plate(project: Project, manifest: Manifest, shots: Sequence[Shot]) -> Path | None:
    counts: Counter[str] = Counter(s.setting for s in shots if s.setting)
    for setting, _ in counts.most_common():
        scene = manifest.visual_bible.find_set(setting)
        if scene is None:
            continue
        rel = scene.panel("wide")
        if rel:
            path = project.path(rel)
            if path.is_file():
                return path
    return None


def _collect_section_refs(
    project: Project,
    manifest: Manifest,
    shots: Sequence[Shot],
    template: WorkflowTemplate,
    *,
    exclude_param: str | None = None,
) -> tuple[list[tuple[str, str, Path]], Path | None]:
    excluded: set[tuple[str, str]] = set()
    if exclude_param is not None:
        excluded = {(t.node, t.field) for t in template.bindings.inputs.get(exclude_param, ())}

    chars: list[tuple[str | None, Any, str | None]] = []
    seen_char_keys: set[tuple[str | None, str | None]] = set()
    for shot in shots:
        if not shot.type.shows_singer:
            continue
        for cid, char, state_id in manifest.visible_characters(shot):
            ck = (cid, state_id)
            if ck not in seen_char_keys:
                seen_char_keys.add(ck)
                chars.append((cid, char, state_id))

    refs: list[tuple[str, str, Path]] = []
    slot_names = ["reference_image", *(f"reference_image_{n}" for n in range(2, 11))]
    slot_idx = 0
    for _cid, char, state_id in chars:
        ref_set = char.reference_for(state_id)
        if ref_set is None:
            continue
        rel = ref_set.panel("front") or ref_set.sheet
        if not rel:
            continue
        path = project.path(rel)
        if not path.is_file():
            continue
        while slot_idx < len(slot_names):
            slot = slot_names[slot_idx]
            slot_idx += 1
            targets = _has_distinct_target(template, slot, excluded_targets=excluded)
            if targets is not None:
                excluded.update(targets)
                refs.append((slot, slot, path))
                break

    plate = _majority_set_plate(project, manifest, shots)
    if plate is not None:
        targets = _has_distinct_target(template, "set_image", excluded_targets=excluded)
        if targets is not None:
            excluded.update(targets)
            refs.append(("set_image", "set_image", plate))
    return refs, plate


def select_r2v_template(
    templates: dict[str, WorkflowTemplate], shots: Sequence[Shot]
) -> WorkflowTemplate:
    has_lipsync = any(s.type is ShotType.LIPSYNC for s in shots)
    if "r2v" in templates:
        return templates["r2v"]
    lipsync = templates.get("lipsync")
    if (
        has_lipsync
        and lipsync is not None
        and (
            lipsync.accepts("draft_grid_image")
            or lipsync.accepts("reference_image")
            or lipsync.accepts("set_image")
        )
    ):
        return lipsync
    if "i2v" in templates:
        return templates["i2v"]
    if lipsync is not None:
        return lipsync
    raise ConfigError("Configure an 'r2v', 'lipsync', or 'i2v' workflow for draft_grid mode='r2v'.")


def _slice_video_clip(src: Path, dst: Path, start: float, duration: float) -> Path:
    """Slice [start, start+duration] seconds from ``src`` into ``dst`` atomically."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{dst.stem}.", suffix=".mp4", dir=dst.parent)
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        media.run(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-protocol_whitelist",
                "file",
                "-ss",
                f"{max(0.0, start):.4f}",
                "-t",
                f"{max(0.05, duration):.4f}",
                "-i",
                str(src.resolve()),
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "16",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(tmp_path.resolve()),
            ]
        )
        os.replace(tmp_path, dst)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    return dst


def generate_section_sheet(
    project: Project,
    manifest: Manifest,
    comfy: ComfyClient,
    templates: dict[str, WorkflowTemplate],
    *,
    section: str | None = None,
    shot_ids: Sequence[str] | None = None,
    layout: Literal["auto", "2x2", "3x2"] = "auto",
    mode: Literal["sheet", "stills", "r2v", "both"] = "stills",
    draft: bool = True,
    overwrite: bool = False,
    apply_to_shots: bool = True,
    gates: QualityGates | None = None,
) -> list[DraftGridResult]:
    """Generate per-section 2x2 or 3x2 storyboard contact sheets and slice stills / R2V clips."""
    if layout not in ("auto", "2x2", "3x2"):
        raise ConfigError(f"Unknown draft_grid layout {layout!r}; use 'auto', '2x2', or '3x2'.")
    if mode not in ("sheet", "stills", "r2v", "both"):
        raise ConfigError(
            f"Unknown draft_grid mode {mode!r}; use 'sheet', 'stills', 'r2v', or 'both'."
        )

    max_panels = 4 if layout == "2x2" else 6
    chunks = _plan_section_chunks(
        manifest, section=section, shot_ids=shot_ids, max_panels=max_panels
    )
    if mode in ("r2v", "both"):
        r2v_template = select_r2v_template(templates, [s for c in chunks for s in c.shots])
    else:
        r2v_template = None

    bible = manifest.visual_bible
    video_cfg = project.config.video
    notation = project.config.comfy.reference_notation
    max_chars = project.config.comfy.max_prompt_chars
    inset = project.config.reference.panel_inset
    cell_size = video_cfg.generation_size(draft=draft)
    negative_prompt = prompts.merge_negative_prompt(bible.negative_prompt, bible.exclusions)
    project.draft_grids_dir.mkdir(parents=True, exist_ok=True)

    beat_offsets_all: list[float] = []
    try:
        beat_offsets_all = list(project.load_analysis().beats)
    except Exception:  # noqa: BLE001
        beat_offsets_all = []

    results: list[DraftGridResult] = []
    for chunk in chunks:
        sec_label = chunk.section
        chunk_shots = chunk.shots
        chunk_idx = chunk.global_chunk_idx
        eff_layout: Literal["2x2", "3x2"] = (
            "2x2" if (layout == "2x2" or (layout == "auto" and len(chunk_shots) <= 4)) else "3x2"
        )
        sheet_id = _build_sheet_id(
            chunk_idx, sec_label, chunk_shots, is_full_chunk=chunk.is_full_chunk
        )
        sheet_path = project.draft_grid_path(sheet_id, ".png")
        rel_sheet = project.rel(sheet_path)

        if overwrite or not sheet_path.is_file():
            still_tpl = templates.get("still")
            t2i_tpl = templates.get("t2i")
            if still_tpl is not None and still_tpl.accepts("reference_image"):
                sheet_tpl = still_tpl
            elif t2i_tpl is not None:
                sheet_tpl = t2i_tpl
            elif still_tpl is not None:
                sheet_tpl = still_tpl
            else:
                raise ConfigError("A 'still' or 't2i' workflow is required for draft_grid.")

            refs, plate_path = _collect_section_refs(project, manifest, chunk_shots, sheet_tpl)
            if not refs and t2i_tpl is not None and sheet_tpl is not t2i_tpl:
                sheet_tpl = t2i_tpl
                refs, plate_path = _collect_section_refs(project, manifest, chunk_shots, sheet_tpl)

            bound_roles = [role for _, role, _ in refs] if refs else None
            prompt = prompts.draft_grid_prompt(
                manifest,
                chunk_shots,
                eff_layout,
                reference_notation=notation,
                bound_references=bound_roles,
                plate_attached=plate_path is not None,
                pin_character_details=project.config.planning.pin_character_details,
            )
            if max_chars is not None:
                prompt = prompts.fit_prompt_budget(prompt, max_chars)

            sheet_w, sheet_h = grid_render_size(project, eff_layout, draft=draft)
            params: dict[str, Any] = {
                "positive_prompt": prompt,
                "negative_prompt": negative_prompt,
                "seed": project.config.seed + 5000 + chunk_idx * 100,
                "width": sheet_w,
                "height": sheet_h,
            }
            for param, _, path in refs:
                params[param] = comfy.upload(path)
            comfy.run(sheet_tpl.render(params), sheet_tpl.bindings.output, sheet_path)

        sheet_img = load_rgb_candidate(sheet_path, f"Draft grid '{sheet_id}'")
        panel_paths: dict[str, Path] = {}
        panel_rel_map: dict[str, str] = {}

        if mode in ("stills", "both"):
            sliced = slice_contact_sheet(
                sheet_img, eff_layout, len(chunk_shots), inset, target_size=cell_size
            )
            for idx, (shot, panel_img) in enumerate(zip(chunk_shots, sliced)):
                number = len(shot.still.attempts) + 1
                dst = project.asset_path(shot, AssetKind.STILL, number)
                atomic_save_png(panel_img, dst)
                rel_panel = project.rel(dst)
                panel_paths[shot.id] = dst
                panel_rel_map[shot.id] = rel_panel
                if gates is not None:
                    qc_report = gates.image.evaluate(
                        dst,
                        shot,
                        manifest,
                        None,
                        expected_size=cell_size if draft else None,
                    )
                else:
                    qc_report = QCReport(passed=True, score=1.0, checks=[])
                attempt = Attempt(
                    number=number,
                    strategy=RetryStrategy.INITIAL,
                    seed=project.config.seed + 5000 + chunk_idx * 100,
                    prompt=shot.still_prompt,
                    params={"draft_grid_sheet": rel_sheet, "draft_grid_panel": idx + 1},
                    path=rel_panel,
                    qc=qc_report,
                )
                if apply_to_shots:
                    shot.still.attempts.append(attempt)
                    shot.still.accepted_by_reviewer = False
                    shot.draft_grid_sheet = rel_sheet
                    shot.review = Review()  # an earlier review no longer describes this shot
                    if qc_report.passed:
                        shot.still.status = AssetStatus.PASSED
                        shot.still.selected_attempt = number
                    else:
                        shot.still.status = AssetStatus.PENDING
                    if shot.video.selected_attempt is not None:
                        shot.video.reset(project.config.attempt_budget(shot.start))

        video_path: Path | None = None
        if mode in ("r2v", "both"):
            assert r2v_template is not None
            video_path = project.draft_grid_path(sheet_id, "_r2v.mp4")
            panel1_path = panel_paths.get(chunk_shots[0].id)
            if panel1_path is None:
                p1_img = slice_contact_sheet(
                    sheet_img, eff_layout, 1, inset, target_size=cell_size
                )[0]
                panel1_path = project.draft_grid_path(sheet_id, "_panel1.png")
                atomic_save_png(p1_img, panel1_path)

            chunk_start = min(s.audio_slice.start for s in chunk_shots)
            chunk_end = max(s.audio_slice.end for s in chunk_shots)
            total_dur = max(0.2, chunk_end - chunk_start)
            num_frames = frames_for(
                total_dur, video_cfg.fps, video_cfg.frame_multiple, video_cfg.frame_offset
            )

            video_refs, _ = _collect_section_refs(
                project, manifest, chunk_shots, r2v_template, exclude_param="image"
            )
            used_targets: set[tuple[str, str]] = {
                (t.node, t.field)
                for p, _, _ in video_refs
                for t in r2v_template.bindings.inputs.get(p, ())
            }
            if r2v_template.accepts("image"):
                used_targets.update(
                    (t.node, t.field) for t in r2v_template.bindings.inputs.get("image", ())
                )
            if _has_distinct_target(
                r2v_template, "draft_grid_image", excluded_targets=used_targets
            ):
                video_refs.append(("draft_grid_image", "draft_grid_image", sheet_path))
            else:
                for n in range(2, 11):
                    slot = f"reference_image_{n}"
                    if _has_distinct_target(r2v_template, slot, excluded_targets=used_targets):
                        video_refs.append((slot, "draft_grid_image", sheet_path))
                        break
                else:
                    if _has_distinct_target(
                        r2v_template, "reference_image", excluded_targets=used_targets
                    ):
                        video_refs.append(("reference_image", "draft_grid_image", sheet_path))

            rel_beats = [
                round(b - chunk_start, 4) for b in beat_offsets_all if chunk_start <= b <= chunk_end
            ]
            has_lipsync = any(s.type is ShotType.LIPSYNC for s in chunk_shots)
            uses_audio_video = (
                has_lipsync and r2v_template.accepts("audio_video")
            ) or "audio_video" in r2v_template.bindings.required

            motion_prompt = prompts.draft_grid_motion_prompt(
                manifest,
                chunk_shots,
                total_dur,
                beat_offsets=rel_beats,
                video_config=video_cfg,
                reference_notation=notation,
                has_audio_video=uses_audio_video,
                bound_references=[role for _, role, _ in video_refs] or None,
            )
            if max_chars is not None:
                motion_prompt = prompts.fit_prompt_budget(motion_prompt, max_chars)

            v_params: dict[str, Any] = {
                "positive_prompt": motion_prompt,
                "negative_prompt": negative_prompt,
                "seed": project.config.seed + 5500 + chunk_idx * 100,
                "width": cell_size[0],
                "height": cell_size[1],
                "num_frames": num_frames,
                "duration": round(total_dur, 3),
                "fps": video_cfg.fps,
            }
            if r2v_template.accepts("image"):
                v_params["image"] = comfy.upload(panel1_path)
            for param, _, ref_path in video_refs:
                v_params[param] = comfy.upload(ref_path)

            if has_lipsync or "audio" in r2v_template.bindings.required or uses_audio_video:
                audio_source = (
                    project.input_path(project.config.inputs.vocals, "vocals")
                    if (has_lipsync and project.config.inputs.vocals)
                    else project.input_path(project.config.inputs.mix, "mix")
                )
                audio_dst = project.draft_grid_path(sheet_id, "_audio.wav")
                media.prepare_audio_slice(
                    audio_source,
                    audio_dst,
                    chunk_start,
                    chunk_end,
                    target_duration=total_dur,
                )
                if has_lipsync or "audio" in r2v_template.bindings.required:
                    v_params["audio"] = comfy.upload(audio_dst)
                if uses_audio_video:
                    av_dst = project.draft_grid_path(sheet_id, "_audio.mp4")
                    media.audio_to_black_video(audio_dst, av_dst, fps=video_cfg.fps)
                    v_params["audio_video"] = comfy.upload(av_dst)

            comfy.run(r2v_template.render(v_params), r2v_template.bindings.output, video_path)
            rel_video = project.rel(video_path)

            if apply_to_shots:
                try:
                    probed_dur = media.probe(video_path).duration
                except MediaToolError:
                    probed_dur = 0.0
                for idx, shot in enumerate(chunk_shots):
                    shot.allow_cuts = True
                    shot.draft_grid_sheet = rel_sheet
                    raw_ss = max(0.0, shot.audio_slice.start - chunk_start)
                    raw_dur = max(0.05, shot.audio_slice.end - shot.audio_slice.start)
                    if probed_dur > 0.1:
                        ss = min(raw_ss, max(0.0, probed_dur - 0.05))
                        dur = min(raw_dur, max(0.05, probed_dur - ss))
                    else:
                        ss, dur = raw_ss, raw_dur
                    next_num = len(shot.video.attempts) + 1
                    clip_dst = project.asset_path(shot, AssetKind.VIDEO, next_num)
                    _slice_video_clip(video_path, clip_dst, ss, dur)
                    rel_clip = project.rel(clip_dst)
                    shot.video.attempts.append(
                        Attempt(
                            number=next_num,
                            strategy=RetryStrategy.INITIAL,
                            seed=project.config.seed + 5500 + chunk_idx * 100,
                            prompt=motion_prompt,
                            params={
                                "draft_grid_sheet": rel_sheet,
                                "draft_grid_video": rel_video,
                                "draft_grid_panel": idx + 1,
                            },
                            path=rel_clip,
                            qc=QCReport(passed=True, score=1.0, checks=[]),
                        )
                    )
                    shot.video.status = AssetStatus.PASSED
                    shot.video.selected_attempt = next_num
                    shot.video.accepted_by_reviewer = False
                    shot.review = Review()  # an earlier review no longer describes this shot

        sheet_record = DraftGridSheet(
            id=sheet_id,
            section=sec_label,
            layout=eff_layout,
            sheet_path=rel_sheet,
            shot_ids=[s.id for s in chunk_shots],
            panel_paths=panel_rel_map,
            video_path=project.rel(video_path) if video_path is not None else None,
        )
        manifest.draft_grids = [g for g in manifest.draft_grids if g.id != sheet_id]
        manifest.draft_grids.append(sheet_record)
        results.append(
            DraftGridResult(
                sheet_id=sheet_id,
                section=sec_label,
                layout=eff_layout,
                sheet_path=sheet_path,
                shot_ids=[s.id for s in chunk_shots],
                panel_paths=panel_paths,
                video_path=video_path,
            )
        )

    return results
