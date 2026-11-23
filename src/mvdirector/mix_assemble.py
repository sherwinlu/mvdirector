# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Chunked assembly, circular loop synthesis, HUD rendering, and caching for long-form mixes.

Each track chapter is rendered independently into ``assets/segments/<chapter_id>.mp4`` alongside a
JSON input stamp. Unchanged chapters are reused via stream-copy concatenation so regenerating a
single track splices into a 60-to-90+ minute video in seconds without re-encoding other chapters.
"""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from mvdirector import media
from mvdirector.assemble import NotReadyError, _resolve_output_size
from mvdirector.audio.tracklist import (
    format_chapter_label,
    format_ffmetadata_chapters,
    format_youtube_chapters,
)
from mvdirector.config import MixConfig, MixHudConfig
from mvdirector.errors import ConfigError, MVDError
from mvdirector.models import Manifest, MixChapter
from mvdirector.planning.mix_planner import quantize_loop_clip_seconds
from mvdirector.project import Project, require_safe_id
from mvdirector.review.policy import is_ready, refresh_reviews, uses_stills_only
from mvdirector.store import atomic_write

log = logging.getLogger(__name__)

_FONT_CANDIDATES = (
    "/System/Library/Fonts/SFNS.ttf",
    "/System/Library/Fonts/HelveticaNeue.ttc",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
)


@dataclass(frozen=True)
class LoopUnitSource:
    shot_id: str
    path: Path
    is_video: bool
    head_handle: float


@dataclass(frozen=True)
class HudLayout:
    card_path: Path
    card_x: int
    card_y: int
    spec_w: int
    spec_h: int
    spec_x: int
    spec_y: int


def _even(value: float | int, minimum: int = 2) -> int:
    rounded = max(minimum, int(round(value)))
    return rounded - (rounded % 2)


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in _FONT_CANDIDATES:
        if Path(candidate).is_file():
            try:
                return ImageFont.truetype(candidate, size=size)
            except OSError:
                continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _truncate_text(draw: ImageDraw.ImageDraw, text: str, font: Any, max_px: int) -> str:
    if max_px <= 0 or not text:
        return ""
    bbox = draw.textbbox((0, 0), text, font=font)
    if bbox[2] - bbox[0] <= max_px:
        return text
    ellipsis = "..."
    lo, hi = 0, len(text)
    best = ellipsis
    while lo <= hi:
        mid = (lo + hi) // 2
        cand = text[:mid].rstrip() + ellipsis
        cb = draw.textbbox((0, 0), cand, font=font)
        if cb[2] - cb[0] <= max_px:
            best = cand
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def render_hud_card(
    dst: Path,
    chapter: MixChapter,
    hud: MixHudConfig,
    frame_w: int,
    frame_h: int,
) -> HudLayout | None:
    """Render a compact anti-aliased RGBA HUD pill card with Pillow (no ffmpeg drawtext needed)."""
    if not hud.enabled:
        return None
    has_title = hud.show_title and bool(chapter.title.strip())
    watermark = " ".join(hud.watermark.split())
    has_watermark = bool(watermark)
    has_spec = hud.show_spectrum
    if not (has_title or has_watermark or has_spec):
        return None

    scale = max(0.45, min(2.0, frame_h / 720.0))
    pad_x = max(6, min(round(16 * scale), max(6, frame_w // 8)))
    pad_y = max(4, min(round(12 * scale), max(4, frame_h // 8)))
    gap_y = max(2, min(round(6 * scale), max(2, frame_h // 16)))
    margin_x = min(max(12, round(frame_w * 0.028)), max(0, (frame_w - 64) // 2))
    margin_y = min(max(12, round(frame_h * 0.04)), max(0, (frame_h - 32) // 2))

    max_card_w = _even(max(32, frame_w - 2 * margin_x), minimum=min(32, _even(frame_w, minimum=2)))
    card_w = _even(min(max_card_w, max(160, round(frame_w * 0.36))), minimum=min(64, max_card_w))
    card_w = min(card_w, _even(frame_w, minimum=2))
    inner_w = max(16, card_w - 2 * pad_x)

    meta_font = _load_font(max(9, round(12 * scale)))
    title_font = _load_font(max(11, round(17 * scale)))
    probe_img = Image.new("RGBA", (16, 16), (0, 0, 0, 0))
    probe_draw = ImageDraw.Draw(probe_img)

    y_cursor = pad_y
    meta_line = ""
    if has_watermark:
        meta_line = _truncate_text(probe_draw, watermark.upper(), meta_font, inner_w)
    elif has_title:
        meta_line = f"CHAPTER {chapter.index + 1:02d}"

    meta_h = 0
    if meta_line:
        mb = probe_draw.textbbox((0, 0), meta_line, font=meta_font)
        meta_h = max(10, round(mb[3] - mb[1]))
        y_cursor += meta_h + gap_y

    title_line = ""
    title_h = 0
    if has_title:
        label = format_chapter_label(chapter.title, chapter.artist)
        title_line = _truncate_text(probe_draw, label, title_font, inner_w)
        tb = probe_draw.textbbox((0, 0), title_line, font=title_font)
        title_h = max(12, round(tb[3] - tb[1]))
        y_cursor += title_h + (gap_y if has_spec else 0)

    spec_w = _even(inner_w, minimum=16) if has_spec else 0
    spec_h = _even(max(18, round(28 * scale)), minimum=16) if has_spec else 0
    spec_rel_y = y_cursor
    max_card_h = _even(max(24, frame_h - 2 * margin_y), minimum=min(24, _even(frame_h, minimum=2)))
    max_card_h = min(max_card_h, _even(frame_h, minimum=2))
    if has_spec and spec_rel_y + spec_h + pad_y > max_card_h:
        # Frame too small to fit text + spectrum bar; drop the spectrum bar so the card fits.
        has_spec = False
        spec_w = 0
        spec_h = 0
    if has_spec:
        y_cursor += spec_h

    card_h = min(_even(y_cursor + pad_y, minimum=min(24, max_card_h)), max_card_h)
    card = Image.new("RGBA", (card_w, card_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(card)
    radius = min(max(4, round(10 * scale)), card_w // 2, card_h // 2)
    draw.rounded_rectangle(
        (0, 0, card_w - 1, card_h - 1),
        radius=radius,
        fill=(10, 12, 18, 168),
        outline=(255, 255, 255, 44),
        width=1,
    )

    text_y = pad_y
    if meta_line:
        draw.text((pad_x, text_y), meta_line, font=meta_font, fill=(185, 198, 220, 215))
        text_y += meta_h + gap_y
    if title_line:
        draw.text((pad_x, text_y), title_line, font=title_font, fill=(248, 250, 255, 245))

    dst.parent.mkdir(parents=True, exist_ok=True)
    card.save(dst, format="PNG")

    is_right = hud.position.endswith("right")
    is_top = hud.position.startswith("top")
    card_x = (
        max(0, frame_w - card_w - margin_x) if is_right else max(0, min(margin_x, frame_w - card_w))
    )
    card_y = (
        max(0, min(margin_y, frame_h - card_h)) if is_top else max(0, frame_h - card_h - margin_y)
    )
    return HudLayout(
        card_path=dst,
        card_x=card_x,
        card_y=card_y,
        spec_w=spec_w,
        spec_h=spec_h,
        spec_x=card_x + pad_x,
        spec_y=card_y + spec_rel_y,
    )


def chapter_frame_counts(chapters: Sequence[MixChapter], fps: int) -> list[int]:
    """Frames per chapter so cumulative cuts stay frame-locked across 90+ minutes."""
    return [round(ch.end * fps) - round(ch.start * fps) for ch in chapters]


def _read_stamp(marker: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(marker.read_text("utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _resolve_chapter_sources(
    project: Project,
    manifest: Manifest,
    chapter: MixChapter,
    *,
    draft: bool,
) -> list[LoopUnitSource]:
    stills_only = uses_stills_only(manifest, project.config)
    sources: list[LoopUnitSource] = []
    for shot_id in chapter.shot_ids:
        try:
            shot = manifest.shot(shot_id)
        except KeyError as exc:
            raise ConfigError(
                f"Chapter '{chapter.id}' references unknown shot '{shot_id}'."
            ) from exc
        if not stills_only:
            video = shot.video.selected or shot.video.best_attempt()
            if video is not None and video.path:
                rel = video.path
                if (
                    not draft
                    and shot.video.upscaled_path
                    and shot.video.upscaled_from_attempt == video.number
                    and project.path(shot.video.upscaled_path).is_file()
                ):
                    rel = shot.video.upscaled_path
                path = project.path(rel)
                if path.is_file():
                    sources.append(
                        LoopUnitSource(shot_id, path, True, shot.audio_slice.head_handle)
                    )
                    continue
        still = shot.still.selected or shot.still.best_attempt()
        if still is not None and still.path:
            path = project.path(still.path)
            if path.is_file():
                sources.append(LoopUnitSource(shot_id, path, False, 0.0))
    return sources


def _build_chapter_stamp(
    project: Project,
    chapter: MixChapter,
    frames: int,
    sources: Sequence[LoopUnitSource],
    mix_path: Path,
    *,
    width: int,
    height: int,
    fps: int,
    draft: bool,
) -> dict[str, Any]:
    mix_cfg: MixConfig = project.config.mix
    hud_cfg = mix_cfg.hud
    source_entries: list[dict[str, Any]] = []
    for src in sources:
        st = src.path.stat()
        source_entries.append(
            {
                "shot_id": src.shot_id,
                "rel": project.rel(src.path),
                "is_video": src.is_video,
                "head_handle": round(src.head_handle, 4),
                "size": st.st_size,
                "mtime_ns": st.st_mtime_ns,
            }
        )
    stamp: dict[str, Any] = {
        "chapter_id": chapter.id,
        "index": chapter.index,
        "start": round(chapter.start, 4),
        "end": round(chapter.end, 4),
        "frames": frames,
        "bpm": round(chapter.bpm, 2),
        "downbeat_offset": round(chapter.downbeat_offset, 4),
        "energy": round(chapter.energy, 4),
        "width": width,
        "height": height,
        "fps": fps,
        "draft": draft,
        "visual_source": mix_cfg.visual_source,
        "clip_seconds": round(mix_cfg.clip_seconds, 4),
        "xfade_seconds": round(mix_cfg.xfade_seconds, 4),
        "loop_blend": mix_cfg.loop_blend,
        "chapter_fade_seconds": round(mix_cfg.chapter_fade_seconds, 4),
        "beat_pulse": round(mix_cfg.beat_pulse, 4),
        "hud": hud_cfg.model_dump(mode="json"),
        "title": chapter.title if (hud_cfg.enabled and hud_cfg.show_title) else "",
        "artist": (chapter.artist or "") if (hud_cfg.enabled and hud_cfg.show_title) else "",
        "sources": source_entries,
    }
    if hud_cfg.enabled and hud_cfg.show_spectrum:
        mix_stat = mix_path.stat()
        stamp["mix_audio"] = {
            "name": mix_path.name,
            "size": mix_stat.st_size,
            "mtime_ns": mix_stat.st_mtime_ns,
        }
    return stamp


def cached_segment_count(
    project: Project,
    manifest: Manifest,
    *,
    draft: bool = False,
) -> int:
    """Count chapters whose cached segment MP4 exists and matches current inputs/settings."""
    if not manifest.chapters:
        return 0
    try:
        mix_path = project.input_path(project.config.inputs.mix, "mix")
    except MVDError:
        return 0
    out_w, out_h = _resolve_output_size(project, draft=draft, width=None, height=None)
    hits = 0
    for ch, frames in zip(manifest.chapters, chapter_frame_counts(manifest.chapters, manifest.fps)):
        if frames <= 0:
            continue
        seg_path = project.segment_video_path(ch.id, draft=draft)
        stamp_path = project.segment_stamp_path(ch.id, draft=draft)
        if not seg_path.is_file():
            continue
        sources = _resolve_chapter_sources(project, manifest, ch, draft=draft)
        expected = _build_chapter_stamp(
            project,
            ch,
            frames,
            sources,
            mix_path,
            width=out_w,
            height=out_h,
            fps=manifest.fps,
            draft=draft,
        )
        if _read_stamp(stamp_path) == expected:
            hits += 1
    return hits


def _conform_unit_clip(
    src: LoopUnitSource,
    dst: Path,
    *,
    variant_idx: int,
    unit_frames: int,
    width: int,
    height: int,
    fps: int,
) -> None:
    """Conform one video clip or hero still into an exact ``unit_frames`` MP4."""
    if src.is_video:
        fit = (
            f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1,format=yuv420p"
        )
        vf = f"fps={fps},tpad=stop_mode=clone:stop_duration=4,{fit}"
        cmd = [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-protocol_whitelist",
            "file",
            "-ss",
            f"{src.head_handle:.4f}",
            "-i",
            str(src.path.resolve()),
            "-vf",
            vf,
            "-frames:v",
            str(unit_frames),
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "14",
            "-pix_fmt",
            "yuv420p",
            "-r",
            str(fps),
            str(dst),
        ]
        media.run(cmd)
        return

    # Hero still: 2x oversampled canvas + per-frame drift crop -> subpixel-smooth camera motion.
    win_w, win_h = width * 2, height * 2
    canvas_w, canvas_h = _even(win_w * 1.08), _even(win_h * 1.08)
    dx = max(0, canvas_w - win_w)
    dy = max(0, canvas_h - win_h)
    denom = max(1, unit_frames - 1)
    mode = variant_idx % 4
    x_expr = f"{dx}*n/{denom}" if mode in (0, 2) else f"{dx}*(1-n/{denom})"
    y_expr = f"{dy}*n/{denom}" if mode in (0, 1) else f"{dy}*(1-n/{denom})"
    vf = (
        f"scale={canvas_w}:{canvas_h}:force_original_aspect_ratio=increase,"
        f"crop={canvas_w}:{canvas_h},"
        f"crop=w={win_w}:h={win_h}:x='{x_expr}':y='{y_expr}',"
        f"scale={width}:{height}:flags=bilinear,setsar=1,format=yuv420p"
    )
    cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-protocol_whitelist",
        "file",
        "-loop",
        "1",
        "-framerate",
        str(fps),
        "-i",
        str(src.path.resolve()),
        "-vf",
        vf,
        "-frames:v",
        str(unit_frames),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-crf",
        "14",
        "-pix_fmt",
        "yuv420p",
        "-r",
        str(fps),
        str(dst),
    ]
    media.run(cmd)


def loop_xfade_transition(loop_blend: str) -> str:
    """FFmpeg ``xfade`` transition options for a chapter's loop-unit boundaries.

    ``smoothstep`` is a Hermite S-curve (``3P^2 - 2P^3``) dissolve (inspired by
    atomtanstudio/sound-and-vision ``server/sv/video/loops.py``). In a custom ``xfade``
    expression ``P`` falls from 1 at the start of the transition to 0 at its end, so the
    outgoing input ``A`` takes ``smoothstep(P)`` and the incoming input ``B`` takes the rest.
    """
    if loop_blend == "smoothstep":
        return "transition=custom:expr='A*(3*P*P-2*P*P*P)+B*(1-(3*P*P-2*P*P*P))'"
    return "transition=fade"


def _build_seamless_loop(
    sources: Sequence[LoopUnitSource],
    loop_dst: Path,
    work_dir: Path,
    chapter: MixChapter,
    project: Project,
    *,
    width: int,
    height: int,
    fps: int,
) -> Path:
    """Synthesize a circularly crossfaded master loop MP4 from ``sources`` (<= K xfade nodes)."""
    mix_cfg = project.config.mix
    video_cfg = project.config.video
    target_clip = quantize_loop_clip_seconds(
        chapter.bpm,
        video_cfg.beats_per_bar,
        mix_cfg.clip_seconds,
        mix_cfg.xfade_seconds,
        video_cfg.max_clip_seconds,
    )

    # Clamp unit length to the shortest actual video clip so loops never freeze on tpad frames.
    avail_list: list[float] = []
    for src in sources:
        if src.is_video:
            avail = max(0.0, media.probe(src.path).duration - src.head_handle)
            if avail >= 0.5:
                avail_list.append(avail)
    effective_clip = min([target_clip, *avail_list]) if avail_list else target_clip

    unit_frames = max(6, round(effective_clip * fps))
    desired_xfade = max(1, round(mix_cfg.xfade_seconds * fps))
    xfade_frames = min(desired_xfade, max(1, unit_frames // 3))
    stride_frames = max(2, unit_frames - xfade_frames)
    unit_frames = stride_frames + xfade_frames

    unit_paths: list[Path] = []
    for idx, src in enumerate(sources):
        unit_path = work_dir / f"unit_{idx:02d}.mp4"
        _conform_unit_clip(
            src,
            unit_path,
            variant_idx=idx,
            unit_frames=unit_frames,
            width=width,
            height=height,
            fps=fps,
        )
        unit_paths.append(unit_path)

    k = len(unit_paths)
    last = k - 1
    xfade_sec = xfade_frames / fps

    filter_parts: list[str] = [
        f"[{last}:v]split=2[last_tail][last_head]",
        f"[last_tail]trim=start_frame={stride_frames}:end_frame={unit_frames},setpts=PTS-STARTPTS[s0]",
    ]
    for idx in range(k - 1):
        filter_parts.append(
            f"[{idx}:v]trim=start_frame=0:end_frame={unit_frames},setpts=PTS-STARTPTS[s{idx + 1}]"
        )
    filter_parts.append(
        f"[last_head]trim=start_frame=0:end_frame={stride_frames},setpts=PTS-STARTPTS[s{k}]"
    )

    prev_label = "s0"
    xfade_mode = loop_xfade_transition(mix_cfg.loop_blend)
    for step in range(1, k + 1):
        out_label = "vloop" if step == k else f"xf{step}"
        offset_sec = ((step - 1) * stride_frames) / fps
        filter_parts.append(
            f"[{prev_label}][s{step}]xfade={xfade_mode}:"
            f"duration={xfade_sec:.6f}:offset={offset_sec:.6f}[{out_label}]"
        )
        prev_label = out_label

    inputs_argv: list[str] = []
    for unit_path in unit_paths:
        inputs_argv.extend(["-protocol_whitelist", "file", "-i", str(unit_path.resolve())])

    media.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            *inputs_argv,
            "-filter_complex",
            ";".join(filter_parts),
            "-map",
            "[vloop]",
            "-frames:v",
            str(k * stride_frames),
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "15",
            "-pix_fmt",
            "yuv420p",
            "-r",
            str(fps),
            str(loop_dst),
        ]
    )
    return loop_dst


def _beat_pulse_filter(chapter: MixChapter, beats_per_bar: int, beat_pulse: float) -> str | None:
    """Delimiter-free ffmpeg ``eq=eval=frame`` expression synced to chapter BPM and downbeats."""
    if beat_pulse <= 0.0 or chapter.bpm <= 0.0:
        return None
    strength = round(beat_pulse * (0.5 + 0.5 * chapter.energy), 4)
    if strength <= 1e-4:
        return None
    omega_b = round(2.0 * math.pi * (chapter.bpm / 60.0), 5)
    omega_bar = round(omega_b / max(1, beats_per_bar), 5)
    phi = round(max(0.0, chapter.downbeat_offset - chapter.start), 4)

    beat_cos = f"(0.5+0.5*cos({omega_b}*(t-{phi})))"
    beat_p = f"({beat_cos}*{beat_cos}*{beat_cos}*{beat_cos})"
    bar_cos = f"(0.5+0.5*cos({omega_bar}*(t-{phi})))"
    bar_p = f"({bar_cos}*{bar_cos}*{bar_cos}*{bar_cos})"

    b_amp = round(0.026 * strength, 5)
    c_amp = round(0.050 * strength, 5)
    s_amp = round(0.070 * strength, 5)
    return (
        f"eq=eval=frame:"
        f"brightness='{b_amp}*{bar_p}':"
        f"contrast='1+{c_amp}*(0.4*{beat_p}+0.6*{bar_p})':"
        f"saturation='1+{s_amp}*{bar_p}'"
    )


def _render_chapter_segment(
    project: Project,
    chapter: MixChapter,
    frames: int,
    sources: Sequence[LoopUnitSource],
    mix_path: Path,
    seg_dst: Path,
    stamp_dst: Path,
    stamp: dict[str, Any],
    *,
    width: int,
    height: int,
    fps: int,
) -> None:
    """Render one chapter MP4 atomically with loop streaming, beat pulse, and HUD."""
    mix_cfg = project.config.mix
    chapter_dur = frames / fps
    seg_dst.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=f"mvd_seg_{chapter.id}_") as tmp:
        work = Path(tmp)
        inputs_argv: list[str] = []
        if sources:
            loop_mp4 = _build_seamless_loop(
                sources,
                work / "loop.mp4",
                work,
                chapter,
                project,
                width=width,
                height=height,
                fps=fps,
            )
            inputs_argv.extend(
                ["-protocol_whitelist", "file", "-stream_loop", "-1", "-i", str(loop_mp4.resolve())]
            )
        else:
            inputs_argv.extend(["-f", "lavfi", "-i", f"color=c=black:s={width}x{height}:r={fps}"])

        hud_layout = render_hud_card(
            work / "hud.png", chapter, mix_cfg.hud, frame_w=width, frame_h=height
        )
        next_input = 1
        hud_input_idx: int | None = None
        spec_input_idx: int | None = None

        if hud_layout is not None:
            hud_input_idx = next_input
            next_input += 1
            inputs_argv.extend(
                [
                    "-protocol_whitelist",
                    "file",
                    "-loop",
                    "1",
                    "-framerate",
                    str(fps),
                    "-i",
                    str(hud_layout.card_path.resolve()),
                ]
            )
            if hud_layout.spec_w > 0 and hud_layout.spec_h > 0:
                spec_input_idx = next_input
                next_input += 1
                inputs_argv.extend(
                    [
                        "-protocol_whitelist",
                        "file",
                        "-ss",
                        f"{chapter.start:.4f}",
                        "-t",
                        f"{chapter_dur:.4f}",
                        "-i",
                        str(mix_path.resolve()),
                    ]
                )

        base_chain = [f"fps={fps}", "setsar=1", "format=yuv420p"]
        if sources:
            pulse = _beat_pulse_filter(
                chapter, project.config.video.beats_per_bar, mix_cfg.beat_pulse
            )
            if pulse is not None:
                base_chain.append(pulse)

        graph_parts: list[str] = [f"[0:v]{','.join(base_chain)}[vbase]"]
        current_label = "vbase"

        if hud_layout is not None and hud_input_idx is not None:
            graph_parts.append(
                f"[{current_label}][{hud_input_idx}:v]overlay="
                f"x={hud_layout.card_x}:y={hud_layout.card_y}:format=auto[vhud]"
            )
            current_label = "vhud"
            if spec_input_idx is not None:
                graph_parts.append(
                    f"[{spec_input_idx}:a:0]apad=whole_dur={chapter_dur + 1.0:.3f},"
                    f"showfreqs=s={hud_layout.spec_w}x{hud_layout.spec_h}:rate={fps}:"
                    f"mode=bar:fscale=log:ascale=cbrt:averaging=3:cmode=combined:"
                    f"colors=white@0.72,format=rgba[vspec]"
                )
                graph_parts.append(
                    f"[{current_label}][vspec]overlay="
                    f"x={hud_layout.spec_x}:y={hud_layout.spec_y}:eof_action=pass:format=auto[vcomp]"
                )
                current_label = "vcomp"

        tail_filters: list[str] = []
        eff_fade = min(mix_cfg.chapter_fade_seconds, chapter_dur * 0.25)
        if eff_fade >= 0.04:
            fade_out_start = max(0.0, chapter_dur - eff_fade)
            tail_filters.append(f"fade=t=in:st=0:d={eff_fade:.4f}")
            tail_filters.append(f"fade=t=out:st={fade_out_start:.4f}:d={eff_fade:.4f}")
        tail_filters.append("format=yuv420p")
        graph_parts.append(f"[{current_label}]{','.join(tail_filters)}[vout]")

        fd, part_name = tempfile.mkstemp(
            prefix=f".mvd_{chapter.id}_", suffix=".part.mp4", dir=seg_dst.parent
        )
        os.close(fd)
        part_path = Path(part_name)
        try:
            media.run(
                [
                    "ffmpeg",
                    "-y",
                    "-v",
                    "error",
                    *inputs_argv,
                    "-filter_complex",
                    ";".join(graph_parts),
                    "-map",
                    "[vout]",
                    "-frames:v",
                    str(frames),
                    "-an",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "medium",
                    "-crf",
                    "18",
                    "-pix_fmt",
                    "yuv420p",
                    "-r",
                    str(fps),
                    "-video_track_timescale",
                    "90000",
                    "-g",
                    str(max(12, fps * 2)),
                    str(part_path.resolve()),
                ]
            )
            os.replace(part_path, seg_dst)
            atomic_write(stamp_dst, json.dumps(stamp, indent=2) + "\n")
        except BaseException:
            part_path.unlink(missing_ok=True)
            raise


def chapters_sidecar_path(output: Path) -> Path:
    """Sidecar YouTube chapter list path alongside ``output`` (``<stem>.chapters.txt``)."""
    return output.with_name(f"{output.stem}.chapters.txt")


def assemble_mix(
    project: Project,
    manifest: Manifest,
    output: Path,
    force: bool = False,
    *,
    draft: bool = False,
    width: int | None = None,
    height: int | None = None,
    chapters: Collection[str] | None = None,
    rebuild_cache: bool = False,
) -> Path:
    """Render dirty chapter segments into ``assets/segments/`` and stream-copy mux the mix."""
    if not manifest.chapters:
        raise MVDError("Mix manifest contains no chapters to assemble.")

    out_w, out_h = _resolve_output_size(project, draft=draft, width=width, height=height)
    refresh_reviews(manifest, project.config)

    stills_only = uses_stills_only(manifest, project.config)
    blocked = [s.id for s in manifest.shots if not is_ready(s, stills_only=stills_only)]
    if blocked and not force:
        raise NotReadyError(
            f"{len(blocked)} loop unit(s) are unfinished or awaiting review: "
            f"{', '.join(blocked[:12])}{' ...' if len(blocked) > 12 else ''}. "
            "Finish them, or pass --force for a draft."
        )

    forced_ids: set[str] = set()
    for cid in chapters or ():
        try:
            forced_ids.add(manifest.chapter(cid).id)
        except KeyError as exc:
            raise MVDError(str(exc)) from exc

    mix_path = project.input_path(project.config.inputs.mix, "mix")
    output.parent.mkdir(parents=True, exist_ok=True)
    project.segments_dir.mkdir(parents=True, exist_ok=True)

    rendered_count = 0
    reused_count = 0
    concat_entries: list[str] = []

    for ch, frames in zip(manifest.chapters, chapter_frame_counts(manifest.chapters, manifest.fps)):
        if frames <= 0:
            continue
        safe_id = require_safe_id(ch.id, "chapter id")
        seg_path = project.segment_video_path(safe_id, draft=draft)
        stamp_path = project.segment_stamp_path(safe_id, draft=draft)
        sources = _resolve_chapter_sources(project, manifest, ch, draft=draft)
        stamp = _build_chapter_stamp(
            project,
            ch,
            frames,
            sources,
            mix_path,
            width=out_w,
            height=out_h,
            fps=manifest.fps,
            draft=draft,
        )

        is_hit = (
            not rebuild_cache
            and safe_id not in forced_ids
            and seg_path.is_file()
            and _read_stamp(stamp_path) == stamp
        )
        if is_hit:
            reused_count += 1
            log.info("Chapter %s: reusing cached segment (%s)", safe_id, seg_path.name)
        else:
            log.info(
                "Chapter %s (%s): rendering %d frames (%.1fs)...",
                safe_id,
                ch.title,
                frames,
                frames / manifest.fps,
            )
            _render_chapter_segment(
                project,
                ch,
                frames,
                sources,
                mix_path,
                seg_path,
                stamp_path,
                stamp,
                width=out_w,
                height=out_h,
                fps=manifest.fps,
            )
            rendered_count += 1
        concat_entries.append(f"file '{seg_path.name}'")

    if not concat_entries:
        raise MVDError("No chapters with frames to assemble.")

    fd_c, concat_tmp = tempfile.mkstemp(
        prefix=".mvd_concat_", suffix=".txt", dir=project.segments_dir
    )
    os.close(fd_c)
    concat_path = Path(concat_tmp)

    fd_m, meta_tmp = tempfile.mkstemp(prefix=".mvd_ffmeta_", suffix=".txt", dir=output.parent)
    os.close(fd_m)
    meta_path = Path(meta_tmp)

    fd_o, out_tmp = tempfile.mkstemp(
        prefix=f".mvd_{output.stem}_", suffix=".part.mp4", dir=output.parent
    )
    os.close(fd_o)
    out_part = Path(out_tmp)

    try:
        concat_path.write_text("\n".join(concat_entries) + "\n", encoding="utf-8")
        meta_path.write_text(
            format_ffmetadata_chapters(manifest.chapters, manifest.title), encoding="utf-8"
        )
        media.run(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-protocol_whitelist",
                "file",
                "-f",
                "concat",
                "-i",
                str(concat_path.resolve()),
                "-protocol_whitelist",
                "file",
                "-i",
                str(mix_path.resolve()),
                "-protocol_whitelist",
                "file",
                "-f",
                "ffmetadata",
                "-i",
                str(meta_path.resolve()),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-map_metadata",
                "2",
                "-map_chapters",
                "2",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-b:a",
                "320k",
                "-shortest",
                "-movflags",
                "+faststart",
                str(out_part.resolve()),
            ]
        )
        os.replace(out_part, output)
    finally:
        concat_path.unlink(missing_ok=True)
        meta_path.unlink(missing_ok=True)
        out_part.unlink(missing_ok=True)

    if project.config.mix.chapters_sidecar:
        yt_text, yt_warnings = format_youtube_chapters(manifest.chapters, manifest.duration)
        for warning in yt_warnings:
            log.warning("YouTube chapters: %s", warning)
        sidecar = chapters_sidecar_path(output)
        atomic_write(sidecar, yt_text)
        log.info("Wrote YouTube chapters sidecar: %s", sidecar)

    from mvdirector.audio import broadcast_qa as broadcast_qa_mod

    chapter_boundaries = [ch.start for ch in manifest.chapters if ch.start > 0.0]
    try:
        bqa = broadcast_qa_mod.analyze_broadcast_audio(
            mix_path, boundary_times=chapter_boundaries or None
        )
        level = logging.WARNING if bqa.problems() else logging.INFO
        log.log(level, "Mix audio QA (clicks checked at chapter joins): %s", bqa.summary())
    except Exception as exc:  # noqa: BLE001
        log.warning("Skipped mix broadcast audio QA on %s: %s", mix_path.name, exc)

    log.info(
        "Wrote %s (%d chapters: %d rendered, %d cached)",
        output,
        len(concat_entries),
        rendered_count,
        reused_count,
    )
    return output
