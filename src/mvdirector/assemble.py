# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Final assembly: trim each clip to its shot, join them and lay the original mix on top.

Clip lengths are computed in whole frames from the cumulative timeline, so rounding
never accumulates and the picture stays locked to the music from start to finish.
"""

from __future__ import annotations

import logging
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from mvdirector import media
from mvdirector.audio import broadcast_qa as broadcast_qa_mod
from mvdirector.audio import foley as foley_mod
from mvdirector.errors import ConfigError, MVDError
from mvdirector.models import (
    Analysis,
    Attempt,
    ContinuityMode,
    Manifest,
    RetryStrategy,
    Shot,
    ShotType,
)
from mvdirector.project import Project, require_safe_id
from mvdirector.qc.color_drift import fit_color_drift_affine
from mvdirector.review.policy import is_ready, refresh_reviews

log = logging.getLogger(__name__)

_ENCODE = ["-an", "-c:v", "libx264", "-preset", "medium", "-crf", "16", "-pix_fmt", "yuv420p"]
_XFADE_MAP = {
    "blend": "fade",
    "whip": "smoothleft",
    "flash": "fadewhite",
}


def color_grade_filter(grade: str, film_grain: float = 0.0) -> str | None:
    """Build an RGB-space (``gbrp``) FFmpeg film color grade + optional grain filter chain.

    Orders YUV ``eq`` strictly before ``format=gbrp`` before RGB curves/colorbalance so highlight
    shifts never introduce a YUV chroma magenta cast.

    Credits & Attribution:
        RGB-space (``gbrp``) film grading ordering and presets inspired by
        ``hexeosis-prime/nightshift``
        (https://github.com/hexeosis-prime/nightshift — ``nightshift/grade.py``).
    """
    stages: list[str] = []
    if grade == "portra_dusk":
        stages.append(
            "eq=contrast=1.04:saturation=1.06,format=gbrp,"
            "colorbalance=rs=0.03:gs=0.01:bs=-0.02:rm=0.02:gm=0.0:bm=-0.03:rh=0.04:gh=0.02:bh=-0.03,"
            "format=yuv420p"
        )
    elif grade == "cinestill_night":
        stages.append(
            "eq=contrast=1.08:saturation=1.08,format=gbrp,"
            "colorbalance=rs=-0.04:gs=0.01:bs=0.05:rm=-0.02:gm=0.0:bm=0.03:rh=0.05:gh=0.01:bh=-0.02,"
            "format=yuv420p"
        )
    elif grade == "faded_pastel":
        stages.append(
            "eq=contrast=0.92:brightness=0.02:saturation=0.88,format=gbrp,"
            "colorbalance=rs=0.02:gs=0.02:bs=0.03:rh=0.02:gh=0.01:bh=0.0,"
            "format=yuv420p"
        )
    elif grade == "neutral_film":
        stages.append(
            "eq=contrast=1.03:saturation=0.98,format=gbrp,"
            "colorbalance=rs=0.01:bs=-0.01:rh=0.01:bh=-0.01,"
            "format=yuv420p"
        )
    if film_grain > 0.0:
        strength = max(0.0, min(25.0, float(film_grain)))
        stages.append(f"noise=alls={strength:.2f}:allf=t+u")
    return ",".join(stages) if stages else None


class NotReadyError(MVDError):
    """Some shots are unfinished or awaiting review."""


@dataclass(frozen=True)
class ClipPlan:
    shot_id: str
    frames: int
    source: Path | None  # video clip, still image, or None for black
    is_video: bool
    head_handle: float
    sync_delay: float = 0.0
    drop_first_frame: bool = False
    continuity: ContinuityMode = ContinuityMode.CUT
    tail_handle: float = 0.0


def frame_counts(shots: list[Shot], fps: int) -> list[int]:
    """Frames per shot such that every cut lands on the nearest frame of the timeline."""
    return [round(shot.end * fps) - round(shot.start * fps) for shot in shots]


@dataclass(frozen=True)
class ClipTiming:
    """Which frames of a shot's generated clip are played at assembly.

    The single source of truth shared with the runner, so continuity handoff frames and seam QC
    look at exactly the frames that end up on screen.
    """

    in_point: float  # seconds into the source clip (ffmpeg -ss), lip-sync delay included
    sync_delay: float  # lip-sync offset applied, clamped to the shot's audio handles
    drop_first_frame: bool  # a CONTINUE clip's frame 0 repeats the previous clip's last frame
    continuity: ContinuityMode  # CONTINUE only when the still really was a handoff frame

    def first_played_frame(self, fps: int) -> int:
        return round(self.in_point * fps) + (1 if self.drop_first_frame else 0)

    def last_played_frame(self, frames: int, fps: int) -> int:
        return self.first_played_frame(fps) + max(1, frames) - 1


def clip_timing(
    shot: Shot, video: Attempt, still: Attempt | None, *, sync_delay_compensation: bool
) -> ClipTiming:
    """Timing of ``shot``'s clip ``video`` as assembly will play it."""
    head = shot.audio_slice.head_handle
    tail = max(0.0, shot.audio_slice.end - shot.end)
    if shot.type is ShotType.LIPSYNC:
        raw_offset = video.sync_delay_seconds if sync_delay_compensation else 0.0
    else:
        raw_offset = video.optimal_in_offset
    delay = round(max(-head, min(tail, raw_offset)), 4)
    used_handoff = (
        shot.index > 0
        and shot.continuity is ContinuityMode.CONTINUE
        and still is not None
        and still.strategy is RetryStrategy.HANDOFF
    )
    continuity = shot.continuity
    if continuity is ContinuityMode.CONTINUE and not used_handoff:
        # A fresh (non-handoff) still makes the join a plain cut: no seam treatment applies.
        continuity = ContinuityMode.CUT
    return ClipTiming(
        in_point=max(0.0, head + delay),
        sync_delay=delay,
        drop_first_frame=used_handoff and head == 0.0,
        continuity=continuity,
    )


def plan_clips(
    project: Project, manifest: Manifest, force: bool, *, draft: bool = False
) -> list[ClipPlan]:
    blocked = [s.id for s in manifest.shots if not is_ready(s)]
    if blocked and not force:
        raise NotReadyError(
            f"{len(blocked)} shot(s) are unfinished or awaiting review: {', '.join(blocked[:12])}"
            f"{' ...' if len(blocked) > 12 else ''}. Finish them, or pass --force for a draft."
        )
    use_sync_delay = project.config.video.sync_delay_compensation
    plans = []
    for shot, frames in zip(manifest.shots, frame_counts(manifest.shots, manifest.fps)):
        shot_id = require_safe_id(shot.id, "shot id")
        video = shot.video.selected or shot.video.best_attempt()
        still = shot.still.selected or shot.still.best_attempt()
        tail_handle = max(0.0, shot.audio_slice.end - shot.end)
        if video is not None and video.path:
            source_rel = video.path
            if (
                not draft
                and shot.video.upscaled_path
                and shot.video.upscaled_from_attempt == video.number
                and project.path(shot.video.upscaled_path).is_file()
            ):
                source_rel = shot.video.upscaled_path
            timing = clip_timing(shot, video, still, sync_delay_compensation=use_sync_delay)
            plans.append(
                ClipPlan(
                    shot_id,
                    frames,
                    project.path(source_rel),
                    True,
                    shot.audio_slice.head_handle,
                    sync_delay=timing.sync_delay,
                    drop_first_frame=timing.drop_first_frame,
                    continuity=timing.continuity,
                    tail_handle=tail_handle,
                )
            )
        elif still is not None and still.path:  # draft: hold the still
            plans.append(ClipPlan(shot_id, frames, project.path(still.path), False, 0.0))
        else:  # draft: black placeholder
            plans.append(ClipPlan(shot_id, frames, None, False, 0.0))
    return plans


def _intercut_run(
    run: list[tuple[Shot, ClipPlan]],
    beats: list[float],
    fps: int,
    *,
    tempo_bpm: float | None = None,
    min_intensity: float,
    anticipation_frames: int,
) -> list[ClipPlan]:
    """Beat-lock intercut one contiguous non-lipsync video run, preserving exact frame count."""
    originals = [p for _, p in run]
    if len(run) < 2 or fps <= 0:
        return originals

    total_frames = sum(p.frames for p in originals)
    if total_frames <= 0:
        return originals

    weighted_intensity = sum(s.intensity * p.frames for s, p in run) / total_frames
    if weighted_intensity < min_intensity:
        return originals

    run_start = run[0][0].start
    run_start_frame = round(run_start * fps)
    run_end_frame = run_start_frame + total_frames
    run_end = run_end_frame / fps

    run_beats = [b for b in beats if run_start + 1e-3 < b < run_end - 1e-3]
    if len(run_beats) < 2 and tempo_bpm is not None and 50.0 <= tempo_bpm <= 220.0:
        step_s = 60.0 / tempo_bpm
        k = 1
        synth: list[float] = []
        while run_start + k * step_s < run_end - 1e-3:
            synth.append(run_start + k * step_s)
            k += 1
        run_beats = synth
    if len(run_beats) < 2:
        return originals

    beat_step_s = float(np.median(np.diff(run_beats)))
    min_sub_frames = max(10, round(0.42 * fps))
    step_pattern: tuple[int, ...]
    if weighted_intensity >= 0.85 and beat_step_s >= 0.42:
        step_pattern = (2, 2, 1, 1)
    elif beat_step_s >= 0.70:
        step_pattern = (1,)
    else:
        step_pattern = (2,)

    frame_edges = [run_start_frame]
    pat_idx = 0
    beat_idx = step_pattern[0] - 1
    while beat_idx < len(run_beats):
        cut_frame = round(run_beats[beat_idx] * fps) - anticipation_frames
        if (
            cut_frame - frame_edges[-1] >= min_sub_frames
            and run_end_frame - cut_frame >= min_sub_frames
        ):
            frame_edges.append(cut_frame)
            pat_idx += 1
            beat_idx += step_pattern[pat_idx % len(step_pattern)]
        else:
            beat_idx += 1
    frame_edges.append(run_end_frame)
    if len(frame_edges) <= 2:
        return originals

    cursors = [max(0.0, p.head_handle + p.sync_delay) for p in originals]
    # Keep a small tail-drift margin when generous tail handle exists, while never dipping below
    # the clip's own frame budget.
    remaining_frames = [
        p.frames + max(0, round((p.tail_handle - min(0.10, p.tail_handle)) * fps))
        for p in originals
    ]

    sub_plans: list[ClipPlan] = []
    prev_src = -1

    def _emit(src_idx: int, frames: int) -> None:
        nonlocal prev_src
        src_plan = originals[src_idx]
        sub_dur = frames / fps
        sub_plans.append(
            ClipPlan(
                shot_id=f"{src_plan.shot_id}_ic{len(sub_plans):02d}",
                frames=frames,
                source=src_plan.source,
                is_video=True,
                head_handle=round(cursors[src_idx], 4),
                sync_delay=0.0,
                drop_first_frame=False,
                continuity=ContinuityMode.CUT,
                tail_handle=round(max(0.0, (remaining_frames[src_idx] - frames) / fps), 4),
            )
        )
        cursors[src_idx] += sub_dur
        remaining_frames[src_idx] -= frames
        prev_src = src_idx

    for f_a, f_b in zip(frame_edges[:-1], frame_edges[1:]):
        need = f_b - f_a
        while need > 0:
            full = [
                idx
                for idx in range(len(originals))
                if idx != prev_src and remaining_frames[idx] >= need
            ]
            if full:
                chosen = max(full, key=lambda idx: (remaining_frames[idx], -idx))
                _emit(chosen, need)
                need = 0
                break
            # If the remaining slice exceeds any single source's remaining footage (e.g. final
            # boundary), split cleanly across two distinct sources rather than aborting the run.
            partial = [
                idx
                for idx in range(len(originals))
                if idx != prev_src and remaining_frames[idx] >= min(need, min_sub_frames)
            ]
            if not partial:
                # If no alternate source has enough safe frames left for a final sliver, extend
                # the previous sub-clip continuously using its own remaining frames (no jump cut).
                if sub_plans and prev_src >= 0 and remaining_frames[prev_src] >= need:
                    prev_plan = sub_plans[-1]
                    sub_plans[-1] = ClipPlan(
                        shot_id=prev_plan.shot_id,
                        frames=prev_plan.frames + need,
                        source=prev_plan.source,
                        is_video=prev_plan.is_video,
                        head_handle=prev_plan.head_handle,
                        sync_delay=prev_plan.sync_delay,
                        drop_first_frame=prev_plan.drop_first_frame,
                        continuity=prev_plan.continuity,
                        tail_handle=round(max(0.0, (remaining_frames[prev_src] - need) / fps), 4),
                    )
                    cursors[prev_src] += need / fps
                    remaining_frames[prev_src] -= need
                    need = 0
                    break
                return originals
            chosen = max(partial, key=lambda idx: (remaining_frames[idx], -idx))
            take = min(need, remaining_frames[chosen])
            if need - take > 0 and need - take < min_sub_frames:
                take = need - min_sub_frames
                if take < min_sub_frames or take > remaining_frames[chosen]:
                    return originals
            _emit(chosen, take)
            need -= take

    return sub_plans


def intercut_plans(
    plans: list[ClipPlan],
    manifest: Manifest,
    analysis: Analysis | None,
    *,
    min_intensity: float = 0.70,
    anticipation_frames: int = 1,
) -> list[ClipPlan]:
    """Beat-lock intercut high-intensity contiguous non-lipsync video runs at assembly time.

    Lip-sync shots, calm sections (< ``min_intensity``), still holds, and ``CONTINUE`` chains
    are preserved untouched. Total frame count is conserved identically for every run.
    """
    if analysis is None or len(plans) != len(manifest.shots) or len(plans) < 2:
        return list(plans)

    result: list[ClipPlan] = []
    run: list[tuple[Shot, ClipPlan]] = []

    def _flush() -> None:
        if not run:
            return
        result.extend(
            _intercut_run(
                run,
                analysis.beats,
                manifest.fps,
                tempo_bpm=analysis.tempo_bpm,
                min_intensity=min_intensity,
                anticipation_frames=anticipation_frames,
            )
        )
        run.clear()

    for idx, (shot, plan) in enumerate(zip(manifest.shots, plans)):
        next_is_continue = (
            idx + 1 < len(plans) and plans[idx + 1].continuity is ContinuityMode.CONTINUE
        )
        eligible = (
            shot.type is not ShotType.LIPSYNC
            and plan.is_video
            and plan.source is not None
            and plan.continuity is ContinuityMode.CUT
            and not next_is_continue
            and plan.frames > 0
        )
        if eligible:
            run.append((shot, plan))
        else:
            _flush()
            result.append(plan)
    _flush()
    return result


def clip_command(
    plan: ClipPlan,
    output: Path,
    width: int,
    height: int,
    fps: int,
    *,
    extra_frames: int = 0,
    extra_vf: str | None = None,
    encode_args: list[str] | None = None,
) -> list[str]:
    """ffmpeg command that conforms one source to ``plan.frames + extra_frames`` frames.

    Applies per-take ``sync_delay`` offset compensation (inspired by
    ``VisualFrisson/ComfyUI-OneShotMV``) and ``trim=start_frame=1`` seam-frame drop on
    ``CONTINUE`` one-take shots (inspired by ``afj176/open-cinema-studio``).
    """
    fit = (
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1"
    )
    target_frames = plan.frames + max(0, extra_frames)
    if plan.source is None:
        source = ["-f", "lavfi", "-i", f"color=c=black:s={width}x{height}:r={fps}"]
        filters = "setsar=1"
    elif plan.is_video:
        effective_ss = max(0.0, plan.head_handle + plan.sync_delay)
        source = [
            "-protocol_whitelist",
            "file",
            "-ss",
            f"{effective_ss:.4f}",
            "-i",
            str(plan.source.resolve()),
        ]
        # Drop duplicated handoff frame 0 on CONTINUE one-take chains; tpad covers slight shorts.
        chain = [f"fps={fps}"]
        if plan.drop_first_frame and target_frames > 1:
            chain.append("trim=start_frame=1,setpts=PTS-STARTPTS")
        chain.append("tpad=stop_mode=clone:stop_duration=2")
        chain.append(fit)
        filters = ",".join(chain)
    else:
        source = [
            "-protocol_whitelist",
            "file",
            "-loop",
            "1",
            "-framerate",
            str(fps),
            "-i",
            str(plan.source.resolve()),
        ]
        filters = fit
    if extra_vf:
        filters = f"{filters},{extra_vf}"
    return [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        *source,
        "-vf",
        filters,
        "-frames:v",
        str(target_frames),
        *(encode_args if encode_args is not None else _ENCODE),
        str(output.resolve()),
    ]


# Largest width/height accepted as a size override (covers 8K UHD and DCI 8K). Frames are
# conformed in memory at this size, so an unbounded override (e.g. 100000x100000 from an MCP
# caller) asked PIL and ffmpeg for tens of gigabytes per frame.
MAX_OUTPUT_DIMENSION = 8192


def validate_override_size(width: int, height: int, label: str) -> None:
    """Reject a width/height override that is odd, below 64, or above MAX_OUTPUT_DIMENSION."""
    if (
        width < 64
        or height < 64
        or width % 2
        or height % 2
        or width > MAX_OUTPUT_DIMENSION
        or height > MAX_OUTPUT_DIMENSION
    ):
        raise ConfigError(
            f"{label} dimensions must be even integers between 64 and {MAX_OUTPUT_DIMENSION}, "
            f"got {width}x{height}."
        )


def _resolve_output_size(
    project: Project, *, draft: bool, width: int | None, height: int | None
) -> tuple[int, int]:
    if (width is None) != (height is None):
        raise ConfigError("Set both width and height together when overriding assembly size.")
    if width is not None and height is not None:
        validate_override_size(width, height, "Assembly")
        return (width, height)
    video = project.config.video
    return video.generation_size(draft=True) if draft else (video.width, video.height)


def _fit_active_pair(
    prev_img: Image.Image, curr_img: Image.Image, out_w: int, out_h: int
) -> tuple[np.ndarray, np.ndarray]:
    """Conform ``curr_img`` identically to FFmpeg scale+pad and crop out letterbox margins."""
    prev_rgb = prev_img.convert("RGB")
    curr_rgb = curr_img.convert("RGB")
    scale = min(out_w / max(1, curr_rgb.width), out_h / max(1, curr_rgb.height))
    nw = max(1, int(round(curr_rgb.width * scale)))
    nh = max(1, int(round(curr_rgb.height * scale)))
    ox = max(0, (out_w - nw) // 2)
    oy = max(0, (out_h - nh) // 2)
    if prev_rgb.size != (out_w, out_h):
        prev_rgb = prev_rgb.resize((out_w, out_h), Image.Resampling.BILINEAR)
    prev_active = prev_rgb.crop((ox, oy, ox + nw, oy + nh))
    curr_active = curr_rgb.resize((nw, nh), Image.Resampling.BILINEAR)
    return np.asarray(prev_active), np.asarray(curr_active)


def _transition_pieces(
    clips: list[tuple[ClipPlan, Path, int]],
    work: Path,
    fps: int,
    transition: str,
    *,
    encode_args: list[str] | None = None,
    workers: int = 1,
) -> list[Path]:
    """Body and crossfade pieces that concatenate into the frame-exact, beat-locked timeline.

    Clip ``i`` was conformed with ``overlap_i`` spare tail frames. Its crossfade into clip
    ``i + 1`` blends exactly those frames with the first ``overlap_i`` frames of the next clip,
    which then continues from frame ``overlap_i``. Each crossfade is rendered from just its two
    clips: one ffmpeg graph over every clip holds a decoder per shot open at once and needs
    gigabytes of memory for a full-length video at 1080p.
    """
    codec_flags = encode_args if encode_args is not None else _ENCODE
    tb = f"1/{fps}"
    tasks: list[tuple[Path, list[str]]] = []
    head_skip = 0  # frames of this clip already shown inside the previous crossfade
    for idx, (plan, clip_path, overlap) in enumerate(clips):
        if plan.frames > head_skip:
            body = work / f"{clip_path.stem}_body.mp4"
            tasks.append(
                (
                    body,
                    [
                        "ffmpeg",
                        "-y",
                        "-v",
                        "error",
                        "-protocol_whitelist",
                        "file",
                        "-i",
                        str(clip_path.resolve()),
                        "-vf",
                        f"trim=start_frame={head_skip}:end_frame={plan.frames},setpts=PTS-STARTPTS",
                        "-frames:v",
                        str(plan.frames - head_skip),
                        *codec_flags,
                        str(body.resolve()),
                    ],
                )
            )
        head_skip = 0
        if overlap > 0 and idx + 1 < len(clips):
            blend = work / f"{clip_path.stem}_xfade.mp4"
            tail_end = plan.frames + overlap
            graph = (
                f"[0:v]trim=start_frame={plan.frames}:end_frame={tail_end},"
                f"setpts=PTS-STARTPTS,settb={tb}[a];"
                f"[1:v]trim=start_frame=0:end_frame={overlap},setpts=PTS-STARTPTS,settb={tb}[b];"
                f"[a][b]xfade=transition={transition}:duration={overlap / fps:.6f}:offset=0[v]"
            )
            tasks.append(
                (
                    blend,
                    [
                        "ffmpeg",
                        "-y",
                        "-v",
                        "error",
                        "-protocol_whitelist",
                        "file",
                        "-i",
                        str(clip_path.resolve()),
                        "-protocol_whitelist",
                        "file",
                        "-i",
                        str(clips[idx + 1][1].resolve()),
                        "-filter_complex",
                        graph,
                        "-map",
                        "[v]",
                        "-frames:v",
                        str(overlap),
                        *codec_flags,
                        str(blend.resolve()),
                    ],
                )
            )
            head_skip = overlap
    if workers > 1 and len(tasks) > 1:
        with ThreadPoolExecutor(max_workers=min(workers, len(tasks))) as pool:
            list(pool.map(lambda item: media.run(item[1]), tasks))
    else:
        for _, cmd in tasks:
            media.run(cmd)
    return [piece for piece, _ in tasks]


def assemble(
    project: Project,
    manifest: Manifest,
    output: Path,
    force: bool = False,
    *,
    draft: bool = False,
    width: int | None = None,
    height: int | None = None,
) -> Path:
    """Build the final video. Raises NotReadyError unless every shot is ready or ``force``."""
    out_w, out_h = _resolve_output_size(project, draft=draft, width=width, height=height)
    refresh_reviews(manifest, project.config)
    plans = plan_clips(project, manifest, force, draft=draft)
    video_cfg = project.config.video
    if video_cfg.assembly_intercut and project.analysis_path.is_file():
        try:
            analysis = project.load_analysis()
        except Exception as exc:  # noqa: BLE001
            log.warning("Skipping assembly intercut: could not load analysis (%s)", exc)
        else:
            before_count = len(plans)
            plans = intercut_plans(
                plans,
                manifest,
                analysis,
                min_intensity=video_cfg.assembly_intercut_min_intensity,
                anticipation_frames=video_cfg.assembly_intercut_anticipation_frames,
            )
            if len(plans) != before_count:
                log.info(
                    "Assembly intercut expanded %d shot(s) into %d beat-locked clip(s).",
                    before_count,
                    len(plans),
                )
    mix = project.input_path(project.config.inputs.mix, "mix")
    output.parent.mkdir(parents=True, exist_ok=True)

    grade_vf = color_grade_filter(video_cfg.color_grade, video_cfg.film_grain)
    xfade_name = _XFADE_MAP.get(video_cfg.transition)
    use_xfade = xfade_name is not None and video_cfg.transition_seconds > 0.0

    with tempfile.TemporaryDirectory(prefix="mvd_assemble_") as tmp:
        work = Path(tmp)
        active_plans = [p for p in plans if p.frames > 0]
        if not active_plans:
            raise MVDError("No shots with frames to assemble.")

        xfade_frames = (
            max(1, round(video_cfg.transition_seconds * manifest.fps)) if use_xfade else 0
        )
        encoder = media.resolve_video_encoder(video_cfg.encoder)
        encode_args = media.encode_args_for(encoder)
        workers = max(1, min(8, video_cfg.assembly_workers))

        clips: list[tuple[ClipPlan, Path, int]] = []
        for idx, plan in enumerate(active_plans):
            safe_id = require_safe_id(plan.shot_id, "shot id")
            clip = work / f"{safe_id}.mp4"
            is_last = idx == len(active_plans) - 1
            # Beat-preserving xfade tail borrowing (inspired by thefullnacho/on-the-one conform.py):
            # borrow extra tail frames on shot i so shot i+1 still starts at its exact musical beat.
            next_plan = active_plans[idx + 1] if not is_last else None
            next_frames = next_plan.frames if next_plan is not None else 0
            next_is_continue = (
                next_plan is not None and next_plan.continuity is ContinuityMode.CONTINUE
            )
            if (
                use_xfade
                and not is_last
                and not next_is_continue
                and plan.frames >= 3
                and next_frames >= 3
            ):
                overlap = min(xfade_frames, plan.frames // 3, next_frames // 3)
                if plan.is_video:
                    avail_tail = max(
                        0, round((plan.tail_handle - max(0.0, plan.sync_delay)) * manifest.fps)
                    )
                    overlap = min(overlap, avail_tail)
            else:
                overlap = 0
            clips.append((plan, clip, overlap))

        # Partition clips into independent chains so CONTINUE color-drift joins run sequentially
        # while all independent CUT clips conform in parallel.
        chains: list[list[int]] = []
        for idx, (plan, _, _) in enumerate(clips):
            needs_prev = (
                video_cfg.color_drift_correction
                and plan.continuity is ContinuityMode.CONTINUE
                and plan.is_video
                and plan.source is not None
                and idx > 0
            )
            if needs_prev and chains:
                chains[-1].append(idx)
            else:
                chains.append([idx])

        def _conform_chain(indices: list[int]) -> None:
            prev_clip_info: tuple[Path, int] | None = None
            for c_idx in indices:
                plan, clip, overlap = clips[c_idx]
                safe_id = require_safe_id(plan.shot_id, "shot id")
                vf_parts: list[str] = []
                if (
                    video_cfg.color_drift_correction
                    and plan.continuity is ContinuityMode.CONTINUE
                    and prev_clip_info is not None
                    and plan.is_video
                    and plan.source is not None
                ):
                    # Clean-room VAE color-drift affine match using exact boundary frames only
                    # (inspired by SonderSaid/ComfyUI-Sonder-Editor color_correction.py).
                    prev_path, prev_last_idx = prev_clip_info
                    prev_png = work / f"{safe_id}_prev_ref.png"
                    curr_png = work / f"{safe_id}_curr_ref.png"
                    # First frame this clip actually plays (after any dropped handoff seam frame).
                    in_frame = round(max(0.0, plan.head_handle + plan.sync_delay) * manifest.fps)
                    start_idx = in_frame + (1 if plan.drop_first_frame else 0)
                    media.extract_frame_at_index(prev_path, max(0, prev_last_idx), prev_png)
                    media.extract_frame_at_index(plan.source, start_idx, curr_png)
                    with Image.open(prev_png) as p_img, Image.open(curr_png) as c_img:
                        prev_arr, curr_arr = _fit_active_pair(p_img, c_img, out_w, out_h)
                    affine = fit_color_drift_affine([curr_arr], [prev_arr])
                    if mixer := affine.to_ffmpeg_colorchannelmixer():
                        vf_parts.append(mixer)
                # When color_drift_correction is enabled, apply grade_vf once after concat/xfade so
                # chained CONTINUE clips are never double-graded against already-graded neighbors.
                if grade_vf and not video_cfg.color_drift_correction:
                    vf_parts.append(grade_vf)
                media.run(
                    clip_command(
                        plan,
                        clip,
                        out_w,
                        out_h,
                        manifest.fps,
                        extra_frames=overlap,
                        extra_vf=",".join(vf_parts) if vf_parts else None,
                        encode_args=encode_args,
                    )
                )
                prev_clip_info = (
                    (clip, plan.frames - 1) if (plan.is_video and plan.source is not None) else None
                )

        if workers > 1 and len(chains) > 1:
            with ThreadPoolExecutor(max_workers=min(workers, len(chains))) as pool:
                list(pool.map(_conform_chain, chains))
        else:
            for chain in chains:
                _conform_chain(chain)

        silent = work / "picture.mp4"
        post_grade_vf = grade_vf if video_cfg.color_drift_correction else None
        has_any_xfade = use_xfade and len(clips) > 1 and any(ov > 0 for _, _, ov in clips[:-1])
        pieces = (
            _transition_pieces(
                clips,
                work,
                manifest.fps,
                xfade_name,
                encode_args=encode_args,
                workers=workers,
            )
            if has_any_xfade and xfade_name is not None
            else [clip_path for _, clip_path, _ in clips]
        )
        entries = [f"file '{piece.name}'" for piece in pieces]
        concat_list = work / "clips.txt"
        concat_list.write_text("\n".join(entries) + "\n", encoding="utf-8")
        concat_codec = ["-vf", post_grade_vf, *encode_args] if post_grade_vf else ["-c", "copy"]
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
                str(concat_list.resolve()),
                *concat_codec,
                str(silent.resolve()),
            ]
        )
        # Automatically level & duck diegetic SFX cues over the mix when any shot carries sfx_path.
        mux_audio = mix
        sfx_cues: list[foley_mod.DiegeticCue] = []
        for shot in manifest.shots:
            if not shot.sfx_path:
                continue
            sfx_src = project.path(shot.sfx_path)
            if not sfx_src.is_file():
                log.warning("%s: sfx_path not found (%s); skipping cue", shot.id, shot.sfx_path)
                continue
            safe_sfx_id = require_safe_id(shot.id, "shot id")
            leveled_sfx = work / f"{safe_sfx_id}_sfx.wav"
            # Only the shot's span of the SFX is played, so only that much is read and levelled.
            foley_mod.write_leveled_foley(
                sfx_src, leveled_sfx, max_seconds=max(0.05, shot.end - shot.start)
            )
            sfx_cues.append(
                foley_mod.DiegeticCue(
                    audio_path=leveled_sfx,
                    start=shot.start,
                    end=shot.end,
                    gain_db=shot.sfx_gain_db,
                    duck_db=shot.sfx_duck_db,
                )
            )
        if sfx_cues:
            mux_audio = foley_mod.build_ducked_audio_mix(mix, sfx_cues, work / "mix_ducked.wav")
            log.info("Blended %d diegetic SFX cue(s) with automatic song ducking", len(sfx_cues))

        try:
            bqa = broadcast_qa_mod.analyze_broadcast_audio(mux_audio)
            if bqa.problems():
                log.warning("Final audio QA: %s", bqa.summary())
        except Exception as exc:  # noqa: BLE001
            log.warning("Skipped broadcast audio QA on %s: %s", mux_audio.name, exc)

        fd_out, out_tmp = tempfile.mkstemp(
            prefix=f".mvd_{output.stem}_", suffix=".part.mp4", dir=output.parent
        )
        os.close(fd_out)
        out_part = Path(out_tmp)
        try:
            mux_cmd = [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-protocol_whitelist",
                "file",
                "-i",
                str(silent.resolve()),
                "-protocol_whitelist",
                "file",
                "-i",
                str(mux_audio.resolve()),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-b:a",
                "320k",
                "-shortest",
            ]
            if video_cfg.strip_metadata:
                mux_cmd.extend(["-map_metadata", "-1"])
            mux_cmd.extend(["-movflags", "+faststart", str(out_part.resolve())])
            media.run(mux_cmd)
            os.replace(out_part, output)
        except BaseException:
            out_part.unlink(missing_ok=True)
            raise
    log.info("Wrote %s (%d shots)", output, len(plans))
    return output
