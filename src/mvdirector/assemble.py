# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Final assembly: trim each clip to its shot, join them and lay the original mix on top.

Clip lengths are computed in whole frames from the cumulative timeline, so rounding
never accumulates and the picture stays locked to the music from start to finish.
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path

from mvdirector import media
from mvdirector.errors import ConfigError, MVDError
from mvdirector.models import Manifest, Shot
from mvdirector.project import Project, require_safe_id
from mvdirector.review.policy import is_ready, refresh_reviews

log = logging.getLogger(__name__)

_ENCODE = ["-an", "-c:v", "libx264", "-preset", "medium", "-crf", "16", "-pix_fmt", "yuv420p"]


class NotReadyError(MVDError):
    """Some shots are unfinished or awaiting review."""


@dataclass(frozen=True)
class ClipPlan:
    shot_id: str
    frames: int
    source: Path | None  # video clip, still image, or None for black
    is_video: bool
    head_handle: float


def frame_counts(shots: list[Shot], fps: int) -> list[int]:
    """Frames per shot such that every cut lands on the nearest frame of the timeline."""
    return [round(shot.end * fps) - round(shot.start * fps) for shot in shots]


def plan_clips(
    project: Project, manifest: Manifest, force: bool, *, draft: bool = False
) -> list[ClipPlan]:
    blocked = [s.id for s in manifest.shots if not is_ready(s)]
    if blocked and not force:
        raise NotReadyError(
            f"{len(blocked)} shot(s) are unfinished or awaiting review: {', '.join(blocked[:12])}"
            f"{' ...' if len(blocked) > 12 else ''}. Finish them, or pass --force for a draft."
        )
    plans = []
    for shot, frames in zip(manifest.shots, frame_counts(manifest.shots, manifest.fps)):
        shot_id = require_safe_id(shot.id, "shot id")
        video = shot.video.selected or shot.video.best_attempt()
        still = shot.still.selected or shot.still.best_attempt()
        if video is not None and video.path:
            source_rel = video.path
            if (
                not draft
                and shot.video.upscaled_path
                and shot.video.upscaled_from_attempt == video.number
                and project.path(shot.video.upscaled_path).is_file()
            ):
                source_rel = shot.video.upscaled_path
            plans.append(
                ClipPlan(
                    shot_id, frames, project.path(source_rel), True, shot.audio_slice.head_handle
                )
            )
        elif still is not None and still.path:  # draft: hold the still
            plans.append(ClipPlan(shot_id, frames, project.path(still.path), False, 0.0))
        else:  # draft: black placeholder
            plans.append(ClipPlan(shot_id, frames, None, False, 0.0))
    return plans


def clip_command(plan: ClipPlan, output: Path, width: int, height: int, fps: int) -> list[str]:
    """ffmpeg command that conforms one source to exactly ``plan.frames`` frames."""
    fit = (
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1"
    )
    if plan.source is None:
        source = ["-f", "lavfi", "-i", f"color=c=black:s={width}x{height}:r={fps}"]
        filters = "setsar=1"
    elif plan.is_video:
        source = ["-ss", f"{plan.head_handle:.4f}", "-i", str(plan.source)]
        # tpad repeats the last frame if a clip came back slightly short.
        filters = f"fps={fps},tpad=stop_mode=clone:stop_duration=2,{fit}"
    else:
        source = ["-loop", "1", "-framerate", str(fps), "-i", str(plan.source)]
        filters = fit
    return [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        *source,
        "-vf",
        filters,
        "-frames:v",
        str(plan.frames),
        *_ENCODE,
        str(output),
    ]


def _resolve_output_size(
    project: Project, *, draft: bool, width: int | None, height: int | None
) -> tuple[int, int]:
    if (width is None) != (height is None):
        raise ConfigError("Set both width and height together when overriding assembly size.")
    if width is not None and height is not None:
        if width < 64 or height < 64 or width % 2 or height % 2:
            raise ConfigError(
                f"Assembly dimensions must be even integers >= 64, got {width}x{height}."
            )
        return (width, height)
    video = project.config.video
    return video.generation_size(draft=True) if draft else (video.width, video.height)


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
    mix = project.input_path(project.config.inputs.mix, "mix")
    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="mvd_assemble_") as tmp:
        work = Path(tmp)
        entries = []
        for plan in plans:
            if plan.frames <= 0:
                continue
            safe_id = require_safe_id(plan.shot_id, "shot id")
            clip = work / f"{safe_id}.mp4"
            media.run(clip_command(plan, clip, out_w, out_h, manifest.fps))
            entries.append(f"file '{clip.name}'")
        if not entries:
            raise MVDError("No shots with frames to assemble.")
        concat_list = work / "clips.txt"
        concat_list.write_text("\n".join(entries) + "\n", encoding="utf-8")
        silent = work / "picture.mp4"
        media.run(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-f",
                "concat",
                "-i",
                str(concat_list.resolve()),
                "-c",
                "copy",
                str(silent.resolve()),
            ]
        )
        media.run(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-i",
                str(silent.resolve()),
                "-i",
                str(mix.resolve()),
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
                "-movflags",
                "+faststart",
                str(output.resolve()),
            ]
        )
    log.info("Wrote %s (%d shots)", output, len(plans))
    return output
