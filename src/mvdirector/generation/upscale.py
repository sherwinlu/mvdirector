# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Selective video upscaling: promote approved draft clips to target resolution via ComfyUI."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from mvdirector.assemble import validate_override_size
from mvdirector.comfy.client import ComfyClient
from mvdirector.comfy.workflow import WorkflowTemplate
from mvdirector.errors import ConfigError
from mvdirector.generation.candidates import ensure_upload_audio_track
from mvdirector.models import AssetStatus, Manifest
from mvdirector.project import Project
from mvdirector.review import policy as review_policy

log = logging.getLogger(__name__)


def upscale_videos(
    project: Project,
    manifest: Manifest,
    comfy: ComfyClient,
    template: WorkflowTemplate,
    *,
    shot_ids: list[str] | None = None,
    force: bool = False,
    overwrite: bool = False,
    width: int | None = None,
    height: int | None = None,
    on_shot_done: Callable[[str, int, int], None] | None = None,
) -> list[Path]:
    """Run the ``upscale`` ComfyUI workflow on selected shots that are ready (or ``force``).

    Skips shots whose selected attempt has already been upscaled unless ``overwrite`` is true.
    """
    if (width is None) != (height is None):
        raise ConfigError("Set both width and height together when overriding upscale size.")
    if width is not None and height is not None:
        validate_override_size(width, height, "Upscale")

    review_policy.refresh_reviews(manifest, project.config)
    video = project.config.video
    target_w = width if width is not None else video.width
    target_h = height if height is not None else video.height
    target_shots = [s for s in manifest.shots if shot_ids is None or s.id in shot_ids]
    total = len(target_shots)
    upscaled: list[Path] = []
    for idx, shot in enumerate(target_shots, start=1):
        selected = shot.video.selected or (shot.video.best_attempt() if force else None)
        if selected is None or not selected.path:
            if on_shot_done is not None:
                on_shot_done(shot.id, idx, total)
            continue
        if not force and not review_policy.is_ready(shot):
            if shot.video.status is not AssetStatus.PASSED:
                log.info("%s: video is %s; skipping upscale", shot.id, shot.video.status.value)
            else:
                log.info("%s: awaiting review decision; skipping upscale", shot.id)
            if on_shot_done is not None:
                on_shot_done(shot.id, idx, total)
            continue
        if (
            not overwrite
            and shot.video.upscaled_path
            and shot.video.upscaled_from_attempt == selected.number
            and project.path(shot.video.upscaled_path).is_file()
        ):
            log.info("%s: attempt %d already upscaled; skipping", shot.id, selected.number)
            if on_shot_done is not None:
                on_shot_done(shot.id, idx, total)
            continue

        source_path = project.path(selected.path)
        if not source_path.is_file():
            raise ConfigError(f"Selected video file for {shot.id} is missing: {source_path}")
        destination = project.upscaled_video_path(shot.id, selected.number)
        params: dict[str, Any] = {
            "video": comfy.upload(ensure_upload_audio_track(project, source_path)),
            "width": target_w,
            "height": target_h,
            "fps": video.fps,
            "seed": project.config.seed + shot.index,
            "positive_prompt": selected.prompt,
            "negative_prompt": manifest.visual_bible.negative_prompt,
        }
        if template.accepts("audio") and shot.audio_slice.path:
            audio_path = project.path(shot.audio_slice.path)
            if audio_path.is_file():
                params["audio"] = comfy.upload(audio_path)
        if template.accepts("image") and shot.still.selected and shot.still.selected.path:
            still_path = project.path(shot.still.selected.path)
            if still_path.is_file():
                params["image"] = comfy.upload(still_path)
        comfy.run(template.render(params), template.bindings.output, destination)
        shot.video.upscaled_path = project.rel(destination)
        shot.video.upscaled_from_attempt = selected.number
        project.save_manifest(manifest)
        upscaled.append(destination)
        if on_shot_done is not None:
            on_shot_done(shot.id, idx, total)
        log.info("%s: upscaled attempt %d -> %s", shot.id, selected.number, destination)
    return upscaled
