# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""3D blocking previz planning: turn a shot spec and set plate into flat-shaded 3D staging.

Mirrors the 3D-blocking workflow where an LLM translates a set image and camera choreography
into plain, untextured geometric stand-ins and smooth camera keyframes so a downstream video
model locks onto exact pacing, angles, and framing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from mvdirector.errors import ConfigError, LLMError
from mvdirector.llm import LLMClient, image_block, text_block
from mvdirector.models import Framing, Manifest, Shot, ShotType

log = logging.getLogger(__name__)

PREVIZ_SYSTEM = """You are a 3D previz supervisor planning simple flat-shaded 3D blocking for an
AI music video shot. Your output drives a plain 3D block-figure renderer whose output will be fed
as a previz reference video into a generative video model.

Rules:
- Keep everything plain: flat solid colors, no textures. Characters are simple block figures with
  no limb animation; set structures and machines are simple boxes or cylinders; props are boxes
  or rings.
- Coordinate system in meters: X is left/right, Y is vertical up (ground floor is Y = 0), Z is
  depth (negative Z is in front of the default camera).
- Make camera motion smooth and continuous across keyframes over the shot duration: no jitter,
  no sudden jumps, and never place the camera inside or passing through solid objects.
- Give the primary singer a dark charcoal color (around [38, 42, 52]) and any secondary figures
  or extras a lighter neutral grey (around [155, 160, 170]) so the stand-in mapping is clear.
- Write a concise 'stand_in_mapping' sentence in English explaining which colored blocks represent
  the real subjects, extras, and props."""


class _PrevizBase(BaseModel):
    model_config = ConfigDict(extra="ignore", allow_inf_nan=False)


RGBChannel = Annotated[int, Field(ge=0, le=255)]
RGBColor = tuple[RGBChannel, RGBChannel, RGBChannel]
PositiveExtent = Annotated[float, Field(gt=0.0, le=100.0)]


class BlockingKeyframe(_PrevizBase):
    time: float = Field(ge=0.0)
    position: tuple[float, float, float]
    yaw_deg: float = 0.0
    scale: float = Field(default=1.0, gt=0.05, le=10.0)
    visible: bool = True


class BlockingActor(_PrevizBase):
    id: str
    kind: Literal["figure", "box", "cylinder", "ring", "stage"] = "box"
    role: Literal["primary", "extra", "set", "prop"] = "set"
    color: RGBColor = (90, 100, 125)
    size: tuple[PositiveExtent, PositiveExtent, PositiveExtent] = (0.6, 1.7, 0.4)
    keyframes: list[BlockingKeyframe] = Field(default_factory=list)


class CameraKeyframe(_PrevizBase):
    time: float = Field(ge=0.0)
    eye: tuple[float, float, float]
    look_at: tuple[float, float, float]
    fov_deg: float = Field(default=48.0, ge=18.0, le=100.0)


class PrevizPlan(_PrevizBase):
    background_color: RGBColor = (18, 22, 32)
    grid_color: RGBColor = (48, 125, 145)
    stand_in_mapping: str = ""
    actors: list[BlockingActor] = Field(default_factory=list)
    camera: list[CameraKeyframe] = Field(default_factory=list)


_CAMERA_BY_FRAMING: dict[Framing, tuple[tuple[float, float, float], tuple[float, float, float]]] = {
    Framing.CLOSE_UP: ((0.0, 1.48, 1.35), (0.0, 1.45, 0.0)),
    Framing.MEDIUM_CLOSE: ((0.15, 1.42, 2.2), (-0.1, 1.35, 0.0)),
    Framing.MEDIUM: ((0.4, 1.38, 3.6), (-0.3, 1.25, 0.0)),
    Framing.WIDE: ((1.2, 2.2, 6.2), (-0.4, 1.2, 0.0)),
}


def default_previz_plan(shot: Shot, manifest: Manifest) -> PrevizPlan:
    """Deterministic flat-shaded blocking layout matching shot framing and cast."""
    duration = max(0.5, round(shot.audio_slice.end - shot.audio_slice.start, 3))
    actors: list[BlockingActor] = [
        BlockingActor(
            id="stage",
            kind="stage",
            role="set",
            color=(32, 38, 50),
            size=(4.6, 0.14, 4.6),
            keyframes=[BlockingKeyframe(time=0.0, position=(0.0, 0.07, 0.0))],
        ),
        BlockingActor(
            id="pillar_left",
            kind="box",
            role="set",
            color=(64, 76, 102),
            size=(0.8, 3.2, 0.8),
            keyframes=[BlockingKeyframe(time=0.0, position=(-2.8, 1.6, -2.5))],
        ),
        BlockingActor(
            id="pillar_right",
            kind="box",
            role="set",
            color=(64, 76, 102),
            size=(0.8, 3.2, 0.8),
            keyframes=[BlockingKeyframe(time=0.0, position=(2.8, 1.6, -2.5))],
        ),
    ]
    mapping_parts: list[str] = []
    if shot.type.shows_singer:
        visible = manifest.visible_characters(shot)
        primary_label = (
            (visible[0][1].name or visible[0][0] or "the singer") if visible else "the singer"
        )
        actors.append(
            BlockingActor(
                id="primary_singer",
                kind="figure",
                role="primary",
                color=(38, 42, 52),
                size=(0.55, 1.72, 0.35),
                keyframes=[
                    BlockingKeyframe(time=0.0, position=(0.0, 0.0, 0.0), yaw_deg=0.0),
                    BlockingKeyframe(time=duration, position=(0.0, 0.0, 0.0), yaw_deg=8.0),
                ],
            )
        )
        mapping_parts.append(f"The dark figure in the center becomes {primary_label}.")
        for idx, (cid, char, _) in enumerate(visible[1:], start=1):
            label = char.name or cid or f"performer {idx + 1}"
            x_offset = (-1.1 if idx % 2 else 1.1) * ((idx + 1) // 2)
            actors.append(
                BlockingActor(
                    id=f"extra_{idx}",
                    kind="figure",
                    role="extra",
                    color=(155, 160, 170),
                    size=(0.52, 1.68, 0.34),
                    keyframes=[
                        BlockingKeyframe(time=0.0, position=(x_offset, 0.0, -0.6), yaw_deg=0.0)
                    ],
                )
            )
            mapping_parts.append(f"The light grey figure at the side becomes {label}.")
    if shot.type is ShotType.BROLL:
        mapping_parts.append("The geometric blocks define architectural framing and scale.")

    start_eye, end_eye = _CAMERA_BY_FRAMING[shot.framing]
    target_y = 1.48 if shot.framing in (Framing.CLOSE_UP, Framing.MEDIUM_CLOSE) else 1.1
    return PrevizPlan(
        stand_in_mapping=" ".join(mapping_parts),
        actors=actors,
        camera=[
            CameraKeyframe(time=0.0, eye=start_eye, look_at=(0.0, target_y, 0.0), fov_deg=46.0),
            CameraKeyframe(
                time=duration,
                eye=(end_eye[0], end_eye[1], max(1.15, end_eye[2] * 0.85)),
                look_at=(0.0, target_y, 0.0),
                fov_deg=44.0,
            ),
        ],
    )


def _shot_previz_prompt(shot: Shot, manifest: Manifest) -> str:
    bible = manifest.visual_bible
    scene = bible.find_set(shot.setting)
    duration = max(0.5, round(shot.audio_slice.end - shot.audio_slice.start, 3))
    cast = [
        f"{cid or 'singer'} ({char.name or 'lead'})"
        for cid, char, _ in manifest.visible_characters(shot)
    ]
    return "\n".join(
        [
            f"Shot ID: {shot.id}",
            f"Duration: {duration:.2f}s",
            f"Type: {shot.type.value}   Framing: {shot.framing.value}",
            f"Setting: {shot.setting}" + (f" — {scene.description}" if scene else ""),
            f"Visible cast: {', '.join(cast) if cast else 'none (b-roll)'}",
            f"First frame: {shot.still_prompt}",
            f"Motion & camera: {shot.motion_prompt}",
        ]
    )


def plan_previz(
    shot: Shot,
    manifest: Manifest,
    llm: LLMClient | None,
    *,
    set_plate_path: Path | None = None,
) -> PrevizPlan:
    """Design a 3D blocking plan for ``shot`` via LLM (using set plate if present) or fallback."""
    if llm is None:
        return default_previz_plan(shot, manifest)

    content = []
    if set_plate_path is not None and set_plate_path.is_file():
        content.append(text_block("SET PLATE IMAGE (match this room layout and proportions):"))
        content.append(image_block(set_plate_path))
    content.append(text_block(_shot_previz_prompt(shot, manifest)))
    try:
        plan = llm.complete_json(system=PREVIZ_SYSTEM, content=content, schema=PrevizPlan)
        if not plan.actors or len(plan.camera) < 1:
            log.warning("%s: LLM previz plan was incomplete; using default blocking", shot.id)
            return default_previz_plan(shot, manifest)
        return plan
    except LLMError as exc:
        log.warning("%s: previz LLM planning failed (%s); using default blocking", shot.id, exc)
        return default_previz_plan(shot, manifest)


def refine_previz(
    shot: Shot,
    manifest: Manifest,
    current_plan: PrevizPlan,
    feedback: str,
    llm: LLMClient | None,
    *,
    set_plate_path: Path | None = None,
) -> PrevizPlan:
    """Update an existing 3D blocking plan according to explicit user camera/staging feedback."""
    cleaned = feedback.strip()
    if not cleaned:
        raise ConfigError("Provide non-empty --feedback to refine the previz plan.")
    if llm is None:
        raise ConfigError("`mvd previz refine` requires llm.enabled: true in project.yaml.")

    content = []
    if set_plate_path is not None and set_plate_path.is_file():
        content.append(text_block("SET PLATE IMAGE (match this room layout and proportions):"))
        content.append(image_block(set_plate_path))
    content.append(
        text_block(
            f"{_shot_previz_prompt(shot, manifest)}\n\n"
            f"CURRENT 3D BLOCKING PLAN:\n{current_plan.model_dump_json(indent=2)}\n\n"
            f"REVISION FEEDBACK:\n{cleaned}"
        )
    )
    refined = llm.complete_json(system=PREVIZ_SYSTEM, content=content, schema=PrevizPlan)
    if not refined.actors or len(refined.camera) < 1:
        raise ConfigError(f"{shot.id}: LLM returned an incomplete refined previz plan.")
    return refined
