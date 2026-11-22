# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Render and lock 3D blocking previz videos for shot camera/staging guidance."""

from __future__ import annotations

import logging
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from pydantic import ValidationError

from mvdirector import media
from mvdirector.errors import ConfigError, MediaToolError
from mvdirector.llm import LLMClient
from mvdirector.models import Manifest, Shot
from mvdirector.planning.previz import (
    BlockingActor,
    BlockingKeyframe,
    CameraKeyframe,
    PrevizPlan,
    plan_previz,
    refine_previz,
)
from mvdirector.project import Project

log = logging.getLogger(__name__)

_NEAR_Z = 0.15
_LIGHT_DIR = np.array([0.45, 0.82, 0.35], dtype=np.float64)
_LIGHT_DIR /= np.linalg.norm(_LIGHT_DIR)
_ALLOWED_VIDEO_EXTENSIONS = frozenset({".mp4", ".mov", ".webm", ".mkv"})


def _smoothstep(u: float) -> float:
    u = max(0.0, min(1.0, u))
    return u * u * (3.0 - 2.0 * u)


def _eval_camera(keyframes: list[CameraKeyframe], t: float) -> tuple[np.ndarray, np.ndarray, float]:
    if not keyframes:
        return np.array([0.0, 1.5, 4.0]), np.array([0.0, 1.1, 0.0]), 46.0
    ordered = sorted(keyframes, key=lambda k: k.time)
    if len(ordered) == 1 or t <= ordered[0].time:
        k = ordered[0]
        return np.array(k.eye, dtype=np.float64), np.array(k.look_at, dtype=np.float64), k.fov_deg
    if t >= ordered[-1].time:
        k = ordered[-1]
        return np.array(k.eye, dtype=np.float64), np.array(k.look_at, dtype=np.float64), k.fov_deg
    for k0, k1 in zip(ordered[:-1], ordered[1:]):
        if k0.time <= t <= k1.time:
            span = max(1e-6, k1.time - k0.time)
            alpha = _smoothstep((t - k0.time) / span)
            eye = (1.0 - alpha) * np.array(k0.eye) + alpha * np.array(k1.eye)
            look = (1.0 - alpha) * np.array(k0.look_at) + alpha * np.array(k1.look_at)
            fov = (1.0 - alpha) * k0.fov_deg + alpha * k1.fov_deg
            return eye, look, fov
    k = ordered[-1]
    return np.array(k.eye, dtype=np.float64), np.array(k.look_at, dtype=np.float64), k.fov_deg


def _eval_actor(
    keyframes: list[BlockingKeyframe], t: float
) -> tuple[np.ndarray, float, float, bool]:
    if not keyframes:
        return np.zeros(3, dtype=np.float64), 0.0, 1.0, True
    ordered = sorted(keyframes, key=lambda k: k.time)
    if len(ordered) == 1 or t <= ordered[0].time:
        k = ordered[0]
        return np.array(k.position, dtype=np.float64), k.yaw_deg, k.scale, k.visible
    if t >= ordered[-1].time:
        k = ordered[-1]
        return np.array(k.position, dtype=np.float64), k.yaw_deg, k.scale, k.visible
    for k0, k1 in zip(ordered[:-1], ordered[1:]):
        if k0.time <= t <= k1.time:
            span = max(1e-6, k1.time - k0.time)
            alpha = _smoothstep((t - k0.time) / span)
            pos = (1.0 - alpha) * np.array(k0.position) + alpha * np.array(k1.position)
            yaw = (1.0 - alpha) * k0.yaw_deg + alpha * k1.yaw_deg
            scale = (1.0 - alpha) * k0.scale + alpha * k1.scale
            return pos, yaw, scale, k0.visible
    k = ordered[-1]
    return np.array(k.position, dtype=np.float64), k.yaw_deg, k.scale, k.visible


def _box_triangles(
    center: np.ndarray, size: tuple[float, float, float], yaw_rad: float
) -> list[np.ndarray]:
    hx, hy, hz = size[0] * 0.5, size[1] * 0.5, size[2] * 0.5
    local = np.array(
        [
            [-hx, -hy, -hz],
            [hx, -hy, -hz],
            [hx, hy, -hz],
            [-hx, hy, -hz],
            [-hx, -hy, hz],
            [hx, -hy, hz],
            [hx, hy, hz],
            [-hx, hy, hz],
        ],
        dtype=np.float64,
    )
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    rot = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)
    world = local @ rot.T + center
    faces = [
        (0, 1, 2, 3),  # -z
        (5, 4, 7, 6),  # +z
        (4, 0, 3, 7),  # -x
        (1, 5, 6, 2),  # +x
        (3, 2, 6, 7),  # +y
        (4, 5, 1, 0),  # -y
    ]
    tris: list[np.ndarray] = []
    for a, b, c_i, d in faces:
        tris.append(world[[a, b, c_i]])
        tris.append(world[[a, c_i, d]])
    return tris


def _actor_triangles(actor: BlockingActor, t: float) -> tuple[list[np.ndarray], bool]:
    pos, yaw_deg, scale, visible = _eval_actor(actor.keyframes, t)
    if not visible:
        return [], False
    sx, sy, sz = (dim * scale for dim in actor.size)
    yaw = math.radians(yaw_deg)
    if actor.kind == "figure":
        # pos is feet-on-ground center; stack torso + head for unambiguous facing silhouette.
        torso = _box_triangles(pos + np.array([0.0, sy * 0.41, 0.0]), (sx, sy * 0.82, sz), yaw)
        head_s = min(sx, sz) * 0.72
        head = _box_triangles(
            pos + np.array([0.0, sy * 0.91, 0.0]), (head_s, sy * 0.18, head_s), yaw
        )
        return torso + head, False
    if actor.kind == "ring":
        # Approximate a horizontal/tilted flat ring as 8 segments around center.
        tris: list[np.ndarray] = []
        r = max(sx, sz) * 0.5
        seg_w = max(0.06, r * 0.45)
        for idx in range(8):
            ang = yaw + idx * math.pi / 4.0
            offset = np.array([math.cos(ang) * r, 0.0, math.sin(ang) * r], dtype=np.float64)
            tris.extend(_box_triangles(pos + offset, (seg_w, sy, seg_w * 0.55), -ang))
        return tris, False
    return _box_triangles(pos, (sx, sy, sz), yaw), actor.kind == "stage"


def _camera_basis(
    eye: np.ndarray, look_at: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    forward = look_at - eye
    norm = float(np.linalg.norm(forward))
    forward = np.array([0.0, 0.0, -1.0]) if norm < 1e-6 else forward / norm
    world_up = (
        np.array([0.0, 0.0, -1.0])
        if abs(float(np.dot(forward, np.array([0.0, 1.0, 0.0])))) > 0.99
        else np.array([0.0, 1.0, 0.0])
    )
    right = np.cross(forward, world_up)
    right /= max(1e-6, float(np.linalg.norm(right)))
    up = np.cross(right, forward)
    return right, up, forward


def render_blocking_frame(
    plan: PrevizPlan,
    t: float,
    width: int,
    height: int,
    *,
    depth_pass: bool = False,
    depth_range_m: float = 16.0,
) -> Image.Image:
    """Render one flat-shaded 3D blocking frame (or linear view-space depth pass) at ``t``.

    When ``depth_pass=True``, renders linear view-space depth mapped ``white=near`` (255) to
    ``black=far`` (0) over ``[0, depth_range_m]`` meters for depth-ControlNet / IC-LoRA / R2V
    video conditioning.

    Credits & Attribution:
        Linear view-space depth pass alongside 3D camera previz (clean-room reimplementation)
        inspired by ``heroesjouney/aimoviestudiov2``
        (https://github.com/heroesjouney/aimoviestudiov2 — ``ViewfinderScene.tsx``).
    """
    if width <= 0 or height <= 0:
        raise ConfigError(f"Previz frame dimensions must be positive, got {width}x{height}.")
    bg_color = (0, 0, 0) if depth_pass else tuple(plan.background_color)
    img = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(img)

    eye, look_at, fov_deg = _eval_camera(plan.camera, t)
    right, up, forward = _camera_basis(eye, look_at)
    focal = (0.5 * height) / math.tan(math.radians(fov_deg) * 0.5)
    cx, cy = width * 0.5, height * 0.5

    def to_cam(pt: np.ndarray) -> np.ndarray:
        d = pt - eye
        return np.array([float(np.dot(d, right)), float(np.dot(d, up)), float(np.dot(d, forward))])

    def project(cam_pt: np.ndarray) -> tuple[float, float]:
        z = cam_pt[2]
        return cx + (cam_pt[0] / z) * focal, cy - (cam_pt[1] / z) * focal

    # Ground plane perspective grid (Y = 0) — omitted in clean linear depth pass.
    if not depth_pass:
        grid_rgb = tuple(plan.grid_color)
        for coord in range(-8, 9, 2):
            for p0_w, p1_w in (
                (np.array([float(coord), 0.0, -10.0]), np.array([float(coord), 0.0, 10.0])),
                (np.array([-10.0, 0.0, float(coord)]), np.array([10.0, 0.0, float(coord)])),
            ):
                c0, c1 = to_cam(p0_w), to_cam(p1_w)
                if c0[2] <= _NEAR_Z and c1[2] <= _NEAR_Z:
                    continue
                if c0[2] <= _NEAR_Z:
                    frac = (_NEAR_Z - c0[2]) / max(1e-6, c1[2] - c0[2])
                    c0 = c0 + frac * (c1 - c0)
                elif c1[2] <= _NEAR_Z:
                    frac = (_NEAR_Z - c1[2]) / max(1e-6, c0[2] - c1[2])
                    c1 = c1 + frac * (c0 - c1)
                draw.line([project(c0), project(c1)], fill=grid_rgb, width=1)

    stage_polys: list[tuple[float, list[tuple[float, float]], tuple[int, int, int]]] = []
    actor_polys: list[tuple[float, list[tuple[float, float]], tuple[int, int, int]]] = []
    safe_range = max(1.0, float(depth_range_m))
    base_colors = tuple(plan.actors)
    for actor in base_colors:
        tris, is_stage = _actor_triangles(actor, t)
        r, g, b = actor.color
        for tri in tris:
            c_pts = np.array([to_cam(tri[0]), to_cam(tri[1]), to_cam(tri[2])])
            if np.any(c_pts[:, 2] <= _NEAR_Z):
                continue
            depth = float(c_pts[:, 2].mean())
            if depth_pass:
                norm_d = max(0.0, min(1.0, 1.0 - (depth - _NEAR_Z) / safe_range))
                grey = int(round(norm_d * 255.0))
                color = (grey, grey, grey)
            else:
                normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
                norm = float(np.linalg.norm(normal))
                shade = 0.72
                if norm > 1e-8:
                    normal /= norm
                    shade = 0.45 + 0.55 * max(0.0, float(np.dot(normal, _LIGHT_DIR)))
                color = (
                    min(255, int(r * shade)),
                    min(255, int(g * shade)),
                    min(255, int(b * shade)),
                )
            poly = [project(c_pts[0]), project(c_pts[1]), project(c_pts[2])]
            (stage_polys if is_stage else actor_polys).append((depth, poly, color))

    for bucket in (stage_polys, actor_polys):
        bucket.sort(key=lambda item: item[0], reverse=True)
        for _, poly, color in bucket:
            draw.polygon(poly, fill=color)
    return img


def render_blocking_video(
    plan: PrevizPlan,
    dst: Path,
    duration: float,
    fps: int,
    width: int = 640,
    height: int = 360,
    *,
    depth_pass: bool = False,
    depth_range_m: float = 16.0,
) -> Path:
    """Render ``plan`` across ``duration`` seconds and encode to MP4 via ffmpeg."""
    media.require_tool("ffmpeg")
    # Ensure even dimensions for yuv420p H.264.
    even_w = max(64, width - (width % 2))
    even_h = max(64, height - (height % 2))
    total_frames = max(2, int(round(duration * fps)))
    dst.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{even_w}x{even_h}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(dst.resolve()),
    ]
    with subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE) as proc:
        assert proc.stdin is not None
        try:
            for frame_idx in range(total_frames):
                t = (frame_idx / max(1, total_frames - 1)) * duration
                frame = render_blocking_frame(
                    plan,
                    t,
                    even_w,
                    even_h,
                    depth_pass=depth_pass,
                    depth_range_m=depth_range_m,
                )
                proc.stdin.write(frame.tobytes())
            proc.stdin.close()
            stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
            returncode = proc.wait()
        except (OSError, ValueError) as exc:
            proc.kill()
            proc.wait()
            raise MediaToolError(f"ffmpeg previz encode failed: {exc}") from exc
    if returncode != 0:
        raise MediaToolError(f"ffmpeg previz encode failed (exit {returncode}):\n{stderr}")
    return dst


def _render_and_save_plan(
    project: Project,
    shot: Shot,
    plan: PrevizPlan,
    *,
    overwrite_mapping: bool = False,
) -> Path:
    video = project.config.video
    scale = min(1.0, 640.0 / max(1, video.width))
    render_w = max(64, int(round(video.width * scale)))
    render_h = max(64, int(round(video.height * scale)))
    dst = project.previz_path(shot.id)
    depth_dst = dst.with_name(f"{dst.stem}_depth.mp4")
    duration = max(0.5, shot.audio_slice.end - shot.audio_slice.start)
    render_blocking_video(plan, dst, duration, video.fps, width=render_w, height=render_h)
    media.ensure_audio_track(dst, dst)
    render_blocking_video(
        plan, depth_dst, duration, video.fps, width=render_w, height=render_h, depth_pass=True
    )
    media.ensure_audio_track(depth_dst, depth_dst)
    plan_path = project.previz_plan_path(shot.id)
    project.write_text(project.rel(plan_path), plan.model_dump_json(indent=2) + "\n")
    shot.previz_video = project.rel(dst)
    shot.previz_depth_video = project.rel(depth_dst)
    if plan.stand_in_mapping and (overwrite_mapping or not shot.previz_mapping):
        shot.previz_mapping = plan.stand_in_mapping
    return dst


def generate_previz(
    project: Project,
    manifest: Manifest,
    llm: LLMClient | None,
    *,
    shot_ids: list[str] | None = None,
    overwrite: bool = False,
) -> list[Path]:
    """Plan and render 3D blocking previz MP4s for the selected shots."""
    written: list[Path] = []
    for shot in manifest.shots:
        if shot_ids is not None and shot.id not in shot_ids:
            continue
        if shot.previz_video and project.path(shot.previz_video).is_file() and not overwrite:
            log.info("%s already has previz (%s); skipping", shot.id, shot.previz_video)
            continue
        scene = manifest.visual_bible.find_set(shot.setting)
        plate_path = project.path(scene.plate) if (scene and scene.plate) else None
        plan = plan_previz(shot, manifest, llm, set_plate_path=plate_path)
        written.append(_render_and_save_plan(project, shot, plan))
    return written


def refine_previz_shot(
    project: Project,
    manifest: Manifest,
    llm: LLMClient | None,
    shot_id: str,
    feedback: str,
) -> Path:
    """Refine a shot's saved 3D blocking plan with user feedback and re-render its MP4."""
    if not feedback.strip():
        raise ConfigError("Provide non-empty --feedback to refine the previz plan.")
    if llm is None:
        raise ConfigError("`mvd previz refine` requires llm.enabled: true in project.yaml.")
    try:
        shot = manifest.shot(shot_id)
    except KeyError as exc:
        raise ConfigError(f"No shot with id '{shot_id}'") from exc
    plan_path = project.previz_plan_path(shot.id)
    if not plan_path.is_file():
        raise ConfigError(
            f"No editable 3D blocking plan found for '{shot.id}'. "
            f"Run `mvd previz generate --shot {shot.id}` first."
        )
    try:
        current_plan = PrevizPlan.model_validate_json(plan_path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as exc:
        raise ConfigError(f"Could not read previz plan {plan_path}: {exc}") from exc

    scene = manifest.visual_bible.find_set(shot.setting)
    plate_path = project.path(scene.plate) if (scene and scene.plate) else None
    updated = refine_previz(shot, manifest, current_plan, feedback, llm, set_plate_path=plate_path)
    return _render_and_save_plan(project, shot, updated, overwrite_mapping=True)


def lock_previz(
    project: Project,
    manifest: Manifest,
    video_path: Path,
    shot_id: str,
    *,
    mapping: str | None = None,
) -> Path:
    """Validate and attach an external 3D blocking video (e.g. from Three.js or Blender)."""
    if not video_path.is_file():
        raise ConfigError(f"Previz video not found: {video_path}")
    suffix = video_path.suffix.lower()
    if suffix not in _ALLOWED_VIDEO_EXTENSIONS:
        allowed = ", ".join(sorted(_ALLOWED_VIDEO_EXTENSIONS))
        raise ConfigError(
            f"Unsupported previz video extension {suffix!r}; expected one of: {allowed}."
        )
    info = media.probe(video_path)
    if info.width is None or info.height is None:
        raise ConfigError(f"Previz file has no video stream: {video_path}")
    try:
        shot = manifest.shot(shot_id)
    except KeyError as exc:
        raise ConfigError(f"No shot with id '{shot_id}'") from exc
    dst = project.previz_path(shot.id, suffix=suffix)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if video_path.resolve() != dst.resolve():
        shutil.copyfile(video_path, dst)
    shot.previz_video = project.rel(dst)
    # A depth pass rendered for earlier auto-previz shows a different camera move; sending both
    # would give the video model two conflicting guides.
    shot.previz_depth_video = None
    if mapping is not None:
        shot.previz_mapping = mapping
    return dst
