# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Thin, testable wrappers around ffmpeg and ffprobe."""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np
from PIL import Image

from mvdirector.errors import MediaToolError

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class MediaInfo:
    duration: float
    width: int | None
    height: int | None
    fps: float | None
    has_audio: bool


def require_tool(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise MediaToolError(
            f"'{name}' was not found on PATH. Install ffmpeg (brew install ffmpeg)."
        )
    return path


def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    """Run a command, raising MediaToolError with the tail of stderr on failure."""
    require_tool(cmd[0])
    log.debug("exec: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        tail = "\n".join(result.stderr.strip().splitlines()[-8:])
        raise MediaToolError(f"{cmd[0]} failed (exit {result.returncode}):\n{tail}")
    return result


def probe(path: Path) -> MediaInfo:
    result = run(
        [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(path.resolve()),
        ]
    )
    data = json.loads(result.stdout)
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    fps = None
    if video and video.get("avg_frame_rate") not in (None, "0/0"):
        fps = float(Fraction(video["avg_frame_rate"]))
    return MediaInfo(
        duration=float(data.get("format", {}).get("duration", 0.0)),
        width=int(video["width"]) if video else None,
        height=int(video["height"]) if video else None,
        fps=fps,
        has_audio=any(s.get("codec_type") == "audio" for s in streams),
    )


def transcode_to_wav(src: Path, dst: Path) -> Path:
    """Decode an audio file (.mp3, .wav, etc.) to a 16-bit PCM WAV atomically."""
    if not src.is_file():
        raise MediaToolError(f"Audio file not found: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".mvd_wav_", suffix=".wav", dir=dst.parent)
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        run(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-i",
                str(src.resolve()),
                "-vn",
                "-c:a",
                "pcm_s16le",
                str(tmp_path.resolve()),
            ]
        )
        os.replace(tmp_path, dst)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    return dst


def slice_audio(src: Path, dst: Path, start: float, end: float) -> None:
    """Cut [start, end] seconds of ``src`` into a mono-compatible PCM wav."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-ss",
            f"{start:.4f}",
            "-t",
            f"{end - start:.4f}",
            "-i",
            str(src.resolve()),
            "-c:a",
            "pcm_s16le",
            str(dst.resolve()),
        ]
    )


def audio_to_black_video(
    audio: Path, dst: Path, width: int = 64, height: int = 64, fps: int = 24
) -> None:
    """Wrap an audio clip in a lightweight black-screen MP4 for video-input lip-sync models."""
    if not audio.is_file():
        raise MediaToolError(f"Audio clip not found: {audio}")
    if width < 2 or height < 2 or width % 2 or height % 2:
        raise MediaToolError(f"Video dimensions must be even integers >= 2, got {width}x{height}.")
    if fps <= 0:
        raise MediaToolError(f"FPS must be positive, got {fps}.")
    dst.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"color=c=black:s={width}x{height}:r={fps}",
            "-i",
            str(audio.resolve()),
            "-shortest",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            str(dst.resolve()),
        ]
    )


def extract_frames(video: Path, width: int, fps: float | None = None) -> list[np.ndarray]:
    """Decode a video into RGB uint8 arrays, optionally resampled to ``fps``."""
    filters = [f"scale={width}:-2"]
    if fps is not None:
        filters.insert(0, f"fps={fps}")
    with tempfile.TemporaryDirectory(prefix="mvd_frames_") as tmp:
        run(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-i",
                str(video.resolve()),
                "-vf",
                ",".join(filters),
                str(Path(tmp) / "f_%06d.png"),
            ]
        )
        frames = []
        for frame_path in sorted(Path(tmp).glob("f_*.png")):
            with Image.open(frame_path) as image:
                frames.append(np.asarray(image.convert("RGB")))
    if not frames:
        raise MediaToolError(f"No frames could be decoded from {video}")
    return frames
