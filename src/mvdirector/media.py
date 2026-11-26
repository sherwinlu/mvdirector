# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Thin, testable wrappers around ffmpeg and ffprobe."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np
import soundfile as sf
from PIL import Image

from mvdirector.errors import MediaToolError

log = logging.getLogger(__name__)

_HW_ENCODERS = ("h264_nvenc", "h264_videotoolbox")
_ENCODERS_CACHE: frozenset[str] | None = None


def clear_encoder_cache() -> None:
    """Clear the cached encoder probe result (intended for tests)."""
    global _ENCODERS_CACHE
    _ENCODERS_CACHE = None


def _probe_encoder_works(encoder: str) -> bool:
    """Run a 1-frame null-mux smoke test to confirm a listed hardware encoder actually works."""
    try:
        res = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "color=c=black:s=64x64:d=0.04",
                "-frames:v",
                "1",
                "-c:v",
                encoder,
                "-f",
                "null",
                "-",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return res.returncode == 0


def available_encoders() -> frozenset[str]:
    """Return verified H.264 encoders usable on this host (cached)."""
    global _ENCODERS_CACHE
    if _ENCODERS_CACHE is not None:
        return _ENCODERS_CACHE
    found: set[str] = set()
    try:
        out = run(["ffmpeg", "-hide_banner", "-encoders"], timeout=10).stdout
    except MediaToolError:
        _ENCODERS_CACHE = frozenset({"libx264"})
        return _ENCODERS_CACHE
    listed: set[str] = set()
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].startswith("V"):
            listed.add(parts[1])
    if "libx264" in listed:
        found.add("libx264")
    for hw in _HW_ENCODERS:
        if hw in listed and _probe_encoder_works(hw):
            found.add(hw)
    if not found:
        found.add("libx264")
    _ENCODERS_CACHE = frozenset(found)
    return _ENCODERS_CACHE


def resolve_video_encoder(requested: str) -> str:
    """Resolve ``libx264``/``auto``/HW encoder name to a verified encoder on this host."""
    if requested == "libx264":
        return "libx264"
    avail = available_encoders()
    if requested == "auto":
        for hw in _HW_ENCODERS:
            if hw in avail:
                return hw
        return "libx264"
    if requested in avail:
        return requested
    log.warning(
        "Requested video encoder %r is unavailable on this host; falling back to libx264.",
        requested,
    )
    return "libx264"


def encode_args_for(encoder: str) -> list[str]:
    """Return ffmpeg video-encoding flags for ``encoder``."""
    if encoder == "h264_nvenc":
        return ["-an", "-c:v", "h264_nvenc", "-preset", "p4", "-cq", "16", "-pix_fmt", "yuv420p"]
    if encoder == "h264_videotoolbox":
        return ["-an", "-c:v", "h264_videotoolbox", "-q:v", "75", "-pix_fmt", "yuv420p"]
    return ["-an", "-c:v", "libx264", "-preset", "medium", "-crf", "16", "-pix_fmt", "yuv420p"]


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


def run(cmd: list[str], *, timeout: float | None = None) -> subprocess.CompletedProcess[str]:
    """Run a command, raising MediaToolError with the tail of stderr on failure."""
    require_tool(cmd[0])
    log.debug("exec: %s", " ".join(cmd))
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise MediaToolError(f"{cmd[0]} timed out after {timeout:.0f}s") from exc
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
            "-protocol_whitelist",
            "file",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            "--",
            str(path.resolve()),
        ]
    )
    try:
        data = json.loads(result.stdout)
    except ValueError as exc:
        raise MediaToolError(f"ffprobe returned invalid JSON for {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise MediaToolError(f"ffprobe returned unexpected JSON for {path}")
    raw_streams = data.get("streams", [])
    streams = raw_streams if isinstance(raw_streams, list) else []
    video = next(
        (s for s in streams if isinstance(s, dict) and s.get("codec_type") == "video"), None
    )
    fps: float | None = None
    if video and video.get("avg_frame_rate") not in (None, "", "0/0", "N/A"):
        try:
            frac = Fraction(str(video["avg_frame_rate"]))
            if frac > 0:
                fps = float(frac)
        except (ValueError, ZeroDivisionError):
            fps = None
    raw_duration = (data.get("format") or {}).get("duration", 0.0)
    try:
        duration = float(raw_duration)
        if not math.isfinite(duration) or duration < 0.0:
            duration = 0.0
    except (TypeError, ValueError):
        duration = 0.0
    width: int | None = None
    height: int | None = None
    if video is not None:
        try:
            if video.get("width") is not None:
                width = int(video["width"])
            if video.get("height") is not None:
                height = int(video["height"])
        except (TypeError, ValueError):
            width = height = None
    return MediaInfo(
        duration=duration,
        width=width,
        height=height,
        fps=fps,
        has_audio=any(isinstance(s, dict) and s.get("codec_type") == "audio" for s in streams),
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
                "-protocol_whitelist",
                "file",
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
    """Cut [start, end] seconds of ``src`` into a mono-compatible PCM wav atomically."""
    if not (math.isfinite(start) and math.isfinite(end)) or start < 0.0 or end <= start:
        raise MediaToolError(f"Invalid audio slice span [{start}, {end}].")
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".mvd_slice_", suffix=".wav", dir=dst.parent)
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        run(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-protocol_whitelist",
                "file",
                "-ss",
                f"{start:.4f}",
                "-t",
                f"{end - start:.4f}",
                "-i",
                str(src.resolve()),
                "-c:a",
                "pcm_s16le",
                str(tmp_path.resolve()),
            ]
        )
        os.replace(tmp_path, dst)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def mute_audio_spans(
    wav_path: Path, spans: list[tuple[float, float]], *, fade_seconds: float = 0.015
) -> None:
    """Silence relative ``(start, end)`` spans in ``wav_path`` in-place with smooth cosine fades."""
    if not spans:
        return
    info = sf.info(str(wav_path))
    samples, sr = sf.read(str(wav_path), dtype="float64", always_2d=True)
    n = len(samples)
    if n == 0:
        return
    mask = np.ones(n, dtype=bool)
    for rel_start, rel_end in spans:
        if not (math.isfinite(rel_start) and math.isfinite(rel_end)):
            continue
        s = max(0, min(n, int(round(rel_start * sr))))
        e = max(s, min(n, int(round(rel_end * sr))))
        if e > s:
            mask[s:e] = False
    if mask.all():
        return

    gain = mask.astype(np.float64)
    safe_fade = fade_seconds if math.isfinite(fade_seconds) and fade_seconds >= 0.0 else 0.0
    fade_len = max(1, int(round(safe_fade * sr)))
    padded = np.concatenate([[True], mask, [True]])
    starts = np.flatnonzero(padded[:-1] & ~padded[1:])
    ends = np.flatnonzero(~padded[:-1] & padded[1:])
    for run_s, run_e in zip(starts.tolist(), ends.tolist()):
        if run_s > 0:
            lo = max(0, run_s - fade_len)
            ramp_down = 0.5 * (1.0 + np.cos(np.linspace(0.0, np.pi, run_s - lo)))
            gain[lo:run_s] = np.minimum(gain[lo:run_s], ramp_down)
        if run_e < n:
            hi = min(n, run_e + fade_len)
            ramp_up = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, hi - run_e)))
            gain[run_e:hi] = np.minimum(gain[run_e:hi], ramp_up)

    gated = samples * gain[:, None]
    fd, tmp_name = tempfile.mkstemp(prefix=".mvd_mute_", suffix=".wav", dir=wav_path.parent)
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        sf.write(str(tmp_path), gated, sr, subtype=info.subtype or "PCM_16")
        os.replace(tmp_path, wav_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


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
    fd, tmp_name = tempfile.mkstemp(prefix=".mvd_black_", suffix=".mp4", dir=dst.parent)
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
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
                "-protocol_whitelist",
                "file",
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
                str(tmp_path.resolve()),
            ]
        )
        os.replace(tmp_path, dst)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


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
                "-protocol_whitelist",
                "file",
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


def mirror_pad_audio(samples: np.ndarray, target_length: int) -> np.ndarray:
    """Extend ``samples`` to ``target_length`` using a periodic triangle-wave mirror gather.

    Avoids zero-padding silence cliffs at clip tails even when the requested padding exceeds
    the original signal length.

    Credits & Attribution:
        Clean-room reimplementation of periodic mirror tail padding inspired by
        ``SonderSaid/ComfyUI-Sonder-Editor``
        (https://github.com/SonderSaid/ComfyUI-Sonder-Editor — ``server/audio_grid.py``).
    """
    arr = np.asarray(samples)
    n = arr.shape[0]
    if target_length <= n or n == 0:
        return arr[: max(0, target_length)].copy()
    if n == 1:
        reps = (target_length, *([1] * (arr.ndim - 1)))
        return np.tile(arr, reps)

    # Triangle-wave fold of sample indices onto [0, n - 1] with period 2*(n - 1).
    period = 2 * (n - 1)
    idx = np.arange(target_length, dtype=np.int64) % period
    folded = np.where(idx < n, idx, period - idx)
    return arr[folded]


def prepare_audio_slice(
    src: Path,
    dst: Path,
    start: float,
    end: float,
    *,
    target_duration: float | None = None,
    vae_offset_seconds: float = 0.0,
    mirror_pad: bool = True,
) -> Path:
    """Slice ``[start, end]`` with optional rational VAE front-crop offset & mirror tail padding.

    When ``vae_offset_seconds`` is non-zero (e.g. ``0.046`` for LTX-2.3/2.5 or ``0.0125`` for
    MiniMax H3), shifts the cut window using exact ``Fraction`` sample arithmetic so ComfyUI's
    VAE front-crop lands in phase with the video frames. When ``mirror_pad`` is true and
    ``target_duration`` exceeds available audio, extends the tail smoothly via
    ``mirror_pad_audio`` instead of leaving silence cliffs.

    Credits & Attribution:
        Clean-room reimplementation inspired by ``SonderSaid/ComfyUI-Sonder-Editor``
        (https://github.com/SonderSaid/ComfyUI-Sonder-Editor — ``server/audio_grid.py``).
    """
    if not math.isfinite(start) or not math.isfinite(end) or end <= start:
        raise MediaToolError(f"Invalid audio slice span [{start}, {end}].")
    if not math.isfinite(vae_offset_seconds):
        raise MediaToolError(f"Invalid vae_offset_seconds: {vae_offset_seconds}")

    raw_start = start + vae_offset_seconds
    adj_start = max(0.0, raw_start)
    lead_silence = max(0.0, -raw_start)
    raw_duration = max(0.01, end - start)
    adj_end = adj_start + max(0.01, raw_duration - lead_silence)
    slice_audio(src, dst, adj_start, adj_end)

    needs_lead = lead_silence > 1e-6
    needs_pad = target_duration is not None and mirror_pad and math.isfinite(target_duration)
    if not needs_lead and not needs_pad:
        return dst

    info = sf.info(str(dst))
    samples, sr = sf.read(str(dst), dtype="float64", always_2d=True)
    if needs_lead:
        lead_samples = int(round(Fraction(str(round(lead_silence, 6))) * sr))
        if lead_samples > 0:
            zeros = np.zeros((lead_samples, samples.shape[1]), dtype=np.float64)
            samples = np.concatenate([zeros, samples], axis=0)
    if needs_pad and target_duration is not None:
        target_frac = Fraction(str(round(max(raw_duration, target_duration), 6)))
        target_samples = int(round(target_frac * sr))
        if samples.shape[0] < target_samples and samples.shape[0] > 0:
            samples = mirror_pad_audio(samples, target_samples)
    elif not needs_lead:
        return dst

    fd, tmp_name = tempfile.mkstemp(prefix=".mvd_pad_", suffix=".wav", dir=dst.parent)
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        sf.write(str(tmp_path), samples, sr, subtype=info.subtype or "PCM_16")
        os.replace(tmp_path, dst)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    return dst


def extract_frame_at_index(video: Path, frame_index: int, dst: Path) -> Path:
    """Extract the exact 0-based ``frame_index`` frame from ``video`` as a PNG atomically.

    Used for trim-aware cross-shot handoff frames (ensuring ``continue`` / ``reframe`` shots
    anchor onto the exact last frame that survives FFmpeg trimming, rather than an overshoot
    lattice frame).

    Credits & Attribution:
        Trim-aware handoff frame extraction inspired by
        ``GeekatplayStudio/ComfyUI-Music-to-Video``
        (https://github.com/GeekatplayStudio/ComfyUI-Music-to-Video — ``nodes.py``) and
        ``afj176/open-cinema-studio``
        (https://github.com/afj176/open-cinema-studio — ``app/studio.py``).
    """
    if not video.is_file():
        raise MediaToolError(f"Video file not found: {video}")
    idx = max(0, int(frame_index))
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".mvd_frame_", suffix=".png", dir=dst.parent)
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        run(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-protocol_whitelist",
                "file",
                "-i",
                str(video.resolve()),
                "-vf",
                f"select=eq(n\\,{idx})",
                "-vframes",
                "1",
                str(tmp_path.resolve()),
            ]
        )
        if not tmp_path.is_file() or tmp_path.stat().st_size == 0:
            # Fallback to the final decodable frame if the clip was slightly shorter than idx.
            run(
                [
                    "ffmpeg",
                    "-y",
                    "-v",
                    "error",
                    "-protocol_whitelist",
                    "file",
                    "-sseof",
                    "-0.2",
                    "-i",
                    str(video.resolve()),
                    "-vf",
                    "reverse",
                    "-vframes",
                    "1",
                    str(tmp_path.resolve()),
                ]
            )
        if not tmp_path.is_file() or tmp_path.stat().st_size == 0:
            raise MediaToolError(f"Could not extract frame {idx} from {video}")
        os.replace(tmp_path, dst)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    return dst


def zoom_crop_image(src: Path, dst: Path, *, zoom_factor: float = 1.4) -> Path:
    """Generate a Lanczos-zoomed center crop of ``src`` for empty-wide camera-path end guides.

    Credits & Attribution:
        Zoomed-crop camera-path end guides inspired by ``afj176/open-cinema-studio``
        (https://github.com/afj176/open-cinema-studio — ``app/studio.py:_path_guide``) and
        ``hexeosis-prime/nightshift``
        (https://github.com/hexeosis-prime/nightshift — ``nightshift/build.py:zoom_crop``).
    """
    if not src.is_file():
        raise MediaToolError(f"Source image not found: {src}")
    if not math.isfinite(zoom_factor) or zoom_factor <= 1.0 or zoom_factor > 4.0:
        raise MediaToolError(f"zoom_factor must be in (1.0, 4.0], got {zoom_factor}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as image:
        rgb = image.convert("RGB")
        w, h = rgb.size
        crop_w = max(2, int(round(w / zoom_factor)))
        crop_h = max(2, int(round(h / zoom_factor)))
        left = (w - crop_w) // 2
        top = (h - crop_h) // 2
        zoomed = rgb.crop((left, top, left + crop_w, top + crop_h)).resize(
            (w, h), Image.Resampling.LANCZOS
        )
        fd, tmp_name = tempfile.mkstemp(prefix=".mvd_zoom_", suffix=".png", dir=dst.parent)
        os.close(fd)
        tmp_path = Path(tmp_name)
        try:
            zoomed.save(tmp_path, format="PNG")
            os.replace(tmp_path, dst)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise
    return dst


def _write_with_silent_audio(video_path: Path, dst: Path) -> None:
    """Write ``video_path`` plus a silent stereo AAC track to ``dst`` (an MP4) atomically.

    The video stream is copied when MP4 can carry it and re-encoded to H.264 otherwise (for
    example ProRes ``.mov`` or VP8 ``.webm`` references).
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".mvd_aud_", suffix=".mp4", dir=dst.parent)
    os.close(fd)
    tmp_path = Path(tmp_name)
    copy_video = ["-c:v", "copy"]
    reencode_video = ["-c:v", "libx264", "-preset", "medium", "-crf", "16", "-pix_fmt", "yuv420p"]
    try:
        for video_codec in (copy_video, reencode_video):
            try:
                run(
                    [
                        "ffmpeg",
                        "-y",
                        "-v",
                        "error",
                        "-protocol_whitelist",
                        "file",
                        "-i",
                        str(video_path.resolve()),
                        "-f",
                        "lavfi",
                        "-i",
                        "anullsrc=channel_layout=stereo:sample_rate=44100",
                        "-map",
                        "0:v:0",
                        "-map",
                        "1:a:0",
                        *video_codec,
                        "-c:a",
                        "aac",
                        "-b:a",
                        "128k",
                        "-shortest",
                        "-movflags",
                        "+faststart",
                        str(tmp_path.resolve()),
                    ]
                )
                break
            except MediaToolError:
                if video_codec is reencode_video:
                    raise
        os.replace(tmp_path, dst)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def ensure_audio_track(video_path: Path, dst: Path) -> Path:
    """Return ``video_path`` if it has an audio stream, else write an MP4 copy with silent audio.

    Prevents ComfyUI video-loader nodes (such as ``VHS_LoadVideoPath``) from crashing on silent
    reference or previz clips. ``dst`` may equal ``video_path`` only for an MP4 the pipeline
    rendered itself; use :func:`with_audio_track` for files a person supplied.

    Credits & Attribution:
        Clean-room silent-audio guard inspired by ``heroesjouney/aimoviestudiov2``
        (https://github.com/heroesjouney/aimoviestudiov2 — ``comfy_video.py``).
    """
    if probe(video_path).has_audio:
        return video_path
    _write_with_silent_audio(video_path, dst)
    return dst


def with_audio_track(video_path: Path, cache_dir: Path) -> Path:
    """``video_path`` if it has audio, else a cached MP4 copy carrying a silent audio track.

    The source is never modified, whatever its container. The copy keeps the source's name (so
    uploads look the same in ComfyUI) inside a per-source folder of ``cache_dir``, and is rebuilt
    only when the source is newer than it.
    """
    if probe(video_path).has_audio:
        return video_path
    key = hashlib.sha256(str(video_path.resolve()).encode("utf-8")).hexdigest()[:12]
    dst = cache_dir / key / f"{video_path.stem}.mp4"
    if dst.is_file() and dst.stat().st_mtime_ns >= video_path.stat().st_mtime_ns:
        return dst
    _write_with_silent_audio(video_path, dst)
    return dst
