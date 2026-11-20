# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Audio-visual lip-sync scoring using ByteDance LatentSync's StableSyncNet + MediaPipe.

Pairs LatentSync's pixel-space ``stable_syncnet.pt`` checkpoint (OpenRAIL++ weights, Apache-2.0
architecture) with MediaPipe Face Detector (Apache-2.0) for face cropping instead of Oxford VGG's
research-only ``syncnet_v2.model`` and S3FD.

PyTorch is imported lazily only when ``StableSyncNetScorer`` loads the neural model, so the rest of
``mvdirector`` stays CPU-only and dependency-light when ``qc.lipsync_backend: mediapipe`` is used.
"""

from __future__ import annotations

import contextlib
import functools
import logging
import os
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import librosa
import numpy as np
from PIL import Image

from mvdirector.errors import DependencyMissing
from mvdirector.qc.common import download_cached_model
from mvdirector.qc.faces import FaceAnalyzer, MediaPipeFaceAnalyzer
from mvdirector.qc.lipsync import SyncMeasurement, fill_gaps, smooth

log = logging.getLogger(__name__)

# StableSyncNet (configs/syncnet/syncnet_16_pixel_attn.yaml + configs/audio.yaml) contract.
SYNCNET_FPS = 25.0
SYNCNET_SAMPLE_RATE = 16000
SYNCNET_NUM_FRAMES = 16
SYNCNET_MEL_WINDOW = 52  # math.ceil(16 / 5 * 16)
SYNCNET_RESOLUTION = 256
SYNCNET_LOWER_HALF_HEIGHT = 128

_AUDIO_N_FFT = 800
_AUDIO_HOP_SIZE = 200
_AUDIO_WIN_SIZE = 800
_AUDIO_NUM_MELS = 80
_AUDIO_FMIN = 55.0
_AUDIO_FMAX = 7600.0
_AUDIO_PREEMPHASIS = 0.97
_AUDIO_REF_LEVEL_DB = 20.0
_AUDIO_MIN_LEVEL_DB = -100.0
_AUDIO_MAX_ABS_VALUE = 4.0

STABLE_SYNCNET_FILENAME = "stable_syncnet.pt"
STABLE_SYNCNET_URL = (
    "https://huggingface.co/ByteDance/LatentSync-1.6/resolve/main/stable_syncnet.pt"
)

_SYNCNET_ARCH: dict[str, dict[str, Any]] = {
    "audio_encoder": {
        "in_channels": 1,
        "block_out_channels": [32, 64, 128, 256, 512, 1024, 2048],
        "downsample_factors": [(2, 1), 2, 2, 1, 2, 2, (2, 3)],
        "attn_blocks": [0, 0, 0, 1, 1, 0, 0],
        "dropout": 0.0,
    },
    "visual_encoder": {
        "in_channels": 48,
        "block_out_channels": [64, 128, 256, 256, 512, 1024, 2048, 2048],
        "downsample_factors": [(1, 2), 2, 2, 2, 2, 2, 2, 2],
        "attn_blocks": [0, 0, 0, 0, 1, 1, 0, 0],
        "dropout": 0.0,
    },
}


# --------------------------------------------------------------------------- cache & download


def _syncnet_cache_dir() -> Path:
    override = os.environ.get("MVD_SYNCNET_CACHE_DIR", "").strip()
    if override:
        return Path(override).expanduser()
    xdg_cache = os.environ.get("XDG_CACHE_HOME", "").strip()
    base = Path(xdg_cache).expanduser() if xdg_cache else Path.home() / ".cache"
    return base / "mvdirector" / "syncnet"


def ensure_syncnet_model(model_path: Path | None = None) -> Path:
    """Resolve an explicit ``stable_syncnet.pt`` path or download and cache it."""
    return download_cached_model(
        STABLE_SYNCNET_FILENAME,
        STABLE_SYNCNET_URL,
        _syncnet_cache_dir(),
        model_path=model_path,
        label="StableSyncNet checkpoint",
        env_var="MVD_SYNCNET_CACHE_DIR",
        timeout=300.0,
        client_factory=httpx.Client,
    )


# --------------------------------------------------------------------------- audio mel features


@functools.lru_cache(maxsize=1)
def _mel_basis() -> np.ndarray:
    return librosa.filters.mel(
        sr=SYNCNET_SAMPLE_RATE,
        n_fft=_AUDIO_N_FFT,
        n_mels=_AUDIO_NUM_MELS,
        fmin=_AUDIO_FMIN,
        fmax=_AUDIO_FMAX,
    )


def mel_spectrogram(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    """Compute the normalized ``(80, T_mel)`` log-mel spectrogram expected by StableSyncNet."""
    wav = np.asarray(samples, dtype=np.float32)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if wav.size == 0 or sample_rate <= 0:
        return np.zeros((_AUDIO_NUM_MELS, 0), dtype=np.float32)
    if sample_rate != SYNCNET_SAMPLE_RATE:
        wav = librosa.resample(wav, orig_sr=sample_rate, target_sr=SYNCNET_SAMPLE_RATE)
    if wav.size > 1:
        wav = np.concatenate((wav[:1], wav[1:] - _AUDIO_PREEMPHASIS * wav[:-1]))
    stft = librosa.stft(
        y=wav,
        n_fft=_AUDIO_N_FFT,
        hop_length=_AUDIO_HOP_SIZE,
        win_length=_AUDIO_WIN_SIZE,
    )
    mel = np.dot(_mel_basis(), np.abs(stft))
    min_level = np.exp(_AUDIO_MIN_LEVEL_DB / 20.0 * np.log(10.0))
    db = 20.0 * np.log10(np.maximum(min_level, mel)) - _AUDIO_REF_LEVEL_DB
    normalized = (2.0 * _AUDIO_MAX_ABS_VALUE) * (
        (db - _AUDIO_MIN_LEVEL_DB) / (-_AUDIO_MIN_LEVEL_DB)
    ) - _AUDIO_MAX_ABS_VALUE
    return np.clip(normalized, -_AUDIO_MAX_ABS_VALUE, _AUDIO_MAX_ABS_VALUE).astype(np.float32)


def mel_window_for_frame(mel: np.ndarray, frame_index: int) -> np.ndarray | None:
    """Extract the ``(80, 52)`` mel slice aligned to a 25-fps video frame index."""
    start = int(80.0 * (frame_index / SYNCNET_FPS))
    end = start + SYNCNET_MEL_WINDOW
    if start < 0 or end > mel.shape[1]:
        return None
    return mel[:, start:end]


# --------------------------------------------------------------------------- video & face crop


def resample_frames_to_25fps(frames: list[np.ndarray], fps: float) -> list[np.ndarray]:
    """Nearest-frame resample to StableSyncNet's 25-fps cadence."""
    if not frames or fps <= 0:
        return []
    if abs(fps - SYNCNET_FPS) < 1e-3:
        return list(frames)
    target_len = max(1, round(len(frames) * SYNCNET_FPS / fps))
    indices = np.clip(
        np.round(np.arange(target_len) * (fps / SYNCNET_FPS)).astype(int),
        0,
        len(frames) - 1,
    )
    return [frames[int(i)] for i in indices]


def extract_face_tracks(
    frames: list[np.ndarray],
    face_analyzer: FaceAnalyzer,
    max_faces: int = 1,
) -> list[tuple[float, list[tuple[int, int, int, int]]]]:
    """Track up to ``max_faces`` face bounding boxes across frames using ``face_analyzer``.

    Returns ``(face_presence, smoothed_boxes)`` for every active track.
    """
    if not frames:
        return []
    limit = max(1, max_faces)
    n_frames = len(frames)
    frame_h, frame_w = frames[0].shape[0], frames[0].shape[1]
    coords = [np.full((n_frames, 4), np.nan, dtype=np.float64) for _ in range(limit)]
    anchors: list[float | None] = [None] * limit

    for idx, frame in enumerate(frames):
        detected = [f.bbox for f in face_analyzer.faces(frame) if f.bbox is not None]
        if not detected:
            continue
        if limit == 1:
            # FaceAnalyzer returns faces largest-first; keep the foreground singer rather than
            # sorting left-to-right (which could latch onto a smaller background face).
            coords[0][idx] = detected[0]
            anchors[0] = (detected[0][0] + detected[0][2]) * 0.5 / max(frame_w, 1)
            continue
        detected.sort(key=lambda b: (b[0] + b[2]) * 0.5)
        used: set[int] = set()
        for box in detected:
            cx = (box[0] + box[2]) * 0.5 / max(frame_w, 1)
            initialized = [
                (i, anchors[i]) for i in range(limit) if i not in used and anchors[i] is not None
            ]
            if initialized:
                slot = min(initialized, key=lambda pair: abs(cx - pair[1]))[0]  # type: ignore[operator]
                if abs(cx - anchors[slot]) > 0.25:  # type: ignore[operator]
                    empty = next(
                        (i for i in range(limit) if i not in used and anchors[i] is None),
                        None,
                    )
                    if empty is not None:
                        slot = empty
            else:
                slot = next((i for i in range(limit) if i not in used), -1)
                if slot < 0:
                    continue
            used.add(slot)
            anchors[slot] = cx
            coords[slot][idx] = box

    results: list[tuple[float, list[tuple[int, int, int, int]]]] = []
    for track, anchor in zip(coords, anchors):
        if anchor is None:
            continue
        presence = float(np.mean(~np.isnan(track[:, 0])))
        smoothed = np.column_stack([smooth(fill_gaps(track[:, col]), width=5) for col in range(4)])
        boxes: list[tuple[int, int, int, int]] = []
        for row in smoothed:
            x1 = max(0, min(frame_w - 1, int(round(row[0]))))
            y1 = max(0, min(frame_h - 1, int(round(row[1]))))
            x2 = max(x1 + 1, min(frame_w, int(round(row[2]))))
            y2 = max(y1 + 1, min(frame_h, int(round(row[3]))))
            boxes.append((x1, y1, x2, y2))
        results.append((presence, boxes))
    return results


def crop_lower_face(
    frame: np.ndarray,
    bbox: tuple[int, int, int, int],
    *,
    margin: float = 1.3,
) -> np.ndarray:
    """Square-crop around ``bbox``, resize to 256x256, normalize to ``[-1, 1]``, keep lower half.

    Returns a ``(3, 128, 256)`` float32 CHW array matching StableSyncNet's pixel-space input.
    """
    frame_h, frame_w = frame.shape[0], frame.shape[1]
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    half = max(x2 - x1, y2 - y1) * margin * 0.5
    sx1 = max(0, int(round(cx - half)))
    sy1 = max(0, int(round(cy - half)))
    sx2 = min(frame_w, max(sx1 + 1, int(round(cx + half))))
    sy2 = min(frame_h, max(sy1 + 1, int(round(cy + half))))

    crop = Image.fromarray(frame[sy1:sy2, sx1:sx2]).resize(
        (SYNCNET_RESOLUTION, SYNCNET_RESOLUTION),
        Image.Resampling.BICUBIC,
    )
    arr = np.asarray(crop, dtype=np.float32) / 127.5 - 1.0
    lower = arr[SYNCNET_LOWER_HALF_HEIGHT:, :, :]
    return np.transpose(lower, (2, 0, 1))


# --------------------------------------------------------------------------- embedding lag sweep


def score_embedding_sequences(
    vision_embeds: np.ndarray,
    audio_embeds: np.ndarray,
    max_lag_frames: int,
    face_presence: float,
) -> SyncMeasurement:
    """Find the best mean cosine similarity and lag across ``±max_lag_frames`` at 25 fps."""
    n_v, n_a = len(vision_embeds), len(audio_embeds)
    if n_v == 0 or n_a == 0:
        return SyncMeasurement(
            correlation=0.0,
            lag_frames=0,
            face_presence=face_presence,
            eval_fps=SYNCNET_FPS,
            confidence=0.0,
        )

    sims: list[tuple[int, float]] = []
    for lag in range(-max_lag_frames, max_lag_frames + 1):
        if lag >= 0:
            overlap = min(n_v - lag, n_a)
            if overlap <= 0:
                continue
            v_slice = vision_embeds[lag : lag + overlap]
            a_slice = audio_embeds[:overlap]
        else:
            overlap = min(n_v, n_a + lag)
            if overlap <= 0:
                continue
            v_slice = vision_embeds[:overlap]
            a_slice = audio_embeds[-lag : -lag + overlap]
        sim = float(np.mean(np.sum(v_slice * a_slice, axis=1)))
        sims.append((lag, sim))

    if not sims:
        return SyncMeasurement(
            correlation=0.0,
            lag_frames=0,
            face_presence=face_presence,
            eval_fps=SYNCNET_FPS,
            confidence=0.0,
        )

    best_lag, best_sim = max(sims, key=lambda pair: pair[1])
    median_sim = float(np.median([s for _, s in sims]))
    confidence = max(0.0, best_sim - median_sim)
    return SyncMeasurement(
        correlation=max(0.0, best_sim),
        lag_frames=best_lag,
        face_presence=face_presence,
        eval_fps=SYNCNET_FPS,
        confidence=confidence,
    )


# --------------------------------------------------------------------------- pure-PyTorch model


def _import_torch() -> Any:
    try:
        import torch

        return torch
    except ImportError as exc:
        raise DependencyMissing(
            "PyTorch is not installed. Install the 'syncnet' extra "
            '(`pip install -e ".[qc,syncnet]"`) to use qc.lipsync_backend: stable_syncnet, '
            "or switch to qc.lipsync_backend: mediapipe."
        ) from exc


def _resolve_device(torch_mod: Any, requested: str) -> str:
    if requested != "auto":
        return requested
    if torch_mod.cuda.is_available():
        return "cuda"
    mps = getattr(getattr(torch_mod, "backends", None), "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def build_stable_syncnet(torch_mod: Any) -> Any:
    """Construct ByteDance LatentSync's pixel-space ``StableSyncNet`` without diffusers/einops.

    Parameter names and shapes match ``checkpoints/stable_syncnet.pt``
    (``configs/syncnet/syncnet_16_pixel_attn.yaml``) verbatim so ``load_state_dict`` succeeds
    with ``strict=True``.
    """
    nn = torch_mod.nn
    F = torch_mod.nn.functional
    module_base: Any = nn.Module

    class _GEGLU(module_base):
        def __init__(self, dim_in: int, dim_out: int) -> None:
            super().__init__()
            self.proj = nn.Linear(dim_in, dim_out * 2)

        def forward(self, hidden_states: Any) -> Any:
            hidden, gate = self.proj(hidden_states).chunk(2, dim=-1)
            return hidden * F.gelu(gate)

    class _FeedForward(module_base):
        def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0) -> None:
            super().__init__()
            inner_dim = int(dim * mult)
            self.net = nn.ModuleList(
                [_GEGLU(dim, inner_dim), nn.Dropout(dropout), nn.Linear(inner_dim, dim)]
            )

        def forward(self, hidden_states: Any) -> Any:
            for layer in self.net:
                hidden_states = layer(hidden_states)
            return hidden_states

    class _SelfAttention(module_base):
        def __init__(
            self,
            query_dim: int,
            heads: int = 8,
            dim_head: int = 64,
            dropout: float = 0.0,
            bias: bool = True,
        ) -> None:
            super().__init__()
            inner_dim = dim_head * heads
            self.heads = heads
            self.to_q = nn.Linear(query_dim, inner_dim, bias=bias)
            self.to_k = nn.Linear(query_dim, inner_dim, bias=bias)
            self.to_v = nn.Linear(query_dim, inner_dim, bias=bias)
            self.to_out = nn.ModuleList([nn.Linear(inner_dim, query_dim), nn.Dropout(dropout)])

        def forward(self, hidden_states: Any) -> Any:
            b, seq_len, dim = hidden_states.shape
            head_dim = dim // self.heads
            q = (
                self.to_q(hidden_states)
                .reshape(b, seq_len, self.heads, head_dim)
                .permute(0, 2, 1, 3)
            )
            k = (
                self.to_k(hidden_states)
                .reshape(b, seq_len, self.heads, head_dim)
                .permute(0, 2, 1, 3)
            )
            v = (
                self.to_v(hidden_states)
                .reshape(b, seq_len, self.heads, head_dim)
                .permute(0, 2, 1, 3)
            )
            out = F.scaled_dot_product_attention(q, k, v)
            out = out.permute(0, 2, 1, 3).reshape(b, seq_len, dim)
            out = self.to_out[0](out)
            return self.to_out[1](out)

    class _ResnetBlock2D(module_base):
        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            dropout: float = 0.0,
            norm_num_groups: int = 32,
            eps: float = 1e-6,
            downsample_factor: int | tuple[int, int] = 2,
        ) -> None:
            super().__init__()
            self.norm1 = nn.GroupNorm(norm_num_groups, in_channels, eps=eps, affine=True)
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.norm2 = nn.GroupNorm(norm_num_groups, out_channels, eps=eps, affine=True)
            self.dropout = nn.Dropout(dropout)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.act_fn = nn.SiLU()
            self.conv_shortcut = (
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
                if in_channels != out_channels
                else None
            )
            if downsample_factor == 1:
                self.downsample_conv = None
                self.pad = (0, 0, 0, 0)
            else:
                self.downsample_conv = nn.Conv2d(
                    out_channels, out_channels, kernel_size=3, stride=downsample_factor, padding=0
                )
                self.pad = (0, 1, 0, 1)
                if isinstance(downsample_factor, tuple):
                    if downsample_factor[0] == 1:
                        self.pad = (0, 1, 1, 1)
                    elif downsample_factor[1] == 1:
                        self.pad = (1, 1, 0, 1)

        def forward(self, input_tensor: Any) -> Any:
            h = self.conv1(self.act_fn(self.norm1(input_tensor)))
            h = self.conv2(self.dropout(self.act_fn(self.norm2(h))))
            if self.conv_shortcut is not None:
                input_tensor = self.conv_shortcut(input_tensor)
            h = h + input_tensor
            if self.downsample_conv is not None:
                h = self.downsample_conv(F.pad(h, self.pad, mode="constant", value=0))
            return h

    class _AttentionBlock2D(module_base):
        def __init__(self, query_dim: int, norm_num_groups: int = 32, dropout: float = 0.0) -> None:
            super().__init__()
            self.norm1 = nn.GroupNorm(norm_num_groups, query_dim, eps=1e-6, affine=True)
            self.norm2 = nn.LayerNorm(query_dim)
            self.norm3 = nn.LayerNorm(query_dim)
            self.ff = _FeedForward(query_dim, dropout=dropout)
            self.conv_in = nn.Conv2d(query_dim, query_dim, kernel_size=1, stride=1, padding=0)
            self.conv_out = nn.Conv2d(query_dim, query_dim, kernel_size=1, stride=1, padding=0)
            self.attn = _SelfAttention(
                query_dim=query_dim,
                heads=8,
                dim_head=query_dim // 8,
                dropout=dropout,
                bias=True,
            )

        def forward(self, hidden_states: Any) -> Any:
            b, c, h, w = hidden_states.shape
            residual = hidden_states
            x = self.conv_in(self.norm1(hidden_states))
            x = x.permute(0, 2, 3, 1).reshape(b, h * w, c)
            x = self.attn(self.norm2(x)) + x
            x = self.ff(self.norm3(x)) + x
            x = x.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()
            return self.conv_out(x) + residual

    class _DownEncoder2D(module_base):
        def __init__(self, cfg: dict[str, Any]) -> None:
            super().__init__()
            channels: list[int] = cfg["block_out_channels"]
            factors: list[int | tuple[int, int]] = cfg["downsample_factors"]
            attns: list[int] = cfg["attn_blocks"]
            dropout: float = cfg["dropout"]
            self.conv_in = nn.Conv2d(
                cfg["in_channels"], channels[0], kernel_size=3, stride=1, padding=1
            )
            blocks: list[Any] = []
            prev = channels[0]
            for out_c, factor, use_attn in zip(channels, factors, attns):
                blocks.append(
                    _ResnetBlock2D(prev, out_c, dropout=dropout, downsample_factor=factor)
                )
                if use_attn:
                    blocks.append(_AttentionBlock2D(out_c, dropout=dropout))
                prev = out_c
            self.down_blocks = nn.ModuleList(blocks)
            self.norm_out = nn.GroupNorm(32, channels[-1], eps=1e-6)
            self.act_fn_out = nn.ReLU()

        def forward(self, hidden_states: Any) -> Any:
            x = self.conv_in(hidden_states)
            for block in self.down_blocks:
                x = block(x)
            return self.act_fn_out(self.norm_out(x))

    class _StableSyncNet(module_base):
        def __init__(self) -> None:
            super().__init__()
            self.audio_encoder = _DownEncoder2D(_SYNCNET_ARCH["audio_encoder"])
            self.visual_encoder = _DownEncoder2D(_SYNCNET_ARCH["visual_encoder"])
            self.eval()

        def encode_visual(self, image_sequences: Any) -> Any:
            embeds = self.visual_encoder(image_sequences)
            return F.normalize(embeds.reshape(embeds.shape[0], -1), p=2, dim=1)

        def encode_audio(self, audio_sequences: Any) -> Any:
            embeds = self.audio_encoder(audio_sequences)
            return F.normalize(embeds.reshape(embeds.shape[0], -1), p=2, dim=1)

        def forward(self, image_sequences: Any, audio_sequences: Any) -> tuple[Any, Any]:
            return self.encode_visual(image_sequences), self.encode_audio(audio_sequences)

    return _StableSyncNet()


def _load_stable_syncnet_checkpoint(torch_mod: Any, model_path: Path | None, device: str) -> Any:
    """Load ``stable_syncnet.pt`` with ``weights_only=True`` (re-downloading once if corrupt)."""
    resolved = ensure_syncnet_model(model_path=model_path)

    def _load(path: Path) -> Any:
        model = build_stable_syncnet(torch_mod)
        checkpoint = torch_mod.load(str(path), map_location=device, weights_only=True)
        state_dict = (
            checkpoint["state_dict"]
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint
            else checkpoint
        )
        model.load_state_dict(state_dict, strict=True)
        model.to(device)
        model.eval()
        return model

    load_errors = (RuntimeError, ValueError, OSError, KeyError, EOFError, zipfile.BadZipFile)
    try:
        return _load(resolved)
    except load_errors as exc:
        if model_path is not None:
            raise DependencyMissing(
                f"StableSyncNet could not load checkpoint {resolved}: {exc}"
            ) from exc
        log.warning(
            "Cached StableSyncNet checkpoint %s failed to load (%s); downloading again",
            resolved,
            exc,
        )
        with contextlib.suppress(OSError):
            resolved.unlink(missing_ok=True)
        resolved = ensure_syncnet_model(model_path=None)
        try:
            return _load(resolved)
        except load_errors as retry_exc:
            with contextlib.suppress(OSError):
                resolved.unlink(missing_ok=True)
            raise DependencyMissing(
                f"StableSyncNet could not load the freshly downloaded checkpoint {resolved}: "
                f"{retry_exc}. Retry, or set qc.lipsync_backend: mediapipe in project.yaml."
            ) from retry_exc


# --------------------------------------------------------------------------- scorer


class StableSyncNetScorer:
    """LipSyncScorer using ByteDance LatentSync's ``stable_syncnet.pt`` + MediaPipe."""

    def __init__(
        self,
        model_path: Path | None = None,
        *,
        device: str = "auto",
        face_analyzer: FaceAnalyzer | None = None,
        batch_size: int = 16,
        _encoder: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]] | None = None,
    ) -> None:
        self._batch_size = max(1, batch_size)
        self._closed = False
        self._owns_faces = face_analyzer is None
        self._faces: FaceAnalyzer = (
            face_analyzer if face_analyzer is not None else MediaPipeFaceAnalyzer()
        )
        self._encoder = _encoder
        self._torch: Any = None
        self._device = "cpu"
        self._model: Any = None
        if _encoder is None:
            try:
                self._torch = _import_torch()
                self._device = _resolve_device(self._torch, device)
                self._model = _load_stable_syncnet_checkpoint(self._torch, model_path, self._device)
            except Exception:
                self.close()
                raise

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._model = None
        if self._owns_faces:
            closer = getattr(self._faces, "close", None)
            if callable(closer):
                closer()

    def __enter__(self) -> StableSyncNetScorer:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _encode_visual(self, visual_windows: np.ndarray) -> np.ndarray:
        torch_mod = self._torch
        parts: list[np.ndarray] = []
        with torch_mod.no_grad():
            for start in range(0, len(visual_windows), self._batch_size):
                batch = torch_mod.from_numpy(visual_windows[start : start + self._batch_size]).to(
                    self._device, dtype=torch_mod.float32
                )
                parts.append(self._model.encode_visual(batch).cpu().numpy())
        return np.concatenate(parts, axis=0) if parts else np.zeros((0, 0), dtype=np.float32)

    def _encode_audio(self, audio_windows: np.ndarray) -> np.ndarray:
        torch_mod = self._torch
        parts: list[np.ndarray] = []
        with torch_mod.no_grad():
            for start in range(0, len(audio_windows), self._batch_size):
                batch = torch_mod.from_numpy(audio_windows[start : start + self._batch_size]).to(
                    self._device, dtype=torch_mod.float32
                )
                parts.append(self._model.encode_audio(batch).cpu().numpy())
        return np.concatenate(parts, axis=0) if parts else np.zeros((0, 0), dtype=np.float32)

    def _encode_batches(
        self, visual_windows: np.ndarray, audio_windows: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if self._encoder is not None:
            return self._encoder(visual_windows, audio_windows)
        return self._encode_visual(visual_windows), self._encode_audio(audio_windows)

    def score_all(
        self,
        frames: list[np.ndarray],
        fps: float,
        samples: np.ndarray,
        sample_rate: int,
        max_lag_seconds: float,
        max_faces: int = 1,
    ) -> list[SyncMeasurement]:
        """Score every tracked face in ``frames`` (up to ``max_faces``)."""
        if self._closed:
            raise RuntimeError("StableSyncNetScorer has been closed")
        frames_25 = resample_frames_to_25fps(frames, fps)
        if not frames_25:
            return []
        tracks = extract_face_tracks(frames_25, self._faces, max_faces=max_faces)
        if not tracks:
            return []

        n_frames = len(frames_25)
        n_visual_windows = n_frames - SYNCNET_NUM_FRAMES + 1
        if n_visual_windows <= 0:
            return [
                SyncMeasurement(
                    correlation=0.0,
                    lag_frames=0,
                    face_presence=p,
                    eval_fps=SYNCNET_FPS,
                    confidence=0.0,
                )
                for p, _ in tracks
            ]

        mel = mel_spectrogram(samples, sample_rate)
        audio_list: list[np.ndarray] = []
        for idx in range(n_visual_windows):
            window = mel_window_for_frame(mel, idx)
            if window is None:
                break
            audio_list.append(window[np.newaxis, :, :])  # (1, 80, 52)
        if not audio_list:
            return [
                SyncMeasurement(
                    correlation=0.0,
                    lag_frames=0,
                    face_presence=p,
                    eval_fps=SYNCNET_FPS,
                    confidence=0.0,
                )
                for p, _ in tracks
            ]

        audio_windows = np.stack(audio_list, axis=0).astype(np.float32)
        cached_audio_embeds = self._encode_audio(audio_windows) if self._encoder is None else None
        max_lag_frames = max(1, round(max_lag_seconds * SYNCNET_FPS))

        measurements: list[SyncMeasurement] = []
        for presence, boxes in tracks:
            crops = [crop_lower_face(frame, box) for frame, box in zip(frames_25, boxes)]
            # Stack 16 (3, 128, 256) crops -> (48, 128, 256) per window (b f c h w -> b (f c) h w).
            visual_windows = np.stack(
                [
                    np.concatenate(crops[i : i + SYNCNET_NUM_FRAMES], axis=0)
                    for i in range(n_visual_windows)
                ],
                axis=0,
            ).astype(np.float32)
            if cached_audio_embeds is not None:
                v_embeds, a_embeds = self._encode_visual(visual_windows), cached_audio_embeds
            else:
                v_embeds, a_embeds = self._encode_batches(visual_windows, audio_windows)
            measurements.append(
                score_embedding_sequences(v_embeds, a_embeds, max_lag_frames, presence)
            )
        return measurements

    def score(
        self,
        frames: list[np.ndarray],
        fps: float,
        samples: np.ndarray,
        sample_rate: int,
        max_lag_seconds: float,
        max_faces: int = 1,
    ) -> SyncMeasurement:
        measurements = self.score_all(
            frames, fps, samples, sample_rate, max_lag_seconds, max_faces=max_faces
        )
        if not measurements:
            return SyncMeasurement(
                correlation=0.0,
                lag_frames=0,
                face_presence=0.0,
                eval_fps=SYNCNET_FPS,
                confidence=0.0,
            )
        present = [m for m in measurements if m.face_presence >= 0.5]
        pool = present or measurements
        return max(pool, key=lambda m: (m.correlation, m.face_presence))
