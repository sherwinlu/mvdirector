# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""VAE color-drift cancellation via ridge-regularized 3x4 RGB affine transforms.

Chained diffusion takes (continue / reframe / FLF2V extensions) accumulate subtle color and
contrast drift across repeated VAE encode-decode cycles. Comparing boundary context frames against
a reference anchor solves a bounded 3x4 RGB affine matrix toward identity, and bridging two
differently graded clips smoothly interpolates from a head affine transform to a tail affine
transform across the span.

Credits & Attribution:
    Clean-room reimplementation of the mathematical ridge-regularized 3x4 RGB affine color-drift
    cancellation and dual-anchor blended ramp concept inspired by
    ``SonderSaid/ComfyUI-Sonder-Editor``
    (https://github.com/SonderSaid/ComfyUI-Sonder-Editor — ``server/color_correction.py``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from PIL import Image

_FIT_MAX_SIDE = 320
_DEFAULT_RIDGE_LAMBDA = 1e-3
_MAX_MATRIX_DEVIATION = 0.15
_MAX_BIAS_UNIT = 10.0 / 255.0


@dataclass(frozen=True)
class ColorAffine:
    """Clamped 3x4 RGB affine transform ``y = x @ matrix.T + bias`` on ``[0, 1]`` RGB."""

    matrix: np.ndarray  # shape (3, 3), float64
    bias: np.ndarray  # shape (3,), float64

    @classmethod
    def identity(cls) -> ColorAffine:
        return cls(matrix=np.eye(3, dtype=np.float64), bias=np.zeros(3, dtype=np.float64))

    def is_identity(self, tol: float = 1e-4) -> bool:
        return bool(
            np.max(np.abs(self.matrix - np.eye(3))) <= tol and np.max(np.abs(self.bias)) <= tol
        )

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """Apply the affine transform to an RGB uint8 ``(H, W, 3)`` frame."""
        rgb = np.asarray(frame, dtype=np.float64) / 255.0
        corrected = rgb @ self.matrix.T + self.bias[None, None, :]
        return np.clip(np.rint(corrected * 255.0), 0.0, 255.0).astype(np.uint8)

    def to_ffmpeg_colorchannelmixer(self) -> str | None:
        """Express the affine transform as FFmpeg ``colorchannelmixer`` (+ ``lutrgb`` bias)."""
        if self.is_identity():
            return None
        parts: list[str] = []
        if float(np.max(np.abs(self.matrix - np.eye(3)))) > 1e-4:
            m = self.matrix
            parts.append(
                f"colorchannelmixer="
                f"rr={m[0, 0]:.5f}:rg={m[0, 1]:.5f}:rb={m[0, 2]:.5f}:"
                f"gr={m[1, 0]:.5f}:gg={m[1, 1]:.5f}:gb={m[1, 2]:.5f}:"
                f"br={m[2, 0]:.5f}:bg={m[2, 1]:.5f}:bb={m[2, 2]:.5f}"
            )
        if float(np.max(np.abs(self.bias))) > 1e-4:
            br, bg, bb = (float(self.bias[i]) * 255.0 for i in range(3))
            parts.append(
                f"lutrgb="
                f"r='clip(val+{br:+.3f},0,255)':"
                f"g='clip(val+{bg:+.3f},0,255)':"
                f"b='clip(val+{bb:+.3f},0,255)'"
            )
        return ",".join(parts) if parts else None


def _downscale_rgb(frame: np.ndarray, max_side: int = _FIT_MAX_SIDE) -> np.ndarray:
    img = Image.fromarray(frame).convert("RGB")
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        img = img.resize(
            (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
            Image.Resampling.BOX,
        )
    return np.asarray(img, dtype=np.float64) / 255.0


def fit_color_drift_affine(
    source_frames: Sequence[np.ndarray],
    target_frames: Sequence[np.ndarray],
    *,
    ridge_lambda: float = _DEFAULT_RIDGE_LAMBDA,
    max_matrix_deviation: float = _MAX_MATRIX_DEVIATION,
    max_bias: float = _MAX_BIAS_UNIT,
) -> ColorAffine:
    """Solve a ridge-regularized 3x4 RGB affine map from ``source_frames`` to ``target_frames``.

    Regularizes strictly toward the identity transform and clamps matrix deviations and bias so
    content differences between frames never produce wild color shifts.
    """
    n_pairs = min(len(source_frames), len(target_frames))
    if n_pairs == 0:
        return ColorAffine.identity()

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for src, tgt in zip(source_frames[:n_pairs], target_frames[:n_pairs]):
        s_small = _downscale_rgb(src)
        t_small = _downscale_rgb(tgt)
        if s_small.shape != t_small.shape:
            t_img = (
                Image.fromarray(tgt)
                .convert("RGB")
                .resize((s_small.shape[1], s_small.shape[0]), Image.Resampling.BOX)
            )
            t_small = np.asarray(t_img, dtype=np.float64) / 255.0
        xs.append(s_small.reshape(-1, 3))
        ys.append(t_small.reshape(-1, 3))

    x_rgb = np.concatenate(xs, axis=0)
    y_rgb = np.concatenate(ys, axis=0)
    if x_rgb.shape[0] < 16:
        return ColorAffine.identity()

    # Augmented design matrix [R, G, B, 1] -> solve (X^T X + lam * I) W = X^T Y + lam * W_id.
    n_pix = float(x_rgb.shape[0])
    ones = np.ones((x_rgb.shape[0], 1), dtype=np.float64)
    x_aug = np.hstack([x_rgb, ones])
    w_id = np.vstack([np.eye(3, dtype=np.float64), np.zeros((1, 3), dtype=np.float64)])
    lam = max(1e-6, ridge_lambda) * n_pix
    lhs = x_aug.T @ x_aug + lam * np.eye(4, dtype=np.float64)
    rhs = x_aug.T @ y_rgb + lam * w_id
    try:
        weights = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        return ColorAffine.identity()

    matrix = weights[:3, :].T
    bias = weights[3, :]
    identity = np.eye(3, dtype=np.float64)
    clamped_matrix = identity + np.clip(
        matrix - identity, -max_matrix_deviation, max_matrix_deviation
    )
    clamped_bias = np.clip(bias, -max_bias, max_bias)
    if not (np.all(np.isfinite(clamped_matrix)) and np.all(np.isfinite(clamped_bias))):
        return ColorAffine.identity()
    return ColorAffine(matrix=clamped_matrix, bias=clamped_bias)
