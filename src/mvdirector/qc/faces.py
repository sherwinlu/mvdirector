# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Face analysis behind small protocols, with optional-dependency implementations.

The QC checks depend only on the protocols, so tests use fakes and the heavy
libraries (insightface, mediapipe) are needed only when the checks really run.
"""

from __future__ import annotations

import contextlib
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TypeVar

import httpx
import numpy as np
from PIL import Image

from mvdirector.errors import DependencyMissing
from mvdirector.qc.common import download_cached_model

log = logging.getLogger(__name__)
_ModelT = TypeVar("_ModelT")

_FACE_DETECTOR_MODEL = (
    "blaze_face_full_range.tflite",
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_full_range/float16/1/blaze_face_full_range.tflite",
)
_FACE_LANDMARKER_MODEL = (
    "face_landmarker.task",
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task",
)


def _mediapipe_cache_dir() -> Path:
    override = os.environ.get("MVD_MEDIAPIPE_CACHE_DIR", "").strip()
    if override:
        return Path(override).expanduser()
    xdg_cache = os.environ.get("XDG_CACHE_HOME", "").strip()
    base = Path(xdg_cache).expanduser() if xdg_cache else Path.home() / ".cache"
    return base / "mvdirector" / "mediapipe"


def _ensure_mediapipe_model(filename: str, url: str, model_path: Path | None = None) -> Path:
    return download_cached_model(
        filename,
        url,
        _mediapipe_cache_dir(),
        model_path=model_path,
        label="MediaPipe model",
        env_var="MVD_MEDIAPIPE_CACHE_DIR",
        timeout=60.0,
        client_factory=httpx.Client,
    )


def _load_mediapipe_model(
    create: Callable[[Path], _ModelT], filename: str, url: str, model_path: Path | None
) -> tuple[_ModelT, Path]:
    """Resolve a model file and build the task, re-downloading a corrupt cached copy once."""
    resolved = _ensure_mediapipe_model(filename, url, model_path=model_path)
    try:
        return create(resolved), resolved
    except (RuntimeError, ValueError, OSError) as exc:
        if model_path is not None:
            raise DependencyMissing(
                f"MediaPipe could not load the model file {resolved}: {exc}"
            ) from exc
        log.warning(
            "Cached MediaPipe model %s failed to load (%s); downloading again", resolved, exc
        )
        try:
            resolved.unlink(missing_ok=True)
        except OSError as unlink_exc:
            raise DependencyMissing(
                f"Cached MediaPipe model {resolved} failed to load ({exc}) and could not be "
                f"removed ({unlink_exc}). Delete or replace it, or set "
                "qc.on_missing_dependency: skip in project.yaml."
            ) from exc
        resolved = _ensure_mediapipe_model(filename, url)
        try:
            return create(resolved), resolved
        except (RuntimeError, ValueError, OSError) as retry_exc:
            with contextlib.suppress(OSError):
                resolved.unlink(missing_ok=True)
            raise DependencyMissing(
                f"MediaPipe could not load the freshly downloaded model {resolved}: {retry_exc}. "
                "Retry, or set qc.on_missing_dependency: skip in project.yaml."
            ) from retry_exc


def _import_mediapipe_tasks(feature: str) -> tuple[Any, Any, Any]:
    try:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        _ = mp.Image, mp.ImageFormat, vision.FaceDetector, vision.FaceLandmarker
        return mp, BaseOptions, vision
    except (ImportError, AttributeError) as exc:
        raise DependencyMissing(
            f"mediapipe with the Tasks Vision API ({feature}) is not available. "
            "Install the 'qc' extra to enable face/lip-sync checks, or set "
            "qc.on_missing_dependency: skip in project.yaml."
        ) from exc


def _validate_rgb_image(image: np.ndarray) -> bool:
    """Validate an HxWx3 RGB array; return False if empty, raise ValueError if malformed."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected an HxWx3 RGB image, got shape {image.shape}")
    return image.shape[0] > 0 and image.shape[1] > 0


def _to_mp_image(mp_module: Any, image: np.ndarray) -> Any:
    rgb = np.ascontiguousarray(image, dtype=np.uint8)
    return mp_module.Image(image_format=mp_module.ImageFormat.SRGB, data=rgb)


@dataclass(frozen=True)
class Face:
    height_fraction: float  # face box height / frame height
    embedding: np.ndarray | None = None  # L2-normalised identity vector, when available
    bbox: tuple[int, int, int, int] | None = None  # (x1, y1, x2, y2) in frame pixels, when known
    confidence: float | None = None  # detector confidence in [0, 1], when reported


class FaceAnalyzer(Protocol):
    def faces(self, image: np.ndarray) -> list[Face]:
        """Faces in an RGB image, largest first."""
        ...


class MouthTracker(Protocol):
    def track(self, frames: list[np.ndarray]) -> np.ndarray:
        """Mouth opening per frame (lip gap / face height); NaN where no face is found."""
        ...


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator > 0 else 0.0


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


class IdentityReference:
    """The singer's identity embedding, computed once per run.

    Given several images (the face and angled panels of a reference sheet), it averages their
    embeddings. A single frontal image under-scores a correct face seen at an angle; the average
    over several angles is a steadier target.
    """

    def __init__(self, analyzer: FaceAnalyzer, reference_images: Path | list[Path]) -> None:
        self._analyzer = analyzer
        self._paths = (
            [reference_images] if isinstance(reference_images, Path) else list(reference_images)
        )
        self._embedding: np.ndarray | None = None

    @property
    def paths(self) -> list[Path]:
        """Reference images, most reliable identity anchor first (face, front, three-quarter)."""
        return list(self._paths)

    @property
    def embedding(self) -> np.ndarray:
        if self._embedding is None:
            vectors = []
            for path in self._paths:
                faces = self._analyzer.faces(load_rgb(path))
                if faces and faces[0].embedding is not None:
                    vector = np.asarray(faces[0].embedding, dtype=np.float64)
                    vectors.append(vector / max(float(np.linalg.norm(vector)), 1e-12))
            if not vectors:
                names = ", ".join(path.name for path in self._paths)
                raise ValueError(f"No face found in the reference image(s): {names}")
            mean = np.mean(vectors, axis=0)
            self._embedding = mean / max(float(np.linalg.norm(mean)), 1e-12)
        return self._embedding

    def similarity(self, face: Face) -> float | None:
        if face.embedding is None:
            return None
        return cosine_similarity(self.embedding, face.embedding)


class MediaPipeFaceAnalyzer:
    """Commercially permissive (Apache-2.0) FaceAnalyzer using MediaPipe Face Detector.

    Detects faces and measures ``height_fraction`` for ``face_present`` and ``face_framing``
    without non-commercial embedding weights (leaving identity checks to ``vision_character``).
    """

    def __init__(self, model_path: Path | None = None) -> None:
        mp, base_options_cls, vision = _import_mediapipe_tasks("FaceDetector")
        self._mp = mp

        def create(path: Path) -> Any:
            options = vision.FaceDetectorOptions(
                base_options=base_options_cls(model_asset_path=str(path)),
                running_mode=vision.RunningMode.IMAGE,
                min_detection_confidence=0.5,
            )
            return vision.FaceDetector.create_from_options(options)

        self._detector, _ = _load_mediapipe_model(create, *_FACE_DETECTOR_MODEL, model_path)
        self._closed = False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._detector.close()

    def __enter__(self) -> MediaPipeFaceAnalyzer:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def faces(self, image: np.ndarray) -> list[Face]:
        if self._closed:
            raise RuntimeError("MediaPipeFaceAnalyzer has been closed")
        if not _validate_rgb_image(image):
            return []
        frame_height, frame_width = image.shape[0], image.shape[1]
        result = self._detector.detect(_to_mp_image(self._mp, image))
        found: list[Face] = []
        for detection in result.detections or []:
            box = detection.bounding_box
            height_fraction = float(np.clip(box.height / frame_height, 0.0, 1.0))
            if height_fraction <= 0:
                continue
            bbox: tuple[int, int, int, int] | None = None
            origin_x = getattr(box, "origin_x", None)
            origin_y = getattr(box, "origin_y", None)
            box_w = getattr(box, "width", None)
            if origin_x is not None and origin_y is not None and box_w is not None:
                x1 = max(0, min(frame_width, int(round(origin_x))))
                y1 = max(0, min(frame_height, int(round(origin_y))))
                x2 = max(x1, min(frame_width, int(round(origin_x + box_w))))
                y2 = max(y1, min(frame_height, int(round(origin_y + box.height))))
                if x2 > x1 and y2 > y1:
                    bbox = (x1, y1, x2, y2)
            cats = getattr(detection, "categories", None) or []
            raw_score = getattr(cats[0], "score", None) if cats else None
            conf = float(raw_score) if raw_score is not None else None
            found.append(
                Face(height_fraction=height_fraction, embedding=None, bbox=bbox, confidence=conf)
            )
        return sorted(found, key=lambda face: face.height_fraction, reverse=True)


class InsightFaceAnalyzer:
    """FaceAnalyzer using insightface's buffalo_l models on CPU (non-commercial weights)."""

    def __init__(self) -> None:
        try:
            from insightface.app import FaceAnalysis
        except ImportError as exc:
            raise DependencyMissing(
                "insightface is not installed. Install the 'insightface' extra to enable "
                "InsightFace embedding checks."
            ) from exc
        self._app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        self._app.prepare(ctx_id=-1, det_size=(640, 640))

    def faces(self, image: np.ndarray) -> list[Face]:
        if not _validate_rgb_image(image):
            return []
        height, width = image.shape[0], image.shape[1]
        found = []
        for f in self._app.get(image[:, :, ::-1]):  # insightface expects BGR
            x1 = max(0, min(width, int(round(float(f.bbox[0])))))
            y1 = max(0, min(height, int(round(float(f.bbox[1])))))
            x2 = max(x1, min(width, int(round(float(f.bbox[2])))))
            y2 = max(y1, min(height, int(round(float(f.bbox[3])))))
            height_fraction = float(np.clip(float(f.bbox[3] - f.bbox[1]) / height, 0.0, 1.0))
            embedding = np.asarray(f.normed_embedding) if f.normed_embedding is not None else None
            found.append(
                Face(
                    height_fraction=height_fraction,
                    embedding=embedding,
                    bbox=(x1, y1, x2, y2) if (x2 > x1 and y2 > y1) else None,
                )
            )
        return sorted(found, key=lambda face: face.height_fraction, reverse=True)


class MediaPipeMouthTracker:
    """MouthTracker using MediaPipe Face Landmarker."""

    # Face Landmarker landmark indices: inner upper/lower lip, forehead top, chin, nose tip.
    _UPPER_LIP, _LOWER_LIP, _FOREHEAD, _CHIN, _CENTER = 13, 14, 10, 152, 1

    def __init__(self, model_path: Path | None = None) -> None:
        mp, base_options_cls, vision = _import_mediapipe_tasks("FaceLandmarker")
        self._mp = mp
        self._base_options_cls = base_options_cls
        self._vision = vision

        def probe(path: Path) -> Path:
            # Open the landmarker once so a corrupt cached model fails here, not mid-run.
            with vision.FaceLandmarker.create_from_options(self._options(path, 1)):
                return path

        _, self._model_path = _load_mediapipe_model(probe, *_FACE_LANDMARKER_MODEL, model_path)

    def _options(self, path: Path, num_faces: int) -> Any:
        return self._vision.FaceLandmarkerOptions(
            base_options=self._base_options_cls(model_asset_path=str(path)),
            running_mode=self._vision.RunningMode.VIDEO,
            num_faces=num_faces,
        )

    @classmethod
    def _mouth_measurement(cls, points: list[Any]) -> tuple[float, float] | None:
        if len(points) <= cls._CHIN:
            return None
        chin_y = points[cls._CHIN].y
        forehead_y = points[cls._FOREHEAD].y
        upper_y = points[cls._UPPER_LIP].y
        lower_y = points[cls._LOWER_LIP].y
        center_x = points[cls._CENTER].x
        if None in (chin_y, forehead_y, upper_y, lower_y, center_x):
            return None
        face_height = abs(chin_y - forehead_y)
        if face_height <= 0:
            return None
        lip_gap = abs(lower_y - upper_y)
        return float(center_x), float(lip_gap / face_height)

    def track(self, frames: list[np.ndarray]) -> np.ndarray:
        return self.track_all(frames, max_faces=1)[0]

    def track_all(self, frames: list[np.ndarray], max_faces: int = 2) -> list[np.ndarray]:
        """Track up to ``max_faces`` mouths across frames, ordered stably across frames."""
        limit = max(1, max_faces)
        tracks = [np.full(len(frames), np.nan) for _ in range(limit)]
        if not frames:
            return tracks[:1]
        anchors: list[float | None] = [None] * limit
        options = self._options(self._model_path, limit)
        with self._vision.FaceLandmarker.create_from_options(options) as landmarker:
            for index, frame in enumerate(frames):
                if not _validate_rgb_image(frame):
                    continue
                # VIDEO mode requires strictly monotonically increasing timestamps in milliseconds.
                result = landmarker.detect_for_video(_to_mp_image(self._mp, frame), index * 33)
                if not result.face_landmarks:
                    continue
                detections = [
                    m
                    for face in result.face_landmarks
                    if (m := self._mouth_measurement(face)) is not None
                ]
                detections.sort(key=lambda item: item[0])
                if limit == 1:
                    if detections:
                        tracks[0][index] = detections[0][1]
                    continue
                used: set[int] = set()
                for x_pos, openness in detections:
                    initialized = [
                        (i, anchors[i])
                        for i in range(limit)
                        if i not in used and anchors[i] is not None
                    ]
                    if initialized:
                        slot = min(initialized, key=lambda pair: abs(x_pos - pair[1]))[0]  # type: ignore[operator]
                        if abs(x_pos - anchors[slot]) > 0.25:  # type: ignore[operator]
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
                    anchors[slot] = x_pos
                    tracks[slot][index] = openness
        active = [t for t, a in zip(tracks, anchors) if a is not None]
        return active or tracks[:1]
