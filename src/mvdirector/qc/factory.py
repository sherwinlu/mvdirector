# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Assemble the quality gates from config, handling optional dependencies in one place."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from mvdirector.config import ProjectConfig
from mvdirector.errors import DependencyMissing
from mvdirector.llm import LLMClient
from mvdirector.qc.faces import (
    FaceAnalyzer,
    InsightFaceAnalyzer,
    MediaPipeFaceAnalyzer,
    MediaPipeMouthTracker,
    MouthTracker,
)
from mvdirector.qc.image import ImageQC
from mvdirector.qc.lipsync import LipSyncScorer
from mvdirector.qc.syncnet import StableSyncNetScorer
from mvdirector.qc.video import VideoQC
from mvdirector.qc.vision import VisionJudge

log = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass(frozen=True)
class QualityGates:
    image: ImageQC
    video: VideoQC
    face_analyzer: FaceAnalyzer | None
    mouth_tracker: MouthTracker | None = None
    lipsync_scorer: LipSyncScorer | None = None
    vision_enabled: bool = False
    face_backend: str = "mediapipe"
    lipsync_backend: str = "mediapipe"
    vision_reason: str | None = None

    def close(self) -> None:
        """Release native resources held by the face and lip-sync backends (a no-op for fakes)."""
        for component in (self.lipsync_scorer, self.face_analyzer, self.mouth_tracker):
            closer = getattr(component, "close", None)
            if callable(closer):
                closer()

    def check_summary(self) -> tuple[list[str], list[tuple[str, str]]]:
        """Return ``(active_checks, skipped_checks_with_reason)`` for run startup reporting."""
        active = ["dimensions", "sharpness", "duration", "temporal"]
        skipped: list[tuple[str, str]] = []
        if self.face_analyzer is not None:
            active.extend(["face_present", "face_framing", "no_people"])
        else:
            skipped.append(
                (
                    "face_present, face_framing, no_people",
                    f"{self.face_backend} unavailable",
                )
            )
        if self.lipsync_scorer is not None or self.mouth_tracker is not None:
            active.extend(["face_presence", "lipsync"])
        else:
            skipped.append(("face_presence, lipsync", f"{self.lipsync_backend} unavailable"))
        if self.face_backend == "insightface" and self.face_analyzer is not None:
            active.append("identity")
        elif self.face_backend == "mediapipe":
            skipped.append(
                (
                    "identity",
                    "mediapipe omits face embeddings; vision_character checks identity",
                )
            )
        else:
            skipped.append(("identity", "insightface unavailable"))
        if self.vision_enabled:
            active.extend(["vision_spec", "vision_artifacts", "vision_character"])
        else:
            skipped.append(
                (
                    "vision_spec, vision_artifacts, vision_character",
                    self.vision_reason or "vision checks disabled",
                )
            )
        return active, skipped


def _optional(factory: Callable[[], T], policy: str) -> T | None:
    try:
        return factory()
    except DependencyMissing as exc:
        if policy == "fail":
            raise
        log.warning("%s The related checks will be recorded as skipped.", exc)
        return None


def build_quality_gates(config: ProjectConfig, llm: LLMClient | None) -> QualityGates:
    policy = config.qc.on_missing_dependency
    face_cls: Callable[[], FaceAnalyzer] = (
        InsightFaceAnalyzer if config.qc.face_backend == "insightface" else MediaPipeFaceAnalyzer
    )
    faces: FaceAnalyzer | None = _optional(face_cls, policy)
    mouth: MouthTracker | None = None
    scorer: LipSyncScorer | None = None
    try:
        if config.qc.lipsync_backend == "stable_syncnet":
            model_path = (
                Path(config.qc.syncnet_model_path) if config.qc.syncnet_model_path else None
            )
            shared_mp = faces if isinstance(faces, MediaPipeFaceAnalyzer) else None
            scorer = _optional(
                lambda: StableSyncNetScorer(
                    model_path=model_path,
                    device=config.qc.syncnet_device,
                    face_analyzer=shared_mp,
                ),
                policy,
            )
        else:
            mouth = _optional(MediaPipeMouthTracker, policy)
    except Exception:
        closer = getattr(faces, "close", None)
        if callable(closer):
            closer()
        raise
    vision = VisionJudge(llm) if (llm is not None and config.qc.vision_enabled) else None
    vision_reason: str | None = None
    if vision is None:
        vision_reason = "llm.enabled is false" if llm is None else "qc.vision_enabled is false"
        log.warning("Vision checks are off (%s).", vision_reason)
    return QualityGates(
        image=ImageQC(config.qc, config.video, faces, vision),
        video=VideoQC(config.qc, config.video, faces, mouth, vision, lipsync_scorer=scorer),
        face_analyzer=faces,
        mouth_tracker=mouth,
        lipsync_scorer=scorer,
        vision_enabled=vision is not None,
        face_backend=config.qc.face_backend,
        lipsync_backend=config.qc.lipsync_backend,
        vision_reason=vision_reason,
    )
