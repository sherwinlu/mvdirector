# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Quality gate for stills. Runs before video generation, where failures are cheapest."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image

from mvdirector.config import QCConfig, VideoConfig
from mvdirector.models import (
    AssetKind,
    CheckResult,
    CheckStatus,
    Manifest,
    QCReport,
    Shot,
    ShotType,
)
from mvdirector.qc.common import dimensions_check, identity_check, vision_checks
from mvdirector.qc.faces import FaceAnalyzer, IdentityReference, load_rgb
from mvdirector.qc.scoring import aggregate, linear_score, skipped
from mvdirector.qc.vision import VisionJudge

log = logging.getLogger(__name__)

_SHARPNESS_SIDE = 512


def sharpness(image: np.ndarray) -> float:
    """Variance of the Laplacian of a downscaled greyscale copy; low means blurry."""
    grey = Image.fromarray(image).convert("L")
    grey.thumbnail((_SHARPNESS_SIDE, _SHARPNESS_SIDE))
    pixels = np.asarray(grey, dtype=np.float64)
    laplacian = (
        -4.0 * pixels[1:-1, 1:-1]
        + pixels[:-2, 1:-1]
        + pixels[2:, 1:-1]
        + pixels[1:-1, :-2]
        + pixels[1:-1, 2:]
    )
    return float(laplacian.var())


class ImageQC:
    def __init__(
        self,
        qc: QCConfig,
        video: VideoConfig,
        face_analyzer: FaceAnalyzer | None,
        vision: VisionJudge | None,
    ) -> None:
        self._qc = qc
        self._cfg = qc.image
        self._video = video
        self._faces = face_analyzer
        self._vision = vision

    def evaluate(
        self,
        path: Path,
        shot: Shot,
        manifest: Manifest,
        reference: IdentityReference | None,
        *,
        reference_image: Path | None = None,
        extra_references: list[tuple[str, Path]] | None = None,
        expected_size: tuple[int, int] | None = None,
    ) -> QCReport:
        image = load_rgb(path)
        height, width = image.shape[:2]
        target_w, target_h = expected_size or (self._video.width, self._video.height)
        checks = [
            dimensions_check(
                width,
                height,
                target_w,
                target_h,
                self._cfg.aspect_tolerance,
                self._cfg.min_resolution_scale,
            ),
            self._sharpness_check(image),
        ]
        if shot.type.shows_singer:
            checks += self._face_checks(image, shot, manifest, reference)
        else:
            checks += self._broll_face_checks(image)
        checks += vision_checks(
            self._vision,
            [path],
            shot,
            manifest,
            AssetKind.STILL,
            self._cfg.vision_hard_min,
            self._cfg.vision_character_hard_min,
            reference_image,
            extra_references=extra_references,
            spec_hard_min=self._cfg.vision_spec_hard_min,
        )
        return aggregate(checks, self._qc.weights, self._qc.pass_score)

    def _broll_face_checks(self, image: np.ndarray) -> list[CheckResult]:
        if self._faces is None:
            return []
        faces = self._faces.faces(image)
        biggest = faces[0].height_fraction if faces else 0.0
        unexpected = biggest >= self._cfg.broll_max_face_fraction
        return [
            CheckResult(
                name="no_people",
                status=CheckStatus.FAILED if unexpected else CheckStatus.PASSED,
                hard=True,
                score=0.0 if unexpected else None,
                detail=(
                    f"unexpected face detected in cutaway ({biggest:.0%} of frame height)"
                    if unexpected
                    else ""
                ),
                fix_hint=(
                    "Remove all people, faces, and human figures; show only the empty location "
                    "or object."
                    if unexpected
                    else None
                ),
                metrics={"broll_face_height_fraction": round(biggest, 4)},
            )
        ]

    def _sharpness_check(self, image: np.ndarray) -> CheckResult:
        value = sharpness(image)
        score = linear_score(value, self._cfg.sharpness_bad, self._cfg.sharpness_good)
        failed = value < self._cfg.sharpness_bad
        return CheckResult(
            name="sharpness",
            status=CheckStatus.FAILED if failed else CheckStatus.PASSED,
            score=score,
            detail="image is blurry" if failed else "",
            fix_hint="Ask for sharp focus on the singer's face." if failed else None,
            metrics={"laplacian_variance": round(value, 2)},
        )

    def _face_checks(
        self,
        image: np.ndarray,
        shot: Shot,
        manifest: Manifest,
        reference: IdentityReference | None,
    ) -> list[CheckResult]:
        if self._faces is None:
            return [
                skipped(name, "face analysis unavailable")
                for name in ("face_present", "face_framing", "identity")
            ]
        faces = self._faces.faces(image)
        if not faces:
            return [
                CheckResult(
                    name="face_present",
                    status=CheckStatus.FAILED,
                    hard=True,
                    detail="no face detected",
                    fix_hint="Show the singer facing the camera with the face clearly visible.",
                )
            ]
        singing_count = len(manifest.singing_characters(shot))
        if singing_count >= 2 and len(faces) < 2:
            return [
                CheckResult(
                    name="face_present",
                    status=CheckStatus.FAILED,
                    hard=True,
                    detail="only 1 face detected; shot requires both singers visible",
                    fix_hint=(
                        "Show both singers clearly in a medium close-up two-shot with both faces "
                        "and mouths unobstructed."
                    ),
                    metrics={"detected_faces": float(len(faces))},
                )
            ]
        scored_faces = (
            [(f, s) for f in faces if (s := reference.similarity(f)) is not None]
            if reference is not None
            else []
        )
        singer, similarity = (
            max(scored_faces, key=lambda pair: pair[1]) if scored_faces else (faces[0], None)
        )
        minimum = self._cfg.min_face_fraction.get(shot.framing, 0.0)
        maximum = self._cfg.max_face_fraction.get(shot.framing, 1.0)
        largest_fraction = (
            max(singer.height_fraction, faces[0].height_fraction)
            if singing_count >= 2
            else singer.height_fraction
        )
        co_singer_minimum = minimum * 0.75
        co_singer_small = (
            singing_count >= 2
            and min(singer.height_fraction, faces[1].height_fraction) < co_singer_minimum
        )
        too_small = singer.height_fraction < minimum if singing_count < 2 else co_singer_small
        too_large = largest_fraction > maximum
        if too_large:
            framing = CheckResult(
                name="face_framing",
                status=CheckStatus.FAILED,
                hard=True,
                score=linear_score(largest_fraction, maximum * 1.5, maximum),
                detail=(
                    f"face is {largest_fraction:.0%} of frame height; "
                    f"{shot.framing.value} allows at most {maximum:.0%} "
                    "(subject is too large in frame or vertically squashed)"
                ),
                fix_hint=(
                    f"Pull back to a true {shot.framing.value.replace('_', ' ')} shot with "
                    "realistic human scale against architecture and natural full-length body "
                    "proportions."
                ),
                metrics={"face_height_fraction": round(largest_fraction, 4)},
            )
        elif co_singer_small:
            smaller = min(singer.height_fraction, faces[1].height_fraction)
            framing = CheckResult(
                name="face_framing",
                status=CheckStatus.FAILED,
                hard=True,
                score=linear_score(smaller, co_singer_minimum * 0.5, co_singer_minimum),
                detail=(
                    f"co-singer face is {smaller:.0%} of frame height; "
                    f"{shot.framing.value} two-shot needs at least {co_singer_minimum:.0%}"
                ),
                fix_hint=(
                    "Stage both singers side by side at similar distance in a medium close-up "
                    "two-shot so both mouths are large enough to lip-sync."
                ),
                metrics={
                    "face_height_fraction": round(singer.height_fraction, 4),
                    "second_face_height_fraction": round(faces[1].height_fraction, 4),
                },
            )
        else:
            framing = CheckResult(
                name="face_framing",
                status=CheckStatus.FAILED if too_small else CheckStatus.PASSED,
                # The mouth must be large enough to animate, so this is hard for lip-sync shots.
                hard=shot.type is ShotType.LIPSYNC,
                score=linear_score(singer.height_fraction, minimum * 0.5, minimum),
                detail=(
                    f"face is {singer.height_fraction:.0%} of frame height; "
                    f"{shot.framing.value} needs at least {minimum:.0%}"
                )
                if too_small
                else "",
                fix_hint=(
                    f"Frame tighter: a true {shot.framing.value.replace('_', ' ')} on the singer."
                    if too_small
                    else None
                ),
                metrics={"face_height_fraction": round(singer.height_fraction, 4)},
            )
        present = CheckResult(name="face_present", status=CheckStatus.PASSED, hard=True)
        identity = identity_check(similarity, self._cfg.identity_hard_min, self._cfg.identity_good)
        return [present, framing, identity]
