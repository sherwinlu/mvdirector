# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Checks shared by the image and video quality gates."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from mvdirector.errors import ImagesNotSupportedError, LLMError, LLMUnavailableError
from mvdirector.models import AssetKind, CheckResult, CheckStatus, Manifest, Shot
from mvdirector.qc.scoring import linear_score, skipped
from mvdirector.qc.vision import VisionJudge

log = logging.getLogger(__name__)


def dimensions_check(
    width: int,
    height: int,
    expected_width: int,
    expected_height: int,
    aspect_tolerance: float,
    min_scale: float,
) -> CheckResult:
    """Hard check: right aspect ratio and not far below the target resolution."""
    expected_aspect = expected_width / expected_height
    aspect_error = abs(width / height - expected_aspect) / expected_aspect
    scale = height / expected_height
    ok = aspect_error <= aspect_tolerance and scale >= min_scale
    return CheckResult(
        name="dimensions",
        status=CheckStatus.PASSED if ok else CheckStatus.FAILED,
        hard=True,
        detail=""
        if ok
        else f"got {width}x{height}, expected about {expected_width}x{expected_height}",
        metrics={"width": width, "height": height},
    )


def identity_check(similarity: float | None, hard_min: float, good: float) -> CheckResult:
    """Compare the singer's face with the locked reference portrait."""
    if similarity is None:
        return skipped("identity", "no face available to compare")
    failed = similarity < hard_min
    return CheckResult(
        name="identity",
        status=CheckStatus.FAILED if failed else CheckStatus.PASSED,
        hard=True,
        score=linear_score(similarity, hard_min, good),
        detail=f"face similarity to reference {similarity:.2f}" if failed else "",
        fix_hint=(
            "Keep the singer's face matching the reference portrait: avoid extreme angles, "
            "heavy shadow or anything covering the face."
        )
        if failed
        else None,
        metrics={"similarity": round(similarity, 4)},
    )


def vision_checks(
    judge: VisionJudge | None,
    images: list[Path | np.ndarray],
    shot: Shot,
    manifest: Manifest,
    kind: AssetKind,
    hard_min: float,
    character_hard_min: float,
    reference_image: Path | None = None,
    extra_references: list[tuple[str, Path]] | None = None,
) -> list[CheckResult]:
    """Three results from one model call: shot spec, artifacts and the character's identity."""
    names = ("vision_spec", "vision_artifacts", "vision_character")
    if judge is None:
        return [skipped(name, "vision check disabled") for name in names]
    if judge.disabled_reason:
        return [skipped(name, judge.disabled_reason) for name in names]
    try:
        if extra_references:
            verdict = judge.judge(
                images, shot, manifest, kind, reference_image, extra_references=extra_references
            )
        else:
            verdict = judge.judge(images, shot, manifest, kind, reference_image)
    except ImagesNotSupportedError as exc:
        reason = f"the model does not accept images ({exc})"
        judge.disable(reason)
        log.warning(
            "Vision checks are off for the rest of this run: %s. They catch wrong costumes, "
            "mirrored marks and malformed hands. Choose a model that accepts images, or set "
            "qc.vision_enabled: false to silence this.",
            reason,
        )
        return [skipped(name, reason) for name in names]
    except LLMError as exc:
        # A flaky judge must not block the pipeline; the other checks still apply.
        log.warning("Vision check failed for %s: %s", shot.id, exc)
        if isinstance(exc, LLMUnavailableError) and judge.record_unavailable(str(exc)):
            log.warning(
                "Vision checks are off for the rest of this run: %s. Attempts made without them "
                "are recorded with the vision checks skipped. Fix the model (`mvd llm-check` "
                "tests it) and generate those shots again.",
                judge.disabled_reason,
            )
        return [skipped(name, f"vision check errored: {exc}") for name in names]

    issues = "; ".join(verdict.issues)
    scored = [
        ("vision_spec", verdict.spec_match, False, hard_min),
        ("vision_artifacts", verdict.artifact_free, True, hard_min),
        ("vision_character", verdict.character_match, True, character_hard_min),
    ]
    results = []
    for name, value, is_hard, minimum in scored:
        if value is None:
            results.append(skipped(name, "no singer expected in this shot"))
            continue
        failed = value < minimum
        results.append(
            CheckResult(
                name=name,
                status=CheckStatus.FAILED if failed else CheckStatus.PASSED,
                hard=is_hard,
                score=value,
                detail=issues if failed else "",
                fix_hint=(verdict.fix or None) if failed else None,
            )
        )
    return results
