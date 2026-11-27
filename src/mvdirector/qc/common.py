# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Checks shared by the image and video quality gates."""

from __future__ import annotations

import logging
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
import numpy as np
from PIL import Image, UnidentifiedImageError

from mvdirector.errors import (
    DependencyMissing,
    ImagesNotSupportedError,
    LLMError,
    LLMUnavailableError,
)
from mvdirector.models import AssetKind, CheckResult, CheckStatus, Manifest, Shot
from mvdirector.qc.scoring import floor_passing_score, linear_score, skipped
from mvdirector.qc.vision import VisionJudge

if TYPE_CHECKING:  # faces.py imports this module, so only import its types for checking
    from mvdirector.qc.faces import Face, IdentityReference

log = logging.getLogger(__name__)

_MAX_MODEL_DOWNLOAD_BYTES = 2 * 1024 * 1024 * 1024  # 2 GiB safety cap on model downloads


def download_cached_model(
    filename: str,
    url: str,
    cache_dir: Path,
    *,
    model_path: Path | None = None,
    label: str = "model",
    env_var: str | None = None,
    timeout: float = 60.0,
    client_factory: Callable[..., Any] | None = None,
) -> Path:
    """Resolve an explicit model path or atomically download and cache it under ``cache_dir``."""
    if (
        not filename
        or "\x00" in filename
        or Path(filename).name != filename
        or filename in (".", "..")
    ):
        raise DependencyMissing(f"Invalid {label} filename: {filename!r}")
    if model_path is not None:
        raw = str(model_path)
        if "\x00" in raw:
            raise DependencyMissing(f"Invalid {label} path: null byte not allowed")
        resolved = model_path.expanduser()
        if resolved.is_file() and resolved.stat().st_size > 0:
            return resolved
        raise DependencyMissing(f"{label} file not found: {resolved}")

    target = cache_dir / filename
    if target.is_file() and target.stat().st_size > 0:
        return target

    log.info("Downloading %s %s to %s", label, filename, target)
    make_client = client_factory or httpx.Client
    tmp_path: Path | None = None
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=target.parent, prefix=f".{filename}.", suffix=".part", delete=False
        ) as handle:
            tmp_path = Path(handle.name)
            with (
                make_client(follow_redirects=True, timeout=timeout) as client,
                client.stream("GET", url) as response,
            ):
                response.raise_for_status()
                encoding = response.headers.get("content-encoding", "identity").strip().lower()
                expected = (
                    response.headers.get("content-length") if encoding in ("", "identity") else None
                )
                if expected and expected.isdigit() and int(expected) > _MAX_MODEL_DOWNLOAD_BYTES:
                    raise DependencyMissing(
                        f"Refusing to download {label} '{filename}': reported size {expected} "
                        f"bytes exceeds the {_MAX_MODEL_DOWNLOAD_BYTES}-byte safety limit."
                    )
                written = 0
                for chunk in response.iter_bytes():
                    written += len(chunk)
                    if written > _MAX_MODEL_DOWNLOAD_BYTES:
                        raise DependencyMissing(
                            f"Aborted download of {label} '{filename}': exceeded "
                            f"{_MAX_MODEL_DOWNLOAD_BYTES}-byte safety limit."
                        )
                    handle.write(chunk)
        size = tmp_path.stat().st_size
        if expected and expected.isdigit() and size != int(expected):
            raise DependencyMissing(
                f"Incomplete download of {label} '{filename}' ({size} of {expected} "
                f"bytes). Retry, or place the file at {target}."
            )
        if size == 0:
            raise DependencyMissing(
                f"Downloaded empty {label} from {url}. "
                f"Place the file at {target} or set qc.on_missing_dependency: skip."
            )
        # NamedTemporaryFile uses 0600; make cached models readable by non-root container users.
        os.chmod(tmp_path, 0o644)
        os.replace(tmp_path, target)
        tmp_path = None
        return target
    except DependencyMissing:
        raise
    except (httpx.HTTPError, OSError) as exc:
        env_hint = f" (or set {env_var})" if env_var else ""
        raise DependencyMissing(
            f"Failed to download {label} '{filename}' from {url} ({exc}). "
            f"Place the file at {target}{env_hint}, or set "
            "qc.on_missing_dependency: skip in project.yaml."
        ) from exc
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


def dimensions_check(
    width: int,
    height: int,
    expected_width: int,
    expected_height: int,
    aspect_tolerance: float,
    min_scale: float,
) -> CheckResult:
    """Hard check: right aspect ratio and not far below the target resolution."""
    if width <= 0 or height <= 0 or expected_width <= 0 or expected_height <= 0:
        return CheckResult(
            name="dimensions",
            status=CheckStatus.FAILED,
            hard=True,
            detail=(
                f"invalid dimensions {width}x{height}, expected about "
                f"{expected_width}x{expected_height}"
            ),
            metrics={"width": width, "height": height},
        )
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


#: Faces compared per frame by the vision identity check (largest first); covers duets.
_VISION_IDENTITY_MAX_FACES = 2
#: Reference panels sent with each comparison (face panel first, then an angled view).
_VISION_IDENTITY_MAX_REFERENCES = 2


def crop_face(image: np.ndarray, bbox: tuple[int, int, int, int], pad: float = 0.35) -> np.ndarray:
    """The face box with some forehead, chin and ear context, clamped to the frame."""
    x1, y1, x2, y2 = bbox
    height, width = image.shape[:2]
    if (
        height <= 0
        or width <= 0
        or x2 <= x1
        or y2 <= y1
        or x2 <= 0
        or y2 <= 0
        or x1 >= width
        or y1 >= height
    ):
        return image[0:0, 0:0]
    margin = max(0, int(round((y2 - y1) * pad)))
    top = max(0, min(height, y1 - margin))
    bottom = max(0, min(height, y2 + margin))
    left = max(0, min(width, x1 - margin))
    right = max(0, min(width, x2 + margin))
    return image[top:bottom, left:right]


_IMAGE_ERRORS = (OSError, UnidentifiedImageError, ValueError, Image.DecompressionBombError)


def _readable_image(path: Path) -> bool:
    """True when ``path`` is a non-empty regular file that Pillow recognizes as an image."""
    try:
        if not path.is_file() or path.stat().st_size <= 0:
            return False
        with Image.open(path) as opened:
            opened.verify()
        return True
    except _IMAGE_ERRORS:
        log.warning("Skipping unreadable reference face image: %s", path)
        return False


def _skip_incomplete_identity(
    last_error: LLMError | None,
    last_image_error: Exception | None,
    disabled_reason: str | None,
) -> CheckResult:
    if last_error is not None:
        return skipped("identity", f"vision identity check errored: {last_error}")
    if last_image_error is not None:
        return skipped("identity", f"vision identity image unreadable: {last_image_error}")
    if disabled_reason:
        return skipped("identity", disabled_reason)
    return skipped("identity", "no face available to compare")


def _disable_for_unsupported_images(judge: VisionJudge, exc: ImagesNotSupportedError) -> str:
    reason = f"the model does not accept images ({exc})"
    if judge.disable(reason):
        log.warning(
            "Vision checks are off for the rest of this run: %s. They catch wrong costumes, "
            "mirrored marks and malformed hands. Choose a model that accepts images, or set "
            "qc.vision_enabled: false to silence this.",
            reason,
        )
    return reason


def _record_vision_unavailable(judge: VisionJudge, exc: LLMError) -> None:
    if isinstance(exc, LLMUnavailableError) and judge.record_unavailable(str(exc)):
        log.warning(
            "Vision checks are off for the rest of this run: %s. Attempts made without them "
            "are recorded with the vision checks skipped. Fix the model (`mvd llm-check` "
            "tests it) and generate those shots again.",
            judge.disabled_reason,
        )


def vision_identity_check(
    judge: VisionJudge | None,
    samples: list[tuple[np.ndarray, list[Face]]],
    reference: IdentityReference | None,
    *,
    min_face_px: int,
    hard_min: float,
    good: float,
    pass_score: float = 0.0,
) -> CheckResult:
    """Identity from face crops judged by the vision model, for face backends without embeddings.

    ``samples`` are frames with their detected faces (one frame for a still, a few for a clip).
    Each frame scores its best-matching face, so a duet partner does not fail the check; frames
    are combined by the median, which tolerates one turned-away frame. Faces under
    ``min_face_px`` tall are not judged: at that size a different person in the same costume
    reads as a match.
    """
    if judge is None or reference is None:
        return skipped("identity", "no face embeddings and no vision model to compare faces")
    if judge.disabled_reason:
        return skipped("identity", judge.disabled_reason)
    refs: list[Path] = []
    for path in reference.paths:
        if _readable_image(path):
            refs.append(path)
            if len(refs) >= _VISION_IDENTITY_MAX_REFERENCES:
                break
    if not refs:
        return skipped("identity", "reference face image unavailable")
    judgeable: list[list[np.ndarray]] = []
    largest_px = 0
    for frame, faces in samples:
        frame_h, frame_w = frame.shape[:2]
        valid: list[tuple[int, np.ndarray]] = []
        for face in faces:
            if face.bbox is None:
                continue
            x1, y1, x2, y2 = face.bbox
            if y2 <= y1 or x2 <= x1:
                continue
            vis_h = min(frame_h, y2) - max(0, y1)
            vis_w = min(frame_w, x2) - max(0, x1)
            if vis_h <= 0 or vis_w <= 0:
                continue
            crop = crop_face(frame, face.bbox)
            if crop.size == 0:
                continue
            valid.append((vis_h, crop))
        if not valid:
            continue
        valid.sort(key=lambda item: item[0], reverse=True)
        largest_px = max(largest_px, valid[0][0])
        sized = [crop for h, crop in valid if h >= min_face_px]
        if sized:
            judgeable.append(sized[:_VISION_IDENTITY_MAX_FACES])
    if not judgeable:
        if largest_px == 0:
            return skipped("identity", "no face available to compare")
        return skipped(
            "identity",
            f"face too small to judge identity ({largest_px}px tall; needs {min_face_px}px)",
        )
    frame_scores: list[float] = []
    reasons: list[str] = []
    last_error: LLMError | None = None
    last_image_error: Exception | None = None
    for crops in judgeable:
        best: tuple[float, str] | None = None
        for crop in crops:
            if judge.disabled_reason:
                if best is not None and best[0] < hard_min:
                    best = None
                break
            try:
                verdict = judge.judge_face_identity(crop, refs)
            except ImagesNotSupportedError as exc:
                reason = _disable_for_unsupported_images(judge, exc)
                return skipped("identity", reason)
            except (*_IMAGE_ERRORS, TypeError) as exc:
                log.warning("Vision identity check could not encode image: %s", exc)
                last_image_error = exc
                if best is not None and best[0] < hard_min:
                    best = None
                break
            except LLMError as exc:
                # A flaky judge must not block the pipeline; the other checks still apply.
                log.warning("Vision identity check failed: %s", exc)
                _record_vision_unavailable(judge, exc)
                last_error = exc
                # Keep an already-passing match (e.g. face 1 in a duet), but never hard-fail on
                # a non-matching face 1 when face 2 could not be checked.
                if best is not None and best[0] < hard_min:
                    best = None
                break
            if best is None or verdict.same_person > best[0]:
                best = (verdict.same_person, verdict.reason)
            if best[0] >= good:
                # Already a clear match for this frame; skip remaining duet faces.
                break
        if best is not None:
            frame_scores.append(best[0])
            reasons.append(best[1])
        if judge.disabled_reason:
            break
    if not frame_scores:
        return _skip_incomplete_identity(last_error, last_image_error, judge.disabled_reason)
    missing = len(judgeable) - len(frame_scores)
    score = float(np.median(frame_scores))
    failed = score < hard_min
    if failed and missing > 0:
        # Never hard-fail a multi-frame clip on incomplete evidence when unjudged frames
        # (dropped by a flaky judge or encoding error) could have lifted the median to pass.
        optimistic = float(np.median([*frame_scores, *([1.0] * missing)]))
        if optimistic >= hard_min:
            return _skip_incomplete_identity(last_error, last_image_error, judge.disabled_reason)
    worst_reason = reasons[int(np.argmin(frame_scores))]
    check_score = 0.0 if failed else floor_passing_score(score, hard_min, good, pass_score)
    return CheckResult(
        name="identity",
        status=CheckStatus.FAILED if failed else CheckStatus.PASSED,
        hard=True,
        score=check_score,
        detail=f"face match to reference {score:.2f} (vision): {worst_reason}" if failed else "",
        fix_hint=(
            "Keep the singer's face matching the reference sheet: same face shape, eyes, brows, "
            "nose and jaw; avoid extreme angles or anything covering the face."
        )
        if failed
        else None,
        metrics={"vision_identity": round(score, 4), "frames_judged": float(len(frame_scores))},
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
    *,
    spec_hard_min: float | None = None,
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
        reason = _disable_for_unsupported_images(judge, exc)
        return [skipped(name, reason) for name in names]
    except LLMError as exc:
        # A flaky judge must not block the pipeline; the other checks still apply.
        log.warning("Vision check failed for %s: %s", shot.id, exc)
        _record_vision_unavailable(judge, exc)
        return [skipped(name, f"vision check errored: {exc}") for name in names]

    issues = "; ".join(verdict.issues)
    effective_spec_min = spec_hard_min if spec_hard_min is not None else hard_min
    scored = [
        ("vision_spec", verdict.spec_match, True, effective_spec_min),
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
                metrics={name: round(value, 4)},
            )
        )
    return results
