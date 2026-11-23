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
from typing import Any

import httpx
import numpy as np

from mvdirector.errors import (
    DependencyMissing,
    ImagesNotSupportedError,
    LLMError,
    LLMUnavailableError,
)
from mvdirector.models import AssetKind, CheckResult, CheckStatus, Manifest, Shot
from mvdirector.qc.scoring import linear_score, skipped
from mvdirector.qc.vision import VisionJudge

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
