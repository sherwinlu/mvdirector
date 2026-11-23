# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Review policy: which shots a person must look at, and what their decisions do.

Quality checks and retries run the same way in both modes. The mode only changes
which finished shots are queued for a person:

- ``strict``: only shots that used up their attempts.
- ``light`` (default): those, plus a risk-weighted sample of shots that passed.
"""

from __future__ import annotations

import hashlib

from mvdirector.config import ProjectConfig, ReviewMode
from mvdirector.models import (
    AssetKind,
    AssetStatus,
    CheckStatus,
    Framing,
    Manifest,
    ProjectMode,
    Review,
    ReviewDecision,
    ReviewReason,
    Shot,
    utc_now,
)


def uses_stills_only(manifest: Manifest, config: ProjectConfig) -> bool:
    """True when the project generates only stills (mix mode with visual_source='stills')."""
    return manifest.mode is ProjectMode.MIX and config.mix.visual_source == "stills"


def is_sampled(shot_id: str, rate: float, seed: int) -> bool:
    """Deterministic sampling, so the same shots are picked on every run."""
    digest = hashlib.sha256(f"{seed}:{shot_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64 < rate


def review_reasons(shot: Shot, manifest: Manifest, config: ProjectConfig) -> list[ReviewReason]:
    """Why a finished shot needs a person; empty if it does not (or is not finished)."""
    stills_only = uses_stills_only(manifest, config)
    if stills_only:
        if shot.still.accepted_by_reviewer:
            return []
        if shot.still.status is AssetStatus.EXHAUSTED:
            return [ReviewReason.EXHAUSTED]
        if shot.still.status is not AssetStatus.PASSED or config.review.mode is ReviewMode.STRICT:
            return []
        review, reasons = config.review, []
        if review.include_retried and len(shot.still.attempts) > 1:
            reasons.append(ReviewReason.RETRIED)
        selected_still = shot.still.selected
        if (
            review.include_near_threshold
            and selected_still is not None
            and selected_still.qc is not None
            and selected_still.qc.score < config.qc.pass_score + review.near_threshold_margin
        ):
            reasons.append(ReviewReason.NEAR_THRESHOLD)
        if shot.type in manifest.escalated_types:
            reasons.append(ReviewReason.ESCALATED)
        rate = review.closeup_rate if shot.framing is Framing.CLOSE_UP else review.other_rate
        if is_sampled(shot.id, rate, config.seed):
            reasons.append(ReviewReason.SAMPLED)
        return reasons

    if AssetStatus.EXHAUSTED in (shot.still.status, shot.video.status):
        return [ReviewReason.EXHAUSTED]
    if shot.video.status is not AssetStatus.PASSED or config.review.mode is ReviewMode.STRICT:
        return []

    review, reasons = config.review, []
    if review.include_retried and (len(shot.still.attempts) > 1 or len(shot.video.attempts) > 1):
        reasons.append(ReviewReason.RETRIED)
    selected = shot.video.selected
    if (
        review.include_near_threshold
        and selected is not None
        and selected.qc is not None
        and selected.qc.score < config.qc.pass_score + review.near_threshold_margin
    ):
        reasons.append(ReviewReason.NEAR_THRESHOLD)
    if (
        selected is not None
        and selected.qc is not None
        and any(
            c.name == "lipsync" and c.status is CheckStatus.FAILED and not c.hard
            for c in selected.qc.checks
        )
    ):
        reasons.append(ReviewReason.LIPSYNC)
    if shot.fallback_from is not None:
        reasons.append(ReviewReason.FALLBACK)
    if shot.type in manifest.escalated_types:
        reasons.append(ReviewReason.ESCALATED)
    rate = review.closeup_rate if shot.framing is Framing.CLOSE_UP else review.other_rate
    if is_sampled(shot.id, rate, config.seed):
        reasons.append(ReviewReason.SAMPLED)
    return reasons


def refresh_reviews(manifest: Manifest, config: ProjectConfig) -> None:
    """Recompute the review queue. Decisions already made are left alone.

    Safe to call at any time, which is what makes switching modes mid-project safe.
    """
    for shot in manifest.shots:
        if shot.review.decision is not None:
            continue
        reasons = review_reasons(shot, manifest, config)
        shot.review = Review(required=bool(reasons), reasons=reasons)


def is_ready(shot: Shot, *, stills_only: bool = False) -> bool:
    """True when the shot has a selected asset that needs no further human input."""
    if stills_only:
        if shot.still.selected is None or shot.still.status is AssetStatus.PENDING:
            return False
        return (
            not shot.review.required
            or shot.review.decision is ReviewDecision.APPROVED
            or shot.still.accepted_by_reviewer
        )
    if shot.video.selected is None or shot.video.status is AssetStatus.PENDING:
        return False
    return not shot.review.required or shot.review.decision is ReviewDecision.APPROVED


def pending_review(manifest: Manifest) -> list[Shot]:
    return [s for s in manifest.shots if s.review.required and s.review.decision is None]


def approve(
    manifest: Manifest,
    shot_id: str,
    note: str | None = None,
    *,
    stills_only: bool = False,
) -> None:
    shot = manifest.shot(shot_id)
    if stills_only:
        if shot.still.selected is None:
            raise ValueError(f"{shot_id} has no still to approve")
    elif shot.video.selected is None:
        raise ValueError(f"{shot_id} has no video to approve")
    shot.review.required = True
    shot.review.decision = ReviewDecision.APPROVED
    shot.review.note = note
    shot.review.decided_at = utc_now()


def accept_still(manifest: Manifest, shot_id: str) -> None:
    """Accept the best attempt of a still that used up its attempts, unblocking its video."""
    shot = manifest.shot(shot_id)
    if shot.still.status is not AssetStatus.EXHAUSTED or shot.still.selected is None:
        raise ValueError(f"{shot_id} has no exhausted still with a usable attempt")
    shot.still.status = AssetStatus.PASSED
    shot.still.accepted_by_reviewer = True
    shot.review = Review()


def reject(
    manifest: Manifest, shot_id: str, note: str, target: AssetKind, config: ProjectConfig
) -> None:
    """Send a shot back for regeneration with the reviewer's note as direction.

    Rejecting the still also resets the video, since the video is built from it.
    The shot's type is escalated so similar shots are all spot-checked from now on.
    """
    shot = manifest.shot(shot_id)
    extra = config.retry.max_attempts
    if target is AssetKind.STILL:
        shot.still.reset(extra, note)
        shot.video.reset(extra)
    else:
        shot.video.reset(extra, note)
    # The decision is recorded on the old review; the runner clears it on regeneration.
    shot.review = Review(
        required=True,
        reasons=shot.review.reasons,
        decision=ReviewDecision.REJECTED,
        note=note,
        decided_at=utc_now(),
    )
    if shot.type not in manifest.escalated_types:
        manifest.escalated_types.append(shot.type)
