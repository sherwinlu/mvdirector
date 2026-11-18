# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Streamlit review UI. Launch with ``mvd review --project <folder>``.

Shows the shots the review policy has queued, with the clip, its still, the QC
breakdown and approve/reject controls. All decision logic lives in
``mvdirector.review.policy``; this file is presentation only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import streamlit as st

from mvdirector.config import ReviewMode
from mvdirector.models import AssetKind, AssetStatus, CheckStatus, Shot
from mvdirector.project import Project
from mvdirector.review import policy

_STATUS_ICON = {CheckStatus.PASSED: "✅", CheckStatus.FAILED: "❌", CheckStatus.SKIPPED: "➖"}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--review-mode", choices=[m.value for m in ReviewMode])
    return parser.parse_args()


def _open_project(args: argparse.Namespace) -> Project:
    project = Project.open(args.project)
    if args.review_mode:
        project.config.review.mode = ReviewMode(args.review_mode)
    return project


def _save_decision(project: Project, action) -> None:  # noqa: ANN001
    """Re-read, change and write the manifest so stale browser state is never saved."""
    if project.is_locked:
        st.error("A generation run is in progress. Decisions can be saved once it finishes.")
        return
    with project.edit_manifest() as manifest:
        action(manifest)
    st.rerun()


def _render_shot(project: Project, shot: Shot) -> None:
    reasons = ", ".join(reason.value for reason in shot.review.reasons)
    st.subheader(f"{shot.id} · {shot.type.value} · {shot.framing.value}")
    st.caption(f"{shot.start:.2f}s to {shot.end:.2f}s · queued because: {reasons}")
    if shot.lyrics:
        st.markdown(f"**Lyrics:** {shot.lyrics}")

    video = shot.video.selected or shot.video.best_attempt()
    still = shot.still.selected or shot.still.best_attempt()
    left, right = st.columns(2)
    with left:
        if video and video.path:
            st.video(str(project.path(video.path)))
        else:
            st.warning("No video yet: the still did not pass QC. Accept it or redo it.")
    with right:
        if still and still.path:
            st.image(str(project.path(still.path)), caption="Still (first frame)")

    report = (video.qc if video else None) or (still.qc if still else None)
    if report:
        st.markdown(
            f"**QC score {report.score:.2f}** over {len(shot.video.attempts)} video "
            f"and {len(shot.still.attempts)} still attempt(s)"
        )
        st.table(
            [
                {
                    "check": f"{_STATUS_ICON[c.status]} {c.name}",
                    "score": "" if c.score is None else f"{c.score:.2f}",
                    "detail": c.detail,
                }
                for c in report.checks
            ]
        )

    note = st.text_input(
        "Note for regeneration",
        key=f"note_{shot.id}",
        placeholder="e.g. less head movement, warmer light",
    )
    has_video = bool(video and video.path)
    still_blocked = shot.still.status is AssetStatus.EXHAUSTED
    approve_col, video_col, still_col = st.columns(3)
    if still_blocked:
        if approve_col.button(
            "Use this still",
            key=f"accept_{shot.id}",
            type="primary",
            disabled=not (still and still.path),
        ):
            _save_decision(project, lambda m: policy.accept_still(m, shot.id))
    elif approve_col.button(
        "Approve", key=f"approve_{shot.id}", type="primary", disabled=not has_video
    ):
        _save_decision(project, lambda m: policy.approve(m, shot.id, note or None))
    if video_col.button("Redo video", key=f"video_{shot.id}", disabled=still_blocked):
        _save_decision(
            project, lambda m: policy.reject(m, shot.id, note, AssetKind.VIDEO, project.config)
        )
    if still_col.button("Redo still + video", key=f"still_{shot.id}"):
        _save_decision(
            project, lambda m: policy.reject(m, shot.id, note, AssetKind.STILL, project.config)
        )
    st.divider()


def main() -> None:
    st.set_page_config(page_title="Shot review", layout="wide")
    project = _open_project(_arguments())
    manifest = project.load_manifest()
    policy.refresh_reviews(manifest, project.config)

    queue = policy.pending_review(manifest)
    ready = sum(1 for shot in manifest.shots if policy.is_ready(shot))
    st.title(manifest.title)
    st.caption(
        f"Review mode: {project.config.review.mode.value} · "
        f"{ready}/{len(manifest.shots)} shots ready · {len(queue)} awaiting review"
    )
    if project.is_locked:
        st.info("A generation run is in progress; this page is read-only until it finishes.")
    if not queue:
        st.success("Nothing to review. Run `mvd assemble` when all shots are ready.")
        return
    for shot in queue:
        _render_shot(project, shot)


main()
