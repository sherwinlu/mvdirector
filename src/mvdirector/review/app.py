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

from mvdirector import mix_assemble
from mvdirector.audio.tracklist import format_chapter_label, format_youtube_chapters
from mvdirector.config import ReviewMode
from mvdirector.errors import ConfigError, MVDError, ProjectLocked
from mvdirector.models import (
    AssetKind,
    AssetStatus,
    CheckStatus,
    Manifest,
    MixChapter,
    ProjectMode,
    Shot,
)
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
    try:
        with project.edit_manifest() as manifest:
            action(manifest)
    except ProjectLocked:
        st.error("A generation run is in progress. Decisions can be saved once it finishes.")
        return
    except (MVDError, KeyError, ValueError) as exc:
        st.error(str(exc))
        return
    st.rerun()


def _safe_asset_path(project: Project, rel: str | None) -> Path | None:
    if not rel:
        return None
    try:
        resolved = project.path(rel)
    except ConfigError:
        return None
    return resolved if resolved.is_file() else None


def _render_shot(project: Project, shot: Shot, *, stills_only: bool = False) -> None:
    reasons = ", ".join(reason.value for reason in shot.review.reasons)
    st.subheader(f"{shot.id} · {shot.type.value} · {shot.framing.value}")
    st.caption(f"{shot.start:.2f}s to {shot.end:.2f}s · queued because: {reasons}")
    if shot.lyrics:
        st.markdown(f"**Lyrics:** {shot.lyrics}")

    video = shot.video.selected or shot.video.best_attempt()
    still = shot.still.selected or shot.still.best_attempt()
    video_file = _safe_asset_path(project, video.path if video else None)
    still_file = _safe_asset_path(project, still.path if still else None)
    if stills_only:
        if still_file is not None:
            st.image(str(still_file), caption="Hero still")
        else:
            st.warning("No usable hero still yet.")
    else:
        left, right = st.columns(2)
        with left:
            if video_file is not None:
                st.video(str(video_file))
            else:
                st.warning("No video yet: the still did not pass QC. Accept it or redo it.")
        with right:
            if still_file is not None:
                st.image(str(still_file), caption="Still (first frame)")

    report = (None if stills_only else (video.qc if video else None)) or (
        still.qc if still else None
    )
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
        placeholder="e.g. calmer drift, warmer lighting",
    )
    can_approve_video = shot.video.selected is not None and video_file is not None
    can_approve_still = shot.still.selected is not None and still_file is not None
    still_blocked = (
        shot.still.status is AssetStatus.EXHAUSTED and not shot.still.accepted_by_reviewer
    )
    approve_col, video_col, still_col = st.columns(3)
    if still_blocked:
        if approve_col.button(
            "Use this still",
            key=f"accept_{shot.id}",
            type="primary",
            disabled=not can_approve_still,
        ):
            _save_decision(project, lambda m: policy.accept_still(m, shot.id))
    elif approve_col.button(
        "Approve",
        key=f"approve_{shot.id}",
        type="primary",
        disabled=not (can_approve_still if stills_only else can_approve_video),
    ):
        _save_decision(
            project,
            lambda m: policy.approve(m, shot.id, note or None, stills_only=stills_only),
        )
    if not stills_only and video_col.button(
        "Redo video", key=f"video_{shot.id}", disabled=still_blocked
    ):
        _save_decision(
            project, lambda m: policy.reject(m, shot.id, note, AssetKind.VIDEO, project.config)
        )
    redo_label = "Redo still" if stills_only else "Redo still + video"
    if still_col.button(redo_label, key=f"still_{shot.id}"):
        _save_decision(
            project, lambda m: policy.reject(m, shot.id, note, AssetKind.STILL, project.config)
        )
    st.divider()


def _render_mix_chapter_header(chapter: MixChapter) -> None:
    tags = ", ".join([*chapter.genre_tags, *chapter.mood_tags])
    tag_note = f" · {tags}" if tags else ""
    label = format_chapter_label(chapter.title, chapter.artist)
    st.markdown(
        f"### {chapter.id} · {label} ({chapter.start:.0f}s–{chapter.end:.0f}s · "
        f"{chapter.bpm:.0f} BPM{tag_note})"
    )
    if chapter.environment:
        st.caption(f"Theme: {chapter.environment}")


def _render_mix_queue(
    project: Project, manifest: Manifest, queue: list[Shot], *, stills_only: bool
) -> None:
    queued_by_chapter: dict[str, list[Shot]] = {}
    for shot in queue:
        key = shot.chapter_id or ""
        queued_by_chapter.setdefault(key, []).append(shot)

    known_ids = {ch.id for ch in manifest.chapters}
    orphan_shots = [s for s in queue if (s.chapter_id or "") not in known_ids]

    options = ["All queued chapters"] + [
        f"{ch.id} — {format_chapter_label(ch.title, ch.artist)} "
        f"({len(queued_by_chapter.get(ch.id, []))} queued)"
        for ch in manifest.chapters
        if ch.id in queued_by_chapter
    ]
    if orphan_shots:
        options.append(f"__unassigned__ — Unassigned ({len(orphan_shots)} queued)")
    selected_opt = st.selectbox("Chapter filter", options)
    chosen_id = None if selected_opt == options[0] else selected_opt.split(" — ", 1)[0]

    for ch in manifest.chapters:
        if chosen_id is not None and ch.id != chosen_id:
            continue
        ch_shots = queued_by_chapter.get(ch.id, [])
        if not ch_shots:
            continue
        _render_mix_chapter_header(ch)
        for shot in ch_shots:
            _render_shot(project, shot, stills_only=stills_only)

    if orphan_shots and chosen_id in (None, "__unassigned__"):
        st.markdown("### Unassigned shots")
        for shot in orphan_shots:
            _render_shot(project, shot, stills_only=stills_only)


def main() -> None:
    st.set_page_config(page_title="Shot review", layout="wide")
    project = _open_project(_arguments())
    manifest = project.load_manifest()
    policy.refresh_reviews(manifest, project.config)
    stills_only = policy.uses_stills_only(manifest, project.config)

    queue = policy.pending_review(manifest)
    ready = sum(1 for shot in manifest.shots if policy.is_ready(shot, stills_only=stills_only))
    st.title(manifest.title)
    if manifest.mode is ProjectMode.MIX:
        cached = mix_assemble.cached_segment_count(project, manifest)
        st.caption(
            f"Mode: mix ({project.config.mix.visual_source}) · "
            f"{len(manifest.chapters)} chapters ({cached}/{len(manifest.chapters)} cached) · "
            f"Review mode: {project.config.review.mode.value} · "
            f"{ready}/{len(manifest.shots)} loop units ready · {len(queue)} awaiting review"
        )
        yt_text, _ = format_youtube_chapters(manifest.chapters, manifest.duration)
        if yt_text:
            with st.expander("YouTube Chapters Preview"):
                st.code(yt_text.strip(), language="text")
    else:
        st.caption(
            f"Review mode: {project.config.review.mode.value} · "
            f"{ready}/{len(manifest.shots)} shots ready · {len(queue)} awaiting review"
        )

    if project.is_locked:
        st.info("A generation run is in progress; this page is read-only until it finishes.")
    if not queue:
        st.success("Nothing to review. Run `mvd assemble` when all shots are ready.")
        return
    if manifest.mode is ProjectMode.MIX:
        _render_mix_queue(project, manifest, queue, stills_only=stills_only)
    else:
        for shot in queue:
            _render_shot(project, shot)


if __name__ == "__main__":
    main()
