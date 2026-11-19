# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""The generate -> quality-check -> retry loop for stills and videos.

For each shot and asset kind the runner keeps generating until an attempt passes QC
or the attempt allowance is used up, saving the manifest after every attempt so a
run can be interrupted and resumed at any point.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mvdirector import media
from mvdirector.comfy.client import ComfyClient
from mvdirector.comfy.workflow import WorkflowTemplate
from mvdirector.errors import BudgetExceeded, ComfyJobError, ConfigError, MediaToolError
from mvdirector.models import (
    AssetKind,
    AssetState,
    AssetStatus,
    Attempt,
    Manifest,
    QCReport,
    ReferenceSet,
    Review,
    Shot,
    ShotType,
)
from mvdirector.planning import prompts
from mvdirector.project import Project
from mvdirector.qc.faces import IdentityReference
from mvdirector.qc.factory import QualityGates
from mvdirector.retry import AttemptPlan, RetryPolicy
from mvdirector.store import atomic_write

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunEvent:
    """One structured progress event from a generation run.

    ``kind`` is one of ``warning`` (preflight problems, in ``message``), ``qc_summary`` (the
    active and skipped checks, before any GPU time is spent), ``phase_started``,
    ``attempt_started``, ``attempt_finished``, ``asset_passed``, ``asset_exhausted`` and
    ``run_finished``. Anything building a UI or persisting progress subscribes to these instead
    of parsing log lines.
    """

    kind: str
    asset: str = ""  # "still" or "video"
    shot_id: str = ""
    attempt: int = 0
    strategy: str = ""
    passed: bool | None = None
    score: float | None = None
    issues: list[str] = field(default_factory=list)
    error: str | None = None
    generations_used: int = 0
    message: str = ""  # human-readable text for ``warning`` events
    active_checks: list[str] = field(default_factory=list)  # ``qc_summary`` only
    skipped_checks: list[tuple[str, str]] = field(default_factory=list)  # (names, reason)


RunObserver = Callable[[RunEvent], None]


class RunBudget:
    """Caps generations per run so a stubborn shot cannot run up the GPU bill."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.used = 0

    def consume(self) -> None:
        if self.used >= self.limit:
            raise BudgetExceeded(
                f"Reached the limit of {self.limit} generations for this run "
                "(retry.max_generations_per_run). Progress is saved; run again to continue."
            )
        self.used += 1


def load_templates(project: Project, names: Iterable[str]) -> dict[str, WorkflowTemplate]:
    comfy = project.config.comfy
    templates = {}
    for name in names:
        if name not in comfy.workflows:
            raise ConfigError(f"comfy.workflows has no entry for '{name}'")
        bindings_path = project.path(Path(comfy.workflows_dir) / comfy.workflows[name])
        templates[name] = WorkflowTemplate.load(name, bindings_path, root=project.root)
    return templates


def _read_stamp(marker: Path) -> dict[str, Any] | None:
    """What an audio slice was cut from, or None when it is missing or unreadable."""
    try:
        stamp = json.loads(marker.read_text("utf-8"))
    except (OSError, ValueError):
        return None
    return stamp if isinstance(stamp, dict) else None


class ShotRunner:
    def __init__(
        self,
        project: Project,
        manifest: Manifest,
        comfy: ComfyClient,
        gates: QualityGates,
        policy: RetryPolicy,
        templates: dict[str, WorkflowTemplate],
        budget: RunBudget,
        observer: RunObserver | None = None,
        draft: bool = False,
    ) -> None:
        self._project = project
        self._manifest = manifest
        self._comfy = comfy
        self._gates = gates
        self._policy = policy
        self._templates = templates
        self._budget = budget
        self._observer = observer
        self._draft = draft
        self._uploads: dict[Path, str] = {}
        self._identities: dict[tuple[str, ...], IdentityReference] = {}
        self._warned: set[str] = set()

    # ---------------------------------------------------------------- public

    def run(self, kinds: list[AssetKind], shot_ids: list[str] | None = None) -> None:
        """Process the given asset kinds for the selected shots (all shots by default)."""
        shots = [s for s in self._manifest.shots if shot_ids is None or s.id in shot_ids]
        try:
            for index, kind in enumerate(kinds):  # stills first: failures are cheapest there
                if index > 0:
                    self._comfy.free_memory()
                self._emit(RunEvent("phase_started", asset=kind.value))
                for shot in shots:
                    self._run_asset(shot, kind)
        finally:
            self._project.save_manifest(self._manifest)
            self._emit(RunEvent("run_finished", generations_used=self._budget.used))

    def _emit(self, event: RunEvent) -> None:
        if self._observer is not None:
            self._observer(event)

    # --------------------------------------------------------------- private

    def _reference_set(self, shot: Shot) -> ReferenceSet | None:
        """The reference sheet for the shot's character and wardrobe state, else the base sheet."""
        character = self._manifest.shot_character(shot)
        reference = character.reference_for(shot.wardrobe)
        warn_key = f"{shot.character or ''}:{shot.wardrobe or ''}"
        if (
            reference is not None
            and shot.wardrobe
            and shot.wardrobe not in character.references
            and warn_key not in self._warned
        ):
            self._warned.add(warn_key)
            log.warning(
                "No reference sheet for wardrobe state '%s'; using the base sheet instead.",
                shot.wardrobe,
            )
        return reference

    def _identity_for(self, shot: Shot) -> IdentityReference | None:
        """Identity embedding built from the shot's reference sheet, cached per set of images."""
        reference = self._reference_set(shot)
        if reference is None or self._gates.face_analyzer is None:
            return None
        images = tuple(reference.identity_images())
        if not images:
            return None
        if images not in self._identities:
            self._identities[images] = IdentityReference(
                self._gates.face_analyzer, [self._project.path(path) for path in images]
            )
        return self._identities[images]

    def _plate_for(self, shot: Shot) -> Path | None:
        """The shot's set plate, when the still workflow can take one and one is locked."""
        template = self._templates.get("still")
        if template is None or not template.accepts("set_image") or not shot.type.shows_singer:
            return None
        scene = self._manifest.visual_bible.find_set(shot.setting)
        if scene is None or not scene.plate:
            return None
        path = self._project.path(scene.plate)
        return path if path.is_file() else None

    def _prop_paths(self, shot: Shot) -> list[tuple[str, Path]]:
        """Locked prop reference images for ``shot.props`` that exist on disk."""
        resolved: list[tuple[str, Path]] = []
        for prop in self._manifest.visual_bible.resolve_props(shot.props):
            if not prop.plate:
                continue
            path = self._project.path(prop.plate)
            if path.is_file():
                resolved.append((prop.name, path))
        return resolved

    def _attach_prop_refs(
        self, params: dict[str, Any], template: WorkflowTemplate, prop_refs: list[tuple[str, Path]]
    ) -> None:
        if prop_refs and template.accepts("prop_image"):
            params["prop_image"] = self._upload(prop_refs[0][1])
            if len(prop_refs) > 1 and template.accepts("prop_image_2"):
                params["prop_image_2"] = self._upload(prop_refs[1][1])

    def _previz_for(self, shot: Shot, template: WorkflowTemplate) -> Path | None:
        if not shot.previz_video or not template.accepts("previz_video"):
            return None
        path = self._project.path(shot.previz_video)
        return path if path.is_file() else None

    def _run_asset(self, shot: Shot, kind: AssetKind) -> None:
        state = shot.asset(kind)
        if kind is AssetKind.VIDEO and shot.still.status is not AssetStatus.PASSED:
            # Never spend video time on a still that has not passed QC or been accepted.
            log.info("%s: still is %s; skipping video", shot.id, shot.still.status.value)
            return
        while state.status is AssetStatus.PENDING:
            if len(state.attempts) >= state.attempts_allowed:
                self._mark_exhausted(shot, kind, state)
                if kind is AssetKind.VIDEO and self._fall_back_to_performance(shot, state):
                    continue
                break
            self._budget.consume()
            plan = self._policy.prepare(shot, kind)
            shot.review = Review()  # any earlier review no longer describes this shot
            self._emit(
                RunEvent(
                    "attempt_started",
                    asset=kind.value,
                    shot_id=shot.id,
                    attempt=plan.number,
                    strategy=plan.strategy.value,
                    generations_used=self._budget.used,
                )
            )
            attempt = self._attempt(shot, kind, plan)
            state.attempts.append(attempt)
            self._manifest.total_generations += 1
            self._emit(
                RunEvent(
                    "attempt_finished",
                    asset=kind.value,
                    shot_id=shot.id,
                    attempt=attempt.number,
                    strategy=attempt.strategy.value,
                    passed=attempt.qc.passed if attempt.qc else None,
                    score=attempt.qc.score if attempt.qc else None,
                    issues=attempt.qc.issues if attempt.qc else [],
                    error=attempt.error,
                    generations_used=self._budget.used,
                )
            )
            if attempt.qc is not None and attempt.qc.passed:
                state.status = AssetStatus.PASSED
                state.selected_attempt = attempt.number
                self._emit(
                    RunEvent(
                        "asset_passed", asset=kind.value, shot_id=shot.id, attempt=attempt.number
                    )
                )
            self._project.save_manifest(self._manifest)
            if (
                attempt.error is not None
                and len(state.attempts) > 1
                and state.attempts[-2].error == attempt.error
            ):
                # A different seed or prompt cannot fix a missing model or a dead network,
                # so stop here and leave the shot pending for the next run.
                log.error(
                    "%s %s: the same error twice in a row, so more attempts would not help. "
                    "Fix it and run again: %s",
                    shot.id,
                    kind.value,
                    attempt.error,
                )
                break

    def _attempt(self, shot: Shot, kind: AssetKind, plan: AttemptPlan) -> Attempt:
        state = shot.asset(kind)
        plate = self._plate_for(shot) if kind is AssetKind.STILL else None
        notation = self._project.config.comfy.reference_notation
        still_refs: list[tuple[str, Path]] = []
        prop_refs = self._prop_paths(shot)
        if kind is AssetKind.STILL:
            still_refs = self._still_reference_paths(shot, plate)
            still_tpl = self._templates["still" if shot.type.shows_singer else "t2i"]
            prompt = prompts.compose_still_prompt(
                self._manifest,
                shot,
                plate_attached=plate is not None,
                props_attached=bool(prop_refs and still_tpl.accepts("prop_image")),
                reference_notation=notation,
                bound_references=[name for name, _ in still_refs] if still_refs else None,
            )
        else:
            video_template = self._video_template(shot)
            previz = self._previz_for(shot, video_template)
            prompt = prompts.compose_motion_prompt(
                self._manifest,
                shot,
                video_config=self._project.config.video,
                reference_notation=notation,
                has_audio_video=self._video_uses_audio_video(shot, video_template),
                previz_attached=previz is not None,
            )
        attempt = Attempt(
            number=plan.number,
            strategy=plan.strategy,
            seed=plan.seed,
            prompt=prompt,
            params=dict(state.param_overrides),
        )
        destination = self._project.asset_path(shot, kind, plan.number)
        log.info("%s %s attempt %d (%s)", shot.id, kind.value, plan.number, plan.strategy.value)
        try:
            if kind is AssetKind.STILL:
                self._generate_still(shot, attempt, destination, still_refs, prop_refs)
            else:
                self._generate_video(shot, attempt, destination, prop_refs)
            attempt.path = self._project.rel(destination)
            attempt.qc = self._check(shot, kind, destination)
        except (ComfyJobError, MediaToolError) as exc:
            # The job itself failed. Record it and let the loop try again.
            log.warning("%s %s attempt %d errored: %s", shot.id, kind.value, plan.number, exc)
            attempt.error = str(exc)
            return attempt
        outcome = "passed" if attempt.qc.passed else "failed: " + "; ".join(attempt.qc.issues)
        log.info(
            "%s %s attempt %d QC %.2f %s",
            shot.id,
            kind.value,
            plan.number,
            attempt.qc.score,
            outcome,
        )
        return attempt

    def _still_reference_paths(self, shot: Shot, plate: Path | None) -> list[tuple[str, Path]]:
        """Ordered (binding_name, path) pairs uploaded to the ``still`` workflow."""
        if not shot.type.shows_singer:
            return []
        template = self._templates["still"]
        refs: list[tuple[str, Path]] = []
        reference = self._reference_set(shot)
        front = reference.panel("front") if reference else None
        if template.accepts("reference_image"):
            if reference is None or front is None:
                raise ConfigError(
                    "No reference sheet is locked. Run `mvd reference generate` and "
                    "`mvd reference lock <image>` before generating stills."
                )
            refs.append(("reference_image", self._project.path(front)))
        # Extra angles go only to workflows that bind them, so nothing is uploaded for nothing.
        if reference is not None:
            for name in ("profile", "three_quarter", "face"):
                path = reference.panel(name)
                param = f"reference_{name}"
                if path and template.accepts(param):
                    refs.append((param, self._project.path(path)))
        visible = self._manifest.visible_characters(shot)
        if len(visible) > 1 and template.accepts("reference_image_2"):
            _, sec_char, sec_state = visible[1]
            sec_ref = sec_char.reference_for(sec_state)
            sec_front = sec_ref.panel("front") if sec_ref else None
            if sec_front is not None:
                refs.append(("reference_image_2", self._project.path(sec_front)))
        if plate is not None:
            refs.append(("set_image", plate))
        return refs

    def _generate_still(
        self,
        shot: Shot,
        attempt: Attempt,
        destination: Path,
        still_refs: list[tuple[str, Path]],
        prop_refs: list[tuple[str, Path]],
    ) -> None:
        video = self._project.config.video
        width, height = video.generation_size(draft=self._draft)
        params: dict[str, Any] = {
            "positive_prompt": attempt.prompt,
            "negative_prompt": self._manifest.visual_bible.negative_prompt,
            "seed": attempt.seed,
            "width": width,
            "height": height,
            **attempt.params,
        }
        if shot.type.shows_singer:
            template = self._templates["still"]
            for param, ref_path in still_refs:
                params[param] = self._upload(ref_path)
        else:
            template = self._templates["t2i"]
        self._attach_prop_refs(params, template, prop_refs)
        self._comfy.run(template.render(params), template.bindings.output, destination)

    def _video_template(self, shot: Shot) -> WorkflowTemplate:
        return (
            self._templates["lipsync"] if shot.type is ShotType.LIPSYNC else self._templates["i2v"]
        )

    @staticmethod
    def _video_uses_audio_video(shot: Shot, template: WorkflowTemplate) -> bool:
        return (
            shot.type is ShotType.LIPSYNC and template.accepts("audio_video")
        ) or "audio_video" in template.bindings.required

    def _generate_video(
        self,
        shot: Shot,
        attempt: Attempt,
        destination: Path,
        prop_refs: list[tuple[str, Path]],
    ) -> None:
        video = self._project.config.video
        width, height = video.generation_size(draft=self._draft)
        still = shot.still.selected
        assert still is not None and still.path is not None  # guarded in _run_asset
        clip_duration = round(shot.audio_slice.end - shot.audio_slice.start, 3)
        template = self._video_template(shot)
        params: dict[str, Any] = {
            "positive_prompt": attempt.prompt,
            "negative_prompt": self._manifest.visual_bible.negative_prompt,
            "lyrics": shot.lyrics,
            "seed": attempt.seed,
            "image": self._upload(self._project.path(still.path)),
            "width": width,
            "height": height,
            "num_frames": shot.num_frames,
            "duration": clip_duration,
            "fps": video.fps,
            **attempt.params,
        }
        if shot.type is ShotType.LIPSYNC or "audio" in template.bindings.required:
            params["audio"] = self._upload(self._ensure_audio_slice(shot))
        if self._video_uses_audio_video(shot, template):
            params["audio_video"] = self._upload(self._ensure_audio_video_slice(shot))
        if shot.type.shows_singer and template.accepts("reference_video"):
            ref = self._reference_set(shot)
            if ref is not None and ref.video:
                ref_video_path = self._project.path(ref.video)
                if ref_video_path.is_file():
                    params["reference_video"] = self._upload(ref_video_path)
        self._attach_prop_refs(params, template, prop_refs)
        previz = self._previz_for(shot, template)
        if previz is not None:
            params["previz_video"] = self._upload(previz)
        self._comfy.run(template.render(params), template.bindings.output, destination)

    def _extra_reference_images(self, shot: Shot) -> list[tuple[str, Path]]:
        """Reference sheets for secondary visible performers and props in this shot."""
        extras: list[tuple[str, Path]] = []
        if shot.type.shows_singer:
            for sec_id, sec_char, sec_state in self._manifest.visible_characters(shot)[1:]:
                sec_ref = sec_char.reference_for(sec_state)
                sec_sheet = (sec_ref.sheet or sec_ref.panel("front")) if sec_ref else None
                if sec_sheet:
                    label = sec_char.name or sec_id or "performer"
                    extras.append((label, self._project.path(sec_sheet)))
        for prop_name, prop_path in self._prop_paths(shot):
            extras.append((f"Prop: {prop_name}", prop_path))
        return extras

    def _check(self, shot: Shot, kind: AssetKind, path: Path) -> QCReport:
        # The vision judge sees the whole reference sheet (every angle). It gets the image as a
        # path, so it works even when face analysis is unavailable.
        reference = self._reference_set(shot) if shot.type.shows_singer else None
        sheet = reference.sheet or reference.panel("front") if reference else None
        reference_image = self._project.path(sheet) if sheet else None
        extra_references = self._extra_reference_images(shot)
        identity = self._identity_for(shot)
        expected_size = (
            self._project.config.video.generation_size(draft=True) if self._draft else None
        )
        if kind is AssetKind.STILL:
            return self._gates.image.evaluate(
                path,
                shot,
                self._manifest,
                identity,
                reference_image=reference_image,
                extra_references=extra_references or None,
                expected_size=expected_size,
            )
        return self._gates.video.evaluate(
            path,
            shot,
            self._manifest,
            self._ensure_audio_slice(shot),
            identity,
            reference_image=reference_image,
            extra_references=extra_references or None,
            expected_size=expected_size,
        )

    def _fall_back_to_performance(self, shot: Shot, state: AssetState) -> bool:
        """Turn a singing shot that could not be made into one where the singer is not singing.

        The still stays; the video is rendered again as a performance shot, which needs no
        lip-sync and so does not depend on the face being tracked. Earlier attempts stay in
        the history.
        """
        if not self._project.config.retry.lipsync_fallback or shot.type is not ShotType.LIPSYNC:
            return False
        shot.fallback_from, shot.type = shot.type, ShotType.PERFORMANCE
        shot.motion_prompt = prompts.PERFORMANCE_MOTION
        state.reset(self._project.config.retry.max_attempts)
        state.prompt_override = None
        state.param_overrides = {}
        shot.review = Review()
        log.warning(
            "%s: no usable lip-sync video, so it is now a performance shot (the singer on "
            "camera, not singing)",
            shot.id,
        )
        self._emit(RunEvent("shot_converted", shot_id=shot.id, asset=AssetKind.VIDEO.value))
        self._project.save_manifest(self._manifest)
        return True

    def _mark_exhausted(self, shot: Shot, kind: AssetKind, state: AssetState) -> None:
        best = state.best_attempt()
        state.status = AssetStatus.EXHAUSTED
        state.selected_attempt = best.number if best else None
        self._emit(
            RunEvent(
                "asset_exhausted",
                asset=kind.value,
                shot_id=shot.id,
                attempt=best.number if best else 0,
                generations_used=self._budget.used,
            )
        )
        log.warning(
            "%s %s exhausted after %d attempts; %s",
            shot.id,
            kind.value,
            len(state.attempts),
            f"keeping attempt {best.number} for review" if best else "nothing usable was produced",
        )
        self._project.save_manifest(self._manifest)

    def _ensure_audio_slice(self, shot: Shot) -> Path:
        """The shot's audio slice, cut again whenever its timing or its source audio changed.

        A sidecar next to the slice records what it was cut from. A slice left over from an
        earlier plan (other start and end), or from another mix, is not reused: it would feed
        the lip-sync workflow and the lip-sync check audio from the wrong part of the song.
        """
        path = self._project.path(shot.audio_slice.path)
        inputs, source = (
            self._project.config.inputs,
            self._project.config.video.lipsync_audio_source,
        )
        try:
            source_path = self._project.input_path(
                inputs.vocals if source == "vocals" else inputs.mix, source
            )
        except ConfigError:
            if path.is_file():
                return path  # the source audio is gone; keep the slice that was already cut
            raise
        source_stat = source_path.stat()
        stamp = {
            "source": source_path.name,
            "source_size": source_stat.st_size,
            "source_mtime_ns": source_stat.st_mtime_ns,
            "start": round(shot.audio_slice.start, 4),
            "end": round(shot.audio_slice.end, 4),
        }
        marker = path.with_name(f"{path.name}.json")
        if not path.is_file() or _read_stamp(marker) != stamp:
            media.slice_audio(source_path, path, shot.audio_slice.start, shot.audio_slice.end)
            atomic_write(marker, json.dumps(stamp))
        return path

    def _ensure_audio_video_slice(self, shot: Shot) -> Path:
        wav_path = self._ensure_audio_slice(shot)
        mp4_path = self._project.audio_video_slice_path(shot.id)
        if not mp4_path.is_file() or mp4_path.stat().st_mtime < wav_path.stat().st_mtime:
            media.audio_to_black_video(wav_path, mp4_path, fps=self._project.config.video.fps)
        return mp4_path

    def _upload(self, path: Path) -> str:
        if path not in self._uploads:
            self._uploads[path] = self._comfy.upload(path)
        return self._uploads[path]
