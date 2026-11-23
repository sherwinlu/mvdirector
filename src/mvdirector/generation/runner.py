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

from mvdirector import assemble, media
from mvdirector.comfy.cache import ComfyRenderCache, canonical_graph_key, file_sha256
from mvdirector.comfy.client import ComfyClient
from mvdirector.comfy.workflow import OutputSpec, WorkflowTemplate
from mvdirector.errors import BudgetExceeded, ComfyJobError, ConfigError, MediaToolError
from mvdirector.generation.candidates import ensure_upload_audio_track
from mvdirector.models import (
    AssetKind,
    AssetState,
    AssetStatus,
    Attempt,
    ContinuityMode,
    Manifest,
    QCReport,
    ReferenceSet,
    RetryStrategy,
    Review,
    Shot,
    ShotType,
)
from mvdirector.planning import prompts
from mvdirector.project import Project, require_safe_id
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
    ``attempt_started``, ``attempt_finished``, ``asset_passed``, ``shot_converted``,
    ``asset_exhausted`` and ``run_finished``. Anything building a UI or persisting progress
    subscribes to these instead of parsing log lines.
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
        templates[name] = WorkflowTemplate.load(
            name, bindings_path, root=project.root, environ=project.environ()
        )
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
        self._uploads: dict[tuple[Path, int, int], str] = {}
        self._uploaded_local: dict[str, Path] = {}
        self._uploaded_digests: dict[str, str] = {}
        self._file_hashes: dict[tuple[Path, int, int], str] = {}
        self._identities: dict[tuple[str, ...], IdentityReference] = {}
        self._warned: set[str] = set()
        self._analysis_beats: list[float] | None = None
        self._cache: ComfyRenderCache | None = (
            ComfyRenderCache(project.path(".comfy_cache"))
            if project.config.comfy.cache_enabled
            else None
        )

    # ---------------------------------------------------------------- public

    def run(self, kinds: list[AssetKind], shot_ids: list[str] | None = None) -> None:
        """Process the given asset kinds for the selected shots (all shots by default)."""
        shots = [s for s in self._manifest.shots if shot_ids is None or s.id in shot_ids]
        has_chained = any(s.continuity is not ContinuityMode.CUT for s in shots)
        # Chained shots run after the shot they continue from, whichever kinds this run covers:
        # a video-only run must render the previous clip before taking a deferred handoff still.
        waves = self._chain_waves(shots) if has_chained else [shots]
        previous_kind: AssetKind | None = None
        try:
            for wave_idx, wave_shots in enumerate(waves):
                for kind in kinds:  # stills first: failures are cheapest there
                    if previous_kind is not None and kind is not previous_kind:
                        self._comfy.free_memory()  # unload the previous phase's model
                    previous_kind = kind
                    self._emit(RunEvent("phase_started", asset=kind.value))
                    ordered = (
                        self._sort_for_cache(wave_shots, kind)
                        if (self._project.config.comfy.sort_batch_for_cache and wave_idx == 0)
                        else wave_shots
                    )
                    for shot in ordered:
                        self._run_asset(shot, kind)
        finally:
            self._project.save_manifest(self._manifest)
            self._emit(RunEvent("run_finished", generations_used=self._budget.used))

    def _chain_waves(self, shots: list[Shot]) -> list[list[Shot]]:
        """Partition shots into dependency waves so CUT shots batch while chains run in order."""
        selected_ids = {s.id for s in shots}
        all_shots = self._manifest.shots
        pos_by_id = {s.id: i for i, s in enumerate(all_shots)}
        depth_by_id: dict[str, int] = {}
        waves: dict[int, list[Shot]] = {}
        for shot in shots:
            pos = pos_by_id.get(shot.id)
            if (
                shot.continuity is not ContinuityMode.CUT
                and pos is not None
                and pos > 0
                and all_shots[pos - 1].id in selected_ids
            ):
                depth = depth_by_id.get(all_shots[pos - 1].id, 0) + 1
            else:
                depth = 0
            depth_by_id[shot.id] = depth
            waves.setdefault(depth, []).append(shot)
        return [waves[d] for d in sorted(waves)]

    @staticmethod
    def _sort_for_cache(shots: list[Shot], kind: AssetKind) -> list[Shot]:
        """Group independent shots by workflow/setting/prompt prefix to keep ComfyUI caches warm.

        Inspired by ``tysonfarrington-cmyk/xenoform`` (``final_tier.py``) and
        ``tsondo/musicvision`` (``assets/consistency.py``).
        """
        if kind is AssetKind.STILL:
            return sorted(
                shots,
                key=lambda s: (
                    0 if s.type.shows_singer else 1,
                    s.character or "",
                    s.wardrobe or "",
                    s.setting,
                    s.still_prompt[:48],
                    s.index,
                ),
            )
        return sorted(
            shots,
            key=lambda s: (
                0 if s.type is ShotType.LIPSYNC else 1,
                s.setting,
                s.motion_prompt[:48],
                s.index,
            ),
        )

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

    def _previz_depth_for(self, shot: Shot, template: WorkflowTemplate) -> Path | None:
        if not shot.previz_depth_video or not template.accepts("previz_depth_video"):
            return None
        path = self._project.path(shot.previz_depth_video)
        return path if path.is_file() else None

    def _previous_shot(self, shot: Shot) -> Shot | None:
        shots = self._manifest.shots
        pos = next((i for i, item in enumerate(shots) if item.id == shot.id), None)
        return shots[pos - 1] if pos is not None and pos > 0 else None

    def _played_tail(self, prev: Shot) -> tuple[Path, int, Attempt] | None:
        """``(clip, index of its last frame played at assembly, attempt)`` for ``prev``.

        Uses assembly's own timing (lip-sync delay, a dropped handoff seam frame), so chained
        shots continue from exactly the frame viewers see last.

        Credits & Attribution:
            Trim-aware cross-shot handoff frame chaining inspired by
            ``GeekatplayStudio/ComfyUI-Music-to-Video`` (``nodes.py``) and
            ``afj176/open-cinema-studio`` (``app/studio.py``).
        """
        prev_video = prev.video.selected or prev.video.best_attempt()
        if prev_video is None or not prev_video.path:
            return None
        video_path = self._project.path(prev_video.path)
        if not video_path.is_file():
            return None
        timing = assemble.clip_timing(
            prev,
            prev_video,
            prev.still.selected or prev.still.best_attempt(),
            sync_delay_compensation=self._project.config.video.sync_delay_compensation,
        )
        fps = self._manifest.fps
        frames = assemble.frame_counts([prev], fps)[0]
        return video_path, timing.last_played_frame(frames, fps), prev_video

    def _previous_handoff_frame(self, shot: Shot) -> Path | None:
        """The previous clip's last played frame, used as a ``reframe`` shot's reference."""
        prev = self._previous_shot(shot)
        tail = self._played_tail(prev) if prev is not None else None
        if prev is None or tail is None:
            return None
        video_path, frame_index, _ = tail
        safe_prev_id = require_safe_id(prev.id, "shot id")
        handoff_dst = self._project.path(f"assets/stills/{safe_prev_id}_handoff.png")
        return media.extract_frame_at_index(video_path, frame_index, handoff_dst)

    def _take_handoff_still(self, shot: Shot, state: AssetState) -> bool:
        """Make a ``continue`` shot's still the previous clip's last played frame.

        Returns True when the still is settled here: handed off, or deferred until the previous
        shot has a clip to continue from. Returns False when the still has to be generated
        normally instead, which makes this join a plain cut.
        """
        prev = self._previous_shot(shot)
        if prev is None:
            return False
        if state.director_note:
            log.warning(
                "%s: a reviewer asked for changes to this continuation still; it gets a newly "
                "generated still, so the join with %s renders as a cut.",
                shot.id,
                prev.id,
            )
            return False
        if prev.video.status is AssetStatus.PENDING:
            # Not final yet: even an existing take may still be retried or replaced.
            log.warning(
                "%s: continuity 'continue' waits for %s's clip; this still will be that clip's "
                "last frame once its video is finished (`mvd videos` or `mvd run`).",
                shot.id,
                prev.id,
            )
            return True
        tail = self._played_tail(prev)
        if tail is None:
            log.warning(
                "%s: %s has no usable clip to continue from; this shot gets a newly generated "
                "still, so the join renders as a cut.",
                shot.id,
                prev.id,
            )
            return False
        video_path, frame_index, prev_video = tail
        if any(
            a.strategy is RetryStrategy.HANDOFF
            and a.params.get("handoff_source") == prev_video.path
            for a in state.attempts
        ):
            # Handing off the same frame again would only repeat the attempt that was sent back.
            log.warning(
                "%s: the handoff frame from %s's current clip was already used and sent back; "
                "this shot gets a newly generated still, so the join renders as a cut. "
                "Re-render %s's video to continue from a new frame.",
                shot.id,
                prev.id,
                prev.id,
            )
            return False
        number = len(state.attempts) + 1
        destination = self._project.asset_path(shot, AssetKind.STILL, number)
        media.extract_frame_at_index(video_path, frame_index, destination)
        # The clip starts directly from the handoff frame: align the audio slice to shot.start
        # (zero head handle) so frame 0 is the handoff boundary and stays locked to the audio.
        shot.audio_slice.start = round(shot.start, 3)
        shot.audio_slice.head_handle = 0.0
        attempt = Attempt(
            number=number,
            strategy=RetryStrategy.HANDOFF,
            seed=0,
            prompt="(continued from previous shot handoff frame)",
            params={"handoff_source": prev_video.path, "handoff_frame": frame_index},
            path=self._project.rel(destination),
            qc=QCReport(passed=True, score=1.0),
        )
        state.attempts.append(attempt)
        state.status = AssetStatus.PASSED
        state.selected_attempt = number
        self._project.save_manifest(self._manifest)
        self._emit(
            RunEvent("asset_passed", asset=AssetKind.STILL.value, shot_id=shot.id, attempt=number)
        )
        return True

    def _run_asset(self, shot: Shot, kind: AssetKind) -> None:
        state = shot.asset(kind)
        if shot.continuity is ContinuityMode.CONTINUE and shot.still.status is AssetStatus.PENDING:
            # A one-take continuation's still is the previous clip's last played frame. The video
            # phase takes it as well, for stills an earlier still-only run had to defer.
            settled = self._take_handoff_still(shot, shot.still)
            if kind is AssetKind.STILL and settled:
                return
        if kind is AssetKind.VIDEO and shot.still.status is not AssetStatus.PASSED:
            # Never spend video time on a still that has not passed QC or been accepted.
            log.info("%s: still is %s; skipping video", shot.id, shot.still.status.value)
            return
        while state.status is AssetStatus.PENDING:
            if len(state.attempts) >= state.attempts_allowed:
                if kind is AssetKind.VIDEO and self._fall_back_from_lipsync(shot, state):
                    continue
                self._mark_exhausted(shot, kind, state)
                break
            self._budget.consume()
            still_tpl_key = "still" if shot.type.shows_singer else "t2i"
            can_repair = (
                kind is AssetKind.STILL
                and still_tpl_key in self._templates
                and self._templates[still_tpl_key].accepts("reference_image")
            )
            plan = self._policy.prepare(shot, kind, can_repair=can_repair)
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
            prev_attempt = state.attempts[-1] if state.attempts else None
            attempt = self._attempt(shot, kind, plan, previous_attempt=prev_attempt)
            # Anti-regression guard for targeted in-place still repair
            # (inspired by afj176/open-cinema-studio studio.py:_repair_frame).
            if (
                plan.strategy is RetryStrategy.REPAIR_STILL
                and prev_attempt is not None
                and prev_attempt.qc is not None
                and attempt.qc is not None
                and not attempt.qc.passed
            ):
                old_failed = {c.name for c in prev_attempt.qc.failed_checks}
                new_failed = {c.name for c in attempt.qc.failed_checks}
                improved = new_failed <= old_failed and len(attempt.qc.issues) < len(
                    prev_attempt.qc.issues
                )
                if not improved:
                    attempt.regressed = True
                    log.info(
                        "%s still repair did not strictly reduce issues (%s -> %s); "
                        "excluding regressed repair from best_attempt selection",
                        shot.id,
                        len(prev_attempt.qc.issues),
                        len(attempt.qc.issues),
                    )
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
                # Smart Cutaway Rescue: if lip-sync correlation is critically low on an advisory
                # pass and smart_cutaway_rescue is enabled, convert to a b-roll cutaway instead
                # of keeping a poor mouth-sync clip (inspired by VisualFrisson/ComfyUI-OneShotMV).
                if (
                    kind is AssetKind.VIDEO
                    and self._should_rescue_weak_lipsync(shot, attempt)
                    and self._convert_lipsync_to_broll(shot, state)
                ):
                    continue
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

    def _attempt(
        self,
        shot: Shot,
        kind: AssetKind,
        plan: AttemptPlan,
        *,
        previous_attempt: Attempt | None = None,
    ) -> Attempt:
        state = shot.asset(kind)
        plate = self._plate_for(shot) if kind is AssetKind.STILL else None
        notation = self._project.config.comfy.reference_notation
        max_chars = self._project.config.comfy.max_prompt_chars
        still_refs: list[tuple[str, Path]] = []
        prop_refs = self._prop_paths(shot)
        repair_source: Path | None = None
        if kind is AssetKind.STILL:
            still_refs = self._still_reference_paths(shot, plate)
            still_tpl = self._templates["still" if shot.type.shows_singer else "t2i"]
            if (
                plan.strategy is RetryStrategy.REPAIR_STILL
                and previous_attempt is not None
                and previous_attempt.path
                and still_tpl.accepts("reference_image")
            ):
                candidate = self._project.path(previous_attempt.path)
                if candidate.is_file():
                    repair_source = candidate
                    issues = (
                        "; ".join(previous_attempt.qc.issues) if previous_attempt.qc else "defects"
                    )
                    hints = " ".join(previous_attempt.qc.fix_hints) if previous_attempt.qc else ""
                    prompt = (
                        "Keep the exact composition, framing, lighting, pose, and character "
                        f"identity from the input image. Fix ONLY these defects: {issues}. "
                        f"{hints}"
                    ).strip()
                else:
                    prompt = prompts.compose_still_prompt(
                        self._manifest,
                        shot,
                        plate_attached=plate is not None,
                        props_attached=bool(prop_refs and still_tpl.accepts("prop_image")),
                        reference_notation=notation,
                        bound_references=[name for name, _ in still_refs] if still_refs else None,
                    )
            else:
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
        if max_chars is not None:
            prompt = prompts.fit_prompt_budget(prompt, max_chars)
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
                self._generate_still(
                    shot, attempt, destination, still_refs, prop_refs, repair_source=repair_source
                )
            else:
                self._generate_video(shot, attempt, destination, prop_refs)
            attempt.path = self._project.rel(destination)
            attempt.qc = self._check(shot, kind, destination)
            if kind is AssetKind.VIDEO and attempt.qc is not None:
                attempt.sync_delay_seconds = self._extract_sync_delay(shot, attempt.qc)
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

    @staticmethod
    def _extract_sync_delay(shot: Shot, report: QCReport) -> float:
        """Clamp measured lip-sync playback offset safely within the shot's audio handles."""
        sync_check = next((c for c in report.checks if c.name == "lipsync"), None)
        if sync_check is None:
            return 0.0
        raw_delay = sync_check.metrics.get("sync_delay_seconds")
        if raw_delay is None:
            return 0.0
        head = max(0.0, shot.audio_slice.head_handle)
        tail = max(0.0, shot.audio_slice.end - shot.end)
        return round(max(-head, min(tail, float(raw_delay))), 4)

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

    def _still_negative_prompt(self, shot: Shot) -> str:
        parts = [self._manifest.visual_bible.negative_prompt]
        raw_still = shot.still.prompt_override or shot.still_prompt
        parts.extend(prompts.extract_negative_exclusions(raw_still))
        if shot.type is ShotType.BROLL:
            parts.append("people, human figures, faces")
        else:
            parts.append("vertically squashed body, stubby legs, oversized head")
        return ", ".join(p.strip(", ") for p in parts if p and p.strip(", "))

    def _generate_still(
        self,
        shot: Shot,
        attempt: Attempt,
        destination: Path,
        still_refs: list[tuple[str, Path]],
        prop_refs: list[tuple[str, Path]],
        *,
        repair_source: Path | None = None,
    ) -> None:
        video = self._project.config.video
        width, height = video.generation_size(draft=self._draft)
        params: dict[str, Any] = {
            "positive_prompt": attempt.prompt,
            "negative_prompt": self._still_negative_prompt(shot),
            "seed": attempt.seed,
            "width": width,
            "height": height,
            **attempt.params,
        }
        if shot.type.shows_singer:
            template = self._templates["still"]
            if repair_source is not None:
                params["reference_image"] = self._upload(repair_source)
            else:
                for param, ref_path in still_refs:
                    params[param] = self._upload(ref_path)
        else:
            template = self._templates["t2i"]
            if repair_source is not None and template.accepts("reference_image"):
                params["reference_image"] = self._upload(repair_source)
        if (
            repair_source is None
            and shot.continuity is ContinuityMode.REFRAME
            and template.accepts("previous_frame")
        ):
            handoff = self._previous_handoff_frame(shot)
            if handoff is not None:
                params["previous_frame"] = self._upload(handoff)
        self._attach_prop_refs(params, template, prop_refs)
        self._run_comfy(template.render(params), template.bindings.output, destination)

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
        still_path = self._project.path(still.path)
        clip_duration = round(shot.audio_slice.end - shot.audio_slice.start, 3)
        template = self._video_template(shot)
        params: dict[str, Any] = {
            "positive_prompt": attempt.prompt,
            "negative_prompt": self._manifest.visual_bible.negative_prompt,
            "lyrics": shot.lyrics,
            "seed": attempt.seed,
            "image": self._upload(still_path),
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
                    params["reference_video"] = self._upload(self._with_audio(ref_video_path))
        if template.accepts("end_image"):
            if shot.end_frame_path:
                end_path = self._project.path(shot.end_frame_path)
                if end_path.is_file():
                    params["end_image"] = self._upload(end_path)
            elif (
                shot.camera_path_zoom is not None
                and not shot.type.shows_singer
                and still_path.is_file()
            ):
                # Zoomed-crop end guide for empty wide camera moves (disabled on human shots
                # to prevent pose-freezing; inspired by afj176/open-cinema-studio).
                safe_shot_id = require_safe_id(shot.id, "shot id")
                guide_dst = self._project.path(f"assets/stills/{safe_shot_id}_path_guide.png")
                media.zoom_crop_image(still_path, guide_dst, zoom_factor=shot.camera_path_zoom)
                params["end_image"] = self._upload(guide_dst)
        self._attach_prop_refs(params, template, prop_refs)
        previz = self._previz_for(shot, template)
        if previz is not None:
            params["previz_video"] = self._upload(self._with_audio(previz))
        previz_depth = self._previz_depth_for(shot, template)
        if previz_depth is not None:
            params["previz_depth_video"] = self._upload(self._with_audio(previz_depth))
        self._run_comfy(template.render(params), template.bindings.output, destination)

    def _with_audio(self, video_path: Path) -> Path:
        """``video_path``, or a cached copy of it with a silent audio track to upload instead."""
        return ensure_upload_audio_track(self._project, video_path)

    def _run_comfy(self, graph: dict[str, Any], output: OutputSpec, destination: Path) -> Path:
        """Render ``graph``, using the content-addressed cache when it can.

        The cache is an optimization, so none of its failure modes may change the outcome of a
        render. `canonical_graph_key` rejects a non-finite float in the graph and `store` rejects
        an empty or symlinked source, both with a bare `ValueError` that `_attempt` does not
        catch (it handles only ComfyJobError/MediaToolError) — so a cache problem used to abort
        the entire run with a traceback, discarding every shot's progress, even though the render
        itself had succeeded. Each cache interaction is now best-effort and logged.
        """
        key: str | None = None
        if self._cache is not None:
            try:
                key = canonical_graph_key(
                    graph,
                    self._uploaded_local,
                    uploaded_digests=self._uploaded_digests,
                    output_node=output.node,
                    extensions=tuple(output.extensions),
                )
            except (ValueError, OSError) as exc:
                log.warning("Skipping ComfyUI cache for this render (unusable key): %s", exc)
            if key is not None:
                try:
                    if self._cache.lookup(key, destination):
                        return destination
                except (ValueError, OSError) as exc:
                    log.warning("ComfyUI cache lookup failed; rendering instead: %s", exc)
        result = self._comfy.run(graph, output, destination)
        if self._cache is not None and key is not None:
            try:
                self._cache.store(key, result)
            except (ValueError, OSError) as exc:
                log.warning("Could not cache render %s: %s", result.name, exc)
        return result

    def _extra_reference_images(self, shot: Shot) -> list[tuple[str, Path]]:
        """Reference sheets for secondary visible performers and props in this shot."""
        extras: list[tuple[str, Path]] = []
        if shot.type.shows_singer:
            for sec_id, sec_char, sec_state in self._manifest.visible_characters(shot)[1:]:
                sec_ref = sec_char.reference_for(sec_state)
                sec_sheet = (sec_ref.sheet or sec_ref.panel("front")) if sec_ref else None
                if sec_sheet:
                    sec_path = self._project.path(sec_sheet)
                    if sec_path.is_file():
                        label = sec_char.name or sec_id or "performer"
                        extras.append((label, sec_path))
        for prop_name, prop_path in self._prop_paths(shot):
            extras.append((f"Prop: {prop_name}", prop_path))
        return extras

    def _check(self, shot: Shot, kind: AssetKind, path: Path) -> QCReport:
        # The vision judge sees the whole reference sheet (every angle). It gets the image as a
        # path, so it works even when face analysis is unavailable.
        reference = self._reference_set(shot) if shot.type.shows_singer else None
        sheet = reference.sheet or reference.panel("front") if reference else None
        ref_candidate = self._project.path(sheet) if sheet else None
        reference_image = (
            ref_candidate if (ref_candidate is not None and ref_candidate.is_file()) else None
        )
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
            prev_tail_frames=self._previous_tail_frames(shot),
            beat_times=self._beats_for_shot(shot),
        )

    def _previous_tail_frames(self, shot: Shot) -> list[Any] | None:
        """Last 4 played frames of the previous shot when auditing a CONTINUE seam."""
        still = shot.still.selected
        if (
            shot.continuity is not ContinuityMode.CONTINUE
            or still is None
            or still.strategy is not RetryStrategy.HANDOFF
        ):
            return None  # without a handoff still the join is a plain cut, not a seam
        prev = self._previous_shot(shot)
        tail = self._played_tail(prev) if prev is not None else None
        if tail is None:
            return None
        video_path, last_index, _ = tail
        try:
            frames = media.extract_frames(video_path, 128)
        except MediaToolError:
            return None
        end_idx = min(len(frames), last_index + 1)
        return frames[max(0, end_idx - 4) : end_idx] or None

    def _beats_for_shot(self, shot: Shot) -> list[float] | None:
        """Beat timestamps relative to ``shot.audio_slice.start`` for beat-motion telemetry."""
        if self._analysis_beats is None:
            try:
                self._analysis_beats = list(self._project.load_analysis().beats)
            except Exception:  # noqa: BLE001
                self._analysis_beats = []
        if not self._analysis_beats:
            return None
        start = shot.audio_slice.start
        end = shot.audio_slice.end
        rel = [round(b - start, 4) for b in self._analysis_beats if start <= b <= end]
        return rel if len(rel) >= 2 else None

    def _can_become_cutaway(self, shot: Shot) -> bool:
        """True if turning ``shot`` into ``broll`` keeps consecutive cutaways <= 2."""
        shots = self._manifest.shots
        pos = next((i for i, item in enumerate(shots) if item.id == shot.id), None)
        if pos is None:
            return False
        run = 1
        for step in (-1, 1):
            at = pos + step
            while 0 <= at < len(shots) and shots[at].type is ShotType.BROLL:
                run += 1
                at += step
        return run <= 2

    def _should_rescue_weak_lipsync(self, shot: Shot, attempt: Attempt) -> bool:
        """True when smart_cutaway_rescue should replace a weak lip-sync take with a cutaway."""
        retry_cfg = self._project.config.retry
        if not retry_cfg.smart_cutaway_rescue or shot.type is not ShotType.LIPSYNC:
            return False
        if attempt.qc is None:
            return False
        sync_check = next((c for c in attempt.qc.checks if c.name == "lipsync"), None)
        if sync_check is None or "correlation" not in sync_check.metrics:
            return False
        return sync_check.metrics["correlation"] < retry_cfg.smart_cutaway_max_correlation

    def _convert_lipsync_to_broll(self, shot: Shot, state: AssetState) -> bool:
        """Convert a weak or exhausted singing shot into a cutaway if within the cutaway run cap.

        Credits & Attribution:
            Smart Cutaway rescue prioritizing weak-lip-sync singing shots inspired by
            ``VisualFrisson/ComfyUI-OneShotMV`` (``fv/broll.py``).
        """
        if (
            not self._can_become_cutaway(shot)
            or "t2i" not in self._templates
            or "i2v" not in self._templates
        ):
            return False
        # Snapshot mutable shot fields so a failed cutaway still or exception rolls back cleanly.
        snapshot = (
            shot.fallback_from,
            shot.type,
            shot.continuity,
            shot.character,
            list(shot.characters),
            list(shot.singers),
            shot.wardrobe,
            dict(shot.character_wardrobes),
            shot.still_prompt,
            shot.motion_prompt,
            shot.still.model_copy(deep=True),
        )
        try:
            shot.fallback_from, shot.type = shot.type, ShotType.BROLL
            shot.continuity = ContinuityMode.CUT
            shot.character = None
            shot.characters = []
            shot.singers = []
            shot.wardrobe = None
            shot.character_wardrobes = {}
            shot.still_prompt = (
                f"Empty atmospheric cutaway in {shot.setting or 'the scene'}, "
                "evoking the mood of the song with dramatic light and texture. No people."
            )
            shot.motion_prompt = (
                "Light and shadow shift steadily across the frame with natural movement. "
                "No people. Gentle push-in."
            )
            shot.still.reset(self._project.config.retry.max_attempts)
            shot.still.prompt_override = None
            shot.still.param_overrides = {}
            self._run_asset(shot, AssetKind.STILL)
        except BaseException:
            (
                shot.fallback_from,
                shot.type,
                shot.continuity,
                shot.character,
                shot.characters,
                shot.singers,
                shot.wardrobe,
                shot.character_wardrobes,
                shot.still_prompt,
                shot.motion_prompt,
                shot.still,
            ) = snapshot
            raise
        if shot.still.status is not AssetStatus.PASSED:
            (
                shot.fallback_from,
                shot.type,
                shot.continuity,
                shot.character,
                shot.characters,
                shot.singers,
                shot.wardrobe,
                shot.character_wardrobes,
                shot.still_prompt,
                shot.motion_prompt,
                shot.still,
            ) = snapshot
            return False
        state.reset(self._project.config.retry.max_attempts)
        state.prompt_override = None
        state.param_overrides = {}
        shot.review = Review()
        log.warning(
            "%s: weak/exhausted lip-sync rescued as a b-roll cutaway (within 2-cutaway limit)",
            shot.id,
        )
        self._emit(RunEvent("shot_converted", shot_id=shot.id, asset=AssetKind.VIDEO.value))
        self._project.save_manifest(self._manifest)
        return True

    def _fall_back_from_lipsync(self, shot: Shot, state: AssetState) -> bool:
        """Convert an exhausted singing shot to b-roll (when configured) or performance."""
        if (
            not self._project.config.retry.lipsync_fallback
            or shot.type is not ShotType.LIPSYNC
            or "i2v" not in self._templates
        ):
            return False
        if (
            self._project.config.retry.lipsync_fallback_mode == "broll"
            and self._convert_lipsync_to_broll(shot, state)
        ):
            return True
        return self._fall_back_to_performance(shot, state)

    def _fall_back_to_performance(self, shot: Shot, state: AssetState) -> bool:
        """Turn a singing shot that could not be made into one where the singer is not singing.

        The still stays; the video is rendered again as a performance shot, which needs no
        lip-sync and so does not depend on the face being tracked. Earlier attempts stay in
        the history.
        """
        if (
            not self._project.config.retry.lipsync_fallback
            or shot.type is not ShotType.LIPSYNC
            or "i2v" not in self._templates
        ):
            return False
        shot.fallback_from, shot.type = shot.type, ShotType.PERFORMANCE
        shot.singers = []
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

    def _effective_mute_spans(self, shot: Shot) -> list[tuple[float, float]]:
        """Song-timeline spans silenced inside ``shot.audio_slice`` to prevent duet voice bleed."""
        if shot.type is not ShotType.LIPSYNC:
            return []
        s_start, s_end = shot.audio_slice.start, shot.audio_slice.end
        spans: list[tuple[float, float]] = [
            (max(s_start, m.start), min(s_end, m.end))
            for m in shot.audio_slice.mute_spans
            if min(s_end, m.end) - max(s_start, m.start) >= 0.01
        ]
        # Also guard adjacent-shot singer handoffs on older manifests that predate mute_spans.
        active = self._manifest.active_singer_ids(shot)
        if len(self._manifest.characters) > 1 and active:
            shots = self._manifest.shots
            pos = next((i for i, item in enumerate(shots) if item.id == shot.id), None)
            if pos is not None:
                if pos > 0 and s_start < shot.start:
                    prev = shots[pos - 1]
                    prev_singers = self._manifest.active_singer_ids(prev) or (
                        {rc}
                        if prev.character
                        and (rc := self._manifest.resolve_character_id(prev.character))
                        else set()
                    )
                    if prev_singers and prev_singers != active:
                        spans.append((s_start, shot.start))
                if pos + 1 < len(shots) and s_end > shot.end:
                    nxt = shots[pos + 1]
                    next_singers = self._manifest.active_singer_ids(nxt) or (
                        {rc}
                        if nxt.character
                        and (rc := self._manifest.resolve_character_id(nxt.character))
                        else set()
                    )
                    if next_singers and next_singers != active:
                        spans.append((shot.end, s_end))
        if not spans:
            return []
        ordered = sorted(spans)
        merged: list[list[float]] = [[ordered[0][0], ordered[0][1]]]
        for a, b in ordered[1:]:
            if a <= merged[-1][1] + 0.01:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        return [(round(a, 4), round(b, 4)) for a, b in merged if b - a >= 0.01]

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
        use_vocals = source == "vocals" and (
            shot.type is ShotType.LIPSYNC
            or bool(inputs.vocals and self._project.path(inputs.vocals).is_file())
        )
        chosen_rel = inputs.vocals if use_vocals else inputs.mix
        chosen_label = "vocals" if use_vocals else "mix"
        try:
            source_path = self._project.input_path(chosen_rel, chosen_label)
        except ConfigError:
            if path.is_file():
                return path  # the source audio is gone; keep the slice that was already cut
            raise
        source_stat = source_path.stat()
        mute_spans = self._effective_mute_spans(shot)
        video_cfg = self._project.config.video
        target_dur = round(shot.num_frames / max(video_cfg.fps, 1), 6)
        stamp: dict[str, Any] = {
            "source": source_path.name,
            "source_size": source_stat.st_size,
            "source_mtime_ns": source_stat.st_mtime_ns,
            "start": round(shot.audio_slice.start, 4),
            "end": round(shot.audio_slice.end, 4),
            "vae_offset": round(video_cfg.audio_vae_offset_seconds, 6),
            "mirror_pad": video_cfg.audio_mirror_pad,
            "target_duration": target_dur,
        }
        if mute_spans:
            stamp["mute_spans"] = [list(span) for span in mute_spans]
        marker = path.with_name(f"{path.name}.json")
        if not path.is_file() or _read_stamp(marker) != stamp:
            media.prepare_audio_slice(
                source_path,
                path,
                shot.audio_slice.start,
                shot.audio_slice.end,
                target_duration=target_dur if video_cfg.audio_mirror_pad else None,
                vae_offset_seconds=video_cfg.audio_vae_offset_seconds,
                mirror_pad=video_cfg.audio_mirror_pad,
            )
            if mute_spans:
                # Align relative mute offsets with prepare_audio_slice's VAE-shifted start.
                effective_slice_start = shot.audio_slice.start + video_cfg.audio_vae_offset_seconds
                rel_mutes = [
                    (a - effective_slice_start, b - effective_slice_start) for a, b in mute_spans
                ]
                # If the slice end itself was muted (duet handoff) and mirror-padded, keep the
                # padded overrun region muted too so foreign vocals are never mirrored back in.
                slice_len = shot.audio_slice.end - shot.audio_slice.start
                if (
                    video_cfg.audio_mirror_pad
                    and target_dur > slice_len
                    and any(b >= slice_len - 0.02 for _, b in rel_mutes)
                ):
                    rel_mutes.append((slice_len, target_dur))
                media.mute_audio_spans(path, rel_mutes)
            atomic_write(marker, json.dumps(stamp))
        return path

    def _ensure_audio_video_slice(self, shot: Shot) -> Path:
        wav_path = self._ensure_audio_slice(shot)
        mp4_path = self._project.audio_video_slice_path(shot.id)
        wav_stat = wav_path.stat()
        fps = self._project.config.video.fps
        stamp: dict[str, Any] = {
            "wav_size": wav_stat.st_size,
            "wav_mtime_ns": wav_stat.st_mtime_ns,
            "fps": fps,
        }
        marker = mp4_path.with_name(f"{mp4_path.name}.json")
        if not mp4_path.is_file() or _read_stamp(marker) != stamp:
            media.audio_to_black_video(wav_path, mp4_path, fps=fps)
            atomic_write(marker, json.dumps(stamp))
        return mp4_path

    def _upload(self, path: Path) -> str:
        if not path.is_file():
            raise ConfigError(f"Asset file not found for upload: {path}")
        st = path.stat()
        cache_key = (path, st.st_size, st.st_mtime_ns)
        if cache_key not in self._uploads:
            remote = self._comfy.upload(path)
            self._uploads[cache_key] = remote
            self._uploaded_local[remote] = path
            if self._cache is not None:
                digest = self._file_hashes.get(cache_key)
                if digest is None:
                    digest = file_sha256(path)
                    self._file_hashes[cache_key] = digest
                self._uploaded_digests[remote] = digest
        return self._uploads[cache_key]
