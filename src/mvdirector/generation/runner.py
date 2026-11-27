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
import math
import os
import re
import sys
import threading
import weakref
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import AbstractContextManager, nullcontext, suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mvdirector import assemble, media, plugins
from mvdirector.comfy.cache import ComfyRenderCache, canonical_graph_key, file_sha256
from mvdirector.comfy.client import ComfyClient
from mvdirector.comfy.provider import ComfyEndpointProvider, StaticComfyProvider
from mvdirector.comfy.workflow import OutputSpec, WorkflowTemplate
from mvdirector.config import VideoConfig, validate_max_concurrency
from mvdirector.errors import BudgetExceeded, ComfyJobError, ConfigError, MediaToolError
from mvdirector.generation import plates
from mvdirector.generation.candidates import ensure_upload_audio_track
from mvdirector.models import (
    AssetKind,
    AssetState,
    AssetStatus,
    Attempt,
    ContinuityMode,
    Framing,
    Manifest,
    QCReport,
    ReferenceSet,
    RetryStrategy,
    Review,
    Shot,
    ShotType,
)
from mvdirector.planning import prompts
from mvdirector.planning.segmenter import frames_for
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
        self._used = 0
        self._lock = threading.Lock()

    @property
    def used(self) -> int:
        with self._lock:
            return self._used

    @used.setter
    def used(self, value: int) -> None:
        with self._lock:
            self._used = value

    def consume(self) -> int:
        with self._lock:
            if self._used >= self.limit:
                raise BudgetExceeded(
                    f"Reached the limit of {self.limit} generations for this run "
                    "(retry.max_generations_per_run). Progress is saved; run again to continue."
                )
            self._used += 1
            return self._used

    def refund(self) -> None:
        """Return one reserved slot when endpoint acquisition fails before an attempt starts."""
        with self._lock:
            if self._used > 0:
                self._used -= 1


@dataclass
class _UploadTable:
    """Per-client upload cache and remote-name -> local-file mapping for graph caching."""

    uploads: dict[tuple[Path, int, int], str] = field(default_factory=dict)
    uploaded_local: dict[str, Path] = field(default_factory=dict)
    uploaded_digests: dict[str, str] = field(default_factory=dict)
    in_flight: dict[tuple[Path, int, int], threading.Event] = field(default_factory=dict)


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
    except (OSError, ValueError, RecursionError):
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
        *,
        max_concurrency: int | None = None,
        comfy_provider: ComfyEndpointProvider | None = None,
    ) -> None:
        raw_concurrency = (
            project.config.comfy.max_concurrency if max_concurrency is None else max_concurrency
        )
        self._max_concurrency = validate_max_concurrency(raw_concurrency)
        self._project = project
        self._manifest = manifest
        self._comfy = comfy
        self._comfy_provider = comfy_provider
        self._gates = gates
        self._policy = policy
        self._templates = templates
        self._budget = budget
        self._observer = observer
        self._draft = draft

        primary_table = _UploadTable()
        self._primary_upload_table = primary_table
        self._uploads = primary_table.uploads
        self._uploaded_local = primary_table.uploaded_local
        self._uploaded_digests = primary_table.uploaded_digests
        self._upload_tables: weakref.WeakKeyDictionary[Any, _UploadTable] = (
            weakref.WeakKeyDictionary()
        )
        # Keep a strong reference alongside the table so GC + allocator address reuse cannot
        # map a newly allocated non-weakrefable client (e.g. a different pod) onto stale uploads.
        self._unweakref_upload_tables: dict[int, tuple[Any, _UploadTable]] = {}
        with suppress(TypeError):
            self._upload_tables[comfy] = primary_table

        self._file_hashes: dict[tuple[Path, int, int], str] = {}
        self._identities: dict[tuple[str, ...], IdentityReference] = {}
        self._warned: set[str] = set()
        self._analysis_beats: list[float] | None = None
        self._cache: ComfyRenderCache | None = (
            ComfyRenderCache(project.path(".comfy_cache"))
            if project.config.comfy.cache_enabled
            else None
        )

        # Lock hierarchy (strictly one-way to prevent deadlocks):
        # _qc_lock -> {_state_lock, _manifest_lock}; all remaining locks are leaves.
        self._qc_lock = threading.Lock()
        self._manifest_lock = threading.RLock()
        self._observer_lock = threading.Lock()
        self._upload_lock = threading.Lock()
        self._cache_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._client_pool_lock = threading.Lock()
        self._media_cache_lock = threading.Lock()
        self._primary_busy = False
        self._abort_event = threading.Event()

    # ---------------------------------------------------------------- public

    def run(self, kinds: list[AssetKind], shot_ids: list[str] | None = None) -> None:
        """Process the given asset kinds for the selected shots (all shots by default)."""
        self._abort_event.clear()
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
                        self._sort_for_cache(wave_shots, kind, video=self._project.config.video)
                        if (self._project.config.comfy.sort_batch_for_cache and wave_idx == 0)
                        else wave_shots
                    )
                    self._run_wave(ordered, kind)
        finally:
            with self._upload_lock:
                self._unweakref_upload_tables.clear()
            with self._manifest_lock:
                self._project.save_manifest(self._manifest)
            self._emit(RunEvent("run_finished", generations_used=self._budget.used))

    @staticmethod
    def _shot_needs_work(shot: Shot, kind: AssetKind) -> bool:
        """True when ``shot`` may perform generation work in this phase."""
        if shot.continuity is ContinuityMode.CONTINUE and shot.still.status is AssetStatus.PENDING:
            return True
        state = shot.asset(kind)
        if state.status is not AssetStatus.PENDING:
            return False
        if kind is AssetKind.VIDEO:
            return shot.still.status is AssetStatus.PASSED
        return True

    def _run_wave(self, ordered: list[Shot], kind: AssetKind) -> None:
        """Execute one dependency wave, concurrently when ``max_concurrency > 1``."""
        if self._max_concurrency <= 1:
            for shot in ordered:
                self._run_asset(shot, kind, client=self._comfy)
            return
        work_count = sum(1 for shot in ordered if self._shot_needs_work(shot, kind))
        worker_count = min(self._max_concurrency, work_count)
        if worker_count <= 1:
            for shot in ordered:
                self._run_asset(shot, kind, client=self._comfy)
            return

        self._abort_event.clear()
        first_error: BaseException | None = None
        error_lock = threading.Lock()

        def _record_error(exc: BaseException) -> None:
            nonlocal first_error
            with error_lock:
                if first_error is None or (
                    isinstance(first_error, BudgetExceeded) and not isinstance(exc, BudgetExceeded)
                ):
                    first_error = exc
            self._abort_event.set()

        def _worker(shot: Shot) -> None:
            if self._abort_event.is_set():
                return
            try:
                self._run_asset(shot, kind)
            except BaseException as exc:
                _record_error(exc)

        with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="mvd-shot") as pool:
            futures = [pool.submit(_worker, shot) for shot in ordered]
            try:
                for fut in as_completed(futures):
                    if self._abort_event.is_set():
                        for pending in futures:
                            pending.cancel()
                    if not fut.cancelled():
                        fut.result()
            except BaseException:
                self._abort_event.set()
                for pending in futures:
                    pending.cancel()
                raise

        if first_error is not None:
            raise first_error

    def _acquire_shot_client(
        self,
    ) -> tuple[ComfyClient, AbstractContextManager[ComfyClient] | str | None]:
        """Claim the primary client if idle, else lease an extra client from ``comfy_provider``."""
        if self._comfy_provider is None or isinstance(self._comfy_provider, StaticComfyProvider):
            # A static lease only re-opens the primary's own server, so share the primary
            # (httpx.Client is thread-safe; its VRAM floor check runs once per phase under a
            # lock). A fresh client per shot would re-upload every reference, and its own first
            # VRAM floor check would see the phase model held by another worker's running job
            # and call /free, unloading that model mid-phase or aborting with
            # ComfyResourceError when VRAM stays occupied.
            return self._comfy, None
        with self._client_pool_lock:
            if not self._primary_busy:
                self._primary_busy = True
                return self._comfy, "primary"
        cm = self._comfy_provider.lease(self._project)
        client = cm.__enter__()
        return client, cm

    def _release_shot_client(
        self,
        client: ComfyClient,
        lease_handle: AbstractContextManager[ComfyClient] | str | None,
        exc_info: tuple[Any, Any, Any] = (None, None, None),
    ) -> None:
        if lease_handle is None:
            return
        if lease_handle == "primary":
            with self._client_pool_lock:
                self._primary_busy = False
            return
        assert not isinstance(lease_handle, str)
        lease_handle.__exit__(*exc_info)

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
    def _effective_video_grid(shot: Shot, video: VideoConfig) -> tuple[int, int]:
        """Return ``(fps, num_frames)`` used for ``shot``'s video generation."""
        if (
            shot.type is not ShotType.LIPSYNC
            and video.i2v_fps is not None
            and video.i2v_fps < video.fps
        ):
            clip_duration = max(0.0, shot.audio_slice.end - shot.audio_slice.start)
            return (
                video.i2v_fps,
                frames_for(clip_duration, video.i2v_fps, video.frame_multiple, video.frame_offset),
            )
        return (video.fps, shot.num_frames)

    @staticmethod
    def _video_frame_bucket(num_frames: int) -> int:
        """Round ``num_frames`` to a 16-frame shape bucket to keep torch.compile caches warm."""
        clamped = max(1, int(num_frames))
        return ((clamped + 15) // 16) * 16

    @staticmethod
    def _sort_for_cache(
        shots: list[Shot], kind: AssetKind, *, video: VideoConfig | None = None
    ) -> list[Shot]:
        """Group independent shots by workflow/shape-bucket/setting to keep ComfyUI caches warm.

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
                ShotRunner._video_frame_bucket(
                    ShotRunner._effective_video_grid(s, video)[1]
                    if video is not None
                    else s.num_frames
                ),
                s.setting,
                s.motion_prompt[:48],
                s.index,
            ),
        )

    def _emit(self, event: RunEvent) -> None:
        if self._observer is not None:
            with self._observer_lock:
                self._observer(event)

    # --------------------------------------------------------------- private

    def _reference_set(self, shot: Shot) -> ReferenceSet | None:
        """The reference sheet for the shot's character and wardrobe state, else the base sheet."""
        character = self._manifest.shot_character(shot)
        reference = character.reference_for(shot.wardrobe)
        warn_key = f"{shot.character or ''}:{shot.wardrobe or ''}"
        if reference is not None and shot.wardrobe and shot.wardrobe not in character.references:
            should_warn = False
            with self._state_lock:
                if warn_key not in self._warned:
                    self._warned.add(warn_key)
                    should_warn = True
            if should_warn:
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

    def _plate_for(self, shot: Shot, template: WorkflowTemplate | None = None) -> Path | None:
        """The shot's set plate when the target workflow accepts ``set_image`` and one exists."""
        active_template = template if template is not None else self._templates.get("still")
        if active_template is None or not active_template.accepts("set_image"):
            return None
        if template is None and not shot.type.shows_singer:
            return None
        scene = self._manifest.visual_bible.find_set(shot.setting)
        if scene is None:
            return None
        angle = plates.select_plate_angle(shot, scene)
        candidates: list[str] = []
        if angle and angle in scene.panels:
            candidates.append(scene.panels[angle])
        if scene.plate:
            candidates.append(scene.plate)
        if "wide" in scene.panels:
            candidates.append(scene.panels["wide"])
        candidates.extend(scene.panels.values())
        for rel in dict.fromkeys(candidates):
            path = self._project.path(rel)
            if path.is_file():
                return path
        return None

    def _sketch_for(self, shot: Shot) -> Path | None:
        if not shot.sketch_image:
            return None
        path = self._project.path(shot.sketch_image)
        return path if path.is_file() else None

    def _sketch_binding(
        self,
        shot: Shot,
        template: WorkflowTemplate,
        existing_refs: list[tuple[str, str, Path]],
    ) -> tuple[str, str, Path] | None:
        sketch = self._sketch_for(shot)
        if sketch is None:
            return None
        used_params = {param for param, _, _ in existing_refs}
        used_targets = {
            (t.node, t.field)
            for param in used_params
            for t in template.bindings.inputs.get(param, ())
        }

        def _is_free(param: str) -> bool:
            if param in used_params or not template.accepts(param):
                return False
            targets = {(t.node, t.field) for t in template.bindings.inputs.get(param, ())}
            return bool(targets) and targets.isdisjoint(used_targets)

        if _is_free("sketch_image"):
            return ("sketch_image", "sketch_image", sketch)
        if not shot.type.shows_singer and _is_free("reference_image"):
            return ("reference_image", "sketch_image", sketch)
        for idx in range(2, 11):
            slot = f"reference_image_{idx}"
            if _is_free(slot):
                return (slot, "sketch_image", sketch)
        return None

    def _prop_paths(self, shot: Shot) -> list[tuple[str, Path]]:
        """Locked prop reference images for ``shot.props``, sorted by shot-prompt mention."""
        haystack = (
            f"{shot.still.prompt_override or shot.still_prompt}\n"
            f"{shot.video.prompt_override or shot.motion_prompt}"
        ).casefold()
        candidates: list[tuple[tuple[int, int, int], str, Path]] = []
        for orig_idx, prop in enumerate(self._manifest.visual_bible.resolve_props(shot.props)):
            if not prop.plate:
                continue
            path = self._project.path(prop.plate)
            if not path.is_file():
                continue
            terms = [
                t
                for t in dict.fromkeys(
                    (
                        prop.name.strip().casefold(),
                        prop.id.replace("_", " ").strip().casefold(),
                        prop.id.strip().casefold(),
                    )
                )
                if t
            ]
            first_pos: int | None = None
            for term in terms:
                if term.replace(" ", "").isalnum():
                    match = re.search(rf"\b{re.escape(term)}\b", haystack)
                    pos = match.start() if match else -1
                else:
                    pos = haystack.find(term)
                if pos >= 0 and (first_pos is None or pos < first_pos):
                    first_pos = pos
            rank = (0 if first_pos is not None else 1, first_pos or 0, orig_idx)
            candidates.append((rank, prop.name, path))
        candidates.sort(key=lambda item: item[0])
        return [(name, path) for _, name, path in candidates]

    @staticmethod
    def _prop_slot_names(template: WorkflowTemplate) -> list[str]:
        slots: list[tuple[int, str]] = []
        for name in template.bindings.inputs:
            if name == "prop_image":
                slots.append((1, name))
            elif name.startswith("prop_image_"):
                suffix = name[len("prop_image_") :]
                if suffix.isascii() and suffix.isdigit() and not suffix.startswith("0"):
                    slot_num = int(suffix)
                    if slot_num >= 2:
                        slots.append((slot_num, name))
        slots.sort(key=lambda item: item[0])
        return [name for _, name in slots]

    @staticmethod
    def _targets_for_params(
        template: WorkflowTemplate, param_names: Iterable[str]
    ) -> set[tuple[str, str]]:
        return {
            (t.node, t.field)
            for param in param_names
            for t in template.bindings.inputs.get(param, ())
        }

    def _collect_prop_refs(
        self,
        template: WorkflowTemplate,
        prop_refs: list[tuple[str, Path]],
        *,
        exclude_param: str | None = None,
        used_targets: set[tuple[str, str]] | None = None,
    ) -> tuple[list[tuple[str, str, Path]], dict[str, str]]:
        """Bind ``prop_refs`` onto available ``prop_image`` / ``prop_image_N`` slots."""
        if not prop_refs:
            return [], {}
        slots = self._prop_slot_names(template)
        if not slots:
            return [], {}
        seen_targets = set(used_targets or ())
        bound: list[tuple[str, str, Path]] = []
        bound_names: dict[str, str] = {}
        prop_iter = iter(prop_refs)
        for slot in slots:
            if not self._has_distinct_ref_target(
                template, slot, exclude_param=exclude_param, used_targets=seen_targets
            ):
                continue
            try:
                prop_name, prop_path = next(prop_iter)
            except StopIteration:
                break
            bound.append((slot, slot, prop_path))
            bound_names[slot] = prop_name
            seen_targets.update((t.node, t.field) for t in template.bindings.inputs.get(slot, ()))
        return bound, bound_names

    def _attach_prop_refs(
        self,
        params: dict[str, Any],
        template: WorkflowTemplate,
        prop_refs: list[tuple[str, Path]],
        *,
        client: ComfyClient | None = None,
        exclude_param: str | None = None,
        used_targets: set[tuple[str, str]] | None = None,
    ) -> None:
        prop_slots = set(self._prop_slot_names(template))
        effective_targets = set(used_targets or ()) | self._targets_for_params(
            template, (param for param in params if param not in prop_slots)
        )
        bound, _ = self._collect_prop_refs(
            template,
            prop_refs,
            exclude_param=exclude_param,
            used_targets=effective_targets,
        )
        for param, _, path in bound:
            if param not in params:
                params[param] = self._upload(path, client=client)

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
        source_fps: float = float(fps)
        if self._project.config.video.i2v_fps is not None:
            with suppress(MediaToolError):
                probed = media.probe(video_path)
                if probed.fps is not None and probed.fps > 0 and abs(probed.fps - fps) > 0.5:
                    source_fps = probed.fps
        last_idx = timing.last_played_frame(frames, fps, source_fps)
        return video_path, last_idx, prev_video

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
        attempt = Attempt(
            number=number,
            strategy=RetryStrategy.HANDOFF,
            seed=0,
            prompt="(continued from previous shot handoff frame)",
            params={"handoff_source": prev_video.path, "handoff_frame": frame_index},
            path=self._project.rel(destination),
            qc=QCReport(passed=True, score=1.0),
        )
        with self._manifest_lock:
            shot.audio_slice.start = round(shot.start, 3)
            shot.audio_slice.head_handle = 0.0
            state.attempts.append(attempt)
            state.status = AssetStatus.PASSED
            state.selected_attempt = number
            self._project.save_manifest(self._manifest)
        self._emit(
            RunEvent("asset_passed", asset=AssetKind.STILL.value, shot_id=shot.id, attempt=number)
        )
        return True

    def _run_asset(self, shot: Shot, kind: AssetKind, *, client: ComfyClient | None = None) -> None:
        if self._abort_event.is_set():
            return
        state = shot.asset(kind)
        if (
            shot.continuity is ContinuityMode.CONTINUE
            and shot.still.status is AssetStatus.PENDING
            and shot.still.pending_strategy is None
        ):
            # A one-take continuation's still is the previous clip's last played frame. The video
            # phase takes it as well, for stills an earlier still-only run had to defer.
            settled = self._take_handoff_still(shot, shot.still)
            if kind is AssetKind.STILL and settled:
                return
        if kind is AssetKind.VIDEO and shot.still.status is not AssetStatus.PASSED:
            # Never spend video time on a still that has not passed QC or been accepted.
            log.info("%s: still is %s; skipping video", shot.id, shot.still.status.value)
            return
        active_client: ComfyClient | None = client
        lease_handle: AbstractContextManager[ComfyClient] | str | None = None
        exc_info: tuple[Any, Any, Any] = (None, None, None)
        try:
            while state.status is AssetStatus.PENDING:
                if self._abort_event.is_set():
                    break
                if len(state.attempts) >= state.attempts_allowed:
                    if kind is AssetKind.VIDEO and self._fall_back_from_lipsync(
                        shot, state, client=active_client
                    ):
                        continue
                    self._mark_exhausted(shot, kind, state)
                    break
                generations_used = self._budget.consume()
                if active_client is None:
                    try:
                        active_client, lease_handle = self._acquire_shot_client()
                    except BaseException:
                        self._budget.refund()
                        raise
                can_repair = (
                    kind is AssetKind.STILL and self._still_repair_template(shot) is not None
                )
                try:
                    # Serialize RetryPolicy.prepare under _qc_lock so prompt rewrites and Vision QC
                    # never race on the shared LLMClient instance (_qc_lock -> _manifest_lock).
                    with self._qc_lock:
                        plan = self._policy.prepare(shot, kind, can_repair=can_repair)
                except BaseException:
                    self._budget.refund()
                    raise
                with self._manifest_lock:
                    shot.review = Review()  # any earlier review no longer describes this shot
                try:
                    self._emit(
                        RunEvent(
                            "attempt_started",
                            asset=kind.value,
                            shot_id=shot.id,
                            attempt=plan.number,
                            strategy=plan.strategy.value,
                            generations_used=generations_used,
                        )
                    )
                except BaseException:
                    # An observer that stops the run here (MCP cooperative cancel raises on
                    # attempt_started) does so before anything is queued on the GPU, so the
                    # reserved slot was never spent.
                    self._budget.refund()
                    raise
                repair_instruction = state.repair_instruction
                has_explicit_instruction = bool(repair_instruction and repair_instruction.strip())
                prev_attempt = None
                if (
                    plan.strategy is RetryStrategy.REPAIR_STILL
                    and state.repair_source_attempt is not None
                ):
                    prev_attempt = next(
                        (a for a in state.attempts if a.number == state.repair_source_attempt),
                        None,
                    )
                if prev_attempt is None and state.attempts:
                    prev_attempt = state.attempts[-1]
                attempt = self._attempt(
                    shot,
                    kind,
                    plan,
                    previous_attempt=prev_attempt,
                    repair_instruction=repair_instruction,
                    client=active_client,
                )
                # Anti-regression guard for targeted in-place still repair
                # (inspired by afj176/open-cinema-studio studio.py:_repair_frame).
                if (
                    plan.strategy is RetryStrategy.REPAIR_STILL
                    and prev_attempt is not None
                    and prev_attempt.qc is not None
                    and attempt.qc is not None
                    and not attempt.qc.passed
                ):
                    if has_explicit_instruction and not prev_attempt.qc.passed:
                        regressed = (
                            attempt.qc.has_hard_failure and not prev_attempt.qc.has_hard_failure
                        ) or attempt.qc.score < prev_attempt.qc.score
                    else:
                        old_failed = {c.name for c in prev_attempt.qc.failed_checks}
                        new_failed = {c.name for c in attempt.qc.failed_checks}
                        improved = new_failed <= old_failed and len(attempt.qc.issues) < len(
                            prev_attempt.qc.issues
                        )
                        regressed = not improved
                    if regressed:
                        attempt.regressed = True
                        log.info(
                            "%s still repair did not improve on attempt %d (%s -> %s); "
                            "excluding regressed repair from best_attempt selection",
                            shot.id,
                            prev_attempt.number,
                            len(prev_attempt.qc.issues),
                            len(attempt.qc.issues),
                        )
                with self._manifest_lock:
                    if attempt.error is None or plan.strategy is not RetryStrategy.REPAIR_STILL:
                        state.clear_pending_overrides()
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
                    # Smart Cutaway Rescue: if lip-sync correlation is critically low on an
                    # advisory pass and smart_cutaway_rescue is enabled, convert to a b-roll
                    # cutaway instead of keeping a poor mouth-sync clip
                    # (inspired by VisualFrisson/ComfyUI-OneShotMV).
                    if (
                        kind is AssetKind.VIDEO
                        and self._should_rescue_weak_lipsync(shot, attempt)
                        and self._convert_lipsync_to_broll(shot, state, client=active_client)
                    ):
                        continue
                    with self._manifest_lock:
                        state.status = AssetStatus.PASSED
                        state.selected_attempt = attempt.number
                    self._emit(
                        RunEvent(
                            "asset_passed",
                            asset=kind.value,
                            shot_id=shot.id,
                            attempt=attempt.number,
                        )
                    )
                with self._manifest_lock:
                    self._project.save_manifest(self._manifest)
                if (
                    attempt.error is not None
                    and len(state.attempts) > 1
                    and state.attempts[-2].error == attempt.error
                ):
                    # A different seed or prompt cannot fix a missing model or a dead network,
                    # so stop here and leave the shot pending for the next run.
                    with self._manifest_lock:
                        state.clear_pending_overrides()
                    log.error(
                        "%s %s: the same error twice in a row, so more attempts would not help. "
                        "Fix it and run again: %s",
                        shot.id,
                        kind.value,
                        attempt.error,
                    )
                    break
        except BaseException:
            exc_info = sys.exc_info()
            raise
        finally:
            if active_client is not None and lease_handle is not None:
                self._release_shot_client(active_client, lease_handle, exc_info)

    def _attempt(
        self,
        shot: Shot,
        kind: AssetKind,
        plan: AttemptPlan,
        *,
        previous_attempt: Attempt | None = None,
        repair_instruction: str | None = None,
        client: ComfyClient | None = None,
    ) -> Attempt:
        state = shot.asset(kind)
        plate = self._plate_for(shot) if kind is AssetKind.STILL else None
        notation = self._project.config.comfy.reference_notation
        max_chars = self._project.config.comfy.max_prompt_chars
        still_refs: list[tuple[str, str, Path]] = []
        still_targets: set[tuple[str, str]] = set()
        video_refs: list[tuple[str, str, Path]] = []
        video_targets: set[tuple[str, str]] = set()
        prop_refs = self._prop_paths(shot)
        repair_source: Path | None = None
        if kind is AssetKind.STILL:
            repair_tpl = (
                self._still_repair_template(shot)
                if plan.strategy is RetryStrategy.REPAIR_STILL
                else None
            )
            if (
                plan.strategy is RetryStrategy.REPAIR_STILL
                and previous_attempt is not None
                and previous_attempt.path
                and repair_tpl is not None
            ):
                candidate = self._project.path(previous_attempt.path)
                if candidate.is_file():
                    repair_source = candidate
            if repair_source is not None:
                cleaned_instruction = (repair_instruction or "").strip()
                if cleaned_instruction:
                    prompt = (
                        "Keep the exact composition, framing, lighting, pose, and character "
                        "identity from the input image. Apply ONLY this edit: "
                        f"{cleaned_instruction}"
                    )
                elif (
                    previous_attempt is not None
                    and previous_attempt.qc is not None
                    and (previous_attempt.qc.issues or not previous_attempt.qc.passed)
                ):
                    issues = "; ".join(previous_attempt.qc.issues) or "defects"
                    hints = " ".join(previous_attempt.qc.fix_hints)
                    prompt = (
                        "Keep the exact composition, framing, lighting, pose, and character "
                        f"identity from the input image. Fix ONLY these defects: {issues}. {hints}"
                    ).strip()
                else:
                    prompt = (
                        "Keep the exact composition, framing, lighting, pose, and character "
                        "identity from the input image. Refine image quality and fix any subtle "
                        "visual artifacts."
                    )
            else:
                still_refs = self._still_reference_paths(shot, plate)
                still_tpl = self._templates["still" if shot.type.shows_singer else "t2i"]
                still_targets = self._targets_for_params(
                    still_tpl, (param for param, _, _ in still_refs)
                )
                bound_props, bound_prop_names = self._collect_prop_refs(
                    still_tpl, prop_refs, used_targets=still_targets
                )
                if bound_props:
                    still_refs = [*still_refs, *bound_props]
                sketch_binding = self._sketch_binding(shot, still_tpl, still_refs)
                if sketch_binding is not None:
                    still_refs = [*still_refs, sketch_binding]
                bound_ref_roles = [role for _, role, _ in still_refs] if still_refs else None
                prompt = prompts.compose_still_prompt(
                    self._manifest,
                    shot,
                    plate_attached=plate is not None,
                    props_attached=bool(bound_props),
                    reference_notation=notation,
                    bound_references=bound_ref_roles,
                    bound_prop_names=bound_prop_names or None,
                    sketch_attached=sketch_binding is not None,
                    sketch_guidance=shot.sketch_guidance.strip() or None,
                    pin_character_details=self._project.config.planning.pin_character_details,
                )
        else:
            video_template = self._video_template(shot)
            video_refs = self._video_reference_paths(shot, video_template)
            video_targets = self._targets_for_params(
                video_template, ("image", *(param for param, _, _ in video_refs))
            )
            bound_props, bound_prop_names = self._collect_prop_refs(
                video_template,
                prop_refs,
                exclude_param="image",
                used_targets=video_targets,
            )
            if bound_props:
                # Keep priority order: lead > set > secondaries > props (> draft_grid_image).
                grid_idx = next(
                    (i for i, (_, role, _) in enumerate(video_refs) if role == "draft_grid_image"),
                    len(video_refs),
                )
                video_refs = [
                    *video_refs[:grid_idx],
                    *bound_props,
                    *video_refs[grid_idx:],
                ]
            video_ref_roles = [role for _, role, _ in video_refs] if video_refs else None
            previz = self._previz_for(shot, video_template)
            prompt = prompts.compose_motion_prompt(
                self._manifest,
                shot,
                video_config=self._project.config.video,
                reference_notation=notation,
                has_audio_video=self._video_uses_audio_video(shot, video_template),
                previz_attached=previz is not None,
                bound_references=video_ref_roles,
                bound_prop_names=bound_prop_names or None,
                pin_character_details=self._project.config.planning.pin_character_details,
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
                    shot,
                    attempt,
                    destination,
                    still_refs,
                    prop_refs,
                    repair_source=repair_source,
                    used_targets=still_targets,
                    client=client,
                )
            else:
                self._generate_video(
                    shot,
                    attempt,
                    destination,
                    prop_refs,
                    video_refs=video_refs,
                    used_targets=video_targets,
                    client=client,
                )
            attempt.path = self._project.rel(destination)
            with self._qc_lock:
                attempt.qc = self._check(shot, kind, destination)
            if kind is AssetKind.VIDEO and attempt.qc is not None:
                attempt.sync_delay_seconds = self._extract_sync_delay(shot, attempt.qc)
                if (
                    attempt.qc.passed
                    and shot.type is not ShotType.LIPSYNC
                    and shot.continuity is not ContinuityMode.CONTINUE
                ):
                    attempt.optimal_in_offset = self._select_optimal_in_offset(shot, destination)
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

    def _select_optimal_in_offset(self, shot: Shot, video_path: Path) -> float:
        """Run optional clip_selector plugin on non-lipsync videos, clamped to handles."""
        selector_name = self._project.config.plugins.clip_selector
        if (
            not selector_name
            or shot.type is ShotType.LIPSYNC
            or shot.continuity is ContinuityMode.CONTINUE
        ):
            return 0.0
        head = max(0.0, shot.audio_slice.head_handle)
        tail = max(0.0, shot.audio_slice.end - shot.end)
        if head <= 0.0 and tail <= 0.0:
            return 0.0
        try:
            fn = plugins.load_plugin(plugins.CLIP_SELECTORS_GROUP, selector_name)
            raw = fn(video_path, shot, self._project.config.video.fps, head, tail)
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise TypeError(f"clip_selector {selector_name!r} returned {type(raw).__name__}")
            val = float(raw)
            if not math.isfinite(val):
                raise ValueError("non-finite offset")
            return round(max(-head, min(tail, val)), 4)
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "Clip selector plugin %r failed on %s (%s); using offset 0.0.",
                selector_name,
                shot.id,
                exc,
            )
            return 0.0

    @staticmethod
    def _has_distinct_ref_target(
        template: WorkflowTemplate,
        param: str,
        *,
        exclude_param: str | None = None,
        used_targets: set[tuple[str, str]] | None = None,
    ) -> bool:
        if not template.accepts(param):
            return False
        targets = {(t.node, t.field) for t in template.bindings.inputs.get(param, ())}
        if not targets:
            return False
        if exclude_param is not None:
            excluded = {(t.node, t.field) for t in template.bindings.inputs.get(exclude_param, ())}
            if not targets.isdisjoint(excluded):
                return False
        return not (used_targets and not targets.isdisjoint(used_targets))

    def _collect_reference_paths(
        self,
        shot: Shot,
        template: WorkflowTemplate,
        plate: Path | None,
        *,
        require_primary_reference: bool,
        exclude_param: str | None = None,
    ) -> list[tuple[str, str, Path]]:
        """Ordered ``(binding_param, prompt_role, path)`` triples uploaded to ``template``.

        Priority order: lead character > set plate > secondary characters (with optional face
        panels). Shared by ``still`` and R2V video workflows (``lipsync`` / ``i2v``). On solo shots
        where the workflow exposes ``reference_image_2`` (but not a dedicated ``reference_face``
        slot), the singer's face-detail panel is routed into ``reference_image_2`` with prompt role
        ``reference_face`` so multi-reference models get both full-body wardrobe continuity and
        high-resolution facial identity in one pass.
        """
        refs: list[tuple[str, str, Path]] = []
        used_targets: set[tuple[str, str]] = set()

        def _append_ref(param: str, role: str, path: Path) -> None:
            refs.append((param, role, path))
            used_targets.update((t.node, t.field) for t in template.bindings.inputs.get(param, ()))

        def _append_set_plate() -> None:
            if plate is not None and self._has_distinct_ref_target(
                template, "set_image", exclude_param=exclude_param, used_targets=used_targets
            ):
                _append_ref("set_image", "set_image", plate)

        if shot.type.shows_singer:
            reference = self._reference_set(shot)
            front = reference.panel("front") if reference else None
            raw_still = shot.still.prompt_override or shot.still_prompt
            is_rear_view = shot.type is not ShotType.LIPSYNC and prompts.is_rear_view_prompt(
                raw_still
            )
            is_profile_view = (
                shot.type is not ShotType.LIPSYNC
                and not is_rear_view
                and prompts.is_profile_view_prompt(raw_still)
            )
            is_hands_detail = (
                shot.type is ShotType.PERFORMANCE
                and shot.framing is Framing.CLOSE_UP
                and prompts.is_hands_detail_prompt(raw_still)
            )
            avoid_frontal_face = is_rear_view or is_profile_view or is_hands_detail
            front_path: Path | None = None

            def _first_existing_panel(
                ref_set: ReferenceSet | None, names: tuple[str, ...]
            ) -> str | None:
                if ref_set is None:
                    return None
                for panel_name in names:
                    rel = ref_set.panel(panel_name)
                    if rel is not None and self._project.path(rel).is_file():
                        return rel
                return None

            if self._has_distinct_ref_target(
                template, "reference_image", exclude_param=exclude_param, used_targets=used_targets
            ):
                if reference is None or front is None:
                    if require_primary_reference:
                        raise ConfigError(
                            "No reference sheet is locked. Run `mvd reference generate` and "
                            "`mvd reference lock <image>` before generating stills."
                        )
                else:
                    # Single-reference workflows lose facial detail and pull framing too wide on
                    # close-ups when given the full-body front panel; prefer the macro face panel
                    # when available. On rear-view/profile shots, prefer back/profile panels so the
                    # model is not anchored to a frontal stare; on hands-only inserts, avoid face.
                    face_rel = reference.panel("face")
                    use_face = (
                        not avoid_frontal_face
                        and self._project.config.reference.closeup_face_panel
                        and shot.framing is Framing.CLOSE_UP
                        and not self._has_distinct_ref_target(
                            template, "reference_face", exclude_param=exclude_param
                        )
                        and face_rel is not None
                        and self._project.path(face_rel).is_file()
                    )
                    if is_rear_view:
                        primary_panel = (
                            _first_existing_panel(reference, ("back", "profile", "front")) or front
                        )
                    elif is_profile_view:
                        primary_panel = (
                            _first_existing_panel(reference, ("profile", "three_quarter", "front"))
                            or front
                        )
                    elif use_face and face_rel is not None:
                        primary_panel = face_rel
                    else:
                        primary_panel = front
                    candidate_front = self._project.path(primary_panel)
                    if require_primary_reference or candidate_front.is_file():
                        front_path = candidate_front
                        _append_ref("reference_image", "reference_image", front_path)
            # Extra angles go only to workflows that bind them, so nothing is uploaded for nothing.
            has_face_ref = False
            if reference is not None:
                extra_panels = (
                    ("profile", "three_quarter")
                    if avoid_frontal_face
                    else ("profile", "three_quarter", "face")
                )
                for name in extra_panels:
                    path = reference.panel(name)
                    param = f"reference_{name}"
                    if path and self._has_distinct_ref_target(
                        template, param, exclude_param=exclude_param, used_targets=used_targets
                    ):
                        panel_path = self._project.path(path)
                        if panel_path.is_file():
                            _append_ref(param, param, panel_path)
                            if name == "face":
                                has_face_ref = True
            visible = self._manifest.visible_characters(shot)
            if (
                len(visible) == 1
                and not has_face_ref
                and reference is not None
                and self._has_distinct_ref_target(
                    template,
                    "reference_image_2",
                    exclude_param=exclude_param,
                    used_targets=used_targets,
                )
                and not avoid_frontal_face
            ):
                face = reference.panel("face")
                if face:
                    face_path = self._project.path(face)
                    same_as_front = (
                        front_path is not None and face_path.resolve() == front_path.resolve()
                    )
                    if not same_as_front and face_path.is_file():
                        _append_ref("reference_image_2", "reference_face", face_path)
            _append_set_plate()
            if len(visible) > 1:
                for idx, (_, sec_char, sec_state) in enumerate(visible[1:], start=2):
                    param = f"reference_image_{idx}"
                    sec_ref = sec_char.reference_for(sec_state)
                    sec_path: Path | None = None
                    if self._has_distinct_ref_target(
                        template,
                        param,
                        exclude_param=exclude_param,
                        used_targets=used_targets,
                    ):
                        if is_rear_view:
                            sec_panel = _first_existing_panel(sec_ref, ("back", "profile", "front"))
                        elif is_profile_view:
                            sec_panel = _first_existing_panel(
                                sec_ref, ("profile", "three_quarter", "front")
                            )
                        else:
                            sec_panel = sec_ref.panel("front") if sec_ref else None
                        if sec_panel is not None:
                            candidate = self._project.path(sec_panel)
                            if candidate.is_file():
                                sec_path = candidate
                                _append_ref(param, param, sec_path)
                    face_param = f"reference_image_{idx}_face"
                    if (
                        not avoid_frontal_face
                        and sec_ref is not None
                        and self._has_distinct_ref_target(
                            template,
                            face_param,
                            exclude_param=exclude_param,
                            used_targets=used_targets,
                        )
                    ):
                        sec_face = sec_ref.panel("face")
                        if sec_face is not None:
                            sec_face_path = self._project.path(sec_face)
                            same_as_primary = (
                                sec_path is not None
                                and sec_face_path.resolve() == sec_path.resolve()
                            )
                            if not same_as_primary and sec_face_path.is_file():
                                _append_ref(face_param, face_param, sec_face_path)
        else:
            _append_set_plate()
        return refs

    def _still_reference_paths(self, shot: Shot, plate: Path | None) -> list[tuple[str, str, Path]]:
        """Ordered ``(binding_param, prompt_role, path)`` triples uploaded to ``still``."""
        if not shot.type.shows_singer:
            return []
        return self._collect_reference_paths(
            shot,
            self._templates["still"],
            plate,
            require_primary_reference=True,
        )

    def _video_reference_paths(
        self, shot: Shot, template: WorkflowTemplate
    ) -> list[tuple[str, str, Path]]:
        """Optional R2V character/scene reference triples uploaded to ``lipsync`` / ``i2v``."""
        plate = self._plate_for(shot, template=template)
        refs = self._collect_reference_paths(
            shot,
            template,
            plate,
            require_primary_reference=False,
            exclude_param="image",
        )
        if shot.draft_grid_sheet:
            sheet_path = self._project.path(shot.draft_grid_sheet)
            if sheet_path.is_file():
                used_params = {param for param, _, _ in refs} | {"image"}
                used_targets = {
                    (t.node, t.field)
                    for param in used_params
                    for t in template.bindings.inputs.get(param, ())
                }

                def _is_free(param: str) -> bool:
                    if param in used_params or not template.accepts(param):
                        return False
                    targets = {(t.node, t.field) for t in template.bindings.inputs.get(param, ())}
                    return bool(targets) and targets.isdisjoint(used_targets)

                if _is_free("draft_grid_image"):
                    refs.append(("draft_grid_image", "draft_grid_image", sheet_path))
                elif shot.allow_cuts:
                    for idx in range(2, 11):
                        slot = f"reference_image_{idx}"
                        if _is_free(slot):
                            refs.append((slot, "draft_grid_image", sheet_path))
                            break
        return refs

    def _still_repair_template(self, shot: Shot) -> WorkflowTemplate | None:
        """Workflow template capable of in-place still editing (``reference_image``)."""
        if shot.type.shows_singer:
            tpl = self._templates.get("still")
            return tpl if tpl is not None and tpl.accepts("reference_image") else None
        t2i = self._templates.get("t2i")
        if t2i is not None and t2i.accepts("reference_image"):
            return t2i
        still = self._templates.get("still")
        if still is not None and still.accepts("reference_image"):
            return still
        return None

    def _still_negative_prompt(self, shot: Shot) -> str:
        parts = [self._manifest.visual_bible.negative_prompt]
        raw_still = shot.still.prompt_override or shot.still_prompt
        parts.extend(prompts.extract_negative_exclusions(raw_still))
        if shot.type is ShotType.BROLL:
            parts.append("people, human figures, faces")
        else:
            parts.append("vertically squashed body, stubby legs, oversized head")
            if shot.type is not ShotType.LIPSYNC and prompts.is_rear_view_prompt(raw_still):
                parts.append("frontal face, looking at camera")
        combined = ", ".join(p.strip(", ") for p in parts if p and p.strip(", "))
        return prompts.merge_negative_prompt(combined, self._manifest.visual_bible.exclusions)

    def _generate_still(
        self,
        shot: Shot,
        attempt: Attempt,
        destination: Path,
        still_refs: list[tuple[str, str, Path]],
        prop_refs: list[tuple[str, Path]],
        *,
        repair_source: Path | None = None,
        used_targets: set[tuple[str, str]] | None = None,
        client: ComfyClient | None = None,
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
        if repair_source is not None:
            template = self._still_repair_template(shot)
            assert template is not None  # guarded in _attempt
            params["reference_image"] = self._upload(repair_source, client=client)
        elif shot.type.shows_singer:
            template = self._templates["still"]
            for param, _, ref_path in still_refs:
                params[param] = self._upload(ref_path, client=client)
        else:
            template = self._templates["t2i"]
            for param, _, ref_path in still_refs:
                params[param] = self._upload(ref_path, client=client)
        if (
            repair_source is None
            and shot.continuity is ContinuityMode.REFRAME
            and template.accepts("previous_frame")
        ):
            handoff = self._previous_handoff_frame(shot)
            if handoff is not None:
                params["previous_frame"] = self._upload(handoff, client=client)
        if repair_source is None:
            self._attach_prop_refs(
                params,
                template,
                prop_refs,
                client=client,
                exclude_param="image",
                used_targets=used_targets,
            )
        self._run_comfy(
            template.render(params), template.bindings.output, destination, client=client
        )

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
        *,
        video_refs: list[tuple[str, str, Path]] | None = None,
        used_targets: set[tuple[str, str]] | None = None,
        client: ComfyClient | None = None,
    ) -> None:
        video = self._project.config.video
        width, height = video.generation_size(draft=self._draft)
        still = shot.still.selected
        assert still is not None and still.path is not None  # guarded in _run_asset
        still_path = self._project.path(still.path)
        clip_duration = round(shot.audio_slice.end - shot.audio_slice.start, 3)
        template = self._video_template(shot)
        gen_fps, gen_frames = self._effective_video_grid(shot, video)
        neg_prompt = prompts.merge_negative_prompt(
            self._manifest.visual_bible.negative_prompt,
            self._manifest.visual_bible.exclusions,
        )
        params: dict[str, Any] = {
            "positive_prompt": attempt.prompt,
            "negative_prompt": neg_prompt,
            "nag_enabled": bool(self._project.config.comfy.nag_enabled and neg_prompt.strip()),
            "lyrics": shot.lyrics,
            "seed": attempt.seed,
            "image": self._upload(still_path, client=client),
            "width": width,
            "height": height,
            "num_frames": gen_frames,
            "duration": clip_duration,
            "fps": gen_fps,
            **attempt.params,
        }
        for param, _, ref_path in video_refs or ():
            params[param] = self._upload(ref_path, client=client)
        if shot.type is ShotType.LIPSYNC or "audio" in template.bindings.required:
            audio_path = (
                self._ensure_conditioning_audio(shot)
                if shot.type is ShotType.LIPSYNC
                else self._ensure_audio_slice(shot)
            )
            params["audio"] = self._upload(audio_path, client=client)
        if self._video_uses_audio_video(shot, template):
            params["audio_video"] = self._upload(
                self._ensure_audio_video_slice(shot), client=client
            )
        if shot.type.shows_singer and template.accepts("reference_video"):
            ref = self._reference_set(shot)
            if ref is not None and ref.video:
                ref_video_path = self._project.path(ref.video)
                if ref_video_path.is_file():
                    params["reference_video"] = self._upload(
                        self._with_audio(ref_video_path), client=client
                    )
        if template.accepts("end_image"):
            if shot.end_frame_path:
                end_path = self._project.path(shot.end_frame_path)
                if end_path.is_file():
                    params["end_image"] = self._upload(end_path, client=client)
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
                params["end_image"] = self._upload(guide_dst, client=client)
        self._attach_prop_refs(
            params,
            template,
            prop_refs,
            client=client,
            exclude_param="image",
            used_targets=used_targets,
        )
        previz = self._previz_for(shot, template)
        if previz is not None:
            params["previz_video"] = self._upload(self._with_audio(previz), client=client)
        previz_depth = self._previz_depth_for(shot, template)
        if previz_depth is not None:
            params["previz_depth_video"] = self._upload(
                self._with_audio(previz_depth), client=client
            )
        self._run_comfy(
            template.render(params), template.bindings.output, destination, client=client
        )

    def _with_audio(self, video_path: Path) -> Path:
        """``video_path``, or a cached copy of it with a silent audio track to upload instead."""
        with self._media_cache_lock:
            return ensure_upload_audio_track(self._project, video_path)

    def _run_comfy(
        self,
        graph: dict[str, Any],
        output: OutputSpec,
        destination: Path,
        *,
        client: ComfyClient | None = None,
    ) -> Path:
        """Render ``graph``, using the content-addressed cache when it can.

        The cache is an optimization, so none of its failure modes may change the outcome of a
        render. `canonical_graph_key` rejects a non-finite float in the graph and `store` rejects
        an empty or symlinked source, both with a bare `ValueError` that `_attempt` does not
        catch (it handles only ComfyJobError/MediaToolError) — so a cache problem used to abort
        the entire run with a traceback, discarding every shot's progress, even though the render
        itself had succeeded. Each cache interaction is now best-effort and logged.
        """
        active_client = self._comfy if client is None else client
        cache_lock = getattr(self, "_cache_lock", nullcontext())
        key: str | None = None
        if self._cache is not None:
            if hasattr(self, "_upload_lock") and hasattr(self, "_primary_upload_table"):
                table = self._upload_table_for(active_client)
                with self._upload_lock:
                    uploaded_local = dict(table.uploaded_local)
                    uploaded_digests = dict(table.uploaded_digests)
            else:
                uploaded_local = self._uploaded_local
                uploaded_digests = self._uploaded_digests
            try:
                key = canonical_graph_key(
                    graph,
                    uploaded_local,
                    uploaded_digests=uploaded_digests,
                    output_node=output.node,
                    extensions=tuple(output.extensions),
                )
            except (ValueError, OSError) as exc:
                log.warning("Skipping ComfyUI cache for this render (unusable key): %s", exc)
            if key is not None:
                try:
                    with cache_lock:
                        hit = self._cache.lookup(key, destination)
                    if hit:
                        return destination
                except (ValueError, OSError) as exc:
                    log.warning("ComfyUI cache lookup failed; rendering instead: %s", exc)
        result = active_client.run(graph, output, destination)
        if self._cache is not None and key is not None:
            try:
                with cache_lock:
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

    def _convert_lipsync_to_broll(
        self,
        shot: Shot,
        state: AssetState,
        *,
        client: ComfyClient | None = None,
    ) -> bool:
        """Convert a weak or exhausted singing shot into a cutaway if within the cutaway run cap.

        Credits & Attribution:
            Smart Cutaway rescue prioritizing weak-lip-sync singing shots inspired by
            ``VisualFrisson/ComfyUI-OneShotMV`` (``fv/broll.py``).
        """
        with self._manifest_lock:
            if (
                not self._can_become_cutaway(shot)
                or "t2i" not in self._templates
                or "i2v" not in self._templates
            ):
                return False
            # Snapshot mutable shot fields so a failed cutaway still or exception rolls back
            # cleanly. Mutating shot.type to BROLL under _manifest_lock also reserves this
            # slot against adjacent parallel workers checking the 2-cutaway run cap.
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

        def _restore() -> None:
            with self._manifest_lock:
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

        try:
            self._run_asset(shot, AssetKind.STILL, client=client)
        except BaseException:
            _restore()
            raise
        if shot.still.status is not AssetStatus.PASSED:
            _restore()
            return False
        with self._manifest_lock:
            state.reset(self._project.config.retry.max_attempts)
            state.prompt_override = None
            state.param_overrides = {}
            shot.review = Review()
            self._project.save_manifest(self._manifest)
        log.warning(
            "%s: weak/exhausted lip-sync rescued as a b-roll cutaway (within 2-cutaway limit)",
            shot.id,
        )
        self._emit(RunEvent("shot_converted", shot_id=shot.id, asset=AssetKind.VIDEO.value))
        return True

    def _fall_back_from_lipsync(
        self,
        shot: Shot,
        state: AssetState,
        *,
        client: ComfyClient | None = None,
    ) -> bool:
        """Convert an exhausted singing shot to b-roll (when configured) or performance."""
        if (
            not self._project.config.retry.lipsync_fallback
            or shot.type is not ShotType.LIPSYNC
            or "i2v" not in self._templates
        ):
            return False
        if (
            self._project.config.retry.lipsync_fallback_mode == "broll"
            and self._convert_lipsync_to_broll(shot, state, client=client)
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
        with self._manifest_lock:
            shot.fallback_from, shot.type = shot.type, ShotType.PERFORMANCE
            shot.singers = []
            shot.motion_prompt = prompts.PERFORMANCE_MOTION
            state.reset(self._project.config.retry.max_attempts)
            state.prompt_override = None
            state.param_overrides = {}
            shot.review = Review()
            self._project.save_manifest(self._manifest)
        log.warning(
            "%s: no usable lip-sync video, so it is now a performance shot (the singer on "
            "camera, not singing)",
            shot.id,
        )
        self._emit(RunEvent("shot_converted", shot_id=shot.id, asset=AssetKind.VIDEO.value))
        return True

    def _mark_exhausted(self, shot: Shot, kind: AssetKind, state: AssetState) -> None:
        with self._manifest_lock:
            best = state.best_attempt()
            state.status = AssetStatus.EXHAUSTED
            state.selected_attempt = best.number if best else None
            state.clear_pending_overrides()
            if kind is AssetKind.VIDEO and not self._project.config.retry.keep_still_on_video_fail:
                shot.still.reset(self._project.config.retry.max_attempts)
            self._project.save_manifest(self._manifest)
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

    def _effective_mute_spans(self, shot: Shot) -> list[tuple[float, float]]:
        """Song-timeline spans silenced inside ``shot.audio_slice`` to prevent duet voice bleed."""
        with self._manifest_lock:
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

    def _ensure_conditioning_audio(self, shot: Shot) -> Path:
        """Conditioned vocal slice for lip-sync generation; QC still scores the original slice."""
        raw_path = self._ensure_audio_slice(shot)
        plugin_name = self._project.config.plugins.audio_conditioner
        if not plugin_name or shot.type is not ShotType.LIPSYNC:
            return raw_path
        try:
            fn = plugins.load_plugin(plugins.AUDIO_CONDITIONERS_GROUP, plugin_name)
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "Audio conditioner plugin %r unavailable (%s); using raw slice.", plugin_name, exc
            )
            return raw_path
        cond_path = raw_path.with_name(f"{raw_path.stem}.cond{raw_path.suffix}")
        project_root = self._project.root.resolve()
        cond_resolved = cond_path.resolve()
        raw_stat = raw_path.stat()
        stamp: dict[str, Any] = {
            "raw_size": raw_stat.st_size,
            "raw_mtime_ns": raw_stat.st_mtime_ns,
            "plugin": plugin_name,
            "version": plugins.plugin_version(fn),
        }
        marker = cond_path.with_name(f"{cond_path.name}.json")
        if (
            not cond_path.is_symlink()
            and cond_resolved.is_relative_to(project_root)
            and cond_resolved.is_file()
            and cond_resolved.stat().st_size > 0
            and _read_stamp(marker) == stamp
        ):
            return cond_path
        try:
            if cond_path.is_symlink():
                raise ValueError(f"Conditioned audio path must not be a symlink: {cond_path}")
            analysis = None
            with suppress(Exception):
                analysis = self._project.load_analysis()
            out = fn(raw_path, cond_path, shot, analysis)
            if out is None:
                candidate = cond_path
            elif isinstance(out, (str, os.PathLike)):
                candidate = Path(out)
            else:
                raise TypeError(
                    f"audio_conditioner {plugin_name!r} returned invalid path type "
                    f"{type(out).__name__}"
                )
            resolved = candidate.resolve()
            if not resolved.is_relative_to(project_root):
                raise ValueError(
                    f"audio_conditioner {plugin_name!r} returned path outside project root: "
                    f"{candidate}"
                )
            if not resolved.is_file() or resolved.stat().st_size <= 0:
                raise OSError(f"audio_conditioner {plugin_name!r} did not write a non-empty file")
            if resolved == cond_path.resolve():
                atomic_write(marker, json.dumps(stamp))
                return cond_path
            marker.unlink(missing_ok=True)
            return resolved
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "Audio conditioner plugin %r failed on %s (%s); falling back to raw slice.",
                plugin_name,
                shot.id,
                exc,
            )
            return raw_path

    def _ensure_audio_video_slice(self, shot: Shot) -> Path:
        wav_path = (
            self._ensure_conditioning_audio(shot)
            if shot.type is ShotType.LIPSYNC
            else self._ensure_audio_slice(shot)
        )
        mp4_path = self._project.audio_video_slice_path(shot.id)
        wav_stat = wav_path.stat()
        wav_resolved = wav_path.resolve()
        project_root = self._project.root.resolve()
        wav_key = (
            wav_resolved.relative_to(project_root).as_posix()
            if wav_resolved.is_relative_to(project_root)
            else str(wav_resolved)
        )
        fps = self._project.config.video.fps
        stamp: dict[str, Any] = {
            "wav_path": wav_key,
            "wav_size": wav_stat.st_size,
            "wav_mtime_ns": wav_stat.st_mtime_ns,
            "fps": fps,
        }
        marker = mp4_path.with_name(f"{mp4_path.name}.json")
        if not mp4_path.is_file() or _read_stamp(marker) != stamp:
            media.audio_to_black_video(wav_path, mp4_path, fps=fps)
            atomic_write(marker, json.dumps(stamp))
        return mp4_path

    def _upload_table_for(self, client: ComfyClient) -> _UploadTable:
        if client is self._comfy:
            return self._primary_upload_table
        with self._upload_lock:
            try:
                table = self._upload_tables.get(client)
                if table is None:
                    table = _UploadTable()
                    self._upload_tables[client] = table
                return table
            except TypeError:
                entry = self._unweakref_upload_tables.get(id(client))
                if entry is None:
                    table = _UploadTable()
                    self._unweakref_upload_tables[id(client)] = (client, table)
                    return table
                return entry[1]

    def _upload(self, path: Path, *, client: ComfyClient | None = None) -> str:
        if not path.is_file():
            raise ConfigError(f"Asset file not found for upload: {path}")
        active_client = self._comfy if client is None else client
        st = path.stat()
        cache_key = (path, st.st_size, st.st_mtime_ns)
        table = self._upload_table_for(active_client)
        while True:
            with self._upload_lock:
                existing = table.uploads.get(cache_key)
                if existing is not None:
                    return existing
                waiter = table.in_flight.get(cache_key)
                if waiter is None:
                    owner_event = threading.Event()
                    table.in_flight[cache_key] = owner_event
                    break
            waiter.wait()

        try:
            remote = active_client.upload(path)
            digest: str | None = None
            if self._cache is not None:
                with self._upload_lock:
                    digest = self._file_hashes.get(cache_key)
                if digest is None:
                    digest = file_sha256(path)
            with self._upload_lock:
                table.uploads[cache_key] = remote
                table.uploaded_local[remote] = path
                if digest is not None:
                    self._file_hashes[cache_key] = digest
                    table.uploaded_digests[remote] = digest
            return remote
        finally:
            with self._upload_lock:
                table.in_flight.pop(cache_key, None)
            owner_event.set()
