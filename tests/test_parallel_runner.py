# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Parallel wave execution, per-worker client leasing, and thread-safety tests for ShotRunner."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import ValidationError

from mvdirector import api
from mvdirector.comfy.client import ComfyClient
from mvdirector.comfy.workflow import OutputSpec
from mvdirector.config import ComfyConfig
from mvdirector.errors import BudgetExceeded, ComfyConnectionError, ComfyJobError, ConfigError
from mvdirector.generation import reference
from mvdirector.generation.runner import RunBudget, RunEvent, ShotRunner, load_templates
from mvdirector.models import (
    AssetKind,
    AssetStatus,
    CheckResult,
    CheckStatus,
    ContinuityMode,
    QCReport,
    ShotType,
)
from mvdirector.planning.planner import build_manifest
from mvdirector.qc.factory import QualityGates
from mvdirector.retry import RetryPolicy
from tests.conftest import FakeComfy, needs_ffmpeg


class _StrictSerialGate:
    """QC gate that fails immediately if invoked concurrently from two threads."""

    def __init__(self, *, pass_all: bool = True, correlation: float | None = None) -> None:
        self._lock = threading.Lock()
        self._active = 0
        self.overlap_detected = False
        self.calls = 0
        self._pass_all = pass_all
        self._correlation = correlation

    def evaluate(self, *args: Any, **kwargs: Any) -> QCReport:
        with self._lock:
            self._active += 1
            if self._active > 1:
                self.overlap_detected = True
            self.calls += 1
        try:
            time.sleep(0.01)
            checks = []
            if self._correlation is not None:
                checks.append(
                    CheckResult(
                        name="lipsync",
                        status=CheckStatus.PASSED,
                        score=self._correlation,
                        metrics={"correlation": self._correlation},
                    )
                )
            return QCReport(
                passed=self._pass_all, score=0.9 if self._pass_all else 0.2, checks=checks
            )
        finally:
            with self._lock:
                self._active -= 1


class _ConcurrentStubComfy:
    """Thread-safe stub simulating either a Serverless endpoint or an isolated GPU pod."""

    def __init__(
        self,
        name: str,
        png: bytes,
        mp4: bytes = b"mp4-bytes",
        *,
        barrier: threading.Barrier | None = None,
        barrier_done: threading.Event | None = None,
        delay: float = 0.0,
        require_local_upload: bool = True,
    ) -> None:
        self.name = name
        self._png = png
        self._mp4 = mp4
        self._barrier = barrier
        self._barrier_done = barrier_done if barrier_done is not None else threading.Event()
        self._delay = delay
        self._require_local_upload = require_local_upload
        self._lock = threading.Lock()
        self.uploads: list[Path] = []
        self.uploaded_names: set[str] = set()
        self.active = 0
        self.peak_active = 0
        self.runs = 0
        self.free_calls = 0

    def ping(self) -> dict[str, Any]:
        return {"system": {"comfyui_version": "test"}, "devices": []}

    def free_memory(self) -> None:
        with self._lock:
            self.free_calls += 1

    def upload(self, path: Path) -> str:
        if self._delay:
            time.sleep(self._delay)
        with self._lock:
            idx = len(self.uploads)
            self.uploads.append(path)
            remote = f"{self.name}/upload_{idx}_{path.name}"
            self.uploaded_names.add(remote)
            return remote

    def run(self, graph: dict[str, Any], output: OutputSpec, destination: Path) -> Path:
        if self._require_local_upload:
            load_inputs = graph.get("3", {}).get("inputs", {})
            for field in ("image", "audio"):
                val = load_inputs.get(field)
                if val:
                    with self._lock:
                        present = val in self.uploaded_names
                    if not present:
                        raise ComfyJobError(
                            f"{self.name}: referenced {val!r} was never uploaded to this client"
                        )
        with self._lock:
            self.active += 1
            self.peak_active = max(self.peak_active, self.active)
            self.runs += 1
        try:
            if self._barrier is not None and not self._barrier_done.is_set():
                self._barrier.wait()
                self._barrier_done.set()
            if self._delay:
                time.sleep(self._delay)
            destination.parent.mkdir(parents=True, exist_ok=True)
            is_video = ".mp4" in output.extensions
            destination.write_bytes(self._mp4 if is_video else self._png)
            return destination
        finally:
            with self._lock:
                self.active -= 1


@pytest.fixture
def planned_six(project, analysis, tmp_path, sample_png):
    """Project with 6 planned shots and a locked reference portrait."""
    project.config.video.max_clip_seconds = 4.0
    manifest = build_manifest(project.config, analysis, "lyrics", "style", llm=None)
    assert len(manifest.shots) == 6
    portrait = tmp_path / "portrait.png"
    portrait.write_bytes(sample_png)
    reference.lock_reference(project, manifest, portrait)
    project.save_manifest(manifest)
    return project, manifest


def _make_runner(
    project,
    manifest,
    comfy,
    image_gate,
    video_gate,
    *,
    budget: int = 100,
    observer=None,
    max_concurrency: int | None = None,
    comfy_provider=None,
) -> ShotRunner:
    gates = QualityGates(image=image_gate, video=video_gate, face_analyzer=None)
    policy = RetryPolicy(project.config.retry, project.config.seed, llm=None)
    templates = load_templates(project, ["t2i", "still", "lipsync", "i2v"])
    return ShotRunner(
        project,
        manifest,
        comfy,
        gates,
        policy,
        templates,
        RunBudget(budget),
        observer=observer,
        max_concurrency=max_concurrency,
        comfy_provider=comfy_provider,
    )


def test_parallel_wave_executes_concurrently_and_dedups_shared_uploads(planned_six, sample_png):
    """Shared client (Serverless pattern): 3 shots run in parallel and upload portrait once."""
    project, manifest = planned_six
    barrier = threading.Barrier(3, timeout=5.0)
    comfy = _ConcurrentStubComfy("serverless", sample_png, barrier=barrier, delay=0.02)
    image_gate = _StrictSerialGate()
    video_gate = _StrictSerialGate()

    observer_lock = threading.Lock()
    observer_active = 0
    observer_overlap = False
    events: list[RunEvent] = []

    def observer(event: RunEvent) -> None:
        nonlocal observer_active, observer_overlap
        with observer_lock:
            observer_active += 1
            if observer_active > 1:
                observer_overlap = True
        try:
            time.sleep(0.005)
            events.append(event)
        finally:
            with observer_lock:
                observer_active -= 1

    runner = _make_runner(
        project,
        manifest,
        comfy,
        image_gate,
        video_gate,
        observer=observer,
        max_concurrency=3,
    )
    runner.run([AssetKind.STILL])

    assert comfy.peak_active == 3
    assert comfy.runs == 6
    # All 6 shots share the same reference portrait; single-flight uploads it exactly once.
    assert len(comfy.uploads) == 1
    assert all(s.still.status is AssetStatus.PASSED for s in manifest.shots)
    assert manifest.total_generations == 6
    assert not image_gate.overlap_detected
    assert not observer_overlap
    assert events[-1].kind == "run_finished" and events[-1].generations_used == 6
    saved = project.load_manifest()
    assert saved.total_generations == 6
    assert all(s.still.status is AssetStatus.PASSED for s in saved.shots)


def test_parallel_wave_leases_per_worker_clients_and_isolates_uploads(planned_six, sample_png):
    """Multi-pod pool pattern: each worker leases its own pod and uploads references to its pod."""
    project, manifest = planned_six
    barrier = threading.Barrier(3, timeout=5.0)
    barrier_done = threading.Event()
    primary = _ConcurrentStubComfy(
        "pod0", sample_png, barrier=barrier, barrier_done=barrier_done, delay=0.02
    )

    lock = threading.Lock()
    leased_clients: list[_ConcurrentStubComfy] = []
    active_leases = 0
    total_leases = 0

    class PoolProvider:
        @contextmanager
        def lease(self, proj) -> Iterator[Any]:
            nonlocal active_leases, total_leases
            assert proj is project
            with lock:
                total_leases += 1
                active_leases += 1
                client = _ConcurrentStubComfy(
                    f"pod{total_leases}",
                    sample_png,
                    barrier=barrier,
                    barrier_done=barrier_done,
                    delay=0.02,
                )
                leased_clients.append(client)
            try:
                yield client
            finally:
                with lock:
                    active_leases -= 1

    runner = _make_runner(
        project,
        manifest,
        primary,
        _StrictSerialGate(),
        _StrictSerialGate(),
        max_concurrency=3,
        comfy_provider=PoolProvider(),
    )
    runner.run([AssetKind.STILL])

    assert all(s.still.status is AssetStatus.PASSED for s in manifest.shots)
    # Primary handled at least 1 shot and extra pods handled concurrent shots.
    assert primary.runs >= 1
    assert total_leases >= 2
    assert active_leases == 0  # every leased client was cleanly released
    # Every pod that ran a shot received its own upload of the portrait.
    for client in [primary, *leased_clients]:
        if client.runs > 0:
            assert len(client.uploads) == 1


def test_per_shot_leasing_never_deadlocks_when_concurrency_exceeds_pool_size(
    planned_six, sample_png
):
    """If max_concurrency > pool capacity, waiting workers proceed as soon as a pod finishes."""
    project, manifest = planned_six
    primary = _ConcurrentStubComfy("pod0", sample_png, delay=0.01)
    pod1 = _ConcurrentStubComfy("pod1", sample_png, delay=0.01)
    sem = threading.Semaphore(1)  # only 1 extra pod available while max_concurrency=4
    active_extra = 0
    peak_extra = 0
    lock = threading.Lock()

    class BoundedPoolProvider:
        @contextmanager
        def lease(self, proj) -> Iterator[Any]:
            nonlocal active_extra, peak_extra
            sem.acquire()
            with lock:
                active_extra += 1
                peak_extra = max(peak_extra, active_extra)
            try:
                yield pod1
            finally:
                with lock:
                    active_extra -= 1
                sem.release()

    runner = _make_runner(
        project,
        manifest,
        primary,
        _StrictSerialGate(),
        _StrictSerialGate(),
        max_concurrency=4,
        comfy_provider=BoundedPoolProvider(),
    )
    runner.run([AssetKind.STILL])

    assert all(s.still.status is AssetStatus.PASSED for s in manifest.shots)
    assert primary.runs + pod1.runs == 6
    assert peak_extra <= 1
    assert active_extra == 0
    # Because pod1 is the same live client object reused across leases, WeakKeyDictionary
    # keeps its upload cache warm and uploads the portrait to pod1 at most once.
    if pod1.runs > 0:
        assert len(pod1.uploads) == 1


@needs_ffmpeg
def test_parallel_waves_respect_continuity_chain_dependencies(planned_six, sample_png, sample_mp4):
    """Wave 0 shots run concurrently; chained Wave 1 shots start only after Wave 0 finishes."""
    project, manifest = planned_six
    # s001 (CUT) -> s002 (CONTINUE from s001)
    # s003 (CUT) -> s004 (CONTINUE from s003)
    manifest.shots[0].continuity = ContinuityMode.CUT
    manifest.shots[1].continuity = ContinuityMode.CONTINUE
    manifest.shots[2].continuity = ContinuityMode.CUT
    manifest.shots[3].continuity = ContinuityMode.CONTINUE

    events_lock = threading.Lock()
    wave0_barrier = threading.Barrier(2, timeout=5.0)
    wave0_still_done: set[str] = set()
    wave0_video_done: set[str] = set()
    order_violations: list[str] = []

    class TrackingComfy(_ConcurrentStubComfy):
        def run(self, graph: dict[str, Any], output: OutputSpec, destination: Path) -> Path:
            shot_id = destination.name.split("_")[0]
            is_video = ".mp4" in output.extensions
            if not is_video and shot_id in {"s001", "s003"}:
                wave0_barrier.wait()
            with events_lock:
                if shot_id in {"s002", "s004"} and not {"s001", "s003"} <= wave0_video_done:
                    order_violations.append(f"{shot_id} started before wave 0 videos finished")
            result = super().run(graph, output, destination)
            with events_lock:
                if shot_id in {"s001", "s003"}:
                    if is_video:
                        wave0_video_done.add(shot_id)
                    else:
                        wave0_still_done.add(shot_id)
            return result

    comfy = TrackingComfy("serverless", sample_png, sample_mp4, delay=0.01)
    runner = _make_runner(
        project,
        manifest,
        comfy,
        _StrictSerialGate(),
        _StrictSerialGate(),
        max_concurrency=4,
    )
    runner.run([AssetKind.STILL, AssetKind.VIDEO], shot_ids=["s001", "s002", "s003", "s004"])

    assert not order_violations
    assert wave0_still_done == {"s001", "s003"}
    assert wave0_video_done == {"s001", "s003"}
    for sid in ("s001", "s002", "s003", "s004"):
        assert manifest.shot(sid).still.status is AssetStatus.PASSED
        assert manifest.shot(sid).video.status is AssetStatus.PASSED


def test_parallel_budget_stops_cleanly_and_avoids_extra_leases(planned_six, sample_png):
    """RunBudget enforces exact cap under concurrency and saves completed in-flight shots."""
    project, manifest = planned_six
    primary = _ConcurrentStubComfy("pod0", sample_png, delay=0.03)
    lease_calls = 0
    lock = threading.Lock()

    class CountingProvider:
        @contextmanager
        def lease(self, proj) -> Iterator[Any]:
            nonlocal lease_calls
            with lock:
                lease_calls += 1
            yield _ConcurrentStubComfy(f"pod{lease_calls}", sample_png, delay=0.03)

    runner = _make_runner(
        project,
        manifest,
        primary,
        _StrictSerialGate(),
        _StrictSerialGate(),
        budget=2,
        max_concurrency=4,
        comfy_provider=CountingProvider(),
    )
    with pytest.raises(BudgetExceeded):
        runner.run([AssetKind.STILL])

    saved = project.load_manifest()
    passed = [s for s in saved.shots if s.still.status is AssetStatus.PASSED]
    assert len(passed) == 2
    assert saved.total_generations == 2
    # At most 1 extra lease needed for 2 budgeted shots (1 uses primary, 1 uses extra).
    assert lease_calls <= 1


def test_failed_provider_lease_refunds_budget_and_aborts_wave(planned_six, sample_png):
    """An endpoint lease failure does not burn ghost budget and propagates cleanly."""
    project, manifest = planned_six
    primary = _ConcurrentStubComfy("pod0", sample_png, delay=0.05)

    class FailingProvider:
        @contextmanager
        def lease(self, proj) -> Iterator[Any]:
            raise ComfyConnectionError("RunPod endpoint unavailable")
            yield  # pragma: no cover

    budget = RunBudget(10)
    gates = QualityGates(image=_StrictSerialGate(), video=_StrictSerialGate(), face_analyzer=None)
    policy = RetryPolicy(project.config.retry, project.config.seed, llm=None)
    templates = load_templates(project, ["t2i", "still", "lipsync", "i2v"])
    runner = ShotRunner(
        project,
        manifest,
        primary,
        gates,
        policy,
        templates,
        budget,
        max_concurrency=3,
        comfy_provider=FailingProvider(),
    )

    with pytest.raises(ComfyConnectionError, match="RunPod endpoint unavailable"):
        runner.run([AssetKind.STILL])

    # Budget reflects only shots that actually started on primary, never failed leases.
    assert budget.used == primary.runs
    assert primary.runs <= 1


def test_parallel_render_cache_is_thread_safe(planned_six, sample_png):
    """Concurrent workers safely read and populate .comfy_cache without corruption."""
    project, manifest = planned_six
    project.config.comfy.cache_enabled = True
    comfy = _ConcurrentStubComfy("serverless", sample_png, delay=0.01)

    runner = _make_runner(
        project,
        manifest,
        comfy,
        _StrictSerialGate(),
        _StrictSerialGate(),
        max_concurrency=4,
    )
    runner.run([AssetKind.STILL])
    assert comfy.runs == 6
    assert (project.root / ".comfy_cache").is_dir()

    # Reset shot states (including attempt history) so seeds match attempt 1 and hit .comfy_cache.
    for shot in manifest.shots:
        shot.still.attempts.clear()
        shot.still.reset(project.config.retry.max_attempts)
    comfy2 = _ConcurrentStubComfy("serverless", sample_png, delay=0.01)
    runner2 = _make_runner(
        project,
        manifest,
        comfy2,
        _StrictSerialGate(),
        _StrictSerialGate(),
        max_concurrency=4,
    )
    runner2.run([AssetKind.STILL])
    assert comfy2.runs == 0
    assert all(s.still.status is AssetStatus.PASSED for s in manifest.shots)


def test_parallel_smart_cutaway_rescue_respects_two_cutaway_cap_and_reuses_client(
    planned_six, sample_png
):
    """Adjacent parallel lip-sync shots cannot both convert into a 3-cutaway run."""
    project, manifest = planned_six
    project.config.retry.smart_cutaway_rescue = True
    project.config.retry.smart_cutaway_max_correlation = 0.25
    # Layout: [s001: BROLL, s002: LIPSYNC, s003: LIPSYNC, s004: LIPSYNC]
    # Turning s002 into BROLL creates a 2-cutaway run (s001, s002); s003 MUST NOT also convert!
    manifest.shots[0].type = ShotType.BROLL
    for idx in (1, 2, 3):
        manifest.shots[idx].type = ShotType.LIPSYNC

    # Pre-populate passed stills for s001..s004.
    comfy = _ConcurrentStubComfy("pod0", sample_png, delay=0.01)
    runner_stills = _make_runner(
        project, manifest, comfy, _StrictSerialGate(), _StrictSerialGate(), max_concurrency=4
    )
    runner_stills.run([AssetKind.STILL], shot_ids=["s001", "s002", "s003", "s004"])

    # Now run video phase with critically low lip-sync correlation so s002/s003/s004 all try rescue.
    lease_lock = threading.Lock()
    active_leases = 0
    peak_leases = 0

    class TrackingProvider:
        @contextmanager
        def lease(self, proj) -> Iterator[Any]:
            nonlocal active_leases, peak_leases
            with lease_lock:
                active_leases += 1
                peak_leases = max(peak_leases, active_leases)
            try:
                yield _ConcurrentStubComfy(f"extra-{threading.get_ident()}", sample_png, delay=0.01)
            finally:
                with lease_lock:
                    active_leases -= 1

    video_comfy = _ConcurrentStubComfy("pod0-v", sample_png, delay=0.01)
    runner_vid = _make_runner(
        project,
        manifest,
        video_comfy,
        _StrictSerialGate(),
        _StrictSerialGate(pass_all=True, correlation=0.05),
        max_concurrency=3,
        comfy_provider=TrackingProvider(),
    )
    runner_vid.run([AssetKind.VIDEO], shot_ids=["s002", "s003", "s004"])

    # Invariant: no 3 consecutive BROLL shots may exist after concurrent smart cutaway rescue.
    max_broll_run = 0
    cur_run = 0
    for s in manifest.shots[:4]:
        if s.type is ShotType.BROLL:
            cur_run += 1
            max_broll_run = max(max_broll_run, cur_run)
        else:
            cur_run = 0
    assert max_broll_run <= 2
    assert active_leases == 0
    # Recursive cutaway still generation reused each worker's client (never exceeded workers-1).
    assert peak_leases <= 2


def test_sequential_and_single_pending_shot_never_lease_extra_clients(planned_six, sample_png):
    """max_concurrency=1 or a wave with only 1 pending shot never calls comfy_provider.lease."""
    project, manifest = planned_six

    class ForbiddenProvider:
        @contextmanager
        def lease(self, proj) -> Iterator[Any]:
            raise AssertionError("Extra lease must not be requested")
            yield  # pragma: no cover

    # Case 1: max_concurrency=1 with multiple pending shots.
    comfy = _ConcurrentStubComfy("pod0", sample_png)
    runner = _make_runner(
        project,
        manifest,
        comfy,
        _StrictSerialGate(),
        _StrictSerialGate(),
        max_concurrency=1,
        comfy_provider=ForbiddenProvider(),
    )
    runner.run([AssetKind.STILL], shot_ids=["s001", "s002", "s003", "s004", "s005"])
    assert comfy.runs == 5

    # Case 2: max_concurrency=4, but 5 of 6 shots are already PASSED (only s006 is PENDING).
    runner_resume = _make_runner(
        project,
        manifest,
        comfy,
        _StrictSerialGate(),
        _StrictSerialGate(),
        max_concurrency=4,
        comfy_provider=ForbiddenProvider(),
    )
    runner_resume.run([AssetKind.STILL])
    assert comfy.runs == 6
    assert all(s.still.status is AssetStatus.PASSED for s in manifest.shots)


def test_concurrency_validation_and_api_generate_integration(planned_six, sample_png, monkeypatch):
    """Config/API validate max_concurrency bounds and api.generate wires parallel workers."""
    import sys

    project, manifest = planned_six
    with pytest.raises(ValidationError):
        ComfyConfig(max_concurrency=0)
    with pytest.raises(ValidationError):
        ComfyConfig(max_concurrency=65)
    with pytest.raises(ConfigError):
        _make_runner(
            project,
            manifest,
            _ConcurrentStubComfy("c", sample_png),
            _StrictSerialGate(),
            _StrictSerialGate(),
            max_concurrency=True,  # type: ignore[arg-type]
        )

    monkeypatch.setitem(sys.modules, "mediapipe", None)
    project.config.llm.enabled = False
    project.config.qc.on_missing_dependency = "skip"
    project.save_manifest(manifest)

    barrier = threading.Barrier(3, timeout=5.0)
    barrier_done = threading.Event()
    leased = 0
    lock = threading.Lock()

    class ServerlessProvider:
        @contextmanager
        def lease(self, proj) -> Iterator[Any]:
            nonlocal leased
            with lock:
                leased += 1
            yield _ConcurrentStubComfy(
                f"worker-{leased}",
                sample_png,
                barrier=barrier,
                barrier_done=barrier_done,
                delay=0.01,
            )

    with pytest.raises(ConfigError):
        api.generate(
            project,
            [AssetKind.STILL],
            skip_preflight=True,
            comfy_provider=ServerlessProvider(),
            max_concurrency=0,
        )

    result = api.generate(
        project,
        [AssetKind.STILL],
        skip_preflight=True,
        comfy_provider=ServerlessProvider(),
        max_concurrency=3,
    )
    assert all(s.still.status is AssetStatus.PASSED for s in result.manifest.shots)
    assert leased >= 3  # 1 primary lease in api.generate + >=2 concurrent worker leases


def test_shared_comfy_client_vram_floor_check_runs_once_under_concurrency(tmp_path, sample_png):
    """ComfyClient._vram_lock serializes initial VRAM check (including /free retry) cleanly."""
    stats_calls = 0
    free_calls = 0
    lock = threading.Lock()
    fake = FakeComfy(png=sample_png)
    base_handler = fake._handle

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal stats_calls, free_calls
        if request.url.path == "/system_stats":
            with lock:
                stats_calls += 1
                # First probe is below floor so check_vram_floor() re-entrantly calls free_memory().
                vram_gb = 4 if stats_calls == 1 else 24
            time.sleep(0.01)
            return httpx.Response(200, json={"devices": [{"vram_free": vram_gb * 1024**3}]})
        if request.url.path == "/free":
            with lock:
                free_calls += 1
            return httpx.Response(200, json={})
        return base_handler(request)

    cfg = ComfyConfig(base_url="http://comfy.test", min_free_vram_gb=12.0, poll_interval=0.0)
    client = ComfyClient(cfg, transport=httpx.MockTransport(handler), sleep=lambda _: None)
    spec = OutputSpec(node="9", extensions=[".png"])
    graph = {
        "1": {"inputs": {"text": "hi"}},
        "2": {"inputs": {"seed": 1, "length": 0}},
        "9": {"inputs": {}},
    }

    start_barrier = threading.Barrier(4, timeout=5.0)

    def run_one(idx: int) -> Path:
        start_barrier.wait()
        return client.run(graph, spec, tmp_path / f"out_{idx}.png")

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=4) as pool:
        paths = list(pool.map(run_one, range(4)))

    assert len(paths) == 4 and all(p.is_file() for p in paths)
    # One initial low probe + one post-/free settle probe across all 4 concurrent callers.
    assert free_calls == 1
    assert stats_calls == 2
    client.close()


def test_non_weakrefable_clients_avoid_aba_upload_cache_collision(tmp_path: Path) -> None:
    """Freshly leased slotted clients must not inherit uploads from GC'd clients at reused ids."""
    import gc
    import weakref
    from dataclasses import dataclass

    from mvdirector.generation.runner import _UploadTable

    @dataclass(slots=True)
    class _SlottedClient:
        pod_name: str
        uploaded: list[str]

        def upload(self, path: Path) -> str:
            self.uploaded.append(path.name)
            return f"{self.pod_name}/{path.name}"

    runner = ShotRunner.__new__(ShotRunner)
    runner._comfy = object()  # type: ignore[assignment]
    runner._primary_upload_table = _UploadTable()
    runner._upload_tables = weakref.WeakKeyDictionary()
    runner._unweakref_upload_tables = {}
    runner._upload_lock = threading.Lock()
    runner._cache = None
    runner._file_hashes = {}

    asset = tmp_path / "ref.png"
    asset.write_bytes(b"png")

    for i in range(40):
        client = _SlottedClient(f"pod-{i}", [])
        remote = runner._upload(asset, client=client)  # type: ignore[arg-type]
        assert client.uploaded == ["ref.png"]
        assert remote == f"pod-{i}/ref.png"
        del client
        gc.collect()


def test_policy_prepare_serialized_under_qc_lock_and_refunds_budget_on_error(
    planned_six, sample_png
) -> None:
    """RetryPolicy.prepare runs under _qc_lock and refunds RunBudget if prepare raises."""
    from mvdirector.generation.runner import load_templates
    from mvdirector.models import ReferenceSet

    proj, manifest = planned_six
    ref_file = proj.path("assets/reference/base_front.png")
    ref_file.parent.mkdir(parents=True, exist_ok=True)
    ref_file.write_bytes(sample_png)
    manifest.character.references["base"] = ReferenceSet(
        sheet="assets/reference/base_front.png",
        panels={"front": "assets/reference/base_front.png"},
    )
    lock_held_during_prepare: list[bool] = []

    class _CheckingPolicy(RetryPolicy):
        def prepare(self, shot, kind, *, can_repair: bool = False):  # type: ignore[override]
            lock_held_during_prepare.append(runner._qc_lock.locked())
            raise RuntimeError("boom during prepare")

    fake = FakeComfy(png=sample_png)
    comfy = ComfyClient(
        ComfyConfig(base_url="http://comfy.test", poll_interval=0.0),
        transport=httpx.MockTransport(fake._handle),
        sleep=lambda _: None,
    )
    budget = RunBudget(limit=5)
    runner = ShotRunner(
        project=proj,
        manifest=manifest,
        comfy=comfy,
        gates=QualityGates(image=None, video=None, face_analyzer=None),  # type: ignore[arg-type]
        policy=_CheckingPolicy(proj.config.retry, proj.config.seed, llm=None),
        templates=load_templates(proj, ["t2i", "still", "i2v", "lipsync"]),
        budget=budget,
        max_concurrency=2,
    )
    try:
        with pytest.raises(RuntimeError, match="boom during prepare"):
            runner.run([AssetKind.STILL], shot_ids=[manifest.shots[0].id])
    finally:
        comfy.close()

    assert lock_held_during_prepare == [True]
    assert budget.used == 0
