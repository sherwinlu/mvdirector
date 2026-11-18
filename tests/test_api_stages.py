# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""api stages run against a custom Project, plus real lock contention between threads."""

from __future__ import annotations

import threading
from contextlib import contextmanager

import pytest

from mvdirector import api
from mvdirector.comfy.client import ComfyClient
from mvdirector.errors import MVDError, ProjectLocked
from mvdirector.models import (
    AssetKind,
    AssetState,
    AssetStatus,
    Attempt,
    QCReport,
    RetryStrategy,
)
from mvdirector.planning.planner import build_manifest
from mvdirector.project import Project
from mvdirector.review import policy as review
from tests.conftest import (
    FakeComfy,
    MemoryState,
    RecordingStorage,
    needs_ffmpeg,
    write_upscale_workflow,
)


def _passed_state(project, shot, kind, data):
    """A passed asset with one attempt whose file is on disk, as a finished run leaves it."""
    path = project.asset_path(shot, kind, 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    attempt = Attempt(
        number=1,
        strategy=RetryStrategy.INITIAL,
        seed=1,
        prompt="p",
        path=project.rel(path),
        qc=QCReport(passed=True, score=0.9),
    )
    return AssetState(
        status=AssetStatus.PASSED, selected_attempt=1, attempts_allowed=4, attempts=[attempt]
    )


def _with_passed_videos(project, manifest, sample_png, sample_mp4, shot_ids=None):
    for shot in manifest.shots:
        if shot_ids is None or shot.id in shot_ids:
            shot.still = _passed_state(project, shot, AssetKind.STILL, sample_png)
            shot.video = _passed_state(project, shot, AssetKind.VIDEO, sample_mp4)


def _custom_project(project, manifest):
    state, storage = MemoryState(), RecordingStorage()
    state.save_manifest(manifest)
    return Project(project.root, project.config, state=state, storage=storage), state, storage


class _LeaseRecorder:
    """A provider that serves a fake ComfyUI and records each lease and release."""

    def __init__(self, fake: FakeComfy) -> None:
        self.fake, self.trace = fake, []

    @contextmanager
    def lease(self, project):
        self.trace.append("lease")
        client = ComfyClient(
            project.config.comfy, transport=self.fake.transport(), sleep=lambda _: None
        )
        try:
            yield client
        finally:
            client.close()
            self.trace.append("release")


@needs_ffmpeg
def test_upscale_through_a_custom_project_and_provider(project, analysis, sample_png, sample_mp4):
    project.config.video.max_clip_seconds = 4.0
    write_upscale_workflow(project, length=25)  # FakeComfy returns an .mp4 for length > 0
    project.config.comfy.workflows["upscale"] = "upscale.bindings.json"
    manifest = build_manifest(project.config, analysis, "lyrics", "style", llm=None)
    _with_passed_videos(project, manifest, sample_png, sample_mp4, shot_ids={"s001"})
    review.approve(manifest, "s001")
    custom, state, storage = _custom_project(project, manifest)
    provider = _LeaseRecorder(FakeComfy(mp4=sample_mp4))

    written = api.upscale(custom, ["s001"], comfy_provider=provider)

    assert [p.name for p in written] == ["s001_a01_upscaled.mp4"]
    assert state.manifest.shot("s001").video.upscaled_path == "assets/videos/s001_a01_upscaled.mp4"
    assert provider.trace == ["lease", "release"]
    assert "push" in storage.events  # the manifest save published the new clip
    assert not custom.is_locked

    assert api.upscale(custom, ["s001"], comfy_provider=provider) == []  # idempotent
    assert len(provider.fake.queued) == 1


def test_upscale_without_the_workflow_fails_before_leasing(project, analysis):
    manifest = build_manifest(project.config, analysis, "lyrics", "style", llm=None)
    custom, _, _ = _custom_project(project, manifest)
    provider = _LeaseRecorder(FakeComfy())

    with pytest.raises(MVDError, match="No 'upscale' workflow"):
        api.upscale(custom, comfy_provider=provider)
    assert provider.trace == []
    assert not custom.is_locked


@needs_ffmpeg
def test_assemble_through_a_custom_project_publishes_the_final_cut(
    project, analysis, sample_png, sample_mp4
):
    from mvdirector import media
    from mvdirector.assemble import NotReadyError

    project.config.video.max_clip_seconds = 4.0
    manifest = build_manifest(project.config, analysis, "lyrics", "style", llm=None)
    _with_passed_videos(project, manifest, sample_png, sample_mp4)
    review.refresh_reviews(manifest, project.config)  # a finished run leaves the queue computed
    queued = [s.id for s in review.pending_review(manifest)]
    # Light mode samples shots for review using a seeded hash, so this set is deterministic. If
    # the default review rates or seed change and nothing is queued, this fails loudly.
    assert queued
    custom, state, storage = _custom_project(project, manifest)

    with pytest.raises(NotReadyError):
        api.assemble(custom)

    api.approve(custom, queued)
    storage.events.clear()
    final = api.assemble(custom)

    assert final == custom.output_dir / "Test_Song.mp4"
    info = media.probe(final)
    assert info.has_audio and abs(info.duration - analysis.duration) < 0.1
    assert storage.events[-1] == "push"  # the cut is not in the manifest, so assemble pushed it


def test_reports_run_against_a_custom_project(project, analysis):
    manifest = build_manifest(project.config, analysis, "lyrics", "style", llm=None)
    custom, _, storage = _custom_project(project, manifest)

    assert "Stills" in api.qc_report(custom, markdown=True)
    assert api.status_counts(custom.load_manifest()) == {}
    with pytest.raises(MVDError, match="No labels"):
        api.calibrate(custom)
    assert "pull" not in storage.events  # reports only read state


def test_a_second_writer_is_refused_while_the_first_holds_the_lock(project, analysis):
    project.save_manifest(build_manifest(project.config, analysis, "lyrics", "style", llm=None))
    holding, release = threading.Event(), threading.Event()

    def hold() -> None:
        with project.run_lock():
            holding.set()
            release.wait(5)

    thread = threading.Thread(target=hold)
    thread.start()
    try:
        assert holding.wait(5)
        with pytest.raises(ProjectLocked):
            api.approve(project, [])
        assert project.is_locked
    finally:
        release.set()
        thread.join(5)

    api.approve(project, [])  # free again, so the same call now succeeds
    assert not project.is_locked


def test_exactly_one_of_many_concurrent_plans_writes_the_manifest(project, analysis):
    project.save_analysis(analysis)
    start = threading.Barrier(6)
    outcomes: list[str] = []
    guard = threading.Lock()

    def attempt() -> None:
        start.wait(5)
        try:
            api.plan(project)
            result = "planned"
        except ProjectLocked:
            result = "locked"
        except MVDError as exc:
            result = "exists" if "already exists" in str(exc) else f"other: {exc}"
        with guard:
            outcomes.append(result)

    threads = [threading.Thread(target=attempt) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)

    assert not any(t.is_alive() for t in threads), "a planning thread never finished"
    assert len(outcomes) == 6, outcomes
    assert outcomes.count("planned") == 1, outcomes
    assert set(outcomes) <= {"planned", "locked", "exists"}, outcomes
    assert project.state.has_manifest() and not project.is_locked
