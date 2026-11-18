# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""The StateStore / AssetStorage seams: a Project works against non-file implementations."""

from __future__ import annotations

from contextlib import contextmanager

import pytest

from mvdirector import api
from mvdirector.errors import MVDError, ProjectLocked
from mvdirector.models import PropDesign, SetDesign
from mvdirector.planning.planner import build_manifest
from mvdirector.project import Project
from mvdirector.store import AssetStorage, LocalAssetStorage, LocalStateStore, StateStore
from tests.conftest import MemoryState, RecordingStorage


def test_local_implementations_satisfy_the_protocols(tmp_path):
    assert isinstance(LocalStateStore(tmp_path), StateStore)
    assert isinstance(LocalAssetStorage(), AssetStorage)
    assert isinstance(MemoryState(), StateStore)


def test_project_uses_injected_state_and_publishes_before_saving(project, analysis):
    state, storage = MemoryState(), RecordingStorage()
    custom = Project(project.root, project.config, state=state, storage=storage)
    manifest = build_manifest(project.config, analysis, "lyrics", "style", llm=None)

    custom.save_manifest(manifest)

    assert state.has_manifest()
    assert storage.events == ["push"]
    assert not (custom.root / "manifest.json").exists()  # nothing written to the workspace


def test_lock_is_exclusive_through_the_state_store(project):
    state = MemoryState()
    custom = Project(project.root, project.config, state=state)
    with custom.run_lock():
        assert custom.is_locked
        with pytest.raises(ProjectLocked), custom.run_lock():
            pass
    assert not custom.is_locked


def test_api_accepts_a_prebuilt_project(project, analysis):
    state = MemoryState()
    state.save_manifest(build_manifest(project.config, analysis, "lyrics", "style", llm=None))
    custom = Project(project.root, project.config, state=state)

    api.approve(custom, [])  # no shots: just proves edit_manifest goes through the store

    assert state.manifest is not None


def test_api_leases_comfy_from_the_injected_provider(project):
    leased = []

    class Provider:
        @contextmanager
        def lease(self, proj):
            leased.append(proj)
            raise MVDError("stop after lease")
            yield  # makes this a generator, as @contextmanager requires

    checks = api.validate_project(project, comfy_provider=Provider())

    assert leased == [project]
    assert any("stop after lease" in (c.problem or "") for c in checks)


def test_plan_checks_for_an_existing_manifest_while_holding_the_lock(project, analysis):
    seen = []

    class Spy(MemoryState):
        def has_manifest(self) -> bool:
            seen.append(self.held)
            return super().has_manifest()

    state = Spy()
    state.save_analysis(analysis)
    custom = Project(project.root, project.config, state=state)

    api.plan(custom)
    with pytest.raises(MVDError, match="already exists"):
        api.plan(custom)

    assert seen == [True, True]


def test_save_manifest_stamps_updated_at_for_any_store(project, analysis):
    state = MemoryState()
    custom = Project(project.root, project.config, state=state)
    manifest = build_manifest(project.config, analysis, "lyrics", "style", llm=None)
    before = manifest.updated_at

    custom.save_manifest(manifest)

    assert state.manifest is not None and state.manifest.updated_at >= before


def test_review_mode_override_does_not_leak_into_the_callers_project(project):
    from mvdirector.config import ReviewMode

    original = project.config.review.mode
    other = next(m for m in ReviewMode if m != original)

    overridden = api.open_project(project, other)

    assert overridden.config.review.mode == other
    assert project.config.review.mode == original
    assert overridden.state is project.state


def test_state_only_stages_skip_pull_but_workspace_stages_pull(project, analysis):
    state, storage = MemoryState(), RecordingStorage()
    state.save_manifest(build_manifest(project.config, analysis, "lyrics", "style", llm=None))
    custom = Project(project.root, project.config, state=state, storage=storage)

    api.approve(custom, [])
    api.load_status(custom)
    assert "pull" not in storage.events

    api.open_project(custom)
    assert storage.events.count("pull") == 1


def test_brief_publishes_the_files_it_writes(project, monkeypatch):
    from mvdirector.planning.brief import BriefReply

    storage = RecordingStorage()
    custom = Project(project.root, project.config, storage=storage)
    reply = BriefReply(lyrics="[Verse]\nla la", suno_style="synthwave", visual_direction="neon")
    monkeypatch.setattr(api, "build_llm", lambda _cfg: None)
    monkeypatch.setattr(api, "generate_brief", lambda *a, **k: reply)

    api.brief(custom, "an idea", overwrite=True)

    assert storage.events.count("push") >= 1


def test_comfy_uploads_are_namespaced_by_the_state_stores_project_id(project):
    custom = Project(project.root, project.config, state=MemoryState())
    other = Project(project.root, project.config, state=MemoryState())

    assert custom.comfy_subfolder == other.comfy_subfolder  # same db key, any workspace
    assert not (custom.root / ".mvd_id").exists()  # nothing written into the workspace


def test_local_project_id_is_persisted_in_the_workspace(project):
    first = project.state.project_id
    assert first and (project.root / ".mvd_id").read_text().strip() == first
    assert project.state.project_id == first


def _generation_project(project, analysis, monkeypatch, state=None, storage=None):
    import sys

    monkeypatch.setitem(sys.modules, "mediapipe", None)
    project.config.llm.enabled = False
    project.config.qc.on_missing_dependency = "skip"
    custom = Project(project.root, project.config, state=state or MemoryState(), storage=storage)
    custom.save_manifest(build_manifest(project.config, analysis, "lyrics", "style", llm=None))
    return custom


def test_generate_streams_preflight_and_qc_events_then_releases_lock_on_failure(
    project, analysis, monkeypatch
):
    custom = _generation_project(project, analysis, monkeypatch)
    events, trace = [], []

    class Provider:
        @contextmanager
        def lease(self, proj):
            trace.append("lease")
            try:
                raise MVDError("endpoint down")
                yield
            finally:
                trace.append("release")

    from mvdirector.models import AssetKind

    with pytest.raises(MVDError, match="endpoint down"):
        api.generate(
            custom,
            [AssetKind.STILL],
            skip_preflight=True,
            observer=events.append,
            comfy_provider=Provider(),
        )

    kinds = [e.kind for e in events]
    assert "qc_summary" in kinds and kinds.index("qc_summary") == len(
        [k for k in kinds if k == "warning"]
    )
    summary = next(e for e in events if e.kind == "qc_summary")
    assert "dimensions" in summary.active_checks and summary.skipped_checks
    assert trace == ["lease", "release"]
    assert not custom.is_locked


def test_generate_reloads_the_manifest_under_the_lock(project, analysis, monkeypatch):
    seen = []

    class Spy(MemoryState):
        def load_manifest(self):
            seen.append(self.held)
            return super().load_manifest()

    custom = _generation_project(project, analysis, monkeypatch, state=Spy())
    seen.clear()

    class StopProvider:
        @contextmanager
        def lease(self, proj):
            raise MVDError("stop")
            yield

    from mvdirector.models import AssetKind

    # Preflight loads once without the lock; the run reloads once while holding it.
    with pytest.raises(MVDError, match="stop"):
        api.generate(custom, [AssetKind.STILL], skip_preflight=True, comfy_provider=StopProvider())
    assert seen == [False]  # lease failed before the locked reload ...

    class FakeComfyClient:
        def ping(self):
            raise MVDError("ping")

    class OkProvider:
        @contextmanager
        def lease(self, proj):
            yield FakeComfyClient()

    seen.clear()
    with pytest.raises(MVDError, match="ping"):
        api.generate(custom, [AssetKind.STILL], skip_preflight=True, comfy_provider=OkProvider())
    assert seen == [False, True]  # ... and with a lease the second load happens under the lock


def test_a_failed_save_does_not_hide_the_runs_own_error(project, analysis, monkeypatch, caplog):
    class BrokenStorage(RecordingStorage):
        def push(self) -> None:
            raise OSError("bucket unreachable")

    custom = _generation_project(project, analysis, monkeypatch)
    manifest = custom.load_manifest()
    custom.storage = BrokenStorage()

    with pytest.raises(MVDError, match="the real failure"):
        try:
            raise MVDError("the real failure")
        finally:
            api._save_after_run(custom, manifest)
    assert "Could not save the manifest" in caplog.text

    with pytest.raises(OSError, match="bucket unreachable"):  # no run error: surface the save
        api._save_after_run(custom, manifest)


def test_candidate_stages_publish_their_files_even_without_a_manifest_save(
    project, analysis, sample_png
):
    from mvdirector.comfy.client import ComfyClient
    from tests.conftest import FakeComfy

    fake = FakeComfy(png=sample_png)
    storage = RecordingStorage()
    state = MemoryState()
    manifest = build_manifest(project.config, analysis, "lyrics", "style", llm=None)
    manifest.visual_bible.settings = [SetDesign(name="Neon City", description="skyline")]
    manifest.visual_bible.props = [PropDesign(id="guitar", name="Guitar", description="red guitar")]
    state.save_manifest(manifest)
    custom = Project(project.root, project.config, state=state, storage=storage)

    class Provider:
        def lease(self, proj):
            return ComfyClient(proj.config.comfy, transport=fake.transport(), sleep=lambda _: None)

    api.generate_plates(custom, comfy_provider=Provider())
    api.generate_props(custom, comfy_provider=Provider())

    assert storage.events.count("push") == 2  # no manifest save happened to trigger them
