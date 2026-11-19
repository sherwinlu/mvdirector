# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import pytest
from typer.testing import CliRunner

from mvdirector import api
from mvdirector.cli import app
from mvdirector.models import AssetKind, AssetStatus, Attempt, RetryStrategy
from mvdirector.planning.planner import build_manifest


@pytest.fixture
def manifest(project, analysis):
    project.config.video.max_clip_seconds = 4.0
    manifest = build_manifest(project.config, analysis, "lyrics", "style", llm=None)
    project.save_manifest(manifest)
    return manifest


def _take(project, shot, kind, number, *, selected=False):
    """Record an attempt with a real file behind it."""
    path = project.asset_path(shot, kind, number)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * 100)
    state = shot.asset(kind)
    state.attempts.append(
        Attempt(
            number=number,
            strategy=RetryStrategy.INITIAL,
            seed=number,
            prompt="p",
            path=project.rel(path),
        )
    )
    if selected:
        state.status, state.selected_attempt = AssetStatus.PASSED, number
    return path


def _save(project, manifest):
    project.save_manifest(manifest)


def test_only_the_takes_beaten_by_a_selected_attempt_are_listed(project, manifest):
    shot = manifest.shots[0]
    bad = _take(project, shot, AssetKind.VIDEO, 1)
    good = _take(project, shot, AssetKind.VIDEO, 2, selected=True)
    _save(project, manifest)
    plan = api.clean(project)
    assert plan.superseded == [bad] and good.is_file() and plan.size_bytes == 100


def test_unfinished_assets_keep_every_take(project, manifest):
    shot = manifest.shots[0]
    _take(project, shot, AssetKind.VIDEO, 1)
    _take(project, shot, AssetKind.VIDEO, 2)
    shot.video.status = AssetStatus.EXHAUSTED
    _save(project, manifest)
    assert api.clean(project).files == []


def test_generations_the_manifest_never_mentions_are_orphans(project, manifest):
    stray = project.path("assets/stills/s999_a01.png")
    stray.parent.mkdir(parents=True, exist_ok=True)
    stray.write_bytes(b"x")
    reference = project.path("assets/reference/ned.png")
    reference.parent.mkdir(parents=True, exist_ok=True)
    reference.write_bytes(b"x")
    notes = project.path("assets/videos/notes.txt")
    notes.parent.mkdir(parents=True, exist_ok=True)
    notes.write_text("keep me")
    plan = api.clean(project)
    assert plan.orphaned == [stray]  # not the reference sheet, not an unrelated file


def test_deleting_removes_files_and_clears_paths_but_keeps_the_record(project, manifest):
    shot = manifest.shots[0]
    bad = _take(project, shot, AssetKind.STILL, 1)
    good = _take(project, shot, AssetKind.STILL, 2, selected=True)
    _save(project, manifest)
    api.clean(project, delete=True)
    assert not bad.exists() and good.is_file()
    attempts = project.load_manifest().shots[0].still.attempts
    assert [a.path for a in attempts] == [None, project.rel(good)]
    assert [a.number for a in attempts] == [1, 2]
    assert api.clean(project).files == []  # nothing left to do


def test_the_cli_lists_by_default_and_deletes_with_yes(project, manifest):
    shot = manifest.shots[0]
    bad = _take(project, shot, AssetKind.VIDEO, 1)
    _take(project, shot, AssetKind.VIDEO, 2, selected=True)
    _save(project, manifest)
    runner = CliRunner()
    listed = runner.invoke(app, ["clean", "-p", str(project.root)])
    assert "Would delete 1 superseded" in listed.output and bad.is_file()
    done = runner.invoke(app, ["clean", "--yes", "-p", str(project.root)])
    assert "Deleted 1 superseded" in done.output and not bad.exists()


def test_clean_is_exported_and_pulls_workspace_storage(project, manifest):
    from mvdirector.project import Project
    from tests.test_store import RecordingStorage

    assert "clean" in api.__all__ and "CleanupPlan" in api.__all__
    storage = RecordingStorage()
    custom = Project(project.root, project.config, storage=storage)
    api.clean(custom, delete=True)
    assert storage.events == ["pull"]  # nothing to delete: no unnecessary manifest save/push
