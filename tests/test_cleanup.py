# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import os
from datetime import datetime, timedelta, timezone

import pytest
from typer.testing import CliRunner

from mvdirector import api, cleanup
from mvdirector.cli import app
from mvdirector.errors import ConfigError
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


def test_retention_days_filters_by_created_at_and_mtime_and_preserves_active_assets(
    project, manifest
):
    now = datetime(2026, 11, 27, 12, 0, 0, tzinfo=timezone.utc)
    old_iso = (now - timedelta(days=10)).isoformat().replace("+00:00", "Z")
    fresh_iso = (now - timedelta(days=2)).isoformat().replace("+00:00", "Z")
    old_ts = (now - timedelta(days=10)).timestamp()
    fresh_ts = (now - timedelta(days=2)).timestamp()

    shot = manifest.shots[0]
    # Superseded attempt 1: old created_at (10d) even though file mtime is fresh -> deleted
    old_bad = _take(project, shot, AssetKind.VIDEO, 1)
    shot.video.attempts[0].created_at = old_iso
    os.utime(old_bad, (fresh_ts, fresh_ts))

    # Superseded attempt 2: fresh created_at (2d) even though file mtime is old -> kept
    fresh_bad = _take(project, shot, AssetKind.VIDEO, 2)
    shot.video.attempts[1].created_at = fresh_iso
    os.utime(fresh_bad, (old_ts, old_ts))

    # Selected attempt 3: very old (365d) -> MUST NEVER be deleted
    very_old_ts = (now - timedelta(days=365)).timestamp()
    active_good = _take(project, shot, AssetKind.VIDEO, 3, selected=True)
    shot.video.attempts[2].created_at = (now - timedelta(days=365)).isoformat()
    os.utime(active_good, (very_old_ts, very_old_ts))

    # Orphaned & cache files: filtered by mtime
    old_orphan = project.path("assets/stills/s999_a01.png")
    old_orphan.parent.mkdir(parents=True, exist_ok=True)
    old_orphan.write_bytes(b"o")
    os.utime(old_orphan, (old_ts, old_ts))

    fresh_orphan = project.path("assets/stills/s998_a01.png")
    fresh_orphan.write_bytes(b"f")
    os.utime(fresh_orphan, (fresh_ts, fresh_ts))

    old_cache = project.root / ".comfy_cache" / "old.bin"
    old_cache.parent.mkdir(parents=True, exist_ok=True)
    old_cache.write_bytes(b"c")
    os.utime(old_cache, (old_ts, old_ts))

    fresh_cache = project.root / ".comfy_cache" / "fresh.bin"
    fresh_cache.write_bytes(b"d")
    os.utime(fresh_cache, (fresh_ts, fresh_ts))

    _save(project, manifest)

    plan_7d = cleanup.plan_cleanup(project, manifest, retention_days=7, now=now)
    assert plan_7d.superseded == [old_bad]
    assert plan_7d.orphaned == [old_orphan]
    assert plan_7d.cache_files == [old_cache]

    # retention_days=0 deletes all superseded/orphaned/cached <= now, still preserving active_good
    plan_0d = cleanup.plan_cleanup(project, manifest, retention_days=0, now=now)
    assert set(plan_0d.superseded) == {old_bad, fresh_bad}
    assert active_good not in plan_0d.files

    for bad_days in (-1, True, False):
        with pytest.raises(ConfigError):
            cleanup.plan_cleanup(project, manifest, retention_days=bad_days)  # type: ignore[arg-type]

    # CLI --retention-days deletes only expired files when --yes is passed
    runner = CliRunner()
    res = runner.invoke(app, ["clean", "--yes", "--retention-days", "7", "-p", str(project.root)])
    assert res.exit_code == 0
    assert not old_bad.exists()
    assert fresh_bad.is_file()
    assert active_good.is_file()
    assert not old_orphan.exists()
    assert fresh_orphan.is_file()


def test_plan_cleanup_skips_escaping_symlink_in_attempt_path(tmp_path, project, manifest) -> None:
    outside_secret = tmp_path / "outside_secret.png"
    outside_secret.write_bytes(b"secret")

    shot = manifest.shots[0]
    escape_link = project.asset_path(shot, AssetKind.STILL, 5)
    escape_link.parent.mkdir(parents=True, exist_ok=True)
    escape_link.symlink_to(outside_secret)

    # 1. Superseded escaping symlink in settled shot
    winner = _take(project, shot, AssetKind.STILL, 1, selected=True)
    shot.still.attempts.append(
        Attempt(
            number=5,
            strategy=RetryStrategy.NEW_SEED,
            seed=5,
            prompt="p",
            path=f"assets/stills/{escape_link.name}",
        )
    )
    # 2. Unsettled escaping symlink in another shot
    shot2 = manifest.shots[1]
    escape_link_2 = project.asset_path(shot2, AssetKind.VIDEO, 1)
    escape_link_2.parent.mkdir(parents=True, exist_ok=True)
    escape_link_2.symlink_to(outside_secret)
    shot2.video.attempts.append(
        Attempt(
            number=1,
            strategy=RetryStrategy.INITIAL,
            seed=1,
            prompt="p",
            path=f"assets/videos/{escape_link_2.name}",
        )
    )
    _save(project, manifest)

    plan = cleanup.plan_cleanup(project, manifest, retention_days=7)
    assert escape_link not in plan.files
    assert escape_link_2 not in plan.files
    assert winner not in plan.files
    cleanup.apply_cleanup(project, manifest, plan)
    assert outside_secret.read_bytes() == b"secret"


def test_symlinked_selected_attempt_still_protects_its_in_project_target(project, manifest):
    """Regression: a selected take stored as a symlink must keep the file it points at.

    Skipping symlinked attempts entirely (to avoid resolving links that escape the project)
    dropped their targets from the keep set, so the orphan scan deleted the selected take's
    real file.
    """
    shot = manifest.shots[0]
    target = project.asset_path(shot, AssetKind.STILL, 7)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"selected pixels")
    link = project.asset_path(shot, AssetKind.STILL, 1)
    link.symlink_to(target.name)
    shot.still.attempts.append(
        Attempt(
            number=1,
            strategy=RetryStrategy.INITIAL,
            seed=1,
            prompt="p",
            path=project.rel(link.parent) + "/" + link.name,
        )
    )
    shot.still.status, shot.still.selected_attempt = AssetStatus.PASSED, 1
    upscaled_target = project.path("assets/videos") / f"{shot.id}_a09_upscaled.mp4"
    upscaled_target.parent.mkdir(parents=True, exist_ok=True)
    upscaled_target.write_bytes(b"upscaled")
    upscaled_link = upscaled_target.with_name(f"{shot.id}_a02_upscaled.mp4")
    upscaled_link.symlink_to(upscaled_target.name)
    shot.video.upscaled_path = f"assets/videos/{upscaled_link.name}"
    _save(project, manifest)

    plan = api.clean(project, delete=True)

    assert target not in plan.files and target.read_bytes() == b"selected pixels"
    assert upscaled_target not in plan.files and upscaled_target.read_bytes() == b"upscaled"
    assert link.is_symlink() and upscaled_link.is_symlink()


@pytest.mark.parametrize("days", [100_000_000, 999_999_999, 10**12])
def test_huge_retention_days_keeps_everything_instead_of_overflowing(project, manifest, days):
    shot = manifest.shots[0]
    _take(project, shot, AssetKind.VIDEO, 1)
    _take(project, shot, AssetKind.VIDEO, 2, selected=True)
    _save(project, manifest)
    plan = cleanup.plan_cleanup(project, manifest, retention_days=days)
    assert plan.files == []
    res = CliRunner().invoke(app, ["clean", "--retention-days", str(days), "-p", str(project.root)])
    assert res.exit_code == 0, res.output
    assert "Would delete 0 superseded, 0 orphaned" in res.output
