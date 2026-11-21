# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""End-to-end FFmpeg tests for Long-Form Mix Mode assembly, HUD/spectrum, and segment caching."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import soundfile as sf
import yaml

from mvdirector import api, media, mix_assemble
from mvdirector.config import ProjectConfig, ReviewMode
from mvdirector.models import (
    AssetKind,
    AssetStatus,
    Attempt,
    ProjectMode,
    QCReport,
    RetryStrategy,
    Shot,
)
from mvdirector.project import Project
from tests.conftest import SR, click_track, needs_ffmpeg, write_workflows


def _mark_passed(
    project: Project, shot: Shot, kind: AssetKind, payload: bytes, attempt_no: int = 1
) -> None:
    dst = project.asset_path(shot, kind, attempt_no)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(payload)
    attempt = Attempt(
        number=attempt_no,
        strategy=RetryStrategy.INITIAL,
        seed=attempt_no,
        prompt="ambient scene",
        path=project.rel(dst),
        qc=QCReport(passed=True, score=0.92),
    )
    state = shot.asset(kind)
    state.attempts.append(attempt)
    state.status = AssetStatus.PASSED
    state.selected_attempt = attempt_no


def _setup_mix_for_assembly(
    tmp_path: Path,
    *,
    visual_source: str = "clips",
) -> Project:
    root = tmp_path / f"mix_{visual_source}"
    (root / "inputs").mkdir(parents=True)
    sf.write(root / "inputs" / "mix.wav", click_track(seconds=18.0, bpm=120.0), SR)
    # Include tricky punctuation in track titles (quotes, colons, %, brackets) that break drawtext!
    (root / "inputs" / "tracklist.json").write_text(
        json.dumps(
            {
                "title": "Test Mix",
                "artist": "DJ Test",
                "tracks": [
                    {"start": "00:00", "title": "Opening: Don't Stop [100%]", "bpm": 120.0},
                    {"start": "00:06", "title": "Middle 'VIP' — Cut #2", "bpm": 124.0},
                    {"start": "00:12", "title": "Final Drift", "bpm": 118.0},
                ],
            }
        ),
        encoding="utf-8",
    )
    write_workflows(root / "workflows", {"t2i": ".png", "i2v": ".mp4"})

    cfg = ProjectConfig(title="Neon Mix", mode=ProjectMode.MIX)
    cfg.inputs.tracklist = "inputs/tracklist.json"
    cfg.llm.enabled = False
    cfg.review.mode = ReviewMode.STRICT
    cfg.video.width = 320
    cfg.video.height = 180
    cfg.video.fps = 12
    cfg.mix.visual_source = visual_source  # type: ignore[assignment]
    cfg.mix.clips_per_segment = 2
    cfg.mix.clip_seconds = 4.0
    cfg.mix.xfade_seconds = 0.75
    cfg.mix.chapter_fade_seconds = 0.3
    cfg.mix.beat_pulse = 0.4
    cfg.mix.hud.enabled = True
    cfg.mix.hud.watermark = "MVD LIVE"
    cfg.mix.hud.show_spectrum = True
    cfg.comfy.workflows = {"t2i": "t2i.bindings.json", "i2v": "i2v.bindings.json"}
    (root / "project.yaml").write_text(yaml.safe_dump(cfg.model_dump(mode="json")))
    return Project.open(root)


@needs_ffmpeg
def test_mix_assemble_caches_segments_splices_single_chapter_and_writes_chapters(
    tmp_path: Path, sample_png: bytes, sample_mp4: bytes
) -> None:
    project = _setup_mix_for_assembly(tmp_path, visual_source="clips")
    api.analyze(project)
    _, manifest = api.plan(project)
    for shot in manifest.shots:
        _mark_passed(project, shot, AssetKind.STILL, sample_png)
        _mark_passed(project, shot, AssetKind.VIDEO, sample_mp4)
    project.save_manifest(manifest)

    # 1. Initial full render creates all 3 segment caches, MP4 chapters, and .chapters.txt.
    out_mp4 = api.assemble(project)
    assert out_mp4.is_file()
    info = media.probe(out_mp4)
    assert info.width == 320 and info.height == 180 and info.has_audio
    assert abs(info.duration - 18.0) < 0.35

    sidecar = mix_assemble.chapters_sidecar_path(out_mp4)
    assert sidecar.is_file()
    yt_lines = sidecar.read_text(encoding="utf-8").splitlines()
    assert yt_lines == [
        "00:00 DJ Test - Opening: Don't Stop [100%]",
        "00:06 DJ Test - Middle 'VIP' — Cut #2",
        "00:12 DJ Test - Final Drift",
    ]

    # Verify native MP4 [CHAPTER] markers were embedded via ffmetadata.
    probe_ch = json.loads(
        subprocess.run(
            ["ffprobe", "-v", "error", "-print_format", "json", "-show_chapters", str(out_mp4)],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    )
    assert len(probe_ch.get("chapters", [])) == 3
    assert mix_assemble.cached_segment_count(project, manifest) == 3

    seg1 = project.segment_video_path("ch01")
    seg2 = project.segment_video_path("ch02")
    seg3 = project.segment_video_path("ch03")
    mtimes_first = (seg1.stat().st_mtime_ns, seg2.stat().st_mtime_ns, seg3.stat().st_mtime_ns)

    # 2. Second assemble is a 100% cache hit: none of the 3 segment files are re-encoded.
    api.assemble(project)
    assert (
        seg1.stat().st_mtime_ns,
        seg2.stat().st_mtime_ns,
        seg3.stat().st_mtime_ns,
    ) == mtimes_first

    # 3. Regenerating one shot in ch02 invalidates ONLY ch02; ch01 and ch03 are spliced untouched!
    with project.edit_manifest() as edited:
        shot_b = edited.shot("ch02_c01")
        _mark_passed(project, shot_b, AssetKind.VIDEO, sample_mp4 + b"\x00", attempt_no=2)

    assert mix_assemble.cached_segment_count(project, project.load_manifest()) == 2
    api.assemble(project)
    assert seg1.stat().st_mtime_ns == mtimes_first[0]
    assert seg2.stat().st_mtime_ns != mtimes_first[1]
    assert seg3.stat().st_mtime_ns == mtimes_first[2]

    # 4. `mvd clean` keeps active segment caches while removing orphaned chapter caches.
    orphan = project.segments_dir / "ch99.mp4"
    orphan.write_bytes(b"stale")
    plan = api.clean(project, delete=True)
    assert orphan in plan.orphaned and not orphan.exists()
    assert seg1.is_file() and seg2.is_file() and seg3.is_file()


@needs_ffmpeg
def test_mix_assemble_stills_only_animates_hero_stills_with_hud(
    tmp_path: Path, sample_png: bytes
) -> None:
    project = _setup_mix_for_assembly(tmp_path, visual_source="stills")
    api.analyze(project)
    _, manifest = api.plan(project)
    for shot in manifest.shots:
        _mark_passed(project, shot, AssetKind.STILL, sample_png)
    project.save_manifest(manifest)

    out_mp4 = api.assemble(project)
    info = media.probe(out_mp4)
    assert info.width == 320 and info.height == 180 and info.has_audio
    assert abs(info.duration - 18.0) < 0.35
    assert mix_assemble.cached_segment_count(project, manifest) == 3


@needs_ffmpeg
def test_mix_assemble_force_handles_single_clip_and_empty_placeholder_chapters(
    tmp_path: Path, sample_mp4: bytes
) -> None:
    project = _setup_mix_for_assembly(tmp_path, visual_source="clips")
    project.config.mix.clips_per_segment = 1
    api.analyze(project)
    _, manifest = api.plan(project)
    # Populate only ch01's single clip; leave ch02 and ch03 completely empty to test --force!
    _mark_passed(project, manifest.shot("ch01_c01"), AssetKind.VIDEO, sample_mp4)
    project.save_manifest(manifest)

    out_mp4 = api.assemble(project, force=True)
    info = media.probe(out_mp4)
    assert info.width == 320 and info.height == 180 and info.has_audio
    assert abs(info.duration - 18.0) < 0.35
