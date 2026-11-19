# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Runner behaviour and a full pipeline pass against a fake ComfyUI server."""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from mvdirector import media
from mvdirector.assemble import NotReadyError, assemble, clip_command, frame_counts, plan_clips
from mvdirector.audio.analysis import analyze_project
from mvdirector.comfy.client import ComfyClient
from mvdirector.errors import BudgetExceeded, ComfyConnectionError
from mvdirector.generation import reference
from mvdirector.generation.runner import RunBudget, ShotRunner, load_templates
from mvdirector.models import (
    AssetKind,
    AssetStatus,
    CheckResult,
    CheckStatus,
    Framing,
    QCReport,
    RetryStrategy,
    ShotType,
)
from mvdirector.planning.planner import build_manifest
from mvdirector.qc import lipsync
from mvdirector.qc.factory import QualityGates
from mvdirector.qc.image import ImageQC
from mvdirector.qc.video import VideoQC
from mvdirector.retry import RetryPolicy
from mvdirector.review import policy as review
from tests.conftest import SR, FakeComfy, needs_ffmpeg, vocal_track


class ScriptedGate:
    """Quality gate whose verdicts are fixed in advance, one per call."""

    def __init__(self, verdicts):
        self._verdicts = list(verdicts)
        self.calls = 0
        self.kwargs = {}

    def evaluate(self, *args, **kwargs):
        self.kwargs = kwargs
        passed = self._verdicts[min(self.calls, len(self._verdicts) - 1)]
        self.calls += 1
        checks = (
            []
            if passed
            else [
                CheckResult(
                    name="vision_artifacts",
                    status=CheckStatus.FAILED,
                    hard=True,
                    detail="extra fingers",
                    fix_hint="Keep hands out of frame.",
                )
            ]
        )
        return QCReport(passed=passed, score=0.9 if passed else 0.2, checks=checks)


@pytest.fixture
def planned(project, analysis, tmp_path, sample_png):
    """The project with a manifest and a locked reference portrait."""
    project.config.video.max_clip_seconds = 4.0
    manifest = build_manifest(project.config, analysis, "lyrics", "style", llm=None)
    portrait = tmp_path / "portrait.png"
    portrait.write_bytes(sample_png)
    reference.lock_reference(project, manifest, portrait)
    project.save_manifest(manifest)
    return project, manifest


def _runner(project, manifest, fake, image_gate, video_gate, budget=100, observer=None):
    comfy = ComfyClient(project.config.comfy, transport=fake.transport(), sleep=lambda _: None)
    gates = QualityGates(image=image_gate, video=video_gate, face_analyzer=None)
    policy = RetryPolicy(project.config.retry, project.config.seed, llm=None)
    templates = load_templates(project, ["t2i", "still", "lipsync", "i2v"])
    return ShotRunner(
        project, manifest, comfy, gates, policy, templates, RunBudget(budget), observer=observer
    )


def test_failed_stills_are_regenerated_until_one_passes(planned, sample_png):
    project, manifest = planned
    fake = FakeComfy(png=sample_png)
    runner = _runner(
        project, manifest, fake, ScriptedGate([False, False, True]), ScriptedGate([True])
    )
    runner.run([AssetKind.STILL], shot_ids=["s002"])

    state = manifest.shot("s002").still
    assert state.status is AssetStatus.PASSED and state.selected_attempt == 3
    assert [a.strategy for a in state.attempts] == [
        RetryStrategy.INITIAL,
        RetryStrategy.NEW_SEED,
        RetryStrategy.REWRITE_PROMPT,
    ]
    assert len({a.seed for a in state.attempts}) == 3
    assert "Keep hands out of frame." in state.attempts[2].prompt
    assert all(project.path(a.path).is_file() for a in state.attempts)
    # The reference portrait was uploaded once and wired into every still request.
    assert fake.uploads == ["upload_0"]
    assert all(g["3"]["inputs"]["image"] == "mvdirector/upload_0" for g in fake.queued)
    # Progress was persisted and untouched shots were left alone.
    saved = project.load_manifest()
    assert saved.shot("s002").still.status is AssetStatus.PASSED
    assert saved.shot("s001").still.attempts == [] and saved.total_generations == 3


def test_exhausted_asset_keeps_best_attempt_and_blocks_video(planned, sample_png):
    project, manifest = planned
    fake = FakeComfy(png=sample_png)
    video_gate = ScriptedGate([True])
    runner = _runner(project, manifest, fake, ScriptedGate([False]), video_gate)
    runner.run([AssetKind.STILL, AssetKind.VIDEO], shot_ids=["s002"])

    shot = manifest.shot("s002")
    assert shot.still.status is AssetStatus.EXHAUSTED
    assert len(shot.still.attempts) == project.config.retry.max_attempts
    assert shot.still.selected_attempt is not None
    assert shot.still.param_overrides == {"steps": 40}
    assert fake.queued[-1]["2"]["inputs"]["steps"] == 40
    # No video time is spent on a still that failed QC until a person accepts it.
    assert video_gate.calls == 0 and shot.video.attempts == []
    review.refresh_reviews(manifest, project.config)
    assert shot.review.reasons == [review.ReviewReason.EXHAUSTED]

    review.accept_still(manifest, shot.id)
    fake.mp4 = b"video"
    runner.run([AssetKind.VIDEO], shot_ids=["s002"])
    assert shot.still.accepted_by_reviewer and shot.video.status is AssetStatus.PASSED


def test_job_errors_count_as_attempts_and_are_recorded(planned, sample_png):
    project, manifest = planned
    fake = FakeComfy(png=sample_png)
    fake.fail_next_jobs = 1
    runner = _runner(project, manifest, fake, ScriptedGate([True]), ScriptedGate([True]))
    runner.run([AssetKind.STILL], shot_ids=["s002"])
    attempts = manifest.shot("s002").still.attempts
    assert "out of memory" in attempts[0].error and attempts[0].qc is None
    assert attempts[1].strategy is RetryStrategy.NEW_SEED and attempts[1].qc.passed


def test_budget_stops_the_run_and_saves_progress(planned, sample_png):
    project, manifest = planned
    runner = _runner(
        project,
        manifest,
        FakeComfy(png=sample_png),
        ScriptedGate([True]),
        ScriptedGate([True]),
        budget=2,
    )
    with pytest.raises(BudgetExceeded):
        runner.run([AssetKind.STILL])
    saved = project.load_manifest()
    assert sum(1 for s in saved.shots if s.still.status is AssetStatus.PASSED) == 2


def test_unreachable_comfy_stops_the_run_without_burning_attempts(planned):
    project, manifest = planned

    class Down(FakeComfy):
        def _handle(self, request):
            import httpx

            raise httpx.ConnectError("refused")

    runner = _runner(project, manifest, Down(), ScriptedGate([True]), ScriptedGate([True]))
    with pytest.raises(ComfyConnectionError):
        runner.run([AssetKind.STILL])
    assert all(not s.still.attempts for s in manifest.shots)


def test_frame_counts_never_drift_from_timeline(planned):
    _, manifest = planned
    counts = frame_counts(manifest.shots, manifest.fps)
    assert sum(counts) == round(manifest.duration * manifest.fps)


def test_clip_command_trims_handle_and_caps_frames(tmp_path):
    from mvdirector.assemble import ClipPlan

    command = clip_command(
        ClipPlan("s002", 96, tmp_path / "in.mp4", True, 0.25), tmp_path / "out.mp4", 1280, 720, 24
    )
    assert command[command.index("-ss") + 1] == "0.2500"
    assert command.index("-ss") < command.index("-i")
    assert command[command.index("-frames:v") + 1] == "96"


def test_assemble_refuses_unfinished_shots(planned):
    project, manifest = planned
    with pytest.raises(NotReadyError, match="6 shot"):
        plan_clips(project, manifest, force=False)
    assert all(p.source is None for p in plan_clips(project, manifest, force=True))


class EnvelopeMouth:
    """Mouth tracker whose output follows (or ignores) the clip's vocal loudness."""

    def __init__(self, audio_path, follow=True):
        self._audio, self._follow = audio_path, follow

    def track(self, frames):
        if not self._follow:
            return np.random.default_rng(3).random(len(frames))
        samples, rate = sf.read(str(self._audio), always_2d=True)
        return 0.1 * lipsync.envelope_at_fps(samples.mean(axis=1), rate, 24.0, len(frames))


@needs_ffmpeg
@pytest.mark.parametrize(
    ("follow", "blocking", "expected"),
    [(True, True, True), (False, True, False), (True, False, True), (False, False, True)],
)
def test_video_gate_lipsync_check_end_to_end(
    planned, tmp_path, sample_mp4, follow, blocking, expected
):
    project, manifest = planned
    project.config.qc.video.lipsync_blocking = blocking
    shot = manifest.shot("s002")
    shot.framing = Framing.CLOSE_UP
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(sample_mp4)
    audio = tmp_path / "slice.wav"
    # Three sung phrases with gaps, so loudness rises and falls within the clip.
    sf.write(audio, vocal_track([(0.5, 1.5), (2.5, 3.5), (5.0, 7.0)], seconds=9.0), SR)
    gate = VideoQC(
        project.config.qc, project.config.video, None, EnvelopeMouth(audio, follow), None
    )
    report = gate.evaluate(clip, shot, manifest, audio, reference=None)
    sync = next(c for c in report.checks if c.name == "lipsync")
    assert (sync.status is CheckStatus.PASSED) is follow
    assert report.passed is expected


@needs_ffmpeg
def test_full_pipeline_from_audio_to_final_video(
    project, tmp_path, sample_png, sample_mp4, monkeypatch
):
    """analyze -> plan -> stills -> videos -> review -> assemble, with real QC and ffmpeg."""
    # The synthetic vocal is a tone with no words: keep transcription out of this test.
    monkeypatch.setattr("mvdirector.audio.lyrics.load_or_transcribe_words", lambda _: [])
    project.config.video.max_clip_seconds = 4.0
    analysis = analyze_project(project)
    assert abs(analysis.tempo_bpm - 120) < 4
    assert len(analysis.vocal_windows) == 1
    assert all(line.start is not None for line in analysis.lyric_lines)

    manifest = build_manifest(project.config, analysis, "lyrics", "style", llm=None)
    assert {s.type for s in manifest.shots} == {ShotType.LIPSYNC, ShotType.PERFORMANCE}
    portrait = tmp_path / "portrait.png"
    portrait.write_bytes(sample_png)
    reference.lock_reference(project, manifest, portrait)

    fake = FakeComfy(png=sample_png, mp4=sample_mp4)
    config = project.config
    image_gate = ImageQC(config.qc, config.video, None, None)
    video_gate = VideoQC(config.qc, config.video, None, None, None)
    _runner(project, manifest, fake, image_gate, video_gate).run([AssetKind.STILL, AssetKind.VIDEO])

    assert all(s.video.status is AssetStatus.PASSED for s in manifest.shots)
    lipsync_graph = next(g for g in fake.queued if g["3"]["inputs"]["audio"])
    assert lipsync_graph["2"]["inputs"]["length"] in {s.num_frames for s in manifest.shots}
    assert all(
        project.path(s.audio_slice.path).is_file()
        for s in manifest.shots
        if s.type is ShotType.LIPSYNC
    )

    # Light mode queues close-ups; the video cannot be built until they are approved.
    review.refresh_reviews(manifest, config)
    queued = review.pending_review(manifest)
    assert queued and all(s.framing is Framing.CLOSE_UP or s.review.reasons for s in queued)
    with pytest.raises(NotReadyError):
        assemble(project, manifest, project.output_dir / "final.mp4")
    for shot in queued:
        review.approve(manifest, shot.id)

    final = assemble(project, manifest, project.output_dir / "final.mp4")
    info = media.probe(final)
    assert info.has_audio and (info.width, info.height) == (1280, 720)
    assert abs(info.duration - analysis.duration) < 0.1


def test_vision_judge_is_given_the_locked_portrait_path(planned, sample_png):
    project, manifest = planned
    gate = ScriptedGate([True])
    runner = _runner(project, manifest, FakeComfy(png=sample_png), gate, ScriptedGate([True]))
    runner.run([AssetKind.STILL], shot_ids=["s002"])
    assert gate.kwargs["reference_image"] == project.path("assets/reference/base_sheet.png")


def test_runner_frees_comfy_vram_between_still_and_video_phases(planned, sample_png):
    project, manifest = planned
    fake = FakeComfy(png=sample_png, mp4=b"video")
    runner = _runner(project, manifest, fake, ScriptedGate([True]), ScriptedGate([True]))
    runner.run([AssetKind.STILL, AssetKind.VIDEO], shot_ids=["s002"])
    assert fake.free_calls == [{"unload_models": True, "free_memory": True}]


def test_t2i_only_still_workflow_and_duration_bound_video_workflow(project, analysis, sample_png):
    import json

    from mvdirector.generation.preflight import check_assets

    manifest = build_manifest(project.config, analysis, "lyrics", "style", llm=None)
    # Remove reference_image from 'still' bindings (e.g. Z-Image T2I-only workflow)
    # and bind 'duration' on 'lipsync' (e.g. MiniMax H3 workflow).
    still_bindings_path = project.path("workflows/still.bindings.json")
    still_bindings = json.loads(still_bindings_path.read_text())
    still_bindings["inputs"].pop("reference_image", None)
    still_bindings_path.write_text(json.dumps(still_bindings))

    lipsync_bindings_path = project.path("workflows/lipsync.bindings.json")
    lipsync_bindings = json.loads(lipsync_bindings_path.read_text())
    lipsync_bindings["inputs"]["duration"] = [{"node": "2", "field": "length"}]
    lipsync_bindings["inputs"].pop("num_frames", None)
    lipsync_bindings_path.write_text(json.dumps(lipsync_bindings))

    templates = load_templates(project, ["t2i", "still", "lipsync", "i2v"])
    # Preflight does not block when 'still' does not accept reference_image.
    assert check_assets(project, manifest, templates).problems == []

    fake = FakeComfy(png=sample_png, mp4=b"video")
    runner = _runner(project, manifest, fake, ScriptedGate([True]), ScriptedGate([True]))
    runner.run([AssetKind.STILL, AssetKind.VIDEO], shot_ids=["s002"])
    shot = manifest.shot("s002")
    assert shot.still.status is AssetStatus.PASSED
    assert shot.video.status is AssetStatus.PASSED
    expected_duration = round(shot.audio_slice.end - shot.audio_slice.start, 3)
    assert fake.queued[-1]["2"]["inputs"]["length"] == expected_duration


# ------------------------------------------------------------ run events and reporting


def test_runner_emits_structured_events_for_every_attempt(planned, sample_png):
    project, manifest = planned
    fake = FakeComfy(png=sample_png)
    events = []
    runner = _runner(
        project,
        manifest,
        fake,
        ScriptedGate([False, True]),
        ScriptedGate([True]),
        observer=events.append,
    )
    runner.run([AssetKind.STILL], shot_ids=["s001"])

    kinds = [e.kind for e in events]
    assert kinds == [
        "phase_started",
        "attempt_started",
        "attempt_finished",
        "attempt_started",
        "attempt_finished",
        "asset_passed",
        "run_finished",
    ]
    first, second = (e for e in events if e.kind == "attempt_finished")
    assert (first.shot_id, first.asset, first.attempt, first.passed) == ("s001", "still", 1, False)
    assert first.issues == ["vision_artifacts: extra fingers"] and first.score == 0.2
    assert (second.attempt, second.strategy, second.passed, second.score) == (
        2,
        "new_seed",
        True,
        0.9,
    )
    assert events[-1].generations_used == 2


@needs_ffmpeg
def test_qc_report_and_calibrate_commands_read_the_manifest(planned, sample_png, sample_mp4):
    from typer.testing import CliRunner

    from mvdirector.cli import app

    project, manifest = planned
    fake = FakeComfy(png=sample_png, mp4=sample_mp4)
    _runner(project, manifest, fake, ScriptedGate([False, True]), ScriptedGate([True])).run(
        [AssetKind.STILL, AssetKind.VIDEO], shot_ids=["s001", "s002"]
    )
    review.approve(manifest, "s001")
    project.save_manifest(manifest)

    result = CliRunner().invoke(app, ["qc-report", "--markdown", "-p", str(project.root)])
    assert result.exit_code == 0, result.output
    assert (
        "## Stills" in result.output and "| `vision_artifacts` | 1 | 0 | 1 | 0 |" in result.output
    )
    assert "6 shots, 3 attempts (0 errored). Passed: 2 (1 on the first attempt" in result.output

    # Only one review decision and the scripted gate records no metrics, so nothing separates.
    result = CliRunner().invoke(app, ["calibrate", "-p", str(project.root)])
    assert result.exit_code == 0, result.output
    assert "no metric had both good and bad examples" in result.output

    labels = project.root / "labels.csv"
    labels.write_text("shot,kind,attempt,label\ns001,video,1,good\ns009,video,1,bad\n")
    result = CliRunner().invoke(
        app, ["calibrate", "--labels", str(labels), "-p", str(project.root)]
    )
    assert result.exit_code == 1 and "unknown shot 's009'" in result.output


def test_the_same_job_error_twice_stops_the_shot_without_exhausting_it(planned, sample_png):
    project, manifest = planned
    fake = FakeComfy(png=sample_png)
    fake.fail_next_jobs = 10
    runner = _runner(project, manifest, fake, ScriptedGate([True]), ScriptedGate([True]))
    runner.run([AssetKind.STILL], shot_ids=["s002"])
    state = manifest.shot("s002").still
    assert len(state.attempts) == 2 and all("out of memory" in a.error for a in state.attempts)
    assert state.status is AssetStatus.PENDING


def _fallback_run(planned, sample_png, *, fallback):
    project, manifest = planned
    project.config.retry.lipsync_fallback = fallback
    shot = manifest.shot("s002")
    shot.type = ShotType.LIPSYNC
    fake = FakeComfy(png=sample_png, mp4=b"video")
    video_gate = ScriptedGate([False] * 4 + [True])
    events = []
    runner = _runner(
        project, manifest, fake, ScriptedGate([True]), video_gate, observer=events.append
    )
    runner.run([AssetKind.STILL, AssetKind.VIDEO], shot_ids=["s002"])
    return project, manifest, shot, events


def test_a_singing_shot_that_cannot_be_made_becomes_a_performance_shot(planned, sample_png):
    project, manifest, shot, events = _fallback_run(planned, sample_png, fallback=True)
    assert shot.type is ShotType.PERFORMANCE and shot.fallback_from is ShotType.LIPSYNC
    assert "not singing" in shot.motion_prompt
    assert shot.video.status is AssetStatus.PASSED and len(shot.video.attempts) == 5
    assert [a.strategy for a in shot.video.attempts] == [
        RetryStrategy.INITIAL,
        RetryStrategy.NEW_SEED,
        RetryStrategy.REWRITE_PROMPT,
        RetryStrategy.ADJUST_PARAMS,
        RetryStrategy.INITIAL,
    ]
    assert shot.video.param_overrides == {}
    kinds = [e.kind for e in events]
    assert "shot_converted" in kinds and "asset_exhausted" not in kinds
    assert shot.still.status is AssetStatus.PASSED  # the still is kept
    review.refresh_reviews(manifest, project.config)
    assert review.ReviewReason.FALLBACK in shot.review.reasons
    assert project.load_manifest().shot("s002").fallback_from is ShotType.LIPSYNC


def test_the_fallback_can_be_turned_off(planned, sample_png):
    _, _, shot, events = _fallback_run(planned, sample_png, fallback=False)
    assert shot.type is ShotType.LIPSYNC and shot.fallback_from is None
    assert shot.video.status is AssetStatus.EXHAUSTED
    kinds = [e.kind for e in events]
    assert "asset_exhausted" in kinds and "shot_converted" not in kinds
