# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Tests for Doc 2 primitives: keep_still_on_video_fail, pin_character_details, interrupt."""

from __future__ import annotations

import httpx
import pytest

from mvdirector.comfy.client import ComfyClient
from mvdirector.config import PlanningConfig, RetryConfig
from mvdirector.errors import ComfyJobError
from mvdirector.generation.runner import RunBudget, ShotRunner, load_templates
from mvdirector.models import (
    AssetKind,
    AssetStatus,
    Attempt,
    Character,
    Framing,
    QCReport,
    RetryStrategy,
    ShotType,
)
from mvdirector.planning import prompts
from mvdirector.planning.planner import build_manifest
from mvdirector.qc.factory import QualityGates
from mvdirector.retry import RetryPolicy
from mvdirector.review import policy as review
from tests.conftest import FakeComfy
from tests.test_pipeline import ScriptedGate


def test_doc2_config_defaults_and_keep_still_on_video_fail_wiring(
    project, analysis, sample_png, sample_mp4
) -> None:
    assert RetryConfig().keep_still_on_video_fail is True
    assert PlanningConfig().pin_character_details is True

    manifest = build_manifest(project.config, analysis, "lyrics", "style", llm=None)
    shot = manifest.shots[0]
    shot.type = ShotType.BROLL
    shot.still.attempts = [
        Attempt(
            number=1,
            strategy=RetryStrategy.INITIAL,
            seed=1,
            prompt="still prompt",
            path="assets/stills/s001_a01.png",
            qc=QCReport(passed=True, score=0.9),
        )
    ]
    still_file = project.path("assets/stills/s001_a01.png")
    still_file.parent.mkdir(parents=True, exist_ok=True)
    still_file.write_bytes(sample_png)
    shot.still.status = AssetStatus.PASSED
    shot.still.selected_attempt = 1
    shot.video.attempts_allowed = 1

    # 1. Default keep_still_on_video_fail=True preserves PASSED still on video exhaustion & reject
    fake = FakeComfy(png=sample_png, mp4=sample_mp4)
    runner = ShotRunner(
        project,
        manifest,
        ComfyClient(project.config.comfy, transport=fake.transport(), sleep=lambda _: None),
        QualityGates(image=ScriptedGate([True]), video=ScriptedGate([False]), face_analyzer=None),
        RetryPolicy(project.config.retry, project.config.seed, llm=None),
        load_templates(project, ["t2i", "still", "lipsync", "i2v"]),
        RunBudget(10),
    )
    runner.run([AssetKind.VIDEO], shot_ids=[shot.id])
    assert shot.video.status is AssetStatus.EXHAUSTED
    assert shot.still.status is AssetStatus.PASSED
    assert shot.still.selected_attempt == 1

    review.reject(manifest, shot.id, "flicker", AssetKind.VIDEO, project.config)
    assert shot.video.status is AssetStatus.PENDING
    assert shot.still.status is AssetStatus.PASSED

    # 2. keep_still_on_video_fail=False resets shot.still on reject and video exhaustion
    project.config.retry.keep_still_on_video_fail = False
    review.reject(manifest, shot.id, "re-render from scratch", AssetKind.VIDEO, project.config)
    assert shot.still.status is AssetStatus.PENDING
    assert shot.still.selected_attempt is None

    # Restore PASSED still and verify exhaustion with keep_still_on_video_fail=False resets still
    shot.still.status = AssetStatus.PASSED
    shot.still.selected_attempt = 1
    shot.video.attempts_allowed = len(shot.video.attempts) + 1
    runner.run([AssetKind.VIDEO], shot_ids=[shot.id])
    assert shot.still.status is AssetStatus.PENDING
    assert shot.still.selected_attempt is None
    # The video goes back to pending too: left EXHAUSTED it would never re-render from the new
    # still, and its kept clip would no longer match the still it was made from.
    assert shot.video.status is AssetStatus.PENDING
    assert shot.video.selected_attempt is None


def test_keep_still_false_rerenders_video_from_new_still_after_exhaustion(
    project, analysis, sample_png, sample_mp4
) -> None:
    """Regression: a video exhaustion that resets the still must not strand the video."""
    project.config.retry.keep_still_on_video_fail = False
    project.config.retry.max_attempts = 1
    manifest = build_manifest(project.config, analysis, "lyrics", "style", llm=None)
    shot = manifest.shots[0]
    shot.type = ShotType.BROLL
    shot.still.attempts_allowed = 1
    shot.video.attempts_allowed = 1
    fake = FakeComfy(png=sample_png, mp4=sample_mp4)

    def _runner(video_passes: list[bool]) -> ShotRunner:
        return ShotRunner(
            project,
            manifest,
            ComfyClient(project.config.comfy, transport=fake.transport(), sleep=lambda _: None),
            QualityGates(
                image=ScriptedGate([True, True]),
                video=ScriptedGate(video_passes),
                face_analyzer=None,
            ),
            RetryPolicy(project.config.retry, project.config.seed, llm=None),
            load_templates(project, ["t2i", "still", "lipsync", "i2v"]),
            RunBudget(10),
        )

    _runner([False]).run([AssetKind.STILL, AssetKind.VIDEO], shot_ids=[shot.id])
    assert shot.still.status is AssetStatus.PENDING
    assert shot.video.status is AssetStatus.PENDING

    # The next run renders a fresh still and then a video made from it.
    _runner([True]).run([AssetKind.STILL, AssetKind.VIDEO], shot_ids=[shot.id])
    assert shot.still.status is AssetStatus.PASSED
    assert shot.still.selected_attempt == 2
    assert shot.video.status is AssetStatus.PASSED
    assert shot.video.selected_attempt == 2


def test_pin_character_details_controls_still_motion_draft_grid_and_runner(
    project, analysis, sample_png, sample_mp4
) -> None:
    manifest = build_manifest(project.config, analysis, "lyrics", "style", llm=None)
    ref_file = project.path("assets/reference/lead.png")
    ref_file.parent.mkdir(parents=True, exist_ok=True)
    ref_file.write_bytes(sample_png)
    manifest.character = Character(
        name="Aria",
        appearance="Tall woman with silver bob hair",
        invariants="scar above left eyebrow, silver cuff",
        references={"base": {"sheet": project.rel(ref_file)}},  # type: ignore[dict-item]
    )
    shot = next(s for s in manifest.shots if s.type is ShotType.LIPSYNC)
    shot.framing = Framing.MEDIUM_CLOSE

    still_pinned = prompts.compose_still_prompt(manifest, shot, pin_character_details=True)
    still_unpinned = prompts.compose_still_prompt(manifest, shot, pin_character_details=False)
    assert "Always: scar above left eyebrow, silver cuff." in still_pinned
    assert "Always:" not in still_unpinned
    assert "scar above left eyebrow" not in still_unpinned

    motion_pinned = prompts.compose_motion_prompt(manifest, shot, pin_character_details=True)
    motion_unpinned = prompts.compose_motion_prompt(manifest, shot, pin_character_details=False)
    assert "Always: scar above left eyebrow, silver cuff." in motion_pinned
    assert "Always:" not in motion_unpinned

    grid_pinned = prompts.draft_grid_prompt(manifest, [shot], "2x2", pin_character_details=True)
    grid_unpinned = prompts.draft_grid_prompt(manifest, [shot], "2x2", pin_character_details=False)
    assert "Always: scar above left eyebrow, silver cuff." in grid_pinned
    assert "Always:" not in grid_unpinned

    # End-to-end ShotRunner honors PlanningConfig.pin_character_details=False
    project.config.planning.pin_character_details = False
    fake = FakeComfy(png=sample_png, mp4=sample_mp4)
    runner = ShotRunner(
        project,
        manifest,
        ComfyClient(project.config.comfy, transport=fake.transport(), sleep=lambda _: None),
        QualityGates(image=ScriptedGate([True]), video=ScriptedGate([True]), face_analyzer=None),
        RetryPolicy(project.config.retry, project.config.seed, llm=None),
        load_templates(project, ["t2i", "still", "lipsync", "i2v"]),
        RunBudget(10),
    )
    runner.run([AssetKind.STILL, AssetKind.VIDEO], shot_ids=[shot.id])
    assert "Always:" not in shot.still.attempts[0].prompt
    assert "Always:" not in shot.video.attempts[0].prompt


def test_comfy_client_interrupt_validation_headers_and_error_handling(project) -> None:
    seen_headers: list[dict[str, str]] = []
    fake = FakeComfy()

    def _capture_transport(request: httpx.Request) -> httpx.Response:
        seen_headers.append(dict(request.headers))
        return fake._handle(request)

    project.config.comfy.headers = {"Authorization": "Bearer secret-token"}
    client = ComfyClient(
        project.config.comfy,
        transport=httpx.MockTransport(_capture_transport),
        sleep=lambda _: None,
    )
    client.interrupt()
    client.interrupt("prompt-42")
    assert fake.interrupt_calls == [{}, {"prompt_id": "prompt-42"}]
    assert seen_headers[0]["authorization"] == "Bearer secret-token"

    for bad_id in ("", "../escape", "a/b", "a\\b", "p?x=1", "p#frag", "p\nid"):
        with pytest.raises(ComfyJobError):
            client.interrupt(bad_id)

    # HTTP 500 and TransportError are handled gracefully without raising
    err_client = ComfyClient(
        project.config.comfy,
        transport=httpx.MockTransport(lambda _req: httpx.Response(500)),
        sleep=lambda _: None,
    )
    err_client.interrupt("p1")

    def _fail_transport(_req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline")

    net_client = ComfyClient(
        project.config.comfy,
        transport=httpx.MockTransport(_fail_transport),
        sleep=lambda _: None,
    )
    net_client.interrupt()


def test_pin_character_details_empty_appearance_and_closeup_facial_invariants(
    project, analysis
) -> None:
    manifest = build_manifest(project.config, analysis, "lyrics", "style", llm=None)
    shot = next(s for s in manifest.shots if s.type is ShotType.LIPSYNC)
    shot.framing = Framing.CLOSE_UP

    # 1. Empty appearance: both still and motion prompts omit pinned details consistently
    manifest.character = Character(name="Aria", appearance="", invariants="silver hoop earrings")
    still_empty = prompts.compose_still_prompt(manifest, shot, pin_character_details=True)
    motion_empty = prompts.compose_motion_prompt(manifest, shot, pin_character_details=True)
    assert "Always:" not in still_empty
    assert "Always:" not in motion_empty

    # 2. CLOSE_UP with facial invariant + lower-body term (and no upper-body garment word)
    #    keeps the facial invariant and strips only the lower-body clause.
    manifest.character = Character(
        name="Aria",
        appearance="Tall woman with dark hair",
        invariants="freckles across nose, silver belt",
    )
    still_cu = prompts.compose_still_prompt(manifest, shot, pin_character_details=True)
    motion_cu = prompts.compose_motion_prompt(manifest, shot, pin_character_details=True)
    assert "Always: freckles across nose." in still_cu
    assert "Always: freckles across nose." in motion_cu
    assert "silver belt" not in still_cu
    assert "silver belt" not in motion_cu
    assert (
        prompts.scrub_lower_body_wardrobe("She wears simple, flat leather shoes.", allow_empty=True)
        == ""
    )


def test_reject_rate_survives_regeneration_and_later_approval(
    project, analysis, sample_png, sample_mp4
) -> None:
    """Regression: the runner clears ``shot.review`` on regeneration, which erased rejections.

    After the usual reject -> run -> approve cycle the human reject rate read 0%.
    """
    from mvdirector import report

    manifest = build_manifest(project.config, analysis, "lyrics", "style", llm=None)
    shot = manifest.shots[0]
    shot.type = ShotType.BROLL
    fake = FakeComfy(png=sample_png, mp4=sample_mp4)

    def _run() -> None:
        ShotRunner(
            project,
            manifest,
            ComfyClient(project.config.comfy, transport=fake.transport(), sleep=lambda _: None),
            QualityGates(
                image=ScriptedGate([True] * 4), video=ScriptedGate([True] * 4), face_analyzer=None
            ),
            RetryPolicy(project.config.retry, project.config.seed, llm=None),
            load_templates(project, ["t2i", "still", "lipsync", "i2v"]),
            RunBudget(10),
        ).run([AssetKind.STILL, AssetKind.VIDEO], shot_ids=[shot.id])

    _run()
    review.reject(manifest, shot.id, "too dark", AssetKind.VIDEO, project.config)
    _run()
    review.reject(manifest, shot.id, "still too dark", AssetKind.VIDEO, project.config)
    _run()
    assert shot.review.decision is None  # the runner cleared the last rejection
    review.approve(manifest, shot.id)
    assert shot.rejection_count == 2

    stats = report.attempt_stats(manifest)
    assert stats.rejected_count == 1
    assert stats.total_rejections == 2
    assert stats.approved_count == 1
    assert stats.reviewed_count == 1
    assert stats.human_reject_rate == 1.0

    # The counter is persisted, so a reloaded manifest reports the same numbers.
    project.save_manifest(manifest)
    assert report.attempt_stats(project.load_manifest()).to_dict() == stats.to_dict()


def test_rejection_count_is_backward_compatible_and_validated(project, analysis) -> None:
    import json

    from pydantic import ValidationError

    from mvdirector.models import Manifest

    manifest = build_manifest(project.config, analysis, "lyrics", "style", llm=None)
    raw = json.loads(manifest.model_dump_json())
    for shot in raw["shots"]:
        shot.pop("rejection_count")  # a manifest written before the field existed
    assert Manifest.model_validate(raw).shots[0].rejection_count == 0
    for bad in (True, -1, 1.5):
        raw["shots"][0]["rejection_count"] = bad
        with pytest.raises(ValidationError):
            Manifest.model_validate(raw)


def test_resets_grant_the_opening_bonus_like_planning(project, analysis) -> None:
    """Opening-window shots keep retry.opening_extra_attempts when a reviewer resets them."""
    cfg = project.config
    cfg.retry.max_attempts, cfg.retry.opening_extra_attempts = 4, 2
    cfg.planning.opening_seconds = 8.0
    assert cfg.attempt_budget(0.0) == 6
    assert cfg.attempt_budget(7.99) == 6
    assert cfg.attempt_budget(8.0) == 4
    cfg.planning.opening_seconds = 0.0  # disabled
    assert cfg.attempt_budget(0.0) == 4
    cfg.planning.opening_seconds = 8.0

    manifest = build_manifest(cfg, analysis, "lyrics", "style", llm=None)
    opening = next(s for s in manifest.shots if s.start < 8.0)
    later = next(s for s in manifest.shots if s.start >= 8.0)
    assert opening.still.attempts_allowed == 6 and later.still.attempts_allowed == 4

    for shot in (opening, later):
        review.reject(manifest, shot.id, "again", AssetKind.STILL, cfg)
    assert opening.still.attempts_allowed == 6 and opening.video.attempts_allowed == 6
    assert later.still.attempts_allowed == 4 and later.video.attempts_allowed == 4


def test_still_restarts_are_capped_then_the_shot_goes_to_review(
    project, analysis, sample_png, sample_mp4
) -> None:
    """With keep_still_on_video_fail false, a video that keeps failing restarts from a new
    still at most retry.max_still_restarts times, then stays EXHAUSTED for review."""
    project.config.retry.keep_still_on_video_fail = False
    project.config.retry.max_attempts = 1
    project.config.retry.opening_extra_attempts = 0
    project.config.retry.max_still_restarts = 1
    manifest = build_manifest(project.config, analysis, "lyrics", "style", llm=None)
    shot = manifest.shots[0]
    shot.type = ShotType.BROLL
    fake = FakeComfy(png=sample_png, mp4=sample_mp4)
    events: list[str] = []

    def _run() -> None:
        events.clear()
        ShotRunner(
            project,
            manifest,
            ComfyClient(project.config.comfy, transport=fake.transport(), sleep=lambda _: None),
            QualityGates(
                image=ScriptedGate([True] * 4), video=ScriptedGate([False] * 4), face_analyzer=None
            ),
            RetryPolicy(project.config.retry, project.config.seed, llm=None),
            load_templates(project, ["t2i", "still", "lipsync", "i2v"]),
            RunBudget(10),
            observer=lambda event: events.append(event.kind),
        ).run([AssetKind.STILL, AssetKind.VIDEO], shot_ids=[shot.id])

    _run()  # first exhaustion: restart from a new still
    assert shot.still_restarts == 1
    assert shot.still.status is AssetStatus.PENDING
    assert shot.video.status is AssetStatus.PENDING
    # Regression: a restart was reported as "exhausted; queued for review".
    assert "asset_restarted" in events and "asset_exhausted" not in events

    _run()  # second exhaustion: the cap is reached, so it is kept for review
    assert "asset_exhausted" in events and "asset_restarted" not in events
    assert shot.still_restarts == 1
    assert shot.still.status is AssetStatus.PASSED
    assert shot.video.status is AssetStatus.EXHAUSTED
    assert shot.video.selected_attempt is not None
    assert review.review_reasons(shot, manifest, project.config) == [review.ReviewReason.EXHAUSTED]

    attempts_before = len(shot.still.attempts) + len(shot.video.attempts)
    _run()  # nothing left to do: no further spend
    assert len(shot.still.attempts) + len(shot.video.attempts) == attempts_before

    project.config.retry.max_still_restarts = 0  # 0 disables restarts entirely
    other = manifest.shots[1]
    other.type = ShotType.BROLL
    shot = other
    _run()
    assert other.still_restarts == 0 and other.video.status is AssetStatus.EXHAUSTED


def test_cli_progress_reports_restart_as_pending_not_review(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from mvdirector.cli import _progress
    from mvdirector.generation.runner import RunEvent

    _progress(RunEvent("asset_restarted", asset="video", shot_id="s01", attempt=2))
    out = capsys.readouterr().out
    assert "run again" in out and "queued for review" not in out
