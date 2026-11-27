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
    assert shot.video.status is AssetStatus.EXHAUSTED
    assert shot.still.status is AssetStatus.PENDING
    assert shot.still.selected_attempt is None


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
