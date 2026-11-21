# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import pytest

from mvdirector.config import ProjectConfig, ReviewMode
from mvdirector.errors import LLMError
from mvdirector.models import (
    AssetKind,
    AssetStatus,
    Attempt,
    CheckResult,
    CheckStatus,
    Framing,
    QCReport,
    RetryStrategy,
    ReviewDecision,
    ReviewReason,
    ShotType,
)
from mvdirector.planning.planner import build_manifest
from mvdirector.retry import RetryPolicy
from mvdirector.review import policy as review


@pytest.fixture
def config():
    config = ProjectConfig()
    config.video.max_clip_seconds = 4.0
    config.planning.max_lipsync_share = None  # these tests need every sung shot on camera
    return config


@pytest.fixture
def manifest(config, analysis):
    return build_manifest(config, analysis, "lyrics", "style", llm=None)


def _report(passed, score, hint=None):
    checks = (
        []
        if passed
        else [
            CheckResult(
                name="lipsync",
                status=CheckStatus.FAILED,
                hard=True,
                detail="out of sync",
                fix_hint=hint,
            )
        ]
    )
    return QCReport(passed=passed, score=score, checks=checks)


def _record(state, passed, score=0.9, hint=None, strategy=RetryStrategy.INITIAL):
    number = len(state.attempts) + 1
    state.attempts.append(
        Attempt(
            number=number,
            strategy=strategy,
            seed=number,
            prompt="p",
            path=f"a{number}.mp4",
            qc=_report(passed, score, hint),
        )
    )
    if passed:
        state.status, state.selected_attempt = AssetStatus.PASSED, number


def _finish(shot, video_score=0.9, video_attempts=1):
    _record(shot.still, True)
    for _ in range(video_attempts - 1):
        _record(shot.video, False, 0.2)
    _record(shot.video, True, video_score)


# --------------------------------------------------------------------------- retry


def test_strategies_escalate_in_configured_order(config, manifest):
    policy = RetryPolicy(config.retry, config.seed, llm=None)
    shot = manifest.shots[1]
    seen = []
    for _ in range(5):
        plan = policy.prepare(shot, AssetKind.VIDEO)
        seen.append(plan.strategy)
        _record(shot.video, False, 0.2, hint="Keep the face frontal.", strategy=plan.strategy)
    assert seen == [
        RetryStrategy.INITIAL,
        RetryStrategy.NEW_SEED,
        RetryStrategy.REWRITE_PROMPT,
        RetryStrategy.ADJUST_PARAMS,
        RetryStrategy.ADJUST_PARAMS,
    ]
    assert "Keep the face frontal." in shot.video.prompt_override
    assert shot.video.param_overrides == {"steps": 40}


def test_seeds_are_deterministic_and_distinct(config, manifest):
    policy = RetryPolicy(config.retry, config.seed, llm=None)
    shot = manifest.shots[0]
    seeds = {policy.seed_for(shot, AssetKind.STILL, n) for n in range(1, 6)}
    assert len(seeds) == 5
    assert policy.seed_for(shot, AssetKind.STILL, 1) == RetryPolicy(
        config.retry, config.seed, None
    ).seed_for(shot, AssetKind.STILL, 1)
    assert policy.seed_for(shot, AssetKind.STILL, 1) != policy.seed_for(shot, AssetKind.VIDEO, 1)


def test_infrastructure_errors_do_not_escalate_strategy(config, manifest):
    policy = RetryPolicy(config.retry, config.seed, llm=None)
    state = manifest.shots[0].still
    state.attempts.append(
        Attempt(number=1, strategy=RetryStrategy.INITIAL, seed=1, prompt="p", error="out of memory")
    )
    state.attempts.append(
        Attempt(
            number=2, strategy=RetryStrategy.NEW_SEED, seed=2, prompt="p", error="out of memory"
        )
    )
    assert policy.prepare(manifest.shots[0], AssetKind.STILL).strategy is RetryStrategy.NEW_SEED


class RewritingLLM:
    def __init__(self, fail=False):
        self.fail, self.requests = fail, []

    def complete_json(self, *, system, content, schema):
        self.requests.append(content[0]["text"])
        if self.fail:
            raise LLMError("api down")
        return schema(prompt="Rewritten: singer faces camera, hands out of frame.")


def test_rewrite_uses_llm_with_failure_reasons(config, manifest):
    llm = RewritingLLM()
    policy = RetryPolicy(config.retry, config.seed, llm)
    shot = manifest.shots[1]
    _record(shot.video, False, hint="Keep the face frontal.")
    _record(shot.video, False, hint="Keep the face frontal.")
    plan = policy.prepare(shot, AssetKind.VIDEO)
    assert plan.strategy is RetryStrategy.REWRITE_PROMPT
    assert shot.video.prompt_override.startswith("Rewritten:")
    assert "lipsync: out of sync" in llm.requests[0] and "Keep the face frontal." in llm.requests[0]


def test_rewrite_falls_back_to_hints_when_llm_fails(config, manifest):
    policy = RetryPolicy(config.retry, config.seed, RewritingLLM(fail=True))
    shot = manifest.shots[1]
    _record(shot.video, False, hint="Keep the face frontal.")
    _record(shot.video, False, hint="Keep the face frontal.")
    policy.prepare(shot, AssetKind.VIDEO)
    assert shot.video.prompt_override == f"{shot.motion_prompt} Keep the face frontal."


def test_director_note_forces_rewrite_and_is_consumed(config, manifest):
    policy = RetryPolicy(config.retry, config.seed, llm=None)
    shot = manifest.shots[1]
    _finish(shot)
    review.reject(manifest, shot.id, "less head movement", AssetKind.VIDEO, config)
    plan = policy.prepare(shot, AssetKind.VIDEO)
    assert plan.strategy is RetryStrategy.REWRITE_PROMPT and plan.number == 2
    assert "less head movement" in shot.video.prompt_override
    assert shot.video.director_note is None


# -------------------------------------------------------------------------- review


def test_strict_mode_queues_only_exhausted_shots(config, manifest):
    config.review.mode = ReviewMode.STRICT
    for shot in manifest.shots:
        _finish(shot, video_attempts=3)
    failed = manifest.shots[0]
    failed.video.status = AssetStatus.EXHAUSTED
    review.refresh_reviews(manifest, config)
    assert [s.id for s in review.pending_review(manifest)] == [failed.id]
    assert failed.review.reasons == [ReviewReason.EXHAUSTED]
    assert all(review.is_ready(s) for s in manifest.shots[1:]) and not review.is_ready(failed)


def test_light_mode_queues_closeups_retried_and_near_threshold(config, manifest):
    config.review.other_rate = 0.0
    clean, closeup, retried, borderline = manifest.shots[:4]
    for shot in (clean, retried, borderline):
        shot.framing = Framing.MEDIUM
    closeup.framing = Framing.CLOSE_UP
    _finish(clean)
    _finish(closeup)
    _finish(retried, video_attempts=2)
    _finish(borderline, video_score=config.qc.pass_score + 0.01)
    review.refresh_reviews(manifest, config)
    assert clean.review.reasons == [] and review.is_ready(clean)
    assert closeup.review.reasons == [ReviewReason.SAMPLED]
    assert retried.review.reasons == [ReviewReason.RETRIED]
    assert borderline.review.reasons == [ReviewReason.NEAR_THRESHOLD]
    assert not review.is_ready(closeup)
    # Unfinished shots are never queued.
    assert manifest.shots[4].review.required is False


def test_sampling_is_deterministic_and_tracks_rate():
    ids = [f"s{i:03d}" for i in range(400)]
    picked = [i for i in ids if review.is_sampled(i, 0.25, seed=1)]
    assert picked == [i for i in ids if review.is_sampled(i, 0.25, seed=1)]
    assert 70 <= len(picked) <= 130
    assert not any(review.is_sampled(i, 0.0, seed=1) for i in ids)
    assert all(review.is_sampled(i, 1.0, seed=1) for i in ids)


def test_switching_modes_keeps_decisions(config, manifest):
    shot = manifest.shots[1]
    shot.framing = Framing.CLOSE_UP
    _finish(shot)
    review.refresh_reviews(manifest, config)
    review.approve(manifest, shot.id)
    config.review.mode = ReviewMode.STRICT
    review.refresh_reviews(manifest, config)
    assert shot.review.decision is ReviewDecision.APPROVED and review.is_ready(shot)


def test_reject_video_resets_it_and_escalates_shot_type(config, manifest):
    config.review.other_rate = 0.0
    rejected, sibling = manifest.shots[1], manifest.shots[2]
    assert rejected.type is sibling.type is ShotType.LIPSYNC
    for shot in (rejected, sibling):
        shot.framing = Framing.MEDIUM
        _finish(shot)
    review.refresh_reviews(manifest, config)
    assert not sibling.review.required

    review.reject(manifest, rejected.id, "eyes look wrong", AssetKind.VIDEO, config)
    assert rejected.video.status is AssetStatus.PENDING
    assert rejected.video.attempts_allowed == 1 + config.retry.max_attempts
    assert rejected.video.director_note == "eyes look wrong"
    assert rejected.still.status is AssetStatus.PASSED
    assert not review.is_ready(rejected)

    review.refresh_reviews(manifest, config)
    assert sibling.review.reasons == [ReviewReason.ESCALATED]


def test_reject_still_resets_video_too(config, manifest):
    shot = manifest.shots[1]
    _finish(shot)
    review.reject(manifest, shot.id, "wrong outfit", AssetKind.STILL, config)
    assert shot.still.status is shot.video.status is AssetStatus.PENDING
    assert shot.still.director_note == "wrong outfit" and shot.video.director_note is None
    assert len(shot.video.attempts) == 1  # history is kept


def test_approving_exhausted_shot_makes_it_ready(config, manifest):
    shot = manifest.shots[1]
    _record(shot.still, True)
    _record(shot.video, False, 0.5)
    shot.video.status = AssetStatus.EXHAUSTED
    shot.video.selected_attempt = shot.video.best_attempt().number
    review.refresh_reviews(manifest, config)
    assert not review.is_ready(shot)
    review.approve(manifest, shot.id)
    assert review.is_ready(shot)


def test_best_attempt_prefers_no_hard_failure_then_score(manifest):
    state = manifest.shots[0].video
    _record(state, False, 0.9)  # hard failure, high score
    state.attempts.append(
        Attempt(
            number=2,
            strategy=RetryStrategy.NEW_SEED,
            seed=2,
            prompt="p",
            path="a2.mp4",
            qc=QCReport(passed=False, score=0.6),
        )
    )
    state.attempts.append(
        Attempt(number=3, strategy=RetryStrategy.NEW_SEED, seed=3, prompt="p", error="failed")
    )
    assert state.best_attempt().number == 2


def test_advisory_lipsync_failure_queues_a_passed_shot_for_review(config, manifest):
    config.review.other_rate = 0.0
    shot = manifest.shots[0]
    shot.framing = Framing.MEDIUM
    _finish(shot)
    shot.video.selected.qc.checks.append(
        CheckResult(name="lipsync", status=CheckStatus.FAILED, detail="correlation 0.29")
    )
    review.refresh_reviews(manifest, config)
    assert shot.review.reasons == [ReviewReason.LIPSYNC] and not review.is_ready(shot)


def test_blocking_lipsync_failure_is_not_also_queued_for_review(config, manifest):
    config.review.other_rate = 0.0
    shot = manifest.shots[0]
    shot.framing = Framing.MEDIUM
    _finish(shot)
    shot.video.selected.qc.checks.append(
        CheckResult(name="lipsync", status=CheckStatus.FAILED, hard=True)
    )
    review.refresh_reviews(manifest, config)
    assert ReviewReason.LIPSYNC not in shot.review.reasons


def test_fallback_reset_restarts_the_strategy_ladder_and_scopes_best_attempt(config, manifest):
    policy = RetryPolicy(config.retry, config.seed, llm=None)
    shot = manifest.shots[1]
    for _ in range(4):
        plan = policy.prepare(shot, AssetKind.VIDEO)
        _record(shot.video, False, 0.7, hint="Sing more clearly.", strategy=plan.strategy)
    assert shot.video.param_overrides == {"steps": 40}

    # Convert to performance and reset for a fresh pass.
    shot.fallback_from, shot.type = shot.type, ShotType.PERFORMANCE
    shot.motion_prompt = "The singer moves with the music, mouth closed, not singing."
    shot.video.reset(config.retry.max_attempts)
    shot.video.prompt_override = None
    shot.video.param_overrides = {}

    seen = []
    for i in range(4):
        plan = policy.prepare(shot, AssetKind.VIDEO)
        seen.append(plan.strategy)
        if i == 0:
            assert shot.video.param_overrides == {}
        _record(shot.video, False, 0.4 + i * 0.05, hint="Steady camera.", strategy=plan.strategy)

    assert seen == [
        RetryStrategy.INITIAL,
        RetryStrategy.NEW_SEED,
        RetryStrategy.REWRITE_PROMPT,
        RetryStrategy.ADJUST_PARAMS,
    ]
    assert "Steady camera." in shot.video.prompt_override
    assert "Sing more clearly." not in shot.video.prompt_override
    # best_attempt picks from the performance pass (5..8), never the earlier lipsync pass (1..4).
    assert shot.video.best_attempt().number == 8


def test_prompt_spec_failure_jumps_immediately_to_rewrite_prompt_on_attempt_two(config, manifest):
    llm = RewritingLLM()
    policy = RetryPolicy(config.retry, config.seed, llm)
    shot = manifest.shots[0]
    shot.type = ShotType.BROLL
    shot.still_prompt = "Grandfather clock, no minute hand."

    p1 = policy.prepare(shot, AssetKind.STILL)
    assert p1.strategy is RetryStrategy.INITIAL
    shot.still.attempts.append(
        Attempt(
            number=1,
            strategy=p1.strategy,
            seed=p1.seed,
            prompt=shot.still_prompt,
            path="s001_a01.png",
            qc=QCReport(
                passed=False,
                score=0.72,
                checks=[
                    CheckResult(
                        name="vision_spec",
                        status=CheckStatus.FAILED,
                        hard=True,
                        score=0.2,
                        detail="clock has a minute hand",
                        fix_hint="Show only one short hour hand on a bare center pin.",
                    )
                ],
            ),
        )
    )

    # Attempt 2 rewrites immediately rather than wasting a seed on 'no minute hand'.
    p2 = policy.prepare(shot, AssetKind.STILL)
    assert p2.strategy is RetryStrategy.REWRITE_PROMPT
    assert shot.still.prompt_override.startswith("Rewritten:")
    shot.still.attempts.append(
        Attempt(
            number=2,
            strategy=p2.strategy,
            seed=p2.seed,
            prompt=shot.still.prompt_override,
            path="s001_a02.png",
            qc=_report(False, 0.3, hint="still noisy"),
        )
    )

    # Subsequent attempts try NEW_SEED and ADJUST_PARAMS on the rewritten prompt.
    p3 = policy.prepare(shot, AssetKind.STILL)
    assert p3.strategy is RetryStrategy.NEW_SEED
    assert shot.still.prompt_override.startswith("Rewritten:")
    shot.still.attempts.append(
        Attempt(
            number=3,
            strategy=p3.strategy,
            seed=p3.seed,
            prompt=shot.still.prompt_override,
            path="s001_a03.png",
            qc=_report(False, 0.3),
        )
    )
    p4 = policy.prepare(shot, AssetKind.STILL)
    assert p4.strategy is RetryStrategy.ADJUST_PARAMS
