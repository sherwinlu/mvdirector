# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Tests for explicit in-place still repair vs. fresh recreate API and runner behavior."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from mvdirector import api
from mvdirector import cli as cli_mod
from mvdirector.config import ProjectMode, ReviewMode
from mvdirector.errors import ConfigError, MVDError
from mvdirector.generation import plates, reference
from mvdirector.models import (
    AssetKind,
    AssetState,
    AssetStatus,
    Attempt,
    AudioSlice,
    CheckResult,
    CheckStatus,
    ContinuityMode,
    Framing,
    Manifest,
    QCReport,
    RetryStrategy,
    ReviewDecision,
    SetDesign,
    Shot,
    ShotType,
    VisualBible,
)
from mvdirector.planning.planner import build_manifest
from mvdirector.qc.factory import QualityGates
from mvdirector.retry import RetryPolicy
from mvdirector.review import policy as review_policy
from tests.conftest import WORKFLOW_GRAPH, FakeComfy
from tests.test_api_stages import _custom_project, _LeaseRecorder
from tests.test_pipeline import ScriptedGate


def _seed_attempt(
    project,
    shot,
    number: int,
    png_bytes: bytes,
    *,
    passed: bool = True,
    score: float = 0.9,
    strategy: RetryStrategy = RetryStrategy.INITIAL,
    regressed: bool = False,
    issues: list[str] | None = None,
    checks: list[CheckResult] | None = None,
    select: bool = True,
) -> Path:
    path = project.asset_path(shot, AssetKind.STILL, number)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png_bytes)
    report_checks = checks or (
        []
        if passed
        else [
            CheckResult(
                name="vision_artifacts",
                status=CheckStatus.FAILED,
                hard=False,
                detail=(issues[0] if issues else "extra finger"),
                fix_hint="fix hand",
            )
        ]
    )
    attempt = Attempt(
        number=number,
        strategy=strategy,
        seed=number,
        prompt=f"attempt {number}",
        path=project.rel(path),
        qc=QCReport(passed=passed, score=score, checks=report_checks),
        regressed=regressed,
    )
    shot.still.attempts.append(attempt)
    if select:
        shot.still.selected_attempt = number
        shot.still.status = AssetStatus.PASSED if passed else AssetStatus.EXHAUSTED
    return path


def _seed_video(project, shot, mp4_bytes: bytes) -> None:
    vpath = project.asset_path(shot, AssetKind.VIDEO, 1)
    vpath.parent.mkdir(parents=True, exist_ok=True)
    vpath.write_bytes(mp4_bytes)
    shot.video = AssetState(
        status=AssetStatus.PASSED,
        attempts_allowed=4,
        selected_attempt=1,
        attempts=[
            Attempt(
                number=1,
                strategy=RetryStrategy.INITIAL,
                seed=1,
                prompt="v",
                path=project.rel(vpath),
                qc=QCReport(passed=True, score=0.9),
            )
        ],
    )


def _patch_gates(monkeypatch: pytest.MonkeyPatch, outcomes: list[bool | QCReport]) -> None:
    class _ReportGate:
        def __init__(self, items: list[bool | QCReport]) -> None:
            self._items = items
            self._calls = 0

        def evaluate(self, *_args, **_kwargs) -> QCReport:
            item = self._items[min(self._calls, len(self._items) - 1)]
            self._calls += 1
            if isinstance(item, QCReport):
                return item
            return ScriptedGate([item]).evaluate()

    def _factory(_config, _llm):
        return QualityGates(
            image=_ReportGate(outcomes), video=ScriptedGate([True]), face_analyzer=None
        )

    monkeypatch.setattr(api, "build_quality_gates", _factory)


def test_repair_still_edits_passed_approved_singer_still_in_place(
    project, analysis, sample_png, sample_mp4, tmp_path, monkeypatch
) -> None:
    project.config.llm.enabled = False
    project.config.retry.in_place_repair = False
    manifest = build_manifest(project.config, analysis, "lyrics", "style", llm=None)
    shot = next(s for s in manifest.shots if s.type is ShotType.LIPSYNC)

    # Give the shot a reference sheet, plate, and passed video to verify repair omits them.
    ref_img = tmp_path / "ref.png"
    ref_img.write_bytes(sample_png)
    reference.lock_reference(project, manifest, ref_img)
    manifest.visual_bible.settings = [SetDesign(name=shot.setting, description="stage")]
    plate_img = tmp_path / "plate.png"
    plate_img.write_bytes(sample_png)
    plates.lock_plate(project, manifest, plate_img, shot.setting)

    _seed_attempt(project, shot, 1, sample_png, passed=True, score=0.92, select=True)
    _seed_video(project, shot, sample_mp4)
    review_policy.approve(manifest, shot.id, stills_only=True)

    custom, state, _ = _custom_project(project, manifest)
    provider = _LeaseRecorder(FakeComfy(png=sample_png))
    _patch_gates(monkeypatch, [True])

    result = api.repair_still(
        custom,
        shot.id,
        instruction="remove stray microphone stand on left",
        skip_preflight=True,
        comfy_provider=provider,
    )
    updated = result.manifest.shot(shot.id)
    assert updated.still.status is AssetStatus.PASSED
    assert updated.still.selected_attempt == 2
    assert updated.still.pending_strategy is None
    assert updated.still.repair_instruction is None
    assert updated.still.repair_source_attempt is None
    assert updated.video.status is AssetStatus.PENDING
    assert updated.video.selected_attempt is None

    new_attempt = updated.still.attempts[-1]
    assert new_attempt.strategy is RetryStrategy.REPAIR_STILL
    assert "Apply ONLY this edit: remove stray microphone stand on left" in new_attempt.prompt
    assert provider.fake.upload_files == [f"{shot.id}_a01.png"]
    assert state.manifest.shot(shot.id).still.selected_attempt == 2


def test_repair_still_empty_instruction_and_qc_failed_prompts(
    project, analysis, sample_png, monkeypatch
) -> None:
    project.config.llm.enabled = False
    manifest = build_manifest(project.config, analysis, "lyrics", "style", llm=None)
    shot = next(s for s in manifest.shots if s.type is ShotType.LIPSYNC)
    _seed_attempt(project, shot, 1, sample_png, passed=True, score=0.9, select=True)

    custom, _, _ = _custom_project(project, manifest)
    provider = _LeaseRecorder(FakeComfy(png=sample_png))
    _patch_gates(monkeypatch, [True, True])

    # Empty instruction on a QC-passing frame uses clean refinement prompt.
    res1 = api.repair_still(custom, shot.id, "", skip_preflight=True, comfy_provider=provider)
    p1 = res1.manifest.shot(shot.id).still.attempts[-1].prompt
    assert "Refine image quality and fix any subtle visual artifacts." in p1
    assert "Fix ONLY these defects" not in p1

    # Consecutive second repair_still also runs REPAIR_STILL (bypasses one-in-a-row guard)
    # and when prior QC failed without instruction, uses "Fix ONLY these defects:".
    shot2 = res1.manifest.shot(shot.id)
    shot2.still.attempts[-1].qc = QCReport(
        passed=False,
        score=0.5,
        checks=[
            CheckResult(
                name="vision_artifacts",
                status=CheckStatus.FAILED,
                detail="warped hand",
                fix_hint="keep hand relaxed",
            )
        ],
    )
    shot2.still.status = AssetStatus.EXHAUSTED
    custom.save_manifest(res1.manifest)

    res2 = api.repair_still(custom, shot.id, "", skip_preflight=True, comfy_provider=provider)
    a3 = res2.manifest.shot(shot.id).still.attempts[-1]
    assert a3.strategy is RetryStrategy.REPAIR_STILL
    assert "Fix ONLY these defects: vision_artifacts: warped hand." in a3.prompt


def test_repair_still_uses_selected_attempt_not_last_regressed_attempt(
    project, analysis, sample_png, monkeypatch
) -> None:
    project.config.llm.enabled = False
    manifest = build_manifest(project.config, analysis, "lyrics", "style", llm=None)
    shot = next(s for s in manifest.shots if s.type is ShotType.LIPSYNC)

    # Attempt 1 is selected; attempt 2 was a regressed/errored attempt.
    _seed_attempt(project, shot, 1, sample_png, passed=True, score=0.88, select=True)
    _seed_attempt(
        project,
        shot,
        2,
        sample_png,
        passed=False,
        score=0.2,
        strategy=RetryStrategy.REPAIR_STILL,
        regressed=True,
        select=False,
    )
    shot.still.selected_attempt = 1
    shot.still.status = AssetStatus.PASSED

    custom, _, _ = _custom_project(project, manifest)
    provider = _LeaseRecorder(FakeComfy(png=sample_png))
    # New edit fails QC -> anti-regression compares against attempt 1 (selected), keeping attempt 1.
    _patch_gates(monkeypatch, [False])

    res = api.repair_still(
        custom,
        shot.id,
        instruction="brighten collar",
        skip_preflight=True,
        comfy_provider=provider,
    )
    st = res.manifest.shot(shot.id).still
    assert provider.fake.upload_files == [f"{shot.id}_a01.png"]
    assert st.attempts[-1].regressed is True
    assert st.selected_attempt == 1
    assert st.status is AssetStatus.EXHAUSTED


def test_repair_still_broll_falls_back_from_t2i_to_still_and_continue_bypasses_handoff(
    project, analysis, sample_png, monkeypatch
) -> None:
    project.config.llm.enabled = False
    # Strip reference_image from t2i so B-roll must fall back to 'still'.
    wf_dir = project.path("workflows")
    t2i_bpath = wf_dir / "t2i.bindings.json"
    t2i_bindings = json.loads(t2i_bpath.read_text())
    t2i_bindings["inputs"].pop("reference_image", None)
    t2i_bpath.write_text(json.dumps(t2i_bindings))

    manifest = build_manifest(project.config, analysis, "lyrics", "style", llm=None)
    broll = manifest.shots[0]
    broll.type = ShotType.BROLL
    broll.character = None
    broll.characters = []
    broll.continuity = ContinuityMode.CONTINUE
    _seed_attempt(project, broll, 1, sample_png, passed=True, score=0.85, select=True)

    custom, _, _ = _custom_project(project, manifest)
    provider = _LeaseRecorder(FakeComfy(png=sample_png))
    _patch_gates(monkeypatch, [True])

    res = api.repair_still(
        custom,
        broll.id,
        instruction="warm up window glow",
        skip_preflight=True,
        comfy_provider=provider,
    )
    updated = res.manifest.shot(broll.id)
    assert updated.still.attempts[-1].strategy is RetryStrategy.REPAIR_STILL
    assert provider.fake.upload_files == [f"{broll.id}_a01.png"]


def test_repair_still_anti_regression_allows_improved_score_for_explicit_instruction(
    project, analysis, sample_png, monkeypatch
) -> None:
    project.config.llm.enabled = False
    manifest = build_manifest(project.config, analysis, "lyrics", "style", llm=None)
    shot = next(s for s in manifest.shots if s.type is ShotType.LIPSYNC)
    _seed_attempt(
        project,
        shot,
        1,
        sample_png,
        passed=False,
        score=0.40,
        checks=[
            CheckResult(
                name="vision_artifacts",
                status=CheckStatus.FAILED,
                hard=False,
                detail="blurry left hand",
            )
        ],
        select=True,
    )
    custom, _, _ = _custom_project(project, manifest)
    provider = _LeaseRecorder(FakeComfy(png=sample_png))
    # Edit still fails with 1 issue, but score improved 0.40 -> 0.65 (no new hard failure).
    improved_checks = [
        CheckResult(
            name="vision_artifacts",
            status=CheckStatus.FAILED,
            hard=False,
            detail="slight finger curl",
        )
    ]
    _patch_gates(monkeypatch, [QCReport(passed=False, score=0.65, checks=improved_checks)])

    res = api.repair_still(
        custom,
        shot.id,
        instruction="fix left hand fingers",
        skip_preflight=True,
        comfy_provider=provider,
    )
    st = res.manifest.shot(shot.id).still
    assert st.attempts[-1].regressed is False
    assert st.selected_attempt == 2


def test_repair_still_error_cases_fail_fast_before_leasing(
    project, analysis, sample_png, tmp_path
) -> None:
    project.config.llm.enabled = False
    manifest = build_manifest(project.config, analysis, "lyrics", "style", llm=None)
    shot = next(s for s in manifest.shots if s.type is ShotType.LIPSYNC)
    custom, _, _ = _custom_project(project, manifest)
    provider = _LeaseRecorder(FakeComfy(png=sample_png))

    # 1. No existing still file on disk.
    with pytest.raises(ConfigError, match="no active still image"):
        api.repair_still(custom, shot.id, "fix", comfy_provider=provider)
    assert provider.trace == []

    # 2. Invalid strategy / unknown shot / NUL byte / bad attempts.
    with pytest.raises(ConfigError, match="Invalid still regeneration strategy"):
        api.regenerate_still(custom, shot.id, strategy="bogus", comfy_provider=provider)  # type: ignore[arg-type]
    with pytest.raises(MVDError, match="No shot with id"):
        api.repair_still(custom, "no_such_shot", "fix", comfy_provider=provider)
    with pytest.raises(ConfigError, match="NUL bytes"):
        api.repair_still(custom, shot.id, "bad\x00note", comfy_provider=provider)
    with pytest.raises(ConfigError, match="attempts must be an integer >= 1"):
        api.repair_still(custom, shot.id, "fix", attempts=0, comfy_provider=provider)
    assert provider.trace == []

    # 3. Neither 'still' nor 't2i' accepts reference_image.
    _seed_attempt(project, shot, 1, sample_png, passed=True, select=True)
    custom.save_manifest(manifest)
    wf_dir = project.path("workflows")
    for wf_name in ("still", "t2i"):
        bpath = wf_dir / f"{wf_name}.bindings.json"
        data = json.loads(bpath.read_text())
        data["inputs"].pop("reference_image", None)
        bpath.write_text(json.dumps(data))
    # Also verify graph doesn't break and error triggers cleanly before lease.
    with pytest.raises(ConfigError, match="accepts 'reference_image'"):
        api.repair_still(custom, shot.id, "fix", comfy_provider=provider)
    assert provider.trace == []

    # 4. RetryPolicy.prepare with pending REPAIR_STILL and can_repair=False raises ConfigError.
    policy = RetryPolicy(project.config.retry, project.config.seed, llm=None)
    shot.still.pending_strategy = RetryStrategy.REPAIR_STILL
    with pytest.raises(ConfigError, match="reference_image"):
        policy.prepare(shot, AssetKind.STILL, can_repair=False)


def test_regenerate_still_recreate_overrides_in_place_repair_and_clears_overrides(
    project, analysis, sample_png, sample_mp4, tmp_path, monkeypatch
) -> None:
    project.config.llm.enabled = False
    project.config.retry.in_place_repair = True
    manifest = build_manifest(project.config, analysis, "lyrics", "style", llm=None)
    shot = next(s for s in manifest.shots if s.type is ShotType.LIPSYNC)

    ref_img = tmp_path / "ref.png"
    ref_img.write_bytes(sample_png)
    reference.lock_reference(project, manifest, ref_img)

    # Seed a repairable failed still + stale overrides + passed video.
    _seed_attempt(
        project,
        shot,
        1,
        sample_png,
        passed=False,
        score=0.4,
        checks=[
            CheckResult(
                name="vision_artifacts",
                status=CheckStatus.FAILED,
                detail="extra finger",
                fix_hint="relax hand",
            )
        ],
        select=True,
    )
    shot.still.prompt_override = "stale override"
    shot.still.param_overrides = {"steps": 99}
    _seed_video(project, shot, sample_mp4)

    custom, _, _ = _custom_project(project, manifest)
    provider = _LeaseRecorder(FakeComfy(png=sample_png))
    _patch_gates(monkeypatch, [True, True])

    # Strategy="recreate" without instruction -> NEW_SEED, uploads character reference.
    res1 = api.regenerate_still(
        custom,
        shot.id,
        strategy="recreate",
        instruction="",
        skip_preflight=True,
        comfy_provider=provider,
    )
    st1 = res1.manifest.shot(shot.id).still
    assert st1.attempts[-1].strategy is RetryStrategy.NEW_SEED
    assert st1.param_overrides == {}
    assert res1.manifest.shot(shot.id).video.status is AssetStatus.PENDING
    assert provider.fake.upload_files == ["base_front.png"]

    # Strategy="recreate" with instruction -> REWRITE_PROMPT (never REPAIR_STILL).
    res2 = api.regenerate_still(
        custom,
        shot.id,
        strategy="recreate",
        instruction="pull camera back slightly",
        skip_preflight=True,
        comfy_provider=provider,
    )
    st2 = res2.manifest.shot(shot.id).still
    assert st2.attempts[-1].strategy is RetryStrategy.REWRITE_PROMPT
    assert "pull camera back slightly" in (st2.prompt_override or "")

    # Strategy="repair" via regenerate_still matches repair_still.
    res3 = api.regenerate_still(
        custom,
        shot.id,
        strategy="repair",
        instruction="soften key light",
        skip_preflight=True,
        comfy_provider=provider,
    )
    assert res3.manifest.shot(shot.id).still.attempts[-1].strategy is RetryStrategy.REPAIR_STILL


def test_api_generate_loads_still_for_pending_broll_repair_and_cli_smoke(
    project, analysis, sample_png, monkeypatch
) -> None:
    project.config.llm.enabled = False
    # Remove reference_image from t2i so api.generate must load 'still' for a B-roll pending repair.
    wf_dir = project.path("workflows")
    t2i_bpath = wf_dir / "t2i.bindings.json"
    t2i_bindings = json.loads(t2i_bpath.read_text())
    t2i_bindings["inputs"].pop("reference_image", None)
    t2i_bpath.write_text(json.dumps(t2i_bindings))

    # Keep workflow graph simple with valid json.
    assert json.loads((wf_dir / "graph.api.json").read_text()) == WORKFLOW_GRAPH

    manifest = build_manifest(project.config, analysis, "lyrics", "style", llm=None)
    broll = manifest.shots[0]
    broll.type = ShotType.BROLL
    broll.character = None
    broll.characters = []
    _seed_attempt(project, broll, 1, sample_png, passed=True, select=True)
    broll.still.reset(1)
    broll.still.pending_strategy = RetryStrategy.REPAIR_STILL
    broll.still.repair_source_attempt = 1
    broll.still.repair_instruction = "add warm mist"

    custom, _, _ = _custom_project(project, manifest)
    provider = _LeaseRecorder(FakeComfy(png=sample_png))
    _patch_gates(monkeypatch, [True])

    res = api.generate(
        custom,
        [AssetKind.STILL],
        shots=[broll.id],
        review_mode=ReviewMode.STRICT,
        skip_preflight=True,
        comfy_provider=provider,
    )
    assert res.manifest.shot(broll.id).still.attempts[-1].strategy is RetryStrategy.REPAIR_STILL

    # CLI invalid strategy fails cleanly.
    bad_cli = CliRunner().invoke(
        cli_mod.app,
        ["regenerate-still", broll.id, "--strategy", "invalid", "-p", str(project.root)],
    )
    assert bad_cli.exit_code != 0


def test_repair_still_preserves_repair_intent_across_transient_comfy_errors(
    project, analysis, sample_png, monkeypatch
) -> None:
    project.config.llm.enabled = False
    manifest = build_manifest(project.config, analysis, "lyrics", "style", llm=None)
    shot = next(s for s in manifest.shots if s.type is ShotType.LIPSYNC)
    # Intentionally omit character reference sheet: retry must remain REPAIR_STILL, not recreate.
    _seed_attempt(project, shot, 1, sample_png, passed=True, score=0.9, select=True)

    custom, _, _ = _custom_project(project, manifest)
    fake = FakeComfy(png=sample_png)
    fake.fail_next_jobs = 1
    provider = _LeaseRecorder(fake)
    _patch_gates(monkeypatch, [True])

    res = api.repair_still(
        custom,
        shot.id,
        instruction="fix earring",
        attempts=2,
        skip_preflight=True,
        comfy_provider=provider,
    )
    st = res.manifest.shot(shot.id).still
    assert len(st.attempts) == 3
    assert st.attempts[1].strategy is RetryStrategy.REPAIR_STILL
    assert st.attempts[1].error is not None
    assert st.attempts[2].strategy is RetryStrategy.REPAIR_STILL
    assert st.attempts[2].error is None
    assert "Apply ONLY this edit: fix earring" in st.attempts[2].prompt
    assert fake.upload_files == [f"{shot.id}_a01.png"]
    assert len(fake.queued) == 2
    assert fake.queued[1]["3"]["inputs"]["image"] == fake.queued[0]["3"]["inputs"]["image"]
    assert st.status is AssetStatus.PASSED
    assert st.selected_attempt == 3
    assert st.pending_strategy is None
    assert st.repair_instruction is None
    assert st.repair_source_attempt is None

    # Double transient failure exhausts/breaks and still clears pending overrides cleanly.
    fake2 = FakeComfy(png=sample_png)
    fake2.fail_next_jobs = 2
    res2 = api.repair_still(
        custom,
        shot.id,
        instruction="fix earring again",
        attempts=2,
        skip_preflight=True,
        comfy_provider=_LeaseRecorder(fake2),
    )
    st2 = res2.manifest.shot(shot.id).still
    assert st2.attempts[-2].strategy is RetryStrategy.REPAIR_STILL
    assert st2.attempts[-1].strategy is RetryStrategy.REPAIR_STILL
    assert st2.pending_strategy is None
    assert st2.repair_instruction is None
    assert st2.repair_source_attempt is None


def test_api_generate_mix_mode_loads_still_for_pending_repair(
    project, sample_png, monkeypatch
) -> None:
    project.config.llm.enabled = False
    project.config.mode = ProjectMode.MIX
    wf_dir = project.path("workflows")
    t2i_bpath = wf_dir / "t2i.bindings.json"
    t2i_bindings = json.loads(t2i_bpath.read_text())
    t2i_bindings["inputs"].pop("reference_image", None)
    t2i_bpath.write_text(json.dumps(t2i_bindings))

    shot = Shot(
        id="ch01_c01",
        index=0,
        start=0.0,
        end=4.0,
        type=ShotType.BROLL,
        framing=Framing.WIDE,
        setting="Hall",
        still_prompt="empty alley at dusk",
        motion_prompt="slow push in",
        audio_slice=AudioSlice(path="a.wav", start=0.0, end=4.0, head_handle=0.0),
        num_frames=97,
        still=AssetState(attempts_allowed=1),
        video=AssetState(attempts_allowed=1),
    )
    manifest = Manifest(
        mode=ProjectMode.MIX,
        title="Mix Repair",
        duration=4.0,
        fps=24,
        visual_bible=VisualBible(style="35mm"),
        shots=[shot],
    )
    _seed_attempt(project, shot, 1, sample_png, passed=True, select=True)
    shot.still.reset(1)
    shot.still.pending_strategy = RetryStrategy.REPAIR_STILL
    shot.still.repair_source_attempt = 1
    shot.still.repair_instruction = "remove power lines"

    custom, _, _ = _custom_project(project, manifest)
    provider = _LeaseRecorder(FakeComfy(png=sample_png))
    _patch_gates(monkeypatch, [True])

    res = api.generate(
        custom,
        [AssetKind.STILL],
        shots=[shot.id],
        review_mode=ReviewMode.STRICT,
        skip_preflight=True,
        comfy_provider=provider,
    )
    updated = res.manifest.shot(shot.id).still
    assert updated.status is AssetStatus.PASSED
    assert updated.attempts[-1].strategy is RetryStrategy.REPAIR_STILL
    assert "remove power lines" in updated.attempts[-1].prompt
    assert provider.fake.upload_files == [f"{shot.id}_a01.png"]


def test_repair_still_budget_exceeded_preserves_existing_state_when_zero_attempts_run(
    project, analysis, sample_png, sample_mp4, monkeypatch
) -> None:
    # Avoid native MediaPipe (needs libEGL on Linux CI); no attempt runs, so QC is never used.
    monkeypatch.setattr(
        api,
        "build_quality_gates",
        lambda _config, _llm: QualityGates(
            image=ScriptedGate([True]), video=ScriptedGate([True]), face_analyzer=None
        ),
    )
    project.config.llm.enabled = False
    project.config.retry.max_generations_per_run = 0
    manifest = build_manifest(project.config, analysis, "lyrics", "style", llm=None)
    shot = next(s for s in manifest.shots if s.type is ShotType.LIPSYNC)
    _seed_attempt(project, shot, 1, sample_png, passed=True, score=0.91, select=True)
    _seed_video(project, shot, sample_mp4)
    review_policy.approve(manifest, shot.id, stills_only=True)

    custom, state, _ = _custom_project(project, manifest)
    provider = _LeaseRecorder(FakeComfy(png=sample_png))

    res = api.repair_still(
        custom,
        shot.id,
        instruction="fix hat",
        skip_preflight=True,
        comfy_provider=provider,
    )
    assert res.budget_message is not None
    for checked_shot in (res.manifest.shot(shot.id), state.manifest.shot(shot.id)):
        assert len(checked_shot.still.attempts) == 1
        assert checked_shot.still.status is AssetStatus.PASSED
        assert checked_shot.still.selected_attempt == 1
        assert checked_shot.still.pending_strategy is None
        assert checked_shot.still.repair_instruction is None
        assert checked_shot.still.repair_source_attempt is None
        assert checked_shot.video.status is AssetStatus.PASSED
        assert checked_shot.video.selected_attempt == 1
        assert checked_shot.review.decision is ReviewDecision.APPROVED
