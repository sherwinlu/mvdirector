# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Track 2 benchmark: Vision QC replay, threshold calibration, and Net Effective Cost."""

from __future__ import annotations

import base64
import csv
import hashlib
import html
import io
import json
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from pydantic import BaseModel, Field

from mvdirector.bench.config import (
    BenchmarkConfig,
    GPUCostConfig,
    ModelArm,
    open_bench_project,
    resolve_bench_environ,
)
from mvdirector.bench.dataset import (
    FailureTag,
    QCLabelItem,
    resolve_still_and_references,
)
from mvdirector.bench.planning import md_cell
from mvdirector.bench.telemetry import (
    CallRecord,
    InstrumentedLLM,
    TelemetrySummary,
    summarize_records,
)
from mvdirector.calibrate import _best_cut
from mvdirector.config import ImageQCConfig
from mvdirector.errors import ConfigError, ImagesNotSupportedError, LLMError
from mvdirector.llm.base import is_refusal_error
from mvdirector.llm.factory import build_llm
from mvdirector.models import AssetKind, Manifest
from mvdirector.project import Project
from mvdirector.qc.vision import VisionJudge
from mvdirector.store import atomic_write

log = logging.getLogger(__name__)

_DEFAULT_IMAGE_QC = ImageQCConfig()
_DEFAULT_THRESHOLDS = {
    "vision_spec": _DEFAULT_IMAGE_QC.vision_spec_hard_min,
    "vision_artifacts": _DEFAULT_IMAGE_QC.vision_hard_min,
    "vision_character": _DEFAULT_IMAGE_QC.vision_character_hard_min,
}
_CHECK_TO_TAG: dict[str, FailureTag] = {
    "vision_spec": "spec",
    "vision_artifacts": "artifact",
    "vision_character": "character",
}


class CheckCalibrationMetrics(BaseModel):
    """Per-check balanced accuracy and optimal cut matching ``mvd calibrate``."""

    check: str
    good_count: int = 0
    bad_count: int = 0
    optimal_threshold: float | None = None
    balanced_accuracy: float | None = None
    good_median: float | None = None
    bad_median: float | None = None
    # Tag-targeted calibration (e.g. vision_spec evaluated specifically against 'spec' failure tags)
    tag_name: str = ""
    tag_good_count: int = 0
    tag_bad_count: int = 0
    tag_optimal_threshold: float | None = None
    tag_balanced_accuracy: float | None = None


class GateOutcomeMetrics(BaseModel):
    """Combined hard-gate classification performance and Net Effective Cost / 90-shot Video."""

    spec_threshold: float
    artifacts_threshold: float
    character_threshold: float
    tp: int = 0
    tn: int = 0
    fp: int = 0  # Good still wrongly rejected -> wasted still GPU retry
    fn: int = 0  # Bad still wrongly approved -> wasted video GPU render
    balanced_accuracy: float = 0.0
    fp_rate: float = 0.0
    fn_rate: float = 0.0
    # Scaled to gpu_costs.shots_per_video (default 90)
    shots_per_video: int = 90
    llm_qc_spend_usd: float = 0.0
    expected_fp_count: float = 0.0
    expected_fn_count: float = 0.0
    fp_still_gpu_cost_usd: float = 0.0
    fn_video_gpu_cost_usd: float = 0.0
    net_effective_cost_usd: float = 0.0


class QCItemVerdict(BaseModel):
    """One model arm's replay verdict on one labeled still."""

    item_id: str
    project: str
    shot_id: str
    attempt: int | None = None
    still_path: str
    reference_image: str | None = None
    ground_truth_good: bool
    failure_tags: list[FailureTag] = Field(default_factory=list)
    notes: str = ""
    vision_spec: float | None = None
    vision_artifacts: float | None = None
    vision_character: float | None = None
    issues: list[str] = Field(default_factory=list)
    fix: str = ""
    refused: bool = False
    error: str | None = None
    latency_seconds: float = 0.0
    estimated_usd: float = 0.0
    cache_fingerprint: str = ""
    call: CallRecord | None = None


class VisionQCArmResult(BaseModel):
    """Aggregated Track 2 Vision QC benchmark outcome for one model arm."""

    arm_id: str
    arm_label: str
    provider: str
    model: str
    blind_observer: bool = False
    cache_fingerprint: str = ""
    total_items: int = 0
    evaluated_items: int = 0
    refusals: int = 0
    errors: int = 0
    refusal_rate: float = 0.0
    wall_seconds: float = 0.0
    telemetry: TelemetrySummary = Field(default_factory=TelemetrySummary)
    per_check: dict[str, CheckCalibrationMetrics] = Field(default_factory=dict)
    default_gate: GateOutcomeMetrics | None = None
    calibrated_gate: GateOutcomeMetrics | None = None
    verdicts: list[QCItemVerdict] = Field(default_factory=list)


def _calibrate_check(
    check_name: str,
    verdicts: Sequence[QCItemVerdict],
) -> CheckCalibrationMetrics:
    tag = _CHECK_TO_TAG[check_name]
    overall_good: list[float] = []
    overall_bad: list[float] = []
    tag_good: list[float] = []
    tag_bad: list[float] = []

    for v in verdicts:
        score = getattr(v, check_name)
        if score is None:
            continue
        if v.ground_truth_good:
            overall_good.append(float(score))
            tag_good.append(float(score))
        else:
            overall_bad.append(float(score))
            if tag in v.failure_tags:
                tag_bad.append(float(score))
            elif v.failure_tags:
                # Bad for a *different* reason (e.g. character mismatch when spec was clean).
                tag_good.append(float(score))

    metrics = CheckCalibrationMetrics(
        check=check_name,
        good_count=len(overall_good),
        bad_count=len(overall_bad),
        tag_name=tag,
        tag_good_count=len(tag_good),
        tag_bad_count=len(tag_bad),
    )
    if overall_good and overall_bad:
        g_arr = np.array(overall_good, dtype=np.float64)
        b_arr = np.array(overall_bad, dtype=np.float64)
        cut, acc = _best_cut(g_arr, b_arr, higher_is_better=True)
        metrics.optimal_threshold = cut
        metrics.balanced_accuracy = acc
        metrics.good_median = round(float(np.median(g_arr)), 4)
        metrics.bad_median = round(float(np.median(b_arr)), 4)
    if tag_good and tag_bad:
        tg_arr = np.array(tag_good, dtype=np.float64)
        tb_arr = np.array(tag_bad, dtype=np.float64)
        t_cut, t_acc = _best_cut(tg_arr, tb_arr, higher_is_better=True)
        metrics.tag_optimal_threshold = t_cut
        metrics.tag_balanced_accuracy = t_acc
    return metrics


def _verdict_passes_gate(
    v: QCItemVerdict,
    *,
    spec_min: float,
    artifacts_min: float,
    character_min: float,
) -> bool:
    """True when ``v`` passes the combined hard gate (refused/errored/missing calls pass)."""
    if v.error is not None or v.refused or v.vision_spec is None or v.vision_artifacts is None:
        return True
    return (
        v.vision_spec >= spec_min
        and v.vision_artifacts >= artifacts_min
        and (v.vision_character is None or v.vision_character >= character_min)
    )


def _qc_cache_fingerprint(
    arm: ModelArm,
    *,
    blind_observer: bool,
    include_plates: bool,
) -> str:
    payload = {
        "provider": arm.provider,
        "model": arm.model,
        "blind_observer": blind_observer,
        "include_plates": include_plates,
        "max_tokens": arm.max_tokens,
        "json_mode": arm.json_mode,
        "base_url": arm.base_url,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def _load_compatible_cached_verdicts(
    arm_dir: Path,
    arm: ModelArm,
    *,
    blind_observer: bool,
    include_plates: bool,
) -> dict[str, QCItemVerdict]:
    """Load cached non-error verdicts matching the current arm/mode fingerprint."""
    verdicts_file = arm_dir / "verdicts.json"
    if not verdicts_file.is_file():
        return {}
    expected_fp = _qc_cache_fingerprint(
        arm, blind_observer=blind_observer, include_plates=include_plates
    )
    metrics_file = arm_dir / "metrics.json"
    if metrics_file.is_file():
        try:
            prev_metrics = VisionQCArmResult.model_validate_json(
                metrics_file.read_text(encoding="utf-8")
            )
            if (
                prev_metrics.provider != arm.provider
                or prev_metrics.model != arm.model
                or prev_metrics.blind_observer != blind_observer
                or (
                    prev_metrics.cache_fingerprint and prev_metrics.cache_fingerprint != expected_fp
                )
            ):
                return {}
        except (OSError, ValueError):
            pass
    cached_by_id: dict[str, QCItemVerdict] = {}
    try:
        raw_list = json.loads(verdicts_file.read_text(encoding="utf-8"))
        if isinstance(raw_list, list):
            for entry in raw_list:
                parsed = QCItemVerdict.model_validate(entry)
                # Never reuse transient non-refusal errors; retry them on re-run.
                if parsed.error is not None and not parsed.refused:
                    continue
                if parsed.cache_fingerprint and parsed.cache_fingerprint != expected_fp:
                    continue
                cached_by_id[parsed.item_id] = parsed
    except (OSError, ValueError):
        return {}
    return cached_by_id


def evaluate_hard_gate(
    verdicts: Sequence[QCItemVerdict],
    *,
    spec_min: float,
    artifacts_min: float,
    character_min: float,
    mean_llm_usd_per_still: float,
    gpu_costs: GPUCostConfig,
) -> GateOutcomeMetrics:
    """Compute combined hard-gate FP/FN rates, balanced accuracy, and Net Effective Cost.

    Matches ``mvdirector.qc.common.vision_checks``: a still passes the vision hard gate when
    no active vision check drops below its hard minimum (note that refused/errored vision calls
    are skipped in production and therefore do not block an escaping bad still).
    """
    tp = tn = fp = fn = 0
    for v in verdicts:
        predicted_pass = _verdict_passes_gate(
            v,
            spec_min=spec_min,
            artifacts_min=artifacts_min,
            character_min=character_min,
        )
        if v.ground_truth_good:
            if predicted_pass:
                tp += 1
            else:
                fp += 1
        else:
            if predicted_pass:
                fn += 1
            else:
                tn += 1

    good_total = tp + fp
    bad_total = tn + fn
    total = good_total + bad_total
    tpr = tp / good_total if good_total > 0 else 0.0
    tnr = tn / bad_total if bad_total > 0 else 0.0
    fp_rate = fp / good_total if good_total > 0 else 0.0
    fn_rate = fn / bad_total if bad_total > 0 else 0.0
    bal_acc = (tpr + tnr) / 2.0 if (good_total > 0 and bad_total > 0) else (tpr or tnr)

    shots = gpu_costs.shots_per_video
    scale = (shots / total) if total > 0 else 0.0
    expected_fp = fp * scale
    expected_fn = fn * scale
    llm_spend = mean_llm_usd_per_still * shots
    fp_gpu = expected_fp * gpu_costs.still_usd
    fn_gpu = expected_fn * gpu_costs.video_usd
    net_cost = llm_spend + fp_gpu + fn_gpu

    return GateOutcomeMetrics(
        spec_threshold=round(spec_min, 4),
        artifacts_threshold=round(artifacts_min, 4),
        character_threshold=round(character_min, 4),
        tp=tp,
        tn=tn,
        fp=fp,
        fn=fn,
        balanced_accuracy=round(bal_acc, 4),
        fp_rate=round(fp_rate, 4),
        fn_rate=round(fn_rate, 4),
        shots_per_video=shots,
        llm_qc_spend_usd=round(llm_spend, 4),
        expected_fp_count=round(expected_fp, 2),
        expected_fn_count=round(expected_fn, 2),
        fp_still_gpu_cost_usd=round(fp_gpu, 4),
        fn_video_gpu_cost_usd=round(fn_gpu, 4),
        net_effective_cost_usd=round(net_cost, 4),
    )


def summarize_vision_qc_arm(
    arm: ModelArm,
    verdicts: list[QCItemVerdict],
    *,
    gpu_costs: GPUCostConfig,
    wall_seconds: float,
    blind_observer: bool = False,
    cache_fingerprint: str = "",
) -> VisionQCArmResult:
    """Compute per-check calibration, combined gate accuracy, and Net Effective Cost for an arm."""
    calls = [v.call for v in verdicts if v.call is not None]
    telemetry = summarize_records(calls)
    total = len(verdicts)
    refusals = sum(1 for v in verdicts if v.refused)
    errors = sum(1 for v in verdicts if v.error is not None and not v.refused)
    evaluated = sum(
        1 for v in verdicts if not v.refused and v.error is None and v.vision_spec is not None
    )

    per_check = {
        name: _calibrate_check(name, verdicts)
        for name in ("vision_spec", "vision_artifacts", "vision_character")
    }
    mean_usd_per_still = (sum(v.estimated_usd for v in verdicts) / total) if total > 0 else 0.0

    default_gate = evaluate_hard_gate(
        verdicts,
        spec_min=_DEFAULT_THRESHOLDS["vision_spec"],
        artifacts_min=_DEFAULT_THRESHOLDS["vision_artifacts"],
        character_min=_DEFAULT_THRESHOLDS["vision_character"],
        mean_llm_usd_per_still=mean_usd_per_still,
        gpu_costs=gpu_costs,
    )

    def _pick_opt(name: str) -> float:
        m = per_check[name]
        if m.tag_optimal_threshold is not None:
            return m.tag_optimal_threshold
        if m.optimal_threshold is not None:
            return m.optimal_threshold
        return _DEFAULT_THRESHOLDS[name]

    calibrated_gate = evaluate_hard_gate(
        verdicts,
        spec_min=_pick_opt("vision_spec"),
        artifacts_min=_pick_opt("vision_artifacts"),
        character_min=_pick_opt("vision_character"),
        mean_llm_usd_per_still=mean_usd_per_still,
        gpu_costs=gpu_costs,
    )

    return VisionQCArmResult(
        arm_id=arm.id,
        arm_label=arm.label,
        provider=arm.provider,
        model=arm.model,
        blind_observer=blind_observer,
        cache_fingerprint=cache_fingerprint,
        total_items=total,
        evaluated_items=evaluated,
        refusals=refusals,
        errors=errors,
        refusal_rate=round(refusals / total, 4) if total > 0 else 0.0,
        wall_seconds=round(wall_seconds, 3),
        telemetry=telemetry,
        per_check=per_check,
        default_gate=default_gate,
        calibrated_gate=calibrated_gate,
        verdicts=verdicts,
    )


def run_vision_qc_arm(
    dataset: Sequence[QCLabelItem],
    arm: ModelArm,
    out_root: Path,
    *,
    gpu_costs: GPUCostConfig,
    blind_observer: bool = False,
    include_plates: bool = False,
    overwrite: bool = False,
    max_budget_usd: float | None = None,
    environ: Mapping[str, str] | None = None,
    llm_factory: Callable[[ModelArm, Mapping[str, str] | None], Any] | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> VisionQCArmResult:
    """Replay all labeled stills in ``dataset`` through ``arm`` using ``VisionJudge``."""
    arm_dir = out_root / "vision_qc" / arm.id
    arm_dir.mkdir(parents=True, exist_ok=True)
    verdicts_file = arm_dir / "verdicts.json"
    cache_fp = _qc_cache_fingerprint(
        arm, blind_observer=blind_observer, include_plates=include_plates
    )

    cached_by_id: dict[str, QCItemVerdict] = (
        {}
        if overwrite
        else _load_compatible_cached_verdicts(
            arm_dir, arm, blind_observer=blind_observer, include_plates=include_plates
        )
    )

    project_cache: dict[str, tuple[Project, Manifest]] = {}

    def _open_cached(proj_str: str) -> tuple[Project, Manifest]:
        if proj_str not in project_cache:
            proj = open_bench_project(Path(proj_str))
            project_cache[proj_str] = (proj, proj.load_manifest())
        return project_cache[proj_str]

    # Resolve all items upfront so missing files fail fast before spending API tokens.
    resolved_items = []
    for item in dataset:
        proj, manifest = _open_cached(item.project)
        shot, still_path, ref_img, extra_refs = resolve_still_and_references(
            proj, manifest, item, include_plates=include_plates
        )
        resolved_items.append((item, manifest, shot, still_path, ref_img, extra_refs))

    started = time.monotonic()
    need_live = any(item.id not in cached_by_id for item, *_ in resolved_items) and (
        max_budget_usd is None or max_budget_usd > 0.0
    )

    raw_llm = None
    instrumented: InstrumentedLLM | None = None
    verdicts: list[QCItemVerdict] = []
    live_spent_usd = 0.0
    budget_hit = False

    try:
        if need_live:
            first_proj = next(iter(project_cache.values()))[0]
            effective_env = resolve_bench_environ(first_proj.environ(), environ)
            raw_llm = (
                llm_factory(arm, effective_env)
                if llm_factory is not None
                else build_llm(arm.to_llm_config(environ=effective_env), environ=effective_env)
            )
            if raw_llm is None:
                raise ConfigError(f"Arm {arm.id} produced no LLM client.")
            instrumented = InstrumentedLLM(raw_llm, arm)
            judge = VisionJudge(instrumented, blind_observer=blind_observer)
        else:
            judge = None

        for idx, (item, manifest, shot, still_path, ref_img, extra_refs) in enumerate(
            resolved_items, start=1
        ):
            if item.id in cached_by_id:
                cached = cached_by_id[item.id]
                # Refresh ground-truth label/tags from CSV so manual label edits rescore for $0.
                verdicts.append(
                    cached.model_copy(
                        update={
                            "ground_truth_good": item.good,
                            "failure_tags": list(item.tags),
                            "notes": item.notes,
                        }
                    )
                )
                continue
            if max_budget_usd is not None and live_spent_usd >= max_budget_usd:
                if on_progress and not budget_hit:
                    on_progress(
                        f"Budget cap reached during Arm {arm.id} "
                        f"(${live_spent_usd:.4f} >= ${max_budget_usd:.2f}); "
                        "skipping remaining uncached stills."
                    )
                budget_hit = True
                continue
            assert instrumented is not None and judge is not None
            if on_progress:
                on_progress(f"Vision QC [{arm.id}] ({idx}/{len(resolved_items)}) {item.id}")

            before_count = len(instrumented.records)
            item_started = time.monotonic()
            v_spec: float | None = None
            v_art: float | None = None
            v_char: float | None = None
            issues: list[str] = []
            fix = ""
            refused = False
            error_text: str | None = None
            try:
                with instrumented.stage(f"vision_qc:{item.id}"):
                    verdict = judge.judge(
                        [still_path],
                        shot,
                        manifest,
                        AssetKind.STILL,
                        reference_image=ref_img,
                        extra_references=extra_refs or None,
                    )
                v_spec = round(float(verdict.spec_match), 4)
                v_art = round(float(verdict.artifact_free), 4)
                v_char = (
                    round(float(verdict.character_match), 4)
                    if verdict.character_match is not None
                    else None
                )
                issues = list(verdict.issues)
                fix = verdict.fix
            except ImagesNotSupportedError as exc:
                error_text = str(exc)
            except LLMError as exc:
                error_text = str(exc)
                refused = is_refusal_error(exc)

            new_calls = instrumented.records[before_count:]
            if any(c.refused for c in new_calls):
                refused = True
            item_usd = round(sum(c.estimated_usd for c in new_calls), 6)
            item_lat = round(max(0.0, time.monotonic() - item_started), 4)
            primary_call = new_calls[-1] if new_calls else None
            if len(new_calls) > 1:
                # Combine multi-pass blind-observer calls into one item-level summary record.
                primary_call = CallRecord(
                    stage=f"vision_qc:{item.id}",
                    schema_name="VisionVerdict",
                    model=arm.model,
                    latency_seconds=item_lat,
                    attempts=sum(c.attempts for c in new_calls),
                    schema_retries=sum(c.schema_retries for c in new_calls),
                    first_pass_ok=all(c.first_pass_ok for c in new_calls),
                    succeeded=all(c.succeeded for c in new_calls),
                    truncated=any(c.truncated for c in new_calls),
                    refused=refused,
                    has_audio=any(c.has_audio for c in new_calls),
                    has_images=any(c.has_images for c in new_calls),
                    input_tokens=sum(c.input_tokens for c in new_calls),
                    output_tokens=sum(c.output_tokens for c in new_calls),
                    thoughts_tokens=sum(c.thoughts_tokens for c in new_calls),
                    cached_tokens=sum(c.cached_tokens for c in new_calls),
                    estimated_usd=item_usd,
                    error=error_text,
                )

            item_verdict = QCItemVerdict(
                item_id=item.id,
                project=item.project,
                shot_id=item.shot_id,
                attempt=item.attempt,
                still_path=str(still_path),
                reference_image=str(ref_img) if ref_img else None,
                ground_truth_good=item.good,
                failure_tags=list(item.tags),
                notes=item.notes,
                vision_spec=v_spec,
                vision_artifacts=v_art,
                vision_character=v_char,
                issues=issues,
                fix=fix,
                refused=refused,
                error=error_text,
                latency_seconds=item_lat,
                estimated_usd=item_usd,
                cache_fingerprint=cache_fp,
                call=primary_call,
            )
            live_spent_usd += item_usd
            verdicts.append(item_verdict)
            if item_verdict.error is None or item_verdict.refused:
                cached_by_id[item.id] = item_verdict
            atomic_write(
                verdicts_file,
                json.dumps([v.model_dump(mode="json") for v in cached_by_id.values()], indent=2)
                + "\n",
            )
    finally:
        if instrumented is not None:
            instrumented.close()
        elif raw_llm is not None:
            closer = getattr(raw_llm, "close", None)
            if callable(closer):
                closer()

    wall = max(0.0, time.monotonic() - started)
    if not need_live:
        wall = sum(v.latency_seconds for v in verdicts)
    result = summarize_vision_qc_arm(
        arm,
        verdicts,
        gpu_costs=gpu_costs,
        wall_seconds=wall,
        blind_observer=blind_observer,
        cache_fingerprint=cache_fp,
    )
    atomic_write(arm_dir / "metrics.json", result.model_dump_json(indent=2) + "\n")
    return result


def load_vision_qc_results(qc_dir: Path) -> list[VisionQCArmResult]:
    """Load saved ``VisionQCArmResult`` files from ``<out>/vision_qc/<arm>/metrics.json``."""
    results: list[VisionQCArmResult] = []
    if not qc_dir.is_dir():
        return results
    for metrics_file in sorted(qc_dir.glob("*/metrics.json")):
        try:
            results.append(
                VisionQCArmResult.model_validate_json(metrics_file.read_text(encoding="utf-8"))
            )
        except (OSError, ValueError) as exc:
            log.warning("Skipping unreadable Vision QC metrics %s: %s", metrics_file, exc)
    return results


def run_vision_qc_benchmark(
    dataset: Sequence[QCLabelItem],
    config: BenchmarkConfig,
    out_root: Path,
    *,
    arm_ids: Sequence[str] | None = None,
    include_controls: bool = False,
    blind_observer: bool = False,
    include_plates: bool = False,
    overwrite: bool = False,
    max_budget_usd: float | None = None,
    environ: Mapping[str, str] | None = None,
    llm_factory: Callable[[ModelArm, Mapping[str, str] | None], Any] | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> list[VisionQCArmResult]:
    """Run Track 2 Vision QC across selected arms and write merged summary + error reports."""
    if not dataset:
        raise ConfigError("Vision QC dataset is empty.")
    arms = config.select_arms(arm_ids, include_controls=include_controls)
    dataset_ids = {item.id for item in dataset}
    results: list[VisionQCArmResult] = []
    spent_usd = 0.0
    budget_hit = False
    for arm in arms:
        arm_dir = out_root / "vision_qc" / arm.id
        prior_cached = (
            {}
            if overwrite
            else _load_compatible_cached_verdicts(
                arm_dir, arm, blind_observer=blind_observer, include_plates=include_plates
            )
        )
        fully_cached = dataset_ids.issubset(prior_cached)
        if max_budget_usd is not None and not fully_cached and spent_usd >= max_budget_usd:
            if on_progress and not budget_hit:
                on_progress(
                    f"Budget cap reached (${spent_usd:.4f} >= ${max_budget_usd:.2f}); "
                    "skipping remaining uncached arms."
                )
            budget_hit = True
            continue
        if on_progress:
            on_progress(f"Starting Track 2 Vision QC for Arm {arm.id} ({arm.model})...")
        remaining_budget = (
            max(0.0, max_budget_usd - spent_usd) if max_budget_usd is not None else None
        )
        res = run_vision_qc_arm(
            dataset,
            arm,
            out_root,
            gpu_costs=config.gpu_costs,
            blind_observer=blind_observer,
            include_plates=include_plates,
            overwrite=overwrite,
            max_budget_usd=remaining_budget,
            environ=environ,
            llm_factory=llm_factory,
            on_progress=on_progress,
        )
        spent_usd += sum(v.estimated_usd for v in res.verdicts if v.item_id not in prior_cached)
        results.append(res)
    merged = load_vision_qc_results(out_root / "vision_qc")
    write_vision_qc_reports(merged or results, out_root / "vision_qc")
    return results


def _format_check_cell(result: VisionQCArmResult, name: str) -> str:
    c = result.per_check.get(name)
    if c is None or c.balanced_accuracy is None:
        return "n/a"
    base = f"{c.balanced_accuracy:.3f} ({c.optimal_threshold:.2f})"
    if c.tag_balanced_accuracy is not None:
        base += f" [tag:{c.tag_balanced_accuracy:.3f}]"
    return base


def render_vision_qc_markdown(results: Sequence[VisionQCArmResult]) -> str:
    """Render Track 2 Vision QC results including balanced accuracy, FP/FN, and effective cost."""
    lines = [
        "# Track 2: Vision QC Benchmark Summary",
        "",
        "| Arm | Model | Spec Bal.Acc (cut) | Art. Bal.Acc (cut) | Char. Bal.Acc (cut) | "
        "Gate Bal.Acc (def / cal) | FP Rate (def / cal) | FN Rate (def / cal) | "
        "Refusal | Mean Lat (s) | QC API / 90s | **Net Eff. Cost / 90s (def / cal)** |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | "
        ":---: | :---: | :---: | :---: |",
    ]
    for r in results:
        dg = r.default_gate
        cg = r.calibrated_gate
        gate_acc = (
            f"{dg.balanced_accuracy:.3f} / {cg.balanced_accuracy:.3f}" if (dg and cg) else "n/a"
        )
        fp = f"{dg.fp_rate:.1%} / {cg.fp_rate:.1%}" if (dg and cg) else "n/a"
        fn = f"{dg.fn_rate:.1%} / {cg.fn_rate:.1%}" if (dg and cg) else "n/a"
        api_90 = f"${dg.llm_qc_spend_usd:.2f}" if dg else "n/a"
        net_90 = (
            f"**${dg.net_effective_cost_usd:.2f}** / **${cg.net_effective_cost_usd:.2f}**"
            if (dg and cg)
            else "n/a"
        )
        lines.append(
            f"| **{md_cell(r.arm_id)}** | `{md_cell(r.model)}` | "
            f"{_format_check_cell(r, 'vision_spec')} | "
            f"{_format_check_cell(r, 'vision_artifacts')} | "
            f"{_format_check_cell(r, 'vision_character')} | "
            f"{gate_acc} | {fp} | {fn} | {r.refusal_rate:.1%} ({r.refusals}) | "
            f"{r.telemetry.mean_latency_seconds:.2f} | {api_90} | {net_90} |"
        )
    return "\n".join(lines) + "\n"


def render_vision_qc_csv(results: Sequence[VisionQCArmResult]) -> str:
    buf = io.StringIO()
    fields = [
        "arm_id",
        "arm_label",
        "provider",
        "model",
        "total_items",
        "evaluated_items",
        "refusals",
        "refusal_rate",
        "errors",
        "mean_latency_seconds",
        "total_api_usd",
        "spec_bal_acc",
        "spec_opt_cut",
        "artifacts_bal_acc",
        "artifacts_opt_cut",
        "character_bal_acc",
        "character_opt_cut",
        "default_gate_bal_acc",
        "default_fp_rate",
        "default_fn_rate",
        "default_net_effective_cost_usd",
        "calibrated_gate_bal_acc",
        "calibrated_fp_rate",
        "calibrated_fn_rate",
        "calibrated_net_effective_cost_usd",
    ]
    writer = csv.DictWriter(buf, fieldnames=fields)
    writer.writeheader()
    for r in results:
        sp = r.per_check.get("vision_spec")
        ar = r.per_check.get("vision_artifacts")
        ch = r.per_check.get("vision_character")
        dg = r.default_gate
        cg = r.calibrated_gate
        writer.writerow(
            {
                "arm_id": r.arm_id,
                "arm_label": r.arm_label,
                "provider": r.provider,
                "model": r.model,
                "total_items": r.total_items,
                "evaluated_items": r.evaluated_items,
                "refusals": r.refusals,
                "refusal_rate": r.refusal_rate,
                "errors": r.errors,
                "mean_latency_seconds": r.telemetry.mean_latency_seconds,
                "total_api_usd": r.telemetry.estimated_usd,
                "spec_bal_acc": sp.balanced_accuracy if sp else None,
                "spec_opt_cut": sp.optimal_threshold if sp else None,
                "artifacts_bal_acc": ar.balanced_accuracy if ar else None,
                "artifacts_opt_cut": ar.optimal_threshold if ar else None,
                "character_bal_acc": ch.balanced_accuracy if ch else None,
                "character_opt_cut": ch.optimal_threshold if ch else None,
                "default_gate_bal_acc": dg.balanced_accuracy if dg else None,
                "default_fp_rate": dg.fp_rate if dg else None,
                "default_fn_rate": dg.fn_rate if dg else None,
                "default_net_effective_cost_usd": dg.net_effective_cost_usd if dg else None,
                "calibrated_gate_bal_acc": cg.balanced_accuracy if cg else None,
                "calibrated_fp_rate": cg.fp_rate if cg else None,
                "calibrated_fn_rate": cg.fn_rate if cg else None,
                "calibrated_net_effective_cost_usd": cg.net_effective_cost_usd if cg else None,
            }
        )
    return buf.getvalue()


def _thumbnail_data_uri(path: str, max_side: int = 320) -> str:
    p = Path(path)
    if not p.is_file():
        return ""
    try:
        with Image.open(p) as opened:
            max_pixels = Image.MAX_IMAGE_PIXELS
            if max_pixels is not None and (opened.width * opened.height) > max_pixels:
                return ""
            pil = opened.convert("RGB")
        pil.thumbnail((max_side, max_side))
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=75)
        encoded = base64.standard_b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"
    except (OSError, Image.DecompressionBombError, ValueError):
        return ""


def render_vision_qc_errors_html(results: Sequence[VisionQCArmResult]) -> str:
    """Render an inspectable static HTML report of each arm's FPs, FNs, and safety refusals."""
    sections: list[str] = []
    thumb_cache: dict[str, str] = {}

    def _thumb(path: str) -> str:
        if path not in thumb_cache:
            thumb_cache[path] = _thumbnail_data_uri(path)
        return thumb_cache[path]

    for r in results:
        dg = r.default_gate
        if dg is None:
            continue
        cards: list[str] = []
        for v in r.verdicts:
            if v.refused or v.error is not None:
                kind = "REFUSAL" if v.refused else "ERROR"
            else:
                passed = _verdict_passes_gate(
                    v,
                    spec_min=dg.spec_threshold,
                    artifacts_min=dg.artifacts_threshold,
                    character_min=dg.character_threshold,
                )
                if passed == v.ground_truth_good:
                    continue
                kind = "FP (good rejected)" if not passed else "FN (bad approved)"
            uri = _thumb(v.still_path)
            img_tag = f'<img src="{uri}" alt="{html.escape(v.item_id)}">' if uri else ""
            scores = f"spec={v.vision_spec} · art={v.vision_artifacts} · char={v.vision_character}"
            detail = v.error or "; ".join(v.issues) or "(no issues listed)"
            cards.append(
                f"<div class='card'>{img_tag}"
                f"<div class='meta'><b>{html.escape(kind)}</b> · "
                f"<code>{html.escape(v.item_id)}</code>"
                f"<br>Truth: {'PASS' if v.ground_truth_good else 'FAIL'} "
                f"{html.escape(', '.join(v.failure_tags))}"
                f"<br>{html.escape(scores)}"
                f"<br><span class='muted'>{html.escape(detail)}</span></div></div>"
            )
        summary_line = (
            f"Balanced Acc: {dg.balanced_accuracy:.3f} · FP: {dg.fp_rate:.1%} · "
            f"FN: {dg.fn_rate:.1%} · Refusals: {r.refusals} · "
            f"Net Effective Cost / {dg.shots_per_video} shots: ${dg.net_effective_cost_usd:.2f}"
        )
        body = "".join(cards) if cards else "<p class='muted'>No errors or refusals.</p>"
        sections.append(
            f"<section><h2>Arm {html.escape(r.arm_id)} — {html.escape(r.arm_label)} "
            f"(<code>{html.escape(r.model)}</code>)</h2>"
            f"<p class='muted'>{html.escape(summary_line)}</p>"
            f"<div class='grid'>{body}</div></section>"
        )

    css = (
        "body{font-family:-apple-system,BlinkMacSystemFont,sans-serif;"
        "background:#0f1117;color:#e8ecf5;margin:20px;}"
        ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));"
        "gap:12px;margin-bottom:28px;}"
        ".card{background:#181c26;border:1px solid #2e364a;border-radius:8px;overflow:hidden;}"
        ".card img{width:100%;height:180px;object-fit:cover;display:block;}"
        ".meta{padding:10px 12px;font-size:12.5px;line-height:1.45;}"
        ".muted{color:#98a2b8;}"
    )
    return (
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
        "<title>mvdirector — Track 2 Vision QC Error Inspector</title>"
        f"<style>{css}</style></head><body>"
        "<h1>Track 2: Vision QC False Positives, False Negatives &amp; Refusals</h1>"
        + "".join(sections)
        + "</body></html>\n"
    )


def write_vision_qc_reports(results: Sequence[VisionQCArmResult], qc_dir: Path) -> None:
    """Write ``qc_results.json``, ``qc_summary.csv``, ``qc_summary.md``, and ``qc_errors.html``."""
    qc_dir.mkdir(parents=True, exist_ok=True)
    atomic_write(
        qc_dir / "qc_results.json",
        json.dumps([r.model_dump(mode="json") for r in results], indent=2) + "\n",
    )
    atomic_write(qc_dir / "qc_summary.csv", render_vision_qc_csv(results))
    atomic_write(qc_dir / "qc_summary.md", render_vision_qc_markdown(results))
    atomic_write(qc_dir / "qc_errors.html", render_vision_qc_errors_html(results))
