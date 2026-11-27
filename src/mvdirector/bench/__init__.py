# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Multi-model A/B/C benchmark harness for mvdirector planning and Vision QC."""

from __future__ import annotations

from mvdirector.bench.blind_viewer import (
    build_blind_viewer_payload,
    generate_blind_viewer,
    select_matched_shots,
)
from mvdirector.bench.config import (
    DEFAULT_PRICING,
    BenchmarkConfig,
    GPUCostConfig,
    ModelArm,
    ModelPricing,
    default_arms,
    load_benchmark_config,
    write_default_benchmark_config,
)
from mvdirector.bench.dataset import (
    QCLabelItem,
    load_qc_dataset,
    resolve_still_and_references,
    scaffold_qc_dataset,
)
from mvdirector.bench.planning import (
    PlanningArmResult,
    PlanningQualityMetrics,
    evaluate_planning_quality,
    load_planning_results,
    render_planning_markdown,
    run_planning_arm,
    run_planning_benchmark,
)
from mvdirector.bench.telemetry import (
    CallRecord,
    InstrumentedLLM,
    TelemetrySummary,
    estimate_cost_usd,
)
from mvdirector.bench.vision_qc import (
    CheckCalibrationMetrics,
    GateOutcomeMetrics,
    QCItemVerdict,
    VisionQCArmResult,
    evaluate_hard_gate,
    load_vision_qc_results,
    render_vision_qc_markdown,
    run_vision_qc_arm,
    run_vision_qc_benchmark,
)

__all__ = [
    "DEFAULT_PRICING",
    "BenchmarkConfig",
    "CallRecord",
    "CheckCalibrationMetrics",
    "GPUCostConfig",
    "GateOutcomeMetrics",
    "InstrumentedLLM",
    "ModelArm",
    "ModelPricing",
    "PlanningArmResult",
    "PlanningQualityMetrics",
    "QCLabelItem",
    "QCItemVerdict",
    "TelemetrySummary",
    "VisionQCArmResult",
    "build_blind_viewer_payload",
    "default_arms",
    "estimate_cost_usd",
    "evaluate_hard_gate",
    "evaluate_planning_quality",
    "generate_blind_viewer",
    "load_benchmark_config",
    "load_planning_results",
    "load_qc_dataset",
    "load_vision_qc_results",
    "render_planning_markdown",
    "render_vision_qc_markdown",
    "resolve_still_and_references",
    "run_planning_arm",
    "run_planning_benchmark",
    "run_vision_qc_arm",
    "run_vision_qc_benchmark",
    "scaffold_qc_dataset",
    "select_matched_shots",
    "write_default_benchmark_config",
]
