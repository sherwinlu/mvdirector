# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Track 1 benchmark: multi-model planning (Visual Bible + batched shot direction)."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import re
import shutil
import statistics
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from mvdirector.bench.config import (
    BenchmarkConfig,
    ModelArm,
    open_bench_project,
    resolve_bench_environ,
    validate_budget_usd,
)
from mvdirector.bench.telemetry import CallRecord, InstrumentedLLM, TelemetrySummary
from mvdirector.config import CONFIG_FILENAME, ProjectConfig
from mvdirector.errors import ConfigError, MVDError
from mvdirector.llm.factory import build_llm
from mvdirector.llm.http import redact_url
from mvdirector.models import Character, Framing, Manifest, Shot, ShotType, VideoStyle
from mvdirector.planning import prompts
from mvdirector.planning.planner import build_manifest
from mvdirector.project import Project
from mvdirector.store import atomic_write

log = logging.getLogger(__name__)

_SCALE = {Framing.WIDE: 0, Framing.MEDIUM: 1, Framing.MEDIUM_CLOSE: 2, Framing.CLOSE_UP: 3}
_FALLBACK_LOG_MARKERS = (
    "using rule-based cutaway",
    "using rule-based performance direction",
    "keeping performance shot with rule-based direction",
    "they get rule-based direction instead",
    "using rule-based direction for remaining batches",
)
_REPAIR_LOG_MARKERS = (
    "unknown wardrobe state",
    "would undo an earlier change",
    "unknown character",
    "dropping unmentioned secondary character",
    "omitted co-singer; replacing with two-shot",
    "is not a set in the bible",
    "is not in the visual bible; ignoring",
    "converting to broll cutaway",
    "keeping performance type because",
)
_SHOT_ID_PREFIX_RE = re.compile(r"^(s\d+):")
_WIDE_MICRO_DETAIL_RE = re.compile(
    r"\b(?:freckles?|pores?|eyelashes?|pupils?|irises?|beauty\s+mark|mole\s+on|"
    r"tiny\s+(?:earring|stud|pendant|ring|mark|scar)|skin\s+texture|fine\s+wrinkles?)\b",
    re.IGNORECASE,
)
_TIGHT_WIDER_LABEL_RE = re.compile(
    r"\b(?:wide\s+shot|full[\s-]body\s+shot|medium\s+shot|from\s+(?:the|his|her|their)\s+"
    r"(?:waist|hips?|knees?)\s+up)\b",
    re.IGNORECASE,
)
_WARDROBE_NO_JEWELRY_RE = re.compile(
    r"\b(?:no|without(?:\s+any)?)\s+"
    r"(?:jewelry|jewellery|accessories|earrings?|necklace|pendant|ring)\b",
    re.IGNORECASE,
)
_INVARIANT_JEWELRY_RE = re.compile(
    r"\b(?:earrings?|studs?|hoops?|necklace|pendant|locket|choker|"
    r"ring|bracelet|brooch|pin|jewelry|jewellery)\b",
    re.IGNORECASE,
)


class PlanningQualityMetrics(BaseModel):
    """Automated Track 1 quality and guardrail-compliance metrics for one planned manifest."""

    model_config = ConfigDict(allow_inf_nan=False)

    total_shots: int = 0
    singer_shots: int = 0
    broll_shots: int = 0
    fallback_shots: int = 0
    fallback_rate: float = 0.0
    code_fixed_shots: int = 0
    distinct_sets_used: int = 0
    bible_sets_defined: int = 0
    wardrobe_states_defined: int = 0
    pace_correlation: float | None = None
    invariants_defined: bool = False
    invariant_always_coverage: float = 0.0
    invariant_wardrobe_contradictions: int = 0
    framing_detail_compliance: float = 1.0
    framing_detail_violations: int = 0
    affirmative_phrasing_compliance: float = 1.0
    negative_exclusion_shots: int = 0
    negative_exclusion_hits: int = 0
    motion_still_word_leaks: int = 0
    planner_warnings: list[str] = Field(default_factory=list)


class PlanningArmResult(BaseModel):
    """Saved benchmark outcome for one (track, model arm) planning run."""

    model_config = ConfigDict(allow_inf_nan=False)

    track: str
    source_project: str
    arm_id: str
    arm_label: str
    provider: str
    model: str
    attach_audio: bool
    config_fingerprint: str = ""
    output_dir: str
    manifest_path: str | None = None
    status: str = "ok"  # "ok" | "error"
    error: str | None = None
    wall_seconds: float = 0.0
    telemetry: TelemetrySummary = Field(default_factory=TelemetrySummary)
    quality: PlanningQualityMetrics = Field(default_factory=PlanningQualityMetrics)
    calls: list[CallRecord] = Field(default_factory=list)


class _PlannerWarningCollector(logging.Handler):
    """Captures warning/info messages emitted by ``mvdirector.planning.planner`` during a run."""

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def _is_rule_based_shot(shot: Shot) -> bool:
    """Detect deterministic prompt signatures produced by ``planner._apply_rule_based``."""
    sp = shot.still_prompt.strip()
    mp = shot.motion_prompt.strip()
    if (
        "of the singer singing directly to camera," in sp
        and mp == "The singer performs the line to camera with natural head movement. Slow push-in."
    ):
        return True
    if "of an object or corner of " in sp and "that echoes the song, no people," in sp:
        return True
    if (
        "inside the uninhabited " in sp
        and (", empty space, " in sp or ", empty architectural space, " in sp)
        and "uninhabited space" in mp
    ):
        return True
    return "of the singer moving to the music, not singing," in sp and mp.startswith(
        "The singer moves with the music, mouth closed, not singing."
    )


def _character_list(manifest: Manifest) -> list[Character]:
    if manifest.characters:
        return list(manifest.characters.values())
    if manifest.character.appearance.strip() or manifest.character.invariants.strip():
        return [manifest.character]
    return []


def _count_invariant_contradictions(characters: list[Character]) -> int:
    contradictions = 0
    for char in characters:
        pinned = char.pinned()
        if not pinned or not _INVARIANT_JEWELRY_RE.search(pinned):
            continue
        for state in char.wardrobe:
            if _WARDROBE_NO_JEWELRY_RE.search(state.description):
                contradictions += 1
    return contradictions


def evaluate_planning_quality(
    manifest: Manifest,
    *,
    planner_messages: Sequence[str] = (),
) -> PlanningQualityMetrics:
    """Compute automated Track 1 compliance & fallback metrics from a planned ``Manifest``."""
    shots = manifest.shots
    total = len(shots)
    singer_shots = [s for s in shots if s.type.shows_singer]
    broll_shots = [s for s in shots if s.type is ShotType.BROLL]

    fallback_ids: set[str] = set()
    fixed_ids: set[str] = set()
    warnings_kept: list[str] = []
    for msg in planner_messages:
        lowered = msg.lower()
        match = _SHOT_ID_PREFIX_RE.match(msg.strip())
        if any(marker in lowered for marker in _FALLBACK_LOG_MARKERS):
            warnings_kept.append(msg)
            if match:
                fallback_ids.add(match.group(1))
        elif any(marker in lowered for marker in _REPAIR_LOG_MARKERS):
            warnings_kept.append(msg)
            if match:
                fixed_ids.add(match.group(1))

    for shot in shots:
        if _is_rule_based_shot(shot):
            fallback_ids.add(shot.id)

    distinct_sets = {s.setting.strip().casefold() for s in shots if s.setting.strip()}

    pace_corr: float | None = None
    if (
        len(shots) >= 4
        and len({s.framing for s in shots}) >= 2
        and len({s.intensity for s in shots}) >= 2
    ):
        pace_corr = round(
            statistics.correlation(
                [s.intensity for s in shots], [float(_SCALE[s.framing]) for s in shots]
            ),
            4,
        )

    chars = _character_list(manifest)
    chars_with_invariants = [c for c in chars if c.pinned()]
    invariants_defined = bool(chars_with_invariants)
    invariant_contradictions = _count_invariant_contradictions(chars)

    if singer_shots:
        always_hits = 0
        eligible_always = 0
        for shot in singer_shots:
            char = manifest.shot_character(shot)
            if char.pinned():
                eligible_always += 1
                composed = prompts.compose_still_prompt(manifest, shot)
                if "Always:" in composed:
                    always_hits += 1
        always_cov = round(always_hits / eligible_always, 4) if eligible_always > 0 else 0.0
    else:
        always_cov = 1.0 if not invariants_defined else 0.0

    framing_violations = 0
    neg_shots = 0
    neg_hits = 0
    motion_still_leaks = 0
    for shot in shots:
        raw_still = shot.still_prompt
        violated = False
        if shot.type.shows_singer and shot.framing in (Framing.CLOSE_UP, Framing.MEDIUM_CLOSE):
            lower_re = (
                prompts._LOWER_BODY_STRICT_RE
                if shot.framing is Framing.CLOSE_UP
                else prompts._LOWER_BODY_MEDIUM_RE
            )
            if lower_re.search(raw_still) or _TIGHT_WIDER_LABEL_RE.search(raw_still):
                violated = True
            if shot.framing is Framing.CLOSE_UP and re.search(
                r"\b(?:from\s+the\s+)?(?:upper\s+)?chest\s+up\b", raw_still, re.IGNORECASE
            ):
                violated = True
        elif shot.framing is Framing.WIDE and _WIDE_MICRO_DETAIL_RE.search(raw_still):
            violated = True
        if violated:
            framing_violations += 1

        exclusions = prompts.extract_negative_exclusions(raw_still)
        if exclusions:
            neg_shots += 1
            neg_hits += len(exclusions)

        if prompts._STILL_WORDS_RE.search(shot.motion_prompt):
            motion_still_leaks += 1

    wardrobe_count = sum(len(c.wardrobe) for c in chars)
    return PlanningQualityMetrics(
        total_shots=total,
        singer_shots=len(singer_shots),
        broll_shots=len(broll_shots),
        fallback_shots=len(fallback_ids),
        fallback_rate=round(len(fallback_ids) / total, 4) if total else 0.0,
        code_fixed_shots=len(fixed_ids),
        distinct_sets_used=len(distinct_sets),
        bible_sets_defined=len(manifest.visual_bible.settings),
        wardrobe_states_defined=wardrobe_count,
        pace_correlation=pace_corr,
        invariants_defined=invariants_defined,
        invariant_always_coverage=always_cov,
        invariant_wardrobe_contradictions=invariant_contradictions,
        framing_detail_compliance=round(1.0 - (framing_violations / total), 4) if total else 1.0,
        framing_detail_violations=framing_violations,
        affirmative_phrasing_compliance=round(1.0 - (neg_shots / total), 4) if total else 1.0,
        negative_exclusion_shots=neg_shots,
        negative_exclusion_hits=neg_hits,
        motion_still_word_leaks=motion_still_leaks,
        planner_warnings=warnings_kept,
    )


def _planning_cache_fingerprint(arm: ModelArm) -> str:
    payload = {
        "provider": arm.provider,
        "model": arm.model,
        "attach_audio": arm.attach_audio,
        "max_tokens": arm.max_tokens,
        "shots_per_request": arm.shots_per_request,
        "json_mode": arm.json_mode,
        "base_url": arm.base_url,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def _load_valid_cached_planning(
    metrics_path: Path,
    manifest_path: Path,
    arm: ModelArm,
) -> PlanningArmResult | None:
    if not (metrics_path.is_file() and manifest_path.is_file()):
        return None
    try:
        cached = PlanningArmResult.model_validate_json(metrics_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if cached.status != "ok":
        return None
    if (
        cached.provider != arm.provider
        or cached.model != arm.model
        or cached.attach_audio != arm.attach_audio
    ):
        return None
    expected_fp = _planning_cache_fingerprint(arm)
    if cached.config_fingerprint and cached.config_fingerprint != expected_fp:
        return None
    return cached


def _redact_literal_base_url(section: object) -> object:
    if not isinstance(section, dict):
        return section
    copied = dict(section)
    raw_url = copied.get("base_url")
    if isinstance(raw_url, str) and "${" not in raw_url:
        copied["base_url"] = redact_url(raw_url)
    return copied


def _sanitized_arm_yaml(source_project: Project, arm: ModelArm, title: str) -> dict[str, Any]:
    """Build an isolated project.yaml mapping without expanding or persisting env secrets."""
    cfg_path = source_project.root / CONFIG_FILENAME
    raw_yaml: dict[str, Any] = {}
    if cfg_path.is_file():
        try:
            loaded = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                raw_yaml = loaded
        except (OSError, UnicodeDecodeError, yaml.YAMLError):
            raw_yaml = {}
    raw_yaml["title"] = title
    raw_yaml["llm"] = {
        "enabled": True,
        "provider": arm.provider,
        "model": arm.model,
        "base_url": (
            redact_url(arm.base_url)
            if isinstance(arm.base_url, str) and "${" not in arm.base_url
            else arm.base_url
        ),
        "api_key_env": arm.api_key_env,
        "max_tokens": arm.max_tokens,
        "shots_per_request": arm.shots_per_request,
        "timeout_seconds": arm.timeout_seconds,
        "json_mode": arm.json_mode,
        "stream": arm.stream,
    }
    # Benchmarks never connect to ComfyUI; strip any connection secrets if present literally.
    comfy_raw = _redact_literal_base_url(raw_yaml.get("comfy"))
    if isinstance(comfy_raw, dict):
        comfy_raw.pop("headers", None)
        comfy_raw.pop("api_key_comfy_org", None)
        raw_yaml["comfy"] = comfy_raw
    for parent_key, child_key in (("qc", "llm"), ("planning", "audio_llm")):
        parent = raw_yaml.get(parent_key)
        if isinstance(parent, dict) and child_key in parent:
            parent_copy = dict(parent)
            parent_copy[child_key] = _redact_literal_base_url(parent_copy[child_key])
            raw_yaml[parent_key] = parent_copy
    planning_raw = raw_yaml.get("planning")
    planning_copy = dict(planning_raw) if isinstance(planning_raw, dict) else {}
    planning_copy["audio_listen"] = "never"
    raw_yaml["planning"] = planning_copy
    return raw_yaml


def _stage_isolated_arm_folder(
    source_project: Project,
    arm: ModelArm,
    arm_dir: Path,
    *,
    environ: Mapping[str, str] | None = None,
) -> tuple[ProjectConfig, str, str, Path | None]:
    """Copy frozen planning inputs into ``arm_dir`` and return patched config + text inputs."""
    arm_dir.mkdir(parents=True, exist_ok=True)
    inputs_dir = arm_dir / "inputs"
    inputs_dir.mkdir(exist_ok=True)

    src_analysis = source_project.root / "analysis.json"
    if not src_analysis.is_file():
        raise ConfigError(
            f"Source project {source_project.root} has no frozen analysis.json. "
            "Run `mvd analyze` first."
        )
    shutil.copy2(src_analysis, arm_dir / "analysis.json")

    cfg = source_project.config.model_copy(deep=True)
    cfg.llm = arm.to_llm_config(environ=environ)
    cfg.planning.audio_listen = "never"
    cfg.title = f"{source_project.config.title} [{arm.id}: {arm.label}]"

    style_path = source_project.input_path(cfg.inputs.style, "style")
    style_text = style_path.read_text(encoding="utf-8-sig")
    (inputs_dir / "style.md").write_text(style_text, encoding="utf-8")

    lyrics_text = ""
    if cfg.planning.video_style is VideoStyle.LIPSYNC:
        lyrics_path = source_project.input_path(cfg.inputs.lyrics, "lyrics")
        lyrics_text = lyrics_path.read_text(encoding="utf-8-sig")
        (inputs_dir / "lyrics.txt").write_text(lyrics_text, encoding="utf-8")
    elif cfg.inputs.lyrics:
        cand = source_project.path(cfg.inputs.lyrics)
        if cand.is_file():
            lyrics_text = cand.read_text(encoding="utf-8-sig")
            (inputs_dir / "lyrics.txt").write_text(lyrics_text, encoding="utf-8")

    # Write a secret-free project.yaml into the isolated benchmark folder for reproducibility.
    raw_yaml = _sanitized_arm_yaml(source_project, arm, cfg.title)
    atomic_write(arm_dir / CONFIG_FILENAME, yaml.safe_dump(raw_yaml, sort_keys=False))

    mix_path: Path | None = None
    if arm.attach_audio and cfg.inputs.mix:
        cand_mix = source_project.path(cfg.inputs.mix)
        if cand_mix.is_file():
            mix_path = cand_mix
    return cfg, lyrics_text, style_text, mix_path


_SAFE_TRACK_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}\Z")


def _safe_track_slug(raw: str) -> str:
    cleaned = raw.strip()
    if not cleaned or ".." in cleaned or "/" in cleaned or "\\" in cleaned:
        raise ConfigError(f"Invalid benchmark track name: {raw!r}")
    if not _SAFE_TRACK_RE.match(cleaned):
        slug = re.sub(r"[^A-Za-z0-9._-]+", "-", cleaned).strip("._-")
        if not slug:
            raise ConfigError(f"Invalid benchmark track name: {raw!r}")
        return slug[:96]
    return cleaned


def run_planning_arm(
    source_dir: Path,
    arm: ModelArm,
    out_root: Path,
    *,
    track_name: str | None = None,
    overwrite: bool = False,
    environ: Mapping[str, str] | None = None,
    llm_factory: Callable[[ModelArm, Mapping[str, str] | None], Any] | None = None,
) -> PlanningArmResult:
    """Run ``mvd plan`` for one (track, arm) pair in an isolated directory."""
    source_project = open_bench_project(source_dir)
    track = _safe_track_slug(track_name or source_project.root.name)
    arm_dir = out_root / "planning" / track / arm.id
    metrics_path = arm_dir / "metrics.json"
    manifest_path = arm_dir / "manifest.json"

    if not overwrite:
        cached = _load_valid_cached_planning(metrics_path, manifest_path, arm)
        if cached is not None:
            log.info("Skipping cached Track 1 run %s / %s (%s)", track, arm.id, arm_dir)
            return cached

    if overwrite and manifest_path.is_file():
        manifest_path.unlink(missing_ok=True)

    effective_env = resolve_bench_environ(source_project.environ(), environ)
    cfg, lyrics_text, style_text, mix_path = _stage_isolated_arm_folder(
        source_project, arm, arm_dir, environ=effective_env
    )
    analysis = source_project.load_analysis()
    cache_fp = _planning_cache_fingerprint(arm)

    started = time.monotonic()
    collector = _PlannerWarningCollector()
    planner_logger = logging.getLogger("mvdirector.planning.planner")
    prev_logger_level = planner_logger.level
    if planner_logger.getEffectiveLevel() > logging.INFO:
        planner_logger.setLevel(logging.INFO)
    planner_logger.addHandler(collector)

    raw_llm = None
    instrumented: InstrumentedLLM | None = None
    try:
        raw_llm = (
            llm_factory(arm, effective_env)
            if llm_factory is not None
            else build_llm(cfg.llm, environ=effective_env)
        )
        if raw_llm is None:
            raise ConfigError(f"Arm {arm.id} produced no LLM client.")
        instrumented = InstrumentedLLM(raw_llm, arm)
        with instrumented.stage("planning"):
            manifest = build_manifest(
                cfg,
                analysis,
                lyrics_text,
                style_text,
                instrumented,
                mix_path=mix_path if arm.attach_audio else None,
            )
        atomic_write(manifest_path, manifest.model_dump_json(indent=2) + "\n")
        quality = evaluate_planning_quality(manifest, planner_messages=collector.messages)
        result = PlanningArmResult(
            track=track,
            source_project=str(source_project.root),
            arm_id=arm.id,
            arm_label=arm.label,
            provider=arm.provider,
            model=arm.model,
            attach_audio=arm.attach_audio,
            config_fingerprint=cache_fp,
            output_dir=str(arm_dir),
            manifest_path=str(manifest_path),
            status="ok",
            wall_seconds=round(max(0.0, time.monotonic() - started), 3),
            telemetry=instrumented.summary(),
            quality=quality,
            calls=instrumented.records,
        )
    except MVDError as exc:
        log.warning("Track 1 arm %s (%s) failed on %s: %s", arm.id, arm.model, track, exc)
        result = PlanningArmResult(
            track=track,
            source_project=str(source_project.root),
            arm_id=arm.id,
            arm_label=arm.label,
            provider=arm.provider,
            model=arm.model,
            attach_audio=arm.attach_audio,
            config_fingerprint=cache_fp,
            output_dir=str(arm_dir),
            manifest_path=None,
            status="error",
            error=str(exc),
            wall_seconds=round(max(0.0, time.monotonic() - started), 3),
            telemetry=instrumented.summary() if instrumented is not None else TelemetrySummary(),
            calls=instrumented.records if instrumented is not None else [],
        )
    finally:
        planner_logger.removeHandler(collector)
        planner_logger.setLevel(prev_logger_level)
        if instrumented is not None:
            instrumented.close()
        elif raw_llm is not None:
            closer = getattr(raw_llm, "close", None)
            if callable(closer):
                closer()

    atomic_write(metrics_path, result.model_dump_json(indent=2) + "\n")
    return result


def run_planning_benchmark(
    projects: Sequence[Path],
    config: BenchmarkConfig,
    out_root: Path,
    *,
    arm_ids: Sequence[str] | None = None,
    include_controls: bool = False,
    overwrite: bool = False,
    max_budget_usd: float | None = None,
    environ: Mapping[str, str] | None = None,
    llm_factory: Callable[[ModelArm, Mapping[str, str] | None], Any] | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> list[PlanningArmResult]:
    """Run Track 1 planning across selected ``projects`` and ``arms`` and write merged summaries."""
    if not projects:
        raise ConfigError("Provide at least one project directory for Track 1 planning benchmark.")
    budget_cap = validate_budget_usd(max_budget_usd)
    arms = config.select_arms(arm_ids, include_controls=include_controls)
    results: list[PlanningArmResult] = []
    spent_usd = 0.0
    budget_hit = False
    used_tracks: set[str] = set()
    for project_dir in projects:
        base_track = _safe_track_slug(project_dir.resolve().name)
        track = base_track
        suffix = 2
        while track.casefold() in used_tracks:
            track = f"{base_track}-{suffix}"
            suffix += 1
        used_tracks.add(track.casefold())
        for arm in arms:
            arm_dir = out_root / "planning" / track / arm.id
            already_cached = (
                not overwrite
                and _load_valid_cached_planning(
                    arm_dir / "metrics.json", arm_dir / "manifest.json", arm
                )
                is not None
            )
            if budget_cap is not None and not already_cached and spent_usd >= budget_cap:
                if on_progress and not budget_hit:
                    on_progress(
                        f"Budget cap reached (${spent_usd:.4f} >= ${budget_cap:.2f}); "
                        "skipping remaining uncached arms."
                    )
                budget_hit = True
                continue
            if on_progress:
                audio_tag = "audio" if arm.attach_audio else "text"
                on_progress(f"Planning [{track}] Arm {arm.id} ({arm.model}, {audio_tag})...")
            res = run_planning_arm(
                project_dir,
                arm,
                out_root,
                track_name=track,
                overwrite=overwrite,
                environ=environ,
                llm_factory=llm_factory,
            )
            if not already_cached:
                spent_usd += res.telemetry.estimated_usd
            results.append(res)
    merged = load_planning_results(out_root / "planning")
    write_planning_reports(merged or results, out_root / "planning")
    return results


def md_cell(value: str) -> str:
    """Escape pipes, backticks, and line breaks for Markdown table cells."""
    return value.replace("|", "\\|").replace("`", "'").replace("\r", " ").replace("\n", " ")


def render_planning_markdown(results: Sequence[PlanningArmResult]) -> str:
    """Render a concise GitHub-flavored Markdown comparison table for Track 1 results."""
    lines = [
        "# Track 1: Planning Benchmark Summary",
        "",
        "| Track | Arm | Model | Audio | Status | 1st-Pass JSON | Retries | Trunc | Refusals | "
        "Fallback Shots | Invariants (`Always:`) | Framing OK | Affirm. OK | Sets | "
        "Latency (s) | Est. USD |",
        "| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | "
        ":---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in results:
        t = r.telemetry
        q = r.quality
        audio = "yes" if r.attach_audio else "no"
        warn = (
            f" ({q.invariant_wardrobe_contradictions}⚠)"
            if q.invariant_wardrobe_contradictions
            else ""
        )
        inv = f"{q.invariant_always_coverage:.0%}{warn}" if q.invariants_defined else "none"
        lines.append(
            f"| {md_cell(r.track)} | **{md_cell(r.arm_id)}** | `{md_cell(r.model)}` | {audio} | "
            f"{md_cell(r.status)} | "
            f"{t.first_pass_rate:.0%} ({t.first_pass_ok}/{t.calls}) | {t.schema_retries} | "
            f"{t.truncations} | {t.refusals} | {q.fallback_shots}/{q.total_shots} | {inv} | "
            f"{q.framing_detail_compliance:.0%} | {q.affirmative_phrasing_compliance:.0%} | "
            f"{q.distinct_sets_used}/{q.bible_sets_defined} | {r.wall_seconds:.1f} | "
            f"${t.estimated_usd:.4f} |"
        )
    return "\n".join(lines) + "\n"


def render_planning_csv(results: Sequence[PlanningArmResult]) -> str:
    buf = io.StringIO()
    fields = [
        "track",
        "arm_id",
        "arm_label",
        "provider",
        "model",
        "attach_audio",
        "status",
        "wall_seconds",
        "calls",
        "first_pass_rate",
        "schema_retries",
        "truncations",
        "refusals",
        "input_tokens",
        "output_tokens",
        "thoughts_tokens",
        "estimated_usd",
        "total_shots",
        "fallback_shots",
        "code_fixed_shots",
        "distinct_sets_used",
        "bible_sets_defined",
        "pace_correlation",
        "invariants_defined",
        "invariant_always_coverage",
        "invariant_wardrobe_contradictions",
        "framing_detail_compliance",
        "affirmative_phrasing_compliance",
        "motion_still_word_leaks",
        "error",
    ]
    writer = csv.DictWriter(buf, fieldnames=fields)
    writer.writeheader()
    for r in results:
        writer.writerow(
            {
                "track": r.track,
                "arm_id": r.arm_id,
                "arm_label": r.arm_label,
                "provider": r.provider,
                "model": r.model,
                "attach_audio": r.attach_audio,
                "status": r.status,
                "wall_seconds": r.wall_seconds,
                "calls": r.telemetry.calls,
                "first_pass_rate": r.telemetry.first_pass_rate,
                "schema_retries": r.telemetry.schema_retries,
                "truncations": r.telemetry.truncations,
                "refusals": r.telemetry.refusals,
                "input_tokens": r.telemetry.input_tokens,
                "output_tokens": r.telemetry.output_tokens,
                "thoughts_tokens": r.telemetry.thoughts_tokens,
                "estimated_usd": r.telemetry.estimated_usd,
                "total_shots": r.quality.total_shots,
                "fallback_shots": r.quality.fallback_shots,
                "code_fixed_shots": r.quality.code_fixed_shots,
                "distinct_sets_used": r.quality.distinct_sets_used,
                "bible_sets_defined": r.quality.bible_sets_defined,
                "pace_correlation": r.quality.pace_correlation,
                "invariants_defined": r.quality.invariants_defined,
                "invariant_always_coverage": r.quality.invariant_always_coverage,
                "invariant_wardrobe_contradictions": r.quality.invariant_wardrobe_contradictions,
                "framing_detail_compliance": r.quality.framing_detail_compliance,
                "affirmative_phrasing_compliance": r.quality.affirmative_phrasing_compliance,
                "motion_still_word_leaks": r.quality.motion_still_word_leaks,
                "error": r.error or "",
            }
        )
    return buf.getvalue()


def write_planning_reports(results: Sequence[PlanningArmResult], planning_dir: Path) -> None:
    """Write ``summary.json``, ``summary.csv``, and ``summary.md`` under ``planning_dir``."""
    planning_dir.mkdir(parents=True, exist_ok=True)
    atomic_write(
        planning_dir / "summary.json",
        json.dumps([r.model_dump(mode="json") for r in results], indent=2) + "\n",
    )
    atomic_write(planning_dir / "summary.csv", render_planning_csv(results))
    atomic_write(planning_dir / "summary.md", render_planning_markdown(results))


def load_planning_results(planning_dir: Path) -> list[PlanningArmResult]:
    """Load saved ``PlanningArmResult`` files from ``<out>/planning/<track>/<arm>/metrics.json``."""
    results: list[PlanningArmResult] = []
    if not planning_dir.is_dir():
        return results
    for metrics_file in sorted(planning_dir.glob("*/*/metrics.json")):
        try:
            results.append(
                PlanningArmResult.model_validate_json(metrics_file.read_text(encoding="utf-8"))
            )
        except (OSError, ValueError) as exc:
            log.warning("Skipping unreadable benchmark metrics %s: %s", metrics_file, exc)
    return results
