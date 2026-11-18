# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Service layer: every pipeline stage as a plain function.

The CLI is a thin front end over these functions, and other programs (notebooks, workers, web
services) can call them directly. Functions take a project folder, raise `MVDError` on failure,
and return data instead of printing. Progress is reported through callbacks, never `print`.
"""

from __future__ import annotations

import json
import logging
import shutil
import sys
from collections import Counter
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from functools import partial
from importlib import resources
from pathlib import Path

from mvdirector import assemble as assemble_mod
from mvdirector import calibrate as calibrate_mod
from mvdirector import media
from mvdirector import report as report_mod
from mvdirector.audio.analysis import analyze_project
from mvdirector.comfy.client import ComfyClient
from mvdirector.comfy.provider import ComfyEndpointProvider, StaticComfyProvider
from mvdirector.config import CONFIG_FILENAME, ReviewMode
from mvdirector.errors import BudgetExceeded, LLMError, MVDError
from mvdirector.generation import plates as plates_mod
from mvdirector.generation import previz as previz_mod
from mvdirector.generation import props as props_mod
from mvdirector.generation import reference as reference_mod
from mvdirector.generation import upscale as upscale_mod
from mvdirector.generation.preflight import check_assets
from mvdirector.generation.runner import RunBudget, RunEvent, ShotRunner, load_templates
from mvdirector.llm import build_llm
from mvdirector.llm.check import LLMCheckResult, ping_llm, run_llm_check
from mvdirector.models import Analysis, AssetKind, Manifest, ReferenceSet
from mvdirector.planning.brief import ensure_writable, generate_brief, write_brief
from mvdirector.planning.planner import build_manifest
from mvdirector.project import Project
from mvdirector.qc.factory import build_quality_gates
from mvdirector.retry import RetryPolicy
from mvdirector.review import policy as review_policy

__all__ = [
    # types
    "BriefResult",
    "GenerateResult",
    "LLMCheckOutcome",
    "Observer",
    "ProjectRef",
    "ValidationCheck",
    # stages
    "analyze",
    "approve",
    "assemble",
    "brief",
    "calibrate",
    "check_llm",
    "export_schemas",
    "generate",
    "generate_plates",
    "generate_previz",
    "generate_props",
    "generate_reference_candidates",
    "generate_reference_videos",
    "init_project",
    "load_status",
    "lock_plate",
    "lock_previz",
    "lock_prop",
    "lock_reference",
    "lock_reference_video",
    "open_project",
    "plan",
    "qc_report",
    "refine_previz",
    "reject",
    "status_counts",
    "upscale",
    "validate_project",
]

log = logging.getLogger(__name__)

Observer = Callable[[RunEvent], None]
ProjectRef = Path | Project
"""A project folder, or a `Project` already wired to custom state and storage."""


def _lease(
    project: Project, provider: ComfyEndpointProvider | None
) -> AbstractContextManager[ComfyClient]:
    return (provider or StaticComfyProvider()).lease(project)


def open_project(
    project_dir: ProjectRef, review_mode: ReviewMode | None = None, *, pull: bool = True
) -> Project:
    """Open a project folder, optionally overriding `review.mode` for the returned project.

    Pass an already-built `Project` (for example one wired to a database `StateStore` and an
    object-store `AssetStorage`) to run any stage against custom persistence. A review-mode
    override never changes the `Project` you passed in; the returned project carries a copy of
    its config. `pull=False` skips `AssetStorage.pull()` for stages that only touch state.
    """
    project = project_dir if isinstance(project_dir, Project) else Project.open(project_dir)
    if pull:
        project.storage.pull()
    if review_mode is not None and project.config.review.mode != review_mode:
        config = project.config.model_copy(deep=True)
        config.review.mode = review_mode
        project = Project(project.root, config, state=project.state, storage=project.storage)
    return project


# --- setup ------------------------------------------------------------------------------------


def init_project(directory: Path) -> Path:
    """Create a project folder with a starter config and workflow binding examples."""
    if (directory / CONFIG_FILENAME).exists():
        raise MVDError(f"{directory} already contains {CONFIG_FILENAME}")
    template_root = resources.files("mvdirector") / "templates"
    with resources.as_file(template_root) as source:
        shutil.copytree(source, directory, dirs_exist_ok=True)
    for folder in ("inputs", "assets", "output"):
        (directory / folder).mkdir(exist_ok=True)
    return directory


@dataclass
class ValidationCheck:
    label: str
    problem: str | None = None

    @property
    def ok(self) -> bool:
        return self.problem is None


def validate_project(
    project_dir: ProjectRef,
    comfy_provider: ComfyEndpointProvider | None = None,
) -> list[ValidationCheck]:
    """Check tools, workflow bindings and the ComfyUI connection before spending GPU time."""
    project = open_project(project_dir)
    checks: list[ValidationCheck] = []

    def attempt(label: str, action: Callable[[], object]) -> None:
        try:
            action()
            checks.append(ValidationCheck(label))
        except MVDError as exc:
            checks.append(ValidationCheck(label, str(exc)))

    for tool in ("ffmpeg", "ffprobe"):
        attempt(tool, partial(media.require_tool, tool))
    for name in project.config.comfy.workflows:
        attempt(f"workflow '{name}'", partial(load_templates, project, [name]))

    def ping() -> None:
        with _lease(project, comfy_provider) as comfy:
            comfy.ping()

    attempt(f"ComfyUI at {project.config.comfy.base_url}", ping)
    llm_config = project.config.llm
    if llm_config.enabled:
        attempt(
            f"LLM settings ({llm_config.provider}, {llm_config.model}); `mvd llm-check` tests them",
            lambda: build_llm(llm_config),
        )
    return checks


@dataclass
class LLMCheckOutcome:
    provider: str
    model: str
    enabled: bool
    vision_needed: bool = False
    tested_images: bool = False
    result: LLMCheckResult | None = None

    @property
    def vision_failed(self) -> bool:
        r = self.result
        return bool(r and self.tested_images and r.vision not in ("skipped", "ok"))

    @property
    def ok(self) -> bool:
        return not self.enabled or bool(
            self.result and self.result.json_ok and not self.vision_failed
        )


def check_llm(project_dir: ProjectRef, skip_image: bool = False) -> LLMCheckOutcome:
    """Test the configured model: credentials, JSON output and image input."""
    project = open_project(project_dir)
    config = project.config.llm
    outcome = LLMCheckOutcome(config.provider, config.model, config.enabled)
    if not config.enabled:
        return outcome
    client = build_llm(config)
    if client is None:  # build_llm returns None only for a disabled model, handled above
        raise MVDError("No language model is configured.")
    # The image test only matters when vision checks are switched on.
    outcome.vision_needed = project.config.qc.vision_enabled
    outcome.tested_images = outcome.vision_needed and not skip_image
    outcome.result = run_llm_check(client, check_vision=outcome.tested_images)
    return outcome


# --- planning ---------------------------------------------------------------------------------


def analyze(project_dir: ProjectRef) -> tuple[Project, Analysis]:
    """Analyze the audio: beats, bars, vocal activity and lyric timing."""
    project = open_project(project_dir)
    analysis = analyze_project(project)
    project.save_analysis(analysis)
    return project, analysis


@dataclass
class BriefResult:
    lyrics_path: Path
    style_path: Path
    suno_style: str


def brief(
    project_dir: ProjectRef,
    idea: str,
    duration: int | None = None,
    vibe: str | None = None,
    overwrite: bool = False,
) -> BriefResult:
    """Draft Suno-ready lyrics.txt and style.md from a concept."""
    project = open_project(project_dir)
    ensure_writable(project, overwrite=overwrite)
    llm = build_llm(project.config.llm)
    generated = generate_brief(idea, llm=llm, duration_seconds=duration, vibe=vibe)
    with project.run_lock():
        ensure_writable(project, overwrite=overwrite)  # re-check: another job may have written
        lyrics_path, style_path = write_brief(project, generated, overwrite=overwrite)
        project.storage.push()
    return BriefResult(lyrics_path, style_path, generated.suno_style)


def plan(project_dir: ProjectRef, overwrite: bool = False) -> tuple[Project, Manifest]:
    """Create the shot manifest from the analysis, lyrics and style brief."""
    project = open_project(project_dir)
    with project.run_lock():
        if project.state.has_manifest() and not overwrite:
            raise MVDError("manifest.json already exists. Pass --overwrite to replace it.")
        inputs = project.config.inputs
        manifest = build_manifest(
            project.config,
            project.load_analysis(),
            project.input_path(inputs.lyrics, "lyrics").read_text("utf-8"),
            project.input_path(inputs.style, "style").read_text("utf-8"),
            build_llm(project.config.llm),
        )
        project.save_manifest(manifest)
    return project, manifest


# --- references, plates, props, previz --------------------------------------------------------


def generate_reference_candidates(
    project_dir: ProjectRef,
    state: str | None = None,
    character: str | None = None,
    from_character: str | None = None,
    count: int | None = None,
    comfy_provider: ComfyEndpointProvider | None = None,
) -> list[Path]:
    """Generate candidate reference sheets (front, profile, 3/4 and face panels)."""
    project = open_project(project_dir)
    manifest = project.load_manifest()
    needs_still = reference_mod.uses_still_workflow(
        manifest, state, character, from_character=from_character
    )
    templates = load_templates(project, ["still"] if needs_still else ["t2i"])
    with _lease(project, comfy_provider) as comfy:
        result = reference_mod.generate_candidates(
            project,
            manifest,
            comfy,
            templates,
            state,
            count or project.config.reference.candidates,
            character_id=character,
            from_character=from_character,
        )
    project.storage.push()  # candidates are not in the manifest yet; publish them now
    return result


def lock_reference(
    project_dir: ProjectRef,
    image: Path,
    state: str | None = None,
    character: str | None = None,
    single: bool = False,
) -> ReferenceSet:
    """Lock a reference sheet, cut it into panels, and use it for this wardrobe state."""
    project = open_project(project_dir)
    with project.edit_manifest() as manifest:
        return reference_mod.lock_reference(
            project, manifest, image, state, split=not single, character_id=character
        )


def generate_reference_videos(
    project_dir: ProjectRef,
    state: str | None = None,
    character: str | None = None,
    count: int | None = None,
    comfy_provider: ComfyEndpointProvider | None = None,
) -> list[Path]:
    """Generate candidate silent turnaround reference videos from a locked sheet."""
    project = open_project(project_dir)
    manifest = project.load_manifest()
    workflow_name = "ref_video" if "ref_video" in project.config.comfy.workflows else "i2v"
    templates = load_templates(project, [workflow_name])
    with _lease(project, comfy_provider) as comfy:
        result = reference_mod.generate_video_candidates(
            project,
            manifest,
            comfy,
            templates,
            state,
            count or project.config.reference.video_candidates,
            character_id=character,
        )
    project.storage.push()  # candidates are not in the manifest yet; publish them now
    return result


def lock_reference_video(
    project_dir: ProjectRef,
    video: Path,
    state: str | None = None,
    character: str | None = None,
) -> str:
    """Lock a silent turnaround reference video for this character/wardrobe state."""
    project = open_project(project_dir)
    with project.edit_manifest() as manifest:
        return reference_mod.lock_reference_video(
            project, manifest, video, state, character_id=character
        )


def generate_plates(
    project_dir: ProjectRef,
    set_name: str | None = None,
    style_ref: Path | None = None,
    embed_set: str | None = None,
    from_set: str | None = None,
    count: int | None = None,
    comfy_provider: ComfyEndpointProvider | None = None,
) -> dict[str, list[Path]]:
    """Generate candidate images of the empty sets."""
    project = open_project(project_dir)
    manifest = project.load_manifest()
    needs_still = (
        style_ref is not None
        or embed_set is not None
        or from_set is not None
        or any(
            s.parent_set
            for s in manifest.visual_bible.settings
            if set_name is None or s.name.strip().lower() == set_name.strip().lower()
        )
    )
    templates = load_templates(project, ["t2i", "still"] if needs_still else ["t2i"])
    with _lease(project, comfy_provider) as comfy:
        result = plates_mod.generate_candidates(
            project,
            manifest,
            comfy,
            templates if needs_still else templates["t2i"],
            set_name,
            count or project.config.plates.candidates,
            style_ref=style_ref,
            embed_set=embed_set,
            from_set=from_set,
        )
    project.storage.push()  # candidates are not in the manifest yet; publish them now
    return result


def lock_plate(project_dir: ProjectRef, image: Path, set_name: str) -> Path:
    """Lock an image as a set's plate, used as a second reference for stills."""
    project = open_project(project_dir)
    with project.edit_manifest() as manifest:
        return plates_mod.lock_plate(project, manifest, image, set_name)


def generate_props(
    project_dir: ProjectRef,
    prop: str | None = None,
    count: int | None = None,
    comfy_provider: ComfyEndpointProvider | None = None,
) -> dict[str, list[Path]]:
    """Generate candidate two-angle reference sheets for recurring hero props."""
    project = open_project(project_dir)
    manifest = project.load_manifest()
    template = load_templates(project, ["t2i"])["t2i"]
    with _lease(project, comfy_provider) as comfy:
        result = props_mod.generate_candidates(
            project, manifest, comfy, template, prop, count or project.config.props.candidates
        )
    project.storage.push()  # candidates are not in the manifest yet; publish them now
    return result


def lock_prop(project_dir: ProjectRef, image: Path, prop: str) -> Path:
    """Lock a two-angle prop sheet, used as a reference on shots containing that prop."""
    project = open_project(project_dir)
    with project.edit_manifest() as manifest:
        return props_mod.lock_prop(project, manifest, image, prop)


def generate_previz(
    project_dir: ProjectRef, shot_ids: list[str] | None = None, overwrite: bool = False
) -> list[Path]:
    """Plan and render flat-shaded 3D blocking previz MP4s for selected shots."""
    project = open_project(project_dir)
    llm = build_llm(project.config.llm)
    with project.edit_manifest() as manifest:
        return previz_mod.generate_previz(
            project, manifest, llm, shot_ids=shot_ids, overwrite=overwrite
        )


def lock_previz(
    project_dir: ProjectRef, video: Path, shot_id: str, mapping: str | None = None
) -> Path:
    """Attach a custom 3D blocking video to a shot."""
    project = open_project(project_dir)
    with project.edit_manifest() as manifest:
        return previz_mod.lock_previz(project, manifest, video, shot_id, mapping=mapping)


def refine_previz(project_dir: ProjectRef, shot_id: str, feedback: str) -> Path:
    """Refine a shot's 3D blocking plan with LLM feedback and re-render its previz MP4."""
    project = open_project(project_dir)
    llm = build_llm(project.config.llm)
    with project.edit_manifest() as manifest:
        return previz_mod.refine_previz_shot(project, manifest, llm, shot_id, feedback)


# --- generation -------------------------------------------------------------------------------


@dataclass
class GenerateResult:
    project: Project
    manifest: Manifest
    active_checks: list[str] = field(default_factory=list)
    skipped_checks: list[tuple[str, str]] = field(default_factory=list)
    budget_message: str | None = None


def generate(
    project_dir: ProjectRef,
    kinds: list[AssetKind],
    shots: list[str] | None = None,
    review_mode: ReviewMode | None = None,
    skip_preflight: bool = False,
    draft: bool = False,
    observer: Observer | None = None,
    comfy_provider: ComfyEndpointProvider | None = None,
) -> GenerateResult:
    """Generate and quality-check stills and/or videos, retrying failures automatically.

    `observer` receives every `RunEvent`: preflight `warning`s and the `qc_summary` before any
    GPU time is spent, then the per-attempt events of the run. It is the one stream to persist.

    The writer lock is taken before the ComfyUI endpoint is leased, so a provider that waits
    for capacity does so while holding the lock. The manifest is reloaded under the lock, so
    changes made by another job during preflight are never overwritten.
    """

    def emit(event: RunEvent) -> None:
        if observer is not None:
            observer(event)

    project = open_project(project_dir, review_mode)
    manifest = project.load_manifest()
    llm = build_llm(project.config.llm)
    names = ["still", "t2i"] if AssetKind.STILL in kinds else []
    names += ["lipsync", "i2v"] if AssetKind.VIDEO in kinds else []
    templates = load_templates(project, names)
    if AssetKind.STILL in kinds:
        ready = check_assets(project, manifest, templates, shots)
        for warning in ready.warnings:
            emit(RunEvent("warning", message=f"Warning: {warning}"))
        if skip_preflight:
            for problem in ready.problems:
                emit(RunEvent("warning", message=f"Skipping check: {problem}"))
        else:
            ready.raise_if_blocked()
    if llm is not None:
        # Fail before any GPU time is spent, not as a stream of skipped checks afterwards.
        try:
            ping_llm(llm)
        except LLMError as exc:
            raise LLMError(
                f"The language model is not usable: {exc}\nFix it (`mvd llm-check` tests the "
                "settings), or set llm.enabled: false to run without a model."
            ) from exc
    result = GenerateResult(project, manifest)
    gates = build_quality_gates(project.config, llm)
    try:
        result.active_checks, result.skipped_checks = gates.check_summary()
        emit(
            RunEvent(
                "qc_summary",
                active_checks=result.active_checks,
                skipped_checks=result.skipped_checks,
            )
        )
        policy = RetryPolicy(project.config.retry, project.config.seed, llm)
        budget = RunBudget(project.config.retry.max_generations_per_run)
        with project.run_lock(), _lease(project, comfy_provider) as comfy:
            manifest = result.manifest = project.load_manifest()
            comfy.ping()
            runner = ShotRunner(
                project,
                manifest,
                comfy,
                gates,
                policy,
                templates,
                budget,
                observer=observer,
                draft=draft,
            )
            try:
                runner.run(kinds, shots)
            except BudgetExceeded as exc:
                result.budget_message = str(exc)
            finally:
                _save_after_run(project, manifest)
    finally:
        gates.close()
    return result


def _save_after_run(project: Project, manifest: Manifest) -> None:
    """Persist the manifest after a run without letting a failed save hide the run's error."""
    in_flight = sys.exc_info()[1]  # set when called from a `finally` during a failing run
    review_policy.refresh_reviews(manifest, project.config)
    try:
        project.save_manifest(manifest)
    except Exception:
        if in_flight is None:  # nothing else went wrong, so the save failure is the news
            raise
        log.exception("Could not save the manifest after a failed run")


def upscale(
    project_dir: ProjectRef,
    shots: list[str] | None = None,
    force: bool = False,
    overwrite: bool = False,
    comfy_provider: ComfyEndpointProvider | None = None,
) -> list[Path]:
    """Upscale winning/approved draft clips through the optional 'upscale' workflow."""
    project = open_project(project_dir)
    if "upscale" not in project.config.comfy.workflows:
        raise MVDError(
            "No 'upscale' workflow configured. Add comfy.workflows.upscale in project.yaml."
        )
    template = load_templates(project, ["upscale"])["upscale"]
    with (
        project.run_lock(),
        _lease(project, comfy_provider) as comfy,
    ):
        manifest = project.load_manifest()
        comfy.ping()
        return upscale_mod.upscale_videos(
            project, manifest, comfy, template, shot_ids=shots, force=force, overwrite=overwrite
        )


# --- status and review ------------------------------------------------------------------------


def load_status(
    project_dir: ProjectRef, review_mode: ReviewMode | None = None, refresh: bool = True
) -> tuple[Project, Manifest]:
    """Load a project and manifest, refreshing review queues unless `refresh` is false."""
    project = open_project(project_dir, review_mode, pull=False)
    manifest = project.load_manifest()
    if refresh:
        review_policy.refresh_reviews(manifest, project.config)
    return project, manifest


def approve(project_dir: ProjectRef, shot_ids: list[str], still: bool = False) -> None:
    """Approve shots, or accept an exhausted still so its video can be generated."""
    decide = review_policy.accept_still if still else review_policy.approve
    _decide(project_dir, lambda manifest, _: [decide(manifest, s) for s in shot_ids])


def reject(
    project_dir: ProjectRef, shot_id: str, note: str, target: AssetKind = AssetKind.VIDEO
) -> None:
    """Reject a shot and send it back for regeneration with the reviewer's note."""
    _decide(
        project_dir,
        lambda manifest, project: review_policy.reject(
            manifest, shot_id, note, target, project.config
        ),
    )


def _decide(project_dir: ProjectRef, action: Callable[[Manifest, Project], object]) -> None:
    project = open_project(project_dir, pull=False)
    try:
        with project.edit_manifest() as manifest:
            action(manifest, project)
    except (KeyError, ValueError) as exc:
        raise MVDError(str(exc)) from exc


def assemble(
    project_dir: ProjectRef,
    output: Path | None = None,
    force: bool = False,
    review_mode: ReviewMode | None = None,
) -> Path:
    """Join the approved clips and lay the original mix over them."""
    project = open_project(project_dir, review_mode)
    manifest = project.load_manifest()
    safe_title = "".join(c if c.isalnum() else "_" for c in manifest.title).strip("_") or "video"
    target = output or project.output_dir / f"{safe_title}{'_draft' if force else ''}.mp4"
    result = assemble_mod.assemble(project, manifest, target, force=force)
    project.storage.push()  # the final cut is not referenced by the manifest
    return result


# --- reports ----------------------------------------------------------------------------------


def qc_report(project_dir: ProjectRef, markdown: bool = False) -> str:
    """Count, per check, how many attempts passed, failed or were skipped, and why."""
    manifest = open_project(project_dir, pull=False).load_manifest()
    return report_mod.render(report_mod.summarize(manifest), markdown=markdown)


def calibrate(project_dir: ProjectRef, labels: Path | None = None) -> str:
    """Suggest QC thresholds from attempts labelled good or bad."""
    manifest = open_project(project_dir, pull=False).load_manifest()
    found = (
        calibrate_mod.load_labels(labels) if labels else calibrate_mod.labels_from_reviews(manifest)
    )
    if not found:
        raise MVDError(
            "No labels. Approve or reject shots in the review UI, or pass --labels with a "
            "CSV of shot,kind,attempt,label rows."
        )
    samples = calibrate_mod.collect_samples(manifest, found)
    return calibrate_mod.render(calibrate_mod.suggest(samples), len(found))


def export_schemas(directory: Path) -> list[Path]:
    """Write JSON Schemas for manifest.json and analysis.json."""
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for name, model in (("manifest", Manifest), ("analysis", Analysis)):
        path = directory / f"{name}.schema.json"
        path.write_text(json.dumps(model.model_json_schema(), indent=2) + "\n", encoding="utf-8")
        written.append(path)
    return written


def status_counts(manifest: Manifest) -> dict[str, Counter[str]]:
    """Per-check passed/failed/skipped counts across every attempt."""
    check_counts: dict[str, Counter[str]] = {}
    for shot in manifest.shots:
        for state in (shot.still, shot.video):
            for attempt in state.attempts:
                if attempt.qc is None:
                    continue
                for check in attempt.qc.checks:
                    check_counts.setdefault(check.name, Counter())[check.status.value] += 1
    return check_counts
