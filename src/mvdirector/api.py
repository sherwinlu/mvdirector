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
import unicodedata
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field
from functools import partial
from importlib import resources
from pathlib import Path

from mvdirector import assemble as assemble_mod
from mvdirector import calibrate as calibrate_mod
from mvdirector import media, mix_assemble
from mvdirector import report as report_mod
from mvdirector import setup as setup_mod
from mvdirector import tuning as tuning_mod
from mvdirector.audio.analysis import analyze_project
from mvdirector.audio.broadcast_qa import BroadcastAudioReport, analyze_broadcast_audio
from mvdirector.audio.lyrics import Transcriber
from mvdirector.audio.mix_analysis import analyze_mix_project
from mvdirector.audio.prepare import PrepareResult, prepare_project
from mvdirector.audio.stems import StemSeparator
from mvdirector.cleanup import CleanupPlan, apply_cleanup, plan_cleanup
from mvdirector.comfy.client import ComfyClient
from mvdirector.comfy.provider import ComfyEndpointProvider, StaticComfyProvider
from mvdirector.config import CONFIG_FILENAME, ReviewMode, validate_max_concurrency
from mvdirector.errors import BudgetExceeded, ConfigError, LLMError, MVDError
from mvdirector.generation import plates as plates_mod
from mvdirector.generation import previz as previz_mod
from mvdirector.generation import props as props_mod
from mvdirector.generation import reference as reference_mod
from mvdirector.generation import upscale as upscale_mod
from mvdirector.generation.preflight import check_assets
from mvdirector.generation.runner import RunBudget, RunEvent, ShotRunner, load_templates
from mvdirector.llm import LLMClient, build_llm
from mvdirector.llm.check import LLMCheckResult, ping_llm, run_llm_check
from mvdirector.llm.http import redact_url
from mvdirector.models import (
    Analysis,
    AssetKind,
    Manifest,
    ProjectMode,
    ReferenceSet,
    ShotType,
    TrackProfile,
    VideoStyle,
)
from mvdirector.planning.brief import ensure_writable, generate_brief, write_brief
from mvdirector.planning.mix_planner import build_mix_manifest
from mvdirector.planning.planner import build_manifest
from mvdirector.project import Project
from mvdirector.qc.factory import build_quality_gates
from mvdirector.retry import RetryPolicy
from mvdirector.review import policy as review_policy
from mvdirector.store import path_is_file

__all__ = [
    # types
    "BriefResult",
    "BroadcastAudioReport",
    "CleanupPlan",
    "EnvironmentReport",
    "GenerateResult",
    "LLMCheckOutcome",
    "Observer",
    "PrepareResult",
    "ProjectRef",
    "SetupOptions",
    "TuneResult",
    "ValidationCheck",
    # stages
    "analyze",
    "approve",
    "assemble",
    "audio_qc",
    "brief",
    "calibrate",
    "check_llm",
    "clean",
    "export_schemas",
    "generate",
    "generate_plates",
    "generate_previz",
    "generate_props",
    "generate_reference_candidates",
    "generate_reference_videos",
    "init_project",
    "inspect_environment",
    "load_status",
    "lock_plate",
    "lock_previz",
    "lock_prop",
    "lock_reference",
    "lock_reference_video",
    "open_project",
    "plan",
    "prepare",
    "qc_report",
    "refine_previz",
    "reject",
    "setup_project",
    "status_counts",
    "tune",
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


@contextmanager
def _close_llm(llm: LLMClient | None) -> Iterator[LLMClient | None]:
    try:
        yield llm
    finally:
        closer = getattr(llm, "close", None)
        if callable(closer):
            closer()


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


def _with_mode(project: Project, mode: ProjectMode | None) -> Project:
    if mode is not None and project.config.mode != mode:
        config = project.config.model_copy(deep=True)
        config.mode = mode
        return Project(project.root, config, state=project.state, storage=project.storage)
    return project


def _resolve_tracklist_arg(project: Project, tracklist: Path | None) -> Path | None:
    if tracklist is None:
        return None
    if path_is_file(tracklist):
        return tracklist
    if not tracklist.is_absolute():
        return project.path(tracklist)
    return tracklist


# --- setup ------------------------------------------------------------------------------------

_STARTER_MIX_TRACKLIST = {
    "title": "My Mix",
    "source": "weaveshift",
    "tracks": [
        {
            "start": "00:00",
            "title": "Opening Horizon",
            "bpm": 122.0,
            "mood_tags": ["atmospheric", "deep"],
            "genre_tags": ["melodic house"],
        },
        {
            "start": "04:00",
            "title": "Kinetic Pulse",
            "bpm": 126.0,
            "mood_tags": ["driving", "hypnotic"],
            "genre_tags": ["progressive techno"],
        },
        {
            "start": "08:00",
            "title": "Afterglow Drift",
            "bpm": 124.0,
            "mood_tags": ["warm", "expansive"],
            "genre_tags": ["electronica"],
        },
    ],
}


#: Platform and tooling debris that must never be copied into a user's new project. An editable
#: install serves `templates/` straight from the source checkout, so whatever Finder, Windows
#: Explorer or Python left lying there would otherwise land in every `mvd init`.
_JUNK_TEMPLATE_NAMES = (
    ".DS_Store",
    "Thumbs.db",
    "desktop.ini",
    "__pycache__",
    ".ipynb_checkpoints",
)


def _ignore_junk_files(_directory: str, names: list[str]) -> set[str]:
    return {
        name
        for name in names
        if name in _JUNK_TEMPLATE_NAMES or name.startswith("._") or name.endswith(".pyc")
    }


def init_project(directory: Path, *, mode: ProjectMode = ProjectMode.STANDARD) -> Path:
    """Create a project folder with a starter config and workflow binding examples."""
    if directory.exists() and not directory.is_dir():
        raise MVDError(f"{directory} already exists and is not a directory")
    if (directory / CONFIG_FILENAME).exists():
        raise MVDError(f"{directory} already contains {CONFIG_FILENAME}")
    template_root = resources.files("mvdirector") / "templates"
    with resources.as_file(template_root) as source:
        shutil.copytree(source, directory, dirs_exist_ok=True, ignore=_ignore_junk_files)
    for folder in ("inputs", "assets", "output"):
        (directory / folder).mkdir(exist_ok=True)
    if mode is ProjectMode.MIX:
        cfg_file = directory / CONFIG_FILENAME
        cfg_text = cfg_file.read_text(encoding="utf-8")
        cfg_text = cfg_text.replace("mode: standard", "mode: mix", 1)
        cfg_text = cfg_text.replace("tracklist: null", "tracklist: inputs/tracklist.json", 1)
        cfg_file.write_text(cfg_text, encoding="utf-8")
        tracklist_file = directory / "inputs" / "tracklist.json"
        tracklist_file.write_text(
            json.dumps(_STARTER_MIX_TRACKLIST, indent=2) + "\n", encoding="utf-8"
        )
    return directory


EnvironmentReport = setup_mod.EnvironmentReport
SetupOptions = setup_mod.SetupOptions


def inspect_environment(
    *,
    comfy_url: str | None = None,
    comfy_token_env: str | None = None,
    ollama_url: str | None = None,
) -> EnvironmentReport:
    """Inspect local tools, Python extras, Ollama, and ComfyUI readiness."""
    return setup_mod.inspect_environment(
        comfy_url=comfy_url,
        comfy_token_env=comfy_token_env,
        ollama_url=ollama_url,
    )


def setup_project(
    directory: Path,
    options: SetupOptions,
    *,
    mode: ProjectMode = ProjectMode.STANDARD,
) -> Path:
    """Initialize ``directory`` if needed and apply ``options`` to its ``project.yaml``."""
    setup_mod.validate_setup_options(options)
    if not (directory / CONFIG_FILENAME).exists():
        init_project(directory, mode=options.mode or mode)
    setup_mod.apply_setup_config(directory, options)
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
    if project.config.mode is ProjectMode.MIX:
        workflow_names = ["t2i"] if project.config.mix.visual_source == "stills" else ["t2i", "i2v"]
    elif project.config.planning.video_style is VideoStyle.VISUAL:
        workflow_names = ["t2i", "i2v"]
    elif project.config.planning.video_style is VideoStyle.NARRATIVE:
        workflow_names = ["t2i", "still", "i2v"]
    else:
        workflow_names = list(project.config.comfy.workflows)
    loaded_templates = {}
    for name in workflow_names:
        try:
            loaded = load_templates(project, [name])
            loaded_templates.update(loaded)
            checks.append(ValidationCheck(f"workflow '{name}'"))
        except MVDError as exc:
            checks.append(ValidationCheck(f"workflow '{name}'", str(exc)))

    live_wf_checks: list[ValidationCheck] = []

    def ping_and_probe() -> None:
        with _lease(project, comfy_provider) as comfy:
            comfy.ping()
            # Live /object_info node/model validation when server exposes /object_info
            # (inspired by VisualFrisson/ComfyUI-OneShotMV setup_check.py).
            info_fn = getattr(comfy, "object_info", None)
            if callable(info_fn):
                try:
                    info = info_fn()
                except MVDError:
                    info = {}
                if isinstance(info, dict) and info:
                    for wf_name, tpl in loaded_templates.items():
                        bound_targets = {
                            (str(t.node), t.field)
                            for targets in tpl.bindings.inputs.values()
                            for t in targets
                        }
                        issues = comfy.validate_graph_live(
                            tpl.graph, object_info=info, skip_inputs=bound_targets
                        )
                        if issues:
                            live_wf_checks.append(
                                ValidationCheck(
                                    f"ComfyUI dependencies for '{wf_name}'",
                                    "; ".join(issues[:5]),
                                )
                            )

    attempt(f"ComfyUI at {redact_url(project.config.comfy.base_url)}", ping_and_probe)
    checks.extend(live_wf_checks)
    llm_config = project.config.llm
    if llm_config.enabled:

        def _check_llm_config() -> None:
            with _close_llm(build_llm(llm_config)):
                pass

        attempt(
            f"LLM settings ({llm_config.provider}, {llm_config.model}); `mvd llm-check` tests them",
            _check_llm_config,
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
    with _close_llm(build_llm(config)) as client:
        if client is None:  # build_llm returns None only for a disabled model, handled above
            raise MVDError("No language model is configured.")
        # The image test only matters when vision checks are switched on.
        outcome.vision_needed = project.config.qc.vision_enabled
        outcome.tested_images = outcome.vision_needed and not skip_image
        outcome.result = run_llm_check(client, check_vision=outcome.tested_images)
    return outcome


# --- planning ---------------------------------------------------------------------------------

_UNSET = object()


def prepare(
    project_dir: ProjectRef,
    audio: Path | Sequence[Path] | None = None,
    *,
    overwrite: bool = False,
    separator: StemSeparator | None = None,
    transcriber: Transcriber | None = None,
    llm: LLMClient | None | object = _UNSET,
) -> PrepareResult:
    """Prepare a project from audio files or Suno/DAW stem packs (.zip, folders, or loose stems)."""
    project = open_project(project_dir)
    if llm is _UNSET:
        with _close_llm(build_llm(project.config.llm)) as owned_llm:
            return prepare_project(
                project,
                audio=audio,
                overwrite=overwrite,
                separator=separator,
                transcriber=transcriber,
                llm=owned_llm,
            )
    return prepare_project(
        project,
        audio=audio,
        overwrite=overwrite,
        separator=separator,
        transcriber=transcriber,
        llm=llm,  # type: ignore[arg-type]
    )


@dataclass(frozen=True)
class TuneResult:
    profile: TrackProfile
    adjustments: list[tuning_mod.Adjustment]
    applied_to_yaml: bool
    report: str


def analyze(
    project_dir: ProjectRef,
    *,
    apply_tuning: bool = False,
    mode: ProjectMode | None = None,
    tracklist: Path | None = None,
) -> tuple[Project, Analysis]:
    """Analyze the audio: beats, bars, vocal activity, lyric timing, profile, or mix chapters."""
    project = _with_mode(open_project(project_dir), mode)
    resolved_tracklist = _resolve_tracklist_arg(project, tracklist)
    if project.config.mode is ProjectMode.MIX:
        analysis = analyze_mix_project(project, tracklist_override=resolved_tracklist)
        project.save_analysis(analysis)
        return project, analysis
    if resolved_tracklist is not None:
        raise ConfigError(
            "--tracklist requires mix mode (--mode mix or mode: mix in project.yaml)."
        )
    analysis = analyze_project(project)
    if apply_tuning:
        with project.run_lock():
            project.save_analysis(analysis)
            if analysis.track_profile is not None:
                adjustments = tuning_mod.recommend_adjustments(
                    project.config, analysis.track_profile
                )
                if tuning_mod.apply_adjustments_to_yaml(
                    project.root / CONFIG_FILENAME, adjustments
                ):
                    project = Project.open(
                        project.root, state=project.state, storage=project.storage
                    )
    else:
        project.save_analysis(analysis)
    return project, analysis


def tune(
    project_dir: ProjectRef,
    *,
    apply: bool = False,
    force: bool = False,
) -> TuneResult:
    """Profile track characteristics from lyrics, style, and analysis, and tune project.yaml."""
    project = open_project(project_dir, pull=False)
    lyrics_text, style_text = tuning_mod.read_optional_project_inputs(project)
    try:
        analysis: Analysis | None = project.load_analysis()
    except MVDError:
        analysis = None
    profile = tuning_mod.profile_track(
        lyrics_text=lyrics_text,
        style_text=style_text,
        title_hint=project.config.title,
        analysis=analysis,
    )
    adjustments = tuning_mod.recommend_adjustments(project.config, profile, force=force)
    wrote_yaml = False
    if apply and any(adj.applied for adj in adjustments):
        with project.run_lock():
            project = open_project(project_dir, pull=False)
            adjustments = tuning_mod.recommend_adjustments(project.config, profile, force=force)
            if any(adj.applied for adj in adjustments):
                tuning_mod.apply_adjustments_to_yaml(project.root / CONFIG_FILENAME, adjustments)
                wrote_yaml = True
    report = tuning_mod.render_tune_report(profile, adjustments, applied_to_yaml=wrote_yaml)
    return TuneResult(
        profile=profile,
        adjustments=adjustments,
        applied_to_yaml=wrote_yaml,
        report=report,
    )


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
    with _close_llm(build_llm(project.config.llm)) as llm:
        generated = generate_brief(idea, llm=llm, duration_seconds=duration, vibe=vibe)
    with project.run_lock():
        ensure_writable(project, overwrite=overwrite)  # re-check: another job may have written
        lyrics_path, style_path = write_brief(project, generated, overwrite=overwrite)
        project.storage.push()
    return BriefResult(lyrics_path, style_path, generated.suno_style)


def _read_text_input(path: Path, label: str) -> str:
    try:
        return path.read_text("utf-8-sig")
    except (OSError, UnicodeDecodeError) as exc:
        raise ConfigError(f"Could not read {label} file {path}: {exc}") from exc


def plan(
    project_dir: ProjectRef,
    overwrite: bool = False,
    *,
    mode: ProjectMode | None = None,
    tracklist: Path | None = None,
    video_style: VideoStyle | None = None,
) -> tuple[Project, Manifest]:
    """Create the shot manifest from the analysis, lyrics/tracklist and style brief."""
    project = _with_mode(open_project(project_dir), mode)
    with project.run_lock():
        if project.state.has_manifest() and not overwrite:
            raise MVDError("manifest.json already exists. Pass --overwrite to replace it.")
        inputs = project.config.inputs
        if project.config.mode is ProjectMode.MIX:
            if video_style is not None:
                raise ConfigError("--style / video_style applies to standard mode, not mix mode.")
            resolved_tracklist = _resolve_tracklist_arg(project, tracklist)
            if resolved_tracklist is not None:
                analysis = analyze_mix_project(project, tracklist_override=resolved_tracklist)
                project.save_analysis(analysis)
            else:
                analysis = project.load_analysis()
            style_text = ""
            if inputs.style:
                style_path = project.path(inputs.style)
                if style_path.is_file():
                    style_text = _read_text_input(style_path, "style")
            with _close_llm(build_llm(project.config.llm)) as llm:
                manifest = build_mix_manifest(
                    project.config,
                    analysis,
                    style_text,
                    llm,
                )
        else:
            if tracklist is not None:
                raise ConfigError(
                    "--tracklist requires mix mode (--mode mix or mode: mix in project.yaml)."
                )
            plan_config = project.config
            if video_style is not None:
                plan_config = project.config.model_copy(deep=True)
                plan_config.planning.video_style = video_style
            mix_candidate = project.path(inputs.mix) if inputs.mix else None
            mix_path = mix_candidate if (mix_candidate and mix_candidate.is_file()) else None
            if plan_config.planning.video_style is VideoStyle.LIPSYNC:
                lyrics_text = _read_text_input(
                    project.input_path(inputs.lyrics, "lyrics"), "lyrics"
                )
            else:
                lyrics_candidate = project.path(inputs.lyrics) if inputs.lyrics else None
                lyrics_text = (
                    _read_text_input(lyrics_candidate, "lyrics")
                    if lyrics_candidate is not None and lyrics_candidate.is_file()
                    else ""
                )
            with _close_llm(build_llm(plan_config.llm)) as llm:
                manifest = build_manifest(
                    plan_config,
                    project.load_analysis(),
                    lyrics_text,
                    _read_text_input(project.input_path(inputs.style, "style"), "style"),
                    llm,
                    mix_path=mix_path,
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
            project.config.reference.candidates if count is None else count,
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
            project.config.reference.video_candidates if count is None else count,
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
    target_scene = manifest.visual_bible.find_set(set_name) if set_name is not None else None
    needs_still = (
        style_ref is not None
        or embed_set is not None
        or from_set is not None
        or (
            bool(target_scene and target_scene.parent_set)
            if set_name is not None
            else any(s.parent_set for s in manifest.visual_bible.settings)
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
            project.config.plates.candidates if count is None else count,
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
            project,
            manifest,
            comfy,
            template,
            prop,
            project.config.props.candidates if count is None else count,
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
    with _close_llm(build_llm(project.config.llm)) as llm, project.edit_manifest() as manifest:
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
    with _close_llm(build_llm(project.config.llm)) as llm, project.edit_manifest() as manifest:
        return previz_mod.refine_previz_shot(project, manifest, llm, shot_id, feedback)


# --- generation -------------------------------------------------------------------------------


@dataclass
class GenerateResult:
    project: Project
    manifest: Manifest
    active_checks: list[str] = field(default_factory=list)
    skipped_checks: list[tuple[str, str]] = field(default_factory=list)
    budget_message: str | None = None


def _resolve_target_shots(
    manifest: Manifest,
    shots: list[str] | None,
    chapters: list[str] | None,
) -> list[str] | None:
    if chapters is None:
        return list(shots) if shots is not None else None
    if manifest.mode is not ProjectMode.MIX:
        raise MVDError("--chapter is only supported on mix-mode projects.")
    try:
        chapter_shots = manifest.shots_for_chapters(chapters)
    except KeyError as exc:
        raise MVDError(str(exc)) from exc
    if shots is None:
        return chapter_shots
    combined = list(shots)
    seen = set(combined)
    combined.extend(sid for sid in chapter_shots if sid not in seen)
    return combined


def generate(
    project_dir: ProjectRef,
    kinds: list[AssetKind],
    shots: list[str] | None = None,
    review_mode: ReviewMode | None = None,
    skip_preflight: bool = False,
    draft: bool = False,
    observer: Observer | None = None,
    comfy_provider: ComfyEndpointProvider | None = None,
    *,
    chapters: list[str] | None = None,
    max_concurrency: int | None = None,
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
    effective_concurrency = validate_max_concurrency(
        project.config.comfy.max_concurrency if max_concurrency is None else max_concurrency
    )
    manifest = project.load_manifest()
    selected_shots = _resolve_target_shots(manifest, shots, chapters)

    effective_kinds = list(kinds)
    if manifest.mode is ProjectMode.MIX and project.config.mix.visual_source == "stills":
        if effective_kinds == [AssetKind.VIDEO]:
            raise MVDError(
                "mix.visual_source is 'stills'; set mix.visual_source: clips in project.yaml "
                "to generate video clips."
            )
        effective_kinds = [k for k in effective_kinds if k is not AssetKind.VIDEO]

    with _close_llm(build_llm(project.config.llm)) as llm:
        if manifest.mode is ProjectMode.MIX:
            names = ["t2i"] if AssetKind.STILL in effective_kinds else []
            names += ["i2v"] if AssetKind.VIDEO in effective_kinds else []
        else:
            active_shots = [
                s for s in manifest.shots if selected_shots is None or s.id in selected_shots
            ]
            names = []
            if AssetKind.STILL in effective_kinds:
                if not active_shots or any(s.type.shows_singer for s in active_shots):
                    names.append("still")
                if not active_shots or any(not s.type.shows_singer for s in active_shots):
                    names.append("t2i")
            if AssetKind.VIDEO in effective_kinds:
                has_lipsync = any(s.type is ShotType.LIPSYNC for s in active_shots)
                if has_lipsync:
                    names.append("lipsync")
                names.append("i2v")
                retry_cfg = project.config.retry
                wants_broll_rescue = has_lipsync and (
                    retry_cfg.smart_cutaway_rescue
                    or (retry_cfg.lipsync_fallback and retry_cfg.lipsync_fallback_mode == "broll")
                )
                if (
                    wants_broll_rescue
                    and "t2i" not in names
                    and "t2i" in project.config.comfy.workflows
                ):
                    t2i_bindings = project.path(
                        Path(project.config.comfy.workflows_dir)
                        / project.config.comfy.workflows["t2i"]
                    )
                    if t2i_bindings.is_file():
                        names.append("t2i")
        templates = load_templates(project, names)
        if AssetKind.STILL in effective_kinds and manifest.mode is not ProjectMode.MIX:
            ready = check_assets(project, manifest, templates, selected_shots)
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
            effective_provider: ComfyEndpointProvider = comfy_provider or StaticComfyProvider()
            with project.run_lock(), effective_provider.lease(project) as comfy:
                manifest = result.manifest = project.load_manifest()
                locked_shots = _resolve_target_shots(manifest, shots, chapters)
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
                    max_concurrency=effective_concurrency,
                    comfy_provider=effective_provider,
                )
                try:
                    runner.run(effective_kinds, locked_shots)
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
    *,
    width: int | None = None,
    height: int | None = None,
    on_shot_done: Callable[[str, int, int], None] | None = None,
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
            project,
            manifest,
            comfy,
            template,
            shot_ids=shots,
            force=force,
            overwrite=overwrite,
            width=width,
            height=height,
            on_shot_done=on_shot_done,
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


def approve(
    project_dir: ProjectRef,
    shot_ids: list[str] | None = None,
    still: bool = False,
    *,
    chapters: list[str] | None = None,
) -> None:
    """Approve shots (or chapters in mix mode), or accept an exhausted still."""

    def _apply(manifest: Manifest, project: Project) -> None:
        if shot_ids is None and chapters is None:
            raise MVDError("Pass at least one shot id or --chapter to approve.")
        targets = _resolve_target_shots(manifest, shot_ids, chapters) or []
        if still:
            for sid in targets:
                review_policy.accept_still(manifest, sid)
        else:
            stills_only = review_policy.uses_stills_only(manifest, project.config)
            for sid in targets:
                review_policy.approve(manifest, sid, stills_only=stills_only)

    _decide(project_dir, _apply)


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


def clean(project_dir: ProjectRef, *, delete: bool = False) -> CleanupPlan:
    """Plan, and with ``delete=True`` carry out, the removal of generations no longer needed.

    Deleting takes the run lock, so it cannot overlap a run that is still writing.
    """
    project = open_project(project_dir)
    if not delete:
        return plan_cleanup(project, project.load_manifest())
    with project.run_lock():
        manifest = project.load_manifest()
        plan = plan_cleanup(project, manifest)
        if plan.files:
            apply_cleanup(project, manifest, plan)
            project.save_manifest(manifest)
    return plan


def _decide(project_dir: ProjectRef, action: Callable[[Manifest, Project], object]) -> None:
    project = open_project(project_dir, pull=False)
    try:
        with project.edit_manifest() as manifest:
            action(manifest, project)
    except (KeyError, ValueError) as exc:
        raise MVDError(str(exc)) from exc


def _safe_title(title: str) -> str:
    chars: list[str] = []
    for ch in unicodedata.normalize("NFC", title):
        if ch.isalnum() or (
            chars and chars[-1] != "_" and unicodedata.category(ch).startswith("M")
        ):
            chars.append(ch)
        else:
            chars.append("_")
    return "".join(chars).strip("_")[:120].strip("_") or "video"


def assemble(
    project_dir: ProjectRef,
    output: Path | None = None,
    force: bool = False,
    review_mode: ReviewMode | None = None,
    *,
    draft: bool = False,
    width: int | None = None,
    height: int | None = None,
    chapters: list[str] | None = None,
    rebuild_cache: bool = False,
) -> Path:
    """Join the approved clips (or cached mix chapters) and lay the original mix over them."""
    project = open_project(project_dir, review_mode)
    manifest = project.load_manifest()
    safe_title = _safe_title(manifest.title)
    suffix = "_draft" if (force or draft) else ""
    target = output or project.output_dir / f"{safe_title}{suffix}.mp4"
    if manifest.mode is ProjectMode.MIX:
        result = mix_assemble.assemble_mix(
            project,
            manifest,
            target,
            force=force,
            draft=draft,
            width=width,
            height=height,
            chapters=chapters,
            rebuild_cache=rebuild_cache,
        )
    else:
        if chapters is not None or rebuild_cache:
            raise MVDError("--chapter and --rebuild-cache are only supported on mix-mode projects.")
        result = assemble_mod.assemble(
            project, manifest, target, force=force, draft=draft, width=width, height=height
        )
    project.storage.push()  # the final cut is not referenced by the manifest
    return result


# --- reports ----------------------------------------------------------------------------------


def audio_qc(project_dir: ProjectRef, audio_path: Path | None = None) -> BroadcastAudioReport:
    """Measure broadcast loudness (LUFS), 4x True Peak, and splice clicks on the project mix."""
    project = open_project(project_dir, pull=False)
    target = (
        audio_path
        if audio_path is not None
        else project.input_path(project.config.inputs.mix, "mix")
    )
    boundary_times: list[float] | None = None
    drum_onsets: list[float] = []
    try:
        analysis = project.load_analysis()
        drum_onsets = list(analysis.beats)
        if analysis.chapters:
            boundary_times = [ch.start for ch in analysis.chapters if ch.start > 0.0]
    except MVDError:
        pass
    return analyze_broadcast_audio(
        target,
        boundary_times=boundary_times,
        drum_onset_times=drum_onsets,
    )


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
