# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Command-line interface. Each pipeline stage is one command and can be re-run alone."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import typer

from mvdirector import api, mix_assemble
from mvdirector.config import ReviewMode
from mvdirector.errors import MVDError
from mvdirector.generation.runner import RunEvent
from mvdirector.models import AssetKind, Character, Manifest, ProjectMode, VideoStyle
from mvdirector.project import Project
from mvdirector.review import policy as review_policy

app = typer.Typer(add_completion=False, no_args_is_help=True, help=__doc__)
reference_app = typer.Typer(
    no_args_is_help=True, help="Create and lock the singer's reference sheets."
)
app.add_typer(reference_app, name="reference")
plates_app = typer.Typer(no_args_is_help=True, help="Create and lock an image of each set.")
app.add_typer(plates_app, name="plates")
props_app = typer.Typer(
    no_args_is_help=True, help="Create and lock two-angle reference sheets for hero props."
)
app.add_typer(props_app, name="props")
previz_app = typer.Typer(
    no_args_is_help=True, help="Generate or lock 3D blocking previz clips for shots."
)
app.add_typer(previz_app, name="previz")

log = logging.getLogger("mvdirector")

ProjectDir = typer.Option(Path("."), "--project", "-p", help="Project folder.")
ModeOption = typer.Option(None, "--review-mode", help="Override review.mode from the config.")
ShotsOption = typer.Option(None, "--shot", "-s", help="Limit to these shot ids (repeatable).")
ChaptersOption = typer.Option(
    None, "--chapter", "-C", help="Limit to these mix chapter ids (repeatable)."
)
SkipPreflight = typer.Option(
    False,
    "--skip-preflight",
    help="Generate stills even if reference sheets or plates are missing.",
)
DraftOption = typer.Option(
    False,
    "--draft",
    help="Render at draft resolution (video.draft_width/height, or half target resolution).",
)
StateOption = typer.Option(None, "--state", help="Wardrobe state id. Defaults to the first.")
CharacterOption = typer.Option(
    None,
    "--character",
    "-c",
    help="Character id (for duets or bands). Defaults to the primary character.",
)


def _version_callback(value: bool) -> None:
    if value:
        from importlib.metadata import PackageNotFoundError, version

        try:
            typer.echo(f"mvd {version('mvdirector')}")
        except PackageNotFoundError:  # running from a source tree without an install
            typer.echo("mvd (unknown version)")
        raise typer.Exit()


@app.callback()
def main(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug logging."),
    version: bool = typer.Option(
        False,
        "--version",
        help="Print the mvdirector version and exit.",
        callback=_version_callback,
        is_eager=True,
    ),
) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)


def _fail(exc: MVDError) -> typer.Exit:
    typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
    return typer.Exit(code=1)


def _yellow(text: str) -> None:
    typer.secho(text, fg=typer.colors.YELLOW)


def _lines(paths) -> str:  # noqa: ANN001
    return "\n".join(f"  {p}" for p in paths)


@app.command()
def init(
    directory: Path = typer.Argument(..., help="Folder to create the project in."),
    mode: ProjectMode = typer.Option(
        ProjectMode.STANDARD,
        "--mode",
        help="Pipeline mode: 'standard' (single track) or 'mix' (long-form continuous mix).",
    ),
) -> None:
    """Create a project folder with a starter config and workflow binding examples."""
    try:
        api.init_project(directory, mode=mode)
    except MVDError as exc:
        raise _fail(exc) from exc
    next_hint = (
        "add your mix audio and edit inputs/tracklist.json, then run `mvd analyze`."
        if mode is ProjectMode.MIX
        else "add your audio to inputs/, then read workflows/README.md."
    )
    typer.echo(f"Created {mode.value} project in {directory}. Next: {next_hint}")


@app.command()
def prepare(
    audio: list[Path] | None = typer.Argument(
        None,
        help=(
            "Audio file(s), stem folder, or Suno stem .zip archive(s) "
            "(defaults to inputs/mix.wav, inputs/mix.mp3, or stems dropped in inputs/stems/)."
        ),
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Re-extract/re-separate stems and replace existing mix, lyrics.txt, and style.md.",
    ),
    project_dir: Path = ProjectDir,
) -> None:
    """Build mix, vocal stem, lyrics.txt, and style.md from audio or Suno/DAW stem packs."""
    try:
        audio_arg: Path | list[Path] | None = (
            None if not audio else (audio[0] if len(audio) == 1 else audio)
        )
        result = api.prepare(project_dir, audio=audio_arg, overwrite=overwrite)
    except MVDError as exc:
        raise _fail(exc) from exc
    mode = "LLM" if result.inferred_with_llm else "fallback (llm disabled)"
    stems_note = f", {result.stems_ingested} stems" if result.stems_ingested else ""
    typer.echo(
        f"Prepared inputs from {result.mix_path} ({result.duration:.1f}s, "
        f"{result.tempo_bpm:.0f} BPM, {result.vocal_windows} vocal windows, "
        f"{result.transcribed_words} transcribed words{stems_note}, inferred via {mode}):"
    )
    written = [result.mix_path, result.vocals_path]
    if result.drums_path is not None:
        written.append(result.drums_path)
    written.extend([result.lyrics_path, result.style_path])
    if result.word_timestamps_path is not None:
        written.append(result.word_timestamps_path)
    typer.echo(_lines(written))
    typer.echo(
        "Review/edit lyrics.txt and style.md if desired, then run `mvd analyze` and `mvd plan`."
    )


@app.command()
def analyze(
    project_dir: Path = ProjectDir,
    tune_config: bool = typer.Option(
        False,
        "--tune",
        help="Write auto-tuned track parameters into project.yaml.",
    ),
    mode: ProjectMode | None = typer.Option(
        None, "--mode", help="Override project mode ('standard' or 'mix')."
    ),
    tracklist: Path | None = typer.Option(
        None,
        "--tracklist",
        help="Cue sheet or tracklist (.json, .yaml, .cue, .txt) when running in mix mode.",
    ),
) -> None:
    """Analyze the audio: beats, bars, vocal activity, lyric timing, profile, or mix chapters."""
    try:
        project, analysis = api.analyze(
            project_dir, apply_tuning=tune_config, mode=mode, tracklist=tracklist
        )
    except MVDError as exc:
        raise _fail(exc) from exc
    if analysis.chapters:
        typer.echo(
            f"Wrote {project.analysis_path}: {len(analysis.chapters)} chapters, "
            f"{analysis.tempo_bpm:.0f} BPM mean, {len(analysis.bars)} bars."
        )
    else:
        typer.echo(
            f"Wrote {project.analysis_path}: {analysis.tempo_bpm:.0f} BPM, "
            f"{len(analysis.bars)} bars, {len(analysis.vocal_windows)} vocal windows."
        )
    if tune_config and analysis.track_profile is not None:
        typer.echo(f"Updated {project.root / 'project.yaml'} with track-specific settings.")


@app.command()
def tune(
    apply: bool = typer.Option(
        False,
        "--apply",
        help="Write recommended parameter adjustments into project.yaml.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Override non-default custom settings in project.yaml as well.",
    ),
    project_dir: Path = ProjectDir,
) -> None:
    """Inspect track lyrics, style, and audio analysis to recommend or apply project.yaml tuning."""
    try:
        result = api.tune(project_dir, apply=apply, force=force)
    except MVDError as exc:
        raise _fail(exc) from exc
    typer.echo(result.report)


@app.command()
def brief(
    idea: str = typer.Option(..., "--idea", help="Song story and visual concept."),
    duration: int | None = typer.Option(
        None, "--duration", min=5, max=600, help="Target song length in seconds (e.g. 40)."
    ),
    vibe: str | None = typer.Option(
        None, "--vibe", help="Optional genre, vocal style, and visual aesthetic cues."
    ),
    overwrite: bool = typer.Option(
        False, "--overwrite", help="Replace non-starter lyrics.txt / style.md files."
    ),
    project_dir: Path = ProjectDir,
) -> None:
    """Draft Suno-ready lyrics.txt and style.md from a concept before generating your song."""
    try:
        result = api.brief(project_dir, idea, duration=duration, vibe=vibe, overwrite=overwrite)
    except MVDError as exc:
        raise _fail(exc) from exc
    typer.echo(f"Wrote {result.lyrics_path} and {result.style_path}.")
    typer.echo(f"\nSuno style prompt:\n{result.suno_style.strip()}")


@app.command()
def plan(
    project_dir: Path = ProjectDir,
    overwrite: bool = typer.Option(False, help="Replace an existing manifest and its history."),
    mode: ProjectMode | None = typer.Option(
        None, "--mode", help="Override project mode ('standard' or 'mix')."
    ),
    tracklist: Path | None = typer.Option(
        None,
        "--tracklist",
        help="Cue sheet or tracklist (.json, .yaml, .cue, .txt) when planning in mix mode.",
    ),
    style: VideoStyle | None = typer.Option(
        None,
        "--style",
        help=(
            "Override standard-mode video style: 'lipsync', 'narrative' "
            "(characters, no lip-sync), or 'visual' (100% b-roll, no characters)."
        ),
    ),
) -> None:
    """Create the shot manifest from the analysis, lyrics/tracklist and style brief."""
    try:
        project, manifest = api.plan(
            project_dir, overwrite=overwrite, mode=mode, tracklist=tracklist, video_style=style
        )
    except MVDError as exc:
        raise _fail(exc) from exc
    if manifest.mode is ProjectMode.MIX:
        typer.echo(
            f"Wrote {project.manifest_path}: {len(manifest.chapters)} chapters, "
            f"{len(manifest.shots)} loop units (visual_source={project.config.mix.visual_source})."
        )
        return
    kinds = Counter(shot.type.value for shot in manifest.shots)
    typer.echo(
        f"Wrote {project.manifest_path}: {len(manifest.shots)} shots {dict(kinds)}. "
        "Review and edit it before generating."
    )


@app.command()
def validate(project_dir: Path = ProjectDir) -> None:
    """Check tools, workflow bindings and the ComfyUI connection before spending GPU time."""
    try:
        checks = api.validate_project(project_dir)
    except MVDError as exc:
        raise _fail(exc) from exc
    for check in checks:
        mark, colour = ("ok", typer.colors.GREEN) if check.ok else ("FAIL", typer.colors.RED)
        typer.secho(
            f"[{mark:>4}] {check.label}" + (f": {check.problem}" if check.problem else ""),
            fg=colour,
        )
    if not all(check.ok for check in checks):
        raise typer.Exit(code=1)


@app.command("llm-check")
def llm_check(
    project_dir: Path = ProjectDir,
    skip_image: bool = typer.Option(False, "--skip-image", help="Do not test image input."),
) -> None:
    """Test the configured model: credentials, JSON output and image input."""
    try:
        outcome = api.check_llm(project_dir, skip_image=skip_image)
    except MVDError as exc:
        raise _fail(exc) from exc
    if not outcome.enabled or outcome.result is None:
        typer.echo("llm.enabled is false: planning is rule-based and no model is used.")
        return
    result = outcome.result
    typer.echo(f"Provider: {outcome.provider}   Model: {outcome.model}")

    def line(ok: bool, text: str) -> None:
        colour = typer.colors.GREEN if ok else typer.colors.RED
        typer.secho(f"[{'ok' if ok else 'FAIL':>4}] {text}", fg=colour)

    line(
        result.json_ok,
        f"JSON output ({result.json_seconds:.1f} s)"
        + (f": {result.json_detail}" if result.json_detail else ""),
    )
    if not outcome.vision_needed:
        _yellow("[skip] Image input not tested: qc.vision_enabled is false.")
    elif outcome.tested_images and result.vision != "skipped":
        detail = f": {result.vision_detail}" if result.vision_detail else ""
        line(not outcome.vision_failed, f"Image input ({result.vision_seconds:.1f} s){detail}")
        if result.vision in ("unsupported", "wrong"):
            _yellow(
                "Without image input the vision checks cannot run. Choose a model that accepts "
                "images, or set qc.vision_enabled: false in project.yaml."
            )
    if not outcome.ok:
        raise typer.Exit(code=1)


@reference_app.command("generate")
def reference_generate(
    project_dir: Path = ProjectDir,
    state: str = StateOption,
    character: str = CharacterOption,
    from_character: str = typer.Option(
        None,
        "--from-character",
        help="Derive this character's base sheet by editing another character's locked sheet.",
    ),
    count: int = typer.Option(None, help="Number of candidates. Defaults to reference.candidates."),
) -> None:
    """Generate candidate reference sheets (front, profile, 3/4 and face panels)."""
    try:
        paths = api.generate_reference_candidates(
            project_dir, state, character, from_character=from_character, count=count
        )
    except MVDError as exc:
        raise _fail(exc) from exc
    typer.echo("Candidates:\n" + _lines(paths))
    char_flag = f" --character {character}" if character else ""
    flag = f" --state {state}" if state else ""
    typer.echo(f"Pick one with: mvd reference lock <path>{char_flag}{flag}")


@reference_app.command("lock")
def reference_lock(
    image: Path = typer.Argument(..., help="Sheet to use (a candidate, or your own image)."),
    project_dir: Path = ProjectDir,
    state: str = StateOption,
    character: str = CharacterOption,
    single: bool = typer.Option(
        False, "--single", help="The image is one portrait, not a four-panel sheet."
    ),
) -> None:
    """Lock a reference sheet, cut it into panels, and use it for this wardrobe state."""
    try:
        locked = api.lock_reference(project_dir, image, state, character, single=single)
    except MVDError as exc:
        raise _fail(exc) from exc
    typer.echo(f"Locked {locked.sheet} with {len(locked.panels)} panel(s).")


@reference_app.command("video")
def reference_video(
    project_dir: Path = ProjectDir,
    state: str = StateOption,
    character: str = CharacterOption,
    count: int = typer.Option(
        None, help="Number of candidates. Defaults to reference.video_candidates."
    ),
) -> None:
    """Generate candidate silent turnaround reference videos from a locked reference sheet."""
    try:
        paths = api.generate_reference_videos(project_dir, state, character, count=count)
    except MVDError as exc:
        raise _fail(exc) from exc
    typer.echo("Reference video candidates:\n" + _lines(paths))
    char_flag = f" --character {character}" if character else ""
    flag = f" --state {state}" if state else ""
    typer.echo(f"Pick one with: mvd reference lock-video <path>{char_flag}{flag}")


@reference_app.command("lock-video")
def reference_lock_video(
    video: Path = typer.Argument(..., help="Turnaround reference video to lock for this state."),
    project_dir: Path = ProjectDir,
    state: str = StateOption,
    character: str = CharacterOption,
) -> None:
    """Lock a silent turnaround reference video for this character/wardrobe state."""
    try:
        locked = api.lock_reference_video(project_dir, video, state, character)
    except MVDError as exc:
        raise _fail(exc) from exc
    typer.echo(f"Locked reference video: {locked}")


@plates_app.command("generate")
def plates_generate(
    project_dir: Path = ProjectDir,
    set_name: str = typer.Option(None, "--set", help="Set name. Defaults to every set."),
    style_ref: Path = typer.Option(
        None, "--style-ref", help="Optional reference image to seed the visual style."
    ),
    embed_set: str = typer.Option(
        None,
        "--embed-set",
        help="Locked set plate to embed inside a screen/diorama in the target set.",
    ),
    from_set: str = typer.Option(
        None, "--from-set", help="Derive this set's plate by editing another locked set plate."
    ),
    count: int = typer.Option(None, help="Candidates per set. Defaults to plates.candidates."),
) -> None:
    """Generate candidate images of the empty sets."""
    try:
        results = api.generate_plates(
            project_dir,
            set_name,
            style_ref=style_ref,
            embed_set=embed_set,
            from_set=from_set,
            count=count,
        )
    except MVDError as exc:
        raise _fail(exc) from exc
    for name, paths in results.items():
        typer.echo(f"{name}:\n" + _lines(paths))
    typer.echo('Pick one per set with: mvd plates lock <path> --set "<set name>"')


@plates_app.command("lock")
def plates_lock(
    image: Path = typer.Argument(..., help="Image to use for the set."),
    set_name: str = typer.Option(..., "--set", help="Set name."),
    project_dir: Path = ProjectDir,
) -> None:
    """Lock an image as a set's plate, used as a second reference for stills."""
    try:
        locked = api.lock_plate(project_dir, image, set_name)
    except MVDError as exc:
        raise _fail(exc) from exc
    typer.echo(f"Locked plate: {locked}")


def _reference_status(char: Character, key: str, index: int) -> str:
    reference = char.references.get(key) or (char.default_reference if index == 0 else None)
    if not reference:
        return "MISSING"
    video_note = " + video" if reference.video else ""
    return f"locked, {len(reference.panels)} panel(s){video_note}"


@props_app.command("generate")
def props_generate(
    project_dir: Path = ProjectDir,
    prop: str = typer.Option(None, "--prop", help="Prop id or name. Defaults to every prop."),
    count: int = typer.Option(None, help="Candidates per prop. Defaults to props.candidates."),
) -> None:
    """Generate candidate two-angle reference sheets for recurring hero props."""
    try:
        results = api.generate_props(project_dir, prop, count=count)
    except MVDError as exc:
        raise _fail(exc) from exc
    for name, paths in results.items():
        typer.echo(f"{name}:\n" + _lines(paths))
    typer.echo('Pick one per prop with: mvd props lock <path> --prop "<prop id or name>"')


@props_app.command("lock")
def props_lock(
    image: Path = typer.Argument(..., help="Image to use for the prop."),
    prop: str = typer.Option(..., "--prop", help="Prop id or name."),
    project_dir: Path = ProjectDir,
) -> None:
    """Lock a two-angle prop sheet, used as a reference on shots containing that prop."""
    try:
        locked = api.lock_prop(project_dir, image, prop)
    except MVDError as exc:
        raise _fail(exc) from exc
    typer.echo(f"Locked prop sheet: {locked}")


@previz_app.command("generate")
def previz_generate(
    project_dir: Path = ProjectDir,
    shot: list[str] = ShotsOption,
    overwrite: bool = typer.Option(False, help="Replace existing previz videos."),
) -> None:
    """Plan and render flat-shaded 3D blocking previz MP4s for selected shots."""
    try:
        written = api.generate_previz(project_dir, shot or None, overwrite=overwrite)
    except MVDError as exc:
        raise _fail(exc) from exc
    if not written:
        typer.echo("No new previz clips written (pass --overwrite to regenerate).")
        return
    typer.echo("Generated previz:\n" + _lines(written))


@previz_app.command("lock")
def previz_lock(
    video: Path = typer.Argument(..., help="External 3D blocking MP4 to attach to the shot."),
    shot_id: str = typer.Option(..., "--shot", "-s", help="Target shot id (e.g. s001)."),
    mapping: str = typer.Option(None, "--mapping", help="Stand-in mapping description."),
    project_dir: Path = ProjectDir,
) -> None:
    """Attach a custom 3D blocking video (from Three.js, Blender, etc.) to a shot."""
    try:
        locked = api.lock_previz(project_dir, video, shot_id, mapping=mapping)
    except MVDError as exc:
        raise _fail(exc) from exc
    typer.echo(f"Locked previz for {shot_id}: {locked}")


@previz_app.command("refine")
def previz_refine(
    shot_id: str = typer.Option(..., "--shot", "-s", help="Target shot id (e.g. s001)."),
    feedback: str = typer.Option(
        ..., "--feedback", "-f", help="Camera or staging changes to apply to the existing plan."
    ),
    project_dir: Path = ProjectDir,
) -> None:
    """Refine a shot's 3D blocking plan with LLM feedback and re-render its previz MP4."""
    try:
        rendered = api.refine_previz(project_dir, shot_id, feedback)
    except MVDError as exc:
        raise _fail(exc) from exc
    typer.echo(f"Refined previz for {shot_id}: {rendered}")


@app.command()
def assets(project_dir: Path = ProjectDir) -> None:
    """Show which reference sheets and set plates are locked."""
    try:
        manifest = api.open_project(project_dir).load_manifest()
    except MVDError as exc:
        raise _fail(exc) from exc
    character, bible = manifest.character, manifest.visual_bible
    typer.echo("Reference sheets")
    if len(manifest.characters) > 1:
        for char_id, char in manifest.characters.items():
            for index, key in enumerate(char.wardrobe_ids or ["base"]):
                typer.echo(f"  {f'{char_id}:{key}':<24} {_reference_status(char, key, index)}")
    else:
        for index, key in enumerate(character.wardrobe_ids or ["base"]):
            typer.echo(f"  {key:<16} {_reference_status(character, key, index)}")
    typer.echo("Set plates")
    for scene in bible.settings:
        typer.echo(f"  {scene.name:<28} {'locked' if scene.plate else 'not set'}")
    if not bible.settings:
        typer.echo("  (no sets in the bible)")
    if bible.props:
        typer.echo("Prop sheets")
        for prop in bible.props:
            typer.echo(f"  {prop.name:<28} {'locked' if prop.plate else 'not set'}")


def _generate(
    project_dir: Path,
    kinds: list[AssetKind],
    shots: list[str] | None,
    mode: ReviewMode | None,
    skip_preflight: bool = False,
    draft: bool = False,
    chapters: list[str] | None = None,
) -> None:
    try:
        result = api.generate(
            project_dir,
            kinds,
            shots,
            mode,
            skip_preflight=skip_preflight,
            draft=draft,
            observer=_progress,
            chapters=chapters,
        )
    except MVDError as exc:
        raise _fail(exc) from exc
    if result.budget_message:
        _yellow(result.budget_message)
    _print_status(result.project, result.manifest)


@app.command()
def stills(
    project_dir: Path = ProjectDir,
    shot: list[str] = ShotsOption,
    chapter: list[str] = ChaptersOption,
    skip_preflight: bool = SkipPreflight,
    draft: bool = DraftOption,
) -> None:
    """Generate and quality-check stills, regenerating failures automatically."""
    _generate(
        project_dir,
        [AssetKind.STILL],
        shot or None,
        None,
        skip_preflight,
        draft=draft,
        chapters=chapter or None,
    )


@app.command()
def videos(
    project_dir: Path = ProjectDir,
    shot: list[str] = ShotsOption,
    chapter: list[str] = ChaptersOption,
    review_mode: ReviewMode = ModeOption,
    draft: bool = DraftOption,
) -> None:
    """Generate and quality-check videos for shots whose still has passed."""
    _generate(
        project_dir,
        [AssetKind.VIDEO],
        shot or None,
        review_mode,
        draft=draft,
        chapters=chapter or None,
    )


@app.command()
def run(
    project_dir: Path = ProjectDir,
    shot: list[str] = ShotsOption,
    chapter: list[str] = ChaptersOption,
    review_mode: ReviewMode = ModeOption,
    skip_preflight: bool = SkipPreflight,
    draft: bool = DraftOption,
) -> None:
    """Generate stills, then videos, with automatic quality checks and retries."""
    _generate(
        project_dir,
        [AssetKind.STILL, AssetKind.VIDEO],
        shot or None,
        review_mode,
        skip_preflight,
        draft=draft,
        chapters=chapter or None,
    )


@app.command()
def upscale(
    project_dir: Path = ProjectDir,
    shot: list[str] = ShotsOption,
    force: bool = typer.Option(
        False, "--force", help="Upscale the best available clip even if pending review."
    ),
    overwrite: bool = typer.Option(False, help="Re-upscale clips that were already upscaled."),
    width: int | None = typer.Option(None, help="Override target upscale width (even >= 64)."),
    height: int | None = typer.Option(None, help="Override target upscale height (even >= 64)."),
) -> None:
    """Upscale winning/approved draft clips through the optional 'upscale' workflow."""
    try:
        paths = api.upscale(
            project_dir,
            shot or None,
            force=force,
            overwrite=overwrite,
            width=width,
            height=height,
        )
    except MVDError as exc:
        raise _fail(exc) from exc
    if not paths:
        typer.echo("No clips needed upscaling.")
        return
    typer.echo(f"Upscaled {len(paths)} clip(s):\n" + _lines(paths))


def _progress(event: RunEvent) -> None:
    """Print one line per attempt so a long run shows progress without debug logging."""
    if event.kind == "warning":
        _yellow(event.message)
    elif event.kind == "qc_summary":
        typer.echo(f"QC active: {', '.join(event.active_checks)}")
        if event.skipped_checks:
            _yellow(
                "QC skipped: "
                + "; ".join(f"{names} ({reason})" for names, reason in event.skipped_checks)
            )
    elif event.kind == "phase_started":
        typer.echo(f"== {event.asset}s ==")
    elif event.kind == "attempt_finished":
        if event.error:
            outcome = f"error: {event.error}"
        elif event.passed:
            outcome = f"passed ({event.score:.2f})"
        else:
            outcome = f"failed ({event.score:.2f}): " + "; ".join(event.issues)
        typer.echo(
            f"[{event.generations_used:>3}] {event.shot_id} {event.asset} attempt "
            f"{event.attempt} ({event.strategy}) {outcome}"
        )
    elif event.kind == "shot_converted":
        typer.secho(
            f"      {event.shot_id} {event.asset} converted from lipsync to performance; retrying",
            fg=typer.colors.YELLOW,
        )
    elif event.kind == "asset_exhausted":
        typer.secho(
            f"      {event.shot_id} {event.asset} exhausted; queued for review",
            fg=typer.colors.YELLOW,
        )


@app.command("qc-report")
def qc_report(
    project_dir: Path = ProjectDir,
    markdown: bool = typer.Option(False, "--markdown", help="Markdown tables, for a README."),
) -> None:
    """Count, per check, how many attempts passed, failed or were skipped, and why."""
    try:
        text = api.qc_report(project_dir, markdown=markdown)
    except MVDError as exc:
        raise _fail(exc) from exc
    typer.echo(text, nl=False)


@app.command()
def calibrate(
    project_dir: Path = ProjectDir,
    labels: Path | None = typer.Option(
        None,
        "--labels",
        help="CSV or JSON with shot, kind, attempt, label (good|bad). Default: review decisions.",
    ),
) -> None:
    """Suggest QC thresholds from attempts you have labelled good or bad."""
    try:
        text = api.calibrate(project_dir, labels)
    except MVDError as exc:
        raise _fail(exc) from exc
    typer.echo(text, nl=False)


@app.command()
def status(project_dir: Path = ProjectDir, review_mode: ReviewMode = ModeOption) -> None:
    """Show progress and the review queue."""
    try:
        project, manifest = api.load_status(project_dir, review_mode)
    except MVDError as exc:
        raise _fail(exc) from exc
    _print_status(project, manifest)


def _print_status(project: Project, manifest: Manifest) -> None:
    stills_only = review_policy.uses_stills_only(manifest, project.config)
    stills_by = Counter(s.still.status.value for s in manifest.shots)
    videos_by = Counter(s.video.status.value for s in manifest.shots)
    queue = review_policy.pending_review(manifest)
    ready = sum(1 for s in manifest.shots if review_policy.is_ready(s, stills_only=stills_only))
    if manifest.mode is ProjectMode.MIX:
        cached = mix_assemble.cached_segment_count(project, manifest)
        typer.echo(
            f"Mode: mix   Chapters: {len(manifest.chapters)}   "
            f"Cached segments: {cached}/{len(manifest.chapters)}"
        )
    typer.echo(f"Shots: {len(manifest.shots)}   review mode: {project.config.review.mode.value}")
    typer.echo(f"Stills: {dict(stills_by)}")
    if not stills_only:
        typer.echo(f"Videos: {dict(videos_by)}")
    typer.echo(
        f"Ready for assembly: {ready}/{len(manifest.shots)}   "
        f"generations so far: {manifest.total_generations}"
    )
    check_counts = api.status_counts(manifest)
    if check_counts:
        typer.echo("QC checks across attempts:")
        for name, counts in sorted(check_counts.items()):
            typer.echo(
                f"  {name:<18} passed={counts['passed']}  "
                f"failed={counts['failed']}  skipped={counts['skipped']}"
            )
    if queue:
        typer.echo(f"Awaiting your review ({len(queue)}):")
        for shot in queue:
            reasons = ", ".join(r.value for r in shot.review.reasons)
            typer.echo(f"  {shot.id}  {shot.type.value:<11} {shot.framing.value:<12} [{reasons}]")


@app.command()
def approve(
    shot_ids: list[str] = typer.Argument(None, help="Shot ids to approve."),
    chapter: list[str] = ChaptersOption,
    still: bool = typer.Option(
        False, "--still", help="Accept an exhausted still so its video can be generated."
    ),
    project_dir: Path = ProjectDir,
) -> None:
    """Approve shots (or --chapter in mix mode) from the command line."""
    if not shot_ids and not chapter:
        raise _fail(MVDError("Pass at least one shot id or --chapter to approve."))
    try:
        api.approve(project_dir, shot_ids or None, still=still, chapters=chapter or None)
    except MVDError as exc:
        raise _fail(exc) from exc
    typer.echo("Saved.")


@app.command()
def clean(
    yes: bool = typer.Option(
        False, "--yes", help="Delete the files. Without it, only list what would go."
    ),
    project_dir: Path = ProjectDir,
) -> None:
    """Delete takes that lost to a selected attempt, and generations the manifest never used.

    Assets that are pending or exhausted keep all their takes, and references, plates, props,
    audio and your inputs are never touched.
    """
    try:
        plan = api.clean(project_dir, delete=yes)
    except MVDError as exc:
        raise _fail(exc) from exc
    verb = "Deleted" if yes else "Would delete"
    typer.echo(
        f"{verb} {len(plan.superseded)} superseded and {len(plan.orphaned)} orphaned files "
        f"({plan.size_bytes / 1e6:.0f} MB)." + ("" if yes else " Run again with --yes to delete.")
    )


@app.command()
def reject(
    shot_id: str = typer.Argument(...),
    note: str = typer.Option(..., "--note", "-n", help="What to change; guides the next attempt."),
    target: AssetKind = typer.Option(
        AssetKind.VIDEO, help="Regenerate the video, or the still too."
    ),
    project_dir: Path = ProjectDir,
) -> None:
    """Reject a shot and send it back for regeneration with your note."""
    try:
        api.reject(project_dir, shot_id, note, target)
    except MVDError as exc:
        raise _fail(exc) from exc
    typer.echo("Saved.")


@app.command()
def review(project_dir: Path = ProjectDir, review_mode: ReviewMode = ModeOption) -> None:
    """Open the review UI in your browser."""
    app_path = Path(__file__).parent / "review" / "app.py"
    bind_address = os.environ.get("STREAMLIT_SERVER_ADDRESS", "127.0.0.1")
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.address",
        bind_address,
        "--",
        "--project",
        str(project_dir.resolve()),
    ]
    if review_mode is not None:
        command += ["--review-mode", review_mode.value]
    try:
        raise typer.Exit(code=subprocess.call(command))
    except FileNotFoundError as exc:
        raise _fail(MVDError("Could not start Streamlit. Install the 'ui' extra.")) from exc


@app.command()
def assemble(
    project_dir: Path = ProjectDir,
    output: Path = typer.Option(None, help="Output file. Defaults to output/<title>.mp4."),
    force: bool = typer.Option(False, help="Build a draft even if shots are unfinished."),
    draft: bool = typer.Option(
        False, "--draft", help="Assemble at draft resolution and ignore upscaled clips."
    ),
    width: int | None = typer.Option(None, help="Override assembly width (even >= 64)."),
    height: int | None = typer.Option(None, help="Override assembly height (even >= 64)."),
    chapter: list[str] = ChaptersOption,
    rebuild_cache: bool = typer.Option(
        False,
        "--rebuild-cache",
        help="Re-render all mix chapters instead of reusing cached chapter MP4s.",
    ),
    review_mode: ReviewMode = ModeOption,
) -> None:
    """Join the approved clips (or cached mix chapters) and lay the original mix over them."""
    try:
        result = api.assemble(
            project_dir,
            output,
            force=force,
            review_mode=review_mode,
            draft=draft,
            width=width,
            height=height,
            chapters=chapter or None,
            rebuild_cache=rebuild_cache,
        )
    except MVDError as exc:
        raise _fail(exc) from exc
    typer.echo(f"Wrote {result}")
    sidecar = mix_assemble.chapters_sidecar_path(result)
    if sidecar.is_file():
        typer.echo(f"Wrote {sidecar}")


@app.command()
def schema(
    directory: Path = typer.Argument(Path("schemas"), help="Where to write schemas."),
) -> None:
    """Export JSON Schemas for manifest.json and analysis.json."""
    for path in api.export_schemas(directory):
        typer.echo(f"Wrote {path}")


@app.command()
def mcp(
    http: bool = typer.Option(
        False,
        "--http",
        help="Serve Streamable HTTP (/mcp) instead of stdio.",
    ),
    host: str = typer.Option(
        "127.0.0.1",
        "--host",
        help="Bind host for --http mode (defaults to loopback 127.0.0.1).",
    ),
    port: int = typer.Option(
        8765,
        "--port",
        min=1,
        max=65535,
        help="Bind port for --http mode.",
    ),
    token: str | None = typer.Option(
        None,
        "--token",
        envvar="MVD_MCP_TOKEN",
        help="Bearer token required for --http requests (required for non-loopback binds).",
    ),
    allow_unauthenticated: bool = typer.Option(
        False,
        "--allow-unauthenticated",
        help="Allow binding --http on a non-loopback address without a bearer token.",
    ),
    project: Path | None = typer.Option(
        None,
        "--project",
        "-p",
        help="Default project folder used when tool calls omit project_dir.",
    ),
    workspace: list[Path] | None = typer.Option(
        None,
        "--workspace",
        "-w",
        help="Restrict all project and file paths to these root folders (repeatable).",
    ),
) -> None:
    """Run the mvdirector MCP server over stdio (default) or Streamable HTTP (--http)."""
    from mvdirector.mcp import MCPServer, serve_http, serve_stdio

    allowed_roots: list[Path] | None = list(workspace) if workspace else None
    if http and allowed_roots is None:
        # Always contain HTTP callers to the configured project or current working directory.
        allowed_roots = [project.resolve() if project is not None else Path.cwd().resolve()]

    try:
        server = MCPServer(default_project=project, allowed_roots=allowed_roots)
        if http:
            serve_http(
                server,
                host=host,
                port=port,
                token=token,
                allow_unauthenticated=allow_unauthenticated,
            )
        else:
            serve_stdio(server)
    except MVDError as exc:
        raise _fail(exc) from exc


if __name__ == "__main__":
    app()
