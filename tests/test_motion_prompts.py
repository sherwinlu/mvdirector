# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Tests for motion prompt guards, tag-based reference notation, and black-screen MP4 audio."""

from __future__ import annotations

import copy
import json

import pytest

from mvdirector import media
from mvdirector.comfy.client import ComfyClient
from mvdirector.config import ProjectConfig, VideoConfig
from mvdirector.errors import ConfigError, MediaToolError
from mvdirector.generation import reference
from mvdirector.generation.runner import RunBudget, ShotRunner, load_templates
from mvdirector.models import (
    AssetKind,
    AssetStatus,
    AudioSlice,
    Character,
    Framing,
    Manifest,
    SetDesign,
    Shot,
    ShotType,
    VisualBible,
    WardrobeState,
)
from mvdirector.planning import prompts
from mvdirector.planning.planner import build_manifest
from mvdirector.qc.factory import QualityGates
from mvdirector.retry import RetryPolicy
from tests.conftest import WORKFLOW_GRAPH, FakeComfy, needs_ffmpeg
from tests.test_pipeline import ScriptedGate


def _sample_manifest() -> tuple[Manifest, Shot]:
    shot = Shot(
        id="s001",
        index=0,
        start=0.0,
        end=4.0,
        type=ShotType.LIPSYNC,
        framing=Framing.MEDIUM_CLOSE,
        setting="Stage",
        wardrobe="suit",
        lyrics="Neon on my skin",
        still_prompt="Singer on stage.",
        motion_prompt="0-2.0s: She steps forward. 2.0-4.0s: Slow push-in.",
        audio_slice=AudioSlice(
            path="assets/audio_slices/s001.wav", start=0.0, end=4.25, head_handle=0.0
        ),
        num_frames=105,
    )
    manifest = Manifest(
        title="Test",
        duration=4.0,
        fps=24,
        character=Character(
            appearance="Korean woman with silver ponytail.",
            wardrobe=[WardrobeState(id="suit", description="Black techwear jacket.")],
        ),
        visual_bible=VisualBible(
            style="35mm film, neon",
            settings=[SetDesign(name="Stage", description="a dark neon stage")],
        ),
        shots=[shot],
    )
    return manifest, shot


def test_motion_prompt_adds_single_take_framing_and_media_guards() -> None:
    manifest, shot = _sample_manifest()
    prompt = prompts.compose_motion_prompt(manifest, shot)
    assert "Maintain medium close-up framing throughout. Single continuous shot, no cuts." in prompt
    assert "No subtitles, no text." in prompt
    assert "No music." in prompt
    assert "Neon on my skin" not in prompt

    cfg = VideoConfig(
        motion_single_take=False,
        motion_no_subtitles=False,
        motion_no_music=False,
        motion_include_lyrics=True,
    )
    with_lyrics = prompts.compose_motion_prompt(manifest, shot, video_config=cfg)
    assert 'The song lyrics are: "Neon on my skin".' in with_lyrics
    assert "Single continuous shot" not in with_lyrics
    assert "No subtitles" not in with_lyrics
    assert "No music" not in with_lyrics

    shot.type = ShotType.PERFORMANCE
    non_vocal = prompts.compose_motion_prompt(manifest, shot, video_config=cfg)
    assert "Neon on my skin" not in non_vocal


def test_motion_prompt_tag_notation_mentions_image_and_audio_video() -> None:
    manifest, shot = _sample_manifest()
    prompt = prompts.compose_motion_prompt(
        manifest, shot, reference_notation="tags", has_audio_video=True
    )
    assert "Animate starting from @image1." in prompt
    assert "vocal in @video1." in prompt


def test_still_prompt_supports_tag_notation_and_multi_reference_prose() -> None:
    manifest, shot = _sample_manifest()
    tagged = prompts.compose_still_prompt(
        manifest,
        shot,
        plate_attached=True,
        reference_notation="tags",
        bound_references=["reference_image", "reference_image_2", "set_image"],
    )
    assert "@image1 shows the singer" in tagged
    assert "@image2 shows the secondary performer" in tagged
    assert "@image3 shows the location" in tagged

    prose = prompts.compose_still_prompt(
        manifest,
        shot,
        plate_attached=True,
        reference_notation="prose",
        bound_references=["reference_image", "reference_image_2", "set_image"],
    )
    assert "first reference image shows the singer" in prose
    assert "second reference image shows the secondary performer" in prose
    assert "third reference image shows the location" in prose


def test_audio_to_black_video_validates_inputs(tmp_path) -> None:
    missing = tmp_path / "missing.wav"
    with pytest.raises(MediaToolError, match="not found"):
        media.audio_to_black_video(missing, tmp_path / "out.mp4")
    real_wav = tmp_path / "real.wav"
    real_wav.write_bytes(b"00")
    with pytest.raises(MediaToolError, match="even integers"):
        media.audio_to_black_video(real_wav, tmp_path / "out.mp4", width=63, height=64)
    with pytest.raises(MediaToolError, match="FPS"):
        media.audio_to_black_video(real_wav, tmp_path / "out.mp4", fps=0)


def test_planner_system_prompt_documents_single_take_and_timestamped_beats() -> None:
    normalized = " ".join(prompts.SHOTS_SYSTEM.split())
    assert "single continuous take" in normalized
    assert "timestamped sub-beats" in normalized


def test_audio_video_slice_path_rejects_unsafe_shot_id(project) -> None:
    with pytest.raises(ConfigError, match="shot id"):
        project.audio_video_slice_path("../outside")


@needs_ffmpeg
def test_audio_to_black_video_creates_playable_mp4(project, tmp_path) -> None:
    wav = project.input_path(project.config.inputs.vocals, "vocals")
    short_wav = tmp_path / "short.wav"
    media.slice_audio(wav, short_wav, 4.0, 5.5)
    mp4 = tmp_path / "short.mp4"
    media.audio_to_black_video(short_wav, mp4, fps=24)
    info = media.probe(mp4)
    assert info.has_audio and info.width == 64 and info.height == 64
    assert 1.3 <= info.duration <= 1.7


@needs_ffmpeg
def test_runner_uploads_audio_video_and_uses_tag_prompts(
    project, analysis, tmp_path, sample_png, sample_mp4
) -> None:
    project.config.video.max_clip_seconds = 4.0
    project.config.video.motion_include_lyrics = True
    project.config.comfy.reference_notation = "tags"

    # Extend toy workflow graph with an audio_video slot on node 3 and bind it on lipsync.
    graph = copy.deepcopy(WORKFLOW_GRAPH)
    graph["3"]["inputs"]["audio_video"] = ""
    wf_dir = project.path("workflows")
    (wf_dir / "graph.api.json").write_text(json.dumps(graph))
    lipsync_path = wf_dir / "lipsync.bindings.json"
    bindings = json.loads(lipsync_path.read_text())
    bindings["inputs"]["audio_video"] = [{"node": "3", "field": "audio_video"}]
    lipsync_path.write_text(json.dumps(bindings))

    manifest = build_manifest(project.config, analysis, "lyrics", "style", llm=None)
    portrait = tmp_path / "portrait.png"
    portrait.write_bytes(sample_png)
    reference.lock_reference(project, manifest, portrait)
    shot = manifest.shot("s002")
    shot.lyrics = "first line"
    shot.still.status = AssetStatus.PASSED
    shot.still.attempts = []
    # Seed a passed still so video generation runs immediately.
    still_path = project.asset_path(shot, AssetKind.STILL, 1)
    still_path.parent.mkdir(parents=True, exist_ok=True)
    still_path.write_bytes(sample_png)
    from mvdirector.models import Attempt, RetryStrategy

    shot.still.attempts.append(
        Attempt(
            number=1,
            strategy=RetryStrategy.INITIAL,
            seed=1,
            prompt="still",
            path=project.rel(still_path),
        )
    )
    shot.still.selected_attempt = 1

    fake = FakeComfy(png=sample_png, mp4=sample_mp4)
    comfy = ComfyClient(project.config.comfy, transport=fake.transport(), sleep=lambda _: None)
    gates = QualityGates(image=ScriptedGate([True]), video=ScriptedGate([True]), face_analyzer=None)
    policy = RetryPolicy(project.config.retry, project.config.seed, llm=None)
    templates = load_templates(project, ["t2i", "still", "lipsync", "i2v"])
    runner = ShotRunner(project, manifest, comfy, gates, policy, templates, RunBudget(10))
    runner.run([AssetKind.VIDEO], shot_ids=["s002"])

    assert shot.video.status is AssetStatus.PASSED
    assert project.audio_video_slice_path("s002").is_file()
    assert "s002.mp4" in fake.upload_files
    assert fake.queued[0]["3"]["inputs"]["audio_video"].startswith("mvdirector/upload_")
    prompt = shot.video.attempts[0].prompt
    assert "@image1" in prompt and "@video1" in prompt
    assert 'The song lyrics are: "first line".' in prompt
    assert "No subtitles, no text." in prompt and "No music." in prompt


def test_seedance_bindings_template_validates() -> None:
    from importlib import resources

    from mvdirector.comfy.workflow import WorkflowBindings

    text = (
        resources.files("mvdirector") / "templates" / "workflows" / "seedance_lipsync.bindings.json"
    ).read_text(encoding="utf-8")
    parsed = WorkflowBindings.model_validate_json(text)
    assert "audio_video" in parsed.inputs and "audio_video" in parsed.required
    # Check against ProjectConfig defaults check as well
    assert ProjectConfig().comfy.reference_notation == "prose"


def test_r2v_audiolock_and_latent_refine_templates_validate() -> None:
    from importlib import resources

    from mvdirector.comfy.workflow import WorkflowBindings

    wf_dir = resources.files("mvdirector") / "templates" / "workflows"
    r2v = WorkflowBindings.model_validate_json(
        (wf_dir / "r2v_audiolock_lipsync.bindings.json").read_text(encoding="utf-8")
    )
    assert {"image", "audio", "reference_image", "reference_image_2", "set_image"} <= set(
        r2v.inputs
    )
    assert "reference_image" not in r2v.required and "set_image" not in r2v.required

    upscale = WorkflowBindings.model_validate_json(
        (wf_dir / "latent_refine_upscale.bindings.json").read_text(encoding="utf-8")
    )
    assert {"video", "width", "height", "positive_prompt", "negative_prompt", "seed"} <= set(
        upscale.inputs
    )


def test_motion_prompt_includes_r2v_bound_references_in_tags_and_prose() -> None:
    manifest, shot = _sample_manifest()
    tagged = prompts.compose_motion_prompt(
        manifest,
        shot,
        reference_notation="tags",
        bound_references=["reference_image", "reference_image_2", "set_image"],
    )
    assert "Animate starting from @image1." in tagged
    assert "@image2 shows the singer" in tagged
    assert "@image3 shows the secondary performer" in tagged
    assert "@image4 shows the location" in tagged

    prose = prompts.compose_motion_prompt(
        manifest,
        shot,
        reference_notation="prose",
        bound_references=["reference_image", "set_image"],
    )
    assert "first attached reference image shows the singer" in prose
    assert "second attached reference image shows the location" in prose


def test_bible_system_prompt_encourages_grounded_people_and_shootable_locations() -> None:
    normalized = " ".join(prompts.BIBLE_SYSTEM.split()).lower()
    assert "ordinary, relatable real-world people" in normalized
    assert "physically shootable real-world locations" in normalized


@needs_ffmpeg
def test_runner_binds_optional_r2v_video_references_and_prunes_broll(
    project, analysis, tmp_path, sample_png, sample_mp4
) -> None:
    from mvdirector.generation import plates
    from mvdirector.models import Attempt, RetryStrategy

    project.config.video.max_clip_seconds = 4.0
    project.config.comfy.reference_notation = "tags"

    # Build a video graph with optional LoadImage nodes for reference_image and set_image.
    graph = copy.deepcopy(WORKFLOW_GRAPH)
    graph["10"] = {"class_type": "LoadImage", "inputs": {"image": "unused_ref.png"}}
    graph["11"] = {"class_type": "LoadImage", "inputs": {"image": "unused_set.png"}}
    graph["2"]["inputs"]["ref_char"] = ["10", 0]
    graph["2"]["inputs"]["ref_set"] = ["11", 0]
    wf_dir = project.path("workflows")
    (wf_dir / "graph.api.json").write_text(json.dumps(graph))
    for name in ("lipsync", "i2v"):
        bpath = wf_dir / f"{name}.bindings.json"
        bdata = json.loads(bpath.read_text())
        bdata["inputs"]["reference_image"] = [{"node": "10", "field": "image"}]
        bdata["inputs"]["set_image"] = [{"node": "11", "field": "image"}]
        bpath.write_text(json.dumps(bdata))

    manifest = build_manifest(project.config, analysis, "lyrics", "style", llm=None)
    portrait = tmp_path / "portrait.png"
    portrait.write_bytes(sample_png)
    reference.lock_reference(project, manifest, portrait)

    # Lock a set plate for the singer shot's setting.
    singer_shot = next(s for s in manifest.shots if s.type is ShotType.LIPSYNC)
    if manifest.visual_bible.find_set(singer_shot.setting) is None:
        manifest.visual_bible.settings.append(
            SetDesign(name=singer_shot.setting, description="neon performance stage")
        )
    plate_img = tmp_path / "plate.png"
    plate_img.write_bytes(sample_png)
    plates.lock_plate(project, manifest, plate_img, singer_shot.setting)

    # Also test a b-roll shot with no singer (reference_image LoadImage must be pruned).
    broll_shot = next(s for s in manifest.shots if s.id != singer_shot.id)
    broll_shot.type = ShotType.BROLL
    broll_shot.character = None
    broll_shot.characters = []
    broll_shot.setting = singer_shot.setting

    for shot in (singer_shot, broll_shot):
        still_path = project.asset_path(shot, AssetKind.STILL, 1)
        still_path.parent.mkdir(parents=True, exist_ok=True)
        still_path.write_bytes(sample_png)
        shot.still.status = AssetStatus.PASSED
        shot.still.attempts = [
            Attempt(
                number=1,
                strategy=RetryStrategy.INITIAL,
                seed=1,
                prompt="still",
                path=project.rel(still_path),
            )
        ]
        shot.still.selected_attempt = 1

    fake = FakeComfy(png=sample_png, mp4=sample_mp4)
    comfy = ComfyClient(project.config.comfy, transport=fake.transport(), sleep=lambda _: None)
    gates = QualityGates(
        image=ScriptedGate([True, True]),
        video=ScriptedGate([True, True]),
        face_analyzer=None,
    )
    policy = RetryPolicy(project.config.retry, project.config.seed, llm=None)
    templates = load_templates(project, ["t2i", "still", "lipsync", "i2v"])
    runner = ShotRunner(project, manifest, comfy, gates, policy, templates, RunBudget(10))
    runner.run([AssetKind.VIDEO], shot_ids=[singer_shot.id, broll_shot.id])

    assert len(fake.queued) == 2
    ordered_ids = [s.id for s in manifest.shots if s.id in {singer_shot.id, broll_shot.id}]
    queued_by_shot = dict(zip(ordered_ids, fake.queued, strict=True))
    singer_graph = queued_by_shot[singer_shot.id]
    broll_graph = queued_by_shot[broll_shot.id]
    assert "10" in singer_graph and "11" in singer_graph
    assert singer_graph["10"]["inputs"]["image"].startswith("mvdirector/upload_")
    assert singer_graph["11"]["inputs"]["image"].startswith("mvdirector/upload_")
    assert "@image2 shows the singer" in singer_shot.video.attempts[0].prompt
    assert "@image3 shows the location" in singer_shot.video.attempts[0].prompt

    # B-roll has no singer -> reference_image loader node 10 is pruned, while set_image stays.
    assert "10" not in broll_graph and "ref_char" not in broll_graph["2"]["inputs"]
    assert "11" in broll_graph and broll_graph["11"]["inputs"]["image"].startswith(
        "mvdirector/upload_"
    )
    assert "@image2 shows the location" in broll_shot.video.attempts[0].prompt


def test_motion_prompt_hygiene_and_patina_cues_in_system_prompts() -> None:
    shots_norm = " ".join(prompts.SHOTS_SYSTEM.split()).lower()
    assert "never re-describe locked wardrobe" in shots_norm
    assert "permanent facial features" in shots_norm
    assert "static room/set decor" in shots_norm

    visual_shots_norm = " ".join(prompts.VISUAL_SHOTS_SYSTEM.split()).lower()
    assert "never re-describe static architecture" in visual_shots_norm

    rewrite_norm = " ".join(prompts.REWRITE_SYSTEM.split()).lower()
    assert "when rewriting a video motion_prompt" in rewrite_norm
    assert "strip any re-described locked wardrobe" in rewrite_norm

    for system in (prompts.BIBLE_SYSTEM, prompts.VISUAL_BIBLE_SYSTEM):
        norm = " ".join(system.split()).lower()
        assert "lived-in wear and patina" in norm
        assert "sleek, pristine, or futuristic" in norm


def test_plate_and_prop_prompts_apply_patina_by_default_and_suppress_for_sleek() -> None:
    from mvdirector.models import PropDesign, VisualBible

    worn_manifest = Manifest(
        title="Worn",
        duration=4.0,
        fps=24,
        visual_bible=VisualBible(
            style="35mm gritty neo-noir",
            palette="amber and slate",
            lighting="sodium streetlamp",
            settings=[SetDesign(name="Alley", description="brick alley with wet asphalt")],
            props=[PropDesign(id="watch", name="Pocket Watch", description="brass 5cm watch")],
        ),
    )
    for rendered in (
        prompts.plate_prompt(worn_manifest, "Alley"),
        prompts.plate_style_prompt(worn_manifest, "Alley"),
        prompts.plate_edit_prompt(worn_manifest, "Alley", "Old Alley"),
        prompts.prop_prompt(worn_manifest, "watch"),
    ):
        assert prompts._SUBTLE_PATINA_CUE in rendered
    assert "glossy plastic sheen" in prompts.PROP_NEGATIVE

    sleek_manifest = Manifest(
        title="Sleek",
        duration=4.0,
        fps=24,
        visual_bible=VisualBible(
            style="sleek futuristic sci-fi chrome minimalism",
            palette="ice blue and silver",
            lighting="clean studio edge light",
            settings=[SetDesign(name="Lab", description="glass corridor")],
            props=[PropDesign(id="visor", name="Visor", description="mirror visor 18cm wide")],
        ),
    )
    for rendered in (
        prompts.plate_prompt(sleek_manifest, "Lab"),
        prompts.plate_style_prompt(sleek_manifest, "Lab"),
        prompts.plate_edit_prompt(sleek_manifest, "Lab", "Old Lab"),
        prompts.prop_prompt(sleek_manifest, "visor"),
    ):
        assert prompts._SUBTLE_PATINA_CUE not in rendered
        assert "  " not in rendered
