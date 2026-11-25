# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Tests for global exclusions / do_not_show propagation and downstream enforcement."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from mvdirector import api as api_mod
from mvdirector import cli as cli_mod
from mvdirector import tuning as tuning_mod
from mvdirector.comfy.client import ComfyClient
from mvdirector.config import ProjectConfig, ProjectMode
from mvdirector.generation import plates, props
from mvdirector.generation.runner import RunBudget, ShotRunner, load_templates
from mvdirector.models import (
    Analysis,
    AnalyzedChapter,
    AssetKind,
    AssetState,
    AssetStatus,
    Attempt,
    AudioSlice,
    Framing,
    Manifest,
    PropDesign,
    QCReport,
    RetryStrategy,
    SetDesign,
    Shot,
    ShotType,
    VisualBible,
)
from mvdirector.planning import prompts
from mvdirector.planning.brief import BriefReply, generate_brief, write_brief
from mvdirector.planning.mix_planner import (
    _ChapterThemeReply,
    _MixThemesBatchReply,
    build_mix_manifest,
)
from mvdirector.planning.planner import (
    BibleReply,
    SetReply,
    ShotBatchReply,
    ShotCreative,
    build_manifest,
)
from mvdirector.qc.factory import QualityGates
from mvdirector.qc.vision import VisionJudge
from mvdirector.retry import RetryPolicy
from tests.conftest import FakeComfy, needs_ffmpeg
from tests.test_pipeline import ScriptedGate


def test_parse_exclusions_and_tuning_section_stop() -> None:
    style_text = (
        "Music style: upbeat synthwave, female vocal\n\n"
        "Do not show: male vocals, saxophone, red balloons\n"
        "- ghost's face\n\n"
        "Visual direction: neon rooftop.\n"
    )
    assert prompts.parse_exclusions_from_style(style_text) == [
        "male vocals",
        "saxophone",
        "red balloons",
        "ghost's face",
    ]
    musical = tuning_mod._musical_style_portion(style_text)
    assert "female vocal" in musical
    assert "male vocals" not in musical
    assert "saxophone" not in musical


def test_parse_exclusions_markdown_bold_colon_headers_never_leak_asterisks() -> None:
    inline_bold = (
        "Music style: dark indie\n\n**Never show:** guitars, neon\n\nVisual direction: street\n"
    )
    assert prompts.parse_exclusions_from_style(inline_bold) == ["guitars", "neon"]

    bullet_bold = (
        "Music style: dark indie, female vocal\n\n"
        "**Never show:**\n"
        "- male singer\n"
        "- saxophone\n"
        "**Visual direction:** misty street\n"
    )
    assert prompts.parse_exclusions_from_style(bullet_bold) == ["male singer", "saxophone"]

    musical = tuning_mod._musical_style_portion(bullet_bold)
    assert "dark indie" in musical
    assert "male singer" not in musical
    assert "saxophone" not in musical

    assert prompts.parse_exclusions_from_style("**Never show**: guitars, neon") == [
        "guitars",
        "neon",
    ]
    assert prompts.parse_exclusions_from_style("*Never show:* guitars, neon") == [
        "guitars",
        "neon",
    ]

    bible = VisualBible(style="35mm")
    prompts.sync_bible_exclusions(
        bible, extra_exclusions=prompts.parse_exclusions_from_style(inline_bold)
    )
    assert bible.exclusions == ["guitars", "neon"]
    assert "**" not in bible.negative_prompt
    assert all("**" not in rule for rule in bible.rules)
    assert "guitars" in bible.negative_prompt
    assert "neon" in bible.negative_prompt


def test_sync_bible_exclusions_populates_rules_and_negative_idempotently() -> None:
    bible = VisualBible(
        style="35mm",
        negative_prompt="blurry, watermark, Red Balloons",
        rules=["Keep camera at eye height."],
        exclusions=["never show the ghost's face", "red balloons", "do not show clocks"],
    )
    prompts.sync_bible_exclusions(bible)
    assert bible.exclusions == [
        "never show the ghost's face",
        "red balloons",
        "do not show clocks",
    ]
    assert bible.negative_prompt == "blurry, watermark, Red Balloons, the ghost's face, clocks"
    assert any("Never show the ghost's face in any shot" in r for r in bible.rules)
    assert any("Never show red balloons in any shot" in r for r in bible.rules)
    assert any("Never show clocks in any shot" in r for r in bible.rules)
    rules_count = len(bible.rules)
    prompts.sync_bible_exclusions(bible, extra_exclusions=["RED BALLOONS"])
    assert len(bible.rules) == rules_count
    assert bible.negative_prompt == "blurry, watermark, Red Balloons, the ghost's face, clocks"


def test_brief_flows_exclusions_to_llm_and_style_md(tmp_path: Path, monkeypatch) -> None:
    class _FakeBriefLLM:
        def __init__(self) -> None:
            self.seen_prompt = ""

        def complete_json(self, *, system, content, schema):
            assert schema is BriefReply
            self.seen_prompt = content[0]["text"]
            return BriefReply(
                lyrics="[Verse]\nShadows fall",
                suno_style="indie rock",
                visual_direction="misty hallway",
            )

    llm = _FakeBriefLLM()
    reply = generate_brief(
        "Haunted hotel",
        llm=llm,
        do_not_show=["never show the ghost's face", "skulls"],
    )
    assert "GLOBAL EXCLUSIONS (NEVER SHOW):" in llm.seen_prompt
    assert "never show the ghost's face" in llm.seen_prompt
    assert reply.exclusions == ["never show the ghost's face", "skulls"]

    proj_dir = tmp_path / "song"
    assert CliRunner().invoke(cli_mod.app, ["init", str(proj_dir)]).exit_code == 0
    monkeypatch.setattr(api_mod, "build_llm", lambda _cfg: _FakeBriefLLM())
    res = CliRunner().invoke(
        cli_mod.app,
        [
            "brief",
            "--idea",
            "Haunted hotel",
            "--exclude",
            "the ghost's face",
            "--exclude",
            "mirrors",
            "-p",
            str(proj_dir),
        ],
    )
    assert res.exit_code == 0
    style_md = (proj_dir / "inputs" / "style.md").read_text(encoding="utf-8")
    assert "Exclusions: the ghost's face, mirrors" in style_md


def test_planner_propagates_exclusions_for_both_llm_and_fallback(
    project, analysis: Analysis
) -> None:
    config = ProjectConfig(inputs={"do_not_show": ["never show the ghost's face"]})
    config.video.max_clip_seconds = 4.0

    # Fallback path (llm=None)
    fallback = build_manifest(
        config,
        analysis,
        "lyrics",
        "indie noir\n\nExclusions: red balloons",
        llm=None,
        exclusions=["skulls"],
    )
    assert fallback.visual_bible.exclusions == [
        "never show the ghost's face",
        "red balloons",
        "skulls",
    ]
    assert "the ghost's face" in fallback.visual_bible.negative_prompt
    assert "red balloons" in fallback.visual_bible.negative_prompt
    assert "skulls" in fallback.visual_bible.negative_prompt
    assert any("Never show the ghost's face" in r for r in fallback.visual_bible.rules)

    # LLM path
    class _FakeDirectorLLM:
        def __init__(self) -> None:
            self.bible_prompt = ""
            self.shots_prompt = ""

        def complete_json(self, *, system, content, schema):
            text = content[0]["text"]
            if schema is BibleReply:
                self.bible_prompt = text
                return BibleReply(
                    appearance="Singer with dark coat",
                    style="35mm film",
                    palette="teal and amber",
                    lighting="soft sodium",
                    negative_prompt="blurry, red balloons",
                    settings=[SetReply(name="Hallway", description="dim hotel hallway")],
                    rules=["Hold one set per verse."],
                )
            assert schema is ShotBatchReply
            self.shots_prompt = text
            return ShotBatchReply(
                shots=[
                    ShotCreative(
                        id=s.id,
                        type=s.type,
                        framing=Framing.MEDIUM_CLOSE,
                        setting="Hallway",
                        still_prompt="Singer stands in hallway.",
                        motion_prompt="Slow push-in.",
                    )
                    for s in fallback.shots
                ]
            )

    llm = _FakeDirectorLLM()
    planned = build_manifest(
        config,
        analysis,
        "lyrics",
        "indie noir\n\nExclusions: red balloons",
        llm=llm,
        exclusions=["skulls"],
    )
    assert "GLOBAL EXCLUSIONS (NEVER SHOW IN ANY SHOT):" in llm.bible_prompt
    assert "never show the ghost's face" in llm.bible_prompt
    assert "Never show the ghost's face" in llm.shots_prompt
    assert planned.visual_bible.negative_prompt == "blurry, red balloons, the ghost's face, skulls"
    assert any("Never show red balloons" in r for r in planned.visual_bible.rules)


def test_mix_planner_propagates_exclusions_for_both_llm_and_fallback() -> None:
    cfg = ProjectConfig(
        mode=ProjectMode.MIX,
        inputs={"exclusions": ["never show neon signs"]},
    )
    cfg.mix.clips_per_segment = 1
    analysis = Analysis(
        duration=12.0,
        tempo_bpm=120.0,
        beats=[],
        bars=[],
        bar_intensity=[],
        vocal_windows=[],
        chapters=[
            AnalyzedChapter(
                id="ch01",
                index=0,
                start=0.0,
                end=12.0,
                title="Opening",
                bpm=120.0,
            )
        ],
    )
    fb = build_mix_manifest(cfg, analysis, "ambient\n\nNever show: cars", llm=None)
    assert fb.visual_bible.exclusions == ["never show neon signs", "cars"]
    assert "neon signs" in fb.visual_bible.negative_prompt
    assert any("Never show cars" in r for r in fb.visual_bible.rules)

    class _FakeMixLLM:
        def __init__(self) -> None:
            self.prompt = ""

        def complete_json(self, *, system, content, schema):
            self.prompt = content[0]["text"]
            return _MixThemesBatchReply(
                style="atmospheric photography",
                negative_prompt="people, cars",
                chapters=[
                    _ChapterThemeReply(
                        id="ch01",
                        environment="empty stone hall",
                        palette="amber and slate",
                        lighting="soft rim light",
                        motion_style="drifting fog",
                        clip_prompts=["wide empty stone hall"],
                    )
                ],
            )

    llm = _FakeMixLLM()
    planned = build_mix_manifest(cfg, analysis, "ambient\n\nNever show: cars", llm=llm)
    assert "GLOBAL EXCLUSIONS (NEVER SHOW):" in llm.prompt
    assert planned.visual_bible.negative_prompt == "people, cars, neon signs"


def test_api_plan_accepts_do_not_show_and_write_brief(project, analysis) -> None:
    project.save_analysis(analysis)
    (project.root / "inputs" / "style.md").write_text("dark indie\n", encoding="utf-8")
    _, manifest = api_mod.plan(project, overwrite=True, do_not_show=["never show clocks"])
    assert manifest.visual_bible.exclusions == ["never show clocks"]
    assert "clocks" in manifest.visual_bible.negative_prompt
    assert any("Never show clocks" in r for r in manifest.visual_bible.rules)

    # write_brief with no exclusions preserves original clean format
    lyrics_p, style_p = write_brief(
        project,
        BriefReply(lyrics="[Verse]\nhi", suno_style="pop", visual_direction="clean room"),
        overwrite=True,
    )
    assert lyrics_p.is_file()
    assert "Exclusions:" not in style_p.read_text(encoding="utf-8")


@needs_ffmpeg
def test_downstream_vision_runner_plates_props_enforce_exclusions(
    project, analysis, sample_png, sample_mp4
) -> None:
    manifest = Manifest(
        title="Excl",
        duration=4.0,
        fps=24,
        visual_bible=VisualBible(
            style="35mm film",
            negative_prompt="blurry, the ghost's face",
            exclusions=["never show the ghost's face", "red balloons"],
            settings=[SetDesign(name="Hall", description="wooden hall")],
            props=[PropDesign(id="lantern", name="Lantern", description="brass lantern")],
            rules=["Never show the ghost's face in any shot."],
        ),
        shots=[
            Shot(
                id="s001",
                index=0,
                start=0.0,
                end=4.0,
                type=ShotType.BROLL,
                framing=Framing.WIDE,
                setting="Hall",
                still_prompt="Empty wooden hall, no hats.",
                motion_prompt="Slow drift.",
                audio_slice=AudioSlice(path="a.wav", start=0.0, end=4.0, head_handle=0.0),
                num_frames=97,
                still=AssetState(attempts_allowed=2),
                video=AssetState(attempts_allowed=2),
            )
        ],
    )
    spec = VisionJudge.specification(manifest.shots[0], manifest)
    assert "Must NOT appear in frame (hard fail if visible):" in spec
    assert "the ghost's face" in spec
    assert "red balloons" in spec
    assert "hats" in spec

    # Seed passed still so ShotRunner exercises both _still_negative_prompt and _generate_video.
    fake = FakeComfy(png=sample_png, mp4=sample_mp4)
    comfy = ComfyClient(project.config.comfy, transport=fake.transport(), sleep=lambda _: None)
    gates = QualityGates(image=ScriptedGate([True]), video=ScriptedGate([True]), face_analyzer=None)
    policy = RetryPolicy(project.config.retry, project.config.seed, llm=None)
    templates = load_templates(project, ["t2i", "still", "lipsync", "i2v"])
    runner = ShotRunner(project, manifest, comfy, gates, policy, templates, RunBudget(10))

    still_neg = runner._still_negative_prompt(manifest.shots[0])
    assert "the ghost's face" in still_neg
    assert "red balloons" in still_neg
    assert still_neg.lower().count("ghost's face") == 1

    plates.generate_candidates(project, manifest, comfy, templates, "Hall", count=1)
    props.generate_candidates(project, manifest, comfy, templates["t2i"], "lantern", count=1)
    assert "red balloons" in fake.queued[0]["1"]["inputs"]["negative"]
    assert "red balloons" in fake.queued[1]["1"]["inputs"]["negative"]

    # Also exercise _generate_video negative prompt.
    shot = manifest.shots[0]
    still_path = project.asset_path(shot, AssetKind.STILL, 1)
    still_path.parent.mkdir(parents=True, exist_ok=True)
    still_path.write_bytes(sample_png)
    shot.still.status = AssetStatus.PASSED
    shot.still.selected_attempt = 1
    shot.still.attempts = [
        Attempt(
            number=1,
            strategy=RetryStrategy.INITIAL,
            seed=1,
            prompt="p",
            path=project.rel(still_path),
            qc=QCReport(passed=True, score=0.9),
        )
    ]
    runner.run([AssetKind.VIDEO], shot_ids=["s001"])
    video_neg = fake.queued[2]["1"]["inputs"]["negative"]
    assert "the ghost's face" in video_neg and "red balloons" in video_neg
