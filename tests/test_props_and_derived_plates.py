# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Tests for derived set plates and recurring hero prop sheets."""

from __future__ import annotations

import logging

import pytest
from PIL import Image
from typer.testing import CliRunner

from mvdirector.cli import app
from mvdirector.comfy.client import ComfyClient
from mvdirector.config import ProjectConfig
from mvdirector.errors import ConfigError
from mvdirector.generation import plates, props, reference
from mvdirector.generation.preflight import check_assets
from mvdirector.generation.runner import RunBudget, ShotRunner, load_templates
from mvdirector.models import AssetKind, PropDesign, SetDesign, ShotType
from mvdirector.planning import prompts
from mvdirector.planning.planner import (
    BibleReply,
    PropReply,
    SetReply,
    ShotBatchReply,
    ShotCreative,
    build_manifest,
)
from mvdirector.qc import vision
from mvdirector.qc.factory import QualityGates
from mvdirector.retry import RetryPolicy
from tests.conftest import FakeComfy, write_workflows
from tests.test_pipeline import ScriptedGate


def test_derived_plate_edits_locked_parent_plate(project, analysis, tmp_path, sample_png) -> None:
    manifest = build_manifest(project.config, analysis, "lyrics", "style", llm=None)
    manifest.visual_bible.settings = [
        SetDesign(name="Neon City", description="glowing pagoda skyline and round stage"),
        SetDesign(
            name="Empty Grid",
            description="remove the buildings and stage leaving only grid lines",
            parent_set="Neon City",
        ),
    ]
    templates = load_templates(project, ["t2i", "still"])
    comfy = ComfyClient(
        project.config.comfy, transport=FakeComfy(png=sample_png).transport(), sleep=lambda _: None
    )
    with pytest.raises(ConfigError, match="Lock a plate for 'Neon City'"):
        plates.generate_candidates(project, manifest, comfy, templates, "Empty Grid", count=1)

    parent_img = tmp_path / "neon_city.png"
    Image.new("RGB", (64, 36), "navy").save(parent_img)
    plates.lock_plate(project, manifest, parent_img, "Neon City")

    fake = FakeComfy(png=sample_png)
    comfy = ComfyClient(project.config.comfy, transport=fake.transport(), sleep=lambda _: None)
    derived = plates.generate_candidates(project, manifest, comfy, templates, "Empty Grid", count=1)
    assert len(derived["Empty Grid"]) == 1
    assert fake.upload_files == ["neon_city.png"]
    prompt = fake.queued[0]["1"]["inputs"]["text"]
    assert "Edit the existing empty location photograph (Neon City)" in prompt
    assert "remove the buildings and stage" in prompt

    # Missing parent plate file on disk raises ConfigError rather than FileNotFoundError.
    project.path(manifest.visual_bible.find_set("Neon City").plate).unlink()
    with pytest.raises(ConfigError, match="missing"):
        plates.generate_candidates(project, manifest, comfy, templates, "Empty Grid", count=1)

    with pytest.raises(ConfigError, match="cannot derive from itself"):
        plates.generate_candidates(
            project, manifest, comfy, templates, "Empty Grid", count=1, from_set="Empty Grid"
        )


def test_planner_persists_props_and_parent_set_and_warns_unknown_prop(analysis, caplog) -> None:
    class PropDirector:
        def complete_json(self, *, system, content, schema):
            if schema is BibleReply:
                return BibleReply(
                    appearance="Kira, silver ponytail.",
                    style="35mm neon",
                    palette="cyan and black",
                    lighting="rim light",
                    negative_prompt="blurry",
                    settings=[
                        SetReply(name="Neon City", description="cyber skyline"),
                        SetReply(
                            name="Empty Grid",
                            description="flat grid only",
                            parent_set="Neon City",
                        ),
                    ],
                    props=[
                        PropReply(
                            id="light_disc",
                            name="Light Disc",
                            sections=["Chorus"],
                            description="30 cm glossy black hollow ring with cyan edges, no hub",
                        )
                    ],
                )
            return ShotBatchReply(
                shots=[
                    ShotCreative(
                        id="s001",
                        type=ShotType.PERFORMANCE,
                        framing="medium",
                        setting="Neon City",
                        props=["light_disc", "nonexistent_prop"],
                        still_prompt="Kira holds the disc.",
                        motion_prompt="She spins the disc.",
                    )
                ]
            )

    cfg = ProjectConfig()
    cfg.video.max_clip_seconds = 8.0
    with caplog.at_level(logging.WARNING, logger="mvdirector.planning.planner"):
        manifest = build_manifest(cfg, analysis, "lyrics", "style", PropDirector())

    assert manifest.visual_bible.find_set("Empty Grid").parent_set == "Neon City"
    assert manifest.visual_bible.find_prop("Light Disc").id == "light_disc"
    assert manifest.shot("s001").props == ["light_disc"]
    assert "nonexistent_prop" in caplog.text


def test_props_preflight_runner_vision_and_cli(project, analysis, tmp_path, sample_png) -> None:
    project.config.video.max_clip_seconds = 4.0
    write_workflows(
        project.path("workflows"),
        {"t2i": ".png", "still": ".png", "lipsync": ".mp4", "i2v": ".mp4"},
        still_extra={"prop_image": "profile", "prop_image_2": "three_quarter"},
    )
    manifest = build_manifest(project.config, analysis, "lyrics", "style", llm=None)
    manifest.character.appearance = "Kira, silver ponytail."
    manifest.visual_bible.props = [
        PropDesign(
            id="light_disc",
            name="Light Disc",
            description="30 cm hollow cyan glowing ring, no center hub",
        ),
        PropDesign(id="visor", name="Visor", description="glossy black visor"),
    ]
    shot = manifest.shot("s002")
    shot.props = ["light_disc", "visor"]

    sheet = tmp_path / "sheet.png"
    Image.new("RGB", (2048, 896), "grey").save(sheet)
    reference.lock_reference(project, manifest, sheet)

    templates = load_templates(project, ["t2i", "still"])
    preflight = check_assets(project, manifest, templates, shot_ids=["s002"])
    assert any("Light Disc" in p and "mvd props generate" in p for p in preflight.problems)

    fake_prop = FakeComfy(png=sample_png)
    comfy = ComfyClient(project.config.comfy, transport=fake_prop.transport(), sleep=lambda _: None)
    prop_candidates = props.generate_candidates(
        project, manifest, comfy, templates["t2i"], "light_disc", count=1
    )
    assert prop_candidates["Light Disc"][0].name == "light_disc_candidate_01.png"
    assert "hollow cyan glowing ring" in fake_prop.queued[0]["1"]["inputs"]["text"]

    project.save_manifest(manifest)
    cli = CliRunner()
    res_lock = cli.invoke(
        app,
        [
            "props",
            "lock",
            str(prop_candidates["Light Disc"][0]),
            "--prop",
            "light_disc",
            "-p",
            str(project.root),
        ],
    )
    assert res_lock.exit_code == 0 and "Locked prop sheet" in res_lock.output
    manifest = project.load_manifest()
    shot = manifest.shot("s002")
    props.lock_prop(project, manifest, prop_candidates["Light Disc"][0], "visor")
    assert check_assets(project, manifest, templates, shot_ids=["s002"]).problems == []

    # Assets CLI shows Prop sheets section
    project.save_manifest(manifest)
    assets_out = cli.invoke(app, ["assets", "-p", str(project.root)]).output
    assert "Prop sheets" in assets_out and "Light Disc" in assets_out

    # Runner uploads prop_image + prop_image_2 and passes prop references to VisionJudge
    fake_still = FakeComfy(png=sample_png)
    gate = ScriptedGate([True])
    runner = ShotRunner(
        project,
        manifest,
        ComfyClient(project.config.comfy, transport=fake_still.transport(), sleep=lambda _: None),
        QualityGates(image=gate, video=ScriptedGate([True]), face_analyzer=None),
        RetryPolicy(project.config.retry, project.config.seed, llm=None),
        load_templates(project, ["t2i", "still", "lipsync", "i2v"]),
        RunBudget(10),
    )
    runner.run([AssetKind.STILL], shot_ids=["s002"])
    assert "light_disc.png" in fake_still.upload_files
    assert "visor.png" in fake_still.upload_files
    assert "Match the exact design, scale and details of the prop reference" in (
        shot.still.attempts[0].prompt
    )
    assert ("Prop: Light Disc", project.path("assets/props/light_disc.png")) in gate.kwargs[
        "extra_references"
    ]

    spec = vision.VisionJudge.specification(shot, manifest)
    assert "Hero prop (Light Disc): 30 cm hollow cyan glowing ring" in spec

    # When prop_image is unbound, compose_still_prompt keeps text description and preflight warns.
    write_workflows(
        project.path("workflows"),
        {"t2i": ".png", "still": ".png", "lipsync": ".mp4", "i2v": ".mp4"},
    )
    plain_templates = load_templates(project, ["t2i", "still"])
    warn_result = check_assets(project, manifest, plain_templates)
    assert any("no 'prop_image' binding" in w for w in warn_result.warnings)
    text_prompt = prompts.compose_still_prompt(manifest, shot, props_attached=False)
    assert "Hero prop (Light Disc):" in text_prompt

    # Duplicate prop references (id + display name) resolve once everywhere.
    shot.props = ["light_disc", "Light Disc", "visor"]
    assert [p.id for p in manifest.visual_bible.resolve_props(shot.props)] == [
        "light_disc",
        "visor",
    ]
    assert runner._prop_paths(shot) == [
        ("Light Disc", project.path("assets/props/light_disc.png")),
        ("Visor", project.path("assets/props/visor.png")),
    ]
