# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Tests for F13: optional rough sketch / framing guide attachment and forwarding."""

from __future__ import annotations

import json
from typing import Any

import pytest
from PIL import Image
from typer.testing import CliRunner

from mvdirector import api
from mvdirector.cli import app
from mvdirector.comfy.client import ComfyClient
from mvdirector.comfy.workflow import WorkflowTemplate
from mvdirector.errors import ConfigError, ImagesNotSupportedError
from mvdirector.generation import preflight
from mvdirector.generation.runner import RunBudget, ShotRunner, load_templates
from mvdirector.models import AssetKind, ReferenceSet, RetryStrategy, ShotType
from mvdirector.planning import prompts
from mvdirector.planning.planner import build_manifest
from mvdirector.planning.previz import (
    PrevizPlan,
    describe_sketch_blocking,
    plan_previz,
    refine_previz,
)
from mvdirector.retry import RetryPolicy
from tests.conftest import FakeComfy, offline_gates, write_workflows


def _setup_singer_ref(project, manifest) -> None:
    ref_rel = "assets/reference/base_front.png"
    project.path(ref_rel).parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), "white").save(project.path(ref_rel))
    manifest.character.references["base"] = ReferenceSet(sheet=ref_rel, panels={"front": ref_rel})


def test_attach_and_clear_shot_sketch_and_workflow_pruning(project, analysis, tmp_path) -> None:
    manifest = build_manifest(project.config, analysis, "lyrics", "style", llm=None)
    project.save_manifest(manifest)
    shot_id = manifest.shots[0].id

    # Invalid file raises ConfigError before mutating manifest.
    bad = tmp_path / "bad.txt"
    bad.write_text("not an image", encoding="utf-8")
    with pytest.raises(ConfigError, match="could not be opened"):
        api.attach_shot_sketch(project.root, shot_id, bad)

    # Path escape on shot_id fails safely.
    with pytest.raises(ConfigError, match="shot id"):
        project.sketch_path("../../escape")

    sketch_src = tmp_path / "rough_blocking.png"
    Image.new("RGB", (320, 180), "white").save(sketch_src)
    saved = api.attach_shot_sketch(project.root, shot_id, sketch_src)
    assert saved is not None and saved.is_file()
    reloaded = project.load_manifest()
    assert reloaded.shot(shot_id).sketch_image == f"assets/sketches/{shot_id}.png"

    # WorkflowTemplate prunes unbound sketch_image LoadImage node cleanly.
    graph = {
        "1": {"class_type": "Prompt", "inputs": {"text": "", "negative": ""}},
        "2": {"class_type": "Sampler", "inputs": {"seed": 0}},
        "3": {"class_type": "LoadImage", "inputs": {"image": ""}},
        "4": {"class_type": "ControlNet", "inputs": {"image": ["3", 0]}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["4", 0]}},
    }
    (project.path("workflows/sketch_graph.api.json")).write_text(json.dumps(graph))
    bindings = {
        "workflow": "sketch_graph.api.json",
        "inputs": {
            "positive_prompt": [{"node": "1", "field": "text"}],
            "seed": [{"node": "2", "field": "seed"}],
            "sketch_image": [{"node": "3", "field": "image"}],
        },
        "required": ["positive_prompt", "seed"],
        "output": {"node": "9", "extensions": [".png"]},
    }
    tpl_path = project.path("workflows/sketch.bindings.json")
    tpl_path.write_text(json.dumps(bindings))
    tpl = WorkflowTemplate.load("still", tpl_path, root=project.root)

    rendered_with = tpl.render({"positive_prompt": "hi", "seed": 1, "sketch_image": "s.png"})
    assert "3" in rendered_with and rendered_with["3"]["inputs"]["image"] == "s.png"
    rendered_without = tpl.render({"positive_prompt": "hi", "seed": 1})
    assert "3" not in rendered_without

    # CLI attach & clear round-trip.
    cli_runner = CliRunner()
    assert (
        cli_runner.invoke(
            app, ["sketch", "clear", "--shot", shot_id, "-p", str(project.root)]
        ).exit_code
        == 0
    )
    assert project.load_manifest().shot(shot_id).sketch_image is None
    assert (
        cli_runner.invoke(
            app, ["sketch", "attach", str(sketch_src), "--shot", shot_id, "-p", str(project.root)]
        ).exit_code
        == 0
    )
    assert project.load_manifest().shot(shot_id).sketch_image == f"assets/sketches/{shot_id}.png"


def test_sketch_runner_forwarding_tiers_and_repair_skip(
    project, analysis, tmp_path, sample_png
) -> None:
    manifest = build_manifest(project.config, analysis, "lyrics", "style", llm=None)
    _setup_singer_ref(project, manifest)
    shot = manifest.shots[0]
    shot.type = ShotType.PERFORMANCE
    shot.sketch_guidance = "Subject framed in lower-left third, doorway on right."
    project.save_manifest(manifest)

    sketch_src = tmp_path / "sketch.png"
    Image.new("RGB", (128, 72), "grey").save(sketch_src)
    api.attach_shot_sketch(project.root, shot.id, sketch_src)
    manifest = project.load_manifest()
    shot = manifest.shot(shot.id)

    # Tier 1: dedicated `sketch_image` slot on `still`.
    write_workflows(
        project.path("workflows"),
        {"t2i": ".png", "still": ".png", "lipsync": ".mp4", "i2v": ".mp4"},
        still_extra={"sketch_image": "set"},
    )
    fake_t1 = FakeComfy(png=sample_png)
    comfy_t1 = ComfyClient(
        project.config.comfy, transport=fake_t1.transport(), sleep=lambda _: None
    )
    templates_t1 = load_templates(project, ["still", "t2i", "lipsync", "i2v"])
    runner_t1 = ShotRunner(
        project,
        manifest,
        comfy_t1,
        offline_gates(project.config),
        RetryPolicy(project.config.retry, project.config.seed, llm=None),
        templates_t1,
        RunBudget(5),
    )
    runner_t1.run([AssetKind.STILL], [shot.id])
    assert f"{shot.id}.png" in fake_t1.upload_files
    prompt_t1 = fake_t1.queued[0]["1"]["inputs"]["text"]
    assert "Framing and blocking guide (from rough sketch):" in prompt_t1
    assert "rough composition sketch" in prompt_t1

    # REPAIR_STILL must NOT attach sketch_image or sketch_guidance (edits the rendered frame).
    shot.still.reset(1)
    shot.still.pending_strategy = RetryStrategy.REPAIR_STILL
    shot.still.repair_instruction = "Fix left hand."
    shot.still.repair_source_attempt = 1
    project.save_manifest(manifest)
    fake_repair = FakeComfy(png=sample_png)
    comfy_repair = ComfyClient(
        project.config.comfy, transport=fake_repair.transport(), sleep=lambda _: None
    )
    runner_repair = ShotRunner(
        project,
        manifest,
        comfy_repair,
        offline_gates(project.config),
        RetryPolicy(project.config.retry, project.config.seed, llm=None),
        templates_t1,
        RunBudget(5),
    )
    runner_repair.run([AssetKind.STILL], [shot.id])
    assert f"{shot.id}.png" not in fake_repair.upload_files
    assert "Fix left hand." in fake_repair.queued[0]["1"]["inputs"]["text"]

    # Tier 3: unused reference_image_2 slot when no dedicated sketch_image slot exists.
    write_workflows(
        project.path("workflows"),
        {"t2i": ".png", "still": ".png", "lipsync": ".mp4", "i2v": ".mp4"},
        still_extra={"reference_image_2": "profile"},
    )
    manifest.shots[1].type = ShotType.PERFORMANCE
    api.attach_shot_sketch(project.root, manifest.shots[1].id, sketch_src)
    manifest = project.load_manifest()
    templates_t3 = load_templates(project, ["still", "t2i", "lipsync", "i2v"])
    fake_t3 = FakeComfy(png=sample_png)
    runner_t3 = ShotRunner(
        project,
        manifest,
        ComfyClient(project.config.comfy, transport=fake_t3.transport(), sleep=lambda _: None),
        offline_gates(project.config),
        RetryPolicy(project.config.retry, project.config.seed, llm=None),
        templates_t3,
        RunBudget(5),
    )
    runner_t3.run([AssetKind.STILL], [manifest.shots[1].id])
    assert f"{manifest.shots[1].id}.png" in fake_t3.upload_files

    # Preflight catches missing sketch file on disk.
    project.path(manifest.shots[1].sketch_image).unlink()
    pf = preflight.check_assets(project, manifest, templates_t3, [manifest.shots[1].id])
    assert any("Sketch image" in p for p in pf.problems)


def test_sketch_previz_multimodal_and_fallback(project, analysis, tmp_path) -> None:
    manifest = build_manifest(project.config, analysis, "lyrics", "style", llm=None)
    shot = manifest.shots[0]
    sketch = tmp_path / "sketch.png"
    Image.new("RGB", (64, 36), "white").save(sketch)

    class _FakeLLM:
        def __init__(self, *, reject_images: bool = False) -> None:
            self.calls: list[list[dict[str, Any]]] = []
            self.reject_images = reject_images

        def complete_json(self, *, system: str, content: list[dict[str, Any]], schema: Any) -> Any:
            self.calls.append(content)
            if self.reject_images and any(b.get("type") == "image" for b in content):
                raise ImagesNotSupportedError("no vision")
            if schema.__name__ == "SketchBlockingSummary":
                return schema(guidance="Singer on left third, silhouette doorway on right.")
            return PrevizPlan.model_validate(
                {
                    "stand_in_mapping": "Dark figure is singer.",
                    "actors": [{"id": "singer", "kind": "figure", "role": "primary"}],
                    "camera": [{"time": 0.0, "eye": [0.0, 1.5, 3.0], "look_at": [0.0, 1.2, 0.0]}],
                }
            )

    vision_llm = _FakeLLM(reject_images=False)
    plan = plan_previz(shot, manifest, vision_llm, sketch_path=sketch)
    assert plan.actors
    assert any(b.get("type") == "image" for b in vision_llm.calls[0])

    summary = describe_sketch_blocking(shot, manifest, sketch, vision_llm)
    assert "lower" in summary or "Singer" in summary

    text_only_llm = _FakeLLM(reject_images=True)
    fallback_plan = refine_previz(
        shot, manifest, plan, "Tighter shot.", text_only_llm, sketch_path=sketch
    )
    assert fallback_plan.actors
    assert len(text_only_llm.calls) == 2
    assert describe_sketch_blocking(shot, manifest, sketch, text_only_llm) == ""


def test_broll_sketch_prose_prompt_includes_sketch_instruction_without_guidance(
    project, analysis, tmp_path, sample_png
) -> None:
    manifest = build_manifest(project.config, analysis, "lyrics", "style", llm=None)
    broll = manifest.shots[1]
    broll.type = ShotType.BROLL
    broll.sketch_guidance = ""
    project.save_manifest(manifest)

    sketch_src = tmp_path / "broll_sketch.png"
    Image.new("RGB", (128, 72), "grey").save(sketch_src)
    api.attach_shot_sketch(project.root, broll.id, sketch_src)
    manifest = project.load_manifest()
    broll = manifest.shot(broll.id)

    # Default t2i workflow accepts `reference_image` (Tier 2 single-ref BROLL path).
    fake = FakeComfy(png=sample_png)
    templates = load_templates(project, ["still", "t2i", "lipsync", "i2v"])
    runner = ShotRunner(
        project,
        manifest,
        ComfyClient(project.config.comfy, transport=fake.transport(), sleep=lambda _: None),
        offline_gates(project.config),
        RetryPolicy(project.config.retry, project.config.seed, llm=None),
        templates,
        RunBudget(5),
    )
    runner.run([AssetKind.STILL], [broll.id])
    assert f"{broll.id}.png" in fake.upload_files
    queued_prompt = fake.queued[0]["1"]["inputs"]["text"]
    assert "rough framing and composition sketch" in queued_prompt
    assert "camera angle" in queued_prompt
    assert "blocking" in queued_prompt

    direct_prose = prompts.compose_still_prompt(
        manifest,
        broll,
        plate_attached=False,
        reference_notation="prose",
        bound_references=["sketch_image"],
        sketch_attached=True,
        sketch_guidance="",
    )
    assert "rough framing and composition sketch" in direct_prose
    assert "photorealistic scene details and textures" in direct_prose
