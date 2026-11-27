# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Tests for planning.max_props, generic N-slot ref binding, secondary faces, and prompt tags."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from PIL import Image
from pydantic import ValidationError

from mvdirector.comfy.client import ComfyClient
from mvdirector.comfy.workflow import WorkflowBindings, WorkflowTemplate, is_optional_file_param
from mvdirector.config import PlanningConfig, ProjectConfig
from mvdirector.generation.runner import RunBudget, ShotRunner
from mvdirector.models import (
    AssetKind,
    Character,
    Framing,
    PropDesign,
    ReferenceSet,
    SetDesign,
    ShotType,
)
from mvdirector.planning import prompts
from mvdirector.planning.planner import (
    BibleReply,
    PropReply,
    SetReply,
    ShotBatchReply,
    build_manifest,
)
from mvdirector.qc.factory import QualityGates
from mvdirector.retry import RetryPolicy
from tests.conftest import FakeComfy
from tests.test_pipeline import ScriptedGate


def _make_template(
    name: str,
    ref_params: list[str],
    *,
    extra_params: list[str] | None = None,
    with_video_length: bool = False,
) -> WorkflowTemplate:
    inputs: dict[str, list[dict[str, str]]] = {
        "positive_prompt": [{"node": "6", "field": "text"}],
        "negative_prompt": [{"node": "7", "field": "text"}],
        "seed": [{"node": "3", "field": "seed"}],
        "width": [{"node": "5", "field": "width"}],
        "height": [{"node": "5", "field": "height"}],
    }
    if with_video_length:
        inputs["image"] = [{"node": "1", "field": "image"}]
        inputs["num_frames"] = [{"node": "2", "field": "length"}]
    for p in extra_params or []:
        inputs[p] = [{"node": "2", "field": p}]
    graph: dict[str, Any] = {
        "1": {"class_type": "LoadImage", "inputs": {"image": "stub.png"}},
        "2": {"class_type": "Sampler", "inputs": {"length": 17 if with_video_length else 0}},
        "3": {"class_type": "KSampler", "inputs": {"seed": 0}},
        "5": {"class_type": "EmptyLatent", "inputs": {"width": 512, "height": 512}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": ""}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": ""}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["2", 0]}},
    }
    for idx, param in enumerate(ref_params, start=10):
        nid = str(idx)
        graph[nid] = {"class_type": "LoadImage", "inputs": {"image": f"{param}.png"}}
        inputs[param] = [{"node": nid, "field": "image"}]
        graph["9"]["inputs"][f"link_{param}"] = [nid, 0]
    bindings = WorkflowBindings.model_validate({"workflow": "w.json", "inputs": inputs})
    return WorkflowTemplate(name, bindings, graph)


def _write_png(path: Path, color: tuple[int, int, int] = (200, 120, 40)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), color).save(path)
    return path


def test_max_props_config_validation_and_bible_system_prompt() -> None:
    cfg = PlanningConfig()
    assert cfg.max_props == 3
    assert PlanningConfig(max_props=0).max_props == 0
    assert PlanningConfig(max_props=30).max_props == 30

    for bad in (-1, 31, True, False):
        with pytest.raises(ValidationError):
            PlanningConfig(max_props=bad)  # type: ignore[arg-type]

    assert prompts.bible_system_prompt(prompts.BIBLE_SYSTEM, 3) is prompts.BIBLE_SYSTEM
    assert (
        prompts.bible_system_prompt(prompts.VISUAL_BIBLE_SYSTEM, 3) is prompts.VISUAL_BIBLE_SYSTEM
    )
    zero = prompts.bible_system_prompt(prompts.BIBLE_SYSTEM, 0)
    assert "0 recurring hero objects (always return an empty props list [])" in zero
    eight = prompts.bible_system_prompt(prompts.VISUAL_BIBLE_SYSTEM, 8)
    assert "0 to 8 recurring hero objects" in eight


def test_planner_enforces_max_props_in_system_prompt_and_output_slice(analysis) -> None:
    class _OverPropLLM:
        def __init__(self) -> None:
            self.systems: list[str] = []

        def complete_json(self, *, system, content, schema):
            self.systems.append(system)
            if schema is BibleReply:
                return BibleReply(
                    appearance="Lead singer",
                    style="film",
                    palette="warm",
                    lighting="soft",
                    negative_prompt="blur",
                    settings=[SetReply(name="Studio", description="Brick loft")],
                    props=[
                        PropReply(id=f"prop_{i}", name=f"Prop {i}", description=f"Desc {i}")
                        for i in range(1, 6)
                    ],
                )
            assert schema is ShotBatchReply
            return ShotBatchReply(shots=[])

    # Cap down to 2 props even when LLM returns 5
    llm_cap2 = _OverPropLLM()
    manifest_2 = build_manifest(
        ProjectConfig(planning=PlanningConfig(max_props=2)),
        analysis,
        "lyrics",
        "style",
        llm_cap2,
    )
    assert "0 to 2 recurring hero objects" in llm_cap2.systems[0]
    assert [p.id for p in manifest_2.visual_bible.props] == ["prop_1", "prop_2"]

    # Raise to 5 props
    llm_5 = _OverPropLLM()
    manifest_5 = build_manifest(
        ProjectConfig(planning=PlanningConfig(max_props=5)),
        analysis,
        "lyrics",
        "style",
        llm_5,
    )
    assert "0 to 5 recurring hero objects" in llm_5.systems[0]
    assert len(manifest_5.visual_bible.props) == 5


def test_workflow_optional_params_and_pruning_for_extra_props_and_secondary_faces() -> None:
    for valid in (
        "prop_image",
        "prop_image_2",
        "prop_image_3",
        "prop_image_12",
        "reference_image_2_face",
        "reference_image_4_face",
    ):
        assert is_optional_file_param(valid) is True
    for invalid in (
        "prop_image_0",
        "prop_image_1",
        "prop_image_2\n",
        "prop_image_٢",
        "reference_image_0_face",
        "reference_image_1_face",
        "reference_image_2_face\n",
    ):
        assert is_optional_file_param(invalid) is False

    tpl = _make_template(
        "still",
        [
            "reference_image",
            "reference_image_2",
            "reference_image_2_face",
            "prop_image",
            "prop_image_2",
            "prop_image_3",
        ],
    )
    rendered = tpl.render(
        {
            "positive_prompt": "test",
            "reference_image": "lead.png",
            "reference_image_2": "sec.png",
            "prop_image": "p1.png",
        }
    )
    # reference_image (10), reference_image_2 (11), prop_image (13) stay;
    # unbound reference_image_2_face (12), prop_image_2 (14), prop_image_3 (15) are pruned.
    assert {"10", "11", "13"}.issubset(rendered)
    assert "12" not in rendered
    assert "14" not in rendered
    assert "15" not in rendered
    assert rendered["9"]["inputs"] == {
        "images": ["2", 0],
        "link_reference_image": ["10", 0],
        "link_reference_image_2": ["11", 0],
        "link_prop_image": ["13", 0],
    }


def test_nslot_binding_priority_order_secondary_faces_and_prompt_tags(
    project, analysis, sample_png
) -> None:
    manifest = build_manifest(project.config, analysis, "lyrics", "style", llm=None)
    lead_front = _write_png(project.path("assets/reference/lead_front.png"))
    lead_face = _write_png(project.path("assets/reference/lead_face.png"))
    sec_front = _write_png(project.path("assets/reference/sec_front.png"))
    sec_face = _write_png(project.path("assets/reference/sec_face.png"))
    set_plate = _write_png(project.path("assets/plates/club.png"))
    p_watch = _write_png(project.path("assets/props/watch.png"))
    p_lantern = _write_png(project.path("assets/props/lantern.png"))
    p_compass = _write_png(project.path("assets/props/compass.png"))

    manifest.characters = {
        "lead": Character(
            name="Aria",
            appearance="Tall singer",
            references={
                "base": ReferenceSet(
                    sheet=project.rel(lead_front),
                    panels={"front": project.rel(lead_front), "face": project.rel(lead_face)},
                )
            },
        ),
        "duo": Character(
            name="Kai",
            appearance="Guitarist",
            references={
                "base": ReferenceSet(
                    sheet=project.rel(sec_front),
                    panels={"front": project.rel(sec_front), "face": project.rel(sec_face)},
                )
            },
        ),
    }
    manifest.character = manifest.characters["lead"]
    manifest.visual_bible.settings = [
        SetDesign(name="Club", description="Neon club", plate=project.rel(set_plate))
    ]
    manifest.visual_bible.props = [
        PropDesign(
            id="watch",
            name="Pocket Watch",
            description="Brass watch",
            plate=project.rel(p_watch),
        ),
        PropDesign(
            id="lantern",
            name="Brass Lantern",
            description="Warm lantern",
            plate=project.rel(p_lantern),
        ),
        PropDesign(
            id="compass",
            name="Brass Compass",
            description="Old compass",
            plate=project.rel(p_compass),
        ),
    ]

    shot = next(s for s in manifest.shots if s.type is ShotType.LIPSYNC)
    shot.framing = Framing.MEDIUM_CLOSE
    shot.setting = "Club"
    shot.characters = ["lead", "duo"]
    # Mention Brass Compass first and Brass Lantern second; Pocket Watch is unmentioned.
    shot.props = ["watch", "lantern", "compass"]
    shot.still_prompt = "Aria checks the Brass Compass beside Kai while the Brass Lantern glows."
    shot.motion_prompt = "Aria raises the Brass Compass slowly."

    still_tpl = _make_template(
        "still",
        [
            "reference_image",
            "reference_face",
            "set_image",
            "reference_image_2",
            "reference_image_2_face",
            "prop_image",
            "prop_image_2",
            "prop_image_3",
        ],
    )
    video_tpl = _make_template(
        "lipsync",
        [
            "reference_image",
            "set_image",
            "reference_image_2",
            "reference_image_2_face",
            "prop_image",
            "prop_image_2",
            "prop_image_3",
        ],
        extra_params=["audio", "duration", "fps"],
        with_video_length=True,
    )
    project.config.comfy.reference_notation = "tags"

    fake = FakeComfy(png=sample_png, mp4=b"\x00\x00\x00\x18ftypmp42")
    runner = ShotRunner(
        project,
        manifest,
        ComfyClient(project.config.comfy, transport=fake.transport(), sleep=lambda _: None),
        QualityGates(image=ScriptedGate([True]), video=ScriptedGate([True]), face_analyzer=None),
        RetryPolicy(project.config.retry, project.config.seed, llm=None),
        {"t2i": still_tpl, "still": still_tpl, "lipsync": video_tpl, "i2v": video_tpl},
        RunBudget(10),
    )

    # 1. Verify prop sort order by prompt mention
    assert [name for name, _ in runner._prop_paths(shot)] == [
        "Brass Compass",
        "Brass Lantern",
        "Pocket Watch",
    ]

    # 2. Verify priority order lead > set > secondaries (+ secondary face)
    refs = runner._still_reference_paths(shot, set_plate)
    assert [param for param, _, _ in refs] == [
        "reference_image",
        "reference_face",
        "set_image",
        "reference_image_2",
        "reference_image_2_face",
    ]

    # 3. Rear-view performance shot skips frontal secondary face panel
    rear_shot = shot.model_copy(
        update={
            "type": ShotType.PERFORMANCE,
            "still_prompt": "Rear view from behind of Aria and Kai walking away.",
        }
    )
    rear_refs = runner._still_reference_paths(rear_shot, set_plate)
    assert "reference_image_2_face" not in [p for p, _, _ in rear_refs]
    assert "reference_face" not in [p for p, _, _ in rear_refs]

    # 4. End-to-end still + video run binds all 3 props and tags every reference
    runner.run([AssetKind.STILL], shot_ids=[shot.id])
    still_prompt = shot.still.attempts[0].prompt
    assert "@image1 shows the singer" in still_prompt
    assert "@image2 shows a close-up of the singer's face" in still_prompt
    assert "@image3 shows the location" in still_prompt
    assert "@image4 shows the secondary performer" in still_prompt
    assert "@image5 shows a close-up of the secondary performer's face" in still_prompt
    assert "@image6 shows the hero prop (Brass Compass)" in still_prompt
    assert "@image7 shows the hero prop (Brass Lantern)" in still_prompt
    assert "@image8 shows the hero prop (Pocket Watch)" in still_prompt
    assert (
        "Match the exact design, scale and details of the prop reference "
        "(Brass Compass, Brass Lantern, Pocket Watch)."
    ) in still_prompt
    # Verify all 3 prop files were uploaded to ComfyUI in priority order
    assert fake.upload_files[-3:] == ["compass.png", "lantern.png", "watch.png"]

    # Motion prompt tags (@image1 is starting still, @image2..@image8 are refs + props)
    motion_prompt = prompts.compose_motion_prompt(
        manifest,
        shot,
        video_config=project.config.video,
        reference_notation="tags",
        bound_references=[
            "reference_image",
            "set_image",
            "reference_image_2",
            "reference_image_2_face",
            "prop_image",
            "prop_image_2",
            "prop_image_3",
        ],
        bound_prop_names={
            "prop_image": "Brass Compass",
            "prop_image_2": "Brass Lantern",
            "prop_image_3": "Pocket Watch",
        },
    )
    assert "Animate starting from @image1." in motion_prompt
    assert "@image2 shows the singer" in motion_prompt
    assert "@image3 shows the location" in motion_prompt
    assert "@image4 shows the secondary performer" in motion_prompt
    assert "@image5 shows a close-up of the secondary performer's face" in motion_prompt
    assert "@image6 shows the hero prop (Brass Compass)" in motion_prompt
    assert "@image7 shows the hero prop (Brass Lantern)" in motion_prompt
    assert "@image8 shows the hero prop (Pocket Watch)" in motion_prompt


def test_compose_still_prompt_prose_disambiguates_secondary_face_and_overflow_props(
    project, analysis
) -> None:
    manifest = build_manifest(project.config, analysis, "lyrics", "style", llm=None)
    manifest.characters = {
        "lead": Character(name="Aria", appearance="Lead look"),
        "duo": Character(name="Kai", appearance="Secondary look"),
    }
    manifest.character = manifest.characters["lead"]
    manifest.visual_bible.props = [
        PropDesign(id="compass", name="Brass Compass", description="Engraved brass compass"),
        PropDesign(id="lantern", name="Brass Lantern", description="Dented copper lantern"),
    ]
    shot = next(s for s in manifest.shots if s.type is ShotType.LIPSYNC)
    shot.characters = ["lead", "duo"]
    shot.props = ["compass", "lantern"]

    prose = prompts.compose_still_prompt(
        manifest,
        shot,
        plate_attached=True,
        props_attached=True,
        reference_notation="prose",
        bound_references=[
            "reference_image",
            "set_image",
            "reference_image_2",
            "reference_image_2_face",
            "prop_image",
        ],
        bound_prop_names={"prop_image": "Brass Compass"},
    )
    assert "Picture 4 defines Kai's close-up facial features" in prose
    assert "Picture 5 defines Brass Compass's design" in prose
    assert (
        "Match the exact design, scale and details of the prop reference (Brass Compass)."
    ) in prose
    # Unattached overflow prop retains text description
    assert "Hero prop (Brass Lantern): Dented copper lantern" in prose


def test_colliding_set_and_prop_slot_preserves_set_image_and_tags(
    project, analysis, sample_png
) -> None:
    manifest = build_manifest(project.config, analysis, "lyrics", "style", llm=None)
    lead_front = _write_png(project.path("assets/reference/lead_front.png"), (220, 40, 40))
    set_plate = _write_png(project.path("assets/plates/club.png"), (40, 220, 40))
    p_compass = _write_png(project.path("assets/props/compass.png"), (40, 40, 220))

    manifest.characters = {
        "lead": Character(
            name="Aria",
            appearance="Tall singer",
            references={
                "base": ReferenceSet(
                    sheet=project.rel(lead_front),
                    panels={"front": project.rel(lead_front)},
                )
            },
        )
    }
    manifest.character = manifest.characters["lead"]
    manifest.visual_bible.settings = [
        SetDesign(name="Club", description="Neon club", plate=project.rel(set_plate))
    ]
    manifest.visual_bible.props = [
        PropDesign(
            id="compass",
            name="Brass Compass",
            description="Old compass",
            plate=project.rel(p_compass),
        )
    ]

    shot = next(s for s in manifest.shots if s.type is ShotType.LIPSYNC)
    shot.framing = Framing.MEDIUM_CLOSE
    shot.setting = "Club"
    shot.characters = ["lead"]
    shot.props = ["compass"]
    shot.still_prompt = "Aria checks the Brass Compass."
    shot.motion_prompt = "Aria raises the Brass Compass."

    def _colliding_template(name: str, *, with_video: bool) -> WorkflowTemplate:
        tpl = _make_template(
            name,
            ["reference_image", "set_image", "prop_image_2", "prop_image_3"],
            extra_params=["audio", "duration", "fps"] if with_video else None,
            with_video_length=with_video,
        )
        if with_video:
            tpl.bindings.output.extensions = [".mp4"]
        # Alias prop_image to the exact same LoadImage node ("11") as set_image.
        tpl.bindings.inputs["prop_image"] = list(tpl.bindings.inputs["set_image"])
        return tpl

    still_tpl = _colliding_template("still", with_video=False)
    video_tpl = _colliding_template("lipsync", with_video=True)
    project.config.comfy.reference_notation = "tags"

    fake = FakeComfy(png=sample_png, mp4=b"\x00\x00\x00\x18ftypmp42")
    runner = ShotRunner(
        project,
        manifest,
        ComfyClient(project.config.comfy, transport=fake.transport(), sleep=lambda _: None),
        QualityGates(image=ScriptedGate([True]), video=ScriptedGate([True]), face_analyzer=None),
        RetryPolicy(project.config.retry, project.config.seed, llm=None),
        {"t2i": still_tpl, "still": still_tpl, "lipsync": video_tpl, "i2v": video_tpl},
        RunBudget(10),
    )

    runner.run([AssetKind.STILL, AssetKind.VIDEO], shot_ids=[shot.id])

    upload_by_name = {
        f"mvdirector/{name}": filename
        for name, filename in zip(fake.uploads, fake.upload_files, strict=True)
    }
    # Still render: node 10 = lead, node 11 = set_image (NOT overwritten by prop), node 12 = prop
    still_graph = fake.queued[0]
    assert upload_by_name[still_graph["10"]["inputs"]["image"]] == "lead_front.png"
    assert upload_by_name[still_graph["11"]["inputs"]["image"]] == "club.png"
    assert upload_by_name[still_graph["12"]["inputs"]["image"]] == "compass.png"
    assert "13" not in still_graph  # single prop is not duplicated into prop_image_3

    still_prompt = shot.still.attempts[0].prompt
    assert "@image1 shows the singer" in still_prompt
    assert "@image2 shows the location" in still_prompt
    assert "@image3 shows the hero prop (Brass Compass)" in still_prompt
    assert "@image4" not in still_prompt

    # Video render: node 1 = starting still, node 10 = lead, node 11 = set_image, node 12 = prop
    video_graph = fake.queued[1]
    assert upload_by_name[video_graph["1"]["inputs"]["image"]].endswith("_a01.png")
    assert upload_by_name[video_graph["10"]["inputs"]["image"]] == "lead_front.png"
    assert upload_by_name[video_graph["11"]["inputs"]["image"]] == "club.png"
    assert upload_by_name[video_graph["12"]["inputs"]["image"]] == "compass.png"
    assert "13" not in video_graph

    motion_prompt = shot.video.attempts[0].prompt
    assert "Animate starting from @image1." in motion_prompt
    assert "@image2 shows the singer" in motion_prompt
    assert "@image3 shows the location" in motion_prompt
    assert "@image4 shows the hero prop (Brass Compass)" in motion_prompt
    assert "@image5" not in motion_prompt
