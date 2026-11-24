# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Tests for multi-member band/ensemble reference bindings in Still/Video runner."""

from __future__ import annotations

from pathlib import Path

from mvdirector.comfy.workflow import WorkflowBindings, WorkflowTemplate
from mvdirector.generation.runner import RunBudget, ShotRunner
from mvdirector.models import Character, ReferenceSet, ShotType
from mvdirector.planning.planner import build_manifest


def _make_still_template(ref_params: list[str]) -> WorkflowTemplate:
    graph = {
        "1": {"class_type": "CLIPTextEncode", "inputs": {"text": ""}},
        "2": {"class_type": "KSampler", "inputs": {"seed": 1}},
        "9": {"class_type": "SaveImage", "inputs": {}},
    }
    inputs: dict[str, list[dict[str, str]]] = {
        "positive_prompt": [{"node": "1", "field": "text"}],
        "seed": [{"node": "2", "field": "seed"}],
    }
    for idx, param in enumerate(ref_params, start=10):
        node_id = str(idx)
        graph[node_id] = {"class_type": "LoadImage", "inputs": {"image": ""}}
        inputs[param] = [{"node": node_id, "field": "image"}]
    bindings = {
        "workflow": "stub.api.json",
        "inputs": inputs,
        "required": ["positive_prompt"],
        "output": {"node": "9", "extensions": [".png"]},
    }
    return WorkflowTemplate(
        name="still",
        graph=graph,
        bindings=WorkflowBindings.model_validate(bindings),
    )


def test_still_references_bind_three_band_members(project, analysis):
    manifest = build_manifest(project.config, analysis, "lyrics", "style", llm=None)
    ref_dir = project.reference_dir
    ref_dir.mkdir(parents=True, exist_ok=True)

    chars: dict[str, Character] = {}
    for cid, role in (("lead", "Lead Singer"), ("guitarist", "Guitarist"), ("alto", "Alto")):
        sheet = ref_dir / f"{cid}_base_sheet.png"
        sheet.write_bytes(b"png")
        chars[cid] = Character(
            name=role,
            appearance=f"{role} appearance",
            references={"base": ReferenceSet(sheet=project.rel(sheet), panels={})},
        )
    manifest.characters = chars
    manifest.character = chars["lead"]

    shot = next(s for s in manifest.shots if s.type is ShotType.LIPSYNC)
    shot.characters = ["lead", "guitarist", "alto"]

    # Workflow accepting 3 character references binds all 3 in visible order
    tpl_3 = _make_still_template(["reference_image", "reference_image_2", "reference_image_3"])
    runner_3 = ShotRunner(
        project=project,
        manifest=manifest,
        comfy=None,  # type: ignore[arg-type]
        gates=None,  # type: ignore[arg-type]
        policy=None,  # type: ignore[arg-type]
        templates={"still": tpl_3},
        budget=RunBudget(limit=10),
    )
    refs_3 = runner_3._still_reference_paths(shot, plate=None)
    assert [name for name, _ in refs_3] == [
        "reference_image",
        "reference_image_2",
        "reference_image_3",
    ]
    assert [Path(p).name for _, p in refs_3] == [
        "lead_base_sheet.png",
        "guitarist_base_sheet.png",
        "alto_base_sheet.png",
    ]

    # Workflow accepting only 2 character references gracefully binds the first 2
    tpl_2 = _make_still_template(["reference_image", "reference_image_2"])
    runner_2 = ShotRunner(
        project=project,
        manifest=manifest,
        comfy=None,  # type: ignore[arg-type]
        gates=None,  # type: ignore[arg-type]
        policy=None,  # type: ignore[arg-type]
        templates={"still": tpl_2},
        budget=RunBudget(limit=10),
    )
    refs_2 = runner_2._still_reference_paths(shot, plate=None)
    assert [name for name, _ in refs_2] == ["reference_image", "reference_image_2"]

    # Missing secondary file on disk is skipped cleanly rather than raising FileNotFoundError
    (ref_dir / "guitarist_base_sheet.png").unlink()
    refs_missing = runner_3._still_reference_paths(shot, plate=None)
    assert [name for name, _ in refs_missing] == ["reference_image", "reference_image_3"]


def test_workflow_render_drops_unused_higher_band_reference_loaders() -> None:
    tpl = _make_still_template(
        ["reference_image", "reference_image_2", "reference_image_3", "reference_image_4"]
    )
    # Attach consumer links so we verify upstream links are also cleaned up.
    tpl._graph["9"]["inputs"] = {
        "ref1": ["10", 0],
        "ref2": ["11", 0],
        "ref3": ["12", 0],
        "ref4": ["13", 0],
    }

    solo_graph = tpl.render({"positive_prompt": "solo", "reference_image": "lead.png"})
    assert "10" in solo_graph
    assert "11" not in solo_graph
    assert "12" not in solo_graph
    assert "13" not in solo_graph
    assert solo_graph["9"]["inputs"] == {"ref1": ["10", 0]}

    trio_graph = tpl.render(
        {
            "positive_prompt": "trio",
            "reference_image": "lead.png",
            "reference_image_2": "g.png",
            "reference_image_3": "a.png",
        }
    )
    assert {"10", "11", "12"}.issubset(trio_graph)
    assert "13" not in trio_graph


def test_compose_still_prompt_numbers_three_plus_band_refs_and_plate(project, analysis) -> None:
    from mvdirector.planning import prompts

    manifest = build_manifest(project.config, analysis, "lyrics", "style", llm=None)
    chars: dict[str, Character] = {
        "lead": Character(name="Lead Singer", appearance="Lead look."),
        "guitarist": Character(name="Guitarist", appearance="Guitarist look."),
        "alto": Character(name="Alto", appearance="Alto look."),
    }
    manifest.characters = chars
    manifest.character = chars["lead"]
    shot = next(s for s in manifest.shots if s.type is ShotType.LIPSYNC)
    shot.characters = ["lead", "guitarist", "alto"]

    bound = ["reference_image", "reference_image_2", "reference_image_3", "set_image"]
    tagged = prompts.compose_still_prompt(
        manifest, shot, plate_attached=True, reference_notation="tags", bound_references=bound
    )
    assert "@image1 shows the singer" in tagged
    assert "@image2 shows the secondary performer" in tagged
    assert "@image3 shows performer #3" in tagged
    assert "@image4 shows the location" in tagged

    prose = prompts.compose_still_prompt(
        manifest, shot, plate_attached=True, reference_notation="prose", bound_references=bound
    )
    assert "third reference image shows performer #3" in prose
    assert "fourth reference image shows the location" in prose
    assert "Picture 1 defines Lead Singer's appearance" in prose
    assert "Picture 2 defines Guitarist's appearance" in prose
    assert "Picture 3 defines Alto's appearance" in prose
    assert "Picture 4 defines only the empty location" in prose

    # Sparse binding (member 2 has no ref, member 3 has ref): still labels Alto correctly.
    sparse = prompts.compose_still_prompt(
        manifest,
        shot,
        plate_attached=False,
        reference_notation="prose",
        bound_references=["reference_image", "reference_image_3"],
    )
    assert "second reference image shows performer #3" in sparse
    assert "Picture 2 defines Alto's appearance" in sparse
