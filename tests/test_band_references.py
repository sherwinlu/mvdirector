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
        budget=RunBudget(limit=None),
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
        budget=RunBudget(limit=None),
    )
    refs_2 = runner_2._still_reference_paths(shot, plate=None)
    assert [name for name, _ in refs_2] == ["reference_image", "reference_image_2"]
