# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Tests for dual-panel face+body reference binding and reference-sheet garment QC."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mvdirector.comfy.workflow import WorkflowTemplate
from mvdirector.config import ProjectConfig
from mvdirector.generation.runner import ShotRunner
from mvdirector.models import (
    AudioSlice,
    Character,
    Framing,
    Manifest,
    ReferenceSet,
    Shot,
    ShotType,
    VisualBible,
    WardrobeState,
)
from mvdirector.planning import prompts
from mvdirector.project import Project
from mvdirector.qc import vision


def _write_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Minimal 1x1 PNG bytes sufficient for file existence / upload mocking.
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\xcf\xc0"
        b"\x00\x00\x03\x01\x01\x00\x18\xdd\x8d\xb0\x00\x00\x00\x00IEND\xaeB`\x82"
    )


def _make_template(
    tmp_path: Path, name: str, inputs: dict[str, list[dict[str, str]]]
) -> WorkflowTemplate:
    graph: dict[str, Any] = {
        "1": {"class_type": "PromptNode", "inputs": {"prompt": ""}},
        "2": {"class_type": "NegNode", "inputs": {"prompt": ""}},
        "3": {"class_type": "Sampler", "inputs": {"seed": 0}},
        "10": {"class_type": "LoadImage", "inputs": {"image": ""}},
        "11": {"class_type": "LoadImage", "inputs": {"image": ""}},
        "12": {"class_type": "LoadImage", "inputs": {"image": ""}},
        "13": {"class_type": "LoadImage", "inputs": {"image": ""}},
        "99": {"class_type": "SaveImage", "inputs": {}},
    }
    wf_dir = tmp_path / "wf"
    wf_dir.mkdir(parents=True, exist_ok=True)
    (wf_dir / f"{name}.api.json").write_text(json.dumps(graph), "utf-8")
    bindings = {
        "workflow": f"{name}.api.json",
        "inputs": {
            "positive_prompt": [{"node": "1", "field": "prompt"}],
            "negative_prompt": [{"node": "2", "field": "prompt"}],
            "seed": [{"node": "3", "field": "seed"}],
            **inputs,
        },
        "required": ["positive_prompt", "seed"],
        "output": {"node": "99", "extensions": [".png"]},
    }
    bpath = wf_dir / f"{name}.bindings.json"
    bpath.write_text(json.dumps(bindings), "utf-8")
    return WorkflowTemplate.load(name, bpath)


def _make_project_and_manifest(
    tmp_path: Path, *, with_face_file: bool = True, two_characters: bool = False
) -> tuple[Project, Manifest, Shot]:
    root = tmp_path / "proj"
    root.mkdir(parents=True, exist_ok=True)
    proj = Project(root, ProjectConfig(title="Dual Panel Test"))
    front_rel = "assets/reference/cinder/front.png"
    face_rel = "assets/reference/cinder/face.png"
    _write_png(proj.path(front_rel))
    if with_face_file:
        _write_png(proj.path(face_rel))
    lead = Character(
        name="The Architect",
        appearance="Sharp jawline, blunt black bob.",
        wardrobe=[WardrobeState(id="cinder", description="Scorched black coat and corset.")],
        references={
            "cinder": ReferenceSet(
                sheet=front_rel,
                panels={"front": front_rel, "face": face_rel},
            )
        },
    )
    characters: dict[str, Character] = {}
    shot_chars: list[str] = []
    if two_characters:
        sec_rel = "assets/reference/guest/front.png"
        _write_png(proj.path(sec_rel))
        guest = Character(
            name="Hansel",
            appearance="Tall man in wool coat.",
            wardrobe=[WardrobeState(id="default", description="Dark wool coat.")],
            references={"default": ReferenceSet(sheet=sec_rel, panels={"front": sec_rel})},
        )
        characters = {"lead": lead, "guest": guest}
        shot_chars = ["lead", "guest"]
    shot = Shot(
        id="s001",
        index=0,
        start=0.0,
        end=3.0,
        type=ShotType.LIPSYNC,
        framing=Framing.CLOSE_UP,
        setting="Greenhouse",
        wardrobe="cinder",
        character="lead" if two_characters else None,
        characters=shot_chars,
        still_prompt="Eye-level close-up in the frosted greenhouse.",
        audio_slice=AudioSlice(path="assets/audio/s001.wav", start=0.0, end=3.0, head_handle=0.0),
        num_frames=73,
    )
    manifest = Manifest(
        title="Dual Panel Test",
        duration=3.0,
        fps=24,
        character=lead,
        characters=characters,
        visual_bible=VisualBible(style="Dark cinema"),
        shots=[shot],
    )
    return proj, manifest, shot


def _make_runner(proj: Project, manifest: Manifest, still_tpl: WorkflowTemplate) -> ShotRunner:
    runner = ShotRunner.__new__(ShotRunner)
    runner._project = proj
    runner._manifest = manifest
    runner._templates = {"still": still_tpl}
    return runner


def test_solo_shot_routes_face_panel_into_reference_image_2(tmp_path: Path) -> None:
    proj, manifest, shot = _make_project_and_manifest(tmp_path)
    tpl = _make_template(
        tmp_path,
        "dual_generic",
        {
            "reference_image": [{"node": "10", "field": "image"}],
            "reference_image_2": [{"node": "11", "field": "image"}],
        },
    )
    runner = _make_runner(proj, manifest, tpl)
    refs = runner._still_reference_paths(shot, plate=None)
    assert [(param, role) for param, role, _ in refs] == [
        ("reference_image", "reference_image"),
        ("reference_image_2", "reference_face"),
    ]
    assert refs[0][2] == proj.path("assets/reference/cinder/front.png")
    assert refs[1][2] == proj.path("assets/reference/cinder/face.png")


def test_solo_shot_with_explicit_reference_face_does_not_duplicate_into_slot_2(
    tmp_path: Path,
) -> None:
    proj, manifest, shot = _make_project_and_manifest(tmp_path)
    tpl = _make_template(
        tmp_path,
        "explicit_face",
        {
            "reference_image": [{"node": "10", "field": "image"}],
            "reference_face": [{"node": "11", "field": "image"}],
            "reference_image_2": [{"node": "12", "field": "image"}],
        },
    )
    runner = _make_runner(proj, manifest, tpl)
    refs = runner._still_reference_paths(shot, plate=None)
    assert [(param, role) for param, role, _ in refs] == [
        ("reference_image", "reference_image"),
        ("reference_face", "reference_face"),
    ]


def test_two_character_shot_keeps_secondary_performer_in_reference_image_2(tmp_path: Path) -> None:
    proj, manifest, shot = _make_project_and_manifest(tmp_path, two_characters=True)
    tpl = _make_template(
        tmp_path,
        "duet",
        {
            "reference_image": [{"node": "10", "field": "image"}],
            "reference_image_2": [{"node": "11", "field": "image"}],
        },
    )
    runner = _make_runner(proj, manifest, tpl)
    refs = runner._still_reference_paths(shot, plate=None)
    assert [(param, role) for param, role, _ in refs] == [
        ("reference_image", "reference_image"),
        ("reference_image_2", "reference_image_2"),
    ]
    assert refs[1][2] == proj.path("assets/reference/guest/front.png")


def test_solo_face_fallback_skips_missing_face_file(tmp_path: Path) -> None:
    proj, manifest, shot = _make_project_and_manifest(tmp_path, with_face_file=False)
    tpl = _make_template(
        tmp_path,
        "dual_missing_face",
        {
            "reference_image": [{"node": "10", "field": "image"}],
            "reference_image_2": [{"node": "11", "field": "image"}],
        },
    )
    runner = _make_runner(proj, manifest, tpl)
    refs = runner._still_reference_paths(shot, plate=None)
    assert [(param, role) for param, role, _ in refs] == [("reference_image", "reference_image")]


def test_compose_still_prompt_describes_dual_panel_face_and_body_in_prose_and_tags(
    tmp_path: Path,
) -> None:
    _, manifest, shot = _make_project_and_manifest(tmp_path)
    prose = prompts.compose_still_prompt(
        manifest,
        shot,
        plate_attached=False,
        reference_notation="prose",
        bound_references=["reference_image", "reference_face"],
    )
    assert "The first reference image shows the singer" in prose
    assert "The second reference image shows a close-up of the singer's face" in prose
    assert "Picture 1 defines The Architect's appearance" in prose
    assert "Picture 2 defines The Architect's close-up facial features" in prose

    tagged = prompts.compose_still_prompt(
        manifest,
        shot,
        plate_attached=False,
        reference_notation="tags",
        bound_references=["reference_image", "reference_face"],
    )
    assert "@image1 shows the singer" in tagged
    assert "@image2 shows a close-up of the singer's face" in tagged


def test_compose_still_prompt_preserves_plate_note_for_single_ref_plus_set(
    tmp_path: Path,
) -> None:
    _, manifest, shot = _make_project_and_manifest(tmp_path)
    prose = prompts.compose_still_prompt(
        manifest,
        shot,
        plate_attached=True,
        reference_notation="prose",
        bound_references=["reference_image", "set_image"],
    )
    assert prompts.PLATE_NOTE in prose


def test_vision_prompts_enforce_reference_sheet_garment_fidelity_and_priority() -> None:
    assert "Reference-image priority" in vision._SYSTEM
    assert "turtleneck" in vision._SYSTEM
    assert "character_match <= 0.4" in vision._SYSTEM
    assert "anatomical left/right placement" in vision._SYSTEM
    assert "neckline coverage" in vision._BLIND_OBSERVER_SYSTEM
    assert "character_match <= 0.4" in vision._BLIND_AUDITOR_SYSTEM
    assert "text-vs-reference discrepancies" in vision._BLIND_AUDITOR_SYSTEM


def test_compose_still_prompt_numbers_dual_panel_plus_set_plate_without_contradiction(
    tmp_path: Path,
) -> None:
    _, manifest, shot = _make_project_and_manifest(tmp_path)
    prose = prompts.compose_still_prompt(
        manifest,
        shot,
        plate_attached=True,
        reference_notation="prose",
        bound_references=["reference_image", "reference_face", "set_image"],
    )
    assert "The first reference image shows the singer" in prose
    assert "The second reference image shows a close-up of the singer's face" in prose
    assert "The third reference image shows the location" in prose
    assert "The second reference image shows the location" not in prose
    assert "Picture 1 defines The Architect's appearance" in prose
    assert "Picture 2 defines The Architect's close-up facial features" in prose
    assert "Picture 3 defines only the empty location" in prose


def test_solo_face_fallback_skips_rear_view_shots_and_aliased_front_paths(tmp_path: Path) -> None:
    proj, manifest, shot = _make_project_and_manifest(tmp_path)
    tpl = _make_template(
        tmp_path,
        "dual_rear_and_alias",
        {
            "reference_image": [{"node": "10", "field": "image"}],
            "reference_image_2": [{"node": "11", "field": "image"}],
        },
    )
    runner = _make_runner(proj, manifest, tpl)

    # Rear-view non-lipsync performance shots must not inject a frontal face close-up.
    rear_shot = shot.model_copy(
        update={
            "type": ShotType.PERFORMANCE,
            "still_prompt": "Wide shot from behind looking away from camera down the hall.",
        }
    )
    rear_refs = runner._still_reference_paths(rear_shot, plate=None)
    assert [(param, role) for param, role, _ in rear_refs] == [
        ("reference_image", "reference_image")
    ]

    # Equivalent relative paths resolving to the same file as front must not duplicate into slot 2.
    manifest.character.references["cinder"] = ReferenceSet(
        sheet="assets/reference/cinder/front.png",
        panels={
            "front": "assets/reference/cinder/front.png",
            "face": "./assets/reference/cinder/front.png",
        },
    )
    aliased_refs = runner._still_reference_paths(shot, plate=None)
    assert [(param, role) for param, role, _ in aliased_refs] == [
        ("reference_image", "reference_image")
    ]
