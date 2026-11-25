# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Tests for F10: 2x2 multi-angle location turnaround set plates and shot angle selection."""

from __future__ import annotations

import io
import json

import pytest
from PIL import Image
from typer.testing import CliRunner

from mvdirector import api
from mvdirector.cli import app
from mvdirector.comfy.client import ComfyClient
from mvdirector.comfy.provider import StaticComfyProvider
from mvdirector.errors import ConfigError
from mvdirector.generation import plates, preflight
from mvdirector.generation.runner import RunBudget, ShotRunner, load_templates
from mvdirector.models import (
    PLATE_ANGLE_NAMES,
    Framing,
    ReferenceSet,
    SetDesign,
    ShotType,
)
from mvdirector.planning import prompts
from mvdirector.planning.planner import build_manifest
from mvdirector.qc.factory import build_quality_gates
from mvdirector.retry import RetryPolicy
from tests.conftest import FakeComfy, write_workflows


def _make_png_bytes(width: int, height: int, color: tuple[int, int, int] = (40, 80, 120)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buf, format="PNG")
    return buf.getvalue()


class _FakeProvider(StaticComfyProvider):
    def __init__(self, fake: FakeComfy, project) -> None:
        super().__init__()
        self._fake = fake
        self._project = project

    def lease(self, project):
        from contextlib import contextmanager

        @contextmanager
        def _cm():
            yield ComfyClient(
                self._project.config.comfy,
                transport=self._fake.transport(),
                sleep=lambda _: None,
            )

        return _cm()


def test_plate_turnaround_fails_fast_without_plate_or_ref_slot(
    project, analysis, sample_png
) -> None:
    manifest = build_manifest(project.config, analysis, "lyrics", "style", llm=None)
    manifest.visual_bible.settings = [
        SetDesign(name="Neon Alley", description="Wet cobblestone alley with neon signs.")
    ]
    fake = FakeComfy(png=sample_png)
    comfy = ComfyClient(project.config.comfy, transport=fake.transport(), sleep=lambda _: None)
    templates = load_templates(project, ["still"])

    # No locked plate -> fails before contacting ComfyUI.
    with pytest.raises(ConfigError, match="Lock a plate"):
        plates.generate_turnaround(project, manifest, comfy, templates, "Neon Alley")
    assert fake.queued == []

    # Locked plate, but workflow has no reference_image or set_image -> fails fast.
    src_plate = project.root / "plate.png"
    Image.new("RGB", (128, 72), "navy").save(src_plate)
    plates.lock_plate(project, manifest, src_plate, "Neon Alley")
    write_workflows(project.path("workflows"), {"still": ".png"})
    # Remove reference_image binding from still.bindings.json to simulate t2i-only graph.
    still_b = project.path("workflows/still.bindings.json")
    data = json.loads(still_b.read_text())
    data["inputs"].pop("reference_image", None)
    still_b.write_text(json.dumps(data))
    bad_templates = load_templates(project, ["still"])
    with pytest.raises(ConfigError, match="must bind 'reference_image' or 'set_image'"):
        plates.generate_turnaround(project, manifest, comfy, bad_templates, "Neon Alley")
    assert fake.queued == []


def test_plate_turnaround_patina_policy_and_aspect_ratios(project, analysis, tmp_path) -> None:
    manifest = build_manifest(project.config, analysis, "lyrics", "style", llm=None)
    manifest.visual_bible.settings = [
        SetDesign(name="Glass Hall", description="Tall arched gallery.")
    ]
    manifest.visual_bible.style = "35mm gritty noir"
    gritty = prompts.plate_turnaround_prompt(manifest, "Glass Hall")
    assert "patina" in gritty.lower()
    assert "wide establishing" in gritty.lower()
    assert "180-degree reverse" in gritty.lower()

    manifest.visual_bible.style = "sleek futuristic chrome minimalism"
    sleek = prompts.plate_turnaround_prompt(manifest, "Glass Hall")
    assert "patina" not in sleek.lower()

    # 16:9, 1:1, and 9:16 turnaround sheets all crop cleanly into 4 angle panels.
    for idx, (w, h) in enumerate(((1280, 720), (1024, 1024), (720, 1280))):
        sheet_img = tmp_path / f"turnaround_{idx}.png"
        Image.new("RGB", (w, h), "teal").save(sheet_img)
        scene = plates.lock_turnaround(project, manifest, sheet_img, "Glass Hall")
        assert scene.sheet is not None and project.path(scene.sheet).is_file()
        assert tuple(scene.panels) == PLATE_ANGLE_NAMES
        for angle in PLATE_ANGLE_NAMES:
            assert project.path(scene.panels[angle]).is_file()
        # First lock_turnaround populates scene.plate with wide if unset.
        assert scene.plate == scene.panels["wide"]


def test_plate_turnaround_generate_and_runner_angle_selection(
    project, analysis, sample_png, tmp_path
) -> None:
    write_workflows(
        project.path("workflows"),
        {"t2i": ".png", "still": ".png", "lipsync": ".mp4", "i2v": ".mp4"},
        still_extra={"set_image": "set"},
    )
    manifest = build_manifest(project.config, analysis, "lyrics", "style", llm=None)
    manifest.visual_bible.settings = [
        SetDesign(name="Rooftop", description="Rain-slicked rooftop overlooking the city.")
    ]
    for shot in manifest.shots:
        shot.setting = "Rooftop"
    # Give shots distinct angle cues + an explicit override.
    manifest.shots[0].framing = Framing.WIDE
    manifest.shots[0].still_prompt = "Wide establishing shot of the singer on the rooftop."
    manifest.shots[1].plate_angle = "reverse"
    manifest.shots[2].type = ShotType.BROLL
    manifest.shots[2].framing = Framing.CLOSE_UP
    manifest.shots[2].still_prompt = "Close-up of the rain-slicked brick ledge."
    if len(manifest.shots) > 3:
        manifest.shots[3].still_prompt = "90-degree side profile across the rooftop parapet."

    # Lock primary plate and generate 2x2 turnaround via FakeComfy.
    plate_src = tmp_path / "base_plate.png"
    Image.new("RGB", (128, 72), (10, 20, 30)).save(plate_src)
    plates.lock_plate(project, manifest, plate_src, "Rooftop")
    base_plate_rel = manifest.visual_bible.settings[0].plate
    assert base_plate_rel is not None

    fake = FakeComfy(png=sample_png)
    comfy = ComfyClient(project.config.comfy, transport=fake.transport(), sleep=lambda _: None)
    templates = load_templates(project, ["t2i", "still", "lipsync", "i2v"])
    res = plates.generate_turnaround(project, manifest, comfy, templates, "Rooftop", count=1)

    assert res.set_name == "Rooftop"
    assert res.sheet is not None and project.path(res.sheet).is_file()
    assert set(res.panels) == set(PLATE_ANGLE_NAMES)
    # Existing primary scene.plate is preserved.
    assert manifest.visual_bible.settings[0].plate == base_plate_rel
    assert fake.upload_files == ["rooftop.png"]

    # Attach dummy character ref for ShotRunner._plate_for tests.
    ref_img = tmp_path / "char_front.png"
    Image.new("RGB", (64, 64), "white").save(ref_img)
    rel_ref = "assets/reference/base_front.png"
    project.path(rel_ref).parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), "white").save(project.path(rel_ref))
    manifest.character.references["base"] = ReferenceSet(sheet=rel_ref, panels={"front": rel_ref})

    runner = ShotRunner(
        project,
        manifest,
        comfy,
        build_quality_gates(project.config, None),
        RetryPolicy(project.config.retry, project.config.seed, llm=None),
        templates,
        RunBudget(10),
    )
    scene = manifest.visual_bible.settings[0]
    assert runner._plate_for(manifest.shots[0]) == project.path(scene.panels["wide"])
    assert runner._plate_for(manifest.shots[1]) == project.path(scene.panels["reverse"])
    assert runner._plate_for(manifest.shots[2], template=templates["still"]) == project.path(
        scene.panels["detail"]
    )
    if len(manifest.shots) > 3:
        assert runner._plate_for(manifest.shots[3]) == project.path(scene.panels["side"])

    # Fallback to scene.plate if matched panel file is missing on disk.
    project.path(scene.panels["reverse"]).unlink()
    assert runner._plate_for(manifest.shots[1]) == project.path(base_plate_rel)

    # Preflight catches missing explicit shot.plate_angle panel file.
    preflight_res = preflight.check_assets(project, manifest, templates)
    assert any("reverse" in problem for problem in preflight_res.problems)


def test_plate_turnaround_api_and_cli(project, analysis, sample_png, tmp_path) -> None:
    manifest = build_manifest(project.config, analysis, "lyrics", "style", llm=None)
    manifest.visual_bible.settings = [
        SetDesign(name="Studio Stage", description="Dark stage with warm tungsten spots.")
    ]
    project.save_manifest(manifest)

    plate_src = tmp_path / "stage_plate.png"
    Image.new("RGB", (128, 72), "brown").save(plate_src)
    api.lock_plate(project.root, plate_src, "Studio Stage")

    fake = FakeComfy(png=sample_png)
    provider = _FakeProvider(fake, project)
    result = api.generate_plate_turnaround(
        project.root, "Studio Stage", count=1, lock=True, comfy_provider=provider
    )
    assert result.sheet is not None
    assert set(result.panels) == set(PLATE_ANGLE_NAMES)

    runner = CliRunner()
    assets_out = runner.invoke(app, ["assets", "-p", str(project.root)])
    assert assets_out.exit_code == 0
    assert "locked (+ 4 panel(s))" in assets_out.output

    ext_sheet = tmp_path / "custom_turnaround.png"
    ext_sheet.write_bytes(_make_png_bytes(640, 360))
    lock_out = runner.invoke(
        app,
        [
            "plates",
            "lock-turnaround",
            str(ext_sheet),
            "--set",
            "Studio Stage",
            "-p",
            str(project.root),
        ],
    )
    assert lock_out.exit_code == 0
    assert "4 angle panel(s)" in lock_out.output
