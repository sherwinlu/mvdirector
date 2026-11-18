# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Tests for style-seeded set plates (--style-ref) and nested set-in-set plates (--embed-set)."""

from __future__ import annotations

import pytest
from PIL import Image
from typer.testing import CliRunner

from mvdirector.cli import app
from mvdirector.comfy.client import ComfyClient
from mvdirector.errors import ConfigError
from mvdirector.generation import plates
from mvdirector.generation.runner import load_templates
from mvdirector.models import SetDesign
from mvdirector.planning.planner import build_manifest
from tests.conftest import FakeComfy


def test_plate_style_ref_and_embed_set(project, analysis, tmp_path, sample_png) -> None:
    manifest = build_manifest(project.config, analysis, "lyrics", "style", llm=None)
    manifest.visual_bible.settings = [
        SetDesign(name="Neon City", description="glowing pagoda skyline and round stage"),
        SetDesign(
            name="Retro Arcade",
            description="sunny 1980s arcade with Neon City inside the cabinet glass",
        ),
    ]
    templates = load_templates(project, ["t2i", "still"])

    # Invalid/non-image --style-ref fails cleanly before contacting ComfyUI.
    bad_ref = tmp_path / "not_an_image.txt"
    bad_ref.write_text("hello", encoding="utf-8")
    fake_bad = FakeComfy(png=sample_png)
    comfy_bad = ComfyClient(
        project.config.comfy, transport=fake_bad.transport(), sleep=lambda _: None
    )
    with pytest.raises(ConfigError, match="could not be opened"):
        plates.generate_candidates(
            project, manifest, comfy_bad, templates, "Neon City", count=1, style_ref=bad_ref
        )
    assert fake_bad.queued == []

    # Valid --style-ref uploads the style seed once and uses PLATE_STYLE_TEMPLATE.
    style_img = tmp_path / "moodboard.png"
    Image.new("RGB", (64, 36), "teal").save(style_img)
    fake_style = FakeComfy(png=sample_png)
    comfy_style = ComfyClient(
        project.config.comfy, transport=fake_style.transport(), sleep=lambda _: None
    )
    styled = plates.generate_candidates(
        project, manifest, comfy_style, templates, "Neon City", count=1, style_ref=style_img
    )
    assert len(styled["Neon City"]) == 1
    assert fake_style.upload_files == ["moodboard.png"]
    assert (
        "matching the visual style, color palette, and mood"
        in fake_style.queued[0]["1"]["inputs"]["text"]
    )

    # --embed-set requires --set, rejects self-embedding, and requires a locked embedded plate.
    with pytest.raises(ConfigError, match="Pass --set"):
        plates.generate_candidates(
            project, manifest, comfy_style, templates, None, count=1, embed_set="Neon City"
        )
    with pytest.raises(ConfigError, match="cannot embed itself"):
        plates.generate_candidates(
            project,
            manifest,
            comfy_style,
            templates,
            "Neon City",
            count=1,
            embed_set="Neon City",
        )
    with pytest.raises(ConfigError, match="Lock a plate for 'Neon City'"):
        plates.generate_candidates(
            project,
            manifest,
            comfy_style,
            templates,
            "Retro Arcade",
            count=1,
            embed_set="Neon City",
        )

    # Lock Neon City, then embed it into Retro Arcade without locking perspective.
    city_img = tmp_path / "city.png"
    Image.new("RGB", (64, 36), "navy").save(city_img)
    plates.lock_plate(project, manifest, city_img, "Neon City")

    fake_embed = FakeComfy(png=sample_png)
    comfy_embed = ComfyClient(
        project.config.comfy, transport=fake_embed.transport(), sleep=lambda _: None
    )
    embedded = plates.generate_candidates(
        project,
        manifest,
        comfy_embed,
        templates,
        "Retro Arcade",
        count=1,
        embed_set="Neon City",
    )
    assert len(embedded["Retro Arcade"]) == 1
    assert fake_embed.upload_files == ["neon_city.png"]
    prompt = fake_embed.queued[0]["1"]["inputs"]["text"]
    assert "in its own room layout, camera angle, and lighting" in prompt
    assert "incorporating the scene from the reference image ('Neon City')" in prompt


def test_plates_generate_cli_rejects_both_style_ref_and_embed_set(
    project, analysis, tmp_path, sample_png
) -> None:
    manifest = build_manifest(project.config, analysis, "lyrics", "style", llm=None)
    manifest.visual_bible.settings = [
        SetDesign(name="Neon City", description="cyber skyline"),
        SetDesign(name="Retro Arcade", description="arcade room"),
    ]
    project.save_manifest(manifest)
    style_img = tmp_path / "style.png"
    style_img.write_bytes(sample_png)

    res = CliRunner().invoke(
        app,
        [
            "plates",
            "generate",
            "--set",
            "Retro Arcade",
            "--style-ref",
            str(style_img),
            "--embed-set",
            "Neon City",
            "-p",
            str(project.root),
        ],
    )
    assert res.exit_code == 1
    assert "not multiple" in res.output
