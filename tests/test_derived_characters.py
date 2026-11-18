# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Tests for deriving variant/clone character sheets via --from-character and parent_character."""

from __future__ import annotations

import pytest
from PIL import Image
from pydantic import ValidationError
from typer.testing import CliRunner

from mvdirector.cli import app
from mvdirector.comfy.client import ComfyClient
from mvdirector.errors import ConfigError
from mvdirector.generation import reference
from mvdirector.generation.runner import load_templates
from mvdirector.models import Character, WardrobeState
from mvdirector.planning.planner import build_manifest
from tests.conftest import FakeComfy


def test_derived_character_edits_parent_sheet(project, analysis, tmp_path, sample_png) -> None:
    manifest = build_manifest(project.config, analysis, "lyrics", "style", llm=None)
    manifest.characters = {
        "kira": Character(
            name="Kira",
            appearance="22-year-old Korean woman with silver ponytail.",
            wardrobe=[WardrobeState(id="scifi", description="Black techwear with cyan strips.")],
        ),
        "clone": Character(
            name="Masked Clone",
            appearance="Same build and ponytail, wearing a glossy black full-face helmet.",
            parent_character="kira",
            wardrobe=[WardrobeState(id="scifi", description="Black techwear with pink visor.")],
        ),
    }
    manifest.character = manifest.characters["kira"]
    templates = load_templates(project, ["t2i", "still"])

    fake = FakeComfy(png=sample_png)
    comfy = ComfyClient(project.config.comfy, transport=fake.transport(), sleep=lambda _: None)

    # Requires parent's base sheet to be locked first.
    with pytest.raises(ConfigError, match="Lock the base sheet for 'kira'"):
        reference.generate_candidates(
            project, manifest, comfy, templates, None, count=1, character_id="clone"
        )

    sheet_img = tmp_path / "kira.png"
    Image.new("RGB", (2048, 896), "grey").save(sheet_img)
    reference.lock_reference(project, manifest, sheet_img, character_id="kira")

    # Self-derivation is rejected.
    with pytest.raises(ConfigError, match="cannot derive from itself"):
        reference.generate_candidates(
            project,
            manifest,
            comfy,
            templates,
            None,
            count=1,
            character_id="kira",
            from_character="kira",
        )

    candidates = reference.generate_candidates(
        project, manifest, comfy, templates, None, count=1, character_id="clone"
    )
    assert len(candidates) == 1
    assert fake.upload_files == ["kira_scifi_sheet.png"]
    prompt = fake.queued[0]["1"]["inputs"]["text"]
    assert "Edit it to create a variant character sheet" in prompt
    assert "glossy black full-face helmet" in prompt


def test_derived_character_validates_parent_id_and_cli(
    project, analysis, tmp_path, sample_png, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ValidationError):
        Character(appearance="Clone", parent_character="../escape")

    manifest = build_manifest(project.config, analysis, "lyrics", "style", llm=None)
    manifest.characters = {
        "kira": Character(name="Kira", appearance="Silver ponytail."),
        "clone": Character(name="Clone", appearance="Short red bob cut."),
    }
    manifest.character = manifest.characters["kira"]
    sheet_img = tmp_path / "kira.png"
    Image.new("RGB", (2048, 896), "grey").save(sheet_img)
    reference.lock_reference(project, manifest, sheet_img, character_id="kira")
    project.save_manifest(manifest)

    fake = FakeComfy(png=sample_png)
    from mvdirector.comfy import provider as api_mod

    monkeypatch.setattr(
        api_mod,
        "ComfyClient",
        lambda cfg, upload_subfolder=None: ComfyClient(
            cfg, transport=fake.transport(), sleep=lambda _: None
        ),
    )
    res = CliRunner().invoke(
        app,
        [
            "reference",
            "generate",
            "--character",
            "clone",
            "--from-character",
            "kira",
            "--count",
            "1",
            "-p",
            str(project.root),
        ],
    )
    assert res.exit_code == 0
    assert "clone_base_candidate_01.png" in res.output
    assert fake.upload_files == ["kira_base_sheet.png"]
