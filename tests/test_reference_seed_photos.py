# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Tests for F11: multi-photo character reference seeding (1-10 angle uploads)."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import pytest
from PIL import Image
from typer.testing import CliRunner

from mvdirector import api
from mvdirector.cli import app
from mvdirector.comfy.client import ComfyClient
from mvdirector.comfy.provider import StaticComfyProvider
from mvdirector.errors import ConfigError
from mvdirector.generation import preflight
from mvdirector.generation.runner import RunBudget, ShotRunner, load_templates
from mvdirector.models import Character, ShotType
from mvdirector.planning import prompts
from mvdirector.planning.planner import build_manifest
from mvdirector.retry import RetryPolicy
from tests.conftest import FakeComfy, offline_gates, write_workflows


class _FakeProvider(StaticComfyProvider):
    def __init__(self, fake: FakeComfy, project) -> None:
        super().__init__()
        self._fake = fake
        self._project = project

    def lease(self, project):
        @contextmanager
        def _cm():
            yield ComfyClient(
                self._project.config.comfy,
                transport=self._fake.transport(),
                sleep=lambda _: None,
            )

        return _cm()


def _make_photos(tmp_path: Path, count: int) -> list[Path]:
    paths: list[Path] = []
    for idx in range(count):
        p = tmp_path / f"photo_{idx + 1}.png"
        # Alternate portrait/landscape to verify ImageOps.contain preserves aspect.
        size = (120, 180) if idx % 2 == 0 else (180, 120)
        Image.new("RGB", size, (20 * idx % 255, 100, 180)).save(p)
        paths.append(p)
    return paths


def test_seed_from_photos_validation_errors(project, analysis, tmp_path, sample_png) -> None:
    manifest = build_manifest(project.config, analysis, "lyrics", "style", llm=None)
    project.save_manifest(manifest)

    with pytest.raises(ConfigError, match="at least 1|1 to 10"):
        api.seed_reference_from_photos(project.root, [])

    too_many = _make_photos(tmp_path, 11)
    with pytest.raises(ConfigError, match="1 to 10|At most 10"):
        api.seed_reference_from_photos(project.root, too_many)

    two = _make_photos(tmp_path, 2)
    with pytest.raises(ConfigError, match="Expected 2 angle"):
        api.seed_reference_from_photos(project.root, two, angles=["front"])
    with pytest.raises(ConfigError, match="Unknown seed photo angle"):
        api.seed_reference_from_photos(project.root, two, angles=["front", "overhead"])
    with pytest.raises(ConfigError, match="Duplicate seed photo angle"):
        api.seed_reference_from_photos(project.root, two, angles=["front", "front"])

    bad = tmp_path / "bad.png"
    bad.write_text("not an image", encoding="utf-8")
    with pytest.raises(ConfigError, match="could not be opened"):
        api.seed_reference_from_photos(project.root, [bad])

    # mode="generate" validates still workflow before touching ComfyUI.
    fake = FakeComfy(png=sample_png)
    provider = _FakeProvider(fake, project)
    project.config.comfy.workflows.pop("still", None)
    (project.root / "project.yaml").write_text(
        __import__("yaml").safe_dump(project.config.model_dump(mode="json"))
    )
    with pytest.raises(ConfigError, match="still"):
        api.seed_reference_from_photos(
            project.root, two[:1], mode="generate", comfy_provider=provider
        )
    assert fake.queued == []


def test_seed_from_photos_direct_mode_multi_photo_and_rear_view(
    project, analysis, tmp_path, sample_png
) -> None:
    manifest = build_manifest(project.config, analysis, "lyrics", "style", llm=None)
    manifest.characters = {
        "lead": Character(name="Lead", appearance="Auburn waves."),
        "co": Character(name="Co", appearance="Short black hair."),
    }
    manifest.character = manifest.characters["lead"]
    for shot in manifest.shots:
        shot.character = "lead"
    project.save_manifest(manifest)

    photos = _make_photos(tmp_path, 6)
    res = api.seed_reference_from_photos(
        project.root,
        photos,
        character="lead",
        mode="direct",
    )
    assert res.mode == "direct"
    assert res.sheet is not None and res.sheet.is_file()
    assert res.reference is not None
    assert set(res.panels) == {"front", "profile", "three_quarter", "face", "back"}
    for p in res.panels.values():
        assert p.is_file()

    reloaded = project.load_manifest()
    assert reloaded.characters["lead"].references["base"].sheet == project.rel(res.sheet)
    # Primary character mirrors onto manifest.character.references.
    assert reloaded.character.references["base"].panels == res.reference.panels

    # Verify ShotRunner uses seeded `back` panel on rear-view shots and preflight passes.
    reloaded.shots[0].type = ShotType.PERFORMANCE
    reloaded.shots[0].still_prompt = "Viewed from behind looking down the street."
    templates = load_templates(project, ["still", "t2i", "lipsync", "i2v"])
    fake = FakeComfy(png=sample_png)
    comfy = ComfyClient(project.config.comfy, transport=fake.transport(), sleep=lambda _: None)
    runner = ShotRunner(
        project,
        reloaded,
        comfy,
        offline_gates(project.config),
        RetryPolicy(project.config.retry, project.config.seed, llm=None),
        templates,
        RunBudget(5),
    )
    refs = runner._still_reference_paths(reloaded.shots[0], plate=None)
    assert refs[0][0] == "reference_image"
    assert refs[0][2] == project.path(res.reference.panels["back"])

    pf = preflight.check_assets(project, reloaded, templates, [reloaded.shots[0].id])
    assert not pf.problems


def test_seed_from_photos_generate_mode_single_and_multislot(
    project, analysis, tmp_path, sample_png
) -> None:
    manifest = build_manifest(project.config, analysis, "lyrics", "style", llm=None)
    manifest.character = Character(name="Singer", appearance="Red leather jacket.")
    project.save_manifest(manifest)

    # 1) 1-slot still workflow + 3 photos -> composites collage and uploads once.
    three = _make_photos(tmp_path, 3)
    fake_1slot = FakeComfy(png=sample_png)
    res_1slot = api.seed_reference_from_photos(
        project.root,
        three,
        mode="generate",
        include_back=True,
        comfy_provider=_FakeProvider(fake_1slot, project),
    )
    assert res_1slot.mode == "generate"
    assert res_1slot.layout == "columns"
    assert len(res_1slot.candidates) == 1
    assert res_1slot.sheet is not None and res_1slot.sheet.is_file()
    assert "back" in res_1slot.panels
    assert len(fake_1slot.upload_files) == 1
    assert fake_1slot.upload_files[0].endswith("_seed_input.png")

    # 2) Multi-slot still workflow (reference_image + reference_image_2 + reference_image_3)
    #    uploads all 3 individual photos without collapsing to a collage.
    write_workflows(
        project.path("workflows"),
        {"t2i": ".png", "still": ".png", "lipsync": ".mp4", "i2v": ".mp4"},
        still_extra={"reference_image_2": "profile", "reference_image_3": "three_quarter"},
    )
    fake_3slot = FakeComfy(png=sample_png)
    res_3slot = api.seed_reference_from_photos(
        project.root,
        three,
        mode="generate",
        lock=False,
        comfy_provider=_FakeProvider(fake_3slot, project),
    )
    assert res_3slot.reference is None
    assert len(res_3slot.candidates) == 1
    # Metadata-free re-encoded copies are uploaded, never the original photo files.
    assert fake_3slot.upload_files == [f"base_seed_photo_{n}.png" for n in (1, 2, 3)]

    # 3) CLI `mvd reference seed` direct happy path with explicit --angle flags.
    runner = CliRunner()
    out = runner.invoke(
        app,
        [
            "reference",
            "seed",
            str(three[0]),
            str(three[1]),
            "--angle",
            "front",
            "--angle",
            "rear",
            "-p",
            str(project.root),
        ],
    )
    assert out.exit_code == 0
    assert "2 panel(s)" in out.output
    reloaded = project.load_manifest()
    assert set(reloaded.character.references["base"].panels) == {"front", "back"}


def test_seed_sheet_prompt_only_requests_back_view_when_layout_has_back_panel(
    project, analysis
) -> None:
    manifest = build_manifest(project.config, analysis, "lyrics", "style", llm=None)
    row_prompt = prompts.seed_sheet_prompt(
        manifest, "row", include_back=True, include_grid_back=False
    )
    assert "rear/back view" not in row_prompt
    assert "3/4 turn" in row_prompt

    grid_no_back = prompts.seed_sheet_prompt(
        manifest, "grid", include_back=True, include_grid_back=False
    )
    assert "rear/back view" not in grid_no_back

    grid_with_back = prompts.seed_sheet_prompt(
        manifest, "grid", include_back=True, include_grid_back=True
    )
    assert "rear/back view" in grid_with_back

    cols_prompt = prompts.seed_sheet_prompt(manifest, "columns", include_back=True)
    assert "rear/back view" in cols_prompt
