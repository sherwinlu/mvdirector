# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Regression tests for QA/security audit findings on photo seeding and draft grids."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import httpx
import pytest
from PIL import Image

from mvdirector import api
from mvdirector.comfy.client import ComfyClient
from mvdirector.comfy.provider import StaticComfyProvider
from mvdirector.errors import ComfyJobError, MVDError
from mvdirector.models import AssetStatus, Review, ReviewDecision, Shot
from mvdirector.planning.planner import build_manifest
from tests.conftest import FakeComfy, write_workflows

_PRIVATE_MARKER = "PrivatePhoneMaker-7f3a"


class _Provider(StaticComfyProvider):
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


class _CapturingComfy(FakeComfy):
    """FakeComfy that keeps the raw multipart body of every upload."""

    def __init__(self, png: bytes = b"", mp4: bytes = b"") -> None:
        super().__init__(png=png, mp4=mp4)
        self.upload_bodies: list[bytes] = []

    def _handle(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/upload/image":
            self.upload_bodies.append(bytes(request.content))
        return super()._handle(request)


class _FailAfterFirstJob(FakeComfy):
    """FakeComfy whose second and later queued jobs fail with an execution error."""

    def _handle(self, request: httpx.Request) -> httpx.Response:
        if request.url.path.startswith("/history/"):
            prompt_id = request.url.path.rsplit("/", 1)[1]
            if int(prompt_id[1:]) >= 2:
                self.fail_next_jobs = 1
        return super()._handle(request)


def _rotated_phone_photo(path: Path) -> Path:
    """A 200x100 landscape JPEG whose EXIF says 'rotate 90° CW' and carries a camera Make tag."""
    img = Image.new("RGB", (200, 100), (200, 40, 40))
    exif = img.getexif()
    exif[0x0112] = 6  # Orientation: rotate 90 CW on display -> 100x200 portrait
    exif[0x010F] = _PRIVATE_MARKER  # Make
    img.save(path, format="JPEG", exif=exif.tobytes())
    return path


def test_seed_direct_mode_honors_exif_orientation(project, analysis, tmp_path) -> None:
    manifest = build_manifest(project.config, analysis, "lyrics", "style", llm=None)
    project.save_manifest(manifest)
    photo = _rotated_phone_photo(tmp_path / "phone.jpg")

    result = api.seed_reference_from_photos(project.root, [photo], angles=["front"])

    with Image.open(result.panels["front"]) as panel:
        assert panel.size == (100, 200), "phone photo must be stored upright (EXIF transposed)"


def test_seed_generate_mode_uploads_metadata_stripped_photos(
    project, analysis, tmp_path, sample_png
) -> None:
    manifest = build_manifest(project.config, analysis, "lyrics", "style", llm=None)
    project.save_manifest(manifest)
    write_workflows(
        project.path("workflows"),
        {"t2i": ".png", "still": ".png", "lipsync": ".mp4", "i2v": ".mp4"},
        still_extra={"reference_image_2": "profile"},
    )
    photos = [
        _rotated_phone_photo(tmp_path / "a.jpg"),
        _rotated_phone_photo(tmp_path / "b.jpg"),
    ]
    fake = _CapturingComfy(png=sample_png)

    api.seed_reference_from_photos(
        project.root, photos, mode="generate", lock=False, comfy_provider=_Provider(fake, project)
    )

    assert len(fake.upload_bodies) == 2
    for body in fake.upload_bodies:
        assert _PRIVATE_MARKER.encode() not in body, "EXIF metadata must not leave the machine"
    assert not any(name.endswith(".jpg") for name in fake.upload_files)


def _two_section_project(project, analysis) -> None:
    manifest = build_manifest(project.config, analysis, "lyrics", "style", llm=None)
    base: Shot = manifest.shots[0]
    shots: list[Shot] = []
    for idx, sec in enumerate(["Chorus", "Chorus", "Verse", "Verse"]):
        s = base.model_copy(deep=True)
        s.id = f"s{idx + 1:03d}"
        s.index = idx
        s.start, s.end = float(idx * 2), float((idx + 1) * 2)
        s.audio_slice.start, s.audio_slice.end = s.start, s.end
        s.audio_slice.head_handle = 0.0
        s.section = sec
        shots.append(s)
    manifest.shots = shots
    project.save_manifest(manifest)


def test_draft_grid_clears_stale_review_decision(project, analysis, sample_png) -> None:
    _two_section_project(project, analysis)
    manifest = project.load_manifest()
    for shot in manifest.shots:
        shot.review = Review(decision=ReviewDecision.APPROVED, note="looked great")
    project.save_manifest(manifest)

    api.generate_draft_grid(
        project.root,
        section="Chorus",
        mode="stills",
        comfy_provider=_Provider(FakeComfy(png=sample_png), project),
    )

    reloaded = project.load_manifest()
    for shot in reloaded.shots[:2]:
        assert shot.still.status is AssetStatus.PASSED
        assert shot.review.decision is None, "an approval must not carry over to a new still"
    assert reloaded.shots[2].review.decision is ReviewDecision.APPROVED


def test_draft_grid_persists_completed_sections_when_a_later_section_fails(
    project, analysis, sample_png
) -> None:
    _two_section_project(project, analysis)
    fake = _FailAfterFirstJob(png=sample_png)

    with pytest.raises((ComfyJobError, MVDError)):
        api.generate_draft_grid(
            project.root, mode="stills", comfy_provider=_Provider(fake, project)
        )

    reloaded = project.load_manifest()
    chorus = [s for s in reloaded.shots if s.section == "Chorus"]
    assert all(len(s.still.attempts) == 1 for s in chorus), "finished section must be recorded"
    assert all(project.path(s.still.attempts[0].path).is_file() for s in chorus)
    assert [g.section for g in reloaded.draft_grids] == ["Chorus"]


def test_replacing_a_sketch_drops_guidance_written_for_the_old_one(
    project, analysis, tmp_path
) -> None:
    manifest = build_manifest(project.config, analysis, "lyrics", "style", llm=None)
    project.save_manifest(manifest)
    shot_id = manifest.shots[0].id
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    Image.new("RGB", (128, 72), "grey").save(first)
    Image.new("RGB", (128, 72), "white").save(second)

    api.attach_shot_sketch(project.root, shot_id, first)
    with project.edit_manifest() as locked:
        locked.shot(shot_id).sketch_guidance = "Singer lower-left, doorway on the right."

    api.attach_shot_sketch(project.root, shot_id, second)

    assert project.load_manifest().shot(shot_id).sketch_guidance == ""


def test_lock_plate_turnaround_accepts_string_paths(project, analysis, tmp_path) -> None:
    from mvdirector.models import SetDesign

    manifest = build_manifest(project.config, analysis, "lyrics", "style", llm=None)
    manifest.visual_bible.settings = [SetDesign(name="Alley", description="Brick alley.")]
    project.save_manifest(manifest)
    sheet = tmp_path / "turnaround.png"
    Image.new("RGB", (256, 256), "grey").save(sheet)

    scene = api.lock_plate_turnaround(project.root, str(sheet), "Alley")  # type: ignore[arg-type]

    assert set(scene.panels) == {"wide", "reverse", "side", "detail"}
