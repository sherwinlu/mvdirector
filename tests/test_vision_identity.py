# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Vision identity: face-crop comparison by the vision model when there are no embeddings."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from mvdirector.config import ProjectConfig
from mvdirector.errors import LLMUnavailableError
from mvdirector.models import (
    AssetState,
    AudioSlice,
    Character,
    CheckStatus,
    Framing,
    Manifest,
    SetDesign,
    Shot,
    ShotType,
    VisualBible,
)
from mvdirector.qc.common import crop_face, vision_identity_check
from mvdirector.qc.faces import Face, IdentityReference
from mvdirector.qc.image import ImageQC
from mvdirector.qc.video import VideoQC
from mvdirector.qc.vision import FaceIdentityVerdict, VisionJudge, VisionVerdict


class ScriptedLLM:
    """Answers the main vision verdict with a pass and face-identity calls from a script."""

    def __init__(self, identity: list[float | Exception]):
        self.identity = list(identity)
        self.identity_calls: list[list[dict[str, Any]]] = []

    def complete_json(self, *, system, content, schema):
        if schema is FaceIdentityVerdict:
            self.identity_calls.append(content)
            answer = self.identity.pop(0)
            if isinstance(answer, Exception):
                raise answer
            return FaceIdentityVerdict(same_person=answer, reason=f"scored {answer}")
        return VisionVerdict(spec_match=0.9, artifact_free=0.9, character_match=0.9)


class FakeFaces:
    def __init__(self, faces: list[Face]):
        self._faces = faces

    def faces(self, image):
        return self._faces


def _face(height_px: int, *, embedding: np.ndarray | None = None, x: int = 500) -> Face:
    y1 = 100
    return Face(
        height_fraction=height_px / 704,
        embedding=embedding,
        bbox=(x, y1, x + int(height_px * 0.8), y1 + height_px),
    )


def _shot() -> Shot:
    return Shot(
        id="s024",
        index=23,
        start=0.0,
        end=3.0,
        type=ShotType.LIPSYNC,
        framing=Framing.MEDIUM_CLOSE,
        section="Chorus",
        setting="Cabin",
        still_prompt="Medium-close shot of The Witch singing in the cabin.",
        motion_prompt="She sings to the lens.",
        audio_slice=AudioSlice(path="a.wav", start=0.0, end=3.25, head_handle=0.0),
        num_frames=81,
        still=AssetState(attempts_allowed=4),
        video=AssetState(attempts_allowed=4),
    )


@pytest.fixture
def manifest() -> Manifest:
    return Manifest(
        title="t",
        duration=3.0,
        fps=24,
        character=Character(name="The Witch", appearance="Dark braid, amber eyes."),
        visual_bible=VisualBible(
            style="folk horror", settings=[SetDesign(name="Cabin", description="a dark cabin")]
        ),
    )


@pytest.fixture
def still(tmp_path: Path) -> Path:
    path = tmp_path / "still.png"
    Image.new("RGB", (1280, 704), (40, 30, 20)).save(path)
    return path


@pytest.fixture
def ref_faces(tmp_path: Path) -> list[Path]:
    paths = []
    for name in ("face", "front", "three_quarter"):
        path = tmp_path / f"{name}.png"
        Image.new("RGB", (400, 400), (90, 70, 60)).save(path)
        paths.append(path)
    return paths


def _gate(config: ProjectConfig, faces: list[Face], llm: ScriptedLLM) -> ImageQC:
    return ImageQC(config.qc, config.video, FakeFaces(faces), VisionJudge(llm))  # type: ignore[arg-type]


def _identity(report):
    return next(c for c in report.checks if c.name == "identity")


def test_same_person_passes_and_sends_face_panels_then_the_crop(manifest, still, ref_faces):
    config = ProjectConfig()
    llm = ScriptedLLM([0.95])
    faces = [_face(220)]
    report = _gate(config, faces, llm).evaluate(
        still, _shot(), manifest, IdentityReference(FakeFaces(faces), ref_faces)
    )
    identity = _identity(report)
    assert identity.status is CheckStatus.PASSED
    assert identity.metrics["vision_identity"] == pytest.approx(0.95)
    # Face panel and one angled view (not all three), then the crop to judge.
    texts = [b["text"] for b in llm.identity_calls[0] if b["type"] == "text"]
    assert texts == ["REFERENCE face:", "REFERENCE face:", "FACE TO CHECK:"]


def test_different_person_is_a_hard_identity_failure(manifest, still, ref_faces):
    config = ProjectConfig()
    faces = [_face(220)]
    report = _gate(config, faces, ScriptedLLM([0.1])).evaluate(
        still, _shot(), manifest, IdentityReference(FakeFaces(faces), ref_faces)
    )
    identity = _identity(report)
    assert identity.status is CheckStatus.FAILED and identity.hard
    assert "0.10" in identity.detail and "scored 0.1" in identity.detail
    assert report.has_hard_failure and not report.passed


def test_small_face_is_skipped_without_asking_the_model(manifest, still, ref_faces):
    config = ProjectConfig()
    llm = ScriptedLLM([])
    faces = [_face(60)]
    report = _gate(config, faces, llm).evaluate(
        still, _shot(), manifest, IdentityReference(FakeFaces(faces), ref_faces)
    )
    identity = _identity(report)
    assert identity.status is CheckStatus.SKIPPED
    assert "too small" in identity.detail and "60px" in identity.detail
    assert llm.identity_calls == []


def test_embeddings_keep_the_embedding_identity_check(manifest, still, ref_faces):
    config = ProjectConfig()
    llm = ScriptedLLM([])
    faces = [_face(220, embedding=np.array([1.0, 0.0]))]
    report = _gate(config, faces, llm).evaluate(
        still, _shot(), manifest, IdentityReference(FakeFaces(faces), ref_faces)
    )
    identity = _identity(report)
    assert identity.status is CheckStatus.PASSED
    assert "similarity" in identity.metrics and llm.identity_calls == []


def test_can_be_turned_off(manifest, still, ref_faces):
    config = ProjectConfig()
    config.qc.vision_identity = False
    llm = ScriptedLLM([])
    faces = [_face(220)]
    report = _gate(config, faces, llm).evaluate(
        still, _shot(), manifest, IdentityReference(FakeFaces(faces), ref_faces)
    )
    assert _identity(report).status is CheckStatus.SKIPPED
    assert llm.identity_calls == []


def test_duet_partner_does_not_fail_the_check(manifest, still, ref_faces):
    config = ProjectConfig()
    faces = [_face(260, x=200), _face(230, x=800)]  # partner is larger and listed first
    report = _gate(config, faces, ScriptedLLM([0.1, 0.9])).evaluate(
        still, _shot(), manifest, IdentityReference(FakeFaces(faces), ref_faces)
    )
    identity = _identity(report)
    assert identity.status is CheckStatus.PASSED
    assert identity.metrics["vision_identity"] == pytest.approx(0.9)


def test_model_error_skips_instead_of_failing(manifest, still, ref_faces):
    config = ProjectConfig()
    faces = [_face(220)]
    report = _gate(config, faces, ScriptedLLM([LLMUnavailableError("down")])).evaluate(
        still, _shot(), manifest, IdentityReference(FakeFaces(faces), ref_faces)
    )
    identity = _identity(report)
    assert identity.status is CheckStatus.SKIPPED and "errored" in identity.detail


def test_clip_frames_use_the_median_verdict(ref_faces):
    frame = np.zeros((704, 1280, 3), dtype=np.uint8)
    face = _face(200)
    judge = VisionJudge(ScriptedLLM([0.9, 0.2, 0.85]))  # type: ignore[arg-type]
    result = vision_identity_check(
        judge,
        [(frame, [face]), (frame, [face]), (frame, [face])],
        IdentityReference(FakeFaces([face]), ref_faces),
        min_face_px=100,
        hard_min=0.5,
        good=0.8,
    )
    assert result.status is CheckStatus.PASSED  # one turned-away frame is tolerated
    assert result.metrics["vision_identity"] == pytest.approx(0.85)
    assert result.metrics["frames_judged"] == 3


def test_video_samples_interior_frames_with_faces():
    config = ProjectConfig()
    config.qc.video.vision_identity_frames = 2
    gate = VideoQC(config.qc, config.video, FakeFaces([_face(200)]), None, None)
    frames = [np.full((704, 1280, 3), i, dtype=np.uint8) for i in range(10)]
    samples = gate._vision_identity_samples(frames)
    assert [int(f[0, 0, 0]) for f, _ in samples] == [3, 6]


def test_crop_face_pads_and_clamps_to_the_frame():
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    crop = crop_face(image, (0, 0, 40, 40))
    assert crop.shape[:2] == (54, 54)  # 35% pad below/right, clamped at the top-left edge
