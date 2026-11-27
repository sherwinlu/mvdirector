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


def test_crop_face_clamps_negative_positive_oob_and_inverted_bboxes():
    image = np.zeros((704, 1280, 3), dtype=np.uint8)
    assert crop_face(image, (-500, 100, -300, 300)).size == 0
    # Even when padding would spill across the frame border, an off-screen bbox returns empty.
    assert crop_face(image, (-999, -999, -100, -100)).size == 0
    assert crop_face(image, (1350, 100, 1850, 600)).size == 0
    assert crop_face(image, (2000, 100, 2200, 300)).size == 0
    assert crop_face(image, (300, 300, 100, 100)).size == 0
    assert crop_face(image, (100, 100, 100, 250)).size == 0


def test_vision_identity_handles_oob_crops_and_corrupt_references(tmp_path, ref_faces):
    frame = np.zeros((704, 1280, 3), dtype=np.uint8)
    oob_face = Face(height_fraction=200 / 704, bbox=(2000, 100, 2200, 300))
    judge = VisionJudge(ScriptedLLM([]))  # type: ignore[arg-type]

    # Out-of-bounds bbox degrades cleanly to skipped instead of crashing image_block.
    res_oob = vision_identity_check(
        judge,
        [(frame, [oob_face])],
        IdentityReference(FakeFaces([oob_face]), ref_faces),
        min_face_px=100,
        hard_min=0.5,
        good=0.8,
    )
    assert res_oob.status is CheckStatus.SKIPPED
    assert res_oob.detail == "no face available to compare"

    # Empty / corrupt reference files degrade to skipped instead of raising UnidentifiedImageError.
    empty_ref = tmp_path / "empty.png"
    empty_ref.write_bytes(b"")
    bad_ref = tmp_path / "not_an_image.png"
    bad_ref.write_text("definitely not a png")
    valid_face = _face(200)
    res_corrupt = vision_identity_check(
        judge,
        [(frame, [valid_face])],
        IdentityReference(FakeFaces([valid_face]), [empty_ref, bad_ref]),
        min_face_px=100,
        hard_min=0.5,
        good=0.8,
    )
    assert res_corrupt.status is CheckStatus.SKIPPED
    assert "unavailable" in res_corrupt.detail

    # If a valid reference follows a corrupt one, the valid panel is still used.
    llm_ok = ScriptedLLM([0.92])
    res_fallback = vision_identity_check(
        VisionJudge(llm_ok),  # type: ignore[arg-type]
        [(frame, [valid_face])],
        IdentityReference(FakeFaces([valid_face]), [bad_ref, ref_faces[0]]),
        min_face_px=100,
        hard_min=0.5,
        good=0.8,
    )
    assert res_fallback.status is CheckStatus.PASSED
    assert len(llm_ok.identity_calls) == 1


def test_video_identity_default_min_face_px_judges_512px_frames_and_skips_wide(ref_faces):
    config = ProjectConfig()
    assert config.qc.video.vision_identity_min_face_px == 40
    # 512x282 matches analysis_width=512 extraction of a 1280x704 clip.
    frames = [np.zeros((282, 512, 3), dtype=np.uint8) for _ in range(24)]

    def _scaled_face(px: int) -> Face:
        return Face(height_fraction=px / 282, bbox=(200, 40, 200 + int(px * 0.8), 40 + px))

    # CLOSE_UP (32% = 90px), MEDIUM_CLOSE (25% = 70px), MEDIUM (18% = 51px) all get judged.
    for px in (90, 70, 51):
        llm = ScriptedLLM([0.1, 0.1])
        gate = VideoQC(
            config.qc,
            config.video,
            FakeFaces([_scaled_face(px)]),
            None,
            VisionJudge(llm),  # type: ignore[arg-type]
        )
        res = gate._identity_check(frames, 24.0, IdentityReference(FakeFaces([]), ref_faces))
        assert res.status is CheckStatus.FAILED and res.hard
        assert len(llm.identity_calls) == 2

    # WIDE (8% = 22px) stays skipped without calling the VLM.
    llm_wide = ScriptedLLM([])
    gate_wide = VideoQC(
        config.qc,
        config.video,
        FakeFaces([_scaled_face(22)]),
        None,
        VisionJudge(llm_wide),  # type: ignore[arg-type]
    )
    res_wide = gate_wide._identity_check(frames, 24.0, IdentityReference(FakeFaces([]), ref_faces))
    assert res_wide.status is CheckStatus.SKIPPED
    assert "22px" in res_wide.detail and llm_wide.identity_calls == []


def test_video_identity_avoids_redundant_mediapipe_inference_and_insightface_vlm_fallback(
    ref_faces,
):
    frames = [np.zeros((282, 512, 3), dtype=np.uint8) for _ in range(96)]

    class CountingFaces:
        def __init__(self, result: list[Face]) -> None:
            self.calls = 0
            self._result = result

        def faces(self, image):
            self.calls += 1
            return self._result

    # 1. MediaPipe: runs faces() ONLY on vision_identity_frames (2), not on frames[::step] (8+2).
    cfg_mp = ProjectConfig()
    mp_faces = CountingFaces([Face(height_fraction=70 / 282, bbox=(200, 40, 256, 110))])
    llm_mp = ScriptedLLM([0.9, 0.9])
    gate_mp = VideoQC(
        cfg_mp.qc,
        cfg_mp.video,
        mp_faces,
        None,
        VisionJudge(llm_mp),  # type: ignore[arg-type]
    )
    res_mp = gate_mp._identity_check(frames, 24.0, IdentityReference(mp_faces, ref_faces))
    assert res_mp.status is CheckStatus.PASSED
    assert mp_faces.calls == cfg_mp.qc.video.vision_identity_frames

    # 2. InsightFace with no faces on sampled frames: skips without calling VLM.
    cfg_if = ProjectConfig()
    cfg_if.qc.face_backend = "insightface"
    if_faces = CountingFaces([])
    llm_if = ScriptedLLM([])
    gate_if = VideoQC(
        cfg_if.qc,
        cfg_if.video,
        if_faces,
        None,
        VisionJudge(llm_if),  # type: ignore[arg-type]
    )
    res_if = gate_if._identity_check(frames, 24.0, IdentityReference(if_faces, ref_faces))
    assert res_if.status is CheckStatus.SKIPPED
    assert llm_if.identity_calls == []


def test_borderline_passing_vision_identity_is_floored_at_pass_score(manifest, still, ref_faces):
    config = ProjectConfig()
    # Clean sharp still; vision returns 0.80, identity returns exact hard_min=0.50.
    rng = np.random.default_rng(7)
    Image.fromarray(rng.integers(0, 256, (704, 1280, 3), dtype=np.uint8)).save(still)

    class BorderlineLLM:
        def complete_json(self, *, system, content, schema):
            if schema is FaceIdentityVerdict:
                return FaceIdentityVerdict(same_person=0.50, reason="borderline match")
            return VisionVerdict(spec_match=0.80, artifact_free=0.80, character_match=0.80)

    faces = [_face(220)]
    report = ImageQC(
        config.qc,
        config.video,
        FakeFaces(faces),
        VisionJudge(BorderlineLLM()),  # type: ignore[arg-type]
    ).evaluate(still, _shot(), manifest, IdentityReference(FakeFaces(faces), ref_faces))
    identity = _identity(report)
    assert identity.status is CheckStatus.PASSED
    assert identity.score == pytest.approx(config.qc.pass_score)
    assert report.passed


def test_config_validates_vision_identity_thresholds_and_pass_score():
    from pydantic import ValidationError

    from mvdirector.config import ImageQCConfig, QCConfig, VideoQCConfig

    with pytest.raises(ValidationError):
        ImageQCConfig(vision_identity_hard_min=0.80, vision_identity_good=0.50)
    with pytest.raises(ValidationError):
        VideoQCConfig(vision_identity_hard_min=0.80, vision_identity_good=0.50)
    for bad in (-0.5, 1.5, float("nan"), float("inf")):
        with pytest.raises(ValidationError):
            QCConfig(pass_score=bad)


def test_vision_identity_sorts_unsorted_faces_and_avoids_zero_px_message(ref_faces):
    frame = np.zeros((704, 1280, 3), dtype=np.uint8)
    # Largest face (300px) is listed last; must be selected ahead of smaller faces.
    faces = [_face(110, x=100), _face(120, x=400), _face(300, x=700)]
    llm = ScriptedLLM([0.95])
    res = vision_identity_check(
        VisionJudge(llm),  # type: ignore[arg-type]
        [(frame, faces)],
        IdentityReference(FakeFaces(faces), ref_faces),
        min_face_px=100,
        hard_min=0.5,
        good=0.8,
    )
    assert res.status is CheckStatus.PASSED
    # Short-circuited on the first (largest) face because 0.95 >= 0.80.
    assert len(llm.identity_calls) == 1

    # Empty samples or bbox=None must not report "0px tall".
    res_empty = vision_identity_check(
        VisionJudge(ScriptedLLM([])),  # type: ignore[arg-type]
        [],
        IdentityReference(FakeFaces([]), ref_faces),
        min_face_px=100,
        hard_min=0.5,
        good=0.8,
    )
    assert res_empty.status is CheckStatus.SKIPPED
    assert res_empty.detail == "no face available to compare"

    res_no_bbox = vision_identity_check(
        VisionJudge(ScriptedLLM([])),  # type: ignore[arg-type]
        [(frame, [Face(height_fraction=0.3, bbox=None)])],
        IdentityReference(FakeFaces([]), ref_faces),
        min_face_px=100,
        hard_min=0.5,
        good=0.8,
    )
    assert res_no_bbox.status is CheckStatus.SKIPPED
    assert res_no_bbox.detail == "no face available to compare"


def test_duet_short_circuits_and_handles_partial_errors_safely(ref_faces):
    frame = np.zeros((704, 1280, 3), dtype=np.uint8)
    faces = [_face(260, x=200), _face(230, x=800)]
    ref = IdentityReference(FakeFaces(faces), ref_faces)

    # 1. Face 1 >= good (0.98 >= 0.80): short-circuits after 1 VLM call.
    llm_fast = ScriptedLLM([0.98, 0.1])
    res_fast = vision_identity_check(
        VisionJudge(llm_fast),  # type: ignore[arg-type]
        [(frame, faces)],
        ref,
        min_face_px=100,
        hard_min=0.5,
        good=0.8,
    )
    assert res_fast.status is CheckStatus.PASSED
    assert len(llm_fast.identity_calls) == 1

    # 2. Face 1 passes below good (0.70), face 2 errors: retains face 1's passing verdict.
    llm_partial_pass = ScriptedLLM([0.70, LLMUnavailableError("transient")])
    res_partial_pass = vision_identity_check(
        VisionJudge(llm_partial_pass),  # type: ignore[arg-type]
        [(frame, faces)],
        ref,
        min_face_px=100,
        hard_min=0.5,
        good=0.8,
    )
    assert res_partial_pass.status is CheckStatus.PASSED
    assert res_partial_pass.metrics["vision_identity"] == pytest.approx(0.70)

    # 3. Face 1 is partner (0.10 < hard_min), face 2 errors: skips instead of falsely hard-failing.
    llm_partner_err = ScriptedLLM([0.10, LLMUnavailableError("transient")])
    res_partner_err = vision_identity_check(
        VisionJudge(llm_partner_err),  # type: ignore[arg-type]
        [(frame, faces)],
        ref,
        min_face_px=100,
        hard_min=0.5,
        good=0.8,
    )
    assert res_partner_err.status is CheckStatus.SKIPPED
    assert "errored" in res_partner_err.detail


def test_vision_judge_breaker_thread_safe_and_logs_disable_warning_once(ref_faces, caplog):
    from concurrent.futures import ThreadPoolExecutor

    frame = np.zeros((704, 1280, 3), dtype=np.uint8)
    face = _face(220)
    ref = IdentityReference(FakeFaces([face]), ref_faces)

    class AlwaysDownLLM:
        def complete_json(self, *, system, content, schema):
            raise LLMUnavailableError("backend unreachable")

    judge = VisionJudge(AlwaysDownLLM())  # type: ignore[arg-type]

    # Direct concurrent breaker transitions: record_unavailable returns True strictly once.
    with ThreadPoolExecutor(max_workers=16) as pool:
        transitions = list(pool.map(lambda i: judge.record_unavailable(f"err-{i}"), range(32)))
    assert transitions.count(True) == 1
    assert judge.disabled_reason is not None

    # End-to-end through vision_identity_check: logs run-wide disable warning once.
    judge2 = VisionJudge(AlwaysDownLLM())  # type: ignore[arg-type]
    with caplog.at_level("WARNING", logger="mvdirector.qc.common"):
        for _ in range(5):
            res = vision_identity_check(
                judge2,
                [(frame, [face])],
                ref,
                min_face_px=100,
                hard_min=0.5,
                good=0.8,
            )
            assert res.status is CheckStatus.SKIPPED
    disable_warnings = [
        r for r in caplog.records if "Vision checks are off for the rest of this run" in r.message
    ]
    assert len(disable_warnings) == 1


def test_face_identity_verdict_rejects_booleans_and_bounds_reason():
    from pydantic import ValidationError

    for bad in (True, False, float("nan"), float("inf"), -0.1, 1.1):
        with pytest.raises(ValidationError):
            FaceIdentityVerdict.model_validate({"same_person": bad, "reason": "x"})

    v_int = FaceIdentityVerdict.model_validate({"same_person": 1, "reason": "  ok\n  match  "})
    assert v_int.same_person == 1.0 and v_int.reason == "ok match"

    v_long = FaceIdentityVerdict.model_validate({"same_person": 0.2, "reason": "a" * 100_000})
    assert len(v_long.reason) <= 500
    assert v_long.reason.endswith("…")


def test_quality_gates_check_summary_reports_mediapipe_vision_identity():
    from mvdirector.qc.factory import QualityGates

    config = ProjectConfig()
    dummy_img = ImageQC(config.qc, config.video, None, None)
    dummy_vid = VideoQC(config.qc, config.video, None, None, None)
    faces = FakeFaces([])

    # MediaPipe + vision_enabled + vision_identity_enabled -> identity is active!
    gates_on = QualityGates(
        image=dummy_img,
        video=dummy_vid,
        face_analyzer=faces,
        vision_enabled=True,
        vision_identity_enabled=True,
        face_backend="mediapipe",
    )
    active, skipped_list = gates_on.check_summary()
    assert "identity" in active
    assert not any(name == "identity" for name, _ in skipped_list)

    # MediaPipe + vision_identity_enabled=False -> identity is skipped with clear reason.
    gates_off = QualityGates(
        image=dummy_img,
        video=dummy_vid,
        face_analyzer=faces,
        vision_enabled=True,
        vision_identity_enabled=False,
        face_backend="mediapipe",
    )
    active_off, skipped_off = gates_off.check_summary()
    assert "identity" not in active_off
    assert "vision_identity is false" in dict(skipped_off)["identity"]


def test_vision_identity_handles_decompression_bomb_and_unencodable_frame(tmp_path, ref_faces):
    import struct
    import zlib

    def _chunk(kind: bytes, payload: bytes) -> bytes:
        crc = zlib.crc32(kind + payload) & 0xFFFFFFFF
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", crc)

    # Minimal valid-header PNG declaring 20000x20000 pixels (exceeds Pillow's bomb threshold).
    ihdr = struct.pack(">IIBBBBB", 20_000, 20_000, 8, 2, 0, 0, 0)
    bomb_png = tmp_path / "bomb.png"
    bomb_png.write_bytes(b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", ihdr) + _chunk(b"IEND", b""))

    frame = np.zeros((704, 1280, 3), dtype=np.uint8)
    face = _face(200)
    res_bomb = vision_identity_check(
        VisionJudge(ScriptedLLM([])),  # type: ignore[arg-type]
        [(frame, [face])],
        IdentityReference(FakeFaces([face]), [bomb_png]),
        min_face_px=100,
        hard_min=0.5,
        good=0.8,
    )
    assert res_bomb.status is CheckStatus.SKIPPED
    assert "unavailable" in res_bomb.detail

    # Unexpected non-uint8 frame dtype fails encoding with TypeError -> skipped with neutral detail.
    bad_dtype_frame = np.zeros((704, 1280, 3), dtype=np.float64)
    res_dtype = vision_identity_check(
        VisionJudge(ScriptedLLM([])),  # type: ignore[arg-type]
        [(bad_dtype_frame, [face])],
        IdentityReference(FakeFaces([face]), ref_faces),
        min_face_px=100,
        hard_min=0.5,
        good=0.8,
    )
    assert res_dtype.status is CheckStatus.SKIPPED
    assert "vision identity image unreadable" in res_dtype.detail

    # DecompressionBombError raised during judge encoding also degrades cleanly to skipped.
    class BombOnEncodeJudge(VisionJudge):
        def judge_face_identity(self, face_crop, reference_faces):
            raise Image.DecompressionBombError("oversized image")

    res_encode_bomb = vision_identity_check(
        BombOnEncodeJudge(ScriptedLLM([])),  # type: ignore[arg-type]
        [(frame, [face])],
        IdentityReference(FakeFaces([face]), ref_faces),
        min_face_px=100,
        hard_min=0.5,
        good=0.8,
    )
    assert res_encode_bomb.status is CheckStatus.SKIPPED
    assert "vision identity image unreadable" in res_encode_bomb.detail


def test_vision_identity_skips_incomplete_multiframe_clip_only_when_missing_could_pass(ref_faces):
    frame = np.zeros((704, 1280, 3), dtype=np.uint8)
    face = _face(200)
    ref = IdentityReference(FakeFaces([face]), ref_faces)

    # 1. 2-frame clip: frame 1 = 0.20, frame 2 errors -> optimistic median 0.60 >= 0.50 -> skipped.
    res_2f = vision_identity_check(
        VisionJudge(ScriptedLLM([0.20, LLMUnavailableError("timeout")])),  # type: ignore[arg-type]
        [(frame, [face]), (frame, [face])],
        ref,
        min_face_px=100,
        hard_min=0.5,
        good=0.8,
    )
    assert res_2f.status is CheckStatus.SKIPPED
    assert "errored" in res_2f.detail

    # 2. 2-frame clip: frame 1 = 0.20, judge disabled before frame 2 -> skipped with disable reason.
    class DisableAfterFirstLLM:
        def __init__(self, target_judge: list[VisionJudge]) -> None:
            self._judge_ref = target_judge

        def complete_json(self, *, system, content, schema):
            self._judge_ref[0].disable("breaker tripped concurrently")
            return FaceIdentityVerdict(same_person=0.20, reason="turned away")

    holder: list[VisionJudge] = []
    concurrent_judge = VisionJudge(DisableAfterFirstLLM(holder))  # type: ignore[arg-type]
    holder.append(concurrent_judge)
    res_disabled = vision_identity_check(
        concurrent_judge,
        [(frame, [face]), (frame, [face])],
        ref,
        min_face_px=100,
        hard_min=0.5,
        good=0.8,
    )
    assert res_disabled.status is CheckStatus.SKIPPED
    assert res_disabled.detail == "breaker tripped concurrently"

    # 3. 3-frame clip: 0.10, 0.15, error -> optimistic median [0.10, 0.15, 1.0] = 0.15 -> fails.
    res_3f = vision_identity_check(
        VisionJudge(ScriptedLLM([0.10, 0.15, LLMUnavailableError("timeout")])),  # type: ignore[arg-type]
        [(frame, [face]), (frame, [face]), (frame, [face])],
        ref,
        min_face_px=100,
        hard_min=0.5,
        good=0.8,
    )
    assert res_3f.status is CheckStatus.FAILED and res_3f.hard
    assert res_3f.metrics["frames_judged"] == 2.0


def test_floor_passing_score_rejects_non_finite_thresholds():
    from mvdirector.qc.scoring import floor_passing_score

    for args in (
        (0.6, float("nan"), 0.8, 0.7),
        (0.6, 0.5, float("nan"), 0.7),
        (0.6, float("inf"), 0.8, 0.7),
        (0.6, float("-inf"), 0.8, 0.7),
        (0.6, 0.5, float("inf"), 0.7),
        (float("nan"), 0.5, 0.8, 0.7),
        (0.6, 0.5, 0.8, float("nan")),
    ):
        assert floor_passing_score(*args) == 0.0
