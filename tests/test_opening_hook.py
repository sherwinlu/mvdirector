# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Tests for Doc 1 Idea 6: Opening-hook priority across config, models, planner, QC, and review."""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf
from PIL import Image
from pydantic import ValidationError

from mvdirector.config import PlanningConfig, ProjectConfig, QCConfig, RetryConfig, ReviewMode
from mvdirector.models import (
    MANIFEST_SCHEMA_VERSION,
    AssetStatus,
    Attempt,
    AudioSlice,
    Character,
    Framing,
    Manifest,
    ProjectMode,
    QCReport,
    RetryStrategy,
    ReviewReason,
    Shot,
    ShotType,
)
from mvdirector.planning.planner import (
    BibleReply,
    SetReply,
    ShotBatchReply,
    build_manifest,
)
from mvdirector.qc.faces import Face, IdentityReference
from mvdirector.qc.image import ImageQC
from mvdirector.qc.video import VideoQC
from mvdirector.review import policy as review
from tests.conftest import SR, vocal_track


class _FakeFaces:
    def __init__(self, faces: list[Face]) -> None:
        self._faces = faces

    def faces(self, image: np.ndarray) -> list[Face]:
        return list(self._faces)


def _shot(
    shot_id: str, start: float, end: float, *, shot_type: ShotType = ShotType.LIPSYNC
) -> Shot:
    return Shot(
        id=shot_id,
        index=0,
        start=start,
        end=end,
        type=shot_type,
        framing=Framing.MEDIUM,
        audio_slice=AudioSlice(path="a.wav", start=start, end=end, head_handle=0.0),
        num_frames=97,
    )


def _pass_shot(shot: Shot, *, score: float = 0.90) -> None:
    shot.still.attempts = [
        Attempt(
            number=1,
            strategy=RetryStrategy.INITIAL,
            seed=1,
            prompt="p",
            path=f"assets/stills/{shot.id}_a01.png",
            qc=QCReport(passed=True, score=score),
        )
    ]
    shot.still.status = AssetStatus.PASSED
    shot.still.selected_attempt = 1
    shot.video.attempts = [
        Attempt(
            number=1,
            strategy=RetryStrategy.INITIAL,
            seed=1,
            prompt="p",
            path=f"assets/videos/{shot.id}_a01.mp4",
            qc=QCReport(passed=True, score=score),
        )
    ]
    shot.video.status = AssetStatus.PASSED
    shot.video.selected_attempt = 1


def test_opening_config_validation_and_schema_version() -> None:
    assert MANIFEST_SCHEMA_VERSION == 1
    assert PlanningConfig().opening_seconds == 8.0
    assert RetryConfig().opening_extra_attempts == 2
    assert QCConfig().opening_threshold_margin == 0.0

    for bad_float in (-0.1, 3600.1, float("nan"), float("inf"), True, False):
        with pytest.raises(ValidationError):
            PlanningConfig(opening_seconds=bad_float)  # type: ignore[arg-type]
        with pytest.raises(ValidationError):
            QCConfig(opening_threshold_margin=bad_float)  # type: ignore[arg-type]

    with pytest.raises(ValidationError):
        QCConfig(opening_threshold_margin=0.31)
    for bad_int in (-1, 21, True, False):
        with pytest.raises(ValidationError):
            RetryConfig(opening_extra_attempts=bad_int)  # type: ignore[arg-type]


def test_shot_is_opening_and_is_opening_at_boundary_behavior() -> None:
    s_zero = _shot("s001", 0.0, 4.0)
    s_inside = _shot("s002", 7.99, 12.0)
    s_edge = _shot("s003", 8.0, 12.0)
    assert s_zero.is_opening is True
    assert s_inside.is_opening is True
    assert s_edge.is_opening is False

    assert s_zero.is_opening_at(0.0) is False
    assert s_zero.is_opening_at(-5.0) is False
    assert s_zero.is_opening_at(float("nan")) is False
    assert s_edge.is_opening_at(10.0) is True


def test_planner_grants_extra_attempts_and_opening_prompt_guidance(analysis) -> None:
    class _RecordingLLM:
        def __init__(self) -> None:
            self.contents: list[str] = []

        def complete_json(self, *, system, content, schema):
            if schema is BibleReply:
                return BibleReply(
                    appearance="Lead",
                    style="film",
                    palette="warm",
                    lighting="dramatic",
                    negative_prompt="blur",
                    settings=[SetReply(name="Alley", description="Neon alley")],
                )
            self.contents.append(str(content))
            return ShotBatchReply(shots=[])

    cfg = ProjectConfig()
    cfg.llm.shots_per_request = 1
    llm = _RecordingLLM()
    manifest = build_manifest(cfg, analysis, "lyrics", "style", llm=llm)
    assert len(manifest.shots) == 3
    # s001 starts at 0.0 (< 8.0s), s002 at 8.0s (>= 8.0s), s003 at 16.0s
    assert manifest.shots[0].still.attempts_allowed == 6
    assert manifest.shots[0].video.attempts_allowed == 6
    assert manifest.shots[1].still.attempts_allowed == 4
    assert manifest.shots[2].video.attempts_allowed == 4

    # Batch 1 (opening) gets opening=true and OPENING HOOK PRIORITY; Batch 2 & 3 do not
    assert "opening=true" in llm.contents[0]
    assert "OPENING HOOK PRIORITY" in llm.contents[0]
    assert "opening=true" not in llm.contents[1]
    assert "OPENING HOOK PRIORITY" not in llm.contents[1]


def test_qc_opening_threshold_margin_raises_opening_gate_and_prevents_score_inversion(
    tmp_path, sample_mp4
) -> None:
    img_path = tmp_path / "still.png"
    arr = np.zeros((720, 1280, 3), dtype=np.uint8)
    arr[::2, ::2] = 255
    Image.fromarray(arr).save(img_path)

    cfg = ProjectConfig()
    cfg.qc.pass_score = 0.60
    # Margin 0.30 pushes identity_hard_min (0.30 + 0.30 = 0.60) above identity_good (0.55);
    # ImageQC/VideoQC must clamp good = max(good, effective_hard_min) instead of inverting.
    cfg.qc.opening_threshold_margin = 0.30
    cfg.qc.image.sharpness_bad = 0.0

    # Embedding similarity 0.45 passes normal identity_hard_min (0.30), fails opening (0.60)
    emb_ref = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    emb_mid = np.array([0.45, float(np.sqrt(1 - 0.45**2)), 0.0], dtype=np.float32)
    faces = _FakeFaces([Face(height_fraction=0.22, embedding=emb_mid)])
    ref = IdentityReference(_FakeFaces([Face(height_fraction=0.22, embedding=emb_ref)]), img_path)

    manifest = Manifest(
        title="t",
        duration=16.0,
        fps=24,
        character=Character(name="Lead", appearance="Singer"),
    )
    opening_shot = _shot("s001", 0.0, 4.0)
    later_shot = _shot("s002", 8.0, 12.0)

    img_qc = ImageQC(cfg.qc, cfg.video, faces, None, opening_seconds=cfg.planning.opening_seconds)
    r_open_fail = img_qc.evaluate(img_path, opening_shot, manifest, ref)
    r_later_pass = img_qc.evaluate(img_path, later_shot, manifest, ref)
    assert r_open_fail.passed is False
    assert any(c.name == "identity" and c.status.value == "failed" for c in r_open_fail.checks)
    assert r_later_pass.passed is True

    # Verify passing score above effective_hard_min (0.65 >= 0.60) is NOT inverted to 0.0
    emb_pass = np.array([0.65, float(np.sqrt(1 - 0.65**2)), 0.0], dtype=np.float32)
    faces_pass = _FakeFaces([Face(height_fraction=0.22, embedding=emb_pass)])
    img_qc_pass = ImageQC(cfg.qc, cfg.video, faces_pass, None, opening_seconds=8.0)
    r_open_pass = img_qc_pass.evaluate(img_path, opening_shot, manifest, ref)
    id_check = next(c for c in r_open_pass.checks if c.name == "identity")
    assert id_check.status.value == "passed"
    assert id_check.score == 1.0

    # VideoQC applies margin only on opening shots as well
    wav = tmp_path / "slice.wav"
    sf.write(wav, vocal_track([(0.0, 4.0)]), SR)
    vid_path = tmp_path / "clip.mp4"
    vid_path.write_bytes(sample_mp4)
    vid_cfg = ProjectConfig()
    vid_cfg.qc.pass_score = 0.70
    vid_cfg.qc.opening_threshold_margin = 0.20  # identity_hard_min 0.25 -> 0.45 on opening
    vid_cfg.qc.video.duration_tolerance = 1.0
    emb_vid = np.array([0.35, float(np.sqrt(1 - 0.35**2)), 0.0], dtype=np.float32)
    vid_faces = _FakeFaces([Face(height_fraction=0.22, embedding=emb_vid)])
    vid_qc = VideoQC(vid_cfg.qc, vid_cfg.video, vid_faces, None, None, opening_seconds=8.0)
    open_vid_shot = _shot("s001", 0.0, 9.0, shot_type=ShotType.PERFORMANCE)
    later_vid_shot = _shot("s002", 8.0, 17.0, shot_type=ShotType.PERFORMANCE)
    v_open = vid_qc.evaluate(vid_path, open_vid_shot, manifest, wav, ref)
    v_later = vid_qc.evaluate(vid_path, later_vid_shot, manifest, wav, ref)
    assert next(c for c in v_open.checks if c.name == "identity").status.value == "failed"
    assert next(c for c in v_later.checks if c.name == "identity").status.value == "passed"


def test_review_policy_flags_opening_in_light_and_stills_only_but_not_strict() -> None:
    cfg = ProjectConfig()
    cfg.review.mode = ReviewMode.LIGHT
    cfg.review.closeup_rate = 0.0
    cfg.review.other_rate = 0.0

    open_shot = _shot("s001", 0.0, 4.0)
    later_shot = _shot("s002", 8.0, 12.0)
    _pass_shot(open_shot)
    _pass_shot(later_shot)
    manifest = Manifest(title="t", duration=12.0, fps=24, shots=[open_shot, later_shot])

    review.refresh_reviews(manifest, cfg)
    assert open_shot.review.reasons == [ReviewReason.OPENING]
    assert open_shot.review.required is True
    assert later_shot.review.reasons == []
    assert review.is_ready(later_shot) is True

    # Mix stills_only mode also flags opening shots in LIGHT mode
    manifest_mix = Manifest(
        title="t",
        mode=ProjectMode.MIX,
        duration=12.0,
        fps=24,
        shots=[open_shot.model_copy(deep=True), later_shot.model_copy(deep=True)],
    )
    cfg.mix.visual_source = "stills"
    for s in manifest_mix.shots:
        s.review = s.review.__class__()
    review.refresh_reviews(manifest_mix, cfg)
    assert manifest_mix.shots[0].review.reasons == [ReviewReason.OPENING]
    assert manifest_mix.shots[1].review.reasons == []

    # opening_seconds = 0 disables opening flagging
    cfg.planning.opening_seconds = 0.0
    manifest_mix.shots[0].review = manifest_mix.shots[0].review.__class__()
    review.refresh_reviews(manifest_mix, cfg)
    assert manifest_mix.shots[0].review.reasons == []

    # STRICT mode keeps exhausted-only contract
    cfg.planning.opening_seconds = 8.0
    cfg.review.mode = ReviewMode.STRICT
    open_shot.review = open_shot.review.__class__()
    review.refresh_reviews(manifest, cfg)
    assert open_shot.review.reasons == []
    assert review.is_ready(open_shot) is True
