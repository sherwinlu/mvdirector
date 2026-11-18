# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""The vision judge: what it is shown, and how its verdict becomes quality-check results."""

from typing import Any

import numpy as np
import pytest

from mvdirector.config import ProjectConfig
from mvdirector.errors import LLMError
from mvdirector.models import (
    AssetKind,
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
    WardrobeState,
)
from mvdirector.qc.common import vision_checks
from mvdirector.qc.faces import Face, IdentityReference
from mvdirector.qc.image import ImageQC
from mvdirector.qc.vision import VisionJudge, VisionVerdict


class FakeLLM:
    def __init__(self, verdict: VisionVerdict | Exception):
        self.verdict = verdict
        self.system = ""
        self.content: list[dict[str, Any]] = []

    def complete_json(self, *, system, content, schema):
        self.system, self.content = system, content
        if isinstance(self.verdict, Exception):
            raise self.verdict
        return self.verdict


class FakeFaces:
    def __init__(self, faces):
        self._faces = faces

    def faces(self, image):
        return self._faces


def _verdict(**overrides):
    base = {"spec_match": 0.9, "artifact_free": 0.9, "character_match": 0.9}
    return VisionVerdict(**{**base, **overrides})


def _shot(shot_type=ShotType.LIPSYNC, wardrobe="bare"):
    return Shot(
        id="s010",
        index=9,
        start=0.0,
        end=4.0,
        type=shot_type,
        framing=Framing.CLOSE_UP,
        section="Bridge",
        setting="Quiet Chamber",
        wardrobe=wardrobe,
        still_prompt="Close-up, chin lifted.",
        motion_prompt="She meets the lens.",
        audio_slice=AudioSlice(path="a.wav", start=0.0, end=4.25, head_handle=0.0),
        num_frames=105,
        still=AssetState(attempts_allowed=4),
        video=AssetState(attempts_allowed=4),
    )


@pytest.fixture
def manifest():
    return Manifest(
        title="t",
        duration=4.0,
        fps=24,
        character=Character(
            appearance="Beauty mark under the outer corner of her LEFT eye.",
            wardrobe=[
                WardrobeState(id="cape", description="Bodysuit and a black cape."),
                WardrobeState(id="bare", description="Bodysuit only. No cape."),
            ],
        ),
        visual_bible=VisualBible(
            style="glossy",
            settings=[SetDesign(name="Quiet Chamber", description="a small charcoal room")],
            rules=["Armor stays brushed silver, never chrome."],
        ),
    )


@pytest.fixture
def png(tmp_path, sample_png):
    path = tmp_path / "frame.png"
    path.write_bytes(sample_png)
    return path


def _kinds(content):
    return [block["type"] for block in content]


def test_judge_sees_reference_portrait_then_the_frame_then_the_spec(manifest, png):
    llm = FakeLLM(_verdict())
    VisionJudge(llm).judge([png], _shot(), manifest, AssetKind.STILL, reference_image=png)
    assert _kinds(llm.content) == ["text", "image", "text", "image", "text"]
    assert llm.content[0]["text"].startswith("REFERENCE of the singer")
    spec = llm.content[-1]["text"]
    assert "LEFT eye" in spec and "Wardrobe state 'bare'" in spec and "No cape." in spec
    assert "Setting: Quiet Chamber - a small charcoal room" in spec
    assert "Section of the song: Bridge" in spec and "never chrome" in spec


def test_judge_sends_sampled_video_frames_in_order(manifest, png):
    llm = FakeLLM(_verdict())
    frames = [np.zeros((16, 16, 3), dtype=np.uint8)] * 3
    VisionJudge(llm).judge(frames, _shot(), manifest, AssetKind.VIDEO, reference_image=png)
    assert _kinds(llm.content) == ["text", "image", "text", "image", "image", "image", "text"]
    assert "3 frames sampled in order" in llm.system and "morphing" in llm.system


def test_judge_skips_reference_when_there_is_none_or_no_singer(manifest, png):
    llm = FakeLLM(_verdict(character_match=None))
    VisionJudge(llm).judge([png], _shot(), manifest, AssetKind.STILL)
    assert _kinds(llm.content) == ["text", "image", "text"]
    VisionJudge(llm).judge([png], _shot(ShotType.BROLL, None), manifest, AssetKind.STILL, png)
    assert _kinds(llm.content) == ["text", "image", "text"]
    assert "No singer should be visible" in llm.content[-1]["text"]


def test_system_prompt_flags_mirrored_features_and_costume_state(manifest, png):
    llm = FakeLLM(_verdict())
    VisionJudge(llm).judge([png], _shot(), manifest, AssetKind.STILL)
    assert "mirror-flipped" in llm.system and "cape" in llm.system


def _checks(verdict, manifest, png, kind=AssetKind.STILL):
    results = vision_checks(
        VisionJudge(FakeLLM(verdict)), [png], _shot(), manifest, kind, 0.30, 0.40, png
    )
    return {c.name: c for c in results}


def test_good_verdict_passes_all_three_checks(manifest, png):
    checks = _checks(_verdict(), manifest, png)
    assert set(checks) == {"vision_spec", "vision_artifacts", "vision_character"}
    assert all(c.status is CheckStatus.PASSED for c in checks.values())


def test_character_mismatch_is_a_hard_failure_with_a_fix_hint(manifest, png):
    verdict = _verdict(
        character_match=0.2,
        issues=["beauty mark is under the right eye", "cape still present"],
        fix="Keep the mark under her left eye and remove the cape.",
    )
    checks = _checks(verdict, manifest, png)
    character = checks["vision_character"]
    assert character.status is CheckStatus.FAILED and character.hard
    assert "right eye" in character.detail and "remove the cape" in character.fix_hint
    assert checks["vision_spec"].status is CheckStatus.PASSED


def test_character_match_none_is_skipped(manifest, png):
    checks = _checks(_verdict(character_match=None), manifest, png)
    assert checks["vision_character"].status is CheckStatus.SKIPPED


def test_llm_error_skips_all_vision_checks(manifest, png):
    checks = _checks(LLMError("api down"), manifest, png)
    assert all(c.status is CheckStatus.SKIPPED for c in checks.values())


def test_disabled_judge_skips_all_three(manifest, png):
    results = vision_checks(None, [png], _shot(), manifest, AssetKind.STILL, 0.3, 0.4)
    assert [c.name for c in results] == ["vision_spec", "vision_artifacts", "vision_character"]
    assert all(c.status is CheckStatus.SKIPPED for c in results)


def test_still_with_wrong_costume_is_rejected_even_when_the_face_matches(manifest, png):
    """Face embeddings cannot see a cape or a mirrored mark; the vision check can."""
    config = ProjectConfig()
    same_face = Face(0.4, np.array([1.0, 0.0]))
    reference = IdentityReference(FakeFaces([same_face]), png)
    verdict = _verdict(character_match=0.1, issues=["cape present"], fix="No cape.")
    gate = ImageQC(config.qc, config.video, FakeFaces([same_face]), VisionJudge(FakeLLM(verdict)))
    report = gate.evaluate(png, _shot(), manifest, reference, reference_image=png)
    identity = next(c for c in report.checks if c.name == "identity")
    assert identity.status is CheckStatus.PASSED
    assert not report.passed and report.has_hard_failure
    assert report.issues == ["vision_character: cape present"]
    assert report.fix_hints == ["No cape."]
