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
    assert "Reference-image priority" in llm.system
    assert "ground truth" in llm.system


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


def test_broll_exclusion_violation_like_minute_hand_is_a_hard_qc_failure(manifest, png):
    """Even when sharpness and artifact_free are 1.0, failing vision_spec rejects the still."""
    config = ProjectConfig()
    clock_shot = _shot(ShotType.BROLL, None)
    clock_shot.framing = Framing.MEDIUM
    clock_shot.still_prompt = (
        "Dust motes floating in shafts of light revealing a large silent grandfather clock, "
        "no minute hand, tarnished brass."
    )
    verdict = VisionVerdict(
        spec_match=0.2,
        artifact_free=0.98,
        character_match=None,
        issues=["grandfather clock face has a visible minute hand"],
        fix=(
            "Close-up of the tarnished brass clock dial showing only one short hour hand on a "
            "bare center pin."
        ),
    )
    llm = FakeLLM(verdict)
    gate = ImageQC(config.qc, config.video, FakeFaces([]), VisionJudge(llm))
    report = gate.evaluate(png, clock_shot, manifest, reference=None)
    assert "Must NOT appear in frame (hard fail if visible): minute hand" in llm.content[-1]["text"]
    assert "height-to-width" in llm.system and "no minute hand" in llm.system
    spec = next(c for c in report.checks if c.name == "vision_spec")
    assert spec.status is CheckStatus.FAILED and spec.hard
    assert not report.passed and report.has_hard_failure
    assert "minute hand" in report.issues[0]


def test_vision_specification_distinguishes_solo_listener_and_together_duet():
    duet_manifest = Manifest(
        title="Duet",
        duration=12.0,
        fps=24,
        characters={
            "elias": Character(name="Elias", appearance="Dark-haired baritone."),
            "odalie": Character(name="Odalie", appearance="Soprano with curls."),
        },
        visual_bible=VisualBible(style="35mm film"),
    )

    solo_shot = _shot(ShotType.LIPSYNC, None)
    solo_shot.character = "elias"
    solo_shot.characters = ["elias"]
    solo_shot.singers = ["elias"]
    assert "Vocal delivery: Solo — ONLY Elias is singing." in VisionJudge.specification(
        solo_shot, duet_manifest
    )

    listener_shot = _shot(ShotType.LIPSYNC, None)
    listener_shot.character = "odalie"
    listener_shot.characters = ["odalie", "elias"]
    listener_shot.singers = ["odalie"]
    listener_spec = VisionJudge.specification(listener_shot, duet_manifest)
    assert "ONLY Odalie is singing; Elias is listening with mouth closed." in listener_spec

    together_shot = _shot(ShotType.LIPSYNC, None)
    together_shot.character = "elias"
    together_shot.characters = ["elias", "odalie"]
    together_shot.singers = ["elias", "odalie"]
    together_spec = VisionJudge.specification(together_shot, duet_manifest)
    assert "Simultaneous duet — BOTH Elias and Odalie are singing TOGETHER" in together_spec


def test_vision_system_prompt_flags_overlaid_title_cards_and_credits(manifest, png):
    config = ProjectConfig()
    broll = _shot(ShotType.BROLL, None)
    verdict = VisionVerdict(
        spec_match=0.9,
        artifact_free=0.1,
        character_match=None,
        issues=["centered serif opening credit 'THE WALDREST' overlaid on scene"],
        fix="Keep the frame completely free of overlaid typography or title cards.",
    )
    llm = FakeLLM(verdict)
    gate = ImageQC(config.qc, config.video, FakeFaces([]), VisionJudge(llm))
    report = gate.evaluate(png, broll, manifest, reference=None)
    assert "opening credits" in llm.system and "title cards" in llm.system
    artifacts = next(c for c in report.checks if c.name == "vision_artifacts")
    assert artifacts.status is CheckStatus.FAILED and artifacts.hard
    assert not report.passed


def test_vision_video_scope_and_scoring_calibration(manifest, png):
    verdict = VisionVerdict(spec_match=0.9, artifact_free=0.9, character_match=0.9)

    still_llm = FakeLLM(verdict)
    VisionJudge(still_llm).judge([png], _shot(), manifest, AssetKind.STILL, reference_image=png)
    assert "Scoring calibration:" in still_llm.system
    assert "Video scope:" not in still_llm.system

    video_llm = FakeLLM(verdict)
    frames = [np.zeros((16, 16, 3), dtype=np.uint8) for _ in range(3)]
    VisionJudge(video_llm).judge(frames, _shot(), manifest, AssetKind.VIDEO, reference_image=png)
    assert "Scoring calibration:" in video_llm.system
    assert "Video scope:" in video_llm.system
    assert "already-approved first frame" in video_llm.system

    # Blind-observer mode applies the same video scope and calibration on Pass 2.
    from mvdirector.qc.vision import _BlindObservation

    class _BlindFakeLLM:
        def __init__(self) -> None:
            self.systems: list[str] = []

        def complete_json(self, *, system, content, schema):
            self.systems.append(system)
            if schema is _BlindObservation:
                return _BlindObservation(people_count=1, framing="close_up")
            return verdict

    blind_still = _BlindFakeLLM()
    VisionJudge(blind_still, blind_observer=True).judge(  # type: ignore[arg-type]
        [png], _shot(), manifest, AssetKind.STILL
    )
    assert "Scoring calibration:" in blind_still.systems[1]
    assert "Video scope:" not in blind_still.systems[1]

    blind_video = _BlindFakeLLM()
    VisionJudge(blind_video, blind_observer=True).judge(  # type: ignore[arg-type]
        frames, _shot(), manifest, AssetKind.VIDEO
    )
    assert "Scoring calibration:" in blind_video.systems[1]
    assert "Video scope:" in blind_video.systems[1]
