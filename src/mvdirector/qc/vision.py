# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Model-based visual judgement: does the frame match the shot, the character and the rules?"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from mvdirector.llm import LLMClient, image_block, text_block
from mvdirector.models import AssetKind, Framing, Manifest, Shot, ShotType
from mvdirector.planning.prompts import extract_negative_exclusions, join_names

_SYSTEM = """You are the quality-control supervisor on an AI-generated music video. You are
shown {subject}. Judge it strictly against the shot specification.

Score from 0 to 1:
- spec_match: framing, camera distance, setting, lighting and action match the specification, and
  the production rules that can be judged from {frames_word} are followed.
  * Explicit exclusions & absences: verify every 'no ...', 'without ...', 'missing ...' or
    'Must NOT appear' constraint in the specification (for example 'no minute hand', 'no cape',
    'no people'). If a forbidden object, clock hand, garment or person is visible anyway—or if
    any person or face appears when no singer should be visible—score spec_match <= 0.2 and name
    the offending element in 'issues'.
  * Environmental scale: people and props must sit at realistic physical scale relative to doors,
    windows, furniture and buildings (a person near a house or manor must never appear giant or
    nearly as tall as the architecture; score spec_match <= 0.2 if scale is implausible, or if a
    wide shot is filled top-to-bottom by a foreground figure).
  * Vocal delivery & cast staging: verify 'Vocal delivery' when present—on simultaneous duet
    shots, BOTH named vocalists must appear in frame with unobstructed mouths singing together; on
    solo two-shots with a listener, ONLY the named solo vocalist sings while the listener's mouth
    stays closed (score spec_match <= 0.2 if a co-singer is missing or the wrong person is singing).
  * Reference-image priority: when a character reference image is provided, treat the reference
    image as the visual ground truth for character appearance, garment construction/coverage
    (neckline, collar, undershirt/turtleneck, sleeves, coat), and accessory placement; do not
    penalize spec_match for minor text-vs-reference discrepancies (such as an anatomical
    left/right side difference or a skin detail covered by the reference garment) when the shot
    faithfully matches the reference image.
- artifact_free: no malformed hands or fingers, extra or missing limbs, warped faces,
  melted objects, garbled text, watermarks{video_artifacts}.
  * Overlaid text & title cards: any overlaid typography, opening credits, title cards,
    subtitles, captions, logos, or watermarks—even clean, legible, or stylized serif text
    centered or in a corner—MUST score artifact_free <= 0.2 and be named in 'issues' unless
    the specification explicitly asks for text. Lettering that physically belongs to the scene
    (a shop sign, a book cover, a print on clothing) is not an overlay; penalize it only when
    it is garbled.
  * Body proportions & height-to-width ratios: check human head-to-body, torso-to-leg and
    height-to-width proportions carefully. Vertically squashed or compressed bodies, dwarf or
    stubby legs beneath a wide coat or torso, oversized heads, or stiff front-facing reference
    cutout poses pasted onto a scene MUST score artifact_free <= 0.3.
- character_match: the singer matches the character description and, when a reference
  sheet or portrait is given, looks like the same person. A sheet shows the same person from
  several angles. Judge only details large enough to resolve at the stated framing. Check face
  shape, skin tone, hair cut and length, makeup, build and the costume for this shot's wardrobe
  state (for example a cape must be present or absent exactly as specified). When a reference
  sheet or portrait is given, treat it as the visual ground truth for both facial identity and
  wardrobe construction/coverage—neckline, collar, undershirt or turtleneck vs. bare skin,
  sleeves, coat, and accessories must match the reference sheet for this wardrobe state. Score
  character_match <= 0.4 and name the mismatch in 'issues' if a garment layer shown in the
  reference sheet is dropped or altered (for example, replacing a high-neck turtleneck or
  undershirt with bare skin or an open neckline) even when the text description did not repeat
  that layer. For asymmetric identifying marks or accessories, treat the reference image as ground
  truth for anatomical left/right placement; flag a mirror-flipped face, mark, or accessory only
  when it contradicts the reference image (or the text description when no reference image is
  given).
  Use null if the specification says no singer should be visible.

List each concrete problem in 'issues' (empty if none). In 'fix', give one short,
practical instruction for the prompt that would avoid the worst problem; when a 'no <part>'
exclusion failed, state affirmatively what should appear instead without repeating 'no <part>'."""


# What can fairly be judged at each framing. Small marks and jewelry do not resolve in wide
# shots, so asking for them there causes false failures.
DETAIL_BY_FRAMING = {
    Framing.CLOSE_UP: "Fine details (identifying marks, jewelry, cuffs, hair texture) are visible "
    "and must be correct.",
    Framing.MEDIUM_CLOSE: "Check face, hair, makeup and costume. The finest details may be soft.",
    Framing.MEDIUM: "Check face shape, hair, costume and build. Fine marks and small jewelry may "
    "not resolve, so do not penalize their absence.",
    Framing.WIDE: "Check silhouette, hair, costume, build and colors only. The face is small, so "
    "do not penalize missing or unresolved facial detail.",
}


_BLIND_OBSERVER_SYSTEM = """You are a neutral visual observer describing {subject} for a
downstream continuity auditor. You are NOT given any target specification. Report ONLY what is
actually visible in {frames_word}—never guess or assume unseen details.

Describe concretely:
- people_count: integer count of visible people or human figures (0 if empty scene).
- framing: observed shot distance (extreme_close_up, close_up, medium_close, medium, wide).
- subjects: who is visible, their pose, facing direction, mouth state (open/singing vs closed),
  hair, clothing layers and neckline coverage (e.g. high-neck turtleneck/undershirt vs. open
  neckline/bare skin, coat, sleeves, accessories), anatomical proportions (hands, limbs, face,
  body scale), and any visible identity or garment-coverage differences from the reference image
  when one is provided.
- setting: visible location, architecture, lighting sources, and mood.
- props_and_objects: notable foreground/midground objects and their physical state (e.g. lit vs
  unlit, intact vs broken, hands on a clock).
- artifacts_or_text: any visible garbled text, warped faces/hands, unnatural body compression,
  or morphing across frames, plus any typography overlaid on the picture (title cards, opening
  credits, subtitles, captions, logos, watermarks) even when it is clean and legible. Say for
  each piece of text whether it is overlaid on the picture or physically part of the scene (a
  sign, a book cover, a clothing print). Empty list if none."""

_BLIND_AUDITOR_SYSTEM = """You are a strict continuity auditor comparing neutral visual
observations (produced by a blind observer who never saw the requirements) against a shot
specification. You do NOT see the image; rely ONLY on the observed facts and the specification.

Score from 0 to 1:
- spec_match: whether observed framing, people count, setting, props, vocal delivery, and all
  explicit exclusions ('Must NOT appear') satisfy the specification. If a forbidden element was
  observed, or a required element clearly contradicted the observation, score <= 0.2.
- artifact_free: whether observed anatomy, proportions, faces, hands, and absence of garbled
  text/watermarks/morphing are clean. Any observed overlaid typography (title cards, opening
  credits, subtitles, captions, logos, watermarks), even clean and legible, MUST score
  artifact_free <= 0.2 unless the specification explicitly asks for text; text that physically
  belongs to the scene is penalized only when it is garbled.
- character_match: whether observed appearance, hair, pinned details, and wardrobe match the
  character specification and reference portrait/sheet (use null when the spec says no singer
  should be visible). If the observer reports a garment-layer or neckline-coverage mismatch
  against the reference portrait/sheet (such as dropping a turtleneck or undershirt shown in the
  reference to expose bare skin), score character_match <= 0.4.

List concrete contradictions in 'issues' (empty if none). In 'fix', give one affirmative prompt
instruction avoiding the worst contradiction without repeating 'no <part>'."""


class _BlindObservation(BaseModel):
    model_config = ConfigDict(extra="ignore")
    people_count: int = Field(default=0, ge=0)
    framing: str = ""
    subjects: str = ""
    setting: str = ""
    props_and_objects: str = ""
    artifacts_or_text: list[str] = Field(default_factory=list)


class VisionVerdict(BaseModel):
    model_config = ConfigDict(extra="ignore")
    spec_match: float = Field(ge=0.0, le=1.0)
    artifact_free: float = Field(ge=0.0, le=1.0)
    character_match: float | None = Field(default=None, ge=0.0, le=1.0)
    issues: list[str] = Field(default_factory=list)
    fix: str = ""


MAX_UNAVAILABLE_IN_A_ROW = 3


class VisionJudge:
    """Model-based visual judge supporting single-pass and two-pass Blind-Observer auditing.

    Credits & Attribution:
        Two-pass Blind-Observer / Split-Auditor verification (eliminating VLM confirmation bias
        by decoupling neutral image observation from text-only spec auditing) inspired by
        ``afj176/open-cinema-studio``
        (https://github.com/afj176/open-cinema-studio —
        ``custom_nodes/comfyui-open-cinema/director.py`` & ``app/graphs.py``).
    """

    def __init__(self, llm: LLMClient, *, blind_observer: bool = False) -> None:
        self._llm = llm
        self._blind_observer = blind_observer
        self._disabled_reason: str | None = None
        self._unavailable_in_a_row = 0

    @property
    def disabled_reason(self) -> str | None:
        """Why the judge stopped working for this run, or None while it is available."""
        return self._disabled_reason

    def disable(self, reason: str) -> None:
        """Stop sending requests for the rest of the run."""
        self._disabled_reason = reason

    def record_success(self) -> None:
        self._unavailable_in_a_row = 0

    def record_unavailable(self, error: str) -> bool:
        """Count a failure that retrying cannot fix. Returns True when the judge gives up.

        An unreachable server or a rejected key fails every request, and each failed request can
        wait out a long timeout. A single invalid reply does not count: that is a bad answer from
        a working model, and the next request may be fine.
        """
        self._unavailable_in_a_row += 1
        if self._unavailable_in_a_row < MAX_UNAVAILABLE_IN_A_ROW:
            return False
        self.disable(
            f"the model was unavailable {MAX_UNAVAILABLE_IN_A_ROW} times in a row "
            f"(last error: {error})"
        )
        return True

    def judge(
        self,
        images: list[Path | np.ndarray],
        shot: Shot,
        manifest: Manifest,
        kind: AssetKind,
        reference_image: Path | None = None,
        extra_references: list[tuple[str, Path]] | None = None,
    ) -> VisionVerdict:
        is_video = kind is AssetKind.VIDEO
        if self._blind_observer:
            return self._judge_blind(
                images,
                shot,
                manifest,
                is_video=is_video,
                reference_image=reference_image,
                extra_references=extra_references,
            )
        system = _SYSTEM.format(
            subject=(
                f"{len(images)} frames sampled in order from one clip"
                if is_video
                else "one still image that will be the first frame of a clip"
            ),
            frames_word="the frames" if is_video else "the image",
            video_artifacts=", and no morphing or identity change between frames"
            if is_video
            else "",
        )
        show_singer = shot.type.shows_singer
        content = []
        if show_singer and reference_image is not None:
            content += [
                text_block(
                    "REFERENCE of the singer (a sheet shows the same person from several angles):"
                ),
                image_block(reference_image),
            ]
        if extra_references:
            for label, ref_path in extra_references:
                if label.startswith("Prop:"):
                    content += [
                        text_block(f"REFERENCE for {label}:"),
                        image_block(ref_path),
                    ]
                elif show_singer:
                    content += [
                        text_block(f"REFERENCE for additional character ({label}):"),
                        image_block(ref_path),
                    ]
        content += [
            text_block("IMAGE TO JUDGE:" if not is_video else "FRAMES TO JUDGE, in order:"),
            *(image_block(image) for image in images),
            text_block(self.specification(shot, manifest)),
        ]
        verdict = self._llm.complete_json(system=system, content=content, schema=VisionVerdict)
        self.record_success()
        return verdict

    def _judge_blind(
        self,
        images: list[Path | np.ndarray],
        shot: Shot,
        manifest: Manifest,
        *,
        is_video: bool,
        reference_image: Path | None = None,
        extra_references: list[tuple[str, Path]] | None = None,
    ) -> VisionVerdict:
        """Two-pass blind-observer / split-auditor VLM check eliminating confirmation bias."""
        subject = (
            f"{len(images)} frames sampled in order from one clip"
            if is_video
            else "one still image"
        )
        frames_word = "the frames" if is_video else "the image"
        obs_system = _BLIND_OBSERVER_SYSTEM.format(subject=subject, frames_word=frames_word)
        obs_content = []
        if reference_image is not None and shot.type.shows_singer:
            obs_content += [
                text_block(
                    "REFERENCE PORTRAIT (compare whether the person in the target image looks "
                    "like the same individual and report similarities/differences neutrally):"
                ),
                image_block(reference_image),
            ]
        if extra_references:
            for label, ref_path in extra_references:
                if label.startswith("Prop:"):
                    obs_content += [
                        text_block(f"REFERENCE for {label}:"),
                        image_block(ref_path),
                    ]
                elif shot.type.shows_singer:
                    obs_content += [
                        text_block(f"REFERENCE for additional character ({label}):"),
                        image_block(ref_path),
                    ]
        obs_content += [
            text_block("IMAGE TO OBSERVE:" if not is_video else "FRAMES TO OBSERVE, in order:"),
            *(image_block(img) for img in images),
        ]
        observation = self._llm.complete_json(
            system=obs_system, content=obs_content, schema=_BlindObservation
        )
        obs_summary = (
            "BLIND OBSERVER REPORT\n"
            f"People count: {observation.people_count}\n"
            f"Framing: {observation.framing}\n"
            f"Subjects: {observation.subjects}\n"
            f"Setting: {observation.setting}\n"
            f"Props & objects: {observation.props_and_objects}\n"
            f"Artifacts/text: {', '.join(observation.artifacts_or_text) or 'none'}"
        )
        verdict = self._llm.complete_json(
            system=_BLIND_AUDITOR_SYSTEM,
            content=[text_block(obs_summary), text_block(self.specification(shot, manifest))],
            schema=VisionVerdict,
        )
        self.record_success()
        return verdict

    @staticmethod
    def specification(shot: Shot, manifest: Manifest) -> str:
        bible = manifest.visual_bible
        character = manifest.shot_character(shot)
        scene = bible.find_set(shot.setting)
        lines = [
            "SHOT SPECIFICATION",
            f"Type: {shot.type.value}",
            f"Framing: {shot.framing.value}",
            f"Section of the song: {shot.section or 'n/a'}",
            f"Setting: {shot.setting}" + (f" - {scene.description}" if scene else ""),
            f"First frame: {shot.still_prompt}",
            f"Action: {shot.motion_prompt}",
            f"Look: {bible.style}",
        ]
        if shot.type.shows_singer and len(manifest.characters) > 1:
            singing = manifest.singing_characters(shot)
            visible = manifest.visible_characters(shot)
            if shot.type is ShotType.LIPSYNC and len(singing) >= 2:
                names = join_names([c.name or cid or "performer" for cid, c, _ in singing])
                lead = f"BOTH {names}" if len(singing) == 2 else names
                lines.append(
                    f"Vocal delivery: Simultaneous duet — {lead} are singing TOGETHER; "
                    "all singing faces must be clearly visible with unobstructed singing mouths."
                )
            elif shot.type is ShotType.LIPSYNC and len(singing) == 1 and len(visible) > 1:
                solo_id, solo_char, _ = singing[0]
                solo_label = solo_char.name or solo_id or "lead"
                listener_list = [
                    c.name or cid or "performer" for cid, c, _ in visible if cid != solo_id
                ]
                listeners = join_names(listener_list)
                be_verb = "are" if len(listener_list) > 1 else "is"
                mouth_noun = "mouths" if len(listener_list) > 1 else "mouth"
                lines.append(
                    f"Vocal delivery: Solo in two-shot — ONLY {solo_label} is singing; "
                    f"{listeners} {be_verb} listening with {mouth_noun} closed."
                )
            elif shot.type is ShotType.LIPSYNC and len(singing) == 1:
                solo_id, solo_char, _ = singing[0]
                solo_label = solo_char.name or solo_id or "lead"
                lines.append(f"Vocal delivery: Solo — ONLY {solo_label} is singing.")
            else:
                lines.append(
                    "Vocal delivery: Non-singing performance — mouths closed, nobody singing."
                )
        if shot.type.shows_singer:
            lines.append(
                f"Detail level for a {shot.framing.value} shot: {DETAIL_BY_FRAMING[shot.framing]}"
            )
            if character.appearance:
                lines.append(f"Singer (must match exactly): {character.appearance}")
            if pinned := character.pinned():
                lines.append(
                    f"Pinned details (expected wherever the framing and angle show them): {pinned}"
                )
            state = character.state(shot.wardrobe)
            if state is not None:
                lines.append(f"Wardrobe state '{state.id}' for this shot: {state.description}")
            visible = manifest.visible_characters(shot)
            for sec_id, sec_char, sec_state in visible[1:]:
                if sec_char.appearance:
                    label = sec_char.name or sec_id or "performer"
                    sec_w = sec_char.state(sec_state)
                    w_text = f" Wardrobe: {sec_w.description}" if sec_w else ""
                    lines.append(
                        f"Additional character in shot ({label}): {sec_char.appearance}{w_text}"
                    )
                    if sec_pinned := sec_char.pinned():
                        lines.append(f"Pinned details for {label}: {sec_pinned}")
        else:
            lines.append("No singer should be visible in this shot.")
        if shot.allow_cuts:
            lines.append(
                "Multi-angle montage (allow_cuts=true): deliberate camera angle cuts between "
                "frames are expected; verify character, wardrobe, and set consistency across cuts."
            )
        for prop in bible.resolve_props(shot.props):
            desc = f": {prop.description}" if prop.description else ""
            lines.append(f"Hero prop ({prop.name}){desc}")
        exclusions = extract_negative_exclusions(shot.still_prompt)
        if exclusions:
            lines.append(
                "Must NOT appear in frame (hard fail if visible): " + ", ".join(exclusions)
            )
        if bible.rules:
            lines.append("Production rules:\n" + "\n".join(f"- {rule}" for rule in bible.rules))
        return "\n".join(lines)
