# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Model-based visual judgement: does the frame match the shot, the character and the rules?"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from mvdirector.llm import LLMClient, image_block, text_block
from mvdirector.models import AssetKind, Framing, Manifest, Shot

_SYSTEM = """You are the quality-control supervisor on an AI-generated music video. You are
shown {subject}. Judge it strictly against the shot specification.

Score from 0 to 1:
- spec_match: framing, setting, lighting and action match the specification, and the
  production rules that can be judged from {frames_word} are followed.
- artifact_free: no malformed hands or fingers, extra or missing limbs, warped faces,
  melted objects, garbled text, watermarks{video_artifacts}.
- character_match: the singer matches the character description and, when a reference
  sheet or portrait is given, looks like the same person. A sheet shows the same person from
  several angles. Judge only details large enough to resolve at the stated framing. Check face
  shape, skin tone, hair cut and length, makeup, build and the costume for this shot's wardrobe
  state (for example a cape must be present or absent exactly as specified). Identifying
  marks must be on the stated side of the singer's anatomical left or right; a mirror-flipped
  face or mark is a mismatch.
  Use null if the specification says no singer should be visible.

List each concrete problem in 'issues' (empty if none). In 'fix', give one short,
practical instruction for the prompt that would avoid the worst problem."""


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


class VisionVerdict(BaseModel):
    model_config = ConfigDict(extra="ignore")
    spec_match: float = Field(ge=0.0, le=1.0)
    artifact_free: float = Field(ge=0.0, le=1.0)
    character_match: float | None = Field(default=None, ge=0.0, le=1.0)
    issues: list[str] = Field(default_factory=list)
    fix: str = ""


MAX_UNAVAILABLE_IN_A_ROW = 3


class VisionJudge:
    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm
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
        character = manifest.shot_character(shot)
        show_singer = shot.type.shows_singer and bool(character.appearance)
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
        if shot.type.shows_singer and character.appearance:
            lines.append(
                f"Detail level for a {shot.framing.value} shot: {DETAIL_BY_FRAMING[shot.framing]}"
            )
            lines.append(f"Singer (must match exactly): {character.appearance}")
            if character.pinned():
                lines.append(
                    f"Pinned details (expected wherever the framing and angle show them): "
                    f"{character.pinned()}"
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
                    if sec_char.pinned():
                        lines.append(f"Pinned details for {label}: {sec_char.pinned()}")
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
        if bible.rules:
            lines.append("Production rules:\n" + "\n".join(f"- {rule}" for rule in bible.rules))
        return "\n".join(lines)
