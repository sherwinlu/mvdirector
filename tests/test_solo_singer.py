# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""A one-singer song must plan, prompt, and review exactly as before the duet work."""

from mvdirector.models import Framing, ShotType
from mvdirector.planning import prompts
from mvdirector.planning.planner import build_manifest


def _solo(project, analysis):
    project.config.video.max_clip_seconds = 4.0
    return build_manifest(project.config, analysis, "lyrics", "style", llm=None)


def test_solo_singing_shots_stay_close_and_prompts_have_no_duet_text(project, analysis):
    manifest = _solo(project, analysis)
    sung = [s for s in manifest.shots if s.type is ShotType.LIPSYNC]
    assert sung and all(s.framing in (Framing.CLOSE_UP, Framing.MEDIUM_CLOSE) for s in sung)
    for shot in manifest.shots:
        text = prompts.compose_still_prompt(manifest, shot)
        assert "Also in frame" not in text and "Always:" not in text
        assert len(manifest.visible_characters(shot)) == 1


def test_solo_pinned_details_appear_once_and_only_when_set(project, analysis):
    manifest = _solo(project, analysis)
    shot = next(s for s in manifest.shots if s.type is ShotType.LIPSYNC)
    manifest.character.invariants = "a small scar on the left cheek"
    text = prompts.compose_still_prompt(manifest, shot)
    assert text.count("Always: a small scar on the left cheek.") == 1
