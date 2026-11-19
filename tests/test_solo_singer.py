# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""A one-singer song must plan, prompt, and review exactly as before the duet work."""

from mvdirector.config import ProjectConfig
from mvdirector.models import Framing, Manifest, ReviewReason, ShotType
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


def test_a_manifest_from_before_the_new_fields_still_loads(project, analysis):
    manifest = _solo(project, analysis)
    raw = manifest.model_dump(mode="json")
    for shot in raw["shots"]:
        del shot["fallback_from"]  # added later; older files never wrote it
    loaded = Manifest.model_validate(raw)
    assert all(s.fallback_from is None for s in loaded.shots)
    assert loaded.shots[0].type is manifest.shots[0].type


def test_the_new_defaults_do_not_depend_on_a_duet():
    config = ProjectConfig()
    assert config.qc.video.lipsync_blocking is False
    assert config.retry.lipsync_fallback is True
    assert config.planning.lipsync_widest_framing is Framing.MEDIUM_CLOSE
    assert ReviewReason.FALLBACK.value == "fallback" and ReviewReason.LIPSYNC.value == "lipsync"
