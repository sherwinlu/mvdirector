# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Capping how much of the song is the singer singing to camera."""

from mvdirector.models import AudioSlice, Shot, ShotType
from mvdirector.planning import planner, prompts
from mvdirector.planning.planner import _thin_lipsync, singing_share


def _shot(index, section, intensity, kind=ShotType.LIPSYNC):
    start = index * 4.0
    return Shot(
        id=f"s{index + 1:03d}",
        index=index,
        start=start,
        end=start + 4.0,
        type=kind,
        framing="medium_close",
        setting="Hall",
        section=section,
        intensity=intensity,
        still_prompt="x",
        motion_prompt="y",
        audio_slice=AudioSlice(path="a.wav", start=start, end=start + 4.0, head_handle=0.0),
        num_frames=97,
    )


def _song():
    sections = ["Verse 1", "Verse 1", "Chorus", "Chorus", "Verse 2", "Verse 2", "Bridge", "Chorus"]
    intensity = [0.3, 0.4, 0.9, 1.0, 0.35, 0.5, 0.6, 0.95]
    return [_shot(i, sec, v) for i, (sec, v) in enumerate(zip(sections, intensity))]


def test_the_share_is_unchanged_when_no_cap_is_set_or_the_song_is_under_it():
    shots = _song()
    assert singing_share(shots) == 1.0
    _thin_lipsync(shots, 1.0)
    assert all(s.type is ShotType.LIPSYNC for s in shots)


def test_quietest_non_hook_shots_become_cutaways_until_the_share_fits():
    shots = _song()
    _thin_lipsync(shots, 0.5)
    assert singing_share(shots) <= 0.5
    kinds = {s.id: s.type for s in shots}
    # The quietest verse shots went first; every chorus and the last shot stayed on camera.
    assert kinds["s001"] is ShotType.BROLL and kinds["s005"] is ShotType.BROLL
    assert all(kinds[i] is ShotType.LIPSYNC for i in ("s003", "s004", "s008"))


def test_the_singer_is_never_off_screen_for_more_than_two_shots_in_a_row():
    shots = _song()
    _thin_lipsync(shots, 0.0)
    run = longest = 0
    for shot in shots:
        run = run + 1 if shot.type is ShotType.BROLL else 0
        longest = max(longest, run)
    assert longest <= 2
    # Hook shots and the final sung shot are protected even when the cap cannot be met.
    assert shots[2].type is ShotType.LIPSYNC and shots[-1].type is ShotType.LIPSYNC


def test_the_model_cannot_bring_the_singer_back_into_a_cutaway():
    shot = _shot(0, "Verse 1", 0.3, ShotType.BROLL)
    creative = planner.ShotCreative(
        id=shot.id,
        type=ShotType.PERFORMANCE,
        framing="wide",
        setting="Hall",
        still_prompt="a",
        motion_prompt="b",
    )
    planner._apply_creative(shot, creative, planner.VisualBible())
    assert shot.type is ShotType.BROLL


def test_a_rule_based_cutaway_has_no_people():
    shot = _shot(0, "Verse 1", 0.3, ShotType.BROLL)
    planner._apply_rule_based(shot)
    assert "no people" in shot.still_prompt.lower()
    assert "No people" in shot.motion_prompt


def test_the_direction_prompt_carries_the_section_arc_and_cutaway_rules():
    text = prompts.SHOTS_SYSTEM
    assert "'broll' are cutaways" in text
    assert "Second verse: never a" in text and "Bridge: break" in text
    assert "three distinct looks" in text


def test_pace_follows_intensity_and_a_shift_is_flagged_in_the_model_context():
    assert [planner.pace_label(v) for v in (0.1, 0.5, 0.9)] == ["calm", "moderate", "high"]
    quiet, loud = _shot(0, "Verse 1", 0.2, ShotType.PERFORMANCE), _shot(1, "Chorus", 0.9)
    for shot in (quiet, loud):
        shot.type = ShotType.PERFORMANCE
        planner._apply_rule_based(shot)
    # Scale tightens and the motion note changes as the song builds, with no camera lurch.
    assert quiet.framing.value == "wide" and loud.framing.value == "medium_close"
    assert "Little movement" in quiet.motion_prompt and "stable" in loud.motion_prompt


def test_a_plan_whose_framing_runs_against_the_music_is_reported(caplog):
    shots = []
    for i in range(10):
        shot = _shot(i, "Verse", i / 10)
        shot.framing = "wide" if i >= 5 else "close_up"  # loudest parts are the widest
        shots.append(shot)
    with caplog.at_level("WARNING"):
        planner._report_pace(shots)
    assert "runs against the song's intensity" in caplog.text


def test_the_direction_prompt_names_the_four_pace_levers():
    assert "four levers" in prompts.SHOTS_SYSTEM and "cut speed" in prompts.SHOTS_SYSTEM
