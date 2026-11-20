# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Director craft: singing-share cap, cutaways, pace labeling, and plan diagnostics."""

import pytest
from pydantic import ValidationError

from mvdirector.config import PlanningConfig, ProjectConfig
from mvdirector.models import (
    Analysis,
    AudioSlice,
    Framing,
    LyricLine,
    Manifest,
    Shot,
    ShotType,
    TimeSpan,
)
from mvdirector.planning import planner, prompts
from mvdirector.planning.planner import (
    BibleReply,
    SetReply,
    ShotBatchReply,
    ShotCreative,
    WardrobeReply,
    _thin_lipsync,
    build_manifest,
    singing_share,
)


def _shot(index, section, intensity, kind=ShotType.LIPSYNC, duration=4.0):
    start = round(index * duration, 3)
    end = round(start + duration, 3)
    return Shot(
        id=f"s{index + 1:03d}",
        index=index,
        start=start,
        end=end,
        type=kind,
        framing=Framing.MEDIUM_CLOSE,
        setting="Hall",
        section=section,
        intensity=intensity,
        still_prompt="x",
        motion_prompt="y",
        audio_slice=AudioSlice(path="a.wav", start=start, end=end, head_handle=0.0),
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


def test_pre_chorus_post_chorus_and_substrings_are_not_protected_as_hooks():
    assert not planner._is_hook_section("Pre-Chorus")
    assert not planner._is_hook_section("Prechorus")
    assert not planner._is_hook_section("Pre Chorus")
    assert not planner._is_hook_section("Post-Chorus")
    assert not planner._is_hook_section("Post Chorus")
    assert not planner._is_hook_section("Verse 1 - unhooked")
    assert planner._is_hook_section("Chorus")
    assert planner._is_hook_section("Final Chorus")
    assert planner._is_hook_section("Chorus - big hook")
    assert planner._is_hook_section("Hook")
    assert planner._is_hook_section("Pre-Chorus / Chorus")

    # Interleave a Chorus so _MAX_CUTAWAY_RUN does not block Post-Chorus from thinning.
    sections = [
        "Pre-Chorus",
        "Prechorus",
        "Chorus",
        "Post-Chorus",
        "Verse 1 - unhooked",
        "Chorus - big hook",
        "Hook",
        "Outro",
    ]
    intensities = [0.2, 0.25, 0.9, 0.3, 0.35, 0.95, 0.9, 0.4]
    shots = [_shot(i, sec, v) for i, (sec, v) in enumerate(zip(sections, intensities))]
    _thin_lipsync(shots, 0.5)
    kinds = {s.section: s.type for s in shots}
    assert kinds["Pre-Chorus"] is ShotType.BROLL
    assert kinds["Prechorus"] is ShotType.BROLL
    assert kinds["Post-Chorus"] is ShotType.BROLL
    assert kinds["Verse 1 - unhooked"] is ShotType.BROLL
    assert kinds["Chorus"] is ShotType.LIPSYNC
    assert kinds["Chorus - big hook"] is ShotType.LIPSYNC
    assert kinds["Hook"] is ShotType.LIPSYNC
    assert kinds["Outro"] is ShotType.LIPSYNC


def test_exact_share_boundary_does_not_drop_an_extra_shot_from_float_drift():
    # 10 shots of 1.1s with 3 sung shots = exact 0.3 share (sum([1.1]*3) - 0.3 * 11.0 == 8.88e-16).
    shots = [
        _shot(i, "Verse 1", 0.3, ShotType.LIPSYNC if i < 3 else ShotType.PERFORMANCE, duration=1.1)
        for i in range(10)
    ]
    _thin_lipsync(shots, 0.3)
    assert sum(1 for s in shots if s.type is ShotType.LIPSYNC) == 3


def test_the_model_cannot_bring_the_singer_back_into_a_cutaway(caplog):
    shot = _shot(0, "Verse 1", 0.3, ShotType.BROLL)
    creative = planner.ShotCreative(
        id=shot.id,
        type=ShotType.PERFORMANCE,
        framing=Framing.WIDE,
        setting="Hall",
        still_prompt="The singer leans against the wall.",
        motion_prompt="The singer nods to the beat.",
    )
    with caplog.at_level("WARNING", logger="mvdirector.planning.planner"):
        planner._apply_creative(shot, creative, planner.VisualBible())
    assert shot.type is ShotType.BROLL
    assert "no people" in shot.still_prompt.lower() and "No people" in shot.motion_prompt
    assert "singer" not in shot.still_prompt.lower()
    assert "put the singer back in a cutaway" in caplog.text


def test_llm_performance_to_broll_conversion_respects_max_cutaway_run(caplog):
    beats = [round(0.5 * i, 3) for i in range(40)]
    analysis = Analysis(
        duration=20.0,
        tempo_bpm=120.0,
        beats=beats,
        bars=beats[::4],
        bar_intensity=[0.2, 0.2, 0.3, 0.3, 0.5, 0.5, 0.25, 0.25, 0.9, 0.9],
        vocal_windows=[TimeSpan(start=0.0, end=8.0), TimeSpan(start=12.0, end=20.0)],
        lyric_lines=[
            LyricLine(text="a", section="Verse 1", start=0.0, end=8.0, timing_source="aligned"),
            LyricLine(text="b", section="Chorus", start=12.0, end=20.0, timing_source="aligned"),
        ],
    )

    class AllCutawayDirector:
        def complete_json(self, *, system, content, schema):
            if schema is BibleReply:
                return BibleReply(
                    appearance="Singer",
                    style="film",
                    palette="warm",
                    lighting="soft",
                    negative_prompt="blurry",
                    settings=[
                        SetReply(name="Hall", description="hall"),
                        SetReply(name="Roof", description="roof"),
                        SetReply(name="Street", description="street"),
                    ],
                )
            return ShotBatchReply(
                shots=[
                    ShotCreative(
                        id=f"s{i + 1:03d}",
                        type=ShotType.BROLL,
                        framing=Framing.WIDE,
                        setting="Hall",
                        still_prompt="empty room",
                        motion_prompt="dust drifts",
                    )
                    for i in range(5)
                ]
            )

    config = ProjectConfig()
    config.video.max_clip_seconds = 4.0
    config.planning.max_lipsync_share = 0.4  # thins s001 and s002 to BROLL; s003 is PERFORMANCE
    with caplog.at_level("WARNING", logger="mvdirector.planning.planner"):
        manifest = build_manifest(config, analysis, "lyrics", "style", llm=AllCutawayDirector())
    # s001 and s002 are thinned cutaways, so s003 cannot also turn into BROLL — and it gets
    # singer-on-camera rule-based prompts rather than keeping the rejected cutaway text.
    assert manifest.shot("s001").type is ShotType.BROLL
    assert manifest.shot("s002").type is ShotType.BROLL
    s3 = manifest.shot("s003")
    assert s3.type is ShotType.PERFORMANCE
    assert s3.setting == "Hall"
    assert "singer" in s3.still_prompt.lower() and "empty room" not in s3.still_prompt.lower()
    assert "off screen for more than 2 shots in a row" in caplog.text


def test_solo_broll_cutaways_clear_wardrobe_and_do_not_advance_monotonic_state():
    beats = [round(0.5 * i, 3) for i in range(32)]
    analysis = Analysis(
        duration=16.0,
        tempo_bpm=120.0,
        beats=beats,
        bars=beats[::4],
        bar_intensity=[0.2, 0.2, 0.3, 0.3, 0.9, 0.9, 0.95, 0.95],
        vocal_windows=[TimeSpan(start=0.0, end=16.0)],
        lyric_lines=[
            LyricLine(text="a", section="Verse 1", start=0.0, end=8.0, timing_source="aligned"),
            LyricLine(text="b", section="Chorus", start=8.0, end=16.0, timing_source="aligned"),
        ],
    )

    class SoloWardrobeDirector:
        def complete_json(self, *, system, content, schema):
            if schema is BibleReply:
                return BibleReply(
                    appearance="Singer",
                    style="film",
                    palette="warm",
                    lighting="soft",
                    negative_prompt="blurry",
                    wardrobe=[
                        WardrobeReply(id="coat", description="Long coat."),
                        WardrobeReply(id="suit", description="Tailored suit. No coat."),
                    ],
                    settings=[
                        SetReply(name="Hall", description="hall"),
                        SetReply(name="Roof", description="roof"),
                        SetReply(name="Stage", description="stage"),
                    ],
                )
            # LLM wrongly puts later wardrobe "suit" on the thinned cutaway s001, then asks for
            # "coat" on the on-camera chorus shot s003.
            return ShotBatchReply(
                shots=[
                    ShotCreative(
                        id="s001",
                        type=ShotType.BROLL,
                        framing=Framing.WIDE,
                        setting="Hall",
                        wardrobe="suit",
                        still_prompt="rain on window",
                        motion_prompt="rain falls",
                    ),
                    ShotCreative(
                        id="s002",
                        type=ShotType.BROLL,
                        framing=Framing.WIDE,
                        setting="Roof",
                        wardrobe="suit",
                        still_prompt="empty rooftop",
                        motion_prompt="clouds drift",
                    ),
                    ShotCreative(
                        id="s003",
                        type=ShotType.LIPSYNC,
                        framing=Framing.CLOSE_UP,
                        setting="Stage",
                        wardrobe="coat",
                        still_prompt="singer on stage",
                        motion_prompt="singer sings",
                    ),
                    ShotCreative(
                        id="s004",
                        type=ShotType.LIPSYNC,
                        framing=Framing.CLOSE_UP,
                        setting="Stage",
                        wardrobe="suit",
                        still_prompt="singer on stage",
                        motion_prompt="singer sings",
                    ),
                ]
            )

    config = ProjectConfig()
    config.video.max_clip_seconds = 4.0
    config.planning.max_lipsync_share = 0.5
    manifest = build_manifest(config, analysis, "lyrics", "style", llm=SoloWardrobeDirector())
    assert manifest.shot("s001").type is ShotType.BROLL and manifest.shot("s001").wardrobe is None
    assert manifest.shot("s002").type is ShotType.BROLL and manifest.shot("s002").wardrobe is None
    # Cutaways did not advance monotonic wardrobe, so s003 keeps "coat" before s004 moves to "suit".
    assert manifest.shot("s003").wardrobe == "coat"
    assert manifest.shot("s004").wardrobe == "suit"


def test_a_rule_based_cutaway_has_no_people_and_matches_pace():
    calm = _shot(0, "Verse 1", 0.2, ShotType.BROLL)
    high = _shot(1, "Bridge", 0.85, ShotType.BROLL)
    planner._apply_rule_based(calm)
    planner._apply_rule_based(high)
    assert calm.framing is Framing.WIDE and high.framing is Framing.MEDIUM_CLOSE
    assert "no people" in calm.still_prompt.lower() and "No people" in calm.motion_prompt
    assert "no people" in high.still_prompt.lower() and "No people" in high.motion_prompt
    assert calm.motion_prompt != high.motion_prompt and "Brisk movement" in high.motion_prompt


def test_the_direction_prompt_carries_the_section_arc_and_cutaway_rules():
    text = prompts.SHOTS_SYSTEM
    assert "'broll' are cutaways" in text
    assert "Second verse: never a" in text and "Bridge: break" in text
    assert "three distinct looks" in text


def test_pace_follows_intensity_and_shifts_are_flagged_across_batches():
    assert [planner.pace_label(v) for v in (0.1, 0.5, 0.9)] == ["calm", "moderate", "high"]
    quiet, loud = _shot(0, "Verse 1", 0.2, ShotType.PERFORMANCE), _shot(1, "Chorus", 0.9)
    for shot in (quiet, loud):
        shot.type = ShotType.PERFORMANCE
        planner._apply_rule_based(shot)
    # Scale tightens and the motion note changes as the song builds, with no camera lurch.
    assert quiet.framing is Framing.WIDE and loud.framing is Framing.MEDIUM_CLOSE
    assert "Little movement" in quiet.motion_prompt and "stable" in loud.motion_prompt

    # Verify both within-batch and cross-batch pace shift notes in the LLM prompt context.
    captured: list[str] = []

    class CapturingLLM:
        def complete_json(self, *, system, content, schema):
            captured.append(content[0]["text"])
            return ShotBatchReply(shots=[])

    shots = [
        _shot(0, "Verse 1", 0.2, ShotType.PERFORMANCE),
        _shot(1, "Verse 1", 0.5, ShotType.PERFORMANCE),  # within-batch: calm -> moderate
        _shot(2, "Chorus", 0.9, ShotType.LIPSYNC),  # cross-batch: moderate -> high
        _shot(3, "Chorus", 0.9, ShotType.LIPSYNC),  # steady: high -> high
    ]
    manifest = Manifest(title="Pace Test", duration=16.0, fps=24, shots=shots)
    llm = CapturingLLM()
    planner._ask_for_batch(manifest, shots[:2], 0, None, None, llm)
    planner._ask_for_batch(manifest, shots[2:], 2, None, None, llm)

    assert "id=s001" in captured[0] and "pace=calm " in captured[0]
    assert "(was " not in captured[0].split("id=s002")[0]
    assert "pace=moderate (was calm)" in captured[0]
    assert "pace=high (was moderate)" in captured[1]
    assert "id=s004" in captured[1] and "(was " not in captured[1].split("id=s004")[1]


def test_a_plan_whose_framing_runs_against_the_music_is_reported(caplog):
    shots = []
    for i in range(10):
        shot = _shot(i, "Verse", i / 10)
        shot.framing = Framing.WIDE if i >= 5 else Framing.CLOSE_UP  # loudest parts are widest
        shots.append(shot)
    with caplog.at_level("WARNING", logger="mvdirector.planning.planner"):
        planner._report_pace(shots)
    assert "runs against the song's intensity" in caplog.text


def test_report_pace_warns_on_uniform_framing_but_not_flat_song_intensity(caplog):
    same_framing = [_shot(i, "Verse", i / 10) for i in range(8)]
    with caplog.at_level("WARNING", logger="mvdirector.planning.planner"):
        planner._report_pace(same_framing)
    assert "Every shot has the same framing (medium_close)" in caplog.text

    caplog.clear()
    flat_intensity = []
    for i in range(8):
        shot = _shot(i, "Verse", 0.5)
        shot.framing = Framing.WIDE if i % 2 == 0 else Framing.CLOSE_UP
        flat_intensity.append(shot)
    with caplog.at_level("WARNING", logger="mvdirector.planning.planner"):
        planner._report_pace(flat_intensity)
    assert caplog.text == ""


def test_report_mix_warns_below_three_sets_only_for_bible_plans_with_enough_shots(caplog):
    shots = [_shot(i, "Verse", 0.5) for i in range(4)]
    with caplog.at_level("WARNING", logger="mvdirector.planning.planner"):
        planner._report_mix(shots, has_bible=False)
    assert caplog.text == ""

    with caplog.at_level("WARNING", logger="mvdirector.planning.planner"):
        planner._report_mix(shots[:2], has_bible=True)
    assert caplog.text == ""

    with caplog.at_level("WARNING", logger="mvdirector.planning.planner"):
        planner._report_mix(shots, has_bible=True)
    assert "Only 1 distinct set(s)" in caplog.text


def test_the_direction_prompt_names_the_four_pace_levers():
    assert "four levers" in prompts.SHOTS_SYSTEM and "cut speed" in prompts.SHOTS_SYSTEM


def test_capping_the_singing_share_is_on_by_default_and_validates_bounds():
    default_cap = ProjectConfig().planning.max_lipsync_share
    assert default_cap == 0.45
    assert PlanningConfig(max_lipsync_share=None).max_lipsync_share is None
    assert PlanningConfig(max_lipsync_share=0.0).max_lipsync_share == 0.0
    assert PlanningConfig(max_lipsync_share=1.0).max_lipsync_share == 1.0
    for bad in (-0.1, 1.1, float("nan"), float("inf")):
        with pytest.raises(ValidationError):
            PlanningConfig(max_lipsync_share=bad)

    shots = _song()
    _thin_lipsync(shots, default_cap)
    # Hook shots and the run limit stop this song short of the cap, but it still thins out.
    assert singing_share(shots) <= 0.55


@pytest.mark.parametrize("raw_framing", [None, "", "none", "null", "N/A"])
def test_a_reply_with_no_framing_gets_the_rule_based_one(raw_framing):
    reply = ShotBatchReply.model_validate(
        {
            "shots": [
                {
                    "id": "s001",
                    "type": "broll",
                    "framing": raw_framing,
                    "setting": "Hall",
                    "still_prompt": "a",
                    "motion_prompt": "b",
                },
                {
                    "id": "s002",
                    "type": "lipsync",
                    "framing": raw_framing,
                    "setting": "Hall",
                    "still_prompt": "a",
                    "motion_prompt": "b",
                },
            ]
        }
    )
    assert reply.shots[0].framing is None and reply.shots[1].framing is None
    cutaway = _shot(0, "Verse 1", 0.2, ShotType.BROLL)
    planner._apply_creative(cutaway, reply.shots[0], planner.VisualBible())
    assert cutaway.framing is Framing.WIDE  # calm pace

    # Singing shot with no framing matches _apply_rule_based (high pace -> close_up).
    loud_sung = _shot(1, "Chorus", 0.9, ShotType.LIPSYNC)
    planner._apply_creative(loud_sung, reply.shots[1], planner.VisualBible())
    assert loud_sung.framing is Framing.CLOSE_UP


def test_in_a_duet_each_singer_keeps_half_of_their_sung_shots_on_camera():
    shots = [_shot(i, f"Verse: {who}", 0.2 + i / 100) for i, who in enumerate("EEEEOOOO")]
    singer = {s.id: ("elias" if s.section.endswith("E") else "odalie") for s in shots}
    _thin_lipsync(shots, 0.0, lambda s: singer[s.id])
    for who in ("elias", "odalie"):
        kept = [s for s in shots if singer[s.id] == who and s.type is ShotType.LIPSYNC]
        assert len(kept) >= 2
    # With one singer there is no floor: 6 shots thin to 2 (under half, limited by the run cap).
    solo = [_shot(i, "Verse", 0.2 + i / 100) for i in range(6)]
    _thin_lipsync(solo, 0.0, lambda s: "elias")
    assert sum(s.type is ShotType.LIPSYNC for s in solo) == 2


def test_untagged_sections_in_a_duet_are_not_floored_as_a_third_singer():
    # 4 Elias, 6 untagged Bridge shots (quietest), 4 Odalie: Bridge should thin past half (to 2/6).
    specs = (
        [("Verse 1: Elias", 0.4 + i / 100) for i in range(4)]
        + [("Bridge", 0.1 + i / 100) for i in range(6)]
        + [("Verse 2: Odalie", 0.5 + i / 100) for i in range(4)]
    )
    shots = [_shot(i, sec, v) for i, (sec, v) in enumerate(specs)]
    characters = {
        "elias": planner.Character(name="Elias", appearance="singer A"),
        "odalie": planner.Character(name="Odalie", appearance="singer B"),
    }
    _thin_lipsync(shots, 0.0, lambda s: planner._section_singer(s, characters))
    assert sum(s.section == "Verse 1: Elias" and s.type is ShotType.LIPSYNC for s in shots) == 2
    assert sum(s.section == "Verse 2: Odalie" and s.type is ShotType.LIPSYNC for s in shots) >= 2
    assert sum(s.section == "Bridge" and s.type is ShotType.LIPSYNC for s in shots) == 2


def test_build_manifest_thins_duet_after_bible_using_section_singers():
    sections = ["Verse 1: Elias"] * 4 + ["Verse 2: Odalie"] * 4
    n = len(sections)
    duration = n * 4.0
    analysis = Analysis(
        duration=duration,
        tempo_bpm=120.0,
        beats=[i * 0.5 for i in range(int(duration * 2))],
        bars=[i * 2.0 for i in range(int(duration // 2))],
        bar_intensity=[0.2 + i / 50 for i in range(int(duration // 2))],
        vocal_windows=[TimeSpan(start=0.0, end=duration)],
        lyric_lines=[
            LyricLine(start=i * 4.0, end=(i + 1) * 4.0, text=f"line {i}", section=sec)
            for i, sec in enumerate(sections)
        ],
    )

    class DuetLLM:
        def complete_json(self, *, system, content, schema):
            if schema is BibleReply:
                return BibleReply(
                    concept="duet",
                    style="film",
                    palette="warm",
                    lighting="soft",
                    negative_prompt="blurry",
                    characters=[
                        planner.CharacterReply(id="elias", name="Elias", appearance="A"),
                        planner.CharacterReply(id="odalie", name="Odalie", appearance="B"),
                    ],
                    settings=[SetReply(name="Hall", description="wide hall")],
                )
            assert schema is ShotBatchReply
            return ShotBatchReply(
                shots=[
                    ShotCreative(
                        id=f"s{i + 1:03d}",
                        type=ShotType.BROLL,
                        framing=None,
                        setting="Hall",
                        still_prompt="still",
                        motion_prompt="motion",
                    )
                    for i in range(n)
                ]
            )

    config = ProjectConfig()
    config.video.max_clip_seconds = 4.0
    config.planning.max_lipsync_share = 0.0
    manifest = build_manifest(config, analysis, "", "style", DuetLLM())
    for label in ("Verse 1: Elias", "Verse 2: Odalie"):
        kept = [s for s in manifest.shots if s.section == label and s.type is ShotType.LIPSYNC]
        assert len(kept) >= 2
    assert all(s.still_prompt == "still" for s in manifest.shots)


def test_empty_landscape_performance_prompt_converts_to_broll_or_rule_based_singer():
    manor_prompt = (
        "Wide, static shot of the crooked Victorian manor on a snow-covered hill, "
        "silhouette against a pearl gray and blue sky."
    )
    shot = _shot(0, "Intro", 0.3, ShotType.PERFORMANCE)
    creative = ShotCreative(
        id=shot.id,
        type=ShotType.PERFORMANCE,
        framing=Framing.WIDE,
        setting="Manor",
        still_prompt=manor_prompt,
        motion_prompt="The singer stands in the snow.",
    )
    planner._apply_creative(shot, creative, planner.VisualBible(), allow_broll_conversion=True)
    assert shot.type is ShotType.BROLL
    assert shot.still_prompt == manor_prompt
    assert "No people" in shot.motion_prompt

    # When 2 cutaways already preceded this shot, keep it PERFORMANCE with a real singer prompt.
    capped = _shot(2, "Intro", 0.3, ShotType.PERFORMANCE)
    planner._apply_creative(capped, creative, planner.VisualBible(), allow_broll_conversion=False)
    assert capped.type is ShotType.PERFORMANCE
    assert "singer" in capped.still_prompt.lower()
    assert "Victorian manor" not in capped.still_prompt


def test_wide_still_prompt_scales_singer_and_extracts_negative_exclusions():
    wide_shot = _shot(0, "Intro", 0.3, ShotType.PERFORMANCE)
    wide_shot.framing = Framing.WIDE
    wide_shot.still_prompt = "Wide shot of the singer walking up the snowy path toward the manor."
    manifest = Manifest(
        title="t",
        duration=4.0,
        fps=24,
        character=planner.Character(appearance="Teen boy in dark coat and red scarf."),
        shots=[wide_shot],
    )
    composed = prompts.compose_still_prompt(manifest, wide_shot)
    assert "~7.5 heads tall" in composed and "never towering over buildings" in composed
    assert "human eye height" not in composed

    clock_prompt = (
        "Dust motes floating in shafts of light revealing a large silent grandfather clock, "
        "no minute hand, tarnished brass."
    )
    assert prompts.extract_negative_exclusions(clock_prompt) == ["minute hand"]
    lyric = "A blue that paints the town, no promise of tomorrow"
    assert prompts.extract_negative_exclusions(lyric) == []
