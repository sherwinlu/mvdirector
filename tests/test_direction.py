# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Creative direction: sections, bible structure, wardrobe continuity and prompt composition."""

import re

import pytest

from mvdirector.comfy.client import ComfyClient
from mvdirector.config import ProjectConfig
from mvdirector.errors import LLMError
from mvdirector.generation import reference
from mvdirector.generation.runner import load_templates
from mvdirector.models import (
    Analysis,
    AudioSlice,
    Character,
    LyricLine,
    Manifest,
    SetDesign,
    Shot,
    ShotType,
    TimeSpan,
    VisualBible,
    WardrobeState,
)
from mvdirector.planning import prompts
from mvdirector.planning.planner import (
    BibleReply,
    CharacterReply,
    SetReply,
    ShotBatchReply,
    ShotCreative,
    WardrobeReply,
    build_manifest,
)
from mvdirector.planning.segmenter import section_at, section_timeline, segment
from tests.conftest import FakeComfy

CAPE = "Black bodysuit with silver shoulder caps, wearing a floor-length black cape."
BARE = "Black bodysuit with silver shoulder caps. No cape."


@pytest.fixture
def sectioned() -> Analysis:
    """24 s song: instrumental lead-in, verse from 4 s, chorus from 12 s."""
    beats = [round(0.5 * i, 3) for i in range(48)]
    lines = [
        LyricLine(text="a", section="Verse 1", start=4.0, end=8.0, timing_source="aligned"),
        LyricLine(text="b", section="Verse 1", start=8.0, end=12.0, timing_source="aligned"),
        LyricLine(text="c", section="Chorus", start=12.0, end=16.0, timing_source="aligned"),
    ]
    return Analysis(
        duration=24.0,
        tempo_bpm=120,
        beats=beats,
        bars=beats[::4],
        bar_intensity=[0.5] * 12,
        vocal_windows=[TimeSpan(start=4.0, end=16.0)],
        lyric_lines=lines,
    )


class Director:
    """A scripted director: returns a bible and a fixed (setting, wardrobe) per shot id."""

    def __init__(self, plan):
        self.plan, self.shot_requests = plan, []

    def complete_json(self, *, system, content, schema):
        if schema is BibleReply:
            self.bible_request = content[0]["text"]
            return BibleReply(
                appearance="Singer with a beauty mark under her left eye.",
                style="glossy",
                palette="cobalt",
                lighting="hard",
                negative_prompt="blurry",
                concept="A knight turns a cage into a stage.",
                rules=["Eyes and mouth stay visible.", "The cape never returns after the bridge."],
                wardrobe=[
                    WardrobeReply(id="cape", description=CAPE),
                    WardrobeReply(id="bare", description=BARE),
                ],
                settings=[
                    SetReply(
                        name="Corridor",
                        sections=["Verse"],
                        description="a narrow black corridor",
                        movement="walks toward camera",
                    ),
                    SetReply(
                        name="Stage",
                        sections=["Chorus"],
                        description="a large black stage",
                        movement="angular arm frames",
                    ),
                ],
            )
        text = content[0]["text"]
        self.shot_requests.append(text)
        ids = re.findall(r"- id=(s\d+)", text)
        shots = []
        for shot_id in ids:
            setting, wardrobe = self.plan[shot_id]
            shots.append(
                ShotCreative(
                    id=shot_id,
                    type=ShotType.LIPSYNC,
                    framing="close_up",
                    setting=setting,
                    wardrobe=wardrobe,
                    still_prompt=f"still {shot_id}",
                    motion_prompt=f"motion {shot_id}",
                )
            )
        return ShotBatchReply(shots=shots)


PLAN = {
    "s001": ("Corridor", "cape"),
    "s002": ("Corridor", "cape"),
    "s003": ("stage", "bare"),  # wrong case
    "s004": ("Corridor Set", "cape"),  # fuzzy name; tries to bring the cape back
    "s005": ("Moon base", "bare"),  # not a set
    "s006": ("Stage", "no_such_state"),
}


def _build(sectioned, plan=PLAN, **planning):
    config = ProjectConfig()
    config.video.max_clip_seconds = 4.0
    config.llm.shots_per_request = 2
    config.planning.max_lipsync_share = None  # these scenarios keep sung shots on camera
    for key, value in planning.items():
        setattr(config.planning, key, value)
    director = Director(plan)
    manifest = build_manifest(config, sectioned, "[Verse 1]\na\n", "brief", director)
    return manifest, director


# --------------------------------------------------------------------- sections


def test_section_timeline_marks_each_change(sectioned):
    timeline = section_timeline(sectioned)
    assert timeline == [(4.0, "Verse 1"), (12.0, "Chorus")]
    assert section_at(timeline, 2.0) is None
    assert section_at(timeline, 6.0) == "Verse 1"
    assert section_at(timeline, 20.0) == "Chorus"  # stays until the next section starts


def test_repeated_section_labels_are_separate_runs():
    lines = [
        LyricLine(text="a", section="Chorus", start=0.0, end=2.0, timing_source="aligned"),
        LyricLine(text="b", section="Verse", start=4.0, end=6.0, timing_source="aligned"),
        LyricLine(text="c", section="Chorus", start=8.0, end=10.0, timing_source="aligned"),
    ]
    analysis = Analysis(
        duration=12, tempo_bpm=120, beats=[], bars=[], bar_intensity=[], vocal_windows=[],
        lyric_lines=lines,
    )  # fmt: skip
    assert [label for _, label in section_timeline(analysis)] == ["Chorus", "Verse", "Chorus"]


def test_shots_carry_their_section(sectioned):
    config = ProjectConfig()
    config.video.max_clip_seconds = 4.0
    assert [s.section for s in segment(sectioned, config.video)] == [
        None, "Verse 1", "Verse 1", "Chorus", "Chorus", "Chorus",
    ]  # fmt: skip


# ------------------------------------------------------------- what the model sees


def test_bible_request_lists_song_sections(sectioned):
    config = ProjectConfig()
    director = Director(PLAN)
    lyrics = "[Verse 1]\nline\n[Chorus - big hook]\nline\n[Verse 2]\nline\n[Chorus - big hook]\nx\n"
    build_manifest(config, sectioned, lyrics, "brief", director)
    listed = director.bible_request.split("SONG SECTIONS, IN ORDER:")[1].split("LYRICS:")[0]
    assert re.findall(r"- (.+)", listed) == [
        "Verse 1",
        "Chorus - big hook",
        "Verse 2",
        "Chorus - big hook",
    ]


def test_bible_fields_are_stored(sectioned):
    manifest, _ = _build(sectioned)
    bible, character = manifest.visual_bible, manifest.character
    assert bible.concept.startswith("A knight") and len(bible.rules) == 2
    assert [s.name for s in bible.settings] == ["Corridor", "Stage"]
    assert bible.settings[1].sections == ["Chorus"] and bible.settings[1].movement
    assert character.appearance.startswith("Singer") and character.wardrobe_ids == ["cape", "bare"]


def test_shot_request_carries_bible_section_position_and_context(sectioned):
    manifest, director = _build(sectioned)
    first, second, third = director.shot_requests
    for text in (first, second, third):
        assert "PRODUCTION RULES" in text and "The cape never returns after the bridge." in text
        assert "WARDROBE STATES" in text and "- bare: " in text
        assert "SETS (use these exact names)" in text and "movement: angular arm frames" in text
    assert "section=None" not in first and "(instrumental lead-in)" in first
    assert "17% through the song" in first  # s002 starts at 4 s of 24
    assert "section='Verse 1'" in first and "section='Chorus'" in second
    # The first batch starts in the first costume and has nothing earlier to be consistent with.
    assert "ALREADY DIRECTED" not in first and "CURRENT WARDROBE STATE: cape" in first
    # Later batches see what earlier ones decided.
    assert "ALREADY DIRECTED" in second and "- s002 section='Verse 1' setting='Corridor'" in second
    assert "CURRENT WARDROBE STATE: cape" in second
    assert "CURRENT WARDROBE STATE: bare" in third


# ------------------------------------------------------------------- continuity


def test_wardrobe_changes_only_move_forward(sectioned):
    manifest, _ = _build(sectioned)
    # s004 asked for the cape back, s006 for an unknown state: both keep the current costume.
    assert [s.wardrobe for s in manifest.shots] == ["cape", "cape", "bare", "bare", "bare", "bare"]


def test_wardrobe_may_regress_when_configured(sectioned):
    manifest, _ = _build(sectioned, wardrobe_monotonic=False)
    assert [s.wardrobe for s in manifest.shots][2:5] == ["bare", "cape", "bare"]


def test_settings_are_mapped_onto_bible_sets(sectioned):
    manifest, _ = _build(sectioned)
    assert [s.setting for s in manifest.shots] == [
        "Corridor", "Corridor", "Stage", "Corridor", "Corridor", "Stage",
    ]  # fmt: skip


def test_rule_based_plan_has_no_wardrobe(sectioned):
    config = ProjectConfig()
    manifest = build_manifest(config, sectioned, "lyrics", "brief", llm=None)
    assert all(s.wardrobe is None for s in manifest.shots)


# ----------------------------------------------------------- prompt composition


def test_still_prompt_applies_set_and_the_wardrobe_for_the_shot(sectioned):
    manifest, _ = _build(sectioned)
    early, late = manifest.shots[0], manifest.shots[3]
    early_prompt = prompts.compose_still_prompt(manifest, early)
    late_prompt = prompts.compose_still_prompt(manifest, late)
    assert "Setting: a narrow black corridor" in early_prompt
    assert (
        "beauty mark under her left eye" in early_prompt
        and "wearing a floor-length" in early_prompt
    )
    assert "No cape." in late_prompt and "wearing a floor-length" not in late_prompt
    manifest.shots[3].type = ShotType.BROLL
    assert "The singer:" not in prompts.compose_still_prompt(manifest, manifest.shots[3])


def test_character_describe_and_portrait_text():
    character = Character(
        appearance="Appearance.",
        wardrobe=[
            WardrobeState(id="a", description="First."),
            WardrobeState(id="b", description="Second."),
        ],
    )
    assert character.describe("b") == "Appearance. Wearing: Second."
    assert character.describe(None) == character.describe("missing") == "Appearance."
    assert character.portrait_text() == "Appearance. Wearing: First."
    assert Character(appearance="Only.").portrait_text() == "Only."


def test_bible_set_lookup_ignores_case():
    bible = VisualBible(settings=[SetDesign(name="The Vault")])
    assert bible.find_set("  the vault ").name == "The Vault"
    assert bible.find_set("elsewhere") is None


def test_reference_sheet_uses_appearance_and_first_costume(project, sample_png, sectioned):
    manifest, _ = _build(sectioned)
    fake = FakeComfy(png=sample_png)
    templates = load_templates(project, ["t2i", "still"])
    with ComfyClient(
        project.config.comfy, transport=fake.transport(), sleep=lambda _: None
    ) as comfy:
        reference.generate_candidates(project, manifest, comfy, templates, None, count=1)
    graph = fake.queued[0]
    prompt = graph["1"]["inputs"]["text"]
    assert "beauty mark under her left eye" in prompt and "floor-length black cape" in prompt
    assert "Four equal panels in a single horizontal row" in prompt
    assert (graph["2"]["inputs"]["width"], graph["2"]["inputs"]["height"]) == (2048, 896)


# ------------------------------------------------------ a model that cannot do the job


class FailingBatches(Director):
    """Writes a good bible but cannot produce valid shot direction."""

    def complete_json(self, *, system, content, schema):
        if schema is ShotBatchReply:
            raise LLMError("reply could not be parsed")
        return super().complete_json(system=system, content=content, schema=schema)


def test_a_failed_shot_batch_falls_back_to_rule_based_direction(sectioned, caplog):
    config = ProjectConfig()
    config.video.max_clip_seconds = 4.0
    config.llm.shots_per_request = 2
    with caplog.at_level("WARNING", logger="mvdirector.planning.planner"):
        manifest = build_manifest(config, sectioned, "lyrics", "style", FailingBatches(PLAN))
    assert manifest.visual_bible.concept  # the bible still came from the model
    assert all(s.still_prompt and s.motion_prompt for s in manifest.shots)
    assert manifest.shots[0].still_prompt.startswith(("Close-up", "Medium", "Wide"))
    assert caplog.text.count("could not direct shots") == 3  # three batches of two shots
    assert "llm.shots_per_request" in caplog.text


def test_a_failed_bible_stops_planning_with_advice(sectioned):
    class NoBible(Director):
        def complete_json(self, *, system, content, schema):
            raise LLMError("401 unauthorized")

    with pytest.raises(LLMError, match="visual bible.*401 unauthorized.*mvd llm-check"):
        build_manifest(ProjectConfig(), sectioned, "lyrics", "style", NoBible(PLAN))


# --------------------------------------------------------- duets, bands and ensembles


def test_duet_planning_assigns_characters_wardrobes_and_two_shots():
    beats = [round(0.5 * i, 3) for i in range(32)]
    lines = [
        LyricLine(
            text="Forgive the snow",
            section="Verse 1: Elias, breathless",
            start=0.0,
            end=4.0,
            timing_source="aligned",
        ),
        LyricLine(
            text="You were late",
            section="Odalie, young, hurt",
            start=4.0,
            end=8.0,
            timing_source="aligned",
        ),
        LyricLine(
            text="And not a moment longer",
            section="Odalie, young, hurt",
            start=8.0,
            end=12.0,
            timing_source="aligned",
        ),
        LyricLine(
            text="Wait for me",
            section="Chorus: Elias",
            start=12.0,
            end=16.0,
            timing_source="aligned",
        ),
    ]
    duet_analysis = Analysis(
        duration=16.0,
        tempo_bpm=120,
        beats=beats,
        bars=beats[::4],
        bar_intensity=[0.5] * 8,
        vocal_windows=[TimeSpan(start=0.0, end=16.0)],
        lyric_lines=lines,
    )

    class DuetDirector:
        def complete_json(self, *, system, content, schema):
            if schema is BibleReply:
                return BibleReply(
                    style="gothic",
                    palette="warm gold",
                    lighting="candlelight",
                    negative_prompt="blurry",
                    characters=[
                        CharacterReply(
                            id="Elias",
                            name="Elias",
                            appearance="Elias, a 22-year-old male with green eyes.",
                            wardrobe=[
                                WardrobeReply(id="coat", description="Velvet coat."),
                                WardrobeReply(id="tux", description="Black tuxedo. No coat."),
                            ],
                        ),
                        CharacterReply(
                            id="Odalie",
                            name="Odalie",
                            appearance="Odalie, a 20-year-old female with dark curls.",
                            wardrobe=[
                                WardrobeReply(id="gown", description="Ivory silk gown."),
                            ],
                        ),
                    ],
                    settings=[SetReply(name="Drawing Room", description="an antique room")],
                )
            return ShotBatchReply(
                shots=[
                    ShotCreative(
                        id="s001",
                        type=ShotType.LIPSYNC,
                        framing="medium_close",
                        setting="Drawing Room",
                        character="elias",
                        wardrobe="coat",
                        still_prompt="Elias on the stairs.",
                        motion_prompt="Elias climbs.",
                    ),
                    ShotCreative(
                        id="s002",
                        type=ShotType.LIPSYNC,
                        framing="medium_close",
                        setting="Drawing Room",
                        character="odalie",
                        characters=["odalie", "elias"],
                        wardrobe="gown",
                        still_prompt="Odalie by the window as Elias approaches.",
                        motion_prompt="Odalie sings to Elias.",
                    ),
                    # Model mistakenly writes Elias in a close-up during Odalie's section:
                    ShotCreative(
                        id="s003",
                        type=ShotType.LIPSYNC,
                        framing="close_up",
                        setting="Drawing Room",
                        character="elias",
                        wardrobe="coat",
                        still_prompt="Close-up of Elias singing.",
                        motion_prompt="Elias sings softly.",
                    ),
                    ShotCreative(
                        id="s004",
                        type=ShotType.LIPSYNC,
                        framing="close_up",
                        setting="Drawing Room",
                        character="elias",
                        wardrobe="tux",
                        still_prompt="Close-up of Elias.",
                        motion_prompt="Elias sings.",
                    ),
                ]
            )

    config = ProjectConfig()
    config.planning.max_lipsync_share = None  # this scenario needs every sung shot on camera
    config.video.max_clip_seconds = 4.0
    manifest = build_manifest(config, duet_analysis, "lyrics", "style", DuetDirector())

    assert list(manifest.characters) == ["elias", "odalie"]
    assert manifest.character.appearance.startswith("Elias")

    s1, s2, s3, s4 = manifest.shots
    assert (s1.character, s1.wardrobe, s1.characters) == ("elias", "coat", ["elias"])
    # s002 is a two-shot led by Odalie, with Elias also visible in his current coat state.
    assert (s2.character, s2.wardrobe, s2.characters) == ("odalie", "gown", ["odalie", "elias"])
    assert s2.character_wardrobes == {"odalie": "gown", "elias": "coat"}
    # s003 is corrected to Odalie because the section explicitly names Odalie.
    assert (s3.character, s3.wardrobe, s3.characters) == ("odalie", "gown", ["odalie"])
    assert "Odalie" in s3.still_prompt and "Elias" not in s3.still_prompt
    # s004 advances Elias's wardrobe to tux independently of Odalie's gown.
    assert (s4.character, s4.wardrobe) == ("elias", "tux")

    # Still prompt composition includes both characters on two-shots and only
    # the active singer on solos.
    two_shot_prompt = prompts.compose_still_prompt(manifest, s2)
    assert "The singer (Odalie, matches reference portrait): Odalie" in two_shot_prompt
    assert "Also in frame (Elias): Elias" in two_shot_prompt and "Velvet coat." in two_shot_prompt
    solo_odalie_prompt = prompts.compose_still_prompt(manifest, s3)
    assert "The singer (Odalie): Odalie" in solo_odalie_prompt
    assert "Elias" not in solo_odalie_prompt


def test_character_resolution_and_proper_noun_prompt_rewrite():
    from mvdirector.models import AudioSlice, Character, Shot, ShotType, WardrobeState
    from mvdirector.planning.planner import _fix_mismatched_singer_prompt

    chars = {
        "elias": Character(
            name="Elias Vance",
            appearance="Elias.",
            wardrobe=[WardrobeState(id="w1", description="coat")],
        ),
        "grace": Character(
            name="Grace",
            appearance="Grace.",
            wardrobe=[WardrobeState(id="w1", description="dress")],
        ),
    }
    manifest = Manifest(title="Duet", duration=8.0, fps=24, characters=chars)
    assert manifest.resolve_character_id("a") is None
    assert manifest.resolve_character_id("li") is None
    assert manifest.resolve_character_id("unknown") is None
    assert manifest.resolve_character_id("elias-vance") == "elias"
    assert manifest.resolve_character_id("Elias Vance") == "elias"
    assert manifest.resolve_character_id("Elias") == "elias"
    assert manifest.resolve_character_id("Grace") == "grace"

    shot = Shot(
        id="s001",
        index=0,
        start=0.0,
        end=4.0,
        type=ShotType.LIPSYNC,
        audio_slice=AudioSlice(
            path="assets/audio_slices/s001.wav", start=0.0, end=4.0, head_handle=0.0
        ),
        num_frames=97,
        still_prompt="Grace walks across the floor with quiet grace.",
        motion_prompt="Grace turns slowly with effortless grace.",
    )
    _fix_mismatched_singer_prompt(shot, "elias", chars)
    assert shot.still_prompt == "Elias Vance walks across the floor with quiet grace."
    assert shot.motion_prompt == "Elias Vance turns slowly with effortless grace."


def test_cut_points_are_deterministic_before_bible_call():
    from mvdirector.planning.planner import has_multiple_section_singers

    beats = [round(0.5 * i, 3) for i in range(48)]
    # Section change at 10s (not a multiple of 8s): uniform pacing cuts [0, 8, 16, 24];
    # section-snapped pacing cuts at 10s too.
    solo_analysis = Analysis(
        duration=24.0,
        tempo_bpm=120,
        beats=beats,
        bars=beats[::4],
        bar_intensity=[0.5] * 12,
        vocal_windows=[TimeSpan(start=0.0, end=24.0)],
        lyric_lines=[
            LyricLine(text="a", section="Verse 1", start=0.0, end=10.0, timing_source="aligned"),
            LyricLine(
                text="b", section="Chorus - big hook", start=10.0, end=24.0, timing_source="aligned"
            ),
        ],
    )
    duet_analysis = solo_analysis.model_copy(
        update={
            "lyric_lines": [
                LyricLine(
                    text="a",
                    section="Verse 1: Elias, breathless",
                    start=0.0,
                    end=10.0,
                    timing_source="aligned",
                ),
                LyricLine(
                    text="b",
                    section="Chorus: Odalie",
                    start=10.0,
                    end=24.0,
                    timing_source="aligned",
                ),
            ]
        }
    )
    assert not has_multiple_section_singers(solo_analysis, "[Verse 1]\na\n[Chorus - big hook]\nb\n")
    assert has_multiple_section_singers(duet_analysis)

    class TwoCharLLM:
        def complete_json(self, *, system, content, schema):
            if schema is BibleReply:
                return BibleReply(
                    style="s",
                    palette="p",
                    lighting="l",
                    negative_prompt="n",
                    characters=[
                        CharacterReply(id="elias", name="Elias", appearance="Elias."),
                        CharacterReply(id="odalie", name="Odalie", appearance="Odalie."),
                    ],
                    settings=[SetReply(name="Room", description="a room")],
                )
            return ShotBatchReply(shots=[])

    cfg = ProjectConfig()
    # Even if an LLM returns 2 characters on a solo-tagged song, cut points stay identical
    # to llm=None because timing runs before the bible call.
    solo_none = build_manifest(cfg, solo_analysis, "[Verse 1]\na\n", "style", llm=None)
    solo_two = build_manifest(cfg, solo_analysis, "[Verse 1]\na\n", "style", llm=TwoCharLLM())
    assert [(s.start, s.end) for s in solo_none.shots] == [(s.start, s.end) for s in solo_two.shots]
    assert [(s.start, s.end) for s in solo_none.shots] == [(0.0, 8.0), (8.0, 16.0), (16.0, 24.0)]

    # Duet section tags deterministically enable section snapping regardless of LLM output.
    duet_none = build_manifest(cfg, duet_analysis, "lyrics", "style", llm=None)
    assert any(s.start == 10.0 for s in duet_none.shots)


def test_pinned_details_reach_every_character_in_a_still_prompt():
    def shot(shot_id, character, characters):
        return Shot(
            id=shot_id,
            index=0,
            start=0.0,
            end=4.0,
            type=ShotType.LIPSYNC,
            framing="medium_close",
            setting="Hall",
            character=character,
            characters=characters,
            still_prompt="A two-shot.",
            motion_prompt="They sing.",
            audio_slice=AudioSlice(path="a.wav", start=0.0, end=4.0, head_handle=0.0),
            num_frames=97,
        )

    coat = [WardrobeState(id="coat", description="A coat.")]
    manifest = Manifest(
        title="Duet",
        duration=4.0,
        fps=24,
        visual_bible=VisualBible(),
        characters={
            "odalie": Character(
                name="Odalie",
                appearance="Odalie, fair.",
                invariants="blonde hair in a low loose bun",
                wardrobe=coat,
            ),
            "elias": Character(
                name="Elias",
                appearance="Elias, lean.",
                invariants="very dark, tight curly hair.",
                wardrobe=coat,
            ),
            "ned": Character(name="Ned", appearance="Ned, small.", wardrobe=coat),
        },
        shots=[shot("s1", "odalie", ["odalie", "elias"]), shot("s2", "elias", ["elias"])],
    )
    # The lead and the second character both get their pinned details, trailing period normalized.
    duet = prompts.compose_still_prompt(manifest, manifest.shots[0])
    assert "Always: blonde hair in a low loose bun." in duet
    assert "Always: very dark, tight curly hair." in duet
    # A solo shot pins only its own character.
    solo = prompts.compose_still_prompt(manifest, manifest.shots[1])
    assert "very dark, tight curly hair" in solo and "blonde hair" not in solo
    # A character with nothing pinned adds no empty "Always:" note.
    manifest.shots[1].character, manifest.shots[1].characters = "ned", ["ned"]
    assert "Always" not in prompts.compose_still_prompt(manifest, manifest.shots[1])


def test_pinned_details_edge_cases():
    coat = [WardrobeState(id="coat", description="a long black coat")]
    # A costume with no closing punctuation does not swallow the pinned details.
    worn = Character(appearance="Fair.", invariants="blonde hair in a bun", wardrobe=coat)
    assert prompts._look(worn, "coat") == (
        "Fair. Wearing: a long black coat. Always: blonde hair in a bun."
    )
    # Pinned details need a look to attach to, so they alone say nothing.
    assert prompts._look(Character(invariants="a scar"), None) == ""
    assert prompts._look(Character(appearance="Fair.", invariants=" . "), None) == "Fair."
    assert prompts._look(Character(), None) == ""
    # Trailing sentence punctuation is trimmed; a leading period or a question mark is kept.
    assert Character(invariants=".5mm nose stud.").pinned() == ".5mm nose stud"
    assert Character(invariants="tattoo reading 'Why?'").pinned() == "tattoo reading 'Why?'"
    # A costume with no appearance is not worth a prompt line on its own.
    assert prompts._look(Character(wardrobe=coat), "coat") == ""
    # The base sheet carries the pinned details too, so the locked identity includes them.
    sheet = Manifest(
        title="t",
        duration=1.0,
        fps=24,
        visual_bible=VisualBible(),
        characters={"a": Character(appearance="Fair.", invariants="a scar", wardrobe=coat)},
    )
    assert "Always: a scar." in prompts.sheet_prompt(sheet, "row")
    # A derived (variant) sheet changes the look on purpose, so it is not pinned.
    assert "Always" not in prompts.derived_sheet_prompt(sheet, "row")
    # Existing sentence punctuation is kept rather than doubled.
    marked = Character(appearance="Fair.", invariants="tattoo that says WHY?")
    assert prompts._look(marked, None) == "Fair. Always: tattoo that says WHY?"
    # The vision judge hears the pinned details of every character, including in a montage.
    pinned = Manifest(
        title="t",
        duration=1.0,
        fps=24,
        visual_bible=VisualBible(),
        characters={"a": Character(appearance="Fair.", invariants="a scar", wardrobe=coat)},
    )
    shot = Shot(
        id="s1",
        index=0,
        start=0.0,
        end=1.0,
        type=ShotType.LIPSYNC,
        framing="close_up",
        setting="Hall",
        character="a",
        characters=["a"],
        still_prompt="x",
        motion_prompt="y",
        audio_slice=AudioSlice(path="a.wav", start=0.0, end=1.0, head_handle=0.0),
        num_frames=25,
        allow_cuts=True,
    )
    from mvdirector.qc import vision

    assert "a scar" in vision.VisionJudge.specification(shot, pinned)


def test_off_bar_duet_handoff_cuts_in_vocal_gap_and_maps_acoustic_voices():
    from mvdirector.models import Framing, VocalTurn

    # Bars every 2s [0, 2, ..., 16]. Elias sings 0.5-8.6s; Odalie enters mid-bar at 8.8-15.5s!
    # Snapping to nearest bar (8.0s or 10.0s) would bleed one singer's voice into the other's shot.
    beats = [round(0.5 * i, 3) for i in range(32)]
    duet_analysis = Analysis(
        duration=16.0,
        tempo_bpm=120.0,
        beats=beats,
        bars=beats[::4],
        bar_intensity=[0.6] * 8,
        vocal_windows=[TimeSpan(start=0.5, end=15.5)],
        vocal_turns=[
            VocalTurn(start=0.5, end=8.6, voice="voice_1", median_f0_hz=130.0),
            VocalTurn(start=8.8, end=15.5, voice="voice_2", median_f0_hz=280.0),
        ],
        lyric_lines=[
            LyricLine(
                text="Wait in the snow",
                section="Chorus",
                start=0.5,
                end=4.2,
                timing_source="aligned",
            ),
            LyricLine(
                text="Hold the door",
                section="Chorus",
                start=4.4,
                end=8.6,
                timing_source="aligned",
            ),
            LyricLine(
                text="Too late tonight",
                section="Chorus",
                start=8.8,
                end=12.0,
                timing_source="aligned",
            ),
            LyricLine(
                text="Turn out the light",
                section="Chorus",
                start=12.2,
                end=15.5,
                timing_source="aligned",
            ),
        ],
    )

    class UntaggedChorusDirector:
        def complete_json(self, *, system, content, schema):
            if schema is BibleReply:
                return BibleReply(
                    style="gothic",
                    palette="cool",
                    lighting="moonlight",
                    negative_prompt="blurry",
                    characters=[
                        CharacterReply(
                            id="elias",
                            name="Elias",
                            appearance="22-year-old male baritone with dark hair.",
                            wardrobe=[WardrobeReply(id="coat", description="Dark coat.")],
                        ),
                        CharacterReply(
                            id="odalie",
                            name="Odalie",
                            appearance="20-year-old female soprano with curls.",
                            wardrobe=[WardrobeReply(id="gown", description="Silk gown.")],
                        ),
                    ],
                    settings=[SetReply(name="Manor", description="a Victorian hall")],
                )
            ids = re.findall(r"- id=(s\d+)", content[0]["text"])
            return ShotBatchReply(
                shots=[
                    ShotCreative(
                        id=sid,
                        type=ShotType.LIPSYNC,
                        framing=Framing.MEDIUM_CLOSE,
                        setting="Manor",
                        still_prompt="Close singing shot.",
                        motion_prompt="Sings softly.",
                    )
                    for sid in ids
                ]
            )

    cfg = ProjectConfig()
    cfg.video.max_clip_seconds = 6.0
    cfg.planning.max_lipsync_share = None
    manifest = build_manifest(cfg, duet_analysis, "lyrics", "style", UntaggedChorusDirector())

    # Exact handoff cut lands inside [8.6, 8.8] (at 8.7s), never on bar 8.0 or bar 10.0!
    boundaries = [s.start for s in manifest.shots]
    assert 8.7 in boundaries
    elias_shots = [s for s in manifest.shots if s.end <= 8.7]
    odalie_shots = [s for s in manifest.shots if s.start >= 8.7]
    assert all(s.character == "elias" for s in elias_shots)
    assert all(s.character == "odalie" for s in odalie_shots)
    assert all("Too late" not in s.lyrics for s in elias_shots)
    assert all("Hold the door" not in s.lyrics for s in odalie_shots)
    # Boundary shots record mute_spans across the 8.7s handoff so handles never bleed.
    last_elias = elias_shots[-1]
    first_odalie = odalie_shots[0]
    assert any(m.start <= 8.8 and m.end > 8.7 for m in last_elias.audio_slice.mute_spans)
    assert any(m.start < 8.6 and m.end >= 8.45 for m in first_odalie.audio_slice.mute_spans)


def test_solo_vs_listener_two_shot_vs_both_singing_together_end_to_end():
    from mvdirector.models import Framing, LyricLine, VocalTurn
    from mvdirector.planning.segmenter import dominant_voice

    # External diarizers that emit overlapping solo turns also resolve to 'both'.
    overlapping = [
        VocalTurn(start=8.0, end=12.0, voice="voice_1", median_f0_hz=130.0),
        VocalTurn(start=8.0, end=12.0, voice="voice_2", median_f0_hz=280.0),
    ]
    assert dominant_voice(overlapping, 8.0, 12.0) == "both"

    beats = [round(0.5 * i, 3) for i in range(32)]
    bars = beats[::4]  # 0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0 (duration 16.0)
    duet_analysis = Analysis(
        duration=16.0,
        tempo_bpm=120.0,
        beats=beats,
        bars=bars,
        bar_intensity=[0.5] * len(bars),
        vocal_windows=[TimeSpan(start=0.2, end=15.8)],
        vocal_turns=[
            VocalTurn(start=0.2, end=3.9, voice="voice_1", median_f0_hz=130.0),
            VocalTurn(start=4.1, end=7.9, voice="voice_2", median_f0_hz=280.0),
            VocalTurn(start=8.1, end=11.9, voice="both", median_f0_hz=None),
            # Inside [Chorus: Both], Elias sings the final 4s adlib completely solo:
            VocalTurn(start=12.1, end=15.8, voice="voice_1", median_f0_hz=132.0),
        ],
        lyric_lines=[
            LyricLine(
                text="Cold iron gate",
                section="Verse 1: Elias",
                start=0.3,
                end=3.8,
                timing_source="aligned",
            ),
            LyricLine(
                text="Candle in the hall",
                section="Verse 2: Odalie",
                start=4.2,
                end=7.8,
                timing_source="aligned",
            ),
            LyricLine(
                text="Hold back the dawn",
                section="Chorus: Both",
                start=8.2,
                end=11.8,
                timing_source="aligned",
            ),
            LyricLine(
                text="Stay with me now",
                section="Chorus: Both",
                start=12.2,
                end=15.7,
                timing_source="aligned",
            ),
        ],
    )
    batch_requests: list[str] = []

    class TogetherDirector:
        def complete_json(self, *, system, content, schema):
            if schema is BibleReply:
                return BibleReply(
                    style="gothic cinema",
                    palette="slate and amber",
                    lighting="candlelight",
                    negative_prompt="blurry",
                    characters=[
                        CharacterReply(
                            id="elias",
                            name="Elias",
                            appearance="24-year-old male baritone.",
                            wardrobe=[WardrobeReply(id="coat", description="Dark wool coat.")],
                        ),
                        CharacterReply(
                            id="odalie",
                            name="Odalie",
                            appearance="23-year-old female soprano.",
                            wardrobe=[WardrobeReply(id="gown", description="Ivory silk gown.")],
                        ),
                    ],
                    settings=[SetReply(name="Hall", description="grand staircase")],
                )
            text = content[0]["text"]
            batch_requests.append(text)
            return ShotBatchReply(
                shots=[
                    ShotCreative(
                        id="s001",
                        type=ShotType.LIPSYNC,
                        framing=Framing.CLOSE_UP,
                        setting="Hall",
                        character="elias",
                        characters=["elias"],
                        still_prompt="Close-up of Elias at the gate.",
                        motion_prompt="Elias sings steadily.",
                    ),
                    ShotCreative(
                        id="s002",
                        type=ShotType.LIPSYNC,
                        framing=Framing.MEDIUM_CLOSE,
                        setting="Hall",
                        character="odalie",
                        characters=["odalie", "elias"],
                        still_prompt="Odalie on the stairs while Elias watches.",
                        motion_prompt="Odalie sings as Elias stands nearby.",
                    ),
                    # Model mistakenly picks close_up and omits Odalie on a simultaneous duet shot:
                    ShotCreative(
                        id="s003",
                        type=ShotType.LIPSYNC,
                        framing=Framing.CLOSE_UP,
                        setting="Hall",
                        character="elias",
                        characters=["elias"],
                        still_prompt="Elias and Odalie singing.",
                        motion_prompt="They sing together.",
                    ),
                    ShotCreative(
                        id="s004",
                        type=ShotType.LIPSYNC,
                        framing=Framing.CLOSE_UP,
                        setting="Hall",
                        character="elias",
                        characters=["elias"],
                        still_prompt="Close-up of Elias singing the final line.",
                        motion_prompt="Elias sings quietly.",
                    ),
                ]
            )

    cfg = ProjectConfig()
    cfg.video.max_clip_seconds = 4.5
    cfg.planning.max_lipsync_share = None
    manifest = build_manifest(cfg, duet_analysis, "lyrics", "style", TogetherDirector())

    assert len(batch_requests) == 1
    assert "vocal='solo' singer='elias'" in batch_requests[0]
    assert "vocal='solo' singer='odalie'" in batch_requests[0]
    assert "vocal='together' singers=['elias', 'odalie']" in batch_requests[0]

    s1, s2, s3, s4 = manifest.shots
    # 1. Solo Elias
    assert (s1.character, s1.characters, s1.singers) == ("elias", ["elias"], ["elias"])
    assert "Elias sings with clear, natural mouth movement" in prompts.compose_motion_prompt(
        manifest, s1
    )

    # 2. Solo Odalie in a two-shot with Elias listening silently
    assert (s2.character, s2.characters, s2.singers) == ("odalie", ["odalie", "elias"], ["odalie"])
    s2_still = prompts.compose_still_prompt(manifest, s2)
    s2_motion = prompts.compose_motion_prompt(manifest, s2)
    assert "Only Odalie is singing; Elias listens with mouth closed." in s2_still
    assert "Also in frame (Elias):" in s2_still
    assert "while Elias listens with mouth closed, not singing." in s2_motion

    # 3. Both singing together: widened to MEDIUM_CLOSE, both in characters & singers
    assert s3.framing is Framing.MEDIUM_CLOSE
    assert s3.characters == ["elias", "odalie"]
    assert s3.singers == ["elias", "odalie"]
    s3_still = prompts.compose_still_prompt(manifest, s3)
    s3_motion = prompts.compose_motion_prompt(manifest, s3)
    assert "Both Elias and Odalie singing together" in s3_still
    assert "Co-singer in frame (Odalie):" in s3_still
    assert "Both Elias and Odalie sing together with clear, natural mouth movement" in s3_motion
    # Together shot keeps [8.0, 12.0] unmuted while muting adjacent solo handles!
    assert all(m.end <= s3.start or m.start >= s3.end for m in s3.audio_slice.mute_spans)
    assert any(m.start < s3.start for m in s3.audio_slice.mute_spans)
    assert any(m.end > s3.end for m in s3.audio_slice.mute_spans)

    # 4. Solo adlib inside [Chorus: Both] overridden by 100% voice_1 dominance
    assert (s4.character, s4.singers) == ("elias", ["elias"])

    # Unsafe singer identifiers are strictly rejected at deserialization.
    with pytest.raises(ValueError, match="Invalid character id"):
        s3.model_validate({**s3.model_dump(), "singers": ["../evil"]})
