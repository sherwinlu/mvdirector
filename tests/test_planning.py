# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import pytest

from mvdirector.config import ProjectConfig, VideoConfig
from mvdirector.models import Analysis, Framing, LyricLine, ShotType, TimeSpan
from mvdirector.planning.planner import (
    BibleReply,
    SetReply,
    ShotBatchReply,
    ShotCreative,
    build_manifest,
)
from mvdirector.planning.segmenter import cut_points, frames_for, segment


@pytest.mark.parametrize(
    ("duration", "expected"), [(1.0, 25), (8.5, 209), (0.1, 9), (8.0, 193), (8.05, 201)]
)
def test_frames_for_uses_ltx_frame_rule(duration, expected):
    frames = frames_for(duration, fps=24, multiple=8, offset=1)
    assert frames == expected
    assert (frames - 1) % 8 == 0 and frames >= duration * 24 - 1e-6


def test_segments_cover_song_on_bar_lines(analysis):
    video = VideoConfig()
    segments = segment(analysis, video)
    assert segments[0].start == 0.0 and segments[-1].end == analysis.duration
    assert all(a.end == b.start for a, b in zip(segments, segments[1:]))
    assert all(s.duration <= video.max_clip_seconds + 1e-6 for s in segments)
    assert all(s.start in analysis.bars for s in segments)
    # 2 s bars and an 8 s limit give 4-bar shots.
    assert [s.duration for s in segments] == [8.0, 8.0, 8.0]


def test_dynamic_pacing_cuts_faster_on_high_intensity_and_snaps_to_sections():
    bars = [2.0 * i for i in range(12)]
    # Bars 0-3 calm (0.3), bars 4-7 peak chorus (0.95), bars 8-11 medium bridge (0.65).
    intensities = [0.3] * 4 + [0.95] * 4 + [0.65] * 4
    analysis = Analysis(
        duration=24.0,
        tempo_bpm=120.0,
        beats=[0.5 * i for i in range(48)],
        bars=bars,
        bar_intensity=intensities,
        vocal_windows=[TimeSpan(start=0.0, end=24.0)],
        lyric_lines=[
            LyricLine(text="v1", section="Verse", start=0.0, end=4.0, timing_source="aligned"),
            LyricLine(text="ch", section="Chorus", start=8.0, end=12.0, timing_source="aligned"),
            LyricLine(text="br", section="Bridge", start=16.0, end=20.0, timing_source="aligned"),
        ],
    )
    video = VideoConfig(max_clip_seconds=8.0, min_clip_seconds=2.0, dynamic_pacing=True)
    segments = segment(analysis, video)
    # Verse holds 4 bars (8s), Chorus cuts every bar (2s), Bridge cuts every 2 bars (4s).
    assert [s.duration for s in segments] == [8.0, 2.0, 2.0, 2.0, 2.0, 4.0, 4.0]
    assert all(s.start in bars for s in segments)


def test_fixed_pacing_stays_uniform_across_sections_by_default():
    bars = [2.0 * i for i in range(16)]
    analysis = Analysis(
        duration=32.0,
        tempo_bpm=120.0,
        beats=[0.5 * i for i in range(64)],
        bars=bars,
        bar_intensity=[0.5] * 16,
        vocal_windows=[TimeSpan(start=0.0, end=32.0)],
        lyric_lines=[
            LyricLine(text="v1", section="Verse", start=0.0, end=8.0, timing_source="aligned"),
            # Section starts at 10s (non-multiple of 8s); dynamic_pacing=False must stay uniform.
            LyricLine(text="ch", section="Chorus", start=10.0, end=20.0, timing_source="aligned"),
        ],
    )
    video = VideoConfig(max_clip_seconds=8.0, min_clip_seconds=2.0, dynamic_pacing=False)
    assert cut_points(analysis, video) == [0.0, 8.0, 16.0, 24.0, 32.0]


def test_segment_types_follow_vocal_activity(analysis):
    video = VideoConfig(max_clip_seconds=4.0)
    types = [s.type for s in segment(analysis, video)]
    assert types == [ShotType.PERFORMANCE] + [ShotType.LIPSYNC] * 3 + [ShotType.PERFORMANCE] * 2


def test_each_lyric_line_lands_in_exactly_one_shot(analysis):
    segments = segment(analysis, VideoConfig(max_clip_seconds=4.0))
    assert [s.lyrics for s in segments] == ["", "first line", "second line", "third line", "", ""]


def test_pickup_before_first_bar_does_not_create_a_sliver():
    analysis = Analysis(
        duration=16.3,
        tempo_bpm=120,
        beats=[],
        bars=[0.3 + 2 * i for i in range(8)],
        bar_intensity=[1.0] * 8,
        vocal_windows=[TimeSpan(start=0.0, end=16.3)],
    )
    video = VideoConfig()
    points = cut_points(analysis, video)
    lengths = [b - a for a, b in zip(points, points[1:])]
    assert points[0] == 0.0 and points[-1] == 16.3
    assert min(lengths) >= video.min_clip_seconds
    assert max(lengths) <= video.max_clip_seconds + 1e-6


def test_no_beat_grid_falls_back_to_equal_splits():
    analysis = Analysis(
        duration=20.0, tempo_bpm=0, beats=[], bars=[], bar_intensity=[], vocal_windows=[]
    )
    assert cut_points(analysis, VideoConfig()) == [0.0, 6.667, 13.333, 20.0]


def test_rule_based_manifest_is_complete_without_llm(analysis):
    config = ProjectConfig()
    manifest = build_manifest(config, analysis, "lyrics", "neon style", llm=None)
    assert [s.id for s in manifest.shots] == ["s001", "s002", "s003"]
    for shot in manifest.shots:
        assert shot.still_prompt and shot.motion_prompt
        assert shot.still.attempts_allowed == config.retry.max_attempts
        assert (shot.num_frames - 1) % 8 == 0
        assert shot.num_frames / config.video.fps >= shot.audio_slice.end - shot.audio_slice.start
    first, last = manifest.shots[0], manifest.shots[-1]
    assert first.audio_slice.start == 0.0 and first.audio_slice.head_handle == 0.0
    assert last.audio_slice.end == analysis.duration
    assert manifest.shots[1].audio_slice.head_handle == config.video.handle_seconds


class ScriptedLLM:
    """Returns canned replies; tries to break the rules the planner must enforce."""

    def __init__(self, framing=Framing.WIDE):
        self._framing = framing

    def complete_json(self, *, system, content, schema):
        if schema is BibleReply:
            return BibleReply(
                appearance="A singer",
                style="neon",
                palette="teal",
                lighting="hard",
                negative_prompt="blurry",
                settings=[SetReply(name="rooftop", description="a rain-slick rooftop")],
            )
        assert schema is ShotBatchReply
        return ShotBatchReply(
            shots=[
                # Tries to turn a sung shot into a wide b-roll shot.
                ShotCreative(
                    id="s002",
                    type=ShotType.BROLL,
                    framing=self._framing,
                    setting="rooftop",
                    still_prompt="still two",
                    motion_prompt="motion two",
                ),
                ShotCreative(
                    id="s005",
                    type=ShotType.BROLL,
                    framing=Framing.WIDE,
                    setting="street",
                    still_prompt="still five",
                    motion_prompt="motion five",
                ),
            ]
        )


def test_llm_direction_is_applied_within_rules(analysis):
    config = ProjectConfig()
    config.planning.max_lipsync_share = None  # this scenario needs every sung shot on camera
    config.video.max_clip_seconds = 4.0
    manifest = build_manifest(config, analysis, "lyrics", "style", llm=ScriptedLLM())
    assert manifest.character.appearance == "A singer"
    sung, cutaway, omitted = manifest.shot("s002"), manifest.shot("s005"), manifest.shot("s001")
    # A lip-sync shot stays lip-sync and is tightened to medium close-up at the widest.
    assert sung.type is ShotType.LIPSYNC and sung.framing is Framing.MEDIUM_CLOSE
    assert sung.still_prompt == "still two"
    # A shot with no vocals may become a wide b-roll cutaway.
    assert cutaway.type is ShotType.BROLL and cutaway.framing is Framing.WIDE
    # Shots the model skipped still get usable rule-based direction.
    assert omitted.still_prompt and omitted.motion_prompt


# ----------------------------------------------- tolerance for how a model spells the choices


@pytest.mark.parametrize(
    ("written", "type_", "framing"),
    [
        ("Lip-sync", ShotType.LIPSYNC, Framing.CLOSE_UP),
        ("lip sync", ShotType.LIPSYNC, Framing.MEDIUM),
        ("B-roll", ShotType.BROLL, Framing.WIDE),
        ("PERFORMANCE", ShotType.PERFORMANCE, Framing.MEDIUM_CLOSE),
    ],
)
def test_the_shot_reply_tolerates_natural_spellings(written, type_, framing):
    spelled = {
        Framing.CLOSE_UP: "Close-up",
        Framing.MEDIUM: "Medium shot",
        Framing.WIDE: "wide",
        Framing.MEDIUM_CLOSE: "Medium close-up",
    }[framing]
    creative = ShotCreative(
        id="s001", type=written, framing=spelled, setting="x", still_prompt="a", motion_prompt="b"
    )
    assert (creative.type, creative.framing) == (type_, framing)


def test_the_shot_reply_still_rejects_values_that_mean_nothing():
    with pytest.raises(ValueError, match="framing"):
        ShotCreative(
            id="s001", type="lipsync", framing="dutch tilt", setting="x",
            still_prompt="a", motion_prompt="b",
        )  # fmt: skip


def test_phrase_cut_mode_and_exact_frame_rules_for_other_video_models(analysis):
    from mvdirector.planning.segmenter import cut_points, frames_for

    config = ProjectConfig().video
    config.cut_mode = "phrases"
    config.min_clip_seconds = 3.0
    config.max_clip_seconds = 12.0
    points = cut_points(analysis, config)
    # Vocal window is 4.0s to 16.0s; phrase mode cuts at 4.0s and 16.0s.
    assert points == [0.0, 4.0, 16.0, 24.0]
    # Wan (4n+1) and MiniMax/exact (1n+0) frame rules.
    assert frames_for(4.0, fps=24, multiple=4, offset=1) == 97
    assert frames_for(4.0, fps=24, multiple=1, offset=0) == 96


def test_rule_based_singing_shots_are_never_wider_than_medium_close(analysis):
    config = ProjectConfig()
    config.video.max_clip_seconds = 4.0
    manifest = build_manifest(config, analysis, "lyrics", "style", llm=None)
    sung = [s for s in manifest.shots if s.type is ShotType.LIPSYNC]
    assert sung and all(s.framing in (Framing.CLOSE_UP, Framing.MEDIUM_CLOSE) for s in sung)


def test_the_widest_singing_framing_is_configurable(analysis):
    config = ProjectConfig()
    config.planning.max_lipsync_share = None  # keep s002 as a lipsync shot
    config.video.max_clip_seconds = 4.0
    config.planning.lipsync_widest_framing = Framing.MEDIUM
    manifest = build_manifest(config, analysis, "lyrics", "style", llm=ScriptedLLM(Framing.MEDIUM))
    sung = manifest.shot("s002")
    assert sung.type is ShotType.LIPSYNC and sung.framing is Framing.MEDIUM


def test_broll_guard_blocks_named_cast_but_keeps_anonymous_crowd_silhouettes():
    from mvdirector.models import AudioSlice, Character, SetDesign, Shot, VisualBible
    from mvdirector.planning.planner import _apply_creative

    bible = VisualBible(settings=[SetDesign(name="Neon Club", description="Dark neon hall.")])
    chars = {
        "lead_vocal": Character(name="Seraphina", appearance="Dark hair."),
        "vocalist_1": Character(name="Anya", appearance="Blonde hair."),
    }

    def _make_shot(shot_type: ShotType) -> Shot:
        return Shot(
            id="s010",
            index=10,
            start=10.0,
            end=14.0,
            audio_slice=AudioSlice(path="audio.wav", start=10.0, end=14.0, head_handle=0.0),
            num_frames=97,
            type=shot_type,
        )

    # 1. Cast member named in a broll slot -> falls back to uninhabited cutaway.
    broll_cast = _make_shot(ShotType.BROLL)
    _apply_creative(
        broll_cast,
        ShotCreative(
            id="s010",
            type=ShotType.BROLL,
            framing=Framing.CLOSE_UP,
            setting="Neon Club",
            still_prompt="Close-up on Anya's face bathed in pink light.",
            motion_prompt="Slow push-in.",
        ),
        bible,
        characters=chars,
    )
    assert broll_cast.type is ShotType.BROLL
    assert "Anya" not in broll_cast.still_prompt
    assert "no people" in broll_cast.still_prompt.lower()

    # 2. Raw snake_case character ID in a broll slot -> also caught.
    broll_id = _make_shot(ShotType.BROLL)
    _apply_creative(
        broll_id,
        ShotCreative(
            id="s010",
            type=ShotType.BROLL,
            framing=Framing.MEDIUM,
            setting="Neon Club",
            still_prompt="Medium shot showing lead_vocal watching the dance floor.",
            motion_prompt="Slow pan.",
        ),
        bible,
        characters=chars,
    )
    assert "lead_vocal" not in broll_id.still_prompt

    # 3. Anonymous backlit silhouettes / figurines in a broll slot -> preserved!
    broll_silhouettes = _make_shot(ShotType.BROLL)
    _apply_creative(
        broll_silhouettes,
        ShotCreative(
            id="s010",
            type=ShotType.BROLL,
            framing=Framing.WIDE,
            setting="Neon Club",
            still_prompt="Wide strobe-lit club floor with anonymous backlit dancer silhouettes.",
            motion_prompt="Strobe lights sweep across the silhouettes.",
        ),
        bible,
        characters=chars,
    )
    assert broll_silhouettes.type is ShotType.BROLL
    assert "anonymous backlit dancer silhouettes" in broll_silhouettes.still_prompt

    # 4. Performance slot converted to broll while naming a cast member -> stays performance.
    perf_slot = _make_shot(ShotType.PERFORMANCE)
    _apply_creative(
        perf_slot,
        ShotCreative(
            id="s010",
            type=ShotType.BROLL,
            framing=Framing.MEDIUM,
            setting="Neon Club",
            still_prompt="Spotlight illuminating Anya center stage.",
            motion_prompt="Slow push-in.",
        ),
        bible,
        characters=chars,
    )
    assert perf_slot.type is ShotType.PERFORMANCE
    assert "Anya" in perf_slot.still_prompt


def test_close_and_medium_close_framing_notes_discourage_stiff_mugshot_pose():
    from mvdirector.models import AudioSlice, Manifest, Shot, VisualBible
    from mvdirector.planning.prompts import compose_still_prompt

    manifest = Manifest(
        title="Test",
        duration=10.0,
        fps=24,
        visual_bible=VisualBible(concept="c", style="s", palette="p", lighting="l"),
    )
    for framing in (Framing.CLOSE_UP, Framing.MEDIUM_CLOSE):
        shot = Shot(
            id="s001",
            index=0,
            start=0.0,
            end=4.0,
            type=ShotType.LIPSYNC,
            framing=framing,
            still_prompt="Singer in alley.",
            motion_prompt="Sings.",
            audio_slice=AudioSlice(
                path="assets/audio_slices/s001.wav", start=0.0, end=4.0, head_handle=0.0
            ),
            num_frames=97,
        )
        composed = compose_still_prompt(manifest, shot)
        assert "do not copy the stiff centered frontal pose" in composed
