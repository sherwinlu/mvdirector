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


def test_segment_suppresses_lipsync_on_instrumental_bleed_when_lyrics_aligned():
    bars = [2.0 * i for i in range(12)]
    beats = [0.5 * i for i in range(48)]
    # Demucs synth bleed creates a continuous vocal_window [0, 24], while Whisper aligned
    # lyrics only start at 12.0s.
    analysis = Analysis(
        duration=24.0,
        tempo_bpm=120.0,
        beats=beats,
        bars=bars,
        bar_intensity=[0.6] * len(bars),
        vocal_windows=[TimeSpan(start=0.0, end=24.0)],
        lyric_lines=[
            LyricLine(text="First line", start=12.0, end=16.0, timing_source="aligned"),
            LyricLine(text="Second line", start=16.0, end=20.0, timing_source="aligned"),
        ],
    )
    video = VideoConfig(min_clip_seconds=2.0, max_clip_seconds=4.0)
    segs = segment(analysis, video)
    by_start = {s.start: s for s in segs}
    assert by_start[0.0].type is ShotType.PERFORMANCE
    assert by_start[4.0].type is ShotType.PERFORMANCE
    assert by_start[8.0].type is ShotType.PERFORMANCE
    assert by_start[12.0].type is ShotType.LIPSYNC
    assert by_start[16.0].type is ShotType.LIPSYNC
    assert by_start[20.0].type is ShotType.PERFORMANCE
    # Unaligned / no-lyrics fallback keeps vocal_windows behavior unchanged
    unaligned = analysis.model_copy(update={"lyric_lines": []})
    assert all(s.type is ShotType.LIPSYNC for s in segment(unaligned, video))


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


def test_broll_min_clip_seconds_keeps_sung_shots_long_and_lets_drops_cut_faster():
    # 136 BPM-ish 1.75s bars: bars 0-3 sung (vocal_windows + aligned lyrics),
    # bars 4-7 calm instrumental, bars 8-11 high-intensity instrumental drop.
    bars = [round(1.75 * i, 3) for i in range(12)]
    beats = [round(1.75 / 4 * i, 3) for i in range(48)]
    intensities = [0.95] * 4 + [0.30] * 4 + [0.95] * 4
    analysis = Analysis(
        duration=21.0,
        tempo_bpm=137.1,
        beats=beats,
        bars=bars,
        bar_intensity=intensities,
        vocal_windows=[TimeSpan(start=0.0, end=7.0)],
        lyric_lines=[
            LyricLine(text="Line one", start=0.0, end=3.5, timing_source="aligned"),
            LyricLine(text="Line two", start=3.5, end=7.0, timing_source="aligned"),
        ],
    )
    video = VideoConfig(
        min_clip_seconds=2.5,
        broll_min_clip_seconds=1.5,
        max_clip_seconds=8.0,
        dynamic_pacing=True,
    )
    segs = segment(analysis, video)
    sung = [s for s in segs if s.type is ShotType.LIPSYNC]
    drop = [s for s in segs if s.start >= 14.0]
    calm = [s for s in segs if 7.0 <= s.start < 14.0]
    # Sung shots obey the 2.5s floor (2-bar ~3.5s clips)
    assert sung and all(s.duration >= 2.5 - 1e-3 for s in sung)
    # Calm instrumental section holds long (~7.0s) for pacing contrast
    assert calm and max(s.duration for s in calm) >= 6.5
    # Peak instrumental drop cuts faster down to 1-bar (~1.75s)
    assert drop and all(s.duration < 2.5 for s in drop)

    with pytest.raises(ValueError, match="broll_min_clip_seconds"):
        VideoConfig(min_clip_seconds=2.0, broll_min_clip_seconds=3.0)


def test_assembly_intercut_conserves_frames_protects_lipsync_and_avoids_repeat_cuts(tmp_path):
    from mvdirector.assemble import ClipPlan, intercut_plans
    from mvdirector.models import AudioSlice, ContinuityMode, Manifest, Shot, VisualBible

    fps = 24
    # 136 BPM beats (~0.441s each) over 12s
    beats = [round(0.441 * i, 3) for i in range(1, 28)]
    analysis = Analysis(
        duration=12.0,
        tempo_bpm=136.0,
        beats=beats,
        bars=[0.0, 1.764, 3.528, 5.292, 7.056, 8.82, 10.584],
        bar_intensity=[0.9] * 7,
        vocal_windows=[TimeSpan(start=0.0, end=4.0)],
    )
    clip_a = tmp_path / "a.mp4"
    clip_b = tmp_path / "b.mp4"
    clip_c = tmp_path / "c.mp4"
    for p in (clip_a, clip_b, clip_c):
        p.write_bytes(b"fake")

    shots = [
        Shot(
            id="s001",
            index=0,
            start=0.0,
            end=4.0,
            type=ShotType.LIPSYNC,
            framing=Framing.CLOSE_UP,
            intensity=0.95,
            still_prompt="p",
            motion_prompt="m",
            audio_slice=AudioSlice(path="a.wav", start=0.0, end=4.0, head_handle=0.25),
            num_frames=97,
        ),
        Shot(
            id="s002",
            index=1,
            start=4.0,
            end=8.0,
            type=ShotType.BROLL,
            framing=Framing.WIDE,
            intensity=0.92,
            still_prompt="p",
            motion_prompt="m",
            audio_slice=AudioSlice(path="b.wav", start=4.0, end=8.0, head_handle=0.25),
            num_frames=97,
        ),
        Shot(
            id="s003",
            index=2,
            start=8.0,
            end=12.0,
            type=ShotType.PERFORMANCE,
            framing=Framing.MEDIUM,
            intensity=0.90,
            still_prompt="p",
            motion_prompt="m",
            audio_slice=AudioSlice(path="c.wav", start=8.0, end=12.0, head_handle=0.25),
            num_frames=97,
        ),
    ]
    manifest = Manifest(
        title="Test",
        duration=12.0,
        fps=fps,
        visual_bible=VisualBible(concept="c", style="s", palette="p", lighting="l"),
        shots=shots,
    )
    plans = [
        ClipPlan("s001", 96, clip_a, True, 0.25, tail_handle=0.25),
        ClipPlan("s002", 96, clip_b, True, 0.25, tail_handle=0.25),
        ClipPlan("s003", 96, clip_c, True, 0.25, tail_handle=0.25),
    ]

    out = intercut_plans(plans, manifest, analysis, min_intensity=0.70, anticipation_frames=1)
    # 1. Exact frame conservation
    assert sum(p.frames for p in out) == sum(p.frames for p in plans)
    # 2. Lip-sync shot s001 is completely untouched
    assert out[0] == plans[0]
    # 3. Non-lipsync drop run expanded into balanced 1-bar / half-bar cuts (no <0.5s strobes)
    intercut_sub = out[1:]
    assert len(intercut_sub) >= 4
    assert all(p.frames >= round(0.50 * fps) for p in intercut_sub)
    # 4. No back-to-back jump cuts from the same source clip
    for prev, nxt in zip(intercut_sub[:-1], intercut_sub[1:]):
        assert prev.source != nxt.source
    # 5. Source playback cursors strictly advance forward (no repeated frames)
    last_head: dict[object, float] = {}
    for p in intercut_sub:
        if p.source in last_head:
            assert p.head_handle > last_head[p.source]
        last_head[p.source] = p.head_handle
    # 6. Calm runs below min_intensity stay untouched (contrast preserved!)
    calm_out = intercut_plans(plans, manifest, analysis, min_intensity=0.99)
    assert calm_out == plans
    # 7. CONTINUE chains are never split
    cont_plans = [
        plans[0],
        plans[1],
        ClipPlan(
            "s003", 96, clip_c, True, 0.25, continuity=ContinuityMode.CONTINUE, tail_handle=0.25
        ),
    ]
    assert intercut_plans(cont_plans, manifest, analysis, min_intensity=0.70) == cont_plans


def test_filter_vocal_windows_by_words_drops_bleed_and_trims_edges():
    from mvdirector.audio.analysis import filter_vocal_windows_by_words
    from mvdirector.audio.lyrics import Word

    windows = [
        TimeSpan(start=5.0, end=14.1),  # synth pad with 1 spurious word
        TimeSpan(start=15.0, end=29.8),  # real verse
    ]
    words = [
        Word(word="I", start=5.3, end=6.1),
        Word(word="Static", start=15.3, end=15.9),
        Word(word="in", start=15.9, end=16.2),
        Word(word="skin", start=27.7, end=28.4),
        Word(word="You", start=43.5, end=44.9),  # hallucination in silence
    ]
    filtered = filter_vocal_windows_by_words(windows, words)
    assert filtered == [TimeSpan(start=15.0, end=28.9)]
    # When Whisper found <2 words in every window, keep original windows untouched
    assert filter_vocal_windows_by_words(windows, [Word(word="You", start=43.5, end=44.9)]) == (
        windows
    )


def test_populate_slice_mutes_silences_non_lyric_lead_in_and_preserves_breaths():
    from mvdirector.models import AudioSlice, Manifest, Shot, VisualBible
    from mvdirector.planning.planner import _populate_slice_mutes

    analysis = Analysis(
        duration=20.0,
        tempo_bpm=120.0,
        beats=[0.5 * i for i in range(40)],
        bars=[2.0 * i for i in range(10)],
        bar_intensity=[0.6] * 10,
        vocal_windows=[TimeSpan(start=14.0, end=20.0)],
        lyric_lines=[
            LyricLine(text="First line", start=14.7, end=16.0, timing_source="aligned"),
            LyricLine(text="Next line", start=16.3, end=18.0, timing_source="aligned"),
        ],
    )
    # Shot 14.0-17.0s (slice 13.75-17.25s): lead-in 13.75-14.50s muted, brief 0.3s breath at
    # 16.0-16.3s stays unmuted because +/-0.20s padding bridges it!
    shot = Shot(
        id="s009",
        index=0,
        start=14.0,
        end=17.0,
        type=ShotType.LIPSYNC,
        framing=Framing.MEDIUM_CLOSE,
        still_prompt="Singer.",
        motion_prompt="Sings.",
        audio_slice=AudioSlice(
            path="assets/audio_slices/s009.wav", start=13.75, end=17.25, head_handle=0.25
        ),
        num_frames=85,
    )
    manifest = Manifest(
        title="Test",
        duration=20.0,
        fps=24,
        visual_bible=VisualBible(concept="c", style="s", palette="p", lighting="l"),
        shots=[shot],
    )
    _populate_slice_mutes(manifest, analysis, voice_map={})
    assert shot.audio_slice.mute_spans == [TimeSpan(start=13.75, end=14.5)]
