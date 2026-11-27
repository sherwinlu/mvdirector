# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Regression tests covering all QA audit findings for PR #50."""

from __future__ import annotations

from importlib import metadata
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from mvdirector import assemble as assemble_mod
from mvdirector import media, plugins
from mvdirector.assemble import ClipPlan, intercut_plans
from mvdirector.audio.analysis import filter_vocal_windows_by_words
from mvdirector.audio.lyrics import Word, estimate_missing_timings
from mvdirector.config import ProjectConfig, VideoConfig
from mvdirector.models import (
    Analysis,
    AssetState,
    AssetStatus,
    Attempt,
    AudioSlice,
    Framing,
    LyricLine,
    Manifest,
    RetryStrategy,
    Shot,
    ShotType,
    TimeSpan,
    VisualBible,
)
from mvdirector.planning.planner import _populate_slice_mutes
from mvdirector.planning.segmenter import (
    _aligned_lyric_spans,
    _is_sung_span,
    _validate_pacing_cuts,
    cut_points,
    frames_for,
    segment,
)
from mvdirector.project import Project
from tests.conftest import MemoryState


def _make_broll_run(
    tmp_path: Path,
    *,
    bpm: float,
    shot_seconds: float,
    n_shots: int,
    handle: float,
    intensity: float,
    fps: int = 24,
    frame_multiple: int = 1,
    frame_offset: int = 0,
    sync_delay: float = 0.0,
    tail_handle: float | None = None,
    num_frames_override: int | None = None,
    shot_ids: list[str] | None = None,
) -> tuple[list[ClipPlan], Manifest, Analysis]:
    dur = shot_seconds * n_shots
    beats = [round(i * 60.0 / bpm, 4) for i in range(1, int(dur * bpm / 60.0) + 2)]
    bar_s = 4.0 * 60.0 / bpm
    bars = [round(i * bar_s, 4) for i in range(max(1, int(dur / bar_s) + 1))]
    analysis = Analysis(
        duration=dur,
        tempo_bpm=bpm,
        beats=beats,
        bars=bars,
        bar_intensity=[intensity] * len(bars),
        vocal_windows=[],
    )
    eff_tail = handle if tail_handle is None else tail_handle
    shots: list[Shot] = []
    plans: list[ClipPlan] = []
    for i in range(n_shots):
        s0, s1 = i * shot_seconds, (i + 1) * shot_seconds
        f_cnt = round(s1 * fps) - round(s0 * fps)
        nf = (
            num_frames_override
            if num_frames_override is not None
            else frames_for(shot_seconds + handle + eff_tail, fps, frame_multiple, frame_offset)
        )
        sid = shot_ids[i] if shot_ids is not None else f"s{i + 1:03d}"
        src = tmp_path / f"{sid}.mp4"
        src.write_bytes(b"fake")
        shots.append(
            Shot(
                id=sid,
                index=i,
                start=s0,
                end=s1,
                type=ShotType.BROLL,
                framing=Framing.WIDE,
                intensity=intensity,
                still_prompt="p",
                motion_prompt="m",
                audio_slice=AudioSlice(
                    path=f"{sid}.wav",
                    start=max(0.0, s0 - handle),
                    end=s1 + eff_tail,
                    head_handle=handle,
                ),
                num_frames=nf,
            )
        )
        plans.append(
            ClipPlan(
                sid,
                f_cnt,
                src,
                True,
                handle,
                sync_delay=sync_delay,
                tail_handle=eff_tail,
            )
        )
    manifest = Manifest(
        title="Test",
        duration=dur,
        fps=fps,
        visual_bible=VisualBible(concept="c", style="s", palette="p", lighting="l"),
        shots=shots,
    )
    return plans, manifest, analysis


# --------------------------------------------------------------------------- H1


def test_h1_intercut_allocator_does_not_abort_realistic_runs(tmp_path: Path) -> None:
    """H1: High-intensity multi-shot and zero-handle runs intercut without stranding slivers."""
    # 136 BPM 4.0s B-roll runs of 4, 10, and 50 shots (previously 100% abort rate).
    for n_shots in (4, 10, 50):
        plans, manifest, analysis = _make_broll_run(
            tmp_path,
            bpm=136.0,
            shot_seconds=4.0,
            n_shots=n_shots,
            handle=0.25,
            intensity=0.92,
            num_frames_override=109,
        )
        out = intercut_plans(plans, manifest, analysis, min_intensity=0.70, anticipation_frames=1)
        assert len(out) > len(plans), f"136 BPM {n_shots}-shot run aborted"
        assert sum(p.frames for p in out) == sum(p.frames for p in plans)
        assert all(p.frames >= 12 for p in out)
        for prev, nxt in zip(out[:-1], out[1:]):
            assert prev.source != nxt.source

    # Zero-handle exact (1n+0) high-intensity matrix across BPMs and shot counts.
    for bpm in (80.0, 110.0, 120.0, 136.0, 145.0):
        bar_s = 4.0 * 60.0 / bpm
        for n_shots in (2, 3, 4, 5, 8):
            plans, manifest, analysis = _make_broll_run(
                tmp_path,
                bpm=bpm,
                shot_seconds=2.0 * bar_s,
                n_shots=n_shots,
                handle=0.0,
                intensity=0.90,
                frame_multiple=1,
                frame_offset=0,
            )
            out = intercut_plans(
                plans, manifest, analysis, min_intensity=0.70, anticipation_frames=1
            )
            assert len(out) > len(plans), f"Exact zero-handle aborted at {bpm} BPM, {n_shots} shots"
            assert sum(p.frames for p in out) == sum(p.frames for p in plans)
            for prev, nxt in zip(out[:-1], out[1:]):
                assert prev.source != nxt.source


def test_h1_intercut_skips_absurd_duration_window(tmp_path: Path) -> None:
    """Residual risk: synthetic beat loop is bounded for absurdly huge manifest durations."""
    plans, manifest, analysis = _make_broll_run(
        tmp_path,
        bpm=120.0,
        shot_seconds=2000.0,
        n_shots=2,
        handle=0.0,
        intensity=0.90,
    )
    analysis = analysis.model_copy(update={"beats": []})
    out = intercut_plans(plans, manifest, analysis, min_intensity=0.70, anticipation_frames=1)
    assert out == plans


# --------------------------------------------------------------------------- H2


def test_h2_estimated_lyrics_excluded_from_bleed_guard_and_slice_mutes() -> None:
    """H2: Unaligned lines stretched by estimate_missing_timings must not defeat guards."""
    vocal_windows = [TimeSpan(start=0.0, end=28.0)]
    raw_lines = [
        LyricLine(text="Whoa-oh", start=None, end=None, timing_source="none"),
        LyricLine(text="Verse one", start=18.0, end=22.0, timing_source="aligned"),
        LyricLine(text="Verse two", start=22.0, end=26.0, timing_source="aligned"),
    ]
    lines = estimate_missing_timings(raw_lines, vocal_windows)
    # Confirm estimate_missing_timings stretched line 0 across [0.0, 18.0] as "estimated".
    assert lines[0].timing_source == "estimated"
    assert lines[0].start == 0.0 and lines[0].end == 18.0

    bars = [2.0 * i for i in range(14)]
    beats = [0.5 * i for i in range(56)]
    analysis = Analysis(
        duration=28.0,
        tempo_bpm=120.0,
        beats=beats,
        bars=bars,
        bar_intensity=[0.6] * len(bars),
        vocal_windows=vocal_windows,
        lyric_lines=lines,
    )

    # _aligned_lyric_spans must ignore the "estimated" line.
    assert _aligned_lyric_spans(analysis) == [TimeSpan(start=18.0, end=26.0)]

    # Intro shots 0-16s must stay PERFORMANCE (not LIPSYNC).
    video = VideoConfig(min_clip_seconds=2.0, max_clip_seconds=4.0)
    segs = segment(analysis, video)
    intro_segs = [s for s in segs if s.end <= 16.0 + 1e-6]
    assert intro_segs and all(s.type is ShotType.PERFORMANCE for s in intro_segs)
    verse_segs = [s for s in segs if s.start >= 18.0 - 1e-6 and s.end <= 26.0 + 1e-6]
    assert verse_segs and all(s.type is ShotType.LIPSYNC for s in verse_segs)

    # _populate_slice_mutes must mute non-singing lead-in [13.75, 17.8] despite line 0 estimated.
    shot = Shot(
        id="s005",
        index=0,
        start=14.0,
        end=18.5,
        type=ShotType.LIPSYNC,
        framing=Framing.MEDIUM_CLOSE,
        still_prompt="p",
        motion_prompt="m",
        audio_slice=AudioSlice(
            path="assets/audio_slices/s005.wav", start=13.75, end=18.75, head_handle=0.25
        ),
        num_frames=121,
    )
    manifest = Manifest(
        title="Test",
        duration=28.0,
        fps=24,
        visual_bible=VisualBible(concept="c", style="s", palette="p", lighting="l"),
        shots=[shot],
    )
    _populate_slice_mutes(manifest, analysis, voice_map={})
    assert shot.audio_slice.mute_spans == [TimeSpan(start=13.75, end=17.8)]


# --------------------------------------------------------------------------- M1


def test_m1_intercut_accounts_for_sync_delay_without_source_overrun(tmp_path: Path) -> None:
    """M1: Positive sync_delay reduces tail budget so intercut never reads past shot.num_frames."""
    fps = 24
    plans, manifest, analysis = _make_broll_run(
        tmp_path,
        bpm=136.0,
        shot_seconds=4.0,
        n_shots=2,
        handle=0.25,
        tail_handle=0.50,
        sync_delay=0.50,
        intensity=0.92,
        fps=fps,
        num_frames_override=121,
    )
    out = intercut_plans(plans, manifest, analysis, min_intensity=0.70, anticipation_frames=1)
    assert len(out) > len(plans)
    assert sum(p.frames for p in out) == sum(p.frames for p in plans)
    for clip in out:
        end_frame = round(clip.head_handle * fps) + clip.frames
        assert end_frame <= 121, f"Clip {clip.shot_id} overran source (end_frame={end_frame} > 121)"


# --------------------------------------------------------------------------- M2


def test_m2_broll_min_clip_seconds_works_with_pacing_plugin_and_dp_slicing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """M2: _validate_pacing_cuts and dp_phrase_cuts honor broll_min_clip_seconds on B-roll spans."""
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
    cuts_default = cut_points(analysis, video)
    assert [round(b - a, 3) for a, b in zip(cuts_default[:-1], cuts_default[1:]) if a >= 14.0] == [
        1.75,
        1.75,
        1.75,
        1.75,
    ]

    # 3-arg and 4-arg _validate_pacing_cuts both accept default B-roll cuts.
    assert _validate_pacing_cuts(cuts_default, 21.0, video) == cuts_default
    assert _validate_pacing_cuts(cuts_default, 21.0, video, analysis=analysis) == cuts_default

    # 4-arg _validate_pacing_cuts still rejects sung spans shorter than min_clip_seconds (2.5s).
    bad_sung_cuts = [0.0, 1.75, 3.5, 7.0, 14.0, 15.75, 17.5, 19.25, 21.0]
    assert _validate_pacing_cuts(bad_sung_cuts, 21.0, video, analysis=analysis) is None

    # End-to-end no-op pacing plugin is accepted without falling back.
    class _FakeEP:
        name = "noop_pacer"

        def load(self) -> Any:
            return lambda pts, _a, _v: list(pts)

    plugins.clear_cache()
    monkeypatch.setattr(
        metadata,
        "entry_points",
        lambda *, group: [_FakeEP()] if group == plugins.PACING_GROUP else [],
    )
    try:
        plugin_cuts = cut_points(analysis, video, pacing_plugin="noop_pacer")
        assert plugin_cuts == cuts_default
    finally:
        plugins.clear_cache()

    # Enabling phrase_dp_slicing keeps sung >= 2.5s while letting high-intensity B-roll cut < 2.5s.
    dp_video = video.model_copy(update={"phrase_dp_slicing": True})
    dp_segs = segment(analysis, dp_video)
    sung = [s for s in dp_segs if s.type is ShotType.LIPSYNC]
    drop = [s for s in dp_segs if s.start >= 14.0]
    calm = [s for s in dp_segs if 7.0 <= s.start < 14.0]
    assert sung and all(s.duration >= 2.5 - 1e-3 for s in sung)
    assert calm and max(s.duration for s in calm) >= 6.5
    assert drop and all(s.duration < 2.5 for s in drop)


# --------------------------------------------------------------------------- M3 & M4


def test_m3_m4_assemble_uses_state_store_and_unique_temp_filenames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """M3 & M4: assemble() loads analysis from non-local StateStore and avoids temp-file races."""
    # M4: intercut_plans avoids colliding with an existing shot_id like "s002_ic00".
    plans, manifest, analysis = _make_broll_run(
        tmp_path,
        bpm=136.0,
        shot_seconds=4.0,
        n_shots=3,
        handle=0.25,
        intensity=0.92,
        shot_ids=["s002", "s003", "s002_ic00"],
    )
    out = intercut_plans(plans, manifest, analysis, min_intensity=0.70, anticipation_frames=1)
    ids = [p.shot_id for p in out]
    assert len(ids) == len(set(ids)), f"Colliding intercut shot_ids: {ids}"

    # M3: non-local MemoryState (analysis_path.is_file() is False) still runs intercut_plans.
    store = MemoryState()
    cfg = ProjectConfig(
        title="MemStore",
        video=VideoConfig(
            assembly_intercut=True,
            assembly_intercut_min_intensity=0.70,
            assembly_workers=2,
            min_clip_seconds=2.5,
            max_clip_seconds=8.0,
        ),
    )
    mix_path = tmp_path / "inputs" / "mix.wav"
    mix_path.parent.mkdir(parents=True, exist_ok=True)
    mix_path.write_bytes(b"RIFF")
    project = Project(root=tmp_path, config=cfg, state=store)
    assert not project.analysis_path.is_file()

    clip_dir = tmp_path / "assets" / "videos"
    clip_dir.mkdir(parents=True, exist_ok=True)
    for sid in ("s001", "s002"):
        (clip_dir / f"{sid}.mp4").write_bytes(b"mp4")

    mem_analysis = Analysis(
        duration=8.0,
        tempo_bpm=136.0,
        beats=[round(i * 60.0 / 136.0, 4) for i in range(1, 20)],
        bars=[0.0, 1.765, 3.529, 5.294, 7.059],
        bar_intensity=[0.92] * 5,
        vocal_windows=[],
    )
    store.save_analysis(mem_analysis)

    mem_manifest = Manifest(
        title="MemStore",
        duration=8.0,
        fps=24,
        visual_bible=VisualBible(concept="c", style="s", palette="p", lighting="l"),
        shots=[
            Shot(
                id="s001",
                index=0,
                start=0.0,
                end=4.0,
                type=ShotType.BROLL,
                framing=Framing.WIDE,
                intensity=0.92,
                still_prompt="p",
                motion_prompt="m",
                audio_slice=AudioSlice(path="a.wav", start=0.0, end=4.25, head_handle=0.25),
                num_frames=109,
                video=AssetState(
                    status=AssetStatus.PASSED,
                    selected_attempt=1,
                    attempts=[
                        Attempt(
                            number=1,
                            strategy=RetryStrategy.INITIAL,
                            seed=1,
                            prompt="m",
                            path="assets/videos/s001.mp4",
                        )
                    ],
                ),
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
                audio_slice=AudioSlice(path="b.wav", start=3.75, end=8.25, head_handle=0.25),
                num_frames=109,
                video=AssetState(
                    status=AssetStatus.PASSED,
                    selected_attempt=1,
                    attempts=[
                        Attempt(
                            number=1,
                            strategy=RetryStrategy.INITIAL,
                            seed=2,
                            prompt="m",
                            path="assets/videos/s002.mp4",
                        )
                    ],
                ),
            ),
        ],
    )

    recorded_clip_outputs: list[Path] = []
    intercut_called = False
    orig_intercut = assemble_mod.intercut_plans

    def _spy_intercut(*args: Any, **kwargs: Any) -> list[ClipPlan]:
        nonlocal intercut_called
        intercut_called = True
        return orig_intercut(*args, **kwargs)

    def _fake_run(cmd: list[str]) -> str:
        out_path = Path(cmd[-1])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"ok")
        if out_path.name.endswith(".mp4") and not out_path.name.startswith(
            ("picture", ".mvd_final_")
        ):
            recorded_clip_outputs.append(out_path)
        return ""

    monkeypatch.setattr(assemble_mod, "intercut_plans", _spy_intercut)
    monkeypatch.setattr(media, "run", _fake_run)
    monkeypatch.setattr(media, "resolve_video_encoder", lambda _e: "libx264")
    monkeypatch.setattr(
        assemble_mod.broadcast_qa_mod,
        "analyze_broadcast_audio",
        lambda _p: type("_QA", (), {"problems": lambda self: [], "summary": lambda self: ""})(),
    )

    output_mp4 = tmp_path / "out" / "final.mp4"
    assemble_mod.assemble(project, mem_manifest, output_mp4, force=True)

    assert intercut_called, "assemble() skipped intercut_plans for MemoryStateStore"
    assert len(recorded_clip_outputs) > 2
    # Every temp clip filename is uniquely index-prefixed ({idx:04d}_{safe_id}.mp4).
    names = [p.name for p in recorded_clip_outputs]
    assert len(names) == len(set(names))
    assert all(name[:5].endswith("_") and name[:4].isdigit() for name in names)


# --------------------------------------------------------------------------- M5


def test_m5_filter_vocal_windows_by_words_counts_zero_duration_words() -> None:
    """M5: Zero-duration words (start == end) count as hits; inverted words and NaN pad are safe."""
    windows = [
        TimeSpan(start=5.0, end=14.0),  # bleed window with 2 normal hallucinated words
        TimeSpan(start=15.0, end=25.0),  # real verse with 2 zero-duration + 1 normal word
    ]
    words = [
        Word(word="uh", start=5.5, end=6.0),
        Word(word="ah", start=6.1, end=6.8),
        Word(word="hey", start=16.0, end=16.0),
        Word(word="now", start=17.0, end=17.0),
        Word(word="sing", start=18.0, end=18.2),
        Word(word="inverted", start=20.0, end=19.0),
    ]
    filtered = filter_vocal_windows_by_words(windows, words, pad=0.5)
    assert TimeSpan(start=15.5, end=18.7) in filtered
    assert len(filtered) == 2

    # Non-finite / negative pad falls back to 0.0 cleanly.
    filtered_nan = filter_vocal_windows_by_words(
        [TimeSpan(start=15.0, end=25.0)],
        [Word(word="hey", start=16.0, end=16.0), Word(word="now", start=17.0, end=17.0)],
        pad=float("nan"),
    )
    assert filtered_nan == [TimeSpan(start=16.0, end=17.0)]


# --------------------------------------------------------------------------- L1


def test_l1_intercut_non_final_subclips_zero_tail_handle_prevents_xfade_replay(
    tmp_path: Path,
) -> None:
    """L1: Non-final sub-clips of a source expose tail_handle=0.0 so xfade cannot replay frames."""
    fps = 24
    plans, manifest, analysis = _make_broll_run(
        tmp_path,
        bpm=136.0,
        shot_seconds=4.0,
        n_shots=2,
        handle=0.25,
        intensity=0.92,
        fps=fps,
        num_frames_override=121,
    )
    out = intercut_plans(plans, manifest, analysis, min_intensity=0.70, anticipation_frames=1)
    by_source: dict[Path | None, list[ClipPlan]] = {}
    for clip in out:
        by_source.setdefault(clip.source, []).append(clip)

    assert any(len(clips) >= 2 for clips in by_source.values())
    for clips in by_source.values():
        for earlier, later in zip(clips[:-1], clips[1:]):
            # Zero tail_handle guarantees xfade borrows 0 frames on non-final sub-clips.
            assert earlier.tail_handle == 0.0
            earlier_end = earlier.head_handle + earlier.frames / fps
            assert later.head_handle >= earlier_end - 1e-3


# --------------------------------------------------------------------------- L2


def test_l2_is_sung_span_merges_overlapping_lyrics_and_vocal_windows() -> None:
    """L2: Overlapping duet/echo lyric lines and vocal windows must not be double-counted."""
    analysis = Analysis(
        duration=10.0,
        tempo_bpm=120.0,
        beats=[0.5 * i for i in range(20)],
        bars=[2.0 * i for i in range(5)],
        bar_intensity=[0.6] * 5,
        vocal_windows=[
            TimeSpan(start=3.85, end=4.0),
            TimeSpan(start=3.85, end=4.0),
            TimeSpan(start=0.0, end=4.0),
        ],
        lyric_lines=[
            LyricLine(text="Echo A", start=3.85, end=4.0, timing_source="aligned"),
            LyricLine(text="Echo B", start=3.85, end=4.0, timing_source="aligned"),
        ],
    )
    merged_lyrics = _aligned_lyric_spans(analysis)
    assert merged_lyrics == [TimeSpan(start=3.85, end=4.0)]
    # Actual union overlap is 0.15s < 0.25s threshold, even if caller passes unmerged spans.
    unmerged = [TimeSpan(start=3.85, end=4.0), TimeSpan(start=3.85, end=4.0)]
    assert not _is_sung_span(analysis, 0.0, 4.0, 0.25, merged_lyrics)
    assert not _is_sung_span(analysis, 0.0, 4.0, 0.25, unmerged)

    # Overlapping vocal_windows also must not double-count vocal fraction.
    dup_win_analysis = analysis.model_copy(
        update={
            "vocal_windows": [TimeSpan(start=0.0, end=0.6), TimeSpan(start=0.0, end=0.6)],
            "lyric_lines": [],
        }
    )
    # 0.6s / 4.0s = 0.15 < 0.25 (unmerged would have summed to 1.2s / 4.0s = 0.30).
    assert not _is_sung_span(dup_win_analysis, 0.0, 4.0, 0.25, [])


# --------------------------------------------------------------------------- L3


def test_l3_video_config_legacy_max_clip_and_bool_rejection() -> None:
    """L3: Legacy max_clip_seconds<2.5 configs still load; booleans are rejected on numbers."""
    # Legacy config omitting min_clip_seconds clamps the default cleanly.
    legacy = VideoConfig(max_clip_seconds=2.0)
    assert legacy.max_clip_seconds == 2.0
    assert legacy.min_clip_seconds == 2.0

    # Explicit min_clip_seconds > max_clip_seconds is still rejected.
    with pytest.raises(ValidationError, match="min_clip_seconds"):
        VideoConfig(min_clip_seconds=2.5, max_clip_seconds=2.0)

    # Booleans are rejected on numeric VideoConfig fields instead of coercing to 1 / 0.
    for kwargs in (
        {"broll_min_clip_seconds": True},
        {"assembly_intercut_anticipation_frames": True},
        {"assembly_intercut_min_intensity": False},
        {"min_clip_seconds": True},
        {"max_clip_seconds": True},
        {"assembly_workers": True},
    ):
        with pytest.raises(ValidationError, match="not a boolean"):
            VideoConfig(**kwargs)  # type: ignore[arg-type]
