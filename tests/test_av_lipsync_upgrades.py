# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Tests for audio, video, and lip-sync upgrades across Phases 1-3."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest
import soundfile as sf
from PIL import Image

from mvdirector import assemble, media, mix_assemble
from mvdirector.audio import alignment_qa, broadcast_qa, foley
from mvdirector.audio import lyrics as lyrics_mod
from mvdirector.audio.analysis import analyze_project
from mvdirector.audio.lyrics import (
    Word,
    _drain_segments_with_timeout,
    collapse_repetitions,
    collapse_word_repetitions,
)
from mvdirector.comfy.cache import ComfyRenderCache, canonical_graph_key
from mvdirector.comfy.client import ComfyClient
from mvdirector.comfy.workflow import (
    convert_ui_workflow,
    prune_switch_branches,
)
from mvdirector.config import ComfyConfig, ProjectConfig, VideoConfig
from mvdirector.errors import (
    ComfyJobError,
    ComfyResourceError,
    ConfigError,
    MediaToolError,
)
from mvdirector.generation import previz as previz_gen
from mvdirector.generation.runner import RunBudget, ShotRunner, load_templates
from mvdirector.models import (
    Analysis,
    AssetKind,
    AssetState,
    AssetStatus,
    Attempt,
    AudioSlice,
    Character,
    CheckResult,
    CheckStatus,
    ContinuityMode,
    Framing,
    LyricLine,
    Manifest,
    MixChapter,
    ProjectMode,
    QCReport,
    RetryStrategy,
    SetDesign,
    Shot,
    ShotType,
    TimeSpan,
    VisualBible,
    WorldFact,
)
from mvdirector.planning import prompts
from mvdirector.planning.planner import build_manifest
from mvdirector.planning.previz import default_previz_plan
from mvdirector.planning.segmenter import (
    dp_phrase_cuts,
    shift_to_vocal_onset,
    split_long_span_at_silences,
)
from mvdirector.planning.world_state import (
    IrreversibleFactViolation,
    WorldStateLedger,
    filter_look_for_state,
)
from mvdirector.project import Project
from mvdirector.qc.color_drift import (
    ColorAffine,
    fit_color_drift_affine,
)
from mvdirector.qc.continuity import measure_beat_motion_sync, measure_join_continuity
from mvdirector.qc.factory import QualityGates
from mvdirector.qc.lipsync import SyncMeasurement
from mvdirector.qc.video import VideoQC
from mvdirector.qc.vision import VisionJudge, VisionVerdict
from mvdirector.retry import RetryPolicy, knuth_seed
from tests.conftest import FakeComfy, write_workflows

SR = 22050


def make_video(
    dst: Path,
    *,
    duration: float = 1.0,
    fps: int = 24,
    width: int = 128,
    height: int = 72,
    color: str = "red",
) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    media.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"color=c={color}:s={width}x{height}:r={fps}:d={duration:.3f}",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            str(dst.resolve()),
        ]
    )
    return dst


# --------------------------------------------------------------------------- Audio & Alignment QA


def test_voicing_and_consonant_ratios_distinguish_vocals_from_noise_and_silence():
    t = np.arange(SR, dtype=np.float64) / SR
    voiced = 0.4 * np.sin(2.0 * np.pi * 220.0 * t)
    rng = np.random.default_rng(7)
    noise = 0.25 * rng.standard_normal(SR)
    high_tone = 0.4 * np.sin(2.0 * np.pi * 5500.0 * t)
    silence = np.zeros(SR, dtype=np.float64)

    assert alignment_qa.voicing_ratio(voiced, SR) > 0.7
    assert alignment_qa.voicing_ratio(noise, SR) < 0.25
    assert alignment_qa.voicing_ratio(silence, SR) == 0.0

    assert alignment_qa.consonant_band_ratio(high_tone, SR) > 0.85
    assert alignment_qa.consonant_band_ratio(voiced, SR) < 0.1
    assert alignment_qa.consonant_band_ratio(silence, SR) == 0.0


def test_audit_lyric_alignment_and_boundary_word_cuts():
    # 0-2s voiced sine, 2-4s silence, 4-6s voiced sine
    t = np.arange(6 * SR, dtype=np.float64) / SR
    vocals = np.where((t < 2.0) | (t >= 4.0), 0.4 * np.sin(2.0 * np.pi * 200.0 * t), 0.0)
    lines = [
        LyricLine(text="Clean sung line", start=0.2, end=1.8, timing_source="aligned"),
        LyricLine(text="Ghost hallucinated line", start=2.2, end=3.6, timing_source="aligned"),
    ]
    windows = [TimeSpan(start=0.0, end=2.0), TimeSpan(start=4.0, end=6.0)]
    warnings = alignment_qa.audit_lyric_alignment(vocals, SR, lines, windows)
    kinds = {w.kind for w in warnings}
    assert "unvoiced_lyric" in kinds
    assert "unaligned_vocal_window" in kinds

    cut_warns = alignment_qa.detect_boundary_word_cuts(
        [1.0, 2.0], [("hello", 0.8, 1.2), ("world", 1.5, 1.9)]
    )
    assert len(cut_warns) == 1
    assert cut_warns[0].kind == "boundary_word_cut"
    assert "hello" in cut_warns[0].detail


def test_whisper_repetition_collapsing_and_stall_watchdog():
    assert collapse_repetitions("yeah yeah") == "yeah yeah"
    assert collapse_repetitions("hold on hold on hold on hold on now") == "hold on hold on now"

    # Sung repeats with real timings keep every word (and timestamp) for lyric alignment...
    sung = [Word(word="la", start=i * 0.2, end=(i + 1) * 0.2) for i in range(5)]
    assert collapse_word_repetitions(sung) == sung
    # ...while a decoding loop (zero-length, stacked timestamps) collapses to two words.
    looped = [Word(word="la", start=1.0, end=1.0) for _ in range(5)]
    assert len(collapse_word_repetitions(looped)) == 2

    def _stalled():
        yield "ok"
        time.sleep(1.0)

    with pytest.raises(TimeoutError, match="stalled"):
        _drain_segments_with_timeout(_stalled(), timeout_seconds=0.05)

    def _slow_start():
        time.sleep(0.2)  # feature extraction and language detection precede the first segment
        yield "first"
        yield "second"

    drained = _drain_segments_with_timeout(
        _slow_start(), timeout_seconds=0.05, first_timeout_seconds=5.0
    )
    assert drained == ["first", "second"]
    assert ProjectConfig().prepare.whisper_stall_timeout_seconds == 600.0


def test_broadcast_audio_lufs_true_peak_and_splice_clicks(tmp_path: Path):
    sr = 48000
    t = np.arange(sr, dtype=np.float64) / sr
    # -14 dBFS sine wave (~ -14 to -17 LUFS).
    amp = 10.0 ** (-14.0 / 20.0)
    tone = amp * np.sin(2.0 * np.pi * 1000.0 * t)
    lufs = broadcast_qa.integrated_lufs(tone, sr)
    tp = broadcast_qa.true_peak_dbtp(tone)
    assert -18.0 < lufs < -12.0
    assert -15.0 < tp < -13.0

    # Inject a sharp DC step click at t = 0.50s.
    clicked = tone.copy()
    clicked[sr // 2] += 0.45
    clicks = broadcast_qa.scan_splice_clicks(clicked, sr, boundary_times=[0.5])
    assert clicks and abs(clicks[0] - 0.5) < 0.02

    # Drum-onset exclusion ignores a legitimate drum hit at the same timestamp.
    masked = broadcast_qa.scan_splice_clicks(
        clicked, sr, boundary_times=[0.5], drum_onset_times=[0.5]
    )
    assert masked == []

    wav = tmp_path / "mix.wav"
    sf.write(wav, clicked, sr)
    report = broadcast_qa.analyze_broadcast_audio(wav, boundary_times=[0.5])
    assert report.splice_clicks


# --------------------------------------------------------------------------- Foley & Media


def test_foley_active_rms_leveling_retime_and_ducking(tmp_path: Path):
    sr = 22050
    sig = np.zeros(sr * 2, dtype=np.float64)
    # Short sharp impulse at 0.30s.
    start = int(0.30 * sr)
    sig[start : start + 400] = 0.05 * np.hanning(400) * np.sin(np.linspace(0, 40, 400))

    leveled = foley.level_active_rms(sig, sr, target_db=-26.0, peak_ceiling_db=-6.0)
    assert float(np.max(np.abs(leveled))) <= 10.0 ** (-6.0 / 20.0) + 1e-6
    assert float(np.max(np.abs(leveled))) > float(np.max(np.abs(sig)))

    onsets = foley.detect_foley_onsets(leveled, sr)
    assert onsets and abs(onsets[0] - 0.30) < 0.05

    retimed = foley.retime_foley_hits(leveled, sr, [0.80, 1.40], total_duration=2.0)
    peak_near_08 = float(np.max(np.abs(retimed[int(0.75 * sr) : int(0.90 * sr)])))
    peak_near_14 = float(np.max(np.abs(retimed[int(1.35 * sr) : int(1.50 * sr)])))
    assert peak_near_08 > 0.05 and peak_near_14 > 0.05

    music_wav = tmp_path / "music.wav"
    cue_wav = tmp_path / "cue.wav"
    out_wav = tmp_path / "ducked.wav"
    t = np.arange(sr * 2, dtype=np.float32) / sr
    sf.write(music_wav, (0.3 * np.sin(2.0 * np.pi * 440.0 * t)), sr)
    sf.write(cue_wav, leveled.astype(np.float32), sr)
    cue = foley.DiegeticCue(audio_path=cue_wav, start=0.5, end=1.2, duck_db=-8.0)
    foley.build_ducked_audio_mix(music_wav, [cue], out_wav)
    assert out_wav.is_file() and out_wav.stat().st_size > 0


def test_mirror_pad_audio_and_prepare_audio_slice(tmp_path: Path):
    base = np.linspace(0.0, 1.0, 100, dtype=np.float64)[:, None]
    padded = media.mirror_pad_audio(base, 350)
    assert padded.shape == (350, 1)
    # Boundary at index 99->100 must mirror smoothly without dropping to 0.
    assert abs(float(padded[100, 0]) - float(padded[98, 0])) < 1e-9

    sr = 22050
    t = np.arange(sr * 3, dtype=np.float32) / sr
    src = tmp_path / "src.wav"
    dst = tmp_path / "slice.wav"
    sf.write(src, 0.4 * np.sin(2.0 * np.pi * 220.0 * t), sr)
    media.prepare_audio_slice(
        src, dst, 0.5, 1.5, target_duration=1.25, vae_offset_seconds=0.046, mirror_pad=True
    )
    out, out_sr = sf.read(str(dst))
    assert len(out) / out_sr == pytest.approx(1.25, abs=0.02)


def test_extract_frame_zoom_crop_and_ensure_audio_track(tmp_path: Path):
    clip = tmp_path / "silent.mp4"
    make_video(clip, duration=1.0, fps=24, width=128, height=72)
    assert not media.probe(clip).has_audio

    frame_png = tmp_path / "handoff.png"
    media.extract_frame_at_index(clip, 10, frame_png)
    assert frame_png.is_file()
    with Image.open(frame_png) as img:
        assert img.size == (128, 72)

    zoomed = tmp_path / "zoom.png"
    media.zoom_crop_image(frame_png, zoomed, zoom_factor=1.5)
    with Image.open(zoomed) as zimg:
        assert zimg.size == (128, 72)

    with_aud = tmp_path / "with_audio.mp4"
    result = media.ensure_audio_track(clip, with_aud)
    assert result == with_aud
    assert media.probe(with_aud).has_audio


# --------------------------------------------------------------------------- Segmentation & Prompts


def test_silence_split_dp_phrase_and_vocal_onset():
    counts = split_long_span_at_silences(
        0.0,
        12.0,
        [TimeSpan(start=0.5, end=5.5), TimeSpan(start=6.5, end=11.5)],
        fps=24,
        min_seconds=3.0,
        max_seconds=8.0,
    )
    assert sum(counts) == 12 * 24
    # Cut should land right in the 5.5-6.5s silence gap (frame 144 = 6.0s).
    assert counts[0] == 144

    analysis = Analysis(
        duration=12.0,
        tempo_bpm=120.0,
        beats=[i * 0.5 for i in range(24)],
        bars=[0.0, 2.0, 4.0, 6.0, 8.0, 10.0],
        bar_intensity=[0.5] * 6,
        vocal_windows=[TimeSpan(start=0.5, end=5.8), TimeSpan(start=6.2, end=11.5)],
        lyric_lines=[
            LyricLine(text="Phrase one", start=0.5, end=5.8, timing_source="aligned"),
            LyricLine(text="Phrase two", start=6.2, end=11.5, timing_source="aligned"),
        ],
    )
    video = VideoConfig(min_clip_seconds=2.0, max_clip_seconds=8.0)
    dp_cuts = dp_phrase_cuts(analysis, video, [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0])
    # DP should pick 6.0s (clean gap) and avoid splitting through 2.0/4.0/8.0/10.0 mid-phrase!
    assert dp_cuts == [0.0, 6.0, 12.0]

    shifted = shift_to_vocal_onset(
        [0.0, 4.0, 8.0], [TimeSpan(start=4.3, end=7.5)], VideoConfig(min_clip_seconds=2.0)
    )
    assert shifted == [0.0, 4.3, 8.0]


def test_world_state_ledger_and_prompt_craft_upgrades():
    ledger = WorldStateLedger()
    ledger.set_fact("lantern", "illumination", "unlit", shot_index=0)
    ledger.set_fact(
        "mirror", "integrity", "shattered", shot_index=2, irreversible=True, event_id="s003"
    )
    with pytest.raises(IrreversibleFactViolation):
        ledger.set_fact("mirror", "integrity", "intact", shot_index=4)

    filtered = filter_look_for_state(
        "35mm film grain, warm glowing lantern light, moody blue shadows",
        ledger.state_at(1),
    )
    assert "glowing lantern" not in filtered.lower()
    assert "35mm film grain" in filtered

    assert (
        "photograph"
        not in prompts.strip_still_words("Film still photograph, slow push-in.").lower()
    )
    assert prompts.is_rear_view_prompt("Over Elias's shoulder looking toward the stage")
    scrubbed = prompts.scrub_frontal_facial_details(
        "Tall woman, green eyes, sharp nose, crimson lipstick, long dark wavy hair, leather coat"
    )
    assert "eyes" not in scrubbed.lower() and "long dark wavy hair" in scrubbed.lower()

    budgeted = prompts.fit_prompt_budget(
        "Head sentence. Middle filler one. Middle filler two. Guard one. Guard two.",
        max_chars=52,
        keep_tail=2,
    )
    assert budgeted.startswith("Head sentence.") and budgeted.endswith("Guard two.")

    timed = prompts.inject_multicut_timestamps(
        ["Wide stage sweep", "Close on hands"], shot_duration=4.0, beat_offsets=[0.0, 1.95, 3.9]
    )
    assert "[Shot 1] At 00:00.000" in timed and "[Shot 2] At 00:01.950" in timed

    # Verify WorldFacts + Picture N disambiguation inside compose_still_prompt.
    manifest = Manifest(
        title="Test",
        duration=8.0,
        fps=24,
        characters={
            "elias": Character(name="Elias", appearance="Dark hair, brown eyes."),
            "odalie": Character(name="Odalie", appearance="Blonde hair, blue eyes."),
        },
        visual_bible=VisualBible(
            style="35mm film",
            lighting="warm glowing lantern light, cold moonlight",
            settings=[SetDesign(name="Hall", description="Stone hall.")],
            world_facts=[
                WorldFact(entity="lantern", attribute="illumination", value="unlit", start_shot=0)
            ],
        ),
    )
    shot = Shot(
        id="s001",
        index=0,
        start=0.0,
        end=4.0,
        type=ShotType.PERFORMANCE,
        framing=Framing.MEDIUM,
        setting="Hall",
        character="elias",
        characters=["elias", "odalie"],
        still_prompt="Over Elias's shoulder facing Odalie.",
        audio_slice=AudioSlice(
            path="assets/audio_slices/s001.wav", start=0.0, end=4.0, head_handle=0.0
        ),
        num_frames=97,
    )
    composed = prompts.compose_still_prompt(
        manifest,
        shot,
        plate_attached=True,
        bound_references=["reference_image", "reference_image_2", "set_image"],
    )
    assert "Picture 1 defines Elias" in composed
    assert "Picture 2 defines Odalie" in composed
    assert "Picture 3 defines only the empty location" in composed
    assert "glowing lantern" not in composed.lower()
    assert "brown eyes" not in composed.lower()


# ------------------------------------------------------------------ QC, Color Drift & Continuity


def test_blind_observer_vision_judge_splits_image_and_spec():
    calls: list[dict[str, Any]] = []

    class _FakeLLM:
        def complete_json(self, *, system: str, content: list[dict[str, Any]], schema: Any) -> Any:
            calls.append({"system": system, "content": content, "schema": schema.__name__})
            if len(calls) == 1:
                return schema(
                    people_count=1,
                    framing="close_up",
                    subjects="Singer with dark hair singing.",
                    setting="Dark hall",
                    props_and_objects="Unlit brass lantern",
                    artifacts_or_text=[],
                )
            return VisionVerdict(spec_match=0.95, artifact_free=0.95, character_match=0.9)

    manifest = Manifest(
        title="Blind",
        duration=4.0,
        fps=24,
        character=Character(appearance="Dark hair."),
        visual_bible=VisualBible(style="film"),
    )
    shot = Shot(
        id="s001",
        index=0,
        start=0.0,
        end=4.0,
        type=ShotType.LIPSYNC,
        framing=Framing.CLOSE_UP,
        still_prompt="Close-up singing, no cape.",
        audio_slice=AudioSlice(path="a.wav", start=0.0, end=4.0, head_handle=0.0),
        num_frames=97,
    )
    judge = VisionJudge(_FakeLLM(), blind_observer=True)  # type: ignore[arg-type]
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    verdict = judge.judge([frame], shot, manifest, AssetKind.STILL)
    assert verdict.spec_match == 0.95
    assert len(calls) == 2
    # Both passes carry the overlaid-typography rule (hallucinated title cards / credits).
    assert "title cards" in calls[0]["system"] and "title cards" in calls[1]["system"]
    assert "artifact_free <= 0.2" in calls[1]["system"]
    # Pass 1 (Blind Observer) sees image, NEVER sees target exclusions ("no cape").
    p1_text = " ".join(b.get("text", "") for b in calls[0]["content"] if b.get("type") == "text")
    assert any(b.get("type") == "image" for b in calls[0]["content"])
    assert "no cape" not in p1_text.lower()
    # Pass 2 (Blind Auditor) sees text-only spec + report, NEVER sees image.
    assert not any(b.get("type") == "image" for b in calls[1]["content"])
    p2_text = " ".join(b.get("text", "") for b in calls[1]["content"] if b.get("type") == "text")
    assert "SHOT SPECIFICATION" in p2_text and "Unlit brass lantern" in p2_text


def test_color_drift_affine_recovers_shift_and_clamps_extremes():
    rng = np.random.default_rng(42)
    target = rng.integers(40, 210, size=(64, 64, 3), dtype=np.uint8)
    # Apply a subtle VAE-like warm shift (+6 R, -4 B, slight gain).
    shifted = np.clip(
        target.astype(np.float64) * np.array([0.96, 1.0, 1.04]) + np.array([-5.0, 0.0, 4.0]),
        0,
        255,
    ).astype(np.uint8)
    affine = fit_color_drift_affine([shifted], [target])
    restored = affine.apply(shifted)
    before_err = float(np.mean(np.abs(shifted.astype(np.float64) - target.astype(np.float64))))
    after_err = float(np.mean(np.abs(restored.astype(np.float64) - target.astype(np.float64))))
    assert after_err < before_err * 0.5
    assert affine.to_ffmpeg_colorchannelmixer() is not None


def test_join_continuity_and_beat_motion_sync():
    # Smooth synthetic horizontal pan vs duplicate seam frame.
    grid = np.zeros((96, 96, 3), dtype=np.uint8)
    grid[20:76, 20:76] = 200
    pan = [np.roll(grid, shift=i * 3, axis=1) for i in range(8)]
    smooth_rep = measure_join_continuity(pan[:4], pan[4:])
    assert not smooth_rep.duplicate_boundary_frame

    dup_rep = measure_join_continuity(pan[:4], [pan[3], pan[4], pan[5], pan[6]])
    assert dup_rep.duplicate_boundary_frame
    assert dup_rep.issues

    # Beat-synced pulsing frames every 0.5s at 10 fps.
    fps = 10.0
    frames = []
    for i in range(40):
        val = 220 if (i % 5 == 0) else 40
        frames.append(np.full((48, 48, 3), val, dtype=np.uint8))
    beats = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    sync_rep = measure_beat_motion_sync(frames, fps, beats, permutations=40, seed=1)
    assert sync_rep.correlation > 0.4


def test_knuth_seed_31bit_and_video_qc_sync_delay_metric(tmp_path: Path):
    seeds = [knuth_seed(1234, i) for i in range(1, 20)]
    assert all(0 <= s < 2**31 - 1 for s in seeds)
    assert len(set(seeds)) == len(seeds)

    clip = tmp_path / "sing.mp4"
    make_video(clip, duration=1.5, fps=24, width=128, height=72)
    wav = tmp_path / "vocals.wav"
    t = np.arange(int(1.5 * SR), dtype=np.float32) / SR
    sf.write(wav, 0.5 * np.sin(2.0 * np.pi * 220.0 * t), SR)

    class _FakeScorer:
        def score(self, frames, fps, samples, sample_rate, max_lag_seconds, max_faces=1):
            return SyncMeasurement(
                correlation=0.75, lag_frames=3, face_presence=1.0, eval_fps=25.0, confidence=0.4
            )

    cfg = ProjectConfig()
    cfg.video.width, cfg.video.height = 128, 72
    manifest = Manifest(title="Sync", duration=1.5, fps=24)
    shot = Shot(
        id="s001",
        index=0,
        start=0.25,
        end=1.25,
        type=ShotType.LIPSYNC,
        framing=Framing.CLOSE_UP,
        audio_slice=AudioSlice(path="a.wav", start=0.0, end=1.5, head_handle=0.25),
        num_frames=36,
    )
    qc = VideoQC(cfg.qc, cfg.video, None, None, None, lipsync_scorer=_FakeScorer())  # type: ignore[arg-type]
    report = qc.evaluate(clip, shot, manifest, wav, None)
    sync_check = next(c for c in report.checks if c.name == "lipsync")
    assert sync_check.metrics["sync_delay_seconds"] == pytest.approx(0.12, abs=0.01)


# --------------------------------------------------------------------------- ComfyUI & Runner


def test_comfy_cache_switch_pruning_ui_converter_and_client_guards(tmp_path: Path):
    img_a = tmp_path / "a.png"
    img_a.write_bytes(b"IMG-A")
    g1 = {
        "1": {"class_type": "LoadImage", "inputs": {"image": "sub/a.png"}},
        "2": {"class_type": "SaveImage", "inputs": {"filename_prefix": "run1", "images": ["1", 0]}},
    }
    g2 = {
        "1": {"class_type": "LoadImage", "inputs": {"image": "sub/a.png"}},
        "2": {"class_type": "SaveImage", "inputs": {"filename_prefix": "run2", "images": ["1", 0]}},
    }
    k1 = canonical_graph_key(g1, {"sub/a.png": img_a})
    k2 = canonical_graph_key(g2, {"sub/a.png": img_a})
    assert k1 == k2

    cache = ComfyRenderCache(tmp_path / ".cache")
    src_out = tmp_path / "gen.png"
    src_out.write_bytes(b"RENDERED-PNG")
    cache.store(k1, src_out)
    dst_out = tmp_path / "hit.png"
    assert cache.lookup(k2, dst_out) and dst_out.read_bytes() == b"RENDERED-PNG"

    # Changing input bytes invalidates the cache key.
    img_a.write_bytes(b"IMG-B")
    assert canonical_graph_key(g1, {"sub/a.png": img_a}) != k1

    # Static ComfySwitchNode collapsing & dead-branch pruning.
    sw_graph = {
        "1": {"class_type": "LoadImage", "inputs": {"image": "used.png"}},
        "2": {"class_type": "LoadImage", "inputs": {"image": "unused.png"}},
        "3": {
            "class_type": "ComfySwitchNode",
            "inputs": {"switch": True, "on_true": ["1", 0], "on_false": ["2", 0]},
        },
        "4": {"class_type": "SaveImage", "inputs": {"images": ["3", 0]}},
    }
    pruned = prune_switch_branches(sw_graph, output_node="4")
    assert "3" not in pruned and "2" not in pruned
    assert pruned["4"]["inputs"]["images"] == ["1", 0]

    # Clean-room UI workflow -> API graph conversion.
    ui_graph = {
        "nodes": [
            {"id": 1, "type": "KSampler", "inputs": [], "widgets_values": [42, "randomize", 20]},
            {
                "id": 2,
                "type": "SaveImage",
                "inputs": [{"name": "images", "link": 10}],
                "widgets_values": [],
            },
        ],
        "links": [[10, 1, 0, 2, 0, "IMAGE"]],
    }
    obj_info = {
        "KSampler": {"input": {"required": {"seed": ["INT", {}], "steps": ["INT", {}]}}},
        "SaveImage": {"input": {"required": {"images": ["IMAGE"]}}},
    }
    api_g = convert_ui_workflow(ui_graph, obj_info)
    assert api_g["1"]["inputs"] == {"seed": 42, "steps": 20}
    assert api_g["2"]["inputs"]["images"] == ["1", 0]

    # Client: dedup_uploads HEAD /view, live /object_info probe, VRAM floor, partial targets.
    seen_payloads: list[dict[str, Any]] = []
    free_called = []

    def handler(req: httpx.Request) -> httpx.Response:
        if req.method == "HEAD" and req.url.path == "/view":
            return httpx.Response(200)
        if req.url.path == "/system_stats":
            # 8 GiB free VRAM
            return httpx.Response(200, json={"devices": [{"vram_free": 8 * 1024**3}]})
        if req.url.path == "/free":
            free_called.append(True)
            return httpx.Response(200, json={})
        if req.url.path == "/object_info":
            return httpx.Response(
                200,
                json={
                    "CheckpointLoader": {
                        "input": {"required": {"ckpt_name": [["installed.safetensors"]]}}
                    }
                },
            )
        if req.url.path == "/prompt":
            seen_payloads.append(json.loads(req.content))
            return httpx.Response(200, json={"prompt_id": "p1"})
        return httpx.Response(404)

    cfg = ComfyConfig(base_url="http://comfy.test", dedup_uploads=True, min_free_vram_gb=12.0)
    with ComfyClient(cfg, transport=httpx.MockTransport(handler), sleep=lambda _: None) as comfy:
        remote = comfy.upload(img_a)
        assert remote.startswith("mvdirector/a_") and remote.endswith(".png")
        issues = comfy.validate_graph_live(
            {
                "1": {
                    "class_type": "CheckpointLoader",
                    "inputs": {"ckpt_name": "missing.safetensors"},
                },
                "2": {"class_type": "UnknownNode", "inputs": {}},
            }
        )
        assert len(issues) == 2
        # Stops the run rather than counting (and burning) a failed attempt per shot.
        with pytest.raises(ComfyResourceError, match="min_free_vram_gb"):
            comfy.check_vram_floor()
        assert not issubclass(ComfyResourceError, ComfyJobError)
        assert free_called
        comfy.queue(
            {"1": {"class_type": "SaveImage", "inputs": {}}}, partial_execution_targets=["1"]
        )
        assert seen_payloads[0]["partial_execution_targets"] == ["1"]


def test_previz_renders_linear_depth_pass_alongside_rgb(project: Project, analysis: Analysis):
    manifest = build_manifest(project.config, analysis, "lyrics", "style", llm=None)
    shot = manifest.shots[0]
    plan = default_previz_plan(shot, manifest)
    depth_img = previz_gen.render_blocking_frame(plan, 0.0, 160, 90, depth_pass=True)
    arr = np.asarray(depth_img)
    # Background is black (0); geometry in front of camera has brighter linear depth pixels.
    assert int(arr.min()) == 0 and int(arr.max()) > 50

    previz_gen.generate_previz(project, manifest, llm=None, shot_ids=[shot.id])
    assert shot.previz_video and project.path(shot.previz_video).is_file()
    assert shot.previz_depth_video and project.path(shot.previz_depth_video).is_file()


def test_assemble_sync_delay_xfade_grade_and_mix_smoothstep(
    project: Project, tmp_path: Path, sample_png: bytes
):
    c1 = project.root / "assets" / "videos" / "s001_a01.mp4"
    c2 = project.root / "assets" / "videos" / "s002_a01.mp4"
    make_video(c1, duration=2.0, fps=24, width=128, height=72, color="red")
    make_video(c2, duration=2.0, fps=24, width=128, height=72, color="blue")

    cfg = project.config.model_copy(deep=True)
    cfg.video.width, cfg.video.height, cfg.video.fps = 128, 72, 24
    cfg.video.transition = "blend"
    cfg.video.transition_seconds = 0.25
    cfg.video.color_grade = "portra_dusk"
    cfg.video.film_grain = 2.0
    cfg.video.strip_metadata = True
    cfg.video.sync_delay_compensation = True  # opt-in
    proj = Project(project.root, cfg)

    s1 = Shot(
        id="s001",
        index=0,
        start=0.0,
        end=1.0,
        type=ShotType.LIPSYNC,
        audio_slice=AudioSlice(path="a1.wav", start=0.0, end=1.25, head_handle=0.25),
        num_frames=33,
        video=AssetState(
            status=AssetStatus.PASSED,
            selected_attempt=1,
            attempts=[
                Attempt(
                    number=1,
                    strategy=RetryStrategy.INITIAL,
                    seed=1,
                    prompt="p",
                    path="assets/videos/s001_a01.mp4",
                    sync_delay_seconds=0.12,
                    qc=QCReport(passed=True, score=0.9),
                )
            ],
        ),
    )
    s2 = Shot(
        id="s002",
        index=1,
        start=1.0,
        end=2.0,
        type=ShotType.PERFORMANCE,
        continuity=ContinuityMode.CONTINUE,
        audio_slice=AudioSlice(path="a2.wav", start=1.0, end=2.0, head_handle=0.0),
        num_frames=33,
        still=AssetState(
            status=AssetStatus.PASSED,
            selected_attempt=1,
            attempts=[
                Attempt(
                    number=1,
                    strategy=RetryStrategy.HANDOFF,
                    seed=0,
                    prompt="(continued from previous shot handoff frame)",
                    path="assets/stills/s001_handoff.png",
                    qc=QCReport(passed=True, score=1.0),
                )
            ],
        ),
        video=AssetState(
            status=AssetStatus.PASSED,
            selected_attempt=1,
            attempts=[
                Attempt(
                    number=1,
                    strategy=RetryStrategy.INITIAL,
                    seed=2,
                    prompt="p",
                    path="assets/videos/s002_a01.mp4",
                    qc=QCReport(passed=True, score=0.9),
                )
            ],
        ),
    )
    manifest = Manifest(title="Finish", duration=2.0, fps=24, shots=[s1, s2])
    plans = assemble.plan_clips(proj, manifest, force=True)
    assert plans[0].sync_delay == pytest.approx(0.12, abs=1e-4)
    assert plans[1].head_handle == 0.0
    assert plans[1].drop_first_frame is True
    # Off by default: a measured lag is recorded but never moves the in-point unasked.
    assert project.config.video.sync_delay_compensation is False
    assert assemble.plan_clips(project, manifest, force=True)[0].sync_delay == 0.0

    # Legacy CONTINUE manifest with non-zero head_handle preserves head_handle & avoids desync.
    s2_legacy = s2.model_copy(
        update={"audio_slice": AudioSlice(path="a2.wav", start=0.75, end=2.0, head_handle=0.25)}
    )
    legacy_plans = assemble.plan_clips(
        proj, Manifest(title="Legacy", duration=2.0, fps=24, shots=[s1, s2_legacy]), force=True
    )
    assert legacy_plans[1].head_handle == pytest.approx(0.25)
    assert legacy_plans[1].drop_first_frame is False

    out_mp4 = tmp_path / "final.mp4"
    assemble.assemble(proj, manifest, out_mp4, force=True)
    info = media.probe(out_mp4)
    # Beat-preserving xfade must preserve exact 2.0s total duration!
    assert info.duration == pytest.approx(2.0, abs=0.12)

    # Mix-mode smoothstep loop synthesis.
    mix_cfg = project.config.model_copy(deep=True)
    mix_cfg.mode = ProjectMode.MIX
    mix_cfg.video.width, mix_cfg.video.height, mix_cfg.video.fps = 128, 72, 24
    mix_cfg.mix.loop_blend = "smoothstep"
    mix_cfg.mix.clip_seconds = 4.0
    mix_cfg.mix.xfade_seconds = 0.5
    mix_proj = Project(project.root, mix_cfg)
    loop_dst = tmp_path / "smooth_loop.mp4"
    chapter = MixChapter(
        id="ch01", index=0, start=0.0, end=4.0, title="Smooth", bpm=120.0, shot_ids=["s001"]
    )
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    mix_assemble._build_seamless_loop(
        [mix_assemble.LoopUnitSource("s001", c1, True, 0.0)],
        loop_dst,
        work_dir,
        chapter,
        mix_proj,
        width=128,
        height=72,
        fps=24,
    )
    assert loop_dst.is_file() and media.probe(loop_dst).duration > 1.0


def test_runner_in_place_repair_and_smart_cutaway_rescue(
    project: Project, analysis: Analysis, sample_png: bytes, tmp_path: Path
):
    from mvdirector.generation import reference

    cfg = project.config.model_copy(deep=True)
    cfg.retry.in_place_repair = True
    cfg.retry.smart_cutaway_rescue = True
    cfg.retry.smart_cutaway_max_correlation = 0.30
    cfg.video.max_clip_seconds = 4.0
    cfg.planning.max_lipsync_share = None
    proj = Project(project.root, cfg)

    manifest = build_manifest(cfg, analysis, "lyrics", "style", llm=None)
    manifest.character.appearance = "Singer with dark hair."
    sheet = tmp_path / "sheet.png"
    sheet.write_bytes(sample_png)
    reference.lock_reference(proj, manifest, sheet)

    vid_bytes = tmp_path / "sample.mp4"
    make_video(vid_bytes, duration=4.5, fps=24, width=128, height=72)
    fake = FakeComfy(png=sample_png, mp4=vid_bytes.read_bytes())
    comfy = ComfyClient(cfg.comfy, transport=fake.transport(), sleep=lambda _: None)

    still_calls = {"n": 0}

    class _RepairImageQC:
        def evaluate(self, path, shot, manifest, reference, **_kw):
            still_calls["n"] += 1
            if still_calls["n"] == 1:
                return QCReport(
                    passed=False,
                    score=0.4,
                    checks=[
                        CheckResult(
                            name="vision_spec",
                            status=CheckStatus.FAILED,
                            hard=True,
                            score=0.2,
                            detail="stray hat visible",
                            fix_hint="Remove the hat.",
                        )
                    ],
                )
            return QCReport(passed=True, score=0.95)

    class _WeakSyncVideoQC:
        def evaluate(self, path, shot, manifest, audio_slice, reference, **_kw):
            if shot.type is ShotType.LIPSYNC:
                return QCReport(
                    passed=True,
                    score=0.8,
                    checks=[
                        CheckResult(
                            name="lipsync",
                            status=CheckStatus.FAILED,
                            hard=False,
                            metrics={"correlation": 0.12, "sync_delay_seconds": 0.0},
                        )
                    ],
                )
            return QCReport(passed=True, score=0.9)

    gates = QualityGates(
        image=_RepairImageQC(),  # type: ignore[arg-type]
        video=_WeakSyncVideoQC(),  # type: ignore[arg-type]
        face_analyzer=None,
    )
    write_workflows(
        proj.root / "workflows", {"t2i": ".png", "still": ".png", "lipsync": ".mp4", "i2v": ".mp4"}
    )
    templates = load_templates(proj, ["t2i", "still", "lipsync", "i2v"])
    runner = ShotRunner(
        proj,
        manifest,
        comfy,
        gates,
        RetryPolicy(cfg.retry, cfg.seed, llm=None),
        templates,
        RunBudget(20),
    )

    target = next(s for s in manifest.shots if s.type is ShotType.LIPSYNC)
    runner.run([AssetKind.STILL, AssetKind.VIDEO], shot_ids=[target.id])

    # Attempt 2 on still used REPAIR_STILL strategy!
    assert any(a.strategy is RetryStrategy.REPAIR_STILL for a in target.still.attempts)
    # Weak lip-sync (0.12 < 0.30) triggered Smart Cutaway Rescue -> converted to BROLL!
    assert target.type is ShotType.BROLL
    assert target.fallback_from is ShotType.LIPSYNC
    assert target.video.status is AssetStatus.PASSED


def test_qa_security_boundaries_cache_suffix_object_info_and_foley_nan(tmp_path: Path):
    cache = ComfyRenderCache(tmp_path / ".cache")
    with pytest.raises(ValueError, match="Invalid cache key"):
        cache._entry_path("../escape", ".png")
    # Unsafe suffix falls back safely to .bin rather than traversing.
    entry = cache._entry_path("a" * 64, "..")
    assert entry.suffix == ".bin" and entry.parent.parent == cache.cache_dir

    comfy = ComfyClient(
        ComfyConfig(base_url="http://comfy.test"),
        transport=httpx.MockTransport(lambda _r: httpx.Response(200, json={})),
        sleep=lambda _: None,
    )
    with pytest.raises(ComfyJobError, match="Invalid ComfyUI node class_type"):
        comfy.object_info("../admin")
    comfy.close()

    # Non-finite foley cue fields are ignored rather than emitting malformed FFmpeg expressions.
    gain = foley.ducking_gain(
        np.linspace(0.0, 3.0, 50),
        [
            foley.DiegeticCue(
                audio_path=tmp_path / "x.wav", start=float("nan"), end=2.0, duck_db=-6.0
            )
        ],
    )
    assert np.all(gain == 1.0)


def test_qa_audit_regressions(tmp_path: Path):
    # 1. ColorAffine emits lutrgb when bias is non-zero.
    affine = ColorAffine(matrix=np.eye(3, dtype=np.float64), bias=np.array([0.02, -0.01, 0.0]))
    vf = affine.to_ffmpeg_colorchannelmixer()
    assert vf is not None and "lutrgb=" in vf

    # 2. convert_ui_workflow handles dict-shaped widgets_values cleanly.
    ui_dict = {
        "nodes": [
            {
                "id": 1,
                "type": "CustomNode",
                "inputs": [],
                "widgets_values": {"text": "hello", "steps": 20},
            }
        ],
        "links": [],
    }
    converted = convert_ui_workflow(ui_dict, {})
    assert converted["1"]["inputs"] == {"text": "hello", "steps": 20}

    # 3. broadcast_qa 5.1 channel weights (LFE=0.0) and low-SR guard (sr <= 3400).
    lfe_only = np.zeros((48000, 6), dtype=np.float64)
    lfe_only[:, 3] = 0.5 * np.sin(2.0 * np.pi * 60.0 * np.arange(48000) / 48000.0)
    assert broadcast_qa.integrated_lufs(lfe_only, 48000) == pytest.approx(-70.0)
    low_sr = 0.2 * np.sin(2.0 * np.pi * 200.0 * np.arange(2000) / 2000.0)
    assert np.isfinite(broadcast_qa.integrated_lufs(low_sr, 2000))

    # 4. ComfyRenderCache rejects trailing newline in key and symlinks in lookup().
    cache = ComfyRenderCache(tmp_path / ".cache")
    with pytest.raises(ValueError, match="Invalid cache key"):
        cache._entry_path("a" * 64 + "\n", ".png")
    target = tmp_path / "secret.png"
    target.write_bytes(b"secret")
    key = "b" * 64
    entry = cache._entry_path(key, ".png")
    entry.parent.mkdir(parents=True, exist_ok=True)
    entry.symlink_to(target)
    assert cache.lookup(key, tmp_path / "out.png") is False

    # 5. WorldStateLedger promotes irreversible=True on same-value update & real word-boundary test.
    ledger = WorldStateLedger()
    ledger.set_fact("glass", "state", "shattered", shot_index=0, irreversible=False)
    ledger.set_fact("glass", "state", "shattered", shot_index=1, irreversible=True)
    with pytest.raises(IrreversibleFactViolation):
        ledger.set_fact("glass", "state", "intact", shot_index=2)
    filtered = filter_look_for_state(
        "glowing lantern light, glowing glass lanternbox, blue night sky",
        {"lantern.illumination": "unlit"},
    )
    assert "glowing lantern light" not in filtered
    assert "glowing glass lanternbox" in filtered
    assert "blue night sky" in filtered

    # 6. Cache defaults to opt-in (False) and canonical_graph_key hashes only used files.
    assert ComfyConfig().cache_enabled is False
    used_file = tmp_path / "used.png"
    unused_file = tmp_path / "unused.png"
    used_file.write_bytes(b"used")
    unused_file.write_bytes(b"unused-v1")
    graph = {"1": {"class_type": "LoadImage", "inputs": {"image": "remote_used.png"}}}
    k1 = canonical_graph_key(
        graph, {"remote_used.png": used_file, "remote_unused.png": unused_file}
    )
    unused_file.write_bytes(b"unused-v2-changed")
    k2 = canonical_graph_key(
        graph, {"remote_used.png": used_file, "remote_unused.png": unused_file}
    )
    assert k1 == k2

    # 7. Attempt.regressed excludes higher-scoring regressed repairs from best_attempt().
    state = AssetState(
        attempts=[
            Attempt(
                number=1,
                strategy=RetryStrategy.INITIAL,
                seed=1,
                prompt="p",
                path="a.png",
                qc=QCReport(passed=False, score=0.60),
            ),
            Attempt(
                number=2,
                strategy=RetryStrategy.REPAIR_STILL,
                seed=2,
                prompt="p",
                path="b.png",
                regressed=True,
                qc=QCReport(passed=False, score=0.75),
            ),
        ]
    )
    best = state.best_attempt()
    assert best is not None and best.number == 1


def test_pr_review_wave_chaining_vae_mutes_blind_refs_streaming_bqa_and_auto_wiring(
    project: Project, analysis: Analysis, tmp_path: Path, sample_png: bytes, sample_mp4: bytes
):
    from typer.testing import CliRunner

    from mvdirector import api
    from mvdirector.cli import app
    from mvdirector.errors import ConfigError

    # 1. Negative VAE offset at start=0 prepends lead silence instead of clamping to 0.
    sr = 16000
    t = np.arange(sr * 2, dtype=np.float64) / sr
    tone = 0.5 * np.sin(2.0 * np.pi * 440.0 * t)
    src_wav = tmp_path / "tone.wav"
    sf.write(str(src_wav), tone, sr)
    neg_dst = tmp_path / "neg_slice.wav"
    media.prepare_audio_slice(
        src_wav, neg_dst, 0.0, 1.0, vae_offset_seconds=-0.05, mirror_pad=False
    )
    neg_samples, _ = sf.read(str(neg_dst), dtype="float64")
    lead_n = int(round(0.05 * sr))
    assert float(np.max(np.abs(neg_samples[: lead_n - 10]))) < 1e-6
    assert float(np.max(np.abs(neg_samples[lead_n + 100 :]))) > 0.2

    # 2. Streaming broadcast_qa matches full-file LUFS / True Peak / clicks across chunk boundaries.
    sig = 0.25 * np.sin(2.0 * np.pi * 1000.0 * np.arange(sr * 4) / sr)
    sig[int(1.5 * sr)] = 0.95  # click right at chunk boundary when chunk_seconds=1.5
    stream_wav = tmp_path / "stream.wav"
    sf.write(str(stream_wav), sig, sr)
    full_rep = broadcast_qa.analyze_broadcast_audio(stream_wav, chunk_seconds=60.0)
    chunk_rep = broadcast_qa.analyze_broadcast_audio(stream_wav, chunk_seconds=1.5)
    assert chunk_rep.integrated_lufs == pytest.approx(full_rep.integrated_lufs, abs=0.05)
    assert chunk_rep.true_peak_dbtp == pytest.approx(full_rep.true_peak_dbtp, abs=0.05)
    assert any(abs(c - 1.5) < 0.03 for c in chunk_rep.splice_clicks)

    # 3. CLI `mvd audio-qc` & `api.audio_qc` work end-to-end.
    api_rep = api.audio_qc(project, audio_path=stream_wav)
    assert api_rep.integrated_lufs == pytest.approx(full_rep.integrated_lufs, abs=0.05)
    cli_res = CliRunner().invoke(
        app, ["audio-qc", "-p", str(project.root), "--audio", str(stream_wav)]
    )
    assert cli_res.exit_code == 0 and "LUFS" in cli_res.output and "dBTP" in cli_res.output

    # 4. Blind-observer VisionJudge forwards extra_references into Pass 1.
    class CapturingLLM:
        def __init__(self):
            self.contents: list[list[Any]] = []

        def complete_json(self, system, content, schema):
            self.contents.append(list(content))
            if len(self.contents) == 1:
                return schema(
                    people_count=2,
                    framing="medium_close",
                    subjects="two singers with red guitar",
                    setting="stage",
                    props_and_objects="red guitar",
                )
            return VisionVerdict(spec_match=0.9, artifact_free=0.9, issues=[], fix="")

    prop_img = tmp_path / "prop.png"
    prop_img.write_bytes(sample_png)
    shot_ref = Shot(
        id="s001",
        index=0,
        start=0.0,
        end=2.0,
        type=ShotType.PERFORMANCE,
        audio_slice=AudioSlice(path="a.wav", start=0.0, end=2.0, head_handle=0.0),
        num_frames=49,
    )
    llm_cap = CapturingLLM()
    judge = VisionJudge(llm_cap, blind_observer=True)  # type: ignore[arg-type]
    judge.judge(
        [prop_img],
        shot_ref,
        Manifest(title="Ref", duration=2.0, fps=24, shots=[shot_ref]),
        AssetKind.STILL,
        extra_references=[("Prop: Red Guitar", prop_img)],
    )
    assert any(
        isinstance(blk, dict) and "Prop: Red Guitar" in str(blk.get("text", ""))
        for blk in llm_cap.contents[0]
    )

    # 5. Chained waves, RetryStrategy.HANDOFF zeroed head_handle, _upload overwrite/missing,
    #    mvd clean purging .comfy_cache, and template-gated REPAIR_STILL.
    cfg = project.config.model_copy(deep=True)
    cfg.comfy.cache_enabled = True
    cfg.video.max_clip_seconds = 2.0
    cfg.video.audio_vae_offset_seconds = 0.05
    proj = Project(project.root, cfg)
    manifest = build_manifest(proj.config, analysis, "lyrics", "style", llm=None)
    # Use 3 b-roll shots: s001(CUT) -> s002(CONTINUE) and s003(CUT) so waves batch s001+s003 first.
    s1, s2, s3 = manifest.shots[:3]
    for s in (s1, s2, s3):
        s.type = ShotType.BROLL
    s1.continuity = ContinuityMode.CUT
    s2.continuity = ContinuityMode.CONTINUE
    s3.continuity = ContinuityMode.CUT
    manifest.shots = [s1, s2, s3]
    proj.save_manifest(manifest)

    vid_src = tmp_path / "gen_vid.mp4"
    make_video(vid_src, duration=2.5, fps=24, width=128, height=72, color="blue")
    fake = FakeComfy(png=sample_png, mp4=vid_src.read_bytes())
    comfy = ComfyClient(proj.config.comfy, transport=fake.transport(), sleep=lambda _: None)
    events: list[str] = []

    class PassGate:
        def evaluate(self, *args, **kwargs):
            return QCReport(passed=True, score=0.95)

    runner = ShotRunner(
        proj,
        manifest,
        comfy,
        QualityGates(image=PassGate(), video=PassGate(), face_analyzer=None),  # type: ignore[arg-type]
        RetryPolicy(proj.config.retry, proj.config.seed, llm=None),
        load_templates(proj, ["t2i", "still", "i2v", "lipsync"]),
        RunBudget(20),
        observer=lambda ev: events.append(f"{ev.kind}:{ev.asset}:{ev.shot_id}"),
    )
    runner.run([AssetKind.STILL, AssetKind.VIDEO])

    # Wave 0 batched s1+s3 stills, then s1+s3 videos; Wave 1 did s2 handoff still then s2 video.
    assert not any("chained" in e for e in events)
    assert s2.still.selected is not None
    assert s2.still.selected.strategy is RetryStrategy.HANDOFF
    assert s2.audio_slice.head_handle == 0.0
    assert s2.audio_slice.start == pytest.approx(s2.start, abs=1e-3)

    # _upload re-uploads when file changes and raises ConfigError when missing.
    up_file = tmp_path / "dyn.png"
    up_file.write_bytes(b"v1")
    r1 = runner._upload(up_file)
    up_file.write_bytes(b"v2-longer")
    r2 = runner._upload(up_file)
    assert r1 != r2
    with pytest.raises(ConfigError, match="not found for upload"):
        runner._upload(tmp_path / "missing.png")

    # mvd clean discovers and deletes .comfy_cache files.
    clean_plan = api.clean(proj, delete=False)
    assert len(clean_plan.cache_files) > 0
    api.clean(proj, delete=True)
    assert not (proj.root / ".comfy_cache").exists()

    # 6. Automatic VideoQC join_continuity + beat_motion wiring and assemble foley SFX ducking.
    v1 = proj.root / "assets" / "videos" / "v1.mp4"
    make_video(v1, duration=1.0, fps=24, width=128, height=72, color="green")
    sfx_wav = proj.root / "inputs" / "hit.wav"
    sf.write(str(sfx_wav), 0.4 * np.sin(2.0 * np.pi * 600.0 * np.arange(sr) / sr), sr)

    vqc = VideoQC(proj.config.qc, proj.config.video, None, None, None)
    tail_frames = [np.zeros((72, 128, 3), dtype=np.uint8) for _ in range(4)]
    qc_shot = Shot(
        id="s001",
        index=0,
        start=0.0,
        end=1.0,
        type=ShotType.BROLL,
        sfx_path="inputs/hit.wav",
        audio_slice=AudioSlice(path="a.wav", start=0.0, end=1.0, head_handle=0.0),
        num_frames=25,
        video=AssetState(
            status=AssetStatus.PASSED,
            selected_attempt=1,
            attempts=[
                Attempt(
                    number=1,
                    strategy=RetryStrategy.INITIAL,
                    seed=1,
                    prompt="p",
                    path="assets/videos/v1.mp4",
                    qc=QCReport(passed=True, score=0.9),
                )
            ],
        ),
    )
    qc_manifest = Manifest(title="Auto", duration=1.0, fps=24, shots=[qc_shot])
    vrep = vqc.evaluate(
        v1,
        qc_shot,
        qc_manifest,
        sfx_wav,
        None,
        expected_size=(128, 72),
        prev_tail_frames=tail_frames,
        beat_times=[0.25, 0.75],
    )
    check_names = {c.name for c in vrep.checks}
    assert "join_continuity" in check_names and "beat_motion" in check_names

    # Assemble automatically levels & ducks shot.sfx_path over inputs.mix.
    out_sfx = tmp_path / "with_sfx.mp4"
    assemble.assemble(proj, qc_manifest, out_sfx, force=True)
    assert out_sfx.is_file() and media.probe(out_sfx).has_audio

    # 7. Template-driven REPAIR_STILL gating (works for b-roll if template accepts reference_image,
    #    skipped for singer shots if template does NOT accept reference_image).
    repair_cfg = proj.config.retry.model_copy(update={"in_place_repair": True})
    policy = RetryPolicy(repair_cfg, proj.config.seed, llm=None)
    failed_still = AssetState(
        attempts=[
            Attempt(
                number=1,
                strategy=RetryStrategy.INITIAL,
                seed=1,
                prompt="p",
                path="assets/stills/s001_a01.png",
                qc=QCReport(
                    passed=False,
                    score=0.5,
                    checks=[
                        CheckResult(
                            name="vision_artifacts", status=CheckStatus.FAILED, detail="bad"
                        )
                    ],
                ),
            )
        ]
    )
    broll_shot = Shot(
        id="s010",
        index=0,
        start=0.0,
        end=2.0,
        type=ShotType.BROLL,
        audio_slice=AudioSlice(path="a.wav", start=0.0, end=2.0, head_handle=0.0),
        num_frames=49,
        still=failed_still.model_copy(deep=True),
    )
    singer_shot = broll_shot.model_copy(
        update={"id": "s011", "type": ShotType.LIPSYNC, "still": failed_still.model_copy(deep=True)}
    )
    assert (
        policy.prepare(broll_shot, AssetKind.STILL, can_repair=True).strategy
        is RetryStrategy.REPAIR_STILL
    )
    assert (
        policy.prepare(singer_shot, AssetKind.STILL, can_repair=False).strategy
        is not RetryStrategy.REPAIR_STILL
    )

    # 8. Duet mute_spans stay aligned when video.audio_vae_offset_seconds is non-zero.
    mute_shot = Shot(
        id="s012",
        index=0,
        start=0.2,
        end=1.2,
        type=ShotType.LIPSYNC,
        audio_slice=AudioSlice(
            path="assets/audio_slices/s012.wav",
            start=0.2,
            end=1.2,
            head_handle=0.0,
            mute_spans=[TimeSpan(start=0.5, end=0.8)],
        ),
        num_frames=25,
    )
    sliced_muted = runner._ensure_audio_slice(mute_shot)
    m_samples, m_sr = sf.read(str(sliced_muted), dtype="float64")
    # Slice starts at 0.2 + 0.05 = 0.25s, so mute span [0.5, 0.8] lands at [0.25s, 0.55s].
    mid_muted = m_samples[int(0.35 * m_sr) : int(0.45 * m_sr)]
    assert float(np.max(np.abs(mid_muted))) < 1e-4

    # 9. Video xfade with tail_handle=0.0 caps overlap to 0 (no frozen-frame dissolve).
    zero_tail_cfg = proj.config.model_copy(deep=True)
    zero_tail_cfg.video.width, zero_tail_cfg.video.height, zero_tail_cfg.video.fps = 128, 72, 24
    zero_tail_cfg.video.transition = "blend"
    zero_tail_cfg.video.transition_seconds = 0.25
    zero_tail_proj = Project(proj.root, zero_tail_cfg)
    z1 = qc_shot.model_copy(
        update={"id": "z001", "index": 0, "start": 0.0, "end": 1.0, "sfx_path": None}
    )
    z2 = qc_shot.model_copy(
        update={
            "id": "z002",
            "index": 1,
            "start": 1.0,
            "end": 2.0,
            "sfx_path": None,
            "audio_slice": AudioSlice(path="b.wav", start=1.0, end=2.0, head_handle=0.0),
        }
    )
    z_out = tmp_path / "zero_tail.mp4"
    assemble.assemble(
        zero_tail_proj, Manifest(title="Z", duration=2.0, fps=24, shots=[z1, z2]), z_out, force=True
    )
    assert media.probe(z_out).duration == pytest.approx(2.0, abs=0.12)


# ------------------------------------------------------------------ Review regression tests


def test_mix_smoothstep_loop_blend_dissolves_from_outgoing_to_incoming(tmp_path: Path):
    red = make_video(tmp_path / "red.mp4", duration=2.0, color="red")
    blue = make_video(tmp_path / "blue.mp4", duration=2.0, color="blue")
    assert mix_assemble.loop_xfade_transition("xfade") == "transition=fade"
    out = tmp_path / "blend.mp4"
    transition = mix_assemble.loop_xfade_transition("smoothstep")
    media.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(red),
            "-i",
            str(blue),
            "-filter_complex",
            f"[0:v][1:v]xfade={transition}:duration=1:offset=1",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            str(out),
        ]
    )
    frames = media.extract_frames(out, 32)
    red_minus_blue = [float(f[..., 0].mean()) - float(f[..., 2].mean()) for f in frames]
    # 24 fps: frames 0-23 are red, the 1 s S-curve dissolve spans frames 24-47, then blue. The
    # dissolve must open on the outgoing clip and close on the incoming one (no pop at either end).
    assert red_minus_blue[24] > 150
    assert red_minus_blue[47] < -150
    samples = [red_minus_blue[i] for i in (24, 30, 36, 42, 47)]
    assert samples == sorted(samples, reverse=True)


@pytest.mark.parametrize(("name", "codec"), [("hit.mp3", "libmp3lame"), ("hit.m4a", "aac")])
def test_write_leveled_foley_accepts_compressed_sfx(tmp_path: Path, name: str, codec: str):
    src = tmp_path / name
    media.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=0.5",
            "-c:a",
            codec,
            str(src),
        ]
    )
    dst = foley.write_leveled_foley(src, tmp_path / "leveled.wav")
    assert sf.info(str(dst)).subtype == "PCM_24"
    leveled, _ = sf.read(str(dst))
    assert 0.01 < float(np.max(np.abs(leveled))) <= 10 ** (-6.0 / 20.0) + 1e-3


def test_strip_still_words_keeps_photograph_props():
    prop = "She studies a faded photograph of him as the camera pushes in."
    assert prompts.strip_still_words(prop) == prop
    assert prompts.strip_still_words("Film still photograph, slow push-in.") == "slow push-in."
    assert prompts.strip_still_words("Freeze frame: rain streaks down.") == "rain streaks down."


def test_analyze_project_survives_broadcast_qa_failure(
    project: Project, monkeypatch: pytest.MonkeyPatch
):
    def _boom(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("decoder exploded")

    monkeypatch.setattr(broadcast_qa, "analyze_broadcast_audio", _boom)
    monkeypatch.setattr(lyrics_mod, "load_or_transcribe_words", lambda _project: [])
    analysis = analyze_project(project)
    assert analysis.broadcast_qa is None
    assert analysis.beats  # the core analysis still completed


# --------------------------------------------------------- Ultrareview regression tests


def _clip(path: str = "assets/videos/x.mp4", number: int = 1) -> Attempt:
    return Attempt(
        number=number,
        strategy=RetryStrategy.INITIAL,
        seed=number,
        prompt="p",
        path=path,
        qc=QCReport(passed=True, score=0.9),
    )


def test_clip_timing_continue_chain_plays_past_the_dropped_seam_frame():
    fps, clip = 24, _clip()
    cut = Shot(
        id="s001",
        index=0,
        start=0.0,
        end=1.0,
        type=ShotType.BROLL,
        audio_slice=AudioSlice(path="a.wav", start=-0.25, end=1.25, head_handle=0.25),
        num_frames=41,
    )
    timing = assemble.clip_timing(cut, clip, None, sync_delay_compensation=False)
    assert (timing.first_played_frame(fps), timing.last_played_frame(24, fps)) == (6, 29)

    chained = Shot(
        id="s002",
        index=1,
        start=1.0,
        end=2.0,
        type=ShotType.BROLL,
        continuity=ContinuityMode.CONTINUE,
        audio_slice=AudioSlice(path="b.wav", start=1.0, end=2.25, head_handle=0.0),
        num_frames=33,
    )
    handoff = Attempt(
        number=1,
        strategy=RetryStrategy.HANDOFF,
        seed=0,
        prompt="handoff",
        path="assets/stills/s002_a01.png",
        qc=QCReport(passed=True, score=1.0),
    )
    timing = assemble.clip_timing(chained, clip, handoff, sync_delay_compensation=False)
    # Frame 0 repeats the previous clip's last frame and is dropped, so the last played frame
    # (the next handoff source in a chain) is index 24, not 23.
    assert timing.drop_first_frame and timing.last_played_frame(24, fps) == 24
    assert timing.continuity is ContinuityMode.CONTINUE
    # A 'continue' shot that got a freshly generated still is a plain cut at assembly.
    fresh = assemble.clip_timing(chained, clip, _clip(), sync_delay_compensation=False)
    assert fresh.continuity is ContinuityMode.CUT and not fresh.drop_first_frame


def test_continue_still_defers_in_stills_run_and_hands_off_with_unique_files(
    project: Project, analysis: Analysis, tmp_path: Path, sample_png: bytes
):
    from mvdirector import api
    from mvdirector.review import policy as review_policy

    cfg = project.config.model_copy(deep=True)
    cfg.video.max_clip_seconds = 2.0
    proj = Project(project.root, cfg)
    manifest = build_manifest(proj.config, analysis, "lyrics", "style", llm=None)
    s1, s2 = manifest.shots[:2]
    for shot in (s1, s2):
        shot.type = ShotType.BROLL
    s2.continuity = ContinuityMode.CONTINUE
    manifest.shots = [s1, s2]
    proj.save_manifest(manifest)

    clip_src = make_video(tmp_path / "gen.mp4", duration=3.0, color="blue")
    fake = FakeComfy(png=sample_png, mp4=clip_src.read_bytes())

    class PassGate:
        def evaluate(self, *args, **kwargs):
            return QCReport(passed=True, score=0.95)

    def runner() -> ShotRunner:
        return ShotRunner(
            proj,
            manifest,
            ComfyClient(proj.config.comfy, transport=fake.transport(), sleep=lambda _: None),
            QualityGates(image=PassGate(), video=PassGate(), face_analyzer=None),  # type: ignore[arg-type]
            RetryPolicy(proj.config.retry, proj.config.seed, llm=None),
            load_templates(proj, ["t2i", "i2v"]),
            RunBudget(20),
        )

    # `mvd stills`: the continuation still waits for the previous clip instead of becoming an
    # unrelated still that would turn the join into a cut.
    runner().run([AssetKind.STILL])
    assert s1.still.status is AssetStatus.PASSED
    assert s2.still.status is AssetStatus.PENDING and not s2.still.attempts

    # A take of s001 left mid-retry (e.g. an interrupted run) is not final: still no handoff.
    interrupted = proj.root / "assets" / "videos" / "s001_a01.mp4"
    interrupted.parent.mkdir(parents=True, exist_ok=True)
    interrupted.write_bytes(clip_src.read_bytes())
    failed = _clip("assets/videos/s001_a01.mp4")
    failed.qc = QCReport(passed=False, score=0.2)
    s1.video.attempts.append(failed)
    assert s1.video.status is AssetStatus.PENDING
    assert runner()._take_handoff_still(s2, s2.still) is True and not s2.still.attempts

    # `mvd videos`: s001's clip renders first, then s002 takes its handoff still on demand.
    runner().run([AssetKind.VIDEO])
    first = s2.still.selected
    assert first is not None and first.strategy is RetryStrategy.HANDOFF
    assert first.path == "assets/stills/s002_a01.png"
    assert first.params["handoff_source"] == s1.video.selected.path  # type: ignore[union-attr]
    assert s2.video.status is AssetStatus.PASSED

    # Sent back with a note, or reset without one: the identical frame is not handed off again.
    review_policy.reject(manifest, "s002", "too dark", AssetKind.STILL, proj.config)
    assert runner()._take_handoff_still(s2, s2.still) is False
    s2.still.reset(2)
    assert runner()._take_handoff_still(s2, s2.still) is False

    # A re-rendered previous clip yields a new handoff in its own file, so `mvd clean` deletes
    # only the superseded frame and never the selected one.
    rerender = proj.root / "assets" / "videos" / "s001_a09.mp4"
    rerender.write_bytes(clip_src.read_bytes())
    s1.video.attempts.append(_clip("assets/videos/s001_a09.mp4", number=9))
    s1.video.selected_attempt = 9
    assert runner()._take_handoff_still(s2, s2.still) is True
    second = s2.still.selected
    assert second is not None and second.path == "assets/stills/s002_a02.png"
    proj.save_manifest(manifest)
    plan = api.clean(proj, delete=False)
    assert proj.path("assets/stills/s002_a01.png") in plan.superseded
    assert proj.path("assets/stills/s002_a02.png") not in plan.files


def test_with_audio_track_never_rewrites_a_locked_mov_reference(tmp_path: Path):
    src = tmp_path / "turnaround.mov"
    media.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=red:s=64x64:r=24:d=1",
            "-c:v",
            "prores_ks",
            str(src),
        ]
    )
    before = src.read_bytes()
    cache = tmp_path / "cache"
    out = media.with_audio_track(src, cache)
    assert out.parent.parent == cache and out.name == "turnaround.mp4"
    assert src.read_bytes() == before  # the locked asset is untouched
    assert media.probe(out).has_audio
    stamp = out.stat().st_mtime_ns
    assert media.with_audio_track(src, cache) == out and out.stat().st_mtime_ns == stamp


def test_lock_previz_drops_a_stale_depth_pass(project: Project, analysis: Analysis, tmp_path: Path):
    manifest = build_manifest(project.config, analysis, "lyrics", "style", llm=None)
    shot = manifest.shots[0]
    shot.previz_depth_video = "assets/previz/old_depth.mp4"
    external = make_video(tmp_path / "blender.mp4", duration=1.0)
    previz_gen.lock_previz(project, manifest, external, shot.id)
    assert shot.previz_video and shot.previz_depth_video is None


def test_calibration_keeps_beat_and_telemetry_metrics_out_of_lipsync_threshold():
    from mvdirector import calibrate

    frames = [np.full((36, 64, 3), 40 * (i % 2), dtype=np.uint8) for i in range(24)]
    beat = VideoQC._beat_motion_check(frames, 24.0, [0.0, 0.5])
    assert "correlation" not in beat.metrics and "beat_correlation" in beat.metrics
    lipsync = CheckResult(
        name="lipsync",
        status=CheckStatus.PASSED,
        metrics={"correlation": 0.7, "lag_seconds": 0.04, "sync_delay_seconds": 0.04},
    )
    shot = Shot(
        id="s001",
        index=0,
        start=0.0,
        end=1.0,
        type=ShotType.LIPSYNC,
        audio_slice=AudioSlice(path="a.wav", start=0.0, end=1.0, head_handle=0.0),
        num_frames=25,
        video=AssetState(
            status=AssetStatus.PASSED,
            selected_attempt=1,
            attempts=[
                Attempt(
                    number=1,
                    strategy=RetryStrategy.INITIAL,
                    seed=1,
                    prompt="p",
                    path="v.mp4",
                    qc=QCReport(passed=True, score=0.9, checks=[lipsync, beat]),
                )
            ],
        ),
    )
    manifest = Manifest(title="Cal", duration=1.0, fps=24, shots=[shot])
    samples = calibrate.collect_samples(
        manifest, [calibrate.Label(shot.id, AssetKind.VIDEO, None, True)]
    )
    assert samples == {"correlation": [(0.7, True)]}


def test_vram_floor_waits_for_async_free_and_is_checked_once_per_phase(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    gib = 1024**3
    frees: list[int] = []
    polls_after_free: list[int] = []

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/free":
            frees.append(1)
            return httpx.Response(200, json={})
        if req.url.path == "/system_stats":
            if frees:
                polls_after_free.append(1)
            # /free only flags ComfyUI's worker: VRAM is released a couple of polls later.
            free = 20 * gib if len(polls_after_free) >= 3 else 6 * gib
            return httpx.Response(200, json={"devices": [{"vram_free": free}]})
        return httpx.Response(404)

    cfg = ComfyConfig(base_url="http://comfy.test", min_free_vram_gb=12.0)
    with ComfyClient(cfg, transport=httpx.MockTransport(handler), sleep=lambda _: None) as comfy:
        assert comfy.check_vram_floor() == pytest.approx(20.0)
        assert frees == [1]

        checks: list[int] = []
        monkeypatch.setattr(comfy, "check_vram_floor", lambda *a, **k: checks.append(1))
        monkeypatch.setattr(comfy, "queue", lambda *a, **k: "p1")
        monkeypatch.setattr(comfy, "wait", lambda *a, **k: {})
        monkeypatch.setattr(comfy, "download_output", lambda entry, output, dst: dst)
        for name in ("a.png", "b.png"):  # one phase: the loaded model is expected, not a fault
            comfy.run({}, None, tmp_path / name)  # type: ignore[arg-type]
        assert checks == [1]
        comfy.free_memory()  # next phase
        comfy.run({}, None, tmp_path / "c.png")  # type: ignore[arg-type]
        assert checks == [1, 1]


# ------------------------------------------------------- QA / stress / security regression tests


def test_broadcast_qa_rejects_nan_audio_instead_of_reporting_silence(tmp_path: Path):
    sr = 48000
    tone = 0.3 * np.sin(2 * np.pi * 440 * np.arange(sr * 2) / sr)
    tone[1000:1100] = np.nan
    bad = tmp_path / "nan.wav"
    sf.write(str(bad), tone.astype(np.float32), sr, subtype="FLOAT")
    with pytest.raises(MediaToolError, match="NaN or infinite"):
        broadcast_qa.analyze_broadcast_audio(bad)
    with pytest.raises(MediaToolError, match="nan.wav"):
        from mvdirector.audio.analysis import load_mono

        load_mono(bad)

    silent = broadcast_qa.BroadcastAudioReport(integrated_lufs=-70.0, true_peak_dbtp=-120.0)
    assert silent.problems() == ["silent"] and silent.summary().endswith("(silent)")
    clean = broadcast_qa.BroadcastAudioReport(integrated_lufs=-14.0, true_peak_dbtp=-1.0)
    assert clean.problems() == [] and clean.summary().endswith("(clean)")


def test_analyze_reports_nan_mix_as_a_clean_error(project: Project):
    sr = 22050
    mix = 0.3 * np.sin(2 * np.pi * 220 * np.arange(sr * 6) / sr)
    mix[sr : sr + 40] = np.inf
    sf.write(str(project.root / "inputs" / "mix.wav"), mix.astype(np.float32), sr, subtype="FLOAT")
    with pytest.raises(MediaToolError, match="mix.wav"):
        analyze_project(project)


def test_transitions_assemble_pairwise_without_opening_every_clip(
    project: Project, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    colors = ["red", "lime", "blue"]
    shots = []
    for i, color in enumerate(colors):
        clip = project.root / "assets" / "videos" / f"s00{i + 1}_a01.mp4"
        make_video(clip, duration=1.5, fps=24, width=64, height=36, color=color)
        shots.append(
            Shot(
                id=f"s00{i + 1}",
                index=i,
                start=float(i),
                end=float(i + 1),
                type=ShotType.BROLL,
                audio_slice=AudioSlice(path="a.wav", start=float(i), end=i + 1.25, head_handle=0.0),
                num_frames=37,
                video=AssetState(
                    status=AssetStatus.PASSED,
                    selected_attempt=1,
                    attempts=[_clip(f"assets/videos/s00{i + 1}_a01.mp4")],
                ),
            )
        )
    cfg = project.config.model_copy(deep=True)
    cfg.video.width, cfg.video.height, cfg.video.fps = 64, 36, 24
    cfg.video.transition, cfg.video.transition_seconds = "blend", 0.25
    proj = Project(project.root, cfg)
    manifest = Manifest(title="Blend", duration=3.0, fps=24, shots=shots)

    max_video_inputs = 0
    real_run = media.run

    def counting_run(cmd: list[str], **kwargs: Any) -> Any:
        nonlocal max_video_inputs
        inputs = [cmd[i + 1] for i, arg in enumerate(cmd) if arg == "-i"]
        max_video_inputs = max(max_video_inputs, sum(1 for p in inputs if p.endswith(".mp4")))
        return real_run(cmd, **kwargs)

    monkeypatch.setattr(media, "run", counting_run)
    out = tmp_path / "blend.mp4"
    assemble.assemble(proj, manifest, out, force=True)
    # Memory no longer grows with the shot count: at most two clips are decoded at once.
    assert max_video_inputs <= 2
    frames = media.extract_frames(out, 16)
    assert len(frames) == 72  # 3 shots x 24 frames, beat-locked
    rgb = [tuple(float(c) for c in f.reshape(-1, 3).mean(axis=0)) for f in frames]
    assert rgb[10][0] > 200 and rgb[30][1] > 200 and rgb[60][2] > 200  # bodies
    assert rgb[26][0] > 40 and rgb[26][1] > 40  # mid-crossfade from red into green


def test_ducking_scales_past_one_hundred_cues(tmp_path: Path):
    sr = 22050
    music = tmp_path / "music.wav"
    sf.write(str(music), 0.4 * np.sin(2 * np.pi * 220 * np.arange(sr * 180) / sr), sr)
    hush = tmp_path / "hush.wav"
    sf.write(str(hush), np.zeros(sr), sr)
    cues = [
        foley.DiegeticCue(hush, start=i * 1.2, end=i * 1.2 + 1.0, duck_db=-12.0) for i in range(140)
    ]
    out = foley.build_ducked_audio_mix(music, cues, tmp_path / "ducked.wav")
    mixed, out_sr = sf.read(str(out), dtype="float64")
    mixed = mixed if mixed.ndim == 1 else mixed.mean(axis=1)

    def rms(t0: float, t1: float) -> float:
        seg = mixed[int(t0 * out_sr) : int(t1 * out_sr)]
        return float(np.sqrt(np.mean(seg * seg)))

    # Flat middle of cue 50 (60.0-61.0 s) is ~12 dB down from the untouched tail of the song.
    assert rms(60.35, 60.65) < 0.4 * rms(170.0, 175.0)


def test_leveled_foley_reads_only_the_used_span(tmp_path: Path):
    sr = 22050
    long_sfx = tmp_path / "long.wav"
    sf.write(str(long_sfx), 0.2 * np.sin(2 * np.pi * 500 * np.arange(sr * 30) / sr), sr)
    out = foley.write_leveled_foley(long_sfx, tmp_path / "lvl.wav", max_seconds=1.0)
    assert sf.info(str(out)).duration == pytest.approx(1.0, abs=0.01)


def test_render_cache_refuses_a_symlink_out_of_the_project(
    project: Project, tmp_path: Path, sample_png: bytes
):
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (project.root / ".comfy_cache").symlink_to(outside, target_is_directory=True)
    cfg = project.config.model_copy(deep=True)
    cfg.comfy.cache_enabled = True
    proj = Project(project.root, cfg)
    manifest = Manifest(title="Cache", duration=1.0, fps=24, shots=[])
    with pytest.raises(ConfigError, match="escapes the project folder"):
        ShotRunner(
            proj,
            manifest,
            ComfyClient(proj.config.comfy, transport=FakeComfy(png=sample_png).transport()),
            QualityGates(image=None, video=None, face_analyzer=None),  # type: ignore[arg-type]
            RetryPolicy(proj.config.retry, proj.config.seed, llm=None),
            {},
            RunBudget(1),
        )


def test_ducking_ramps_are_click_free(tmp_path: Path):
    sr = 44100
    music = tmp_path / "sine.wav"
    sf.write(str(music), 0.3 * np.sin(2 * np.pi * 220 * np.arange(sr * 6) / sr), sr)
    hush = tmp_path / "hush.wav"
    sf.write(str(hush), np.zeros(sr), sr)
    cues = [foley.DiegeticCue(hush, start=1.0 + 2.5 * i, end=2.0 + 2.5 * i) for i in range(2)]
    out = foley.build_ducked_audio_mix(music, cues, tmp_path / "ducked.wav")
    mixed, _ = sf.read(str(out), dtype="float64")
    mixed = mixed if mixed.ndim == 1 else mixed.mean(axis=1)
    # A per-frame gain staircase (4096-sample WAV frames) jumped ~10x the sine's own step.
    natural_step = 0.3 * 2 * np.pi * 220 / sr
    assert float(np.max(np.abs(np.diff(mixed)))) < 1.5 * natural_step
    assert broadcast_qa.analyze_broadcast_audio(out).splice_clicks == []
