# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Regression tests for the post-v0.8.0b3 QA audit (PRs #48, #49, #50, #52)."""

from __future__ import annotations

import shutil
import subprocess
import wave
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from PIL import Image
from pydantic import BaseModel, ValidationError

from mvdirector import api, media, plugins
from mvdirector import assemble as assemble_mod
from mvdirector.audio.analysis import filter_vocal_windows_by_words
from mvdirector.audio.lyrics import Word
from mvdirector.bench.blind_viewer import generate_blind_viewer
from mvdirector.bench.config import (
    BenchmarkConfig,
    GPUCostConfig,
    ModelArm,
    open_bench_project,
    write_default_benchmark_config,
)
from mvdirector.bench.dataset import QCLabelItem
from mvdirector.bench.planning import (
    PlanningArmResult,
    PlanningQualityMetrics,
    run_planning_arm,
    run_planning_benchmark,
)
from mvdirector.bench.telemetry import CallRecord
from mvdirector.bench.vision_qc import (
    CheckCalibrationMetrics,
    VisionQCArmResult,
    _format_check_cell,
    evaluate_hard_gate,
    run_vision_qc_arm,
    run_vision_qc_benchmark,
)
from mvdirector.config import (
    ImageQCConfig,
    PluginsConfig,
    ProjectConfig,
    QCConfig,
    VideoConfig,
    VideoQCConfig,
)
from mvdirector.errors import ConfigError
from mvdirector.generation.runner import RunBudget, ShotRunner
from mvdirector.llm.base import ChatBackend, Message, Reply
from mvdirector.models import (
    Analysis,
    AssetState,
    AssetStatus,
    Attempt,
    AudioSlice,
    CheckResult,
    CheckStatus,
    ContinuityMode,
    Framing,
    LyricLine,
    Manifest,
    RetryStrategy,
    Shot,
    ShotType,
    TimeSpan,
    VisualBible,
)
from mvdirector.planning.planner import _non_singing_gaps, _populate_slice_mutes
from mvdirector.planning.segmenter import (
    _aligned_lyric_spans,
    _effective_aligned_lyric_spans,
    cut_points,
    segment,
)
from mvdirector.project import Project
from mvdirector.qc.color_drift import fit_optimal_transport_color_match
from mvdirector.qc.common import vision_identity_check
from mvdirector.qc.faces import Face, IdentityReference
from mvdirector.qc.factory import build_quality_gates
from mvdirector.qc.video import VideoQC
from mvdirector.retry import RetryPolicy
from tests.conftest import FakeComfy, offline_gates
from tests.test_qa_audit_pr50 import _make_broll_run


def test_disjoint_hallucinated_aligned_lyrics_do_not_wipe_out_lipsync_or_mute_handles() -> None:
    """Whisper hallucinations in silence outside vocal_windows must fall back to vocal_windows."""
    bars = [2.0 * i for i in range(12)]
    beats = [0.5 * i for i in range(48)]
    analysis = Analysis(
        duration=24.0,
        tempo_bpm=120.0,
        beats=beats,
        bars=bars,
        bar_intensity=[0.6] * len(bars),
        vocal_windows=[TimeSpan(start=4.0, end=16.0)],
        lyric_lines=[
            LyricLine(text="ghost 1", start=20.32, end=22.54, timing_source="aligned"),
            LyricLine(text="ghost 2", start=22.54, end=23.98, timing_source="aligned"),
        ],
    )
    assert _aligned_lyric_spans(analysis) == [TimeSpan(start=20.32, end=23.98)]
    assert _effective_aligned_lyric_spans(analysis) == []

    video = VideoConfig(min_clip_seconds=2.5, max_clip_seconds=4.0)
    segs = segment(analysis, video)
    sung = [s for s in segs if s.type is ShotType.LIPSYNC]
    assert len(sung) >= 3
    assert all(s.start >= 4.0 and s.end <= 16.0 for s in sung)

    shot = Shot(
        id="s002",
        index=1,
        start=4.0,
        end=8.0,
        type=ShotType.LIPSYNC,
        framing=Framing.CLOSE_UP,
        still_prompt="p",
        motion_prompt="m",
        audio_slice=AudioSlice(
            path="assets/audio_slices/s002.wav", start=3.75, end=8.25, head_handle=0.25
        ),
        num_frames=109,
    )
    manifest = Manifest(
        title="Test",
        duration=24.0,
        fps=24,
        visual_bible=VisualBible(concept="c", style="s", palette="p", lighting="l"),
        shots=[shot],
    )
    _populate_slice_mutes(manifest, analysis, voice_map={})
    # Because aligned lyrics had zero overlap with vocal_windows [4, 16], fallback uses [4, 16]
    # and the +/-0.20s pad covers the 0.25s lead-in without muting the shot body or handles.
    assert shot.audio_slice.mute_spans == []

    # When at least one aligned line overlaps vocal_windows, only overlapping spans are kept.
    mixed = analysis.model_copy(
        update={
            "lyric_lines": [
                LyricLine(text="real", start=8.0, end=12.0, timing_source="aligned"),
                LyricLine(text="ghost", start=20.32, end=23.98, timing_source="aligned"),
            ]
        }
    )
    assert _effective_aligned_lyric_spans(mixed) == [TimeSpan(start=8.0, end=12.0)]


def test_blind_viewer_html_renders_solo_and_duet_cast_lists(tmp_path: Path) -> None:
    """Solo shots (characters=[]) and multi-cast shots must render Cast non-empty in HTML."""
    m = Manifest(
        title="SoloAndDuet",
        duration=8.0,
        fps=24,
        visual_bible=VisualBible(concept="c", style="s", palette="p", lighting="l"),
        shots=[
            Shot(
                id="s001",
                index=0,
                start=0.0,
                end=4.0,
                type=ShotType.LIPSYNC,
                framing=Framing.CLOSE_UP,
                character="aria",
                characters=[],
                still_prompt="Aria singing.",
                motion_prompt="Sings.",
                audio_slice=AudioSlice(path="a.wav", start=0.0, end=4.0, head_handle=0.0),
                num_frames=97,
            ),
            Shot(
                id="s002",
                index=1,
                start=4.0,
                end=8.0,
                type=ShotType.LIPSYNC,
                framing=Framing.MEDIUM_CLOSE,
                character=None,
                characters=["aria", "kael"],
                still_prompt="Both singing.",
                motion_prompt="Sing.",
                audio_slice=AudioSlice(path="b.wav", start=4.0, end=8.0, head_handle=0.0),
                num_frames=97,
            ),
        ],
    )
    m_path = tmp_path / "manifest.json"
    m_path.write_text(m.model_dump_json(), encoding="utf-8")
    res = PlanningArmResult(
        track="t1",
        source_project=str(tmp_path),
        arm_id="A",
        arm_label="Arm A",
        provider="gemini",
        model="gemini-3.8-flash",
        attach_audio=True,
        output_dir=str(tmp_path),
        manifest_path=str(m_path),
        status="ok",
    )
    html_path, _ = generate_blind_viewer([res], tmp_path / "viewer.html", seed=1)
    html_text = html_path.read_text(encoding="utf-8")
    assert "sh.characters && sh.characters.length" in html_text
    assert "castList.length" in html_text


def test_i2v_fps_played_tail_and_color_drift_extract_at_source_fps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When i2v_fps produces a 12fps clip, handoff and color-drift indices scale to 12fps."""
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg/ffprobe required")

    (tmp_path / "inputs").mkdir(parents=True)
    (tmp_path / "assets" / "videos").mkdir(parents=True)
    (tmp_path / "assets" / "stills").mkdir(parents=True)
    sig = (0.1 * np.sin(2 * np.pi * 440.0 * np.arange(8 * 48000) / 48000) * 32767).astype(np.int16)
    with wave.open(str(tmp_path / "inputs" / "mix.wav"), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(48000)
        wf.writeframes(sig.tobytes())

    # Generate a real 12 fps 57-frame (4.75s) clip for s001 and s002.
    raw = bytes(np.full((57, 64, 64, 3), (180, 50, 50), dtype=np.uint8))
    for sid in ("s001", "s002"):
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                "64x64",
                "-r",
                "12",
                "-i",
                "pipe:0",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-pix_fmt",
                "yuv420p",
                str(tmp_path / "assets" / "videos" / f"{sid}.mp4"),
            ],
            input=raw,
            check=True,
        )
    Image.new("RGB", (64, 64), (180, 50, 50)).save(tmp_path / "assets" / "stills" / "s002.png")

    cfg = ProjectConfig(
        title="HalfFPS",
        video=VideoConfig(
            width=64,
            height=64,
            fps=24,
            i2v_fps=12,
            color_drift_correction=True,
        ),
    )
    proj = Project(root=tmp_path, config=cfg)
    s1 = Shot(
        id="s001",
        index=0,
        start=0.0,
        end=4.0,
        type=ShotType.BROLL,
        framing=Framing.WIDE,
        still_prompt="p",
        motion_prompt="m",
        audio_slice=AudioSlice(path="a.wav", start=0.0, end=4.5, head_handle=0.25),
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
    )
    s2 = Shot(
        id="s002",
        index=1,
        start=4.0,
        end=8.0,
        type=ShotType.BROLL,
        framing=Framing.WIDE,
        continuity=ContinuityMode.CONTINUE,
        still_prompt="p",
        motion_prompt="m",
        audio_slice=AudioSlice(path="b.wav", start=3.75, end=8.25, head_handle=0.25),
        num_frames=109,
        still=AssetState(
            status=AssetStatus.PASSED,
            selected_attempt=1,
            attempts=[
                Attempt(
                    number=1,
                    strategy=RetryStrategy.HANDOFF,
                    seed=1,
                    prompt="p",
                    path="assets/stills/s002.png",
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
                    prompt="m",
                    path="assets/videos/s002.mp4",
                )
            ],
        ),
    )
    manifest = Manifest(
        title="HalfFPS",
        duration=8.0,
        fps=24,
        visual_bible=VisualBible(concept="c", style="s", palette="p", lighting="l"),
        shots=[s1, s2],
    )
    runner = ShotRunner(
        proj,
        manifest,
        FakeComfy(),  # type: ignore[arg-type]
        offline_gates(cfg),
        RetryPolicy(cfg.retry, cfg.planning, None),
        {},
        RunBudget(10),
    )
    tail = runner._played_tail(s1)
    assert tail is not None
    # At 12 fps, end_sec = 0.25 + 95/24 = 4.2083s -> frame ~50 (NOT frame 101 > 57!).
    assert 48 <= tail[1] <= 52

    extracted_indices: list[tuple[str, int]] = []
    orig_extract = media.extract_frame_at_index

    def _spy_extract(video: Path, frame_index: int, dst: Path) -> Path:
        extracted_indices.append((video.name, frame_index))
        return orig_extract(video, frame_index, dst)

    monkeypatch.setattr(media, "extract_frame_at_index", _spy_extract)
    assemble_mod.assemble(proj, manifest, tmp_path / "out.mp4", force=True)
    # s002.mp4 has head_handle=0.25 at 12fps -> frame index 3 (not 6 at 24fps!).
    curr_calls = [idx for name, idx in extracted_indices if name == "s002.mp4"]
    assert curr_calls == [3]


def test_conditioning_audio_rejects_in_project_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A symlink at <shot>.cond.wav pointing to project.yaml inside root must never be written."""
    (tmp_path / "inputs").mkdir(parents=True)
    (tmp_path / "assets" / "audio_slices").mkdir(parents=True)
    cfg_file = tmp_path / "project.yaml"
    cfg_file.write_text("title: SecretConfig\n", encoding="utf-8")
    raw_wav = tmp_path / "assets" / "audio_slices" / "s001.wav"
    raw_wav.write_bytes(b"RIFFraw")
    cond_symlink = tmp_path / "assets" / "audio_slices" / "s001.cond.wav"
    cond_symlink.symlink_to(cfg_file)

    called = False

    def _bad_cond(raw_p: Path, dst_p: Path, _shot: Shot, _analysis: Any) -> Path:
        nonlocal called
        called = True
        dst_p.write_bytes(b"OVERWRITTEN")
        return dst_p

    class _EP:
        name = "test_cond"

        def load(self) -> Any:
            return _bad_cond

    plugins.clear_cache()
    monkeypatch.setattr(
        metadata,
        "entry_points",
        lambda *, group: [_EP()] if group == plugins.AUDIO_CONDITIONERS_GROUP else [],
    )
    try:
        cfg = ProjectConfig(
            title="SymlinkTest", plugins=PluginsConfig(audio_conditioner="test_cond")
        )
        proj = Project(root=tmp_path, config=cfg)
        shot = Shot(
            id="s001",
            index=0,
            start=0.0,
            end=4.0,
            type=ShotType.LIPSYNC,
            framing=Framing.CLOSE_UP,
            still_prompt="p",
            motion_prompt="m",
            audio_slice=AudioSlice(
                path="assets/audio_slices/s001.wav", start=0.0, end=4.0, head_handle=0.0
            ),
            num_frames=97,
        )
        manifest = Manifest(
            title="SymlinkTest",
            duration=4.0,
            fps=24,
            visual_bible=VisualBible(concept="c", style="s", palette="p", lighting="l"),
            shots=[shot],
        )
        runner = ShotRunner(
            proj,
            manifest,
            FakeComfy(),  # type: ignore[arg-type]
            offline_gates(cfg),
            RetryPolicy(cfg.retry, cfg.planning, None),
            {},
            RunBudget(10),
        )
        monkeypatch.setattr(runner, "_ensure_audio_slice", lambda _s: raw_wav)
        result_path = runner._ensure_conditioning_audio(shot)
        assert result_path == raw_wav
        assert not called
        assert cfg_file.read_text(encoding="utf-8") == "title: SecretConfig\n"
    finally:
        plugins.clear_cache()


def test_video_qc_plugin_rejects_non_finite_and_bool_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """VideoQC._plugin_checks rejects NaN/Inf/bool plugin metrics cleanly."""
    for bad_metrics in ({"bad": float("nan")}, {"bad": float("inf")}, {"bad": True}):

        def _plugin(
            _frames: Any,
            _fps: float,
            _shot: Shot,
            _m: Manifest,
            _a: Path,
            _bad: dict[str, Any] = bad_metrics,
        ) -> list[Any]:
            return [
                CheckResult.model_construct(
                    name="custom",
                    status=CheckStatus.PASSED,
                    hard=False,
                    score=0.9,
                    weight=1.0,
                    metrics=_bad,  # type: ignore[arg-type]
                )
            ]

        class _EP:
            name = "bad_metrics"

            def load(self) -> Any:
                return _plugin

        plugins.clear_cache()
        monkeypatch.setattr(
            metadata,
            "entry_points",
            lambda *, group: [_EP()] if group == plugins.QC_CHECKS_GROUP else [],
        )
        try:
            cfg = ProjectConfig(qc=QCConfig(on_missing_dependency="fail"))
            vqc = VideoQC(cfg.qc, cfg.video, None, None, None, qc_check_plugins=["bad_metrics"])
            shot = Shot(
                id="s001",
                index=0,
                start=0.0,
                end=4.0,
                type=ShotType.BROLL,
                framing=Framing.WIDE,
                still_prompt="p",
                motion_prompt="m",
                audio_slice=AudioSlice(path="a.wav", start=0.0, end=4.0, head_handle=0.0),
                num_frames=97,
            )
            manifest = Manifest(
                title="T",
                duration=4.0,
                fps=24,
                visual_bible=VisualBible(concept="c", style="s", palette="p", lighting="l"),
                shots=[shot],
            )
            res = vqc._plugin_checks([], 24.0, shot, manifest, tmp_path / "a.wav")
            assert len(res) == 1
            assert res[0].status is CheckStatus.FAILED and res[0].hard
            assert "invalid metric" in res[0].detail
        finally:
            plugins.clear_cache()


def test_vision_identity_uses_visible_clamped_bbox_height_and_two_frame_sampling(
    tmp_path: Path,
) -> None:
    """Partially off-screen bbox must be sized by visible pixels, not unclamped y2 - y1."""
    ref_img = tmp_path / "ref.png"
    Image.new("RGB", (128, 128), (120, 120, 120)).save(ref_img)
    calls = 0

    class _Judge:
        disabled_reason = None

        def judge_face_identity(self, _crop: np.ndarray, _refs: list[Path]) -> Any:
            nonlocal calls
            calls += 1
            return type("V", (), {"same_person": 0.9, "reason": "ok"})()

    class _DummyAnalyzer:
        def faces(self, _img: np.ndarray) -> list[Face]:
            return [Face(height_fraction=0.5, bbox=(10, 10, 110, 110))]

    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    ref = IdentityReference(_DummyAnalyzer(), [ref_img])  # type: ignore[arg-type]
    # Only 10px visible inside the 200x200 frame despite raw bbox height of 150px!
    res = vision_identity_check(
        _Judge(),  # type: ignore[arg-type]
        [(frame, [Face(height_fraction=0.75, bbox=(50, -140, 150, 10))])],
        ref,
        min_face_px=100,
        hard_min=0.5,
        good=0.8,
    )
    assert res.status is CheckStatus.SKIPPED
    assert "10px tall" in res.detail
    assert calls == 0

    # Two-frame clip with vision_identity_frames=2 samples both frames.
    cfg = ProjectConfig()
    cfg.qc.video.vision_identity_frames = 2
    vqc = VideoQC(cfg.qc, cfg.video, _DummyAnalyzer(), None, None)  # type: ignore[arg-type]
    frames = [np.full((64, 64, 3), i, dtype=np.uint8) for i in range(2)]
    sampled = vqc._vision_identity_samples(frames)
    assert [int(f[0, 0, 0]) for f, _ in sampled] == [0, 1]


def test_boundary_hardening_intercut_words_gaps_and_benchmark_budgets(tmp_path: Path) -> None:
    """NaN, negative, and invalid boundary args fail safely or fall back cleanly."""
    plans, manifest, analysis = _make_broll_run(
        tmp_path, bpm=136.0, shot_seconds=4.0, n_shots=3, handle=0.25, intensity=0.90
    )
    assert (
        assemble_mod.intercut_plans(plans, manifest, analysis, min_intensity=float("nan")) == plans
    )
    out_neg = assemble_mod.intercut_plans(
        plans, manifest, analysis, min_intensity=0.70, anticipation_frames=-5
    )
    assert sum(p.frames for p in out_neg) == sum(p.frames for p in plans)

    # clip_timing guards non-finite optimal_in_offset.
    bad_att = Attempt.model_construct(
        number=1,
        strategy=RetryStrategy.INITIAL,
        seed=1,
        prompt="m",
        sync_delay_seconds=0.0,
        optimal_in_offset=float("nan"),
    )
    timing = assemble_mod.clip_timing(
        manifest.shots[0], bad_att, None, sync_delay_compensation=False
    )
    assert timing.sync_delay == 0.0

    # filter_vocal_windows_by_words handles min_words_per_window=0 and blank tokens.
    wins = [TimeSpan(start=0.0, end=5.0)]
    assert (
        filter_vocal_windows_by_words(
            wins,
            [Word(word="   ", start=1.0, end=2.0), Word(word="hi", start=10.0, end=11.0)],
            min_words_per_window=0,
        )
        == wins
    )
    assert _non_singing_gaps(0.0, 5.0, [(1.0, 2.0)], min_gap=-1.0) == [(0.0, 0.8), (2.2, 5.0)]
    assert _non_singing_gaps(0.0, 5.0, [(1.0, 2.0)], min_gap=float("nan")) == [
        (0.0, 0.8),
        (2.2, 5.0),
    ]

    # Benchmark budgets reject NaN, negative, and booleans.
    arm = ModelArm(id="A", label="A", provider="gemini", model="gemini-3.8-flash")
    bcfg = BenchmarkConfig(arms=[arm])
    dummy_item = QCLabelItem(id="p:s1:a1", project=str(tmp_path), shot_id="s001", good=True)
    for bad_budget in (float("nan"), -1.0, True):
        with pytest.raises(ConfigError, match="max_budget_usd"):
            run_planning_benchmark([tmp_path], bcfg, tmp_path / "o", max_budget_usd=bad_budget)  # type: ignore[arg-type]
        with pytest.raises(ConfigError, match="max_budget_usd"):
            run_vision_qc_arm(
                [dummy_item],
                arm,
                tmp_path / "o",
                gpu_costs=GPUCostConfig(),
                max_budget_usd=bad_budget,  # type: ignore[arg-type]
            )
        with pytest.raises(ConfigError, match="max_budget_usd"):
            run_vision_qc_benchmark([dummy_item], bcfg, tmp_path / "o", max_budget_usd=bad_budget)  # type: ignore[arg-type]

    # evaluate_hard_gate rejects NaN / negative spend; _format_check_cell guards None threshold.
    with pytest.raises(ConfigError):
        evaluate_hard_gate(
            [],
            spec_min=float("nan"),
            artifacts_min=0.5,
            character_min=0.5,
            mean_llm_usd_per_still=0.01,
            gpu_costs=GPUCostConfig(),
        )
    res_null_cut = VisionQCArmResult(
        arm_id="A",
        arm_label="A",
        provider="gemini",
        model="m",
        per_check={
            "vision_spec": CheckCalibrationMetrics(check="vision_spec", balanced_accuracy=0.8)
        },
    )
    assert _format_check_cell(res_null_cut, "vision_spec") == "n/a"

    # Bench result models reject NaN/Inf.
    with pytest.raises(ValidationError):
        CallRecord(
            schema_name="S",
            model="m",
            latency_seconds=float("nan"),
            attempts=1,
            schema_retries=0,
            first_pass_ok=True,
            succeeded=True,
            truncated=False,
            refused=False,
            has_audio=False,
            has_images=False,
            input_tokens=0,
            output_tokens=0,
            thoughts_tokens=0,
            cached_tokens=0,
            estimated_usd=0.0,
        )
    with pytest.raises(ValidationError):
        PlanningQualityMetrics(fallback_rate=float("inf"))


def test_planning_benchmark_case_insensitive_tracks_sanitized_yaml_and_non_dict_config(
    tmp_path: Path,
) -> None:
    """Case-colliding project names get distinct track slugs and isolated yaml sets audio_listen."""
    # Non-dict project.yaml raises clean ConfigError in open_bench_project.
    bad_proj = tmp_path / "bad_yaml"
    bad_proj.mkdir()
    (bad_proj / "project.yaml").write_text("- not-a-mapping\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="YAML mapping"):
        open_bench_project(bad_proj)

    # Atomic starter bench.yaml write works and loads back cleanly.
    starter = write_default_benchmark_config(tmp_path / "bench.yaml")
    assert starter.is_file()

    # Two folders whose names differ only in case get distinct track slugs.
    p1 = tmp_path / "dir1" / "MyTrack"
    p2 = tmp_path / "dir2" / "mytrack"
    analysis = Analysis(
        duration=8.0,
        tempo_bpm=120.0,
        beats=[0.5 * i for i in range(16)],
        bars=[2.0 * i for i in range(4)],
        bar_intensity=[0.6] * 4,
        vocal_windows=[],
    )
    for p in (p1, p2):
        (p / "inputs").mkdir(parents=True)
        (p / "inputs" / "style.md").write_text("Minimal style.", encoding="utf-8")
        (p / "inputs" / "lyrics.txt").write_text("", encoding="utf-8")
        (p / "project.yaml").write_text(
            yaml.safe_dump({"title": p.name, "planning": {"audio_listen": "always"}}),
            encoding="utf-8",
        )
        (p / "analysis.json").write_text(analysis.model_dump_json(), encoding="utf-8")

    class _ErrLLM:
        def complete_json(self, **_kw: Any) -> Any:
            raise ConfigError("stop")

    arm = ModelArm(id="A", label="A", provider="gemini", model="gemini-3.8-flash")
    results = run_planning_benchmark(
        [p1, p2],
        BenchmarkConfig(arms=[arm]),
        tmp_path / "bench_out",
        llm_factory=lambda _a, _e: _ErrLLM(),
    )
    assert [r.track.casefold() for r in results] == ["mytrack", "mytrack-2"]
    # Sanitized project.yaml in arm folder matches runtime audio_listen='never'.
    arm_yaml = yaml.safe_load((Path(results[0].output_dir) / "project.yaml").read_text())
    assert arm_yaml["planning"]["audio_listen"] == "never"

    # Single run_planning_arm also writes audio_listen: never.
    r_single = run_planning_arm(p1, arm, tmp_path / "single", llm_factory=lambda _a, _e: _ErrLLM())
    single_yaml = yaml.safe_load((Path(r_single.output_dir) / "project.yaml").read_text())
    assert single_yaml["planning"]["audio_listen"] == "never"


def test_segmenter_broll_min_clip_irregular_bars_early_snap() -> None:
    """Irregular bar grids where broll_steps == allowed_steps still allow B-roll section snap."""
    # Median bar length is 2.6s (so both allowed_steps and broll_steps are [2, 1]),
    # while bar 0 is an irregular 1.8s instrumental bar followed by a new section at bar 1.
    bars = [0.0, 1.8, 4.4, 7.0, 9.6, 12.2]
    analysis = Analysis(
        duration=14.8,
        tempo_bpm=92.0,
        beats=[0.65 * i for i in range(23)],
        bars=bars,
        bar_intensity=[0.3, 0.3, 0.3, 0.9, 0.9, 0.9],
        vocal_windows=[TimeSpan(start=1.8, end=14.8)],
        lyric_lines=[
            LyricLine(text="Verse", section="Verse", start=1.8, end=14.8, timing_source="aligned")
        ],
    )
    video = VideoConfig(
        min_clip_seconds=2.5,
        broll_min_clip_seconds=1.5,
        max_clip_seconds=6.0,
        dynamic_pacing=True,
    )
    cuts = cut_points(analysis, video, snap_sections=True)
    assert 1.8 in cuts


def test_bool_rejection_on_new_fields_and_factory_plugin_summary() -> None:
    """New numeric fields reject booleans; QualityGates reports configured QC plugins."""
    with pytest.raises(ValidationError, match="not a boolean"):
        ImageQCConfig(vision_identity_hard_min=False)  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="not a boolean"):
        VideoQCConfig(vision_identity_frames=True)  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="not a boolean"):
        QCConfig(pass_score=True)  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="not a boolean"):
        Attempt(
            number=1,
            strategy=RetryStrategy.INITIAL,
            seed=1,
            prompt="p",
            optimal_in_offset=True,  # type: ignore[arg-type]
        )
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="not a boolean"):
        fit_optimal_transport_color_match([frame], [frame], regularization=True)  # type: ignore[arg-type]

    cfg = ProjectConfig(
        qc=QCConfig(on_missing_dependency="skip", vision_enabled=False),
        plugins=PluginsConfig(qc_checks=["custom_qc"]),
    )
    gates = build_quality_gates(cfg, None)
    try:
        active, _ = gates.check_summary()
        assert "custom_qc" in active
    finally:
        gates.close()


def test_api_validate_project_fallback_audio_llm_and_check_llm_disabled_qc_llm(
    tmp_path: Path,
) -> None:
    """validate_project checks qc.llm fallback audio_listen; check_llm respects qc.llm.enabled."""
    (tmp_path / "inputs").mkdir(parents=True)
    (tmp_path / "inputs" / "mix.wav").write_bytes(b"RIFF")
    (tmp_path / "inputs" / "style.md").write_text("Style.", encoding="utf-8")
    (tmp_path / "inputs" / "lyrics.txt").write_text("hi", encoding="utf-8")
    (tmp_path / "project.yaml").write_text(
        yaml.safe_dump(
            {
                "title": "FallbackAudio",
                "llm": {"enabled": False},
                "planning": {"audio_listen": "auto"},
                "qc": {
                    "vision_enabled": False,
                    "llm": {
                        "enabled": True,
                        "provider": "gemini",
                        "model": "gemini-3.8-flash",
                        "api_key_env": "MVD_MISSING_FALLBACK_KEY",
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    checks = api.validate_project(tmp_path)
    audio_checks = [c for c in checks if "Audio listen LLM" in c.label]
    assert len(audio_checks) == 1
    assert not audio_checks[0].ok

    # Explicit qc.llm.enabled: false turns off vision_needed in check_llm.
    (tmp_path / "project.yaml").write_text(
        yaml.safe_dump(
            {
                "title": "DisabledQCLLM",
                "llm": {"enabled": False},
                "qc": {
                    "vision_enabled": True,
                    "llm": {"enabled": False, "provider": "gemini", "model": "gemini-3.8-flash"},
                },
            }
        ),
        encoding="utf-8",
    )
    outcome = api.check_llm(tmp_path)
    assert not outcome.vision_needed
    assert not outcome.tested_images


def test_chat_backend_latency_safe_when_hook_set_during_call() -> None:
    """Attaching a telemetry hook mid-call must report realistic elapsed latency, not uptime."""

    class _Schema(BaseModel):
        ok: bool

    captured: list[float] = []

    class _LateHookBackend(ChatBackend):
        def _send(self, system: str, messages: list[Message], schema: type[BaseModel]) -> Reply:
            self.set_telemetry_hook(lambda ev: captured.append(ev.latency_seconds))
            return Reply('{"ok": true}')

    backend = _LateHookBackend(model="m", max_tokens=256)
    out = backend.complete_json(system="s", content=[], schema=_Schema)
    assert out.ok is True
    assert len(captured) == 1
    assert 0.0 <= captured[0] < 10.0
