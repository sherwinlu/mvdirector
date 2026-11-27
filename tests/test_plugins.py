# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Tests for plugin hooks, clip in-point offsets, encoder probing, and parallel assembly."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf
from pydantic import ValidationError

from mvdirector import api, assemble, media, plugins
from mvdirector.comfy.workflow import Binding, OutputSpec, WorkflowBindings, WorkflowTemplate
from mvdirector.config import (
    ComfyConfig,
    PluginsConfig,
    ProjectConfig,
    QCConfig,
    VideoConfig,
    load_config,
)
from mvdirector.errors import ConfigError
from mvdirector.generation.runner import ShotRunner
from mvdirector.models import (
    Analysis,
    AssetKind,
    AssetState,
    AssetStatus,
    Attempt,
    AudioSlice,
    CheckResult,
    CheckStatus,
    ContinuityMode,
    Framing,
    Manifest,
    RetryStrategy,
    Shot,
    ShotType,
    VisualBible,
)
from mvdirector.planning.segmenter import cut_points
from mvdirector.qc.color_drift import fit_optimal_transport_color_match
from mvdirector.qc.scoring import aggregate
from mvdirector.qc.video import VideoQC

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg and ffprobe are required",
)


class _FakeEP:
    def __init__(self, name: str, obj: Any) -> None:
        self.name = name
        self._obj = obj

    def load(self) -> Any:
        return self._obj


def _patch_entry_points(
    monkeypatch: pytest.MonkeyPatch, registry: dict[str, list[_FakeEP]]
) -> None:
    plugins.clear_cache()

    def _fake_eps(*, group: str) -> list[_FakeEP]:
        return list(registry.get(group, ()))

    monkeypatch.setattr(metadata, "entry_points", _fake_eps)


@pytest.fixture(autouse=True)
def _reset_caches() -> Any:
    plugins.clear_cache()
    media.clear_encoder_cache()
    yield
    plugins.clear_cache()
    media.clear_encoder_cache()


# --------------------------------------------------------------------------- plugin registry


@pytest.mark.parametrize(
    "bad",
    ["", "Upper", "pkg.mod:fn", "../evil", "a/b", "1abc", "a-b", "a" * 65, "has space"],
)
def test_plugin_name_validation_rejects_paths_and_bad_identifiers(bad: str) -> None:
    with pytest.raises(ConfigError):
        plugins.validate_plugin_name(bad)
    with pytest.raises(ValidationError):
        PluginsConfig(pacing=bad)
    with pytest.raises(ValidationError):
        PluginsConfig(qc_checks=[bad])


def test_plugins_config_load_config_includes_path_and_rejects_duplicates(tmp_path: Path) -> None:
    bad_yaml = tmp_path / "bad_plugin.yaml"
    bad_yaml.write_text("plugins:\n  pacing: '../evil'\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="bad_plugin.yaml"):
        load_config(bad_yaml)

    with pytest.raises(ValidationError, match="Duplicate plugin name"):
        PluginsConfig(qc_checks=["check_a", "check_a"])


def test_plugin_group_allowlist_and_duplicate_entry_points(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    with pytest.raises(ConfigError, match="Unsupported plugin group"):
        plugins.list_plugins("console_scripts")
    with pytest.raises(ConfigError, match="Unsupported plugin group"):
        plugins.load_plugin("pytest11", "platformdirs")

    first_fn = object()
    second_fn = object()
    _patch_entry_points(
        monkeypatch,
        {
            plugins.PACING_GROUP: [
                _FakeEP("dup_pacer", first_fn),
                _FakeEP("dup_pacer", second_fn),
            ]
        },
    )
    with caplog.at_level("WARNING", logger="mvdirector.plugins"):
        eps = plugins.list_plugins(plugins.PACING_GROUP)
    assert eps["dup_pacer"].load() is first_fn
    assert "Duplicate entry point" in caplog.text


def test_load_plugin_concurrent_cold_cache_loads_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    class _SlowUniqueEP(_FakeEP):
        def load(self) -> Any:
            nonlocal calls
            calls += 1
            time.sleep(0.02)
            return object()

    _patch_entry_points(monkeypatch, {plugins.PACING_GROUP: [_SlowUniqueEP("slow_pacer", None)]})

    with ThreadPoolExecutor(max_workers=16) as pool:
        results = list(
            pool.map(lambda _: plugins.load_plugin(plugins.PACING_GROUP, "slow_pacer"), range(16))
        )

    assert calls == 1
    assert len({id(obj) for obj in results}) == 1
    assert len(plugins._INFLIGHT_LOCKS) == 0


def test_load_plugin_releases_inflight_locks_on_errors_and_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0

    class _FlakyEP(_FakeEP):
        def load(self) -> Any:
            nonlocal attempts
            attempts += 1
            time.sleep(0.01)
            if attempts == 1:
                raise RuntimeError("transient load failure")
            return "recovered_plugin"

    class _AlwaysFailEP(_FakeEP):
        def load(self) -> Any:
            raise RuntimeError("permanent load failure")

    _patch_entry_points(
        monkeypatch,
        {
            plugins.QC_CHECKS_GROUP: [
                _FlakyEP("flaky_qc", None),
                _AlwaysFailEP("broken_qc", None),
            ]
        },
    )

    # Repeated unknown plugin lookups must not leak lock entries.
    for i in range(50):
        with pytest.raises(ConfigError, match="Unknown plugin"):
            plugins.load_plugin(plugins.QC_CHECKS_GROUP, f"missing_{i}")
    assert len(plugins._INFLIGHT_LOCKS) == 0

    # Concurrent failures must release the lock entry once all waiters unwind.
    def _try_broken(_: int) -> bool:
        try:
            plugins.load_plugin(plugins.QC_CHECKS_GROUP, "broken_qc")
        except RuntimeError:
            return True
        return False

    with ThreadPoolExecutor(max_workers=8) as pool:
        assert all(pool.map(_try_broken, range(8)))
    assert len(plugins._INFLIGHT_LOCKS) == 0

    # First call fails transiently; concurrent waiters / retry recover cleanly with zero leaks.
    def _try_flaky(_: int) -> str:
        try:
            return str(plugins.load_plugin(plugins.QC_CHECKS_GROUP, "flaky_qc"))
        except RuntimeError:
            return "failed"

    with ThreadPoolExecutor(max_workers=6) as pool:
        outcomes = list(pool.map(_try_flaky, range(6)))
    assert "failed" in outcomes
    assert "recovered_plugin" in outcomes
    assert plugins.load_plugin(plugins.QC_CHECKS_GROUP, "flaky_qc") == "recovered_plugin"
    assert len(plugins._INFLIGHT_LOCKS) == 0


def test_load_plugin_unknown_and_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def _fn() -> str:
        return "ok"

    class _CountingEP(_FakeEP):
        def load(self) -> Any:
            nonlocal calls
            calls += 1
            return super().load()

    _patch_entry_points(monkeypatch, {plugins.PACING_GROUP: [_CountingEP("my_pacer", _fn)]})
    with pytest.raises(ConfigError, match="Unknown plugin"):
        plugins.load_plugin(plugins.PACING_GROUP, "missing_pacer")

    assert plugins.load_plugin(plugins.PACING_GROUP, "my_pacer") is _fn
    assert plugins.load_plugin(plugins.PACING_GROUP, "my_pacer") is _fn
    assert calls == 1


# --------------------------------------------------------------------------- QC checks plugin


def _make_shot(
    shot_id: str = "s001",
    *,
    shot_type: ShotType = ShotType.BROLL,
    start: float = 0.0,
    end: float = 2.0,
    head: float = 0.25,
    tail: float = 0.25,
) -> Shot:
    return Shot(
        id=shot_id,
        index=0,
        start=start,
        end=end,
        type=shot_type,
        framing=Framing.MEDIUM,
        num_frames=49,
        audio_slice=AudioSlice(
            path=f"assets/audio/{shot_id}.wav",
            start=max(0.0, start - head),
            end=end + tail,
            head_handle=head,
        ),
    )


def _make_manifest(shots: list[Shot]) -> Manifest:
    return Manifest(
        title="Test",
        duration=max((s.end for s in shots), default=4.0),
        fps=24,
        visual_bible=VisualBible(concept="c", style="s", palette="p", lighting="l"),
        shots=shots,
    )


def test_video_qc_plugin_checks_and_failure_containment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    shot = _make_shot()
    manifest = _make_manifest([shot])
    frames = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(4)]

    def _good_check(
        _frames: Any, _fps: float, _shot: Shot, _manifest: Manifest, _audio: Path
    ) -> list[CheckResult]:
        return [CheckResult(name="custom_ok", status=CheckStatus.PASSED, score=0.9, weight=0.0)]

    def _boom(*_args: Any) -> list[CheckResult]:
        raise RuntimeError("boom")

    def _bad_return(*_args: Any) -> Any:
        return "not-a-list"

    _patch_entry_points(
        monkeypatch,
        {
            plugins.QC_CHECKS_GROUP: [
                _FakeEP("good_qc", _good_check),
                _FakeEP("boom_qc", _boom),
                _FakeEP("bad_qc", _bad_return),
            ]
        },
    )

    qc_skip = VideoQC(
        QCConfig(on_missing_dependency="skip"),
        VideoConfig(),
        None,
        None,
        None,
        qc_check_plugins=("good_qc", "boom_qc", "bad_qc"),
    )
    res_skip = qc_skip._plugin_checks(frames, 24.0, shot, manifest, tmp_path / "a.wav")
    by_name = {c.name: c for c in res_skip}
    assert by_name["custom_ok"].status is CheckStatus.PASSED
    assert by_name["boom_qc"].status is CheckStatus.SKIPPED
    assert by_name["bad_qc"].status is CheckStatus.SKIPPED

    qc_fail = VideoQC(
        QCConfig(on_missing_dependency="fail"),
        VideoConfig(),
        None,
        None,
        None,
        qc_check_plugins=("boom_qc",),
    )
    res_fail = qc_fail._plugin_checks(frames, 24.0, shot, manifest, tmp_path / "a.wav")
    assert res_fail[0].status is CheckStatus.FAILED and res_fail[0].hard is True


# --------------------------------------------------------------------------- pacing plugin


def test_pacing_plugin_accepts_valid_and_falls_back_on_invalid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    analysis = Analysis(
        duration=12.0,
        tempo_bpm=120.0,
        beats=[],
        bars=[0.0, 2.0, 4.0, 6.0, 8.0, 10.0],
        bar_intensity=[],
        vocal_windows=[],
    )
    video = VideoConfig(min_clip_seconds=1.5, max_clip_seconds=6.0)
    default_cuts = cut_points(analysis, video)

    def _valid(_cuts: list[float], _a: Analysis, _v: VideoConfig) -> list[float]:
        return [0.0, 3.0, 6.0, 9.0, 12.0]

    def _too_short(_cuts: list[float], _a: Analysis, _v: VideoConfig) -> list[float]:
        return [0.0, 0.4, 6.0, 12.0]

    def _boom(_cuts: list[float], _a: Analysis, _v: VideoConfig) -> list[float]:
        raise ValueError("broken")

    _patch_entry_points(
        monkeypatch,
        {
            plugins.PACING_GROUP: [
                _FakeEP("valid_pace", _valid),
                _FakeEP("short_pace", _too_short),
                _FakeEP("boom_pace", _boom),
            ]
        },
    )
    assert cut_points(analysis, video, pacing_plugin="valid_pace") == [0.0, 3.0, 6.0, 9.0, 12.0]
    assert cut_points(analysis, video, pacing_plugin="short_pace") == default_cuts
    assert cut_points(analysis, video, pacing_plugin="boom_pace") == default_cuts


# --------------------------------------------------------------------------- clip offset & timing


def test_clip_timing_applies_optimal_in_offset_only_on_non_lipsync_and_clamps() -> None:
    broll = _make_shot("s001", shot_type=ShotType.BROLL, start=2.0, end=4.0, head=0.25, tail=0.5)
    att = Attempt(
        number=1,
        strategy=RetryStrategy.INITIAL,
        seed=1,
        prompt="p",
        optimal_in_offset=0.9,  # exceeds tail=0.5 -> clamped to 0.5
        sync_delay_seconds=0.1,
    )
    t_broll = assemble.clip_timing(broll, att, None, sync_delay_compensation=True)
    assert t_broll.sync_delay == 0.5
    assert t_broll.in_point == pytest.approx(0.75)

    att_neg = att.model_copy(update={"optimal_in_offset": -0.9})  # clamped to -head=-0.25
    t_neg = assemble.clip_timing(broll, att_neg, None, sync_delay_compensation=True)
    assert t_neg.sync_delay == -0.25
    assert t_neg.in_point == pytest.approx(0.0)

    # Lip-sync shots MUST ignore optimal_in_offset.
    lipsync = _make_shot(
        "s002", shot_type=ShotType.LIPSYNC, start=2.0, end=4.0, head=0.25, tail=0.5
    )
    t_ls = assemble.clip_timing(lipsync, att, None, sync_delay_compensation=False)
    assert t_ls.sync_delay == 0.0
    assert t_ls.in_point == pytest.approx(0.25)


# --------------------------------------------------------------------------- audio conditioner


@needs_ffmpeg
def test_audio_conditioner_isolated_to_lipsync_and_qc_keeps_raw_slice(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    proj = api.open_project(api.init_project(tmp_path / "proj"))
    sr = 16000
    tone = np.zeros(sr * 4, dtype=np.float32)
    sf.write(proj.path(proj.config.inputs.mix), tone, sr)
    sf.write(proj.path(proj.config.inputs.vocals), tone, sr)

    called_for: list[str] = []

    def _cond(raw_wav: Path, out_wav: Path, shot: Shot, _analysis: Analysis) -> Path:
        called_for.append(shot.id)
        out_wav.write_bytes(raw_wav.read_bytes())
        return out_wav

    _cond.__version__ = "1.0"  # type: ignore[attr-defined]
    _patch_entry_points(
        monkeypatch, {plugins.AUDIO_CONDITIONERS_GROUP: [_FakeEP("test_cond", _cond)]}
    )

    cfg = proj.config.model_copy(
        deep=True,
        update={"plugins": PluginsConfig(audio_conditioner="test_cond")},
    )
    proj.config = cfg
    ls_shot = _make_shot("s001", shot_type=ShotType.LIPSYNC, start=0.5, end=2.5)
    broll_shot = _make_shot("s002", shot_type=ShotType.BROLL, start=0.5, end=2.5)
    manifest = _make_manifest([ls_shot, broll_shot])
    proj.save_analysis(
        Analysis(
            duration=4.0, tempo_bpm=120.0, beats=[], bars=[], bar_intensity=[], vocal_windows=[]
        )
    )
    runner = ShotRunner(proj, manifest, None, None, None, {}, None)  # type: ignore[arg-type]

    raw_ls = runner._ensure_audio_slice(ls_shot)
    cond_ls = runner._ensure_conditioning_audio(ls_shot)
    assert cond_ls != raw_ls
    assert cond_ls.name.endswith(".cond.wav")
    assert called_for == ["s001"]
    # Second call hits cache stamp.
    assert runner._ensure_conditioning_audio(ls_shot) == cond_ls
    assert called_for == ["s001"]
    # B-roll never runs the vocal conditioner.
    assert runner._ensure_conditioning_audio(broll_shot) == runner._ensure_audio_slice(broll_shot)
    assert called_for == ["s001"]


# --------------------------------------------------------------------------- encoder probe


def test_encoder_probe_falls_back_when_nvenc_listed_but_test_encode_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_encoders_out = " V..... libx264\n V..... h264_nvenc\n"

    def _fake_run(
        cmd: list[str], *, timeout: float | None = None
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(cmd, 0, stdout=fake_encoders_out, stderr="")

    def _fake_subproc(cmd: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        # Simulate CPU host where h264_nvenc is listed in ffmpeg -encoders but fails to initialize.
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="No capable devices found")

    monkeypatch.setattr(media, "run", _fake_run)
    monkeypatch.setattr(subprocess, "run", _fake_subproc)

    avail = media.available_encoders()
    assert "libx264" in avail
    assert "h264_nvenc" not in avail
    assert media.resolve_video_encoder("auto") == "libx264"
    assert media.resolve_video_encoder("h264_nvenc") == "libx264"


# --------------------------------------------------------------------------- parallel assembly


def _frame_hashes(video_path: Path) -> list[str]:
    frames = media.extract_frames(video_path, 160)
    return [hashlib.sha256(f.tobytes()).hexdigest() for f in frames]


@needs_ffmpeg
def test_parallel_assembly_matches_sequential_frame_hashes(tmp_path: Path) -> None:
    proj = api.open_project(api.init_project(tmp_path / "asm_proj"))
    sr = 16000
    sf.write(proj.path(proj.config.inputs.mix), np.zeros(sr * 6, dtype=np.float32), sr)

    shots: list[Shot] = []
    colors = ("red", "green", "blue", "yellow")
    for idx, color in enumerate(colors):
        shot_id = f"s{idx + 1:03d}"
        clip_rel = f"assets/videos/{shot_id}_v1.mp4"
        clip_path = proj.path(clip_rel)
        clip_path.parent.mkdir(parents=True, exist_ok=True)
        media.run(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                f"color=c={color}:s=128x72:r=24:d=2.0",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(clip_path),
            ]
        )
        start = float(idx)
        end = start + 1.0
        shots.append(
            Shot(
                id=shot_id,
                index=idx,
                start=start,
                end=end,
                type=ShotType.BROLL,
                framing=Framing.WIDE,
                continuity=ContinuityMode.CUT,
                num_frames=49,
                audio_slice=AudioSlice(
                    path=f"assets/audio/{shot_id}.wav",
                    start=start,
                    end=end + 0.5,
                    head_handle=0.0,
                ),
                video=AssetState(
                    status=AssetStatus.PASSED,
                    selected_attempt=1,
                    attempts=[
                        Attempt(
                            number=1,
                            strategy=RetryStrategy.INITIAL,
                            seed=idx + 1,
                            prompt="p",
                            path=clip_rel,
                        )
                    ],
                ),
            )
        )
    manifest = _make_manifest(shots)

    for transition, t_sec in (("cut", 0.0), ("blend", 0.25)):
        proj.config = ProjectConfig(
            video=VideoConfig(
                width=128,
                height=72,
                fps=24,
                transition=transition,  # type: ignore[arg-type]
                transition_seconds=t_sec,
                encoder="libx264",
                assembly_workers=1,
            )
        )
        out_seq = tmp_path / f"seq_{transition}.mp4"
        assemble.assemble(proj, manifest, out_seq, force=True)

        proj.config = proj.config.model_copy(
            deep=True,
            update={"video": proj.config.video.model_copy(update={"assembly_workers": 4})},
        )
        out_par = tmp_path / f"par_{transition}.mp4"
        assemble.assemble(proj, manifest, out_par, force=True)

        seq_hashes = _frame_hashes(out_seq)
        par_hashes = _frame_hashes(out_par)
        assert len(seq_hashes) == len(par_hashes) == 96
        assert seq_hashes == par_hashes


# --------------------------------------------------------------------------- regression & coverage


def test_video_qc_plugin_isolates_frames_and_clamps_scores(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    shot = _make_shot()
    manifest = _make_manifest([shot])
    frames = [np.zeros((32, 32, 3), dtype=np.uint8) for _ in range(4)]
    observed_counts: list[int] = []
    p1_calls = 0

    def _mutator_inflator(
        plugin_frames: list[np.ndarray],
        _fps: float,
        _shot: Shot,
        _manifest: Manifest,
        _audio: Path,
    ) -> list[CheckResult]:
        nonlocal p1_calls
        p1_calls += 1
        observed_counts.append(len(plugin_frames))
        plugin_frames.clear()
        return [
            CheckResult(name="inflator", status=CheckStatus.PASSED, score=100.0, weight=1.0),
            CheckResult(name="negative", status=CheckStatus.PASSED, score=-5.0, weight=-2.0),
        ]

    def _observer(
        plugin_frames: list[np.ndarray],
        _fps: float,
        _shot: Shot,
        _manifest: Manifest,
        _audio: Path,
    ) -> list[CheckResult]:
        observed_counts.append(len(plugin_frames))
        return [CheckResult(name="observer", status=CheckStatus.PASSED, score=0.5, weight=1.0)]

    _patch_entry_points(
        monkeypatch,
        {
            plugins.QC_CHECKS_GROUP: [
                _FakeEP("mutator", _mutator_inflator),
                _FakeEP("observer", _observer),
            ]
        },
    )

    qc = VideoQC(
        QCConfig(on_missing_dependency="fail"),
        VideoConfig(),
        None,
        None,
        None,
        qc_check_plugins=("mutator", "mutator", "observer"),
    )
    plugin_res = qc._plugin_checks(frames, 24.0, shot, manifest, tmp_path / "a.wav")
    assert p1_calls == 1
    assert observed_counts == [4, 4]
    assert len(frames) == 4
    by_name = {c.name: c for c in plugin_res}
    assert by_name["inflator"].score == 1.0
    assert by_name["negative"].score == 0.0
    assert by_name["negative"].weight == 0.0

    failing_soft = CheckResult(name="soft_fail", status=CheckStatus.FAILED, score=0.1, weight=2.0)
    report = aggregate([failing_soft, *plugin_res], {}, pass_score=0.7)
    # Weighted mean = (0.1*2 + 1.0*1 + 0.5*1) / 4 = 0.425 (never inflated above 1.0).
    assert report.score == pytest.approx(0.425)
    assert report.passed is False


def test_available_encoders_concurrent_cold_cache_runs_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_calls = 0

    def _fake_run(
        cmd: list[str], *, timeout: float | None = None
    ) -> subprocess.CompletedProcess[str]:
        nonlocal run_calls
        run_calls += 1
        time.sleep(0.02)
        return subprocess.CompletedProcess(cmd, 0, stdout=" V..... libx264\n", stderr="")

    monkeypatch.setattr(media, "run", _fake_run)
    with ThreadPoolExecutor(max_workers=12) as pool:
        results = list(pool.map(lambda _: media.available_encoders(), range(12)))

    assert run_calls == 1
    assert all(r == frozenset({"libx264"}) for r in results)


@pytest.mark.parametrize("encoder", ["h264_nvenc", "h264_videotoolbox"])
def test_probe_encoder_works_passes_encode_args_for(
    monkeypatch: pytest.MonkeyPatch, encoder: str
) -> None:
    seen_cmds: list[list[str]] = []

    def _fake_subproc(cmd: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        seen_cmds.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", _fake_subproc)
    assert media._probe_encoder_works(encoder) is True
    assert len(seen_cmds) == 1
    expected_flags = media.encode_args_for(encoder)
    cmd = seen_cmds[0]
    for flag in expected_flags:
        assert flag in cmd


def test_continue_handoff_skips_optimal_in_offset_in_runner_and_assembly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    proj = api.open_project(api.init_project(tmp_path / "handoff_proj"))
    calls = 0

    def _selector(*_args: Any) -> float:
        nonlocal calls
        calls += 1
        return 0.25

    _patch_entry_points(monkeypatch, {plugins.CLIP_SELECTORS_GROUP: [_FakeEP("my_sel", _selector)]})
    proj.config = proj.config.model_copy(
        deep=True, update={"plugins": PluginsConfig(clip_selector="my_sel")}
    )

    cont_shot = Shot(
        id="s002",
        index=1,
        start=2.0,
        end=4.0,
        type=ShotType.BROLL,
        framing=Framing.MEDIUM,
        continuity=ContinuityMode.CONTINUE,
        num_frames=57,
        audio_slice=AudioSlice(path="assets/audio/s002.wav", start=2.0, end=4.25, head_handle=0.0),
    )
    manifest = _make_manifest([_make_shot("s001"), cont_shot])
    runner = ShotRunner(proj, manifest, None, None, None, {}, None)  # type: ignore[arg-type]

    assert runner._select_optimal_in_offset(cont_shot, tmp_path / "v.mp4") == 0.0
    assert calls == 0

    handoff_still = Attempt(number=1, strategy=RetryStrategy.HANDOFF, seed=1, prompt="p")
    video_att = Attempt(
        number=1, strategy=RetryStrategy.INITIAL, seed=1, prompt="p", optimal_in_offset=0.25
    )
    timing = assemble.clip_timing(cont_shot, video_att, handoff_still, sync_delay_compensation=True)
    assert timing.in_point == 0.0
    assert timing.sync_delay == 0.0
    assert timing.drop_first_frame is True
    assert timing.first_played_frame(24) == 1

    # Fresh still demotes CONTINUE to CUT, allowing optimal_in_offset to apply normally.
    fresh_still = Attempt(number=1, strategy=RetryStrategy.INITIAL, seed=1, prompt="p")
    timing_cut = assemble.clip_timing(
        cont_shot, video_att, fresh_still, sync_delay_compensation=True
    )
    assert timing_cut.continuity is ContinuityMode.CUT
    assert timing_cut.sync_delay == 0.25
    assert timing_cut.drop_first_frame is False


@needs_ffmpeg
def test_ensure_conditioning_audio_blocks_escape_and_stale_custom_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    proj = api.open_project(api.init_project(tmp_path / "cond_sec"))
    sr = 16000
    tone = np.zeros(sr * 4, dtype=np.float32)
    sf.write(proj.path(proj.config.inputs.mix), tone, sr)
    sf.write(proj.path(proj.config.inputs.vocals), tone, sr)

    outside_file = tmp_path / "outside_secret.wav"
    outside_file.write_bytes(b"SECRET_HOST_DATA")

    def _escape_plugin(_raw: Path, _out: Path, _shot: Shot, _analysis: Any) -> Path:
        return outside_file

    def _symlink_plugin(_raw: Path, out: Path, _shot: Shot, _analysis: Any) -> Path:
        link = out.with_name("escape_link.wav")
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(outside_file)
        return link

    custom_calls = 0

    def _custom_in_project(raw: Path, out: Path, _shot: Shot, _analysis: Any) -> Path:
        nonlocal custom_calls
        custom_calls += 1
        custom = out.with_name("custom_out.wav")
        custom.write_bytes(b"CUSTOM_PROCESSED_" + str(custom_calls).encode())
        return custom

    _patch_entry_points(
        monkeypatch,
        {
            plugins.AUDIO_CONDITIONERS_GROUP: [
                _FakeEP("escape_cond", _escape_plugin),
                _FakeEP("symlink_cond", _symlink_plugin),
                _FakeEP("custom_cond", _custom_in_project),
            ]
        },
    )

    shot = _make_shot("s001", shot_type=ShotType.LIPSYNC, start=0.5, end=2.5)
    manifest = _make_manifest([shot])
    runner = ShotRunner(proj, manifest, None, None, None, {}, None)  # type: ignore[arg-type]
    raw_path = runner._ensure_audio_slice(shot)

    # 1. Direct out-of-project path is rejected and falls back to raw_path.
    proj.config = proj.config.model_copy(
        deep=True, update={"plugins": PluginsConfig(audio_conditioner="escape_cond")}
    )
    assert runner._ensure_conditioning_audio(shot) == raw_path

    # 2. In-project symlink escaping project root is rejected and falls back to raw_path.
    proj.config = proj.config.model_copy(
        deep=True, update={"plugins": PluginsConfig(audio_conditioner="symlink_cond")}
    )
    assert runner._ensure_conditioning_audio(shot) == raw_path

    # 3. Pre-existing stale cond_path must never be returned when plugin returns a custom path.
    stale_cond = raw_path.with_name(f"{raw_path.stem}.cond{raw_path.suffix}")
    stale_cond.write_bytes(b"STALE_GARBAGE")
    proj.config = proj.config.model_copy(
        deep=True, update={"plugins": PluginsConfig(audio_conditioner="custom_cond")}
    )
    first = runner._ensure_conditioning_audio(shot)
    assert first.name == "custom_out.wav"
    assert first.read_bytes() == b"CUSTOM_PROCESSED_1"
    assert not stale_cond.with_name(f"{stale_cond.name}.json").exists()

    second = runner._ensure_conditioning_audio(shot)
    assert second.name == "custom_out.wav"
    assert second.read_bytes() == b"CUSTOM_PROCESSED_2"
    assert second != stale_cond


def test_audio_video_slice_rebuilds_when_wav_path_changes_with_same_size_and_mtime(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    proj = api.open_project(api.init_project(tmp_path / "av_stamp_proj"))
    shot = _make_shot("s001", shot_type=ShotType.LIPSYNC, start=0.0, end=2.0)
    manifest = _make_manifest([shot])
    runner = ShotRunner(proj, manifest, None, None, None, {}, None)  # type: ignore[arg-type]

    raw_path = proj.audio_slice_path(shot.id)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_bytes(b"RIFF_RAW_AUDIO_BYTES")
    raw_stat = raw_path.stat()

    monkeypatch.setattr(runner, "_ensure_audio_slice", lambda _s: raw_path)

    encoded_sources: list[Path] = []

    def _fake_audio_to_black_video(wav_path: Path, mp4_path: Path, *, fps: int) -> None:
        encoded_sources.append(wav_path)
        mp4_path.parent.mkdir(parents=True, exist_ok=True)
        mp4_path.write_bytes(f"{wav_path.name}:{fps}".encode())

    monkeypatch.setattr(media, "audio_to_black_video", _fake_audio_to_black_video)

    def _same_stat_cond(raw_wav: Path, out_wav: Path, _shot: Shot, _analysis: Any) -> Path:
        out_wav.write_bytes(b"RIFF_CND_AUDIO_BYTES")  # exact same byte length as raw_path
        os.utime(out_wav, ns=(raw_stat.st_atime_ns, raw_stat.st_mtime_ns))
        return out_wav

    _patch_entry_points(
        monkeypatch,
        {plugins.AUDIO_CONDITIONERS_GROUP: [_FakeEP("same_stat_cond", _same_stat_cond)]},
    )

    # 1. Raw slice (conditioner disabled) encodes raw_path.
    runner._ensure_audio_video_slice(shot)
    assert [p.name for p in encoded_sources] == ["s001.wav"]

    # 2. Cached hit with unchanged raw_path does not re-encode.
    runner._ensure_audio_video_slice(shot)
    assert [p.name for p in encoded_sources] == ["s001.wav"]

    # 3. Enabling conditioner with identical size & mtime still invalidates and re-encodes.
    proj.config = proj.config.model_copy(
        deep=True, update={"plugins": PluginsConfig(audio_conditioner="same_stat_cond")}
    )
    runner._ensure_audio_video_slice(shot)
    assert [p.name for p in encoded_sources] == ["s001.wav", "s001.cond.wav"]

    # 4. Disabling conditioner again invalidates back to raw_path.
    proj.config = proj.config.model_copy(
        deep=True, update={"plugins": PluginsConfig(audio_conditioner=None)}
    )
    runner._ensure_audio_video_slice(shot)
    assert [p.name for p in encoded_sources] == ["s001.wav", "s001.cond.wav", "s001.wav"]


def test_i2v_fps_validation_and_frame_grid_alignment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with pytest.raises(ValidationError, match="video.i2v_fps"):
        VideoConfig(fps=24, i2v_fps=24)
    with pytest.raises(ValidationError, match="video.i2v_fps"):
        VideoConfig(fps=24, i2v_fps=30)

    proj = api.open_project(api.init_project(tmp_path / "i2v_proj"))
    still_rel = "assets/stills/s001_v1.png"
    still_path = proj.path(still_rel)
    still_path.parent.mkdir(parents=True, exist_ok=True)
    still_path.write_bytes(b"png")

    captured: list[dict[str, Any]] = []

    class _CapturingTemplate:
        bindings = WorkflowBindings(
            workflow="w.json",
            inputs={"image": [Binding(node="1", field="image")]},
            output=OutputSpec(extensions=[".mp4"]),
        )

        def accepts(self, _param: str) -> bool:
            return False

        def render(self, params: dict[str, Any]) -> dict[str, Any]:
            captured.append(dict(params))
            return {}

    def _run_case(shot_type: ShotType, dur: float, multiple: int, offset: int) -> tuple[int, int]:
        captured.clear()
        proj.config = ProjectConfig(
            video=VideoConfig(fps=24, i2v_fps=12, frame_multiple=multiple, frame_offset=offset)
        )
        shot = Shot(
            id="s001",
            index=0,
            start=0.0,
            end=dur,
            type=shot_type,
            framing=Framing.MEDIUM,
            num_frames=89,
            audio_slice=AudioSlice(
                path="assets/audio/s001.wav", start=0.0, end=dur, head_handle=0.0
            ),
            still=AssetState(
                status=AssetStatus.PASSED,
                selected_attempt=1,
                attempts=[
                    Attempt(
                        number=1, strategy=RetryStrategy.INITIAL, seed=1, prompt="p", path=still_rel
                    )
                ],
            ),
        )
        manifest = _make_manifest([shot])
        tpl = _CapturingTemplate()
        runner = ShotRunner(
            proj,
            manifest,
            None,  # type: ignore[arg-type]
            None,  # type: ignore[arg-type]
            None,  # type: ignore[arg-type]
            {"i2v": tpl, "lipsync": tpl},  # type: ignore[dict-item]
            None,  # type: ignore[arg-type]
        )
        monkeypatch.setattr(runner, "_upload", lambda p, client=None: p.name)
        monkeypatch.setattr(runner, "_run_comfy", lambda *_a, **_kw: tmp_path / "out.mp4")
        monkeypatch.setattr(runner, "_ensure_conditioning_audio", lambda _s: still_path)
        att = Attempt(number=1, strategy=RetryStrategy.INITIAL, seed=7, prompt="motion")
        runner._generate_video(shot, att, tmp_path / "out.mp4", [])
        return int(captured[0]["fps"]), int(captured[0]["num_frames"])

    # 8n+1: 3.5s @ 12fps needs 42 frames -> 8*6 + 1 = 49.
    assert _run_case(ShotType.BROLL, 3.5, 8, 1) == (12, 49)
    # 4n+1: 3.5s @ 12fps needs 42 frames -> 4*11 + 1 = 45.
    assert _run_case(ShotType.BROLL, 3.5, 4, 1) == (12, 45)
    # 1n+0: 2.04s @ 12fps = 24.48 -> ceil 25 frames (never under-generates 24).
    assert _run_case(ShotType.BROLL, 2.04, 1, 0) == (12, 25)
    # LIPSYNC ignores i2v_fps and stays at video.fps + shot.num_frames.
    assert _run_case(ShotType.LIPSYNC, 3.5, 8, 1) == (24, 89)


def test_video_frame_bucket_and_sort_for_cache_with_i2v_fps() -> None:
    assert ShotRunner._video_frame_bucket(0) == 16
    assert ShotRunner._video_frame_bucket(1) == 16
    assert ShotRunner._video_frame_bucket(16) == 16
    assert ShotRunner._video_frame_bucket(17) == 32
    assert ShotRunner._video_frame_bucket(49) == 64
    assert ShotRunner._video_frame_bucket(81) == 96

    # Two BROLL shots with different 24fps num_frames (105 vs 137 -> buckets 112 vs 144)
    # but at i2v_fps=12 (8n+1): 4.2s -> 57 (bucket 64), 5.6s -> 73 (bucket 80),
    # while a 3.5s shot -> 49 (bucket 64) should group with the 4.2s shot!
    s_long = Shot(
        id="s001",
        index=0,
        start=0.0,
        end=5.6,
        type=ShotType.BROLL,
        framing=Framing.WIDE,
        num_frames=137,
        setting="alley",
        audio_slice=AudioSlice(path="assets/audio/s001.wav", start=0.0, end=5.6, head_handle=0.0),
    )
    s_mid = Shot(
        id="s002",
        index=1,
        start=5.6,
        end=9.8,
        type=ShotType.BROLL,
        framing=Framing.WIDE,
        num_frames=105,
        setting="rooftop",
        audio_slice=AudioSlice(path="assets/audio/s002.wav", start=5.6, end=9.8, head_handle=0.0),
    )
    s_short = Shot(
        id="s003",
        index=2,
        start=9.8,
        end=13.3,
        type=ShotType.BROLL,
        framing=Framing.WIDE,
        num_frames=89,
        setting="alley",
        audio_slice=AudioSlice(path="assets/audio/s003.wav", start=9.8, end=13.3, head_handle=0.0),
    )
    video = VideoConfig(fps=24, i2v_fps=12, frame_multiple=8, frame_offset=1)
    ordered = ShotRunner._sort_for_cache([s_long, s_mid, s_short], AssetKind.VIDEO, video=video)
    # s_short (49 -> bucket 64, "alley") and s_mid (57 -> bucket 64, "rooftop") precede
    # s_long (73 -> bucket 80).
    assert [s.id for s in ordered] == ["s003", "s002", "s001"]


def test_nag_enabled_wiring_and_switch_pruning(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    proj = api.open_project(api.init_project(tmp_path / "nag_proj"))
    still_rel = "assets/stills/s001_v1.png"
    still_path = proj.path(still_rel)
    still_path.parent.mkdir(parents=True, exist_ok=True)
    still_path.write_bytes(b"png")

    graph = {
        "1": {"class_type": "LoadImage", "inputs": {"image": "placeholder.png"}},
        "2": {"class_type": "BaseModel", "inputs": {}},
        "3": {"class_type": "LTX2_NAG", "inputs": {"model": ["2", 0], "neg": ""}},
        "4": {
            "class_type": "ComfySwitchNode",
            "inputs": {"switch": False, "on_false": ["2", 0], "on_true": ["3", 0]},
        },
        "5": {"class_type": "Sampler", "inputs": {"model": ["4", 0], "image": ["1", 0]}},
    }
    bindings = WorkflowBindings(
        workflow="w.json",
        inputs={
            "image": [Binding(node="1", field="image")],
            "negative_prompt": [Binding(node="3", field="neg")],
            "nag_enabled": [Binding(node="4", field="switch")],
        },
        output=OutputSpec(node="5", extensions=[".mp4"]),
    )
    template = WorkflowTemplate("i2v", bindings, graph)

    rendered_graphs: list[dict[str, Any]] = []

    def _run_nag(nag_cfg: bool, neg_prompt: str) -> dict[str, Any]:
        rendered_graphs.clear()
        proj.config = ProjectConfig(comfy=ComfyConfig(nag_enabled=nag_cfg))
        shot = _make_shot("s001", shot_type=ShotType.BROLL, start=0.0, end=2.0)
        shot.still = AssetState(
            status=AssetStatus.PASSED,
            selected_attempt=1,
            attempts=[
                Attempt(
                    number=1, strategy=RetryStrategy.INITIAL, seed=1, prompt="p", path=still_rel
                )
            ],
        )
        manifest = Manifest(
            title="Test",
            duration=4.0,
            fps=24,
            visual_bible=VisualBible(
                concept="c",
                style="s",
                palette="p",
                lighting="l",
                negative_prompt=neg_prompt,
                exclusions=[],
            ),
            shots=[shot],
        )
        runner = ShotRunner(
            proj,
            manifest,
            None,  # type: ignore[arg-type]
            None,  # type: ignore[arg-type]
            None,  # type: ignore[arg-type]
            {"i2v": template},
            None,  # type: ignore[arg-type]
        )
        monkeypatch.setattr(runner, "_upload", lambda p, client=None: p.name)
        monkeypatch.setattr(
            runner,
            "_run_comfy",
            lambda g, *_a, **_kw: (rendered_graphs.append(g), tmp_path / "o.mp4")[1],
        )
        att = Attempt(number=1, strategy=RetryStrategy.INITIAL, seed=1, prompt="motion")
        runner._generate_video(shot, att, tmp_path / "o.mp4", [])
        return rendered_graphs[0]

    # Enabled + non-empty negative prompt -> NAG node kept and wired into Sampler.
    g_on = _run_nag(True, "blurry, watermark")
    assert "3" in g_on
    assert g_on["5"]["inputs"]["model"] == ["3", 0]

    # Enabled + whitespace-only negative prompt -> NAG node pruned, Sampler gets BaseModel.
    g_blank = _run_nag(True, "   \t\n  ")
    assert "3" not in g_blank
    assert g_blank["5"]["inputs"]["model"] == ["2", 0]

    # Disabled + non-empty negative prompt -> NAG node pruned, Sampler gets BaseModel.
    g_off = _run_nag(False, "blurry, watermark")
    assert "3" not in g_off
    assert g_off["5"]["inputs"]["model"] == ["2", 0]


def test_fit_optimal_transport_color_match_clamping_accuracy_and_edge_cases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(42)
    src = np.clip(rng.normal(0.50, 0.05, size=(64, 64, 3)) * 255.0, 0, 255).astype(np.uint8)
    tgt = np.clip(rng.normal(0.60, 0.20, size=(64, 64, 3)) * 255.0, 0, 255).astype(np.uint8)

    affine = fit_optimal_transport_color_match(
        [src], [tgt], max_matrix_deviation=0.25, max_bias=0.15
    )
    out = affine.apply(src)
    src_mean = src.mean() / 255.0
    tgt_mean = tgt.mean() / 255.0
    out_mean = out.mean() / 255.0
    # Must move decisively brighter toward tgt_mean (never darker than src_mean).
    assert out_mean > src_mean + 0.05
    assert abs(out_mean - tgt_mean) < 0.02

    # Edge cases: empty, too small, constant images, zero regularization.
    assert fit_optimal_transport_color_match([], [tgt]).is_identity()
    assert fit_optimal_transport_color_match([src], []).is_identity()
    tiny = np.zeros((2, 2, 3), dtype=np.uint8)
    assert fit_optimal_transport_color_match([tiny], [tgt]).is_identity()
    black = np.zeros((32, 32, 3), dtype=np.uint8)
    white = np.full((32, 32, 3), 255, dtype=np.uint8)
    bw = fit_optimal_transport_color_match([black], [white], regularization=0.0)
    assert np.all(np.isfinite(bw.matrix)) and np.all(np.isfinite(bw.bias))

    # Invalid negative or non-finite parameters raise ValueError cleanly.
    for kwargs in (
        {"regularization": -1.0},
        {"regularization": float("inf")},
        {"regularization": float("nan")},
        {"max_matrix_deviation": -0.1},
        {"max_matrix_deviation": float("inf")},
        {"max_bias": -0.05},
        {"max_bias": float("nan")},
    ):
        with pytest.raises(ValueError):
            fit_optimal_transport_color_match([src], [tgt], **kwargs)

    # LinAlgError during eigendecomposition degrades safely to identity.
    def _raise_linalg(_a: Any) -> Any:
        raise np.linalg.LinAlgError("Eigenvalues did not converge")

    monkeypatch.setattr(np.linalg, "eigh", _raise_linalg)
    assert fit_optimal_transport_color_match([src], [tgt]).is_identity()
