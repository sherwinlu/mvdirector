# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Tests for plugin hooks, clip in-point offsets, encoder probing, and parallel assembly."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from mvdirector import api, assemble, media, plugins
from mvdirector.config import PluginsConfig, ProjectConfig, QCConfig, VideoConfig
from mvdirector.errors import ConfigError
from mvdirector.generation.runner import ShotRunner
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
    Manifest,
    RetryStrategy,
    Shot,
    ShotType,
    VisualBible,
)
from mvdirector.planning.segmenter import cut_points
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
    with pytest.raises(ConfigError):
        PluginsConfig(pacing=bad)
    with pytest.raises(ConfigError):
        PluginsConfig(qc_checks=[bad])


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
