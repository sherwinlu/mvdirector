# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""A shot's audio slice follows the shot: it is cut again when the plan or the audio changes."""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from mvdirector.comfy.client import ComfyClient
from mvdirector.errors import ConfigError
from mvdirector.generation import runner as runner_module
from mvdirector.generation.runner import RunBudget, ShotRunner, load_templates
from mvdirector.planning.planner import build_manifest
from mvdirector.qc.factory import QualityGates
from mvdirector.qc.image import ImageQC
from mvdirector.qc.video import VideoQC
from mvdirector.retry import RetryPolicy
from tests.conftest import SR, FakeComfy, needs_ffmpeg, vocal_track

pytestmark = needs_ffmpeg


@pytest.fixture
def runner_and_shot(project, analysis):
    manifest = build_manifest(project.config, analysis, "lyrics", "style", llm=None)
    comfy = ComfyClient(
        project.config.comfy, transport=FakeComfy().transport(), sleep=lambda _: None
    )
    gates = QualityGates(
        image=ImageQC(project.config.qc, project.config.video, face_analyzer=None, vision=None),
        video=VideoQC(
            project.config.qc,
            project.config.video,
            face_analyzer=None,
            mouth_tracker=None,
            vision=None,
        ),
        face_analyzer=None,
    )
    runner = ShotRunner(
        project,
        manifest,
        comfy,
        gates,
        RetryPolicy(project.config.retry, project.config.seed, llm=None),
        load_templates(project, ["t2i", "still", "lipsync", "i2v"]),
        RunBudget(10),
    )
    return runner, manifest.shots[0]


@pytest.fixture
def cuts(monkeypatch):
    """The audio cuts made so far, as (start, end) pairs."""
    made: list[tuple[float, float]] = []
    real = runner_module.media.slice_audio

    def counting(src, dst, start, end):
        made.append((start, end))
        real(src, dst, start, end)

    monkeypatch.setattr(runner_module.media, "slice_audio", counting)
    return made


def _seconds(path) -> float:
    info = sf.info(str(path))
    return info.frames / info.samplerate


def test_slice_is_cut_to_the_shot_and_reused_while_nothing_changes(runner_and_shot, cuts):
    runner, shot = runner_and_shot
    path = runner._ensure_audio_slice(shot)
    assert _seconds(path) == pytest.approx(shot.audio_slice.end - shot.audio_slice.start, abs=0.05)
    assert runner._ensure_audio_slice(shot) == path
    assert len(cuts) == 1


def test_slice_is_cut_again_when_the_plan_moves_the_shot(runner_and_shot, cuts):
    # A re-plan over an existing project keeps the file name (s001.wav) but not the timing.
    runner, shot = runner_and_shot
    path = runner._ensure_audio_slice(shot)
    shot.audio_slice.start += 1.0
    shot.audio_slice.end += 3.0
    runner._ensure_audio_slice(shot)
    assert len(cuts) == 2
    assert cuts[-1] == (shot.audio_slice.start, shot.audio_slice.end)
    assert _seconds(path) == pytest.approx(shot.audio_slice.end - shot.audio_slice.start, abs=0.05)


def test_slice_is_cut_again_when_the_source_audio_is_replaced(runner_and_shot, project, cuts):
    runner, shot = runner_and_shot
    runner._ensure_audio_slice(shot)
    quieter = vocal_track([(4.0, 16.0)]) * np.float32(0.5)
    sf.write(project.path("inputs/vocals.wav"), np.concatenate([quieter, quieter[:SR]]), SR)
    runner._ensure_audio_slice(shot)
    assert len(cuts) == 2


def test_slice_without_a_record_is_cut_again_once(runner_and_shot, cuts):
    # Slices from before the record existed cannot be trusted, so they are replaced.
    runner, shot = runner_and_shot
    path = runner._ensure_audio_slice(shot)
    path.with_name(f"{path.name}.json").unlink()
    runner._ensure_audio_slice(shot)
    runner._ensure_audio_slice(shot)
    assert len(cuts) == 2


def test_existing_slice_is_kept_when_the_source_audio_is_gone(runner_and_shot, project, cuts):
    runner, shot = runner_and_shot
    path = runner._ensure_audio_slice(shot)
    project.path("inputs/vocals.wav").unlink()
    assert runner._ensure_audio_slice(shot) == path
    assert len(cuts) == 1
    path.unlink()
    with pytest.raises(ConfigError, match="not found"):
        runner._ensure_audio_slice(shot)


def test_duet_lipsync_slices_mute_adjacent_and_foreign_singer_bleed(project):
    from mvdirector.models import Analysis, Character, LyricLine, TimeSpan, VocalTurn

    # 8s duet: Elias sings 0.0-4.0s (130 Hz), Odalie sings 4.0-8.0s (280 Hz) with zero gap.
    t = np.arange(int(8.0 * SR), dtype=np.float32) / SR
    duet_vocals = np.where(
        t < 4.0,
        0.5 * np.sin(2.0 * np.pi * 130.0 * t),
        0.5 * np.sin(2.0 * np.pi * 280.0 * t),
    ).astype(np.float32)
    sf.write(project.path("inputs/vocals.wav"), duet_vocals, SR)

    duet_analysis = Analysis(
        duration=8.0,
        tempo_bpm=120.0,
        beats=[round(0.5 * i, 3) for i in range(16)],
        bars=[0.0, 2.0, 4.0, 6.0],
        bar_intensity=[0.6, 0.6, 0.7, 0.7],
        vocal_windows=[TimeSpan(start=0.0, end=8.0)],
        vocal_turns=[
            VocalTurn(start=0.0, end=4.0, voice="voice_1", median_f0_hz=130.0),
            VocalTurn(start=4.0, end=8.0, voice="voice_2", median_f0_hz=280.0),
        ],
        lyric_lines=[
            LyricLine(
                text="Forgive the snow",
                section="Verse 1: Elias",
                start=0.2,
                end=3.8,
                timing_source="aligned",
            ),
            LyricLine(
                text="You were late",
                section="Verse 1: Odalie",
                start=4.1,
                end=7.8,
                timing_source="aligned",
            ),
        ],
    )
    cfg = project.config.model_copy(deep=True)
    cfg.video.max_clip_seconds = 4.0
    cfg.planning.max_lipsync_share = None
    manifest = build_manifest(cfg, duet_analysis, "lyrics", "style", llm=None)
    manifest.characters = {
        "elias": Character(name="Elias", appearance="22-year-old male."),
        "odalie": Character(name="Odalie", appearance="20-year-old female."),
    }
    s1, s2 = manifest.shots[0], manifest.shots[1]
    s1.character, s2.character = "elias", "odalie"

    comfy = ComfyClient(cfg.comfy, transport=FakeComfy().transport(), sleep=lambda _: None)
    gates = QualityGates(
        image=ImageQC(cfg.qc, cfg.video, face_analyzer=None, vision=None),
        video=VideoQC(cfg.qc, cfg.video, None, None, None),
        face_analyzer=None,
    )
    runner = ShotRunner(
        project,
        manifest,
        comfy,
        gates,
        RetryPolicy(cfg.retry, cfg.seed, llm=None),
        load_templates(project, ["t2i", "still", "lipsync", "i2v"]),
        RunBudget(10),
    )

    wav1, sr1 = sf.read(str(runner._ensure_audio_slice(s1)))
    wav2, sr2 = sf.read(str(runner._ensure_audio_slice(s2)))
    # Full slice lengths (including 0.25s handles) are preserved for frame-accurate assembly.
    assert len(wav1) / sr1 == pytest.approx(4.25, abs=0.05)
    assert len(wav2) / sr2 == pytest.approx(4.25, abs=0.05)
    # Elias's tail handle (4.02-4.24s) has Odalie's voice silenced; Elias's body stays loud.
    assert float(np.sqrt(np.mean(wav1[int(1.0 * sr1) : int(3.5 * sr1)] ** 2))) > 0.2
    assert float(np.sqrt(np.mean(wav1[int(4.03 * sr1) :] ** 2))) < 1e-3
    # Odalie's head handle (0.0-0.22s of slice, i.e. 3.75-3.97s) has Elias's voice silenced.
    assert float(np.sqrt(np.mean(wav2[: int(0.22 * sr2)] ** 2))) < 1e-3
    assert float(np.sqrt(np.mean(wav2[int(0.5 * sr2) : int(3.5 * sr2)] ** 2))) > 0.2


def test_mute_audio_spans_ignores_non_finite_spans_and_negative_fade(tmp_path):
    from mvdirector import media

    wav = tmp_path / "tone.wav"
    sr = 22050
    original = np.full(sr, 0.5, dtype=np.float32)
    sf.write(str(wav), original, sr)
    media.mute_audio_spans(
        wav,
        [(float("nan"), 0.5), (0.1, float("inf")), (0.8, 0.2)],
        fade_seconds=-0.5,
    )
    loaded, _ = sf.read(str(wav))
    assert float(np.sqrt(np.mean(loaded**2))) == pytest.approx(0.5, abs=1e-2)
