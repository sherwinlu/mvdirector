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
