# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Shared fixtures: synthetic audio, a ready-made project and a fake ComfyUI server."""

from __future__ import annotations

import copy
import json
import re
import shutil
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest
import soundfile as sf
import yaml
from PIL import Image

from mvdirector.config import ProjectConfig
from mvdirector.errors import ConfigError, ProjectLocked
from mvdirector.models import Analysis, LyricLine, Manifest, TimeSpan
from mvdirector.project import Project

SR = 22050
BPM = 120.0
SONG_SECONDS = 24.0

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def click_track(seconds: float = SONG_SECONDS, bpm: float = BPM) -> np.ndarray:
    """A drum-like track: one decaying noise burst per beat."""
    rng = np.random.default_rng(0)
    samples = np.zeros(int(seconds * SR), dtype=np.float32)
    decay = np.exp(-np.arange(1024) / 150.0)
    for beat in np.arange(0.0, seconds - 0.1, 60.0 / bpm):
        start = int(beat * SR)
        samples[start : start + 1024] += (rng.standard_normal(1024) * decay).astype(np.float32)
    return samples


def vocal_track(spans: list[tuple[float, float]], seconds: float = SONG_SECONDS) -> np.ndarray:
    time = np.arange(int(seconds * SR)) / SR
    samples = np.zeros_like(time, dtype=np.float32)
    for start, end in spans:
        mask = (time >= start) & (time < end)
        samples[mask] = 0.5 * np.sin(2 * np.pi * 220 * time[mask])
    return samples


@pytest.fixture
def analysis() -> Analysis:
    """A hand-built 24 s analysis at 120 BPM: bars every 2 s, singing from 4 s to 16 s."""
    beats = [round(0.5 * i, 3) for i in range(48)]
    bars = beats[::4]
    return Analysis(
        duration=SONG_SECONDS,
        tempo_bpm=BPM,
        beats=beats,
        bars=bars,
        bar_intensity=[0.4] * 4 + [1.0] * 4 + [0.6] * 4,
        vocal_windows=[TimeSpan(start=4.0, end=16.0)],
        lyric_lines=[
            LyricLine(text="first line", start=4.0, end=8.0, timing_source="aligned"),
            LyricLine(text="second line", start=8.0, end=12.0, timing_source="aligned"),
            LyricLine(text="third line", start=12.0, end=16.0, timing_source="aligned"),
        ],
    )


WORKFLOW_GRAPH = {
    "1": {"class_type": "Prompt", "inputs": {"text": "", "negative": ""}},
    "2": {
        "class_type": "Sampler",
        "inputs": {"seed": 0, "steps": 20, "width": 0, "height": 0, "length": 0, "fps": 0},
    },
    "3": {
        "class_type": "Load",
        "inputs": {
            "image": "",
            "audio": "",
            "profile": "",
            "three_quarter": "",
            "face": "",
            "set": "",
        },
    },
    "9": {"class_type": "Save", "inputs": {"prefix": "out"}},
}


def write_workflows(
    folder: Path, names: dict[str, str], still_extra: dict[str, str] | None = None
) -> None:
    """Write a toy workflow and bindings for each logical name -> output extension.

    ``still_extra`` adds optional parameters (name -> field on node 3) to the ``still`` workflow.
    """
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "graph.api.json").write_text(json.dumps(WORKFLOW_GRAPH))
    for name, extension in names.items():
        bindings: dict[str, Any] = {
            "workflow": "graph.api.json",
            "inputs": {
                "positive_prompt": [{"node": "1", "field": "text"}],
                "negative_prompt": [{"node": "1", "field": "negative"}],
                "seed": [{"node": "2", "field": "seed"}],
                "steps": [{"node": "2", "field": "steps"}],
                "width": [{"node": "2", "field": "width"}],
                "height": [{"node": "2", "field": "height"}],
                "num_frames": [{"node": "2", "field": "length"}],
                "fps": [{"node": "2", "field": "fps"}],
                "image": [{"node": "3", "field": "image"}],
                "reference_image": [{"node": "3", "field": "image"}],
                "audio": [{"node": "3", "field": "audio"}],
            },
            "required": ["positive_prompt", "seed"],
            "output": {"node": "9", "extensions": [extension]},
        }
        if name == "still":
            for param, field in (still_extra or {}).items():
                bindings["inputs"][param] = [{"node": "3", "field": field}]
        (folder / f"{name}.bindings.json").write_text(json.dumps(bindings))


def write_upscale_workflow(
    project: Project,
    *,
    extra_inputs: dict[str, list[dict[str, str]]] | None = None,
    length: int = 0,
) -> None:
    """Write graph.api.json and upscale.bindings.json taking a `video` input (not registered).

    ``length`` > 0 is set on node 2 so FakeComfy returns an .mp4 output. Register the workflow
    with ``project.config.comfy.workflows["upscale"] = "upscale.bindings.json"``.
    """
    graph = copy.deepcopy(WORKFLOW_GRAPH)
    graph["3"]["inputs"]["video"] = ""
    if length:
        graph["2"]["inputs"]["length"] = length
    wf_dir = project.path("workflows")
    (wf_dir / "graph.api.json").write_text(json.dumps(graph))
    inputs = {"video": [{"node": "3", "field": "video"}], **(extra_inputs or {})}
    (wf_dir / "upscale.bindings.json").write_text(
        json.dumps(
            {
                "workflow": "graph.api.json",
                "inputs": inputs,
                "required": ["video"],
                "output": {"node": "9", "extensions": [".mp4"]},
            }
        )
    )


@pytest.fixture
def project(tmp_path: Path) -> Project:
    """A project on disk with synthetic audio, toy workflows and the LLM switched off."""
    root = tmp_path / "song"
    (root / "inputs").mkdir(parents=True)
    sf.write(root / "inputs" / "mix.wav", click_track(), SR)
    sf.write(root / "inputs" / "vocals.wav", vocal_track([(4.0, 16.0)]), SR)
    (root / "inputs" / "lyrics.txt").write_text("[Verse]\nfirst line\nsecond line\nthird line\n")
    (root / "inputs" / "style.md").write_text("Synthwave, neon city at night.")
    names = {"t2i": ".png", "still": ".png", "lipsync": ".mp4", "i2v": ".mp4"}
    write_workflows(root / "workflows", names)
    config = ProjectConfig(title="Test Song")
    config.llm.enabled = False
    config.planning.max_lipsync_share = None  # most tests expect every sung shot on camera
    config.comfy.base_url = "http://comfy.test"
    config.comfy.poll_interval = 0.0
    config.comfy.workflows = {name: f"{name}.bindings.json" for name in names}
    (root / "project.yaml").write_text(yaml.safe_dump(config.model_dump(mode="json")))
    return Project.open(root)


@pytest.fixture(scope="session")
def sample_png(tmp_path_factory: pytest.TempPathFactory) -> bytes:
    path = tmp_path_factory.mktemp("media") / "still.png"
    noise = np.random.default_rng(0).integers(0, 255, (720, 1280, 3), dtype=np.uint8)
    Image.fromarray(noise).save(path)
    return path.read_bytes()


@pytest.fixture(scope="session")
def sample_mp4(tmp_path_factory: pytest.TempPathFactory) -> bytes:
    path = tmp_path_factory.mktemp("media") / "clip.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=s=1280x720:r=24:d=9",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
    )
    return path.read_bytes()


class FakeComfy:
    """In-memory stand-in for ComfyUI's HTTP API, usable as an httpx transport."""

    def __init__(self, png: bytes = b"", mp4: bytes = b"") -> None:
        self.png, self.mp4 = png, mp4
        self.queued: list[dict[str, Any]] = []
        self.uploads: list[str] = []
        self.upload_files: list[str] = []  # original filenames, in upload order
        self.free_calls: list[dict[str, Any]] = []
        self.fail_next_jobs = 0

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self._handle)

    def _handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/system_stats":
            return httpx.Response(200, json={"system": {}})
        if path == "/free":
            self.free_calls.append(json.loads(request.content))
            return httpx.Response(200, json={})
        if path == "/upload/image":
            name = f"upload_{len(self.uploads)}"
            self.uploads.append(name)
            match = re.search(rb'filename="([^"]+)"', request.content)
            self.upload_files.append(match.group(1).decode() if match else "")
            return httpx.Response(200, json={"name": name, "subfolder": "mvdirector"})
        if path == "/prompt":
            self.queued.append(json.loads(request.content)["prompt"])
            return httpx.Response(200, json={"prompt_id": f"p{len(self.queued)}"})
        if path.startswith("/history/"):
            prompt_id = path.rsplit("/", 1)[1]
            if self.fail_next_jobs > 0:
                self.fail_next_jobs -= 1
                status = {
                    "status_str": "error",
                    "completed": False,
                    "messages": [
                        [
                            "execution_error",
                            {
                                "node_id": "2",
                                "node_type": "Sampler",
                                "exception_message": "out of memory",
                            },
                        ]
                    ],
                }
                return httpx.Response(200, json={prompt_id: {"status": status, "outputs": {}}})
            graph = self.queued[int(prompt_id[1:]) - 1]
            extension = ".mp4" if graph["2"]["inputs"]["length"] else ".png"
            entry = {
                "status": {"status_str": "success", "completed": True},
                "outputs": {
                    "9": {
                        "images": [
                            {"filename": f"out{extension}", "subfolder": "", "type": "output"}
                        ]
                    }
                },
            }
            return httpx.Response(200, json={prompt_id: entry})
        if path == "/view":
            is_video = request.url.params["filename"].endswith(".mp4")
            return httpx.Response(200, content=self.mp4 if is_video else self.png)
        return httpx.Response(404)


class MemoryState:
    """A database-style StateStore: documents in memory, lock owned by a flag."""

    def __init__(self) -> None:
        self.analysis: Analysis | None = None
        self.manifest: Manifest | None = None
        self.held = False

    def load_analysis(self) -> Analysis:
        if self.analysis is None:
            raise ConfigError("no analysis")
        return self.analysis

    def save_analysis(self, analysis: Analysis) -> None:
        self.analysis = analysis

    def load_manifest(self) -> Manifest:
        if self.manifest is None:
            raise ConfigError("no manifest")
        return self.manifest.model_copy(deep=True)

    def save_manifest(self, manifest: Manifest) -> None:
        self.manifest = manifest.model_copy(deep=True)

    def has_manifest(self) -> bool:
        return self.manifest is not None

    @property
    def project_id(self) -> str:
        return "db-key-42"

    @property
    def is_locked(self) -> bool:
        return self.held

    @contextmanager
    def lock(self) -> Iterator[None]:
        if self.held:
            raise ProjectLocked("held by another job")
        self.held = True
        try:
            yield
        finally:
            self.held = False


class RecordingStorage:
    """An AssetStorage that records pull() and push() calls instead of moving files."""

    def __init__(self) -> None:
        self.events: list[str] = []

    def pull(self) -> None:
        self.events.append("pull")

    def push(self) -> None:
        self.events.append("push")
