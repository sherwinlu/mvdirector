# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Tests for 1-click quality fixes: solo vocal lock, clean reference sheets,
close-up face-panel routing, and OpenAI-compatible SSE streaming."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import BaseModel

from mvdirector.config import LLMConfig, PlanningConfig, ProjectConfig
from mvdirector.errors import LLMError
from mvdirector.generation.runner import ShotRunner
from mvdirector.llm.factory import build_llm
from mvdirector.models import (
    BOTH_VOICES,
    Analysis,
    AudioSlice,
    Character,
    Framing,
    Manifest,
    ReferenceSet,
    Shot,
    ShotType,
    VocalTurn,
)
from mvdirector.planning import prompts
from mvdirector.planning.planner import _resolve_shot_singers, _shot_singer_hint
from mvdirector.project import Project


class _DummyReply(BaseModel):
    answer: str


def test_sheet_studio_forbids_panel_text_labels() -> None:
    assert "FRONT / SIDE" not in prompts.SHEET_STUDIO
    assert "No text, no labels" in prompts.SHEET_STUDIO
    for layout in ("row", "grid", "columns"):
        studio = prompts._sheet_studio(layout, "white")
        assert "FRONT / SIDE" not in studio
        assert "FRONTAL / PROFILE" not in studio
        assert "No text, no labels" in studio


def test_shots_system_includes_lipsync_mouth_and_closeup_and_motion_rules() -> None:
    assert "softly closed, or barely parted" in prompts.SHOTS_SYSTEM
    assert "extreme mouth-only or eye-only" in prompts.SHOTS_SYSTEM
    assert "Hold one primary set across each musical section" in prompts.SHOTS_SYSTEM
    assert "Reliable camera motion" in prompts.SHOTS_SYSTEM


def test_solo_vocals_locks_singing_to_primary_protagonist() -> None:
    characters = {
        "witch_lead": Character(name="The Witch", appearance="female lead"),
        "hansel": Character(name="Hansel", appearance="male youth"),
    }
    shot = Shot(
        id="s010",
        index=9,
        start=10.0,
        end=14.0,
        type=ShotType.LIPSYNC,
        section="Chorus",
        audio_slice=AudioSlice(path="assets/audio/s010.wav", start=10.0, end=14.0, head_handle=0.0),
        num_frames=97,
    )
    analysis = Analysis(
        duration=30.0,
        tempo_bpm=124.0,
        beats=[],
        bars=[],
        bar_intensity=[],
        vocal_windows=[],
        vocal_turns=[
            VocalTurn(start=10.0, end=13.5, voice="voice_2", median_f0_hz=140.0),
            VocalTurn(start=13.5, end=14.0, voice=BOTH_VOICES, median_f0_hz=260.0),
        ],
    )
    voice_map = {"voice_1": "witch_lead", "voice_2": "hansel"}

    # In auto/multi mode, dominant voice_2 maps to hansel.
    assert _resolve_shot_singers(shot, analysis, characters, voice_map, solo_vocals=False) == [
        "hansel"
    ]
    # In solo mode, ONLY the primary protagonist ever sings.
    assert _resolve_shot_singers(shot, analysis, characters, voice_map, solo_vocals=True) == [
        "witch_lead"
    ]
    assert (
        _shot_singer_hint(shot, analysis, characters, voice_map, solo_vocals=True) == "witch_lead"
    )


def test_config_defaults_and_options() -> None:
    cfg = ProjectConfig()
    assert cfg.planning.vocals == "auto"
    assert cfg.llm.stream is False
    assert PlanningConfig(vocals="solo").vocals == "solo"
    assert (
        LLMConfig(
            provider="openai_compatible",
            model="qwen",
            base_url="http://127.0.0.1:8000/v1",
            stream=True,
        ).stream
        is True
    )


def test_closeup_routes_face_panel_for_single_ref_workflow(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    (root / "assets" / "reference").mkdir(parents=True)
    front_file = root / "assets" / "reference" / "front.png"
    face_file = root / "assets" / "reference" / "face.png"
    front_file.write_bytes(b"png")
    face_file.write_bytes(b"png")

    wf_dir = root / "workflows"
    wf_dir.mkdir(parents=True)
    (wf_dir / "still.api.json").write_text(
        json.dumps({"1": {"class_type": "LoadImage", "inputs": {"image": "x.png"}}})
    )
    (wf_dir / "still.bindings.json").write_text(
        json.dumps(
            {
                "workflow": "still.api.json",
                "inputs": {"reference_image": [{"node": "1", "field": "image"}]},
                "required": ["reference_image"],
                "output": {"node": "1", "extensions": [".png"]},
            }
        )
    )

    cfg = ProjectConfig()
    cfg.reference.closeup_face_panel = True
    project = Project(root, cfg)
    ref = ReferenceSet(
        panels={"front": "assets/reference/front.png", "face": "assets/reference/face.png"}
    )
    manifest = Manifest(
        title="Test",
        duration=4.0,
        fps=24,
        character=Character(references={"base": ref}),
        shots=[],
    )
    runner = ShotRunner.__new__(ShotRunner)
    runner._project = project
    runner._manifest = manifest
    from mvdirector.comfy.workflow import WorkflowTemplate

    runner._templates = {
        "still": WorkflowTemplate.load(wf_dir / "still.api.json", wf_dir / "still.bindings.json")
    }

    cu_shot = Shot(
        id="s001",
        index=0,
        start=0.0,
        end=2.0,
        type=ShotType.LIPSYNC,
        framing=Framing.CLOSE_UP,
        wardrobe="base",
        audio_slice=AudioSlice(path="assets/audio/s001.wav", start=0.0, end=2.0, head_handle=0.0),
        num_frames=49,
    )
    mc_shot = Shot(
        id="s002",
        index=1,
        start=2.0,
        end=4.0,
        type=ShotType.LIPSYNC,
        framing=Framing.MEDIUM_CLOSE,
        wardrobe="base",
        audio_slice=AudioSlice(path="assets/audio/s002.wav", start=2.0, end=4.0, head_handle=0.0),
        num_frames=49,
    )

    cu_refs = runner._still_reference_paths(cu_shot, None)
    mc_refs = runner._still_reference_paths(mc_shot, None)
    assert cu_refs == [("reference_image", face_file)]
    assert mc_refs == [("reference_image", front_file)]


def test_openai_compatible_sse_streaming_assembles_chunks() -> None:
    seen_payloads: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen_payloads.append(body)
        sse_lines = [
            'data: {"choices":[{"delta":{"content":"{\\"answer\\""},"finish_reason":null}]}',
            'data: {"choices":[{"delta":{"content":": \\"ok\\"}"},"finish_reason":"stop"}]}',
            "data: [DONE]",
            "",
        ]
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=("\n".join(sse_lines)).encode("utf-8"),
        )

    cfg = LLMConfig(
        provider="openai_compatible",
        model="Qwen/Qwen2.5-VL-32B-Instruct",
        base_url="http://127.0.0.1:8000/v1",
        stream=True,
    )
    client = build_llm(cfg, transport=httpx.MockTransport(handler))
    assert client is not None
    result = client.complete_json(
        system="system", content=[{"type": "text", "text": "user"}], schema=_DummyReply
    )
    assert result.answer == "ok"
    assert seen_payloads[0]["stream"] is True


def test_openai_compatible_sse_streaming_handles_refusal() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        sse = (
            'data: {"choices":[{"delta":{"refusal":"policy"},"finish_reason":"stop"}]}\n'
            "data: [DONE]\n"
        )
        return httpx.Response(200, content=sse.encode("utf-8"))

    cfg = LLMConfig(
        provider="openai_compatible",
        model="test-model",
        base_url="http://127.0.0.1:8000/v1",
        stream=True,
    )
    client = build_llm(cfg, transport=httpx.MockTransport(handler))
    assert client is not None
    with pytest.raises(LLMError, match="refused"):
        client.complete_json(
            system="system", content=[{"type": "text", "text": "user"}], schema=_DummyReply
        )


def test_openai_sse_falls_back_to_json_and_surfaces_stream_errors() -> None:
    cfg = LLMConfig(
        provider="openai_compatible",
        model="test-model",
        base_url="http://127.0.0.1:8000/v1",
        stream=True,
    )

    # Server ignores stream=True and returns standard application/json completion.
    def json_handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            json={"choices": [{"message": {"content": '{"answer": "json-ok"}'}}]},
        )

    client = build_llm(cfg, transport=httpx.MockTransport(json_handler))
    assert client is not None
    reply = client.complete_json(
        system="system", content=[{"type": "text", "text": "user"}], schema=_DummyReply
    )
    assert reply.answer == "json-ok"

    # Server sends an in-stream error payload over HTTP 200 SSE.
    def err_handler(_request: httpx.Request) -> httpx.Response:
        sse = 'data: {"error":{"message":"context length exceeded"}}\n\ndata: [DONE]\n\n'
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=sse.encode("utf-8")
        )

    err_client = build_llm(cfg, transport=httpx.MockTransport(err_handler))
    assert err_client is not None
    with pytest.raises(LLMError, match="context length exceeded"):
        err_client.complete_json(
            system="system", content=[{"type": "text", "text": "user"}], schema=_DummyReply
        )


def test_openai_sse_enforces_byte_cap_and_wall_clock_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mvdirector.errors import LLMUnavailableError
    from mvdirector.llm import http as llm_http

    monkeypatch.setattr(llm_http, "_MAX_SSE_BYTES", 256)

    class _HugeStream(httpx.SyncByteStream):
        def __iter__(self):
            yield b'data: {"choices":[{"delta":{"content":"' + b"x" * 512 + b'"}}]}\n\n'

    cfg = LLMConfig(
        provider="openai_compatible",
        model="test-model",
        base_url="http://127.0.0.1:8000/v1",
        stream=True,
        timeout_seconds=10,
    )
    client = build_llm(
        cfg,
        transport=httpx.MockTransport(
            lambda _r: httpx.Response(
                200, headers={"content-type": "text/event-stream"}, stream=_HugeStream()
            )
        ),
    )
    assert client is not None
    with pytest.raises(LLMError, match="byte limit"):
        client.complete_json(
            system="system", content=[{"type": "text", "text": "user"}], schema=_DummyReply
        )

    # Trickle stream exceeding total wall-clock timeout raises LLMUnavailableError.
    monkeypatch.setattr(llm_http, "_MAX_SSE_BYTES", 1024 * 1024)
    times = iter([0.0, 100.0])
    monkeypatch.setattr(llm_http.time, "monotonic", lambda: next(times, 100.0))

    class _TrickleStream(httpx.SyncByteStream):
        def __iter__(self):
            yield b": keep-alive\n\n"
            yield b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'

    slow_client = build_llm(
        cfg,
        transport=httpx.MockTransport(
            lambda _r: httpx.Response(
                200, headers={"content-type": "text/event-stream"}, stream=_TrickleStream()
            )
        ),
    )
    assert slow_client is not None
    with pytest.raises(LLMUnavailableError, match="timed out"):
        slow_client.complete_json(
            system="system", content=[{"type": "text", "text": "user"}], schema=_DummyReply
        )


def test_closeup_face_panel_falls_back_to_front_when_face_file_missing(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    (root / "assets" / "reference").mkdir(parents=True)
    front_file = root / "assets" / "reference" / "front.png"
    front_file.write_bytes(b"front")

    wf_dir = root / "workflows"
    wf_dir.mkdir(parents=True)
    (wf_dir / "still.api.json").write_text(
        json.dumps({"1": {"class_type": "LoadImage", "inputs": {"image": "x.png"}}})
    )
    (wf_dir / "still.bindings.json").write_text(
        json.dumps(
            {
                "workflow": "still.api.json",
                "inputs": {"reference_image": [{"node": "1", "field": "image"}]},
                "required": ["reference_image"],
                "output": {"node": "1", "extensions": [".png"]},
            }
        )
    )

    cfg = ProjectConfig()
    cfg.reference.closeup_face_panel = True
    project = Project(root, cfg)
    ref = ReferenceSet(
        panels={
            "front": "assets/reference/front.png",
            "face": "assets/reference/missing_face.png",
        }
    )
    manifest = Manifest(
        title="Test",
        duration=4.0,
        fps=24,
        character=Character(references={"base": ref}),
        shots=[],
    )
    runner = ShotRunner.__new__(ShotRunner)
    runner._project = project
    runner._manifest = manifest
    from mvdirector.comfy.workflow import WorkflowTemplate

    runner._templates = {
        "still": WorkflowTemplate.load(wf_dir / "still.api.json", wf_dir / "still.bindings.json")
    }
    cu_shot = Shot(
        id="s001",
        index=0,
        start=0.0,
        end=2.0,
        type=ShotType.LIPSYNC,
        framing=Framing.CLOSE_UP,
        wardrobe="base",
        audio_slice=AudioSlice(path="assets/audio/s001.wav", start=0.0, end=2.0, head_handle=0.0),
        num_frames=49,
    )
    assert runner._still_reference_paths(cu_shot, None) == [("reference_image", front_file)]
