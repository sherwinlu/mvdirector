# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from __future__ import annotations

from pathlib import Path

import httpx

from mvdirector.comfy.client import ComfyClient, ComfyProgress, parse_comfy_ws_event
from mvdirector.comfy.workflow import OutputSpec
from mvdirector.config import ComfyConfig


def test_parse_comfy_ws_progress_event() -> None:
    ev = parse_comfy_ws_event(
        {
            "type": "progress",
            "data": {"prompt_id": "p-1", "value": 7, "max": 20, "node": "12"},
        },
        "p-1",
        elapsed_seconds=4.26,
    )
    assert ev == ComfyProgress(
        prompt_id="p-1",
        state="running",
        step=7,
        total_steps=20,
        percent=35.0,
        node="12",
        elapsed_seconds=4.3,
    )


def test_parse_comfy_ws_ignores_other_prompt() -> None:
    ev = parse_comfy_ws_event(
        {
            "type": "progress",
            "data": {"prompt_id": "other-job", "value": 5, "max": 10},
        },
        "p-1",
    )
    assert ev is None


def test_parse_comfy_ws_progress_state_aggregates_nodes() -> None:
    ev = parse_comfy_ws_event(
        {
            "type": "progress_state",
            "data": {
                "prompt_id": "p-2",
                "nodes": {
                    "3": {"value": 10, "max": 10, "state": "finished"},
                    "8": {"value": 5, "max": 10, "state": "running"},
                },
            },
        },
        "p-2",
        elapsed_seconds=12.0,
    )
    assert ev is not None
    assert ev.step == 15
    assert ev.total_steps == 20
    assert ev.percent == 75.0
    assert ev.node == "8"


def test_parse_comfy_ws_executing_none_marks_downloading() -> None:
    ev = parse_comfy_ws_event(
        {"type": "executing", "data": {"prompt_id": "p-1", "node": None}},
        "p-1",
        elapsed_seconds=9.1,
    )
    assert ev is not None
    assert ev.state == "downloading"
    assert ev.percent == 99.0


def test_comfy_client_emits_queue_and_completion_progress(tmp_path: Path) -> None:
    polls = {"history": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/prompt":
            return httpx.Response(200, json={"prompt_id": "job-42"})
        if request.url.path == "/history/job-42":
            polls["history"] += 1
            if polls["history"] < 3:
                return httpx.Response(200, json={})
            return httpx.Response(
                200,
                json={
                    "job-42": {
                        "status": {"completed": True, "status_str": "success"},
                        "outputs": {
                            "9": {
                                "images": [
                                    {"filename": "out.png", "subfolder": "", "type": "output"}
                                ]
                            }
                        },
                    }
                },
            )
        if request.url.path == "/queue":
            if polls["history"] == 1:
                return httpx.Response(
                    200,
                    json={"queue_running": [], "queue_pending": [[1, "other"], [2, "job-42"]]},
                )
            return httpx.Response(
                200,
                json={"queue_running": [[2, "job-42"]], "queue_pending": []},
            )
        if request.url.path == "/view":
            return httpx.Response(200, content=b"fake-png-bytes")
        return httpx.Response(404)

    seen: list[ComfyProgress] = []
    cfg = ComfyConfig(base_url="http://127.0.0.1:8188", poll_interval=0.0, job_timeout=5.0)
    with ComfyClient(
        cfg,
        transport=httpx.MockTransport(handler),
        sleep=lambda _: None,
        on_progress=seen.append,
    ) as client:
        out = client.run({"9": {}}, OutputSpec(node="9"), tmp_path / "shot.png")

    assert out.read_bytes() == b"fake-png-bytes"
    states = [p.state for p in seen]
    assert states == ["queued", "queued", "running", "downloading", "done"]
    assert seen[1].queue_position == 2
    assert seen[-1].percent == 100.0
