# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import json

import httpx
import pytest

from mvdirector.comfy.client import ComfyClient
from mvdirector.comfy.workflow import OutputSpec, WorkflowBindings, WorkflowTemplate
from mvdirector.config import ComfyConfig
from mvdirector.errors import ComfyConnectionError, ComfyJobError, WorkflowBindingError
from tests.conftest import WORKFLOW_GRAPH, FakeComfy, write_workflows


@pytest.fixture
def template(tmp_path):
    write_workflows(tmp_path, {"still": ".png"})
    return WorkflowTemplate.load("still", tmp_path / "still.bindings.json")


def test_render_writes_params_without_mutating_template(template):
    graph = template.render({"positive_prompt": "a singer", "seed": 7, "steps": 30})
    assert graph["1"]["inputs"]["text"] == "a singer"
    assert graph["2"]["inputs"]["seed"] == 7 and graph["2"]["inputs"]["steps"] == 30
    assert template.render({"positive_prompt": "x", "seed": 1})["2"]["inputs"]["steps"] == 20


def test_render_ignores_unbound_and_none_params(template):
    graph = template.render({"positive_prompt": "x", "seed": 1, "unknown": 5, "audio": None})
    assert graph["3"]["inputs"]["audio"] == ""


def test_render_rejects_missing_required_param(template):
    with pytest.raises(WorkflowBindingError, match="seed"):
        template.render({"positive_prompt": "x"})


def test_load_reports_bindings_that_do_not_match_graph(tmp_path):
    write_workflows(tmp_path, {"still": ".png"})
    path = tmp_path / "still.bindings.json"
    bindings = json.loads(path.read_text())
    bindings["inputs"]["seed"] = [{"node": "99", "field": "seed"}]
    bindings["inputs"]["steps"] = [{"node": "2", "field": "nope"}]
    path.write_text(json.dumps(bindings))
    with pytest.raises(WorkflowBindingError) as error:
        WorkflowTemplate.load("still", path)
    assert "node 99" in str(error.value) and "no input 'nope'" in str(error.value)


def test_load_detects_ui_format_export(tmp_path):
    write_workflows(tmp_path, {"still": ".png"})
    (tmp_path / "graph.api.json").write_text(json.dumps({"nodes": [], "links": []}))
    with pytest.raises(WorkflowBindingError, match="API Format"):
        WorkflowTemplate.load("still", tmp_path / "still.bindings.json")


def _client(fake: FakeComfy) -> ComfyClient:
    config = ComfyConfig(base_url="http://comfy.test", poll_interval=0.0)
    return ComfyClient(config, transport=fake.transport(), sleep=lambda _: None)


def test_run_queues_waits_and_downloads(tmp_path):
    fake = FakeComfy(png=b"PNGDATA")
    destination = tmp_path / "out" / "still.png"
    with _client(fake) as comfy:
        assert comfy.upload(_touch(tmp_path / "ref.png")) == "mvdirector/upload_0"
        result = comfy.run(WORKFLOW_GRAPH, OutputSpec(node="9"), destination)
    assert result.read_bytes() == b"PNGDATA"
    assert fake.queued == [WORKFLOW_GRAPH]


def test_failed_job_raises_job_error_with_node_detail(tmp_path):
    fake = FakeComfy()
    fake.fail_next_jobs = 1
    with _client(fake) as comfy, pytest.raises(ComfyJobError, match="out of memory"):
        comfy.run(WORKFLOW_GRAPH, OutputSpec(node="9"), tmp_path / "x.png")


def test_missing_output_raises_job_error(tmp_path):
    with _client(FakeComfy()) as comfy, pytest.raises(ComfyJobError, match="no .mp4 output"):
        comfy.run(WORKFLOW_GRAPH, OutputSpec(node="9", extensions=[".mp4"]), tmp_path / "x.mp4")


def test_rejected_workflow_reports_node_errors():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={
                "error": {"message": "Prompt outputs failed validation"},
                "node_errors": {
                    "3": {
                        "class_type": "Load",
                        "errors": [{"message": "Value not in list", "details": "image: 'x.png'"}],
                    }
                },
            },
        )

    config = ComfyConfig(base_url="http://comfy.test")
    with (
        ComfyClient(config, transport=httpx.MockTransport(handler)) as comfy,
        pytest.raises(ComfyJobError, match="Value not in list"),
    ):
        comfy.queue(WORKFLOW_GRAPH)


def test_unreachable_server_raises_connection_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    config = ComfyConfig(base_url="http://comfy.test")
    comfy = ComfyClient(config, transport=httpx.MockTransport(handler), sleep=lambda _: None)
    with pytest.raises(ComfyConnectionError, match="COMFY_URL"):
        comfy.ping()


def _touch(path):
    path.write_bytes(b"x")
    return path


@pytest.mark.parametrize(
    ("filename", "subfolder"),
    [
        ("../secret.png", ""),
        ("sub/file.png", ""),
        ("sub\\file.png", ""),
        ("out.png", "../secret"),
        ("out.png", "/abs/sub"),
    ],
)
def test_download_output_rejects_unsafe_server_paths(tmp_path, filename, subfolder):
    entry = {
        "outputs": {
            "9": {"images": [{"filename": filename, "subfolder": subfolder, "type": "output"}]}
        }
    }
    with (
        _client(FakeComfy()) as comfy,
        pytest.raises(ComfyJobError, match="unsafe output path"),
    ):
        comfy.download_output(entry, OutputSpec(node="9"), tmp_path / "out.png")


def test_free_memory_posts_to_free_endpoint():
    fake = FakeComfy()
    with _client(fake) as comfy:
        comfy.free_memory()
    assert fake.free_calls == [{"unload_models": True, "free_memory": True}]


def test_bindings_defaults_expand_allowed_env_vars(tmp_path, monkeypatch):
    monkeypatch.setenv("COMFY_SEEDANCE_API_KEY", "sk-seedance-123")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-123")
    write_workflows(tmp_path, {"still": ".png"})
    path = tmp_path / "still.bindings.json"
    bindings = json.loads(path.read_text())
    bindings["inputs"]["api_key"] = [{"node": "1", "field": "negative"}]
    bindings["defaults"] = {"api_key": "${COMFY_SEEDANCE_API_KEY}"}
    path.write_text(json.dumps(bindings))
    loaded = WorkflowTemplate.load("still", path, root=tmp_path)
    graph = loaded.render({"positive_prompt": "hi", "seed": 1})
    assert graph["1"]["inputs"]["negative"] == "sk-seedance-123"

    bindings["defaults"] = {"api_key": "${ANTHROPIC_API_KEY}"}
    path.write_text(json.dumps(bindings))
    with pytest.raises(WorkflowBindingError, match="not allowed"):
        WorkflowTemplate.load("still", path, root=tmp_path)


def test_download_output_streams_and_enforces_size_limit(tmp_path, monkeypatch):
    from mvdirector.comfy import client as client_mod

    monkeypatch.setattr(client_mod, "_MAX_DOWNLOAD_BYTES", 1024)
    entry = {
        "outputs": {"9": {"images": [{"filename": "big.mp4", "subfolder": "", "type": "output"}]}}
    }
    payload = b"A" * 2048

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=payload)

    config = ComfyConfig(base_url="http://comfy.test")
    destination = tmp_path / "out" / "big.mp4"
    with (
        ComfyClient(config, transport=httpx.MockTransport(handler), sleep=lambda _: None) as comfy,
        pytest.raises(ComfyJobError, match="download limit"),
    ):
        comfy.download_output(entry, OutputSpec(node="9", extensions=[".mp4"]), destination)
    assert not destination.exists()
    assert not list(destination.parent.glob("*.part"))


def test_queue_does_not_retry_on_read_timeout():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("timed out after queuing")

    config = ComfyConfig(base_url="http://comfy.test")
    with (
        ComfyClient(config, transport=httpx.MockTransport(handler), sleep=lambda _: None) as comfy,
        pytest.raises(ComfyConnectionError, match="interrupted"),
    ):
        comfy.queue(WORKFLOW_GRAPH)
    assert calls == 1


def test_upload_uses_project_specific_subfolder(tmp_path):
    import re

    from mvdirector.config import ProjectConfig
    from mvdirector.project import Project

    seen_subfolders: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        match = re.search(rb'name="subfolder"\r\n\r\n([^\r\n]+)', request.content)
        sub = match.group(1).decode() if match else ""
        seen_subfolders.append(sub)
        return httpx.Response(200, json={"name": "ref.png", "subfolder": sub})

    p1 = Project(tmp_path / "song-a", ProjectConfig())
    p2 = Project(tmp_path / "song-b", ProjectConfig())
    assert p1.comfy_subfolder != p2.comfy_subfolder
    ref = _touch(tmp_path / "ref.png")
    config = ComfyConfig(base_url="http://comfy.test")
    with ComfyClient(
        config, transport=httpx.MockTransport(handler), upload_subfolder=p1.comfy_subfolder
    ) as c1:
        returned = c1.upload(ref)
    assert seen_subfolders == [p1.comfy_subfolder]
    assert returned == f"{p1.comfy_subfolder}/ref.png"


def _multi_image_template(**binding_overrides):
    graph = {
        "10": {"class_type": "LoadImage", "inputs": {"image": "portrait.png"}},
        "11": {"class_type": "LoadImage", "inputs": {"image": "plate.png"}},
        "12": {"class_type": "LoadImage", "inputs": {"image": "prop.png"}},
        "20": {
            "class_type": "TextEncodeQwenImage21",
            "inputs": {
                "prompt": "",
                "images.image_1": ["10", 0],
                "images.image_2": ["11", 0],
                "images.image_3": ["12", 0],
            },
        },
    }
    bindings = WorkflowBindings(
        workflow="x.api.json",
        inputs={
            "positive_prompt": [{"node": "20", "field": "prompt"}],
            "reference_image": [{"node": "10", "field": "image"}],
            "set_image": [{"node": "11", "field": "image"}],
            "prop_image": [{"node": "12", "field": "image"}],
        },
        required=["positive_prompt", "reference_image"],
        **binding_overrides,
    )
    return WorkflowTemplate("still", bindings, graph)


def test_render_keeps_optional_reference_images_that_are_supplied():
    graph = _multi_image_template().render(
        {
            "positive_prompt": "x",
            "reference_image": "a.png",
            "set_image": "set.png",
            "prop_image": "box.png",
        }
    )
    encoder = graph["20"]["inputs"]
    assert graph["11"]["inputs"]["image"] == "set.png"
    assert graph["12"]["inputs"]["image"] == "box.png"
    assert encoder["images.image_2"] == ["11", 0] and encoder["images.image_3"] == ["12", 0]


def test_render_cuts_out_optional_reference_images_a_shot_does_not_have():
    template = _multi_image_template()
    graph = template.render(
        {"positive_prompt": "x", "reference_image": "a.png", "set_image": "set.png"}
    )
    assert "12" not in graph
    assert "images.image_3" not in graph["20"]["inputs"]
    assert graph["11"]["inputs"]["image"] == "set.png" and "images.image_2" in graph["20"]["inputs"]
    # The template itself is not changed for the next shot.
    assert "12" in template.render(
        {"positive_prompt": "x", "reference_image": "a.png", "prop_image": "box.png"}
    )


def test_render_never_cuts_out_the_required_reference_or_a_defaulted_image():
    template = _multi_image_template(defaults={"set_image": "blank.png"})
    graph = template.render({"positive_prompt": "x", "reference_image": "a.png"})
    assert "10" in graph and "images.image_1" in graph["20"]["inputs"]
    assert graph["11"]["inputs"]["image"] == "blank.png"
    assert "12" not in graph
