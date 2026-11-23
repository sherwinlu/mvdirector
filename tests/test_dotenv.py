# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Security and isolation tests for project-scoped ``.env`` helpers."""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from mvdirector import api
from mvdirector.comfy.workflow import WorkflowTemplate
from mvdirector.config import CONFIG_FILENAME, load_config
from mvdirector.dotenv import (
    DOTENV_FILENAME,
    is_allowed_dotenv_var,
    merged_project_environ,
    parse_dotenv,
    read_project_dotenv,
    write_project_dotenv_var,
)
from mvdirector.errors import ConfigError
from mvdirector.llm.factory import build_llm
from mvdirector.project import Project
from tests.conftest import write_workflows


@pytest.mark.parametrize(
    "allowed",
    [
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "COMFY_TOKEN",
        "COMFY_URL",
        "COMFY_SEEDANCE_API_KEY",
        "MVD_LLM_URL",
        "MVD_LLM_KEY",
    ],
)
def test_allowed_dotenv_vars(allowed: str) -> None:
    assert is_allowed_dotenv_var(allowed) is True


@pytest.mark.parametrize(
    "blocked",
    [
        "PATH",
        "PYTHONPATH",
        "LD_PRELOAD",
        "HOME",
        "AWS_SECRET_ACCESS_KEY",
        "GITHUB_TOKEN",
        "DATABASE_URL",
        "MVD_ALLOWED_ENV_VARS",
        "MVD_MCP_TOKEN",
        "MVD_BIND_ADDR",
        "MVD_MEDIAPIPE_CACHE_DIR",
        "MVD_SYNCNET_CACHE_DIR",
        "BAD NAME",
        "",
    ],
)
def test_disallowed_dotenv_vars_are_rejected(blocked: str) -> None:
    assert is_allowed_dotenv_var(blocked) is False


def test_parse_dotenv_filters_disallowed_and_handles_quotes() -> None:
    text = (
        "# comment\n"
        'OPENAI_API_KEY="sk-live-\\"quoted\\""\n'
        "export COMFY_TOKEN='tok-123'\n"
        "MVD_ALLOWED_ENV_VARS=AWS_SECRET_ACCESS_KEY\n"
        "PATH=/tmp/evil\n"
        "AWS_SECRET_ACCESS_KEY=secret\n"
        "EMPTY_LINE_NO_EQUALS\n"
        "MVD_LLM_URL=http://192.168.1.50:11434\n"
    )
    parsed = parse_dotenv(text)
    assert parsed == {
        "OPENAI_API_KEY": 'sk-live-"quoted"',
        "COMFY_TOKEN": "tok-123",
        "MVD_LLM_URL": "http://192.168.1.50:11434",
    }


def test_write_project_dotenv_sets_0600_gitignores_and_updates_in_place(tmp_path: Path) -> None:
    proj = tmp_path / "song"
    proj.mkdir()

    dotenv_path = write_project_dotenv_var(proj, "OPENAI_API_KEY", "sk-first")
    assert dotenv_path == proj / DOTENV_FILENAME
    if os.name == "posix":
        assert stat.S_IMODE(dotenv_path.stat().st_mode) == 0o600
    gitignore = (proj / ".gitignore").read_text(encoding="utf-8")
    assert ".env" in gitignore.splitlines()
    assert ".env.*" in gitignore.splitlines()

    # Adding a second key and updating the first preserves both without duplicating lines.
    write_project_dotenv_var(proj, "COMFY_TOKEN", "comfy-tok")
    write_project_dotenv_var(proj, "OPENAI_API_KEY", "sk-updated")
    parsed = read_project_dotenv(proj)
    assert parsed == {"OPENAI_API_KEY": "sk-updated", "COMFY_TOKEN": "comfy-tok"}
    lines = [
        line
        for line in dotenv_path.read_text(encoding="utf-8").splitlines()
        if "OPENAI_API_KEY=" in line
    ]
    assert len(lines) == 1


def test_write_and_read_dotenv_reject_symlinks_oversized_and_injection(tmp_path: Path) -> None:
    proj = tmp_path / "sec"
    proj.mkdir()

    with pytest.raises(ConfigError):
        write_project_dotenv_var(proj, "MVD_ALLOWED_ENV_VARS", "AWS_SECRET_ACCESS_KEY")
    with pytest.raises(ConfigError):
        write_project_dotenv_var(proj, "OPENAI_API_KEY", "sk-bad\nPATH=/evil")

    outside = tmp_path / "outside.txt"
    outside.write_text("OPENAI_API_KEY=leak\n", encoding="utf-8")
    (proj / DOTENV_FILENAME).symlink_to(outside)
    assert read_project_dotenv(proj) == {}
    with pytest.raises(ConfigError, match="symlink"):
        write_project_dotenv_var(proj, "OPENAI_API_KEY", "sk-new")

    (proj / DOTENV_FILENAME).unlink()
    (proj / DOTENV_FILENAME).write_text("OPENAI_API_KEY=" + ("a" * 70_000), encoding="utf-8")
    assert read_project_dotenv(proj) == {}


def test_project_dotenv_is_isolated_and_shell_always_wins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("COMFY_TOKEN", raising=False)

    proj_a = tmp_path / "proj-a"
    proj_b = tmp_path / "proj-b"
    api.init_project(proj_a)
    api.init_project(proj_b)

    for folder, model in ((proj_a, "gpt-4o"), (proj_b, "gpt-4o-mini")):
        cfg_path = folder / CONFIG_FILENAME
        cfg_path.write_text(
            "title: Test\n"
            "comfy:\n"
            "  base_url: http://127.0.0.1:8188\n"
            '  headers: {Authorization: "Bearer ${COMFY_TOKEN}"}\n'
            "llm:\n"
            "  provider: openai\n"
            f"  model: {model}\n",
            encoding="utf-8",
        )

    write_project_dotenv_var(proj_a, "OPENAI_API_KEY", "sk-project-a")
    write_project_dotenv_var(proj_a, "COMFY_TOKEN", "tok-project-a")
    write_project_dotenv_var(proj_b, "OPENAI_API_KEY", "sk-project-b")
    write_project_dotenv_var(proj_b, "COMFY_TOKEN", "tok-project-b")

    # Opening both projects resolves each project's own secrets without mutating os.environ.
    opened_a = Project.open(proj_a)
    opened_b = Project.open(proj_b)
    assert "OPENAI_API_KEY" not in os.environ
    assert "COMFY_TOKEN" not in os.environ
    assert opened_a.config.comfy.headers == {"Authorization": "Bearer tok-project-a"}
    assert opened_b.config.comfy.headers == {"Authorization": "Bearer tok-project-b"}
    assert opened_a.environ()["OPENAI_API_KEY"] == "sk-project-a"
    assert opened_b.environ()["OPENAI_API_KEY"] == "sk-project-b"

    # build_llm and WorkflowTemplate.load read project-scoped .env cleanly.
    client_a = build_llm(opened_a.config.llm, environ=opened_a.environ())
    assert client_a is not None
    client_a.close()

    import json

    write_workflows(proj_a / "workflows", {"still": ".png"})
    bpath = proj_a / "workflows" / "still.bindings.json"
    bdata = json.loads(bpath.read_text(encoding="utf-8"))
    bdata["defaults"] = {"api_key": "${COMFY_TOKEN}"}
    bpath.write_text(json.dumps(bdata), encoding="utf-8")
    tpl = WorkflowTemplate.load("still", bpath, root=proj_a, environ=opened_a.environ())
    assert tpl.bindings.defaults["api_key"] == "tok-project-a"

    # Explicit shell env always overrides .env.
    monkeypatch.setenv("COMFY_TOKEN", "tok-from-shell")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-shell")
    assert merged_project_environ(proj_a)["OPENAI_API_KEY"] == "sk-from-shell"
    assert load_config(proj_a / CONFIG_FILENAME).comfy.headers == {
        "Authorization": "Bearer tok-from-shell"
    }
