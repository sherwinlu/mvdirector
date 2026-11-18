# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Tests for the pre-production Suno lyrics & style brief generator (mvd brief)."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from mvdirector import api as api_mod
from mvdirector import cli as cli_mod
from mvdirector.audio.lyrics import parse_lyrics
from mvdirector.errors import ConfigError
from mvdirector.planning.brief import BriefReply, generate_brief, write_brief


class _FakeBriefLLM:
    def __init__(self, reply: BriefReply | None = None) -> None:
        self.seen_request = ""
        self.reply = reply or BriefReply(
            lyrics="[Intro]\nReady? Go\n\n[Chorus]\nPlayer one\nHigh score",
            suno_style="K-pop mainstream pop, 118 BPM, dark lower-register female vocal",
            visual_direction="Retro arcade transitioning into neon Tron grid world.",
        )

    def complete_json(self, *, system, content, schema):
        assert schema is BriefReply
        self.seen_request = content[0]["text"]
        return self.reply


def test_generate_brief_validates_inputs_and_passes_options() -> None:
    with pytest.raises(ConfigError, match="non-empty --idea"):
        generate_brief("   ", llm=_FakeBriefLLM())
    with pytest.raises(ConfigError, match="llm.enabled: true"):
        generate_brief("Arcade cyber duel", llm=None)
    with pytest.raises(ConfigError, match="between 5 and 600"):
        generate_brief("Arcade cyber duel", llm=_FakeBriefLLM(), duration_seconds=0)
    with pytest.raises(ConfigError, match="empty lyrics"):
        generate_brief(
            "Arcade cyber duel",
            llm=_FakeBriefLLM(BriefReply(lyrics="  ", suno_style="pop", visual_direction="neon")),
        )

    llm = _FakeBriefLLM()
    reply = generate_brief(
        "Girl enters a neon arcade game",
        llm=llm,
        duration_seconds=40,
        vibe="K-pop + Tron",
    )
    assert "~40 seconds" in llm.seen_request and "K-pop + Tron" in llm.seen_request
    assert [ln.section for ln in parse_lyrics(reply.lyrics)] == [
        "Intro",
        "Chorus",
        "Chorus",
    ]


def test_write_brief_guards_existing_files_and_allows_starter_replacement(
    project, tmp_path
) -> None:
    # `project` fixture writes custom lyrics/style; without overwrite=True write_brief must refuse.
    brief = _FakeBriefLLM().reply
    with pytest.raises(ConfigError, match="--overwrite"):
        write_brief(project, brief, overwrite=False)

    lyrics_path, style_path = write_brief(project, brief, overwrite=True)
    assert "Player one" in lyrics_path.read_text(encoding="utf-8")
    assert "Music style: K-pop mainstream pop" in style_path.read_text(encoding="utf-8")
    assert "Visual direction: Retro arcade" in style_path.read_text(encoding="utf-8")

    # On a fresh `mvd init` project with starter placeholders, overwrite is not needed.
    fresh_dir = tmp_path / "fresh"
    res_init = CliRunner().invoke(cli_mod.app, ["init", str(fresh_dir)])
    assert res_init.exit_code == 0
    from mvdirector.project import Project

    fresh_proj = Project.open(fresh_dir)
    fresh_lyrics, fresh_style = write_brief(fresh_proj, brief, overwrite=False)
    assert "Player one" in fresh_lyrics.read_text(encoding="utf-8")
    assert "Music style:" in fresh_style.read_text(encoding="utf-8")


def test_brief_cli_populates_inputs_and_prints_suno_style(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fresh_dir = tmp_path / "song"
    runner = CliRunner()
    assert runner.invoke(cli_mod.app, ["init", str(fresh_dir)]).exit_code == 0

    fake_llm = _FakeBriefLLM()
    monkeypatch.setattr(api_mod, "build_llm", lambda _cfg: fake_llm)
    result = runner.invoke(
        cli_mod.app,
        [
            "brief",
            "--idea",
            "A girl battles masked clones in a neon arcade",
            "--duration",
            "40",
            "--vibe",
            "K-pop sci-fi",
            "-p",
            str(fresh_dir),
        ],
    )
    assert result.exit_code == 0
    assert "Suno style prompt:" in result.output
    assert "K-pop mainstream pop, 118 BPM" in result.output
    assert "Player one" in (fresh_dir / "inputs" / "lyrics.txt").read_text(encoding="utf-8")

    # Overwrite check happens before invoking the LLM so non-starter files don't burn tokens.
    guarded_llm = _FakeBriefLLM()
    monkeypatch.setattr(api_mod, "build_llm", lambda _cfg: guarded_llm)
    blocked = runner.invoke(
        cli_mod.app,
        ["brief", "--idea", "Another concept", "-p", str(fresh_dir)],
    )
    assert blocked.exit_code == 1
    assert "--overwrite" in blocked.output
    assert guarded_llm.seen_request == ""
