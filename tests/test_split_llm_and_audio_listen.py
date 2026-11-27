# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Tests for split Vision QC LLM (`qc.llm`) and single-call native audio listening (`audio_llm`)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from mvdirector import api
from mvdirector.config import LLMConfig, ProjectConfig, load_config
from mvdirector.errors import ConfigError
from mvdirector.llm.base import ContentBlock
from mvdirector.models import Analysis, LyricLine, TimeSpan
from mvdirector.planning.planner import (
    AudioListeningNotes,
    BibleReply,
    ShotBatchReply,
    build_manifest,
)


class _FakeTextPlanner:
    """Text-only planner (like Claude Opus 5.5) that records every prompt it receives."""

    supports_audio = False

    def __init__(self) -> None:
        self.prompts: list[tuple[str, str]] = []

    def complete_json(self, *, system: str, content: list[ContentBlock], schema: type[Any]) -> Any:
        joined = "\n".join(b.get("text", "") for b in content if b.get("type") == "text")
        self.prompts.append((schema.__name__, joined))
        if schema is BibleReply:
            return BibleReply(
                concept="A gothic candlelit chapel in 3/4 waltz time.",
                style="35mm moody film",
                palette="candle amber and cold slate",
                lighting="practical candles",
                negative_prompt="cgi, watermark",
                appearance="Pale woman with dark wavy hair.",
                settings=[{"name": "Chapel", "sections": ["Verse 1"], "description": "Old chapel"}],
                rules=["Keep camera steady."],
            )
        assert schema is ShotBatchReply
        return ShotBatchReply(shots=[])


class _FakeAudioListener:
    """Audio-capable helper (like Gemini Flash) that asserts the MP3 block is attached."""

    supports_audio = True

    def __init__(self) -> None:
        self.calls: list[list[ContentBlock]] = []

    def complete_json(self, *, system: str, content: list[ContentBlock], schema: type[Any]) -> Any:
        assert schema is AudioListeningNotes
        self.calls.append(content)
        return AudioListeningNotes(
            meter_and_groove="3/4 waltz sway with gentle rubato",
            instrumentation_and_texture="antique music box, bowed cello, reverberant wooden room",
            vocal_character="haunted baritone and fragile soprano duet",
            atmosphere_and_subtext="uncanny gothic grief beneath tender nostalgia",
            key_musical_turns=["Bridge drops to solo music box before final harmonic swell"],
        )


def test_split_qc_llm_and_audio_listen_config_validation(tmp_path: Path):
    yaml_text = yaml.safe_dump(
        {
            "title": "Split Config Song",
            "llm": {"provider": "anthropic", "model": "claude-opus-5-5"},
            "planning": {"audio_listen": "auto"},
            "qc": {
                "vision_enabled": True,
                "llm": {"provider": "gemini", "model": "gemini-3.8-flash"},
            },
        }
    )
    cfg_path = tmp_path / "project.yaml"
    cfg_path.write_text(yaml_text, encoding="utf-8")
    cfg = load_config(cfg_path)
    assert cfg.llm.provider == "anthropic"
    assert cfg.qc.llm is not None and cfg.qc.llm.model == "gemini-3.8-flash"
    # Auto audio_listen automatically resolves to qc.llm when qc.llm is Gemini!
    resolved_audio = api._resolve_audio_llm_config(cfg)
    assert resolved_audio is not None and resolved_audio.model == "gemini-3.8-flash"

    # Explicit planning.audio_llm takes precedence over qc.llm
    cfg.planning.audio_llm = LLMConfig(provider="gemini", model="gemini-4-neon")
    resolved_explicit = api._resolve_audio_llm_config(cfg)
    assert resolved_explicit is not None and resolved_explicit.model == "gemini-4-neon"

    # audio_listen: never disables the audio pass completely
    cfg.planning.audio_listen = "never"
    assert api._resolve_audio_llm_config(cfg) is None


def test_build_manifest_runs_audio_listen_pass_for_text_only_planner(tmp_path: Path, monkeypatch):
    mix_file = tmp_path / "mix.wav"
    mix_file.write_bytes(b"fake-wav")
    # Stub audio_block so no ffmpeg subprocess is needed in unit test
    monkeypatch.setattr(
        "mvdirector.llm.base.audio_block",
        lambda _p: {
            "type": "audio",
            "source": {"type": "base64", "media_type": "audio/mpeg", "data": "AA=="},
        },
    )

    cfg = ProjectConfig(
        title="Waltz Test",
        llm=LLMConfig(provider="anthropic", model="claude-opus-5-5"),
    )
    analysis = Analysis(
        duration=8.0,
        tempo_bpm=89.0,
        beats=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        bars=[0.0, 4.0, 8.0],
        bar_intensity=[0.4, 0.7],
        vocal_windows=[TimeSpan(start=0.0, end=8.0)],
        lyric_lines=[LyricLine(text="Half a beat behind", section="Verse 1", start=0.0, end=8.0)],
    )
    planner = _FakeTextPlanner()
    listener = _FakeAudioListener()

    build_manifest(
        cfg,
        analysis,
        "[Verse 1]\nHalf a beat behind\n",
        "Moody duet.",
        planner,
        mix_path=mix_file,
        audio_llm=listener,
    )
    assert len(listener.calls) == 1
    assert any(b.get("type") == "audio" for b in listener.calls[0])
    bible_prompt = next(text for name, text in planner.prompts if name == "BibleReply")
    assert "NATIVE AUDIO LISTENING NOTES" in bible_prompt
    assert "3/4 waltz sway" in bible_prompt
    assert "uncanny gothic grief" in bible_prompt


def test_build_manifest_skips_audio_listen_when_planner_supports_audio_or_disabled(
    tmp_path: Path, monkeypatch
):
    mix_file = tmp_path / "mix.wav"
    mix_file.write_bytes(b"fake-wav")
    monkeypatch.setattr(
        "mvdirector.llm.base.audio_block",
        lambda _p: {
            "type": "audio",
            "source": {"type": "base64", "media_type": "audio/mpeg", "data": "AA=="},
        },
    )
    analysis = Analysis(
        duration=8.0,
        tempo_bpm=89.0,
        beats=[0.0, 2.0, 4.0, 6.0, 8.0],
        bars=[0.0, 4.0, 8.0],
        bar_intensity=[0.4, 0.7],
        vocal_windows=[TimeSpan(start=0.0, end=8.0)],
        lyric_lines=[LyricLine(text="Line", section="Verse 1", start=0.0, end=8.0)],
    )

    # 1. audio_listen: never skips audio_llm
    cfg_never = ProjectConfig(title="No Listen")
    cfg_never.planning.audio_listen = "never"
    planner = _FakeTextPlanner()
    listener = _FakeAudioListener()
    build_manifest(
        cfg_never,
        analysis,
        "[Verse 1]\nLine\n",
        "Style",
        planner,
        mix_path=mix_file,
        audio_llm=listener,
    )
    assert listener.calls == []

    # 2. Audio-capable main planner skips redundant audio_llm call
    planner.supports_audio = True
    cfg_auto = ProjectConfig(title="Native Audio Planner")
    build_manifest(
        cfg_auto,
        analysis,
        "[Verse 1]\nLine\n",
        "Style",
        planner,
        mix_path=mix_file,
        audio_llm=listener,
    )
    assert listener.calls == []


def test_qc_vision_threshold_shorthands_propagate_and_respect_overrides(tmp_path: Path):
    cfg_path = tmp_path / "project.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "title": "Calibrated Hosted Setup",
                "llm": {"provider": "anthropic", "model": "claude-opus-5-5"},
                "planning": {"audio_listen": "auto"},
                "qc": {
                    "llm": {"provider": "gemini", "model": "gemini-3.8-flash"},
                    "vision_spec": 0.71,
                    "vision_artifacts": 0.72,
                    "vision_character": 0.71,
                },
            }
        ),
        encoding="utf-8",
    )
    cfg = load_config(cfg_path)
    assert cfg.qc.vision_spec == 0.71
    assert cfg.qc.vision_artifacts == 0.72
    assert cfg.qc.vision_character == 0.71
    assert cfg.qc.image.vision_spec_hard_min == 0.71
    assert cfg.qc.image.vision_hard_min == 0.72
    assert cfg.qc.image.vision_character_hard_min == 0.71
    assert cfg.qc.video.vision_spec_hard_min == 0.71
    assert cfg.qc.video.vision_hard_min == 0.72
    assert cfg.qc.video.vision_character_hard_min == 0.71

    # Explicit per-media override takes precedence over top-level qc shorthand
    override_path = tmp_path / "override.yaml"
    override_path.write_text(
        yaml.safe_dump(
            {
                "title": "Per-Media Override",
                "qc": {
                    "vision_spec": 0.71,
                    "vision_artifacts": 0.72,
                    "vision_character": 0.71,
                    "video": {"vision_spec_hard_min": 0.65},
                    "image": {"vision_character_hard_min": 0.80},
                },
            }
        ),
        encoding="utf-8",
    )
    override_cfg = load_config(override_path)
    assert override_cfg.qc.image.vision_spec_hard_min == 0.71
    assert override_cfg.qc.video.vision_spec_hard_min == 0.65
    assert override_cfg.qc.image.vision_character_hard_min == 0.80
    assert override_cfg.qc.video.vision_character_hard_min == 0.71
    assert override_cfg.qc.image.vision_hard_min == 0.72
    assert override_cfg.qc.video.vision_hard_min == 0.72

    # Out-of-bounds and NaN values are rejected
    for bad_qc in (
        {"vision_spec": 1.2},
        {"vision_artifacts": -0.05},
        {"vision_character": float("nan")},
    ):
        bad_path = tmp_path / "bad.yaml"
        bad_path.write_text(yaml.safe_dump({"title": "Bad", "qc": bad_qc}), encoding="utf-8")
        with pytest.raises(ConfigError):
            load_config(bad_path)
