# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Tests for the multi-model Track 1 (Planning) & Track 2 (Vision QC) benchmark harness."""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import numpy as np
import pytest
import yaml
from PIL import Image
from pydantic import BaseModel
from typer.testing import CliRunner

from mvdirector.bench import (
    BenchmarkConfig,
    GPUCostConfig,
    InstrumentedLLM,
    ModelArm,
    ModelPricing,
    QCItemVerdict,
    default_arms,
    estimate_cost_usd,
    evaluate_hard_gate,
    evaluate_planning_quality,
    generate_blind_viewer,
    load_benchmark_config,
    load_qc_dataset,
    run_planning_benchmark,
    run_vision_qc_benchmark,
    scaffold_qc_dataset,
    select_matched_shots,
    write_default_benchmark_config,
)
from mvdirector.calibrate import _best_cut
from mvdirector.cli import app
from mvdirector.errors import ConfigError, LLMError
from mvdirector.llm import TokenUsage, build_llm, text_block
from mvdirector.llm.anthropic_backend import AnthropicChat
from mvdirector.models import (
    Analysis,
    AssetStatus,
    Attempt,
    AudioSlice,
    Character,
    CheckResult,
    CheckStatus,
    Framing,
    LyricLine,
    Manifest,
    QCReport,
    ReferenceSet,
    RetryStrategy,
    SetDesign,
    Shot,
    ShotType,
    TimeSpan,
    VisualBible,
    WardrobeState,
)


class _DummyReply(BaseModel):
    ok: bool


class _MockTransport:
    def __init__(self, responses: list[httpx.Response]) -> None:
        self._responses = list(responses)
        self.requests: list[dict[str, Any]] = []

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self._handle)

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(json.loads(request.content))
        if len(self._responses) > 1:
            return self._responses.pop(0)
        return self._responses[0]


def _save_png(path: Path, color: tuple[int, int, int] = (80, 90, 100)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(np.full((64, 64, 3), color, dtype=np.uint8))
    img.save(path)
    return path


def _make_sample_project(root: Path) -> Path:
    """Create a minimal project with frozen analysis.json, lyrics, and style brief."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "inputs").mkdir(exist_ok=True)
    (root / "inputs" / "lyrics.txt").write_text(
        "[Intro]\n\n"
        "[Verse 1]\nSmoke rises over glass\n\n"
        "[Chorus]\nSugar and ash tonight\n\n"
        "[Outro]\nFade into dawn\n",
        encoding="utf-8",
    )
    (root / "inputs" / "style.md").write_text(
        "Moody 35mm indie film aesthetic, solo female vocalist in a rain-lit diner and rooftop.\n",
        encoding="utf-8",
    )
    cfg = {
        "title": "Test Song",
        "video": {
            "width": 1280,
            "height": 720,
            "fps": 24,
            "min_clip_seconds": 2.0,
            "max_clip_seconds": 6.0,
            "cut_mode": "bars",
        },
        "llm": {"enabled": True, "provider": "gemini", "model": "gemini-3.8-flash"},
    }
    (root / "project.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")

    analysis = Analysis(
        duration=24.0,
        tempo_bpm=120.0,
        beats=[float(i) * 0.5 for i in range(49)],
        bars=[0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0],
        bar_intensity=[0.2, 0.35, 0.45, 0.55, 0.85, 0.90, 0.88, 0.40, 0.30, 0.75, 0.92, 0.25],
        vocal_windows=[
            TimeSpan(start=4.0, end=8.0),
            TimeSpan(start=8.0, end=14.0),
            TimeSpan(start=18.0, end=22.0),
        ],
        lyric_lines=[
            LyricLine(text="Smoke rises over glass", section="Verse 1", start=4.0, end=8.0),
            LyricLine(text="Sugar and ash tonight", section="Chorus", start=8.0, end=14.0),
            LyricLine(text="Fade into dawn", section="Outro", start=18.0, end=22.0),
        ],
    )
    (root / "analysis.json").write_text(analysis.model_dump_json(indent=2), encoding="utf-8")
    return root


def _bible_json() -> dict[str, Any]:
    wardrobe = [
        {
            "id": "trench",
            "description": (
                "Charcoal wool trench coat over ribbed turtleneck, black trousers, worn boots."
            ),
        },
        {
            "id": "sweater",
            "description": (
                "Ribbed charcoal turtleneck with sleeves pushed up, black trousers, worn boots."
            ),
        },
    ]
    return {
        "interpretation": {
            "meaning": "A quiet reckoning after a breakup in a rain-drenched city.",
            "tone": "Restrained and smoky.",
            "music_feel": "Slow build into an electric chorus lift.",
            "visual_motifs": ["rain on glass", "cooling coffee"],
            "section_beats": [
                {
                    "section": "Intro",
                    "emotion": "anticipation",
                    "visual_beat": "empty booth in neon rain",
                    "camera_energy": "still",
                },
                {
                    "section": "Verse 1",
                    "emotion": "guarded",
                    "visual_beat": "singer at diner counter",
                    "camera_energy": "slow push-in",
                },
                {
                    "section": "Chorus",
                    "emotion": "release",
                    "visual_beat": "rooftop sodium glow",
                    "camera_energy": "tighter framing",
                },
                {
                    "section": "Outro",
                    "emotion": "calm",
                    "visual_beat": "dawn alley",
                    "camera_energy": "pull-back",
                },
            ],
        },
        "concept": "Three linked urban spaces move from rain-locked diner to open rooftop dawn.",
        "characters": [
            {
                "id": "mara",
                "name": "Mara",
                "appearance": "Late-20s woman with oval face, dark wavy bob, natural skin texture.",
                "invariants": "pair of small silver hoop earrings",
                "wardrobe": wardrobe,
            }
        ],
        "appearance": "Late-20s woman with oval face, dark wavy bob, natural skin texture.",
        "invariants": "pair of small silver hoop earrings",
        "wardrobe": wardrobe,
        "style": "35mm Kodak Vision3 film grain",
        "palette": "amber tungsten and teal rain shadows",
        "lighting": "practical diner neon and wet street reflections",
        "negative_prompt": "cgi, plastic skin, subtitles, watermark",
        "settings": [
            {
                "name": "Rain Diner",
                "sections": ["Intro", "Verse 1"],
                "description": "Empty corner booth beside rain-streaked glass with a ceramic mug.",
                "movement": "Slow push-in, steam curling from the mug.",
            },
            {
                "name": "Wet Rooftop",
                "sections": ["Chorus"],
                "description": "Tar rooftop under sodium streetlights with wet vents.",
                "movement": "Firm handheld sway, wind moving hair.",
            },
            {
                "name": "Dawn Alley",
                "sections": ["Outro"],
                "description": "Quiet brick lane catching pale blue pre-dawn mist.",
                "movement": "Slow pull-back as mist drifts.",
            },
        ],
        "props": [],
        "rules": [
            "Keep camera in open floor space at eye height.",
            "Hold one set per musical section.",
            "Keep singing mouths unobstructed.",
            "Never include on-screen text.",
        ],
    }


def _shots_json_for(prompt_text: str) -> dict[str, Any]:
    """Parse requested shot ids from planner prompt and return valid creative JSON."""
    entries = []
    for match in re.finditer(r"- id=(s\d+)\b.*?type=(lipsync|performance|broll)\b", prompt_text):
        sid, stype = match.group(1), match.group(2)
        idx = int(sid[1:])
        if stype == "broll":
            entries.append(
                {
                    "id": sid,
                    "type": "broll",
                    "framing": "close_up",
                    "setting": "Rain Diner",
                    "still_prompt": (
                        "Close-up of amber neon reflecting across a rain-streaked window pane."
                    ),
                    "motion_prompt": (
                        "Raindrops slide slowly down the glass while neon reflections shimmer."
                    ),
                    "wardrobe": None,
                    "character": None,
                    "characters": [],
                }
            )
        elif stype == "lipsync":
            entries.append(
                {
                    "id": sid,
                    "type": "lipsync",
                    "framing": "close_up" if idx % 2 == 0 else "medium_close",
                    "setting": "Wet Rooftop" if idx >= 3 else "Rain Diner",
                    "still_prompt": (
                        "Close-up of Mara in three-quarter view singing under warm tungsten light."
                        if idx % 2 == 0
                        else "Medium close-up of Mara leaning forward under wet rooftop glow."
                    ),
                    "motion_prompt": (
                        "Mara sings steadily with subtle head tilts and natural breathing sway."
                    ),
                    "wardrobe": "sweater" if idx >= 4 else "trench",
                    "character": "mara",
                    "characters": ["mara"],
                }
            )
        else:
            entries.append(
                {
                    "id": sid,
                    "type": "performance",
                    "framing": "wide" if idx == 1 else "medium",
                    "setting": "Dawn Alley" if idx >= 5 else "Rain Diner",
                    "still_prompt": (
                        "Wide shot of Mara standing beside the rain-lit window in open space."
                    ),
                    "motion_prompt": (
                        "Mara shifts weight slowly while headlights drift past. Slow push-in."
                    ),
                    "wardrobe": "sweater" if idx >= 4 else "trench",
                    "character": "mara",
                    "characters": ["mara"],
                }
            )
    return {"shots": entries}


def _v(
    spec: float,
    art: float,
    char: float | None,
    issues: list[str] | None = None,
    fix: str = "",
) -> dict[str, Any]:
    return {
        "spec_match": spec,
        "artifact_free": art,
        "character_match": char,
        "issues": issues or [],
        "fix": fix,
    }


# ============================================================ Config & Pricing


def test_default_benchmark_matrix_includes_six_arms_plus_b2_control(tmp_path: Path):
    cfg = BenchmarkConfig()
    assert [a.id for a in cfg.arms] == ["A", "B", "B2", "C", "D", "E", "F", "G", "H", "I", "J"]
    assert [a.id for a in cfg.select_arms()] == ["A", "B", "C", "D", "E", "F"]
    assert [a.id for a in cfg.select_arms(include_controls=True)] == [
        "A",
        "B",
        "B2",
        "C",
        "D",
        "E",
        "F",
        "G",
        "H",
        "I",
        "J",
    ]
    assert [a.id for a in cfg.select_arms(["B", "B2", "G", "J"])] == ["B", "B2", "G", "J"]

    b_arm = cfg.select_arms(["B"])[0]
    b2_arm = cfg.select_arms(["B2"])[0]
    c_arm = cfg.select_arms(["C"])[0]
    d_arm = cfg.select_arms(["D"])[0]
    e_arm = cfg.select_arms(["E"])[0]
    f_arm = cfg.select_arms(["F"])[0]
    g_arm = cfg.select_arms(["G"])[0]
    h_arm = cfg.select_arms(["H"])[0]
    i_arm = cfg.select_arms(["I"])[0]
    j_arm = cfg.select_arms(["J"])[0]
    assert b_arm.model == "gemini-4-neon" and b_arm.attach_audio is True
    assert b2_arm.model == "gemini-4-neon" and b2_arm.attach_audio is False
    assert c_arm.model == "claude-sonnet-5-5"
    assert c_arm.effective_pricing() == ModelPricing(input_per_million=2.0, output_per_million=10.0)
    assert d_arm.model == "claude-opus-5-5"
    assert d_arm.effective_pricing() == ModelPricing(input_per_million=4.0, output_per_million=20.0)
    assert e_arm.model == "gpt-6.2-sol"
    assert f_arm.model == "gpt-6.2-astra"
    assert g_arm.model == "gemini-4-carbon" and g_arm.attach_audio is True and g_arm.optional
    assert h_arm.model == "claude-fable-5-2" and h_arm.optional
    assert i_arm.model == "claude-haiku-5-5" and i_arm.optional
    assert j_arm.model == "gpt-6.1-terra" and j_arm.optional

    yaml_path = write_default_benchmark_config(tmp_path / "bench.yaml")
    reloaded = load_benchmark_config(yaml_path)
    assert [a.id for a in reloaded.arms] == [a.id for a in default_arms()]


def test_cost_estimation_respects_provider_token_accounting():
    pricing = ModelPricing(
        input_per_million=2.0,
        output_per_million=10.0,
        cached_input_per_million=0.5,
    )
    usage = TokenUsage(
        input_tokens=1_000_000,
        output_tokens=500_000,
        thoughts_tokens=200_000,
        cached_tokens=400_000,
    )

    # Gemini: promptTokenCount includes cached; thoughts billed separately.
    gemini_usd = estimate_cost_usd(usage, pricing, provider="gemini")
    assert gemini_usd == pytest.approx(0.6 * 2.0 + 0.4 * 0.5 + 0.5 * 10.0 + 0.2 * 10.0)

    # OpenAI: completion_tokens ALREADY includes reasoning_tokens (no double counting).
    openai_usd = estimate_cost_usd(usage, pricing, provider="openai")
    assert openai_usd == pytest.approx(0.6 * 2.0 + 0.4 * 0.5 + 0.5 * 10.0)

    # Anthropic: cache_read_input_tokens reported separately from input_tokens.
    anthropic_usd = estimate_cost_usd(usage, pricing, provider="anthropic")
    assert anthropic_usd == pytest.approx(1.0 * 2.0 + 0.4 * 0.5 + 0.5 * 10.0 + 0.2 * 10.0)


# ============================================================ LLM Telemetry & Refusals


def test_gemini_and_openai_usage_schema_retry_and_refusal_telemetry():
    gem_server = _MockTransport(
        [
            httpx.Response(
                200,
                json={
                    "candidates": [
                        {"content": {"parts": [{"text": "not json"}]}, "finishReason": "STOP"}
                    ],
                    "usageMetadata": {
                        "promptTokenCount": 100,
                        "candidatesTokenCount": 20,
                        "thoughtsTokenCount": 50,
                    },
                },
            ),
            httpx.Response(
                200,
                json={
                    "candidates": [
                        {"content": {"parts": [{"text": '{"ok": true}'}]}, "finishReason": "STOP"}
                    ],
                    "usageMetadata": {
                        "promptTokenCount": 140,
                        "candidatesTokenCount": 10,
                        "thoughtsTokenCount": 30,
                    },
                },
            ),
        ]
    )
    arm_b2 = BenchmarkConfig().select_arms(["B2"])[0]
    raw = build_llm(
        arm_b2.to_llm_config(),
        environ={"GEMINI_API_KEY": "g-key"},
        transport=gem_server.transport(),
        sleep=lambda _: None,
    )
    assert raw is not None
    with InstrumentedLLM(raw, arm_b2) as client:
        assert client.supports_audio is False  # B2 disables audio even on GeminiChat
        ans = client.complete_json(system="test", content=[text_block("hi")], schema=_DummyReply)
        assert ans.ok is True
        summary = client.summary()
        assert summary.calls == 1
        assert summary.first_pass_ok == 0
        assert summary.schema_retries == 1
        assert summary.input_tokens == 240
        assert summary.output_tokens == 30
        assert summary.thoughts_tokens == 80
        assert summary.estimated_usd > 0

    oai_server = _MockTransport(
        [
            httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "refusal": "Policy block",
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 50, "completion_tokens": 10},
                },
            )
        ]
    )
    arm_e = BenchmarkConfig().select_arms(["E"])[0]
    raw_oai = build_llm(
        arm_e.to_llm_config(),
        environ={"OPENAI_API_KEY": "sk-key"},
        transport=oai_server.transport(),
        sleep=lambda _: None,
    )
    assert raw_oai is not None
    with InstrumentedLLM(raw_oai, arm_e) as client:
        with pytest.raises(LLMError, match="refused"):
            client.complete_json(system="test", content=[text_block("hi")], schema=_DummyReply)
        assert client.summary().refusals == 1


def test_anthropic_usage_and_refusal_detection():
    class _FakeAnthropicClient:
        def __init__(self, responses: list[Any]) -> None:
            self._responses = list(responses)
            self.messages = SimpleNamespace(create=self._create)

        def _create(self, **_: Any) -> Any:
            return self._responses.pop(0)

    fake = _FakeAnthropicClient(
        [
            SimpleNamespace(
                content=[SimpleNamespace(type="text", text='{"ok": true}')],
                stop_reason="end_turn",
                usage=SimpleNamespace(
                    input_tokens=200, output_tokens=40, cache_read_input_tokens=50
                ),
            ),
            SimpleNamespace(
                content=[],
                stop_reason="refusal",
                usage=SimpleNamespace(input_tokens=100, output_tokens=0, cache_read_input_tokens=0),
            ),
        ]
    )
    arm_c = BenchmarkConfig().select_arms(["C"])[0]
    backend = AnthropicChat(model=arm_c.model, max_tokens=1024, client=fake)
    with InstrumentedLLM(backend, arm_c) as client:
        res = client.complete_json(system="s", content=[text_block("p")], schema=_DummyReply)
        assert res.ok is True
        with pytest.raises(LLMError, match="refused"):
            client.complete_json(system="s", content=[text_block("p")], schema=_DummyReply)
        summary = client.summary()
        assert summary.calls == 2
        assert summary.input_tokens == 200
        assert summary.output_tokens == 40
        assert summary.cached_tokens == 50
        assert summary.refusals == 1


# ============================================================ Track 1 Planning + Blind Viewer


def test_track1_planning_benchmark_and_blind_viewer(tmp_path: Path):
    proj_dir = _make_sample_project(tmp_path / "song_alpha")
    out_dir = tmp_path / "bench_out"

    def _fake_factory(arm: ModelArm, _env: Any) -> Any:
        def _handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            if "contents" in body:
                text = " ".join(
                    p.get("text", "") for p in body["contents"][0]["parts"] if "text" in p
                )
                payload = _bible_json() if "SONG SECTIONS" in text else _shots_json_for(text)
                return httpx.Response(
                    200,
                    json={
                        "candidates": [
                            {
                                "content": {"parts": [{"text": json.dumps(payload)}]},
                                "finishReason": "STOP",
                            }
                        ],
                        "usageMetadata": {
                            "promptTokenCount": 500,
                            "candidatesTokenCount": 250,
                            "thoughtsTokenCount": 40,
                        },
                    },
                )
            text = body["messages"][1]["content"]
            if isinstance(text, list):
                text = " ".join(p.get("text", "") for p in text)
            payload = _bible_json() if "SONG SECTIONS" in text else _shots_json_for(text)
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": json.dumps(payload)},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 500, "completion_tokens": 250},
                },
            )

        return build_llm(
            arm.to_llm_config(),
            environ={"GEMINI_API_KEY": "g-key", "OPENAI_API_KEY": "sk-key"},
            transport=httpx.MockTransport(_handler),
            sleep=lambda _: None,
        )

    cfg = BenchmarkConfig()
    results = run_planning_benchmark(
        [proj_dir],
        cfg,
        out_dir,
        arm_ids=["A", "B", "B2", "E"],
        llm_factory=_fake_factory,
    )
    assert len(results) == 4
    assert all(r.status == "ok" for r in results)
    assert all(r.telemetry.first_pass_rate == 1.0 for r in results)
    assert all(r.quality.fallback_shots == 0 for r in results)
    assert all(r.quality.invariants_defined for r in results)
    assert all(r.quality.invariant_always_coverage == 1.0 for r in results)
    assert all(r.quality.affirmative_phrasing_compliance == 1.0 for r in results)
    assert (out_dir / "planning" / "summary.md").is_file()
    assert (out_dir / "planning" / "summary.csv").is_file()

    viewer_html, key_json = generate_blind_viewer(
        results, out_dir / "planning" / "blind_review.html", seed=42
    )
    html_text = viewer_html.read_text(encoding="utf-8")
    assert "Director 1" in html_text
    assert "Director 4" in html_text
    # Real model IDs must NOT appear in plaintext outside the base64-encoded key.
    blind_section = html_text.split("const ENCODED_KEY =")[0]
    assert "gemini-4-neon" not in blind_section
    assert "gpt-6.2-sol" not in blind_section

    key_data = json.loads(key_json.read_text(encoding="utf-8"))
    aliases = key_data["tracks"]["song_alpha"]
    assert set(aliases.keys()) == {"Director 1", "Director 2", "Director 3", "Director 4"}
    assert {v["arm_id"] for v in aliases.values()} == {"A", "B", "B2", "E"}
    encoded_str = html_text.split('const ENCODED_KEY = "')[1].split('";')[0]
    assert json.loads(base64.b64decode(encoded_str).decode("utf-8")) == key_data

    # Verify budget cap stops before starting additional uncached arms, and staged runs merge.
    budget_dir = tmp_path / "budget_out"
    stage1 = run_planning_benchmark(
        [proj_dir],
        cfg,
        budget_dir,
        arm_ids=["A", "B", "G"],
        max_budget_usd=0.0001,  # stops right after Arm A finishes
        llm_factory=_fake_factory,
    )
    assert [r.arm_id for r in stage1] == ["A"]
    # Stage 2 appends Arm G without re-running A; summary merges both A and G.
    stage2 = run_planning_benchmark(
        [proj_dir],
        cfg,
        budget_dir,
        arm_ids=["G"],
        llm_factory=_fake_factory,
    )
    assert [r.arm_id for r in stage2] == ["G"]
    merged_summary = json.loads((budget_dir / "planning" / "summary.json").read_text("utf-8"))
    assert {entry["arm_id"] for entry in merged_summary} == {"A", "G"}

    # Verify CLI `mvd bench viewer` regenerates from saved outputs
    runner = CliRunner()
    regen_html = tmp_path / "regen.html"
    cli_res = runner.invoke(
        app, ["bench", "viewer", "--out", str(out_dir), "--html", str(regen_html), "--seed", "7"]
    )
    assert cli_res.exit_code == 0, cli_res.output
    assert regen_html.is_file()


def test_planning_quality_detects_fallbacks_contradictions_and_negations():
    manifest = Manifest(
        title="t",
        duration=12.0,
        fps=24,
        character=Character(
            name="Mara",
            appearance="Woman in late 20s",
            invariants="small silver hoop earrings",
            wardrobe=[WardrobeState(id="base", description="Black coat, no jewelry, black boots.")],
        ),
        visual_bible=VisualBible(
            style="35mm",
            settings=[SetDesign(name="Diner", description="Neon diner")],
        ),
        shots=[
            Shot(
                id="s001",
                index=0,
                start=0.0,
                end=4.0,
                type=ShotType.LIPSYNC,
                framing=Framing.CLOSE_UP,
                setting="Diner",
                wardrobe="base",
                still_prompt="Close-up from the waist up showing boots, no minute hand on clock.",
                motion_prompt="Film still of Mara singing softly.",
                audio_slice=AudioSlice(path="a.wav", start=0.0, end=4.0, head_handle=0.0),
                num_frames=97,
            ),
            Shot(
                id="s002",
                index=1,
                start=4.0,
                end=8.0,
                type=ShotType.BROLL,
                framing=Framing.WIDE,
                setting="Diner",
                still_prompt=(
                    "Wide shot of an object or corner of the Diner that echoes the song, "
                    "no people, calm mood."
                ),
                motion_prompt=(
                    "Light, dust and shadow drift slowly through the frame. "
                    "No people. Steady camera."
                ),
                audio_slice=AudioSlice(path="a.wav", start=4.0, end=8.0, head_handle=0.0),
                num_frames=97,
            ),
        ],
    )
    q = evaluate_planning_quality(manifest)
    assert q.fallback_shots == 1  # s002 matches legacy rule-based signature
    assert q.invariant_wardrobe_contradictions == 1
    assert q.framing_detail_violations == 1  # s001 close_up mentions waist up + boots
    assert q.negative_exclusion_shots == 2  # "no minute hand" + "no people"
    assert q.motion_still_word_leaks == 1  # "Film still" in s001 motion_prompt

    # New affirmative rule-based b-roll fallback is detected without adding negative exclusions.
    from mvdirector.planning import planner as planner_mod

    s003 = Shot(
        id="s003",
        index=2,
        start=8.0,
        end=12.0,
        type=ShotType.BROLL,
        setting="Diner",
        audio_slice=AudioSlice(path="a.wav", start=8.0, end=12.0, head_handle=0.0),
        num_frames=97,
    )
    planner_mod._apply_rule_based(s003)
    manifest.shots.append(s003)
    q2 = evaluate_planning_quality(manifest)
    assert q2.fallback_shots == 2
    assert q2.negative_exclusion_shots == 2  # s003 uses affirmative uninhabited phrasing


# ============================================================ Track 2 Vision QC


def test_track2_vision_qc_dataset_scaffolding_replay_calibration_and_effective_cost(tmp_path: Path):
    proj_dir = _make_sample_project(tmp_path / "qc_proj")
    ref_sheet = _save_png(proj_dir / "assets" / "reference" / "mara_sheet.png", (120, 100, 90))
    still_paths = [
        _save_png(proj_dir / "assets" / "stills" / f"s00{i}_a01.png", (20 * i, 40, 60))
        for i in range(1, 5)
    ]

    shots = []
    for idx, sp in enumerate(still_paths, start=1):
        passed_prior = idx in (1, 2)
        check_status = CheckStatus.PASSED if passed_prior else CheckStatus.FAILED
        shots.append(
            Shot(
                id=f"s00{idx}",
                index=idx - 1,
                start=float((idx - 1) * 4),
                end=float(idx * 4),
                type=ShotType.LIPSYNC if idx != 4 else ShotType.BROLL,
                framing=Framing.CLOSE_UP if idx <= 2 else Framing.WIDE,
                setting="Rain Diner",
                wardrobe="trench" if idx <= 2 else None,
                still_prompt="Mara in dark shadow diner." if idx != 4 else "Rain on diner glass.",
                motion_prompt="Subtle sway.",
                audio_slice=AudioSlice(path="a.wav", start=0.0, end=4.0, head_handle=0.0),
                num_frames=97,
                still={
                    "status": AssetStatus.PASSED if passed_prior else AssetStatus.EXHAUSTED,
                    "selected_attempt": 1,
                    "attempts": [
                        Attempt(
                            number=1,
                            strategy=RetryStrategy.INITIAL,
                            seed=idx,
                            prompt="p",
                            path=str(sp.relative_to(proj_dir)),
                            qc=QCReport(
                                passed=passed_prior,
                                score=0.9 if passed_prior else 0.3,
                                checks=[
                                    CheckResult(
                                        name="vision_spec",
                                        status=check_status,
                                        score=0.9 if passed_prior else 0.2,
                                        metrics={"vision_spec": 0.9 if passed_prior else 0.2},
                                    )
                                ],
                            ),
                        )
                    ],
                },
            )
        )
    manifest = Manifest(
        title="QC Test",
        duration=16.0,
        fps=24,
        character=Character(
            name="Mara",
            appearance="Late-20s woman",
            wardrobe=[WardrobeState(id="trench", description="Charcoal trench")],
            references={"trench": ReferenceSet(sheet=str(ref_sheet.relative_to(proj_dir)))},
        ),
        visual_bible=VisualBible(
            style="35mm",
            lighting="dark moody neon",
            settings=[SetDesign(name="Rain Diner", description="Dim corner booth")],
        ),
        shots=shots,
    )
    (proj_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

    # 1. Scaffold starter CSV dataset and reload it
    labels_csv = tmp_path / "qc_labels.csv"
    scaffolded = scaffold_qc_dataset([proj_dir], labels_csv, target_size=4)
    assert len(scaffolded) == 4
    reloaded = load_qc_dataset(labels_csv)
    assert len(reloaded) == 4
    assert sum(1 for i in reloaded if i.good) == 2
    assert sum(1 for i in reloaded if not i.good) == 2

    # 2. Replay through two mocked arms: Arm A (1 FN, 0 FP) vs Arm B (perfect separator)
    verdicts_by_arm = {
        "A": [
            _v(0.90, 0.92, 0.88),
            _v(0.85, 0.86, 0.80),
            _v(0.72, 0.75, 0.70),  # FN!
            _v(0.20, 0.85, None, ["wrong"], "fix"),
        ],
        "B": [
            _v(0.95, 0.95, 0.92),
            _v(0.91, 0.90, 0.88),
            _v(0.25, 0.80, 0.30, ["bad spec"], "fix"),
            _v(0.15, 0.90, None, ["bad"], "fix"),
        ],
    }

    def _qc_factory(arm: ModelArm, _env: Any) -> Any:
        queue = list(verdicts_by_arm[arm.id])

        def _handler(_request: httpx.Request) -> httpx.Response:
            v = queue.pop(0)
            return httpx.Response(
                200,
                json={
                    "candidates": [
                        {"content": {"parts": [{"text": json.dumps(v)}]}, "finishReason": "STOP"}
                    ],
                    "usageMetadata": {"promptTokenCount": 1200, "candidatesTokenCount": 100},
                },
            )

        return build_llm(
            arm.to_llm_config(),
            environ={"GEMINI_API_KEY": "g-key"},
            transport=httpx.MockTransport(_handler),
            sleep=lambda _: None,
        )

    out_dir = tmp_path / "qc_out"
    gpu_costs = GPUCostConfig(still_usd=0.02, video_usd=0.35, shots_per_video=90)
    cfg = BenchmarkConfig(gpu_costs=gpu_costs)
    results = run_vision_qc_benchmark(
        reloaded,
        cfg,
        out_dir,
        arm_ids=["A", "B"],
        llm_factory=_qc_factory,
    )
    arm_a, arm_b = results[0], results[1]

    # Verify balanced accuracy parity with calibrate._best_cut
    spec_good = np.array([0.90, 0.85])
    spec_bad = np.array([0.72, 0.20])
    expected_cut, expected_acc = _best_cut(spec_good, spec_bad, True)
    assert arm_a.per_check["vision_spec"].optimal_threshold == expected_cut
    assert arm_a.per_check["vision_spec"].balanced_accuracy == expected_acc

    assert arm_a.default_gate is not None and arm_b.default_gate is not None
    assert arm_a.default_gate.fn == 1 and arm_a.default_gate.fn_rate == 0.5
    assert arm_b.default_gate.fn == 0 and arm_b.default_gate.fp == 0
    assert arm_b.default_gate.balanced_accuracy == 1.0
    # Pricier Arm B wins on Net Effective Cost / 90-shot video by preventing wasted video renders.
    assert arm_b.default_gate.net_effective_cost_usd < arm_a.default_gate.net_effective_cost_usd
    assert (out_dir / "vision_qc" / "qc_summary.md").is_file()
    assert (out_dir / "vision_qc" / "qc_summary.csv").is_file()
    assert (out_dir / "vision_qc" / "qc_errors.html").is_file()


def test_net_effective_cost_formula_exact_values():
    verdicts = [
        QCItemVerdict(
            item_id="p:s1:a1",
            project="p",
            shot_id="s1",
            still_path="x.png",
            ground_truth_good=True,
            vision_spec=0.4,  # FP (good rejected)
            vision_artifacts=0.9,
            vision_character=0.9,
            estimated_usd=0.01,
        ),
        QCItemVerdict(
            item_id="p:s2:a1",
            project="p",
            shot_id="s2",
            still_path="y.png",
            ground_truth_good=False,
            vision_spec=0.9,  # FN (bad approved)
            vision_artifacts=0.9,
            vision_character=0.9,
            estimated_usd=0.01,
        ),
    ]
    gate = evaluate_hard_gate(
        verdicts,
        spec_min=0.60,
        artifacts_min=0.50,
        character_min=0.40,
        mean_llm_usd_per_still=0.01,
        gpu_costs=GPUCostConfig(still_usd=0.02, video_usd=0.35, shots_per_video=90),
    )
    # 1 FP and 1 FN out of 2 -> scaled to 90 shots = 45 FP and 45 FN
    assert gate.llm_qc_spend_usd == pytest.approx(0.90)
    assert gate.fp_still_gpu_cost_usd == pytest.approx(45 * 0.02)
    assert gate.fn_video_gpu_cost_usd == pytest.approx(45 * 0.35)
    assert gate.net_effective_cost_usd == pytest.approx(0.90 + 45 * 0.02 + 45 * 0.35)


def test_cli_bench_init_config_and_validation(tmp_path: Path):
    runner = CliRunner()
    cfg_file = tmp_path / "bench.yaml"
    res = runner.invoke(app, ["bench", "init-config", "--output", str(cfg_file)])
    assert res.exit_code == 0, res.output
    assert cfg_file.is_file()
    res2 = runner.invoke(app, ["bench", "init-config", "--output", str(cfg_file)])
    assert res2.exit_code == 1


def test_select_matched_shots_picks_diverse_song_roles():
    shots = [
        Shot(
            id=f"s{i:03d}",
            index=i - 1,
            start=float((i - 1) * 3),
            end=float(i * 3),
            type=ShotType.BROLL if i == 5 else ShotType.LIPSYNC,
            section=(
                "Intro"
                if i == 1
                else "Verse 1"
                if i in (2, 3)
                else "Chorus"
                if i in (4, 5, 6)
                else "Verse 2"
                if i in (7, 8)
                else "Bridge"
                if i == 9
                else "Final Chorus"
                if i in (10, 11)
                else "Outro"
            ),
            intensity=0.95 if i == 10 else 0.5,
            wardrobe="coat" if i < 7 else "jacket",
            audio_slice=AudioSlice(path="a.wav", start=0.0, end=3.0, head_handle=0.0),
            num_frames=73,
        )
        for i in range(1, 13)
    ]
    manifest = Manifest(title="t", duration=36.0, fps=24, shots=shots)
    matched = select_matched_shots([manifest], max_shots=10)
    roles = {role for _, role in matched}
    ids = [sid for sid, _ in matched]
    assert 8 <= len(matched) <= 10
    assert "Intro" in roles
    assert "Chorus Drop" in roles
    assert "Cutaway (B-Roll)" in roles
    assert "Wardrobe Shift" in roles
    assert "Finale" in roles
    assert ids[0] == "s001" and ids[-1] == "s012"


def test_invalid_qc_label_tags_raise_config_error(tmp_path: Path):
    bad = tmp_path / "bad.csv"
    bad.write_text("project,shot,attempt,label,tags\np,s001,1,fail,unknown_tag\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="Unknown failure tag"):
        load_qc_dataset(bad)
