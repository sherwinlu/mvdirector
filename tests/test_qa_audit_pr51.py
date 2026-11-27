# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Regression tests for PR #51 QA, stress, and security audit findings (H1–H3, M1–M6, L1–L4)."""

from __future__ import annotations

import json
import logging
import re
import struct
import zlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from PIL import Image
from pydantic import BaseModel, ValidationError
from typer.testing import CliRunner

from mvdirector import api
from mvdirector.bench.blind_viewer import generate_blind_viewer
from mvdirector.bench.config import (
    BenchmarkConfig,
    GPUCostConfig,
    ModelArm,
    load_benchmark_config,
    lookup_pricing,
)
from mvdirector.bench.dataset import QCLabelItem, load_qc_dataset, scaffold_qc_dataset
from mvdirector.bench.planning import run_planning_arm, run_planning_benchmark
from mvdirector.bench.vision_qc import (
    QCItemVerdict,
    VisionQCArmResult,
    evaluate_hard_gate,
    render_vision_qc_errors_html,
    run_vision_qc_arm,
    run_vision_qc_benchmark,
)
from mvdirector.cli import app
from mvdirector.config import ProjectConfig, QCConfig
from mvdirector.errors import ConfigError, LLMError
from mvdirector.llm.anthropic_backend import _anthropic_usage
from mvdirector.llm.base import ChatBackend, Reply, TokenUsage
from mvdirector.llm.gemini_backend import GeminiChat, _usage_from_data
from mvdirector.llm.openai_backend import OpenAIChat, _openai_usage
from mvdirector.models import (
    Analysis,
    AssetState,
    Attempt,
    Framing,
    LyricLine,
    Manifest,
    QCReport,
    RetryStrategy,
    Shot,
    ShotType,
    TimeSpan,
)
from mvdirector.planning.planner import (
    AudioListeningNotes,
    BibleReply,
    ShotBatchReply,
    _broll_mentions_people,
    build_manifest,
)
from mvdirector.project import Project


def _make_tiny_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), (100, 90, 80)).save(path)


def _make_png_bomb(width: int = 20000, height: int = 20000) -> bytes:
    def chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    idat = zlib.compress(b"\x00" * 16)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")


def _make_shot(idx: int, sid: str, **kw: object) -> Shot:
    start = float(idx * 4.0)
    end = start + 4.0
    return Shot(
        id=sid,
        index=idx,
        start=start,
        end=end,
        audio_slice={"path": "a.wav", "start": start, "end": end, "head_handle": 0.0},
        num_frames=97,
        **kw,
    )


def _setup_project(
    root: Path, *, extra_yaml: dict | None = None, instrumental: bool = False
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "inputs").mkdir(exist_ok=True)
    (root / "inputs" / "lyrics.txt").write_text("[Verse 1]\nLine one\n", encoding="utf-8")
    (root / "inputs" / "style.md").write_text("Moody ballad.", encoding="utf-8")
    cfg: dict = {"title": "Test Song", "llm": {"provider": "ollama", "model": "gemma3:12b"}}
    if extra_yaml:
        cfg.update(extra_yaml)
    (root / "project.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    analysis = Analysis(
        duration=8.0,
        tempo_bpm=90.0,
        beats=[0.0, 2.0, 4.0, 6.0, 8.0],
        bars=[0.0, 4.0, 8.0],
        bar_intensity=[0.4, 0.7],
        vocal_windows=[] if instrumental else [TimeSpan(start=0.0, end=8.0)],
        lyric_lines=(
            []
            if instrumental
            else [LyricLine(text="Line one", section="Verse 1", start=0.0, end=8.0)]
        ),
    )
    (root / "analysis.json").write_text(analysis.model_dump_json(indent=2), encoding="utf-8")
    return root


class _StubPlannerLLM:
    supports_audio = False

    def __init__(self, *, concept: str = "Concept", empty_perf: bool = False) -> None:
        self._concept = concept
        self._empty_perf = empty_perf

    def complete_json(self, *, system: str, content, schema):
        if schema is BibleReply:
            return BibleReply(
                concept=self._concept,
                style="35mm film",
                palette="amber",
                lighting="candlelight",
                negative_prompt="blurry",
                appearance="Singer in coat",
                settings=[{"name": "Hall", "sections": ["Verse 1"], "description": "Dim hall"}],
                rules=["Keep steady"],
            )
        assert schema is ShotBatchReply
        text = content[0]["text"]
        ids = list(dict.fromkeys(re.findall(r"\b(s\d+)\b", text)))
        return ShotBatchReply.model_validate(
            {
                "shots": [
                    {
                        "id": sid,
                        "type": "performance" if self._empty_perf else "lipsync",
                        "framing": "wide" if self._empty_perf else "close_up",
                        "setting": "Hall",
                        "still_prompt": (
                            "Empty rain-slicked marble hallway, completely empty, amber light."
                            if self._empty_perf
                            else f"Close-up of singer {sid} __ENCODED_KEY__"
                        ),
                        "motion_prompt": "Slow push-in.",
                    }
                    for sid in ids
                ]
            }
        )


# ----------------------------------------------------------------------- H1


def test_h1_planning_arm_does_not_persist_expanded_env_secrets(tmp_path: Path):
    proj = _setup_project(
        tmp_path / "song",
        extra_yaml={
            "comfy": {
                "base_url": "https://user:${COMFY_SECRET_PASS}@comfy.internal:8188",
                "headers": {"Authorization": "Bearer ${COMFY_BEARER_TOKEN}"},
                "api_key_comfy_org": "${COMFY_ORG_API_KEY}",
            },
            "qc": {
                "llm": {
                    "provider": "openai_compatible",
                    "model": "vlm-1",
                    "base_url": "https://token:${MVD_QC_TOKEN}@vlm.internal/v1",
                }
            },
        },
    )
    (proj / ".env").write_text(
        "COMFY_SECRET_PASS=super_secret_pw_123\n"
        "COMFY_BEARER_TOKEN=tok_live_secret_456\n"
        "COMFY_ORG_API_KEY=comfy_org_secret_789\n"
        "MVD_QC_TOKEN=qc_secret_000\n",
        encoding="utf-8",
    )
    arm = ModelArm(id="A", label="Arm A", provider="openai", model="gpt-6.2-sol")
    res = run_planning_arm(
        proj, arm, tmp_path / "out", llm_factory=lambda _a, _e: _StubPlannerLLM()
    )
    dumped = (Path(res.output_dir) / "project.yaml").read_text(encoding="utf-8")
    for secret in (
        "super_secret_pw_123",
        "tok_live_secret_456",
        "comfy_org_secret_789",
        "qc_secret_000",
    ):
        assert secret not in dumped
    parsed = yaml.safe_load(dumped)
    assert "headers" not in parsed.get("comfy", {})
    assert "api_key_comfy_org" not in parsed.get("comfy", {})
    assert "${MVD_QC_TOKEN}" in parsed["qc"]["llm"]["base_url"]


# ----------------------------------------------------------------------- H2


def test_h2_blind_viewer_single_pass_replace_and_scoped_storage_key(tmp_path: Path):
    proj = _setup_project(tmp_path / "song")
    arm_a = ModelArm(id="A", label="Model A", provider="openai", model="gpt-6.2-sol")
    arm_b = ModelArm(id="B", label="Model B", provider="gemini", model="gemini-4-neon")
    arm_c = ModelArm(id="C", label="Model C", provider="anthropic", model="claude-opus-5-5")
    res_a = run_planning_arm(
        proj,
        arm_a,
        tmp_path / "out",
        llm_factory=lambda _a, _e: _StubPlannerLLM(concept="Has __ENCODED_KEY__ and __VIEWER_ID__"),
    )
    res_b = run_planning_arm(
        proj, arm_b, tmp_path / "out", llm_factory=lambda _a, _e: _StubPlannerLLM()
    )
    res_c = run_planning_arm(
        proj, arm_c, tmp_path / "out", llm_factory=lambda _a, _e: _StubPlannerLLM()
    )
    html1, key1 = generate_blind_viewer([res_a, res_b, res_c], tmp_path / "v1.html", seed=1)
    html1_same, _ = generate_blind_viewer([res_a, res_b, res_c], tmp_path / "v1b.html", seed=1)
    html2, key2 = generate_blind_viewer([res_a, res_b, res_c], tmp_path / "v2.html", seed=42)
    doc1 = html1.read_text(encoding="utf-8")
    doc1_same = html1_same.read_text(encoding="utf-8")
    doc2 = html2.read_text(encoding="utf-8")

    # __ENCODED_KEY__ inside BLIND_DATA remains intact JSON (no injected double quotes breaking JS).
    m = re.search(r"const BLIND_DATA = (\{.*?\});\n", doc1)
    assert m is not None
    parsed = json.loads(m.group(1))
    concepts = [d["bible"]["concept"] for d in parsed["tracks"][0]["directors"]]
    assert "Has __ENCODED_KEY__ and __VIEWER_ID__" in concepts

    # Distinct shuffles get distinct VIEWER_IDs; identical shuffles keep the same VIEWER_ID.
    assert json.loads(key1.read_text()) != json.loads(key2.read_text())
    id1 = re.search(r'const VIEWER_ID = "([0-9a-f]+)";', doc1)
    id1_same = re.search(r'const VIEWER_ID = "([0-9a-f]+)";', doc1_same)
    id2 = re.search(r'const VIEWER_ID = "([0-9a-f]+)";', doc2)
    assert id1 and id1_same and id2
    assert id1.group(1) == id1_same.group(1)
    assert id1.group(1) != id2.group(1)
    assert "mvd_bench_blind_scores_v1_${VIEWER_ID}" in doc1


# ----------------------------------------------------------------------- H3


def test_h3_planning_cache_invalidation_and_budget_guard(tmp_path: Path):
    proj = _setup_project(tmp_path / "song")
    calls = [0]

    def factory(_a, _e):
        calls[0] += 1
        return _StubPlannerLLM()

    arm_v1 = ModelArm(
        id="A", label="Flash", provider="gemini", model="gemini-3.8-flash", attach_audio=True
    )
    arm_v2 = ModelArm(
        id="A", label="Opus", provider="anthropic", model="claude-opus-5-5", attach_audio=False
    )
    r1 = run_planning_arm(proj, arm_v1, tmp_path / "out", llm_factory=factory)
    assert r1.model == "gemini-3.8-flash" and calls[0] == 1

    # Changing model/provider/attach_audio must invalidate the cache rather than return stale data.
    r2 = run_planning_arm(proj, arm_v2, tmp_path / "out", llm_factory=factory)
    assert r2.model == "claude-opus-5-5" and calls[0] == 2

    # A failed overwrite leaving an older manifest.json must NOT bypass max_budget_usd=0.0.
    class _FailLLM:
        def complete_json(self, **kw):
            raise LLMError("outage")

    err_res = run_planning_arm(
        proj, arm_v2, tmp_path / "out", overwrite=True, llm_factory=lambda _a, _e: _FailLLM()
    )
    assert err_res.status == "error"
    budget_calls = [0]

    def budget_factory(_a, _e):
        budget_calls[0] += 1
        return _StubPlannerLLM()

    run_planning_benchmark(
        [proj],
        BenchmarkConfig(arms=[arm_v2]),
        tmp_path / "out",
        max_budget_usd=0.0,
        llm_factory=budget_factory,
    )
    assert budget_calls[0] == 0


def test_h3_vision_qc_cache_invalidation_preservation_error_retry_and_budget(tmp_path: Path):
    proj = _setup_project(tmp_path / "vc")
    shots = []
    items = []
    for i in range(1, 5):
        rel = f"assets/stills/s{i:02d}_a1.png"
        _make_tiny_png(proj / rel)
        shots.append(
            _make_shot(
                i - 1,
                f"s{i:02d}",
                type=ShotType.BROLL,
                framing=Framing.WIDE,
                still=AssetState(
                    selected_attempt=1,
                    attempts=[
                        Attempt(
                            number=1, strategy=RetryStrategy.INITIAL, seed=1, prompt="p", path=rel
                        )
                    ],
                ),
            )
        )
        items.append(
            QCLabelItem(
                id=f"vc:s{i:02d}:a1", project=str(proj), shot_id=f"s{i:02d}", attempt=1, good=True
            )
        )
    (proj / "manifest.json").write_text(
        Manifest(title="VC", duration=16.0, fps=24, shots=shots).model_dump_json(indent=2),
        encoding="utf-8",
    )

    class _MockJudgeLLM(ChatBackend):
        def __init__(self, *, fail_on: int | None = None, tokens: int = 100) -> None:
            super().__init__(model="m", max_tokens=1000)
            self.calls = 0
            self._fail_on = fail_on
            self._tokens = tokens

        def _send(self, system, messages, schema):
            self.calls += 1
            if self._fail_on and self.calls == self._fail_on:
                raise LLMError("Transient 503 outage")
            if schema.__name__ == "BlindObservation":
                return Reply(
                    json.dumps(
                        {
                            "scene_description": "hall",
                            "visible_people": "none",
                            "framing_observed": "wide",
                            "defects_observed": [],
                        }
                    ),
                    usage=TokenUsage(input_tokens=self._tokens, output_tokens=50),
                )
            return Reply(
                json.dumps(
                    {
                        "spec_match": 0.9,
                        "artifact_free": 0.9,
                        "character_match": None,
                        "issues": [],
                        "fix": "",
                    }
                ),
                usage=TokenUsage(input_tokens=self._tokens, output_tokens=50),
            )

    arm1 = ModelArm(id="A", label="Flash", provider="gemini", model="gemini-3.8-flash")
    llm1 = _MockJudgeLLM()
    run_vision_qc_arm(
        items[:3],
        arm1,
        tmp_path / "out",
        gpu_costs=GPUCostConfig(),
        llm_factory=lambda _a, _e: llm1,
    )
    assert llm1.calls == 3

    # 1. Running a 1-item subset with 1 new item must preserve the 3 prior cached items on disk.
    llm_sub = _MockJudgeLLM()
    run_vision_qc_arm(
        items[3:4],
        arm1,
        tmp_path / "out",
        gpu_costs=GPUCostConfig(),
        llm_factory=lambda _a, _e: llm_sub,
    )
    assert llm_sub.calls == 1
    persisted = json.loads((tmp_path / "out" / "vision_qc" / "A" / "verdicts.json").read_text())
    assert len(persisted) == 4

    # 2. Changing model or blind_observer invalidates stale cache.
    arm2 = ModelArm(id="A", label="Opus", provider="anthropic", model="claude-opus-5-5")
    llm2 = _MockJudgeLLM()
    run_vision_qc_arm(
        items[:2],
        arm2,
        tmp_path / "out",
        gpu_costs=GPUCostConfig(),
        blind_observer=True,
        llm_factory=lambda _a, _e: llm2,
    )
    assert llm2.calls == 4  # 2 items x 2 blind-observer passes

    # 3. Transient LLMError is NOT cached permanently; re-run retries only the failed item.
    arm_err = ModelArm(id="E", label="Err", provider="gemini", model="gemini-3.8-flash")
    run_vision_qc_arm(
        items[:2],
        arm_err,
        tmp_path / "out_err",
        gpu_costs=GPUCostConfig(),
        llm_factory=lambda _a, _e: _MockJudgeLLM(fail_on=2),
    )
    retry_llm = _MockJudgeLLM()
    r_retry = run_vision_qc_arm(
        items[:2],
        arm_err,
        tmp_path / "out_err",
        gpu_costs=GPUCostConfig(),
        llm_factory=lambda _a, _e: retry_llm,
    )
    assert retry_llm.calls == 1 and r_retry.errors == 0

    # 4. Budget cap stops mid-arm instead of evaluating all remaining uncached stills.
    arm_cost = ModelArm(id="D", label="Opus", provider="anthropic", model="claude-opus-5-5")
    cost_llm = _MockJudgeLLM(tokens=100_000)  # ~$0.40 per item on Opus 5.5
    res_budget = run_vision_qc_benchmark(
        items,
        BenchmarkConfig(arms=[arm_cost]),
        tmp_path / "out_budget",
        max_budget_usd=0.10,
        llm_factory=lambda _a, _e: cost_llm,
    )
    assert cost_llm.calls == 1
    assert res_budget[0].evaluated_items == 1


# -------------------------------------------------------------------- M1–M6


def test_m1_llm_check_reports_qc_llm_when_primary_disabled_and_open_qc_llm_respects_vision_flag(
    tmp_path: Path,
):
    proj = _setup_project(
        tmp_path / "qc_only",
        extra_yaml={
            "llm": {"enabled": False},
            "qc": {
                "vision_enabled": True,
                "llm": {
                    "provider": "ollama",
                    "model": "gemma3:12b",
                    "base_url": "http://127.0.0.1:1",
                    "timeout_seconds": 1.0,
                },
            },
        },
    )
    res = CliRunner().invoke(app, ["llm-check", "-p", str(proj)])
    assert res.exit_code == 1
    assert "Vision QC LLM" in res.output

    # When qc.vision_enabled is false, _open_qc_llm yields None instead of opening qc.llm.
    cfg_off = ProjectConfig.model_validate(
        {
            "title": "Off",
            "llm": {"enabled": False},
            "qc": {
                "vision_enabled": False,
                "llm": {"provider": "ollama", "model": "m", "base_url": "http://127.0.0.1:1"},
            },
        }
    )
    with api._open_qc_llm(Project(tmp_path, cfg_off), None) as qc_llm:
        assert qc_llm is None


def test_m2_audio_listen_always_runs_even_for_audio_capable_planner(tmp_path: Path, monkeypatch):
    mix_file = tmp_path / "mix.mp3"
    mix_file.write_bytes(b"fake")
    monkeypatch.setattr(
        "mvdirector.llm.base.audio_block",
        lambda _p: {
            "type": "audio",
            "source": {"type": "base64", "media_type": "audio/mpeg", "data": "AA=="},
        },
    )
    analysis = Analysis(
        duration=8.0,
        tempo_bpm=90.0,
        beats=[0.0, 4.0, 8.0],
        bars=[0.0, 4.0, 8.0],
        bar_intensity=[0.5, 0.5],
        vocal_windows=[TimeSpan(start=0.0, end=8.0)],
        lyric_lines=[LyricLine(text="Hi", section="Verse 1", start=0.0, end=8.0)],
    )

    class _AudioPlanner(_StubPlannerLLM):
        supports_audio = True

    class _SpyListener:
        supports_audio = True

        def __init__(self) -> None:
            self.calls = 0

        def complete_json(self, *, system: str, content, schema):
            assert schema is AudioListeningNotes
            self.calls += 1
            return AudioListeningNotes(meter_and_groove="3/4 waltz")

    cfg = ProjectConfig(title="Always")
    cfg.planning.audio_listen = "always"
    spy = _SpyListener()
    build_manifest(
        cfg, analysis, "[Verse 1]\nHi\n", "Style", _AudioPlanner(), mix_path=mix_file, audio_llm=spy
    )
    assert spy.calls == 1


def test_m3_decompression_bomb_handled_safely_in_error_html(tmp_path: Path):
    bomb = tmp_path / "bomb.png"
    bomb.write_bytes(_make_png_bomb(20000, 20000))
    v = QCItemVerdict(
        item_id="proj:s01:a1",
        project=str(tmp_path),
        shot_id="s01",
        attempt=1,
        still_path=str(bomb),
        ground_truth_good=True,
        vision_spec=0.1,
        vision_artifacts=0.1,
    )
    gate = evaluate_hard_gate(
        [v],
        spec_min=0.6,
        artifacts_min=0.5,
        character_min=0.4,
        mean_llm_usd_per_still=0.0,
        gpu_costs=GPUCostConfig(),
    )
    arm_res = VisionQCArmResult(
        arm_id="A", arm_label="A", provider="gemini", model="m", default_gate=gate, verdicts=[v]
    )
    rendered = render_vision_qc_errors_html([arm_res])
    assert "FP (good rejected)" in rendered
    assert "<img " not in rendered


def test_m4_model_arm_rejects_disallowed_api_key_env_and_wraps_validation_error(tmp_path: Path):
    bad_cfg = tmp_path / "bench.yaml"
    bad_cfg.write_text(
        yaml.safe_dump(
            {
                "arms": [
                    {
                        "id": "X",
                        "label": "BadKey",
                        "provider": "openai",
                        "model": "gpt-6.2-sol",
                        "api_key_env": "AWS_SECRET_ACCESS_KEY",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="not allowed"):
        load_benchmark_config(bad_cfg)

    arm_missing_url = ModelArm(
        id="Y", label="Local", provider="openai_compatible", model="qwen", base_url=None
    )
    with pytest.raises(ConfigError, match="base_url"):
        arm_missing_url.to_llm_config(environ={})


def test_m5_dataset_disambiguates_colliding_project_basenames_and_rejects_nonpositive_size(
    tmp_path: Path,
):
    projects = []
    for sub in ("dir1/song", "dir2/song"):
        p = _setup_project(tmp_path / sub)
        rel = "assets/stills/s01_a1.png"
        _make_tiny_png(p / rel)
        m = Manifest(
            title="Song",
            duration=8.0,
            fps=24,
            shots=[
                _make_shot(
                    0,
                    "s01",
                    type=ShotType.BROLL,
                    framing=Framing.WIDE,
                    still=AssetState(
                        selected_attempt=1,
                        attempts=[
                            Attempt(
                                number=1,
                                strategy=RetryStrategy.INITIAL,
                                seed=1,
                                prompt="p",
                                path=rel,
                                qc=QCReport(passed=True, score=0.9, checks=[]),
                            )
                        ],
                    ),
                )
            ],
        )
        (p / "manifest.json").write_text(m.model_dump_json(indent=2), encoding="utf-8")
        projects.append(p)

    out_csv = tmp_path / "labels.csv"
    scaffolded = scaffold_qc_dataset(projects, out_csv, target_size=10)
    assert len(scaffolded) == 2
    assert len({i.id for i in scaffolded}) == 2
    reloaded = load_qc_dataset(out_csv)
    assert [i.id for i in reloaded] == [i.id for i in scaffolded]

    with pytest.raises(ConfigError, match="target_size"):
        scaffold_qc_dataset(projects, tmp_path / "zero.csv", target_size=0)


def test_m6_openai_stream_usage_refusal_short_circuit_partial_safety_and_hook_isolation():
    class _Tiny(BaseModel):
        x: int

    # 1. OpenAI streaming requests include stream_options.include_usage and fall back on 400.
    seen_payloads: list[dict] = []

    class _FakeHttp:
        def post_sse(self, url, payload, **kw):
            seen_payloads.append(dict(payload))
            if len(seen_payloads) == 1:
                return SimpleNamespace(
                    status_code=400, is_success=False, text="Unknown field: stream_options"
                ), None
            events = [
                json.dumps(
                    {"choices": [{"delta": {"content": '{"x": 7}'}, "finish_reason": "stop"}]}
                ),
                json.dumps({"choices": [], "usage": {"prompt_tokens": 12, "completion_tokens": 5}}),
            ]
            return SimpleNamespace(status_code=200, is_success=True, text=""), events

    oai = OpenAIChat(
        provider="openai",
        model="gpt-6.2-sol",
        max_tokens=100,
        base_url="https://api.openai.com/v1",
        http=_FakeHttp(),  # type: ignore[arg-type]
        json_mode="off",
        token_param="max_completion_tokens",
        key_hint="OPENAI_API_KEY",
        stream=True,
    )
    tel: list = []
    oai.set_telemetry_hook(tel.append)
    res = oai.complete_json(system="s", content=[{"type": "text", "text": "hi"}], schema=_Tiny)
    assert res.x == 7
    assert seen_payloads[0].get("stream_options") == {"include_usage": True}
    assert "stream_options" not in seen_payloads[1]
    assert tel[0].usage == TokenUsage(input_tokens=12, output_tokens=5)

    # 2. Refusal with non-empty prose aborts immediately without a wasted correction retry.
    class _RefusalBackend(ChatBackend):
        def __init__(self) -> None:
            super().__init__(model="claude-opus-5-5", max_tokens=100)
            self.sends = 0

        def _send(self, system, messages, schema):
            self.sends += 1
            return Reply("I cannot assist.", refused=True, finish_reason="refusal")

    rb = _RefusalBackend()
    rb_tel: list = []
    rb.set_telemetry_hook(rb_tel.append)
    with pytest.raises(LLMError, match="refused the request"):
        rb.complete_json(system="s", content=[{"type": "text", "text": "hi"}], schema=_Tiny)
    assert rb.sends == 1 and rb_tel[0].attempts == 1 and rb_tel[0].schema_retries == 0

    # 3. Partial safety blocks set Reply.refused=True on Gemini and OpenAI.
    g_rep = GeminiChat._parse(
        {"candidates": [{"content": {"parts": [{"text": "partial"}]}, "finishReason": "SAFETY"}]}
    )
    assert g_rep.refused is True
    o_rep = oai._parse(
        {"choices": [{"message": {"content": "partial"}, "finish_reason": "content_filter"}]}
    )
    assert o_rep.refused is True

    # 4. Raising telemetry hook never masks validated return values.
    class _OkBackend(ChatBackend):
        def __init__(self) -> None:
            super().__init__(model="m", max_tokens=100)

        def _send(self, system, messages, schema):
            return Reply('{"x": 42}')

    okb = _OkBackend()
    okb.set_telemetry_hook(lambda _t: (_ for _ in ()).throw(RuntimeError("hook boom")))
    assert (
        okb.complete_json(system="s", content=[{"type": "text", "text": "hi"}], schema=_Tiny).x
        == 42
    )


# -------------------------------------------------------------------- L1–L4


def test_l1_planner_warning_collector_captures_info_repairs_at_warning_level(tmp_path: Path):
    proj = _setup_project(tmp_path / "inst", instrumental=True)
    plog = logging.getLogger("mvdirector.planning.planner")
    prev = plog.level
    plog.setLevel(logging.WARNING)
    try:
        res = run_planning_arm(
            proj,
            ModelArm(id="A", label="A", provider="gemini", model="gemini-3.8-flash"),
            tmp_path / "out",
            llm_factory=lambda _a, _e: _StubPlannerLLM(empty_perf=True),
        )
        assert res.quality.code_fixed_shots >= 1
        assert any("converting to broll cutaway" in w for w in res.quality.planner_warnings)
    finally:
        plog.setLevel(prev)


def test_l2_lookup_pricing_avoids_tier_and_numeric_overcharging():
    assert lookup_pricing("claude-opus-4-20250514").input_per_million == 15.0
    assert lookup_pricing("gemini-3.8-flash-preview").input_per_million == 0.30
    assert lookup_pricing("gpt-4o-2024-08-06").input_per_million == 2.50
    for other_tier in (
        "gpt-4o-mini",
        "gpt-5-mini",
        "gemini-2.5-flash-lite",
        "gpt-50",
        "claude-opus-40",
    ):
        assert lookup_pricing(other_tier).input_per_million == 0.0


def test_l3_broll_guardrail_blocks_live_actions_while_keeping_inanimate_and_hands():
    assert _broll_mentions_people(
        "A painted woman stands alone in the hallway, singing softly.", framing=Framing.WIDE
    )
    assert _broll_mentions_people(
        "A wooden man walks across the marble floor.", framing=Framing.MEDIUM
    )
    assert _broll_mentions_people(
        "A woman's hands rest on the piano as she gazes at the camera.", framing=Framing.CLOSE_UP
    )
    assert not _broll_mentions_people(
        "Two carved porcelain figurines on a velvet shelf.", framing=Framing.CLOSE_UP
    )
    assert not _broll_mentions_people(
        "A woman's weathered hands holding a brass pocket watch.", framing=Framing.CLOSE_UP
    )


def test_l4_bool_rejection_and_none_score_gate_consistency():
    with pytest.raises(ValidationError, match="boolean"):
        QCConfig.model_validate({"vision_spec": True})
    assert _openai_usage({"usage": {"prompt_tokens": True}}) is None
    assert _usage_from_data({"usageMetadata": {"promptTokenCount": True}}) is None
    assert (
        _anthropic_usage(
            SimpleNamespace(
                usage=SimpleNamespace(
                    input_tokens=True, output_tokens=False, cache_read_input_tokens=None
                )
            )
        )
        is None
    )

    # vision_spec=None without error is predicted_pass=True in both gate and HTML report.
    v = QCItemVerdict(
        item_id="p:s01:a1",
        project="/tmp",
        shot_id="s01",
        attempt=1,
        still_path="/nonexistent.png",
        ground_truth_good=True,
        vision_spec=None,
        vision_artifacts=None,
    )
    gate = evaluate_hard_gate(
        [v],
        spec_min=0.6,
        artifacts_min=0.5,
        character_min=0.4,
        mean_llm_usd_per_still=0.0,
        gpu_costs=GPUCostConfig(),
    )
    assert gate.fp == 0 and gate.tp == 1
    html_out = render_vision_qc_errors_html(
        [
            VisionQCArmResult(
                arm_id="A",
                arm_label="A",
                provider="gemini",
                model="m",
                default_gate=gate,
                verdicts=[v],
            )
        ]
    )
    assert "FP (good rejected)" not in html_out


# ------------------------------------------------------------------ Round 2


def test_r2_bench_config_recursion_and_bool_rejection(tmp_path: Path):
    deep = tmp_path / "deep.yaml"
    deep.write_text("[" * 2000 + "]" * 2000, encoding="utf-8")
    with pytest.raises(ConfigError, match="Could not read benchmark config"):
        load_benchmark_config(deep)

    with pytest.raises(ValidationError, match="boolean"):
        from mvdirector.bench.config import ModelPricing

        ModelPricing(input_per_million=True, output_per_million=0.0)
    with pytest.raises(ValidationError, match="boolean"):
        GPUCostConfig(still_usd=True)
    with pytest.raises(ValidationError, match="boolean"):
        ModelArm(id="A", label="A", provider="openai", model="m", shots_per_request=True)


def test_r2_duplicate_projects_deduped_and_corrupt_manifest_skipped(tmp_path: Path):
    from mvdirector.bench.blind_viewer import build_blind_viewer_payload
    from mvdirector.bench.planning import PlanningArmResult

    p = _setup_project(tmp_path / "dup")
    rel = "assets/stills/s01_a1.png"
    _make_tiny_png(p / rel)
    m = Manifest(
        title="D",
        duration=8.0,
        fps=24,
        shots=[
            _make_shot(
                0,
                "s01",
                type=ShotType.BROLL,
                framing=Framing.WIDE,
                still=AssetState(
                    selected_attempt=1,
                    attempts=[
                        Attempt(
                            number=1,
                            strategy=RetryStrategy.INITIAL,
                            seed=1,
                            prompt="p",
                            path=rel,
                            qc=QCReport(passed=True, score=0.9, checks=[]),
                        )
                    ],
                ),
            )
        ],
    )
    (p / "manifest.json").write_text(m.model_dump_json(), encoding="utf-8")
    items = scaffold_qc_dataset([p, p], tmp_path / "dup.csv", target_size=10)
    assert len(items) == 1
    assert len(load_qc_dataset(tmp_path / "dup.csv")) == 1

    bad_m = tmp_path / "bad_manifest.json"
    bad_m.write_text("{corrupt", encoding="utf-8")
    with pytest.raises(ConfigError, match="No readable Track 1 manifests"):
        build_blind_viewer_payload(
            [
                PlanningArmResult(
                    track="t",
                    source_project=str(p),
                    arm_id="A",
                    arm_label="A",
                    provider="openai",
                    model="m",
                    attach_audio=False,
                    output_dir=str(tmp_path),
                    manifest_path=str(bad_m),
                    status="ok",
                )
            ]
        )


def test_r2_validate_project_skips_disabled_features_and_planning_arm_isolates_audio_listen(
    tmp_path: Path, monkeypatch
):
    pv = _setup_project(
        tmp_path / "val",
        extra_yaml={
            "llm": {"enabled": False},
            "planning": {
                "audio_listen": "never",
                "audio_llm": {
                    "provider": "gemini",
                    "model": "gemini-3.8-flash",
                    "api_key_env": "MVD_MISSING_AUDIO_KEY",
                },
            },
            "qc": {
                "vision_enabled": False,
                "llm": {
                    "provider": "gemini",
                    "model": "gemini-3.8-flash",
                    "api_key_env": "MVD_MISSING_QC_KEY",
                },
            },
        },
    )
    checks = api.validate_project(pv)
    labels = [c.label for c in checks]
    assert not any("Vision QC LLM" in lbl or "Audio listen LLM" in lbl for lbl in labels)

    # Track 1 arm benchmark isolates planning.audio_listen='always' so Gemini arms don't make an
    # extra AudioListeningNotes pre-pass call.
    p_always = _setup_project(
        tmp_path / "always_src",
        extra_yaml={
            "planning": {"audio_listen": "always"},
            "comfy": {"base_url": "https://user:literal_secret@comfy.internal:8188"},
        },
    )
    (p_always / "inputs" / "mix.wav").write_bytes(b"RIFFfake")
    monkeypatch.setattr(
        "mvdirector.llm.base.audio_block",
        lambda _p: {
            "type": "audio",
            "source": {"type": "base64", "media_type": "audio/mpeg", "data": "AA=="},
        },
    )

    class _AudioArmLLM(_StubPlannerLLM):
        supports_audio = True

        def __init__(self) -> None:
            super().__init__()
            self.schemas: list[str] = []

        def complete_json(self, *, system: str, content, schema):
            self.schemas.append(schema.__name__)
            return super().complete_json(system=system, content=content, schema=schema)

    spy = _AudioArmLLM()
    arm = ModelArm(
        id="A", label="Flash", provider="gemini", model="gemini-3.8-flash", attach_audio=True
    )
    res = run_planning_arm(p_always, arm, tmp_path / "out", llm_factory=lambda _a, _e: spy)
    assert "AudioListeningNotes" not in spy.schemas
    dumped = (Path(res.output_dir) / "project.yaml").read_text(encoding="utf-8")
    assert "literal_secret" not in dumped

    # Failed overwrite removes stale manifest.json from disk.
    assert (Path(res.output_dir) / "manifest.json").is_file()

    class _FailLLM:
        def complete_json(self, **kw):
            raise LLMError("boom")

    err_res = run_planning_arm(
        p_always, arm, tmp_path / "out", overwrite=True, llm_factory=lambda _a, _e: _FailLLM()
    )
    assert err_res.status == "error"
    assert not (Path(res.output_dir) / "manifest.json").exists()
