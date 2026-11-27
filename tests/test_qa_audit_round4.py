# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Regression tests for the round-4 white/black-box QA, stress and security audit."""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from typer.testing import CliRunner

from mvdirector import api
from mvdirector.assemble import ClipPlan, ClipTiming, clip_command, source_frame_index
from mvdirector.bench.config import BenchmarkConfig, GPUCostConfig, ModelArm
from mvdirector.bench.dataset import QCLabelItem
from mvdirector.bench.planning import run_planning_arm
from mvdirector.bench.vision_qc import run_vision_qc_arm, run_vision_qc_benchmark
from mvdirector.cli import app
from mvdirector.dotenv import write_project_dotenv_var
from mvdirector.errors import ConfigError, MVDError, UnknownIdError
from mvdirector.llm.base import ChatBackend, Reply, TokenUsage
from mvdirector.mcp.jobs import JobManager
from mvdirector.models import (
    AssetKind,
    AssetState,
    Attempt,
    ContinuityMode,
    Framing,
    Manifest,
    RetryStrategy,
    ShotType,
    VisualBible,
)
from tests.conftest import needs_ffmpeg
from tests.test_qa_audit_pr51 import (
    _make_shot,
    _make_tiny_png,
    _setup_project,
    _StubPlannerLLM,
)

# --------------------------------------------------------------------------- unknown-id errors


def test_unknown_id_errors_render_without_repr_quotes(tmp_path: Path) -> None:
    """`str(KeyError(msg))` is the repr of msg, which leaked as `Error: "No shot ..."`."""
    manifest = Manifest(
        title="T",
        duration=4.0,
        fps=24,
        visual_bible=VisualBible(concept="c", style="s", palette="p", lighting="l"),
        shots=[
            _make_shot(
                0,
                "s001",
                type=ShotType.BROLL,
                framing=Framing.WIDE,
                still_prompt="p",
                motion_prompt="m",
            )
        ],
    )
    with pytest.raises(KeyError) as info:  # still a KeyError for existing handlers
        manifest.shot("s9\\x")
    assert isinstance(info.value, UnknownIdError)
    assert str(info.value) == "No shot with id 's9\\\\x'"

    with pytest.raises(KeyError) as job_info:
        JobManager().get("job_nope")
    assert str(job_info.value) == "Unknown job_id: 'job_nope'"

    proj = _setup_project(tmp_path / "song")
    (proj / "manifest.json").write_text(manifest.model_dump_json(), encoding="utf-8")
    with pytest.raises(MVDError) as mvd:
        api.reject(proj, "s9999", "note", target=AssetKind.VIDEO)
    assert str(mvd.value) == "No shot with id 's9999'"


# --------------------------------------------------------------------------- bench plan cache


def _counting_factory(calls: list[int]):
    def factory(_arm, _env):
        calls[0] += 1
        return _StubPlannerLLM()

    return factory


def test_planning_cache_invalidates_when_source_inputs_change(tmp_path: Path) -> None:
    """Editing lyrics/style/analysis/planning config must not reuse the stale cached plan."""
    proj = _setup_project(tmp_path / "song")
    out = tmp_path / "out"
    arm = ModelArm(id="A", label="Flash", provider="gemini", model="gemini-3.8-flash")
    calls = [0]
    factory = _counting_factory(calls)

    run_planning_arm(proj, arm, out, llm_factory=factory)
    run_planning_arm(proj, arm, out, llm_factory=factory)
    assert calls[0] == 1, "unchanged inputs should hit the cache"

    (proj / "inputs" / "lyrics.txt").write_text("[Verse 1]\nA new line\n", encoding="utf-8")
    run_planning_arm(proj, arm, out, llm_factory=factory)
    assert calls[0] == 2, "edited lyrics must invalidate the cache"

    (proj / "inputs" / "style.md").write_text("Bright synth-pop.", encoding="utf-8")
    run_planning_arm(proj, arm, out, llm_factory=factory)
    assert calls[0] == 3, "edited style brief must invalidate the cache"

    analysis = json.loads((proj / "analysis.json").read_text(encoding="utf-8"))
    analysis["tempo_bpm"] = 120.0
    (proj / "analysis.json").write_text(json.dumps(analysis), encoding="utf-8")
    run_planning_arm(proj, arm, out, llm_factory=factory)
    assert calls[0] == 4, "re-analysed audio must invalidate the cache"

    cfg = (proj / "project.yaml").read_text(encoding="utf-8")
    (proj / "project.yaml").write_text(cfg + "seed: 99\n", encoding="utf-8")
    run_planning_arm(proj, arm, out, llm_factory=factory)
    assert calls[0] == 5, "changed project config must invalidate the cache"

    run_planning_arm(proj, arm, out, llm_factory=factory)
    assert calls[0] == 5


def test_planning_cache_never_serves_another_projects_plan(tmp_path: Path) -> None:
    """Track slugs are order-dependent basenames; a cache hit must match the source folder."""
    proj_a = _setup_project(tmp_path / "a" / "song")
    proj_b = _setup_project(tmp_path / "b" / "song")
    out = tmp_path / "out"
    arm = ModelArm(id="A", label="Flash", provider="gemini", model="gemini-3.8-flash")
    calls = [0]
    factory = _counting_factory(calls)

    res_a = run_planning_arm(proj_a, arm, out, track_name="song", llm_factory=factory)
    res_b = run_planning_arm(proj_b, arm, out, track_name="song", llm_factory=factory)
    assert calls[0] == 2
    assert res_b.source_project == str(proj_b.resolve()) != res_a.source_project


def test_planning_cache_rejects_results_without_fingerprint(tmp_path: Path) -> None:
    proj = _setup_project(tmp_path / "song")
    out = tmp_path / "out"
    arm = ModelArm(id="A", label="Flash", provider="gemini", model="gemini-3.8-flash")
    calls = [0]
    factory = _counting_factory(calls)
    res = run_planning_arm(proj, arm, out, llm_factory=factory)

    metrics = Path(res.output_dir) / "metrics.json"
    data = json.loads(metrics.read_text(encoding="utf-8"))
    data["config_fingerprint"] = ""
    metrics.write_text(json.dumps(data), encoding="utf-8")
    run_planning_arm(proj, arm, out, llm_factory=factory)
    assert calls[0] == 2


# --------------------------------------------------------------------------- bench qc cache


class _MockJudge(ChatBackend):
    def __init__(self, *, tokens: int = 100) -> None:
        super().__init__(model="m", max_tokens=1000)
        self.calls = 0
        self._tokens = tokens

    def _send(self, system, messages, schema):
        self.calls += 1
        body = {
            "spec_match": 0.9,
            "artifact_free": 0.9,
            "character_match": None,
            "issues": [],
            "fix": "",
        }
        return Reply(
            json.dumps(body), usage=TokenUsage(input_tokens=self._tokens, output_tokens=50)
        )


def _qc_project(root: Path, n: int = 2) -> tuple[Path, list[QCLabelItem]]:
    proj = _setup_project(root)
    shots, items = [], []
    for i in range(1, n + 1):
        rel = f"assets/stills/s{i:02d}_a1.png"
        _make_tiny_png(proj / rel)
        shots.append(
            _make_shot(
                i - 1,
                f"s{i:02d}",
                type=ShotType.BROLL,
                framing=Framing.WIDE,
                still_prompt=f"Empty hall {i}",
                motion_prompt="m",
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
            QCLabelItem(id=f"vc:s{i:02d}:asel", project=str(proj), shot_id=f"s{i:02d}", good=True)
        )
    manifest = Manifest(title="VC", duration=4.0 * n, fps=24, shots=shots)
    (proj / "manifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return proj, items


def test_vision_qc_cache_rejudges_changed_still_or_spec(tmp_path: Path) -> None:
    """A cached verdict belongs to the image and spec judged, not just the shot/attempt id."""
    proj, items = _qc_project(tmp_path / "vc")
    arm = ModelArm(id="A", label="Flash", provider="gemini", model="gemini-3.8-flash")
    out = tmp_path / "out"

    def run() -> int:
        llm = _MockJudge()
        run_vision_qc_arm(items, arm, out, gpu_costs=GPUCostConfig(), llm_factory=lambda *_: llm)
        return llm.calls

    assert run() == 2
    assert run() == 0, "unchanged stills should be served from cache"

    # Regenerated still at the same path: only that item is re-judged.
    Image.new("RGB", (8, 8), (200, 10, 10)).save(proj / "assets/stills/s01_a1.png")
    assert run() == 1

    # Edited shot prompt changes the specification the judge scores against.
    manifest = Manifest.model_validate_json((proj / "manifest.json").read_text(encoding="utf-8"))
    manifest.shots[1].still_prompt = "Crowded neon street"
    (proj / "manifest.json").write_text(manifest.model_dump_json(), encoding="utf-8")
    assert run() == 1
    assert run() == 0

    # Verdicts written before input digests existed cannot prove what they judged.
    verdicts_file = out / "vision_qc" / "A" / "verdicts.json"
    data = json.loads(verdicts_file.read_text(encoding="utf-8"))
    for entry in data:
        entry["input_digest"] = ""
    verdicts_file.write_text(json.dumps(data), encoding="utf-8")
    assert run() == 2


def test_vision_qc_budget_counts_rejudged_stale_items_across_arms(tmp_path: Path) -> None:
    """Re-judging a stale cached id costs money and must count against max_budget_usd."""
    proj, items = _qc_project(tmp_path / "vc", n=1)
    arm_a = ModelArm(id="A", label="Opus", provider="anthropic", model="claude-opus-5-5")
    arm_b = ModelArm(id="B", label="Opus2", provider="anthropic", model="claude-opus-5-5")
    out = tmp_path / "out"
    cfg = BenchmarkConfig(arms=[arm_a, arm_b])
    first = _MockJudge(tokens=100_000)
    run_vision_qc_benchmark(items, cfg, out, arm_ids=["A"], llm_factory=lambda *_: first)
    assert first.calls == 1

    # The still changes, so arm A's cached id is stale and re-judged live (~$0.50 on Opus).
    Image.new("RGB", (8, 8), (10, 200, 10)).save(proj / "assets/stills/s01_a1.png")
    second = _MockJudge(tokens=100_000)
    results = run_vision_qc_benchmark(
        items, cfg, out, max_budget_usd=0.10, llm_factory=lambda *_: second
    )
    assert second.calls == 1, "arm B must be skipped once arm A's live re-judge used the budget"
    assert results[0].live_usd > 0.10


# --------------------------------------------------------------------------- bench plan exit code


def test_bench_plan_exits_nonzero_when_every_arm_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    proj = _setup_project(tmp_path / "song")
    out = tmp_path / "bench"
    runner = CliRunner()

    failed = runner.invoke(app, ["bench", "plan", "-p", str(proj), "--out", str(out), "--arm", "A"])
    assert failed.exit_code == 1
    assert "planning arm(s) failed" in failed.output

    # A zero budget skips the arm: nothing ran, so nothing failed.
    skipped = runner.invoke(
        app,
        ["bench", "plan", "-p", str(proj), "--out", str(out), "--arm", "A", "--budget-usd", "0"],
    )
    assert skipped.exit_code == 0, skipped.output


# --------------------------------------------------------------------------- dotenv FIFO


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs POSIX FIFOs")
def test_dotenv_write_rejects_fifo_instead_of_hanging(tmp_path: Path) -> None:
    os.mkfifo(tmp_path / ".env")
    outcome: dict[str, object] = {}

    def attempt() -> None:
        try:
            write_project_dotenv_var(tmp_path, "OPENAI_API_KEY", "sk-test")
        except ConfigError as exc:
            outcome["error"] = exc

    worker = threading.Thread(target=attempt, daemon=True)
    worker.start()
    worker.join(5.0)
    assert not worker.is_alive(), "opening a FIFO .env blocked waiting for a writer"
    assert "Could not safely read" in str(outcome.get("error"))


# --------------------------------------------------------------------------- review UI markdown


def test_review_ui_escapes_untrusted_markdown() -> None:
    pytest.importorskip("streamlit")
    from mvdirector.review.app import md_text

    lyric = "Hold on ![t](https://tracker.example/p.png) [Chorus](https://x) *sigh* :red[x] $1$"
    escaped = md_text(lyric)
    for token in ("![", "](", "*sigh*", ":red[", "$1$"):
        assert token not in escaped
    # Each escape is a backslash before ASCII punctuation, which CommonMark renders literally.
    assert escaped.replace("\\", "") == lyric
    assert md_text("# Title") == "\\# Title"
    assert md_text("José 💥") == "José 💥"


# --------------------------------------------------------------------------- i2v_fps seam frames


@pytest.mark.parametrize(
    ("in_point", "drop", "played", "src_fps", "expected"),
    [
        # Observed with ffmpeg: a 16 fps source conformed by clip_command to a 24 fps timeline.
        (0.0, True, 0, 16, 0),  # the dropped frame is a timeline frame; source frame 0 repeats
        (0.1, True, 0, 16, 3),
        (0.1, True, 11, 16, 10),
        (0.25, True, 0, 16, 4),
        (0.5, True, 11, 16, 16),
        (0.3125, False, 0, 16, 5),
        # Equal rates and frame-aligned in-points reduce to the plain frame offset.
        (0.25, True, 4, 24, 11),
        # Equal rates, in-point between frames: the next source frame plays first
        # (0.1 s x 24 = frame 2.4 -> 3; round() picked 2).
        (0.1, False, 0, 24, 3),
        (0.1, True, 5, 24, 9),
        (0.137, False, 0, 24, 4),
    ],
)
def test_source_frame_index_matches_ffmpeg_conform(
    in_point: float, drop: bool, played: int, src_fps: int, expected: int
) -> None:
    assert source_frame_index(in_point, drop, played, 24, src_fps) == expected


def test_clip_timing_played_frames_use_conform_mapping() -> None:
    """ShotRunner's handoff frame comes from ClipTiming; it must agree with the shared mapping."""
    timing = ClipTiming(
        in_point=0.1, sync_delay=0.1, drop_first_frame=False, continuity=ContinuityMode.CUT
    )
    assert timing.first_played_frame(24) == 3
    assert timing.last_played_frame(12, 24) == 14
    assert timing.last_played_frame(12, 24, 16.0) == source_frame_index(0.1, False, 11, 24, 16)


@needs_ffmpeg
@pytest.mark.parametrize("src_fps", [16, 24, 25])
def test_source_frame_index_against_real_clip_command(tmp_path: Path, src_fps: int) -> None:
    """Handoff and seam frames must be the frames that actually play after conform."""
    fps = 24
    src = tmp_path / "src.mkv"
    raw = b"".join(np.full((32, 32), 4 * i, np.uint8).tobytes() for i in range(40))
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "gray", "-s", "32x32"]
        + ["-r", str(src_fps), "-i", "-", "-c:v", "ffv1", "-pix_fmt", "gray", str(src)],
        input=raw,
        check=True,
    )
    # Avoid exact half-tick ties (e.g. 0.3125 s at 16 fps): ffmpeg resolves those by the
    # container's timestamp precision, which no closed form can know.
    cases = ((0.0, True), (0.1, True), (0.1, False), (0.137, False), (0.27, True), (0.5, True))
    for in_point, drop in cases:
        plan = ClipPlan(
            "s001",
            12,
            src,
            True,
            in_point,
            drop_first_frame=drop,
            continuity=ContinuityMode.CONTINUE,
        )
        out = tmp_path / "out.mp4"
        subprocess.run(clip_command(plan, out, 32, 32, fps), check=True)
        decoded = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(out), "-f", "rawvideo", "-pix_fmt", "gray", "-"],
            capture_output=True,
            check=True,
        ).stdout
        frames = np.frombuffer(decoded, np.uint8).reshape(-1, 32, 32)
        played = [int(round(f[8:24, 8:24].mean() / 4)) for f in frames]
        expected = [source_frame_index(in_point, drop, j, fps, src_fps) for j in range(12)]
        assert played == expected, (src_fps, in_point, drop)


def test_source_frame_index_is_constant_time_and_total() -> None:
    """A probed rate like 90000 fps (odd timebase) must not turn the lookup into a long scan."""
    started = time.monotonic()
    for _ in range(200):
        source_frame_index(0.1, True, 479, 24, 1_000_000.0)
    assert time.monotonic() - started < 0.5
    assert source_frame_index(0.1, True, 479, 24, 1_000_000.0) == source_frame_index(
        0.1, True, 479, 24, 1_000_000
    )
    # Non-finite or negative in-points degrade to the clip start instead of raising.
    assert source_frame_index(float("nan"), False, 0, 24, 24) == 0
    assert source_frame_index(float("inf"), False, 3, 24, 24) == 3
    assert source_frame_index(-1.0, False, 0, 24, 24) == 0
