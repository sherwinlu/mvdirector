"""Regressions from the QA audit of the large-v3-turbo / non-Latin tuning branch."""

from __future__ import annotations

import json
import math
import random
import threading
from pathlib import Path

import pytest

from mvdirector import media, tuning
from mvdirector.assemble import source_frame_index
from mvdirector.config import ProjectConfig


def _reference_source_frame_index(
    in_point: float, drop: bool, played: int, fps: int, source_fps: float
) -> int:
    """The original unbounded stepping loop the closed form must keep matching."""
    start = max(0.0, in_point)

    def tick(idx: int) -> int:
        return math.floor((idx / source_fps - start) * fps + 0.5)

    first = max(0, math.ceil(start * source_fps - 1e-9))
    target = tick(first) + max(0, played) + (1 if drop else 0)
    idx = first
    while tick(idx + 1) <= target:
        idx += 1
    return idx


def _call_with_deadline(fn, *args, seconds: float = 5.0):  # type: ignore[no-untyped-def]
    result: list[object] = []
    worker = threading.Thread(target=lambda: result.append(fn(*args)), daemon=True)
    worker.start()
    worker.join(seconds)
    assert not worker.is_alive(), f"{fn.__name__}{args} did not return within {seconds}s"
    return result[0]


@pytest.mark.parametrize(
    "args",
    [
        (0.5, False, 0, 24, 1e308),  # absurd probed rate: float can't separate idx from idx + 1
        (1e20, False, 0, 24, 90000.0),
        (1e300, False, 0, 24, 24.0),
        (1e300, False, 0, 24, 1e10),  # in_point * source_fps overflows to inf
    ],
)
def test_source_frame_index_terminates_on_precision_loss(
    args: tuple[float, bool, int, int, float],
) -> None:
    # Previously the settle loops spun forever once in_point * source_fps passed 2**52.
    assert _call_with_deadline(source_frame_index, *args) == 0


def test_source_frame_index_bounded_settle_still_matches_reference() -> None:
    rng = random.Random(20261127)
    for _ in range(3000):
        fps = rng.choice([24, 25, 30, 60])
        source_fps = rng.choice([float(fps), 16.0, 23.976, 29.97, 60.0, rng.uniform(1.0, 120.0)])
        in_point = rng.choice([0.0, rng.random() * 5, rng.randint(0, 50) / 24, rng.random() * 600])
        played = rng.randint(0, 120)
        drop = rng.random() < 0.5
        assert source_frame_index(in_point, drop, played, fps, source_fps) == (
            _reference_source_frame_index(in_point, drop, played, fps, source_fps)
        )
    # Equal-rate exactness at large but legitimate in-points is unchanged.
    assert source_frame_index(1e6, False, 0, 24, 24.0) == 24_000_000
    assert source_frame_index(5.0, False, 10**15, 24, 24.0) == 10**15 + 120


@pytest.mark.parametrize(
    ("rate", "expected"),
    [
        ("24/1", 24.0),
        ("30000/1001", 30000 / 1001),
        ("1000/1", 1000.0),
        ("1001/1", None),  # implausible container rate is reported as unknown
        ("100000000000000000/1", None),
        (f"{10**400}/1", None),  # float(Fraction) overflow used to escape probe()
        ("-24/1", None),
    ],
)
def test_probe_rejects_implausible_or_overflowing_frame_rates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rate: str, expected: float | None
) -> None:
    payload = {
        "format": {"duration": "2.0"},
        "streams": [{"codec_type": "video", "width": 64, "height": 64, "avg_frame_rate": rate}],
    }
    fake_proc = type("P", (), {"returncode": 0, "stdout": json.dumps(payload), "stderr": ""})()
    monkeypatch.setattr("mvdirector.media.require_tool", lambda _: "ffprobe")
    monkeypatch.setattr("subprocess.run", lambda *a, **kw: fake_proc)
    info = media.probe(tmp_path / "clip.mp4")
    assert info.fps == expected
    assert info.duration == 2.0


KOREAN_LYRICS = "[Verse 1]\n사랑해 너를 정말 오늘 밤에\n[Chorus]\n너와 함께 춤을 춰요\n"


def test_legacy_small_whisper_model_still_upgrades_for_non_english() -> None:
    # Projects made from the pre-turbo template carry an explicit `whisper_model: small`;
    # flipping the default must not silently stop their non-English upgrade.
    cfg = ProjectConfig()
    cfg.prepare.whisper_model = "small"
    prepare_cfg, applied = tuning.tune_prepare_config(cfg, lyrics_text=KOREAN_LYRICS)
    assert prepare_cfg.whisper_language == "ko"
    assert prepare_cfg.whisper_model == "large-v3-turbo"
    assert "prepare.whisper_model" in {adj.key for adj in applied}

    # A deliberately chosen non-legacy model is still only recommended, never overwritten.
    cfg.prepare.whisper_model = "medium"
    profile = tuning.profile_track(lyrics_text=KOREAN_LYRICS)
    model_adj = [
        a for a in tuning.recommend_adjustments(cfg, profile) if a.key == "prepare.whisper_model"
    ]
    assert model_adj == []  # 'medium' is not in the upgrade set at all
    cfg.prepare.whisper_model = "base"
    base_adj = {a.key: a for a in tuning.recommend_adjustments(cfg, profile)}
    assert base_adj["prepare.whisper_model"].applied is False


def test_legacy_small_whisper_model_upgrade_persists_to_yaml(tmp_path: Path) -> None:
    yaml_path = tmp_path / "project.yaml"
    yaml_path.write_text(
        "prepare:\n  whisper_model: small           # keep this comment\n"
        "  whisper_language: null\n",
        encoding="utf-8",
    )
    cfg = ProjectConfig()
    cfg.prepare.whisper_model = "small"
    profile = tuning.profile_track(lyrics_text=KOREAN_LYRICS)
    written = tuning.apply_adjustments_to_yaml(
        yaml_path, tuning.recommend_adjustments(cfg, profile)
    )
    assert {"prepare.whisper_model", "prepare.whisper_language"} <= {a.key for a in written}
    text = yaml_path.read_text(encoding="utf-8")
    assert "whisper_model: large-v3-turbo" in text
    assert "keep this comment" in text


@pytest.mark.parametrize(
    ("lyrics", "expected"),
    [
        ("Я люблю тебя и ночь молчит", "ru"),
        ("Я люблю тебе і ніч мовчить, її очі сяють", "uk"),
        ("Ґанок, де ми стояли", "uk"),
        ("Љубав је песма која никад не престаје", "sr"),
        ("Ќе те чекам до утре, ѕвезди сјаат", "mk"),
        ("Менің жүрегім саған қарай ұмтылады", "kk"),
        ("Усё будзе добра, я веру ў цуд", "be"),
        ("Хөх тэнгэрийн дор бүжиглэе", "mn"),  # 'ө/ү' alone used to be read as Kazakh
        ("Сенга бўлган севгим қалбимда", "uz"),  # 'ў' with 'қ' is Uzbek, not Belarusian
        ("Ҳамаи шаб туро меҷӯям", "tg"),
        ("Йырҙар һөйөү тураһында", "ba"),
        ("Җырлыйк бергә, дусларым", "tt"),
    ],
)
def test_cyrillic_lyrics_lock_the_right_whisper_language(lyrics: str, expected: str) -> None:
    # Every Cyrillic lyric used to lock Whisper to Russian, mis-transcribing Ukrainian etc.
    assert tuning.detect_languages(lyrics_text=f"[Verse 1]\n{lyrics}\n") == ([expected], False)
    prepare_cfg, _ = tuning.tune_prepare_config(
        ProjectConfig(), lyrics_text=f"[Verse 1]\n{lyrics}\n"
    )
    assert prepare_cfg.whisper_language == expected


def test_cyrillic_and_english_code_switching_stays_multilingual() -> None:
    langs, multi = tuning.detect_languages(
        lyrics_text="[Verse 1]\nЯ люблю тебе і ніч мовчить\n[Chorus]\n"
        "Dancing through the night with you and never looking back\n"
    )
    assert set(langs) == {"uk", "en"}
    assert multi is True
