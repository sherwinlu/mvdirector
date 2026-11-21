# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Tests for track profiling and automatic parameter tuning (`mvdirector.tuning` / `mvd tune`)."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from typer.testing import CliRunner

from mvdirector import api as api_mod
from mvdirector import cli as cli_mod
from mvdirector import tuning
from mvdirector.audio import lyrics as lyrics_mod
from mvdirector.audio.analysis import analyze_project
from mvdirector.audio.prepare import prepare_project
from mvdirector.audio.stems import StemResult
from mvdirector.config import ProjectConfig, load_config
from mvdirector.models import Analysis, LyricLine, ShotType, TimeSpan, TrackProfile
from mvdirector.planning.planner import build_manifest
from mvdirector.project import Project


def test_bilingual_suno_track_strips_negative_tags_and_tunes_whisper_and_drops() -> None:
    style = (
        "Dark emotional vocal trance / edgy dance pop, bilingual French and English, "
        "126–128 BPM, moody breathy female vocals, four-on-the-floor kick, "
        "filtered vocal chops on drops, ‑acoustic, ‑male vocals, ‑trap"
    )
    lyrics = (
        "[Intro – whispered, distant]\n"
        "Je cherche encore\n"
        "Still searching\n\n"
        "[Verse 1 – French, sparse and intimate]\n"
        "Dans la pluie ton ombre reste sur le trottoir\n"
        "Mais mes pas continuent dans la nuit sans toi\n\n"
        "[Chorus]\n"
        "I was searching in the dark for the light inside my heart\n"
        "And every heartbeat pulls me through the night\n\n"
        "[Drop – short instrumental with vocal chops]\n"
    )
    profile = tuning.profile_track(lyrics_text=lyrics, style_text=style)
    assert set(profile.languages) == {"en", "fr"}
    assert profile.multilingual is True
    # Negative Suno tags ("-male vocals", "-trap") must not trigger duet or rap!
    assert profile.vocal_mode == "solo"
    assert profile.rap_or_fast_vocal is False
    assert profile.drop_or_chop_heavy is True
    assert profile.extreme_vocals is True  # whispered intro

    cfg = ProjectConfig()
    prepare_cfg, whisper_adj = tuning.tune_prepare_config(cfg, lyrics_text=lyrics, style_text=style)
    assert prepare_cfg.whisper_multilingual is True
    assert prepare_cfg.whisper_model == "large-v3-turbo"
    assert {a.key for a in whisper_adj} == {
        "prepare.whisper_multilingual",
        "prepare.whisper_model",
    }

    all_adj = {a.key: a.recommended for a in tuning.recommend_adjustments(cfg, profile)}
    assert all_adj["video.dynamic_pacing"] is True
    assert all_adj["video.lipsync_min_vocal_fraction"] == 0.35
    assert all_adj["planning.max_lipsync_share"] == 0.30


def test_monolingual_non_english_lyrics_ignore_english_section_tags() -> None:
    # Pure Spanish lyrics with standard English [Verse 1] / [Chorus] section tags.
    es_lyrics = (
        "[Verse 1]\n"
        "En la noche sin ti mi corazón sigue ardiendo en el fuego\n"
        "Por las calles vacías siempre busco todo lo que fuimos\n\n"
        "[Chorus]\n"
        "Pero cuando vuelves conmigo ya nada duele en la sombra\n"
    )
    profile = tuning.profile_track(lyrics_text=es_lyrics, style_text="Latin pop ballad, 96 BPM")
    assert profile.languages == ["es"]
    assert profile.multilingual is False

    # English-only *.en Whisper model is safely upgraded instead of failing validation.
    cfg = ProjectConfig()
    cfg.prepare.whisper_model = "small.en"
    tuned_prepare, _ = tuning.tune_prepare_config(cfg, lyrics_text=es_lyrics)
    assert tuned_prepare.whisper_language == "es"
    assert tuned_prepare.whisper_multilingual is False
    assert tuned_prepare.whisper_model == "large-v3-turbo"


def test_suno_production_and_language_section_cues_do_not_false_trigger_duet() -> None:
    headers = [
        "Pre‐Chorus",
        "Post‐Chorus",
        "Re‐Build",
        "Bridge: Mandarin, almost whispered",
        "Chorus: English, then Mandarin, piano and cello",
        "Breakdown: Piano, airy pad, whispered vocal",
        "Build: snare roll, riser, vocal chop",
        "Bridge: bare percussion, whispered",
        "Chorus: Explosive, gritty, belting from the chest",
    ]
    assert not lyrics_mod.labels_have_multiple_singers(headers)
    assert lyrics_mod.is_hook_section("Post‐Chorus") is False
    assert lyrics_mod.is_hook_section("Chorus: Explosive, gritty") is True

    # Real lettered / gendered / named duet and 3-singer group headers still work.
    assert lyrics_mod.labels_have_multiple_singers(["Bridge: Voice A", "Bridge: Voice B"])
    assert lyrics_mod.labels_have_multiple_singers(["Verse: Elias", "Chorus: Odalie"])
    group_profile = tuning.profile_track(
        lyrics_text="[Verse 1: Jihoon]\nhey\n[Verse 2: Minji]\nyo\n[Chorus: Hana]\ngo\n"
    )
    assert group_profile.vocal_mode == "group"


def test_audio_syllable_rate_drops_instrumental_and_waltz_tuning(
    project: Project, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 1. Rapid syllable rate triggers rap_or_fast_vocal -> cut_mode='phrases', share=0.35.
    beats = [0.5 * i for i in range(48)]
    bars = beats[::4]
    rap_analysis = Analysis(
        duration=24.0,
        tempo_bpm=120.0,
        beats=beats,
        bars=bars,
        bar_intensity=[0.6] * 12,
        vocal_windows=[TimeSpan(start=2.0, end=10.0)],
        lyric_lines=[
            LyricLine(
                text="running through the static every millisecond spitting fire automatic",
                section="Verse 1",
                start=2.0,
                end=4.5,
                timing_source="aligned",
            ),
            LyricLine(
                text="never gonna hesitate accelerate across the barricade tonight",
                section="Verse 1",
                start=4.5,
                end=7.0,
                timing_source="aligned",
            ),
        ],
    )
    rap_profile = tuning.profile_track(analysis=rap_analysis)
    assert rap_profile.rap_or_fast_vocal is True
    assert rap_profile.syllables_per_second is not None and rap_profile.syllables_per_second >= 4.5

    caller_cfg = ProjectConfig()
    tuned_cfg, applied = tuning.tune_config_for_planning(caller_cfg, rap_analysis)
    assert caller_cfg.video.cut_mode == "bars"  # caller config is never mutated in place
    assert tuned_cfg.video.cut_mode == "phrases"
    assert tuned_cfg.planning.max_lipsync_share == 0.35
    assert applied

    # 2. Acoustic instrumental drops (loud non-vocal peak bars on a vocal song) trigger drop tuning.
    drop_analysis = Analysis(
        duration=32.0,
        tempo_bpm=128.0,
        beats=[0.5 * i for i in range(64)],
        bars=[2.0 * i for i in range(16)],
        bar_intensity=[0.35] * 8 + [0.95] * 4 + [0.40] * 4,
        vocal_windows=[
            TimeSpan(start=0.0, end=16.0)
        ],  # bars 8-11 (16-24s) are loud non-vocal drop!
        lyric_lines=[
            LyricLine(text="holding on tonight", start=2.0, end=6.0, timing_source="aligned")
        ],
    )
    drop_profile = tuning.profile_track(analysis=drop_analysis)
    assert drop_profile.drop_or_chop_heavy is True
    drop_cfg, _ = tuning.tune_config_for_planning(ProjectConfig(), drop_analysis)
    assert drop_cfg.video.dynamic_pacing is True
    assert drop_cfg.video.lipsync_min_vocal_fraction == 0.35
    assert drop_cfg.planning.max_lipsync_share == 0.35

    # 3. Instrumental track sets max_lipsync_share=0.0 -> zero LIPSYNC shots in manifest.
    inst_analysis = Analysis(
        duration=16.0,
        tempo_bpm=120.0,
        beats=[0.5 * i for i in range(32)],
        bars=[2.0 * i for i in range(8)],
        bar_intensity=[0.5] * 8,
        vocal_windows=[],
        lyric_lines=[],
    )
    manifest = build_manifest(
        ProjectConfig(), inst_analysis, "[Instrumental]\n", "Cinematic synthwave", llm=None
    )
    assert all(shot.type is not ShotType.LIPSYNC for shot in manifest.shots)

    # 4. auto_tune=False disables all runtime overrides.
    disabled_cfg = ProjectConfig(auto_tune=False)
    untouched, adj = tuning.tune_config_for_planning(disabled_cfg, rap_analysis)
    assert untouched.video.cut_mode == "bars" and adj == []

    # 5. analyze_project honors 3/4 waltz meter hint and populates analysis.track_profile.
    project.write_text(
        "inputs/style.md", "Music style: Dark cinematic waltz, 3/4, 120 BPM, French\n"
    )
    project.write_text(
        "inputs/lyrics.txt",
        "[Couplet 1]\nDans la nuit je marche sous la pluie sans toi\nToujours dans mon cœur\n",
    )
    captured_whisper: dict[str, object] = {}

    class FakeWhisperModel:
        def __init__(self, model_size: str, **kwargs: object) -> None:
            captured_whisper["model_size"] = model_size

        def transcribe(self, audio: object, **kwargs: object):
            captured_whisper.update(kwargs)
            return [], types.SimpleNamespace(language="fr")

    fake_fw = types.ModuleType("faster_whisper")
    fake_fw.WhisperModel = FakeWhisperModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_fw)

    analyzed = analyze_project(project)
    assert analyzed.track_profile is not None
    assert analyzed.track_profile.beats_per_bar_hint == 3
    assert analyzed.track_profile.languages == ["fr"]
    assert analyzed.bars == analyzed.beats[::3]
    assert captured_whisper["model_size"] == "large-v3-turbo"
    assert captured_whisper["language"] == "fr"


def test_prepare_uses_filename_script_hint_when_inputs_are_starter(
    project: Project, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Starter lyrics.txt and style.md are still untouched, but source audio filename has Hangul+EN.
    sr = 22050
    samples = 0.2 * np.sin(2 * np.pi * 220 * np.arange(sr * 6, dtype=np.float32) / sr)
    song_wav = tmp_path / "Tight Leash (타이트 리쉬).wav"
    sf.write(str(song_wav), samples, sr, subtype="PCM_16")
    # Remove fixture's pre-created mix.wav so prepare ingests song_wav cleanly without --overwrite.
    project.path("inputs/mix.wav").unlink(missing_ok=True)
    # Restore starter templates on project fixture.
    for name in ("lyrics.txt", "style.md"):
        starter = (
            Path(__file__).parent.parent / "src" / "mvdirector" / "templates" / "inputs" / name
        )
        project.write_text(f"inputs/{name}", starter.read_text(encoding="utf-8"))

    seen_whisper: dict[str, object] = {}

    class FakeWhisperModel:
        def __init__(self, model_size: str, **kwargs: object) -> None:
            seen_whisper["model_size"] = model_size

        def transcribe(self, audio: object, **kwargs: object):
            seen_whisper.update(kwargs)
            seg = types.SimpleNamespace(
                text="Tight leash 타이트 리쉬", start=0.5, end=3.0, words=[]
            )
            return [seg], types.SimpleNamespace(language="ko")

    class FakeSeparator:
        def separate(self, mix_path: Path, vocals_out: Path, *, drums_out: Path | None = None):
            sf.write(str(vocals_out), samples, sr, subtype="PCM_16")
            return StemResult(vocals=vocals_out, drums=None)

    fake_fw = types.ModuleType("faster_whisper")
    fake_fw.WhisperModel = FakeWhisperModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_fw)
    project.config.llm.enabled = False

    prepare_project(project, audio=song_wav, separator=FakeSeparator())
    assert seen_whisper["model_size"] == "large-v3-turbo"
    assert seen_whisper["multilingual"] is True


def test_yaml_persistence_preserves_comments_env_vars_and_custom_settings(tmp_path: Path) -> None:
    proj_dir = tmp_path / "song"
    api_mod.init_project(proj_dir)
    yaml_path = proj_dir / "project.yaml"

    # Customize max_lipsync_share to 0.70 so default `mvd tune --apply` preserves it.
    raw_before = yaml_path.read_text(encoding="utf-8").replace(
        "max_lipsync_share: 0.45", "max_lipsync_share: 0.70"
    )
    yaml_path.write_text(raw_before, encoding="utf-8")

    (proj_dir / "inputs" / "style.md").write_text(
        "Music style: Bilingual Korean and English K-pop hip-hop rap, heavy EDM drop, 132 BPM\n",
        encoding="utf-8",
    )
    (proj_dir / "inputs" / "lyrics.txt").write_text(
        "[Verse 1]\n오늘밤 we run the city lights\n[Drop]\n",
        encoding="utf-8",
    )

    runner = CliRunner()
    # Dry-run `mvd tune` reports recommendations without modifying project.yaml.
    dry = runner.invoke(cli_mod.app, ["tune", "-p", str(proj_dir)])
    assert dry.exit_code == 0, dry.output
    assert "Recommended adjustments:" in dry.output
    assert "Kept custom values" in dry.output
    assert yaml_path.read_text(encoding="utf-8") == raw_before

    # `mvd tune --apply` writes default-valued keys while keeping comments, ${COMFY_URL}, and 0.70.
    applied = runner.invoke(cli_mod.app, ["tune", "--apply", "-p", str(proj_dir)])
    assert applied.exit_code == 0, applied.output
    updated_text = yaml_path.read_text(encoding="utf-8")
    assert "${COMFY_URL:-http://127.0.0.1:8188}" in updated_text
    assert "# faster-whisper model size" in updated_text

    loaded = load_config(yaml_path)
    assert loaded.prepare.whisper_multilingual is True
    assert loaded.prepare.whisper_model == "large-v3-turbo"
    assert loaded.video.cut_mode == "phrases"
    assert loaded.video.lipsync_min_vocal_fraction == 0.35
    assert loaded.planning.max_lipsync_share == 0.70  # custom value preserved!

    # `mvd tune --apply --force` overrides the custom 0.70 value too.
    forced = runner.invoke(cli_mod.app, ["tune", "--apply", "--force", "-p", str(proj_dir)])
    assert forced.exit_code == 0, forced.output
    assert load_config(yaml_path).planning.max_lipsync_share == 0.35

    # YAML 1.1 reserved word language code 'no' (Norwegian) is safely quoted, not boolean False!
    no_profile = TrackProfile(languages=["no"], multilingual=False, drop_or_chop_heavy=True)
    no_adj = tuning.recommend_adjustments(ProjectConfig(), no_profile)
    minimal_yaml = tmp_path / "minimal.yaml"
    minimal_yaml.write_text(
        "prepare:\n  whisper_language:\n  whisper_model: small\n"
        "title: Norsk Sang\n"
        "planning: {}  # empty flow mapping\n",
        encoding="utf-8",
    )
    tuning.apply_adjustments_to_yaml(minimal_yaml, no_adj)
    reloaded = load_config(minimal_yaml)
    assert reloaded.prepare.whisper_language == "no"
    assert reloaded.prepare.whisper_model == "large-v3-turbo"
    assert reloaded.planning.max_lipsync_share == 0.35
    assert minimal_yaml.read_text(encoding="utf-8").count("planning:") == 1


def test_cli_analyze_with_tune_flag_persists_profile_and_config(project: Project) -> None:
    project.write_text(
        "inputs/style.md",
        "Music style: Bilingual Spanish and English reggaeton EDM drop, 128 BPM\n",
    )
    project.write_text(
        "inputs/lyrics.txt",
        "[Verse 1]\nEn la noche con fuego\n[Chorus]\nRunning through the dark tonight\n[Drop]\n",
    )
    project.write_text("inputs/word_timestamps.json", "[]\n")
    yaml_path = project.root / "project.yaml"
    raw = (
        yaml_path.read_text(encoding="utf-8")
        .replace("word_timestamps: null", "word_timestamps: inputs/word_timestamps.json")
        .replace("max_lipsync_share: null", "max_lipsync_share: 0.45")
    )
    yaml_path.write_text(raw, encoding="utf-8")

    runner = CliRunner()
    res = runner.invoke(cli_mod.app, ["analyze", "--tune", "-p", str(project.root)])
    assert res.exit_code == 0, res.output
    assert "Updated" in res.output
    updated_cfg = load_config(yaml_path)
    assert updated_cfg.prepare.whisper_multilingual is True
    assert updated_cfg.prepare.whisper_model == "large-v3-turbo"
    assert updated_cfg.video.dynamic_pacing is True
    assert updated_cfg.planning.max_lipsync_share == 0.35
    saved_analysis = project.load_analysis()
    assert saved_analysis.track_profile is not None
    assert saved_analysis.track_profile.multilingual is True
