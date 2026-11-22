# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Tests for musical-feature analysis, multimodal audio blocks, and song interpretation."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import yaml

from mvdirector import api as api_mod
from mvdirector.audio.analysis import ANALYSIS_SR, analyze_musical_features
from mvdirector.audio.lyrics import Transcript, TranscriptSegment
from mvdirector.audio.prepare import (
    InferredInputsReply,
    fallback_style_brief,
    infer_inputs_from_transcript,
    summarize_audio,
)
from mvdirector.config import ProjectConfig
from mvdirector.errors import ConfigError, LLMError, LLMUnavailableError
from mvdirector.llm.base import (
    AUDIO_TOKEN_ESTIMATE,
    audio_block,
    complete_json_with_optional_audio,
    estimate_tokens,
    maybe_audio_block,
    text_block,
)
from mvdirector.llm.gemini_backend import GeminiChat
from mvdirector.models import (
    Analysis,
    Framing,
    LyricLine,
    MusicalFeatures,
    SectionBeat,
    ShotType,
    SongInterpretation,
    TimeSpan,
    VisualBible,
)
from mvdirector.planning.planner import (
    BibleReply,
    SectionBeatReply,
    SetReply,
    ShotBatchReply,
    ShotCreative,
    SongInterpretationReply,
    build_manifest,
)


def _chord_signal(freqs: list[float], duration: float = 3.0, sr: int = ANALYSIS_SR) -> np.ndarray:
    t = np.arange(int(duration * sr), dtype=np.float32) / sr
    sig = np.zeros_like(t)
    for f in freqs:
        sig += np.sin(2 * np.pi * f * t)
    return (sig / max(len(freqs), 1)).astype(np.float32)


def test_analyze_musical_features_detects_major_and_minor_modes() -> None:
    # C major triad (C4, E4, G4) + C5
    c_major = _chord_signal([261.63, 329.63, 392.00, 523.25])
    mf_maj = analyze_musical_features(c_major, ANALYSIS_SR, tempo_bpm=120.0)
    assert mf_maj.mode == "major"
    assert mf_maj.key == "C major"
    assert "C major" in mf_maj.summary
    assert "~120 BPM" in mf_maj.summary

    # A minor triad (A3, C4, E4) + A4
    a_minor = _chord_signal([220.00, 261.63, 329.63, 440.00])
    mf_min = analyze_musical_features(a_minor, ANALYSIS_SR, tempo_bpm=88.0)
    assert mf_min.mode == "minor"
    assert mf_min.key == "A minor"


def test_analyze_musical_features_distinguishes_brightness_percussiveness_and_dynamics() -> None:
    dark = _chord_signal([110.0, 220.0])
    bright = _chord_signal([4200.0, 5600.0])
    mf_dark = analyze_musical_features(dark, ANALYSIS_SR)
    mf_bright = analyze_musical_features(bright, ANALYSIS_SR)
    assert mf_dark.brightness == "dark"
    assert mf_bright.brightness == "bright"
    assert mf_bright.spectral_centroid_hz > mf_dark.spectral_centroid_hz

    # Sharp percussive impulses vs sustained chord
    n = int(3.0 * ANALYSIS_SR)
    clicks = np.zeros(n, dtype=np.float32)
    rng = np.random.default_rng(42)
    for pos in range(0, n, ANALYSIS_SR // 4):
        end = min(n, pos + 200)
        clicks[pos:end] = rng.standard_normal(end - pos).astype(np.float32) * 0.8
    mf_perc = analyze_musical_features(clicks, ANALYSIS_SR)
    assert mf_perc.percussive_ratio > mf_dark.percussive_ratio

    # Quiet-to-loud swell vs constant amplitude
    t = np.arange(n, dtype=np.float32) / ANALYSIS_SR
    swell = (0.05 + 0.95 * (t / 3.0) ** 3) * np.sin(2 * np.pi * 220.0 * t)
    mf_swell = analyze_musical_features(swell.astype(np.float32), ANALYSIS_SR)
    assert mf_swell.dynamic_range_db > mf_dark.dynamic_range_db + 5.0


@pytest.mark.parametrize(
    "samples",
    [
        np.zeros(0, dtype=np.float32),
        np.zeros(ANALYSIS_SR, dtype=np.float32),
        np.array([np.nan, np.inf, -np.inf, 0.0], dtype=np.float32),
    ],
)
def test_analyze_musical_features_handles_empty_silent_and_non_finite(samples: np.ndarray) -> None:
    mf = analyze_musical_features(samples, ANALYSIS_SR)
    assert mf.key is None
    assert mf.mode is None
    assert mf.summary == ""


def test_audio_block_and_gemini_wire_format(tmp_path: Path) -> None:
    wav = tmp_path / "song.wav"
    sig = _chord_signal([220.0, 261.63, 329.63], duration=2.0)
    sf.write(str(wav), sig, ANALYSIS_SR, subtype="PCM_16")

    block = audio_block(wav, max_seconds=2.0)
    assert block["type"] == "audio"
    assert block["source"]["media_type"] == "audio/mpeg"
    assert block["source"]["data"]

    tokens = estimate_tokens("sys", [{"role": "user", "content": [text_block("hi"), block]}])
    assert tokens >= AUDIO_TOKEN_ESTIMATE

    wire = GeminiChat._wire_contents([{"role": "user", "content": [text_block("listen"), block]}])
    assert wire[0]["parts"][0] == {"text": "listen"}
    assert wire[0]["parts"][1] == {
        "inlineData": {"mimeType": "audio/mpeg", "data": block["source"]["data"]}
    }
    assert GeminiChat.supports_audio is True

    with pytest.raises(LLMError, match="not found"):
        audio_block(tmp_path / "missing.wav")
    with pytest.raises(LLMError, match="exceeds limit"):
        audio_block(wav, max_bytes=16)


def test_maybe_audio_block_respects_backend_capability(tmp_path: Path) -> None:
    wav = tmp_path / "song.wav"
    sf.write(str(wav), _chord_signal([220.0], duration=1.0), ANALYSIS_SR, subtype="PCM_16")

    class _TextOnlyLLM:
        supports_audio = False

        def complete_json(self, *, system, content, schema):
            raise NotImplementedError

    class _AudioLLM:
        supports_audio = True

        def complete_json(self, *, system, content, schema):
            raise NotImplementedError

    assert maybe_audio_block(_TextOnlyLLM(), wav) is None
    attached = maybe_audio_block(_AudioLLM(), wav)
    assert attached is not None and attached["type"] == "audio"
    assert maybe_audio_block(_AudioLLM(), tmp_path / "does-not-exist.wav") is None


def test_prepare_includes_musical_character_and_optional_audio(tmp_path: Path) -> None:
    mix = _chord_signal([220.00, 261.63, 329.63, 440.00], duration=4.0)
    vocals = _chord_signal([440.0], duration=4.0)
    summary = summarize_audio(mix, vocals, beats_per_bar=4)
    assert summary.musical_features is not None
    assert summary.musical_features.key == "A minor"
    assert "A minor" in summary.musical_features.summary

    fallback_style, _ = fallback_style_brief(summary, has_lyrics=True)
    assert "A minor" in fallback_style

    wav = tmp_path / "mix.wav"
    sf.write(str(wav), mix, ANALYSIS_SR, subtype="PCM_16")
    transcript = Transcript(
        segments=[TranscriptSegment(text="shadows on the midnight glass", start=0.5, end=3.5)],
        words=[],
        language="en",
    )

    seen_blocks: list[list[dict]] = []

    class _AudioInferLLM:
        supports_audio = True

        def complete_json(self, *, system, content, schema):
            seen_blocks.append(content)
            return InferredInputsReply(
                lyrics="[Verse 1]\nShadows on the midnight glass",
                music_style="Atmospheric minor-key indie synth ballad",
                visual_direction="Solitary figure behind rain-streaked glass at night",
            )

    _, used_llm = infer_inputs_from_transcript(
        transcript, summary, llm=_AudioInferLLM(), audio_path=wav
    )
    assert used_llm is True
    assert len(seen_blocks) == 1
    assert any(b["type"] == "audio" for b in seen_blocks[0])
    prompt_text = next(b["text"] for b in seen_blocks[0] if b["type"] == "text")
    assert "MUSICAL CHARACTER:" in prompt_text
    assert "A minor" in prompt_text

    # Without audio support, only text block is sent
    seen_blocks.clear()

    class _TextOnlyInferLLM(_AudioInferLLM):
        supports_audio = False

    infer_inputs_from_transcript(transcript, summary, llm=_TextOnlyInferLLM(), audio_path=wav)
    assert all(b["type"] == "text" for b in seen_blocks[0])


def test_visual_bible_find_section_beat_matches_exact_and_singer_tagged() -> None:
    bible = VisualBible(
        interpretation=SongInterpretation(
            meaning="Grief disguised as calm routine.",
            tone="Restrained and haunted.",
            section_beats=[
                SectionBeat(section="Verse 1", emotion="Numb stillness", visual_beat="Empty room"),
                SectionBeat(
                    section="Chorus: Both", emotion="Breaking open", visual_beat="Rain hits glass"
                ),
            ],
        )
    )
    assert bible.find_section_beat("verse 1") is not None
    assert bible.find_section_beat("Verse 1: Elias").emotion == "Numb stillness"  # type: ignore[union-attr]
    assert bible.find_section_beat("Chorus").emotion == "Breaking open"  # type: ignore[union-attr]
    assert bible.find_section_beat("Bridge") is None
    assert bible.find_section_beat(None) is None


def test_planner_passes_interpretation_and_batch_section_arcs(tmp_path: Path) -> None:
    wav = tmp_path / "mix.wav"
    sf.write(str(wav), _chord_signal([220.0, 261.63, 329.63], duration=2.0), ANALYSIS_SR)

    analysis = Analysis(
        duration=16.0,
        tempo_bpm=120.0,
        beats=[0.5 * i for i in range(32)],
        bars=[2.0 * i for i in range(8)],
        bar_intensity=[0.3, 0.3, 0.4, 0.4, 0.9, 0.9, 0.95, 0.95],
        vocal_windows=[TimeSpan(start=0.0, end=16.0)],
        lyric_lines=[
            LyricLine(
                text="quiet room at dawn",
                section="Verse 1",
                start=0.5,
                end=7.5,
                timing_source="aligned",
            ),
            LyricLine(
                text="burn the letters now",
                section="Chorus",
                start=8.5,
                end=15.5,
                timing_source="aligned",
            ),
        ],
        musical_features=MusicalFeatures(
            key="A minor",
            mode="minor",
            brightness="dark",
            percussiveness="melodic",
            dynamic_range="cinematic",
            summary=(
                "A minor, dark timbre, melodic texture, cinematic dynamics "
                "(19.0 dB range), ~120 BPM"
            ),
        ),
    )

    bible_calls: list[list[dict]] = []
    batch_texts: list[str] = []

    class _DirectorLLM:
        supports_audio = True

        def complete_json(self, *, system, content, schema):
            if schema is BibleReply:
                bible_calls.append(content)
                return BibleReply(
                    interpretation=SongInterpretationReply(
                        meaning=(
                            "Letting go of a past relationship by confronting unspoken memories."
                        ),
                        tone="Intimate, aching, and cathartic rather than melodramatic.",
                        music_feel=(
                            "Minor-key swells move from hushed stillness into cathartic release."
                        ),
                        visual_motifs=[
                            "unopened letters",
                            "cold blue dawn turning to ember light",
                        ],
                        section_beats=[
                            SectionBeatReply(
                                section="Verse 1",
                                emotion="Isolated restraint",
                                visual_beat=(
                                    "Singer alone among untouched letters in blue dawn light"
                                ),
                                camera_energy="Still, wide-to-medium, quiet observation",
                            ),
                            SectionBeatReply(
                                section="Chorus",
                                emotion="Cathartic release",
                                visual_beat=(
                                    "Letters catch warm amber light as singer faces the lens"
                                ),
                                camera_energy="Tighter framing, expressive push-in",
                            ),
                        ],
                    ),
                    appearance="A singer in their late 20s with dark hair.",
                    style="35mm film grain, moody naturalism",
                    palette="cold slate blue shifting to warm ember amber",
                    lighting="soft dawn window light shifting to practical tungsten",
                    negative_prompt="glossy CGI",
                    settings=[
                        SetReply(name="bedroom", sections=["Verse 1"], description="quiet bedroom"),
                        SetReply(
                            name="courtyard", sections=["Chorus"], description="dusk courtyard"
                        ),
                    ],
                    concept="A quiet reckoning between memory and release.",
                )
            assert schema is ShotBatchReply
            batch_texts.append(next(b["text"] for b in content if b["type"] == "text"))
            # Parse shot ids from the prompt so we reply for whichever batch was requested.
            shot_ids = [
                line.split("id=", 1)[1].split()[0]
                for line in batch_texts[-1].splitlines()
                if line.startswith("- id=")
            ]
            return ShotBatchReply(
                shots=[
                    ShotCreative(
                        id=sid,
                        type=ShotType.LIPSYNC,
                        framing=Framing.MEDIUM_CLOSE,
                        setting="bedroom" if sid == "s001" else "courtyard",
                        still_prompt=f"Singer in frame for {sid}",
                        motion_prompt=f"Singer sings emotionally in {sid}",
                    )
                    for sid in shot_ids
                ]
            )

    config = ProjectConfig()
    config.llm.shots_per_request = 1  # force 1 shot per batch so we test batch-scoped section arcs
    manifest = build_manifest(
        config,
        analysis,
        "[Verse 1]\nquiet room at dawn\n\n[Chorus]\nburn the letters now\n",
        "Dark minor-key indie ballad with dynamic swells.",
        _DirectorLLM(),
        mix_path=wav,
    )

    assert len(bible_calls) == 1
    assert any(b["type"] == "audio" for b in bible_calls[0])
    bible_text = next(b["text"] for b in bible_calls[0] if b["type"] == "text")
    assert "MUSICAL ANALYSIS:" in bible_text
    assert "A minor, dark timbre" in bible_text

    interp = manifest.visual_bible.interpretation
    assert interp is not None
    assert "unspoken memories" in interp.meaning
    assert len(interp.section_beats) == 2

    assert len(batch_texts) == 2
    assert "SONG INTERPRETATION:" in batch_texts[0]
    assert "unopened letters" in batch_texts[0]
    assert "SECTION ARCS FOR THIS BATCH:" in batch_texts[0]
    assert "Isolated restraint" in batch_texts[0]
    assert "Cathartic release" not in batch_texts[0]
    assert "Cathartic release" in batch_texts[1]
    assert "Isolated restraint" not in batch_texts[1]


def test_planner_falls_back_cleanly_when_interpretation_omitted(analysis: Analysis) -> None:
    batch_texts: list[str] = []

    class _MinimalBibleLLM:
        def complete_json(self, *, system, content, schema):
            if schema is BibleReply:
                return BibleReply(
                    appearance="A singer",
                    style="neon",
                    palette="teal",
                    lighting="hard",
                    negative_prompt="blurry",
                    settings=[SetReply(name="rooftop", description="a rain-slick rooftop")],
                )
            batch_texts.append(content[0]["text"])
            shot_ids = [
                line.split("id=", 1)[1].split()[0]
                for line in batch_texts[-1].splitlines()
                if line.startswith("- id=")
            ]
            return ShotBatchReply(
                shots=[
                    ShotCreative(
                        id=sid,
                        type=ShotType.LIPSYNC,
                        framing=Framing.CLOSE_UP,
                        setting="rooftop",
                        still_prompt="singer on rooftop",
                        motion_prompt="singer sings",
                    )
                    for sid in shot_ids
                ]
            )

    manifest = build_manifest(
        ProjectConfig(),
        analysis,
        "lyrics",
        "Dreamy shoegaze guitars and reverb",
        _MinimalBibleLLM(),
    )
    assert manifest.visual_bible.interpretation is None
    assert batch_texts and "MUSIC STYLE: Dreamy shoegaze guitars and reverb" in batch_texts[0]


def test_analyze_musical_features_handles_short_stereo_extreme_and_long_audio() -> None:
    # Tiny arrays (<256 samples) must return empty features without warnings or bogus keys.
    tiny = analyze_musical_features(np.ones(10, dtype=np.float32) * 0.5, ANALYSIS_SR)
    assert tiny.key is None and tiny.summary == ""

    # 2D stereo arrays are downmixed cleanly instead of crashing.
    mono = _chord_signal([261.63, 329.63, 392.00, 523.25], duration=2.0)
    stereo = np.stack([mono, mono], axis=0)
    mf_stereo = analyze_musical_features(stereo, ANALYSIS_SR)
    assert mf_stereo.key == "C major"

    # Extreme amplitudes (1e30) normalize cleanly with zero overflow.
    extreme = mono * np.float32(1e30)
    mf_extreme = analyze_musical_features(extreme, ANALYSIS_SR)
    assert mf_extreme.key == "C major"

    # Non-positive sr and non-finite tempo do not raise.
    assert analyze_musical_features(mono, sr=0).summary == ""
    assert analyze_musical_features(mono, sr=-22050).summary == ""
    mf_nan_bpm = analyze_musical_features(mono, ANALYSIS_SR, tempo_bpm=float("nan"))
    assert "BPM" not in mf_nan_bpm.summary

    # Long audio (>180s) is bounded via representative window sampling.
    long_mix = np.tile(mono, 110)  # 220s (> 180s cap)
    mf_long = analyze_musical_features(long_mix, ANALYSIS_SR, tempo_bpm=120.0)
    assert mf_long.key == "C major"
    assert "dynamic dynamics" not in mf_long.summary


def test_audio_block_rejects_non_finite_handles_shell_chars_and_concurrency(
    tmp_path: Path,
) -> None:
    sentinel = tmp_path / "pwned"
    tricky = tmp_path / "song; touch pwned $(id).wav"
    sf.write(str(tricky), _chord_signal([220.0], duration=1.0), ANALYSIS_SR, subtype="PCM_16")

    with pytest.raises(LLMError, match="finite"):
        audio_block(tricky, max_seconds=float("nan"))
    with pytest.raises(LLMError, match="finite"):
        audio_block(tricky, max_seconds=float("inf"))

    # Concurrent calls on a filename with shell metacharacters must succeed safely.
    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(lambda _: audio_block(tricky, max_seconds=1.0), range(6)))
    assert not sentinel.exists()
    assert len({r["source"]["data"] for r in results}) == 1


def test_complete_json_with_optional_audio_falls_back_on_provider_rejection(
    tmp_path: Path,
) -> None:
    wav = tmp_path / "song.wav"
    sf.write(str(wav), _chord_signal([220.0], duration=1.0), ANALYSIS_SR, subtype="PCM_16")
    calls: list[list[dict]] = []

    class _RejectAudioLLM:
        supports_audio = True

        def complete_json(self, *, system, content, schema):
            calls.append(content)
            if any(b["type"] == "audio" for b in content):
                raise LLMError("Unsupported audio mimeType")
            return InferredInputsReply(
                lyrics="[Verse]\nhello",
                music_style="indie pop",
                visual_direction="warm indoor performance",
            )

    reply = complete_json_with_optional_audio(
        _RejectAudioLLM(),
        system="sys",
        text="prompt",
        schema=InferredInputsReply,
        audio_path=wav,
    )
    assert reply.music_style == "indie pop"
    assert len(calls) == 2
    assert any(b["type"] == "audio" for b in calls[0])
    assert all(b["type"] == "text" for b in calls[1])

    class _UnavailableLLM:
        supports_audio = True

        def complete_json(self, *, system, content, schema):
            raise LLMUnavailableError("connection timed out")

    with pytest.raises(LLMUnavailableError):
        complete_json_with_optional_audio(
            _UnavailableLLM(),
            system="sys",
            text="prompt",
            schema=InferredInputsReply,
            audio_path=wav,
        )


def test_reply_models_coerce_null_lists_and_strings() -> None:
    batch = ShotBatchReply.model_validate(
        {
            "shots": [
                {
                    "id": "s001",
                    "type": "lipsync",
                    "framing": "close_up",
                    "setting": "bar",
                    "still_prompt": "singer at bar",
                    "motion_prompt": "singer sings",
                    "characters": None,
                    "props": [None, "glass"],
                }
            ]
        }
    )
    assert batch.shots[0].characters == []
    assert batch.shots[0].props == ["glass"]

    bible = BibleReply.model_validate(
        {
            "interpretation": {
                "meaning": None,
                "tone": None,
                "music_feel": None,
                "visual_motifs": ["rain", None],
                "section_beats": [{"section": None, "emotion": None, "visual_beat": None}],
            },
            "style": "noir",
            "palette": "blue",
            "lighting": "dim",
            "negative_prompt": "blur",
            "settings": [{"name": "bar"}],
        }
    )
    assert bible.interpretation is not None
    assert bible.interpretation.meaning == ""
    assert bible.interpretation.visual_motifs == ["rain"]
    assert bible.interpretation.section_beats[0].section == ""


def test_api_plan_rejects_path_traversal_and_symlink_escape_in_mix(
    project, analysis: Analysis, tmp_path: Path
) -> None:
    project.save_analysis(analysis)
    project.write_text("inputs/lyrics.txt", "[Verse 1]\nhello world\n")
    project.write_text("inputs/style.md", "indie rock\n")

    # Path traversal in inputs.mix must be blocked before build_manifest.
    project.config.inputs.mix = "../../etc/passwd"
    (project.root / "project.yaml").write_text(
        yaml.safe_dump(project.config.model_dump(mode="json")), encoding="utf-8"
    )
    with pytest.raises(ConfigError, match="project folder"):
        api_mod.plan(project.root)

    # Symlink escaping the project folder must also be blocked.
    outside = tmp_path / "secret.wav"
    outside.write_bytes(b"RIFF")
    link = project.root / "inputs" / "escape.wav"
    link.symlink_to(outside)
    project.config.inputs.mix = "inputs/escape.wav"
    (project.root / "project.yaml").write_text(
        yaml.safe_dump(project.config.model_dump(mode="json")), encoding="utf-8"
    )
    with pytest.raises(ConfigError, match="project folder"):
        api_mod.plan(project.root)


def test_planner_scales_cleanly_to_large_shot_and_section_counts() -> None:
    num_sections = 20
    bars = [2.0 * i for i in range(num_sections * 4)]
    lyric_lines = [
        LyricLine(
            text=f"line {i}",
            section=f"Section {i + 1}",
            start=i * 8.0 + 0.2,
            end=(i + 1) * 8.0 - 0.2,
            timing_source="aligned",
        )
        for i in range(num_sections)
    ]
    big_analysis = Analysis(
        duration=num_sections * 8.0,
        tempo_bpm=120.0,
        beats=[0.5 * i for i in range(num_sections * 16)],
        bars=bars,
        bar_intensity=[0.6] * len(bars),
        vocal_windows=[TimeSpan(start=0.0, end=num_sections * 8.0)],
        lyric_lines=lyric_lines,
    )
    batch_texts: list[str] = []

    class _ScaleLLM:
        def complete_json(self, *, system, content, schema):
            if schema is BibleReply:
                return BibleReply(
                    interpretation=SongInterpretationReply(
                        meaning="Epic journey.",
                        tone="Sweeping.",
                        music_feel="Steady drive.",
                        section_beats=[
                            SectionBeatReply(
                                section=f"Section {i + 1}",
                                emotion=f"Mood {i + 1}",
                                visual_beat=f"Beat {i + 1}",
                            )
                            for i in range(num_sections)
                        ],
                    ),
                    appearance="A singer",
                    style="film",
                    palette="warm",
                    lighting="soft",
                    negative_prompt="blur",
                    settings=[SetReply(name="stage", description="open stage")],
                )
            text = content[0]["text"]
            batch_texts.append(text)
            shot_ids = [
                line.split("id=", 1)[1].split()[0]
                for line in text.splitlines()
                if line.startswith("- id=")
            ]
            return ShotBatchReply(
                shots=[
                    ShotCreative(
                        id=sid,
                        type=ShotType.LIPSYNC,
                        framing=Framing.MEDIUM_CLOSE,
                        setting="stage",
                        still_prompt="singer",
                        motion_prompt="sings",
                    )
                    for sid in shot_ids
                ]
            )

    cfg = ProjectConfig()
    cfg.video.max_clip_seconds = 2.0  # 80 shots across 20 sections
    cfg.llm.shots_per_request = 4  # 1 section per batch
    manifest = build_manifest(cfg, big_analysis, "lyrics", "rock", _ScaleLLM())
    assert len(manifest.shots) == 80
    assert len(batch_texts) == 20
    # Each batch must contain only its own section's beat, not all 20 sections.
    assert "Mood 1" in batch_texts[0] and "Mood 20" not in batch_texts[0]
    assert "Mood 20" in batch_texts[-1] and "Mood 1 |" not in batch_texts[-1]
