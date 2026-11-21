# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Tests for non-English and multi-language (code-switching) song support."""

from __future__ import annotations

import sys
import types
import unicodedata

import pytest
from pydantic import ValidationError

from mvdirector import api as api_mod
from mvdirector.audio import lyrics
from mvdirector.audio.lyrics import (
    Transcript,
    TranscriptSegment,
    WhisperTranscriber,
    Word,
    align_lines_to_words,
    estimate_missing_timings,
    is_hook_section,
    labels_have_multiple_singers,
    parse_lyrics,
    section_singers,
    tokenize,
)
from mvdirector.audio.prepare import (
    AudioSummary,
    InferredInputsReply,
    infer_inputs_from_transcript,
    lyric_fidelity,
)
from mvdirector.config import PrepareConfig, ProjectConfig
from mvdirector.errors import ConfigError
from mvdirector.generation.plates import plate_slug
from mvdirector.models import (
    Analysis,
    AudioSlice,
    Character,
    Framing,
    LyricLine,
    Manifest,
    SetDesign,
    Shot,
    ShotType,
    TimeSpan,
    VisualBible,
)
from mvdirector.planning import planner, prompts
from mvdirector.planning.planner import (
    BibleReply,
    CharacterReply,
    PropReply,
    SetReply,
    ShotBatchReply,
    WardrobeReply,
    _fix_mismatched_singer_prompt,
    _thin_lipsync,
    build_manifest,
)


def test_unicode_tokenize_handles_all_major_scripts() -> None:
    # Accented Latin, curly apostrophes, and German sharp-s casefolding
    assert tokenize("Corazón, l’amour don't strasse") == [
        "corazón",
        "l'amour",
        "don't",
        "strasse",
    ]
    assert tokenize("Straße") == tokenize("STRASSE")

    # Devanagari and Thai keep combining vowel/tone marks attached to their base letters
    assert tokenize("नमस्ते दुनिया!") == ["नमस्ते", "दुनिया"]
    assert tokenize("สวัสดีครับ") == ["ส", "วั", "ส", "ดี", "ค", "รั", "บ"]

    # Arabic strips tatweel vocal elongation so sung stretched words match written lyrics
    assert tokenize("حبيـــبي يا نور") == ["حبيبي", "يا", "نور"]

    # Cyrillic
    assert tokenize("Привет, ночной город!") == ["привет", "ночной", "город"]

    # Unspaced CJK / Japanese / Korean syllabic tokenization + mixed code-switching
    assert tokenize("Baby 君の eyes に落ちてく 5時") == [
        "baby",
        "君",
        "の",
        "eyes",
        "に",
        "落",
        "ち",
        "て",
        "く",
        "5",
        "時",
    ]
    assert tokenize("밤하늘의 별") == tokenize("밤 하늘의 별")


def test_cjk_korean_indic_and_codeswitching_lines_align_to_words() -> None:
    # Japanese with lenticular section brackets and multi-character Whisper word chunks
    jp_lines = parse_lyrics("【1番Aメロ】\n東京の夜を歩く\n［サビ］\nネオンの雨が降る")
    assert [ln.section for ln in jp_lines] == ["1番Aメロ", "サビ"]
    jp_words = [
        Word(word="東京", start=1.0, end=1.6),
        Word(word="の夜を", start=1.6, end=2.4),
        Word(word="歩く", start=2.4, end=3.1),
        Word(word="ネオンの", start=4.0, end=4.9),
        Word(word="雨が", start=4.9, end=5.5),
        Word(word="降る", start=5.5, end=6.2),
    ]
    jp_aligned = align_lines_to_words(jp_lines, jp_words)
    assert [(ln.start, ln.end, ln.timing_source) for ln in jp_aligned] == [
        (1.0, 3.1, "aligned"),
        (4.0, 6.2, "aligned"),
    ]

    # Korean spacing differences between lyric sheet and Whisper still align cleanly
    ko_lines = parse_lyrics("[1절]\n오늘밤 너만 생각나\n[후렴]\nStay with me 멈추지 마")
    ko_words = [
        Word(word="오늘", start=1.0, end=1.4),
        Word(word="밤", start=1.4, end=1.8),
        Word(word="너만", start=1.9, end=2.4),
        Word(word="생각나", start=2.4, end=3.2),
        Word(word="Stay", start=4.0, end=4.4),
        Word(word="with", start=4.4, end=4.7),
        Word(word="me", start=4.7, end=5.0),
        Word(word="멈추지마", start=5.1, end=6.0),
    ]
    ko_aligned = align_lines_to_words(ko_lines, ko_words)
    assert (ko_aligned[0].start, ko_aligned[0].end) == (1.0, 3.2)
    assert (ko_aligned[1].start, ko_aligned[1].end) == (4.0, 6.0)

    # Hindi (Devanagari combining marks) aligns cleanly
    hi_lines = parse_lyrics("चाँदनी रात में\nतेरा नाम लूँ")
    hi_words = [
        Word(word="चाँदनी", start=0.5, end=1.2),
        Word(word="रात", start=1.2, end=1.6),
        Word(word="में", start=1.6, end=2.0),
        Word(word="तेरा", start=2.5, end=3.0),
        Word(word="नाम", start=3.0, end=3.5),
        Word(word="लूँ", start=3.5, end=4.0),
    ]
    hi_aligned = align_lines_to_words(hi_lines, hi_words)
    assert [(ln.start, ln.end) for ln in hi_aligned] == [(0.5, 2.0), (2.5, 4.0)]


def test_stray_single_cjk_particle_does_not_misanchor_untranscribed_line() -> None:
    lines = parse_lyrics("東京の夜を歩く\n静かな朝の光\nネオンの雨が降る")
    # Middle line was missed by Whisper, except line 3 has particle "の" at 9.0s.
    words = [
        Word(word="東京の夜を歩く", start=1.0, end=3.0),
        Word(word="ネオン", start=8.2, end=8.9),
        Word(word="の", start=9.0, end=9.2),
        Word(word="雨が降る", start=9.2, end=10.5),
    ]
    aligned = align_lines_to_words(lines, words)
    assert aligned[0].timing_source == "aligned" and (aligned[0].start, aligned[0].end) == (
        1.0,
        3.0,
    )
    assert aligned[1].timing_source == "none" and aligned[1].start is None
    assert aligned[2].timing_source == "aligned" and (aligned[2].start, aligned[2].end) == (
        8.2,
        10.5,
    )

    filled = estimate_missing_timings(aligned, [TimeSpan(start=1.0, end=10.5)])
    assert (filled[1].start, filled[1].end, filled[1].timing_source) == (3.0, 8.2, "estimated")


def test_mixed_script_timing_estimation_balances_cjk_and_english_lines() -> None:
    # 8-char Japanese line (~8-12 morae) and 24-char English line (~7 syllables) match in sung time.
    lines = [LyricLine(text="君のことばかり考えて"), LyricLine(text="I keep thinking about you")]
    timed = estimate_missing_timings(lines, [TimeSpan(start=0.0, end=10.0)])
    jp_dur = timed[0].end - timed[0].start  # type: ignore[operator]
    en_dur = timed[1].end - timed[1].start  # type: ignore[operator]
    assert 4.0 <= jp_dur <= 6.5 and 3.5 <= en_dur <= 6.0


def test_non_latin_lyric_fidelity_catches_translation_or_hallucination() -> None:
    summary = AudioSummary(12.0, 120.0, [TimeSpan(start=2.0, end=10.0)], 0.667, [])
    jp_transcript = Transcript(
        segments=[
            TranscriptSegment(text="真夜中の雨の中を歩いて", start=2.0, end=5.5),
            TranscriptSegment(text="君の名前を呼んでいる", start=5.8, end=9.5),
        ],
        words=[],
        language="ja",
    )

    # Model translates Japanese lyrics into English -> fidelity must reject and fall back!
    translated = InferredInputsReply(
        lyrics="[Verse 1]\nWalking through the midnight rain\n\n[Chorus]\nCalling out your name",
        music_style="J-pop synth ballad",
        visual_direction="Rainy Tokyo street at night",
    )
    assert lyric_fidelity(translated.lyrics, jp_transcript) == 0.0
    reply_rejected, _ = infer_inputs_from_transcript(
        jp_transcript,
        summary,
        llm=types.SimpleNamespace(complete_json=lambda **_kw: translated),  # type: ignore[arg-type]
    )
    assert "真夜中の雨の中を歩いて" in reply_rejected.lyrics
    assert "Walking through" not in reply_rejected.lyrics

    # Model hallucinates completely different Japanese lyrics -> also rejected!
    hallucinated_jp = InferredInputsReply(
        lyrics="[Verse 1]\n青い空の下で笑っていた\n\n[Chorus]\n夏の風が吹く海辺",
        music_style="J-pop",
        visual_direction="Sunny beach",
    )
    assert lyric_fidelity(hallucinated_jp.lyrics, jp_transcript) < 0.3

    # Model preserves Japanese lyrics while adding English [Section] headers and punctuation.
    faithful = InferredInputsReply(
        lyrics="[Verse 1]\n真夜中の雨の中を歩いて、\n\n[Chorus]\n君の名前を呼んでいる！",
        music_style="J-pop synth ballad with Japanese female vocals",
        visual_direction="Solo singer under neon umbrellas in rainy Tokyo.",
    )
    seen: dict[str, str] = {}

    class _LLM:
        def complete_json(self, *, system, content, schema):
            seen["system"] = system
            seen["request"] = content[0]["text"]
            return faithful

    jp_multilingual = jp_transcript.model_copy(update={"multilingual": True})
    reply_kept, _ = infer_inputs_from_transcript(jp_multilingual, summary, llm=_LLM())
    assert "[Chorus]\n君の名前を呼んでいる！" in reply_kept.lyrics
    assert "DETECTED LANGUAGE: multilingual (primary: ja)" in seen["request"]
    assert "never translate" in seen["system"].lower()


def test_prepare_config_and_whisper_transcriber_multilingual_settings(
    project, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = PrepareConfig(whisper_language=" JA ", whisper_multilingual=True)
    assert cfg.whisper_language == "ja"
    assert cfg.whisper_multilingual is True

    # English-only *.en Whisper models must be rejected for non-English or multilingual mode.
    with pytest.raises(ValidationError, match="English-only"):
        PrepareConfig(whisper_model="small.en", whisper_language="es")
    with pytest.raises(ValidationError, match="English-only"):
        PrepareConfig(whisper_model="base.en", whisper_multilingual=True)
    assert PrepareConfig(whisper_model="small.en", whisper_language="en").whisper_language == "en"

    with pytest.raises(ConfigError, match="English-only"):
        WhisperTranscriber(model_size="small.en", multilingual=True).transcribe(
            project.path("inputs/vocals.wav")
        )

    captured_kwargs: dict[str, object] = {}

    class FakeWhisperModel:
        def __init__(self, *args, **kwargs):
            pass

        def transcribe(self, audio, **kwargs):
            captured_kwargs.update(kwargs)
            seg = types.SimpleNamespace(
                text="Hola mundo",
                start=1.0,
                end=2.0,
                words=[
                    types.SimpleNamespace(word="Hola", start=1.0, end=1.5),
                    types.SimpleNamespace(word="mundo", start=1.5, end=2.0),
                ],
            )
            return [seg], types.SimpleNamespace(language="es")

    fake_mod = types.ModuleType("faster_whisper")
    fake_mod.WhisperModel = FakeWhisperModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_mod)

    project.config.prepare.whisper_multilingual = True
    words = lyrics.load_or_transcribe_words(project)
    assert captured_kwargs.get("multilingual") is True
    assert [w.word for w in words] == ["Hola", "mundo"]


def test_non_english_solo_sections_never_trigger_false_duet_detection() -> None:
    solo_headers = [
        # Spanish
        ["Introducción", "Verso 1", "Pre-Coro", "Estribillo", "Verso 2", "Puente", "Salida"],
        # French
        ["Intro", "Couplet 1", "Pré-Refrain", "Refrain", "Pont", "Outro"],
        # German
        ["Vorspiel", "Strophe 1", "Refrain", "Brücke", "Nachspiel"],
        # Russian
        ["Интро", "Куплет 1", "Предприпев", "Припев", "Бридж", "Аутро"],
        # Japanese
        ["イントロ", "1番Aメロ", "1番Bメロ", "1番サビ", "間奏", "ラスサビ", "アウトロ"],
        # Chinese
        ["前奏", "第一主歌", "预副歌", "副歌", "间奏", "桥段", "尾奏"],
        # Korean
        ["인트로", "1절", "프리코러스", "후렴", "2절", "브릿지", "아웃트로"],
    ]
    for headers in solo_headers:
        assert not labels_have_multiple_singers(headers), f"False duet on {headers}"


def test_non_english_duet_cues_and_hook_protection_work_across_languages() -> None:
    # Duet cues across languages, including fullwidth colons/brackets and designators
    assert labels_have_multiple_singers(["Verso 1: Hombre", "Verso 2: Mujer"])
    assert labels_have_multiple_singers(["主歌：歌手1", "副歌：歌手2"])
    assert labels_have_multiple_singers(["Aメロ：さくら", "Bメロ：れん"])
    assert labels_have_multiple_singers(["サビ（二人）"])
    assert labels_have_multiple_singers(["副歌：合唱"])
    assert labels_have_multiple_singers(["후렴: 함께"])
    assert labels_have_multiple_singers(["Coro: Elena y Mateo"])
    assert section_singers("主歌：男声") == {"male"}
    assert section_singers("主歌：女声") == {"female"}
    assert section_singers("1절: 가수1") == {"singer 1"}

    # Hook protection recognizes non-English choruses while excluding pre/post-chorus
    for hook in (
        "Estribillo",
        "Coro",
        "Refrão",
        "Refrain",
        "1番サビ",
        "ラスサビ",
        "副歌",
        "후렴",
        "Припев",
    ):
        assert is_hook_section(hook), f"Expected hook for {hook}"
    for non_hook in (
        "Pre-Coro",
        "Post-Coro",
        "プレサビ",
        "预副歌",
        "프리코러스",
        "Предприпев",
        "Verso 1",
    ):
        assert not is_hook_section(non_hook), f"Unexpected hook for {non_hook}"

    # Verify _thin_lipsync protects Japanese [サビ] while thinning [Aメロ] and [プレサビ]
    sections = ["Aメロ", "プレサビ", "サビ", "Bメロ", "ラスサビ", "アウトロ"]
    intensities = [0.2, 0.25, 0.9, 0.3, 0.95, 0.4]
    shots = [
        Shot(
            id=f"s{i + 1:03d}",
            index=i,
            start=i * 4.0,
            end=(i + 1) * 4.0,
            type=ShotType.LIPSYNC,
            framing=Framing.MEDIUM_CLOSE,
            setting="Stage",
            section=sec,
            intensity=val,
            still_prompt="x",
            motion_prompt="y",
            audio_slice=AudioSlice(path="a.wav", start=i * 4.0, end=(i + 1) * 4.0, head_handle=0.0),
            num_frames=97,
        )
        for i, (sec, val) in enumerate(zip(sections, intensities))
    ]
    _thin_lipsync(shots, 0.5)
    by_sec = {s.section: s.type for s in shots}
    assert by_sec["Aメロ"] is ShotType.BROLL
    assert by_sec["プレサビ"] is ShotType.BROLL
    assert by_sec["サビ"] is ShotType.LIPSYNC
    assert by_sec["ラスサビ"] is ShotType.LIPSYNC


def test_non_ascii_ids_character_matching_plates_and_titles() -> None:
    # BibleReply safely converts accented and CJK ids into SAFE_IDs while rejecting path escapes
    bible = BibleReply(
        style="35mm neon",
        palette="cyan and amber",
        lighting="wet neon",
        negative_prompt="blurry",
        characters=[
            CharacterReply(
                id="José",
                name="José García",
                appearance="25-year-old male baritone.",
                wardrobe=[WardrobeReply(id="été", description="Linen shirt.")],
            ),
            CharacterReply(
                id="さくら",
                name="佐藤 さくら",
                appearance="23-year-old female soprano.",
                wardrobe=[WardrobeReply(id="着物", description="Crimson silk kimono.")],
            ),
        ],
        settings=[SetReply(name="渋谷の交差点", description="Rainy crossing.")],
        props=[PropReply(id="光る円盤", name="光る円盤", description="30 cm cyan ring.")],
    )
    assert bible.characters[0].id == "jose"
    assert bible.characters[0].wardrobe[0].id == "ete"
    assert bible.characters[1].id.startswith("char_")
    assert bible.characters[1].wardrobe[0].id.startswith("state_")
    assert bible.props[0].id.startswith("prop_")

    with pytest.raises(ValidationError):
        CharacterReply(id="../escape", appearance="bad")
    with pytest.raises(ValidationError):
        WardrobeReply(id="-flag", description="bad")

    # Manifest character resolution and CJK singer prompt swap fix
    chars = {
        "jose": Character(name="José García", appearance="baritone"),
        "sakura": Character(name="佐藤 さくら", appearance="soprano"),
        "ren": Character(name="れん", appearance="tenor"),
    }
    manifest = Manifest(
        title="東京の夜",
        duration=8.0,
        fps=24,
        characters=chars,
        visual_bible=VisualBible(settings=[SetDesign(name=unicodedata.normalize("NFD", "Café"))]),
    )
    assert manifest.resolve_character_id("Jose Garcia") == "jose"
    assert manifest.resolve_character_id("José") == "jose"
    assert manifest.resolve_character_id("さくら") == "sakura"
    assert manifest.resolve_character_id("佐藤 さくら") == "sakura"
    assert manifest.visual_bible.find_set(unicodedata.normalize("NFC", "café")) is not None

    cjk_shot = Shot(
        id="s001",
        index=0,
        start=0.0,
        end=4.0,
        type=ShotType.LIPSYNC,
        audio_slice=AudioSlice(path="a.wav", start=0.0, end=4.0, head_handle=0.0),
        num_frames=97,
        still_prompt="れんが窓辺で歌う。",
        motion_prompt="れんが静かに歌う。",
    )
    _fix_mismatched_singer_prompt(cjk_shot, "sakura", chars)
    assert "佐藤 さくら" in cjk_shot.still_prompt and "れん" not in cjk_shot.still_prompt
    assert "佐藤 さくら" in cjk_shot.motion_prompt and "れん" not in cjk_shot.motion_prompt

    # Numeric character ids never false-match numbers in prose; spaced Arabic respects word bounds.
    edge_chars = {
        "1": Character(name="", appearance="baritone"),
        "nour": Character(name="نور", appearance="soprano"),
    }
    assert (
        planner._characters_in_text("0-1.5s: 1 spotlight.", edge_chars, proper_noun_only=True) == []
    )
    assert planner._characters_in_text("تنورة طويلة", edge_chars, proper_noun_only=True) == []
    assert planner._characters_in_text("تغني نور في الليل", edge_chars, proper_noun_only=True) == [
        "nour"
    ]

    # Duet planning with Japanese section cues assigns singers accurately
    beats = [round(0.5 * i, 3) for i in range(16)]
    jp_duet_analysis = Analysis(
        duration=8.0,
        tempo_bpm=120.0,
        beats=beats,
        bars=beats[::4],
        bar_intensity=[0.5] * 4,
        vocal_windows=[TimeSpan(start=0.0, end=8.0)],
        lyric_lines=[
            LyricLine(
                text="夜の雨", section="Aメロ：さくら", start=0.0, end=4.0, timing_source="aligned"
            ),
            LyricLine(
                text="朝の光", section="Bメロ：れん", start=4.0, end=8.0, timing_source="aligned"
            ),
        ],
    )

    class JPDuetLLM:
        def complete_json(self, *, system, content, schema):
            if schema is BibleReply:
                return BibleReply(
                    style="film",
                    palette="cool",
                    lighting="neon",
                    negative_prompt="blurry",
                    characters=[
                        CharacterReply(id="sakura", name="さくら", appearance="Female soprano."),
                        CharacterReply(id="ren", name="れん", appearance="Male baritone."),
                    ],
                    settings=[SetReply(name="渋谷の交差点", description="crossing")],
                )
            return ShotBatchReply(shots=[])

    cfg = ProjectConfig()
    cfg.video.max_clip_seconds = 4.0
    cfg.planning.max_lipsync_share = None
    jp_manifest = build_manifest(cfg, jp_duet_analysis, "", "style", JPDuetLLM())
    assert [s.character for s in jp_manifest.shots] == ["sakura", "ren"]

    # Plate slugs stay identical for ASCII sets and collision-free for non-ASCII sets
    assert plate_slug("Neon City") == "neon_city"
    assert plate_slug("渋谷の交差点") != plate_slug("夜の屋上")
    assert plate_slug("Stage 東京") != plate_slug("Stage 大阪")
    assert plate_slug("Café") != plate_slug("Cafe")

    # Output title sanitization preserves combining marks and NFC accents while stripping escapes
    assert api_mod._safe_title("My Song") == "My_Song"
    assert api_mod._safe_title(unicodedata.normalize("NFD", "Café Noir")) == "Café_Noir"
    assert api_mod._safe_title("मेरा गीत") == "मेरा_गीत"
    assert api_mod._safe_title("../../etc/passwd") == "etc_passwd"
    assert api_mod._safe_title("\u0301") == "video"

    # System prompts explicitly steer visual output to English
    assert "in English" in prompts.BIBLE_SYSTEM
    assert "in English" in prompts.SHOTS_SYSTEM
    assert "in English" in prompts.REWRITE_SYSTEM
    assert "in English" in prompts.BRIEF_SYSTEM
    assert "in English" in prompts.INFER_FROM_LYRICS_SYSTEM
    assert planner._is_hook_section("サビ")
