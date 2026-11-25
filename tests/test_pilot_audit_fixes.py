# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Regression tests for pilot audit fixes across lyrics, tuning, config, LLM, and planner."""

from __future__ import annotations

import json

import httpx
import pytest

from mvdirector.audio import lyrics as lyrics_mod
from mvdirector.config import InputsConfig, LLMConfig
from mvdirector.llm.http import HttpJSON
from mvdirector.llm.ollama_backend import OllamaChat
from mvdirector.models import (
    AudioSlice,
    Character,
    Framing,
    Manifest,
    SetDesign,
    Shot,
    ShotType,
    VisualBible,
    WardrobeState,
)
from mvdirector.planning import planner, prompts
from mvdirector.tuning import profile_track


def test_parse_lyrics_keeps_section_headers_over_suno_arrangement_cues():
    text = """
[Build]
[Snare roll accelerating]
[Kick briefly drops out]
One more step and I cross the line

[Drop]
[Massive euphoric trance drop]
[Soaring vocal hook]
Run through the signal

[Main Drop]
[Full power festival energy]
Step into the light

[Outro]
[Drums fade out]
[Atmospheric echo]
Neon overdrive
"""
    parsed = lyrics_mod.parse_lyrics(text)
    assert [ln.section for ln in parsed] == ["Build", "Drop", "Main Drop", "Outro"]


def test_parse_lyrics_extracts_inline_speaker_cues_and_strips_romanization():
    text = """
[Intro]
[all voices] Te perdiste
[all voices] 너는 잃어버렸어 (neo-neun il-eo-beo-ryeoss-eo)
[female voice 1] It's your loss
[Verse 1]
[female voice 2] Mientes, mientes
[female voice 3] 이제 끝이야 (i-je kkeut-i-ya)
"""
    parsed = lyrics_mod.parse_lyrics(text)
    assert [(ln.section, ln.text) for ln in parsed] == [
        ("Intro: all voices", "Te perdiste"),
        ("Intro: all voices", "너는 잃어버렸어 (neo-neun il-eo-beo-ryeoss-eo)"),
        ("Intro: female voice 1", "It's your loss"),
        ("Verse 1: female voice 2", "Mientes, mientes"),
        ("Verse 1: female voice 3", "이제 끝이야 (i-je kkeut-i-ya)"),
    ]
    assert lyrics_mod.section_singers("Intro: all voices") is None
    assert lyrics_mod.section_singers("Intro: female voice 1") == {"female voice 1"}
    assert lyrics_mod.section_singers("Verse 1: female voice 2") == {"female voice 2"}
    assert lyrics_mod.tokenize(parsed[1].text) == ["너", "는", "잃", "어", "버", "렸", "어"]
    # English adlib without non-Latin script is preserved untouched.
    assert lyrics_mod.tokenize("And I tell myself (tell myself)") == [
        "and",
        "i",
        "tell",
        "myself",
        "tell",
        "myself",
    ]


def test_profile_track_does_not_misdetect_french_from_korean_romanization():
    text = """
[Verse 1]
[female voice 1] You said you were working but your phone was on
[female voice 3] 이제 끝이야 (i-je kkeut-i-ya)
[female voice 3] 이제 너야 (i-je neo-ya)
[female voice 3] 이제 다 끝났어 (i-je da kkeut-nass-eo)
[female voice 3] 이제 안 믿어 (i-je an mid-eo)
"""
    profile = profile_track(lyrics_text=text, style_text="K-pop influenced, EDM")
    assert "ko" in profile.languages
    assert "en" in profile.languages
    assert "fr" not in profile.languages


def test_inputs_config_accepts_null_vocals_and_lyrics_and_yaml_off_json_mode():
    inputs = InputsConfig(mix="inputs/mix.wav", vocals=None, lyrics=None)
    assert inputs.vocals is None
    assert inputs.lyrics is None

    llm = LLMConfig(provider="ollama", model="qwen3.8:27b-mlx", json_mode=False)  # type: ignore[arg-type]
    assert llm.json_mode == "off"


def test_bible_reply_coerces_list_invariants_and_string_interpretation():
    reply = planner.BibleReply.model_validate(
        {
            "interpretation": "A song about breaking through pressure.",
            "style": "Cyberpunk cinema",
            "palette": ["Neon cyan", "deep black"],
            "lighting": "High contrast neon",
            "negative_prompt": ["watermark", "blurry"],
            "invariants": ["Silver hoop earrings", "thin silver chain"],
            "settings": [{"name": "Alley", "sections": ["Verse 1"], "description": "Wet alley"}],
            "characters": [
                {
                    "id": "lead_vocal",
                    "name": "Vera",
                    "appearance": "Woman in her mid-twenties.",
                    "invariants": ["Silver hoop earrings in both ears"],
                }
            ],
        }
    )
    assert reply.interpretation is not None
    assert reply.interpretation.meaning == "A song about breaking through pressure."
    assert reply.invariants == "Silver hoop earrings, thin silver chain"
    assert reply.negative_prompt == "watermark, blurry"
    assert reply.characters[0].invariants == "Silver hoop earrings in both ears"


def test_compose_still_prompt_falls_back_on_stale_wardrobe_and_narrative_label():
    char = Character(
        name="Odalie",
        appearance="Woman in her sixties with deep-set grey-blue eyes and fine pores.",
        wardrobe=[WardrobeState(id="dress1", description="Ivory and charcoal Victorian gown.")],
    )
    manifest = Manifest(
        title="Overture",
        duration=30.0,
        fps=24,
        characters={"odalie": char},
        visual_bible=VisualBible(
            style="Gothic Victorian cinema",
            palette="Winter greys",
            lighting="Candlelight",
            settings=[SetDesign(name="Hall", description="Grand hall with antique clocks.")],
        ),
        shots=[
            Shot(
                id="s001",
                index=0,
                start=0.0,
                end=4.0,
                num_frames=97,
                type=ShotType.PERFORMANCE,
                framing=Framing.WIDE,
                audio_slice=AudioSlice(
                    path="assets/audio/s001.wav", start=0.0, end=4.0, head_handle=0.0
                ),
                character="odalie",
                characters=["odalie"],
                wardrobe="dress2",  # stale removed state id
                setting="Hall",
                still_prompt="Odalie stands small in the distant snowy doorway.",
            )
        ],
    )
    composed = prompts.compose_still_prompt(manifest, manifest.shots[0])
    assert "Wearing: Ivory and charcoal Victorian gown." in composed
    assert "The character:" in composed
    assert "The singer" not in composed
    # Wide framing scrubs close-up facial clauses like "eyes".
    assert "grey-blue eyes" not in composed


def test_broll_people_guard_and_missing_shot_uses_bible_setting():
    bible = VisualBible(
        style="Neon",
        palette="Cyan",
        lighting="Low-key",
        settings=[SetDesign(name="Neon Club", description="Empty neon club.")],
    )
    chars = {"anya": Character(name="Anya", appearance="Young woman.")}
    shot_missing = Shot(
        id="s040",
        index=0,
        start=0.0,
        end=4.0,
        num_frames=97,
        type=ShotType.PERFORMANCE,
        framing=Framing.MEDIUM,
        audio_slice=AudioSlice(path="assets/audio/s040.wav", start=0.0, end=4.0, head_handle=0.0),
    )
    planner._apply_creative(shot_missing, None, bible, characters=chars)
    assert shot_missing.setting == "Neon Club"

    shot_broll = Shot(
        id="s067",
        index=1,
        start=4.0,
        end=8.0,
        num_frames=97,
        type=ShotType.BROLL,
        framing=Framing.MEDIUM,
        audio_slice=AudioSlice(path="assets/audio/s067.wav", start=4.0, end=8.0, head_handle=0.0),
    )
    bad_broll = planner.ShotCreative(
        id="s067",
        type=ShotType.BROLL,
        framing=Framing.CLOSE_UP,
        setting="Neon Club",
        still_prompt="Close-up on Anya's face, her eyes filled with sadness.",
        motion_prompt="Slow push-in.",
    )
    planner._apply_creative(shot_broll, bad_broll, bible, characters=chars)
    assert "Anya" not in shot_broll.still_prompt
    assert "no people" in shot_broll.still_prompt.lower()


def test_ollama_chat_sends_think_false_and_falls_back_if_unsupported():
    seen: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        seen.append(body)
        if body.get("think") is False and len(seen) == 1:
            return httpx.Response(400, json={"error": "unknown field 'think'"})
        return httpx.Response(200, json={"message": {"content": '{"meaning": "ok"}'}, "done": True})

    http = HttpJSON(
        headers={},
        timeout=5.0,
        transport=httpx.MockTransport(handler),
        sleep=lambda _: None,
    )
    try:
        chat = OllamaChat(
            model="qwen3.8:27b-mlx",
            max_tokens=512,
            base_url="http://localhost:11434",
            http=http,
            json_mode="off",
        )
        reply = chat._send(
            "sys", [{"role": "user", "content": "hi"}], planner.SongInterpretationReply
        )
        assert reply.text == '{"meaning": "ok"}'
        assert seen[0]["think"] is False
        assert "think" not in seen[1]
    finally:
        http.close()


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Empty neon hallway with rain on glass, no people.", False),
        ("Distant backlit silhouettes beyond neon mist.", False),
        ("Close-up of bartenders pouring drinks at the club bar.", True),
        ("Two figures stand in purple light looking at camera.", True),
    ],
)
def test_broll_mentions_people_distinguishes_allowed_atmosphere(text: str, expected: bool):
    assert planner._broll_mentions_people(text) is expected


def test_tight_framing_scrubs_lower_body_wardrobe_and_prepends_framing_anchor():
    char = Character(
        name="Mira",
        appearance="Woman in her mid-twenties with short burgundy hair.",
        invariants="Silver hoop earrings, sturdy leather boots.",
        wardrobe=[
            WardrobeState(
                id="w1",
                description=(
                    "Oversized charcoal hoodie, black cargo pants with silver chain belt, "
                    "black leather high-top sneakers."
                ),
            )
        ],
    )
    manifest = Manifest(
        title="Test",
        duration=20.0,
        fps=24,
        characters={"mira": char},
        visual_bible=VisualBible(
            style="Neon",
            palette="Purple",
            lighting="Low-key",
            settings=[SetDesign(name="Club", description="Neon club.")],
        ),
        shots=[
            Shot(
                id="s083",
                index=0,
                start=0.0,
                end=4.0,
                num_frames=97,
                type=ShotType.LIPSYNC,
                framing=Framing.CLOSE_UP,
                audio_slice=AudioSlice(
                    path="assets/audio/s083.wav", start=0.0, end=4.0, head_handle=0.0
                ),
                character="mira",
                characters=["mira"],
                wardrobe="w1",
                setting="Club",
                still_prompt="Mira, mouth relaxed, looking slightly off-lens.",
            )
        ],
    )
    composed = prompts.compose_still_prompt(manifest, manifest.shots[0])
    assert composed.startswith("Tight head-and-shoulders close-up portrait")
    assert "Oversized charcoal hoodie" in composed
    assert "cargo pants" not in composed
    assert "sneakers" not in composed
    assert "chain belt" not in composed
    assert "Silver hoop earrings" in composed
    assert "leather boots" not in composed


def test_broll_face_check_ignores_low_confidence_hallucinations_and_skips_rear_view(tmp_path):
    from PIL import Image

    from mvdirector.config import QCConfig, VideoConfig
    from mvdirector.qc.faces import Face
    from mvdirector.qc.image import ImageQC

    class _FakeFaces:
        def __init__(self, result: list[Face]) -> None:
            self._result = result

        def faces(self, _image) -> list[Face]:
            return self._result

    img_path = tmp_path / "still.png"
    Image.new("RGB", (640, 384), (40, 50, 90)).save(img_path)

    manifest = Manifest(
        title="Test",
        duration=20.0,
        fps=24,
        visual_bible=VisualBible(style="Neon", palette="Blue", lighting="Dark"),
        shots=[],
    )
    broll_shot = Shot(
        id="s001",
        index=0,
        start=0.0,
        end=4.0,
        num_frames=97,
        type=ShotType.BROLL,
        framing=Framing.WIDE,
        audio_slice=AudioSlice(path="assets/audio/s001.wav", start=0.0, end=4.0, head_handle=0.0),
        still_prompt="Empty neon alley.",
    )
    # 0.53 confidence glowing-sign hallucination passes no_people.
    qc_low = ImageQC(QCConfig(), VideoConfig(), _FakeFaces([Face(0.15, confidence=0.53)]), None)
    rep_low = qc_low.evaluate(img_path, broll_shot, manifest, None, expected_size=(640, 384))
    assert any(c.name == "no_people" and c.status.value == "passed" for c in rep_low.checks)

    # 0.85 confidence real face still fails no_people.
    qc_high = ImageQC(QCConfig(), VideoConfig(), _FakeFaces([Face(0.15, confidence=0.85)]), None)
    rep_high = qc_high.evaluate(img_path, broll_shot, manifest, None, expected_size=(640, 384))
    assert any(c.name == "no_people" and c.status.value == "failed" for c in rep_high.checks)

    # Rear-view "back to the camera" performance shot skips face_present cleanly.
    assert prompts.is_rear_view_prompt("Nova stands at the center, back to the camera.")
    rear_shot = Shot(
        id="s136",
        index=1,
        start=4.0,
        end=8.0,
        num_frames=97,
        type=ShotType.PERFORMANCE,
        framing=Framing.WIDE,
        audio_slice=AudioSlice(path="assets/audio/s136.wav", start=4.0, end=8.0, head_handle=0.0),
        still_prompt="Nova stands at the center, back to the camera.",
    )
    qc_none = ImageQC(QCConfig(), VideoConfig(), _FakeFaces([]), None)
    rep_rear = qc_none.evaluate(img_path, rear_shot, manifest, None, expected_size=(640, 384))
    assert any(c.name == "face_present" and c.status.value == "skipped" for c in rep_rear.checks)


def test_cluster_two_solo_voices_keeps_same_register_belt_solo_and_splits_duet():
    import numpy as np

    from mvdirector.audio.analysis import _cluster_two_solo_voices

    # Solo female shifting from 301 Hz chest voice to 469 Hz chorus belt with ~0.96 MFCC cosine.
    base_mfcc = np.array([12.0, -4.0, 6.0, -3.0, 2.5, -1.5, 1.0, -0.8, 0.5, -0.4, 0.3, -0.2])
    chest = np.log2([295.0, 300.0, 305.0, 310.0, 298.0, 302.0])
    belt = np.log2([455.0, 465.0, 470.0, 475.0, 460.0, 480.0])
    oct_solo = np.concatenate([chest, belt])
    mfcc_solo = np.vstack(
        [
            np.tile(base_mfcc, (6, 1)) + np.linspace(-0.2, 0.2, 6)[:, None],
            np.tile(base_mfcc * 1.05 + 0.3, (6, 1)) + np.linspace(-0.2, 0.2, 6)[:, None],
        ]
    )
    durs = np.full(12, 3.0)
    assert _cluster_two_solo_voices(oct_solo, mfcc_solo, durs, force_two_voices=False) is None

    # Cross-register male (194 Hz) + female (369 Hz) duet still splits cleanly.
    male = np.log2([185.0, 190.0, 195.0, 200.0, 192.0, 198.0])
    female = np.log2([355.0, 365.0, 370.0, 378.0, 362.0, 375.0])
    oct_duet = np.concatenate([male, female])
    scale = np.array([0.8, 1.4, 0.7, 1.3, 0.6, 1.2, 0.9, 1.1, 0.8, 1.2, 0.9, 1.1])
    mfcc_duet = np.vstack(
        [
            np.tile(base_mfcc, (6, 1)),
            np.tile(base_mfcc * scale, (6, 1)),
        ]
    )
    clustered = _cluster_two_solo_voices(oct_duet, mfcc_duet, durs, force_two_voices=False)
    assert clustered is not None
    assert set(clustered.values()) == {"voice_1", "voice_2"}
    assert all(clustered[i] == "voice_1" for i in range(6))
    assert all(clustered[i] == "voice_2" for i in range(6, 12))


def test_prepare_inference_prompt_does_not_declare_duet_without_voice_2():
    from mvdirector.audio import prepare
    from mvdirector.audio.lyrics import Transcript, TranscriptSegment
    from mvdirector.models import BOTH_VOICES, VocalTurn

    summary = prepare.AudioSummary(
        duration=180.0,
        tempo_bpm=82.0,
        vocal_fraction=0.6,
        vocal_windows=[],
        energy_arc=[],
        vocal_turns=[
            VocalTurn(start=10.0, end=25.0, voice="voice_1"),
            VocalTurn(start=25.0, end=40.0, voice=BOTH_VOICES),
        ],
    )
    transcript = Transcript(
        words=[],
        segments=[
            TranscriptSegment(start=10.0, end=25.0, text="Stay a little longer"),
            TranscriptSegment(start=25.0, end=40.0, text="Hold on to the night"),
        ],
        language="en",
    )
    prompt = prepare._build_inference_prompt(transcript, summary)
    assert "DETECTED VOICES" not in prompt
    assert "(Singer 1)" not in prompt
    assert "(Both)" not in prompt


def test_extract_dir_master_mix_separates_song_titled_mix_from_stems(tmp_path):
    from mvdirector.audio import prepare

    folder = tmp_path / "A Half-Beat Behind Stems"
    folder.mkdir()
    for name in (
        "0 Lead Vocals.wav",
        "1 Backing Vocals.wav",
        "2 Drums.wav",
        "3 Bass.wav",
        "4 Guitar.wav",
        "A Half-Beat Behind.wav",
    ):
        (folder / name).write_bytes(b"RIFF")

    mix, rest = prepare._extract_dir_master_mix(folder, "A Half-Beat Behind")
    assert mix == folder / "A Half-Beat Behind.wav"
    assert len(rest) == 5
    assert mix not in rest


def test_stem_summary_formats_round_minute_transitions_without_2_60():
    from mvdirector.audio.stem_pack import _format_arrangement_summary
    from mvdirector.models import StemTrackInfo, TimeSpan

    tracks = [
        StemTrackInfo(
            id="drums",
            name="Drums",
            role="drums",
            path="inputs/stems/drums.wav",
            active_spans=[TimeSpan(start=0.0, end=180.0)],
            active_fraction=0.8,
        )
    ]
    summary = _format_arrangement_summary(tracks, [], [153.2, 179.7], duration=218.0)
    assert "2:60" not in summary
    assert "3:00" in summary


def test_voice_tag_map_ignores_solo_voice_with_harmony_both():
    from mvdirector.audio.lyrics import segment_speaker_label
    from mvdirector.models import BOTH_VOICES, VocalTurn

    turns = [
        VocalTurn(start=0.0, end=10.0, voice="voice_1"),
        VocalTurn(start=10.0, end=20.0, voice=BOTH_VOICES),
    ]
    assert segment_speaker_label(1.0, 5.0, turns) is None
    assert segment_speaker_label(11.0, 15.0, turns) is None


def test_describe_vocal_register_classifies_solo_and_duet_pitch():
    from mvdirector.audio import prepare
    from mvdirector.audio.analysis import describe_vocal_register
    from mvdirector.audio.lyrics import Transcript, TranscriptSegment
    from mvdirector.models import VocalTurn

    low_hint = describe_vocal_register(
        [VocalTurn(start=0.0, end=5.0, voice="voice_1", median_f0_hz=140.3)]
    )
    assert low_hint is not None and "male" in low_hint.lower() and "140 Hz" in low_hint

    high_hint = describe_vocal_register([], solo_f0_hz=320.0)
    assert high_hint is not None and "female" in high_hint.lower() and "320 Hz" in high_hint

    duet_hint = describe_vocal_register(
        [
            VocalTurn(start=0.0, end=5.0, voice="voice_1", median_f0_hz=193.7),
            VocalTurn(start=5.0, end=10.0, voice="voice_2", median_f0_hz=368.6),
        ]
    )
    assert duet_hint is not None and "2 vocal registers" in duet_hint

    summary = prepare.AudioSummary(
        duration=180.0,
        tempo_bpm=81.0,
        vocal_fraction=0.6,
        vocal_windows=[],
        energy_arc=[],
        vocal_pitch_hz=140.3,
    )
    prompt = prepare._build_inference_prompt(
        Transcript(
            words=[],
            segments=[TranscriptSegment(start=1.0, end=5.0, text="Stay")],
            language="en",
        ),
        summary,
    )
    assert "DETECTED VOCAL REGISTER:" in prompt
    assert "male" in prompt.lower()


def test_broll_mentions_people_rejects_tight_shadowy_figures():
    text = "A medium close-up of a single, shadowy 'angel' figure, silhouetted against fire."
    assert planner._broll_mentions_people(text, framing=Framing.MEDIUM_CLOSE) is True
    assert (
        planner._broll_mentions_people(
            "Empty misty pine forest with embers, no people.", framing=Framing.MEDIUM_CLOSE
        )
        is False
    )
    solo_cast = {"_solo": Character(name="The Keeper")}
    assert (
        planner._broll_mentions_people(
            "A medium shot of The Keeper, framed from the waist up.",
            solo_cast,
            framing=Framing.MEDIUM,
        )
        is True
    )


def test_single_character_batch_prompt_includes_character_name_and_appearance():
    seen: list[str] = []

    class _CaptureLLM:
        def complete_json(self, *, system, content, schema):
            seen.append(content[0]["text"])
            return planner.ShotBatchReply(shots=[])

    manifest = Manifest(
        title="Stay a Little",
        duration=20.0,
        fps=24,
        character=Character(
            name="Arthur",
            appearance="A man in his late 30s with a weathered jaw.",
            wardrobe=[WardrobeState(id="coat", description="Long charcoal wool coat.")],
        ),
        visual_bible=VisualBible(
            style="Dark",
            palette="Slate",
            lighting="Low-key",
            settings=[SetDesign(name="Library", description="Dim circular library.")],
        ),
        shots=[
            Shot(
                id="s001",
                index=0,
                start=0.0,
                end=4.0,
                num_frames=97,
                type=ShotType.LIPSYNC,
                framing=Framing.MEDIUM_CLOSE,
                audio_slice=AudioSlice(
                    path="assets/audio/s001.wav", start=0.0, end=4.0, head_handle=0.0
                ),
            )
        ],
    )
    planner._ask_for_batch(
        manifest,
        manifest.shots,
        0,
        "coat",
        None,
        _CaptureLLM(),  # type: ignore[arg-type]
    )
    assert seen and "Arthur" in seen[0] and "weathered jaw" in seen[0]


def test_chat_backend_accepts_bare_json_array_for_shot_batch_reply():
    from mvdirector.llm.base import ChatBackend, Reply

    class _ArrayChat(ChatBackend):
        def _send(self, system, messages, schema):
            return Reply(
                text='[{"id": "s016", "type": "lipsync", "framing": "medium_close", '
                '"setting": "Stage", "still_prompt": "Close portrait.", "motion_prompt": "Sings."}]'
            )

    chat = _ArrayChat(model="test", max_tokens=1024)
    result = chat.complete_json(
        system="sys", content=[{"type": "text", "text": "go"}], schema=planner.ShotBatchReply
    )
    assert len(result.shots) == 1
    assert result.shots[0].id == "s016"


def test_closeup_compresses_setting_and_scrubs_barefoot_mid_calf_cleanly():
    char = Character(
        name="The Caretaker",
        appearance="Woman in her thirties with dark hair.",
        wardrobe=[
            WardrobeState(
                id="w1",
                description=(
                    "A sleeveless V-neck mid-calf dress of charred black fibers. "
                    "She remains barefoot with no jewelry or accessories. "
                    "She wears simple, flat leather shoes."
                ),
            )
        ],
    )
    manifest = Manifest(
        title="Sugar and Ash",
        duration=20.0,
        fps=24,
        character=char,
        visual_bible=VisualBible(
            style="Dark",
            palette="Amber",
            lighting="Low-key",
            settings=[
                SetDesign(
                    name="The Circular Stage",
                    description=(
                        "An open-air scorched pine stage in a dark forest. "
                        "The floor is covered in breadcrumbs and collapsed walls."
                    ),
                )
            ],
        ),
        shots=[
            Shot(
                id="s037",
                index=0,
                start=0.0,
                end=4.0,
                num_frames=97,
                type=ShotType.LIPSYNC,
                framing=Framing.CLOSE_UP,
                audio_slice=AudioSlice(
                    path="assets/audio/s037.wav", start=0.0, end=4.0, head_handle=0.0
                ),
                wardrobe="w1",
                setting="The Circular Stage",
                still_prompt="Close-up of The Caretaker looking into the lens.",
            )
        ],
    )
    composed = prompts.compose_still_prompt(manifest, manifest.shots[0])
    assert "tight close-up; soft-focus background only" in composed
    assert "An open-air scorched pine stage in a dark forest" in composed
    assert "floor is covered in breadcrumbs" not in composed
    assert "barefoot" not in composed.lower()
    assert "mid-calf" not in composed.lower()
    assert "She wears simple" not in composed
    assert "sleeveless V-neck dress" in composed


def test_rule_based_broll_avoids_duplicate_the_article():
    shot = Shot(
        id="s019",
        index=0,
        start=0.0,
        end=4.0,
        num_frames=97,
        type=ShotType.BROLL,
        framing=Framing.MEDIUM_CLOSE,
        setting="The Circular Stage",
        audio_slice=AudioSlice(path="assets/audio/s019.wav", start=0.0, end=4.0, head_handle=0.0),
    )
    planner._apply_rule_based(shot)
    assert "of the The " not in shot.still_prompt
    assert "of The Circular Stage" in shot.still_prompt


def test_rear_view_panel_routing_prefers_back_then_profile_and_avoids_face(tmp_path):
    from mvdirector.comfy.workflow import WorkflowTemplate
    from mvdirector.config import ProjectConfig
    from mvdirector.generation.runner import ShotRunner
    from mvdirector.models import ReferenceSet
    from mvdirector.project import Project

    proj_dir = tmp_path / "rear_proj"
    ref_dir = proj_dir / "assets" / "reference"
    ref_dir.mkdir(parents=True)
    for name in ("front.png", "profile.png", "back.png", "face.png"):
        (ref_dir / name).write_bytes(b"PNG")
    cfg = ProjectConfig()
    cfg.reference.closeup_face_panel = True
    proj = Project(root=proj_dir, config=cfg)

    shot = Shot(
        id="s136",
        index=0,
        start=0.0,
        end=4.0,
        num_frames=97,
        type=ShotType.PERFORMANCE,
        framing=Framing.CLOSE_UP,
        wardrobe="w1",
        audio_slice=AudioSlice(path="assets/audio/s136.wav", start=0.0, end=4.0, head_handle=0.0),
        still_prompt="Nova stands at the center, back to the camera.",
    )
    manifest = Manifest(
        title="Rear Test",
        duration=20.0,
        fps=24,
        character=Character(
            name="Nova",
            appearance="Woman with short dark hair.",
            wardrobe=[WardrobeState(id="w1", description="Black bodysuit.")],
            references={
                "w1": ReferenceSet(
                    sheet="assets/reference/front.png",
                    panels={
                        "front": "assets/reference/front.png",
                        "profile": "assets/reference/profile.png",
                        "back": "assets/reference/back.png",
                        "face": "assets/reference/face.png",
                    },
                )
            },
        ),
        visual_bible=VisualBible(style="Cyber", palette="Neon", lighting="Dark"),
        shots=[shot],
    )
    wf_dir = proj_dir / "workflows"
    wf_dir.mkdir()
    (wf_dir / "still.api.json").write_text(
        json.dumps({"10": {"class_type": "LoadImage", "inputs": {"image": ""}}})
    )
    bindings_path = wf_dir / "still.bindings.json"
    bindings_path.write_text(
        json.dumps(
            {
                "workflow": "still.api.json",
                "inputs": {"reference_image": [{"node": "10", "field": "image"}]},
                "required": ["reference_image"],
                "output": {"node": "10", "extensions": [".png"]},
            }
        )
    )
    tpl = WorkflowTemplate.load("still", bindings_path)
    runner = ShotRunner.__new__(ShotRunner)
    runner._project = proj
    runner._manifest = manifest
    runner._templates = {"still": tpl}

    # 1. With a back panel (columns layout), rear-view routes to back.png (not front or face).
    refs_with_back = runner._still_reference_paths(shot, plate=None)
    assert refs_with_back[0][2].name == "back.png"

    # 2. Without a back panel (row/grid layout), rear-view routes to profile.png (never face.png).
    del manifest.character.references["w1"].panels["back"]
    refs_without_back = runner._still_reference_paths(shot, plate=None)
    assert refs_without_back[0][2].name == "profile.png"

    # 3. Negative prompt and composed prompt both enforce rear view.
    assert "frontal face" in runner._still_negative_prompt(shot)
    composed = prompts.compose_still_prompt(manifest, shot)
    assert "Camera behind subject (rear view)" in composed


def test_tuning_ignores_visual_direction_costume_ensemble_and_respects_explicit_solo_gender():
    from mvdirector import tuning
    from mvdirector.models import Analysis, TimeSpan, VocalTurn

    style_with_visual_ensemble = (
        "Music style: A high-energy pop track in A major.\n\n"
        "Visual direction: The lead performer changes into a darker ensemble symbolizing freedom."
    )
    prof = tuning.profile_track(
        lyrics_text="[Chorus]\nPut me on the naughty list",
        style_text=style_with_visual_ensemble,
    )
    assert prof.vocal_mode == "solo"

    fake_analysis = Analysis(
        duration=180.0,
        tempo_bpm=130.0,
        beats=[0.0, 0.5],
        bars=[0.0, 2.0],
        bar_intensity=[0.5],
        vocal_windows=[TimeSpan(start=10.0, end=120.0)],
        vocal_turns=[
            VocalTurn(start=10.0, end=60.0, voice="voice_1", median_f0_hz=260.0),
            VocalTurn(start=60.0, end=110.0, voice="voice_2", median_f0_hz=420.0),
        ],
    )
    prof_solo = tuning.profile_track(
        lyrics_text="[Chorus]\nHold the light",
        style_text="uplifting progressive trance, 132 BPM, female lead vocal, layered synths",
        analysis=fake_analysis,
    )
    assert prof_solo.vocal_mode == "solo"


def test_parse_lyrics_strips_suno_production_parens_and_keeps_adlibs():
    from mvdirector.audio import lyrics

    raw = (
        "[Chorus] (Strings swell, slow beat on beats 2 & 4, soaring emotional belt) "
        "We built paper castles right into the flame\n"
        "*(Wordless vocal chop, breathy, rhythmic)*\n"
        "[Drop] (Instrumental — vocals cut out, full sidechained bass and kick take over)\n"
        "[Interlude]\n"
        "(Sugar and ash)\n"
        "[Verse 1] I used to think monsters stayed under the floor\n"
    )
    parsed = lyrics.parse_lyrics(raw)
    assert [ln.text for ln in parsed] == [
        "We built paper castles right into the flame",
        "(Sugar and ash)",
        "I used to think monsters stayed under the floor",
    ]
    assert parsed[0].section == "Chorus"
    assert parsed[2].section == "Verse 1"


def test_clamped_lipsync_reconciles_medium_shot_and_waist_up_prose():
    bible = VisualBible(
        style="Neon",
        palette="Blue",
        lighting="Dark",
        settings=[SetDesign(name="Club", description="Neon club.")],
    )
    shot = Shot(
        id="s041",
        index=0,
        start=0.0,
        end=4.0,
        num_frames=97,
        type=ShotType.LIPSYNC,
        framing=Framing.MEDIUM_CLOSE,
        audio_slice=AudioSlice(path="assets/audio/s041.wav", start=0.0, end=4.0, head_handle=0.0),
    )
    creative = planner.ShotCreative(
        id="s041",
        type=ShotType.LIPSYNC,
        framing=Framing.MEDIUM,
        setting="Club",
        still_prompt=(
            "Medium shot of Kaelen framed from her waist up, looking into the lens. "
            "Her bare feet are planted on the ash, anchoring the composition."
        ),
        motion_prompt="Sings.",
    )
    planner._apply_creative(shot, creative, bible)
    assert shot.framing is Framing.MEDIUM_CLOSE
    assert "Medium close-up of Kaelen framed from the chest up" in shot.still_prompt
    assert "Medium shot" not in shot.still_prompt
    assert "waist up" not in shot.still_prompt
    assert "bare feet" not in shot.still_prompt


def test_extract_json_does_not_descend_into_nested_object_of_malformed_outer_json():
    from mvdirector.llm.base import extract_json

    malformed = '{"interpretation": {"section": "Verse 1"}, "concept": "bad",}'
    extracted = extract_json(malformed)
    assert extracted == malformed


def test_ollama_is_think_error_matches_model_not_supporting_thinking():
    from mvdirector.llm.ollama_backend import _is_think_error

    assert _is_think_error('{"error":"\\"gemma3:12b\\" does not support thinking"}')
    assert _is_think_error("unknown field: think")
    assert not _is_think_error("model not found")
