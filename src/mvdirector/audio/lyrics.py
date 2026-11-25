# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Lyric parsing and timing.

Timing comes from the best available source, in this order:

1. Word timestamps (a JSON file you supply, or faster-whisper if installed) matched
   against the known lyrics, so transcription mistakes do not change the text.
2. An estimate that spreads lines across detected vocal activity by length.
"""

from __future__ import annotations

import json
import logging
import queue
import re
import threading
import unicodedata
from collections.abc import Iterable
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Protocol

import librosa
from pydantic import BaseModel, Field, ValidationError

from mvdirector.errors import ConfigError, MediaToolError
from mvdirector.models import BOTH_VOICES, LyricLine, TimeSpan, VocalTurn
from mvdirector.project import Project

log = logging.getLogger(__name__)

# Square or CJK lenticular brackets mark a section. Parenthesised lines are sung adlibs.
_SECTION_TAG = re.compile(r"^\s*(?:\[([^\]]+)\]|［([^］]+)］|【([^】]+)】)\s*$")
_INLINE_CUE_TAG = re.compile(r"^\s*(?:\[([^\]]+)\]|［([^］]+)］|【([^】]+)】)\s+(.+)$")
_ROMANIZATION_PAREN = re.compile(r"\(([A-Za-z'’\s-]+)\)")
_HYPHENATED_TOKEN = re.compile(r"[A-Za-z]+-[A-Za-z]+")
_BIDI_AND_BOM = (
    "\ufeff\u200b\u200c\u200d\u200e\u200f\u061c"
    "\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069"
)
_IGNORED_TOKEN_CHARS = str.maketrans(
    {
        "\u0640": None,  # Arabic tatweel (sung vocal elongation)
        **{ch: None for ch in _BIDI_AND_BOM},
        "\u2018": "'",
        "\u2019": "'",
        "\u02bc": "'",
    }
)
WHISPER_SAMPLE_RATE = 16000


class Word(BaseModel):
    word: str
    start: float = Field(ge=0.0, allow_inf_nan=False)
    end: float = Field(ge=0.0, allow_inf_nan=False)


class TranscriptSegment(BaseModel):
    text: str
    start: float
    end: float


class Transcript(BaseModel):
    segments: list[TranscriptSegment]
    words: list[Word]
    language: str | None = None
    multilingual: bool = False


class Transcriber(Protocol):
    """Protocol for transcribing a vocal stem into timestamped segments and words."""

    def transcribe(self, audio_path: Path) -> Transcript: ...


def collapse_repetitions(text: str, *, max_ngram: int = 8, min_repeats: int = 4) -> str:
    """Collapse 4+ consecutive verbatim n-gram repeats in ``text`` back down to two.

    Text alone cannot tell a Whisper decoding loop from a sung hook ('na na na na'), so the
    transcriber applies this only to segments whose word timings show a loop (see
    :func:`collapse_word_repetitions`).

    Credits & Attribution:
        Inspired by ``GeekatplayStudio/ComfyUI-Music-to-Video``
        (https://github.com/GeekatplayStudio/ComfyUI-Music-to-Video,
        ``nodes.py:_collapse_repetitions``).
    """
    min_repeats = max(2, min_repeats)
    tokens = text.split()
    if len(tokens) < min_repeats:
        return text.strip()
    out: list[str] = []
    i = 0
    n = len(tokens)
    while i < n:
        collapsed = False
        upper_k = min(max_ngram, (n - i) // min_repeats)
        for k in range(upper_k, 0, -1):
            phrase = [t.casefold() for t in tokens[i : i + k]]
            reps = 1
            while (
                i + (reps + 1) * k <= n
                and [t.casefold() for t in tokens[i + reps * k : i + (reps + 1) * k]] == phrase
            ):
                reps += 1
            if reps >= min_repeats:
                out.extend(tokens[i : i + 2 * k])
                i += reps * k
                collapsed = True
                break
        if not collapsed:
            out.append(tokens[i])
            i += 1
    return " ".join(out)


# A Whisper decoding loop repeats text with zero-length, overlapping or crammed word timings;
# sung repeats ('na na na na') get distinct, increasing timings of normal length.
_LOOP_MIN_WORD_SECONDS = 0.04
_LOOP_MIN_SECONDS_PER_WORD = 0.08
_LOOP_OVERLAP_TOLERANCE = 0.02


def _is_decoding_loop(run: list[Word]) -> bool:
    """True when a run of repeated words has the degenerate timing of a decoding loop."""
    if any(w.end - w.start < _LOOP_MIN_WORD_SECONDS for w in run):
        return True
    if any(b.start < a.end - _LOOP_OVERLAP_TOLERANCE for a, b in zip(run, run[1:])):
        return True
    return run[-1].end - run[0].start < _LOOP_MIN_SECONDS_PER_WORD * len(run)


def collapse_word_repetitions(
    words: list[Word], *, max_ngram: int = 6, min_repeats: int = 4
) -> list[Word]:
    """Collapse Whisper decoding loops (4+ verbatim repeats with degenerate timing) to two.

    Repeats with real, spread-out timings are sung words and keep every timestamp, so lyric
    alignment still sees all of them.
    """
    min_repeats = max(2, min_repeats)
    if len(words) < min_repeats:
        return list(words)
    out: list[Word] = []
    i = 0
    n = len(words)
    norm = [w.word.strip().casefold() for w in words]
    while i < n:
        collapsed = False
        upper_k = min(max_ngram, (n - i) // min_repeats)
        for k in range(upper_k, 0, -1):
            phrase = norm[i : i + k]
            if not any(phrase):
                continue
            reps = 1
            while i + (reps + 1) * k <= n and norm[i + reps * k : i + (reps + 1) * k] == phrase:
                reps += 1
            if reps >= min_repeats:
                run = words[i : i + reps * k]
                out.extend(run[: 2 * k] if _is_decoding_loop(run) else run)
                i += reps * k
                collapsed = True
                break
        if not collapsed:
            out.append(words[i])
            i += 1
    return out


def _drain_segments_with_timeout(
    raw_segments: Iterable[Any],
    timeout_seconds: float,
    *,
    first_timeout_seconds: float | None = None,
) -> list[Any]:
    """Drain a faster-whisper segment generator with a per-segment stall timeout watchdog.

    ``first_timeout_seconds`` (default ``timeout_seconds``) bounds the wait for the first
    segment, which also covers feature extraction and language detection.

    Credits & Attribution:
        Stall-timeout watchdog inspired by ``GeekatplayStudio/ComfyUI-Music-to-Video``
        (https://github.com/GeekatplayStudio/ComfyUI-Music-to-Video,
        ``nodes.py:_drain_whisper_segments``).
    """
    if timeout_seconds <= 0:
        return list(raw_segments)
    sentinel = object()
    q: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=4)
    cancelled = threading.Event()

    def _put(item: tuple[str, Any]) -> bool:
        while not cancelled.is_set():
            try:
                q.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def _worker() -> None:
        try:
            for item in raw_segments:
                if cancelled.is_set() or not _put(("item", item)):
                    return
            _put(("done", sentinel))
        except BaseException as exc:  # noqa: BLE001
            _put(("error", exc))

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    collected: list[Any] = []
    first_wait = timeout_seconds if first_timeout_seconds is None else first_timeout_seconds
    try:
        while True:
            wait = first_wait if not collected else timeout_seconds
            try:
                kind, payload = q.get(timeout=wait)
            except queue.Empty as exc:
                cancelled.set()
                raise TimeoutError(
                    f"Whisper decoder stalled for >{wait:.0f}s without producing a segment."
                ) from exc
            if kind == "done":
                return collected
            if kind == "error":
                raise payload
            collected.append(payload)
    finally:
        cancelled.set()


class WhisperTranscriber:
    """Local vocal transcription backed by faster-whisper.

    Hardened with ``condition_on_previous_text=False``, optional known-lyrics ``initial_prompt``
    vocabulary seeding, a per-segment stall watchdog, and consecutive repetition collapsing
    (inspired by ``GeekatplayStudio/ComfyUI-Music-to-Video``).
    """

    def __init__(
        self,
        model_size: str = "small",
        language: str | None = None,
        *,
        multilingual: bool = False,
        initial_prompt: str | None = None,
        stall_timeout_seconds: float = 600.0,
    ) -> None:
        self.model_size = model_size
        self.language = language.strip().lower() if language else None
        self.multilingual = multilingual
        self.initial_prompt = initial_prompt
        self.stall_timeout_seconds = stall_timeout_seconds

    def transcribe(self, audio_path: Path) -> Transcript:
        if self.model_size.lower().endswith(".en") and (
            self.multilingual or (self.language is not None and self.language != "en")
        ):
            raise ConfigError(
                f"Whisper model {self.model_size!r} is English-only; use a multilingual model "
                "(for example 'small' or 'large-v3-turbo')."
            )
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise ConfigError(
                "Transcribing lyrics requires 'faster-whisper'. "
                "Install it with: pip install -e '.[align]'"
            ) from exc

        log.info(
            "Transcribing vocals with faster-whisper (%s; first run downloads the model)",
            self.model_size,
        )
        # Decode the audio ourselves and pass samples in: faster-whisper's own decoder
        # depends on PyAV, whose version often disagrees with faster-whisper's.
        samples, _ = librosa.load(str(audio_path), sr=WHISPER_SAMPLE_RATE, mono=True)
        model = WhisperModel(self.model_size, compute_type="int8")
        transcribe_kwargs: dict[str, object] = {
            "word_timestamps": True,
            "language": self.language,
            "condition_on_previous_text": False,
        }
        if self.multilingual:
            transcribe_kwargs["multilingual"] = True
        if self.initial_prompt and self.initial_prompt.strip():
            transcribe_kwargs["initial_prompt"] = self.initial_prompt.strip()[:800]
        raw_segments, info = model.transcribe(samples, **transcribe_kwargs)  # type: ignore[arg-type]
        try:
            drained = _drain_segments_with_timeout(
                raw_segments,
                self.stall_timeout_seconds,
                # The first segment also waits for feature extraction and language detection.
                first_timeout_seconds=self.stall_timeout_seconds * 3,
            )
        except TimeoutError as exc:
            raise MediaToolError(
                f"{exc} Raise prepare.whisper_stall_timeout_seconds for slow hardware or large "
                "models (0 disables the watchdog)."
            ) from exc
        segments: list[TranscriptSegment] = []
        words: list[Word] = []
        for seg in drained:
            seg_words = [
                Word(word=w.word, start=round(float(w.start), 3), end=round(float(w.end), 3))
                for w in seg.words or []
                if (getattr(w, "word", None) or "").strip()
            ]
            text = (getattr(seg, "text", None) or "").strip()
            if len(collapse_word_repetitions(seg_words)) < len(seg_words):
                text = collapse_repetitions(text)  # this segment's timing shows a decoding loop
            if text:
                segments.append(
                    TranscriptSegment(
                        text=text, start=round(float(seg.start), 3), end=round(float(seg.end), 3)
                    )
                )
            words.extend(seg_words)
        words = collapse_word_repetitions(words)
        detected_lang = getattr(info, "language", None) or self.language
        return Transcript(
            segments=segments,
            words=words,
            language=detected_lang,
            multilingual=self.multilingual,
        )


def _voice_tag_map(turns: list[VocalTurn] | None) -> dict[str, str]:
    """Map diarized voice ids ('voice_1', 'voice_2', 'both') to human section tags."""
    if not turns:
        return {}
    voices = list(dict.fromkeys(t.voice for t in turns))
    solo_voices = [v for v in voices if v != BOTH_VOICES]
    if len(solo_voices) < 2:
        return {}
    mapping = {voice: f"Singer {idx}" for idx, voice in enumerate(solo_voices, start=1)}
    if BOTH_VOICES in voices:
        mapping[BOTH_VOICES] = "Both"
    return mapping


def segment_speaker_label(start: float, end: float, turns: list[VocalTurn] | None) -> str | None:
    """Return 'Singer 1' / 'Singer 2' / 'Both' for ``[start, end]`` when multi-voice turns exist."""
    tag_map = _voice_tag_map(turns)
    if not tag_map or not turns:
        return None
    overlap_by_voice: dict[str, float] = {}
    for turn in turns:
        shared = turn.overlap(start, end)
        if shared > 0:
            overlap_by_voice[turn.voice] = overlap_by_voice.get(turn.voice, 0.0) + shared
    if not overlap_by_voice:
        return None
    best_voice = max(overlap_by_voice.items(), key=lambda pair: pair[1])[0]
    return tag_map.get(best_voice)


def format_transcript_as_lyrics(
    segments: list[TranscriptSegment],
    pause_threshold: float = 3.0,
    vocal_turns: list[VocalTurn] | None = None,
) -> str:
    """Turn timestamped transcript segments into `[Section]`-tagged lyrics text."""
    cleaned = [seg for seg in segments if seg.text.strip()]
    if not cleaned:
        return "[Instrumental]\n"

    first_speaker = segment_speaker_label(cleaned[0].start, cleaned[0].end, vocal_turns)
    sections: list[tuple[str | None, list[str]]] = [(first_speaker, [cleaned[0].text.strip()])]
    prev_end = cleaned[0].end
    for seg in cleaned[1:]:
        speaker = segment_speaker_label(seg.start, seg.end, vocal_turns)
        prev_speaker = sections[-1][0]
        speaker_changed = (
            speaker is not None and prev_speaker is not None and speaker != prev_speaker
        )
        if seg.start - prev_end >= pause_threshold or speaker_changed:
            sections.append((speaker, [seg.text.strip()]))
        else:
            if prev_speaker is None and speaker is not None:
                sections[-1] = (speaker, sections[-1][1])
            sections[-1][1].append(seg.text.strip())
        prev_end = max(prev_end, seg.end)

    blocks = []
    for idx, (speaker, lines) in enumerate(sections, start=1):
        header = f"[Section {idx}: {speaker}]" if speaker else f"[Section {idx}]"
        blocks.append(f"{header}\n" + "\n".join(lines))
    return "\n\n".join(blocks) + "\n"


_STRUCTURAL_SECTION_WORDS = frozenset(
    {
        # English
        "intro",
        "verse",
        "pre-chorus",
        "prechorus",
        "pre",
        "pré",
        "chorus",
        "post-chorus",
        "postchorus",
        "post",
        "pós",
        "pos",
        "hook",
        "big",
        "refrain",
        "bridge",
        "outro",
        "interlude",
        "instrumental",
        "solo",
        "break",
        "breakdown",
        "drop",
        "build",
        "buildup",
        "build-up",
        "rebuild",
        "re-build",
        "cooldown",
        "cool-down",
        "climax",
        "pause",
        "stop",
        "silence",
        "scream",
        "chant",
        "melody",
        "melodic",
        "tag",
        "coda",
        "spoken",
        "adlib",
        "ad-lib",
        "adlibs",
        "ad-libs",
        "rap",
        "fade",
        "ending",
        "end",
        "part",
        "section",
        "final",
        "reprise",
        "repeat",
        "repeated",
        "again",
        "vamp",
        "turnaround",
        "soft",
        "loud",
        "quiet",
        "whisper",
        "whispered",
        "acoustic",
        "heavy",
        "stripped",
        "building",
        "calm",
        "intense",
        "acapella",
        "capella",
        "fast",
        "slow",
        "high",
        "low",
        "emotional",
        "softly",
        "gently",
        "out",
        "in",
        "up",
        "down",
        "lead",
        "backing",
        "backup",
        "vocal",
        "vocals",
        "vox",
        # Romance (Spanish, Portuguese, French, Italian)
        "introducción",
        "introducao",
        "introdução",
        "introduzione",
        "verso",
        "estrofa",
        "estrofe",
        "couplet",
        "strophe",
        "pre-coro",
        "precoro",
        "pre-estribillo",
        "pré-refrão",
        "pre-refrao",
        "pré-refrain",
        "coro",
        "estribillo",
        "refrão",
        "refrao",
        "ritornello",
        "post-coro",
        "pós-refrão",
        "pos-refrao",
        "puente",
        "ponte",
        "pont",
        "interludio",
        "intermède",
        "intermede",
        "salida",
        "cierre",
        "hablado",
        "parlé",
        "parle",
        # Germanic / Nordic / Polish / Turkish / Indonesian
        "vers",
        "refräng",
        "omkvæd",
        "refrein",
        "refren",
        "zwrotka",
        "mostek",
        "brücke",
        "brucke",
        "vorspiel",
        "zwischenspiel",
        "nachspiel",
        "giriş",
        "kıta",
        "nakarat",
        "köprü",
        "çıkış",
        "bait",
        "reff",
        # Slavic (Russian)
        "интро",
        "вступление",
        "куплет",
        "предприпев",
        "припев",
        "бридж",
        "проигрыш",
        "соло",
        "аутро",
        "концовка",
        # Japanese
        "イントロ",
        "メロ",
        "aメロ",
        "bメロ",
        "cメロ",
        "dメロ",
        "プレサビ",
        "サビ",
        "大サビ",
        "ラスサビ",
        "落ちサビ",
        "間奏",
        "前奏",
        "後奏",
        "アウトロ",
        "ブリッジ",
        "フック",
        "ラップ",
        "セリフ",
        "パート",
        "コーラス",
        # Chinese (Simplified & Traditional)
        "主歌",
        "导歌",
        "導歌",
        "预副歌",
        "預副歌",
        "副歌",
        "间奏",
        "桥段",
        "橋段",
        "尾奏",
        "结尾",
        "結尾",
        "说唱",
        "說唱",
        "独白",
        "獨白",
        "段",
        # Korean
        "인트로",
        "전주",
        "절",
        "프리코러스",
        "코러스",
        "후렴",
        "후렴구",
        "브릿지",
        "브리지",
        "간주",
        "아웃트로",
        "후주",
        "랩",
        "내레이션",
        "마지막",
        # Arabic & Hindi
        "مقدمة",
        "مقطع",
        "لازمة",
        "كورال",
        "خاتمة",
        "موسيقى",
        "मुखड़ा",
        "अंतरा",
    }
)
_ENSEMBLE_SECTION_WORDS = frozenset(
    {
        "duet",
        "dueto",
        "duo",
        "dúo",
        "both",
        "ambos",
        "beide",
        "tous",
        "tous les deux",
        "juntos",
        "ensemble",
        "zusammen",
        "insieme",
        "together",
        "all",
        "all voices",
        "all vocals",
        "all singers",
        "todos",
        "everyone",
        "choir",
        "group",
        "harmony",
        "вместе",
        "оба",
        "дуэт",
        "хор",
        "二人",
        "ふたり",
        "全員",
        "みんなで",
        "合唱",
        "合",
        "함께",
        "같이",
        "듀엣",
        "떼창",
    }
)
_ENGLISH_DESIGNATOR_WORDS = frozenset(
    {"male", "female", "man", "woman", "boy", "girl", "guy", "lady", "he", "she", "him", "her"}
)
_LOW_FOREIGN_DESIGNATORS = frozenset(
    {
        "hombre",
        "chico",
        "homme",
        "garçon",
        "garcon",
        "mann",
        "ragazzo",
        "мужчина",
        "парень",
        "男",
        "男声",
        "男聲",
        "男性",
        "남",
        "남자",
        "남성",
    }
)
_HIGH_FOREIGN_DESIGNATORS = frozenset(
    {
        "mujer",
        "chica",
        "femme",
        "fille",
        "frau",
        "ragazza",
        "женщина",
        "девушка",
        "女",
        "女声",
        "女聲",
        "女性",
        "여",
        "여자",
        "여성",
    }
)
_DESIGNATOR_WORDS = _ENGLISH_DESIGNATOR_WORDS | _LOW_FOREIGN_DESIGNATORS | _HIGH_FOREIGN_DESIGNATORS
_ENGLISH_NUMBERED_DESIGNATORS = frozenset(
    {"singer", "voice", "vocalist", "vocal", "vocals", "rapper", "character", "performer"}
)
_FOREIGN_NUMBERED_DESIGNATORS = frozenset(
    {
        "cantante",
        "voz",
        "chanteur",
        "chanteuse",
        "voix",
        "sänger",
        "sanger",
        "stimme",
        "певец",
        "певица",
        "голос",
        "歌手",
        "ボーカル",
        "シンガー",
        "가수",
        "보컬",
    }
)
_NUMBERED_DESIGNATORS = _ENGLISH_NUMBERED_DESIGNATORS | _FOREIGN_NUMBERED_DESIGNATORS
_COMPACT_NUMBERED_RE = re.compile(r"^([^\W\d_]+)(\d+)$")
_SECTION_ORDINAL_RE = re.compile(
    r"^(?:第?[0-9一二三四五六七八九十]+番?|\d+)|[0-9一二三四五六七八九十]+$"
)
_JP_MELO_RE = re.compile(r"^[a-z]メロ$")
_SINGER_SPLIT = re.compile(
    r"\s*(?:&|\+|/|・|、|\b(?:and|y|e|et|und|и)\b)\s*",
    flags=re.IGNORECASE,
)
_REPEAT_MARKER = re.compile(r"\s*[(\[【]?\s*(?:x|×)\s*\d+\s*[)\]】]?\s*$", flags=re.IGNORECASE)
_BRACKETED_CUE = re.compile(r"^(.*?)\s*[(\[【]([^()\[\]【】]+)[)\]】]\s*$")
_DASH_SPLIT = re.compile(r"\s+[-—–]\s+")
# Pre-chorus and post-chorus equivalents excluded before checking hook/chorus markers.
_NON_HOOK_CHORUS_RE = re.compile(
    r"(?:\b(?:pre|post|pré|pós|pos)[\s-]?"
    r"(?:chorus|coro|estribillo|refrão|refrao|refrain|refren|refrein)\b"
    r"|\bпредприпев\b|プレサビ|预副歌|預副歌|导歌|導歌|프리코러스)",
    flags=re.IGNORECASE,
)
_HOOK_SECTION_RE = re.compile(
    r"(?:\b(?:chorus|hook|refrain|estribillo|coro|refrão|refrao|ritornello|"
    r"refren|refrein|refräng|omkvæd|nakarat|reff|припев|كورال|لازمة)\b"
    r"|サビ|フック|副歌|후렴|코러스|मुखड़ा)",
    flags=re.IGNORECASE,
)


_HYPHEN_TRANS = str.maketrans({"\u2010": "-", "\u2011": "-", "\u2212": "-"})
_NON_SINGER_CUE_WORDS = frozenset(
    {
        # Languages often noted in Suno section headers (e.g. "[Chorus: English, then Mandarin]")
        "english",
        "french",
        "spanish",
        "mandarin",
        "cantonese",
        "chinese",
        "korean",
        "japanese",
        "indonesian",
        "malay",
        "portuguese",
        "italian",
        "german",
        "tagalog",
        "filipino",
        "arabic",
        "hindi",
        "russian",
        # Instruments, arrangement, and vocal delivery cues (e.g. "[Breakdown: Piano, airy pad]")
        "piano",
        "guitar",
        "drums",
        "strings",
        "pads",
        "synth",
        "synths",
        "supersaw",
        "supersaws",
        "arpeggio",
        "bass",
        "kick",
        "snare",
        "riser",
        "percussion",
        "cello",
        "violin",
        "horns",
        "brass",
        "sub",
        "explosive",
        "tender",
        "husky",
        "raw",
        "pleading",
        "restrained",
        "sparse",
        "fragile",
        "very",
        "bare",
        "rising",
        "full",
        "production",
        "energy",
        "volume",
        "maximum",
        "biggest",
        "atmospheric",
        "filtered",
        "stacked",
        "harmonies",
        "echoes",
        "half-spoken",
        "half-time",
        "modern",
        "hybrid",
        "camera",
        "shutter",
        "only",
        "begins",
        "returns",
        "continue",
        "swell",
        "slows",
        "tempo",
    }
)


def is_hook_section(section: str | None) -> bool:
    """True for chorus/hook sections across languages, excluding pre-chorus and post-chorus."""
    if not section:
        return False
    norm = unicodedata.normalize("NFKC", section).translate(_HYPHEN_TRANS)
    cleaned = _NON_HOOK_CHORUS_RE.sub("", norm)
    return bool(_HOOK_SECTION_RE.search(cleaned))


def _strip_section_affixes(word: str) -> str:
    stripped = _SECTION_ORDINAL_RE.sub("", word)
    return "メロ" if _JP_MELO_RE.match(stripped) else stripped


def _is_word_structural(word: str) -> bool:
    stripped = _strip_section_affixes(word)
    if not stripped:
        return False
    if stripped in _STRUCTURAL_SECTION_WORDS:
        return True
    if "-" in stripped:
        parts = [_strip_section_affixes(p) for p in stripped.split("-") if p]
        return bool(parts) and all(p in _STRUCTURAL_SECTION_WORDS for p in parts)
    return False


def _canonical_designator(words: list[str]) -> str:
    if all(w in _LOW_FOREIGN_DESIGNATORS for w in words):
        return "male"
    if all(w in _HIGH_FOREIGN_DESIGNATORS for w in words):
        return "female"
    return " ".join(words)


def _is_structural(words: list[str]) -> bool:
    core = [w for w in words if _strip_section_affixes(w)]
    return bool(core) and all(_is_word_structural(w) for w in core)


def _looks_like_name(text: str) -> bool:
    return all(w[:1].isupper() for w in text.split())


def _split_section_cue(label: str) -> tuple[str, bool, bool]:
    """Return ``(cue, explicit, after_structural_dash)`` for ``label``."""
    norm = unicodedata.normalize("NFKC", label.strip().strip(_BIDI_AND_BOM)).translate(
        _HYPHEN_TRANS
    )
    norm = _REPEAT_MARKER.sub("", norm)
    bracketed = _BRACKETED_CUE.match(norm)
    if bracketed:
        return bracketed.group(2).strip(), True, False
    if ":" in norm:
        return norm.split(":", 1)[1].strip(), True, False
    if _DASH_SPLIT.search(norm):
        lhs, rhs = _DASH_SPLIT.split(norm, maxsplit=1)
        lhs_structural = _is_structural(lhs.casefold().split())
        return (rhs if lhs_structural else lhs).strip(), False, lhs_structural
    return norm, False, False


def section_cue(label: str) -> tuple[str, bool]:
    """Split a section label into its performer cue and whether that cue was explicit."""
    cue, explicit, _ = _split_section_cue(label)
    return cue, explicit


def _has_bicameral_letters(text: str) -> bool:
    return any(ch.isalpha() and ch.casefold() != ch.upper() for ch in text)


def singers_in_cue(
    cue: str,
    *,
    explicit: bool,
    cased: bool,
    allow_bare_names: bool = True,
) -> set[str] | None:
    """Performer keys named by one section cue, or ``None`` when the cue names an ensemble."""
    singers: set[str] = set()
    head = cue.split(",", 1)[0].strip()
    for raw in _SINGER_SPLIT.split(head):
        part = _REPEAT_MARKER.sub("", raw.strip())
        if not part:
            continue
        words = part.casefold().split()
        joined = " ".join(words)
        if joined in _ENSEMBLE_SECTION_WORDS or joined in ("call and response",):
            return None
        if len(words) == 1:
            compact = _COMPACT_NUMBERED_RE.match(words[0])
            if compact and compact.group(1) in _NUMBERED_DESIGNATORS:
                role = (
                    compact.group(1)
                    if compact.group(1) in _ENGLISH_NUMBERED_DESIGNATORS
                    else "singer"
                )
                singers.add(f"{role} {compact.group(2)}")
                continue
        if (
            len(words) == 2
            and words[0] in _NUMBERED_DESIGNATORS
            and (words[1].isdigit() or (len(words[1]) == 1 and words[1].isalpha()))
        ):
            role = words[0] if words[0] in _ENGLISH_NUMBERED_DESIGNATORS else "singer"
            singers.add(f"{role} {words[1]}")
            continue
        if (
            len(words) == 3
            and words[0] in _DESIGNATOR_WORDS
            and words[1] in _NUMBERED_DESIGNATORS
            and (words[2].isdigit() or (len(words[2]) == 1 and words[2].isalpha()))
        ):
            gender = _canonical_designator([words[0]])
            role = words[1] if words[1] in _ENGLISH_NUMBERED_DESIGNATORS else "singer"
            singers.add(f"{gender} {role} {words[2]}")
            continue
        core = [_strip_section_affixes(w) for w in words]
        core = [w for w in core if w]
        if not core:
            continue
        if any(_is_word_structural(w) for w in words):
            rest = [
                _strip_section_affixes(w)
                for w in words
                if not _is_word_structural(w) and _strip_section_affixes(w)
            ]
            if rest and all(w in _DESIGNATOR_WORDS for w in rest):
                singers.add(_canonical_designator(rest))
            continue
        if all(w in _DESIGNATOR_WORDS for w in core):
            singers.add(_canonical_designator(core))
            continue
        if any(w in _NON_SINGER_CUE_WORDS for w in core):
            continue
        if len(core) > 3 or any(w.isdigit() for w in words):
            continue
        if explicit:
            # In a mixed-case header like "[Bridge: bare percussion]", lowercase Latin phrases are
            # stage directions rather than proper names; unicameral scripts (CJK/Kana/Hangul) pass.
            if cased and _has_bicameral_letters(part) and not _looks_like_name(part):
                continue
            singers.add(" ".join(core))
        elif allow_bare_names and cased and _looks_like_name(part):
            singers.add(" ".join(core))
    return singers


def section_singers(label: str | None) -> set[str] | None:
    """Performer keys named by ``label`` (empty if none, ``None`` for ensemble cues)."""
    if not label or not label.strip():
        return set()
    cleaned = unicodedata.normalize("NFKC", label.strip().strip(_BIDI_AND_BOM)).translate(
        _HYPHEN_TRANS
    )
    cue, explicit, after_structural_dash = _split_section_cue(cleaned)
    if not cue:
        return set()
    return singers_in_cue(
        cue,
        explicit=explicit,
        cased=cleaned != cleaned.casefold(),
        allow_bare_names=not after_structural_dash,
    )


def extract_section_tags(text: str) -> list[str]:
    """Return every ``[Section]`` header in ``text`` in order, including empty/instrumental ones."""
    tags: list[str] = []
    for raw in text.lstrip("\ufeff").splitlines():
        cleaned = raw.strip().strip(_BIDI_AND_BOM).strip()
        if not cleaned:
            continue
        match = _SECTION_TAG.match(cleaned)
        if match:
            tag = next((g for g in match.groups() if g is not None), "").strip()
            tag = tag.strip(_BIDI_AND_BOM).strip()
            if tag:
                tags.append(tag)
    return tags


def labels_have_multiple_singers(labels: Iterable[str | None]) -> bool:
    """True when ``labels`` name two or more distinct singers or an ensemble cue."""
    singers: set[str] = set()
    for raw in labels:
        found = section_singers(raw)
        if found is None:
            return True
        singers |= found
    return len(singers) >= 2


_SECTION_MODIFIER_WORDS = frozenset(
    {"final", "main", "big", "first", "second", "third", "last", "soft", "quiet", "loud", "double"}
)
_ARRANGEMENT_NOTE_WORDS = _NON_SINGER_CUE_WORDS | frozenset(
    {
        "bpm",
        "bassline",
        "sidechain",
        "reverb",
        "echo",
        "melody",
        "ambience",
        "progression",
        "instrumentation",
        "pitch",
        "roll",
        "accelerating",
        "pluck",
        "trance",
        "festival",
        "euphoric",
        "soaring",
        "pulsing",
        "breathing",
        "drops",
        "fades",
        "opens",
        "wider",
        "peak",
        "power",
        "emotional",
        "massive",
        "ethereal",
        "uplifting",
    }
)


def _cue_head_words(tag: str) -> list[str]:
    """Casefolded words of ``tag`` before any ``:`` or dash qualifier."""
    norm = unicodedata.normalize("NFKC", tag).translate(_HYPHEN_TRANS)
    head = norm.split(":", 1)[0]
    head = _DASH_SPLIT.split(head, maxsplit=1)[0]
    words = [re.sub(r"^[^\w]+|[^\w]+$", "", w.casefold()) for w in head.split()]
    return [w for w in words if w]


# Words ``_STRUCTURAL_SECTION_WORDS`` accepts that describe delivery rather than a song part.
# Inline before a sung line ("[whispered] ...", "[soft vocals] ...") they must not open a section.
_DELIVERY_CUE_WORDS = frozenset(
    {
        "whisper",
        "whispered",
        "spoken",
        "hablado",
        "parle",
        "scream",
        "soft",
        "softly",
        "gently",
        "quiet",
        "loud",
        "calm",
        "heavy",
        "intense",
        "emotional",
        "fast",
        "slow",
        "high",
        "low",
        "melodic",
        "melody",
        "acapella",
        "capella",
        "acoustic",
        "stripped",
        "vocal",
        "vocals",
        "vox",
        "backing",
        "backup",
        "lead",
        "again",
        "repeat",
        "repeated",
        "adlib",
        "adlibs",
        "ad-lib",
        "ad-libs",
        "in",
        "out",
        "up",
        "down",
    }
)


def _names_a_section(words: list[str]) -> bool:
    """True when a cue starts like a song section (``Verse 2``, ``Final Chorus``)."""
    if not words:
        return False
    if _is_word_structural(words[0]):
        return True
    return words[0] in _SECTION_MODIFIER_WORDS and len(words) >= 2 and _is_word_structural(words[1])


def _opens_section(cue: str) -> bool:
    """True when an inline cue names a song part (``Verse 2``, ``Whispered Bridge``).

    Leading modifiers and delivery words are skipped; the next word must be a structural word
    that is not itself only a delivery cue, so ``[whispered] line`` keeps the current section.
    """
    for word in _cue_head_words(cue):
        core = _strip_section_affixes(word)
        if core in _DELIVERY_CUE_WORDS or core in _SECTION_MODIFIER_WORDS:
            continue
        return _is_word_structural(word)
    return False


def _is_arrangement_note(tag: str) -> bool:
    """True for Suno arrangement/production cues (e.g. '[Kick drops out]') under a header."""
    if section_singers(tag) != set():
        return False
    words = _cue_head_words(tag)
    if _names_a_section(words):
        return False
    return len(words) >= 2 and any(w in _ARRANGEMENT_NOTE_WORDS for w in words)


def strip_romanization_guides(text: str) -> str:
    """Drop parenthetical hyphenated Latin pronunciation guides on lines with non-Latin script."""
    if not any(is_syllabic_char(ch) or ord(ch) > 0x024F for ch in text):
        return text

    def _replace(match: re.Match[str]) -> str:
        inner = match.group(1).strip()
        return "" if _HYPHENATED_TOKEN.search(inner) else match.group(0)

    return _ROMANIZATION_PAREN.sub(_replace, text)


_PRODUCTION_PAREN_TOKEN_RE = re.compile(
    r"\b(?:tempo|bpm|piano|drum|drums|snare|kick|bass|sub-bass|synth|strings|cello|violin|"
    r"violins|guitar|percussion|arpeggio|sidechain|sidechained|crescendo|decelerando|reverb|"
    r"instrumental|vocals?|beat|beats|drop|groove|harmonies|whispery|whispered|soaring|belted|"
    r"fades?|sparse|bars|stereo|toms|pad|pads|delivery|swell|swells|stripped|acoustic|electric|"
    r"chop|chops|breathy|rhythmic|wordless|pitched|noise|filtered)\b",
    re.IGNORECASE,
)


def _strip_production_parens(text: str) -> str:
    """Strip parenthetical Suno stage/arrangement directions while keeping sung adlibs."""

    def _replace(match: re.Match[str]) -> str:
        inner = match.group(1).strip()
        if re.match(r"^(?:tempo|instrumental)\b", inner, re.IGNORECASE):
            return ""
        if len(set(t.lower() for t in _PRODUCTION_PAREN_TOKEN_RE.findall(inner))) >= 2:
            return ""
        return match.group(0)

    cleaned = re.sub(r"\*?\(([^()]+)\)\*?", _replace, text)
    return re.sub(r"\s{2,}", " ", cleaned).strip()


def parse_lyrics(text: str) -> list[LyricLine]:
    """Split lyrics into lines, turning ``[Chorus]``-style tags into section labels.

    Lines in parentheses are kept as lyrics (adlibs are sung and need timing too), while
    parenthetical Suno production/arrangement directions (e.g. ``(Strings swell, slow beat)``)
    are stripped. Inline leading cues are stripped from the sung text: singer cues such as
    ``[female voice 1] line`` are attached to the active section label, section names such as
    ``[Verse 1] line`` start a new section, and delivery cues such as ``[whispered] line`` leave
    the section unchanged.
    """
    lines: list[LyricLine] = []
    section: str | None = None
    for raw in text.lstrip("\ufeff").splitlines():
        stripped = raw.strip()
        if not stripped:
            continue
        cleaned_tag_line = stripped.strip(_BIDI_AND_BOM).strip()
        if not cleaned_tag_line:
            continue
        tag = _SECTION_TAG.match(cleaned_tag_line)
        if tag:
            matched = next((g for g in tag.groups() if g is not None), "")
            candidate = matched.strip().strip(_BIDI_AND_BOM).strip() or None
            if candidate is not None and not (
                section is not None and _is_arrangement_note(candidate)
            ):
                section = candidate
            continue
        line_section = section
        line_text = stripped.strip("\ufeff")
        inline = _INLINE_CUE_TAG.match(cleaned_tag_line)
        if inline:
            cue = next((g for g in inline.groups()[:3] if g is not None), "")
            cue = cue.strip().strip(_BIDI_AND_BOM).strip()
            rest = inline.group(4).strip()
            if cue and rest:
                line_text = rest
                if _opens_section(cue):  # "[Verse 2] ..." or "[Verse 2: Singer 1] ..."
                    section = cue
                    line_section = cue
                elif section_singers(cue) != set():
                    # Swap any singer qualifier on the header ("Verse 1: Singer 1") for this one.
                    base = section.split(":", 1)[0].strip() if section else ""
                    line_section = f"{base}: {cue}" if base else cue
                # Otherwise a delivery/production cue ("[whispered] ...", "[laughs] ..."):
                # drop it from the sung text without renaming the surrounding section.
        line_text = _strip_production_parens(line_text)
        if not line_text or re.fullmatch(r"[\W_]+", line_text):
            continue
        lines.append(LyricLine(text=line_text, section=line_section))
    return lines


def is_syllabic_char(ch: str) -> bool:
    """True for unspaced or syllabic scripts tokenized per character/cluster."""
    cp = ord(ch)
    if cp < 0x0E00:
        return False
    in_range = (
        0x0E00 <= cp <= 0x109F  # Thai, Lao, Tibetan, Myanmar
        or 0x1780 <= cp <= 0x17FF  # Khmer
        or 0x3005 <= cp <= 0x3006  # CJK iteration marks (々, 〆)
        or 0x3040 <= cp <= 0x312F  # Hiragana, Katakana, Bopomofo
        or 0x31F0 <= cp <= 0x31FF  # Katakana phonetic extensions
        or 0x3400 <= cp <= 0x4DBF  # CJK Extension A
        or 0x4E00 <= cp <= 0x9FFF  # CJK Unified Ideographs
        or 0xAC00 <= cp <= 0xD7AF  # Hangul Syllables
        or 0xF900 <= cp <= 0xFAFF  # CJK Compatibility Ideographs
        or 0x20000 <= cp <= 0x323AF  # CJK Extensions B–H
    )
    return in_range and unicodedata.category(ch)[0] in ("L", "N")


def tokenize(text: str) -> list[str]:
    """Split lyric or transcript text into normalized tokens across any Unicode script."""
    cleaned = strip_romanization_guides(text)
    norm = unicodedata.normalize("NFKC", cleaned).translate(_IGNORED_TOKEN_CHARS).casefold()
    tokens: list[str] = []
    buf: list[str] = []

    def flush() -> None:
        if not buf:
            return
        word = "".join(buf).strip("'")
        buf.clear()
        if word:
            tokens.append(word)

    i = 0
    n = len(norm)
    while i < n:
        ch = norm[i]
        if is_syllabic_char(ch):
            flush()
            end = i + 1
            while end < n and unicodedata.category(norm[end]).startswith("M"):
                end += 1
            tokens.append(norm[i:end])
            i = end
            continue
        cat = unicodedata.category(ch)
        if cat[0] in ("L", "N") or (cat[0] == "M" and buf) or (ch == "'" and buf):
            buf.append(ch)
        else:
            flush()
        i += 1
    flush()
    return tokens


def _sung_weight(text: str) -> int:
    """Estimated relative singing duration weight across single- and mixed-script lines."""
    total = 0
    for ch in unicodedata.normalize("NFKC", text):
        if ch.isspace() or unicodedata.category(ch).startswith("M"):
            continue
        total += 3 if is_syllabic_char(ch) else 1
    return max(1, total)


def load_or_transcribe_words(project: Project) -> list[Word]:
    """Word timestamps from the configured JSON file, else faster-whisper, else none."""
    inputs = project.config.inputs
    if inputs.word_timestamps:
        ts_path = project.input_path(inputs.word_timestamps, "word_timestamps")
        try:
            raw = json.loads(ts_path.read_text("utf-8-sig"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ConfigError(f"Could not parse word timestamps file {ts_path}: {exc}") from exc
        except RecursionError:
            raise ConfigError(
                f"Word timestamps file {ts_path} nests JSON too deeply to parse."
            ) from None
        if not isinstance(raw, list):
            raise ConfigError(
                f"Invalid word timestamps file {ts_path}: expected a JSON list of word objects."
            )
        try:
            return [Word.model_validate(item) for item in raw]
        except ValidationError as exc:
            raise ConfigError(f"Invalid word timestamps in {ts_path}:\n{exc}") from exc
    prepare_cfg = project.config.prepare
    if project.config.auto_tune:
        from mvdirector import tuning as tuning_mod

        lyrics_text, style_text = tuning_mod.read_optional_project_inputs(project)
        prepare_cfg, whisper_adjustments = tuning_mod.tune_prepare_config(
            project.config,
            lyrics_text=lyrics_text,
            style_text=style_text,
            title_hint=project.config.title,
        )
        for adj in whisper_adjustments:
            log.info(
                "Auto-tuned %s: %r -> %r (%s)", adj.key, adj.current, adj.recommended, adj.reason
            )
    vocals_path = project.input_path(inputs.vocals, "vocals")
    vocab_hint: str | None = None
    if inputs.lyrics:
        lyrics_file = project.path(inputs.lyrics)
        if lyrics_file.is_file():
            parsed_hint = parse_lyrics(lyrics_file.read_text("utf-8-sig"))
            if parsed_hint:
                vocab_hint = " ".join(ln.text for ln in parsed_hint)[:800]
    transcriber = WhisperTranscriber(
        model_size=prepare_cfg.whisper_model,
        language=prepare_cfg.whisper_language,
        multilingual=prepare_cfg.whisper_multilingual,
        initial_prompt=vocab_hint,
        stall_timeout_seconds=prepare_cfg.whisper_stall_timeout_seconds,
    )
    try:
        return transcriber.transcribe(vocals_path).words
    except ConfigError as exc:
        log.warning(
            "%s Lyric timings will be estimated from vocal activity.",
            exc,
        )
        return []
    except Exception as exc:  # noqa: BLE001 - alignment is an enhancement; never block analysis
        log.warning(
            "Transcription failed (%s: %s); lyric timings will be estimated from vocal activity.",
            type(exc).__name__,
            exc,
        )
        return []


def align_lines_to_words(lines: list[LyricLine], words: list[Word]) -> list[LyricLine]:
    """Give each lyric line the time span of the transcribed words it matches.

    Lines with no matching words are left untimed for ``estimate_missing_timings``.
    """
    lyric_tokens: list[str] = []
    token_line: list[int] = []
    line_token_counts: list[int] = []
    for index, line in enumerate(lines):
        tokens = tokenize(line.text)
        line_token_counts.append(len(tokens))
        for token in tokens:
            lyric_tokens.append(token)
            token_line.append(index)

    word_tokens: list[str] = []
    token_word: list[int] = []
    for index, word in enumerate(words):
        for token in tokenize(word.word):
            word_tokens.append(token)
            token_word.append(index)

    # Record (start, end, block_size, is_syllabic) per matched token on each line so an isolated
    # 1-token stopword/particle match cannot stretch across a singer gap or mis-anchor a CJK line.
    matched_words: dict[int, list[tuple[float, float, int, bool]]] = {}
    matcher = SequenceMatcher(a=lyric_tokens, b=word_tokens, autojunk=False)
    for block in matcher.get_matching_blocks():
        for offset in range(block.size):
            tok_idx = block.a + offset
            line_index = token_line[tok_idx]
            word = words[token_word[block.b + offset]]
            syllabic = is_syllabic_char(lyric_tokens[tok_idx][0])
            matched_words.setdefault(line_index, []).append(
                (word.start, word.end, block.size, syllabic)
            )

    spans: dict[int, tuple[float, float]] = {}
    for line_index, hits in matched_words.items():
        hits.sort(key=lambda item: item[0])
        while len(hits) >= 2 and hits[0][2] == 1 and hits[1][0] - hits[0][1] > 3.5:
            hits = hits[1:]
        while len(hits) >= 2 and hits[-1][2] == 1 and hits[-1][0] - hits[-2][1] > 3.5:
            hits = hits[:-1]
        if len(hits) == 1 and hits[0][2] == 1 and hits[0][3] and line_token_counts[line_index] >= 4:
            continue
        spans[line_index] = (min(h[0] for h in hits), max(h[1] for h in hits))

    aligned = []
    for index, line in enumerate(lines):
        if index in spans:
            start, end = spans[index]
            aligned.append(
                line.model_copy(
                    update={
                        "start": round(start, 3),
                        "end": round(end, 3),
                        "timing_source": "aligned",
                    }
                )
            )
        else:
            aligned.append(line)
    log.info("Aligned %d of %d lyric lines to word timestamps", len(spans), len(lines))
    return aligned


def estimate_missing_timings(lines: list[LyricLine], windows: list[TimeSpan]) -> list[LyricLine]:
    """Fill in untimed lines.

    With no aligned lines at all, lines are spread across vocal activity in proportion
    to their estimated sung length. Otherwise each run of untimed lines shares the gap between its
    timed neighbours.
    """
    if not lines or all(line.timing_source == "aligned" for line in lines):
        return lines
    if not any(line.timing_source == "aligned" for line in lines):
        return _spread_over_windows(lines, windows)

    result = list(lines)
    song_end = max([w.end for w in windows] + [ln.end for ln in lines if ln.end is not None])
    index = 0
    while index < len(result):
        if result[index].start is not None:
            index += 1
            continue
        run_end = index
        while run_end < len(result) and result[run_end].start is None:
            run_end += 1
        gap_start = result[index - 1].end if index > 0 else 0.0
        gap_end = result[run_end].start if run_end < len(result) else song_end
        gap_start = gap_start or 0.0
        gap_end = max(gap_end if gap_end is not None else song_end, gap_start)
        weights = [_sung_weight(result[i].text) for i in range(index, run_end)]
        cursor = gap_start
        for i, weight in zip(range(index, run_end), weights):
            share = (gap_end - gap_start) * weight / sum(weights)
            result[i] = result[i].model_copy(
                update={
                    "start": round(cursor, 3),
                    "end": round(cursor + share, 3),
                    "timing_source": "estimated",
                }
            )
            cursor += share
        index = run_end
    return result


def _spread_over_windows(lines: list[LyricLine], windows: list[TimeSpan]) -> list[LyricLine]:
    total = sum(w.duration for w in windows)
    if total <= 0:
        return lines

    def to_absolute(vocal_time: float, at_end: bool) -> float:
        """Map seconds of cumulative singing back to a position in the song."""
        elapsed = 0.0
        for window in windows:
            reached = elapsed + window.duration
            if vocal_time < reached or (at_end and vocal_time <= reached + 1e-9):
                return window.start + (vocal_time - elapsed)
            elapsed = reached
        return windows[-1].end

    weights = [_sung_weight(line.text) for line in lines]
    result, cursor = [], 0.0
    for line, weight in zip(lines, weights):
        share = total * weight / sum(weights)
        result.append(
            line.model_copy(
                update={
                    "start": round(to_absolute(cursor, at_end=False), 3),
                    "end": round(to_absolute(cursor + share, at_end=True), 3),
                    "timing_source": "estimated",
                }
            )
        )
        cursor += share
    return result
