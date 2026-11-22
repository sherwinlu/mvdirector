# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Track characteristic analysis and automatic pipeline parameter tuning.

Inspects ``lyrics.txt`` and ``style.md`` ahead of transcription (so Whisper model and
language flags match non-English or code-switching songs), then refines pacing, cut mode,
and lip-sync budgets using measured audio dynamics, vocal turns, and syllable rates.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from pydantic import ValidationError

from mvdirector.audio import lyrics as lyrics_mod
from mvdirector.config import CONFIG_FILENAME, PrepareConfig, ProjectConfig
from mvdirector.errors import ConfigError
from mvdirector.models import BOTH_VOICES, Analysis, TrackProfile, VideoStyle, VocalMode
from mvdirector.project import Project
from mvdirector.store import atomic_write

log = logging.getLogger(__name__)

_STARTER_TITLES = frozenset({"", "my song", "untitled"})
_FAST_SYLLABLE_RATE = 4.2
_RAP_BURST_SYLLABLE_RATE = 4.5
_RAP_BURST_MIN_SECONDS = 4.0
_DROP_CONTRAST_MIN = 0.35
_DROP_PEAK_INTENSITY = 0.80
_DROP_MAX_VOCAL_SHARE = 0.15

_NEGATIVE_CLAUSE_RE = re.compile(r"(?:^|(?<=[,;\n]))\s*[-‑–—]\s*[A-Za-z][^,;\n]*")

# Explicit language mentions in positive style prompts or section headers -> ISO-639-1 codes.
_STYLE_LANG_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("en", re.compile(r"\benglish\b", re.IGNORECASE)),
    ("fr", re.compile(r"\b(?:french|français|francais)\b", re.IGNORECASE)),
    ("es", re.compile(r"\b(?:spanish|español|espanol|spanglish)\b", re.IGNORECASE)),
    ("ko", re.compile(r"\b(?:korean|hangul)\b", re.IGNORECASE)),
    (
        "zh",
        re.compile(
            r"\b(?:mandarin|cantonese|chinese|mandopop|cantopop|c-pop|cpop)\b", re.IGNORECASE
        ),
    ),
    ("ja", re.compile(r"\b(?:japanese|j-pop|jpop)\b", re.IGNORECASE)),
    ("id", re.compile(r"\b(?:indonesian|bahasa indonesia)\b", re.IGNORECASE)),
    ("ms", re.compile(r"\b(?:malay|bahasa melayu)\b", re.IGNORECASE)),
    ("pt", re.compile(r"\b(?:portuguese)\b", re.IGNORECASE)),
    ("it", re.compile(r"\b(?:italian|italiano)\b", re.IGNORECASE)),
    ("de", re.compile(r"\b(?:german|deutsch)\b", re.IGNORECASE)),
    ("tl", re.compile(r"\b(?:tagalog|filipino)\b", re.IGNORECASE)),
    ("ru", re.compile(r"\b(?:russian)\b", re.IGNORECASE)),
    ("ar", re.compile(r"\b(?:arabic)\b", re.IGNORECASE)),
    ("hi", re.compile(r"\b(?:hindi)\b", re.IGNORECASE)),
    ("th", re.compile(r"\b(?:thai)\b", re.IGNORECASE)),
)
_EXPLICIT_MULTILINGUAL_RE = re.compile(
    r"\b(?:bilingual|multilingual|trilingual|code[- ]switching|spanglish)\b",
    re.IGNORECASE,
)

# High-precision function words for Latin-script sung lyrics (evaluated only on sung lines).
_LATIN_STOPWORDS: dict[str, frozenset[str]] = {
    "en": frozenset(
        {
            "the",
            "and",
            "you",
            "your",
            "with",
            "that",
            "this",
            "from",
            "have",
            "don't",
            "can't",
            "won't",
            "i'm",
            "we're",
            "through",
            "into",
            "beneath",
            "every",
            "never",
            "still",
        }
    ),
    "es": frozenset(
        {
            "los",
            "las",
            "una",
            "por",
            "para",
            "con",
            "sin",
            "como",
            "pero",
            "más",
            "todo",
            "nada",
            "noche",
            "fuego",
            "corazón",
            "siempre",
            "contigo",
            "cuando",
            "donde",
            "porque",
            "estoy",
            "tengo",
        }
    ),
    "fr": frozenset(
        {
            "je",
            "nous",
            "vous",
            "dans",
            "sur",
            "sous",
            "avec",
            "sans",
            "pour",
            "mais",
            "plus",
            "comme",
            "encore",
            "nuit",
            "lumière",
            "cœur",
            "toujours",
            "jamais",
            "suis",
            "cette",
        }
    ),
    "pt": frozenset(
        {
            "não",
            "você",
            "minha",
            "teu",
            "tua",
            "pelo",
            "pela",
            "tudo",
            "sempre",
            "coração",
            "noite",
            "fogo",
            "quando",
            "estou",
        }
    ),
    "de": frozenset(
        {
            "ich",
            "wir",
            "nicht",
            "und",
            "mit",
            "auf",
            "für",
            "aber",
            "noch",
            "nacht",
            "herz",
            "immer",
            "durch",
            "mein",
            "dein",
        }
    ),
    "it": frozenset(
        {
            "non",
            "che",
            "sono",
            "nel",
            "nella",
            "senza",
            "più",
            "notte",
            "cuore",
            "sempre",
            "fuoco",
            "mio",
            "tua",
        }
    ),
    "id": frozenset(
        {
            "aku",
            "kamu",
            "kau",
            "yang",
            "dan",
            "dari",
            "tidak",
            "tak",
            "jangan",
            "cinta",
            "hati",
            "malam",
            "selalu",
            "dalam",
            "dengan",
        }
    ),
}

_INSTRUMENTAL_STYLE_RE = re.compile(
    r"\b(?:instrumental only|no vocals|purely instrumental)\b", re.IGNORECASE
)
_GROUP_CUE_RE = re.compile(
    r"\b(?:choir|choral|ensemble|girl group|boy group|group vocals?|gang vocals?|"
    r"crowd chants?|multiple vocalists|trio|quartet|full female group|full male group)\b",
    re.IGNORECASE,
)
_DUET_CUE_RE = re.compile(
    r"\b(?:duet|male and female|female and male|call[- ]and[- ]response|two vocalists)\b",
    re.IGNORECASE,
)
_RAP_CUE_RE = re.compile(
    r"\b(?:rap|rapping|hip[- ]hop|trap|fast flow|double[- ]time|spoken word|"
    r"sprechgesang|staccato rap|chopper)\b",
    re.IGNORECASE,
)
_DROP_CUE_RE = re.compile(
    r"\b(?:vocal chops?|bass drop|beat drop|edm|trance|techno|house|dubstep|"
    r"drum and bass|dnb|hardstyle|big room|rave)\b",
    re.IGNORECASE,
)
_DROP_SECTION_RE = re.compile(r"\b(?:drop|bass drop|beat drop|vocal chops?)\b", re.IGNORECASE)
_EXTREME_VOCAL_RE = re.compile(
    r"\b(?:scream(?:ed|ing|s)?|growl(?:ed|ing|s)?|metalcore|death metal|harsh vocals?|"
    r"distorted vocals?|vocoder|robotic vocals?|whispered)\b",
    re.IGNORECASE,
)
_WALTZ_OR_34_RE = re.compile(r"(?:\b3/4\b|\bwaltz\b)", re.IGNORECASE)
_METER_68_RE = re.compile(r"\b6/8\b")
_METER_54_RE = re.compile(r"\b5/4\b")
_METER_78_RE = re.compile(r"\b7/8\b")
_FREE_TIME_RE = re.compile(r"\b(?:odd meter|rubato|free time|beatless)\b", re.IGNORECASE)

_VOWEL_GROUP_RE = re.compile(r"[aeiouyáéíóúàèìòùâêîôûäëïöüãõøœæіїєаеёиоуыэюя]+", re.IGNORECASE)
_LATIN_WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ']+")


@dataclass(frozen=True)
class Adjustment:
    """A single recommended or applied ``project.yaml`` parameter change."""

    key: str
    current: Any
    recommended: Any
    reason: str
    applied: bool = True


def is_starter_input(path: Path, template_name: str) -> bool:
    """True when ``path`` is missing or unchanged from ``templates/inputs/<template_name>``."""
    if not path.is_file():
        return True
    try:
        starter = (
            resources.files("mvdirector") / "templates" / "inputs" / template_name
        ).read_text(encoding="utf-8")
        current = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError):
        return False
    return current.strip() == starter.strip()


def read_optional_project_inputs(project: Project) -> tuple[str, str]:
    """Return ``(lyrics_text, style_text)`` for non-starter project input files, or ``''``."""
    inputs = project.config.inputs
    texts: list[str] = []
    for rel, template_name in ((inputs.lyrics, "lyrics.txt"), (inputs.style, "style.md")):
        if not rel or not rel.strip():
            texts.append("")
            continue
        target = project.path(rel.strip())
        if not target.is_file() or is_starter_input(target, template_name):
            texts.append("")
            continue
        try:
            texts.append(target.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeDecodeError):
            texts.append("")
    return texts[0], texts[1]


def strip_negative_style_tags(style_text: str) -> str:
    """Drop Suno negative exclusions (e.g. ``, -male vocals, ‑trap``) before positive matching."""
    return _NEGATIVE_CLAUSE_RE.sub("", style_text)


def _script_languages(text: str) -> list[str]:
    found: list[str] = []
    has_kana = bool(re.search(r"[\u3040-\u309F\u30A0-\u30FF]", text))
    if re.search(r"[\uAC00-\uD7AF\u1100-\u11FF\u3130-\u318F]", text):
        found.append("ko")
    if has_kana:
        found.append("ja")
    if re.search(r"[\u4E00-\u9FFF]", text) and not has_kana:
        found.append("zh")
    if re.search(r"[\u0400-\u04FF]", text):
        found.append("ru")
    if re.search(r"[\u0600-\u06FF]", text):
        found.append("ar")
    if re.search(r"[\u0E00-\u0E7F]", text):
        found.append("th")
    if re.search(r"[\u0900-\u097F]", text):
        found.append("hi")
    return found


def detect_languages(
    lyrics_text: str = "",
    style_text: str = "",
    *,
    title_hint: str = "",
    transcript_language: str | None = None,
    transcript_multilingual: bool = False,
) -> tuple[list[str], bool]:
    """Return ``(iso_languages, is_multilingual)`` from lyrics, style cues, and transcript hints."""
    pos_style = strip_negative_style_tags(style_text)
    section_headers = " | ".join(lyrics_mod.extract_section_tags(lyrics_text))
    sung_lines = [ln.text for ln in lyrics_mod.parse_lyrics(lyrics_text)]
    sung_text = "\n".join(sung_lines)

    langs: list[str] = []

    def add_lang(code: str) -> None:
        cleaned = code.strip().lower()
        if re.fullmatch(r"[a-z]{2,3}", cleaned) and cleaned not in langs:
            langs.append(cleaned)

    meta_text = f"{pos_style}\n{section_headers}"
    if re.search(r"\bspanglish\b", meta_text, re.IGNORECASE):
        add_lang("es")
        add_lang("en")
    for code, pattern in _STYLE_LANG_PATTERNS:
        if pattern.search(meta_text):
            add_lang(code)

    for code in _script_languages(sung_text):
        add_lang(code)

    # Evaluate Latin-script function words strictly on sung lyrics (never on [Section] headers).
    latin_words = [w.strip("'").casefold() for w in _LATIN_WORD_RE.findall(sung_text)]
    latin_words = [w for w in latin_words if w]
    if latin_words:
        total_latin = len(latin_words)
        scored: list[tuple[int, str]] = []
        for code, vocab in _LATIN_STOPWORDS.items():
            hits = sum(1 for w in latin_words if w in vocab)
            distinct = len({w for w in latin_words if w in vocab})
            if (
                hits >= 4
                or distinct >= 3
                or (total_latin <= 12 and distinct >= 2 and hits / total_latin >= 0.25)
            ):
                scored.append((hits, code))
        for _, code in sorted(scored, reverse=True):
            add_lang(code)

    # Pre-transcription fallback: when no lyrics exist yet, check title/filename scripts.
    title_script_hint = False
    clean_title = title_hint.strip()
    if not sung_text.strip() and clean_title.casefold() not in _STARTER_TITLES:
        title_scripts = _script_languages(clean_title)
        if title_scripts:
            has_latin_title = bool(re.search(r"[A-Za-z]{2,}", clean_title))
            if has_latin_title:
                add_lang("en")
            for code in title_scripts:
                add_lang(code)
            title_script_hint = len(title_scripts) > 1 or has_latin_title

    if transcript_language:
        add_lang(transcript_language)

    if not langs:
        langs.append("en")

    multilingual = (
        len(langs) > 1
        or bool(_EXPLICIT_MULTILINGUAL_RE.search(meta_text))
        or transcript_multilingual
        or title_script_hint
    )
    return langs, multilingual


def _line_syllables(text: str) -> int:
    """Approximate sung syllable count across syllabic (CJK/Kana/Hangul) and alphabetic scripts."""
    syllabic = sum(1 for ch in text if lyrics_mod.is_syllabic_char(ch))
    non_syllabic = "".join(" " if lyrics_mod.is_syllabic_char(ch) else ch for ch in text)
    alpha_syllables = 0
    for token in lyrics_mod.tokenize(non_syllabic):
        if not any(ch.isalpha() for ch in token):
            continue
        groups = _VOWEL_GROUP_RE.findall(token)
        count = len(groups)
        if count > 1 and token.endswith("e") and not token.endswith(("ee", "le", "ie")):
            count -= 1
        alpha_syllables += max(1, count)
    return syllabic + alpha_syllables


def measure_syllable_rate(analysis: Analysis | None) -> tuple[float | None, bool]:
    """Return ``(overall_syllables_per_sec, has_fast_rap_burst)`` from aligned lyric timings."""
    if analysis is None or not analysis.lyric_lines:
        return None, False

    total_syl = 0
    total_dur = 0.0
    fast_dur = 0.0
    for line in analysis.lyric_lines:
        if line.timing_source != "aligned" or line.start is None or line.end is None:
            continue
        dur = line.end - line.start
        if dur < 0.8:
            continue
        syl = _line_syllables(line.text)
        if syl <= 0:
            continue
        total_syl += syl
        total_dur += dur
        if syl / dur >= _RAP_BURST_SYLLABLE_RATE:
            fast_dur += dur

    if total_dur < 3.0 or total_syl == 0:
        return None, False
    overall = round(total_syl / total_dur, 2)
    return overall, (overall >= _FAST_SYLLABLE_RATE or fast_dur >= _RAP_BURST_MIN_SECONDS)


def _measure_intensity_and_drops(analysis: Analysis | None) -> tuple[float | None, bool]:
    """Return ``(intensity_contrast, has_acoustic_instrumental_drops)``."""
    if analysis is None or len(analysis.bar_intensity) < 4:
        return None, False
    arr = np.asarray(analysis.bar_intensity, dtype=np.float64)
    contrast = round(float(np.percentile(arr, 90) - np.percentile(arr, 20)), 3)
    contrast = min(1.0, max(0.0, contrast))

    if len(analysis.bars) < 8 or contrast < _DROP_CONTRAST_MIN or not analysis.vocal_windows:
        return contrast, False

    count = min(len(analysis.bars), len(analysis.bar_intensity))
    edges = [*analysis.bars[:count], analysis.duration]
    peak_bars = 0
    non_vocal_peak_bars = 0
    for idx in range(count):
        if analysis.bar_intensity[idx] < _DROP_PEAK_INTENSITY:
            continue
        start, end = edges[idx], edges[idx + 1]
        bar_dur = end - start
        if bar_dur <= 0:
            continue
        peak_bars += 1
        vocal_sec = sum(w.overlap(start, end) for w in analysis.vocal_windows)
        if vocal_sec / bar_dur < _DROP_MAX_VOCAL_SHARE:
            non_vocal_peak_bars += 1

    has_drops = peak_bars >= 2 and (non_vocal_peak_bars / peak_bars) >= 0.25
    return contrast, has_drops


def _detect_vocal_mode(
    pos_style: str,
    section_tags: list[str],
    sung_line_count: int,
    analysis: Analysis | None,
) -> tuple[VocalMode, str | None]:
    combined_cues = f"{pos_style}\n" + " | ".join(section_tags)
    if _INSTRUMENTAL_STYLE_RE.search(pos_style):
        return "instrumental", "style prompt marks track as instrumental"
    if (
        section_tags
        and sung_line_count == 0
        and all(
            re.fullmatch(r"instrumental(?:\s+.*)?", t.strip(), re.IGNORECASE) for t in section_tags
        )
    ):
        return "instrumental", "lyrics contain only [Instrumental] sections"
    if analysis is not None and analysis.duration > 0:
        vocal_cov = sum(w.duration for w in analysis.vocal_windows) / analysis.duration
        if vocal_cov < 0.05 and len(analysis.lyric_lines) == 0:
            return "instrumental", f"near-zero vocal activity ({vocal_cov * 100:.0f}% coverage)"

    if _GROUP_CUE_RE.search(combined_cues):
        return "group", "group/choir/chant vocals indicated in style or section cues"

    named_singers: set[str] = set()
    has_ensemble_tag = False
    for tag in section_tags:
        found = lyrics_mod.section_singers(tag)
        if found is None:
            has_ensemble_tag = True
        else:
            named_singers |= found
    if analysis is not None:
        for line in analysis.lyric_lines:
            if line.section:
                found = lyrics_mod.section_singers(line.section)
                if found is None:
                    has_ensemble_tag = True
                else:
                    named_singers |= found

    if len(named_singers) >= 3:
        return "group", f"{len(named_singers)} distinct singers named in section tags"
    if _DUET_CUE_RE.search(combined_cues) or len(named_singers) == 2 or has_ensemble_tag:
        return "duet", "two-singer / duet cues detected in style or section tags"

    if analysis is not None and analysis.vocal_turns:
        dur_by_voice: dict[str, float] = {}
        for turn in analysis.vocal_turns:
            dur_by_voice[turn.voice] = dur_by_voice.get(turn.voice, 0.0) + turn.duration
        solo_durations = [d for v, d in dur_by_voice.items() if v != BOTH_VOICES]
        total_voiced = sum(dur_by_voice.values())
        if len(solo_durations) >= 2 and total_voiced > 0:
            minor_share = min(solo_durations) / total_voiced
            if minor_share >= 0.15 or BOTH_VOICES in dur_by_voice:
                return "duet", "acoustic diarization detected 2 active vocal registers"

    return "solo", None


def profile_track(
    *,
    lyrics_text: str = "",
    style_text: str = "",
    title_hint: str = "",
    analysis: Analysis | None = None,
    transcript_language: str | None = None,
    transcript_multilingual: bool = False,
) -> TrackProfile:
    """Build a ``TrackProfile`` from text cues and optional DSP/alignment ``Analysis``."""
    pos_style = strip_negative_style_tags(style_text)
    section_tags = lyrics_mod.extract_section_tags(lyrics_text)
    if analysis is not None and not lyrics_text.strip() and analysis.lyric_lines:
        lyrics_text = "\n".join(
            (f"[{ln.section}]\n{ln.text}" if ln.section else ln.text) for ln in analysis.lyric_lines
        )
        section_tags = [ln.section for ln in analysis.lyric_lines if ln.section]

    sung_lines = lyrics_mod.parse_lyrics(lyrics_text)
    languages, multilingual = detect_languages(
        lyrics_text,
        style_text,
        title_hint=title_hint,
        transcript_language=transcript_language,
        transcript_multilingual=transcript_multilingual,
    )
    reasons: list[str] = []
    if multilingual:
        reasons.append(f"multilingual / code-switching track ({', '.join(languages)})")
    elif languages and languages[0] != "en":
        reasons.append(f"non-English track ({languages[0]})")

    vocal_mode, vocal_reason = _detect_vocal_mode(
        pos_style, section_tags, len(sung_lines), analysis
    )
    if vocal_reason:
        reasons.append(vocal_reason)

    combined_cues = f"{pos_style}\n" + " | ".join(section_tags)
    syllable_rate, fast_burst = measure_syllable_rate(analysis)
    rap_cue = bool(_RAP_CUE_RE.search(combined_cues))
    rap_or_fast = rap_cue or fast_burst
    if rap_cue and syllable_rate is not None:
        reasons.append(f"rap / fast-flow cues ({syllable_rate:.1f} syllables/s)")
    elif rap_cue:
        reasons.append("rap / fast-flow cues in style or section headers")
    elif fast_burst and syllable_rate is not None:
        reasons.append(f"high measured vocal density ({syllable_rate:.1f} syllables/s)")

    contrast, acoustic_drops = _measure_intensity_and_drops(analysis)
    drop_cue = bool(
        _DROP_CUE_RE.search(pos_style) or any(_DROP_SECTION_RE.search(tag) for tag in section_tags)
    )
    drop_or_chop = (drop_cue or acoustic_drops) and vocal_mode != "instrumental"
    if drop_cue:
        reasons.append("EDM / drop / vocal-chop structure in style or section headers")
    elif acoustic_drops:
        reasons.append("high-intensity non-vocal drop bars detected in mix dynamics")

    extreme_vocals = bool(_EXTREME_VOCAL_RE.search(combined_cues))
    if extreme_vocals:
        reasons.append("whispered, screamed, or heavily processed vocal cues")

    beats_hint: int | None = None
    non_4_4 = False
    if _WALTZ_OR_34_RE.search(combined_cues):
        beats_hint = 3
        non_4_4 = True
        reasons.append("3/4 / waltz meter cue")
    elif _METER_68_RE.search(combined_cues):
        beats_hint = 3
        non_4_4 = True
        reasons.append("6/8 compound meter cue")
    elif _METER_54_RE.search(combined_cues):
        beats_hint = 5
        non_4_4 = True
        reasons.append("5/4 odd meter cue")
    elif _METER_78_RE.search(combined_cues):
        beats_hint = 7
        non_4_4 = True
        reasons.append("7/8 odd meter cue")
    if _FREE_TIME_RE.search(combined_cues):
        non_4_4 = True
        reasons.append("rubato / beatless / odd-meter timing cue")

    return TrackProfile(
        languages=languages,
        multilingual=multilingual,
        vocal_mode=vocal_mode,
        rap_or_fast_vocal=rap_or_fast,
        drop_or_chop_heavy=drop_or_chop,
        extreme_vocals=extreme_vocals,
        non_4_4_or_free_time=non_4_4,
        beats_per_bar_hint=beats_hint,
        syllables_per_second=syllable_rate,
        intensity_contrast=contrast,
        reasons=reasons,
    )


def _get_dotted(config: ProjectConfig, key: str) -> Any:
    section_name, field_name = key.split(".", 1)
    return getattr(getattr(config, section_name), field_name)


def _set_dotted(config: ProjectConfig, key: str, value: Any) -> None:
    section_name, field_name = key.split(".", 1)
    setattr(getattr(config, section_name), field_name, value)


def recommend_adjustments(
    config: ProjectConfig,
    profile: TrackProfile,
    *,
    force: bool = False,
) -> list[Adjustment]:
    """Compute parameter adjustments for ``profile`` relative to ``config``.

    When ``force`` is False (the default), settings that the user already customized away
    from ``ProjectConfig`` defaults are preserved and returned with ``applied=False``.
    """
    defaults = ProjectConfig()
    candidates: list[tuple[str, Any, str]] = []

    non_english = any(lang != "en" for lang in profile.languages)
    if profile.multilingual:
        candidates.append(
            (
                "prepare.whisper_multilingual",
                True,
                f"per-segment language detection for {', '.join(profile.languages)}",
            )
        )
    elif non_english and len(profile.languages) == 1:
        candidates.append(
            (
                "prepare.whisper_language",
                profile.languages[0],
                f"lock Whisper language to '{profile.languages[0]}'",
            )
        )

    if profile.multilingual or non_english:
        cur_model = config.prepare.whisper_model.lower()
        if force or cur_model in ("tiny", "base", "small") or cur_model.endswith(".en"):
            candidates.append(
                (
                    "prepare.whisper_model",
                    "large-v3-turbo",
                    "more accurate non-English / multilingual vocal alignment",
                )
            )

    if profile.beats_per_bar_hint is not None and profile.beats_per_bar_hint != 4:
        candidates.append(
            (
                "video.beats_per_bar",
                profile.beats_per_bar_hint,
                f"match {profile.beats_per_bar_hint}-beat meter grouping",
            )
        )

    wants_phrase_cuts = profile.rap_or_fast_vocal or profile.non_4_4_or_free_time
    if wants_phrase_cuts:
        why = (
            "avoid slicing dense rap / fast vocal phrases on bar lines"
            if profile.rap_or_fast_vocal
            else "follow vocal phrases on non-4/4 or free-time meter"
        )
        candidates.append(("video.cut_mode", "phrases", why))

    # Resolve what cut_mode will be in effect so dynamic_pacing is only enabled on bar grids.
    cut_mode_default = _get_dotted(defaults, "video.cut_mode")
    cur_cut_mode = _get_dotted(config, "video.cut_mode")
    effective_cut_mode = (
        "phrases"
        if wants_phrase_cuts and (force or cur_cut_mode == cut_mode_default)
        else cur_cut_mode
    )
    if effective_cut_mode == "bars" and (
        profile.drop_or_chop_heavy or profile.vocal_mode == "group"
    ):
        candidates.append(
            (
                "video.dynamic_pacing",
                True,
                "cut faster on high-intensity drops/choruses and snap to section boundaries",
            )
        )

    lipsync_enabled = config.planning.video_style is VideoStyle.LIPSYNC
    if lipsync_enabled and (profile.drop_or_chop_heavy or profile.extreme_vocals):
        candidates.append(
            (
                "video.lipsync_min_vocal_fraction",
                0.35,
                "filter brief vocal-chop bleed or whispered/FX fragments out of lip-sync shots",
            )
        )

    share_caps: list[tuple[float, str]] = []
    if lipsync_enabled:
        if profile.vocal_mode == "instrumental":
            share_caps.append((0.0, "instrumental track needs no on-camera lip-sync shots"))
        if profile.vocal_mode == "group":
            share_caps.append(
                (0.30, "group/choir vocals cannot be acoustically diarized past 2 soloists")
            )
        if profile.extreme_vocals:
            share_caps.append(
                (0.30, "whispered/harsh/FX vocals have lower lip-sync tracker reliability")
            )
        if profile.drop_or_chop_heavy:
            share_caps.append(
                (0.35, "reserve screen time for b-roll/performance on drops and vocal chops")
            )
        if profile.rap_or_fast_vocal:
            share_caps.append((0.35, "balance rapid rap phrasing with performance/b-roll cutaways"))
    if share_caps:
        best_share, share_reason = min(share_caps, key=lambda item: item[0])
        candidates.append(("planning.max_lipsync_share", best_share, share_reason))

    adjustments: list[Adjustment] = []
    for key, recommended, reason in candidates:
        current = _get_dotted(config, key)
        if current == recommended:
            continue
        default_val = _get_dotted(defaults, key)
        en_model_conflict = (
            key == "prepare.whisper_model"
            and isinstance(current, str)
            and current.lower().endswith(".en")
        )
        can_apply = force or current == default_val or en_model_conflict
        adjustments.append(
            Adjustment(
                key=key,
                current=current,
                recommended=recommended,
                reason=reason,
                applied=can_apply,
            )
        )
    return adjustments


def tune_prepare_config(
    config: ProjectConfig,
    *,
    lyrics_text: str = "",
    style_text: str = "",
    title_hint: str = "",
) -> tuple[PrepareConfig, list[Adjustment]]:
    """Return an auto-tuned ``PrepareConfig`` copy (and applied adjustments) before Whisper runs."""
    if not config.auto_tune:
        return config.prepare, []
    profile = profile_track(lyrics_text=lyrics_text, style_text=style_text, title_hint=title_hint)
    applied = [
        adj
        for adj in recommend_adjustments(config, profile)
        if adj.applied and adj.key.startswith("prepare.")
    ]
    if not applied:
        return config.prepare, []
    updated = config.model_copy(deep=True)
    for adj in applied:
        _set_dotted(updated, adj.key, adj.recommended)
    # Re-validate PrepareConfig invariants (e.g. model name & language format).
    validated_prepare = PrepareConfig.model_validate(updated.prepare.model_dump())
    return validated_prepare, applied


def tune_config_for_planning(
    config: ProjectConfig,
    analysis: Analysis,
    *,
    lyrics_text: str = "",
    style_text: str = "",
) -> tuple[ProjectConfig, list[Adjustment]]:
    """Return a tuned ``ProjectConfig`` copy for ``build_manifest`` without mutating ``config``."""
    if not config.auto_tune:
        return config, []
    # Re-profile when lyrics/style text are provided (e.g. edited after `mvd analyze`) or when
    # loading an older analysis.json that has no stored track_profile.
    if lyrics_text.strip() or style_text.strip() or analysis.track_profile is None:
        profile = profile_track(
            lyrics_text=lyrics_text,
            style_text=style_text,
            title_hint=config.title,
            analysis=analysis,
        )
    else:
        profile = analysis.track_profile

    applied = [
        adj
        for adj in recommend_adjustments(config, profile)
        if adj.applied and (adj.key.startswith("video.") or adj.key.startswith("planning."))
    ]
    if not applied:
        return config, []
    updated = config.model_copy(deep=True)
    for adj in applied:
        _set_dotted(updated, adj.key, adj.recommended)
    return updated, applied


def _format_yaml_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ConfigError(f"Cannot write non-finite float to {CONFIG_FILENAME}: {value!r}")
        return f"{value:.2f}".rstrip("0").rstrip(".") if value != 0.0 else "0.0"
    if isinstance(value, str):
        if re.fullmatch(r"[A-Za-z0-9._-]+", value) and value.lower() not in (
            "true",
            "false",
            "null",
            "yes",
            "no",
            "on",
            "off",
        ):
            return value
        dumped = yaml.safe_dump(value, default_style='"').splitlines()[0]
        return dumped
    raise ConfigError(
        f"Unsupported scalar type for {CONFIG_FILENAME} update: {type(value).__name__}"
    )


_TOP_SECTION_HEADER_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*):[ \t]*(?:#.*)?$")
_TOP_LEVEL_KEY_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*):")


def _update_yaml_scalar(text: str, section: str, field: str, formatted_val: str) -> str:
    lines = text.splitlines(keepends=True)
    sec_start: int | None = None
    sec_end = len(lines)
    for idx, raw in enumerate(lines):
        stripped_line = raw.rstrip("\r\n")
        if sec_start is None:
            m = _TOP_SECTION_HEADER_RE.match(stripped_line)
            if m and m.group(1) == section:
                sec_start = idx
        elif _TOP_LEVEL_KEY_RE.match(stripped_line):
            sec_end = idx
            break

    if sec_start is None:
        sep = "" if (not text or text.endswith("\n")) else "\n"
        prefix = "\n" if text.strip() else ""
        return f"{text}{sep}{prefix}{section}:\n  {field}: {formatted_val}\n"

    # Determine the section's direct child indentation so nested dicts are never matched.
    child_indent: str | None = None
    for idx in range(sec_start + 1, sec_end):
        stripped = lines[idx].strip()
        if stripped and not stripped.startswith("#"):
            leading = lines[idx][: len(lines[idx]) - len(lines[idx].lstrip(" \t"))]
            if leading and (child_indent is None or len(leading) < len(child_indent)):
                child_indent = leading

    indent = child_indent or "  "
    field_re = re.compile(
        rf"^{re.escape(indent)}{re.escape(field)}:[ \t]*([^#\r\n]*?)([ \t]*#[^\r\n]*)?(\r?\n?)$"
    )
    for idx in range(sec_start + 1, sec_end):
        m = field_re.match(lines[idx])
        if m:
            _old_val, comment, newline = m.groups()
            newline = newline or "\n"
            comment_part = ""
            if comment:
                gap = comment[: len(comment) - len(comment.lstrip())] or "  "
                comment_part = f"{gap}{comment.lstrip()}"
            lines[idx] = f"{indent}{field}: {formatted_val}{comment_part}{newline}"
            return "".join(lines)

    # Field is absent from an existing section: insert right after its last direct-child line
    # (or section header if empty), never in the middle of a nested block.
    insert_at = sec_start + 1
    in_nested = False
    for idx in range(sec_start + 1, sec_end):
        raw_line = lines[idx]
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            if in_nested:
                insert_at = idx + 1
            continue
        leading = raw_line[: len(raw_line) - len(raw_line.lstrip(" \t"))]
        if len(leading) == len(indent):
            insert_at = idx + 1
            in_nested = stripped.endswith(":")
        elif len(leading) > len(indent):
            insert_at = idx + 1
            in_nested = True
    if insert_at - 1 < len(lines) and not lines[insert_at - 1].endswith("\n"):
        lines[insert_at - 1] += "\n"
    lines.insert(insert_at, f"{indent}{field}: {formatted_val}\n")
    return "".join(lines)


def apply_adjustments_to_yaml(yaml_path: Path, adjustments: list[Adjustment]) -> list[Adjustment]:
    """Update ``project.yaml`` in place for ``applied`` adjustments while keeping comments."""
    to_write = [adj for adj in adjustments if adj.applied]
    if not to_write:
        return []
    if not yaml_path.is_file():
        raise ConfigError(f"Config file not found: {yaml_path}")
    text = yaml_path.read_text(encoding="utf-8")
    try:
        parsed_before = yaml.safe_load(text) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"{yaml_path} is not valid YAML: {exc}") from exc
    if not isinstance(parsed_before, dict):
        raise ConfigError(f"{yaml_path} must hold a YAML mapping at top level.")

    # When an English-only Whisper model (e.g. 'small.en') is kept while tuning sets a
    # non-English language or multilingual flag, upgrade the model to avoid an invalid config.
    write_keys = {adj.key for adj in to_write}
    if "prepare.whisper_model" not in write_keys:
        prep_before = parsed_before.get("prepare")
        cur_model = prep_before.get("whisper_model") if isinstance(prep_before, dict) else None
        needs_multilingual_model = any(
            (adj.key == "prepare.whisper_multilingual" and adj.recommended is True)
            or (adj.key == "prepare.whisper_language" and adj.recommended not in (None, "en"))
            for adj in to_write
        )
        if (
            needs_multilingual_model
            and isinstance(cur_model, str)
            and cur_model.lower().endswith(".en")
        ):
            to_write.append(
                Adjustment(
                    key="prepare.whisper_model",
                    current=cur_model,
                    recommended="large-v3-turbo",
                    reason="replace English-only '*.en' model for non-English / multilingual audio",
                    applied=True,
                )
            )

    updated_text = text
    for adj in to_write:
        section, field = adj.key.split(".", 1)
        if (
            isinstance(parsed_before.get(section), dict) is False
            and parsed_before.get(section) is not None
        ):
            raise ConfigError(f"Section '{section}' in {yaml_path} must be a mapping or null.")
        # If the section was `section: null`, `~`, or `{}`, normalize to `section:` header first.
        sec_empty_re = re.compile(
            rf"^{re.escape(section)}:[ \t]*(?:null|~|\{{[ \t]*\}})([ \t]*#[^\r\n]*)?$",
            re.MULTILINE,
        )
        updated_text = sec_empty_re.sub(rf"{section}:\1", updated_text)
        updated_text = _update_yaml_scalar(
            updated_text, section, field, _format_yaml_scalar(adj.recommended)
        )

    try:
        parsed_after = yaml.safe_load(updated_text) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"Updated {yaml_path} failed YAML validation: {exc}") from exc
    if not isinstance(parsed_after, dict):
        raise ConfigError(f"Updated {yaml_path} must hold a YAML mapping at top level.")
    for adj in to_write:
        section, field = adj.key.split(".", 1)
        sec_after = parsed_after.get(section)
        actual = sec_after.get(field) if isinstance(sec_after, dict) else None
        if actual != adj.recommended:
            raise ConfigError(
                f"Failed to update {adj.key} in {yaml_path}: "
                f"expected {adj.recommended!r}, got {actual!r}"
            )

    try:
        ProjectConfig.model_validate(parsed_after)
    except ValidationError as exc:
        raise ConfigError(f"Invalid configuration after tuning {yaml_path}:\n{exc}") from exc

    atomic_write(yaml_path, updated_text)
    return to_write


def render_tune_report(
    profile: TrackProfile,
    adjustments: list[Adjustment],
    *,
    applied_to_yaml: bool = False,
) -> str:
    """Format a concise human-readable track profile and parameter tuning summary."""
    langs = ", ".join(profile.languages) or "en"
    lang_label = f"{langs} (multilingual)" if profile.multilingual else langs
    traits: list[str] = [f"languages={lang_label}", f"vocal_mode={profile.vocal_mode}"]
    if profile.rap_or_fast_vocal:
        traits.append("rap/fast-vocal")
    if profile.drop_or_chop_heavy:
        traits.append("drop/chop-heavy")
    if profile.extreme_vocals:
        traits.append("extreme-vocals")
    if profile.non_4_4_or_free_time:
        meter = f"{profile.beats_per_bar_hint}/4" if profile.beats_per_bar_hint else "free/odd"
        traits.append(f"meter={meter}")
    if profile.syllables_per_second is not None:
        traits.append(f"rate={profile.syllables_per_second:.1f} syl/s")
    if profile.intensity_contrast is not None:
        traits.append(f"contrast={profile.intensity_contrast:.2f}")

    lines = [f"Track profile: {', '.join(traits)}"]
    for reason in profile.reasons:
        lines.append(f"  - {reason}")

    if not adjustments:
        lines.append("\nNo parameter changes needed (current project.yaml already matches track).")
        return "\n".join(lines)

    applied = [a for a in adjustments if a.applied]
    skipped = [a for a in adjustments if not a.applied]
    if applied:
        heading = "Applied to project.yaml:" if applied_to_yaml else "Recommended adjustments:"
        lines.append(f"\n{heading}")
        for adj in applied:
            lines.append(f"  {adj.key}: {adj.current!r} -> {adj.recommended!r}  ({adj.reason})")
    if skipped:
        lines.append("\nKept custom values (pass --force to override):")
        for adj in skipped:
            lines.append(
                f"  {adj.key}: kept {adj.current!r} (recommended {adj.recommended!r}: {adj.reason})"
            )
    return "\n".join(lines)
