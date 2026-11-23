# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Mix Mode planner: 1 distinct visual environment per chapter + 2-3 loopable shots.

Bypasses character sheets, lip-sync, and multi-shot narrative storyboarding so 30-to-90+ minute
mixes require only a small pool of reusable loop clips (or hero stills) per track segment.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, ConfigDict, Field, field_validator

from mvdirector.audio.cues import format_cues_summary
from mvdirector.audio.tracklist import resolve_chapters
from mvdirector.config import ProjectConfig
from mvdirector.errors import LLMError, LLMUnavailableError
from mvdirector.llm import LLMClient, text_block
from mvdirector.models import (
    Analysis,
    AnalyzedChapter,
    AssetState,
    AudioSlice,
    Framing,
    Manifest,
    MixChapter,
    ProjectMode,
    SetDesign,
    Shot,
    ShotType,
    VisualBible,
    validate_safe_id,
)
from mvdirector.planning import prompts
from mvdirector.planning.segmenter import frames_for

log = logging.getLogger(__name__)

_DEFAULT_NEGATIVE = (
    "people, human figures, faces, singers, text, subtitles, watermark, logo, "
    "blurry, deformed geometry, strobe flicker, abrupt cut"
)

# Deliberately not "cinematic": movie-trailer phrasing primes distilled (cfg 1) video models to
# fade in hallucinated opening credits and title cards.
_DEFAULT_STYLE = "Atmospheric ambient visual environments"

_VARIANT_FRAMINGS: tuple[Framing, ...] = (
    Framing.WIDE,
    Framing.MEDIUM,
    Framing.MEDIUM_CLOSE,
    Framing.WIDE,
    Framing.MEDIUM,
    Framing.MEDIUM_CLOSE,
)

_VARIANT_ANGLES: tuple[str, ...] = (
    "Sweeping wide architectural establishing composition in open space",
    "Mid-scale atmospheric composition emphasizing depth, surfaces, and shifting light",
    "Tighter textural detail composition with shallow depth of field and drifting particles",
    "Elevated symmetrical vista across the uninhabited environment",
    "Low-horizon perspective framing reflections and volumetric beams",
    "Layered foreground-to-background vista with gentle parallax depth",
)

_THEME_ARCHETYPES: tuple[tuple[str, str, str], ...] = (
    (
        "Flooded brutalist concrete cistern with mirror-still dark water and monumental arches",
        "deep cyan, slate blue, and cold silver highlights",
        "volumetric overhead shafts piercing drifting mist",
    ),
    (
        "Wind-carved red sandstone canyon framing a monolithic mirror obelisk at dusk",
        "burnt amber, terracotta, copper, and deep indigo twilight",
        "low golden-hour rim light with warm atmospheric haze",
    ),
    (
        "Rain-slicked neon atrium of glass walkways and suspended botanical terrariums at night",
        "magenta, electric violet, emerald foliage, and wet obsidian reflections",
        "soft diffused neon glow reflecting across wet glass and mist",
    ),
    (
        "High-altitude alpine observatory deck above a sea of moonlit clouds and aurora ribbons",
        "glacial teal, emerald aurora, crisp moonlight white, and midnight navy",
        "cool celestial glow with slow-shifting aurora reflections on brushed metal",
    ),
    (
        "Subterranean volcanic basalt gallery lit by channels of slow-moving molten glass",
        "obsidian black, molten crimson, ember orange, and smoke grey",
        "warm under-lit magma glow contrasting against cool matte stone",
    ),
    (
        "Sun-bleached salt flat horizon with kinetic brass rings rotating in shallow water",
        "ivory white, champagne gold, pastel sky blue, and soft rose",
        "crisp high-key horizon sunlight with crisp mirror reflections",
    ),
    (
        "Ancient moss-covered cedar courtyard shrouded in bamboo mist and floating paper lanterns",
        "forest jade, warm lantern amber, charcoal wood, and silver fog",
        "warm practical lantern pools diffusing into cool twilight rain mist",
    ),
    (
        "Industrial laser cathedral inside a vast turbine hall with geometric light planes",
        "cobalt blue, tungsten white, gunmetal grey, and ruby laser accents",
        "structured volumetric light curtains sweeping slowly through haze",
    ),
    (
        "Bioluminescent deep-ocean kelp cavern with translucent drifting light spores",
        "abyssal ultramarine, bioluminescent turquoise, coral violet, and seafoam",
        "caustic underwater light ripples and soft internal luminescence",
    ),
    (
        "Minimalist concrete desert pavilion overlooking rolling silver sand dunes under starlight",
        "warm sand, pewter silver, travertine cream, and deep violet night sky",
        "recessed architectural warm strip lighting meeting cool starlight",
    ),
    (
        "Art-deco glass conservatory filled with giant crystalline prisms and tropical ferns",
        "emerald green, prismatic gold, sapphire refraction, and warm bronze",
        "refracted sunbeams casting slow rainbow caustics through humid air",
    ),
    (
        "Cybernetic server canyon of towering matte-black monoliths with pulsing fiber-optic seams",
        "matte carbon, ice cyan, warm amber telemetry, and deep shadow",
        "rhythmic traveling light filaments reflecting off polished dark floor",
    ),
)


class _ChapterThemeReply(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    environment: str
    palette: str = ""
    lighting: str = ""
    motion_style: str = ""
    clip_prompts: list[str] = Field(default_factory=list)

    @field_validator("id")
    @classmethod
    def _check_id(cls, value: str) -> str:
        return validate_safe_id(value, "chapter id")


class _MixThemesBatchReply(BaseModel):
    model_config = ConfigDict(extra="ignore")

    style: str = ""
    negative_prompt: str = ""
    chapters: list[_ChapterThemeReply] = Field(default_factory=list)


def quantize_loop_clip_seconds(
    bpm: float,
    beats_per_bar: int,
    target_clip_seconds: float,
    xfade_seconds: float,
    max_clip_seconds: float,
) -> float:
    """Pick a clip duration ``stride + xfade_seconds`` aligned to whole bars when possible."""
    upper = max(4.0, max_clip_seconds)
    lower = min(4.0, upper)
    min_stride = max(1.5, xfade_seconds * 1.25)
    target_stride = max(min_stride, target_clip_seconds - xfade_seconds)

    if bpm > 0.0:
        beat_sec = 60.0 / bpm
        bar_sec = beat_sec * max(1, beats_per_bar)
        bar_candidates = [
            n * bar_sec
            for n in range(1, 17)
            if min_stride <= n * bar_sec and lower <= (n * bar_sec + xfade_seconds) <= upper
        ]
        if bar_candidates:
            best_stride = min(bar_candidates, key=lambda s: abs(s - target_stride))
            return round(best_stride + xfade_seconds, 3)

        beat_candidates = [
            n * beat_sec
            for n in range(2, 33)
            if min_stride <= n * beat_sec and lower <= (n * beat_sec + xfade_seconds) <= upper
        ]
        if beat_candidates:
            best_stride = min(beat_candidates, key=lambda s: abs(s - target_stride))
            return round(best_stride + xfade_seconds, 3)

    return round(min(upper, max(lower, target_clip_seconds)), 3)


def _motion_for_energy(energy: float, custom: str = "") -> str:
    if custom.strip():
        base = custom.strip().rstrip(".")
        return f"{base}. Smooth continuous ambient motion, uninhabited space."
    if energy >= 0.72:
        return (
            "Steady forward camera glide as volumetric light beams, reflections, and atmospheric "
            "particles flow briskly and continuously through the uninhabited space."
        )
    if energy >= 0.42:
        return (
            "Gentle, steady camera drift while light, mist, and surface reflections shift "
            "smoothly and continuously across the uninhabited environment."
        )
    return (
        "Slow, meditative camera drift as soft mist, ambient light, and subtle reflections "
        "breathe calmly across the uninhabited scene."
    )


def _fallback_theme(
    chapter: AnalyzedChapter,
    clips_per_segment: int,
) -> _ChapterThemeReply:
    arch_env, arch_palette, arch_light = _THEME_ARCHETYPES[chapter.index % len(_THEME_ARCHETYPES)]
    tags = [*chapter.mood_tags, *chapter.genre_tags]
    tag_clause = f" ({', '.join(tags[:4])} atmosphere)" if tags else ""
    env = (chapter.theme_prompt or f"{arch_env}{tag_clause}").strip()
    motion = _motion_for_energy(chapter.energy)

    variants: list[str] = list(chapter.clip_prompts[:clips_per_segment])
    while len(variants) < clips_per_segment:
        v_idx = len(variants)
        angle = _VARIANT_ANGLES[v_idx % len(_VARIANT_ANGLES)]
        variants.append(f"{angle} of {env.rstrip('.')}. Completely uninhabited space.")
    return _ChapterThemeReply(
        id=chapter.id,
        environment=env,
        palette=arch_palette,
        lighting=arch_light,
        motion_style=motion,
        clip_prompts=variants,
    )


def _ensure_analyzed_chapters(config: ProjectConfig, analysis: Analysis) -> list[AnalyzedChapter]:
    if analysis.chapters:
        return list(analysis.chapters)
    resolved = resolve_chapters(
        None,
        analysis.duration,
        default_segment_seconds=config.mix.default_segment_seconds,
        snap_candidates=analysis.bars,
    )
    bpm = analysis.tempo_bpm if analysis.tempo_bpm > 0 else 120.0
    synthesized: list[AnalyzedChapter] = []
    for item in resolved:
        near_bars = [b for b in analysis.bars if item.start <= b < item.end]
        downbeat = near_bars[0] if near_bars else item.start
        synthesized.append(
            AnalyzedChapter(
                id=item.id,
                index=item.index,
                start=item.start,
                end=item.end,
                title=item.title,
                artist=item.artist,
                bpm=bpm,
                downbeat_offset=downbeat,
                energy=0.5,
            )
        )
    return synthesized


def _format_lyric_excerpt(lyrics: str | None, max_chars: int = 280) -> str:
    """Collapse whitespace and control chars before truncating a chapter lyric excerpt."""
    if not lyrics:
        return ""
    cleaned = " ".join("".join(ch if ch.isprintable() else " " for ch in lyrics).split())
    if not cleaned:
        return ""
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[:max_chars].rstrip() + "..."


def _request_llm_themes(
    chapters: list[AnalyzedChapter],
    style_text: str,
    clips_per_segment: int,
    batch_size: int,
    llm: LLMClient,
) -> tuple[str, str, dict[str, _ChapterThemeReply]]:
    by_id: dict[str, _ChapterThemeReply] = {}
    chosen_style = ""
    chosen_neg = ""
    brief = style_text.strip() or _DEFAULT_STYLE

    for offset in range(0, len(chapters), batch_size):
        batch = chapters[offset : offset + batch_size]
        prev_notes = [
            f"- {cid}: palette={reply.palette!r}, env={reply.environment[:80]!r}"
            for cid, reply in list(by_id.items())[-3:]
        ]
        chapter_lines: list[str] = []
        for ch in batch:
            tags = ", ".join([*ch.genre_tags, *ch.mood_tags]) or "electronic / ambient"
            key_note = f" | key={ch.key}" if ch.key else ""
            cue_summary = format_cues_summary(ch.cues)
            cue_note = f" | cues={cue_summary}" if cue_summary else ""
            hint = f" | user_theme_hint={ch.theme_prompt!r}" if ch.theme_prompt else ""
            excerpt = _format_lyric_excerpt(ch.lyrics)
            lyric_note = f" | lyrics_excerpt={excerpt!r}" if excerpt else ""
            chapter_lines.append(
                f"- id={ch.id} | time={ch.start:.0f}-{ch.end:.0f}s | title={ch.title!r} | "
                f"bpm={ch.bpm:.0f}{key_note} | energy={ch.energy:.2f} | tags={tags}"
                f"{cue_note}{hint}{lyric_note}"
            )
        prompt_parts = [
            f"GLOBAL STYLE BRIEF:\n{brief}",
            f"CLIPS PER CHAPTER: {clips_per_segment}",
        ]
        if prev_notes:
            prompt_parts.append(
                "PREVIOUS CHAPTER LOOKS (avoid repeating palette/environment):\n"
                + "\n".join(prev_notes)
            )
        prompt_parts.append("CHAPTERS TO DIRECT:\n" + "\n".join(chapter_lines))

        try:
            reply = llm.complete_json(
                system=prompts.MIX_THEMES_SYSTEM,
                content=[text_block("\n\n".join(prompt_parts))],
                schema=_MixThemesBatchReply,
            )
        except LLMUnavailableError as exc:
            log.warning(
                "Mix theme LLM unavailable at batch %d..%d (%s); using deterministic themes for "
                "remaining chapters.",
                offset + 1,
                offset + len(batch),
                exc,
            )
            break
        except LLMError as exc:
            log.warning(
                "Mix theme LLM batch %d..%d failed (%s); falling back to deterministic themes.",
                offset + 1,
                offset + len(batch),
                exc,
            )
            continue

        if reply.style.strip() and not chosen_style:
            chosen_style = reply.style.strip()
        if reply.negative_prompt.strip() and not chosen_neg:
            chosen_neg = reply.negative_prompt.strip()
        for item in reply.chapters:
            by_id[item.id] = item

    return chosen_style, chosen_neg, by_id


def _chapter_sample_starts(
    ch_in: AnalyzedChapter,
    clip_dur: float,
    clips_per_seg: int,
    total_duration: float,
) -> list[float]:
    """Pick ``clips_per_seg`` audio sample starts, anchoring to structural cues when available."""
    avail_span = max(0.0, ch_in.duration - clip_dur)
    lo = max(0.0, min(ch_in.start, max(0.0, total_duration - clip_dur)))
    hi = max(lo, min(max(0.0, total_duration - clip_dur), ch_in.start + avail_span))
    min_sep = max(1.0, clip_dur * 0.5)

    chosen: list[float] = []
    if ch_in.cues is not None:
        raw_landmarks = (
            ch_in.cues.main_peak,
            ch_in.cues.first_lift,
            ch_in.cues.second_peak,
            ch_in.cues.breakdown,
        )
        for mark in raw_landmarks:
            if len(chosen) >= clips_per_seg:
                break
            if mark is None or not (ch_in.start <= mark <= ch_in.end):
                continue
            cand = round(min(hi, max(lo, mark - clip_dur * 0.5)), 3)
            if all(abs(cand - prev) >= min_sep for prev in chosen):
                chosen.append(cand)

    uniform = [
        round(
            min(
                max(0.0, total_duration - clip_dur),
                ch_in.start
                + avail_span * ((i + 0.5) / clips_per_seg if clips_per_seg > 1 else 0.0),
            ),
            3,
        )
        for i in range(clips_per_seg)
    ]
    for cand in uniform:
        if len(chosen) >= clips_per_seg:
            break
        if all(abs(cand - prev) >= min_sep for prev in chosen):
            chosen.append(cand)
    for cand in uniform:
        if len(chosen) >= clips_per_seg:
            break
        if cand not in chosen:
            chosen.append(cand)
    for cand in uniform:
        if len(chosen) >= clips_per_seg:
            break
        chosen.append(cand)
    return sorted(chosen[:clips_per_seg])


def build_mix_manifest(
    config: ProjectConfig,
    analysis: Analysis,
    style_text: str,
    llm: LLMClient | None,
) -> Manifest:
    """Create a ``ProjectMode.MIX`` manifest with 1 theme + ``clips_per_segment`` shots/chapter."""
    chapters_in = _ensure_analyzed_chapters(config, analysis)
    clips_per_seg = config.mix.clips_per_segment
    clean_style = " ".join(style_text.split()) or _DEFAULT_STYLE

    llm_style = ""
    llm_neg = ""
    directed: dict[str, _ChapterThemeReply] = {}
    if llm is not None:
        llm_style, llm_neg, directed = _request_llm_themes(
            chapters_in,
            style_text,
            clips_per_seg,
            max(1, config.llm.shots_per_request),
            llm,
        )
    else:
        log.info("LLM disabled: generating deterministic per-chapter themes for mix mode.")

    bible_settings: list[SetDesign] = []
    mix_chapters: list[MixChapter] = []
    shots: list[Shot] = []

    for ch_in in chapters_in:
        fallback = _fallback_theme(ch_in, clips_per_seg)
        item = directed.get(ch_in.id)

        environment = (
            ch_in.theme_prompt
            or (item.environment.strip() if item and item.environment.strip() else "")
            or fallback.environment
        )
        palette = (
            item.palette.strip() if item and item.palette.strip() else ""
        ) or fallback.palette
        lighting = (
            item.lighting.strip() if item and item.lighting.strip() else ""
        ) or fallback.lighting
        motion_style = _motion_for_energy(
            ch_in.energy, item.motion_style if item and item.motion_style else ""
        )

        # User-provided clip_prompts win first, then LLM variants, then deterministic angles.
        raw_variants: list[str] = list(ch_in.clip_prompts[:clips_per_seg])
        if item and len(raw_variants) < clips_per_seg:
            for cand in item.clip_prompts:
                if cand.strip() and len(raw_variants) < clips_per_seg:
                    raw_variants.append(cand.strip())
        while len(raw_variants) < clips_per_seg:
            raw_variants.append(fallback.clip_prompts[len(raw_variants)])

        clip_dur = quantize_loop_clip_seconds(
            ch_in.bpm,
            config.video.beats_per_bar,
            config.mix.clip_seconds,
            config.mix.xfade_seconds,
            config.video.max_clip_seconds,
        )
        clip_dur = round(min(clip_dur, analysis.duration), 3)
        num_frames = frames_for(
            clip_dur,
            config.video.fps,
            config.video.frame_multiple,
            config.video.frame_offset,
        )

        chapter_shot_ids: list[str] = []
        sample_starts = _chapter_sample_starts(ch_in, clip_dur, clips_per_seg, analysis.duration)
        for v_idx, sample_start in enumerate(sample_starts):
            shot_id = f"{ch_in.id}_c{v_idx + 1:02d}"
            chapter_shot_ids.append(shot_id)
            sample_end = round(min(analysis.duration, sample_start + clip_dur), 3)
            framing = _VARIANT_FRAMINGS[v_idx % len(_VARIANT_FRAMINGS)]

            shots.append(
                Shot(
                    id=shot_id,
                    index=len(shots),
                    start=sample_start,
                    end=sample_end,
                    type=ShotType.BROLL,
                    framing=framing,
                    section=ch_in.title,
                    chapter_id=ch_in.id,
                    setting=ch_in.id,
                    intensity=ch_in.energy,
                    still_prompt=raw_variants[v_idx],
                    motion_prompt=motion_style,
                    audio_slice=AudioSlice(
                        path=f"assets/audio_slices/{shot_id}.wav",
                        start=sample_start,
                        end=sample_end,
                        head_handle=0.0,
                    ),
                    num_frames=num_frames,
                    still=AssetState(attempts_allowed=config.retry.max_attempts),
                    video=AssetState(attempts_allowed=config.retry.max_attempts),
                )
            )

        bible_settings.append(
            SetDesign(
                name=ch_in.id,
                sections=[ch_in.title],
                description=environment,
                movement=motion_style,
            )
        )
        mix_chapters.append(
            MixChapter(
                id=ch_in.id,
                index=ch_in.index,
                start=ch_in.start,
                end=ch_in.end,
                title=ch_in.title,
                artist=ch_in.artist,
                bpm=ch_in.bpm,
                downbeat_offset=ch_in.downbeat_offset,
                energy=ch_in.energy,
                mood_tags=list(ch_in.mood_tags),
                genre_tags=list(ch_in.genre_tags),
                environment=environment,
                palette=palette,
                lighting=lighting,
                motion_style=motion_style,
                shot_ids=chapter_shot_ids,
            )
        )

    bible = VisualBible(
        concept=(
            f"Long-form continuous visual mix across {len(mix_chapters)} distinct track chapters."
        ),
        style=llm_style or clean_style,
        negative_prompt=llm_neg or _DEFAULT_NEGATIVE,
        settings=bible_settings,
        rules=[
            "Uninhabited environments only; no performers, faces, or singing.",
            "Every clip uses continuous loop-friendly ambient motion with no cuts or text.",
        ],
    )
    log.info(
        "Planned mix manifest: %d chapter(s), %d reusable loop unit(s) (visual_source=%s).",
        len(mix_chapters),
        len(shots),
        config.mix.visual_source,
    )
    return Manifest(
        mode=ProjectMode.MIX,
        title=config.title,
        duration=analysis.duration,
        fps=config.video.fps,
        visual_bible=bible,
        chapters=mix_chapters,
        shots=shots,
    )
