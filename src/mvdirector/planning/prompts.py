# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Prompt text for the planner and for composing generation prompts.

Kept together so wording can be tuned without touching pipeline logic. The vision-check
prompt lives next to its code in ``qc/vision.py``.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import TYPE_CHECKING

from mvdirector.models import Character, Framing, Manifest, ProjectMode, Shot, ShotType
from mvdirector.planning.world_state import build_ledger_from_facts, filter_look_for_state

if TYPE_CHECKING:
    from mvdirector.config import VideoConfig

# Still-photography tokens that can cause I2V/FLF2V video models (e.g. LTX) to freeze or crossfade.
# A bare "photograph" is deliberately absent: it is also an ordinary prop noun ("she studies a
# faded photograph"), and deleting it would corrupt the action description.
# Credit: GeekatplayStudio/ComfyUI-Music-to-Video (nodes.py: STILL_WORDS).
_STILL_WORDS_RE = re.compile(
    r"\b(?:film\s+still(?:\s+photo(?:graph)?)?|still\s+frame|still\s+photo(?:graph)?|"
    r"photographic\s+still|static\s+photo|freeze\s+frame)\b[:,;]?\s*",
    re.IGNORECASE,
)

# Rear-view / over-the-shoulder cues where frontal facial features cause actors to spin around.
# Credit: afj176/open-cinema-studio (app/staging.py).
_REAR_VIEW_RE = re.compile(
    r"\b(?:over[\s-]+(?:the[\s-]+)?(?:[a-z]+'s\s+)?shoulder|from\s+behind|rear\s+view|"
    r"backs?\s+of\s+(?:their|his|her)\s+heads?|seen\s+from\s+behind|facing\s+away|"
    r"backs?\s+to\s+(?:the\s+)?(?:camera|lens)|backs?\s+turned)\b",
    re.IGNORECASE,
)
_FRONTAL_FACIAL_CLAUSE_RE = re.compile(
    r"\b(?:eyes?|eyebrows?|brows?|nose|lips?|mouth|jawline|cheeks?|freckles?|beauty\s+mark|"
    r"eyeliner|lipstick|makeup|pupil|gaze)\b",
    re.IGNORECASE,
)
_LOWER_BODY_STRICT_RE = re.compile(
    r"\b(?:trousers?|pants?|jeans|skirts?|shorts|boots?|shoes?|sandals?|sneakers?|heels?|"
    r"slippers?|socks?|footwear|barefoot|bare\s+feet|feet|anklets?|soles?|leggings?|tights|"
    r"stockings?|hemlines?|hem|belts?|gloves?|wrists?|wristbands?|bracelets?|"
    r"high[\s-]waisted|knee[\s-]high|mid[\s-]calf|ankle[\s-]length|calf[\s-]length|"
    r"floor[\s-]sweeping|full[\s-]length|gathered\s+skirt)\b",
    re.IGNORECASE,
)
_LOWER_BODY_MEDIUM_RE = re.compile(
    r"\b(?:trousers?|pants?|jeans|skirts?|shorts|boots?|shoes?|sandals?|sneakers?|heels?|"
    r"slippers?|socks?|footwear|barefoot|bare\s+feet|feet|anklets?|soles?|leggings?|tights|"
    r"stockings?|hemlines?|hem|knee[\s-]high|mid[\s-]calf|ankle[\s-]length|calf[\s-]length|"
    r"floor[\s-]sweeping|full[\s-]length)\b",
    re.IGNORECASE,
)
_UPPER_BODY_WARDROBE_RE = re.compile(
    r"\b(?:coat|overcoat|jacket|blazer|hoodie|sweater|turtleneck|shirt|blouse|dress|gown|"
    r"top|vest|corset|collar|neckline|lapels?|sleeves?|shoulders?|hood|scarf|hat|cap|"
    r"earrings?|necklace|pendant|choker|locket)\b",
    re.IGNORECASE,
)
_EXPLICIT_TIGHT_FRAMING_RE = re.compile(
    r"\b(?:close[\s_-]*up|head[\s-]+and[\s-]+shoulders|chest[\s-]+up|medium[\s_-]*close|"
    r"tight\s+portrait|bust\s+shot)\b",
    re.IGNORECASE,
)
_SILHOUETTE_RE = re.compile(
    r"\b(?:(?:in|as\s+a|backlit|dark|stark|shadowy|full[- ]body)\s+silhouette|"
    r"silhouetted\s+(?:against|in|by))\b",
    re.IGNORECASE,
)


def strip_still_words(motion_text: str) -> str:
    """Remove still-photography terms from motion prompts so video models stay dynamic.

    Credits & Attribution:
        Inspired by ``GeekatplayStudio/ComfyUI-Music-to-Video``
        (https://github.com/GeekatplayStudio/ComfyUI-Music-to-Video — ``nodes.py:STILL_WORDS``).
    """
    cleaned = _STILL_WORDS_RE.sub("", motion_text)
    return re.sub(r"\s{2,}", " ", cleaned).strip()


def is_rear_view_prompt(text: str) -> bool:
    """True when ``text`` specifies over-the-shoulder or back-of-head staging."""
    return bool(_REAR_VIEW_RE.search(text))


def is_silhouette_prompt(text: str) -> bool:
    """True when ``text`` explicitly stages the subject as a backlit/dark silhouette."""
    return bool(_SILHOUETTE_RE.search(text))


def scrub_frontal_facial_details(description: str) -> str:
    """Remove frontal facial-feature clauses for rear-view/OTS or wide-shot prompt hygiene.

    Credits & Attribution:
        - Rear-view/OTS facial scrubbing inspired by ``afj176/open-cinema-studio``
          (https://github.com/afj176/open-cinema-studio — ``app/staging.py``).
        - Anatomy-noun wide-framing hygiene inspired by ``JavaDerek/music-video-pipeline``
          (https://github.com/JavaDerek/music-video-pipeline — ``prompting.py``).
    """
    clauses = re.split(r"([,;.])", description)
    kept: list[str] = []
    i = 0
    while i < len(clauses):
        part = clauses[i]
        sep = clauses[i + 1] if i + 1 < len(clauses) else ""
        i += 2
        if not part.strip():
            continue
        if _FRONTAL_FACIAL_CLAUSE_RE.search(part):
            continue
        kept.append(part.strip() + sep)
    cleaned = " ".join(kept).strip().rstrip(",;").strip()
    return cleaned or description


def scrub_lower_body_wardrobe(
    description: str, *, include_waist: bool = True, allow_empty: bool = False
) -> str:
    """Remove lower-body garment clauses (pants, skirts, boots) from tight-shot prompts.

    Image-edit models zoom out from close-ups/medium-close shots when the prompt names trousers,
    skirts, belts, or boots. Keeping only upper-body/collar/hair clothing details holds tight
    framing on the face and shoulders.
    """
    pattern = _LOWER_BODY_STRICT_RE if include_waist else _LOWER_BODY_MEDIUM_RE
    kept: list[str] = []
    for raw_sentence in re.split(r"(?<=[.;])\s+", description.strip()):
        sentence = raw_sentence.strip()
        if not sentence:
            continue
        if not pattern.search(sentence):
            kept.append(sentence)
            continue
        # Drop pure lower-body sentences wholesale (e.g. "She wears simple, flat leather shoes.")
        # instead of leaving dangling stubs like "She wears simple."
        if not _UPPER_BODY_WARDROBE_RE.search(sentence):
            continue
        clauses = re.split(r"([,;.])", sentence)
        sent_kept: list[str] = []
        i = 0
        while i < len(clauses):
            part = clauses[i]
            sep = clauses[i + 1] if i + 1 < len(clauses) else ""
            i += 2
            if not part.strip():
                continue
            if pattern.search(part):
                # Keep upper-body garment noun phrases when only a length modifier (e.g. mid-calf)
                # was attached inside the same clause.
                if _UPPER_BODY_WARDROBE_RE.search(part):
                    stripped = re.sub(
                        r"\b(?:mid[\s-]calf|ankle[\s-]length|calf[\s-]length|"
                        r"floor[\s-]sweeping|full[\s-]length)\b",
                        "",
                        part,
                        flags=re.IGNORECASE,
                    )
                    stripped = re.sub(r"\s{2,}", " ", stripped).strip()
                    if stripped and not pattern.search(stripped):
                        sent_kept.append(stripped + sep)
                continue
            sent_kept.append(part.strip() + sep)
        rebuilt = " ".join(sent_kept).strip().rstrip(",;").strip()
        if rebuilt:
            kept.append(rebuilt)
    cleaned = " ".join(kept).strip().rstrip(",;").strip()
    return cleaned if (cleaned or allow_empty) else description


def fit_prompt_budget(prompt: str, max_chars: int, *, keep_tail: int = 2) -> str:
    """Trim ``prompt`` to ``max_chars`` on sentence boundaries, pinning head and continuity tail.

    Credits & Attribution:
        Head + continuity-tail prompt budgeting inspired by
        ``GeekatplayStudio/ComfyUI-Music-to-Video``
        (https://github.com/GeekatplayStudio/ComfyUI-Music-to-Video — ``nodes.py:_fit_prompt``).
    """
    text = prompt.strip()
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if len(sentences) <= 1 + keep_tail:
        return text[:max_chars].rstrip()

    head = sentences[0]
    tail = sentences[-keep_tail:] if keep_tail > 0 else []
    middle = sentences[1 : len(sentences) - len(tail)]
    reserved = len(head) + sum(len(t) + 1 for t in tail)
    if reserved >= max_chars:
        tail_str = " ".join(tail)
        if tail_str and len(tail_str) < max_chars:
            head_budget = max_chars - len(tail_str) - 1
            if head_budget >= 20:
                return f"{head[:head_budget].rstrip()} {tail_str}"
            return tail_str[:max_chars].rstrip()
        joined = " ".join([head, *tail])
        return joined[:max_chars].rstrip()

    chosen_mid: list[str] = []
    used = reserved
    for sent in middle:
        if used + len(sent) + 1 <= max_chars:
            chosen_mid.append(sent)
            used += len(sent) + 1
    return " ".join([head, *chosen_mid, *tail])


def inject_multicut_timestamps(
    sub_descriptions: Sequence[str],
    shot_duration: float,
    beat_offsets: Sequence[float] = (),
) -> str:
    """Deterministically prepend beat-locked ``[Shot N] At MM:SS.mmm`` cues to multi-cut prose.

    Keeps the LLM writer timestamp-free so sub-cut timings never hallucinate or drift out of
    order.

    Credits & Attribution:
        Timestamp-free LLM writer contract inspired by ``hodgemann/mondo-music-video``
        (https://github.com/hodgemann/mondo-music-video — ``mmv/engine/schema.py``).
    """
    cleaned = [s.strip() for s in sub_descriptions if s.strip()]
    if not cleaned:
        return ""
    n = len(cleaned)
    dur = max(0.2, shot_duration)
    raw_times = [dur * k / n for k in range(n)]
    times = [0.0]
    for target in raw_times[1:]:
        if beat_offsets:
            nearest = min(beat_offsets, key=lambda b: abs(b - target))
            chosen = (
                nearest if abs(nearest - target) <= 0.35 and nearest > times[-1] + 0.1 else target
            )
        else:
            chosen = target
        times.append(min(dur - 0.05, max(times[-1] + 0.1, chosen)))

    lines: list[str] = []
    for idx, (t, desc) in enumerate(zip(times, cleaned), start=1):
        mins = int(t // 60)
        secs = t - mins * 60
        verb = "Opens on" if idx == 1 else "Hard cut to"
        lines.append(f"[Shot {idx}] At {mins:02d}:{secs:06.3f}, {verb}: {desc}")
    return " ".join(lines)


BIBLE_SYSTEM = """You are the director of a music video that balances performer shots with
cutaways. First interpret the song's meaning and musical style, then write the visual bible that
every shot will follow.
Write every field in English even when the lyrics or style brief are in another language or mix
languages, because downstream image and video models take English prompts. Every 'id' field MUST
use ASCII lowercase letters, digits, and underscores only (romanize non-Latin names for ids, such
as id='sakura' with name='さくら').

- interpretation: before designing any visuals, analyze what the song is conveying as a director:
  - meaning: two to four sentences on what the song is really about beneath the literal words —
    its core theme, emotional conflict, and subtext.
  - tone: one sentence on the emotional register and attitude (e.g. 'wistful and restrained,
    never melodramatic', 'defiant and ironic', 'haunted and intimate').
  - music_feel: one to two sentences on how the musical style, key/mood, instrumentation, dynamics,
    and vocal delivery translate into visual rhythm, lighting contrast, and movement.
  - visual_motifs: 2 to 5 short recurring visual metaphors or symbols grounded in the song's
    meaning (not merely literal objects named once in a lyric).
  - section_beats: one entry per song section, in order, using the EXACT section names listed
    under SONG SECTIONS (do not invent or rename sections), each with:
    - section: exact section name from SONG SECTIONS.
    - emotion: the emotional state or shift in this section.
    - visual_beat: what happens visually or narratively here to express the lyrics and music.
    - camera_energy: how shot scale, pacing, and performance intensity should feel here.
- concept: two to four sentences on the visual idea that ties the video together, grounded directly
  in 'interpretation' (meaning, tone, music_feel, and visual_motifs).
- characters: every distinct singer, vocalist, or on-camera band/group member indicated by the
  style brief, section headers, or lyrics (one entry for a solo track; two for a duet; one per
  member or named character for a band, group, or theatrical piece). Match every character's gender
  presentation and total cast size strictly to the vocal cues in the style brief and lyrics (for
  example, a solo female vocal track must have exactly one female lead; never invent unmentioned
  co-characters or reuse example names). Every singer or recurring performer MUST have their own
  entry in 'characters' with:
  - id: short ASCII lowercase identifier (letters, digits and underscores, e.g. 'lead_vocal',
    'vocalist_1', 'vocalist_2', 'guitarist').
  - name: display name or role invented specifically for this song (never copy generic example
    names from instructions).
  - appearance: one dense paragraph on that person's PERMANENT physical look: age range, gender
    presentation, face shape, skin tone, natural skin texture (visible pores, subtle natural
    unevenness, real facial asymmetry), eyes, brows, nose, lips, hair (cut, length, parting,
    individual strands and flyaways), makeup, build, and any identifying marks. Give the side of
    any mark as the person's anatomical left or right. Forbid idealized, airbrushed, or glass-skin
    rendering so the person reads as a real unretouched photograph. Unless the style brief
    explicitly calls for surreal, fantasy, or stylized fashion figures, ground characters as
    ordinary, relatable real-world people (everyday faces and believable proportions rather than
    hyper-idealized supermodels). Do not describe clothing here.
    It must be precise enough that an image model draws the same person every time. Do not
    describe a real, identifiable person.
  - wardrobe: that person's costume states in the order they appear in the song. Each has a short
    id (lowercase letters, digits and underscores) and a COMPLETE head-to-toe description of
    everything worn in that state (outerwear/top, trousers/skirt, footwear, fabric texture, weave,
    and natural drape — omitting trousers/footwear causes cropped legs on reference sheets). Never
    contradict the character's 'invariants' (for example, do not write 'no jewelry' if 'invariants'
    includes a necklace or earrings). When a later state removes an outer layer, still describe the
    full remaining outfit head to toe alongside what was removed. Only add a new state when
    clothing actually changes—never for poses, props, or rooms. Use one state if their costume
    never changes.
  - invariants: optional short phrase listing signature accessories or visual details that must
    survive across every costume change (e.g. symmetric earrings or a distinctive pendant). Prefer
    symmetric accessories over tiny single-sided jewelry pinned to anatomical left or right unless
    the user's style brief explicitly specifies an asymmetric mark (image-edit models often mirror
    small single-sided accessories). It is added to every still of this character, so never put
    anything here that a costume change would alter (sleeves, jacket, shoes).
- appearance: copy the primary/lead singer's permanent appearance paragraph here as well.
- invariants: copy the primary/lead singer's invariants here as well.
- wardrobe: copy the primary/lead singer's wardrobe states here as well.
- style, palette, lighting: short phrases an image model can act on. Keep a consistent film stock
  and color grade across all sets while letting practical light sources vary per location.
- negative_prompt: comma-separated things to avoid.
- settings: 3 to 6 distinct sets. Each has a name, the sections of the song it serves
  (sections), a description of how it looks (for an image model, describing the room/environment
  layout and one mid-depth landmark without camera-framing directives like 'focus on a close-up'),
  how the performers move and the camera works there (movement), and an optional parent_set naming
  an earlier set when this location is a stripped, altered, or nested variant of it. Unless the
  style brief explicitly calls for fantasy, sci-fi, or surreal worlds, ground sets in physically
  shootable real-world locations (everyday streets, apartments, transit, cafés, rooftops) that an
  indie crew could film on location.
- props: 0 to 3 recurring hero objects grounded specifically in this song's lyrics or concept (use
  an empty list if none are needed; never invent generic sci-fi discs or unrelated props). Each
  has id (short lowercase identifier), name, sections, and a purely visual description with
  approximate scale in centimeters, finish, and affirmative descriptions of its visible structure
  (no sound/audio descriptions, and in shot prompts refer to props by natural prose, never by
  snake_case ids).
- rules: 4 to 10 short rules that every shot follows (camera language, physical camera placement,
  continuity, effects, how much of the video shows the performers). For duets or groups, write
  rules that apply across the cast rather than locking every shot to a single character. Only
  rules a shot-by-shot director can act on."""

SHOTS_SYSTEM = """You are the director of a music video. Timing is already fixed. For each shot
you are given, choose the creative details.

Rules:
- Return exactly one entry per shot id you were given.
- character: the id of the primary character on camera in this shot (from CHARACTERS, when given).
  On 'lipsync' shots, follow the shot's vocal cue:
  * vocal='solo' singer='<id>': 'character' MUST be that single vocalist, and ONLY that vocalist
    sings in still_prompt and motion_prompt. If a second character appears in a medium_close
    reaction two-shot, describe the second character listening or reacting with mouth closed, never
    singing.
  * vocal='together' singers=[...]: BOTH listed vocalists sing simultaneously in unison or harmony.
    Use framing medium_close (never close_up, which fits only one face), include all singing ids in
    'characters', and stage both vocalists together with both faces and mouths unobstructed and
    actively singing.
  Use null on 'broll'.
- characters: list of all character ids visible in the frame (primary 'character' first). On solo
  close_up shots, include only [character]. On medium_close, medium, or wide shots that show a
  simultaneous duet, reaction two-shot, or band/group performance, list every character visible in
  the frame — and ONLY list characters who actually appear in still_prompt. Never write raw
  snake_case character ids (such as 'lead_vocal' or 'vocalist_1') inside still_prompt or
  motion_prompt; always use the character's display name.
- setting: use the exact name of one set from the visual bible, chosen for the shot's song
  section. Follow each set's sections and movement notes.
- props: list of hero prop ids from PROPS visible in this shot (empty list when none appear).
- wardrobe: use the id of one wardrobe state for the shot's primary character. It is what that
  character wears in the shot's FIRST frame. A costume change may happen during a shot; the next
  shot then uses the new state. Each character's wardrobe only moves forward through their list.
  Never return to an earlier state. Use null on 'broll'.
- 'lipsync' shots show the active singer singing the given lyrics. Use framing close_up or
  medium_close (never medium or wide) so the singer's mouth is clearly visible and unobstructed
  (no hands, microphones, hair or foreground objects across the mouth). In still_prompt for
  'lipsync' shots, keep the singer's mouth relaxed, softly closed, or barely parted (never
  wide-open mid-scream) so audio-driven lip-sync models animate naturally from the vocal track
  and performance fallbacks stay consistent.
- Character close_up framing: frame head-and-shoulders or the full face (forehead to chin) with
  both eyes and mouth clearly inside the frame. Never ask for extreme mouth-only or eye-only
  macros on character shots, which crop off the head and break face framing checks. On close_up
  and medium_close shots, describe ONLY face, neck, shoulders, upper torso, and raised hands—never
  mention boots, shoes, feet, legs, lower skirt hems, or ground/floor at their feet, and never
  write 'waist up' or 'medium shot' for a medium_close shot.
- Shots given as 'performance' have no vocals. Keep them as 'performance' (performer visibly on
  camera, not singing) or change them to 'broll' (no performer) where a cutaway serves the song,
  without putting more than two 'broll' shots in a row. Never leave type as 'performance' if
  still_prompt describes only an empty landscape, building, or object with no performer in frame:
  any shot with no performer MUST be 'broll'.
- Shots given as 'broll' are cutaways: the vocal plays over them and no singer is shown. Keep
  them 'broll', with character and wardrobe null. NEVER place any named cast member, singer, or
  recognizable close-up human face in a 'broll' shot (any shot featuring a named character or
  prominent person MUST be 'performance', not 'broll'). Show what the lyric is about: objects,
  architecture, light, weather, props, inanimate figurines, or—when the setting genuinely calls
  for atmosphere (such as a club or concert)—anonymous distant backlit silhouettes, blurred
  background crowd motion, or hands-only detail inserts with no recognizable faces. Keep the main
  subject sharply in focus (never ask for a completely out-of-focus or blurry whole frame). Still
  give every shot a framing (close_up, medium_close, medium or wide): never null, even on a
  cutaway.
- Match detail to shot size on BOTH singer shots and 'broll' cutaways. Small details (identifying
  marks, jewelry, cuffs, clock hands, inscriptions, small props, screen content) belong in
  close_up or medium_close shots only; never ask for tiny altered object parts in a wide shot. In
  medium and wide shots, describe pose, silhouette, scale against architecture, and action, not
  fine detail.
- State absences affirmatively: image models often draw nouns named after 'no' (e.g. 'no minute
  hand' draws a minute hand). Whenever a part is missing or excluded, describe what IS visibly
  present in its place (e.g. 'clock dial displaying only a single short hour hand on a bare brass
  center pin' instead of bare 'no minute hand').
- Physical camera placement: always position the camera in open floor space at normal human
  standing or seated height on the clear side of any furniture. Never place the lens inside,
  below, level with, or passing through tables, walls, or solid objects, and never slice or
  occlude a performer's body with foreground furniture edges.
- Follow the production rules you are given.
- Serve the SONG INTERPRETATION (meaning, tone, music feel, visual motifs) and the current
  section's SECTION ARCS beat (emotion, visual beat, camera energy) in every shot. Express the
  emotional subtext and musical style through performance, staging, light, and recurring motifs
  rather than literally illustrating isolated lyric words.
- Use the position in the song and the section to shape energy: build tension toward
  choruses, release in them, hold still where the song is stripped back. Match energy to
  intensity (0 calm to 1 peak).
- Vary framing between neighbouring shots, and stay consistent with the shots already directed.
- Shape the video like the song. First verse: restrained, one look. First chorus: the first
  escalation, so change the set, the framing or the light from the verse. Second verse: never a
  repeat of the first; add a complication (a new angle, set, prop or costume beat). Bridge: break
  the pattern of the shots before it (a different set, light or pace). Final chorus: the payoff,
  bringing earlier looks back together. Close on an image that lingers.
- Pace is built from four levers that stack: cut speed, movement in the frame, shot scale (the
  closer the lens, the faster it feels) and camera movement. Each shot has a pace (calm, moderate
  or high) and a note when it changes from the shot before. Raise the pace by tightening the
  framing and adding movement in the frame, and lower it by doing the opposite. A change of pace
  must follow the song (a section change or a rise or fall in intensity), never come from nowhere,
  and the bridge should strip back so the next lift is felt. On 'lipsync' shots the camera stays
  steady, so use scale and movement in the frame, not camera moves, to carry the pace.
- Eye contact: in verses let the singer look slightly off the lens, in three-quarter view or at
  something in the room while keeping their face and mouth clearly visible to the lens. Save a
  look straight into the lens for the pre-chorus build and the chorus hook.
- Use at least three distinct looks across the video (different sets or lighting), not one setup.
  Hold one primary set across each musical section rather than ping-ponging between sets every
  1-2 shots, unless a deliberate cross-cut is called for in a later verse or bridge.
- Reliable camera motion: prefer movements that image-to-video models execute cleanly (slow
  push-in, slow pull-back, gentle handheld drift, subtle pan, subject expression/performance,
  and ambient motion like fire, snow, embers, or light shifts). Avoid complex 360-degree orbits
  or long multi-room tracking shots that models tend to render static.
- Something should visibly change every 2 to 4 seconds without a cut: light shifting, a flame or
  curtain moving, a background element, the performer's posture. Do not use camera moves for this
  on lipsync shots.
- Aim for one striking, printable frame in every 30 seconds: strong symmetry, a silhouette, or a
  single bright light against dark.
- still_prompt: in English, describes the first frame only: framing, physical camera placement in
  open space, who is in frame, pose, action, camera angle. Translate imagery from non-English or
  multi-language lyrics into concrete English visual descriptions rather than quoting raw lyrics.
  It is sent to an image-edit model together with a reference portrait of the primary singer, and
  the set and character descriptions are added automatically, so do not repeat full physical
  descriptions.
- motion_prompt: in English, what happens in the clip: performance, body movement, and physical
  camera operation (describe handheld physics such as subtle operator breathing sway, small weight
  shifts, and soft starts and stops rather than bare labels). Keep camera moves gentle and stable
  on lipsync shots so the singing mouth stays clearly tracked. On 'broll' and 'performance' clips,
  keep motion physically steady—one continuous camera move plus smooth ambient or body motion—and
  avoid sudden violent impacts, collapses, or multi-stage chaotic events that trigger video-model
  flicker or unintended scene cuts. Every shot MUST remain a single continuous take with no
  internal cuts and no subtitles or on-screen text. For shots 4.0 seconds or longer with distinct
  movement beats or camera shifts, you may structure motion_prompt as timestamped sub-beats
  covering the shot duration (e.g. '0-2.0s: ... 2.0-4.0s: ...')."""

NARRATIVE_BIBLE_ADDENDUM = """VIDEO STYLE — NARRATIVE / NON-LIPSYNC PERFORMANCE:
- This video has NO on-camera lip-sync. Characters NEVER sing or mouth lyrics to the camera.
- Design recurring protagonist(s), dancer(s), or musician(s) who express the song's story and
  emotion through movement, action, staging, and atmosphere rather than vocal lip-sync.
- You are free to use wide staging, silhouettes, profiles, masks, choreography, and dynamic
  camera language without close-up mouth-visibility constraints."""

VISUAL_BIBLE_SYSTEM = """You are the director of a purely visual, characterless music video
(100% b-roll: environments, architecture, light, weather, and recurring symbolic objects). First
interpret the music's mood and meaning, then write the visual bible that every shot will follow.
Write every field in English even when the lyrics or style brief are in another language, because
downstream image and video models take English prompts. Every 'id' field MUST use ASCII lowercase
letters, digits, and underscores only.

- interpretation: before designing any visuals, analyze what the track is conveying as a director:
  - meaning: two to four sentences on the emotional core, imagery, and subtext of the music (and
    lyrics, if any).
  - tone: one sentence on the emotional register and attitude (e.g. 'hypnotic and stark',
    'luminous and melancholic').
  - music_feel: one to two sentences on how the musical style, key/mood, instrumentation, and
    dynamics translate into visual rhythm, lighting contrast, and camera movement.
  - visual_motifs: 2 to 5 short recurring visual metaphors, objects, or natural phenomena grounded
    in the music's mood.
  - section_beats: one entry per song section, in order, using the EXACT section names listed
    under SONG SECTIONS (empty list if SONG SECTIONS is '(none)'), each with section, emotion,
    visual_beat, and camera_energy.
- concept: two to four sentences on the overarching visual journey that ties the film together.
- characters: ALWAYS return an empty list []. There are NO recurring human performers or singers.
- appearance: ALWAYS return an empty string "".
- invariants: ALWAYS return an empty string "".
- wardrobe: ALWAYS return an empty list [].
- style, palette, lighting: short phrases an image model can act on. Keep a consistent film stock
  and color grade across all sets while letting practical light sources evolve per location.
- negative_prompt: comma-separated things to avoid (include people, faces, singers, text).
- settings: 3 to 6 distinct uninhabited environments or visual worlds. Each has a name, the
  sections of the song it serves (sections), a concrete description of how it looks (naming one
  primary mid-depth focal object so the lens has a clear plane of focus), how light/atmosphere and
  the camera move there (movement), and an optional parent_set when a location is an altered
  variant of an earlier set.
- props: 0 to 3 recurring hero objects central to the visual storytelling (e.g. a monolith, clock,
  prism, lantern). Each has id (short lowercase identifier), name, sections, and a concrete
  description with approximate scale, finish, and affirmative descriptions of visible structure.
- rules: 4 to 10 short actionable visual direction rules (camera language, physical camera
  placement, lighting evolution, motion rhythm, no people in frame)."""

NARRATIVE_SHOTS_ADDENDUM = """VIDEO STYLE — NARRATIVE / NON-LIPSYNC PERFORMANCE:
- This video has NO lip-sync shots. Every 'performance' shot shows the character(s) acting,
  moving, dancing, playing an instrument, or reacting with mouth closed (NEVER singing to camera).
- Use the full range of framings (close_up, medium_close, medium, wide), angles, silhouettes, and
  dynamic camera movements freely—there are no lip-sync mouth-visibility or steady-camera limits."""

VISUAL_SHOTS_SYSTEM = """You are the director of a purely visual, characterless music video
(100% b-roll). Timing is already fixed. For each shot you are given, choose the creative details.

Rules:
- Return exactly one entry per shot id you were given.
- Every shot MUST keep type='broll', with character=null, characters=[], and wardrobe=null. Never
  introduce people, singers, or human faces.
- setting: use the exact name of one set from the visual bible, chosen for the shot's song
  section and energy. Follow each set's sections and movement notes.
- props: list of hero prop ids from PROPS visible in this shot (empty list when none appear).
- framing: always choose close_up, medium_close, medium, or wide (never null). Vary framing
  between neighbouring shots to create visual rhythm. Match detail to shot size: small textures and
  object details belong in close_up/medium_close; architecture, scale, and atmosphere belong in
  medium/wide.
- State absences affirmatively (describe what IS visibly present rather than bare 'no X').
- Physical camera placement: position the camera in open space with clear sightlines, never
  clipping through walls, floors, or solid objects.
- Serve the SONG INTERPRETATION (meaning, tone, music feel, visual motifs) and SECTION ARCS beat.
  Shape energy across the track: build tension into peaks, strip back in calmer passages, and
  close on a lingering image.
- Use at least three distinct sets or lighting looks across the video. Something should visibly
  evolve every 2 to 4 seconds (light shifting, particles, water, fog, shadows, camera glide).
- still_prompt: in English, describes the first frame only—framing, camera placement, subject
  object/environment, lighting, and atmosphere (no people or faces).
- motion_prompt: in English, describes what moves in the scene and how the camera moves. Every
  shot MUST remain a single continuous take with no internal cuts and no subtitles or on-screen
  text."""

REWRITE_SYSTEM = """You fix prompts for an AI image or video model after a quality check
rejected the result. Rewrite the prompt in English so the listed problems are less likely, keeping
the shot's intent, framing and setting.
- Prefer simplifying (hands out of frame, calmer pose, camera in open floor space clear of
  furniture, plainer background) over piling on instructions.
- Replace negative 'no <part>' phrasing that backfired: image models attend to named nouns and
  draw them anyway. Remove the forbidden noun and describe the visible state affirmatively (for
  example, rewrite 'grandfather clock, no minute hand' as 'close-up of the tarnished brass
  clock dial displaying only one single short hour hand on an empty center pin').
- Fix body proportions and environmental scale explicitly: if a person looked vertically squashed,
  stubby-legged, stiffly pasted, or giant against buildings, specify natural full-length human
  proportions (~7.5 heads tall, normal leg-to-torso ratio), an organic in-scene pose, and concrete
  scale against doors, windows, or architecture.
Return the new prompt only, in the 'prompt' field."""

BRIEF_SYSTEM = """You are a music video writer and Suno song producer. Given a short video
concept, optional target duration, and musical/visual vibe, write:

- lyrics: Suno-ready lyrics in the language(s) requested by the concept or vibe (matching the
  concept's language if none is named, and mixing languages naturally when requested), structured
  with English [Section] headers ([Intro], [Verse], [Pre-Chorus], [Chorus], [Bridge], [Outro]; for
  duets, tag solo sections like [Verse 1: Name] and simultaneous sections like [Chorus: Both]).
  Keep line counts concise for the requested duration. Put ONLY sung lines or sung adlibs in
  (parentheses); never put unspoken SFX or stage directions in parentheses.
- suno_style: one dense Suno Custom Mode style prompt in English specifying genre, vocal
  language(s), vocal register and phrasing, key instruments, BPM, rhythm feel, and stereo
  production polish.
- visual_direction: a short creative brief in English covering the performer look, key costume
  states, locations, lighting, palette, and story beats for mvdirector's visual bible."""

INFER_FROM_LYRICS_SYSTEM = """You are a music video director and producer preparing a project from
a finished audio track with no existing lyrics file or style brief. Given a timestamped vocal
transcript, a compact audio and musical-character summary (duration, BPM, key/mode, timbral
brightness, percussiveness, dynamic range, vocal coverage, energy arc), and optionally the audio
track itself, return:

- lyrics: the transcribed words organized under English [Section] headers ([Intro], [Verse 1],
  [Pre-Chorus], [Chorus], [Verse 2], [Bridge], [Outro]; for duets or alternating vocalists, tag
  solo sections like [Verse 1: Singer 1] and simultaneous duet sections like [Chorus: Both]).
  Strictly preserve the actual sung words, original language(s), and script(s) from the transcript
  (never translate non-English or code-switching lyrics into another language)
  so word-level audio alignment for lip-sync stays accurate—fix only capitalization, punctuation,
  and line breaks, and never invent new verses or rewrite sung words (keep "kind of" as "kind of"
  and contractions as sung, and fix a word only when the transcript is plainly a mishearing).
  Put ONLY sung adlibs in (parentheses); never put unspoken SFX or stage directions in
  parentheses. If the track is instrumental with no transcribed words, return "[Instrumental]".
- music_style: a concise musical style description in English grounded in the audio/musical
  character (key/mode, timbre, percussiveness, dynamics, instrumentation if heard) and the lyrics'
  tone, language(s), structure, and energy arc (genre, vocal language and register/delivery,
  rhythm feel, and production mood).
- visual_direction: a concrete creative brief in English for mvdirector's visual bible grounded in
  the song's deeper meaning, emotional subtext, musical feel, and lyrical imagery. State what the
  song is conveying emotionally, then cover the singer/performer's look and costume arc, 3 to 5
  concrete locations/sets, recurring visual motifs, lighting, color palette, and story beats.
  Explicitly design for on-camera lip-synced singing performance during vocal hooks and key verses
  (clear, well-lit, unobstructed face and mouth with no masks or foreground obstructions) intercut
  with atmospheric/narrative cutaways inspired by the song's meaning.
  Write music_style and visual_direction as plain English prose without markdown (no asterisks,
  headings or bullet syntax). Do not quote a BPM figure; describe the tempo feel in words."""

PERFORMANCE_MOTION = (
    "The singer moves with the music, mouth closed, not singing. Slow, steady camera drift."
)

MIX_THEMES_SYSTEM = """You are the visual director for a long-form continuous DJ or background mix
(30 to 90+ minutes). Viewers play these mixes in the background and scrub across chapters, so each
track chapter MUST have its own immediately recognizable visual environment, color palette, and
lighting mood.

Rules:
- Return one entry per chapter 'id' given, plus top-level 'style' and 'negative_prompt' strings.
- Uninhabited environments only: depict empty architecture, landscapes, kinetic light sculptures,
  weather, interiors, or abstract/cybernetic spaces with zero people, faces, or performers. State
  absences affirmatively by describing the uninhabited space itself.
- High chapter contrast: neighbouring chapters MUST differ clearly in color palette, lighting
  temperature, or spatial scale so timeline scrubbing gives instant visual chapter feedback. Write
  color palettes using natural color words (e.g. 'deep teal, warm amber, slate grey'), never hex
  codes like '#FFA500'.
- No text-bearing props or movie-title framing: avoid signage, neon signs, storefronts, books,
  screens, posters, or plaques; describe 'style' with photographic or architectural terms (e.g.
  'atmospheric ambient photography') rather than 'cinematic' or movie-trailer phrasing so video
  models do not invent opening credits. Never mention text, titles, credits, subtitles, or
  captions in any field except 'negative_prompt', not even to exclude them: naming them in a
  prompt primes video models to render them.
- Seamless loop physics: design every 'clip_prompts' entry and 'motion_style' for short (4-8s)
  crossfade-looped clips. Use continuous cyclical or steady ambient motion (drifting fog, rippling
  water, slow volumetric light sweeps, floating dust/embers, gentle camera glide). Avoid one-off
  explosions, sudden whip-pans, narrative plot beats, internal cuts, or on-screen text.
- Write every field in English. For each chapter, provide exactly the requested number of
  complementary 'clip_prompts' (varying framing/angle within the same chapter environment).
- When 'lyrics_excerpt' is provided for a chapter, draw concrete environmental imagery, motifs,
  and mood from it (translating non-English imagery into English visual descriptions without
  rendering on-screen text)."""

PANEL_LAYOUT = {
    "row": "Four equal panels in a single horizontal row, thin neutral grey dividers.",
    "grid": (
        "Four equal panels in a 2x2 grid, thin neutral grey dividers. Panels 1 and 2 are the top "
        "row and panels 3 and 4 are the bottom row."
    ),
    "columns": (
        "Three-column layout with thin neutral grey dividers: the LEFT column is split "
        "horizontally into two equal-height stacked panels (top and bottom), while the CENTER "
        "and RIGHT columns are each one full-height panel."
    ),
}
PANEL_LIST = (
    "Panel 1, FULL BODY, straight-on frontal, arms at sides, feet in frame, eyes to camera. "
    "Panel 2, FULL BODY, 90-degree profile facing frame-right, feet in frame. "
    "Panel 3, FULL BODY, 3/4 turn toward frame-left, feet in frame. "
    "Panel 4, MACRO CLOSE-UP of the face, forehead to chin filling the panel, tack-sharp, "
    "showing maximum pore detail, natural skin texture, and iris texture."
)
PANEL_LIST_COLUMNS = (
    "Left column, TOP, FRONTAL HEADSHOT: head and shoulders, front view, eyes looking straight "
    "into the lens, neck and collar details visible. "
    "Left column, BOTTOM, SIDE PROFILE HEADSHOT: same crop and scale as the top panel, head "
    "turned to a three-quarter profile showing hair length and ear accessories. "
    "Center column, FULL-BODY FRONT: framed from the neck down (head cropped at the base of the "
    "neck), relaxed upright pose, arms at sides, hands visible and empty, feet in frame. "
    "Right column, FULL-BODY BACK: framed from the neck down at the same crop height, facing "
    "directly away, showing the back of the outfit, back hair, and heels. "
    "The face appears only in the two left-column headshots."
)
_SHEET_BACKGROUNDS = {
    "white": (
        "Pure white seamless studio background in every panel. Soft even key light, identical in "
        "all panels."
    ),
    "grey": (
        "Plain empty neutral grey studio background in every panel. Soft, even, flat diffuse "
        "lighting, identical in all panels."
    ),
}
_SHEET_BODY_REALISM = (
    "Neutral expression and mouth closed in every panel. Photorealistic unretouched "
    "photography: real skin texture with individually visible pores, natural uneven tone, subtle "
    "facial asymmetry, and individual hair strands with flyaways. No smoothing, no airbrushing, "
    "no glass-skin glow, no CGI or 3D-render look."
)
SHEET_STUDIO = (
    f"{_SHEET_BACKGROUNDS['white']} {_SHEET_BODY_REALISM} "
    "No text, no labels, no captions, no watermarks anywhere in or around the panels."
)
# The first sheet is built from a written description.
SHEET_TEMPLATE = (
    "Character reference sheet of a single person. {layout} {panels} The person looks identical in "
    "every panel. {studio} {description}"
)
DERIVED_CHARACTER_SHEET_TEMPLATE = (
    "The reference image is an existing character reference sheet of a person. Edit it to "
    "create a variant character sheet, keeping their poses, body proportions, panel layout, "
    "framing, and background identical unless modified below. {layout} {panels} "
    "Variant appearance and outfit, identical across every panel: {description} {studio}"
)
# Later sheets are edits of the first, so the face is deliberately not described again: words about
# a face placed next to a reference image make the model blend the two.
STATE_SHEET_TEMPLATE = (
    "The reference image is an existing character reference sheet of a person. Create a NEW "
    "character reference sheet of the same person in different clothing. The face must be "
    "identical to the reference: same person, same bone structure, same hairline, same hair and "
    "makeup, instantly recognizable. Only the wardrobe changes.{invariants_note} {layout} "
    "{panels} Wardrobe, identical in all panels: {wardrobe} {studio}"
)
SHEET_NEGATIVE = (
    "duplicate panels, different people in different panels, cropped feet in full-body panels, "
    "colored background, text other than the panel labels, flawless poreless skin, glass-skin "
    "glow, airbrushed or plastic or waxy skin, CGI, 3D render, video-game look, idealized "
    "symmetrical face, blurry face in Panel 4, distorted hands or fingers"
)
PLATE_CAMERA_LIMITATIONS = (
    "Real photograph taken on film with genuine optical lens limitations: only ONE focal plane at "
    "mid-depth is tack-sharp; foreground and far background fall progressively out of focus with "
    "natural optical falloff. Resolution degrades slightly toward the frame edges and corners. "
    "Brightest practical lights roll off and bloom softly; deep corner shadows crush naturally. "
    "Restrained texture: do not render uniform etched micro-detail across every surface; "
    "off-focus surfaces resolve as soft shape and tone only. Subtle 35mm film grain and natural "
    "lens vignette."
)
PLATE_TEMPLATE = (
    "A photograph of an empty location, completely empty of people. {description} "
    f"{PLATE_CAMERA_LIMITATIONS} {{style}} "
    "No people, no figures, no faces, no readable text or logos."
)
PLATE_STYLE_TEMPLATE = (
    "Create a photograph of an empty location matching the visual style, color palette, and mood "
    "of the reference image, completely empty of people. {description} "
    f"{PLATE_CAMERA_LIMITATIONS} {{style}} "
    "No people, no figures, no faces, no readable text or logos."
)
PLATE_EMBED_TEMPLATE = (
    "Create a photograph of an empty location ('{target_name}') in its own room layout, camera "
    "angle, and lighting, incorporating the scene from the reference image ('{source_name}') "
    "inside a screen, window, or miniature diorama as described. {description} "
    f"{PLATE_CAMERA_LIMITATIONS} {{style}} "
    "No people, no figures, no faces, no readable text or logos."
)
PLATE_EDIT_TEMPLATE = (
    "Edit the existing empty location photograph ({source_name}) to depict '{target_name}', "
    "keeping no people and matching the camera perspective, architecture, and film stock where "
    "unchanged. Required changes: {description} {style} "
    "No people, no figures, no faces, no readable text or logos."
)
PLATE_NEGATIVE = (
    "people, figures, faces, readable text, logos, CGI, 3D-render, video-game look, glossy "
    "plastic sheen, uniform sharpness across the whole frame, everything-in-focus deep focus, "
    "hyper-detailed micro-texture on every surface, crisp resolution in frame corners, HDR look, "
    "lifted flat shadows, flat sterile lighting, watermark"
)
PROP_TEMPLATE = (
    "Generate a set of two real photographs of the same hero prop side by side, split down the "
    "middle with a thin neutral grey divider line. Plain empty neutral grey background. Soft, "
    "even, flat diffuse lighting. Left panel: the prop seen flat-on from the front, filling most "
    "of the panel. Right panel: the same prop tilted at a three-quarter angle showing thickness, "
    "edge profile and depth. Same object, same scale and lighting in both panels. {description} "
    "{style} No people, no hands, no readable text, no logos."
)
PROP_NEGATIVE = (
    "people, hands, fingers, readable text, logos, watermarks, CGI, 3D render, video-game look, "
    "mismatched proportions between panels, extra parts"
)
# Added to a still prompt when a set plate is supplied as a second reference image.
PLATE_NOTE = (
    "The first reference image shows the singer: keep the same face, hair and costume. The second "
    "reference image shows the location: place the singer in that location and match its layout, "
    "lighting and color grade exactly."
)
CAMERA_PLACEMENT_NOTE = (
    "Camera positioned in open floor space at human eye height; no solid furniture intersecting "
    "or occluding the subject."
)
_FRAMING_SCALE_NOTE = {
    Framing.CLOSE_UP: CAMERA_PLACEMENT_NOTE,
    Framing.MEDIUM_CLOSE: (
        "Camera positioned in open floor space at human eye height; chest-up medium close-up "
        "with natural head-to-shoulder proportions and no solid furniture intersecting or "
        "occluding the subject."
    ),
    Framing.MEDIUM: (
        "Camera positioned in open floor space at human eye height; waist-up or knee-up medium "
        "framing with natural human height-to-width proportions, relaxed natural posture (do not "
        "copy the stiff studio pose of the reference portrait), and no solid furniture "
        "intersecting or occluding the subject."
    ),
    Framing.WIDE: (
        "True wide environmental framing: scale the performer to realistic human size relative to "
        "doors, windows, and architecture (occupying roughly one-quarter to one-half of frame "
        "height, never towering over buildings), with natural elongated full-body proportions "
        "(~7.5 heads tall, normal leg-to-torso ratio, never vertically squashed) and a natural "
        "in-scene pose rather than the stiff frontal pose of the reference portrait."
    ),
}
_ABSTRACT_NEGATION_HEADS = frozenset(
    {
        "doubt",
        "escape",
        "fear",
        "longer",
        "matter",
        "mercy",
        "more",
        "need",
        "one",
        "other",
        "place",
        "promise",
        "promises",
        "reason",
        "regret",
        "rest",
        "return",
        "rush",
        "sleep",
        "sound",
        "time",
        "tomorrow",
        "turning",
        "use",
        "way",
        "wonder",
    }
)
_EXCLUSION_RE = re.compile(
    r"\b(?:no|without(?:\s+(?:a|an|any))?)\s+"
    r"([a-z][a-z0-9-]*(?:\s+[a-z][a-z0-9-]*){0,2}?)"
    r"(?=[,.;:!?)]|\s+(?:and|or|in|on|at|with|across|over|under|near|by|from|that|which|while)\b|$)",
    flags=re.IGNORECASE,
)


def extract_negative_exclusions(prompt: str) -> list[str]:
    """Extract concrete forbidden objects/parts ('no minute hand' -> 'minute hand') from prose."""
    found: list[str] = []
    for match in _EXCLUSION_RE.finditer(prompt):
        phrase = " ".join(match.group(1).lower().split())
        head = phrase.split()[0] if phrase else ""
        if not head or head in _ABSTRACT_NEGATION_HEADS:
            continue
        if phrase not in found:
            found.append(phrase)
    return found


_FRAMING_LABEL = {
    Framing.CLOSE_UP: "close-up",
    Framing.MEDIUM_CLOSE: "medium close-up",
    Framing.MEDIUM: "medium",
    Framing.WIDE: "wide",
}
_ORDINALS = ("first", "second", "third", "fourth", "fifth", "sixth")
_STILL_REF_ROLES = {
    "reference_image": "the singer: keep the same face, hair and costume",
    "reference_profile": "the singer's side profile angle",
    "reference_three_quarter": "the singer's three-quarter angle",
    "reference_face": (
        "a close-up of the singer's face: match their exact facial features, eyes and hair"
    ),
    "reference_image_2": "the secondary performer: keep their face, hair and costume",
    "set_image": (
        "the location: place the performers in that location and match its layout, "
        "lighting and color grade exactly"
    ),
}


_EXTRA_STILL_REF_RE = re.compile(r"^reference_image_([1-9][0-9]*)\Z")


def _extra_reference_index(name: str) -> int | None:
    """Map ``reference_image_<N>`` (N >= 2) to its 0-based index in ``visible`` (N - 1)."""
    match = _EXTRA_STILL_REF_RE.match(name)
    if match is None:
        return None
    slot = int(match.group(1))
    return slot - 1 if slot >= 2 else None


def _still_ref_role(name: str) -> str | None:
    role = _STILL_REF_ROLES.get(name)
    if role is not None:
        return role
    extra_idx = _extra_reference_index(name)
    if extra_idx is not None:
        return f"performer #{extra_idx + 1}: keep their face, hair and costume"
    return None


def _still_reference_note(
    plate_attached: bool,
    reference_notation: str,
    bound_references: list[str] | None,
) -> str | None:
    if bound_references is None:
        if not plate_attached:
            return None
        if reference_notation == "tags":
            return (
                "References: @image1 shows the singer (keep the same face, hair and costume). "
                "@image2 shows the location (place the singer in that location and match its "
                "layout, lighting and color grade exactly)."
            )
        return PLATE_NOTE
    active: list[tuple[str, str]] = []
    for name in bound_references:
        role = _still_ref_role(name)
        if role is not None:
            active.append((name, role))
    if not active:
        return None
    if reference_notation == "prose":
        if plate_attached and [name for name, _ in active] == ["reference_image", "set_image"]:
            return PLATE_NOTE
        if len(active) <= 1:
            return None
        clauses = []
        for index, (_, role) in enumerate(active):
            ordinal = _ORDINALS[index] if index < len(_ORDINALS) else f"#{index + 1}"
            clauses.append(f"The {ordinal} reference image shows {role}.")
        return " ".join(clauses)
    clauses = [f"@image{index + 1} shows {role}." for index, (_, role) in enumerate(active)]
    return " ".join(clauses)


def _sheet_panels(layout: str) -> str:
    return PANEL_LIST_COLUMNS if layout == "columns" else PANEL_LIST


def _sheet_studio(layout: str, background: str) -> str:
    del layout  # panel layout is already described by _sheet_panels(); omit text labels
    bg = _SHEET_BACKGROUNDS.get(background, _SHEET_BACKGROUNDS["white"])
    return (
        f"{bg} {_SHEET_BODY_REALISM} "
        "No text, no labels, no captions, no watermarks anywhere in or around the panels."
    )


def _format_character_sheet(
    manifest: Manifest,
    template: str,
    layout: str,
    state_id: str | None = None,
    character_id: str | None = None,
    *,
    background: str = "white",
    pin_details: bool,
) -> str:
    character = manifest.character_for(character_id)
    description = (
        character.describe(state_id) if state_id is not None else character.portrait_text()
    )
    if pin_details:
        description = _with_pinned(character, description)
    return template.format(
        layout=PANEL_LAYOUT[layout],
        panels=_sheet_panels(layout),
        studio=_sheet_studio(layout, background),
        description=description,
    )


def sheet_prompt(
    manifest: Manifest,
    layout: str,
    state_id: str | None = None,
    character_id: str | None = None,
    *,
    background: str = "white",
) -> str:
    return _format_character_sheet(
        manifest,
        SHEET_TEMPLATE,
        layout,
        state_id,
        character_id,
        background=background,
        pin_details=True,
    )


def derived_sheet_prompt(
    manifest: Manifest,
    layout: str,
    state_id: str | None = None,
    character_id: str | None = None,
    *,
    background: str = "white",
) -> str:
    return _format_character_sheet(
        manifest,
        DERIVED_CHARACTER_SHEET_TEMPLATE,
        layout,
        state_id,
        character_id,
        background=background,
        pin_details=False,
    )


def state_sheet_prompt(
    manifest: Manifest,
    state_id: str,
    layout: str,
    character_id: str | None = None,
    *,
    background: str = "white",
) -> str:
    character = manifest.character_for(character_id)
    state = character.state(state_id)
    invariants = character.pinned()
    invariants_note = f" Keep {_sentence(invariants)}" if invariants else ""
    return STATE_SHEET_TEMPLATE.format(
        invariants_note=invariants_note,
        layout=PANEL_LAYOUT[layout],
        panels=_sheet_panels(layout),
        wardrobe=state.description if state else "",
        studio=_sheet_studio(layout, background),
    )


def reference_video_prompt(
    manifest: Manifest,
    state_id: str | None = None,
    character_id: str | None = None,
    *,
    layout: str = "row",
    background: str = "white",
    chained_from_base: bool = False,
) -> str:
    """Prompt for generating a silent multi-angle character turnaround reference video."""
    character = manifest.character_for(character_id)
    bg_label = "neutral grey" if background == "grey" else "plain white"
    parts = [
        "A silent character reference video laid out in the exact multi-panel grid of the "
        "reference sheet, showing the same person from multiple angles and with natural "
        "expressions so face, hair and outfit stay consistent.",
        "Static camera in every panel. Single continuous shot with no cuts. All panel dividers "
        "stay fixed in place.",
    ]
    if chained_from_base:
        parts.append(
            "Preserve the exact face, body proportions, and natural movement style from the base "
            "character reference video while wearing the wardrobe shown on the reference sheet."
        )
    if layout == "columns":
        parts.append(
            "Center full-body front panel: casually turns left and right with small natural steps "
            "and relaxed weight shifts. Right full-body back panel: turns smoothly to show side "
            "silhouettes then faces away again. Left top frontal headshot: slowly turns head and "
            "shoulders around to show both profiles and back of the hair. Left bottom profile "
            "headshot: moves naturally through expressions (calm, subtle smirk, serious gaze)."
        )
    else:
        parts.append(
            "Full-body panels shift weight casually and rotate slightly so all sides of the "
            "costume and hair read cleanly; close-up face panel cycles smoothly through subtle "
            "natural expressions (neutral, slight smirk, focused gaze) without exaggerating."
        )
    if pinned := character.pinned():
        parts.append(f"Locked details: {_sentence(pinned)}")
    state = (
        character.state(state_id)
        if state_id
        else (character.wardrobe[0] if character.wardrobe else None)
    )
    if state is not None:
        parts.append(f"Outfit, identical across every panel: {state.description}")
    parts.append(
        f"Same {bg_label} studio background and flat diffuse lighting as the sheet. "
        "Silent, no audio, no music, no text."
    )
    return " ".join(parts)


def _format_plate_prompt(manifest: Manifest, set_name: str, template: str, **extra: str) -> str:
    bible = manifest.visual_bible
    scene = bible.find_set(set_name)
    return template.format(
        target_name=scene.name if scene else set_name,
        description=scene.description if scene else set_name,
        style=_style_line(bible.style, bible.palette, bible.lighting),
        **extra,
    ).replace("  ", " ")


def plate_prompt(manifest: Manifest, set_name: str) -> str:
    return _format_plate_prompt(manifest, set_name, PLATE_TEMPLATE)


def plate_style_prompt(manifest: Manifest, set_name: str) -> str:
    return _format_plate_prompt(manifest, set_name, PLATE_STYLE_TEMPLATE)


def plate_embed_prompt(manifest: Manifest, set_name: str, source_name: str) -> str:
    return _format_plate_prompt(manifest, set_name, PLATE_EMBED_TEMPLATE, source_name=source_name)


def plate_edit_prompt(manifest: Manifest, set_name: str, source_name: str) -> str:
    return _format_plate_prompt(manifest, set_name, PLATE_EDIT_TEMPLATE, source_name=source_name)


def prop_prompt(manifest: Manifest, prop_query: str) -> str:
    bible = manifest.visual_bible
    prop = bible.find_prop(prop_query)
    description = (
        prop.description if prop and prop.description else (prop.name if prop else prop_query)
    )
    return PROP_TEMPLATE.format(
        description=description,
        style=_style_line(bible.style, bible.palette, bible.lighting),
    ).replace("  ", " ")


def join_names(names: Sequence[str]) -> str:
    """Format 1, 2, or 3+ display names cleanly ('A', 'A and B', 'A, B, and C')."""
    items = [n for n in names if n]
    if len(items) <= 2:
        return " and ".join(items)
    return f"{', '.join(items[:-1])}, and {items[-1]}"


def _sentence(text: str) -> str:
    """``text`` ended with a full stop, unless it already ends a sentence."""
    text = text.strip()
    return text if not text or text[-1] in ".!?" else f"{text}."


def _with_pinned(character: Character, text: str, *, framing: Framing | None = None) -> str:
    """``text`` followed by the character's pinned details, as one more sentence."""
    text, pinned = text.strip(), character.pinned()
    if pinned and framing in (Framing.CLOSE_UP, Framing.MEDIUM_CLOSE):
        pinned = scrub_lower_body_wardrobe(
            pinned, include_waist=(framing is Framing.CLOSE_UP), allow_empty=True
        )
    if not pinned:
        return text
    return f"{_sentence(text)} Always: {_sentence(pinned)}".strip()


def _look(
    character: Character,
    state_id: str | None,
    *,
    framing: Framing | None = None,
) -> str:
    """Appearance and costume, plus the details pinned for every still of this character.

    The reference images carry identity, but a second character in a two-shot gets fewer of them
    and drifts (hair, sleeves, hair colour). Writing the pinned details into the prompt holds them.
    Empty for a character with no appearance, so callers skip them (pinned details need a look
    to attach to, and a reference sheet is only made for a character with an appearance).
    If a non-empty ``state_id`` no longer matches any wardrobe state (e.g. after editing a
    manifest), fall back to the first wardrobe state so the character never loses their outfit.
    """
    if not character.appearance.strip():
        return ""
    resolved = state_id
    if resolved and character.state(resolved) is None and character.wardrobe:
        resolved = character.wardrobe[0].id
    described = character.describe(resolved)
    if framing in (Framing.CLOSE_UP, Framing.MEDIUM_CLOSE) and " Wearing: " in described:
        app_part, wear_part = described.split(" Wearing: ", 1)
        trimmed_wear = scrub_lower_body_wardrobe(
            wear_part, include_waist=(framing is Framing.CLOSE_UP)
        )
        described = f"{app_part} Wearing: {trimmed_wear}".strip()
    return _with_pinned(character, described, framing=framing)


def compose_still_prompt(
    manifest: Manifest,
    shot: Shot,
    plate_attached: bool = False,
    *,
    props_attached: bool = False,
    reference_notation: str = "prose",
    bound_references: list[str] | None = None,
) -> str:
    """The prompt for a shot's still.

    With a set plate attached as a second reference image, the plate carries the location, so its
    text description is left out and a note explains the two images instead.
    """
    bible = manifest.visual_bible
    raw_still = (shot.still.prompt_override or shot.still_prompt).strip()
    parts: list[str] = []
    if shot.type.shows_singer and not _EXPLICIT_TIGHT_FRAMING_RE.search(raw_still):
        if shot.framing is Framing.CLOSE_UP:
            parts.append("Tight head-and-shoulders close-up portrait (face large in frame).")
        elif shot.framing is Framing.MEDIUM_CLOSE:
            parts.append("Chest-up medium close-up shot (head and upper chest fill frame).")
    parts.append(raw_still)
    scene = bible.find_set(shot.setting)
    ref_note = _still_reference_note(plate_attached, reference_notation, bound_references)
    if ref_note is not None:
        parts.append(ref_note)
    if not plate_attached and scene is not None and scene.description:
        if shot.framing is Framing.CLOSE_UP and shot.type is ShotType.BROLL:
            if shot.setting:
                parts.append(
                    f"Setting: {shot.setting} "
                    "(tight close-up; keep distant room elements out of frame)."
                )
        elif shot.framing is Framing.CLOSE_UP:
            first_sent = re.split(r"(?<=[.!?])\s+", scene.description.strip(), maxsplit=1)[0]
            first_sent = first_sent.rstrip(".!?").strip()
            parts.append(f"Setting: {first_sent} (tight close-up; soft-focus background only).")
        else:
            parts.append(f"Setting: {scene.description}")
    resolved_props = bible.resolve_props(shot.props)
    if resolved_props:
        if props_attached:
            names = ", ".join(p.name for p in resolved_props)
            parts.append(
                f"Match the exact design, scale and details of the prop reference ({names})."
            )
        else:
            for prop in resolved_props:
                if prop.description:
                    parts.append(f"Hero prop ({prop.name}): {prop.description}")
    visible = manifest.visible_characters(shot) if shot.type.shows_singer else []
    singing = manifest.singing_characters(shot)
    singing_ids = {cid for cid, _, _ in singing if cid is not None}
    if shot.type is ShotType.LIPSYNC:
        if len(singing) >= 2:
            names = join_names([c.name or cid or "performer" for cid, c, _ in singing])
            prefix = "Both" if len(singing) == 2 else "All singers (" + names + ")"
            together_line = (
                f"Both {names} singing together, both faces clearly visible and "
                "mouths unobstructed."
                if len(singing) == 2
                else f"{prefix} singing together, faces clearly visible and mouths unobstructed."
            )
            parts.append(together_line)
        elif len(visible) > 1 and len(singing) == 1:
            solo_id, solo_char, _ = singing[0]
            solo_label = solo_char.name or solo_id or "lead"
            listeners = [c.name or cid or "performer" for cid, c, _ in visible if cid != solo_id]
            listener_text = join_names(listeners)
            verb = "listen" if len(listeners) > 1 else "listens"
            mouth_noun = "mouths" if len(listeners) > 1 else "mouth"
            parts.append(
                f"Face clearly visible, mouth unobstructed. Only {solo_label} is singing; "
                f"{listener_text} {verb} with {mouth_noun} closed."
            )
        else:
            parts.append("Face clearly visible, mouth unobstructed.")
    if shot.type.shows_singer:
        parts.append(_FRAMING_SCALE_NOTE.get(shot.framing, CAMERA_PLACEMENT_NOTE))
        raw_still_text = shot.still.prompt_override or shot.still_prompt
        rear_view = shot.type is not ShotType.LIPSYNC and is_rear_view_prompt(raw_still_text)
        if rear_view:
            parts.append(
                "Camera behind subject (rear view): show only the back of the head and back of "
                "the outfit, turned completely away from the camera with face out of view."
            )
        hide_facial = rear_view or shot.framing is Framing.WIDE
        has_lipsync = any(s.type is ShotType.LIPSYNC for s in manifest.shots)
        lead_noun = "The singer" if has_lipsync else "The character"
        if (
            bound_references
            and len(bound_references) > 1
            and len(visible) >= 1
            and reference_notation != "tags"
        ):
            # Explicit Picture N identity disambiguation (clean-room concept inspired by
            # heroesjouney/aimoviestudiov2 prompt_builder.py).
            slot_notes: list[str] = []
            p_label = visible[0][1].name or visible[0][0] or "the primary singer"
            for slot_idx, param in enumerate(bound_references, start=1):
                if param == "reference_image":
                    slot_notes.append(f"Picture {slot_idx} defines {p_label}'s appearance")
                elif param == "reference_face":
                    slot_notes.append(
                        f"Picture {slot_idx} defines {p_label}'s close-up facial features"
                    )
                elif param == "reference_profile":
                    slot_notes.append(f"Picture {slot_idx} defines {p_label}'s side profile")
                elif param == "reference_three_quarter":
                    slot_notes.append(f"Picture {slot_idx} defines {p_label}'s three-quarter view")
                elif (char_idx := _extra_reference_index(param)) is not None and char_idx < len(
                    visible
                ):
                    default_label = (
                        "the second performer" if char_idx == 1 else f"performer {char_idx + 1}"
                    )
                    s_label = visible[char_idx][1].name or visible[char_idx][0] or default_label
                    slot_notes.append(f"Picture {slot_idx} defines {s_label}'s appearance")
                elif param == "set_image":
                    slot_notes.append(f"Picture {slot_idx} defines only the empty location")
            if slot_notes:
                parts.append("; ".join(slot_notes) + ".")
        if len(visible) == 1:
            char_id, character, state_id = visible[0]
            if look := _look(character, state_id, framing=shot.framing):
                if hide_facial:
                    look = scrub_frontal_facial_details(look)
                if len(manifest.characters) <= 1 or char_id is None:
                    parts.append(f"{lead_noun}: {look}")
                else:
                    label = character.name or char_id
                    parts.append(f"{lead_noun} ({label}): {look}")
        elif len(visible) > 1:
            primary_id, primary_char, primary_state = visible[0]
            if look := _look(primary_char, primary_state, framing=shot.framing):
                if hide_facial:
                    look = scrub_frontal_facial_details(look)
                label = primary_char.name or primary_id or "lead"
                parts.append(f"{lead_noun} ({label}, matches reference portrait): {look}")
            for sec_id, sec_char, sec_state in visible[1:]:
                if look := _look(sec_char, sec_state, framing=shot.framing):
                    if hide_facial:
                        look = scrub_frontal_facial_details(look)
                    sec_label = sec_char.name or sec_id or "performer"
                    role_prefix = (
                        f"Co-singer in frame ({sec_label})"
                        if sec_id in singing_ids
                        else f"Also in frame ({sec_label})"
                    )
                    parts.append(f"{role_prefix}: {look}")
    chapter = manifest.find_chapter(shot.chapter_id)
    palette = chapter.palette if chapter and chapter.palette else bible.palette
    lighting = chapter.lighting if chapter and chapter.lighting else bible.lighting
    if bible.world_facts:
        ledger = build_ledger_from_facts(
            (f.entity, f.attribute, f.value, f.start_shot, f.irreversible)
            for f in bible.world_facts
        )
        active_states = ledger.state_at(shot.index)
        if active_states:
            lighting = filter_look_for_state(lighting, active_states)
            style_filtered = filter_look_for_state(bible.style, active_states)
            parts.append(_style_line(style_filtered, palette, lighting))
        else:
            parts.append(_style_line(bible.style, palette, lighting))
    else:
        parts.append(_style_line(bible.style, palette, lighting))
    return " ".join(part.strip() for part in parts if part.strip())


def _video_reference_note(
    reference_notation: str,
    bound_references: list[str] | None,
) -> str | None:
    """Reference-image role cue for R2V video models (``@image1`` is the starting still frame)."""
    if not bound_references:
        return None
    active: list[tuple[str, str]] = []
    for name in bound_references:
        role = _still_ref_role(name)
        if role is not None:
            active.append((name, role))
    if not active:
        return None
    if reference_notation == "tags":
        clauses = [
            f"@image{index} shows {role}." for index, (_, role) in enumerate(active, start=2)
        ]
        return " ".join(clauses)
    clauses = []
    for index, (_, role) in enumerate(active):
        ordinal = _ORDINALS[index] if index < len(_ORDINALS) else f"#{index + 1}"
        clauses.append(f"The {ordinal} attached reference image shows {role}.")
    return " ".join(clauses)


def compose_motion_prompt(
    manifest: Manifest,
    shot: Shot,
    *,
    video_config: VideoConfig | None = None,
    reference_notation: str = "prose",
    has_audio_video: bool = False,
    previz_attached: bool = False,
    bound_references: list[str] | None = None,
) -> str:
    """Assemble the full motion prompt for video generation."""
    single_take = video_config.motion_single_take if video_config is not None else True
    no_subtitles = video_config.motion_no_subtitles if video_config is not None else True
    no_music = video_config.motion_no_music if video_config is not None else True
    include_lyrics = video_config.motion_include_lyrics if video_config is not None else False

    parts: list[str] = []
    if previz_attached:
        parts.append(
            "Use the previz reference video to define the pacing, camera angles, camera "
            "movement, and framing, mapping realistic textures and lighting onto the scene."
        )
        if shot.previz_mapping.strip():
            parts.append(f"Stand-in mapping: {shot.previz_mapping.strip()}")
    parts.append(strip_still_words(shot.video.prompt_override or shot.motion_prompt))
    if reference_notation == "tags":
        parts.append("Animate starting from @image1.")
    video_ref_note = _video_reference_note(reference_notation, bound_references)
    if video_ref_note is not None:
        parts.append(video_ref_note)
    if shot.allow_cuts:
        parts.append(
            "Hard cuts between distinct camera angles within the clip while keeping the "
            "character, wardrobe, and location consistent across every cut."
        )
    if shot.type is ShotType.LIPSYNC:
        target_audio = (
            "in time with the vocal in @video1"
            if (has_audio_video and reference_notation == "tags")
            else "in time with the vocal"
        )
        singing = manifest.singing_characters(shot)
        visible = manifest.visible_characters(shot)
        if len(singing) >= 2:
            names = join_names([c.name or cid or "performer" for cid, c, _ in singing])
            lead = f"Both {names}" if len(singing) == 2 else names
            parts.append(f"{lead} sing together with clear, natural mouth movement {target_audio}.")
        elif len(singing) == 1 and len(visible) > 1:
            solo_id, solo_char, _ = singing[0]
            solo_label = solo_char.name or solo_id or "The singer"
            listeners = [c.name or cid or "performer" for cid, c, _ in visible if cid != solo_id]
            listener_text = join_names(listeners)
            verb = "listen" if len(listeners) > 1 else "listens"
            mouth_noun = "mouths" if len(listeners) > 1 else "mouth"
            parts.append(
                f"{solo_label} sings with clear, natural mouth movement {target_audio} while "
                f"{listener_text} {verb} with {mouth_noun} closed, not singing."
            )
        elif len(singing) == 1 and len(manifest.characters) > 1 and singing[0][0] is not None:
            solo_id, solo_char, _ = singing[0]
            solo_label = solo_char.name or solo_id or "The singer"
            parts.append(f"{solo_label} sings with clear, natural mouth movement {target_audio}.")
        else:
            parts.append(f"The singer sings with clear, natural mouth movement {target_audio}.")
        lyrics = " ".join(shot.lyrics.replace('"', "'").split())
        if include_lyrics and lyrics:
            parts.append(f'The song lyrics are: "{lyrics}".')
    if single_take and not shot.allow_cuts:
        framing = _FRAMING_LABEL[shot.framing]
        parts.append(f"Maintain {framing} framing throughout. Single continuous shot, no cuts.")
    if no_subtitles and manifest.mode is not ProjectMode.MIX:
        parts.append("No subtitles, no text.")
    if no_music:
        parts.append("No music.")
    if manifest.visual_bible.style:
        parts.append(f"Style: {manifest.visual_bible.style}.")
    return " ".join(part.strip() for part in parts if part.strip())


def _style_line(style: str, palette: str, lighting: str) -> str:
    labelled = [("Style", style), ("Palette", palette), ("Lighting", lighting)]
    return " ".join(f"{label}: {value.rstrip('.')}." for label, value in labelled if value)
