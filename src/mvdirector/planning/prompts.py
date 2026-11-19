# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Prompt text for the planner and for composing generation prompts.

Kept together so wording can be tuned without touching pipeline logic. The vision-check
prompt lives next to its code in ``qc/vision.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mvdirector.models import Character, Framing, Manifest, Shot, ShotType

if TYPE_CHECKING:
    from mvdirector.config import VideoConfig

BIBLE_SYSTEM = """You are the director of a music video that balances performer shots with
cutaways. Write the visual bible that every shot will follow.

- concept: two to four sentences on the idea that ties the video together.
- characters: every distinct singer, vocalist, or on-camera band/group member indicated by the
  style brief, section headers, or lyrics (one entry for a solo track; two for a duet; one per
  member or named character for a band, group, or theatrical piece). Every singer or recurring
  performer MUST have their own entry in 'characters' with:
  - id: short lowercase identifier (letters, digits and underscores, e.g. 'elias', 'odalie',
    'lead_vocal', 'guitarist').
  - name: display name or role (e.g. 'Elias', 'Odalie').
  - appearance: one dense paragraph on that person's PERMANENT physical look: age range, gender
    presentation, face shape, skin tone, natural skin texture (visible pores, subtle natural
    unevenness, real facial asymmetry), eyes, brows, nose, lips, hair (cut, length, parting,
    individual strands and flyaways), makeup, build, and any identifying marks. Give the side of
    any mark as the person's anatomical left or right. Forbid idealized, airbrushed, or glass-skin
    rendering so the person reads as a real unretouched photograph. Do not describe clothing here.
    It must be precise enough that an image model draws the same person every time. Do not
    describe a real, identifiable person.
  - wardrobe: that person's costume states in the order they appear in the song. Each has a short
    id (lowercase letters, digits and underscores) and a complete description of everything worn
    in that state, including fabric texture, weave, and natural drape. State explicitly what is
    absent compared with their previous state (for example 'no cape'). Only add a new state when
    clothing actually changes—never for poses, props, or rooms. Use one state if their costume
    never changes.
  - invariants: optional short phrase listing signature accessories or visual details that must
    survive across every costume change (e.g. 'glowing cyan hoop earrings in both ears'). It is
    added to every still of this character, so never put anything here that a costume change
    would alter (sleeves, jacket, shoes).
- appearance: copy the primary/lead singer's permanent appearance paragraph here as well.
- invariants: copy the primary/lead singer's invariants here as well.
- wardrobe: copy the primary/lead singer's wardrobe states here as well.
- style, palette, lighting: short phrases an image model can act on. Keep a consistent film stock
  and color grade across all sets while letting practical light sources vary per location.
- negative_prompt: comma-separated things to avoid.
- settings: 3 to 6 distinct sets. Each has a name, the sections of the song it serves
  (sections), a description of how it looks (for an image model, naming one primary mid-depth
  focal object so the lens has a clear plane of focus), how the performers move and the camera
  works there (movement), and an optional parent_set naming an earlier set when this location is
  a stripped, altered, or nested variant of it.
- props: 0 to 3 recurring hero objects central to the story or choreography (e.g. a glowing
  disc, vintage guitar, helmet). Each has id (short lowercase identifier), name, sections, and a
  description with approximate scale in centimeters, finish, and explicit negative exclusions
  (e.g. 'no center hub, no spokes').
- rules: 4 to 10 short rules that every shot follows (camera language, physical camera placement,
  continuity, effects, how much of the video shows the performers). For duets or groups, write
  rules that apply across the cast rather than locking every shot to a single character. Only
  rules a shot-by-shot director can act on."""

SHOTS_SYSTEM = """You are the director of a music video. Timing is already fixed. For each shot
you are given, choose the creative details.

Rules:
- Return exactly one entry per shot id you were given.
- character: the id of the primary character on camera in this shot (from CHARACTERS, when given).
  On 'lipsync' shots, 'character' MUST be the vocalist singing that shot's section/lyrics, and
  both still_prompt and motion_prompt must focus on that vocalist singing. Use null on 'broll'.
- characters: list of all character ids visible in the frame (primary 'character' first). On
  close_up shots, include only [character]. On medium_close, medium, or wide shots that show a
  duet interaction, two-shot, or band/group performance, list every character visible in the frame.
- setting: use the exact name of one set from the visual bible, chosen for the shot's song
  section. Follow each set's sections and movement notes.
- props: list of hero prop ids from PROPS visible in this shot (empty list when none appear).
- wardrobe: use the id of one wardrobe state for the shot's primary character. It is what that
  character wears in the shot's FIRST frame. A costume change may happen during a shot; the next
  shot then uses the new state. Each character's wardrobe only moves forward through their list.
  Never return to an earlier state. Use null on 'broll'.
- 'lipsync' shots show the active singer singing the given lyrics. Use framing close_up or
  medium_close (never medium or wide) so the singer's mouth is clearly visible and unobstructed
  (no hands, microphones, hair or foreground objects across the mouth).
- Shots given as 'performance' have no vocals. Keep them as 'performance' (performer on camera,
  not singing) or change them to 'broll' (no performer) where a cutaway serves the song, without
  putting more than two 'broll' shots in a row.
- Shots given as 'broll' are cutaways: the vocal plays over them and no performer is shown. Keep
  them 'broll', with character and wardrobe null, and show what the lyric is about (an object, a
  place, light, weather, or an architectural detail), never a face or performer. Still give
  every shot a framing (close_up, medium_close, medium or wide): never null, even on a cutaway.
- Match detail to shot size. Small details (identifying marks, jewelry, cuffs, props, screen
  content) belong in close_up or medium_close shots only. In medium and wide shots, describe pose,
  silhouette and action, not fine detail.
- Physical camera placement: always position the camera in open floor space at normal human
  standing or seated height on the clear side of any furniture. Never place the lens inside,
  below, level with, or passing through tables, walls, or solid objects, and never slice or
  occlude a performer's body with foreground furniture edges.
- Follow the production rules you are given.
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
- Something should visibly change every 2 to 4 seconds without a cut: light shifting, a flame or
  curtain moving, a background element, the performer's posture. Do not use camera moves for this
  on lipsync shots.
- Aim for one striking, printable frame in every 30 seconds: strong symmetry, a silhouette, or a
  single bright light against dark.
- still_prompt: describes the first frame only: framing, physical camera placement in open space,
  who is in frame, pose, action, camera angle. It is sent to an image-edit model together with a
  reference portrait of the primary singer, and the set and character descriptions are added
  automatically, so do not repeat full physical descriptions.
- motion_prompt: what happens in the clip: performance, body movement, and physical camera
  operation (describe handheld physics such as subtle operator breathing sway, small weight
  shifts, and soft starts and stops rather than bare labels). Keep camera moves gentle and stable
  on lipsync shots so the singing mouth stays clearly tracked. Every shot MUST remain a single
  continuous take with no internal cuts and no subtitles or on-screen text. For shots 4.0 seconds
  or longer with distinct movement beats or camera shifts, you may structure motion_prompt as
  timestamped sub-beats covering the shot duration (e.g. '0-2.0s: ... 2.0-4.0s: ...')."""

REWRITE_SYSTEM = """You fix prompts for an AI image or video model after a quality check
rejected the result. Rewrite the prompt so the listed problems are less likely, keeping the
shot's intent, framing and setting. Prefer simplifying (hands out of frame, calmer pose,
camera in open floor space clear of furniture, plainer background) over piling on instructions.
Return the new prompt only, in the 'prompt' field."""

BRIEF_SYSTEM = """You are a music video writer and Suno song producer. Given a short video
concept, optional target duration, and musical/visual vibe, write:

- lyrics: Suno-ready lyrics structured with [Section] headers ([Intro], [Verse], [Pre-Chorus],
  [Chorus], [Bridge], [Outro]; for duets, tag performer sections like [Verse 1: Name]). Keep line
  counts concise for the requested duration. Put ONLY sung lines or sung adlibs in (parentheses);
  never put unspoken SFX or stage directions in parentheses.
- suno_style: one dense Suno Custom Mode style prompt specifying genre, vocal register and
  phrasing, key instruments, BPM, rhythm feel, and stereo production polish.
- visual_direction: a short creative brief covering the performer look, key costume states,
  locations, lighting, palette, and story beats for mvdirector's visual bible."""

PERFORMANCE_MOTION = (
    "The singer moves with the music, mouth closed, not singing. Slow, steady camera drift."
)

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
    "One small label per panel: FRONT / SIDE / 3-QUARTER / FACE DETAIL."
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
    "reference_face": "a macro close-up of the singer's face",
    "reference_image_2": "the secondary performer: keep their face, hair and costume",
    "set_image": (
        "the location: place the performers in that location and match its layout, "
        "lighting and color grade exactly"
    ),
}


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
    active = [name for name in bound_references if name in _STILL_REF_ROLES]
    if not active:
        return None
    if reference_notation == "prose":
        if plate_attached and "reference_image_2" not in active:
            return PLATE_NOTE
        if "reference_image_2" not in active:
            return None
        clauses = []
        for index, name in enumerate(active):
            role = _STILL_REF_ROLES[name]
            ordinal = _ORDINALS[index] if index < len(_ORDINALS) else f"#{index + 1}"
            clauses.append(f"The {ordinal} reference image shows {role}.")
        return " ".join(clauses)
    clauses = [
        f"@image{index + 1} shows {_STILL_REF_ROLES[name]}." for index, name in enumerate(active)
    ]
    return " ".join(clauses)


def _sheet_panels(layout: str) -> str:
    return PANEL_LIST_COLUMNS if layout == "columns" else PANEL_LIST


def _sheet_studio(layout: str, background: str) -> str:
    bg = _SHEET_BACKGROUNDS.get(background, _SHEET_BACKGROUNDS["white"])
    labels = (
        "FRONTAL / PROFILE / FULL FRONT / FULL BACK."
        if layout == "columns"
        else "FRONT / SIDE / 3-QUARTER / FACE DETAIL."
    )
    return f"{bg} {_SHEET_BODY_REALISM} One small label per panel: {labels}"


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


def _sentence(text: str) -> str:
    """``text`` ended with a full stop, unless it already ends a sentence."""
    text = text.strip()
    return text if not text or text[-1] in ".!?" else f"{text}."


def _with_pinned(character: Character, text: str) -> str:
    """``text`` followed by the character's pinned details, as one more sentence."""
    text, pinned = text.strip(), character.pinned()
    if not pinned:
        return text
    return f"{_sentence(text)} Always: {_sentence(pinned)}".strip()


def _look(character: Character, state_id: str | None) -> str:
    """Appearance and costume, plus the details pinned for every still of this character.

    The reference images carry identity, but a second character in a two-shot gets fewer of them
    and drifts (hair, sleeves, hair colour). Writing the pinned details into the prompt holds them.
    Empty for a character with no appearance, so callers skip them (pinned details need a look
    to attach to, and a reference sheet is only made for a character with an appearance).
    """
    if not character.appearance.strip():
        return ""
    return _with_pinned(character, character.describe(state_id))


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
    parts = [shot.still.prompt_override or shot.still_prompt]
    scene = bible.find_set(shot.setting)
    ref_note = _still_reference_note(plate_attached, reference_notation, bound_references)
    if ref_note is not None:
        parts.append(ref_note)
    if not plate_attached and scene is not None and scene.description:
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
    if shot.type is ShotType.LIPSYNC:
        parts.append("Face clearly visible, mouth unobstructed.")
    if shot.type.shows_singer:
        parts.append(CAMERA_PLACEMENT_NOTE)
        visible = manifest.visible_characters(shot)
        if len(visible) == 1:
            char_id, character, state_id = visible[0]
            if look := _look(character, state_id):
                if len(manifest.characters) <= 1 or char_id is None:
                    parts.append(f"The singer: {look}")
                else:
                    label = character.name or char_id
                    parts.append(f"The singer ({label}): {look}")
        elif len(visible) > 1:
            primary_id, primary_char, primary_state = visible[0]
            if look := _look(primary_char, primary_state):
                label = primary_char.name or primary_id or "lead"
                parts.append(f"The singer ({label}, matches reference portrait): {look}")
            for sec_id, sec_char, sec_state in visible[1:]:
                if look := _look(sec_char, sec_state):
                    sec_label = sec_char.name or sec_id or "performer"
                    parts.append(f"Also in frame ({sec_label}): {look}")
    parts.append(_style_line(bible.style, bible.palette, bible.lighting))
    return " ".join(part.strip() for part in parts if part.strip())


def compose_motion_prompt(
    manifest: Manifest,
    shot: Shot,
    *,
    video_config: VideoConfig | None = None,
    reference_notation: str = "prose",
    has_audio_video: bool = False,
    previz_attached: bool = False,
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
    parts.append(shot.video.prompt_override or shot.motion_prompt)
    if reference_notation == "tags":
        parts.append("Animate starting from @image1.")
    if shot.allow_cuts:
        parts.append(
            "Hard cuts between distinct camera angles within the clip while keeping the "
            "character, wardrobe, and location consistent across every cut."
        )
    if shot.type is ShotType.LIPSYNC:
        if has_audio_video and reference_notation == "tags":
            parts.append(
                "The singer sings with clear, natural mouth movement in time with the vocal "
                "in @video1."
            )
        else:
            parts.append(
                "The singer sings with clear, natural mouth movement in time with the vocal."
            )
        lyrics = " ".join(shot.lyrics.split())
        if include_lyrics and lyrics:
            parts.append(f'The song lyrics are: "{lyrics}".')
    if single_take and not shot.allow_cuts:
        framing = _FRAMING_LABEL[shot.framing]
        parts.append(f"Maintain {framing} framing throughout. Single continuous shot, no cuts.")
    if no_subtitles:
        parts.append("No subtitles, no text.")
    if no_music:
        parts.append("No music.")
    if manifest.visual_bible.style:
        parts.append(f"Style: {manifest.visual_bible.style}.")
    return " ".join(part.strip() for part in parts if part.strip())


def _style_line(style: str, palette: str, lighting: str) -> str:
    labelled = [("Style", style), ("Palette", palette), ("Lighting", lighting)]
    return " ".join(f"{label}: {value.rstrip('.')}." for label, value in labelled if value)
