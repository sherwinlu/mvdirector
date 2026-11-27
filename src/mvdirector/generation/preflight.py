# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Check, before any still is generated, that the assets it depends on are locked.

A series stays consistent when every character sheet and location plate is approved before the
first generation. A missing asset otherwise shows up as a run of failed or inconsistent stills.
"""

from __future__ import annotations

import re
import shlex
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from mvdirector.comfy.workflow import WorkflowTemplate
from mvdirector.errors import ConfigError
from mvdirector.models import Manifest, ascii_fold
from mvdirector.project import Project


@dataclass
class PreflightResult:
    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def raise_if_blocked(self) -> None:
        if self.problems:
            raise ConfigError(
                "Assets are not ready for stills:\n- "
                + "\n- ".join(self.problems)
                + "\nFix these, or pass --skip-preflight to continue with fallbacks."
            )


_EXTRA_PROP_BINDING_RE = re.compile(r"^prop_image_([1-9][0-9]*)\Z")


def _accepts_prop_binding(template: WorkflowTemplate | None) -> bool:
    if template is None:
        return False
    if template.accepts("prop_image"):
        return True
    return any(
        _EXTRA_PROP_BINDING_RE.match(name) and int(name.rsplit("_", 1)[1]) >= 2
        for name in template.bindings.inputs
    )


def check_assets(
    project: Project,
    manifest: Manifest,
    templates: dict[str, WorkflowTemplate],
    shot_ids: list[str] | None = None,
) -> PreflightResult:
    """List what is missing for the selected shots (all shots by default)."""
    result = PreflightResult()
    character, bible = manifest.character, manifest.visual_bible
    shots = [s for s in manifest.shots if shot_ids is None or s.id in shot_ids]
    singer = [s for s in shots if s.type.shows_singer]

    still = templates.get("still")
    lipsync = templates.get("lipsync")
    i2v = templates.get("i2v")
    video_templates = [t for t in (lipsync, i2v) if t is not None]
    needs_ref = (
        (still is None and not video_templates)
        or (still is not None and still.accepts("reference_image"))
        or any("reference_image" in t.bindings.required for t in video_templates)
    )
    if len(manifest.characters) > 1:
        used_by_char: dict[str, set[str]] = {}
        for s in singer:
            for cid, _char, state_id in manifest.visible_characters(s):
                if cid:
                    used_by_char.setdefault(cid, set())
                    if state_id:
                        used_by_char[cid].add(state_id)
        for char_id, char in manifest.characters.items():
            if char_id not in used_by_char:
                continue
            if needs_ref:
                if char.default_reference is None:
                    result.problems.append(
                        f"No reference sheet is locked for '{char_id}'. Run `mvd reference "
                        f"generate --character {char_id}`, then `mvd reference lock <image> "
                        f"--character {char_id}`."
                    )
                for state in char.states_without_reference(used_by_char[char_id]):
                    result.problems.append(
                        f"No reference sheet for '{char_id}' wardrobe state '{state}'. Run "
                        f"`mvd reference generate --character {char_id} --state {state}`, then "
                        f"`mvd reference lock <image> --character {char_id} --state {state}`."
                    )
            for key, reference in char.references.items():
                for path in [reference.sheet, *reference.panels.values()]:
                    if path and not project.path(path).is_file():
                        result.problems.append(
                            f"Reference file for '{char_id}:{key}' is missing: {path}"
                        )
    else:
        if singer and needs_ref:
            if character.default_reference is None:
                result.problems.append(
                    "No reference sheet is locked. Run `mvd reference generate`, then "
                    "`mvd reference lock <image>`."
                )
            used = {s.wardrobe for s in singer if s.wardrobe}
            for state in character.states_without_reference(used):
                result.problems.append(
                    f"No reference sheet for wardrobe state '{state}'. Run `mvd reference generate "
                    f"--state {state}`, then `mvd reference lock <image> --state {state}`."
                )
        if singer:
            for key, reference in character.references.items():
                for path in [reference.sheet, *reference.panels.values()]:
                    if path and not project.path(path).is_file():
                        result.problems.append(f"Reference file for '{key}' is missing: {path}")
    plate_templates = [t for t in (still, lipsync, i2v) if t is not None and t.accepts("set_image")]
    if still is not None and still.accepts("set_image"):
        for name in sorted({s.setting for s in singer if s.setting}):
            scene = bible.find_set(name)
            if scene is None:
                continue
            if not scene.plate and not scene.panels:
                result.problems.append(
                    f"No plate for set '{scene.name}'. Run `mvd plates generate --set "
                    f'"{scene.name}"`, then `mvd plates lock <image> --set "{scene.name}"`.'
                )
            else:
                for path in [scene.plate, scene.sheet, *scene.panels.values()]:
                    if path and not project.path(path).is_file():
                        result.problems.append(
                            f"Plate file for set '{scene.name}' is missing: {path}"
                        )
    elif plate_templates:
        for name in sorted({s.setting for s in shots if s.setting}):
            scene = bible.find_set(name)
            if scene is None:
                continue
            for path in [scene.plate, scene.sheet, *scene.panels.values()]:
                if path and not project.path(path).is_file():
                    result.problems.append(f"Plate file for set '{scene.name}' is missing: {path}")
    else:
        plated = [s.name for s in bible.settings if s.plate or s.panels]
        if plated:
            result.warnings.append(
                f"Plates are locked for {', '.join(plated)}, but the 'still' workflow has no "
                "'set_image' binding, so they will not be used."
            )
    for s in shots:
        if s.plate_angle and s.setting:
            scene = bible.find_set(s.setting)
            if scene is not None and scene.panels:
                panel_rel = scene.panels.get(s.plate_angle)
                if not panel_rel or not project.path(panel_rel).is_file():
                    result.problems.append(
                        f"Turnaround plate angle '{s.plate_angle}' for shot '{s.id}' in set "
                        f"'{scene.name}' is missing."
                    )
    t2i = templates.get("t2i")
    for s in shots:
        if not s.sketch_image:
            continue
        if not project.path(s.sketch_image).is_file():
            result.problems.append(f"Sketch image for '{s.id}' is missing: {s.sketch_image}")
            continue
        tpl = still if s.type.shows_singer else t2i
        if tpl is not None and not (
            tpl.accepts("sketch_image")
            or (not s.type.shows_singer and tpl.accepts("reference_image"))
            or any(tpl.accepts(f"reference_image_{n}") for n in range(2, 11))
            or s.sketch_guidance.strip()
        ):
            result.warnings.append(
                f"Shot '{s.id}' has sketch_image ({s.sketch_image}), but workflow '{tpl.name}' "
                "has no sketch_image or extra reference_image slot and no sketch_guidance text."
            )
    prop_queries = [
        q
        for s in shots
        if s.props
        for q in s.props
        if _accepts_prop_binding(still if s.type.shows_singer else t2i)
    ]
    if prop_queries:
        for prop in bible.resolve_props(prop_queries):
            if not prop.plate:
                result.problems.append(
                    f"No sheet locked for prop '{prop.name}'. Run `mvd props generate --prop "
                    f'"{prop.id}"`, then `mvd props lock <image> --prop "{prop.id}"`.'
                )
            elif not project.path(prop.plate).is_file():
                result.problems.append(f"Prop sheet for '{prop.name}' is missing: {prop.plate}")
    else:
        locked_props = [p.name for p in bible.props if p.plate]
        prop_bound = any(_accepts_prop_binding(t) for t in (still, t2i))
        if locked_props and not prop_bound:
            result.warnings.append(
                f"Prop sheets are locked for {', '.join(locked_props)}, but the workflow has "
                "no 'prop_image' binding, so they will not be used."
            )
    return result


# ------------------------------------------------------------------------- reference coverage lint

_WORD_RE = re.compile(r"[a-z]+(?:-[a-z]+)?")
_PREPOSITION_PREFIX = r"(?:with|holding|carrying|on|beside|near|inside|under|over|by|of)"
_DETERMINER_PREFIX = (
    r"(?:a|an|the|this|that|these|those|their|his|her|its|"
    r"several|some|many|few|two|three|both|each|every)"
)
_CHUNK_RE = re.compile(
    rf"(?:^|[\s,(;:])(?:{_PREPOSITION_PREFIX}(?:\s+{_DETERMINER_PREFIX})?|{_DETERMINER_PREFIX})\s+"
    r"([a-z]{3,}(?:-[a-z]{2,})?(?:\s+[a-z]{3,}(?:-[a-z]{2,})?)?)\b"
)

_MATERIAL_MODIFIERS = frozenset(
    {
        "brass",
        "brick",
        "bronze",
        "ceramic",
        "chrome",
        "copper",
        "crystal",
        "glass",
        "iron",
        "ivory",
        "leather",
        "marble",
        "metal",
        "oak",
        "paper",
        "pine",
        "plastic",
        "silk",
        "steel",
        "stone",
        "velvet",
        "wood",
        "wooden",
    }
)

_SET_LOCATION_NOUNS = frozenset(
    {
        "alley",
        "alleyway",
        "attic",
        "balcony",
        "bar",
        "basement",
        "beach",
        "bedroom",
        "bridge",
        "cafe",
        "canyon",
        "cathedral",
        "chapel",
        "club",
        "corridor",
        "courtyard",
        "desert",
        "diner",
        "forest",
        "garage",
        "hallway",
        "kitchen",
        "library",
        "lobby",
        "meadow",
        "pier",
        "platform",
        "rooftop",
        "stairwell",
        "station",
        "street",
        "studio",
        "subway",
        "theater",
        "theatre",
        "tunnel",
        "warehouse",
    }
)

_SINGULAR_KEEP = frozenset(
    {
        "brass",
        "canvas",
        "compass",
        "dress",
        "glass",
        "grass",
        "lens",
        "pass",
        "series",
        "species",
        "wings",
    }
)

_COVERAGE_STOPWORDS = frozenset(
    {
        # Articles, pronouns, prepositions, conjunctions, common adverbs
        "about",
        "above",
        "across",
        "after",
        "against",
        "almost",
        "alone",
        "along",
        "already",
        "also",
        "although",
        "always",
        "amid",
        "among",
        "another",
        "around",
        "away",
        "back",
        "because",
        "before",
        "behind",
        "below",
        "beneath",
        "beside",
        "between",
        "beyond",
        "both",
        "down",
        "during",
        "each",
        "either",
        "else",
        "enough",
        "even",
        "ever",
        "every",
        "everywhere",
        "from",
        "here",
        "hers",
        "herself",
        "himself",
        "however",
        "inside",
        "into",
        "itself",
        "just",
        "less",
        "like",
        "more",
        "most",
        "much",
        "must",
        "near",
        "neither",
        "never",
        "next",
        "none",
        "nothing",
        "nowhere",
        "only",
        "onto",
        "other",
        "others",
        "ours",
        "outside",
        "over",
        "past",
        "perhaps",
        "quite",
        "rather",
        "really",
        "same",
        "several",
        "should",
        "since",
        "some",
        "someone",
        "something",
        "somewhere",
        "soon",
        "still",
        "such",
        "than",
        "that",
        "their",
        "theirs",
        "them",
        "themselves",
        "then",
        "there",
        "these",
        "they",
        "thing",
        "things",
        "this",
        "those",
        "though",
        "through",
        "throughout",
        "together",
        "toward",
        "towards",
        # Directional adverbs from motion prompts ("a gentle forward push", "tilts upward")
        "backward",
        "backwards",
        "downward",
        "downwards",
        "forward",
        "forwards",
        "inward",
        "onward",
        "outward",
        "sideways",
        "upward",
        "upwards",
        "under",
        "until",
        "upon",
        "very",
        "well",
        "what",
        "whatever",
        "when",
        "where",
        "whether",
        "which",
        "while",
        "whom",
        "whose",
        "will",
        "with",
        "within",
        "without",
        "would",
        "your",
        "yours",
        "yourself",
        # Camera, framing, composition, shot direction, lighting, texture, atmosphere
        "ambient",
        "angle",
        "angles",
        "atmosphere",
        "atmospheric",
        "background",
        "beam",
        "beams",
        "blur",
        "blurred",
        "blurry",
        "bokeh",
        "border",
        "bright",
        "calm",
        "camera",
        "center",
        "centered",
        "cinematic",
        "close",
        "closeup",
        "close-up",
        "color",
        "colors",
        "colour",
        "composition",
        "contrast",
        "crane",
        "crop",
        "cutaway",
        "cuts",
        "dark",
        "darkness",
        "dawn",
        "depth",
        "detail",
        "details",
        "distance",
        "distant",
        "dolly",
        "dramatic",
        "drift",
        "dusk",
        "dust",
        "edge",
        "edges",
        "empty",
        "energy",
        "establishing",
        "evening",
        "exposure",
        "extreme",
        "fade",
        "field",
        "film",
        "flare",
        "floor",
        "focus",
        "focused",
        "fog",
        "foreground",
        "frame",
        "framed",
        "frames",
        "framing",
        "front",
        "full",
        "gentle",
        "glare",
        "glint",
        "glow",
        "glowing",
        "grade",
        "grain",
        "ground",
        "handheld",
        "haze",
        "high",
        "highlight",
        "highlights",
        "horizon",
        "huge",
        "image",
        "images",
        "insert",
        "intensity",
        "interior",
        "intimate",
        "keep",
        "large",
        "left",
        "lens",
        "light",
        "lighting",
        "lights",
        "line",
        "lines",
        "long",
        "look",
        "looks",
        "low",
        "medium",
        "medium-close",
        # Hyphenated shot jargon that reads as a noun phrase ("at eye-level", "a two-shot")
        "deep-focus",
        "eye-level",
        "high-angle",
        "low-angle",
        "three-quarter",
        "two-shot",
        "midground",
        "mist",
        "moment",
        "mood",
        "morning",
        "motion",
        "move",
        "movement",
        "moves",
        "moving",
        "natural",
        "neon",
        "night",
        "noon",
        "open",
        "orbit",
        "overhead",
        "pace",
        "palette",
        "pan",
        "panorama",
        "part",
        "parts",
        "particle",
        "particles",
        "pattern",
        "perspective",
        "place",
        "placement",
        "plain",
        "plane",
        "point",
        "portrait",
        "position",
        "profile",
        "pull",
        "pull-back",
        "push",
        "push-in",
        "rain",
        "rear",
        "reflection",
        "reflections",
        "right",
        "room",
        "scale",
        "scene",
        "second",
        "section",
        "sequence",
        "shadow",
        "shadows",
        "shallow",
        "sharp",
        "shift",
        "shifts",
        "short",
        "shot",
        "shots",
        "side",
        "sightline",
        "silhouette",
        "single",
        "size",
        "slow",
        "small",
        "smoke",
        "soft",
        "space",
        "spark",
        "sparks",
        "static",
        "steady",
        "style",
        "subject",
        "subtle",
        "sunlight",
        "surface",
        "sway",
        "symmetry",
        "take",
        "texture",
        "textures",
        "tight",
        "tilt",
        "time",
        "tiny",
        "tone",
        "tones",
        "track",
        "tracking",
        "view",
        "visual",
        "wall",
        "walls",
        "warm",
        "water",
        "weather",
        "wide",
        "widescreen",
        "wind",
        "zoom",
        # People, body, expression, generic performance & clothing adjectives/verbs
        "acts",
        "acting",
        "arms",
        "audience",
        "body",
        "breath",
        "breathing",
        "brow",
        "brows",
        "character",
        "characters",
        "cheek",
        "cheeks",
        "chest",
        "chin",
        "closed",
        "collarbone",
        "crowd",
        "dance",
        "dancer",
        "dancing",
        "expression",
        "eyes",
        "face",
        "faces",
        "figure",
        "figures",
        "finger",
        "fingers",
        "foot",
        "feet",
        "gaze",
        "gesture",
        "hair",
        "hand",
        "hands",
        "head",
        "heads",
        "human",
        "jaw",
        "lean",
        "leaning",
        "legs",
        "lips",
        "listen",
        "listens",
        "listening",
        "looking",
        "mouth",
        "mouths",
        "musician",
        "neck",
        "nose",
        "people",
        "perform",
        "performance",
        "performer",
        "performers",
        "person",
        "play",
        "playing",
        "plays",
        "pose",
        "posture",
        "react",
        "reacting",
        "reacts",
        "shoulder",
        "shoulders",
        "sing",
        "singer",
        "singers",
        "singing",
        "sings",
        "sits",
        "sitting",
        "smile",
        "smiling",
        "stage",
        "stand",
        "standing",
        "stands",
        "step",
        "steps",
        "stepping",
        "torso",
        "turn",
        "turning",
        "turns",
        "vocal",
        "vocalist",
        "vocals",
        "voice",
        "waist",
        "walk",
        "walking",
        "walks",
        "wear",
        "wearing",
        "wears",
        # Colors, materials/qualities that shouldn't stand alone as props, common verbs
        "amber",
        "appear",
        "appears",
        "bare",
        "begin",
        "begins",
        "black",
        "blue",
        "brightly",
        "broken",
        "brown",
        "carry",
        "carrying",
        "change",
        "changes",
        "changing",
        "clear",
        "cold",
        "come",
        "comes",
        "coming",
        "cool",
        "crimson",
        "cyan",
        "deep",
        "dim",
        "does",
        "done",
        "draw",
        "draws",
        "drop",
        "drops",
        "emerald",
        "enter",
        "enters",
        "fades",
        "fall",
        "falling",
        "falls",
        "feel",
        "feels",
        "fill",
        "fills",
        "find",
        "first",
        "flicker",
        "flickering",
        "float",
        "floating",
        "flow",
        "flowing",
        "gently",
        "gold",
        "golden",
        "gray",
        "green",
        "grey",
        "hang",
        "hanging",
        "hangs",
        "hard",
        "have",
        "held",
        "hold",
        "holding",
        "holds",
        "indigo",
        "keeping",
        "keeps",
        "last",
        "late",
        "lead",
        "leads",
        "leaves",
        "leaving",
        "lift",
        "lifting",
        "lifts",
        "little",
        "live",
        "living",
        "lower",
        "make",
        "makes",
        "making",
        "match",
        "middle",
        "moonlight",
        "orange",
        "pale",
        "pass",
        "passes",
        "passing",
        "pink",
        "purple",
        "quiet",
        "raise",
        "raised",
        "raises",
        "reach",
        "reaches",
        "reaching",
        "rest",
        "resting",
        "rests",
        "rise",
        "rises",
        "rising",
        "ruby",
        "runs",
        "running",
        "rust",
        "rusted",
        "scarlet",
        "seen",
        "settle",
        "settles",
        "shine",
        "shines",
        "shining",
        "show",
        "showing",
        "shows",
        "silver",
        "slowly",
        "smooth",
        "softly",
        "stay",
        "stays",
        "strike",
        "striking",
        "strong",
        "takes",
        "taking",
        "tall",
        "teal",
        "thin",
        "tightly",
        "touch",
        "touches",
        "touching",
        "upper",
        "violet",
        "wait",
        "waiting",
        "waits",
        "warmth",
        "watch",
        "watches",
        "watching",
        "white",
        "worn",
        "yellow",
        # Short (3-letter) non-prop words (beyond dim/fog/jaw/low/pan above)
        "act",
        "age",
        "ago",
        "aid",
        "aim",
        "air",
        "all",
        "and",
        "any",
        "are",
        "arm",
        "art",
        "ash",
        "ask",
        "bad",
        "big",
        "bit",
        "but",
        "can",
        "cue",
        "cut",
        "day",
        "did",
        "dry",
        "due",
        "ear",
        "end",
        "era",
        "eye",
        "far",
        "few",
        "fit",
        "for",
        "fun",
        "get",
        "got",
        "had",
        "has",
        "her",
        "him",
        "his",
        "hot",
        "how",
        "hue",
        "ice",
        "icy",
        "its",
        "leg",
        "let",
        "lip",
        "lit",
        "lot",
        "man",
        "may",
        "men",
        "met",
        "mid",
        "mix",
        "new",
        "nod",
        "nor",
        "not",
        "now",
        "odd",
        "off",
        "old",
        "one",
        "our",
        "out",
        "own",
        "pay",
        "per",
        "pov",
        "put",
        "raw",
        "ray",
        "red",
        "rib",
        "rub",
        "run",
        "sad",
        "sat",
        "saw",
        "say",
        "sea",
        "see",
        "set",
        "she",
        "shy",
        "sit",
        "six",
        "sky",
        "sun",
        "tan",
        "ten",
        "the",
        "too",
        "top",
        "try",
        "two",
        "use",
        "via",
        "was",
        "way",
        "wet",
        "who",
        "why",
        "win",
        "won",
        "yes",
        "yet",
        "you",
    }
)


@dataclass(frozen=True)
class CoverageFinding:
    noun: str
    count: int
    sections: list[str]
    shot_ids: list[str]
    suggested_command: str
    entity_kind: Literal["prop", "set", "character"] = "prop"
    declared: bool = False
    locked: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _singularize(word: str) -> str:
    if len(word) <= 3 or word in _SINGULAR_KEEP:
        return word
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith(("ches", "shes", "xes", "zes", "sses")) and len(word) > 4:
        return word[:-2]
    if word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def _normalize_phrase(text: str) -> str:
    folded = ascii_fold(text).lower()
    words = _WORD_RE.findall(folded)
    if not words:
        return ""
    norm = [_singularize(w) for w in words]
    return " ".join(norm)


# Words that end a noun phrase: once a chunk is followed by one of these (or by punctuation or
# the end of the text), its last word is a noun rather than the verb of a clause.
_PHRASE_CLOSERS = frozenset(
    {
        "above",
        "across",
        "against",
        "along",
        "amid",
        "among",
        "and",
        "around",
        "as",
        "at",
        "behind",
        "below",
        "beneath",
        "beside",
        "between",
        "beyond",
        "but",
        "by",
        "for",
        "from",
        "in",
        "inside",
        "into",
        "near",
        "nor",
        "of",
        "on",
        "onto",
        "or",
        "over",
        "past",
        "through",
        "to",
        "toward",
        "towards",
        "under",
        "upon",
        "while",
        "with",
        "within",
        "without",
    }
)
# Stopword nouns that usually start a clause in shot prompts ("the camera holds steady").
_CLAUSE_SUBJECTS = frozenset({"camera", "focus", "frame", "lens", "light", "lights", "shot"})
_NEXT_WORD_RE = re.compile(r"[ \t]*([a-z]+)?")
_DETERMINER_WORD_RE = re.compile(_DETERMINER_PREFIX)


def _extract_noun_candidates(text: str) -> tuple[set[str], dict[str, str], set[str]]:
    """Return ``(noun_phrases, phrase_to_head, loose_words)`` normalized from ``text``.

    ``noun_phrases`` come from determiner/preposition-led chunks that are closed by punctuation,
    the end of the text, or a preposition/conjunction ("a brass lantern on wet stone"), which is
    reliable evidence the last word is a noun. ``loose_words`` holds everything weaker: heads of
    chunks running into a verb ("the lantern swings" -> "swing") and every remaining content word.
    They only count as mentions of a noun some closed chunk in the project already established.
    """
    folded = ascii_fold(text).lower()
    candidates: set[str] = set()
    phrase_to_head: dict[str, str] = {}
    loose: set[str] = set()

    for match in _CHUNK_RE.finditer(folded):
        raw_words = match.group(1).split()
        closed = False
        if len(raw_words) == 2 and raw_words[1] in _PHRASE_CLOSERS:
            raw_words.pop()
            closed = True
        else:
            nxt = _NEXT_WORD_RE.match(folded, match.end())
            next_word = nxt.group(1) if nxt else None
            # A chunk's last word reads as a verb when an object or adverb follows it
            # ("pushes the door", "swings gently"); otherwise it is taken as the noun.
            closed = (
                next_word is None
                or next_word in _PHRASE_CLOSERS
                or not (_DETERMINER_WORD_RE.fullmatch(next_word) or next_word.endswith("ly"))
            )
        stripped = False
        while raw_words and (
            raw_words[0] in _COVERAGE_STOPWORDS
            or raw_words[0].endswith("ly")
            or len(raw_words[0]) < 3
        ):
            stripped = stripped or raw_words[0] in _CLAUSE_SUBJECTS
            raw_words.pop(0)
        if not raw_words:
            continue
        head = _singularize(raw_words[-1])
        if len(head) < 3 or head in _COVERAGE_STOPWORDS or head.endswith(("ly", "ing", "ed")):
            continue
        # "the camera remains rooted": once a subject noun is dropped as a stopword, the word
        # left over is the clause's verb, not a noun.
        bucket = candidates if closed and not stripped else loose
        if len(raw_words) == 2:
            mod = _singularize(raw_words[0])
            if len(mod) >= 3 and mod not in _COVERAGE_STOPWORDS and not mod.endswith("ly"):
                phrase = f"{mod} {head}"
                bucket.add(phrase)
                phrase_to_head[phrase] = head
        if head not in _MATERIAL_MODIFIERS:
            bucket.add(head)
            phrase_to_head[head] = head

    for token in _WORD_RE.findall(folded):
        if len(token) < 3 or token.endswith(("ly", "ing")):
            continue
        lemma = _singularize(token)
        if (
            len(lemma) < 3
            or lemma in _COVERAGE_STOPWORDS
            or lemma in _MATERIAL_MODIFIERS
            or lemma.endswith("ed")
        ):
            continue
        loose.add(lemma)

    return candidates, phrase_to_head, loose


def _entity_tokens(text: str) -> set[str]:
    """Extract normalized tokens and phrases from an entity name/id/description."""
    norm = _normalize_phrase(text.replace("_", " "))
    out: set[str] = set()
    if norm:
        out.add(norm)
    cands, _, bare = _extract_noun_candidates(text.replace("_", " "))
    out.update(cands)
    out.update(bare)
    return out


def _safe_prop_slug(noun: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", ascii_fold(noun).lower()).strip("_")
    return slug or "prop"


@dataclass
class _DeclaredEntity:
    kind: Literal["prop", "set", "character"]
    key: str
    display_name: str
    aliases: set[str]
    tokens: set[str]
    locked: bool
    suggested_command: str


@dataclass
class _NormalizedShot:
    shot_id: str
    section: str
    setting: str
    props: list[str]
    characters: list[str]
    text: str


def _as_dict_list(value: Any) -> list[Mapping[str, Any]]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [item for item in value if isinstance(item, Mapping)]
    return []


def _extract_inputs(
    manifest: Manifest | Mapping[str, Any],
) -> tuple[list[_NormalizedShot], list[_DeclaredEntity]]:
    entities: list[_DeclaredEntity] = []
    shots_out: list[_NormalizedShot] = []

    if isinstance(manifest, Manifest):
        chars = (
            dict(manifest.characters)
            if manifest.characters
            else (
                {"lead": manifest.character}
                if manifest.character.name or manifest.character.appearance
                else {}
            )
        )
        single_unnamed = not manifest.characters
        for cid, char in chars.items():
            locked = char.default_reference is not None or any(
                bool(ref.sheet or ref.panels) for ref in char.references.values()
            )
            aliases = {
                _normalize_phrase(v)
                for v in (cid.replace("_", " "), char.name)
                if v and _normalize_phrase(v)
            }
            desc_parts = [
                char.name,
                char.appearance,
                char.invariants,
                *(w.description for w in char.wardrobe),
            ]
            tokens = set(aliases)
            for part in desc_parts:
                if part:
                    tokens.update(_entity_tokens(part))
            cmd = (
                "mvd reference generate"
                if single_unnamed
                else f"mvd reference generate --character {shlex.quote(cid)}"
            )
            entities.append(
                _DeclaredEntity(
                    kind="character",
                    key=cid,
                    display_name=char.name or cid,
                    aliases=aliases,
                    tokens=tokens,
                    locked=locked,
                    suggested_command=cmd,
                )
            )
        for scene in manifest.visual_bible.settings:
            locked = bool(scene.plate or scene.panels)
            aliases = {_normalize_phrase(scene.name)} - {""}
            tokens = set(aliases) | _entity_tokens(f"{scene.name} {scene.description}")
            entities.append(
                _DeclaredEntity(
                    kind="set",
                    key=scene.name,
                    display_name=scene.name,
                    aliases=aliases,
                    tokens=tokens,
                    locked=locked,
                    suggested_command=f"mvd plates generate --set {shlex.quote(scene.name)}",
                )
            )
        for prop in manifest.visual_bible.props:
            locked = bool(prop.plate)
            aliases = {
                _normalize_phrase(v)
                for v in (prop.id.replace("_", " "), prop.name)
                if v and _normalize_phrase(v)
            }
            tokens = set(aliases) | _entity_tokens(f"{prop.name} {prop.description}")
            entities.append(
                _DeclaredEntity(
                    kind="prop",
                    key=prop.id,
                    display_name=prop.name,
                    aliases=aliases,
                    tokens=tokens,
                    locked=locked,
                    suggested_command=f"mvd props generate --prop {shlex.quote(prop.id)}",
                )
            )
        for shot in manifest.shots:
            text_parts = [
                shot.still.prompt_override or shot.still_prompt,
                shot.video.prompt_override or shot.motion_prompt,
                shot.previz_mapping,
                shot.sketch_guidance,
            ]
            char_ids = list(shot.characters)
            if shot.character and shot.character not in char_ids:
                char_ids.insert(0, shot.character)
            shots_out.append(
                _NormalizedShot(
                    shot_id=shot.id,
                    section=(shot.section or "").strip(),
                    setting=shot.setting.strip(),
                    props=list(shot.props),
                    characters=char_ids,
                    text="\n".join(p for p in text_parts if p),
                )
            )
        return shots_out, entities

    # Mapping[str, Any] adapter (compatible with shot_lint.py dicts)
    raw_chars = manifest.get("characters")
    char_items: list[tuple[str, Mapping[str, Any], bool]] = []
    if isinstance(raw_chars, Mapping) and raw_chars:
        for cid, cdata in raw_chars.items():
            if isinstance(cdata, Mapping):
                char_items.append((str(cid), cdata, False))
    elif isinstance(raw_chars, Sequence) and not isinstance(raw_chars, (str, bytes)):
        for cdata in raw_chars:
            if isinstance(cdata, Mapping):
                cid = str(cdata.get("id") or cdata.get("name") or "character")
                char_items.append((cid, cdata, False))
    elif isinstance(manifest.get("character"), Mapping):
        cdata = manifest["character"]
        if cdata.get("name") or cdata.get("appearance"):
            char_items.append(("lead", cdata, True))

    for cid, cdata, single_unnamed in char_items:
        refs = cdata.get("references")
        locked = False
        if isinstance(refs, Mapping):
            locked = any(
                isinstance(r, Mapping) and bool(r.get("sheet") or r.get("panels"))
                for r in refs.values()
            )
        cname = str(cdata.get("name") or "")
        aliases = {
            _normalize_phrase(v)
            for v in (cid.replace("_", " "), cname)
            if v and _normalize_phrase(v)
        }
        wardrobe_parts = [
            str(w.get("description", "")) for w in _as_dict_list(cdata.get("wardrobe"))
        ]
        desc_parts = [
            cname,
            str(cdata.get("appearance") or ""),
            str(cdata.get("invariants") or ""),
            *wardrobe_parts,
        ]
        tokens = set(aliases)
        for part in desc_parts:
            if part:
                tokens.update(_entity_tokens(part))
        cmd = (
            "mvd reference generate"
            if single_unnamed
            else f"mvd reference generate --character {shlex.quote(cid)}"
        )
        entities.append(
            _DeclaredEntity(
                kind="character",
                key=cid,
                display_name=cname or cid,
                aliases=aliases,
                tokens=tokens,
                locked=locked,
                suggested_command=cmd,
            )
        )

    vb = manifest.get("visual_bible")
    vb_map = vb if isinstance(vb, Mapping) else {}
    for sdata in _as_dict_list(vb_map.get("settings")):
        sname = str(sdata.get("name") or "").strip()
        if not sname:
            continue
        locked = bool(sdata.get("plate") or sdata.get("panels"))
        aliases = {_normalize_phrase(sname)} - {""}
        tokens = set(aliases) | _entity_tokens(f"{sname} {sdata.get('description') or ''}")
        entities.append(
            _DeclaredEntity(
                kind="set",
                key=sname,
                display_name=sname,
                aliases=aliases,
                tokens=tokens,
                locked=locked,
                suggested_command=f"mvd plates generate --set {shlex.quote(sname)}",
            )
        )
    for pdata in _as_dict_list(vb_map.get("props")):
        pid = str(pdata.get("id") or pdata.get("name") or "").strip()
        pname = str(pdata.get("name") or pid).strip()
        if not pid and not pname:
            continue
        locked = bool(pdata.get("plate"))
        aliases = {
            _normalize_phrase(v)
            for v in (pid.replace("_", " "), pname)
            if v and _normalize_phrase(v)
        }
        tokens = set(aliases) | _entity_tokens(f"{pname} {pdata.get('description') or ''}")
        entities.append(
            _DeclaredEntity(
                kind="prop",
                key=pid or pname,
                display_name=pname or pid,
                aliases=aliases,
                tokens=tokens,
                locked=locked,
                suggested_command=f"mvd props generate --prop {shlex.quote(pid or pname)}",
            )
        )

    for idx, sdata in enumerate(_as_dict_list(manifest.get("shots")), start=1):
        sid = str(sdata.get("id") or f"s{idx:03d}")
        still_obj = sdata.get("still")
        video_obj = sdata.get("video")
        still_override = (
            str(still_obj.get("prompt_override") or "") if isinstance(still_obj, Mapping) else ""
        )
        video_override = (
            str(video_obj.get("prompt_override") or "") if isinstance(video_obj, Mapping) else ""
        )
        text_parts = [
            still_override or str(sdata.get("still_prompt") or ""),
            video_override or str(sdata.get("motion_prompt") or ""),
            str(sdata.get("description") or ""),
            str(sdata.get("continuity_note") or ""),
            str(sdata.get("previz_mapping") or ""),
            str(sdata.get("sketch_guidance") or ""),
            str(sdata.get("prompt") or ""),
        ]
        raw_props = sdata.get("props")
        prop_list = (
            [str(p) for p in raw_props if p]
            if isinstance(raw_props, Sequence) and not isinstance(raw_props, (str, bytes))
            else []
        )
        raw_chars = sdata.get("characters")
        char_list = (
            [str(c) for c in raw_chars if c]
            if isinstance(raw_chars, Sequence) and not isinstance(raw_chars, (str, bytes))
            else []
        )
        primary_char = sdata.get("character")
        if isinstance(primary_char, str) and primary_char and primary_char not in char_list:
            char_list.insert(0, primary_char)
        shots_out.append(
            _NormalizedShot(
                shot_id=sid,
                section=str(sdata.get("section") or "").strip(),
                setting=str(sdata.get("setting") or "").strip(),
                props=prop_list,
                characters=char_list,
                text="\n".join(p for p in text_parts if p),
            )
        )
    return shots_out, entities


def lint_reference_coverage(
    manifest: Manifest | Mapping[str, Any],
    *,
    min_shots: int = 3,
    min_sections: int = 2,
) -> list[CoverageFinding]:
    """Flag recurring objects, places, or performers lacking a locked visual reference.

    Zero-LLM deterministic check over shot prompts and visual-bible metadata. Any candidate
    noun phrase (or declared-but-unlocked entity) appearing in at least ``min_shots`` shots
    or across at least ``min_sections`` distinct song sections without a locked reference is
    reported with its shot IDs and a suggested ``mvd props`` / ``mvd plates`` command.
    """
    if isinstance(min_shots, bool) or min_shots < 1:
        raise ConfigError(f"min_shots must be a positive integer, got {min_shots!r}.")
    if isinstance(min_sections, bool) or min_sections < 1:
        raise ConfigError(f"min_sections must be a positive integer, got {min_sections!r}.")

    shots, entities = _extract_inputs(manifest)
    if not shots:
        return []

    locked_tokens: set[str] = set()
    unlocked_entities: list[_DeclaredEntity] = []
    for entity in entities:
        if entity.locked:
            locked_tokens.update(entity.tokens)
        else:
            unlocked_entities.append(entity)

    # Track occurrences per unlocked declared entity and per undeclared noun phrase.
    entity_shots: dict[tuple[str, str], list[str]] = {}
    entity_sections: dict[tuple[str, str], list[str]] = {}
    noun_shots: dict[str, list[str]] = {}
    noun_sections: dict[str, list[str]] = {}
    phrase_heads: dict[str, str] = {}

    def _record(
        bucket_shots: dict[Any, list[str]],
        bucket_sections: dict[Any, list[str]],
        key: Any,
        shot_id: str,
        section: str,
    ) -> None:
        slist = bucket_shots.setdefault(key, [])
        if shot_id not in slist:
            slist.append(shot_id)
        if section:
            seclist = bucket_sections.setdefault(key, [])
            if section not in seclist:
                seclist.append(section)

    extracted = [_extract_noun_candidates(shot.text) for shot in shots]
    # A loose word counts as a noun mention only once some shot uses it in a closed noun chunk
    # ("beside a lantern."). Otherwise every verb, adverb and camera direction ("forward",
    # "maintain", "level") recurs across sections and floods the report.
    noun_vocabulary: set[str] = {cand for cands, _, _ in extracted for cand in cands}
    for shot, (noun_phrases, head_map, loose_words) in zip(shots, extracted, strict=True):
        cands = noun_phrases | (loose_words & noun_vocabulary)
        head_map = {**{word: word for word in loose_words}, **head_map}
        phrase_heads.update(head_map)
        norm_text = _normalize_phrase(shot.text)
        padded_norm_text = f" {norm_text} " if norm_text else ""
        norm_prop_refs = {_normalize_phrase(p.replace("_", " ")) for p in shot.props if p}
        norm_setting = _normalize_phrase(shot.setting)
        norm_char_refs = {_normalize_phrase(c.replace("_", " ")) for c in shot.characters if c}

        matched_unlocked_tokens: set[str] = set()
        for entity in unlocked_entities:
            ekey = (entity.kind, entity.key)
            explicitly_referenced = (
                (entity.kind == "prop" and bool(norm_prop_refs & entity.aliases))
                or (entity.kind == "set" and norm_setting in entity.aliases)
                or (entity.kind == "character" and bool(norm_char_refs & entity.aliases))
            )
            mentioned_in_text = bool(cands & entity.aliases) or any(
                f" {alias} " in padded_norm_text for alias in entity.aliases if len(alias) >= 3
            )
            if explicitly_referenced or mentioned_in_text:
                _record(entity_shots, entity_sections, ekey, shot.shot_id, shot.section)
                matched_unlocked_tokens.update(entity.aliases)
                matched_unlocked_tokens.update(entity.tokens & cands)

        for cand in cands:
            if cand in locked_tokens:
                continue
            head = head_map.get(cand, cand)
            if head in locked_tokens:
                continue
            if cand in matched_unlocked_tokens:
                continue
            _record(noun_shots, noun_sections, cand, shot.shot_id, shot.section)

    findings: list[CoverageFinding] = []
    for entity in unlocked_entities:
        ekey = (entity.kind, entity.key)
        sids = entity_shots.get(ekey, [])
        secs = entity_sections.get(ekey, [])
        if len(sids) >= min_shots or len(secs) >= min_sections:
            findings.append(
                CoverageFinding(
                    noun=entity.display_name,
                    count=len(sids),
                    sections=secs,
                    shot_ids=sids,
                    suggested_command=entity.suggested_command,
                    entity_kind=entity.kind,
                    declared=True,
                    locked=False,
                )
            )

    qualifying_nouns = {
        cand: sids
        for cand, sids in noun_shots.items()
        if len(sids) >= min_shots or len(noun_sections.get(cand, [])) >= min_sections
    }
    # Suppress single-word head nouns when a 2-word modifier+noun phrase already covers the same
    # or a superset of shots (e.g. "brass lantern" suppresses redundant "lantern").
    suppressed_heads: set[str] = set()
    for cand, sids in qualifying_nouns.items():
        if " " not in cand:
            continue
        head = phrase_heads.get(cand, cand.rsplit(" ", 1)[-1])
        head_sids = qualifying_nouns.get(head)
        if head_sids is not None and set(head_sids) <= set(sids):
            suppressed_heads.add(head)

    for cand, sids in qualifying_nouns.items():
        if cand in suppressed_heads:
            continue
        secs = noun_sections.get(cand, [])
        head = phrase_heads.get(cand, cand.rsplit(" ", 1)[-1])
        if head in _SET_LOCATION_NOUNS:
            entity_kind: Literal["prop", "set"] = "set"
            cmd = f"mvd plates generate --set {shlex.quote(cand)}"
        else:
            entity_kind = "prop"
            cmd = f"mvd props generate --prop {shlex.quote(_safe_prop_slug(cand))}"
        findings.append(
            CoverageFinding(
                noun=cand,
                count=len(sids),
                sections=secs,
                shot_ids=sids,
                suggested_command=cmd,
                entity_kind=entity_kind,
                declared=False,
                locked=False,
            )
        )

    findings.sort(key=lambda f: (-f.count, -len(f.sections), f.noun))
    return findings
