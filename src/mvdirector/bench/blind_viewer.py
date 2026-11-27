# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Static self-contained HTML viewer for blind human evaluation of Track 1 planning runs."""

from __future__ import annotations

import base64
import json
import random
import secrets
from collections import defaultdict
from collections.abc import Sequence
from importlib import resources
from pathlib import Path
from typing import Any

from mvdirector.bench.planning import PlanningArmResult
from mvdirector.errors import ConfigError
from mvdirector.models import Manifest, Shot, ShotType
from mvdirector.planning import prompts
from mvdirector.store import atomic_write

_MAX_MATCHED_SHOTS = 10
_MIN_MATCHED_SHOTS = 8


def _safe_json_for_script(obj: Any) -> str:
    """Serialize ``obj`` safely for embedding inside an HTML ``<script>`` tag."""
    return (
        json.dumps(obj, ensure_ascii=False)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def _sec_lower(shot: Shot) -> str:
    return (shot.section or "").lower()


def select_matched_shots(
    manifests: Sequence[Manifest],
    *,
    max_shots: int = _MAX_MATCHED_SHOTS,
) -> list[tuple[str, str]]:
    """Select 8-10 representative matched ``(shot_id, role_label)`` pairs shared across arms.

    Covers intro, verse 1, chorus drop, cutaway/b-roll, verse 2, wardrobe shift (detected across
    any arm), bridge, peak chorus, and finale, then fills evenly if fewer than 8 roles matched.
    """
    if not manifests or not manifests[0].shots:
        return []
    base_shots = manifests[0].shots
    common_ids = {s.id for s in base_shots}
    for m in manifests[1:]:
        common_ids &= {s.id for s in m.shots}
    shots = [s for s in base_shots if s.id in common_ids]
    if not shots:
        return []
    if len(shots) <= max_shots:
        return [(s.id, s.section or s.type.value) for s in shots]

    chosen: list[tuple[str, str]] = []
    chosen_ids: set[str] = set()

    def _add(shot: Shot | None, label: str) -> None:
        if shot is None or shot.id in chosen_ids or len(chosen) >= max_shots:
            return
        chosen.append((shot.id, label))
        chosen_ids.add(shot.id)

    def _first(pred: Any) -> Shot | None:
        return next((s for s in shots if s.id not in chosen_ids and pred(s)), None)

    _add(shots[0], "Intro")
    _add(_first(lambda s: "verse" in _sec_lower(s)), "Verse 1")
    _add(
        _first(lambda s: "pre-chorus" in _sec_lower(s) or "pre chorus" in _sec_lower(s)),
        "Pre-Chorus",
    )
    _add(
        _first(lambda s: "chorus" in _sec_lower(s) or "drop" in _sec_lower(s)),
        "Chorus Drop",
    )
    _add(_first(lambda s: s.type is ShotType.BROLL), "Cutaway (B-Roll)")

    # First shot where ANY arm transitions to a subsequent wardrobe state.
    wardrobe_shift_id: str | None = None
    for m in manifests:
        initial_w: str | None = None
        for s in m.shots:
            if s.id not in common_ids or not s.wardrobe:
                continue
            if initial_w is None:
                initial_w = s.wardrobe
            elif s.wardrobe != initial_w:
                wardrobe_shift_id = s.id
                break
        if wardrobe_shift_id is not None:
            break
    if wardrobe_shift_id is not None:
        _add(_first(lambda s: s.id == wardrobe_shift_id), "Wardrobe Shift")

    mid_idx = len(shots) // 3
    _add(
        next(
            (s for s in shots[mid_idx:] if s.id not in chosen_ids and "verse" in _sec_lower(s)),
            None,
        ),
        "Verse 2",
    )
    _add(_first(lambda s: "bridge" in _sec_lower(s)), "Bridge")
    late_half = [s for s in shots[len(shots) // 2 : -1] if s.id not in chosen_ids]
    if late_half:
        _add(max(late_half, key=lambda s: s.intensity), "Peak / Final Chorus")
    _add(shots[-1], "Finale")

    target = min(max_shots, max(_MIN_MATCHED_SHOTS, len(chosen)))
    if len(chosen) < target:
        step = max(1, len(shots) // (target + 1))
        for idx in range(step, len(shots) - 1, step):
            if len(chosen) >= target:
                break
            _add(shots[idx], shots[idx].section or f"Beat {idx + 1}")

    order = {s.id: i for i, s in enumerate(shots)}
    chosen.sort(key=lambda pair: order[pair[0]])
    return chosen


def _serialize_bible(manifest: Manifest) -> dict[str, Any]:
    bible = manifest.visual_bible
    chars = (
        [
            {
                "id": cid,
                "name": c.name or cid,
                "appearance": c.appearance,
                "invariants": c.invariants,
                "wardrobe": [{"id": w.id, "description": w.description} for w in c.wardrobe],
            }
            for cid, c in manifest.characters.items()
        ]
        if manifest.characters
        else (
            [
                {
                    "id": "lead",
                    "name": manifest.character.name or "Lead",
                    "appearance": manifest.character.appearance,
                    "invariants": manifest.character.invariants,
                    "wardrobe": [
                        {"id": w.id, "description": w.description}
                        for w in manifest.character.wardrobe
                    ],
                }
            ]
            if manifest.character.appearance or manifest.character.wardrobe
            else []
        )
    )
    interp = bible.interpretation
    return {
        "interpretation": (
            {
                "meaning": interp.meaning,
                "tone": interp.tone,
                "music_feel": interp.music_feel,
                "visual_motifs": list(interp.visual_motifs),
                "section_beats": [b.model_dump() for b in interp.section_beats],
            }
            if interp is not None
            else None
        ),
        "concept": bible.concept,
        "style": bible.style,
        "palette": bible.palette,
        "lighting": bible.lighting,
        "negative_prompt": bible.negative_prompt,
        "characters": chars,
        "settings": [
            {
                "name": s.name,
                "sections": list(s.sections),
                "description": s.description,
                "movement": s.movement,
                "parent_set": s.parent_set,
            }
            for s in bible.settings
        ],
        "props": [
            {
                "id": p.id,
                "name": p.name,
                "sections": list(p.sections),
                "description": p.description,
            }
            for p in bible.props
        ],
        "rules": list(bible.rules),
        "exclusions": list(bible.exclusions),
    }


def _serialize_shot(manifest: Manifest, shot: Shot) -> dict[str, Any]:
    return {
        "id": shot.id,
        "type": shot.type.value,
        "framing": shot.framing.value,
        "setting": shot.setting,
        "character": shot.character,
        "characters": list(shot.characters),
        "wardrobe": shot.wardrobe,
        "props": list(shot.props),
        "still_prompt": shot.still_prompt,
        "composed_still_prompt": prompts.compose_still_prompt(manifest, shot),
        "motion_prompt": shot.motion_prompt,
    }


def build_blind_viewer_payload(
    results: Sequence[PlanningArmResult],
    *,
    seed: int | None = None,
    max_shots: int = _MAX_MATCHED_SHOTS,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build ``(blind_data, reveal_key)`` structures for the static HTML viewer."""
    rng = random.Random(seed if seed is not None else secrets.randbits(64))
    by_track: dict[str, list[PlanningArmResult]] = defaultdict(list)
    for res in results:
        if res.status == "ok" and res.manifest_path and Path(res.manifest_path).is_file():
            by_track[res.track].append(res)
    if not by_track:
        raise ConfigError(
            "No completed Track 1 manifests found to build a blind comparison viewer."
        )

    tracks_blind: list[dict[str, Any]] = []
    reveal_map: dict[str, dict[str, dict[str, Any]]] = {}

    for track, track_results in sorted(by_track.items()):
        loaded: list[tuple[PlanningArmResult, Manifest]] = [
            (r, Manifest.model_validate_json(Path(r.manifest_path).read_text(encoding="utf-8")))
            for r in track_results
            if r.manifest_path is not None
        ]
        if not loaded:
            continue
        matched = select_matched_shots([m for _, m in loaded], max_shots=max_shots)
        base_by_id = {s.id: s for s in loaded[0][1].shots}
        shot_meta = [
            {
                "id": sid,
                "role": role,
                "start": base_by_id[sid].start,
                "end": base_by_id[sid].end,
                "section": base_by_id[sid].section,
                "lyrics": base_by_id[sid].lyrics,
                "intensity": base_by_id[sid].intensity,
                "type": base_by_id[sid].type.value,
            }
            for sid, role in matched
            if sid in base_by_id
        ]

        shuffled = list(loaded)
        rng.shuffle(shuffled)
        directors: list[dict[str, Any]] = []
        track_reveal: dict[str, dict[str, Any]] = {}
        for idx, (res, manifest) in enumerate(shuffled, start=1):
            alias = f"Director {idx}"
            shots_by_id = {s.id: s for s in manifest.shots}
            directors.append(
                {
                    "alias": alias,
                    "bible": _serialize_bible(manifest),
                    "shots": {
                        meta["id"]: _serialize_shot(manifest, shots_by_id[meta["id"]])
                        for meta in shot_meta
                        if meta["id"] in shots_by_id
                    },
                }
            )
            track_reveal[alias] = {
                "arm_id": res.arm_id,
                "arm_label": res.arm_label,
                "provider": res.provider,
                "model": res.model,
                "attach_audio": res.attach_audio,
                "wall_seconds": res.wall_seconds,
                "telemetry": res.telemetry.model_dump(mode="json"),
                "quality": res.quality.model_dump(mode="json"),
            }

        tracks_blind.append(
            {
                "track": track,
                "shot_meta": shot_meta,
                "directors": directors,
            }
        )
        reveal_map[track] = track_reveal

    return {"tracks": tracks_blind}, {"tracks": reveal_map}


def generate_blind_viewer(
    results: Sequence[PlanningArmResult],
    output_html: Path,
    *,
    seed: int | None = None,
    max_shots: int = _MAX_MATCHED_SHOTS,
) -> tuple[Path, Path]:
    """Write a self-contained blind HTML viewer and a companion ``blind_key.json``."""
    blind_data, reveal_key = build_blind_viewer_payload(results, seed=seed, max_shots=max_shots)
    encoded_key = base64.b64encode(
        json.dumps(reveal_key, ensure_ascii=False).encode("utf-8")
    ).decode("ascii")

    template = resources.files("mvdirector.bench").joinpath("blind_viewer.html").read_text("utf-8")
    html_doc = template.replace("__BLIND_DATA__", _safe_json_for_script(blind_data)).replace(
        "__ENCODED_KEY__", _safe_json_for_script(encoded_key)
    )
    output_html.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(output_html, html_doc)
    key_path = output_html.with_name(output_html.stem + "_key.json")
    atomic_write(key_path, json.dumps(reveal_key, indent=2) + "\n")
    return output_html, key_path
