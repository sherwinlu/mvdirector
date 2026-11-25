# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Check, before any still is generated, that the assets it depends on are locked.

A series stays consistent when every character sheet and location plate is approved before the
first generation. A missing asset otherwise shows up as a run of failed or inconsistent stills.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mvdirector.comfy.workflow import WorkflowTemplate
from mvdirector.errors import ConfigError
from mvdirector.models import Manifest
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
        still is None
        or still.accepts("reference_image")
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
            if not scene.plate:
                result.problems.append(
                    f"No plate for set '{scene.name}'. Run `mvd plates generate --set "
                    f'"{scene.name}"`, then `mvd plates lock <image> --set "{scene.name}"`.'
                )
            elif not project.path(scene.plate).is_file():
                result.problems.append(
                    f"Plate file for set '{scene.name}' is missing: {scene.plate}"
                )
    elif plate_templates:
        for name in sorted({s.setting for s in shots if s.setting}):
            scene = bible.find_set(name)
            if scene is not None and scene.plate and not project.path(scene.plate).is_file():
                result.problems.append(
                    f"Plate file for set '{scene.name}' is missing: {scene.plate}"
                )
    else:
        plated = [s.name for s in bible.settings if s.plate]
        if plated:
            result.warnings.append(
                f"Plates are locked for {', '.join(plated)}, but the 'still' workflow has no "
                "'set_image' binding, so they will not be used."
            )
    t2i = templates.get("t2i")
    prop_queries = [
        q
        for s in shots
        if s.props
        for q in s.props
        if (tpl := (still if s.type.shows_singer else t2i)) is not None
        and tpl.accepts("prop_image")
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
        prop_bound = any(t is not None and t.accepts("prop_image") for t in (still, t2i))
        if locked_props and not prop_bound:
            result.warnings.append(
                f"Prop sheets are locked for {', '.join(locked_props)}, but the workflow has "
                "no 'prop_image' binding, so they will not be used."
            )
    return result
