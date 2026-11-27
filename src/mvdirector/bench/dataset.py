# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Labeled still dataset loader and stratified scaffolder for Track 2 Vision QC benchmarks."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from mvdirector.bench.config import open_bench_project
from mvdirector.calibrate import _parse_label
from mvdirector.errors import ConfigError
from mvdirector.models import AssetKind, Framing, Manifest, ReviewDecision, Shot
from mvdirector.project import Project
from mvdirector.store import atomic_write, path_is_file

FailureTag = Literal["spec", "artifact", "character"]
_TAG_ALIASES: dict[str, FailureTag] = {
    "spec": "spec",
    "vision_spec": "spec",
    "prompt": "spec",
    "framing": "spec",
    "artifact": "artifact",
    "artifacts": "artifact",
    "vision_artifacts": "artifact",
    "anatomy": "artifact",
    "text": "artifact",
    "character": "character",
    "char": "character",
    "identity": "character",
    "vision_character": "character",
    "wardrobe": "character",
}
_DARK_SCENE_RE = re.compile(
    r"\b(?:dark|night|shadow|dim|moody|silhouette|candle|noir|midnight|backlit)\b",
    re.IGNORECASE,
)
_ANGLED_FACE_RE = re.compile(
    r"\b(?:three[- ]quarter|3/4|profile|angled|turned|glancing|over[- ]the[- ]shoulder)\b",
    re.IGNORECASE,
)


class QCLabelItem(BaseModel):
    """One labeled still in the Track 2 Vision QC benchmark dataset."""

    id: str = Field(description="Unique stable key, e.g. '<project>:<shot>:a<attempt>'.")
    project: str = Field(description="Project directory path.")
    shot_id: str
    attempt: int | None = None
    good: bool
    tags: list[FailureTag] = Field(default_factory=list)
    notes: str = ""


def parse_failure_tags(raw: object) -> list[FailureTag]:
    """Parse semicolon-, comma-, pipe-, or space-delimited failure tags into canonical names."""
    if raw is None:
        return []
    tokens: list[str]
    if isinstance(raw, list):
        tokens = [str(item) for item in raw if item is not None]
    else:
        tokens = re.split(r"[,;|\s]+", str(raw).strip())
    parsed: list[FailureTag] = []
    for token in tokens:
        cleaned = token.strip().lower()
        if not cleaned:
            continue
        if cleaned not in _TAG_ALIASES:
            raise ValueError(
                f"Unknown failure tag {token!r}; expected one of: spec, artifact, character."
            )
        canonical = _TAG_ALIASES[cleaned]
        if canonical not in parsed:
            parsed.append(canonical)
    return parsed


def _project_id_prefixes(resolved_projects: Sequence[Path]) -> dict[Path, str]:
    """Map resolved project paths to stable ID prefixes, disambiguating shared basenames."""
    unique = list(dict.fromkeys(p.resolve() for p in resolved_projects))
    counts = Counter(p.name for p in unique)
    prefixes: dict[Path, str] = {}
    for p in unique:
        if counts[p.name] > 1:
            digest = hashlib.sha256(str(p).encode("utf-8")).hexdigest()[:6]
            prefixes[p] = f"{p.name}-{digest}"
        else:
            prefixes[p] = p.name
    return prefixes


def load_qc_dataset(
    path: Path,
    *,
    default_project: Path | None = None,
) -> list[QCLabelItem]:
    """Load labeled stills from CSV or JSON for Track 2 Vision QC replay."""
    if not path_is_file(path):
        raise ConfigError(f"Vision QC labels file not found: {path}")
    try:
        text = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError) as exc:
        raise ConfigError(f"Could not read Vision QC labels file {path}: {exc}") from exc

    rows: list[dict[str, object]]
    if path.suffix.lower() == ".json":
        try:
            data = json.loads(text)
        except (json.JSONDecodeError, RecursionError) as exc:
            raise ConfigError(f"{path}: invalid JSON ({exc})") from exc
        if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
            raise ConfigError(f"{path} must hold a JSON list of label objects.")
        rows = [dict(item) for item in data]
    else:
        try:
            rows = [dict(row) for row in csv.DictReader(io.StringIO(text))]
        except csv.Error as exc:
            raise ConfigError(f"{path}: invalid CSV ({exc})") from exc

    base_dir = path.resolve().parent
    resolved_rows: list[tuple[int, dict[str, object], Path]] = []
    for idx, row in enumerate(rows, start=1):
        raw_proj = row.get("project")
        proj_str = "" if raw_proj is None else str(raw_proj).strip()
        if proj_str:
            cand = Path(proj_str)
            if not cand.is_absolute():
                rel_to_file = (base_dir / cand).resolve()
                rel_to_cwd = cand.resolve()
                resolved_proj = rel_to_file if rel_to_file.is_dir() else rel_to_cwd
            else:
                resolved_proj = cand.resolve()
        elif default_project is not None:
            resolved_proj = default_project.resolve()
        else:
            raise ConfigError(f"{path} row {idx}: missing 'project' column (or pass --project)")
        resolved_rows.append((idx, row, resolved_proj))

    prefixes = _project_id_prefixes([proj for _, _, proj in resolved_rows])
    items: list[QCLabelItem] = []
    seen_ids: set[str] = set()
    for idx, row, resolved_proj in resolved_rows:
        try:
            raw_shot = row.get("shot") or row.get("shot_id")
            if raw_shot is None or not str(raw_shot).strip():
                raise ValueError("shot must not be empty")
            shot_id = str(raw_shot).strip()

            raw_attempt = row.get("attempt")
            if isinstance(raw_attempt, bool):
                raise ValueError("attempt must be an integer")
            att_str = "" if raw_attempt is None else str(raw_attempt).strip()
            attempt = int(att_str) if att_str else None
            if attempt is not None and attempt <= 0:
                raise ValueError("attempt must be a positive integer")

            raw_label = row.get("label")
            if raw_label is None or not str(raw_label).strip():
                raise ValueError("label must not be empty")
            good = _parse_label(str(raw_label))
            tags = parse_failure_tags(row.get("tags") or row.get("failure_tags"))
            notes = str(row.get("notes") or "").strip()

            proj_prefix = prefixes[resolved_proj]
            item_id = f"{proj_prefix}:{shot_id}:a{attempt if attempt is not None else 'sel'}"
            if item_id in seen_ids:
                raise ValueError(f"duplicate dataset entry {item_id!r}")
            seen_ids.add(item_id)
            items.append(
                QCLabelItem(
                    id=item_id,
                    project=str(resolved_proj),
                    shot_id=shot_id,
                    attempt=attempt,
                    good=good,
                    tags=tags,
                    notes=notes,
                )
            )
        except (KeyError, ValueError, TypeError) as exc:
            raise ConfigError(f"{path} row {idx}: {exc}") from exc

    if not items:
        raise ConfigError(f"{path} holds no labeled stills.")
    return items


def _infer_tags_from_qc(shot: Shot, attempt_num: int) -> list[FailureTag]:
    att = next((a for a in shot.still.attempts if a.number == attempt_num), None)
    if att is None or att.qc is None:
        return []
    tags: list[FailureTag] = []
    for check in att.qc.checks:
        failed = check.status.value == "failed"
        low = check.score is not None and check.score < 0.70
        if not (failed or low):
            continue
        if check.name == "vision_spec" and "spec" not in tags:
            tags.append("spec")
        elif check.name == "vision_artifacts" and "artifact" not in tags:
            tags.append("artifact")
        elif check.name in ("vision_character", "identity") and "character" not in tags:
            tags.append("character")
    return tags


def _tricky_notes(shot: Shot, manifest: Manifest, initial_wardrobe: str | None) -> list[str]:
    notes: list[str] = []
    text = f"{shot.still_prompt} {manifest.visual_bible.lighting} {shot.setting}"
    if _DARK_SCENE_RE.search(text):
        notes.append("dark/moody")
    if _ANGLED_FACE_RE.search(shot.still_prompt):
        notes.append("angled-face")
    if shot.wardrobe and initial_wardrobe and shot.wardrobe != initial_wardrobe:
        notes.append(f"wardrobe:{shot.wardrobe}")
    if shot.framing is Framing.WIDE:
        notes.append("wide-shot")
    return notes


def scaffold_qc_dataset(
    projects: Sequence[Path],
    output_path: Path,
    *,
    target_size: int = 40,
    overwrite: bool = False,
) -> list[QCLabelItem]:
    """Sample a stratified 30-50 still starter dataset from existing project(s).

    Prioritizes a balanced mix of prior-passed/approved and prior-failed/rejected stills,
    including dark/moody scenes, angled faces, wardrobe changes, and wide shots. Review and edit
    the generated labels before running ``mvd bench qc``.
    """
    if output_path.exists() and not overwrite:
        raise ConfigError(f"{output_path} already exists. Pass --overwrite to replace it.")
    if not projects:
        raise ConfigError("Provide at least one project folder to scaffold a Vision QC dataset.")
    if isinstance(target_size, bool) or target_size < 1:
        raise ConfigError(f"target_size must be a positive integer, got {target_size!r}.")

    unique_projects = list(dict.fromkeys(Path(proj_dir).resolve() for proj_dir in projects))
    opened_projects = [open_bench_project(proj_dir) for proj_dir in unique_projects]
    prefixes = _project_id_prefixes([p.root for p in opened_projects])

    good_by_proj: list[list[tuple[int, QCLabelItem]]] = []
    bad_by_proj: list[list[tuple[int, QCLabelItem]]] = []

    for project in opened_projects:
        manifest = project.load_manifest()
        proj_prefix = prefixes[project.root.resolve()]
        initial_w = next((s.wardrobe for s in manifest.shots if s.wardrobe), None)
        p_good: list[tuple[int, QCLabelItem]] = []
        p_bad: list[tuple[int, QCLabelItem]] = []
        for shot in manifest.shots:
            for att in shot.still.attempts:
                if not att.path or not project.path(att.path).is_file():
                    continue
                is_selected = att.number == shot.still.selected_attempt
                if is_selected and shot.review.decision is ReviewDecision.APPROVED:
                    good = True
                elif is_selected and shot.review.decision is ReviewDecision.REJECTED:
                    good = False
                elif att.qc is not None:
                    good = bool(att.qc.passed)
                else:
                    continue
                tricky = _tricky_notes(shot, manifest, initial_w)
                tags = [] if good else _infer_tags_from_qc(shot, att.number)
                # Favor tricky cases and multi-tag / artifact / character defects.
                defect_bonus = sum(2 if t in ("artifact", "character") else 1 for t in tags)
                priority = len(tricky) * 2 + defect_bonus + (1 if is_selected else 0)
                item = QCLabelItem(
                    id=f"{proj_prefix}:{shot.id}:a{att.number}",
                    project=str(project.root.resolve()),
                    shot_id=shot.id,
                    attempt=att.number,
                    good=good,
                    tags=tags,
                    notes=", ".join(tricky),
                )
                if good:
                    p_good.append((priority, item))
                else:
                    p_bad.append((priority, item))
        p_good.sort(key=lambda pair: (-pair[0], pair[1].id))
        p_bad.sort(key=lambda pair: (-pair[0], pair[1].id))
        good_by_proj.append(p_good)
        bad_by_proj.append(p_bad)

    def _round_robin(groups: list[list[tuple[int, QCLabelItem]]]) -> list[QCLabelItem]:
        queues = [[item for _, item in grp] for grp in groups]
        out: list[QCLabelItem] = []
        while any(queues):
            for q in queues:
                if q:
                    out.append(q.pop(0))
        return out

    good_ordered = _round_robin(good_by_proj)
    bad_ordered = _round_robin(bad_by_proj)
    if not good_ordered and not bad_ordered:
        raise ConfigError("No still attempt image files with QC or review history were found.")

    half = max(1, target_size // 2)
    selected: list[QCLabelItem] = good_ordered[:half]
    selected += bad_ordered[: target_size - len(selected)]
    if len(selected) < target_size:
        used = {item.id for item in selected}
        remainder = [
            item for item in (good_ordered[half:] + bad_ordered[half:]) if item.id not in used
        ]
        selected.extend(remainder[: target_size - len(selected)])

    selected.sort(key=lambda i: (i.project, i.shot_id, i.attempt or 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() == ".json":
        payload = [
            {
                "project": item.project,
                "shot": item.shot_id,
                "attempt": item.attempt,
                "label": "pass" if item.good else "fail",
                "tags": item.tags,
                "notes": item.notes,
            }
            for item in selected
        ]
        atomic_write(output_path, json.dumps(payload, indent=2) + "\n")
    else:
        buf = io.StringIO()
        writer = csv.DictWriter(
            buf, fieldnames=["project", "shot", "attempt", "label", "tags", "notes"]
        )
        writer.writeheader()
        for item in selected:
            writer.writerow(
                {
                    "project": item.project,
                    "shot": item.shot_id,
                    "attempt": item.attempt if item.attempt is not None else "",
                    "label": "pass" if item.good else "fail",
                    "tags": ";".join(item.tags),
                    "notes": item.notes,
                }
            )
        atomic_write(output_path, buf.getvalue())
    return selected


def resolve_still_and_references(
    project: Project,
    manifest: Manifest,
    item: QCLabelItem,
    *,
    include_plates: bool = False,
) -> tuple[Shot, Path, Path | None, list[tuple[str, Path]]]:
    """Resolve the still path and locked reference images matching ``ShotRunner._check``."""
    try:
        shot = manifest.shot(item.shot_id)
    except KeyError as exc:
        raise ConfigError(f"{item.id}: unknown shot {item.shot_id!r} in {project.root}") from exc

    state = shot.asset(AssetKind.STILL)
    attempt_num = item.attempt if item.attempt is not None else state.selected_attempt
    if attempt_num is None and state.attempts:
        best = state.best_attempt() or state.attempts[-1]
        attempt_num = best.number
    attempt = next((a for a in state.attempts if a.number == attempt_num), None)
    if attempt is None or not attempt.path:
        raise ConfigError(f"{item.id}: still attempt {attempt_num} has no image path in manifest.")
    still_path = project.path(attempt.path)
    if not still_path.is_file():
        raise ConfigError(f"{item.id}: still image not found on disk: {still_path}")

    reference_image: Path | None = None
    extra_references: list[tuple[str, Path]] = []
    if shot.type.shows_singer:
        char = manifest.shot_character(shot)
        ref_set = char.reference_for(shot.wardrobe)
        sheet = (ref_set.sheet or ref_set.panel("front")) if ref_set else None
        if sheet:
            cand = project.path(sheet)
            if cand.is_file():
                reference_image = cand
        for sec_id, sec_char, sec_state in manifest.visible_characters(shot)[1:]:
            sec_ref = sec_char.reference_for(sec_state)
            sec_sheet = (sec_ref.sheet or sec_ref.panel("front")) if sec_ref else None
            if sec_sheet:
                sec_path = project.path(sec_sheet)
                if sec_path.is_file():
                    extra_references.append((sec_char.name or sec_id or "performer", sec_path))

    for prop in manifest.visual_bible.resolve_props(shot.props):
        if prop.plate:
            prop_path = project.path(prop.plate)
            if prop_path.is_file():
                extra_references.append((f"Prop: {prop.name}", prop_path))

    if include_plates:
        scene = manifest.visual_bible.find_set(shot.setting)
        if scene is not None and scene.plate:
            plate_path = project.path(scene.plate)
            if plate_path.is_file():
                extra_references.append((f"Prop: Set plate ({scene.name})", plate_path))

    return shot, still_path, reference_image, extra_references
