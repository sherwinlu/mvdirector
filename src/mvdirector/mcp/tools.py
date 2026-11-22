# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""MCP tool definitions and handlers wrapping ``mvdirector.api``."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mvdirector import api
from mvdirector.comfy.provider import ComfyEndpointProvider
from mvdirector.config import ReviewMode
from mvdirector.errors import ConfigError, MVDError
from mvdirector.mcp.jobs import JobContext, JobManager
from mvdirector.models import AssetKind, Manifest, ProjectMode, Shot, VideoStyle

ToolHandler = Callable[[dict[str, Any]], dict[str, Any]]

_READ_ONLY = {"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True}
_MUTATING = {"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False}
_IDEMPOTENT_MUTATING = {"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True}
_DESTRUCTIVE = {"readOnlyHint": False, "destructiveHint": True, "idempotentHint": False}


class WorkspaceGuard:
    """Resolves project and file paths while enforcing optional workspace root containment."""

    def __init__(
        self,
        *,
        default_project: Path | None = None,
        allowed_roots: Iterable[Path] | None = None,
    ) -> None:
        self.default_project = default_project.expanduser().resolve() if default_project else None
        self.allowed_roots: tuple[Path, ...] | None = (
            tuple(Path(r).expanduser().resolve() for r in allowed_roots)
            if allowed_roots is not None
            else None
        )
        if self.default_project is not None and self.allowed_roots is not None:
            self._check_contained(self.default_project, "default project")

    def _check_contained(self, resolved: Path, label: str) -> Path:
        if self.allowed_roots is None:
            return resolved
        if not any(resolved.is_relative_to(root) for root in self.allowed_roots):
            raise ConfigError(f"{label} '{resolved}' is outside the allowed workspace.")
        return resolved

    @staticmethod
    def _reject_nul(raw: str, label: str) -> str:
        if "\x00" in raw:
            raise ConfigError(f"{label} must not contain NUL bytes.")
        return raw

    def _resolve_user_path(self, candidate: Path, label: str) -> Path:
        try:
            return candidate.expanduser().resolve()
        except (RuntimeError, OSError) as exc:
            raise ConfigError(f"Invalid {label} path: {exc}") from exc

    def resolve_project(self, raw: Any) -> Path:
        if raw is None or (isinstance(raw, str) and not raw.strip()):
            if self.default_project is not None:
                return self._check_contained(self.default_project, "project_dir")
            raise ConfigError("project_dir is required (or start `mvd mcp` with --project <dir>).")
        if not isinstance(raw, str):
            raise ConfigError("project_dir must be a string path.")
        cleaned = self._reject_nul(raw.strip(), "project_dir")
        resolved = self._resolve_user_path(Path(cleaned), "project_dir")
        return self._check_contained(resolved, "project_dir")

    def resolve_path(self, project_dir: Path, raw: Any, *, label: str) -> Path:
        if not isinstance(raw, str) or not raw.strip():
            raise ConfigError(f"{label} must be a non-empty string path.")
        cleaned = self._reject_nul(raw.strip(), label)
        try:
            candidate = Path(cleaned).expanduser()
            resolved = (
                candidate.resolve()
                if candidate.is_absolute()
                else (project_dir / candidate).resolve()
            )
        except (RuntimeError, OSError) as exc:
            raise ConfigError(f"Invalid {label} path: {exc}") from exc
        return self._check_contained(resolved, label)

    def resolve_optional_path(self, project_dir: Path, raw: Any, *, label: str) -> Path | None:
        if raw is None or (isinstance(raw, str) and not raw.strip()):
            return None
        return self.resolve_path(project_dir, raw, label=label)


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: ToolHandler
    annotations: dict[str, Any]

    def descriptor(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
            "annotations": dict(self.annotations),
        }


# ------------------------------------------------------------------ argument parsing helpers


def _opt_str(args: dict[str, Any], key: str) -> str | None:
    val = args.get(key)
    if val is None:
        return None
    if not isinstance(val, str):
        raise ConfigError(f"'{key}' must be a string.")
    return val


def _req_str(args: dict[str, Any], key: str) -> str:
    val = _opt_str(args, key)
    if not val or not val.strip():
        raise ConfigError(f"'{key}' is required and must be a non-empty string.")
    return val


def _opt_bool(args: dict[str, Any], key: str, default: bool = False) -> bool:
    val = args.get(key, default)
    if not isinstance(val, bool):
        raise ConfigError(f"'{key}' must be a boolean.")
    return val


def _opt_int(args: dict[str, Any], key: str, *, minimum: int | None = None) -> int | None:
    val = args.get(key)
    if val is None:
        return None
    if isinstance(val, bool) or not isinstance(val, int):
        raise ConfigError(f"'{key}' must be an integer.")
    if minimum is not None and val < minimum:
        raise ConfigError(f"'{key}' must be >= {minimum}.")
    return val


def _opt_str_list(args: dict[str, Any], key: str) -> list[str] | None:
    val = args.get(key)
    if val is None:
        return None
    if not isinstance(val, list) or not all(isinstance(item, str) for item in val):
        raise ConfigError(f"'{key}' must be a list of strings.")
    return val or None


def _opt_enum(args: dict[str, Any], key: str, enum_cls: Any) -> Any:
    raw = _opt_str(args, key)
    if raw is None:
        return None
    try:
        return enum_cls(raw)
    except ValueError as exc:
        valid = ", ".join(m.value for m in enum_cls)
        raise ConfigError(f"Invalid '{key}': {raw!r}. Expected one of: {valid}.") from exc


def _project_prop() -> dict[str, Any]:
    return {
        "type": "string",
        "description": "Path to the mvdirector project folder (optional when --project is set).",
    }


def _background_prop(default: bool) -> dict[str, Any]:
    return {
        "type": "boolean",
        "default": default,
        "description": (
            "Run asynchronously in the background and return a job_id for mvd_task_status "
            f"(default: {str(default).lower()})."
        ),
    }


def _schema(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    body: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
    }
    if required:
        body["required"] = required
    return body


# ------------------------------------------------------------------ serialization helpers


def _review_decision_label(shot: Shot) -> str:
    if shot.review.decision is not None:
        return shot.review.decision.value
    return "pending" if shot.review.required else "none"


def _shot_summary(shot: Shot) -> dict[str, Any]:
    return {
        "id": shot.id,
        "index": shot.index,
        "start": round(shot.start, 3),
        "end": round(shot.end, 3),
        "duration": round(shot.end - shot.start, 3),
        "type": shot.type.value,
        "framing": shot.framing.value,
        "section": shot.section,
        "chapter_id": shot.chapter_id,
        "setting": shot.setting,
        "character": shot.character,
        "wardrobe": shot.wardrobe,
        "still_status": shot.still.status.value,
        "still_attempts": len(shot.still.attempts),
        "video_status": shot.video.status.value,
        "video_attempts": len(shot.video.attempts),
        "review_decision": _review_decision_label(shot),
        "review_required": shot.review.required,
    }


def _shot_detail(shot: Shot) -> dict[str, Any]:
    def _attempts(state: Any) -> list[dict[str, Any]]:
        out = []
        for att in state.attempts:
            entry: dict[str, Any] = {
                "number": att.number,
                "strategy": att.strategy.value,
                "seed": att.seed,
                "path": att.path,
                "error": att.error,
            }
            if att.qc is not None:
                entry["qc"] = {
                    "passed": att.qc.passed,
                    "score": round(att.qc.score, 4),
                    "issues": list(att.qc.issues),
                    "checks": [
                        {
                            "name": c.name,
                            "status": c.status.value,
                            "hard": c.hard,
                            "score": round(c.score, 4) if c.score is not None else None,
                            "detail": c.detail,
                            "fix_hint": c.fix_hint,
                        }
                        for c in att.qc.checks
                    ],
                }
            out.append(entry)
        return out

    return {
        **_shot_summary(shot),
        "characters": list(shot.characters),
        "singers": list(shot.singers),
        "lyrics": shot.lyrics,
        "props": list(shot.props),
        "still_prompt": shot.still_prompt,
        "motion_prompt": shot.motion_prompt,
        "allow_cuts": shot.allow_cuts,
        "previz_video": shot.previz_video,
        "audio_slice": shot.audio_slice.model_dump(mode="json") if shot.audio_slice else None,
        "num_frames": shot.num_frames,
        "still": {
            "status": shot.still.status.value,
            "selected_attempt": shot.still.selected_attempt,
            "attempts_allowed": shot.still.attempts_allowed,
            "accepted_by_reviewer": shot.still.accepted_by_reviewer,
            "attempts": _attempts(shot.still),
        },
        "video": {
            "status": shot.video.status.value,
            "selected_attempt": shot.video.selected_attempt,
            "attempts_allowed": shot.video.attempts_allowed,
            "upscaled_path": shot.video.upscaled_path,
            "attempts": _attempts(shot.video),
        },
        "review": {
            "decision": _review_decision_label(shot),
            "required": shot.review.required,
            "reasons": list(shot.review.reasons),
            "note": shot.review.note,
            "decided_at": shot.review.decided_at,
        },
    }


def _manifest_status_summary(
    manifest: Manifest,
    *,
    video_style: str | None = None,
    include_shots: bool = True,
    shot_ids: list[str] | None = None,
    chapter_ids: list[str] | None = None,
    review_filter: str | None = None,
) -> dict[str, Any]:
    still_counts = Counter(s.still.status.value for s in manifest.shots)
    video_counts = Counter(s.video.status.value for s in manifest.shots)
    review_counts = Counter(_review_decision_label(s) for s in manifest.shots)
    selected = manifest.shots
    if shot_ids is not None:
        wanted = set(shot_ids)
        selected = [s for s in selected if s.id in wanted]
    if chapter_ids is not None:
        wanted_ch = set(chapter_ids)
        selected = [s for s in selected if s.chapter_id in wanted_ch]
    if review_filter is not None:
        selected = [s for s in selected if _review_decision_label(s) == review_filter]

    payload: dict[str, Any] = {
        "title": manifest.title,
        "mode": manifest.mode.value,
        "video_style": video_style,
        "duration": round(manifest.duration, 3),
        "fps": manifest.fps,
        "total_shots": len(manifest.shots),
        "total_generations": manifest.total_generations,
        "still_counts": dict(still_counts),
        "video_counts": dict(video_counts),
        "review_counts": dict(review_counts),
        "qc_check_counts": {k: dict(v) for k, v in api.status_counts(manifest).items()},
        "updated_at": manifest.updated_at,
    }
    if include_shots:
        payload["shots"] = [_shot_summary(s) for s in selected]
    return payload


def _generate_result_summary(res: api.GenerateResult) -> dict[str, Any]:
    return {
        "active_checks": list(res.active_checks),
        "skipped_checks": [
            {"check": name, "reason": reason} for name, reason in res.skipped_checks
        ],
        "budget_message": res.budget_message,
        "status": _manifest_status_summary(
            res.manifest,
            video_style=res.project.config.planning.video_style.value,
            include_shots=False,
        ),
    }


# ------------------------------------------------------------------ registry builder


def build_tool_registry(
    *,
    guard: WorkspaceGuard,
    jobs: JobManager,
    comfy_provider: ComfyEndpointProvider | None = None,
) -> dict[str, ToolSpec]:
    """Build all MCP tool specifications wired to ``mvdirector.api``."""

    def _run_or_submit(
        stage: str,
        project_dir: Path,
        background: bool,
        work: Callable[[JobContext | None], dict[str, Any]],
    ) -> dict[str, Any]:
        if background:
            return jobs.submit(stage, str(project_dir), work)
        return work(None)

    # --- project & diagnostics ------------------------------------------------

    def h_init_project(args: dict[str, Any]) -> dict[str, Any]:
        directory = guard.resolve_project(args.get("directory"))
        mode = _opt_enum(args, "mode", ProjectMode) or ProjectMode.STANDARD
        created = api.init_project(directory, mode=mode)
        return {"project_dir": str(created), "mode": mode.value}

    def h_validate_project(args: dict[str, Any]) -> dict[str, Any]:
        project_dir = guard.resolve_project(args.get("project_dir"))
        checks = api.validate_project(project_dir, comfy_provider=comfy_provider)
        return {
            "project_dir": str(project_dir),
            "ok": all(c.ok for c in checks),
            "checks": [{"label": c.label, "ok": c.ok, "problem": c.problem} for c in checks],
        }

    def h_check_llm(args: dict[str, Any]) -> dict[str, Any]:
        project_dir = guard.resolve_project(args.get("project_dir"))
        skip_image = _opt_bool(args, "skip_image", False)
        outcome = api.check_llm(project_dir, skip_image=skip_image)
        res = outcome.result
        return {
            "project_dir": str(project_dir),
            "enabled": outcome.enabled,
            "provider": outcome.provider,
            "model": outcome.model,
            "ok": outcome.ok,
            "vision_needed": outcome.vision_needed,
            "tested_images": outcome.tested_images,
            "vision_failed": outcome.vision_failed,
            "json_ok": res.json_ok if res else None,
            "vision": res.vision if res else None,
        }

    def h_status(args: dict[str, Any]) -> dict[str, Any]:
        project_dir = guard.resolve_project(args.get("project_dir"))
        review_mode = _opt_enum(args, "review_mode", ReviewMode)
        refresh = _opt_bool(args, "refresh", True)
        include_shots = _opt_bool(args, "include_shots", True)
        shot_ids = _opt_str_list(args, "shot_ids")
        chapter_ids = _opt_str_list(args, "chapters")
        review_filter = _opt_str(args, "review_status")
        project, manifest = api.load_status(project_dir, review_mode=review_mode, refresh=refresh)
        return {
            "project_dir": str(project.root),
            **_manifest_status_summary(
                manifest,
                video_style=project.config.planning.video_style.value,
                include_shots=include_shots,
                shot_ids=shot_ids,
                chapter_ids=chapter_ids,
                review_filter=review_filter,
            ),
        }

    def h_get_shot(args: dict[str, Any]) -> dict[str, Any]:
        project_dir = guard.resolve_project(args.get("project_dir"))
        shot_id = _req_str(args, "shot_id")
        _, manifest = api.load_status(project_dir, refresh=False)
        for shot in manifest.shots:
            if shot.id == shot_id:
                return _shot_detail(shot)
        raise MVDError(f"Shot '{shot_id}' not found in manifest.")

    def h_get_manifest_summary(args: dict[str, Any]) -> dict[str, Any]:
        project_dir = guard.resolve_project(args.get("project_dir"))
        project = api.open_project(project_dir, pull=False)
        manifest = project.load_manifest()
        vb = manifest.visual_bible
        char_map = manifest.characters or {"primary": manifest.character}
        chars = {
            cid: {
                "name": ch.name,
                "appearance": ch.appearance,
                "invariants": list(ch.invariants),
                "parent_character": ch.parent_character,
                "wardrobe_states": [w.id for w in ch.wardrobe],
                "locked_references": {
                    state_id: {
                        "sheet": ref.sheet,
                        "panels": list(ref.panels),
                        "video": ref.video,
                    }
                    for state_id, ref in ch.references.items()
                },
            }
            for cid, ch in char_map.items()
        }
        analysis_summary: dict[str, Any] | None = None
        try:
            analysis = project.load_analysis()
        except MVDError:
            analysis = None
        if analysis is not None:
            analysis_summary = {
                "duration": round(analysis.duration, 3),
                "tempo_bpm": round(analysis.tempo_bpm, 2),
                "beats": len(analysis.beats),
                "bars": len(analysis.bars),
                "vocal_windows": len(analysis.vocal_windows),
                "lyric_lines": len(analysis.lyric_lines),
                "chapters": len(analysis.chapters),
            }
        return {
            "project_dir": str(project.root),
            "title": manifest.title,
            "mode": manifest.mode.value,
            "visual_bible": {
                "concept": vb.concept,
                "style": vb.style,
                "palette": vb.palette,
                "lighting": vb.lighting,
                "rules": list(vb.rules),
                "settings": [
                    {"name": s.name, "plate": s.plate, "parent_set": s.parent_set}
                    for s in vb.settings
                ],
                "props": [{"name": p.name, "plate": p.plate} for p in vb.props],
            },
            "characters": chars,
            "chapters": [
                {"id": ch.id, "title": ch.title, "start": ch.start, "end": ch.end}
                for ch in manifest.chapters
            ],
            "analysis": analysis_summary,
        }

    def h_qc_report(args: dict[str, Any]) -> dict[str, Any]:
        project_dir = guard.resolve_project(args.get("project_dir"))
        markdown = _opt_bool(args, "markdown", False)
        return {"report": api.qc_report(project_dir, markdown=markdown)}

    def h_calibrate(args: dict[str, Any]) -> dict[str, Any]:
        project_dir = guard.resolve_project(args.get("project_dir"))
        labels = guard.resolve_optional_path(project_dir, args.get("labels"), label="labels")
        return {"report": api.calibrate(project_dir, labels=labels)}

    def h_tune(args: dict[str, Any]) -> dict[str, Any]:
        project_dir = guard.resolve_project(args.get("project_dir"))
        apply = _opt_bool(args, "apply", False)
        force = _opt_bool(args, "force", False)
        res = api.tune(project_dir, apply=apply, force=force)
        return {
            "applied_to_yaml": res.applied_to_yaml,
            "adjustments": [
                {
                    "key": adj.key,
                    "current": adj.current,
                    "recommended": adj.recommended,
                    "applied": adj.applied,
                    "reason": adj.reason,
                }
                for adj in res.adjustments
            ],
            "report": res.report,
        }

    # --- brief, prepare, analyze, plan ----------------------------------------

    def h_brief(args: dict[str, Any]) -> dict[str, Any]:
        project_dir = guard.resolve_project(args.get("project_dir"))
        idea = _req_str(args, "idea")
        duration = _opt_int(args, "duration", minimum=1)
        vibe = _opt_str(args, "vibe")
        overwrite = _opt_bool(args, "overwrite", False)
        res = api.brief(project_dir, idea, duration=duration, vibe=vibe, overwrite=overwrite)
        return {
            "lyrics_path": str(res.lyrics_path),
            "style_path": str(res.style_path),
            "suno_style": res.suno_style,
        }

    def h_prepare(args: dict[str, Any]) -> dict[str, Any]:
        project_dir = guard.resolve_project(args.get("project_dir"))
        raw_audio = _opt_str_list(args, "audio")
        audio_paths = (
            [guard.resolve_path(project_dir, p, label="audio") for p in raw_audio]
            if raw_audio is not None
            else None
        )
        overwrite = _opt_bool(args, "overwrite", False)
        background = _opt_bool(args, "background", False)

        def _work(_ctx: JobContext | None) -> dict[str, Any]:
            res = api.prepare(project_dir, audio=audio_paths, overwrite=overwrite)
            return {
                "mix_path": str(res.mix_path),
                "vocals_path": str(res.vocals_path) if res.vocals_path else None,
                "drums_path": str(res.drums_path) if res.drums_path else None,
                "lyrics_path": str(res.lyrics_path) if res.lyrics_path else None,
                "style_path": str(res.style_path) if res.style_path else None,
                "tempo_bpm": round(res.tempo_bpm, 2),
                "duration": round(res.duration, 3),
                "vocal_windows": res.vocal_windows,
                "transcribed_words": res.transcribed_words,
                "inferred_with_llm": res.inferred_with_llm,
                "stems_ingested": res.stems_ingested,
            }

        return _run_or_submit("prepare", project_dir, background, _work)

    def h_analyze(args: dict[str, Any]) -> dict[str, Any]:
        project_dir = guard.resolve_project(args.get("project_dir"))
        apply_tuning = _opt_bool(args, "apply_tuning", False)
        mode = _opt_enum(args, "mode", ProjectMode)
        tracklist = guard.resolve_optional_path(
            project_dir, args.get("tracklist"), label="tracklist"
        )
        background = _opt_bool(args, "background", False)

        def _work(_ctx: JobContext | None) -> dict[str, Any]:
            project, analysis = api.analyze(
                project_dir, apply_tuning=apply_tuning, mode=mode, tracklist=tracklist
            )
            return {
                "project_dir": str(project.root),
                "mode": project.config.mode.value,
                "duration": round(analysis.duration, 3),
                "tempo_bpm": round(analysis.tempo_bpm, 2),
                "beats": len(analysis.beats),
                "bars": len(analysis.bars),
                "vocal_windows": len(analysis.vocal_windows),
                "lyric_lines": len(analysis.lyric_lines),
                "chapters": len(analysis.chapters),
            }

        return _run_or_submit("analyze", project_dir, background, _work)

    def h_plan(args: dict[str, Any]) -> dict[str, Any]:
        project_dir = guard.resolve_project(args.get("project_dir"))
        overwrite = _opt_bool(args, "overwrite", False)
        mode = _opt_enum(args, "mode", ProjectMode)
        tracklist = guard.resolve_optional_path(
            project_dir, args.get("tracklist"), label="tracklist"
        )
        video_style = _opt_enum(args, "video_style", VideoStyle)
        background = _opt_bool(args, "background", False)

        def _work(_ctx: JobContext | None) -> dict[str, Any]:
            project, manifest = api.plan(
                project_dir,
                overwrite=overwrite,
                mode=mode,
                tracklist=tracklist,
                video_style=video_style,
            )
            return {
                "project_dir": str(project.root),
                "title": manifest.title,
                "mode": manifest.mode.value,
                "total_shots": len(manifest.shots),
                "characters": list(manifest.characters.keys()) or ["primary"],
                "settings": [s.name for s in manifest.visual_bible.settings],
                "props": [p.name for p in manifest.visual_bible.props],
            }

        return _run_or_submit("plan", project_dir, background, _work)

    # --- references, plates, props, previz ------------------------------------

    def h_generate_references(args: dict[str, Any]) -> dict[str, Any]:
        project_dir = guard.resolve_project(args.get("project_dir"))
        state = _opt_str(args, "state")
        character = _opt_str(args, "character")
        from_character = _opt_str(args, "from_character")
        count = _opt_int(args, "count", minimum=1)
        background = _opt_bool(args, "background", True)

        def _work(_ctx: JobContext | None) -> dict[str, Any]:
            paths = api.generate_reference_candidates(
                project_dir,
                state=state,
                character=character,
                from_character=from_character,
                count=count,
                comfy_provider=comfy_provider,
            )
            return {"candidates": [str(p) for p in paths]}

        return _run_or_submit("generate_references", project_dir, background, _work)

    def h_lock_reference(args: dict[str, Any]) -> dict[str, Any]:
        project_dir = guard.resolve_project(args.get("project_dir"))
        image = guard.resolve_path(project_dir, args.get("image"), label="image")
        state = _opt_str(args, "state")
        character = _opt_str(args, "character")
        single = _opt_bool(args, "single", False)
        ref = api.lock_reference(
            project_dir, image, state=state, character=character, single=single
        )
        return {"sheet": ref.sheet, "panels": list(ref.panels), "video": ref.video}

    def h_generate_reference_videos(args: dict[str, Any]) -> dict[str, Any]:
        project_dir = guard.resolve_project(args.get("project_dir"))
        state = _opt_str(args, "state")
        character = _opt_str(args, "character")
        count = _opt_int(args, "count", minimum=1)
        background = _opt_bool(args, "background", True)

        def _work(_ctx: JobContext | None) -> dict[str, Any]:
            paths = api.generate_reference_videos(
                project_dir,
                state=state,
                character=character,
                count=count,
                comfy_provider=comfy_provider,
            )
            return {"candidates": [str(p) for p in paths]}

        return _run_or_submit("generate_reference_videos", project_dir, background, _work)

    def h_lock_reference_video(args: dict[str, Any]) -> dict[str, Any]:
        project_dir = guard.resolve_project(args.get("project_dir"))
        video = guard.resolve_path(project_dir, args.get("video"), label="video")
        state = _opt_str(args, "state")
        character = _opt_str(args, "character")
        locked = api.lock_reference_video(project_dir, video, state=state, character=character)
        return {"video": locked}

    def h_generate_plates(args: dict[str, Any]) -> dict[str, Any]:
        project_dir = guard.resolve_project(args.get("project_dir"))
        set_name = _opt_str(args, "set_name")
        style_ref = guard.resolve_optional_path(
            project_dir, args.get("style_ref"), label="style_ref"
        )
        embed_set = _opt_str(args, "embed_set")
        from_set = _opt_str(args, "from_set")
        count = _opt_int(args, "count", minimum=1)
        background = _opt_bool(args, "background", True)

        def _work(_ctx: JobContext | None) -> dict[str, Any]:
            res = api.generate_plates(
                project_dir,
                set_name=set_name,
                style_ref=style_ref,
                embed_set=embed_set,
                from_set=from_set,
                count=count,
                comfy_provider=comfy_provider,
            )
            return {"plates": {name: [str(p) for p in paths] for name, paths in res.items()}}

        return _run_or_submit("generate_plates", project_dir, background, _work)

    def h_lock_plate(args: dict[str, Any]) -> dict[str, Any]:
        project_dir = guard.resolve_project(args.get("project_dir"))
        image = guard.resolve_path(project_dir, args.get("image"), label="image")
        set_name = _req_str(args, "set_name")
        locked = api.lock_plate(project_dir, image, set_name)
        return {"set_name": set_name, "plate": str(locked)}

    def h_generate_props(args: dict[str, Any]) -> dict[str, Any]:
        project_dir = guard.resolve_project(args.get("project_dir"))
        prop = _opt_str(args, "prop")
        count = _opt_int(args, "count", minimum=1)
        background = _opt_bool(args, "background", True)

        def _work(_ctx: JobContext | None) -> dict[str, Any]:
            res = api.generate_props(
                project_dir, prop=prop, count=count, comfy_provider=comfy_provider
            )
            return {"props": {name: [str(p) for p in paths] for name, paths in res.items()}}

        return _run_or_submit("generate_props", project_dir, background, _work)

    def h_lock_prop(args: dict[str, Any]) -> dict[str, Any]:
        project_dir = guard.resolve_project(args.get("project_dir"))
        image = guard.resolve_path(project_dir, args.get("image"), label="image")
        prop = _req_str(args, "prop")
        locked = api.lock_prop(project_dir, image, prop)
        return {"prop": prop, "plate": str(locked)}

    def h_generate_previz(args: dict[str, Any]) -> dict[str, Any]:
        project_dir = guard.resolve_project(args.get("project_dir"))
        shot_ids = _opt_str_list(args, "shot_ids")
        overwrite = _opt_bool(args, "overwrite", False)
        background = _opt_bool(args, "background", True)

        def _work(_ctx: JobContext | None) -> dict[str, Any]:
            paths = api.generate_previz(project_dir, shot_ids=shot_ids, overwrite=overwrite)
            return {"previz_videos": [str(p) for p in paths]}

        return _run_or_submit("generate_previz", project_dir, background, _work)

    def h_lock_previz(args: dict[str, Any]) -> dict[str, Any]:
        project_dir = guard.resolve_project(args.get("project_dir"))
        video = guard.resolve_path(project_dir, args.get("video"), label="video")
        shot_id = _req_str(args, "shot_id")
        mapping = _opt_str(args, "mapping")
        locked = api.lock_previz(project_dir, video, shot_id, mapping=mapping)
        return {"shot_id": shot_id, "previz_video": str(locked)}

    def h_refine_previz(args: dict[str, Any]) -> dict[str, Any]:
        project_dir = guard.resolve_project(args.get("project_dir"))
        shot_id = _req_str(args, "shot_id")
        feedback = _req_str(args, "feedback")
        background = _opt_bool(args, "background", True)

        def _work(_ctx: JobContext | None) -> dict[str, Any]:
            path = api.refine_previz(project_dir, shot_id, feedback)
            return {"shot_id": shot_id, "previz_video": str(path)}

        return _run_or_submit("refine_previz", project_dir, background, _work)

    # --- generation, upscale, review, assemble, clean -------------------------

    def h_generate(args: dict[str, Any]) -> dict[str, Any]:
        project_dir = guard.resolve_project(args.get("project_dir"))
        raw_kinds = _opt_str_list(args, "kinds") or ["still", "video"]
        try:
            kinds = [AssetKind(k) for k in raw_kinds]
        except ValueError as exc:
            raise ConfigError("Invalid 'kinds'. Expected values from: still, video.") from exc
        shots = _opt_str_list(args, "shots")
        chapters = _opt_str_list(args, "chapters")
        review_mode = _opt_enum(args, "review_mode", ReviewMode)
        skip_preflight = _opt_bool(args, "skip_preflight", False)
        draft = _opt_bool(args, "draft", False)
        background = _opt_bool(args, "background", True)

        def _work(ctx: JobContext | None) -> dict[str, Any]:
            observer = ctx.on_run_event if ctx is not None else None
            res = api.generate(
                project_dir,
                kinds,
                shots=shots,
                review_mode=review_mode,
                skip_preflight=skip_preflight,
                draft=draft,
                observer=observer,
                comfy_provider=comfy_provider,
                chapters=chapters,
            )
            return _generate_result_summary(res)

        return _run_or_submit("generate", project_dir, background, _work)

    def h_upscale(args: dict[str, Any]) -> dict[str, Any]:
        project_dir = guard.resolve_project(args.get("project_dir"))
        shots = _opt_str_list(args, "shots")
        force = _opt_bool(args, "force", False)
        overwrite = _opt_bool(args, "overwrite", False)
        width = _opt_int(args, "width", minimum=64)
        height = _opt_int(args, "height", minimum=64)
        background = _opt_bool(args, "background", True)

        def _work(ctx: JobContext | None) -> dict[str, Any]:
            def _on_done(shot_id: str, done: int, total: int) -> None:
                if ctx is not None:
                    ctx.set_progress(last_shot=shot_id, completed=done, total=total)
                    ctx.check_cancelled()

            written = api.upscale(
                project_dir,
                shots=shots,
                force=force,
                overwrite=overwrite,
                comfy_provider=comfy_provider,
                width=width,
                height=height,
                on_shot_done=_on_done if ctx is not None else None,
            )
            return {"upscaled": [str(p) for p in written]}

        return _run_or_submit("upscale", project_dir, background, _work)

    def h_approve(args: dict[str, Any]) -> dict[str, Any]:
        project_dir = guard.resolve_project(args.get("project_dir"))
        shot_ids = _opt_str_list(args, "shot_ids")
        chapters = _opt_str_list(args, "chapters")
        still = _opt_bool(args, "still", False)
        api.approve(project_dir, shot_ids=shot_ids, still=still, chapters=chapters)
        return {
            "approved": True,
            "shot_ids": shot_ids,
            "chapters": chapters,
            "still_only": still,
        }

    def h_reject(args: dict[str, Any]) -> dict[str, Any]:
        project_dir = guard.resolve_project(args.get("project_dir"))
        shot_id = _req_str(args, "shot_id")
        note = _req_str(args, "note")
        target = _opt_enum(args, "target", AssetKind) or AssetKind.VIDEO
        api.reject(project_dir, shot_id, note, target=target)
        return {"rejected": True, "shot_id": shot_id, "target": target.value, "note": note}

    def h_assemble(args: dict[str, Any]) -> dict[str, Any]:
        project_dir = guard.resolve_project(args.get("project_dir"))
        output = guard.resolve_optional_path(project_dir, args.get("output"), label="output")
        force = _opt_bool(args, "force", False)
        review_mode = _opt_enum(args, "review_mode", ReviewMode)
        draft = _opt_bool(args, "draft", False)
        width = _opt_int(args, "width", minimum=64)
        height = _opt_int(args, "height", minimum=64)
        chapters = _opt_str_list(args, "chapters")
        rebuild_cache = _opt_bool(args, "rebuild_cache", False)
        background = _opt_bool(args, "background", True)

        def _work(_ctx: JobContext | None) -> dict[str, Any]:
            result = api.assemble(
                project_dir,
                output=output,
                force=force,
                review_mode=review_mode,
                draft=draft,
                width=width,
                height=height,
                chapters=chapters,
                rebuild_cache=rebuild_cache,
            )
            return {"output": str(result)}

        return _run_or_submit("assemble", project_dir, background, _work)

    def h_clean(args: dict[str, Any]) -> dict[str, Any]:
        project_dir = guard.resolve_project(args.get("project_dir"))
        delete = _opt_bool(args, "delete", False)
        plan = api.clean(project_dir, delete=delete)
        return {
            "deleted": delete,
            "superseded": [str(p) for p in plan.superseded],
            "orphaned": [str(p) for p in plan.orphaned],
            "total_files": len(plan.files),
            "size_bytes": plan.size_bytes,
        }

    # --- background task control ----------------------------------------------

    def h_task_status(args: dict[str, Any]) -> dict[str, Any]:
        job_id = _opt_str(args, "job_id")
        if job_id is None:
            status_filter = _opt_str(args, "status")
            return {"jobs": jobs.list_jobs(status=status_filter)}
        event_limit = _opt_int(args, "event_limit", minimum=0)
        try:
            return jobs.get(job_id, event_limit=20 if event_limit is None else event_limit)
        except KeyError as exc:
            raise MVDError(str(exc)) from exc

    def h_task_stop(args: dict[str, Any]) -> dict[str, Any]:
        job_id = _req_str(args, "job_id")
        try:
            return jobs.stop(job_id)
        except KeyError as exc:
            raise MVDError(str(exc)) from exc

    specs = [
        ToolSpec(
            "mvd_init_project",
            "Create a new mvdirector project folder with starter config and workflow bindings.",
            _schema(
                {
                    "directory": {"type": "string", "description": "Folder to create."},
                    "mode": {
                        "type": "string",
                        "enum": ["standard", "mix"],
                        "default": "standard",
                    },
                },
                required=["directory"],
            ),
            h_init_project,
            _MUTATING,
        ),
        ToolSpec(
            "mvd_validate_project",
            "Check ffmpeg/ffprobe, workflow bindings, ComfyUI connectivity, and LLM config.",
            _schema({"project_dir": _project_prop()}),
            h_validate_project,
            _READ_ONLY,
        ),
        ToolSpec(
            "mvd_check_llm",
            "Test the configured LLM backend (credentials, JSON output, and optional vision).",
            _schema(
                {
                    "project_dir": _project_prop(),
                    "skip_image": {"type": "boolean", "default": False},
                }
            ),
            h_check_llm,
            _READ_ONLY,
        ),
        ToolSpec(
            "mvd_status",
            "Load project status: shot counts by still/video/review status and shot summaries.",
            _schema(
                {
                    "project_dir": _project_prop(),
                    "review_mode": {
                        "type": "string",
                        "enum": [m.value for m in ReviewMode],
                    },
                    "refresh": {"type": "boolean", "default": True},
                    "include_shots": {"type": "boolean", "default": True},
                    "shot_ids": {"type": "array", "items": {"type": "string"}},
                    "chapters": {"type": "array", "items": {"type": "string"}},
                    "review_status": {"type": "string"},
                }
            ),
            h_status,
            _READ_ONLY,
        ),
        ToolSpec(
            "mvd_get_shot",
            "Inspect one shot's prompts, lyrics, attempts, QC checks/issues, and review state.",
            _schema(
                {
                    "project_dir": _project_prop(),
                    "shot_id": {"type": "string", "description": "Shot id (e.g. 's001')."},
                },
                required=["shot_id"],
            ),
            h_get_shot,
            _READ_ONLY,
        ),
        ToolSpec(
            "mvd_get_manifest_summary",
            "Inspect the visual bible, cast/references, sets/plates, props, and analysis summary.",
            _schema({"project_dir": _project_prop()}),
            h_get_manifest_summary,
            _READ_ONLY,
        ),
        ToolSpec(
            "mvd_qc_report",
            "Return per-check passed/failed/skipped QC statistics across all attempts.",
            _schema(
                {
                    "project_dir": _project_prop(),
                    "markdown": {"type": "boolean", "default": False},
                }
            ),
            h_qc_report,
            _READ_ONLY,
        ),
        ToolSpec(
            "mvd_calibrate",
            "Suggest QC threshold adjustments from reviewed shots or a labels CSV.",
            _schema(
                {
                    "project_dir": _project_prop(),
                    "labels": {"type": "string", "description": "Optional CSV path."},
                }
            ),
            h_calibrate,
            _READ_ONLY,
        ),
        ToolSpec(
            "mvd_tune",
            "Profile track characteristics and recommend (or apply) project.yaml tuning.",
            _schema(
                {
                    "project_dir": _project_prop(),
                    "apply": {"type": "boolean", "default": False},
                    "force": {"type": "boolean", "default": False},
                }
            ),
            h_tune,
            _IDEMPOTENT_MUTATING,
        ),
        ToolSpec(
            "mvd_brief",
            "Draft Suno-ready lyrics.txt and style.md from a song concept.",
            _schema(
                {
                    "project_dir": _project_prop(),
                    "idea": {"type": "string", "description": "Song concept or premise."},
                    "duration": {"type": "integer", "minimum": 1},
                    "vibe": {"type": "string"},
                    "overwrite": {"type": "boolean", "default": False},
                },
                required=["idea"],
            ),
            h_brief,
            _MUTATING,
        ),
        ToolSpec(
            "mvd_prepare",
            "Ingest audio files or Suno/DAW stem packs into mix, stems, lyrics, and style.",
            _schema(
                {
                    "project_dir": _project_prop(),
                    "audio": {"type": "array", "items": {"type": "string"}},
                    "overwrite": {"type": "boolean", "default": False},
                    "background": _background_prop(False),
                }
            ),
            h_prepare,
            _MUTATING,
        ),
        ToolSpec(
            "mvd_analyze",
            "Analyze audio for beats, bars, vocal activity, lyric alignment, or mix chapters.",
            _schema(
                {
                    "project_dir": _project_prop(),
                    "apply_tuning": {"type": "boolean", "default": False},
                    "mode": {"type": "string", "enum": ["standard", "mix"]},
                    "tracklist": {"type": "string"},
                    "background": _background_prop(False),
                }
            ),
            h_analyze,
            _IDEMPOTENT_MUTATING,
        ),
        ToolSpec(
            "mvd_plan",
            "Create manifest.json with visual bible, cast, sets, props, and shot prompts.",
            _schema(
                {
                    "project_dir": _project_prop(),
                    "overwrite": {"type": "boolean", "default": False},
                    "mode": {"type": "string", "enum": ["standard", "mix"]},
                    "tracklist": {"type": "string"},
                    "video_style": {
                        "type": "string",
                        "enum": [v.value for v in VideoStyle],
                    },
                    "background": _background_prop(False),
                }
            ),
            h_plan,
            _MUTATING,
        ),
        ToolSpec(
            "mvd_generate_references",
            "Generate candidate 4-panel character reference sheets via ComfyUI.",
            _schema(
                {
                    "project_dir": _project_prop(),
                    "state": {"type": "string"},
                    "character": {"type": "string"},
                    "from_character": {"type": "string"},
                    "count": {"type": "integer", "minimum": 1},
                    "background": _background_prop(True),
                }
            ),
            h_generate_references,
            _MUTATING,
        ),
        ToolSpec(
            "mvd_lock_reference",
            "Lock a reference sheet image (cutting panels unless single=true) for a character.",
            _schema(
                {
                    "project_dir": _project_prop(),
                    "image": {"type": "string"},
                    "state": {"type": "string"},
                    "character": {"type": "string"},
                    "single": {"type": "boolean", "default": False},
                },
                required=["image"],
            ),
            h_lock_reference,
            _IDEMPOTENT_MUTATING,
        ),
        ToolSpec(
            "mvd_generate_reference_videos",
            "Generate candidate silent turnaround videos from a locked reference sheet.",
            _schema(
                {
                    "project_dir": _project_prop(),
                    "state": {"type": "string"},
                    "character": {"type": "string"},
                    "count": {"type": "integer", "minimum": 1},
                    "background": _background_prop(True),
                }
            ),
            h_generate_reference_videos,
            _MUTATING,
        ),
        ToolSpec(
            "mvd_lock_reference_video",
            "Lock a silent turnaround reference video for a character/wardrobe state.",
            _schema(
                {
                    "project_dir": _project_prop(),
                    "video": {"type": "string"},
                    "state": {"type": "string"},
                    "character": {"type": "string"},
                },
                required=["video"],
            ),
            h_lock_reference_video,
            _IDEMPOTENT_MUTATING,
        ),
        ToolSpec(
            "mvd_generate_plates",
            "Generate candidate empty set plate images via ComfyUI.",
            _schema(
                {
                    "project_dir": _project_prop(),
                    "set_name": {"type": "string"},
                    "style_ref": {"type": "string"},
                    "embed_set": {"type": "string"},
                    "from_set": {"type": "string"},
                    "count": {"type": "integer", "minimum": 1},
                    "background": _background_prop(True),
                }
            ),
            h_generate_plates,
            _MUTATING,
        ),
        ToolSpec(
            "mvd_lock_plate",
            "Lock an image as a set's empty plate.",
            _schema(
                {
                    "project_dir": _project_prop(),
                    "image": {"type": "string"},
                    "set_name": {"type": "string"},
                },
                required=["image", "set_name"],
            ),
            h_lock_plate,
            _IDEMPOTENT_MUTATING,
        ),
        ToolSpec(
            "mvd_generate_props",
            "Generate candidate two-angle reference sheets for recurring hero props.",
            _schema(
                {
                    "project_dir": _project_prop(),
                    "prop": {"type": "string"},
                    "count": {"type": "integer", "minimum": 1},
                    "background": _background_prop(True),
                }
            ),
            h_generate_props,
            _MUTATING,
        ),
        ToolSpec(
            "mvd_lock_prop",
            "Lock a two-angle prop reference sheet.",
            _schema(
                {
                    "project_dir": _project_prop(),
                    "image": {"type": "string"},
                    "prop": {"type": "string"},
                },
                required=["image", "prop"],
            ),
            h_lock_prop,
            _IDEMPOTENT_MUTATING,
        ),
        ToolSpec(
            "mvd_generate_previz",
            "Plan and render flat-shaded 3D blocking previz MP4s for shots.",
            _schema(
                {
                    "project_dir": _project_prop(),
                    "shot_ids": {"type": "array", "items": {"type": "string"}},
                    "overwrite": {"type": "boolean", "default": False},
                    "background": _background_prop(True),
                }
            ),
            h_generate_previz,
            _MUTATING,
        ),
        ToolSpec(
            "mvd_lock_previz",
            "Attach a custom 3D blocking video to a shot.",
            _schema(
                {
                    "project_dir": _project_prop(),
                    "video": {"type": "string"},
                    "shot_id": {"type": "string"},
                    "mapping": {"type": "string"},
                },
                required=["video", "shot_id"],
            ),
            h_lock_previz,
            _IDEMPOTENT_MUTATING,
        ),
        ToolSpec(
            "mvd_refine_previz",
            "Refine a shot's 3D blocking plan with LLM feedback and re-render its previz MP4.",
            _schema(
                {
                    "project_dir": _project_prop(),
                    "shot_id": {"type": "string"},
                    "feedback": {"type": "string"},
                    "background": _background_prop(True),
                },
                required=["shot_id", "feedback"],
            ),
            h_refine_previz,
            _MUTATING,
        ),
        ToolSpec(
            "mvd_generate",
            "Generate and QC stills and/or videos with automatic retries (async by default).",
            _schema(
                {
                    "project_dir": _project_prop(),
                    "kinds": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["still", "video"]},
                        "default": ["still", "video"],
                    },
                    "shots": {"type": "array", "items": {"type": "string"}},
                    "chapters": {"type": "array", "items": {"type": "string"}},
                    "review_mode": {
                        "type": "string",
                        "enum": [m.value for m in ReviewMode],
                    },
                    "skip_preflight": {"type": "boolean", "default": False},
                    "draft": {"type": "boolean", "default": False},
                    "background": _background_prop(True),
                }
            ),
            h_generate,
            _MUTATING,
        ),
        ToolSpec(
            "mvd_upscale",
            "Upscale winning/approved draft clips through the configured upscale workflow.",
            _schema(
                {
                    "project_dir": _project_prop(),
                    "shots": {"type": "array", "items": {"type": "string"}},
                    "force": {"type": "boolean", "default": False},
                    "overwrite": {"type": "boolean", "default": False},
                    "width": {"type": "integer", "minimum": 64},
                    "height": {"type": "integer", "minimum": 64},
                    "background": _background_prop(True),
                }
            ),
            h_upscale,
            _MUTATING,
        ),
        ToolSpec(
            "mvd_approve",
            "Approve shots (or mix chapters), or accept an exhausted still.",
            _schema(
                {
                    "project_dir": _project_prop(),
                    "shot_ids": {"type": "array", "items": {"type": "string"}},
                    "chapters": {"type": "array", "items": {"type": "string"}},
                    "still": {"type": "boolean", "default": False},
                }
            ),
            h_approve,
            _IDEMPOTENT_MUTATING,
        ),
        ToolSpec(
            "mvd_reject",
            "Reject a shot and queue it for regeneration with a director's note.",
            _schema(
                {
                    "project_dir": _project_prop(),
                    "shot_id": {"type": "string"},
                    "note": {"type": "string"},
                    "target": {
                        "type": "string",
                        "enum": ["still", "video"],
                        "default": "video",
                    },
                },
                required=["shot_id", "note"],
            ),
            h_reject,
            _MUTATING,
        ),
        ToolSpec(
            "mvd_assemble",
            "Assemble approved clips (or mix chapters) with the original audio mix into an MP4.",
            _schema(
                {
                    "project_dir": _project_prop(),
                    "output": {"type": "string"},
                    "force": {"type": "boolean", "default": False},
                    "review_mode": {
                        "type": "string",
                        "enum": [m.value for m in ReviewMode],
                    },
                    "draft": {"type": "boolean", "default": False},
                    "width": {"type": "integer", "minimum": 64},
                    "height": {"type": "integer", "minimum": 64},
                    "chapters": {"type": "array", "items": {"type": "string"}},
                    "rebuild_cache": {"type": "boolean", "default": False},
                    "background": _background_prop(True),
                }
            ),
            h_assemble,
            _MUTATING,
        ),
        ToolSpec(
            "mvd_clean",
            "Plan (default dry-run) or delete superseded/orphaned asset files.",
            _schema(
                {
                    "project_dir": _project_prop(),
                    "delete": {"type": "boolean", "default": False},
                }
            ),
            h_clean,
            _DESTRUCTIVE,
        ),
        ToolSpec(
            "mvd_task_status",
            "Check progress, recent RunEvents, and results of a background job (or list all jobs).",
            _schema(
                {
                    "job_id": {"type": "string"},
                    "status": {
                        "type": "string",
                        "enum": ["queued", "running", "succeeded", "failed", "cancelled"],
                    },
                    "event_limit": {"type": "integer", "minimum": 0, "default": 20},
                }
            ),
            h_task_status,
            _READ_ONLY,
        ),
        ToolSpec(
            "mvd_task_stop",
            "Request cooperative cancellation of a running background job (progress is saved).",
            _schema(
                {"job_id": {"type": "string"}},
                required=["job_id"],
            ),
            h_task_stop,
            _IDEMPOTENT_MUTATING,
        ),
    ]
    return {spec.name: spec for spec in specs}
