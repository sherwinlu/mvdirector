---
name: mvdirector
description: Direct, generate, quality-check, review, and assemble music videos using the mvdirector MCP server.
---

# mvdirector Agent Skill

Use the `mvdirector` MCP tools (`mvd_*`) to drive an end-to-end music video project without leaving your coding or director agent.

## Recommended Pipeline Order

1. **Bootstrap & Verify**
   - `mvd_init_project(directory, mode="standard" | "mix")` to create a project folder.
   - `mvd_validate_project(project_dir)` to verify `ffmpeg`/`ffprobe`, workflow bindings, and ComfyUI reachability.
   - `mvd_check_llm(project_dir)` if LLM planning or Vision QC is enabled.

2. **Audio Intake, Analysis & Planning**
   - `mvd_brief(project_dir, idea, ...)` (optional) to draft Suno-ready `lyrics.txt` and `style.md`.
   - `mvd_prepare(project_dir, audio=[...])` to ingest audio files or Suno/DAW stem packs (`.zip`, folders, or loose stems).
   - `mvd_analyze(project_dir, apply_tuning=True)` to compute beats, bars, vocal windows, lyric alignment, and track profile (or mix chapters).
   - `mvd_tune(project_dir, apply=True)` to inspect or apply automatic `project.yaml` pacing/QC tuning.
   - `mvd_plan(project_dir, video_style="lipsync" | "narrative" | "visual")` to build `manifest.json`.
   - `mvd_get_manifest_summary(project_dir)` to review the visual bible, characters, wardrobe states, sets, and props.

3. **Lock Pre-Production Continuity Assets (Before Generating Shots)**
   - Characters: `mvd_generate_references` -> inspect candidates -> `mvd_lock_reference`.
   - Turnaround videos (optional): `mvd_generate_reference_videos` -> `mvd_lock_reference_video`.
   - Empty set plates: `mvd_generate_plates` -> `mvd_lock_plate`.
   - Hero props: `mvd_generate_props` -> `mvd_lock_prop`.
   - 3D camera-blocking previz (optional): `mvd_generate_previz` / `mvd_refine_previz` / `mvd_lock_previz`.

4. **Generate, Inspect & Review**
   - Start stills first (cheapest to iterate): `mvd_generate(project_dir, kinds=["still"], draft=True)` (or `kinds=["still", "video"]`).
   - GPU-heavy tools run asynchronously by default and return `{ "job_id": "..." }`.
   - Poll `mvd_task_status(job_id=...)` to inspect live `RunEvent` progress, `generations_used`, and final results.
   - Call `mvd_task_stop(job_id=...)` to cancel cooperatively between shots/attempts (all completed work is saved to `manifest.json`).
   - Check overall progress with `mvd_status(project_dir)` and per-check metrics with `mvd_qc_report(project_dir)`.
   - Inspect any problematic shot with `mvd_get_shot(project_dir, shot_id=...)`.
   - Use `mvd_approve(project_dir, shot_ids=[...])` or `mvd_reject(project_dir, shot_id=..., note=..., target="still" | "video")` and re-run `mvd_generate`.

5. **Upscale, Assemble & Clean**
   - Optional draft-to-final upscale: `mvd_upscale(project_dir)`.
   - Final cut: `mvd_assemble(project_dir)`.
   - Reclaim disk space: dry-run `mvd_clean(project_dir, delete=False)`, then confirm with `delete=True`.

## Operational Tips

- Pass `background=false` on any async tool only for tiny 1-shot runs; keep `background=true` for full runs so MCP client request timeouts are never hit.
- Only one writer can mutate a project at a time (`Project.run_lock()`). Check `mvd_task_status()` before launching a second write job on the same project.
- Never bypass preflight (`skip_preflight=true`) unless you intentionally want to generate without locked reference sheets or plates.
