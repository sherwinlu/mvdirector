# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project follows [Semantic Versioning](https://semver.org/). While the version is 0.x, a minor release may still change the public API; every such change is listed here.

## [Unreleased]

### Fixed
- A model reply with `framing: null` for a cutaway no longer fails validation and sends the whole batch to rule-based direction. The shot gets the framing the rule-based planner would pick, and the direction prompt says framing is never null.
- The singing-share cap keeps at least half of each singer's sung shots on camera in a duet, so one singer's verse is no longer turned into cutaways wholesale. A solo song is unaffected. The cap now runs after the visual bible, so each section can be matched to the singer it names.

## [0.3.0] - 2026-11-19

### Changed
- Behaviour change: new plans now cap the singer singing to camera at 45% of the song by default (`planning.max_lipsync_share: 0.45`); the rest of the sung shots become cutaways with the vocal playing. Set `planning.max_lipsync_share: null` to keep every sung shot on camera as before. Existing manifests are untouched until you run `mvd plan` again.

### Added
- `planning.max_lipsync_share` (default `0.45`; `null` turns it off): caps the share of the song that is the singer singing to camera. Sung shots over the cap become cutaways with the vocal playing over them, quietest non-chorus/non-hook shots first; chorus, hook and final-line shots stay, and the singer is never off screen for more than two shots in a row. Fewer faces on screen means less identity drift and fewer lip-sync renders.
- The shot-direction prompt now asks for a section arc, off-lens verse looks, at least three looks, a change every 2 to 4 seconds without a cut and a striking frame every 30 seconds. A planner-made cutaway stays a cutaway (with wardrobe cleared), and the rule-based planner writes a pace-matched no-people cutaway for it.
- Pace: each shot gets a `calm`, `moderate` or `high` label from the song's intensity, shown to the model with the change from the previous shot (including across batch boundaries), and the direction prompt explains the four levers (cut speed, movement in frame, shot scale, camera movement). The rule-based planner frames performance shots, cutaways, and singing shots by pace. `mvd plan` warns when every shot shares the same framing or when shot scale runs strongly against intensity.
- `mvd plan` logs the share of the song spent singing to camera and the number of distinct sets, and warns below three sets on visual-bible plans with at least three shots.

### Documentation
- README and docs describe the verified local setup (Ollama `gemma3:12b`, Wan 2.1 + InfiniteTalk on RunPod), advisory lip-sync and the performance fallback, duet reference limits and the InfiniteTalk and RunPod gotchas. Claude remains the built-in default LLM.

## [0.2.1] - 2026-11-18

### Fixed
- `Character.invariants` is now written into every still prompt, for each character in the shot. It was documented as carried across every prompt but was only used for reference sheets and the reference video. In a two-shot the second character gets only one reference image, and without the pinned details their hair, sleeves and hair colour drifted between stills. Each character's details are added as `Always: ...`, and nothing is added when `invariants` is empty. They are also added to the base reference sheet (regenerate it after editing `invariants`; derived variant sheets deliberately ignore them) and the vision judge now checks them. Keep `invariants` to details a costume change can't alter, since they apply in every wardrobe state: if an existing manifest pins clothing there, move it to the wardrobe states.

### Behaviour changes
- Stills and sheets you regenerate pick up `invariants`; stills and sheets already generated are untouched.
- The vision judge also checks pinned details, so a shot that passed before can fail on a missing one and be retried.
- No manifest, CLI or public API changes.

## [0.2.0] - 2026-11-18

### Added
- `video.size_multiple`: snaps generation sizes to the grid the workflow renders on, keeping the aspect ratio within the 3% quality-check tolerance. LTX needs 64; its default 640x360 draft came back as 640x320 and failed every dimension check, and retries could not fix that.
- The shipped `qwen_edit_still` bindings include `set_image` and `prop_image`.
- `mvd clean`: lists, and with `--yes` deletes, the takes that lost to a selected attempt and generations the manifest never mentions. Pending and exhausted assets keep every take; references, plates, props, audio and your inputs are never touched. Cleared attempts keep their QC record with no file path.
- `retry.lipsync_fallback` (default true): a singing shot whose video uses up its attempts becomes a performance shot (the singer on camera, not singing) and is rendered again, instead of staying stuck. The shot records `fallback_from` and is queued for review with the new `fallback` reason.
- `planning.lipsync_widest_framing` (default `medium_close`): singing shots are never planned wider than this. On `medium` shots the mouth was too small to drive, and 5 of 7 came back static with no face tracked.
- `review` reason `lipsync`: a clip that passes but scored below `qc.video.lipsync_min` goes to the review queue with its score instead of being retried.

### Changed
- Lip-sync is advisory by default (`qc.video.lipsync_blocking: false`). On sung vocals the mouth-versus-loudness correlation did not separate clips a person judged in sync from ones out of sync: 6 of 8 acceptable clips scored below the old 0.40 floor, and a clip judged slightly off scored higher than one judged fine. Set `lipsync_blocking: true` for the old reject-and-retry behaviour.

### Fixed
- A job that fails with the same error twice in a row (a missing model, no network) now stops that shot, leaving it pending, instead of spending every remaining attempt on an error a new seed cannot fix.
- A shot's audio slice (`assets/audio_slices/<shot>.wav`) was reused whenever the file existed, so a re-plan, a change to `handle_seconds`, or a replaced mix silently fed the lip-sync workflow and the lip-sync check audio from the wrong part of the song. Each slice now has a record of the start, end and source audio it was cut from (`<shot>.wav.json`) and is cut again when any of them changes. Slices from earlier versions have no record and are cut again once.
- An optional reference image a shot does not have (`set_image`, `prop_image`, `reference_video`, ...) is now cut out of the ComfyUI job. Before, its `LoadImage` kept whatever file the export named, so shots without a plate or prop sent a stale image or failed validation.

## [0.1.0] - 2026-11-17

First release.

### Pipeline
- `mvd analyze`: tempo, beat and bar grid, per-bar intensity, vocal activity windows, and lyric alignment or estimation. Dynamic intensity-driven pacing and section snapping (`video.dynamic_pacing`).
- `mvd brief`: drafts Suno-ready lyrics and a style prompt from a concept.
- `mvd plan`: deterministic bar-line shot segmentation plus an LLM-directed visual bible and shot list, with a rule-based fallback when the model is disabled. Multi-character, duet and ensemble casts, per-character wardrobe tracking, and opt-in multi-cut montage shots.
- Reference material: four-panel character reference sheets per wardrobe state and derived characters (`mvd reference generate` / `lock`, `--from-character`), turnaround reference videos, empty set plates with derived, style-seeded and embedded variants (`mvd plates`), recurring hero props (`mvd props`), and 3D camera-blocking previz (`mvd previz generate` / `lock` / `refine`).
- Generation: `mvd stills`, `mvd videos` and `mvd run` against ComfyUI through workflow templates and `*.bindings.json` files (with starter templates for LTX, Seedance, InfiniteTalk and LatentSync lip-sync), preflight asset checks, draft-resolution rendering (`--draft`), and selective post-review upscaling (`mvd upscale`).
- Quality control: image and video gates (dimensions, sharpness, duration, temporal, face presence and framing, lip-sync correlation, optional vision-model checks), a retry ladder (`new_seed`, `rewrite_prompt`, `adjust_params`), generation budgets, and `mvd qc-report` and `mvd calibrate` for inspecting and tuning thresholds. A missing `qc` extra fails closed by default (`qc.on_missing_dependency: fail`).
- Review and assembly: `light` and `strict` review policies, `mvd status` / `approve` / `reject`, a Streamlit review UI (`mvd review`), and frame-exact FFmpeg assembly with handle trimming and the original mix muxed in (`mvd assemble`).
- LLM providers: `anthropic`, `openai`, `gemini`, `ollama` and `openai_compatible` (LM Studio, vLLM, llama.cpp, OpenRouter), with `mvd llm-check` and automatic fallback of vision checks on text-only or unreachable models.

### Library
- `mvdirector.api` exposes every pipeline stage as a plain function; the CLI is a thin wrapper over it. See `docs/library.md`.
- Extension points: `StateStore` (analysis, manifest, writer lock, project id), `AssetStorage` (pull and push of the workspace) and `ComfyEndpointProvider`, with local defaults. Every `api` function accepts a prebuilt `Project`.
- One `observer` callback receives every `RunEvent`: preflight `warning`s, the `qc_summary`, and each phase, attempt, pass and exhaustion.
- Typed package (`py.typed`) with lazy top-level exports (`Project`, `Manifest`, `Analysis`, `RunEvent`, `MVDError`, and the protocols).

### Security
- Strict project-root path containment, safe-id validation at model and LLM boundaries, a ComfyUI download size cap, and streamed downloads written atomically.
- `${NAME}` expansion is limited to `MVD_*`, `COMFY_*`, `RUNPOD_TOKEN` and names listed in `MVD_ALLOWED_ENV_VARS`; `llm.api_key_env` accepts only provider key names, `MVD_*` or allowlisted names.
- Docker Compose ports and `mvd review` bind to `127.0.0.1` by default.
- Commercially permissive default face backend (`qc.face_backend: mediapipe`); InsightFace's non-commercial weights stay in an opt-in extra.

### Requirements and license
- Python 3.11 or later. `anthropic` is an optional extra. Upstream Docker tags are pinned.
- Licensed under the Mozilla Public License 2.0. Every source file carries the notice, enforced by `tests/test_license_headers.py`.
