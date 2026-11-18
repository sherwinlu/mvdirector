# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project follows [Semantic Versioning](https://semver.org/). While the version is 0.x, a minor release may still change the public API; every such change is listed here.

## [Unreleased]

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
