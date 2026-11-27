# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project follows [Semantic Versioning](https://semver.org/). While the version is 0.x, a minor release may still change the public API; every such change is listed here.

## [Unreleased]

### Added
- **Beat-locked assembly intercut (`video.assembly_intercut`):** partitions contiguous high-intensity non-lipsync video runs into temporally local 2–3 shot windows and intercuts them on musical beats (`video.assembly_intercut_min_intensity`, `video.assembly_intercut_anticipation_frames`) with exact frame conservation and zero extra GPU renders, leaving lip-sync shots, calm holds, and `CONTINUE` chains untouched.
- **Separate B-roll pacing floor (`video.broll_min_clip_seconds`):** allows high-intensity non-vocal passages to cut faster than sung shots across bar pacing, phrase DP slicing, and pacing plugins while keeping lip-sync shots at `>= video.min_clip_seconds`.
- **Entry-point plugin hooks (`plugins`):** added `mvdirector.plugins` registry supporting `mvdirector.qc_checks`, `mvdirector.audio_conditioners`, `mvdirector.pacing`, and `mvdirector.clip_selectors` by allowlisted entry-point name (never raw import paths), plus `Attempt.optimal_in_offset`.
- **Parallel & hardware-accelerated FFmpeg assembly:** added `video.assembly_workers` (1–8, default 1) and `video.encoder` (`libx264`, `auto`, `h264_nvenc`, `h264_videotoolbox`) with cached 1-frame encoder probing (`media.available_encoders()`).
- **Workflow & cache hooks:** added frame-bucket batch sorting (`_sort_for_cache`), Monge-Kantorovich optimal-transport color matching (`qc.color_drift.fit_optimal_transport_color_match`), optional `video.i2v_fps` for half-FPS non-lipsync I2V generation, and `comfy.nag_enabled` for NAG-gated workflows when `negative_prompt` is non-empty.
- **Vision identity check:** with a face backend that has no identity embeddings (the default MediaPipe), the `identity` check no longer always skips. The vision model compares a crop of the singer's face with the reference sheet's face and three-quarter panels, judging facial structure only, not costume, hair or lighting. Faces under `qc.image.vision_identity_min_face_px` (default 100) for stills or `qc.video.vision_identity_min_face_px` (default 40 on `analysis_width`-scaled frames) for clips are skipped as too small to judge; duets score their best-matching face; clips use the median over `qc.video.vision_identity_frames` sampled frames. Hard-fails below `vision_identity_hard_min` (0.50). One extra model call per still or clip with a large enough face; turn off with `qc.vision_identity: false`. InsightFace embedding identity is unchanged.
- **Director prompt craft:** planner prompts now guide subtext over literal nouns, diffusion-feasible single-action shots, eye-trace continuity across short cuts, section color/light progression, candid off-center close-up framing (anti-mugshot), and B-roll prop diversity.

### Changed
- **Default `video.min_clip_seconds` raised from `1.5` to `2.5`:** keeps sung lip-sync shots readable by default. Existing configs that set `video.max_clip_seconds < 2.5` without an explicit `min_clip_seconds` automatically clamp the default to `max_clip_seconds`; numeric `VideoConfig` fields now also reject YAML booleans instead of coercing them.

### Fixed
- **Vocal-bleed lip-sync guard & non-singing gap muting:** `mvd analyze` now refines Demucs `vocal_windows` with Whisper speech clusters (including in `lipsync` mode without `lyrics.txt`), the segmenter suppresses false `lipsync` classification on instrumental bleed outside aligned lyrics, and `_populate_slice_mutes` mutes non-singing lead-in, tail, and internal gaps inside lip-sync slices.
- **Short xfade concat seam preservation:** `mvd assemble` falls back to a clean hard cut whenever available tail frames clamp an `xfade` overlap below 3 frames, preventing `libx264` B-frame / non-B-frame MP4 concat seam duplication.
- **Fewer close-up still retries:** every close-up still of a singer now starts with a concrete size cue ("the face fills at least a third of the frame height, framed from just above the head to the collarbone"), even when the planner's prompt already says "close-up". Before, the cue was skipped whenever the prompt contained "close-up", and image models read a bare close-up as chest-up: in a 92-shot project 17 of 26 close-ups failed `face_framing` (median face 26% vs the 30% minimum) and needed a retry. The planner's shot instructions now define close_up by size and rule out "upper chest up".
- **Near-static lip-sync clips are no longer retried:** in a singing close-up the mouth is most of the motion, so clips with strong lip-sync (correlation 0.8) failed `temporal` as "almost static" by a hair. When the lip-sync check passes, the static result is accepted (non-hard, unscored, `static_waived` in its metrics); a frozen clip still fails lip-sync.
- **Passing lip-sync QC score floor:** a lip-sync check that passes now scores at least `qc.pass_score` so its heavy weight cannot drag an otherwise-passing clip into a false retry fallback.

## [0.8.0b3] - 2026-11-25

Third **Beta** release, adding global exclusions, an explicit still-repair API, multi-angle set plates, multi-photo character seeding, a draft grid, and a sketch guide, plus b-roll cast matching and profile/hands reference routing fixes.

### Added
- **Motion-prompt hygiene & lived-in detail:** `SHOTS_SYSTEM` now keeps motion prompts to movement, acting and camera only; `BIBLE_SYSTEM`, `PLATE_TEMPLATE` and `PROP_TEMPLATE` add lived-in patina and wear cues.
- **Global exclusions:** a `do_not_show` brief/config field propagates into `visual_bible.rules` and `visual_bible.negative_prompt`.
- **Still repair API:** `api.repair_still()` and `api.regenerate_still(..., strategy="repair"|"recreate")` choose between editing a still in place and recreating it.
- **Multi-angle set plates, multi-photo character seeding, draft grid and sketch guide:** seed photos are EXIF-orientation corrected and uploaded to ComfyUI as re-encoded, metadata-free copies; draft grid sheets clear stale review decisions when they replace a still or video; `lock_plate_turnaround` accepts `str` paths.

### Fixed
- B-roll cast guards match epithet names (`Corbin` matches `Corbin the Host`) without matching plain first names against common words.
- Side-profile `PERFORMANCE` shots route to the profile panel, and hands-only close-up inserts skip `face.png` and MediaPipe face checks; wider shots that mention hands keep full face QC.
- Draft close-ups no longer hard-fail `min_face_fraction` by one pixel of bounding-box quantization.
- Rear-view detection matches plural phrasing (`backs turned`, `backs to camera`, `backs of heads`).
- Six new-feature tests run without native MediaPipe so Linux CI passes.

## [0.8.0b2] - 2026-11-25

Second **Beta** release, packing parallel wave execution, multi-reference R2V video & latent-refine upscale support, dual-panel character consistency, guided onboarding with `0600` project `.env` storage, and three rounds of pilot and QA/security hardening.

### Added
- **Multi-Reference (R2V) Video Workflows & Two-Stage Latent-Refine Upscaling:**
  - `ShotRunner` now uploads optional character and set plate references (`reference_image`, `reference_face`, `reference_profile`, `reference_three_quarter`, `reference_image_2`, …, `set_image`) on `lipsync` / `i2v` workflows when bound to distinct targets from `image`, annotates composed motion prompts in both `tags` (`@image2+`, after starting still `@image1`) and `prose` notation, and automatically prunes unbound optional `LoadImage` / `LoadAudio` nodes per shot.
  - `mvd upscale` (`upscale_videos`) now forwards `negative_prompt` plus optional `audio` and `image` uploads for two-stage 3D latent upscaler + 1-step tiled refine graphs (with automatic pruning when unbound).
  - Added starter placeholder bindings `r2v_audiolock_lipsync.bindings.json` and `latent_refine_upscale.bindings.json`, plus updated `docs/workflows.md` and `BIBLE_SYSTEM` defaults encouraging ordinary, relatable real-world people and physically shootable locations unless the style brief asks otherwise.
- **Parallel Shot Execution (`comfy.max_concurrency`, `--concurrency`):**
  - Parallelized `ShotRunner` dependency waves with per-worker client leasing (sharing the primary client for static endpoints so VRAM floor checks and reference uploads stay synchronized), thread-safe single-flight upload deduplication, atomic budget accounting, and cooperative cancellation slot refunds.
- **Multi-Member & Dual-Panel Character Consistency:**
  - Generalized `reference_image_N` bindings for bands/ensembles (`reference_image_2`, `reference_image_3`, …) with explicit `Picture N` prompt disambiguation.
  - Solo shots on still workflows binding `reference_image_2` (without `reference_face`) now automatically route the singer's close-up `face` panel into `reference_image_2` (skipped on rear-view shots), paired with reference-sheet garment coverage checks in `VisionJudge`.
- **Guided `mvd setup` Onboarding, Live Sanity Checks & Project-Scoped `.env`:**
  - Interactive `mvd setup` now supports provider-first LLM setup (accepting aliases like `chatgpt` and `claude`): when `ollama` is chosen and not running locally, it offers to install Ollama (via Homebrew on macOS or winget on Windows), start `ollama serve`, and pull the chosen model; when a cloud LLM is chosen, it skips Ollama, prompts (masked) for the API key if unset, and offers to save secrets to an owner-only (`0600`), gitignored `<project>/.env` file (or keep them shell-only).
  - Project-scoped `.env` loading (`mvdirector.dotenv`) reads only allowlisted LLM/ComfyUI secret keys with `O_NOFOLLOW` symlink checks and a 64 KiB cap (enforced on both read and write) without mutating `os.environ`, and shell exports always take precedence.
  - Added `LLMQuotaError` (`LLMUnavailableError` subclass) to fail fast without wasted retries when a provider reports billing/quota exhaustion (`insufficient_quota`, `credit balance is too low`, HTTP `402`, etc.) and trip the vision QC circuit breaker cleanly.
  - Interactive `mvd setup` (and `--non-interactive --verify`) runs a live LLM JSON + vision sanity check (distinguishing invalid keys, quota limits, unknown models, and text-only models with interactive retry/switch recovery), a ComfyUI `/object_info` + workflow wiring check (`api.check_comfy_wiring`), and auto-optimization (`optimize_setup_options`) that preserves hand-tuned keys on re-run.
- **ComfyUI Live Progress Streaming & 1-Click Quality Knobs:**
  - Added best-effort RFC 6455 WebSocket step progress streaming (`_ComfyWsProgressListener`, TLS 1.2 minimum, HTTP `/queue` fallback) in `ComfyClient` and opted-in `comfy.api_key_comfy_org` forwarding.
  - Added `llm.stream` (SSE streaming for `openai` / `openai_compatible` backends to avoid proxy idle timeouts), `planning.vocals: solo` (locking singing shots to the protagonist), and `reference.closeup_face_panel` (preferring the macro face panel on close-ups for single-reference workflows), plus reference-image priority, video-scope calibration, and excess-flicker video QC scoring.

### Fixed
- **Audio, Lyrics, Planning & QC Hardening (Pilot Audit):**
  - Hardened vocal diarization against splitting same-register solo vocals, surfaced acoustic vocal registers into casting cues, detected master mixes inside stem directories, fixed `mm:ss` rounding (`2:60`), stripped Suno production/inline delivery cues without corrupting section headers or instrument range words (`tenor sax`), and normalized romanization guides before language detection.
  - Sent `think=False` to Ollama `/api/chat` with automatic fallback, added single-retry Gemini `thinkingBudget=1024` recovery when reasoning exhausts `MAX_TOKENS`, allowed nullable `inputs.vocals`/`inputs.lyrics`, coerced YAML `json_mode: off` (`False`), and hardened bare JSON list extraction (`_try_bare_list_for_schema`).
  - Improved rear-view panel routing (`back` -> `profile` -> `front`), tight framing prose reconciliation, silhouette/non-human performance face-check handling, and raised b-roll MediaPipe face confidence to `0.80` to avoid object pareidolia.
- **QA & Security Hardening (Rounds 2 & 3):**
  - Validated `assemble` output paths before rendering (video suffix, non-directory, refusing to clobber project state/clips outside `output/`), capped `--width`/`--height` overrides at 8192, limited `clean --delete` strictly to generation files under `assets/`, preserved existing files on `mvd init` (and rejected broken symlinks), capped/sanitized review notes (2,000 chars, no NUL bytes), and failed fast on unknown shot IDs.
  - Hardened MCP HTTP with a single-watchdog total read deadline, header-stage Host/Origin/Bearer precheck (case-insensitive `Bearer`), and body draining on rejection.
  - Fixed O(N²) SSE line splitting and bounded non-200 error body reads (256 KiB) in `HttpJSON.post_sse`; guarded `redact_url`/`origin` on malformed IPv6 URLs; enforced `http`/`https` base URL validation in `ComfyClient` and rejected query/fragment strings in `setup.validate_http_url`.
  - Fixed `LLMConfig` `environ={}` host-env isolation, rejected control characters in LLM API keys, rejected symlinked `project.yaml` in `setup`/`tuning`, loaded project `.env` during `tuning.apply_adjustments_to_yaml` validation, caught traversal workflow paths in `check_comfy_wiring`, ran preflight on video-only runs (`mvd videos`), and caught `RecursionError` on deeply nested JSON/YAML across LLM, ComfyUI, Ollama probes, setup, tuning, lock, and stamp parsers.

## [0.8.0b1] - 2026-11-22

First public **Beta** release, verified end-to-end both locally and with ComfyUI running on a remote host.

### Added
- **Guided Setup & Local/Remote Onboarding (`mvd setup`, `docs/local-quickstart.md`):**
  - Added interactive and `--non-interactive` `mvd setup` (plus `--check-only` diagnostics) probing `ffmpeg`/`ffprobe`, installed extras, Ollama (local or remote), and ComfyUI (local, home LAN/Tailscale computer, or authenticated cloud HTTPS server via `--comfy-token-env`), with model-family video presets (`ltx`, `wan`, `wan-i2v`, `exact`, `custom`).
  - Added a step-by-step [Local Quickstart Guide](docs/local-quickstart.md) covering fully local, home-LAN, and cloud ComfyUI setups.
- **MCP Server & Agent Integration (`mvdirector.mcp`, `mvd mcp`):**
  - Zero-dependency Model Context Protocol server exposing pipeline stages and async background jobs over `stdio` and Streamable HTTP (`--http`) with bearer-token auth, loopback/Origin/Host guards, workspace containment, and the bundled `mvdirector://skill` workflow guide.
- **Structural Cues & Mix-Mode Upgrades (`mvdirector.audio.cues`):**
  - Added structural energy landmarks, WeaveShift tracklist/CSV ingestion, and DSP polish across analysis and Long-Form Mix Mode.
- **Audio & Lip-Sync Precision:**
  - `video.audio_vae_offset_seconds` (default `0.0`) and `video.audio_mirror_pad` (default `false`) for rational VAE front-crop audio shift alignment (including lead-silence padding for negative offsets at `start == 0`) and periodic triangle-wave mirror tail padding without silence cliffs.
  - `video.sync_delay_compensation` (default `false`: opt-in, because the default MediaPipe lag estimate is noisy on sung vocals), `qc.video.lipsync_min_confidence`, `qc.video.desync_face_fraction_min`, and `Attempt.sync_delay_seconds` to apply measured per-take lip-sync lag compensation (clamped to audio handles) during assembly and compute viewer-visible `desync_risk`.
  - Pre-spend lyric alignment diagnostics (`mvdirector.audio.alignment_qa`, `Analysis.alignment_warnings`) flagging unvoiced/noisy vocal spans, consonant-heavy spans, and lyric words cut across bar lines.
  - Hardened Whisper transcription (`condition_on_previous_text=False`, optional `initial_prompt` vocabulary seeding, a stall-timeout watchdog (`prepare.whisper_stall_timeout_seconds`, default 600 s and three times that for the first segment; `0` disables it), and collapsing of 4+ consecutive repeats only when their word timings show a decoding loop, so sung repeats such as "na na na na" keep every timestamp).
  - Global shortest-path phrase-aware DP slicing (`video.phrase_dp_slicing`, default `false`) and local vocal-onset cut shifting (`video.prefer_vocal_onset`, default `false`).
  - Streaming ITU-R BS.1770-4 LUFS loudness, 4× True Peak, and splice-click QA (`mvdirector.audio.broadcast_qa`, `Analysis.broadcast_qa`, `mvd audio-qc`), automatically run during `mvd analyze`, `mvd assemble`, and mix assembly with bounded memory for multi-hour mixes. Audio containing NaN/Inf samples is reported as an error (it would otherwise read as silence), and a silent mix is flagged.
  - Automatic diegetic SFX / foley active-RMS leveling and sample-accurate song ducking (a NumPy gain envelope, so fades are click-free and any number of cues works) (`mvdirector.audio.foley`, `Shot.sfx_path` in any ffmpeg-decodable format, `Shot.sfx_gain_db`, `Shot.sfx_duck_db`) blended automatically during `mvd assemble`.
- **Visual Continuity, Prompting & Quality Gates:**
  - Shot continuity chaining (`Shot.continuity`: `cut`, `reframe`, `continue`) with dependency-wave scheduling in `ShotRunner` (batching independent `CUT` shots while running chained depths in order with `free_memory()` between phases), explicit `RetryStrategy.HANDOFF` tracking with zeroed head-handle alignment (each handoff still is its own attempt file; a handoff that was sent back is not repeated from the same clip; a `continue` still waits for the previous clip, also across separate `mvd stills` and `mvd videos` runs), and optional `previous_frame` / `end_image` / `camera_path_zoom` workflow bindings.
  - Ridge-regularized 3×4 RGB affine VAE color-drift cancellation across `CONTINUE` joins (`video.color_drift_correction`, default `false`) with letterbox-free active-region fitting and FFmpeg `colorchannelmixer` + `lutrgb` emission.
  - Automatic advisory `join_continuity` (phase-correlation velocity jump / duplicate frame / speed dip) and `beat_motion` (Gaussian beat-pulse cross-correlation with permutation z-score) checks in `VideoQC` (`mvdirector.qc.continuity`).
  - Targeted in-place still repair (`retry.in_place_repair`, `RetryStrategy.REPAIR_STILL`) gated on templates accepting `reference_image`, with `Attempt.regressed` exclusion from `best_attempt()` when repairs fail to reduce QC issues.
  - Smart Cutaway Rescue (`retry.smart_cutaway_rescue`, `retry.smart_cutaway_max_correlation`) converting weak lip-sync takes into atmospheric b-roll cutaways within the 2-cutaway run limit, with full exception-safe rollback.
  - Two-pass blind-observer / split-auditor VLM judging (`qc.vision_blind_observer`, default `false`) forwarding primary and extra references without confirmation bias.
  - Monotonic World-State Ledger (`VisualBible.world_facts`, `WorldFact`, `mvdirector.planning.world_state`) filtering contradicted state adjectives from scene looks across shots, plus rear-view facial detail scrubbing and multi-reference `Picture N` disambiguation in `prompts.py`.
  - Linear view-space depth previz pass (`Shot.previz_depth_video`) and a silent AAC audio track added to cached upload copies of previz and reference videos for ComfyUI video-loader compatibility (`media.with_audio_track`; the source file is never rewritten, and codecs MP4 cannot carry are re-encoded). Locking an external previz with `lock_previz` drops a depth pass rendered for the earlier auto-previz.
- **Assembly, Mix Mode & ComfyUI Infrastructure:**
  - RGB-space (`gbrp`) cinematic color grade presets (`video.color_grade`, `video.film_grain`), beat-preserving `xfade` transitions (`video.transition`, `video.transition_seconds`, capped to real tail handles, disabled across `CONTINUE` seams, and rendered one crossfade at a time so memory does not grow with the shot count), and metadata stripping (`video.strip_metadata`).
  - Smoothstep seamless loop cross-dissolve option (`mix.loop_blend`: `xfade` | `smoothstep`) for long-form mix chapters.
  - Opt-in content-addressed ComfyUI render cache (`comfy.cache_enabled`, default `false`, stored under `<project>/.comfy_cache`, purged by `mvd clean`), content-addressed upload deduplication (`comfy.dedup_uploads`), batch cache sorting (`comfy.sort_batch_for_cache`), VRAM floor probing (`comfy.min_free_vram_gb`, checked once per phase after `/free` has had time to release memory; when it cannot be met the run stops with `ComfyResourceError` instead of using up shot attempts), prompt length budgeting (`comfy.max_prompt_chars`), ComfyUI UI-to-API workflow conversion (`convert_ui_workflow`), and live `/object_info` pre-flight dependency checks in `mvd validate`.

### Fixed
- **QA, Security & ComfyUI Version-Agnostic Hardening:**
  - Hardened `ComfyClient` and `convert_ui_workflow` across ComfyUI versions: sets a default `mvdirector/<version>` `User-Agent` (respecting user overrides), handles V1 and V3 `/object_info` combo schemas, safely parses varied `/history` status/output/error shapes, supports both list and dict `links` and `control_after_generate` metadata in UI exports, and catches HTTP/auth errors cleanly during `probe_comfy`.
  - Comprehensive white/black-box QA and security hardening across path/symlink containment, env-var allowlists, MCP HTTP transport, atomic writes, FFmpeg subprocess boundaries, and numeric edge cases.

## [0.7.0] - 2026-11-22

### Added
- **Suno & DAW multi-stem pack ingestion (`mvdirector.audio.stem_pack`, `inputs.stems_dir`):**
  - `mvd prepare` now accepts one or more `.zip` archives, folders, or loose `.wav`/`.mp3`/`.flac` stem files (or automatically discovers stem packs dropped into `inputs/stems/` or `inputs/*.zip`), supporting Suno Auto Split, Split from Mix, and Premier Advanced Split.
  - Classifies stems into canonical musical roles (`lead_vocals`, `backing_vocals`, `vocals`, `drums`, `percussion`, `bass`, `guitar`, `keys`, `strings`, `synth`, `brass`, `woodwinds`, `instrumental`, `fx`, `other`), routing clean `lead_vocals` (excluding `backing_vocals`) to `inputs/vocals.wav` and skipping Demucs, using `drums`+`percussion` for beat tracking, using harmonic stems for cleaner key/mode estimation, and synthesizing a peak-normalized `inputs/mix.wav` when no master mix is provided.
  - Hardened archive extraction rejects zip-slip traversal, symlinks, null-byte paths, entry-flood DoS, and zip bombs with bounded streaming size limits and atomic writes.
  - Persists `Analysis.stems` (`StemArrangement` + `StemTrackInfo`, updated `schemas/analysis.schema.json`) and feeds instrumentation/arrangement summaries and per-shot active stems (`stems=[...]`) into the Visual Bible and shot-direction prompts.
- **Non-lipsync standard-mode video styles (`planning.video_style`, `mvd plan --style`):**
  - Added `VideoStyle` (`lipsync` default, `narrative` for recurring characters without lip-sync, and `visual` for 100% characterless b-roll) with dedicated narrative/visual Bible and shot-direction prompts.
  - `mvd analyze` and `mvd plan` no longer require `inputs.vocals` or `inputs.lyrics` for non-lipsync styles; `tuning` skips lip-sync knobs; `generate` and `validate_project` only require the ComfyUI workflows needed by the selected style.

### Fixed
- `ChatBackend.complete_json` now falls back to `json.loads(..., strict=False)` when open-weights models emit literal newlines inside multiline JSON string values (such as lyrics).

## [0.6.0] - 2026-11-21

### Added
- **Song interpretation & acoustic musical-style analysis:**
  - `mvdirector.audio.analysis.analyze_musical_features` extracts musical key/mode, spectral brightness, harmonic vs. percussive texture, dynamic range, and onset density into `Analysis.musical_features` (`MusicalFeatures`), enriching `mvd prepare`, `mvd analyze`, and `mvd plan`.
  - `VisualBible.interpretation` (`SongInterpretation` + `SectionBeat`) records the director's reading of the song's meaning, tone, musical feel, recurring visual motifs, and per-section emotional/narrative/camera arc (`VisualBible.find_section_beat`), threading section beats and motifs into every shot-direction batch.
  - Optional multimodal audio attachment (`audio_block`, `maybe_audio_block`, `complete_json_with_optional_audio`, and Gemini `inlineData` audio support): audio-capable backends receive a compact mono MP3 of `inputs/mix.wav` during `mvd prepare` and `mvd plan`, falling back automatically to text-only prompts with `musical_features` on non-audio providers.
- **Per-chapter lyrics excerpts in Long-Form Mix Mode:**
  - Tracklist entries (`TrackEntryInput`, `ResolvedTrack`, `AnalyzedChapter`) now accept an optional `lyrics` field (aliases `lyric_excerpt`, `lyrics_excerpt`); `mix_planner` sanitizes and forwards excerpts into chapter theme planning so generated environments and loop prompts draw concrete imagery from each track's lyrics.

### Changed
- Removed RunPod-specific references and generalized cloud GPU setup (`docker/Dockerfile.runpod` → `docker/Dockerfile.allinone`, `docker/runpod-entrypoint.sh` → `docker/allinone-entrypoint.sh`, plus updated docs, config comments, and issue templates).
- Updated `schemas/analysis.schema.json` and `schemas/manifest.schema.json` with `MusicalFeatures`, `SongInterpretation`, `SectionBeat`, and `AnalyzedChapter.lyrics`.

### Fixed
- `OllamaChat` now falls back to prompt-based JSON extraction on HTTP `501` structured-output rejections (in addition to HTTP `400`).

## [0.5.0] - 2026-11-20

### Added
- **Track profiling & automatic parameter tuning (`mvd tune`, `mvd analyze --tune`, `auto_tune: true`):**
  - `mvdirector.tuning` (`profile_track`, `recommend_adjustments`, `tune_prepare_config`, `tune_config_for_planning`, `apply_adjustments_to_yaml`) and `api.tune` / `TuneResult` inspect `inputs/lyrics.txt`, `inputs/style.md` (stripping Suno negative `-tag` exclusions), audio dynamics, vocal turns, and aligned syllable rates to populate `Analysis.track_profile` (`TrackProfile`).
  - Automatically detects languages & code-switching, vocal mode (`solo`, `duet`, `group`, `instrumental`), rap/fast-flow density, EDM drops/vocal chops, extreme/processed vocals, and non-4/4 or free-time meter cues, tuning default `prepare.whisper_*`, `video.beats_per_bar`, `video.cut_mode`, `video.dynamic_pacing`, `video.lipsync_min_vocal_fraction`, and `planning.max_lipsync_share` values at runtime while preserving user-customized settings unless `--force` is passed.
  - Comment-preserving in-place `project.yaml` updates via `mvd tune --apply` and `mvd analyze --tune`.
- **Long-Form Mix Mode (`mode: mix`, `mvd init --mode mix`):**
  - `ProjectMode.MIX`, `inputs.tracklist` / `--tracklist` ingestion (`mvdirector.audio.tracklist`) supporting JSON/YAML (including WeaveShift setlist exports), `.cue` sheets, plain-text timestamp lists (`00:00 Artist - Title`), or fixed-interval `mix.default_segment_seconds` chapters snapped to low-energy mix dips.
  - Windowed, bounded-memory per-chapter audio analysis (`mvdirector.audio.mix_analysis` -> `Analysis.chapters` / `AnalyzedChapter`) measuring per-chapter BPM, downbeat phase, and normalized energy across 30-to-90+ minute mixes without requiring vocal stems or lyrics.
  - Per-chapter visual theme planning (`mvdirector.planning.mix_planner` -> `Manifest.chapters` / `MixChapter`) generating 1 uninhabited visual environment and `mix.clips_per_segment` (2–3) loopable `broll` units per chapter (`mix.visual_source: clips` for `t2i`+`i2v` loops or `stills` for `t2i` hero stills with camera drift).
  - Seamless circular loop synthesis, BPM/downbeat-synced visual pulse (`mix.beat_pulse`), Pillow-rendered RGBA track-title/watermark/spectrum HUD cards (`mix.hud`), incremental per-chapter MP4 segment caching (`assets/segments/<chapter_id>.mp4` + `.mp4.json` input stamps, cleaned by `mvd clean`), FFmpeg stream-copy muxing with embedded MP4 chapter markers, and YouTube-ready `<output>.chapters.txt` sidecars (`mvdirector.mix_assemble`).
  - `--chapter` / `-C` targeting across `mvd stills`, `mvd videos`, `mvd run`, `mvd approve`, and `mvd assemble` (plus `--rebuild-cache`), and chapter-grouped review in `mvd review`.

### Changed
- Updated `schemas/analysis.schema.json` and `schemas/manifest.schema.json` with `TrackProfile`, `AnalyzedChapter`, `ProjectMode`, `MixChapter`, and `Shot.chapter_id`.
- `ProjectConfig.auto_tune` defaults to `true`, applying track-appropriate parameter defaults automatically during `mvd prepare`, `mvd analyze`, and `mvd plan` without overriding custom user settings.

## [0.4.0] - 2026-11-20

### Added
- `mvd prepare [AUDIO]` (and `api.prepare`): start from a single `.mp3` or `.wav` file. It writes `inputs/mix.wav`, separates `inputs/vocals.wav` (and `inputs/drums.wav` when `inputs.drums` is set) with Demucs, transcribes the vocals with faster-whisper, and infers `[Section]`-tagged `inputs/lyrics.txt` and a lip-sync-ready `inputs/style.md` from the lyrics and the audio's tempo and energy (an LLM when enabled, otherwise a rule-based fallback). It refuses to replace existing files without `--overwrite`.
- `prepare:` config section (`stem_model`, `stem_device`, `whisper_model`, `whisper_language`, `whisper_multilingual`) and the `stems` extra (`pip install -e ".[stems,align]"`).
- `Transcriber` / `WhisperTranscriber` and `StemSeparator` / `DemucsSeparator` extension points.
- Neural lip-sync scoring via `qc.lipsync_backend: stable_syncnet` (ByteDance LatentSync `stable_syncnet.pt` + MediaPipe Face Detector, `syncnet` extra).
- Solo vs. simultaneous duet diarization (`Analysis.vocal_turns`, `Shot.singers`, `AudioSlice.mute_spans`) with vocal-handoff cut placement, cosine-faded foreign-voice muting (`media.mute_audio_spans`), and two-face simultaneous duet QC.
- Image proportion, architectural scale, and negative-exclusion QC (`ImageQCConfig.max_face_fraction`, `broll_max_face_fraction` / `no_people`, hard `vision_spec` check, `extract_negative_exclusions`, and automatic prompt-rewrite escalation).
- `--draft`, `--width`, and `--height` flags on `mvd assemble`, and `--width` / `--height` on `mvd upscale`.
- Multilingual and non-English song support: Unicode/CJK/Hangul/Indic/Thai/Arabic lyric tokenization and alignment, per-segment `prepare.whisper_multilingual`, multilingual `[Section]` / duet / hook recognition with UTF-8 BOM and RTL bidi stripping, English visual prompt directives, and collision-safe non-ASCII IDs, set plate slugs, and output filenames.

### Fixed
- Tempo detection folds half-time and double-time readings into 70-160 BPM (a slow blues no longer reads as 185 BPM), keeping the beat grid consistent. This changes `analysis.json` bars for songs whose tempo was outside that range; re-run `mvd analyze` and `mvd plan --overwrite` to pick it up.
- `mvd prepare` falls back to the transcript's lyrics when the model's lyrics match less than 85% of the transcribed words, so a rewritten song cannot break word alignment. The inference prompt asks for plain-prose style fields and no BPM figure.

### Changed
- `llm.timeout_seconds` now defaults to 600 (was 180) so local models on a busy GPU do not time out.
- Behaviour change: the language model now runs locally by default (`llm.provider: ollama`, `llm.model: gemma3:12b`), so no API key is needed. To keep using Claude, set `llm.provider: anthropic` and name a model. A config that sets `provider: anthropic` without a `model` is now rejected, like the other cloud providers. Projects from the earlier template name both `provider` and `model` and are unaffected; a `project.yaml` that left `provider` out relied on the old default and must add `provider: anthropic`.
- The Whisper model size, language, and multilingual mode used by `mvd analyze` now follow `prepare.whisper_model`, `prepare.whisper_language`, and `prepare.whisper_multilingual` (defaults unchanged: `small`, auto-detect, single-language).

## [0.3.1] - 2026-11-19

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
