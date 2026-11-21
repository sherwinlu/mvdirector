# Pipeline, Assets, Quality Checks & Review

## Project Layout

```text
my-song/
  project.yaml          configuration
  inputs/               mix, stems, lyrics, style brief
  workflows/            your ComfyUI workflows and their bindings
  analysis.json         written by `mvd analyze`
  manifest.json         the shot list and full generation history
  assets/
    reference/          sheet candidates, locked <state>_sheet.png, panel crops, and turnaround videos
    plates/             set plate candidates and locked <set>.png
    props/              hero prop candidates and locked <prop>.png
    previz/             3D camera-blocking plans (*.plan.json) and rendered MP4s
    audio_slices/       per-shot vocal slices (and black-screen audio_video MP4s)
    stills/             s001_a01.png ... (shot id, attempt number)
    videos/             s001_a01.mp4 ... (and upscaled clips from `mvd upscale`)
  output/               final and draft videos
```

Every attempt is kept on disk and recorded in the manifest with its seed, prompt, parameters and QC report, so any result can be reproduced or audited. All paths resolved through `Project` and workflow bindings enforce strict containment inside the project root, and shot, wardrobe, and reference identifiers are validated against `SAFE_ID`. JSON Schemas for both state files are in `schemas/`, with filled-in examples in `examples/`.

---

## Audio-Only Preparation (`mvd prepare`)

If you only have a single `.mp3` or `.wav` file (no stem split, no lyrics file, and no style brief), install the `stems` and `align` extras (`pip install -e ".[stems,align]"`) and run:

```bash
mvd prepare /path/to/song.mp3
```

`mvd prepare` transcodes the track to `inputs/mix.wav`, separates `inputs/vocals.wav` (and `inputs/drums.wav` if `inputs.drums` is configured) with Demucs so lip-sync and vocal detection work, transcribes the isolated vocals with `faster-whisper`, and uses the configured LLM (or a deterministic fallback when `llm.enabled: false`) to structure `[Section]`-tagged `inputs/lyrics.txt` and infer a lip-sync-ready `inputs/style.md` from the lyrics and audio dynamics. Pass `--overwrite` to re-separate stems or replace non-starter `lyrics.txt` / `style.md`. Set `inputs.word_timestamps` (for example `inputs/word_timestamps.json`) before running `mvd prepare` to also save the word timings; `mvd analyze` then reuses them instead of transcribing the vocals a second time. Review the inferred `lyrics.txt` and `style.md` before `mvd plan`: they come from automatic transcription and a model's reading of the lyrics.

---

## Pre-Production Brief (`mvd brief`)

If you do not already have lyrics and a Suno style prompt, `mvd brief` uses the configured LLM to draft `inputs/lyrics.txt` (with `[Section]` tags) and `inputs/style.md` from a short concept:

```bash
mvd brief --idea "rain-soaked cyberpunk rooftop duet" --duration 180 --vibe "synthwave, melancholic"
```

Pass `--overwrite` to regenerate existing `inputs/lyrics.txt` and `inputs/style.md`.

---

## How the Model Directs the Video

`mvd plan` makes two kinds of call to the configured LLM, both returning JSON that is validated before use.

### 1. The Visual Bible (one call)

Given your style brief, lyrics, and song sections in order, the LLM returns:

| Field | What it is | Where it is used |
|---|---|---|
| `concept` | The idea tying the video together | Shown to the LLM on every shot call |
| `characters` | Permanent cast appearance (`appearance`, `invariants`, `parent_character`, `wardrobe`, `references`); solo projects use one entry (`Manifest.character` aliases the primary entry) | Added to every singer still prompt; checked by the vision judge |
| `wardrobe` | Costume states per character in order of appearance, each with an id (`cape`, `bare`) and a full description of fabric, drape and what is absent | Tracked per character (`wardrobe` + `character_wardrobes`) |
| `style`, `palette`, `lighting`, `negative_prompt` | Look of the video (consistent film stock and grade across sets) | Added to still prompts; the negative prompt goes to every generation |
| `settings` | Sets (`SetDesign`), each with a name, song sections, description (naming a mid-depth focal object), optional `parent_set`, and movement notes | Description added to each shot's still prompt; movement notes guide the LLM |
| `props` | Optional hero objects (`PropDesign`) reused across shots | Locked as `assets/props/<prop>.png` and forwarded to `still` + `VisionJudge` |
| `rules` | Short rules every shot follows (camera language, physical camera placement, continuity, effects) | Shown to the LLM on every shot call and to the vision judge |

### 2. Shot Direction (batched calls)

For each shot the LLM receives its song section, time, position in the song, type, duration, intensity, pace label (`calm`, `moderate` or `high`, plus the shift from the previous shot across batch boundaries), and lyrics, along with the concept, rules, wardrobe states, sets, and the last four shots already directed so batches stay consistent. It returns the framing, set, wardrobe state, `still_prompt`, and `motion_prompt`.

The shot-direction prompt also steers the creative arc: a section arc (restrained first verse, escalation at the first chorus, a second verse that is not a repeat of the first, a bridge that breaks the pattern, a final-chorus payoff), the four pace levers (cut speed, movement in the frame, shot scale, and camera movement), off-lens looks in verses with direct eye contact on the pre-chorus build and chorus hook, at least three looks, a visible change every 2 to 4 seconds without a camera move on singing shots, and a striking frame every 30 seconds. These steer the model rather than being hard-enforced.

### Deterministic Rules Enforced in Code

- **Cut points, voice diarization, and solo vs simultaneous duet staging come from the audio.** Shots cover the song with no gaps and land on bar lines (or vocal phrases when `video.cut_mode: phrases`). `mvd analyze` diarizes multi-singer vocal stems into solo voice clusters (`voice_1`, `voice_2`) and simultaneous harmony/unison turns (`both`) in `Analysis.vocal_turns`, splitting even zero-gap transitions (`solo <-> solo` and `solo <-> both`). When lyric tags name multiple vocalists (`[Verse 1: Elias]`, `[Chorus (Odalie)]`, `[Chorus: Both]`, `[Chorus: Elias & Odalie]`) or `vocal_turns` detects multiple voices, cuts land on vocal transitions rather than snapping across active singing, `Shot.singers` records who is actively singing (widening simultaneous duet shots to `medium_close` two-shots, instructing silent listeners in solo two-shots to keep mouths closed, and requiring two present/synced faces in `ImageQC` and `VideoQC`), and `AudioSlice.mute_spans` silences foreign vocals across handles and solo turns so singers never lip-sync each other's lines. Setting `video.dynamic_pacing: true` also scales shot length inversely with bar loudness and snaps structural sections to bar lines.
- **Empty-scene `performance` guard:** If the LLM writes an empty landscape/object `still_prompt` on an instrumental `performance` shot without setting `type: broll`, `mvd plan` automatically converts it to `broll` (when within the 2-cutaway run limit) so the singer portrait is not pasted onto an empty establishing shot.
- **Lip-sync framing:** A shot starts as `lipsync` when enough of it is sung (`video.lipsync_min_vocal_fraction`), before `planning.max_lipsync_share` thins excess sung shots into cutaways. Lip-sync shots get the vocal-stem slice and are never framed wider than `planning.lipsync_widest_framing` (`medium_close` by default).
- **How much of the song is the singer singing to camera:** `planning.max_lipsync_share` caps it, by screen time. A rough guide from music-video practice: `0.7` for a performance-driven video, `0.45` for a hybrid (the usual commercial mix) and `0.2` for a narrative one. Sung shots over the cap become cutaways: no singer, the vocal plays over an object, place or detail that fits the lyric. The quietest non-chorus/non-hook shots go first (including verses, pre-chorus, bridge or outro); chorus and hook shots and the last sung shot always stay, in a duet each singer keeps at least half of their sung shots on camera, and the singer is never off screen for more than two shots in a row (including when the LLM turns instrumental `performance` shots into `broll`). Cutaways need no face to stay consistent and no lip-sync, so they also cut render time. The default is `0.45`, the hybrid mix; set `null` to keep every sung shot on camera, or `0.7` for a performance-driven video. `mvd plan` logs the resulting share and the number of distinct sets, and warns when an LLM plan has fewer than three sets (on plans with at least three shots).
- **Pace checks and rule-based framing:** Without an LLM, performance shots and cutaways are framed wide, medium or medium-close by pace, with matching movement notes, and singing shots tighten from `medium_close` to `close_up` as pace rises while keeping a steady camera. `mvd plan` warns when every shot has the same framing or when shot scale runs strongly against the song's intensity (`r < -0.25`).
- **Set fallback:** A set name not in the bible is matched loosely, else replaced with the first set.
- **Monotonic wardrobe (`planning.wardrobe_monotonic`):** Wardrobe moves forward through each character's list by default; set to `false` if costumes change back and forth. A shot's wardrobe state is what is worn in its **first frame**.
- **Motion prompt guards:** `video.motion_single_take` (keep shots as one continuous camera move unless `planning.allow_multicut_shots` / `Shot.allow_cuts` opts into multi-cut montage), `video.motion_no_subtitles`, `video.motion_no_music`, and `video.motion_include_lyrics` append deterministic video-generation constraints to `motion_prompt`.
- **Clips use handles** (`video.handle_seconds` on each side) and are trimmed to frame-exact length at `mvd assemble`, with the original mix replacing all generated audio.
- **Stills gate videos:** No video time is spent on a still that has not passed QC or been approved by a person.

**Prompt text lives in two places:** `src/mvdirector/planning/prompts.py` and `src/mvdirector/qc/vision.py`.

---

## Reference Sheets, Turnaround Videos, Plates & Props

Stills only look like one series if every shot starts from the same approved images. `mvd stills` and `mvd run` refuse to start while any locked asset needed by the selected shots is missing (`--skip-preflight` overrides).

### Character Reference Sheets & Turnaround Videos

A reference sheet has four panels (front, profile, three-quarter, and macro face detail) on a plain studio background:

- **Layouts & background:** `reference.layout` supports `row` (4x1), `grid` (2x2), or `columns` (1x4); `reference.background` supports `white` or `grey` (neutral mid-grey avoids white-halo bleed on dark hair and clothing).
- **Invariants & derived characters:** `Character.invariants` lists non-negotiable traits (hair style and colour, signature details) carried across every costume and written into every still prompt as `Always: ...`, for each character in the shot, and into the base reference sheet (regenerate the sheet after editing; derived variant sheets ignore them). Use it for anything that drifts between stills: in a two-shot the second character gets only one reference image, so the text is what holds their look. For alternate ages, personas, or close variants of an existing character, set `Character.parent_character` or run `mvd reference generate --from-character <id>` to edit from the parent's locked sheet rather than generating from scratch.
- **Costume states:** The base sheet comes from `t2i`; later wardrobe states (`mvd reference generate --state <id>`) edit the locked base sheet (`still`) changing only the clothes, without re-describing the face. Lock a single portrait photo with `--single`.
- **Silent turnaround reference videos:** Workflows that bind `reference_video` (such as Seedance) can receive a locked 360-degree expression/turnaround clip generated via `mvd reference video` and locked with `mvd reference lock-video`.
- **Panel cropping & multi-angle identity:** `mvd reference lock` cuts the sheet into `front`, `profile`, `three_quarter`, and `face` panels. The `front` panel feeds `reference_image` (plus optional `reference_profile`, `reference_three_quarter`, `reference_face`, and `reference_image_2` on two-shots). The identity check averages `face`, `front`, and `three_quarter` embeddings, and `VisionJudge` inspects the whole sheet.

### Set Plates & Hero Props

- **Set plates (`mvd plates generate` / `mvd plates lock`):** Empty-location plates use an optical lens limitation block (single tack-sharp mid-depth focal plane, progressive optical falloff, highlight bloom) so backgrounds read like photographed locations rather than uniform CGI.
  - **Derived & embedded plates:** Use `SetDesign.parent_set` or `--from-set <set>` to derive a variant lighting/angle plate from an existing locked set. Use `--style-ref <image>` to seed a plate from an external style reference, or `--embed-set <set>` to place an already-locked set inside a screen/window/portal of a new set.
  - When `still` binds `set_image`, each singer still receives its set's locked plate as a second reference image.
- **Hero props (`mvd props generate` / `mvd props lock`):** Lock recurring objects (`assets/props/<prop>.png`); shots referencing those props forward `prop_image` / `prop_image_2` to `still` and `VisionJudge`.

### Matching Detail to Shot Size

Small details (a mark under one eye, jewelry, cuffs) do not resolve in a wide shot. The planner keeps them to close-ups, and the vision judge is told what can be judged at each framing so wide shots are not failed for invisible details.

---

## 3D Blocking Previz (`mvd previz`)

When standard I2V struggles with complex camera choreography (orbits, push-throughs, crane moves), `mvd previz` generates a deterministic Blender/procedural 3D blocking clip (`previz_video`) to guide the video model:

```bash
mvd previz generate --shot s004
mvd previz refine --shot s004 --feedback "push in faster on the last bar"
mvd previz lock assets/previz/s004_candidate_01.mp4 --shot s004
```

Each candidate persists its `.plan.json` alongside the `.mp4` so `mvd previz refine` can adjust camera keyframes iteratively before locking.

---

## Draft Previews, Upscaling & Multi-Cut Montage

- **Draft Mode (`--draft`):** Pass `--draft` to `mvd stills`, `mvd videos`, or `mvd run` to render at reduced draft resolution (with draft-aware QC) before spending full GPU/API credits.
- **Upscaling (`mvd upscale`):** Promotes approved or selected video clips through your `upscale` ComfyUI workflow (`upscaled_path` in the manifest). `mvd assemble` automatically prefers `upscaled_path` when present.
- **Multi-cut montage (`allow_cuts`):** By default, shots are single continuous takes (`motion_single_take`). Enable `planning.allow_multicut_shots: true` (or set `allow_cuts: true` on individual shots in `manifest.json`) for dynamic multi-angle montage clips; `VideoQC` relaxes temporal cut-density penalties on those shots.

---

## Quality Checks & Automatic Regeneration

A failed **hard** check rejects the attempt outright. Otherwise the attempt passes when the weighted average of its scores reaches `qc.pass_score`.

**Stills**

| Check | Hard | What it measures |
|---|---|---|
| `dimensions` | yes | Aspect ratio and minimum resolution |
| `sharpness` | no | Laplacian variance |
| `face_present` | yes | A face is detected (singer shots; MediaPipe by default) |
| `face_framing` | lip-sync or oversized: yes | Face height within `[min_face_fraction, max_face_fraction]` for the shot's framing (catches tiny mouths and oversized/squashed foreground figures in wide/medium shots) |
| `no_people` | yes | Rejects unexpected faces (`>= broll_max_face_fraction`) in `broll` cutaways |
| `identity` | yes | Face-embedding similarity to the shot's reference sheet (`qc.face_backend: insightface` only; skipped by default in favor of `vision_character`) |
| `vision_spec` | yes | Vision LLM (`vision_spec_hard_min`): framing, set, environmental scale, action, production rules, and explicit exclusions (`no minute hand`, `no people`) |
| `vision_artifacts` | yes | Vision LLM (`vision_hard_min`): hands, limbs, faces, anatomical height-to-width / head-to-body proportions, text, watermarks |
| `vision_character` | yes | Vision LLM, shown the reference sheet: same person, correct costume for the shot's wardrobe state, marks on the correct side (detail expectations scaled by framing) |

**Videos**

| Check | Hard | What it measures |
|---|---|---|
| `dimensions`, `duration` | yes | Usable file, long enough for the shot plus its handle |
| `temporal` | static or cut: yes | Frozen clips, abrupt scene changes (relaxed for `allow_cuts` montage shots), flicker |
| `face_presence` | yes | Face tracked through the clip (lip-sync shots) |
| `lipsync` | advisory | Mouth-to-vocal sync within `qc.video.lipsync_max_lag_seconds`: either MediaPipe mouth-opening vs. vocal-loudness correlation (`qc.lipsync_backend: mediapipe`, default) or ByteDance LatentSync `stable_syncnet.pt` cosine similarity + MediaPipe face cropping (`qc.lipsync_backend: stable_syncnet`); a weak score sends the shot to review (`lipsync` reason) unless `qc.video.lipsync_blocking: true` |
| `identity` | yes | Median similarity to the portrait across sampled frames |
| `vision_spec`, `vision_artifacts`, `vision_character` | as above | Vision LLM on frames sampled across the clip, including morphing |

Checks that cannot run (missing extra, wide framing, silent slice) are recorded as `skipped` with a reason, so the manifest always shows what was and was not verified. Every `mvd run`, `mvd stills`, and `mvd videos` prints which checks are active or skipped at startup.

### Retry Ladder

Applied per asset (`retry.strategies`) until one attempt passes or `retry.max_attempts` is exhausted:

1. `new_seed`: same prompt, new seed (used first on stochastic artifact/sync failures).
2. `rewrite_prompt`: the LLM rewrites the prompt from the QC failure reasons and fix hints, replacing backfiring `'no <part>'` phrasing with affirmative visual descriptions (jumped to immediately on Attempt 2 when `vision_spec`, `no_people`, or `face_framing` fails).
3. `adjust_params`: applies `retry.adjust_params` (for example more steps).

When attempts run out, the best-scoring attempt is kept and the shot is queued for review. A failed ComfyUI job counts as an attempt; an unreachable server stops the run without using any. `retry.max_generations_per_run` caps total spend per run.

### Calibrating the Quality Checks

1. Generate a batch, then label clips good or bad by eye. Approve and reject them in the review UI, or write a CSV with `shot,kind,attempt,label` rows.
2. Run `mvd calibrate` (or `mvd calibrate --labels labels.csv`). For every recorded metric it prints the cut that best separates your good and bad labels, the balanced accuracy of that cut, and the `project.yaml` key it belongs to. A balanced accuracy near `0.5` means the metric does not separate your clips; leave that threshold alone.
3. Run `mvd qc-report` (or `mvd qc-report --markdown`) to see per-check pass, fail, and skip counts with skip reasons.

By hand, compare your labels with the `metrics` recorded on each check in `manifest.json` and tune thresholds under `qc.image` and `qc.video` in `project.yaml` (all fields and defaults are in [`src/mvdirector/config.py`](../src/mvdirector/config.py)).

The lip-sync score assumes the mouth opens when the voice is loud. It is reliable on close and medium shots of clearly sung lines and weaker on mumbled, heavily processed, or very fast vocals. In a manual comparison on sung vocals (9 clips judged by eye), it did not separate in-sync from out-of-sync clips, so it is advisory by default: a weak score queues the shot for review with the score shown instead of failing and retrying it. Set `qc.video.lipsync_blocking: true` to make it a hard check. By default `qc.video.lipsync_max_lag_seconds` is `0.25` (±6 frames at 24 fps) so small workflow AV offsets do not trigger a cliff of failures; tighten it (e.g. `0.125`) if your workflow is tightly aligned, or widen it (e.g. `0.35`) if your encoder adds constant latency.

---

## Review Modes

Set `review.mode` in `project.yaml` or pass `--review-mode light|strict`. QC and retries behave identically in both; the mode only decides what reaches you.

| | `light` (default) | `strict` |
|---|---|---|
| Shots that ran out of attempts | Queued | Queued |
| Shots that passed QC | A deterministic sample is queued | Not queued |
| Assembly | Waits for decisions on queued shots | Waits only for exhausted shots |

The `light`-mode sample contains every close-up (`closeup_rate`), a share of other shots (`other_rate`), every shot that needed a retry, and every shot that passed narrowly. Rejecting a shot escalates its shot type: all other shots of that type are queued too. Switching modes never discards a decision you have made.

In `mvd review` (or with `mvd approve` / `mvd reject --note "..."`), you can approve a shot, redo its video, or redo its still and video. Your note becomes direction for the next attempt. For a still that ran out of attempts, `mvd approve --still` accepts the best one so its video can be generated.
