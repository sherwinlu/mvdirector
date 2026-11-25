# Workflows

All image and video generation in `mvdirector` runs through **ComfyUI** — locally, on another
computer in your home LAN/Tailscale, or on a remote cloud server. This folder connects the
pipeline to **your** ComfyUI workflows—whether they run open-weight GPU models (Qwen, Z-Image,
Krea, Flux, LTX, Wan, HunyuanVideo, MOSS, etc.) or **paid cloud models** via ComfyUI Partner/API
nodes (Nano Banana, Seedance, Grok, P-H3 / MiniMax, Kling, FAL.ai, Replicate). Because graphs
are exported directly from your own ComfyUI server and `mvdirector` relies only on ComfyUI's
core HTTP API (`/system_stats`, `/upload/image`, `/prompt`, `/history`, `/view`) with graceful
fallbacks for optional endpoints, workflows stay portable across ComfyUI versions.

Each workflow needs two files:

| File | What it is | Where it comes from |
|---|---|---|
| `<name>.api.json` | The workflow graph | You export it from ComfyUI |
| `<name>.bindings.json` | Which node inputs the pipeline may set | Shipped here as an example; you edit the node ids |

## Setting up or swapping a workflow

1. Build the workflow in ComfyUI and confirm it works by hand with a test image, prompt
   and (for lip-sync) audio clip.
2. Export it with **Save (API Format)** (enable dev mode in ComfyUI settings if the option
   is hidden) and save it here under the name in the bindings file's `workflow` field.
3. Open the exported JSON. Each top-level key is a node id. For every entry under `inputs`
   in the bindings file, set `node` to the id of the node that takes that value and `field`
   to the input's name on that node.
4. Set `output.node` to the id of the node that saves the image or video (`SaveImage`,
   `SaveVideo`, `VHS_VideoCombine`, etc.).
5. Run `mvd validate`. It reports any binding that points at a missing node or input, and lists
   the available input fields on that node if you mistyped a field name.

## The four workflows

| Name | Used for | Parameters the pipeline sends |
|---|---|---|
| `t2i` | Base reference sheet, set plate candidates, b-roll stills | `positive_prompt`, `negative_prompt`, `seed`, `width`, `height` |
| `still` | Singer stills, and reference sheets for later costumes | the above plus `reference_image` (omit `reference_image` for T2I-only / LoRA models such as Z-Image or Krea T2I) |
| `lipsync` | Lip-sync clips | `positive_prompt`, `negative_prompt`, `lyrics`, `seed`, `image`, `audio`, `audio_video`, `reference_video`, `previz_video`, optional R2V refs (`reference_image`, `reference_face`, `reference_profile`, `reference_three_quarter`, `reference_image_2`, …, `set_image`, `prop_image`), `width`, `height`, `num_frames`, `duration`, `fps` |
| `i2v` | All other clips | as `lipsync`, without `audio` / `audio_video` (unless listed in `required`) |
| `upscale` (optional) | Selective post-review clip upscaling (`mvd upscale`) | `video`, `width`, `height`, `fps`, `seed`, `positive_prompt`, `negative_prompt`, optional `audio`, optional `image` |

**Dedicated singing lip-sync & latent-refine templates shipped here:** LTX audio-conditioned A2V can drift on fast/sung syllables. Starter bindings for tighter mouth sync and 1080p tiled latent refinement are included alongside the defaults:
- `infinitetalk_lipsync.bindings.json` (Wan 2.1 + InfiniteTalk Single, Apache-2.0): starter placeholder bindings (replace example node ids with ids from your exported workflow); set `comfy.workflows.lipsync` to this file and `video.frame_multiple: 4`, `video.frame_offset: 1` in `project.yaml`.
- `latentsync_lipsync.bindings.json` (I2V + LatentSync mouth-dub pass): starter placeholder bindings (replace example node ids with ids from your exported workflow); keep your I2V motion (LTX or Wan) and pass the video + vocal slice through LatentSync (weights are OpenRAIL++ — review use restrictions before commercial use).
- `seedance_lipsync.bindings.json` (Seedance Reference-to-Video via `audio_video`): starter placeholder bindings using `audio_video` (an auto-generated black-screen MP4 containing the vocal slice, which improves lip-sync on video-reference nodes) and `comfy.reference_notation: tags` (`@image1`, `@video1`) with `video.motion_include_lyrics: true`.
- `r2v_audiolock_lipsync.bindings.json` (Multi-Reference R2V + Audio-Lock lip-sync): starter placeholder bindings wiring `image`, isolated vocal `audio`, and optional `reference_image` / `reference_image_2` / `set_image` slots for R2V models (Wan VACE/Phantom, LTX-2.x AV Audio-Lock, or commercially licensed MiniMax H3).
- `latent_refine_upscale.bindings.json` (Two-Stage Latent Upscale + 1-Step Tiled Refine): starter placeholder bindings for `mvd upscale` using a 3D latent upscaler + tiled low-denoise refine pass to promote approved draft clips to 1080p on 16–24 GB GPUs with natural skin texture.

Rules:

- A parameter with no binding is ignored, so leave out anything your workflow does not take
  (for example, bind `duration` instead of `num_frames` for P-H3 / Seedance, or omit
  `negative_prompt` for Flux / Krea / Nano Banana).
- A parameter listed in `required` must have a binding.
- One parameter can drive several node inputs (see `fps` in the LTX examples).
- `defaults` sets values for any bound parameter, for example `{"steps": 30}`. Strings in
  `defaults` support `${ENV_VAR}` expansion for `COMFY_*` and `MVD_*`
  (for example `{"api_key": "${COMFY_SEEDANCE_API_KEY}"}`), so you never have to hard-code paid
  API keys in `.api.json` or `.bindings.json`.
- An optional reference image a shot does not have (`set_image`, `prop_image`, and the others
  in the table below) is left out of that job: the `Load...` node
  bound to it is removed. Connect each one to an optional input of the model node.
- Extra parameters from `retry.adjust_params` in `project.yaml` (such as `steps`) are sent
  on later attempts and need a binding to have any effect.
- `image`, `reference_image` and `audio` receive the name of a file the pipeline has
  uploaded to ComfyUI's input folder, in the form `LoadImage` / `LoadAudio` nodes expect.

## Using paid cloud models in ComfyUI (Nano Banana, Seedance, Grok, P-H3)

Paid models still run through ComfyUI using ComfyUI Official Partner Nodes or community API
nodes:

- **Image generation & editing (e.g. Nano Banana / Grok):** Wire a `LoadImage` node (bound to
  `reference_image`) into the paid image-edit node and connect its output to `SaveImage`
  (bound to `output.node`).
- **Video generation (e.g. Seedance / P-H3 / Kling):** Wire a `LoadImage` node (bound to
  `image`) and optional `LoadAudio` node (bound to `audio`) into the paid video node, bind
  `duration` (or `num_frames`), and connect its output to `SaveVideo` / `VHS_VideoCombine`.
- **Controlling paid API spend:** Lower `retry.max_attempts` and `retry.max_generations_per_run`
  in `project.yaml` so QC retries stay within your budget.

## Reference sheets and set plates

Reference sheets are wide images (2048x896 for `row`, 1536x1536 for `grid`, or 1536x1024 for
`columns`, configured under `reference:` in `project.yaml`), so `t2i` and `still` must bind
`width` and `height` or the sheets come out at the wrong size. Use `reference.background: grey`
for flat neutral-grey lighting when bright/neon costumes clip on pure white.

You can also render and lock a silent 4-panel turnaround reference video per wardrobe state
(`mvd reference video`, `mvd reference lock-video <mp4>`). When `lipsync` or `i2v` binds
`reference_video`, the shot's locked turnaround video is uploaded alongside the still.

For a singer still, `reference_image` receives the **front panel** of the sheet for the shot's
wardrobe state. These parameters are optional. A workflow that has no binding for one simply
does not receive it:

| Parameter | What it receives | Bind it when |
|---|---|---|
| `reference_profile` | Profile panel of the shot's sheet | Your edit model takes several input images |
| `reference_three_quarter` | Three-quarter panel | Same |
| `reference_face` | Face detail panel | Same |
| `reference_image_2` | Second character's front panel on two-shots | Your edit model supports two subject images |
| `set_image` | The set's locked plate | You want the still placed in an approved location |
| `prop_image` | Primary hero prop sheet for the shot (`prop_image_2` for a second) | Shots feature recurring hero props (`mvd props`) |

A reference image a shot does not have (for example `prop_image` on a shot without props,
`reference_image_2` on a solo shot, or `set_image` when running with `--skip-preflight`) is not
sent, and the `Load...` node bound to it is removed from that job together with the input it
fed. Feed each one into an optional slot, such as one image of a multi-image edit node, so the
graph still validates without it. A value set under `defaults` is used instead of removing the
node.

Binding `set_image` makes plates **required** in preflight: `mvd stills` stops until every set
used by the selected singer shots has a locked plate (`mvd plates generate`, then
`mvd plates lock`), unless `--skip-preflight` is passed. Use
`mvd plates generate --set "Empty Room" --from-set "Room"` (or `parent_set` in the bible) to
derive a variant plate by editing an existing locked plate. Leave `set_image` unbound to keep
describing sets in text only. Binding `prop_image` similarly requires locked prop sheets
(`mvd props generate`, `mvd props lock`) for shots that list props.

When a plate is supplied, the still prompt says "the first reference image shows the singer, the
second shows the location" (or `@image1` / `@image2` when `comfy.reference_notation: tags` is set
in `project.yaml` for Seedream/Seedance/OpenArt workflows). If your edit model refers to input
images differently, change `PLATE_NOTE` in `planning/prompts.py`.

## 3D Blocking Previz (`previz_video`)

For shots with complex camera paths or multi-figure choreography, generate flat-shaded 3D blocking
videos with `mvd previz generate --shot s001` (or attach an external Blender/Three.js render via
`mvd previz lock <clip.mp4> --shot s001 --mapping "..."`). If your `i2v` or `lipsync` workflow
binds `previz_video`, `ShotRunner` uploads the blocking video and prepends the stand-in mapping
instructions to the motion prompt.
