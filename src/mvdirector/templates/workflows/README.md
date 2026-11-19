# Workflows

All image and video generation in `mvdirector` runs through **ComfyUI**. This folder connects
the pipeline to **your** ComfyUI workflows—whether they run open-weight GPU models (Qwen,
Z-Image, Krea, Flux, LTX, Wan, MOSS) or **paid cloud models** via ComfyUI Partner/API nodes
(Nano Banana, Seedance, Grok, P-H3 / MiniMax, Kling, FAL.ai, Replicate).

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
| `lipsync` | Lip-sync clips | `positive_prompt`, `negative_prompt`, `lyrics`, `seed`, `image`, `audio`, `audio_video`, `reference_video`, `previz_video`, `width`, `height`, `num_frames`, `duration`, `fps` |
| `i2v` | All other clips | as `lipsync`, without `audio` / `audio_video` (unless listed in `required`) |
| `upscale` (optional) | Selective post-review clip upscaling (`mvd upscale`) | `video`, `width`, `height`, `fps`, `seed`, `positive_prompt` |

**Dedicated singing lip-sync templates shipped here:** LTX audio-conditioned A2V can drift on fast/sung syllables. Starter bindings for tighter mouth sync are included alongside the defaults:
- `infinitetalk_lipsync.bindings.json` (Wan 2.1 + InfiniteTalk Single, Apache-2.0): starter placeholder bindings (replace example node ids with ids from your exported workflow); set `comfy.workflows.lipsync` to this file and `video.frame_multiple: 4`, `video.frame_offset: 1` in `project.yaml`.
- `latentsync_lipsync.bindings.json` (I2V + LatentSync mouth-dub pass): starter placeholder bindings (replace example node ids with ids from your exported workflow); keep your I2V motion (LTX or Wan) and pass the video + vocal slice through LatentSync (weights are OpenRAIL++ — review use restrictions before commercial use).
- `seedance_lipsync.bindings.json` (Seedance Reference-to-Video via `audio_video`): starter placeholder bindings using `audio_video` (an auto-generated black-screen MP4 containing the vocal slice, which improves lip-sync on video-reference nodes) and `comfy.reference_notation: tags` (`@image1`, `@video1`) with `video.motion_include_lyrics: true`.

Rules:

- A parameter with no binding is ignored, so leave out anything your workflow does not take
  (for example, bind `duration` instead of `num_frames` for P-H3 / Seedance, or omit
  `negative_prompt` for Flux / Krea / Nano Banana).
- A parameter listed in `required` must have a binding.
- One parameter can drive several node inputs (see `fps` in the LTX examples).
- `defaults` sets values for any bound parameter, for example `{"steps": 30}`. Strings in
  `defaults` support `${ENV_VAR}` expansion for `COMFY_*`, `MVD_*`, and `RUNPOD_TOKEN`
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

A reference image a shot does not have (no plate for its set, no props) is not sent, and the
`Load...` node bound to it is removed from that job together with the input it fed. Feed each
one into an optional slot, such as one image of a multi-image edit node, so the graph still
validates without it. A value set under `defaults` is used instead of removing the node.

Binding `set_image` makes plates **required**: `mvd stills` stops until every set used by the
selected singer shots has a locked plate (`mvd plates generate`, then `mvd plates lock`). Use
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
