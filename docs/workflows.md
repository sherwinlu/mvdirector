# ComfyUI Workflows & Model Bindings

Each workflow is a pair of files in the project's `workflows/` folder:

- `<name>.api.json`: your workflow, exported from ComfyUI with **Save (API Format)**.
- `<name>.bindings.json`: maps logical parameters (`positive_prompt`, `seed`, `image`, `audio`, `num_frames`, etc.) to a node ID and input field in that export.

## Standard Workflows

| Name | Used for |
|---|---|
| `t2i` | Base character reference sheets, set plate candidates, hero prop candidates, and b-roll stills (e.g. Qwen-Image, Z-Image, Flux, Seedream) |
| `still` | Singer stills edited from a reference sheet panel, derived character/outfit sheets, derived/embedded set plates, and derived prop sheets (e.g. Qwen-Image-Edit, Flux Redux/Kontext, Seedream Edit, or a T2I/LoRA workflow without `reference_image`) |
| `lipsync` | Image + vocal audio (or `audio_video` black-screen MP4) to video for singing shots (e.g. LTX, Wan + InfiniteTalk/LatentSync, Seedance, MiniMax H3) |
| `i2v` | Image to video for non-singing shots and silent character turnaround reference videos (e.g. LTX, Wan 2.1/2.2, Seedance, MiniMax H3, HunyuanVideo) |
| `upscale` *(optional)* | Video upscaler used by `mvd upscale` to promote `--draft` clips to final resolution |

`workflows/README.md` in each initialized project has the step-by-step setup and the full parameter list. When `mvd run` finishes all stills and moves to videos, it sends `POST /free` to ComfyUI to unload the image model from VRAM before loading the video model.

---

## Customizing for Any Image or Video Model (Open-Weight or Paid API Nodes)

**All image and video generation runs through ComfyUI.** Because `mvdirector` talks to ComfyUI purely through your exported `<name>.api.json` graph and `<name>.bindings.json` map, you can swap in any open-weight model **or** any paid cloud model (via ComfyUI Official Partner Nodes or custom API nodes such as FAL.ai / Replicate / vendor nodes) without changing Python code:

1. **Build and test the workflow in ComfyUI** (local GPU, RunPod, or CPU-only ComfyUI if using cloud API nodes).
2. **Export it with Save (API Format)** into your project's `workflows/` folder.
3. **Map its inputs in `<name>.bindings.json`**. The pipeline only sends parameters you bind; unbound parameters are ignored:
   - **Stills (`t2i`, `still`):** `positive_prompt`, `negative_prompt`, `seed`, `width`, `height`, plus `reference_image` (and optional `reference_profile`, `reference_three_quarter`, `reference_face`, `reference_image_2` on two-shots, `set_image`, `prop_image`, `prop_image_2`) on `still`.
   - **Videos (`lipsync`, `i2v`):** `positive_prompt`, `negative_prompt`, `lyrics`, `seed`, `image`, `audio` (on `lipsync`, or on `i2v` if listed in `required`), `audio_video` (optional black-screen `.mp4` carrying the shot's audio track for video-to-video lip-sync nodes such as Seedance), `reference_video` (optional locked silent turnaround video of the character), `previz_video` (optional locked 3D camera-blocking video for the shot), `width`, `height`, `num_frames`, `duration` (seconds), and `fps`.
   - **Upscale (`upscale`):** `video`, `width`, `height`, `seed`, `fps`.
4. **Point `comfy.workflows` in `project.yaml`** at your new `.bindings.json` files and set the matching `video:` timing rules.
5. **Run `mvd validate`**—it checks every node ID and input field name against your `.api.json` (and lists available fields on the node if you mistype one).

### Reference Notation (`prose` vs `tags`)

Some multi-input video/image nodes (such as Seedance / Omni-Reference workflows) prefer `@image1` / `@video1` handles rather than prose phrases like `"the first reference image"`. Set `comfy.reference_notation` in `project.yaml`:

- `prose` *(default)*: composes natural-language references (`"Matching the character in the reference image..."`).
- `tags`: composes `@image1`, `@video1`, etc. tokens matching your bound slot order.

### Black-Screen Audio Video (`audio_video`)

For lip-sync models that ingest an audio-bearing video clip rather than a raw `.wav` file, bind `"audio_video"` in `lipsync.bindings.json` (for example using the starter `seedance_lipsync.bindings.json` template). `mvdirector` automatically wraps the shot's vocal slice into a matching black-screen `.mp4` and uploads it alongside the still.

---

## Model-Family Settings Reference

| Model family (in ComfyUI) | `video:` settings in `project.yaml` | Workflow & binding notes |
|---|---|---|
| **LTX-2 / LTX-2.3 / LTX-2.5** (open-weight default) | `frame_multiple: 8`, `frame_offset: 1`, `cut_mode: bars`, `max_clip_seconds: 8.0` | Bind `num_frames` ($8n + 1$) or `duration` (`i2v` works cleanly; for tight singing `lipsync`, pair with a LatentSync pass or use InfiniteTalk below). |
| **Wan 2.1 + InfiniteTalk** (Apache-2.0 singing lip-sync) | `frame_multiple: 4`, `frame_offset: 1`, `cut_mode: bars`, `max_clip_seconds: 8.0` | Point `comfy.workflows.lipsync` at `infinitetalk_lipsync.bindings.json` (start from ComfyUI's `wan2_1_infinitetalk_music` template and bind `fps` to 24). |
| **I2V + LatentSync 1.6** (two-stage mouth dub) | Match your base I2V model (`8n+1` for LTX, `4n+1` for Wan) | Point `comfy.workflows.lipsync` at `latentsync_lipsync.bindings.json` to keep base I2V motion while re-dubbing the mouth from `audio` (weights are OpenRAIL++ — review its use-case restrictions before commercial use). |
| **Wan 2.1 / 2.2, HunyuanVideo** (open-weight I2V) | `frame_multiple: 4`, `frame_offset: 1`, `cut_mode: bars`, `max_clip_seconds: 5.0` | Bind `num_frames` ($4n + 1$). In `lipsync`, chain I2V into an audio lip-sync node (InfiniteTalk / Sonic / LatentSync / FantasyTalking). |
| **MOSS** (open-weight audio/lyric video) | `frame_multiple: 4` (or `1`), `frame_offset: 1` (or `0`), `cut_mode: bars` | Bind `image`, `audio`, `positive_prompt`, and optionally `lyrics` in `lipsync.bindings.json`. |
| **Krea / Z-Image / Flux / SDXL** (open-weight or API stills) | Any video settings above | For `t2i` + edit (Flux Kontext/Redux, Qwen-Edit), bind `reference_image` on `still`. For T2I-only or character LoRA workflows without an edit input, omit `reference_image` from `still.bindings.json`. |
| **Nano Banana / Grok / Seedream** (paid image API nodes in ComfyUI) | Any video settings above | Place the image/edit node in ComfyUI between `LoadImage` and `SaveImage`. Bind `positive_prompt`, `reference_image` (for multi-turn character/outfit edits), and `seed`. |
| **Seedance** (paid audio-driven video API node) | `frame_multiple: 1`, `frame_offset: 0`, `cut_mode: bars`, `max_clip_seconds: 10.0` (or `15.0`) | Point `comfy.workflows.lipsync` at `seedance_lipsync.bindings.json` to send `audio_video` (black-screen `.mp4`), `reference_video`, `previz_video`, and `lyrics`, and set `comfy.reference_notation: tags` if your node expects `@image1`/`@video1` handles. |
| **Kling / Runway** (paid I2V API nodes in ComfyUI) | `frame_multiple: 1`, `frame_offset: 0`, `cut_mode: bars`, `max_clip_seconds: 5.0` or `10.0` | Bind `image`, `positive_prompt`, `seed`, and `duration` (or `num_frames`). For lip-sync, chain the I2V node into a ComfyUI lip-sync node (or use the vendor's lip-sync node). |
| **P-H3 / MiniMax H3** (paid or open phrase-driven video in ComfyUI) | `frame_multiple: 1`, `frame_offset: 0`, `cut_mode: phrases`, `min_clip_seconds: 5.0`, `max_clip_seconds: 15.0` | Bind `duration` (seconds), `image`, and `audio`. `cut_mode: phrases` cuts at vocal windows and lyric sections. |

---

## Tips for Paid Cloud API Nodes in ComfyUI

- **ComfyUI is still the runner, even without a local GPU:** When your workflows use ComfyUI Partner Nodes or cloud API nodes, ComfyUI acts as a lightweight local orchestrator (`http://127.0.0.1:8188`) that uploads reference images/audio, calls the paid API, and hands the output to `mvdirector` for quality checks and assembly.
- **Never commit API keys in `.api.json`:** Configure your API key in ComfyUI's server environment / Partner Node settings, or bind an `"api_key"` input in `<name>.bindings.json` and pass it via environment variable expansion in `defaults`. Only `MVD_*`, `COMFY_*`, and `RUNPOD_TOKEN` names expand by default (export `MVD_ALLOWED_ENV_VARS=VAR1,VAR2` in your shell to allow others), so a project file you did not write can never read your other secrets:
  ```json
  "inputs": {
    "api_key": [{"node": "10", "field": "api_key"}]
  },
  "defaults": {
    "api_key": "${COMFY_SEEDANCE_API_KEY}"
  }
  ```
- **Cap retries and preview in Draft Mode to control spend:** Every failed QC check triggers a regeneration attempt. When using paid per-generation nodes, lower `retry.max_attempts` (e.g. `2`) and `retry.max_generations_per_run` (e.g. `40`) in `project.yaml`, test on two shots first with `mvd run --shot s001 --shot s002`, or render low-resolution previews with `--draft` before promoting winners with `mvd upscale`.
