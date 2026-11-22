# Local Quickstart Guide

This guide walks you from a clean machine to your first rendered draft music video running **100% locally** — no cloud API keys required.

Everything runs on your own machine:
- **`mvdirector` (`mvd`)**: Python orchestrator, audio analysis, quality checks, and FFmpeg assembly.
- **Ollama**: local multimodal LLM (`gemma3:12b` by default) for directing the shot list, writing prompts, and running vision QC.
- **ComfyUI**: local image and video generation server (`http://127.0.0.1:8188`).

> **Hardware guidelines**
> - **Apple Silicon Mac (M1–M4):** 32 GB+ unified memory is comfortable for running Ollama (`gemma3:12b`) and ComfyUI; on 16–24 GB Macs, use a smaller vision model such as `qwen2.5vl:7b` and keep only one heavy model loaded at a time. Run Ollama and ComfyUI **natively on macOS** (Docker Desktop on macOS cannot access Apple Metal/MPS).
> - **Windows / Linux with NVIDIA GPU:** 16–24 GB+ VRAM is recommended for local open-weight video models (LTX, Wan 2.1 + InfiniteTalk). You can run natively or via Docker Compose (`--profile comfy-gpu`).
> - **No local GPU?** You can still run `mvdirector` + Ollama locally and point a lightweight CPU-only ComfyUI at paid cloud API nodes (see [ComfyUI Workflows & Model Bindings](workflows.md#tips-for-paid-cloud-api-nodes-in-comfyui)).

---

## Step 1: Install Python 3.11+ and FFmpeg

Make sure `python3` (3.11 or newer), `ffmpeg`, and `ffprobe` are on your `PATH`:

**macOS (Homebrew):**
```bash
brew install python@3.12 ffmpeg
```

**Ubuntu / Debian:**
```bash
sudo apt update && sudo apt install -y python3 python3-venv python3-pip ffmpeg
```

**Windows (PowerShell with winget):**
```powershell
winget install Python.Python.3.12 Gyan.FFmpeg
```
*(Restart your terminal after installing on Windows so `python` and `ffmpeg` are on `PATH`.)*

---

## Step 2: Clone and install `mvdirector`

**macOS / Linux:**
```bash
git clone https://github.com/sherwinlu/mvdirector.git
cd mvdirector
python3 -m venv .venv
source .venv/bin/activate
pip install -c constraints.txt -e ".[qc,align,ui]"
```

**Windows (PowerShell):**
```powershell
git clone https://github.com/sherwinlu/mvdirector.git
cd mvdirector
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -c constraints.txt -e ".[qc,align,ui]"
```

> **Optional extra:** If you want to start from a single `.mp3` or `.wav` file and have `mvd prepare` automatically separate vocals/drums with Demucs, add `stems`:
> ```bash
> pip install -c constraints.txt -e ".[qc,align,ui,stems]"
> ```

---

## Step 3: Install and start Ollama

1. Download and install [Ollama](https://ollama.com).
2. Ensure the Ollama server is running (`ollama serve` if it isn't already running as a background app/service).
3. Pull a local vision-capable model (vision support is needed for character/costume/framing QC):

```bash
ollama pull gemma3:12b          # default recommended local model
# Or on 16-24 GB machines:
# ollama pull qwen2.5vl:7b
```

---

## Step 4: Install and start ComfyUI

### Option A: Native install (recommended for macOS Apple Silicon; works on Linux/Windows too)

1. Install [ComfyUI](https://github.com/comfyanonymous/ComfyUI) (or ComfyUI Desktop) and download the checkpoints/custom nodes for the image and video models you plan to use (for example: Qwen-Image + Qwen-Image-Edit for stills, and LTX-2.x or Wan 2.1 + InfiniteTalk for video).
2. Start ComfyUI so it listens on `http://127.0.0.1:8188`:
   ```bash
   python main.py --listen 127.0.0.1 --port 8188
   ```
3. Open `http://127.0.0.1:8188` in your browser and verify you can generate a test image.

### Option B: Docker Compose (Linux / Windows with NVIDIA GPU)

If you have Docker + the NVIDIA Container Toolkit installed, you can launch GPU ComfyUI, GPU Ollama, and the review UI together from the repo root:

```bash
COMFY_URL=http://comfy-gpu:8188 MVD_LLM_URL=http://ollama-gpu:11434 \
  docker compose --profile comfy-gpu --profile llm-gpu --profile ui up -d
```

See [Installation, Docker & Language Models](setup.md#running-with-docker-local-or-cloud-gpu-stack) for all Compose profiles and volume mounts.

---

## Step 5: Run `mvd setup`

`mvd setup` checks `ffmpeg`/`ffprobe`, installed Python extras, local Ollama (and its pulled models), and local ComfyUI (and its GPU/VRAM), then interactively creates and configures your project's `project.yaml`.

To inspect your machine without creating a project yet:
```bash
mvd setup --check-only
```

To create and configure your first project interactively:
```bash
mvd setup projects/my-song
```

`mvd setup` will prompt you for:
- **Project title**
- **ComfyUI URL** (defaults to `http://127.0.0.1:8188` or `COMFY_URL`)
- **LLM provider & model** (defaults to `ollama` and automatically picks `gemma3:12b` or another vision model you already pulled)
- **Video model family preset** (`ltx`, `wan`, `wan-i2v`, or `custom`), which writes the matching `frame_multiple`, `frame_offset`, `size_multiple`, and `max_clip_seconds` into `project.yaml`

You can also run it non-interactively with flags:
```bash
mvd setup projects/my-song --non-interactive --title "My Song" --video-preset ltx
```

---

## Step 6: Add your song inputs

```bash
cd projects/my-song
```

Choose whichever starting point matches your files:

- **Starting from a single `.mp3` or `.wav` file** (requires the `stems` and `align` extras):
  ```bash
  mvd prepare /path/to/song.mp3
  ```
  This transcodes `inputs/mix.wav`, separates `inputs/vocals.wav` with Demucs, transcribes lyrics with `faster-whisper`, and drafts `inputs/lyrics.txt` and `inputs/style.md`.
- **Starting from Suno stems / existing files:** Place these in `inputs/`:
  - `inputs/mix.wav` — full stereo mix
  - `inputs/vocals.wav` — isolated vocal stem (drives lip-sync)
  - `inputs/lyrics.txt` — lyrics with `[Verse 1]` / `[Chorus]` section tags
  - `inputs/style.md` — visual style brief, character description, and setting notes
- **Starting from just an idea** (before generating the song in Suno):
  ```bash
  mvd brief --idea "A neon cyberpunk ballad in the rain"
  ```

---

## Step 7: Export and bind your ComfyUI workflows

`mvdirector` ships with starter `.bindings.json` files in `workflows/`, but **you must supply the matching `.api.json` workflow exports from your own ComfyUI instance**:

1. In ComfyUI's settings (gear icon), enable **Dev Mode Options**.
2. Open and test your image (`t2i`, `still`) and video (`i2v`, `lipsync`) workflows directly in ComfyUI until they produce clean outputs.
3. Click **Save (API Format)** for each workflow and save the `.api.json` files into `projects/my-song/workflows/`.
4. Open the corresponding `.bindings.json` files in `workflows/` and verify each node ID and input field name matches your exported `.api.json`.

See `projects/my-song/workflows/README.md` and [ComfyUI Workflows & Model Bindings](workflows.md) for the complete list of supported bindings and model-family settings.

---

## Step 8: Validate, plan, lock assets, and render a 2-shot draft

Always run the two preflight checks before spending GPU time:

```bash
mvd validate     # checks ffmpeg, workflow bindings, and ComfyUI /object_info
mvd llm-check    # verifies Ollama JSON output + image vision support
```

Once both pass, run the pipeline on **two shots in draft mode first** so you can verify everything end-to-end quickly:

```bash
mvd analyze
mvd plan

# Generate and lock your character reference sheet and set plates
mvd reference generate
mvd reference lock assets/reference/<best_candidate>.png
mvd plates generate
mvd plates lock assets/plates/<best_candidate>.png --set "<Set Name>"
mvd assets                                      # confirm required assets are locked

# Render a fast 2-shot draft preview
mvd run --draft --shot s001 --shot s002
mvd review                                      # opens Streamlit UI at http://127.0.0.1:8501
mvd assemble --force --draft                    # preview output/<title>.mp4
```

When the 2-shot draft looks good, run the full video:
```bash
mvd run
mvd review
mvd assemble
```

---

## Troubleshooting Local Setups

- **Ollama times out during `mvd stills` or `mvd run`:** Ollama and ComfyUI share system RAM/VRAM on a single machine. Make sure no other large model is loaded in Ollama, switch to a smaller vision model (`qwen2.5vl:7b`), or raise `llm.timeout_seconds` in `project.yaml`. You can also lower `llm.shots_per_request` to `5`–`10` for smaller local models.
- **`MediaPipe` / `qc` dependency error on startup:** Install the `qc` extra (`pip install -c constraints.txt -e ".[qc]"`), or temporarily set `qc.on_missing_dependency: skip` in `project.yaml` while testing.
- **Dimension or frame-count errors from ComfyUI:** Re-check your `video.size_multiple`, `video.frame_multiple`, and `video.frame_offset` values against the table in [ComfyUI Workflows & Model Bindings](workflows.md#model-family-settings-reference) (or re-run `mvd setup --video-preset <ltx|wan|wan-i2v>`).
- **Apple Silicon Docker container cannot see the GPU:** Run Ollama and ComfyUI natively on macOS and point Dockerized `mvd` at `COMFY_URL=http://host.docker.internal:8188` and `MVD_LLM_URL=http://host.docker.internal:11434`.
