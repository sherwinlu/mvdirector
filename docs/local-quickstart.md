# Running Locally: Cloudless Mode & Cloud Mode

This guide walks you from a clean laptop to your first rendered draft music video on **`mvdirector` `v1.0.0`** in either **Cloudless Mode** (100% local, zero cloud APIs) or **Cloud Mode** (running `mvd` on your laptop while hitting real cloud LLMs and a real cloud GPU or paid ComfyUI API nodes).

`mvdirector` is **setup-, model-, and ComfyUI-version-agnostic**:
- **`mvdirector` (`mvd`)** always runs on your laptop (orchestration, audio analysis, quality checks, FFmpeg assembly, and Streamlit review UI).
- **LLM** directs the shot list, judges visual QC, and rewrites failed prompts — either locally (Ollama / LM Studio / llama.cpp) or in the cloud (Anthropic, Gemini, OpenAI, or any OpenAI-compatible endpoint). You can even **split LLMs** (e.g. Claude for planning, Gemini for Vision QC and native MP3 listening).
- **ComfyUI** executes all image, video, and lip-sync workflows over HTTP(S) — either locally on your laptop/workstation, on a home-LAN/Tailscale GPU box, on a cloud GPU host (RunPod, Vast.ai, Lambda, etc.), or as a CPU-only ComfyUI instance calling paid cloud API nodes. Because `mvd` uploads assets over HTTP and polls `/history`, **no shared filesystem is required**.

## Choose Your Mode

| Mode | LLM | ComfyUI / GPU | Cloud API keys? | Best for |
| :--- | :--- | :--- | :--- | :--- |
| **Cloudless: Full Local** | Local Ollama (`gemma3:12b` or `qwen2.5vl:7b`) | Local ComfyUI (Apple Silicon M1–M4 or NVIDIA GPU) | None | Privacy, zero per-run cost, offline work |
| **Cloudless: No-LLM (Rule-Based)** | Disabled (`llm.enabled: false`) | Local ComfyUI | None | Deterministic planning & fast ComfyUI pipeline testing |
| **Cloudless: Dev / Mock Harness** | `FakeLLMServer` (`pytest`) | `FakeComfy` + real `ffmpeg` (`pytest`) | None | Developing or verifying the full pipeline with **no GPU and no LLM** |
| **Cloud: Cloud LLM + Cloud GPU** | Anthropic / Gemini / OpenAI (or Split-LLM) | Remote GPU ComfyUI (HTTPS + Bearer or SSH tunnel) | Yes (LLM + optional `COMFY_TOKEN`) | Production quality & fast renders from a thin laptop |
| **Cloud: Cloud LLM + Paid API Nodes** | Anthropic / Gemini / OpenAI | Local CPU-only ComfyUI calling Partner/API nodes (Seedance, Nano Banana, Kling, MiniMax H3) | Yes (LLM + `COMFY_ORG_API_KEY` / vendor key) | Zero GPU management; pay-per-generation from any laptop |

*(Hybrid setups work identically — for example, Cloud LLM + Local GPU, or Local Ollama + Remote Cloud GPU.)*

---

## Step 1: Install Python 3.11+ and FFmpeg

Make sure `python3` (3.11 or newer; 3.12 recommended), `ffmpeg`, and `ffprobe` are on your `PATH`:

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

## Step 2: Clone and Install `mvdirector`

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
.\.venv\Scripts\Activate.ps1   # if blocked, run: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install -c constraints.txt -e ".[qc,align,ui]"
```

**Optional extras depending on your workflow:**
- **Using Anthropic Claude in Cloud Mode:** add `anthropic` (`pip install -c constraints.txt -e ".[qc,align,ui,anthropic]"`).
- **Starting from a single `.mp3`/`.wav` file (`mvd prepare` with Demucs stem separation):** add `stems`.
- **Running the test suite / Cloudless mock harness:** add `dev`.
- **Neural lip-sync scoring (`qc.lipsync_backend: stable_syncnet`):** add `syncnet`.

---

## Track A: Cloudless Mode (100% Local on Your Laptop)

Follow this track if you want everything running locally with zero cloud API calls.

### A1. Start Local Ollama (Vision LLM)

1. Install [Ollama](https://ollama.com) and ensure the server is running (`ollama serve` if it is not already running as a background app/service).
2. Pull a local vision-capable model (vision support is required for character/costume/framing QC):
   ```bash
   ollama pull gemma3:12b          # default recommended (32 GB+ RAM/VRAM)
   # On 16-24 GB laptops:
   # ollama pull qwen2.5vl:7b
   ```

> **Apple Silicon (M1–M4) memory note:** Run Ollama and ComfyUI **natively on macOS** (Docker Desktop on macOS cannot access Apple Metal/MPS). Keep only one heavy model loaded in Ollama at a time; on 16–24 GB Macs, use `qwen2.5vl:7b` and lower `llm.shots_per_request` to `5`–`10` in `project.yaml`.

### A2. Start Local ComfyUI

- **Native install (recommended for macOS Apple Silicon; works on Linux/Windows too):**
  1. Install [ComfyUI](https://github.com/comfyanonymous/ComfyUI) (or ComfyUI Desktop) and place checkpoints/custom nodes for your chosen open-weight models (e.g., Qwen-Image + Qwen-Image-Edit for stills; LTX-2.x or Wan 2.1 + InfiniteTalk for video).
  2. Start ComfyUI on loopback:
     ```bash
     python main.py --listen 127.0.0.1 --port 8188
     ```
  3. Open `http://127.0.0.1:8188` in your browser and verify you can generate a test image.

- **Docker Compose (Linux / Windows with NVIDIA GPU + NVIDIA Container Toolkit):**
  ```bash
  COMFY_URL=http://comfy-gpu:8188 MVD_LLM_URL=http://ollama-gpu:11434 \
    docker compose --profile comfy-gpu --profile llm-gpu --profile ui up -d
  ```
  *(On Windows PowerShell, set `$env:COMFY_URL = "http://comfy-gpu:8188"; $env:MVD_LLM_URL = "http://ollama-gpu:11434"` before `docker compose`.)* `COMFY_RESERVE_VRAM` defaults to `10` GiB so Ollama and ComfyUI can share one GPU.

### A3. Fully Offline / Air-Gapped & No-LLM Notes

- **Pre-warming local QC/alignment caches:** On first run, `qc` downloads two ~5 MB MediaPipe models into `~/.cache/mvdirector/mediapipe` (override with `MVD_MEDIAPIPE_CACHE_DIR`), `align` downloads `faster-whisper` weights into Hugging Face cache, and optional `syncnet` downloads `stable_syncnet.pt` (~1.6 GB) into `~/.cache/mvdirector/syncnet` (`MVD_SYNCNET_CACHE_DIR`). Run one `mvd validate` / `mvd analyze` while online (or copy those cache directories) before going completely offline.
- **No-LLM / Rule-Based Mode:** To run without any LLM at all (deterministic beat/bar planning, no vision QC, no prompt rewrites), set in `project.yaml`:
  ```yaml
  llm:
    enabled: false
  qc:
    vision_enabled: false
  ```
- **Zero-GPU / Zero-LLM Engineering Test Harness:** If you are developing on a laptop without ComfyUI or Ollama installed, you can still run the entire end-to-end pipeline (`analyze` → `plan` → `stills` → `videos` → `QC` → `retry` → `assemble`) locally using the built-in fake ComfyUI and multi-protocol `FakeLLMServer` with real `ffmpeg`:
  ```bash
  pip install -c constraints.txt -e ".[dev,qc,align,ui]"
  pytest tests/test_pipeline.py tests/test_llm_server.py
  ```

---

## Track B: Cloud Mode (Real Cloud LLM + Real Cloud GPU)

Follow this track if you are running `mvd` on your laptop and hitting real cloud LLMs and a real cloud GPU (or paid ComfyUI cloud API nodes).

### B1. Configure Cloud LLM Credentials & Split-LLM Routing

Never put API keys in `project.yaml`. Export them in your shell or let `mvd setup` write them to an owner-only (`0600`), gitignored `<project>/.env` file (shell exports always override `.env`):

```bash
# Pick the provider(s) you use:
export ANTHROPIC_API_KEY="sk-ant-..."     # requires the [anthropic] extra
export GEMINI_API_KEY="AIza..."           # (GOOGLE_API_KEY also works)
export OPENAI_API_KEY="sk-..."
```

In `project.yaml`, configure either a **single cloud LLM** or the recommended **Split-LLM** setup (a strong planning model + a fast/cheap multimodal model for Vision QC and native MP3 audio listening):

```yaml
# Example: Split-LLM Cloud Mode (Claude plans, Gemini listens to MP3 & runs Vision QC)
llm:
  provider: anthropic          # or: gemini | openai | openai_compatible
  model: claude-sonnet-4-6     # use any current model name your provider offers
  max_tokens: 8000
  timeout_seconds: 180

planning:
  audio_listen: auto           # 1-call native MP3 listen when main planner is text-only
  # audio_llm defaults to qc.llm when qc.llm is audio-capable (e.g. gemini)

qc:
  vision_enabled: true
  llm:
    provider: gemini
    model: gemini-2.5-flash    # fast, low-cost vision judge
```

*(If you use a single Gemini model for everything, omit `qc.llm` and `planning.audio_llm` — Gemini handles planning, native MP3 listening, and Vision QC directly. If you point `openai_compatible` at a hosted vLLM server behind Cloudflare or a reverse proxy, set `llm.stream: true` to prevent 100-second idle-connection `524` timeouts.)*

### B2. Connect Your Laptop to a Cloud GPU (or Cloud API Nodes)

Choose one of three ways to reach ComfyUI in Cloud Mode:

#### Option 1: Remote Cloud GPU over HTTPS + Bearer Token (RunPod, Vast.ai, Lambda, etc.)

On your cloud GPU instance, run ComfyUI (or the all-in-one image `docker/Dockerfile.allinone` which bundles ComfyUI + CUDA + `mvd`) behind HTTPS + Bearer authentication. **Never expose unauthenticated ComfyUI to the public internet.**

On your laptop:
```bash
export COMFY_URL="https://gpu.example.com:8188"
export COMFY_TOKEN="your-bearer-token"
```
And in `project.yaml`:
```yaml
comfy:
  base_url: ${COMFY_URL}
  headers:
    Authorization: "Bearer ${COMFY_TOKEN}"
```
Verify reachability before creating a project:
```bash
mvd setup --check-only --comfy-url "$COMFY_URL" --comfy-token-env COMFY_TOKEN
```

#### Option 2: Remote Cloud GPU over an SSH Tunnel (Simplest for Cloud VMs)

If your cloud GPU VM only exposes SSH and ComfyUI listens on `127.0.0.1:8188` on the remote box, forward port `8188` to your laptop:
```bash
ssh -N -L 8188:127.0.0.1:8188 user@your-cloud-gpu-host
```
Then leave `COMFY_URL=http://127.0.0.1:8188` on your laptop — `mvd` talks to the cloud GPU seamlessly over the encrypted tunnel with no public port exposure.

#### Option 3: Local CPU-Only ComfyUI + Paid Cloud API Nodes (No GPU VM Needed)

If you want to call paid cloud image/video models (Seedance, Nano Banana, Kling, Grok, MiniMax H3, FAL.ai, etc.) directly through ComfyUI Partner/API nodes without renting a GPU VM:

1. Run a lightweight CPU-only ComfyUI on your laptop:
   ```bash
   docker compose --profile comfy-cpu up -d
   # Or natively: python main.py --cpu --listen 127.0.0.1 --port 8188
   ```
2. Export your Comfy.org API key and opt in explicitly in `project.yaml` (an exported key is **never** forwarded unless `comfy.api_key_comfy_org` is configured, preventing accidental leakage to third-party servers):
   ```bash
   export COMFY_ORG_API_KEY="comfyui-..."
   ```
   ```yaml
   comfy:
     base_url: http://127.0.0.1:8188
     api_key_comfy_org: ${COMFY_ORG_API_KEY}
   ```

### B3. Cloud Mode Spend & Performance Guardrails

Before running a full song against paid cloud APIs or billed GPU hours:
- **Cap retries and total spend** in `project.yaml`:
  ```yaml
  retry:
    max_attempts: 2
    max_generations_per_run: 40
  ```
- **Avoid re-uploading and re-rendering** on iterative runs:
  ```yaml
  comfy:
    dedup_uploads: true     # content-addressed SHA-256 uploads checked via HEAD /view
    cache_enabled: true     # local .comfy_cache for identical graph + input hashes
  ```
- **Parallelize on serverless / multi-pod ComfyUI:** Set `comfy.max_concurrency: 4` (or pass `-j 4`) when your cloud ComfyUI endpoint fans out across multiple workers. Keep `1` for a single GPU box.
- **Always test 2 shots in `--draft` mode first** (`mvd run --draft --shot s001 --shot s002`) before running the whole song.

---

## Step 3: Run `mvd setup` to Create Your Project

`mvd setup` inspects `ffmpeg`/`ffprobe`, Python extras, Ollama, and ComfyUI (GPU/VRAM + `/object_info`), then writes a configured `project.yaml` and runs live LLM + ComfyUI sanity checks.

```bash
# Interactive wizard (recommended for both Cloudless and Cloud Mode):
mvd setup projects/my-song
```

Or run non-interactively:
```bash
# Cloudless (local Ollama + local ComfyUI):
mvd setup projects/my-song --non-interactive --title "My Song" --video-preset ltx

# Home LAN / Tailscale ComfyUI:
mvd setup projects/my-song --non-interactive --title "My Song" \
  --comfy-url http://192.168.1.50:8188 --video-preset wan

# Cloud Mode (HTTPS ComfyUI + Bearer auth + exact-duration model):
mvd setup projects/my-song --non-interactive --title "My Song" \
  --comfy-url https://gpu.example.com:8188 --comfy-token-env COMFY_TOKEN --video-preset exact
```

`mvd setup` checks:
- **LLM connectivity (JSON + Vision):** immediately flags bad API keys, billing/quota limits (`LLMQuotaError`), unknown models, or text-only models.
- **ComfyUI connectivity & wiring:** confirms `/system_stats` and `/object_info`, and validates any `.api.json` / `.bindings.json` files already in `workflows/`. Pass `--no-verify` to skip live probes.

---

## Step 4: Add Your Song Inputs

```bash
cd projects/my-song
```

Choose whichever starting point matches your files:

- **Single `.mp3` or `.wav` file** (requires the `stems` and `align` extras):
  ```bash
  mvd prepare /path/to/song.mp3
  ```
  Transcodes `inputs/mix.wav`, separates `inputs/vocals.wav` with Demucs, transcribes with `faster-whisper`, and drafts `inputs/lyrics.txt` + `inputs/style.md`.
- **Existing stems / Suno export:** Place in `inputs/`:
  - `inputs/mix.wav` — full stereo mix (also the final soundtrack)
  - `inputs/vocals.wav` — isolated vocal stem (drives lip-sync)
  - `inputs/lyrics.txt` — lyrics with `[Verse 1]` / `[Chorus]` section tags
  - `inputs/style.md` — visual style brief, character description, and setting notes
- **Starting from an idea** (before generating the song):
  ```bash
  mvd brief --idea "A neon cyberpunk ballad in the rain"
  ```

---

## Step 5: Export and Bind Your ComfyUI Workflows

`mvdirector` ships with starter `.bindings.json` files in `workflows/`, but **you must supply the matching `.api.json` workflow exports from your ComfyUI instance** (local or cloud):

1. In ComfyUI's settings (gear icon), enable **Dev Mode Options**.
2. Open and test your image (`t2i`, `still`) and video (`i2v`, `lipsync`) workflows directly in ComfyUI until they produce clean outputs.
3. Click **Save (API Format)** for each workflow and save the `.api.json` files into `projects/my-song/workflows/`.
4. Open the corresponding `.bindings.json` files in `workflows/` and verify each node ID and input field name matches your exported `.api.json`.

See `projects/my-song/workflows/README.md` and [ComfyUI Workflows & Model Bindings](workflows.md) for all supported bindings and model-family settings.

---

## Step 6: Validate, Plan, Lock Assets, and Render a 2-Shot Draft

Always run the two preflight checks before spending GPU time or cloud credits:

```bash
mvd validate     # checks ffmpeg, workflow bindings, API keys, and ComfyUI /object_info
mvd llm-check    # verifies LLM JSON output + image vision support
```

Once both pass, run the pipeline on **two shots in draft mode first**:

```bash
mvd analyze
mvd plan

# Generate and lock your character reference sheet and set plates
mvd reference generate
mvd reference lock assets/reference/<best_candidate>.png
mvd plates generate
mvd plates lock assets/plates/<best_candidate>.png --set "<Set Name>"
mvd assets --lint                               # confirm required assets are locked

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

## Troubleshooting (Cloudless & Cloud)

- **Ollama times out during `mvd stills` or `mvd run` (Cloudless):** Ollama and ComfyUI share RAM/VRAM on one machine. Make sure no other large model is loaded in Ollama, switch to `qwen2.5vl:7b`, raise `llm.timeout_seconds`, or lower `llm.shots_per_request` to `5`–`10`.
- **Reverse-proxy HTTP 524 / idle timeout on cloud vLLM / RunPod (Cloud):** Set `llm.stream: true` in `project.yaml` so responses stream over SSE and keep the connection alive.
- **Cloud proxy returns HTTP 403 on ComfyUI requests (Cloud):** `ComfyClient` already sends `User-Agent: mvdirector/<version>` instead of `python-httpx/...`. If your proxy expects a specific header or token, set it under `comfy.headers`.
- **Fresh cloud GPU container fails `mvd validate` (Cloud):** Ephemeral cloud containers often boot without your checkpoints or custom nodes; always run `mvd validate` after restarting a cloud instance.
- **`MediaPipe` / `qc` dependency error on startup:** Install the `qc` extra (`pip install -c constraints.txt -e ".[qc]"`), or temporarily set `qc.on_missing_dependency: skip` in `project.yaml`.
- **Dimension or frame-count errors from ComfyUI:** Check `video.size_multiple`, `video.frame_multiple`, and `video.frame_offset` against [ComfyUI Workflows & Model Bindings](workflows.md#model-family-settings-reference), or re-run `mvd setup --video-preset <ltx|wan|wan-i2v|exact|custom>`.
- **Apple Silicon Docker container cannot see the GPU (Cloudless):** Run Ollama and ComfyUI natively on macOS and point Dockerized `mvd` at `COMFY_URL=http://host.docker.internal:8188` and `MVD_LLM_URL=http://host.docker.internal:11434`.
