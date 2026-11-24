# Installation, Docker & Language Models

> Looking for a step-by-step walkthrough (fully local, home-LAN ComfyUI, or remote cloud GPU)? See the [Quickstart Guide](local-quickstart.md) and the interactive `mvd setup` command.

## Supported Topologies (Setup- & Model-Neutral)

`mvdirector` (`v0.8.0b1` Beta) has been tested end-to-end **both fully locally and with ComfyUI running on a remote host**. Both ComfyUI and the LLM can run wherever works best for your hardware, using whichever models you choose:

| Topology | Where ComfyUI runs | Where `mvd` & LLM run | Example `COMFY_URL` |
|---|---|---|---|
| **Fully local** | Same machine (Apple Silicon Mac, Linux/Windows GPU, or CPU ComfyUI with paid API nodes) | Same machine (local Ollama/LM Studio, or cloud LLM API) | `http://127.0.0.1:8188` |
| **Home LAN / second computer** | Another PC or workstation on your home LAN or Tailscale network | Your laptop/desktop (local or remote LLM) | `http://192.168.1.50:8188` |
| **Remote cloud server** | Cloud GPU VM or container behind HTTPS / reverse proxy (or SSH tunnel) | Your laptop/desktop (local or cloud LLM) | `https://gpu.example.com:8188` *(with `comfy.headers` Bearer auth)* |

Because `mvdirector` uploads reference images and audio slices over HTTP (`POST /upload/image`) and polls `GET /history/{id}`, **no shared filesystem or NFS mount is required** between `mvd` and ComfyUI.

## Install

Requires Python 3.11+ and `ffmpeg`. Installing the `mvdirector` package provides the `mvd` command-line tool.

**macOS / Linux:**

```bash
brew install ffmpeg                                         # or: sudo apt-get install ffmpeg
python3 -m venv .venv && source .venv/bin/activate
pip install -c constraints.txt -e ".[qc,align,ui,dev]"      # add "anthropic" to use llm.provider: anthropic
ollama pull gemma3:12b                                      # default local LLM; or use any Ollama/OpenAI-compatible/cloud model
export COMFY_URL=http://127.0.0.1:8188                      # local, home LAN/Tailscale, or remote cloud HTTPS URL
```

**Windows (PowerShell):**

```powershell
winget install Python.Python.3.12 Gyan.FFmpeg               # restart terminal after installing
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1                                # if blocked, run: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install -c constraints.txt -e ".[qc,align,ui,dev]"      # add "anthropic" to use llm.provider: anthropic
ollama pull gemma3:12b                                      # default local LLM; or use any Ollama/OpenAI-compatible/cloud model
$env:COMFY_URL = "http://127.0.0.1:8188"                    # local, home LAN/Tailscale, or remote cloud HTTPS URL
```

The extras let you install only what you use; by default `qc.on_missing_dependency` is `fail` so missing `qc` packages never downgrade checks silently (set `qc.on_missing_dependency: skip` in `project.yaml` to opt out):

| Extra | Adds | Without it |
|---|---|---|
| `anthropic` | Official Anthropic SDK (`llm.provider: anthropic`) | Use `openai`, `gemini`, `ollama`, or `openai_compatible` without installing the Anthropic SDK. |
| `qc` | MediaPipe Tasks (Apache-2.0): face detection, face framing, face presence and lip-sync. The two model files (~5 MB) download once into `~/.cache/mvdirector/mediapipe` (override with `MVD_MEDIAPIPE_CACHE_DIR`); the Docker images pre-fetch them. | Runs stop at startup unless `qc.on_missing_dependency: skip` is set (which records those checks as `skipped`). |
| `syncnet` | PyTorch + MediaPipe for `qc.lipsync_backend: stable_syncnet` (ByteDance LatentSync's pixel-space `stable_syncnet.pt` under OpenRAIL++ paired with Apache-2.0 MediaPipe Face Detector for face cropping, avoiding Oxford VGG's research-only `syncnet_v2.model` and S3FD). Checkpoint (~1.6 GB) downloads once into `~/.cache/mvdirector/syncnet` (override with `MVD_SYNCNET_CACHE_DIR` or `qc.syncnet_model_path`). | Uses the default CPU-only MediaPipe mouth-vs-loudness check. |
| `align` | `faster-whisper` word timestamps | Lyric timing is estimated from vocal activity. |
| `ui` | Streamlit review UI | Use `mvd status`, `mvd approve`, `mvd reject`. |
| `insightface` | InsightFace `buffalo_l` face-embedding similarity (`qc.face_backend: insightface`) | Identity is checked by the multimodal LLM (`vision_character`) against the reference sheet. |

**Commercial and monetized use (YouTube, client work):** By default (`qc.face_backend: mediapipe`), every library and model bundled or downloaded by the `qc`, `align` and `ui` extras is permissively licensed for commercial use (Apache-2.0 / MIT / BSD), and singer identity is verified by `vision_character` using your reference sheet. Do **not** enable the optional `insightface` extra (`qc.face_backend: insightface`) on commercial or monetized projects, as InsightFace's pretrained `buffalo_l` weights are licensed for non-commercial research only.

---

## Running with Docker (Local or Cloud GPU Stack)

`docker/` provides three image definitions and a profile-driven `docker-compose.yml`:

| File | Purpose |
|---|---|
| `docker/Dockerfile.mvd` | `mvd` CLI + `ffmpeg` + MediaPipe QC + `faster-whisper` + Streamlit review UI (`:8501`) |
| `docker/Dockerfile.comfy` | Multi-target ComfyUI image: `gpu` target (NVIDIA CUDA + `--reserve-vram`) and `cpu` target (for laptops using paid cloud API nodes) |
| `docker/Dockerfile.allinone` | Single-container all-in-one image (ComfyUI + Ollama + `mvd`) for single-VM cloud GPU instances (`A100 80GB`, `RTX 6000 Ada 48GB`) with persistent volume caching (`MVD_VOLUME_PATH`, default `/data`) |

Pick and choose which services run via `docker compose --profile`:

```bash
# 1. Full local NVIDIA desktop/laptop stack (GPU ComfyUI + GPU Ollama + Review UI)
COMFY_URL=http://comfy-gpu:8188 MVD_LLM_URL=http://ollama-gpu:11434 \
  docker compose --profile comfy-gpu --profile llm-gpu --profile ui up -d

# 2. Laptop using paid ComfyUI API nodes (CPU ComfyUI) + local Ollama
COMFY_URL=http://comfy-cpu:8188 MVD_LLM_URL=http://ollama:11434 \
  docker compose --profile comfy-cpu --profile llm --profile ui up -d

# 3. Laptop using cloud LLM (Claude/Gemini/OpenAI) + CPU ComfyUI
COMFY_URL=http://comfy-cpu:8188 \
  docker compose --profile comfy-cpu --profile ui up -d

# 4. Run any CLI command inside the mvd container
docker compose run --rm mvd validate -p /workspace/projects/my-song
```

*(On Windows PowerShell, set environment variables before running `docker compose`, e.g. `$env:COMFY_URL = "http://comfy-gpu:8188"; $env:MVD_LLM_URL = "http://ollama-gpu:11434"; docker compose --profile comfy-gpu --profile llm-gpu --profile ui up -d`.)*

**Pinning, ComfyUI version-agnosticism & loopback binding:** The optional Dockerfiles pin tested upstream tags (`COMFYUI_REF=v0.45.4`, `OLLAMA_VERSION=0.40.16`, `CURL_IMAGE_TAG=8.23.0`) for reproducible container builds (override at build time with `--build-arg COMFYUI_REF=<tag>`), and `Dockerfile.mvd` uses `constraints.txt`. `mvdirector` itself is **ComfyUI version-agnostic** and works against any local, home-LAN, or cloud ComfyUI installation using standard API-format workflow exports (see [ComfyUI Version Compatibility](workflows.md#comfyui-version-compatibility)). `docker-compose.yml` binds all published ports (`8188`, `11434`, `8501`) to `127.0.0.1` by default; override `MVD_BIND_ADDR` (for example on a trusted private network) only if you intentionally want LAN access.

**Apple Silicon (M1–M4) MacBooks:** Docker Desktop on macOS runs containers inside a Linux VM without Apple Metal/MPS GPU passthrough. To use your Mac's GPU for local open-weight models (`qwen2.5vl:7b`, `gemma3:12b`), run Ollama and ComfyUI natively on macOS and point the `mvd` container at `COMFY_URL=http://host.docker.internal:8188` and `MVD_LLM_URL=http://host.docker.internal:11434`.

**Remote ComfyUI & proxy notes.** Some hosted HTTP proxies reject Python's default HTTP library `User-Agent` (`python-httpx/...` / `Python-urllib/...`) with 403. `ComfyClient` automatically sends `User-Agent: mvdirector/<version>` by default and respects any custom `User-Agent` header configured in `comfy.headers` (if a specific proxy requires a custom string, set it there). Fresh cloud container disks also start without your model files or symlinks, so run `mvd validate` before a long job.

---

## Choosing the Language Model

One model does every text task: writing a Suno brief (`mvd brief`), planning the visual bible and shots (`mvd plan`), planning 3D blocking previz (`mvd previz generate`), judging stills and clips, and rewriting prompts. You pick it once in `project.yaml`.

```yaml
llm:
  provider: ollama           # ollama (local) | anthropic | openai | gemini | openai_compatible
  model: gemma3:12b          # any model name your provider offers; required for every
                             #   provider except ollama
  base_url: null             # server address (or set MVD_LLM_URL); required for openai_compatible
  api_key_env: null          # override the key variable (MVD_* or listed in MVD_ALLOWED_ENV_VARS)
  max_tokens: 8000
  shots_per_request: 20      # use 5 to 10 for small models
  timeout_seconds: 600       # a local model can be slow when the GPU is busy
  json_mode: auto            # auto | off | object | schema
  context_tokens: null       # ollama: context window to request (default 16384)
```

| Provider | For | Key variable | Default address |
|---|---|---|---|
| `anthropic` | Claude models | `ANTHROPIC_API_KEY` | the SDK default |
| `openai` | OpenAI models | `OPENAI_API_KEY` | `https://api.openai.com/v1` |
| `gemini` | Gemini models | `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) | Google's Generative Language API |
| `ollama` | Local or remote models served by Ollama (e.g. `qwen2.5vl:7b`, `gemma3:12b`, or any vision model) | none | `${MVD_LLM_URL:-http://localhost:11434}` |
| `openai_compatible` | LM Studio, llama.cpp server, vLLM, OpenRouter and similar | optional (`api_key_env`) | **`base_url` or `MVD_LLM_URL` is required** |

The default is a local Ollama model (`gemma3:12b`), so nothing leaves your machine and no API key is needed. Install [Ollama](https://ollama.com), then run `ollama pull gemma3:12b` and `mvd llm-check`. Model names change often, so use the name your provider or local server lists. `llm.model` has a default only for `ollama`. For any other provider the config is rejected until you name a model, so a forgotten model never reaches the server as an Ollama name. Examples:

```yaml
llm: {provider: anthropic, model: <claude model name>}
llm: {provider: gemini, model: <gemini model name>}
llm: {provider: openai, model: <openai model name>}
llm: {provider: ollama, model: <a local model that accepts images>}
llm: {provider: openai_compatible, base_url: "http://localhost:1234/v1", model: <loaded model>}
```

**Test it first.** `mvd llm-check` sends one tiny JSON request and one request with a test image (a blue square) and reports each result, so a wrong key, model name or server address, or a text-only model, shows up in seconds instead of mid-run. It exits with an error if the JSON test fails, or if the image test fails while `qc.vision_enabled` is true. With `qc.vision_enabled: false` the image test is skipped. `mvd validate` also checks that the key variable is set.

**The model should accept images.** The vision checks send frames and the reference sheet. With a text-only model, set `qc.vision_enabled: false`. Planning and prompt rewrites still work, but you lose the checks that catch a wrong costume, a mirrored mark and malformed hands. If you forget, the app turns the vision checks off after the server rejects the first image, with one warning and a note in each affected check, instead of failing every attempt.

**Audio-capable models (optional).** Backends that accept audio input (such as `gemini`) automatically receive a compact mono MP3 of `inputs/mix.wav` during `mvd prepare` and `mvd plan` to deepen musical-style and song-meaning interpretation. Other providers automatically fall back to text-only prompts enriched with the extracted acoustic `musical_features` summary.

### Local Models

- **Verified local setup (Ollama), the default.** Planning, vision checks and prompt rewrites ran on a local Ollama model with no API key. `llm.timeout_seconds` defaults to 600 so a slow or busy GPU does not time out:

  ```yaml
  llm:
    model: gemma3:12b        # accepts images; use whatever your Ollama lists
  ```

  Run `mvd llm-check` first. Install with `pip install -e ".[qc,align,ui,dev]"`; the `anthropic` extra is only needed for `llm.provider: anthropic`.
- **One Ollama model at a time.** If another job holds a large model in Ollama, the vision checks queue behind it and time out, and a model that does not fit in memory alongside it makes things worse (a 21 GB model next to `gemma3:12b` did not fit in 39 GB). Pause the other job, or restart Ollama if requests hang.
- Ollama's default context window is small, and a server may silently drop the start of a prompt that does not fit. The `ollama` provider sets the window per request (`16384` by default, `llm.context_tokens` to change it) and warns when the server reports that the prompt filled it. The `openai_compatible` provider cannot set it: configure the window in the server.
- Use fewer shots per request (`5` to `10`) so each reply stays short and valid.
- A model that writes `<think>` reasoning before its answer is handled: the reasoning is ignored.
- Small differences in how a model spells the shot choices are accepted ("Close-up" for `close_up`, "B-roll" for `broll`). A value that means nothing is still rejected.
- Small vision models are usually more lenient than large ones, so expect more false passes from the vision checks. Compare against clips you have labeled before trusting them.

### How Failures Behave

- Invalid JSON: the model is asked once to correct it, quoting the error.
- A reply cut off at `max_tokens`: an error that tells you to raise `llm.max_tokens` or lower `llm.shots_per_request`. It is not retried, since another try would be cut off the same way.
- A shot batch the model cannot direct: those shots get rule-based direction and a warning, and planning continues. A failed visual bible stops planning, because everything depends on it.
- Rate limits and server errors are retried three times, except a server error that says the model cannot read images, which is recognised at once and not retried. A missing key stops the command at the start and names the variable to set.
- **Before a generation run** the app sends the model one tiny request. If the server is unreachable, the key is rejected or the model does not exist, `mvd stills` and `mvd run` stop before any GPU time is spent, and say how to fix it or to set `llm.enabled: false`.
- **During a run**, if the model is unavailable three times in a row (unreachable, key rejected, model not found, or a timeout), the vision checks turn off for the rest of the run with one warning, and the affected attempts are recorded with those checks skipped. An invalid reply from a working model never counts toward that: the next request may be fine. Generate the affected shots again once the model is fixed.
- A server that rejects `response_format` is asked again without it, once, and the app remembers.

### Privacy and Cost

With a cloud provider, your lyrics, style brief and shot descriptions are sent to it, and so are generated frames and reference sheets while vision checks are on. A local model keeps everything on your machine. The quality thresholds were not calibrated for any particular model, so re-check them on labeled clips after you switch.
