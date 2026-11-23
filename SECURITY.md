# Security Policy

## Supported Versions

| Version | Supported |
| ------- | --------- |
| `0.8.x` (Beta) | Yes |
| `< 0.8.0b1` | No |

## Reporting a Vulnerability

Please **do not** report security vulnerabilities through public GitHub issues.

Instead, use **[GitHub Private Vulnerability Reporting](https://github.com/sherwinlu/mvdirector/security/advisories/new)** on this repository (under the **Security** tab -> **Report a vulnerability**) to disclose the issue privately. Include:

- A description of the vulnerability and its potential impact.
- Steps or a minimal configuration/manifest that reproduces the issue.
- Affected version or commit hash.

You can expect an acknowledgment within 72 hours and a coordinated fix before public disclosure.

---

## Built-in Security Controls

`mvdirector` processes local media files, LLM-generated JSON responses, user-supplied ComfyUI workflow bindings, and remote HTTP responses. The codebase enforces the following boundaries:

1. **Project Root Containment (`Project.path`, `Project.rel`, `WorkflowTemplate.load`):**
   All relative asset paths, input paths, and workflow graph paths inside `*.bindings.json` are resolved and checked with `Path.is_relative_to(project.root)`. Any path that uses `..` or absolute paths to escape the project root raises `ConfigError` or `WorkflowBindingError`.
2. **Safe Identifier Validation (`SAFE_ID`):**
   Identifiers interpolated into filenames (`Shot.id`, `Shot.character`, `Shot.characters`, `Shot.wardrobe`, `Shot.character_wardrobes`, `WardrobeState.id`, `Manifest.characters` keys, `Character.references` keys, and LLM reply IDs) must match `^[A-Za-z0-9][A-Za-z0-9_-]*$` at both the Pydantic schema boundary and the generation boundary.
3. **Strict Environment Variable Expansion (`is_allowed_env_var`, `is_allowed_api_key_env`):**
   `${NAME}` expansion only runs on connection fields (`comfy.base_url`, `comfy.headers`, `llm.base_url`) in `project.yaml` and on `defaults` in `*.bindings.json`, and is restricted to `MVD_*`, `COMFY_*`, and any non-blocked names explicitly opted into via `MVD_ALLOWED_ENV_VARS` (comma-separated). LLM provider keys (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`, `GOOGLE_API_KEY`) and unrelated environment variables (such as `DATABASE_URL`) are rejected by `expand_env` by default so they cannot be interpolated into `comfy.headers` or workflow defaults. `llm.api_key_env` is validated separately by `is_allowed_api_key_env`: because the same file controls `llm.base_url`, only the standard provider key names, `MVD_*` names, and names listed in `MVD_ALLOWED_ENV_VARS` are accepted, so an arbitrary `*_KEY` / `*_TOKEN` credential cannot be sent as a bearer token to a server named in the YAML.
4. **Subprocess & FFmpeg Hardening (`media.py`, `assemble.py`):**
   All external tools (`ffmpeg`, `ffprobe`) are invoked as argument lists without `shell=True`, with resolved paths so filenames starting with `-` cannot be parsed as CLI flags, and `ffmpeg -f concat` runs without `-safe 0`.
5. **ComfyUI Output & Upload Hardening (`ComfyClient.download_output`, `ComfyClient.upload`):**
   Server-returned `filename` and `subfolder` fields are checked against path traversal before requesting `/view`, downloads stream in chunks to a temporary `.part` file and abort at a 512 MiB cap before atomic rename, uploads isolate each project under a portable `mvdirector/<slug>_<hash8>` subfolder (independent of absolute host path), and `POST /prompt` skips `ReadTimeout`/`500` retries so timeouts do not queue duplicate GPU jobs.
6. **MCP Server Hardening (`mvdirector.mcp`):**
   `mvd mcp --http` binds to `127.0.0.1` by default, refuses non-loopback binds without a bearer token (`--token` / `MVD_MCP_TOKEN`, compared with `hmac.compare_digest`) unless `--allow-unauthenticated` is explicitly passed, rejects non-loopback `Host` (on loopback binds) and `Origin` headers to block DNS rebinding and CSRF, caps request bodies at 1 MiB, and confines all project and file path arguments to the configured `--workspace` / `--project` root(s).

---

## Operational Security Guidelines for Users

- **Never expose unauthenticated ComfyUI (`:8188`), Ollama (`:11434`), or the Streamlit Review UI (`:8501`) to untrusted networks or the public internet.** Neither ComfyUI, Ollama, nor the review UI includes built-in user authentication. `docker-compose.yml` binds host ports to `127.0.0.1` by default and `mvd review` binds Streamlit to `127.0.0.1` locally; when running ComfyUI on another computer in your home network or on a remote cloud server, keep it on a private LAN/Tailscale network, use SSH port forwarding, or put it behind HTTPS with an `Authorization` header in `comfy.headers`.
- **Treat ComfyUI workflow files (`*.api.json`) as executable code.** Custom nodes in ComfyUI can execute arbitrary Python on the GPU host. Only run workflow graphs you exported yourself or audited from a trusted source.
- **Review model weight licenses before commercial use.** By default `qc.face_backend: mediapipe` uses Apache-2.0 components; opt-in `insightface` (`buffalo_l`) is non-commercial research only, and LatentSync weights carry an OpenRAIL++ license with use-case restrictions that should be reviewed prior to commercial deployment.
- **Keep API keys in environment variables.** Never commit real API keys or tokens into `project.yaml`. Reference LLM keys by variable name via `llm.api_key_env` (e.g. `ANTHROPIC_API_KEY`), and reference ComfyUI connection tokens via `COMFY_*` or `MVD_*` (e.g. `${COMFY_TOKEN}`).
