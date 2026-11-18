#!/usr/bin/env bash
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

set -euo pipefail

# Persist models on RunPod's network volume (/runpod-volume) when mounted so cold starts
# do not re-download LLM or ComfyUI weights.
VOLUME_ROOT="${RUNPOD_VOLUME_PATH:-/runpod-volume}"
if [ -d "${VOLUME_ROOT}" ]; then
    export OLLAMA_MODELS="${OLLAMA_MODELS:-${VOLUME_ROOT}/ollama}"
    export HF_HOME="${HF_HOME:-${VOLUME_ROOT}/huggingface}"
    mkdir -p "${OLLAMA_MODELS}" "${HF_HOME}" "${VOLUME_ROOT}/comfy-models"
fi

# 1. Optionally start local Ollama server and pull the configured vision-LLM (Qwen-VL / Gemma 3).
if [ "${ENABLE_OLLAMA:-true}" = "true" ]; then
    export OLLAMA_HOST="${OLLAMA_HOST:-0.0.0.0:11434}"
    export OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:--1}"
    echo "[entrypoint] Starting Ollama on ${OLLAMA_HOST}..."
    ollama serve &
    OLLAMA_PID=$!

    for _ in $(seq 1 30); do
        if curl -sf "http://127.0.0.1:11434/api/tags" >/dev/null 2>&1; then
            break
        fi
        sleep 1
    done

    if [ -n "${MVD_OLLAMA_MODEL:-}" ]; then
        echo "[entrypoint] Ensuring Ollama model '${MVD_OLLAMA_MODEL}' is pulled..."
        ollama pull "${MVD_OLLAMA_MODEL}" || echo "[entrypoint] Warning: failed to pull ${MVD_OLLAMA_MODEL}"
    fi
fi

# 2. Start ComfyUI (unless disabled), reserving VRAM for the local LLM on the same GPU.
if [ "${ENABLE_COMFY:-true}" = "true" ]; then
    RESERVE_ARGS=""
    if [ "${ENABLE_OLLAMA:-true}" = "true" ] && [ -n "${COMFY_RESERVE_VRAM:-}" ]; then
        RESERVE_ARGS="--reserve-vram ${COMFY_RESERVE_VRAM}"
    fi
    echo "[entrypoint] Starting ComfyUI on port ${COMFY_PORT:-8188} (${RESERVE_ARGS})..."
    cd /opt/ComfyUI
    python3 main.py --listen 0.0.0.0 --port "${COMFY_PORT:-8188}" ${RESERVE_ARGS} ${COMFY_EXTRA_ARGS:-} &
fi

# If custom command arguments were passed (e.g., `mvd run`), execute them; otherwise wait on services.
if [ "$#" -gt 0 ]; then
    exec "$@"
fi

wait -n
