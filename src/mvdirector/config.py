# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Project configuration, loaded from ``project.yaml`` and validated with pydantic.

Thresholds live here, not in code, so they can be calibrated per project.
Connection and workflow-default strings may reference allowed environment variables
(``MVD_*`` or ``COMFY_*``) as ``${NAME}`` or ``${NAME:-default}``.
"""

from __future__ import annotations

import os
import re
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from mvdirector.errors import ConfigError
from mvdirector.models import Framing, ProjectMode, RetryStrategy, VideoStyle

CONFIG_FILENAME = "project.yaml"
_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_ALLOWED_ENV_PREFIXES = ("MVD_", "COMFY_")
_ALLOWED_LLM_KEY_NAMES = {
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
}
_BLOCKED_ENV_PREFIXES = (
    "AWS_",
    "AZURE_",
    "DOCKER_",
    "GCP_",
    "GH_",
    "GITHUB_",
    "GITLAB_",
    "GPG_",
    "KUBE",
    "NPM_",
    "PYPI_",
    "SSH_",
)


def _extra_allowed_env_vars() -> set[str]:
    """Optional user-configured comma-separated allowlist from ``MVD_ALLOWED_ENV_VARS``."""
    raw = os.environ.get("MVD_ALLOWED_ENV_VARS", "")
    allowed: set[str] = set()
    for item in raw.split(","):
        name = item.strip()
        if not name or not _ENV_NAME.match(name):
            continue
        upper = name.upper()
        if not upper.startswith(_BLOCKED_ENV_PREFIXES):
            allowed.add(upper)
    return allowed


def is_allowed_env_var(name: str) -> bool:
    """Least-privilege allowlist for ``${NAME}`` expansion in connection/workflow fields.

    Only ``MVD_*``, ``COMFY_*``, and names explicitly listed in ``MVD_ALLOWED_ENV_VARS``
    may be expanded into strings such as ``comfy.base_url``, ``comfy.headers``,
    ``llm.base_url``, or workflow ``defaults``. LLM API keys and unrelated secrets
    (such as ``DATABASE_URL``) are never expanded by default.
    """
    if not _ENV_NAME.match(name):
        return False
    upper = name.upper()
    if upper.startswith(_BLOCKED_ENV_PREFIXES):
        return False
    return upper.startswith(_ALLOWED_ENV_PREFIXES) or upper in _extra_allowed_env_vars()


def is_allowed_api_key_env(name: str) -> bool:
    """Validate ``llm.api_key_env``, the variable whose value is sent to ``llm.base_url``.

    Because ``project.yaml`` also controls ``llm.base_url``, this is as sensitive as header
    expansion: only the standard provider key names, ``MVD_*`` names, and names explicitly
    listed in ``MVD_ALLOWED_ENV_VARS`` are accepted. Arbitrary ``*_KEY`` / ``*_TOKEN`` names
    are rejected so an unrelated credential cannot be sent to a server named in the YAML.
    """
    if not _ENV_NAME.match(name):
        return False
    upper = name.upper()
    if upper.startswith(_BLOCKED_ENV_PREFIXES):
        return False
    return (
        upper in _ALLOWED_LLM_KEY_NAMES
        or upper.startswith("MVD_")
        or upper in _extra_allowed_env_vars()
    )


class _Section(BaseModel):
    """Base for every `project.yaml` section.

    ``allow_inf_nan=False`` matters most for the unbounded thresholds: YAML spells NaN as
    ``.nan``, and a NaN ``qc.pass_score`` makes every ``score >= pass_score`` comparison false,
    so every take is rejected and the whole generation budget burns with nothing to show and no
    error anywhere. Bounded fields already rejected it by accident; now all of them do on purpose.
    """

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class InputsConfig(_Section):
    mix: str = "inputs/mix.wav"
    vocals: str = "inputs/vocals.wav"
    drums: str | None = None
    stems_dir: str = "inputs/stems"
    lyrics: str = "inputs/lyrics.txt"
    style: str = "inputs/style.md"
    word_timestamps: str | None = Field(
        default=None,
        description="Optional JSON list of {word, start, end}; skips transcription.",
    )
    tracklist: str | None = Field(
        default=None,
        description=(
            "Optional cue sheet or tracklist (.json, .yaml, .cue, or .txt) used when mode is 'mix'."
        ),
    )


_SAFE_MODEL_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_SAFE_LANG_CODE = re.compile(r"^[a-z]{2,3}$")


class PrepareConfig(_Section):
    stem_model: str = Field(
        default="htdemucs",
        description="Demucs model name used by `mvd prepare` to separate vocals (and drums).",
    )
    stem_device: Literal["auto", "cpu", "cuda", "mps"] = Field(
        default="auto",
        description="PyTorch device for stem separation ('auto' lets Demucs choose cuda/mps/cpu).",
    )
    whisper_model: str = Field(
        default="small",
        description="faster-whisper model size used for lyric transcription and alignment.",
    )
    whisper_language: str | None = Field(
        default=None,
        description="Optional ISO-639-1 language code for transcription (e.g. 'en'); null = auto.",
    )
    whisper_multilingual: bool = Field(
        default=False,
        description=(
            "Detect language per segment in faster-whisper (enable for code-switching or "
            "multi-language songs)."
        ),
    )
    whisper_stall_timeout_seconds: float = Field(
        default=600.0,
        ge=0.0,
        allow_inf_nan=False,
        description=(
            "Abort transcription when faster-whisper produces no segment for this many seconds "
            "(three times as long for the first segment); 0 disables the watchdog."
        ),
    )

    @field_validator("stem_model", "whisper_model")
    @classmethod
    def _check_model_name(cls, value: str) -> str:
        if not _SAFE_MODEL_NAME.match(value):
            raise ValueError(f"Invalid model name {value!r}: use letters, digits, '.', '_' or '-'.")
        return value

    @field_validator("whisper_language")
    @classmethod
    def _check_language(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip().lower()
        if not _SAFE_LANG_CODE.match(cleaned):
            raise ValueError(
                f"Invalid language code {value!r}: expected a 2- or 3-letter ISO code."
            )
        return cleaned

    @model_validator(mode="after")
    def _check_english_only_whisper(self) -> PrepareConfig:
        if self.whisper_model.lower().endswith(".en") and (
            self.whisper_multilingual
            or (self.whisper_language is not None and self.whisper_language != "en")
        ):
            raise ValueError(
                f"Whisper model {self.whisper_model!r} is English-only; use a multilingual model "
                "(for example 'small' or 'large-v3-turbo')."
            )
        return self


_SNAP_ASPECT_TOLERANCE = 0.03  # the default qc aspect_tolerance


def _snap_size(
    width: int, height: int, multiple: int, tolerance: float = _SNAP_ASPECT_TOLERANCE
) -> tuple[int, int]:
    """The (width, height) on the ``multiple`` grid closest to the request.

    Candidates must keep the aspect ratio within ``tolerance``; when none does, each side is
    rounded to the grid on its own.
    """
    if multiple <= 1:
        return (width, height)
    ratio = width / height
    best: tuple[int, tuple[int, int]] | None = None
    first = max(multiple, (width // multiple - 4) * multiple)
    for w in range(first, width + 4 * multiple + 1, multiple):
        centre = round(w / ratio / multiple)
        for h in (max(multiple, (centre + k) * multiple) for k in (-1, 0, 1)):
            if abs(w / h / ratio - 1) > tolerance:
                continue
            distance = abs(w - width) + abs(h - height)
            if best is None or distance < best[0]:
                best = (distance, (w, h))
    if best is not None:
        return best[1]
    return (
        max(multiple, round(width / multiple) * multiple),
        max(multiple, round(height / multiple) * multiple),
    )


class VideoConfig(_Section):
    width: int = Field(default=1280, ge=64)
    height: int = Field(default=720, ge=64)
    draft_width: int | None = Field(
        default=None, ge=64, description="Optional width used when running with --draft."
    )
    draft_height: int | None = Field(
        default=None, ge=64, description="Optional height used when running with --draft."
    )
    size_multiple: int = Field(
        default=1,
        ge=1,
        description=(
            "Grid your workflow renders on (for example 64 for LTX, which halves the size, "
            "snaps it to 32 and upscales 2x). Above 1, generation sizes move to the nearest "
            "width and height on that grid that keep the aspect ratio within 3%, so the "
            "clip comes back at the size the quality check expects. Sizes set in "
            "draft_width and draft_height are used as written."
        ),
    )
    fps: int = Field(default=24, ge=1)
    max_clip_seconds: float = Field(default=8.0, gt=0.0, allow_inf_nan=False)
    min_clip_seconds: float = Field(default=1.5, gt=0.0, allow_inf_nan=False)
    handle_seconds: float = Field(
        default=0.25,
        ge=0.0,
        allow_inf_nan=False,
        description="Extra audio/video on each side of a shot, trimmed at assembly.",
    )
    # Frame count rule: multiple * n + offset (LTX: 8n+1, Wan/Hunyuan: 4n+1, MiniMax/exact: 1n+0).
    frame_multiple: int = Field(default=8, ge=1)
    frame_offset: int = Field(default=1, ge=0)
    beats_per_bar: int = Field(default=4, ge=1)
    cut_mode: Literal["bars", "phrases"] = Field(
        default="bars",
        description=(
            "'bars' cuts on musical bar lines (best for 2-8s LTX/Wan clips); 'phrases' cuts on "
            "vocal phrase and section boundaries (best for 5-15s MiniMax H3 clips)."
        ),
    )
    dynamic_pacing: bool = Field(
        default=False,
        description="Cut faster during high-intensity bars and snap cuts to section boundaries.",
    )
    snap_sections: Literal["auto", "always", "never"] = Field(
        default="auto",
        description=(
            "'auto' snaps bar cuts to lyric section boundaries when the section tags name more "
            "than one singer; 'always' / 'never' override that detection."
        ),
    )
    lipsync_min_vocal_fraction: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
        description="Share of a shot that must be sung to make it a lip-sync shot.",
    )
    lipsync_audio_source: Literal["vocals", "mix"] = "vocals"
    motion_single_take: bool = Field(
        default=True,
        description="Append framing continuity and single-take (no cuts) cues to motion prompts.",
    )
    motion_no_subtitles: bool = Field(
        default=True,
        description=(
            "Append a 'No subtitles, no text.' guard to composed motion prompts "
            "(omitted in mode='mix', where negative text phrasing can prime distilled video "
            "models to render title cards)."
        ),
    )
    motion_no_music: bool = Field(
        default=True,
        description=(
            "Append a 'No music.' guard on generated clips so audio-video models do not invent "
            "competing background music."
        ),
    )
    motion_include_lyrics: bool = Field(
        default=False,
        description=(
            "Embed the shot's lyrics in composed motion prompts on lip-sync shots (useful for "
            "video models like Seedance that read lyrics from the prompt rather than an input)."
        ),
    )
    # Audio-grid alignment, lip-sync offset repair, phrase slicing & finishing options.
    audio_vae_offset_seconds: float = Field(
        default=0.0,
        ge=-0.25,
        le=0.25,
        allow_inf_nan=False,
        description=(
            "Optional VAE front-crop audio shift compensation in seconds (e.g. 0.046 for "
            "LTX-2.3/2.5 or 0.0125 for MiniMax H3; clean-room concept inspired by "
            "ComfyUI-Sonder-Editor)."
        ),
    )
    audio_mirror_pad: bool = Field(
        default=False,
        description=(
            "Pad clip audio tails up to the quantized frame-grid duration using a periodic "
            "triangle-wave mirror gather rather than leaving silence cliffs on joint AV models "
            "(clean-room concept inspired by ComfyUI-Sonder-Editor)."
        ),
    )
    sync_delay_compensation: bool = Field(
        default=False,
        description=(
            "Apply measured per-take lip-sync lag (clamped within handles) to clip in-points at "
            "assembly (inspired by VisualFrisson/ComfyUI-OneShotMV). Opt-in: the default "
            "MediaPipe lag estimate is noisy on sung vocals, and a wrong shift of up to a handle "
            "length (0.25 s by default) is clearly visible. Enable it once lag estimates are "
            "trustworthy, e.g. qc.lipsync_backend: stable_syncnet with a "
            "qc.video.lipsync_min_confidence floor."
        ),
    )
    phrase_dp_slicing: bool = Field(
        default=False,
        description=(
            "Use shortest-path dynamic programming over valid frame-grid lengths to minimize "
            "mid-phrase splits (inspired by JavaDerek/music-video-pipeline)."
        ),
    )
    prefer_vocal_onset: bool = Field(
        default=False,
        description=(
            "Locally shift cut boundaries (without cascading) to minimize leading vocal dead air "
            "on lip-sync shots (inspired by JavaDerek/music-video-pipeline)."
        ),
    )
    color_drift_correction: bool = Field(
        default=False,
        description=(
            "Apply ridge-regularized 3x4 RGB affine color-drift cancellation across chained "
            "shots during assembly (clean-room concept inspired by ComfyUI-Sonder-Editor)."
        ),
    )
    transition: Literal["cut", "blend", "whip", "flash"] = Field(
        default="cut",
        description=(
            "Standard-mode transition between shots; non-'cut' values borrow spare tail handle "
            "frames so cuts stay beat-locked (inspired by thefullnacho/on-the-one)."
        ),
    )
    transition_seconds: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
        description="Overlap duration in seconds for beat-preserving xfade transitions.",
    )
    color_grade: Literal[
        "none", "portra_dusk", "cinestill_night", "faded_pastel", "neutral_film"
    ] = Field(
        default="none",
        description=(
            "Optional RGB-space (gbrp) film color grade applied during assembly "
            "(inspired by hexeosis-prime/nightshift)."
        ),
    )
    film_grain: float = Field(
        default=0.0,
        ge=0.0,
        le=25.0,
        allow_inf_nan=False,
        description="Optional subtle film grain strength applied during assembly (0 disables).",
    )
    strip_metadata: bool = Field(
        default=True,
        description=(
            "Strip container/workflow metadata from assembled MP4s "
            "(inspired by hexeosis-prime/nightshift)."
        ),
    )

    @model_validator(mode="after")
    def _check_draft_dimensions(self) -> VideoConfig:
        if (self.draft_width is None) != (self.draft_height is None):
            raise ValueError("Set both video.draft_width and video.draft_height together.")
        if self.min_clip_seconds > self.max_clip_seconds:
            raise ValueError(
                f"video.min_clip_seconds ({self.min_clip_seconds}) must not exceed "
                f"video.max_clip_seconds ({self.max_clip_seconds})."
            )
        for label, val in (
            ("video.width", self.width),
            ("video.height", self.height),
            ("video.draft_width", self.draft_width),
            ("video.draft_height", self.draft_height),
        ):
            if val is not None and val % 2 != 0:
                raise ValueError(f"{label} must be an even integer, got {val}.")
        return self

    def generation_size(self, draft: bool = False) -> tuple[int, int]:
        """Target (width, height) for still/video generation."""
        if not draft:
            return _snap_size(self.width, self.height, self.size_multiple)
        if self.draft_width is not None and self.draft_height is not None:
            return (self.draft_width, self.draft_height)
        dw = max(64, (self.width // 2) - ((self.width // 2) % 2))
        dh = max(64, (self.height // 2) - ((self.height // 2) % 2))
        return _snap_size(dw, dh, self.size_multiple)


class ComfyConfig(_Section):
    base_url: str = Field(default="${COMFY_URL:-http://127.0.0.1:8188}", validate_default=True)
    headers: dict[str, str] = Field(default_factory=dict)

    @field_validator("base_url", "headers", mode="before")
    @classmethod
    def _expand_connection_env(cls, value: Any) -> Any:
        """Expand ``${NAME}`` here (defaults included) rather than on the raw YAML."""
        return expand_env(value)

    @field_validator("headers")
    @classmethod
    def _check_headers(cls, value: dict[str, str]) -> dict[str, str]:
        for key, header_val in value.items():
            if not key.strip() or any(ord(ch) < 32 or ord(ch) == 127 for ch in key):
                raise ValueError(f"Invalid ComfyUI header name: {key!r}")
            if any(ord(ch) < 32 or ord(ch) == 127 for ch in header_val):
                raise ValueError(f"ComfyUI header '{key}' must not contain control characters.")
        return value

    request_timeout: float = Field(default=60.0, gt=0.0, allow_inf_nan=False)
    job_timeout: float = Field(default=1800.0, gt=0.0, allow_inf_nan=False)
    poll_interval: float = Field(default=3.0, ge=0.0, allow_inf_nan=False)
    workflows_dir: str = "workflows"
    reference_notation: Literal["prose", "tags"] = Field(
        default="prose",
        description=(
            "How composed prompts refer to multi-image/video workflow inputs: 'prose' "
            "('first/second reference image') or 'tags' ('@image1', '@image2', '@video1')."
        ),
    )
    cache_enabled: bool = Field(
        default=False,
        description=(
            "Opt-in content-addressed cache of ComfyUI outputs under <project>/.comfy_cache "
            "(keyed by canonical graph + input file SHA-256; note that changing model weights "
            "without renaming them is not detected — run `mvd clean` to purge). Inspired by "
            "tysonfarrington-cmyk/xenoform."
        ),
    )
    dedup_uploads: bool = Field(
        default=False,
        description=(
            "Upload reference files under content-addressed SHA-256 names and skip re-uploading "
            "when ComfyUI confirms via HEAD /view (inspired by afj176/open-cinema-studio)."
        ),
    )
    sort_batch_for_cache: bool = Field(
        default=False,
        description=(
            "Sort shot generation order by workflow/setting/prompt prefix to keep ComfyUI text "
            "encoder and reference caches warm (inspired by xenoform and musicvision)."
        ),
    )
    min_free_vram_gb: float | None = Field(
        default=None,
        ge=0.5,
        le=256.0,
        allow_inf_nan=False,
        description=(
            "Optional minimum free GPU VRAM (GiB) probed via /system_stats before queueing a "
            "render job (inspired by JavaDerek/music-video-pipeline)."
        ),
    )
    max_prompt_chars: int | None = Field(
        default=None,
        ge=120,
        le=8000,
        description=(
            "Optional prompt character budget preserving head + continuity tail sentences "
            "(inspired by GeekatplayStudio/ComfyUI-Music-to-Video)."
        ),
    )
    # Logical workflow name -> bindings file inside workflows_dir.
    workflows: dict[str, str] = Field(
        default_factory=lambda: {
            "t2i": "qwen_t2i.bindings.json",
            "still": "qwen_edit_still.bindings.json",
            "lipsync": "ltx_lipsync.bindings.json",
            "i2v": "ltx_i2v.bindings.json",
        }
    )


LLMProvider = Literal["anthropic", "openai", "gemini", "ollama", "openai_compatible"]
# The default model is a local one, so no API key or network service is needed out of the box.
DEFAULT_LLM_PROVIDER: LLMProvider = "ollama"
DEFAULT_LLM_MODEL = "gemma3:12b"


class LLMConfig(_Section):
    """One model for every text task: planning, vision checks and prompt rewrites."""

    enabled: bool = True
    provider: LLMProvider = Field(
        default=DEFAULT_LLM_PROVIDER,
        description="ollama (native API, local, the default), anthropic, openai, gemini, or "
        "openai_compatible (LM Studio, llama.cpp, vLLM, OpenRouter and similar).",
    )
    model: str = DEFAULT_LLM_MODEL
    base_url: str | None = Field(
        default=None,
        description="Server address. Required for openai_compatible; optional for other providers.",
    )
    api_key_env: str | None = Field(
        default=None, description="Name of the environment variable that holds the API key."
    )
    max_tokens: int = Field(default=8000, ge=256)
    shots_per_request: int = Field(default=20, ge=1)
    timeout_seconds: float = Field(default=600.0, gt=0, allow_inf_nan=False)
    json_mode: Literal["auto", "off", "object", "schema"] = Field(
        default="auto",
        description="How to ask the server for JSON: 'object' (valid JSON), 'schema' (JSON "
        "Schema, where supported), 'off' (prompt only). 'auto' picks a safe default per provider.",
    )
    context_tokens: int | None = Field(
        default=None,
        ge=1024,
        description="Context window to request. Used by ollama (default 16384) and for warnings.",
    )

    @field_validator("base_url", mode="before")
    @classmethod
    def _expand_base_url_env(cls, value: Any) -> Any:
        return expand_env(value)

    @model_validator(mode="after")
    def _check_provider(self) -> LLMConfig:
        if not self.model.strip():
            raise ValueError("llm.model must not be empty")
        if self.provider != DEFAULT_LLM_PROVIDER and "model" not in self.model_fields_set:
            # The default names an Ollama model, which no other provider has.
            raise ValueError(
                f"llm.model must be set when llm.provider is '{self.provider}'. Use the model "
                "name your provider or server lists."
            )
        if self.provider == "openai_compatible" and not self.base_url:
            env_url = os.environ.get("MVD_LLM_URL", "").strip()
            if env_url:
                self.base_url = env_url
            else:
                raise ValueError(
                    "llm.base_url is required when llm.provider is 'openai_compatible', for "
                    "example http://localhost:1234/v1 (or set MVD_LLM_URL)"
                )
        if self.api_key_env is not None and not is_allowed_api_key_env(self.api_key_env):
            raise ValueError(
                f"llm.api_key_env {self.api_key_env!r} is not allowed. Use a provider key name "
                "(ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY, GOOGLE_API_KEY), an MVD_* "
                "name (for example MVD_LLM_KEY), or a name listed in MVD_ALLOWED_ENV_VARS."
            )
        return self


class PlanningConfig(_Section):
    video_style: VideoStyle = Field(
        default=VideoStyle.LIPSYNC,
        description=(
            "Direction style for standard-mode music videos: 'lipsync' (default: on-camera "
            "singing intercut with performance and b-roll), 'narrative' (no lip-sync: recurring "
            "character(s) in performance shots intercut with b-roll), or 'visual' (no lip-sync "
            "and no recurring characters: 100% b-roll visual/atmospheric film)."
        ),
    )
    wardrobe_monotonic: bool = Field(
        default=True,
        description="A costume change is one-way: once a later wardrobe state appears, "
        "earlier states never return (e.g. a cape taken off stays off).",
    )
    allow_multicut_shots: bool = Field(
        default=False,
        description="Allow the planner to flag high-energy shots as multi-angle montage cuts.",
    )
    lipsync_widest_framing: Framing = Field(
        default=Framing.MEDIUM_CLOSE,
        description=(
            "The widest framing a singing shot may have; wider ones are tightened to it. "
            "On a medium shot the mouth is too small for the lip-sync workflow to drive "
            "or the face tracker to find, and clips came back static."
        ),
    )
    max_lipsync_share: float | None = Field(
        default=0.45,
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
        description=(
            "The most screen time, as a share of the song, that may be the singer singing to "
            "camera. Sung shots over the budget become cutaways (no singer, the vocal plays over "
            "them), starting with the quietest non-chorus/non-hook shots; chorus, hook and "
            "final-line shots stay, and in a duet each singer keeps at least half of their sung "
            "shots on camera. Rough guide: 0.7 for a performance video, 0.45 for a hybrid, 0.2 "
            "for a narrative one. The default is the hybrid mix. None leaves every sung shot on "
            "camera."
        ),
    )


class ReferenceConfig(_Section):
    layout: Literal["row", "grid", "columns"] = Field(
        default="row",
        description=(
            "Reference sheet layout: 'row' (1x4), 'grid' (2x2), or 'columns' (left stacked "
            "headshots + full-height neck-down front/back body panels)."
        ),
    )
    background: Literal["white", "grey"] = Field(
        default="white",
        description=(
            "Studio background for reference sheets: 'white' seamless or 'grey' neutral flat-lit "
            "(avoids clipping on bright or glowing outfits)."
        ),
    )
    width: int | None = Field(
        default=None, ge=64, description="Sheet width. Default depends on layout."
    )
    height: int | None = Field(
        default=None, ge=64, description="Sheet height. Default depends on layout."
    )
    candidates: int = Field(default=4, ge=1)
    video_candidates: int = Field(
        default=2,
        ge=1,
        description="Candidate turnaround reference videos generated per run.",
    )
    panel_inset: float = Field(
        default=0.012,
        ge=0.0,
        lt=0.2,
        description="Share of each panel trimmed on every side, to drop dividers and labels.",
    )

    def size(self) -> tuple[int, int]:
        defaults = {"row": (2048, 896), "grid": (1536, 1536), "columns": (1536, 1024)}
        default = defaults[self.layout]
        return (self.width or default[0], self.height or default[1])


class PlatesConfig(_Section):
    candidates: int = Field(default=3, ge=1)


class PropsConfig(_Section):
    candidates: int = Field(default=3, ge=1)


class ImageQCConfig(_Section):
    aspect_tolerance: float = 0.03
    min_resolution_scale: float = 0.5
    sharpness_bad: float = 40.0
    sharpness_good: float = 150.0
    identity_hard_min: float = 0.30
    identity_good: float = 0.55
    # Minimum and maximum face height as a share of frame height, per framing.
    min_face_fraction: dict[Framing, float] = Field(
        default_factory=lambda: {
            Framing.CLOSE_UP: 0.30,
            Framing.MEDIUM_CLOSE: 0.18,
            Framing.MEDIUM: 0.09,
            Framing.WIDE: 0.03,
        }
    )
    max_face_fraction: dict[Framing, float] = Field(
        default_factory=lambda: {
            Framing.CLOSE_UP: 0.90,
            Framing.MEDIUM_CLOSE: 0.45,
            Framing.MEDIUM: 0.25,
            Framing.WIDE: 0.10,
        }
    )
    broll_max_face_fraction: float = Field(
        default=0.04,
        ge=0.0,
        le=1.0,
        description="Largest face box allowed on a b-roll cutaway before failing 'no_people'.",
    )
    vision_hard_min: float = Field(default=0.50, ge=0.0, le=1.0)
    vision_spec_hard_min: float = Field(default=0.60, ge=0.0, le=1.0)
    vision_character_hard_min: float = Field(default=0.40, ge=0.0, le=1.0)


class VideoQCConfig(_Section):
    duration_tolerance: float = 0.3
    aspect_tolerance: float = 0.03
    min_resolution_scale: float = 0.5
    analysis_width: int = Field(default=512, ge=16)
    # Lip-sync: correlation between mouth opening and vocal loudness.
    # The 0.40 floor suppresses random-noise false passes while the ±0.25s (±6 frames at 24fps)
    # lag window avoids rejecting workflows with a small constant AV offset.
    lipsync_bad: float = 0.25
    lipsync_good: float = 0.65
    lipsync_min: float = 0.40
    lipsync_min_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum SyncNet confidence required before applying automatic sync_delay offset "
            "compensation at assembly."
        ),
    )
    desync_face_fraction_min: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description=(
            "Face height fraction above which lip-sync offset becomes viewer-visible for "
            "desync_risk triage (inspired by JavaDerek/music-video-pipeline)."
        ),
    )
    lipsync_max_lag_seconds: float = Field(default=0.25, ge=0.0, allow_inf_nan=False)
    # Advisory by default: on sung vocals the mouth-vs-loudness correlation did not separate
    # clips a person judged in sync from ones out of sync, so a low score sends the shot to
    # review (with the score shown) instead of failing it. Set true to reject and retry.
    lipsync_blocking: bool = False
    lipsync_framings: list[Framing] = Field(
        default_factory=lambda: [Framing.CLOSE_UP, Framing.MEDIUM_CLOSE, Framing.MEDIUM],
        description="Mouths are too small to measure in wider shots.",
    )
    face_presence_min: float = 0.90
    identity_sample_seconds: float = 0.5
    identity_hard_min: float = 0.25
    identity_good: float = 0.50
    # Mean absolute frame-to-frame difference, 0-1.
    static_max_motion: float = 0.0008
    jump_cut_threshold: float = 0.25
    flicker_good: float = 0.03
    flicker_bad: float = 0.12
    vision_frames: int = Field(default=5, ge=1)
    vision_hard_min: float = Field(default=0.50, ge=0.0, le=1.0)
    vision_spec_hard_min: float = Field(default=0.60, ge=0.0, le=1.0)
    vision_character_hard_min: float = Field(default=0.40, ge=0.0, le=1.0)


class QCConfig(_Section):
    pass_score: float = 0.70
    vision_enabled: bool = True
    vision_blind_observer: bool = Field(
        default=False,
        description=(
            "Run two-pass Blind-Observer / Split-Auditor vision checks (Pass 1 describes the "
            "image without seeing requirements; Pass 2 audits text-only against the spec; "
            "inspired by afj176/open-cinema-studio)."
        ),
    )
    face_backend: Literal["mediapipe", "insightface"] = Field(
        default="mediapipe",
        description=(
            "Face analyzer backend: 'mediapipe' (default, Apache-2.0, commercially permissive; "
            "checks face_present and face_framing while vision_character checks identity) or "
            "'insightface' (requires the 'insightface' extra; adds buffalo_l face-embedding "
            "identity checks, whose pretrained weights are non-commercial research only)."
        ),
    )
    lipsync_backend: Literal["mediapipe", "stable_syncnet"] = Field(
        default="mediapipe",
        description=(
            "Lip-sync scorer backend: 'mediapipe' (default, Apache-2.0, CPU-only mouth-opening "
            "vs. vocal-loudness correlation) or 'stable_syncnet' (ByteDance LatentSync "
            "stable_syncnet.pt under OpenRAIL++ paired with Apache-2.0 MediaPipe Face Detector "
            "for face cropping; requires the 'qc' and 'syncnet' extras)."
        ),
    )
    syncnet_model_path: str | None = Field(
        default=None,
        description=(
            "Optional local path to ByteDance LatentSync's stable_syncnet.pt checkpoint. "
            "When unset, downloaded and cached automatically on first use."
        ),
    )
    syncnet_device: Literal["auto", "cpu", "cuda", "mps"] = Field(
        default="auto",
        description=(
            "PyTorch device for qc.lipsync_backend: stable_syncnet ('auto' picks cuda/mps/cpu)."
        ),
    )
    on_missing_dependency: Literal["skip", "fail"] = Field(
        default="fail",
        description=(
            "'fail' stops the run when required QC extras are missing; "
            "'skip' logs a warning and omits the check."
        ),
    )

    @field_validator("syncnet_model_path", mode="before")
    @classmethod
    def _expand_syncnet_model_path(cls, value: Any) -> Any:
        return expand_env(value)

    weights: dict[str, float] = Field(
        default_factory=lambda: {
            "lipsync": 3.0,
            "identity": 2.0,
            "vision_artifacts": 2.0,
            "vision_spec": 2.0,
            "vision_character": 2.0,
            "temporal": 1.0,
            "sharpness": 0.5,
            "face_framing": 1.0,
        }
    )
    image: ImageQCConfig = Field(default_factory=ImageQCConfig)
    video: VideoQCConfig = Field(default_factory=VideoQCConfig)


class RetryConfig(_Section):
    max_attempts: int = Field(default=4, ge=1)
    strategies: list[RetryStrategy] = Field(
        default_factory=lambda: [
            RetryStrategy.NEW_SEED,
            RetryStrategy.REWRITE_PROMPT,
            RetryStrategy.ADJUST_PARAMS,
        ],
        min_length=1,
        description="Applied in order from attempt 2 on; the last one repeats.",
    )
    max_generations_per_run: int = Field(default=300, ge=1)
    lipsync_fallback: bool = Field(
        default=True,
        description=(
            "When a singing shot's video uses up its attempts, turn it into a performance shot "
            "(the singer on camera, not singing) or b-roll cutaway and render that instead of "
            "leaving the shot stuck. The conversion is flagged for review."
        ),
    )
    lipsync_fallback_mode: Literal["performance", "broll"] = Field(
        default="performance",
        description=(
            "Target shot type when a lip-sync shot exhausts retries ('performance' keeps the "
            "singer on camera silently; 'broll' rescues it with a cutaway when within the "
            "2-cutaway run limit, inspired by VisualFrisson/ComfyUI-OneShotMV)."
        ),
    )
    smart_cutaway_rescue: bool = Field(
        default=False,
        description=(
            "Automatically convert critically weak lip-sync takes below "
            "smart_cutaway_max_correlation into b-roll cutaways when within the 2-cutaway run "
            "limit (inspired by VisualFrisson/ComfyUI-OneShotMV)."
        ),
    )
    smart_cutaway_max_correlation: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="Lip-sync correlation ceiling below which smart_cutaway_rescue triggers.",
    )
    in_place_repair: bool = Field(
        default=False,
        description=(
            "Attempt a targeted in-place image-edit repair on stills that pass face checks but "
            "fail isolated vision_spec/artifact issues, accepting only strict non-regressions "
            "(inspired by afj176/open-cinema-studio)."
        ),
    )
    # Workflow parameters applied by the adjust_params strategy.
    adjust_params: dict[str, dict[str, Any]] = Field(
        default_factory=lambda: {"still": {"steps": 40}, "video": {"steps": 40}}
    )

    @model_validator(mode="after")
    def _no_initial(self) -> RetryConfig:
        if RetryStrategy.INITIAL in self.strategies or not self.strategies:
            raise ValueError("retry.strategies must be non-empty and must not contain 'initial'")
        return self


class ReviewMode(str, Enum):
    LIGHT = "light"  # spot-check a sample of approved shots, plus every exhausted shot
    STRICT = "strict"  # only exhausted shots reach the reviewer


class ReviewConfig(_Section):
    mode: ReviewMode = ReviewMode.LIGHT
    closeup_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    other_rate: float = Field(default=0.25, ge=0.0, le=1.0)
    include_retried: bool = True
    include_near_threshold: bool = True
    near_threshold_margin: float = 0.05


class MixHudConfig(_Section):
    enabled: bool = True
    show_title: bool = True
    watermark: str = ""
    show_spectrum: bool = True
    position: Literal["bottom_left", "bottom_right", "top_left", "top_right"] = "bottom_left"


class MixConfig(_Section):
    default_segment_seconds: float = Field(
        default=240.0,
        ge=30.0,
        le=1800.0,
        allow_inf_nan=False,
        description="Fallback chapter length in seconds when inputs.tracklist is not provided.",
    )
    visual_source: Literal["clips", "stills"] = Field(
        default="clips",
        description=(
            "'clips' generates short loopable t2i+i2v clips per chapter; 'stills' generates only "
            "high-res t2i hero stills and animates them with camera drift and beat pulse."
        ),
    )
    clips_per_segment: int = Field(
        default=3,
        ge=1,
        le=6,
        description="Number of loopable visual units generated per track chapter.",
    )
    clip_seconds: float = Field(
        default=6.0,
        ge=4.0,
        le=8.0,
        allow_inf_nan=False,
        description="Target duration in seconds for each generated loop unit.",
    )
    xfade_seconds: float = Field(
        default=1.0,
        ge=0.25,
        le=3.0,
        allow_inf_nan=False,
        description="Crossfade overlap in seconds inside each chapter's seamless loop cycle.",
    )
    loop_blend: Literal["xfade", "smoothstep"] = Field(
        default="xfade",
        description=(
            "Loop boundary blend curve ('xfade' linear FFmpeg fade or 'smoothstep' S-curve "
            "tail-over-head closure inspired by atomtanstudio/sound-and-vision)."
        ),
    )
    chapter_fade_seconds: float = Field(
        default=0.75,
        ge=0.0,
        le=5.0,
        allow_inf_nan=False,
        description="Fade-in and fade-out duration in seconds at chapter boundaries.",
    )
    beat_pulse: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
        description="Strength of BPM/downbeat-synced subtle visual pulse (0 disables).",
    )
    chapters_sidecar: bool = Field(
        default=True,
        description="Write a YouTube-compatible <output>.chapters.txt sidecar during assembly.",
    )
    hud: MixHudConfig = Field(default_factory=MixHudConfig)

    @model_validator(mode="after")
    def _check_xfade(self) -> MixConfig:
        if self.xfade_seconds * 2.0 >= self.clip_seconds:
            raise ValueError(
                f"mix.xfade_seconds ({self.xfade_seconds}) must be less than half of "
                f"mix.clip_seconds ({self.clip_seconds}) so loop strides stay positive."
            )
        return self


class ProjectConfig(_Section):
    title: str = "Untitled"
    mode: ProjectMode = Field(
        default=ProjectMode.STANDARD,
        description="Pipeline mode: 'standard' (single-track narrative/lip-sync) or 'mix' "
        "(30-90+ minute continuous mixes with per-chapter themes and loop caching).",
    )
    seed: int = 1234
    auto_tune: bool = Field(
        default=True,
        description=(
            "Automatically adjust default transcription, beat-grid, cut-mode, pacing, and "
            "lip-sync settings based on detected track characteristics."
        ),
    )
    inputs: InputsConfig = Field(default_factory=InputsConfig)
    prepare: PrepareConfig = Field(default_factory=PrepareConfig)
    video: VideoConfig = Field(default_factory=VideoConfig)
    mix: MixConfig = Field(default_factory=MixConfig)
    comfy: ComfyConfig = Field(default_factory=ComfyConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    planning: PlanningConfig = Field(default_factory=PlanningConfig)
    reference: ReferenceConfig = Field(default_factory=ReferenceConfig)
    plates: PlatesConfig = Field(default_factory=PlatesConfig)
    props: PropsConfig = Field(default_factory=PropsConfig)
    qc: QCConfig = Field(default_factory=QCConfig)
    retry: RetryConfig = Field(default_factory=RetryConfig)
    review: ReviewConfig = Field(default_factory=ReviewConfig)


def expand_env(value: Any) -> Any:
    """Recursively replace ``${NAME}`` / ``${NAME:-default}`` in strings."""
    if isinstance(value, str):

        def replace(match: re.Match[str]) -> str:
            name, default = match.group(1), match.group(2)
            if not is_allowed_env_var(name):
                raise ConfigError(
                    f"Environment variable '{name}' is not allowed in {CONFIG_FILENAME}. "
                    "Only MVD_*, COMFY_*, or names listed in MVD_ALLOWED_ENV_VARS "
                    "may be expanded."
                )
            resolved = os.environ.get(name, default)
            if resolved is None:
                raise ConfigError(f"Environment variable {name} is not set")
            return resolved

        return _ENV_PATTERN.sub(replace, value)
    if isinstance(value, dict):
        return {key: expand_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [expand_env(item) for item in value]
    return value


def mask_unset_allowed_env_refs(value: Any) -> Any:
    """Replace unset allowed ``${VAR}`` placeholders so validation works before export."""
    if isinstance(value, str):

        def _sub(match: re.Match[str]) -> str:
            var_name = match.group(1)
            if is_allowed_env_var(var_name) and os.environ.get(var_name) is None:
                return "placeholder"
            return match.group(0)

        return _ENV_PATTERN.sub(_sub, value)
    if isinstance(value, dict):
        return {key: mask_unset_allowed_env_refs(item) for key, item in value.items()}
    if isinstance(value, list):
        return [mask_unset_allowed_env_refs(item) for item in value]
    return value


def load_config(path: Path) -> ProjectConfig:
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ConfigError(f"Could not read {path}: {exc}") from exc
    try:
        raw = yaml.safe_load(text) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path} is not valid YAML: {exc}") from exc
    except RecursionError:
        # PyYAML recurses per nesting level, so a deeply nested document exhausts the stack
        # instead of raising YAMLError.
        raise ConfigError(f"{path} nests collections too deeply to parse.") from None
    try:
        return ProjectConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"Invalid configuration in {path}:\n{exc}") from exc
