# Contributing to mvdirector

Thank you for your interest in contributing to `mvdirector`! This document covers local development setup, quality gates, and architectural conventions.

## Development Setup

Requires Python 3.11+ (Python 3.12 recommended) and `ffmpeg` / `ffprobe` on your `PATH`.

```bash
brew install ffmpeg
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,qc,align,ui]"
```

## Running Checks Before Opening a Pull Request

Every pull request must pass the full test suite, linter, formatter, and type checker:

```bash
# 1. Run the test suite (requires ffmpeg)
pytest

# 2. Check linting and formatting
ruff check .
ruff format --check .

# 3. Run static type checking
mypy
```

## Changing `models.py` (Schemas & Examples)

`schemas/manifest.schema.json`, `schemas/analysis.schema.json`, and `examples/*.example.json` are tested against the Pydantic models in `src/mvdirector/models.py` (`tests/test_schemas.py`).

If you add or modify fields in `src/mvdirector/models.py`:

1. Give new fields sensible defaults so existing manifests remain valid where possible.
2. If you make a breaking change to `Manifest`, bump `MANIFEST_SCHEMA_VERSION` and add a `mode="before"` migration validator so older manifests upgrade cleanly in memory.
3. Regenerate the JSON Schemas and commit the result:
   ```bash
   mvd schema schemas
   ```
4. Update `examples/manifest.example.json` or `examples/analysis.example.json` if needed and run `pytest tests/test_schemas.py`.

## Architectural Conventions

- **State lives in `manifest.json` and `analysis.json`, never in memory between commands.** Always use `project.save_manifest()` or `project.edit_manifest()` so writes go through the `StateStore` and respect the writer lock.
- **Keep external systems behind protocols or injectable transports.** `LLMClient`, `FaceAnalyzer`, `MouthTracker`, and `ComfyClient` (via `httpx.BaseTransport`) allow deterministic unit and integration tests without live cloud APIs or GPUs.
- **Raise `MVDError` subclasses for expected operational errors.** Anything that a user can hit through configuration, network, or media issues should raise an `MVDError` subclass (`src/mvdirector/errors.py`) with an actionable message rather than a raw traceback.
- **Keep prompts and thresholds as data.** Prompt templates belong in `src/mvdirector/planning/prompts.py` and `src/mvdirector/qc/vision.py`; configurable thresholds belong in `src/mvdirector/config.py`.
- **Licensing.** mvdirector is under the MPL-2.0, and contributions are accepted under the same license. Every new source file (`.py` and shell scripts) starts with the three-line MPL notice used in the existing files; `tests/test_license_headers.py` fails if one is missing.
- **Maintain commercial-safe defaults.** Default dependencies and `qc` settings (`qc.face_backend: mediapipe`) must remain under permissive open-source licenses (MIT / BSD / Apache-2.0). Non-commercial research dependencies must stay opt-in.
