# Architecture, Development & Limitations

## Code Map

```text
src/mvdirector/
  api.py              service layer: one function per stage; the CLI and other callers use it
  cli.py              thin typer wrapper over api.py: options in, text out
  config.py           project.yaml schema, defaults and allowed env-var expansion
  models.py           Analysis and Manifest (persisted state, safe IDs and schema version checks)
  project.py          paths + root containment; delegates state to a StateStore and assets to
                      an AssetStorage
  store.py            StateStore / AssetStorage protocols and the local file + PID-lock defaults
  errors.py           MVDError hierarchy for expected operational failures
  media.py            ffmpeg / ffprobe wrappers (including black-screen audio_video packaging)
  llm/                LLMClient protocol, request loop (base.py), shared HTTP/retry (http.py),
                      one backend per provider, factory.py (build_llm), check.py (mvd llm-check)
  audio/              analysis.py (beats, vocals), lyrics.py (parsing, transcription, alignment),
                      stems.py (Demucs stem separation), prepare.py (`mvd prepare` audio-only ingest)
  planning/           segmenter.py (timing, dynamic pacing, sections), planner.py (direction),
                      prompts.py (prompt text), brief.py (`mvd brief`), previz.py (3D camera plans)
  comfy/              workflow.py (templates + bindings), client.py (HTTP),
                      provider.py (ComfyEndpointProvider: which endpoint serves a stage)
  qc/                 scoring, common.py, image and video gates, lipsync maths, syncnet
                      (LatentSync StableSyncNet + MediaPipe), faces, vision (judge + prompt),
                      factory
  retry.py            retry policy
  report.py           per-check tallies for `mvd qc-report`
  cleanup.py          superseded and orphaned generations for `mvd clean`
  calibrate.py        threshold suggestions from labels for `mvd calibrate`
  generation/         runner.py (generate -> QC -> retry loop), candidates.py,
                      reference.py, plates.py, props.py, previz.py, upscale.py, preflight.py
  review/             policy.py (queue and decisions), app.py (Streamlit)
  assemble.py         final cut
  templates/          files copied by `mvd init`
```

## Architectural Conventions

- **State lives in the manifest**, never in memory between commands. Writes are atomic via `Project.edit_manifest()`.
- **Progress is an event stream.** `ShotRunner` takes an `observer` callback and emits a `RunEvent` for every phase, attempt, pass and exhaustion. The CLI prints them; anything building a UI subscribes to them instead of parsing logs.
- **External systems sit behind protocols** (`LLMClient`, `FaceAnalyzer`, `MouthTracker`) or injectable transports (`ComfyClient`), so tests use fakes.
- **Expected failures raise `MVDError` subclasses** (`errors.py`); the CLI prints them without a traceback. Anything else is a bug and shows one.
- **Thresholds and prompt text are data:** `config.py`, `planning/prompts.py` and `qc/vision.py`.

---

## Development

```bash
pip install -c constraints.txt -e ".[dev,anthropic]"   # the versions CI and the author tested
pytest                 # needs ffmpeg for the pipeline tests
ruff check . && ruff format --check .
mypy                   # type check; keep clean
mvd schema schemas     # after changing models.py; a test fails if schemas are stale
```

Changing `models.py` changes the manifest format. Once a release has shipped, bump `MANIFEST_SCHEMA_VERSION` and write an in-memory migration so existing projects keep working.

---

## Extending

- **Another face or lip-sync backend:** implement `FaceAnalyzer` / `MouthTracker` (`qc/faces.py`) or `LipSyncScorer` (`qc/lipsync.py`, as `StableSyncNetScorer` in `qc/syncnet.py` does) and construct it in `qc/factory.py`.
- **A new check:** return a `CheckResult` from the image or video gate and give it a weight under `qc.weights`.
- **A new retry strategy:** add it to `RetryStrategy` and handle it in `RetryPolicy.prepare`.
- **Another model provider:** subclass `ChatBackend` in `llm/` (implement `_send`, translating the neutral content blocks) and add it to `PROVIDERS` and `build_llm` in `llm/factory.py`.

---

## Known Limitations

- **Beat/bar grid assumptions:** Bars assume the first detected beat is a downbeat and a fixed `beats_per_bar`. If cuts feel a beat early or late, edit `bars` in `analysis.json` and re-run `mvd plan`.
- **Section-aware pacing:** By default on single-character songs (`video.dynamic_pacing: false`), shots are a fixed number of bars long. Enable `video.dynamic_pacing: true` in `project.yaml` to cut faster on high-intensity bars and snap cuts to `[Section]` transitions; songs whose lyric section tags name multiple singers automatically snap bar cuts to lyric section transitions before calling the LLM so adjacent vocalists do not share a shot. Only explicit performer cues count (`[Verse 1: Elias]`, `[Chorus (Odalie)]`, `[Female Verse]`, `[Singer 2]`, or a bare `[Elias]`); structural tags such as `[Chorus x2]`, `[Guitar Solo]` or `[Verse 1 - angry]` do not. Set `video.snap_sections: always` or `never` to override detection. Cuts still land on bar lines rather than mid-bar lyric phrases.
- **Vocal bleed in AI stems:** Suno stems are AI-separated. Bleed in the vocal stem can mark instrumental passages as sung; check `vocal_windows` in `analysis.json` if shot types look wrong.
- **Costume changes depend on the image-edit model:** Stills are edits of a reference sheet; when you lock a separate sheet per wardrobe state, each shot starts from its own costume's front panel. Without a per-state sheet, the prompt says what is absent (for example `"No cape"`), and `vision_character` catches mistakes at the cost of extra retries.
- **Vision judge token overhead:** `vision_character` adds the reference sheet image (and optional hero prop image) to each call.
- **Sheet panels are cut by position:** The image model is asked for four equal panels and the code cuts the image into equal parts. If it draws uneven panels, the crops will be off. Check `assets/reference/<state>/` after locking, or lock a hand-made sheet with `--single`.
- **Multi-image editing:** Whether a specific ComfyUI image-edit workflow follows multi-reference wording (`prose` vs `tags`) and accepts several input images (`reference_image`, `set_image`, `prop_image`) depends on the underlying model/workflow.
- **Sequential generation:** Generation runs sequentially against a single ComfyUI endpoint.
- **Re-planning overwrites history:** `mvd plan --overwrite` discards generation history. Edit `manifest.json` by hand to change direction for individual shots instead.
