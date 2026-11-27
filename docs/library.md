# Using mvdirector as a library

The `mvd` command is a thin front end over `mvdirector.api`. Anything the CLI does, a script,
notebook or worker can do by calling the same functions.

```python
from pathlib import Path
from mvdirector import api
from mvdirector.models import AssetKind

project_dir = Path("my-song")
api.analyze(project_dir)
api.plan(project_dir)
result = api.generate(
    project_dir,
    [AssetKind.STILL, AssetKind.VIDEO],
    observer=lambda event: print(event.kind, event.shot_id),
)
print(api.assemble(project_dir))
```

Functions raise `mvdirector.MVDError` (or a subclass) on failure and return data instead of
printing. `generate` reports progress through one callback, `observer`, which receives a
`RunEvent` for each preflight `warning`, the `qc_summary` (active and skipped checks), and every
phase, attempt, pass and exhaustion. That stream is what a service should persist and fan out.

## Public surface

Semver applies. While the version is 0.x, a minor release may still change these names, and the changelog lists every such change. These names are public:

| Name | Purpose |
| --- | --- |
| `mvdirector.api` | One function per pipeline stage (`analyze`, `plan`, `lint_assets`, `generate`, `attempt_stats`, `approve`, `reject`, `assemble`, `clean`, etc. — see `api.__all__`). |
| `Project` | A workspace folder plus its config, state store and asset storage. |
| `StateStore` | Where `analysis` and `manifest` live, plus the single-writer lock. |
| `AssetStorage` | How the workspace is pulled from and pushed to durable storage (e.g., Cloudflare R2 / S3). |
| `ComfyEndpointProvider` | Where each stage gets its `ComfyClient` (including `ComfyClient.interrupt()` for cooperative cancellation). |
| `Manifest`, `Analysis`, `RunEvent`, `MVDError` | Data and error types. |

Everything else, including `mvdirector.generation`, `.qc`, `.planning` and `.comfy.client`, is
internal and may change in any release. If you need something from there, open an issue so it can
be promoted to the public surface.

## Extension points

The engine works on a local workspace folder, because ffmpeg, OpenCV and ComfyUI uploads need
real files. Three small interfaces let a service wrap that without forking the engine.

### `StateStore`

Owns `analysis.json`, `manifest.json`, the writer lock and a stable `project_id`.
`LocalStateStore` (the default) uses files and a PID lock that only sees the local machine.

A store shared between workers must implement:

- **`lock()`**: exclusive write access, raising `ProjectLocked` when another job holds it. The
  lock must expire or be reclaimable when its holder dies (a lease with a heartbeat or TTL),
  otherwise one crashed worker blocks the project forever.
- **`project_id`**: a stable identity (for example a database key) that namespaces ComfyUI
  uploads. The local store persists a random id in `.mvd_id` inside the workspace. If your
  workspace is recreated for every job, return your own id instead, or each job re-uploads its
  inputs into a new subfolder that is never cleaned up. Return `""` to derive one from the config.
- **`save_manifest`** only has to persist what it is given; `Project` stamps `updated_at`.

### `AssetStorage`

`pull()` runs in the caller's thread when a stage that reads or writes workspace files opens
the project through `api`, before any lock is taken, so inputs and earlier assets exist in the
workspace. State-only stages (status, approve, reject, reports) skip it. `push()` runs before every manifest save, so a stored manifest never references an
asset that was not published; if `push()` raises, the manifest is not saved. Stages that write files without saving the
manifest (candidate generation for references, plates and props, and `assemble`) call `push()`
themselves before returning. It must be safe to
call repeatedly. `LocalAssetStorage` does
nothing because the workspace already is the storage.

### `ComfyEndpointProvider`

`lease(project)` returns a context manager that yields a `ComfyClient`. The default
`StaticComfyProvider` uses `comfy.base_url` from the project config. Implement the protocol to
route each stage to a pod pool, check health, enforce concurrency limits or cold-start capacity.
Stages that generate take the writer lock first and then lease, so a provider that waits for
capacity does so while holding the lock. Keep waits short, or fail fast and let the job queue retry.
Pass it as `comfy_provider=` to any api function that talks to ComfyUI.

### Wiring it together

```python
from mvdirector import api, Project

project = Project.open(workspace, state=MyDbState(job_id), storage=MyS3Storage(prefix))
api.generate(project, kinds, comfy_provider=MyCloudComfyRouter())
```

Every `api` function accepts a `Project` in place of a folder path. A `review_mode=` override
applies to that call only; the `Project` you pass in is never modified.

### Entry-point plugins (`plugins`)

Optional downstream extensions can register Python entry points in four groups without modifying
`mvdirector`:

- `mvdirector.qc_checks` (`plugins.qc_checks`) — extra video QC checks `(frames, fps, shot, manifest, audio_slice) -> list[CheckResult]`.
- `mvdirector.audio_conditioners` (`plugins.audio_conditioner`) — conditioning-only vocal DSP for `LIPSYNC` shots (QC still scores the raw slice).
- `mvdirector.pacing` (`plugins.pacing`) — post-processing of cut timestamps, re-validated against min/max clip rules with fallback on error.
- `mvdirector.clip_selectors` (`plugins.clip_selector`) — selects `Attempt.optimal_in_offset` within handle slack on non-lip-sync shots.

Config refers to plugins by registered allowlisted name (`^[a-z][a-z0-9_]{0,63}$`), never by import path.

## Hosted use

The default config trusts the person running the CLI: endpoints, workflow files and `${ENV}`
expansion come from `project.yaml`. A multi-tenant service must build the `Project` and
`ComfyEndpointProvider` itself and never let tenants control `comfy.base_url`, `llm.base_url`,
workflow files or environment expansion.

Several `api` functions accept filesystem paths that are not inside the project: `image`,
`video` and `style_ref` (copied into the project), `labels` (read), and `output` and `directory`
(written). That is right for the CLI. A service must validate or replace them with paths inside a
per-job directory before calling.

The Streamlit review UI (`mvd review`) opens the project folder directly and only works with the
local stores. A hosted product replaces it with its own review UI over `api.approve`,
`api.reject` and `api.load_status`.
