# MCP Server & Agent Integration

`mvdirector` ships with a zero-dependency **Model Context Protocol (MCP)** server (`mvd mcp`) over `mvdirector.api`, plus a bundled Agent Skill (`mvdirector://skill`). It supports both:

- **`stdio` transport** (default) for local coding/director agents (Claude Code, Cursor, Antigravity).
- **Streamable HTTP transport** (`--http`) for driving a remote GPU workstation from a laptop agent without running an LLM on the GPU box.

## Quick Start (stdio)

Run `mvd mcp` from your agent's MCP server configuration:

```json
{
  "mcpServers": {
    "mvdirector": {
      "command": "mvd",
      "args": ["mcp", "--workspace", "/path/to/projects"]
    }
  }
}
```

If you work on a single project, pass `--project /path/to/projects/my-song` so tool calls can omit `project_dir`:

```json
{
  "mcpServers": {
    "mvdirector": {
      "command": "mvd",
      "args": ["mcp", "--project", "/path/to/projects/my-song"]
    }
  }
}
```

## Remote GPU Box (Streamable HTTP)

On the GPU host, start the Streamable HTTP endpoint (`POST /mcp`, `DELETE /mcp`, `GET /health`):

```bash
export MVD_MCP_TOKEN="replace-with-a-strong-random-token"
mvd mcp --http --host 0.0.0.0 --port 8765 --workspace /srv/mvdirector/projects
```

Point your MCP client at `http://<gpu-host>:8765/mcp` with header `Authorization: Bearer <token>`.

### Built-In Security Controls

1. **Loopback by default:** `--http` binds to `127.0.0.1:8765` unless `--host` is explicitly changed.
2. **Mandatory auth on non-loopback binds:** Binding to a non-loopback address (`0.0.0.0`, LAN IP) refuses to start without `--token` / `MVD_MCP_TOKEN` (unless `--allow-unauthenticated` is explicitly passed for a trusted private network). Tokens are verified with constant-time `hmac.compare_digest`.
3. **Workspace containment:** In `--http` mode, all `project_dir` and file path arguments (`image`, `video`, `output`, `audio`, `tracklist`, `labels`, `style_ref`) are strictly confined to `--workspace` (or `--project`, or the current working directory if neither is passed). `..` traversal and symlink escapes are rejected.
4. **DNS rebinding & CSRF guards:** Loopback binds reject non-loopback `Host` headers, and non-loopback `Origin` headers are rejected with `403 Forbidden`. Request bodies are capped at 1 MiB.

## Async Background Jobs

GPU-heavy tools (`mvd_generate`, `mvd_upscale`, `mvd_assemble`, `mvd_generate_references`, `mvd_generate_reference_videos`, `mvd_generate_plates`, `mvd_generate_props`, `mvd_generate_previz`, `mvd_refine_previz`) default to `background: true` and immediately return a `job_id`:

1. Call `mvd_generate(project_dir=..., kinds=["still"])` → returns `{"job_id": "job_...", "status": "queued"}`.
2. Poll `mvd_task_status(job_id="job_...")` to view live `RunEvent` progress, `generations_used`, and the final result.
3. Call `mvd_task_stop(job_id="job_...")` to cancel cooperatively between shots/attempts (completed attempts are saved to `manifest.json`).

Pass `background: false` on any async tool when you want to run synchronously (for example, a quick 1-shot test).

## Tool Reference

| Category | Tools |
| :--- | :--- |
| **Project & Diagnostics** | `mvd_init_project`, `mvd_validate_project`, `mvd_check_llm`, `mvd_status`, `mvd_get_shot`, `mvd_get_manifest_summary`, `mvd_qc_report`, `mvd_calibrate`, `mvd_tune` |
| **Brief, Audio & Plan** | `mvd_brief`, `mvd_prepare`, `mvd_analyze`, `mvd_plan` |
| **Pre-Production Assets** | `mvd_generate_references`, `mvd_lock_reference`, `mvd_generate_reference_videos`, `mvd_lock_reference_video`, `mvd_generate_plates`, `mvd_lock_plate`, `mvd_generate_props`, `mvd_lock_prop`, `mvd_generate_previz`, `mvd_lock_previz`, `mvd_refine_previz` |
| **Generate, Review & Export** | `mvd_generate`, `mvd_upscale`, `mvd_approve`, `mvd_reject`, `mvd_assemble`, `mvd_clean` |
| **Async Job Control** | `mvd_task_status`, `mvd_task_stop` |

## Bundled Agent Skill

The server exposes its workflow guide as MCP resource `mvdirector://skill` (`src/mvdirector/mcp/SKILL.md`), which MCP-capable agents can read via `resources/read` to learn the recommended stage order and operational guardrails.
