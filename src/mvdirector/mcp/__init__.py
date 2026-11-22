# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Model Context Protocol (MCP) server exposing ``mvdirector.api`` over stdio and HTTP."""

from __future__ import annotations

from mvdirector.mcp.jobs import JobCancelled, JobManager
from mvdirector.mcp.server import (
    SKILL_RESOURCE_URI,
    MCPServer,
    build_http_server,
    handle_http_request,
    is_loopback_host,
    load_skill_markdown,
    serve_http,
    serve_stdio,
)
from mvdirector.mcp.tools import WorkspaceGuard

__all__ = [
    "JobCancelled",
    "JobManager",
    "MCPServer",
    "SKILL_RESOURCE_URI",
    "WorkspaceGuard",
    "build_http_server",
    "handle_http_request",
    "is_loopback_host",
    "load_skill_markdown",
    "serve_http",
    "serve_stdio",
]
