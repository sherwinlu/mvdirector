# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Zero-dependency MCP (Model Context Protocol) server over stdio and Streamable HTTP."""

from __future__ import annotations

import hmac
import ipaddress
import json
import logging
import sys
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from typing import Any, TextIO
from urllib.parse import urlparse

from mvdirector import __version__
from mvdirector.comfy.provider import ComfyEndpointProvider
from mvdirector.errors import ConfigError, MVDError
from mvdirector.mcp.jobs import JobManager
from mvdirector.mcp.tools import ToolSpec, WorkspaceGuard, build_tool_registry

log = logging.getLogger(__name__)

SUPPORTED_PROTOCOL_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")
DEFAULT_PROTOCOL_VERSION = "2025-03-26"
SKILL_RESOURCE_URI = "mvdirector://skill"
MAX_HTTP_BODY_BYTES = 1_048_576  # 1 MiB

_SERVER_INSTRUCTIONS = (
    "mvdirector MCP server: plan, generate, quality-check, review, and assemble music videos. "
    "GPU-heavy stages (mvd_generate, mvd_upscale, mvd_assemble, reference/plate/prop/previz "
    "generation) run asynchronously by default and return a job_id; poll progress with "
    "mvd_task_status and cancel cooperatively with mvd_task_stop. Read resource "
    "'mvdirector://skill' for the recommended end-to-end directing workflow."
)


def is_loopback_host(host: str) -> bool:
    """Return True if ``host`` names a loopback interface (localhost, 127.0.0.0/8, ::1)."""
    cleaned = host.strip().lower()
    if cleaned.startswith("[") and cleaned.endswith("]"):
        cleaned = cleaned[1:-1]
    if cleaned == "localhost":
        return True
    try:
        return ipaddress.ip_address(cleaned).is_loopback
    except ValueError:
        return False


def _reject_json_constant(token: str) -> Any:
    raise ValueError(f"Non-standard JSON constant not allowed: {token!r}")


def _loads_strict_json(raw: str) -> Any:
    return json.loads(raw, parse_constant=_reject_json_constant)


def _host_without_port(host_header: str) -> str:
    """Extract the host part from an HTTP Host header, returning ``''`` if malformed."""
    value = host_header.strip()
    if not value or any(ch.isspace() or ch in ",/@" for ch in value):
        return ""
    if value.startswith("["):
        end = value.find("]")
        if end == -1:
            return ""
        host = value[1:end]
        rest = value[end + 1 :]
        if not rest:
            return host
        if not rest.startswith(":"):
            return ""
        port_str = rest[1:]
        if not port_str.isdigit() or not (1 <= int(port_str) <= 65535):
            return ""
        return host
    if ":" in value:
        host, port_str = value.rsplit(":", 1)
        if ":" in host or not port_str.isdigit() or not (1 <= int(port_str) <= 65535):
            return ""
        return host
    return value


def _is_loopback_origin(origin: str) -> bool:
    parsed = urlparse(origin)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return False
    return is_loopback_host(parsed.hostname)


def _rpc_error(req_id: Any, code: int, message: str) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": code, "message": message},
    }


def _rpc_result(req_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def load_skill_markdown() -> str:
    """Load the bundled Agent Skill guide shipped inside the package."""
    return (resources.files("mvdirector.mcp") / "SKILL.md").read_text(encoding="utf-8")


class MCPServer:
    """Protocol-level MCP JSON-RPC 2.0 server wrapping ``mvdirector.api``."""

    def __init__(
        self,
        *,
        default_project: Path | None = None,
        allowed_roots: list[Path] | tuple[Path, ...] | None = None,
        comfy_provider: ComfyEndpointProvider | None = None,
        jobs: JobManager | None = None,
    ) -> None:
        self.guard = WorkspaceGuard(default_project=default_project, allowed_roots=allowed_roots)
        self.jobs = jobs or JobManager()
        self._tools: dict[str, ToolSpec] = build_tool_registry(
            guard=self.guard,
            jobs=self.jobs,
            comfy_provider=comfy_provider,
        )

    @property
    def tool_names(self) -> list[str]:
        return list(self._tools)

    def handle_message(self, message: Any) -> dict[str, Any] | None:
        """Process one decoded JSON-RPC 2.0 message; return None for notifications."""
        if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
            return _rpc_error(None, -32600, "Invalid JSON-RPC 2.0 request.")

        method = message.get("method")
        has_id = "id" in message
        req_id = message.get("id")
        if has_id and (isinstance(req_id, bool) or not isinstance(req_id, (str, int, type(None)))):
            return _rpc_error(None, -32600, "Invalid 'id': must be a string, integer, or null.")
        if not isinstance(method, str) or not method:
            return _rpc_error(req_id if has_id else None, -32600, "Missing or invalid 'method'.")

        params = message.get("params")
        if params is None:
            params = {}
        elif not isinstance(params, dict):
            if not has_id:
                return None
            return _rpc_error(req_id, -32602, "'params' must be a JSON object.")

        if not has_id:
            # Notifications never receive a JSON-RPC response.
            return None

        if method == "initialize":
            requested = params.get("protocolVersion")
            version = (
                requested
                if isinstance(requested, str) and requested in SUPPORTED_PROTOCOL_VERSIONS
                else DEFAULT_PROTOCOL_VERSION
            )
            return _rpc_result(
                req_id,
                {
                    "protocolVersion": version,
                    "capabilities": {
                        "tools": {"listChanged": False},
                        "resources": {"subscribe": False, "listChanged": False},
                    },
                    "serverInfo": {"name": "mvdirector", "version": __version__},
                    "instructions": _SERVER_INSTRUCTIONS,
                },
            )

        if method == "ping":
            return _rpc_result(req_id, {})

        if method == "tools/list":
            return _rpc_result(
                req_id,
                {"tools": [spec.descriptor() for spec in self._tools.values()]},
            )

        if method == "tools/call":
            return self._handle_tool_call(req_id, params)

        if method == "resources/list":
            return _rpc_result(
                req_id,
                {
                    "resources": [
                        {
                            "uri": SKILL_RESOURCE_URI,
                            "name": "mvdirector Agent Skill",
                            "description": (
                                "End-to-end guide for directing music videos via mvdirector MCP."
                            ),
                            "mimeType": "text/markdown",
                        }
                    ]
                },
            )

        if method == "resources/read":
            uri = params.get("uri")
            if uri != SKILL_RESOURCE_URI:
                return _rpc_error(req_id, -32602, f"Unknown resource URI: {uri!r}")
            return _rpc_result(
                req_id,
                {
                    "contents": [
                        {
                            "uri": SKILL_RESOURCE_URI,
                            "mimeType": "text/markdown",
                            "text": load_skill_markdown(),
                        }
                    ]
                },
            )

        return _rpc_error(req_id, -32601, f"Method not found: {method!r}")

    def _handle_tool_call(self, req_id: Any, params: dict[str, Any]) -> dict[str, Any]:
        name = params.get("name")
        if not isinstance(name, str) or name not in self._tools:
            return _rpc_error(req_id, -32602, f"Unknown tool: {name!r}")
        arguments = params.get("arguments")
        if arguments is None:
            arguments = {}
        elif not isinstance(arguments, dict):
            return _rpc_error(req_id, -32602, "'arguments' must be a JSON object.")

        spec = self._tools[name]
        allowed_keys = set(spec.input_schema.get("properties", {}))
        unknown = sorted(set(arguments) - allowed_keys)
        if unknown:
            return _rpc_error(
                req_id, -32602, f"Unknown argument(s) for '{name}': {', '.join(unknown)}."
            )
        try:
            payload = spec.handler(arguments)
        except MVDError as exc:
            error_body = {"error": str(exc), "error_type": type(exc).__name__}
            return _rpc_result(
                req_id,
                {
                    "content": [
                        {"type": "text", "text": json.dumps(error_body, indent=2, allow_nan=False)}
                    ],
                    "isError": True,
                },
            )
        except Exception:
            log.exception("Unhandled exception in MCP tool %s", name)
            error_body = {"error": "Internal server error.", "error_type": "InternalError"}
            return _rpc_result(
                req_id,
                {
                    "content": [
                        {"type": "text", "text": json.dumps(error_body, indent=2, allow_nan=False)}
                    ],
                    "isError": True,
                },
            )

        return _rpc_result(
            req_id,
            {
                "content": [
                    {"type": "text", "text": json.dumps(payload, indent=2, allow_nan=False)}
                ],
                "isError": False,
            },
        )


def serve_stdio(
    server: MCPServer,
    *,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
) -> None:
    """Serve newline-delimited JSON-RPC 2.0 messages over stdio until EOF."""
    in_stream = stdin or sys.stdin
    out_stream = stdout or sys.stdout
    for raw_line in in_stream:
        line = raw_line.strip()
        if not line:
            continue
        try:
            message = _loads_strict_json(line)
        except ValueError as exc:
            response: dict[str, Any] | None = _rpc_error(None, -32700, f"Parse error: {exc}")
        else:
            response = server.handle_message(message)
        if response is not None:
            out_stream.write(json.dumps(response, separators=(",", ":"), allow_nan=False) + "\n")
            out_stream.flush()


def _json_response(
    status: int,
    payload: dict[str, Any] | None,
    *,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, str], bytes]:
    body = (
        json.dumps(payload, separators=(",", ":"), allow_nan=False).encode("utf-8")
        if payload is not None
        else b""
    )
    out_headers: dict[str, str] = {"Cache-Control": "no-store"}
    if payload is not None:
        out_headers["Content-Type"] = "application/json; charset=utf-8"
    if headers:
        out_headers.update(headers)
    return int(status), out_headers, body


def handle_http_request(
    mcp_server: MCPServer,
    method: str,
    raw_path: str,
    headers: dict[str, str],
    body: bytes = b"",
    *,
    token: str | None = None,
    loopback_only: bool = True,
) -> tuple[int, dict[str, str], bytes]:
    """Process one Streamable HTTP MCP request and return ``(status, headers, body)``."""
    path = urlparse(raw_path).path
    norm_headers = {k.lower(): v for k, v in headers.items()}

    if method == "GET" and path == "/health":
        return _json_response(HTTPStatus.OK, {"status": "ok"})

    if path != "/mcp":
        return _json_response(HTTPStatus.NOT_FOUND, {"error": "Not found."})

    # 1. Block DNS rebinding when bound to loopback.
    if loopback_only:
        host_header = norm_headers.get("host", "")
        if host_header and not is_loopback_host(_host_without_port(host_header)):
            return _json_response(
                HTTPStatus.FORBIDDEN,
                {"error": "Non-loopback Host header rejected."},
            )

    # 2. Block cross-origin browser requests (CSRF).
    origin = norm_headers.get("origin")
    if origin and not _is_loopback_origin(origin):
        return _json_response(
            HTTPStatus.FORBIDDEN,
            {"error": "Cross-origin requests are not allowed."},
        )

    # 3. Verify bearer token when configured.
    if token:
        auth = norm_headers.get("authorization", "")
        prefix = "Bearer "
        supplied = auth[len(prefix) :] if auth.startswith(prefix) else ""
        if not hmac.compare_digest(supplied.encode("utf-8"), token.encode("utf-8")):
            return _json_response(
                HTTPStatus.UNAUTHORIZED,
                {"error": "Missing or invalid Bearer token."},
                headers={"WWW-Authenticate": "Bearer"},
            )

    if method == "GET":
        return _json_response(
            HTTPStatus.METHOD_NOT_ALLOWED,
            {"error": "Use POST /mcp for JSON-RPC requests."},
            headers={"Allow": "POST, DELETE"},
        )

    if method == "DELETE":
        return _json_response(HTTPStatus.OK, {"closed": True})

    if method != "POST":
        return _json_response(
            HTTPStatus.METHOD_NOT_ALLOWED,
            {"error": f"Method {method!r} not allowed."},
            headers={"Allow": "POST, DELETE"},
        )

    if len(body) > MAX_HTTP_BODY_BYTES:
        return _json_response(
            HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
            {"error": f"Request body exceeds {MAX_HTTP_BODY_BYTES} bytes."},
        )

    try:
        message = _loads_strict_json(body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        return _json_response(HTTPStatus.OK, _rpc_error(None, -32700, f"Parse error: {exc}"))

    response = mcp_server.handle_message(message)
    if response is None:
        return _json_response(HTTPStatus.ACCEPTED, None)

    extra_headers: dict[str, str] = {}
    if isinstance(message, dict) and message.get("method") == "initialize":
        # Always mint a server-generated session id (never reflect untrusted client headers).
        extra_headers["Mcp-Session-Id"] = uuid.uuid4().hex
    return _json_response(HTTPStatus.OK, response, headers=extra_headers)


class _MCPHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        mcp_server: MCPServer,
        *,
        token: str | None,
        loopback_only: bool,
    ) -> None:
        self.mcp_server = mcp_server
        self.token = token
        self.loopback_only = loopback_only
        super().__init__(server_address, _MCPHTTPRequestHandler)


class _MCPHTTPRequestHandler(BaseHTTPRequestHandler):
    server: _MCPHTTPServer
    timeout = 30

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        log.info("mcp-http %s - %s", self.address_string(), format % args)

    def _write_response(self, status: int, headers: dict[str, str], body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        for key, value in headers.items():
            self.send_header(key, value)
        self.end_headers()
        if body:
            self.wfile.write(body)

    def _dispatch(self, method: str, body: bytes = b"") -> None:
        req_headers = {k: v for k, v in self.headers.items()}
        status, headers, resp_body = handle_http_request(
            self.server.mcp_server,
            method,
            self.path,
            req_headers,
            body,
            token=self.server.token,
            loopback_only=self.server.loopback_only,
        )
        self._write_response(status, headers, resp_body)

    def do_GET(self) -> None:  # noqa: N802
        self._dispatch("GET")

    def do_HEAD(self) -> None:  # noqa: N802
        self._dispatch("HEAD")

    def do_PUT(self) -> None:  # noqa: N802
        self._dispatch("PUT")

    def do_PATCH(self) -> None:  # noqa: N802
        self._dispatch("PATCH")

    def do_OPTIONS(self) -> None:  # noqa: N802
        self._dispatch("OPTIONS")

    def do_DELETE(self) -> None:  # noqa: N802
        self._dispatch("DELETE")

    def do_POST(self) -> None:  # noqa: N802
        raw_len = self.headers.get("Content-Length")
        if raw_len is None:
            status, headers, body = _json_response(
                HTTPStatus.BAD_REQUEST, {"error": "Missing Content-Length header."}
            )
            self._write_response(status, headers, body)
            return
        try:
            length = int(raw_len)
        except ValueError:
            status, headers, body = _json_response(
                HTTPStatus.BAD_REQUEST, {"error": "Invalid Content-Length header."}
            )
            self._write_response(status, headers, body)
            return
        if length < 0 or length > MAX_HTTP_BODY_BYTES:
            status, headers, body = _json_response(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                {"error": f"Request body exceeds {MAX_HTTP_BODY_BYTES} bytes."},
            )
            self._write_response(status, headers, body)
            return
        self._dispatch("POST", self.rfile.read(length))


def build_http_server(
    server: MCPServer,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    token: str | None = None,
    allow_unauthenticated: bool = False,
) -> ThreadingHTTPServer:
    """Build a Streamable HTTP MCP server with loopback/token/Origin/Host security guards."""
    clean_token: str | None = None
    if token is not None:
        clean_token = token.strip()
        if not clean_token:
            raise ConfigError("Bearer token must not be empty or whitespace.")
        if any(ord(ch) < 32 or ord(ch) == 127 for ch in clean_token):
            raise ConfigError("Bearer token must not contain control characters.")
    loopback = is_loopback_host(host)
    if not loopback and not clean_token and not allow_unauthenticated:
        raise ConfigError(
            f"Refusing to bind MCP HTTP server to non-loopback host '{host}' without a bearer "
            "token. Set --token / MVD_MCP_TOKEN, or pass --allow-unauthenticated on a trusted "
            "private network."
        )
    return _MCPHTTPServer(
        (host, port),
        server,
        token=clean_token,
        loopback_only=loopback,
    )


def serve_http(
    server: MCPServer,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    token: str | None = None,
    allow_unauthenticated: bool = False,
) -> None:
    """Start the Streamable HTTP MCP server and block until interrupted."""
    httpd = build_http_server(
        server,
        host=host,
        port=port,
        token=token,
        allow_unauthenticated=allow_unauthenticated,
    )
    bound_host, bound_port = httpd.server_address[:2]
    log.info("mvdirector MCP Streamable HTTP listening on http://%s:%s/mcp", bound_host, bound_port)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
