"""Process MCP server — expose the runner as MCP tools to Claude Desktop.

Step 6.2 of the open follow-ups (plan.md §5). Closes the
recorder→synth→runner→Claude loop: the synthesizer can now invoke
the runner's "start a run" / "what's the status" / "approve this
destructive step" surface as MCP tool calls from inside Claude
Desktop, instead of needing the user to drive the dashboard
manually.

Architecture
------------

This module hosts an MCP server in **stdio** mode — the standard
transport every MCP client (Claude Desktop, the synthesizer's MCP
catalog probe, etc.) speaks. The server is **a thin client of the
gateway's HTTP API**, not an in-process collaborator of
:class:`runner.run_manager.RunManager`. Two reasons:

1. The gateway already owns the canonical runner state (run index
   SQLite, broadcaster, kill switch, dispatcher pools). Sharing
   that state across a separate Python process would race; talking
   to it over HTTP is the cleanest separation.
2. Claude Desktop launches each MCP server as a stdio subprocess
   per client session. We want subprocess startup to be cheap —
   no Playwright probe, no SQLite open, no MCP-client probe. A
   thin HTTP shim costs ~50ms to start; in-process would cost
   seconds.

Tool surface
------------

Six tools, mapping 1:1 onto the existing gateway endpoints:

* ``list_skills``       → ``GET  /skills``
* ``list_runs``         → ``GET  /runs``
* ``start_run``         → ``POST /run/start``
* ``get_run_status``    → ``GET  /run/{run_id}``
* ``confirm_run``       → ``POST /run/{run_id}/confirm``
* ``abort_run``         → ``POST /run/{run_id}/abort``

We deliberately do NOT expose ``GET /run/{id}/events`` or the WS
stream as tools — Claude Desktop polls instead, and the per-event
fan-out of a long run would burn the model's context.

Configuration
-------------

The server reads ``$TRACE_GATEWAY_URL`` (default
``http://127.0.0.1:8765``) so a non-default gateway port can be
set without re-installing the Claude Desktop config.

Claude Desktop config snippet to register this server::

    {
      "mcpServers": {
        "trace-runner": {
          "command": "uv",
          "args": [
            "--directory", "/path/to/trace",
            "run", "python", "-m", "runner.mcp_server"
          ]
        }
      }
    }

The runner gateway must be running (``uvicorn gateway.main:app``)
before Claude tries to invoke a tool — otherwise the HTTP call
returns a connection error which the server surfaces verbatim.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import Awaitable, Callable
from typing import Any

import httpx
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

__all__ = [
    "DEFAULT_GATEWAY_URL",
    "DEFAULT_TIMEOUT_SECONDS",
    "build_server",
    "main",
    "run_stdio_server",
]

LOGGER = logging.getLogger(__name__)

#: Default URL of the trace gateway — matches the conventional bind
#: in :mod:`gateway.main`. Override via ``$TRACE_GATEWAY_URL`` when
#: running on a non-default port.
DEFAULT_GATEWAY_URL: str = "http://127.0.0.1:8765"

#: Per-tool HTTP timeout. Generous enough for ``start_run`` (which
#: kicks off a background task) but tight enough that a wedged
#: gateway doesn't keep Claude Desktop hanging.
DEFAULT_TIMEOUT_SECONDS: float = 15.0

#: Type alias for an HTTP-backed tool handler.
ToolHandler = Callable[
    [httpx.AsyncClient, dict[str, Any]], Awaitable[list[TextContent]]
]


# ---------------------------------------------------------------- tool defs


def _tools() -> list[Tool]:
    """The static set of tools this server exposes.

    Each tool's ``inputSchema`` is the canonical JSON-schema shape
    Claude Desktop will validate against before sending the call —
    keeping shapes tight here means a malformed call surfaces as a
    400-style error from Claude's side rather than a 422 from the
    gateway.
    """
    return [
        Tool(
            name="list_skills",
            description=(
                "List every skill installed in the trace runner's skills "
                "directory. Use this before start_run to discover available "
                "slugs and the parameters each skill expects."
            ),
            inputSchema={"type": "object", "properties": {}, "additionalProperties": False},
        ),
        Tool(
            name="list_runs",
            description=(
                "List recent runs from the runner's index. Optionally filter "
                "by skill_slug. The most recent runs come first."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "skill_slug": {
                        "type": "string",
                        "description": "Optional filter — only runs of this skill.",
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 500,
                        "default": 50,
                    },
                    "offset": {"type": "integer", "minimum": 0, "default": 0},
                },
                "additionalProperties": False,
            },
        ),
        Tool(
            name="start_run",
            description=(
                "Start a new run of the named skill. Returns a run_id "
                "immediately; poll get_run_status for completion. mode=dry_run "
                "is the default and safe — it replays against trajectory "
                "screenshots; mode=execute drives the real machine and is "
                "gated by the TRACE_ALLOW_LIVE env var on the runner."
            ),
            inputSchema={
                "type": "object",
                "required": ["skill_slug"],
                "properties": {
                    "skill_slug": {"type": "string"},
                    "parameters": {
                        "type": "object",
                        "description": (
                            "Map of {parameter_name: value}. Must cover every "
                            "required parameter declared in the skill's meta."
                        ),
                        "additionalProperties": {"type": "string"},
                        "default": {},
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["dry_run", "execute"],
                        "default": "dry_run",
                    },
                },
                "additionalProperties": False,
            },
        ),
        Tool(
            name="get_run_status",
            description=(
                "Return run_metadata.json for a run by id. Use to poll a "
                "run's status (running / awaiting_confirmation / succeeded "
                "/ failed / aborted)."
            ),
            inputSchema={
                "type": "object",
                "required": ["run_id"],
                "properties": {"run_id": {"type": "string"}},
                "additionalProperties": False,
            },
        ),
        Tool(
            name="confirm_run",
            description=(
                "Submit a destructive-action confirmation decision. Used when "
                "get_run_status reports status=awaiting_confirmation. "
                "decision=confirm proceeds; decision=abort finalises the run "
                "without firing the destructive action."
            ),
            inputSchema={
                "type": "object",
                "required": ["run_id", "decision"],
                "properties": {
                    "run_id": {"type": "string"},
                    "decision": {"type": "string", "enum": ["confirm", "abort"]},
                    "reason": {
                        "type": "string",
                        "description": (
                            "Optional free-text reason recorded in the audit "
                            "trail. Useful for declines."
                        ),
                    },
                },
                "additionalProperties": False,
            },
        ),
        Tool(
            name="abort_run",
            description=(
                "Trigger the run's kill switch. Idempotent: an already-"
                "finished run reports {killed: false, aborted: true}. Use to "
                "stop a runaway run without going through the confirmation "
                "queue."
            ),
            inputSchema={
                "type": "object",
                "required": ["run_id"],
                "properties": {"run_id": {"type": "string"}},
                "additionalProperties": False,
            },
        ),
    ]


# -------------------------------------------------------------- handlers


async def _handle_list_skills(
    client: httpx.AsyncClient, _arguments: dict[str, Any]
) -> list[TextContent]:
    resp = await client.get("/skills")
    return _wrap_response(resp)


async def _handle_list_runs(
    client: httpx.AsyncClient, arguments: dict[str, Any]
) -> list[TextContent]:
    params: dict[str, Any] = {}
    if "skill_slug" in arguments:
        params["skill_slug"] = arguments["skill_slug"]
    if "limit" in arguments:
        params["limit"] = int(arguments["limit"])
    if "offset" in arguments:
        params["offset"] = int(arguments["offset"])
    resp = await client.get("/runs", params=params)
    return _wrap_response(resp)


async def _handle_start_run(
    client: httpx.AsyncClient, arguments: dict[str, Any]
) -> list[TextContent]:
    body: dict[str, Any] = {
        "skill_slug": arguments["skill_slug"],
        "parameters": arguments.get("parameters") or {},
        "mode": arguments.get("mode", "dry_run"),
    }
    resp = await client.post("/run/start", json=body)
    return _wrap_response(resp)


async def _handle_get_run_status(
    client: httpx.AsyncClient, arguments: dict[str, Any]
) -> list[TextContent]:
    run_id = str(arguments["run_id"])
    resp = await client.get(f"/run/{run_id}")
    return _wrap_response(resp)


async def _handle_confirm_run(
    client: httpx.AsyncClient, arguments: dict[str, Any]
) -> list[TextContent]:
    run_id = str(arguments["run_id"])
    body: dict[str, Any] = {"decision": arguments["decision"]}
    if "reason" in arguments:
        body["reason"] = arguments["reason"]
    resp = await client.post(f"/run/{run_id}/confirm", json=body)
    return _wrap_response(resp)


async def _handle_abort_run(
    client: httpx.AsyncClient, arguments: dict[str, Any]
) -> list[TextContent]:
    run_id = str(arguments["run_id"])
    resp = await client.post(f"/run/{run_id}/abort", json={})
    return _wrap_response(resp)


_HANDLERS: dict[str, ToolHandler] = {
    "list_skills": _handle_list_skills,
    "list_runs": _handle_list_runs,
    "start_run": _handle_start_run,
    "get_run_status": _handle_get_run_status,
    "confirm_run": _handle_confirm_run,
    "abort_run": _handle_abort_run,
}


def _wrap_response(resp: httpx.Response) -> list[TextContent]:
    """Translate an httpx response into MCP tool output.

    On 2xx: pretty-print the JSON body so the agent reads structured
    output without having to ask "what does this look like".

    On 4xx/5xx: render an error string carrying the status, the
    FastAPI ``detail`` (when present), and the raw body fragment.
    The MCP client sees this as a normal text result and the
    invoking agent can decide to retry.
    """
    if 200 <= resp.status_code < 300:
        try:
            body: Any = resp.json()
        except json.JSONDecodeError:
            return [TextContent(type="text", text=resp.text)]
        return [
            TextContent(
                type="text", text=json.dumps(body, indent=2, ensure_ascii=False)
            )
        ]

    detail = ""
    try:
        payload = resp.json()
        if isinstance(payload, dict) and "detail" in payload:
            detail = str(payload["detail"])
    except json.JSONDecodeError:
        detail = resp.text

    suffix = f" — {detail}" if detail else ""
    return [
        TextContent(
            type="text",
            text=f"runner gateway returned HTTP {resp.status_code}{suffix}",
        )
    ]


# ----------------------------------------------------------------- server


def build_server(
    *,
    gateway_url: str = DEFAULT_GATEWAY_URL,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    client_factory: Callable[[], httpx.AsyncClient] | None = None,
) -> Server:
    """Construct an MCP :class:`Server` wired to the runner gateway.

    Tests pass ``client_factory`` to inject an :class:`httpx.AsyncClient`
    pinned to a :class:`httpx.ASGITransport` over the FastAPI app —
    no real network or sockets needed.
    """

    def _default_factory() -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=gateway_url, timeout=timeout_seconds
        )

    factory = client_factory or _default_factory

    server: Server[Any, Any] = Server("trace-runner")

    # The MCP SDK's decorators are untyped (no inputSchema → handler
    # type inference), so mypy strict flags the registration. Both
    # decorators are documented + stable in the SDK; the # type:
    # ignore narrows the strict-mode noise without losing real
    # type-checking on the function bodies themselves.

    @server.list_tools()  # type: ignore[no-untyped-call,untyped-decorator]
    async def _list_tools() -> list[Tool]:
        return _tools()

    @server.call_tool()  # type: ignore[untyped-decorator]
    async def _call_tool(
        name: str, arguments: dict[str, Any]
    ) -> list[TextContent]:
        handler = _HANDLERS.get(name)
        if handler is None:
            return [
                TextContent(
                    type="text",
                    text=(
                        f"trace-runner MCP server does not implement tool "
                        f"{name!r}. Known: {sorted(_HANDLERS)}"
                    ),
                )
            ]
        async with factory() as client:
            try:
                return await handler(client, arguments or {})
            except httpx.RequestError as exc:
                return [
                    TextContent(
                        type="text",
                        text=(
                            f"runner gateway request failed: {type(exc).__name__}: "
                            f"{exc}. Is the gateway running on "
                            f"{gateway_url}?"
                        ),
                    )
                ]

    return server


async def run_stdio_server(
    *,
    gateway_url: str = DEFAULT_GATEWAY_URL,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> None:
    """Run the MCP server over stdio. Blocks until the client disconnects."""
    server = build_server(
        gateway_url=gateway_url, timeout_seconds=timeout_seconds
    )
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream, write_stream, server.create_initialization_options()
        )


def main() -> None:
    """CLI entry: ``uv run python -m runner.mcp_server`` → stdio MCP server.

    Reads ``$TRACE_GATEWAY_URL`` to override the gateway URL (e.g.
    when the user moves the gateway off port 8765). Logs to stderr —
    stdout is reserved for the MCP wire protocol.
    """
    gateway_url = os.environ.get("TRACE_GATEWAY_URL", DEFAULT_GATEWAY_URL)
    logging.basicConfig(level=logging.INFO)
    LOGGER.info("trace-runner MCP server connecting to %s", gateway_url)
    asyncio.run(run_stdio_server(gateway_url=gateway_url))


if __name__ == "__main__":
    main()
