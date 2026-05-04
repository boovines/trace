"""Process MCP server tests — Step 6.2.

Verifies the wiring layer over the gateway HTTP API:

* ``list_tools`` returns the canonical six-tool surface.
* Each tool dispatches to the right gateway endpoint, with
  arguments translated correctly into query params / body.
* Successful responses are pretty-printed JSON; failure
  responses surface the FastAPI ``detail`` field; connection
  errors surface a "is the gateway running" message.
* Unknown tool names return a structured error rather than
  crashing the server.

The tests inject an ``httpx.AsyncClient`` over an
``ASGITransport`` pinned to a live FastAPI app, so the gateway
endpoints actually run — the behaviour exercised here is the
HTTP shim, not the runner internals (those have their own
tests).
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from httpx import ASGITransport
from mcp.types import CallToolResult, TextContent

from runner.api import router
from runner.kill_switch import KillSwitch
from runner.mcp_server import build_server
from runner.run_manager import RunManager

# --- gateway app fixture (mirrors test_api.py's pattern) -----------------


@pytest.fixture
def runs_root(tmp_path: Path) -> Path:
    root = tmp_path / "runs"
    root.mkdir()
    return root


@pytest.fixture
def skills_root(tmp_path: Path) -> Path:
    root = tmp_path / "skills"
    # Two valid skills + one malformed (missing skill.meta.json) so
    # ``list_skills`` exercises the "skip malformed" branch.
    for slug, name, params in [
        ("hello_world", "Hello World", []),
        (
            "send_email",
            "Send Email",
            [{"name": "recipient", "type": "string", "required": True}],
        ),
    ]:
        d = root / slug
        d.mkdir(parents=True)
        (d / "SKILL.md").write_text(f"# {name}\n\n## Steps\n\n1. Do thing.\n")
        (d / "skill.meta.json").write_text(
            json.dumps(
                {
                    "slug": slug,
                    "name": name,
                    "trajectory_id": "00000000-0000-0000-0000-000000000000",
                    "created_at": "2026-01-01T00:00:00+00:00",
                    "parameters": params,
                    "destructive_steps": [],
                    "preconditions": [],
                    "step_count": 1,
                }
            )
        )
    # Malformed: directory but no skill.meta.json. Should be skipped.
    (root / "broken").mkdir(parents=True)
    return root


@pytest.fixture
def gateway_app(
    tmp_path: Path,
    runs_root: Path,
    skills_root: Path,
) -> FastAPI:
    app = FastAPI()
    manager = RunManager(
        runs_root=runs_root,
        skills_root=skills_root,
        trajectories_root=tmp_path / "trajectories",
        costs_path=tmp_path / "costs.jsonl",
        kill_switch=KillSwitch(),
        # Skip the MCP + browser_dom probes — both touch external
        # subprocesses we don't need exercised here.
        capability_registry=None,
        probe_browser_dom=False,
    )
    app.state.run_manager = manager
    app.include_router(router)
    return app


def _client_factory(app: FastAPI) -> Any:
    def _factory() -> httpx.AsyncClient:
        return httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://gateway"
        )

    return _factory


async def _call(
    app: FastAPI, name: str, arguments: dict[str, Any] | None = None
) -> CallToolResult | list[TextContent]:
    """Build a server, look up the call_tool handler, and invoke it.

    Returns either a :class:`CallToolResult` (newer MCP SDKs wrap the
    handler output) or a raw ``list[TextContent]`` (older). Tests
    normalise via :func:`_text` below.
    """
    server = build_server(client_factory=_client_factory(app))
    handler = server.request_handlers
    # The decorated ``call_tool`` handler is registered on the server;
    # invoke it directly. The MCP SDK exposes registered handlers via
    # the request_handlers map keyed by the request type. We avoid
    # depending on the wire-protocol JSON-RPC layer for unit tests.
    from mcp.types import CallToolRequest

    req = CallToolRequest(
        method="tools/call",
        params={"name": name, "arguments": arguments or {}},
    )
    response = await handler[CallToolRequest](req)  # type: ignore[index]
    return response  # type: ignore[no-any-return]


def _text(result: Any) -> str:
    """Extract concatenated text from a tool-call result."""
    # Newer MCP wraps the handler output in a CallToolResult /
    # ServerResult envelope; older versions return content directly.
    inner = getattr(result, "root", result)
    content = getattr(inner, "content", inner)
    if isinstance(content, list):
        return "\n".join(
            item.text for item in content if hasattr(item, "text")
        )
    return str(content)


# --- list_tools ---------------------------------------------------------


@pytest.mark.asyncio
async def test_list_tools_exposes_canonical_surface(
    gateway_app: FastAPI,
) -> None:
    server = build_server(client_factory=_client_factory(gateway_app))
    from mcp.types import ListToolsRequest

    req = ListToolsRequest(method="tools/list")
    handler = server.request_handlers
    response = await handler[ListToolsRequest](req)  # type: ignore[index]
    inner = getattr(response, "root", response)
    tools = getattr(inner, "tools", [])
    names = {t.name for t in tools}
    assert names == {
        "list_skills",
        "list_runs",
        "start_run",
        "get_run_status",
        "confirm_run",
        "abort_run",
    }


# --- list_skills --------------------------------------------------------


@pytest.mark.asyncio
async def test_list_skills_returns_pretty_json(gateway_app: FastAPI) -> None:
    result = await _call(gateway_app, "list_skills")
    text = _text(result)
    payload = json.loads(text)
    assert isinstance(payload, list)
    slugs = {row["slug"] for row in payload}
    # broken/ skipped (no meta); two valid skills surface.
    assert slugs == {"hello_world", "send_email"}
    # Each row carries the contract fields the tool's docstring
    # promises.
    for row in payload:
        assert {"slug", "name", "parameters", "step_count"} <= row.keys()


# --- list_runs -----------------------------------------------------------


@pytest.mark.asyncio
async def test_list_runs_no_runs_returns_empty_array(
    gateway_app: FastAPI,
) -> None:
    result = await _call(gateway_app, "list_runs")
    text = _text(result)
    assert json.loads(text) == []


@pytest.mark.asyncio
async def test_list_runs_passes_through_filters(
    gateway_app: FastAPI,
) -> None:
    """``skill_slug`` / ``limit`` / ``offset`` reach the gateway as
    query params. We verify by triggering a 404-free call with a
    filter that matches no runs — the empty-array result confirms
    the URL was well-formed."""
    result = await _call(
        gateway_app,
        "list_runs",
        {"skill_slug": "hello_world", "limit": 5, "offset": 0},
    )
    text = _text(result)
    assert json.loads(text) == []


# --- get_run_status (404 path surfaces the gateway's `detail`) ---------


@pytest.mark.asyncio
async def test_get_run_status_unknown_run_surfaces_404(
    gateway_app: FastAPI,
) -> None:
    bogus = str(uuid.uuid4())
    result = await _call(gateway_app, "get_run_status", {"run_id": bogus})
    text = _text(result)
    # The error path renders a single-line "HTTP 404 — <detail>"
    # message rather than a JSON dict.
    assert "404" in text
    assert "run not found" in text


# --- confirm_run + abort_run validation -------------------------------


@pytest.mark.asyncio
async def test_confirm_run_unknown_returns_404(gateway_app: FastAPI) -> None:
    bogus = str(uuid.uuid4())
    result = await _call(
        gateway_app,
        "confirm_run",
        {"run_id": bogus, "decision": "confirm"},
    )
    text = _text(result)
    assert "404" in text


@pytest.mark.asyncio
async def test_abort_run_unknown_is_idempotent(gateway_app: FastAPI) -> None:
    """abort always 200s — even on an unknown id — because the kill
    switch can't tell "never started" from "already finished" and
    either way there's nothing else to do."""
    bogus = str(uuid.uuid4())
    result = await _call(gateway_app, "abort_run", {"run_id": bogus})
    text = _text(result)
    payload = json.loads(text)
    assert payload["aborted"] is True


# --- unknown tool name -------------------------------------------------


@pytest.mark.asyncio
async def test_unknown_tool_returns_structured_error(
    gateway_app: FastAPI,
) -> None:
    result = await _call(gateway_app, "wave_hands", {})
    text = _text(result)
    assert "wave_hands" in text
    assert "does not implement" in text


# --- gateway-down error path -----------------------------------------


@pytest.mark.asyncio
async def test_gateway_down_surfaces_helpful_message() -> None:
    """When the gateway isn't reachable, the tool surfaces a message
    that names the gateway URL so the user knows where to look."""

    def _broken_factory() -> httpx.AsyncClient:
        # Localhost port that nothing should be listening on.
        return httpx.AsyncClient(
            base_url="http://127.0.0.1:1", timeout=0.5
        )

    # The error message names the *configured* gateway_url (what the
    # user pointed the server at) rather than the URL the broken
    # factory ended up using — which is what actually surfaces in the
    # Claude Desktop transcript so the user knows where to look.
    server = build_server(
        client_factory=_broken_factory, gateway_url="http://127.0.0.1:1"
    )
    from mcp.types import CallToolRequest

    req = CallToolRequest(
        method="tools/call",
        params={"name": "list_skills", "arguments": {}},
    )
    handler = server.request_handlers
    response = await handler[CallToolRequest](req)  # type: ignore[index]
    text = _text(response)
    assert "request failed" in text
    assert "127.0.0.1:1" in text
