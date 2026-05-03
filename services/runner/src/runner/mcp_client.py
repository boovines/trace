"""MCP client + capability probe for the runner's tiered execution layer.

Step 3a of the tiered-execution rollout. Step 2 ([PR #7]) shipped a
:class:`~runner.execution_hints.CapabilityRegistry` whose default state
declared every MCP server unavailable; this module replaces that default
with a *real* probe that connects to each configured MCP server, lists
its tools, and seeds the registry with the live capability surface.

What this module does:

* **Config**: read ``~/.config/trace/mcp_servers.json`` (or
  ``$TRACE_MCP_CONFIG_PATH``) describing each MCP server as a stdio
  subprocess (command, args, env). The shape mirrors Anthropic's Claude
  desktop config so users can paste their existing setup.
* **Probe**: connect to each configured server briefly (with a
  per-server timeout), list its ``tools/`` namespace, and return a
  :class:`~runner.execution_hints.CapabilityRegistry` whose
  ``mcp_servers`` set names every healthy server and whose
  ``mcp_functions`` dict maps each server to the set of tool names it
  publishes.
* **Failure modes**: a server that fails to start, doesn't respond, or
  raises during ``initialize`` is logged and skipped — the runner still
  starts with a partially-populated registry. We never let one broken
  MCP take the run down.

What this module does NOT do (deferred to Step 3b):

* Actually invoke MCP tools at execution time. The dispatcher knows
  about ``tier=mcp`` hints; for now they get logged as "would dispatch"
  and the executor still drives computer-use for the action itself.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from runner.execution_hints import CapabilityRegistry

__all__ = [
    "DEFAULT_CALL_TIMEOUT_SECONDS",
    "DEFAULT_PROBE_TIMEOUT_SECONDS",
    "MCPCallDispatcher",
    "MCPCallError",
    "MCPCallResult",
    "MCPProbeError",
    "MCPProbeReport",
    "MCPServerConfig",
    "default_config_path",
    "load_server_configs",
    "probe_capabilities",
    "probe_capabilities_sync",
    "substitute_parameters",
]

LOGGER = logging.getLogger(__name__)

#: How long to wait per server for its full initialize+list_tools handshake.
DEFAULT_PROBE_TIMEOUT_SECONDS: float = 5.0

#: Per-call timeout for ``MCPCallDispatcher.call``. Generous enough for
#: real-world tools that touch the network; tight enough that a hung MCP
#: doesn't stall the whole run.
DEFAULT_CALL_TIMEOUT_SECONDS: float = 30.0


# Pattern matching ``{name}`` references in MCP-hint argument values. Mirrors
# the synthesizer's parameter identifier shape (1..30 chars, lowercase, must
# start with a letter). Anchored so nested ``{{x}}`` JSON-template-style
# tokens don't accidentally match.
_PARAM_REF_RE = re.compile(r"\{([a-z][a-z0-9_]{0,29})\}")


@dataclass(frozen=True)
class MCPServerConfig:
    """Stdio-launchable MCP server, as declared in the runner's config.

    Mirrors the schema Claude desktop uses so users can copy-paste their
    existing ``mcpServers`` block. ``env`` overlays onto the runner's
    environment when the server is launched; ``args`` is appended to
    ``command`` verbatim.
    """

    name: str
    command: str
    args: tuple[str, ...] = ()
    env: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class MCPProbeReport:
    """Result of probing every configured server.

    * ``registry`` — the populated :class:`CapabilityRegistry` ready to
      hand to :class:`~runner.executor.Executor`.
    * ``per_server`` — per-server outcome dict. Keys are server names;
      values are either the set of tool names (success) or an error
      string (failure). Useful for surfacing probe diagnostics in the
      run's event stream.
    """

    registry: CapabilityRegistry
    per_server: dict[str, frozenset[str] | str]


class MCPProbeError(Exception):
    """Raised only when the *config* itself is broken (parse / shape).

    Per-server connect failures don't raise — they're recorded as error
    strings on :attr:`MCPProbeReport.per_server` so the run keeps going
    with whatever servers are healthy.
    """


# --------------------------------------------------------------------- config


def default_config_path() -> Path:
    """Return ``~/.config/trace/mcp_servers.json``, honouring TRACE_MCP_CONFIG_PATH."""
    override = os.environ.get("TRACE_MCP_CONFIG_PATH")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".config" / "trace" / "mcp_servers.json"


def load_server_configs(path: Path) -> list[MCPServerConfig]:
    """Parse ``mcp_servers.json`` and return one config per declared server.

    Schema (matches Claude desktop's ``claude_desktop_config.json``)::

        {
          "mcpServers": {
            "<server_name>": {
              "command": "<executable>",
              "args": ["..."],
              "env":  {"K": "V"}
            }
          }
        }

    Returns an empty list when ``path`` doesn't exist (no config →
    nothing to probe → registry stays at its conservative default).

    Raises :class:`MCPProbeError` if the file is unreadable JSON or if
    the top-level shape is wrong; we'd rather fail loudly than silently
    skip a misconfigured config the user thought was active.
    """
    if not path.is_file():
        LOGGER.info("MCP config %s does not exist; no servers to probe", path)
        return []

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise MCPProbeError(
            f"MCP config at {path} is not valid JSON: {exc.msg} "
            f"(line {exc.lineno}, col {exc.colno})"
        ) from exc

    if not isinstance(raw, dict):
        raise MCPProbeError(
            f"MCP config at {path} must be a JSON object at the top level"
        )

    servers_block = raw.get("mcpServers", {})
    if not isinstance(servers_block, dict):
        raise MCPProbeError(
            f"MCP config {path}: 'mcpServers' must be a JSON object"
        )

    configs: list[MCPServerConfig] = []
    for name, entry in servers_block.items():
        if not isinstance(name, str) or not name:
            raise MCPProbeError(
                f"MCP config {path}: server names must be non-empty strings"
            )
        if not isinstance(entry, dict):
            raise MCPProbeError(
                f"MCP config {path}: server {name!r} entry must be a JSON object"
            )
        command = entry.get("command")
        if not isinstance(command, str) or not command:
            raise MCPProbeError(
                f"MCP config {path}: server {name!r} missing required string 'command'"
            )
        args_raw = entry.get("args", []) or []
        if not isinstance(args_raw, list) or not all(
            isinstance(a, str) for a in args_raw
        ):
            raise MCPProbeError(
                f"MCP config {path}: server {name!r} 'args' must be a list of strings"
            )
        env_raw = entry.get("env", {}) or {}
        if not isinstance(env_raw, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in env_raw.items()
        ):
            raise MCPProbeError(
                f"MCP config {path}: server {name!r} 'env' must be string→string"
            )
        configs.append(
            MCPServerConfig(
                name=name,
                command=command,
                args=tuple(args_raw),
                env=dict(env_raw),
            )
        )
    return configs


# ---------------------------------------------------------------------- probe


async def _probe_single_server(
    config: MCPServerConfig, timeout: float
) -> frozenset[str] | str:
    """Connect to one server, list its tools, and return the tool name set.

    Returns the (possibly empty) set on success or a short error string
    on any failure — handshake failure, list_tools error, timeout, or
    transport hang. Never raises into the caller; one bad server should
    never abort the probe of the rest.
    """
    params = StdioServerParameters(
        command=config.command,
        args=list(config.args),
        env={**os.environ, **dict(config.env)} if config.env else None,
    )
    try:
        async with asyncio.timeout(timeout):
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    listing = await session.list_tools()
    except TimeoutError:
        return f"timeout after {timeout:.1f}s"
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"

    names = {
        tool.name
        for tool in getattr(listing, "tools", [])
        if isinstance(getattr(tool, "name", None), str)
    }
    return frozenset(names)


async def probe_capabilities(
    configs: list[MCPServerConfig] | None = None,
    *,
    config_path: Path | None = None,
    per_server_timeout: float = DEFAULT_PROBE_TIMEOUT_SECONDS,
) -> MCPProbeReport:
    """Probe every configured server in parallel and return a registry.

    Either pass ``configs`` directly (test path) or let the function
    load them from ``config_path`` (or :func:`default_config_path` when
    omitted). Servers are probed concurrently because each probe is
    bounded by the same per-server timeout — serial probing would scale
    linearly with server count for no benefit.

    On success, ``registry.mcp_servers`` lists every server that
    responded and ``registry.mcp_functions[server]`` is the set of tool
    names that server published. Failed servers stay out of the
    registry but appear in ``per_server`` with an error string so
    callers can log diagnostics.
    """
    if configs is None:
        configs = load_server_configs(config_path or default_config_path())

    if not configs:
        return MCPProbeReport(
            registry=CapabilityRegistry(computer_use=True),
            per_server={},
        )

    tasks = [_probe_single_server(c, per_server_timeout) for c in configs]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    per_server: dict[str, frozenset[str] | str] = {}
    healthy_servers: set[str] = set()
    healthy_functions: dict[str, frozenset[str]] = {}

    for config, result in zip(configs, results, strict=True):
        if isinstance(result, BaseException):
            # _probe_single_server normally returns; this branch is for
            # truly catastrophic asyncio-level failures.
            per_server[config.name] = f"{type(result).__name__}: {result}"
            continue
        if isinstance(result, str):
            per_server[config.name] = result
            LOGGER.warning(
                "MCP server %r probe failed: %s", config.name, result
            )
            continue
        per_server[config.name] = result
        healthy_servers.add(config.name)
        healthy_functions[config.name] = result

    registry = CapabilityRegistry(
        mcp_servers=frozenset(healthy_servers),
        mcp_functions={k: v for k, v in healthy_functions.items()},
        computer_use=True,
    )
    return MCPProbeReport(registry=registry, per_server=per_server)


def probe_capabilities_sync(
    configs: list[MCPServerConfig] | None = None,
    *,
    config_path: Path | None = None,
    per_server_timeout: float = DEFAULT_PROBE_TIMEOUT_SECONDS,
) -> MCPProbeReport:
    """Sync wrapper around :func:`probe_capabilities` for non-async callers.

    The :class:`~runner.executor.Executor` itself runs inside an
    asyncio event loop, so it should call :func:`probe_capabilities`
    directly. This helper exists for the API layer's startup path and
    for tests that want a one-shot probe without spinning up their own
    loop.
    """
    return asyncio.run(
        probe_capabilities(
            configs,
            config_path=config_path,
            per_server_timeout=per_server_timeout,
        )
    )


# ---------------------------------------------------------------------- utils


def format_probe_report(report: MCPProbeReport) -> str:
    """Render a probe report as a single multi-line string for run logs.

    Used by :class:`~runner.executor.Executor` to emit a single
    ``mcp_probe_report`` event at run start so the run log preserves a
    snapshot of which servers were live and which weren't.
    """
    lines: list[str] = []
    healthy = sorted(report.registry.mcp_servers)
    failed = sorted(
        name
        for name, val in report.per_server.items()
        if isinstance(val, str)
    )
    total_functions = sum(
        len(v) for v in report.per_server.values() if isinstance(v, frozenset)
    )
    lines.append(
        f"healthy={len(healthy)} failed={len(failed)} "
        f"total_functions={total_functions}"
    )
    for name in healthy:
        fns = report.per_server[name]
        if isinstance(fns, frozenset):
            tools = ", ".join(sorted(fns)[:6])
            tail = "" if len(fns) <= 6 else f" (+{len(fns) - 6} more)"
            lines.append(f"  {name}: {tools}{tail}")
    for name in failed:
        err = report.per_server[name]
        with suppress(Exception):
            lines.append(f"  {name}: ERROR {err}")
    return "\n".join(lines)


def _coerce_any(value: Any) -> Any:
    """Type-narrowing shim so ``Any`` returns from the SDK don't infect callers."""
    return value


# --------------------------------------------------------------- substitution


class MCPCallError(RuntimeError):
    """Raised when an MCP call cannot be issued or the server returns an error.

    Carries the original server + function names so the executor's event
    log can name exactly which call failed.
    """

    def __init__(self, *, server: str, function: str, message: str) -> None:
        super().__init__(f"MCP {server}.{function}: {message}")
        self.server = server
        self.function = function
        self.detail = message


def substitute_parameters(
    arguments: Mapping[str, Any], parameters: Mapping[str, str]
) -> dict[str, Any]:
    """Replace ``{name}`` references in MCP-hint argument values.

    The synthesizer emits hints like
    ``{"to": "{recipient}", "body": "{reply_body}"}``. At dispatch time
    those parameter references are resolved against the run's actual
    parameter map. References to undeclared parameters raise
    :class:`KeyError` with the missing name so the dispatcher can fall
    through to the next tier rather than send a half-formed call.

    String values get full substitution; non-string values pass through
    unchanged. Nested dicts/lists also get recursively substituted.
    """

    def _replace_in_string(value: str) -> str:
        def _sub(match: re.Match[str]) -> str:
            name = match.group(1)
            if name not in parameters:
                raise KeyError(name)
            return parameters[name]

        return _PARAM_REF_RE.sub(_sub, value)

    def _walk(value: Any) -> Any:
        if isinstance(value, str):
            return _replace_in_string(value)
        if isinstance(value, list):
            return [_walk(item) for item in value]
        if isinstance(value, dict):
            return {k: _walk(v) for k, v in value.items()}
        return value

    out: dict[str, Any] = {}
    for key, value in arguments.items():
        out[key] = _walk(value)
    return out


# ------------------------------------------------------------------ dispatch


@dataclass(frozen=True)
class MCPCallResult:
    """Outcome of one ``tools/call`` invocation.

    * ``ok`` — True when the server returned a non-error result.
    * ``server`` / ``function`` — the call identity, for log lines.
    * ``content_text`` — the textual content blocks concatenated; this
      is what the executor injects into the agent's transcript so the
      agent knows what came back. ``None`` when the result has no text
      content.
    * ``raw`` — the original ``CallToolResult`` for callers that need
      structured access (e.g. parsing JSON-encoded tool output).
    * ``error`` — error message when ``ok`` is False; ``None`` otherwise.
    """

    ok: bool
    server: str
    function: str
    content_text: str | None
    raw: Any
    error: str | None = None


class MCPCallDispatcher:
    """Long-lived pool of MCP sessions for in-run tool calls.

    Probe-time connections (in :func:`probe_capabilities`) are short
    lived — open, list_tools, close. Dispatch-time connections need to
    *stay open* across every call in a run so we don't pay the
    initialize handshake on each step. This class owns one
    :class:`ClientSession` per configured server and round-trips
    ``tools/call`` requests through the matching session.

    Lifecycle: instantiate, ``async with dispatcher:`` (opens all
    sessions in parallel), do calls, exit the context (closes all
    sessions). The :class:`Executor` enters the context once at run
    start and exits when the run finishes. Servers that fail to open
    are skipped silently — the dispatcher just refuses ``call(...)`` for
    them (returns an error result), matching the probe layer's "one
    bad MCP shouldn't sink the rest" stance.
    """

    def __init__(
        self,
        configs: list[MCPServerConfig],
        *,
        per_call_timeout: float = DEFAULT_CALL_TIMEOUT_SECONDS,
    ) -> None:
        self._configs = list(configs)
        self._per_call_timeout = per_call_timeout
        # Per-server async-context stack so we can open and close every
        # connection in parallel as one async-context-managed unit.
        self._sessions: dict[str, ClientSession] = {}
        self._exit_callbacks: list[Any] = []

    async def __aenter__(self) -> MCPCallDispatcher:
        # Open every server's session in parallel. Sessions that fail to
        # open are dropped from ``self._sessions`` and any subsequent
        # ``call(server, ...)`` for them returns an error result.
        async def _open(config: MCPServerConfig) -> tuple[str, Any] | None:
            params = StdioServerParameters(
                command=config.command,
                args=list(config.args),
                env={**os.environ, **dict(config.env)} if config.env else None,
            )
            try:
                stdio_ctx = stdio_client(params)
                read, write = await stdio_ctx.__aenter__()
                session_ctx = ClientSession(read, write)
                session = await session_ctx.__aenter__()
                await session.initialize()
            except Exception as exc:
                LOGGER.warning(
                    "MCP server %r connection failed: %s",
                    config.name,
                    exc,
                )
                return None
            self._exit_callbacks.append((stdio_ctx, session_ctx))
            return (config.name, session)

        results = await asyncio.gather(
            *(_open(c) for c in self._configs),
            return_exceptions=True,
        )
        for entry in results:
            if isinstance(entry, BaseException) or entry is None:
                continue
            name, session = entry
            self._sessions[name] = session
        return self

    async def __aexit__(self, *_: Any) -> None:
        # Close in reverse order so stdio_client's subprocess teardown
        # mirrors its open. Never raise out of __aexit__ — server
        # processes that crashed should be cleanly logged, not bubbled.
        for stdio_ctx, session_ctx in reversed(self._exit_callbacks):
            with suppress(Exception):
                await session_ctx.__aexit__(None, None, None)
            with suppress(Exception):
                await stdio_ctx.__aexit__(None, None, None)
        self._sessions.clear()
        self._exit_callbacks.clear()

    @property
    def connected_servers(self) -> frozenset[str]:
        """Names of servers whose ``initialize`` handshake succeeded."""
        return frozenset(self._sessions.keys())

    async def call(
        self,
        *,
        server: str,
        function: str,
        arguments: Mapping[str, Any],
    ) -> MCPCallResult:
        """Invoke ``tools/call`` on a connected server.

        Returns an :class:`MCPCallResult` whose ``ok`` flag mirrors the
        server's ``isError`` response. Connection failures, timeouts,
        and unknown servers all map to ``ok=False`` with a descriptive
        ``error`` string — the executor's caller is expected to log the
        error and fall through to the next tier rather than abort the
        run.
        """
        session = self._sessions.get(server)
        if session is None:
            return MCPCallResult(
                ok=False,
                server=server,
                function=function,
                content_text=None,
                raw=None,
                error=(
                    f"server {server!r} is not connected; "
                    f"available: {sorted(self._sessions)}"
                ),
            )

        try:
            async with asyncio.timeout(self._per_call_timeout):
                result = await session.call_tool(
                    function,
                    arguments=dict(arguments),
                    read_timeout_seconds=timedelta(seconds=self._per_call_timeout),
                )
        except TimeoutError:
            return MCPCallResult(
                ok=False,
                server=server,
                function=function,
                content_text=None,
                raw=None,
                error=f"call timed out after {self._per_call_timeout:.1f}s",
            )
        except Exception as exc:
            return MCPCallResult(
                ok=False,
                server=server,
                function=function,
                content_text=None,
                raw=None,
                error=f"{type(exc).__name__}: {exc}",
            )

        text_blocks: list[str] = []
        for block in getattr(result, "content", []) or []:
            text = getattr(block, "text", None)
            if isinstance(text, str):
                text_blocks.append(text)
        content_text = "\n".join(text_blocks) if text_blocks else None

        if getattr(result, "isError", False):
            return MCPCallResult(
                ok=False,
                server=server,
                function=function,
                content_text=content_text,
                raw=result,
                error=content_text or "server returned isError without text",
            )

        return MCPCallResult(
            ok=True,
            server=server,
            function=function,
            content_text=content_text,
            raw=result,
        )
