"""Tests for ``runner.mcp_client`` — Step 3a probe layer.

Covers:

* ``load_server_configs`` parses well-formed configs and raises on every
  malformed-shape branch.
* ``probe_capabilities`` returns a populated registry against a fake
  in-process MCP server (we patch :func:`runner.mcp_client._probe_single_server`
  to bypass the stdio transport — testing the SDK itself isn't this PR's
  scope, only the probe orchestration around it).
* Concurrent probing — two healthy servers + one failing one all show up
  in ``per_server``; only healthy ones land in ``mcp_servers``.
* ``CapabilityRegistry`` extension: function-set membership is enforced
  by ``supports()`` once ``mcp_functions`` is populated.
* ``format_probe_report`` shape spot-check.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from runner.execution_hints import CapabilityRegistry, Tier, pick_hint
from runner.mcp_client import (
    MCPProbeError,
    MCPProbeReport,
    MCPServerConfig,
    default_config_path,
    format_probe_report,
    load_server_configs,
    probe_capabilities,
)

# --- load_server_configs ---------------------------------------------------


def test_default_config_path_honours_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    target = tmp_path / "custom.json"
    monkeypatch.setenv("TRACE_MCP_CONFIG_PATH", str(target))
    assert default_config_path() == target


def test_default_config_path_falls_back_to_xdg(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TRACE_MCP_CONFIG_PATH", raising=False)
    path = default_config_path()
    assert path.parts[-3:] == (".config", "trace", "mcp_servers.json")


def test_load_server_configs_missing_file_returns_empty(tmp_path: Path) -> None:
    assert load_server_configs(tmp_path / "no_such.json") == []


def test_load_server_configs_happy_path(tmp_path: Path) -> None:
    cfg = tmp_path / "mcp.json"
    cfg.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "gmail": {
                        "command": "node",
                        "args": ["server.js"],
                        "env": {"TOKEN": "xyz"},
                    },
                    "slack": {"command": "uvx", "args": ["mcp-slack"]},
                }
            }
        )
    )
    configs = load_server_configs(cfg)
    by_name = {c.name: c for c in configs}
    assert set(by_name) == {"gmail", "slack"}
    assert by_name["gmail"].command == "node"
    assert by_name["gmail"].args == ("server.js",)
    assert by_name["gmail"].env == {"TOKEN": "xyz"}
    assert by_name["slack"].args == ("mcp-slack",)
    assert dict(by_name["slack"].env) == {}


def test_load_server_configs_invalid_json_raises(tmp_path: Path) -> None:
    cfg = tmp_path / "mcp.json"
    cfg.write_text("{not valid json")
    with pytest.raises(MCPProbeError, match="not valid JSON"):
        load_server_configs(cfg)


def test_load_server_configs_top_level_must_be_object(tmp_path: Path) -> None:
    cfg = tmp_path / "mcp.json"
    cfg.write_text("[]")
    with pytest.raises(MCPProbeError, match="JSON object at the top level"):
        load_server_configs(cfg)


def test_load_server_configs_missing_command(tmp_path: Path) -> None:
    cfg = tmp_path / "mcp.json"
    cfg.write_text(json.dumps({"mcpServers": {"x": {"args": []}}}))
    with pytest.raises(MCPProbeError, match="missing required string 'command'"):
        load_server_configs(cfg)


def test_load_server_configs_args_must_be_list_of_strings(tmp_path: Path) -> None:
    cfg = tmp_path / "mcp.json"
    cfg.write_text(
        json.dumps({"mcpServers": {"x": {"command": "y", "args": [1, 2, 3]}}})
    )
    with pytest.raises(MCPProbeError, match="'args' must be a list of strings"):
        load_server_configs(cfg)


def test_load_server_configs_env_must_be_string_to_string(tmp_path: Path) -> None:
    cfg = tmp_path / "mcp.json"
    cfg.write_text(
        json.dumps({"mcpServers": {"x": {"command": "y", "env": {"k": 1}}}})
    )
    with pytest.raises(MCPProbeError, match="'env' must be string"):
        load_server_configs(cfg)


# --- probe_capabilities (with patched single-server probe) ----------------


@pytest.fixture
def patched_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, frozenset[str] | str]:
    """Replace _probe_single_server with a deterministic stub.

    Tests populate the returned dict with ``server_name -> tool set or
    error string``; the probe returns whatever's keyed under the config's
    ``name`` (or, when missing, raises a ConnectionRefusedError-equivalent
    string so the error path is also exercised by default).
    """
    canned: dict[str, frozenset[str] | str] = {}

    async def _fake(
        config: MCPServerConfig, timeout: float
    ) -> frozenset[str] | str:
        return canned.get(config.name, "ConnectionError: stub default")

    monkeypatch.setattr(
        "runner.mcp_client._probe_single_server", _fake
    )
    return canned


@pytest.mark.asyncio
async def test_probe_capabilities_no_configs_returns_default_registry() -> None:
    report = await probe_capabilities(configs=[])
    assert isinstance(report, MCPProbeReport)
    assert report.registry.mcp_servers == frozenset()
    assert report.registry.computer_use is True
    assert report.per_server == {}


@pytest.mark.asyncio
async def test_probe_capabilities_healthy_server_populates_registry(
    patched_probe: dict[str, frozenset[str] | str],
) -> None:
    patched_probe["gmail"] = frozenset({"create_draft", "search_threads"})
    report = await probe_capabilities(
        configs=[MCPServerConfig(name="gmail", command="x")]
    )
    assert report.registry.mcp_servers == frozenset({"gmail"})
    assert report.registry.mcp_functions["gmail"] == frozenset(
        {"create_draft", "search_threads"}
    )
    assert report.per_server["gmail"] == frozenset(
        {"create_draft", "search_threads"}
    )


@pytest.mark.asyncio
async def test_probe_capabilities_failed_server_excluded_from_registry(
    patched_probe: dict[str, frozenset[str] | str],
) -> None:
    patched_probe["gmail"] = frozenset({"create_draft"})
    patched_probe["broken"] = "ConnectionError: refused"
    report = await probe_capabilities(
        configs=[
            MCPServerConfig(name="gmail", command="x"),
            MCPServerConfig(name="broken", command="y"),
        ]
    )
    assert report.registry.mcp_servers == frozenset({"gmail"})
    assert "broken" not in report.registry.mcp_functions
    assert report.per_server["broken"] == "ConnectionError: refused"


@pytest.mark.asyncio
async def test_probe_capabilities_loads_from_config_path_when_configs_omitted(
    tmp_path: Path,
    patched_probe: dict[str, frozenset[str] | str],
) -> None:
    cfg = tmp_path / "mcp.json"
    cfg.write_text(
        json.dumps(
            {"mcpServers": {"slack": {"command": "uvx", "args": ["mcp-slack"]}}}
        )
    )
    patched_probe["slack"] = frozenset({"post_message"})
    report = await probe_capabilities(config_path=cfg)
    assert report.registry.mcp_servers == frozenset({"slack"})
    assert "post_message" in report.registry.mcp_functions["slack"]


def test_probe_capabilities_sync_runs_event_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``probe_capabilities_sync`` returns the same report shape."""
    from runner.mcp_client import probe_capabilities_sync

    async def _fake(
        config: MCPServerConfig, timeout: float
    ) -> frozenset[str] | str:
        return frozenset({"do_thing"})

    monkeypatch.setattr("runner.mcp_client._probe_single_server", _fake)
    report = probe_capabilities_sync(
        configs=[MCPServerConfig(name="gmail", command="x")]
    )
    assert report.registry.mcp_servers == frozenset({"gmail"})


@pytest.mark.asyncio
async def test_probe_capabilities_sync_safe_inside_running_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The sync wrapper must not blow up when called from inside a loop.

    ``RunManager.start_run`` is async and runs on the FastAPI loop;
    inside it, ``_get_capability_registry`` (sync) calls this wrapper.
    A naive ``asyncio.run`` raises RuntimeError in that situation —
    this test pins the deferred-to-thread fix in place.
    """
    from runner.mcp_client import probe_capabilities_sync

    async def _fake(
        config: MCPServerConfig, timeout: float
    ) -> frozenset[str] | str:
        return frozenset({"do_thing"})

    monkeypatch.setattr("runner.mcp_client._probe_single_server", _fake)
    report = probe_capabilities_sync(
        configs=[MCPServerConfig(name="gmail", command="x")]
    )
    assert report.registry.mcp_servers == frozenset({"gmail"})


# --- CapabilityRegistry — function-name filtering after probe -------------


def test_supports_filters_by_function_when_mcp_functions_populated() -> None:
    reg = CapabilityRegistry(
        mcp_servers=frozenset({"gmail"}),
        mcp_functions={"gmail": frozenset({"create_draft"})},
    )
    assert reg.supports(
        {"tier": "mcp", "mcp_server": "gmail", "function": "create_draft"}
    )
    # Same server, but a function that's not in the live tool list.
    assert not reg.supports(
        {"tier": "mcp", "mcp_server": "gmail", "function": "send_draft"}
    )


def test_supports_falls_back_to_server_check_when_functions_empty() -> None:
    """Pre-probe registries (no mcp_functions) accept any function on the server."""
    reg = CapabilityRegistry(mcp_servers=frozenset({"gmail"}))  # no functions
    assert reg.supports(
        {"tier": "mcp", "mcp_server": "gmail", "function": "anything"}
    )


def test_pick_hint_walks_to_next_tier_when_function_unavailable() -> None:
    """A live server without the named function is not a match — keep walking."""
    reg = CapabilityRegistry(
        mcp_servers=frozenset({"gmail"}),
        mcp_functions={"gmail": frozenset({"create_draft"})},
        computer_use=True,
    )
    meta = {
        "step_count": 1,
        "steps": [
            {
                "number": 1,
                "execution_hints": [
                    {
                        "tier": "mcp",
                        "mcp_server": "gmail",
                        "function": "send_draft",  # not in live set
                        "arguments": {},
                    },
                    {"tier": "computer_use", "summary": "fallback"},
                ],
            }
        ],
    }
    decision = pick_hint(step_number=1, meta=meta, registry=reg)
    assert decision.chosen_tier == Tier.COMPUTER_USE
    assert decision.fell_back is True
    assert "not in live tool list" in decision.unsupported_reasons[0]
    assert "create_draft" in decision.unsupported_reasons[0]


# --- format_probe_report ---------------------------------------------------


def test_format_probe_report_lists_servers_and_failures() -> None:
    report = MCPProbeReport(
        registry=CapabilityRegistry(
            mcp_servers=frozenset({"gmail"}),
            mcp_functions={"gmail": frozenset({"create_draft", "send_draft"})},
        ),
        per_server={
            "gmail": frozenset({"create_draft", "send_draft"}),
            "broken": "TimeoutError: timeout after 5.0s",
        },
    )
    rendered = format_probe_report(report)
    assert "healthy=1 failed=1" in rendered
    assert "gmail:" in rendered
    assert "create_draft" in rendered
    assert "broken: ERROR" in rendered


def test_format_probe_report_truncates_long_function_lists() -> None:
    fns = frozenset(f"fn{i}" for i in range(30))
    report = MCPProbeReport(
        registry=CapabilityRegistry(
            mcp_servers=frozenset({"big"}),
            mcp_functions={"big": fns},
        ),
        per_server={"big": fns},
    )
    rendered = format_probe_report(report)
    assert "(+24 more)" in rendered  # 30 - 6 shown


# --- asyncio fixture configuration ---------------------------------------


def test_event_loop_runs_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sanity: probe_capabilities works under a fresh event loop."""

    async def _fake(
        config: MCPServerConfig, timeout: float
    ) -> frozenset[str] | str:
        return frozenset({"do_thing"})

    monkeypatch.setattr("runner.mcp_client._probe_single_server", _fake)
    report = asyncio.run(
        probe_capabilities(configs=[MCPServerConfig(name="gmail", command="x")])
    )
    assert report.registry.mcp_servers == frozenset({"gmail"})
