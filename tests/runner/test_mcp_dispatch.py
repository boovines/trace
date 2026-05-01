"""Tests for ``MCPCallDispatcher`` + ``substitute_parameters`` (Step 3b).

Step 3a probed servers and populated the registry; this PR adds the
dispatch path: opening a long-lived session pool, substituting
``{param}`` references, calling ``tools/call``, and surfacing structured
results.

Coverage:

* ``substitute_parameters`` — replaces top-level strings, walks lists
  and nested dicts, raises on missing parameters, leaves non-string
  values untouched.
* ``MCPCallDispatcher`` async-context lifecycle — opens connections,
  ``connected_servers`` reflects only the ones that succeeded, exits
  cleanly even on partial failures.
* ``MCPCallDispatcher.call`` — happy path returns ``ok=True`` with
  concatenated text content; isError result returns ``ok=False`` with
  the error text; unknown server returns ``ok=False`` without dialing.
* The :class:`Executor` integration — ``_pre_execute_mcp_steps`` walks
  ``meta.steps``, dispatches non-destructive MCP hints, skips
  destructive ones, logs ``mcp_dispatched`` / ``mcp_skipped`` / ``mcp_failed``
  events. Tests use a fake dispatcher that never opens real subprocesses.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from unittest.mock import MagicMock

import pytest

from runner.execution_hints import CapabilityRegistry
from runner.mcp_client import (
    MCPCallDispatcher,
    MCPCallResult,
    MCPServerConfig,
    substitute_parameters,
)

# --- substitute_parameters ------------------------------------------------


def test_substitute_parameters_top_level_string() -> None:
    out = substitute_parameters({"to": "{recipient}"}, {"recipient": "alice@x"})
    assert out == {"to": "alice@x"}


def test_substitute_parameters_multiple_refs_in_one_value() -> None:
    out = substitute_parameters(
        {"subject": "Re: {topic} from {sender}"},
        {"topic": "Q2 review", "sender": "Alice"},
    )
    assert out == {"subject": "Re: Q2 review from Alice"}


def test_substitute_parameters_nested_dict() -> None:
    out = substitute_parameters(
        {"payload": {"to": "{recipient}", "body": "Hi"}},
        {"recipient": "x@y"},
    )
    assert out == {"payload": {"to": "x@y", "body": "Hi"}}


def test_substitute_parameters_list_walks_elements() -> None:
    out = substitute_parameters(
        {"recipients": ["{a}", "{b}", "literal"]},
        {"a": "alice", "b": "bob"},
    )
    assert out == {"recipients": ["alice", "bob", "literal"]}


def test_substitute_parameters_non_strings_pass_through() -> None:
    out = substitute_parameters(
        {"send": True, "max": 5, "tags": None}, {}
    )
    assert out == {"send": True, "max": 5, "tags": None}


def test_substitute_parameters_missing_param_raises_keyerror() -> None:
    with pytest.raises(KeyError) as excinfo:
        substitute_parameters({"to": "{missing}"}, {})
    assert excinfo.value.args[0] == "missing"


def test_substitute_parameters_no_refs_passthrough() -> None:
    out = substitute_parameters({"x": "literal", "y": 42}, {"a": "b"})
    assert out == {"x": "literal", "y": 42}


# --- MCPCallDispatcher (with patched stdio) -------------------------------


@pytest.fixture
def patched_session(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Patch the dispatcher's stdio_client/ClientSession so no subprocess starts.

    Returns a dict the test fills in:

    * ``"sessions"`` (dict[str, MagicMock]) — key=server name, value=
      a mock session whose ``call_tool`` returns whatever the test
      configures via ``configure_call``.
    * ``"open_failures"`` (set[str]) — server names that should fail
      to initialize.
    """
    state: dict[str, Any] = {"sessions": {}, "open_failures": set()}

    class _FakeStdioCtx:
        def __init__(self, server_name: str) -> None:
            self.server_name = server_name

        async def __aenter__(self) -> tuple[Any, Any]:
            return ("read", "write")

        async def __aexit__(self, *_: Any) -> None:
            return None

    class _FakeSessionCtx:
        def __init__(self, server_name: str) -> None:
            self.server_name = server_name

        async def __aenter__(self) -> Any:
            if self.server_name in state["open_failures"]:
                raise ConnectionRefusedError("simulated open failure")
            return state["sessions"][self.server_name]

        async def __aexit__(self, *_: Any) -> None:
            return None

    # Stash the server name on the SessionCtx by reading it back out of
    # StdioServerParameters at construction time. We track the most-
    # recently-built ctx so the next ClientSession() picks it up.
    last_server: dict[str, str | None] = {"name": None}

    def _fake_stdio_client(params: Any) -> Any:
        # Recover server name from the command field — tests pass
        # the server name as the command for simplicity.
        last_server["name"] = params.command
        return _FakeStdioCtx(params.command)

    def _fake_client_session(_read: Any, _write: Any) -> Any:
        name = last_server["name"]
        assert name is not None
        return _FakeSessionCtx(name)

    monkeypatch.setattr("runner.mcp_client.stdio_client", _fake_stdio_client)
    monkeypatch.setattr("runner.mcp_client.ClientSession", _fake_client_session)

    return state


def _mock_session_with_call(
    call_result: Any | Exception,
) -> MagicMock:
    """Build a MagicMock session whose call_tool returns ``call_result``.

    Pass an exception instance to simulate a server-side raise.
    """
    session = MagicMock()
    session.initialize = _AsyncReturn(None)
    if isinstance(call_result, Exception):
        session.call_tool = _AsyncRaise(call_result)
    else:
        session.call_tool = _AsyncReturn(call_result)
    return session


class _AsyncReturn:
    def __init__(self, value: Any) -> None:
        self._value = value

    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._value


class _AsyncRaise:
    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise self._exc


def _ok_result(text: str = "ok") -> Any:
    """Build a fake CallToolResult-shaped object."""
    block = MagicMock()
    block.text = text
    result = MagicMock()
    result.isError = False
    result.content = [block]
    return result


def _err_result(text: str = "boom") -> Any:
    block = MagicMock()
    block.text = text
    result = MagicMock()
    result.isError = True
    result.content = [block]
    return result


@pytest.mark.asyncio
async def test_dispatcher_opens_and_lists_connected_servers(
    patched_session: dict[str, Any],
) -> None:
    patched_session["sessions"]["gmail"] = _mock_session_with_call(_ok_result())
    patched_session["sessions"]["slack"] = _mock_session_with_call(_ok_result())
    dispatcher = MCPCallDispatcher(
        configs=[
            MCPServerConfig(name="gmail", command="gmail"),
            MCPServerConfig(name="slack", command="slack"),
        ]
    )
    async with dispatcher:
        assert dispatcher.connected_servers == frozenset({"gmail", "slack"})


@pytest.mark.asyncio
async def test_dispatcher_skips_servers_that_fail_to_open(
    patched_session: dict[str, Any],
) -> None:
    patched_session["sessions"]["gmail"] = _mock_session_with_call(_ok_result())
    patched_session["open_failures"].add("broken")
    dispatcher = MCPCallDispatcher(
        configs=[
            MCPServerConfig(name="gmail", command="gmail"),
            MCPServerConfig(name="broken", command="broken"),
        ]
    )
    async with dispatcher:
        assert dispatcher.connected_servers == frozenset({"gmail"})


@pytest.mark.asyncio
async def test_call_happy_path_returns_text_content(
    patched_session: dict[str, Any],
) -> None:
    patched_session["sessions"]["gmail"] = _mock_session_with_call(
        _ok_result("thread_id=abc123")
    )
    dispatcher = MCPCallDispatcher(
        configs=[MCPServerConfig(name="gmail", command="gmail")]
    )
    async with dispatcher:
        result = await dispatcher.call(
            server="gmail",
            function="search_threads",
            arguments={"query": "from:alice"},
        )
    assert result.ok is True
    assert result.content_text == "thread_id=abc123"
    assert result.error is None


@pytest.mark.asyncio
async def test_call_returns_error_when_server_responds_iserror(
    patched_session: dict[str, Any],
) -> None:
    patched_session["sessions"]["gmail"] = _mock_session_with_call(
        _err_result("rate limited")
    )
    dispatcher = MCPCallDispatcher(
        configs=[MCPServerConfig(name="gmail", command="gmail")]
    )
    async with dispatcher:
        result = await dispatcher.call(
            server="gmail",
            function="search_threads",
            arguments={"query": "x"},
        )
    assert result.ok is False
    assert result.error == "rate limited"


@pytest.mark.asyncio
async def test_call_returns_error_when_server_raises(
    patched_session: dict[str, Any],
) -> None:
    patched_session["sessions"]["gmail"] = _mock_session_with_call(
        ConnectionResetError("pipe closed")
    )
    dispatcher = MCPCallDispatcher(
        configs=[MCPServerConfig(name="gmail", command="gmail")]
    )
    async with dispatcher:
        result = await dispatcher.call(
            server="gmail",
            function="search_threads",
            arguments={},
        )
    assert result.ok is False
    assert "ConnectionResetError" in (result.error or "")


@pytest.mark.asyncio
async def test_call_unknown_server_returns_error_without_dialing(
    patched_session: dict[str, Any],
) -> None:
    patched_session["sessions"]["gmail"] = _mock_session_with_call(_ok_result())
    dispatcher = MCPCallDispatcher(
        configs=[MCPServerConfig(name="gmail", command="gmail")]
    )
    async with dispatcher:
        result = await dispatcher.call(
            server="not_connected",
            function="x",
            arguments={},
        )
    assert result.ok is False
    assert "not connected" in (result.error or "")


# --- Executor._pre_execute_mcp_steps integration -------------------------


class _FakeDispatcher:
    """Stand-in for MCPCallDispatcher with no async-context overhead.

    Configurable per-call: maps ``(server, function)`` → ``MCPCallResult``.
    """

    def __init__(
        self, results: Mapping[tuple[str, str], MCPCallResult]
    ) -> None:
        self._results = dict(results)
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    async def __aenter__(self) -> _FakeDispatcher:
        return self

    async def __aexit__(self, *_: Any) -> None:
        return None

    async def call(
        self,
        *,
        server: str,
        function: str,
        arguments: Mapping[str, Any],
    ) -> MCPCallResult:
        self.calls.append((server, function, dict(arguments)))
        return self._results.get(
            (server, function),
            MCPCallResult(
                ok=False,
                server=server,
                function=function,
                content_text=None,
                raw=None,
                error="no canned response",
            ),
        )


def _meta_with_mcp_step(
    *,
    step_number: int,
    server: str,
    function: str,
    arguments: dict[str, Any],
    destructive: bool = False,
) -> dict[str, Any]:
    return {
        "slug": "x",
        "step_count": step_number,
        "destructive_steps": [step_number] if destructive else [],
        "steps": [
            {
                "number": step_number,
                "intent": "send_email",
                "execution_hints": [
                    {
                        "tier": "mcp",
                        "mcp_server": server,
                        "function": function,
                        "arguments": arguments,
                    }
                ],
            }
        ],
    }


def _make_executor_for_mcp(
    *,
    meta: dict[str, Any],
    parameters: dict[str, str],
    dispatcher: _FakeDispatcher | None,
    registry: CapabilityRegistry,
) -> Any:
    """Build a minimally-wired Executor whose collaborators are mocks.

    Most collaborators are MagicMock(); we exercise only the
    pre-execution path here.
    """
    from runner.executor import Executor

    loaded_skill = MagicMock()
    loaded_skill.meta = meta
    parsed = MagicMock()
    parsed.steps = []
    loaded_skill.parsed_skill = parsed

    writer = MagicMock()
    writer.append_event = MagicMock()
    queue = MagicMock()

    executor = Executor(
        loaded_skill=loaded_skill,
        parameters=parameters,
        mode="dry_run",
        agent_runtime=MagicMock(),
        input_adapter=MagicMock(),
        screen_source=MagicMock(),
        ax_resolver=MagicMock(),
        budget=MagicMock(),
        writer=writer,
        confirmation_queue=queue,
        run_id="00000000-0000-0000-0000-000000000000",
        capability_registry=registry,
        mcp_dispatcher=dispatcher,  # type: ignore[arg-type]
    )
    return executor


@pytest.mark.asyncio
async def test_pre_execute_dispatches_non_destructive_mcp_step() -> None:
    fake = _FakeDispatcher(
        results={
            ("gmail", "search_threads"): MCPCallResult(
                ok=True,
                server="gmail",
                function="search_threads",
                content_text="thread_id=t1",
                raw=None,
            )
        }
    )
    meta = _meta_with_mcp_step(
        step_number=2,
        server="gmail",
        function="search_threads",
        arguments={"query": "from:{recipient}"},
    )
    registry = CapabilityRegistry(
        mcp_servers=frozenset({"gmail"}),
        mcp_functions={"gmail": frozenset({"search_threads"})},
    )
    executor = _make_executor_for_mcp(
        meta=meta,
        parameters={"recipient": "alice"},
        dispatcher=fake,
        registry=registry,
    )

    skill_proxy = MagicMock(meta=meta)
    await executor._pre_execute_mcp_steps(skill_proxy)

    assert len(fake.calls) == 1
    server, function, args = fake.calls[0]
    assert (server, function) == ("gmail", "search_threads")
    assert args == {"query": "from:alice"}
    assert 2 in executor._mcp_pre_executions


@pytest.mark.asyncio
async def test_pre_execute_skips_destructive_mcp_step() -> None:
    fake = _FakeDispatcher(results={})
    meta = _meta_with_mcp_step(
        step_number=5,
        server="gmail",
        function="reply_to_thread",
        arguments={"thread_id": "x", "body": "hi"},
        destructive=True,
    )
    registry = CapabilityRegistry(
        mcp_servers=frozenset({"gmail"}),
        mcp_functions={"gmail": frozenset({"reply_to_thread"})},
    )
    executor = _make_executor_for_mcp(
        meta=meta,
        parameters={},
        dispatcher=fake,
        registry=registry,
    )
    skill_proxy = MagicMock(meta=meta)
    await executor._pre_execute_mcp_steps(skill_proxy)

    assert fake.calls == []  # never dispatched
    assert 5 not in executor._mcp_pre_executions


@pytest.mark.asyncio
async def test_pre_execute_skips_when_param_missing() -> None:
    fake = _FakeDispatcher(results={})
    meta = _meta_with_mcp_step(
        step_number=2,
        server="gmail",
        function="search_threads",
        arguments={"query": "from:{never_provided}"},
    )
    registry = CapabilityRegistry(
        mcp_servers=frozenset({"gmail"}),
        mcp_functions={"gmail": frozenset({"search_threads"})},
    )
    executor = _make_executor_for_mcp(
        meta=meta,
        parameters={},
        dispatcher=fake,
        registry=registry,
    )
    skill_proxy = MagicMock(meta=meta)
    await executor._pre_execute_mcp_steps(skill_proxy)
    assert fake.calls == []
    assert 2 not in executor._mcp_pre_executions


@pytest.mark.asyncio
async def test_pre_execute_no_dispatcher_is_noop() -> None:
    meta = _meta_with_mcp_step(
        step_number=2,
        server="gmail",
        function="search_threads",
        arguments={"query": "x"},
    )
    registry = CapabilityRegistry(
        mcp_servers=frozenset({"gmail"}),
        mcp_functions={"gmail": frozenset({"search_threads"})},
    )
    executor = _make_executor_for_mcp(
        meta=meta,
        parameters={},
        dispatcher=None,
        registry=registry,
    )
    skill_proxy = MagicMock(meta=meta)
    await executor._pre_execute_mcp_steps(skill_proxy)
    assert executor._mcp_pre_executions == {}


@pytest.mark.asyncio
async def test_pre_execute_records_failure_event_on_server_error() -> None:
    fake = _FakeDispatcher(
        results={
            ("gmail", "search_threads"): MCPCallResult(
                ok=False,
                server="gmail",
                function="search_threads",
                content_text=None,
                raw=None,
                error="rate limited",
            )
        }
    )
    meta = _meta_with_mcp_step(
        step_number=2,
        server="gmail",
        function="search_threads",
        arguments={"query": "x"},
    )
    registry = CapabilityRegistry(
        mcp_servers=frozenset({"gmail"}),
        mcp_functions={"gmail": frozenset({"search_threads"})},
    )
    executor = _make_executor_for_mcp(
        meta=meta,
        parameters={},
        dispatcher=fake,
        registry=registry,
    )
    skill_proxy = MagicMock(meta=meta)
    await executor._pre_execute_mcp_steps(skill_proxy)

    # The failed call should NOT land in pre_executions, but it should
    # be visible as an mcp_failed event on the writer mock.
    assert 2 not in executor._mcp_pre_executions
    event_types = [
        call.kwargs.get("event_type")
        for call in executor._writer.append_event.call_args_list
    ]
    assert "mcp_failed" in event_types


# --- _build_mcp_primer_message --------------------------------------------


def test_primer_message_none_when_nothing_pre_executed() -> None:
    meta = _meta_with_mcp_step(
        step_number=2,
        server="gmail",
        function="search_threads",
        arguments={"query": "x"},
    )
    executor = _make_executor_for_mcp(
        meta=meta,
        parameters={},
        dispatcher=None,
        registry=CapabilityRegistry(),
    )
    skill_proxy = MagicMock(meta=meta)
    assert executor._build_mcp_primer_message(skill_proxy) is None


def test_primer_message_includes_per_step_results() -> None:
    meta = {
        "slug": "x",
        "step_count": 3,
        "destructive_steps": [],
        "steps": [
            {"number": 2, "execution_hints": []},
        ],
    }
    executor = _make_executor_for_mcp(
        meta=meta,
        parameters={},
        dispatcher=None,
        registry=CapabilityRegistry(),
    )
    executor._mcp_pre_executions[2] = MCPCallResult(
        ok=True,
        server="gmail",
        function="search_threads",
        content_text="thread_id=t1",
        raw=None,
    )
    skill_proxy = MagicMock(meta=meta)
    msg = executor._build_mcp_primer_message(skill_proxy)
    assert msg is not None
    assert msg["role"] == "user"
    text = msg["content"][0]["text"]
    assert "[2]" in text
    assert "step 3 in the SKILL.md" in text
    assert "gmail.search_threads" in text
    assert "thread_id=t1" in text
