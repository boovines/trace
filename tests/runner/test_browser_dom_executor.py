"""Executor wiring for ``tier=browser_dom`` — Step 4.1 commit 3.

Mirror of ``tests/runner/test_mcp_dispatch.py`` for the browser_dom
tier. Covers the executor's pre-execution path:

* Non-destructive browser_dom step → dispatcher.dispatch called with
  the resolved hint, result stashed for the primer builder.
* Destructive browser_dom step → confirmation queued, dispatch only
  on confirm, step lands in ``destructive_steps_executed``.
* Decline / kill / timeout during destructive confirmation aborts the
  whole run before the agent loop ever starts.
* ``{param}`` substitution in url_pattern / selector / value happens
  in the executor (so the destructive-confirmation request shows the
  literal target the user is approving).
* Missing-param skips the step rather than aborting (computer_use
  tier picks up the slack from the SKILL.md prose).
* ``BrowserDOMCallError`` from the dispatcher (unsupported action,
  missing required field) lands as a ``browser_dom_skipped`` event.
* Action-level failure (page raised inside dispatch) lands as
  ``browser_dom_failed`` but doesn't abort the run.
* The primer message names every browser_dom-pre-executed step so the
  agent knows to skip them.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from unittest.mock import MagicMock

import pytest

from runner.browser_dom_dispatcher import (
    BrowserDOMCallError,
    BrowserDOMResult,
)
from runner.browser_dom_probe import BrowserDOMCapability
from runner.confirmation import ConfirmationDecision
from runner.execution_hints import CapabilityRegistry

# --- _FakeBrowserDOMDispatcher --------------------------------------------


class _FakeBrowserDOMDispatcher:
    """Stand-in for BrowserDOMDispatcher with no Playwright overhead.

    Configurable per-action: ``raise_call_error`` and
    ``raise_call_error_for`` simulate hint-shape errors;
    ``next_results`` lets a test queue specific outcomes for ordered
    dispatches. Records every dispatch for assertion.
    """

    def __init__(
        self,
        *,
        results: list[BrowserDOMResult] | None = None,
        raise_call_error: BrowserDOMCallError | None = None,
    ) -> None:
        self._results: list[BrowserDOMResult] = list(results or [])
        self._raise_call_error = raise_call_error
        self.calls: list[tuple[dict[str, Any], dict[str, str]]] = []

    async def __aenter__(self) -> _FakeBrowserDOMDispatcher:
        return self

    async def __aexit__(self, *_: Any) -> None:
        return None

    async def dispatch(
        self,
        hint: Mapping[str, Any],
        parameters: Mapping[str, str],
    ) -> BrowserDOMResult:
        self.calls.append((dict(hint), dict(parameters)))
        if self._raise_call_error is not None:
            raise self._raise_call_error
        if self._results:
            return self._results.pop(0)
        return BrowserDOMResult(
            ok=False,
            action=str(hint.get("action", "<unknown>")),
            selector=hint.get("selector"),
            url=None,
            content_text=None,
            duration_ms=0.0,
            error="no canned response",
        )


_BROWSER_DOM_AVAILABLE = BrowserDOMCapability(
    cdp_endpoint=None, executable_path="/fake/chromium"
)


def _meta_with_browser_dom_step(
    *,
    step_number: int,
    action: str,
    selector: str | None = None,
    url_pattern: str | None = None,
    value: str | None = None,
    destructive: bool = False,
) -> dict[str, Any]:
    hint: dict[str, Any] = {"tier": "browser_dom", "action": action}
    if selector is not None:
        hint["selector"] = selector
    if url_pattern is not None:
        hint["url_pattern"] = url_pattern
    if value is not None:
        hint["value"] = value
    return {
        "slug": "x",
        "step_count": step_number,
        "destructive_steps": [step_number] if destructive else [],
        "steps": [
            {
                "number": step_number,
                "intent": "do_thing",
                "execution_hints": [hint],
            }
        ],
    }


def _make_executor_for_browser_dom(
    *,
    meta: dict[str, Any],
    parameters: dict[str, str],
    dispatcher: _FakeBrowserDOMDispatcher | None,
    registry: CapabilityRegistry,
) -> Any:
    """Build a minimally-wired Executor whose collaborators are mocks."""
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
        browser_dom_dispatcher=dispatcher,  # type: ignore[arg-type]
    )
    return executor


def _wire_confirmation(
    executor: Any, decision: ConfirmationDecision
) -> None:
    """Patch the confirmation-await + screenshot-ref hooks."""

    async def _fake_await() -> ConfirmationDecision:
        return decision

    executor._await_decision_or_kill = _fake_await
    executor._last_screenshot_ref = MagicMock(return_value=None)


# --- non-destructive dispatch --------------------------------------------


@pytest.mark.asyncio
async def test_pre_execute_dispatches_non_destructive_browser_dom_step() -> None:
    fake = _FakeBrowserDOMDispatcher(
        results=[
            BrowserDOMResult(
                ok=True,
                action="navigate",
                selector=None,
                url="https://x.example/inbox",
                content_text="navigated to https://x.example/inbox",
                duration_ms=12.0,
            )
        ]
    )
    meta = _meta_with_browser_dom_step(
        step_number=1,
        action="navigate",
        url_pattern="https://x.example/{slug}",
    )
    registry = CapabilityRegistry(
        browser_dom_capability=_BROWSER_DOM_AVAILABLE
    )
    executor = _make_executor_for_browser_dom(
        meta=meta,
        parameters={"slug": "inbox"},
        dispatcher=fake,
        registry=registry,
    )

    skill_proxy = MagicMock(meta=meta)
    abort = await executor._pre_execute_browser_dom_steps(
        skill_proxy, MagicMock()
    )

    assert abort is None
    assert len(fake.calls) == 1
    hint, params = fake.calls[0]
    # Substitution happened in the executor; dispatcher gets resolved
    # values + empty parameters.
    assert hint["action"] == "navigate"
    assert hint["url_pattern"] == "https://x.example/inbox"
    assert params == {}
    assert 1 in executor._browser_dom_pre_executions


@pytest.mark.asyncio
async def test_substitution_resolves_selector_and_value() -> None:
    fake = _FakeBrowserDOMDispatcher(
        results=[
            BrowserDOMResult(
                ok=True, action="type", selector="input[name='to']",
                url="about:blank", content_text="typed into input[name='to']",
                duration_ms=1.0,
            )
        ]
    )
    meta = _meta_with_browser_dom_step(
        step_number=1,
        action="type",
        selector="input[name='{field}']",
        value="{recipient}",
    )
    registry = CapabilityRegistry(
        browser_dom_capability=_BROWSER_DOM_AVAILABLE
    )
    executor = _make_executor_for_browser_dom(
        meta=meta,
        parameters={"field": "to", "recipient": "alice@example.com"},
        dispatcher=fake,
        registry=registry,
    )

    await executor._pre_execute_browser_dom_steps(
        MagicMock(meta=meta), MagicMock()
    )
    hint, _ = fake.calls[0]
    assert hint["selector"] == "input[name='to']"
    assert hint["value"] == "alice@example.com"


@pytest.mark.asyncio
async def test_skips_step_when_param_missing() -> None:
    fake = _FakeBrowserDOMDispatcher()
    meta = _meta_with_browser_dom_step(
        step_number=1, action="type", selector="input", value="{recipient}"
    )
    registry = CapabilityRegistry(
        browser_dom_capability=_BROWSER_DOM_AVAILABLE
    )
    executor = _make_executor_for_browser_dom(
        meta=meta, parameters={}, dispatcher=fake, registry=registry,
    )
    abort = await executor._pre_execute_browser_dom_steps(
        MagicMock(meta=meta), MagicMock()
    )
    # Missing-param does not abort the run; the agent loop will run
    # the step via computer_use.
    assert abort is None
    assert fake.calls == []
    assert 1 not in executor._browser_dom_pre_executions


# --- destructive confirmation flow ----------------------------------------


@pytest.mark.asyncio
async def test_destructive_step_dispatches_after_confirm() -> None:
    fake = _FakeBrowserDOMDispatcher(
        results=[
            BrowserDOMResult(
                ok=True, action="click", selector="button.send",
                url="about:blank", content_text="clicked button.send",
                duration_ms=5.0,
            )
        ]
    )
    meta = _meta_with_browser_dom_step(
        step_number=4, action="click", selector="button.send",
        destructive=True,
    )
    registry = CapabilityRegistry(
        browser_dom_capability=_BROWSER_DOM_AVAILABLE
    )
    executor = _make_executor_for_browser_dom(
        meta=meta, parameters={}, dispatcher=fake, registry=registry,
    )
    _wire_confirmation(executor, ConfirmationDecision(action="confirm"))

    abort = await executor._pre_execute_browser_dom_steps(
        MagicMock(meta=meta), MagicMock()
    )
    assert abort is None
    assert len(fake.calls) == 1
    assert executor._destructive_steps_executed == [4]
    assert executor._confirmation_count == 1


@pytest.mark.asyncio
async def test_destructive_decline_aborts_run() -> None:
    fake = _FakeBrowserDOMDispatcher()
    meta = _meta_with_browser_dom_step(
        step_number=4, action="click", selector="button.send",
        destructive=True,
    )
    registry = CapabilityRegistry(
        browser_dom_capability=_BROWSER_DOM_AVAILABLE
    )
    executor = _make_executor_for_browser_dom(
        meta=meta, parameters={}, dispatcher=fake, registry=registry,
    )
    decline = ConfirmationDecision(action="abort", reason="user_abort")
    _wire_confirmation(executor, decline)

    abort = await executor._pre_execute_browser_dom_steps(
        MagicMock(meta=meta), MagicMock()
    )
    assert abort is decline
    # Dispatcher must NOT have been called — we aborted before action.
    assert fake.calls == []
    assert executor._destructive_steps_executed == []


# --- error paths ---------------------------------------------------------


@pytest.mark.asyncio
async def test_call_error_logs_skip_and_continues() -> None:
    fake = _FakeBrowserDOMDispatcher(
        raise_call_error=BrowserDOMCallError(
            action="click", selector="x", message="unsupported action 'wave_hands'"
        )
    )
    meta = _meta_with_browser_dom_step(
        step_number=1, action="click", selector="x"
    )
    registry = CapabilityRegistry(
        browser_dom_capability=_BROWSER_DOM_AVAILABLE
    )
    executor = _make_executor_for_browser_dom(
        meta=meta, parameters={}, dispatcher=fake, registry=registry,
    )
    abort = await executor._pre_execute_browser_dom_steps(
        MagicMock(meta=meta), MagicMock()
    )
    assert abort is None
    assert 1 not in executor._browser_dom_pre_executions


@pytest.mark.asyncio
async def test_action_failure_does_not_abort_run() -> None:
    fake = _FakeBrowserDOMDispatcher(
        results=[
            BrowserDOMResult(
                ok=False, action="click", selector="button.x",
                url="about:blank", content_text=None, duration_ms=10.0,
                error="Timeout: selector not found",
            )
        ]
    )
    meta = _meta_with_browser_dom_step(
        step_number=1, action="click", selector="button.x"
    )
    registry = CapabilityRegistry(
        browser_dom_capability=_BROWSER_DOM_AVAILABLE
    )
    executor = _make_executor_for_browser_dom(
        meta=meta, parameters={}, dispatcher=fake, registry=registry,
    )
    abort = await executor._pre_execute_browser_dom_steps(
        MagicMock(meta=meta), MagicMock()
    )
    assert abort is None
    # Failed dispatch is NOT stashed for the primer (the agent should
    # see only steps that actually succeeded).
    assert 1 not in executor._browser_dom_pre_executions


# --- skip when no dispatcher --------------------------------------------


@pytest.mark.asyncio
async def test_no_op_when_no_dispatcher() -> None:
    meta = _meta_with_browser_dom_step(
        step_number=1, action="click", selector="x"
    )
    registry = CapabilityRegistry(
        browser_dom_capability=_BROWSER_DOM_AVAILABLE
    )
    executor = _make_executor_for_browser_dom(
        meta=meta, parameters={}, dispatcher=None, registry=registry,
    )
    abort = await executor._pre_execute_browser_dom_steps(
        MagicMock(meta=meta), MagicMock()
    )
    assert abort is None
    assert executor._browser_dom_pre_executions == {}


# --- primer message ------------------------------------------------------


@pytest.mark.asyncio
async def test_primer_lists_pre_executed_steps() -> None:
    fake = _FakeBrowserDOMDispatcher(
        results=[
            BrowserDOMResult(
                ok=True, action="navigate", selector=None,
                url="https://x/inbox",
                content_text="navigated to https://x/inbox",
                duration_ms=4.0,
            )
        ]
    )
    meta = _meta_with_browser_dom_step(
        step_number=1, action="navigate", url_pattern="https://x/inbox",
    )
    meta["step_count"] = 5  # there are more steps after this one
    registry = CapabilityRegistry(
        browser_dom_capability=_BROWSER_DOM_AVAILABLE
    )
    executor = _make_executor_for_browser_dom(
        meta=meta, parameters={}, dispatcher=fake, registry=registry,
    )
    await executor._pre_execute_browser_dom_steps(
        MagicMock(meta=meta), MagicMock()
    )

    primer = executor._build_browser_dom_primer_message(MagicMock(meta=meta))
    assert primer is not None
    text = primer["content"][0]["text"]  # type: ignore[index]
    assert "[1]" in text
    assert "browser DOM" in text
    assert "navigate" in text
    # Should direct the agent to continue from step 2.
    assert "step 2" in text


def test_primer_returns_none_when_no_pre_executions() -> None:
    fake = _FakeBrowserDOMDispatcher()
    meta = _meta_with_browser_dom_step(
        step_number=1, action="click", selector="x"
    )
    registry = CapabilityRegistry(
        browser_dom_capability=_BROWSER_DOM_AVAILABLE
    )
    executor = _make_executor_for_browser_dom(
        meta=meta, parameters={}, dispatcher=fake, registry=registry,
    )
    primer = executor._build_browser_dom_primer_message(MagicMock(meta=meta))
    assert primer is None
