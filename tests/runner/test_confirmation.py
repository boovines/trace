"""Tests for X-016: ConfirmationQueue + WebSocket protocol."""

from __future__ import annotations

import asyncio
import logging

import pytest
from runner.confirmation import (
    ConfirmationDecision,
    ConfirmationQueue,
    ConfirmationRequest,
    IllegalState,
    make_request_message,
    parse_response_message,
)

pytestmark = pytest.mark.asyncio


def _push(queue: ConfirmationQueue, run_id: str = "run-1", step: int = 3) -> ConfirmationRequest:
    return queue.push_request(
        run_id=run_id,
        step_number=step,
        step_text="Click Send",
        screenshot_ref="0005.png",
        destructive_reason="label matches 'send'",
    )


async def test_push_request_records_pending() -> None:
    queue = ConfirmationQueue()
    assert not queue.has_pending("run-1")
    request = _push(queue)
    assert isinstance(request, ConfirmationRequest)
    assert request.run_id == "run-1"
    assert request.step_number == 3
    assert queue.has_pending("run-1")
    assert queue.get_pending("run-1") == request


async def test_full_confirm_flow_returns_decision() -> None:
    queue = ConfirmationQueue()
    _push(queue)

    async def submit_after_delay() -> None:
        await asyncio.sleep(0)
        assert queue.submit_decision("run-1", ConfirmationDecision(action="confirm")) is True

    async with asyncio.TaskGroup() as tg:
        tg.create_task(submit_after_delay())
        decision_task = tg.create_task(queue.await_decision("run-1", timeout_seconds=5.0))

    decision = decision_task.result()
    assert decision.action == "confirm"
    assert decision.reason is None
    # Pending cleaned up.
    assert not queue.has_pending("run-1")


async def test_full_abort_flow_returns_decision() -> None:
    queue = ConfirmationQueue()
    _push(queue)

    async def submit_after_delay() -> None:
        await asyncio.sleep(0)
        queue.submit_decision(
            "run-1", ConfirmationDecision(action="abort", reason="user_said_no")
        )

    async with asyncio.TaskGroup() as tg:
        tg.create_task(submit_after_delay())
        decision_task = tg.create_task(queue.await_decision("run-1", timeout_seconds=5.0))

    decision = decision_task.result()
    assert decision.action == "abort"
    assert decision.reason == "user_said_no"
    assert not queue.has_pending("run-1")


async def test_await_decision_times_out_to_abort() -> None:
    queue = ConfirmationQueue()
    _push(queue)
    decision = await queue.await_decision("run-1", timeout_seconds=0.05)
    assert decision.action == "abort"
    assert decision.reason == "user_timeout"
    # Timed-out request is cleaned up so a future push for the same run_id is ok.
    assert not queue.has_pending("run-1")
    _push(queue)
    assert queue.has_pending("run-1")


async def test_double_push_raises_illegal_state() -> None:
    queue = ConfirmationQueue()
    _push(queue)
    with pytest.raises(IllegalState, match="already has a pending confirmation"):
        _push(queue)


async def test_push_after_decision_resolved_is_allowed() -> None:
    queue = ConfirmationQueue()
    _push(queue)
    queue.submit_decision("run-1", ConfirmationDecision(action="confirm"))
    decision = await queue.await_decision("run-1", timeout_seconds=1.0)
    assert decision.action == "confirm"
    # After resolution, a fresh push should succeed (next destructive step).
    req = _push(queue, step=7)
    assert req.step_number == 7


async def test_submit_decision_without_pending_is_noop(caplog: pytest.LogCaptureFixture) -> None:
    queue = ConfirmationQueue()
    with caplog.at_level(logging.WARNING, logger="runner.confirmation"):
        result = queue.submit_decision("run-missing", ConfirmationDecision(action="confirm"))
    assert result is False
    assert any("no pending request" in rec.message for rec in caplog.records)


async def test_await_without_pending_raises() -> None:
    queue = ConfirmationQueue()
    with pytest.raises(IllegalState, match="no pending confirmation to await"):
        await queue.await_decision("run-missing", timeout_seconds=0.01)


async def test_kill_run_injects_abort_decision() -> None:
    queue = ConfirmationQueue()
    _push(queue)

    async def kill_soon() -> None:
        await asyncio.sleep(0)
        assert queue.kill_run("run-1", reason="user_hotkey") is True

    async with asyncio.TaskGroup() as tg:
        tg.create_task(kill_soon())
        decision_task = tg.create_task(queue.await_decision("run-1", timeout_seconds=5.0))

    decision = decision_task.result()
    assert decision.action == "abort"
    assert decision.reason == "user_hotkey"
    assert not queue.has_pending("run-1")


async def test_kill_run_without_pending_is_noop() -> None:
    queue = ConfirmationQueue()
    assert queue.kill_run("run-missing") is False


async def test_kill_run_default_reason_is_kill_switch() -> None:
    queue = ConfirmationQueue()
    _push(queue)
    queue.kill_run("run-1")
    decision = await queue.await_decision("run-1", timeout_seconds=1.0)
    assert decision.action == "abort"
    assert decision.reason == "kill_switch"


async def test_submit_twice_second_is_dropped(caplog: pytest.LogCaptureFixture) -> None:
    queue = ConfirmationQueue()
    _push(queue)
    assert queue.submit_decision("run-1", ConfirmationDecision(action="confirm")) is True
    with caplog.at_level(logging.WARNING, logger="runner.confirmation"):
        second = queue.submit_decision("run-1", ConfirmationDecision(action="abort"))
    assert second is False
    assert any("decision already pending" in rec.message for rec in caplog.records)
    # The first decision survives.
    decision = await queue.await_decision("run-1", timeout_seconds=1.0)
    assert decision.action == "confirm"


async def test_independent_run_ids_do_not_interfere() -> None:
    queue = ConfirmationQueue()
    _push(queue, run_id="run-a", step=1)
    _push(queue, run_id="run-b", step=2)
    queue.submit_decision("run-b", ConfirmationDecision(action="confirm"))
    b_decision = await queue.await_decision("run-b", timeout_seconds=1.0)
    assert b_decision.action == "confirm"
    # run-a is still pending.
    assert queue.has_pending("run-a")
    queue.kill_run("run-a")
    a_decision = await queue.await_decision("run-a", timeout_seconds=1.0)
    assert a_decision.action == "abort"


async def test_timeout_decision_has_expected_shape() -> None:
    decision = ConfirmationDecision(action="abort", reason="user_timeout")
    assert decision.action == "abort"
    assert decision.reason == "user_timeout"


async def test_make_request_message_shape() -> None:
    request = ConfirmationRequest(
        run_id="r-1",
        step_number=4,
        step_text="Click Publish",
        screenshot_ref="0010.png",
        destructive_reason="label matches 'publish'",
    )
    msg = make_request_message(request, screenshot_url="/runs/r-1/screenshots/0010.png")
    assert msg == {
        "type": "confirmation_request",
        "run_id": "r-1",
        "step_number": 4,
        "step_text": "Click Publish",
        "destructive_reason": "label matches 'publish'",
        "screenshot_url": "/runs/r-1/screenshots/0010.png",
    }


async def test_make_request_message_with_no_screenshot_url() -> None:
    request = ConfirmationRequest(
        run_id="r-1",
        step_number=4,
        step_text="Click Publish",
        screenshot_ref=None,
        destructive_reason="belt-and-suspenders",
    )
    msg = make_request_message(request)
    assert msg["screenshot_url"] is None


async def test_parse_response_message_confirm() -> None:
    run_id, decision = parse_response_message(
        {"type": "confirmation_response", "run_id": "r-1", "decision": "confirm"}
    )
    assert run_id == "r-1"
    assert decision.action == "confirm"
    assert decision.reason is None


async def test_parse_response_message_abort_with_reason() -> None:
    run_id, decision = parse_response_message(
        {
            "type": "confirmation_response",
            "run_id": "r-1",
            "decision": "abort",
            "reason": "looked wrong",
        }
    )
    assert run_id == "r-1"
    assert decision.action == "abort"
    assert decision.reason == "looked wrong"


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        (None, "must be an object"),
        ([], "must be an object"),
        ({}, "type="),
        ({"type": "other", "run_id": "r", "decision": "confirm"}, "type="),
        ({"type": "confirmation_response", "decision": "confirm"}, "run_id"),
        ({"type": "confirmation_response", "run_id": "", "decision": "confirm"}, "run_id"),
        ({"type": "confirmation_response", "run_id": 1, "decision": "confirm"}, "run_id"),
        ({"type": "confirmation_response", "run_id": "r", "decision": "yes"}, "decision"),
        ({"type": "confirmation_response", "run_id": "r", "decision": None}, "decision"),
        (
            {"type": "confirmation_response", "run_id": "r", "decision": "confirm", "reason": 7},
            "reason",
        ),
    ],
)
async def test_parse_response_message_rejects_malformed(payload: object, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        parse_response_message(payload)
