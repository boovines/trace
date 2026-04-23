"""Confirmation queue and WebSocket protocol (X-016).

The runner pauses before every destructive action and asks the user to confirm
or abort. The pause mechanism is this module's :class:`ConfirmationQueue`:

* The executor calls :meth:`ConfirmationQueue.push_request` with the pending
  step's details. The queue records that the run is awaiting confirmation and
  buffers a one-slot decision channel for it.
* The API layer forwards the request to the UI over WebSocket via
  :func:`make_request_message` and, when the UI responds, calls
  :meth:`ConfirmationQueue.submit_decision`.
* The executor awaits the decision via :meth:`ConfirmationQueue.await_decision`,
  which returns ``'confirm'`` / ``'abort'`` (or a synthetic ``'user_timeout'``
  abort if the 5-minute deadline elapses).
* The kill switch (X-018) calls :meth:`ConfirmationQueue.kill_run` to force an
  abort into any pending queue for a run_id.

Invariants:

* At most ONE pending confirmation per run_id at any time. A second
  ``push_request`` before the first is resolved raises :class:`IllegalState`.
* ``submit_decision`` for a run_id with no pending request is a no-op (warning
  logged) — this happens legitimately when a timeout races with a late UI
  response.
* ``await_decision`` always cleans up its own state (whether it returns a
  decision, times out, or the kill switch fires).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Final, Literal

_logger = logging.getLogger(__name__)

DEFAULT_CONFIRMATION_TIMEOUT_SECONDS: Final[float] = 300.0

DecisionAction = Literal["confirm", "abort"]

_MSG_TYPE_REQUEST: Final[str] = "confirmation_request"
_MSG_TYPE_RESPONSE: Final[str] = "confirmation_response"

_USER_TIMEOUT_REASON: Final[str] = "user_timeout"
_KILL_SWITCH_REASON: Final[str] = "kill_switch"


def _put_decision_nowait(
    channel: asyncio.Queue[ConfirmationDecision],
    decision: ConfirmationDecision,
) -> None:
    """Best-effort ``put_nowait`` that swallows ``QueueFull``."""
    try:
        channel.put_nowait(decision)
    except asyncio.QueueFull:
        _logger.warning(
            "put_decision dropped: decision already pending on the channel"
        )


class IllegalState(RuntimeError):
    """Raised when the queue is used in a way the state machine forbids.

    Today the only case is ``push_request`` called twice for the same run_id
    before the first decision is resolved — the executor should never do this,
    but we fail loudly rather than silently overwriting the pending request.
    """


@dataclass(frozen=True, slots=True)
class ConfirmationRequest:
    """A pending confirmation for a destructive step, awaiting user decision.

    The queue hands one of these to the API layer which serializes it for the
    UI (see :func:`make_request_message`).
    """

    run_id: str
    step_number: int
    step_text: str
    screenshot_ref: str | None
    destructive_reason: str


@dataclass(frozen=True, slots=True)
class ConfirmationDecision:
    """The user's response to a confirmation request.

    ``reason`` is optional free text — for example ``'user_timeout'`` when the
    decision was synthesized by a timeout rather than a real UI response.
    """

    action: DecisionAction
    reason: str | None = None


class ConfirmationQueue:
    """Per-run_id confirmation state: at most one pending request each.

    Backed by a dict of single-slot :class:`asyncio.Queue` instances so
    ``await_decision`` can block until the API calls ``submit_decision``
    without polling. An :class:`asyncio.Queue` is single-event-loop safe by
    construction, which matches the runner's async architecture.
    """

    def __init__(self) -> None:
        self._pending: dict[str, ConfirmationRequest] = {}
        self._channels: dict[str, asyncio.Queue[ConfirmationDecision]] = {}
        self._channel_loops: dict[str, asyncio.AbstractEventLoop] = {}

    def push_request(
        self,
        *,
        run_id: str,
        step_number: int,
        step_text: str,
        screenshot_ref: str | None,
        destructive_reason: str,
    ) -> ConfirmationRequest:
        """Register a new pending confirmation.

        Raises :class:`IllegalState` if a request for this run_id is already
        pending. The caller (the executor) is responsible for also writing
        ``awaiting_confirmation`` to ``run_metadata.json`` via the RunWriter.
        """
        if run_id in self._pending:
            raise IllegalState(
                f"run_id={run_id!r} already has a pending confirmation; "
                "submit or abort it before pushing another"
            )
        request = ConfirmationRequest(
            run_id=run_id,
            step_number=step_number,
            step_text=step_text,
            screenshot_ref=screenshot_ref,
            destructive_reason=destructive_reason,
        )
        self._pending[run_id] = request
        self._channels[run_id] = asyncio.Queue(maxsize=1)
        self._channel_loops[run_id] = asyncio.get_event_loop()
        return request

    async def await_decision(
        self,
        run_id: str,
        timeout_seconds: float = DEFAULT_CONFIRMATION_TIMEOUT_SECONDS,
    ) -> ConfirmationDecision:
        """Block until the UI responds or ``timeout_seconds`` elapse.

        On timeout returns a synthetic abort with ``reason='user_timeout'``;
        the executor transitions the run to ``aborted`` in either abort case.
        Cleans up the pending state before returning so the run_id is free for
        a future confirmation (none expected — typically the run ends here).
        """
        channel = self._channels.get(run_id)
        if channel is None:
            raise IllegalState(
                f"run_id={run_id!r} has no pending confirmation to await"
            )
        try:
            decision = await asyncio.wait_for(channel.get(), timeout=timeout_seconds)
            return decision
        except TimeoutError:
            return ConfirmationDecision(action="abort", reason=_USER_TIMEOUT_REASON)
        finally:
            self._pending.pop(run_id, None)
            self._channels.pop(run_id, None)
            self._channel_loops.pop(run_id, None)

    def submit_decision(
        self,
        run_id: str,
        decision: ConfirmationDecision,
    ) -> bool:
        """Deliver the UI's decision to the waiting ``await_decision`` call.

        Returns ``True`` when the decision was delivered, ``False`` (with a
        warning logged) when there was no pending request for that run_id.
        The no-pending case races legitimately with timeouts and kill-switch
        aborts, so it is a non-fatal no-op.
        """
        channel = self._channels.get(run_id)
        loop = self._channel_loops.get(run_id)
        if channel is None or loop is None:
            _logger.warning(
                "submit_decision for run_id=%s dropped: no pending request "
                "(already resolved or timed out?)",
                run_id,
            )
            return False
        # The channel was created on the executor's loop (possibly a
        # background thread). When the caller is on the same loop (most
        # tests), put synchronously so we can surface ``QueueFull`` as a
        # ``False`` return. When on a different loop (production: UI HTTP
        # submit from the request-handler thread), hop loops safely.
        try:
            current = asyncio.get_running_loop()
        except RuntimeError:
            current = None
        if current is loop:
            try:
                channel.put_nowait(decision)
            except asyncio.QueueFull:
                _logger.warning(
                    "submit_decision for run_id=%s dropped: decision already "
                    "pending on the channel",
                    run_id,
                )
                return False
            return True
        try:
            loop.call_soon_threadsafe(
                _put_decision_nowait, channel, decision
            )
        except RuntimeError:
            _logger.warning(
                "submit_decision for run_id=%s dropped: channel loop closed",
                run_id,
            )
            return False
        return True

    def kill_run(self, run_id: str, reason: str = _KILL_SWITCH_REASON) -> bool:
        """Force an abort decision for any pending confirmation on ``run_id``.

        Used by the X-018 kill switch. Returns ``True`` if an abort was
        injected, ``False`` if there was nothing pending.
        """
        return self.submit_decision(
            run_id, ConfirmationDecision(action="abort", reason=reason)
        )

    def has_pending(self, run_id: str) -> bool:
        """Return True iff this run_id currently has a pending request."""
        return run_id in self._pending

    def get_pending(self, run_id: str) -> ConfirmationRequest | None:
        """Return the pending request for ``run_id``, or None if none."""
        return self._pending.get(run_id)


def make_request_message(
    request: ConfirmationRequest, *, screenshot_url: str | None = None
) -> dict[str, Any]:
    """Serialize a :class:`ConfirmationRequest` for the runner → UI WebSocket.

    Shape (per PRD X-016):
    ``{type: 'confirmation_request', run_id, step_number, step_text,
    destructive_reason, screenshot_url}``. ``screenshot_url`` is included as
    ``null`` when the API layer has nothing to point at (no screenshot ref).
    """
    return {
        "type": _MSG_TYPE_REQUEST,
        "run_id": request.run_id,
        "step_number": request.step_number,
        "step_text": request.step_text,
        "destructive_reason": request.destructive_reason,
        "screenshot_url": screenshot_url,
    }


def parse_response_message(payload: object) -> tuple[str, ConfirmationDecision]:
    """Parse a UI → runner WebSocket payload into ``(run_id, decision)``.

    Shape (per PRD X-016):
    ``{type: 'confirmation_response', run_id, decision: 'confirm'|'abort',
    reason?: string}``. Raises :class:`ValueError` on any structural problem
    — the WebSocket handler is expected to translate that into a 4xx close.
    """
    if not isinstance(payload, dict):
        raise ValueError(f"confirmation response must be an object, got {type(payload).__name__}")
    if payload.get("type") != _MSG_TYPE_RESPONSE:
        raise ValueError(
            f"confirmation response must have type={_MSG_TYPE_RESPONSE!r}, "
            f"got {payload.get('type')!r}"
        )
    run_id = payload.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("confirmation response missing non-empty run_id")
    action = payload.get("decision")
    if action not in ("confirm", "abort"):
        raise ValueError(
            f"confirmation response decision must be 'confirm' or 'abort', "
            f"got {action!r}"
        )
    reason = payload.get("reason")
    if reason is not None and not isinstance(reason, str):
        raise ValueError(
            "confirmation response reason must be string or absent, "
            f"got {type(reason).__name__}"
        )
    return run_id, ConfirmationDecision(action=action, reason=reason)


__all__ = [
    "DEFAULT_CONFIRMATION_TIMEOUT_SECONDS",
    "ConfirmationDecision",
    "ConfirmationQueue",
    "ConfirmationRequest",
    "DecisionAction",
    "IllegalState",
    "make_request_message",
    "parse_response_message",
]
