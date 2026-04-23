"""Kill switch (X-018): 2-second abort guarantee.

A :class:`KillSwitch` is a per-service registry of per-run
:class:`asyncio.Event` kill signals. The executor registers the run at start
and unregisters on completion; the API layer — POST
``/run/{run_id}/abort`` — and the Tauri global hotkey both call
:meth:`KillSwitch.kill` to request an abort.

Total time from ``kill()`` invocation to ``status=aborted`` on disk must be
under 2 seconds in the 95th percentile. The two cost centers are:

1. Cancelling the in-flight Anthropic HTTP request. The executor wraps each
   turn in :func:`asyncio.create_task` and races it against the kill event;
   on kill we :meth:`~asyncio.Task.cancel` the task, which surfaces as an
   ``httpx`` request cancellation at the next socket-read await point.
2. Flushing the final ``run_metadata.json``. The RunWriter uses atomic
   tmp+rename so this is one ``fsync`` away.

Invariants:

* ``kill()`` is idempotent — repeat calls for the same run_id are no-ops and
  logged.
* ``kill()`` before :meth:`register` is a no-op (logged): the run hasn't
  started, there is nothing to cancel.
* ``kill()`` after :meth:`unregister` is a no-op (logged): the run has
  already finalized.
* The executor is responsible for routing a kill into any pending
  confirmation via the :class:`~runner.confirmation.ConfirmationQueue`; this
  module only owns the event registry.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Final

_logger = logging.getLogger(__name__)

DEFAULT_KILL_REASON: Final[str] = "kill_switch"


@dataclass
class _RunState:
    event: asyncio.Event
    reason: str | None = None
    killed: bool = False


class KillSwitch:
    """Per-run kill-event registry.

    A single process-global instance is shared between the executor and the
    API layer. The executor calls :meth:`register` at run start to get its
    kill event, races that event in its main loop, then calls
    :meth:`unregister` on terminal status. The API layer looks up the run_id
    and calls :meth:`kill`.
    """

    def __init__(self) -> None:
        self._runs: dict[str, _RunState] = {}

    def register(self, run_id: str) -> asyncio.Event:
        """Return the :class:`asyncio.Event` for ``run_id``, creating it if new.

        Called by the Executor at run start. Safe to call twice for the same
        run_id — returns the existing event so the kill-before-register race
        is observable to the caller.
        """
        state = self._runs.get(run_id)
        if state is None:
            state = _RunState(event=asyncio.Event())
            self._runs[run_id] = state
        return state.event

    def unregister(self, run_id: str) -> None:
        """Forget ``run_id`` after its terminal status has been flushed.

        No-op if the run is not registered. Subsequent :meth:`kill` calls on
        the run_id become no-ops (logged), matching the "kill after finish"
        acceptance criterion.
        """
        self._runs.pop(run_id, None)

    def kill(self, run_id: str, reason: str = DEFAULT_KILL_REASON) -> bool:
        """Signal the run to abort.

        Returns ``True`` if a new kill signal was delivered, ``False`` when
        the call is a no-op — run not registered (never started / already
        finished) or already killed. The caller — typically the API handler —
        can treat the return value as "abort accepted" for the 200 response.
        """
        state = self._runs.get(run_id)
        if state is None:
            _logger.info(
                "kill(%r) dropped: run_id not registered "
                "(never started, already finished, or unknown)",
                run_id,
            )
            return False
        if state.killed:
            _logger.info("kill(%r) dropped: already killed", run_id)
            return False
        state.killed = True
        state.reason = reason
        state.event.set()
        return True

    def is_killed(self, run_id: str) -> bool:
        """Return True iff ``run_id`` is registered and a kill was delivered."""
        state = self._runs.get(run_id)
        return state is not None and state.killed

    def reason(self, run_id: str) -> str | None:
        """Return the kill reason for ``run_id`` or ``None`` if not killed."""
        state = self._runs.get(run_id)
        return state.reason if state is not None else None

    def is_registered(self, run_id: str) -> bool:
        """Return True iff ``run_id`` is currently tracked by the switch."""
        return run_id in self._runs


_GLOBAL_KILL_SWITCH: KillSwitch = KillSwitch()


def get_global_kill_switch() -> KillSwitch:
    """Return the process-global :class:`KillSwitch` singleton.

    The API layer uses this to look up the switch from request handlers, and
    the executor factory uses it when no explicit switch is injected. Tests
    that want isolation should construct their own :class:`KillSwitch` and
    pass it in directly rather than touching the global.
    """
    return _GLOBAL_KILL_SWITCH


__all__ = [
    "DEFAULT_KILL_REASON",
    "KillSwitch",
    "get_global_kill_switch",
]
