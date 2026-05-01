"""Per-run WebSocket event broadcaster.

The API layer's WebSocket handler subscribes to a run's events. The executor
publishes via an :class:`ObservingRunWriter` wrapper plus direct publishes
from the :class:`~runner.run_manager.RunManager` (confirmation requests and
the terminal ``done`` message). The broadcaster fan-outs each event to every
current subscriber's queue; there is no replay for late subscribers, so the
UI is expected to subscribe before the run starts or immediately after.

Cross-loop safety: subscribers' queues are bound to whichever loop called
:meth:`EventBroadcaster.subscribe` (typically the WebSocket handler's loop).
The run task runs on a dedicated background loop owned by the
:class:`~runner.run_manager.RunManager`. Publishing uses
``loop.call_soon_threadsafe`` so an event from the background loop correctly
wakes up a consumer awaiting on the WS handler's loop.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator
from typing import Any


class EventBroadcaster:
    """Per-run_id fan-out of events to N subscriber queues.

    Queues are unbounded; publishers are non-blocking. Subscribers consume via
    :meth:`subscribe` which returns an async generator that yields events
    until the run is marked done by :meth:`close`.
    """

    def __init__(self) -> None:
        self._subs: dict[str, list[_Subscription]] = {}
        self._lock = threading.Lock()

    def subscribe(self, run_id: str) -> _Subscription:
        """Create a new subscription for ``run_id`` bound to the current loop.

        The returned object is an async iterator that yields events until the
        run completes. The caller owns its lifecycle and should enter the
        ``async with`` block (or call :meth:`_Subscription.close`) to
        deregister on disconnect.
        """
        loop = asyncio.get_event_loop()
        queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        sub = _Subscription(self, run_id, queue, loop)
        with self._lock:
            self._subs.setdefault(run_id, []).append(sub)
        return sub

    def publish(self, run_id: str, event: dict[str, Any]) -> None:
        """Publish ``event`` to every current subscriber for ``run_id``.

        No-op if no subscribers. Safe to call from any thread: each
        subscriber's queue is driven via ``call_soon_threadsafe`` on its own
        loop so producers from a background thread correctly wake consumers.
        """
        with self._lock:
            subs = list(self._subs.get(run_id, ()))
        for sub in subs:
            sub._put(event)

    def close(self, run_id: str) -> None:
        """Signal end-of-stream to every subscriber for ``run_id``.

        The sentinel (``None``) causes ``subscribe`` iterators to stop. After
        close, any new subscription for this run_id is still valid (the UI
        may subscribe after the run completes) but will yield nothing.
        """
        with self._lock:
            subs = list(self._subs.get(run_id, ()))
        for sub in subs:
            sub._put(None)

    def _remove(self, run_id: str, sub: _Subscription) -> None:
        with self._lock:
            subs = self._subs.get(run_id)
            if subs is None:
                return
            try:
                subs.remove(sub)
            except ValueError:  # pragma: no cover - already removed
                return
            if not subs:
                self._subs.pop(run_id, None)


class _Subscription:
    """Async iterator over broadcaster events for a single subscriber."""

    def __init__(
        self,
        broadcaster: EventBroadcaster,
        run_id: str,
        queue: asyncio.Queue[dict[str, Any] | None],
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        self._broadcaster = broadcaster
        self._run_id = run_id
        self._queue = queue
        self._loop = loop
        self._closed = False

    def _put(self, event: dict[str, Any] | None) -> None:
        """Deliver ``event`` to the queue, crossing loops safely."""
        try:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, event)
        except RuntimeError:
            # The subscriber's loop has been closed — drop the event.
            return

    async def __aenter__(self) -> _Subscription:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: Any,
    ) -> None:
        self.close()

    def __aiter__(self) -> AsyncIterator[dict[str, Any]]:
        return self._iter()

    async def _iter(self) -> AsyncIterator[dict[str, Any]]:
        while True:
            event = await self._queue.get()
            if event is None:
                return
            yield event

    async def get(self, timeout: float | None = None) -> dict[str, Any] | None:
        """Return the next event or ``None`` if the stream closed.

        If ``timeout`` is set and elapses with no event, raises
        :class:`TimeoutError`.
        """
        if timeout is None:
            return await self._queue.get()
        return await asyncio.wait_for(self._queue.get(), timeout=timeout)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._broadcaster._remove(self._run_id, self)


__all__ = ["EventBroadcaster"]
