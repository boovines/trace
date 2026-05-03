"""RunWriter subclass that publishes to an :class:`EventBroadcaster`.

Wraps every write path so the WebSocket stream can forward the same events
the executor emits to on-disk artifacts. The subclass owns no additional
state — every public method delegates to the parent and then publishes a
WebSocket-shaped event to the broadcaster.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from runner.event_stream import EventBroadcaster
from runner.run_index import RunIndex
from runner.run_writer import RunWriter
from runner.schema import RunMetadata


class ObservingRunWriter(RunWriter):
    """A :class:`RunWriter` that also publishes every mutation as a stream event.

    Event shapes match what the WebSocket consumer expects:

    * ``{type: 'status_change', run_id, status, metadata}`` after each write
      that changes ``status``. ``metadata`` is the schema-valid dict form.
    * ``{type: 'turn_complete', run_id, turn_number, input_tokens,
      output_tokens}`` after every transcript append.
    * ``{type: 'event', run_id, seq, event_type, message, step_number,
      screenshot_ref}`` after every event append.
    """

    def __init__(
        self,
        *,
        run_id: str,
        skill_slug: str,
        mode: str,
        runs_root: Path,
        broadcaster: EventBroadcaster,
        run_index: RunIndex | None = None,
    ) -> None:
        super().__init__(
            run_id=run_id,
            skill_slug=skill_slug,
            mode=mode,
            runs_root=runs_root,
            run_index=run_index,
        )
        self._broadcaster = broadcaster
        self._last_published_status: str | None = None

    def write_metadata(self, metadata: RunMetadata) -> None:
        super().write_metadata(metadata)
        if metadata.status != self._last_published_status:
            self._last_published_status = metadata.status
            self._broadcaster.publish(
                self.run_id,
                {
                    "type": "status_change",
                    "run_id": self.run_id,
                    "status": metadata.status,
                    "metadata": metadata.to_dict(),
                },
            )

    def append_event(
        self,
        *,
        seq: int,
        event_type: str,
        message: str,
        step_number: int | None = None,
        screenshot_seq: int | None = None,
        timestamp_ms: int | None = None,
    ) -> None:
        super().append_event(
            seq=seq,
            event_type=event_type,
            message=message,
            step_number=step_number,
            screenshot_seq=screenshot_seq,
            timestamp_ms=timestamp_ms,
        )
        self._broadcaster.publish(
            self.run_id,
            {
                "type": "event",
                "run_id": self.run_id,
                "seq": seq,
                "event_type": event_type,
                "message": message,
                "step_number": step_number,
                "screenshot_ref": (
                    f"{screenshot_seq:04d}.png" if screenshot_seq is not None else None
                ),
            },
        )

    def append_transcript(
        self,
        *,
        turn: int,
        role: str,
        content: list[Any],
        input_tokens: int,
        output_tokens: int,
        timestamp_ms: int | None = None,
    ) -> None:
        super().append_transcript(
            turn=turn,
            role=role,
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            timestamp_ms=timestamp_ms,
        )
        if role == "assistant":
            self._broadcaster.publish(
                self.run_id,
                {
                    "type": "turn_complete",
                    "run_id": self.run_id,
                    "turn_number": turn,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                },
            )


__all__ = ["ObservingRunWriter"]
