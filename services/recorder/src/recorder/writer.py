"""Trajectory directory writer.

The :class:`TrajectoryWriter` owns the on-disk layout for a single recording
(``<root>/<uuid>/{metadata.json, events.jsonl, screenshots/NNNN.png}``) and is
the only module allowed to mutate it.  It serialises writes behind a
``threading.Lock`` so that events produced on the CGEventTap background thread
and events produced by the main-thread text aggregator can never collide on
``seq`` assignment or on the ``events.jsonl`` file handle.

All on-disk writes that must survive a crash go through ``_atomic_write_json``
(write to ``.tmp`` → fsync → rename).  ``events.jsonl`` is append-only and is
flushed after every event; a durable fsync on every line is too expensive for
the expected event rate and is explicitly accepted as a v1 tradeoff (see
``scripts/ralph/prd.json`` open questions).
"""

from __future__ import annotations

import contextlib
import json
import os
import threading
import uuid as uuid_module
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any, TextIO

from recorder.index_db import IndexDB
from recorder.schema import validate_event, validate_metadata

__all__ = ["TrajectoryWriter"]


def _utcnow_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def _duration_ms_from_metadata(metadata: dict[str, Any]) -> int | None:
    started = metadata.get("started_at")
    stopped = metadata.get("stopped_at")
    if not isinstance(started, str) or not isinstance(stopped, str):
        return None
    try:
        start_dt = datetime.fromisoformat(started)
        stop_dt = datetime.fromisoformat(stopped)
    except ValueError:
        return None
    return int((stop_dt - start_dt).total_seconds() * 1000)


class TrajectoryWriter:
    """Write a single trajectory directory.

    Thread-safe: every mutation of ``events.jsonl``, ``metadata.json``, or the
    internal ``seq`` counter is guarded by a single ``threading.Lock``.
    """

    def __init__(
        self,
        root: Path | str,
        label: str,
        *,
        trajectory_id: str | None = None,
        index_db: IndexDB | None = None,
    ) -> None:
        self.id: str = trajectory_id or str(uuid_module.uuid4())
        self.label: str = label
        self.root: Path = Path(root)
        self.dir: Path = self.root / self.id
        self.screenshots_dir: Path = self.dir / "screenshots"

        self.root.mkdir(parents=True, exist_ok=True)
        self.dir.mkdir(parents=True, exist_ok=False)
        self.screenshots_dir.mkdir(parents=True, exist_ok=False)
        os.chmod(self.dir, 0o700)
        os.chmod(self.screenshots_dir, 0o700)

        self._events_path: Path = self.dir / "events.jsonl"
        self._metadata_path: Path = self.dir / "metadata.json"
        self._events_fh: TextIO | None = self._events_path.open("a", encoding="utf-8")

        self._lock: threading.Lock = threading.Lock()
        self._next_seq: int = 1
        self._metadata: dict[str, Any] | None = None
        self._closed: bool = False
        self._index_db: IndexDB | None = index_db

        # Insert a minimal row now so GET /trajectories sees an in-flight
        # recording even before write_metadata() lands; started_at + label
        # will be filled in as soon as the session calls write_metadata().
        # Index is a cache, not authoritative — swallow failures.
        if self._index_db is not None:
            with contextlib.suppress(Exception):
                self._index_db.upsert(
                    trajectory_id=self.id,
                    label=label or None,
                    started_at=None,
                    stopped_at=None,
                    event_count=0,
                    duration_ms=None,
                )

    # ----------------------------------------------------------------- metadata

    def write_metadata(self, metadata: dict[str, Any]) -> None:
        """Validate and atomically write ``metadata.json``."""
        validate_metadata(metadata)
        snapshot = dict(metadata)
        with self._lock:
            self._atomic_write_json(self._metadata_path, snapshot)
            self._metadata = snapshot
            event_count = max(0, self._next_seq - 1)

        if self._index_db is not None:
            label = (
                snapshot.get("label") if isinstance(snapshot.get("label"), str) else None
            )
            started_at = (
                snapshot.get("started_at")
                if isinstance(snapshot.get("started_at"), str)
                else None
            )
            stopped_at = (
                snapshot.get("stopped_at")
                if isinstance(snapshot.get("stopped_at"), str)
                else None
            )
            duration_ms = _duration_ms_from_metadata(snapshot)
            with contextlib.suppress(Exception):
                self._index_db.upsert(
                    trajectory_id=self.id,
                    label=label,
                    started_at=started_at,
                    stopped_at=stopped_at,
                    event_count=event_count,
                    duration_ms=duration_ms,
                )

    # ------------------------------------------------------------------- events

    def append_event(self, event: dict[str, Any]) -> dict[str, Any]:
        """Validate ``event`` against the trajectory schema and append it.

        Assigns ``seq`` if the caller omitted it (or passed ``None``).  The
        assigned event is validated *before* touching disk; a schema violation
        raises :class:`jsonschema.ValidationError` and writes nothing.

        Returns the dict that was actually written (useful for callers that
        want to know which ``seq`` was assigned).
        """
        with self._lock:
            if self._closed or self._events_fh is None:
                raise RuntimeError("TrajectoryWriter is closed")

            event_to_write = dict(event)
            if event_to_write.get("seq") is None:
                event_to_write["seq"] = self._next_seq

            # Validate BEFORE writing so bad events never hit the file.
            validate_event(event_to_write)

            seq = int(event_to_write["seq"])
            line = json.dumps(event_to_write, separators=(",", ":"), ensure_ascii=False)
            self._events_fh.write(line + "\n")
            self._events_fh.flush()

            # Keep the counter strictly ahead of any seq we've seen so callers
            # that pre-assign seq (e.g. the session orchestrator) can't collide
            # with future auto-assignments.
            self._next_seq = max(self._next_seq, seq) + 1
            return event_to_write

    # --------------------------------------------------------------- screenshots

    def write_screenshot(self, seq: int, png_bytes: bytes) -> Path:
        """Write ``screenshots/NNNN.png`` (4-digit zero-padded)."""
        if seq < 1:
            raise ValueError(f"seq must be >= 1, got {seq}")
        filename = f"{seq:04d}.png"
        path = self.screenshots_dir / filename
        tmp = path.with_name(path.name + ".tmp")
        with tmp.open("wb") as fh:
            fh.write(png_bytes)
            fh.flush()
            os.fsync(fh.fileno())
        tmp.replace(path)
        return path

    # --------------------------------------------------------------------- close

    def close(self) -> None:
        """Finalise ``stopped_at`` in metadata and close the events handle.

        Safe to call multiple times; subsequent calls are no-ops.
        """
        with self._lock:
            if self._closed:
                return
            self._closed = True

            if self._events_fh is not None:
                try:
                    self._events_fh.flush()
                    # Best-effort fsync; some filesystems (tmpfs in tests)
                    # don't support it and raise EINVAL.
                    with contextlib.suppress(OSError):
                        os.fsync(self._events_fh.fileno())
                finally:
                    self._events_fh.close()
                    self._events_fh = None

            metadata = self._metadata
            if metadata is None and self._metadata_path.is_file():
                with self._metadata_path.open("r", encoding="utf-8") as fh:
                    metadata = json.load(fh)

            if metadata is not None:
                metadata = dict(metadata)
                metadata["stopped_at"] = _utcnow_iso()
                validate_metadata(metadata)
                self._atomic_write_json(self._metadata_path, metadata)
                self._metadata = metadata

            event_count = max(0, self._next_seq - 1)

        if self._index_db is not None:
            final_meta: dict[str, Any] | None = self._metadata
            stopped_at = (
                final_meta.get("stopped_at")
                if final_meta is not None
                and isinstance(final_meta.get("stopped_at"), str)
                else None
            )
            duration_ms = (
                _duration_ms_from_metadata(final_meta)
                if final_meta is not None
                else None
            )
            with contextlib.suppress(Exception):
                updated = self._index_db.mark_closed(
                    self.id,
                    stopped_at=stopped_at,
                    event_count=event_count,
                    duration_ms=duration_ms,
                )
                if not updated and final_meta is not None:
                    # No row yet (write_metadata was never called) — upsert
                    # a complete row from the metadata we just finalised.
                    label = (
                        final_meta.get("label")
                        if isinstance(final_meta.get("label"), str)
                        else None
                    )
                    started_at = (
                        final_meta.get("started_at")
                        if isinstance(final_meta.get("started_at"), str)
                        else None
                    )
                    self._index_db.upsert(
                        trajectory_id=self.id,
                        label=label,
                        started_at=started_at,
                        stopped_at=stopped_at,
                        event_count=event_count,
                        duration_ms=duration_ms,
                    )

    # ---------------------------------------------------------------- internals

    @staticmethod
    def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
        tmp = path.with_name(path.name + ".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, separators=(",", ":"))
            fh.flush()
            os.fsync(fh.fileno())
        tmp.replace(path)

    # ---------------------------------------------------------------- context

    def __enter__(self) -> TrajectoryWriter:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
