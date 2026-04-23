"""SQLite index for trajectory metadata.

Backed by a single ``index.db`` file (next to the ``trajectories/`` root)
so ``GET /trajectories`` stays fast once hundreds of recordings accumulate
— scanning the filesystem every list call means reading every
``metadata.json`` and counting every ``events.jsonl``, which is linear in
total recording history.

The index is a *cache* — the on-disk trajectory directories are the source
of truth.  :meth:`IndexDB.reconcile` rebuilds the cache from the
filesystem and is invoked on service startup to recover from crashes,
manual edits, or moves done outside the service.

Threading model: ``check_same_thread=False`` + a module-level
:class:`threading.Lock` guarding every statement.  A single recorder
process has at most one writer thread (the trajectory writer) and one
reader thread (the HTTP list endpoint); the lock is not a bottleneck in
practice.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import Any, TypedDict

__all__ = ["IndexDB", "TrajectoryRow"]

logger = logging.getLogger(__name__)


class TrajectoryRow(TypedDict):
    """Shape of a row in the ``trajectories`` table."""

    id: str
    label: str | None
    started_at: str | None
    stopped_at: str | None
    event_count: int
    duration_ms: int | None


def _compute_duration_ms(started_at: str | None, stopped_at: str | None) -> int | None:
    if not isinstance(started_at, str) or not isinstance(stopped_at, str):
        return None
    try:
        start_dt = datetime.fromisoformat(started_at)
        stop_dt = datetime.fromisoformat(stopped_at)
    except ValueError:
        return None
    return int((stop_dt - start_dt).total_seconds() * 1000)


def _count_events_file(events_path: Path) -> int:
    if not events_path.is_file():
        return 0
    try:
        with events_path.open(encoding="utf-8") as fh:
            return sum(1 for line in fh if line.strip())
    except OSError:
        return 0


def _row_to_typed(row: sqlite3.Row) -> TrajectoryRow:
    return TrajectoryRow(
        id=row["id"],
        label=row["label"],
        started_at=row["started_at"],
        stopped_at=row["stopped_at"],
        event_count=int(row["event_count"]) if row["event_count"] is not None else 0,
        duration_ms=(
            int(row["duration_ms"]) if row["duration_ms"] is not None else None
        ),
    )


class IndexDB:
    """SQLite-backed index of trajectories.

    Schema::

        CREATE TABLE trajectories (
            id TEXT PRIMARY KEY,
            label TEXT,
            started_at TEXT,
            stopped_at TEXT,
            event_count INTEGER NOT NULL DEFAULT 0,
            duration_ms INTEGER
        )
    """

    def __init__(self, path: Path | str) -> None:
        self.path: Path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection = sqlite3.connect(
            str(self.path), check_same_thread=False
        )
        self._conn.row_factory = sqlite3.Row
        self._create_schema()

    # ------------------------------------------------------------- schema

    def _create_schema(self) -> None:
        with self._lock:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS trajectories ("
                "id TEXT PRIMARY KEY, "
                "label TEXT, "
                "started_at TEXT, "
                "stopped_at TEXT, "
                "event_count INTEGER NOT NULL DEFAULT 0, "
                "duration_ms INTEGER"
                ")"
            )
            self._conn.commit()

    # ------------------------------------------------------------ writes

    def upsert(
        self,
        *,
        trajectory_id: str,
        label: str | None,
        started_at: str | None,
        stopped_at: str | None = None,
        event_count: int = 0,
        duration_ms: int | None = None,
    ) -> None:
        """Insert a new row, or replace an existing one with the same id."""
        with self._lock:
            self._conn.execute(
                "INSERT INTO trajectories "
                "(id, label, started_at, stopped_at, event_count, duration_ms) "
                "VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET "
                "  label = excluded.label, "
                "  started_at = excluded.started_at, "
                "  stopped_at = excluded.stopped_at, "
                "  event_count = excluded.event_count, "
                "  duration_ms = excluded.duration_ms",
                (
                    trajectory_id,
                    label,
                    started_at,
                    stopped_at,
                    int(event_count),
                    duration_ms,
                ),
            )
            self._conn.commit()

    def mark_closed(
        self,
        trajectory_id: str,
        *,
        stopped_at: str | None,
        event_count: int,
        duration_ms: int | None,
    ) -> bool:
        """Update the close-time fields on an existing row.

        Returns ``True`` if a row was updated; ``False`` if no row exists
        for ``trajectory_id`` (caller can choose to upsert instead).
        """
        with self._lock:
            cursor = self._conn.execute(
                "UPDATE trajectories "
                "SET stopped_at = ?, event_count = ?, duration_ms = ? "
                "WHERE id = ?",
                (stopped_at, int(event_count), duration_ms, trajectory_id),
            )
            self._conn.commit()
            return cursor.rowcount > 0

    def delete(self, trajectory_id: str) -> bool:
        """Remove the row for ``trajectory_id``.  Returns ``True`` if present."""
        with self._lock:
            cursor = self._conn.execute(
                "DELETE FROM trajectories WHERE id = ?", (trajectory_id,)
            )
            self._conn.commit()
            return cursor.rowcount > 0

    # ------------------------------------------------------------- reads

    def get(self, trajectory_id: str) -> TrajectoryRow | None:
        with self._lock:
            cursor = self._conn.execute(
                "SELECT id, label, started_at, stopped_at, event_count, duration_ms "
                "FROM trajectories WHERE id = ?",
                (trajectory_id,),
            )
            row = cursor.fetchone()
        if row is None:
            return None
        return _row_to_typed(row)

    def list_all(self) -> list[TrajectoryRow]:
        """Return every trajectory row ordered by ``started_at`` ascending.

        Rows whose ``started_at`` is ``NULL`` (in-progress or corrupt) sort
        first so they stay visible in the UI.
        """
        with self._lock:
            cursor = self._conn.execute(
                "SELECT id, label, started_at, stopped_at, event_count, duration_ms "
                "FROM trajectories "
                "ORDER BY started_at IS NULL DESC, started_at ASC, id ASC"
            )
            rows = cursor.fetchall()
        return [_row_to_typed(r) for r in rows]

    def ids(self) -> set[str]:
        with self._lock:
            cursor = self._conn.execute("SELECT id FROM trajectories")
            return {row["id"] for row in cursor.fetchall()}

    # --------------------------------------------------------- reconcile

    def reconcile(self, trajectories_root: Path) -> dict[str, int]:
        """Rebuild the index from ``trajectories_root`` on disk.

        * For every subdirectory with a readable ``metadata.json``, upsert
          a row derived from the metadata + an ``events.jsonl`` line count.
        * For every indexed id whose directory is gone, delete the row.

        Returns a small ``{added_or_updated, removed}`` summary dict for
        logging.
        """
        added_or_updated = 0
        removed = 0

        on_disk_ids: set[str] = set()
        if trajectories_root.is_dir():
            for entry in sorted(trajectories_root.iterdir()):
                if not entry.is_dir():
                    continue
                metadata = _read_metadata_safe(entry)
                if metadata is None:
                    continue
                tid = metadata.get("id")
                if not isinstance(tid, str) or not tid:
                    continue
                on_disk_ids.add(tid)
                label = metadata.get("label") if isinstance(metadata.get("label"), str) else None
                started_at = (
                    metadata.get("started_at")
                    if isinstance(metadata.get("started_at"), str)
                    else None
                )
                stopped_at = (
                    metadata.get("stopped_at")
                    if isinstance(metadata.get("stopped_at"), str)
                    else None
                )
                event_count = _count_events_file(entry / "events.jsonl")
                duration_ms = _compute_duration_ms(started_at, stopped_at)
                self.upsert(
                    trajectory_id=tid,
                    label=label,
                    started_at=started_at,
                    stopped_at=stopped_at,
                    event_count=event_count,
                    duration_ms=duration_ms,
                )
                added_or_updated += 1

        for stale_id in self.ids() - on_disk_ids:
            if self.delete(stale_id):
                removed += 1

        logger.info(
            "IndexDB.reconcile: %s added/updated, %s removed",
            added_or_updated,
            removed,
        )
        return {"added_or_updated": added_or_updated, "removed": removed}

    # --------------------------------------------------------------- close

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ------------------------------------------------------------- context

    def __enter__(self) -> IndexDB:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


def _read_metadata_safe(traj_dir: Path) -> dict[str, Any] | None:
    meta_path = traj_dir / "metadata.json"
    if not meta_path.is_file():
        return None
    try:
        with meta_path.open(encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        logger.warning("Could not read metadata for %s", traj_dir.name, exc_info=True)
        return None
    if not isinstance(data, dict):
        return None
    return data
