"""SQLite index of runs for fast ``GET /runs`` queries (X-020).

The index lives alongside the skill/trajectory data at
``~/Library/Application Support/Trace[-dev]/index.db``. Each service owns
its own table; the runner owns the ``runs`` table here.

Why a SQLite index rather than a directory scan:

* ``GET /runs`` should remain fast as runs accumulate over months of use.
  Scanning every ``run_metadata.json`` file for a single list view does not
  scale.
* Filtering by ``skill_slug`` and ordering by ``started_at DESC`` both
  become index lookups.

The RunWriter inserts a row on construction (``status='pending'``) and
upserts on every metadata write, so the index is always at-most-one-write
behind the on-disk metadata. A ``reconcile(runs_root)`` pass at service
startup picks up any rows that were orphaned by a crash and marks
non-terminal/non-waiting rows as ``failed`` with
``abort_reason='incomplete_on_restart'`` so the UI does not display a
stale ``running`` status forever.
"""

from __future__ import annotations

import contextlib
import json
import logging
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Final

from runner.schema import TERMINAL_STATUSES, WAITING_STATUS, RunMetadata

logger = logging.getLogger(__name__)

_SCHEMA_SQL: Final[str] = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    skill_slug TEXT NOT NULL,
    status TEXT NOT NULL,
    mode TEXT NOT NULL,
    started_at TEXT,
    ended_at TEXT,
    duration_seconds REAL,
    total_cost_usd REAL,
    final_step_reached INTEGER,
    abort_reason TEXT
);
CREATE INDEX IF NOT EXISTS runs_started_at_idx ON runs (started_at DESC);
CREATE INDEX IF NOT EXISTS runs_skill_slug_idx ON runs (skill_slug);
"""

INCOMPLETE_ON_RESTART_REASON: Final[str] = "incomplete_on_restart"


class RunIndex:
    """SQLite-backed index of run summaries.

    Connections are opened with ``check_same_thread=False`` and every write
    path is guarded by an internal lock, so the RunWriter (which may be
    invoked from a background thread) and reconcile (invoked on the main
    thread at startup) can share a single instance safely.
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            str(db_path),
            check_same_thread=False,
            isolation_level=None,  # autocommit
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        with self._lock:
            self._conn.executescript(_SCHEMA_SQL)

    @property
    def db_path(self) -> Path:
        return self._db_path

    def close(self) -> None:
        """Close the underlying connection. Idempotent."""
        with self._lock, contextlib.suppress(sqlite3.ProgrammingError):
            self._conn.close()

    def __enter__(self) -> RunIndex:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def upsert(
        self,
        *,
        run_id: str,
        skill_slug: str,
        status: str,
        mode: str,
        started_at: str | None = None,
        ended_at: str | None = None,
        duration_seconds: float | None = None,
        total_cost_usd: float | None = None,
        final_step_reached: int | None = None,
        abort_reason: str | None = None,
    ) -> None:
        """Insert or update a row for ``run_id``.

        The primary-key ``ON CONFLICT`` clause makes repeated upserts a no-op
        when nothing changed and an in-place update otherwise.
        """
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO runs (
                    run_id, skill_slug, status, mode, started_at, ended_at,
                    duration_seconds, total_cost_usd, final_step_reached,
                    abort_reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    skill_slug=excluded.skill_slug,
                    status=excluded.status,
                    mode=excluded.mode,
                    started_at=excluded.started_at,
                    ended_at=excluded.ended_at,
                    duration_seconds=excluded.duration_seconds,
                    total_cost_usd=excluded.total_cost_usd,
                    final_step_reached=excluded.final_step_reached,
                    abort_reason=excluded.abort_reason
                """,
                (
                    run_id,
                    skill_slug,
                    status,
                    mode,
                    started_at,
                    ended_at,
                    duration_seconds,
                    total_cost_usd,
                    final_step_reached,
                    abort_reason,
                ),
            )

    def upsert_from_metadata(self, metadata: RunMetadata) -> None:
        """Upsert a row from a :class:`~runner.schema.RunMetadata`."""
        data = metadata.to_dict()
        started = data.get("started_at")
        ended = data.get("ended_at")
        duration = _compute_duration(started, ended)
        self.upsert(
            run_id=str(metadata.run_id),
            skill_slug=metadata.skill_slug,
            status=metadata.status,
            mode=metadata.mode,
            started_at=started,
            ended_at=ended,
            duration_seconds=duration,
            total_cost_usd=data.get("total_cost_usd"),
            final_step_reached=data.get("final_step_reached"),
            abort_reason=data.get("abort_reason"),
        )

    def get(self, run_id: str) -> dict[str, Any] | None:
        """Return the row for ``run_id`` as a dict, or ``None`` if missing."""
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM runs WHERE run_id = ?", (run_id,)
            )
            row = cur.fetchone()
        if row is None:
            return None
        return _row_to_dict(row, cur.description)

    def list(
        self,
        *,
        skill_slug: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Return run summaries newest-first, optionally filtered by slug.

        Ordering uses ``COALESCE(started_at, '')`` so rows still in the
        ``pending`` state (no ``started_at`` yet) sort to the bottom instead
        of being treated as newest by a NULL-first collation.
        """
        if limit < 0:
            raise ValueError(f"limit must be non-negative, got {limit}")
        if offset < 0:
            raise ValueError(f"offset must be non-negative, got {offset}")
        sql = (
            "SELECT * FROM runs"
            + (" WHERE skill_slug = ?" if skill_slug is not None else "")
            + " ORDER BY COALESCE(started_at, '') DESC, run_id ASC"
            + " LIMIT ? OFFSET ?"
        )
        params: tuple[Any, ...] = (
            (skill_slug, limit, offset)
            if skill_slug is not None
            else (limit, offset)
        )
        with self._lock:
            cur = self._conn.execute(sql, params)
            rows = cur.fetchall()
            description = cur.description
        return [_row_to_dict(r, description) for r in rows]

    def reconcile(self, runs_root: Path) -> int:
        """Rebuild the index from on-disk run directories.

        Scans every ``<runs_root>/<run_id>/run_metadata.json`` file and
        upserts a row for it. Rows whose on-disk status is non-terminal and
        non-waiting (i.e. ``pending`` or ``running``) are treated as crashed
        mid-run: the row is written as ``failed`` with
        ``abort_reason='incomplete_on_restart'`` and the on-disk metadata is
        rewritten so the UI no longer sees a permanent ``running`` row.

        Returns the number of rows synced.
        """
        if not runs_root.is_dir():
            return 0

        synced = 0
        from runner.run_writer import _atomic_write_bytes

        for child in sorted(runs_root.iterdir()):
            metadata_path = child / "run_metadata.json"
            if not metadata_path.is_file():
                continue
            try:
                raw = json.loads(metadata_path.read_text(encoding="utf-8"))
                metadata = RunMetadata.model_validate(raw)
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                logger.warning(
                    "reconcile skipping unreadable metadata at %s: %s",
                    metadata_path,
                    exc,
                )
                continue

            if (
                metadata.status not in TERMINAL_STATUSES
                and metadata.status != WAITING_STATUS
            ):
                merged = metadata.model_dump() | {
                    "status": "failed",
                    "abort_reason": INCOMPLETE_ON_RESTART_REASON,
                    "error_message": INCOMPLETE_ON_RESTART_REASON,
                }
                metadata = RunMetadata.model_validate(merged)
                payload = metadata.to_dict()
                serialized = (
                    json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
                    + b"\n"
                )
                try:
                    _atomic_write_bytes(metadata_path, serialized)
                except OSError as exc:  # pragma: no cover - best-effort
                    logger.warning(
                        "reconcile could not rewrite %s: %s", metadata_path, exc
                    )

            self.upsert_from_metadata(metadata)
            synced += 1

        return synced


def _compute_duration(started: str | None, ended: str | None) -> float | None:
    """Return ``ended_at - started_at`` in seconds, or ``None`` if either missing."""
    if not isinstance(started, str) or not isinstance(ended, str):
        return None
    try:
        s = datetime.fromisoformat(started.replace("Z", "+00:00"))
        e = datetime.fromisoformat(ended.replace("Z", "+00:00"))
    except ValueError:
        return None
    return (e - s).total_seconds()


def _row_to_dict(
    row: tuple[Any, ...], description: Any
) -> dict[str, Any]:
    return {col[0]: row[i] for i, col in enumerate(description)}


__all__ = [
    "INCOMPLETE_ON_RESTART_REASON",
    "RunIndex",
]
