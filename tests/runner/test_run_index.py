"""Tests for runner.run_index (X-020).

Covers the acceptance matrix:

* ``upsert`` inserts a row, subsequent calls update the same row.
* ``list`` returns newest-first and supports a ``skill_slug`` filter.
* ``list`` supports ``limit`` / ``offset`` pagination.
* RunWriter integration: constructing a writer with a RunIndex inserts a
  row, and every ``update_status`` upserts the new values.
* ``reconcile`` scans the on-disk runs root and rebuilds the index.
* ``reconcile`` turns a crashed-mid-run (``running`` on disk) into a
  terminal ``failed`` with ``abort_reason='incomplete_on_restart'`` and
  rewrites the metadata file so the UI does not see a stale status.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pytest
from runner.run_index import INCOMPLETE_ON_RESTART_REASON, RunIndex
from runner.run_writer import RunWriter
from runner.schema import RunMetadata


def _metadata(
    run_id: str,
    *,
    skill_slug: str = "gmail_reply",
    status: str = "pending",
    started_at: datetime | None = None,
    ended_at: datetime | None = None,
    total_cost_usd: float | None = None,
    final_step_reached: int | None = None,
    abort_reason: str | None = None,
    mode: str = "dry_run",
) -> RunMetadata:
    return RunMetadata(
        run_id=uuid.UUID(run_id),
        skill_slug=skill_slug,
        started_at=started_at or datetime(2026, 4, 22, 18, 0, 0, tzinfo=UTC),
        ended_at=ended_at,
        status=status,  # type: ignore[arg-type]
        mode=mode,  # type: ignore[arg-type]
        total_cost_usd=total_cost_usd,
        final_step_reached=final_step_reached,
        abort_reason=abort_reason,
    )


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "index.db"


def test_init_creates_table_and_indexes(db_path: Path) -> None:
    import sqlite3

    RunIndex(db_path).close()
    assert db_path.is_file()
    conn = sqlite3.connect(str(db_path))
    try:
        names = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type IN ('table', 'index') ORDER BY name"
            ).fetchall()
        }
    finally:
        conn.close()
    assert "runs" in names
    assert "runs_started_at_idx" in names
    assert "runs_skill_slug_idx" in names


def test_upsert_and_get_roundtrip(db_path: Path) -> None:
    with RunIndex(db_path) as index:
        run_id = str(uuid.uuid4())
        index.upsert(
            run_id=run_id,
            skill_slug="gmail_reply",
            status="pending",
            mode="dry_run",
        )
        row = index.get(run_id)
        assert row is not None
        assert row["run_id"] == run_id
        assert row["skill_slug"] == "gmail_reply"
        assert row["status"] == "pending"
        assert row["mode"] == "dry_run"
        assert row["started_at"] is None
        assert row["ended_at"] is None


def test_upsert_updates_existing_row(db_path: Path) -> None:
    with RunIndex(db_path) as index:
        run_id = str(uuid.uuid4())
        index.upsert(
            run_id=run_id,
            skill_slug="gmail_reply",
            status="pending",
            mode="dry_run",
        )
        index.upsert(
            run_id=run_id,
            skill_slug="gmail_reply",
            status="succeeded",
            mode="dry_run",
            started_at="2026-04-22T18:00:00Z",
            ended_at="2026-04-22T18:05:00Z",
            duration_seconds=300.0,
            total_cost_usd=0.01,
            final_step_reached=5,
        )
        row = index.get(run_id)
        assert row is not None
        assert row["status"] == "succeeded"
        assert row["duration_seconds"] == 300.0
        assert row["total_cost_usd"] == pytest.approx(0.01)
        assert row["final_step_reached"] == 5


def test_upsert_from_metadata_computes_duration(db_path: Path) -> None:
    with RunIndex(db_path) as index:
        run_id = str(uuid.uuid4())
        metadata = _metadata(
            run_id,
            status="succeeded",
            started_at=datetime(2026, 4, 22, 18, 0, 0, tzinfo=UTC),
            ended_at=datetime(2026, 4, 22, 18, 2, 30, tzinfo=UTC),
            total_cost_usd=0.02,
        )
        index.upsert_from_metadata(metadata)
        row = index.get(run_id)
        assert row is not None
        assert row["duration_seconds"] == pytest.approx(150.0)
        assert row["total_cost_usd"] == pytest.approx(0.02)


def test_list_ordering_newest_first(db_path: Path) -> None:
    with RunIndex(db_path) as index:
        older_id = str(uuid.uuid4())
        newer_id = str(uuid.uuid4())
        index.upsert(
            run_id=older_id,
            skill_slug="gmail_reply",
            status="succeeded",
            mode="dry_run",
            started_at="2026-04-22T10:00:00Z",
        )
        index.upsert(
            run_id=newer_id,
            skill_slug="gmail_reply",
            status="succeeded",
            mode="dry_run",
            started_at="2026-04-22T18:00:00Z",
        )
        rows = index.list()
        assert [r["run_id"] for r in rows] == [newer_id, older_id]


def test_list_filter_by_skill_slug(db_path: Path) -> None:
    with RunIndex(db_path) as index:
        gmail_id = str(uuid.uuid4())
        slack_id = str(uuid.uuid4())
        index.upsert(
            run_id=gmail_id,
            skill_slug="gmail_reply",
            status="succeeded",
            mode="dry_run",
            started_at="2026-04-22T10:00:00Z",
        )
        index.upsert(
            run_id=slack_id,
            skill_slug="slack_status",
            status="succeeded",
            mode="dry_run",
            started_at="2026-04-22T11:00:00Z",
        )
        rows = index.list(skill_slug="gmail_reply")
        assert len(rows) == 1
        assert rows[0]["run_id"] == gmail_id


def test_list_pagination(db_path: Path) -> None:
    with RunIndex(db_path) as index:
        ids = []
        for i in range(5):
            rid = str(uuid.uuid4())
            ids.append(rid)
            index.upsert(
                run_id=rid,
                skill_slug="gmail_reply",
                status="succeeded",
                mode="dry_run",
                started_at=f"2026-04-22T1{i}:00:00Z",
            )
        first = index.list(limit=2, offset=0)
        second = index.list(limit=2, offset=2)
        third = index.list(limit=2, offset=4)
        assert len(first) == 2
        assert len(second) == 2
        assert len(third) == 1
        # No overlap, all distinct.
        seen = {r["run_id"] for r in first + second + third}
        assert seen == set(ids)


def test_list_rejects_negative_limit_or_offset(db_path: Path) -> None:
    with RunIndex(db_path) as index:
        with pytest.raises(ValueError, match="limit"):
            index.list(limit=-1)
        with pytest.raises(ValueError, match="offset"):
            index.list(offset=-1)


def test_list_with_null_started_at_sorts_last(db_path: Path) -> None:
    with RunIndex(db_path) as index:
        pending_id = str(uuid.uuid4())
        done_id = str(uuid.uuid4())
        index.upsert(
            run_id=pending_id,
            skill_slug="gmail_reply",
            status="pending",
            mode="dry_run",
        )
        index.upsert(
            run_id=done_id,
            skill_slug="gmail_reply",
            status="succeeded",
            mode="dry_run",
            started_at="2026-04-22T10:00:00Z",
        )
        rows = index.list()
        assert [r["run_id"] for r in rows] == [done_id, pending_id]


# ---------- RunWriter integration ----------


def test_run_writer_inserts_pending_row_on_construction(
    tmp_path: Path, db_path: Path
) -> None:
    with RunIndex(db_path) as index:
        run_id = str(uuid.uuid4())
        RunWriter(
            run_id=run_id,
            skill_slug="gmail_reply",
            mode="dry_run",
            runs_root=tmp_path,
            run_index=index,
        )
        row = index.get(run_id)
        assert row is not None
        assert row["status"] == "pending"
        assert row["mode"] == "dry_run"
        assert row["skill_slug"] == "gmail_reply"


def test_run_writer_upserts_on_status_change(
    tmp_path: Path, db_path: Path
) -> None:
    with RunIndex(db_path) as index:
        run_id = str(uuid.uuid4())
        writer = RunWriter(
            run_id=run_id,
            skill_slug="gmail_reply",
            mode="dry_run",
            runs_root=tmp_path,
            run_index=index,
        )
        metadata = _metadata(run_id, status="running")
        writer.write_metadata(metadata)
        assert index.get(run_id)["status"] == "running"  # type: ignore[index]

        writer.update_status(
            metadata,
            "succeeded",
            ended_at=datetime(2026, 4, 22, 18, 2, 30, tzinfo=UTC),
            total_cost_usd=0.03,
            final_step_reached=7,
        )
        row = index.get(run_id)
        assert row is not None
        assert row["status"] == "succeeded"
        assert row["duration_seconds"] == pytest.approx(150.0)
        assert row["total_cost_usd"] == pytest.approx(0.03)
        assert row["final_step_reached"] == 7


def test_run_writer_without_index_is_still_functional(tmp_path: Path) -> None:
    # Back-compat: index is opt-in.
    run_id = str(uuid.uuid4())
    writer = RunWriter(
        run_id=run_id,
        skill_slug="gmail_reply",
        mode="dry_run",
        runs_root=tmp_path,
    )
    writer.write_metadata(_metadata(run_id))
    assert (writer.run_dir / "run_metadata.json").is_file()


# ---------- reconcile ----------


def _write_metadata_file(
    runs_root: Path, metadata: RunMetadata
) -> Path:
    run_dir = runs_root / str(metadata.run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "run_metadata.json"
    path.write_text(
        json.dumps(metadata.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def test_reconcile_populates_index_from_disk(
    tmp_path: Path, db_path: Path
) -> None:
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    done = _metadata(
        str(uuid.uuid4()),
        status="succeeded",
        ended_at=datetime(2026, 4, 22, 18, 5, 0, tzinfo=UTC),
        total_cost_usd=0.04,
    )
    _write_metadata_file(runs_root, done)

    with RunIndex(db_path) as index:
        count = index.reconcile(runs_root)
        assert count == 1
        row = index.get(str(done.run_id))
        assert row is not None
        assert row["status"] == "succeeded"
        assert row["total_cost_usd"] == pytest.approx(0.04)


def test_reconcile_handles_crash_mid_run(
    tmp_path: Path, db_path: Path
) -> None:
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    crashed = _metadata(
        str(uuid.uuid4()),
        status="running",
    )
    path = _write_metadata_file(runs_root, crashed)

    with RunIndex(db_path) as index:
        count = index.reconcile(runs_root)
        assert count == 1
        row = index.get(str(crashed.run_id))
        assert row is not None
        assert row["status"] == "failed"
        assert row["abort_reason"] == INCOMPLETE_ON_RESTART_REASON

    # On-disk metadata was rewritten so the UI no longer sees `running`.
    rewritten = json.loads(path.read_text(encoding="utf-8"))
    assert rewritten["status"] == "failed"
    assert rewritten["abort_reason"] == INCOMPLETE_ON_RESTART_REASON


def test_reconcile_leaves_terminal_runs_alone(
    tmp_path: Path, db_path: Path
) -> None:
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    aborted = _metadata(
        str(uuid.uuid4()),
        status="aborted",
        ended_at=datetime(2026, 4, 22, 18, 5, 0, tzinfo=UTC),
        abort_reason="user_abort",
    )
    path = _write_metadata_file(runs_root, aborted)
    original_bytes = path.read_bytes()

    with RunIndex(db_path) as index:
        index.reconcile(runs_root)
        row = index.get(str(aborted.run_id))
        assert row is not None
        assert row["status"] == "aborted"
        assert row["abort_reason"] == "user_abort"

    assert path.read_bytes() == original_bytes


def test_reconcile_leaves_awaiting_confirmation_alone(
    tmp_path: Path, db_path: Path
) -> None:
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    waiting = _metadata(
        str(uuid.uuid4()),
        status="awaiting_confirmation",
    )
    path = _write_metadata_file(runs_root, waiting)
    original_bytes = path.read_bytes()

    with RunIndex(db_path) as index:
        index.reconcile(runs_root)
        row = index.get(str(waiting.run_id))
        assert row is not None
        # Waiting is NOT a terminal status but we still do not overwrite it.
        assert row["status"] == "awaiting_confirmation"

    assert path.read_bytes() == original_bytes


def test_reconcile_skips_unreadable_metadata(
    tmp_path: Path, db_path: Path
) -> None:
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    (runs_root / "garbage").mkdir()
    (runs_root / "garbage" / "run_metadata.json").write_text(
        "{ not valid json", encoding="utf-8"
    )
    good = _metadata(str(uuid.uuid4()), status="succeeded")
    _write_metadata_file(runs_root, good)

    with RunIndex(db_path) as index:
        count = index.reconcile(runs_root)
        assert count == 1
        assert index.get(str(good.run_id)) is not None


def test_reconcile_returns_zero_when_runs_root_missing(db_path: Path) -> None:
    with RunIndex(db_path) as index:
        assert index.reconcile(Path("/nonexistent/path/for/ralph/test")) == 0
