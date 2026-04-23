"""Tests for :mod:`recorder.index_db` (R-012).

Covers the acceptance criteria from ``scripts/ralph/prd.json``:

* insert / update / delete / list round-trips
* reconcile from filesystem (add, update, remove stale rows)
* reconcile removes rows when trajectory directories are deleted by hand
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from recorder.index_db import IndexDB
from recorder.writer import TrajectoryWriter


def _write_disk_trajectory(
    root: Path,
    trajectory_id: str,
    *,
    label: str = "demo",
    started_at: str = "2026-04-23T12:00:00.000+00:00",
    stopped_at: str | None = "2026-04-23T12:00:02.000+00:00",
    event_count: int = 3,
) -> Path:
    traj_dir = root / trajectory_id
    traj_dir.mkdir(parents=True)
    metadata = {
        "id": trajectory_id,
        "label": label,
        "started_at": started_at,
        "stopped_at": stopped_at,
        "display_info": {"width": 1, "height": 1, "scale_factor": 1.0},
        "app_focus_history": [],
    }
    (traj_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    events_lines = [
        json.dumps(
            {
                "seq": i + 1,
                "timestamp_ms": 1_700_000_000_000 + i,
                "type": "keyframe",
                "screenshot_ref": None,
                "app": {"bundle_id": "c.t", "name": "T", "pid": 1},
                "target": None,
                "payload": {"reason": "periodic"},
            },
            separators=(",", ":"),
        )
        for i in range(event_count)
    ]
    (traj_dir / "events.jsonl").write_text("\n".join(events_lines) + "\n", encoding="utf-8")
    return traj_dir


# --------------------------------------------------------------- construction


def test_index_db_creates_file_and_parent(tmp_path: Path) -> None:
    db_path = tmp_path / "nested" / "dir" / "index.db"
    db = IndexDB(db_path)
    try:
        assert db_path.is_file()
    finally:
        db.close()


def test_index_db_reopen_is_idempotent(tmp_path: Path) -> None:
    db_path = tmp_path / "index.db"
    db1 = IndexDB(db_path)
    db1.upsert(
        trajectory_id="a",
        label="x",
        started_at="2026-04-23T00:00:00+00:00",
    )
    db1.close()

    db2 = IndexDB(db_path)
    try:
        assert db2.get("a") is not None
    finally:
        db2.close()


# ----------------------------------------------------------------- CRUD


def test_upsert_then_get(tmp_path: Path) -> None:
    with IndexDB(tmp_path / "index.db") as db:
        db.upsert(
            trajectory_id="abc",
            label="demo",
            started_at="2026-04-23T12:00:00+00:00",
            stopped_at="2026-04-23T12:00:01+00:00",
            event_count=7,
            duration_ms=1000,
        )
        row = db.get("abc")
        assert row is not None
        assert row["id"] == "abc"
        assert row["label"] == "demo"
        assert row["event_count"] == 7
        assert row["duration_ms"] == 1000


def test_upsert_replaces_existing_row(tmp_path: Path) -> None:
    with IndexDB(tmp_path / "index.db") as db:
        db.upsert(
            trajectory_id="abc",
            label="first",
            started_at="2026-04-23T12:00:00+00:00",
            event_count=1,
        )
        db.upsert(
            trajectory_id="abc",
            label="second",
            started_at="2026-04-23T13:00:00+00:00",
            event_count=5,
            duration_ms=500,
        )
        row = db.get("abc")
        assert row is not None
        assert row["label"] == "second"
        assert row["event_count"] == 5
        assert row["duration_ms"] == 500


def test_mark_closed_updates_fields(tmp_path: Path) -> None:
    with IndexDB(tmp_path / "index.db") as db:
        db.upsert(
            trajectory_id="abc",
            label="demo",
            started_at="2026-04-23T12:00:00+00:00",
        )
        updated = db.mark_closed(
            "abc",
            stopped_at="2026-04-23T12:00:05+00:00",
            event_count=10,
            duration_ms=5000,
        )
        assert updated is True
        row = db.get("abc")
        assert row is not None
        assert row["stopped_at"] == "2026-04-23T12:00:05+00:00"
        assert row["event_count"] == 10
        assert row["duration_ms"] == 5000


def test_mark_closed_returns_false_when_missing(tmp_path: Path) -> None:
    with IndexDB(tmp_path / "index.db") as db:
        updated = db.mark_closed(
            "nope", stopped_at=None, event_count=0, duration_ms=None
        )
        assert updated is False


def test_delete_removes_row(tmp_path: Path) -> None:
    with IndexDB(tmp_path / "index.db") as db:
        db.upsert(trajectory_id="abc", label=None, started_at=None)
        assert db.delete("abc") is True
        assert db.get("abc") is None
        assert db.delete("abc") is False  # idempotent-ish; returns False on missing


def test_list_all_is_ordered_by_started_at(tmp_path: Path) -> None:
    with IndexDB(tmp_path / "index.db") as db:
        db.upsert(trajectory_id="c", label="c", started_at="2026-04-23T12:00:02+00:00")
        db.upsert(trajectory_id="a", label="a", started_at="2026-04-23T12:00:00+00:00")
        db.upsert(trajectory_id="b", label="b", started_at="2026-04-23T12:00:01+00:00")
        rows = db.list_all()
        assert [r["id"] for r in rows] == ["a", "b", "c"]


def test_list_all_surfaces_null_started_at_first(tmp_path: Path) -> None:
    """Rows without a started_at (in-progress / not yet finalised) sort first."""
    with IndexDB(tmp_path / "index.db") as db:
        db.upsert(trajectory_id="done", label=None, started_at="2026-04-23T12:00:00+00:00")
        db.upsert(trajectory_id="pending", label=None, started_at=None)
        rows = db.list_all()
        assert rows[0]["id"] == "pending"
        assert rows[1]["id"] == "done"


# ---------------------------------------------------------------- reconcile


def test_reconcile_adds_rows_from_disk(tmp_path: Path) -> None:
    root = tmp_path / "trajectories"
    root.mkdir()
    _write_disk_trajectory(root, "aaa", label="alpha", event_count=4)
    _write_disk_trajectory(root, "bbb", label="beta", event_count=1)

    with IndexDB(tmp_path / "index.db") as db:
        result = db.reconcile(root)
        assert result["added_or_updated"] == 2
        ids = {r["id"] for r in db.list_all()}
        assert ids == {"aaa", "bbb"}
        row = db.get("aaa")
        assert row is not None
        assert row["label"] == "alpha"
        assert row["event_count"] == 4
        assert row["duration_ms"] == 2000  # 12:00:02 - 12:00:00


def test_reconcile_removes_rows_without_disk_dir(tmp_path: Path) -> None:
    """Manual deletion of a trajectory directory should purge the row on reconcile."""
    root = tmp_path / "trajectories"
    root.mkdir()
    _write_disk_trajectory(root, "survive", label="stays", event_count=2)
    _write_disk_trajectory(root, "ghost", label="disappears", event_count=2)

    with IndexDB(tmp_path / "index.db") as db:
        db.reconcile(root)
        assert db.get("ghost") is not None

        # Filesystem-level deletion simulating a user ``rm -rf`` of a
        # trajectory directory while the row still exists.
        import shutil

        shutil.rmtree(root / "ghost")

        result = db.reconcile(root)
        assert result["removed"] == 1
        assert db.get("ghost") is None
        assert db.get("survive") is not None


def test_reconcile_ignores_dirs_without_metadata(tmp_path: Path) -> None:
    root = tmp_path / "trajectories"
    root.mkdir()
    (root / "stray").mkdir()
    (root / "stray" / "garbage.txt").write_text("nope")

    with IndexDB(tmp_path / "index.db") as db:
        result = db.reconcile(root)
        assert result["added_or_updated"] == 0
        assert db.list_all() == []


def test_reconcile_ignores_corrupt_metadata(tmp_path: Path) -> None:
    root = tmp_path / "trajectories"
    root.mkdir()
    broken = root / "broken"
    broken.mkdir()
    (broken / "metadata.json").write_text("{ not valid json")

    with IndexDB(tmp_path / "index.db") as db:
        result = db.reconcile(root)
        assert result["added_or_updated"] == 0


def test_reconcile_on_nonexistent_root_is_safe(tmp_path: Path) -> None:
    with IndexDB(tmp_path / "index.db") as db:
        result = db.reconcile(tmp_path / "does-not-exist")
        assert result == {"added_or_updated": 0, "removed": 0}


def test_reconcile_updates_event_count_from_disk(tmp_path: Path) -> None:
    """If events.jsonl grew between captures, reconcile should re-count."""
    root = tmp_path / "trajectories"
    root.mkdir()
    _write_disk_trajectory(root, "abc", event_count=2)

    with IndexDB(tmp_path / "index.db") as db:
        db.reconcile(root)
        row = db.get("abc")
        assert row is not None
        assert row["event_count"] == 2

        # Append more events by rewriting the file with a longer count.
        _write_disk_trajectory(root / "..", "abc", event_count=5)  # overwrite
        # The trajectory dir already exists so rewrite is in-place; the
        # helper recreates the dir when the parent is the trajectories
        # root — here we write events.jsonl directly instead.
        traj_dir = root / "abc"
        lines = (traj_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
        extra = "\n".join(lines) + "\n" + lines[0] + "\n" + lines[0] + "\n"
        (traj_dir / "events.jsonl").write_text(extra, encoding="utf-8")

        db.reconcile(root)
        row = db.get("abc")
        assert row is not None
        assert row["event_count"] >= 4


# ----------------------------------------------------- writer integration


def test_trajectory_writer_inserts_row_on_creation(tmp_path: Path) -> None:
    db = IndexDB(tmp_path / "index.db")
    try:
        writer = TrajectoryWriter(tmp_path / "trajectories", "demo", index_db=db)
        row = db.get(writer.id)
        assert row is not None
        assert row["label"] == "demo"
        assert row["started_at"] is None  # write_metadata hasn't been called yet
        writer.close()
    finally:
        db.close()


def test_trajectory_writer_updates_row_on_close(tmp_path: Path) -> None:
    db = IndexDB(tmp_path / "index.db")
    try:
        writer = TrajectoryWriter(tmp_path / "trajectories", "demo", index_db=db)
        writer.write_metadata(
            {
                "id": writer.id,
                "label": "demo",
                "started_at": "2026-04-23T12:00:00.000+00:00",
                "stopped_at": None,
                "display_info": {"width": 1920, "height": 1080, "scale_factor": 2.0},
                "app_focus_history": [],
            }
        )
        for i in range(3):
            writer.append_event(
                {
                    "timestamp_ms": 1_700_000_000_000 + i,
                    "type": "keyframe",
                    "screenshot_ref": None,
                    "app": {"bundle_id": "c.t", "name": "T", "pid": 1},
                    "target": None,
                    "payload": {"reason": "periodic"},
                }
            )
        writer.close()

        row = db.get(writer.id)
        assert row is not None
        assert row["label"] == "demo"
        assert row["started_at"] == "2026-04-23T12:00:00.000+00:00"
        assert row["stopped_at"] is not None
        assert row["event_count"] == 3
        # duration_ms is computed from started_at (fixed in metadata) - now(),
        # so the sign depends on wall-clock time vs. the test's literal
        # timestamp. We only care that SOME value landed in the row.
        assert row["duration_ms"] is not None
    finally:
        db.close()


def test_trajectory_writer_without_index_db_is_noop(tmp_path: Path) -> None:
    """Writer must remain usable without an index (backward compat)."""
    writer = TrajectoryWriter(tmp_path / "trajectories", "demo")
    writer.write_metadata(
        {
            "id": writer.id,
            "label": "demo",
            "started_at": "2026-04-23T12:00:00.000+00:00",
            "stopped_at": None,
            "display_info": {"width": 1, "height": 1, "scale_factor": 1.0},
            "app_focus_history": [],
        }
    )
    writer.close()


# ---------------------------------------------------------- thread safety


def test_concurrent_upsert_is_safe(tmp_path: Path) -> None:
    db = IndexDB(tmp_path / "index.db")
    errors: list[Exception] = []

    def worker(idx: int) -> None:
        try:
            for i in range(10):
                db.upsert(
                    trajectory_id=f"t{idx}-{i}",
                    label="x",
                    started_at=None,
                )
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    try:
        assert errors == []
        assert len(db.list_all()) == 40
    finally:
        db.close()


# ------------------------------------------------------- storage helpers


def test_default_index_db_path_follows_profile(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from recorder.storage import default_index_db_path, default_trajectories_root

    monkeypatch.setenv("TRACE_PROFILE_DIR", str(tmp_path))
    root = default_trajectories_root()
    index_path = default_index_db_path()
    assert index_path == root.parent / "index.db"
