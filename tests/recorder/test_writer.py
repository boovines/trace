"""Tests for :mod:`recorder.writer` (R-003).

Covers the acceptance criteria from ``scripts/ralph/prd.json``:

* directory layout + 0700 permissions
* atomic metadata write
* append_event validates before writing
* zero-padded screenshot filenames
* concurrent appends from multiple threads keep seq distinct and monotonic
* round-trip: write → close → re-read
* idempotent close
"""

from __future__ import annotations

import json
import os
import stat
import threading
from pathlib import Path
from typing import Any

import pytest
from jsonschema.exceptions import ValidationError

from recorder.writer import TrajectoryWriter

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _metadata(trajectory_id: str, label: str = "demo") -> dict[str, Any]:
    return {
        "id": trajectory_id,
        "label": label,
        "started_at": "2026-04-22T10:00:00+00:00",
        "stopped_at": None,
        "display_info": {"width": 1920, "height": 1080, "scale_factor": 2.0},
        "app_focus_history": [],
    }


def _keyframe(seq: int | None = None, ts: int = 1_700_000_000_000) -> dict[str, Any]:
    event: dict[str, Any] = {
        "timestamp_ms": ts,
        "type": "keyframe",
        "screenshot_ref": None,
        "app": {"bundle_id": "com.apple.finder", "name": "Finder", "pid": 1},
        "target": None,
        "payload": {"reason": "periodic"},
    }
    if seq is not None:
        event["seq"] = seq
    return event


def test_constructor_creates_trajectory_dirs_with_0700(tmp_path: Path) -> None:
    w = TrajectoryWriter(tmp_path, label="demo")
    try:
        assert w.dir == tmp_path / w.id
        assert w.dir.is_dir()
        assert (w.dir / "screenshots").is_dir()
        assert stat.S_IMODE(os.stat(w.dir).st_mode) == 0o700
        assert stat.S_IMODE(os.stat(w.dir / "screenshots").st_mode) == 0o700
    finally:
        w.close()


def test_constructor_rejects_existing_trajectory_id(tmp_path: Path) -> None:
    w = TrajectoryWriter(tmp_path, label="demo")
    existing_id = w.id
    w.close()
    # Same root + same id → directory already exists → FileExistsError.
    with pytest.raises(FileExistsError):
        TrajectoryWriter(tmp_path, label="demo", trajectory_id=existing_id)


def test_round_trip_ten_events_three_screenshots(tmp_path: Path) -> None:
    w = TrajectoryWriter(tmp_path, label="demo")
    w.write_metadata(_metadata(w.id))
    for i in range(10):
        w.append_event(_keyframe(ts=1_700_000_000_000 + i))
    for s in (1, 5, 10):
        w.write_screenshot(s, PNG_MAGIC + b"fake-body")
    w.close()

    # metadata.json has stopped_at populated.
    md = json.loads((w.dir / "metadata.json").read_text(encoding="utf-8"))
    assert md["id"] == w.id
    assert md["label"] == "demo"
    assert md["stopped_at"] is not None

    # events.jsonl: 10 lines, seq 1..10, each valid JSON.
    lines = [
        json.loads(line)
        for line in (w.dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(lines) == 10
    assert [e["seq"] for e in lines] == list(range(1, 11))
    assert all(e["type"] == "keyframe" for e in lines)

    # screenshots are zero-padded and non-empty.
    for s in (1, 5, 10):
        p = w.dir / "screenshots" / f"{s:04d}.png"
        assert p.is_file()
        assert p.read_bytes().startswith(PNG_MAGIC)


def test_append_event_with_bad_payload_raises_before_writing(tmp_path: Path) -> None:
    w = TrajectoryWriter(tmp_path, label="demo")
    try:
        bad = _keyframe()
        bad["payload"] = {"reason": "not-an-enum-member"}
        with pytest.raises(ValidationError):
            w.append_event(bad)
        # events.jsonl should be empty — bad event must not leak to disk.
        contents = ""
        if (w.dir / "events.jsonl").is_file():
            contents = (w.dir / "events.jsonl").read_text(encoding="utf-8")
        assert contents == ""
    finally:
        w.close()


def test_write_metadata_rejects_invalid_shape(tmp_path: Path) -> None:
    w = TrajectoryWriter(tmp_path, label="demo")
    try:
        md = _metadata(w.id)
        del md["display_info"]  # required
        with pytest.raises(ValidationError):
            w.write_metadata(md)
        assert not (w.dir / "metadata.json").is_file()
    finally:
        w.close()


def test_concurrent_append_assigns_distinct_monotonic_seq(tmp_path: Path) -> None:
    w = TrajectoryWriter(tmp_path, label="concurrent")
    w.write_metadata(_metadata(w.id))
    errors: list[BaseException] = []
    per_thread = 20
    thread_count = 5

    def worker() -> None:
        try:
            for _ in range(per_thread):
                w.append_event(_keyframe())
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(thread_count)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    w.close()

    assert errors == []

    lines = [
        json.loads(line)
        for line in (w.dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    total = per_thread * thread_count
    assert len(lines) == total

    seqs = [e["seq"] for e in lines]
    # Distinct: no collisions.
    assert len(set(seqs)) == total
    # Monotonic on disk: the lock serialises append_event so write order == seq order.
    assert seqs == sorted(seqs)
    assert seqs[0] == 1
    assert seqs[-1] == total


def test_preserves_caller_provided_seq_and_advances_counter(tmp_path: Path) -> None:
    w = TrajectoryWriter(tmp_path, label="demo")
    try:
        w.append_event(_keyframe(seq=7))
        w.append_event(_keyframe())  # should auto-assign 8
        lines = [
            json.loads(line)
            for line in (w.dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert [e["seq"] for e in lines] == [7, 8]
    finally:
        w.close()


def test_close_is_idempotent(tmp_path: Path) -> None:
    w = TrajectoryWriter(tmp_path, label="demo")
    w.write_metadata(_metadata(w.id))
    w.close()
    w.close()  # must not raise
    with pytest.raises(RuntimeError):
        w.append_event(_keyframe())


def test_screenshot_filename_is_zero_padded_to_four_digits(tmp_path: Path) -> None:
    w = TrajectoryWriter(tmp_path, label="demo")
    try:
        w.write_screenshot(42, PNG_MAGIC)
        assert (w.dir / "screenshots" / "0042.png").is_file()
        # 5-digit seqs extend naturally past 4 digits (schema allows 4+).
        w.write_screenshot(12345, PNG_MAGIC)
        assert (w.dir / "screenshots" / "12345.png").is_file()
    finally:
        w.close()


def test_close_without_metadata_is_safe(tmp_path: Path) -> None:
    w = TrajectoryWriter(tmp_path, label="demo")
    # Never called write_metadata. close() should not raise and should not
    # leave a stray metadata.json lying around.
    w.close()
    assert not (w.dir / "metadata.json").is_file()


def test_context_manager_closes_writer(tmp_path: Path) -> None:
    with TrajectoryWriter(tmp_path, label="demo") as w:
        w.write_metadata(_metadata(w.id))
        w.append_event(_keyframe())
    # After context exit, appending raises.
    with pytest.raises(RuntimeError):
        w.append_event(_keyframe())
    md = json.loads((w.dir / "metadata.json").read_text(encoding="utf-8"))
    assert md["stopped_at"] is not None
