"""Tests for runner.run_writer.

Covers the full X-005 acceptance matrix:

* Full lifecycle: directory creation, status updates, 10 events, 5
  screenshots, 3 transcript turns, close.
* Concurrent append safety from 3 threads producing 30 events.
* Atomic-write crash semantics: a simulated fsync failure leaves NO ``.tmp``
  file behind and keeps the prior ``run_metadata.json`` intact.
* PNG magic-byte validation on ``write_screenshot``.
* ``close()`` idempotency.
* Run directory permissions are ``0o700``.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest import mock

import pytest
from runner.run_writer import RunWriter, get_run_dir_perms
from runner.schema import RunMetadata

_MINIMAL_PNG: bytes = (
    # 8-byte PNG signature
    b"\x89PNG\r\n\x1a\n"
    # IHDR chunk (13 bytes of data: 1x1, 8-bit, RGBA)
    b"\x00\x00\x00\rIHDR"
    b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00"
    b"\x1f\x15\xc4\x89"
    # IDAT chunk (minimal zlib stream)
    b"\x00\x00\x00\nIDAT"
    b"x\x9cc\x00\x01\x00\x00\x05\x00\x01"
    b"\r\n-\xb4"
    # IEND
    b"\x00\x00\x00\x00IEND\xaeB`\x82"
)


def _make_writer(tmp_path: Path, run_id: str | None = None) -> tuple[RunWriter, str]:
    run_id = run_id or str(uuid.uuid4())
    writer = RunWriter(
        run_id=run_id,
        skill_slug="gmail_reply",
        mode="dry_run",
        runs_root=tmp_path,
    )
    return writer, run_id


def _base_metadata(run_id: str, status: str = "pending") -> RunMetadata:
    return RunMetadata(
        run_id=uuid.UUID(run_id),
        skill_slug="gmail_reply",
        started_at=datetime(2026, 4, 22, 18, 0, 0, tzinfo=UTC),
        ended_at=None,
        status=status,  # type: ignore[arg-type]
        mode="dry_run",
    )


def test_run_directory_created_with_0700_perms(tmp_path: Path) -> None:
    writer, run_id = _make_writer(tmp_path)
    assert writer.run_dir == tmp_path / run_id
    assert writer.run_dir.is_dir()
    assert get_run_dir_perms(writer.run_dir) == 0o700
    assert (writer.run_dir / "screenshots").is_dir()
    assert get_run_dir_perms(writer.run_dir / "screenshots") == 0o700


def test_full_lifecycle(tmp_path: Path) -> None:
    writer, run_id = _make_writer(tmp_path)
    metadata = _base_metadata(run_id, status="pending")
    writer.write_metadata(metadata)

    metadata = writer.update_status(metadata, "running")
    assert metadata.status == "running"

    for i in range(10):
        writer.append_event(
            seq=i,
            event_type="step_start",
            message=f"starting step {i}",
            step_number=i,
            timestamp_ms=1_000 + i,
        )

    for i in range(5):
        writer.write_screenshot(i, _MINIMAL_PNG)

    for turn in range(3):
        writer.append_transcript(
            turn=turn,
            role="assistant",
            content=[{"type": "text", "text": f"turn {turn}"}],
            input_tokens=100,
            output_tokens=50,
            timestamp_ms=2_000 + turn,
        )

    final = writer.update_status(
        metadata,
        "succeeded",
        ended_at=datetime(2026, 4, 22, 18, 5, 0, tzinfo=UTC),
        final_step_reached=9,
    )
    writer.close()
    assert writer.closed

    # metadata round-trip
    saved = json.loads((writer.run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert saved["status"] == "succeeded"
    assert saved["final_step_reached"] == 9
    assert saved["ended_at"] == "2026-04-22T18:05:00Z"
    assert final.is_terminal()

    # events
    event_lines = (writer.run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(event_lines) == 10
    parsed = [json.loads(line) for line in event_lines]
    assert [e["seq"] for e in parsed] == list(range(10))

    # transcripts
    transcript_lines = (
        writer.run_dir / "transcript.jsonl"
    ).read_text(encoding="utf-8").splitlines()
    assert len(transcript_lines) == 3

    # screenshots
    shots = sorted((writer.run_dir / "screenshots").iterdir())
    assert [s.name for s in shots] == [f"{i:04d}.png" for i in range(5)]
    for s in shots:
        assert s.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_metadata_write_is_idempotent(tmp_path: Path) -> None:
    writer, run_id = _make_writer(tmp_path)
    metadata = _base_metadata(run_id)
    writer.write_metadata(metadata)
    first_mtime = (writer.run_dir / "run_metadata.json").stat().st_mtime_ns
    first_bytes = (writer.run_dir / "run_metadata.json").read_bytes()

    writer.write_metadata(metadata)

    # Same content → no rewrite → same file bytes.
    assert (writer.run_dir / "run_metadata.json").read_bytes() == first_bytes
    # mtime unchanged confirms the no-op fast path was taken.
    assert (writer.run_dir / "run_metadata.json").stat().st_mtime_ns == first_mtime


def test_metadata_runid_mismatch_raises(tmp_path: Path) -> None:
    writer, _run_id = _make_writer(tmp_path)
    other = _base_metadata(str(uuid.uuid4()))
    with pytest.raises(ValueError, match="run_id"):
        writer.write_metadata(other)


def test_metadata_slug_mismatch_raises(tmp_path: Path) -> None:
    writer, run_id = _make_writer(tmp_path)
    bad = _base_metadata(run_id).model_copy(update={"skill_slug": "other_skill"})
    with pytest.raises(ValueError, match="skill_slug"):
        writer.write_metadata(bad)


def test_metadata_mode_mismatch_raises(tmp_path: Path) -> None:
    writer, run_id = _make_writer(tmp_path)
    bad = _base_metadata(run_id).model_copy(update={"mode": "execute"})
    with pytest.raises(ValueError, match="mode"):
        writer.write_metadata(bad)


def test_update_status_rejects_invalid_enum(tmp_path: Path) -> None:
    writer, run_id = _make_writer(tmp_path)
    metadata = _base_metadata(run_id)
    writer.write_metadata(metadata)
    # pydantic ValidationError is the concrete class raised, but to avoid
    # importing pydantic's error hierarchy in this test we match by name
    # through a broad parent — ruff flags blind `Exception` so we use
    # ``pydantic.ValidationError`` directly.
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        writer.update_status(metadata, "banana")


def test_concurrent_append_events_from_three_threads(tmp_path: Path) -> None:
    writer, _run_id = _make_writer(tmp_path)
    total = 30
    seq_counter = iter(range(total))
    counter_lock = threading.Lock()

    def worker(label: str) -> None:
        for _ in range(10):
            with counter_lock:
                seq = next(seq_counter)
            writer.append_event(
                seq=seq,
                event_type="tool_call",
                message=f"{label}-{seq}",
            )

    threads = [
        threading.Thread(target=worker, args=(f"t{i}",), name=f"t{i}")
        for i in range(3)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    lines = (writer.run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == total
    records: list[dict[str, Any]] = [json.loads(line) for line in lines]
    # Every line must parse (no partial writes interleaved).
    assert all("seq" in r for r in records)
    # All 30 distinct seq values are present.
    assert sorted(r["seq"] for r in records) == list(range(total))


def test_fsync_failure_removes_tmp_and_preserves_existing_metadata(tmp_path: Path) -> None:
    writer, run_id = _make_writer(tmp_path)
    metadata = _base_metadata(run_id)
    writer.write_metadata(metadata)

    original_bytes = (writer.run_dir / "run_metadata.json").read_bytes()
    original_mtime = (writer.run_dir / "run_metadata.json").stat().st_mtime_ns

    # Mutate the metadata so the fast-path idempotency check does NOT short-circuit.
    next_metadata = metadata.model_copy(update={"status": "running"})

    real_fsync = os.fsync

    def flaky_fsync(fd: int) -> None:
        # Fail the fsync that occurs during atomic metadata write, but only after
        # the tmp file was created — simulating a mid-write crash.
        raise OSError("simulated disk failure during fsync")

    with (
        mock.patch("runner.run_writer.os.fsync", side_effect=flaky_fsync),
        pytest.raises(OSError, match="simulated disk failure"),
    ):
        writer.write_metadata(next_metadata)

    # .tmp sibling must have been cleaned up.
    tmp_candidates = list(writer.run_dir.glob("run_metadata.json.tmp"))
    assert tmp_candidates == []
    # Existing metadata untouched.
    assert (writer.run_dir / "run_metadata.json").read_bytes() == original_bytes
    assert (writer.run_dir / "run_metadata.json").stat().st_mtime_ns == original_mtime
    # Subsequent writes still work because fsync is restored after the patch.
    assert os.fsync is real_fsync
    writer.write_metadata(next_metadata)
    assert b'"running"' in (writer.run_dir / "run_metadata.json").read_bytes()


def test_write_screenshot_rejects_invalid_png(tmp_path: Path) -> None:
    writer, _run_id = _make_writer(tmp_path)
    with pytest.raises(ValueError, match="PNG magic"):
        writer.write_screenshot(0, b"not a png at all")
    # No file created on failure.
    assert list((writer.run_dir / "screenshots").iterdir()) == []


def test_write_screenshot_rejects_negative_seq(tmp_path: Path) -> None:
    writer, _run_id = _make_writer(tmp_path)
    with pytest.raises(ValueError, match="non-negative"):
        writer.write_screenshot(-1, _MINIMAL_PNG)


def test_write_screenshot_filename_is_zero_padded(tmp_path: Path) -> None:
    writer, _run_id = _make_writer(tmp_path)
    path = writer.write_screenshot(42, _MINIMAL_PNG)
    assert path.name == "0042.png"


def test_close_is_idempotent(tmp_path: Path) -> None:
    writer, _run_id = _make_writer(tmp_path)
    writer.close()
    writer.close()
    assert writer.closed


def test_context_manager_closes(tmp_path: Path) -> None:
    run_id = str(uuid.uuid4())
    with RunWriter(
        run_id=run_id,
        skill_slug="gmail_reply",
        mode="dry_run",
        runs_root=tmp_path,
    ) as writer:
        assert not writer.closed
    assert writer.closed


def test_events_jsonl_lines_are_individually_parseable(tmp_path: Path) -> None:
    writer, _run_id = _make_writer(tmp_path)
    writer.append_event(seq=0, event_type="step_start", message="ok", step_number=1)
    writer.append_event(
        seq=1,
        event_type="confirmation_requested",
        message="waiting on user",
        step_number=5,
        screenshot_seq=3,
    )
    raw = (writer.run_dir / "events.jsonl").read_text(encoding="utf-8")
    first, second, _ = raw.split("\n")
    a = json.loads(first)
    b = json.loads(second)
    assert a["screenshot_ref"] is None
    assert b["screenshot_ref"] == "0003.png"
    assert b["step_number"] == 5


def test_transcript_jsonl_preserves_content_blocks(tmp_path: Path) -> None:
    writer, _run_id = _make_writer(tmp_path)
    content = [{"type": "text", "text": "hello"}, {"type": "tool_use", "id": "t1"}]
    writer.append_transcript(
        turn=0,
        role="assistant",
        content=content,
        input_tokens=10,
        output_tokens=5,
    )
    line = (writer.run_dir / "transcript.jsonl").read_text(encoding="utf-8").strip()
    record = json.loads(line)
    assert record["content"] == content
    assert record["input_tokens_this_turn"] == 10
    assert record["output_tokens_this_turn"] == 5
