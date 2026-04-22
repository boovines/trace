"""Tests for :mod:`synthesizer.trajectory_reader`.

Covers the S-004 acceptance criteria:

* Each of the 5 fixture trajectories loads and produces a plausible summary.
* Malformed ``events.jsonl`` raises with line number AND event ``seq``.
* A missing screenshot returns ``None`` and logs a warning.
* ``metadata.json`` with extra keys is accepted; missing required keys raises.
* The reader never mutates files on disk.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import pytest
from synthesizer.trajectory_reader import (
    Event,
    TrajectoryReader,
    TrajectoryReadError,
)

REFERENCE_SLUGS = (
    "gmail_reply",
    "calendar_block",
    "finder_organize",
    "slack_status",
    "notes_daily",
)


def _fixtures_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "fixtures" / "trajectories").is_dir():
            return candidate / "fixtures" / "trajectories"
    raise FileNotFoundError("Could not locate fixtures/trajectories from test file")


FIXTURES_ROOT = _fixtures_root()


@pytest.fixture
def fixture_trajectory(tmp_path: Path, request: pytest.FixtureRequest) -> Path:
    """Copy a reference fixture trajectory into ``tmp_path`` so tests can mutate it.

    The fixtures under ``fixtures/trajectories/`` are checked-in ground truth;
    tests that mangle events.jsonl or metadata.json must operate on a copy.
    """
    slug: str = request.param
    dest = tmp_path / slug
    shutil.copytree(FIXTURES_ROOT / slug, dest)
    return dest


# --- Happy paths on every reference fixture -------------------------------


@pytest.mark.parametrize("slug", REFERENCE_SLUGS)
def test_reads_reference_fixture(slug: str) -> None:
    reader = TrajectoryReader(FIXTURES_ROOT / slug)
    summary = reader.summary()
    assert summary["event_count"] >= 4
    assert summary["duration_ms"] > 0
    assert isinstance(summary["app_focus_history"], list)
    assert summary["app_focus_history"], "fixture should record at least one app focus"
    # Every reference fixture includes exactly one keyframe screenshot.
    assert summary["keyframe_count"] == 1
    # Every reference fixture starts with an app_switch event.
    assert summary["app_switch_count"] >= 1


@pytest.mark.parametrize("slug", REFERENCE_SLUGS)
def test_iter_events_is_seq_ordered(slug: str) -> None:
    reader = TrajectoryReader(FIXTURES_ROOT / slug)
    seqs = [e.seq for e in reader.iter_events()]
    assert seqs == sorted(seqs)
    assert seqs[0] >= 1
    assert len(set(seqs)) == len(seqs)


def test_iter_events_by_type_filters() -> None:
    reader = TrajectoryReader(FIXTURES_ROOT / "gmail_reply")
    clicks = list(reader.iter_events_by_type("click"))
    assert all(isinstance(e, Event) for e in clicks)
    assert all(e.kind == "click" for e in clicks)
    # gmail_reply has 2 clicks: Reply, Send
    assert len(clicks) == 2
    labels = [e.target["label"] for e in clicks if e.target is not None]
    assert "Reply" in labels and "Send" in labels


def test_summary_counts_match_fixture_content() -> None:
    reader = TrajectoryReader(FIXTURES_ROOT / "gmail_reply")
    summary = reader.summary()
    # gmail_reply: 1 app_switch + 2 clicks + 1 text_input = 4 events
    assert summary["event_count"] == 4
    assert summary["click_count"] == 2
    assert summary["text_input_count"] == 1
    assert summary["app_switch_count"] == 1


# --- Constructor behaviour ------------------------------------------------


def test_missing_directory_raises_file_not_found(tmp_path: Path) -> None:
    missing = tmp_path / "nope"
    with pytest.raises(FileNotFoundError, match="does not exist"):
        TrajectoryReader(missing)


def test_file_instead_of_directory_raises(tmp_path: Path) -> None:
    path = tmp_path / "traj.txt"
    path.write_text("not a directory", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="not a directory"):
        TrajectoryReader(path)


def test_missing_metadata_raises(tmp_path: Path) -> None:
    empty = tmp_path / "empty_trajectory"
    empty.mkdir()
    with pytest.raises(TrajectoryReadError, match=r"metadata\.json not found"):
        TrajectoryReader(empty)


# --- Malformed events.jsonl ----------------------------------------------


@pytest.mark.parametrize("fixture_trajectory", ["gmail_reply"], indirect=True)
def test_malformed_event_reports_line_and_seq(fixture_trajectory: Path) -> None:
    """A bad event line must surface both the line number AND the event seq."""
    events_path = fixture_trajectory / "events.jsonl"
    lines = events_path.read_text(encoding="utf-8").splitlines()
    # Corrupt line 2 (seq=2) — remove the required 'kind' field.
    corrupted = json.loads(lines[1])
    del corrupted["kind"]
    lines[1] = json.dumps(corrupted)
    events_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(TrajectoryReadError) as excinfo:
        TrajectoryReader(fixture_trajectory)
    message = str(excinfo.value)
    assert "line 2" in message
    assert "seq=2" in message


@pytest.mark.parametrize("fixture_trajectory", ["gmail_reply"], indirect=True)
def test_invalid_json_in_events_reports_line(fixture_trajectory: Path) -> None:
    events_path = fixture_trajectory / "events.jsonl"
    lines = events_path.read_text(encoding="utf-8").splitlines()
    lines[2] = "{not valid json"
    events_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(TrajectoryReadError) as excinfo:
        TrajectoryReader(fixture_trajectory)
    assert "line 3" in str(excinfo.value)
    assert "invalid JSON" in str(excinfo.value)


@pytest.mark.parametrize("fixture_trajectory", ["gmail_reply"], indirect=True)
def test_duplicate_seq_raises(fixture_trajectory: Path) -> None:
    events_path = fixture_trajectory / "events.jsonl"
    lines = events_path.read_text(encoding="utf-8").splitlines()
    # Rewrite line 3 to share seq with line 2.
    event = json.loads(lines[2])
    event["seq"] = 2
    lines[2] = json.dumps(event)
    events_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(TrajectoryReadError, match="duplicate seq=2"):
        TrajectoryReader(fixture_trajectory)


@pytest.mark.parametrize("fixture_trajectory", ["gmail_reply"], indirect=True)
def test_blank_lines_in_events_are_skipped(fixture_trajectory: Path) -> None:
    events_path = fixture_trajectory / "events.jsonl"
    body = events_path.read_text(encoding="utf-8")
    events_path.write_text("\n\n" + body + "\n\n", encoding="utf-8")

    reader = TrajectoryReader(fixture_trajectory)
    # gmail_reply: 4 real events; blank lines are ignored.
    assert reader.summary()["event_count"] == 4


# --- Screenshots ---------------------------------------------------------


def test_get_screenshot_path_returns_existing_file() -> None:
    reader = TrajectoryReader(FIXTURES_ROOT / "gmail_reply")
    path = reader.get_screenshot_path(1)
    assert path is not None
    assert path.is_file()
    assert path.name == "0001.png"


def test_get_screenshot_path_none_when_event_has_no_ref() -> None:
    reader = TrajectoryReader(FIXTURES_ROOT / "gmail_reply")
    # seq=2 is a click with no screenshot_ref
    assert reader.get_screenshot_path(2) is None


def test_get_screenshot_path_none_when_seq_missing() -> None:
    reader = TrajectoryReader(FIXTURES_ROOT / "gmail_reply")
    assert reader.get_screenshot_path(9999) is None


@pytest.mark.parametrize("fixture_trajectory", ["gmail_reply"], indirect=True)
def test_missing_screenshot_returns_none_and_warns(
    fixture_trajectory: Path, caplog: pytest.LogCaptureFixture
) -> None:
    (fixture_trajectory / "screenshots" / "0001.png").unlink()
    reader = TrajectoryReader(fixture_trajectory)
    with caplog.at_level(logging.WARNING, logger="synthesizer.trajectory_reader"):
        path = reader.get_screenshot_path(1)
    assert path is None
    assert any("missing on disk" in r.message for r in caplog.records)


# --- Metadata forward-compat vs required-key enforcement ------------------


@pytest.mark.parametrize("fixture_trajectory", ["gmail_reply"], indirect=True)
def test_metadata_with_extra_keys_is_accepted(fixture_trajectory: Path) -> None:
    meta_path = fixture_trajectory / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["future_field"] = "something the recorder will add later"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    reader = TrajectoryReader(fixture_trajectory)
    assert reader.metadata["future_field"] == "something the recorder will add later"


@pytest.mark.parametrize("fixture_trajectory", ["gmail_reply"], indirect=True)
def test_metadata_missing_required_key_raises(fixture_trajectory: Path) -> None:
    meta_path = fixture_trajectory / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    del meta["started_at"]
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(TrajectoryReadError, match="schema validation"):
        TrajectoryReader(fixture_trajectory)


@pytest.mark.parametrize("fixture_trajectory", ["gmail_reply"], indirect=True)
def test_metadata_invalid_uuid_raises(fixture_trajectory: Path) -> None:
    meta_path = fixture_trajectory / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["id"] = "not-a-uuid"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(TrajectoryReadError, match="schema validation"):
        TrajectoryReader(fixture_trajectory)


# --- Read-only semantics -------------------------------------------------


@pytest.mark.parametrize("slug", REFERENCE_SLUGS)
def test_reader_does_not_mutate_trajectory(slug: str, tmp_path: Path) -> None:
    """After construction + every accessor, the on-disk bytes are unchanged."""
    src = FIXTURES_ROOT / slug
    dest = tmp_path / slug
    shutil.copytree(src, dest)

    before = {p.relative_to(dest): p.read_bytes() for p in dest.rglob("*") if p.is_file()}

    reader = TrajectoryReader(dest)
    list(reader.iter_events())
    list(reader.iter_events_by_type("click"))
    reader.get_screenshot_path(1)
    reader.summary()

    after = {p.relative_to(dest): p.read_bytes() for p in dest.rglob("*") if p.is_file()}
    assert before == after
