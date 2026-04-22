"""Tests for ``synthesizer.preprocess``.

Covers the PRD S-005 acceptance bar:

* scroll collapsing folds 50 same-app scrolls inside 2s into one ``scroll_run``
* ≥5s gaps between retained events inject an ``idle`` entry
* keyframe selection caps at 20 even when every event carries a screenshot
* clicks and text_inputs survive unmodified through preprocessing
* the rough-token estimator agrees with a hand count to within 20%

The shared ``_make_trajectory`` helper writes a minimal on-disk trajectory so
tests exercise the real :class:`~synthesizer.trajectory_reader.TrajectoryReader`
loading path rather than mocking it — keeps the preprocessor honest about the
actual event shapes shipped by Module 1.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError
from synthesizer.preprocess import (
    MAX_KEYFRAMES,
    TOKENS_PER_IMAGE,
    DigestEntry,
    PreprocessedTrajectory,
    preprocess_trajectory,
)
from synthesizer.trajectory_reader import TrajectoryReader

# --- helpers ---------------------------------------------------------------


def _iso(seconds_offset: float, base: datetime | None = None) -> str:
    """Return an ISO-8601 UTC timestamp offset from ``base`` (default: epoch+0)."""
    base = base or datetime(2026, 4, 22, 14, 0, 0, tzinfo=UTC)
    return (base + timedelta(seconds=seconds_offset)).isoformat().replace("+00:00", "Z")


def _make_trajectory(
    tmp_path: Path,
    events: list[dict[str, Any]],
    *,
    started_at: str = "2026-04-22T14:00:00Z",
    stopped_at: str | None = None,
    bundle_id: str = "com.google.Chrome",
) -> TrajectoryReader:
    """Write a minimal trajectory directory and return a reader on it."""
    if stopped_at is None:
        # Pick a stopped_at later than the last event timestamp.
        if events:
            last_t = events[-1]["t"]
            parsed = last_t.replace("Z", "+00:00")
            dt = datetime.fromisoformat(parsed) + timedelta(seconds=1)
            stopped_at = dt.isoformat().replace("+00:00", "Z")
        else:
            stopped_at = started_at

    traj_dir = tmp_path / f"traj-{uuid.uuid4().hex[:8]}"
    traj_dir.mkdir()
    (traj_dir / "screenshots").mkdir()

    metadata = {
        "id": str(uuid.uuid4()),
        "started_at": started_at,
        "stopped_at": stopped_at,
        "label": "test",
        "display_info": {"width": 2560, "height": 1440, "scale": 2.0},
        "app_focus_history": [
            {"at": started_at, "bundle_id": bundle_id, "title": "Test"}
        ],
    }
    (traj_dir / "metadata.json").write_text(json.dumps(metadata))

    # Write a single tiny PNG used by any event that references a screenshot.
    # 67-byte 1x1 grayscale PNG matches the fixture-trajectory style.
    png_bytes = bytes.fromhex(
        "89504e470d0a1a0a0000000d49484452000000010000000108000000"
        "003b7e9b550000000a49444154789c6300010000000500010d0a2db4"
        "0000000049454e44ae426082"
    )

    refs_written: set[str] = set()
    for event in events:
        ref = event.get("screenshot_ref")
        if ref and ref not in refs_written:
            (traj_dir / ref).parent.mkdir(parents=True, exist_ok=True)
            (traj_dir / ref).write_bytes(png_bytes)
            refs_written.add(ref)

    with (traj_dir / "events.jsonl").open("w") as f:
        for event in events:
            f.write(json.dumps(event) + "\n")

    return TrajectoryReader(traj_dir)


# --- scroll collapsing -----------------------------------------------------


def test_collapses_dense_scroll_run(tmp_path: Path) -> None:
    # 50 scrolls over 2 seconds in the same app → one scroll_run
    events: list[dict[str, Any]] = []
    for i in range(50):
        events.append(
            {
                "seq": i + 1,
                "t": _iso(i * 0.04),  # 25 Hz, 2s total
                "kind": "scroll",
                "bundle_id": "com.google.Chrome",
                "y": -10.0,
            }
        )
    reader = _make_trajectory(tmp_path, events)
    result = preprocess_trajectory(reader)

    scroll_runs = [e for e in result.digest if e.kind == "scroll_run"]
    assert len(scroll_runs) == 1
    run = scroll_runs[0]
    assert run.payload["total_delta"] == pytest.approx(-500.0)
    assert run.payload["duration_ms"] >= 1_900
    assert run.payload["duration_ms"] <= 2_100
    assert run.app_bundle_id == "com.google.Chrome"
    assert result.original_event_count == 50
    assert result.digest_entry_count == 1


def test_scroll_run_broken_by_app_change(tmp_path: Path) -> None:
    events = [
        {"seq": 1, "t": _iso(0.0), "kind": "scroll", "bundle_id": "com.a", "y": -5.0},
        {"seq": 2, "t": _iso(0.5), "kind": "scroll", "bundle_id": "com.a", "y": -5.0},
        {"seq": 3, "t": _iso(1.0), "kind": "scroll", "bundle_id": "com.b", "y": -5.0},
    ]
    result = preprocess_trajectory(_make_trajectory(tmp_path, events))
    scroll_runs = [e for e in result.digest if e.kind == "scroll_run"]
    assert len(scroll_runs) == 2


def test_scroll_run_broken_by_long_gap(tmp_path: Path) -> None:
    events = [
        {"seq": 1, "t": _iso(0.0), "kind": "scroll", "bundle_id": "com.a", "y": -5.0},
        {"seq": 2, "t": _iso(0.5), "kind": "scroll", "bundle_id": "com.a", "y": -5.0},
        # 4s gap → still within 3s? no. 4s > 3s gap → new run
        {"seq": 3, "t": _iso(4.6), "kind": "scroll", "bundle_id": "com.a", "y": -5.0},
    ]
    result = preprocess_trajectory(_make_trajectory(tmp_path, events))
    scroll_runs = [e for e in result.digest if e.kind == "scroll_run"]
    assert len(scroll_runs) == 2


# --- idle detection --------------------------------------------------------


def test_idle_entry_injected_for_long_gap(tmp_path: Path) -> None:
    events = [
        {
            "seq": 1,
            "t": _iso(0.0),
            "kind": "click",
            "x": 10.0,
            "y": 20.0,
            "button": "left",
            "bundle_id": "com.a",
            "target": {"label": "Hello", "role": "button"},
        },
        {
            "seq": 2,
            "t": _iso(10.0),  # 10s idle
            "kind": "click",
            "x": 11.0,
            "y": 22.0,
            "button": "left",
            "bundle_id": "com.a",
            "target": {"label": "World", "role": "button"},
        },
    ]
    result = preprocess_trajectory(_make_trajectory(tmp_path, events))
    idles = [e for e in result.digest if e.kind == "idle"]
    assert len(idles) == 1
    assert idles[0].payload["duration_ms"] == 10_000
    # Ensure the idle entry is sandwiched between the two clicks
    kinds = [e.kind for e in result.digest]
    assert kinds == ["click", "idle", "click"]


def test_idle_not_injected_for_short_gap(tmp_path: Path) -> None:
    events = [
        {
            "seq": 1,
            "t": _iso(0.0),
            "kind": "click",
            "x": 10.0,
            "y": 20.0,
            "button": "left",
            "bundle_id": "com.a",
            "target": {"label": "A", "role": "button"},
        },
        {
            "seq": 2,
            "t": _iso(4.5),  # under 5s
            "kind": "click",
            "x": 11.0,
            "y": 22.0,
            "button": "left",
            "bundle_id": "com.a",
            "target": {"label": "B", "role": "button"},
        },
    ]
    result = preprocess_trajectory(_make_trajectory(tmp_path, events))
    assert not any(e.kind == "idle" for e in result.digest)


# --- clicks / text_inputs never dropped ------------------------------------


def test_all_clicks_and_text_inputs_preserved(tmp_path: Path) -> None:
    events: list[dict[str, Any]] = []
    seq = 0
    # 30 clicks, one every 100ms so they stay in a single app context
    for i in range(30):
        seq += 1
        events.append(
            {
                "seq": seq,
                "t": _iso(seq * 0.1),
                "kind": "click",
                "x": float(i),
                "y": float(i),
                "button": "left",
                "bundle_id": "com.a",
                "target": {"label": f"Btn{i}", "role": "button"},
            }
        )
    # 10 text_inputs interleaved afterward
    for i in range(10):
        seq += 1
        events.append(
            {
                "seq": seq,
                "t": _iso(seq * 0.1),
                "kind": "text_input",
                "text": f"hello {i}",
                "bundle_id": "com.a",
            }
        )

    result = preprocess_trajectory(_make_trajectory(tmp_path, events))
    clicks = [e for e in result.digest if e.kind == "click"]
    inputs = [e for e in result.digest if e.kind == "text_input"]
    assert len(clicks) == 30
    assert len(inputs) == 10
    # Every click surfaces its target label in summary text
    for i, click in enumerate(clicks):
        assert f"Btn{i}" in click.summary_text
    # text_input summaries carry the typed text (truncated at 60 chars)
    for i, entry in enumerate(inputs):
        assert f"hello {i}" in entry.summary_text


def test_mouse_noise_events_are_dropped(tmp_path: Path) -> None:
    events: list[dict[str, Any]] = [
        {"seq": 1, "t": _iso(0.0), "kind": "mouse_move", "x": 1.0, "y": 1.0, "bundle_id": "com.a"},
        {"seq": 2, "t": _iso(0.1), "kind": "mouse_down", "x": 1.0, "y": 1.0, "bundle_id": "com.a"},
        {"seq": 3, "t": _iso(0.2), "kind": "mouse_up", "x": 1.0, "y": 1.0, "bundle_id": "com.a"},
        {
            "seq": 4,
            "t": _iso(0.3),
            "kind": "click",
            "x": 1.0,
            "y": 1.0,
            "button": "left",
            "bundle_id": "com.a",
            "target": {"label": "Go", "role": "button"},
        },
    ]
    result = preprocess_trajectory(_make_trajectory(tmp_path, events))
    kinds = [e.kind for e in result.digest]
    assert kinds == ["click"]


# --- keyframe selection ----------------------------------------------------


def test_keyframe_cap_at_twenty_uniform_case(tmp_path: Path) -> None:
    # 100 screenshot-bearing events of kind 'screenshot' (no app_switch,
    # no clicks) → priority 1 and 2 yield zero, priority 3 fills exactly 20.
    events: list[dict[str, Any]] = []
    for i in range(100):
        events.append(
            {
                "seq": i + 1,
                "t": _iso(i * 0.5),  # 50s span
                "kind": "screenshot",
                "bundle_id": "com.a",
                "screenshot_ref": f"screenshots/{i + 1:04d}.png",
            }
        )
    result = preprocess_trajectory(_make_trajectory(tmp_path, events))
    with_shots = [e for e in result.digest if e.screenshot_ref is not None]
    assert len(with_shots) == MAX_KEYFRAMES
    assert result.screenshots_included == MAX_KEYFRAMES


def test_keyframe_priority_app_switches_win(tmp_path: Path) -> None:
    # 30 events total: 5 app_switches early, then 25 generic screenshots.
    # We expect all 5 app_switches to be among the selected keyframes.
    events: list[dict[str, Any]] = []
    seq = 0
    for i in range(5):
        seq += 1
        events.append(
            {
                "seq": seq,
                "t": _iso(seq * 0.1),
                "kind": "app_switch",
                "bundle_id": f"com.app{i}",
                "screenshot_ref": f"screenshots/{seq:04d}.png",
            }
        )
    for _ in range(25):
        seq += 1
        events.append(
            {
                "seq": seq,
                "t": _iso(seq * 0.1),
                "kind": "screenshot",
                "bundle_id": "com.a",
                "screenshot_ref": f"screenshots/{seq:04d}.png",
            }
        )

    result = preprocess_trajectory(_make_trajectory(tmp_path, events))
    picked_kinds = [e.kind for e in result.digest if e.screenshot_ref is not None]
    # At least the 5 app_switches made the cut (priority 1)
    app_switch_count = sum(1 for k in picked_kinds if k == "app_switch")
    assert app_switch_count == 5
    assert len(picked_kinds) == MAX_KEYFRAMES


def test_keyframe_click_cluster_preceders_win(tmp_path: Path) -> None:
    # A screenshot immediately preceding each click cluster (3 clusters total)
    # should be among the selected keyframes even when there are far more
    # competing screenshots.
    events: list[dict[str, Any]] = []
    seq = 0

    # Cluster pattern x3: screenshot -> click -> click -> long wait, repeated.
    for cluster in range(3):
        # filler screenshots before the cluster so the preceding-keyframe is
        # distinct from every other screenshot.
        base = cluster * 20.0
        seq += 1
        events.append(
            {
                "seq": seq,
                "t": _iso(base),
                "kind": "screenshot",
                "bundle_id": "com.a",
                "screenshot_ref": f"screenshots/{seq:04d}.png",
            }
        )
        seq += 1
        events.append(
            {
                "seq": seq,
                "t": _iso(base + 0.5),
                "kind": "click",
                "x": 1.0,
                "y": 1.0,
                "button": "left",
                "bundle_id": "com.a",
                "target": {"label": f"Go{cluster}", "role": "button"},
            }
        )
        seq += 1
        events.append(
            {
                "seq": seq,
                "t": _iso(base + 0.8),
                "kind": "click",
                "x": 1.0,
                "y": 1.0,
                "button": "left",
                "bundle_id": "com.a",
                "target": {"label": f"Again{cluster}", "role": "button"},
            }
        )

    # Add 50 extra screenshots after the clusters to force the cap
    tail_base = 3 * 20.0 + 2.0
    for i in range(50):
        seq += 1
        events.append(
            {
                "seq": seq,
                "t": _iso(tail_base + i * 0.1),
                "kind": "screenshot",
                "bundle_id": "com.a",
                "screenshot_ref": f"screenshots/{seq:04d}.png",
            }
        )

    result = preprocess_trajectory(_make_trajectory(tmp_path, events))
    # The 3 cluster-preceding screenshots (seq 1, 4, 7 in our layout) must
    # appear as keyframes.
    selected_seqs = {e.seq for e in result.digest if e.screenshot_ref is not None}
    assert 1 in selected_seqs
    assert 4 in selected_seqs
    assert 7 in selected_seqs


def test_all_screenshots_kept_when_under_cap(tmp_path: Path) -> None:
    events: list[dict[str, Any]] = [
        {
            "seq": i + 1,
            "t": _iso(i * 0.5),
            "kind": "screenshot",
            "bundle_id": "com.a",
            "screenshot_ref": f"screenshots/{i + 1:04d}.png",
        }
        for i in range(5)
    ]
    result = preprocess_trajectory(_make_trajectory(tmp_path, events))
    assert result.screenshots_included == 5


# --- statistics + integration on fixtures ----------------------------------


def test_stats_are_coherent_on_reference_fixture() -> None:
    fixtures_root = (
        Path(__file__).resolve().parents[2] / "fixtures" / "trajectories" / "gmail_reply"
    )
    reader = TrajectoryReader(fixtures_root)
    result = preprocess_trajectory(reader)

    assert result.original_event_count == 4
    # Fixture gaps are all ≥5s (5s/5s/15s) so an idle entry is injected
    # between each real event. 4 real + 3 idle = 7 digest entries.
    assert result.digest_entry_count == 7
    kinds = [e.kind for e in result.digest]
    assert kinds == [
        "app_switch",
        "idle",
        "click",
        "idle",
        "text_input",
        "idle",
        "click",
    ]
    # The one screenshot on the app_switch is under the cap, so it is kept.
    assert result.screenshots_included == 1


def test_token_estimate_within_twenty_percent(tmp_path: Path) -> None:
    events = [
        {
            "seq": 1,
            "t": _iso(0.0),
            "kind": "click",
            "x": 1.0,
            "y": 1.0,
            "button": "left",
            "bundle_id": "com.google.Chrome",
            "target": {"label": "Reply", "role": "button"},
            "screenshot_ref": "screenshots/0001.png",
        },
        {
            "seq": 2,
            "t": _iso(0.5),
            "kind": "text_input",
            "text": "Hello world, this is a short reply.",
            "bundle_id": "com.google.Chrome",
        },
        {
            "seq": 3,
            "t": _iso(1.0),
            "kind": "click",
            "x": 1.0,
            "y": 1.0,
            "button": "left",
            "bundle_id": "com.google.Chrome",
            "target": {"label": "Send", "role": "button"},
        },
    ]
    result = preprocess_trajectory(_make_trajectory(tmp_path, events))

    # Hand computation: 4 chars/token on summary text with +16 framing/entry,
    # plus one screenshot at TOKENS_PER_IMAGE.
    text_chars = sum(len(e.summary_text) + 16 for e in result.digest)
    expected = int(text_chars / 4.0) + TOKENS_PER_IMAGE
    # Within 20%
    low = int(expected * 0.8)
    high = int(expected * 1.2)
    assert low <= result.estimated_input_tokens <= high


def test_empty_trajectory_produces_empty_digest(tmp_path: Path) -> None:
    result = preprocess_trajectory(_make_trajectory(tmp_path, []))
    assert isinstance(result, PreprocessedTrajectory)
    assert result.digest == []
    assert result.original_event_count == 0
    assert result.estimated_input_tokens == 0


def test_digest_entries_are_frozen(tmp_path: Path) -> None:
    events = [
        {
            "seq": 1,
            "t": _iso(0.0),
            "kind": "click",
            "x": 1.0,
            "y": 1.0,
            "button": "left",
            "bundle_id": "com.a",
            "target": {"label": "X", "role": "button"},
        }
    ]
    result = preprocess_trajectory(_make_trajectory(tmp_path, events))
    entry: DigestEntry = result.digest[0]
    with pytest.raises(ValidationError):
        entry.summary_text = "mutated"  # type: ignore[misc]
