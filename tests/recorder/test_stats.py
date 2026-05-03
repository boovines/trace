"""Tests for :mod:`recorder.stats`.

Drives :func:`compute_summary` against a hand-rolled trajectory tree on
disk so the aggregation logic (focus-time, event mix, hour-of-day,
top-windows, daily timeseries) is exercised without any of the recorder's
runtime dependencies (PyObjC, CGEventTap, NSWorkspace).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI

from recorder.api import (
    RecorderState,
    get_recorder_state,
    stats_router,
)
from recorder.index_db import IndexDB
from recorder.stats import compute_summary

# ----------------------------------------------------------------- fixtures


def _iso(dt: datetime) -> str:
    return dt.isoformat(timespec="milliseconds")


def _write_trajectory(
    root: Path,
    *,
    tid: str,
    label: str,
    started: datetime,
    stopped: datetime,
    app_focus_history: list[dict[str, Any]],
    events: list[dict[str, Any]],
) -> None:
    traj_dir = root / tid
    traj_dir.mkdir(parents=True)
    metadata = {
        "id": tid,
        "label": label,
        "started_at": _iso(started),
        "stopped_at": _iso(stopped),
        "display_info": {"width": 1920, "height": 1080, "scale_factor": 2.0},
        "app_focus_history": app_focus_history,
    }
    (traj_dir / "metadata.json").write_text(json.dumps(metadata))
    with (traj_dir / "events.jsonl").open("w") as fh:
        for ev in events:
            fh.write(json.dumps(ev))
            fh.write("\n")


@pytest.fixture
def populated_root(tmp_path: Path) -> tuple[Path, IndexDB, datetime]:
    """Seed two trajectories on different days under ``tmp_path``."""
    root = tmp_path / "trajectories"
    root.mkdir()
    now = datetime(2026, 5, 1, 18, 0, 0, tzinfo=UTC)

    # ---- Trajectory A: yesterday, 10 minutes in Chrome ----
    a_start = now - timedelta(days=1)
    a_stop = a_start + timedelta(minutes=10)
    _write_trajectory(
        root,
        tid="aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
        label="email triage",
        started=a_start,
        stopped=a_stop,
        app_focus_history=[
            {
                "bundle_id": "com.google.Chrome",
                "name": "Google Chrome",
                "entered_at": _iso(a_start),
                "exited_at": _iso(a_start + timedelta(minutes=6)),
            },
            {
                "bundle_id": "com.apple.mail",
                "name": "Mail",
                "entered_at": _iso(a_start + timedelta(minutes=6)),
                "exited_at": _iso(a_stop),
            },
        ],
        events=[
            _ev(1, a_start, "click",
                {"bundle_id": "com.google.Chrome", "name": "Google Chrome", "pid": 100},
                payload={"button": "left"}),
            _ev(2, a_start + timedelta(seconds=5), "window_focus",
                {"bundle_id": "com.google.Chrome", "name": "Google Chrome", "pid": 100},
                payload={"window_title": "Inbox - Gmail"}),
            _ev(3, a_start + timedelta(seconds=10), "keypress",
                {"bundle_id": "com.google.Chrome", "name": "Google Chrome", "pid": 100},
                payload={"keys": ["a"]}),
            _ev(4, a_start + timedelta(minutes=6), "app_switch",
                {"bundle_id": "com.apple.mail", "name": "Mail", "pid": 200},
                payload={"to_bundle_id": "com.apple.mail"}),
            _ev(5, a_start + timedelta(minutes=7), "text_input",
                {"bundle_id": "com.apple.mail", "name": "Mail", "pid": 200},
                payload={"text": "Hello world", "field_label": None}),
        ],
    )

    # ---- Trajectory B: today, 4 minutes in Chrome again ----
    b_start = now - timedelta(minutes=10)
    b_stop = now - timedelta(minutes=6)
    _write_trajectory(
        root,
        tid="bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
        label="research",
        started=b_start,
        stopped=b_stop,
        app_focus_history=[
            {
                "bundle_id": "com.google.Chrome",
                "name": "Google Chrome",
                "entered_at": _iso(b_start),
                "exited_at": _iso(b_stop),
            },
        ],
        events=[
            _ev(1, b_start, "window_focus",
                {"bundle_id": "com.google.Chrome", "name": "Google Chrome", "pid": 101},
                payload={"window_title": "Inbox - Gmail"}),
            _ev(2, b_start + timedelta(seconds=30), "scroll",
                {"bundle_id": "com.google.Chrome", "name": "Google Chrome", "pid": 101},
                payload={"direction": "down", "delta": 3.0}),
            _ev(3, b_start + timedelta(minutes=2), "keyframe",
                {"bundle_id": "com.google.Chrome", "name": "Google Chrome", "pid": 101},
                payload={"reason": "periodic"}),
        ],
    )

    db = IndexDB(tmp_path / "index.db")
    db.reconcile(root)
    return root, db, now


def _ev(
    seq: int,
    when: datetime,
    type_: str,
    app: dict[str, Any],
    *,
    payload: dict[str, Any],
) -> dict[str, Any]:
    return {
        "seq": seq,
        "timestamp_ms": int(when.timestamp() * 1000),
        "type": type_,
        "app": app,
        "target": None,
        "payload": payload,
    }


# ------------------------------------------------------------- unit tests


def test_summary_aggregates_focus_time_by_app(
    populated_root: tuple[Path, IndexDB, datetime],
) -> None:
    root, db, now = populated_root
    out = compute_summary(
        index_db=db, trajectories_root=root, window_days=7, now=now
    )
    apps_by_bundle = {a["bundle_id"]: a for a in out["top_apps"]}

    # Chrome appears in both trajectories: 6 minutes (A) + 4 minutes (B) = 600s
    assert apps_by_bundle["com.google.Chrome"]["seconds"] == pytest.approx(600.0)
    assert apps_by_bundle["com.google.Chrome"]["sessions"] == 2
    # Mail: 4 minutes
    assert apps_by_bundle["com.apple.mail"]["seconds"] == pytest.approx(240.0)
    # Chrome should rank above Mail
    assert out["top_apps"][0]["bundle_id"] == "com.google.Chrome"


def test_summary_event_counts_exclude_keyframe_from_hour_buckets(
    populated_root: tuple[Path, IndexDB, datetime],
) -> None:
    root, db, now = populated_root
    out = compute_summary(
        index_db=db, trajectories_root=root, window_days=7, now=now
    )
    # 6 events across both trajectories — keyframe IS counted in the totals…
    assert out["event_counts"]["click"] == 1
    assert out["event_counts"]["keyframe"] == 1
    # …but NOT in the hour-of-day distribution (only interactive events count).
    assert sum(out["hour_of_day"]) == 7  # 5 + 3 - 1 keyframe


def test_summary_top_windows_collapse_duplicates(
    populated_root: tuple[Path, IndexDB, datetime],
) -> None:
    root, db, now = populated_root
    out = compute_summary(
        index_db=db, trajectories_root=root, window_days=7, now=now
    )
    # "Inbox - Gmail" appears in both trajectories: should be collapsed to one
    # row with count=2.
    titles = {(w["app_name"], w["window_title"]): w["count"] for w in out["top_windows"]}
    assert titles[("Google Chrome", "Inbox - Gmail")] == 2


def test_summary_text_input_chars_total(
    populated_root: tuple[Path, IndexDB, datetime],
) -> None:
    root, db, now = populated_root
    out = compute_summary(
        index_db=db, trajectories_root=root, window_days=7, now=now
    )
    assert out["text_input_chars"] == len("Hello world")


def test_summary_daily_buckets_have_correct_length_and_dates(
    populated_root: tuple[Path, IndexDB, datetime],
) -> None:
    root, db, now = populated_root
    out = compute_summary(
        index_db=db, trajectories_root=root, window_days=7, now=now
    )
    assert out["window_days"] == 7
    assert len(out["daily"]) == 7
    # Last bucket = today (UTC)
    assert out["daily"][-1]["date"] == now.date().isoformat()
    # Bucket for "yesterday" must show ~600s recorded (10 min trajectory A)
    yday = (now.date() - timedelta(days=1)).isoformat()
    yday_bucket = next(b for b in out["daily"] if b["date"] == yday)
    assert yday_bucket["recorded_seconds"] == pytest.approx(600.0)


def test_summary_filters_trajectories_outside_window(
    populated_root: tuple[Path, IndexDB, datetime],
) -> None:
    root, db, now = populated_root
    # Add an old trajectory 10 days ago — should be excluded from a 7-day window.
    old_start = now - timedelta(days=10)
    _write_trajectory(
        root,
        tid="cccccccc-cccc-cccc-cccc-cccccccccccc",
        label="old",
        started=old_start,
        stopped=old_start + timedelta(minutes=30),
        app_focus_history=[{
            "bundle_id": "com.example.OldApp",
            "name": "OldApp",
            "entered_at": _iso(old_start),
            "exited_at": _iso(old_start + timedelta(minutes=30)),
        }],
        events=[],
    )
    db.reconcile(root)
    out = compute_summary(
        index_db=db, trajectories_root=root, window_days=7, now=now
    )
    bundles = {a["bundle_id"] for a in out["top_apps"]}
    assert "com.example.OldApp" not in bundles
    assert out["trajectory_count"] == 2


def test_summary_handles_empty_index(tmp_path: Path) -> None:
    root = tmp_path / "trajectories"
    root.mkdir()
    db = IndexDB(tmp_path / "index.db")
    out = compute_summary(index_db=db, trajectories_root=root, window_days=7)
    assert out["trajectory_count"] == 0
    assert out["recorded_seconds"] == 0.0
    assert out["top_apps"] == []
    assert out["top_windows"] == []
    assert sum(out["hour_of_day"]) == 0
    assert len(out["daily"]) == 7


def test_summary_window_days_validated(
    populated_root: tuple[Path, IndexDB, datetime],
) -> None:
    root, db, _now = populated_root
    with pytest.raises(ValueError):
        compute_summary(index_db=db, trajectories_root=root, window_days=0)


# ------------------------------------------------------------- HTTP surface


@pytest.mark.anyio
async def test_stats_summary_endpoint(
    populated_root: tuple[Path, IndexDB, datetime],
) -> None:
    root, db, _now = populated_root
    state = RecorderState(root, index_db=db, reconcile_on_init=False)
    app = FastAPI()
    app.include_router(stats_router)
    app.dependency_overrides[get_recorder_state] = lambda: state

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get("/stats/summary?days=7")
        assert r.status_code == 200
        body = r.json()
        assert body["window_days"] == 7
        assert body["trajectory_count"] == 2
        assert "top_apps" in body
        assert any(a["bundle_id"] == "com.google.Chrome" for a in body["top_apps"])

        bad = await client.get("/stats/summary?days=0")
        assert bad.status_code == 400


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"
