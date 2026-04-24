"""End-to-end HTTP-driven recorder test (R-014).

The point of this test is to exercise the full recorder slice — FastAPI
router → :class:`recorder.api.RecorderState` → real
:class:`recorder.session.RecordingSession` → real
:class:`recorder.writer.TrajectoryWriter` → real
:class:`recorder.index_db.IndexDB` → on-disk trajectory — with every
macOS-dependent dependency replaced by a synthetic fake so it runs
hermetically in Ralph's sandbox and CI.

Unlike :mod:`tests.recorder.test_api`, which substitutes ``FakeSession``
for the whole session class, this test keeps the real ``RecordingSession``
and only swaps out the four things that need PyObjC
(``EventTap``, ``FocusTracker``, ``TextAggregator``, the screenshot
capturer, the AX resolver, and the permissions check).  That means the
click envelope, the text_input emission path, the periodic-keyframe
loop, and the metadata finalisation are all real code under test.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI

from recorder.api import (
    RecorderState,
    get_recorder_state,
    trajectories_router,
)
from recorder.api import (
    router as recorder_router,
)
from recorder.focus_tracker import (
    AppInfo,
    AppSwitchPayload,
    WindowFocusPayload,
)
from recorder.index_db import IndexDB
from recorder.keyframe_policy import KeyframePolicy
from recorder.schema import validate_event, validate_metadata
from recorder.session import RecordingSession

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
PNG_BYTES = PNG_MAGIC + b"\x00" * 32


# --------------------------------------------------------- fakes (no PyObjC)


class FakeEventTap:
    """Stand-in for :class:`recorder.event_tap.EventTap` — same shape as
    the one used in :mod:`tests.recorder.test_session`.
    """

    def __init__(self, callback: Any) -> None:
        self.callback = callback
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def fire(self, event: dict[str, Any]) -> None:
        self.callback(event)


class FakeFocusTracker:
    def __init__(self) -> None:
        self.app_switch_cbs: list[Any] = []
        self.window_focus_cbs: list[Any] = []
        self.current_app: AppInfo | None = {
            "bundle_id": "com.test.app",
            "name": "TestApp",
            "pid": 1000,
        }
        self.history: list[dict[str, Any]] = [
            {
                "bundle_id": "com.test.app",
                "name": "TestApp",
                "entered_at": "2026-04-23T12:00:00.000+00:00",
                "exited_at": None,
            }
        ]
        self.started = False
        self.stopped = False

    def on_app_switch(self, callback: Any) -> None:
        self.app_switch_cbs.append(callback)

    def on_window_focus_change(self, callback: Any) -> None:
        self.window_focus_cbs.append(callback)

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True
        if self.history and self.history[-1].get("exited_at") is None:
            self.history[-1]["exited_at"] = "2026-04-23T12:00:05.000+00:00"

    def get_current_app(self) -> AppInfo | None:
        return self.current_app

    def get_app_focus_history(self) -> list[dict[str, Any]]:
        return list(self.history)

    def fire_app_switch(self, payload: AppSwitchPayload) -> None:
        new_app: AppInfo = {
            "bundle_id": payload["to_bundle_id"],
            "name": payload.get("to_name") or "",
            "pid": 1000 + len(self.history),
        }
        entered_at = f"2026-04-23T12:00:{len(self.history):02d}.000+00:00"
        if self.history and self.history[-1].get("exited_at") is None:
            self.history[-1]["exited_at"] = entered_at
        self.history.append(
            {
                "bundle_id": new_app["bundle_id"],
                "name": new_app["name"],
                "entered_at": entered_at,
                "exited_at": None,
            }
        )
        self.current_app = new_app
        for cb in self.app_switch_cbs:
            cb(payload)

    def fire_window_focus(self, payload: WindowFocusPayload) -> None:
        for cb in self.window_focus_cbs:
            cb(payload)


class FakeTextAggregator:
    def __init__(self, emit: Any) -> None:
        self.emit = emit
        self.set_focus_calls: list[tuple[Any, ...]] = []
        self.key_events: list[dict[str, Any]] = []
        self.stopped = False

    def set_focus(self, bundle_id: Any, key: Any, label: Any) -> None:
        self.set_focus_calls.append((bundle_id, key, label))

    def handle_key_event(self, event: dict[str, Any]) -> None:
        self.key_events.append(event)

    def stop(self) -> None:
        self.stopped = True

    def fire_text_input(
        self, text: str, field_label: str | None, bundle_id: str
    ) -> None:
        self.emit(
            {
                "text": text,
                "field_label": field_label,
                "app_bundle_id": bundle_id,
            }
        )


# ---------------------------------------------- session factory for the app


class _Bag:
    """Holds references to the active fakes so the test can drive events."""

    event_tap: FakeEventTap | None = None
    focus_tracker: FakeFocusTracker | None = None
    text_aggregator: FakeTextAggregator | None = None


def _make_session_factory(
    bag: _Bag,
) -> Any:
    """Build a ``SessionFactory`` that wires fakes into each new session."""

    def factory(root: Path, index_db: IndexDB | None) -> RecordingSession:
        def event_tap_factory(cb: Any) -> FakeEventTap:
            tap = FakeEventTap(cb)
            bag.event_tap = tap
            return tap

        def focus_tracker_factory() -> FakeFocusTracker:
            ft = FakeFocusTracker()
            bag.focus_tracker = ft
            return ft

        def text_aggregator_factory(emit: Any) -> FakeTextAggregator:
            ta = FakeTextAggregator(emit)
            bag.text_aggregator = ta
            return ta

        # A very-long periodic interval keeps the periodic keyframe loop
        # from firing during the test window — we want deterministic event
        # counts.  Post-click delay = 0 so the post-click keyframe fires
        # synchronously, which makes the assertion easier.
        policy = KeyframePolicy(
            periodic_interval_seconds=3600.0,
            post_click_delay_seconds=0.0,
        )
        return RecordingSession(
            root,
            index_db=index_db,
            event_tap_factory=event_tap_factory,
            focus_tracker_factory=focus_tracker_factory,
            text_aggregator_factory=text_aggregator_factory,
            permissions_check=lambda: None,
            ax_resolver=lambda _x, _y: {
                "role": "AXButton",
                "label": "Send",
                "description": None,
                "frame": {"x": 10.0, "y": 20.0, "w": 30.0, "h": 40.0},
                "ax_identifier": "send",
            },
            screenshot_capturer=lambda: PNG_BYTES,
            display_info_provider=lambda: {
                "width": 1920,
                "height": 1080,
                "scale_factor": 2.0,
            },
            keyframe_policy=policy,
        )

    return factory


# ----------------------------------------------------------- fixtures


@pytest.fixture
def bag() -> _Bag:
    return _Bag()


@pytest.fixture
def app(tmp_path: Path, bag: _Bag) -> FastAPI:
    state = RecorderState(
        tmp_path / "trajectories",
        session_factory=_make_session_factory(bag),
        index_db_path=tmp_path / "index.db",
        reconcile_on_init=False,
    )
    app = FastAPI()
    app.include_router(recorder_router, prefix="/recorder")
    app.include_router(trajectories_router)
    app.dependency_overrides[get_recorder_state] = lambda: state
    return app


@pytest.fixture
async def client(app: FastAPI) -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac


# ----------------------------------------------------------- helpers


def _fire_click(bag: _Bag, x: float, y: float) -> None:
    assert bag.event_tap is not None
    bag.event_tap.fire(
        {
            "cg_event_type": "left_mouse_down",
            "timestamp_ms": int(x * 1000 + y),
            "location_x": x,
            "location_y": y,
            "modifiers": [],
        }
    )


# ------------------------------------------------------------- the test


async def test_e2e_record_read_delete(client: httpx.AsyncClient, bag: _Bag) -> None:
    """Drive the full recorder slice over HTTP with mocked macOS pieces.

    This exercises the acceptance criteria in R-014 in one flow:
      1. ``POST /recorder/start``  →  ``{trajectory_id}``
      2. Synthesise 5 clicks, 2 text inputs, 1 app switch
      3. ``POST /recorder/stop``  →  ``{trajectory_id, event_count >= 8}``
      4. ``GET /trajectories/{id}``  → full metadata + events validate
      5. ``DELETE /trajectories/{id}``  →  200, then ``GET`` returns 404
    """

    # 1. start ----------------------------------------------------------------
    start_resp = await client.post("/recorder/start", json={"label": "e2e"})
    assert start_resp.status_code == 200
    tid = start_resp.json()["trajectory_id"]
    uuid.UUID(tid)

    status_resp = await client.get("/recorder/status")
    assert status_resp.status_code == 200
    status_body = status_resp.json()
    assert status_body["recording"] is True
    assert status_body["current_trajectory_id"] == tid

    # 2. drive synthetic capture events --------------------------------------
    assert bag.event_tap is not None
    assert bag.focus_tracker is not None
    assert bag.text_aggregator is not None

    for idx in range(5):
        _fire_click(bag, x=100.0 + idx, y=200.0 + idx)

    bag.text_aggregator.fire_text_input(
        text="hello world", field_label="Subject", bundle_id="com.test.app"
    )
    bag.text_aggregator.fire_text_input(
        text="cheers", field_label="Body", bundle_id="com.test.app"
    )

    bag.focus_tracker.fire_app_switch(
        {
            "from_bundle_id": "com.test.app",
            "to_bundle_id": "com.test.other",
            "from_name": "TestApp",
            "to_name": "OtherApp",
        }
    )

    # 3. stop -----------------------------------------------------------------
    stop_resp = await client.post("/recorder/stop")
    assert stop_resp.status_code == 200
    stop_body = stop_resp.json()
    assert stop_body["trajectory_id"] == tid
    # Event budget (minimum):
    #   5 clicks * (pre_click + click + post_click) = 15
    #   2 text_input = 2
    #   1 app_switch + its post-switch keyframe = 2
    # => well over the AC's >= 8 threshold.
    assert stop_body["event_count"] >= 8
    assert stop_body["duration_ms"] >= 0

    # Capture components should have been stopped.
    assert bag.event_tap.stopped is True
    assert bag.focus_tracker.stopped is True
    assert bag.text_aggregator.stopped is True

    status_after = await client.get("/recorder/status")
    assert status_after.json()["recording"] is False

    # 4. fetch and validate ---------------------------------------------------
    get_resp = await client.get(f"/trajectories/{tid}")
    assert get_resp.status_code == 200
    detail = get_resp.json()
    assert detail["id"] == tid

    metadata = detail["metadata"]
    assert metadata["id"] == tid
    assert metadata["label"] == "e2e"
    assert metadata["stopped_at"] is not None
    assert metadata["display_info"] == {
        "width": 1920,
        "height": 1080,
        "scale_factor": 2.0,
    }
    # Schema validation on the full metadata.
    validate_metadata(metadata)

    # Every event on disk must validate against the locked schema.
    events: list[dict[str, Any]] = detail["events"]
    assert len(events) == stop_body["event_count"]
    for event in events:
        validate_event(event)

    # Structural sanity on the event mix.
    types = [e["type"] for e in events]
    assert types.count("click") == 5
    assert types.count("text_input") == 2
    assert types.count("app_switch") == 1
    assert "keyframe" in types

    # seq values are contiguous and monotonic.
    seqs = [e["seq"] for e in events]
    assert seqs == list(range(1, len(events) + 1))

    # Screenshots were captured — at least one keyframe per click envelope.
    assert len(detail["screenshots"]) >= 5
    for url in detail["screenshots"]:
        assert url.startswith("http://testserver/trajectories/")
        assert url.endswith(".png")

    # Index row matches.
    list_resp = await client.get("/trajectories")
    assert list_resp.status_code == 200
    rows = list_resp.json()
    assert len(rows) == 1
    row = rows[0]
    assert row["id"] == tid
    assert row["label"] == "e2e"
    assert row["event_count"] == stop_body["event_count"]

    # 5. delete ---------------------------------------------------------------
    delete_resp = await client.delete(f"/trajectories/{tid}")
    assert delete_resp.status_code == 200
    assert delete_resp.json() == {"deleted": tid}

    follow_up = await client.get(f"/trajectories/{tid}")
    assert follow_up.status_code == 404

    list_after = await client.get("/trajectories")
    assert list_after.json() == []
