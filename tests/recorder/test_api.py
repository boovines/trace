"""Tests for :mod:`recorder.api` (R-011).

Hermetic coverage of the HTTP surface:

* All endpoints are driven via :class:`httpx.AsyncClient` against the
  FastAPI app (ASGI transport — no real TCP socket).
* The :class:`recorder.session.RecordingSession` is replaced with a
  ``FakeSession`` that writes a real on-disk trajectory directory but does
  NOT touch any PyObjC framework, so the suite stays green in Ralph's
  sandbox.
* The :func:`recorder.api.get_recorder_state` dependency is overridden via
  FastAPI's ``app.dependency_overrides`` machinery with a fresh
  :class:`RecorderState` rooted at ``tmp_path`` per test.
"""

from __future__ import annotations

import json
import threading
import time
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
from recorder.index_db import IndexDB
from recorder.permissions import PermissionsError
from recorder.session import (
    PermissionsMissingError,
    SessionAlreadyActiveError,
    SessionSummary,
)
from recorder.writer import TrajectoryWriter

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


# -------------------------------------------------------------- fake session


class FakeSession:
    """Stand-in for :class:`recorder.session.RecordingSession`.

    Writes a real on-disk trajectory directory through
    :class:`TrajectoryWriter` (so list / get / delete endpoints have real
    files to operate on) but never touches PyObjC.
    """

    permissions_error: PermissionsError | None = None

    def __init__(self, root: Path, index_db: IndexDB | None = None) -> None:
        self.root = root
        self.index_db = index_db
        self.label: str = ""
        self._writer: TrajectoryWriter | None = None
        self._active: bool = False
        self._summary: SessionSummary | None = None
        self._started_monotonic: float = 0.0

    def start(self, label: str) -> str:
        if FakeSession.permissions_error is not None:
            raise PermissionsMissingError(FakeSession.permissions_error)
        if self._active:
            raise SessionAlreadyActiveError("already active")
        writer = TrajectoryWriter(self.root, label, index_db=self.index_db)
        writer.write_metadata(
            {
                "id": writer.id,
                "label": label,
                "started_at": "2026-04-23T12:00:00.000+00:00",
                "stopped_at": None,
                "display_info": {"width": 1920, "height": 1080, "scale_factor": 2.0},
                "app_focus_history": [],
            }
        )
        # Inject a couple of fake events + a screenshot so list/get/delete
        # endpoints have meaningful payloads.
        writer.append_event(
            {
                "timestamp_ms": 1_700_000_000_000,
                "type": "keyframe",
                "screenshot_ref": "screenshots/0001.png",
                "app": {"bundle_id": "com.test", "name": "Test", "pid": 1},
                "target": None,
                "payload": {"reason": "periodic"},
            }
        )
        writer.write_screenshot(1, PNG_MAGIC + b"\x00" * 16)
        writer.append_event(
            {
                "timestamp_ms": 1_700_000_000_500,
                "type": "click",
                "screenshot_ref": None,
                "app": {"bundle_id": "com.test", "name": "Test", "pid": 1},
                "target": None,
                "payload": {"button": "left"},
            }
        )
        self._writer = writer
        self._active = True
        self.label = label
        self._started_monotonic = time.monotonic()
        return writer.id

    def stop(self) -> SessionSummary:
        assert self._writer is not None
        # Close first (it writes a fresh stopped_at) then overwrite metadata
        # with deterministic timestamps so the duration_ms assertion is stable.
        self._writer.close()
        metadata_path = self._writer.dir / "metadata.json"
        metadata_path.write_text(
            json.dumps(
                {
                    "id": self._writer.id,
                    "label": self.label,
                    "started_at": "2026-04-23T12:00:00.000+00:00",
                    "stopped_at": "2026-04-23T12:00:01.500+00:00",
                    "display_info": {
                        "width": 1920,
                        "height": 1080,
                        "scale_factor": 2.0,
                    },
                    "app_focus_history": [],
                },
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
        duration_ms = int((time.monotonic() - self._started_monotonic) * 1000)
        # Overwrite the index row with the deterministic values from the
        # hand-written metadata so list_trajectories assertions are stable.
        if self.index_db is not None:
            self.index_db.upsert(
                trajectory_id=self._writer.id,
                label=self.label or None,
                started_at="2026-04-23T12:00:00.000+00:00",
                stopped_at="2026-04-23T12:00:01.500+00:00",
                event_count=2,
                duration_ms=1500,
            )
        summary: SessionSummary = {
            "trajectory_id": self._writer.id,
            "event_count": 2,
            "duration_ms": duration_ms,
        }
        self._active = False
        self._summary = summary
        return summary

    def is_active(self) -> bool:
        return self._active

    @property
    def trajectory_id(self) -> str | None:
        if self._writer is not None:
            return self._writer.id
        return self._summary["trajectory_id"] if self._summary else None


# -------------------------------------------------------- fixtures / app


@pytest.fixture(autouse=True)
def _reset_fake_session_class_state() -> Any:
    """Each test starts with a clean ``permissions_error`` class attribute."""
    FakeSession.permissions_error = None
    yield
    FakeSession.permissions_error = None


@pytest.fixture
def app(tmp_path: Path) -> FastAPI:
    state = RecorderState(tmp_path / "trajectories", session_factory=FakeSession)
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


def _state(app: FastAPI) -> RecorderState:
    """Resolve the overridden :class:`RecorderState` for assertions."""
    return app.dependency_overrides[get_recorder_state]()


# ------------------------------------------------------------ status route


async def test_status_idle(client: httpx.AsyncClient) -> None:
    resp = await client.get("/recorder/status")
    assert resp.status_code == 200
    assert resp.json() == {
        "recording": False,
        "current_trajectory_id": None,
        "started_at": None,
    }


# ------------------------------------------------------------ start route


async def test_start_returns_trajectory_id(client: httpx.AsyncClient) -> None:
    resp = await client.post("/recorder/start", json={"label": "demo"})
    assert resp.status_code == 200
    body = resp.json()
    assert "trajectory_id" in body
    uuid.UUID(body["trajectory_id"])  # parses


async def test_start_without_body_uses_empty_label(client: httpx.AsyncClient) -> None:
    resp = await client.post("/recorder/start", json={})
    assert resp.status_code == 200


async def test_status_after_start_shows_recording(client: httpx.AsyncClient) -> None:
    started = await client.post("/recorder/start", json={"label": "x"})
    tid = started.json()["trajectory_id"]
    status_resp = await client.get("/recorder/status")
    assert status_resp.status_code == 200
    body = status_resp.json()
    assert body["recording"] is True
    assert body["current_trajectory_id"] == tid
    assert body["started_at"] is not None


async def test_start_while_recording_returns_409(client: httpx.AsyncClient) -> None:
    first = await client.post("/recorder/start", json={"label": "first"})
    assert first.status_code == 200
    second = await client.post("/recorder/start", json={"label": "second"})
    assert second.status_code == 409


async def test_start_without_permissions_returns_403_with_structured_error(
    client: httpx.AsyncClient,
) -> None:
    FakeSession.permissions_error = {
        "error": "missing_permission",
        "permissions": ["accessibility", "screen_recording"],
        "how_to_grant": {
            "accessibility": "grant accessibility",
            "screen_recording": "grant screen recording",
        },
    }
    resp = await client.post("/recorder/start", json={"label": "x"})
    assert resp.status_code == 403
    body = resp.json()
    assert body["error"] == "missing_permission"
    assert body["permissions"] == ["accessibility", "screen_recording"]
    assert body["how_to_grant"]["accessibility"] == "grant accessibility"
    assert body["how_to_grant"]["screen_recording"] == "grant screen recording"


# ------------------------------------------------------------- stop route


async def test_stop_returns_summary(client: httpx.AsyncClient) -> None:
    started = await client.post("/recorder/start", json={"label": "x"})
    tid = started.json()["trajectory_id"]
    resp = await client.post("/recorder/stop")
    assert resp.status_code == 200
    body = resp.json()
    assert body["trajectory_id"] == tid
    assert body["event_count"] == 2
    assert body["duration_ms"] >= 0


async def test_stop_without_recording_returns_409(client: httpx.AsyncClient) -> None:
    resp = await client.post("/recorder/stop")
    assert resp.status_code == 409


async def test_status_after_stop_returns_idle(client: httpx.AsyncClient) -> None:
    await client.post("/recorder/start", json={"label": "x"})
    await client.post("/recorder/stop")
    status_resp = await client.get("/recorder/status")
    assert status_resp.json()["recording"] is False
    assert status_resp.json()["current_trajectory_id"] is None


# ------------------------------------------------------- list trajectories


async def test_list_trajectories_empty(client: httpx.AsyncClient) -> None:
    resp = await client.get("/trajectories")
    assert resp.status_code == 200
    assert resp.json() == []


async def test_list_trajectories_after_recording(client: httpx.AsyncClient) -> None:
    started = await client.post("/recorder/start", json={"label": "demo"})
    tid = started.json()["trajectory_id"]
    await client.post("/recorder/stop")

    resp = await client.get("/trajectories")
    assert resp.status_code == 200
    items = resp.json()
    assert len(items) == 1
    item = items[0]
    assert item["id"] == tid
    assert item["label"] == "demo"
    assert item["event_count"] == 2
    assert item["duration_ms"] == 1500
    assert item["started_at"] == "2026-04-23T12:00:00.000+00:00"
    assert item["stopped_at"] == "2026-04-23T12:00:01.500+00:00"


async def test_list_skips_dirs_without_metadata(
    client: httpx.AsyncClient, app: FastAPI
) -> None:
    state = _state(app)
    (state.root / "stray").mkdir(parents=True)
    (state.root / "stray" / "garbage.txt").write_text("nope")
    resp = await client.get("/trajectories")
    assert resp.status_code == 200
    assert resp.json() == []


# -------------------------------------------------------- get trajectory


async def test_get_trajectory_returns_full_payload(client: httpx.AsyncClient) -> None:
    started = await client.post("/recorder/start", json={"label": "demo"})
    tid = started.json()["trajectory_id"]
    await client.post("/recorder/stop")

    resp = await client.get(f"/trajectories/{tid}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["id"] == tid
    assert body["metadata"]["id"] == tid
    assert body["metadata"]["label"] == "demo"
    assert len(body["events"]) == 2
    assert body["events"][0]["seq"] == 1
    assert body["events"][1]["seq"] == 2
    assert len(body["screenshots"]) == 1
    # URL is fully qualified for the UI to fetch directly.
    assert body["screenshots"][0].endswith(f"/trajectories/{tid}/screenshots/0001.png")


async def test_get_trajectory_404_when_missing(client: httpx.AsyncClient) -> None:
    resp = await client.get("/trajectories/does-not-exist")
    assert resp.status_code == 404


# ------------------------------------------------------- delete trajectory


async def test_delete_trajectory_removes_directory(
    client: httpx.AsyncClient, app: FastAPI
) -> None:
    started = await client.post("/recorder/start", json={"label": "demo"})
    tid = started.json()["trajectory_id"]
    await client.post("/recorder/stop")

    state = _state(app)
    assert (state.root / tid).is_dir()

    resp = await client.delete(f"/trajectories/{tid}")
    assert resp.status_code == 200
    assert resp.json() == {"deleted": tid}
    assert not (state.root / tid).exists()

    follow_up = await client.get(f"/trajectories/{tid}")
    assert follow_up.status_code == 404


async def test_delete_404_when_missing(client: httpx.AsyncClient) -> None:
    resp = await client.delete("/trajectories/never-existed")
    assert resp.status_code == 404


async def test_delete_active_trajectory_returns_409(client: httpx.AsyncClient) -> None:
    started = await client.post("/recorder/start", json={"label": "demo"})
    tid = started.json()["trajectory_id"]
    resp = await client.delete(f"/trajectories/{tid}")
    assert resp.status_code == 409


# --------------------------------------------------- screenshot endpoint


async def test_get_screenshot_serves_png_bytes(client: httpx.AsyncClient) -> None:
    started = await client.post("/recorder/start", json={"label": "demo"})
    tid = started.json()["trajectory_id"]
    await client.post("/recorder/stop")

    resp = await client.get(f"/trajectories/{tid}/screenshots/0001.png")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"
    assert resp.content.startswith(PNG_MAGIC)


async def test_get_screenshot_404_when_missing(client: httpx.AsyncClient) -> None:
    started = await client.post("/recorder/start", json={"label": "demo"})
    tid = started.json()["trajectory_id"]
    await client.post("/recorder/stop")

    resp = await client.get(f"/trajectories/{tid}/screenshots/9999.png")
    assert resp.status_code == 404


async def test_get_screenshot_rejects_non_png_suffix(
    client: httpx.AsyncClient,
) -> None:
    started = await client.post("/recorder/start", json={"label": "demo"})
    tid = started.json()["trajectory_id"]
    await client.post("/recorder/stop")

    resp = await client.get(f"/trajectories/{tid}/screenshots/evil.txt")
    assert resp.status_code == 404


@pytest.mark.parametrize(
    "filename",
    ["..", ".", "", "../metadata.json", "..\\metadata.json", "evil.txt"],
)
def test_safe_screenshot_path_rejects_unsafe_inputs(
    tmp_path: Path, filename: str
) -> None:
    """Direct unit test for ``_safe_screenshot_path`` — guards against
    path-traversal attempts that would otherwise be normalised away by
    httpx before the request reaches the handler."""
    from recorder.api import _safe_screenshot_path

    assert _safe_screenshot_path(tmp_path, "some-id", filename) is None


def test_safe_screenshot_path_accepts_well_formed_filename(tmp_path: Path) -> None:
    from recorder.api import _safe_screenshot_path

    target = tmp_path / "abc" / "screenshots" / "0001.png"
    target.parent.mkdir(parents=True)
    target.write_bytes(PNG_MAGIC)
    resolved = _safe_screenshot_path(tmp_path, "abc", "0001.png")
    assert resolved == target.resolve()


# ------------------------------------------------------- state class tests


def test_recorder_state_creates_root_directory(tmp_path: Path) -> None:
    root = tmp_path / "trajectories"
    assert not root.exists()
    RecorderState(root, session_factory=FakeSession)
    assert root.is_dir()


def test_recorder_state_status_idle(tmp_path: Path) -> None:
    state = RecorderState(tmp_path / "trajectories", session_factory=FakeSession)
    assert state.active_trajectory_id() is None
    assert state.status().recording is False


def test_recorder_state_start_stop_round_trip(tmp_path: Path) -> None:
    state = RecorderState(tmp_path / "trajectories", session_factory=FakeSession)
    tid = state.start("demo")
    assert state.active_trajectory_id() == tid
    summary = state.stop()
    assert summary["trajectory_id"] == tid
    assert state.active_trajectory_id() is None


def test_recorder_state_start_twice_raises(tmp_path: Path) -> None:
    state = RecorderState(tmp_path / "trajectories", session_factory=FakeSession)
    state.start("first")
    with pytest.raises(SessionAlreadyActiveError):
        state.start("second")


def test_recorder_state_concurrent_start_yields_one_session(tmp_path: Path) -> None:
    """The lock around start() must serialise concurrent attempts."""
    state = RecorderState(tmp_path / "trajectories", session_factory=FakeSession)
    successes: list[str] = []
    failures: list[Exception] = []
    barrier = threading.Barrier(8)

    def attempt() -> None:
        barrier.wait()
        try:
            successes.append(state.start("race"))
        except SessionAlreadyActiveError as exc:
            failures.append(exc)

    threads = [threading.Thread(target=attempt) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(successes) == 1
    assert len(failures) == 7


# ----------------------------------------------- list <-> on-disk consistency


def test_list_trajectories_skips_unreadable_metadata(tmp_path: Path) -> None:
    """A directory whose metadata.json is corrupt is skipped, not crashed on."""
    from recorder.api import list_trajectories

    root = tmp_path / "trajectories"
    state = RecorderState(root, session_factory=FakeSession)
    bad = root / "broken"
    bad.mkdir()
    (bad / "metadata.json").write_text("{ this is not json")
    assert list_trajectories(state) == []
