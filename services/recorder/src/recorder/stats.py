"""Daily-usage statistics aggregated from on-disk trajectories.

The dashboard endpoints serve a different access pattern from the recorder
itself: they want *cross-trajectory* aggregates ("how much time did I spend
in Chrome over the last 7 days?") rather than the per-recording detail the
existing ``/trajectories/{id}`` endpoint returns.

Source-of-truth here is the same one the recorder writes:

* :class:`recorder.index_db.IndexDB` for the trajectory listing (fast).
* ``metadata.json``'s ``app_focus_history`` for **time per app** (the gold —
  one entry per app activation, with entered_at/exited_at).
* ``events.jsonl`` for **event-mix counts**, **top window titles**, the
  **hour-of-day activity heatmap**, and **text-input volume**.

Everything is pure functions over paths so callers (and tests) can stub the
filesystem without bringing up the full recorder service.

Performance: the dashboard caps queries to a window (default 7 days). For a
power-user with hundreds of recordings per week this still means scanning
hundreds of ``events.jsonl`` files; that's acceptable today (the lines are
small, parsing is streamed) but is the obvious thing to cache later.
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any

from typing_extensions import TypedDict

from recorder.index_db import IndexDB, TrajectoryRow
from recorder.storage import trajectory_dir

__all__ = [
    "AppUsage",
    "DailyBucket",
    "StatsSummary",
    "WindowUsage",
    "compute_summary",
]

logger = logging.getLogger(__name__)

#: Event types that count toward "active interaction" hour-of-day buckets.
#: ``keyframe`` and ``tap_reenabled`` are infrastructural noise — including
#: them would inflate hours when the user is idle (keyframes fire on a
#: timer).
_INTERACTIVE_EVENT_TYPES: frozenset[str] = frozenset(
    {"click", "keypress", "scroll", "app_switch", "window_focus", "text_input"}
)


class AppUsage(TypedDict):
    """One row of the "top apps" table."""

    bundle_id: str
    name: str
    seconds: float
    sessions: int


class WindowUsage(TypedDict):
    """One row of the "top windows / pages" table."""

    app_name: str
    window_title: str
    count: int


class DailyBucket(TypedDict):
    """One row of the daily timeseries."""

    date: str  # YYYY-MM-DD (UTC)
    recorded_seconds: float
    event_count: int
    click_count: int
    keypress_count: int
    text_input_chars: int


class StatsSummary(TypedDict):
    """The full payload the ``/stats/summary`` endpoint returns."""

    window_days: int
    range_start: str  # YYYY-MM-DD
    range_end: str  # YYYY-MM-DD
    trajectory_count: int
    recorded_seconds: float
    event_counts: dict[str, int]
    text_input_chars: int
    top_apps: list[AppUsage]
    top_windows: list[WindowUsage]
    hour_of_day: list[int]  # length 24
    daily: list[DailyBucket]


# ----------------------------------------------------------- public entry

def compute_summary(
    *,
    index_db: IndexDB,
    trajectories_root: Path,
    window_days: int = 7,
    now: datetime | None = None,
    top_n: int = 10,
) -> StatsSummary:
    """Aggregate stats across every trajectory whose ``started_at`` falls in
    the last ``window_days`` days (UTC, inclusive of today).

    ``now`` is injectable so tests can pin the wall clock.
    """
    if window_days < 1:
        raise ValueError("window_days must be >= 1")
    now = now or datetime.now(UTC)
    range_end = now.date()
    range_start = range_end - timedelta(days=window_days - 1)

    rows = _select_rows_in_window(index_db, range_start, range_end)

    app_seconds: dict[str, float] = defaultdict(float)
    app_names: dict[str, str] = {}
    app_sessions: dict[str, int] = defaultdict(int)

    event_counts: Counter[str] = Counter()
    text_chars = 0

    window_counter: Counter[tuple[str, str]] = Counter()
    hour_buckets = [0] * 24

    daily_seconds: dict[date, float] = defaultdict(float)
    daily_events: Counter[date] = Counter()
    daily_clicks: Counter[date] = Counter()
    daily_keys: Counter[date] = Counter()
    daily_text_chars: Counter[date] = Counter()

    total_recorded_seconds = 0.0

    for row in rows:
        traj_dir = trajectory_dir(trajectories_root, row["id"])
        if not traj_dir.is_dir():
            continue
        meta = _read_metadata(traj_dir)
        if meta is None:
            continue

        # ---- time per app from focus history ----
        for entry in meta.get("app_focus_history") or []:
            seconds, day = _focus_entry_seconds(entry, fallback_stop=row["stopped_at"])
            if seconds <= 0:
                continue
            bundle_id = str(entry.get("bundle_id") or "")
            name = str(entry.get("name") or bundle_id or "Unknown")
            if not bundle_id:
                bundle_id = f"name:{name}"
            app_seconds[bundle_id] += seconds
            app_sessions[bundle_id] += 1
            app_names.setdefault(bundle_id, name)
            if day is not None and range_start <= day <= range_end:
                daily_seconds[day] += seconds
            total_recorded_seconds += seconds

        # ---- event-derived stats ----
        for ev in _iter_events(traj_dir / "events.jsonl"):
            etype = ev.get("type")
            if not isinstance(etype, str):
                continue
            event_counts[etype] += 1

            ts = ev.get("timestamp_ms")
            event_day: date | None = None
            if isinstance(ts, int):
                dt = datetime.fromtimestamp(ts / 1000.0, tz=UTC)
                event_day = dt.date()
                if etype in _INTERACTIVE_EVENT_TYPES:
                    hour_buckets[dt.hour] += 1
                if event_day is not None and range_start <= event_day <= range_end:
                    daily_events[event_day] += 1

            payload = ev.get("payload") or {}
            app_field = ev.get("app") or {}

            if etype == "click" and event_day is not None:
                daily_clicks[event_day] += 1
            elif etype == "keypress" and event_day is not None:
                daily_keys[event_day] += 1
            elif etype == "text_input":
                text = payload.get("text") if isinstance(payload, dict) else None
                if isinstance(text, str):
                    text_chars += len(text)
                    if event_day is not None:
                        daily_text_chars[event_day] += len(text)
            elif etype == "window_focus":
                title = payload.get("window_title") if isinstance(payload, dict) else None
                if isinstance(title, str) and title.strip():
                    app_name = (
                        app_field.get("name")
                        if isinstance(app_field, dict)
                        else None
                    )
                    if not isinstance(app_name, str) or not app_name:
                        app_name = "Unknown"
                    window_counter[(app_name, title.strip())] += 1

    top_apps: list[AppUsage] = sorted(
        (
            AppUsage(
                bundle_id=bid,
                name=app_names.get(bid, bid),
                seconds=round(secs, 2),
                sessions=app_sessions[bid],
            )
            for bid, secs in app_seconds.items()
        ),
        key=lambda a: a["seconds"],
        reverse=True,
    )[:top_n]

    top_windows: list[WindowUsage] = [
        WindowUsage(app_name=app, window_title=title, count=count)
        for (app, title), count in window_counter.most_common(top_n)
    ]

    daily: list[DailyBucket] = []
    for offset in range(window_days):
        d = range_start + timedelta(days=offset)
        daily.append(
            DailyBucket(
                date=d.isoformat(),
                recorded_seconds=round(daily_seconds.get(d, 0.0), 2),
                event_count=daily_events.get(d, 0),
                click_count=daily_clicks.get(d, 0),
                keypress_count=daily_keys.get(d, 0),
                text_input_chars=daily_text_chars.get(d, 0),
            )
        )

    return StatsSummary(
        window_days=window_days,
        range_start=range_start.isoformat(),
        range_end=range_end.isoformat(),
        trajectory_count=len(rows),
        recorded_seconds=round(total_recorded_seconds, 2),
        event_counts=dict(event_counts),
        text_input_chars=text_chars,
        top_apps=top_apps,
        top_windows=top_windows,
        hour_of_day=hour_buckets,
        daily=daily,
    )


# ----------------------------------------------------------------- helpers


def _select_rows_in_window(
    index_db: IndexDB,
    range_start: date,
    range_end: date,
) -> list[TrajectoryRow]:
    start_dt = datetime.combine(range_start, time.min, tzinfo=UTC)
    end_dt = datetime.combine(range_end, time.max, tzinfo=UTC)
    selected: list[TrajectoryRow] = []
    for row in index_db.list_all():
        started = row["started_at"]
        if not isinstance(started, str):
            continue
        try:
            started_dt = datetime.fromisoformat(started)
        except ValueError:
            continue
        if started_dt.tzinfo is None:
            started_dt = started_dt.replace(tzinfo=UTC)
        if start_dt <= started_dt <= end_dt:
            selected.append(row)
    return selected


def _read_metadata(traj_dir: Path) -> dict[str, Any] | None:
    path = traj_dir / "metadata.json"
    if not path.is_file():
        return None
    try:
        with path.open(encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        logger.warning("stats: could not read %s", path, exc_info=True)
        return None
    return data if isinstance(data, dict) else None


def _iter_events(path: Path) -> Iterator[dict[str, Any]]:
    if not path.is_file():
        return
    try:
        fh = path.open(encoding="utf-8")
    except OSError:
        return
    with fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj


def _focus_entry_seconds(
    entry: dict[str, Any],
    *,
    fallback_stop: str | None,
) -> tuple[float, date | None]:
    """Return ``(duration_seconds, day_of_entered_at)`` for a focus entry.

    A null ``exited_at`` (recording crashed or trajectory was an interrupted
    one) falls back to the trajectory's ``stopped_at`` so the time isn't
    silently lost. If both are missing we return zero.
    """
    entered = entry.get("entered_at")
    exited = entry.get("exited_at") or fallback_stop
    if not isinstance(entered, str) or not isinstance(exited, str):
        return 0.0, None
    try:
        e_dt = datetime.fromisoformat(entered)
        x_dt = datetime.fromisoformat(exited)
    except ValueError:
        return 0.0, None
    if e_dt.tzinfo is None:
        e_dt = e_dt.replace(tzinfo=UTC)
    if x_dt.tzinfo is None:
        x_dt = x_dt.replace(tzinfo=UTC)
    seconds = (x_dt - e_dt).total_seconds()
    if seconds < 0:
        return 0.0, None
    return seconds, e_dt.date()


def daily_buckets_skeleton(days: int, end: date) -> Iterable[date]:
    """Inclusive series of UTC dates ending on ``end``.

    Exposed for tests and for callers that want to drive their own iteration.
    """
    start = end - timedelta(days=days - 1)
    for offset in range(days):
        yield start + timedelta(days=offset)
