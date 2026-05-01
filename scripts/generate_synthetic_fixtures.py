"""Synthesize the five reference-workflow trajectory fixtures.

Fixtures live under ``fixtures/trajectories/<slug>/`` and are consumed by
the Synthesizer and Runner module tests. For v1 (personal-use MVP) they
are hand-crafted JSON + tiny synthetic PNGs that pass the locked
``contracts/trajectory.schema.json`` contract — a human re-records them
on a real Mac during smoke testing (see ``scripts/regenerate_fixtures.sh``).

Run from the repo root:

    uv run python scripts/generate_synthetic_fixtures.py

The script validates every event it writes via ``recorder.schema``.
"""

from __future__ import annotations

import json
import shutil
import struct
import sys
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Allow running as a plain script without `uv run ...` bootstrapping.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "services" / "recorder" / "src"))

from recorder.schema import validate_event, validate_metadata  # noqa: E402

FIXTURES_ROOT = REPO_ROOT / "fixtures" / "trajectories"
BASE_EPOCH_MS = 1_774_000_000_000  # 2026-04-01T00:26:40Z — deterministic.
DISPLAY_INFO = {"width": 1440, "height": 900, "scale_factor": 2.0}
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    crc = zlib.crc32(kind + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", crc)


def _make_tiny_png(width: int = 4, height: int = 4) -> bytes:
    """Structurally valid PNG: magic + IHDR + one IDAT + IEND.

    Small enough (~60 bytes) that 5 fixtures * 8 keyframes each stay well
    under the 5 MB fixtures-directory cap called out in R-013.
    """
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)  # 8-bit RGB
    raw_row = b"\x00" + b"\x00\x00\x00" * width  # filter byte + RGB pixels
    raw = raw_row * height
    idat = zlib.compress(raw, 6)
    return (
        PNG_MAGIC
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", idat)
        + _png_chunk(b"IEND", b"")
    )


# ---------------------------------------------------------------------------
# Workflow definitions — intentionally verbose but flat so humans can eyeball
# each trajectory. Each ``events`` list is a sequence of partial dicts; the
# writer below fills in ``seq`` / ``timestamp_ms`` / ``app`` deterministically.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AppCtx:
    bundle_id: str
    name: str
    pid: int


CHROME = AppCtx("com.google.Chrome", "Google Chrome", 501)
FINDER = AppCtx("com.apple.finder", "Finder", 502)
SLACK = AppCtx("com.tinyspeck.slackmacgap", "Slack", 503)
NOTES = AppCtx("com.apple.Notes", "Notes", 504)
SAFARI = AppCtx("com.apple.Safari", "Safari", 505)


def _app(ctx: AppCtx) -> dict[str, Any]:
    return {"bundle_id": ctx.bundle_id, "name": ctx.name, "pid": ctx.pid}


def _target(
    role: str,
    label: str | None,
    frame: tuple[float, float, float, float],
    *,
    description: str | None = None,
    ax_identifier: str | None = None,
) -> dict[str, Any]:
    x, y, w, h = frame
    return {
        "role": role,
        "label": label,
        "description": description,
        "frame": {"x": x, "y": y, "w": w, "h": h},
        "ax_identifier": ax_identifier,
    }


def _click(
    app_ctx: AppCtx,
    x: float,
    y: float,
    *,
    target: dict[str, Any] | None,
    button: str = "left",
    modifiers: list[str] | None = None,
    screenshot_ref: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"button": button}
    if modifiers is not None:
        payload["modifiers"] = modifiers
    ev: dict[str, Any] = {
        "type": "click",
        "app": _app(app_ctx),
        "target": target,
        "payload": payload,
    }
    if screenshot_ref is not None:
        ev["screenshot_ref"] = screenshot_ref
    return ev


def _keyframe(app_ctx: AppCtx, reason: str, screenshot_ref: str) -> dict[str, Any]:
    return {
        "type": "keyframe",
        "app": _app(app_ctx),
        "screenshot_ref": screenshot_ref,
        "payload": {"reason": reason},
    }


def _app_switch(
    app_ctx: AppCtx,
    *,
    from_bundle_id: str | None,
    from_name: str | None,
    to_bundle_id: str,
    to_name: str,
    screenshot_ref: str | None = None,
) -> dict[str, Any]:
    ev: dict[str, Any] = {
        "type": "app_switch",
        "app": _app(app_ctx),
        "payload": {
            "from_bundle_id": from_bundle_id,
            "from_name": from_name,
            "to_bundle_id": to_bundle_id,
            "to_name": to_name,
        },
    }
    if screenshot_ref is not None:
        ev["screenshot_ref"] = screenshot_ref
    return ev


def _text_input(
    app_ctx: AppCtx,
    text: str,
    field_label: str | None,
) -> dict[str, Any]:
    return {
        "type": "text_input",
        "app": _app(app_ctx),
        "payload": {"text": text, "field_label": field_label},
    }


def _keypress(
    app_ctx: AppCtx, keys: list[str], modifiers: list[str] | None = None
) -> dict[str, Any]:
    payload: dict[str, Any] = {"keys": keys}
    if modifiers is not None:
        payload["modifiers"] = modifiers
    return {"type": "keypress", "app": _app(app_ctx), "payload": payload}


def _scroll(app_ctx: AppCtx, direction: str, delta: float) -> dict[str, Any]:
    return {
        "type": "scroll",
        "app": _app(app_ctx),
        "payload": {"direction": direction, "delta": delta},
    }


def _window_focus(app_ctx: AppCtx, title: str) -> dict[str, Any]:
    return {
        "type": "window_focus",
        "app": _app(app_ctx),
        "payload": {"window_title": title},
    }


# ---------------------------------------------------------------------------
# Per-workflow event scripts
# ---------------------------------------------------------------------------


def _gmail_reply_events() -> list[dict[str, Any]]:
    return [
        _app_switch(
            CHROME,
            from_bundle_id=None,
            from_name=None,
            to_bundle_id=CHROME.bundle_id,
            to_name=CHROME.name,
            screenshot_ref="screenshots/0001.png",
        ),
        _window_focus(CHROME, "Inbox (3) - you@example.com - Gmail"),
        _keyframe(CHROME, "periodic", "screenshots/0003.png"),
        _keyframe(CHROME, "pre_click", "screenshots/0004.png"),
        _click(
            CHROME,
            380.0,
            220.0,
            target=_target(
                "AXRow",
                "Unread — Alice Smith — Q2 review",
                (120.0, 210.0, 900.0, 48.0),
                description="Email list row",
                ax_identifier="gmail-thread-0001",
            ),
            screenshot_ref="screenshots/0005.png",
        ),
        _keyframe(CHROME, "post_click", "screenshots/0006.png"),
        _click(
            CHROME,
            520.0,
            640.0,
            target=_target(
                "AXButton",
                "Reply",
                (500.0, 620.0, 80.0, 32.0),
                ax_identifier="gmail-reply-btn",
            ),
        ),
        _text_input(
            CHROME,
            "Thanks — I'll review this today and send notes before 5pm.",
            "Message body",
        ),
        _keypress(CHROME, ["tab"]),
        _click(
            CHROME,
            470.0,
            780.0,
            target=_target(
                "AXButton",
                "Send",
                (450.0, 760.0, 72.0, 32.0),
                ax_identifier="gmail-send-btn",
            ),
        ),
        _keyframe(CHROME, "post_click", "screenshots/0011.png"),
    ]


def _calendar_block_events() -> list[dict[str, Any]]:
    return [
        _app_switch(
            CHROME,
            from_bundle_id=FINDER.bundle_id,
            from_name=FINDER.name,
            to_bundle_id=CHROME.bundle_id,
            to_name=CHROME.name,
            screenshot_ref="screenshots/0001.png",
        ),
        _window_focus(CHROME, "Google Calendar - Week of Apr 20"),
        _keyframe(CHROME, "periodic", "screenshots/0003.png"),
        _click(
            CHROME,
            50.0,
            130.0,
            target=_target(
                "AXButton",
                "Create",
                (24.0, 110.0, 96.0, 40.0),
                ax_identifier="gcal-create-btn",
            ),
            screenshot_ref="screenshots/0004.png",
        ),
        _click(
            CHROME,
            110.0,
            180.0,
            target=_target(
                "AXMenuItem",
                "Event",
                (80.0, 170.0, 160.0, 32.0),
                ax_identifier="gcal-create-event",
            ),
        ),
        _keyframe(CHROME, "post_click", "screenshots/0006.png"),
        _text_input(CHROME, "Focus block", "Add title"),
        _click(
            CHROME,
            420.0,
            260.0,
            target=_target(
                "AXButton",
                "2:00 PM - 2:30 PM",
                (400.0, 250.0, 200.0, 28.0),
                ax_identifier="gcal-time-picker",
            ),
        ),
        _keypress(CHROME, ["s"], modifiers=["cmd"]),
        _click(
            CHROME,
            680.0,
            320.0,
            target=_target(
                "AXButton",
                "Save",
                (660.0, 310.0, 80.0, 32.0),
                ax_identifier="gcal-save",
            ),
            screenshot_ref="screenshots/0010.png",
        ),
        _keyframe(CHROME, "periodic", "screenshots/0011.png"),
    ]


def _finder_organize_events() -> list[dict[str, Any]]:
    return [
        _app_switch(
            FINDER,
            from_bundle_id=CHROME.bundle_id,
            from_name=CHROME.name,
            to_bundle_id=FINDER.bundle_id,
            to_name=FINDER.name,
            screenshot_ref="screenshots/0001.png",
        ),
        _window_focus(FINDER, "Downloads"),
        _click(
            FINDER,
            80.0,
            180.0,
            target=_target(
                "AXRow",
                "Downloads",
                (60.0, 170.0, 180.0, 24.0),
                description="Sidebar shortcut",
                ax_identifier="finder-sidebar-downloads",
            ),
        ),
        _keyframe(FINDER, "post_click", "screenshots/0004.png"),
        _scroll(FINDER, "down", 120.0),
        _click(
            FINDER,
            360.0,
            300.0,
            target=_target(
                "AXRow",
                "old-invoice-2025.pdf",
                (320.0, 290.0, 320.0, 24.0),
                ax_identifier="finder-file-old-invoice-2025.pdf",
            ),
            screenshot_ref="screenshots/0006.png",
        ),
        _keypress(FINDER, ["c"], modifiers=["cmd"]),
        _text_input(FINDER, "Archive", "Go to folder"),
        _click(
            FINDER,
            100.0,
            420.0,
            target=_target(
                "AXRow",
                "Archive",
                (80.0, 410.0, 180.0, 24.0),
                ax_identifier="finder-sidebar-archive",
            ),
        ),
        _keypress(FINDER, ["v"], modifiers=["cmd", "opt"]),
        _keyframe(FINDER, "post_click", "screenshots/0011.png"),
        _keyframe(FINDER, "periodic", "screenshots/0012.png"),
    ]


def _slack_status_events() -> list[dict[str, Any]]:
    return [
        _app_switch(
            SLACK,
            from_bundle_id=CHROME.bundle_id,
            from_name=CHROME.name,
            to_bundle_id=SLACK.bundle_id,
            to_name=SLACK.name,
            screenshot_ref="screenshots/0001.png",
        ),
        _window_focus(SLACK, "Slack — Acme HQ"),
        _click(
            SLACK,
            1380.0,
            80.0,
            target=_target(
                "AXButton",
                "Your profile",
                (1360.0, 60.0, 40.0, 40.0),
                ax_identifier="slack-avatar",
            ),
            screenshot_ref="screenshots/0003.png",
        ),
        _keyframe(SLACK, "post_click", "screenshots/0004.png"),
        _click(
            SLACK,
            1200.0,
            200.0,
            target=_target(
                "AXMenuItem",
                "Set a status",
                (1180.0, 190.0, 220.0, 28.0),
                ax_identifier="slack-set-status",
            ),
        ),
        _text_input(SLACK, "🎯 heads down", "What's your status?"),
        _click(
            SLACK,
            960.0,
            420.0,
            target=_target(
                "AXPopUpButton",
                "Clear after",
                (940.0, 410.0, 160.0, 28.0),
                ax_identifier="slack-clear-after",
            ),
        ),
        _click(
            SLACK,
            960.0,
            480.0,
            target=_target(
                "AXMenuItem",
                "2 hours",
                (940.0, 470.0, 160.0, 28.0),
                ax_identifier="slack-clear-2h",
            ),
            screenshot_ref="screenshots/0008.png",
        ),
        _click(
            SLACK,
            1080.0,
            560.0,
            target=_target(
                "AXButton",
                "Save",
                (1060.0, 550.0, 80.0, 32.0),
                ax_identifier="slack-save-status",
            ),
        ),
        _keyframe(SLACK, "periodic", "screenshots/0010.png"),
    ]


def _notes_daily_events() -> list[dict[str, Any]]:
    return [
        _app_switch(
            NOTES,
            from_bundle_id=SAFARI.bundle_id,
            from_name=SAFARI.name,
            to_bundle_id=NOTES.bundle_id,
            to_name=NOTES.name,
            screenshot_ref="screenshots/0001.png",
        ),
        _window_focus(NOTES, "All iCloud"),
        _keypress(NOTES, ["n"], modifiers=["cmd"]),
        _keyframe(NOTES, "post_click", "screenshots/0004.png"),
        _click(
            NOTES,
            520.0,
            180.0,
            target=_target(
                "AXTextField",
                "Title",
                (500.0, 170.0, 400.0, 28.0),
                ax_identifier="notes-title",
            ),
        ),
        _text_input(NOTES, "2026-04-23 Daily", "Title"),
        _click(
            NOTES,
            520.0,
            260.0,
            target=_target(
                "AXTextArea",
                "Body",
                (500.0, 240.0, 600.0, 400.0),
                ax_identifier="notes-body",
            ),
            screenshot_ref="screenshots/0007.png",
        ),
        _text_input(
            NOTES,
            "## Plan\n- [ ] focus block 2pm\n## Wins\n- \n## Notes\n- ",
            "Body",
        ),
        _keypress(NOTES, ["s"], modifiers=["cmd"]),
        _keyframe(NOTES, "periodic", "screenshots/0010.png"),
    ]


WORKFLOWS: dict[str, dict[str, Any]] = {
    "gmail_reply": {
        "label": "Reply to most recent unread email from Alice Smith",
        "uuid": "11111111-1111-4111-8111-111111111111",
        "started_at": "2026-04-01T10:00:00.000+00:00",
        "stopped_at": "2026-04-01T10:02:30.000+00:00",
        "app_focus_history": [
            {
                "bundle_id": CHROME.bundle_id,
                "name": CHROME.name,
                "entered_at": "2026-04-01T10:00:00.000+00:00",
                "exited_at": "2026-04-01T10:02:30.000+00:00",
            }
        ],
        "events": _gmail_reply_events(),
    },
    "calendar_block": {
        "label": "Create 30-minute focus block tomorrow at 2pm",
        "uuid": "22222222-2222-4222-8222-222222222222",
        "started_at": "2026-04-01T11:00:00.000+00:00",
        "stopped_at": "2026-04-01T11:02:00.000+00:00",
        "app_focus_history": [
            {
                "bundle_id": FINDER.bundle_id,
                "name": FINDER.name,
                "entered_at": "2026-04-01T11:00:00.000+00:00",
                "exited_at": "2026-04-01T11:00:02.000+00:00",
            },
            {
                "bundle_id": CHROME.bundle_id,
                "name": CHROME.name,
                "entered_at": "2026-04-01T11:00:02.000+00:00",
                "exited_at": "2026-04-01T11:02:00.000+00:00",
            },
        ],
        "events": _calendar_block_events(),
    },
    "finder_organize": {
        "label": "Move old PDFs from Downloads to Archive",
        "uuid": "33333333-3333-4333-8333-333333333333",
        "started_at": "2026-04-01T12:00:00.000+00:00",
        "stopped_at": "2026-04-01T12:03:00.000+00:00",
        "app_focus_history": [
            {
                "bundle_id": CHROME.bundle_id,
                "name": CHROME.name,
                "entered_at": "2026-04-01T12:00:00.000+00:00",
                "exited_at": "2026-04-01T12:00:01.000+00:00",
            },
            {
                "bundle_id": FINDER.bundle_id,
                "name": FINDER.name,
                "entered_at": "2026-04-01T12:00:01.000+00:00",
                "exited_at": "2026-04-01T12:03:00.000+00:00",
            },
        ],
        "events": _finder_organize_events(),
    },
    "slack_status": {
        "label": "Set Slack status to heads down for 2 hours",
        "uuid": "44444444-4444-4444-8444-444444444444",
        "started_at": "2026-04-01T13:00:00.000+00:00",
        "stopped_at": "2026-04-01T13:01:40.000+00:00",
        "app_focus_history": [
            {
                "bundle_id": CHROME.bundle_id,
                "name": CHROME.name,
                "entered_at": "2026-04-01T13:00:00.000+00:00",
                "exited_at": "2026-04-01T13:00:01.000+00:00",
            },
            {
                "bundle_id": SLACK.bundle_id,
                "name": SLACK.name,
                "entered_at": "2026-04-01T13:00:01.000+00:00",
                "exited_at": "2026-04-01T13:01:40.000+00:00",
            },
        ],
        "events": _slack_status_events(),
    },
    "notes_daily": {
        "label": "Create today's daily note from template",
        "uuid": "55555555-5555-4555-8555-555555555555",
        "started_at": "2026-04-01T14:00:00.000+00:00",
        "stopped_at": "2026-04-01T14:01:30.000+00:00",
        "app_focus_history": [
            {
                "bundle_id": SAFARI.bundle_id,
                "name": SAFARI.name,
                "entered_at": "2026-04-01T14:00:00.000+00:00",
                "exited_at": "2026-04-01T14:00:01.000+00:00",
            },
            {
                "bundle_id": NOTES.bundle_id,
                "name": NOTES.name,
                "entered_at": "2026-04-01T14:00:01.000+00:00",
                "exited_at": "2026-04-01T14:01:30.000+00:00",
            },
        ],
        "events": _notes_daily_events(),
    },
}


def _finalise_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fill in seq + timestamp_ms deterministically and validate each event."""
    finalised: list[dict[str, Any]] = []
    for idx, raw in enumerate(events, start=1):
        ev = dict(raw)
        ev["seq"] = idx
        ev["timestamp_ms"] = BASE_EPOCH_MS + (idx * 1500)
        # target is optional at the schema level; keep the key present
        # (possibly null) only for event types that carry semantic target data.
        if ev["type"] == "click" and "target" not in ev:
            ev["target"] = None
        validate_event(ev)
        finalised.append(ev)
    return finalised


def _write_fixture(slug: str, spec: dict[str, Any]) -> None:
    dest = FIXTURES_ROOT / slug
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    (dest / "screenshots").mkdir()

    metadata = {
        "id": spec["uuid"],
        "label": spec["label"],
        "started_at": spec["started_at"],
        "stopped_at": spec["stopped_at"],
        "display_info": dict(DISPLAY_INFO),
        "app_focus_history": list(spec["app_focus_history"]),
    }
    validate_metadata(metadata)
    (dest / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    events = _finalise_events(spec["events"])
    with (dest / "events.jsonl").open("w", encoding="utf-8") as fh:
        for ev in events:
            fh.write(json.dumps(ev) + "\n")

    refs = {ev["screenshot_ref"] for ev in events if ev.get("screenshot_ref")}
    png = _make_tiny_png()
    for ref in sorted(refs):
        rel = Path(ref)
        (dest / rel).write_bytes(png)


def main() -> None:
    FIXTURES_ROOT.mkdir(parents=True, exist_ok=True)
    # Keep the gitkeep so an empty regen (e.g. via shell script without this
    # python running first) doesn't orphan the directory.
    gitkeep = FIXTURES_ROOT / ".gitkeep"
    gitkeep.touch(exist_ok=True)
    for slug, spec in WORKFLOWS.items():
        _write_fixture(slug, spec)
        print(f"wrote fixtures/trajectories/{slug}/")


if __name__ == "__main__":
    main()
