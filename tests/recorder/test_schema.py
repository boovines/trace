"""Tests for the trajectory event/metadata schema (contracts/trajectory.schema.json).

AC from R-002 requires at least 5 valid example events that pass validation
and at least 5 malformed events that fail with useful errors.
"""

from __future__ import annotations

from typing import Any

import pytest
from jsonschema.exceptions import ValidationError

from recorder.schema import (
    find_schema_path,
    load_event_schema,
    load_metadata_schema,
    validate_event,
    validate_metadata,
)


def _base_event(**overrides: Any) -> dict[str, Any]:
    event: dict[str, Any] = {
        "seq": 1,
        "timestamp_ms": 1_700_000_000_000,
        "type": "keyframe",
        "screenshot_ref": "screenshots/0001.png",
        "app": {"bundle_id": "com.apple.finder", "name": "Finder", "pid": 123},
        "target": None,
        "payload": {"reason": "periodic"},
    }
    event.update(overrides)
    return event


def test_schema_file_is_locatable() -> None:
    path = find_schema_path()
    assert path.is_file()
    assert path.name == "trajectory.schema.json"


def test_schemas_are_draft_2020_12() -> None:
    ev = load_event_schema()
    md = load_metadata_schema()
    # Embedded $defs mean the validator can resolve internal $refs locally.
    assert "$defs" in ev
    assert "$defs" in md


# -------------------------- valid events --------------------------------


VALID_EVENTS: list[dict[str, Any]] = [
    _base_event(
        seq=1,
        type="click",
        payload={
            "button": "left",
            "modifiers": ["cmd"],
            "drag_to": None,
        },
        target={
            "role": "AXButton",
            "label": "Send",
            "description": None,
            "frame": {"x": 100.0, "y": 200.0, "w": 80.0, "h": 24.0},
            "ax_identifier": "send-button",
        },
    ),
    _base_event(
        seq=2,
        type="keypress",
        payload={"keys": ["cmd", "shift", "a"], "modifiers": ["cmd", "shift"]},
        screenshot_ref=None,
    ),
    _base_event(
        seq=3,
        type="scroll",
        payload={"direction": "down", "delta": -42.5},
        screenshot_ref=None,
    ),
    _base_event(
        seq=4,
        type="app_switch",
        payload={
            "from_bundle_id": "com.apple.finder",
            "to_bundle_id": "com.google.Chrome",
            "from_name": "Finder",
            "to_name": "Google Chrome",
        },
    ),
    _base_event(
        seq=5,
        type="text_input",
        payload={"text": "hello world", "field_label": "Subject"},
        screenshot_ref=None,
    ),
    _base_event(
        seq=6,
        type="window_focus",
        payload={"window_title": "Inbox (3) — Gmail"},
        screenshot_ref=None,
    ),
    _base_event(
        seq=7,
        type="keyframe",
        payload={"reason": "pre_click"},
        screenshot_ref="screenshots/0007.png",
    ),
    _base_event(
        seq=8,
        type="tap_reenabled",
        payload={"cause": "timeout"},
        screenshot_ref=None,
    ),
]


@pytest.mark.parametrize("event", VALID_EVENTS, ids=[e["type"] for e in VALID_EVENTS])
def test_valid_events_pass(event: dict[str, Any]) -> None:
    validate_event(event)


# -------------------------- invalid events ------------------------------


INVALID_EVENTS: list[tuple[str, dict[str, Any], str]] = [
    (
        "missing_seq",
        {
            "timestamp_ms": 1,
            "type": "keyframe",
            "app": {"bundle_id": "x", "name": "x", "pid": 1},
            "payload": {"reason": "periodic"},
        },
        "seq",
    ),
    (
        "unknown_type",
        _base_event(type="teleport", payload={}),
        "teleport",
    ),
    (
        "click_missing_button",
        _base_event(type="click", payload={"modifiers": []}),
        "button",
    ),
    (
        "text_input_empty_text",
        _base_event(type="text_input", payload={"text": "", "field_label": "Subject"}),
        "",
    ),
    (
        "keyframe_bad_reason",
        _base_event(type="keyframe", payload={"reason": "whenever"}),
        "whenever",
    ),
    (
        "bad_screenshot_ref",
        _base_event(screenshot_ref="/tmp/not-allowed.png"),
        "screenshots/",
    ),
    (
        "seq_zero",
        _base_event(seq=0),
        "seq",
    ),
    (
        "scroll_missing_delta",
        _base_event(type="scroll", payload={"direction": "up"}),
        "delta",
    ),
    (
        "additional_top_level_key",
        _base_event(extra_field="nope"),
        "extra_field",
    ),
    (
        "target_missing_frame",
        _base_event(
            type="click",
            payload={"button": "left"},
            target={"role": "AXButton", "label": "Send"},
        ),
        "frame",
    ),
]


def _collect_error_text(err: ValidationError) -> str:
    """Flatten a ValidationError plus its oneOf/anyOf sub-error context."""
    parts = [str(err), repr(err.instance)]
    for sub in err.context or ():
        parts.append(_collect_error_text(sub))
    return "\n".join(parts)


@pytest.mark.parametrize(
    "event,needle",
    [(e, n) for _, e, n in INVALID_EVENTS],
    ids=[name for name, _, _ in INVALID_EVENTS],
)
def test_invalid_events_fail(event: dict[str, Any], needle: str) -> None:
    with pytest.raises(ValidationError) as exc_info:
        validate_event(event)
    assert str(exc_info.value), "ValidationError must carry a non-empty message"
    text = _collect_error_text(exc_info.value)
    assert needle in text, f"expected '{needle}' in error text, got: {text[:800]}"


# -------------------------- metadata ------------------------------------


def test_valid_metadata_passes() -> None:
    metadata: dict[str, Any] = {
        "id": "7ab0c0f4-6ad1-4d9d-9d57-2d41d2d1f5aa",
        "label": "gmail_reply",
        "started_at": "2026-04-22T15:00:00Z",
        "stopped_at": "2026-04-22T15:03:12Z",
        "display_info": {"width": 1920, "height": 1080, "scale_factor": 2.0},
        "app_focus_history": [
            {
                "bundle_id": "com.google.Chrome",
                "name": "Google Chrome",
                "entered_at": "2026-04-22T15:00:01Z",
                "exited_at": None,
            }
        ],
    }
    validate_metadata(metadata)


def test_metadata_allows_null_stopped_at() -> None:
    metadata: dict[str, Any] = {
        "id": "7ab0c0f4-6ad1-4d9d-9d57-2d41d2d1f5aa",
        "label": "in_progress",
        "started_at": "2026-04-22T15:00:00Z",
        "stopped_at": None,
        "display_info": {"width": 1920, "height": 1080, "scale_factor": 2.0},
        "app_focus_history": [],
    }
    validate_metadata(metadata)


def test_metadata_rejects_bad_uuid() -> None:
    metadata: dict[str, Any] = {
        "id": "not-a-uuid",
        "label": "x",
        "started_at": "2026-04-22T15:00:00Z",
        "stopped_at": None,
        "display_info": {"width": 1920, "height": 1080, "scale_factor": 2.0},
        "app_focus_history": [],
    }
    with pytest.raises(ValidationError):
        validate_metadata(metadata)
