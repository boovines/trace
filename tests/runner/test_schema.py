"""Tests for ``runner.schema`` — the RunMetadata schema and pydantic model.

Two-layer validation is under test:

* the JSON schema at ``contracts/run-metadata.schema.json`` (via
  ``validate_run_metadata``), which catches drift in on-disk artifacts; and
* the ``RunMetadata`` pydantic model, which catches drift in runtime code.

Both must agree. If one accepts something the other rejects, a run could
write a file the UI cannot read.
"""

from __future__ import annotations

import uuid
from typing import Any

import jsonschema
import pytest
from pydantic import ValidationError
from runner.schema import (
    TERMINAL_STATUSES,
    WAITING_STATUS,
    RunMetadata,
    load_run_metadata_schema,
    validate_run_metadata,
)

_UUID = "12345678-1234-4abc-8def-1234567890ab"
# Use the "Z" suffix — the canonical UTC form that pydantic emits via
# model_dump(mode="json"), so round-trip comparisons hold byte-for-byte.
_STARTED = "2026-04-22T15:33:38Z"
_ENDED = "2026-04-22T15:35:00Z"


def _minimal(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "run_id": _UUID,
        "skill_slug": "gmail_reply",
        "started_at": _STARTED,
        "ended_at": None,
        "status": "running",
        "mode": "dry_run",
    }
    base.update(overrides)
    return base


VALID_EXAMPLES: list[dict[str, Any]] = [
    _minimal(),
    _minimal(status="pending", mode="execute"),
    _minimal(
        status="succeeded",
        ended_at=_ENDED,
        total_cost_usd=0.042,
        input_tokens_total=1200,
        output_tokens_total=300,
        confirmation_count=2,
        destructive_actions_executed=[3, 7],
        parameters={"recipient": "alex@example.com", "template": "weekly_update"},
    ),
    _minimal(
        status="awaiting_confirmation",
        final_step_reached=4,
        abort_reason=None,
    ),
    _minimal(
        status="aborted",
        ended_at=_ENDED,
        abort_reason="user pressed kill switch",
        error_message=None,
    ),
    _minimal(
        status="budget_exceeded",
        ended_at=_ENDED,
        total_cost_usd=5.0,
        error_message="per-run cost cap reached",
    ),
]


@pytest.mark.parametrize("example", VALID_EXAMPLES)
def test_valid_metadata_passes_schema(example: dict[str, Any]) -> None:
    validate_run_metadata(example)


@pytest.mark.parametrize("example", VALID_EXAMPLES)
def test_valid_metadata_passes_model(example: dict[str, Any]) -> None:
    model = RunMetadata.from_dict(example)
    assert str(model.run_id) == _UUID


MALFORMED_EXAMPLES: list[tuple[str, dict[str, Any]]] = [
    ("bad_status", _minimal(status="totally_fine")),
    ("bad_mode", _minimal(mode="LIVE")),
    ("bad_uuid", _minimal(run_id="not-a-uuid")),
    ("unknown_field", _minimal(cost_usd=0.01)),
    ("missing_run_id", {k: v for k, v in _minimal().items() if k != "run_id"}),
    ("missing_skill_slug", {k: v for k, v in _minimal().items() if k != "skill_slug"}),
    ("missing_mode", {k: v for k, v in _minimal().items() if k != "mode"}),
    ("wrong_type_started_at", _minimal(started_at={"not": "a date"})),
    ("wrong_type_final_step", _minimal(final_step_reached="four")),
    ("wrong_type_destructive_list", _minimal(destructive_actions_executed=[1, "two"])),
    ("negative_tokens", _minimal(input_tokens_total=-5)),
    ("parameters_non_string_value", _minimal(parameters={"recipient": 42})),
]


@pytest.mark.parametrize("label,example", MALFORMED_EXAMPLES)
def test_malformed_fails_schema(label: str, example: dict[str, Any]) -> None:
    with pytest.raises(jsonschema.ValidationError):
        validate_run_metadata(example)


@pytest.mark.parametrize("label,example", MALFORMED_EXAMPLES)
def test_malformed_fails_model(label: str, example: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        RunMetadata.from_dict(example)


@pytest.mark.parametrize("example", VALID_EXAMPLES)
def test_round_trip_identity(example: dict[str, Any]) -> None:
    model = RunMetadata.from_dict(example)
    round_tripped = model.to_dict()
    assert round_tripped == example


def test_to_dict_drops_unset_optional_fields() -> None:
    model = RunMetadata.from_dict(_minimal())
    out = model.to_dict()
    for optional_key in (
        "parameters",
        "final_step_reached",
        "total_cost_usd",
        "abort_reason",
        "input_tokens_total",
        "output_tokens_total",
        "error_message",
        "confirmation_count",
        "destructive_actions_executed",
    ):
        assert optional_key not in out, f"unset field {optional_key} leaked into to_dict()"


@pytest.mark.parametrize(
    "status,expected_terminal",
    [
        ("pending", False),
        ("running", False),
        ("awaiting_confirmation", False),
        ("succeeded", True),
        ("failed", True),
        ("aborted", True),
        ("budget_exceeded", True),
        ("rate_limited", True),
    ],
)
def test_is_terminal(status: str, expected_terminal: bool) -> None:
    model = RunMetadata.from_dict(_minimal(status=status))
    assert model.is_terminal() is expected_terminal


@pytest.mark.parametrize(
    "status,expected_waiting",
    [
        ("pending", False),
        ("running", False),
        ("awaiting_confirmation", True),
        ("succeeded", False),
        ("failed", False),
        ("aborted", False),
        ("budget_exceeded", False),
        ("rate_limited", False),
    ],
)
def test_is_waiting(status: str, expected_waiting: bool) -> None:
    model = RunMetadata.from_dict(_minimal(status=status))
    assert model.is_waiting() is expected_waiting


def test_terminal_and_waiting_constants_match_enum() -> None:
    schema = load_run_metadata_schema()
    status_values = set(schema["properties"]["status"]["enum"])
    # Every terminal status is in the schema enum.
    assert status_values >= TERMINAL_STATUSES
    # The waiting sentinel is in the schema enum.
    assert WAITING_STATUS in status_values
    # Terminal and waiting are disjoint.
    assert WAITING_STATUS not in TERMINAL_STATUSES


def test_load_schema_has_required_fields() -> None:
    schema = load_run_metadata_schema()
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {
        "run_id",
        "skill_slug",
        "started_at",
        "ended_at",
        "status",
        "mode",
    }


def test_run_id_accepts_uuid_object() -> None:
    data = _minimal(run_id=uuid.UUID(_UUID))
    model = RunMetadata.from_dict(data)
    assert str(model.run_id) == _UUID
