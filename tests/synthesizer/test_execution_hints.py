"""Tests for the ``meta.steps[].execution_hints`` extension.

Covers:

* ``skill-meta.schema.json`` accepts a meta with ``steps[]`` populated and
  rejects malformed entries.
* ``validate_meta_against_markdown`` cross-checks step numbers and
  validates each hint against :mod:`synthesizer.mcp_catalog`.
* The MCP-tier hint shape is enforced (server + function + arguments
  required and present in the catalog); browser_dom and computer_use
  hints have their own minimums.
* Backward compat: a meta with no ``steps[]`` (the original shape) still
  validates and cross-checks.
"""

from __future__ import annotations

from typing import Any

import pytest
from jsonschema import ValidationError as SchemaValidationError

from synthesizer.schema import (
    ValidationError,
    validate_meta,
    validate_meta_against_markdown,
)

# --- helpers ---------------------------------------------------------------


_BASE_MD = """\
# Reply to email

Reply to the most recent unread email.

## Parameters

- `recipient` (string, required): Address of the reply target.

## Preconditions

- Gmail is open in Chrome.

## Steps

1. Open Gmail.
2. Click the email from {recipient}.
3. ⚠️ [DESTRUCTIVE] Click the Send button.

## Expected outcome

The email is sent.
"""


def _base_meta(*, with_steps: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "slug": "reply_to_email",
        "name": "Reply to email",
        "trajectory_id": "11111111-1111-4111-8111-111111111111",
        "created_at": "2026-04-01T10:00:00+00:00",
        "parameters": [
            {"name": "recipient", "type": "string", "required": True},
        ],
        "destructive_steps": [3],
        "preconditions": ["Gmail is open in Chrome."],
        "step_count": 3,
    }
    if with_steps is not None:
        meta["steps"] = with_steps
    return meta


# --- backward compat: no steps[] still works -------------------------------


def test_meta_without_steps_array_validates() -> None:
    meta = _base_meta()
    validate_meta(meta)
    validate_meta_against_markdown(meta, _BASE_MD)


# --- happy path: full hint chain ------------------------------------------


def test_meta_with_full_hint_chain_validates() -> None:
    meta = _base_meta(
        with_steps=[
            {
                "number": 3,
                "intent": "send_email",
                "execution_hints": [
                    {
                        "tier": "mcp",
                        "mcp_server": "gmail",
                        "function": "reply_to_thread",
                        "arguments": {
                            "thread_id": "{thread_id}",
                            "body": "{body}",
                            "send": True,
                        },
                    },
                    {
                        "tier": "browser_dom",
                        "url_pattern": "https://mail.google.com/",
                        "selector": "div[role='button'][data-tooltip*='Send']",
                        "action": "click",
                    },
                    {
                        "tier": "computer_use",
                        "summary": "Click the Send button at the bottom of the reply pane.",
                    },
                ],
            },
        ]
    )
    validate_meta(meta)
    validate_meta_against_markdown(meta, _BASE_MD)


# --- per-step number bounds -----------------------------------------------


def test_step_number_above_step_count_is_rejected() -> None:
    meta = _base_meta(
        with_steps=[
            {"number": 99, "intent": "x"},
        ]
    )
    validate_meta(meta)  # JSON schema doesn't know step_count vs steps[]
    with pytest.raises(ValidationError, match="outside the valid range"):
        validate_meta_against_markdown(meta, _BASE_MD)


def test_duplicate_step_numbers_rejected() -> None:
    meta = _base_meta(
        with_steps=[
            {"number": 1, "intent": "open"},
            {"number": 1, "intent": "still_one"},
        ]
    )
    validate_meta(meta)
    with pytest.raises(ValidationError, match="duplicate"):
        validate_meta_against_markdown(meta, _BASE_MD)


# --- mcp tier validation ---------------------------------------------------


def test_mcp_hint_with_unknown_server_rejected() -> None:
    meta = _base_meta(
        with_steps=[
            {
                "number": 3,
                "execution_hints": [
                    {
                        "tier": "mcp",
                        "mcp_server": "ghost_service",
                        "function": "do_thing",
                        "arguments": {},
                    }
                ],
            }
        ]
    )
    validate_meta(meta)
    with pytest.raises(ValidationError, match="unknown MCP server"):
        validate_meta_against_markdown(meta, _BASE_MD)


def test_mcp_hint_with_unknown_function_rejected() -> None:
    meta = _base_meta(
        with_steps=[
            {
                "number": 3,
                "execution_hints": [
                    {
                        "tier": "mcp",
                        "mcp_server": "gmail",
                        "function": "telepathically_send",
                        "arguments": {},
                    }
                ],
            }
        ]
    )
    validate_meta(meta)
    with pytest.raises(ValidationError, match="no function"):
        validate_meta_against_markdown(meta, _BASE_MD)


def test_mcp_hint_missing_required_arg_rejected() -> None:
    meta = _base_meta(
        with_steps=[
            {
                "number": 3,
                "execution_hints": [
                    {
                        "tier": "mcp",
                        "mcp_server": "gmail",
                        "function": "create_draft",
                        # missing subject + body
                        "arguments": {"to": "alice@example.com"},
                    }
                ],
            }
        ]
    )
    validate_meta(meta)
    with pytest.raises(ValidationError, match="missing required argument"):
        validate_meta_against_markdown(meta, _BASE_MD)


def test_mcp_hint_with_extra_arg_rejected() -> None:
    meta = _base_meta(
        with_steps=[
            {
                "number": 3,
                "execution_hints": [
                    {
                        "tier": "mcp",
                        "mcp_server": "gmail",
                        "function": "create_draft",
                        "arguments": {
                            "to": "x@y",
                            "subject": "s",
                            "body": "b",
                            "made_up_arg": "nope",
                        },
                    }
                ],
            }
        ]
    )
    validate_meta(meta)
    with pytest.raises(ValidationError, match="unknown argument"):
        validate_meta_against_markdown(meta, _BASE_MD)


# --- browser_dom + computer_use tiers --------------------------------------


def test_browser_dom_hint_missing_action_rejected() -> None:
    meta = _base_meta(
        with_steps=[
            {
                "number": 3,
                "execution_hints": [
                    {"tier": "browser_dom", "selector": "button.send"},
                ],
            }
        ]
    )
    with pytest.raises(SchemaValidationError):
        # JSON schema enum check is enough — `action` isn't strictly required
        # at schema level, so this case actually passes the JSON-schema gate
        # but the cross-check also enforces it.
        validate_meta(meta)
        validate_meta_against_markdown(meta, _BASE_MD)


def test_browser_dom_hint_type_action_requires_value() -> None:
    meta = _base_meta(
        with_steps=[
            {
                "number": 2,
                "execution_hints": [
                    {
                        "tier": "browser_dom",
                        "selector": "input[name='to']",
                        "action": "type",
                        # no `value` field
                    }
                ],
            }
        ]
    )
    validate_meta(meta)
    with pytest.raises(ValidationError, match="requires a 'value' field"):
        validate_meta_against_markdown(meta, _BASE_MD)


def test_computer_use_hint_missing_summary_rejected() -> None:
    meta = _base_meta(
        with_steps=[
            {
                "number": 1,
                "execution_hints": [
                    {"tier": "computer_use"},  # no summary
                ],
            }
        ]
    )
    validate_meta(meta)
    with pytest.raises(ValidationError, match="missing 'summary'"):
        validate_meta_against_markdown(meta, _BASE_MD)


def test_unknown_tier_rejected_by_schema() -> None:
    meta = _base_meta(
        with_steps=[
            {
                "number": 1,
                "execution_hints": [
                    {"tier": "magic_tier", "summary": "wave hands"}
                ],
            }
        ]
    )
    with pytest.raises(SchemaValidationError):
        validate_meta(meta)
