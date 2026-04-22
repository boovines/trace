"""Tests for ``synthesizer.schema`` — schema validation + markdown cross-check.

Covers the S-002 acceptance criteria:

* 5+ valid meta dicts pass validation.
* 10+ malformed meta dicts fail with specific, useful messages.
* Cross-check detects markdown/meta drift on destructive steps.
* Cross-check detects markdown/meta drift on parameters.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest
from jsonschema import ValidationError
from synthesizer.schema import (
    load_meta_schema,
    validate_meta,
    validate_meta_against_markdown,
)

# ---------- Fixtures / helpers -------------------------------------------------


def _base_meta() -> dict[str, Any]:
    """A minimal, valid meta dict — tests mutate copies of this."""
    return {
        "slug": "gmail_reply",
        "name": "Reply to a Gmail thread",
        "trajectory_id": "0c3f1f4e-1d2a-4c9d-8f9a-1a2b3c4d5e6f",
        "created_at": "2026-04-22T10:00:00Z",
        "parameters": [
            {"name": "recipient", "type": "string", "required": True},
            {
                "name": "body",
                "type": "string",
                "required": False,
                "default": "Thanks!",
            },
        ],
        "destructive_steps": [4],
        "preconditions": ["Chrome is running", "Gmail tab is open"],
        "step_count": 4,
    }


def _base_markdown() -> str:
    return (
        "# Reply to a Gmail thread\n"
        "\n"
        "Send a template reply to the most recent unread message from a "
        "given recipient.\n"
        "\n"
        "## Parameters\n"
        "\n"
        "- recipient (string, required)\n"
        "- body (string, optional, default: Thanks!)\n"
        "\n"
        "## Preconditions\n"
        "\n"
        "- Chrome is running\n"
        "- Gmail tab is open\n"
        "\n"
        "## Steps\n"
        "\n"
        "1. Focus the Gmail tab.\n"
        "2. Open the most recent unread thread from {recipient}.\n"
        "3. Type {body} into the reply field.\n"
        "4. ⚠️ [DESTRUCTIVE] Click the Send button.\n"
        "\n"
        "## Expected outcome\n"
        "\n"
        "The reply is sent and the thread is marked read.\n"
    )


# ---------- Schema loading -----------------------------------------------------


def test_schema_loads_and_is_draft_2020_12() -> None:
    schema = load_meta_schema()
    assert schema["$schema"].endswith("/draft/2020-12/schema")
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False


# ---------- Valid meta dicts (5+) ---------------------------------------------


def test_valid_minimal_meta() -> None:
    meta = _base_meta()
    # strip parameters + destructive_steps to the minimum; still valid
    meta["parameters"] = []
    meta["destructive_steps"] = []
    meta["preconditions"] = []
    meta["step_count"] = 1
    validate_meta(meta)


def test_valid_base_meta() -> None:
    validate_meta(_base_meta())


def test_valid_many_parameters() -> None:
    meta = _base_meta()
    meta["parameters"] = [
        {"name": f"p{i}", "type": "string", "required": False, "default": None}
        for i in range(10)
    ]
    meta["step_count"] = 3
    validate_meta(meta)


def test_valid_integer_and_boolean_params() -> None:
    meta = _base_meta()
    meta["parameters"] = [
        {"name": "count", "type": "integer", "required": True},
        {"name": "dry_run", "type": "boolean", "required": False, "default": True},
    ]
    validate_meta(meta)


def test_valid_default_null() -> None:
    meta = _base_meta()
    meta["parameters"] = [
        {"name": "note", "type": "string", "required": False, "default": None}
    ]
    validate_meta(meta)


def test_valid_long_slug_at_max_length() -> None:
    meta = _base_meta()
    meta["slug"] = "a" + "b" * 39  # 40 chars total, matches upper bound
    validate_meta(meta)


def test_valid_many_destructive_steps() -> None:
    meta = _base_meta()
    meta["destructive_steps"] = [1, 2, 3, 4]
    meta["step_count"] = 4
    validate_meta(meta)


# ---------- Malformed meta dicts (10+) ----------------------------------------


def test_malformed_slug_uppercase() -> None:
    meta = _base_meta()
    meta["slug"] = "Gmail_Reply"
    with pytest.raises(ValidationError) as exc:
        validate_meta(meta)
    assert "slug" in str(exc.value).lower()


def test_malformed_slug_starts_with_digit() -> None:
    meta = _base_meta()
    meta["slug"] = "1gmail"
    with pytest.raises(ValidationError):
        validate_meta(meta)


def test_malformed_slug_too_short() -> None:
    meta = _base_meta()
    meta["slug"] = "ab"  # 2 chars, min is 3
    with pytest.raises(ValidationError):
        validate_meta(meta)


def test_malformed_slug_too_long() -> None:
    meta = _base_meta()
    meta["slug"] = "a" + "b" * 40  # 41 chars, max is 40
    with pytest.raises(ValidationError):
        validate_meta(meta)


def test_malformed_slug_with_hyphen() -> None:
    meta = _base_meta()
    meta["slug"] = "gmail-reply"
    with pytest.raises(ValidationError):
        validate_meta(meta)


def test_malformed_missing_required_slug() -> None:
    meta = _base_meta()
    del meta["slug"]
    with pytest.raises(ValidationError) as exc:
        validate_meta(meta)
    assert "slug" in str(exc.value)


def test_malformed_missing_required_step_count() -> None:
    meta = _base_meta()
    del meta["step_count"]
    with pytest.raises(ValidationError):
        validate_meta(meta)


def test_malformed_wrong_type_step_count_as_string() -> None:
    meta = _base_meta()
    meta["step_count"] = "4"
    with pytest.raises(ValidationError):
        validate_meta(meta)


def test_malformed_extra_top_level_key_rejected() -> None:
    """additionalProperties: false catches accidental typos like stepp_count."""
    meta = _base_meta()
    meta["stepp_count"] = 4
    with pytest.raises(ValidationError) as exc:
        validate_meta(meta)
    assert "additional" in str(exc.value).lower() or "stepp_count" in str(exc.value)


def test_malformed_destructive_step_zero() -> None:
    meta = _base_meta()
    meta["destructive_steps"] = [0]
    with pytest.raises(ValidationError):
        validate_meta(meta)


def test_malformed_destructive_step_negative() -> None:
    meta = _base_meta()
    meta["destructive_steps"] = [-1]
    with pytest.raises(ValidationError):
        validate_meta(meta)


def test_malformed_parameter_name_with_hyphen() -> None:
    meta = _base_meta()
    meta["parameters"] = [{"name": "recipient-email", "type": "string", "required": True}]
    with pytest.raises(ValidationError):
        validate_meta(meta)


def test_malformed_parameter_type_not_in_enum() -> None:
    meta = _base_meta()
    meta["parameters"] = [{"name": "amount", "type": "float", "required": True}]
    with pytest.raises(ValidationError):
        validate_meta(meta)


def test_malformed_parameter_extra_key() -> None:
    meta = _base_meta()
    meta["parameters"] = [
        {"name": "x", "type": "string", "required": True, "bogus": 1}
    ]
    with pytest.raises(ValidationError):
        validate_meta(meta)


def test_malformed_parameter_missing_required_field() -> None:
    meta = _base_meta()
    meta["parameters"] = [{"name": "x", "type": "string"}]  # missing "required"
    with pytest.raises(ValidationError):
        validate_meta(meta)


def test_malformed_step_count_zero() -> None:
    meta = _base_meta()
    meta["step_count"] = 0
    with pytest.raises(ValidationError):
        validate_meta(meta)


def test_malformed_trajectory_id_not_uuid() -> None:
    meta = _base_meta()
    meta["trajectory_id"] = "not-a-uuid"
    with pytest.raises(ValidationError):
        validate_meta(meta)


def test_malformed_created_at_wrong_type() -> None:
    """created_at must be a string (the date-time *format* check requires the
    optional ``rfc3339-validator`` dep which we do not pull in for v1 — the
    type check alone catches the most common drift).
    """
    meta = _base_meta()
    meta["created_at"] = 1714000000  # epoch int, not string
    with pytest.raises(ValidationError):
        validate_meta(meta)


def test_malformed_name_too_long() -> None:
    meta = _base_meta()
    meta["name"] = "x" * 101
    with pytest.raises(ValidationError):
        validate_meta(meta)


def test_malformed_name_empty() -> None:
    meta = _base_meta()
    meta["name"] = ""
    with pytest.raises(ValidationError):
        validate_meta(meta)


def test_malformed_parameter_name_starts_with_digit() -> None:
    meta = _base_meta()
    meta["parameters"] = [{"name": "1x", "type": "string", "required": True}]
    with pytest.raises(ValidationError):
        validate_meta(meta)


# ---------- Cross-check: markdown <-> meta agreement --------------------------


def test_cross_check_happy_path() -> None:
    validate_meta_against_markdown(_base_meta(), _base_markdown())


def test_cross_check_catches_destructive_missing_from_meta() -> None:
    """Markdown flags step 4 as ⚠️ but meta.destructive_steps is empty."""
    meta = _base_meta()
    meta["destructive_steps"] = []
    with pytest.raises(ValidationError) as exc:
        validate_meta_against_markdown(meta, _base_markdown())
    msg = str(exc.value)
    assert "destructive" in msg.lower()
    assert "4" in msg


def test_cross_check_catches_destructive_missing_from_markdown() -> None:
    """meta says step 2 is destructive but markdown has no ⚠️ on step 2."""
    meta = _base_meta()
    meta["destructive_steps"] = [2, 4]
    with pytest.raises(ValidationError) as exc:
        validate_meta_against_markdown(meta, _base_markdown())
    assert "destructive" in str(exc.value).lower()


def test_cross_check_catches_parameter_in_markdown_not_meta() -> None:
    """Markdown uses {foo} but meta.parameters doesn't list foo."""
    md = _base_markdown().replace("{body}", "{foo}")
    with pytest.raises(ValidationError) as exc:
        validate_meta_against_markdown(_base_meta(), md)
    assert "foo" in str(exc.value)


def test_cross_check_catches_parameter_in_meta_not_markdown() -> None:
    """meta lists a parameter that is never referenced in markdown."""
    meta = _base_meta()
    meta["parameters"].append(
        {"name": "phantom", "type": "string", "required": False}
    )
    with pytest.raises(ValidationError) as exc:
        validate_meta_against_markdown(meta, _base_markdown())
    assert "phantom" in str(exc.value)


def test_cross_check_step_count_mismatch() -> None:
    meta = _base_meta()
    meta["step_count"] = 3  # markdown has 4
    with pytest.raises(ValidationError) as exc:
        validate_meta_against_markdown(meta, _base_markdown())
    assert "step_count" in str(exc.value)


def test_cross_check_step_numbering_out_of_order() -> None:
    md = _base_markdown().replace("3. Type", "5. Type")
    with pytest.raises(ValidationError):
        validate_meta_against_markdown(_base_meta(), md)


def test_cross_check_ignores_params_inside_fenced_code() -> None:
    """{example} inside a ``` block should not count as a parameter ref."""
    md = _base_markdown() + "\n\n## Notes\n\n```\nExample: {example}\n```\n"
    # Should still pass — example is not a real parameter ref.
    validate_meta_against_markdown(_base_meta(), md)


def test_cross_check_accepts_repeated_parameter_references() -> None:
    """A parameter used in multiple steps still counts as one entry."""
    md = _base_markdown().replace(
        "4. ⚠️ [DESTRUCTIVE] Click the Send button.",
        "4. ⚠️ [DESTRUCTIVE] Click Send to {recipient}.",
    )
    validate_meta_against_markdown(_base_meta(), md)


# ---------- Regression: ensure base fixtures are themselves valid -------------


def test_base_meta_is_self_consistent() -> None:
    meta = _base_meta()
    validate_meta(meta)
    validate_meta_against_markdown(meta, _base_markdown())


def test_deep_copy_independence() -> None:
    """Sanity: mutations in one test's base don't leak into another."""
    a = _base_meta()
    b = _base_meta()
    a["slug"] = "changed"
    assert b["slug"] == "gmail_reply"
    # And copy.deepcopy round-trips cleanly
    validate_meta(copy.deepcopy(b))
