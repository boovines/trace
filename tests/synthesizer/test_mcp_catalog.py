"""Tests for ``synthesizer.mcp_catalog``.

Covers (a) catalog shape — every server has at least one function, every
function declares args, descriptions are non-empty; (b) :func:`validate_hint`
across the unknown-server / unknown-function / missing-arg / unknown-arg
matrix; and (c) :func:`format_for_prompt` produces stable markdown that
mentions every server.
"""

from __future__ import annotations

import pytest

from synthesizer.mcp_catalog import (
    CATALOG_VERSION,
    MCP_CATALOG,
    format_for_prompt,
    lookup_function,
    validate_hint,
)

# --- catalog shape ---------------------------------------------------------


def test_catalog_version_is_positive_int() -> None:
    assert isinstance(CATALOG_VERSION, int) and CATALOG_VERSION >= 1


def test_every_server_has_at_least_one_function() -> None:
    assert MCP_CATALOG, "catalog must declare at least one server"
    for server, fns in MCP_CATALOG.items():
        assert fns, f"server {server!r} has no functions"


def test_every_function_has_args_and_description() -> None:
    for server, fns in MCP_CATALOG.items():
        for fn_name, spec in fns.items():
            assert "args" in spec, f"{server}.{fn_name} missing 'args'"
            assert "description" in spec, f"{server}.{fn_name} missing 'description'"
            assert spec["description"].strip(), f"{server}.{fn_name} has empty description"


def test_arg_types_are_known_primitives() -> None:
    allowed = {"string", "integer", "boolean"}
    for server, fns in MCP_CATALOG.items():
        for fn_name, spec in fns.items():
            for arg_name, arg_type in {**spec.get("args", {}), **spec.get("optional", {})}.items():
                assert arg_type in allowed, (
                    f"{server}.{fn_name}.{arg_name}={arg_type!r} not one of {sorted(allowed)}"
                )


# --- lookup_function -------------------------------------------------------


def test_lookup_function_known() -> None:
    spec = lookup_function("gmail", "create_draft")
    assert spec is not None
    assert "to" in spec["args"]


def test_lookup_function_unknown_server_returns_none() -> None:
    assert lookup_function("not_a_server", "anything") is None


def test_lookup_function_unknown_function_returns_none() -> None:
    assert lookup_function("gmail", "definitely_not_a_real_fn") is None


# --- validate_hint ---------------------------------------------------------


def test_validate_hint_happy_path() -> None:
    err = validate_hint(
        server="gmail",
        function="create_draft",
        arguments={"to": "{recipient}", "subject": "{subject}", "body": "{body}"},
    )
    assert err is None


def test_validate_hint_optional_args_accepted() -> None:
    err = validate_hint(
        server="gmail",
        function="create_draft",
        arguments={
            "to": "alice@example.com",
            "subject": "Hi",
            "body": "Hello",
            "cc": "bob@example.com",
        },
    )
    assert err is None


def test_validate_hint_unknown_server() -> None:
    err = validate_hint(
        server="not_a_server", function="x", arguments={}
    )
    assert err is not None
    assert "not_a_server" in err
    assert "Catalog servers" in err


def test_validate_hint_unknown_function_lists_alternatives() -> None:
    err = validate_hint(
        server="gmail", function="not_a_function", arguments={}
    )
    assert err is not None
    assert "not_a_function" in err
    assert "Available functions" in err
    # Spot-check that a real function name appears in the error
    assert "create_draft" in err


def test_validate_hint_missing_required_arg() -> None:
    err = validate_hint(
        server="gmail", function="create_draft", arguments={"to": "x@y"}
    )
    assert err is not None
    assert "missing required argument" in err
    assert "subject" in err and "body" in err


def test_validate_hint_unknown_arg() -> None:
    err = validate_hint(
        server="gmail",
        function="create_draft",
        arguments={
            "to": "x@y",
            "subject": "s",
            "body": "b",
            "definitely_not_an_arg": "oops",
        },
    )
    assert err is not None
    assert "unknown argument" in err
    assert "definitely_not_an_arg" in err


# --- format_for_prompt -----------------------------------------------------


def test_format_for_prompt_mentions_every_server() -> None:
    rendered = format_for_prompt()
    for server in MCP_CATALOG:
        assert f"### {server}" in rendered, f"prompt is missing {server}"


def test_format_for_prompt_includes_function_signatures() -> None:
    rendered = format_for_prompt()
    # Spot-check: a Gmail function with a known arg should show up with the arg name.
    assert "create_draft" in rendered
    assert "to: string" in rendered  # required-arg formatting
    assert "cc?: string" in rendered  # optional-arg formatting


def test_format_for_prompt_is_deterministic() -> None:
    a = format_for_prompt()
    b = format_for_prompt()
    assert a == b


@pytest.mark.parametrize(
    "tier_function",
    [
        ("gmail", "search_threads"),
        ("gmail", "send_draft"),
        ("google_calendar", "create_event"),
        ("slack", "post_message"),
        ("notion", "search"),
    ],
)
def test_known_workflow_targets_have_catalog_entries(tier_function: tuple[str, str]) -> None:
    """Sanity: the five reference workflows should have at least one
    plausible MCP function each. Spot-check the obvious mappings."""
    server, fn = tier_function
    assert lookup_function(server, fn) is not None, (
        f"catalog missing {server}.{fn} — workflows that need it would have to "
        "fall back to computer-use unnecessarily"
    )
