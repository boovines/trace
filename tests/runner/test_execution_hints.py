"""Tests for ``runner.execution_hints``.

Covers:

* :class:`CapabilityRegistry` shape — defaults, ``supports`` matrix
  across all three tiers, MCP-server filtering.
* :func:`extract_step_hints` — ``meta.steps`` absent / empty / matched /
  malformed.
* :func:`pick_hint` — happy MCP path, MCP server unavailable falls
  through, all candidates unsupported emits the synthetic computer_use
  fallback, computer_use disabled returns ``chosen=None``.
* The :class:`HintDecision` ``fell_back`` flag is correctly set.

These cover Step 2's logging-only resolver. Step 3 (real MCP dispatch)
and Step 4 (Playwright DOM) will add execution-side tests on top.
"""

from __future__ import annotations

from typing import Any

import pytest

from runner.browser_dom_probe import BrowserDOMCapability
from runner.execution_hints import (
    CapabilityRegistry,
    HintDecision,
    Tier,
    default_capability_registry,
    extract_step_hints,
    iter_step_numbers,
    pick_hint,
)

_BROWSER_DOM_AVAILABLE = BrowserDOMCapability(
    cdp_endpoint=None, executable_path="/fake/chromium"
)

# --- CapabilityRegistry ---------------------------------------------------


def test_default_registry_only_allows_computer_use() -> None:
    reg = default_capability_registry()
    assert reg.computer_use is True
    assert reg.browser_dom is False
    assert reg.mcp_servers == frozenset()


def test_supports_mcp_only_when_server_in_set() -> None:
    reg = CapabilityRegistry(mcp_servers=frozenset({"gmail"}))
    assert reg.supports({"tier": "mcp", "mcp_server": "gmail"}) is True
    assert reg.supports({"tier": "mcp", "mcp_server": "slack"}) is False
    # missing mcp_server → unsupported
    assert reg.supports({"tier": "mcp"}) is False


def test_supports_browser_dom_flag() -> None:
    on = CapabilityRegistry(browser_dom_capability=_BROWSER_DOM_AVAILABLE)
    off = CapabilityRegistry(browser_dom_capability=None)
    hint = {"tier": "browser_dom", "selector": "button.send", "action": "click"}
    assert on.supports(hint) is True
    assert off.supports(hint) is False
    assert on.browser_dom is True
    assert off.browser_dom is False


def test_supports_computer_use_flag() -> None:
    on = CapabilityRegistry(computer_use=True)
    off = CapabilityRegistry(computer_use=False)
    hint = {"tier": "computer_use", "summary": "click stuff"}
    assert on.supports(hint) is True
    assert off.supports(hint) is False


def test_supports_unknown_tier_returns_false() -> None:
    reg = CapabilityRegistry(computer_use=True)
    assert reg.supports({"tier": "magic_tier"}) is False


# --- extract_step_hints ---------------------------------------------------


def _meta_with_steps(steps: list[dict[str, Any]]) -> dict[str, Any]:
    return {"slug": "x", "step_count": 5, "steps": steps}


def test_extract_step_hints_no_steps_array_returns_none_pair() -> None:
    intent, hints = extract_step_hints({"slug": "x"}, step_number=1)
    assert intent is None
    assert hints == []


def test_extract_step_hints_returns_intent_and_hints() -> None:
    meta = _meta_with_steps(
        [
            {
                "number": 3,
                "intent": "send_email",
                "execution_hints": [{"tier": "computer_use", "summary": "x"}],
            }
        ]
    )
    intent, hints = extract_step_hints(meta, step_number=3)
    assert intent == "send_email"
    assert hints == [{"tier": "computer_use", "summary": "x"}]


def test_extract_step_hints_unmatched_step_returns_empty() -> None:
    meta = _meta_with_steps([{"number": 1, "intent": "open"}])
    intent, hints = extract_step_hints(meta, step_number=99)
    assert intent is None
    assert hints == []


def test_extract_step_hints_skips_malformed_entries() -> None:
    meta = {"steps": ["not a dict", {"number": 2, "intent": "good"}]}
    intent, hints = extract_step_hints(meta, step_number=2)
    assert intent == "good"
    assert hints == []


# --- pick_hint -----------------------------------------------------------


def test_pick_hint_chooses_first_supported_mcp() -> None:
    reg = CapabilityRegistry(
        mcp_servers=frozenset({"gmail"}),
        browser_dom_capability=_BROWSER_DOM_AVAILABLE,
    )
    meta = _meta_with_steps(
        [
            {
                "number": 1,
                "intent": "send_email",
                "execution_hints": [
                    {
                        "tier": "mcp",
                        "mcp_server": "gmail",
                        "function": "send_draft",
                        "arguments": {"draft_id": "x"},
                    },
                    {"tier": "browser_dom", "selector": "btn", "action": "click"},
                    {"tier": "computer_use", "summary": "fallback"},
                ],
            }
        ]
    )
    decision = pick_hint(step_number=1, meta=meta, registry=reg)
    assert decision.chosen_tier == Tier.MCP
    assert decision.intent == "send_email"
    assert decision.fell_back is False
    assert decision.chosen is not None
    assert decision.chosen["mcp_server"] == "gmail"
    assert len(decision.considered) == 3
    # First reason is empty (supported); the others should still be present.
    assert decision.unsupported_reasons[0] == ""


def test_pick_hint_falls_through_when_mcp_server_not_connected() -> None:
    reg = CapabilityRegistry(mcp_servers=frozenset(), browser_dom_capability=None)
    meta = _meta_with_steps(
        [
            {
                "number": 1,
                "execution_hints": [
                    {
                        "tier": "mcp",
                        "mcp_server": "gmail",
                        "function": "send_draft",
                        "arguments": {},
                    },
                    {"tier": "computer_use", "summary": "fallback"},
                ],
            }
        ]
    )
    decision = pick_hint(step_number=1, meta=meta, registry=reg)
    assert decision.chosen_tier == Tier.COMPUTER_USE
    assert decision.fell_back is True
    assert "not connected" in decision.unsupported_reasons[0]


def test_pick_hint_synthetic_fallback_when_no_hints_supported() -> None:
    reg = CapabilityRegistry(computer_use=True)
    meta = _meta_with_steps(
        [
            {
                "number": 1,
                "execution_hints": [
                    {
                        "tier": "mcp",
                        "mcp_server": "slack",
                        "function": "post_message",
                        "arguments": {},
                    },
                    {"tier": "browser_dom", "selector": "x", "action": "click"},
                ],
            }
        ]
    )
    decision = pick_hint(step_number=1, meta=meta, registry=reg)
    assert decision.chosen_tier == Tier.COMPUTER_USE
    assert decision.chosen is not None
    # The chosen hint is synthetic — not one of the originals.
    assert decision.chosen.get("synthetic") is True
    assert decision.fell_back is True
    # Both originals should appear in `considered` with their reasons.
    assert len(decision.considered) == 2
    assert all(r for r in decision.unsupported_reasons[:2])


def test_pick_hint_no_steps_metadata_falls_back_to_computer_use() -> None:
    reg = default_capability_registry()
    decision = pick_hint(step_number=1, meta={"slug": "x"}, registry=reg)
    assert decision.chosen_tier == Tier.COMPUTER_USE
    assert decision.chosen is not None
    assert decision.chosen.get("synthetic") is True
    assert decision.intent is None
    assert decision.considered == ()
    assert decision.fell_back is False  # no candidates were tried


def test_pick_hint_returns_none_when_computer_use_disabled() -> None:
    reg = CapabilityRegistry(computer_use=False)
    meta = _meta_with_steps(
        [
            {
                "number": 1,
                "execution_hints": [
                    {
                        "tier": "mcp",
                        "mcp_server": "gmail",
                        "function": "send_draft",
                        "arguments": {},
                    }
                ],
            }
        ]
    )
    decision = pick_hint(step_number=1, meta=meta, registry=reg)
    assert decision.chosen is None
    assert decision.chosen_tier is None


@pytest.mark.parametrize(
    "registry,expected_tier",
    [
        (
            CapabilityRegistry(mcp_servers=frozenset({"gmail"})),
            Tier.MCP,
        ),
        (
            CapabilityRegistry(
                mcp_servers=frozenset(),
                browser_dom_capability=_BROWSER_DOM_AVAILABLE,
            ),
            Tier.BROWSER_DOM,
        ),
        (
            CapabilityRegistry(),
            Tier.COMPUTER_USE,
        ),
    ],
)
def test_pick_hint_tier_priority_order(
    registry: CapabilityRegistry, expected_tier: Tier
) -> None:
    """A workflow with all three tiers should pick the highest one available."""
    meta = _meta_with_steps(
        [
            {
                "number": 1,
                "execution_hints": [
                    {
                        "tier": "mcp",
                        "mcp_server": "gmail",
                        "function": "send_draft",
                        "arguments": {"draft_id": "x"},
                    },
                    {"tier": "browser_dom", "selector": "btn", "action": "click"},
                    {"tier": "computer_use", "summary": "fallback"},
                ],
            }
        ]
    )
    decision = pick_hint(step_number=1, meta=meta, registry=registry)
    assert decision.chosen_tier == expected_tier


# --- iter_step_numbers ----------------------------------------------------


def test_iter_step_numbers_yields_in_order() -> None:
    meta = _meta_with_steps(
        [
            {"number": 5},
            {"number": 2},
            {"number": 7},
        ]
    )
    assert list(iter_step_numbers(meta)) == [5, 2, 7]


def test_iter_step_numbers_skips_invalid_entries() -> None:
    meta = {
        "steps": [
            "garbage",
            {"number": "not an int"},
            {"number": 3},
            {"missing_number": True},
        ]
    }
    assert list(iter_step_numbers(meta)) == [3]


# --- HintDecision integration --------------------------------------------


def test_hint_decision_fell_back_true_when_first_choice_unsupported() -> None:
    reg = CapabilityRegistry()  # no MCP, no browser_dom
    decision = HintDecision(
        step_number=1,
        intent=None,
        chosen={"tier": "computer_use", "summary": "x", "synthetic": True},
        chosen_tier=Tier.COMPUTER_USE,
        considered=(
            {"tier": "mcp", "mcp_server": "gmail", "function": "x"},
        ),
        unsupported_reasons=("server gmail not connected",),
    )
    assert decision.fell_back is True
    _ = reg  # suppress unused
