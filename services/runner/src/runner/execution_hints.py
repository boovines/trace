"""Tier resolution for ``skill.meta.json`` execution hints.

The synthesizer emits ``meta.steps[].execution_hints`` as an ordered list of
candidates per step (most-preferred first) drawn from three tiers:

* ``mcp`` — direct call into a published MCP server function.
* ``browser_dom`` — DOM action via Playwright against a known selector.
* ``computer_use`` — pixel-grounded fallback via the existing agent loop.

This module owns the **decision** of which tier the runner should use for
a given step, given a runtime :class:`CapabilityRegistry` describing what
the host actually has connected. The decision is recorded into the run's
event stream so users (and downstream tools) can inspect what happened.

Step 2 of the tiered-execution rollout is intentionally **logging-only**:
:func:`pick_hint` returns the chosen hint, but the executor still goes
through computer-use for the actual action. Step 3 wires real MCP
dispatch; Step 4 wires Playwright. Both will hang off the same seam.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from runner.browser_dom_probe import BrowserDOMCapability

__all__ = [
    "CapabilityRegistry",
    "HintDecision",
    "Tier",
    "default_capability_registry",
    "extract_step_hints",
    "pick_hint",
]


class Tier(StrEnum):
    """Execution tiers in synth's preference order (most preferred first)."""

    MCP = "mcp"
    BROWSER_DOM = "browser_dom"
    COMPUTER_USE = "computer_use"


@dataclass(frozen=True)
class CapabilityRegistry:
    """Snapshot of which execution tiers the host can currently use.

    The :attr:`mcp_servers` set names the MCP servers the runner has live
    connections to; tier-mcp hints whose ``mcp_server`` is not in this
    set are skipped. The optional :attr:`mcp_functions` map narrows that
    further: when populated, a tier-mcp hint is only supported if its
    ``function`` name is also in ``mcp_functions[server]``. The probe
    layer (:mod:`runner.mcp_client`) populates this map from the
    server's actual ``tools/list`` response so the synthesizer can't
    name a tool the live server doesn't actually expose.

    ``browser_dom_capability`` is the populated probe result when
    Playwright + Chromium are available (or a CDP endpoint is set);
    ``None`` means the tier is unsupported. Stored as a structured
    capability rather than a bool so the dispatcher (Step 4.1 commit 2)
    can read ``cdp_endpoint`` and ``executable_path`` without re-probing.
    ``computer_use`` is a simple flag — it has no per-target nuance,
    just a kill-switch for tests that want to force tier=None.
    """

    mcp_servers: frozenset[str] = field(default_factory=frozenset)
    mcp_functions: Mapping[str, frozenset[str]] = field(default_factory=dict)
    browser_dom_capability: BrowserDOMCapability | None = None
    computer_use: bool = True

    @property
    def browser_dom(self) -> bool:
        """Convenience flag: ``True`` when ``tier=browser_dom`` is dispatchable."""
        return self.browser_dom_capability is not None

    def supports(self, hint: Mapping[str, Any]) -> bool:
        """Return True when ``hint`` is dispatchable under this registry.

        Checks the tier-level capability, the named server (for MCP),
        and (when populated) the function name against
        ``mcp_functions[server]``. Argument validity is the catalog's
        job at synth time and the dispatcher's job at run time — this
        is the cheap pre-check the resolver uses to walk the candidate
        list.
        """
        tier = hint.get("tier")
        if tier == Tier.MCP.value:
            server = hint.get("mcp_server")
            if not isinstance(server, str) or server not in self.mcp_servers:
                return False
            # When the registry didn't probe function names (legacy /
            # default), accept any function on the server. When it did,
            # the function must be present.
            fn_set = self.mcp_functions.get(server)
            if fn_set is None or not fn_set:
                return True
            function = hint.get("function")
            return isinstance(function, str) and function in fn_set
        if tier == Tier.BROWSER_DOM.value:
            return self.browser_dom
        if tier == Tier.COMPUTER_USE.value:
            return self.computer_use
        # Unknown tier — pretend unsupported. The synthesizer's schema
        # validator should have caught this before the meta hit disk, but
        # be defensive at the runner boundary anyway.
        return False


def default_capability_registry() -> CapabilityRegistry:
    """The conservative default: only computer_use is online.

    Real MCP probing + browser detection are explicit follow-on work
    (Steps 3 and 4 of the tiered-execution rollout). Until then, every
    workflow runs through the existing pixel-grounded path; the new
    routing layer just *records* what other tier would have applied.
    """
    return CapabilityRegistry()


@dataclass(frozen=True)
class HintDecision:
    """The outcome of running :func:`pick_hint` on one step.

    Fields:

    * ``step_number`` — 1-based index, matches ``meta.steps[].number``.
    * ``intent`` — short verb-phrase from the synth (e.g. ``send_email``);
      ``None`` when the synth didn't supply one.
    * ``chosen`` — the winning hint dict (verbatim from meta), or
      ``None`` when the step had no hints at all (or every hint was
      unsupported and ``computer_use`` had no fallback).
    * ``chosen_tier`` — the chosen hint's tier, or ``None`` when
      ``chosen`` is ``None``.
    * ``considered`` — the full ordered list of candidates the resolver
      walked, including the unsupported ones.
    * ``unsupported_reasons`` — parallel list of reason strings for any
      candidate the resolver had to skip (non-empty entries align with
      ``considered`` indices).
    """

    step_number: int
    intent: str | None
    chosen: dict[str, Any] | None
    chosen_tier: Tier | None
    considered: tuple[dict[str, Any], ...]
    unsupported_reasons: tuple[str, ...]

    @property
    def fell_back(self) -> bool:
        """True when at least one preferred candidate was skipped."""
        return any(self.unsupported_reasons)


def _reason_unsupported(hint: Mapping[str, Any], registry: CapabilityRegistry) -> str:
    tier = hint.get("tier")
    if tier == Tier.MCP.value:
        server = hint.get("mcp_server", "<missing>")
        if server not in registry.mcp_servers:
            connected = ", ".join(sorted(registry.mcp_servers)) or "(none)"
            return (
                f"mcp server {server!r} not connected; live servers: {connected}"
            )
        # Server is healthy but the function isn't in the live tool list.
        function = hint.get("function", "<missing>")
        fn_set = registry.mcp_functions.get(server)
        if fn_set is not None and fn_set:
            available = ", ".join(sorted(fn_set)[:6])
            tail = "" if len(fn_set) <= 6 else f" (+{len(fn_set) - 6} more)"
            return (
                f"mcp {server}.{function} not in live tool list; "
                f"available: {available}{tail}"
            )
        return f"mcp {server}.{function} unsupported (no probe data)"
    if tier == Tier.BROWSER_DOM.value:
        return (
            "browser_dom tier unavailable; install chromium with "
            "`uv run playwright install chromium` or set TRACE_CDP_ENDPOINT"
        )
    if tier == Tier.COMPUTER_USE.value:
        return "computer_use tier disabled in this registry"
    return f"unknown tier {tier!r}"


def extract_step_hints(meta: Mapping[str, Any], step_number: int) -> tuple[
    str | None, list[dict[str, Any]]
]:
    """Pull the (intent, execution_hints) pair for a given step.

    Returns ``(None, [])`` when ``meta`` has no ``steps`` array, when no
    entry matches ``step_number``, or when the matching entry has no
    ``execution_hints``. The synthesizer treats ``steps`` as optional;
    absence means "fall back to computer-use using the SKILL.md prose."
    """
    steps_raw = meta.get("steps") or []
    for entry in steps_raw:
        if not isinstance(entry, dict):
            continue
        if entry.get("number") != step_number:
            continue
        intent = entry.get("intent")
        hints_raw = entry.get("execution_hints") or []
        hints: list[dict[str, Any]] = [h for h in hints_raw if isinstance(h, dict)]
        return (intent if isinstance(intent, str) else None, hints)
    return (None, [])


def pick_hint(
    *,
    step_number: int,
    meta: Mapping[str, Any],
    registry: CapabilityRegistry,
) -> HintDecision:
    """Resolve which tier should execute a given step under the registry.

    Walks ``meta.steps[number==step_number].execution_hints`` in order
    and returns the first hint :meth:`CapabilityRegistry.supports`
    accepts. When every hint is unsupported the decision falls back to a
    synthetic ``computer_use`` hint *if* the registry allows that tier;
    otherwise returns ``chosen=None`` so the executor knows the step
    can't be dispatched at any tier.

    The full ordered candidate list is preserved on the result so the
    executor can log it for inspection.
    """
    intent, hints = extract_step_hints(meta, step_number)

    chosen: dict[str, Any] | None = None
    chosen_tier: Tier | None = None
    reasons: list[str] = []

    for hint in hints:
        if registry.supports(hint):
            chosen = dict(hint)
            chosen_tier = Tier(hint["tier"])
            reasons.append("")  # supported — empty reason marker
            break
        reasons.append(_reason_unsupported(hint, registry))

    # Synthetic fallback: if nothing in the hint chain is supported but
    # the registry still allows computer_use, dispatch via the existing
    # pixel-grounded path. The runner already drives that path from the
    # SKILL.md prose; the synthetic hint simply makes the choice explicit
    # in the run log.
    if chosen is None and registry.computer_use:
        chosen = {
            "tier": Tier.COMPUTER_USE.value,
            "summary": (
                f"Fall back to computer-use for step {step_number} "
                f"(no supported hint{'; ' + str(len(hints)) + ' tried' if hints else ''})."
            ),
            "synthetic": True,
        }
        chosen_tier = Tier.COMPUTER_USE

    # ``considered`` should reflect every candidate we walked, even the
    # unsupported ones, so the run log shows the full preference chain.
    return HintDecision(
        step_number=step_number,
        intent=intent,
        chosen=chosen,
        chosen_tier=chosen_tier,
        considered=tuple(dict(h) for h in hints),
        unsupported_reasons=tuple(reasons),
    )


def iter_step_numbers(meta: Mapping[str, Any]) -> Iterable[int]:
    """Yield every ``number`` declared in ``meta.steps[]``, in input order."""
    for entry in meta.get("steps") or []:
        if isinstance(entry, dict) and isinstance(entry.get("number"), int):
            yield entry["number"]
