"""Pre-action destructive keyword matcher — the harness-layer gate (X-015).

This is the **third** and strictest of the three destructive-action defenses:

1. Prompt layer: the skill's ``⚠️`` markers become per-step instructions that
   ask the model to emit ``<needs_confirmation step="N"/>`` before acting.
2. Parser layer: :mod:`runner.parser` drops any concurrent ``tool_use`` block
   when a ``<needs_confirmation …/>`` tag is present in the same turn.
3. *This layer:* before dispatching a click-class tool call we query the AX
   element at the target coordinates and force a confirmation pause if its
   label matches a destructive keyword — even if the skill didn't flag the
   step and even if Claude didn't emit ``<needs_confirmation>``.

Layers 1 and 2 trust the LLM. This layer does **not** — it inspects the actual
screen state at action time. If Claude hallucinates a click on the "Send"
button without a confirmation tag, this gate catches it.

Scope
-----
Only ``left_click``/``right_click``/``double_click``/``middle_click`` actions
are gated. ``type``, ``scroll``, ``mouse_move``, ``wait``, and ``screenshot``
are never gated (they cannot trigger destructive side effects without a prior
click on a destructive control).

The gate only fires when the AX element under the cursor is **actionable** —
``AXButton``, ``AXLink``, ``AXMenuItem``, or ``AXCheckBox``. A destructive
keyword on an ``AXStaticText`` element (e.g. a warning paragraph that mentions
"delete") is NOT enough to gate a click.

Dry-run mode is a no-op — ``apply_gate_to_tool_call(..., mode="dry_run")``
always returns :class:`AllowAction`. Dry-run clicks do not drive the live
input adapter, so there is nothing to gate; gating in dry-run would produce
false positives against trajectory screenshots.

AXResolver protocol
-------------------
The actual AX resolution is owned by the recorder module's ``ax_resolver``
(not yet shipped on ``feat/recorder``). Once that ships, its resolver will
conform to the :class:`AXResolver` Protocol declared here — no copy-paste of
resolution logic lives in the runner. For now we only depend on the protocol
shape, which is the contract both modules share.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Protocol

from runner.coords import ImageMapping, resized_pixels_to_points
from runner.destructive import matches_destructive_keyword
from runner.parser import ToolCallAction

ACTIONABLE_AX_ROLES: Final[frozenset[str]] = frozenset(
    {"AXButton", "AXLink", "AXMenuItem", "AXCheckBox"}
)

CLICK_ACTION_NAMES: Final[frozenset[str]] = frozenset(
    {"left_click", "right_click", "double_click", "middle_click"}
)

DRY_RUN_MODE: Final[str] = "dry_run"
EXECUTE_MODE: Final[str] = "execute"


@dataclass(frozen=True, slots=True)
class AXTarget:
    """Minimal shape the gate needs from an AX resolver.

    The recorder's forthcoming ``ax_resolver`` module will return a richer
    object; the gate only consumes ``role`` and ``label`` so we keep the
    protocol surface small.
    """

    role: str
    label: str


class AXResolver(Protocol):
    """Resolves the AX element at a display-point coordinate."""

    def resolve_at(self, x_pt: float, y_pt: float) -> AXTarget | None:
        """Return the AX target at ``(x_pt, y_pt)`` or ``None`` if unknown.

        ``None`` covers both "no AX element found" (e.g. clicking empty
        desktop) and "resolution timed out / permission denied" — the gate
        treats both as :class:`Unknown` and leaves the policy decision to
        the executor.
        """


@dataclass(frozen=True, slots=True)
class AllowAction:
    """Proceed with the tool call — no destructive element detected."""


@dataclass(frozen=True, slots=True)
class RequireConfirmation:
    """Force a confirmation pause before dispatching the tool call."""

    label: str
    reason: str


@dataclass(frozen=True, slots=True)
class Unknown:
    """AX resolution did not return a target — executor policy decides.

    The executor's default policy is to ALLOW (we do not over-block on an
    imperfect AX tree) but it MUST log the occurrence so we can spot patterns
    where AX routinely fails and revisit the gate.
    """


GateDecision = AllowAction | RequireConfirmation | Unknown


def inspect_click_target(
    x_pt: float, y_pt: float, ax_resolver: AXResolver
) -> GateDecision:
    """Ask ``ax_resolver`` what's under ``(x_pt, y_pt)`` and classify.

    Display-point coordinates — the caller is expected to have converted
    Claude's resized-pixel space via :func:`runner.coords.resized_pixels_to_points`.

    Decision tree
    -------------
    * Resolver raises or returns ``None`` → :class:`Unknown`.
    * Target role is not in :data:`ACTIONABLE_AX_ROLES` → :class:`AllowAction`.
    * Target label matches a destructive keyword → :class:`RequireConfirmation`.
    * Otherwise → :class:`AllowAction`.

    Never raises — any exception from the resolver is caught and demoted to
    :class:`Unknown`; the harness should degrade open rather than crash.
    """

    try:
        target = ax_resolver.resolve_at(x_pt, y_pt)
    except Exception:
        return Unknown()

    if target is None:
        return Unknown()

    if target.role not in ACTIONABLE_AX_ROLES:
        return AllowAction()

    if matches_destructive_keyword(target.label):
        return RequireConfirmation(
            label=target.label,
            reason=(
                f"AX target role={target.role} label={target.label!r} "
                "matches destructive keyword"
            ),
        )

    return AllowAction()


def apply_gate_to_tool_call(
    action: ToolCallAction,
    image_mapping: ImageMapping,
    ax_resolver: AXResolver,
    mode: str,
) -> GateDecision:
    """Route a parsed tool call through the harness-layer gate.

    * ``mode == "dry_run"`` → always :class:`AllowAction`.
    * Non-click action → :class:`AllowAction` (nothing to inspect).
    * Malformed/out-of-shape coordinate → :class:`AllowAction` (the
      dispatcher will produce a ``tool_result`` error on the same turn; the
      gate has no reason to force a confirmation on a bogus input).
    * Well-formed click → delegate to :func:`inspect_click_target` after
      mapping the resized-pixel coordinate to display points.

    The mapping-validation policy deliberately mirrors the dispatcher's
    tolerance: the gate is a SAFETY check, not a schema check, so we never
    up-convert a malformed tool_input into a confirmation pause.
    """

    if mode == DRY_RUN_MODE:
        return AllowAction()

    action_name = action.tool_input.get("action")
    if not isinstance(action_name, str) or action_name not in CLICK_ACTION_NAMES:
        return AllowAction()

    raw_coord = action.tool_input.get("coordinate")
    if not isinstance(raw_coord, list | tuple) or len(raw_coord) != 2:
        return AllowAction()
    x, y = raw_coord
    if isinstance(x, bool) or isinstance(y, bool):
        return AllowAction()
    if not isinstance(x, int | float) or not isinstance(y, int | float):
        return AllowAction()

    x_pt, y_pt = resized_pixels_to_points(float(x), float(y), image_mapping)
    return inspect_click_target(x_pt, y_pt, ax_resolver)


__all__ = [
    "ACTIONABLE_AX_ROLES",
    "CLICK_ACTION_NAMES",
    "DRY_RUN_MODE",
    "EXECUTE_MODE",
    "AXResolver",
    "AXTarget",
    "AllowAction",
    "GateDecision",
    "RequireConfirmation",
    "Unknown",
    "apply_gate_to_tool_call",
    "inspect_click_target",
]
