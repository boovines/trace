"""Safety flag plumbing for the Trace runner.

The runner can drive real mouse/keyboard input via CGEventPost (macOS). That is
a foot-gun for Ralph iterations and automated tests, so every live-mode code
path MUST first route through ``require_live_mode()``. The gate is the value of
the ``TRACE_ALLOW_LIVE`` environment variable — it must equal the string ``"1"``
exactly. Any other value (including ``"true"``, ``"yes"``, ``"0"``, empty, or
unset) is treated as live-mode-disabled.

This is the outermost layer of the three-layer destructive-action defense:

1. The skill's ⚠️ markers become per-step prompt instructions.
2. The harness inspects target AX-element labels against the shared keyword
   list and forces confirmation on match.
3. A hard per-run token budget and per-minute action rate limit prevent
   runaway loops.

``TRACE_ALLOW_LIVE`` gates whether ANY of the live adapters (LiveInputAdapter,
LiveScreenSource) can even be instantiated.
"""

from __future__ import annotations

import os

LIVE_MODE_ENV_VAR = "TRACE_ALLOW_LIVE"
_LIVE_MODE_ENABLED_VALUE = "1"


class LiveModeNotAllowed(RuntimeError):
    """Raised when a live-mode-only code path runs without TRACE_ALLOW_LIVE=1.

    Do NOT catch this to "fall through" to a dry-run. The caller is attempting
    to post real events; the only correct response is to refuse.
    """


def is_live_mode_allowed() -> bool:
    """Return True iff the environment grants permission to post real events.

    The check is strict string equality against ``"1"``. ``"true"``, ``"yes"``,
    ``"0"``, and any other value all return False. This prevents fat-finger
    configuration from silently enabling live mode.
    """
    return os.environ.get(LIVE_MODE_ENV_VAR) == _LIVE_MODE_ENABLED_VALUE


def require_live_mode() -> None:
    """Raise ``LiveModeNotAllowed`` unless ``TRACE_ALLOW_LIVE=1`` is set.

    Every class that posts CGEventPost, captures the live screen, or otherwise
    drives the real machine MUST call this in its ``__init__``. Belt-and-
    suspenders: the safety gate lives at instantiation time so that no code
    path accidentally constructs a live adapter in a dry-run context.
    """
    if not is_live_mode_allowed():
        raise LiveModeNotAllowed(
            f"Live mode is not allowed: environment variable {LIVE_MODE_ENV_VAR!r} "
            f"must equal '1' to enable real mouse/keyboard input or screen capture. "
            f"Ralph iterations MUST NOT set this flag; it is reserved for the "
            f"human-driven live-execution test phase."
        )
