"""Decide when the recorder should capture a keyframe screenshot.

The recorder does not screenshot every event — that would balloon disk
usage and slow capture.  Instead, keyframes are captured on specific
triggers:

* **periodic** — at least :data:`PERIODIC_INTERVAL_SECONDS` of wall-clock
  time have passed since the last keyframe, and the session emitted a
  periodic tick.
* **app_switch** — a ``NSWorkspaceDidActivateApplicationNotification``
  fired and the focused app changed.
* **pre_click** — a mouse-down is imminent; the caller captures *before*
  dispatching the click event so the frame reflects what the user saw at
  the moment they clicked.
* **post_click** — :data:`POST_CLICK_DELAY_SECONDS` after the click
  completed; captures the result of the click (a dialog opening, a
  selection changing, a page navigating).

:class:`KeyframePolicy` is intentionally stateless.  The caller (the
session orchestrator in R-010) tracks wall-clock time between keyframes
and passes ``seconds_since_last_keyframe`` on each ``tick``.  A stateless
policy is easier to test, thread-safe by construction, and composes
cleanly with any timer implementation (``threading.Timer``, ``asyncio``,
tests driving time manually).
"""

from __future__ import annotations

from typing import Final

__all__ = [
    "KEYFRAME_REASONS",
    "PERIODIC_INTERVAL_SECONDS",
    "POST_CLICK_DELAY_SECONDS",
    "KeyframePolicy",
]

#: Minimum wall-clock gap between periodic keyframes.  Matches the PRD
#: guidance "every 5 seconds while recording".
PERIODIC_INTERVAL_SECONDS: Final[float] = 5.0

#: Delay between a click event and the post-click keyframe.  100 ms is a
#: compromise between "long enough to capture the UI reaction" and "short
#: enough that the user has not moved on".
POST_CLICK_DELAY_SECONDS: Final[float] = 0.1

#: The set of reasons the policy ever emits — also the enum in
#: ``contracts/trajectory.schema.json#/$defs/keyframePayload``.
KEYFRAME_REASONS: Final[tuple[str, ...]] = (
    "periodic",
    "app_switch",
    "pre_click",
    "post_click",
)

# Event-type strings the session orchestrator passes to the policy.
_EVENT_TICK = "tick"
_EVENT_APP_SWITCH = "app_switch"
_EVENT_PRE_CLICK = "pre_click"
_EVENT_POST_CLICK = "post_click"


class KeyframePolicy:
    """Stateless policy for keyframe capture decisions.

    Construct with optional overrides for the timing constants (tests
    pass tighter intervals to exercise boundary behaviour; production
    code should use the defaults).
    """

    def __init__(
        self,
        *,
        periodic_interval_seconds: float = PERIODIC_INTERVAL_SECONDS,
        post_click_delay_seconds: float = POST_CLICK_DELAY_SECONDS,
    ) -> None:
        if periodic_interval_seconds <= 0:
            raise ValueError("periodic_interval_seconds must be > 0")
        if post_click_delay_seconds < 0:
            raise ValueError("post_click_delay_seconds must be >= 0")
        self._periodic_interval_seconds = float(periodic_interval_seconds)
        self._post_click_delay_seconds = float(post_click_delay_seconds)

    @property
    def periodic_interval_seconds(self) -> float:
        """Configured periodic-keyframe interval in seconds."""
        return self._periodic_interval_seconds

    @property
    def post_click_delay_seconds(self) -> float:
        """Configured post-click keyframe delay in seconds."""
        return self._post_click_delay_seconds

    def should_capture(
        self, event_type: str, seconds_since_last_keyframe: float
    ) -> bool:
        """Return ``True`` if a keyframe should be captured now.

        ``event_type`` is one of:

        * ``"tick"`` — periodic timer fired; captures iff
          ``seconds_since_last_keyframe >= periodic_interval_seconds``.
        * ``"app_switch"`` — the focused app just changed; always captures.
        * ``"pre_click"`` / ``"post_click"`` — click envelope; always
          captures.

        Any other value returns ``False``.  This is the canonical gate
        the session orchestrator uses before emitting a ``keyframe`` event.
        """
        return self.reason_for(event_type, seconds_since_last_keyframe) is not None

    def reason_for(
        self, event_type: str, seconds_since_last_keyframe: float
    ) -> str | None:
        """Return the keyframe reason string, or ``None`` if we should skip.

        The returned value, when not ``None``, is guaranteed to be a
        member of :data:`KEYFRAME_REASONS` — the session orchestrator can
        pass it straight into the ``keyframe`` event payload without
        further translation.
        """
        if event_type == _EVENT_TICK:
            if seconds_since_last_keyframe >= self._periodic_interval_seconds:
                return "periodic"
            return None
        if event_type == _EVENT_APP_SWITCH:
            return "app_switch"
        if event_type == _EVENT_PRE_CLICK:
            return "pre_click"
        if event_type == _EVENT_POST_CLICK:
            return "post_click"
        return None
