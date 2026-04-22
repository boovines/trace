"""Tests for ``recorder.keyframe_policy``.

The policy is stateless so every test is a direct call-and-assert. We
cover each trigger condition, the periodic boundary (strictly >=), and
the "any unknown event_type is ignored" behaviour.
"""

from __future__ import annotations

import pytest

from recorder.keyframe_policy import (
    KEYFRAME_REASONS,
    PERIODIC_INTERVAL_SECONDS,
    POST_CLICK_DELAY_SECONDS,
    KeyframePolicy,
)

# ---------------------------------------------------------------------------
# Defaults / construction
# ---------------------------------------------------------------------------


def test_defaults_match_public_constants() -> None:
    policy = KeyframePolicy()
    assert policy.periodic_interval_seconds == PERIODIC_INTERVAL_SECONDS == 5.0
    assert policy.post_click_delay_seconds == POST_CLICK_DELAY_SECONDS == 0.1


def test_reasons_match_schema_enum() -> None:
    # The schema's keyframePayload.reason enum is a locked contract; this
    # test pins the policy's output vocabulary to it.
    assert set(KEYFRAME_REASONS) == {"periodic", "app_switch", "pre_click", "post_click"}


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_periodic_interval_must_be_positive(bad: float) -> None:
    with pytest.raises(ValueError, match="periodic_interval_seconds"):
        KeyframePolicy(periodic_interval_seconds=bad)


def test_post_click_delay_must_be_non_negative() -> None:
    # 0 is explicitly allowed (e.g. tests that don't want to wait).
    KeyframePolicy(post_click_delay_seconds=0.0)
    with pytest.raises(ValueError, match="post_click_delay_seconds"):
        KeyframePolicy(post_click_delay_seconds=-0.01)


# ---------------------------------------------------------------------------
# Periodic (tick) trigger
# ---------------------------------------------------------------------------


def test_tick_fires_when_interval_met() -> None:
    policy = KeyframePolicy()
    assert policy.should_capture("tick", 5.0) is True
    assert policy.reason_for("tick", 5.0) == "periodic"


def test_tick_fires_when_interval_exceeded() -> None:
    policy = KeyframePolicy()
    assert policy.should_capture("tick", 12.5) is True
    assert policy.reason_for("tick", 12.5) == "periodic"


def test_tick_does_not_fire_below_interval() -> None:
    policy = KeyframePolicy()
    assert policy.should_capture("tick", 4.999) is False
    assert policy.reason_for("tick", 4.999) is None


def test_tick_does_not_fire_at_zero() -> None:
    policy = KeyframePolicy()
    assert policy.should_capture("tick", 0.0) is False


def test_custom_periodic_interval_respected() -> None:
    policy = KeyframePolicy(periodic_interval_seconds=1.0)
    assert policy.should_capture("tick", 0.99) is False
    assert policy.should_capture("tick", 1.0) is True


# ---------------------------------------------------------------------------
# Event-triggered captures (app_switch, pre_click, post_click)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("event_type", "reason"),
    [
        ("app_switch", "app_switch"),
        ("pre_click", "pre_click"),
        ("post_click", "post_click"),
    ],
)
def test_event_triggers_capture_regardless_of_elapsed(
    event_type: str, reason: str
) -> None:
    policy = KeyframePolicy()
    # Always fires — even 0 seconds since last keyframe is fine.
    assert policy.should_capture(event_type, 0.0) is True
    assert policy.reason_for(event_type, 0.0) == reason
    # And also after a long gap (confirms elapsed is irrelevant here).
    assert policy.should_capture(event_type, 60.0) is True
    assert policy.reason_for(event_type, 60.0) == reason


# ---------------------------------------------------------------------------
# Unknown / ignored event types
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "event_type",
    [
        "click",  # the raw click itself — pre/post_click are the keyframe triggers
        "keypress",
        "scroll",
        "text_input",
        "window_focus",
        "tap_reenabled",
        "",  # empty string
        "TICK",  # wrong case — policy is exact-match
    ],
)
def test_unknown_event_types_do_not_trigger(event_type: str) -> None:
    policy = KeyframePolicy()
    assert policy.should_capture(event_type, 99.0) is False
    assert policy.reason_for(event_type, 99.0) is None
