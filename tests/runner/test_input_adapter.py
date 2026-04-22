"""Tests for :mod:`runner.input_adapter` — dry-run recorder contract.

``DryRunInputAdapter`` is the adapter Ralph iterations use; every test should
confirm that method calls are recorded with exact shape and order but that
nothing is posted to the real event system (there is nothing to observe on
that side — its absence is verified at the contract level by never importing
PyObjC from this module).

The live adapter's safety gate and CGEventPost plumbing are covered in
``test_live_input.py``.
"""

from __future__ import annotations

from runner.input_adapter import (
    DryRunInputAdapter,
    InputAdapter,
)


def test_dry_run_adapter_satisfies_protocol() -> None:
    adapter = DryRunInputAdapter()
    assert isinstance(adapter, InputAdapter)


def test_records_click_with_default_button_and_modifiers() -> None:
    adapter = DryRunInputAdapter()
    adapter.click(100.0, 200.0)
    assert adapter.get_recorded_calls() == [
        ("click", (100.0, 200.0, "left", ()), {}),
    ]


def test_records_click_with_button_and_modifiers() -> None:
    adapter = DryRunInputAdapter()
    adapter.click(50.0, 75.0, button="right", modifiers=["cmd", "shift"])
    assert adapter.get_recorded_calls() == [
        ("click", (50.0, 75.0, "right", ("cmd", "shift")), {}),
    ]


def test_records_type_text() -> None:
    adapter = DryRunInputAdapter()
    adapter.type_text("hello world")
    assert adapter.get_recorded_calls() == [("type_text", ("hello world",), {})]


def test_records_unicode_type_text() -> None:
    adapter = DryRunInputAdapter()
    adapter.type_text("héllo 👋")
    assert adapter.get_recorded_calls() == [("type_text", ("héllo 👋",), {})]


def test_records_key_press_preserves_order_as_tuple() -> None:
    adapter = DryRunInputAdapter()
    adapter.key_press(["cmd", "shift", "n"])
    assert adapter.get_recorded_calls() == [
        ("key_press", (("cmd", "shift", "n"),), {}),
    ]


def test_records_scroll() -> None:
    adapter = DryRunInputAdapter()
    adapter.scroll(100.0, 200.0, "down", 5)
    assert adapter.get_recorded_calls() == [
        ("scroll", (100.0, 200.0, "down", 5), {}),
    ]


def test_records_move_mouse() -> None:
    adapter = DryRunInputAdapter()
    adapter.move_mouse(640.0, 480.0)
    assert adapter.get_recorded_calls() == [("move_mouse", (640.0, 480.0), {})]


def test_records_full_sequence_in_order() -> None:
    adapter = DryRunInputAdapter()
    adapter.click(10.0, 20.0)
    adapter.type_text("Hi")
    adapter.key_press(["cmd", "a"])
    adapter.scroll(10.0, 20.0, "up", 3)
    adapter.move_mouse(0.0, 0.0)

    recorded = adapter.get_recorded_calls()
    assert [name for name, _, _ in recorded] == [
        "click",
        "type_text",
        "key_press",
        "scroll",
        "move_mouse",
    ]
    assert recorded == [
        ("click", (10.0, 20.0, "left", ()), {}),
        ("type_text", ("Hi",), {}),
        ("key_press", (("cmd", "a"),), {}),
        ("scroll", (10.0, 20.0, "up", 3), {}),
        ("move_mouse", (0.0, 0.0), {}),
    ]


def test_get_recorded_calls_returns_a_copy() -> None:
    adapter = DryRunInputAdapter()
    adapter.click(1.0, 2.0)
    snapshot = adapter.get_recorded_calls()
    snapshot.clear()
    # Internal state must be untouched.
    assert len(adapter.get_recorded_calls()) == 1


def test_clear_drops_all_recorded_calls() -> None:
    adapter = DryRunInputAdapter()
    adapter.click(1.0, 2.0)
    adapter.type_text("x")
    assert len(adapter.get_recorded_calls()) == 2
    adapter.clear()
    assert adapter.get_recorded_calls() == []


def test_adapter_is_reusable_after_clear() -> None:
    adapter = DryRunInputAdapter()
    adapter.click(1.0, 2.0)
    adapter.clear()
    adapter.type_text("second life")
    assert adapter.get_recorded_calls() == [("type_text", ("second life",), {})]


