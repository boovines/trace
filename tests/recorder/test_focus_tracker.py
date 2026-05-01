"""Tests for ``recorder.focus_tracker``.

The state machine (``handle_app_activated`` / ``refresh_window_title``) is
driven directly in unit tests — no PyObjC involved.  The NSWorkspace
subscription path is exercised by stubbing ``AppKit`` / ``ApplicationServices``
via ``sys.modules`` so the tests stay hermetic on the Ralph sandbox.

One ``@pytest.mark.macos`` integration test launches two real apps with
``osascript`` and asserts the tracker fires the expected callbacks on a real
host.  It is skipped by default (see ``conftest.py``).
"""

from __future__ import annotations

import subprocess
import sys
import time
import types
from typing import Any

import pytest

from recorder.focus_tracker import (
    APP_ACTIVATED_NOTIFICATION,
    AppInfo,
    AppSwitchPayload,
    FocusTracker,
    WindowFocusPayload,
    _extract_app_info,
    _ns_running_app_to_dict,
)

# ---------------------------------------------------------------------------
# stubs
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _disable_lsappinfo_probe(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    """Force the lsappinfo frontmost probe to return None in every test.

    Real hosts have ``lsappinfo`` installed and it shells out to return the
    actual frontmost app, bypassing the hermetic AppKit/ApplicationServices
    stubs each test installs. Stubbing it out at the module level keeps the
    tests pinned to the NSWorkspace fallback path they were written for.

    Tests marked ``@pytest.mark.uses_lsappinfo`` (the ones that exercise
    the helper itself) opt out and install their own subprocess stubs.
    """
    if request.node.get_closest_marker("uses_lsappinfo"):
        return
    monkeypatch.setattr(
        "recorder.focus_tracker._query_frontmost_via_lsappinfo",
        lambda: None,
    )


class _FakeRunningApp:
    def __init__(self, bundle_id: str, name: str, pid: int) -> None:
        self._bundle_id = bundle_id
        self._name = name
        self._pid = pid

    def bundleIdentifier(self) -> str:
        return self._bundle_id

    def localizedName(self) -> str:
        return self._name

    def processIdentifier(self) -> int:
        return self._pid


class _FakeUserInfo(dict[str, Any]):
    """Behaves like an NSDictionary; supports the ``.get`` access path."""

    def objectForKey_(self, key: str) -> Any:
        return self.get(key)


class _ObjCOnlyUserInfo:
    """NSDictionary-style stub that exposes only ``objectForKey_``.

    Forces the ``_extract_app_info`` fallback path to exercise the ObjC
    accessor rather than the Python-dict ``.get`` shortcut.
    """

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def objectForKey_(self, key: str) -> Any:
        return self._data.get(key)


class _FakeNotification:
    def __init__(
        self,
        app: _FakeRunningApp | None,
        *,
        via_objc: bool = False,
    ) -> None:
        user_info: Any
        if app is None:
            user_info = None
        elif via_objc:
            user_info = _ObjCOnlyUserInfo({"NSWorkspaceApplicationKey": app})
        else:
            user_info = _FakeUserInfo({"NSWorkspaceApplicationKey": app})
        self._user_info = user_info

    def userInfo(self) -> Any:
        return self._user_info


def _install_appkit_stub(
    monkeypatch: pytest.MonkeyPatch,
    *,
    frontmost: _FakeRunningApp | None = None,
    capture_observer: list[tuple[str, Any]] | None = None,
) -> types.ModuleType:
    stub = types.ModuleType("AppKit")

    class _Center:
        def __init__(self) -> None:
            self.added: list[tuple[str, Any]] = (
                capture_observer if capture_observer is not None else []
            )
            self.removed: list[Any] = []

        def addObserverForName_object_queue_usingBlock_(
            self, name: str, obj: Any, queue: Any, block: Any
        ) -> object:
            handle = object()
            self.added.append((name, block))
            return handle

        def removeObserver_(self, handle: Any) -> None:
            self.removed.append(handle)

    class _Workspace:
        _center = _Center()

        @classmethod
        def sharedWorkspace(cls) -> _Workspace:
            return cls()

        def notificationCenter(self) -> _Center:
            return _Workspace._center

        def frontmostApplication(self) -> _FakeRunningApp | None:
            return frontmost

    stub.NSWorkspace = _Workspace  # type: ignore[attr-defined]
    stub._center = _Workspace._center  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "AppKit", stub)
    return stub


def _install_application_services_stub(
    monkeypatch: pytest.MonkeyPatch,
    *,
    windows_by_pid: dict[int, str | None] | None = None,
    create_returns: dict[int, Any] | None = None,
    raise_on_attr: str | None = None,
) -> types.ModuleType:
    stub = types.ModuleType("ApplicationServices")
    titles = windows_by_pid or {}
    create_map = create_returns or {}

    class _AppElement:
        def __init__(self, pid: int) -> None:
            self.pid = pid

    def AXUIElementCreateApplication(pid: int) -> Any:
        if pid in create_map:
            return create_map[pid]
        return _AppElement(pid)

    def AXUIElementCopyAttributeValue(element: Any, attr: str, _: Any) -> tuple[int, Any]:
        if raise_on_attr == attr:
            raise RuntimeError(f"simulated failure on {attr}")
        if attr == "AXFocusedWindow":
            if isinstance(element, _AppElement):
                title = titles.get(element.pid)
                if title is None:
                    return (0, None)
                return (0, {"__title__": title})
            return (-1, None)
        if attr == "AXTitle":
            if isinstance(element, dict) and "__title__" in element:
                return (0, element["__title__"])
            return (-1, None)
        return (-1, None)

    stub.AXUIElementCreateApplication = AXUIElementCreateApplication  # type: ignore[attr-defined]
    stub.AXUIElementCopyAttributeValue = AXUIElementCopyAttributeValue  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ApplicationServices", stub)
    return stub


# ---------------------------------------------------------------------------
# state machine: handle_app_activated
# ---------------------------------------------------------------------------


def test_first_activation_emits_switch_from_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_application_services_stub(monkeypatch, windows_by_pid={42: "Inbox"})
    switches: list[AppSwitchPayload] = []
    focus: list[WindowFocusPayload] = []
    tracker = FocusTracker()
    tracker.on_app_switch(switches.append)
    tracker.on_window_focus_change(focus.append)

    tracker.handle_app_activated(
        AppInfo(bundle_id="com.google.Chrome", name="Chrome", pid=42)
    )

    assert len(switches) == 1
    assert switches[0]["from_bundle_id"] is None
    assert switches[0]["from_name"] is None
    assert switches[0]["to_bundle_id"] == "com.google.Chrome"
    assert switches[0]["to_name"] == "Chrome"
    assert tracker.get_current_app() == AppInfo(
        bundle_id="com.google.Chrome", name="Chrome", pid=42
    )
    assert focus == [
        WindowFocusPayload(window_title="Inbox", app_bundle_id="com.google.Chrome")
    ]
    assert tracker.get_current_window_title() == "Inbox"


def test_second_activation_emits_from_previous(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_application_services_stub(
        monkeypatch, windows_by_pid={1: "one", 2: "two"}
    )
    switches: list[AppSwitchPayload] = []
    tracker = FocusTracker()
    tracker.on_app_switch(switches.append)

    tracker.handle_app_activated(AppInfo(bundle_id="com.a", name="A", pid=1))
    tracker.handle_app_activated(AppInfo(bundle_id="com.b", name="B", pid=2))

    assert len(switches) == 2
    assert switches[1]["from_bundle_id"] == "com.a"
    assert switches[1]["from_name"] == "A"
    assert switches[1]["to_bundle_id"] == "com.b"
    assert switches[1]["to_name"] == "B"


def test_reactivating_same_app_is_a_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_application_services_stub(monkeypatch, windows_by_pid={9: "win"})
    switches: list[AppSwitchPayload] = []
    tracker = FocusTracker()
    tracker.on_app_switch(switches.append)

    app = AppInfo(bundle_id="com.a", name="A", pid=9)
    tracker.handle_app_activated(app)
    tracker.handle_app_activated(app)

    assert len(switches) == 1  # second activation is suppressed


def test_app_focus_history_finalises_previous_entries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_application_services_stub(monkeypatch)
    tracker = FocusTracker()

    tracker.handle_app_activated(AppInfo(bundle_id="com.a", name="A", pid=1))
    tracker.handle_app_activated(AppInfo(bundle_id="com.b", name="B", pid=2))
    tracker.handle_app_activated(AppInfo(bundle_id="com.c", name="C", pid=3))

    history = tracker.get_app_focus_history()
    assert [h["bundle_id"] for h in history] == ["com.a", "com.b", "com.c"]
    # All but the current entry should have an exited_at timestamp.
    assert history[0]["exited_at"] is not None
    assert history[1]["exited_at"] is not None
    assert history[2]["exited_at"] is None


def test_stop_finalises_last_history_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_appkit_stub(monkeypatch)
    _install_application_services_stub(monkeypatch)
    tracker = FocusTracker()
    tracker.start()
    tracker.handle_app_activated(AppInfo(bundle_id="com.a", name="A", pid=1))
    tracker.stop()
    history = tracker.get_app_focus_history()
    assert history[-1]["exited_at"] is not None


def test_app_switch_callback_exception_does_not_propagate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_application_services_stub(monkeypatch)
    good: list[AppSwitchPayload] = []
    tracker = FocusTracker()

    def bad(_: AppSwitchPayload) -> None:
        raise RuntimeError("consumer bug")

    tracker.on_app_switch(bad)
    tracker.on_app_switch(good.append)
    tracker.handle_app_activated(AppInfo(bundle_id="com.a", name="A", pid=1))
    # Good callback still ran despite the bad one blowing up.
    assert len(good) == 1


# ---------------------------------------------------------------------------
# window title behaviour
# ---------------------------------------------------------------------------


def test_window_title_refresh_only_fires_on_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    titles: dict[int, str | None] = {42: "first"}
    _install_application_services_stub(monkeypatch, windows_by_pid=titles)
    fires: list[WindowFocusPayload] = []
    tracker = FocusTracker()
    tracker.on_window_focus_change(fires.append)

    tracker.handle_app_activated(
        AppInfo(bundle_id="com.a", name="A", pid=42)
    )
    assert [f["window_title"] for f in fires] == ["first"]

    # No change → no additional callback fires.
    tracker.refresh_window_title()
    assert len(fires) == 1

    # Underlying title changes → callback fires exactly once.
    titles[42] = "second"
    tracker.refresh_window_title()
    assert [f["window_title"] for f in fires] == ["first", "second"]
    assert tracker.get_current_window_title() == "second"


def test_window_title_is_none_when_ax_returns_no_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_application_services_stub(monkeypatch, windows_by_pid={42: None})
    fires: list[WindowFocusPayload] = []
    tracker = FocusTracker()
    tracker.on_window_focus_change(fires.append)

    tracker.handle_app_activated(AppInfo(bundle_id="com.a", name="A", pid=42))
    # First observed title is None, which is still a "change" from the
    # initial unknown state — but we explicitly treat None→None as *no*
    # callback to avoid spamming consumers for apps with no window.
    assert fires == []
    assert tracker.get_current_window_title() is None


def test_window_title_refresh_with_no_current_app_is_safe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_application_services_stub(monkeypatch)
    tracker = FocusTracker()
    # Before any activation — must not raise.
    tracker.refresh_window_title()
    assert tracker.get_current_window_title() is None


def test_window_title_ax_raises_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_application_services_stub(
        monkeypatch,
        windows_by_pid={1: "ignored"},
        raise_on_attr="AXFocusedWindow",
    )
    tracker = FocusTracker()
    tracker.handle_app_activated(AppInfo(bundle_id="com.a", name="A", pid=1))
    assert tracker.get_current_window_title() is None


def test_window_title_applicationservices_importerror_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Install a stub that is missing the two symbols — ``from X import Y``
    # raises ImportError which the module must swallow.
    broken = types.ModuleType("ApplicationServices")
    monkeypatch.setitem(sys.modules, "ApplicationServices", broken)
    tracker = FocusTracker()
    tracker.handle_app_activated(AppInfo(bundle_id="com.a", name="A", pid=1))
    assert tracker.get_current_window_title() is None


def test_window_title_create_returns_none_gives_none_title(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_application_services_stub(
        monkeypatch,
        windows_by_pid={1: "won't be seen"},
        create_returns={1: None},
    )
    tracker = FocusTracker()
    tracker.handle_app_activated(AppInfo(bundle_id="com.a", name="A", pid=1))
    assert tracker.get_current_window_title() is None


# ---------------------------------------------------------------------------
# NSWorkspace subscription plumbing
# ---------------------------------------------------------------------------


def test_start_subscribes_to_workspace_notifications(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    appkit = _install_appkit_stub(monkeypatch)
    _install_application_services_stub(monkeypatch)
    tracker = FocusTracker()
    tracker.start()
    try:
        center = appkit._center  # type: ignore[attr-defined]
        assert len(center.added) == 1
        name, block = center.added[0]
        assert name == APP_ACTIVATED_NOTIFICATION
        assert callable(block)
    finally:
        tracker.stop()
        center = appkit._center  # type: ignore[attr-defined]
        assert len(center.removed) == 1


def test_start_twice_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_appkit_stub(monkeypatch)
    _install_application_services_stub(monkeypatch)
    tracker = FocusTracker()
    tracker.start()
    try:
        with pytest.raises(RuntimeError, match="more than once"):
            tracker.start()
    finally:
        tracker.stop()


def test_stop_before_start_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_appkit_stub(monkeypatch)
    tracker = FocusTracker()
    tracker.stop()  # must not raise


def test_appkit_importerror_disables_subscription_gracefully(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Install an AppKit stub missing NSWorkspace — triggers ImportError on
    # ``from AppKit import NSWorkspace`` and the tracker should just log.
    broken = types.ModuleType("AppKit")
    monkeypatch.setitem(sys.modules, "AppKit", broken)
    _install_application_services_stub(monkeypatch)
    tracker = FocusTracker()
    tracker.start()
    # Still drivable via handle_app_activated even without NSWorkspace.
    received: list[AppSwitchPayload] = []
    tracker.on_app_switch(received.append)
    tracker.handle_app_activated(AppInfo(bundle_id="com.a", name="A", pid=1))
    assert len(received) == 1
    tracker.stop()


def test_block_delivers_via_notification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The block registered with NSWorkspace drives the state machine."""
    captured: list[tuple[str, Any]] = []
    _install_appkit_stub(monkeypatch, capture_observer=captured)
    _install_application_services_stub(monkeypatch, windows_by_pid={7: "hello"})
    received: list[AppSwitchPayload] = []
    focus: list[WindowFocusPayload] = []
    tracker = FocusTracker()
    tracker.on_app_switch(received.append)
    tracker.on_window_focus_change(focus.append)
    tracker.start()
    try:
        assert len(captured) == 1
        _, block = captured[0]
        fake_app = _FakeRunningApp("com.x", "X", 7)
        block(_FakeNotification(fake_app))
        assert len(received) == 1
        assert received[0]["to_bundle_id"] == "com.x"
        assert focus == [
            WindowFocusPayload(window_title="hello", app_bundle_id="com.x")
        ]
    finally:
        tracker.stop()


def test_block_with_missing_user_info_is_safe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[tuple[str, Any]] = []
    _install_appkit_stub(monkeypatch, capture_observer=captured)
    _install_application_services_stub(monkeypatch)
    tracker = FocusTracker()
    received: list[AppSwitchPayload] = []
    tracker.on_app_switch(received.append)
    tracker.start()
    try:
        _, block = captured[0]
        block(_FakeNotification(None))  # no running app in userInfo
        assert received == []
    finally:
        tracker.stop()


def test_frontmost_app_seeded_on_start(monkeypatch: pytest.MonkeyPatch) -> None:
    """If a frontmost app exists at start(), the tracker seeds its state."""
    frontmost = _FakeRunningApp("com.seed", "Seed", 99)
    _install_appkit_stub(monkeypatch, frontmost=frontmost)
    _install_application_services_stub(
        monkeypatch, windows_by_pid={99: "Seed Window"}
    )
    received: list[AppSwitchPayload] = []
    tracker = FocusTracker()
    tracker.on_app_switch(received.append)
    tracker.start()
    try:
        assert tracker.get_current_app() == AppInfo(
            bundle_id="com.seed", name="Seed", pid=99
        )
        assert tracker.get_current_window_title() == "Seed Window"
        assert len(received) == 1
    finally:
        tracker.stop()


# ---------------------------------------------------------------------------
# helper coverage
# ---------------------------------------------------------------------------


def test_extract_app_info_via_get_path() -> None:
    info = _extract_app_info(_FakeNotification(_FakeRunningApp("com.a", "A", 5)))
    assert info == AppInfo(bundle_id="com.a", name="A", pid=5)


def test_extract_app_info_via_objc_path() -> None:
    info = _extract_app_info(
        _FakeNotification(_FakeRunningApp("com.a", "A", 6), via_objc=True)
    )
    assert info == AppInfo(bundle_id="com.a", name="A", pid=6)


def test_extract_app_info_with_no_user_info_returns_none() -> None:
    assert _extract_app_info(_FakeNotification(None)) is None


def test_ns_running_app_bad_pid_returns_none() -> None:
    class _Broken:
        def bundleIdentifier(self) -> str:
            return "com.x"

        def localizedName(self) -> str:
            return "X"

        def processIdentifier(self) -> None:
            return None

    assert _ns_running_app_to_dict(_Broken()) is None


def test_ns_running_app_empty_fields_become_empty_strings() -> None:
    class _Empty:
        def bundleIdentifier(self) -> str:
            return ""

        def localizedName(self) -> str:
            return ""

        def processIdentifier(self) -> int:
            return 3

    info = _ns_running_app_to_dict(_Empty())
    assert info == AppInfo(bundle_id="", name="", pid=3)


# ---------------------------------------------------------------------------
# real-macOS integration smoke (opt-in)
# ---------------------------------------------------------------------------


@pytest.mark.macos
def test_macos_real_app_switch_fires_callback() -> None:  # pragma: no cover
    """Activate two real apps via osascript and verify callbacks fire.

    Requires AppKit to be importable and the user's terminal to have
    permission to post Apple Events.  Skipped on the default run.
    """
    pytest.importorskip("AppKit")
    switches: list[AppSwitchPayload] = []
    tracker = FocusTracker()
    tracker.on_app_switch(switches.append)
    tracker.start()
    try:
        # Activate Finder and then the Terminal so focus changes at least once.
        subprocess.run(
            ["osascript", "-e", 'tell application "Finder" to activate'],
            check=False,
        )
        time.sleep(0.5)
        subprocess.run(
            ["osascript", "-e", 'tell application "Terminal" to activate'],
            check=False,
        )
        deadline = time.time() + 3.0
        while time.time() < deadline and len(switches) < 1:
            time.sleep(0.1)
    finally:
        tracker.stop()
    assert switches, "expected at least one app-switch callback"


# ---------------------------------------------------------------------------
# lsappinfo helper
# ---------------------------------------------------------------------------


def _make_run_result(stdout: str, returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr="")


@pytest.mark.uses_lsappinfo
def test_lsappinfo_parses_three_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    from recorder.focus_tracker import _query_frontmost_via_lsappinfo

    monkeypatch.setattr("shutil.which", lambda _name: "/usr/bin/lsappinfo")
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        if cmd[:2] == ["lsappinfo", "front"]:
            return _make_run_result("ASN:0x0-0x123456:\n")
        return _make_run_result(
            '"pid"=4242\n'
            '"CFBundleIdentifier"="com.apple.TextEdit"\n'
            '"LSDisplayName"="TextEdit"\n'
        )

    monkeypatch.setattr("subprocess.run", fake_run)
    result = _query_frontmost_via_lsappinfo()
    assert result == {"bundle_id": "com.apple.TextEdit", "name": "TextEdit", "pid": 4242}
    # Second call asn should be the exact ASN from the first call's stdout.
    assert calls[1][-1] == "ASN:0x0-0x123456"


@pytest.mark.uses_lsappinfo
def test_lsappinfo_missing_binary_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    from recorder.focus_tracker import _query_frontmost_via_lsappinfo

    monkeypatch.setattr("shutil.which", lambda _name: None)
    # subprocess.run should never be called; make it blow up if it is.
    monkeypatch.setattr(
        "subprocess.run",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("should not run")),
    )
    assert _query_frontmost_via_lsappinfo() is None


@pytest.mark.uses_lsappinfo
def test_lsappinfo_timeout_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    from recorder.focus_tracker import _query_frontmost_via_lsappinfo

    monkeypatch.setattr("shutil.which", lambda _name: "/usr/bin/lsappinfo")

    def raise_timeout(*_: Any, **__: Any) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(cmd="lsappinfo", timeout=1.0)

    monkeypatch.setattr("subprocess.run", raise_timeout)
    assert _query_frontmost_via_lsappinfo() is None


@pytest.mark.uses_lsappinfo
def test_lsappinfo_info_nonzero_returncode_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from recorder.focus_tracker import _query_frontmost_via_lsappinfo

    monkeypatch.setattr("shutil.which", lambda _name: "/usr/bin/lsappinfo")

    def fake_run(cmd: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        if cmd[:2] == ["lsappinfo", "front"]:
            return _make_run_result("ASN:0x0-0x1:\n")
        return _make_run_result("", returncode=1)

    monkeypatch.setattr("subprocess.run", fake_run)
    assert _query_frontmost_via_lsappinfo() is None


@pytest.mark.uses_lsappinfo
def test_lsappinfo_partial_fields_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    from recorder.focus_tracker import _query_frontmost_via_lsappinfo

    monkeypatch.setattr("shutil.which", lambda _name: "/usr/bin/lsappinfo")

    def fake_run(cmd: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        if cmd[:2] == ["lsappinfo", "front"]:
            return _make_run_result("ASN:0x0-0x1:\n")
        return _make_run_result('"pid"=42\n"CFBundleIdentifier"=""\n"LSDisplayName"="X"\n')

    monkeypatch.setattr("subprocess.run", fake_run)
    assert _query_frontmost_via_lsappinfo() is None
