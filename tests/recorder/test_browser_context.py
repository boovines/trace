"""Tests for ``recorder.browser_context`` — Step 6.1.

Covers:

* ``is_browser_bundle`` membership for the canonical Chromium-family
  + Safari + Arc + Firefox bundles, and the negative case.
* ``resolve_browser_context`` happy path: a fake AppleScript runner
  returns a "<url>\\n<title>" payload and we get back the parsed
  envelope.
* Failure paths: non-browser bundle (no runner call), empty
  AppleScript result (no front window), runner raises (timeout),
  Firefox-style "registered but unscriptable" entry.
* Result parsing: title-less results, leading/trailing whitespace.

All tests use a fake runner so the real ``Foundation.NSAppleScript``
path is never exercised — that path is macOS-only and mocking
NSAppleScript inside a unit test would be more code than the
behaviour deserves.
"""

from __future__ import annotations

from typing import Any

from recorder.browser_context import (
    KNOWN_BROWSER_BUNDLES,
    BrowserContext,
    is_browser_bundle,
    resolve_browser_context,
)

# --- is_browser_bundle ---------------------------------------------------


def test_is_browser_bundle_recognises_chrome_family() -> None:
    assert is_browser_bundle("com.google.Chrome") is True
    assert is_browser_bundle("com.brave.Browser") is True
    assert is_browser_bundle("com.microsoft.edgemac") is True


def test_is_browser_bundle_recognises_safari_and_arc() -> None:
    assert is_browser_bundle("com.apple.Safari") is True
    assert is_browser_bundle("company.thebrowser.Browser") is True


def test_is_browser_bundle_rejects_non_browser() -> None:
    assert is_browser_bundle("com.apple.finder") is False
    assert is_browser_bundle(None) is False
    assert is_browser_bundle("") is False


def test_known_browser_bundles_is_a_frozenset() -> None:
    """KNOWN_BROWSER_BUNDLES is exposed for callers to short-circuit
    AppleScript dispatch on non-browser apps. It should be immutable
    so callers can't accidentally mutate the global registry."""
    assert isinstance(KNOWN_BROWSER_BUNDLES, frozenset)


# --- resolve_browser_context: happy path ---------------------------------


def test_resolve_returns_url_and_title_from_runner() -> None:
    captured: dict[str, Any] = {}

    def _fake(script: str, timeout_seconds: float) -> str:
        captured["script"] = script
        captured["timeout"] = timeout_seconds
        return "https://example.com/inbox\nInbox - Example"

    ctx = resolve_browser_context("com.google.Chrome", runner=_fake)
    assert ctx == {"url": "https://example.com/inbox", "title": "Inbox - Example"}
    # Script ran for the right bundle (active tab pattern, not Safari's).
    assert "com.google.Chrome" in captured["script"]
    assert "active tab" in captured["script"]


def test_resolve_uses_safari_specific_script() -> None:
    captured: dict[str, str] = {}

    def _fake(script: str, timeout_seconds: float) -> str:
        captured["script"] = script
        return "https://apple.com/\nApple"

    ctx = resolve_browser_context("com.apple.Safari", runner=_fake)
    assert ctx is not None
    assert "front document" in captured["script"]


def test_resolve_handles_url_only_response() -> None:
    """Some scripts may return only the URL (e.g. when title can't be
    queried). The result envelope should still validate."""

    def _fake(script: str, timeout_seconds: float) -> str:
        return "https://example.com/"

    ctx = resolve_browser_context("com.google.Chrome", runner=_fake)
    assert ctx == {"url": "https://example.com/", "title": None}


def test_resolve_strips_whitespace_around_lines() -> None:
    def _fake(script: str, timeout_seconds: float) -> str:
        return "  https://example.com/  \n  Page Title  "

    ctx = resolve_browser_context("com.google.Chrome", runner=_fake)
    assert ctx == {"url": "https://example.com/", "title": "Page Title"}


# --- resolve_browser_context: failure paths ------------------------------


def test_resolve_returns_none_for_non_browser() -> None:
    """Non-browser bundles short-circuit before ever calling the
    runner — confirms callers don't pay AppleScript dispatch cost
    on Finder / Slack / etc."""
    runner_called = False

    def _fake(script: str, timeout_seconds: float) -> str:
        nonlocal runner_called
        runner_called = True
        return ""

    ctx = resolve_browser_context("com.apple.finder", runner=_fake)
    assert ctx is None
    assert runner_called is False


def test_resolve_returns_none_for_empty_result() -> None:
    """Empty AppleScript result means no front window or private-tab
    redaction. Treated as "no useful capture"."""

    def _fake(script: str, timeout_seconds: float) -> str:
        return ""

    ctx = resolve_browser_context("com.google.Chrome", runner=_fake)
    assert ctx is None


def test_resolve_returns_none_when_runner_raises() -> None:
    """Timeout / permission-denied / app-quit-mid-query all surface
    as exceptions from the runner. The resolver swallows them and
    returns None so the click hot path keeps moving."""

    def _fake(script: str, timeout_seconds: float) -> str:
        raise TimeoutError("AppleScript dispatch exceeded 0.20s")

    ctx = resolve_browser_context("com.google.Chrome", runner=_fake)
    assert ctx is None


def test_resolve_returns_none_for_firefox_unscriptable() -> None:
    """Firefox is registered as a known browser (so callers don't
    fall through to non-browser handling) but the AppleScript layer
    can't query its tabs — script string is empty, resolver short-
    circuits before invoking the runner."""
    runner_called = False

    def _fake(script: str, timeout_seconds: float) -> str:
        nonlocal runner_called
        runner_called = True
        return ""

    ctx = resolve_browser_context("org.mozilla.firefox", runner=_fake)
    assert ctx is None
    assert runner_called is False


def test_resolve_returns_none_for_url_only_whitespace() -> None:
    """A response that's just newlines doesn't parse to a real URL."""

    def _fake(script: str, timeout_seconds: float) -> str:
        return "\n\n"

    ctx = resolve_browser_context("com.google.Chrome", runner=_fake)
    assert ctx is None


# --- type contract check -------------------------------------------------


def test_browsercontext_typed_dict_shape() -> None:
    """BrowserContext is the trajectory-schema-compatible envelope."""
    sample: BrowserContext = {"url": "https://x", "title": None}
    assert sample["url"] == "https://x"
    assert sample["title"] is None
