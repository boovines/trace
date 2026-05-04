"""Tests for ``runner.browser_dom_probe`` — Step 4.1 commit 1.

Covers:

* ``probe_browser_dom`` honours an explicit ``cdp_endpoint`` argument
  without ever touching Playwright (so this branch is fast + offline).
* ``$TRACE_CDP_ENDPOINT`` is read from the environment when no argument
  is passed.
* When neither a CDP endpoint nor a usable Chromium is available, the
  probe returns ``None`` and never raises.
* The Playwright-import path uses ``asyncio.to_thread`` so we patch
  the underlying ``_check_playwright_chromium`` helper to assert
  success/failure handling without launching a real browser.
* ``BrowserDOMCapability.is_remote`` flips on the CDP-endpoint branch.
* ``format_probe_diagnostic`` produces a stable, single-line summary
  for run-log emission.
* ``probe_browser_dom_sync`` is a working sync wrapper.
"""

from __future__ import annotations

import asyncio
import os

import pytest

from runner import browser_dom_probe
from runner.browser_dom_probe import (
    BrowserDOMCapability,
    format_probe_diagnostic,
    probe_browser_dom,
    probe_browser_dom_sync,
)

# --- BrowserDOMCapability shape -------------------------------------------


def test_capability_is_remote_true_for_cdp_endpoint() -> None:
    cap = BrowserDOMCapability(
        cdp_endpoint="ws://localhost:9222/devtools/browser/abc",
        executable_path=None,
    )
    assert cap.is_remote is True


def test_capability_is_remote_false_for_local_executable() -> None:
    cap = BrowserDOMCapability(
        cdp_endpoint=None,
        executable_path="/path/to/chromium",
    )
    assert cap.is_remote is False


# --- probe_browser_dom: CDP-endpoint branch -------------------------------


def test_probe_honours_explicit_cdp_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Make sure the env-var fallback can't accidentally satisfy the call.
    monkeypatch.delenv("TRACE_CDP_ENDPOINT", raising=False)

    async def _no_chromium_check() -> tuple[str | None, str | None]:
        raise AssertionError(
            "Playwright probe should not run when an explicit CDP endpoint is given"
        )

    monkeypatch.setattr(
        browser_dom_probe,
        "_check_playwright_chromium",
        _no_chromium_check,
    )
    capability = asyncio.run(
        probe_browser_dom(cdp_endpoint="ws://localhost:9222/devtools/browser/x")
    )
    assert capability is not None
    assert capability.cdp_endpoint == "ws://localhost:9222/devtools/browser/x"
    assert capability.executable_path is None
    assert capability.is_remote is True


def test_probe_reads_cdp_endpoint_env_when_no_argument(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRACE_CDP_ENDPOINT", "ws://example.com:1234/x")

    def _no_chromium_check() -> tuple[str | None, str | None]:
        raise AssertionError("env override should bypass the chromium probe")

    monkeypatch.setattr(
        browser_dom_probe,
        "_check_playwright_chromium",
        _no_chromium_check,
    )
    capability = asyncio.run(probe_browser_dom())
    assert capability is not None
    assert capability.cdp_endpoint == "ws://example.com:1234/x"


# --- probe_browser_dom: chromium-on-disk branch ---------------------------


def test_probe_returns_capability_when_chromium_present(
    monkeypatch: pytest.MonkeyPatch, tmp_path: object
) -> None:
    monkeypatch.delenv("TRACE_CDP_ENDPOINT", raising=False)

    fake_path = "/fake/chromium"

    def _ok() -> tuple[str | None, str | None]:
        return fake_path, None

    monkeypatch.setattr(browser_dom_probe, "_check_playwright_chromium", _ok)
    capability = asyncio.run(probe_browser_dom())
    assert capability is not None
    assert capability.cdp_endpoint is None
    assert capability.executable_path == fake_path
    assert capability.is_remote is False


def test_probe_returns_none_when_playwright_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TRACE_CDP_ENDPOINT", raising=False)

    def _missing() -> tuple[str | None, str | None]:
        return None, "playwright not installed"

    monkeypatch.setattr(browser_dom_probe, "_check_playwright_chromium", _missing)
    capability = asyncio.run(probe_browser_dom())
    assert capability is None


def test_probe_returns_none_when_chromium_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TRACE_CDP_ENDPOINT", raising=False)

    def _no_chromium() -> tuple[str | None, str | None]:
        return None, "chromium executable not found at /missing"

    monkeypatch.setattr(
        browser_dom_probe, "_check_playwright_chromium", _no_chromium
    )
    capability = asyncio.run(probe_browser_dom())
    assert capability is None


def test_probe_times_out_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TRACE_CDP_ENDPOINT", raising=False)

    def _hang() -> tuple[str | None, str | None]:
        # asyncio.to_thread runs this on a worker; we simulate a stall by
        # sleeping past the probe timeout.
        import time as _time

        _time.sleep(1.0)
        return None, "should not be reached"

    monkeypatch.setattr(browser_dom_probe, "_check_playwright_chromium", _hang)
    capability = asyncio.run(probe_browser_dom(timeout=0.05))
    assert capability is None


# --- probe_browser_dom_sync ----------------------------------------------


def test_probe_sync_wraps_async(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TRACE_CDP_ENDPOINT", raising=False)
    monkeypatch.setattr(
        browser_dom_probe,
        "_check_playwright_chromium",
        lambda: ("/fake/chromium", None),
    )
    capability = probe_browser_dom_sync()
    assert capability is not None
    assert capability.executable_path == "/fake/chromium"


@pytest.mark.asyncio
async def test_probe_sync_safe_inside_running_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The sync wrapper must not blow up when called from inside a loop.

    ``RunManager.start_run`` is async and runs on the FastAPI loop;
    inside it, ``_get_capability_registry`` (sync) calls this wrapper.
    A naive ``asyncio.run`` raises RuntimeError in that situation. This
    test pins the deferred-to-thread fix in place.
    """
    monkeypatch.delenv("TRACE_CDP_ENDPOINT", raising=False)
    monkeypatch.setattr(
        browser_dom_probe,
        "_check_playwright_chromium",
        lambda: ("/fake/chromium", None),
    )
    capability = probe_browser_dom_sync()
    assert capability is not None
    assert capability.executable_path == "/fake/chromium"


# --- format_probe_diagnostic ---------------------------------------------


def test_format_diagnostic_unavailable() -> None:
    line = format_probe_diagnostic(None)
    assert "unavailable" in line
    assert "browser_dom" in line


def test_format_diagnostic_local() -> None:
    cap = BrowserDOMCapability(cdp_endpoint=None, executable_path="/x/chrome")
    line = format_probe_diagnostic(cap)
    assert "launch local" in line
    assert "/x/chrome" in line


def test_format_diagnostic_cdp() -> None:
    cap = BrowserDOMCapability(
        cdp_endpoint="ws://localhost:9222/devtools/browser/x",
        executable_path=None,
    )
    line = format_probe_diagnostic(cap)
    assert "CDP" in line
    assert "ws://localhost:9222" in line


# --- module-level invariant ----------------------------------------------


def test_module_does_not_import_playwright_eagerly() -> None:
    """The probe module must defer the playwright import.

    Importing playwright eagerly would force every runner process to
    pay the import cost even when browser_dom isn't used. The lazy
    import inside ``_check_playwright_chromium`` is part of the
    module's contract.
    """
    src = browser_dom_probe.__file__
    assert src is not None and os.path.isfile(src)
    with open(src, encoding="utf-8") as fh:
        text = fh.read()
    # No top-level ``from playwright`` / ``import playwright``.
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if stripped.startswith("from playwright") or stripped.startswith(
            "import playwright"
        ):
            # Allowed only inside a function (indented lines).
            assert line.startswith(" "), (
                f"top-level playwright import found: {line!r}"
            )
