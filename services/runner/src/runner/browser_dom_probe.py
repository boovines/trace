"""Browser-DOM capability probe for the runner's tiered execution layer.

Step 4.1 (commit 1) of the tiered-execution rollout. Mirrors
:mod:`runner.mcp_client`'s probe layer: at run-manager startup we ask
"can we drive a Chromium instance via Playwright?", cache the answer,
and feed it into :class:`~runner.execution_hints.CapabilityRegistry` so
``tier=browser_dom`` hints can be selected (or skipped with a clear
unsupported_reason) by :func:`~runner.execution_hints.pick_hint`.

The probe is intentionally **non-launching**: we only want to know
whether a Chromium binary is installed for Playwright to drive, not to
actually open a browser at startup. A real browser launch happens in
Step 4.1 commit 2 inside ``BrowserDOMDispatcher`` once the registry has
already accepted a ``tier=browser_dom`` hint.

Two acceptance paths:

* ``$TRACE_CDP_ENDPOINT`` is set → we trust the operator: the
  dispatcher will attach to the existing CDP endpoint, no local
  Chromium needed. The probe records the endpoint for the dispatcher
  to read, and reports ``cdp_endpoint`` in its diagnostics string.
* ``$TRACE_CDP_ENDPOINT`` is absent → import :mod:`playwright` and ask
  for ``chromium.executable_path``. If Playwright isn't installed (the
  runtime dep is missing), or no Chromium browser binary is present
  (``playwright install chromium`` was never run), the probe returns
  ``None`` and the runner silently degrades — the tier just stays
  unavailable, with a per-step ``unsupported_reason`` explaining why.

Failure to probe is **never** fatal: one bad browser environment
should not take down a run that doesn't need browser_dom anyway. This
matches the precedent set by :mod:`runner.mcp_client`.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import os
from dataclasses import dataclass

__all__ = [
    "DEFAULT_PROBE_TIMEOUT_SECONDS",
    "BrowserDOMCapability",
    "format_probe_diagnostic",
    "probe_browser_dom",
    "probe_browser_dom_sync",
]

LOGGER = logging.getLogger(__name__)

#: How long to wait for the Playwright import + executable-path lookup
#: before giving up. The lookup is in-process (no subprocess), so this
#: only ever fires if the playwright import itself hangs — generous
#: enough that a slow first import on a cold cache succeeds.
DEFAULT_PROBE_TIMEOUT_SECONDS: float = 5.0

#: Environment variable the dispatcher reads to attach to a running
#: Chromium over CDP rather than launching its own. Documented here
#: because the probe layer also has to honour it: when set, we don't
#: require a locally-installed Chromium.
_CDP_ENDPOINT_ENV: str = "TRACE_CDP_ENDPOINT"


@dataclass(frozen=True)
class BrowserDOMCapability:
    """Snapshot of the host's ability to drive Chromium for DOM actions.

    Returned by :func:`probe_browser_dom` and stored on
    :class:`~runner.execution_hints.CapabilityRegistry`. The dispatcher
    (Step 4.1 commit 2) reads :attr:`cdp_endpoint` to decide between
    ``connect_over_cdp`` and ``launch``; ``executable_path`` is purely
    diagnostic and lets the run log show *which* Chromium would be
    driven.
    """

    cdp_endpoint: str | None
    executable_path: str | None

    @property
    def is_remote(self) -> bool:
        """True when the dispatcher should attach over CDP rather than launch."""
        return self.cdp_endpoint is not None


def _check_playwright_chromium() -> tuple[str | None, str | None]:
    """Return ``(executable_path, error)`` for the local Chromium install.

    Pure import-and-lookup — no subprocess, no browser launch. Either
    field is ``None`` to signal "no usable Chromium":

    * ``ImportError`` → ``("", "playwright not installed")``-style error
      string, no path. Keeps the runner from crashing when the
      `playwright` Python package isn't on the path despite the dep
      declaration (e.g. the user is running an older venv).
    * ``executable_path`` returns a non-existent path → treat as missing
      (``playwright install chromium`` was never run).
    * Any other exception inside Playwright → log and report as error.
    """
    try:
        # Imported lazily so `runner` modules with no browser-DOM need
        # don't pay the playwright import cost on startup. Playwright's
        # sync API is what we want here — there's no event loop in the
        # probe path, and starting one for a one-shot lookup is silly.
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        return None, f"playwright not installed: {exc}"

    try:
        with sync_playwright() as p:
            executable_path = p.chromium.executable_path
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"

    if not executable_path:
        return None, (
            "playwright reported no chromium executable path; "
            "did you run `uv run playwright install chromium`?"
        )
    if not os.path.exists(executable_path):
        return None, (
            f"chromium executable not found at {executable_path}; "
            f"did you run `uv run playwright install chromium`?"
        )
    return executable_path, None


async def probe_browser_dom(
    *,
    timeout: float = DEFAULT_PROBE_TIMEOUT_SECONDS,
    cdp_endpoint: str | None = None,
) -> BrowserDOMCapability | None:
    """Probe whether the runner can drive Chromium for DOM actions.

    Returns a populated :class:`BrowserDOMCapability` on success, or
    ``None`` when no usable browser is available. The capability is
    handed to :class:`~runner.execution_hints.CapabilityRegistry`; a
    ``None`` return there preserves the conservative default and keeps
    ``tier=browser_dom`` hints unsupported.

    Resolution order:

    1. ``cdp_endpoint`` argument (test override) takes precedence.
    2. ``$TRACE_CDP_ENDPOINT`` if set in the environment.
    3. Otherwise look up Playwright's local Chromium binary.

    The Playwright lookup is run in a worker thread because Playwright's
    own ``sync_playwright()`` cannot be invoked from inside an asyncio
    event loop. Spinning up a fresh thread per probe is fine — it
    happens at most once per run-manager process.
    """
    endpoint = cdp_endpoint or os.environ.get(_CDP_ENDPOINT_ENV)
    if endpoint:
        return BrowserDOMCapability(
            cdp_endpoint=endpoint,
            executable_path=None,
        )

    try:
        async with asyncio.timeout(timeout):
            executable_path, error = await asyncio.to_thread(
                _check_playwright_chromium
            )
    except TimeoutError:
        LOGGER.warning(
            "browser_dom probe timed out after %.1fs; tier disabled",
            timeout,
        )
        return None

    if error is not None:
        LOGGER.info("browser_dom tier unavailable: %s", error)
        return None

    return BrowserDOMCapability(
        cdp_endpoint=None,
        executable_path=executable_path,
    )


def probe_browser_dom_sync(
    *,
    timeout: float = DEFAULT_PROBE_TIMEOUT_SECONDS,
    cdp_endpoint: str | None = None,
) -> BrowserDOMCapability | None:
    """Sync wrapper around :func:`probe_browser_dom` for non-async callers.

    Mirrors :func:`runner.mcp_client.probe_capabilities_sync`: runs the
    async probe on a single-shot worker thread so this helper is safe
    to call from inside another event loop's call stack. The typical
    case is :class:`~runner.run_manager.RunManager.start_run` (async on
    the FastAPI loop) calling the sync ``_get_capability_registry``
    which calls this.
    """

    def _run() -> BrowserDOMCapability | None:
        return asyncio.run(
            probe_browser_dom(timeout=timeout, cdp_endpoint=cdp_endpoint)
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(_run).result()


def format_probe_diagnostic(capability: BrowserDOMCapability | None) -> str:
    """Render the probe outcome as a single line for run-log emission.

    Symmetric with :func:`runner.mcp_client.format_probe_report` so the
    run log can show MCP and browser-DOM probe results side-by-side.
    """
    if capability is None:
        return "browser_dom: unavailable (no chromium / playwright)"
    if capability.cdp_endpoint:
        return f"browser_dom: attach via CDP {capability.cdp_endpoint}"
    return f"browser_dom: launch local chromium ({capability.executable_path})"
