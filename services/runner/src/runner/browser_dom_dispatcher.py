"""Playwright-backed dispatcher for ``tier=browser_dom`` execution hints.

Step 4.1 commit 2 of the tiered-execution rollout. Mirrors
:class:`runner.mcp_client.MCPCallDispatcher`: a long-lived,
async-context-managed pool of browser state that the executor enters
once at run start and exits when the run finishes. Inside the context
the executor calls :meth:`BrowserDOMDispatcher.dispatch` once per step
that resolved to ``tier=browser_dom``.

Wiring into the executor's pre-execution flow is commit 3; this commit
is the dispatcher itself plus a fake-driven test surface so CI never
launches a real browser.

Hint shape (from ``contracts/skill-meta.schema.json``)::

    {
      "tier": "browser_dom",
      "url_pattern": "https://mail.google.com/mail/u/0/#inbox",  # optional
      "selector": "button[aria-label='Send']",                    # optional
      "action": "click" | "type" | "navigate" | "scroll" | "submit",
      "value": "literal text or URL for type/navigate",           # optional
    }

``{param}`` references in any string field are substituted from the
run's parameter map at dispatch time. Missing parameters raise
``KeyError`` so the dispatcher can fall through to the next tier
rather than send a half-formed action — same contract as the MCP
dispatcher's :func:`runner.mcp_client.substitute_parameters`.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Awaitable, Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, cast

from runner.browser_dom_probe import BrowserDOMCapability

if TYPE_CHECKING:
    from contextlib import AbstractAsyncContextManager

__all__ = [
    "DEFAULT_ACTION_TIMEOUT_MS",
    "DEFAULT_FRAME_JPEG_QUALITY",
    "BrowserDOMCallError",
    "BrowserDOMDispatcher",
    "BrowserDOMResult",
    "FrameSink",
    "PageLike",
    "substitute_parameters_in_string",
]

#: JPEG quality for live-stream frames captured after each action. 70
#: keeps the file size around 60-100 KB at typical 1280-wide viewports
#: while staying readable; the dashboard's live-frame card downscales
#: anyway. Tunable via ``BrowserDOMDispatcher`` constructor if a future
#: caller wants higher fidelity (e.g. an offline replay tool).
DEFAULT_FRAME_JPEG_QUALITY: int = 70

#: Callback the dispatcher invokes with the latest action's screenshot
#: bytes (always JPEG). The ``BrowserDOMDispatcher`` itself never
#: writes to disk or broadcasts events — that's the executor's
#: responsibility through this sink so the dispatcher stays unaware of
#: the run-writer / event-broadcaster layering.
FrameSink = Callable[[bytes], Awaitable[None]]

LOGGER = logging.getLogger(__name__)

#: Per-action wait budget. Generous for cold-start network but tight
#: enough that a stuck selector doesn't stall the whole run. In ms,
#: matching Playwright's API conventions.
DEFAULT_ACTION_TIMEOUT_MS: int = 10_000

#: Mirrors ``runner.mcp_client._PARAM_REF_RE``. Re-stated here so the
#: dispatcher doesn't depend on a private symbol from a sibling module
#: (and so the regex lives next to its only consumer in this module).
_PARAM_REF_RE = re.compile(r"\{([a-z][a-z0-9_]{0,29})\}")

#: Browser_dom hint actions the dispatcher knows how to perform. Kept
#: in lockstep with the ``action`` enum in ``skill-meta.schema.json``.
_SUPPORTED_ACTIONS: frozenset[str] = frozenset(
    {"click", "type", "navigate", "scroll", "submit"}
)


def substitute_parameters_in_string(
    value: str, parameters: Mapping[str, str]
) -> str:
    """Replace ``{name}`` references in a single hint-string value.

    Raises :class:`KeyError` (with the missing name) when a referenced
    parameter is not declared on the run, matching the MCP dispatcher's
    convention so the executor's fall-through logic is identical
    across tiers.
    """

    def _sub(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in parameters:
            raise KeyError(name)
        return parameters[name]

    return _PARAM_REF_RE.sub(_sub, value)


class BrowserDOMCallError(RuntimeError):
    """Raised when a browser_dom hint cannot be issued at all.

    Distinct from a :class:`BrowserDOMResult` with ``ok=False`` (which
    means the action ran but the page rejected it). Carries the action
    + selector for log emission.
    """

    def __init__(self, *, action: str, selector: str | None, message: str) -> None:
        loc = selector or "<no-selector>"
        super().__init__(f"browser_dom {action}({loc}): {message}")
        self.action = action
        self.selector = selector
        self.detail = message


@dataclass(frozen=True)
class BrowserDOMResult:
    """Outcome of one ``BrowserDOMDispatcher.dispatch`` call.

    Mirrors :class:`runner.mcp_client.MCPCallResult`:

    * ``ok`` — True when the action succeeded.
    * ``action`` / ``selector`` / ``url`` — call identity for log lines.
    * ``content_text`` — short human-readable summary the executor
      injects into the agent's primer transcript so the agent knows
      what the dispatcher already did. ``None`` only on hard errors
      with no useful description.
    * ``duration_ms`` — wall-clock elapsed time for the action.
    * ``error`` — error message when ``ok`` is False; ``None`` otherwise.
    """

    ok: bool
    action: str
    selector: str | None
    url: str | None
    content_text: str | None
    duration_ms: float
    error: str | None = None


class PageLike(Protocol):
    """Minimum :class:`playwright.async_api.Page` surface the dispatcher uses.

    Defining the dispatcher against this protocol — rather than
    Playwright's concrete ``Page`` — keeps the test surface tight: a
    fake page only has to implement the methods listed here, not the
    full Playwright API. The protocol matches Playwright's signatures
    so a real ``Page`` is automatically structurally-compatible.
    """

    url: str

    async def goto(
        self,
        url: str,
        *,
        timeout: float | None = ...,
        wait_until: str | None = ...,
    ) -> Any: ...

    async def click(self, selector: str, *, timeout: float | None = ...) -> None: ...

    async def fill(
        self, selector: str, value: str, *, timeout: float | None = ...
    ) -> None: ...

    async def press(
        self, selector: str, key: str, *, timeout: float | None = ...
    ) -> None: ...

    async def evaluate(self, expression: str) -> Any: ...

    async def title(self) -> str: ...

    async def screenshot(
        self,
        *,
        type: str = ...,
        quality: int | None = ...,
        full_page: bool = ...,
    ) -> bytes: ...


class BrowserDOMDispatcher:
    """Long-lived browser state for in-run DOM actions.

    Lifecycle: instantiate, ``async with dispatcher:`` (launches Chromium
    or attaches over CDP), do dispatches, exit the context (closes the
    browser). The :class:`runner.executor.Executor` enters the context
    once at run start and exits when the run finishes — so the per-run
    cold-start cost of launching Chromium is paid once, not per step.

    Browser_dom hints with action ``navigate`` perform the navigation
    directly. Hints with any other action consult ``url_pattern``: if
    the current page's URL doesn't match (after parameter
    substitution), the dispatcher navigates to it first. This frees
    the synthesizer from having to author per-step navigation hints.

    The constructor takes an optional ``page_provider`` for tests: a
    callable returning an async-context-manager that yields a
    :class:`PageLike`. When provided, the dispatcher uses the
    yielded page instead of launching real Chromium. Production code
    leaves it ``None`` and the dispatcher launches Chromium itself
    (or attaches over CDP from :attr:`BrowserDOMCapability.cdp_endpoint`).
    """

    def __init__(
        self,
        capability: BrowserDOMCapability,
        *,
        action_timeout_ms: int = DEFAULT_ACTION_TIMEOUT_MS,
        page_provider: PageProvider | None = None,
        on_frame: FrameSink | None = None,
        frame_jpeg_quality: int = DEFAULT_FRAME_JPEG_QUALITY,
    ) -> None:
        self._capability = capability
        self._action_timeout_ms = action_timeout_ms
        self._page_provider = page_provider
        self._page: PageLike | None = None
        # When ``on_frame`` is set, a JPEG of the page is captured
        # after every successful action and handed off via the sink.
        # Capture failures never abort the dispatch — frame capture is
        # observability, the action result is what the run depends on.
        self._on_frame = on_frame
        self._frame_jpeg_quality = frame_jpeg_quality
        # Real-Playwright teardown stack — populated only when no
        # page_provider override is set (production path).
        self._teardown_stack: list[Any] = []
        self._provider_ctx: AbstractAsyncContextManager[PageLike] | None = None

    async def __aenter__(self) -> BrowserDOMDispatcher:
        if self._page_provider is not None:
            self._provider_ctx = self._page_provider()
            self._page = await self._provider_ctx.__aenter__()
            return self
        await self._launch_real_browser()
        return self

    async def __aexit__(self, *_: Any) -> None:
        if self._provider_ctx is not None:
            with suppress(Exception):
                await self._provider_ctx.__aexit__(None, None, None)
            self._provider_ctx = None
            self._page = None
            return
        # Real path: tear down in reverse open order. Never raise out of
        # __aexit__ — a crashed browser process should be cleanly logged,
        # not bubbled to the run task.
        for ctx in reversed(self._teardown_stack):
            with suppress(Exception):
                await ctx.__aexit__(None, None, None)
        self._teardown_stack.clear()
        self._page = None

    async def _launch_real_browser(self) -> None:
        # Imported lazily so importing this module never forces the
        # playwright import cost on runners that don't need browser_dom.
        from playwright.async_api import async_playwright

        pw_ctx = async_playwright()
        playwright = await pw_ctx.__aenter__()
        self._teardown_stack.append(pw_ctx)

        if self._capability.cdp_endpoint is not None:
            browser = await playwright.chromium.connect_over_cdp(
                self._capability.cdp_endpoint
            )
        else:
            browser = await playwright.chromium.launch(headless=True)
        # Browser owns context+pages; close it on exit and the rest follows.
        self._teardown_stack.append(_BrowserCloser(browser))

        # Reuse an existing page when attaching over CDP (the operator
        # may already have a tab they want driven); otherwise open a
        # fresh page on a new context for isolation.
        # ``Page`` is structurally compatible with ``PageLike`` but
        # mypy can't always prove it (Playwright's signatures use
        # ``Optional[float]`` defaults which don't match the Protocol's
        # ``...`` defaults bit-for-bit). Cast explicitly at the seam.
        if (
            self._capability.cdp_endpoint is not None
            and browser.contexts
            and browser.contexts[0].pages
        ):
            self._page = cast(PageLike, browser.contexts[0].pages[0])
        else:
            context = await browser.new_context()
            self._page = cast(PageLike, await context.new_page())

    @property
    def page(self) -> PageLike | None:
        """Current page, or ``None`` outside the async-context lifetime."""
        return self._page

    async def dispatch(
        self,
        hint: Mapping[str, Any],
        parameters: Mapping[str, str],
    ) -> BrowserDOMResult:
        """Execute one browser_dom hint against the active page.

        Substitutes parameters into ``url_pattern`` / ``selector`` /
        ``value`` first; if any reference is undeclared, raises
        :class:`KeyError` so the executor can fall through to the next
        tier. Otherwise navigates if needed, performs the action, and
        returns a structured :class:`BrowserDOMResult`. Action-level
        failures return ``ok=False`` rather than raising — same
        contract as :meth:`runner.mcp_client.MCPCallDispatcher.call`.
        """
        if self._page is None:
            return BrowserDOMResult(
                ok=False,
                action=str(hint.get("action", "<missing>")),
                selector=hint.get("selector"),
                url=None,
                content_text=None,
                duration_ms=0.0,
                error="dispatcher entered before action; call inside `async with`",
            )

        action = hint.get("action")
        if action not in _SUPPORTED_ACTIONS:
            raise BrowserDOMCallError(
                action=str(action),
                selector=hint.get("selector"),
                message=(
                    f"unsupported action {action!r}; expected one of "
                    f"{sorted(_SUPPORTED_ACTIONS)}"
                ),
            )

        # Parameter substitution. Missing-param raises KeyError up to
        # the executor, which matches the MCP dispatcher's contract.
        url_pattern = hint.get("url_pattern")
        selector = hint.get("selector")
        value = hint.get("value")
        if isinstance(url_pattern, str):
            url_pattern = substitute_parameters_in_string(url_pattern, parameters)
        if isinstance(selector, str):
            selector = substitute_parameters_in_string(selector, parameters)
        if isinstance(value, str):
            value = substitute_parameters_in_string(value, parameters)

        started = time.monotonic()
        try:
            content_text = await self._dispatch_action(
                action=action,
                url_pattern=url_pattern,
                selector=selector,
                value=value,
            )
        except BrowserDOMCallError:
            raise
        except Exception as exc:
            return BrowserDOMResult(
                ok=False,
                action=action,
                selector=selector,
                url=url_pattern,
                content_text=None,
                duration_ms=(time.monotonic() - started) * 1000.0,
                error=f"{type(exc).__name__}: {exc}",
            )

        # Capture a frame for the live-stream dashboard. Wrapped in
        # try/except because the run shouldn't fail if a screenshot
        # blip happens (e.g. page context was torn down by a redirect
        # mid-action). Frame capture is purely observability.
        if self._on_frame is not None:
            try:
                jpg = await self._page.screenshot(
                    type="jpeg", quality=self._frame_jpeg_quality
                )
                await self._on_frame(jpg)
            except Exception as exc:
                LOGGER.warning(
                    "browser_dom frame capture failed: %s", exc
                )

        return BrowserDOMResult(
            ok=True,
            action=action,
            selector=selector,
            url=self._page.url,
            content_text=content_text,
            duration_ms=(time.monotonic() - started) * 1000.0,
        )

    async def _dispatch_action(
        self,
        *,
        action: str,
        url_pattern: str | None,
        selector: str | None,
        value: str | None,
    ) -> str:
        """Action-by-action implementation; returns a content_text summary."""
        assert self._page is not None  # narrowed by dispatch()
        timeout = float(self._action_timeout_ms)

        if action == "navigate":
            target = value or url_pattern
            if not target:
                raise BrowserDOMCallError(
                    action=action,
                    selector=selector,
                    message="navigate requires either `value` or `url_pattern`",
                )
            await self._page.goto(target, timeout=timeout)
            return f"navigated to {target}"

        # All other actions need a target page; navigate first if the
        # synth gave us a URL pattern and we're not already there.
        if url_pattern and self._page.url != url_pattern:
            await self._page.goto(url_pattern, timeout=timeout)

        if action == "click":
            if not selector:
                raise BrowserDOMCallError(
                    action=action, selector=selector,
                    message="click requires `selector`",
                )
            await self._page.click(selector, timeout=timeout)
            return f"clicked {selector}"

        if action == "type":
            if not selector:
                raise BrowserDOMCallError(
                    action=action, selector=selector,
                    message="type requires `selector`",
                )
            if value is None:
                raise BrowserDOMCallError(
                    action=action, selector=selector,
                    message="type requires `value`",
                )
            await self._page.fill(selector, value, timeout=timeout)
            return f"typed into {selector}"

        if action == "submit":
            if not selector:
                raise BrowserDOMCallError(
                    action=action, selector=selector,
                    message="submit requires `selector`",
                )
            # Pressing Enter inside the targeted field submits the form
            # without depending on a separate Submit button selector —
            # matches what synth's prose typically describes.
            await self._page.press(selector, "Enter", timeout=timeout)
            return f"submitted via {selector}"

        if action == "scroll":
            # Page-level scroll. Future synth could add `value` for a
            # scroll amount; for now the dispatcher scrolls one viewport
            # down, which matches the most common synth-emitted hint.
            await self._page.evaluate(
                "window.scrollBy({top: window.innerHeight, behavior: 'instant'})"
            )
            return "scrolled one viewport"

        # Defensive — _SUPPORTED_ACTIONS membership was checked above.
        raise BrowserDOMCallError(
            action=action, selector=selector,
            message="unreachable: action passed enum check but no branch matched",
        )


class _BrowserCloser:
    """Adapt a Playwright ``Browser`` to the async-context-manager protocol.

    The teardown stack stores async-context-managers; ``Browser`` itself
    is not one but exposes ``close()``. Wrapping it lets us treat
    everything in ``_teardown_stack`` uniformly.
    """

    def __init__(self, browser: Any) -> None:
        self._browser = browser

    async def __aenter__(self) -> _BrowserCloser:
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self._browser.close()


class PageProvider(Protocol):
    """Test-only callable that yields a :class:`PageLike` async-context.

    Production code relies on the real Playwright launch path; tests
    inject a fake provider so CI never starts a real browser.
    """

    def __call__(self) -> AbstractAsyncContextManager[PageLike]: ...
