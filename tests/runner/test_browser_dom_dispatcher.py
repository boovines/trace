"""Tests for ``runner.browser_dom_dispatcher`` — Step 4.1 commit 2.

Covers:

* ``substitute_parameters_in_string`` happy path + missing-param raises
  ``KeyError``.
* ``BrowserDOMDispatcher`` lifecycle with a fake page provider — the
  context manager opens and closes cleanly.
* Each ``action`` (navigate / click / type / submit / scroll) dispatches
  the right ``PageLike`` call, returns a populated
  :class:`BrowserDOMResult`, and substitutes ``{param}`` references.
* Implicit navigation: a non-navigate action with ``url_pattern`` set
  navigates first when the page is on a different URL, but skips the
  goto when the page already matches.
* Action-level failures (page raises) yield ``ok=False`` with an
  error string rather than bubbling.
* Hint-shape failures (unknown action, missing required field, missing
  param) raise ``BrowserDOMCallError`` / ``KeyError`` so the executor
  can fall through to the next tier.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from typing import Any

import pytest

from runner.browser_dom_dispatcher import (
    BrowserDOMCallError,
    BrowserDOMDispatcher,
    BrowserDOMResult,
    PageLike,
    substitute_parameters_in_string,
)
from runner.browser_dom_probe import BrowserDOMCapability

_FAKE_CAPABILITY = BrowserDOMCapability(
    cdp_endpoint=None, executable_path="/fake/chromium"
)


# --- substitute_parameters_in_string -------------------------------------


def test_substitute_replaces_single_param() -> None:
    out = substitute_parameters_in_string(
        "https://x/{slug}", {"slug": "inbox"}
    )
    assert out == "https://x/inbox"


def test_substitute_replaces_multiple_params() -> None:
    out = substitute_parameters_in_string(
        "{a}/{b}", {"a": "1", "b": "2"}
    )
    assert out == "1/2"


def test_substitute_passes_strings_without_refs() -> None:
    out = substitute_parameters_in_string("plain text", {"x": "y"})
    assert out == "plain text"


def test_substitute_raises_keyerror_on_missing_param() -> None:
    with pytest.raises(KeyError) as exc_info:
        substitute_parameters_in_string(
            "{recipient}", {"sender": "alice"}
        )
    assert exc_info.value.args[0] == "recipient"


# --- _FakePage ------------------------------------------------------------


class _FakePage:
    """Records action calls. Implements just enough of :class:`PageLike`."""

    def __init__(self, *, initial_url: str = "about:blank") -> None:
        self.url = initial_url
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        # Optional per-method side-effects: callable invoked when the
        # corresponding method is called. Lets a single test simulate
        # a page that raises on, say, click.
        self.click_side_effect: Any = None

    async def goto(
        self,
        url: str,
        *,
        timeout: float | None = None,
        wait_until: str | None = None,
    ) -> None:
        self.calls.append(("goto", (url,), {"timeout": timeout}))
        self.url = url

    async def click(self, selector: str, *, timeout: float | None = None) -> None:
        self.calls.append(("click", (selector,), {"timeout": timeout}))
        if self.click_side_effect is not None:
            raise self.click_side_effect

    async def fill(
        self, selector: str, value: str, *, timeout: float | None = None
    ) -> None:
        self.calls.append(("fill", (selector, value), {"timeout": timeout}))

    async def press(
        self, selector: str, key: str, *, timeout: float | None = None
    ) -> None:
        self.calls.append(("press", (selector, key), {"timeout": timeout}))

    async def evaluate(self, expression: str) -> Any:
        self.calls.append(("evaluate", (expression,), {}))
        return None

    async def title(self) -> str:
        return "fake page"


def _provider_for(page: PageLike) -> Any:
    @asynccontextmanager
    async def _provider() -> AsyncIterator[PageLike]:
        yield page

    return _provider


# --- BrowserDOMDispatcher lifecycle --------------------------------------


@pytest.mark.asyncio
async def test_dispatcher_opens_and_closes_with_fake_provider() -> None:
    page = _FakePage()
    dispatcher = BrowserDOMDispatcher(
        _FAKE_CAPABILITY, page_provider=_provider_for(page)
    )
    async with dispatcher as d:
        assert d.page is page
    # After exit, page reference should be released.
    assert dispatcher.page is None


@pytest.mark.asyncio
async def test_dispatch_outside_context_returns_error() -> None:
    page = _FakePage()
    dispatcher = BrowserDOMDispatcher(
        _FAKE_CAPABILITY, page_provider=_provider_for(page)
    )
    # No ``async with`` — page never set.
    result = await dispatcher.dispatch(
        {"tier": "browser_dom", "action": "click", "selector": "x"},
        parameters={},
    )
    assert isinstance(result, BrowserDOMResult)
    assert result.ok is False
    assert result.error is not None
    assert "entered before" in result.error


# --- per-action dispatch ---------------------------------------------------


async def _dispatch_one(
    page: _FakePage,
    hint: Mapping[str, Any],
    parameters: Mapping[str, str] | None = None,
) -> BrowserDOMResult:
    dispatcher = BrowserDOMDispatcher(
        _FAKE_CAPABILITY, page_provider=_provider_for(page)
    )
    async with dispatcher:
        return await dispatcher.dispatch(hint, parameters or {})


@pytest.mark.asyncio
async def test_navigate_uses_value_first() -> None:
    page = _FakePage()
    result = await _dispatch_one(
        page,
        {
            "tier": "browser_dom",
            "action": "navigate",
            "value": "https://x.example/{slug}",
        },
        {"slug": "inbox"},
    )
    assert result.ok is True
    assert result.action == "navigate"
    assert result.url == "https://x.example/inbox"
    assert result.content_text == "navigated to https://x.example/inbox"
    assert page.calls == [
        ("goto", ("https://x.example/inbox",), {"timeout": 10000.0}),
    ]


@pytest.mark.asyncio
async def test_navigate_falls_back_to_url_pattern() -> None:
    page = _FakePage()
    result = await _dispatch_one(
        page,
        {
            "tier": "browser_dom",
            "action": "navigate",
            "url_pattern": "https://x.example/inbox",
        },
    )
    assert result.ok is True
    assert page.calls[0][1] == ("https://x.example/inbox",)


@pytest.mark.asyncio
async def test_navigate_without_target_raises() -> None:
    page = _FakePage()
    with pytest.raises(BrowserDOMCallError, match="requires either"):
        await _dispatch_one(
            page,
            {"tier": "browser_dom", "action": "navigate"},
        )


@pytest.mark.asyncio
async def test_click_navigates_first_when_url_pattern_set() -> None:
    page = _FakePage(initial_url="https://other/")
    result = await _dispatch_one(
        page,
        {
            "tier": "browser_dom",
            "action": "click",
            "url_pattern": "https://x.example/inbox",
            "selector": "button.send",
        },
    )
    assert result.ok is True
    assert result.action == "click"
    assert [c[0] for c in page.calls] == ["goto", "click"]
    assert page.calls[0][1] == ("https://x.example/inbox",)
    assert page.calls[1][1] == ("button.send",)


@pytest.mark.asyncio
async def test_click_skips_goto_when_already_on_url() -> None:
    page = _FakePage(initial_url="https://x.example/inbox")
    await _dispatch_one(
        page,
        {
            "tier": "browser_dom",
            "action": "click",
            "url_pattern": "https://x.example/inbox",
            "selector": "button.send",
        },
    )
    # Only the click should have happened — no implicit navigation.
    assert [c[0] for c in page.calls] == ["click"]


@pytest.mark.asyncio
async def test_click_without_selector_raises() -> None:
    page = _FakePage()
    with pytest.raises(BrowserDOMCallError, match="requires `selector`"):
        await _dispatch_one(
            page, {"tier": "browser_dom", "action": "click"}
        )


@pytest.mark.asyncio
async def test_type_fills_field_with_substituted_value() -> None:
    page = _FakePage()
    result = await _dispatch_one(
        page,
        {
            "tier": "browser_dom",
            "action": "type",
            "selector": "input[name='to']",
            "value": "{recipient}",
        },
        {"recipient": "alice@example.com"},
    )
    assert result.ok is True
    assert page.calls[0] == (
        "fill",
        ("input[name='to']", "alice@example.com"),
        {"timeout": 10000.0},
    )


@pytest.mark.asyncio
async def test_type_without_value_raises() -> None:
    page = _FakePage()
    with pytest.raises(BrowserDOMCallError, match="requires `value`"):
        await _dispatch_one(
            page,
            {"tier": "browser_dom", "action": "type", "selector": "input"},
        )


@pytest.mark.asyncio
async def test_submit_presses_enter_in_field() -> None:
    page = _FakePage()
    result = await _dispatch_one(
        page,
        {
            "tier": "browser_dom",
            "action": "submit",
            "selector": "form#reply",
        },
    )
    assert result.ok is True
    assert page.calls[0] == (
        "press",
        ("form#reply", "Enter"),
        {"timeout": 10000.0},
    )


@pytest.mark.asyncio
async def test_scroll_evaluates_one_viewport() -> None:
    page = _FakePage()
    result = await _dispatch_one(
        page,
        {"tier": "browser_dom", "action": "scroll"},
    )
    assert result.ok is True
    assert page.calls[0][0] == "evaluate"
    assert "scrollBy" in page.calls[0][1][0]


# --- failure paths -------------------------------------------------------


@pytest.mark.asyncio
async def test_unknown_action_raises_call_error() -> None:
    page = _FakePage()
    with pytest.raises(BrowserDOMCallError, match="unsupported action"):
        await _dispatch_one(
            page, {"tier": "browser_dom", "action": "wave_hands"}
        )


@pytest.mark.asyncio
async def test_missing_param_raises_keyerror() -> None:
    page = _FakePage()
    with pytest.raises(KeyError, match="recipient"):
        await _dispatch_one(
            page,
            {
                "tier": "browser_dom",
                "action": "type",
                "selector": "input",
                "value": "{recipient}",
            },
            {},  # no params declared
        )


@pytest.mark.asyncio
async def test_action_failure_returns_ok_false() -> None:
    page = _FakePage()
    page.click_side_effect = RuntimeError("Timeout: selector not found")
    result = await _dispatch_one(
        page,
        {"tier": "browser_dom", "action": "click", "selector": "button.send"},
    )
    assert result.ok is False
    assert result.action == "click"
    assert result.selector == "button.send"
    assert result.error is not None
    assert "Timeout" in result.error
    assert result.duration_ms >= 0.0
