"""Shared pytest fixtures for the synthesizer test suite.

Every LLM-touching test uses the ``anthropic_mock`` fixture so no test ever
reaches ``api.anthropic.com``. The fixture is a respx router scoped to the
Anthropic base URL; tests register routes on it to script responses.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest
import respx


@pytest.fixture(autouse=True)
def _force_fake_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every synthesizer test runs with fake-mode LLM + a dummy API key.

    S-006 adds the client that honors ``TRACE_LLM_FAKE_MODE``; setting it here
    in S-001 is a forward-compatible guard. ``ANTHROPIC_API_KEY`` gets a dummy
    value so constructors that only validate presence don't fail; any real HTTP
    call is still blocked by ``anthropic_mock`` when that fixture is in scope.
    """
    monkeypatch.setenv("TRACE_LLM_FAKE_MODE", "1")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-dummy-key")
    # Isolate on-disk writes from the user's real profile.
    monkeypatch.setenv("TRACE_PROFILE", "dev")


@pytest.fixture
def anthropic_mock() -> Iterator[respx.MockRouter]:
    """A respx router that intercepts every call to api.anthropic.com.

    Usage::

        def test_thing(anthropic_mock):
            anthropic_mock.post("/v1/messages").respond(
                200, json={"id": "msg_x", "content": [...]}
            )

    ``assert_all_called=False`` lets tests register optional routes without
    failing teardown; when a test needs exhaustiveness it can set the flag on
    the returned router.
    """
    with respx.mock(
        base_url="https://api.anthropic.com",
        assert_all_called=False,
    ) as router:
        yield router


@pytest.fixture(autouse=True)
def _no_stray_anthropic_calls(anthropic_mock: respx.MockRouter) -> None:
    """Belt-and-suspenders: install the mock router on every test so an
    unmocked call raises rather than silently hitting the network. Tests that
    want explicit control can still use the ``anthropic_mock`` fixture to
    register routes.
    """
    # Touching anthropic_mock installs the respx transport for the duration of
    # the test; assertion happens implicitly because respx raises on any
    # unmatched request when there is no passthrough.
    assert anthropic_mock is not None
    # Ensure we did not accidentally leave an ANTHROPIC_API_KEY pointing at a
    # real account (ci guard).
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    assert key.startswith("test-") or key == "", (
        "ANTHROPIC_API_KEY should be a test-only value during the test suite"
    )
