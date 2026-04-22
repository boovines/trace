"""Shared pytest fixtures for the runner module.

Two invariants this file enforces for every test:

1. ``TRACE_ALLOW_LIVE`` is force-unset before every test, so a leaked
   environment variable from a developer shell can never cause a Ralph or CI
   run to instantiate a live adapter. Tests that need live mode must opt in
   explicitly (see the ``live_mode_allowed`` fixture below).
2. An ``anthropic_mock`` fixture backed by respx is available, so runner tests
   never make real network calls to ``api.anthropic.com``.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
import respx
from runner.safety import LIVE_MODE_ENV_VAR


@pytest.fixture(autouse=True)
def _force_live_mode_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Guarantee every runner test starts with TRACE_ALLOW_LIVE unset.

    Autouse so a developer cannot accidentally skip it. Tests that need live
    mode must use the ``live_mode_allowed`` fixture, which sets the flag AFTER
    this one has cleared it.
    """
    monkeypatch.delenv(LIVE_MODE_ENV_VAR, raising=False)


@pytest.fixture
def live_mode_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Opt-in: enable TRACE_ALLOW_LIVE=1 for a single test.

    Used only by tests that specifically exercise the safety gate OR live
    adapter instantiation. No test should mix this with real CGEventPost calls
    unless it is also marked ``@pytest.mark.live_input``.
    """
    monkeypatch.setenv(LIVE_MODE_ENV_VAR, "1")


@pytest.fixture
def anthropic_mock() -> Iterator[respx.MockRouter]:
    """Mock ``api.anthropic.com`` for any test that might otherwise hit it.

    Tests that need specific canned responses should add routes via
    ``anthropic_mock.post("/v1/messages").respond(...)``. Leaving the mock
    un-configured will cause any accidental outbound request to fail loudly
    rather than silently hit the real API.
    """
    with respx.mock(base_url="https://api.anthropic.com", assert_all_called=False) as router:
        yield router
