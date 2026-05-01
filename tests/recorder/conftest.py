"""Shared pytest fixtures and collection hooks for the recorder test suite.

Tests marked ``@pytest.mark.macos`` exercise real macOS frameworks
(Accessibility, CGEventTap, Screen Recording) that require permissions a
headless / sandboxed environment does not have.  We skip them by default and
opt in with ``TRACE_RUN_MACOS_TESTS=1`` — that way the Ralph loop stays
green without ever silently passing a broken integration test.
"""

from __future__ import annotations

import os

import pytest


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if os.environ.get("TRACE_RUN_MACOS_TESTS") == "1":
        return
    skip_macos = pytest.mark.skip(
        reason="macOS integration test; set TRACE_RUN_MACOS_TESTS=1 to run."
    )
    for item in items:
        if "macos" in item.keywords:
            item.add_marker(skip_macos)
