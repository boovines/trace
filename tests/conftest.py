"""Root pytest conftest.

Hosts CLI options that must be registered before any subdirectory conftest
runs — pytest's ``pytest_addoption`` hook must live at the test-tree root.
"""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register ``--run-live-input`` to opt into the live CGEventPost tests."""
    parser.addoption(
        "--run-live-input",
        action="store_true",
        default=False,
        help=(
            "Run tests marked @pytest.mark.live_input. These post real events "
            "via CGEventPost — only enable on a disposable test machine."
        ),
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip ``live_input`` tests unless the caller passes ``--run-live-input``."""
    if config.getoption("--run-live-input"):
        return
    skip_marker = pytest.mark.skip(
        reason="live_input tests require --run-live-input (posts real events)"
    )
    for item in items:
        if "live_input" in item.keywords:
            item.add_marker(skip_marker)
