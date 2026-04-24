"""Placeholder test — replaced when R-002 lands real schema tests.

Ensures `uv run pytest tests/recorder/` exits 0 on a fresh scaffold so the
R-001 quality gate passes. Also proves the `recorder` package is importable
from the installed workspace member.
"""

from __future__ import annotations

import recorder


def test_recorder_importable() -> None:
    assert recorder.__version__ == "0.1.0"
