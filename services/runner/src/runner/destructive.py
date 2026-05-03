"""Shared destructive-keyword matcher.

The source of truth for the destructive-keyword list is
``contracts/destructive_keywords.json`` at the repo root. Both the synthesizer
(which flags ⚠️ steps during skill authoring) and the runner (harness-layer
pre-action gate, see X-015) import from the same file so the two sides cannot
drift.

This module is middle layer of the three-layer destructive-action defense:

1. The skill's ⚠️ markers become per-step prompt instructions.
2. *This layer:* the harness inspects target AX-element labels and step text
   against the shared keyword list and forces a confirmation on match.
3. A hard per-run token budget and per-minute action rate limit prevent
   runaway loops.

``matches_destructive_keyword`` does **word-boundary**, case-insensitive
matching so ``"sender"`` does not trip ``"send"`` and ``"submitted"`` does not
trip ``"submit"``. Text is unicode-normalised (NFKC) before matching so that
fullwidth characters and non-breaking whitespace behave like their ASCII
equivalents.
"""

from __future__ import annotations

import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Final

_KEYWORDS_PATH: Final[Path] = (
    Path(__file__).resolve().parents[4] / "contracts" / "destructive_keywords.json"
)


@lru_cache(maxsize=1)
def load_destructive_keywords() -> frozenset[str]:
    """Return the locked destructive-keyword list as a frozenset.

    Cached because the list is immutable at runtime and the harness-layer gate
    calls this before every action. Returns a ``frozenset[str]`` so callers
    get O(1) membership and cannot mutate the shared list.
    """
    raw = json.loads(_KEYWORDS_PATH.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(
            f"{_KEYWORDS_PATH} must contain a JSON array of lowercase strings"
        )
    return frozenset(str(word) for word in raw)


@lru_cache(maxsize=1)
def _compiled_pattern() -> re.Pattern[str]:
    keywords = sorted(load_destructive_keywords(), key=len, reverse=True)
    escaped = "|".join(re.escape(word) for word in keywords)
    return re.compile(rf"\b(?:{escaped})\b", re.IGNORECASE)


def matches_destructive_keyword(text: str) -> bool:
    """Return True iff ``text`` contains any destructive keyword.

    Matching is case-insensitive and word-boundary aware (``\\b``), so
    ``"sender"`` does NOT match ``"send"`` and ``"postpone"`` does NOT match
    ``"post"``. Text is NFKC-normalised first to collapse fullwidth
    characters and exotic whitespace onto their ASCII form.

    Raises ``TypeError`` if ``text`` is not a string — silently returning
    False on a programming error would hide bugs in callers.
    """
    if not isinstance(text, str):
        raise TypeError(
            f"matches_destructive_keyword expected str, got {type(text).__name__}"
        )
    if not text:
        return False
    normalised = unicodedata.normalize("NFKC", text)
    return _compiled_pattern().search(normalised) is not None
