"""Tests for ``runner.destructive`` — shared destructive keyword matcher.

The JSON list at ``contracts/destructive_keywords.json`` is the single source
of truth shared with the synthesizer. These tests lock both the matching
semantics (word-boundary, case-insensitive, unicode-normalised) and the list
contents (14 entries, all lowercase).
"""

from __future__ import annotations

from typing import Any

import pytest
from runner.destructive import load_destructive_keywords, matches_destructive_keyword

EXPECTED_KEYWORDS = frozenset(
    {
        "send",
        "submit",
        "delete",
        "remove",
        "publish",
        "post",
        "purchase",
        "pay",
        "buy",
        "transfer",
        "confirm",
        "authorize",
        "approve",
        "share",
    }
)


def test_load_destructive_keywords_returns_frozenset() -> None:
    keywords = load_destructive_keywords()
    assert isinstance(keywords, frozenset)
    assert keywords == EXPECTED_KEYWORDS


def test_load_destructive_keywords_is_cached() -> None:
    assert load_destructive_keywords() is load_destructive_keywords()


def test_load_destructive_keywords_all_lowercase() -> None:
    for word in load_destructive_keywords():
        assert word == word.lower()


@pytest.mark.parametrize("keyword", sorted(EXPECTED_KEYWORDS))
def test_each_keyword_matches_as_word_boundary(keyword: str) -> None:
    assert matches_destructive_keyword(keyword) is True
    assert matches_destructive_keyword(f"please {keyword} it") is True
    assert matches_destructive_keyword(f"{keyword}.") is True
    assert matches_destructive_keyword(f"({keyword})") is True


@pytest.mark.parametrize(
    "text",
    [
        "sender",
        "senders",
        "postpone",
        "posthumous",
        "submitted",
        "submittal",
        "removed",
        "remover",
        "deleted",
        "deletion",
        "published",
        "payment",
        "buyer",
        "shared",
        "confirmed",
        "authorized",
        "approval",
        "transferred",
        "purchased",
    ],
)
def test_substring_does_not_match_word_boundary(text: str) -> None:
    assert matches_destructive_keyword(text) is False


def test_multi_word_text_matches() -> None:
    assert matches_destructive_keyword("Click Send") is True
    assert matches_destructive_keyword("Please click the submit button") is True


def test_uppercase_matches_case_insensitively() -> None:
    assert matches_destructive_keyword("SEND") is True
    assert matches_destructive_keyword("Delete") is True
    assert matches_destructive_keyword("PURCHASE NOW") is True


def test_empty_string_returns_false() -> None:
    assert matches_destructive_keyword("") is False


def test_whitespace_only_returns_false() -> None:
    assert matches_destructive_keyword("   \t\n  ") is False


def test_leading_and_trailing_whitespace_still_matches() -> None:
    assert matches_destructive_keyword("   send   ") is True
    assert matches_destructive_keyword("\n\tsubmit\n") is True


def test_punctuation_surrounding_keyword_matches() -> None:
    assert matches_destructive_keyword("Send!") is True
    assert matches_destructive_keyword('"Delete"') is True
    assert matches_destructive_keyword("confirm?") is True
    assert matches_destructive_keyword("...post...") is True


def test_none_input_raises_type_error() -> None:
    with pytest.raises(TypeError):
        matches_destructive_keyword(None)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", [123, 1.5, [], {}, b"send", object()])
def test_non_string_input_raises_type_error(bad: Any) -> None:
    with pytest.raises(TypeError):
        matches_destructive_keyword(bad)


def test_unicode_fullwidth_normalises() -> None:
    # U+FF33 U+FF25 U+FF2E U+FF24 → "SEND" under NFKC
    fullwidth = "\uff33\uff25\uff2e\uff24"
    assert matches_destructive_keyword(fullwidth) is True


def test_unicode_non_breaking_whitespace() -> None:
    # Non-breaking space (U+00A0) surrounding the keyword.
    text = "click\u00a0send\u00a0now"
    assert matches_destructive_keyword(text) is True


def test_non_destructive_sentence_does_not_match() -> None:
    assert matches_destructive_keyword("Open the inbox and read the message") is False
    assert matches_destructive_keyword("Navigate to the preferences pane") is False


def test_multiple_keywords_in_one_text_still_true() -> None:
    assert matches_destructive_keyword("confirm and then send") is True
