"""Tests for :mod:`synthesizer.slug` — S-009.

Covers:

* :func:`slugify` across punctuation, accents, leading digits, emoji, and
  edge cases (empty, too-short, too-long, only-special-chars).
* :func:`resolve_unique_slug` collision handling (numeric suffixes only).
* :func:`validate_user_slug` accepts valid and rejects invalid patterns with
  human-readable messages.
* Directory-safety: slug outputs are always safe macOS directory names.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from synthesizer.slug import (
    SLUG_MAX_LEN,
    SLUG_MIN_LEN,
    SLUG_PATTERN,
    SLUG_REGEX,
    SlugError,
    resolve_unique_slug,
    slugify,
    validate_user_slug,
)

# ---------------------------------------------------------------------------
# slugify
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Gmail Reply", "gmail_reply"),
        ("gmail reply", "gmail_reply"),
        ("GMAIL REPLY", "gmail_reply"),
        # Punctuation
        ("Jake's Email!", "jake_s_email"),
        ("hello.world", "hello_world"),
        ("foo-bar-baz", "foo_bar_baz"),
        ("under_score_ok", "under_score_ok"),
        # Multiple separators collapse
        ("foo   bar", "foo_bar"),
        ("foo---bar", "foo_bar"),
        ("foo  -  bar", "foo_bar"),
        # Accents transliterated
        ("Émilie's Notes", "emilie_s_notes"),
        ("Café Menu", "cafe_menu"),
        ("Résumé Template", "resume_template"),
        # Leading digits stripped
        ("123 Notes Daily", "notes_daily"),
        ("42abc", "abc"),
        # Emoji dropped (English-only slug)
        ("🎯 heads down", "heads_down"),
        ("🚀🚀launch", "launch"),
        # Trailing junk stripped
        ("hello!!!", "hello"),
        ("_foo_bar_", "foo_bar"),
        # Digits and underscores inside the body preserved
        ("skill 2 variant", "skill_2_variant"),
        ("workflow_42", "workflow_42"),
    ],
)
def test_slugify_happy_path(name: str, expected: str) -> None:
    assert slugify(name) == expected


def test_slugify_truncates_to_max_length() -> None:
    long_name = "a" * 200
    result = slugify(long_name)
    assert len(result) == SLUG_MAX_LEN
    assert result == "a" * SLUG_MAX_LEN


def test_slugify_truncation_strips_trailing_underscore() -> None:
    # Build a 40-char candidate where position 40 would be an underscore; after
    # truncation the trailing '_' must be stripped so the slug still matches
    # the pattern.
    name = "a" * 39 + " " + "bbbbbbbbbb"
    result = slugify(name)
    assert not result.endswith("_")
    assert SLUG_REGEX.match(result)


@pytest.mark.parametrize(
    "name",
    [
        "",
        "   ",
        "!!!",
        "🎯",
        "12",
        "123",  # all digits → no leading letter → empty after trim
        "__",
        "a",  # too short
        "ab",  # too short
    ],
)
def test_slugify_raises_slug_error_on_empty_or_too_short(name: str) -> None:
    with pytest.raises(SlugError):
        slugify(name)


def test_slugify_raises_on_non_string() -> None:
    with pytest.raises(SlugError):
        slugify(123)  # type: ignore[arg-type]


def test_slugify_output_always_matches_pattern() -> None:
    """Any successful slugify output matches the locked pattern."""
    for name in [
        "Gmail Reply",
        "Émilie's 2nd Workflow",
        "🎯 heads down",
        "foo---bar",
        "A" * 100,
    ]:
        assert SLUG_REGEX.match(slugify(name))


def test_slugify_output_is_safe_macos_directory_name() -> None:
    """macOS reserved/problematic chars must never appear in slug output."""
    bad_chars = set("/\\:*?\"<>|")
    for name in [
        "some/path/like name",
        "a:b:c",
        "foo\\bar",
        'with "quotes"',
        "pipe|name",
        "<angle>",
    ]:
        slug = slugify(name)
        assert not (bad_chars & set(slug))
        # No leading dot (hidden on Unix, problematic on Finder).
        assert not slug.startswith(".")
        # No trailing dot (Windows would reject, macOS tolerates).
        assert not slug.endswith(".")


# ---------------------------------------------------------------------------
# resolve_unique_slug
# ---------------------------------------------------------------------------


def test_resolve_unique_slug_returns_base_when_free(tmp_path: Path) -> None:
    assert resolve_unique_slug("gmail_reply", tmp_path) == "gmail_reply"


def test_resolve_unique_slug_returns_base_when_dir_missing(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist"
    assert resolve_unique_slug("gmail_reply", missing) == "gmail_reply"


def test_resolve_unique_slug_with_one_collision(tmp_path: Path) -> None:
    (tmp_path / "gmail_reply").mkdir()
    assert resolve_unique_slug("gmail_reply", tmp_path) == "gmail_reply_2"


def test_resolve_unique_slug_with_five_collisions_produces_base_6(
    tmp_path: Path,
) -> None:
    (tmp_path / "notes_daily").mkdir()
    for i in range(2, 6):
        (tmp_path / f"notes_daily_{i}").mkdir()
    assert resolve_unique_slug("notes_daily", tmp_path) == "notes_daily_6"


def test_resolve_unique_slug_skips_files_too(tmp_path: Path) -> None:
    """A regular file at skills_dir/slug blocks the slug too."""
    (tmp_path / "foo").write_text("not a dir")
    assert resolve_unique_slug("foo", tmp_path) == "foo_2"


def test_resolve_unique_slug_rejects_invalid_base(tmp_path: Path) -> None:
    with pytest.raises(SlugError):
        resolve_unique_slug("Bad-Base", tmp_path)
    with pytest.raises(SlugError):
        resolve_unique_slug("9abc", tmp_path)


def test_resolve_unique_slug_hits_max_length_boundary(tmp_path: Path) -> None:
    """If the numeric suffix would overflow SLUG_MAX_LEN, raise SlugError."""
    base = "a" * SLUG_MAX_LEN  # 40 chars
    (tmp_path / base).mkdir()
    with pytest.raises(SlugError):
        resolve_unique_slug(base, tmp_path)


# ---------------------------------------------------------------------------
# validate_user_slug
# ---------------------------------------------------------------------------


def test_validate_user_slug_accepts_valid(tmp_path: Path) -> None:
    ok, slug = validate_user_slug("my_workflow", tmp_path)
    assert ok is True
    assert slug == "my_workflow"


def test_validate_user_slug_trims_surrounding_whitespace(tmp_path: Path) -> None:
    ok, slug = validate_user_slug("  gmail_reply  ", tmp_path)
    assert ok is True
    assert slug == "gmail_reply"


def test_validate_user_slug_rejects_empty(tmp_path: Path) -> None:
    ok, reason = validate_user_slug("", tmp_path)
    assert ok is False
    assert "empty" in reason.lower()


def test_validate_user_slug_rejects_too_short(tmp_path: Path) -> None:
    ok, reason = validate_user_slug("ab", tmp_path)
    assert ok is False
    assert "short" in reason.lower()
    assert str(SLUG_MIN_LEN) in reason


def test_validate_user_slug_rejects_too_long(tmp_path: Path) -> None:
    ok, reason = validate_user_slug("a" * (SLUG_MAX_LEN + 1), tmp_path)
    assert ok is False
    assert "long" in reason.lower()
    assert str(SLUG_MAX_LEN) in reason


def test_validate_user_slug_rejects_uppercase(tmp_path: Path) -> None:
    ok, reason = validate_user_slug("Gmail_Reply", tmp_path)
    assert ok is False
    assert "lowercase" in reason.lower()


def test_validate_user_slug_rejects_leading_digit(tmp_path: Path) -> None:
    ok, reason = validate_user_slug("9abc", tmp_path)
    assert ok is False
    assert "start" in reason.lower()


def test_validate_user_slug_rejects_leading_underscore(tmp_path: Path) -> None:
    ok, reason = validate_user_slug("_abc", tmp_path)
    assert ok is False
    assert "start" in reason.lower()


def test_validate_user_slug_rejects_hyphen(tmp_path: Path) -> None:
    ok, reason = validate_user_slug("foo-bar", tmp_path)
    assert ok is False
    assert "-" in reason or "disallowed" in reason.lower()


def test_validate_user_slug_rejects_special_chars(tmp_path: Path) -> None:
    ok, reason = validate_user_slug("foo.bar", tmp_path)
    assert ok is False
    assert "disallowed" in reason.lower() or "." in reason


def test_validate_user_slug_rejects_collision(tmp_path: Path) -> None:
    (tmp_path / "gmail_reply").mkdir()
    ok, reason = validate_user_slug("gmail_reply", tmp_path)
    assert ok is False
    assert "exists" in reason.lower() or "already" in reason.lower()


def test_validate_user_slug_accepts_when_skills_dir_missing(tmp_path: Path) -> None:
    missing = tmp_path / "nope"
    ok, slug = validate_user_slug("gmail_reply", missing)
    assert ok is True
    assert slug == "gmail_reply"


def test_validate_user_slug_rejects_non_string(tmp_path: Path) -> None:
    ok, reason = validate_user_slug(123, tmp_path)  # type: ignore[arg-type]
    assert ok is False
    assert "string" in reason.lower()


def test_validate_user_slug_accepts_minimum_length(tmp_path: Path) -> None:
    ok, slug = validate_user_slug("abc", tmp_path)
    assert ok is True
    assert slug == "abc"


def test_validate_user_slug_accepts_maximum_length(tmp_path: Path) -> None:
    longest = "a" + "b" * (SLUG_MAX_LEN - 1)
    assert len(longest) == SLUG_MAX_LEN
    ok, slug = validate_user_slug(longest, tmp_path)
    assert ok is True
    assert slug == longest


def test_slug_pattern_regex_consistency() -> None:
    """The exported pattern string must compile to the exported regex."""
    assert re.compile(SLUG_PATTERN).pattern == SLUG_REGEX.pattern
