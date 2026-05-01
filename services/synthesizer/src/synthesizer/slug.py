"""Slug generation, collision resolution, and user-supplied slug validation.

Skill slugs are filesystem-safe, URL-safe, and uniquely identify a skill in
``~/Library/Application Support/Trace[-dev]/skills/<slug>/``. The locked
contract (``contracts/skill-meta.schema.json``) pins them to the pattern
``^[a-z][a-z0-9_]{2,39}$`` — 3-40 characters, first char is a letter,
remaining chars are lowercase letters, digits, or underscores.

Public API:

* :func:`slugify` — derive a slug from a skill name.
* :func:`resolve_unique_slug` — append ``_2``, ``_3``, ... as needed to avoid
  collisions with existing skill directories.
* :func:`validate_user_slug` — check whether a user-supplied slug is legal and
  available.
* :class:`SlugError` — raised when slugification cannot produce a valid result.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

__all__ = [
    "SLUG_MAX_LEN",
    "SLUG_MIN_LEN",
    "SLUG_PATTERN",
    "SLUG_REGEX",
    "SlugError",
    "resolve_unique_slug",
    "slugify",
    "validate_user_slug",
]

SLUG_PATTERN = r"^[a-z][a-z0-9_]{2,39}$"
SLUG_REGEX = re.compile(SLUG_PATTERN)
SLUG_MIN_LEN = 3
SLUG_MAX_LEN = 40


class SlugError(ValueError):
    """Raised when a name cannot be turned into a valid slug."""


def _transliterate(name: str) -> str:
    """Best-effort ASCII fold.

    Uses NFKD normalization to split combining marks off their base characters,
    then drops any codepoint that is not plain ASCII. Emoji and other non-Latin
    scripts are dropped entirely — this is English-only by design (see CLAUDE.md
    and the PRD non-goal).
    """
    normalized = unicodedata.normalize("NFKD", name)
    return normalized.encode("ascii", "ignore").decode("ascii")


def slugify(name: str) -> str:
    """Derive a slug matching :data:`SLUG_PATTERN` from a human-readable name.

    Transformations (in order):

    1. Best-effort transliterate to ASCII (accents stripped, emoji dropped).
    2. Lowercase.
    3. Replace any run of whitespace, hyphens, dots, or other non-alphanumeric
       characters with a single underscore.
    4. Strip leading characters that are not lowercase letters (the final slug
       must start with ``[a-z]``).
    5. Strip trailing underscores.
    6. Truncate to :data:`SLUG_MAX_LEN` characters.

    Raises :class:`SlugError` if the result is shorter than
    :data:`SLUG_MIN_LEN` characters or does not match :data:`SLUG_PATTERN`.
    """
    if not isinstance(name, str):  # defensive: caller handed us the wrong type
        raise SlugError(f"slugify requires a string, got {type(name).__name__}")

    ascii_name = _transliterate(name).lower()
    # Collapse any run of characters that are NOT [a-z0-9_] into a single underscore.
    collapsed = re.sub(r"[^a-z0-9_]+", "_", ascii_name)
    # Collapse consecutive underscores (can arise from adjacent non-alnum chars).
    collapsed = re.sub(r"_+", "_", collapsed)
    # Strip leading characters until we hit [a-z]. Digits and underscores at the
    # start are dropped because the slug must begin with a letter.
    trimmed = re.sub(r"^[^a-z]+", "", collapsed)
    # Strip trailing underscores so "Gmail Reply " doesn't become "gmail_reply_".
    trimmed = trimmed.rstrip("_")

    if len(trimmed) > SLUG_MAX_LEN:
        trimmed = trimmed[:SLUG_MAX_LEN].rstrip("_")

    if len(trimmed) < SLUG_MIN_LEN:
        raise SlugError(
            f"slugify({name!r}) produced {trimmed!r}, which is shorter than "
            f"the minimum slug length of {SLUG_MIN_LEN} characters"
        )
    if not SLUG_REGEX.match(trimmed):
        # Shouldn't happen given the transformations above, but guard anyway —
        # a bug here would produce an invalid directory name downstream.
        raise SlugError(
            f"slugify({name!r}) produced {trimmed!r}, which does not match "
            f"{SLUG_PATTERN}"
        )
    return trimmed


def resolve_unique_slug(base: str, skills_dir: Path) -> str:
    """Return ``base`` if unused, else ``base_2``, ``base_3``, ... until free.

    Collisions are detected purely by directory presence under ``skills_dir``
    (``skills_dir / candidate`` exists). The caller is responsible for ensuring
    ``skills_dir`` itself exists; if it does not, every candidate is considered
    free and ``base`` is returned unchanged.

    The numeric-suffix-only rule (no ``_v2`` / ``_copy`` / ``_new``) is
    deliberate — see the PRD S-009 notes.

    Raises :class:`SlugError` if ``base`` itself is not a valid slug, or if the
    numeric suffix would push the candidate over :data:`SLUG_MAX_LEN`
    characters (extremely unlikely in practice).
    """
    if not SLUG_REGEX.match(base):
        raise SlugError(f"resolve_unique_slug: base {base!r} is not a valid slug")

    if not _slug_exists(base, skills_dir):
        return base

    suffix = 2
    while True:
        candidate = f"{base}_{suffix}"
        if len(candidate) > SLUG_MAX_LEN:
            raise SlugError(
                f"resolve_unique_slug: cannot append _{suffix} to {base!r} "
                f"without exceeding max slug length {SLUG_MAX_LEN}"
            )
        if not _slug_exists(candidate, skills_dir):
            return candidate
        suffix += 1


def validate_user_slug(user_input: str, skills_dir: Path) -> tuple[bool, str]:
    """Validate a user-supplied slug candidate.

    Returns ``(True, slug)`` if the input is legal and no existing skill
    directory collides. Returns ``(False, reason)`` with a human-readable
    message otherwise — the message is suitable to surface verbatim to the
    user in the Q&A UI.
    """
    if not isinstance(user_input, str):
        return False, f"slug must be a string, got {type(user_input).__name__}"

    stripped = user_input.strip()
    if not stripped:
        return False, "slug cannot be empty"

    if len(stripped) < SLUG_MIN_LEN:
        return False, (
            f"slug is too short ({len(stripped)} chars); minimum is "
            f"{SLUG_MIN_LEN} characters"
        )
    if len(stripped) > SLUG_MAX_LEN:
        return False, (
            f"slug is too long ({len(stripped)} chars); maximum is "
            f"{SLUG_MAX_LEN} characters"
        )

    if stripped != stripped.lower():
        return False, "slug must be lowercase (no uppercase letters allowed)"

    first = stripped[0]
    if not ("a" <= first <= "z"):
        return False, (
            f"slug must start with a lowercase letter, got {first!r}"
        )

    if not SLUG_REGEX.match(stripped):
        # Give a specific reason when we can; fall back to pattern.
        bad_chars = sorted({c for c in stripped if not re.match(r"[a-z0-9_]", c)})
        if bad_chars:
            return False, (
                "slug can only contain lowercase letters, digits, and "
                f"underscores; disallowed characters: {''.join(bad_chars)!r}"
            )
        return False, f"slug does not match required pattern {SLUG_PATTERN}"

    if _slug_exists(stripped, skills_dir):
        return False, f"slug {stripped!r} already exists in the skills directory"

    return True, stripped


def _slug_exists(slug: str, skills_dir: Path) -> bool:
    """Return True if ``skills_dir / slug`` is an existing directory.

    A regular file sitting where a skill directory would live is ALSO considered
    a collision — overwriting random files is worse than choosing a different
    slug.
    """
    if not skills_dir.exists():
        return False
    candidate = skills_dir / slug
    return candidate.exists()
