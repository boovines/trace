"""Loading and validating ``skill.meta.json`` against the locked contract.

The schema itself lives at ``contracts/skill-meta.schema.json`` at the repo root
(a single source of truth shared across modules). This module exposes:

* :func:`load_meta_schema` — reads and caches the schema.
* :func:`validate_meta` — schema-validates a meta dict.
* :func:`validate_meta_against_markdown` — cross-checks that a meta dict and its
  sibling SKILL.md agree on parameters, destructive-step flags, and step count.

The cross-check is load-bearing: the LLM will occasionally drift between the
markdown it writes and the meta JSON it writes, and we'd rather catch the drift
structurally than hope for prompt obedience.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from jsonschema import (
    Draft202012Validator,
    FormatChecker,
    ValidationError,
)

__all__ = [
    "CONTRACTS_DIR",
    "SCHEMA_PATH",
    "ValidationError",
    "load_meta_schema",
    "validate_meta",
    "validate_meta_against_markdown",
]


def _find_repo_root(start: Path) -> Path:
    """Walk up from ``start`` to the first directory containing ``contracts/``.

    The synthesizer package can be imported from the editable install under the
    workspace or from a published wheel — we don't assume a fixed relative
    layout, we just walk up looking for the sibling ``contracts`` dir.
    """
    for candidate in (start, *start.parents):
        if (candidate / "contracts" / "skill-meta.schema.json").is_file():
            return candidate
    raise FileNotFoundError(
        "Could not locate contracts/skill-meta.schema.json by walking up from "
        f"{start}. Is the schema file present in the repository?"
    )


CONTRACTS_DIR: Path = _find_repo_root(Path(__file__).resolve()) / "contracts"
SCHEMA_PATH: Path = CONTRACTS_DIR / "skill-meta.schema.json"


@lru_cache(maxsize=1)
def load_meta_schema() -> dict[str, Any]:
    """Return the parsed ``skill-meta.schema.json`` as a dict.

    Cached across calls — the schema is immutable at runtime.
    """
    with SCHEMA_PATH.open("r", encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)
    return data


@lru_cache(maxsize=1)
def _validator() -> Draft202012Validator:
    return Draft202012Validator(load_meta_schema(), format_checker=FormatChecker())


def validate_meta(meta: dict[str, Any]) -> None:
    """Validate ``meta`` against the JSON schema.

    Raises :class:`jsonschema.ValidationError` on failure. The message includes
    the JSON pointer path to the offending field, so callers can surface it
    directly to users.
    """
    _validator().validate(meta)


# --- Markdown cross-check ---------------------------------------------------

# Matches a {param_name} reference where param_name is our identifier pattern.
# We deliberately don't match on escaped \{ or inside fenced code blocks; those
# cases are handled in S-003's proper parser. For the cross-check at this layer
# we pre-strip fenced code blocks to avoid false positives.
_PARAM_RE = re.compile(r"\{([a-z][a-z0-9_]{0,29})\}")

# A numbered-step line under ## Steps. We tolerate any amount of leading
# whitespace after the period, and capture the rest of the line as step text.
_STEP_LINE_RE = re.compile(r"^(\d+)\.\s+(.*)$")

# The destructive sentinel the synthesizer's draft prompt is instructed to
# emit. Must be an exact string — the matcher's job is to flag drift if the LLM
# produces a similar-but-different prefix.
_DESTRUCTIVE_MARKER = "⚠️ [DESTRUCTIVE]"

_FENCED_CODE_RE = re.compile(r"```.*?```", re.DOTALL)


def _strip_fenced_code(markdown: str) -> str:
    """Remove triple-backtick fenced code blocks from ``markdown``.

    Parameter-like tokens inside code examples should not count as parameter
    references. Inline backticks are left intact — they occur in normal prose
    and the S-003 parser handles them properly.
    """
    return _FENCED_CODE_RE.sub("", markdown)


def _extract_param_refs(markdown: str) -> set[str]:
    return set(_PARAM_RE.findall(_strip_fenced_code(markdown)))


def _extract_steps_section(markdown: str) -> list[str]:
    """Return the lines between ``## Steps`` and the next H2 (or EOF)."""
    lines = markdown.splitlines()
    out: list[str] = []
    in_steps = False
    for line in lines:
        stripped = line.strip()
        if stripped.lower() == "## steps":
            in_steps = True
            continue
        if in_steps and stripped.startswith("## "):
            break
        if in_steps:
            out.append(line)
    return out


def _iter_numbered_steps(markdown: str) -> list[tuple[int, str]]:
    """Return a list of ``(step_number, step_text)`` from the ## Steps section.

    A step is any line that starts with ``<digits>. `` inside the Steps section.
    Continuation lines (indented under a step) are NOT treated as new steps.
    """
    steps: list[tuple[int, str]] = []
    for line in _extract_steps_section(markdown):
        match = _STEP_LINE_RE.match(line)
        if match:
            steps.append((int(match.group(1)), match.group(2)))
    return steps


def _raise_cross_check(message: str, path: str) -> None:
    """Raise a :class:`ValidationError` shaped like a schema failure.

    Using the same exception type as :func:`validate_meta` lets callers have a
    single ``except jsonschema.ValidationError`` and get useful messages from
    either source.
    """
    err = ValidationError(message)
    # ``absolute_path`` is a ``deque[str | int]`` at runtime, but the
    # ``types-jsonschema`` stubs widen it to ``Sequence[str | int]`` which
    # doesn't expose ``.append``. Suppress the false positive.
    err.absolute_path.append(path)  # type: ignore[attr-defined]
    raise err


def validate_meta_against_markdown(meta: dict[str, Any], markdown: str) -> None:
    """Cross-check that ``meta`` and ``markdown`` agree on invariants.

    Three checks:

    1. ``step_count`` matches the number of numbered steps in ``## Steps``.
    2. ``destructive_steps`` matches the step numbers whose text contains the
       ``⚠️ [DESTRUCTIVE]`` marker.
    3. Every ``{name}`` reference in the markdown has a matching entry in
       ``meta['parameters']``, and vice versa.

    Raises :class:`jsonschema.ValidationError` on any mismatch; the exception's
    message names the specific drift so the caller can show it to the user (or
    feed it back to the LLM for a retry).

    This function assumes ``meta`` has already passed :func:`validate_meta`.
    """
    numbered = _iter_numbered_steps(markdown)
    step_numbers = [n for n, _ in numbered]

    expected_count = meta.get("step_count")
    if expected_count != len(numbered):
        _raise_cross_check(
            f"step_count mismatch: meta says {expected_count}, markdown has "
            f"{len(numbered)} numbered steps under ## Steps",
            "step_count",
        )

    # Verify numbering is 1..N sequential (parser-lite; full check is in S-003).
    if step_numbers != list(range(1, len(numbered) + 1)):
        _raise_cross_check(
            "step numbering must be 1-indexed and sequential; "
            f"found {step_numbers}",
            "step_count",
        )

    md_destructive: set[int] = {n for n, text in numbered if _DESTRUCTIVE_MARKER in text}
    meta_destructive: set[int] = set(meta.get("destructive_steps", []))
    if md_destructive != meta_destructive:
        only_md_steps = sorted(md_destructive - meta_destructive)
        only_meta_steps = sorted(meta_destructive - md_destructive)
        _raise_cross_check(
            "destructive_steps disagreement — "
            f"only in markdown: {only_md_steps}; only in meta: {only_meta_steps}",
            "destructive_steps",
        )

    md_params: set[str] = _extract_param_refs(markdown)
    meta_params: set[str] = {p["name"] for p in meta.get("parameters", [])}
    if md_params != meta_params:
        only_md_params = sorted(md_params - meta_params)
        only_meta_params = sorted(meta_params - md_params)
        _raise_cross_check(
            "parameter references disagreement — "
            f"in markdown but not meta: {only_md_params}; "
            f"in meta but not markdown: {only_meta_params}",
            "parameters",
        )

    # ---- steps[].number bounds + hint shape ---------------------------
    # ``steps`` is optional; absence means the runner falls through to
    # computer-use for every step. When present, every entry must point
    # at a real step number (1..len(numbered)) and any tier=mcp hints
    # must reference a server/function declared in the MCP catalog.
    meta_steps_raw = meta.get("steps") or []
    seen_numbers: set[int] = set()
    for i, entry in enumerate(meta_steps_raw):
        if not isinstance(entry, dict):
            _raise_cross_check(
                f"steps[{i}] must be a JSON object, got {type(entry).__name__}",
                f"steps/{i}",
            )
        num = entry.get("number")
        if not isinstance(num, int) or num < 1 or num > len(numbered):
            _raise_cross_check(
                f"steps[{i}].number={num!r} is outside the valid range "
                f"1..{len(numbered)} (markdown has {len(numbered)} steps)",
                f"steps/{i}/number",
            )
        if num in seen_numbers:
            _raise_cross_check(
                f"duplicate steps[].number={num} — each step number may "
                "appear at most once in meta.steps",
                f"steps/{i}/number",
            )
        seen_numbers.add(num)
        for j, hint in enumerate(entry.get("execution_hints") or []):
            err = _validate_execution_hint(hint)
            if err is not None:
                _raise_cross_check(
                    f"steps[{i}].execution_hints[{j}]: {err}",
                    f"steps/{i}/execution_hints/{j}",
                )


def _validate_execution_hint(hint: Any) -> str | None:
    """Return an error message if ``hint`` is malformed, else ``None``.

    Tier-specific shape requirements (mcp needs server+function+arguments;
    browser_dom needs selector+action; computer_use needs summary) are
    enforced here because the JSON schema can only express it via a
    fragile ``oneOf`` block. tier=mcp hints are also looked up against
    :mod:`synthesizer.mcp_catalog` so the LLM can't hallucinate function
    names that aren't on the wire.
    """
    from synthesizer.mcp_catalog import validate_hint as _validate_mcp_hint

    if not isinstance(hint, dict):
        return f"hint must be a JSON object, got {type(hint).__name__}"
    tier = hint.get("tier")
    if tier == "mcp":
        for required in ("mcp_server", "function", "arguments"):
            if required not in hint:
                return f"tier=mcp hint missing {required!r}"
        if not isinstance(hint["arguments"], dict):
            return "tier=mcp hint.arguments must be an object"
        catalog_err = _validate_mcp_hint(
            server=hint["mcp_server"],
            function=hint["function"],
            arguments=hint["arguments"],
        )
        if catalog_err is not None:
            return catalog_err
        return None
    if tier == "browser_dom":
        for required in ("selector", "action"):
            if required not in hint:
                return f"tier=browser_dom hint missing {required!r}"
        if hint["action"] in ("type", "navigate") and "value" not in hint:
            return f"tier=browser_dom action={hint['action']!r} requires a 'value' field"
        return None
    if tier == "computer_use":
        if "summary" not in hint:
            return "tier=computer_use hint missing 'summary'"
        return None
    return f"unknown tier {tier!r}; expected mcp, browser_dom, or computer_use"
