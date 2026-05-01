"""Secondary destructive-step matcher (belt-and-suspenders).

The draft prompt instructs Claude to flag destructive steps with
``⚠️ [DESTRUCTIVE]`` in the markdown and to list them in
``meta.destructive_steps``. That guidance is not enough by itself — models
occasionally miss a flag, so this module runs an independent keyword scan
over each step's source click ``target.label`` from the raw trajectory and
flags any that matches a known-destructive verb. The intent is to make the
skill file on disk correct even when the LLM slips.

Matching rules (v1 — English-only):

* Case-insensitive, word-boundary (``re.IGNORECASE`` + ``\\b``) matches
  against :data:`DESTRUCTIVE_KEYWORDS` in the click's ``target.label``.
* Step↔click binding walks clicks in ``seq`` order; for each click whose
  ``target.label`` contains a destructive keyword, the matcher binds it to
  the earliest un-bound step whose text case-insensitively contains that
  same label substring. Steps not bound to any destructive click — non-click
  steps, clicks with harmless labels, or clicks whose label does not appear
  in any step text — are never flagged.
* **Additive only.** A step already flagged by the LLM is left alone; the
  matcher can add flags but never removes them. Disagreement (matcher +
  LLM) resolves to flagged.
* The matcher looks at the click ``target.label`` (the physical UI label).
  The label-in-step-text binding is a safety rail against flagging a step
  that genuinely describes a different action: "Submit a query" against a
  click on "Search" has no label overlap and is left alone.

The Runner has its own live-execution keyword matcher as a third line of
defense; writing the flags into the skill file at synthesis time means the
disk artifact is already correct regardless of downstream reprocessing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from synthesizer.skill_doc import ParsedSkill, Step

if TYPE_CHECKING:
    from synthesizer.trajectory_reader import TrajectoryReader

__all__ = [
    "DESTRUCTIVE_KEYWORDS",
    "MatcherReport",
    "MatcherResult",
    "apply_destructive_matcher",
    "label_has_destructive_keyword",
]


DESTRUCTIVE_KEYWORDS: tuple[str, ...] = (
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
)
"""English verbs that indicate a destructive UI action. Sourced from
``CLAUDE.md`` — kept in sync with the Runner's own matcher.
"""


_KEYWORD_RE: re.Pattern[str] = re.compile(
    r"\b(?:" + "|".join(DESTRUCTIVE_KEYWORDS) + r")\b",
    re.IGNORECASE,
)


def label_has_destructive_keyword(label: str | None) -> bool:
    """Return ``True`` when ``label`` contains a word-boundary destructive keyword.

    Empty / ``None`` labels return ``False``. The check is case-insensitive
    (``Send``, ``SEND``, ``send`` all match) and uses ASCII word boundaries
    so ``sender`` does NOT match ``send``.
    """
    if not label:
        return False
    return _KEYWORD_RE.search(label) is not None


@dataclass(frozen=True)
class MatcherReport:
    """Telemetry from :func:`apply_destructive_matcher`.

    * ``added_flags`` — step numbers the matcher newly flagged destructive.
    * ``unchanged`` — step numbers the LLM had already flagged and the matcher
      left in place.
    * ``llm_flags`` — all step numbers the LLM flagged in the input
      (superset-ish view into what the model produced before the matcher ran).
    """

    added_flags: list[int] = field(default_factory=list)
    unchanged: list[int] = field(default_factory=list)
    llm_flags: list[int] = field(default_factory=list)


@dataclass(frozen=True)
class MatcherResult:
    """Return of :func:`apply_destructive_matcher`: updated skill + telemetry."""

    parsed: ParsedSkill
    report: MatcherReport


def apply_destructive_matcher(
    parsed: ParsedSkill, reader: TrajectoryReader
) -> MatcherResult:
    """Apply the destructive-keyword secondary matcher to a :class:`ParsedSkill`.

    Returns a :class:`MatcherResult` whose ``parsed`` field is a new
    :class:`ParsedSkill` with the combined LLM + matcher flags. The original
    skill is not mutated (``ParsedSkill`` is frozen).

    Binding: for each click event whose ``target.label`` contains a
    destructive keyword (walked in ``seq`` order), find the earliest
    un-bound step whose text case-insensitively contains the label
    substring. Non-click steps, clicks with harmless labels, and clicks
    whose label does not appear in any step text are never flagged.
    """
    llm_flags: list[int] = [s.number for s in parsed.steps if s.destructive]
    added_flags: list[int] = []

    destructive_clicks: list[str] = []
    for click in reader.iter_events_by_type("click"):
        if click.target is None:
            continue
        raw = click.target.get("label")
        label = raw if isinstance(raw, str) else None
        if label and label_has_destructive_keyword(label):
            destructive_clicks.append(label)

    new_steps: list[Step] = list(parsed.steps)
    bound: set[int] = set()

    for label in destructive_clicks:
        lowered = label.lower()
        for i, step in enumerate(new_steps):
            if step.number in bound:
                continue
            if lowered in step.text.lower():
                bound.add(step.number)
                if not step.destructive:
                    added_flags.append(step.number)
                    new_steps[i] = step.model_copy(update={"destructive": True})
                break

    updated = parsed.model_copy(update={"steps": new_steps})
    report = MatcherReport(
        added_flags=sorted(added_flags),
        unchanged=sorted(llm_flags),
        llm_flags=sorted(llm_flags),
    )
    return MatcherResult(parsed=updated, report=report)
