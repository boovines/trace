"""Cross-consistency checks between SKILL.md and skill.meta.json.

The synthesizer writes two sibling files for every skill: ``SKILL.md`` (human-
readable, parsed by :mod:`synthesizer.skill_doc`) and ``skill.meta.json`` (the
machine-readable companion, schema at ``contracts/skill-meta.schema.json``).

JSON-schema validation on the meta file catches structural problems. This
module catches *semantic* drift between the two files — specifically, the
case where a step is marked ⚠️ in the markdown but missing from
``meta.destructive_steps`` (or vice versa). Either direction is a bug that
would let a destructive step slip past the runner's per-step destructive
gate.
"""

from __future__ import annotations

from typing import Any

from synthesizer.skill_doc import ParsedSkill


class SkillMetaMismatchError(ValueError):
    """Raised when skill.meta.json and SKILL.md disagree on destructive steps."""


def validate_meta_against_markdown(meta: dict[str, Any], parsed: ParsedSkill) -> None:
    """Cross-validate ``skill.meta.json`` against a parsed SKILL.md.

    Currently enforces that ``meta["destructive_steps"]`` (1-based indices)
    exactly matches the set of steps whose text was prefixed with ⚠️. Extra
    checks can be added here as the contract grows, but every check must be
    symmetric — any drift between the two files in either direction is an
    error, not a warning.

    Raises ``SkillMetaMismatchError`` on any disagreement.
    """
    declared_raw = meta.get("destructive_steps", [])
    if not isinstance(declared_raw, list):
        raise SkillMetaMismatchError(
            "skill.meta.json 'destructive_steps' must be a list of integers"
        )
    declared = {int(n) for n in declared_raw}
    observed = {step.number for step in parsed.steps if step.destructive}

    if declared != observed:
        missing_in_meta = sorted(observed - declared)
        missing_in_md = sorted(declared - observed)
        parts: list[str] = []
        if missing_in_meta:
            parts.append(
                "steps marked ⚠️ in SKILL.md but missing from meta.destructive_steps: "
                f"{missing_in_meta}"
            )
        if missing_in_md:
            parts.append(
                "steps in meta.destructive_steps but not marked ⚠️ in SKILL.md: "
                f"{missing_in_md}"
            )
        raise SkillMetaMismatchError("; ".join(parts))


__all__ = ["SkillMetaMismatchError", "validate_meta_against_markdown"]
