"""Validate every hand-crafted golden skill fixture under ``fixtures/skills/``.

Invoked by :file:`scripts/check_fixtures.sh`. Runs in-process (no subprocess
per fixture) so the full pass over the five reference workflows completes in
well under five seconds.

Validation per fixture directory:

1. ``skill.meta.json`` conforms to ``contracts/skill-meta.schema.json``.
2. ``SKILL.md`` round-trips through
   :func:`synthesizer.skill_doc.parse_skill_md` /
   :func:`synthesizer.skill_doc.render_skill_md`.
3. :func:`synthesizer.schema.validate_meta_against_markdown` agrees on step
   count, destructive flags, and parameter references.
4. ``meta.trajectory_id`` resolves to an existing
   ``fixtures/trajectories/<dir>/metadata.json`` with a matching ``id`` field.

Exits ``0`` when every fixture passes; on failure, prints the offending file
path and a concrete reason and exits non-zero.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from synthesizer.schema import (
    CONTRACTS_DIR,
    ValidationError,
    validate_meta,
    validate_meta_against_markdown,
)
from synthesizer.skill_doc import (
    SkillParseError,
    parse_skill_md,
    render_skill_md,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

__all__ = ["CheckResult", "check_all_fixtures", "main"]


REPO_ROOT: Path = CONTRACTS_DIR.parent
FIXTURES_SKILLS_DIR: Path = REPO_ROOT / "fixtures" / "skills"
FIXTURES_TRAJECTORIES_DIR: Path = REPO_ROOT / "fixtures" / "trajectories"


class CheckResult:
    """Collected failures for a ``check_all_fixtures`` run."""

    def __init__(self) -> None:
        self.failures: list[str] = []

    def fail(self, path: Path, reason: str) -> None:
        self.failures.append(f"{path}: {reason}")

    @property
    def ok(self) -> bool:
        return not self.failures


def _iter_fixture_dirs(skills_dir: Path) -> Iterable[Path]:
    if not skills_dir.is_dir():
        return ()
    return sorted(p for p in skills_dir.iterdir() if p.is_dir())


def _check_one(fixture_dir: Path, result: CheckResult) -> None:
    skill_md = fixture_dir / "SKILL.md"
    meta_json = fixture_dir / "skill.meta.json"

    if not skill_md.is_file():
        result.fail(fixture_dir, "missing SKILL.md")
        return
    if not meta_json.is_file():
        result.fail(fixture_dir, "missing skill.meta.json")
        return

    try:
        meta = json.loads(meta_json.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        result.fail(meta_json, f"invalid JSON: {e}")
        return

    try:
        validate_meta(meta)
    except ValidationError as e:
        result.fail(meta_json, f"schema validation failed: {e.message}")
        return

    markdown = skill_md.read_text(encoding="utf-8")

    try:
        parsed = parse_skill_md(markdown)
    except SkillParseError as e:
        result.fail(skill_md, f"parse failed: {e}")
        return

    try:
        re_parsed = parse_skill_md(render_skill_md(parsed))
    except SkillParseError as e:
        result.fail(skill_md, f"round-trip re-parse failed: {e}")
        return

    if re_parsed != parsed:
        result.fail(skill_md, "round-trip inequality: parse(render(parse(f))) != parse(f)")
        return

    try:
        validate_meta_against_markdown(meta, markdown)
    except ValidationError as e:
        result.fail(fixture_dir, f"markdown/meta cross-check failed: {e.message}")
        return

    expected_slug = fixture_dir.name
    if meta["slug"] != expected_slug:
        result.fail(
            meta_json,
            f"meta.slug '{meta['slug']}' does not match directory name '{expected_slug}'",
        )
        return

    trajectory_id = meta["trajectory_id"]
    _check_trajectory_link(fixture_dir, trajectory_id, result)


def _check_trajectory_link(fixture_dir: Path, trajectory_id: str, result: CheckResult) -> None:
    """Ensure ``meta.trajectory_id`` points at a real fixture trajectory.

    The contract is looser than "directory name equals trajectory id": the
    fixture folders under ``fixtures/trajectories/`` are named by workflow
    slug (e.g., ``gmail_reply/``), and the trajectory's id lives inside its
    ``metadata.json``. We accept any trajectory directory whose ``metadata.json``
    carries a matching ``id`` — this way a rename of the slug does not break
    the link.
    """
    if not FIXTURES_TRAJECTORIES_DIR.is_dir():
        result.fail(
            fixture_dir,
            "fixtures/trajectories/ directory is missing; "
            f"cannot resolve trajectory_id {trajectory_id!r}",
        )
        return

    for traj_dir in FIXTURES_TRAJECTORIES_DIR.iterdir():
        if not traj_dir.is_dir():
            continue
        metadata_path = traj_dir / "metadata.json"
        if not metadata_path.is_file():
            continue
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if metadata.get("id") == trajectory_id:
            return

    result.fail(
        fixture_dir,
        f"meta.trajectory_id {trajectory_id!r} does not match any "
        "fixtures/trajectories/<dir>/metadata.json id",
    )


def check_all_fixtures(skills_dir: Path = FIXTURES_SKILLS_DIR) -> CheckResult:
    """Validate every golden skill fixture under ``skills_dir``.

    Returns a :class:`CheckResult`; inspect ``.ok`` and ``.failures`` on the
    returned object.
    """
    result = CheckResult()
    fixtures = list(_iter_fixture_dirs(skills_dir))
    if not fixtures:
        result.fail(skills_dir, "no golden skill fixtures found")
        return result
    for fixture_dir in fixtures:
        _check_one(fixture_dir, result)
    return result


def main(argv: list[str] | None = None) -> int:
    del argv  # reserved for future --fixtures-dir override
    result = check_all_fixtures()
    if result.ok:
        sys.stdout.write("fixtures: OK\n")
        return 0
    for failure in result.failures:
        sys.stderr.write(f"FAIL: {failure}\n")
    sys.stderr.write(f"fixtures: {len(result.failures)} failure(s)\n")
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
