"""Cross-module contract and fixture consistency check.

Invoked by :file:`scripts/check_contracts.sh`. The script is the final safety
net that catches breaking changes to the locked JSON schemas under
``contracts/`` and misalignments between the fixture trajectories (Recorder
ground truth) and the golden skill fixtures (Synthesizer ground truth).

Checks performed, in order, on the current working copy:

1. Every ``*.schema.json`` under ``contracts/`` is syntactically valid JSON
   and a valid JSON Schema draft 2020-12 (``Draft202012Validator.check_schema``
   does not raise).
2. Every trajectory under ``fixtures/trajectories/<dir>/`` passes
   :class:`synthesizer.trajectory_reader.TrajectoryReader` (which validates
   ``metadata.json`` and every line of ``events.jsonl`` against the trajectory
   schema).
3. Every golden skill under ``fixtures/skills/<dir>/`` passes schema +
   round-trip + markdown/meta cross-check via
   :func:`synthesizer.check_fixtures.check_all_fixtures`.
4. Every golden skill's ``meta.trajectory_id`` resolves to a trajectory
   directory whose ``metadata.json.id`` matches (delegated to the same helper
   in :mod:`synthesizer.check_fixtures`).

Exits ``0`` when every check passes; on failure prints concrete file paths
and reasons to stderr and exits ``1``. Designed to complete in well under ten
seconds on the full fixture set so it can sit in the Ralph loop's quality
gates next to ``ruff`` / ``mypy`` / ``pytest``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from jsonschema import (
    Draft202012Validator,
    SchemaError,
)

from synthesizer.check_fixtures import (
    FIXTURES_SKILLS_DIR,
    FIXTURES_TRAJECTORIES_DIR,
    CheckResult,
    check_all_fixtures,
)
from synthesizer.schema import CONTRACTS_DIR
from synthesizer.trajectory_reader import TrajectoryReader, TrajectoryReadError

if TYPE_CHECKING:
    from collections.abc import Iterable

__all__ = [
    "check_contracts_are_valid_json_schema",
    "check_everything",
    "check_fixture_trajectories",
    "main",
]


REPO_ROOT: Path = CONTRACTS_DIR.parent


# --- individual checks ------------------------------------------------------


def _iter_schema_files(contracts_dir: Path) -> Iterable[Path]:
    if not contracts_dir.is_dir():
        return ()
    return sorted(contracts_dir.glob("*.schema.json"))


def check_contracts_are_valid_json_schema(
    contracts_dir: Path, result: CheckResult
) -> None:
    """Ensure every ``*.schema.json`` under ``contracts/`` is a valid schema."""
    schema_files = list(_iter_schema_files(contracts_dir))
    if not schema_files:
        result.fail(contracts_dir, "no *.schema.json files found under contracts/")
        return

    for schema_path in schema_files:
        try:
            schema = json.loads(schema_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            result.fail(schema_path, f"invalid JSON: {e}")
            continue

        declared = schema.get("$schema")
        if declared != "https://json-schema.org/draft/2020-12/schema":
            result.fail(
                schema_path,
                f"$schema should be 'https://json-schema.org/draft/2020-12/schema'; "
                f"found {declared!r}",
            )
            continue

        try:
            Draft202012Validator.check_schema(schema)
        except SchemaError as e:
            pointer = "/".join(str(p) for p in e.absolute_path) or "<root>"
            result.fail(
                schema_path,
                f"not a valid JSON Schema draft 2020-12 at {pointer}: {e.message}",
            )


def _iter_trajectory_dirs(trajectories_dir: Path) -> Iterable[Path]:
    if not trajectories_dir.is_dir():
        return ()
    return sorted(p for p in trajectories_dir.iterdir() if p.is_dir())


def check_fixture_trajectories(
    trajectories_dir: Path, result: CheckResult
) -> None:
    """Validate every trajectory fixture via :class:`TrajectoryReader`."""
    trajectory_dirs = list(_iter_trajectory_dirs(trajectories_dir))
    if not trajectory_dirs:
        result.fail(trajectories_dir, "no fixture trajectories found")
        return

    for traj_dir in trajectory_dirs:
        try:
            TrajectoryReader(traj_dir)
        except (TrajectoryReadError, FileNotFoundError) as e:
            result.fail(traj_dir, f"trajectory validation failed: {e}")
        except Exception as e:  # pragma: no cover — defensive
            result.fail(traj_dir, f"unexpected error loading trajectory: {e}")


# --- orchestrator -----------------------------------------------------------


def check_everything(
    contracts_dir: Path = CONTRACTS_DIR,
    trajectories_dir: Path = FIXTURES_TRAJECTORIES_DIR,
    skills_dir: Path = FIXTURES_SKILLS_DIR,
) -> CheckResult:
    """Run all four cross-module checks and return a combined result."""
    result = CheckResult()
    check_contracts_are_valid_json_schema(contracts_dir, result)
    check_fixture_trajectories(trajectories_dir, result)

    # Golden skill fixtures (schema + round-trip + cross-check +
    # trajectory_id linkage) are checked by the same helper S-016 wired up;
    # merge its failures into our combined result.
    skill_result = check_all_fixtures(skills_dir)
    result.failures.extend(skill_result.failures)

    return result


def main(argv: list[str] | None = None) -> int:
    del argv  # reserved for future --contracts-dir / --fixtures-dir overrides
    result = check_everything()
    if result.ok:
        sys.stdout.write("contracts: OK\n")
        return 0
    for failure in result.failures:
        sys.stderr.write(f"FAIL: {failure}\n")
    sys.stderr.write(f"contracts: {len(result.failures)} failure(s)\n")
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
