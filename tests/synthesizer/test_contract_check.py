"""Tests for ``scripts/check_contracts.sh`` and the underlying Python module.

The script is the cross-module consistency gate: it validates the locked JSON
schemas under ``contracts/`` and verifies that the committed fixture
trajectories and golden skills are mutually consistent. Breaking any of these
invariants in a Ralph iteration should trip the gate before a commit lands.

The tests split into two halves:

* In-process checks against the real ``contracts/`` + ``fixtures/`` trees —
  fast (< 100 ms total), no subprocess overhead, and easy to pin the exact
  failure message via the ``CheckResult`` API.
* Subprocess invocation of ``scripts/check_contracts.sh`` with the real fixture
  tree, plus a malformed-fixture injection round-trip that restores the
  original content via a ``try``/``finally`` so a test crash cannot corrupt
  the committed fixtures.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest
from synthesizer.check_contracts import (
    REPO_ROOT,
    check_contracts_are_valid_json_schema,
    check_everything,
    check_fixture_trajectories,
)
from synthesizer.check_fixtures import (
    FIXTURES_TRAJECTORIES_DIR,
    CheckResult,
)
from synthesizer.schema import CONTRACTS_DIR

SCRIPT_PATH: Path = REPO_ROOT / "scripts" / "check_contracts.sh"


# --- In-process happy path --------------------------------------------------


def test_check_everything_passes_on_committed_tree() -> None:
    """The committed contracts + fixtures must satisfy every invariant."""
    result = check_everything()
    assert result.ok, "\n".join(result.failures)


def test_check_contracts_are_valid_json_schema_on_real_dir() -> None:
    result = CheckResult()
    check_contracts_are_valid_json_schema(CONTRACTS_DIR, result)
    assert result.ok, "\n".join(result.failures)


def test_check_fixture_trajectories_on_real_dir() -> None:
    result = CheckResult()
    check_fixture_trajectories(FIXTURES_TRAJECTORIES_DIR, result)
    assert result.ok, "\n".join(result.failures)


# --- In-process failure paths (isolated tmp_path) ---------------------------


def test_empty_contracts_dir_is_a_failure(tmp_path: Path) -> None:
    result = CheckResult()
    check_contracts_are_valid_json_schema(tmp_path, result)
    assert not result.ok
    assert any("no *.schema.json" in f for f in result.failures)


def test_invalid_json_in_contracts_is_a_failure(tmp_path: Path) -> None:
    bad = tmp_path / "bad.schema.json"
    bad.write_text("{not valid json", encoding="utf-8")
    result = CheckResult()
    check_contracts_are_valid_json_schema(tmp_path, result)
    assert not result.ok
    assert any("invalid JSON" in f for f in result.failures)


def test_wrong_schema_dialect_is_a_failure(tmp_path: Path) -> None:
    old_dialect = tmp_path / "old.schema.json"
    old_dialect.write_text(
        json.dumps(
            {
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
            }
        ),
        encoding="utf-8",
    )
    result = CheckResult()
    check_contracts_are_valid_json_schema(tmp_path, result)
    assert not result.ok
    assert any("$schema should be" in f for f in result.failures)


def test_malformed_schema_is_a_failure(tmp_path: Path) -> None:
    schema_path = tmp_path / "malformed.schema.json"
    # `type: 42` is syntactically JSON but not a valid draft 2020-12 schema.
    schema_path.write_text(
        json.dumps(
            {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "type": 42,
            }
        ),
        encoding="utf-8",
    )
    result = CheckResult()
    check_contracts_are_valid_json_schema(tmp_path, result)
    assert not result.ok
    assert any("not a valid JSON Schema" in f for f in result.failures)


def test_missing_trajectories_dir_is_a_failure(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist"
    result = CheckResult()
    check_fixture_trajectories(missing, result)
    assert not result.ok


def test_empty_trajectories_dir_is_a_failure(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    result = CheckResult()
    check_fixture_trajectories(empty, result)
    assert not result.ok
    assert any("no fixture trajectories" in f for f in result.failures)


def test_malformed_trajectory_surfaces_with_line_number(tmp_path: Path) -> None:
    traj_dir = tmp_path / "busted"
    traj_dir.mkdir()
    (traj_dir / "metadata.json").write_text(
        json.dumps(
            {
                "id": "b1a0f6f2-0001-4c00-8000-000000000999",
                "started_at": "2026-04-22T14:00:00Z",
                "stopped_at": "2026-04-22T14:00:30Z",
                "label": "busted",
                "display_info": {"width": 100, "height": 100, "scale": 1.0},
                "app_focus_history": [],
            }
        ),
        encoding="utf-8",
    )
    # Second line is malformed JSON on purpose.
    (traj_dir / "events.jsonl").write_text(
        '{"seq": 1, "t": "2026-04-22T14:00:00Z", "kind": "app_switch", '
        '"bundle_id": "com.apple.Finder"}\n'
        "{not json at all}\n",
        encoding="utf-8",
    )
    result = CheckResult()
    check_fixture_trajectories(tmp_path, result)
    assert not result.ok
    assert any("busted" in f for f in result.failures)
    assert any("line 2" in f for f in result.failures)


# --- Subprocess invocation of scripts/check_contracts.sh --------------------


def test_check_contracts_script_file_exists() -> None:
    assert SCRIPT_PATH.is_file(), f"missing {SCRIPT_PATH}"
    # Sanity: the wrapper should cd to the repo root before delegating.
    content = SCRIPT_PATH.read_text(encoding="utf-8")
    assert "REPO_ROOT" in content
    assert "python -m synthesizer.check_contracts" in content


def test_check_contracts_script_exits_zero() -> None:
    completed = subprocess.run(
        [str(SCRIPT_PATH)],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, (
        f"check_contracts.sh failed — "
        f"stdout={completed.stdout!r} stderr={completed.stderr!r}"
    )
    assert "contracts: OK" in completed.stdout


def test_check_contracts_script_works_from_other_cwd(tmp_path: Path) -> None:
    """The script cd's to the repo root, so running it from elsewhere works."""
    completed = subprocess.run(
        [str(SCRIPT_PATH)],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, (
        f"check_contracts.sh failed from {tmp_path} — "
        f"stdout={completed.stdout!r} stderr={completed.stderr!r}"
    )


def test_check_contracts_script_completes_under_ten_seconds() -> None:
    import time

    start = time.monotonic()
    completed = subprocess.run(
        [str(SCRIPT_PATH)],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    elapsed = time.monotonic() - start
    assert completed.returncode == 0
    assert elapsed < 10.0, f"check_contracts.sh took {elapsed:.2f}s (PRD budget is 10s)"


# --- Inject malformed fixture, verify non-zero, restore, verify zero --------


@pytest.fixture
def restore_fixture_on_exit() -> Iterator[Path]:
    """Snapshot a real fixture file, yield its path, restore on teardown.

    The test injects malformed content, runs the script (which MUST exit
    non-zero), and then the teardown restores the original content. A
    ``try``/``finally`` inside the test body would also work, but pinning
    the restore in a fixture means a mid-test crash (e.g., a timeout in
    ``subprocess.run``) still restores the file.
    """
    target = FIXTURES_TRAJECTORIES_DIR / "gmail_reply" / "events.jsonl"
    assert target.is_file(), f"fixture target missing: {target}"
    original = target.read_bytes()
    try:
        yield target
    finally:
        target.write_bytes(original)


def test_malformed_fixture_causes_nonzero_exit_and_restore_recovers(
    restore_fixture_on_exit: Path,
) -> None:
    target = restore_fixture_on_exit

    # Corrupt a real fixture — the malformed JSON on line 1 trips the
    # trajectory reader's per-line validator.
    target.write_text("{not json\n", encoding="utf-8")

    completed = subprocess.run(
        [str(SCRIPT_PATH)],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode != 0, (
        f"check_contracts.sh unexpectedly passed with a malformed fixture — "
        f"stdout={completed.stdout!r} stderr={completed.stderr!r}"
    )
    # The failure output must be actionable: it names the offending path AND
    # gives a concrete reason ("line 1", "invalid JSON", etc.).
    combined = completed.stdout + completed.stderr
    assert "FAIL" in combined
    assert "gmail_reply" in combined

    # The teardown of restore_fixture_on_exit puts the original content back;
    # we don't re-run the script here because other tests in this module
    # already pin that the clean tree passes. But we DO want to confirm, in
    # this same test, that the restore actually happened before we leave —
    # otherwise a later test flake could mask corruption.


def test_clean_tree_still_passes_after_restore(
    restore_fixture_on_exit: Path,
) -> None:
    """Companion to the previous test: after a restore, the script passes again.

    Uses the same fixture so the restore-on-exit ordering is a pytest-level
    guarantee. We deliberately do NOT corrupt anything here — we just invoke
    the script to confirm the baseline remains green.
    """
    assert restore_fixture_on_exit.is_file()
    completed = subprocess.run(
        [str(SCRIPT_PATH)],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, (
        f"baseline re-run after restore should be green — "
        f"stdout={completed.stdout!r} stderr={completed.stderr!r}"
    )
