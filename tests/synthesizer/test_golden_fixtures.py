"""Validate the five hand-crafted golden skill fixtures under ``fixtures/skills/``.

The golden skills are the ground truth for the snapshot similarity test
(S-014) and the real-mode smoke test (S-017). They are hand-crafted and MUST
NOT be regenerated automatically — see ``fixtures/skills/README.md``.

This test module pins the invariants every fixture is expected to satisfy:

* ``skill.meta.json`` passes the locked JSON Schema
  (``contracts/skill-meta.schema.json``).
* ``SKILL.md`` round-trips cleanly through
  :func:`synthesizer.skill_doc.parse_skill_md` /
  :func:`synthesizer.skill_doc.render_skill_md`.
* :func:`synthesizer.schema.validate_meta_against_markdown` agrees on step
  count, destructive flags, and parameter references.
* Each golden skill has at least four steps and at least one parameter.
* ``gmail_reply``, ``finder_organize``, and ``slack_status`` each declare at
  least one destructive step; ``calendar_block`` and ``notes_daily`` may
  declare zero (per the PRD — reversible workflows).
* ``meta.trajectory_id`` resolves to an existing
  ``fixtures/trajectories/<dir>/metadata.json`` with a matching ``id``.
* Total size of ``fixtures/skills/`` is under 1 MB.
* ``scripts/check_fixtures.sh`` exits zero on the committed fixtures.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from synthesizer.check_fixtures import (
    FIXTURES_SKILLS_DIR,
    FIXTURES_TRAJECTORIES_DIR,
    REPO_ROOT,
    check_all_fixtures,
)
from synthesizer.schema import validate_meta, validate_meta_against_markdown
from synthesizer.skill_doc import parse_skill_md, render_skill_md

GOLDEN_SLUGS: tuple[str, ...] = (
    "gmail_reply",
    "calendar_block",
    "finder_organize",
    "slack_status",
    "notes_daily",
)

# Per the PRD, these workflows are irreversible enough to demand at least one
# destructive step. ``calendar_block`` and ``notes_daily`` may legitimately
# have zero destructive steps because their actions are easily undone.
DESTRUCTIVE_REQUIRED: frozenset[str] = frozenset(
    {"gmail_reply", "finder_organize", "slack_status"}
)


def _fixture_dir(slug: str) -> Path:
    return FIXTURES_SKILLS_DIR / slug


def _load_pair(slug: str) -> tuple[str, dict[str, object]]:
    skill_md = (_fixture_dir(slug) / "SKILL.md").read_text(encoding="utf-8")
    meta = json.loads((_fixture_dir(slug) / "skill.meta.json").read_text(encoding="utf-8"))
    assert isinstance(meta, dict)
    return skill_md, meta


@pytest.mark.parametrize("slug", GOLDEN_SLUGS)
def test_golden_skill_dir_exists(slug: str) -> None:
    assert _fixture_dir(slug).is_dir(), f"missing fixtures/skills/{slug}/"
    assert (_fixture_dir(slug) / "SKILL.md").is_file()
    assert (_fixture_dir(slug) / "skill.meta.json").is_file()


@pytest.mark.parametrize("slug", GOLDEN_SLUGS)
def test_meta_passes_schema(slug: str) -> None:
    _, meta = _load_pair(slug)
    validate_meta(meta)  # raises on failure


@pytest.mark.parametrize("slug", GOLDEN_SLUGS)
def test_meta_slug_matches_directory(slug: str) -> None:
    _, meta = _load_pair(slug)
    assert meta["slug"] == slug


@pytest.mark.parametrize("slug", GOLDEN_SLUGS)
def test_markdown_round_trip(slug: str) -> None:
    markdown, _ = _load_pair(slug)
    parsed = parse_skill_md(markdown)
    re_parsed = parse_skill_md(render_skill_md(parsed))
    assert re_parsed == parsed, (
        f"{slug} failed the round-trip invariant — parsing the rendered form "
        "returned a structurally different ParsedSkill"
    )
    # And once more: re-rendering the re-parsed form matches the first render.
    assert render_skill_md(re_parsed) == render_skill_md(parsed)


@pytest.mark.parametrize("slug", GOLDEN_SLUGS)
def test_markdown_meta_cross_check(slug: str) -> None:
    markdown, meta = _load_pair(slug)
    validate_meta_against_markdown(meta, markdown)  # raises on failure


@pytest.mark.parametrize("slug", GOLDEN_SLUGS)
def test_minimum_step_count(slug: str) -> None:
    markdown, _ = _load_pair(slug)
    parsed = parse_skill_md(markdown)
    assert len(parsed.steps) >= 4, (
        f"{slug} has only {len(parsed.steps)} steps; golden fixtures must have at least 4"
    )


@pytest.mark.parametrize("slug", GOLDEN_SLUGS)
def test_at_least_one_parameter(slug: str) -> None:
    _, meta = _load_pair(slug)
    params = meta["parameters"]
    assert isinstance(params, list)
    assert len(params) >= 1, f"{slug} has zero parameters; golden fixtures must have at least 1"


@pytest.mark.parametrize("slug", GOLDEN_SLUGS)
def test_destructive_step_policy(slug: str) -> None:
    _, meta = _load_pair(slug)
    destructive = meta["destructive_steps"]
    assert isinstance(destructive, list)
    if slug in DESTRUCTIVE_REQUIRED:
        assert len(destructive) >= 1, (
            f"{slug} declares zero destructive steps, but its real-world workflow "
            "is irreversible — the golden fixture must flag at least one step"
        )


@pytest.mark.parametrize("slug", GOLDEN_SLUGS)
def test_trajectory_id_resolves(slug: str) -> None:
    _, meta = _load_pair(slug)
    trajectory_id = meta["trajectory_id"]
    found = False
    for traj_dir in FIXTURES_TRAJECTORIES_DIR.iterdir():
        metadata_path = traj_dir / "metadata.json"
        if not metadata_path.is_file():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("id") == trajectory_id:
            found = True
            break
    assert found, (
        f"{slug}: meta.trajectory_id {trajectory_id!r} does not match any "
        "fixtures/trajectories/<dir>/metadata.json id field"
    )


def test_fixtures_skills_total_size_under_1mb() -> None:
    total = 0
    for path in FIXTURES_SKILLS_DIR.rglob("*"):
        if path.is_file():
            total += path.stat().st_size
    assert total < 1_000_000, (
        f"fixtures/skills/ is {total} bytes; golden fixtures must stay under 1 MB "
        "(screenshots belong in fixtures/trajectories/, not here)"
    )


def test_check_all_fixtures_passes() -> None:
    result = check_all_fixtures()
    assert result.ok, "\n".join(result.failures)


def test_check_fixtures_script_exits_zero() -> None:
    script = REPO_ROOT / "scripts" / "check_fixtures.sh"
    assert script.is_file(), f"missing {script}"
    completed = subprocess.run(
        [str(script)],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, (
        f"check_fixtures.sh failed — stdout={completed.stdout!r} stderr={completed.stderr!r}"
    )
