"""Real-API smoke test for synthesizer draft generation (S-017).

Gated behind ``TRACE_REAL_API_TESTS=1``; skipped during the standard test run
and during every Ralph iteration. Requires a real ``ANTHROPIC_API_KEY`` in the
environment.

For each of the five reference trajectories under
``fixtures/trajectories/``, this test runs :func:`generate_draft
<synthesizer.draft.generate_draft>` against the live Sonnet 4.5 endpoint and
then scores the produced :class:`~synthesizer.skill_doc.ParsedSkill` against
the hand-crafted golden via :func:`score_skill_similarity
<synthesizer.similarity.score_skill_similarity>` (Haiku 4.5). Per-fixture
results are aggregated into ``tests/synthesizer/smoke_report.json`` so a human
reviewer can inspect drift before merging ``feat/synthesizer`` to ``main``.

Assertions (per the PRD's S-017 acceptance criteria):

* ``destructive_match == 1.0`` on all five fixtures — non-negotiable. The
  secondary keyword matcher (S-008) is the structural enforcement.
* ``overall >= 0.80`` on at least four of five fixtures — tolerates LLM drift.
* Total cost across the run < $2 (caps runaway spend).

This test does NOT block the Ralph completion promise. ``SYNTHESIZER_DONE`` is
based on the fake-mode tests only; the smoke test is a manual pre-merge gate.
See ``tests/synthesizer/README.md`` for instructions on running it.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from synthesizer.draft import DraftResult, generate_draft
from synthesizer.llm_client import LLMClient
from synthesizer.preprocess import preprocess_trajectory
from synthesizer.schema import validate_meta, validate_meta_against_markdown
from synthesizer.similarity import SimilarityScore, score_skill_similarity
from synthesizer.skill_doc import ParsedSkill, parse_skill_md
from synthesizer.trajectory_reader import TrajectoryReader

REFERENCE_SLUGS: tuple[str, ...] = (
    "gmail_reply",
    "calendar_block",
    "finder_organize",
    "slack_status",
    "notes_daily",
)

OVERALL_SCORE_THRESHOLD: float = 0.80
"""Per-fixture minimum on :attr:`SimilarityScore.overall` to count as passing."""

MIN_FIXTURES_PASSING_OVERALL: int = 4
"""At least four of the five reference fixtures must clear the threshold;
LLM non-determinism makes a strict 5/5 too brittle for a CI-style assertion."""

TOTAL_COST_CAP_USD: float = 2.0
"""Hard cap on cumulative spend across the whole smoke run. Matches the PRD's
``apiCostBudgetForSmoke`` constraint and protects against a runaway retry loop."""


# Module-level skip: when ``TRACE_REAL_API_TESTS`` is unset, every test in this
# module skips before fixture setup runs, so nothing else in the module needs
# guarding.
pytestmark = pytest.mark.skipif(
    not os.environ.get("TRACE_REAL_API_TESTS"),
    reason="Set TRACE_REAL_API_TESTS=1 to run the real-API smoke test.",
)


# --- Fixture overrides ----------------------------------------------------
#
# The package-wide ``conftest.py`` autouses ``_force_fake_mode`` (forces
# ``TRACE_LLM_FAKE_MODE=1`` + dummy API key) and ``_no_stray_anthropic_calls``
# (installs a respx router that turns any call to api.anthropic.com into an
# error). The whole point of this test is to hit the real endpoint, so we
# replace both fixtures locally — pytest fixture resolution prefers the
# fixture closest to the test, so module-scoped overrides win.


@pytest.fixture(autouse=True)
def _force_fake_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Override the conftest fake-mode forcing fixture.

    Unsets ``TRACE_LLM_FAKE_MODE`` and verifies a real ``ANTHROPIC_API_KEY`` is
    present; ``TRACE_PROFILE=dev`` keeps any disk writes (cost log, etc.)
    inside the dev profile so the smoke test does not pollute the prod data
    directory.
    """
    monkeypatch.delenv("TRACE_LLM_FAKE_MODE", raising=False)
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key or api_key.startswith("test-"):
        pytest.fail(
            "TRACE_REAL_API_TESTS=1 requires a real ANTHROPIC_API_KEY in the env "
            "(must not start with 'test-')."
        )
    monkeypatch.setenv("TRACE_PROFILE", "dev")


@pytest.fixture(autouse=True)
def _no_stray_anthropic_calls() -> None:
    """Override the conftest network guard so calls to api.anthropic.com flow."""
    return None


@pytest.fixture
def anthropic_mock() -> Iterator[None]:
    """Override the conftest respx router with a no-op so nothing is mocked."""
    yield None


# --- Path resolution ------------------------------------------------------


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "fixtures" / "trajectories").is_dir():
            return candidate
    raise FileNotFoundError("Could not locate repo root with fixtures/")


REPO_ROOT: Path = _repo_root()
TRAJECTORIES_ROOT: Path = REPO_ROOT / "fixtures" / "trajectories"
GOLDEN_SKILLS_ROOT: Path = REPO_ROOT / "fixtures" / "skills"
SMOKE_REPORT_PATH: Path = Path(__file__).resolve().parent / "smoke_report.json"


# --- Fixture-loading helpers ----------------------------------------------


def _load_golden(slug: str) -> ParsedSkill:
    md_path = GOLDEN_SKILLS_ROOT / slug / "SKILL.md"
    meta_path = GOLDEN_SKILLS_ROOT / slug / "skill.meta.json"
    markdown = md_path.read_text(encoding="utf-8")
    parsed = parse_skill_md(markdown)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    # Fail fast if the committed golden is itself invalid; the smoke test
    # would otherwise mask a fixture regression as a model regression.
    validate_meta(meta)
    validate_meta_against_markdown(meta, markdown)
    return parsed


def _generate_draft_for_slug(slug: str, client: LLMClient) -> DraftResult:
    reader = TrajectoryReader(TRAJECTORIES_ROOT / slug)
    preprocessed = preprocess_trajectory(reader)
    return generate_draft(
        preprocessed,
        client,
        reader=reader,
        context_label=f"smoke:{slug}:draft",
    )


# --- Smoke test -----------------------------------------------------------


def test_real_api_smoke_against_golden_fixtures() -> None:
    """End-to-end real-API draft + similarity check on all 5 reference fixtures.

    See the module docstring for thresholds and rationale. Writes a per-run
    report to ``tests/synthesizer/smoke_report.json``.
    """
    client = LLMClient()
    per_fixture: list[dict[str, Any]] = []
    total_cost_usd = 0.0
    overall_pass_count = 0
    destructive_pass_count = 0

    for slug in REFERENCE_SLUGS:
        draft = _generate_draft_for_slug(slug, client)
        golden = _load_golden(slug)
        score: SimilarityScore = score_skill_similarity(
            draft.parsed,
            golden,
            client,
            context_label=f"smoke:{slug}:similarity",
        )

        total_cost_usd += draft.total_cost_usd
        if score.overall >= OVERALL_SCORE_THRESHOLD:
            overall_pass_count += 1
        if score.destructive_match == 1.0:
            destructive_pass_count += 1

        per_fixture.append(
            {
                "slug": slug,
                "draft_llm_calls": draft.llm_calls,
                "draft_cost_usd": draft.total_cost_usd,
                "step_count": len(draft.parsed.steps),
                "destructive_step_count": sum(
                    1 for s in draft.parsed.steps if s.destructive
                ),
                "scores": {
                    "overall": score.overall,
                    "step_coverage": score.step_coverage,
                    "parameter_match": score.parameter_match,
                    "destructive_match": score.destructive_match,
                },
                "reasoning": score.reasoning,
            }
        )

    report: dict[str, Any] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "model_draft": "claude-sonnet-4-5",
        "model_similarity": "claude-haiku-4-5",
        "thresholds": {
            "overall_score_threshold": OVERALL_SCORE_THRESHOLD,
            "min_fixtures_passing_overall": MIN_FIXTURES_PASSING_OVERALL,
            "total_cost_cap_usd": TOTAL_COST_CAP_USD,
        },
        "results": {
            "total_cost_usd": total_cost_usd,
            "overall_pass_count": overall_pass_count,
            "destructive_pass_count": destructive_pass_count,
            "fixture_count": len(REFERENCE_SLUGS),
        },
        "fixtures": per_fixture,
    }
    SMOKE_REPORT_PATH.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    # Cost cap first — surfaces a runaway spend before any score assertion.
    assert total_cost_usd < TOTAL_COST_CAP_USD, (
        f"Smoke run exceeded cost cap: ${total_cost_usd:.4f} >= "
        f"${TOTAL_COST_CAP_USD:.2f}. See {SMOKE_REPORT_PATH} for the breakdown."
    )
    # Destructive flagging is safety-critical; the secondary matcher in S-008
    # makes 5/5 deterministic on the reference fixtures.
    assert destructive_pass_count == len(REFERENCE_SLUGS), (
        f"destructive_match != 1.0 on "
        f"{len(REFERENCE_SLUGS) - destructive_pass_count} fixture(s). "
        f"Destructive flagging must match the golden exactly. "
        f"See {SMOKE_REPORT_PATH}."
    )
    assert overall_pass_count >= MIN_FIXTURES_PASSING_OVERALL, (
        f"Only {overall_pass_count}/{len(REFERENCE_SLUGS)} fixtures scored "
        f">= {OVERALL_SCORE_THRESHOLD} on overall similarity (need "
        f">= {MIN_FIXTURES_PASSING_OVERALL}). See {SMOKE_REPORT_PATH}."
    )
