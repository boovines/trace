"""Runner cost budgets: per-execution and per-day caps (X-023).

A runaway retry loop or a stuck agent can silently rack up Anthropic bills.
This module is the policy layer on top of :mod:`runner.budget` (which
enforces per-run token/time/action caps) and adds:

* **Per-execution USD cap** — enforced via :class:`runner.budget.BudgetTracker`
  by setting ``RunBudget.max_cost_usd``. Trip maps to
  ``abort_reason="per_run_cost_cap"``.
* **Daily USD cap** — enforced at ``POST /run/start`` by summing today's
  ``runner`` entries in ``costs.jsonl``.
* **80% warnings** — emitted to stderr and broadcast as a ``warning``
  WebSocket message so the UI can show a banner.

The dev and prod profiles have different defaults: dev is tight (Ralph
iterations run in dev, and we never want an iteration to burn more than a
few dollars/day on the runner), prod is looser but still finite.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Final

from runner.paths import PROFILE_ENV_VAR, profile_root

logger = logging.getLogger(__name__)

CONFIG_FILENAME: Final[str] = "config.json"
WARNING_THRESHOLD: Final[float] = 0.80

PROD_PER_EXECUTION_USD_CAP: Final[float] = 2.00
PROD_DAILY_USD_CAP: Final[float] = 20.00
DEV_PER_EXECUTION_USD_CAP: Final[float] = 0.50
DEV_DAILY_USD_CAP: Final[float] = 2.00

_VALID_KEYS: Final[frozenset[str]] = frozenset(
    {"run_per_execution_usd_cap", "run_daily_usd_cap"}
)


class ConfigurationError(ValueError):
    """Raised when ``config.json`` is present but malformed."""


@dataclass(frozen=True, slots=True)
class RunnerBudgetConfig:
    """Per-execution and per-day runner cost caps, both in USD."""

    run_per_execution_usd_cap: float
    run_daily_usd_cap: float

    def __post_init__(self) -> None:
        for name in ("run_per_execution_usd_cap", "run_daily_usd_cap"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ConfigurationError(
                    f"{name} must be a number, got {type(value).__name__}"
                )
            if value <= 0:
                raise ConfigurationError(
                    f"{name} must be > 0, got {value!r}"
                )


def default_config_for_profile() -> RunnerBudgetConfig:
    """Return the default caps for the active profile (dev vs prod)."""

    if os.environ.get(PROFILE_ENV_VAR) == "prod":
        return RunnerBudgetConfig(
            run_per_execution_usd_cap=PROD_PER_EXECUTION_USD_CAP,
            run_daily_usd_cap=PROD_DAILY_USD_CAP,
        )
    return RunnerBudgetConfig(
        run_per_execution_usd_cap=DEV_PER_EXECUTION_USD_CAP,
        run_daily_usd_cap=DEV_DAILY_USD_CAP,
    )


def load_runner_budget(config_path: Path | None = None) -> RunnerBudgetConfig:
    """Load runner caps from ``<profile>/config.json`` with profile defaults.

    Missing file → profile defaults. Malformed file (invalid JSON, wrong
    shape, non-numeric values, unknown keys, non-positive values) → raise
    :class:`ConfigurationError` so the caller can surface a 500 instead of
    silently ignoring configuration drift.
    """

    path = config_path if config_path is not None else profile_root() / CONFIG_FILENAME
    defaults = default_config_for_profile()
    if not path.is_file():
        return defaults

    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigurationError(f"could not read {path}: {exc}") from exc

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ConfigurationError(
            f"{path} is not valid JSON: {exc}"
        ) from exc

    if not isinstance(data, dict):
        raise ConfigurationError(
            f"{path} must contain a JSON object, got {type(data).__name__}"
        )

    unknown = set(data) - _VALID_KEYS
    if unknown:
        raise ConfigurationError(
            f"{path} contains unknown keys: {sorted(unknown)}"
        )

    per_run = data.get("run_per_execution_usd_cap", defaults.run_per_execution_usd_cap)
    per_day = data.get("run_daily_usd_cap", defaults.run_daily_usd_cap)
    return RunnerBudgetConfig(
        run_per_execution_usd_cap=per_run,
        run_daily_usd_cap=per_day,
    )


def sum_daily_runner_cost_usd(
    costs_path: Path, *, now: datetime | None = None
) -> float:
    """Sum ``cost_estimate_usd`` of today's ``runner`` entries in costs.jsonl.

    "Today" is UTC — ``costs.jsonl`` entries are written by
    :class:`runner.claude_runtime.ClaudeRuntime` with
    ``timestamp_ms = int(time.time() * 1000)``, which is UTC-based.

    Missing file → 0.0 (a fresh install has no cost history). Malformed
    lines are skipped with a warning — the daily cap must remain enforceable
    even if a past writer crashed mid-line.
    """

    if not costs_path.is_file():
        return 0.0

    reference = (now or datetime.now(UTC)).astimezone(UTC).date()
    total = 0.0
    try:
        text = costs_path.read_text(encoding="utf-8")
    except OSError as exc:  # pragma: no cover - defensive
        logger.warning("could not read %s: %s", costs_path, exc)
        return 0.0

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            record = json.loads(stripped)
        except json.JSONDecodeError:
            logger.warning("skipping malformed costs.jsonl line: %r", stripped[:120])
            continue
        if not isinstance(record, dict):
            continue
        if record.get("module") != "runner":
            continue
        ts_ms = record.get("timestamp_ms")
        if not isinstance(ts_ms, (int, float)) or isinstance(ts_ms, bool):
            continue
        entry_date = _utc_date_from_ms(int(ts_ms))
        if entry_date != reference:
            continue
        cost = record.get("cost_estimate_usd")
        if isinstance(cost, (int, float)) and not isinstance(cost, bool):
            total += float(cost)
    return total


def _utc_date_from_ms(timestamp_ms: int) -> date:
    return datetime.fromtimestamp(timestamp_ms / 1000.0, tz=UTC).date()


def crossed_warning_threshold(
    prev_value: float, new_value: float, cap: float
) -> bool:
    """True iff ``new_value`` just crossed the 80% warning threshold.

    Uses a prev/new comparison so repeated checks after the initial crossing
    do not re-log. Cap must be > 0 for the threshold to be meaningful.
    """

    if cap <= 0:
        return False
    threshold = WARNING_THRESHOLD * cap
    return prev_value < threshold <= new_value


__all__ = [
    "CONFIG_FILENAME",
    "DEV_DAILY_USD_CAP",
    "DEV_PER_EXECUTION_USD_CAP",
    "PROD_DAILY_USD_CAP",
    "PROD_PER_EXECUTION_USD_CAP",
    "WARNING_THRESHOLD",
    "ConfigurationError",
    "RunnerBudgetConfig",
    "crossed_warning_threshold",
    "default_config_for_profile",
    "load_runner_budget",
    "sum_daily_runner_cost_usd",
]
