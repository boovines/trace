"""Cost budget guardrails for the synthesizer.

Two caps defend against runaway spend:

* **Per-session cap** — the cumulative cost of draft + revisions for a single
  synthesis session. Checked after every LLM-costing operation; when exceeded,
  the owning :class:`~synthesizer.session.SynthesisSession` transitions to
  ``errored`` with ``error='per-session cost cap exceeded'``.
* **Daily cap** — the total cost of all synthesizer entries in ``costs.jsonl``
  for today (UTC). Checked before accepting ``POST /synthesize/start``; when
  exceeded, the request returns HTTP 429.

Caps live in ``<trace_data_dir>/config.json``. Defaults (1.00 / 5.00 USD) kick
in when the file is absent. Invalid values (non-numeric, zero, negative) raise
:class:`~synthesizer.llm_client.ConfigurationError` so the operator notices
before a live session silently skips the guardrail.

Warnings fire to stderr at 80% of either cap — once per session for the
per-session cap, once per monitor lifetime for the daily cap. The EXCEEDED
status is load-bearing (drives state transitions); the WARNING is a soft
signal that lets the UI suggest saving work before the cap hits.
"""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from synthesizer.llm_client import ConfigurationError, costs_log_path, trace_data_dir

__all__ = [
    "BUDGET_CONFIG_FILENAME",
    "DEFAULT_DAILY_CAP_USD",
    "DEFAULT_SESSION_CAP_USD",
    "WARNING_THRESHOLD",
    "BudgetCheckResult",
    "BudgetConfig",
    "BudgetMonitor",
    "BudgetStatus",
    "budget_config_path",
    "load_budget_config",
]

LOGGER = logging.getLogger(__name__)

DEFAULT_SESSION_CAP_USD: float = 1.00
DEFAULT_DAILY_CAP_USD: float = 5.00
WARNING_THRESHOLD: float = 0.80
BUDGET_CONFIG_FILENAME: str = "config.json"


class BudgetStatus(StrEnum):
    """Tri-state budget classification: OK, WARNING (at 80%), EXCEEDED (at 100%)."""

    OK = "ok"
    WARNING = "warning"
    EXCEEDED = "exceeded"


@dataclass(frozen=True)
class BudgetConfig:
    """Loaded per-profile budget caps."""

    synthesis_per_session_usd_cap: float
    synthesis_daily_usd_cap: float


@dataclass(frozen=True)
class BudgetCheckResult:
    """Outcome of a single budget check.

    ``current_usd`` is the running total that was measured; ``cap_usd`` is the
    ceiling the monitor compared against; ``status`` is the tri-state
    classification. Callers transition session state off ``status``, not off the
    raw numbers — the monitor owns the threshold arithmetic.
    """

    status: BudgetStatus
    current_usd: float
    cap_usd: float

    @property
    def percent_used(self) -> float:
        if self.cap_usd <= 0:
            return 0.0
        return self.current_usd / self.cap_usd


# --- Config loading --------------------------------------------------------


def budget_config_path() -> Path:
    """Per-profile path to ``config.json``."""
    return trace_data_dir() / BUDGET_CONFIG_FILENAME


def _validate_cap(key: str, value: Any) -> float:
    # Reject booleans explicitly — ``bool`` is a subclass of ``int`` in Python
    # and would otherwise sneak through the numeric check.
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigurationError(
            f"{key} must be a positive number, got {type(value).__name__}"
        )
    if value <= 0:
        raise ConfigurationError(f"{key} must be positive, got {value}")
    return float(value)


def load_budget_config(path: Path | None = None) -> BudgetConfig:
    """Read ``config.json`` and return a :class:`BudgetConfig`.

    Missing file → defaults. Malformed JSON, wrong top-level type, or invalid
    cap values → :class:`ConfigurationError`. Unknown keys are ignored so the
    config can grow over time without breaking older profiles.
    """
    target = path if path is not None else budget_config_path()
    if not target.exists():
        return BudgetConfig(
            synthesis_per_session_usd_cap=DEFAULT_SESSION_CAP_USD,
            synthesis_daily_usd_cap=DEFAULT_DAILY_CAP_USD,
        )
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError as err:
        raise ConfigurationError(
            f"{target} is not valid JSON: {err}"
        ) from err
    if not isinstance(raw, dict):
        raise ConfigurationError(
            f"{target} must be a JSON object, got {type(raw).__name__}"
        )

    per_session = raw.get("synthesis_per_session_usd_cap", DEFAULT_SESSION_CAP_USD)
    daily = raw.get("synthesis_daily_usd_cap", DEFAULT_DAILY_CAP_USD)

    return BudgetConfig(
        synthesis_per_session_usd_cap=_validate_cap(
            "synthesis_per_session_usd_cap", per_session
        ),
        synthesis_daily_usd_cap=_validate_cap("synthesis_daily_usd_cap", daily),
    )


# --- Monitor ---------------------------------------------------------------


class BudgetMonitor:
    """Per-synthesizer-process budget enforcer.

    The monitor is intentionally cheap to construct — config is loaded once at
    init (or injected for tests), and each :meth:`check_daily_cost` call
    re-scans ``costs.jsonl`` from scratch. The daily log is small in practice
    (one line per LLM call, retention undefined in v1) so a linear scan is
    fine; if that changes, cache by file mtime.

    Warnings are deduplicated: per-session warnings fire at most once per
    session id, and the daily-cap warning fires at most once per monitor
    lifetime (the monitor is owned by the HTTP layer, so this means at most
    once per synthesizer process per 80% crossing).
    """

    def __init__(
        self,
        *,
        config: BudgetConfig | None = None,
        costs_path: Path | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._config: BudgetConfig = (
            config if config is not None else load_budget_config()
        )
        self._costs_path_override = costs_path
        self._clock: Callable[[], datetime] = clock or (lambda: datetime.now(UTC))
        self._warned_sessions: set[str] = set()
        self._warned_daily: bool = False

    @property
    def config(self) -> BudgetConfig:
        return self._config

    def _costs_path(self) -> Path:
        if self._costs_path_override is not None:
            return self._costs_path_override
        return costs_log_path()

    # -- per-session -------------------------------------------------------

    def check_session_cost(
        self, session_id: str, current_cost: float
    ) -> BudgetCheckResult:
        """Classify ``current_cost`` against the per-session cap.

        Emits a one-shot stderr warning the first time a given ``session_id``
        crosses 80%. The caller decides what to do on ``EXCEEDED`` — the
        session layer transitions to ``errored``; the API layer can short-
        circuit before the next LLM call.
        """
        cap = self._config.synthesis_per_session_usd_cap
        result = self._classify(current_cost, cap)
        if (
            result.status == BudgetStatus.WARNING
            and session_id not in self._warned_sessions
        ):
            self._warned_sessions.add(session_id)
            self._emit_warning(
                f"synthesis session {session_id} at "
                f"{result.percent_used * 100:.0f}% of ${cap:.2f} per-session cap "
                f"(current ${current_cost:.4f})"
            )
        return result

    # -- daily --------------------------------------------------------------

    def check_daily_cost(self) -> BudgetCheckResult:
        """Classify today's cumulative synthesizer cost against the daily cap.

        "Today" is the UTC calendar day of :func:`datetime.now` — keeps the
        accounting consistent with the UTC ``timestamp_iso`` written into
        ``costs.jsonl``.
        """
        today = self._clock().date()
        current = self._sum_today(today)
        cap = self._config.synthesis_daily_usd_cap
        result = self._classify(current, cap)
        if result.status == BudgetStatus.WARNING and not self._warned_daily:
            self._warned_daily = True
            self._emit_warning(
                f"synthesizer at {result.percent_used * 100:.0f}% of "
                f"${cap:.2f} daily cap (current ${current:.4f})"
            )
        return result

    def _sum_today(self, today: date) -> float:
        path = self._costs_path()
        if not path.exists():
            return 0.0
        total = 0.0
        with path.open("r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    # A malformed line is a bug elsewhere but must not take
                    # down the budget check — skip and keep going.
                    LOGGER.warning("Skipping malformed costs.jsonl line: %r", line)
                    continue
                if entry.get("module") != "synthesizer":
                    continue
                ts = entry.get("timestamp_iso")
                if not isinstance(ts, str):
                    continue
                try:
                    dt = datetime.fromisoformat(ts)
                except ValueError:
                    continue
                if dt.date() != today:
                    continue
                cost = entry.get("cost_estimate_usd")
                if isinstance(cost, bool):
                    continue
                if isinstance(cost, int | float):
                    total += float(cost)
        return total

    # -- shared -------------------------------------------------------------

    def _classify(self, current: float, cap: float) -> BudgetCheckResult:
        if cap <= 0:
            # Guard against a misconfigured cap making everything look OK —
            # :func:`load_budget_config` rejects this, but a caller that
            # constructs a :class:`BudgetConfig` directly could sneak one in.
            return BudgetCheckResult(
                status=BudgetStatus.OK, current_usd=current, cap_usd=cap
            )
        ratio = current / cap
        if ratio >= 1.0:
            status = BudgetStatus.EXCEEDED
        elif ratio >= WARNING_THRESHOLD:
            status = BudgetStatus.WARNING
        else:
            status = BudgetStatus.OK
        return BudgetCheckResult(
            status=status, current_usd=current, cap_usd=cap
        )

    def _emit_warning(self, message: str) -> None:
        # stderr directly rather than LOGGER.warning so the line reaches the
        # user's console even when logging is configured at a higher level
        # (the PRD calls this out explicitly).
        print(f"WARNING: {message}", file=sys.stderr)
        LOGGER.warning(message)
