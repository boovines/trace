"""Runtime orchestration of runs: start, track, and stream their lifecycle.

The FastAPI layer (``runner.api``) is a thin translator between HTTP and this
module. Here is where all the executor collaborators are wired up:

* Per-run :class:`~runner.run_writer.RunWriter` wrapped with
  :class:`~runner.observing_writer.ObservingRunWriter` so status changes,
  events, and turn-completions are fan-outed to WebSocket subscribers.
* Per-run :class:`~runner.confirmation.ConfirmationQueue` wrapped so a
  ``push_request`` also publishes a ``confirmation_request`` stream event.
* Shared process-global :class:`~runner.kill_switch.KillSwitch` reached via
  :func:`~runner.kill_switch.get_global_kill_switch`.
* Per-run :class:`~runner.claude_runtime.ClaudeRuntime` in fake-mode for
  Ralph iterations (the fake-mode env var is the default contract) and in
  real-API mode otherwise.

The manager does NOT own the event loop; it is mounted inside FastAPI's own
loop and every task it creates is scheduled on that loop. The
:class:`RunManager` is a singleton for the process, held by the FastAPI app
state.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from runner import paths
from runner.budget import BudgetTracker, RunBudget
from runner.budget_config import (
    RunnerBudgetConfig,
    crossed_warning_threshold,
    load_runner_budget,
    sum_daily_runner_cost_usd,
)
from runner.claude_runtime import ClaudeRuntime
from runner.confirmation import (
    ConfirmationDecision,
    ConfirmationQueue,
    ConfirmationRequest,
    make_request_message,
)
from runner.event_stream import EventBroadcaster
from runner.execution_hints import CapabilityRegistry, default_capability_registry
from runner.executor import Executor
from runner.input_adapter import DryRunInputAdapter, InputAdapter
from runner.kill_switch import KillSwitch, get_global_kill_switch
from runner.mcp_client import (
    MCPCallDispatcher,
    MCPProbeError,
    MCPServerConfig,
    format_probe_report,
    load_server_configs,
    probe_capabilities_sync,
)
from runner.observing_writer import ObservingRunWriter
from runner.pre_action_gate import AXResolver, AXTarget
from runner.run_index import RunIndex
from runner.safety import is_live_mode_allowed
from runner.schema import RunMetadata, RunMode
from runner.screen_source import ScreenSource, TrajectoryScreenSource
from runner.skill_loader import (
    LoadedSkill,
    SkillNotFoundError,
    load_skill,
)

logger = logging.getLogger(__name__)


class _BackgroundLoop:
    """Owns a thread running a dedicated asyncio event loop for run tasks.

    Run tasks must outlive the HTTP request that created them. In production
    FastAPI runs a single long-lived loop, but test clients (and any harness
    that uses httpx's sync transport) spin up a fresh loop per request and
    tear it down on return — which would otherwise cancel the run task
    immediately. Keeping run tasks on this loop makes run lifetime
    independent of any request handler's loop.
    """

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="runner-bg-loop", daemon=True
        )
        self._thread.start()
        self._ready.wait()

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._ready.set()
        self._loop.run_forever()

    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        return self._loop

    def submit(
        self, coro: Any
    ) -> concurrent.futures.Future[Any]:
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    def stop(self) -> None:
        if self._loop.is_closed():
            return
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=2.0)


class DailyCapExceeded(RuntimeError):
    """Raised when the daily runner cost cap blocks a new run start.

    The API layer maps this to HTTP 429 so the UI can surface the hit cap
    without crashing. The message names the cap and today's spend.
    """

    def __init__(self, *, today_spend_usd: float, cap_usd: float) -> None:
        super().__init__(
            f"daily runner cost cap reached: ${today_spend_usd:.4f} spent "
            f"today >= ${cap_usd:.2f} cap"
        )
        self.today_spend_usd = today_spend_usd
        self.cap_usd = cap_usd


class LiveModeNotEnabled(RuntimeError):
    """Raised when a caller asks for execute mode with TRACE_ALLOW_LIVE unset.

    The API layer catches this and returns a 400 with a message that names
    the env var, so the UI can display an actionable error instead of a 500.
    """


class RunNotFound(KeyError):
    """Raised when ``run_id`` isn't a directory under the runs root."""


class InvalidRunState(RuntimeError):
    """Raised when an operation is illegal for the run's current state.

    Used for ``/confirm`` on a run that isn't awaiting confirmation.
    """


class _NullAXResolver:
    """AX resolver that always returns Unknown.

    Used by the dry-run path (gate short-circuits to Allow) and as a
    placeholder in execute mode until the recorder's ax_resolver ships on
    ``feat/recorder``.
    """

    def resolve_at(self, x_pt: float, y_pt: float) -> AXTarget | None:
        return None


class _BroadcastingConfirmationQueue(ConfirmationQueue):
    """ConfirmationQueue that also publishes push_request to the broadcaster.

    Subclass so the executor's existing callsites keep working — the runner
    already calls ``push_request``/``await_decision``/``submit_decision``
    through the :class:`ConfirmationQueue` API.
    """

    def __init__(self, broadcaster: EventBroadcaster) -> None:
        super().__init__()
        self._broadcaster = broadcaster

    def push_request(
        self,
        *,
        run_id: str,
        step_number: int,
        step_text: str,
        screenshot_ref: str | None,
        destructive_reason: str,
    ) -> ConfirmationRequest:
        request = super().push_request(
            run_id=run_id,
            step_number=step_number,
            step_text=step_text,
            screenshot_ref=screenshot_ref,
            destructive_reason=destructive_reason,
        )
        screenshot_url = (
            f"/run/{run_id}/screenshots/{screenshot_ref}"
            if screenshot_ref is not None
            else None
        )
        message = make_request_message(request, screenshot_url=screenshot_url)
        self._broadcaster.publish(run_id, message)
        return request


@dataclass
class RunHandle:
    """Per-run bookkeeping held by the :class:`RunManager`.

    ``task`` is an :class:`asyncio.Future` wrapping a
    :class:`concurrent.futures.Future` from the background event loop, so
    callers can ``await`` it from the current loop even though the run
    itself executes on the :class:`RunManager`'s background loop.
    """

    run_id: str
    skill_slug: str
    mode: RunMode
    task: asyncio.Future[RunMetadata]
    confirmation_queue: _BroadcastingConfirmationQueue
    broadcaster: EventBroadcaster
    kill_switch: KillSwitch
    run_dir: Path
    final_metadata: RunMetadata | None = None
    pending_confirmation: ConfirmationRequest | None = field(default=None)


@dataclass(frozen=True, slots=True)
class AdapterBundle:
    """Executor collaborators assembled by :meth:`RunManager._build_adapters`.

    Split out so tests can pre-build a bundle and inject it via
    :meth:`RunManager.set_adapter_factory` without reimplementing the live-mode
    wire-up.
    """

    input_adapter: InputAdapter
    screen_source: ScreenSource
    ax_resolver: AXResolver


AdapterFactory = Any  # Callable[[LoadedSkill, RunMode], AdapterBundle]


class RunManager:
    """Orchestrates run start, WebSocket streaming, confirmation, abort.

    Constructed once per FastAPI app. Paths default to
    :func:`runner.paths.runs_root` / :func:`runner.paths.skills_root`; tests
    override them for isolation.
    """

    def __init__(
        self,
        *,
        runs_root: Path | None = None,
        skills_root: Path | None = None,
        trajectories_root: Path | None = None,
        costs_path: Path | None = None,
        index_path: Path | None = None,
        kill_switch: KillSwitch | None = None,
        adapter_factory: AdapterFactory | None = None,
        runtime_factory: Any = None,
        config_path: Path | None = None,
        capability_registry: CapabilityRegistry | None = None,
        mcp_config_path: Path | None = None,
    ) -> None:
        self._runs_root = runs_root
        self._skills_root = skills_root
        self._trajectories_root = trajectories_root
        self._costs_path = costs_path
        self._index_path = index_path
        self._kill_switch = kill_switch or get_global_kill_switch()
        self._adapter_factory = adapter_factory
        self._runtime_factory = runtime_factory
        self._config_path = config_path
        self._mcp_config_path = mcp_config_path
        self._capability_registry: CapabilityRegistry | None = capability_registry
        self._capability_registry_lock = threading.Lock()
        # MCP server configs are loaded lazily alongside the registry
        # probe; the dispatcher (live during a run) needs them to open
        # its own session pool. Cache them so we don't re-read disk per
        # run.
        self._mcp_server_configs: list[MCPServerConfig] | None = None
        self._runs: dict[str, RunHandle] = {}
        self._broadcaster = EventBroadcaster()
        self._background: _BackgroundLoop | None = None
        self._background_lock = threading.Lock()
        self._index: RunIndex | None = None
        self._index_lock = threading.Lock()
        self._daily_warning_emitted_for: str | None = None

    def _get_background(self) -> _BackgroundLoop:
        with self._background_lock:
            if self._background is None:
                self._background = _BackgroundLoop()
            return self._background

    def _get_capability_registry(self) -> CapabilityRegistry:
        """Probe MCP servers once and cache the resulting registry.

        First call connects to every server in the configured
        ``mcp_servers.json`` (or ``$TRACE_MCP_CONFIG_PATH``), lists
        their tools, and caches the populated registry. Subsequent
        calls reuse the cache so per-run startup cost is constant.

        A broken config is logged but not fatal — we fall back to the
        conservative default (computer-use only). One bad MCP server
        likewise just stays out of the registry, see ``mcp_client``.
        """
        with self._capability_registry_lock:
            if self._capability_registry is not None:
                return self._capability_registry
            try:
                from runner.mcp_client import default_config_path

                config_path = self._mcp_config_path or default_config_path()
                self._mcp_server_configs = load_server_configs(config_path)
                report = probe_capabilities_sync(
                    configs=self._mcp_server_configs,
                    config_path=config_path,
                )
            except MCPProbeError as exc:
                logger.warning(
                    "MCP probe skipped (config error): %s — using "
                    "computer-use-only registry",
                    exc,
                )
                self._capability_registry = default_capability_registry()
                self._mcp_server_configs = []
                return self._capability_registry
            except Exception:
                logger.exception(
                    "MCP probe raised unexpectedly — falling back to "
                    "computer-use-only registry"
                )
                self._capability_registry = default_capability_registry()
                self._mcp_server_configs = []
                return self._capability_registry
            logger.info(
                "MCP probe complete:\n%s", format_probe_report(report)
            )
            self._capability_registry = report.registry
            return self._capability_registry

    def _make_mcp_dispatcher(self) -> MCPCallDispatcher | None:
        """Build a fresh dispatcher for one run, or ``None`` if no servers.

        Sessions opened by the dispatcher live for the run; we make a
        fresh dispatcher per run rather than sharing across runs so one
        run's tool-call state can't leak into another's. Restricts to
        servers that probed healthy — there's no point dialing a
        known-broken server during a run.
        """
        # Force the registry probe to run first so configs are cached.
        self._get_capability_registry()
        configs = self._mcp_server_configs or []
        if not configs:
            return None
        registry = self._capability_registry or default_capability_registry()
        live = [c for c in configs if c.name in registry.mcp_servers]
        if not live:
            return None
        return MCPCallDispatcher(live)

    def shutdown(self) -> None:
        """Stop the background loop. Idempotent; safe to skip in tests."""
        with self._background_lock:
            if self._background is not None:
                self._background.stop()
                self._background = None
        with self._index_lock:
            if self._index is not None:
                self._index.close()
                self._index = None

    @property
    def broadcaster(self) -> EventBroadcaster:
        return self._broadcaster

    @property
    def runs_root(self) -> Path:
        return self._runs_root if self._runs_root is not None else paths.runs_root()

    @property
    def skills_root(self) -> Path:
        return (
            self._skills_root
            if self._skills_root is not None
            else paths.skills_root()
        )

    @property
    def trajectories_root(self) -> Path:
        return (
            self._trajectories_root
            if self._trajectories_root is not None
            else paths.trajectories_root()
        )

    @property
    def costs_path(self) -> Path:
        return (
            self._costs_path if self._costs_path is not None else paths.costs_path()
        )

    @property
    def index_path(self) -> Path:
        if self._index_path is not None:
            return self._index_path
        return self.runs_root.parent / "index.db"

    def _get_index(self) -> RunIndex:
        with self._index_lock:
            if self._index is None:
                self._index = RunIndex(self.index_path)
            return self._index

    def reconcile_index(self) -> int:
        """Rebuild the SQLite index from disk. Called at service startup.

        Short-circuits when ``runs_root`` does not yet exist — a fresh install
        has nothing to reconcile and we would rather not materialise the index
        file in the user profile until the first real run is written.
        """
        if not self.runs_root.is_dir():
            return 0
        return self._get_index().reconcile(self.runs_root)

    def get(self, run_id: str) -> RunHandle | None:
        return self._runs.get(run_id)

    def _load_skill(self, slug: str) -> LoadedSkill:
        try:
            return load_skill(slug, self.skills_root)
        except SkillNotFoundError as exc:
            raise RunNotFound(f"skill not found: {slug}") from exc

    def _build_adapters(
        self, skill: LoadedSkill, mode: RunMode
    ) -> AdapterBundle:
        if self._adapter_factory is not None:
            bundle = self._adapter_factory(skill, mode)
            assert isinstance(bundle, AdapterBundle)
            return bundle
        if mode == "dry_run":
            trajectory_id = str(skill.meta.get("trajectory_ref", ""))
            return AdapterBundle(
                input_adapter=DryRunInputAdapter(),
                screen_source=TrajectoryScreenSource(
                    trajectory_id,
                    trajectories_root=self.trajectories_root,
                ),
                ax_resolver=_NullAXResolver(),
            )
        # execute mode: import live adapters lazily so the dry-run path has no
        # PyObjC dependency at import time.
        from runner.live_input import LiveInputAdapter
        from runner.live_screen import LiveScreenSource

        return AdapterBundle(
            input_adapter=LiveInputAdapter(),
            screen_source=LiveScreenSource(),
            ax_resolver=_NullAXResolver(),
        )

    def _build_runtime(self, run_id: str) -> Any:
        if self._runtime_factory is not None:
            return self._runtime_factory(run_id)
        return ClaudeRuntime(run_id=run_id, costs_path=self.costs_path)

    def load_budget_config(self) -> RunnerBudgetConfig:
        """Load runner cost caps for the active profile. Overridable in tests."""

        return load_runner_budget(self._config_path)

    async def start_run(
        self,
        *,
        skill_slug: str,
        parameters: dict[str, str],
        mode: RunMode,
    ) -> str:
        """Start a new run and return its ``run_id`` immediately.

        Raises:
            RunNotFound: ``skill_slug`` has no corresponding skill directory.
            LiveModeNotEnabled: ``mode='execute'`` while TRACE_ALLOW_LIVE is
                not set.
            DailyCapExceeded: today's summed runner cost is at or above the
                daily USD cap.
        """
        if mode == "execute" and not is_live_mode_allowed():
            raise LiveModeNotEnabled(
                "mode='execute' rejected: TRACE_ALLOW_LIVE is not set. "
                "Live runs require the operator to opt in explicitly."
            )

        budget_config = self.load_budget_config()
        today_spend = sum_daily_runner_cost_usd(self.costs_path)
        if today_spend >= budget_config.run_daily_usd_cap:
            raise DailyCapExceeded(
                today_spend_usd=today_spend,
                cap_usd=budget_config.run_daily_usd_cap,
            )
        self._maybe_log_daily_warning(today_spend, budget_config.run_daily_usd_cap)

        loaded = self._load_skill(skill_slug)
        bundle = self._build_adapters(loaded, mode)
        run_id = str(uuid.uuid4())

        writer = ObservingRunWriter(
            run_id=run_id,
            skill_slug=skill_slug,
            mode=mode,
            runs_root=self.runs_root,
            broadcaster=self._broadcaster,
            run_index=self._get_index(),
        )
        queue = _BroadcastingConfirmationQueue(self._broadcaster)
        skill_budget = RunBudget.from_skill_meta(loaded.meta)
        budget = _merge_cost_cap(
            skill_budget, budget_config.run_per_execution_usd_cap
        )
        tracker = BudgetTracker(budget=budget)
        runtime = self._build_runtime(run_id)

        broadcaster = self._broadcaster

        def _publish_per_run_warning(cost_usd: float, cap_usd: float) -> None:
            broadcaster.publish(
                run_id,
                {
                    "type": "warning",
                    "run_id": run_id,
                    "kind": "per_run_cost_80pct",
                    "cost_usd": cost_usd,
                    "cap_usd": cap_usd,
                },
            )

        executor = Executor(
            loaded_skill=loaded,
            parameters=parameters,
            mode=mode,
            agent_runtime=runtime,
            input_adapter=bundle.input_adapter,
            screen_source=bundle.screen_source,
            ax_resolver=bundle.ax_resolver,
            budget=tracker,
            writer=writer,
            confirmation_queue=queue,
            run_id=run_id,
            kill_switch=self._kill_switch,
            cost_warning_sink=_publish_per_run_warning,
            capability_registry=self._get_capability_registry(),
            mcp_dispatcher=self._make_mcp_dispatcher(),
        )

        async def _run_and_finalize() -> RunMetadata:
            try:
                final = await executor.run()
            except Exception:
                logger.exception("run %s crashed", run_id)
                self._broadcaster.publish(
                    run_id,
                    {
                        "type": "status_change",
                        "run_id": run_id,
                        "status": "failed",
                    },
                )
                raise
            finally:
                handle = self._runs.get(run_id)
                if handle is not None and "final" in locals():
                    handle.final_metadata = final
                    self._broadcaster.publish(
                        run_id,
                        {
                            "type": "done",
                            "run_id": run_id,
                            "final_metadata": final.to_dict(),
                        },
                    )
                self._broadcaster.close(run_id)
            return final

        background = self._get_background()
        concurrent_future = background.submit(_run_and_finalize())
        task: asyncio.Future[RunMetadata] = asyncio.wrap_future(
            concurrent_future, loop=asyncio.get_running_loop()
        )
        handle = RunHandle(
            run_id=run_id,
            skill_slug=skill_slug,
            mode=mode,
            task=task,
            confirmation_queue=queue,
            broadcaster=self._broadcaster,
            kill_switch=self._kill_switch,
            run_dir=writer.run_dir,
        )
        self._runs[run_id] = handle
        return run_id

    def submit_decision(
        self, run_id: str, decision: ConfirmationDecision
    ) -> None:
        """Forward a UI confirmation decision to the run's queue.

        Raises:
            RunNotFound: the run isn't tracked.
            InvalidRunState: the run has no pending confirmation.
        """
        handle = self._runs.get(run_id)
        if handle is None:
            raise RunNotFound(run_id)
        if not handle.confirmation_queue.has_pending(run_id):
            raise InvalidRunState(
                f"run {run_id!r} is not awaiting_confirmation"
            )
        handle.confirmation_queue.submit_decision(run_id, decision)

    def abort(self, run_id: str) -> bool:
        """Signal the run's kill switch. Idempotent.

        Returns ``True`` if a new abort signal was delivered, ``False``
        otherwise (already finished, never started, already aborting).
        """
        return self._kill_switch.kill(run_id, reason="user_abort")

    def list_runs(
        self,
        *,
        skill_slug: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Return a list of run summaries sorted by ``started_at`` descending.

        Served from the SQLite index (X-020). ``started_at`` may be ``None``
        for rows that have not transitioned past ``pending`` yet; the index
        sorts those below any row with a timestamp.
        """
        index = self._get_index()
        rows = index.list(skill_slug=skill_slug, limit=limit, offset=offset)
        return [
            {
                "run_id": r["run_id"],
                "skill_slug": r["skill_slug"],
                "status": r["status"],
                "mode": r["mode"],
                "started_at": r["started_at"],
                "ended_at": r["ended_at"],
                "duration_seconds": r["duration_seconds"],
                "total_cost_usd": r["total_cost_usd"],
            }
            for r in rows
        ]

    def get_metadata(self, run_id: str) -> dict[str, Any]:
        """Return the parsed ``run_metadata.json`` for ``run_id``."""
        import json

        path = self.runs_root / run_id / "run_metadata.json"
        if not path.is_file():
            raise RunNotFound(run_id)
        data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return data

    def get_events(self, run_id: str) -> list[dict[str, Any]]:
        """Return ``events.jsonl`` contents for ``run_id`` as a JSON array."""
        import json

        base = self.runs_root / run_id
        if not base.is_dir():
            raise RunNotFound(run_id)
        path = base / "events.jsonl"
        if not path.is_file():
            return []
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def _maybe_log_daily_warning(self, today_spend: float, cap_usd: float) -> None:
        """Log a stderr warning the first time today's spend crosses 80% of cap.

        Keyed by ``YYYY-MM-DD`` so a fresh UTC day re-arms the warning.
        """
        import datetime as _dt

        today = _dt.datetime.now(_dt.UTC).date().isoformat()
        if self._daily_warning_emitted_for == today:
            return
        if not crossed_warning_threshold(0.0, today_spend, cap_usd):
            return
        self._daily_warning_emitted_for = today
        logger.warning(
            "runner daily cost $%.4f exceeded 80%% of $%.2f cap (profile %s)",
            today_spend,
            cap_usd,
            today,
        )

    def screenshot_path(self, run_id: str, filename: str) -> Path:
        """Return the filesystem path for a run's screenshot.

        Rejects any ``filename`` containing a path separator so a caller
        cannot escape the screenshots directory.
        """
        if "/" in filename or "\\" in filename or filename in ("", ".", ".."):
            raise RunNotFound(f"invalid screenshot filename: {filename!r}")
        base = self.runs_root / run_id / "screenshots"
        if not base.is_dir():
            raise RunNotFound(run_id)
        candidate = base / filename
        if not candidate.is_file():
            raise RunNotFound(f"screenshot not found: {filename}")
        return candidate


def _merge_cost_cap(base: RunBudget, cost_cap_usd: float) -> RunBudget:
    """Return a copy of ``base`` with ``max_cost_usd`` set to the smaller value.

    If the skill already requested a cost cap via ``runtime_limits``, the
    minimum of the two wins — neither layer should be able to loosen the
    other. ``dataclasses.replace`` re-runs ``__post_init__`` so the copy is
    still validated.
    """
    from dataclasses import replace

    existing = base.max_cost_usd
    effective = cost_cap_usd if existing is None else min(existing, cost_cap_usd)
    return replace(base, max_cost_usd=effective)


__all__ = [
    "AdapterBundle",
    "DailyCapExceeded",
    "InvalidRunState",
    "LiveModeNotEnabled",
    "RunHandle",
    "RunManager",
    "RunNotFound",
]
