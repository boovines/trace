"""Main execution loop for the runner (X-017).

Ties together the LLM runtime, parser, dispatcher, pre-action gate, budget
tracker, confirmation queue, and run writer into a single orchestration
boundary. The API layer and CLI both enter the runner through
:meth:`Executor.run`; everything else is a collaborator injected at
construction time.

Per-turn flow
-------------

1. Pre-turn budget check. Hard trip → abort with ``budget_exceeded``; rate
   trip → ``asyncio.sleep`` for the suggested wait and try again.
2. ``agent_runtime.run_turn(...)``.
3. Accumulate token usage; re-check budget so one giant turn can still trip
   caps on the way back in.
4. Append the assistant turn to the transcript and the in-memory message
   list.
5. Parse the response into a :class:`~runner.parser.ParsedAction`.
6. Dispatch by action kind:

   * :class:`~runner.parser.WorkflowComplete` → ``status=succeeded``.
   * :class:`~runner.parser.WorkflowFailed` → ``status=failed`` with
     ``error_message`` set to the reason.
   * :class:`~runner.parser.ConfirmationRequest` → push to the queue,
     await the user's decision, then either inject a ``Confirmed``
     user message and continue or end the run with ``status=aborted``.
   * :class:`~runner.parser.ToolCallAction` → in execute mode, first run
     the pre-action gate. A ``RequireConfirmation`` gate result runs the
     same queue flow as a parser-level confirmation. If allowed, dispatch
     via :mod:`runner.dispatcher`, capture any screenshot the dispatcher
     produced, and feed the ``tool_result`` back as the next user message.
   * :class:`~runner.parser.UnknownAction` → nudge the model with a user
     message. The fourth consecutive ``UnknownAction`` aborts with
     ``error_message='agent_stuck'`` so a malfunctioning model cannot
     burn tokens forever.

Image mapping
-------------
The executor owns ``current_image_mapping`` and syncs it to the runtime
(``agent_runtime.set_image_mapping(...)``) after every new screenshot so the
computer tool's ``display_width_px``/``display_height_px`` stays accurate.
Before the very first screenshot the mapping is the default derived from
:data:`runner.coords.DryRunDisplayInfo` downscaled to longest-edge 1568,
matching what :class:`runner.claude_runtime.ClaudeRuntime` would pick.

Dry-run vs execute
------------------
The executor itself has no live/dry branching beyond (a) the prompt mode
passed to :func:`runner.execution_prompt.build_execution_prompt` and (b) the
mode passed to :func:`runner.pre_action_gate.apply_gate_to_tool_call`. The
caller decides which :class:`~runner.input_adapter.InputAdapter` and
:class:`~runner.screen_source.ScreenSource` back the run; the dry-run
combination is ``DryRunInputAdapter`` + ``TrajectoryScreenSource``.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, Final, cast
from uuid import UUID

from runner.agent_runtime import AgentResponse, AgentRuntime, Message
from runner.browser_dom_dispatcher import (
    BrowserDOMCallError,
    BrowserDOMDispatcher,
    BrowserDOMResult,
    substitute_parameters_in_string,
)
from runner.budget import (
    BudgetReason,
    BudgetStatusKind,
    BudgetTracker,
)
from runner.budget_config import crossed_warning_threshold
from runner.claude_runtime import estimate_cost_usd
from runner.confirmation import (
    DEFAULT_CONFIRMATION_TIMEOUT_SECONDS,
    ConfirmationDecision,
    ConfirmationQueue,
)
from runner.coords import DryRunDisplayInfo, ImageMapping
from runner.dispatcher import dispatch_tool_call
from runner.execution_hints import (
    CapabilityRegistry,
    HintDecision,
    Tier,
    default_capability_registry,
    iter_step_numbers,
    pick_hint,
)
from runner.execution_prompt import build_execution_prompt
from runner.input_adapter import InputAdapter
from runner.kill_switch import KillSwitch
from runner.mcp_client import (
    MCPCallDispatcher,
    MCPCallResult,
)
from runner.mcp_client import (
    substitute_parameters as substitute_mcp_parameters,
)
from runner.parser import (
    ConfirmationRequest as ParserConfirmationRequest,
)
from runner.parser import (
    ToolCallAction,
    UnknownAction,
    WorkflowComplete,
    WorkflowFailed,
    parse_agent_response,
)
from runner.pre_action_gate import (
    AXResolver,
    RequireConfirmation,
    apply_gate_to_tool_call,
)
from runner.run_writer import RunWriter
from runner.schema import RunMetadata, RunMode
from runner.screen_source import ScreenSource
from runner.skill_loader import LoadedSkill, substitute_parameters

logger = logging.getLogger(__name__)

UNKNOWN_ACTION_ABORT_THRESHOLD: Final[int] = 4
AGENT_STUCK_REASON: Final[str] = "agent_stuck"
USER_ABORT_REASON: Final[str] = "user_abort"
KILL_SWITCH_REASON: Final[str] = "kill_switch"
PER_RUN_COST_CAP_REASON: Final[str] = "per_run_cost_cap"
DEFAULT_TARGET_LONGEST_EDGE: Final[int] = 1568

CostWarningSink = Callable[[float, float], None]


def _default_image_mapping() -> ImageMapping:
    """Canonical :class:`ImageMapping` before any screenshot has landed.

    Matches what :class:`runner.claude_runtime.ClaudeRuntime` reports as the
    tool's ``display_width_px``/``display_height_px`` before
    ``set_image_mapping`` is called — keeps the two layers in lockstep so the
    very first tool call decodes coordinates against the same dims the model
    was shown.
    """

    orig_w = DryRunDisplayInfo.width_pixels
    orig_h = DryRunDisplayInfo.height_pixels
    longest = max(orig_w, orig_h)
    if longest <= DEFAULT_TARGET_LONGEST_EDGE:
        resized_w, resized_h = orig_w, orig_h
    else:
        ratio = DEFAULT_TARGET_LONGEST_EDGE / longest
        resized_w = round(orig_w * ratio)
        resized_h = round(orig_h * ratio)
    return ImageMapping(
        original_pixels=(orig_w, orig_h),
        resized_pixels=(resized_w, resized_h),
        scale_from_resized_to_points=(orig_w / resized_w)
        / DryRunDisplayInfo.scale_factor,
    )


def _find_step_text(skill: LoadedSkill, step_number: int) -> str:
    for step in skill.parsed_skill.steps:
        if step.number == step_number:
            return step.text
    return ""


def _now_utc() -> datetime:
    return datetime.now(UTC)


def _screenshot_filename(seq: int) -> str:
    return f"{seq:04d}.png"


class Executor:
    """Orchestrates a single run from first turn to final status.

    ``run()`` is the only public entry point and returns the final
    :class:`RunMetadata` on normal completion (including ``aborted``,
    ``failed``, ``budget_exceeded``). Exceptions from collaborators
    (adapter, screen source) propagate after the writer records
    ``status=failed``; the API layer catches them for a 500 response.
    """

    def __init__(
        self,
        *,
        loaded_skill: LoadedSkill,
        parameters: dict[str, str],
        mode: RunMode,
        agent_runtime: AgentRuntime,
        input_adapter: InputAdapter,
        screen_source: ScreenSource,
        ax_resolver: AXResolver,
        budget: BudgetTracker,
        writer: RunWriter,
        confirmation_queue: ConfirmationQueue,
        run_id: str,
        confirmation_timeout_seconds: float = DEFAULT_CONFIRMATION_TIMEOUT_SECONDS,
        kill_switch: KillSwitch | None = None,
        cost_warning_sink: CostWarningSink | None = None,
        capability_registry: CapabilityRegistry | None = None,
        mcp_dispatcher: MCPCallDispatcher | None = None,
        browser_dom_dispatcher: BrowserDOMDispatcher | None = None,
    ) -> None:
        self._loaded_skill = loaded_skill
        self._parameters = dict(parameters)
        self._mode: RunMode = mode
        self._agent_runtime = agent_runtime
        self._input_adapter = input_adapter
        self._screen_source = screen_source
        self._ax_resolver = ax_resolver
        self._budget = budget
        self._writer = writer
        self._confirmation_queue = confirmation_queue
        self._run_id = run_id
        self._confirmation_timeout_seconds = confirmation_timeout_seconds
        self._kill_switch = kill_switch
        self._kill_event: asyncio.Event | None = None
        self._cost_warning_sink = cost_warning_sink
        self._cost_warning_emitted = False
        self._capability_registry: CapabilityRegistry = (
            capability_registry
            if capability_registry is not None
            else default_capability_registry()
        )
        self._mcp_dispatcher: MCPCallDispatcher | None = mcp_dispatcher
        # Pre-execution outcome map populated by _pre_execute_mcp_steps,
        # keyed by step number → MCPCallResult. Read by the primer-message
        # builder. Empty when no dispatcher was supplied or no steps had
        # supported MCP hints.
        self._mcp_pre_executions: dict[int, MCPCallResult] = {}
        self._browser_dom_dispatcher: BrowserDOMDispatcher | None = (
            browser_dom_dispatcher
        )
        # Same shape as ``_mcp_pre_executions`` — populated by the
        # browser_dom pre-execution path so the primer-builder can list
        # which steps the dispatcher already drove.
        self._browser_dom_pre_executions: dict[int, BrowserDOMResult] = {}

        self._event_seq = 0
        self._screenshot_seq = 0
        self._current_image_mapping: ImageMapping = _default_image_mapping()
        self._current_step = 0
        self._max_step_seen = 0
        self._unknown_action_count = 0
        self._input_tokens_total = 0
        self._output_tokens_total = 0
        self._cost_usd_total = 0.0
        self._confirmation_count = 0
        self._destructive_steps_executed: list[int] = []
        self._started_at: datetime | None = None

    async def run(self) -> RunMetadata:
        """Execute the run to completion and return the final metadata."""

        if self._kill_switch is not None:
            self._kill_event = self._kill_switch.register(self._run_id)
            if self._kill_event.is_set():
                return await self._run_killed_before_start()

        try:
            # Both dispatchers are async-context-managed and live for
            # the run's duration. AsyncExitStack lets us enter whichever
            # ones are present without nested-if branching.
            async with contextlib.AsyncExitStack() as stack:
                if self._mcp_dispatcher is not None:
                    await stack.enter_async_context(self._mcp_dispatcher)
                if self._browser_dom_dispatcher is not None:
                    await stack.enter_async_context(self._browser_dom_dispatcher)
                return await self._run_body()
        finally:
            if self._kill_switch is not None:
                self._kill_switch.unregister(self._run_id)
            self._writer.close()

    async def _run_killed_before_start(self) -> RunMetadata:
        """Finalize without ever hitting the LLM when kill fired pre-run."""

        self._started_at = _now_utc()
        metadata = RunMetadata(
            run_id=UUID(self._run_id),
            skill_slug=str(self._loaded_skill.meta["slug"]),
            started_at=self._started_at,
            ended_at=None,
            status="running",
            mode=self._mode,
            parameters=self._parameters if self._parameters else None,
        )
        self._writer.write_metadata(metadata)
        reason = self._kill_reason_or_default()
        decision = ConfirmationDecision(action="abort", reason=reason)
        return self._finalize_aborted(metadata, decision)

    async def _run_body(self) -> RunMetadata:
        resolved_skill = substitute_parameters(
            self._loaded_skill, self._parameters
        )
        prompt_mode: Final = "live" if self._mode == "execute" else "dry_run"
        system_prompt = build_execution_prompt(resolved_skill, prompt_mode)

        self._started_at = _now_utc()
        metadata = RunMetadata(
            run_id=UUID(self._run_id),
            skill_slug=str(self._loaded_skill.meta["slug"]),
            started_at=self._started_at,
            ended_at=None,
            status="running",
            mode=self._mode,
            parameters=self._parameters if self._parameters else None,
            input_tokens_total=0,
            output_tokens_total=0,
            total_cost_usd=0.0,
            confirmation_count=0,
            destructive_actions_executed=[],
        )
        self._writer.write_metadata(metadata)
        self._agent_runtime.set_image_mapping(self._current_image_mapping)
        self._append_event(
            "step_start",
            f"run_started slug={metadata.skill_slug}",
            step_number=None,
        )
        self._record_tier_decisions(resolved_skill)

        # Steps 3b/3c: dispatch MCP-tier hints before the agent loop
        # kicks off. Non-destructive steps go through immediately;
        # destructive ones queue a confirmation request and dispatch
        # only on confirm. The agent sees a primer user-message naming
        # everything that was already done so it can skip those steps.
        # User abort during the destructive confirmation aborts the
        # whole run (matches the prompt-tag flow's contract — there's
        # no "abort just this step" path because that would leave the
        # workflow inconsistent).
        abort_decision = await self._pre_execute_mcp_steps(
            resolved_skill, metadata
        )
        if abort_decision is not None:
            return self._finalize_aborted(metadata, abort_decision)

        # Same flow as MCP, but for tier=browser_dom. A step's resolved
        # tier is exclusive (pick_hint chooses one), so iterating once
        # per tier never double-dispatches the same step. We do MCP
        # first because it's the higher-priority tier; either tier's
        # destructive abort short-circuits the whole run.
        abort_decision = await self._pre_execute_browser_dom_steps(
            resolved_skill, metadata
        )
        if abort_decision is not None:
            return self._finalize_aborted(metadata, abort_decision)

        messages: list[Message] = []
        primer = self._build_mcp_primer_message(resolved_skill)
        if primer is not None:
            messages.append(primer)
        browser_dom_primer = self._build_browser_dom_primer_message(
            resolved_skill
        )
        if browser_dom_primer is not None:
            messages.append(browser_dom_primer)
        try:
            return await self._run_loop(
                resolved_skill, system_prompt, messages, metadata
            )
        except Exception as exc:
            self._writer.update_status(
                metadata,
                status="failed",
                ended_at=_now_utc(),
                error_message=str(exc),
                input_tokens_total=self._input_tokens_total,
                output_tokens_total=self._output_tokens_total,
                total_cost_usd=self._cost_usd_total,
                confirmation_count=self._confirmation_count,
                destructive_actions_executed=list(
                    self._destructive_steps_executed
                ),
                final_step_reached=self._max_step_seen or None,
                abort_reason="exception",
            )
            raise

    async def _run_loop(
        self,
        resolved_skill: LoadedSkill,
        system_prompt: str,
        messages: list[Message],
        metadata: RunMetadata,
    ) -> RunMetadata:
        while True:
            if self._kill_triggered():
                return self._finalize_aborted_by_kill(metadata)

            rate_limit_result = await self._check_and_wait_rate_limit()
            if rate_limit_result is not None:
                return self._finalize_budget_exceeded(
                    metadata, rate_limit_result
                )

            turn_result = await self._run_turn_cancellable(
                system_prompt, messages
            )
            if turn_result is None:
                return self._finalize_aborted_by_kill(metadata)
            response = turn_result
            self._budget.record_turn(
                response.input_tokens, response.output_tokens
            )
            turn_cost = estimate_cost_usd(
                response.input_tokens, response.output_tokens
            )
            self._input_tokens_total += response.input_tokens
            self._output_tokens_total += response.output_tokens
            prev_cost = self._cost_usd_total
            self._cost_usd_total += turn_cost
            self._budget.record_cost(turn_cost)
            self._maybe_emit_cost_warning(prev_cost, self._cost_usd_total)
            self._writer.append_transcript(
                turn=response.turn_number,
                role="assistant",
                content=response.content_blocks,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
            )
            messages.append(
                {"role": "assistant", "content": response.content_blocks}
            )

            post_status = self._budget.check()
            if post_status.kind == BudgetStatusKind.BUDGET_EXCEEDED:
                return self._finalize_budget_exceeded(
                    metadata, post_status.reason
                )

            parsed = parse_agent_response(response)

            if isinstance(parsed, WorkflowComplete):
                return self._finalize_succeeded(metadata, resolved_skill)

            if isinstance(parsed, WorkflowFailed):
                return self._finalize_failed(metadata, parsed.reason)

            if isinstance(parsed, ParserConfirmationRequest):
                self._unknown_action_count = 0
                decision = await self._handle_prompt_confirmation(
                    resolved_skill, parsed.step_number
                )
                if decision.action == "abort":
                    return self._finalize_aborted(metadata, decision)
                messages.append(self._confirmation_user_message(parsed.step_number))
                continue

            if isinstance(parsed, UnknownAction):
                self._unknown_action_count += 1
                if self._unknown_action_count >= UNKNOWN_ACTION_ABORT_THRESHOLD:
                    return self._finalize_agent_stuck(metadata)
                messages.append(self._nudge_user_message())
                continue

            if isinstance(parsed, ToolCallAction):
                self._unknown_action_count = 0
                gate_outcome = await self._apply_pre_action_gate(parsed)
                if isinstance(gate_outcome, _GateAbort):
                    return self._finalize_aborted(
                        metadata, gate_outcome.decision
                    )

                self._budget.record_action()
                action_status = self._budget.check()
                if action_status.kind == BudgetStatusKind.BUDGET_EXCEEDED:
                    return self._finalize_budget_exceeded(
                        metadata, action_status.reason
                    )
                if (
                    action_status.kind == BudgetStatusKind.RATE_LIMITED
                    and action_status.wait_seconds is not None
                ):
                    await asyncio.sleep(action_status.wait_seconds)

                if self._kill_triggered():
                    return self._finalize_aborted_by_kill(metadata)

                tool_result = dispatch_tool_call(
                    parsed,
                    self._input_adapter,
                    self._screen_source,
                    self._current_image_mapping,
                )
                action_name = parsed.tool_input.get("action", "<unknown>")
                screenshot_seq_for_event: int | None = None
                if tool_result.new_image_mapping is not None:
                    self._current_image_mapping = tool_result.new_image_mapping
                    self._agent_runtime.set_image_mapping(
                        tool_result.new_image_mapping
                    )
                    for block in tool_result.content_blocks:
                        if (
                            isinstance(block, dict)
                            and block.get("type") == "image"
                        ):
                            screenshot_seq_for_event = (
                                self._persist_screenshot(block)
                            )
                            break

                self._append_event(
                    "tool_call",
                    f"action={action_name} is_error={tool_result.is_error}",
                    step_number=self._current_step or None,
                    screenshot_seq=screenshot_seq_for_event,
                )

                messages.append(
                    cast(
                        Message,
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": parsed.tool_use_id,
                                    "content": tool_result.content_blocks,
                                    "is_error": tool_result.is_error,
                                }
                            ],
                        },
                    )
                )
                continue

    def _kill_triggered(self) -> bool:
        """True if a kill has been signaled for this run."""

        return self._kill_event is not None and self._kill_event.is_set()

    def _kill_reason_or_default(self) -> str:
        """Return the kill reason from the switch, falling back to a constant."""

        if self._kill_switch is not None:
            reason = self._kill_switch.reason(self._run_id)
            if reason:
                return reason
        return KILL_SWITCH_REASON

    async def _run_turn_cancellable(
        self, system_prompt: str, messages: list[Message]
    ) -> AgentResponse | None:
        """Run one turn, racing the LLM call against the kill event.

        Returns the :class:`AgentResponse` on completion, or ``None`` if the
        kill event fires first. On kill we cancel the turn task so the
        in-flight ``httpx`` request is aborted at the next await point, which
        is what gives the 2-second abort guarantee from
        ``CLAUDE.md`` its teeth.
        """

        turn_task = asyncio.create_task(
            self._agent_runtime.run_turn(system_prompt, messages)
        )
        if self._kill_event is None:
            return await turn_task
        kill_task = asyncio.create_task(self._kill_event.wait())
        try:
            done, _pending = await asyncio.wait(
                [turn_task, kill_task],
                return_when=asyncio.FIRST_COMPLETED,
            )
            if turn_task in done:
                return turn_task.result()
            turn_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await turn_task
            return None
        finally:
            if not kill_task.done():
                kill_task.cancel()

    async def _await_decision_or_kill(self) -> ConfirmationDecision:
        """Await a user decision; race against the kill event.

        On kill, synthesize an abort decision. We also inject the synthetic
        decision into the queue so the underlying ``await_decision`` task can
        complete and clean up the pending state.
        """

        decision_task = asyncio.create_task(
            self._confirmation_queue.await_decision(
                self._run_id, self._confirmation_timeout_seconds
            )
        )
        if self._kill_event is None:
            return await decision_task
        kill_task = asyncio.create_task(self._kill_event.wait())
        try:
            done, _pending = await asyncio.wait(
                [decision_task, kill_task],
                return_when=asyncio.FIRST_COMPLETED,
            )
            if decision_task in done:
                return decision_task.result()
            reason = self._kill_reason_or_default()
            self._confirmation_queue.submit_decision(
                self._run_id,
                ConfirmationDecision(action="abort", reason=reason),
            )
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await decision_task
            return ConfirmationDecision(action="abort", reason=reason)
        finally:
            if not kill_task.done():
                kill_task.cancel()

    def _finalize_aborted_by_kill(self, metadata: RunMetadata) -> RunMetadata:
        """Finalize the run as ``aborted`` due to a kill-switch signal."""

        reason = self._kill_reason_or_default()
        return self._finalize_aborted(
            metadata, ConfirmationDecision(action="abort", reason=reason)
        )

    async def _check_and_wait_rate_limit(self) -> BudgetReason | None:
        """Pre-turn budget check. Returns a reason if hard-tripped, else None.

        A soft rate-limit triggers an ``asyncio.sleep`` and the loop retries
        until either the limit clears or a hard cap trips.
        """

        while True:
            status = self._budget.check()
            if status.kind == BudgetStatusKind.BUDGET_EXCEEDED:
                return status.reason
            if (
                status.kind == BudgetStatusKind.RATE_LIMITED
                and status.wait_seconds is not None
            ):
                await asyncio.sleep(status.wait_seconds)
                continue
            return None

    async def _handle_prompt_confirmation(
        self, resolved_skill: LoadedSkill, step_number: int
    ) -> ConfirmationDecision:
        """Queue a parser-level ``<needs_confirmation>`` and await the user."""

        self._current_step = step_number
        self._max_step_seen = max(self._max_step_seen, step_number)
        self._confirmation_count += 1
        step_text = _find_step_text(resolved_skill, step_number)
        self._confirmation_queue.push_request(
            run_id=self._run_id,
            step_number=step_number,
            step_text=step_text,
            screenshot_ref=self._last_screenshot_ref(),
            destructive_reason="prompt_tag",
        )
        self._append_event(
            "confirmation_requested",
            f"step={step_number} reason=prompt_tag",
            step_number=step_number,
        )
        self._writer.update_status(
            self._snapshot_metadata("awaiting_confirmation", ended_at=None),
            status="awaiting_confirmation",
            confirmation_count=self._confirmation_count,
        )
        decision = await self._await_decision_or_kill()
        if decision.action == "confirm":
            self._destructive_steps_executed.append(step_number)
            self._append_event(
                "confirmed",
                f"step={step_number}",
                step_number=step_number,
            )
            self._writer.update_status(
                self._snapshot_metadata("running", ended_at=None),
                status="running",
            )
        return decision

    async def _apply_pre_action_gate(
        self, action: ToolCallAction
    ) -> _GateAbort | _GateAllow:
        """Run the harness-layer gate on a ``ToolCallAction``.

        Dry-run short-circuits to allow (the gate itself enforces this too,
        but we save the resolver call). On ``RequireConfirmation`` we run the
        same queue flow as a parser-level confirmation; an abort decision
        returns ``_GateAbort`` so the caller finalizes the run.
        """

        decision = apply_gate_to_tool_call(
            action,
            self._current_image_mapping,
            self._ax_resolver,
            self._mode,
        )
        if not isinstance(decision, RequireConfirmation):
            return _GateAllow()

        step_number = self._current_step or 0
        self._confirmation_count += 1
        self._confirmation_queue.push_request(
            run_id=self._run_id,
            step_number=step_number,
            step_text=decision.label,
            screenshot_ref=self._last_screenshot_ref(),
            destructive_reason=decision.reason,
        )
        self._append_event(
            "confirmation_requested",
            f"step={step_number} reason=pre_action_gate label={decision.label!r}",
            step_number=step_number or None,
        )
        self._writer.update_status(
            self._snapshot_metadata("awaiting_confirmation", ended_at=None),
            status="awaiting_confirmation",
            confirmation_count=self._confirmation_count,
        )
        user_decision = await self._await_decision_or_kill()
        if user_decision.action == "abort":
            return _GateAbort(decision=user_decision)
        if step_number:
            self._destructive_steps_executed.append(step_number)
        self._append_event(
            "confirmed",
            f"step={step_number} via pre_action_gate",
            step_number=step_number or None,
        )
        self._writer.update_status(
            self._snapshot_metadata("running", ended_at=None),
            status="running",
        )
        return _GateAllow()

    def _persist_screenshot(self, image_block: dict[str, Any]) -> int | None:
        source = image_block.get("source", {})
        data = source.get("data") if isinstance(source, dict) else None
        if not isinstance(data, str):
            return None
        try:
            png_bytes = base64.standard_b64decode(data)
        except (ValueError, TypeError):
            return None
        if not png_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
            return None
        seq = self._screenshot_seq
        self._writer.write_screenshot(seq, png_bytes)
        self._screenshot_seq += 1
        return seq

    def _last_screenshot_ref(self) -> str | None:
        if self._screenshot_seq == 0:
            return None
        return _screenshot_filename(self._screenshot_seq - 1)

    def _append_event(
        self,
        event_type: str,
        message: str,
        *,
        step_number: int | None = None,
        screenshot_seq: int | None = None,
    ) -> None:
        self._writer.append_event(
            seq=self._event_seq,
            event_type=event_type,
            message=message,
            step_number=step_number,
            screenshot_seq=screenshot_seq,
        )
        self._event_seq += 1

    def _record_tier_decisions(self, resolved_skill: LoadedSkill) -> None:
        """Walk every step with hints and log which tier the runner picked.

        Step 2 of the tiered-execution rollout is logging-only: we record
        the decision into the run's event stream so downstream tools (and
        humans) can see what the runner *would* dispatch once Steps 3
        (real MCP) and 4 (Playwright DOM) ship. The actual action this
        run takes still flows through the existing computer-use path.

        Steps without ``meta.steps`` entries are silent — the runner just
        falls through to the SKILL.md prose for them, same as before.
        """
        meta = resolved_skill.meta
        for step_number in iter_step_numbers(meta):
            decision = pick_hint(
                step_number=step_number,
                meta=meta,
                registry=self._capability_registry,
            )
            self._append_event(
                "tier_selected",
                self._format_tier_decision(decision),
                step_number=decision.step_number,
            )

    async def _pre_execute_mcp_steps(
        self, resolved_skill: LoadedSkill, metadata: RunMetadata
    ) -> ConfirmationDecision | None:
        """Run real MCP calls for ``tier=mcp`` steps before the agent loop.

        Walks every step number declared in ``meta.steps`` in input
        order. For each step whose resolved tier is ``mcp``:

        * Non-destructive: dispatch immediately, stash result for the
          primer builder. Failures are logged but never abort the run.
        * Destructive: queue a confirmation request and await the
          user's decision. On confirm, dispatch and increment the run's
          destructive-action counter. On abort or kill, return the
          decision so the caller can finalize the run as ``aborted``
          before the agent loop ever starts.

        Returns ``None`` when the run should proceed (every destructive
        step was confirmed or none existed) or a :class:`ConfirmationDecision`
        when the user aborted (or kill fired) and the caller must
        finalize. Mirrors the prompt-tag flow's "abort = end the run"
        contract — there's no "abort just this step" path because that
        would leave the workflow in an inconsistent state.
        """
        dispatcher = self._mcp_dispatcher
        if dispatcher is None:
            return None
        meta = resolved_skill.meta
        destructive = set(meta.get("destructive_steps") or [])

        for step_number in iter_step_numbers(meta):
            decision = pick_hint(
                step_number=step_number,
                meta=meta,
                registry=self._capability_registry,
            )
            if decision.chosen is None or decision.chosen_tier != Tier.MCP:
                continue

            server = decision.chosen.get("mcp_server")
            function = decision.chosen.get("function")
            raw_args = decision.chosen.get("arguments") or {}
            if not isinstance(server, str) or not isinstance(function, str):
                self._append_event(
                    "mcp_skipped",
                    f"step={step_number} malformed mcp hint "
                    f"(server={server!r} function={function!r})",
                    step_number=step_number,
                )
                continue

            try:
                arguments = substitute_mcp_parameters(raw_args, self._parameters)
            except KeyError as exc:
                self._append_event(
                    "mcp_skipped",
                    f"step={step_number} missing run parameter {{{exc.args[0]}}} "
                    f"for {server}.{function}",
                    step_number=step_number,
                )
                continue

            is_destructive = step_number in destructive
            if is_destructive:
                user_decision = await self._confirm_mcp_destructive(
                    metadata=metadata,
                    step_number=step_number,
                    server=server,
                    function=function,
                    arguments=arguments,
                    skill=resolved_skill,
                )
                if user_decision.action != "confirm":
                    self._append_event(
                        "mcp_aborted",
                        f"step={step_number} {server}.{function} "
                        f"aborted_by_user reason="
                        f"{user_decision.reason or USER_ABORT_REASON}",
                        step_number=step_number,
                    )
                    return user_decision

            await self._dispatch_one_mcp(
                step_number=step_number,
                server=server,
                function=function,
                arguments=arguments,
                destructive=is_destructive,
            )
        return None

    async def _confirm_mcp_destructive(
        self,
        *,
        metadata: RunMetadata,
        step_number: int,
        server: str,
        function: str,
        arguments: dict[str, Any],
        skill: LoadedSkill,
    ) -> ConfirmationDecision:
        """Push a confirmation request for a destructive MCP-tier step.

        Mirrors the prompt-tag confirmation flow in
        :meth:`_request_confirmation_via_prompt` but identifies the
        request as ``mcp_destructive`` so the UI can render the
        prospective tool call (server.function + args) instead of the
        SKILL.md prose. Increments ``_confirmation_count``, switches
        the run to ``awaiting_confirmation`` while we wait, and
        flips back to ``running`` on confirm.
        """
        self._current_step = step_number
        self._max_step_seen = max(self._max_step_seen, step_number)
        self._confirmation_count += 1
        step_text = _find_step_text(skill, step_number) or (
            f"MCP {server}.{function}"
        )
        self._confirmation_queue.push_request(
            run_id=self._run_id,
            step_number=step_number,
            step_text=step_text,
            screenshot_ref=self._last_screenshot_ref(),
            destructive_reason="mcp_destructive",
        )
        # Render a compact, log-friendly view of the prospective call
        # so the audit trail records the exact server/function/args
        # the user was asked to confirm.
        arg_preview = ", ".join(f"{k}={v!r}" for k, v in arguments.items())
        if len(arg_preview) > 160:
            arg_preview = arg_preview[:157] + "..."
        self._append_event(
            "confirmation_requested",
            f"step={step_number} reason=mcp_destructive "
            f"call={server}.{function}({arg_preview})",
            step_number=step_number,
        )
        self._writer.update_status(
            self._snapshot_metadata("awaiting_confirmation", ended_at=None),
            status="awaiting_confirmation",
            confirmation_count=self._confirmation_count,
        )
        decision = await self._await_decision_or_kill()
        if decision.action == "confirm":
            self._destructive_steps_executed.append(step_number)
            self._append_event(
                "confirmed",
                f"step={step_number} reason=mcp_destructive",
                step_number=step_number,
            )
            self._writer.update_status(
                self._snapshot_metadata("running", ended_at=None),
                status="running",
            )
        return decision

    async def _dispatch_one_mcp(
        self,
        *,
        step_number: int,
        server: str,
        function: str,
        arguments: dict[str, Any],
        destructive: bool,
    ) -> None:
        """Issue a single MCP ``tools/call`` and record the result.

        Splits out so both the destructive-after-confirm path and the
        non-destructive path use identical logging + result-stashing
        semantics. ``destructive=True`` adds a marker to the event
        message so the run log makes the distinction obvious.
        """
        dispatcher = self._mcp_dispatcher
        assert dispatcher is not None  # guarded by caller

        result = await dispatcher.call(
            server=server, function=function, arguments=arguments
        )
        marker = " destructive" if destructive else ""
        if result.ok:
            self._mcp_pre_executions[step_number] = result
            preview = (result.content_text or "").strip().replace("\n", " ")
            if len(preview) > 160:
                preview = preview[:157] + "..."
            self._append_event(
                "mcp_dispatched",
                f"step={step_number}{marker} {server}.{function} ok"
                + (f" → {preview}" if preview else ""),
                step_number=step_number,
            )
        else:
            self._append_event(
                "mcp_failed",
                f"step={step_number}{marker} {server}.{function} error: "
                f"{result.error}",
                step_number=step_number,
            )

    def _build_mcp_primer_message(
        self, resolved_skill: LoadedSkill
    ) -> Message | None:
        """Build a primer user-message naming MCP-pre-executed steps.

        Returns ``None`` when no steps were pre-executed. Otherwise the
        agent's first turn sees a message like:

            Steps 2, 3 have been pre-executed for you via MCP. Their
            results are below — please skip those steps and continue
            from step 4 in the SKILL.md.

            Step 2 (gmail.search_threads):
            <result text>

        This is intentionally directive: the SKILL.md prose still
        contains every step, but the primer tells the agent which
        ones are done and what came back. The agent shouldn't redo
        them.
        """
        if not self._mcp_pre_executions:
            return None
        executed = sorted(self._mcp_pre_executions)
        next_step = max(executed) + 1
        total_steps = int(resolved_skill.meta.get("step_count") or 0)
        lines: list[str] = [
            f"Steps {executed} have been pre-executed for you via MCP. "
            f"Their results are below — please skip those steps and "
            f"continue from step {next_step} in the SKILL.md."
        ]
        if total_steps and next_step > total_steps:
            lines = [
                f"Every step in the workflow ({executed}) was pre-executed "
                "for you via MCP. Acknowledge completion with the workflow-"
                "complete signal."
            ]
        for step_number in executed:
            result = self._mcp_pre_executions[step_number]
            content = (result.content_text or "(no content returned)").strip()
            lines.append("")
            lines.append(
                f"Step {step_number} ({result.server}.{result.function}):"
            )
            # Truncate per-step content so the primer doesn't blow up
            # the context for huge tool outputs. Full content stays in
            # ``self._mcp_pre_executions`` for downstream tools that
            # want it.
            if len(content) > 1200:
                content = content[:1200] + "\n...(truncated)"
            lines.append(content)
        text = "\n".join(lines)
        return {
            "role": "user",
            "content": [{"type": "text", "text": text}],
        }

    async def _pre_execute_browser_dom_steps(
        self, resolved_skill: LoadedSkill, metadata: RunMetadata
    ) -> ConfirmationDecision | None:
        """Run real Playwright DOM actions for ``tier=browser_dom`` steps.

        Mirror of :meth:`_pre_execute_mcp_steps` for the browser_dom
        tier. A step's resolved tier is exclusive — :func:`pick_hint`
        picks one — so this method only acts on steps the resolver
        already routed to browser_dom; everything else is silently
        skipped.

        For each browser_dom step:

        * Substitute ``{param}`` references in ``url_pattern`` /
          ``selector`` / ``value``. A missing parameter logs a
          ``browser_dom_skipped`` event and falls through to the
          agent loop (computer_use will pick up the slack).
        * Non-destructive: dispatch immediately, stash the result for
          the primer builder.
        * Destructive: queue a confirmation request and await the
          user's decision. On confirm, dispatch and increment the
          destructive-action counter. On abort/kill, return the
          decision so the caller finalizes the run as ``aborted``.
        """
        dispatcher = self._browser_dom_dispatcher
        if dispatcher is None:
            return None
        meta = resolved_skill.meta
        destructive = set(meta.get("destructive_steps") or [])

        for step_number in iter_step_numbers(meta):
            decision = pick_hint(
                step_number=step_number,
                meta=meta,
                registry=self._capability_registry,
            )
            if (
                decision.chosen is None
                or decision.chosen_tier != Tier.BROWSER_DOM
            ):
                continue

            action = decision.chosen.get("action")
            if not isinstance(action, str):
                self._append_event(
                    "browser_dom_skipped",
                    f"step={step_number} malformed browser_dom hint "
                    f"(missing or non-string `action`)",
                    step_number=step_number,
                )
                continue

            try:
                resolved_hint = self._resolve_browser_dom_hint(
                    decision.chosen
                )
            except KeyError as exc:
                self._append_event(
                    "browser_dom_skipped",
                    f"step={step_number} missing run parameter "
                    f"{{{exc.args[0]}}} for browser_dom {action}",
                    step_number=step_number,
                )
                continue

            is_destructive = step_number in destructive
            if is_destructive:
                user_decision = await self._confirm_browser_dom_destructive(
                    metadata=metadata,
                    step_number=step_number,
                    resolved_hint=resolved_hint,
                    skill=resolved_skill,
                )
                if user_decision.action != "confirm":
                    self._append_event(
                        "browser_dom_aborted",
                        f"step={step_number} {action} "
                        f"aborted_by_user reason="
                        f"{user_decision.reason or USER_ABORT_REASON}",
                        step_number=step_number,
                    )
                    return user_decision

            await self._dispatch_one_browser_dom(
                step_number=step_number,
                resolved_hint=resolved_hint,
                destructive=is_destructive,
            )
        return None

    def _resolve_browser_dom_hint(
        self, hint: dict[str, Any]
    ) -> dict[str, Any]:
        """Substitute ``{param}`` references and return a resolved hint dict.

        The dispatcher does its own substitution at dispatch time, but
        the executor needs the substituted view *first* so the
        destructive-confirmation request can name the literal target
        the user is approving. Substituting once here and passing the
        resolved hint back to the dispatcher with ``parameters={}``
        means we don't substitute twice — the dispatcher's pass becomes
        a no-op on already-resolved strings.
        """
        out: dict[str, Any] = {"tier": "browser_dom"}
        for key in ("action", "url_pattern", "selector", "value"):
            raw = hint.get(key)
            if isinstance(raw, str):
                out[key] = substitute_parameters_in_string(raw, self._parameters)
            elif raw is not None:
                out[key] = raw
        return out

    async def _confirm_browser_dom_destructive(
        self,
        *,
        metadata: RunMetadata,
        step_number: int,
        resolved_hint: dict[str, Any],
        skill: LoadedSkill,
    ) -> ConfirmationDecision:
        """Push a confirmation request for a destructive browser_dom step.

        Mirrors :meth:`_confirm_mcp_destructive` — same state changes
        (``awaiting_confirmation`` → ``running``), same counter bumps,
        same kill-aware decision wait — but the request reason is
        ``browser_dom_destructive`` and the audit-trail event names
        the prospective DOM action instead of an MCP function call.
        """
        self._current_step = step_number
        self._max_step_seen = max(self._max_step_seen, step_number)
        self._confirmation_count += 1
        action = str(resolved_hint.get("action", "<unknown>"))
        selector = resolved_hint.get("selector")
        step_text = _find_step_text(skill, step_number) or (
            f"browser_dom {action}"
            + (f" on {selector}" if selector else "")
        )
        self._confirmation_queue.push_request(
            run_id=self._run_id,
            step_number=step_number,
            step_text=step_text,
            screenshot_ref=self._last_screenshot_ref(),
            destructive_reason="browser_dom_destructive",
        )
        # Audit-trail line names every relevant resolved field; long
        # values are truncated so log lines stay grep-friendly.
        action_preview = self._format_browser_dom_action(resolved_hint)
        self._append_event(
            "confirmation_requested",
            f"step={step_number} reason=browser_dom_destructive "
            f"{action_preview}",
            step_number=step_number,
        )
        self._writer.update_status(
            self._snapshot_metadata("awaiting_confirmation", ended_at=None),
            status="awaiting_confirmation",
            confirmation_count=self._confirmation_count,
        )
        decision = await self._await_decision_or_kill()
        if decision.action == "confirm":
            self._destructive_steps_executed.append(step_number)
            self._append_event(
                "confirmed",
                f"step={step_number} reason=browser_dom_destructive",
                step_number=step_number,
            )
            self._writer.update_status(
                self._snapshot_metadata("running", ended_at=None),
                status="running",
            )
        return decision

    async def _dispatch_one_browser_dom(
        self,
        *,
        step_number: int,
        resolved_hint: dict[str, Any],
        destructive: bool,
    ) -> None:
        """Issue a single browser_dom action and record the outcome.

        Mirror of :meth:`_dispatch_one_mcp`. Catches
        :class:`BrowserDOMCallError` (hint-shape error — unsupported
        action, missing required field) and logs it as a skip; runtime
        action failures land as ``browser_dom_failed`` with the error
        message. Successful dispatches stash the result so the primer
        builder can name the step in the agent's first user message.
        """
        dispatcher = self._browser_dom_dispatcher
        assert dispatcher is not None  # guarded by caller

        action = str(resolved_hint.get("action", "<unknown>"))
        marker = " destructive" if destructive else ""
        try:
            # Pre-substituted hint + empty params: dispatcher's own
            # substitution becomes a no-op.
            result = await dispatcher.dispatch(resolved_hint, parameters={})
        except BrowserDOMCallError as exc:
            self._append_event(
                "browser_dom_skipped",
                f"step={step_number}{marker} {action} "
                f"unissuable: {exc.detail}",
                step_number=step_number,
            )
            return

        action_preview = self._format_browser_dom_action(resolved_hint)
        if result.ok:
            self._browser_dom_pre_executions[step_number] = result
            content_preview = (result.content_text or "").strip()
            if len(content_preview) > 160:
                content_preview = content_preview[:157] + "..."
            self._append_event(
                "browser_dom_dispatched",
                f"step={step_number}{marker} {action_preview} ok"
                + (f" → {content_preview}" if content_preview else ""),
                step_number=step_number,
            )
        else:
            self._append_event(
                "browser_dom_failed",
                f"step={step_number}{marker} {action_preview} "
                f"error: {result.error}",
                step_number=step_number,
            )

    @staticmethod
    def _format_browser_dom_action(hint: dict[str, Any]) -> str:
        """Compact one-line render of a resolved browser_dom hint."""
        action = hint.get("action", "<missing>")
        bits: list[str] = [str(action)]
        if hint.get("selector"):
            bits.append(f"selector={hint['selector']!r}")
        if hint.get("url_pattern"):
            bits.append(f"url={hint['url_pattern']!r}")
        if hint.get("value"):
            value_repr = repr(hint["value"])
            if len(value_repr) > 80:
                value_repr = value_repr[:77] + "..."
            bits.append(f"value={value_repr}")
        return " ".join(bits)

    def _build_browser_dom_primer_message(
        self, resolved_skill: LoadedSkill
    ) -> Message | None:
        """Build a primer user-message naming browser_dom-pre-executed steps.

        Mirror of :meth:`_build_mcp_primer_message`. Returns ``None``
        when no steps were pre-executed at the browser_dom tier.
        Otherwise the agent's first turn sees a directive message
        listing which steps the dispatcher already drove and what the
        action was, so the agent skips those steps.
        """
        if not self._browser_dom_pre_executions:
            return None
        executed = sorted(self._browser_dom_pre_executions)
        next_step = max(executed) + 1
        total_steps = int(resolved_skill.meta.get("step_count") or 0)
        lines: list[str] = [
            f"Steps {executed} have been pre-executed for you via "
            f"browser DOM actions. Their results are below — please "
            f"skip those steps and continue from step {next_step} in "
            "the SKILL.md."
        ]
        if total_steps and next_step > total_steps:
            lines = [
                f"Every step in the workflow ({executed}) was pre-executed "
                "for you via browser DOM actions. Acknowledge completion "
                "with the workflow-complete signal."
            ]
        for step_number in executed:
            result = self._browser_dom_pre_executions[step_number]
            content = (result.content_text or "(no content returned)").strip()
            lines.append("")
            target = result.selector or result.url or "<unspecified>"
            lines.append(
                f"Step {step_number} ({result.action} {target}):"
            )
            if len(content) > 1200:
                content = content[:1200] + "\n...(truncated)"
            lines.append(content)
        text = "\n".join(lines)
        return {
            "role": "user",
            "content": [{"type": "text", "text": text}],
        }

    @staticmethod
    def _format_tier_decision(decision: HintDecision) -> str:
        """Render a HintDecision as a single-line event message."""
        intent = decision.intent or "(no intent)"
        if decision.chosen is None:
            return (
                f"step={decision.step_number} intent={intent} "
                "tier=<none> - no supported execution hint and computer_use disabled"
            )
        tier = decision.chosen_tier.value if decision.chosen_tier else "<unknown>"
        detail = ""
        if tier == "mcp":
            detail = (
                f" mcp={decision.chosen.get('mcp_server')}."
                f"{decision.chosen.get('function')}"
            )
        elif tier == "browser_dom":
            detail = f" selector={decision.chosen.get('selector')!r}"
        fell_back = " (fell_back=true)" if decision.fell_back else ""
        synthetic = " synthetic" if decision.chosen.get("synthetic") else ""
        return (
            f"step={decision.step_number} intent={intent} "
            f"tier={tier}{synthetic}{detail}{fell_back}"
        )

    def _snapshot_metadata(
        self, status: str, *, ended_at: datetime | None
    ) -> RunMetadata:
        """Build a ``RunMetadata`` reflecting the executor's live counters.

        Used as the ``RunWriter.update_status`` base — the writer re-runs
        pydantic validation so a bogus status enum still trips loudly.
        """

        return RunMetadata(
            run_id=UUID(self._run_id),
            skill_slug=str(self._loaded_skill.meta["slug"]),
            started_at=self._started_at or _now_utc(),
            ended_at=ended_at,
            status="running",
            mode=self._mode,
            parameters=self._parameters if self._parameters else None,
            input_tokens_total=self._input_tokens_total,
            output_tokens_total=self._output_tokens_total,
            total_cost_usd=self._cost_usd_total,
            confirmation_count=self._confirmation_count,
            destructive_actions_executed=list(
                self._destructive_steps_executed
            ),
            final_step_reached=self._max_step_seen or None,
        )

    def _finalize_succeeded(
        self, metadata: RunMetadata, resolved_skill: LoadedSkill
    ) -> RunMetadata:
        total_steps = len(resolved_skill.parsed_skill.steps)
        final_step = max(self._max_step_seen, total_steps)
        self._append_event(
            "step_complete",
            f"run_succeeded total_steps={total_steps}",
            step_number=final_step,
        )
        return self._writer.update_status(
            metadata,
            status="succeeded",
            ended_at=_now_utc(),
            input_tokens_total=self._input_tokens_total,
            output_tokens_total=self._output_tokens_total,
            total_cost_usd=self._cost_usd_total,
            confirmation_count=self._confirmation_count,
            destructive_actions_executed=list(
                self._destructive_steps_executed
            ),
            final_step_reached=final_step,
        )

    def _finalize_failed(
        self, metadata: RunMetadata, reason: str
    ) -> RunMetadata:
        self._append_event(
            "error",
            f"workflow_failed reason={reason}",
            step_number=self._current_step or None,
        )
        return self._writer.update_status(
            metadata,
            status="failed",
            ended_at=_now_utc(),
            error_message=reason,
            input_tokens_total=self._input_tokens_total,
            output_tokens_total=self._output_tokens_total,
            total_cost_usd=self._cost_usd_total,
            confirmation_count=self._confirmation_count,
            destructive_actions_executed=list(
                self._destructive_steps_executed
            ),
            final_step_reached=self._max_step_seen or None,
        )

    def _finalize_aborted(
        self, metadata: RunMetadata, decision: ConfirmationDecision
    ) -> RunMetadata:
        reason = decision.reason or USER_ABORT_REASON
        self._append_event(
            "aborted",
            f"run_aborted reason={reason}",
            step_number=self._current_step or None,
        )
        return self._writer.update_status(
            metadata,
            status="aborted",
            ended_at=_now_utc(),
            abort_reason=reason,
            input_tokens_total=self._input_tokens_total,
            output_tokens_total=self._output_tokens_total,
            total_cost_usd=self._cost_usd_total,
            confirmation_count=self._confirmation_count,
            destructive_actions_executed=list(
                self._destructive_steps_executed
            ),
            final_step_reached=self._max_step_seen or None,
        )

    def _maybe_emit_cost_warning(self, prev_cost: float, new_cost: float) -> None:
        """Emit an 80%-of-cap warning once, on the first crossing."""

        if self._cost_warning_emitted:
            return
        cap = self._budget.budget.max_cost_usd
        if cap is None:
            return
        if not crossed_warning_threshold(prev_cost, new_cost, cap):
            return
        self._cost_warning_emitted = True
        logger.warning(
            "runner cost for run %s at $%.4f exceeded 80%% of $%.2f cap",
            self._run_id,
            new_cost,
            cap,
        )
        if self._cost_warning_sink is not None:
            try:
                self._cost_warning_sink(new_cost, cap)
            except Exception:  # pragma: no cover - defensive
                logger.exception("cost warning sink raised")

    def _finalize_budget_exceeded(
        self, metadata: RunMetadata, reason: BudgetReason | None
    ) -> RunMetadata:
        if reason == BudgetReason.COST:
            abort_reason = PER_RUN_COST_CAP_REASON
        elif reason is not None:
            abort_reason = f"budget_exceeded:{reason}"
        else:
            abort_reason = "budget_exceeded"
        self._append_event(
            "error",
            f"budget_exceeded reason={reason}",
            step_number=self._current_step or None,
        )
        return self._writer.update_status(
            metadata,
            status="budget_exceeded",
            ended_at=_now_utc(),
            abort_reason=abort_reason,
            input_tokens_total=self._input_tokens_total,
            output_tokens_total=self._output_tokens_total,
            total_cost_usd=self._cost_usd_total,
            confirmation_count=self._confirmation_count,
            destructive_actions_executed=list(
                self._destructive_steps_executed
            ),
            final_step_reached=self._max_step_seen or None,
        )

    def _finalize_agent_stuck(self, metadata: RunMetadata) -> RunMetadata:
        self._append_event(
            "error",
            f"agent_stuck after {self._unknown_action_count} consecutive UnknownAction turns",
            step_number=self._current_step or None,
        )
        return self._writer.update_status(
            metadata,
            status="failed",
            ended_at=_now_utc(),
            error_message=AGENT_STUCK_REASON,
            abort_reason=AGENT_STUCK_REASON,
            input_tokens_total=self._input_tokens_total,
            output_tokens_total=self._output_tokens_total,
            total_cost_usd=self._cost_usd_total,
            confirmation_count=self._confirmation_count,
            destructive_actions_executed=list(
                self._destructive_steps_executed
            ),
            final_step_reached=self._max_step_seen or None,
        )

    def _confirmation_user_message(self, step_number: int) -> Message:
        text = (
            f"User confirmed step {step_number}. Proceed with the "
            "destructive action now."
        )
        return {"role": "user", "content": [{"type": "text", "text": text}]}

    def _nudge_user_message(self) -> Message:
        text = (
            "Your response did not include a tool call or known tag; please "
            "continue or emit <workflow_complete/>."
        )
        return {"role": "user", "content": [{"type": "text", "text": text}]}


class _GateAllow:
    """Pre-action gate allows the call to proceed."""


class _GateAbort:
    """Pre-action gate's confirmation flow resolved with an abort."""

    __slots__ = ("decision",)

    def __init__(self, *, decision: ConfirmationDecision) -> None:
        self.decision = decision


__all__ = [
    "AGENT_STUCK_REASON",
    "KILL_SWITCH_REASON",
    "PER_RUN_COST_CAP_REASON",
    "UNKNOWN_ACTION_ABORT_THRESHOLD",
    "USER_ABORT_REASON",
    "CostWarningSink",
    "Executor",
]
