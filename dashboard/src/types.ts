// Mirror of `recorder.stats.StatsSummary` (see contracts: services/recorder/src/recorder/stats.py).
// Kept as plain type aliases — there's no runtime validation here on purpose:
// the gateway is local, trusted, and fully tested. If the Python contract
// changes, we'll catch the mismatch in a failing chart render.

export interface AppUsage {
  bundle_id: string;
  name: string;
  seconds: number;
  sessions: number;
}

export interface WindowUsage {
  app_name: string;
  window_title: string;
  count: number;
}

export interface DailyBucket {
  date: string;
  recorded_seconds: number;
  event_count: number;
  click_count: number;
  keypress_count: number;
  text_input_chars: number;
}

export interface StatsSummary {
  window_days: number;
  range_start: string;
  range_end: string;
  trajectory_count: number;
  recorded_seconds: number;
  event_counts: Record<string, number>;
  text_input_chars: number;
  top_apps: AppUsage[];
  top_windows: WindowUsage[];
  hour_of_day: number[];
  daily: DailyBucket[];
}

// Runner contracts. Mirror of:
//   services/runner/src/runner/run_index.py    (RunSummary row shape)
//   services/runner/src/runner/schema.py       (RunMetadata)
//   services/runner/src/runner/run_writer.py   (events.jsonl row shape)
// Stays as plain type aliases — gateway is local + trusted.

export type RunStatus =
  | "pending"
  | "running"
  | "awaiting_confirmation"
  | "succeeded"
  | "failed"
  | "aborted"
  | "budget_exceeded";

export type RunMode = "dry_run" | "execute";

export interface RunSummary {
  run_id: string;
  skill_slug: string;
  status: RunStatus;
  mode: RunMode;
  started_at: string | null;
  ended_at: string | null;
  duration_seconds: number | null;
  total_cost_usd: number | null;
}

// ``run_metadata.json`` is richer than RunSummary; we surface the
// subset the dashboard renders today. The gateway returns the file
// verbatim so additional fields can be added to ``RunMetadata`` without
// invalidating this type — `Record<string, unknown>` shape keeps tsc
// happy on unknown extras.
export interface RunMetadata {
  run_id: string;
  skill_slug: string;
  status: RunStatus;
  mode: RunMode;
  started_at: string | null;
  ended_at: string | null;
  parameters: Record<string, string> | null;
  input_tokens_total: number | null;
  output_tokens_total: number | null;
  total_cost_usd: number | null;
  confirmation_count: number | null;
  destructive_actions_executed: number[] | null;
  final_step_reached: number | null;
  error_message: string | null;
  abort_reason: string | null;
}

// One row from ``events.jsonl`` as served by GET /run/{id}/events.
// Note: the inner event-type field is ``type`` here (matches the
// runner's RunWriter serialization). The WebSocket stream uses a
// different envelope where ``type`` names the *envelope* type and
// ``event_type`` carries this same value — see ``WSEvent`` once that
// lands in commit C.
export interface RunEvent {
  seq: number;
  type: string;
  message: string;
  step_number: number | null;
  screenshot_ref: string | null;
  timestamp_ms: number | null;
}

// WebSocket envelope shapes broadcast by the runner over
// /run/{run_id}/stream. Source of truth:
//   services/runner/src/runner/observing_writer.py   (event, status_change, turn_complete)
//   services/runner/src/runner/run_manager.py        (warning, done, status_change on crash)
//   services/runner/src/runner/confirmation.py       (confirmation_request)
//
// Note the asymmetry vs RunEvent: the WS envelope's outer ``type``
// names the *envelope* kind (event / status_change / ...), and the
// inner event type — when an envelope wraps an events.jsonl row —
// rides on ``event_type``.

export interface WSEventMessage {
  type: "event";
  run_id: string;
  seq: number;
  event_type: string;
  message: string;
  step_number: number | null;
  screenshot_ref: string | null;
}

export interface WSStatusChangeMessage {
  type: "status_change";
  run_id: string;
  status: RunStatus;
  metadata?: Record<string, unknown>;
}

export interface WSTurnCompleteMessage {
  type: "turn_complete";
  run_id: string;
  turn_number: number;
  input_tokens: number;
  output_tokens: number;
}

export interface WSConfirmationRequestMessage {
  type: "confirmation_request";
  run_id: string;
  step_number: number;
  step_text: string;
  destructive_reason: string;
  screenshot_url: string | null;
}

export interface WSWarningMessage {
  type: "warning";
  run_id: string;
  kind: string;
  cost_usd?: number;
  cap_usd?: number;
}

export interface WSDoneMessage {
  type: "done";
  run_id: string;
  final_metadata: Record<string, unknown>;
}

// Live JPEG frame captured by the browser_dom tier after each action.
// ``url`` is the path under the gateway (``/run/RUN_ID/dom_frames/...``)
// — set ``<img src>`` directly.
export interface WSDomFrameMessage {
  type: "dom_frame";
  run_id: string;
  seq: number;
  filename: string;
  url: string;
}

export type WSMessage =
  | WSEventMessage
  | WSStatusChangeMessage
  | WSTurnCompleteMessage
  | WSConfirmationRequestMessage
  | WSWarningMessage
  | WSDoneMessage
  | WSDomFrameMessage;
