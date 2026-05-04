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
