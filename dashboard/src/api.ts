import type {
  RunEvent,
  RunMetadata,
  RunSummary,
  StatsSummary,
} from "./types";

export async function fetchSummary(days: number): Promise<StatsSummary> {
  const resp = await fetch(`/stats/summary?days=${encodeURIComponent(days)}`);
  if (!resp.ok) {
    throw new Error(`Stats request failed: HTTP ${resp.status}`);
  }
  return (await resp.json()) as StatsSummary;
}

export interface ListRunsParams {
  skillSlug?: string;
  limit?: number;
  offset?: number;
}

export async function fetchRuns(
  params: ListRunsParams = {},
): Promise<RunSummary[]> {
  const qs = new URLSearchParams();
  if (params.skillSlug) qs.set("skill_slug", params.skillSlug);
  if (params.limit !== undefined) qs.set("limit", String(params.limit));
  if (params.offset !== undefined) qs.set("offset", String(params.offset));
  const url = qs.toString() ? `/runs?${qs}` : "/runs";
  const resp = await fetch(url);
  if (!resp.ok) {
    throw new Error(`Runs request failed: HTTP ${resp.status}`);
  }
  return (await resp.json()) as RunSummary[];
}

export async function fetchRunMetadata(runId: string): Promise<RunMetadata> {
  const resp = await fetch(`/run/${encodeURIComponent(runId)}`);
  if (!resp.ok) {
    throw new Error(`Run metadata request failed: HTTP ${resp.status}`);
  }
  return (await resp.json()) as RunMetadata;
}

export async function fetchRunEvents(runId: string): Promise<RunEvent[]> {
  const resp = await fetch(`/run/${encodeURIComponent(runId)}/events`);
  if (!resp.ok) {
    throw new Error(`Run events request failed: HTTP ${resp.status}`);
  }
  return (await resp.json()) as RunEvent[];
}

/**
 * Submit a destructive-action confirmation decision.
 *
 * The runner's POST /run/{id}/confirm endpoint accepts
 * ``decision: "confirm" | "abort"`` plus an optional ``reason`` for
 * audit trail. We expose two thin functions so callers don't have to
 * remember the body shape.
 */
export async function confirmRun(
  runId: string,
  decision: "confirm" | "abort",
  reason?: string,
): Promise<void> {
  const resp = await fetch(`/run/${encodeURIComponent(runId)}/confirm`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ decision, reason }),
  });
  if (!resp.ok) {
    let detail = "";
    try {
      detail = ((await resp.json()) as { detail?: string }).detail ?? "";
    } catch {
      // Non-JSON response — fall through with empty detail.
    }
    throw new Error(
      `Confirm request failed: HTTP ${resp.status}${detail ? ` — ${detail}` : ""}`,
    );
  }
}

/**
 * Trigger the run's kill switch (used by the modal's Decline button so
 * a "no, kill it" decision both rejects the destructive action and
 * stops the run before it can take a worse path elsewhere).
 *
 * Always 200s on the server side — the runner treats abort as
 * idempotent — so the only failure mode is network. Caller surfaces
 * the error toast.
 */
export async function abortRun(runId: string): Promise<void> {
  const resp = await fetch(`/run/${encodeURIComponent(runId)}/abort`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
  });
  if (!resp.ok) {
    throw new Error(`Abort request failed: HTTP ${resp.status}`);
  }
}
