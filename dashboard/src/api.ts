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
