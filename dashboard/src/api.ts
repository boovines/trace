import type { StatsSummary } from "./types";

export async function fetchSummary(days: number): Promise<StatsSummary> {
  const resp = await fetch(`/stats/summary?days=${encodeURIComponent(days)}`);
  if (!resp.ok) {
    throw new Error(`Stats request failed: HTTP ${resp.status}`);
  }
  return (await resp.json()) as StatsSummary;
}
