// Recorder usage stats — the original single-screen UI, lifted into a
// tab component so the new App.tsx can host it alongside Runs and
// Browser Agent. No behavior change from PR #11; just a relocation.

import { useCallback, useEffect, useState } from "react";
import { fetchSummary } from "../api";
import { DailyChart } from "../components/DailyChart";
import { EventMixChart } from "../components/EventMixChart";
import { HourChart } from "../components/HourChart";
import { MetricCards } from "../components/MetricCards";
import { TopAppsChart } from "../components/TopAppsChart";
import { TopWindowsTable } from "../components/TopWindowsTable";
import type { StatsSummary } from "../types";

const RANGE_OPTIONS: ReadonlyArray<{ value: number; label: string }> = [
  { value: 1, label: "Today" },
  { value: 7, label: "Last 7 days" },
  { value: 14, label: "Last 14 days" },
  { value: 30, label: "Last 30 days" },
  { value: 90, label: "Last 90 days" },
];

export function StatsTab() {
  const [days, setDays] = useState(7);
  const [data, setData] = useState<StatsSummary | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const load = useCallback(async (windowDays: number) => {
    setLoading(true);
    setError(null);
    try {
      const summary = await fetchSummary(windowDays);
      setData(summary);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load(days);
  }, [days, load]);

  return (
    <>
      <div className="tab-controls">
        <label htmlFor="range">Range</label>
        <select
          id="range"
          value={days}
          onChange={(e) => setDays(Number(e.target.value))}
        >
          {RANGE_OPTIONS.map((o) => (
            <option key={o.value} value={o.value}>
              {o.label}
            </option>
          ))}
        </select>
        <button
          type="button"
          onClick={() => void load(days)}
          disabled={loading}
        >
          {loading ? "Loading…" : "Refresh"}
        </button>
      </div>
      {error && (
        <div className="err">
          Failed to load stats: {error}
          <br />
          <small>
            Is the gateway running on <code>127.0.0.1:8765</code>?
          </small>
        </div>
      )}
      {data && (
        <>
          <div className="range-meta">
            {data.range_start} → {data.range_end} · {data.window_days} day
            window
          </div>
          <MetricCards data={data} />
          <div className="grid">
            <div className="card span-2">
              <h2>Daily activity</h2>
              <DailyChart daily={data.daily} />
            </div>
            <div className="card span-2">
              <h2>Hour of day</h2>
              <HourChart hours={data.hour_of_day} />
            </div>
            <div className="card span-2">
              <h2>Top apps by time</h2>
              <TopAppsChart apps={data.top_apps} />
            </div>
            <div className="card span-2">
              <h2>Event mix</h2>
              <EventMixChart counts={data.event_counts} />
            </div>
            <div className="card span-4">
              <h2>Top windows / pages</h2>
              <TopWindowsTable windows={data.top_windows} />
            </div>
          </div>
        </>
      )}
      {!data && !error && loading && <div className="empty">Loading…</div>}
    </>
  );
}
