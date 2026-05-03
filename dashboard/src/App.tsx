import { useCallback, useEffect, useState } from "react";
import { fetchSummary } from "./api";
import type { StatsSummary } from "./types";
import { Header } from "./components/Header";
import { MetricCards } from "./components/MetricCards";
import { DailyChart } from "./components/DailyChart";
import { HourChart } from "./components/HourChart";
import { TopAppsChart } from "./components/TopAppsChart";
import { EventMixChart } from "./components/EventMixChart";
import { TopWindowsTable } from "./components/TopWindowsTable";

const RANGE_OPTIONS: ReadonlyArray<{ value: number; label: string }> = [
  { value: 1, label: "Today" },
  { value: 7, label: "Last 7 days" },
  { value: 14, label: "Last 14 days" },
  { value: 30, label: "Last 30 days" },
  { value: 90, label: "Last 90 days" },
];

export function App() {
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
      <Header
        days={days}
        options={RANGE_OPTIONS}
        loading={loading}
        onChangeDays={setDays}
        onRefresh={() => void load(days)}
      />
      <main>
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
      </main>
    </>
  );
}
