// Runs tab: list-and-detail view backed by GET /runs and
// GET /run/{id}{,/events}. The hash sub-route (``rest`` from
// useHashTab) names the focused run id; an empty rest means "show the
// list". Clicking a row updates the rest, which the App passes down so
// the URL becomes ``#/runs/<run_id>`` and the detail view renders.

import { useCallback, useEffect, useMemo, useState } from "react";
import { fetchRunEvents, fetchRunMetadata, fetchRuns } from "../api";
import type { RunEvent, RunMetadata, RunStatus, RunSummary } from "../types";

interface Props {
  rest: string;
  setRest: (rest: string) => void;
}

export function RunsTab({ rest, setRest }: Props) {
  const focusedRunId = rest || null;
  return (
    <>
      {focusedRunId === null ? (
        <RunsList onSelect={(id) => setRest(id)} />
      ) : (
        <RunDetail
          runId={focusedRunId}
          onBack={() => setRest("")}
        />
      )}
    </>
  );
}

// --- List view ----------------------------------------------------------

interface ListProps {
  onSelect: (runId: string) => void;
}

function RunsList({ onSelect }: ListProps) {
  const [rows, setRows] = useState<RunSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await fetchRuns({ limit: 100 });
      setRows(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  return (
    <>
      <div className="tab-controls">
        <span className="muted-label">
          {rows ? `${rows.length} run${rows.length === 1 ? "" : "s"}` : "—"}
        </span>
        <button type="button" onClick={() => void load()} disabled={loading}>
          {loading ? "Loading…" : "Refresh"}
        </button>
      </div>
      {error && (
        <div className="err">
          Failed to load runs: {error}
          <br />
          <small>
            Is the gateway running on <code>127.0.0.1:8765</code>?
          </small>
        </div>
      )}
      {rows && rows.length === 0 && !error && (
        <div className="empty">No runs yet — start one to see it here.</div>
      )}
      {rows && rows.length > 0 && (
        <div className="card span-4">
          <table className="table">
            <thead>
              <tr>
                <th>Started</th>
                <th>Skill</th>
                <th>Status</th>
                <th>Mode</th>
                <th className="num">Duration</th>
                <th className="num">Cost (USD)</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr
                  key={row.run_id}
                  className="row-clickable"
                  onClick={() => onSelect(row.run_id)}
                >
                  <td>{formatTimestamp(row.started_at)}</td>
                  <td>
                    <code>{row.skill_slug}</code>
                  </td>
                  <td>
                    <StatusPill status={row.status} />
                  </td>
                  <td>{row.mode}</td>
                  <td className="num">
                    {formatDuration(row.duration_seconds)}
                  </td>
                  <td className="num">{formatCost(row.total_cost_usd)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </>
  );
}

// --- Detail view --------------------------------------------------------

interface DetailProps {
  runId: string;
  onBack: () => void;
}

function RunDetail({ runId, onBack }: DetailProps) {
  const [meta, setMeta] = useState<RunMetadata | null>(null);
  const [events, setEvents] = useState<RunEvent[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      // Parallel fetch — metadata and events are independent endpoints.
      const [m, e] = await Promise.all([
        fetchRunMetadata(runId),
        fetchRunEvents(runId),
      ]);
      setMeta(m);
      setEvents(e);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, [runId]);

  useEffect(() => {
    void load();
  }, [load]);

  const sortedEvents = useMemo(
    () => (events ? [...events].sort((a, b) => a.seq - b.seq) : null),
    [events],
  );

  return (
    <>
      <div className="tab-controls">
        <button type="button" onClick={onBack}>
          ← All runs
        </button>
        <span className="muted-label">
          run <code>{runId}</code>
        </span>
        <button type="button" onClick={() => void load()} disabled={loading}>
          {loading ? "Loading…" : "Refresh"}
        </button>
      </div>
      {error && (
        <div className="err">
          Failed to load run: {error}
        </div>
      )}
      {meta && (
        <div className="grid">
          <div className="card span-2">
            <h2>Run metadata</h2>
            <dl className="kv">
              <dt>Skill</dt>
              <dd>
                <code>{meta.skill_slug}</code>
              </dd>
              <dt>Status</dt>
              <dd>
                <StatusPill status={meta.status} />
              </dd>
              <dt>Mode</dt>
              <dd>{meta.mode}</dd>
              <dt>Started</dt>
              <dd>{formatTimestamp(meta.started_at)}</dd>
              <dt>Ended</dt>
              <dd>{formatTimestamp(meta.ended_at)}</dd>
              <dt>Final step</dt>
              <dd>{meta.final_step_reached ?? "—"}</dd>
              <dt>Confirmations</dt>
              <dd>{meta.confirmation_count ?? 0}</dd>
              <dt>Destructive steps</dt>
              <dd>
                {meta.destructive_actions_executed
                  ? meta.destructive_actions_executed.join(", ") || "—"
                  : "—"}
              </dd>
              <dt>Tokens (in/out)</dt>
              <dd>
                {(meta.input_tokens_total ?? 0).toLocaleString()} /{" "}
                {(meta.output_tokens_total ?? 0).toLocaleString()}
              </dd>
              <dt>Cost</dt>
              <dd>{formatCost(meta.total_cost_usd)}</dd>
              {meta.error_message && (
                <>
                  <dt>Error</dt>
                  <dd>
                    <code>{meta.error_message}</code>
                  </dd>
                </>
              )}
              {meta.abort_reason && (
                <>
                  <dt>Abort reason</dt>
                  <dd>
                    <code>{meta.abort_reason}</code>
                  </dd>
                </>
              )}
              {meta.parameters && (
                <>
                  <dt>Parameters</dt>
                  <dd>
                    <pre className="json-block">
                      {JSON.stringify(meta.parameters, null, 2)}
                    </pre>
                  </dd>
                </>
              )}
            </dl>
          </div>
          <div className="card span-2">
            <h2>Events ({sortedEvents?.length ?? 0})</h2>
            {sortedEvents && sortedEvents.length === 0 && (
              <div className="empty">No events yet.</div>
            )}
            {sortedEvents && sortedEvents.length > 0 && (
              <ul className="event-list">
                {sortedEvents.map((evt) => (
                  <li key={evt.seq} className="event">
                    <span className="event-seq">#{evt.seq}</span>
                    <span className="event-type">{evt.type}</span>
                    {evt.step_number !== null && (
                      <span className="event-step">step {evt.step_number}</span>
                    )}
                    <span className="event-message">{evt.message}</span>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </div>
      )}
    </>
  );
}

// --- formatting helpers -------------------------------------------------

function formatTimestamp(value: string | null): string {
  if (!value) return "—";
  // The runner emits ISO8601 with timezone; let the browser localise.
  const d = new Date(value);
  if (Number.isNaN(d.getTime())) return value;
  return d.toLocaleString(undefined, {
    year: "numeric",
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

function formatDuration(seconds: number | null): string {
  if (seconds === null || seconds === undefined) return "—";
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  const mins = Math.floor(seconds / 60);
  const rem = Math.round(seconds - mins * 60);
  return `${mins}m ${rem}s`;
}

function formatCost(usd: number | null): string {
  if (usd === null || usd === undefined) return "—";
  return `$${usd.toFixed(4)}`;
}

function StatusPill({ status }: { status: RunStatus }) {
  return <span className={`pill pill-${status}`}>{status}</span>;
}
