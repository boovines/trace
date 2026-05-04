// Browser Agent tab — Step 4.2 commit C scaffolding.
//
// This commit ships the WebSocket subscription plumbing: pick a run,
// open a typed subscription to /run/{id}/stream via ``subscribeToRun``,
// and render the incoming WSMessage list as a chronological log. The
// real observability surface (per-step tier ribbons, live Playwright
// frames, DOM action log, MCP timeline, confirmation modal) is the
// scope of Step 4.3 — but the WS pipe is the foundation everything
// else hangs off, so it ships here in 4.2.
//
// The tab uses the same hash sub-route convention as Runs: an empty
// ``rest`` shows the run picker; ``rest`` carrying a run id opens the
// live view.

import { useEffect, useMemo, useState } from "react";
import { fetchRuns } from "../api";
import type {
  RunSummary,
  WSConfirmationRequestMessage,
  WSEventMessage,
  WSMessage,
  WSStatusChangeMessage,
} from "../types";
import { subscribeToRun } from "../ws";

interface Props {
  rest: string;
  setRest: (rest: string) => void;
}

export function BrowserAgentTab({ rest, setRest }: Props) {
  const focusedRunId = rest || null;
  return (
    <>
      {focusedRunId === null ? (
        <RunPicker onSelect={(id) => setRest(id)} />
      ) : (
        <LiveRunView runId={focusedRunId} onBack={() => setRest("")} />
      )}
    </>
  );
}

// --- Run picker ---------------------------------------------------------

function RunPicker({ onSelect }: { onSelect: (id: string) => void }) {
  const [rows, setRows] = useState<RunSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        // Surface in-progress runs first; the picker is most useful
        // for live observability of an actively-running workflow.
        const all = await fetchRuns({ limit: 50 });
        if (cancelled) return;
        setRows(all);
      } catch (e) {
        if (cancelled) return;
        setError(e instanceof Error ? e.message : String(e));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const liveRuns = useMemo(
    () =>
      (rows ?? []).filter(
        (r) =>
          r.status === "running" ||
          r.status === "pending" ||
          r.status === "awaiting_confirmation",
      ),
    [rows],
  );

  return (
    <>
      <div className="tab-controls">
        <span className="muted-label">
          Pick a run to live-stream. Active runs appear first.
        </span>
      </div>
      {error && <div className="err">Failed to load runs: {error}</div>}
      {rows && (
        <div className="grid">
          <div className="card span-4">
            <h2>Live ({liveRuns.length})</h2>
            {liveRuns.length === 0 ? (
              <div className="empty">No active runs right now.</div>
            ) : (
              <RunPickerTable rows={liveRuns} onSelect={onSelect} />
            )}
          </div>
          <div className="card span-4">
            <h2>Recent</h2>
            <RunPickerTable rows={rows.slice(0, 20)} onSelect={onSelect} />
          </div>
        </div>
      )}
    </>
  );
}

function RunPickerTable({
  rows,
  onSelect,
}: {
  rows: RunSummary[];
  onSelect: (id: string) => void;
}) {
  return (
    <table className="table">
      <thead>
        <tr>
          <th>Started</th>
          <th>Skill</th>
          <th>Status</th>
          <th>Run ID</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((r) => (
          <tr
            key={r.run_id}
            className="row-clickable"
            onClick={() => onSelect(r.run_id)}
          >
            <td>{r.started_at ?? "—"}</td>
            <td>
              <code>{r.skill_slug}</code>
            </td>
            <td>
              <span className={`pill pill-${r.status}`}>{r.status}</span>
            </td>
            <td>
              <code>{r.run_id.slice(0, 8)}…</code>
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

// --- Live view ----------------------------------------------------------

interface LiveViewProps {
  runId: string;
  onBack: () => void;
}

function LiveRunView({ runId, onBack }: LiveViewProps) {
  const [messages, setMessages] = useState<WSMessage[]>([]);
  const [connected, setConnected] = useState(false);
  const [wsError, setWsError] = useState<string | null>(null);
  const [confirmationPending, setConfirmationPending] =
    useState<WSConfirmationRequestMessage | null>(null);
  const [latestStatus, setLatestStatus] =
    useState<WSStatusChangeMessage | null>(null);

  useEffect(() => {
    setMessages([]);
    setConnected(false);
    setWsError(null);
    setConfirmationPending(null);
    setLatestStatus(null);

    const sub = subscribeToRun(runId, {
      onOpen: () => setConnected(true),
      onClose: () => setConnected(false),
      onMessage: (msg) => {
        setMessages((xs) => [...xs, msg]);
        if (msg.type === "status_change") {
          setLatestStatus(msg);
        } else if (msg.type === "confirmation_request") {
          setConfirmationPending(msg);
        } else if (msg.type === "done") {
          setConfirmationPending(null);
        }
      },
      onError: (e) => setWsError(e.message),
    });
    return () => sub.close();
  }, [runId]);

  const eventMessages = useMemo(
    () => messages.filter((m): m is WSEventMessage => m.type === "event"),
    [messages],
  );

  return (
    <>
      <div className="tab-controls">
        <button type="button" onClick={onBack}>
          ← Pick another run
        </button>
        <span className="muted-label">
          run <code>{runId}</code>
          {" · "}
          <span className={connected ? "ws-on" : "ws-off"}>
            {connected ? "● live" : "○ offline"}
          </span>
          {latestStatus && (
            <>
              {" · status: "}
              <span className={`pill pill-${latestStatus.status}`}>
                {latestStatus.status}
              </span>
            </>
          )}
        </span>
      </div>
      {wsError && (
        <div className="err">
          WebSocket: {wsError}
          <br />
          <small>Will keep retrying with exponential backoff.</small>
        </div>
      )}
      {confirmationPending && (
        <div className="confirm-banner">
          <strong>Confirmation requested</strong> · step{" "}
          {confirmationPending.step_number} ·{" "}
          {confirmationPending.destructive_reason}
          <div className="confirm-step-text">
            {confirmationPending.step_text}
          </div>
          <small>
            Step 4.4 will wire Approve/Decline buttons here against the
            existing /run/{"{id}"}/confirm endpoint.
          </small>
        </div>
      )}
      <div className="grid">
        <div className="card span-4">
          <h2>Event stream ({eventMessages.length})</h2>
          {eventMessages.length === 0 ? (
            <div className="empty">
              {connected
                ? "Listening — no events yet."
                : "Connecting…"}
            </div>
          ) : (
            <ul className="event-list">
              {eventMessages.map((evt) => (
                <li key={evt.seq} className="event">
                  <span className="event-seq">#{evt.seq}</span>
                  <span className="event-type">{evt.event_type}</span>
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
    </>
  );
}
