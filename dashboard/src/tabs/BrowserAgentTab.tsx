// Browser Agent tab — Step 4.3 commit B.
//
// Live observability surface for an active run. The tab subscribes to
// /run/{run_id}/stream (typed via subscribeToRun) and renders five
// derived views off the message stream:
//
//   - Live frame viewer: latest JPEG from the runner's
//     dom_frame WS event, served by GET /run/{id}/dom_frames/<file>.
//   - Tier ribbon: per-step lit pill (MCP / DOM / COMPUTER_USE / NONE)
//     parsed from the runner's tier_selected events.
//   - DOM action log: ordered list of browser_dom_* events with the
//     resolved selector / action / value spelled out.
//   - MCP call timeline: server.function and content_text response
//     for each mcp_dispatched / mcp_failed event.
//   - Full event stream (collapsed by default).
//
// Confirmation modal polish + Approve/Decline wiring lands in commit
// C of this PR / Step 4.4 respectively.

import { useEffect, useMemo, useState } from "react";
import { fetchRuns } from "../api";
import type {
  RunSummary,
  WSConfirmationRequestMessage,
  WSDomFrameMessage,
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
// (Unchanged from commit C of 4.2 — kept here so the file is self-contained.)

function RunPicker({ onSelect }: { onSelect: (id: string) => void }) {
  const [rows, setRows] = useState<RunSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
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
  const [latestFrame, setLatestFrame] =
    useState<WSDomFrameMessage | null>(null);

  useEffect(() => {
    setMessages([]);
    setConnected(false);
    setWsError(null);
    setConfirmationPending(null);
    setLatestStatus(null);
    setLatestFrame(null);

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
        } else if (msg.type === "dom_frame") {
          setLatestFrame(msg);
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
        <div className="card span-2">
          <h2>Live frame</h2>
          <LiveFrame frame={latestFrame} />
        </div>
        <div className="card span-2">
          <h2>Tier ribbon</h2>
          <TierRibbon events={eventMessages} />
        </div>
        <div className="card span-2">
          <h2>DOM actions</h2>
          <DOMActionLog events={eventMessages} />
        </div>
        <div className="card span-2">
          <h2>MCP calls</h2>
          <MCPTimeline events={eventMessages} />
        </div>
        <div className="card span-4">
          <h2>Full event stream ({eventMessages.length})</h2>
          {eventMessages.length === 0 ? (
            <div className="empty">
              {connected ? "Listening — no events yet." : "Connecting…"}
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

// --- Live frame viewer --------------------------------------------------

function LiveFrame({ frame }: { frame: WSDomFrameMessage | null }) {
  if (frame === null) {
    return (
      <div className="empty">
        No browser_dom frames yet. The runner emits a frame after every
        successful DOM action.
      </div>
    );
  }
  return (
    <div className="frame-viewport">
      <img
        src={frame.url}
        alt={`Frame ${frame.seq}`}
        className="frame-img"
      />
      <div className="frame-meta">
        frame #{frame.seq} · <code>{frame.filename}</code>
      </div>
    </div>
  );
}

// --- Tier ribbon --------------------------------------------------------

interface TierEntry {
  step: number;
  tier: string;
  message: string;
  fellBack: boolean;
}

const TIER_PILL_CLASS: Record<string, string> = {
  mcp: "tier-mcp",
  browser_dom: "tier-dom",
  computer_use: "tier-cu",
};

// Parses a tier_selected event's message string into structured form.
// Message shape (from runner.executor._format_tier_decision):
//   "step={n} intent={intent} tier={tier}[ synthetic][ details][ (fell_back=true)]"
// The runner enforces this shape so a brittle string parse is fine
// here — when the runner changes the format, the dashboard needs to
// follow.
function parseTierEvent(evt: WSEventMessage): TierEntry | null {
  if (evt.event_type !== "tier_selected") return null;
  const stepMatch = /step=(\d+)/.exec(evt.message);
  const tierMatch = /tier=([\w<>_]+)/.exec(evt.message);
  if (!stepMatch || !tierMatch) return null;
  return {
    step: Number(stepMatch[1]),
    tier: tierMatch[1],
    message: evt.message,
    fellBack: evt.message.includes("fell_back=true"),
  };
}

function TierRibbon({ events }: { events: WSEventMessage[] }) {
  const entries = useMemo(() => {
    const seen = new Map<number, TierEntry>();
    for (const evt of events) {
      const entry = parseTierEvent(evt);
      if (entry !== null) {
        // Last decision per step wins (the runner only emits once per
        // step today, but defensive against future iterations).
        seen.set(entry.step, entry);
      }
    }
    return [...seen.values()].sort((a, b) => a.step - b.step);
  }, [events]);

  if (entries.length === 0) {
    return (
      <div className="empty">
        No tier decisions yet — the runner emits one tier_selected event
        per step at run start.
      </div>
    );
  }
  return (
    <ul className="tier-ribbon">
      {entries.map((e) => (
        <li key={e.step} className="tier-row" title={e.message}>
          <span className="tier-step">step {e.step}</span>
          <span
            className={
              "tier-pill " + (TIER_PILL_CLASS[e.tier] ?? "tier-other")
            }
          >
            {e.tier}
          </span>
          {e.fellBack && <span className="tier-flag">fell back</span>}
        </li>
      ))}
    </ul>
  );
}

// --- DOM action log -----------------------------------------------------

const DOM_EVENT_TYPES: ReadonlySet<string> = new Set([
  "browser_dom_dispatched",
  "browser_dom_failed",
  "browser_dom_skipped",
  "browser_dom_aborted",
]);

function DOMActionLog({ events }: { events: WSEventMessage[] }) {
  const rows = useMemo(
    () => events.filter((e) => DOM_EVENT_TYPES.has(e.event_type)),
    [events],
  );
  if (rows.length === 0) {
    return (
      <div className="empty">No DOM actions yet.</div>
    );
  }
  return (
    <ul className="event-list">
      {rows.map((e) => (
        <li key={e.seq} className="event">
          <span className="event-seq">#{e.seq}</span>
          <span
            className={
              "event-type " + (e.event_type.endsWith("_failed") ? "event-bad" : "")
            }
          >
            {e.event_type.replace("browser_dom_", "")}
          </span>
          {e.step_number !== null && (
            <span className="event-step">step {e.step_number}</span>
          )}
          <span className="event-message">{e.message}</span>
        </li>
      ))}
    </ul>
  );
}

// --- MCP timeline -------------------------------------------------------

const MCP_EVENT_TYPES: ReadonlySet<string> = new Set([
  "mcp_dispatched",
  "mcp_failed",
  "mcp_skipped",
  "mcp_aborted",
]);

function MCPTimeline({ events }: { events: WSEventMessage[] }) {
  const rows = useMemo(
    () => events.filter((e) => MCP_EVENT_TYPES.has(e.event_type)),
    [events],
  );
  if (rows.length === 0) {
    return <div className="empty">No MCP calls yet.</div>;
  }
  return (
    <ul className="event-list">
      {rows.map((e) => (
        <li key={e.seq} className="event">
          <span className="event-seq">#{e.seq}</span>
          <span
            className={
              "event-type " + (e.event_type.endsWith("_failed") ? "event-bad" : "")
            }
          >
            {e.event_type.replace("mcp_", "")}
          </span>
          {e.step_number !== null && (
            <span className="event-step">step {e.step_number}</span>
          )}
          <span className="event-message">{e.message}</span>
        </li>
      ))}
    </ul>
  );
}
