// Typed WebSocket client for /run/{run_id}/stream.
//
// Wraps the browser's ``WebSocket`` with three things the bare API
// doesn't give us:
//
//   1. Typed message decoding — JSON parse + dispatch into the
//      ``WSMessage`` discriminated union, with malformed payloads
//      surfaced through an ``onError`` callback rather than silently
//      throwing inside ``onmessage``.
//   2. Exponential-backoff reconnection — the runner's WS handler
//      closes the socket once a run finishes, but a UI that's still
//      open should reconnect cheaply if the user restarts the run on
//      the same id (rare in practice, but trivial to support and it
//      also guards against transient network blips during a long
//      live run).
//   3. A tidy ``close()`` that prevents the auto-reconnect from
//      firing — important because React effects need a clean teardown
//      that doesn't leave orphan timers running.
//
// Production usage from a React component:
//
//     useEffect(() => {
//       const sub = subscribeToRun(runId, {
//         onMessage: (m) => setMessages((xs) => [...xs, m]),
//         onError: (e) => console.warn("ws", e),
//       });
//       return () => sub.close();
//     }, [runId]);

import type { WSMessage } from "./types";

export interface SubscribeOptions {
  onMessage: (msg: WSMessage) => void;
  onOpen?: () => void;
  onClose?: (event: CloseEvent) => void;
  onError?: (err: Error) => void;
}

export interface Subscription {
  /** Stop receiving messages and prevent auto-reconnect. Idempotent. */
  close: () => void;
  /** ``true`` while the underlying socket is open. */
  readonly isOpen: () => boolean;
}

const INITIAL_BACKOFF_MS = 500;
const MAX_BACKOFF_MS = 10_000;

/**
 * Open a typed subscription to ``/run/{run_id}/stream``.
 *
 * The dashboard is served from the same origin as the gateway in prod
 * and through a ``ws: true`` Vite proxy in dev, so we resolve the URL
 * relative to ``window.location`` rather than hard-coding 127.0.0.1.
 * That keeps this file agnostic of dev-vs-prod hosting.
 */
export function subscribeToRun(
  runId: string,
  options: SubscribeOptions,
): Subscription {
  let socket: WebSocket | null = null;
  let backoff = INITIAL_BACKOFF_MS;
  let reconnectTimer: number | null = null;
  let closed = false;

  const url = wsUrlForRun(runId);

  const connect = () => {
    if (closed) return;
    let ws: WebSocket;
    try {
      ws = new WebSocket(url);
    } catch (e) {
      // Construction can throw on a bad URL or blocked WebSocket
      // permissions. Surface to the caller and bail — there's no
      // socket to reconnect.
      options.onError?.(e instanceof Error ? e : new Error(String(e)));
      return;
    }
    socket = ws;

    ws.onopen = () => {
      // Reset backoff on every successful open so a *new* failure
      // starts from the small backoff again.
      backoff = INITIAL_BACKOFF_MS;
      options.onOpen?.();
    };

    ws.onmessage = (event) => {
      let payload: unknown;
      try {
        payload = JSON.parse(event.data as string);
      } catch (e) {
        options.onError?.(
          e instanceof Error ? e : new Error("WS payload not JSON"),
        );
        return;
      }
      // Trust the runner: we don't validate every field, just that the
      // payload has a ``type`` discriminator. Anything richer would
      // require a runtime schema and the gateway is local + trusted.
      if (
        typeof payload === "object" &&
        payload !== null &&
        typeof (payload as { type?: unknown }).type === "string"
      ) {
        options.onMessage(payload as WSMessage);
      } else {
        options.onError?.(new Error("WS message missing `type` discriminator"));
      }
    };

    ws.onerror = () => {
      // The browser's ``ErrorEvent`` is intentionally opaque (security
      // reason), so we just signal "something went wrong" and let
      // ``onclose`` carry the reconnect logic.
      options.onError?.(new Error("WebSocket error"));
    };

    ws.onclose = (event) => {
      socket = null;
      options.onClose?.(event);
      if (closed) return;
      // Reconnect with exponential backoff. A clean close (1000) from
      // the server still triggers a reconnect attempt — the run might
      // have completed and the user could legitimately re-open the
      // same run id (rare). The cheap retries cap out fast at
      // MAX_BACKOFF_MS so we don't burn CPU on a permanently-dead
      // socket.
      reconnectTimer = window.setTimeout(connect, backoff);
      backoff = Math.min(backoff * 2, MAX_BACKOFF_MS);
    };
  };

  connect();

  return {
    close: () => {
      closed = true;
      if (reconnectTimer !== null) {
        window.clearTimeout(reconnectTimer);
        reconnectTimer = null;
      }
      if (socket !== null) {
        // Use ``close()`` (not ``terminate``) so the server-side
        // handler sees a clean close event and tears down its
        // subscription cleanly.
        socket.close();
        socket = null;
      }
    },
    isOpen: () => socket !== null && socket.readyState === WebSocket.OPEN,
  };
}

/**
 * Resolve the absolute ``ws[s]://...`` URL for a run-stream
 * subscription. Exported so tests can assert the URL shape without
 * actually opening a socket.
 */
export function wsUrlForRun(runId: string): string {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${window.location.host}/run/${encodeURIComponent(runId)}/stream`;
}
