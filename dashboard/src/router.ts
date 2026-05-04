// Minimal hash-based tab router. No external dep.
//
// `window.location.hash` is the source of truth for "which tab is open"
// — that gives us deep-linking and back/forward navigation for free, no
// extra state management library required. Each tab also gets its own
// path inside the hash (e.g. `#/runs/abc-123`) so tabs that drill into a
// detail view (Runs → run detail) don't need their own state plumbing.
//
// The hook re-renders the calling component whenever the hash changes,
// whether from a `setTab(...)` call here or a manual edit / browser
// back-button. SSR isn't a concern (this app is client-only) so we read
// `window.location` directly.

import { useCallback, useEffect, useState } from "react";

export type TabId = "stats" | "runs" | "browser";

export const DEFAULT_TAB: TabId = "stats";

const KNOWN_TABS: ReadonlySet<TabId> = new Set<TabId>([
  "stats",
  "runs",
  "browser",
]);

/**
 * Parse `#/<tab>/<rest>` into `(tab, rest)`. An empty / unknown / malformed
 * hash falls back to ``DEFAULT_TAB`` with empty ``rest``.
 */
export function parseHash(hash: string): { tab: TabId; rest: string } {
  // Strip the leading "#" and an optional leading "/"; what's left is the
  // tab id and everything after it (sub-route, ids, query bits — opaque
  // to the router, owned by the active tab).
  const stripped = hash.replace(/^#\/?/, "");
  if (!stripped) {
    return { tab: DEFAULT_TAB, rest: "" };
  }
  const [head, ...tail] = stripped.split("/");
  const candidate = head as TabId;
  if (!KNOWN_TABS.has(candidate)) {
    return { tab: DEFAULT_TAB, rest: "" };
  }
  return { tab: candidate, rest: tail.join("/") };
}

/**
 * Build the `#/<tab>/<rest>` URL form for a given tab + sub-route.
 *
 * Empty `rest` collapses to just `#/<tab>`. We always emit the leading
 * `#/` so the active tab is unambiguous in the URL bar.
 */
export function buildHash(tab: TabId, rest: string = ""): string {
  return rest ? `#/${tab}/${rest}` : `#/${tab}`;
}

interface UseHashTab {
  tab: TabId;
  /** Sub-route owned by the active tab (everything after `#/<tab>/`). */
  rest: string;
  /** Switch to ``tab`` and reset the sub-route. */
  setTab: (tab: TabId) => void;
  /** Update the sub-route within the active tab. */
  setRest: (rest: string) => void;
}

/**
 * React hook backing the hash-based tab system.
 *
 * Subscribes to ``hashchange``; returns the current `(tab, rest)` plus
 * setters that write back to ``window.location.hash`` (which then loops
 * back through the same listener and triggers a re-render).
 */
export function useHashTab(): UseHashTab {
  const [state, setState] = useState(() => parseHash(window.location.hash));

  useEffect(() => {
    const onHashChange = () => setState(parseHash(window.location.hash));
    window.addEventListener("hashchange", onHashChange);
    // If the page loaded with no hash, normalize to the default tab so
    // bookmarks / shares always have an explicit `#/<tab>` prefix.
    if (!window.location.hash) {
      window.location.hash = buildHash(DEFAULT_TAB);
    }
    return () => window.removeEventListener("hashchange", onHashChange);
  }, []);

  const setTab = useCallback((tab: TabId) => {
    window.location.hash = buildHash(tab);
  }, []);

  const setRest = useCallback((rest: string) => {
    window.location.hash = buildHash(parseHash(window.location.hash).tab, rest);
  }, []);

  return { tab: state.tab, rest: state.rest, setTab, setRest };
}
