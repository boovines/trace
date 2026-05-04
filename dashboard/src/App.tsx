// Top-level dashboard shell. Owns the tab bar; delegates body rendering
// to the active tab component. Tab state lives in the URL hash via
// ``useHashTab`` so deep-links and back/forward navigation work without
// any extra state library.
//
// Browser Agent gets a tab placeholder here in 4.2; the real
// observability UI lands in 4.3. Runs is a real list-and-detail tab as
// of this PR.

import { type TabId, useHashTab } from "./router";
import { BrowserAgentTab } from "./tabs/BrowserAgentTab";
import { RunsTab } from "./tabs/RunsTab";
import { StatsTab } from "./tabs/StatsTab";

interface TabSpec {
  id: TabId;
  label: string;
}

const TABS: ReadonlyArray<TabSpec> = [
  { id: "stats", label: "Stats" },
  { id: "runs", label: "Runs" },
  { id: "browser", label: "Browser Agent" },
];

export function App() {
  const { tab, rest, setTab, setRest } = useHashTab();
  return (
    <>
      <header>
        <h1>
          Trace <span>· dashboard</span>
        </h1>
        <nav className="tabs" aria-label="Sections">
          {TABS.map((t) => (
            <button
              key={t.id}
              type="button"
              className={"tab" + (tab === t.id ? " tab-active" : "")}
              onClick={() => setTab(t.id)}
              aria-pressed={tab === t.id}
            >
              {t.label}
            </button>
          ))}
        </nav>
      </header>
      <main>
        {tab === "stats" && <StatsTab />}
        {tab === "runs" && <RunsTab rest={rest} setRest={setRest} />}
        {tab === "browser" && <BrowserAgentTab rest={rest} setRest={setRest} />}
      </main>
    </>
  );
}
