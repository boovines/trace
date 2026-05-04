// Browser Agent tab — placeholder during Step 4.2.
// The live observability UI (tier ribbons, frame stream, DOM action log,
// MCP timeline, confirmation modal) lands in Step 4.3.

interface Props {
  rest: string;
  setRest: (rest: string) => void;
}

export function BrowserAgentTab(_props: Props) {
  return (
    <div className="empty">
      Browser Agent tab — live run observability lands in Step 4.3.
    </div>
  );
}
