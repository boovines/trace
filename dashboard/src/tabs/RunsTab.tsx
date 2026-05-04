// Runs tab — placeholder during commit A of Step 4.2 (tab framework).
// The real list-and-detail UI lands in the next commit.

interface Props {
  rest: string;
  setRest: (rest: string) => void;
}

export function RunsTab(_props: Props) {
  return (
    <div className="empty">
      Runs tab — list view coming up in the next commit of this PR.
    </div>
  );
}
