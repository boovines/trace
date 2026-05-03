import { fmtNumber, fmtSeconds } from "../format";
import type { StatsSummary } from "../types";

export function MetricCards({ data }: { data: StatsSummary }) {
  const totalEvents = Object.values(data.event_counts).reduce(
    (a, b) => a + b,
    0,
  );

  return (
    <div className="grid">
      <Metric label="Recorded time" value={fmtSeconds(data.recorded_seconds)} hint="across recordings" />
      <Metric label="Events" value={fmtNumber(totalEvents)} hint="captured" />
      <Metric label="Recordings" value={fmtNumber(data.trajectory_count)} hint="sessions" />
      <Metric label="Characters typed" value={fmtNumber(data.text_input_chars)} hint="typed" />
    </div>
  );
}

function Metric({
  label,
  value,
  hint,
}: {
  label: string;
  value: string;
  hint?: string;
}) {
  return (
    <div className="card">
      <h2>{label}</h2>
      <div className="metric">
        {value} {hint && <small>{hint}</small>}
      </div>
    </div>
  );
}
