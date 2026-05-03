import {
  Bar,
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { DailyBucket } from "../types";
import { shortDate } from "../format";
import { CHART_COLORS, GRID, MUTED, TOOLTIP_STYLE } from "./theme";

export function DailyChart({ daily }: { daily: DailyBucket[] }) {
  const data = daily.map((d) => ({
    date: shortDate(d.date),
    minutes: Math.round(d.recorded_seconds / 60),
    events: d.event_count,
  }));

  return (
    <div style={{ width: "100%", height: 260 }}>
      <ResponsiveContainer>
        <ComposedChart data={data} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
          <CartesianGrid stroke={GRID} strokeDasharray="3 3" />
          <XAxis dataKey="date" stroke={MUTED} fontSize={12} />
          <YAxis yAxisId="left" stroke={MUTED} fontSize={12} />
          <YAxis yAxisId="right" orientation="right" stroke={MUTED} fontSize={12} />
          <Tooltip contentStyle={TOOLTIP_STYLE} />
          <Legend />
          <Bar
            yAxisId="left"
            dataKey="minutes"
            name="Minutes recorded"
            fill={CHART_COLORS[0]}
            radius={[2, 2, 0, 0]}
          />
          <Line
            yAxisId="right"
            dataKey="events"
            name="Events"
            stroke={CHART_COLORS[1]}
            strokeWidth={2}
            dot={false}
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
