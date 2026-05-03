import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { CHART_COLORS, GRID, MUTED, TOOLTIP_STYLE } from "./theme";

export function HourChart({ hours }: { hours: number[] }) {
  const data = hours.map((count, hour) => ({ hour: String(hour), count }));
  return (
    <div style={{ width: "100%", height: 260 }}>
      <ResponsiveContainer>
        <BarChart data={data} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
          <CartesianGrid stroke={GRID} strokeDasharray="3 3" />
          <XAxis dataKey="hour" stroke={MUTED} fontSize={12} />
          <YAxis stroke={MUTED} fontSize={12} />
          <Tooltip contentStyle={TOOLTIP_STYLE} />
          <Bar dataKey="count" name="Interactive events" fill={CHART_COLORS[2]} radius={[2, 2, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
