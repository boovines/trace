import {
  Cell,
  Legend,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
} from "recharts";
import { CHART_COLORS, TOOLTIP_STYLE } from "./theme";

const HIDDEN_TYPES = new Set(["keyframe", "tap_reenabled"]);

export function EventMixChart({ counts }: { counts: Record<string, number> }) {
  const entries = Object.entries(counts)
    .filter(([k]) => !HIDDEN_TYPES.has(k))
    .filter(([, v]) => v > 0)
    .sort((a, b) => b[1] - a[1])
    .map(([name, value]) => ({ name, value }));

  if (!entries.length) {
    return <div className="empty">No interactive events in range.</div>;
  }

  return (
    <div style={{ width: "100%", height: 320 }}>
      <ResponsiveContainer>
        <PieChart>
          <Tooltip contentStyle={TOOLTIP_STYLE} />
          <Legend layout="vertical" verticalAlign="middle" align="right" />
          <Pie
            data={entries}
            dataKey="value"
            nameKey="name"
            innerRadius={60}
            outerRadius={100}
            paddingAngle={1}
            stroke="#161b22"
          >
            {entries.map((_, i) => (
              <Cell key={i} fill={CHART_COLORS[i % CHART_COLORS.length]} />
            ))}
          </Pie>
        </PieChart>
      </ResponsiveContainer>
    </div>
  );
}
