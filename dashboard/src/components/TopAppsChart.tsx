import {
  Bar,
  BarChart,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { AppUsage } from "../types";
import { fmtSeconds } from "../format";
import { CHART_COLORS, MUTED, TOOLTIP_STYLE } from "./theme";

interface TooltipPayloadItem {
  payload?: AppUsage;
}

interface TooltipProps {
  active?: boolean;
  payload?: TooltipPayloadItem[];
}

function AppTooltip({ active, payload }: TooltipProps) {
  if (!active || !payload?.length) return null;
  const a = payload[0]?.payload;
  if (!a) return null;
  return (
    <div style={TOOLTIP_STYLE}>
      <div style={{ fontWeight: 600 }}>{a.name}</div>
      <div>{fmtSeconds(a.seconds)}</div>
      <div style={{ color: MUTED }}>
        {a.sessions} session{a.sessions === 1 ? "" : "s"}
      </div>
    </div>
  );
}

export function TopAppsChart({ apps }: { apps: AppUsage[] }) {
  if (!apps.length) {
    return <div className="empty">No focus history captured yet.</div>;
  }
  const data = apps.map((a) => ({ ...a, minutes: Math.round(a.seconds / 60) }));
  return (
    <div style={{ width: "100%", height: 320 }}>
      <ResponsiveContainer>
        <BarChart
          data={data}
          layout="vertical"
          margin={{ top: 8, right: 16, left: 8, bottom: 0 }}
        >
          <XAxis type="number" stroke={MUTED} fontSize={12} />
          <YAxis dataKey="name" type="category" stroke={MUTED} fontSize={12} width={110} />
          <Tooltip content={<AppTooltip />} cursor={{ fill: "rgba(255,255,255,0.04)" }} />
          <Bar dataKey="minutes" radius={[0, 2, 2, 0]}>
            {data.map((_, i) => (
              <Cell key={i} fill={CHART_COLORS[i % CHART_COLORS.length]} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
