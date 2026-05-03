import type { WindowUsage } from "../types";
import { fmtNumber } from "../format";

export function TopWindowsTable({ windows }: { windows: WindowUsage[] }) {
  if (!windows.length) {
    return (
      <div className="empty">
        No window-focus events captured yet in this range.
      </div>
    );
  }
  return (
    <table className="table">
      <thead>
        <tr>
          <th>App</th>
          <th>Window / page title</th>
          <th className="num">Visits</th>
        </tr>
      </thead>
      <tbody>
        {windows.map((w, i) => (
          <tr key={i}>
            <td>{w.app_name}</td>
            <td className="title">
              <div title={w.window_title}>{w.window_title}</div>
            </td>
            <td className="num">{fmtNumber(w.count)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
