interface Props {
  days: number;
  options: ReadonlyArray<{ value: number; label: string }>;
  loading: boolean;
  onChangeDays: (days: number) => void;
  onRefresh: () => void;
}

export function Header({
  days,
  options,
  loading,
  onChangeDays,
  onRefresh,
}: Props) {
  return (
    <header>
      <h1>
        Trace <span>· usage dashboard</span>
      </h1>
      <div className="controls">
        <label htmlFor="range">Range</label>
        <select
          id="range"
          value={days}
          onChange={(e) => onChangeDays(Number(e.target.value))}
        >
          {options.map((o) => (
            <option key={o.value} value={o.value}>
              {o.label}
            </option>
          ))}
        </select>
        <button type="button" onClick={onRefresh} disabled={loading}>
          {loading ? "Loading…" : "Refresh"}
        </button>
      </div>
    </header>
  );
}
