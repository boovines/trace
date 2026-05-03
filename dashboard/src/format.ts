export function fmtSeconds(s: number): string {
  if (!s || s < 1) return "0m";
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  if (h > 0) return `${h}h ${m}m`;
  return `${m}m`;
}

export function fmtNumber(n: number): string {
  return new Intl.NumberFormat().format(n || 0);
}

export function shortDate(iso: string): string {
  // YYYY-MM-DD → "Apr 30". Treated as UTC so daily buckets render
  // consistently regardless of the viewer's timezone.
  const d = new Date(iso + "T00:00:00Z");
  return d.toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
    timeZone: "UTC",
  });
}
