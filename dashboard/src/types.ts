// Mirror of `recorder.stats.StatsSummary` (see contracts: services/recorder/src/recorder/stats.py).
// Kept as plain type aliases — there's no runtime validation here on purpose:
// the gateway is local, trusted, and fully tested. If the Python contract
// changes, we'll catch the mismatch in a failing chart render.

export interface AppUsage {
  bundle_id: string;
  name: string;
  seconds: number;
  sessions: number;
}

export interface WindowUsage {
  app_name: string;
  window_title: string;
  count: number;
}

export interface DailyBucket {
  date: string;
  recorded_seconds: number;
  event_count: number;
  click_count: number;
  keypress_count: number;
  text_input_chars: number;
}

export interface StatsSummary {
  window_days: number;
  range_start: string;
  range_end: string;
  trajectory_count: number;
  recorded_seconds: number;
  event_counts: Record<string, number>;
  text_input_chars: number;
  top_apps: AppUsage[];
  top_windows: WindowUsage[];
  hour_of_day: number[];
  daily: DailyBucket[];
}
