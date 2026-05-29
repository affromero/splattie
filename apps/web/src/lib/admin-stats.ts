/** Server-side client for the backend's protected analytics endpoint. */

export interface StatsSummary {
  pageviews: number;
  visitors: number;
  avatar_creates: number;
  avatar_views: number;
  bots: number;
  pageviews_today: number;
  pageviews_7d: number;
}

export interface AdminStats {
  range_days: number;
  summary: StatsSummary;
  timeseries: { day: string; pageviews: number; visitors: number }[];
  top_paths: { path: string; views: number }[];
  top_referrers: { referrer: string; count: number }[];
  devices: { device: string; count: number }[];
}

function backendBase(): string {
  return (
    process.env.BACKEND_INTERNAL_URL ??
    process.env.NEXT_PUBLIC_API_URL ??
    'http://localhost:8000'
  );
}

/**
 * Fetch aggregate analytics from the backend using the shared ADMIN_API_TOKEN.
 * Runs server-side only; the token never reaches the browser. Returns null on
 * any failure so the dashboard can render an error state instead of crashing.
 */
export async function fetchAdminStats(days = 30): Promise<AdminStats | null> {
  const token = process.env.ADMIN_API_TOKEN ?? '';
  if (!token) return null;

  try {
    const res = await fetch(`${backendBase()}/admin/stats?days=${days}`, {
      headers: { authorization: `Bearer ${token}` },
      cache: 'no-store',
    });
    if (!res.ok) return null;
    return (await res.json()) as AdminStats;
  } catch {
    return null;
  }
}
