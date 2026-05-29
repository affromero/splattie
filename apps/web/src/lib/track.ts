import { API_URL } from '@/lib/api-client';

export type TrackType = 'pageview' | 'avatar_view' | 'avatar_create';

/**
 * Fire-and-forget analytics beacon to the backend.
 *
 * Sent as text/plain so it stays a CORS-simple request (no preflight) and works
 * with navigator.sendBeacon across the splattie.app -> api.splattie.ai origin.
 * Analytics must never throw into the page.
 */
export function track(type: TrackType, path: string, meta?: Record<string, unknown>): void {
  if (typeof window === 'undefined') return;

  const body = JSON.stringify({
    type,
    path,
    referrer: document.referrer || null,
    meta: meta ?? null,
  });
  const url = `${API_URL}/track`;

  try {
    if (typeof navigator !== 'undefined' && typeof navigator.sendBeacon === 'function') {
      const blob = new Blob([body], { type: 'text/plain' });
      if (navigator.sendBeacon(url, blob)) return;
    }
    void fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'text/plain' },
      body,
      keepalive: true,
    }).catch(() => undefined);
  } catch {
    // swallow — analytics failures must never break the page
  }
}
