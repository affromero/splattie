/**
 * In-memory, per-key login throttle for the admin panel.
 *
 * After MAX_ATTEMPTS failed tries within WINDOW_MS, the key (client IP) is
 * locked out for LOCKOUT_MS. State is process-local — fine for the single
 * standalone web container; a restart simply clears counters (fail-open, but
 * an attacker can't trigger a restart). Successful logins reset the counter.
 */

export const MAX_ATTEMPTS = 3;
const WINDOW_MS = 15 * 60 * 1000;
const LOCKOUT_MS = 15 * 60 * 1000;
const MAX_TRACKED_KEYS = 10_000; // bound memory against distinct-IP flooding

interface Attempt {
  count: number;
  firstAt: number;
  lockedUntil: number;
}

const attempts = new Map<string, Attempt>();

export interface RateResult {
  allowed: boolean;
  remaining: number;
  retryAfterSeconds: number;
}

function prune(now: number): void {
  if (attempts.size < MAX_TRACKED_KEYS) return;
  for (const [key, a] of attempts) {
    if (a.lockedUntil <= now && now - a.firstAt > WINDOW_MS) attempts.delete(key);
  }
}

/** Whether `key` may attempt a login right now (does not consume an attempt). */
export function checkRateLimit(key: string, now: number = Date.now()): RateResult {
  const a = attempts.get(key);
  if (a && a.lockedUntil > now) {
    return { allowed: false, remaining: 0, retryAfterSeconds: Math.ceil((a.lockedUntil - now) / 1000) };
  }
  const used = a && now - a.firstAt <= WINDOW_MS ? a.count : 0;
  return { allowed: true, remaining: Math.max(0, MAX_ATTEMPTS - used), retryAfterSeconds: 0 };
}

/** Record a failed attempt and report the resulting lockout state. */
export function recordFailure(key: string, now: number = Date.now()): RateResult {
  prune(now);
  let a = attempts.get(key);
  if (!a || now - a.firstAt > WINDOW_MS) {
    a = { count: 0, firstAt: now, lockedUntil: 0 };
  }
  a.count += 1;
  if (a.count >= MAX_ATTEMPTS) {
    a.lockedUntil = now + LOCKOUT_MS;
  }
  attempts.set(key, a);

  const locked = a.lockedUntil > now;
  return {
    allowed: !locked,
    remaining: Math.max(0, MAX_ATTEMPTS - a.count),
    retryAfterSeconds: locked ? Math.ceil((a.lockedUntil - now) / 1000) : 0,
  };
}

/** Clear all throttle state for `key` (called on successful login). */
export function resetRateLimit(key: string): void {
  attempts.delete(key);
}

/** Test-only: wipe all tracked state. */
export function __resetAll(): void {
  attempts.clear();
}
