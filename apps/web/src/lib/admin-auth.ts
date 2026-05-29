import { createHmac, timingSafeEqual } from 'node:crypto';

/**
 * Stateless admin session auth. The session cookie holds `${exp}.${hmac}` where
 * the HMAC is keyed by SESSION_SECRET, so no server-side session store is needed.
 * Server-only — never import from a client component.
 */

export const ADMIN_COOKIE = 'splattie_admin';
export const SESSION_MAX_AGE = 7 * 24 * 60 * 60; // 7 days, in seconds

function sessionSecret(): string {
  return process.env.SESSION_SECRET ?? '';
}

function constantTimeEquals(a: string, b: string): boolean {
  const bufA = Buffer.from(a);
  const bufB = Buffer.from(b);
  if (bufA.length !== bufB.length) return false;
  return timingSafeEqual(bufA, bufB);
}

function sign(payload: string): string {
  return createHmac('sha256', sessionSecret()).update(payload).digest('hex');
}

/** Constant-time comparison of the submitted password against ADMIN_PASSWORD. */
export function verifyPassword(input: string): boolean {
  const expected = process.env.ADMIN_PASSWORD ?? '';
  if (!expected || !input) return false;
  return constantTimeEquals(input, expected);
}

/** Build a signed session token valid for SESSION_MAX_AGE. */
export function createSessionToken(nowMs: number = Date.now()): string {
  const exp = Math.floor(nowMs / 1000) + SESSION_MAX_AGE;
  const payload = String(exp);
  return `${payload}.${sign(payload)}`;
}

/** Validate a session token's signature and expiry. */
export function verifySessionToken(
  token: string | undefined,
  nowMs: number = Date.now()
): boolean {
  if (!token || !sessionSecret()) return false;

  const dot = token.lastIndexOf('.');
  if (dot <= 0) return false;

  const payload = token.slice(0, dot);
  const signature = token.slice(dot + 1);
  if (!constantTimeEquals(signature, sign(payload))) return false;

  const exp = Number(payload);
  if (!Number.isFinite(exp)) return false;
  return exp > Math.floor(nowMs / 1000);
}
