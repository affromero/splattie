import { NextResponse } from 'next/server';
import { ADMIN_COOKIE, SESSION_MAX_AGE, createSessionToken, verifyPassword } from '@/lib/admin-auth';
import { checkRateLimit, recordFailure, resetRateLimit } from '@/lib/rate-limit';

/** Throttle key: client IP behind the reverse proxy, else a shared bucket. */
function clientKey(request: Request): string {
  const forwarded = request.headers.get('x-forwarded-for');
  return (forwarded ? forwarded.split(',')[0]?.trim() : '') || 'unknown';
}

function lockedResponse(retryAfterSeconds: number): NextResponse {
  return NextResponse.json(
    { error: 'Too many attempts. Try again later.', retryAfter: retryAfterSeconds },
    { status: 429, headers: { 'Retry-After': String(retryAfterSeconds) } }
  );
}

export async function POST(request: Request): Promise<NextResponse> {
  const key = clientKey(request);

  const gate = checkRateLimit(key);
  if (!gate.allowed) {
    return lockedResponse(gate.retryAfterSeconds);
  }

  let password = '';
  try {
    const body = (await request.json()) as { password?: unknown };
    password = typeof body.password === 'string' ? body.password : '';
  } catch {
    password = '';
  }

  if (!verifyPassword(password)) {
    const after = recordFailure(key);
    if (!after.allowed) {
      return lockedResponse(after.retryAfterSeconds);
    }
    return NextResponse.json(
      { error: 'Invalid password', remaining: after.remaining },
      { status: 401 }
    );
  }

  resetRateLimit(key);

  const response = NextResponse.json({ ok: true });
  response.cookies.set(ADMIN_COOKIE, createSessionToken(), {
    httpOnly: true,
    secure: process.env.NODE_ENV === 'production',
    sameSite: 'lax',
    path: '/',
    maxAge: SESSION_MAX_AGE,
  });
  return response;
}
