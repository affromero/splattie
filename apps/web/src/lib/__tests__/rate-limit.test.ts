import { beforeEach, describe, expect, it } from 'vitest';
import {
  MAX_ATTEMPTS,
  __resetAll,
  checkRateLimit,
  recordFailure,
  resetRateLimit,
} from '@/lib/rate-limit';

const T0 = 1_700_000_000_000;

describe('rate-limit', () => {
  beforeEach(() => {
    __resetAll();
  });

  it('allows attempts up to the limit, then locks out', () => {
    expect(checkRateLimit('1.1.1.1', T0)).toMatchObject({ allowed: true, remaining: MAX_ATTEMPTS });

    for (let i = 1; i < MAX_ATTEMPTS; i++) {
      const r = recordFailure('1.1.1.1', T0);
      expect(r.allowed).toBe(true);
    }

    const last = recordFailure('1.1.1.1', T0);
    expect(last.allowed).toBe(false);
    expect(last.retryAfterSeconds).toBeGreaterThan(0);

    const gate = checkRateLimit('1.1.1.1', T0);
    expect(gate.allowed).toBe(false);
  });

  it('keeps separate counters per key', () => {
    for (let i = 0; i < MAX_ATTEMPTS; i++) recordFailure('attacker', T0);
    expect(checkRateLimit('attacker', T0).allowed).toBe(false);
    expect(checkRateLimit('innocent', T0).allowed).toBe(true);
  });

  it('clears the lockout after a successful login (reset)', () => {
    for (let i = 0; i < MAX_ATTEMPTS; i++) recordFailure('2.2.2.2', T0);
    expect(checkRateLimit('2.2.2.2', T0).allowed).toBe(false);

    resetRateLimit('2.2.2.2');
    expect(checkRateLimit('2.2.2.2', T0)).toMatchObject({ allowed: true, remaining: MAX_ATTEMPTS });
  });

  it('lifts the lockout once the window has elapsed', () => {
    for (let i = 0; i < MAX_ATTEMPTS; i++) recordFailure('3.3.3.3', T0);
    expect(checkRateLimit('3.3.3.3', T0).allowed).toBe(false);

    // 16 minutes later, both the lockout and the attempt window have expired.
    const later = T0 + 16 * 60 * 1000;
    expect(checkRateLimit('3.3.3.3', later).allowed).toBe(true);
  });
});
