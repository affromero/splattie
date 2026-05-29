import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { createSessionToken, verifyPassword, verifySessionToken } from '@/lib/admin-auth';

describe('admin-auth', () => {
  beforeEach(() => {
    process.env.ADMIN_PASSWORD = 'hunter2';
    process.env.SESSION_SECRET = 'test-secret';
  });

  afterEach(() => {
    delete process.env.ADMIN_PASSWORD;
    delete process.env.SESSION_SECRET;
  });

  it('accepts the correct password and rejects others', () => {
    expect(verifyPassword('hunter2')).toBe(true);
    expect(verifyPassword('wrong')).toBe(false);
    expect(verifyPassword('')).toBe(false);
  });

  it('rejects all passwords when ADMIN_PASSWORD is unset', () => {
    delete process.env.ADMIN_PASSWORD;
    expect(verifyPassword('hunter2')).toBe(false);
  });

  it('round-trips a valid session token', () => {
    expect(verifySessionToken(createSessionToken())).toBe(true);
  });

  it('rejects a tampered token', () => {
    const token = createSessionToken();
    const tampered = token.slice(0, -1) + (token.endsWith('a') ? 'b' : 'a');
    expect(verifySessionToken(tampered)).toBe(false);
  });

  it('honors expiry', () => {
    const now = 1_700_000_000_000;
    const token = createSessionToken(now);
    expect(verifySessionToken(token, now + 1000)).toBe(true);
    expect(verifySessionToken(token, now + 8 * 24 * 60 * 60 * 1000)).toBe(false);
  });

  it('rejects a token signed with a different secret', () => {
    const token = createSessionToken();
    process.env.SESSION_SECRET = 'different-secret';
    expect(verifySessionToken(token)).toBe(false);
  });

  it('rejects empty or malformed tokens', () => {
    expect(verifySessionToken(undefined)).toBe(false);
    expect(verifySessionToken('')).toBe(false);
    expect(verifySessionToken('no-dot-here')).toBe(false);
  });
});
