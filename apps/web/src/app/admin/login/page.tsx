'use client';

import { useRouter } from 'next/navigation';
import { useState } from 'react';
import styles from './page.module.css';

interface LoginError {
  error?: string;
  remaining?: number;
  retryAfter?: number;
}

export default function AdminLoginPage() {
  const router = useRouter();
  const [password, setPassword] = useState('');
  const [message, setMessage] = useState('');
  const [locked, setLocked] = useState(false);
  const [busy, setBusy] = useState(false);

  async function onSubmit(event: React.FormEvent) {
    event.preventDefault();
    setBusy(true);
    setMessage('');
    try {
      const res = await fetch('/api/admin/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ password }),
      });

      if (res.ok) {
        router.replace('/admin');
        router.refresh();
        return;
      }

      const data = (await res.json().catch(() => ({}))) as LoginError;
      setPassword('');

      if (res.status === 429) {
        const mins = Math.max(1, Math.ceil((data.retryAfter ?? 0) / 60));
        setLocked(true);
        setMessage(`Too many attempts. Try again in about ${mins} min.`);
      } else if (typeof data.remaining === 'number') {
        setMessage(`Invalid password. ${data.remaining} attempt${data.remaining === 1 ? '' : 's'} left.`);
      } else {
        setMessage('Invalid password.');
      }
    } catch {
      setMessage('Something went wrong. Try again.');
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className={styles.main}>
      <form className={styles.card} onSubmit={onSubmit}>
        <h1 className={styles.title}>Admin</h1>
        <p className={styles.subtitle}>Enter the admin password to view analytics.</p>
        <input
          className={styles.input}
          type="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          placeholder="Password"
          autoFocus
          autoComplete="current-password"
          disabled={locked}
        />
        {message && <p className={styles.error}>{message}</p>}
        <button className={styles.button} type="submit" disabled={busy || locked || !password}>
          {busy ? 'Signing in…' : 'Sign in'}
        </button>
      </form>
    </main>
  );
}
