import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// Anti-clickjacking + no-store for the admin panel and its API, so the login
// can't be framed into a phishing page and stats pages aren't cached.
const ADMIN_SECURITY_HEADERS = [
  { key: 'X-Frame-Options', value: 'DENY' },
  { key: 'Content-Security-Policy', value: "frame-ancestors 'none'" },
  { key: 'X-Content-Type-Options', value: 'nosniff' },
  { key: 'Referrer-Policy', value: 'no-referrer' },
  { key: 'Cache-Control', value: 'no-store, max-age=0' },
];

/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'standalone',
  outputFileTracingRoot: path.join(__dirname, '../../'),
  async headers() {
    return [
      { source: '/admin', headers: ADMIN_SECURITY_HEADERS },
      { source: '/admin/:path*', headers: ADMIN_SECURITY_HEADERS },
      { source: '/api/admin/:path*', headers: ADMIN_SECURITY_HEADERS },
    ];
  },
};

export default nextConfig;
