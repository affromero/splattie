# Security

Splattie is a public web app (splattie.app) backed by a GPU generation
service. If you find a vulnerability, open a private security advisory on
this repository (or contact the maintainer directly) rather than a public
issue.

## Model

- The generation API (`/generate`, `/generate-from-upload`) is public and
  unauthenticated by design. Uploaded images are processed to a `.splattie`
  bundle and streamed back; the backend never executes user-provided code.
- The admin panel (`/admin`) is the only authenticated surface.
  `ADMIN_PASSWORD` is compared in constant time; sessions are HMAC-SHA256
  signed cookies keyed by `SESSION_SECRET`, `HttpOnly`, `Secure`, and expire
  after seven days. There is no server-side session store.
- Admin login is throttled per client IP: three failed attempts in fifteen
  minutes locks the address out for fifteen minutes. Tracked keys are bounded
  so distinct-IP flooding cannot exhaust memory.
- All responses send `Content-Security-Policy: frame-ancestors 'none'` and
  `X-Frame-Options: DENY`, so the site cannot be framed into a phishing page.
  Stats pages are never cached.
- Secrets live in Infisical and are injected at deploy time. No credentials
  are committed; `.env` files are gitignored.
- The GPU backend runs vendored research code (LAM, LHM, TRELLIS, Puppeteer,
  SMAL) inside a CUDA container. Those submodules are excluded from static
  analysis and are trusted only as far as their upstream.
- Supply chain: CodeQL (JavaScript/TypeScript and Python), gitleaks, and
  Dependabot (weekly, 7-day cooldown, one grouped PR per ecosystem) run on
  every push. Workflow actions are pinned by commit SHA.
