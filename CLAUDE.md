# CLAUDE.md — Mirada

> Upload a photo. Get a 3D head whose eyes follow your cursor.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Next.js 15 App Router, TypeScript strict, CSS Modules |
| 3D Rendering | Spark 2.0 (@sparkjsdev/spark), Three.js, WebGL2 |
| Segmentation | SAM 3 ONNX (client-side WebGPU) + server fallback |
| Backend (GPU) | Python 3.11, FastAPI, uv + pyproject.toml |
| Head Generation | LAM (SIGGRAPH 2025) — swappable via HeadGenerationMethod protocol |
| Animation | FLAME LBS (client-side, no neural network) |
| Compression | SPZ v4 (Niantic) — ~10x over PLY |
| Deploy | Docker + Caddy + GitHub Actions SSH |

## Monorepo

npm workspace: `apps/web/` (Next.js). Backend: `backend/` (Python, not an npm workspace).

## Commands

### Frontend

```bash
npm run dev          # Dev server on port 4001
npm run build        # Production build
npm run lint         # ESLint
npm run type-check   # TypeScript strict check
npm run test         # Vitest
npm run ci           # lint + type-check + test + build
```

### Backend

```bash
cd backend
uv sync
uv run uvicorn mirada.api.app:create_app --factory --reload --port 8000
uv run pytest
uv run pre-commit run --all-files
```

## Key Rules

1. **CSS Modules only** — no Tailwind, no inline styles
2. **TypeScript strict** — no `any`
3. **Server Components by default** — `'use client'` only for interactive
4. **Python: uv only** — pip is FORBIDDEN
5. **Swappable methods** — all head generation goes through HeadGenerationMethod protocol
6. **SPZ compression** — all models compressed to <2MB before serving to client
7. **Client-side animation** — FLAME LBS runs in browser, no server roundtrip for gaze
8. **Port 4001** — dev and production

## Design System

- Background: `#0A0A0F` / Surface: `#12121A` / Elevated: `#1E1E2A`
- Primary: `#60A5FA` (blue) / Accent: `#F59E0B` (amber)
- Fonts: Space Grotesk (heading/body) + JetBrains Mono (code)
