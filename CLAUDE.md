# CLAUDE.md — Splattie

> Upload a photo. Get a 3D head whose eyes follow your cursor.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Next.js 15 App Router, TypeScript strict, CSS Modules |
| 3D Rendering | LAM_WebRender (gaussian-splat-renderer-for-lam), WebGL |
| Backend (GPU) | Python 3.10, FastAPI, uv + pyproject.toml |
| Head Generation | LAM (SIGGRAPH 2025) — swappable via HeadGenerationMethod protocol |
| Animation | ARKit blendshapes via FLAME mesh (client-side, 60fps) |
| Face Detection | readPixels after WebGL render (pixel-perfect) |
| Compression | @playcanvas/splat-transform (PLY → SPZ) |
| Deploy | Docker (CUDA) + Caddy + GitHub Actions SSH |

## Monorepo

- `apps/web/` — Next.js frontend (npm workspace)
- `backend/` — FastAPI GPU service (Python, NOT npm workspace)
- `packages/lam-renderer/` — LAM_WebRender submodule (built with Vite)
- `backend/vendor/LAM/` — LAM submodule

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
uv run uvicorn splattie.api.app:create_app --factory --reload --port 8000
uv run pytest
uv run ruff check src/ tests/
```

### Renderer

```bash
npm run build:renderer   # Build + patch + deploy to apps/web/public/demo/
```

### GPU Setup (on gcloud-h100)

```bash
cd backend
bash scripts/setup-gpu.sh   # torch, CUDA extensions, LAM weights, Blender, FBX SDK
```

### Docker

```bash
docker compose -f docker-compose.prod.yml build
docker compose -f docker-compose.prod.yml up -d
```

## Key Rules

1. **CSS Modules only** — no Tailwind, no inline styles
2. **TypeScript strict** — no `any`
3. **Server Components by default** — `'use client'` only for interactive
4. **Python: uv only** — pip is FORBIDDEN
5. **Swappable methods** — all head generation goes through HeadGenerationMethod protocol
6. **Client-side animation** — ARKit blendshapes run in browser, no server roundtrip
7. **Port 4001** — frontend dev and production
8. **Port 8000** — backend API
9. **Renderer patches** — setExpression + readPixels injected via `scripts/build-renderer.sh`

## Design System

- Background: `#08080C` / Surface: `#0E0E14` / Elevated: `#16161F`
- Primary: `#7EB8F0` / Accent: `#C4A0F0`
- Fonts: Space Grotesk (heading/body) + JetBrains Mono (code)
