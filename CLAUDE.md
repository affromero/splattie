# CLAUDE.md — Splattie

> Generate interactive 3D avatars and objects for the web.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Next.js 15 App Router, TypeScript strict, CSS Modules |
| 3D Rendering | LAM_WebRender (gaussian-splat-renderer-for-lam), WebGL |
| Backend (GPU) | Python 3.11, FastAPI, uv + pyproject.toml |
| Asset Generation | LAM (SIGGRAPH 2025) heads, LHM/SMPL-X bodies, and TRELLIS + Puppeteer objects — swappable via AssetGenerationMethod protocol (`asset_type`: head/body/object) |
| Animation | ARKit blendshapes via FLAME mesh (client-side, 60fps) |
| Face Detection | readPixels after WebGL render (pixel-perfect) |
| Compression | @playcanvas/splat-transform (PLY → SPZ) |
| Deploy | Docker (CUDA) + Caddy + GitHub Actions SSH |

## Monorepo

- `apps/web/` — Next.js frontend (npm workspace)
- `backend/` — FastAPI GPU service (Python, NOT npm workspace)
- `packages/lam-renderer/` — LAM_WebRender submodule (built with Vite)
- `backend/vendor/LAM/` — LAM head-generation submodule
- `backend/vendor/LHM/` — LHM body-generation submodule
- `backend/vendor/TRELLIS/` — TRELLIS image-to-3D reconstruction submodule
- `backend/vendor/Puppeteer/` — Puppeteer skeleton + skinning submodule

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
bash scripts/setup-gpu.sh   # torch, CUDA extensions, LAM/LHM/TRELLIS/Puppeteer weights
```

### Docker

```bash
docker compose -f deploy/docker-compose.prod.yml build
docker compose -f deploy/docker-compose.prod.yml up -d
docker compose -f deploy/docker-compose.gpu.yml up -d --build  # self-host GPU stack
```

## Key Rules

1. **CSS Modules only** — no Tailwind, no inline styles
2. **TypeScript strict** — no `any`
3. **Server Components by default** — `'use client'` only for interactive
4. **Python: uv only** — pip is FORBIDDEN
5. **Swappable methods** — all asset generation goes through the `AssetGenerationMethod` protocol with an `asset_type` discriminator (head/body/object)
6. **Client-side animation** — ARKit blendshapes run in browser, no server roundtrip
7. **Port 4001** — frontend dev and production
8. **Port 8000** — backend API
9. **Renderer patches** — setExpression + readPixels injected via `scripts/build-renderer.sh`
10. **Python schemas** — first-party Python must not use `dict` typing; use Pydantic dataclasses/models for known payloads and precise non-dict types elsewhere. `Mapping`/`MutableMapping` are only acceptable for open-ended JSON or infrastructure plumbing. Stdlib `@dataclass` is forbidden.

## Design System

- Background: `#08080C` / Surface: `#0E0E14` / Elevated: `#16161F`
- Primary: `#7EB8F0` / Accent: `#C4A0F0`
- Fonts: Space Grotesk (heading/body) + JetBrains Mono (code)
