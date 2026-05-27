<div align="center">

<img src="apps/web/public/logo.svg" alt="Mirada" width="120" />

# Mirada

**Upload a photo. Get a 3D head whose eyes follow you.**

*Powered by LAM (SIGGRAPH 2025) + 3D Gaussian Splatting + FLAME animation*

[![Stage](https://img.shields.io/badge/stage-prototype-orange)]()
[![License](https://img.shields.io/badge/license-private-lightgrey)]()
[![CI](https://github.com/affromero/mirada/actions/workflows/ci.yml/badge.svg)](https://github.com/affromero/mirada/actions/workflows/ci.yml)
[![Next.js](https://img.shields.io/badge/Next.js-15-black?logo=next.js)](https://nextjs.org)
[![TypeScript](https://img.shields.io/badge/TypeScript-strict-blue?logo=typescript&logoColor=white)](https://typescriptlang.org)
[![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)](https://python.org)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Docker](https://img.shields.io/badge/Docker-GPU-2496ED?logo=docker&logoColor=white)](https://docker.com)

</div>

---

Mirada turns a single photograph into an interactive 3D gaussian splatting head. The eyes follow your cursor, the face blinks naturally, and reacts when you hover over it — all rendered client-side at 60fps.

## How It Works

1. **Upload** a photo with a visible face
2. **Generate** a 3D head on the GPU backend using LAM (~30s on H100)
3. **View** the interactive head — eyes follow your cursor, face reacts to hover

## Quick Start

### Frontend (any machine)

```bash
npm install
npm run dev
```

Open [http://localhost:4001](http://localhost:4001).

### Backend (GPU server)

```bash
cd backend
bash scripts/setup-gpu.sh    # installs torch, LAM, CUDA extensions, model weights
uv run uvicorn mirada.api.app:create_app --factory --port 8000
```

### Docker (GPU server)

```bash
docker compose -f docker-compose.prod.yml build
docker compose -f docker-compose.prod.yml up -d
```

## Architecture

```
mirada/
├── apps/web/                  # Next.js 15 frontend (port 4001)
├── backend/                   # FastAPI GPU service (port 8000)
│   ├── src/mirada/            # API routes, LAM method, segmentation
│   ├── vendor/LAM/            # LAM submodule (SIGGRAPH 2025)
│   └── scripts/setup-gpu.sh   # GPU environment setup
├── packages/lam-renderer/     # LAM_WebRender submodule (3DGS viewer)
├── scripts/build-renderer.sh  # Build + patch the renderer
├── Dockerfile.backend         # GPU Docker image (~15GB)
├── Dockerfile.web             # Frontend Docker image (~100MB)
└── Caddyfile                  # Reverse proxy config
```

| Component | Technology |
|-----------|-----------|
| Frontend | Next.js 15, TypeScript strict, CSS Modules |
| 3D Rendering | LAM_WebRender (gaussian-splat-renderer-for-lam) |
| Backend | FastAPI, Python 3.10, uv |
| Head Generation | LAM (SIGGRAPH 2025) — swappable via HeadGenerationMethod protocol |
| Animation | ARKit blendshapes via FLAME mesh (client-side, 60fps) |
| Face Detection | readPixels after WebGL render (pixel-perfect) |
| Compression | @playcanvas/splat-transform (PLY → SPZ, ~5x) |
| Deploy | Docker + Caddy + GitHub Actions SSH |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `POST /generate-from-upload` | multipart | Upload photo → LAM → ZIP bundle |
| `POST /segment` | multipart | Server-side SAM 3 segmentation |
| `POST /generate` | SSE | Generate from image/mask URLs with progress |
| `GET /models` | JSON | List available generation methods |
| `GET /health` | JSON | Service status, GPU info, model loaded |
| `GET /storage/{id}/{file}` | static | Serve generated assets |

## Development

```bash
# Frontend
npm run ci                     # lint + type-check + test + build

# Backend
cd backend
uv sync                        # install Python deps
uv run pytest                  # run tests
uv run ruff check src/ tests/  # lint

# Renderer (rebuild after changes to packages/lam-renderer/src/)
npm run build:renderer
```

## Environment

Copy `.env.example` to `.env.local`:

```bash
cp .env.example .env.local
```

Set `NEXT_PUBLIC_API_URL` to your GPU backend URL.
