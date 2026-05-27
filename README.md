<div align="center">

<img src="apps/web/public/logo.svg" alt="Splattie" width="120" />

# Splattie

**Upload a photo. Get a 3D head whose eyes follow you.**

*Powered by LAM (SIGGRAPH 2025) + 3D Gaussian Splatting + FLAME animation*

[![Stage](https://img.shields.io/badge/stage-prototype-orange)]()
[![License](https://img.shields.io/badge/license-private-lightgrey)]()
[![CI](https://github.com/affromero/splattie/actions/workflows/ci.yml/badge.svg)](https://github.com/affromero/splattie/actions/workflows/ci.yml)
[![Next.js](https://img.shields.io/badge/Next.js-15-black?logo=next.js)](https://nextjs.org)
[![TypeScript](https://img.shields.io/badge/TypeScript-strict-blue?logo=typescript&logoColor=white)](https://typescriptlang.org)
[![Spark](https://img.shields.io/badge/Spark_2.0-MIT-green?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgaGVpZ2h0PSIxNiI+PGNpcmNsZSBjeD0iOCIgY3k9IjgiIHI9IjYiIGZpbGw9IiNmZmYiLz48L3N2Zz4=)](https://github.com/sparkjsdev/spark)
[![Three.js](https://img.shields.io/badge/Three.js-r170-black?logo=three.js)](https://threejs.org)
[![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)](https://python.org)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Docker](https://img.shields.io/badge/Docker-GPU-2496ED?logo=docker&logoColor=white)](https://docker.com)
[![Tests](https://img.shields.io/badge/tests-63_passing-brightgreen)]()

</div>

---

Splattie turns a single photograph into an interactive 3D gaussian splatting head. The eyes follow your cursor, the face blinks naturally, and reacts when you hover over it — all rendered client-side at 60fps.

## How It Works

1. **Upload** a photo with a visible face
2. **Generate** a 3D head on the GPU backend using LAM (~30s on H100)
3. **View** the interactive head — eyes follow your cursor, face reacts to hover

## `<splattie-widget>` — Interactive 3DGS as a Web Component

The core rendering is packaged as a standalone web component that makes gaussian splats reactive — like Rive or Lottie but for 3D.

```html
<splattie-widget
  src="avatar.ply"
  bones="bone_tree.json"
  weights="lbs_weight_20k.json"
  config="states.json"
  expression-basis="expression_basis.bin"
></splattie-widget>
```

### Five Dimensions of State

Each interaction state (idle, hover, click) defines all five:

| Dimension | What it controls | Implementation |
|-----------|-----------------|----------------|
| Ghost | Floating/bobbing motion | Sinusoidal mesh offset |
| Expression | FLAME blendshapes + bone rotations | SplatSkinning (DQ) + expression basis |
| Camera | Spherical position (theta/phi/radius) | Three.js PerspectiveCamera |
| Rotation | Object pitch/yaw/roll | mesh.rotation |
| Tracking | Cursor-following intensity (eyes, head) | Per-bone NDC projection |

### Expression System

Two layers work together:

- **SplatSkinning** (5 FLAME bones) — jaw open, neck pitch/yaw/roll, eye gaze, brow raise (virtual bones)
- **Expression basis** (10 FLAME PCA coefficients) — smile, lip shapes, jaw movement, deforming all 20K splats coherently via per-splat position offsets

### Visual State Editor

An on-canvas editor with live sliders for all dimensions. Design idle → hover → click states, export as `.splattie` (ZIP bundle: PLY + bones + weights + states.json + expression_basis.bin).

### `.splattie` Format

A single embeddable file (like `.lottie` for Lottie):

```
avatar.splattie (ZIP)
├── model.ply              # 20K gaussian splats
├── bone_tree.json         # 5 FLAME bones
├── lbs_weight_20k.json    # LBS weights per splat
├── expression_basis.bin   # FLAME blendshape basis (2.3 MB)
└── states.json            # Interaction state definitions
```

## Quick Start

### Frontend (any machine)

```bash
npm install
npm run dev
```

Open [http://localhost:4001](http://localhost:4001).

### Splat Widget Editor

```bash
cd packages/splattie-widget
npm run dev
```

Open [http://localhost:4002](http://localhost:4002).

### Backend (GPU server)

```bash
cd backend
bash scripts/setup-gpu.sh
uv run uvicorn splattie.api.app:create_app --factory --port 8000
```

### Export Expression Basis

```bash
cd backend
python3 scripts/export_expression_basis.py \
  --output ../packages/splattie-widget/public/expression_basis.bin \
  --num-expressions 10
```

## Architecture

```
splattie/
├── apps/web/                     # Next.js 15 frontend (port 4001)
├── backend/                      # FastAPI GPU service (port 8000)
│   ├── src/splattie/             # API routes, LAM method, segmentation
│   ├── vendor/LAM/               # LAM submodule (SIGGRAPH 2025, forked)
│   └── scripts/
│       ├── setup-gpu.sh          # GPU environment setup
│       └── export_expression_basis.py  # FLAME blendshape export
├── packages/splattie-widget/        # <splattie-widget> web component
│   ├── src/
│   │   ├── SplatWidget.ts        # Custom element, render loop
│   │   ├── renderer/SparkSetup.ts # Spark 2.0 + SplatSkinning init
│   │   ├── state/StateMachine.ts  # State transitions + interpolation
│   │   ├── features/
│   │   │   ├── ExpressionBasis.ts # FLAME blendshape per-splat deformation
│   │   │   └── AutoBlink.ts      # Natural random blinking
│   │   ├── dimensions/           # Ghost, Camera, Rotation, Tracking
│   │   └── interaction/          # CursorTracker, HitDetector, Events
│   ├── public/                   # Demo assets (PLY, bones, weights, basis)
│   └── tests/                    # 25 unit tests
├── Dockerfile.backend            # GPU Docker image (~15GB)
├── Dockerfile.web                # Frontend Docker image (~100MB)
└── Caddyfile                     # Reverse proxy config
```

| Component | Technology |
|-----------|-----------|
| Frontend | Next.js 15, TypeScript strict, CSS Modules |
| 3D Rendering | Spark 2.0 (MIT, by World Labs) + Three.js |
| Animation | SplatSkinning (FLAME bones, dual quaternion) |
| Blendshapes | FLAME expression basis (20K vertices, 10 PCA coefficients) |
| Backend | FastAPI, Python 3.10, uv |
| Head Generation | LAM (SIGGRAPH 2025) — swappable via HeadGenerationMethod protocol |
| Face Detection | readPixels after WebGL render (pixel-perfect) |
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

# Splat Widget
cd packages/splattie-widget
npm run dev                    # dev server on port 4002
npx vitest run                 # 25 tests

# Backend
cd backend
uv sync
uv run pytest                  # 13 tests
uv run ruff check src/ tests/
```

## Environment

Copy `.env.example` to `.env.local`:

```bash
cp .env.example .env.local
```

Set `NEXT_PUBLIC_API_URL` to your GPU backend URL.

## Submodules

Both submodules are forked under `affromero`:

| Submodule | Path | Fork |
|-----------|------|------|
| LAM | `backend/vendor/LAM` | [affromero/LAM](https://github.com/affromero/LAM) |

Model weights (`model_zoo/`) are not in git — download them into the submodule path on the GPU server.
