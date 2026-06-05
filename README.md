<div align="center">

<img src="apps/web/public/logo.svg" alt="Splattie" width="120" />

# Splattie

**Interactive rigged 3D Gaussian assets from one image.**

*Heads, bodies, and objects that follow the cursor, pose on hover, and run at 60fps in any browser.*

[![Live](https://img.shields.io/badge/live-splattie.app-7eb8f0)](https://splattie.app)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![npm](https://img.shields.io/npm/v/@afromero/splattie-widget?color=blue)](https://www.npmjs.com/package/@afromero/splattie-widget)
[![CI](https://github.com/affromero/splattie/actions/workflows/ci.yml/badge.svg)](https://github.com/affromero/splattie/actions/workflows/ci.yml)
[![Next.js](https://img.shields.io/badge/Next.js-15-black?logo=next.js)](https://nextjs.org)
[![TypeScript](https://img.shields.io/badge/TypeScript-strict-blue?logo=typescript&logoColor=white)](https://typescriptlang.org)
[![Spark](https://img.shields.io/badge/Spark_2.0-MIT-green)](https://github.com/sparkjsdev/spark)
[![Three.js](https://img.shields.io/badge/Three.js-r170-black?logo=three.js)](https://threejs.org)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://python.org)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

</div>

---

<p align="center">
  <img src="docs/assets/demo.gif" alt="Splattie Demo" width="600" />
</p>

Splattie turns one image into an **interactive rigged 3D Gaussian Splatting asset** — a **head**, **full body**, or **object** — that lives on your website. Eyes follow the cursor, heads and torsos turn toward visitors, bodies add cursor-driven arm IK, and objects expose skeleton handles for pose editing. Hover and click trigger smooth state transitions. Rendered client-side. One file, one tag.

```html
<splattie-widget src="asset.splattie"></splattie-widget>
<script src="https://unpkg.com/@afromero/splattie-widget"></script>
```

This repo contains the **landing page**, the **GPU generation pipeline**, and the **`.splattie` format spec**. The web component itself lives in a separate repo: [**affromero/splattie-widget**](https://github.com/affromero/splattie-widget) (published as [`@afromero/splattie-widget`](https://www.npmjs.com/package/@afromero/splattie-widget) on npm). If you only want to embed a `.splattie` asset on your site, you don't need this repo.

<details>
<summary>Contents</summary>

- [Why](#why)
- [Try it](#try-it)
- [Run it locally](#run-it-locally)
  - [Frontend only](#frontend-only-no-gpu-1-minute)
  - [Self-host the full app](#self-host-the-full-app-with-gpu)
  - [Generate `.splattie` files from the CLI](#generate-splattie-files-from-the-cli-gpu)
  - [GPU runtime and memory](#gpu-runtime-and-memory)
  - [Widget development](#widget-development)
- [The `.splattie` format](#the-splattie-format)
- [Architecture](#architecture)
- [API](#api)
- [Contributing](#contributing)
- [Acknowledgements](#acknowledgements)
- [License](#license)

</details>

## Why

Spark renders splats. SuperSplat edits them. StorySplat hosts them. **Nothing makes them react.** Splattie is the interaction layer - a portable `.splattie` bundle plus a web component that reads it.

## Try it

- **Hosted** - [splattie.app](https://splattie.app) - click any of the 48 demo assets (12 heads + 12 bodies + 12 objects + 12 animals), play with the sliders, download the customised `.splattie`.
- **Embed the widget** - `npm install @afromero/splattie-widget` and drop the tag on your page. The 48 demo `.splattie` files in [`apps/web/public/demos/`](apps/web/public/demos) are **AI-generated (synthetic)** — no real people, no attribution required.
- **Self-host** with your own GPU - see below.

## Run it locally

### Frontend only (no GPU, 1 minute)

```bash
git clone https://github.com/affromero/splattie.git
cd splattie
git submodule update --init packages/splattie-widget
npm install
npm run dev
```

Open [http://localhost:4001](http://localhost:4001). All 48 demo assets (heads + bodies + objects + animals) work, the state editor works, downloads work. No backend needed.

### Self-host the full app (with GPU)

The hosted [splattie.app](https://splattie.app) doesn't expose the upload flow - that needs a GPU. If you have one, set `NEXT_PUBLIC_SELF_HOST=true` and the **Create** route lights up:

```bash
cp .env.example .env.local
# edit .env.local:
#   NEXT_PUBLIC_SELF_HOST=true
#   NEXT_PUBLIC_API_URL=http://localhost:8000

# install GPU deps (CUDA 12.x, ~20 min — downloads LAM head, LHM body,
# TRELLIS + TripoSplat reconstruction, Puppeteer rigging, and the
# SMAL + SuperAnimal quadruped runtime)
cd backend
bash scripts/setup-gpu.sh

# run the backend
uv run uvicorn splattie.api.app:create_app --factory --reload --port 8000

# in another shell, run the frontend
cd ../
npm run dev
```

Docker self-host path:

```bash
git submodule update --init --recursive packages/splattie-widget backend/vendor/LAM backend/vendor/LHM backend/vendor/TRELLIS backend/vendor/TripoSplat backend/vendor/Puppeteer
ADMIN_PASSWORD=change-me SESSION_SECRET=change-me ADMIN_API_TOKEN=change-me \
  docker compose -f deploy/docker-compose.gpu.yml up -d --build
```

Now [http://localhost:4001/create](http://localhost:4001/create) accepts images and generates a 3D **head** (LAM), **body** (LHM), **object** (TRELLIS + Puppeteer), or **quadruped mammal** (TripoSplat/TRELLIS + SuperAnimal-anchored SMAL, with cursor-follow head tracking), then routes you to the state editor where you can tune the interactions and download the `.splattie`. No CLI required.

### Generate `.splattie` files from the CLI (GPU)

If you have shell access on a GPU machine and only want output files, use the backend batch CLI directly. Run this after the GPU setup step above; the command loads the selected generator once and writes one `.splattie` per input image.

```bash
cd backend
bash scripts/setup-gpu.sh

mkdir -p /tmp/splattie-out
uv run splattie generate-splattie-batch \
  --images-dir /path/to/source-images \
  --output-dir /tmp/splattie-out \
  --asset-type head
```

Use `--asset-type body` for LHM/SMPL-X bodies, `--asset-type object` for TRELLIS + Puppeteer rigged objects, or `--asset-type quadruped_mammal` for SMAL-rigged animals with cursor-follow head tracking. Input images can be `.jpg`, `.jpeg`, or `.png`; outputs are named after the input stems.

The quadruped pipeline reconstructs the gaussian splat with **TripoSplat** by default (VAST-AI; reconstructs animal muzzles/faces far more cleanly than TRELLIS) and fits a **SMAL** skeleton anchored to **SuperAnimal-Quadruped** keypoints. Set `SPLATTIE_QUADRUPED_BACKEND=trellis` to use TRELLIS reconstruction instead. SMAL weights are license-gated (MPI, non-commercial) and download separately — see the setup script.

If you prefer Docker on the GPU host:

```bash
git submodule update --init --recursive packages/splattie-widget backend/vendor/LAM backend/vendor/LHM backend/vendor/TRELLIS backend/vendor/TripoSplat backend/vendor/Puppeteer
ADMIN_PASSWORD=change-me SESSION_SECRET=change-me ADMIN_API_TOKEN=change-me \
  docker compose -f deploy/docker-compose.gpu.yml build gpu-backend

docker run --rm --gpus all --shm-size 16g \
  -v /path/to/source-images:/inputs:ro \
  -v /tmp/splattie-out:/outputs \
  deploy-gpu-backend:latest \
  uv run --no-sync splattie generate-splattie-batch \
    --images-dir /inputs \
    --output-dir /outputs \
    --asset-type object
```

### GPU runtime and memory

Measured on an NVIDIA H100 80GB HBM3 from the uv-managed `backend/.venv`, using one demo image and the cold CLI command above after weights were already downloaded (head/body/object on June 1, 2026; quadruped on June 5, 2026). Peak GPU memory is the highest `nvidia-smi` sample during the run; it includes framework allocator reservations and should not be read as a hard minimum.

| Asset type | Pipeline | Input | Cold CLI time | Peak GPU memory | Output bundle |
|------------|----------|-------|---------------|-----------------|---------------|
| Head | LAM-20K + FLAME tracking | `h1.jpg` | 98 s | 12.0 GiB (12,240 MiB) | 1.7 MiB |
| Body | LHM-500M + Multi-HMR | `b1.jpg` | 100 s | 15.1 GiB (15,430 MiB) | 2.4 MiB |
| Object | TRELLIS-image-large + Puppeteer | `o1.jpg` | 118 s | 70.6 GiB (72,324 MiB) | 16.6 MiB |
| Quadruped mammal | TripoSplat + SuperAnimal-anchored SMAL | `q1.jpg` | 115 s | 7.5 GiB (7,677 MiB) | 11.6 MiB |

The API server and batch CLI load each method once and reuse it for multiple images, so multi-image batches amortize model startup. Different GPUs, drivers, PyTorch allocator behavior, input complexity, and TRELLIS output density can move both time and peak memory.

### Widget development

```bash
cd packages/splattie-widget
npm run dev
```

Open [http://localhost:4002](http://localhost:4002). Hot-reload for the widget itself, sliders for all five state dimensions, drag-drop any `.splattie` to load it.

## The `.splattie` format

A ZIP bundle with a required `manifest.json` declaring every asset and locking the format version to the widget version. Full spec: [FORMAT.md in the widget repo](https://github.com/affromero/splattie-widget/blob/main/FORMAT.md).

```
asset.splattie
├── manifest.json             # required - declares assets + assetType (head/body/object) + formatVersion
├── *.ply or *.spz            # required - Gaussian splats
│
│  # head (assetType: head) — FLAME rig:
├── bone_tree.json            # optional - skeleton (FLAME 5 bones)
├── lbs_weight_20k.json       # optional - per-splat skinning weights
├── expression_basis.bin      # optional - FLAME PCA blendshape basis
│
│  # body (assetType: body) — SMPL-X rig:
├── skeleton.json             # optional - skeleton (SMPL-X 55 joints, baked-pose rest)
├── lbs_weights.json          # optional - per-gaussian sparse LBS weights
│
│  # object (assetType: object) — arbitrary skeleton rig:
├── skeleton.json             # optional - object joint hierarchy + rest positions
├── lbs_weights.bin           # optional - binary sparse per-gaussian LBS weights
│
└── states.json               # optional - idle/hover/click definitions
```

Each state defines all five interaction dimensions: **ghost** (floating motion), **expression** (FLAME blendshapes + bones for heads; rig sliders for bodies/objects), **camera** (spherical position), **rotation** (asset pitch/yaw/roll), **tracking** (cursor-follow intensity per asset type). Heads use FLAME SplatSkinning, bodies use SMPL-X linear blend skinning plus two-bone arm IK, and objects use arbitrary skeleton LBS with drag-to-pose terminal handles. The format is one bundle either way — the widget branches on `assetType`.

## Architecture

```
splattie/
├── apps/web/                       # Next.js 15 landing + editor (port 4001)
│   └── src/app/                    # /, /create, /view/[id]
├── packages/splattie-widget/       # <splattie-widget> web component (MIT)
│   ├── src/                        # SplatWidget, StateMachine, dimensions (look-at, IK)
│   └── FORMAT.md                   # .splattie format spec
├── backend/                        # FastAPI GPU service (port 8000)
│   ├── src/splattie/methods/lam/   # LAM head generation (FLAME)
│   ├── src/splattie/methods/lhm/   # LHM body generation (SMPL-X)
│   ├── src/splattie/methods/object/ # TRELLIS + Puppeteer object rig generation
│   ├── src/splattie/methods/quadruped_mammal/ # TripoSplat/TRELLIS + SuperAnimal-anchored SMAL
│   ├── scripts/setup-gpu.sh        # CUDA + LAM/LHM/TRELLIS/TripoSplat/Puppeteer/SMAL setup
│   ├── src/splattie/cli/demos.py   # Gemini demo image generation + demo install
│   ├── vendor/LAM/                 # LAM submodule (SIGGRAPH 2025)
│   ├── vendor/LHM/                 # LHM submodule (SIGGRAPH 2025)
│   ├── vendor/TRELLIS/             # TRELLIS submodule (MIT)
│   ├── vendor/TripoSplat/          # TripoSplat submodule (VAST-AI; optional animal backend)
│   ├── vendor/Puppeteer/           # Puppeteer submodule (Apache 2.0)
│   └── vendor/SMAL/                # SMAL parametric quadruped (MPI non-commercial; gitignored weights)
├── deploy/                         # Compose recipes + Caddy fragment
│   ├── docker-compose.dev.yml      # Local Docker stack
│   ├── docker-compose.prod.yml     # CPU serving stack for splattie.app
│   ├── docker-compose.gpu.yml      # Self-host GPU stack
│   └── Caddyfile                   # Reverse proxy fragment for splattie.app
└── docs/assets/demo.gif            # README demo media
```

| Component | Technology |
|-----------|-----------|
| Frontend | Next.js 15, TypeScript strict, CSS Modules |
| Rendering | Spark 2.0 (World Labs, MIT) + Three.js |
| Animation (heads) | FLAME SplatSkinning (dual quaternion) + PCA blendshapes |
| Animation (bodies) | SMPL-X linear blend skinning + head/torso look-at + two-bone arm IK |
| Animation (objects) | Arbitrary skeleton LBS + root/joint follow + drag-to-pose skeleton handles |
| Animation (quadrupeds) | SMAL linear blend skinning + cursor-follow head/neck tracking (gaze) |
| Backend | FastAPI, Python 3.11, uv |
| Asset generation | LAM (heads, FLAME) + LHM (bodies, SMPL-X) + TRELLIS/Puppeteer (objects) + TripoSplat/SMAL (quadruped mammals); swappable via the `AssetGenerationMethod` protocol (`asset_type`: head/body/object/quadruped_mammal) |
| Format | ZIP with `manifest.json`, version-locked to the widget |

## API

The widget exposes a simple custom-element API:

```html
<splattie-widget
  src="asset.splattie"
  background="#0e0e14"
  width="100%"
  height="400px"
></splattie-widget>
```

```javascript
widget.addEventListener('splatload',  () => {});
widget.addEventListener('splathover', () => {});
widget.addEventListener('splatclick', () => {});
widget.setState('hover');
```

React wrapper available via `@afromero/splattie-widget/react`. Full reference in the [widget README](https://github.com/affromero/splattie-widget#readme).

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Acknowledgements

Splattie builds on outstanding open-source research:

- **[LAM](https://github.com/aigc3d/LAM)** (SIGGRAPH 2025) - Large Avatar Model for single-image 3DGS **head** generation. By Zixuan Zeng et al., AIGC3D.
- **[LHM](https://github.com/aigc3d/LHM)** (SIGGRAPH 2025) - Large Animatable Human Model for single-image 3DGS **body** generation. By AIGC3D.
- **[TRELLIS](https://github.com/microsoft/TRELLIS)** - single-image 3D asset reconstruction. By Microsoft.
- **[TripoSplat](https://github.com/VAST-AI-Research/TripoSplat)** - single-image image-to-3D-gaussian reconstruction (default animal backend; cleaner muzzles/faces than TRELLIS). By VAST-AI Research.
- **[Puppeteer](https://github.com/snap-research/Puppeteer)** - automatic skeleton and skinning for generated 3D assets. By Snap Research.
- **[SMAL](https://smal.is.tue.mpg.de/)** - Skinned Multi-Animal Linear model; parametric quadruped skeleton + shape (quadruped mammals). By Zuffi, Kanazawa, Jacobs, Black (MPI).
- **[SuperAnimal-Quadruped / DeepLabCut](https://github.com/DeepLabCut/DeepLabCut)** - foundation animal pose estimation; supplies the keypoints the SMAL fit is anchored to. By Mathis lab et al.
- **[FLAME](https://flame.is.tue.mpg.de/)** - 3D face shape, expression, and pose model (heads). By Tianye Li, Timo Bolkart, Michael J. Black, Hao Li, Javier Romero.
- **[SMPL-X](https://smpl-x.is.tue.mpg.de/)** - Expressive 3D body model (bodies). By Pavlakos, Choutas, Ghorbani, Bolkart, Osman, Tzionas, Black (MPI).
- **[Spark 2.0](https://github.com/sparkjsdev/spark)** - MIT-licensed 3DGS renderer for Three.js, by World Labs.
- **[3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)** - Real-time radiance field rendering. Kerbl, Kopanas, Leimkühler, Drettakis (INRIA).
- Demo assets are **AI-generated (synthetic)** — they are not real people.

## License

The Splattie source code is **MIT-licensed** and commercial-use safe:

- the `<splattie-widget>` web component (`@afromero/splattie-widget` on npm)
- the `.splattie` format and `manifest.json` schema
- the web app (`apps/web/`)

The reference GPU pipeline (`backend/`) wraps **LAM** (Apache 2.0), **LHM** (Apache 2.0), **TRELLIS** (MIT), **TripoSplat** (VAST-AI), **Puppeteer** (Apache 2.0), and **gsplat** (Apache 2.0) on top of parametric models — **FLAME** (heads), **SMPL-X** (bodies), and **SMAL** (quadruped mammals). The non-commercial pieces are those three parametric models (FLAME, SMPL-X, SMAL — all MPI) plus SuperAnimal-Quadruped weights; the object path is MIT / Apache 2.0. The widget itself does not require any of this at runtime - it only needs a valid `.splattie` file.

**Paths to commercial use** (see [`NOTICE`](NOTICE) for the full breakdown):

1. **Widget-only** - use the widget freely; generate `.splattie` files through your own pipeline.
2. **License FLAME / SMPL-X / SMAL** - contact [MPI for Intelligent Systems](https://www.is.mpg.de) for commercial FLAME (heads), SMPL-X (bodies), and SMAL (quadruped mammals) terms. The rest of the stack is already Apache 2.0 / MIT.
3. **Object-only path** - use TRELLIS + Puppeteer-generated object `.splattie` files without FLAME, SMPL-X, or SMAL.
4. **Drop-in replacement** - implement an alternative asset-generation method behind the `AssetGenerationMethod` protocol in [`backend/src/splattie/methods/`](backend/src/splattie/methods/), declaring its `asset_type` (head/body/object/quadruped_mammal). The format is method-agnostic.
