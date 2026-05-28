<div align="center">

<img src="apps/web/public/logo.svg" alt="Splattie" width="120" />

# Splattie

**Interactive 3D avatars from a single photo.**

*Eyes follow visitors. Head reacts to hover. Runs at 60fps in any browser.*

[![Live](https://img.shields.io/badge/live-splattie.app-7eb8f0)](https://splattie.app)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![npm](https://img.shields.io/npm/v/@afromero/splattie-widget?color=blue)](https://www.npmjs.com/package/@afromero/splattie-widget)
[![CI](https://github.com/affromero/splattie/actions/workflows/ci.yml/badge.svg)](https://github.com/affromero/splattie/actions/workflows/ci.yml)
[![Next.js](https://img.shields.io/badge/Next.js-15-black?logo=next.js)](https://nextjs.org)
[![TypeScript](https://img.shields.io/badge/TypeScript-strict-blue?logo=typescript&logoColor=white)](https://typescriptlang.org)
[![Spark](https://img.shields.io/badge/Spark_2.0-MIT-green)](https://github.com/sparkjsdev/spark)
[![Three.js](https://img.shields.io/badge/Three.js-r170-black?logo=three.js)](https://threejs.org)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

</div>

---

<p align="center">
  <img src="demo.gif" alt="Splattie Demo" width="600" />
</p>

Splattie turns a single photograph into an **interactive 3D Gaussian Splatting head** that lives on your website. Eyes follow the cursor, the face blinks naturally, hover and click trigger smooth state transitions. Rendered client-side. One file, one tag.

```html
<splattie-widget src="avatar.splattie"></splattie-widget>
<script src="https://unpkg.com/@afromero/splattie-widget"></script>
```

This repo contains the **landing page**, the **GPU generation pipeline**, and the **`.splattie` format spec**. The web component itself lives in a separate repo: [**affromero/splattie-widget**](https://github.com/affromero/splattie-widget) (published as [`@afromero/splattie-widget`](https://www.npmjs.com/package/@afromero/splattie-widget) on npm). If you only want to embed an avatar on your site, you don't need this repo.

## Why

Spark renders splats. SuperSplat edits them. StorySplat hosts them. **Nothing makes them react.** Splattie is the interaction layer - a portable `.splattie` bundle plus a web component that reads it.

## Try it

- **Hosted** - [splattie.app](https://splattie.app) - click any of the 6 demo portraits, play with the sliders, download the customised `.splattie`.
- **Embed the widget** - `npm install @afromero/splattie-widget` and drop the tag on your page. The 6 demo `.splattie` files in [`apps/web/public/demos/`](apps/web/public/demos) are MIT-friendly (Pexels-licensed portraits).
- **Self-host** with your own GPU - see below.

## Run it locally

### Frontend only (no GPU, 1 minute)

```bash
git clone https://github.com/affromero/splattie.git
cd splattie
git submodule update --init
npm install
npm run dev
```

Open [http://localhost:4001](http://localhost:4001). All 6 demo avatars work, the state editor works, downloads work. No backend needed.

### Self-host the full app (with GPU)

The hosted [splattie.app](https://splattie.app) doesn't expose the upload flow - that needs a GPU. If you have one, set `NEXT_PUBLIC_SELF_HOST=true` and the **Create** route lights up:

```bash
cp .env.example .env.local
# edit .env.local:
#   NEXT_PUBLIC_SELF_HOST=true
#   NEXT_PUBLIC_API_URL=http://localhost:8000

# install GPU deps (CUDA 12.x, ~20 min — downloads LAM weights)
cd backend
bash scripts/setup-gpu.sh

# run the backend
uv run uvicorn splattie.api.app:create_app --factory --reload --port 8000

# in another shell, run the frontend
cd ../
npm run dev
```

Now [http://localhost:4001/create](http://localhost:4001/create) accepts photos, generates a 3D head in ~30 s on an H100, and routes you to the state editor where you can tune the interactions and download the `.splattie`. No CLI required.

### Widget development

```bash
cd packages/splattie-widget
npm run dev
```

Open [http://localhost:4002](http://localhost:4002). Hot-reload for the widget itself, sliders for all five state dimensions, drag-drop any `.splattie` to load it.

## The `.splattie` format

A ZIP bundle with a required `manifest.json` declaring every asset and locking the format version to the widget version. Full spec: [FORMAT.md in the widget repo](https://github.com/affromero/splattie-widget/blob/main/FORMAT.md).

```
avatar.splattie
├── manifest.json             # required - declares assets + formatVersion
├── *.ply or *.spz            # required - Gaussian splats
├── bone_tree.json            # optional - skeleton (FLAME 5 bones)
├── lbs_weight_20k.json       # optional - per-splat skinning weights
├── expression_basis.bin      # optional - FLAME PCA blendshape basis
└── states.json               # optional - idle/hover/click definitions
```

Each state defines all five interaction dimensions: **ghost** (floating motion), **expression** (FLAME blendshapes + bones), **camera** (spherical position), **rotation** (object pitch/yaw/roll), **tracking** (cursor-follow intensity per bone).

## Architecture

```
splattie/
├── apps/web/                       # Next.js 15 landing + editor (port 4001)
│   └── src/app/                    # /, /create, /view/[id]
├── packages/splattie-widget/       # <splattie-widget> web component (MIT)
│   ├── src/                        # SplatWidget, StateMachine, dimensions
│   └── FORMAT.md                   # .splattie format spec
├── backend/                        # FastAPI GPU service (port 8000)
│   ├── src/splattie/methods/lam/   # LAM head generation
│   ├── scripts/setup-gpu.sh        # CUDA + LAM weights setup
│   ├── scripts/generate_splattie_batch.py  # CLI batch generation
│   └── vendor/LAM/                 # LAM submodule (SIGGRAPH 2025)
├── Dockerfile.backend              # GPU image
├── apps/web/Dockerfile             # Web image (port 4001)
└── deploy/Caddyfile                # Reverse proxy fragment for splattie.app
```

| Component | Technology |
|-----------|-----------|
| Frontend | Next.js 15, TypeScript strict, CSS Modules |
| Rendering | Spark 2.0 (World Labs, MIT) + Three.js |
| Animation | FLAME SplatSkinning (dual quaternion) + PCA blendshapes |
| Backend | FastAPI, Python 3.10, uv |
| Head generation | LAM (SIGGRAPH 2025) - swappable via the `HeadGenerationMethod` protocol |
| Format | ZIP with `manifest.json`, version-locked to the widget |

## API

The widget exposes a simple custom-element API:

```html
<splattie-widget
  src="avatar.splattie"
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

- **[LAM](https://github.com/aigc3d/LAM)** (SIGGRAPH 2025) - Large Avatar Model for single-image 3DGS head generation. By Zixuan Zeng et al., AIGC3D.
- **[FLAME](https://flame.is.tue.mpg.de/)** - 3D face shape, expression, and pose model. By Tianye Li, Timo Bolkart, Michael J. Black, Hao Li, Javier Romero.
- **[Spark 2.0](https://github.com/sparkjsdev/spark)** - MIT-licensed 3DGS renderer for Three.js, by World Labs.
- **[3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)** - Real-time radiance field rendering. Kerbl, Kopanas, Leimkühler, Drettakis (INRIA).
- Demo portraits from **[Pexels](https://www.pexels.com)** (free license, attribution on the landing page).

## License

The Splattie source code is **MIT-licensed** and commercial-use safe:

- the `<splattie-widget>` web component (`@afromero/splattie-widget` on npm)
- the `.splattie` format and `manifest.json` schema
- the web app (`apps/web/`)

The reference GPU pipeline (`backend/`) wraps **LAM** (Apache 2.0) + **gsplat** (Apache 2.0) + **FLAME**. The only non-commercial piece is FLAME's face model. The widget itself does not require any of this at runtime - it only needs a valid `.splattie` file.

**Three paths to commercial use** (see [`NOTICE`](NOTICE) for the full breakdown):

1. **Widget-only** - use the widget freely; generate `.splattie` files through your own pipeline.
2. **License FLAME** - contact [MPI for Intelligent Systems](https://flame.is.tue.mpg.de) for commercial FLAME terms. The rest of the stack is already Apache 2.0 / MIT.
3. **Drop-in replacement** - implement an alternative head-generation method behind the `HeadGenerationMethod` protocol in [`backend/src/splattie/methods/`](backend/src/splattie/methods/). The format is method-agnostic.
