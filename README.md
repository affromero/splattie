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
- [Method landscape](#method-landscape)
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

The quadruped pipeline reconstructs the gaussian splat with **TripoSplat** by default (VAST-AI; reconstructs animal muzzles/faces far more cleanly than TRELLIS) and fits a **SMAL** skeleton anchored to **SuperAnimal-Quadruped** keypoints. The reconstruction model is a per-request choice (the `ReconstructBackend` enum): pass `--backend trellis` to the CLI, `?backend=trellis` to `/generate-from-upload`, or pick it from the **Reconstruction model** toggle on [`/create`](http://localhost:4001/create) when the Animal category is selected. SMAL weights are license-gated (MPI, non-commercial) and download separately — see the setup script.

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

Measured on an NVIDIA H100 80GB HBM3 from the uv-managed `backend/.venv`, using one demo image and the cold CLI command above after weights were already downloaded (head/body/object on June 1, 2026; quadruped on June 5, 2026). Peak GPU memory is the highest `nvidia-smi` sample during the run; it includes framework allocator reservations and should not be read as a hard minimum. Bundle sizes are the current checked-in demo files for the listed input. The splat count is `manifest.avatar.splat.numGaussians`, which is the main reason object and animal files stay heavier.

| Asset type | Pipeline | Input | Splats | Cold CLI time | Peak GPU memory | Demo bundle |
|------------|----------|-------|--------|---------------|-----------------|-------------|
| Head | LAM-20K + FLAME tracking | `h1.jpg` | 20,018 | 98 s | 12.0 GiB (12,240 MiB) | 1.7 MiB, raw PLY |
| Body | LHM-500M + Multi-HMR | `b1.jpg` | 40,000 | 100 s | 15.1 GiB (15,430 MiB) | 2.4 MiB, raw PLY |
| Object | TRELLIS-image-large + Puppeteer | `o1.jpg` | 336,736 | 118 s | 70.6 GiB (72,324 MiB) | 5.1 MiB, rig-aware compressed PLY |
| Quadruped mammal | TripoSplat + SuperAnimal-anchored SMAL | `q1.jpg` | 262,144 | 115 s | 7.5 GiB (7,677 MiB) | 5.2 MiB, rig-aware compressed PLY |

The API server and batch CLI load each method once and reuse it for multiple images, so multi-image batches amortize model startup. Different GPUs, drivers, PyTorch allocator behavior, input complexity, and TRELLIS output density can move both time and peak memory. Raw generated object/quadruped bundles are larger until post-processed with rig-aware PlayCanvas compressed PLY; that compression preserves rigging by permuting the per-splat LBS weights to the compressed splat order.

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

## Method landscape

Splattie's bar for a **drop-in** asset generator is narrow: **one image in → feed-forward (no per-subject training) → a 3D Gaussian Splatting asset rigged to a model the widget animates client-side (FLAME / SMPL-X / SMAL / arbitrary-skeleton LBS) → under a commercially-usable license.** Most published avatar/object work misses at least one axis — it wants multi-view or video capture, optimizes per subject, outputs a mesh or NeRF instead of splats, or is research-only. These tables track the popular/recent approaches against that bar so we don't re-evaluate the same papers twice. *Verified June 2026 — PRs welcome.*

**Legend** &nbsp;·&nbsp; ✅ baseline / clean swap &nbsp;·&nbsp; ⚠️ adaptable (needs a conversion or dependency swap) &nbsp;·&nbsp; ❌ wrong input, representation, or license &nbsp;·&nbsp; ⏳ no code released yet.
**Feed-fwd** ✅ = single forward pass, ❌ = per-subject optimization/training. The **License** column flags commercial use of the code **and** of every required weight/dataset/dependency: a permissive code license riding on a non-commercial model (FLAME / SMPL-X / SMAL / INRIA `diff-gaussian-rasterization`) is marked ❌. Splattie's own FLAME / SMPL-X / SMAL baselines are research-licensed too (see [License](#license) for commercial paths); the `<splattie-widget>` runtime needs none of them.

### Heads — vs. LAM (single image → FLAME-rigged 3DGS → ARKit client animation)

| Method | Input | Feed-fwd | Output | License (commercial) | Plug-and-play |
|--------|-------|----------|--------|----------------------|---------------|
| **[LAM](https://github.com/aigc3d/LAM)** · SIGGRAPH '25 — *baseline* | 1 image | ✅ | FLAME 3DGS | Apache 2.0 ✅ | ✅ **what Splattie uses** |
| [GAGAvatar](https://github.com/xg-chu/GAGAvatar) · NeurIPS '24 | 1 image | ✅ | feature-3DGS + bespoke neural renderer | MIT, FLAME dep ⚠️ | ❌ custom renderer + driving-image rig, not a static ARKit splat |
| [FlexAvatar — Kirschstein](https://github.com/tobias-kirschstein/flexavatar) · arXiv '25 | 1 image | ✅ | neural-decoder 3DGS | CC BY-NC ❌ | ❌ non-commercial; gaussians come from a live neural decoder |
| [FlexAvatar — Peng](https://arxiv.org/abs/2512.17717) · arXiv '25 | 1 image | ✅ | FLAME-UV 3DGS + LBS | unreleased ⏳ | ⏳ no code; NeRSemble-trained (NC) — architecturally the closest |
| [Portrait4D-v2](https://github.com/YuDeng/Portrait-4D) · ECCV '24 | 1 image | ✅ | tri-plane NeRF | MIT, FLAME dep ⚠️ | ❌ NeRF not splat; driven by a driving frame, not ARKit |
| [GPAvatar](https://github.com/xg-chu/GPAvatar) · ICLR '24 | few images | ✅ | tri-plane NeRF | MIT, FLAME/EMOCA NC ❌ | ❌ NeRF not splat; non-commercial deps |
| [HeadGAP](https://arxiv.org/abs/2408.06019) · 3DV '25 | few images | ❌ | FLAME 3DGS | unreleased ⏳ | ⏳ no code; ~5 min/identity fitting |
| [CAP4D](https://github.com/felixtaubner/cap4d) · CVPR '25 | few images | ❌ | FLAME-mesh 3DGS | CC BY-NC ❌ | ❌ per-subject training; non-commercial |
| [GaussianAvatars](https://github.com/ShenhanQian/GaussianAvatars) · CVPR '24 | multi-view | ❌ | FLAME-mesh 3DGS | CC BY-NC-SA + Toyota ❌ | ❌ 16-camera capture; non-commercial |
| [Gaussian Head Avatar](https://github.com/YuelangX/Gaussian-Head-Avatar) · CVPR '24 | multi-view | ❌ | MLP-deform 3DGS | research-only ❌ | ❌ 16-camera capture; non-commercial |
| [FlashAvatar](https://github.com/USTC3DV/FlashAvatar-code) · CVPR '24 | mono video | ❌ | FLAME-UV 3DGS | MIT, FLAME dep ❌ | ❌ video + per-subject training |
| [SplattingAvatar](https://github.com/initialneil/SplattingAvatar) · CVPR '24 | mono video | ❌ | mesh-bound 3DGS | CC BY-NC-SA ❌ | ❌ video + per-subject; non-commercial |
| [MonoGaussianAvatar](https://github.com/yufan1012/MonoGaussianAvatar) · SIGGRAPH '24 | mono video | ❌ | deform-field 3DGS | MIT, FLAME dep ❌ | ❌ video + per-subject training |
| [INSTA](https://github.com/Zielon/INSTA) · CVPR '23 | mono video | ❌ | NeRF (instant-ngp) | MPI research-only ❌ | ❌ video; NeRF not splat; non-commercial |
| [Relightable Gaussian Codec Avatars](https://github.com/facebookresearch/goliath) · CVPR '24 | multi-view dome | ❌ | relightable 3DGS | CC BY-NC ❌ | ❌ light-stage capture; non-commercial |

### Bodies — vs. LHM (single image → SMPL-X-rigged 3DGS → LBS + arm IK)

| Method | Input | Feed-fwd | Output | License (commercial) | Plug-and-play |
|--------|-------|----------|--------|----------------------|---------------|
| **[LHM](https://github.com/aigc3d/LHM)** · ICCV '25 — *baseline* | 1 image | ✅ | SMPL-X 3DGS | Apache 2.0, SMPL-X dep ✅† | ✅ **what Splattie uses** |
| [IDOL](https://github.com/yiyuzhuang/IDOL) · CVPR '25 | 1 image | ✅ | SMPL-X 3DGS | no LICENSE file + NC dataset ❌ | ❌ closest match, but NC weights/data + SMPL-X |
| [AniGS](https://github.com/aigc3d/AniGS) · CVPR '25 | 1 image | ❌ | SMPL-X 3DGS | code unreleased ⏳ | ⏳ no code; ~5 min/subject 4D-GS optimization |
| [Human3Diffusion](https://github.com/YuxuanSnow/Human3Diffusion) · NeurIPS '24 | 1 image | ✅ | unrigged 3DGS | MIT ⚠️ | ❌ template-free, no rig — the body can't be animated |
| [SiTH](https://github.com/SiTH-Diffusion/SiTH) · CVPR '24 | 1 image | ✅ | textured mesh | MIT, SMPL-X dep ❌ | ❌ mesh not splat; non-commercial dep |
| [PSHuman](https://github.com/pengHTYX/PSHuman) · CVPR '25 | 1 image | ✅ | textured mesh | MIT, SMPL-X dep ❌ | ❌ mesh not splat; non-commercial dep |
| [En3D](https://github.com/menyifang/En3D) · CVPR '24 | text / seed | ❌ | mesh + auto-rig (FBX) | Apache 2.0 ✅ | ❌ generator, not single-image reconstruction; mesh not splat |
| [GART](https://github.com/JiahuiLei/GART) · CVPR '24 | mono video | ❌ | SMPL 3DGS | MIT, SMPL/SMAL/INRIA NC ❌ | ❌ video + per-subject; non-commercial deps |
| [Animatable Gaussians](https://github.com/lizhe00/AnimatableGaussians) · CVPR '24 | multi-view video | ❌ | SMPL-X 3DGS maps | research-only ❌ | ❌ multi-view capture; non-commercial |
| [ExAvatar](https://github.com/mks0601/ExAvatar_RELEASE) · ECCV '24 | mono video | ❌ | SMPL-X mesh-bound 3DGS | MIT, SMPL-X dep ❌ | ❌ video + per-subject; non-commercial dep |
| [GaussianAvatar](https://github.com/aipixel/GaussianAvatar) · CVPR '24 | mono video | ❌ | SMPL 3DGS | MIT, SMPL dep ❌ | ❌ video + per-subject; non-commercial dep |
| [3DGS-Avatar](https://github.com/mikeqzy/3dgs-avatar-release) · CVPR '24 | mono video | ❌ | SMPL deform 3DGS | MIT, INRIA/SMPL NC ❌ | ❌ video + per-subject; non-commercial deps |
| [HUGS](https://github.com/apple/ml-hugs) · CVPR '24 | mono video | ❌ | SMPL 3DGS | Apple SCL, SMPL dep ❌ | ❌ video + per-subject; non-commercial dep |
| [AvatarPopUp](https://www.nikoskolot.com/avatarpopup/) · ECCV '24 | 1 image | ✅ | GHUM-rigged mesh | unreleased ⏳ | ⏳ no code; mesh + GHUM rig (not splat) |

### Objects — vs. TRELLIS → Puppeteer (single image → 3DGS → auto-skeleton + LBS)

Two stages: a **generator** (image → 3D) and a **rigger** (3D → skeleton + skinning). A generator must emit gaussians (a mesh needs a mesh→splat step); a rigger must put skinning weights on the splat points (mesh-vertex weights need a transfer step).

| Method | Role | Input | Output | License (commercial) | Plug-and-play |
|--------|------|-------|--------|----------------------|---------------|
| **[TRELLIS](https://github.com/microsoft/TRELLIS)** · CVPR '25 — *baseline* | generate | 1 image | native 3DGS | MIT ✅ | ✅ **default generator** |
| **[Puppeteer](https://github.com/Seed3D/Puppeteer)** · NeurIPS '25 — *baseline* | rig | 3D model | skeleton + skinning | Apache 2.0 ✅ | ✅ **default rigger** |
| [LGM](https://github.com/3DTopia/LGM) · ECCV '24 | generate | 1 image | native 3DGS | MIT, INRIA rasterizer NC ⚠️ | ⚠️ swap the INRIA rasterizer for a commercial one (e.g. gsplat) |
| [TripoSG](https://github.com/VAST-AI-Research/TripoSG) · arXiv '25 | generate | 1 image | mesh | MIT ✅ | ⚠️ needs a mesh→splat step |
| [InstantMesh](https://github.com/TencentARC/InstantMesh) · arXiv '24 | generate | 1 image | mesh | Apache 2.0 ✅ | ⚠️ needs a mesh→splat step |
| [CRM](https://github.com/thu-ml/CRM) · ECCV '24 | generate | 1 image | mesh | MIT ✅ | ⚠️ needs a mesh→splat step |
| [Hunyuan3D 2.0/2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2) · arXiv '25 | generate | 1 image | textured mesh | Tencent Community (region + MAU caps) ⚠️ | ⚠️ mesh→splat step + restricted license |
| [Wonder3D](https://github.com/xxlong0/Wonder3D) · CVPR '24 | generate | 1 image | mesh | MIT ✅ | ⚠️ mesh→splat + per-subject NeuS extraction |
| [Rodin Gen-1 / Hyper3D](https://hyper3d.ai/) · CLAY, SIGGRAPH '24 | generate | 1 image | textured mesh | closed paid API ❌ | ❌ no code/weights; mesh; closed API |
| [UniRig](https://github.com/VAST-AI-Research/UniRig) · SIGGRAPH '25 | rig | 3D mesh | skeleton + skinning | MIT ✅ | ⚠️ rigs mesh vertices → transfer weights to splat points |
| [MagicArticulate](https://github.com/Seed3D/MagicArticulate) · CVPR '25 | rig | 3D mesh | skeleton + skinning | Apache 2.0 ✅ | ⚠️ mesh-vertex weights → splat transfer |
| [RigNet](https://github.com/zhan-xu/RigNet) · SIGGRAPH '20 | rig | 3D mesh | skeleton + skinning | GPLv3 ❌ | ❌ copyleft conflicts with closed product; mesh verts not splats |
| [Anything-World](https://anything.world/) · commercial | rig | 3D model | rigged mesh | closed paid API ❌ | ❌ closed API; mesh not splat |
| [Neural Blend Shapes](https://github.com/PeizhuoLi/neural-blend-shapes) · SIGGRAPH '21 | rig | 3D mesh | biped skeleton + skinning | BSD-2 ✅ | ❌ biped-only; mesh verts not splats |

### Quadruped mammals — vs. TripoSplat/TRELLIS → SuperAnimal → SMAL (single image → 3DGS → keypoint-anchored SMAL fit)

Three stages: **reconstruct** the splat, detect **keypoints**, fit the parametric **SMAL** rig. The detection gate is deliberate — no SuperAnimal detection ⇒ `NotAQuadrupedMammalError`, no fallback rig.

| Method | Role | Input | Output | License (commercial) | Plug-and-play |
|--------|------|-------|--------|----------------------|---------------|
| **[TripoSplat](https://github.com/VAST-AI-Research/TripoSplat)** — *baseline* | reconstruct | 1 image | 3DGS | MIT ✅ | ✅ **default recon backend** (cleaner muzzles than TRELLIS) |
| **[TRELLIS](https://github.com/microsoft/TRELLIS)** · CVPR '25 — *baseline* | reconstruct | 1 image | 3DGS | MIT ✅ | ✅ alternate recon backend (`--backend trellis`) |
| **[SuperAnimal-Quadruped](https://github.com/DeepLabCut/DeepLabCut)** — *baseline* | keypoints | 1 image | 39 2D keypoints | code LGPL-3.0, weights NC ✅† | ✅ keypoint anchor (weights research-only) |
| **[SMAL](https://smal.is.tue.mpg.de/)** · CVPR '17 — *baseline* | rig model | — | quadruped mesh + skeleton | MPI research-only ✅† | ✅ the rig model (research-only) |
| [3D-Fauna](https://github.com/3DAnimals/3DAnimals) · CVPR '24 | end-to-end | 1 image | DMTet mesh + learned skeleton | MIT ✅ | ⚠️ commercial-clean, but mesh + its own rig — replaces the whole pipeline |
| [MagicPony](https://github.com/elliottwu/MagicPony) · CVPR '23 | end-to-end | 1 image | DMTet mesh + learned skeleton | MIT ✅ | ⚠️ mesh + own (non-SMAL) rig, not a splat |
| [Ponymation](https://github.com/3DAnimals/3DAnimals) · ECCV '24 | end-to-end | 1 image | mesh + heuristic skeleton + motion | MIT ✅ | ⚠️ generative mesh motion, not splat + SMAL |
| [SMALify / SMALR](https://github.com/benjiebob/SMALify) | fit | 1 image + kpts | SMAL mesh | MIT, SMAL dep ❌ | ❌ needs masks/keypoints; mesh not splat; SMAL NC |
| [BARC](https://github.com/runa91/barc_release) · CVPR '22 | fit | 1 image | SMAL params | MPI research-only ❌ | ❌ dogs only; non-commercial |
| [BITE](https://github.com/runa91/bite_release) · CVPR '23 | fit | 1 image | D-SMAL params | MPI research-only ❌ | ❌ dogs only; non-commercial |
| [Animal Avatars](https://github.com/facebookresearch/AnimalAvatar) · ECCV '24 | reconstruct | mono video | SMAL mesh + neural texture | CC BY-NC ❌ | ❌ video + per-subject; dog-centric; non-commercial |
| [RAC](https://github.com/gengshan-y/rac) · CVPR '23 | reconstruct | multi-video | NeRF + category skeleton | no LICENSE ⚠️ | ❌ multi-video per category; NeRF not splat |
| [BANMo](https://github.com/facebookresearch/banmo) · CVPR '22 | end-to-end | multi-video | NeRF + neural bones | CC BY-NC ❌ | ❌ many videos + per-subject; NeRF; non-commercial |
| [CASA](https://github.com/Iven-Wu/CASA) · NeurIPS '22 | end-to-end | mono video | mesh + skeleton | contradictory NC ❌ | ❌ video + per-subject; non-commercial |

† The parametric models and pose weights Splattie already depends on (SMPL-X, SMAL, SuperAnimal) are MPI/EPFL research-only. Splattie ships as the reference pipeline under those terms; commercial deployment needs the licenses in [License](#license). The splat output and the widget are unaffected.

## Acknowledgements

Splattie builds on outstanding open-source research:

- **[LAM](https://github.com/aigc3d/LAM)** (SIGGRAPH 2025) - Large Avatar Model for single-image 3DGS **head** generation. By Zixuan Zeng et al., AIGC3D.
- **[LHM](https://github.com/aigc3d/LHM)** (SIGGRAPH 2025) - Large Animatable Human Model for single-image 3DGS **body** generation. By AIGC3D.
- **[TRELLIS](https://github.com/microsoft/TRELLIS)** - single-image 3D asset reconstruction. By Microsoft.
- **[TripoSplat](https://github.com/VAST-AI-Research/TripoSplat)** - single-image image-to-3D-gaussian reconstruction (default animal backend; cleaner muzzles/faces than TRELLIS). By VAST-AI Research.
- **[Puppeteer](https://github.com/Seed3D/Puppeteer)** (NeurIPS 2025) - automatic skeleton and skinning for generated 3D assets. By ByteDance Seed, NTU & A*STAR.
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
