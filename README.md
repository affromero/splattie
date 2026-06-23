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

Splattie's bar for a **drop-in** asset generator is narrow: **one image in → feed-forward (no per-subject training) → a 3D Gaussian Splatting asset rigged to a model the widget animates client-side (FLAME / SMPL-X / SMAL / arbitrary-skeleton LBS) → under a commercially-usable license.** Most published avatar/object work misses at least one axis — it needs multi-view or video capture, optimizes per subject, outputs a mesh / NeRF / tri-plane instead of splats, or is research-only. These tables track the notable and recent approaches against that bar (roughly the 20 most relevant per category — full coverage of the field is impractical), so the same papers don't get re-evaluated twice. Rows run baseline first, then nearest-miss, then per-subject / multi-view references. *Verified June 2026 — corrections and additions welcome via PR.*

**How to read these.** *Input* is what one asset needs (1 image / few images / video / multi-view / a 3D mesh, for riggers / text). *FF* (feed-forward): `yes` = one forward pass, `no` = per-subject optimization or training. *Output* is the representation produced — Splattie needs a Gaussian splat, so mesh / NeRF / tri-plane don't drop into the renderer. *License* covers the code **and** every required weight/dataset/dependency; a `*` means the code license is permissive but a required model (FLAME / SMPL-X / SMAL / NeRSemble / INRIA `diff-gaussian-rasterization`) is non-commercial. *Status*: **Baseline** = what Splattie uses; **Adapt** = right shape, needs a conversion or dependency swap; **No** = wrong input, representation, or license; **No code** = paper-only or unreleased.

Splattie's own FLAME / SMPL-X / SMAL baselines (and LHM's released weights) are research-licensed too — see [License](#license) for the commercial paths; the `<splattie-widget>` runtime needs none of them.

### Heads — vs. LAM (single image → FLAME-rigged 3DGS → ARKit client animation)

| Method | Input | FF | Output | License | Status |
|--------|-------|----|--------|---------|--------|
| **[LAM](https://github.com/aigc3d/LAM)** · SIGGRAPH '25 | 1 image | yes | FLAME 3DGS | Apache 2.0 | **Baseline** — what Splattie uses |
| [GAGAvatar](https://github.com/xg-chu/GAGAvatar) · NeurIPS '24 | 1 image | yes | feature-3DGS (neural renderer) | MIT* | **No** — bespoke renderer + driving-image rig, not a static ARKit splat |
| [FlexAvatar — Kirschstein](https://github.com/tobias-kirschstein/flexavatar) · CVPR '26 | 1 image | yes | neural-decoder 3DGS | CC BY-NC | **No** — non-commercial; gaussians from a live neural decoder |
| [FlexAvatar — Peng](https://pengc02.github.io/flexavatar/) · arXiv '25 | 1 image | yes | FLAME-UV 3DGS | none | **No code** — paper-only; NeRSemble-trained (NC) |
| [Avat3r](https://tobias-kirschstein.github.io/avat3r/) · ICCV '25 | few images | yes | 3DGS (LRM) | none* | **No code** — README-only repo; NeRSemble-trained |
| [UIKA](https://zijian-wu.github.io/uika-page/) · CVPR '26 | few images | yes | 3DGS (UV-guided) | MIT* | **No** — code MIT, but FLAME + training data non-commercial |
| [PanoLAM](https://panolam.github.io/) · arXiv '25 | 1 image | yes | 3DGS (LAM-lineage) | none | **No code** — "code coming" |
| [Bringing Your Portrait to 3D Presence](https://github.com/zjwfufu/Bringing-Your-Portrait-to-3D-Presence) · CVPR '26 | 1 image | yes | tri-plane | Apache 2.0* | **No code** — weights withheld; tri-plane, not splat |
| [Real3D-Portrait](https://github.com/yerfor/Real3DPortrait) · ICLR '24 | 1 image | yes | tri-plane | MIT* | **No** — tri-plane talking-head video, not a FLAME-rigged splat |
| [Portrait4D-v2](https://github.com/YuDeng/Portrait-4D) · ECCV '24 | 1 image | yes | tri-plane NeRF | MIT* | **No** — NeRF, driven by a driving frame, not ARKit |
| [GPAvatar](https://github.com/xg-chu/GPAvatar) · ICLR '24 | few images | yes | tri-plane NeRF | MIT* | **No** — NeRF not splat; FLAME/EMOCA non-commercial |
| [HeadGAP](https://headgap.github.io/) · 3DV '25 | few images | no | FLAME 3DGS | none | **No code** — ~5 min/identity fitting |
| [CAP4D](https://github.com/felixtaubner/cap4d) · CVPR '25 (Oral) | few images | no | FLAME-mesh 3DGS | CC BY-NC | **No** — per-subject diffusion + training; non-commercial |
| [HRAvatar](https://github.com/Pixel-Talk/HRAvatar) · CVPR '25 | mono video | no | relightable 3DGS | Apache 2.0* | **No** — per-subject video training; FLAME dep |
| [3D Gaussian Blendshapes](https://github.com/zjumsj/GaussianBlendshapes) · SIGGRAPH '24 | mono video | no | 3DGS blendshapes | GPL-3.0 | **No** — per-subject video; GPL + FLAME (NC) |
| [FlashAvatar](https://github.com/USTC3DV/FlashAvatar-code) · CVPR '24 | mono video | no | FLAME-UV 3DGS | MIT* | **No** — per-subject video training |
| [SplattingAvatar](https://github.com/initialneil/SplattingAvatar) · CVPR '24 | mono video | no | mesh-bound 3DGS | CC BY-NC-SA | **No** — per-subject video; non-commercial |
| [MonoGaussianAvatar](https://github.com/yufan1012/MonoGaussianAvatar) · SIGGRAPH '24 | mono video | no | deform-field 3DGS | MIT* | **No** — per-subject video training |
| [GaussianAvatars](https://github.com/ShenhanQian/GaussianAvatars) · CVPR '24 (Highlight) | multi-view | no | FLAME-mesh 3DGS | CC BY-NC-SA | **No** — 16-cam capture; non-commercial (defines the rigged-GS rep) |
| [Gaussian Head Avatar](https://github.com/YuelangX/Gaussian-Head-Avatar) · CVPR '24 | multi-view | no | MLP-deform 3DGS | research-only | **No** — 16-cam capture; non-commercial |
| [INSTA](https://github.com/Zielon/INSTA) · CVPR '23 | mono video | no | NeRF (instant-ngp) | research-only | **No** — video; NeRF; non-commercial |
| [Relightable Gaussian Codec Avatars](https://shunsukesaito.github.io/rgca/) · CVPR '24 (Oral) | multi-view dome | no | relightable 3DGS | CC BY-NC | **No** — light-stage capture; non-commercial |

### Bodies — vs. LHM (single image → SMPL-X-rigged 3DGS → LBS + arm IK)

| Method | Input | FF | Output | License | Status |
|--------|-------|----|--------|---------|--------|
| **[LHM](https://github.com/aigc3d/LHM)** · ICCV '25 | 1 image | yes | SMPL-X 3DGS | Apache 2.0* | **Baseline** — what Splattie uses (weights CC BY-NC) |
| [LHM++](https://github.com/aigc3d/LHM-plusplus) · ICLR '26 | 1 image | yes | SMPL-X 3DGS | Apache 2.0* | **Adapt** — drop-in LHM upgrade, but weights still CC BY-NC |
| [IDOL](https://github.com/yiyuzhuang/IDOL) · CVPR '25 | 1 image | yes | SMPL-X 3DGS | MIT badge, no LICENSE* | **No** — closest match, but NC (HuGe100K) data + SMPL-X |
| [AniGS](https://github.com/aigc3d/AniGS) · CVPR '25 | 1 image | no | SMPL-X 3DGS | unreleased | **No code** — ~5 min/subject 4D-GS optimization |
| [HumanSplat](https://github.com/humansplat/humansplat) · NeurIPS '24 | 1 image | yes | unrigged 3DGS | MIT* | **No** — static/unrigged GS for novel-view; not animatable as-is |
| [Human3Diffusion](https://github.com/YuxuanSnow/Human3Diffusion) · NeurIPS '24 | 1 image | yes | unrigged 3DGS | MIT | **No** — template-free, no rig — the body can't be animated |
| [HGM](https://github.com/jinnan-chen/HGM) · ICLR '25 | 1 image | yes | 3DGS + mesh | none | **No** — no license; needs an SMPL fit to animate |
| [GST](https://github.com/abdullahamdi/GST) · CVPR '25 | 1 image | yes | SMPL 3DGS | BSD-3* | **No** — SMPL pose-conditioned GS; SMPL dep (NC) |
| [GUAVA](https://github.com/Pixel-Talk/GUAVA) · ICCV '25 | 1 image | yes | upper-body 3DGS | Apache 2.0* | **No** — code Apache, but built on SMPL-X/FLAME (NC) |
| [HumanNOVA](https://humannova.github.io/) · CVPR '26 | 1 image | yes | tri-plane | unlicensed | **No** — neural field, not splat; no license |
| [NoPo-Avatar](https://github.com/wenj/NoPo-Avatar) · NeurIPS '25 | few images | yes | SMPL-X 3DGS | Apache 2.0* | **No** — pose-free feed-forward GS, but SMPL-X dep (NC) |
| [SiTH](https://github.com/SiTH-Diffusion/SiTH) · CVPR '24 | 1 image | yes | textured mesh | MIT* | **No** — mesh not splat; SMPL-X (NC) |
| [PSHuman](https://github.com/pengHTYX/PSHuman) · CVPR '25 | 1 image | yes | textured mesh | MIT* | **No** — mesh not splat; SMPL-X (NC) |
| [AvatarPopUp](https://www.nikoskolot.com/avatarpopup/) · ECCV '24 | 1 image | yes | GHUM-rigged mesh | unreleased | **No code** — mesh + GHUM rig, not splat |
| [En3D](https://github.com/menyifang/En3D) · CVPR '24 | text / seed | no | mesh + auto-rig (FBX) | Apache 2.0 | **No** — generator, not single-image reconstruction; mesh not splat |
| [TaoAvatar](https://github.com/PixelAI-Team/TaoAvatar) · CVPR '25 | multi-view | no | SMPL-X mesh-bound 3DGS | unreleased | **No code** — on-device ARKit GS, but multi-view + no license |
| [GART](https://github.com/JiahuiLei/GART) · CVPR '24 | mono video | no | SMPL 3DGS | MIT* | **No** — per-subject video; SMPL/SMAL/INRIA (NC) |
| [ExAvatar](https://github.com/mks0601/ExAvatar_RELEASE) · ECCV '24 | mono video | no | SMPL-X mesh-bound 3DGS | MIT* | **No** — per-subject video; SMPL-X (NC) |
| [GaussianAvatar](https://github.com/aipixel/GaussianAvatar) · CVPR '24 | mono video | no | SMPL 3DGS | MIT* | **No** — per-subject video; SMPL (NC) |
| [3DGS-Avatar](https://github.com/mikeqzy/3dgs-avatar-release) · CVPR '24 | mono video | no | SMPL deform 3DGS | MIT* | **No** — per-subject video; INRIA/SMPL (NC) |
| [HUGS](https://github.com/apple/ml-hugs) · CVPR '24 | mono video | no | SMPL 3DGS | Apple SCL* | **No** — per-subject video; SMPL (NC) |
| [Animatable Gaussians](https://github.com/lizhe00/AnimatableGaussians) · CVPR '24 | multi-view video | no | SMPL-X 3DGS maps | research-only | **No** — multi-view capture; non-commercial |

### Objects — vs. TRELLIS → Puppeteer (single image → 3DGS → auto-skeleton + LBS)

Two stages: a **generator** (image → 3D) and a **rigger** (3D → skeleton + skinning). A generator must emit gaussians (a mesh needs a mesh→splat step); a rigger must put skinning weights on the splat points (mesh-vertex weights need a transfer step).

| Method | Role | Input | Output | License | Status |
|--------|------|-------|--------|---------|--------|
| **[TRELLIS](https://github.com/microsoft/TRELLIS)** · CVPR '25 | generate | 1 image | native 3DGS | MIT | **Baseline** — default generator |
| **[Puppeteer](https://github.com/Seed3D/Puppeteer)** · NeurIPS '25 | rig | 3D model | skeleton + skinning | Apache 2.0 | **Baseline** — default rigger |
| [LGM](https://github.com/3DTopia/LGM) · ECCV '24 | generate | 1 image | native 3DGS | MIT* | **Adapt** — swap the INRIA rasterizer (e.g. gsplat) for commercial use |
| [Splatter Image](https://github.com/szymanowiczs/splatter-image) · CVPR '24 | generate | 1 image | native 3DGS | BSD-3 | **Adapt** — native GS, but single-object / low-fidelity; still needs rigging |
| [TripoSG](https://github.com/VAST-AI-Research/TripoSG) · arXiv '25 | generate | 1 image | mesh | MIT | **Adapt** — mesh→splat step |
| [InstantMesh](https://github.com/TencentARC/InstantMesh) · arXiv '24 | generate | 1 image | mesh | Apache 2.0 | **Adapt** — mesh→splat step |
| [CRM](https://github.com/thu-ml/CRM) · ECCV '24 | generate | 1 image | mesh | MIT | **Adapt** — mesh→splat step |
| [TripoSR](https://github.com/VAST-AI-Research/TripoSR) · arXiv '24 | generate | 1 image | mesh (NeRF) | MIT | **Adapt** — mesh→splat step; fast but unrigged |
| [SF3D](https://github.com/Stability-AI/stable-fast-3d) · arXiv '24 | generate | 1 image | textured mesh | Stability Community | **Adapt** — mesh→splat + restricted license |
| [3DTopia-XL](https://github.com/3DTopia/3DTopia-XL) · CVPR '25 | generate | 1 image | mesh (PBR) | Apache 2.0 | **Adapt** — mesh→splat step |
| [Step1X-3D](https://github.com/stepfun-ai/Step1X-3D) · arXiv '25 | generate | 1 image | textured mesh | Apache 2.0 | **Adapt** — mesh→splat step |
| [Hunyuan3D 2.0/2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2) · arXiv '25 | generate | 1 image | textured mesh | Tencent Community | **Adapt** — mesh→splat + region/MAU license caps |
| [Wonder3D](https://github.com/xxlong0/Wonder3D) · CVPR '24 | generate | 1 image | mesh | MIT | **Adapt** — mesh→splat + per-subject NeuS step |
| [CLAY / Rodin Gen-1](https://github.com/CLAY-3D/OpenCLAY) · SIGGRAPH '24 | generate | 1 image | textured mesh | closed paid API | **No** — no open code/weights; mesh; paid API |
| [Tripo / Meshy](https://www.tripo3d.ai/) · commercial | generate | 1 image | rigged mesh | closed paid API | **No** — closed API; mesh + mesh-LBS, not splat |
| [UniRig](https://github.com/VAST-AI-Research/UniRig) · SIGGRAPH '25 | rig | 3D mesh | skeleton + skinning | MIT | **Adapt** — mesh-vertex weights → transfer to splat points |
| [MagicArticulate](https://github.com/Seed3D/MagicArticulate) · CVPR '25 | rig | 3D mesh | skeleton + skinning | Apache 2.0 | **Adapt** — mesh-vertex weights → splat transfer |
| [RigAnything](https://github.com/Isabella98Liu/RigAnything) · SIGGRAPH '25 | rig | 3D mesh | skeleton + skinning | Adobe (NC) | **No** — non-commercial license; mesh verts, not splats |
| [Auto-Connect](https://autoconnectrig.github.io/) · NeurIPS '25 | rig | 3D mesh | skeleton + skinning | unreleased | **No code** — rigging code "coming"; mesh-based |
| [RigNet](https://github.com/zhan-xu/RigNet) · SIGGRAPH '20 | rig | 3D mesh | skeleton + skinning | GPL-3.0 | **No** — copyleft; mesh verts, not splats |
| [Anything-World](https://anything.world/) · commercial | rig | 3D model | rigged mesh | closed paid API | **No** — closed API; mesh, not splat |
| [Neural Blend Shapes](https://github.com/PeizhuoLi/neural-blend-shapes) · SIGGRAPH '21 | rig | 3D mesh | biped skeleton + skinning | BSD-2 | **No** — biped-only; mesh verts, not splats |

### Quadruped mammals — vs. TripoSplat/TRELLIS → SuperAnimal → SMAL (single image → 3DGS → keypoint-anchored SMAL fit)

Three stages: **reconstruct** the splat, detect **keypoints**, fit the parametric **SMAL** rig. The detection gate is deliberate — no SuperAnimal detection ⇒ `NotAQuadrupedMammalError`, no fallback rig.

| Method | Role | Input | Output | License | Status |
|--------|------|-------|--------|---------|--------|
| **[TripoSplat](https://github.com/VAST-AI-Research/TripoSplat)** · '26 | reconstruct | 1 image | 3DGS | MIT | **Baseline** — default recon backend (cleaner muzzles than TRELLIS) |
| **[TRELLIS](https://github.com/microsoft/TRELLIS)** · CVPR '25 | reconstruct | 1 image | 3DGS | MIT | **Baseline** — alternate recon backend (`--backend trellis`) |
| **[SuperAnimal-Quadruped](https://github.com/DeepLabCut/DeepLabCut)** · Nat. Comms. '24 | keypoints | 1 image | 39 keypoints | LGPL code / NC weights | **Baseline** — keypoint anchor (weights research-only) |
| **[SMAL](https://smal.is.tue.mpg.de/)** · CVPR '17 | rig model | — | quadruped mesh + skeleton | MPI research-only | **Baseline** — the rig model (research-only) |
| [DualPM](https://github.com/DualPM/DualPM_Paper) · CVPR '25 | end-to-end | 1 image | dual point-maps + skeleton | BSD-3 | **Adapt** — feed-forward & commercial-clean, but point-maps + own rig |
| [3D-Fauna](https://github.com/3DAnimals/3DAnimals) · CVPR '24 | end-to-end | 1 image | DMTet mesh + learned skeleton | MIT | **Adapt** — commercial-clean, but mesh + own rig; replaces the pipeline |
| [MagicPony](https://github.com/elliottwu/MagicPony) · CVPR '23 | end-to-end | 1 image | DMTet mesh + learned skeleton | MIT | **Adapt** — mesh + own (non-SMAL) rig, not a splat |
| [Ponymation](https://github.com/3DAnimals/3DAnimals) · ECCV '24 | end-to-end | 1 image | mesh + heuristic skeleton + motion | MIT | **Adapt** — generative mesh motion, not splat + SMAL |
| [AniMer](https://github.com/luoxue-star/AniMer) · CVPR '25 | fit | 1 image | SMAL params (mesh) | MIT* | **No** — single-image SMAL regression, but mesh not splat; SMAL (NC) |
| [AniMer+](https://github.com/AniMerPlus/AniMerPlus) · TPAMI '25 | fit | 1 image | SMAL/AVES params | MIT* | **No** — adds birds; mesh not splat; SMAL (NC) |
| [Dessie](https://github.com/celiali/Dessie) · ACCV '24 | fit | 1 image | SMAL params (horses) | MIT* | **No** — horses-only; mesh; SMAL (NC) |
| [BARC](https://github.com/runa91/barc_release) · CVPR '22 | fit | 1 image | SMAL params | MPI research-only | **No** — dogs only; non-commercial |
| [BITE](https://github.com/runa91/bite_release) · CVPR '23 | fit | 1 image | D-SMAL params | MPI research-only | **No** — dogs only; non-commercial |
| [SMALify / SMALR](https://github.com/benjiebob/SMALify) · ECCV '20 | fit | 1 image + kpts | SMAL mesh | MIT* | **No** — needs masks/keypoints; mesh not splat; SMAL (NC) |
| [Hi-LASSIE](https://github.com/google/hi-lassie) · CVPR '23 | end-to-end | few images | mesh + skeleton | Apache 2.0 | **No** — needs a per-category image set; mesh not splat |
| [LASSIE](https://github.com/google/lassie) · NeurIPS '22 | end-to-end | few images | mesh + skeleton | Apache 2.0 | **No** — per-category image set; mesh not splat |
| [ARTIC3D](https://github.com/chhankyao/artic3d_recon) · NeurIPS '23 | end-to-end | few images | mesh + skeleton | Apache 2.0 | **No** — sparse image set + per-subject opt; mesh not splat |
| [Animal Avatars](https://github.com/facebookresearch/AnimalAvatar) · ECCV '24 (Oral) | reconstruct | mono video | SMAL mesh + neural texture | CC BY-NC | **No** — per-subject video; dog-centric; non-commercial |
| [RAC](https://github.com/gengshan-y/rac) · CVPR '23 | reconstruct | multi-video | NeRF + category skeleton | none | **No** — multi-video per category; NeRF not splat |
| [BANMo](https://github.com/facebookresearch/banmo) · CVPR '22 | end-to-end | multi-video | NeRF + neural bones | CC BY-NC | **No** — many videos + per-subject; NeRF; non-commercial |
| [CASA](https://github.com/Iven-Wu/CASA) · NeurIPS '22 | end-to-end | mono video | mesh + skeleton | contradictory NC | **No** — per-subject video; non-commercial |

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
