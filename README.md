<div align="center">

<img src="apps/web/public/logo.svg" alt="Mirada" width="120" />

# Mirada

**Upload a photo. Get a 3D head whose eyes follow you.**

*Powered by 3D Gaussian Splatting + FLAME animation*

[![Stage](https://img.shields.io/badge/stage-prototype-orange)]()
[![License](https://img.shields.io/badge/license-private-lightgrey)]()
[![Next.js](https://img.shields.io/badge/Next.js-15-black?logo=next.js)](https://nextjs.org)
[![TypeScript](https://img.shields.io/badge/TypeScript-strict-blue?logo=typescript&logoColor=white)](https://typescriptlang.org)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://python.org)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Docker](https://img.shields.io/badge/Docker-deployed-2496ED?logo=docker&logoColor=white)](https://docker.com)
[![Spark](https://img.shields.io/badge/Spark-2.0-orange)](https://sparkjs.dev)

</div>

---

Mirada turns a single photograph into an interactive 3D head that tracks your cursor in real time. The pipeline segments the head (SAM 3), reconstructs a 3D Gaussian Splatting avatar (LAM), compresses it to under 2MB (SPZ), and renders it in any browser with FLAME-driven eye animation — no GPU required on the viewer side.

## Quick Start

```bash
npm install
npm run dev
```

Open [http://localhost:4001](http://localhost:4001).

## Architecture

| Component | Role |
|-----------|------|
| `apps/web/` | Next.js 15 frontend — upload, segment, view |
| `backend/` | FastAPI GPU service — LAM inference, SPZ compression |
| Spark 2.0 | WebGL2 3DGS renderer |
| FLAME LBS | Client-side eye animation (no neural network) |

## How It Works

1. **Upload** a photo with a visible face
2. **Segment** the head automatically (SAM 3 runs in-browser via WebGPU)
3. **Generate** a 3D Gaussian head on the GPU backend (~2 seconds)
4. **View** the interactive head — eyes follow your mouse cursor at 60fps
