# syntax=docker/dockerfile:1.7
# Splattie GPU backend
#
# CUDA 12.1 + Python 3.11 + uv + the full generation stack:
# LAM heads, LHM bodies, TRELLIS reconstruction, and Puppeteer rigging.
FROM nvidia/cuda:13.3.1-devel-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV FORCE_CUDA=1
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0"
ENV UV_LINK_MODE=copy

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      ca-certificates \
      cmake \
      curl \
      git \
      libegl1 \
      libgl1 \
      libglib2.0-0 \
      libgomp1 \
      libsm6 \
      libx11-6 \
      libxext6 \
      libxi6 \
      libxkbcommon0 \
      libxrender1 \
      libxrandr2 \
      ninja-build \
      pkg-config \
      wget \
      xz-utils && \
    rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH=/root/.local/bin:${PATH}

RUN --mount=type=cache,target=/root/.cache/uv \
    uv python install 3.11.15

WORKDIR /app

# Copy metadata, setup script, and vendored generator source before app source
# so the slow CUDA/weight layer is cacheable.
COPY backend/pyproject.toml backend/uv.lock backend/.python-version backend/
COPY backend/scripts/setup-gpu.sh backend/scripts/setup-gpu.sh
COPY backend/vendor/ backend/vendor/

WORKDIR /app/backend

RUN uv venv --python 3.11 .venv
ENV PATH=/app/backend/.venv/bin:${PATH}
ENV VIRTUAL_ENV=/app/backend/.venv

RUN --mount=type=cache,target=/root/.cache/uv \
    SPLATTIE_SKIP_SUBMODULE_UPDATE=1 \
    SPLATTIE_UV_SYNC_FLAGS="--frozen --no-install-project" \
    SPLATTIE_UV_RUN_FLAGS="--no-sync" \
    bash scripts/setup-gpu.sh

COPY packages/splattie-widget/package.json /app/packages/splattie-widget/package.json
COPY packages/splattie-widget/public/ /app/packages/splattie-widget/public/

RUN uv run --no-sync python - <<'PY'
from rembg import new_session
import torch

torch.hub.load_state_dict_from_url(
    "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth",
    map_location="cpu",
)
new_session("u2net")
PY

COPY backend/src/ /app/backend/src/
RUN uv pip install -e /app/backend

RUN mkdir -p /app/backend/data/uploads /app/backend/data/generations

EXPOSE 8000
CMD ["uvicorn", "splattie.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
