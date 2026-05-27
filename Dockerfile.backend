# ── Splattie GPU Backend ────────────────────────────────────────────────
# nvidia/cuda 12.1.1 + Python 3.10 + uv + Blender 4.2 + FBX SDK + LAM
# ──────────────────────────────────────────────────────────────────────

FROM nvidia/cuda:12.1.1-devel-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# ── System packages + Python 3.10 via deadsnakes ─────────────────────
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
      python3.10 python3.10-dev python3.10-venv python3.10-distutils \
      curl ca-certificates git wget xz-utils libgl1 libglib2.0-0 \
      libxrender1 libsm6 libxext6 libgomp1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    rm -rf /var/lib/apt/lists/*

# ── uv ───────────────────────────────────────────────────────────────
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# ── Blender 4.2 headless ─────────────────────────────────────────────
RUN wget -q https://mirror.clarkson.edu/blender/release/Blender4.2/blender-4.2.0-linux-x64.tar.xz \
      -O /tmp/blender.tar.xz && \
    tar -xJf /tmp/blender.tar.xz -C /opt/ && \
    ln -s /opt/blender-4.2.0-linux-x64/blender /usr/local/bin/blender && \
    rm /tmp/blender.tar.xz
ENV BLENDER_PATH="/opt/blender-4.2.0-linux-x64"

# ── FBX SDK ──────────────────────────────────────────────────────────
RUN wget -q https://damassets.autodesk.net/content/dam/autodesk/www/files/fbx202037_fbxpythonsdk_linux.tar.gz \
      -O /tmp/fbx_sdk.tar.gz && \
    mkdir -p /tmp/fbx_sdk && \
    tar -xzf /tmp/fbx_sdk.tar.gz -C /tmp/fbx_sdk && \
    yes yes | python3 /tmp/fbx_sdk/fbx202037_fbxpythonsdk_linux /opt/fbx_sdk && \
    cp /opt/fbx_sdk/lib/Python310_x64/* \
       /usr/lib/python3/dist-packages/ 2>/dev/null || \
    cp /opt/fbx_sdk/lib/Python310_x64/* \
       /usr/local/lib/python3.10/dist-packages/ 2>/dev/null || true && \
    rm -rf /tmp/fbx_sdk /tmp/fbx_sdk.tar.gz

WORKDIR /app

# ── Python dependencies ──────────────────────────────────────────────
# Copy backend project metadata first for layer caching
COPY backend/pyproject.toml backend/.python-version backend/
WORKDIR /app/backend

# Create venv and install base deps
RUN uv venv --python 3.10 .venv
ENV PATH="/app/backend/.venv/bin:$PATH"
ENV VIRTUAL_ENV="/app/backend/.venv"

RUN uv sync --extra gpu --inexact --no-install-project

# ── PyTorch + xformers (CUDA 12.1) ──────────────────────────────────
RUN uv pip install \
      torch==2.3.0 torchvision==0.18.0 \
      --index-url https://download.pytorch.org/whl/cu121 && \
    uv pip install \
      xformers==0.0.26.post1 \
      --index-url https://download.pytorch.org/whl/cu121

# ── LAM vendor + CUDA extensions ─────────────────────────────────────
COPY backend/vendor/LAM /app/backend/vendor/LAM

RUN uv pip install --no-build-isolation \
      "git+https://github.com/ashawkey/diff-gaussian-rasterization/" \
      "git+https://github.com/camenduru/simple-knn/" \
      "nvdiffrast@git+https://github.com/ShenhanQian/nvdiffrast@backface-culling" \
      "git+https://github.com/facebookresearch/pytorch3d.git"

# ── LAM Python dependencies ──────────────────────────────────────────
RUN uv pip install --no-build-isolation \
      face-detection-tflite scikit-image==0.20.0 "rembg[gpu]" chumpy accelerate \
      huggingface-hub==0.23.2

# ── Build FaceBoxes C extension ──────────────────────────────────────
RUN cd /app/backend/vendor/LAM/external/landmark_detection/FaceBoxesV2/utils/ && \
    sh make.sh

# ── Download model weights from HuggingFace ──────────────────────────
RUN python3 -c " \
from huggingface_hub import snapshot_download; \
import os; \
lam_dir = '/app/backend/vendor/LAM'; \
os.makedirs(f'{lam_dir}/model_zoo/lam_models/releases/lam', exist_ok=True); \
snapshot_download('3DAIGC/LAM-20K', local_dir=f'{lam_dir}/model_zoo/lam_models/releases/lam/lam-20k'); \
os.makedirs(f'{lam_dir}/model_zoo/lam_models/releases/lam/lam-20k/step_045500', exist_ok=True); \
os.path.exists(f'{lam_dir}/model_zoo/lam_models/releases/lam/lam-20k/step_045500/model.safetensors') or \
  os.symlink('../model.safetensors', f'{lam_dir}/model_zoo/lam_models/releases/lam/lam-20k/step_045500/model.safetensors'); \
os.makedirs(f'{lam_dir}/model_zoo/human_parametric_models', exist_ok=True); \
snapshot_download('3DAIGC/LAM-assets', local_dir=f'{lam_dir}/model_zoo/human_parametric_models'); \
import tarfile, glob; \
[tarfile.open(p).extractall(f'{lam_dir}/model_zoo/human_parametric_models') for p in glob.glob(f'{lam_dir}/model_zoo/human_parametric_models/*.tar')]; \
print('Weights ready'); \
"

# ── Download OAC template assets ─────────────────────────────────────
RUN wget -q https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/data/LAM/sample_oac.tar \
      -O /tmp/sample_oac.tar && \
    tar -xf /tmp/sample_oac.tar -C /app/backend/vendor/LAM/ && \
    rm /tmp/sample_oac.tar

# ── Backend source code ─────────────────────────────────────────────
COPY backend/src/ /app/backend/src/
RUN uv pip install -e /app/backend --no-build-isolation

RUN mkdir -p /app/backend/data/uploads /app/backend/data/generations

EXPOSE 8000
CMD ["uvicorn", "splattie.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
