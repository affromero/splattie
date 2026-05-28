#!/bin/bash
set -e

echo "=== Splattie GPU Backend Setup ==="
echo "Requires: CUDA 12.x, Python 3.10+"

cd "$(dirname "$0")/.."

echo "[1/4] Installing Python deps via uv sync --extra gpu..."
uv sync --extra gpu

# chumpy is needed to unpickle FLAME's flame2023.pkl, but its build needs pip
# at build time and breaks under uv's build isolation, so it's installed here
# (not declared in pyproject). A runtime shim in lam/method.py restores the
# py3.11/numpy>=1.24 names chumpy 0.70 expects.
uv pip install chumpy

echo "[2/4] Installing CUDA build-from-source extensions..."
# These need --no-build-isolation so they share the active torch install
# during their CUDA compilation. They are not in pyproject.toml because
# uv's resolver builds them eagerly without seeing torch.
uv pip install --no-build-isolation \
  "git+https://github.com/camenduru/simple-knn/" \
  "nvdiffrast@git+https://github.com/ShenhanQian/nvdiffrast@backface-culling" \
  "git+https://github.com/facebookresearch/pytorch3d.git"

echo "[3/4] Building FaceBoxes CUDA extension..."
cd vendor/LAM/external/landmark_detection/FaceBoxesV2/utils/
sh make.sh
cd ../../../../../..

echo "[4/4] Downloading model weights..."
uv run python <<'PY'
import os
import tarfile
import glob
from huggingface_hub import snapshot_download

lam_dir = "vendor/LAM"
os.makedirs(f"{lam_dir}/model_zoo/lam_models/releases/lam", exist_ok=True)
snapshot_download("3DAIGC/LAM-20K", local_dir=f"{lam_dir}/model_zoo/lam_models/releases/lam/lam-20k")

step_dir = f"{lam_dir}/model_zoo/lam_models/releases/lam/lam-20k/step_045500"
os.makedirs(step_dir, exist_ok=True)
link = f"{step_dir}/model.safetensors"
if not os.path.exists(link):
    os.symlink("../model.safetensors", link)

os.makedirs(f"{lam_dir}/model_zoo/human_parametric_models", exist_ok=True)
snapshot_download("3DAIGC/LAM-assets", local_dir=f"{lam_dir}/model_zoo/human_parametric_models")
for tar_path in glob.glob(f"{lam_dir}/model_zoo/human_parametric_models/*.tar"):
    with tarfile.open(tar_path) as tf:
        tf.extractall(f"{lam_dir}/model_zoo/human_parametric_models")
print("Weights ready")
PY

echo "=== Setup complete ==="
echo "Run: uv run uvicorn splattie.api.app:create_app --factory --port 8000"
