#!/bin/bash
set -e

echo "=== Splattie GPU Backend Setup ==="
echo "Requires: CUDA 12.x, Python 3.10+"

cd "$(dirname "$0")/.."

echo "[1/3] Installing all Python deps via uv sync --extra gpu --extra cuda..."
# Everything is declared in pyproject — no pip. The `gpu` extra is wheels; the
# `cuda` extra is the build-from-source pytorch3d/nvdiffrast/simple-knn + chumpy,
# compiled by uv against the in-env torch via [tool.uv] no-build-isolation-package
# + [tool.uv.extra-build-dependencies] + [tool.uv.sources]. A runtime shim in
# lam/method.py restores the py3.11/numpy names chumpy 0.70 needs when FLAME's
# flame2023.pkl unpickles.
uv sync --extra gpu --extra cuda

echo "[2/3] Building the FaceBoxes Cython extension (in-repo, not a package)..."
# build.py must run with the venv's python (the repo's make.sh hardcodes system
# python3, producing a .so for the wrong Python ABI).
BACKEND_DIR="$PWD"
cd vendor/LAM/external/landmark_detection/FaceBoxesV2/utils/
uv run --project "$BACKEND_DIR" python build.py build_ext --inplace
cd "$BACKEND_DIR"

echo "[3/3] Downloading model weights..."
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
