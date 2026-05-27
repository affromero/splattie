#!/bin/bash
set -e

echo "=== Mirada GPU Backend Setup ==="
echo "Requires: CUDA 12.x, Python 3.10+"

cd "$(dirname "$0")/.."

echo "[1/5] Creating venv and installing base deps..."
uv sync --extra gpu

echo "[2/5] Installing PyTorch + xformers (CUDA 12.1)..."
uv pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121
uv pip install xformers==0.0.26.post1 --index-url https://download.pytorch.org/whl/cu121

echo "[3/5] Installing CUDA extensions (requires compilation)..."
uv pip install --no-build-isolation \
  "git+https://github.com/ashawkey/diff-gaussian-rasterization/" \
  "git+https://github.com/camenduru/simple-knn/" \
  "nvdiffrast@git+https://github.com/ShenhanQian/nvdiffrast@backface-culling" \
  "git+https://github.com/facebookresearch/pytorch3d.git"

echo "[4/5] Installing LAM dependencies..."
uv pip install --no-build-isolation \
  face-detection-tflite scikit-image==0.20.0 "rembg[gpu]" chumpy accelerate \
  huggingface-hub==0.23.2

echo "[5/5] Building FaceBoxes extension..."
cd vendor/LAM/external/landmark_detection/FaceBoxesV2/utils/
sh make.sh
cd ../../../../../..

echo "[6/6] Downloading model weights..."
uv run python -c "
from huggingface_hub import snapshot_download
import os
lam_dir = 'vendor/LAM'
os.makedirs(f'{lam_dir}/model_zoo/lam_models/releases/lam', exist_ok=True)
snapshot_download('3DAIGC/LAM-20K', local_dir=f'{lam_dir}/model_zoo/lam_models/releases/lam/lam-20k')
os.makedirs(f'{lam_dir}/model_zoo/lam_models/releases/lam/lam-20k/step_045500', exist_ok=True)
os.symlink('../model.safetensors', f'{lam_dir}/model_zoo/lam_models/releases/lam/lam-20k/step_045500/model.safetensors') if not os.path.exists(f'{lam_dir}/model_zoo/lam_models/releases/lam/lam-20k/step_045500/model.safetensors') else None
os.makedirs(f'{lam_dir}/model_zoo/human_parametric_models', exist_ok=True)
snapshot_download('3DAIGC/LAM-assets', local_dir=f'{lam_dir}/model_zoo/human_parametric_models')
import tarfile, glob
for tar_path in glob.glob(f'{lam_dir}/model_zoo/human_parametric_models/*.tar'):
    with tarfile.open(tar_path) as tf:
        tf.extractall(f'{lam_dir}/model_zoo/human_parametric_models')
print('Weights ready')
"

echo "=== Setup complete ==="
echo "Run: uv run uvicorn mirada.api.app:create_app --factory --port 8000"
