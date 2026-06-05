#!/bin/bash
set -e

echo "=== Splattie GPU Backend Setup ==="
echo "Requires: CUDA 12.x, Python 3.11+"

cd "$(dirname "$0")/.."

echo "[1/7] Initializing vendor submodules recursively..."
if [[ "${SPLATTIE_SKIP_SUBMODULE_UPDATE:-0}" == "1" ]]; then
  echo "Skipping submodule update (SPLATTIE_SKIP_SUBMODULE_UPDATE=1)"
else
  git submodule update --init --recursive vendor/LAM vendor/LHM vendor/TRELLIS vendor/TripoSplat vendor/Puppeteer
fi

echo "[2/7] Installing all Python deps via uv sync --extra gpu --extra cuda..."
# Everything is resolved by uv from pyproject. chumpy's legacy setup.py imports
# the pip module during a no-build-isolation build, so seed that module without
# ever invoking raw pip. The `gpu` extra is wheels; the `cuda` extra is the
# build-from-source pytorch3d/nvdiffrast/simple-knn + chumpy, compiled by uv
# against the in-env torch via [tool.uv] no-build-isolation-package +
# [tool.uv.extra-build-dependencies] + [tool.uv.sources]. A runtime shim in
# lam/method.py restores the py3.11/numpy names chumpy 0.70 needs when FLAME's
# flame2023.pkl unpickles.
uv pip install pip setuptools wheel
uv sync --extra gpu --extra cuda ${SPLATTIE_UV_SYNC_FLAGS:-}

echo "[3/7] Building the FaceBoxes Cython extension (in-repo, not a package)..."
# build.py must run with the venv's python (the repo's make.sh hardcodes system
# python3, producing a .so for the wrong Python ABI).
BACKEND_DIR="$PWD"
cd vendor/LAM/external/landmark_detection/FaceBoxesV2/utils/
uv run --project "$BACKEND_DIR" ${SPLATTIE_UV_RUN_FLAGS:-} python build.py build_ext --inplace
cd "$BACKEND_DIR"

echo "[4/7] Downloading LHM model weights..."
uv run ${SPLATTIE_UV_RUN_FLAGS:-} python <<'PY'
import tarfile
import urllib.request
from pathlib import Path

from huggingface_hub import snapshot_download

lhm_dir = Path("vendor/LHM")
snapshot_download("3DAIGC/LHM-500M", cache_dir=lhm_dir / "pretrained_models" / "huggingface")

human_files = lhm_dir / "pretrained_models" / "human_model_files"
if not human_files.exists():
    prior_tar = lhm_dir / "LHM_prior_model.tar"
    urllib.request.urlretrieve(
        "https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/data/for_lingteng/LHM/LHM_prior_model.tar",
        prior_tar,
    )
    with tarfile.open(prior_tar) as tf:
        tf.extractall(lhm_dir)
    prior_tar.unlink()

dense_points = lhm_dir / "pretrained_models" / "dense_sample_points" / "1_20000.ply"
if not dense_points.exists():
    dense_points.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(
        "https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/data/LHM/1_20000.ply",
        dense_points,
    )

print("LHM weights ready")
PY

echo "[5/7] Downloading LAM model weights..."
uv run ${SPLATTIE_UV_RUN_FLAGS:-} python <<'PY'
import glob
import os
import shutil
import tarfile

from huggingface_hub import snapshot_download

lam_dir = "vendor/LAM"

# 1. LAM-20K head-reconstruction model.
os.makedirs(f"{lam_dir}/model_zoo/lam_models/releases/lam", exist_ok=True)
snapshot_download("3DAIGC/LAM-20K", local_dir=f"{lam_dir}/model_zoo/lam_models/releases/lam/lam-20k")
step_dir = f"{lam_dir}/model_zoo/lam_models/releases/lam/lam-20k/step_045500"
os.makedirs(step_dir, exist_ok=True)
link = f"{step_dir}/model.safetensors"
if not os.path.exists(link):
    os.symlink("../model.safetensors", link)

# 2. LAM-assets. Each tar already carries its own top-level prefix
# (model_zoo/, assets/, pretrained_models/, thirdparties/), so extract into
# lam_dir ITSELF. Extracting into a subdir double-nests everything (e.g.
# model_zoo/human_parametric_models/model_zoo/flame_tracking_models/...).
staging = f"{lam_dir}/.lam_assets_download"
snapshot_download("3DAIGC/LAM-assets", local_dir=staging)
for tar_path in glob.glob(f"{staging}/*.tar"):
    with tarfile.open(tar_path) as tf:
        tf.extractall(lam_dir)

# 3. flame_assets (flame2023.pkl + FLAME_masks/landmark_embedding/head_template)
# is FLAME-license-gated and NOT in LAM-assets. Reuse the identical FLAME 2023
# assets vendored with LHM (download LHM weights first). flame.py reads them from
# model_zoo/human_parametric_models/flame_assets/.
flame_src = "vendor/LHM/pretrained_models/human_model_files/flame_assets"
flame_dst = f"{lam_dir}/model_zoo/human_parametric_models/flame_assets"
if os.path.isdir(flame_src) and not os.path.exists(flame_dst):
    os.makedirs(os.path.dirname(flame_dst), exist_ok=True)
    shutil.copytree(flame_src, flame_dst)
    print(f"Copied flame_assets from {flame_src}")
elif not os.path.exists(flame_dst):
    print(f"WARNING: {flame_dst} missing — populate with FLAME 2023 (license-gated) or fetch LHM weights first")
print("Weights ready")
PY

echo "[6/7] Downloading object-generation weights..."
uv run ${SPLATTIE_UV_RUN_FLAGS:-} python <<'PY'
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

puppeteer = Path("vendor/Puppeteer")

# TRELLIS image pipeline weights; from_pretrained can fetch lazily, but setup-gpu
# makes production readiness explicit and avoids failing the first user request.
snapshot_download("microsoft/TRELLIS-image-large")

# TripoSplat image->3D-gaussian weights (~3.8 GB) — the default reconstruction backend for the
# quadruped_mammal pipeline (much cleaner animal faces than TRELLIS); objects still use TRELLIS.
# Land them under vendor/TripoSplat/ckpts so the paths match triposplat.py's expected ckpts/ layout.
snapshot_download("VAST-AI/TripoSplat", local_dir=str(Path("vendor/TripoSplat") / "ckpts"))

# Puppeteer skeleton generation.
hf_hub_download(
    repo_id="Maikou/Michelangelo",
    filename="checkpoints/aligned_shape_latents/shapevae-256.ckpt",
    local_dir=puppeteer / "skeleton" / "third_partys" / "Michelangelo",
)
hf_hub_download(
    repo_id="Seed3D/Puppeteer",
    filename="skeleton_ckpts/puppeteer_skeleton_w_diverse_pose.pth",
    local_dir=puppeteer / "skeleton",
)

# Puppeteer skinning. The skinning code imports Michelangelo from its own
# third_partys folder, so point it at the skeleton-stage copy.
src = (puppeteer / "skeleton" / "third_partys" / "Michelangelo").resolve()
dst = puppeteer / "skinning" / "third_partys" / "Michelangelo"
if not dst.exists():
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.symlink_to(src, target_is_directory=True)

hf_hub_download(
    repo_id="mikaelaangel/partfield-ckpt",
    filename="model_objaverse.ckpt",
    local_dir=puppeteer / "skinning" / "third_partys" / "PartField" / "ckpt",
)
hf_hub_download(
    repo_id="Seed3D/Puppeteer",
    filename="skinning_ckpts/puppeteer_skin_w_diverse_pose_depth1.pth",
    local_dir=puppeteer / "skinning",
)
print("Object weights ready")
PY

echo "[7/7] Setting up the quadruped-mammal runtime (SMAL + DeepLabCut/SuperAnimal)..."
# SMAL parametric quadruped model (MPI noncommercial). Gitignored under vendor/SMAL.
if [[ -n "${MPI_USERNAME:-}" && -n "${MPI_PASSWORD:-}" ]]; then
  uv run ${SPLATTIE_UV_RUN_FLAGS:-} python scripts/download_smal.py
else
  echo "  Skipping SMAL download (MPI_USERNAME/MPI_PASSWORD unset). Fetch later via:"
  echo "    doppler run -p splattie -c prd -- uv run python backend/scripts/download_smal.py"
fi
# SuperAnimal-Quadruped runs in an ISOLATED DeepLabCut venv: deeplabcut's deps conflict with
# the backend's numpy<2 pin, so it cannot share this env. This is the one sanctioned uv-pip
# exception — an out-of-project tool venv invoked only as a subprocess.
DLC_PYTHON="${SPLATTIE_DLC_PYTHON:-$HOME/dlc-venv/bin/python}"
DLC_VENV_DIR="$(dirname "$(dirname "$DLC_PYTHON")")"
if [[ ! -x "$DLC_PYTHON" ]]; then
  echo "  Creating DeepLabCut venv at $DLC_VENV_DIR ..."
  uv venv "$DLC_VENV_DIR" --python 3.11
  uv pip install --python "$DLC_PYTHON" "deeplabcut>=3.0"
fi
echo "  Quadruped runtime ready (SuperAnimal-Quadruped weights download lazily on first inference)."

echo "=== Setup complete ==="
echo "Run: uv run uvicorn splattie.api.app:create_app --factory --port 8000"
