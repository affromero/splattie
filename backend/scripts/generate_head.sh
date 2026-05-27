#!/bin/bash
set -e

IMAGE_PATH="$1"
OUTPUT_DIR="$2"

if [ -z "$IMAGE_PATH" ] || [ -z "$OUTPUT_DIR" ]; then
  echo "Usage: $0 <image_path> <output_dir>"
  exit 1
fi

BASENAME=$(basename "$IMAGE_PATH" | sed 's/\.[^.]*$//')
LAM_DIR="$(dirname "$0")/../vendor/LAM"

cd "$LAM_DIR"
source "$(dirname "$0")/../.venv/bin/activate"

echo "[1/4] Running FLAME tracking..."
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PYTHONPATH:. python -m lam.launch infer.lam \
  --config configs/inference/lam-20k-8gpu.yaml \
  model_name=model_zoo/lam_models/releases/lam/lam-20k/step_045500/ \
  image_input="$IMAGE_PATH" \
  save_ply=true save_img=true export_video=true export_mesh=false \
  vis_motion=false render_fps=30 \
  motion_seqs_dir="tracking_output/export/${BASENAME}/" \
  motion_img_dir=null rank=0 nodes=0 2>&1 | tail -5

echo "[2/4] Collecting outputs..."
mkdir -p "$OUTPUT_DIR/${BASENAME}"
cp "exps/cano_gs/${BASENAME}_gs_offset.ply" "$OUTPUT_DIR/${BASENAME}/offset.ply"

echo "[3/4] Adding skeleton assets..."
DEMO_ZIP="$(dirname "$0")/../vendor/LAM_WebRender/asset/arkit/p2-1.zip"
if [ -f "$DEMO_ZIP" ]; then
  TMP=$(mktemp -d)
  unzip -o "$DEMO_ZIP" -d "$TMP" > /dev/null
  cp "$TMP/p2-1/skin.glb" "$OUTPUT_DIR/${BASENAME}/"
  cp "$TMP/p2-1/animation.glb" "$OUTPUT_DIR/${BASENAME}/"
  cp "$TMP/p2-1/vertex_order.json" "$OUTPUT_DIR/${BASENAME}/"
  rm -rf "$TMP"
fi

echo "[4/4] Creating ZIP bundle..."
cd "$OUTPUT_DIR"
zip -r "${BASENAME}.zip" "${BASENAME}/" > /dev/null
echo "Bundle: $OUTPUT_DIR/${BASENAME}.zip ($(du -h "${BASENAME}.zip" | cut -f1))"
