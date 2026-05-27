#!/bin/bash
# Build LAM_WebRender, apply runtime patches, and copy to apps/web/public/demo/
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RENDERER_DIR="$REPO_ROOT/packages/lam-renderer"
OUTPUT_JS="$RENDERER_DIR/dist/assets/index.js"
DEST="$REPO_ROOT/apps/web/public/demo/lam-renderer.js"

cd "$RENDERER_DIR"

echo "[1/4] Building LAM_WebRender..."
./node_modules/.bin/vite build

if [ ! -f "$OUTPUT_JS" ]; then
  # vite may hash the filename — find the actual JS output
  OUTPUT_JS=$(find "$RENDERER_DIR/dist/assets" -name '*.js' -type f | head -1)
  if [ -z "$OUTPUT_JS" ]; then
    echo "ERROR: No JS output found in dist/assets/" >&2
    exit 1
  fi
fi

echo "[2/4] Patching setExpression() into FLAME code path..."
# Before viewer.update, inject setExpression call
sed -i.bak 's/this\.viewer\.update(this\.viewer\.renderer, this\.viewer\.camera)/this.setExpression(); this.viewer.update(this.viewer.renderer, this.viewer.camera)/g' "$OUTPUT_JS"

echo "[3/4] Patching readPixels for face detection..."
# After consecutiveRenderFrames++, inject the pixel reading code for hit testing
READPIXELS_CODE='if (window.__mouseX !== undefined \&\& this.viewer.renderer) { try { const gl = this.viewer.renderer.getContext(); const dpr = window.devicePixelRatio || 1; const px = Math.floor(window.__mouseX * dpr); const py = Math.floor((gl.drawingBufferHeight - window.__mouseY * dpr)); const pixel = new Uint8Array(4); gl.readPixels(px, py, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, pixel); const bg = 14 + 14 + 20; const sum = pixel[0] + pixel[1] + pixel[2]; window.__onSplat = Math.abs(sum - bg) > 40; } catch(e) {} }'

sed -i.bak "s/this\.viewer\.consecutiveRenderFrames++/this.viewer.consecutiveRenderFrames++; ${READPIXELS_CODE}/" "$OUTPUT_JS"

# Clean up backup files
rm -f "$OUTPUT_JS.bak"

echo "[4/4] Copying to apps/web/public/demo/lam-renderer.js..."
cp "$OUTPUT_JS" "$DEST"

echo "Done. Output: $DEST ($(wc -c < "$DEST" | tr -d ' ') bytes)"
