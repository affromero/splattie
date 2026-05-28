#!/usr/bin/env python3
"""Add manifest.json to existing .splattie files.

Idempotent. For each .splattie in the input directory:
  1. Open the ZIP
  2. Skip if it already has a manifest.json with a matching formatVersion
  3. Parse PLY header for vertex count
  4. Compute SHA-256 of the corresponding source thumbnail (if present)
  5. Build a manifest matching the current widget package.json version
  6. Re-bundle with the manifest as the first entry

Runs on any machine (no GPU required) - just rewrites ZIPs.

Usage:
    python backend/scripts/add_manifest_to_splattie.py \\
        --splatties-dir apps/web/public/demos \\
        --thumbs-dir apps/web/public/demos/thumbs
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent if SCRIPT_DIR.name == "scripts" else SCRIPT_DIR
REPO_ROOT = BACKEND_DIR.parent
WIDGET_PKG_JSON = REPO_ROOT / "packages" / "splattie-widget" / "package.json"

ATTRIBUTIONS = {
    "3762763": "Shiny Diamond",
    "3754430": "TUBARONES PHOTOGRAPHY",
    "7705909": "ShotPot",
    "8727488": "Tima Miroshnichenko",
    "8727554": "Tima Miroshnichenko",
    "35466969": "Daniel Hoffman Jackson",
}


def read_widget_version() -> str:
    return json.loads(WIDGET_PKG_JSON.read_text())["version"]


def count_ply_vertices_bytes(data: bytes) -> int:
    """Count vertices from raw PLY bytes."""
    buf = io.BytesIO(data)
    for raw in buf:
        line = raw.decode("ascii", errors="ignore").strip()
        if line.startswith("element vertex"):
            return int(line.split()[-1])
        if line == "end_header":
            break
    raise ValueError("No vertex count in PLY header")


def find_splat_entry(zf: zipfile.ZipFile) -> tuple[str, str]:
    """Return (entry_name, format) for the splat file in the archive."""
    for name in zf.namelist():
        if name.endswith(".ply"):
            return name, "ply"
        if name.endswith(".spz"):
            return name, "spz"
    raise FileNotFoundError("No .ply or .spz entry in archive")


def find_thumb(thumbs_dir: Path, stem: str) -> Path | None:
    for ext in (".jpg", ".jpeg", ".png"):
        candidate = thumbs_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def build_manifest(
    stem: str,
    splat_entry: str,
    splat_format: str,
    num_gaussians: int,
    has_skeleton: bool,
    has_weights: bool,
    has_states: bool,
    thumb_path: Path | None,
    widget_version: str,
) -> dict:
    manifest: dict = {
        "format": "splattie",
        "formatVersion": widget_version,
        "generator": {
            "method": "lam",
            "methodVersion": "20k-siggraph2025",
            "tool": "add_manifest_to_splattie.py",
            "createdAt": datetime.now(timezone.utc).isoformat(),
        },
        "avatar": {
            "splat": {
                "file": splat_entry,
                "format": splat_format,
                "numGaussians": num_gaussians,
                "topology": "flame-20k",
            },
        },
        "animation": {
            "type": "lbs",
            "expression": {"system": "flame-pca", "basis": None},
        },
        "widget": {"config": "states.json" if has_states else "states.json"},
    }
    if has_skeleton:
        manifest["animation"]["skeleton"] = {"file": "bone_tree.json", "rig": "flame"}
    if has_weights:
        manifest["animation"]["weights"] = {"file": "lbs_weight_20k.json"}

    metadata: dict = {}
    if thumb_path is not None:
        source_hash = hashlib.sha256(thumb_path.read_bytes()).hexdigest()
        metadata["sourceImageHash"] = f"sha256:{source_hash}"
    attribution = ATTRIBUTIONS.get(stem)
    if attribution:
        metadata["attribution"] = f"Photo by {attribution} on Pexels"
        metadata["license"] = "Pexels (free license)"
    if metadata:
        manifest["metadata"] = metadata

    return manifest


def rebundle(splattie_path: Path, thumbs_dir: Path, widget_version: str) -> str:
    """Returns a short status string for logging."""
    stem = splattie_path.stem

    with zipfile.ZipFile(str(splattie_path), "r") as zf:
        names = set(zf.namelist())

        if "manifest.json" in names:
            existing = json.loads(zf.read("manifest.json").decode("utf-8"))
            if existing.get("formatVersion") == widget_version:
                return f"skip (already v{widget_version})"

        splat_entry, splat_format = find_splat_entry(zf)
        splat_bytes = zf.read(splat_entry)
        num_gaussians = count_ply_vertices_bytes(splat_bytes) if splat_format == "ply" else 0

        has_skeleton = "bone_tree.json" in names
        has_weights = "lbs_weight_20k.json" in names
        has_states = "states.json" in names

        payload: dict[str, bytes] = {name: zf.read(name) for name in zf.namelist() if not name.endswith("/")}

    thumb_path = find_thumb(thumbs_dir, stem)
    manifest = build_manifest(
        stem=stem,
        splat_entry=splat_entry,
        splat_format=splat_format,
        num_gaussians=num_gaussians,
        has_skeleton=has_skeleton,
        has_weights=has_weights,
        has_states=has_states,
        thumb_path=thumb_path,
        widget_version=widget_version,
    )

    tmp_path = splattie_path.with_suffix(splattie_path.suffix + ".tmp")
    with zipfile.ZipFile(str(tmp_path), "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps(manifest, indent=2))
        for name, data in payload.items():
            if name == "manifest.json":
                continue
            zf.writestr(name, data)
    tmp_path.replace(splattie_path)
    return f"rebundled (v{widget_version}, {num_gaussians} gaussians)"


def main() -> None:
    parser = argparse.ArgumentParser(description="Add manifest.json to existing .splattie files")
    parser.add_argument("--splatties-dir", type=str, required=True)
    parser.add_argument("--thumbs-dir", type=str, required=True)
    args = parser.parse_args()

    splatties_dir = Path(args.splatties_dir).resolve()
    thumbs_dir = Path(args.thumbs_dir).resolve()
    if not splatties_dir.is_dir():
        raise SystemExit(f"Not a directory: {splatties_dir}")
    if not thumbs_dir.is_dir():
        raise SystemExit(f"Not a directory: {thumbs_dir}")

    widget_version = read_widget_version()
    print(f"Widget version: {widget_version}")
    print(f"Splatties dir:  {splatties_dir}")
    print(f"Thumbs dir:     {thumbs_dir}\n")

    splatties = sorted(splatties_dir.glob("*.splattie"))
    if not splatties:
        print(f"No .splattie files in {splatties_dir}")
        return

    for p in splatties:
        status = rebundle(p, thumbs_dir, widget_version)
        print(f"  {p.name}: {status}")

    print(f"\nProcessed {len(splatties)} file(s).")


if __name__ == "__main__":
    main()
