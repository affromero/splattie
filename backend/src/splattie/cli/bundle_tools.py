"""Bundle maintenance commands for existing `.splattie` assets."""

from __future__ import annotations

import hashlib
import io
import json
import tempfile
import zipfile
from collections.abc import Mapping, MutableMapping
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from klogr import get_logger

from splattie.compression.compressed_ply import compress_ply
from splattie.methods.bundle_common import read_widget_version
from splattie.types import AssetType

logger = get_logger()

ATTRIBUTIONS = {
    "3762763": "Shiny Diamond",
    "3754430": "TUBARONES PHOTOGRAPHY",
    "7705909": "ShotPot",
    "8727488": "Tima Miroshnichenko",
    "8727554": "Tima Miroshnichenko",
    "35466969": "Daniel Hoffman Jackson",
}


def count_ply_vertices_bytes(data: bytes) -> int:
    """Count vertices from raw PLY bytes."""
    buf = io.BytesIO(data)
    for raw in buf:
        line = raw.decode("ascii", errors="ignore").strip()
        if line.startswith("element vertex"):
            return int(line.split()[-1])
        if line == "end_header":
            break
    msg = "No vertex count in PLY header"
    raise ValueError(msg)


def find_splat_entry(zf: zipfile.ZipFile) -> tuple[str, str]:
    """Return (entry_name, format) for the splat file in the archive."""
    for name in zf.namelist():
        if name.endswith(".ply"):
            return name, "ply"
        if name.endswith(".spz"):
            return name, "spz"
    msg = "No .ply or .spz entry in archive"
    raise FileNotFoundError(msg)


def find_thumb(thumbs_dir: Path, stem: str) -> Path | None:
    """Return the source thumbnail for a stem, trying common extensions."""
    for ext in (".jpg", ".jpeg", ".png"):
        candidate = thumbs_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def _animation_manifest(names: set[str], asset_type: AssetType) -> Mapping[str, object]:
    if asset_type is AssetType.body:
        animation: MutableMapping[str, object] = {"type": "lbs", "expression": None}
        if "skeleton.json" in names:
            animation["skeleton"] = {"file": "skeleton.json", "rig": "smplx"}
        if "lbs_weights.json" in names:
            animation["weights"] = {"file": "lbs_weights.json"}
        return animation

    if asset_type in (AssetType.object, AssetType.quadruped_mammal):
        # quadruped bundles are object-format (LBS skeleton.json + lbs_weights.bin), built
        # via build_object_splattie; the animation manifest is identical to objects.
        animation = {"type": "lbs", "expression": None}
        if "skeleton.json" in names:
            animation["skeleton"] = {"file": "skeleton.json", "rig": "puppeteer-object"}
        if "lbs_weights.bin" in names:
            animation["weights"] = {"file": "lbs_weights.bin", "format": "lbsw-v1"}
    else:
        animation = {"type": "lbs", "expression": {"system": "flame-pca", "basis": None}}
        if "bone_tree.json" in names:
            animation["skeleton"] = {"file": "bone_tree.json", "rig": "flame"}
        if "lbs_weight_20k.json" in names:
            animation["weights"] = {"file": "lbs_weight_20k.json"}
    return animation


def _topology(asset_type: AssetType) -> str:
    if asset_type is AssetType.body:
        return "smplx-voxel"
    if asset_type in (AssetType.object, AssetType.quadruped_mammal):
        return "object-auto"
    return "flame-20k"


def _generator_method(asset_type: AssetType) -> str:
    if asset_type is AssetType.body:
        return "lhm"
    if asset_type is AssetType.object:
        return "trellis-puppeteer"
    if asset_type is AssetType.quadruped_mammal:
        return "trellis-smal-quadruped"
    return "lam"


def build_legacy_manifest(
    *,
    stem: str,
    splat_entry: str,
    splat_format: str,
    num_gaussians: int,
    thumb_path: Path | None,
    widget_version: str,
    asset_type: AssetType,
    names: set[str],
) -> MutableMapping[str, object]:
    """Build the .splattie manifest payload for one legacy bundle."""
    manifest: MutableMapping[str, object] = {
        "format": "splattie",
        "formatVersion": widget_version,
        "assetType": asset_type.value,
        "generator": {
            "method": _generator_method(asset_type),
            "methodVersion": "20k-siggraph2025" if asset_type is AssetType.head else None,
            "tool": "splattie add-manifest",
            "createdAt": datetime.now(UTC).isoformat(),
        },
        "avatar": {
            "splat": {
                "file": splat_entry,
                "format": splat_format,
                "numGaussians": num_gaussians,
                "topology": _topology(asset_type),
            },
        },
        "animation": _animation_manifest(names, asset_type),
        "widget": {"config": "states.json"},
    }
    if manifest["generator"]["methodVersion"] is None:
        del manifest["generator"]["methodVersion"]

    metadata: MutableMapping[str, object] = {}
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


def rebundle(
    splattie_path: Path,
    thumbs_dir: Path,
    widget_version: str,
    asset_type: AssetType,
    *,
    compress: bool = False,
) -> str:
    """Re-bundle one .splattie with a current manifest; return a status string."""
    stem = splattie_path.stem
    note = ""

    with zipfile.ZipFile(str(splattie_path), "r") as zf:
        names = set(zf.namelist())
        existing = json.loads(zf.read("manifest.json").decode("utf-8")) if "manifest.json" in names else None
        splat_entry, splat_format = find_splat_entry(zf)
        splat_bytes = zf.read(splat_entry)
        num_gaussians = count_ply_vertices_bytes(splat_bytes)
        already_compressed = b"packed_position" in splat_bytes[:1024]
        already_current = (
            existing is not None
            and existing.get("formatVersion") == widget_version
            and existing.get("assetType") == asset_type.value
        )
        if already_current and not (compress and not already_compressed):
            return f"skip (already v{widget_version}, {asset_type.value})"
        payload = {name: zf.read(name) for name in zf.namelist() if not name.endswith("/")}

    if compress and not already_compressed:
        with tempfile.TemporaryDirectory() as td:
            ply_tmp = Path(td) / f"{stem}.ply"
            comp_tmp = Path(td) / f"{stem}.compressed.ply"
            ply_tmp.write_bytes(splat_bytes)
            compress_ply(ply_tmp, comp_tmp)
            comp_bytes = comp_tmp.read_bytes()
        payload[splat_entry] = comp_bytes
        note = f", {len(splat_bytes) // 1024}KB ply -> {len(comp_bytes) // 1024}KB compressed.ply"

    manifest = build_legacy_manifest(
        stem=stem,
        splat_entry=splat_entry,
        splat_format=splat_format,
        num_gaussians=num_gaussians,
        thumb_path=find_thumb(thumbs_dir, stem),
        widget_version=widget_version,
        asset_type=asset_type,
        names=set(payload),
    )

    tmp_path = splattie_path.with_suffix(splattie_path.suffix + ".tmp")
    with zipfile.ZipFile(str(tmp_path), "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps(manifest, indent=2))
        for name, data in payload.items():
            if name != "manifest.json":
                zf.writestr(name, data)
    tmp_path.replace(splattie_path)
    return f"rebundled (v{widget_version}, {num_gaussians} gaussians, format={splat_format}{note})"


def add_manifest(
    splatties_dir: Path,
    thumbs_dir: Path,
    asset_type: AssetType = AssetType.head,
    *,
    compress: bool = False,
) -> None:
    """Add or update manifest.json in every .splattie in a directory.

    Args:
        splatties_dir: Directory of .splattie bundles to process.
        thumbs_dir: Directory of source thumbnails (for sourceImageHash + attribution).
        asset_type: Asset type recorded in the manifest (head/body/object).
        compress: Compress PLY payloads to compressed PLY before re-bundling.

    """
    splatties_dir = splatties_dir.resolve()
    thumbs_dir = thumbs_dir.resolve()
    if not splatties_dir.is_dir():
        msg = f"Not a directory: {splatties_dir}"
        raise SystemExit(msg)
    if not thumbs_dir.is_dir():
        msg = f"Not a directory: {thumbs_dir}"
        raise SystemExit(msg)

    widget_version = read_widget_version()
    logger.info(f"Widget version: {widget_version}")
    logger.info(f"Splatties dir:  {splatties_dir}")
    logger.info(f"Thumbs dir:     {thumbs_dir}")

    splatties = sorted(splatties_dir.glob("*.splattie"))
    if not splatties:
        logger.info(f"No .splattie files in {splatties_dir}")
        return

    for path in splatties:
        status = rebundle(path, thumbs_dir, widget_version, asset_type, compress=compress)
        logger.info(f"  {path.name}: {status}")

    logger.info(f"Processed {len(splatties)} file(s).")


def shrink_expression_basis(basis_path: Path, output: Path | None = None) -> None:
    """Transcode an expression-basis .bin from float32 EXPR to float16 EXPH.

    Args:
        basis_path: Path to the float32 (EXPR) expression_basis.bin.
        output: Where to write the float16 (EXPH) .bin. Defaults to overwriting basis_path.

    """
    out = output or basis_path
    data = basis_path.read_bytes()
    magic = data[:4]
    if magic == b"EXPH":
        logger.info(f"{basis_path.name} is already float16 (EXPH); nothing to do.")
        return
    if magic != b"EXPR":
        msg = f"Unexpected magic {magic!r} in {basis_path} (expected EXPR)"
        raise ValueError(msg)

    num_verts, num_expr = np.frombuffer(data, dtype="<u4", count=2, offset=4).tolist()
    basis32 = np.frombuffer(data, dtype="<f4", offset=12).reshape(num_verts, num_expr, 3)
    basis16 = basis32.astype(np.float16)

    with out.open("wb") as f:
        f.write(b"EXPH")
        f.write(np.asarray([num_verts, num_expr], dtype="<u4").tobytes())
        f.write(basis16.tobytes())

    before, after = len(data), out.stat().st_size
    logger.info(
        f"{out.name}: {before // 1024} KB (f32) -> {after // 1024} KB (f16), {num_verts} verts x {num_expr} expr"
    )

    json_path = basis_path.with_suffix(".json")
    if json_path.exists():
        meta = json.loads(json_path.read_text())
        meta["bytes"] = int(after)
        meta["format"] = "float16_le, shape (num_vertices, num_expressions, 3)"
        out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
        logger.info(f"Updated sidecar: {out.with_suffix('.json').name}")
