"""Bundle maintenance commands for existing `.splattie` assets."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from collections.abc import Mapping, MutableMapping
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from klogr import get_logger

from splattie.compression.bundle import compress_bundle_rigged, is_compressed_ply_bytes
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
        expression_basis = "expression_basis.bin" if "expression_basis.bin" in names else None
        animation = {"type": "lbs", "expression": {"system": "flame-pca", "basis": expression_basis}}
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
        already_compressed = is_compressed_ply_bytes(splat_bytes)
        expression_basis_current = True
        if existing is not None and asset_type is AssetType.head and "expression_basis.bin" in names:
            animation = existing.get("animation")
            expression = animation.get("expression") if isinstance(animation, Mapping) else None
            expression_basis_current = (
                isinstance(expression, Mapping) and expression.get("basis") == "expression_basis.bin"
            )
        already_current = (
            existing is not None
            and existing.get("formatVersion") == widget_version
            and existing.get("assetType") == asset_type.value
            and expression_basis_current
        )
        if already_current and not (compress and not already_compressed):
            return f"skip (already v{widget_version}, {asset_type.value})"
        payload = {name: zf.read(name) for name in zf.namelist() if not name.endswith("/")}

    if compress and not already_compressed and asset_type is AssetType.head:
        msg = "--compress is disabled for head bundles because FLAME topology depends on splat order"
        raise ValueError(msg)

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

    if compress and not already_compressed:
        result = compress_bundle_rigged(tmp_path, tmp_path)
        note = f", {result.size_before // 1024}KB bundle -> {result.size_after // 1024}KB rigged compressed"

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


def _subset_expression_basis(data: bytes, entry: str, idx: "np.ndarray", num_ply_verts: int) -> bytes:
    """Subset EXPH/EXPR expression-basis rows to the given vertex indices.

    Always emits EXPH float16 — the widget's basis loader accepts nothing else.
    """
    magic = data[:4]
    if magic not in (b"EXPH", b"EXPR"):
        msg = f"Unexpected magic {magic!r} in {entry}"
        raise ValueError(msg)
    dtype = "<f2" if magic == b"EXPH" else "<f4"
    num_verts, num_expr = np.frombuffer(data, dtype="<u4", count=2, offset=4).tolist()
    if num_verts != num_ply_verts:
        msg = f"{entry}: {num_verts} basis rows but PLY has {num_ply_verts} vertices"
        raise ValueError(msg)
    basis = np.frombuffer(data, dtype=dtype, offset=12).reshape(num_verts, num_expr, 3)
    subset = basis[idx].astype("<f2")
    return b"EXPH" + np.asarray([len(idx), num_expr], dtype="<u4").tobytes() + subset.tobytes()


def _select_important(vertices: "np.ndarray", max_gaussians: int) -> "np.ndarray":
    """Pick the gaussians that matter most for coverage, in original splat order.

    Uniform stride sampling tears holes in the surface: it discards large,
    opaque, load-bearing splats as readily as tiny film-grain ones, and the
    survivors' scales are tuned for the dense cloud. Rank by rendered
    contribution instead — opacity x volume (the LightGaussian pruning
    heuristic) — and keep the top scorers. Scored in log space:
    log(sigmoid(opacity_logit)) + sum of log-scales = log(alpha * volume).
    """
    fields = set(vertices.dtype.names or ())
    required = {"opacity", "scale_0", "scale_1", "scale_2"}
    missing = sorted(required - fields)
    if missing:
        msg = f"PLY is missing fields needed for importance pruning: {missing}"
        raise ValueError(msg)
    log_volume = (
        vertices["scale_0"].astype(np.float64)
        + vertices["scale_1"].astype(np.float64)
        + vertices["scale_2"].astype(np.float64)
    )
    # log(sigmoid(o)) + log-volume: product of opacity and volume in log space.
    importance = -np.logaddexp(0.0, -vertices["opacity"].astype(np.float64)) + log_volume
    if not np.isfinite(importance).all():
        msg = "PLY has non-finite opacity/scale values; refusing to rank a corrupt splat"
        raise ValueError(msg)
    keep = min(max_gaussians, len(vertices))
    top = np.argpartition(-importance, keep - 1)[:keep]
    return np.sort(top)


def _compensate_scales(vertices: "np.ndarray", factor: float) -> "np.ndarray":
    """Grow surviving splats so a sparser cloud still covers the surface.

    Scales are stored as logs, so a linear size factor is an additive shift.
    """
    out = vertices.copy()
    shift = np.float32(np.log(factor))
    for f in ("scale_0", "scale_1", "scale_2"):
        out[f] = out[f] + shift
    return out


def _load_head_bundle(input_path: Path) -> tuple[MutableMapping[str, object], MutableMapping[str, bytes], str]:
    """Read a head `.splattie` and validate it is downsample-able."""
    with zipfile.ZipFile(str(input_path), "r") as zf:
        manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
        payload = {name: zf.read(name) for name in zf.namelist() if not name.endswith("/")}

    if manifest.get("assetType") != AssetType.head.value:
        msg = f"downsample only supports head bundles, got assetType={manifest.get('assetType')!r}"
        raise ValueError(msg)
    # The manifest is authoritative for which entry is the splat — the widget
    # loads manifest.avatar.splat.file, not the first .ply in zip order.
    splat_entry = manifest["avatar"]["splat"]["file"]
    if splat_entry not in payload:
        msg = f"manifest declares splat file {splat_entry!r} but the bundle has no such entry"
        raise ValueError(msg)
    if is_compressed_ply_bytes(payload[splat_entry]):
        msg = "downsample requires an uncompressed PLY (compressed chunks cannot be subset)"
        raise ValueError(msg)
    return manifest, payload, splat_entry


def downsample(
    input_path: Path,
    output: Path,
    max_gaussians: int = 8000,
    *,
    keep_expression_basis: bool = False,
    scale_compensation: float = 1.0,
) -> None:
    """Create a reduced-gaussian head `.splattie` for low-bandwidth targets.

    Keeps the highest-contribution gaussians (opacity x volume) and applies the
    same index subset to the PLY vertices, the dense LBS weight rows, and the
    expression-basis rows, preserving the splat-order correspondence FLAME
    rigging depends on. The expression basis is dropped by default: it is by
    far the largest entry
    and only drives FLAME expression offsets (editor sliders); autoBlink is
    SDF-based in the widget and unaffected.

    Args:
        input_path: Source `.splattie` head bundle (uncompressed PLY).
        output: Where to write the downsampled bundle.
        max_gaussians: Target gaussian count.
        keep_expression_basis: Keep (and subset) `expression_basis.bin` instead
            of dropping it.
        scale_compensation: Linear size factor applied to surviving splats.
            Importance pruning keeps surfaces closed at moderate ratios, so
            this defaults to 1.0 (off); raise it (e.g. (n/kept)^(1/3)) only if
            an extreme ratio tears holes.

    """
    from tempfile import TemporaryDirectory

    from splattie.methods.object.bundle import read_binary_ply, write_binary_ply

    if max_gaussians < 1:
        msg = f"max_gaussians must be >= 1, got {max_gaussians}"
        raise ValueError(msg)
    if not (np.isfinite(scale_compensation) and scale_compensation > 0):
        msg = f"scale_compensation must be a finite positive factor, got {scale_compensation}"
        raise ValueError(msg)

    manifest, payload, splat_entry = _load_head_bundle(input_path)

    with TemporaryDirectory() as tmp:
        # Fixed name — the archive-controlled entry name must not touch paths.
        ply_path = Path(tmp) / "splat.ply"
        ply_path.write_bytes(payload[splat_entry])
        ply = read_binary_ply(ply_path)
        n = len(ply.vertices)
        idx = _select_important(ply.vertices, max_gaussians)

        subset = ply.vertices[idx]
        if scale_compensation != 1.0:
            subset = _compensate_scales(subset, scale_compensation)
        write_binary_ply(ply_path, type(ply)(vertices=subset, properties=ply.properties))
        payload[splat_entry] = ply_path.read_bytes()

    weights_entry = (manifest.get("animation", {}).get("weights") or {}).get("file")
    if not weights_entry or weights_entry not in payload:
        msg = f"head bundle is missing its LBS weights entry ({weights_entry!r})"
        raise ValueError(msg)
    weights = json.loads(payload[weights_entry].decode("utf-8"))
    if not isinstance(weights, list) or len(weights) != n:
        msg = f"{weights_entry}: expected a dense list of {n} weight rows"
        raise ValueError(msg)
    rounded = [[round(w, 5) for w in weights[i]] for i in idx.tolist()]
    payload[weights_entry] = json.dumps(rounded).encode("utf-8")

    basis_entry = (manifest.get("animation", {}).get("expression") or {}).get("basis")
    if basis_entry:
        if keep_expression_basis:
            payload[basis_entry] = _subset_expression_basis(payload[basis_entry], basis_entry, idx, n)
        else:
            del payload[basis_entry]
            manifest["animation"]["expression"]["basis"] = None

    manifest["avatar"]["splat"]["numGaussians"] = len(idx)

    with zipfile.ZipFile(str(output), "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps(manifest, indent=2))
        for name, data in payload.items():
            if name != "manifest.json":
                zf.writestr(name, data)

    before, after = input_path.stat().st_size, output.stat().st_size
    logger.info(
        f"{output.name}: {n} -> {len(idx)} gaussians, "
        f"{before // 1024} KB -> {after // 1024} KB"
        f"{'' if keep_expression_basis or not basis_entry else ' (expression basis dropped)'}"
    )
