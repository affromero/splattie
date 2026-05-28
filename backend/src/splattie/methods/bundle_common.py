"""Shared `.splattie` bundler — the single source of truth for the bundle shape.

Both the demo batch builder (`scripts/generate_splattie_batch.py`) and the LAM
API path (`methods/lam/method.py`) call this, so every served `.splattie` has an
identical, widget-loadable shape: `manifest.json` + splat file + rig files
(skeleton + weights) + `states.json`. The LHM body path (`methods/lhm/bundle.py`)
reuses `build_manifest` / `bundle_splattie` with body-flavored inputs.

The widget loader (`packages/splattie-widget/src/SplatWidget.ts`) requires the
manifest plus every file it references — a bundle missing any of them fails to
load. Keeping one bundler guarantees the API path and the batch demos stay in
lockstep.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from splattie.types import AssetType, SplatFormat

# methods/bundle_common.py -> methods -> splattie -> src -> backend -> repo root
REPO_ROOT = Path(__file__).resolve().parents[4]
WIDGET_PKG_JSON = REPO_ROOT / "packages" / "splattie-widget" / "package.json"
WIDGET_PUBLIC = REPO_ROOT / "packages" / "splattie-widget" / "public"


@dataclass(frozen=True)
class RigSpec:
    """The rig-specific shape of a `.splattie` for one asset type.

    Captures the skeleton/weights filenames, splat container format, topology
    label and expression block so they live as named data instead of magic
    strings scattered through manifest builders. One instance per asset type
    (`HEAD_RIG` here; the LHM body path defines its own in 1.C).
    """

    rig: str
    topology: str
    splat_format: SplatFormat
    skeleton_file: str
    weights_file: str
    expression: dict | None


# The FLAME-20k head rig. Skeleton + weights are canonical (identical for every
# head) and live in the shared widget public dir.
HEAD_RIG = RigSpec(
    rig="flame",
    topology="flame-20k",
    splat_format=SplatFormat.PLY,
    skeleton_file="bone_tree.json",
    weights_file="lbs_weight_20k.json",
    expression={"system": "flame-pca", "basis": None},
)

# Default widget states baked into a head `.splattie`. Mirrors the head branch of
# the widget's `createDefaultConfig('head')` (StateConfig.ts). Kept here because the
# Python bundler cannot import the TS source.
DEFAULT_STATES_HEAD: dict = {
    "defaults": {
        "camera": {"theta": 0, "phi": 75, "radius": 0.5, "fov": 60},
        "autoBlink": {"interval": [2000, 7000], "duration": 150},
    },
    "states": {
        "idle": {
            "ghost": {"amplitude": 0.003, "frequency": 0.4, "wobble": 0.2},
            "expression": {},
            "camera": {"theta": 0, "phi": 75, "radius": 0.5, "fov": 60},
            "rotation": [0, 0, 0],
            "tracking": {"eyes": 1.0, "head": 0.1},
        },
        "hover": {
            "ghost": {"amplitude": 0.005, "frequency": 0.6, "wobble": 0.4},
            "expression": {},
            "camera": {"theta": 0, "phi": 75, "radius": 0.45, "fov": 60},
            "rotation": [-2, 0, -1],
            "tracking": {"eyes": 1.0, "head": 0.3},
        },
        "click": {
            "ghost": {"amplitude": 0.001, "frequency": 1.0, "wobble": 0.1},
            "expression": {},
            "camera": {"theta": 0, "phi": 70, "radius": 0.4, "fov": 65},
            "rotation": [3, 0, 0],
            "tracking": {"eyes": 0.5, "head": 0.0},
        },
    },
    "transitions": {
        "idle->hover": {"duration": 0.3, "easing": "ease-out"},
        "hover->idle": {"duration": 0.5, "easing": "ease-in"},
        "*->click": {"duration": 0.1, "easing": "snap"},
    },
}


def read_widget_version() -> str:
    """Read the widget package version that drives the manifest formatVersion."""
    return json.loads(WIDGET_PKG_JSON.read_text())["version"]


def count_ply_vertices(ply_path: Path) -> int:
    """Read the `element vertex` count from a PLY header."""
    with ply_path.open("rb") as f:
        for raw in f:
            line = raw.decode("ascii", errors="ignore").strip()
            if line.startswith("element vertex"):
                return int(line.split()[-1])
            if line == "end_header":
                break
    msg = f"No vertex count in PLY header: {ply_path}"
    raise ValueError(msg)


def build_manifest(
    *,
    splat_filename: str,
    num_gaussians: int,
    widget_version: str,
    asset_type: AssetType,
    rig: RigSpec,
    generator_method: str = "lam",
    generator_method_version: str = "20k-siggraph2025",
    generator_tool: str,
    source_image_path: Path | None = None,
) -> dict:
    """Build a `.splattie` manifest from an asset type and its `RigSpec`."""
    manifest: dict = {
        "format": "splattie",
        "formatVersion": widget_version,
        "assetType": AssetType(asset_type).value,
        "generator": {
            "method": generator_method,
            "methodVersion": generator_method_version,
            "tool": generator_tool,
            "createdAt": datetime.now(timezone.utc).isoformat(),
        },
        "avatar": {
            "splat": {
                "file": splat_filename,
                "format": rig.splat_format.value,
                "numGaussians": num_gaussians,
                "topology": rig.topology,
            },
        },
        "animation": {
            "type": "lbs",
            "skeleton": {"file": rig.skeleton_file, "rig": rig.rig},
            "weights": {"file": rig.weights_file},
            "expression": rig.expression,
        },
        "widget": {"config": "states.json"},
    }
    if source_image_path is not None:
        source_hash = hashlib.sha256(source_image_path.read_bytes()).hexdigest()
        manifest["metadata"] = {"sourceImageHash": f"sha256:{source_hash}"}
    return manifest


def bundle_splattie(
    *,
    output_path: Path,
    splat_path: Path,
    manifest: dict,
    states: dict,
    rig_files: dict[str, Path] | None = None,
) -> Path:
    """Write a widget-loadable `.splattie` ZIP.

    The splat is written under ``manifest.avatar.splat.file``. Skeleton and weights
    are written under their manifest filenames; their sources come from ``rig_files``
    (arcname -> path) when provided (the body path passes generated files), otherwise
    they are resolved from the shared widget public dir (the head path, where the
    FLAME rig is canonical and identical for every head).
    """
    splat_arc = manifest["avatar"]["splat"]["file"]
    referenced: list[tuple[str, Path]] = [(splat_arc, splat_path)]

    def _resolve(arc: str) -> Path:
        if rig_files is not None and arc in rig_files:
            return rig_files[arc]
        return WIDGET_PUBLIC / arc

    skeleton = manifest["animation"].get("skeleton")
    if skeleton:
        referenced.append((skeleton["file"], _resolve(skeleton["file"])))
    weights = manifest["animation"].get("weights")
    if weights:
        referenced.append((weights["file"], _resolve(weights["file"])))

    for arc_name, src in referenced:
        if not src.exists():
            msg = f"Manifest references {arc_name!r} but source {src} does not exist"
            raise FileNotFoundError(msg)

    with zipfile.ZipFile(str(output_path), "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps(manifest, indent=2))
        for arc_name, src in referenced:
            zf.write(str(src), arc_name)
        zf.writestr(manifest["widget"]["config"], json.dumps(states, indent=2))

    return output_path
