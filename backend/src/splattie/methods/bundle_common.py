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
from collections.abc import Mapping, MutableMapping
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ConfigDict, Field, TypeAdapter
from pydantic.dataclasses import dataclass

from splattie.types import AssetType, SplatFormat

# methods/bundle_common.py -> methods -> splattie -> src -> backend -> repo root
REPO_ROOT = Path(__file__).resolve().parents[4]
WIDGET_PKG_JSON = REPO_ROOT / "packages" / "splattie-widget" / "package.json"
WIDGET_PUBLIC = REPO_ROOT / "packages" / "splattie-widget" / "public"


@dataclass(config=ConfigDict(frozen=True), kw_only=True)
class RigExpressionSpec:
    """Manifest expression metadata for expression-capable rigs."""

    system: str
    basis: object | None


@dataclass(config=ConfigDict(frozen=True), kw_only=True)
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
    expression: RigExpressionSpec | None


# The FLAME-20k head rig. Skeleton + weights are canonical (identical for every
# head) and live in the shared widget public dir.
HEAD_RIG = RigSpec(
    rig="flame",
    topology="flame-20k",
    splat_format=SplatFormat.PLY,
    skeleton_file="bone_tree.json",
    weights_file="lbs_weight_20k.json",
    expression=RigExpressionSpec(system="flame-pca", basis=None),
)

_PYDANTIC_CONFIG = ConfigDict(extra="forbid", populate_by_name=True)


@dataclass(config=_PYDANTIC_CONFIG, kw_only=True)
class HeadCameraConfig:
    """Widget camera config serialized into head states.json."""

    theta: float
    phi: float
    radius: float
    fov: float


@dataclass(config=_PYDANTIC_CONFIG, kw_only=True)
class HeadAutoBlinkConfig:
    """Head auto-blink defaults."""

    interval: tuple[int, int]
    duration: int


@dataclass(config=_PYDANTIC_CONFIG, kw_only=True)
class HeadGhostConfig:
    """Subtle idle motion config."""

    amplitude: float
    frequency: float
    wobble: float


@dataclass(config=_PYDANTIC_CONFIG, kw_only=True)
class HeadExpressionConfig:
    """Per-state FLAME expression coefficients."""


@dataclass(config=_PYDANTIC_CONFIG, kw_only=True)
class HeadTrackingConfig:
    """Head widget pointer tracking gains."""

    eyes: float
    head: float


@dataclass(config=_PYDANTIC_CONFIG, kw_only=True)
class HeadStateDefinition:
    """One head interaction state."""

    ghost: HeadGhostConfig
    expression: HeadExpressionConfig
    camera: HeadCameraConfig
    rotation: tuple[float, float, float]
    tracking: HeadTrackingConfig


@dataclass(config=_PYDANTIC_CONFIG, kw_only=True)
class HeadStateSet:
    """Head widget states."""

    idle: HeadStateDefinition
    hover: HeadStateDefinition
    click: HeadStateDefinition


@dataclass(config=_PYDANTIC_CONFIG, kw_only=True)
class HeadTransitionConfig:
    """Widget transition config."""

    duration: float
    easing: str


@dataclass(config=_PYDANTIC_CONFIG, kw_only=True)
class HeadTransitions:
    """Head state transition configs."""

    idle_hover: HeadTransitionConfig
    hover_idle: HeadTransitionConfig
    any_click: HeadTransitionConfig

    def jsonable(self) -> Mapping[str, object]:
        """Return the widget's transition-key shape."""
        return {
            "idle->hover": TypeAdapter(HeadTransitionConfig).dump_python(self.idle_hover, mode="json"),
            "hover->idle": TypeAdapter(HeadTransitionConfig).dump_python(self.hover_idle, mode="json"),
            "*->click": TypeAdapter(HeadTransitionConfig).dump_python(self.any_click, mode="json"),
        }


@dataclass(config=_PYDANTIC_CONFIG, kw_only=True)
class HeadWidgetDefaults:
    """Head widget default config."""

    camera: HeadCameraConfig
    auto_blink: HeadAutoBlinkConfig = Field(..., alias="autoBlink")


@dataclass(config=_PYDANTIC_CONFIG, kw_only=True)
class HeadWidgetConfig:
    """Full head states.json payload."""

    defaults: HeadWidgetDefaults
    states: HeadStateSet
    transitions: HeadTransitions

    def jsonable(self) -> Mapping[str, object]:
        """Return the widget states.json shape."""
        payload = TypeAdapter(HeadWidgetConfig).dump_python(self, mode="json", by_alias=True)
        payload["transitions"] = self.transitions.jsonable()
        return payload


# Default widget states baked into a head `.splattie`. Mirrors the head branch of
# the widget's `createDefaultConfig('head')` (StateConfig.ts). Kept here because the
# Python bundler cannot import the TS source.
HEAD_CAMERA = HeadCameraConfig(theta=0, phi=75, radius=0.5, fov=60)
DEFAULT_STATES_HEAD = HeadWidgetConfig(
    defaults=HeadWidgetDefaults(
        camera=HEAD_CAMERA,
        auto_blink=HeadAutoBlinkConfig(interval=(2000, 7000), duration=150),
    ),
    states=HeadStateSet(
        idle=HeadStateDefinition(
            ghost=HeadGhostConfig(amplitude=0.003, frequency=0.4, wobble=0.2),
            expression=HeadExpressionConfig(),
            camera=HEAD_CAMERA,
            rotation=(0, 0, 0),
            tracking=HeadTrackingConfig(eyes=1.0, head=0.1),
        ),
        hover=HeadStateDefinition(
            ghost=HeadGhostConfig(amplitude=0.005, frequency=0.6, wobble=0.4),
            expression=HeadExpressionConfig(),
            camera=HeadCameraConfig(theta=0, phi=75, radius=0.45, fov=60),
            rotation=(-2, 0, -1),
            tracking=HeadTrackingConfig(eyes=1.0, head=0.3),
        ),
        click=HeadStateDefinition(
            ghost=HeadGhostConfig(amplitude=0.001, frequency=1.0, wobble=0.1),
            expression=HeadExpressionConfig(),
            camera=HeadCameraConfig(theta=0, phi=70, radius=0.4, fov=65),
            rotation=(3, 0, 0),
            tracking=HeadTrackingConfig(eyes=0.5, head=0.0),
        ),
    ),
    transitions=HeadTransitions(
        idle_hover=HeadTransitionConfig(duration=0.3, easing="ease-out"),
        hover_idle=HeadTransitionConfig(duration=0.5, easing="ease-in"),
        any_click=HeadTransitionConfig(duration=0.1, easing="snap"),
    ),
)


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
) -> MutableMapping[str, object]:
    """Build a `.splattie` manifest from an asset type and its `RigSpec`."""
    expression = (
        TypeAdapter(RigExpressionSpec).dump_python(rig.expression, mode="json") if rig.expression is not None else None
    )
    manifest: MutableMapping[str, object] = {
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
            "expression": expression,
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
    manifest: Mapping[str, object],
    states: Mapping[str, object],
    rig_files: Mapping[str, Path] | None = None,
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
