"""Shared Pydantic models and enums for the Splattie API."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel


class CamelModel(BaseModel):
    """API base model: serialize camelCase for the TS frontend, accept snake_case too.

    `populate_by_name` keeps Python-side construction (`MethodInfo(asset_type=...)`)
    working, while `model_dump(by_alias=True)` / FastAPI responses emit `assetType`.
    `protected_namespaces=()` allows `model_id` without Pydantic's `model_` warning.
    """

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        protected_namespaces=(),
    )


class AssetType(StrEnum):
    """The kind of asset a generation method produces.

    `str`-valued so it serializes to its plain value (`"head"`) in JSON/Pydantic
    and compares equal to that string, while staying a single typed source of truth.
    """

    head = "head"
    body = "body"
    object = "object"
    quadruped_mammal = "quadruped_mammal"


class SplatFormat(StrEnum):
    """On-disk gaussian-splat container format inside a `.splattie` bundle."""

    PLY = "ply"
    SPZ = "spz"


class MethodInfo(CamelModel):
    """Metadata about an asset generation method."""

    id: str
    name: str
    description: str
    paper_url: str
    repo_url: str
    asset_type: AssetType


class MethodCapabilities(CamelModel):
    """What an asset generation method can do."""

    supports_single_image: bool
    supports_expression: bool
    max_output_gaussians: int
    typical_inference_seconds: float


class GenerationResult(CamelModel):
    """Result of an asset generation request.

    The asset is a single `.splattie` zip bundle (manifest + splat file + rig files);
    the widget loads it from one URL and unpacks the rig itself, so there is no
    separate rig-params URL.
    """

    model_id: str
    splattie_url: str
    splattie_size_bytes: int
    num_gaussians: int
    method_id: str


class SegmentationResult(CamelModel):
    """Result of a segmentation request."""

    mask_url: str
    preview_url: str
    bbox: tuple[int, int, int, int]
