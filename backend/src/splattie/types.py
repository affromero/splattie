"""Shared Pydantic models and enums for the Splattie API."""

from enum import Enum

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


class AssetType(str, Enum):
    """The kind of asset a generation method produces.

    `str`-valued so it serializes to its plain value (`"head"`) in JSON/Pydantic
    and compares equal to that string, while staying a single typed source of truth.
    """

    HEAD = "head"
    BODY = "body"
    OBJECT = "object"


class SplatFormat(str, Enum):
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
    """Result of an asset generation request."""

    model_id: str
    spz_url: str
    spz_size_bytes: int
    num_gaussians: int
    method_id: str
    rig_params_url: str


class SegmentationResult(CamelModel):
    """Result of a segmentation request."""

    mask_url: str
    preview_url: str
    bbox: tuple[int, int, int, int]
