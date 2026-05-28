"""Shared Pydantic models and enums for the Splattie API."""

from enum import Enum

from pydantic import BaseModel


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


class MethodInfo(BaseModel):
    """Metadata about an asset generation method."""

    id: str
    name: str
    description: str
    paper_url: str
    repo_url: str
    asset_type: AssetType


class MethodCapabilities(BaseModel):
    """What an asset generation method can do."""

    supports_single_image: bool
    supports_expression: bool
    max_output_gaussians: int
    typical_inference_seconds: float


class GenerationResult(BaseModel):
    """Result of an asset generation request."""

    model_id: str
    spz_url: str
    spz_size_bytes: int
    num_gaussians: int
    method_id: str
    rig_params_url: str


class SegmentationResult(BaseModel):
    """Result of a segmentation request."""

    mask_url: str
    preview_url: str
    bbox: tuple[int, int, int, int]
