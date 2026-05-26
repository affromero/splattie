"""Shared Pydantic models for the Mirada API."""

from pydantic import BaseModel


class MethodInfo(BaseModel):
    """Metadata about a head generation method."""

    id: str
    name: str
    description: str
    paper_url: str
    repo_url: str


class MethodCapabilities(BaseModel):
    """What a head generation method can do."""

    supports_single_image: bool
    supports_expression: bool
    max_output_gaussians: int
    typical_inference_seconds: float


class GenerationResult(BaseModel):
    """Result of a head generation request."""

    model_id: str
    spz_url: str
    spz_size_bytes: int
    num_gaussians: int
    method_id: str
    flame_params_url: str


class SegmentationResult(BaseModel):
    """Result of a segmentation request."""

    mask_url: str
    preview_url: str
    bbox: tuple[int, int, int, int]
