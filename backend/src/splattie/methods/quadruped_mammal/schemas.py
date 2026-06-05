"""Typed payloads passed between the quadruped pipeline stages.

Tensor/array carriers are Pydantic dataclasses (``arbitrary_types_allowed``), mirroring how
``methods/object/bundle.py`` models ``BinaryPly``. Scalar diagnostics are a ``CamelModel`` so
they can be logged / surfaced. No bare ``dict``-typed fields (CLAUDE.md rule 10).
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from splattie.types import CamelModel

_ARRAY_CONFIG = ConfigDict(arbitrary_types_allowed=True, frozen=True)


class NotAQuadrupedMammalError(RuntimeError):
    """The input isn't a recognizable quadruped mammal (SuperAnimal/SMAL gate failed).

    Raised instead of silently shipping a garbage rig — the category has no fallback path.
    """


@dataclass(config=_ARRAY_CONFIG, kw_only=True)
class Keypoints3D:
    """Triangulated 3D SuperAnimal landmarks in the splat's world frame."""

    bodyparts: list[str]
    positions: np.ndarray  # (N, 3) float32
    support: np.ndarray  # (N,) view-count per landmark
    mean_confidence: float

    def lookup(self, *, min_support: int = 3) -> Mapping[str, np.ndarray]:
        """Return ``name -> xyz`` for well-triangulated landmarks (local compute helper)."""
        return {
            name: self.positions[i]
            for i, name in enumerate(self.bodyparts)
            if self.support[i] >= min_support and np.isfinite(self.positions[i]).all()
        }


class FitDiagnostics(CamelModel):
    """Scalar quality signals from the SMAL fit; drives the mammal-detection gate + logs."""

    chamfer: float
    anchor_rms: float
    lr_residual: float
    lr_swap: bool
    n_anchors: int
    triangulated_count: int
    mean_keypoint_confidence: float
    shape_norm: float  # |betas| — extreme for out-of-family shapes (e.g. elephant); see method gate


@dataclass(config=_ARRAY_CONFIG, kw_only=True)
class QuadrupedFit:
    """Optimised SMAL parameters (SMAL-space) plus fit diagnostics."""

    betas: torch.Tensor
    pose: torch.Tensor
    scale: float
    trans: torch.Tensor
    diagnostics: FitDiagnostics
