"""Protocol for swappable asset generation methods."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt
from jaxtyping import Bool, UInt8

from splattie.types import GenerationResult, MethodCapabilities, MethodInfo


@runtime_checkable
class AssetGenerationMethod(Protocol):
    """Contract for any 3DGS asset generation backend (head, body, object).

    Adding a new method = implement this protocol + register in the registry.
    The asset type is declared via ``MethodInfo.asset_type``.
    """

    @property
    def info(self) -> MethodInfo: ...

    @property
    def capabilities(self) -> MethodCapabilities: ...

    def load(self) -> None:
        """Load model weights into GPU memory."""
        ...

    def generate(
        self,
        image: UInt8[npt.NDArray[np.uint8], "h w 3"],
        mask: Bool[npt.NDArray[np.bool_], "h w"],
    ) -> GenerationResult:
        """Generate a 3DGS asset from a segmented image."""
        ...

    def unload(self) -> None:
        """Release GPU memory."""
        ...
