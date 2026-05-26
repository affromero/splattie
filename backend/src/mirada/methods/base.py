"""Protocol for swappable head generation methods."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from mirada.types import GenerationResult, MethodCapabilities, MethodInfo


@runtime_checkable
class HeadGenerationMethod(Protocol):
    """Contract for any 3DGS head generation backend.

    Adding a new method = implement this protocol + register in the registry.
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
        image: NDArray[np.uint8],
        mask: NDArray[np.bool_],
    ) -> GenerationResult:
        """Generate a 3DGS head from a segmented image."""
        ...

    def unload(self) -> None:
        """Release GPU memory."""
        ...
