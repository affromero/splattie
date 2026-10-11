"""Expression exports restrict supplied shape checkpoints and authenticate masks."""

from __future__ import annotations

import pickle
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from splattie.cli.flame_exports import export_expression_basis  # noqa: E402
from tests.methods.test_lam_assets import UntrustedMask  # noqa: E402


def test_shape_parameter_checkpoint_rejects_executable_pickle_before_loading_flame(tmp_path: Path) -> None:
    marker = tmp_path / "executed"
    checkpoint = tmp_path / "shape.pt"
    torch.save(UntrustedMask(marker), checkpoint)

    with pytest.raises(pickle.UnpicklingError):
        export_expression_basis(tmp_path / "basis.bin", shape_param=checkpoint, device="cpu")

    assert not marker.exists()
