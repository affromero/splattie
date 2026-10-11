"""Resolve the immutable LHM model snapshot used by setup and inference."""

from __future__ import annotations

import argparse
from pathlib import Path

MODEL_REPO = "3DAIGC/LHM-500M"
MODEL_REVISION = "7b9c7036404f9c21d8c4891ee4053e58c6ca2427"


def model_snapshot(vendor: Path, *, local_files_only: bool = True) -> Path:
    """Resolve the pinned snapshot, downloading only when setup requests it."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import LocalEntryNotFoundError

    try:
        snapshot = snapshot_download(
            MODEL_REPO,
            revision=MODEL_REVISION,
            cache_dir=vendor / "pretrained_models" / "huggingface",
            local_files_only=local_files_only,
        )
    except LocalEntryNotFoundError as exc:
        msg = "Pinned LHM model snapshot is missing. Run backend/scripts/setup-gpu.sh to download it."
        raise FileNotFoundError(msg) from exc
    return Path(snapshot)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("vendor", type=Path)
    args = parser.parse_args()
    model_snapshot(args.vendor, local_files_only=False)
