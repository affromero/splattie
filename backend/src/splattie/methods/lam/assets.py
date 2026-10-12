"""Integrity checks for the fixed legacy assets consumed by vendored LAM."""

from __future__ import annotations

import argparse
import hashlib
import pickle
import tarfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import cast

LAM_ASSETS_REVISION = "284d3198e8a89cefba386ef83cb5f19034ff3910"
LAM_MODEL_REVISION = "a710cd3c40c86ffe3fc572e895ed12f3ead47289"
LAM_ARCHIVE_DIGESTS: Mapping[str, str] = {
    "LAM_assets.tar": "f8853a6678df2e08f4c27b3ed8ae625d29bc64500330b0bfebf45e64ff5af78f",
    "LAM_human_model.tar": "f8a1e10301937ac9d6169bf46d1bb9a1e73725a31e380f67d5fca3f12f92ef24",
    "thirdparties.tar": "86b5b7dafb858e4cb14764ac63317e9d7e6923ebbab1abf70b8679163f0eb339",
    "thirdparty_models.tar": "2ac5eba7c1abca0ab6c682ad900ddb8819b713506043b04874599c82572931c8",
}
FLAME_MASKS_DIGEST = "ccefbe1ac0774ff78c68caf2c627b4abc067a6555ebeb0be5d5b0812366ab492"

# Hashes were measured from thirdparty_models.tar at LAM_ASSETS_REVISION.
# Vendor readers reopen these operator-owned paths after verification. This
# authenticates downloaded assets, without claiming protection against a local
# attacker concurrently rewriting the model directory.
LAM_LEGACY_ASSET_DIGESTS: Mapping[str, str] = {
    "model_zoo/human_parametric_models/flame_vhap/uv_masks.npz": "34b7e9f2a9feb35255ffc539f15209de2d14f1a558d7cd3b8da4dde20194587a",
    "model_zoo/human_parametric_models/flame_assets/flame/flame2023.pkl": "8fb1af0db1abb51053ead8fd1f2624a63d01c9602f4a4fb4ea23bd2c82017fa0",
    "model_zoo/human_parametric_models/flame_assets/flame/FLAME_masks.pkl": "ccefbe1ac0774ff78c68caf2c627b4abc067a6555ebeb0be5d5b0812366ab492",
    "model_zoo/human_parametric_models/flame_assets/flame/landmark_embedding_with_eyes.npy": "a9050103cc98d5505beadca2f9b101acce9a78ebdd5d10cb001fa164ab7d309b",
    "model_zoo/human_parametric_models/flame_vhap/flame2023.pkl": "8fb1af0db1abb51053ead8fd1f2624a63d01c9602f4a4fb4ea23bd2c82017fa0",
    "model_zoo/human_parametric_models/flame_vhap/FLAME_masks.pkl": "ccefbe1ac0774ff78c68caf2c627b4abc067a6555ebeb0be5d5b0812366ab492",
    "model_zoo/human_parametric_models/flame_vhap/landmark_embedding_with_eyes.npy": "a9050103cc98d5505beadca2f9b101acce9a78ebdd5d10cb001fa164ab7d309b",
    "model_zoo/flame_tracking_models/matting/stylematte_synth.pt": "5ce985571e909b6677d7d25e560216fa3f620e5cd337a8382ee0799c6d9af16c",
    "model_zoo/flame_tracking_models/FaceBoxesV2.pth": "aae07fec4b62ac655508c06336662538803407852312ca5009fd93fb487d8cd7",
    "model_zoo/flame_tracking_models/68_keypoints_model.pkl": "0f8a1532772f3baaba042117ca1399bad8d4e3c959bed6a991eaaae246352b86",
    "model_zoo/flame_tracking_models/vgghead/lmks_2d.pt": "a0d0f89d3830954b6adb8d2e9e67fc656636db61c612a47b935b444f3ce69504",
    "model_zoo/flame_tracking_models/vgghead/vgg_heads_l.trcd": "18acb79c53032db11e8f502c12fdd34b5f642e9bc9041bce152c7b716c1b6f74",
}


def verify_lam_assets(lam_directory: Path, *, tracker: bool = False) -> None:
    """Reject missing or changed legacy assets before vendor model construction."""
    prefix = "model_zoo/human_parametric_models/flame_assets/flame/"
    for relative_path, expected in LAM_LEGACY_ASSET_DIGESTS.items():
        if not tracker and not relative_path.startswith(prefix):
            continue
        path = lam_directory / relative_path
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected:
            msg = f"LAM asset integrity check failed: {path}. Run scripts/setup-gpu.sh to restore pinned assets."
            raise ValueError(msg)


def load_flame_masks(path: Path) -> Mapping[str, Iterable[int]]:
    """Read the authenticated raw FLAME pickle from the exact verified bytes."""
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != FLAME_MASKS_DIGEST:
        msg = f"FLAME masks integrity check failed: {path}"
        raise ValueError(msg)
    masks = pickle.loads(content, encoding="latin1")  # noqa: S301 - Exact bytes match the pinned asset digest.
    if not isinstance(masks, Mapping) or any(not isinstance(name, str) for name in masks):
        msg = f"Invalid FLAME mask mapping: {path}"
        raise ValueError(msg)
    return cast("Mapping[str, Iterable[int]]", masks)


def extract_lam_archive(path: Path, destination: Path) -> None:
    """Authenticate a pinned LAM archive and reject unsafe extraction paths."""
    expected = LAM_ARCHIVE_DIGESTS[path.name]
    with path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
            msg = f"LAM archive integrity check failed: {path}"
            raise ValueError(msg)
        stream.seek(0)
        with tarfile.open(fileobj=stream) as archive:
            archive.extractall(destination, filter="data")


def download_lam_assets(lam_directory: Path) -> None:
    """Download immutable model snapshots and authenticate legacy archives."""
    from huggingface_hub import snapshot_download

    model_directory = lam_directory / "model_zoo/lam_models/releases/lam/lam-20k"
    snapshot_download("3DAIGC/LAM-20K", revision=LAM_MODEL_REVISION, local_dir=model_directory)
    checkpoint = model_directory / "step_045500/model.safetensors"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if not checkpoint.exists():
        checkpoint.symlink_to("../model.safetensors")

    staging = lam_directory / ".lam_assets_download"
    snapshot_download("3DAIGC/LAM-assets", revision=LAM_ASSETS_REVISION, local_dir=staging)
    for archive_name in LAM_ARCHIVE_DIGESTS:
        extract_lam_archive(staging / archive_name, lam_directory)
    verify_lam_assets(lam_directory, tracker=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and verify pinned LAM assets.")
    parser.add_argument("directory", type=Path)
    download_lam_assets(parser.parse_args().directory)
