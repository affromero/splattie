"""Resolve and authenticate the fixed LHM models used by setup and inference."""

from __future__ import annotations

import argparse
import hashlib
import tarfile
import tempfile
import urllib.request
from pathlib import Path

MODEL_REPO = "3DAIGC/LHM-500M"
MODEL_REVISION = "7b9c7036404f9c21d8c4891ee4053e58c6ca2427"
PRIOR_REPO = "3DAIGC/LHM"
PRIOR_REVISION = "5730e2d4559091657c3cd3fe248ced49efb1607f"
PRIOR_ARCHIVE_DIGEST = "b3b4d9e34ce41f96a92420fadb5addb5f62ceaf2f64d99f8afa9acf955498fa6"
DENSE_POINTS_PATH = "pretrained_models/dense_sample_points/1_20000.ply"
DENSE_POINTS_DIGEST = "088b1a919b794fb8486aa72ebc19ef4009b6cf92b2ac61ecfc21fc5940e25732"
DENSE_POINTS_URL = "https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/data/LHM/1_20000.ply"

# These operator-owned paths are reopened by canonical vendor readers after
# verification. Concurrent local rewrites of the model directory are outside
# this download-integrity boundary, as with the LAM assets.
REQUIRED_PRIOR_DIGESTS = {
    "pretrained_models/human_model_files/smplx/SMPLX_NEUTRAL.npz": "376021446ddc86e99acacd795182bbef903e61d33b76b9d8b359c2b0865bd992",
    "pretrained_models/human_model_files/smplx/SMPLX_MALE.npz": "ab318e3f37d2bfaae26abf4e6fab445c2a610e1d63714794d60379cc263bc2a5",
    "pretrained_models/human_model_files/smplx/SMPLX_FEMALE.npz": "b2a3686c9d6d218ff6822fba411c607a3c8125a70af340f384ce68bebecabe0e",
    "pretrained_models/human_model_files/smplx/MANO_SMPLX_vertex_ids.pkl": "e5abe70b6574de25470475091e8008314a5b90127eb48c3e63bfa0adf8c04dcf",
    "pretrained_models/human_model_files/flame/FLAME_NEUTRAL.pkl": "efcd14cc4a69f3a3d9af8ded80146b5b6b50df3bd74cf69108213b144eba725b",
    "pretrained_models/human_model_files/flame/flame_static_embedding.pkl": "2881bdad8e61e87aad83a3df2e04655c534694089f68ca7144e7350da9e8ad62",
    "pretrained_models/human_model_files/flame/2019/generic_model.pkl": "e7f4f6ac7aa97045b3989893386aa94dc5245ded7105bbbf44116bfdf801a050",
    "pretrained_models/arcface_resnet18.pth": "69554b3184575bdda7ecf7f40caf2b0c5ecbd0617b676d2209d8bf0e7e3d8042",
    "pretrained_models/sapiens/pretrained/checkpoints/sapiens_1b/sapiens_1b_epoch_173_torchscript.pt2": "b5094b60a11ea8f94ef77b9e0a98439f677a5451c995e4dfda1de29538e8d861",
}
OPTIONAL_PRIOR_DIGESTS = {
    "pretrained_models/gagatracker/vgghead/vgg_heads_l.trcd": "18acb79c53032db11e8f502c12fdd34b5f642e9bc9041bce152c7b716c1b6f74",
    "pretrained_models/human_model_files/pose_estimate/multiHMR_896_L.pt": "c1b55933eb35df9b27966e1d8d3a69c4bc5e28eee0f0dd7f0cdc41c3f27202ca",
}


def verify_file(path: Path, expected: str) -> None:
    """Reject a changed asset without deserializing any of its contents."""
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != expected:
        msg = f"LHM asset integrity check failed: {path}. Run backend/scripts/setup-gpu.sh to restore pinned assets."
        raise ValueError(msg)


def verify_prior_assets(vendor: Path) -> None:
    """Authenticate core models and every present optional executable asset."""
    for relative_path, expected in REQUIRED_PRIOR_DIGESTS.items():
        verify_file(vendor / relative_path, expected)
    for relative_path, expected in OPTIONAL_PRIOR_DIGESTS.items():
        path = vendor / relative_path
        if path.exists():
            verify_file(path, expected)


def verify_lhm_assets(vendor: Path) -> None:
    """Fail before vendor imports if trusted assets are missing or changed."""
    verify_prior_assets(vendor)
    verify_file(vendor / DENSE_POINTS_PATH, DENSE_POINTS_DIGEST)


def extract_prior_archive(path: Path, destination: Path) -> None:
    """Verify the exact archive bytes before extracting safe data paths."""
    with path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != PRIOR_ARCHIVE_DIGEST:
            msg = f"LHM archive integrity check failed: {path}"
            raise ValueError(msg)
        stream.seek(0)
        with tarfile.open(fileobj=stream) as archive:
            archive.extractall(destination, filter="data")


def download_prior_assets(vendor: Path) -> None:
    """Restore pinned prior assets and authenticate existing installations."""
    from huggingface_hub import hf_hub_download

    try:
        verify_prior_assets(vendor)
    except (FileNotFoundError, ValueError):
        archive = hf_hub_download(
            PRIOR_REPO,
            "LHM_prior_model.tar",
            revision=PRIOR_REVISION,
            cache_dir=vendor / ".lhm_assets_download",
        )
        extract_prior_archive(Path(archive), vendor)

    points = vendor / DENSE_POINTS_PATH
    try:
        verify_file(points, DENSE_POINTS_DIGEST)
    except (FileNotFoundError, ValueError):
        points.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=points.parent, delete=False) as stream:
            staged = Path(stream.name)
        try:
            urllib.request.urlretrieve(DENSE_POINTS_URL, staged)
            verify_file(staged, DENSE_POINTS_DIGEST)
            staged.replace(points)
        finally:
            staged.unlink(missing_ok=True)
    verify_lhm_assets(vendor)


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
    download_prior_assets(args.vendor)
    model_snapshot(args.vendor, local_files_only=False)
