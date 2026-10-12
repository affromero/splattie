"""Authenticated legacy asset loading and extraction boundaries."""

from __future__ import annotations

import hashlib
import io
import pickle
import tarfile
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest

from splattie.methods.lam import assets, method


def _write_marker(path: str) -> None:
    Path(path).write_text("executed")


class UntrustedMask:
    """Executable pickle fixture whose only side effect creates a test marker."""

    def __init__(self, marker: Path) -> None:
        self.marker = marker

    def __reduce__(self) -> tuple[object, tuple[str]]:
        """Encode an observable side effect without executing it during serialization."""
        return _write_marker, (str(self.marker),)


def _trust_fixture(monkeypatch: pytest.MonkeyPatch, content: bytes) -> None:
    monkeypatch.setattr(assets, "FLAME_MASKS_DIGEST", hashlib.sha256(content).hexdigest())


@pytest.mark.parametrize("protocol", [0, 2])
def test_authenticated_raw_masks_preserve_numpy_vertex_indices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, protocol: int
) -> None:
    content = pickle.dumps({"jaw": np.array([1, 3, 5], dtype=np.int64)}, protocol=protocol)
    path = tmp_path / "FLAME_masks.pkl"
    path.write_bytes(content)
    _trust_fixture(monkeypatch, content)
    masks = assets.load_flame_masks(path)
    np.testing.assert_array_equal(masks["jaw"], [1, 3, 5])


def test_untrusted_masks_are_rejected_without_executing_pickle(tmp_path: Path) -> None:
    marker = tmp_path / "executed"
    path = tmp_path / "FLAME_masks.pkl"
    path.write_bytes(pickle.dumps(UntrustedMask(marker)))
    with pytest.raises(ValueError, match="integrity check failed"):
        assets.load_flame_masks(path)
    assert not marker.exists()


def test_mask_reader_uses_the_verified_bytes_when_path_changes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    content = pickle.dumps({"eyes": [2, 4]})
    path = tmp_path / "FLAME_masks.pkl"
    path.write_bytes(content)
    _trust_fixture(monkeypatch, content)
    original_read = Path.read_bytes

    def replace_after_read(file_path: Path) -> bytes:
        result = original_read(file_path)
        file_path.write_bytes(b"changed after read")
        return result

    monkeypatch.setattr(Path, "read_bytes", replace_after_read)
    assert list(assets.load_flame_masks(path)["eyes"]) == [2, 4]


def test_missing_masks_raise_instead_of_producing_empty_regions(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        assets.load_flame_masks(tmp_path / "missing.pkl")


def test_authenticated_non_mapping_masks_are_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    content = pickle.dumps([1, 2, 3])
    path = tmp_path / "FLAME_masks.pkl"
    path.write_bytes(content)
    _trust_fixture(monkeypatch, content)
    with pytest.raises(ValueError, match="Invalid FLAME mask mapping"):
        assets.load_flame_masks(path)


def _configure_asset_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Mapping[str, str]:
    content = b"operator asset"
    paths = [
        "model_zoo/human_parametric_models/flame_assets/flame/flame2023.pkl",
        "model_zoo/human_parametric_models/flame_vhap/uv_masks.npz",
        "model_zoo/flame_tracking_models/vgghead/vgg_heads_l.trcd",
    ]
    manifest = {path: hashlib.sha256(content).hexdigest() for path in paths}
    for relative in paths:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    monkeypatch.setattr(assets, "LAM_LEGACY_ASSET_DIGESTS", manifest)
    return manifest


def test_tracker_requires_authenticated_assets_including_uv_masks_and_vgg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _configure_asset_fixture(tmp_path, monkeypatch)
    assets.verify_lam_assets(tmp_path, tracker=True)
    for relative in manifest:
        path = tmp_path / relative
        original = path.read_bytes()
        path.write_bytes(b"tampered")
        with pytest.raises(ValueError, match="integrity check failed"):
            assets.verify_lam_assets(tmp_path, tracker=True)
        path.write_bytes(original)


def test_head_loader_rejects_changed_assets_before_importing_vendor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _configure_asset_fixture(tmp_path, monkeypatch)
    (tmp_path / "model_zoo/human_parametric_models/flame_assets/flame/flame2023.pkl").write_bytes(b"tampered")
    monkeypatch.setattr(method, "VENDOR_LAM", tmp_path)
    with pytest.raises(ValueError, match="integrity check failed"):
        method.LAMMethod().load()


def test_tracker_rejects_missing_legacy_files(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        assets.verify_lam_assets(tmp_path, tracker=True)


def _archive_fixture(tmp_path: Path, name: str, *, link: bool = False) -> Path:
    path = tmp_path / "thirdparty_models.tar"
    with tarfile.open(path, "w") as archive:
        member = tarfile.TarInfo(name)
        if link:
            member.type = tarfile.SYMTYPE
            member.linkname = "../../escaped"
            archive.addfile(member)
        else:
            content = b"authenticated model bytes"
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
    return path


def _trust_archive(monkeypatch: pytest.MonkeyPatch, path: Path) -> None:
    monkeypatch.setattr(assets, "LAM_ARCHIVE_DIGESTS", {path.name: hashlib.sha256(path.read_bytes()).hexdigest()})


def test_authenticated_archive_preserves_the_vendor_directory_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _archive_fixture(tmp_path, "model_zoo/flame/model.bin")
    _trust_archive(monkeypatch, path)
    destination = tmp_path / "LAM"
    assets.extract_lam_archive(path, destination)
    assert (destination / "model_zoo/flame/model.bin").read_bytes() == b"authenticated model bytes"


def test_changed_archive_is_rejected_before_any_extraction(tmp_path: Path) -> None:
    path = _archive_fixture(tmp_path, "model.bin")
    destination = tmp_path / "LAM"
    with pytest.raises(ValueError, match="integrity check failed"):
        assets.extract_lam_archive(path, destination)
    assert not destination.exists()


@pytest.mark.parametrize(("name", "link"), [("../escaped", False), ("model_zoo/link", True)])
def test_authenticated_archive_still_rejects_escaping_members(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str, *, link: bool
) -> None:
    path = _archive_fixture(tmp_path, name, link=link)
    _trust_archive(monkeypatch, path)
    with pytest.raises(tarfile.FilterError):
        assets.extract_lam_archive(path, tmp_path / "LAM")
    assert not (tmp_path / "escaped").exists()
