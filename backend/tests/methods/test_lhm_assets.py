"""LHM archive and installed-model integrity checks without GPU construction."""

from __future__ import annotations

import hashlib
import io
import tarfile
from pathlib import Path

import pytest

from splattie.methods.lhm import assets


def archive_file(path: Path, name: str, content: bytes) -> None:
    with tarfile.open(path, "w") as archive:
        entry = tarfile.TarInfo(name)
        entry.size = len(content)
        archive.addfile(entry, io.BytesIO(content))


@pytest.fixture
def installation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    content = b"known prior bytes"
    digest = hashlib.sha256(content).hexdigest()
    monkeypatch.setattr(assets, "REQUIRED_PRIOR_DIGESTS", {"prior.pkl": digest})
    monkeypatch.setattr(assets, "OPTIONAL_PRIOR_DIGESTS", {"optional.pt": digest})
    monkeypatch.setattr(assets, "DENSE_POINTS_DIGEST", digest)
    (tmp_path / "prior.pkl").write_bytes(content)
    points = tmp_path / assets.DENSE_POINTS_PATH
    points.parent.mkdir(parents=True)
    points.write_bytes(content)
    return tmp_path


def test_verified_installation_preserves_optional_asset_absence(installation: Path) -> None:
    assets.verify_lhm_assets(installation)
    assert not (installation / "optional.pt").exists()


def test_present_optional_assets_must_match_their_digest(installation: Path) -> None:
    optional = installation / "optional.pt"
    optional.write_bytes(b"known prior bytes")
    assets.verify_lhm_assets(installation)
    optional.write_bytes(b"changed prior bytes")
    with pytest.raises(ValueError, match=r"optional\.pt"):
        assets.verify_lhm_assets(installation)


def test_missing_core_assets_are_rejected(installation: Path) -> None:
    (installation / "prior.pkl").unlink()
    with pytest.raises(FileNotFoundError):
        assets.verify_lhm_assets(installation)


def test_changed_archives_cannot_extract_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    archive = tmp_path / "prior.tar"
    archive_file(archive, "prior.pkl", b"changed archive")
    monkeypatch.setattr(assets, "PRIOR_ARCHIVE_DIGEST", "0" * 64)
    destination = tmp_path / "models"
    with pytest.raises(ValueError, match="archive integrity"):
        assets.extract_prior_archive(archive, destination)
    assert not destination.exists()


def test_verified_archives_preserve_asset_contents(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    archive = tmp_path / "prior.tar"
    archive_file(archive, "models/prior.pkl", b"known prior bytes")
    monkeypatch.setattr(assets, "PRIOR_ARCHIVE_DIGEST", hashlib.sha256(archive.read_bytes()).hexdigest())
    destination = tmp_path / "assets"
    assets.extract_prior_archive(archive, destination)
    assert (destination / "models/prior.pkl").read_bytes() == b"known prior bytes"


def test_verified_archives_cannot_escape_destination(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    archive = tmp_path / "prior.tar"
    archive_file(archive, "../escaped.pkl", b"outside asset directory")
    monkeypatch.setattr(assets, "PRIOR_ARCHIVE_DIGEST", hashlib.sha256(archive.read_bytes()).hexdigest())
    with pytest.raises(tarfile.FilterError):
        assets.extract_prior_archive(archive, tmp_path / "assets")
    assert not (tmp_path / "escaped.pkl").exists()


def test_existing_assets_are_verified_without_downloads(installation: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    assets.download_prior_assets(installation)
    assets.verify_lhm_assets(installation)


def test_setup_restores_changed_priors_from_an_authenticated_archive(
    installation: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import huggingface_hub

    prior = installation / "prior.pkl"
    prior.write_bytes(b"changed prior")
    archive = installation / "prior.tar"
    archive_file(archive, "prior.pkl", b"known prior bytes")
    monkeypatch.setattr(assets, "PRIOR_ARCHIVE_DIGEST", hashlib.sha256(archive.read_bytes()).hexdigest())

    def download(repo_id: str, filename: str, *, revision: str, cache_dir: Path) -> str:
        assert repo_id == assets.PRIOR_REPO
        assert filename == "LHM_prior_model.tar"
        assert revision == assets.PRIOR_REVISION
        assert cache_dir.is_relative_to(installation)
        return str(archive)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    assets.download_prior_assets(installation)
    assert prior.read_bytes() == b"known prior bytes"
    assets.verify_lhm_assets(installation)


@pytest.mark.parametrize("valid", [False, True])
def test_downloaded_geometry_is_authenticated_before_replacing_installed_bytes(
    installation: Path, monkeypatch: pytest.MonkeyPatch, *, valid: bool
) -> None:
    points = installation / assets.DENSE_POINTS_PATH
    points.write_bytes(b"old changed geometry")

    def download(url: str, filename: Path) -> None:
        assert url == assets.DENSE_POINTS_URL
        filename.write_bytes(b"known prior bytes" if valid else b"unauthenticated geometry")

    monkeypatch.setattr(assets.urllib.request, "urlretrieve", download)
    if valid:
        assets.download_prior_assets(installation)
        assert points.read_bytes() == b"known prior bytes"
    else:
        with pytest.raises(ValueError, match="asset integrity"):
            assets.download_prior_assets(installation)
        assert points.read_bytes() == b"old changed geometry"
