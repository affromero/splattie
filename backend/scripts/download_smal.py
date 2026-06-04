"""Download the SMAL parametric quadruped model from MPI (noncommercial license).

SMAL's download is session-gated — the difflocks-style POST to ``download.php`` 403s for the
SMAL domain — so this authenticates against the MPI shared login and downloads with that
session cookie. Credentials come from ``MPI_USERNAME`` / ``MPI_PASSWORD`` (inject via
``doppler run -p splattie -c prd -- ...``). Idempotent: skips if the model already exists.

URLs are overridable (``SMAL_LOGIN_URL`` / ``SMAL_DOWNLOAD_URL``) since MPI occasionally moves
endpoints; defaults target the published SMAL release archive.

Usage:
    doppler run -p splattie -c prd -- uv run python backend/scripts/download_smal.py
"""

from __future__ import annotations

import os
import sys
import zipfile
from pathlib import Path

import requests

VENDOR_SMAL = Path(__file__).resolve().parents[1] / "vendor" / "SMAL"
TARGET = VENDOR_SMAL / "smal_online_V1.0" / "smal_CVPR2017.pkl"
LOGIN_URL = os.environ.get("SMAL_LOGIN_URL", "https://download.is.tue.mpg.de/login.php")
DOWNLOAD_URL = os.environ.get(
    "SMAL_DOWNLOAD_URL",
    "https://download.is.tue.mpg.de/download.php?domain=smal&sfile=smal_online_V1.0.zip",
)


def main() -> None:
    """Authenticate with MPI and download + extract the SMAL model (idempotent)."""
    if TARGET.exists():
        print(f"SMAL already present: {TARGET}")
        return
    username = os.environ.get("MPI_USERNAME")
    password = os.environ.get("MPI_PASSWORD")
    if not (username and password):
        sys.exit("MPI_USERNAME / MPI_PASSWORD missing — run under `doppler run -p splattie -c prd -- ...`")

    VENDOR_SMAL.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = "Mozilla/5.0 (splattie-setup)"
    # MPI shared auth: POST credentials to obtain a session cookie, then download with it.
    login = session.post(
        LOGIN_URL,
        data={"username": username, "password": password, "commit": "Log in"},
        timeout=60,
    )
    login.raise_for_status()

    archive = VENDOR_SMAL / "smal_online_V1.0.zip"
    with session.get(DOWNLOAD_URL, stream=True, timeout=600) as response:
        response.raise_for_status()
        if "text/html" in response.headers.get("content-type", ""):
            sys.exit(
                "Received an HTML page instead of the archive — login likely failed. Confirm the "
                "MPI credentials and that the account is registered for SMAL, or set SMAL_DOWNLOAD_URL."
            )
        with archive.open("wb") as handle:
            for chunk in response.iter_content(1 << 20):
                handle.write(chunk)

    with zipfile.ZipFile(archive) as zf:
        zf.extractall(VENDOR_SMAL)
    archive.unlink()
    if not TARGET.exists():
        sys.exit(f"Extracted the archive but {TARGET} is missing — check the release layout.")
    print(f"SMAL ready: {TARGET}")


if __name__ == "__main__":
    main()
