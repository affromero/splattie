"""Download the SMAL parametric quadruped model from MPI (noncommercial license).

Verified flow (the simple POST-to-download.php that works for other MPI domains 403s/HTMLs
for SMAL): authenticate on the SMAL *project* host first, THEN POST the same credentials to
the *download* host (a different host, so its session is separate). Credentials come from
``MPI_USERNAME`` / ``MPI_PASSWORD`` — inject via ``doppler run -p splattie -c prd -- ...``.
Idempotent: skips if the model already exists.

Usage:
    doppler run -p splattie -c prd -- uv run python backend/scripts/download_smal.py
"""

from __future__ import annotations

import os
import sys
import tarfile
from pathlib import Path

import requests

VENDOR_SMAL = Path(__file__).resolve().parents[1] / "vendor" / "SMAL"
TARGET = VENDOR_SMAL / "smal_online_V1.0" / "smal_CVPR2017.pkl"
PROJECT_LOGIN_URL = os.environ.get("SMAL_LOGIN_URL", "https://smal.is.tue.mpg.de/login.php")
DOWNLOAD_URL = os.environ.get(
    "SMAL_DOWNLOAD_URL",
    "https://download.is.tue.mpg.de/download.php?domain=smal&resume=1&sfile=smalV1.0.tgz",
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
    # 1. Register the SMAL project session (seed cookies, then log in).
    session.get(PROJECT_LOGIN_URL, timeout=60)
    session.post(
        PROJECT_LOGIN_URL,
        data={"username": username, "password": password, "commit": "Log in"},
        timeout=60,
    )

    # 2. The download host has a separate session — POST the credentials directly to it.
    archive = VENDOR_SMAL / "smalV1.0.tgz"
    with session.post(
        DOWNLOAD_URL, data={"username": username, "password": password}, stream=True, timeout=600
    ) as resp:
        resp.raise_for_status()
        if "text/html" in resp.headers.get("content-type", ""):
            sys.exit(
                "Received HTML instead of the archive — login likely failed. Confirm the MPI "
                "credentials and that the account is registered for SMAL, or set SMAL_DOWNLOAD_URL."
            )
        with archive.open("wb") as handle:
            for chunk in resp.iter_content(1 << 20):
                handle.write(chunk)

    with tarfile.open(archive) as tar:
        tar.extractall(VENDOR_SMAL)  # noqa: S202  (trusted MPI archive)
    archive.unlink()
    if not TARGET.exists():
        sys.exit(f"Extracted the archive but {TARGET} is missing — check the release layout.")
    print(f"SMAL ready: {TARGET}")


if __name__ == "__main__":
    main()
