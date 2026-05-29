"""Splattie backend — single-image → animatable 3D avatar (heads via LAM, bodies via LHM)."""

from pathlib import Path

from dotenv import load_dotenv

# Load secrets/config from backend/.env for every entry point (API, CLI, tests, GPU
# generation scripts). In production these are injected by the secret manager
# (`doppler run`); .env is the local / GPU-box fallback. No-op when the file is absent.
load_dotenv(Path(__file__).resolve().parents[2] / ".env")
