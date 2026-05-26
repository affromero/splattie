"""Local file storage for uploads and generated models."""

from __future__ import annotations

import uuid
from pathlib import Path

UPLOAD_DIR = Path("data/uploads")
GENERATION_DIR = Path("data/generations")


def save_upload(data: bytes, suffix: str = ".png") -> tuple[str, Path]:
    """Save an uploaded file and return (id, path)."""
    file_id = uuid.uuid4().hex[:12]
    upload_dir = UPLOAD_DIR / file_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / f"original{suffix}"
    file_path.write_bytes(data)
    return file_id, file_path


def get_storage_path(model_id: str, filename: str) -> Path | None:
    """Get the filesystem path for a stored file."""
    path = GENERATION_DIR / model_id / filename
    if path.exists():
        return path
    upload_path = UPLOAD_DIR / model_id / filename
    if upload_path.exists():
        return upload_path
    return None
