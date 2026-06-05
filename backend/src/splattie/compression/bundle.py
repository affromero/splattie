"""Recompress a `.splattie` bundle's gaussian PLY to compressed PLY.

Swaps only the inner splat PLY for its splat-transform compressed-PLY form
(~4x smaller, self-describing via the `packed_position` header) and leaves the
manifest and every other entry byte-for-byte identical. The manifest keeps
`format: "ply"` because compressed PLY is still a PLY that the widget's Spark
reader parses through its (correct) PLY path -- so no manifest or widget-version
change is needed. Idempotent: an already-compressed bundle is passed through
unchanged.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import zipfile
from pathlib import Path

from klogr import get_logger
from pydantic import BaseModel

from splattie.compression.compressed_ply import compress_ply

logger = get_logger()

# splat-transform's compressed PLY carries a `packed_position` chunk property in its
# header; raw PLY never does. Cheap, reliable marker to keep recompression idempotent.
_COMPRESSED_MARKER = b"packed_position"


class CompressBundleResult(BaseModel):
    """Outcome of compressing one bundle's splat payload."""

    splat_entry: str
    already_compressed: bool
    size_before: int
    size_after: int


def _splat_entry_name(names: list[str], manifest_bytes: bytes | None) -> str:
    """Inner gaussian PLY name: the manifest's `avatar.splat.file`, else the lone `.ply`."""
    if manifest_bytes is not None:
        manifest = json.loads(manifest_bytes.decode("utf-8"))
        try:
            file = manifest["avatar"]["splat"]["file"]
        except (KeyError, TypeError):
            file = None
        if isinstance(file, str) and file in names:
            return file
    plys = [name for name in names if name.endswith(".ply")]
    if len(plys) != 1:
        msg = f"cannot locate splat PLY: expected exactly one .ply entry, found {plys}"
        raise ValueError(msg)
    return plys[0]


def compress_bundle(src: Path, dst: Path) -> CompressBundleResult:
    """Rewrite `src` into `dst` with its inner PLY replaced by compressed PLY.

    Every other entry (manifest, skeleton, weights, states) and the original entry
    order are preserved exactly. `src` and `dst` may be the same path (in place).
    Idempotent: a bundle whose splat is already compressed is passed through unchanged.
    """
    src = src.resolve()
    size_before = src.stat().st_size
    with zipfile.ZipFile(str(src), "r") as zf:
        names = [name for name in zf.namelist() if not name.endswith("/")]
        manifest_bytes = zf.read("manifest.json") if "manifest.json" in names else None
        entry = _splat_entry_name(names, manifest_bytes)
        payload = {name: zf.read(name) for name in names}

    if _COMPRESSED_MARKER in payload[entry][:1024]:
        if src != dst.resolve():
            shutil.copy(src, dst)
        return CompressBundleResult(
            splat_entry=entry, already_compressed=True, size_before=size_before, size_after=size_before
        )

    with tempfile.TemporaryDirectory() as td:
        ply_tmp = Path(td) / "splat.ply"
        comp_tmp = Path(td) / "splat.compressed.ply"
        ply_tmp.write_bytes(payload[entry])
        compress_ply(ply_tmp, comp_tmp)
        payload[entry] = comp_tmp.read_bytes()

    tmp_out = dst.with_suffix(dst.suffix + ".tmp")
    with zipfile.ZipFile(str(tmp_out), "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in payload.items():
            zf.writestr(name, data)
    tmp_out.replace(dst)
    size_after = dst.stat().st_size
    logger.info(f"Compressed bundle {src.name}: {size_before // 1024} KB -> {size_after // 1024} KB")
    return CompressBundleResult(
        splat_entry=entry, already_compressed=False, size_before=size_before, size_after=size_after
    )
