"""Back-apply the lower-arm rigid re-skin to existing body demo `.splattie` files.

`reweight_lower_arm_rigid` is baked into `bundle.py` for newly generated bodies; this
one-off migrates the already-shipped demos so they match (their lower arms swing as
solid limbs instead of stretching toward the feet when zoomed out). Idempotent — re-
running on an already-migrated bundle re-derives the same binding.

    cd backend && uv run python scripts/reweight_demo_arms.py ../apps/web/public/demos/bodies/*.splattie
"""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

from splattie.methods.lhm.bundle import parse_ply_xyz, reweight_lower_arm_rigid


def migrate(path: Path) -> int:
    """Re-skin one body `.splattie` in place; returns its gaussian count."""
    with zipfile.ZipFile(path) as zf:
        files = {n: zf.read(n) for n in zf.namelist()}
        ply_name = next(n for n in files if n.endswith(".ply"))
    skeleton = json.loads(files["skeleton.json"])
    weights = json.loads(files["lbs_weights.json"])
    reweight_lower_arm_rigid(parse_ply_xyz(files[ply_name]), skeleton, weights)
    files["lbs_weights.json"] = json.dumps(weights).encode()
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, blob in files.items():
            zf.writestr(name, blob)
    return int(weights["numGaussians"])


if __name__ == "__main__":
    paths = [Path(p) for p in sys.argv[1:]]
    if not paths:
        print(__doc__)
        raise SystemExit(2)
    for p in paths:
        print(f"reweighted {p.name} ({migrate(p)} gaussians)")
