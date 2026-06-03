"""Load SMAL and extract its neutral mesh + 33-joint skeleton (the rig we'll bind)."""

# chumpy 0.70 + numpy<2 compat shim for Python 3.11 (chumpy unpickles SMAL).
import inspect

if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec
import numpy as np

for _n, _t in [("bool", bool), ("int", int), ("float", float), ("complex", complex),
               ("object", object), ("str", str), ("unicode", str), ("bool8", bool)]:
    if not hasattr(np, _n):
        setattr(np, _n, _t)
import chumpy  # noqa: F401  -- import now so the shim is in effect before unpickling

import pickle
from pathlib import Path

SMAL = Path("/home/ubuntu/Code/splattie-wt-quadruped-spike/backend/vendor/SMAL/smal_online_V1.0/smal_CVPR2017.pkl")
with open(SMAL, "rb") as f:
    m = pickle.load(f, encoding="latin1")

print("type:", type(m).__name__, "| keys:", sorted(k for k in m))


def arr(x):
    return np.asarray(x.r if hasattr(x, "r") else x, dtype=np.float64)


vt = arr(m["v_template"])
faces = np.asarray(m["f"])
weights = arr(m["weights"])
kintree = np.asarray(m["kintree_table"])
# Joint regressor: sparse or dense -> J = Jreg @ v_template
Jreg = m["J_regressor"]
J = np.asarray(Jreg.dot(vt)) if hasattr(Jreg, "dot") else arr(Jreg) @ vt

print(f"\nv_template: {vt.shape}  faces: {faces.shape}  weights(LBS): {weights.shape}")
print(f"joints (J): {J.shape}  -> {J.shape[0]} joints")
print(f"kintree_table: {kintree.shape}  (parents row = {kintree[0].tolist()})")
print(f"shapedirs: {arr(m['shapedirs']).shape if 'shapedirs' in m else 'n/a'}")
print(f"posedirs: {arr(m['posedirs']).shape if 'posedirs' in m else 'n/a'}")
print(f"\nneutral mesh bbox: min {vt.min(0).round(3).tolist()}  max {vt.max(0).round(3).tolist()}")
print(f"neutral joints bbox: min {J.min(0).round(3).tolist()}  max {J.max(0).round(3).tolist()}")
print("up-axis guess (largest joint spread):", "xyz"[int(np.argmax(J.max(0) - J.min(0)))])
print("\nfirst 8 joint positions:")
for i in range(min(8, len(J))):
    print(f"  joint{i:2d} parent={kintree[0, i]:>3}  ({J[i, 0]:+.3f},{J[i, 1]:+.3f},{J[i, 2]:+.3f})")
