"""Compare two directories written by capture_reference.py.

Usage: python tools/compare_reference.py DIR_A DIR_B
"""
import glob
import os
import sys

import numpy as np

a_dir, b_dir = sys.argv[1], sys.argv[2]
files = sorted(os.path.basename(f) for f in glob.glob(os.path.join(a_dir, "*.npz")))
diffs = {}
for name in files:
    a, b = np.load(os.path.join(a_dir, name)), np.load(os.path.join(b_dir, name))
    for key in a.files:
        x, y = a[key], b[key]
        if x.shape != y.shape:
            diffs.setdefault(key, []).append(f"{name}: shape {x.shape} vs {y.shape}")
        elif not np.array_equal(x, y, equal_nan=x.dtype.kind == "f"):
            if x.dtype.kind in "fc":
                d = np.nanmax(np.abs(x.astype(float) - y.astype(float)))
                diffs.setdefault(key, []).append(f"{name}: max abs diff {d:.3g}")
            else:
                n = int(np.sum(x != y))
                diffs.setdefault(key, []).append(f"{name}: {n}/{x.size} elements differ")

print(f"Compared {len(files)} frames")
if not diffs:
    print("IDENTICAL")
for key, msgs in diffs.items():
    print(f"{key}: differs in {len(msgs)} frames, first: {msgs[0]}")
sys.exit(1 if diffs else 0)
