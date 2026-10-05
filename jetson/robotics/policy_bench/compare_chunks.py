"""Compare saved action chunks: |a-b| per joint, against a same-precision seed-to-seed baseline.

    python compare_chunks.py fp32_s0.npy bf16_s0.npy fp32_s1.npy
The third file (same precision, different noise seed) sets the scale: a precision
change smaller than the model's own sampling spread is harmless.
"""
import sys
import numpy as np
ref, test, *base = [np.load(f).reshape(-1, np.load(f).shape[-1]) for f in sys.argv[1:]]
def row(name, d):
    print(f"{name:<28} mean {np.abs(d).mean():7.3f}   max {np.abs(d).max():7.3f}   per-joint mean "
          + " ".join(f"{x:6.2f}" for x in np.abs(d).mean(0)))
print(f"chunk {ref.shape[0]} steps x {ref.shape[1]} joints (joint units, degrees for SO-101)")
row("precision change", test - ref)
if base:
    row("noise seed change (same dt)", base[0] - ref)
