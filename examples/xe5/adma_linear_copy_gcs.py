# MIT License — see LICENSE file for details.
"""XE5 linear GCS copy

Functional port of the sycl-tla example examples/cute/tutorial/xe5/adma_linear_copy_gcs.cpp, runnable on the CPU.
A copy-only example: the GCS (.gs) load fills SLM from a smaller global source, zero-filling the remainder.

Run:  python examples/xe5/adma_linear_copy_gcs.py
Each section below is one operation; the same content is in adma_linear_copy_gcs.ipynb as cells."""

# ----------------------------------------------------------------------
# Setup — data + a tiny layout printer
# ----------------------------------------------------------------------
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=120)
from tensor_layouts import Layout, size
from tensor_layouts.analysis import is_bijective
from tensor_layouts.atoms_xe_common import make_slm_layout_elem, sizeof_bits

def show_layout(layout, n_rows, n_cols, rl="m", cl="k", max_r=8, max_c=8):
    "Print coord -> memory offset for a rank-2 layout (truncated)."
    R, C = min(n_rows, max_r), min(n_cols, max_c)
    print("      " + "".join((cl + str(j)).ljust(5) for j in range(C)) + (" ..." if C < n_cols else ""))
    for i in range(R):
        print((" " + rl + str(i)).ljust(6) + "".join(str(layout(i, j)).ljust(5) for j in range(C))
              + (" ..." if C < n_cols else ""))
    if R < n_rows:
        print("  ...  (%dx%d total)" % (n_rows, n_cols))

M, N, K = 32, 32, 16          # A rows, B rows, contraction
rng = np.random.default_rng(0)
A = rng.integers(-2, 3, size=(M, K)).astype(np.float32)   # A  (M x K)
B = rng.integers(-2, 3, size=(N, K)).astype(np.float32)   # B  (N x K), used as B^T
print("A", A.shape, " B", B.shape)
print("A[:4]:\n", A[:4])

# ----------------------------------------------------------------------
# Step — GCS load with zero-fill
# ----------------------------------------------------------------------
# Linear GCS (.gs) load: the global source is *smaller* than the SLM
# destination; the hardware zero-fills the remainder.
glb = rng.integers(1, 5, size=24).astype(np.float32)   # 24 elements available
slm_fill = np.zeros(32, dtype=np.float32)              # 32-element SLM slot
slm_fill[:glb.size] = glb                              # GCS load + zero-fill
print("global source (24):", glb)
print("SLM after GCS load (32):", slm_fill)
print("tail zero-filled:", np.all(slm_fill[glb.size:] == 0))

# ----------------------------------------------------------------------
# Recap
# ----------------------------------------------------------------------
print('Recap: GCS load: global bytes → SLM fill.')
