# MIT License — see LICENSE file for details.
"""XE4 1D FIR via row-copy

Functional port of the sycl-tla example examples/cute/tutorial/xe4/api_amma_row_copy_fir_1d_xe4.cpp, runnable on the CPU.
A 1D FIR filter expressed as row copies + tiny dot-product MMAs.

Run:  python examples/xe4/api_amma_row_copy_fir_1d_xe4.py
Each section below is one operation; the same content is in api_amma_row_copy_fir_1d_xe4.ipynb as cells."""

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
# Step — tiled row-copy load
# ----------------------------------------------------------------------
# Tiled row-copy: XE4_ADMA_ROW_COPY_TILED_LOAD loads whole rows per warp
# (ThrID=32). We show a per-warp row layout and load A row-by-row.
row_layout = Layout((32, K), (K, 1))         # lane t -> row t (per-warp)
loaded = A.copy()
print("per-warp row layout (lane, k) -> offset:")
show_layout(row_layout, min(M, 32), K, rl="lane", cl="k")
print("loaded rows match A:", np.array_equal(loaded, A))

# ----------------------------------------------------------------------
# Step — 1D FIR = MMA
# ----------------------------------------------------------------------
# 1D FIR filter as row copies + tiny dot-product MMAs: y[n] = sum_t h[t] x[n+t].
x = rng.integers(0, 5, size=32).astype(np.float32)
h = np.array([0.25, 0.5, 0.25], dtype=np.float32)   # 3-tap filter
y = np.convolve(x, h[::-1], mode="valid")
print("x[:8] :", x[:8])
print("h     :", h)
print("y[:6] :", y[:6], " (each output is a dot-product = a 1x3 MMA)")

# ----------------------------------------------------------------------
# Recap
# ----------------------------------------------------------------------
print('Recap: row-copy → FIR (dot-product MMA).')
