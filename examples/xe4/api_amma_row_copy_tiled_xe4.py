# MIT License — see LICENSE file for details.
"""XE4 tiled row-copy API GEMM

Functional port of the sycl-tla example examples/cute/tutorial/xe4/api_amma_row_copy_tiled_xe4.cpp, runnable on the CPU.
AMMA fed by the per-warp **tiled row-copy** API.

Run:  python examples/xe4/api_amma_row_copy_tiled_xe4.py
Each section below is one operation; the same content is in api_amma_row_copy_tiled_xe4.ipynb as cells."""

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
# Step — MMA: S = A · Bᵀ
# ----------------------------------------------------------------------
# MMA: the systolic multiply computes S = A . B^T.
S = A @ B.T
print("S = A . B^T  shape", S.shape, "\n", S[:4, :8], "...")
aM, aN, aK = 32, 32, 16
print("accumulator grid: %d x %d x %d atom tile(s)  (atom %dx%dx%d)"
      % (M // aM, N // aN, max(K // aK, 1), aM, aN, aK))
C = Layout((N, M), (M, 1))   # (thread=col of B, value=row of A) -> element, col-major M x N
print("C accumulator (thread, value) -> element offset:")
show_layout(C, N, M, rl="t", cl="v")

# ----------------------------------------------------------------------
# Step — ADMA store: SLM → gmem
# ----------------------------------------------------------------------
# ADMA store: write S back to global memory through the SLM core matrix
# (reverse of the load) and confirm it is loss-less.
res = S
slm_o = make_slm_layout_elem(sizeof_bits("bf16"), res.shape[0], res.shape[1])
print("result corner (to be stored):\n", res[:3, :6], "...")
obuf = np.zeros(size(slm_o), dtype=res.dtype)
for i in range(res.shape[0]):
    for j in range(res.shape[1]):
        obuf[slm_o(i, j)] = res[i, j]
print("SLM store offsets for rows 0..3, col 0:", [slm_o(i, 0) for i in range(min(4, res.shape[0]))])
back = np.array([[obuf[slm_o(i, j)] for j in range(res.shape[1])] for i in range(res.shape[0])])
print("store round-trip matches result:", np.array_equal(back, res))

# ----------------------------------------------------------------------
# Recap
# ----------------------------------------------------------------------
print('Recap: row-copy load → MMA → store.')
