# MIT License — see LICENSE file for details.
"""XE5 LARGE core-matrix AMMA GEMM

Functional port of the sycl-tla example examples/cute/tutorial/xe5/xe5_adma_amma_large.cpp, runnable on the CPU.
LARGE SLM core matrices (8192-byte descriptor tiles).

Run:  python examples/xe5/xe5_adma_amma_large.py
Each section below is one operation; the same content is in xe5_adma_amma_large.ipynb as cells."""

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
# Step — LARGE SLM load
# ----------------------------------------------------------------------
# Xe5 LARGE core matrix: the SLM descriptor tile is larger (8192 B / 128 cols
# vs the 1024 B normal tile). The element layout is still a bijection.
slm_large = make_slm_layout_elem(sizeof_bits("bf16"), M, K)
print("LARGE SLM tile:", slm_large, " size:", size(slm_large), " bijective:", is_bijective(slm_large))
buf = np.zeros(size(slm_large), dtype=np.float32)
for m in range(M):
    for k in range(K):
        buf[slm_large(m, k)] = A[m, k]
A_back = np.array([[buf[slm_large(m, k)] for k in range(K)] for m in range(M)])
print("round-trip:", np.array_equal(A_back, A))

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
print('Recap: large SLM load → MMA → store.')
