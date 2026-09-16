# MIT License — see LICENSE file for details.
"""XE4 GEMM subgroup-split reduce

Functional port of the sycl-tla example examples/cute/tutorial/xe4/gemm_sgsplit_reduce_xe4.cpp, runnable on the CPU.
K split across subgroups; partials reduced via store-reduce.

Run:  python examples/xe4/gemm_sgsplit_reduce_xe4.py
Each section below is one operation; the same content is in gemm_sgsplit_reduce_xe4.ipynb as cells."""

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
# Step — ADMA load: gmem → SLM (bank-swizzled)
# ----------------------------------------------------------------------
# ADMA load: move A from global memory into SLM through the Xe bank-swizzled
# core-matrix layout, then read it back to prove the mapping is loss-less.
slm = make_slm_layout_elem(sizeof_bits("bf16"), M, K)
print("SLM layout:", slm, " bijective:", is_bijective(slm), " size:", size(slm))
print("\n(m,k) -> SLM offset  (note the bank jumps down a column):")
show_layout(slm, M, K)
slm_buf = np.zeros(size(slm), dtype=np.float32)
for m in range(M):
    for k in range(K):
        slm_buf[slm(m, k)] = A[m, k]
A_back = np.array([[slm_buf[slm(m, k)] for k in range(K)] for m in range(M)])
print("\nSLM round-trip recovers A exactly:", np.array_equal(A_back, A))

# ----------------------------------------------------------------------
# Step — subgroup-split-K + reduce
# ----------------------------------------------------------------------
# Split-K reduction: the K dimension is split across workers; each computes a
# partial S, and the partials are summed by an atomic store-reduce.
half = K // 2
S_lo = A[:, :half] @ B[:, :half].T
S_hi = A[:, half:] @ B[:, half:].T
S = S_lo + S_hi                      # the store-reduce (atomic add) step
print("partial S_lo[:2,:4]:\n", S_lo[:2, :4])
print("partial S_hi[:2,:4]:\n", S_hi[:2, :4])
print("reduced  S   [:2,:4]:\n", S[:2, :4])
print("equals full A.B^T:", np.allclose(S, A @ B.T))

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
print('Recap: load → subgroup split-K → reduce.')
