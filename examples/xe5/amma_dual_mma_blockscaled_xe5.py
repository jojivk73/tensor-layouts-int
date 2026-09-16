# MIT License — see LICENSE file for details.
"""XE5 dual-group block-scaled GEMM (cluster)

Functional port of the sycl-tla example examples/cute/tutorial/xe5/amma_dual_mma_blockscaled_xe5.cpp, runnable on the CPU.
Dual-group + block-scaled MX.

Run:  python examples/xe5/amma_dual_mma_blockscaled_xe5.py
Each section below is one operation; the same content is in amma_dual_mma_blockscaled_xe5.ipynb as cells."""

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
# Step — dual-group split
# ----------------------------------------------------------------------
# Xe5 dual-group MMA: a lane *pair* (ThrID=2) splits M in half. Lane 0 owns
# rows [0, M/2); lane 1 owns rows [M/2, M).
Cdual = Layout((2, (M // 2, N)), (M // 2, (1, M)))   # (thread=2, value=(M/2,N)) -> element
S = A @ B.T
nv = size(Cdual.shape[1])
rows0 = {Cdual(0, v) % M for v in range(nv)}
rows1 = {Cdual(1, v) % M for v in range(nv)}
print("dual C layout:", Cdual, " ThrID = 2")
print("lane 0 owns M-rows [%d, %d]" % (min(rows0), max(rows0)))
print("lane 1 owns M-rows [%d, %d]" % (min(rows1), max(rows1)))
print("S[:2,:4]:\n", S[:2, :4])

# ----------------------------------------------------------------------
# Step — block-scaled MMA
# ----------------------------------------------------------------------
# Block-scaled (MX): A and B are low-precision mantissas; each K-block of 8
# elements carries a shared scale. The MMA multiplies mantissas then applies the
# per-block scales: S = (A*sfA) . (B*sfB)^T.
BK = 8
sfA = rng.uniform(0.5, 2.0, size=(M, K // BK)).astype(np.float32)
sfB = rng.uniform(0.5, 2.0, size=(N, K // BK)).astype(np.float32)
S = (A * np.repeat(sfA, BK, axis=1)) @ (B * np.repeat(sfB, BK, axis=1)).T
print("scale-factor block size (K):", BK, " sfA shape", sfA.shape)
print("scaled S[:3,:6]:\n", S[:3, :6])

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
print('Recap: load → dual split → scaled MMA → store.')
