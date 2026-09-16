# MIT License — see LICENSE file for details.
"""XE4 FMHA4 forward attention

Functional port of the sycl-tla example examples/xe4/fmha4/xe4_fmha_fwd.cpp, runnable on the CPU.
Flash-attention forward: S = scale·Q·Kᵀ, row softmax, O = P·V — two AMMA GEMMs bridged by an LDSM warp-row softmax, fed by ADMA loads.

Run:  python examples/xe4/fmha4_fwd_xe4.py
Each section below is one operation; the same content is in fmha4_fwd_xe4.ipynb as cells."""

# ----------------------------------------------------------------------
# Setup — data + a tiny layout printer
# ----------------------------------------------------------------------
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=120)
from tensor_layouts import Layout, size
from tensor_layouts.analysis import is_bijective
from tensor_layouts.atoms_xe_common import make_slm_layout_elem, sizeof_bits
from tensor_layouts.atoms_xe4 import make_xe4_ldstm

def show_layout(layout, n_rows, n_cols, rl="m", cl="k", max_r=8, max_c=8):
    "Print coord -> memory offset for a rank-2 layout (truncated)."
    R, C = min(n_rows, max_r), min(n_cols, max_c)
    print("      " + "".join((cl + str(j)).ljust(5) for j in range(C)) + (" ..." if C < n_cols else ""))
    for i in range(R):
        print((" " + rl + str(i)).ljust(6) + "".join(str(layout(i, j)).ljust(5) for j in range(C))
              + (" ..." if C < n_cols else ""))
    if R < n_rows:
        print("  ...  (%dx%d total)" % (n_rows, n_cols))

# Real FMHA4 tile is <64,128,128,128> = <Q-blk, V-dim, KV-blk, QK-dim>; we use a
# small block so the tables print. Q=queries, K=keys, V=values.
seq_q, seq_k, head = 32, 32, 16
rng = np.random.default_rng(0)
Q = rng.integers(-2, 3, size=(seq_q, head)).astype(np.float32)   # (seq_q x head)
Kk = rng.integers(-2, 3, size=(seq_k, head)).astype(np.float32)  # (seq_k x head)
V = rng.integers(-2, 3, size=(seq_k, head)).astype(np.float32)   # (seq_k x head)
scale = 1.0 / np.sqrt(head)
print("Q", Q.shape, " K", Kk.shape, " V", V.shape, " softmax scale =", round(scale, 4))

# ----------------------------------------------------------------------
# Step 1 — ADMA load Q,K into SLM
# ----------------------------------------------------------------------
# ADMA_Q / ADMA_K: load Q (and K) into SLM through the bank-swizzled core
# matrix, then read Q back to prove the mapping is loss-less.
slm = make_slm_layout_elem(sizeof_bits("bf16"), seq_q, head)
print("SLM layout:", slm, " bijective:", is_bijective(slm))
print("\n(row, head) -> SLM offset:")
show_layout(slm, seq_q, head, rl="q", cl="d")
buf = np.zeros(size(slm), dtype=np.float32)
for i in range(seq_q):
    for j in range(head):
        buf[slm(i, j)] = Q[i, j]
Q_back = np.array([[buf[slm(i, j)] for j in range(head)] for i in range(seq_q)])
print("\nSLM round-trip recovers Q:", np.array_equal(Q_back, Q))

# ----------------------------------------------------------------------
# Step 2 — GEMM-1: S = scale·Q·Kᵀ (TiledMmaQK)
# ----------------------------------------------------------------------
# GEMM-1  (TiledMmaQK):  S = scale * Q . K^T     (seq_q x seq_k scores)
S = scale * (Q @ Kk.T)
print("S shape", S.shape, "\n", S[:4, :8], "...")
# accumulator ownership: thread = key column, value = query row (col-major)
C = Layout((seq_k, seq_q), (seq_q, 1))
print("\nQK accumulator (thread=key, value=query) -> element offset:")
show_layout(C, seq_k, seq_q, rl="t", cl="v")

# ----------------------------------------------------------------------
# Step 3 — row softmax: S → P
# ----------------------------------------------------------------------
# softmax(S) over the seq_k keys of each query row (numerically stable).
row_max = S.max(axis=1, keepdims=True)
S_exp = np.exp(S - row_max)
row_sum = S_exp.sum(axis=1, keepdims=True)
P = S_exp / row_sum
print("row_max:", row_max.ravel()[:6], "...")
print("P row 0 sums to", P[0].sum(), " P[0,:6]:", P[0, :6])

# ----------------------------------------------------------------------
# Step 4 — LDSM warp-row (fred.max co-residence)
# ----------------------------------------------------------------------
# LDSM warp-row: the fred.max/sum need each query row co-resident in one
# 32-lane subgroup. Show the lane assignment and confirm the in-warp reduction.
ldsm = make_xe4_ldstm("LDSM", VS=8, s="bf16")
print("LDSM atom:", ldsm.name, " ThrID =", size(ldsm.thr_id), "lanes")
warp = {t: S[0, t] for t in range(min(seq_k, 32))}
print("row 0 lanes 0..7:", [warp[t] for t in range(8)])
print("in-warp max =", max(warp.values()), " == numpy:", max(warp.values()) == S[0, :min(seq_k, 32)].max())

# ----------------------------------------------------------------------
# Step 5 — GEMM-2: O = P·V (TiledMmaPV)
# ----------------------------------------------------------------------
# GEMM-2  (TiledMmaPV):  O = P . V     (seq_q x head output)
O = P @ V
print("O shape", O.shape, "\n", O[:4, :8], "...")

# ----------------------------------------------------------------------
# Step 6 — ADMA store O
# ----------------------------------------------------------------------
# ADMA store: write O back through the SLM core matrix and confirm loss-less.
slm_o = make_slm_layout_elem(sizeof_bits("bf16"), O.shape[0], O.shape[1])
print("O corner (to store):\n", O[:3, :6], "...")
obuf = np.zeros(size(slm_o), dtype=O.dtype)
for i in range(O.shape[0]):
    for j in range(O.shape[1]):
        obuf[slm_o(i, j)] = O[i, j]
back = np.array([[obuf[slm_o(i, j)] for j in range(O.shape[1])] for i in range(O.shape[0])])
print("store round-trip matches O:", np.array_equal(back, O))

# ----------------------------------------------------------------------
# Recap
# ----------------------------------------------------------------------
print('Recap: ADMA load → QK GEMM → softmax → LDSM warp-row → PV GEMM → store.')
