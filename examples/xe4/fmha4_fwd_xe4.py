# MIT License — see LICENSE file for details.
"""XE4 FMHA4 forward attention

Full-execution-flow functional port of the sycl-tla example examples/xe4/fmha4/xe4_fmha_fwd.cpp.
Flash-attention forward: S=scale*Q*K^T, softmax, O=P*V — two AMMA GEMMs bridged by an LDSM warp-row softmax.

Run:  python examples/xe4/fmha4_fwd_xe4.py"""

# ----------------------------------------------------------------------
# Helpers — display + SLM copy
# ----------------------------------------------------------------------
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=120)
from tensor_layouts import Layout, size
from tensor_layouts.analysis import is_bijective
from tensor_layouts.atoms_xe_common import make_slm_layout_elem, sizeof_bits

# ---- display helpers (reused by every cell below) ----
def show_mat(name, X, r=4, c=8):
    X = np.asarray(X)
    print(name, " shape", X.shape)
    print(X[:r, :c] if X.ndim == 2 else X[:c])
    if X.ndim == 2 and (X.shape[0] > r or X.shape[1] > c):
        print("   ...(showing %dx%d of %dx%d)" % (min(r, X.shape[0]), min(c, X.shape[1]), *X.shape))

def show_layout(layout, n_rows, n_cols, rl="m", cl="k", max_r=8, max_c=8):
    R, C = min(n_rows, max_r), min(n_cols, max_c)
    print("      " + "".join((cl + str(j)).ljust(5) for j in range(C)) + (" ..." if C < n_cols else ""))
    for i in range(R):
        print((" " + rl + str(i)).ljust(6) + "".join(str(layout(i, j)).ljust(5) for j in range(C))
              + (" ..." if C < n_cols else ""))
    if R < n_rows:
        print("   ...(%dx%d total)" % (n_rows, n_cols))

def check(name, cond):
    print(("[PASS] " if cond else "[FAIL] ") + name)

# ---- SLM staging helpers: gmem tile <-> bank-swizzled SLM buffer ----
def load_to_slm(mat, lay):
    buf = np.zeros(size(lay), dtype=mat.dtype)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            buf[lay(i, j)] = mat[i, j]
    return buf

def read_slm(buf, lay, r, c):
    return np.array([[buf[lay(i, j)] for j in range(c)] for i in range(r)])

print("helpers ready: show_mat, show_layout, check, load_to_slm, read_slm")

# ----------------------------------------------------------------------
# Setup — allocate + initialize operands
# ----------------------------------------------------------------------
# Flash-attention forward. Real tile <64,128,128,128>; small here to print.
seq_q, seq_k, head = 32, 32, 16
BLK_K = 16; K_TILE = head // BLK_K; K_PIPE = 2
scale = 1.0 / np.sqrt(head); DT = "bf16"; DBITS = sizeof_bits(DT)
rng = np.random.default_rng(0)
Q  = rng.integers(-2, 3, size=(seq_q, head)).astype(np.float32)
Kk = rng.integers(-2, 3, size=(seq_k, head)).astype(np.float32)
V  = rng.integers(-2, 3, size=(seq_k, head)).astype(np.float32)
show_mat("Q", Q); show_mat("K", Kk); show_mat("V", V); print("softmax scale:", round(scale, 4))

# ----------------------------------------------------------------------
# Config — two TiledMMAs (QK, PV) + SLM
# ----------------------------------------------------------------------
from tensor_layouts.atoms_xe4 import make_xe4_amma_atom, make_xe4_ldstm
mmaQK = make_xe4_amma_atom("f32", DT, DT, "f32", seq_q, seq_k, BLK_K, tracking="AB")
mmaPV = make_xe4_amma_atom("f32", DT, DT, "f32", seq_q, head, BLK_K, tracking="AB")
slmQ = make_slm_layout_elem(DBITS, seq_q, head); slmK = make_slm_layout_elem(DBITS, seq_k, head)
print("TiledMmaQK:", mmaQK.name); print("TiledMmaPV:", mmaPV.name, " K-tiles:", K_TILE)

# ----------------------------------------------------------------------
# ADMA load Q, K into SLM
# ----------------------------------------------------------------------
# ADMA_Q / ADMA_K: stage Q and K through the swizzled SLM.
qbuf = load_to_slm(Q, slmQ)
check("Q SLM round-trip", np.array_equal(read_slm(qbuf, slmQ, seq_q, head), Q))
print("SLM Q layout (row, head) -> offset:"); show_layout(slmQ, seq_q, head, rl="q", cl="d")

# ----------------------------------------------------------------------
# GEMM-1 mainloop: S = scale*Q*K^T
# ----------------------------------------------------------------------
# GEMM-1 mainloop (TiledMmaQK): S = scale * sum_k Q_k . K_k^T.
S = np.zeros((seq_q, seq_k), np.float32)
for kt in range(K_TILE):
    S += Q[:, kt*BLK_K:(kt+1)*BLK_K] @ Kk[:, kt*BLK_K:(kt+1)*BLK_K].T
S *= scale
show_mat("S = scale * Q . K^T", S)

# ----------------------------------------------------------------------
# Softmax: S -> P
# ----------------------------------------------------------------------
# Row softmax over seq_k keys.
row_max = S.max(1, keepdims=True); P = np.exp(S - row_max); P = P / P.sum(1, keepdims=True)
show_mat("row_max", row_max); show_mat("P = softmax(S)", P); check("rows sum to 1", np.allclose(P.sum(1), 1))

# ----------------------------------------------------------------------
# LDSM warp-row (fred.max co-residence)
# ----------------------------------------------------------------------
# LDSM warp-row: a query row's scores co-resident in one 32-lane subgroup.
ldsm = make_xe4_ldstm("LDSM", VS=8, s=DT)
print("LDSM atom:", ldsm.name, " ThrID =", size(ldsm.thr_id))
warp = {t: S[0, t] for t in range(min(seq_k, 32))}
check("in-warp max == numpy", max(warp.values()) == S[0, :min(seq_k, 32)].max())

# ----------------------------------------------------------------------
# GEMM-2 mainloop: O = P*V
# ----------------------------------------------------------------------
# GEMM-2 mainloop (TiledMmaPV): O = sum_k P_k . V_k.
O = np.zeros((seq_q, head), np.float32)
for kt in range(seq_k // BLK_K):
    O += P[:, kt*BLK_K:(kt+1)*BLK_K] @ V[kt*BLK_K:(kt+1)*BLK_K, :]
show_mat("O = P . V", O)

# ----------------------------------------------------------------------
# Store O
# ----------------------------------------------------------------------
slmO = make_slm_layout_elem(DBITS, seq_q, head); obuf = load_to_slm(O, slmO)
check("O store round-trip", np.array_equal(read_slm(obuf, slmO, seq_q, head), O))

# ----------------------------------------------------------------------
# Reference attention + validation
# ----------------------------------------------------------------------
S_ref = scale * (Q @ Kk.T)
Pr = np.exp(S_ref - S_ref.max(1, keepdims=True)); Pr = Pr / Pr.sum(1, keepdims=True)
O_ref = Pr @ V
err = np.abs(O - O_ref).max(); print("max abs error vs reference attention:", err)
check("FMHA4 verification", err < 1e-5)

# ----------------------------------------------------------------------
# Recap
# ----------------------------------------------------------------------
print('Recap: load Q,K → QK K-loop → softmax → LDSM → PV K-loop → store → validate.')
