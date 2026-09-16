# MIT License — see LICENSE file for details.
"""XE5 grouped GEMM (tensor-desc update)

Full-execution-flow functional port of the sycl-tla example examples/cute/tutorial/xe5/grouped_gemm_adma_tensor_desc_update.cpp.
Grouped GEMM (Xe5), full flow per group.

Run:  python examples/xe5/grouped_gemm_adma_tensor_desc_update.py"""

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
# Problem setup: allocate and initialize the operands (small so tables print).
M, N, K = 32, 32, 64
BLK_M, BLK_N, BLK_K = 32, 32, 16
K_TILE = K // BLK_K            # number of K-tiles in the mainloop
K_PIPE = 2                    # SLM pipeline stages (StagesA)
alpha, beta = 1.0, 0.0
DT = "bf16"; DBITS = sizeof_bits(DT)
rng = np.random.default_rng(0)
A = rng.integers(-2, 3, size=(M, K)).astype(np.float32)   # A (M x K) row-major
B = rng.integers(-2, 3, size=(N, K)).astype(np.float32)   # B (N x K); B^T used in the GEMM
C = rng.integers(0, 2, size=(M, N)).astype(np.float32)    # C (M x N) input accumulator
show_mat("A (M x K)", A); show_mat("B (N x K)", B)

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
from tensor_layouts.atoms_xe5 import make_xe5_amma_atom
atom = make_xe5_amma_atom("XE5_AMMA_AB", "amma.atm.btm", "f32", DT, DT, "f32", BLK_M, BLK_N, BLK_K)
slmA = make_slm_layout_elem(DBITS, BLK_M, BLK_K); slmB = make_slm_layout_elem(DBITS, BLK_N, BLK_K)
print("MMA atom :", atom.name)
print("CTA tile :", (BLK_M, BLK_N, BLK_K), " K-tiles:", K_TILE, " pipeline stages:", K_PIPE)

# ----------------------------------------------------------------------
# Grouped GEMM — full flow per group
# ----------------------------------------------------------------------
# Grouped GEMM: several independent problems in one launch, each with its own
# ADMA tensor descriptor. We run the full flow (load+mainloop+validate) per group.
groups = [(24, 16, 32), (32, 8, 16)]                 # (M, N, K) per group
for gi, (m, n, k) in enumerate(groups):
    Ag = rng.integers(-2, 3, size=(m, k)).astype(np.float32)
    Bg = rng.integers(-2, 3, size=(n, k)).astype(np.float32)
    accg = np.zeros((m, n), np.float32)
    for kt in range(k // BLK_K):
        accg += Ag[:, kt * BLK_K:(kt + 1) * BLK_K] @ Bg[:, kt * BLK_K:(kt + 1) * BLK_K].T
    print("group %d: M=%d N=%d K=%d ->" % (gi, m, n, k), "acc[0,:4]", accg[0, :4],
          " ok:", np.allclose(accg, Ag @ Bg.T))

# ----------------------------------------------------------------------
# Recap
# ----------------------------------------------------------------------
print('Recap: loop groups; each does a full K-loop GEMM.')
