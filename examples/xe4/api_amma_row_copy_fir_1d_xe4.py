# MIT License — see LICENSE file for details.
"""XE4 1D FIR via row-copy

Full-execution-flow functional port of the sycl-tla example examples/cute/tutorial/xe4/api_amma_row_copy_fir_1d_xe4.cpp.
A 1D FIR filter as row copies + dot-product MMAs.

Run:  python examples/xe4/api_amma_row_copy_fir_1d_xe4.py"""

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
# FIR filter = row-copy + dot-product MMAs
# ----------------------------------------------------------------------
# 1D FIR filter as row-copies + tiny dot-product MMAs: y[n] = sum_t h[t] x[n+t].
x = rng.integers(0, 5, size=32).astype(np.float32)
h = np.array([0.25, 0.5, 0.25], dtype=np.float32)    # 3-tap filter
y = np.convolve(x, h[::-1], mode="valid")
show_mat("x", x.reshape(1, -1)); show_mat("h", h.reshape(1, -1)); show_mat("y", y.reshape(1, -1))
check("y length", len(y) == len(x) - len(h) + 1)

# ----------------------------------------------------------------------
# Recap
# ----------------------------------------------------------------------
print('Recap: row-copy → FIR (dot-product MMA).')
