# MIT License — see LICENSE file for details.
"""XE4 tiled TMM GEMM

Full-execution-flow functional port of the sycl-tla example examples/cute/tutorial/xe4/gemm_tiled_tmma.cpp.
Register TMM MMA (M=32, lane i owns row i).

Run:  python examples/xe4/gemm_tiled_tmma.py"""

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
# Config — tile shape, MMA atom, SLM layouts, pipeline stages
# ----------------------------------------------------------------------
from tensor_layouts.atoms_xe4 import make_xe4_tmm_atom
atom = make_xe4_tmm_atom("XE4_TMM", "f32", DT, DT, "f32", N=BLK_N)
slmA = make_slm_layout_elem(DBITS, BLK_M, BLK_K)
slmB = make_slm_layout_elem(DBITS, BLK_N, BLK_K)
print("MMA atom :", atom.name, " (register TMM, lane i owns row i)")
print("CTA tile :", (BLK_M, BLK_N, BLK_K), " K-tiles:", K_TILE)

# ----------------------------------------------------------------------
# TMM register ownership
# ----------------------------------------------------------------------
# TMM register ownership: lane i owns row i of A and of the accumulator.
Areg = Layout((BLK_M, BLK_K), (1, BLK_M))
print("A register layout (lane, value) -> element (M-major):")
show_layout(Areg, BLK_M, BLK_K, rl="lane", cl="v")

# ----------------------------------------------------------------------
# Tiling — CTA grid + K-tiles
# ----------------------------------------------------------------------
# Partition the problem: CTA grid over (M,N), and K split into K_TILE tiles.
print("CTA grid (M,N):", (M // BLK_M, N // BLK_N), "   K-tiles per CTA:", K_TILE)
a_kt = [A[:, k * BLK_K:(k + 1) * BLK_K] for k in range(K_TILE)]   # A K-tiles (M x BLK_K)
b_kt = [B[:, k * BLK_K:(k + 1) * BLK_K] for k in range(K_TILE)]   # B K-tiles (N x BLK_K)
show_mat("A K-tile 0 (M x BLK_K)", a_kt[0])

# ----------------------------------------------------------------------
# SLM staging — bank-swizzled operand tiles
# ----------------------------------------------------------------------
# SLM staging: each K-tile is copied into a bank-swizzled SLM buffer.
print("SLM A stage layout (m,k) -> SLM offset (bank-swizzled):")
show_layout(slmA, BLK_M, BLK_K)
_probe = load_to_slm(a_kt[0], slmA)
check("SLM stage round-trip is loss-less", np.array_equal(read_slm(_probe, slmA, BLK_M, BLK_K), a_kt[0]))

# ----------------------------------------------------------------------
# Clear the accumulator
# ----------------------------------------------------------------------
# Clear the accumulator fragment (lives in registers on hardware).
acc = np.zeros((BLK_M, BLK_N), dtype=np.float32)
show_mat("cleared accumulator", acc)

# ----------------------------------------------------------------------
# Prologue — fill the SLM pipeline stages
# ----------------------------------------------------------------------
# Prologue: fill the first K_PIPE SLM pipeline stages (double-buffering).
slm_pipe_a = [load_to_slm(a_kt[p], slmA) for p in range(min(K_PIPE, K_TILE))]
slm_pipe_b = [load_to_slm(b_kt[p], slmB) for p in range(min(K_PIPE, K_TILE))]
print("prologue filled", len(slm_pipe_a), "of", K_PIPE, "SLM stages;", K_TILE, "K-tiles total")

# ----------------------------------------------------------------------
# Mainloop — K-tile load + MMA accumulation
# ----------------------------------------------------------------------
# Mainloop: for each K-tile, load it into SLM (producer) and accumulate the
# partial product into acc (consumer MMA). This is the K-reduction.
for kt in range(K_TILE):
    a_k = read_slm(load_to_slm(a_kt[kt], slmA), slmA, BLK_M, BLK_K)   # staged through SLM
    b_k = b_kt[kt]
    acc += a_k @ b_k.T                                               # cute::gemm accumulate
    if kt == 0:
        show_mat("acc after K-tile 0", acc)
show_mat("acc after full K-loop", acc)
check("acc == A . B^T", np.allclose(acc, A @ B.T))

# ----------------------------------------------------------------------
# Epilogue
# ----------------------------------------------------------------------
# Epilogue: D = alpha*acc + beta*C.
D = alpha * acc + beta * C
show_mat("D = alpha*acc + beta*C", D)

# ----------------------------------------------------------------------
# Store → gmem
# ----------------------------------------------------------------------
# Store D to gmem through the SLM core matrix; verify loss-less.
slmD = make_slm_layout_elem(DBITS, M, N)
dbuf = load_to_slm(D, slmD)
check("store round-trip", np.array_equal(read_slm(dbuf, slmD, M, N), D))

# ----------------------------------------------------------------------
# Reference + validation
# ----------------------------------------------------------------------
# Reference GEMM + validation (like validate_gemm_result).
D_ref = alpha * (A @ B.T) + beta * C
err = np.abs(D - D_ref).max()
print("max abs error vs reference:", err)
check("GEMM verification", err < 1e-3)

# ----------------------------------------------------------------------
# Recap
# ----------------------------------------------------------------------
print('Recap: load → tile → stage → clear → prologue → K-loop MMA → epilogue → store → validate.')
