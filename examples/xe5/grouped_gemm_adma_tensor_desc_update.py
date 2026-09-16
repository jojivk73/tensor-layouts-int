# MIT License — see LICENSE file for details.
"""XE5 grouped GEMM (tensor-desc update)

Functional port of the sycl-tla example examples/cute/tutorial/xe5/grouped_gemm_adma_tensor_desc_update.cpp, runnable on the CPU.
Grouped GEMM (Xe5), per-group ADMA descriptors.

Run:  python examples/xe5/grouped_gemm_adma_tensor_desc_update.py
Each section below is one operation; the same content is in grouped_gemm_adma_tensor_desc_update.ipynb as cells."""

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
# Step — grouped GEMM
# ----------------------------------------------------------------------
# Grouped GEMM: several independent problems share one launch. Each group has
# its own A/B (different sizes) and its own ADMA tensor descriptor.
groups = [(24, 16, 8), (32, 8, 16)]          # (M, N, K) per group
for gi, (m, n, k) in enumerate(groups):
    Ag = rng.integers(-2, 3, size=(m, k)).astype(np.float32)
    Bg = rng.integers(-2, 3, size=(n, k)).astype(np.float32)
    Sg = Ag @ Bg.T
    print("group", gi, "M=%d N=%d K=%d ->" % (m, n, k), "S", Sg.shape, " S[0,:4]", Sg[0, :4])

# ----------------------------------------------------------------------
# Recap
# ----------------------------------------------------------------------
print('Recap: loop over groups, one GEMM each.')
