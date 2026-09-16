# MIT License
#
# Copyright (c) 2026 Meta Platforms, Inc. and affiliates.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Intel Xe5 CuTe atoms — AMMA (single-group, dual-group, LARGE, block-scaled)
and ADMA copies (inherited Xe4, dual-group, linear-GCS).

Translated from the sycl-tla (CUTLASS-Xe) traits:
  include/cute/arch/mma_xe5{,_amma,_blockscaled,_desc}.hpp
  include/cute/atom/mma_traits_xe5_{amma,blockscaled}.hpp
  include/cute/arch/copy_xe5_adma.hpp ; atom/copy_traits_xe5_adma.hpp

What Xe5 adds over Xe4:

* **Dual-group MMA** — a lane *pair* (ThrID = 2) splits M (and N) in half; the
  SM100 "2x1SM" convention.  This is the ``[DUAL]`` layout pattern below.
* **Cluster vs. workgroup barrier scope** — ``_CLUSTER`` (masked, ``.atmm``) vs.
  ``_WG`` (maskless, ``.atm``).  As with Xe4 tracking, scope is a *type*
  parameter, not a layout property; variants share A/B/C layouts.
* **LARGE core matrices** — bigger SLM descriptor tiles (same TV layout).
* **Block-scaled MX with fp6** (e3m2 / e2m3, stored in 8-bit containers) and
  Type-III scale-factor descriptors.
* **Linear-load GCS** (``.gs``) copies whose global source and SLM destination
  differ in size (hardware zero-fills the SLM remainder).
"""

from __future__ import annotations

from .atoms import MMAAtom, CopyAtom
from .layouts import Layout
from .atoms_xe_common import sizeof_bits, make_slm_layout_elem


# =============================================================================
# AMMA descriptor TV layouts — [SG] single-group and [DUAL] dual-group
#   (mma_traits_xe5_amma.hpp).  Shape_MNK is always (M, N, K).
# =============================================================================

def _sg_layouts(M, N, K):
    """Single-group [SG]: ThrID=1, operands are one-thread descriptors."""
    a = Layout((1, (M, K)), (0, (1, M)))
    b = Layout((1, (N, K)), (0, (1, N)))
    c = Layout((1, (M, N)), (0, (1, M)))
    return Layout(1), a, b, c


def _dual_layouts(M, N, K):
    """Dual-group [DUAL]: ThrID=2, M and N split across the lane pair."""
    a = Layout((2, (M // 2, K)), (M // 2, (1, M)))
    b = Layout((2, (N // 2, K)), (N // 2, (1, N)))
    c = Layout((2, (M // 2, N)), (M // 2, (1, M)))
    return Layout(2), a, b, c


# Wired single-group tracking variants (mma_xe5_amma.hpp).
XE5_SG_TRACKING = {
    "":            "amma",
    "GROUPSYNC":   "amma.groupsync",
    "D":           "amma.dtm",
    "AB":          "amma.atm.btm",
    "DAB":         "amma.dtm.atm.btm",
    "DB":          "amma.dtm.btm",
    "AB_CLUSTER":  "amma.atmm.btmm (cluster)",
    "DB_CLUSTER":  "amma.dtm.btmm (cluster)",
    "DAB_CLUSTER": "amma.dtm.atmm.btmm (cluster)",
}

# Wired dual-group variants: (suffix, ptx-scope).
XE5_DUAL_VARIANTS = {
    "AB_CLUSTER_DUAL":  "amma.atmm.btmm (cluster, dual)",
    "DB_CLUSTER_DUAL":  "amma.dtm.btmm (cluster, dual)",
    "DAB_CLUSTER_DUAL": "amma.dtm.atmm.btmm (cluster, dual)",
    "D_WG_DUAL":        "amma.dtm (workgroup, dual)",
    "AB_WG_DUAL":       "amma.atm.btm (workgroup, dual)",
    "DB_WG_DUAL":       "amma.dtm.btm (workgroup, dual)",
    "DAB_WG_DUAL":      "amma.dtm.atm.btm (workgroup, dual)",
}

# Single-group LARGE variants (bigger SLM core matrix; same TV layout).
XE5_LARGE_VARIANTS = {
    "LARGE":           "amma (large cm)",
    "GROUPSYNC_LARGE": "amma.groupsync (large cm)",
    "D_LARGE":         "amma.dtm (large cm)",
    "DB_LARGE":        "amma.dtm.btm (large cm)",
    "AB_LARGE":        "amma.atm.btm (large cm)",
    "DAB_LARGE":       "amma.dtm.atm.btm (large cm)",
}


def make_xe5_amma_atom(prefix, ptx, d, a, b, c, M, N, K, dual=False, block_scaled=False):
    thr, a_l, b_l, c_l = (_dual_layouts(M, N, K) if dual else _sg_layouts(M, N, K))
    bs = ".ascale.bscale" if block_scaled else ""
    name = f"{prefix}_{M}x{N}x{K}_{d.upper()}{a.upper()}{b.upper()}{c.upper()}"
    return MMAAtom(
        name=name, ptx=ptx + bs, shape_mnk=(M, N, K),
        thr_id=thr, a_layout=a_l, b_layout=b_l, c_layout=c_l,
    )


# Representative shape 128x128x32 bf16.
_M, _N, _K = 128, 128, 32

MMA_ATOMS_XE5_SG = [
    make_xe5_amma_atom(f"XE5_AMMA_{t}" if t else "XE5_AMMA", p,
                       "f32", "bf16", "bf16", "f32", _M, _N, _K)
    for t, p in XE5_SG_TRACKING.items()
]
MMA_ATOMS_XE5_DUAL = [
    make_xe5_amma_atom(f"XE5_AMMA_{t}", p,
                       "f32", "bf16", "bf16", "f32", _M, _N, _K, dual=True)
    for t, p in XE5_DUAL_VARIANTS.items()
]
MMA_ATOMS_XE5_LARGE = [
    make_xe5_amma_atom(f"XE5_AMMA_{t}", p,
                       "f32", "bf16", "bf16", "f32", _M, _N, _K)
    for t, p in XE5_LARGE_VARIANTS.items()
]
# Block-scaled MX (incl. fp6 e3m2 / e2m3) — single-group + dual.
MMA_ATOMS_XE5_BS = [
    make_xe5_amma_atom("XE5_AMMA_BlockScaled", "amma",
                       "f32", "e2m3", "e2m3", "f32", _M, _N, _K, block_scaled=True),
    make_xe5_amma_atom("XE5_AMMA_BlockScaled_LARGE", "amma (large cm)",
                       "f32", "e3m2", "e3m2", "f32", _M, _N, _K, block_scaled=True),
    make_xe5_amma_atom("XE5_AMMA_BlockScaled_AB_CLUSTER_DUAL", "amma.atmm.btmm (cluster, dual)",
                       "f32", "e4m3", "e4m3", "f32", _M, _N, _K, dual=True, block_scaled=True),
]

MMA_ATOMS_XE5 = (MMA_ATOMS_XE5_SG + MMA_ATOMS_XE5_DUAL
                 + MMA_ATOMS_XE5_LARGE + MMA_ATOMS_XE5_BS)

for _atom in MMA_ATOMS_XE5:
    globals()[_atom.name] = _atom


def slm_operand_layout(dtype, rows, K):
    """Physical SLM element layout for an Xe5 AMMA operand (rows x K tile)."""
    return make_slm_layout_elem(sizeof_bits(dtype), rows, K)


# =============================================================================
# XE5 ADMA copy atoms — copy_traits_xe5_adma.hpp
# =============================================================================

def _adma_placeholder(name, ptx, num_bits, thr=1):
    L = Layout((1, num_bits))
    return CopyAtom(name=name, ptx=ptx, thr_id=Layout(thr),
                    src_layout_bits=L, dst_layout_bits=L)


# Inherited Xe4 aliases (identical structs -> Xe4 traits).
COPY_ATOMS_XE5_INHERITED = [
    _adma_placeholder("XE5_ADMA_LOAD", "= XE4_ADMA_LOAD", 1024),
    _adma_placeholder("XE5_ADMA_STORE", "= XE4_ADMA_STORE", 1024),
    _adma_placeholder("XE5_ADMA_LOAD_MULTICAST", "= XE4_ADMA_LOAD_MULTICAST", 1024),
    _adma_placeholder("XE5_ADMA_PREFETCH", "= XE4_ADMA_PREFETCH", 1024),
    _adma_placeholder("XE5_ADMA_STORE_REDUCE", "= XE4_ADMA_STORE_REDUCE", 1024),
]

# Dual-group tensor loads (even-routing via %clusterwglinearid).
COPY_ATOMS_XE5_DUAL = [
    _adma_placeholder("XE5_ADMA_LOAD_DUAL", "adma .dual.shared_cluster (wg_mask)", 1024),
    _adma_placeholder("XE5_ADMA_LOAD_DUAL_WG", "adma .dual.shared_workgroup", 1024),
]


def make_xe5_linear_gcs(name, ptx, glb_bytes, copy_bytes):
    """Linear-load GCS (.gs): global src bytes -> SLM dst fill (HW zero-fills).

    Src (global) and Dst (SLM) differ in size; RefLayout = Dst.
    """
    return CopyAtom(
        name=name, ptx=ptx, thr_id=Layout(1),
        src_layout_bits=Layout((1, glb_bytes * 8)),
        dst_layout_bits=Layout((1, copy_bytes * 8)))


COPY_ATOMS_XE5_GCS = [
    make_xe5_linear_gcs("XE5_ADMA_LINEAR_LOAD_GCS",
                        "adma linear load .gs (gmem->SLM, zero-fill)", 48, 64),
    make_xe5_linear_gcs("XE5_ADMA_LINEAR_LOAD_MULTICAST_GCS",
                        "adma linear load multicast .gs", 48, 64),
]

COPY_ATOMS_XE5 = (COPY_ATOMS_XE5_INHERITED + COPY_ATOMS_XE5_DUAL
                  + COPY_ATOMS_XE5_GCS)

for _atom in COPY_ATOMS_XE5:
    globals()[_atom.name] = _atom
