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

"""Intel Xe4 CuTe atoms — MMA (TMM, AMMA) and Copy (ADMA, LDSM, EU_COPY).

Translated from the sycl-tla (CUTLASS-Xe) traits:
  include/cute/arch/mma_xe4{,_amma,_tmm}.hpp
  include/cute/atom/mma_traits_xe4{,_amma,_tmm}.hpp
  include/cute/arch/copy_xe4_{adma,dma_legacy,eu_copy,ldsm}.hpp
  include/cute/atom/copy_traits_xe4_{adma,dma_legacy,eu_copy,ldsm}.hpp

Two structural facts drive this module (and differ from NVIDIA / Xe-legacy):

* **Xe4 subgroup = 32 lanes.**
* **AMMA / ADMA operands are SLM matrix *descriptors*.**  Their MMA_Traits /
  Copy_Traits carry a single-thread placeholder layout ``(1, M, K)`` /
  ``(1, NumBits)`` — the *physical* data movement is the SLM core-matrix layout
  built by :func:`atoms_xe_common.make_slm_layout_elem`.  We attach both: the
  faithful trait placeholder (so the atom matches the C++ trait) and, for MMA
  atoms, an ``slm_a/slm_b`` element layout so notebooks can show the real tile.

* **Barrier tracking (None/GroupSync/D/DB/AB/DAB + cluster) is a *type*
  parameter, not a layout property.**  Every tracking variant shares identical
  A/B/C descriptor layouts, so we build one factory and stamp the named
  variants (matching ``AMMA::Tracking``); the difference lives in ``.ptx``.
"""

from __future__ import annotations

from .atoms import MMAAtom, CopyAtom
from .layouts import Layout
from .atoms_xe_common import (
    sizeof_bits, make_ordered_layout, make_slm_layout_elem, SG_SIZE_XE4,
)


# =============================================================================
# XE4_TMM<d,a,b,c, N, kSubGroupSize=32>  —  synchronous register DPAS
#   arch/mma_xe4_tmm.hpp:137 ; atom/mma_traits_xe4_tmm.hpp:247
#   M is fixed at 32.  K = 8 (tf32) / 16 (fp16,bf16) / 32 (fp8,int8).
# =============================================================================

_TMM_M = 32


def _tmm_k(a, b):
    bits = max(sizeof_bits(a), sizeof_bits(b))
    return {32: 8, 16: 16, 8: 32, 4: 32}[bits]


def make_xe4_tmm_atom(name, d, a, b, c, N=32):
    """Create an XE4_TMM atom (register-resident, M=32, subgroup=32).

    A and C layouts are exact (M-major, lane i owns row i).  The B layout's
    true form depends on the ``packF``/``nRegsB``/``tmmLayoutBMaxLanes`` helpers
    (VNNI register packing); we model B as the natural (N, K) col-major view and
    flag it — the register-packing detail is out of scope for the emulator.
    """
    M, K = _TMM_M, _tmm_k(a, b)
    return MMAAtom(
        name=name,
        ptx=f"tmm {M}x{N}x{K} ({a}x{b}->{d}) [B layout simplified]",
        shape_mnk=(M, N, K),
        thr_id=Layout(SG_SIZE_XE4),
        a_layout=Layout((M, K), (1, M)),          # tmm::ALayout — M-major
        b_layout=Layout((N, K), (1, N)),          # simplified; true form VNNI-packed
        c_layout=Layout((M, N), (1, M)),          # tmm::CLayout = ALayout<c,M,N>
    )


MMA_ATOMS_XE4_TMM = [
    make_xe4_tmm_atom("XE4_TMM_32x32x16_F32BF16BF16F32", "f32", "bf16", "bf16", "f32"),
    make_xe4_tmm_atom("XE4_TMM_32x32x16_F32F16F16F32", "f32", "f16", "f16", "f32"),
    make_xe4_tmm_atom("XE4_TMM_32x32x8_F32TF32TF32F32", "f32", "tf32", "tf32", "f32"),
    make_xe4_tmm_atom("XE4_TMM_32x32x32_S32S8S8S32", "s32", "s8", "s8", "s32"),
    make_xe4_tmm_atom("XE4_TMM_32x32x32_F32E4M3E4M3F32", "f32", "e4m3", "e4m3", "f32"),
]


# =============================================================================
# XE4_AMMA<d,a,b,c, M, N, K, a_major, b_major>  —  async SLM-descriptor MMA
#   arch/mma_xe4_amma.hpp ; atom/mma_traits_xe4_amma.hpp
#   ThrID = 1 ; A/B/C are single-thread descriptor placeholders.
# =============================================================================

# AMMA::Tracking enumerators that are actually wired to an atom struct.
AMMA_TRACKING = {
    "":          "amma",              # None
    "GROUPSYNC": "amma.groupsync",
    "D":         "amma.dtm",
    "DB":        "amma.dtm.btm",
    "AB":        "amma.atm.btm",
    "DAB":       "amma.dtm.atm.btm",
    "AB_CLUSTER":  "amma.atmm.btmm (cluster, a/b mask)",
    "DB_CLUSTER":  "amma.dtm.btmm (cluster, b mask)",
    "DAB_CLUSTER": "amma.dtm.atmm.btmm (cluster, a/b mask)",
}


def _amma_layouts(M, N, K):
    """Single-thread descriptor placeholder layouts (mma_traits_xe4_amma.hpp)."""
    a = Layout((1, (M, K)), (0, (1, M)))
    b = Layout((1, (N, K)), (0, (1, N)))
    c = Layout((1, (M, N)), (0, (1, M)))
    return a, b, c


def make_xe4_amma_atom(d, a, b, c, M, N, K, tracking="", block_scaled=False):
    """Create an XE4_AMMA atom for a given barrier-tracking variant.

    All tracking variants share the descriptor A/B/C layouts; ``tracking``
    only changes the barrier suffix (``.ptx``).  The physical SLM tile for A/B
    is available via :func:`make_slm_layout_elem` (see ``slm_a_layout`` /
    ``slm_b_layout`` helpers below).
    """
    a_l, b_l, c_l = _amma_layouts(M, N, K)
    bs = "_BlockScaled" if block_scaled else ""
    trk = f"_{tracking}" if tracking else ""
    name = f"XE4_AMMA{bs}{trk}_{M}x{N}x{K}_{d.upper()}{a.upper()}{b.upper()}{c.upper()}"
    return MMAAtom(
        name=name,
        ptx=AMMA_TRACKING[tracking] + (".ascale.bscale" if block_scaled else ""),
        shape_mnk=(M, N, K),
        thr_id=Layout(1),
        a_layout=a_l, b_layout=b_l, c_layout=c_l,
    )


def slm_a_layout(a_dtype, M, K):
    """Physical SLM element layout for the AMMA A operand (M x K tile)."""
    return make_slm_layout_elem(sizeof_bits(a_dtype), M, K)


def slm_b_layout(b_dtype, N, K):
    """Physical SLM element layout for the AMMA B operand (N x K tile)."""
    return make_slm_layout_elem(sizeof_bits(b_dtype), N, K)


# One representative shape (128x128x32 bf16), all wired tracking variants.
MMA_ATOMS_XE4_AMMA = [
    make_xe4_amma_atom("f32", "bf16", "bf16", "f32", 128, 128, 32, tracking=t)
    for t in AMMA_TRACKING
]
# Block-scaled (MX) variants — same layouts + scale-factor descriptor.
MMA_ATOMS_XE4_AMMA_BS = [
    make_xe4_amma_atom("f32", "e4m3", "e4m3", "f32", 128, 128, 32,
                       tracking=t, block_scaled=True)
    for t in ("", "D", "AB", "DAB", "AB_CLUSTER", "DAB_CLUSTER")
]

for _atom in MMA_ATOMS_XE4_TMM + MMA_ATOMS_XE4_AMMA + MMA_ATOMS_XE4_AMMA_BS:
    globals()[_atom.name] = _atom


# =============================================================================
# XE4 ADMA copy atoms — copy_traits_xe4_adma.hpp
#   Async DMA (gmem<->SLM) copy engine.  Trait layouts are single-thread (or
#   32-thread per-warp) bit placeholders; the real tile is the SLM layout.
# =============================================================================

def _adma_linear(name, ptx, num_bytes):
    bits = num_bytes * 8
    L = Layout((1, bits))
    return CopyAtom(name=name, ptx=ptx, thr_id=Layout(1),
                    src_layout_bits=L, dst_layout_bits=L)


def _adma_row(name, ptx, num_bytes, per_warp):
    bits = num_bytes * 8
    if per_warp:
        thr = Layout(SG_SIZE_XE4)
        L = Layout((SG_SIZE_XE4, bits), (bits, 1))
    else:
        thr = Layout(1)
        L = Layout((1, bits))
    return CopyAtom(name=name, ptx=ptx, thr_id=thr,
                    src_layout_bits=L, dst_layout_bits=L)


COPY_ATOMS_XE4_ADMA_LINEAR = [
    _adma_linear("XE4_ADMA_LINEAR_LOAD", "adma linear load (gmem->SLM)", 64),
    _adma_linear("XE4_ADMA_LINEAR_STORE", "adma linear store (SLM->gmem)", 64),
    _adma_linear("XE4_ADMA_LINEAR_LOAD_MULTICAST_CLUSTER",
                 "adma linear load multicast (cluster)", 64),
    _adma_linear("XE4_ADMA_LINEAR_LOAD_LOCAL_TO_REMOTE_SLM_CLUSTER",
                 "adma SLM->remote-SLM (cluster)", 64),
    _adma_linear("XE4_ADMA_LINEAR_PREFETCH", "adma linear prefetch", 64),
    _adma_linear("XE4_ADMA_LINEAR_REDUCE", "adma linear reduce (SLM->gmem atomic)", 64),
]

COPY_ATOMS_XE4_ADMA_ROW = [
    _adma_row("XE4_ADMA_ROW_COPY_LINEAR_LOAD", "adma row load (per-lane)", 32, False),
    _adma_row("XE4_ADMA_ROW_COPY_LINEAR_LOAD_WARP", "adma row load (per-warp)", 32, True),
    _adma_row("XE4_ADMA_ROW_COPY_LINEAR_STORE", "adma row store (per-lane)", 32, False),
    _adma_row("XE4_ADMA_ROW_PREFETCH", "adma row prefetch", 32, False),
    _adma_row("XE4_ADMA_ROW_COPY_TILED_LOAD", "adma row tiled load", 32, True),
]

COPY_ATOMS_XE4_ADMA_TENSOR = [
    _adma_linear("XE4_ADMA_LOAD", "adma tensor load (descriptor)", 128),
    _adma_linear("XE4_ADMA_STORE", "adma tensor store (descriptor)", 128),
    _adma_linear("XE4_ADMA_LOAD_MULTICAST", "adma tensor load multicast", 128),
    _adma_linear("XE4_ADMA_PREFETCH", "adma tensor prefetch", 128),
    _adma_linear("XE4_ADMA_STORE_REDUCE", "adma tensor store-reduce", 128),
]


# =============================================================================
# XE4_LDSM / XE4_STSM  —  copy_traits_xe4_dma_legacy.hpp:318,328
#   ThrID = 32 ; Src/Dst = make_ordered_layout((32,(1, bits*VS)), Step<2,(1,0)>)
#   This is the exact SLM<->reg warp copy layout.
# =============================================================================

def make_xe4_ldstm(op, VS=8, s="bf16", d=None):
    d = d or s
    bs, bd = sizeof_bits(s), sizeof_bits(d)
    src = make_ordered_layout((SG_SIZE_XE4, (1, bs * VS)), (2, (1, 0)))
    dst = make_ordered_layout((SG_SIZE_XE4, (1, bd * VS)), (2, (1, 0)))
    return CopyAtom(
        name=f"XE4_{op}_VS{VS}_{s.upper()}",
        ptx=f"xe4 {'ldsm' if op == 'LDSM' else 'stsm'} vs={VS} ({s})",
        thr_id=Layout(SG_SIZE_XE4),
        src_layout_bits=src, dst_layout_bits=dst)


COPY_ATOMS_XE4_LDSM = [
    make_xe4_ldstm("LDSM", VS=8, s="bf16"),
    make_xe4_ldstm("LDSM", VS=16, s="s8"),
    make_xe4_ldstm("STSM", VS=8, s="bf16"),
]


# =============================================================================
# XE4_LOAD_MATRIX / XE4_STORE_MATRIX / XE4_REDUCE_MATRIX
#   copy_traits_xe4_ldsm.hpp — DataSize = BitWidth*Vlen*Alen placeholder;
#   executable trait adds ThrLayout = Layout<_32>.
# =============================================================================

def make_xe4_matrix_op(op, s="bf16", vlen=8, alen=1):
    data_bits = sizeof_bits(s) * vlen * alen
    L = Layout((1, data_bits))
    return CopyAtom(
        name=f"XE4_{op}_MATRIX_{s.upper()}_v{vlen}",
        ptx=f"xe4 {op.lower()}_matrix vlen={vlen} alen={alen} ({s})",
        thr_id=Layout(SG_SIZE_XE4),      # executable ThrLayout = _32
        src_layout_bits=L, dst_layout_bits=L)


COPY_ATOMS_XE4_MATRIX = [
    make_xe4_matrix_op("LOAD", "bf16", vlen=8),
    make_xe4_matrix_op("STORE", "bf16", vlen=8),
    make_xe4_matrix_op("REDUCE", "f32", vlen=8),   # reduce-in-flight (Xe4-only)
]


# =============================================================================
# XE4_EU_COPY_{G2S,S2G,S2R}  —  copy_traits_xe4_eu_copy.hpp
#   Copies that run on the EU compute pipeline.  The meaningful transformation
#   is the full linear (M x K, row-major) <-> SLM core-matrix layout.
#   We expose that pair directly (element coordinates), plus a CopyAtom whose
#   src/dst bit-layouts are the linear and SLM tiles.
# =============================================================================

def make_xe4_eu_copy(kind, t="bf16", M=32, K=32, src_stride=None):
    """kind in {G2S, S2G, S2R}.  Linear side is (M,K):(src_stride,1)."""
    src_stride = src_stride or K
    linear = Layout((M, K), (src_stride, 1))            # FullSrc/FullDst linear
    slm = make_slm_layout_elem(sizeof_bits(t), M, K)    # FullDst/FullSrc SLM
    if kind == "G2S":
        src, dst = linear, slm
    else:  # S2G, S2R  (SLM -> linear/reg)
        src, dst = slm, linear
    return CopyAtom(
        name=f"XE4_EU_COPY_{kind}_{t.upper()}_{M}x{K}",
        ptx=f"xe4 eu_copy {kind} ({t}) {M}x{K} [element coords]",
        thr_id=Layout(SG_SIZE_XE4),
        src_layout_bits=src, dst_layout_bits=dst)


COPY_ATOMS_XE4_EU = [
    make_xe4_eu_copy("G2S", "bf16", 32, 32),
    make_xe4_eu_copy("S2G", "bf16", 32, 32),
    make_xe4_eu_copy("S2R", "bf16", 32, 32),
]


COPY_ATOMS_XE4 = (
    COPY_ATOMS_XE4_ADMA_LINEAR + COPY_ATOMS_XE4_ADMA_ROW
    + COPY_ATOMS_XE4_ADMA_TENSOR + COPY_ATOMS_XE4_LDSM
    + COPY_ATOMS_XE4_MATRIX + COPY_ATOMS_XE4_EU
)
MMA_ATOMS_XE4 = MMA_ATOMS_XE4_TMM + MMA_ATOMS_XE4_AMMA + MMA_ATOMS_XE4_AMMA_BS

for _atom in COPY_ATOMS_XE4:
    globals()[_atom.name] = _atom
