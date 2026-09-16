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

"""Intel Xe GPU DPAS (Dot Product Accumulate Systolic) atom definitions.

Mirrors the lane-to-element mapping of Intel's DPAS instruction on Xe
architectures:
  - Xe-HPC (Ponte Vecchio / Data Center Max): subgroup_size = 8
  - Xe-HPG (Arc / DG2):                       subgroup_size = 16

Each DPAS atom maps (thread_idx, value_idx) -> element coordinate using
column-major encoding:
    A: (T, V) -> m + k*M   in the M × K matrix
    B: (T, V) -> n + k*N   in the N × K matrix
    C: (T, V) -> m + n*M   in the M × N matrix

DPAS Register Layout
====================

DPAS is a subgroup-cooperative systolic instruction:

    dpas.{systolic_depth}x{repeat_count} (exec_size) dst src0 src1 src2

    systolic_depth (sd):  8 (fixed on all Xe architectures) — the K dimension
    repeat_count (rc):    1–8 — the M dimension (typically 8 for peak throughput)
    exec_size:            N dimension = subgroup_size (8 for Xe-HPC, 16 for Xe-HPG)

C accumulator (M × N):
    Each work item (lane t) owns one column of C.
    Value v (0..rc-1) selects the row.

        col = t,  row = v
        offset = row + col * M = v + t * M

A operand (M × K):
    A is broadcast across the subgroup — all lanes see the same A data.
    Thread dimension has stride 0.

        offset = m + k * M    (col-major in M × K)

B operand (N × K, CuTe convention):
    Each lane t owns one column of B (one N-position).
    Value v selects the K-position.

        n = t,  k = v
        offset = n + k * N = t + v * N

References
----------

- Intel Graphics Compiler (IGC) VISA specification: DPAS.md
  https://github.com/intel/intel-graphics-compiler/blob/master/documentation/visa/instructions/DPAS.md
- Khronos OpenCL extension: cl_intel_subgroup_matrix_multiply_accumulate
- Intel XeTLA (Xe Templates for Linear Algebra):
  https://intel.github.io/xetla/
- CUTLASS experimental Xe backend (CuTe atom definitions for DPAS)

.. note:: Community feedback welcome

   These atom definitions were derived from public Intel ISA documentation
   and the CUTLASS experimental Xe backend.  If you find an incorrect
   mapping, please open an issue at
   https://github.com/facebookresearch/tensor-layouts/issues — both bug
   reports and confirmations that layouts match real hardware are valuable.

Usage::

    from tensor_layouts.atoms_xe import XeHPC_8x8x8_F32F16F16_DPAS
    print(XeHPC_8x8x8_F32F16F16_DPAS.c_layout)
"""

from .atoms import MMAAtom, CopyAtom
from .layouts import Layout, as_tuple
from .atoms_xe_common import (
    sizeof_bits, wi_interleave, xe_interleaved_layout, SG_SIZE_XE,
)


# =============================================================================
# Helper: construct CuTe layouts from DPAS structural parameters
# =============================================================================

def _dpas_c_layout(m: int, n: int) -> Layout:
    """Build the (T_n, V_m) -> col-major(M, N) accumulator layout.

    Lane t owns column t, value v selects row v.
    offset = v + t * M
    """
    return Layout(
        (n, m),
        (m, 1),
    )


def _dpas_a_layout(m: int, k: int, n: int) -> Layout:
    """Build the (T_n, V_{m*k}) -> col-major(M, K) input layout for A.

    A is broadcast across the subgroup (stride 0 on thread dimension).
    All lanes see the same M × K tile.
    """
    return Layout(
        (n, (m, k)),
        (0, (1, m)),
    )


def _dpas_b_layout(n: int, k: int) -> Layout:
    """Build the (T_n, V_k) -> col-major(N, K) input layout for B.

    Lane t owns column t (N dimension), value v selects K position.
    offset = t + v * N
    """
    return Layout(
        (n, k),
        (1, n),
    )


def make_dpas_atom(
    name: str,
    inst: str,
    m: int,
    n: int,
    k: int,
) -> MMAAtom:
    """Create an Intel Xe DPAS atom.

    Args:
        m: repeat_count (rows of output, typically 8)
        n: exec_size = subgroup_size (8 for Xe-HPC, 16 for Xe-HPG)
        k: systolic_depth (always 8 on current Xe)
    """
    c_layout = _dpas_c_layout(m, n)
    a_layout = _dpas_a_layout(m, k, n)
    b_layout = _dpas_b_layout(n, k)

    return MMAAtom(
        name=name,
        ptx=inst,
        shape_mnk=(m, n, k),
        thr_id=None,   # identity: lane_id = thread_idx % subgroup_size
        a_layout=a_layout,
        b_layout=b_layout,
        c_layout=c_layout,
    )


# =============================================================================
# Xe-HPC (Ponte Vecchio / Data Center Max) — subgroup_size = 8
# DPAS shape: rc × 8 × sd = 8 × 8 × 8
# =============================================================================

# --- FP16 input, FP32 accumulator ---
XeHPC_8x8x8_F32F16F16_DPAS = make_dpas_atom(
    name="XeHPC_8x8x8_F32F16F16_DPAS",
    inst="dpas.8x8 (exec_size=8, FP16)",
    m=8, n=8, k=8,
)

# --- BF16 input, FP32 accumulator ---
XeHPC_8x8x8_F32BF16BF16_DPAS = make_dpas_atom(
    name="XeHPC_8x8x8_F32BF16BF16_DPAS",
    inst="dpas.8x8 (exec_size=8, BF16)",
    m=8, n=8, k=8,
)

# --- TF32 input, FP32 accumulator ---
XeHPC_8x8x8_F32TF32TF32_DPAS = make_dpas_atom(
    name="XeHPC_8x8x8_F32TF32TF32_DPAS",
    inst="dpas.8x8 (exec_size=8, TF32)",
    m=8, n=8, k=8,
)

# --- INT8 input, INT32 accumulator ---
XeHPC_8x8x8_I32I8I8_DPAS = make_dpas_atom(
    name="XeHPC_8x8x8_I32I8I8_DPAS",
    inst="dpas.8x8 (exec_size=8, INT8)",
    m=8, n=8, k=8,
)


# =============================================================================
# Xe-HPG (Arc / DG2) — subgroup_size = 16
# DPAS shape: rc × 16 × sd = 8 × 16 × 8
# =============================================================================

# --- FP16 input, FP32 accumulator ---
XeHPG_8x16x8_F32F16F16_DPAS = make_dpas_atom(
    name="XeHPG_8x16x8_F32F16F16_DPAS",
    inst="dpas.8x8 (exec_size=16, FP16)",
    m=8, n=16, k=8,
)

# --- BF16 input, FP32 accumulator ---
XeHPG_8x16x8_F32BF16BF16_DPAS = make_dpas_atom(
    name="XeHPG_8x16x8_F32BF16BF16_DPAS",
    inst="dpas.8x8 (exec_size=16, BF16)",
    m=8, n=16, k=8,
)

# --- INT8 input, INT32 accumulator ---
XeHPG_8x16x8_I32I8I8_DPAS = make_dpas_atom(
    name="XeHPG_8x16x8_I32I8I8_DPAS",
    inst="dpas.8x8 (exec_size=16, INT8)",
    m=8, n=16, k=8,
)


# =============================================================================
# Convenience lists
# =============================================================================

MMA_ATOMS_XeHPC = [
    XeHPC_8x8x8_F32F16F16_DPAS,
    XeHPC_8x8x8_F32BF16BF16_DPAS,
    XeHPC_8x8x8_F32TF32TF32_DPAS,
    XeHPC_8x8x8_I32I8I8_DPAS,
]

MMA_ATOMS_XeHPG = [
    XeHPG_8x16x8_F32F16F16_DPAS,
    XeHPG_8x16x8_F32BF16BF16_DPAS,
    XeHPG_8x16x8_I32I8I8_DPAS,
]


# =============================================================================
# =============================================================================
# Faithful sycl-tla (CUTLASS-Xe) atoms — translated from the actual traits in
# include/cute/atom/mma_traits_xe.hpp and copy_traits_xe{,_2d}.hpp.
#
# These differ from the pedagogical DPAS atoms above: the real Xe subgroup is
# **16 lanes** (N is fixed at 16), K = 256 / max(bits(A), bits(B)), and the A/B
# operands are VNNI / work-item interleaved via detail::wi_interleave.
# =============================================================================
# =============================================================================

# -----------------------------------------------------------------------------
# XE_DPAS_TT<M, TD, TA, TB, TC>  —  mma_traits_xe.hpp:66
# -----------------------------------------------------------------------------

def make_xe_dpas_atom(name, td, ta, tb, tc, M=8):
    """Create a faithful XE_DPAS_TT atom (subgroup = 16 lanes).

    Args:
        td, ta, tb, tc: dtype names (keys of atoms_xe_common.SIZEOF_BITS) for
                        D (accum out), A, B, C (accum in).
        M: repeat_count (rows of D), 1..8.  N is always 16.
    """
    bA, bB = sizeof_bits(ta), sizeof_bits(tb)
    K = 256 // max(bA, bB)
    BV = 32 // bB
    a_layout = wi_interleave(bA, Layout((K, M), (M, 1)), SG_SIZE_XE)
    b_layout = wi_interleave(bB, Layout((BV, 16, K // BV), (16, 1, 16 * BV)),
                             SG_SIZE_XE)
    c_layout = Layout((16, M), (M, 1))
    return MMAAtom(
        name=name,
        ptx=f"dpas.{K // BV}x{M} (exec=16, {ta}x{tb}->{td})",
        shape_mnk=(M, 16, K),
        thr_id=Layout(SG_SIZE_XE),
        a_layout=a_layout, b_layout=b_layout, c_layout=c_layout,
    )


# (D, A, B, C) tuples always compiled in — mma_xe.hpp:230-252.  N=16, M=8.
_XE_DPAS_TYPES = [
    ("f32", "tf32", "tf32", "f32"),
    ("f32", "bf16", "bf16", "f32"),
    ("bf16", "bf16", "bf16", "f32"),
    ("f32", "bf16", "bf16", "bf16"),
    ("bf16", "bf16", "bf16", "bf16"),
    ("f32", "f16", "f16", "f32"),
    ("f32", "f16", "f16", "f16"),
    ("f16", "f16", "f16", "f32"),
    ("f16", "f16", "f16", "f16"),
    ("u32", "u8", "u8", "u32"),
    ("s32", "u8", "u8", "s32"),
    ("s32", "u8", "s8", "s32"),
    ("s32", "s8", "u8", "s32"),
    ("s32", "s8", "s8", "s32"),
    ("u32", "u4", "u4", "u32"),
    ("s32", "u4", "u4", "s32"),
    ("s32", "u4", "s4", "s32"),
    ("s32", "s4", "u4", "s32"),
    ("s32", "s4", "s4", "s32"),
]


def _dpas_name(td, ta, tb, tc, M):
    up = lambda s: s.upper()  # noqa: E731
    K = 256 // max(sizeof_bits(ta), sizeof_bits(tb))
    return f"XE_DPAS_{M}x16x{K}_{up(td)}{up(ta)}{up(tb)}{up(tc)}"


MMA_ATOMS_XE = [
    make_xe_dpas_atom(_dpas_name(td, ta, tb, tc, 8), td, ta, tb, tc, M=8)
    for (td, ta, tb, tc) in _XE_DPAS_TYPES
]

# Bind each as a module-level name (XE_DPAS_8x16x16_F32BF16BF16F32, ...).
for _atom in MMA_ATOMS_XE:
    globals()[_atom.name] = _atom


# =============================================================================
# XE 1D copy atoms — copy_traits_xe.hpp (bit-coordinate layouts)
# =============================================================================

def _xe_1d_atomic(s="f32", d=None):
    d = d or s
    return CopyAtom(
        name=f"XE_ATOMIC_{s.upper()}",
        ptx="atomic_ref add (global)",
        thr_id=Layout(1),
        src_layout_bits=Layout((1, sizeof_bits(s))),
        dst_layout_bits=Layout((1, sizeof_bits(d))))


def _xe_1d_slm(op, s="bf16", d=None):
    """XE_1D_LDSM (SLM->reg) / XE_1D_STSM (reg->SLM): single-thread, all bits."""
    d = d or s
    return CopyAtom(
        name=f"XE_1D_{op}_{s.upper()}",
        ptx=f"lsc_{'load' if op == 'LDSM' else 'store'}.slm",
        thr_id=Layout(1),
        src_layout_bits=Layout((1, sizeof_bits(d))),
        dst_layout_bits=Layout((1, sizeof_bits(d))))


def make_xe_1d_load_global(s="bf16", d=None):
    """XE_1D_LOAD_GLOBAL — subgroup (16-lane) global load. copy_traits_xe.hpp:68."""
    d = d or s
    bs, bd = sizeof_bits(s), sizeof_bits(d)
    return CopyAtom(
        name=f"XE_1D_LOAD_GLOBAL_{s.upper()}",
        ptx="lsc load (global, subgroup)",
        thr_id=Layout(SG_SIZE_XE),
        src_layout_bits=Layout((SG_SIZE_XE, bs), (0, 1)),
        dst_layout_bits=Layout((SG_SIZE_XE, (bd // bs, bs)),
                               (bs, (bs * SG_SIZE_XE, 1))))


def make_xe_1d_store_global(s="bf16", d=None):
    """XE_1D_STORE_GLOBAL — subgroup (16-lane) global store. copy_traits_xe.hpp:93."""
    d = d or s
    bs, bd = sizeof_bits(s), sizeof_bits(d)
    return CopyAtom(
        name=f"XE_1D_STORE_GLOBAL_{d.upper()}",
        ptx="lsc store (global, subgroup)",
        thr_id=Layout(SG_SIZE_XE),
        src_layout_bits=Layout((SG_SIZE_XE, (bs // bd, bd)),
                               (bd, (bd * SG_SIZE_XE, 1))),
        dst_layout_bits=Layout((SG_SIZE_XE, bd), (0, 1)))


COPY_ATOMS_XE_1D = [
    _xe_1d_atomic("f32"),
    _xe_1d_slm("LDSM", "bf16"),
    _xe_1d_slm("STSM", "bf16"),
    make_xe_1d_load_global("bf16"),
    make_xe_1d_load_global("f32"),
    make_xe_1d_store_global("bf16"),
]

for _atom in COPY_ATOMS_XE_1D:
    globals()[_atom.name] = _atom


# =============================================================================
# XE 2D block copy atoms — copy_traits_xe_2d.hpp
# The (thr, val) -> (x-bit, y) layouts are built by XeInterleavedLayout.
# =============================================================================

def _replace_mode0_broadcast(layout, threads=SG_SIZE_XE):
    """replace<0>(L, Layout<Shape<SGSize>, Stride<_0>>) — for 2D load/store src."""
    shape = (threads,) + tuple(as_tuple(layout.shape))[1:]
    stride = (0,) + tuple(as_tuple(layout.stride))[1:]
    return Layout(shape, stride)


def make_xe_load_2d(bits, H, W, block_w=None, val="bf16"):
    """XE_LOAD_2D<CopyBits,H,W,BlockWidth> — 2D block load .nn. traits :483."""
    block_w = block_w or W
    vb = sizeof_bits(val)
    dst = xe_interleaved_layout(
        Layout((block_w, H, W // block_w), (1, W, block_w)), bits, vb)
    src = _replace_mode0_broadcast(dst)
    return CopyAtom(
        name=f"XE_LOAD_2D_{bits}b_{H}x{W}",
        ptx=f"load 2d .nn {H}x{W} ({bits}b)",
        thr_id=Layout(SG_SIZE_XE),
        src_layout_bits=src, dst_layout_bits=dst)


def make_xe_load_2d_transpose(bits, H, W, val="f32"):
    """XE_LOAD_2D_TRANSPOSE<CopyBits,H,W> — transposing load .tn. traits :523."""
    vb = sizeof_bits(val)
    dst = xe_interleaved_layout(Layout((H, W), (W, 1)), bits, vb)
    src = _replace_mode0_broadcast(dst)
    return CopyAtom(
        name=f"XE_LOAD_2D_TRANSPOSE_{bits}b_{H}x{W}",
        ptx=f"load 2d .tn (transpose) {H}x{W} ({bits}b)",
        thr_id=Layout(SG_SIZE_XE),
        src_layout_bits=src, dst_layout_bits=dst)


def make_xe_store_2d(bits, H, W, val="f32"):
    """XE_STORE_2D<CopyBits,H,W> — 2D block store .nn. traits :542."""
    vb = sizeof_bits(val)
    src = xe_interleaved_layout(Layout((W, H)), bits, vb)
    dst = _replace_mode0_broadcast(src)
    return CopyAtom(
        name=f"XE_STORE_2D_{bits}b_{H}x{W}",
        ptx=f"store 2d .nn {H}x{W} ({bits}b)",
        thr_id=Layout(SG_SIZE_XE),
        src_layout_bits=src, dst_layout_bits=dst)


def make_xe_prefetch_2d(bits, H, W, val="bf16"):
    """XE_PREFETCH_2D<CopyBits,H,W> — 2D block prefetch to %null. traits :591."""
    vb = sizeof_bits(val)
    dst = xe_interleaved_layout(Layout((W, H)), bits, vb)
    return CopyAtom(
        name=f"XE_PREFETCH_2D_{bits}b_{H}x{W}",
        ptx=f"prefetch 2d {H}x{W} ({bits}b)",
        thr_id=Layout(SG_SIZE_XE),
        src_layout_bits=dst, dst_layout_bits=dst)


COPY_ATOMS_XE_2D = [
    make_xe_load_2d(16, 16, 16, val="bf16"),
    make_xe_load_2d(32, 8, 16, val="f32"),
    make_xe_load_2d_transpose(32, 16, 8, val="f32"),
    make_xe_store_2d(32, 8, 16, val="f32"),
    make_xe_prefetch_2d(16, 16, 16, val="bf16"),
]

for _atom in COPY_ATOMS_XE_2D:
    globals()[_atom.name] = _atom


COPY_ATOMS_XE = COPY_ATOMS_XE_1D + COPY_ATOMS_XE_2D
