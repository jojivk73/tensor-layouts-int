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

"""Shared helpers for the Intel Xe / Xe4 / Xe5 CuTe atom emulations.

These reproduce, in pure Python over ``tensor_layouts.Layout``, the small set
of layout-algebra helpers that the sycl-tla (CUTLASS-Xe) traits use to build
their (thread, value) layouts:

- :func:`sizeof_bits` — element bit widths, mirroring ``cute::sizeof_bits_v``.
- :func:`by_mode_coalesce` — ``coalesce(L, Step<_1, _1, ...>)`` (per-mode).
- :func:`wi_interleave` — the DPAS work-item / VNNI interleave
  (``detail::wi_interleave`` in ``atom/mma_traits_xe.hpp``).
- :func:`xe_interleaved_layout` — the 2D block-copy TV splitter
  (``XeInterleavedLayoutHelper`` in ``atom/copy_traits_xe_2d.hpp``).
- :func:`make_ordered_layout` — ``cute::make_ordered_layout`` (used by XE4_LDSM).
- SLM *core-matrix* builders (``xe4_async_gmma_slm_layout.hpp``), the physical
  SLM layout that the AMMA/ADMA matrix descriptors actually reference.

Everything here is executable on a CPU: a ``Layout`` is a callable
``L(coord...) -> offset`` and the algebra ops are the pure-Python
``tensor_layouts`` implementations.

The subgroup / SIMD width on Xe is 16 lanes (Xe-legacy) and 32 lanes (Xe4/Xe5).
"""

from __future__ import annotations

from .layouts import Layout, compose, coalesce, size, as_tuple, is_tuple

# Subgroup (SIMD) sizes.
SG_SIZE_XE = 16     # Xe-legacy (Xe-HPC/Xe-HPG) and DPAS exec size
SG_SIZE_XE4 = 32    # Xe4 / Xe5 subgroup


# =============================================================================
# Element bit widths — mirrors cute::sizeof_bits_v<T>
# =============================================================================

#: Bit width of every dtype the Xe DPAS / AMMA / ADMA atoms name.
SIZEOF_BITS = {
    "f32": 32, "float": 32, "f": 32, "d": 32, "ud": 32, "s32": 32, "int32": 32,
    "tf32": 32,
    "bf16": 16, "bf": 16, "hf16": 16, "hf": 16, "f16": 16, "half": 16,
    "s16": 16, "u16": 16,
    "bf8": 8, "hf8": 8, "e4m3": 8, "e5m2": 8, "s8": 8, "u8": 8, "int8": 8,
    "e3m2": 8, "e2m3": 8,          # Xe5 MX fp6 containers (stored in 8 bits)
    "e2m1": 4, "u4": 4, "s4": 4,   # fp4 / int4
}


def sizeof_bits(dtype) -> int:
    """Return the bit width of ``dtype`` (str name or int bit count)."""
    if isinstance(dtype, int):
        return dtype
    return SIZEOF_BITS[dtype]


def ceil_div(a: int, b: int) -> int:
    return -(-a // b)


# =============================================================================
# by_mode_coalesce — coalesce(L, Step<_1, _1, ...>)
# =============================================================================
#
# CuTe's `coalesce(layout, target_profile)` with a Step of all-ones coalesces
# each top-level mode *independently* and keeps that many modes.  The package's
# `coalesce(L)` flattens fully (can merge across modes), so we coalesce each
# top-level mode on its own and re-assemble.


def _mode(layout: Layout, i: int) -> Layout:
    """The i-th top-level mode of ``layout`` as a standalone Layout."""
    shp = layout.shape[i]
    std = layout.stride[i]
    return Layout(shp, std)


def by_mode_coalesce(layout: Layout) -> Layout:
    """``coalesce(layout, Step<_1, _1, ...>)`` — coalesce each mode separately.

    Keeps the number of top-level modes fixed while collapsing redundant
    sub-modes inside each one.  Used by :func:`wi_interleave` and
    :func:`xe_interleaved_layout`, mirroring the ``Step<_1,_1>`` argument in the
    C++ traits.
    """
    if not is_tuple(layout.shape):
        return coalesce(layout)
    modes = [coalesce(_mode(layout, i)) for i in range(len(as_tuple(layout.shape)))]
    shape = tuple(m.shape for m in modes)
    stride = tuple(m.stride for m in modes)
    return Layout(shape, stride)


# =============================================================================
# wi_interleave — DPAS work-item (VNNI) interleave
#   detail::wi_interleave  (include/cute/atom/mma_traits_xe.hpp)
# =============================================================================


def wi_interleave(val_bits: int, base: Layout, sg_size: int = SG_SIZE_XE) -> Layout:
    """Reproduce ``detail::wi_interleave<ValType>(base)``.

    Splits ``base`` across ``sg_size`` work-items, interleaving sub-byte /
    packed values (VNNI) so each lane holds a contiguous register fragment::

        per_byte = ceil_div(8, val_bits)
        vals     = ceil_div(size(base), sg_size)
        tv = ((sg, (per_byte, vals/per_byte)) : (per_byte, (1, sg*per_byte)))
        return coalesce(compose(base, tv), Step<_1,_1>)

    Args:
        val_bits: element bit width of the operand type (e.g. 16 for bf16).
        base:     the pre-interleave (K, M)-style logical layout.
        sg_size:  subgroup width (16 on Xe-legacy).
    """
    per_byte = ceil_div(8, val_bits)
    vals = ceil_div(size(base), sg_size)
    tv = Layout(
        (sg_size, (per_byte, vals // per_byte)),
        (per_byte, (1, sg_size * per_byte)),
    )
    return by_mode_coalesce(compose(base, tv))


# =============================================================================
# xe_interleaved_layout — 2D block-copy TV splitter
#   XeInterleavedLayoutHelper  (include/cute/atom/copy_traits_xe_2d.hpp)
# =============================================================================


def xe_interleaved_layout(inlayout: Layout, copy_bits: int, val_bits: int,
                          threads: int = SG_SIZE_XE) -> Layout:
    """Reproduce ``XeInterleavedLayout<InLayout, CopyBits, ValBits, Threads>``.

    Splits a subgroup-level element layout into a (thread, value) layout in
    *bit* coordinates::

        VecTypeBits = max(ValBits, 8)
        Expanded = logical_product((CopyBits/VecTypeBits), InLayout)
        TV       = compose(Expanded, (Threads, size(Expanded)/Threads))
        PreResult= blocked_product((1, VecTypeBits), TV)
        Result   = coalesce(PreResult, Step<_1,_1>)
    """
    from .layouts import logical_product, blocked_product

    vec_type_bits = max(val_bits, 8)
    expanded = logical_product(Layout(copy_bits // vec_type_bits), inlayout)
    tv = compose(expanded, Layout((threads, size(expanded) // threads)))
    pre = blocked_product(Layout((1, vec_type_bits)), tv)
    return by_mode_coalesce(pre)


# =============================================================================
# make_ordered_layout — cute::make_ordered_layout(shape, order)
# =============================================================================


def _flatten_with_path(x, prefix=()):
    """Yield (path, leaf_int) for every leaf of a nested int/tuple tree."""
    if is_tuple(x):
        for i, sub in enumerate(as_tuple(x)):
            yield from _flatten_with_path(sub, prefix + (i,))
    else:
        yield prefix, x


def _rebuild(shape, path_to_val):
    """Rebuild a nested tuple congruent to ``shape`` from a {path: val} map."""
    def rec(x, prefix):
        if is_tuple(x):
            return tuple(rec(sub, prefix + (i,))
                         for i, sub in enumerate(as_tuple(x)))
        return path_to_val[prefix]
    return rec(shape, ())


def make_ordered_layout(shape, order) -> Layout:
    """Reproduce ``cute::make_ordered_layout(shape, order)``.

    Builds the compact (bijective) layout whose mode-traversal order is given
    by ``order``: the leaf with the smallest order value is innermost (stride
    1), and each subsequent leaf's stride is the product of the sizes of all
    leaves with a strictly smaller order value.  Ties in ``order`` break by the
    leaf's position in the shape tree (matching CuTe's stable behaviour).
    """
    leaves = list(_flatten_with_path(shape))            # [(path, size), ...]
    orders = dict(_flatten_with_path(order))            # {path: order_value}
    # Sort leaves by (order value, path) to get the traversal sequence.
    seq = sorted(leaves, key=lambda pv: (orders[pv[0]], pv[0]))
    strides = {}
    running = 1
    for path, sz in seq:
        strides[path] = running
        running *= sz
    stride_tree = _rebuild(shape, strides)
    return Layout(shape, stride_tree)


# =============================================================================
# SLM core-matrix layouts — xe4_async_gmma_slm_layout.hpp
#   namespace cute::xe4::slm::type1::kmajor
# =============================================================================
#
# This is the *physical* SLM layout that the AMMA / ADMA matrix descriptors
# reference (the `Layout<Shape<_1, NumBits>>` in the AMMA/ADMA traits is only a
# placeholder — the real tile shape lives here).  Byte-space constants:

SLM_CORE_MATRIX_BYTES = 1024
SLM_ROWS_PER_CM_TILE = 2
SLM_ESUB_BANKS = 8
SLM_HALF_ESUB_BANKS = 4
SLM_MMA_BANKS = 4
SLM_BYTES_PER_CM_ROW = 32
SLM_CORE_MATRIX_M = 32
SLM_ESUB_BANK_BYTES = 64


def make_single_core_matrix_slm_layout(val_bits: int) -> Layout:
    """One 1024-byte SLM core matrix, in *element* coordinates.

    ``Shape<Shape<2,4,4>, cm_k> : Stride<Stride<32/e, 256/e, 64/e>, 1>`` where
    ``e = val_bits/8`` (bytes/element) and ``cm_k = 1024 / (32 * e)``.
    """
    elem = max(val_bits // 8, 1)
    cm_k = SLM_CORE_MATRIX_BYTES // (SLM_BYTES_PER_CM_ROW * elem)
    return Layout(
        ((SLM_ROWS_PER_CM_TILE, SLM_HALF_ESUB_BANKS, SLM_MMA_BANKS), cm_k),
        ((SLM_BYTES_PER_CM_ROW // elem,                          # 32/e
          (SLM_HALF_ESUB_BANKS * SLM_ESUB_BANK_BYTES) // elem,   # 256/e
          SLM_ESUB_BANK_BYTES // elem),                          # 64/e
         1),
    )


def make_slm_layout_elem(val_bits: int, M: int, K: int) -> Layout:
    """Full M x K SLM tile in *element* coordinates (``make_slm_layout_elem``).

    Tiles the single core matrix over ``num_cm_rows = M / 32`` row blocks and
    ``num_cm_cols = K / cm_k`` column blocks::

        Shape ((2,4,4,num_cm_rows), (cm_k, num_cm_cols))
        Stride((32/e,256/e,64/e, num_cm_cols*1024/e), (1, 1024/e))
    """
    elem = max(val_bits // 8, 1)
    cm_k = SLM_CORE_MATRIX_BYTES // (SLM_BYTES_PER_CM_ROW * elem)
    num_cm_rows = M // SLM_CORE_MATRIX_M
    num_cm_cols = K // cm_k
    cm_bytes_e = SLM_CORE_MATRIX_BYTES // elem
    return Layout(
        ((SLM_ROWS_PER_CM_TILE, SLM_HALF_ESUB_BANKS, SLM_MMA_BANKS, num_cm_rows),
         (cm_k, num_cm_cols)),
        ((SLM_BYTES_PER_CM_ROW // elem,
          (SLM_HALF_ESUB_BANKS * SLM_ESUB_BANK_BYTES) // elem,
          SLM_ESUB_BANK_BYTES // elem,
          num_cm_cols * cm_bytes_e),
         (1, cm_bytes_e)),
    )
