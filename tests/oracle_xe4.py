# MIT License — see LICENSE file for details.

"""Structural tests for Intel Xe4 CuTe atoms (TMM, AMMA, ADMA, LDSM, EU_COPY).

Validates the emulated (thread, value) -> element/bit mappings against the
algebraic invariants every well-formed atom must satisfy — no GPU required.
"""

import pytest

from tensor_layouts import size, rank
from tensor_layouts.analysis import is_bijective
from tensor_layouts.atoms_xe4 import (
    MMA_ATOMS_XE4, COPY_ATOMS_XE4, MMA_ATOMS_XE4_TMM, MMA_ATOMS_XE4_AMMA,
    slm_a_layout,
)
from tensor_layouts.atoms_xe_common import make_slm_layout_elem


def _thr(layout):
    return size(layout.shape[0]) if isinstance(layout.shape, tuple) else size(layout.shape)


def _val(layout):
    return size(layout.shape[1]) if isinstance(layout.shape, tuple) else 1


def _c_offsets(atom):
    c = atom.c_layout
    return [c(t, v) for t in range(_thr(c)) for v in range(_val(c))]


@pytest.mark.parametrize("atom", MMA_ATOMS_XE4, ids=lambda a: a.name)
class TestXe4MMAStructural:
    def test_shape_positive(self, atom):
        assert all(d > 0 for d in atom.shape_mnk)

    def test_c_covers_mn(self, atom):
        m, n, k = atom.shape_mnk
        offs = _c_offsets(atom)
        assert set(offs) == set(range(m * n))
        assert len(offs) == m * n

    def test_thrid_matches_c_threads(self, atom):
        assert size(atom.thr_id) == _thr(atom.c_layout)

    def test_a_covers_mk(self, atom):
        m, n, k = atom.shape_mnk
        a = atom.a_layout
        offs = {a(t, v) for t in range(_thr(a)) for v in range(_val(a))}
        assert offs == set(range(m * k))

    def test_b_covers_nk(self, atom):
        m, n, k = atom.shape_mnk
        b = atom.b_layout
        offs = {b(t, v) for t in range(_thr(b)) for v in range(_val(b))}
        # TMM B is a simplified (N,K) view; it still must cover N*K.
        assert offs == set(range(n * k))


class TestXe4TMM:
    """TMM is register-resident with M fixed at 32 and 32-lane subgroup."""

    @pytest.mark.parametrize("atom", MMA_ATOMS_XE4_TMM, ids=lambda a: a.name)
    def test_m_is_32(self, atom):
        assert atom.shape_mnk[0] == 32

    @pytest.mark.parametrize("atom", MMA_ATOMS_XE4_TMM, ids=lambda a: a.name)
    def test_subgroup_32(self, atom):
        assert size(atom.thr_id) == 32


class TestXe4AMMADescriptor:
    """AMMA operands are single-thread SLM descriptors (ThrID = 1)."""

    @pytest.mark.parametrize("atom", MMA_ATOMS_XE4_AMMA, ids=lambda a: a.name)
    def test_single_thread(self, atom):
        assert size(atom.thr_id) == 1

    @pytest.mark.parametrize("atom", MMA_ATOMS_XE4_AMMA, ids=lambda a: a.name)
    def test_all_tracking_share_layout(self, atom):
        base = MMA_ATOMS_XE4_AMMA[0]
        assert atom.a_layout == base.a_layout
        assert atom.c_layout == base.c_layout


class TestXe4SLMCoreMatrix:
    """The physical SLM tile the AMMA descriptors reference is bijective."""

    @pytest.mark.parametrize("bits", [32, 16, 8])
    def test_single_core_matrix_is_1024_bytes(self, bits):
        L = make_slm_layout_elem(bits, 32, (1024 // (32 * (bits // 8 or 1))))
        # single core matrix = 1024 bytes = 1024 / (bits/8) elements
        assert size(L) == 1024 // (bits // 8 or 1)
        assert is_bijective(L)

    @pytest.mark.parametrize("M,K", [(128, 32), (256, 64), (64, 32)])
    def test_slm_operand_bijective(self, M, K):
        L = slm_a_layout("bf16", M, K)
        assert size(L) == M * K
        assert is_bijective(L)


@pytest.mark.parametrize("atom", COPY_ATOMS_XE4, ids=lambda a: a.name)
class TestXe4CopyStructural:
    def test_layouts_callable(self, atom):
        s, d = atom.src_layout_bits, atom.dst_layout_bits
        assert s(0, 0) == 0 or s(0, 0) >= 0
        assert d(0, 0) >= 0

    def test_thrid_positive(self, atom):
        assert size(atom.thr_id) >= 1

    def test_rank_two_layouts(self, atom):
        assert rank(atom.src_layout_bits) == 2
        assert rank(atom.dst_layout_bits) == 2


if __name__ == "__main__":
    import subprocess
    import sys
    raise SystemExit(subprocess.call([sys.executable, "-m", "pytest", __file__, "-v"]))
