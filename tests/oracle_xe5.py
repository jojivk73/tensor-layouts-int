# MIT License — see LICENSE file for details.

"""Structural tests for Intel Xe5 CuTe atoms (single/dual-group AMMA, ADMA).

Validates the emulated (thread, value) -> element/bit mappings against the
algebraic invariants every well-formed atom must satisfy — no GPU required.
"""

import pytest

from tensor_layouts import size, rank
from tensor_layouts.atoms_xe5 import (
    MMA_ATOMS_XE5, COPY_ATOMS_XE5, MMA_ATOMS_XE5_SG, MMA_ATOMS_XE5_DUAL,
)


def _thr(layout):
    return size(layout.shape[0]) if isinstance(layout.shape, tuple) else size(layout.shape)


def _val(layout):
    return size(layout.shape[1]) if isinstance(layout.shape, tuple) else 1


def _covers(layout, total):
    offs = [layout(t, v) for t in range(_thr(layout)) for v in range(_val(layout))]
    return set(offs) == set(range(total)) and len(offs) == total


@pytest.mark.parametrize("atom", MMA_ATOMS_XE5, ids=lambda a: a.name)
class TestXe5MMAStructural:
    def test_c_covers_mn(self, atom):
        m, n, k = atom.shape_mnk
        assert _covers(atom.c_layout, m * n)

    def test_a_covers_mk(self, atom):
        m, n, k = atom.shape_mnk
        assert _covers(atom.a_layout, m * k)

    def test_b_covers_nk(self, atom):
        m, n, k = atom.shape_mnk
        assert _covers(atom.b_layout, n * k)

    def test_thrid_matches_c(self, atom):
        assert size(atom.thr_id) == _thr(atom.c_layout)


class TestXe5SingleGroup:
    @pytest.mark.parametrize("atom", MMA_ATOMS_XE5_SG, ids=lambda a: a.name)
    def test_single_thread(self, atom):
        assert size(atom.thr_id) == 1


class TestXe5DualGroup:
    """Dual-group atoms split the work across a lane pair (ThrID = 2)."""

    @pytest.mark.parametrize("atom", MMA_ATOMS_XE5_DUAL, ids=lambda a: a.name)
    def test_thrid_is_2(self, atom):
        assert size(atom.thr_id) == 2

    @pytest.mark.parametrize("atom", MMA_ATOMS_XE5_DUAL, ids=lambda a: a.name)
    def test_each_lane_owns_half_of_m(self, atom):
        m, n, k = atom.shape_mnk
        c = atom.c_layout
        # Lane 0 owns rows [0, M/2); lane 1 owns rows [M/2, M).
        rows0 = {c(0, v) % m for v in range(_val(c))}
        rows1 = {c(1, v) % m for v in range(_val(c))}
        assert max(rows0) < m // 2
        assert min(rows1) >= m // 2


@pytest.mark.parametrize("atom", COPY_ATOMS_XE5, ids=lambda a: a.name)
class TestXe5CopyStructural:
    def test_rank_two(self, atom):
        assert rank(atom.src_layout_bits) == 2
        assert rank(atom.dst_layout_bits) == 2

    def test_thrid_positive(self, atom):
        assert size(atom.thr_id) >= 1


if __name__ == "__main__":
    import subprocess
    import sys
    raise SystemExit(subprocess.call([sys.executable, "-m", "pytest", __file__, "-v"]))
