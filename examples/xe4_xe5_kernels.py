# MIT License — see LICENSE file for details.

"""Emulated Xe4 / Xe5 kernel configurations, mirroring the sycl-tla examples.

For each real example under ``examples/xe4/`` in the sycl-tla (CUTLASS-Xe) repo,
this module encodes the *same* CTA tile shape, dtypes and atom family, then uses
the pure-Python atom emulations (`tensor_layouts.atoms_xe4` / `atoms_xe5`) to
show how the CTA tile decomposes into **MMA-atom tiles** and how the operands
stage through the **SLM core-matrix layout** — all on the CPU, no GPU required.

Sources (sycl-tla ``examples/xe4/``):

- ``gemm/gemm.cpp``            — CtaTileShape 256x256x128, fp16, cluster 2x1x1
- ``grouped_gemm/...``         — TileShape 256x256x128
- ``gemm/gemm_blockscaled.cpp``— CtaTileShape 128x256x128, fp8 + MX scales
- ``fmha4/xe4_fmha_fwd.cpp``   — TileShape <64,128,128,128> = <Qblk, Vdim, KVblk, QKdim>,
                                 two GEMMs (Q.K^T -> S -> softmax -> P.V -> O)

There is no ``examples/xe5/`` in the tree yet, so the Xe5 entry projects the
xe4 GEMM template onto the Xe5 **dual-group** AMMA atom (ThrID = 2).

Run ``python examples/xe4_xe5_kernels.py`` to print every decomposition report.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tensor_layouts import size
from tensor_layouts.analysis import is_bijective
from tensor_layouts.atoms_xe_common import sizeof_bits, make_slm_layout_elem
from tensor_layouts.atoms_xe4 import make_xe4_amma_atom, make_xe4_ldstm
from tensor_layouts.atoms_xe5 import make_xe5_amma_atom


def _grid(tile, atom):
    """(#atoms, remainder) per dimension for tiling ``tile`` by ``atom``."""
    return [(t // a, t % a) for t, a in zip(tile, atom)]


def _fmt(layout):
    return f"{layout}  (size={size(layout)}, bijective={is_bijective(layout)})"


# =============================================================================
# GEMM-shaped configs
# =============================================================================

@dataclass
class GemmConfig:
    name: str
    source: str
    tile_mnk: tuple            # CTA tile (M, N, K)
    atom_mnk: tuple            # MMA atom (M, N, K)
    a_dtype: str
    b_dtype: str
    dual: bool = False         # Xe5 dual-group?
    block_scaled: bool = False

    def atom(self):
        d, a, b, c = "f32", self.a_dtype, self.b_dtype, "f32"
        M, N, K = self.atom_mnk
        if self.dual:
            return make_xe5_amma_atom("XE5_AMMA_AB_CLUSTER_DUAL",
                                      "amma.atmm.btmm (cluster, dual)",
                                      d, a, b, c, M, N, K, dual=True,
                                      block_scaled=self.block_scaled)
        return make_xe4_amma_atom(d, a, b, c, M, N, K,
                                  tracking="AB", block_scaled=self.block_scaled)

    def describe(self):
        M, N, K = self.tile_mnk
        aM, aN, aK = self.atom_mnk
        atom = self.atom()
        g = _grid(self.tile_mnk, self.atom_mnk)
        n_atoms = g[0][0] * g[1][0] * g[2][0]
        slm_a = make_slm_layout_elem(sizeof_bits(self.a_dtype), M, K)
        slm_b = make_slm_layout_elem(sizeof_bits(self.b_dtype), N, K)
        print(f"\n{'='*72}\n{self.name}\n  (from {self.source})\n{'='*72}")
        print(f"  CTA tile  M×N×K : {M} × {N} × {K}   dtypes A={self.a_dtype} "
              f"B={self.b_dtype}{'  [dual-group]' if self.dual else ''}"
              f"{'  [block-scaled MX]' if self.block_scaled else ''}")
        print(f"  MMA atom  M×N×K : {aM} × {aN} × {aK}   ({atom.name})")
        print(f"  atom grid       : {g[0][0]}×{g[1][0]}×{g[2][0]} "
              f"= {n_atoms} atom invocations / CTA "
              f"({g[0][0]*g[1][0]} accumulator tiles × {g[2][0]} K-steps)")
        for dim, (cnt, rem) in zip("MNK", g):
            assert rem == 0, f"{self.name}: {dim} tile {dict(M=M,N=N,K=K)[dim]} not divisible by atom {(aM,aN,aK)['MNK'.index(dim)]}"
        print(f"  SLM A tile      : {_fmt(slm_a)}")
        print(f"  SLM B tile      : {_fmt(slm_b)}")
        assert is_bijective(slm_a) and is_bijective(slm_b)
        assert size(slm_a) == M * K and size(slm_b) == N * K
        return dict(name=self.name, n_atoms=n_atoms, slm_a=slm_a, slm_b=slm_b)


# =============================================================================
# FMHA4 config — two GEMMs + softmax bridge
# =============================================================================

@dataclass
class Fmha4Config:
    name: str
    source: str
    # TileShape = <BLK_M_Q, V_head_dim, KV_blk, QK_head_dim>
    tile: tuple
    qkv_dtype: str = "f16"
    p_dtype: str = "f16"
    atom_mnk: tuple = (64, 128, 32)   # per-GEMM AMMA atom

    @property
    def qk_mnk(self):   # S = Q·Kᵀ : select<0,2,3> = <Qblk, KVblk, QKdim>
        return (self.tile[0], self.tile[2], self.tile[3])

    @property
    def pv_mnk(self):   # O = P·V : select<0,1,2> = <Qblk, Vdim, KVblk>
        return (self.tile[0], self.tile[1], self.tile[2])

    def describe(self):
        d = "f32"
        qk = self.qk_mnk
        pv = self.pv_mnk
        atom_qk = make_xe4_amma_atom(d, self.qkv_dtype, self.qkv_dtype, d,
                                     *_fit_atom(qk, self.atom_mnk), tracking="AB")
        atom_pv = make_xe4_amma_atom(d, self.p_dtype, self.qkv_dtype, d,
                                     *_fit_atom(pv, self.atom_mnk), tracking="AB")
        print(f"\n{'='*72}\n{self.name}\n  (from {self.source})\n{'='*72}")
        print(f"  TileShape <Qblk, Vdim, KVblk, QKdim> = {self.tile}")
        print(f"  dtypes: Q,K,V={self.qkv_dtype}  P={self.p_dtype}  accum=f32")
        print(f"\n  GEMM-1  S = Q · Kᵀ   (M×N×K = {qk[0]}×{qk[1]}×{qk[2]})")
        print(f"    Q -> A operand,  K -> B operand   via ADMA_Q / ADMA_K loads")
        _report_gemm(qk, atom_qk, self.qkv_dtype, self.qkv_dtype)
        print(f"\n  softmax(S) -> P   (row-wise over N = {qk[1]} keys)")
        # The S->P bridge uses an LDSM warp-row copy so row-mates share a 32-lane
        # warp for in-warp fred.max reductions (Xe4-specific).
        ldsm = make_xe4_ldstm("LDSM", VS=8, s=self.p_dtype)
        print(f"    LDSM warp-row load: {ldsm.name}  ThrID={size(ldsm.thr_id)} "
              f"(row-mates co-resident for fred.max)")
        print(f"    S tile has {qk[0]} query rows; each reduced across {qk[1]} keys")
        print(f"\n  GEMM-2  O = P · V   (M×N×K = {pv[0]}×{pv[1]}×{pv[2]})")
        print(f"    P -> A operand,  V -> B operand   via ADMA_V load")
        _report_gemm(pv, atom_pv, self.p_dtype, self.qkv_dtype)
        return dict(name=self.name, qk=qk, pv=pv)


def _fit_atom(tile_mnk, atom_mnk):
    """Clamp each atom dim to divide the tile (keeps the emulation exact)."""
    from math import gcd
    return tuple(gcd(t, a) for t, a in zip(tile_mnk, atom_mnk))


def _report_gemm(tile, atom, a_dtype, b_dtype):
    M, N, K = tile
    g = _grid(tile, atom.shape_mnk)
    n_atoms = g[0][0] * g[1][0] * g[2][0]
    slm_a = make_slm_layout_elem(sizeof_bits(a_dtype), M, K)
    slm_b = make_slm_layout_elem(sizeof_bits(b_dtype), N, K)
    print(f"      atom {atom.shape_mnk} -> grid {g[0][0]}×{g[1][0]}×{g[2][0]} "
          f"= {n_atoms} invocations")
    print(f"      SLM A {_fmt(slm_a)}")
    print(f"      SLM B {_fmt(slm_b)}")
    assert is_bijective(slm_a) and is_bijective(slm_b)


# =============================================================================
# The example configs (mirroring sycl-tla examples/xe4/*)
# =============================================================================

GEMM_CONFIGS = [
    GemmConfig("XE4 GEMM (fp16, RowMajor)", "examples/xe4/gemm/gemm.cpp",
               tile_mnk=(256, 256, 128), atom_mnk=(128, 128, 32),
               a_dtype="f16", b_dtype="f16"),
    GemmConfig("XE4 Grouped GEMM", "examples/xe4/grouped_gemm/grouped_gemm.cpp",
               tile_mnk=(256, 256, 128), atom_mnk=(128, 128, 32),
               a_dtype="f16", b_dtype="f16"),
    GemmConfig("XE4 Block-scaled GEMM (fp8 + MX)", "examples/xe4/gemm/gemm_blockscaled.cpp",
               tile_mnk=(128, 256, 128), atom_mnk=(128, 128, 32),
               a_dtype="e4m3", b_dtype="e4m3", block_scaled=True),
    GemmConfig("XE5 GEMM (dual-group, projected from xe4 template)",
               "examples/xe4/gemm/gemm.cpp + XE5 dual-group AMMA",
               tile_mnk=(256, 256, 128), atom_mnk=(128, 128, 32),
               a_dtype="bf16", b_dtype="bf16", dual=True),
]

FMHA_CONFIGS = [
    Fmha4Config("XE4 FMHA4 forward (fp16)", "examples/xe4/fmha4/xe4_fmha_fwd.cpp",
                tile=(64, 128, 128, 128), qkv_dtype="f16", p_dtype="f16"),
]


def main():
    print("Emulated Xe4 / Xe5 kernel layout decompositions "
          "(mirroring sycl-tla examples/xe4/*)")
    for cfg in GEMM_CONFIGS:
        cfg.describe()
    for cfg in FMHA_CONFIGS:
        cfg.describe()
    print("\nAll decompositions tile exactly and all SLM tiles are bijective.")


if __name__ == "__main__":
    main()
