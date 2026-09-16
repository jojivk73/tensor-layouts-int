# Intel Xe / Xe4 / Xe5 atom emulation

Pure-Python, CPU-executable emulations of the Intel Xe GPU **MMA and Copy
atoms** defined in the sycl-tla / CUTLASS-Xe headers
(`include/cute/{arch,atom}/*xe*`).  Each atom is a set of
`(thread, value) -> element` maps carried on the same `MMAAtom` / `CopyAtom`
dataclasses used for the NVIDIA and AMD backends, so all of the existing
analysis (`atom_summary`) and visualization (`draw_mma_layout`,
`draw_copy_atom`, `draw_layout`) tooling works on them unchanged.

See the executable walkthrough in [`examples/xe_atoms.ipynb`](../examples/xe_atoms.ipynb).

## Modules

| Module | Contents |
|--------|----------|
| `tensor_layouts.atoms_xe_common` | Shared helpers: `sizeof_bits`, `wi_interleave` (DPAS VNNI interleave), `xe_interleaved_layout` (2D-copy TV splitter), `make_ordered_layout`, and the SLM core-matrix builders (`make_single_core_matrix_slm_layout`, `make_slm_layout_elem`). |
| `tensor_layouts.atoms_xe`  | Legacy Xe-HPC/HPG. Pedagogical `XeHPC_*/XeHPG_*` DPAS atoms **plus** the sycl-tla-faithful `XE_DPAS_TT` MMA atoms (`MMA_ATOMS_XE`) and 1D/2D block copies (`COPY_ATOMS_XE`). Subgroup = 16. |
| `tensor_layouts.atoms_xe4` | Xe4 (subgroup 32): `XE4_TMM` (register MMA), `XE4_AMMA` (SLM-descriptor MMA, all wired barrier-tracking + block-scaled variants), ADMA / LDSM / EU_COPY / matrix copies. |
| `tensor_layouts.atoms_xe5` | Xe5: single-group + **dual-group** (`ThrID=2`) AMMA, LARGE core-matrix variants, block-scaled MX (incl. fp6 `e3m2`/`e2m3`), and ADMA dual / linear-GCS copies. |

## Fidelity notes

The emulator reproduces the **trait `(thread, value)` layouts** verbatim from
the C++ `MMA_Traits<>` / `Copy_Traits<>` specializations, and the derived
transforms (`wi_interleave`, `XeInterleavedLayout`, the SLM core matrix) as
executable layout-algebra.  Two things are intentionally simplified, matching
the "start with what's tractable" scope:

- **AMMA / ADMA descriptor placeholders.** The traits carry a single-thread
  `(1, M, K)` / `(1, NumBits)` placeholder — the *physical* SLM tile lives in
  the matrix descriptor. We keep the placeholder (so the atom matches the
  trait) and additionally expose the real SLM layout via `make_slm_layout_elem`
  / `slm_a_layout` / `slm_b_layout`. Barrier tracking is a *type* parameter, so
  all tracking variants share one A/B/C layout (only `.ptx` differs).
- **`XE4_TMM` B operand.** Modelled as the natural `(N, K)` col-major view; its
  true register-VNNI packing (`packF` / `nRegsB` / `tmmLayoutBMaxLanes`) is not
  reproduced.

Not ported (out of scope): the 100 raw `AsyncMMA` intrinsic specializations,
the full descriptor unions, the EU-copy even/odd swizzle internals, and the
runtime selector/factory machinery — these carry no distinct CuTe TV-layout.

## Worked examples (mirroring the sycl-tla `examples/xe4/`)

`examples/xe4_xe5_kernels.py` encodes the real xe4 example configurations and
shows how each CTA tile decomposes into MMA-atom tiles staged through SLM
(`python examples/xe4_xe5_kernels.py`):

| Example | sycl-tla source | CTA tile M×N×K |
|---------|-----------------|----------------|
| XE4 GEMM (fp16, RowMajor)  | `examples/xe4/gemm/gemm.cpp`             | 256×256×128 |
| XE4 Grouped GEMM           | `examples/xe4/grouped_gemm/`             | 256×256×128 |
| XE4 Block-scaled (fp8+MX)  | `examples/xe4/gemm/gemm_blockscaled.cpp` | 128×256×128 |
| XE5 GEMM (dual-group)      | projected from the xe4 GEMM template     | 256×256×128 |
| **XE4 FMHA4 forward**      | `examples/xe4/fmha4/xe4_fmha_fwd.cpp`    | ⟨64,128,128,128⟩ |

### Per-example scripts and notebooks

`examples/xe4/` and `examples/xe5/` contain **one self-contained functional
example per tutorial example** from `examples/cute/tutorial/{xe4,xe5}/` (plus the
**FMHA4 forward** kernel, `examples/xe4/fmha4_fwd_xe4.{ipynb,py}`). Each example
ships **both** a notebook (`<name>.ipynb`) and an equivalent runnable script
(`<name>.py`) with the same inlined code — no shared runtime — and works cell by
cell / section by section
through the kernel's operations (load → MMA → reduce/softmax/dual-split →
epilogue → store) on small concrete data, printing the actual numbers and the
matching Xe layout at each step (SLM core-matrix swizzle with loss-less
round-trips, accumulator ownership, LDSM warp-row co-residence, block-scale
application, dual-group M-split, GCS zero-fill, …). See the per-directory
`README.md`. Two family notebooks — `examples/xe4_atoms.ipynb` and
`examples/xe5_atoms.ipynb` — walk every atom in each family.

The **FMHA4** entry mirrors the real forward kernel: two GEMMs
(`S = Q·Kᵀ` via `TiledMmaQK`, then softmax, then `O = P·V` via `TiledMmaPV`,
with `TileShapeQK = select<0,2,3>` and `TileShapePV = select<0,1,2>`), the
`ADMA_Q`/`ADMA_K`/`ADMA_V` operand loads, and the `XE4_LDSM` warp-row copy that
bridges S→P so row-mates share a 32-lane warp for the softmax `fred.max`
reduction.  The notebook (`examples/xe_atoms.ipynb`, §9) visualizes the GEMM
accumulator grid and the FMHA4 softmax-bridge copy atom.

## Tests

Structural invariants (coverage, bijectivity, thread counts, dual-group M/N
split) live in `tests/oracle_xe.py`, `tests/oracle_xe4.py`,
`tests/oracle_xe5.py`.
