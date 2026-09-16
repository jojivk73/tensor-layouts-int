<p align="center">
  <img src="https://raw.githubusercontent.com/facebookresearch/tensor-layouts/main/docs/images/logo.svg" alt="tensor-layouts" width="520">
</p>

[![Lint](https://github.com/facebookresearch/tensor-layouts/actions/workflows/lint.yml/badge.svg)](https://github.com/facebookresearch/tensor-layouts/actions/workflows/lint.yml)
[![Tests](https://github.com/facebookresearch/tensor-layouts/actions/workflows/tests.yml/badge.svg)](https://github.com/facebookresearch/tensor-layouts/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/tensor-layouts)](https://pypi.org/project/tensor-layouts/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/facebookresearch/tensor-layouts/blob/main/LICENSE)

A pure-Python implementation of the [NVIDIA CuTe](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/cute/00_quickstart.md) layout algebra. **No GPU required.**

Note that tensor-layouts is now maintained out of [https://github.com/jduprat/tensor-layouts.git](https://github.com/jduprat/tensor-layouts.git)


CuTe layouts describe how logical coordinates map to memory offsets on GPUs.
This library lets you construct, compose, and visualize those layouts using
plain Python — useful for understanding tensor core access patterns, debugging
swizzled shared memory, and prototyping tiled GPU kernels without compiling any
CUDA. The code in `src/tensor_layouts/layouts.py` is intentionally written to
be readable and pedagogical for people learning layout algebra.
The visualization layer is also designed to be pedagogical: for example,
hierarchical layout views can explicitly show nested row/column coordinates and
the resulting offset for each displayed cell.

## Project Status

The core implementation is already in very good shape: the layout algebra,
tensor semantics, analysis helpers, and visualization surface are correct,
usable, and mature. Most of the remaining work is in documentation, examples,
packaging, and repository polish rather than in redesigning the underlying
model.

## Installation

```bash
pip install tensor-layouts
```

For visualization support:

```bash
pip install tensor-layouts[viz]
```

## Quick Start

```python
from tensor_layouts import Layout, compose, complement, logical_divide

# A 4x8 column-major layout: offset(i,j) = i + j*4
layout = Layout((4, 8), (1, 4))
print(layout)       # (4, 8) : (1, 4)
print(layout(2, 3)) # 14

# Compose two layouts
a = Layout((4, 2), (1, 4))
b = Layout((2, 4), (4, 1))
print(compose(a, b))

# Tile a layout into 2x4 blocks
tiler = Layout((2, 4))
print(logical_divide(layout, tiler))
```

## Core Concepts

A `Layout` is a function from logical coordinates to memory offsets, defined by
`(shape, stride)`:

| Layout | Description |
|--------|-------------|
| `Layout((4, 8), (8, 1))` | 4x8 row-major |
| `Layout((4, 8), (1, 4))` | 4x8 column-major |
| `Layout(((2,4), 8), ((1,16), 2))` | Hierarchical (tiled) |

The algebra provides four key operations:

- **`compose(A, B)`** — Function composition: apply B's indexing to A's codomain
- **`complement(L)`** — The "missing half" of a layout's codomain
- **`logical_divide(L, T)`** — Factor a layout into tiles of shape T
- **`logical_product(A, B)`** — Replicate A's pattern across B's domain

Plus `Swizzle(B, M, S)` for XOR-based bank conflict avoidance patterns.

## MMA Atoms

The library includes matrix-multiply atom definitions for NVIDIA, AMD, Intel Xe,
and Intel AMX targets. The Intel mappings were derived from public Intel ISA
documentation; the Xe DPAS layouts were derived primarily from Intel's public
DPAS/VISA instruction documentation and cross-checked against the CUTLASS
experimental Xe backend.

### NVIDIA Atoms

```python
from tensor_layouts.atoms_nv import *

atom = SM90_64x64x16_F16F16F16_SS
print(atom.name)        # SM90_64x64x16_F16F16F16_SS
print(atom.shape_mnk)   # (64, 64, 16)
print(atom.c_layout)    # Thread-value layout for C accumulator
```

Supported architectures: SM70 (Volta), SM75 (Turing), SM80 (Ampere),
SM89 (Ada Lovelace), SM90 (Hopper GMMA), SM100 (Blackwell UMMA),
SM120 (Blackwell B200).

### AMD Atoms

```python
from tensor_layouts.atoms_amd import *

atom = CDNA3_32x32x16_F32F8F8_MFMA
print(atom.name)        # CDNA3_32x32x16_F32F8F8_MFMA
print(atom.shape_mnk)   # (32, 32, 16)
print(atom.c_layout)    # Thread-value layout for C accumulator
```

Supported architectures: CDNA1 (gfx908 / MI100), CDNA2 (gfx90a / MI200),
CDNA3 (gfx942 / MI300), CDNA3+ (gfx950).

### Intel Xe DPAS Atoms

```python
from tensor_layouts.atoms_xe import *

atom = XeHPC_8x8x8_F32F16F16_DPAS
print(atom.name)        # XeHPC_8x8x8_F32F16F16_DPAS
print(atom.shape_mnk)   # (8, 8, 8)
print(atom.c_layout)    # Thread-value layout for C accumulator
```

Supported targets: Xe-HPC (Ponte Vecchio / Data Center Max) and
Xe-HPG (Arc / DG2) subgroup DPAS instructions.

### Intel Xe / Xe4 / Xe5 atoms (sycl-tla-faithful)

Beyond the pedagogical DPAS atoms above, the library ships **CPU-executable
emulations of every MMA and Copy atom** in the Intel sycl-tla (CUTLASS-Xe)
fork, translated directly from the C++ `MMA_Traits<>` / `Copy_Traits<>`:

```python
from tensor_layouts.atoms_xe  import MMA_ATOMS_XE, COPY_ATOMS_XE   # XE_DPAS_TT, 1D/2D copies
from tensor_layouts.atoms_xe4 import MMA_ATOMS_XE4, COPY_ATOMS_XE4  # TMM, AMMA, ADMA, LDSM, EU_COPY
from tensor_layouts.atoms_xe5 import MMA_ATOMS_XE5, COPY_ATOMS_XE5  # single/dual-group AMMA, MX
from tensor_layouts.atoms_xe_common import wi_interleave, make_slm_layout_elem
```

- **`atoms_xe`** — faithful `XE_DPAS_TT` (subgroup 16, VNNI `wi_interleave`) plus
  the Xe 1D/2D block copies.
- **`atoms_xe4`** — `XE4_TMM` (register MMA), `XE4_AMMA` (SLM-descriptor MMA, all
  barrier-tracking + block-scaled variants), and the ADMA / LDSM / EU_COPY /
  matrix copies. Subgroup 32.
- **`atoms_xe5`** — single-group **and dual-group** (`ThrID=2`) AMMA, LARGE
  core-matrix and block-scaled MX (incl. fp6) variants, plus ADMA dual / GCS
  copies.
- **`atoms_xe_common`** — the shared transforms: `wi_interleave`,
  `xe_interleaved_layout`, `make_ordered_layout`, and the bank-swizzled SLM
  **core-matrix** layout builders that the AMMA/ADMA descriptors reference.

Structural tests live in `tests/oracle_xe{,4,5}.py`; see
[`docs/xe_atoms.md`](docs/xe_atoms.md) and the worked example notebooks below.

### Intel AMX Atoms

```python
from tensor_layouts.atoms_amx import *

atom = AMX_16x16x32_F32BF16BF16F32
print(atom.name)        # AMX_16x16x32_F32BF16BF16F32
print(atom.shape_mnk)   # (16, 16, 32)
print(atom.c_layout)    # Thread-value layout for C accumulator
```

Supported targets: Intel AMX tile matrix-multiply instructions
(`tdpbf16ps`, `tdpfp16ps`, `tdpbssd`, `tdpbsud`, `tdpbusd`, `tdpbuud`).

## Visualization

With `pip install tensor-layouts[viz]`:

```python
from tensor_layouts import Layout, Swizzle
from tensor_layouts.viz import draw_layout, draw_swizzle

draw_layout(Layout((8, 8), (8, 1)), title="Row-Major 8x8", colorize=True)
draw_swizzle(Layout((8, 8), (8, 1)), Swizzle(3, 0, 3), colorize=True)
```

<p align="center">
  <img src="https://raw.githubusercontent.com/facebookresearch/tensor-layouts/main/docs/images/row_major_8x8.png" alt="Row-Major 8x8 layout" width="400">
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/facebookresearch/tensor-layouts/main/docs/images/swizzle_8x8.png" alt="Swizzle(3, 0, 3) applied to row-major 8x8" width="800">
</p>

See [`examples/viz.ipynb`](https://github.com/facebookresearch/tensor-layouts/blob/main/examples/viz.ipynb) for a full
gallery of layout, swizzle, MMA atom, and tiled MMA visualizations, and the
[Notebook gallery](#notebook-gallery) below for the full set.

## Documentation

- Example scripts assume `tensor-layouts` is installed.
  From a repo checkout, run `pip install -e .` first, or `pip install -e ".[viz]"`
  for visualization examples.
- [Layout Algebra API](https://github.com/facebookresearch/tensor-layouts/blob/main/docs/layout_api.md) — construction, querying, compose, complement, divide, product
- [Visualization API](https://github.com/facebookresearch/tensor-layouts/blob/main/docs/viz_api.md) — draw_layout, draw_swizzle, draw_mma_layout, and more
- [Layout Examples](https://github.com/facebookresearch/tensor-layouts/blob/main/examples/layouts.py) — runnable script covering the full algebra (`python3 examples/layouts.py`)
- [Visualization Examples](https://github.com/facebookresearch/tensor-layouts/blob/main/examples/viz.py) — runnable script generating all visualization types (`python3 examples/viz.py`)

### Notebook gallery

- [Visualization Notebook](https://github.com/facebookresearch/tensor-layouts/blob/main/examples/viz.ipynb) — `viz.ipynb`: layout, swizzle, MMA atom, and tiled MMA visualizations
- [Algorithms Notebook](https://github.com/facebookresearch/tensor-layouts/blob/main/examples/algorithms.ipynb) — `algorithms.ipynb`: derivations of compose, complement, divide, product
- [Applications Notebook](https://github.com/facebookresearch/tensor-layouts/blob/main/examples/applications.ipynb) — `applications.ipynb`: applied examples (paper-style)
- [GEMM Notebook](https://github.com/facebookresearch/tensor-layouts/blob/main/examples/gemm.ipynb) — `gemm.ipynb`: a fully explained NVIDIA GEMM kernel built up using layout algebra

#### Intel Xe4 / Xe5 atom & kernel notebooks

- [`examples/xe_atoms.ipynb`](examples/xe_atoms.ipynb) — overview of the Xe/Xe4/Xe5 atom families and their layout transformations (DPAS `wi_interleave`, SLM core matrix, EU copy, dual-group).
- [`examples/xe4_atoms.ipynb`](examples/xe4_atoms.ipynb) / [`examples/xe5_atoms.ipynb`](examples/xe5_atoms.ipynb) — every atom in each family, layouts printed and drawn.
- **[`examples/xe4/`](examples/xe4/) and [`examples/xe5/`](examples/xe5/)** — one **self-contained functional notebook + runnable script** per sycl-tla tutorial example under `examples/cute/tutorial/{xe4,xe5}/`, including the **FMHA4 forward** attention ([`examples/xe4/fmha4_fwd_xe4.ipynb`](examples/xe4/fmha4_fwd_xe4.ipynb)). Each works cell by cell through load → MMA → reduce/softmax/dual-split → epilogue → store on small concrete data, printing the actual numbers and the matching Xe layout at each step (SLM swizzle round-trips, accumulator ownership, LDSM warp-row co-residence, block-scale application, GCS zero-fill). See the per-directory `README.md`.
- [`examples/xe4_xe5_kernels.py`](examples/xe4_xe5_kernels.py) — cross-config tiling summary (GEMM 256×256×128, grouped, block-scaled, dual-group, FMHA4) showing how each CTA tile decomposes into MMA-atom tiles.

## What's new — Intel Xe4 / Xe5 support

This fork adds a complete, GPU-free emulation of the Intel Xe4/Xe5 CuTe atom
layer on top of the original NVIDIA/AMD layout-algebra library:

- **New source modules** `atoms_xe_common`, `atoms_xe4`, `atoms_xe5` and an
  extended `atoms_xe` (see [Intel Xe / Xe4 / Xe5 atoms](#intel-xe--xe4--xe5-atoms-sycl-tla-faithful)).
- **Structural test suites** `tests/oracle_xe{,4,5}.py`.
- **Educational example suite**: 28 self-contained functional notebooks + scripts
  (`examples/xe4/`, `examples/xe5/`) porting the sycl-tla tutorial GEMMs and the
  FMHA4 forward kernel, plus atom-family and overview notebooks.
- **Docs**: [`docs/xe_atoms.md`](docs/xe_atoms.md).

## Testing

```bash
pip install -e ".[test]"
pytest tests/
```

For local linting, install the dev extras and run Ruff on the Python sources:

```bash
pip install -e ".[dev]"
ruff check src/ tests/ examples/
```

The default Ruff configuration excludes `*.ipynb`; notebooks are treated as
worked material rather than part of the Python lint surface.

Oracle tests cross-validate against vendor reference implementations and are
skipped automatically if the corresponding tool is unavailable:

```bash
# NVIDIA pycute oracle
pip install -e ".[test,oracle-nv]"
pytest tests/oracle_nv.py

# Direct CuTe C++ oracle
# Requires a C++ compiler plus CUTLASS/CUDA headers in the active environment.
pytest tests/oracle_cute_cpp.py

# AMD (cross-validation against amd_matrix_instruction_calculator)
# AMD's tool is not on PyPI; clone it from GitHub and put it on PYTHONPATH:
#   git clone https://github.com/ROCm/amd_matrix_instruction_calculator
#   export PYTHONPATH="$PWD/amd_matrix_instruction_calculator:$PYTHONPATH"
pip install -e ".[test]"
pytest tests/oracle_amd.py
```

## References

- [CuTe Layout Representation and Algebra — Cris Cecka](https://arxiv.org/abs/2603.02298)
- [Categorical Foundations for CuTe Layouts — Jack Carlisle, Jay Shah, Reuben Stern, Paul VanKoughnett](https://arxiv.org/abs/2601.05972)
- [CuTe Documentation](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/cute/00_quickstart.md)
- [NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [NVIDIA Cutlass](https://github.com/NVIDIA/cutlass)
- [Cutlass MMA Atoms](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/cute/0t_mma_atom.md)
- [AMD Matrix Instruction Calculator](https://github.com/ROCm/amd_matrix_instruction_calculator)
- [AMD Matrix Cores Lab Notes](https://gpuopen.com/learn/amd-lab-notes/amd-lab-notes-matrix-cores-readme/)
- [Intel DPAS / VISA instruction specification](https://github.com/intel/intel-graphics-compiler/blob/master/documentation/visa/instructions/DPAS.md)
- [Intel Architecture Instruction Set Extensions Programming Reference (AMX)](https://www.intel.com/content/dam/develop/external/us/en/documents/architecture-instruction-set-extensions-programming-reference)
- [Intel 64 and IA-32 Architectures Software Developer's Manual, Volume 2A](https://www.intel.com/content/dam/www/public/us/en/documents/manuals/64-ia-32-architectures-software-developer-vol-2a-manual.pdf)

## License

MIT License. See [LICENSE](https://github.com/facebookresearch/tensor-layouts/blob/main/LICENSE) for details.

## Repo

tensor-layouts is maintained out of [https://github.com/jduprat/tensor-layouts.git](https://github.com/jduprat/tensor-layouts.git)
