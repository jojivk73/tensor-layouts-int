# Emulated Intel XE4 tutorial GEMM/attention examples

Self-contained, **functional** ports of each sycl-tla example under
`examples/cute/tutorial/xe4/` (plus the FMHA4 forward kernel for xe4). Each
example ships **both** a Jupyter notebook (`<name>.ipynb`) and an equivalent
runnable script (`<name>.py`) with the *same* inlined code. Every cell/section
performs one operation (load / MMA / reduce / softmax / dual-split / store …) on
small concrete data and prints the resulting **data layout** — full when small, a
slice when large. Run a script with e.g. `python examples/xe4/amma_xe4.py`,
or open the notebook and run the cells top to bottom.

| Example (.ipynb / .py) | Mirrors | Demonstrates |
|------------------------|---------|--------------|
| [`amma_adma_xe4.ipynb`](amma_adma_xe4.ipynb) / [`amma_adma_xe4.py`](amma_adma_xe4.py) | `amma_adma_xe4.cpp` | XE4 AMMA + ADMA GEMM |
| [`amma_adma_xe4_blockscaled_gemm.ipynb`](amma_adma_xe4_blockscaled_gemm.ipynb) / [`amma_adma_xe4_blockscaled_gemm.py`](amma_adma_xe4_blockscaled_gemm.py) | `amma_adma_xe4_blockscaled_gemm.cpp` | XE4 block-scaled (MX) GEMM |
| [`amma_adma_xe4_blockscaled_gemm_cluster.ipynb`](amma_adma_xe4_blockscaled_gemm_cluster.ipynb) / [`amma_adma_xe4_blockscaled_gemm_cluster.py`](amma_adma_xe4_blockscaled_gemm_cluster.py) | `amma_adma_xe4_blockscaled_gemm_cluster.cpp` | XE4 block-scaled GEMM (cluster) |
| [`amma_adma_xe4_cluster.ipynb`](amma_adma_xe4_cluster.ipynb) / [`amma_adma_xe4_cluster.py`](amma_adma_xe4_cluster.py) | `amma_adma_xe4_cluster.cpp` | XE4 AMMA + ADMA GEMM (cluster) |
| [`amma_xe4.ipynb`](amma_xe4.ipynb) / [`amma_xe4.py`](amma_xe4.py) | `amma_xe4.cpp` | XE4 basic AMMA GEMM |
| [`api_amma_row_copy_fir_1d_xe4.ipynb`](api_amma_row_copy_fir_1d_xe4.ipynb) / [`api_amma_row_copy_fir_1d_xe4.py`](api_amma_row_copy_fir_1d_xe4.py) | `api_amma_row_copy_fir_1d_xe4.cpp` | XE4 1D FIR via row-copy |
| [`api_amma_row_copy_tiled_xe4.ipynb`](api_amma_row_copy_tiled_xe4.ipynb) / [`api_amma_row_copy_tiled_xe4.py`](api_amma_row_copy_tiled_xe4.py) | `api_amma_row_copy_tiled_xe4.cpp` | XE4 tiled row-copy API GEMM |
| [`fmha4_fwd_xe4.ipynb`](fmha4_fwd_xe4.ipynb) / [`fmha4_fwd_xe4.py`](fmha4_fwd_xe4.py) | `xe4_fmha_fwd.cpp` | XE4 FMHA4 forward attention |
| [`gemm_async_reduce_xe4.ipynb`](gemm_async_reduce_xe4.ipynb) / [`gemm_async_reduce_xe4.py`](gemm_async_reduce_xe4.py) | `gemm_async_reduce_xe4.cpp` | XE4 GEMM async store-reduce |
| [`gemm_epilogue_adma_amma.ipynb`](gemm_epilogue_adma_amma.ipynb) / [`gemm_epilogue_adma_amma.py`](gemm_epilogue_adma_amma.py) | `gemm_epilogue_adma_amma.cpp` | XE4 fused ADMA+AMMA epilogue |
| [`gemm_epilogue_asymmetric_adma_amma.ipynb`](gemm_epilogue_asymmetric_adma_amma.ipynb) / [`gemm_epilogue_asymmetric_adma_amma.py`](gemm_epilogue_asymmetric_adma_amma.py) | `gemm_epilogue_asymmetric_adma_amma.cpp` | XE4 asymmetric ADMA+AMMA epilogue |
| [`gemm_eu_copy_matrix_atoms.ipynb`](gemm_eu_copy_matrix_atoms.ipynb) / [`gemm_eu_copy_matrix_atoms.py`](gemm_eu_copy_matrix_atoms.py) | `gemm_eu_copy_matrix_atoms.cpp` | XE4 EU-copy + matrix atoms GEMM |
| [`gemm_ldsm_warp_row_epilogue_xe4.ipynb`](gemm_ldsm_warp_row_epilogue_xe4.ipynb) / [`gemm_ldsm_warp_row_epilogue_xe4.py`](gemm_ldsm_warp_row_epilogue_xe4.py) | `gemm_ldsm_warp_row_epilogue_xe4.cpp` | XE4 GEMM LDSM warp-row epilogue |
| [`gemm_pipe_gmem_reduce_cluster_xe4.ipynb`](gemm_pipe_gmem_reduce_cluster_xe4.ipynb) / [`gemm_pipe_gmem_reduce_cluster_xe4.py`](gemm_pipe_gmem_reduce_cluster_xe4.py) | `gemm_pipe_gmem_reduce_cluster_xe4.cpp` | XE4 GEMM pipelined cluster gmem-reduce |
| [`gemm_sgsplit_reduce_xe4.ipynb`](gemm_sgsplit_reduce_xe4.ipynb) / [`gemm_sgsplit_reduce_xe4.py`](gemm_sgsplit_reduce_xe4.py) | `gemm_sgsplit_reduce_xe4.cpp` | XE4 GEMM subgroup-split reduce |
| [`gemm_softmax_xe4.ipynb`](gemm_softmax_xe4.ipynb) / [`gemm_softmax_xe4.py`](gemm_softmax_xe4.py) | `gemm_softmax_xe4.cpp` | XE4 GEMM + row softmax |
| [`gemm_tiled_tmma.ipynb`](gemm_tiled_tmma.ipynb) / [`gemm_tiled_tmma.py`](gemm_tiled_tmma.py) | `gemm_tiled_tmma.cpp` | XE4 tiled TMM GEMM |
| [`grouped_gemm_adma_tensor_desc_update.ipynb`](grouped_gemm_adma_tensor_desc_update.ipynb) / [`grouped_gemm_adma_tensor_desc_update.py`](grouped_gemm_adma_tensor_desc_update.py) | `grouped_gemm_adma_tensor_desc_update.cpp` | XE4 grouped GEMM (tensor-desc update) |
| [`grouped_gemm_ptr_only_uniform.ipynb`](grouped_gemm_ptr_only_uniform.ipynb) / [`grouped_gemm_ptr_only_uniform.py`](grouped_gemm_ptr_only_uniform.py) | `grouped_gemm_ptr_only_uniform.cpp` | XE4 grouped GEMM (ptr-array uniform) |
