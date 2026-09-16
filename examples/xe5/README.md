# Emulated Intel XE5 tutorial GEMM/attention examples

Self-contained, **functional** ports of each sycl-tla example under
`examples/cute/tutorial/xe5/` (plus the FMHA4 forward kernel for xe4). Each
example ships **both** a Jupyter notebook (`<name>.ipynb`) and an equivalent
runnable script (`<name>.py`) with the *same* inlined code. Every cell/section
performs one operation (load / MMA / reduce / softmax / dual-split / store …) on
small concrete data and prints the resulting **data layout** — full when small, a
slice when large. Run a script with e.g. `python examples/xe5/amma_xe5.py`,
or open the notebook and run the cells top to bottom.

| Example (.ipynb / .py) | Mirrors | Demonstrates |
|------------------------|---------|--------------|
| [`adma_linear_copy_gcs.ipynb`](adma_linear_copy_gcs.ipynb) / [`adma_linear_copy_gcs.py`](adma_linear_copy_gcs.py) | `adma_linear_copy_gcs.cpp` | XE5 linear GCS copy |
| [`amma_adma_xe5_runtime_mma.ipynb`](amma_adma_xe5_runtime_mma.ipynb) / [`amma_adma_xe5_runtime_mma.py`](amma_adma_xe5_runtime_mma.py) | `amma_adma_xe5_runtime_mma.cpp` | XE5 runtime (shapeless) AMMA GEMM |
| [`amma_dual_mma_blockscaled_wg_xe5.ipynb`](amma_dual_mma_blockscaled_wg_xe5.ipynb) / [`amma_dual_mma_blockscaled_wg_xe5.py`](amma_dual_mma_blockscaled_wg_xe5.py) | `amma_dual_mma_blockscaled_wg_xe5.cpp` | XE5 dual-group block-scaled GEMM (workgroup) |
| [`amma_dual_mma_blockscaled_xe5.ipynb`](amma_dual_mma_blockscaled_xe5.ipynb) / [`amma_dual_mma_blockscaled_xe5.py`](amma_dual_mma_blockscaled_xe5.py) | `amma_dual_mma_blockscaled_xe5.cpp` | XE5 dual-group block-scaled GEMM (cluster) |
| [`amma_dual_mma_wg_xe5.ipynb`](amma_dual_mma_wg_xe5.ipynb) / [`amma_dual_mma_wg_xe5.py`](amma_dual_mma_wg_xe5.py) | `amma_dual_mma_wg_xe5.cpp` | XE5 dual-group AMMA GEMM (workgroup) |
| [`amma_dual_mma_xe5.ipynb`](amma_dual_mma_xe5.ipynb) / [`amma_dual_mma_xe5.py`](amma_dual_mma_xe5.py) | `amma_dual_mma_xe5.cpp` | XE5 dual-group AMMA GEMM (cluster) |
| [`grouped_gemm_adma_tensor_desc_update.ipynb`](grouped_gemm_adma_tensor_desc_update.ipynb) / [`grouped_gemm_adma_tensor_desc_update.py`](grouped_gemm_adma_tensor_desc_update.py) | `grouped_gemm_adma_tensor_desc_update.cpp` | XE5 grouped GEMM (tensor-desc update) |
| [`xe5_adma_amma_large.ipynb`](xe5_adma_amma_large.ipynb) / [`xe5_adma_amma_large.py`](xe5_adma_amma_large.py) | `xe5_adma_amma_large.cpp` | XE5 LARGE core-matrix AMMA GEMM |
| [`xe5_mx_sf_blockscaled_gemm.ipynb`](xe5_mx_sf_blockscaled_gemm.ipynb) / [`xe5_mx_sf_blockscaled_gemm.py`](xe5_mx_sf_blockscaled_gemm.py) | `xe5_mx_sf_blockscaled_gemm.cpp` | XE5 MX scale-factor block-scaled GEMM |
