# Agentic Kernel Appraiser (AKA)

This project develops the **Agentic Kernel Appraiser (AKA)** —  
an architecture-aware value model designed to **rank CUDA/Triton kernel candidates** based on:

- **Correctness** (compilation + runtime validity)  
- **Expected speedup** over PyTorch native implementations  
- **Compile/runtime/OOM risk**  

The goal is to prune bad runs early and save GPU hours, while guiding agentic code generation frameworks toward promising kernel implementations.

---

## Repository Structure

- **`scripts/`**
  - `load_sakana.py` – load the Sakana.ai CUDA dataset into Parquet.  
  - `make_features.py` – build feature tables (vectorization, shared memory, unrolling, code length, etc.).  
  - `eda.py` – exploratory data analysis and error categorization.  
  - `train_baseline.py` – XGBoost baselines for correctness + speedup, with SHAP feature importance plots.  
  - `ingest_cutlass.py` – ingestion pipeline for CUTLASS kernels, runs the profiler, parses CSV, and saves to Parquet. Supports optional Nsight Compute counters.  
  - `ncu_utils.py` – helpers for attaching Nsight Compute and collecting hardware counters.  

- **`ksn/`**
  - `features.py` – feature extraction utilities for Sakana and CUTLASS.  

---

## Requirements

- Python 3.10+  
- Packages: `pandas`, `numpy`, `xgboost`, `scikit-learn`, `shap`, `matplotlib`  
- CUTLASS with `cmake` + `nvcc`  
- Nsight Compute (`ncu`) for hardware counters  
