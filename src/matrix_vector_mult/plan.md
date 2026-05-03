# SpMV Phase Implementation Plan

## Overview

Implement sparse matrix-vector multiplication (SpMV) with progressive parallelization from laptop prototype → shared-memory (OpenMP) → distributed-memory (MPI). User has 8-core laptop (4 usable) + UTD Sysbio/Ganymede HPC cluster access.

**User Profile:**
- Laptop: 8 cores, 8GB RAM (typically 4 cores available)
- HPC: UTD Sysbio/Ganymede cluster
- Implementation approach: Python (prototyping) → C/C++ (OpenMP/MPI optimization)
- Matrices: Synthetic (Laplacian, asymmetric, symmetric) + external (SuiteSparse)
- Metrics: Wall-clock time, speedup, GFlop/s, memory bandwidth

---

## Phase Decomposition & Step-by-Step Plan

### PHASE 1: Serial Prototype on Laptop

**Goal:** Establish baseline SpMV implementation, correctness validation, and serial timings.

#### 1a. Create Python SpMV Implementation
- **Location:** `src/matrix_vector_mult/spmv_python.py` (NEW)
- **Implement:** `spmv(A: spmatrix, x: ndarray) -> ndarray` for COO, CSR, CSC, LIL
- **Correctness:** Cross-check all formats against dense reference `y = A @ x_dense`
- **Timing:** Use `time.perf_counter()` + `tracemalloc` (consistent with assembly phase)
- **Output:** Result vector + timings dict `{format: avg_time_s, memory_mb}`
- **Test matrices:**
  - Synthetic: Laplacian (10×10 to 100×100 grid), asymmetric (n=1000 to 10000)
  - External: Load 1-2 matrices from SuiteSparse Matrix Collection (use `scipy.io.mmread`)

#### 1b. Implement Matrix Loading Module
- **Location:** `src/load_matrices.py` (EXTEND existing if present)
- **Add functions:**
  - `load_matrix_market(url_or_path)` — load .mtx files from SuiteSparse
  - `load_synthetic_matrices()` — return dict of assembly-generated matrices
  - `verify_matrix_properties(A)` — check dimensions, nnz, condition number
- **Output:** Matrices in COO format (for consistency); converts to CSR/CSC as needed

#### 1c. Create Baseline Benchmark Script
- **Location:** `src/matrix_vector_mult/benchmark_spmv_serial.py` (NEW)
- **Signature:** `benchmark_spmv(matrix_sizes: List[int], repeats: int, outdir: str)`
- **Tests:**
  - For each size in [1000, 5000, 10000] (small, on laptop)
  - For each format in [CSR, CSC, COO, LIL]
  - Generate random x ∈ ℝⁿ, repeat SpMV `repeats` times
  - Collect: avg_time_s, min_time_s, max_time_s, memory_mb, GFlop/s (= 2*nnz / time / 1e9)
- **Output CSV:** `src/matrix_vector_mult/results/spmv_serial_baseline.csv` with columns:
  - N, NNZ, Format, Avg_Time_s, Min_Time_s, Max_Time_s, Memory_MB, GFlops
- **Validation:** Verify y matches dense result for first run
- **CLI:** `python benchmark_spmv_serial.py --matrix-sizes 1000 5000 10000 --repeats 5 --outdir results`

#### 1d. Analyze Serial Results
- **Metrics to report:**
  - Best format (fastest) by matrix size
  - Memory footprint comparison
  - Baseline GFlop/s (for speedup comparison later)
- **Visualization:** `spmv_serial_comparison.png` with 2 panels:
  - Panel 1: Time vs format (bar chart or line plot by matrix size)
  - Panel 2: GFlop/s vs format

---

### PHASE 2: Shared-Memory Parallelization (OpenMP) on Laptop

**Goal:** Implement OpenMP C/C++ SpMV, test thread scaling (1→2→4 threads), measure speedup/efficiency.

#### 2a. Implement C++ SpMV with OpenMP
- **Location:** `src/matrix_vector_mult/spmv_openmp.cpp` + `spmv_openmp.h` (NEW)
- **Formats:** CSR primary (most thread-friendly), plus CSC, COO fallback
- **Parallelization:**
  - CSR: Parallelize outer loop (`#pragma omp parallel for`) over rows
  - CSC: Similar row distribution or column stripping
  - Use `#pragma omp atomic` for result accumulation on shared y
- **Build:** Add to Makefile or CMakeLists.txt with `-fopenmp`
- **API:** `void spmv_csr_openmp(int n, const int* row_ptr, const int* col_idx, const double* values, const double* x, double* y, int num_threads)`

#### 2b. Create Python Wrapper (ctypes or cffi)
- **Location:** `src/matrix_vector_mult/spmv_wrapper.py` (NEW)
- **Wrapper:** Call C++ functions from Python, pass NumPy arrays
- **Fallback:** If binding fails, use slow pure-Python loop (for development)
- **Test:** Compare wrapper results against NumPy reference

#### 2c. Implement Thread Scaling Benchmark
- **Location:** `src/matrix_vector_mult/benchmark_spmv_openmp.py` (NEW)
- **Signature:** `benchmark_spmv_openmp(matrix_sizes: List[int], threads: List[int], repeats: int, outdir: str)`
- **Tests:**
  - For each size in [1000, 5000, 10000]
  - For each thread count in [1, 2, 4] (laptop limit)
  - For each format (CSR primary, CSC fallback)
  - Warm-up: 1 run before timing (cache effects)
  - Timed: `repeats` runs with thread limit set via `omp_set_num_threads()`
- **Output CSV:** `src/matrix_vector_mult/results/spmv_openmp_scaling.csv` with columns:
  - N, NNZ, Format, NumThreads, Avg_Time_s, GFlops, Speedup_vs_1Thread
  - Speedup = time_1thread / time_N_threads
- **Validation:** Verify results match serial version at num_threads=1

#### 2d. Analyze Thread Scaling Results
- **Metrics:**
  - Speedup curve (1→2→4 threads): Should approach linear scaling
  - Efficiency: Speedup / num_threads (ideal = 1.0, realistic = 0.7-0.9)
  - Identify bottlenecks (memory bandwidth limited? Load balancing?)
- **Visualization:** `spmv_openmp_scaling.png` with 2 panels:
  - Panel 1: Speedup vs thread count (line plot with markers per matrix size)
  - Panel 2: GFlop/s vs thread count

#### 2e. Memory Bandwidth Analysis (Optional but Recommended)
- **Measurement:** Capture memory traffic using perf (Linux) or Instruments (macOS)
- **Calculate:** Bytes per second (x vector = 8n, y vector = 8n, matrix = 8*nnz + 4*nnz)
- **Compare:** vs peak bandwidth (laptop ≈ 30-50 GB/s typical)
- **Roofline:** Plot performance vs arithmetic intensity (GFlops / GB/s)

---

### PHASE 3: Distributed Memory Parallelization (MPI) on HPC Cluster

**Goal:** Implement MPI SpMV with process-level scaling, test on UTD cluster with larger matrices.

#### 3a. Setup HPC Environment
- **Cluster:** UTD Sysbio/Ganymede
- **Modules to load:** `gcc`, `openmpi`, `cmake` (or check cluster docs)
- **Test access:** `ssh cluster.utd.edu; module load openmpi; mpicc --version`
- **Compute node specs:** Look up core count, memory, interconnect (e.g., Infiniband vs Ethernet)

#### 3b. Implement MPI SpMV in C/C++
- **Location:** `src/matrix_vector_mult/spmv_mpi.cpp` + `spmv_mpi.h` (NEW)
- **Distribution strategy:**
  - Row-wise: Distribute matrix rows across ranks; each rank owns subset of rows
  - All-gather x: Each rank broadcasts its local x values (or use MPI_Allgather)
  - Local SpMV: Each rank computes y_local = A_local @ x_global
  - Result: Distributed y vector (or gather to rank 0)
- **Format:** CSR best for row-distribution
- **Build:** Compile with `mpicc` or `mpicxx`
- **API:** `void spmv_csr_mpi(int n, const int* row_ptr, const int* col_idx, const double* values, const double* x, double* y)`

#### 3c. Create MPI Python Wrapper (mpi4py)
- **Location:** `src/matrix_vector_mult/spmv_mpi_wrapper.py` (NEW)
- **Library:** `mpi4py` (install on cluster)
- **Wrapper:** Pure Python SpMV with MPI_Scatter/Gather for matrix distribution
- **Alternative:** Call C/C++ wrapper via ctypes + MPI communication layer
- **Test:** Run on small matrix with 2, 4, 8 processes

#### 3d. Implement Process Scaling Benchmark on Cluster
- **Location:** `src/matrix_vector_mult/benchmark_spmv_mpi.py` (NEW)
- **Signature:** `benchmark_spmv_mpi(matrix_sizes: List[int], processes: List[int], repeats: int, outdir: str)`
- **Tests:**
  - For each size in [10000, 50000, 100000] (medium→large, suited to cluster)
  - For each process count in [1, 2, 4, 8, 16, ...] up to cluster node count
  - Matrices: Synthetic + 1-2 external (SuiteSparse)
  - Warm-up + timed runs
- **Output CSV:** `src/matrix_vector_mult/results/spmv_mpi_scaling.csv` with columns:
  - N, NNZ, NumProcs, Avg_Time_s, GFlops, Speedup_vs_1Proc, Efficiency
  - Speedup_vs_1Proc = time_1proc / time_N_procs
  - Efficiency = Speedup / NumProcs
- **Job submission:** SLURM script (or cluster scheduler):
  ```bash
  #!/bin/bash
  #SBATCH --nodes=2 --ntasks-per-node=8 --time=01:00:00
  mpirun python benchmark_spmv_mpi.py --matrix-sizes 10000 50000 100000 --processes 1 2 4 8 16 --repeats 3 --outdir results_cluster
  ```

#### 3e. Analyze MPI Scaling Results
- **Metrics:**
  - Speedup vs process count: Linear ideal, realistic ≈ 0.6-0.8 (communication overhead)
  - Efficiency curves
  - Strong scaling: Fix matrix size, vary process count (typically poor scaling for large p)
  - Weak scaling: Scale matrix with process count (keep nnz/proc constant)
- **Visualization:** `spmv_mpi_scaling.png` with 2-3 panels:
  - Panel 1: Speedup vs process count (1→16)
  - Panel 2: Efficiency vs process count
  - Panel 3 (optional): Strong vs weak scaling comparison

---

### PHASE 4: Consolidated Analysis & Reporting

#### 4a. Cross-Phase Comparison
- **Location:** `src/matrix_vector_mult/analyze_spmv_results.py` (NEW)
- **Combine:**
  - Serial baseline (Phase 1)
  - OpenMP scaling (Phase 2, laptop)
  - MPI scaling (Phase 3, cluster)
- **Normalize:** All times relative to serial baseline (best format on laptop)
- **Metrics:** Speedup, efficiency, absolute GFlop/s

#### 4b. Visualization: Comprehensive Report
- **File:** `src/matrix_vector_mult/results/SPMV_SCALING_REPORT.png` (multi-panel figure)
  - Panel 1: **Speedup curves** (x=procs/threads, y=speedup, lines for serial→OpenMP→MPI)
  - Panel 2: **Efficiency** (x=procs/threads, y=efficiency %)
  - Panel 3: **Absolute GFlop/s** (x=procs/threads, y=GFlop/s)
  - Panel 4: **Memory bandwidth utilization** (if measured)
  - Ideal/roofline lines for reference
- **Summary table:** `SPMV_SUMMARY.txt`
  - Best configuration per platform
  - Bottleneck analysis (compute vs memory vs communication)
  - Recommendations for each phase

#### 4c. Correctness Validation Report
- **File:** `src/matrix_vector_mult/results/CORRECTNESS_VALIDATION.txt`
- **Verify:** All implementations match dense reference for small matrices
- **Cross-format:** CSR, CSC, COO all produce same result
- **Cross-phase:** Serial, OpenMP, MPI produce consistent results (within numerical precision)

---

## Project Structure (Updated)

```
utd-hpc-sparse-matrix-project/
├── src/
│   ├── assembly/                         # (existing, Phase 1 complete)
│   │   ├── *.py                          # laplacian, symmetric, asymmetric
│   │   └── results/                      # Assembly benchmarks
│   │
│   ├── matrix_vector_mult/               # (NEW, SpMV/SpMM area)
│   │   ├── spmv_python.py                # Serial Python implementation
│   │   ├── spmv_openmp.cpp/.h            # OpenMP C++ implementation
│   │   ├── spmv_mpi.cpp/.h               # MPI C++ implementation
│   │   ├── spmv_wrapper.py               # ctypes/cffi wrapper
│   │   ├── spmv_mpi_wrapper.py           # mpi4py wrapper
│   │   ├── benchmark_spmv_serial.py      # Phase 1 benchmark
│   │   ├── benchmark_spmv_openmp.py      # Phase 2 benchmark
│   │   ├── benchmark_spmv_mpi.py         # Phase 3 benchmark
│   │   ├── analyze_spmv_results.py       # Cross-phase analysis
│   │   ├── run_spmv_laptop.sh            # Laptop execution script
│   │   ├── run_spmv_cluster.sh           # Cluster SLURM script
│   │   ├── Makefile / CMakeLists.txt     # Build system
│   │   ├── plan.md                       # This file
│   │   │
│   │   └── results/                      # (NEW, SpMV results subdirectory)
│   │       ├── spmv_serial_baseline.csv     # Phase 1 output
│   │       ├── spmv_serial_comparison.png
│   │       ├── spmv_openmp_scaling.csv     # Phase 2 output
│   │       ├── spmv_openmp_scaling.png
│   │       ├── spmv_mpi_scaling.csv        # Phase 3 output
│   │       ├── spmv_mpi_scaling.png
│   │       ├── SPMV_SCALING_REPORT.png     # Phase 4 consolidated
│   │       ├── SPMV_SUMMARY.txt
│   │       └── CORRECTNESS_VALIDATION.txt
│   │
│   ├── load_matrices.py                  # (EXTEND: add SuiteSparse loading)
│   │
│   └── utils/                            # (existing, reuse if available)
│       ├── timing.py
│       ├── validation.py
│       └── ...
│
└── README.md (update with SpMV section)
```

---

## Verification Steps

**Phase 1 (Serial):**
1. Run `benchmark_spmv_serial.py` with 3 matrix sizes, 3 repeats
2. Verify all formats produce identical y (within 1e-12 tolerance)
3. Check GFlop/s is positive and reasonable (0.1-5 GFlop/s for laptop serial)

**Phase 2 (OpenMP):**
1. Verify OpenMP version matches serial within tolerance at num_threads=1
2. Speedup should be: 1.0 (1 thread), ~1.8-1.95 (2 threads), ~3.5-3.9 (4 threads)
3. GFlop/s should increase with thread count

**Phase 3 (MPI):**
1. Run on 1 process (serial baseline comparison)
2. Run on 2, 4, 8 processes, verify speedup < linear (communication penalty)
3. Weak scaling test: Efficiency should stay >0.6 as problem scales

**Phase 4:**
1. All three phases present in consolidated plots
2. No data gaps or anomalies
3. Physical reasonableness check (no negative speedup, efficiency ≤ 1.0)

---

## Dependencies & Installation

**Laptop (Python):**
- `numpy`, `scipy`, `pandas`, `matplotlib` (existing from assembly)
- OpenMP: Ship with compiler (gcc/clang)
- Optional: `mpi4py` (if prototyping MPI in Python)

**Laptop (C/C++):**
- Compiler with OpenMP support (gcc -fopenmp)
- CMAKE (optional, for build system)

**Cluster (MPI):**
- Module load gcc, openmpi (or check cluster docs)
- mpi4py (if using Python wrapper)

---

## Timeline & Execution Order

1. **Week 1 (Phase 1):** Implement Python SpMV, load matrices, run serial baseline
   - Deliverable: `spmv_serial_baseline.csv` + plot
2. **Week 2 (Phase 2):** Implement OpenMP C++, wrapper, thread scaling benchmark
   - Deliverable: `spmv_openmp_scaling.csv` + speedup plot (should see near-linear speedup to 4 threads)
3. **Week 3 (Phase 3):** Set up cluster access, implement MPI, run process scaling
   - Deliverable: `spmv_mpi_scaling.csv` + process scaling plot
4. **Week 4 (Phase 4):** Consolidate results, write analysis, generate final report
   - Deliverable: `SPMV_SCALING_REPORT.png` + summary + recommendations

---

## Key Decisions Made (from user input)

- ✅ Start with Python (Phase 1), then C/C++ (Phases 2-3)
- ✅ Use both synthetic + external (SuiteSparse) matrices
- ✅ Three-phase progression: Serial → OpenMP → MPI
- ✅ Thread scaling: Powers of 2 (1, 2, 4) on laptop
- ✅ Comprehensive metrics: Time, speedup, GFlop/s, memory bandwidth
- ✅ UTD cluster (Sysbio/Ganymede) for MPI testing

---

## Further Considerations

1. **Matrix Selection for Each Phase:**
   - Phase 1 (serial): Small matrices (n=1K-10K) for quick validation
   - Phase 2 (OpenMP): Medium matrices (n=5K-100K) to see parallelization benefit
   - Phase 3 (MPI): Large matrices (n=50K-1M+) to justify distributed memory overhead

2. **Weak Scaling Analysis (Optional):**
   - Phase 3 could include weak scaling test: Grow matrix size with process count
   - More realistic for HPC (usually p cores get p-sized problems)
   - Recommendation: Add as secondary analysis if time permits

3. **Format Optimization:**
   - Phase 1 tests all 4 formats; Phase 2-3 can focus on best performer (likely CSR)
   - Trade-off: Comprehensive vs focused benchmarking

4. **Load Matrix Fallback:**
   - If SuiteSparse download fails, have offline .mtx files or fallback to synthetic only
   - Recommendation: Include 2-3 standard test matrices in repo (small size)
