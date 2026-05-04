# Phase 2: OpenMP SpMV Implementation

## Overview

Phase 2 introduces shared-memory parallelization using OpenMP. We parallelize the CSR SpMV computation across multiple threads on a single machine, testing thread scaling (1, 2, 4 threads).

**Key Goal:** Measure speedup from parallelization and identify scalability limits on multi-core hardware.

---

## Files

### `spmv_openmp.cpp`
C++ implementation with OpenMP parallelization.

**Key Features:**
- CSR format specialized (fastest serial format from Phase 1)
- Row-parallel computation: `#pragma omp parallel for schedule(static)`
- Thread count can be set at runtime
- Functions:
  - `spmv_csr_omp()` - Main SpMV with CSR + OpenMP
  - `get_num_threads()` - Query available threads
  - `set_num_threads()` - Set thread count

**Compilation:**
```bash
python build.py
# OR manually:
gcc -O3 -march=native -fopenmp -fPIC -shared -o spmv_openmp.so spmv_openmp.cpp
```

### `spmv_wrapper.py`
Python ctypes wrapper for compiled C extension.

**Key Functions:**
- `spmv_csr_omp(A, x, num_threads)` - Call OpenMP SpMV with specified threads
- `spmv_omp(A, x, num_threads)` - Dispatcher (auto-converts to CSR)
- `get_num_threads()` - Query max available threads
- `set_num_threads(n)` - Set default thread count

**Usage:**
```python
import spmv_wrapper
from scipy.sparse import random as sparse_random
import numpy as np

A = sparse_random(1000, 1000, density=0.005, format='csr')
x = np.random.randn(1000)

# Run with 2 threads
y = spmv_wrapper.spmv_csr_omp(A, x, num_threads=2)
```

### `benchmark_spmv_openmp.py`
Phase 2 benchmarking driver and visualization.

**Key Functions:**
- `benchmark_spmv_omp(A, x, num_threads, repeats)` - Time a single configuration
- `benchmark_spmv_openmp()` - Main loop over sizes and thread counts
- `load_serial_baseline()` - Load Phase 1 results for comparison
- `compute_speedup_efficiency()` - Calculate speedup and efficiency
- `create_visualizations()` - 4-panel plot (time, GFlop/s, speedup, efficiency)

**Output:**
- CSV: `results/spmv_openmp_results.csv` - Raw timing data
- CSV: `results/spmv_phase2_analysis.csv` - Speedup/efficiency analysis
- PNG: `results/spmv_openmp_scaling.png` - 4-panel visualization

### `build.py`
Automated compilation script.

**Usage:**
```bash
python build.py
```

Compiles `spmv_openmp.cpp` with:
- `-O3` optimization
- `-march=native` CPU-specific optimization
- `-fopenmp` OpenMP support
- `-fPIC -shared` for shared library

---

## Quick Start (Laptop)

### 1. Compile
```bash
cd src/matrix_vector_mult
python build.py
```

Expected output:
```
Building OpenMP SpMV shared library
...
✓ Build successful: spmv_openmp.so
```

### 2. Run Benchmark
```bash
python benchmark_spmv_openmp.py \
    --matrix-sizes 1000 5000 10000 \
    --thread-counts 1 2 4 \
    --repeats 5 \
    --nnz-ratio 5.0 \
    --outdir results
```

Adjust `--thread-counts` to match your CPU cores (max 4 for testing).

### 3. View Results
```bash
cat results/spmv_phase2_analysis.csv  # Speedup/efficiency data
open results/spmv_openmp_scaling.png   # Visualization
```

### Alternative: Using Bash Script (Similar to nbody_omp)

You can also run benchmarks using the provided bash script, which automatically tests multiple matrix sizes and thread counts:

```bash
# Make script executable
chmod +x run_spmv_omp.sh

# Run the script
./run_spmv_omp.sh
```

**What it does:**
1. Compiles OpenMP library (`python build.py`)
2. Loops through predefined matrix sizes (1000, 2000, 4000, 8000, 16000)
3. Tests each size with different thread counts (1, 2, 4)
4. Collects timing and GFlops data
5. Saves results to `results/spmv_openmp_runtimes.txt`

**Output file format:**
```
MatrixSize,Threads,Avg_Time_s,GFlops
1000,1,0.001234,5.23
1000,2,0.000678,9.52
1000,4,0.000389,16.58
2000,1,0.004890,5.15
...
```

**View results:**
```bash
cat results/spmv_openmp_runtimes.txt
```

**Customize the script:**
Edit `run_spmv_omp.sh` to change matrix sizes or thread counts:
```bash
MATRIX_SIZES=(1000 5000 10000 20000)  # Change these
THREAD_COUNTS=(1 2 4 8)               # Or these
```

---

## Expected Results

**Speedup Pattern (on 4-core laptop):**
| Threads | Expected Speedup | Expected Efficiency |
|---------|------------------|-------------------|
| 1       | 1.0              | 1.0               |
| 2       | ~1.8-1.95        | ~0.9-0.975        |
| 4       | ~3.5-3.9         | ~0.88-0.98        |

**Why not perfect scaling?**
- Memory bandwidth saturation (all cores share L3 cache, memory bus)
- OpenMP overhead (thread creation, synchronization)
- Load imbalance if matrix is structured (rare for random matrices)

---

## Testing Checklist

- [ ] `python build.py` completes successfully (spmv_openmp.so created)
- [ ] `spmv_wrapper.py` imports without errors
- [ ] `benchmark_spmv_openmp.py` runs to completion
- [ ] CSV files created with expected columns
- [ ] PNG visualization created (4 panels)
- [ ] Speedup values reasonable (1.0 < speedup < num_threads)
- [ ] Efficiency values reasonable (0 < efficiency ≤ 1.0)

---

## Customization

**Test fewer threads:**
```bash
python benchmark_spmv_openmp.py --thread-counts 1 2
```

**Larger matrices (more time):**
```bash
python benchmark_spmv_openmp.py --matrix-sizes 5000 10000 20000 --repeats 3
```

**Sparser/denser matrices:**
```bash
python benchmark_spmv_openmp.py --nnz-ratio 2.0  # Sparser
python benchmark_spmv_openmp.py --nnz-ratio 10.0 # Denser
```

---

## Troubleshooting

**`FileNotFoundError: spmv_openmp.so not found`**
- Run `python build.py` first to compile

**`gcc: command not found`**
- macOS: `xcode-select --install`
- Linux: `sudo apt-get install build-essential`

**Poor speedup (< 1.5 on 2 threads)**
- Matrix too small (overhead dominates)
- Try larger matrices: `--matrix-sizes 10000 20000`
- Check CPU isn't throttled: `sysctl machdep.cpu.frequency_max`

**Segmentation fault**
- Check matrix format (must be CSR after conversion)
- Ensure input arrays are contiguous (ctypes requirement)

---

## Next Phase

Phase 3 will extend to **distributed memory parallelization using MPI**, enabling scaling across multiple nodes on the HPC cluster. Results from Phase 2 will serve as the shared-memory baseline for speedup comparison.
