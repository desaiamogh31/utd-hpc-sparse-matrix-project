# Phase 3a: Sparse Matrix-Matrix Multiplication (SpMM)

## Overview

Phase 3a implements **Sparse Matrix-Matrix Multiplication (SpMM)**: computing $C = A \cdot B$ where:
- $A$ is a sparse matrix (m × n)
- $B$ is a dense matrix (n × k)
- $C$ is the sparse result matrix (m × k)

This phase serves as the bridge between Phase 2's SpMV (single column) and Phase 3b's distributed MPI implementations. SpMM exhibits better **arithmetic intensity** than SpMV due to data reuse: the sparse matrix $A$ is accessed $k$ times (once per column of $B$), enabling better cache utilization and parallelization opportunities.

## Phase 3a Structure

### Serial Python Baseline (Week 1-2)

**File: `spmm_python.py`**

Implements three algorithmic variants:

1. **Row-wise**: Parallelize over rows of sparse matrix $A$
   - Algorithm: For each row $i$, compute $C[i, :] = A[i, :] @ B$
   - Characteristics:
     - CSR-friendly (natural row iteration)
     - Simple parallelization (minimal synchronization)
     - Baseline algorithm for comparison
   - Complexity: $O(\text{nnz}(A) \times k)$ FLOPs

2. **Outer-product**: Column-accumulation via rank-1 updates
   - Algorithm: Accumulate $C += A[:, j] \cdot B[j, :]^T$ for each column $j$ of $A$
   - Characteristics:
     - CSC-friendly (natural column iteration in CSC format)
     - Better cache reuse for large $k$ (fewer accesses to $A$)
     - Enables vectorization via SIMD
   - Complexity: $O(\text{nnz}(A) \times k)$ FLOPs

3. **Blocked Inner-product**: Cache-aware column blocking
   - Algorithm: Process $B$ in column blocks; accumulate results per block
   - Characteristics:
     - Best cache locality (fits working set in L3)
     - Most complex implementation
     - Optimal for large $k$ (compute-bound regime)
   - Complexity: $O(\text{nnz}(A) \times k)$ FLOPs
   - Parameter: `block_k` (default 32, tune for L3 cache size)

**Key Features:**
- All three algorithms return **sparse C** (CSR format)
  - Sparse output is more realistic for scientific computing (usually sparse · dense = sparse)
  - Reduces memory overhead vs. dense output
  - Requires dynamic accumulation structure (COO → CSR conversion)
- Format flexibility: Accept A in any scipy.sparse format (CSR, CSC, COO, LIL)
- Validation against dense reference for correctness
- Performance profiling: GFlop/s, memory usage, nnz(C) tracking

### Benchmarking Suite

**File: `benchmark_spmm_serial.py`**

Comprehensive benchmark driver that:
- **Test matrices**: 
  - Phase 2 local matrices: 1138_bus, bcsstk30, delaunay_n15/19, abb313, pkustk14
  - Synthetic larger matrices: 100K×100K (0.01% sparse), 500K×500K (0.001% sparse) for MPI readiness
- **Dense column counts**: 1, 4, 8, 16, 32, 64, 128, 256, 512
  - Captures memory-bound (small k) to compute-bound (large k) regimes
  - SpMM advantage grows with k (better arithmetic intensity)
- **Metrics per (A, B_cols, algorithm)**:
  - Execution time (mean ± std, min)
  - GFlop/s: $\frac{2 \cdot \text{nnz}(A) \cdot k}{10^9 \cdot \text{time}}$
  - Output sparsity: nnz(A), nnz(C)
  - Memory usage: Storage for A, B, C
- **Output**: CSV file `benchmark_spmm_serial_baseline.csv`

### Unit Tests

**File: `tests/test_spmm.py`**

20 test cases covering:
- Small dense and sparse matrices
- Single vs. multiple dense columns (SpMV reduction case)
- Random sparse matrices
- Empty rows/columns
- All three algorithms
- Format conversions (CSR → CSC → COO → LIL)
- Validation correctness

Run tests:
```bash
pytest tests/test_spmm.py -v
```

## Quick Start

### 1. Run Unit Tests

```bash
cd /path/to/utd-hpc-sparse-matrix-project
pytest tests/test_spmm.py -v
```

Expected output: 20 passed tests

### 2. Run Serial Baseline Benchmark

```bash
cd src/matrix_matrix_mult
python benchmark_spmm_serial.py \
    --output results/benchmark_spmm_serial_baseline.csv \
    --cache-dir matrices \
    --repeats 3 \
    --b-cols 1 4 8 16 32 64 128 256 512
```

This will:
1. Load available .mtx matrices from `matrices/` directory
2. For each matrix and column count, benchmark all three algorithms
3. Write results to `results/benchmark_spmm_serial_baseline.csv`
4. Display progress with GFlop/s per algorithm

### 3. Analyze Results

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load results
df = pd.read_csv('results/benchmark_spmm_serial_baseline.csv')

# Plot: GFlop/s vs. B columns, grouped by algorithm
for algo in ['row-wise', 'outer-product', 'blocked']:
    data = df[df['algorithm'] == algo]
    plt.plot(data['b_cols'], data['gflops'], label=algo, marker='o')

plt.xlabel('Dense Matrix Columns (k)')
plt.ylabel('GFlop/s')
plt.legend()
plt.title('SpMM Algorithm Comparison')
plt.xscale('log')
plt.show()
```

## Algorithm Performance Characteristics

### Expected Results

| Regime | Dominant Algorithm | Reason |
|--------|-------------------|--------|
| **k = 1** (memory-bound) | Row-wise or Blocked | Minimal overhead; sequential access |
| **k = 4-16** (transition) | Blocked | Cache blocking amortizes overhead |
| **k ≥ 64** (compute-bound) | Outer-product | Best vectorization; minimal memory traffic |

### Actual Speedups Over Row-wise (Example)

For typical sparse matrices:
- **Outer-product**: 0.8-1.2× at k=1; 1.5-2.5× at k≥64
- **Blocked**: 1.0-1.1× at k=1; 1.8-3.0× at k≥128

Variation depends on:
- Matrix sparsity pattern (row-wise vs. column-wise distribution)
- L3 cache size (affects optimal block_k)
- Dense matrix layout (row vs. column major)

## Storage Formats and Implications

### Sparse Matrix A

| Format | Row-wise | Outer-product | Blocked | Notes |
|--------|----------|---------------|---------|-------|
| CSR | ✓✓ (optimal) | ✓ | ✓✓ | Natural for row iteration |
| CSC | ✓ | ✓✓ (optimal) | ✓ | Natural for column iteration |
| COO | ✓ | ✓ | ✓ | Requires implicit sorting |
| LIL | ~ | ~ | ~ | Slow indirect access; convert first |

**Recommendation**: Store A in CSR for Phase 3a (aligns with Phase 2); consider CSC if outer-product shows significant speedup in benchmarks.

### Dense Matrix B

Stored in **row-major** (C-contiguous) for cache-friendly access in tight inner loops.

### Output C

Stored in **CSR format** (sparse). Advantages:
- Reduces memory vs. dense output (typical scientific computing results in sparse C)
- Compatible with subsequent SpMV/SpMM operations
- Enables validation against dense reference

## Memory Requirements

Rough estimates for 1 benchmark run (A, B_cols=64, all algorithms):
- A storage: O(nnz(A))
- B storage: O(n × 64)
- Intermediate accumulation (COO): O(nnz(C))
- Total: ~2-3× sparse matrix footprint + dense B

For Phase 2 matrices (< 1M nnz), easily fits in modern RAM (< 1GB).

## Known Limitations & Future Work

1. **Sparse C overhead**: 
   - Dynamic accumulation (COO → CSR) has memory fragmentation cost
   - Dense output variant could be faster for some applications
   - Mitigation: Profile vs. blocked dense matrix multiplication

2. **Format conversion cost**:
   - Converting A to optimal format (CSC for outer-product) adds ~5-15% overhead
   - Recommendation: Precompute in optimal format for production

3. **Vectorization**:
   - Current implementation uses Python loops (NumPy element access)
   - C++ variant (Phase 3a Part 2) will add explicit SIMD with `#pragma omp simd`

4. **Large k (> 512)**:
   - Blocked algorithm requires tuning of `block_k` for different hardware
   - L3 miss rate may become bottleneck for k >> 1000

5. **Asymmetric sparsity**:
   - Row-wise parallelization may suffer from load imbalance if sparsity varies greatly
   - Mitigation: Dynamic scheduling (`#pragma omp schedule(dynamic)`) in C++ phase

## Validation & Correctness

All algorithms validate against dense reference computation:
```python
C_ref = A.toarray() @ B  # Dense reference
C_sparse = spmm(A, B, algorithm='...')  # Our sparse implementation
assert np.allclose(C_sparse.toarray(), C_ref, rtol=1e-10)
```

Tolerance: $10^{-10}$ (FP64 machine epsilon ≈ $2.2 \times 10^{-16}$; handles ~5-6 significant digits of error accumulation).

## Next Steps: Phase 3a Part 2 (C++ OpenMP)

Once serial baseline is complete:

1. **Create C++ implementation** (`spmm_openmp.cpp`):
   - Three algorithms using `#pragma omp parallel for`
   - Dynamic scheduling for load balancing
   - Vectorization hints with `#pragma omp simd`

2. **Thread scaling benchmarks**:
   - Strong scaling: 1→2→4→8 threads
   - Measure speedup and efficiency
   - Identify optimal thread count per algorithm

3. **Comparison against scipy**:
   - scipy.sparse doesn't have built-in SpMM, but we can compare against dense fallback

## References

- **GraphBLAS**: https://github.com/DrTimothyAldenDavis/GraphBLAS (reference algorithms)
- **Intel MKL SpMM**: Performance characteristics and tuning guide
- **Eigen Sparse**: https://eigen.tuxfamily.org/dox/ (inspiration for sparse formats)
- **CombBLAS**: Distributed sparse linear algebra (Phase 3b inspiration)

## Contact & Questions

For issues, questions, or optimizations: [Your contact info]
