"""
Serial SpMM (Sparse Matrix-Matrix multiplication) helpers.

Provides SpMM operations for COO, CSR, CSC, and LIL sparse matrix formats
with three algorithmic variants:
  1. Row-wise: Parallelize over rows; each row processes all dense columns
  2. Outer-product: Column-accumulation style; process sparse matrix column-by-column
  3. Blocked inner-product: Block dense matrix for cache efficiency

Returns sparse output matrix C in CSR format.
"""

from __future__ import annotations
import gc
import time
import tracemalloc
from typing import Dict, Tuple
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix, lil_matrix, spmatrix
from scipy import sparse


def spmm_dense_reference(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Dense reference implementation C = A @ B.
    
    Parameters:
    - A: Dense matrix (m × n)
    - B: Dense matrix (n × k)
    
    Returns:
    - C: Dense result matrix (m × k)
    """
    return A @ B


def spmm_row_wise(A_csr: csr_matrix, B: np.ndarray) -> csr_matrix:
    """
    Row-wise SpMM: parallelize over rows of sparse matrix A.
    
    For each row i of A, compute C[i, :] = A[i, :] @ B
    This algorithm is CSR-friendly and enables natural parallelization.
    
    Parameters:
    - A_csr: Sparse matrix in CSR format (m × n)
    - B: Dense matrix (n × k)
    
    Returns:
    - C: Sparse result matrix (m × k) in CSR format
    
    Complexity: O(nnz(A) × k) FLOPs
    """
    m, n = A_csr.shape
    k = B.shape[1]
    
    # Result will be stored as COO initially, then converted to CSR
    row_indices = []
    col_indices = []
    values = []
    
    # Process each row of A
    for i in range(m):
        # Get row i from CSR
        row_start = A_csr.indptr[i]
        row_end = A_csr.indptr[i + 1]
        
        if row_start == row_end:
            # Row is empty, skip
            continue
        
        # Get non-zero indices and values for row i
        col_idx = A_csr.indices[row_start:row_end]
        row_vals = A_csr.data[row_start:row_end]
        
        # Compute C[i, :] = A[i, :] @ B
        # For each column j in result
        for j in range(k):
            # Accumulate: C[i,j] = sum_l A[i,l] * B[l,j]
            c_val = 0.0
            for l_idx, l in enumerate(col_idx):
                c_val += row_vals[l_idx] * B[l, j]
            
            if c_val != 0.0:
                row_indices.append(i)
                col_indices.append(j)
                values.append(c_val)
    
    # Convert COO to CSR
    C_coo = coo_matrix(
        (values, (row_indices, col_indices)), shape=(m, k)
    )
    return C_coo.tocsr()


def spmm_outer_product(A_csc: csc_matrix, B: np.ndarray) -> csr_matrix:
    """
    Outer-product SpMM: column-accumulation style.
    
    Accumulate rank-1 updates: C += A[:, j] * B[j, :]^T for each column j.
    This algorithm is CSC-friendly and better for vectorization.
    
    Parameters:
    - A_csc: Sparse matrix in CSC format (m × n)
    - B: Dense matrix (n × k)
    
    Returns:
    - C: Sparse result matrix (m × k) in CSR format
    
    Complexity: O(nnz(A) × k) FLOPs
    """
    m, n = A_csc.shape
    k = B.shape[1]
    
    # Use LIL format for efficient row-wise accumulation
    C_lil = lil_matrix((m, k), dtype=B.dtype)
    
    # For each column j of A
    for j in range(n):
        col_start = A_csc.indptr[j]
        col_end = A_csc.indptr[j + 1]
        
        if col_start == col_end:
            # Column is empty, skip
            continue
        
        # Get non-zero row indices and values for column j
        row_idx = A_csc.indices[col_start:col_end]
        col_vals = A_csc.data[col_start:col_end]
        
        # For each row i where A[i,j] is non-zero
        for i_idx, i in enumerate(row_idx):
            a_val = col_vals[i_idx]
            # Rank-1 update: C[i, :] += a_val * B[j, :]
            for jj in range(k):
                C_lil[i, jj] += a_val * B[j, jj]
    
    return C_lil.tocsr()


def spmm_blocked_inner_product(
    A_csr: csr_matrix, B: np.ndarray, block_k: int = 32
) -> csr_matrix:
    """
    Blocked inner-product SpMM: block dense matrix for cache efficiency.
    
    Process B in blocks of columns to improve cache reuse of A.
    This algorithm has better cache locality than row-wise for large k.
    
    Parameters:
    - A_csr: Sparse matrix in CSR format (m × n)
    - B: Dense matrix (n × k)
    - block_k: Block size for column dimension (tune for L3 cache)
    
    Returns:
    - C: Sparse result matrix (m × k) in CSR format
    
    Complexity: O(nnz(A) × k) FLOPs
    """
    m, n = A_csr.shape
    k = B.shape[1]
    
    # Result stored as COO, then converted to CSR
    row_indices = []
    col_indices = []
    values = []
    
    # Process B in column blocks
    for bk_start in range(0, k, block_k):
        bk_end = min(bk_start + block_k, k)
        block_cols = bk_end - bk_start
        
        # Process each row of A
        for i in range(m):
            row_start = A_csr.indptr[i]
            row_end = A_csr.indptr[i + 1]
            
            if row_start == row_end:
                continue
            
            col_idx = A_csr.indices[row_start:row_end]
            row_vals = A_csr.data[row_start:row_end]
            
            # Compute C[i, bk_start:bk_end] = A[i, :] @ B[:, bk_start:bk_end]
            for jj in range(block_cols):
                j = bk_start + jj
                c_val = 0.0
                for l_idx, l in enumerate(col_idx):
                    c_val += row_vals[l_idx] * B[l, j]
                
                if c_val != 0.0:
                    row_indices.append(i)
                    col_indices.append(j)
                    values.append(c_val)
    
    # Convert COO to CSR
    C_coo = coo_matrix(
        (values, (row_indices, col_indices)), shape=(m, k)
    )
    return C_coo.tocsr()


def convert_sparse_format(A: spmatrix, format_name: str) -> spmatrix:
    """
    Convert a sparse matrix to the requested format.

    Parameters:
    - A: Sparse matrix in any scipy.sparse format
    - format_name: Target format ('coo', 'csr', 'csc', 'lil')

    Returns:
    - Sparse matrix converted to the requested format
    """
    if format_name == "coo":
        return A.tocoo()
    elif format_name == "csr":
        return A.tocsr()
    elif format_name == "csc":
        return A.tocsc()
    elif format_name == "lil":
        return A.tolil()
    else:
        raise ValueError(f"Unknown format: {format_name}")


def spmm(
    A: spmatrix,
    B: np.ndarray,
    algorithm: str = "row-wise",
    format_name: str = "csr",
) -> csr_matrix:
    """
    Sparse matrix-matrix multiplication dispatcher.
    
    Parameters:
    - A: Sparse matrix in any scipy.sparse format (m × n)
    - B: Dense matrix (n × k)
    - algorithm: Algorithm to use ('row-wise', 'outer-product', 'blocked')
    - format_name: Format to convert A to ('csr', 'csc', 'coo', 'lil')
    
    Returns:
    - C: Sparse result matrix (m × k) in CSR format
    """
    if algorithm == "row-wise":
        A_conv = convert_sparse_format(A, "csr")
        return spmm_row_wise(A_conv, B)
    elif algorithm == "outer-product":
        A_conv = convert_sparse_format(A, "csc")
        return spmm_outer_product(A_conv, B)
    elif algorithm == "blocked":
        A_conv = convert_sparse_format(A, "csr")
        return spmm_blocked_inner_product(A_conv, B)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def validate_spmm(
    A: spmatrix, B: np.ndarray, C: csr_matrix, tol: float = 1e-10
) -> bool:
    """
    Validate SpMM result against dense matrix multiplication.
    
    Parameters:
    - A: Sparse matrix (m × n)
    - B: Dense matrix (n × k)
    - C: Computed sparse result (m × k)
    - tol: Tolerance for numerical comparison
    
    Returns:
    - True if result is correct within tolerance, False otherwise
    """
    # Compute dense reference
    A_dense = A.toarray()
    C_ref_dense = A_dense @ B
    C_dense = C.toarray()
    
    # Check shapes
    if C_dense.shape != C_ref_dense.shape:
        print(f"Shape mismatch: {C_dense.shape} vs {C_ref_dense.shape}")
        return False
    
    # Check values
    diff = np.abs(C_dense - C_ref_dense)
    max_diff = np.max(diff)
    
    if max_diff > tol:
        print(f"Validation failed: max difference = {max_diff} (tolerance = {tol})")
        return False
    
    return True


def measure_spmm_peak_memory_mb(
    A: spmatrix, B: np.ndarray, algorithm: str
) -> float:
    """
    Measure peak Python-level memory allocated by one SpMM call.

    This is run outside the timed benchmark loop so runtime measurements
    are not distorted by tracemalloc overhead.
    """
    gc.collect()
    tracemalloc.start()
    _ = spmm(A, B, algorithm=algorithm)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return float(peak / (1024 * 1024))


def benchmark_spmm_algorithm(
    A: spmatrix, B: np.ndarray, algorithm: str, repeats: int = 5
) -> Dict[str, float]:
    """
    Benchmark SpMM for a specific algorithm.
    
    Parameters:
    - A: Sparse matrix (m × n)
    - B: Dense matrix (n × k)
    - algorithm: Algorithm to benchmark ('row-wise', 'outer-product', 'blocked')
    - repeats: Number of repetitions
    
    Returns:
    - Dict with timing and memory statistics
    """
    # Run once for correctness validation
    C = spmm(A, B, algorithm=algorithm)
    if not validate_spmm(A, B, C):
        print(f"Warning: Validation failed for algorithm {algorithm}")
    
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        C = spmm(A, B, algorithm=algorithm)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    
    times = np.array(times)
    m, n = A.shape
    k = B.shape[1]
    nnz_a = A.nnz
    nnz_c = C.nnz
    
    # Compute metrics
    mean_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    
    # GFlops = 2 * nnz(A) * k / (time in seconds) / 1e9
    gflops = (2 * nnz_a * k) / (mean_time * 1e9) if mean_time > 0 else 0.0
    
    # Memory estimate: A storage + B storage + C storage
    memory_a_mb = (A.data.nbytes + A.indices.nbytes + A.indptr.nbytes) / (1024 * 1024)
    memory_b_mb = B.nbytes / (1024 * 1024)
    memory_c_mb = (C.data.nbytes + C.indices.nbytes + C.indptr.nbytes) / (1024 * 1024)
    
    return {
        "algorithm": algorithm,
        "mean_time": mean_time,
        "std_time": std_time,
        "min_time": min_time,
        "gflops": gflops,
        "nnz_a": nnz_a,
        "nnz_c": nnz_c,
        "memory_a_mb": memory_a_mb,
        "memory_b_mb": memory_b_mb,
        "memory_c_mb": memory_c_mb,
    }


def benchmark_spmm_all_algorithms(
    A: spmatrix, B: np.ndarray, repeats: int = 5
) -> Dict[str, Dict[str, float]]:
    """
    Benchmark SpMM for all three algorithms.
    
    Parameters:
    - A: Sparse matrix (m × n)
    - B: Dense matrix (n × k)
    - repeats: Number of repetitions per algorithm
    
    Returns:
    - Dict mapping algorithm name to performance metrics
    """
    results = {}
    for algo in ["row-wise", "outer-product", "blocked"]:
        print(f"  Benchmarking {algo}...", end=" ", flush=True)
        results[algo] = benchmark_spmm_algorithm(A, B, algo, repeats=repeats)
        print(f"✓ ({results[algo]['gflops']:.2f} GFlop/s)")
    
    return results


# ============================================================================
# SPARSE B SUPPORT: Algorithms for sparse matrix-matrix multiplication
# ============================================================================

def spmm_row_wise_sparse_b(A_csr: csr_matrix, B_csr: csr_matrix) -> csr_matrix:
    """
    Row-wise SpMM with sparse B: C = A @ B (both sparse).
    
    Uses efficient sparse lookup via CSC format for B.
    
    Parameters:
    - A_csr: Sparse matrix in CSR format (m × n)
    - B_csr: Sparse matrix in CSR format (n × k)
    
    Returns:
    - C: Sparse result matrix (m × k) in CSR format
    
    Complexity: O(nnz(A) × average_nnz_per_col(B))
    """
    m, n = A_csr.shape
    k = B_csr.shape[1]
    
    # Convert B to CSC for efficient column access
    B_csc = B_csr.tocsc()
    
    row_indices = []
    col_indices = []
    values = []
    
    # Create dict for O(1) lookup: B_csc.indices and values by (row, col)
    B_dict = {}
    for col_j in range(k):
        col_start = B_csc.indptr[col_j]
        col_end = B_csc.indptr[col_j + 1]
        for idx in range(col_start, col_end):
            row_i = B_csc.indices[idx]
            val = B_csc.data[idx]
            B_dict[(row_i, col_j)] = val
    
    # Process each row of A
    for i in range(m):
        row_start = A_csr.indptr[i]
        row_end = A_csr.indptr[i + 1]
        
        if row_start == row_end:
            continue
        
        col_idx = A_csr.indices[row_start:row_end]
        row_vals = A_csr.data[row_start:row_end]
        
        # Compute C[i, :] = A[i, :] @ B
        for j in range(k):
            c_val = 0.0
            
            # Accumulate: C[i,j] = sum over l where both A[i,l] and B[l,j] are non-zero
            for l_idx, l in enumerate(col_idx):
                if (l, j) in B_dict:
                    c_val += row_vals[l_idx] * B_dict[(l, j)]
            
            if c_val != 0.0:
                row_indices.append(i)
                col_indices.append(j)
                values.append(c_val)
    
    C_coo = coo_matrix(
        (values, (row_indices, col_indices)), shape=(m, k)
    )
    return C_coo.tocsr()


def spmm_outer_product_sparse_b(A_csc: csc_matrix, B_csr: csr_matrix) -> csr_matrix:
    """
    Outer-product SpMM with sparse B: accumulate rank-1 updates.
    
    C += A[:, j] * B[j, :]^T for each non-zero row j of B (in CSR format).
    
    Parameters:
    - A_csc: Sparse matrix in CSC format (m × n)
    - B_csr: Sparse matrix in CSR format (n × k)
    
    Returns:
    - C: Sparse result matrix (m × k) in CSR format
    
    Complexity: O(nnz(A) × nnz(B) / n)
    """
    m, n = A_csc.shape
    k = B_csr.shape[1]
    
    C_lil = lil_matrix((m, k), dtype=B_csr.dtype)
    
    # For each row j of B
    for j in range(n):
        # Get column j from A
        col_start_a = A_csc.indptr[j]
        col_end_a = A_csc.indptr[j + 1]
        
        if col_start_a == col_end_a:
            continue  # Column is empty
        
        a_rows = A_csc.indices[col_start_a:col_end_a]
        a_vals = A_csc.data[col_start_a:col_end_a]
        
        # Get row j from B
        row_start_b = B_csr.indptr[j]
        row_end_b = B_csr.indptr[j + 1]
        
        if row_start_b == row_end_b:
            continue  # Row is empty
        
        b_cols = B_csr.indices[row_start_b:row_end_b]
        b_vals = B_csr.data[row_start_b:row_end_b]
        
        # Rank-1 update: C[i, jj] += A[i, j] * B[j, jj]
        for i_idx, i in enumerate(a_rows):
            a_val = a_vals[i_idx]
            for jj_idx, jj in enumerate(b_cols):
                C_lil[i, jj] += a_val * b_vals[jj_idx]
    
    return C_lil.tocsr()


def spmm_blocked_inner_product_sparse_b(
    A_csr: csr_matrix, B_csr: csr_matrix, block_k: int = 32
) -> csr_matrix:
    """
    Blocked inner-product SpMM with sparse B.
    
    Process B in column blocks for cache efficiency.
    Uses dict-based lookup for fast (row, col) access.
    
    Parameters:
    - A_csr: Sparse matrix in CSR format (m × n)
    - B_csr: Sparse matrix in CSR format (n × k)
    - block_k: Block size for column dimension
    
    Returns:
    - C: Sparse result matrix (m × k) in CSR format
    
    Complexity: O(nnz(A) × average_nnz_per_col(B))
    """
    m, n = A_csr.shape
    k = B_csr.shape[1]
    
    # Convert B to CSC for column access
    B_csc = B_csr.tocsc()
    
    # Create fast lookup dict for B
    B_dict = {}
    for col_j in range(k):
        col_start = B_csc.indptr[col_j]
        col_end = B_csc.indptr[col_j + 1]
        for idx in range(col_start, col_end):
            row_i = B_csc.indices[idx]
            val = B_csc.data[idx]
            B_dict[(row_i, col_j)] = val
    
    row_indices = []
    col_indices = []
    values = []
    
    # Process B in column blocks
    for bk_start in range(0, k, block_k):
        bk_end = min(bk_start + block_k, k)
        
        # Process each row of A
        for i in range(m):
            row_start = A_csr.indptr[i]
            row_end = A_csr.indptr[i + 1]
            
            if row_start == row_end:
                continue
            
            col_idx = A_csr.indices[row_start:row_end]
            row_vals = A_csr.data[row_start:row_end]
            
            # Process columns in [bk_start, bk_end)
            for jj in range(bk_start, bk_end):
                c_val = 0.0
                
                # Accumulate C[i, jj]
                for l_idx, l in enumerate(col_idx):
                    if (l, jj) in B_dict:
                        c_val += row_vals[l_idx] * B_dict[(l, jj)]
                
                if c_val != 0.0:
                    row_indices.append(i)
                    col_indices.append(jj)
                    values.append(c_val)
    
    C_coo = coo_matrix(
        (values, (row_indices, col_indices)), shape=(m, k)
    )
    return C_coo.tocsr()


def spmm_scipy_builtin(A: spmatrix, B: spmatrix) -> csr_matrix:
    """
    SciPy built-in sparse matrix multiplication for reference/comparison.
    
    Parameters:
    - A: Sparse matrix (m × n)
    - B: Sparse matrix (n × k)
    
    Returns:
    - C: Sparse result matrix (m × k) in CSR format
    
    This uses scipy.sparse's optimized @ operator.
    """
    A_csr = A.tocsr() if not isinstance(A, csr_matrix) else A
    B_csr = B.tocsr() if not isinstance(B, csr_matrix) else B
    return (A_csr @ B_csr).tocsr()


def benchmark_spmm_algorithm_sparse_b(
    A: spmatrix,
    B: spmatrix,
    algorithm: str = "row-wise",
    repeats: int = 5,
) -> Dict[str, float]:
    """
    Benchmark a single SpMM algorithm with sparse B.
    
    Parameters:
    - A: Sparse matrix (m × n)
    - B: Sparse matrix (n × k)
    - algorithm: Algorithm to benchmark
    - repeats: Number of repetitions
    
    Returns:
    - Dict with performance metrics
    """
    A_csr = A.tocsr() if not isinstance(A, csr_matrix) else A
    B_csr = B.tocsr() if not isinstance(B, csr_matrix) else B
    
    # Warm-up run
    if algorithm == "row-wise":
        _ = spmm_row_wise_sparse_b(A_csr, B_csr)
    elif algorithm == "outer-product":
        A_csc = A_csr.tocsc()
        _ = spmm_outer_product_sparse_b(A_csc, B_csr)
    elif algorithm == "blocked":
        _ = spmm_blocked_inner_product_sparse_b(A_csr, B_csr)
    elif algorithm == "scipy":
        _ = spmm_scipy_builtin(A_csr, B_csr)
    
    # Timed runs
    times = []
    tracemalloc.start()
    
    for _ in range(repeats):
        gc.collect()
        start = time.perf_counter()
        
        if algorithm == "row-wise":
            C = spmm_row_wise_sparse_b(A_csr, B_csr)
        elif algorithm == "outer-product":
            A_csc = A_csr.tocsc()
            C = spmm_outer_product_sparse_b(A_csc, B_csr)
        elif algorithm == "blocked":
            C = spmm_blocked_inner_product_sparse_b(A_csr, B_csr)
        elif algorithm == "scipy":
            C = spmm_scipy_builtin(A_csr, B_csr)
        
        end = time.perf_counter()
        times.append(end - start)
    
    tracemalloc.stop()
    
    mean_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    
    # GFlops calculation: 2 * nnz(A) * nnz(B) / n (approximation for sparse-sparse)
    nnz_a = A_csr.nnz
    nnz_b = B_csr.nnz
    nnz_c = C.nnz
    flops = 2 * nnz_a * nnz_b / A_csr.shape[1] if A_csr.shape[1] > 0 else 0
    gflops = flops / (mean_time * 1e9) if mean_time > 0 else 0
    
    memory_a_mb = (A_csr.data.nbytes + A_csr.indices.nbytes + A_csr.indptr.nbytes) / (1024 * 1024)
    memory_b_mb = (B_csr.data.nbytes + B_csr.indices.nbytes + B_csr.indptr.nbytes) / (1024 * 1024)
    memory_c_mb = (C.data.nbytes + C.indices.nbytes + C.indptr.nbytes) / (1024 * 1024)
    
    return {
        "algorithm": algorithm,
        "mean_time": mean_time,
        "std_time": std_time,
        "min_time": min_time,
        "gflops": gflops,
        "nnz_a": nnz_a,
        "nnz_b": nnz_b,
        "nnz_c": nnz_c,
        "memory_a_mb": memory_a_mb,
        "memory_b_mb": memory_b_mb,
        "memory_c_mb": memory_c_mb,
    }


def benchmark_spmm_all_algorithms_sparse_b(
    A: spmatrix, B: spmatrix, repeats: int = 5
) -> Dict[str, Dict[str, float]]:
    """
    Benchmark all SpMM algorithms with sparse B (including scipy).
    
    Parameters:
    - A: Sparse matrix (m × n)
    - B: Sparse matrix (n × k)
    - repeats: Number of repetitions per algorithm
    
    Returns:
    - Dict mapping algorithm name to performance metrics
    """
    results = {}
    for algo in ["row-wise", "outer-product", "blocked", "scipy"]:
        print(f"  Benchmarking {algo}...", end=" ", flush=True)
        results[algo] = benchmark_spmm_algorithm_sparse_b(A, B, algo, repeats=repeats)
        print(f"✓ ({results[algo]['gflops']:.2f} GFlop/s)")
    
    return results
