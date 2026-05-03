"""
Serial SpMV (Sparse Matrix-Vector multiplication) helpers.

Provides SpMV operations for COO, CSR, CSC, and LIL sparse matrix formats
using SciPy's built-in sparse kernels, with timing and memory profiling
for benchmarking.
"""

from __future__ import annotations
import gc
import time
import tracemalloc
from typing import Dict
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix, lil_matrix, spmatrix


def spmv_coo(A_coo: coo_matrix, x: np.ndarray) -> np.ndarray:
    """
    Sparse matrix-vector multiplication using COO (Coordinate) format.
    
    Parameters:
    - A_coo: Sparse matrix in COO format (m × n)
    - x: Dense vector (n,)
    
    Returns:
    - y: Dense result vector (m,)
    """
    return A_coo @ x


def spmv_csr(A_csr: csr_matrix, x: np.ndarray) -> np.ndarray:
    """
    Sparse matrix-vector multiplication using CSR (Compressed Sparse Row) format.
    
    Parameters:
    - A_csr: Sparse matrix in CSR format (m × n)
    - x: Dense vector (n,)
    
    Returns:
    - y: Dense result vector (m,)
    """
    return A_csr @ x


def spmv_csc(A_csc: csc_matrix, x: np.ndarray) -> np.ndarray:
    """
    Sparse matrix-vector multiplication using CSC (Compressed Sparse Column) format.
    
    Parameters:
    - A_csc: Sparse matrix in CSC format (m × n)
    - x: Dense vector (n,)
    
    Returns:
    - y: Dense result vector (m,)
    """
    return A_csc @ x


def spmv_lil(A_lil: lil_matrix, x: np.ndarray) -> np.ndarray:
    """
    Sparse matrix-vector multiplication using LIL (List of Lists) format.
    
    Parameters:
    - A_lil: Sparse matrix in LIL format (m × n)
    - x: Dense vector (n,)
    
    Returns:
    - y: Dense result vector (m,)
    """
    return A_lil @ x


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


def spmv_preformatted(A: spmatrix, x: np.ndarray, format_name: str) -> np.ndarray:
    """
    Run SpMV assuming the matrix is already stored in the requested format.
    """
    if format_name == "coo":
        return spmv_coo(A, x)
    elif format_name == "csr":
        return spmv_csr(A, x)
    elif format_name == "csc":
        return spmv_csc(A, x)
    elif format_name == "lil":
        return spmv_lil(A, x)
    else:
        raise ValueError(f"Unknown format: {format_name}")


def spmv(A: spmatrix, x: np.ndarray, format_name: str = "csr") -> np.ndarray:
    """
    Sparse matrix-vector multiplication dispatcher.
    
    Parameters:
    - A: Sparse matrix in any scipy.sparse format
    - x: Dense vector
    - format_name: Format to use ('coo', 'csr', 'csc', 'lil')
    
    Returns:
    - y: Dense result vector
    """
    A_conv = convert_sparse_format(A, format_name)
    return spmv_preformatted(A_conv, x, format_name)


def validate_spmv(A: spmatrix, x: np.ndarray, y: np.ndarray, tol: float = 1e-12) -> bool:
    """
    Validate SpMV result against dense matrix-vector multiplication.
    
    Parameters:
    - A: Sparse matrix
    - x: Input vector
    - y: Computed result vector
    - tol: Tolerance for numerical comparison
    
    Returns:
    - True if result is correct within tolerance, False otherwise
    """
    y_ref = A @ x
    
    # Check shapes
    if y.shape != y_ref.shape:
        print(f"Shape mismatch: {y.shape} vs {y_ref.shape}")
        return False
    
    # Check values
    diff = np.abs(y - y_ref)
    max_diff = np.max(diff)
    
    if max_diff > tol:
        print(f"Validation failed: max difference = {max_diff} (tolerance = {tol})")
        return False
    
    return True


def measure_spmv_peak_memory_mb(
    A: spmatrix, x: np.ndarray, format_name: str
) -> float:
    """
    Measure peak Python-level memory allocated by one SpMV call.

    This is run outside the timed benchmark loop so runtime measurements
    are not distorted by tracemalloc overhead.
    """
    gc.collect()
    tracemalloc.start()
    _ = spmv_preformatted(A, x, format_name)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return float(peak / (1024 * 1024))


def benchmark_spmv_format(
    A: spmatrix, x: np.ndarray, format_name: str, repeats: int = 5
) -> Dict[str, float]:
    """
    Benchmark SpMV for a specific format.
    
    Parameters:
    - A: Sparse matrix
    - x: Input vector
    - format_name: Format to benchmark ('coo', 'csr', 'csc', 'lil')
    - repeats: Number of repetitions
    
    Returns:
    - Dict with timing and memory statistics
    """
    # Convert once before timing so the measured runtime is only SpMV.
    A_bench = convert_sparse_format(A, format_name)
    y = spmv_preformatted(A_bench, x, format_name)
    if not validate_spmv(A, x, y):
        print(f"Warning: Validation failed for format {format_name}")

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        _ = spmv_preformatted(A_bench, x, format_name)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)

    avg_time = float(np.mean(times))
    min_time = float(np.min(times))
    max_time = float(np.max(times))
    peak_memory_mb = measure_spmv_peak_memory_mb(A_bench, x, format_name)
    
    # GFlop/s = 2 * nnz / time / 1e9
    gflops = (2.0 * A.nnz / avg_time / 1e9) if avg_time > 0 else 0.0
    
    return {
        "avg_time_s": avg_time,
        "min_time_s": min_time,
        "max_time_s": max_time,
        "memory_mb": peak_memory_mb,
        "gflops": gflops,
    }
