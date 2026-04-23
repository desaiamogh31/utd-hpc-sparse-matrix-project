from __future__ import annotations
import argparse
import os
import time
import warnings
from typing import Callable, Dict, Tuple
import numpy as np
import pandas as pd

"""
Benchmark 2D Laplacian sparse matrix assembly across COO, CSR, LIL, and CSC.

The 2D Laplacian on an n×n grid is represented as an (n²)×(n²) matrix with
the 5-point stencil pattern (center and 4 neighbors).

Usage:
    python laplacian_matrix.py --n 100 --repeats 3
"""

from scipy.sparse import (
    SparseEfficiencyWarning,
    coo_matrix,
    csc_matrix,
    csr_matrix,
    lil_matrix,
    spmatrix,
)


def generate_laplacian_entries(n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate entries for the 2D Laplacian matrix on an n×n grid.
    
    Uses the 5-point stencil:
    - Diagonal: -4
    - Neighbors (up, down, left, right): +1
    
    Boundary conditions: Dirichlet (no entries outside domain)
    
    Parameters:
    - n: Grid dimension (n×n grid yields n²×n² matrix)
    
    Returns:
    - rows: Row indices
    - cols: Column indices
    - vals: Values
    """
    rows_list = []
    cols_list = []
    vals_list = []
    
    # Iterate through all grid points
    for i in range(n):
        for j in range(n):
            idx = i * n + j  # Linear index in the matrix
            
            # Diagonal entry: -4
            rows_list.append(idx)
            cols_list.append(idx)
            vals_list.append(-4.0)
            
            # Up neighbor (i-1, j)
            if i > 0:
                neighbor_idx = (i - 1) * n + j
                rows_list.append(idx)
                cols_list.append(neighbor_idx)
                vals_list.append(1.0)
            
            # Down neighbor (i+1, j)
            if i < n - 1:
                neighbor_idx = (i + 1) * n + j
                rows_list.append(idx)
                cols_list.append(neighbor_idx)
                vals_list.append(1.0)
            
            # Left neighbor (i, j-1)
            if j > 0:
                neighbor_idx = i * n + (j - 1)
                rows_list.append(idx)
                cols_list.append(neighbor_idx)
                vals_list.append(1.0)
            
            # Right neighbor (i, j+1)
            if j < n - 1:
                neighbor_idx = i * n + (j + 1)
                rows_list.append(idx)
                cols_list.append(neighbor_idx)
                vals_list.append(1.0)
    
    return np.array(rows_list), np.array(cols_list), np.array(vals_list)


def assemble_coo(
    n: int, rows: np.ndarray, cols: np.ndarray, vals: np.ndarray
) -> spmatrix:
    """
    Assemble Laplacian matrix directly in COO format.
    """
    mat_size = n * n
    a = coo_matrix((vals, (rows, cols)), shape=(mat_size, mat_size))
    return a


def assemble_lil(
    n: int, rows: np.ndarray, cols: np.ndarray, vals: np.ndarray
) -> spmatrix:
    """
    Assemble Laplacian matrix via incremental updates in LIL.
    """
    mat_size = n * n
    a = lil_matrix((mat_size, mat_size), dtype=np.float64)
    for r, c, v in zip(rows, cols, vals):
        a[r, c] += v
    return a


def assemble_csr(
    n: int, rows: np.ndarray, cols: np.ndarray, vals: np.ndarray
) -> spmatrix:
    """
    Assemble Laplacian matrix via incremental updates in CSR.
    """
    mat_size = n * n
    a = csr_matrix((mat_size, mat_size), dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SparseEfficiencyWarning)
        for r, c, v in zip(rows, cols, vals):
            a[r, c] += v
    return a


def assemble_csc(
    n: int, rows: np.ndarray, cols: np.ndarray, vals: np.ndarray
) -> spmatrix:
    """
    Assemble Laplacian matrix via incremental updates in CSC.
    """
    mat_size = n * n
    a = csc_matrix((mat_size, mat_size), dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SparseEfficiencyWarning)
        for r, c, v in zip(rows, cols, vals):
            a[r, c] += v
    return a


def benchmark(
    fn: Callable[[int, np.ndarray, np.ndarray, np.ndarray], spmatrix],
    n: int,
    rows: np.ndarray,
    cols: np.ndarray,
    vals: np.ndarray,
    repeats: int,
) -> Dict[str, float]:
    """
    Benchmark the given assembly function and return timing and matrix info.
    
    Parameters:
    - fn: Assembly function to benchmark.
    - n: Grid dimension.
    - rows, cols, vals: Laplacian entries.
    - repeats: Number of benchmark repetitions.
    """
    times = []
    last_mat = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        last_mat = fn(n, rows, cols, vals)
        times.append(time.perf_counter() - t0)

    assert last_mat is not None
    return {
        "avg_s": float(np.mean(times)),
        "min_s": float(np.min(times)),
        "max_s": float(np.max(times)),
        "nnz": int(last_mat.nnz),
        "matrix": last_mat,
    }


def matrices_equal(a: spmatrix, b: spmatrix, tol: float = 1e-12) -> bool:
    """Check if two sparse matrices are approximately equal."""
    d = (a.tocsr() - b.tocsr()).tocoo()
    if d.nnz == 0:
        return True
    return np.max(np.abs(d.data)) <= tol


def benchmark_conversions(
    source_matrices: Dict[str, spmatrix], repeats: int
) -> Dict[str, Dict[str, float]]:
    """
    Benchmark conversions from COO/LIL to CSR/CSC formats.
    
    Parameters:
    - source_matrices: Dict mapping format name (COO/LIL) to the assembled matrix
    - repeats: Number of benchmark repetitions
    
    Returns:
    - Dict mapping 'COO->CSR', 'COO->CSC', 'LIL->CSR', 'LIL->CSC' to timing info
    """
    conversion_pairs = [
        ("COO", "CSR"),
        ("COO", "CSC"),
        ("LIL", "CSR"),
        ("LIL", "CSC"),
    ]
    
    results = {}
    
    for src_format, tgt_format in conversion_pairs:
        if src_format not in source_matrices:
            continue
            
        src_mat = source_matrices[src_format]
        times = []
        
        for _ in range(repeats):
            t0 = time.perf_counter()
            if tgt_format == "CSR":
                _ = src_mat.tocsr()
            elif tgt_format == "CSC":
                _ = src_mat.tocsc()
            times.append(time.perf_counter() - t0)
        
        conversion_key = f"{src_format}->{tgt_format}"
        results[conversion_key] = {
            "avg_s": float(np.mean(times)),
            "min_s": float(np.min(times)),
            "max_s": float(np.max(times)),
        }
    
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark 2D Laplacian sparse matrix assembly for COO/CSR/LIL/CSC."
    )
    parser.add_argument(
        "--n",
        type=int,
        default=100,
        help="Grid dimension (n×n grid yields n²×n² matrix).",
    )
    parser.add_argument("--repeats", type=int, default=3, help="Benchmark repetitions.")
    parser.add_argument("--outdir", type=str, default=".", help="Output directory for results.")
    args = parser.parse_args()

    n = args.n
    rows, cols, vals = generate_laplacian_entries(n)
    mat_size = n * n
    
    # Ensure output directory exists
    os.makedirs(args.outdir, exist_ok=True)

    builders = {
        "COO": assemble_coo,
        "LIL": assemble_lil,
        "CSR": assemble_csr,
        "CSC": assemble_csc,
    }

    results = {}
    for name, fn in builders.items():
        results[name] = benchmark(fn, n, rows, cols, vals, args.repeats)

    # Correctness: compare all against COO result.
    ref = results["COO"]["matrix"]
    print(f"\n2D Laplacian assembly benchmark (grid n={n}, matrix size {mat_size}×{mat_size})")
    print("-" * 74)
    print(
        f"{'Format':<8}{'Avg(s)':>12}{'Min(s)':>12}{'Max(s)':>12}{'NNZ':>12}{'EqualToCOO':>18}"
    )
    print("-" * 74)
    for name in ["COO", "LIL", "CSR", "CSC"]:
        r = results[name]
        ok = matrices_equal(r["matrix"], ref)
        print(
            f"{name:<8}{r['avg_s']:>12.6f}{r['min_s']:>12.6f}{r['max_s']:>12.6f}"
            f"{r['nnz']:>12d}{str(ok):>18}"
        )
    
    table = pd.DataFrame(
        [
            {
                "Format": name,
                "N" : mat_size,
                "Avg(s)": results[name]["avg_s"],
                "Min(s)": results[name]["min_s"],
                "Max(s)": results[name]["max_s"],
                "NNZ": results[name]["nnz"],
                "EqualToCOO": matrices_equal(results[name]["matrix"], ref),
            }
            for name in ["COO", "LIL", "CSR", "CSC"]
        ]
    )

    # Ensure output directory exists
    os.makedirs(args.outdir, exist_ok=True)
    
    output_path = f"{args.outdir}/laplacian_matrix_benchmark.csv"
    table.to_csv(output_path, index=False)
    print(f"\nSaved table to {output_path}")
    print(table.to_string(index=False))
    print("-" * 74)
    print("Note: Direct incremental assembly is usually fastest in LIL/COO,")
    print("while direct CSR/CSC insertion is typically much slower.")
    
    # Benchmark conversions from COO/LIL to CSR/CSC
    print("\n" + "=" * 74)
    print("Format Conversion Benchmarks")
    print("=" * 74)
    
    source_matrices = {
        "COO": results["COO"]["matrix"],
        "LIL": results["LIL"]["matrix"],
    }
    
    conversion_results = benchmark_conversions(source_matrices, args.repeats)
    
    print(f"\n{'Conversion':<15}{'Avg(s)':>12}{'Min(s)':>12}{'Max(s)':>12}")
    print("-" * 51)
    for name in sorted(conversion_results.keys()):
        r = conversion_results[name]
        print(
            f"{name:<15}{r['avg_s']:>12.6f}{r['min_s']:>12.6f}{r['max_s']:>12.6f}"
        )
    
    conversion_table = pd.DataFrame(
        [
            {
                "Conversion": name,
                "Avg(s)": conversion_results[name]["avg_s"],
                "Min(s)": conversion_results[name]["min_s"],
                "Max(s)": conversion_results[name]["max_s"],
            }
            for name in sorted(conversion_results.keys())
        ]
    )
    
    conversion_output_path = f"{args.outdir}/laplacian_conversion_benchmark.csv"
    conversion_table.to_csv(conversion_output_path, index=False)
    print(f"\nSaved conversion table to {conversion_output_path}")
    print(conversion_table.to_string(index=False))
    print("-" * 74)


if __name__ == "__main__":
    main()
