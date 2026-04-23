from __future__ import annotations
import argparse
import os
import time
import warnings
from typing import Callable, Dict, List, Tuple
import numpy as np
import pandas as pd

"""
Benchmark symmetric sparse matrix assembly across COO, CSR, LIL, and CSC.

Two modes available:
1. Single-N benchmark: Benchmark all four formats for a specific matrix size.
   Usage: python symmetric_matrix.py --mode single --n 5000 --upper-nnz 40000 --repeats 3 --seed 42

2. Scaling benchmark: Compare COO and LIL runtimes across different matrix sizes with NNZ scaled by ratio.
   Usage: python symmetric_matrix.py --mode scaling --matrix-sizes 1000 1500 2000 2500 --nnz-ratio 5.0 --repeats 3
"""
''' SAMPLE USAGE:
# Single matrix benchmark
python symmetric_matrix.py --mode single --n 2000 --upper-nnz 10000 --repeats 3 --outdir results

# Scaling with custom sizes and ratio
python symmetric_matrix.py --mode scaling --matrix-sizes 1000 2000 3000 4000 --nnz-ratio 8.0 --repeats 2 --outdir results
'''

from scipy.sparse import (
    SparseEfficiencyWarning,
    coo_matrix,
    csc_matrix,
    csr_matrix,
    lil_matrix,
    spmatrix,
)
import matplotlib.pyplot as plt


def generate_upper_tri_entries(
    n: int, upper_nnz: int, seed: int = 0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate random entries for the upper triangle (including diagonal).
    Duplicates are allowed and intentionally kept to model realistic assembly.

    Parameters:
    - n: Matrix dimension.
    - upper_nnz: Number of upper-triangle entries to generate (with duplicates).
    - seed: Random seed for reproducibility.

    Returns:
    - i: Row indices of upper-triangle entries.
    - j: Column indices of upper-triangle entries.
    - v: Values of upper-triangle entries.
    """
    rng = np.random.default_rng(seed)
    i = rng.integers(0, n, size=upper_nnz, endpoint=False)
    j = rng.integers(0, n, size=upper_nnz, endpoint=False)

    # Force (i, j) into upper triangle so i <= j.
    lo = np.minimum(i, j)
    hi = np.maximum(i, j)
    vals = rng.standard_normal(upper_nnz)

    return lo, hi, vals

def assemble_coo(n: int, i: np.ndarray, j: np.ndarray, v: np.ndarray) -> spmatrix:
    """
    Assemble symmetric matrix directly in COO.
    """
    off_diag = i != j
    rows = np.concatenate([i, j[off_diag]])
    cols = np.concatenate([j, i[off_diag]])
    data = np.concatenate([v, v[off_diag]])

    a = coo_matrix((data, (rows, cols)), shape=(n, n))
    a.sum_duplicates()
    return a


def assemble_lil(n: int, i: np.ndarray, j: np.ndarray, v: np.ndarray) -> spmatrix:
    """
    Assemble symmetric matrix via incremental updates in LIL.
    """
    a = lil_matrix((n, n), dtype=np.float64)
    for r, c, x in zip(i, j, v):
        a[r, c] += x
        if r != c:
            a[c, r] += x
    return a


def assemble_csr(n: int, i: np.ndarray, j: np.ndarray, v: np.ndarray) -> spmatrix:
    """
    Assemble symmetric matrix via incremental updates in CSR (typically slow).
    Included for direct structure comparison.
    """
    a = csr_matrix((n, n), dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SparseEfficiencyWarning)
        for r, c, x in zip(i, j, v):
            a[r, c] += x
            if r != c:
                a[c, r] += x
    return a


def assemble_csc(n: int, i: np.ndarray, j: np.ndarray, v: np.ndarray) -> spmatrix:
    """
    Assemble symmetric matrix via incremental updates in CSC (typically slow).
    Included for direct structure comparison.
    """
    a = csc_matrix((n, n), dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SparseEfficiencyWarning)
        for r, c, x in zip(i, j, v):
            a[r, c] += x
            if r != c:
                a[c, r] += x
    return a


def benchmark(
    fn: Callable[[int, np.ndarray, np.ndarray, np.ndarray], spmatrix],
    n: int,
    i: np.ndarray,
    j: np.ndarray,
    v: np.ndarray,
    repeats: int
) -> Dict[str, float]:
    '''Benchmark the given assembly function and return timing and matrix info. 
    Parameters:
    - fn: Assembly function to benchmark.
    - n: Matrix dimension.
    - i, j, v: Upper-triangle entries to assemble.
    - repeats: Number of benchmark repetitions.
    - outdir: Output directory for results.'''
    times = []
    last_mat = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        last_mat = fn(n, i, j, v)
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
    d = (a.tocsr() - b.tocsr()).tocoo()
    if d.nnz == 0:
        return True
    return np.max(np.abs(d.data)) <= tol


def benchmark_single_n(
    n: int, upper_nnz: int, repeats: int, seed: int, outdir: str = "."
) -> None:
    """
    Benchmark assembly of a single symmetric matrix with given parameters.
    Produces a table with runtimes for COO, CSR, LIL, and CSC formats.
    
    Parameters:
    - n: Matrix dimension.
    - upper_nnz: Number of generated upper-triangle entries.
    - repeats: Number of benchmark repetitions.
    - seed: Random seed for reproducibility.
    - outdir: Output directory for results.
    """
    if upper_nnz > (n * (n + 1)) // 2:
        raise ValueError(
            f"upper-nnz {upper_nnz} exceeds total upper-triangle entries "
            f"{(n * (n + 1)) // 2} for n={n}"
        )
    
    i, j, v = generate_upper_tri_entries(n, upper_nnz, seed)

    builders = {
        "COO": assemble_coo,
        "LIL": assemble_lil,
        "CSR": assemble_csr,
        "CSC": assemble_csc,
    }

    results = {}
    for name, fn in builders.items():
        results[name] = benchmark(fn, n, i, j, v, repeats)

    # Correctness: compare all against COO result.
    ref = results["COO"]["matrix"]
    print(f"\nSymmetric assembly benchmark (n={n}, upper_nnz={upper_nnz})")
    print("-" * 74)
    print(f"{'Format':<8}{'Avg(s)':>12}{'Min(s)':>12}{'Max(s)':>12}{'NNZ':>12}{'EqualToCOO':>18}")
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
    os.makedirs(outdir, exist_ok=True)
    
    output_path = f"{outdir}/symmetric_matrix_benchmark.csv"
    table.to_csv(output_path, index=False)
    print(f"Saved table to {output_path}")
    print(table.to_string(index=False))
    print("-" * 74)
    print("Note: Direct incremental assembly is usually fastest in LIL/COO,")
    print("while direct CSR/CSC insertion is typically much slower.")


def benchmark_scaling(
    matrix_sizes: List[int], nnz_ratio: float, repeats: int, seed: int, outdir: str = "."
) -> None:
    """
    Benchmark assembly runtimes for COO and LIL matrices across different sizes.
    Generates a plot of runtime vs matrix dimension and saves timing data.
    
    Parameters:
    - matrix_sizes: List of matrix dimensions to benchmark.
    - nnz_ratio: Ratio of nnz to n (upper_nnz = int(n * nnz_ratio) for each n).
    - repeats: Number of benchmark repetitions per matrix size.
    - seed: Random seed for reproducibility.
    """
    sizes = matrix_sizes
    coo_times = []
    lil_times = []
    nnz_per_size = []
    
    print(f"\nScaling benchmark: COO vs LIL")
    print(f"Matrix sizes: {sizes}")
    print(f"NNZ ratio (nnz/n): {nnz_ratio}")
    print(f"Repeats per size: {repeats}")
    print("-" * 70)
    print(f"{'N':<8}{'Upper NNZ':<12}{'COO Avg(s)':>15}{'LIL Avg(s)':>15}")
    print("-" * 70)
    
    for n in sizes:
        # Calculate upper_nnz based on ratio, rounded down
        upper_nnz = int(n * nnz_ratio)
        nnz_per_size.append(upper_nnz)
        
        i, j, v = generate_upper_tri_entries(n, upper_nnz, seed)
        
        # Benchmark COO
        coo_result = benchmark(assemble_coo, n, i, j, v, repeats)
        coo_times.append(coo_result["avg_s"])
        
        # Benchmark LIL
        lil_result = benchmark(assemble_lil, n, i, j, v, repeats)
        lil_times.append(lil_result["avg_s"])
        
        print(f"{n:<8}{upper_nnz:<12}{coo_result['avg_s']:>15.6f}{lil_result['avg_s']:>15.6f}")
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, coo_times, marker='o', label='COO', linewidth=2, markersize=8)
    plt.plot(sizes, lil_times, marker='s', label='LIL', linewidth=2, markersize=8)
    plt.xlabel('Matrix Dimension (n)', fontsize=12)
    plt.ylabel('Assembly Time (seconds)', fontsize=12)
    plt.xscale('log')
    plt.yscale('log')
    plt.title(f'Symmetric Matrix Assembly Time vs Size\n(nnz_ratio={nnz_ratio}, repeats={repeats})', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Ensure output directory exists
    os.makedirs(outdir, exist_ok=True)
    
    plot_path = f"{outdir}/symmetric_matrix_scaling.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to {plot_path}")
    
    # Save timing data to CSV
    scaling_table = pd.DataFrame({
        "N": sizes,
        "Upper_NNZ": nnz_per_size,
        "COO_Avg_s": coo_times,
        "LIL_Avg_s": lil_times,
    })
    
    csv_path = f"{outdir}/symmetric_matrix_scaling.csv"
    scaling_table.to_csv(csv_path, index=False)
    print(f"Saved timing data to {csv_path}")
    print(scaling_table.to_string(index=False))
    print("-" * 70)
    print(scaling_table.to_string(index=False))
    print("-" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark symmetric sparse matrix assembly."
    )
    
    # Mode selection
    parser.add_argument(
        "--mode",
        type=str,
        default="single",
        choices=["single", "scaling"],
        help="Benchmark mode: 'single' for a single matrix size, 'scaling' for multiple sizes.",
    )
    
    # Arguments for single-N benchmark
    parser.add_argument("--n", type=int, default=2000, help="Matrix dimension (used in 'single' mode).")
    parser.add_argument(
        "--upper-nnz",
        type=int,
        default=20000,
        help="Number of generated upper-triangle entries (with duplicates).",
    )
    parser.add_argument("--repeats", type=int, default=3, help="Benchmark repetitions.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    
    # Arguments for scaling benchmark
    parser.add_argument(
        "--matrix-sizes",
        type=int,
        nargs='+',
        default=[1000, 1500, 2000, 2500, 3000],
        help="List of matrix dimensions to benchmark (used in 'scaling' mode).",
    )
    parser.add_argument(
        "--nnz-ratio",
        type=float,
        default=5.0,
        help="Ratio of nnz to n for each matrix size (upper_nnz = int(n * nnz_ratio), used in 'scaling' mode).",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=".",
        help="Output directory for plots and CSV files (used in 'scaling' mode).",
    )
    args = parser.parse_args()
    
    if args.mode == "single":
        benchmark_single_n(args.n, args.upper_nnz, args.repeats, args.seed, args.outdir)
    elif args.mode == "scaling":
        benchmark_scaling(args.matrix_sizes, args.nnz_ratio, args.repeats, args.seed, args.outdir)

if __name__ == "__main__":
    main()
