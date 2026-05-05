from __future__ import annotations
import argparse
import os
import time
import warnings
import tracemalloc
from typing import Callable, Dict, List, Tuple
import numpy as np
import pandas as pd

"""
Benchmark 2D Laplacian sparse matrix assembly across COO, CSR, LIL, and CSC.

The 2D Laplacian on an n×n grid is represented as an (n²)×(n²) matrix with
the 5-point stencil pattern (center and 4 neighbors).

This script provides two modes:
"""
''' SAMPLE USAGE:
# Single grid dimension benchmark
python laplacian_matrix.py --mode single --n 100 --repeats 3 --outdir results

# Multiple grid dimensions with single mode
python laplacian_matrix.py --mode single --n 50 100 150 200 --repeats 3 --outdir results

# Scaling with custom grid sizes
python laplacian_matrix.py --mode scaling --matrix-sizes 10 20 50 100 150 --repeats 2 --outdir results
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
    peak_memory = 0
    last_mat = None
    
    for _ in range(repeats):
        tracemalloc.start()
        t0 = time.perf_counter()
        last_mat = fn(n, rows, cols, vals)
        times.append(time.perf_counter() - t0)
        _, peak = tracemalloc.get_traced_memory()
        peak_memory = max(peak_memory, peak)
        tracemalloc.stop()

    assert last_mat is not None
    return {
        "avg_s": float(np.mean(times)),
        "min_s": float(np.min(times)),
        "max_s": float(np.max(times)),
        "nnz": int(last_mat.nnz),
        "peak_memory_mb": float(peak_memory / (1024 * 1024)),
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
    Benchmark conversions from COO/LIL to CSR/CSC formats with memory tracking.
    
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
        peak_memory = 0
        
        for _ in range(repeats):
            tracemalloc.start()
            t0 = time.perf_counter()
            if tgt_format == "CSR":
                _ = src_mat.tocsr()
            elif tgt_format == "CSC":
                _ = src_mat.tocsc()
            times.append(time.perf_counter() - t0)
            _, peak = tracemalloc.get_traced_memory()
            peak_memory = max(peak_memory, peak)
            tracemalloc.stop()
        
        conversion_key = f"{src_format}->{tgt_format}"
        results[conversion_key] = {
            "avg_s": float(np.mean(times)),
            "min_s": float(np.min(times)),
            "max_s": float(np.max(times)),
            "peak_memory_mb": float(peak_memory / (1024 * 1024)),
        }
    
    return results


def benchmark_scaling(
    matrix_sizes: List[int], repeats: int, seed: int, outdir: str = "."
) -> None:
    """
    Benchmark assembly runtimes and memory for Laplacian across different sizes.
    Generates plots of runtime and memory vs matrix dimension.
    
    Parameters:
    - matrix_sizes: List of grid dimensions to benchmark.
    - repeats: Number of benchmark repetitions per matrix size.
    - seed: Random seed (unused for laplacian but kept for consistency).
    - outdir: Output directory for results.
    """
    sizes = matrix_sizes
    coo_times = []
    lil_times = []
    coo_memory = []
    lil_memory = []
    mat_sizes = []
    
    print(f"\nScaling benchmark: COO vs LIL for Laplacian")
    print(f"Grid sizes: {sizes}")
    print(f"Repeats per size: {repeats}")
    print("-" * 85)
    print(f"{'N':<8}{'Matrix Size':<15}{'COO Avg(s)':>15}{'LIL Avg(s)':>15}{'COO Mem(MB)':>12}{'LIL Mem(MB)':>12}")
    print("-" * 85)
    
    for n in sizes:
        mat_size = n * n
        mat_sizes.append(mat_size)
        
        start_time = time.perf_counter()
        rows, cols, vals = generate_laplacian_entries(n)
        end_time = time.perf_counter()
        time_taken_for_generation = end_time - start_time
        print(f"Generated Laplacian entries for n={n} (matrix size {mat_size}×{mat_size}) in {time_taken_for_generation:.6f} seconds.")
        # Benchmark COO
        coo_result = benchmark(assemble_coo, n, rows, cols, vals, repeats)
        coo_times.append(coo_result["avg_s"])
        coo_memory.append(coo_result["peak_memory_mb"])
        
        # Benchmark LIL
        lil_result = benchmark(assemble_lil, n, rows, cols, vals, repeats)
        lil_times.append(lil_result["avg_s"])
        lil_memory.append(lil_result["peak_memory_mb"])
        
        print(f"{n:<8}{mat_size:<15}{coo_result['avg_s']:>15.6f}{lil_result['avg_s']:>15.6f}"
              f"{coo_result['peak_memory_mb']:>12.2f}{lil_result['peak_memory_mb']:>12.2f}")
    
    # Create plots
    os.makedirs(outdir, exist_ok=True)
    
    # Time plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(sizes, coo_times, marker='o', label='COO', linewidth=2, markersize=8)
    ax1.plot(sizes, lil_times, marker='s', label='LIL', linewidth=2, markersize=8)
    ax1.set_xlabel('Grid Dimension (n)', fontsize=12)
    ax1.set_ylabel('Assembly Time (seconds)', fontsize=12)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_title('Laplacian Assembly Time vs Size', fontsize=13)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Memory plot
    ax2.plot(sizes, coo_memory, marker='o', label='COO', linewidth=2, markersize=8)
    ax2.plot(sizes, lil_memory, marker='s', label='LIL', linewidth=2, markersize=8)
    ax2.set_xlabel('Grid Dimension (n)', fontsize=12)
    ax2.set_ylabel('Peak Memory (MB)', fontsize=12)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_title('Laplacian Assembly Memory vs Size', fontsize=13)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = f"{outdir}/laplacian_scaling.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved scaling plots to {plot_path}")
    
    # Save timing data to CSV
    scaling_table = pd.DataFrame({
        "N": sizes,
        "Matrix_Size": mat_sizes,
        "COO_Avg_s": coo_times,
        "LIL_Avg_s": lil_times,
        "COO_Peak_Memory_MB": coo_memory,
        "LIL_Peak_Memory_MB": lil_memory,
    })
    
    csv_path = f"{outdir}/laplacian_scaling.csv"
    scaling_table.to_csv(csv_path, index=False)
    print(f"Saved timing data to {csv_path}")
    print(scaling_table.to_string(index=False))
    print("-" * 85)


def benchmark_single_n(
    n_values: List[int], repeats: int, outdir: str = "."
) -> None:
    """
    Benchmark assembly of Laplacian matrices with given grid dimensions.
    Produces a table with runtimes for COO, CSR, LIL, and CSC formats across multiple N values.
    
    Parameters:
    - n_values: List of grid dimensions to benchmark.
    - repeats: Number of benchmark repetitions.
    - outdir: Output directory for results.
    """
    all_results = []
    
    print(f"\nLaplacian assembly benchmark (repeats={repeats})")
    print("-" * 160)
    print(f"{'N':<8}{'Matrix_Size':<15}{'Format':<8}{'Avg(s)':>12}{'Min(s)':>12}{'Max(s)':>12}{'NNZ':>12}{'Memory(MB)':>12}{'EqualToCOO':>18}")
    print("-" * 160)
    
    for n in n_values:
        mat_size = n * n
        
        rows, cols, vals = generate_laplacian_entries(n)
        builders = {
            "COO": assemble_coo,
            "LIL": assemble_lil,
            "CSR": assemble_csr,
            "CSC": assemble_csc,
        }

        results = {}
        for name, fn in builders.items():
            results[name] = benchmark(fn, n, rows, cols, vals, repeats)

        # Correctness: compare all against COO result.
        ref = results["COO"]["matrix"]
        
        for name in ["COO", "LIL", "CSR", "CSC"]:
            r = results[name]
            ok = matrices_equal(r["matrix"], ref)
            print(
                f"{n:<8}{mat_size:<15}{name:<8}{r['avg_s']:>12.6f}{r['min_s']:>12.6f}{r['max_s']:>12.6f}"
                f"{r['nnz']:>12d}{r['peak_memory_mb']:>12.2f}{str(ok):>18}"
            )
            
            all_results.append({
                "N": n,
                "Matrix_Size": mat_size,
                "Format": name,
                "Avg(s)": r["avg_s"],
                "Min(s)": r["min_s"],
                "Max(s)": r["max_s"],
                "NNZ": r["nnz"],
                "Peak_Memory_MB": r["peak_memory_mb"],
                "EqualToCOO": ok,
            })

    table = pd.DataFrame(all_results)

    # Ensure output directory exists
    os.makedirs(outdir, exist_ok=True)
    
    output_path = f"{outdir}/laplacian_matrix_benchmark.csv"
    table.to_csv(output_path, index=False)
    print(f"\nSaved table to {output_path}")
    print(table.to_string(index=False))
    print("-" * 160)
    print("Note: Direct incremental assembly is usually fastest in LIL/COO,")
    print("while direct CSR/CSC insertion is typically much slower.")
    
    # Benchmark conversions from COO/LIL to CSR/CSC for the first N value
    if n_values:
        print("\n" + "=" * 160)
        print(f"Format Conversion Benchmarks (N={n_values[0]})")
        print("=" * 160)
        
        # Re-generate entries for first N to benchmark conversions
        n = n_values[0]
        rows, cols, vals = generate_laplacian_entries(n)
        
        builders = {
            "COO": assemble_coo,
            "LIL": assemble_lil,
        }
        
        source_matrices = {}
        for name, fn in builders.items():
            source_matrices[name] = benchmark(fn, n, rows, cols, vals, repeats)["matrix"]
        
        conversion_results = benchmark_conversions(source_matrices, repeats)
        
        print(f"\n{'Conversion':<15}{'Avg(s)':>12}{'Min(s)':>12}{'Max(s)':>12}{'Memory(MB)':>12}")
        print("-" * 63)
        for name in sorted(conversion_results.keys()):
            r = conversion_results[name]
            print(
                f"{name:<15}{r['avg_s']:>12.6f}{r['min_s']:>12.6f}{r['max_s']:>12.6f}{r['peak_memory_mb']:>12.2f}"
            )
        
        conversion_table = pd.DataFrame(
            [
                {
                    "Conversion": name,
                    "Avg(s)": conversion_results[name]["avg_s"],
                    "Min(s)": conversion_results[name]["min_s"],
                    "Max(s)": conversion_results[name]["max_s"],
                    "Peak_Memory_MB": conversion_results[name]["peak_memory_mb"],
                }
                for name in sorted(conversion_results.keys())
            ]
        )
        
        conversion_output_path = f"{outdir}/laplacian_conversion_benchmark.csv"
        conversion_table.to_csv(conversion_output_path, index=False)
        print(f"\nSaved conversion table to {conversion_output_path}")
        print(conversion_table.to_string(index=False))
        print("-" * 160)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark 2D Laplacian sparse matrix assembly for COO/CSR/LIL/CSC."
    )
    
    # Mode selection
    parser.add_argument(
        "--mode",
        type=str,
        default="single",
        choices=["single", "scaling"],
        help="Benchmark mode: 'single' for one or more grid sizes, 'scaling' for multiple sizes.",
    )
    
    # Arguments for single-N benchmark
    parser.add_argument(
        "--n",
        type=int,
        nargs='+',
        default=[100],
        help="Grid dimensions to benchmark (used in 'single' mode). Can specify multiple: --n 50 100 150 200"
    )
    parser.add_argument("--repeats", type=int, default=3, help="Benchmark repetitions.")
    
    # Arguments for scaling benchmark
    parser.add_argument(
        "--matrix-sizes",
        type=int,
        nargs='+',
        default=[10, 20, 50, 100],
        help="List of grid dimensions to benchmark (used in 'scaling' mode).",
    )
    parser.add_argument("--outdir", type=str, default=".", help="Output directory for results.")
    args = parser.parse_args()

    if args.mode == "single":
        benchmark_single_n(args.n, args.repeats, args.outdir)
    elif args.mode == "scaling":
        benchmark_scaling(args.matrix_sizes, args.repeats, seed=0, outdir=args.outdir)


if __name__ == "__main__":
    main()
