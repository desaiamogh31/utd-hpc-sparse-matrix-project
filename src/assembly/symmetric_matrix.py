from __future__ import annotations
import argparse
import os
import time
import warnings
import tracemalloc
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

"""
Benchmark symmetric sparse matrix assembly across COO, CSR, LIL, and CSC.

This script provides two modes:
"""
''' SAMPLE USAGE:
# Single matrix benchmark
python symmetric_matrix.py --mode single --n 2000 --nnz-ratio 10.0 --repeats 3 --outdir results

# Multiple matrix sizes with single ratio
python symmetric_matrix.py --mode single --n 1000 2000 5000 --nnz-ratio 5.0 --repeats 3 --outdir results

# Multiple matrix sizes with multiple ratios
python symmetric_matrix.py --mode single --n 1000 10000 --nnz-ratio 4.0 8.0 16.0 --repeats 3 --outdir results

# Backward-compatible explicit upper-triangle entry counts
python symmetric_matrix.py --mode single --n 2000 --upper-nnz 10000 --repeats 3 --outdir results

# Scaling with custom sizes and ratio
python symmetric_matrix.py --mode scaling --matrix-sizes 1000 10000 100000 1000000 --nnz-ratio 4.0 8.0 16.0 --repeats 2 --outdir results
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

#Helper functions for symmetric matrix assembly and benchmarking
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
    - repeats: Number of benchmark repetitions.'''
    times = []
    peak_memory = 0
    last_mat = None
    for _ in range(repeats):
        tracemalloc.start()
        t0 = time.perf_counter()
        last_mat = fn(n, i, j, v)
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


def benchmark_single_n(
    n_values: List[int],
    nnz_ratios: List[float],
    repeats: int,
    seed: int,
    outdir: str = ".",
    upper_nnz_values: Optional[List[int]] = None,
) -> None:
    """
    Benchmark assembly of symmetric matrices with given parameters.
    Produces a table with runtimes for COO, CSR, LIL, and CSC formats across
    multiple N values and NNZ ratios.
    
    Parameters:
    - n_values: List of matrix dimensions to benchmark.
    - nnz_ratios: List of upper-triangle nnz ratios
      (upper_nnz = int(n * nnz_ratio) for each n).
    - repeats: Number of benchmark repetitions.
    - seed: Random seed for reproducibility.
    - outdir: Output directory for results.
    - upper_nnz_values: Optional explicit upper-triangle entry counts. If
      provided, these take precedence over nnz_ratios for backward
      compatibility.
    """
    all_results = []
    benchmark_cases = []

    if upper_nnz_values:
        case_label = f"upper_nnz_values={upper_nnz_values}"
        for n in n_values:
            for upper_nnz in upper_nnz_values:
                if upper_nnz > (n * (n + 1)) // 2:
                    raise ValueError(
                        f"upper-nnz {upper_nnz} exceeds total upper-triangle entries "
                        f"{(n * (n + 1)) // 2} for n={n}"
                    )
                benchmark_cases.append((n, upper_nnz, upper_nnz / n))
    else:
        case_label = f"nnz_ratios={nnz_ratios}"
        for n in n_values:
            for nnz_ratio in nnz_ratios:
                if nnz_ratio <= 0:
                    raise ValueError(f"NNZ ratio must be positive, got {nnz_ratio}")
                upper_nnz = int(n * nnz_ratio)
                if upper_nnz > (n * (n + 1)) // 2:
                    raise ValueError(
                        f"upper_nnz {upper_nnz} exceeds total upper-triangle entries "
                        f"{(n * (n + 1)) // 2} for n={n} (nnz_ratio={nnz_ratio})"
                    )
                benchmark_cases.append((n, upper_nnz, nnz_ratio))

    print(f"\nSymmetric assembly benchmark ({case_label}, repeats={repeats})")
    print("-" * 150)
    print(f"{'N':<8}{'NNZ':<12}{'Ratio':<8}{'Format':<8}{'Avg(s)':>12}{'Min(s)':>12}{'Max(s)':>12}{'NNZ_Result':>12}{'Memory(MB)':>12}{'EqualToCOO':>18}")
    print("-" * 150)

    for n, upper_nnz, nnz_ratio in benchmark_cases:
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

        ref = results["COO"]["matrix"]
        for name in ["COO", "LIL", "CSR", "CSC"]:
            r = results[name]
            ok = matrices_equal(r["matrix"], ref)
            print(
                f"{n:<8}{upper_nnz:<12}{nnz_ratio:<8.1f}{name:<8}{r['avg_s']:>12.6f}{r['min_s']:>12.6f}{r['max_s']:>12.6f}"
                f"{r['nnz']:>12d}{r['peak_memory_mb']:>12.2f}{str(ok):>18}"
            )
            all_results.append({
                "N": n,
                "NNZ": upper_nnz,
                "NNZ_Ratio": nnz_ratio,
                "Format": name,
                "Avg(s)": r["avg_s"],
                "Min(s)": r["min_s"],
                "Max(s)": r["max_s"],
                "NNZ_Result": r["nnz"],
                "Peak_Memory_MB": r["peak_memory_mb"],
                "EqualToCOO": ok,
            })

    table = pd.DataFrame(all_results)

    # Ensure output directory exists
    os.makedirs(outdir, exist_ok=True)
    
    output_path = f"{outdir}/symmetric_matrix_benchmark.csv"
    table.to_csv(output_path, index=False)
    print(f"\nSaved table to {output_path}")
    print(table.to_string(index=False))
    print("-" * 150)
    print("Note: Direct incremental assembly is usually fastest in LIL/COO,")
    print("while direct CSR/CSC insertion is typically much slower.")
    
    # Benchmark conversions from COO/LIL to CSR/CSC for the first N value and first ratio.
    if benchmark_cases:
        first_n, first_upper_nnz, first_ratio = benchmark_cases[0]
        print("\n" + "=" * 150)
        print(f"Format Conversion Benchmarks (N={first_n}, nnz_ratio={first_ratio})")
        print("=" * 150)

        i, j, v = generate_upper_tri_entries(first_n, first_upper_nnz, seed)
        builders = {
            "COO": assemble_coo,
            "LIL": assemble_lil,
        }

        source_matrices = {}
        for name, fn in builders.items():
            source_matrices[name] = benchmark(fn, first_n, i, j, v, repeats)["matrix"]

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

        conversion_output_path = f"{outdir}/symmetric_matrix_conversion_benchmark.csv"
        conversion_table.to_csv(conversion_output_path, index=False)
        print(f"\nSaved conversion table to {conversion_output_path}")
        print(conversion_table.to_string(index=False))
        print("-" * 150)


def benchmark_scaling(
    matrix_sizes: List[int], nnz_ratio_sizes: List[float], repeats: int, seed: int, outdir: str = "."
) -> None:
    """
    Benchmark assembly runtimes and memory for COO and LIL matrices across different sizes and NNZ ratios.
    Generates plots with separate traces for each NNZ ratio (solid COO, dashed LIL) and saves timing data.
    
    Parameters:
    - matrix_sizes: List of matrix dimensions to benchmark.
    - nnz_ratio_sizes: List of ratios of nnz to n (upper_nnz = int(n * nnz_ratio) for each n).
    - repeats: Number of benchmark repetitions per matrix size.
    - seed: Random seed for reproducibility.
    """
    sizes = matrix_sizes
    
    # Data organized by nnz_ratio: nnz_ratio -> {"coo_times": [...], "lil_times": [...], ...}
    data_by_ratio = {ratio: {"coo_times": [], "lil_times": [], "coo_memory": [], "lil_memory": []} 
                     for ratio in nnz_ratio_sizes}
    all_results = []  # For comprehensive table with entry generation times
    
    print(f"\nScaling benchmark: COO vs LIL with multiple NNZ ratios")
    print(f"Matrix sizes: {sizes}")
    print(f"NNZ ratios (nnz/n): {nnz_ratio_sizes}")
    print(f"Repeats per size: {repeats}")
    print("-" * 140)
    print(f"{'N':<8}{'NNZ_Ratio':<12}{'NNZ':<12}{'Entry_Gen(s)':>13}{'COO_Avg(s)':>15}{'LIL_Avg(s)':>15}{'COO_Mem(MB)':>12}{'LIL_Mem(MB)':>12}")
    print("-" * 140)
    
    for n in sizes:
        for nnz_ratio in nnz_ratio_sizes:
            if nnz_ratio <= 0:
                raise ValueError(f"NNZ ratio must be positive, got {nnz_ratio}")
            upper_nnz = int(n * nnz_ratio)
            if upper_nnz > (n * (n + 1)) // 2:
                raise ValueError(
                    f"upper_nnz {upper_nnz} exceeds total upper-triangle entries "
                    f"{(n * (n + 1)) // 2} for n={n}"
                )
            
            # Time entry generation
            start_time = time.perf_counter()
            i, j, v = generate_upper_tri_entries(n, upper_nnz, seed)
            end_time = time.perf_counter()
            gen_time = end_time - start_time
            
            # Benchmark COO
            coo_result = benchmark(assemble_coo, n, i, j, v, repeats)
            
            # Benchmark LIL
            lil_result = benchmark(assemble_lil, n, i, j, v, repeats)
            
            # Store data by ratio
            data_by_ratio[nnz_ratio]["coo_times"].append(coo_result["avg_s"])
            data_by_ratio[nnz_ratio]["lil_times"].append(lil_result["avg_s"])
            data_by_ratio[nnz_ratio]["coo_memory"].append(coo_result["peak_memory_mb"])
            data_by_ratio[nnz_ratio]["lil_memory"].append(lil_result["peak_memory_mb"])
            
            # Store for comprehensive table
            all_results.append({
                "N": n,
                "NNZ_Ratio": nnz_ratio,
                "NNZ": upper_nnz,
                "Entry_Gen_Time_s": gen_time,
                "COO_Avg_s": coo_result["avg_s"],
                "LIL_Avg_s": lil_result["avg_s"],
                "COO_Peak_Memory_MB": coo_result["peak_memory_mb"],
                "LIL_Peak_Memory_MB": lil_result["peak_memory_mb"],
            })
            
            print(f"{n:<8}{nnz_ratio:<12.1f}{upper_nnz:<12}{gen_time:>13.6f}{coo_result['avg_s']:>15.6f}"
                  f"{lil_result['avg_s']:>15.6f}{coo_result['peak_memory_mb']:>12.2f}{lil_result['peak_memory_mb']:>12.2f}")
    
    # Create plots
    os.makedirs(outdir, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    
    markers = ['o', 's', '^', 'D', 'v', 'p']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # Plot 1: Time vs matrix size for each NNZ ratio
    for idx, nnz_ratio in enumerate(nnz_ratio_sizes):
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        
        # COO: solid line
        ax1.plot(sizes, data_by_ratio[nnz_ratio]["coo_times"], 
                linestyle='-', marker=marker, color=color,
                label=f'COO (ratio={nnz_ratio})', linewidth=2, markersize=8)
        
        # LIL: dashed line
        ax1.plot(sizes, data_by_ratio[nnz_ratio]["lil_times"], 
                linestyle='--', marker=marker, color=color, 
                label=f'LIL (ratio={nnz_ratio})', linewidth=2, markersize=8, alpha=0.7)
    
    ax1.set_xlabel('Matrix Dimension (n)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Assembly Time (seconds)', fontsize=12, fontweight='bold')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_title('Time Scaling: COO (solid) vs LIL (dashed)', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10, loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Memory vs matrix size for each NNZ ratio
    for idx, nnz_ratio in enumerate(nnz_ratio_sizes):
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        
        # COO: solid line
        ax2.plot(sizes, data_by_ratio[nnz_ratio]["coo_memory"],
                linestyle='-', marker=marker, color=color,
                label=f'COO (ratio={nnz_ratio})', linewidth=2, markersize=8)
        
        # LIL: dashed line
        ax2.plot(sizes, data_by_ratio[nnz_ratio]["lil_memory"], 
                linestyle='--', marker=marker, color=color, 
                label=f'LIL (ratio={nnz_ratio})', linewidth=2, markersize=8, alpha=0.7)
    
    ax2.set_xlabel('Matrix Dimension (n)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Peak Memory (MB)', fontsize=12, fontweight='bold')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_title('Memory Scaling: COO (solid) vs LIL (dashed)', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10, loc='best')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = f"{outdir}/symmetric_matrix_scaling.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved scaling plots to {plot_path}")
    
    # Save comprehensive timing data to CSV
    scaling_table = pd.DataFrame(all_results)
    csv_path = f"{outdir}/symmetric_matrix_scaling.csv"
    scaling_table.to_csv(csv_path, index=False)
    print(f"Saved timing data to {csv_path}")
    print("\n" + "="*140)
    print("COMPREHENSIVE SCALING TABLE")
    print("="*140)
    print(scaling_table.to_string(index=False))
    print("-" * 95)

#main function and additional plotting functions for comparison across matrix types
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
        help="Benchmark mode: 'single' for one or more matrix sizes and ratios, 'scaling' for multiple sizes with one or more ratios.",
    )
    
    # Arguments for single-N benchmark
    parser.add_argument(
        "--n",
        type=int,
        nargs='+',
        default=[2000],
        help="Matrix dimensions to benchmark (used in 'single' mode). Can specify multiple: --n 1000 2000 5000",
    )
    parser.add_argument(
        "--upper-nnz",
        type=int,
        nargs='+',
        default=None,
        help="Optional explicit upper-triangle entry counts for 'single' mode. Can specify multiple: --upper-nnz 5000 10000",
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
        nargs='+',
        default=[10.0],
        help="List of upper-triangle NNZ ratios (upper_nnz/n). Used in 'single' mode unless --upper-nnz is provided, and in 'scaling' mode. Can specify multiple: --nnz-ratio 4.0 8.0 16.0",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=".",
        help="Output directory for plots and CSV files (used in 'scaling' mode).",
    )
    args = parser.parse_args()
    
    if args.mode == "single":
        benchmark_single_n(
            args.n,
            args.nnz_ratio,
            args.repeats,
            args.seed,
            args.outdir,
            upper_nnz_values=args.upper_nnz,
        )
    elif args.mode == "scaling":
        benchmark_scaling(args.matrix_sizes, args.nnz_ratio, args.repeats, args.seed, args.outdir)

if __name__ == "__main__":
    main()
