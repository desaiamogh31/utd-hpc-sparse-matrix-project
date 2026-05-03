"""
Phase 1: Serial SpMV benchmarking on laptop.

Benchmarks SpMV across selected sparse formats with multiple matrix sizes.
Generates CSV output and visualization comparing format performance.
"""
#SAMPLE USAGE:
# Quick test (smaller matrices, fewer repeats)
#python benchmark_spmv_serial.py --matrix-sizes 500 1000 --repeats 2 --nnz-ratio 3.0 --formats coo csr csc --outdir results

# Detailed test (larger matrices, more repeats)
#python benchmark_spmv_serial.py --matrix-sizes 1000 5000 10000 --repeats 10 --formats coo csr csc lil --nnz-ratios 8.0 16.0 --outdir results

from __future__ import annotations
import argparse
import os
import time
from typing import Dict, List, Sequence, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix

from spmv_python import benchmark_spmv_format


def generate_test_matrix(
    n: int, nnz_ratio: float, seed: int = 0
) -> Tuple[coo_matrix, np.ndarray]:
    """
    Generate a random sparse matrix and dense vector for testing.
    
    Parameters:
    - n: Matrix dimension (n × n)
    - nnz_ratio: Ratio of nnz to n (nnz = int(n * nnz_ratio))
    - seed: Random seed
    
    Returns:
    - A_coo: Random sparse matrix in COO format
    - x: Random dense vector
    """
    nnz = int(n * nnz_ratio)
    rng = np.random.default_rng(seed)

    # Generate sparse coordinates directly so memory scales with nnz rather
    # than the full n x n index space. Duplicates are acceptable at these
    # densities and are merged afterward.
    row = rng.integers(0, n, size=nnz, dtype=np.int32)
    col = rng.integers(0, n, size=nnz, dtype=np.int32)
    data = rng.standard_normal(nnz)
    A = coo_matrix((data, (row, col)), shape=(n, n))
    A.sum_duplicates()

    # Generate random vector
    x = rng.standard_normal(n)
    
    return A, x


def format_ratio_suffix(nnz_ratio: float) -> str:
    """
    Format an nnz ratio for use in output filenames.
    """
    ratio_str = f"{nnz_ratio:g}"
    return ratio_str.replace("-", "neg").replace(".", "p")


def benchmark_spmv_serial(
    matrix_sizes: List[int],
    repeats: int = 5,
    nnz_ratios: Sequence[float] = (5.0,),
    outdir: str = "results",
    seed: int = 0,
    formats: Sequence[str] = ("coo", "csr", "csc", "lil"),
) -> None:
    """
    Benchmark serial SpMV across multiple matrix sizes and formats.
    
    Parameters:
    - matrix_sizes: List of matrix dimensions to test
    - repeats: Number of repetitions per configuration
    - nnz_ratios: Ratios of nnz to n
    - outdir: Output directory for results
    - seed: Random seed
    - formats: Sparse formats to benchmark
    """
    formats = [fmt.upper() for fmt in formats]
    nnz_ratios = [float(ratio) for ratio in nnz_ratios]
    
    print("\n" + "=" * 120)
    print("PHASE 1: SERIAL SPMV BASELINE BENCHMARK")
    print("=" * 120)
    print(f"Matrix sizes: {matrix_sizes}")
    print(f"NNZ ratios (nnz/n): {nnz_ratios}")
    print(f"Repeats per size: {repeats}")
    print(f"Formats: {formats}")
    print(f"Output directory: {outdir}")
    print("-" * 120)
    print(f"{'N':<10}{'NNZ_R':<8}{'NNZ':<12}{'Format':<8}{'Avg_Time(s)':>15}{'Min(s)':>12}{'Max(s)':>12}{'Memory(MB)':>12}{'GFlop/s':>12}")
    print("-" * 120)
    
    # Create output directory
    os.makedirs(outdir, exist_ok=True)
    
    for nnz_ratio in nnz_ratios:
        ratio_results = []
        ratio_suffix = format_ratio_suffix(nnz_ratio)
        print(f"\nNNZ ratio (nnz/n) = {nnz_ratio}")
        for n in matrix_sizes:
            print(f"\nBenchmarking matrix size n={n}...")
            A, x = generate_test_matrix(n, nnz_ratio, seed)
            nnz = A.nnz

            for fmt in formats:
                result = benchmark_spmv_format(A, x, fmt.lower(), repeats)

                ratio_results.append({
                    "N": n,
                    "NNZ_Ratio": nnz_ratio,
                    "NNZ": nnz,
                    "Format": fmt,
                    "Avg_Time_s": result["avg_time_s"],
                    "Min_Time_s": result["min_time_s"],
                    "Max_Time_s": result["max_time_s"],
                    "Memory_MB": result["memory_mb"],
                    "GFlops": result["gflops"],
                })

                print(f"{n:<10}{nnz_ratio:<8.1f}{nnz:<12}{fmt:<8}{result['avg_time_s']:>15.6f}{result['min_time_s']:>12.6f}"
                      f"{result['max_time_s']:>12.6f}{result['memory_mb']:>12.2f}{result['gflops']:>12.2f}")

        df_ratio = pd.DataFrame(ratio_results)
        csv_path = os.path.join(outdir, f"spmv_serial_baseline_nnzr_{ratio_suffix}.csv")
        df_ratio.to_csv(csv_path, index=False)
        print("\n" + "-" * 120)
        print(f"Saved results to {csv_path}")

        create_visualization(df_ratio, outdir, ratio_suffix)
    
    print("=" * 120)
    print(f"Benchmark complete. Output: {outdir}/")
    print("=" * 120)


def create_visualization(df: pd.DataFrame, outdir: str, ratio_suffix: str) -> None:
    """
    Create visualization comparing format performance.
    
    Parameters:
    - df: Results DataFrame
    - outdir: Output directory
    - ratio_suffix: Filename-safe nnz ratio suffix
    """
    formats = sorted(df["Format"].unique())
    nnz_ratio = float(df["NNZ_Ratio"].iloc[0]) if "NNZ_Ratio" in df.columns else None
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = {'COO': '#1f77b4', 'CSR': '#ff7f0e', 'CSC': '#2ca02c', 'LIL': '#d62728'}
    markers = {'COO': 'o', 'CSR': 's', 'CSC': '^', 'LIL': 'D'}
    
    # Panel 1: Time comparison
    for fmt in formats:
        fmt_data = df[df["Format"] == fmt]
        fmt_data_sorted = fmt_data.sort_values("N")
        ax1.plot(fmt_data_sorted["N"], fmt_data_sorted["Avg_Time_s"],
                marker=markers[fmt], label=fmt, linewidth=2, markersize=8,
                color=colors[fmt])
    
    ax1.set_xlabel("Matrix Dimension (n)", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Time (seconds)", fontsize=12, fontweight='bold')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    title_suffix = f", nnz/n={nnz_ratio:g}" if nnz_ratio is not None else ""
    ax1.set_title(f"SpMV Time Comparison (Serial{title_suffix})", fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: GFlop/s comparison
    for fmt in formats:
        fmt_data = df[df["Format"] == fmt]
        fmt_data_sorted = fmt_data.sort_values("N")
        ax2.plot(fmt_data_sorted["N"], fmt_data_sorted["GFlops"],
                marker=markers[fmt], label=fmt, linewidth=2, markersize=8,
                color=colors[fmt])
    
    ax2.set_xlabel("Matrix Dimension (n)", fontsize=12, fontweight='bold')
    ax2.set_ylabel("GFlop/s", fontsize=12, fontweight='bold')
    ax2.set_xscale('log')
    ax2.set_title(f"SpMV Performance (Serial{title_suffix})", fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(outdir, f"spmv_serial_comparison_nnzr_{ratio_suffix}.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {plot_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Phase 1: Serial SpMV benchmarking on laptop."
    )
    parser.add_argument(
        "--matrix-sizes",
        type=int,
        nargs='+',
        default=[1000, 5000, 10000],
        help="List of matrix dimensions to benchmark.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Number of repetitions per configuration.",
    )
    parser.add_argument(
        "--nnz-ratio",
        type=float,
        default=5.0,
        help="Ratio of nnz to n (nnz = int(n * nnz_ratio)).",
    )
    parser.add_argument(
        "--nnz-ratios",
        type=float,
        nargs="+",
        help="One or more nnz/n ratios to benchmark in a single run.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="results",
        help="Output directory for results.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--formats",
        type=str,
        nargs="+",
        choices=["coo", "csr", "csc", "lil"],
        default=["coo", "csr", "csc", "lil"],
        help="Sparse formats to benchmark.",
    )
    
    args = parser.parse_args()
    nnz_ratios = args.nnz_ratios if args.nnz_ratios is not None else [args.nnz_ratio]
    
    benchmark_spmv_serial(
        args.matrix_sizes,
        args.repeats,
        nnz_ratios,
        args.outdir,
        args.seed,
        args.formats,
    )


if __name__ == "__main__":
    main()
