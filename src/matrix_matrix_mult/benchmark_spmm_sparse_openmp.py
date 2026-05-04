"""
Sparse B SpMM Benchmarking Suite (OpenMP Local Baseline)

Tests sparse matrix-matrix multiplication C = A @ B where:
  - A: Sparse matrix (same preset matrix set and size guards as the serial sparse-B benchmark)
  - B: Randomly generated sparse matrix
  - Three OpenMP custom algorithms + SciPy built-in for comparison

Sample usage:
    python benchmark_spmm_sparse_openmp.py --outdir results_omp
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

sys.path.insert(0, os.path.dirname(__file__))

from benchmark_spmm_sparse_serial import generate_sparse_b, load_local_matrix
from build import build as build_openmp


try:
    from spmm_openmp_wrapper import benchmark_spmm_algorithm_sparse_b_openmp
except FileNotFoundError:
    if not build_openmp():
        raise
    from spmm_openmp_wrapper import benchmark_spmm_algorithm_sparse_b_openmp


def benchmark_matrix_sparse_b_openmp(
    A,
    matrix_name: str,
    b_cols: list[int],
    sparsities: list[float],
    thread_counts: list[int],
    repeats: int = 3,
) -> list[dict[str, float]]:
    """Benchmark one sparse matrix against multiple sparse-B and thread settings."""
    results: list[dict[str, float]] = []
    m, n = A.shape

    print(f"\nBenchmarking {matrix_name} ({m}x{n}, nnz={A.nnz}):")

    for k in b_cols:
        for sparsity in sparsities:
            print(f"\n  B: {n}x{k}, sparsity={sparsity:.2%} (nnz={int(n * k * sparsity)})")
            B = generate_sparse_b(n, k, sparsity=sparsity, dtype=A.dtype)

            for num_threads in thread_counts:
                print(f"    Threads={num_threads}")
                for algo in ["row-wise", "outer-product", "blocked", "scipy"]:
                    print(f"      Benchmarking {algo}...", end=" ", flush=True)

                    try:
                        metrics = benchmark_spmm_algorithm_sparse_b_openmp(
                            A,
                            B,
                            algorithm=algo,
                            repeats=repeats,
                            num_threads=num_threads,
                        )
                        print(f"✓ ({metrics['gflops']:.2f} GFlop/s)")

                        results.append(
                            {
                                "matrix_name": matrix_name,
                                "m": m,
                                "n": n,
                                "nnz_a": metrics["nnz_a"],
                                "k": k,
                                "nnz_b": metrics["nnz_b"],
                                "sparsity_b": sparsity,
                                "num_threads": num_threads,
                                "algorithm": algo,
                                "nnz_c": metrics["nnz_c"],
                                "mean_time_sec": metrics["mean_time"],
                                "std_time_sec": metrics["std_time"],
                                "min_time_sec": metrics["min_time"],
                                "gflops": metrics["gflops"],
                                "memory_a_mb": metrics["memory_a_mb"],
                                "memory_b_mb": metrics["memory_b_mb"],
                                "memory_c_mb": metrics["memory_c_mb"],
                            }
                        )
                    except Exception as exc:
                        print(f"✗ Error: {exc}")

    return results


def benchmark_all_matrices_sparse_b_openmp(
    cache_dir: str,
    b_cols: list[int],
    sparsities: list[float],
    thread_counts: list[int],
    repeats: int,
    output_csv: str,
    output_dir: str,
) -> None:
    """Run the sparse-B OpenMP benchmark on the same allowed matrix set as the serial version."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_csv)

    all_results: list[dict[str, float]] = []

    print("Loading matrices...")
    test_matrices = [
        ("1138_bus", load_local_matrix("1138_bus", cache_dir)),
        ("abb313", load_local_matrix("abb313", cache_dir)),
        ("delaunay_n15", load_local_matrix("delaunay_n15", cache_dir)),
    ]

    for matrix_name, (A, source) in test_matrices:
        if A is None:
            print(f"✗ Could not load {matrix_name}, skipping...")
            continue

        m, n = A.shape
        max_b_k = max(b_cols)
        max_b_size_mb = (n * max_b_k * 8) / (1024 * 1024)

        if A.nnz > 2e6:
            print(f"✗ {matrix_name}: nnz={A.nnz:.2e} > 2M threshold, skipping...")
            continue
        if max_b_size_mb > 2000:
            print(
                f"✗ {matrix_name}: estimated B size {max_b_size_mb:.1f} MB > 2000 MB, skipping..."
            )
            continue

        print(f"✓ Loaded from {source}")
        matrix_results = benchmark_matrix_sparse_b_openmp(
            A,
            matrix_name,
            b_cols,
            sparsities,
            thread_counts,
            repeats=repeats,
        )
        all_results.extend(matrix_results)

    if not all_results:
        print("\n✗ No results to write")
        return

    fieldnames = [
        "matrix_name",
        "m",
        "n",
        "nnz_a",
        "k",
        "nnz_b",
        "sparsity_b",
        "num_threads",
        "algorithm",
        "nnz_c",
        "mean_time_sec",
        "std_time_sec",
        "min_time_sec",
        "gflops",
        "memory_a_mb",
        "memory_b_mb",
        "memory_c_mb",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\n✓ Results written to {output_path}")
    print(f"  Total runs: {len(all_results)}")

    if MATPLOTLIB_AVAILABLE:
        print("\nGenerating plots...")
        plot_results(pd.DataFrame(all_results), output_dir)
    else:
        print("\n⚠ matplotlib not available—skipping plots")


def plot_results(df: pd.DataFrame, output_dir: str = "results/") -> None:
    """Generate comparison, scaling, and thread-runtime plots for the OpenMP benchmark."""
    if not MATPLOTLIB_AVAILABLE:
        return

    os.makedirs(output_dir, exist_ok=True)
    sparsity_label = ", ".join(f"{value:.0%}" for value in sorted(df["sparsity_b"].unique()))

    colors = {
        "row-wise": "#1f77b4",
        "outer-product": "#ff7f0e",
        "blocked": "#2ca02c",
        "scipy": "#d62728",
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Sparse B SpMM OpenMP: Algorithm Comparison (Custom vs SciPy)\nB sparsity: {sparsity_label}",
        fontsize=14,
        fontweight="bold",
    )

    matrices = sorted(df["matrix_name"].unique())
    x = np.arange(len(matrices))
    width = 0.2

    ax = axes[0, 0]
    for i, algo in enumerate(["row-wise", "outer-product", "blocked", "scipy"]):
        algo_times = [
            df[(df["matrix_name"] == matrix_name) & (df["algorithm"] == algo)][
                "mean_time_sec"
            ].mean()
            for matrix_name in matrices
        ]
        ax.bar(x + (i - 1.5) * width, algo_times, width, label=algo, color=colors[algo])
    ax.set_xlabel("Matrix")
    ax.set_ylabel("Mean Time (seconds)")
    ax.set_title("Execution Time by Matrix, Averaged over B columns and thread counts")
    ax.set_xticks(x)
    ax.set_xticklabels(matrices, rotation=45)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend()

    ax = axes[0, 1]
    for i, algo in enumerate(["row-wise", "outer-product", "blocked", "scipy"]):
        algo_gflops = [
            df[(df["matrix_name"] == matrix_name) & (df["algorithm"] == algo)]["gflops"].mean()
            for matrix_name in matrices
        ]
        ax.bar(
            x + (i - 1.5) * width,
            algo_gflops,
            width,
            label=algo,
            color=colors[algo],
        )
    ax.set_xlabel("Matrix")
    ax.set_ylabel("GFlop/s")
    ax.set_title("Performance by Matrix, Averaged over B columns and thread counts")
    ax.set_xticks(x)
    ax.set_xticklabels(matrices, rotation=45)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend()

    ax = axes[1, 0]
    for algo in ["row-wise", "outer-product", "blocked", "scipy"]:
        algo_df = df[df["algorithm"] == algo]
        nnz_perf = algo_df.groupby("nnz_b")["mean_time_sec"].mean().sort_index()
        ax.plot(
            nnz_perf.index,
            nnz_perf.values,
            label=algo,
            color=colors[algo],
            alpha=0.85,
            marker="o",
            linewidth=2,
        )
    ax.set_xlabel("nnz(B)")
    ax.set_ylabel("Mean Time (seconds)")
    ax.set_title("Execution Time vs B Sparsity")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 1]
    scipy_times = df[df["algorithm"] == "scipy"].set_index(
        ["matrix_name", "k", "sparsity_b", "num_threads"]
    )["mean_time_sec"]
    speedups = []
    labels = []
    for algo in ["row-wise", "outer-product", "blocked"]:
        algo_df = df[df["algorithm"] == algo].set_index(
            ["matrix_name", "k", "sparsity_b", "num_threads"]
        )
        speedup = algo_df["mean_time_sec"] / scipy_times
        speedups.append(speedup.dropna().values)
        labels.append(algo)

    bp = ax.boxplot(speedups, tick_labels=labels, patch_artist=True)
    for patch, algo in zip(bp["boxes"], labels):
        patch.set_facecolor(colors[algo])
        patch.set_alpha(0.7)
    ax.set_ylabel("Speedup Ratio (Custom / SciPy)")
    ax.set_title("SciPy Speedup over Custom Implementations")
    ax.set_yscale("log")
    ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.5, label="1x (SciPy baseline)")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend()

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "spmm_sparse_openmp_algorithm_comparison.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"  ✓ Saved: {plot_path}")
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Sparse B SpMM OpenMP: B Column Count and Performance\nB sparsity: {sparsity_label}",
        fontsize=14,
        fontweight="bold",
    )

    max_thread = int(df["num_threads"].max())
    max_thread_df = df[df["num_threads"] == max_thread]

    ax = axes[0]
    for matrix_name in sorted(max_thread_df["matrix_name"].unique()):
        matrix_df = max_thread_df[
            (max_thread_df["matrix_name"] == matrix_name)
            & (max_thread_df["algorithm"] == "scipy")
        ]
        col_perf = matrix_df.groupby("k")["mean_time_sec"].mean().sort_index()
        ax.plot(col_perf.index, col_perf.values, marker="o", label=matrix_name, linewidth=2)
    ax.set_xlabel("B Column Count (k)")
    ax.set_ylabel("Mean Time (seconds)")
    ax.set_title(f"Execution Time vs B Columns (SciPy, {max_thread} threads setting)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1]
    for matrix_name in sorted(max_thread_df["matrix_name"].unique()):
        matrix_df = max_thread_df[
            (max_thread_df["matrix_name"] == matrix_name)
            & (max_thread_df["algorithm"] == "scipy")
        ]
        col_perf = matrix_df.groupby("k")["gflops"].mean().sort_index()
        ax.plot(col_perf.index, col_perf.values, marker="s", label=matrix_name, linewidth=2)
    ax.set_xlabel("B Column Count (k)")
    ax.set_ylabel("GFlop/s")
    ax.set_title(f"Performance vs B Columns (SciPy, {max_thread} threads setting)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "spmm_sparse_openmp_scaling.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"  ✓ Saved: {plot_path}")
    plt.close()

    largest_matrix_row = (
        df[["matrix_name", "nnz_a"]].drop_duplicates().sort_values(["nnz_a", "matrix_name"]).iloc[-1]
    )
    largest_matrix = largest_matrix_row["matrix_name"]
    thread_df = df[df["matrix_name"] == largest_matrix]

    fig, ax = plt.subplots(figsize=(8, 5))
    for algo in ["row-wise", "outer-product", "blocked", "scipy"]:
        algo_df = thread_df[thread_df["algorithm"] == algo]
        series = algo_df.groupby("num_threads")["mean_time_sec"].mean().sort_index()
        ax.plot(
            series.index,
            series.values,
            marker="o",
            linewidth=2,
            label=algo,
            color=colors[algo],
        )
    ax.set_xlabel("Number of Threads")
    ax.set_ylabel("Mean Runtime (seconds)")
    ax.set_title(
        f"Thread Scaling on {largest_matrix}\nAveraged over sparse B column counts, B sparsity: {sparsity_label}"
    )
    ax.set_xticks(sorted(thread_df["num_threads"].unique()))
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "spmm_sparse_openmp_thread_scaling.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"  ✓ Saved: {plot_path}")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark sparse-B SpMM with OpenMP (local laptop baseline)."
    )
    parser.add_argument(
        "--output",
        "-o",
        default="benchmark_spmm_sparse_openmp.csv",
        help="Output CSV filename (default: benchmark_spmm_sparse_openmp.csv)",
    )
    parser.add_argument(
        "--outdir",
        default="results/",
        help="Output directory (default: results/)",
    )
    parser.add_argument(
        "--cache-dir",
        default="matrices/",
        help="Matrix cache location (default: matrices/)",
    )
    parser.add_argument(
        "--repeats",
        "-r",
        type=int,
        default=5,
        help="Repetitions per benchmark (default: 5)",
    )
    parser.add_argument(
        "--b-cols",
        type=int,
        nargs="+",
        default=[4, 8, 16],
        help="Column counts for sparse B (default: 4 8 16)",
    )
    parser.add_argument(
        "--sparsity",
        type=float,
        nargs="+",
        default=[0.10],
        help="Sparse B density levels (default: 0.10)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        nargs="+",
        default=[1, 2, 4],
        help="OpenMP thread counts to benchmark (default: 1 2 4)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Sparse B SpMM OpenMP Benchmark")
    print("=" * 70)
    print(f"Output: {args.outdir}/{args.output}")
    print(f"B columns: {args.b_cols}")
    print(f"B sparsity: {args.sparsity}")
    print(f"Thread counts: {args.threads}")
    print(f"Repeats: {args.repeats}")
    print("=" * 70)

    benchmark_all_matrices_sparse_b_openmp(
        cache_dir=args.cache_dir,
        b_cols=args.b_cols,
        sparsities=args.sparsity,
        thread_counts=args.threads,
        repeats=args.repeats,
        output_csv=args.output,
        output_dir=args.outdir,
    )


if __name__ == "__main__":
    main()
