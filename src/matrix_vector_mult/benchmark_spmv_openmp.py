"""
Phase 2: OpenMP SpMV benchmarking with thread scaling.

Benchmarks OpenMP-accelerated SpMV on increasing thread counts (1, 2, 4).
Compares to serial baseline from Phase 1 to calculate speedup and efficiency.
"""

from __future__ import annotations
import argparse
import os
import time
import tracemalloc
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix, random as sparse_random, csr_matrix

import spmv_wrapper
from spmv_python import validate_spmv


def generate_test_matrix(
    n: int, nnz_ratio: float, seed: int = 0
) -> Tuple[csr_matrix, np.ndarray]:
    """Generate a random sparse matrix and dense vector."""
    nnz = int(n * nnz_ratio)
    density = nnz / (n * n)
    A = sparse_random(n, n, density=density, format='csr', random_state=seed)
    
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    
    return A, x


def benchmark_spmv_omp(
    A: csr_matrix, x: np.ndarray, num_threads: int, repeats: int = 5
) -> Dict[str, float]:
    """
    Benchmark OpenMP SpMV for a specific thread count.
    
    Parameters:
    - A: Sparse matrix in CSR format
    - x: Input vector
    - num_threads: Number of threads
    - repeats: Number of repetitions
    
    Returns:
    - Dict with timing and memory statistics
    """
    times = []
    peak_memory = 0.0
    
    # Set thread count
    spmv_wrapper.set_num_threads(num_threads)
    
    for _ in range(repeats):
        tracemalloc.start()
        t0 = time.perf_counter()
        y = spmv_wrapper.spmv_csr_omp(A, x, num_threads)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        
        _, peak = tracemalloc.get_traced_memory()
        peak_memory = max(peak_memory, peak)
        tracemalloc.stop()
    
    # Validate result on first run
    if repeats >= 1:
        y = spmv_wrapper.spmv_csr_omp(A, x, num_threads)
        if not validate_spmv(A, x, y):
            print(f"Warning: Validation failed for num_threads={num_threads}")
    
    avg_time = float(np.mean(times))
    min_time = float(np.min(times))
    max_time = float(np.max(times))
    peak_memory_mb = float(peak_memory / (1024 * 1024))
    gflops = (2.0 * A.nnz / avg_time / 1e9) if avg_time > 0 else 0.0
    
    return {
        "avg_time_s": avg_time,
        "min_time_s": min_time,
        "max_time_s": max_time,
        "memory_mb": peak_memory_mb,
        "gflops": gflops,
    }


def load_serial_baseline(results_dir: str = "results") -> pd.DataFrame:
    """
    Load serial baseline results from Phase 1.
    
    Parameters:
    - results_dir: Directory containing Phase 1 results
    
    Returns:
    - DataFrame with serial baseline (CSR format only)
    
    Raises:
    - FileNotFoundError: If baseline CSV not found
    """
    csv_path = os.path.join(results_dir, "spmv_serial_baseline.csv")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Serial baseline not found at {csv_path}\n"
            "Please run Phase 1 benchmark first:\n"
            "  python benchmark_spmv_serial.py --outdir results"
        )
    
    df = pd.read_csv(csv_path)
    # Filter to CSR format (fastest serial format)
    csr_df = df[df["Format"] == "CSR"].copy()
    
    return csr_df


def benchmark_spmv_openmp(
    matrix_sizes: List[int],
    thread_counts: List[int],
    nnz_ratios: List[float] = None,
    repeats: int = 5,
    outdir: str = "results",
    seed: int = 0,
) -> None:
    """
    Benchmark OpenMP SpMV across matrix sizes, thread counts, and NNZ ratios.
    
    Parameters:
    - matrix_sizes: List of matrix dimensions
    - thread_counts: List of thread counts to test (e.g., [1, 2, 4])
    - nnz_ratios: List of NNZ ratios to test (default: [5.0])
    - repeats: Number of repetitions per configuration
    - outdir: Output directory
    - seed: Random seed
    """
    if nnz_ratios is None:
        nnz_ratios = [5.0]
    
    all_results = []
    
    print("\n" + "=" * 140)
    print("PHASE 2: OPENMP SPMV BENCHMARKING")
    print("=" * 140)
    print(f"Matrix sizes: {matrix_sizes}")
    print(f"Thread counts: {thread_counts}")
    print(f"NNZ ratios (nnz/n): {nnz_ratios}")
    print(f"Repeats per config: {repeats}")
    print(f"Output directory: {outdir}")
    print("-" * 140)
    print(f"{'N':<10}{'NNZ':<12}{'Ratio':<8}{'Threads':<10}{'Avg_Time(s)':>15}{'Min(s)':>12}{'Max(s)':>12}{'Memory(MB)':>12}{'GFlop/s':>12}")
    print("-" * 140)
    
    os.makedirs(outdir, exist_ok=True)
    
    for ratio in nnz_ratios:
        for n in matrix_sizes:
            print(f"\nBenchmarking matrix size n={n}, nnz_ratio={ratio}...")
            A, x = generate_test_matrix(n, ratio, seed)
            nnz = A.nnz
        
        for num_threads in thread_counts:
            result = benchmark_spmv_omp(A, x, num_threads, repeats)
            
            all_results.append({
                "N": n,
                "NNZ": nnz,
                "NNZ_Ratio": ratio,
                "Threads": num_threads,
                "Avg_Time_s": result["avg_time_s"],
                "Min_Time_s": result["min_time_s"],
                "Max_Time_s": result["max_time_s"],
                "Memory_MB": result["memory_mb"],
                "GFlops": result["gflops"],
            })
            
            print(f"{n:<10}{nnz:<12}{ratio:<8.1f}{num_threads:<10}{result['avg_time_s']:>15.6f}{result['min_time_s']:>12.6f}"
                  f"{result['max_time_s']:>12.6f}{result['memory_mb']:>12.2f}{result['gflops']:>12.2f}")
    
    # Save OpenMP results
    df_omp = pd.DataFrame(all_results)
    csv_path = os.path.join(outdir, "spmv_openmp_results.csv")
    df_omp.to_csv(csv_path, index=False)
    print("\n" + "-" * 140)
    print(f"Saved OpenMP results to {csv_path}")
    
    # Load serial baseline and compute speedup/efficiency
    try:
        df_serial = load_serial_baseline(outdir)
        df_combined = compute_speedup_efficiency(df_omp, df_serial)
        combined_csv = os.path.join(outdir, "spmv_phase2_analysis.csv")
        df_combined.to_csv(combined_csv, index=False)
        print(f"Saved speedup/efficiency analysis to {combined_csv}")
        
        # Create visualizations
        create_visualizations(df_omp, df_combined, outdir)
    except FileNotFoundError as e:
        print(f"Warning: {e}")
        print("Skipping speedup/efficiency calculation and visualization")
    
    print("=" * 140)
    print(f"Phase 2 benchmark complete. Output: {outdir}/")
    print("=" * 140)


def compute_speedup_efficiency(
    df_omp: pd.DataFrame, df_serial: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute speedup and efficiency relative to serial baseline.
    
    Parameters:
    - df_omp: OpenMP results
    - df_serial: Serial baseline results (CSR only)
    
    Returns:
    - Combined DataFrame with speedup and efficiency columns
    """
    df_combined = df_omp.copy()
    
    # For each configuration, find corresponding serial baseline
    speedups = []
    efficiencies = []
    
    for idx, row in df_omp.iterrows():
        n = row["N"]
        threads = row["Threads"]
        omp_time = row["Avg_Time_s"]
        
        # Find serial baseline for this matrix size
        # Note: serial baseline may not have NNZ_Ratio column if from old run
        serial_rows = df_serial[df_serial["N"] == n]
        
        if len(serial_rows) > 0:
            serial_time = serial_rows.iloc[0]["Avg_Time_s"]
            speedup = serial_time / omp_time if omp_time > 0 else 0.0
            efficiency = speedup / threads if threads > 0 else 0.0
        else:
            speedup = 0.0
            efficiency = 0.0
        
        speedups.append(speedup)
        efficiencies.append(efficiency)
    
    df_combined["Speedup"] = speedups
    df_combined["Efficiency"] = efficiencies
    
    return df_combined


def create_visualizations(
    df_omp: pd.DataFrame, df_combined: pd.DataFrame, outdir: str
) -> None:
    """
    Create visualizations comparing thread scaling.
    
    Parameters:
    - df_omp: OpenMP benchmark results
    - df_combined: Results with speedup/efficiency
    - outdir: Output directory
    """
    sizes = sorted(df_omp["N"].unique())
    thread_counts = sorted(df_omp["Threads"].unique())
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    colors = {1: '#1f77b4', 2: '#ff7f0e', 4: '#2ca02c', 8: '#d62728', 16: '#9467bd'}
    markers = {1: 'o', 2: 's', 4: '^', 8: 'D', 16: 'v'}
    
    # Panel 1: Time vs Matrix Size (absolute)
    ax = axes[0, 0]
    for t in thread_counts:
        t_data = df_omp[df_omp["Threads"] == t].sort_values("N")
        ax.plot(t_data["N"], t_data["Avg_Time_s"],
               marker=markers.get(t, 'o'), label=f"{t} threads", linewidth=2, markersize=8,
               color=colors.get(t, '#000000'))
    ax.set_xlabel("Matrix Dimension (n)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Time (seconds)", fontsize=11, fontweight='bold')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title("Execution Time vs Matrix Size", fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Panel 2: GFlop/s vs Matrix Size
    ax = axes[0, 1]
    for t in thread_counts:
        t_data = df_omp[df_omp["Threads"] == t].sort_values("N")
        ax.plot(t_data["N"], t_data["GFlops"],
               marker=markers.get(t, 'o'), label=f"{t} threads", linewidth=2, markersize=8,
               color=colors.get(t, '#000000'))
    ax.set_xlabel("Matrix Dimension (n)", fontsize=11, fontweight='bold')
    ax.set_ylabel("GFlop/s", fontsize=11, fontweight='bold')
    ax.set_xscale('log')
    ax.set_title("Performance (GFlop/s) vs Matrix Size", fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Panel 3: Speedup vs Thread Count
    ax = axes[1, 0]
    for n in sizes:
        n_data = df_combined[df_combined["N"] == n].sort_values("Threads")
        ax.plot(n_data["Threads"], n_data["Speedup"],
               marker='o', label=f"n={n}", linewidth=2, markersize=8)
    # Ideal speedup line
    ideal_threads = sorted(thread_counts)
    ax.plot(ideal_threads, ideal_threads, 'k--', label='Ideal', linewidth=2, alpha=0.5)
    ax.set_xlabel("Number of Threads", fontsize=11, fontweight='bold')
    ax.set_ylabel("Speedup", fontsize=11, fontweight='bold')
    ax.set_title("Speedup vs Thread Count", fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)
    
    # Panel 4: Efficiency vs Thread Count
    ax = axes[1, 1]
    for n in sizes:
        n_data = df_combined[df_combined["N"] == n].sort_values("Threads")
        ax.plot(n_data["Threads"], n_data["Efficiency"],
               marker='o', label=f"n={n}", linewidth=2, markersize=8)
    # Ideal efficiency line (1.0)
    ax.axhline(y=1.0, color='k', linestyle='--', linewidth=2, alpha=0.5, label='Ideal')
    ax.set_xlabel("Number of Threads", fontsize=11, fontweight='bold')
    ax.set_ylabel("Efficiency", fontsize=11, fontweight='bold')
    ax.set_title("Parallel Efficiency vs Thread Count", fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(outdir, "spmv_openmp_scaling.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {plot_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Phase 2: OpenMP SpMV benchmarking with thread scaling."
    )
    parser.add_argument(
        "--matrix-sizes",
        type=int,
        nargs='+',
        default=[1000, 5000, 10000],
        help="List of matrix dimensions to benchmark.",
    )
    parser.add_argument(
        "--thread-counts",
        type=int,
        nargs='+',
        default=[1, 2, 4],
        help="List of thread counts to test.",
    )
    parser.add_argument(
        "--nnz-ratio",
        type=float,
        nargs='+',
        default=[5.0],
        help="List of NNZ ratios to test (nnz/n).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Number of repetitions per configuration.",
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
    
    args = parser.parse_args()
    
    benchmark_spmv_openmp(
        args.matrix_sizes,
        args.thread_counts,
        args.nnz_ratio,
        args.repeats,
        args.outdir,
        args.seed
    )


if __name__ == "__main__":
    main()
