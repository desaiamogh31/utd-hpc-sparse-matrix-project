#!/usr/bin/env python3
"""
Phase 2 HPC Benchmarking: OpenMP Thread Scaling on Large Matrices

Tests OpenMP SpMV scaling with:
- Large matrices (100K to 1M elements) with varying nnz/row ratios
- Many thread counts (1, 2, 4, 8, 16, 32, 64, ...)
- Strong scaling analysis (fix matrix, vary threads)
- Optional real matrix validation

Usage:
    python benchmark_phase2_hpc.py --matrix-sizes 100000 500000 1000000 \\
                                    --nnz-ratios 5.0 10.0 20.0 \\
                                    --threads 1 2 4 8 16 32 64 \\
                                    --repeats 3 \\
                                    --outdir results_hpc_phase2

    # With real matrix validation
    python benchmark_phase2_hpc.py --matrix-sizes 100000 500000 \\
                                    --nnz-ratios 5.0 10.0 \\
                                    --threads 1 2 4 8 16 32 \\
                                    --repeats 3 \\
                                    --include-real-matrices \\
                                    --outdir results_hpc_phase2
"""

from __future__ import annotations
import argparse
import os
import sys
import time
import tracemalloc
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import random as sparse_random, csr_matrix

import spmv_wrapper
from spmv_python import validate_spmv

try:
    from load_real_matrices import load_suite_sparse_matrix
    HAS_REAL_MATRIX_LOADER = True
except ImportError:
    HAS_REAL_MATRIX_LOADER = False


def generate_random_csr_matrix(n: int, nnz_ratio: float, seed: int = 42) -> csr_matrix:
    """
    Generate random CSR matrix with consistent properties.
    
    Parameters:
    - n: Matrix dimension
    - nnz_ratio: Nonzeros per row (nnz / n)
    - seed: Random seed for reproducibility
    
    Returns:
    - CSR matrix
    """
    np.random.seed(seed)
    density = nnz_ratio / n  # Convert nnz_per_row to density
    A = sparse_random(n, n, density=density, format='csr', random_state=seed)
    return A


def benchmark_spmv_omp(
    A: csr_matrix,
    x: np.ndarray,
    num_threads: int,
    repeats: int = 3
) -> Dict[str, float]:
    """
    Benchmark OpenMP SpMV for a specific thread count.
    
    Parameters:
    - A: Sparse matrix in CSR format
    - x: Input vector
    - num_threads: Number of threads
    - repeats: Number of repetitions
    
    Returns:
    - Dict with timing and performance statistics
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
    
    # Validate correctness
    y_check = spmv_wrapper.spmv_csr_omp(A, x, num_threads)
    if not validate_spmv(A, x, y_check):
        print(f"  ⚠️  Validation warning for threads={num_threads}")
    
    avg_time = float(np.mean(times))
    min_time = float(np.min(times))
    max_time = float(np.max(times))
    std_time = float(np.std(times))
    peak_memory_mb = float(peak_memory / (1024 * 1024))
    gflops = (2.0 * A.nnz / avg_time / 1e9) if avg_time > 0 else 0.0
    
    return {
        "avg_time_s": avg_time,
        "min_time_s": min_time,
        "max_time_s": max_time,
        "std_time_s": std_time,
        "memory_mb": peak_memory_mb,
        "gflops": gflops,
    }


def benchmark_matrix(
    A: csr_matrix,
    matrix_name: str,
    matrix_type: str,
    thread_counts: List[int],
    repeats: int = 3,
) -> List[Dict]:
    """
    Benchmark a single matrix across multiple thread counts.
    
    Parameters:
    - A: Sparse matrix in CSR format
    - matrix_name: Name/identifier for matrix
    - matrix_type: "random" or "real"
    - thread_counts: Thread counts to test
    - repeats: Repetitions per configuration
    
    Returns:
    - List of result dictionaries
    """
    print(f"\n  Matrix: {matrix_name} ({matrix_type})")
    print(f"    Shape: {A.shape}, NNZ: {A.nnz}")
    print(f"    Density: {A.nnz / (A.shape[0] * A.shape[1]):.2e}")
    
    results = []
    
    # Generate input vector
    rng = np.random.default_rng(42)
    x = rng.standard_normal(A.shape[1])
    
    # Get baseline at 1 thread for speedup calculation
    baseline_result = benchmark_spmv_omp(A, x, num_threads=1, repeats=repeats)
    baseline_time = baseline_result["avg_time_s"]
    
    print(f"  Benchmarking across {len(thread_counts)} thread counts...")
    for num_threads in thread_counts:
        result = benchmark_spmv_omp(A, x, num_threads, repeats)
        speedup = baseline_time / result["avg_time_s"]
        
        results.append({
            "MatrixType": matrix_type,
            "MatrixName": matrix_name,
            "N": A.shape[0],
            "NNZ": A.nnz,
            "Density": A.nnz / (A.shape[0] * A.shape[1]),
            "Threads": num_threads,
            "AvgTime_s": result["avg_time_s"],
            "MinTime_s": result["min_time_s"],
            "MaxTime_s": result["max_time_s"],
            "StdTime_s": result["std_time_s"],
            "Memory_MB": result["memory_mb"],
            "GFlops": result["gflops"],
            "Speedup": speedup,
        })
        
        print(f"    Threads={num_threads:2d}: {result['avg_time_s']:8.6f}s, " +
              f"Speedup={speedup:6.2f}x, {result['gflops']:7.2f} GFlop/s")
    
    return results


def benchmark_phase2_hpc(
    matrix_sizes: List[int],
    nnz_ratios: List[float],
    thread_counts: List[int],
    repeats: int = 3,
    include_real_matrices: bool = False,
    cache_dir: str = "matrices",
    outdir: str = "results_hpc_phase2",
) -> pd.DataFrame:
    """
    Benchmark Phase 2 (OpenMP) on HPC with strong scaling tests.
    
    Parameters:
    - matrix_sizes: List of matrix dimensions to test
    - nnz_ratios: List of sparsity ratios (nnz / (n*n)) to test
    - thread_counts: Thread counts to test
    - repeats: Repetitions per configuration
    - include_real_matrices: Whether to include SuiteSparse validation
    - cache_dir: Cache directory for matrices
    - outdir: Output directory
    
    Returns:
    - DataFrame with benchmark results
    """
    os.makedirs(outdir, exist_ok=True)
    
    print("\n" + "=" * 100)
    print("PHASE 2 HPC: OpenMP Thread Scaling Benchmark")
    print("=" * 100)
    print(f"Matrix sizes: {matrix_sizes}")
    print(f"NNZ ratios: {nnz_ratios}")
    print(f"Thread counts: {thread_counts}")
    print(f"Repeats: {repeats}")
    print(f"Output directory: {outdir}")
    print("=" * 100)
    
    all_results = []
    
    # ===== TIER 1: RANDOM MATRICES (Primary Strong Scaling) =====
    print("\n" + "=" * 100)
    print("TIER 1: Random Matrices (Strong Scaling)")
    print("=" * 100)
    
    for nnz_ratio in nnz_ratios:
        for n in matrix_sizes:
            nnz = int(n * nnz_ratio)  # nnz_ratio is nonzeros per row
            print(f"\nGenerating random matrix: n={n}, nnz_ratio={nnz_ratio:.2f}, nnz={nnz}")
            
            A = generate_random_csr_matrix(n, nnz_ratio=nnz_ratio, seed=42)
            matrix_name = f"random_n{n}_nnzr{nnz_ratio:.2f}"
            
            results = benchmark_matrix(A, matrix_name, "random", thread_counts, repeats)
            all_results.extend(results)
    
    # ===== TIER 2: REAL MATRICES (Validation, Optional) =====
    if include_real_matrices and HAS_REAL_MATRIX_LOADER:
        print("\n" + "=" * 100)
        print("TIER 2: Real Matrices (Validation)")
        print("=" * 100)
        
        real_matrices = [
            ("HB", "bcsstk30"),
        ]
        
        for group, name in real_matrices:
            print(f"\nLoading real matrix: {group}/{name}")
            try:
                A, info = load_suite_sparse_matrix(group, name, cache_dir)
                
                # Limit thread count for real matrices (to save time)
                limited_threads = [t for t in thread_counts if t <= 16]
                results = benchmark_matrix(A, name, "real", limited_threads, repeats)
                all_results.extend(results)
            except Exception as e:
                print(f"  ✗ Failed to load {group}/{name}: {e}")
                continue
    
    # ===== CREATE DATAFRAME AND SAVE =====
    df = pd.DataFrame(all_results)
    
    csv_path = os.path.join(outdir, "phase2_hpc_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n✓ CSV saved: {csv_path}")
    
    return df


def save_results_and_plots(
    df: pd.DataFrame,
    outdir: str = "results_hpc_phase2",
) -> None:
    """
    Create visualization plots for Phase 2 HPC results.
    
    Parameters:
    - df: DataFrame with benchmark results
    - outdir: Output directory
    """
    os.makedirs(outdir, exist_ok=True)
    
    if df.empty:
        print("WARNING: DataFrame is empty, skipping plots")
        return
    
    # Get unique matrices
    matrices = df["MatrixName"].unique()
    thread_counts = sorted(df["Threads"].unique())
    
    # Helper function to get matrix info string
    def get_matrix_info(matrix_name, df):
        mat_data = df[df["MatrixName"] == matrix_name].iloc[0]
        n = int(mat_data["N"])
        nnz = int(mat_data["NNZ"])
        return f"{matrix_name} (N={n:,}, NNZ={nnz:,})"
    
    # ===== PLOT 1: SPEEDUP =====
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for matrix in matrices:
        mat_data = df[df["MatrixName"] == matrix].sort_values("Threads")
        matrix_label = get_matrix_info(matrix, df)
        
        ax.plot(
            mat_data["Threads"],
            mat_data["Speedup"],
            marker='o',
            label=matrix_label,
            linewidth=2.5,
            markersize=8
        )
    
    # Add ideal speedup line
    ax.plot(thread_counts, thread_counts, 'k--', linewidth=2, alpha=0.6, label='Ideal Speedup')
    
    ax.set_xlabel("Number of Threads", fontsize=16, fontweight='bold')
    ax.set_ylabel("Speedup", fontsize=16, fontweight='bold')
    ax.set_title("Phase 2 HPC: Speedup vs Thread Count", fontsize=18, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.legend(fontsize=13, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plot_path = os.path.join(outdir, "plot_speedup.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Plot saved: {plot_path}")
    
    # ===== PLOT 2: GFLOPS =====
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for matrix in matrices:
        mat_data = df[df["MatrixName"] == matrix].sort_values("Threads")
        matrix_label = get_matrix_info(matrix, df)
        
        ax.plot(
            mat_data["Threads"],
            mat_data["GFlops"],
            marker='^',
            label=matrix_label,
            linewidth=2.5,
            markersize=8
        )
    
    ax.set_xlabel("Number of Threads", fontsize=16, fontweight='bold')
    ax.set_ylabel("GFlop/s", fontsize=16, fontweight='bold')
    ax.set_title("Phase 2 HPC: Performance Scaling", fontsize=18, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.legend(fontsize=13, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plot_path = os.path.join(outdir, "plot_gflops.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Plot saved: {plot_path}")
    
    # ===== PLOT 3: RUNTIME =====
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for matrix in matrices:
        mat_data = df[df["MatrixName"] == matrix].sort_values("Threads")
        matrix_label = get_matrix_info(matrix, df)
        
        ax.plot(
            mat_data["Threads"],
            mat_data["AvgTime_s"],
            marker='s',
            label=matrix_label,
            linewidth=2.5,
            markersize=8
        )
    
    ax.set_xlabel("Number of Threads", fontsize=16, fontweight='bold')
    ax.set_ylabel("Runtime (seconds)", fontsize=16, fontweight='bold')
    ax.set_title("Phase 2 HPC: Runtime Scaling", fontsize=18, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.legend(fontsize=13, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plot_path = os.path.join(outdir, "plot_runtime.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Plot saved: {plot_path}")
    
    # ===== PLOT 4: EFFICIENCY =====
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for matrix in matrices:
        mat_data = df[df["MatrixName"] == matrix].sort_values("Threads")
        matrix_label = get_matrix_info(matrix, df)
        efficiency = mat_data["Speedup"] / mat_data["Threads"] * 100  # as percentage
        
        ax.plot(
            mat_data["Threads"],
            efficiency,
            marker='D',
            label=matrix_label,
            linewidth=2.5,
            markersize=8
        )
    
    ax.axhline(y=100, color='k', linestyle='--', linewidth=2, alpha=0.6, label='Ideal (100%)')
    
    ax.set_xlabel("Number of Threads", fontsize=16, fontweight='bold')
    ax.set_ylabel("Efficiency (%)", fontsize=16, fontweight='bold')
    ax.set_title("Phase 2 HPC: Parallel Efficiency", fontsize=18, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.legend(fontsize=13, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim([0, 120])
    plt.tight_layout()
    plot_path = os.path.join(outdir, "plot_efficiency.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Plot saved: {plot_path}")
    
    print("\n" + "=" * 100)
    print("All plots generated successfully!")
    print("=" * 100)


def main():
    parser = argparse.ArgumentParser(
        description="Phase 2 HPC: Benchmark OpenMP SpMV with strong scaling on large matrices",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--matrix-sizes",
        type=int,
        nargs='+',
        default=[100000, 500000, 1000000],
        help="Matrix dimensions to test (default: 100000 500000 1000000)",
    )
    
    parser.add_argument(
        "--nnz-ratios",
        type=float,
        nargs='+',
        default=[5.0, 10.0],
        help="NNZ ratios (nonzeros per row) to test (default: 5.0 10.0)",
    )
    
    parser.add_argument(
        "--threads",
        type=int,
        nargs='+',
        default=[1, 2, 4, 8, 16, 32, 64],
        help="Thread counts to test (default: 1 2 4 8 16 32 64)",
    )
    
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Repetitions per configuration (default: 3)",
    )
    
    parser.add_argument(
        "--include-real-matrices",
        action="store_true",
        help="Include real matrices from SuiteSparse for validation",
    )
    
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="matrices",
        help="Cache directory for matrices (default: matrices)",
    )
    
    parser.add_argument(
        "--outdir",
        type=str,
        default="results_hpc_phase2",
        help="Output directory (default: results_hpc_phase2)",
    )
    
    args = parser.parse_args()
    
    # Run benchmark
    df = benchmark_phase2_hpc(
        matrix_sizes=args.matrix_sizes,
        nnz_ratios=args.nnz_ratios,
        thread_counts=args.threads,
        repeats=args.repeats,
        include_real_matrices=args.include_real_matrices,
        cache_dir=args.cache_dir,
        outdir=args.outdir,
    )
    
    # Generate plots
    save_results_and_plots(df, args.outdir)
    
    # Print summary
    print("\nBENCHMARK SUMMARY:")
    print(df.to_string(index=False))
    
    print("\n" + "=" * 100)
    print(f"Results saved to: {args.outdir}/")
    print("=" * 100)


if __name__ == "__main__":
    main()
