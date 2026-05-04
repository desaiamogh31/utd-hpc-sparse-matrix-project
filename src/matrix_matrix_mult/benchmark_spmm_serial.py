"""
Benchmark Suite for Serial SpMM (Sparse Matrix-Matrix Multiplication).

Tests three algorithms (row-wise, outer-product, blocked) across:
- Multiple sparse matrices (Phase 2 test set + larger matrices)
- Varying dense matrix column counts (1, 4, 8, 16, 32, 64, 128, 256, 512)
- Multiple algorithm variants

Outputs CSV results with timing, GFlop/s, memory usage, and sparsity metrics.

USAGE EXAMPLES:

1. Default run (uses local matrices/, outputs to results/):
    python benchmark_spmm_serial.py

2. Custom output directory:
    python benchmark_spmm_serial.py --outdir my_benchmarks

3. Custom output filename and directory:
    python benchmark_spmm_serial.py --outdir phase3a_results --output spmm_algo_comparison.csv

4. Reduce column counts for quick testing:
    python benchmark_spmm_serial.py --b-cols 1 4 8 16 --repeats 2

5. Subset of dense columns (focus on compute-bound regime):
    python benchmark_spmm_serial.py --outdir compute_bound_results --b-cols 64 128 256 512

6. Single repeat for debugging:
    python benchmark_spmm_serial.py --repeats 1 --b-cols 1 4 8

OPTIONS:
    --output, -o FILE       Output CSV filename (default: benchmark_spmm_serial_baseline.csv)
    --outdir DIR            Output directory for results (default: results/)
    --cache-dir DIR         Directory with .mtx matrices (default: matrices/, falls back to Phase 2)
    --repeats, -r N         Repetitions per benchmark (default: 3)
    --b-cols K1 K2 ...      Column counts to test (default: 1 4 8 16 32 64 128 256 512)
    --help, -h              Show this help message

MATRIX LOADING:
    - Checks local matrices/ directory first
    - Falls back to Phase 2 (../matrix_vector_mult/matrices/)
    - Falls back to SuiteSparse download if available
    - No manual copying needed—fully automatic!
"""

import os
import sys
import time
import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, random as sparse_random
from scipy.io import mmread

# Try to import matplotlib for plotting
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "matrix_vector_mult"))

# Import our modules
from spmm_python import spmm, benchmark_spmm_all_algorithms, validate_spmm
from load_real_matrices import load_suite_sparse_matrix, load_matrix_ssgetpy, download_matrix

# Mapping of matrix names to SuiteSparse collection info (group, name)
MATRIX_METADATA = {
    "1138_bus": ("HB", "1138_bus"),
    "abb313": ("HB", "abb313"),
    "bcsstk30": ("HB", "bcsstk30"),
    "delaunay_n15": ("DIMACS10", "delaunay_n15"),
    "delaunay_n19": ("DIMACS10", "delaunay_n19"),
    "pkustk14": ("Chen", "pkustk14"),
}


def load_local_matrix(matrix_name: str, cache_dir: str = "matrices") -> Tuple[csr_matrix, Dict]:
    """
    Load a matrix from local cache by name.
    If not found locally, attempt to download from SuiteSparse Matrix Collection.
    
    Returns: (A, info_dict)
    """
    from scipy.io import mmread
    
    # Ensure cache directory exists
    os.makedirs(cache_dir, exist_ok=True)
    
    filepath = os.path.join(cache_dir, f"{matrix_name}.mtx")
    
    # Try to load from local cache first
    if os.path.exists(filepath):
        A = mmread(filepath)
        info = {
            "name": matrix_name,
            "source": "local cache",
            "shape": A.shape,
            "nnz": A.nnz,
        }
        return A.tocsr(), info
    
    # Fallback: check Phase 2 matrices directory
    phase2_path = os.path.normpath(os.path.join(cache_dir, "..", "..", "matrix_vector_mult", "matrices", f"{matrix_name}.mtx"))
    if os.path.exists(phase2_path):
        print(f"  Found {matrix_name} in Phase 2 directory, loading...")
        A = mmread(phase2_path)
        info = {
            "name": matrix_name,
            "source": "Phase 2 matrices",
            "shape": A.shape,
            "nnz": A.nnz,
        }
        return A.tocsr(), info
    
    # Final fallback: try to download from SuiteSparse
    if matrix_name in MATRIX_METADATA:
        group, name = MATRIX_METADATA[matrix_name]
        print(f"  {matrix_name} not found locally, downloading from SuiteSparse ({group}/{name})...")
        try:
            filepath = download_matrix(group, name, cache_dir)
            A = mmread(filepath)
            info = {
                "name": matrix_name,
                "source": "downloaded from SuiteSparse",
                "shape": A.shape,
                "nnz": A.nnz,
            }
            return A.tocsr(), info
        except Exception as e:
            raise FileNotFoundError(f"Failed to load {matrix_name}: {e}")
    else:
        raise FileNotFoundError(f"Matrix {matrix_name} not found and no download info available")


def create_synthetic_matrix(
    m: int, n: int, sparsity: float = 0.01, seed: int = 42
) -> Tuple[csr_matrix, Dict]:
    """
    Create a synthetic sparse matrix using random sparse generation.
    
    Parameters:
    - m, n: Matrix dimensions
    - sparsity: Fraction of non-zero elements
    - seed: Random seed for reproducibility
    
    Returns: (A, info_dict)
    """
    np.random.seed(seed)
    A = sparse_random(m, n, density=sparsity, format="csr")
    info = {
        "name": f"synthetic_{m}x{n}_sparsity{sparsity}",
        "source": "synthetic",
        "shape": A.shape,
        "nnz": A.nnz,
    }
    return A, info


def load_test_matrices(cache_dir: str = "matrices") -> List[Tuple[csr_matrix, Dict]]:
    """
    Load all available test matrices (Phase 2 set + synthetic larger matrices).
    
    Returns: List of (matrix, info) tuples
    """
    matrices = []
    
    # Phase 2 local matrices
    local_matrices = [
        "1138_bus",
        "abb313",
        "bcsstk30",
        "delaunay_n15",
        "delaunay_n19",
        "pkustk14",
    ]
    
    for name in local_matrices:
        try:
            A, info = load_local_matrix(name, cache_dir)
            matrices.append((A, info))
            print(f"✓ Loaded {name}: {A.shape} with {A.nnz} nnz")
        except Exception as e:
            print(f"✗ Could not load {name}: {e}")
    
    # Larger synthetic matrices for distributed memory readiness
    # NOTE: Disabled for serial baseline; very large matrices best suited for MPI testing
    # synthetic_configs = [
    #     (100000, 100000, 0.0001),  # 100K x 100K, 0.01% sparse ≈ 10K nnz
    #     (500000, 500000, 0.00001),  # 500K x 500K, 0.001% sparse ≈ 2.5K nnz (undirected graph)
    # ]
    # 
    # for m, n, sparsity in synthetic_configs:
    #     try:
    #         A, info = create_synthetic_matrix(m, n, sparsity)
    #         matrices.append((A, info))
    #         print(f"✓ Created synthetic {info['name']}: {A.shape} with {A.nnz} nnz")
    #     except Exception as e:
    #         print(f"✗ Failed to create synthetic matrix: {e}")
    
    return matrices


def benchmark_matrix_algorithm_variations(
    A: csr_matrix,
    b_col_counts: List[int],
    repeats: int = 3,
) -> List[Dict]:
    """
    Benchmark all algorithms on a single matrix with varying B column counts.
    
    Parameters:
    - A: Sparse matrix to benchmark
    - b_col_counts: List of column counts to test
    - repeats: Repetitions per algorithm variant
    
    Returns: List of result dictionaries
    """
    results = []
    m, n = A.shape
    
    for k in b_col_counts:
        print(f"\n  Testing with B having {k} columns...")
        
        # Generate random dense matrix B (n x k)
        B = np.random.randn(n, k).astype(np.float64)
        
        # Benchmark all three algorithms
        algo_results = benchmark_spmm_all_algorithms(A, B, repeats=repeats)
        
        # Store results
        for algo_name, metrics in algo_results.items():
            result = {
                "matrix_name": f"Shape{m}x{n}",
                "nnz_a": metrics["nnz_a"],
                "nnz_c": metrics["nnz_c"],
                "b_cols": k,
                "algorithm": algo_name,
                "mean_time_sec": metrics["mean_time"],
                "std_time_sec": metrics["std_time"],
                "min_time_sec": metrics["min_time"],
                "gflops": metrics["gflops"],
                "memory_a_mb": metrics["memory_a_mb"],
                "memory_b_mb": metrics["memory_b_mb"],
                "memory_c_mb": metrics["memory_c_mb"],
            }
            results.append(result)
    
    return results


def benchmark_all_matrices(
    cache_dir: str = "matrices",
    b_col_counts: List[int] = None,
    repeats: int = 3,
    output_csv: str = "benchmark_spmm_serial_baseline.csv",
    output_dir: str = ".",
) -> None:
    """
    Benchmark SpMM on all available test matrices.
    
    Writes results to CSV file in output_dir.
    """
    if b_col_counts is None:
        b_col_counts = [1, 4, 8, 16, 32, 64, 128, 256, 512]
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Construct full output path
    output_path = os.path.join(output_dir, output_csv)
    
    print("=" * 80)
    print("SPARSE MATRIX-MATRIX MULTIPLICATION (SpMM) SERIAL BENCHMARK")
    print("=" * 80)
    print()
    
    # Load test matrices
    print("Loading test matrices...")
    matrices = load_test_matrices(cache_dir)
    
    if not matrices:
        print("ERROR: No matrices found. Check cache_dir and ensure Phase 2 matrices exist.")
        return
    
    all_results = []
    
    # Benchmark each matrix
    for A, info in matrices:
        print(f"\nBenchmarking {info['name']} ({A.shape[0]}x{A.shape[1]}, {A.nnz} nnz)...")
        
        # Skip very large matrices for serial benchmarking (memory-intensive)
        # Check both nnz and memory estimate for dense B matrix (largest column count)
        max_b_cols = max(b_col_counts)
        estimated_b_memory_mb = (A.shape[0] * max_b_cols * 8) / (1024 * 1024)  # 8 bytes per float64
        
        if A.nnz > 2e6 or estimated_b_memory_mb > 2000:
            print(f"  Skipping {info['name']}: too large for serial baseline")
            print(f"    (nnz={A.nnz:.0e}, estimated B matrix size={estimated_b_memory_mb:.1f} MB for {max_b_cols} columns)")
            continue
        
        results = benchmark_matrix_algorithm_variations(A, b_col_counts, repeats=repeats)
        all_results.extend(results)
    
    # Write results to CSV
    print(f"\nWriting results to {output_path}...")
    if all_results:
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)
        print(f"✓ Wrote {len(all_results)} result rows to {output_path}")
        
        # Generate plots if matplotlib is available
        if MATPLOTLIB_AVAILABLE:
            print("\nGenerating plots...")
            plot_results(pd.DataFrame(all_results), output_dir)
        else:
            print("\n⚠ matplotlib not available—skipping plots (install with: pip install matplotlib)")
    else:
        print("ERROR: No results to write.")
    
    print("\n" + "=" * 80)
    print("Benchmark Complete")
    print("=" * 80)


def plot_results(df: pd.DataFrame, output_dir: str = "results/") -> None:
    """
    Generate plots for dense B benchmark results.
    
    Parameters:
    - df: DataFrame with benchmark results
    - output_dir: Directory to save plots
    """
    if not MATPLOTLIB_AVAILABLE:
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot 1: Algorithm Comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Dense B SpMM: Algorithm Comparison", fontsize=14, fontweight='bold')
    
    colors = {
        'row-wise': '#1f77b4',
        'outer-product': '#ff7f0e',
        'blocked': '#2ca02c',
    }
    
    # Panel 1: Time by matrix
    ax = axes[0, 0]
    matrices = sorted(df['matrix_name'].unique())
    x = np.arange(len(matrices))
    width = 0.25
    for i, algo in enumerate(['row-wise', 'outer-product', 'blocked']):
        algo_times = [df[(df['matrix_name'] == m) & (df['algorithm'] == algo)]['mean_time_sec'].mean() 
                      for m in matrices]
        ax.bar(x + (i - 1) * width, algo_times, width, label=algo, color=colors[algo])
    ax.set_xlabel('Matrix')
    ax.set_ylabel('Mean Time (seconds)')
    ax.set_title('Execution Time by Matrix, Averaged over B column counts')
    ax.set_xticks(x)
    ax.set_xticklabels(matrices, rotation=45)
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Panel 2: GFlop/s by matrix
    ax = axes[0, 1]
    for i, algo in enumerate(['row-wise', 'outer-product', 'blocked']):
        algo_gflops = [df[(df['matrix_name'] == m) & (df['algorithm'] == algo)]['gflops'].mean() 
                       for m in matrices]
        ax.bar(x + (i - 1) * width, algo_gflops, width, label=algo, color=colors[algo])
    ax.set_xlabel('Matrix')
    ax.set_ylabel('GFlop/s')
    ax.set_title('Performance (GFlop/s) by Matrix')
    ax.set_xticks(x)
    ax.set_xticklabels(matrices, rotation=45)
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Panel 3: Time vs B column count
    ax = axes[1, 0]
    for algo in ['row-wise', 'outer-product', 'blocked']:
        algo_df = df[df['algorithm'] == algo]
        col_perf = algo_df.groupby('b_cols')['mean_time_sec'].mean().sort_index()
        ax.plot(col_perf.index, col_perf.values, marker='o', label=algo, linewidth=2, color=colors[algo])
    ax.set_xlabel('B Column Count (k)')
    ax.set_ylabel('Mean Time (seconds)')
    ax.set_title('Execution Time vs B Columns')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel 4: GFlop/s vs B column count
    ax = axes[1, 1]
    for algo in ['row-wise', 'outer-product', 'blocked']:
        algo_df = df[df['algorithm'] == algo]
        col_perf = algo_df.groupby('b_cols')['gflops'].mean().sort_index()
        ax.plot(col_perf.index, col_perf.values, marker='s', label=algo, linewidth=2, color=colors[algo])
    ax.set_xlabel('B Column Count (k)')
    ax.set_ylabel('GFlop/s')
    ax.set_title('Performance vs B Columns (Dense B)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'spmm_dense_algorithm_comparison.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {plot_path}")
    plt.close()
    
    # Plot 2: Scaling Analysis
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Dense B SpMM: Scaling Analysis", fontsize=14, fontweight='bold')
    
    # Panel 1: Time vs B column count (by matrix)
    ax = axes[0]
    for matrix in sorted(df['matrix_name'].unique()):
        matrix_df = df[df['matrix_name'] == matrix]
        col_perf = matrix_df.groupby('b_cols')['mean_time_sec'].mean().sort_index()
        ax.plot(col_perf.index, col_perf.values, marker='o', label=matrix, linewidth=2)
    ax.set_xlabel('B Column Count (k)')
    ax.set_ylabel('Mean Time (seconds)')
    ax.set_title('Execution Time vs B Columns')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel 2: Output sparsity analysis
    ax = axes[1]
    for matrix in sorted(df['matrix_name'].unique()):
        matrix_df = df[df['matrix_name'] == matrix]
        ax.scatter(matrix_df['nnz_a'], matrix_df['nnz_c'], label=matrix, s=100, alpha=0.7)
    ax.set_xlabel('nnz(A)')
    ax.set_ylabel('nnz(C)')
    ax.set_title('Output Sparsity: nnz(C) vs nnz(A)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'spmm_dense_scaling.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {plot_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark serial SpMM implementations"
    )
    parser.add_argument(
        "--output",
        "-o",
        default="benchmark_spmm_serial_baseline.csv",
        help="Output CSV filename (default: benchmark_spmm_serial_baseline.csv)",
    )
    parser.add_argument(
        "--outdir",
        default="results",
        help="Output directory for results (default: results/)",
    )
    parser.add_argument(
        "--cache-dir",
        default="matrices",
        help="Directory containing .mtx test matrices (default: matrices/)",
    )
    parser.add_argument(
        "--repeats",
        "-r",
        type=int,
        default=3,
        help="Repetitions per benchmark (default: 3)",
    )
    parser.add_argument(
        "--b-cols",
        nargs="+",
        type=int,
        default=[1, 4, 8, 16, 32, 64, 128, 256, 512],
        help="Column counts to test (default: 1 4 8 16 32 64 128 256 512)",
    )
    
    args = parser.parse_args()
    
    benchmark_all_matrices(
        cache_dir=args.cache_dir,
        b_col_counts=args.b_cols,
        repeats=args.repeats,
        output_csv=args.output,
        output_dir=args.outdir,
    )


if __name__ == "__main__":
    main()
