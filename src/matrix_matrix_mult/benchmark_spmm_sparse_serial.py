"""
Sparse B SpMM Benchmarking Suite (Serial Python Baseline)

Tests sparse matrix-matrix multiplication C = A @ B where:
  - A: Sparse matrix (loaded from Phase 2 or local matrices/)
  - B: Randomly generated sparse matrix (10% sparsity / 90% non-zero)
  - Three custom algorithms + SciPy built-in for comparison

Usage:
    python benchmark_spmm_sparse_serial.py

EXAMPLES:
    1. Default run (all matrices, default columns, 3 repeats):
       python benchmark_spmm_sparse_serial.py

    2. Custom output filename and directory:
       python benchmark_spmm_sparse_serial.py --output results.csv --outdir sparse_results/

    3. Quick test with fewer repeats:
       python benchmark_spmm_sparse_serial.py --repeats 1 --b-cols 4 8 16

    4. Only benchmark delaunay_n15:
       python benchmark_spmm_sparse_serial.py --matrices delaunay_n15

    5. Detailed sparsity testing (compute-bound regime):
       python benchmark_spmm_sparse_serial.py --sparsity 0.05 0.10 0.25 0.50

OPTIONS:
    --output, -o FILE       Output CSV filename (default: benchmark_spmm_sparse_serial.csv)
    --outdir DIR            Output directory for results (default: results/)
    --cache-dir DIR         Directory with .mtx matrices (default: matrices/)
    --repeats, -r N         Repetitions per benchmark (default: 3)
    --b-cols K1 K2 ...      Column counts for sparse B (default: 4 8 16 32 64 128)
    --sparsity P1 P2 ...    Sparsity levels: 0.05 = 5% nonzero, 0.50 = 50% nonzero (default: 0.10)
    --matrices M1 M2 ...    Specific matrices to benchmark (default: all eligible)
    --help, -h              Show this help message

SPARSE MATRIX LOADING:
    - Checks local matrices/ directory first
    - Falls back to Phase 2 (../matrix_vector_mult/matrices/)
    - Falls back to SuiteSparse download if available
"""

import os
import sys
import time
import argparse
import csv
import gc
import numpy as np
from scipy.sparse import csr_matrix, random as sp_random
import pandas as pd

# Try to import matplotlib for plotting
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "matrix_vector_mult"))

from spmm_python import (
    benchmark_spmm_algorithm_sparse_b,
)
from load_real_matrices import download_matrix


# Metadata for Phase 2 matrices
MATRIX_METADATA = {
    "1138_bus": ("HB", "1138_bus"),
    "abb313": ("HB", "abb313"),
    "bcsstk30": ("HB", "bcsstk30"),
    "delaunay_n15": ("DIMACS10", "delaunay_n15"),
    "delaunay_n19": ("DIMACS10", "delaunay_n19"),
    "pkustk14": ("Chen", "pkustk14"),
}


def load_local_matrix(matrix_name, cache_dir="matrices"):
    """
    Load a sparse matrix using three-tier fallback strategy.
    
    Priority:
    1. Local matrices/ directory
    2. Phase 2 ../matrix_vector_mult/matrices/ directory
    3. SuiteSparse download
    
    Returns:
    - (csr_matrix, source_string)
    """
    # Tier 1: Local matrices directory
    local_path = os.path.join(cache_dir, f"{matrix_name}.mtx")
    if os.path.exists(local_path):
        try:
            from scipy.io import mmread
            A = csr_matrix(mmread(local_path))
            return A, f"local ({local_path})"
        except Exception as e:
            print(f"  Warning: Failed to load from {local_path}: {e}")
    
    # Tier 2: Phase 2 matrices directory
    phase2_path = os.path.normpath(
        os.path.join(cache_dir, "..", "..", "matrix_vector_mult", "matrices", f"{matrix_name}.mtx")
    )
    if os.path.exists(phase2_path):
        try:
            from scipy.io import mmread
            A = csr_matrix(mmread(phase2_path))
            return A, f"Phase 2 ({phase2_path})"
        except Exception as e:
            print(f"  Warning: Failed to load from Phase 2: {e}")
    
    # Tier 3: SuiteSparse download
    if matrix_name in MATRIX_METADATA:
        try:
            group, name = MATRIX_METADATA[matrix_name]
            print(f"  {matrix_name} not found in cache; downloading {group}/{name}...")
            filepath = download_matrix(group, name, cache_dir)
            from scipy.io import mmread
            A = csr_matrix(mmread(filepath))
            return A, f"SuiteSparse download ({group}/{name})"
        except Exception as e:
            print(f"  Warning: Failed to download from SuiteSparse: {e}")
    
    return None, "Failed (not found)"


def generate_sparse_b(n, k, sparsity=0.10, dtype=np.float64):
    """
    Generate a random sparse matrix B (n × k) with given sparsity.
    
    Parameters:
    - n: Number of rows
    - k: Number of columns
    - sparsity: Density of non-zero elements (e.g., 0.10 = 10% nonzero, 90% zeros)
    - dtype: Data type (default: float64)
    
    Returns:
    - Sparse matrix in CSR format
    """
    B = sp_random(n, k, density=sparsity, format='csr', dtype=dtype, random_state=42)
    return B


def benchmark_matrix_sparse_b(A, matrix_name, b_cols, sparsities, repeats=3):
    """
    Benchmark one sparse matrix against multiple sparse B configurations.
    
    Parameters:
    - A: Sparse matrix
    - matrix_name: Name for logging
    - b_cols: List of column counts for B
    - sparsities: List of sparsity levels to test (e.g., [0.05, 0.10])
    - repeats: Repetitions per benchmark
    
    Returns:
    - List of result dicts for CSV output
    """
    results = []
    m, n = A.shape
    
    print(f"\nBenchmarking {matrix_name} ({m}×{n}, nnz={A.nnz}):")
    
    for k in b_cols:
        for sparsity in sparsities:
            print(f"\n  B: {n}×{k}, sparsity={sparsity:.2%} (nnz={int(n*k*sparsity)})")
            
            # Generate sparse B
            B = generate_sparse_b(n, k, sparsity=sparsity, dtype=A.dtype)
            
            # Benchmark all algorithms
            for algo in ["row-wise", "outer-product", "blocked", "scipy"]:
                print(f"    Benchmarking {algo}...", end=" ", flush=True)
                
                try:
                    metrics = benchmark_spmm_algorithm_sparse_b(A, B, algo, repeats=repeats)
                    print(f"✓ ({metrics['gflops']:.2f} GFlop/s)")
                    
                    result = {
                        "matrix_name": matrix_name,
                        "m": m,
                        "n": n,
                        "nnz_a": metrics["nnz_a"],
                        "k": k,
                        "nnz_b": metrics["nnz_b"],
                        "sparsity_b": sparsity,
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
                    results.append(result)
                
                except Exception as e:
                    print(f"✗ Error: {e}")
    
    return results


def benchmark_all_matrices_sparse_b(cache_dir, b_cols, sparsities, repeats, output_csv, output_dir):
    """
    Master benchmarking loop for all sparse B tests.
    
    Parameters:
    - cache_dir: Directory with .mtx matrices
    - b_cols: List of column counts
    - sparsities: List of sparsity levels
    - repeats: Repetitions per benchmark
    - output_csv: Output CSV filename
    - output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_csv)
    
    all_results = []
    
    # Load test matrices
    print("Loading matrices...")
    test_matrices = [
        ("1138_bus", load_local_matrix("1138_bus", cache_dir)),
        ("abb313", load_local_matrix("abb313", cache_dir)),
        ("delaunay_n15", load_local_matrix("delaunay_n15", cache_dir)),
    ]
    
    # Benchmark eligible matrices
    for matrix_name, (A, source) in test_matrices:
        if A is None:
            print(f"✗ Could not load {matrix_name}, skipping...")
            continue
        
        # Size filtering
        m, n = A.shape
        max_b_k = max(b_cols)
        max_b_size_mb = (n * max_b_k * 8) / (1024 * 1024)
        
        # Skip very large matrices
        if A.nnz > 2e6:
            print(f"✗ {matrix_name}: nnz={A.nnz:.2e} > 2M threshold, skipping...")
            continue
        if max_b_size_mb > 2000:
            print(f"✗ {matrix_name}: estimated B size {max_b_size_mb:.1f} MB > 2000 MB, skipping...")
            continue
        
        print(f"✓ Loaded from {source}")
        
        # Benchmark
        results = benchmark_matrix_sparse_b(A, matrix_name, b_cols, sparsities, repeats)
        all_results.extend(results)
    
    # Write CSV
    if all_results:
        fieldnames = [
            "matrix_name", "m", "n", "nnz_a", "k", "nnz_b", "sparsity_b",
            "algorithm", "nnz_c",
            "mean_time_sec", "std_time_sec", "min_time_sec", "gflops",
            "memory_a_mb", "memory_b_mb", "memory_c_mb",
        ]
        
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        
        print(f"\n✓ Results written to {output_path}")
        print(f"  Total runs: {len(all_results)}")
        
        # Generate plots if matplotlib is available
        if MATPLOTLIB_AVAILABLE:
            print("\nGenerating plots...")
            plot_results(pd.DataFrame(all_results), output_dir)
        else:
            print("\n⚠ matplotlib not available—skipping plots (install with: pip install matplotlib)")
    else:
        print("\n✗ No results to write")


def plot_results(df: pd.DataFrame, output_dir: str = "results/") -> None:
    """
    Generate plots comparing algorithms and analyzing performance.
    
    Parameters:
    - df: DataFrame with benchmark results
    - output_dir: Directory to save plots
    """
    if not MATPLOTLIB_AVAILABLE:
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot 1: Algorithm Comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Sparse B SpMM: Algorithm Comparison (Custom vs SciPy)", fontsize=14, fontweight='bold')
    
    colors = {
        'row-wise': '#1f77b4',
        'outer-product': '#ff7f0e',
        'blocked': '#2ca02c',
        'scipy': '#d62728'
    }
    
    # Panel 1: Time by matrix
    ax = axes[0, 0]
    matrices = sorted(df['matrix_name'].unique())
    x = np.arange(len(matrices))
    width = 0.2
    for i, algo in enumerate(['row-wise', 'outer-product', 'blocked', 'scipy']):
        algo_times = [df[(df['matrix_name'] == m) & (df['algorithm'] == algo)]['mean_time_sec'].mean() 
                      for m in matrices]
        ax.bar(x + (i - 1.5) * width, algo_times, width, label=algo, color=colors[algo])
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
    for i, algo in enumerate(['row-wise', 'outer-product', 'blocked', 'scipy']):
        algo_gflops = [df[(df['matrix_name'] == m) & (df['algorithm'] == algo)]['gflops'].mean() 
                       for m in matrices]
        ax.bar(x + (i - 1.5) * width, algo_gflops, width, label=algo, color=colors[algo])
    ax.set_xlabel('Matrix')
    ax.set_ylabel('GFlop/s')
    ax.set_title('Performance (GFlop/s) by Matrix')
    ax.set_xticks(x)
    ax.set_xticklabels(matrices, rotation=45)
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Panel 3: Time vs nnz(B)
    ax = axes[1, 0]
    for algo in ['row-wise', 'outer-product', 'blocked', 'scipy']:
        algo_df = df[df['algorithm'] == algo]
        ax.scatter(algo_df['nnz_b'], algo_df['mean_time_sec'], label=algo, color=colors[algo], alpha=0.7, marker='o')
    ax.set_xlabel('nnz(B)')
    ax.set_ylabel('Mean Time (seconds)')
    ax.set_title('Execution Time vs B Sparsity')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel 4: Speedup of SciPy
    ax = axes[1, 1]
    scipy_times = df[df['algorithm'] == 'scipy'].set_index(['matrix_name', 'k', 'sparsity_b'])['mean_time_sec']
    speedups = []
    labels = []
    for algo in ['row-wise', 'outer-product', 'blocked']:
        algo_df = df[df['algorithm'] == algo].set_index(['matrix_name', 'k', 'sparsity_b'])
        speedup = algo_df['mean_time_sec'] / scipy_times
        speedups.append(speedup.dropna().values)
        labels.append(algo)
    
    bp = ax.boxplot(speedups, tick_labels=labels, patch_artist=True)
    for patch, algo in zip(bp['boxes'], labels):
        patch.set_facecolor(colors[algo])
        patch.set_alpha(0.7)
    ax.set_ylabel('Speedup Ratio (Custom / SciPy)')
    ax.set_title('SciPy Speedup over Custom Implementations')
    ax.set_yscale('log')
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='1x (SciPy baseline)')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend()
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'spmm_sparse_algorithm_comparison.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {plot_path}")
    plt.close()
    
    # Plot 2: Scaling Analysis
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("SpMM Scaling: B Column Count and Performance", fontsize=14, fontweight='bold')
    
    scipy_df = df[df['algorithm'] == 'scipy']
    
    # Panel 1: Time vs column count (by matrix)
    ax = axes[0]
    for matrix in sorted(scipy_df['matrix_name'].unique()):
        matrix_df = scipy_df[scipy_df['matrix_name'] == matrix]
        col_perf = matrix_df.groupby('k')['mean_time_sec'].mean().sort_index()
        ax.plot(col_perf.index, col_perf.values, marker='o', label=matrix, linewidth=2)
    ax.set_xlabel('B Column Count (k)')
    ax.set_ylabel('Mean Time (seconds)')
    ax.set_title('Execution Time vs B Columns')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel 2: GFlop/s vs column count
    ax = axes[1]
    for matrix in sorted(scipy_df['matrix_name'].unique()):
        matrix_df = scipy_df[scipy_df['matrix_name'] == matrix]
        col_perf = matrix_df.groupby('k')['gflops'].mean().sort_index()
        ax.plot(col_perf.index, col_perf.values, marker='s', label=matrix, linewidth=2)
    ax.set_xlabel('B Column Count (k)')
    ax.set_ylabel('GFlop/s')
    ax.set_title('Performance vs B Columns')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'spmm_sparse_scaling.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {plot_path}")
    plt.close()


def main():
    """Parse arguments and run benchmarks."""
    parser = argparse.ArgumentParser(
        description="Benchmark sparse B SpMM (Serial Python)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument(
        "--output", "-o",
        default="benchmark_spmm_sparse_serial.csv",
        help="Output CSV filename (default: benchmark_spmm_sparse_serial.csv)",
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
        "--repeats", "-r",
        type=int,
        default=3,
        help="Repetitions per benchmark (default: 3)",
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
        help="Sparsity levels: 0.05 = 5%% nonzero (default: 0.10)",
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Sparse B SpMM Serial Baseline Benchmark")
    print("=" * 70)
    print(f"Output: {args.outdir}/{args.output}")
    print(f"B columns: {args.b_cols}")
    print(f"B sparsity: {args.sparsity}")
    print(f"Repeats: {args.repeats}")
    print("=" * 70)
    
    benchmark_all_matrices_sparse_b(
        cache_dir=args.cache_dir,
        b_cols=args.b_cols,
        sparsities=args.sparsity,
        repeats=args.repeats,
        output_csv=args.output,
        output_dir=args.outdir,
    )


if __name__ == "__main__":
    main()
