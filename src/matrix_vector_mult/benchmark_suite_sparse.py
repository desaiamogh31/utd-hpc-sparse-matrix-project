"""
Benchmark OpenMP SpMV on real sparse matrices from SuiteSparse.

USAGE EXAMPLES:
===============

1. Preset sizes (small, medium, large):
   $ python benchmark_suite_sparse.py --preset small
   $ python benchmark_suite_sparse.py --preset medium
   $ python benchmark_suite_sparse.py --preset large

2. Specify exact matrices by name (GROUP:NAME):
   $ python benchmark_suite_sparse.py --matrix DIMACS10:rgg_15
   $ python benchmark_suite_sparse.py --matrix Florida:FL_t99 Florida:FL_t60

3. Search by criteria (1K-10K nonzeros):
   $ python benchmark_suite_sparse.py --search

4. Larger matrices and higher repetitions:
   $ python benchmark_suite_sparse.py --preset large --threads 1 2 4 8 --repeats 5

5. Full example:
   $ python benchmark_suite_sparse.py \
       --preset medium \
       --threads 1 2 4 8 \
       --repeats 5 \
       --outdir ./results \
       --cache-dir ./matrices

Outputs:
  - CSV file: results/benchmark_results.csv
  - Plot (Runtime): results/plot_runtime.png
  - Plot (Memory): results/plot_memory.png
  - Plot (GFlops): results/plot_gflops.png
  - Plot (Speedup): results/plot_speedup.png

Available presets:
  - small: Small matrices (~1K-10K nonzeros)
  - medium: Medium matrices (~10K-100K nonzeros)
  - large: Large matrices (~100K-1M nonzeros)
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
from scipy.sparse import csr_matrix

import spmv_wrapper
from spmv_python import validate_spmv

try:
    from load_real_matrices import load_matrix_ssgetpy, load_suite_sparse_matrix
    HAS_SSGETPY_MODULE = True
except ImportError:
    HAS_SSGETPY_MODULE = False

try:
    import ssgetpy
    HAS_SSGETPY = True
except ImportError:
    HAS_SSGETPY = False

# Preset matrices by size
PRESETS = {
    "small": {
        "matrices": [("DIMACS10", "rgg_15"), ("Florida", "FL_t99")],
        "description": "Small matrices (1K-10K nonzeros)",
    },
    "medium": {
        "matrices": [("DIMACS10", "delaunay_n13"), ("Florida", "FL_t121")],
        "description": "Medium matrices (10K-100K nonzeros)",
    },
    "large": {
        "matrices": [("DIMACS10", "delaunay_n15"), ("Florida", "FL_t131")],
        "description": "Large matrices (100K-1M nonzeros)",
    },
}


def discover_local_matrices(cache_dir: str = "matrices") -> List[str]:
    """
    Discover .mtx files available locally in the cache directory.
    
    Returns:
    - List of matrix names (without .mtx extension)
    """
    if not os.path.exists(cache_dir):
        return []
    
    mtx_files = [f[:-4] for f in os.listdir(cache_dir) if f.endswith('.mtx')]
    return sorted(mtx_files)


def load_local_matrix(matrix_name: str, cache_dir: str = "matrices") -> Tuple[csr_matrix, Dict]:
    """
    Load a matrix from local cache by name.
    
    Parameters:
    - matrix_name: Name of matrix (without .mtx extension)
    - cache_dir: Directory where matrices are cached
    
    Returns:
    - Tuple of (CSR matrix, info dict)
    """
    from scipy.io import mmread
    
    filepath = os.path.join(cache_dir, f"{matrix_name}.mtx")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Matrix file not found: {filepath}")
    
    print(f"    Loading from cache: {filepath}")
    A = mmread(filepath)
    
    # Convert to CSR if needed
    if not isinstance(A, csr_matrix):
        A = A.tocsr()
    
    info = {
        "name": matrix_name,
        "group": "Local",
        "shape": A.shape,
        "nnz": A.nnz,
        "density": A.nnz / (A.shape[0] * A.shape[1])
    }
    
    return A, info




def benchmark_spmv_omp(
    A: csr_matrix, x: np.ndarray, num_threads: int, repeats: int = 3
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
    matrix_group: str,
    thread_counts: List[int],
    repeats: int = 3,
) -> List[Dict]:
    """
    Benchmark a single matrix across multiple thread counts.
    
    Returns list of result dictionaries.
    """
    print(f"\n  Matrix: {matrix_name}")
    print(f"    Shape: {A.shape}, NNZ: {A.nnz}")
    print(f"    Density: {A.nnz / (A.shape[0] * A.shape[1]):.2e}")
    
    results = []
    
    # Generate input vector
    rng = np.random.default_rng(42)
    x = rng.standard_normal(A.shape[1])
    
    sparsity_ratio = A.nnz / A.shape[0]
    
    print(f"  Benchmarking across {len(thread_counts)} thread counts...")
    for num_threads in thread_counts:
        result = benchmark_spmv_omp(A, x, num_threads, repeats)
        
        results.append({
            "MatrixGroup": matrix_group,
            "MatrixName": matrix_name,
            "N": A.shape[0],
            "NNZ": A.nnz,
            "SparsityRatio": sparsity_ratio,
            "Threads": num_threads,
            "AvgTime_s": result["avg_time_s"],
            "MinTime_s": result["min_time_s"],
            "MaxTime_s": result["max_time_s"],
            "StdTime_s": result["std_time_s"],
            "Memory_MB": result["memory_mb"],
            "GFlops": result["gflops"],
        })
        
        print(f"    Threads={num_threads:2d}: {result['avg_time_s']:8.6f}s, " +
              f"Mem={result['memory_mb']:7.1f}MB, {result['gflops']:7.2f} GFlop/s")
    
    return results


def benchmark_matrices(
    matrices: List[Tuple[str, str]],
    thread_counts: List[int] = None,
    repeats: int = 3,
    cache_dir: str = "matrices",
) -> pd.DataFrame:
    """
    Benchmark multiple matrices specified as (group, name) tuples.
    
    Parameters:
    - matrices: List of (group, name) tuples
    - thread_counts: Thread counts to test
    - repeats: Repetitions per configuration
    - cache_dir: Directory to cache matrices
    
    Returns:
    - DataFrame with benchmark results
    """
    if thread_counts is None:
        thread_counts = [1, 2, 4]
    
    print("\n" + "=" * 100)
    print("BENCHMARKING MATRICES FROM SUITESPARSE")
    print("=" * 100)
    print(f"Matrices to benchmark: {len(matrices)}")
    print(f"Thread counts: {thread_counts}")
    print(f"Repeats: {repeats}")
    print("=" * 100)
    
    all_results = []
    
    for group, matrix_name in matrices:
        print(f"\n  Loading {group}/{matrix_name}...")
        
        A = None
        info = None
        
        # Strategy 1: Try loading from local cache
        try:
            A, info = load_local_matrix(matrix_name, cache_dir)
            print(f"    ✓ Loaded from local cache")
        except FileNotFoundError:
            print(f"    Not in local cache, attempting download...")
            
            # Strategy 2: Try direct download
            try:
                A, info = load_suite_sparse_matrix(group, matrix_name, cache_dir)
                print(f"    ✓ Loaded via direct download")
            except Exception as e:
                print(f"    Direct download failed: {e}")
                
                # Strategy 3: Try ssgetpy fallback
                if HAS_SSGETPY_MODULE and HAS_SSGETPY:
                    print(f"    Trying ssgetpy...")
                    try:
                        result = ssgetpy.search(group=group, name=matrix_name, limit=1)
                        if result:
                            A_tmp, info_tmp = load_matrix_ssgetpy(
                                search_params={"limit": 1},
                                cache_dir=cache_dir
                            )
                            A = A_tmp
                            info = info_tmp
                            print(f"    ✓ Loaded via ssgetpy")
                        else:
                            print(f"    Matrix not found in ssgetpy")
                    except Exception as e2:
                        print(f"    ssgetpy also failed: {e2}")
                else:
                    print(f"    ssgetpy not available")
        
        # If all strategies failed, skip this matrix
        if A is None:
            print(f"    ✗ Could not load matrix {group}/{matrix_name}, skipping...")
            continue
        
        # Benchmark this matrix
        results = benchmark_matrix(A, matrix_name, group, thread_counts, repeats)
        all_results.extend(results)
    
    df = pd.DataFrame(all_results)
    print("\n" + "=" * 100)
    return df


def save_results_and_plots(
    df: pd.DataFrame,
    outdir: str = "results",
) -> None:
    """
    Save results to CSV and create plots for runtime, memory, and GFlops.
    
    Parameters:
    - df: DataFrame with benchmark results
    - outdir: Output directory
    """
    os.makedirs(outdir, exist_ok=True)
    
    # Check if DataFrame is empty
    if df.empty:
        print("\n" + "=" * 100)
        print("ERROR: No benchmark results to plot. Failed to load or download any matrices.")
        print("=" * 100)
        return
    
    # Save CSV
    # Get unique matrices first to create filename suffix
    matrices = df["MatrixName"].unique()
    
    # Create filename suffix based on matrix names
    if len(matrices) == 1:
        filename_suffix = f"_{matrices[0]}"
    else:
        filename_suffix = f"_{len(matrices)}matrices"
    
    csv_path = os.path.join(outdir, f"benchmark_results{filename_suffix}.csv")
    df.to_csv(csv_path, index=False)
    print(f"✓ CSV saved: {csv_path}")
    
    # Get thread counts
    thread_counts = sorted(df["Threads"].unique())
    
    # Helper function to get matrix info string
    def get_matrix_info(matrix_name, df):
        mat_data = df[df["MatrixName"] == matrix_name].iloc[0]
        n = int(mat_data["N"])
        nnz = int(mat_data["NNZ"])
        return f"{matrix_name} (N={n:,}, NNZ={nnz:,})"
    
    # --- Plot 1: Runtime ---
    fig, ax = plt.subplots(figsize=(11, 7))
    for matrix in matrices:
        mat_data = df[df["MatrixName"] == matrix].sort_values("Threads")
        matrix_label = get_matrix_info(matrix, df)
        ax.plot(
            mat_data["Threads"], 
            mat_data["AvgTime_s"],
            marker='o', 
            label=matrix_label, 
            linewidth=2.5, 
            markersize=8
        )
    ax.set_xlabel("Number of Threads", fontsize=16, fontweight='bold')
    ax.set_ylabel("Runtime (seconds)", fontsize=16, fontweight='bold')
    ax.set_title("SpMV Runtime vs Thread Count", fontsize=18, fontweight='bold')
    ax.set_yscale('log')
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.legend(fontsize=14, loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plot_path = os.path.join(outdir, f"plot_runtime{filename_suffix}.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Plot saved: {plot_path}")
    
    # --- Plot 2: Memory ---
    fig, ax = plt.subplots(figsize=(11, 7))
    for matrix in matrices:
        mat_data = df[df["MatrixName"] == matrix].sort_values("Threads")
        matrix_label = get_matrix_info(matrix, df)
        ax.plot(
            mat_data["Threads"], 
            mat_data["Memory_MB"],
            marker='s', 
            label=matrix_label, 
            linewidth=2.5, 
            markersize=8
        )
    ax.set_xlabel("Number of Threads", fontsize=16, fontweight='bold')
    ax.set_ylabel("Memory (MB)", fontsize=16, fontweight='bold')
    ax.set_title("SpMV Memory Usage vs Thread Count", fontsize=18, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.legend(fontsize=14, loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plot_path = os.path.join(outdir, f"plot_memory{filename_suffix}.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Plot saved: {plot_path}")
    
    # --- Plot 3: GFlops ---
    fig, ax = plt.subplots(figsize=(11, 7))
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
    ax.set_title("SpMV Performance vs Thread Count", fontsize=18, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.legend(fontsize=14, loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plot_path = os.path.join(outdir, f"plot_gflops{filename_suffix}.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Plot saved: {plot_path}")
    
    # --- Plot 4: Speedup ---
    fig, ax = plt.subplots(figsize=(11, 7))
    for matrix in matrices:
        mat_data = df[df["MatrixName"] == matrix].sort_values("Threads")
        matrix_label = get_matrix_info(matrix, df)
        baseline = mat_data[mat_data["Threads"] == thread_counts[0]]["AvgTime_s"].values
        if len(baseline) > 0:
            speedup = baseline[0] / mat_data["AvgTime_s"].values
            ax.plot(
                mat_data["Threads"], 
                speedup,
                marker='D', 
                label=matrix_label, 
                linewidth=2.5, 
                markersize=8
            )
    # Ideal speedup
    ax.plot(thread_counts, thread_counts, 'k--', label='Ideal', linewidth=1.5, alpha=0.7)
    ax.set_xlabel("Number of Threads", fontsize=16, fontweight='bold')
    ax.set_ylabel("Speedup", fontsize=16, fontweight='bold')
    ax.set_title("Speedup vs Thread Count", fontsize=18, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.legend(fontsize=14, loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plot_path = os.path.join(outdir, f"plot_speedup{filename_suffix}.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Plot saved: {plot_path}")
    
    print("\n" + "=" * 100)
    print("All results and plots saved successfully!")
    print("=" * 100)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark OpenMP SpMV on real sparse matrices from SuiteSparse.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    # Preset sizes
    parser.add_argument(
        "--preset",
        type=str,
        choices=["small", "medium", "large"],
        default="small",
        help="Use preset matrices (small, medium, or large). Default: small",
    )
    
    # Specific matrices
    parser.add_argument(
        "--matrix",
        type=str,
        nargs='+',
        help="Specific matrices to benchmark (format: 'GROUP:NAME' e.g., 'DIMACS10:rgg_15 Florida:FL_t99')",
    )
    
    # Search by criteria
    parser.add_argument(
        "--search",
        action="store_true",
        help="Search by criteria instead of using presets (requires ssgetpy)",
    )
    parser.add_argument(
        "--nnz-min",
        type=int,
        default=1000,
        help="Minimum nonzeros for search (used with --search, default: 1000)",
    )
    parser.add_argument(
        "--nnz-max",
        type=int,
        default=10000,
        help="Maximum nonzeros for search (used with --search, default: 10000)",
    )
    parser.add_argument(
        "--spd",
        action="store_true",
        help="Search for symmetric positive definite matrices (used with --search)",
    )
    
    # Common options
    parser.add_argument(
        "--threads",
        type=int,
        nargs='+',
        default=[1, 2, 4],
        help="Thread counts to test (default: 1 2 4)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Repetitions per configuration (default: 3)",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="results",
        help="Output directory for results and plots (default: results)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="matrices",
        help="Directory to cache downloaded matrices (default: matrices)",
    )
    
    # List available local matrices
    parser.add_argument(
        "--list-local",
        action="store_true",
        help="List available matrices in local cache and exit",
    )
    
    args = parser.parse_args()
    
    # Handle --list-local
    if args.list_local:
        local_matrices = discover_local_matrices(args.cache_dir)
        if not local_matrices:
            print(f"No matrices found in '{args.cache_dir}/'")
            print(f"Create the directory or download matrices to get started.")
        else:
            print(f"\nAvailable local matrices in '{args.cache_dir}/':")
            print("=" * 100)
            for i, matrix_name in enumerate(local_matrices, 1):
                print(f"  {i}. {matrix_name}")
            print("=" * 100)
            print(f"\nTo benchmark a local matrix:")
            print(f"  python benchmark_suite_sparse.py --matrix Local:{local_matrices[0]}")
        sys.exit(0)
    
    # Determine which matrices to benchmark
    matrices = None
    
    if args.matrix:
        # Parse specific matrices
        matrices = []
        for m in args.matrix:
            if ':' in m:
                group, name = m.split(':', 1)
                matrices.append((group, name))
            else:
                print(f"ERROR: Invalid matrix format '{m}'. Use 'GROUP:NAME'")
                sys.exit(1)
    elif args.search:
        # Search by criteria - use ssgetpy
        if not HAS_SSGETPY_MODULE or not HAS_SSGETPY:
            print("ERROR: ssgetpy not installed!")
            print("Install with: pip install ssgetpy")
            sys.exit(1)
        
        print("\n" + "=" * 100)
        print("SEARCHING FOR MATRIX IN SUITESPARSE")
        print("=" * 100)
        print(f"Search criteria:")
        print(f"  NNZ range: {args.nnz_min:,} - {args.nnz_max:,}")
        print(f"  SPD: {args.spd}")
        print("=" * 100)
        
        A, info = load_matrix_ssgetpy(
            search_params={
                "nzbounds": (args.nnz_min, args.nnz_max),
                "isspd": args.spd,
                "limit": 1,
            },
            cache_dir=args.cache_dir
        )
        
        matrix_name = info.get("name", "unknown")
        matrix_group = info.get("group", "unknown")
        matrices = [(matrix_group, matrix_name)]
    else:
        # Use preset
        if args.preset not in PRESETS:
            print(f"ERROR: Unknown preset '{args.preset}'")
            print(f"Available presets: {', '.join(PRESETS.keys())}")
            sys.exit(1)
        
        preset_info = PRESETS[args.preset]
        matrices = preset_info["matrices"]
        print(f"\nUsing preset: {args.preset} ({preset_info['description']})")
        print(f"Matrices: {matrices}")
    
    if not matrices:
        print("ERROR: No matrices to benchmark")
        sys.exit(1)
    
    # Benchmark
    df = benchmark_matrices(
        matrices=matrices,
        thread_counts=args.threads,
        repeats=args.repeats,
        cache_dir=args.cache_dir,
    )
    
    # Save results and create plots
    save_results_and_plots(df, args.outdir)
    
    # Print summary
    print("\nBENCHMARK SUMMARY:")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
