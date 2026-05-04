"""
Plot OpenMP SpMV benchmarking results from run_spmv_omp.sh output.

Reads spmv_openmp_runtimes.txt and creates visualizations showing:
- Execution time vs matrix size and threads
- GFlop/s vs matrix size and threads
- Speedup vs thread count (relative to 1 thread)
- Efficiency vs thread count
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def load_omp_results(csv_path: str = "results/spmv_openmp_runtimes.txt") -> pd.DataFrame:
    """Load OpenMP benchmark results from CSV."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Results file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    # Handle potential whitespace in column names
    df.columns = df.columns.str.strip()
    return df


def compute_speedup_efficiency(df: pd.DataFrame) -> pd.DataFrame:
    """Compute speedup and efficiency relative to single-thread baseline."""
    df_with_metrics = df.copy()
    df_with_metrics["Speedup"] = 1.0
    df_with_metrics["Efficiency"] = 1.0
    
    # For each matrix size, compute speedup relative to 1 thread
    for matrix_size in df["MatrixSize"].unique():
        size_data = df[df["MatrixSize"] == matrix_size]
        baseline_time = size_data[size_data["Threads"] == 1]["Avg_Time_s"].values
        
        if len(baseline_time) > 0:
            baseline_time = baseline_time[0]
            
            # Compute speedup and efficiency for all thread counts
            mask = df_with_metrics["MatrixSize"] == matrix_size
            speedups = baseline_time / df_with_metrics.loc[mask, "Avg_Time_s"]
            threads = df_with_metrics.loc[mask, "Threads"]
            efficiencies = speedups / threads
            
            df_with_metrics.loc[mask, "Speedup"] = speedups.values
            df_with_metrics.loc[mask, "Efficiency"] = efficiencies.values
    
    return df_with_metrics


def plot_results(df: pd.DataFrame, output_dir: str = "results") -> None:
    """Create separate 4-panel visualizations for each NNZ ratio."""
    
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if NNZ_Ratio column exists
    if "NNZ_Ratio" not in df.columns:
        # Add default ratio if not present
        df["NNZ_Ratio"] = 5.0
    
    # Get unique ratios
    ratios = sorted(df["NNZ_Ratio"].unique())
    
    for ratio in ratios:
        # Filter data for this ratio
        df_ratio = df[df["NNZ_Ratio"] == ratio].copy()
        
        sizes = sorted(df_ratio["MatrixSize"].unique())
        thread_counts = sorted(df_ratio["Threads"].unique())
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Color and marker scheme
        colors = {1: '#1f77b4', 2: '#ff7f0e', 4: '#2ca02c', 8: '#d62728', 16: '#9467bd'}
        markers = {1: 'o', 2: 's', 4: '^', 8: 'D', 16: 'v'}
        
        # Panel 1: Execution Time vs Matrix Size
        ax = axes[0, 0]
        for threads in thread_counts:
            thread_data = df_ratio[df_ratio["Threads"] == threads].sort_values("MatrixSize")
            ax.plot(thread_data["MatrixSize"], thread_data["Avg_Time_s"],
                   marker=markers.get(threads, 'o'), label=f"{threads} thread(s)", 
                   linewidth=2, markersize=8, color=colors.get(threads, '#000000'))
        
        ax.set_xlabel("Matrix Dimension (n)", fontsize=11, fontweight='bold')
        ax.set_ylabel("Time (seconds)", fontsize=11, fontweight='bold')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title(f"Execution Time vs Matrix Size (nnz_ratio={ratio})", 
                    fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Panel 2: GFlop/s vs Matrix Size
        ax = axes[0, 1]
        for threads in thread_counts:
            thread_data = df_ratio[df_ratio["Threads"] == threads].sort_values("MatrixSize")
            ax.plot(thread_data["MatrixSize"], thread_data["GFlops"],
                   marker=markers.get(threads, 'o'), label=f"{threads} thread(s)", 
                   linewidth=2, markersize=8, color=colors.get(threads, '#000000'))
        
        ax.set_xlabel("Matrix Dimension (n)", fontsize=11, fontweight='bold')
        ax.set_ylabel("GFlop/s", fontsize=11, fontweight='bold')
        ax.set_xscale('log')
        ax.set_title(f"Performance vs Matrix Size (nnz_ratio={ratio})", 
                    fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Panel 3: Speedup vs Thread Count
        ax = axes[1, 0]
        for size in sizes:
            size_data = df_ratio[df_ratio["MatrixSize"] == size].sort_values("Threads")
            ax.plot(size_data["Threads"], size_data["Speedup"],
                   marker='o', label=f"n={size}", linewidth=2, markersize=8)
        
        # Ideal speedup line (y=x)
        ideal_threads = sorted(thread_counts)
        ax.plot(ideal_threads, ideal_threads, 'k--', label='Ideal', 
               linewidth=2, alpha=0.5)
        
        ax.set_xlabel("Number of Threads", fontsize=11, fontweight='bold')
        ax.set_ylabel("Speedup", fontsize=11, fontweight='bold')
        ax.set_title(f"Speedup vs Thread Count (nnz_ratio={ratio})", 
                    fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)
        
        # Panel 4: Efficiency vs Thread Count
        ax = axes[1, 1]
        for size in sizes:
            size_data = df_ratio[df_ratio["MatrixSize"] == size].sort_values("Threads")
            ax.plot(size_data["Threads"], size_data["Efficiency"],
                   marker='o', label=f"n={size}", linewidth=2, markersize=8)
        
        # Ideal efficiency line (y=1.0)
        ax.axhline(y=1.0, color='k', linestyle='--', linewidth=2, alpha=0.5, 
                  label='Ideal (1.0)')
        
        ax.set_xlabel("Number of Threads", fontsize=11, fontweight='bold')
        ax.set_ylabel("Efficiency", fontsize=11, fontweight='bold')
        ax.set_title(f"Parallel Efficiency vs Thread Count (nnz_ratio={ratio})", 
                    fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save with ratio in filename
        ratio_str = f"{ratio:.1f}".replace('.', '_')
        plot_path = os.path.join(output_dir, f"spmv_openmp_plot_ratio_{ratio_str}.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"✓ Plot saved to {plot_path}")
        plt.close()


def print_summary(df: pd.DataFrame) -> None:
    """Print a summary of results."""
    print("\n" + "=" * 80)
    print("OPENMP SPMV BENCHMARK SUMMARY")
    print("=" * 80)
    print("\nResults by Matrix Size:")
    print("-" * 80)
    
    for size in sorted(df["MatrixSize"].unique()):
        print(f"\nMatrix Size: {size}")
        size_data = df[df["MatrixSize"] == size].sort_values("Threads")
        print(f"  {'Threads':<10}{'Time(s)':<15}{'GFlops':<15}{'Speedup':<12}{'Efficiency':<12}")
        print(f"  {'-'*60}")
        for _, row in size_data.iterrows():
            print(f"  {int(row['Threads']):<10}{row['Avg_Time_s']:<15.6f}{row['GFlops']:<15.2f}"
                  f"{row['Speedup']:<12.2f}{row['Efficiency']:<12.2f}")
    
    print("\n" + "=" * 80)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Plot OpenMP SpMV benchmark results from run_spmv_omp.sh"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="results/spmv_openmp_runtimes.txt",
        help="Path to results CSV file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results",
        help="Output directory for plots",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print summary statistics",
    )
    
    args = parser.parse_args()
    
    # Load results
    print(f"Loading results from {args.input}...")
    df = load_omp_results(args.input)
    
    # Compute speedup and efficiency
    print("Computing speedup and efficiency metrics...")
    df = compute_speedup_efficiency(df)
    
    # Print summary if requested
    if args.summary:
        print_summary(df)
    
    # Create plots (separate for each NNZ ratio)
    print(f"Creating visualizations...")
    plot_results(df, args.output)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
