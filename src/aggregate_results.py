"""
Results aggregation and visualization for sparse matrix assembly benchmarks.

Compares performance across Laplacian, Symmetric, and Asymmetric matrices.
Generates unified plots and summary reports.
"""

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_benchmark_csvs(results_dir: str = "src/assembly/results"):
    """Load all benchmark CSV files from results directory."""
    
    results = {
        "laplacian": {},
        "symmetric": {},
        "asymmetric": {},
    }
    
    if not os.path.exists(results_dir):
        print(f"Results directory '{results_dir}' not found.")
        return results
    
    # Load single-matrix benchmarks
    for matrix_type in ["laplacian", "symmetric", "asymmetric"]:
        # Assembly benchmarks
        asm_pattern = f"{results_dir}/{matrix_type}_matrix_benchmark.csv"
        if glob.glob(asm_pattern):
            results[matrix_type]["assembly"] = pd.read_csv(glob.glob(asm_pattern)[0])
            print(f"Loaded: {matrix_type} assembly benchmark")
        
        # Scaling benchmarks
        scaling_pattern = f"{results_dir}/{matrix_type}_*scaling.csv"
        scaling_files = glob.glob(scaling_pattern)
        for sf in scaling_files:
            results[matrix_type]["scaling"] = pd.read_csv(sf)
            print(f"Loaded: {os.path.basename(sf)}")
        
        # Conversion benchmarks
        conv_pattern = f"{results_dir}/{matrix_type}_*conversion*.csv"
        conv_files = glob.glob(conv_pattern)
        for cf in conv_files:
            results[matrix_type]["conversion"] = pd.read_csv(cf)
            print(f"Loaded: {os.path.basename(cf)}")
    
    return results


def plot_assembly_comparison(n:int, results: dict, outdir: str = "src/assembly/results") -> None:
    """Compare assembly performance across matrix types."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Collect data for each matrix type
    matrix_types = ["laplacian", "symmetric", "asymmetric"]
    colors = {'COO': '#1f77b4', 'LIL': '#ff7f0e', 'CSR': '#2ca02c', 'CSC': '#d62728'}
    
    for idx, (ax, metric) in enumerate(zip(axes.flat[:3], 
                                           ["Avg(s)", "Peak_Memory_MB", "NNZ"])):
        for matrix_type in matrix_types:
            if "assembly" in results[matrix_type]:
                df = results[matrix_type]["assembly"]
                if metric in df.columns:
                    x_pos = np.arange(len(df))
                    width = 0.25
                    offset = (list(matrix_types).index(matrix_type) - 1) * width
                    
                    ax.bar(x_pos + offset, df[metric], width, 
                          label=matrix_type.capitalize(), alpha=0.8)
        
        ax.set_ylabel(metric, fontsize=11)
        ax.set_xlabel('Format', fontsize=11)
        ax.set_title(f'Assembly {metric} Comparison', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Set x-axis labels to format names
        if "assembly" in results[matrix_types[0]]:
            ax.set_xticks(np.arange(len(results[matrix_types[0]]["assembly"])))
            ax.set_xticklabels(results[matrix_types[0]]["assembly"]["Format"], rotation=45)
    
    # Scaling comparison - Time
    ax = axes.flat[3]
    for matrix_type in matrix_types:
        if "scaling" in results[matrix_type]:
            df = results[matrix_type]["scaling"]
            if "N" in df.columns and "COO_Avg_s" in df.columns:
                ax.plot(df["N"], df["COO_Avg_s"], marker='o', label=f"{matrix_type.capitalize()} COO")
    
    ax.set_xlabel('Matrix Dimension (n)', fontsize=11)
    ax.set_ylabel('Assembly Time (s)', fontsize=11)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title(f'Scaling: COO Assembly Time, n={n}', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = f"{outdir}/comparison_assembly.png"
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved comparison plot to {plot_path}")
    plt.close()


def plot_memory_comparison(results: dict, outdir: str = "src/assembly/results") -> None:
    """Compare memory usage across scaling studies."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Scaling memory comparison
    for matrix_type in ["laplacian", "symmetric", "asymmetric"]:
        if "scaling" in results[matrix_type]:
            df = results[matrix_type]["scaling"]
            if "N" in df.columns and "COO_Peak_Memory_MB" in df.columns:
                ax1.plot(df["N"], df["COO_Peak_Memory_MB"], marker='o', 
                        label=f"{matrix_type.capitalize()}", linewidth=2)
    
    ax1.set_xlabel('Matrix Dimension (n)', fontsize=12)
    ax1.set_ylabel('Peak Memory (MB)', fontsize=12)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_title(f'Memory Scaling: COO Format, n={n}', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # LIL memory comparison
    for matrix_type in ["laplacian", "symmetric", "asymmetric"]:
        if "scaling" in results[matrix_type]:
            df = results[matrix_type]["scaling"]
            if "N" in df.columns and "LIL_Peak_Memory_MB" in df.columns:
                ax2.plot(df["N"], df["LIL_Peak_Memory_MB"], marker='s', 
                        label=f"{matrix_type.capitalize()}", linewidth=2)
    
    ax2.set_xlabel('Matrix Dimension (n)', fontsize=12)
    ax2.set_ylabel('Peak Memory (MB)', fontsize=12)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_title(f'Memory Scaling: LIL Format, n={n}', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = f"{outdir}/comparison_memory.png"
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved memory comparison plot to {plot_path}")
    plt.close()


def generate_summary_report(results: dict, outdir: str = "src/assembly/results") -> None:
    """Generate a text summary of all benchmarks."""
    
    report_path = f"{outdir}/BENCHMARK_SUMMARY.txt"
    os.makedirs(outdir, exist_ok=True)
    
    with open(report_path, "w") as f:
        f.write("=" * 100 + "\n")
        f.write("SPARSE MATRIX ASSEMBLY BENCHMARK SUMMARY\n")
        f.write("=" * 100 + "\n\n")
        
        # Assembly benchmarks
        f.write("SINGLE-MATRIX ASSEMBLY BENCHMARKS\n")
        f.write("-" * 100 + "\n\n")
        
        for matrix_type in ["laplacian", "symmetric", "asymmetric"]:
            f.write(f"{matrix_type.upper()} MATRIX\n")
            f.write("-" * 50 + "\n")
            
            if "assembly" in results[matrix_type]:
                df = results[matrix_type]["assembly"]
                f.write(df.to_string(index=False))
                f.write("\n\n")
                
                # Summary stats
                if "Avg(s)" in df.columns:
                    f.write(f"Fastest format: {df.loc[df['Avg(s)'].idxmin(), 'Format']}\n")
                    f.write(f"Avg time range: {df['Avg(s)'].min():.6f}s - {df['Avg(s)'].max():.6f}s\n")
                
                if "Peak_Memory_MB" in df.columns:
                    f.write(f"Memory range: {df['Peak_Memory_MB'].min():.2f}MB - {df['Peak_Memory_MB'].max():.2f}MB\n")
                
                f.write("\n")
        
        # Scaling benchmarks
        f.write("\n" + "=" * 100 + "\n")
        f.write("SCALING BENCHMARKS\n")
        f.write("-" * 100 + "\n\n")
        
        for matrix_type in ["laplacian", "symmetric", "asymmetric"]:
            f.write(f"{matrix_type.upper()} MATRIX - SCALING RESULTS\n")
            f.write("-" * 50 + "\n")
            
            if "scaling" in results[matrix_type]:
                df = results[matrix_type]["scaling"]
                f.write(df.to_string(index=False))
                f.write("\n\n")
        
        # Conversion benchmarks
        f.write("\n" + "=" * 100 + "\n")
        f.write("FORMAT CONVERSION BENCHMARKS\n")
        f.write("-" * 100 + "\n\n")
        
        for matrix_type in ["laplacian", "symmetric", "asymmetric"]:
            f.write(f"{matrix_type.upper()} MATRIX - CONVERSIONS\n")
            f.write("-" * 50 + "\n")
            
            if "conversion" in results[matrix_type]:
                df = results[matrix_type]["conversion"]
                f.write(df.to_string(index=False))
                f.write("\n\n")
        
        f.write("\n" + "=" * 100 + "\n")
        f.write("END OF SUMMARY\n")
    
    print(f"Saved summary report to {report_path}")


def main():
    """Main aggregation workflow."""
    
    print("=" * 80)
    print("SPARSE MATRIX BENCHMARK RESULTS AGGREGATION")
    print("=" * 80 + "\n")
    
    # Load results
    results = load_benchmark_csvs("src/assembly/results")
    
    print("\nGenerating visualizations...\n")
    
    # Generate plots
    plot_assembly_comparison(2000, results)
    plot_memory_comparison(2000, results)
    
    # Generate summary report
    generate_summary_report(results)
    
    print("\n" + "=" * 80)
    print("AGGREGATION COMPLETE")
    print("=" * 80)
    print("\nOutput files:")
    print("  - src/assembly/results/comparison_assembly.png")
    print("  - src/assembly/results/comparison_memory.png")
    print("  - src/assembly/results/BENCHMARK_SUMMARY.txt")


if __name__ == "__main__":
    main()
