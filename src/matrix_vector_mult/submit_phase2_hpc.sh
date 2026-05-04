#!/bin/bash
#SBATCH --job-name=phase2_hpc_benchmark
#SBATCH --output=logs/phase2_hpc_benchmark_%j.log
#SBATCH --error=logs/phase2_hpc_benchmark_%j.err
#SBATCH --partition=cmt
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=32G

################################################################################
# Phase 2 HPC Benchmarking: OpenMP Thread Scaling (1-64 threads)
#
# This script benchmarks sparse matrix-vector multiplication (SpMV) on a single
# HPC node using OpenMP for shared-memory parallelism.
#
# Features:
# - Random matrices: 100K to 1M+ nnz with multiple sparsity levels
# - Strong scaling: 1-64 threads on single node
# - Multiple metrics: Runtime, Speedup, GFlops, Efficiency
# - Automatic visualization: 4 plots (speedup, gflops, runtime, efficiency)
#
# Usage:
#   sbatch submit_phase2_hpc.sh
#
# To customize parameters, edit the python command at the bottom or run:
#   srun -N1 -n1 -c64 -t04:00:00 python benchmark_phase2_hpc.py ...
#
################################################################################

set -e  # Exit on any error

# Print job information
echo "=========================================="
echo "PHASE 2 HPC BENCHMARK - Job Started"
echo "=========================================="
echo "Job ID:          $SLURM_JOB_ID"
echo "Job Name:        $SLURM_JOB_NAME"
echo "Hostname:        $(hostname)"
echo "Num CPUs:        $SLURM_CPUS_PER_TASK"
echo "Memory:          $SLURM_MEM_PER_NODE"
echo "Time Limit:      $SLURM_TIME_LIMIT"
echo "Start Time:      $(date)"
echo "=========================================="

# Create logs and results directories if they don't exist
mkdir -p logs
mkdir -p results_hpc_phase2

# Record system information
echo ""
echo "System Information:"
uname -a
echo "CPU Count: $(nproc)"
echo "Memory: $(free -h | head -2)"
echo ""

# Load modules if needed (adjust for your cluster)
module load gcc || true
#module load python || true

# Compile OpenMP library if needed
echo "Checking C++ OpenMP library..."
if [ ! -f "spmv_openmp.so" ]; then
    echo "Compiling spmv_openmp.cpp..."
    python build.py
    echo "✓ Compilation complete"
else
    echo "✓ spmv_openmp.so already compiled"
fi

echo ""
echo "=========================================="
echo "Starting Benchmarks"
echo "=========================================="
echo ""

# Run benchmark with strong scaling:
# - Matrix sizes: 100K, 500K, 1M
# - NNZ ratios: 5, 10, 20 (nonzeros per row)
# - Thread counts: 1-64 (powers of 2 and intermediates)
# - 3 repetitions for statistical significance

python benchmark_phase2_hpc.py \
    --matrix-sizes 10000 50000 100000 \
    --nnz-ratios 5.0 10.0 20.0 \
    --threads 1 2 4 8 16 32 64 \
    --repeats 3 \
    --outdir results_hpc_phase2

# Optional: Include real matrix validation (uncomment to enable)
# This adds bcsstk30 from SuiteSparse for reference comparison
# python benchmark_phase2_hpc.py \
#     --matrix-sizes 100000 500000 1000000 \
#     --nnz-ratios 5.0 10.0 20.0 \
#     --threads 1 2 4 8 16 32 64 \
#     --repeats 3 \
#     --include-real-matrices \
#     --outdir results_hpc_phase2

echo ""
echo "=========================================="
echo "Benchmark Complete"
echo "=========================================="
echo "End Time:        $(date)"
echo "=========================================="
echo "Results Location:"
echo "  CSV:   results_hpc_phase2/phase2_hpc_results.csv"
echo "  Plots: results_hpc_phase2/plot_*.png"
echo "  Logs:  logs/phase2_hpc_benchmark_$SLURM_JOB_ID.log"
echo "=========================================="

echo "=========================================="
echo "End time: $(date)"
echo "=========================================="
