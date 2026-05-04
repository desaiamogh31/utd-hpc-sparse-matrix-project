#!/bin/bash
#SBATCH --job-name=spmm_sparse_hpc
#SBATCH --output=logs/spmm_sparse_hpc_%j.log
#SBATCH --error=logs/spmm_sparse_hpc_%j.err
#SBATCH --partition=cmt
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G

################################################################################
# HPC Benchmarking: Sparse-B SpMM OpenMP on Real Sparse Matrices
#
# Benchmarks sparse matrix-matrix multiplication C = A @ B using:
# - OpenMP custom kernels + SciPy baseline
# - Real matrices from the local cache / Phase 2 cache
# - Strong scaling across a single HPC node
# - The larger matrices that are skipped in the laptop benchmark
#
# Usage:
#   sbatch submit_spmm_sparse_hpc.sh
#
################################################################################

set -e

echo "=========================================="
echo "SPMM SPARSE-B HPC BENCHMARK - Job Started"
echo "=========================================="
echo "Job ID:          $SLURM_JOB_ID"
echo "Job Name:        $SLURM_JOB_NAME"
echo "Hostname:        $(hostname)"
echo "Num CPUs:        $SLURM_CPUS_PER_TASK"
echo "Memory:          $SLURM_MEM_PER_NODE"
echo "Time Limit:      $SLURM_TIME_LIMIT"
echo "Start Time:      $(date)"
echo "=========================================="

mkdir -p logs
mkdir -p results_hpc_spmm

echo ""
echo "System Information:"
uname -a
echo "CPU Count: $(nproc)"
echo "Memory: $(free -h | head -2 || true)"
echo ""

# Load cluster modules if available
module load gcc || true
# module load python || true

# Keep matplotlib writable on shared systems
export MPLCONFIGDIR="${PWD}/.matplotlib"
mkdir -p "$MPLCONFIGDIR"

echo "Checking C++ OpenMP library..."
if [ ! -f "spmm_openmp.so" ]; then
    echo "Compiling spmm_openmp.cpp..."
    python build.py
    echo "✓ Compilation complete"
else
    echo "✓ spmm_openmp.so already compiled"
fi

echo ""
echo "=========================================="
echo "Starting Sparse-B SpMM HPC Benchmark"
echo "=========================================="
echo ""

python benchmark_spmm_sparse_hpc.py \
    --threads 1 2 4 8 16 32 64 \
    --b-cols 4 8 16 \
    --sparsity 0.10 \
    --repeats 3 \
    --outdir results_hpc_spmm

echo ""
echo "=========================================="
echo "Benchmark Complete"
echo "=========================================="
echo "Results saved to: results_hpc_spmm/"
echo "End Time: $(date)"
echo "=========================================="
