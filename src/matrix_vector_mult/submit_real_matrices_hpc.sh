#!/bin/bash
#SBATCH --job-name=real_matrices_hpc
#SBATCH --output=logs/real_matrices_hpc_%j.log
#SBATCH --error=logs/real_matrices_hpc_%j.err
#SBATCH --partition=cmt
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G

################################################################################
# HPC Benchmarking: OpenMP Thread Scaling on Real SuiteSparse Matrices
#
# Benchmarks sparse matrix-vector multiplication (SpMV) using:
# - Real matrices from local cache (delaunay_n15, bcsstk30, pkustk14, etc.)
# - Strong scaling: 1-64 threads on single node
# - Multiple metrics: Runtime, GFlops, Speedup
#
# Usage:
#   sbatch submit_real_matrices_hpc.sh
#
################################################################################

set -e  # Exit on any error

# Print job information
echo "=========================================="
echo "REAL MATRICES HPC BENCHMARK - Job Started"
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
mkdir -p results_real_matrices

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
echo "Starting Benchmarks on Real Matrices"
echo "=========================================="
echo ""

# Benchmark your local matrices with OpenMP strong scaling
# Matrices: delaunay_n15 (32K×32K), bcsstk30 (28K×28K), pkustk14 (152K×152K)
python benchmark_suite_sparse.py \
    --matrix DIMACS10:delaunay_n15 \
    --threads 1 2 4 8 16 32 64 \
    --repeats 3 \
    --outdir results_real_matrices

python benchmark_suite_sparse.py \
    --matrix DIMACS10:bcsstk30 \
    --threads 1 2 4 8 16 32 64 \
    --repeats 3 \
    --outdir results_real_matrices

python benchmark_suite_sparse.py \
    --matrix Chen:pkustk14 \
    --threads 1 2 4 8 16 32 64 \
    --repeats 3 \
    --outdir results_real_matrices

# Benchmark larger matrix (delaunay_n19: ~524K nodes, ~9M nonzeros)
echo ""
echo "Benchmarking delaunay_n19 (larger matrix)..."
python benchmark_suite_sparse.py \
    --matrix DIMACS10:delaunay_n19 \
    --threads 1 2 4 8 16 32 64 \
    --repeats 3 \
    --outdir results_real_matrices

echo ""
echo "=========================================="
echo "Benchmark Complete"
echo "=========================================="
echo "Results saved to: results_real_matrices/"
echo "End Time: $(date)"
echo "=========================================="
