#!/bin/bash
#SBATCH --job-name=spmm_sparse_mpi
#SBATCH --output=logs/spmm_sparse_mpi_%j.log
#SBATCH --error=logs/spmm_sparse_mpi_%j.err
#SBATCH --partition=cmt
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G

################################################################################
# HPC Benchmarking: Sparse-B SpMM with MPI
#
# Mirrors the OpenMP sparse-B SpMM benchmark, but uses MPI process scaling
# through repeated mpirun launches from a single SLURM job.
#
# Usage:
#   sbatch submit_spmm_sparse_mpi.sh
#
################################################################################

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
MPI_LAUNCHER="${MPI_LAUNCHER:-srun}"

echo "=========================================="
echo "SPMM SPARSE-B MPI BENCHMARK - Job Started"
echo "=========================================="
echo "Job ID:          $SLURM_JOB_ID"
echo "Job Name:        $SLURM_JOB_NAME"
echo "Hostname:        $(hostname)"
echo "Num Tasks:       $SLURM_NTASKS"
echo "CPUs per Task:   $SLURM_CPUS_PER_TASK"
echo "Working Dir:     $SCRIPT_DIR"
echo "Python:          $PYTHON_BIN"
echo "MPI Launcher:    $MPI_LAUNCHER"
echo "Memory:          $SLURM_MEM_PER_NODE"
echo "Time Limit:      $SLURM_TIME_LIMIT"
echo "Start Time:      $(date)"
echo "=========================================="

cd "$SCRIPT_DIR"

mkdir -p logs
mkdir -p results_hpc_spmm_mpi

echo ""
echo "System Information:"
uname -a
echo "CPU Count: $(nproc)"
echo "Memory: $(free -h | head -2 || true)"
echo ""

# Load cluster modules only if your site requires them.
# module load gcc
# module load openmpi

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="${PWD}/.matplotlib"
mkdir -p "$MPLCONFIGDIR"

echo ""
echo "=========================================="
echo "Starting Sparse-B SpMM MPI Benchmark"
echo "=========================================="
echo ""

"$PYTHON_BIN" "$SCRIPT_DIR/benchmark_spmm_sparse_mpi.py" \
    --processes 1 2 4 8 16 \
    --b-cols 4 8 16 \
    --sparsity 0.10 \
    --repeats 3 \
    --mpi-launcher "$MPI_LAUNCHER" \
    --outdir results_hpc_spmm_mpi

echo ""
echo "=========================================="
echo "Benchmark Complete"
echo "=========================================="
echo "Results saved to: results_hpc_spmm_mpi/"
echo "End Time: $(date)"
echo "=========================================="
