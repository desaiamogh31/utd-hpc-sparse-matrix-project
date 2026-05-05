#!/bin/bash
#SBATCH --job-name=spmm_sparse_mpi
#SBATCH --output=spmm_sparse_mpi_%j.log
#SBATCH --error=spmm_sparse_mpi_%j.err
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
PYTHON_BIN="${PYTHON_BIN:-$(command -v python || command -v python3 || true)}"
MPI_LAUNCHER="${MPI_LAUNCHER:-srun}"
RUN_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
RESULTS_DIR="${RUN_DIR}/results_hpc_spmm_mpi"

echo "=========================================="
echo "SPMM SPARSE-B MPI BENCHMARK - Job Started"
echo "=========================================="
echo "Job ID:          $SLURM_JOB_ID"
echo "Job Name:        $SLURM_JOB_NAME"
echo "Hostname:        $(hostname)"
echo "Num Tasks:       $SLURM_NTASKS"
echo "CPUs per Task:   $SLURM_CPUS_PER_TASK"
echo "Code Dir:        $SCRIPT_DIR"
echo "Run Dir:         $RUN_DIR"
echo "Python:          $PYTHON_BIN"
echo "MPI Launcher:    $MPI_LAUNCHER"
echo "Memory:          $SLURM_MEM_PER_NODE"
echo "Time Limit:      $SLURM_TIME_LIMIT"
echo "Start Time:      $(date)"
echo "=========================================="

if [ ! -d "$RUN_DIR" ]; then
    echo "ERROR: Run directory does not exist: $RUN_DIR"
    exit 1
fi

if [ ! -w "$RUN_DIR" ]; then
    echo "ERROR: Run directory is not writable: $RUN_DIR"
    exit 1
fi

cd "$RUN_DIR"

mkdir -p "$RESULTS_DIR"

if [ -z "$PYTHON_BIN" ]; then
    echo "ERROR: No python interpreter found in PATH."
    echo "Set PYTHON_BIN explicitly, for example:"
    echo "  sbatch --export=PYTHON_BIN=/home/\$USER/miniforge3/envs/hpc-s26/bin/python submit_spmm_sparse_mpi.sh"
    exit 1
fi

if [ ! -x "$PYTHON_BIN" ]; then
    echo "ERROR: Python interpreter is not executable: $PYTHON_BIN"
    exit 1
fi

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
export MPLCONFIGDIR="${RUN_DIR}/.matplotlib"
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
    --cache-dir "$RUN_DIR/../matrix_matrix_mult/matrices/" \
    --outdir "$RESULTS_DIR"

echo ""
echo "=========================================="
echo "Benchmark Complete"
echo "=========================================="
echo "Results saved to: $RESULTS_DIR/"
echo "End Time: $(date)"
echo "=========================================="
