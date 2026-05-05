#!/bin/bash
#SBATCH --job-name=spmm_sparse_mpi
#SBATCH --partition=cmt
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --time=08:00:00
#SBATCH --output=spmm_sparse_mpi_%j.log
#SBATCH --error=spmm_sparse_mpi_%j.err

################################################################################
# Native C++ MPI SpMM benchmark submission script.
#
# Run this from src/matrix_mult_mpi with:
#   sbatch submit_spmm_sparse_mpi.sh
#
# This mirrors the working nbody MPI workflow:
#   1. compile one C++ MPI executable with mpicxx
#   2. launch it directly with srun for multiple process counts
#   3. write one combined CSV in the current directory
################################################################################

set -e

MPI_CXX="${MPI_CXX:-mpicxx}"
EXECUTABLE="spmm_sparse_mpi"
SOURCE_FILE="spmm_sparse_mpi.cpp"
RESULTS_DIR="results_hpc_spmm_mpi"
OUTPUT_CSV="${RESULTS_DIR}/benchmark_spmm_sparse_mpi.csv"
PROCESS_VALUES=(1 2 4 8 16)
MAX_PROCS="${SLURM_NTASKS:-16}"

echo "=========================================="
echo "SPMM SPARSE-B MPI BENCHMARK - Job Started"
echo "=========================================="
echo "Job ID:          $SLURM_JOB_ID"
echo "Job Name:        $SLURM_JOB_NAME"
echo "Hostname:        $(hostname)"
echo "Num Tasks:       $SLURM_NTASKS"
echo "CPUs per Task:   $SLURM_CPUS_PER_TASK"
echo "Compiler:        $MPI_CXX"
echo "Working Dir:     $(pwd)"
echo "Start Time:      $(date)"
echo "=========================================="

mkdir -p "$RESULTS_DIR"
rm -f "$OUTPUT_CSV"

echo ""
echo "Compiling ${SOURCE_FILE}..."
"$MPI_CXX" -O3 -std=c++17 -o "$EXECUTABLE" "$SOURCE_FILE"
echo "Compilation complete"

export OMP_NUM_THREADS=1

echo ""
echo "=========================================="
echo "Starting Sparse-B SpMM MPI Benchmark"
echo "=========================================="
echo ""

for procs in "${PROCESS_VALUES[@]}"; do
    if [ "$procs" -gt "$MAX_PROCS" ]; then
        echo "Skipping process count ${procs} (exceeds allocated tasks)"
        continue
    fi

    echo "Running benchmark with ${procs} MPI process(es)..."
    srun --ntasks="$procs" --cpus-per-task=1 "./$EXECUTABLE" \
        --outdir "$RESULTS_DIR" \
        --output "$(basename "$OUTPUT_CSV")" \
        --cache-dir ../matrix_matrix_mult/matrices \
        --repeats 3 \
        --b-cols 4 8 16 \
        --sparsity 0.10 \
        --matrices 1138_bus abb313 delaunay_n15 bcsstk30 delaunay_n19 pkustk14
    echo ""
done

echo "=========================================="
echo "Benchmark Complete"
echo "=========================================="
echo "Results saved to: $OUTPUT_CSV"
echo "End Time: $(date)"
echo "=========================================="
