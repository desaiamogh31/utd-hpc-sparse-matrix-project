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

set -euo pipefail

MPI_CXX="mpicxx"
EXECUTABLE="spmm_sparse_mpi"
SOURCE_FILE="spmm_sparse_mpi.cpp"
RESULTS_DIR="results_hpc_spmm_mpi"
OUTPUT_CSV="${RESULTS_DIR}/benchmark_spmm_sparse_mpi.csv"
PROCESS_VALUES=(1 2 4 8 16)
MAX_PROCS="${SLURM_NTASKS:-16}"
JOB_LOG="${RESULTS_DIR}/job_${SLURM_JOB_ID:-manual}.log"
JOB_ERR="${RESULTS_DIR}/job_${SLURM_JOB_ID:-manual}.err"

mkdir -p "$RESULTS_DIR"
exec > >(tee -a "$JOB_LOG") 2> >(tee -a "$JOB_ERR" >&2)

MPI_CXX_PATH="$(command -v "$MPI_CXX")"
MPI_BIN_DIR="$(dirname "$MPI_CXX_PATH")"
MPI_LAUNCHER="${MPI_LAUNCHER:-$MPI_BIN_DIR/mpirun}"

echo "=========================================="
echo "SPMM SPARSE-B MPI BENCHMARK - Job Started"
echo "=========================================="
echo "Job ID:          $SLURM_JOB_ID"
echo "Job Name:        $SLURM_JOB_NAME"
echo "Hostname:        $(hostname)"
echo "Num Tasks:       $SLURM_NTASKS"
echo "CPUs per Task:   $SLURM_CPUS_PER_TASK"
echo "Compiler:        $MPI_CXX"
echo "MPI Launcher:    $MPI_LAUNCHER"
echo "Working Dir:     $(pwd)"
echo "Start Time:      $(date)"
echo "Job Log:         $JOB_LOG"
echo "Job Err:         $JOB_ERR"
echo "=========================================="

rm -f "$OUTPUT_CSV"

echo ""
echo "Compiling ${SOURCE_FILE}..."
echo "$MPI_CXX_PATH"
"$MPI_CXX" --version || true
command -v "$MPI_LAUNCHER"
"$MPI_LAUNCHER" --version || true
"$MPI_CXX" -O3 -std=c++17 -o "$EXECUTABLE" "$SOURCE_FILE" -lm
echo "Compilation complete"
ls -lh "$EXECUTABLE"

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
    if [ "$MPI_LAUNCHER" = "srun" ]; then
        srun \
            --ntasks="$procs" \
            --cpus-per-task=1 \
            --kill-on-bad-exit=1 \
            --output "${RESULTS_DIR}/srun_np${procs}_%t.log" \
            --error "${RESULTS_DIR}/srun_np${procs}_%t.err" \
            "./$EXECUTABLE" \
            --outdir "$RESULTS_DIR" \
            --output "$(basename "$OUTPUT_CSV")" \
            --cache-dir ../matrix_matrix_mult/matrices \
            --repeats 3 \
            --b-cols 4 8 16 \
            --sparsity 0.10 \
            --matrices 1138_bus abb313 delaunay_n15 bcsstk30 delaunay_n19 pkustk14
    else
        "$MPI_LAUNCHER" -np "$procs" "./$EXECUTABLE" \
            --outdir "$RESULTS_DIR" \
            --output "$(basename "$OUTPUT_CSV")" \
            --cache-dir ../matrix_matrix_mult/matrices \
            --repeats 3 \
            --b-cols 4 8 16 \
            --sparsity 0.10 \
            --matrices 1138_bus abb313 delaunay_n15 bcsstk30 delaunay_n19 pkustk14 \
            > "${RESULTS_DIR}/mpirun_np${procs}.log" \
            2> "${RESULTS_DIR}/mpirun_np${procs}.err"
    fi
    echo ""
done

echo "=========================================="
echo "Benchmark Complete"
echo "=========================================="
echo "Results saved to: $OUTPUT_CSV"
echo "End Time: $(date)"
echo "=========================================="
