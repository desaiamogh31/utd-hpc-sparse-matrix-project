#!/bin/bash
#SBATCH --job-name=spmv-phase1
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/spmv_phase1_%j.out
#SBATCH --error=logs/spmv_phase1_%j.err

# Phase 1: Serial SpMV benchmarking on HPC (larger matrices)

set -e

# Load necessary modules
module load python/3.11
module load openblas  # Optional: for optimized linear algebra

# Create logs directory if it doesn't exist
mkdir -p logs

# Print job info
echo "=========================================="
echo "PHASE 1: SERIAL SPMV ON HPC"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "=========================================="

# Navigate to working directory
cd /home/$(whoami)/utd-hpc-sparse-matrix-project/src/matrix_vector_mult || exit 1

# Create results directory
mkdir -p results

# Run Phase 1 benchmark with larger matrix sizes for HPC
echo "Running benchmark with larger matrices..."
python benchmark_spmv_serial.py \
    --matrix-sizes 50000 100000 200000 \
    --repeats 5 \
    --nnz-ratio 5.0 \
    --outdir results \
    --seed 0

echo "=========================================="
echo "End time: $(date)"
echo "=========================================="
