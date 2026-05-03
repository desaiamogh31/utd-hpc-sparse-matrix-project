#!/bin/bash
#SBATCH --job-name=spmv-phase1-extended
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/spmv_phase1_extended_%j.out
#SBATCH --error=logs/spmv_phase1_extended_%j.err

# Phase 1 Extended: Serial SpMV benchmarking on HPC (very large matrices)

set -e

# Load necessary modules
module load python/3.11
module load openblas

# Create logs directory
mkdir -p logs

echo "=========================================="
echo "PHASE 1 EXTENDED: SERIAL SPMV (LARGE)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Memory available: $(free -h | grep Mem | awk '{print $2}')"
echo "Start time: $(date)"
echo "=========================================="

# Navigate to working directory
cd /home/$(whoami)/utd-hpc-sparse-matrix-project/src/matrix_vector_mult || exit 1

mkdir -p results

# Run with very large matrices (will be slower, more memory intensive)
echo "Running benchmark with very large matrices..."
python benchmark_spmv_serial.py \
    --matrix-sizes 500000 1000000 \
    --repeats 3 \
    --formats coo csr csc \
    --nnz-ratio 3.0 \
    --outdir results_extended \
    --seed 0

echo "=========================================="
echo "End time: $(date)"
echo "=========================================="
