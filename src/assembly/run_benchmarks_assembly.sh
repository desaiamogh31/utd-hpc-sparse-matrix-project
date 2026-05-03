#!/bin/bash
# Full benchmark suite runner for sparse matrix assembly
# Runs laplacian, symmetric, and asymmetric scaling + conversions
# Generates aggregated results and visualizations

set -e  # Exit on any error

PROJECT_ROOT="/Users/a.n.d./Documents/UTD/Coursework/HPC/utd-hpc-sparse-matrix-project"
ASSEMBLY_DIR="$PROJECT_ROOT/src/assembly"
RESULTS_DIR="$ASSEMBLY_DIR/results"

echo "================================================================================"
echo "SPARSE MATRIX ASSEMBLY BENCHMARK SUITE"
echo "================================================================================"
echo ""

# Create results directory if it doesn't exist
mkdir -p "$RESULTS_DIR"

# 1. LAPLACIAN SCALING
echo "[1/6] Running Laplacian scaling benchmark (matrix-sizes: 10 20 50 100)..."
cd "$ASSEMBLY_DIR"
python laplacian_matrix.py --mode scaling --matrix-sizes 10 20 50 100 --repeats 5 --outdir "$RESULTS_DIR"
echo "✓ Laplacian scaling complete"
echo ""

# 2. SYMMETRIC SINGLE
echo "[2/6] Running Symmetric single-matrix benchmark (n=100, upper_nnz=500)..."
cd "$ASSEMBLY_DIR"
python symmetric_matrix.py --mode single --n 100 --upper-nnz 500 --repeats 5 --outdir "$RESULTS_DIR"
echo "✓ Symmetric single benchmark complete"
echo ""

# 3. SYMMETRIC SCALING
echo "[3/6] Running Symmetric scaling benchmark (matrix-sizes: 50 100 200 500)..."
cd "$ASSEMBLY_DIR"
python symmetric_matrix.py --mode scaling --matrix-sizes 50 100 200 500 --nnz-ratio 5.0 --repeats 5 --outdir "$RESULTS_DIR"
echo "✓ Symmetric scaling complete"
echo ""

# 4. ASYMMETRIC SINGLE
echo "[4/6] Running Asymmetric single-matrix benchmark (n=100, nnz=500)..."
cd "$ASSEMBLY_DIR"
python asymmetric_matrix.py --mode single --n 100 --nnz 500 --repeats 5 --outdir "$RESULTS_DIR"
echo "✓ Asymmetric single benchmark complete"
echo ""

# 5. ASYMMETRIC SCALING
echo "[5/6] Running Asymmetric scaling benchmark (matrix-sizes: 50 100 200 500)..."
cd "$ASSEMBLY_DIR"
python asymmetric_matrix.py --mode scaling --matrix-sizes 50 100 200 500 --nnz-ratio 5.0 --repeats 5 --outdir "$RESULTS_DIR"
echo "✓ Asymmetric scaling complete"
echo ""

# 6. AGGREGATION
echo "[6/6] Running results aggregation and visualization..."
cd "$PROJECT_ROOT"
python src/aggregate_results.py
echo "✓ Aggregation complete"
echo ""

echo "================================================================================"
echo "BENCHMARK SUITE COMPLETE"
echo "================================================================================"
echo ""
echo "Results available in: $RESULTS_DIR/"
echo ""
echo "Output files:"
echo "  - Assembly benchmarks:"
echo "    • laplacian_matrix_benchmark.csv"
echo "    • symmetric_matrix_benchmark.csv"
echo "    • asymmetric_matrix_benchmark.csv"
echo "  - Scaling benchmarks:"
echo "    • laplacian_scaling.csv + laplacian_scaling.png"
echo "    • symmetric_scaling.csv + symmetric_matrix_scaling.png"
echo "    • asymmetric_scaling.csv + asymmetric_matrix_scaling.png"
echo "  - Conversion benchmarks:"
echo "    • laplacian_conversion_benchmark.csv"
echo "    • symmetric_matrix_conversion_benchmark.csv"
echo "    • asymmetric_matrix_conversion_benchmark.csv"
echo "  - Aggregation results:"
echo "    • comparison_assembly.png"
echo "    • comparison_memory.png"
echo "    • BENCHMARK_SUMMARY.txt"
echo ""
