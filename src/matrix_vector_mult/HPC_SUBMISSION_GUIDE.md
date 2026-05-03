# Phase 1 HPC Job Submission

This directory contains SLURM job scripts to run Phase 1 (serial SpMV benchmarking) on the UTD HPC cluster.

## Scripts

### `submit_phase1_hpc.sh`
Standard Phase 1 benchmark on HPC with larger matrices.

**Configuration:**
- Partition: <your partition>
- Nodes: 1
- CPUs: 4 (serial execution, extra cores for I/O)
- Memory: 32 GB
- Walltime: 2 hours
- Matrix sizes: 50,000 → 100,000 → 200,000
- NNZ ratio: 5.0
- Repeats: 5 per size

**Usage:**
```bash
sbatch submit_phase1_hpc.sh
```

**Monitor:**
```bash
squeue -u $USER  # Check job status
tail -f logs/spmv_phase1_<JOBID>.out  # Watch output
```

---

### `submit_phase1_hpc_extended.sh`
Extended Phase 1 benchmark for very large matrices (memory intensive).

**Configuration:**
- Partition: gpu
- Nodes: 1
- CPUs: 4
- Memory: 64 GB
- Walltime: 4 hours
- Matrix sizes: 500,000 → 1,000,000
- NNZ ratio: 3.0
- Repeats: 3 per size (fewer repeats due to size)

**Usage:**
```bash
sbatch submit_phase1_hpc_extended.sh
```

**Note:** Only run after standard phase1 completes successfully.

---

## Before First Submission

1. **Update path in script:** Edit the `cd` line to match your HPC home directory:
   ```bash
   cd /home/YOUR_USERNAME/utd-hpc-sparse-matrix-project/src/matrix_vector_mult
   ```

2. **Check available partitions:**
   ```bash
   sinfo
   ```
   Adjust `--partition=gpu` if your cluster uses different names (e.g., `compute`, `batch`, etc.)

3. **Verify modules:**
   ```bash
   module avail python
   module avail openblas
   ```
   Update module names if needed.

4. **Make scripts executable:**
   ```bash
   chmod +x submit_phase1_hpc.sh submit_phase1_hpc_extended.sh
   ```

---

## Expected Output

After job completes, results will be in:
- `results/spmv_serial_baseline.csv` — Timing/memory data (standard)
- `results/spmv_serial_comparison.png` — Visualization (standard)
- `results_extended/spmv_serial_baseline.csv` — Timing/memory data (extended)
- `results_extended/spmv_serial_comparison.png` — Visualization (extended)

Job log will be in `logs/spmv_phase1_<JOBID>.out`

---

## Troubleshooting

**Out of memory (OOM) killed:**
- Reduce `--matrix-sizes` in script
- Increase `--mem` in `#SBATCH` directive
- Reduce `--repeats` to 2 or 3

**Module not found:**
```bash
module load python/3.11  # Try different versions
python --version  # Verify after load
```

**Job takes too long:**
- Reduce largest matrix size
- Decrease `--repeats`
- Increase `--cpus-per-task` if I/O is bottleneck

**Missing dependencies (numpy, scipy):**
```bash
# Install in user environment
python -m pip install --user numpy scipy pandas matplotlib
```

---

## Customization

To create a custom job script, copy one of these templates and modify:

```bash
#SBATCH --matrix-sizes 10000 25000 50000  # Your sizes
#SBATCH --repeats 10                       # More repeats for statistical significance
#SBATCH --nnz-ratio 2.0                    # Sparser matrices
#SBATCH --mem=128G                         # More memory if available
#SBATCH --time=01:00:00                    # Shorter walltime if possible
```

---

## Combining with Laptop Results

After running all three benchmarks:
1. Laptop Phase 1 (serial, small matrices): `results/spmv_serial_baseline.csv`
2. HPC Phase 1 (serial, medium matrices): `results/spmv_serial_baseline.csv`
3. HPC Phase 1 Extended (serial, large matrices): `results_extended/spmv_serial_baseline.csv`

You can combine them in Phase 4 analysis for full scaling picture across all matrix sizes.
