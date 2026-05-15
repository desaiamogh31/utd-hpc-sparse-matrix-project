# Sparse Matrix Storage Formats — UTD HPC Course Project

A high-performance computing (HPC) course project at the **University of Texas at Dallas** that implements, benchmarks, and compares sparse matrix storage formats for **finite-element assembly** and **sparse matrix–vector / matrix–matrix multiplication**.

---

## Table of Contents

- [Overview](#overview)
- [Project Status](#project-status)
- [Storage Formats](#storage-formats)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Build](#build)
  - [Run](#run)
- [References](#references)
- [License](#license)

---

## Overview

Sparse matrices arise naturally in scientific computing whenever most entries of a matrix are zero — common examples include discretized partial differential equations (PDEs), graph Laplacians, and finite-element stiffness matrices. Efficient storage and computation on these matrices is critical for HPC applications.

This project implements:

1. Multiple sparse matrix storage formats (COO, CSR, CSC, LIL) from scratch
2. Finite-element assembly routines for creating global sparse matrices
3. Sparse matrix–vector multiplication (SpMV) with serial and OpenMP implementations
4. Sparse matrix–matrix multiplication (SpMM) with serial, OpenMP, and MPI implementations
5. Comprehensive benchmarking across all formats and parallelization strategies

---

## Project Status

| Phase | Component | Status |
|-------|-----------|--------|
| **Phase 1** | Finite-element assembly (COO, CSR, CSC, LIL) | 
| **Phase 2** | SpMV: Serial Python + OpenMP (1-8 threads) | 
| **Phase 3a** | SpMM: Serial Python (3 algorithms) + OpenMP |
| **Phase 3b** | SpMM: MPI distributed-memory framework |

---

## Storage Formats

| Format | Full Name | Best Use Case |
|--------|-----------|---------------|
| **COO** | Coordinate (triplet) | Assembly, incremental construction |
| **CSR** | Compressed Sparse Row | Row-wise access, SpMV, row-parallel SpMM |
| **CSC** | Compressed Sparse Column | Column-wise access, outer-product SpMM |
| **LIL** | List of Lists | Assembly, incremental construction |

---

## Project Structure

```
src/
├── assembly/                    # Phase 1: Finite-element assembly
│   ├── laplacian_matrix.py      # 2D Laplacian (5-point stencil)
│   ├── symmetric_matrix.py      # Random symmetric sparse matrices
│   ├── asymmetric_matrix.py     # General asymmetric sparse matrices
│   ├── run_benchmarks_assembly.sh
│   └── results/                 # Assembly and conversion benchmarks
│
├── matrix_vector_mult/          # Phase 2: SpMV
│   ├── spmv_python.py           # Serial SpMV implementations (COO, CSR, CSC, LIL)
│   ├── spmv_openmp.cpp          # OpenMP SpMV (row-parallel CSR)
│   ├── spmv_wrapper.py          # Python ctypes wrapper for C++ OpenMP
│   ├── benchmark_spmv_serial.py # Serial format comparison
│   ├── benchmark_spmv_openmp.py # OpenMP thread scaling (1-8 threads)
│   ├── benchmark_suite_sparse.py # Real-world SuiteSparse matrices
│   ├── matrices/                # Real sparse matrices (.mtx files)
│   └── results/                 # SpMV benchmark results
│
├── matrix_matrix_mult/          # Phase 3a: SpMM (serial + OpenMP)
│   ├── spmm_python.py           # Serial SpMM: row-wise, outer-product, blocked inner-product
│   ├── spmm_openmp.cpp          # OpenMP SpMM (parallel variants)
│   ├── spmm_openmp_wrapper.py   # Python ctypes wrapper
│   ├── benchmark_spmm_serial.py # Serial algorithm comparison
│   ├── benchmark_spmm_sparse_openmp.py # OpenMP scaling
│   ├── build.py                 # Compiles C++ extensions
│   └── results/                 # SpMM benchmark results
│
├── matrix_mult_mpi/             # Phase 3b: SpMM MPI
│   ├── spmm_mpi.py              # MPI helper (row partitioning, distributed SpMM)
│   ├── spmm_sparse_mpi.cpp      # Native C++ MPI implementation (~600 lines)
│   ├── benchmark_spmm_sparse_mpi.py # MPI benchmark driver
│   └── smoke_results_hpc/       # Smoke test results
│
├── load_matrices.py             # Matrix I/O utilities (skeleton)
└── aggregate_results.py         # Results aggregation (skeleton)

tests/
├── test_assembly.py             # Assembly correctness tests
├── test_spmm.py                 # SpMM algorithm tests (20+ cases)
├── test_spmm_openmp.py          # OpenMP wrapper tests
├── test_spmm_mpi_helpers.py     # MPI helper logic tests
└── test_spmm_mpi_native.py      # Native C++ MPI smoke test

scratch/                         # Sample notebooks and test matrices
```

---

## Getting Started

### Prerequisites

- **Python 3.8+** — Main implementation language
- **NumPy/SciPy** — Sparse matrix support, dense reference
- **GCC/Clang 9+** — C/C++ compiler for OpenMP/MPI extensions
- **OpenMP 4.5+** — Shared-memory parallelism (Part of GCC/Clang)
- **MPI (OpenMPI 3.x or MPICH 3.x)** — Optional, for Phase 3b distributed-memory
- **make** — Build system

On a Linux HPC system:
```bash
module load gcc openmpi
```

### Build

**Build Phase 1, 2, 3a (Python + OpenMP):**
```bash
cd src/matrix_vector_mult
python build.py  # Compiles spmv_openmp.so

cd ../matrix_matrix_mult
python build.py  # Compiles spmm_openmp.so
```

**Optional: Build Phase 3b native C++ MPI binary:**
```bash
cd src/matrix_mult_mpi
mpicxx -O3 -fopenmp spmm_sparse_mpi.cpp -o spmm_sparse_mpi
```

### Run

**Phase 1: Finite-element Assembly**
```bash
cd src/assembly
bash run_benchmarks_assembly.sh
# Results saved to results/
```

**Phase 2: SpMV Benchmarks**
```bash
cd src/matrix_vector_mult

# Serial baseline (all formats)
python benchmark_spmv_serial.py

# OpenMP scaling (1-8 threads)
python benchmark_spmv_openmp.py

# Real-world SuiteSparse matrices
python benchmark_suite_sparse.py
```

**Phase 3a: SpMM Benchmarks**
```bash
cd src/matrix_matrix_mult

# Serial algorithm comparison (row-wise, outer-product, blocked)
python benchmark_spmm_serial.py

# OpenMP thread scaling
python benchmark_spmm_sparse_openmp.py
```

**Phase 3b: SpMM MPI (Python interface)**
```bash
cd src/matrix_mult_mpi

# Requires MPI installation
python benchmark_spmm_sparse_mpi.py
```

**Run Tests**
```bash
cd tests/

# Assembly tests
python test_assembly.py

# SpMM tests
python test_spmm.py
python test_spmm_openmp.py

# MPI helper tests (no MPI launch)
python test_spmm_mpi_helpers.py

# MPI native test (requires mpicxx)
python test_spmm_mpi_native.py
```

---

## Benchmarking Notes

All benchmark results are saved as CSV files with the following metrics:
- **Wall-clock time** (milliseconds)
- **GFlop/s** (effective floating-point operations per second)
- **Memory bandwidth** (GB/s, where applicable)

Test matrices are drawn from:
- **Synthetic**: Laplacian (2D FEM), random symmetric, random asymmetric
- **Real-world**: SuiteSparse Matrix Collection (bus, bcsstk30, delaunay, abb313, pkustk14, etc.)

---

## References

- Y. Saad, *Iterative Methods for Sparse Linear Systems*, 2nd ed., SIAM, 2003.
- T. A. Davis, *Direct Methods for Sparse Linear Systems*, SIAM, 2006.
- [SuiteSparse Matrix Collection](https://sparse.tamu.edu/)
- [Matrix Market file format](https://math.nist.gov/MatrixMarket/formats.html)

---

## License

Licensed under the [MIT License](LICENSE).  
© 2026 Amogh Neelkanth Desai — University of Texas at Dallas
