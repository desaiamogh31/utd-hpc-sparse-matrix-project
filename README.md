# Sparse Matrix Storage Formats — UTD HPC Course Project

A high-performance computing (HPC) course project at the **University of Texas at Dallas** that implements, benchmarks, and compares sparse matrix storage formats for **finite-element assembly** and **sparse matrix–vector / matrix–matrix multiplication**.

---

## Table of Contents

- [Overview](#overview)
- [Storage Formats](#storage-formats)
- [Features](#features)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Build](#build)
  - [Run](#run)
- [Benchmarks](#benchmarks)
- [Parallelization](#parallelization)
- [References](#references)
- [License](#license)

---

## Overview

Sparse matrices arise naturally in scientific computing whenever most entries of a matrix are zero — common examples include discretized partial differential equations (PDEs), graph Laplacians, and finite-element stiffness matrices. Efficient storage and computation on these matrices is critical for HPC applications.

This project:

1. Implements several widely-used sparse matrix storage formats from scratch.
2. Assembles global sparse matrices from local element matrices (finite-element assembly).
3. Benchmarks **SpMV** (sparse matrix–vector multiplication) and **SpMM** (sparse matrix–matrix multiplication) across formats.
4. Explores parallel implementations using **OpenMP** and/or **MPI**.

---

## Storage Formats

| Format | Full Name | Best Use Case |
|--------|-----------|---------------|
| **COO** | Coordinate (triplet) | Assembly, incremental construction |
| **CSR** | Compressed Sparse Row | Row-wise access, SpMV |
| **CSC** | Compressed Sparse Column | Column-wise access, sparse direct solvers |
| **LIL** | List of Lists

---

## Features

- **Sparse matrix assembly** — constructs global sparse matrices by accumulating local element contributions.
- **Format conversion** — routines to convert between COO, CSR, CSC, LIL.
- **SpMV** — `y = A * x` benchmarked for all formats.
- **SpMM** — `C = A * B` for pairs of sparse matrices.
- **Correctness validation** — results are cross-checked against a dense reference implementation.
- **Performance profiling** — wall-clock timings, GFlop/s, and memory-bandwidth utilization reported for each format and kernel.

---

## Project Structure

```
utd-hpc-sparse-matrix-project/
├── src/                  # Source files
│   ├── formats/          # Storage format implementations (COO, CSR, CSC, ...)
│   ├── assembly/         # Finite-element assembly routines
│   ├── kernels/          # SpMV and SpMM kernels
│   └── utils/            # I/O, timing, validation helpers
├── tests/                # Unit and integration tests
└── README.md
```

---

## Getting Started

### Prerequisites

| Tool | Minimum Version | Purpose |
|------|-----------------|---------|
| GCC / Clang | 9+ | C/C++ compiler |
| OpenMP | 4.5+ | Shared-memory parallelism |
| MPI (OpenMPI / MPICH) | 3.x | Distributed-memory parallelism (optional) |
| CUDA Toolkit | 11+ | GPU kernels (optional) |
| Python 3 | 3.8+ | Plotting / analysis scripts |
| make | — | Build system |

On a typical Linux HPC system (e.g., UTD Sysbio/Ganymede cluster):

```bash
module load gcc openmpi cuda
```

### Build

```bash
# Clone the repository
git clone https://github.com/desaiamogh31/utd-hpc-sparse-matrix-project.git
cd utd-hpc-sparse-matrix-project

# Build all targets (serial + OpenMP)
make

# Build with MPI support
make MPI=1

# Build with CUDA support
make CUDA=1

# Build tests
make tests
```

### Run

```bash
# Run SpMV benchmark on a Matrix Market file with CSR format
./bin/spmv --format csr --matrix data/sample.mtx

# Run finite-element assembly benchmark
./bin/assembly --elements 10000 --dof 3

# Run the full benchmark suite and dump results to results/
make bench
```

---

## Benchmarks

Results are measured on the UTD HPC cluster nodes. Key metrics:

- **Wall-clock time** (ms)
- **Effective GFlop/s**
- **Memory bandwidth** (GB/s)

Benchmark matrices are drawn from the [SuiteSparse Matrix Collection](https://sparse.tamu.edu/) (formerly University of Florida Sparse Matrix Collection).

---

## Parallelization

| Strategy | Scope | Notes |
|----------|-------|-------|
| OpenMP | Shared-memory (single node) | Loop-level parallelism in SpMV / assembly |
| MPI | Distributed-memory (multi-node) | Row-wise matrix partitioning |
| CUDA | GPU | ELL / CSR formats via custom CUDA kernels |

---

## References

- Y. Saad, *Iterative Methods for Sparse Linear Systems*, 2nd ed., SIAM, 2003.
- T. A. Davis, *Direct Methods for Sparse Linear Systems*, SIAM, 2006.
- [SuiteSparse Matrix Collection](https://sparse.tamu.edu/)
- [Matrix Market file format](https://math.nist.gov/MatrixMarket/formats.html)
- Bell & Garland, *Implementing Sparse Matrix-Vector Multiplication on Throughput-Oriented Processors*, SC '09.

---

## License

This project is licensed under the [MIT License](LICENSE).  
© 2026 Amogh Neelkanth Desai — University of Texas at Dallas
