"""
MPI helpers for sparse-B SpMM.

This module implements a first distributed-memory SpMM path using row-wise
partitioning of A across ranks and replication of sparse B on each rank.
Local SpMM work reuses the existing sparse-B kernels from matrix_matrix_mult.
"""

from __future__ import annotations

import os
import sys
from typing import List, Tuple

import numpy as np
from scipy.sparse import csr_matrix, spmatrix, vstack

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "matrix_matrix_mult"))

from spmm_python import (  # noqa: E402
    spmm_blocked_inner_product_sparse_b,
    spmm_outer_product_sparse_b,
    spmm_row_wise_sparse_b,
)


def compute_row_partitions(num_rows: int, num_ranks: int) -> List[Tuple[int, int]]:
    """Return contiguous row ranges for each rank."""
    base = num_rows // num_ranks
    remainder = num_rows % num_ranks
    partitions: List[Tuple[int, int]] = []
    start = 0
    for rank in range(num_ranks):
        rows = base + (1 if rank < remainder else 0)
        end = start + rows
        partitions.append((start, end))
        start = end
    return partitions


def local_spmm_sparse_b(A_local: spmatrix, B_csr: csr_matrix, algorithm: str) -> csr_matrix:
    """Execute one local sparse-B SpMM kernel on a CSR row block."""
    A_csr = A_local.tocsr() if not isinstance(A_local, csr_matrix) else A_local

    if algorithm == "row-wise":
        return spmm_row_wise_sparse_b(A_csr, B_csr)
    if algorithm == "outer-product":
        return spmm_outer_product_sparse_b(A_csr.tocsc(), B_csr)
    if algorithm == "blocked":
        return spmm_blocked_inner_product_sparse_b(A_csr, B_csr)
    if algorithm == "scipy":
        return (A_csr @ B_csr).tocsr()

    raise ValueError(f"Unknown algorithm: {algorithm}")


def _sparse_memory_mb(matrix: csr_matrix) -> float:
    return (
        matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes
    ) / (1024 * 1024)


def benchmark_spmm_algorithm_sparse_b_mpi(
    A: spmatrix | None,
    B: spmatrix | None,
    algorithm: str,
    repeats: int = 3,
    validate: bool = False,
):
    """
    Benchmark one sparse-B SpMM algorithm using MPI row partitioning.

    Root rank owns the full input matrices. Other ranks should pass None.
    """
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        if A is None or B is None:
            raise ValueError("Root rank must provide both A and B")
        A_csr = A.tocsr() if not isinstance(A, csr_matrix) else A
        B_csr = B.tocsr() if not isinstance(B, csr_matrix) else B
        partitions = compute_row_partitions(A_csr.shape[0], size)
        local_blocks = [A_csr[row_start:row_end] for row_start, row_end in partitions]
        meta = {
            "shape_a": A_csr.shape,
            "shape_b": B_csr.shape,
            "nnz_a": int(A_csr.nnz),
            "nnz_b": int(B_csr.nnz),
            "partitions": partitions,
        }
    else:
        local_blocks = None
        B_csr = None
        meta = None
        A_csr = None

    local_A = comm.scatter(local_blocks, root=0)
    B_csr = comm.bcast(B_csr, root=0)
    meta = comm.bcast(meta, root=0)

    C_local = local_spmm_sparse_b(local_A, B_csr, algorithm)

    times: List[float] = []
    for _ in range(repeats):
        comm.Barrier()
        start = MPI.Wtime()
        C_local = local_spmm_sparse_b(local_A, B_csr, algorithm)
        local_elapsed = MPI.Wtime() - start
        elapsed = comm.allreduce(local_elapsed, op=MPI.MAX)
        times.append(elapsed)

    local_nnz_c = int(C_local.nnz)
    total_nnz_c = comm.allreduce(local_nnz_c, op=MPI.SUM)

    local_mem_a = _sparse_memory_mb(local_A.tocsr())
    local_mem_c = _sparse_memory_mb(C_local)
    total_mem_a = comm.allreduce(local_mem_a, op=MPI.SUM)
    total_mem_c = comm.allreduce(local_mem_c, op=MPI.SUM)
    memory_b_per_rank = _sparse_memory_mb(B_csr)

    mean_time = float(np.mean(times))
    std_time = float(np.std(times))
    min_time = float(np.min(times))

    shared_dim = meta["shape_a"][1]
    flops = 2.0 * meta["nnz_a"] * meta["nnz_b"] / shared_dim if shared_dim > 0 else 0.0
    gflops = flops / (mean_time * 1e9) if mean_time > 0 else 0.0

    validation_ok = None
    if validate:
        gathered = comm.gather(C_local, root=0)
        if rank == 0:
            C_full = vstack(gathered, format="csr")
            C_ref = (A_csr @ B_csr).tocsr()
            diff = (C_full - C_ref).tocsr()
            validation_ok = diff.nnz == 0
        validation_ok = comm.bcast(validation_ok, root=0)

    metrics = {
        "algorithm": algorithm,
        "mean_time": mean_time,
        "std_time": std_time,
        "min_time": min_time,
        "gflops": gflops,
        "nnz_a": meta["nnz_a"],
        "nnz_b": meta["nnz_b"],
        "nnz_c": total_nnz_c,
        "memory_a_mb_total": float(total_mem_a),
        "memory_b_mb_per_rank": float(memory_b_per_rank),
        "memory_c_mb_total": float(total_mem_c),
        "num_procs": size,
        "local_rows": int(local_A.shape[0]),
        "validation_ok": validation_ok,
    }

    if rank == 0:
        return metrics
    return None
