"""
Unit tests for MPI SpMM helper logic that does not require launching MPI.
"""

from __future__ import annotations

import os
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "src", "matrix_mult_mpi")
)

from benchmark_spmm_sparse_mpi import build_worker_launch_cmd  # noqa: E402
from spmm_mpi import compute_row_partitions, local_spmm_sparse_b  # noqa: E402


def test_compute_row_partitions_balances_rows():
    parts = compute_row_partitions(10, 3)
    assert parts == [(0, 4), (4, 7), (7, 10)]


def test_local_spmm_sparse_b_matches_scipy():
    A = csr_matrix(np.array([[1, 0, 2], [0, 3, 0], [4, 0, 5]], dtype=np.float64))
    B = csr_matrix(np.array([[1, 0], [0, 2], [3, 4]], dtype=np.float64))
    C_ref = (A @ B).tocsr()

    for algorithm in ["row-wise", "outer-product", "blocked", "scipy"]:
        C = local_spmm_sparse_b(A, B, algorithm)
        np.testing.assert_allclose(C.toarray(), C_ref.toarray(), rtol=1e-12, atol=1e-12)


def test_build_worker_launch_cmd_uses_slurm_friendly_srun_flags():
    args = Namespace(
        mpi_launcher="srun",
        validate=True,
        matrices=["1138_bus"],
        b_cols=[4, 8],
        sparsity=[0.10],
        outdir="results_hpc_spmm_mpi",
        cache_dir="cache",
        repeats=3,
    )

    cmd = build_worker_launch_cmd(args, 4, Path("results_hpc_spmm_mpi/_mpi_worker_runs/worker_np4.csv"))

    assert cmd[:5] == ["srun", "--overlap", "--export=ALL", "--ntasks", "4"]
    assert "--mode" in cmd and "worker" in cmd
    assert "--validate" in cmd
    assert "--matrices" in cmd and "1138_bus" in cmd
