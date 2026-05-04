"""
Unit tests for MPI SpMM helper logic that does not require launching MPI.
"""

from __future__ import annotations

import os
import sys

import numpy as np
from scipy.sparse import csr_matrix

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "src", "matrix_mult_mpi")
)

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
