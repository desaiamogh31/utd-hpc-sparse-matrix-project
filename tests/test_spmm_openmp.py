"""
Unit tests for OpenMP sparse-B SpMM wrappers.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
from scipy.sparse import csr_matrix, random as sparse_random


sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "src", "matrix_matrix_mult")
)

try:
    from spmm_openmp_wrapper import (
        spmm_blocked_inner_product_sparse_b_omp,
        spmm_outer_product_sparse_b_omp,
        spmm_row_wise_sparse_b_omp,
    )
except FileNotFoundError:
    pytest.skip("spmm_openmp.so not built; skipping OpenMP SpMM tests", allow_module_level=True)


@pytest.mark.parametrize(
    ("fn_name", "kernel"),
    [
        ("row-wise", spmm_row_wise_sparse_b_omp),
        ("outer-product", spmm_outer_product_sparse_b_omp),
        ("blocked", spmm_blocked_inner_product_sparse_b_omp),
    ],
)
def test_openmp_sparse_b_matches_scipy(fn_name, kernel):
    rng = np.random.default_rng(42)
    A = sparse_random(8, 8, density=0.25, format="csr", random_state=42)
    B = sparse_random(8, 5, density=0.30, format="csr", random_state=24)

    C = kernel(A, B, num_threads=2)
    C_ref = (A @ B).tocsr()

    np.testing.assert_allclose(
        C.toarray(),
        C_ref.toarray(),
        rtol=1e-12,
        atol=1e-12,
        err_msg=f"Mismatch in {fn_name} OpenMP kernel",
    )


def test_openmp_sparse_b_handles_empty_rows():
    A = csr_matrix(np.array([[0, 0, 0], [1, 2, 0], [0, 0, 0]], dtype=np.float64))
    B = csr_matrix(np.array([[1, 0], [0, 3], [4, 5]], dtype=np.float64))

    C = spmm_row_wise_sparse_b_omp(A, B, num_threads=2)
    C_ref = (A @ B).tocsr()

    np.testing.assert_allclose(C.toarray(), C_ref.toarray(), rtol=1e-12, atol=1e-12)
