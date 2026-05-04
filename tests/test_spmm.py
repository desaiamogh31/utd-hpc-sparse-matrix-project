"""
Unit tests for SpMM (Sparse Matrix-Matrix Multiplication) implementations.

Tests correctness of three algorithms against dense reference:
- Row-wise
- Outer-product
- Blocked inner-product

Usage:
    pytest tests/test_spmm.py -v
"""

import numpy as np
import pytest
from scipy.sparse import csr_matrix, csc_matrix, coo_matrix, lil_matrix, random as sparse_random

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "matrix_matrix_mult"))

from spmm_python import (
    spmm_row_wise,
    spmm_outer_product,
    spmm_blocked_inner_product,
    spmm,
    validate_spmm,
)


class TestSpmmRowWise:
    """Test row-wise SpMM algorithm."""
    
    def test_small_dense_matrix(self):
        """Test on small fully-dense matrix."""
        A_dense = np.array([[1, 2], [3, 4]], dtype=np.float64)
        B = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.float64)
        
        A_csr = csr_matrix(A_dense)
        C = spmm_row_wise(A_csr, B)
        C_ref = A_dense @ B
        
        np.testing.assert_allclose(C.toarray(), C_ref, rtol=1e-12)
    
    def test_sparse_matrix_single_column(self):
        """Test sparse matrix with single dense column (reduces to SpMV)."""
        A_dense = np.array([[1, 0, 2], [0, 3, 0], [4, 0, 5]], dtype=np.float64)
        B = np.array([[1], [2], [3]], dtype=np.float64)
        
        A_csr = csr_matrix(A_dense)
        C = spmm_row_wise(A_csr, B)
        C_ref = A_dense @ B
        
        np.testing.assert_allclose(C.toarray(), C_ref, rtol=1e-12)
    
    def test_sparse_matrix_multiple_columns(self):
        """Test sparse matrix with multiple dense columns."""
        A_dense = np.array([[1, 0, 2], [0, 3, 0], [4, 0, 5]], dtype=np.float64)
        B = np.random.randn(3, 4).astype(np.float64)
        
        A_csr = csr_matrix(A_dense)
        C = spmm_row_wise(A_csr, B)
        C_ref = A_dense @ B
        
        np.testing.assert_allclose(C.toarray(), C_ref, rtol=1e-12)
    
    def test_random_sparse_matrix(self):
        """Test on random sparse matrix."""
        np.random.seed(42)
        A = sparse_random(50, 50, density=0.1, format="csr")
        B = np.random.randn(50, 8).astype(np.float64)
        
        C = spmm_row_wise(A, B)
        C_ref = A.toarray() @ B
        
        np.testing.assert_allclose(C.toarray(), C_ref, rtol=1e-12)
    
    def test_empty_rows(self):
        """Test matrix with empty rows."""
        A_dense = np.array([[0, 0, 0], [1, 2, 3], [0, 0, 0]], dtype=np.float64)
        B = np.random.randn(3, 5).astype(np.float64)
        
        A_csr = csr_matrix(A_dense)
        C = spmm_row_wise(A_csr, B)
        C_ref = A_dense @ B
        
        np.testing.assert_allclose(C.toarray(), C_ref, rtol=1e-12)


class TestSpmmOuterProduct:
    """Test outer-product SpMM algorithm."""
    
    def test_small_dense_matrix(self):
        """Test on small fully-dense matrix."""
        A_dense = np.array([[1, 2], [3, 4]], dtype=np.float64)
        B = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.float64)
        
        A_csc = csc_matrix(A_dense)
        C = spmm_outer_product(A_csc, B)
        C_ref = A_dense @ B
        
        np.testing.assert_allclose(C.toarray(), C_ref, rtol=1e-12)
    
    def test_sparse_matrix_single_column(self):
        """Test sparse matrix with single dense column."""
        A_dense = np.array([[1, 0, 2], [0, 3, 0], [4, 0, 5]], dtype=np.float64)
        B = np.array([[1], [2], [3]], dtype=np.float64)
        
        A_csc = csc_matrix(A_dense)
        C = spmm_outer_product(A_csc, B)
        C_ref = A_dense @ B
        
        np.testing.assert_allclose(C.toarray(), C_ref, rtol=1e-12)
    
    def test_sparse_matrix_multiple_columns(self):
        """Test sparse matrix with multiple dense columns."""
        A_dense = np.array([[1, 0, 2], [0, 3, 0], [4, 0, 5]], dtype=np.float64)
        B = np.random.randn(3, 4).astype(np.float64)
        
        A_csc = csc_matrix(A_dense)
        C = spmm_outer_product(A_csc, B)
        C_ref = A_dense @ B
        
        np.testing.assert_allclose(C.toarray(), C_ref, rtol=1e-12)
    
    def test_random_sparse_matrix(self):
        """Test on random sparse matrix."""
        np.random.seed(42)
        A = sparse_random(50, 50, density=0.1, format="csc")
        B = np.random.randn(50, 8).astype(np.float64)
        
        C = spmm_outer_product(A, B)
        C_ref = A.toarray() @ B
        
        np.testing.assert_allclose(C.toarray(), C_ref, rtol=1e-12)


class TestSpmmBlocked:
    """Test blocked inner-product SpMM algorithm."""
    
    def test_small_dense_matrix(self):
        """Test on small fully-dense matrix."""
        A_dense = np.array([[1, 2], [3, 4]], dtype=np.float64)
        B = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.float64)
        
        A_csr = csr_matrix(A_dense)
        C = spmm_blocked_inner_product(A_csr, B)
        C_ref = A_dense @ B
        
        np.testing.assert_allclose(C.toarray(), C_ref, rtol=1e-12)
    
    def test_sparse_matrix_single_column(self):
        """Test sparse matrix with single dense column."""
        A_dense = np.array([[1, 0, 2], [0, 3, 0], [4, 0, 5]], dtype=np.float64)
        B = np.array([[1], [2], [3]], dtype=np.float64)
        
        A_csr = csr_matrix(A_dense)
        C = spmm_blocked_inner_product(A_csr, B)
        C_ref = A_dense @ B
        
        np.testing.assert_allclose(C.toarray(), C_ref, rtol=1e-12)
    
    def test_sparse_matrix_large_block(self):
        """Test sparse matrix with large number of columns (tests blocking)."""
        A_dense = np.array([[1, 0, 2], [0, 3, 0], [4, 0, 5]], dtype=np.float64)
        B = np.random.randn(3, 128).astype(np.float64)
        
        A_csr = csr_matrix(A_dense)
        C = spmm_blocked_inner_product(A_csr, B, block_k=32)
        C_ref = A_dense @ B
        
        np.testing.assert_allclose(C.toarray(), C_ref, rtol=1e-12)
    
    def test_random_sparse_matrix(self):
        """Test on random sparse matrix."""
        np.random.seed(42)
        A = sparse_random(50, 50, density=0.1, format="csr")
        B = np.random.randn(50, 16).astype(np.float64)
        
        C = spmm_blocked_inner_product(A, B, block_k=8)
        C_ref = A.toarray() @ B
        
        np.testing.assert_allclose(C.toarray(), C_ref, rtol=1e-12)


class TestSpmmDispatcher:
    """Test SpMM dispatcher function."""
    
    def test_row_wise_dispatch(self):
        """Test row-wise algorithm via dispatcher."""
        np.random.seed(42)
        A = sparse_random(30, 30, density=0.15, format="csr")
        B = np.random.randn(30, 10).astype(np.float64)
        
        C = spmm(A, B, algorithm="row-wise")
        C_ref = A.toarray() @ B
        
        np.testing.assert_allclose(C.toarray(), C_ref, rtol=1e-12)
    
    def test_outer_product_dispatch(self):
        """Test outer-product algorithm via dispatcher."""
        np.random.seed(42)
        A = sparse_random(30, 30, density=0.15, format="csr")
        B = np.random.randn(30, 10).astype(np.float64)
        
        C = spmm(A, B, algorithm="outer-product")
        C_ref = A.toarray() @ B
        
        np.testing.assert_allclose(C.toarray(), C_ref, rtol=1e-12)
    
    def test_blocked_dispatch(self):
        """Test blocked algorithm via dispatcher."""
        np.random.seed(42)
        A = sparse_random(30, 30, density=0.15, format="csr")
        B = np.random.randn(30, 10).astype(np.float64)
        
        C = spmm(A, B, algorithm="blocked")
        C_ref = A.toarray() @ B
        
        np.testing.assert_allclose(C.toarray(), C_ref, rtol=1e-12)
    
    def test_format_conversion(self):
        """Test that different input formats all work."""
        np.random.seed(42)
        A_csr = sparse_random(30, 30, density=0.15, format="csr")
        B = np.random.randn(30, 8).astype(np.float64)
        C_ref = A_csr.toarray() @ B
        
        # Test with different formats as input
        for A in [A_csr, A_csr.tocsc(), A_csr.tocoo(), A_csr.tolil()]:
            C = spmm(A, B, algorithm="row-wise")
            np.testing.assert_allclose(C.toarray(), C_ref, rtol=1e-12)
    
    def test_invalid_algorithm(self):
        """Test that invalid algorithm raises error."""
        A = sparse_random(10, 10, density=0.2, format="csr")
        B = np.random.randn(10, 5).astype(np.float64)
        
        with pytest.raises(ValueError, match="Unknown algorithm"):
            spmm(A, B, algorithm="nonexistent")


class TestValidateSpmm:
    """Test SpMM validation function."""
    
    def test_correct_result_passes(self):
        """Test that correct result passes validation."""
        np.random.seed(42)
        A = sparse_random(20, 20, density=0.2, format="csr")
        B = np.random.randn(20, 5).astype(np.float64)
        
        C = spmm(A, B, algorithm="row-wise")
        assert validate_spmm(A, B, C, tol=1e-10)
    
    def test_incorrect_result_fails(self):
        """Test that incorrect result fails validation."""
        np.random.seed(42)
        A = sparse_random(20, 20, density=0.2, format="csr")
        B = np.random.randn(20, 5).astype(np.float64)
        
        # Create intentionally wrong result
        wrong_result = sparse_random(20, 5, density=0.5, format="csr")
        
        assert not validate_spmm(A, B, wrong_result, tol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
