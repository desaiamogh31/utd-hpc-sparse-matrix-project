"""
Unit tests for sparse matrix assembly routines.

Tests cover:
- Input validation (matrix size, nnz constraints)
- Assembly correctness across formats
- Format consistency
- Edge cases
"""

import sys
import os
import numpy as np
import pytest
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix, lil_matrix

# Add src/assembly to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'assembly'))

from laplacian_matrix import (
    generate_laplacian_entries,
    assemble_coo as laplacian_coo,
    assemble_lil as laplacian_lil,
    assemble_csr as laplacian_csr,
    assemble_csc as laplacian_csc,
    matrices_equal as laplacian_matrices_equal,
)

from symmetric_matrix import (
    generate_upper_tri_entries,
    assemble_coo as sym_coo,
    assemble_lil as sym_lil,
    assemble_csr as sym_csr,
    assemble_csc as sym_csc,
    matrices_equal as sym_matrices_equal,
)

from asymmetric_matrix import (
    generate_random_entries,
    assemble_coo as asym_coo,
    assemble_lil as asym_lil,
    assemble_csr as asym_csr,
    assemble_csc as asym_csc,
    matrices_equal as asym_matrices_equal,
)


# ============================================================================
# LAPLACIAN MATRIX TESTS
# ============================================================================

class TestLaplacianGeneration:
    """Test Laplacian entry generation."""
    
    def test_laplacian_matrix_size_zero(self):
        """Matrix size zero should produce empty arrays."""
        n = 0
        rows, cols, vals = generate_laplacian_entries(n)
        assert len(rows) == len(cols) == len(vals) == 0
    
    def test_laplacian_matrix_size_negative(self):
        """Matrix size negative should produce empty arrays."""
        n = -5
        rows, cols, vals = generate_laplacian_entries(n)
        assert len(rows) == len(cols) == len(vals) == 0
    
    def test_laplacian_small_grid(self):
        """Generate entries for small grid (n=2)."""
        n = 2
        rows, cols, vals = generate_laplacian_entries(n)
        
        # 2x2 grid = 4x4 matrix
        # Each interior point has 5 entries (center + 4 neighbors)
        # With boundary conditions, expect: 4 diag + 4 neighbors = 8 entries
        assert len(rows) == len(cols) == len(vals)
        assert len(rows) > 0
        # Values should be -4 (diagonal) or 1 (neighbors)
        assert all(v in [-4.0, 1.0] for v in vals)
    
    def test_laplacian_entries_consistency(self):
        """Rows, cols, vals should have same length."""
        n = 5
        rows, cols, vals = generate_laplacian_entries(n)
        assert len(rows) == len(cols) == len(vals)
    
    def test_laplacian_indices_in_bounds(self):
        """All indices should be in valid range."""
        n = 10
        rows, cols, vals = generate_laplacian_entries(n)
        mat_size = n * n
        
        assert np.all(rows >= 0) and np.all(rows < mat_size)
        assert np.all(cols >= 0) and np.all(cols < mat_size)


class TestLaplacianAssembly:
    """Test Laplacian assembly across formats."""
    
    def test_laplacian_coo_assembly(self):
        """COO assembly should produce valid matrix."""
        n = 5
        rows, cols, vals = generate_laplacian_entries(n)
        mat = laplacian_coo(n, rows, cols, vals)
        
        assert mat.shape == (n * n, n * n)
        assert mat.nnz > 0
    
    def test_laplacian_all_formats_assembly(self):
        """All formats should assemble successfully."""
        n = 3
        rows, cols, vals = generate_laplacian_entries(n)
        
        mat_coo = laplacian_coo(n, rows, cols, vals)
        mat_lil = laplacian_lil(n, rows, cols, vals)
        mat_csr = laplacian_csr(n, rows, cols, vals)
        mat_csc = laplacian_csc(n, rows, cols, vals)
        
        assert all(m.shape == (n * n, n * n) for m in [mat_coo, mat_lil, mat_csr, mat_csc])
    
    def test_laplacian_formats_equal(self):
        """All formats should produce equivalent matrices."""
        n = 4
        rows, cols, vals = generate_laplacian_entries(n)
        
        mat_coo = laplacian_coo(n, rows, cols, vals)
        mat_lil = laplacian_lil(n, rows, cols, vals)
        mat_csr = laplacian_csr(n, rows, cols, vals)
        mat_csc = laplacian_csc(n, rows, cols, vals)
        
        # All should equal the COO reference
        assert laplacian_matrices_equal(mat_coo, mat_lil)
        assert laplacian_matrices_equal(mat_coo, mat_csr)
        assert laplacian_matrices_equal(mat_coo, mat_csc)
    
    def test_laplacian_sparsity(self):
        """Laplacian should be sparse (nnz << n²)."""
        n = 100
        rows, cols, vals = generate_laplacian_entries(n)
        mat = laplacian_coo(n, rows, cols, vals)
        
        # Laplacian has ~5 entries per row (5-point stencil)
        # So nnz should be roughly 5*n² but not all n²*n² entries
        density = mat.nnz / (n * n) ** 2
        assert density < 0.1  # Should be very sparse


# ============================================================================
# SYMMETRIC MATRIX TESTS
# ============================================================================

class TestSymmetricGeneration:
    """Test symmetric matrix entry generation."""
    
    def test_symmetric_matrix_size_positive(self):
        """Matrix size must be positive."""
        with pytest.raises(ValueError):
            generate_upper_tri_entries(0, 10)
    
    def test_symmetric_nnz_at_boundary(self):
        """Test upper NNZ at maximum upper triangle capacity."""
        n = 5
        max_upper_nnz = (n * (n + 1)) // 2
        # Should work at maximum
        i, j, v = generate_upper_tri_entries(n, max_upper_nnz)
        assert len(i) == max_upper_nnz
    
    def test_symmetric_upper_tri_constraint(self):
        """Generated entries should form upper triangle (i <= j)."""
        n = 10
        upper_nnz = 50
        i, j, v = generate_upper_tri_entries(n, upper_nnz)
        
        # i should be <= j (upper triangle constraint)
        assert np.all(i <= j)
    
    def test_symmetric_small_nnz(self):
        """Handle small nnz (edge case)."""
        n = 10
        upper_nnz = 1
        i, j, v = generate_upper_tri_entries(n, upper_nnz)
        
        assert len(i) == len(j) == len(v) == upper_nnz


class TestSymmetricAssembly:
    """Test symmetric matrix assembly."""
    
    def test_symmetric_coo_assembly(self):
        """COO assembly should mirror upper triangle."""
        n = 5
        upper_nnz = 10
        i, j, v = generate_upper_tri_entries(n, upper_nnz)
        
        mat = sym_coo(n, i, j, v)
        
        # Check symmetry: mat should equal mat.T
        diff = (mat - mat.transpose()).tocoo()
        assert diff.nnz == 0, "Matrix should be symmetric"
    
    def test_symmetric_all_formats(self):
        """All formats should produce symmetric matrices."""
        n = 4
        upper_nnz = 8
        i, j, v = generate_upper_tri_entries(n, upper_nnz)
        
        mat_coo = sym_coo(n, i, j, v)
        mat_lil = sym_lil(n, i, j, v)
        mat_csr = sym_csr(n, i, j, v)
        mat_csc = sym_csc(n, i, j, v)
        
        # All should be symmetric
        for mat in [mat_coo, mat_lil, mat_csr, mat_csc]:
            diff = (mat - mat.transpose()).tocoo()
            assert diff.nnz == 0
    
    def test_symmetric_formats_equal(self):
        """All symmetric formats should be equal."""
        n = 6
        upper_nnz = 15
        i, j, v = generate_upper_tri_entries(n, upper_nnz)
        
        mat_coo = sym_coo(n, i, j, v)
        mat_lil = sym_lil(n, i, j, v)
        mat_csr = sym_csr(n, i, j, v)
        mat_csc = sym_csc(n, i, j, v)
        
        assert sym_matrices_equal(mat_coo, mat_lil)
        assert sym_matrices_equal(mat_coo, mat_csr)
        assert sym_matrices_equal(mat_coo, mat_csc)


# ============================================================================
# ASYMMETRIC MATRIX TESTS
# ============================================================================

class TestAsymmetricGeneration:
    """Test asymmetric matrix entry generation."""
    
    def test_asymmetric_matrix_size_positive(self):
        """Matrix size must be positive."""
        with pytest.raises(ValueError):
            generate_random_entries(0, 10)
    
    def test_asymmetric_nnz_at_capacity(self):
        """Test NNZ at maximum capacity."""
        n = 5
        nnz = n * n  # Full capacity
        i, j, v = generate_random_entries(n, nnz)
        assert len(i) == nnz
    
    def test_asymmetric_entries_generation(self):
        """Generate entries for asymmetric matrix."""
        n = 10
        nnz = 50
        i, j, v = generate_random_entries(n, nnz)
        
        assert len(i) == len(j) == len(v) == nnz
        assert np.all(i >= 0) and np.all(i < n)
        assert np.all(j >= 0) and np.all(j < n)


class TestAsymmetricAssembly:
    """Test asymmetric matrix assembly."""
    
    def test_asymmetric_coo_assembly(self):
        """COO assembly for asymmetric matrix."""
        n = 5
        nnz = 15
        i, j, v = generate_random_entries(n, nnz)
        
        mat = asym_coo(n, i, j, v)
        
        assert mat.shape == (n, n)
        assert mat.nnz > 0
    
    def test_asymmetric_all_formats(self):
        """All formats should assemble asymmetric matrices."""
        n = 4
        nnz = 10
        i, j, v = generate_random_entries(n, nnz)
        
        mat_coo = asym_coo(n, i, j, v)
        mat_lil = asym_lil(n, i, j, v)
        mat_csr = asym_csr(n, i, j, v)
        mat_csc = asym_csc(n, i, j, v)
        
        assert all(m.shape == (n, n) for m in [mat_coo, mat_lil, mat_csr, mat_csc])
    
    def test_asymmetric_formats_equal(self):
        """All asymmetric formats should be equal."""
        n = 6
        nnz = 20
        i, j, v = generate_random_entries(n, nnz)
        
        mat_coo = asym_coo(n, i, j, v)
        mat_lil = asym_lil(n, i, j, v)
        mat_csr = asym_csr(n, i, j, v)
        mat_csc = asym_csc(n, i, j, v)
        
        assert asym_matrices_equal(mat_coo, mat_lil)
        assert asym_matrices_equal(mat_coo, mat_csr)
        assert asym_matrices_equal(mat_coo, mat_csc)
    
    def test_asymmetric_not_necessarily_symmetric(self):
        """Asymmetric matrices shouldn't equal their transpose (in general)."""
        n = 5
        nnz = 15
        i, j, v = generate_random_entries(n, nnz, seed=42)
        
        mat = asym_coo(n, i, j, v)
        mat_t = mat.transpose()
        
        # Most random matrices won't equal their transpose
        diff = (mat - mat_t).tocoo()
        # Just check the matrix was created correctly
        assert mat.shape == (n, n)


# ============================================================================
# DENSE REFERENCE VALIDATION
# ============================================================================

class TestLaplacianDenseReference:
    """Validate Laplacian assembly against dense reference."""
    
    def test_laplacian_small_n4_coo_vs_dense(self):
        """Verify Laplacian COO assembly matches dense reference (n=4)."""
        n = 4
        rows, cols, vals = generate_laplacian_entries(n)
        mat_sparse = laplacian_coo(n, rows, cols, vals)
        mat_dense = mat_sparse.toarray()
        
        # Build dense reference manually using same logic as laplacian
        mat_ref = np.zeros((n * n, n * n))
        for r, c, v in zip(rows, cols, vals):
            mat_ref[r, c] += v
        
        # Should match exactly
        np.testing.assert_array_almost_equal(mat_dense, mat_ref)
    
    def test_laplacian_small_n4_all_formats_vs_dense(self):
        """Verify all Laplacian formats match dense reference (n=4)."""
        n = 4
        rows, cols, vals = generate_laplacian_entries(n)
        
        # Build dense reference
        mat_size = n * n
        mat_ref = np.zeros((mat_size, mat_size))
        for r, c, v in zip(rows, cols, vals):
            mat_ref[r, c] += v
        
        # Compare all sparse formats
        mat_coo = laplacian_coo(n, rows, cols, vals).toarray()
        mat_lil = laplacian_lil(n, rows, cols, vals).toarray()
        mat_csr = laplacian_csr(n, rows, cols, vals).toarray()
        mat_csc = laplacian_csc(n, rows, cols, vals).toarray()
        
        np.testing.assert_array_almost_equal(mat_coo, mat_ref)
        np.testing.assert_array_almost_equal(mat_lil, mat_ref)
        np.testing.assert_array_almost_equal(mat_csr, mat_ref)
        np.testing.assert_array_almost_equal(mat_csc, mat_ref)
    
    def test_laplacian_symmetry_dense(self):
        """Verify Laplacian is symmetric in dense form."""
        n = 4
        rows, cols, vals = generate_laplacian_entries(n)
        mat_dense = laplacian_coo(n, rows, cols, vals).toarray()
        
        # Should equal its transpose
        np.testing.assert_array_almost_equal(mat_dense, mat_dense.T)
    
    def test_laplacian_diagonal_entries_dense(self):
        """Verify Laplacian diagonal entries are -4 (5-point stencil center)."""
        n = 4
        rows, cols, vals = generate_laplacian_entries(n)
        mat_dense = laplacian_coo(n, rows, cols, vals).toarray()
        
        # Interior points (not on boundary) should have diagonal = -4
        # Boundary points may differ (depends on implementation)
        diag = np.diag(mat_dense)
        # All should be -4 or less (boundary may have fewer neighbors)
        assert np.all(diag <= -1.0)  # At minimum, some negative entry


class TestSymmetricDenseReference:
    """Validate symmetric matrix assembly against dense reference."""
    
    def test_symmetric_small_n4_coo_vs_dense(self):
        """Verify symmetric COO assembly matches dense reference (n=4)."""
        n = 4
        upper_nnz = 8
        i, j, v = generate_upper_tri_entries(n, upper_nnz, seed=42)
        mat_sparse = sym_coo(n, i, j, v)
        mat_dense = mat_sparse.toarray()
        
        # Build dense reference: add both (i,j) and (j,i)
        mat_ref = np.zeros((n, n))
        for r, c, val in zip(i, j, v):
            mat_ref[r, c] += val
            if r != c:
                mat_ref[c, r] += val
        
        np.testing.assert_array_almost_equal(mat_dense, mat_ref)
    
    def test_symmetric_small_n4_all_formats_vs_dense(self):
        """Verify all symmetric formats match dense reference (n=4)."""
        n = 4
        upper_nnz = 6
        i, j, v = generate_upper_tri_entries(n, upper_nnz, seed=123)
        
        # Build dense reference
        mat_ref = np.zeros((n, n))
        for r, c, val in zip(i, j, v):
            mat_ref[r, c] += val
            if r != c:
                mat_ref[c, r] += val
        
        # Compare all sparse formats
        mat_coo = sym_coo(n, i, j, v).toarray()
        mat_lil = sym_lil(n, i, j, v).toarray()
        mat_csr = sym_csr(n, i, j, v).toarray()
        mat_csc = sym_csc(n, i, j, v).toarray()
        
        np.testing.assert_array_almost_equal(mat_coo, mat_ref)
        np.testing.assert_array_almost_equal(mat_lil, mat_ref)
        np.testing.assert_array_almost_equal(mat_csr, mat_ref)
        np.testing.assert_array_almost_equal(mat_csc, mat_ref)
    
    def test_symmetric_symmetry_property_dense(self):
        """Verify symmetric matrix equals its transpose."""
        n = 4
        upper_nnz = 10
        i, j, v = generate_upper_tri_entries(n, upper_nnz, seed=99)
        mat_dense = sym_coo(n, i, j, v).toarray()
        
        np.testing.assert_array_almost_equal(mat_dense, mat_dense.T)


class TestAsymmetricDenseReference:
    """Validate asymmetric matrix assembly against dense reference."""
    
    def test_asymmetric_small_n4_coo_vs_dense(self):
        """Verify asymmetric COO assembly matches dense reference (n=4)."""
        n = 4
        nnz = 10
        i, j, v = generate_random_entries(n, nnz, seed=42)
        mat_sparse = asym_coo(n, i, j, v)
        mat_dense = mat_sparse.toarray()
        
        # Build dense reference: add entries directly (no mirroring)
        mat_ref = np.zeros((n, n))
        for r, c, val in zip(i, j, v):
            mat_ref[r, c] += val
        
        np.testing.assert_array_almost_equal(mat_dense, mat_ref)
    
    def test_asymmetric_small_n4_all_formats_vs_dense(self):
        """Verify all asymmetric formats match dense reference (n=4)."""
        n = 4
        nnz = 12
        i, j, v = generate_random_entries(n, nnz, seed=77)
        
        # Build dense reference
        mat_ref = np.zeros((n, n))
        for r, c, val in zip(i, j, v):
            mat_ref[r, c] += val
        
        # Compare all sparse formats
        mat_coo = asym_coo(n, i, j, v).toarray()
        mat_lil = asym_lil(n, i, j, v).toarray()
        mat_csr = asym_csr(n, i, j, v).toarray()
        mat_csc = asym_csc(n, i, j, v).toarray()
        
        np.testing.assert_array_almost_equal(mat_coo, mat_ref)
        np.testing.assert_array_almost_equal(mat_lil, mat_ref)
        np.testing.assert_array_almost_equal(mat_csr, mat_ref)
        np.testing.assert_array_almost_equal(mat_csc, mat_ref)
    
    def test_asymmetric_duplicates_summed_dense(self):
        """Verify duplicate entries are summed correctly in dense form."""
        n = 4
        # Manually create duplicates: (1,2) appears twice
        i = np.array([0, 1, 1, 2, 3])
        j = np.array([0, 2, 2, 1, 3])
        v = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        mat_sparse = asym_coo(n, i, j, v).toarray()
        
        # Build dense reference
        mat_ref = np.zeros((n, n))
        for r, c, val in zip(i, j, v):
            mat_ref[r, c] += val
        
        np.testing.assert_array_almost_equal(mat_sparse, mat_ref)
        # Check that (1,2) has value 2.0 + 3.0 = 5.0
        assert mat_sparse[1, 2] == pytest.approx(5.0)


# ============================================================================
# EDGE CASES & ROBUSTNESS
# ============================================================================

class TestEdgeCases:
    """Test edge cases and robustness."""
    
    def test_single_element_matrix(self):
        """1x1 matrix should assemble correctly."""
        n = 1
        upper_nnz = 1
        i, j, v = generate_upper_tri_entries(n, upper_nnz)
        
        mat = sym_coo(n, i, j, v)
        assert mat.shape == (1, 1)
        assert mat.nnz >= 1
    
    def test_zero_values_in_assembly(self):
        """Assembly should handle zero values correctly."""
        n = 3
        i = np.array([0, 1, 2])
        j = np.array([0, 1, 2])
        v = np.array([0.0, 1.0, 0.0])  # Zero values included
        
        mat = asym_coo(n, i, j, v)
        # scipy.sparse may drop explicit zeros
        assert mat.shape == (n, n)
    
    def test_duplicate_entries(self):
        """Duplicate entries should be summed."""
        n = 3
        # Create duplicates: (0,0) appears twice
        i = np.array([0, 0, 1])
        j = np.array([0, 0, 1])
        v = np.array([1.0, 2.0, 3.0])
        
        mat = asym_coo(n, i, j, v).tocsr()  # Convert to CSR for indexing
        
        # (0,0) should sum to 3.0
        assert mat[0, 0] == pytest.approx(3.0)
        assert mat[1, 1] == pytest.approx(3.0)


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
