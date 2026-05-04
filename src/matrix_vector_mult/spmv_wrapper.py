"""
ctypes wrapper for OpenMP-accelerated SpMV (C extension).

Provides a Python interface to the compiled spmv_openmp.so shared library.
Handles conversion between scipy sparse matrices and C arrays.
"""

from __future__ import annotations
import ctypes
import os
import numpy as np
from scipy.sparse import csr_matrix, spmatrix


# Load the compiled shared library
_lib_path = os.path.join(os.path.dirname(__file__), "spmv_openmp.so")

if not os.path.exists(_lib_path):
    raise FileNotFoundError(
        f"OpenMP library not found at {_lib_path}\n"
        "Please compile first: python build.py"
    )

_lib = ctypes.CDLL(_lib_path)

# Define function signatures
_spmv_csr_omp = _lib.spmv_csr_omp
_spmv_csr_omp.argtypes = [
    ctypes.c_int,                           # m
    ctypes.c_int,                           # n
    ctypes.c_int,                           # nnz
    ctypes.POINTER(ctypes.c_double),        # data
    ctypes.POINTER(ctypes.c_int),           # indices
    ctypes.POINTER(ctypes.c_int),           # indptr
    ctypes.POINTER(ctypes.c_double),        # x
    ctypes.POINTER(ctypes.c_double),        # y
    ctypes.c_int,                           # num_threads
]
_spmv_csr_omp.restype = None

_get_num_threads = _lib.get_num_threads
_get_num_threads.argtypes = []
_get_num_threads.restype = ctypes.c_int

_set_num_threads = _lib.set_num_threads
_set_num_threads.argtypes = [ctypes.c_int]
_set_num_threads.restype = None


def get_num_threads() -> int:
    """Get the maximum number of OpenMP threads available."""
    return _get_num_threads()


def set_num_threads(num_threads: int) -> None:
    """Set the number of OpenMP threads for subsequent operations."""
    if num_threads <= 0:
        raise ValueError("num_threads must be > 0")
    _set_num_threads(num_threads)


def spmv_csr_omp(
    A: csr_matrix, x: np.ndarray, num_threads: int = 1
) -> np.ndarray:
    """
    Sparse matrix-vector multiplication using OpenMP-accelerated CSR.
    
    Parameters:
    - A: Sparse matrix in CSR format (m × n)
    - x: Dense input vector (n,)
    - num_threads: Number of OpenMP threads to use (default: 1)
    
    Returns:
    - y: Dense result vector (m,)
    
    Raises:
    - ValueError: If A is not CSR format or dimensions don't match
    - TypeError: If x is not a numpy array
    """
    if not isinstance(A, csr_matrix):
        raise ValueError("A must be a scipy.sparse.csr_matrix")
    
    if not isinstance(x, np.ndarray):
        raise TypeError("x must be a numpy array")
    
    m, n = A.shape
    if x.shape[0] != n:
        raise ValueError(f"Dimension mismatch: A is {m}×{n}, x is {x.shape[0]}")
    
    if num_threads <= 0:
        raise ValueError("num_threads must be > 0")
    
    # Ensure arrays are C-contiguous and proper dtype
    data = np.asarray(A.data, dtype=np.float64, order='C')
    indices = np.asarray(A.indices, dtype=np.int32, order='C')
    indptr = np.asarray(A.indptr, dtype=np.int32, order='C')
    x_c = np.asarray(x, dtype=np.float64, order='C')
    
    # Allocate output vector
    y = np.zeros(m, dtype=np.float64)
    
    # Call C function
    _spmv_csr_omp(
        m,
        n,
        A.nnz,
        data.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        indptr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        x_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        y.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        num_threads,
    )
    
    return y


def spmv_omp(A: spmatrix, x: np.ndarray, num_threads: int = 1) -> np.ndarray:
    """
    Sparse matrix-vector multiplication with automatic format conversion.
    
    Parameters:
    - A: Sparse matrix (any scipy.sparse format)
    - x: Dense input vector
    - num_threads: Number of OpenMP threads
    
    Returns:
    - y: Dense result vector
    """
    # Convert to CSR if needed
    if not isinstance(A, csr_matrix):
        A = A.tocsr()
    
    return spmv_csr_omp(A, x, num_threads)
