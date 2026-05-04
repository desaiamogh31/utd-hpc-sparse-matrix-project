"""
ctypes wrapper for OpenMP-accelerated sparse-B SpMM.

Provides Python bindings for the compiled spmm_openmp shared library and
benchmarks aligned with the sparse-B serial baseline.
"""

from __future__ import annotations

import ctypes
import gc
import os
import time
from typing import Callable, Dict

import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, spmatrix


_LIB_PATH = os.path.join(os.path.dirname(__file__), "spmm_openmp.so")

if not os.path.exists(_LIB_PATH):
    raise FileNotFoundError(
        f"OpenMP library not found at {_LIB_PATH}\n"
        "Please compile it first with: python build.py"
    )

_lib = ctypes.CDLL(_LIB_PATH)


_TripletArgs = [
    ctypes.c_int,                            # m
    ctypes.c_int,                            # n/shared dimension
    ctypes.c_int,                            # k
    ctypes.POINTER(ctypes.c_double),         # a_data
    ctypes.POINTER(ctypes.c_int),            # a_indices
    ctypes.POINTER(ctypes.c_int),            # a_indptr
    ctypes.POINTER(ctypes.c_double),         # b_data
    ctypes.POINTER(ctypes.c_int),            # b_indices
    ctypes.POINTER(ctypes.c_int),            # b_indptr
    ctypes.POINTER(ctypes.POINTER(ctypes.c_int)),     # out_rows
    ctypes.POINTER(ctypes.POINTER(ctypes.c_int)),     # out_cols
    ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),  # out_vals
    ctypes.POINTER(ctypes.c_int),            # out_nnz
    ctypes.c_int,                            # num_threads
]

_spmm_row_wise_sparse_b_omp = _lib.spmm_row_wise_sparse_b_omp
_spmm_row_wise_sparse_b_omp.argtypes = _TripletArgs
_spmm_row_wise_sparse_b_omp.restype = None

_spmm_outer_product_sparse_b_omp = _lib.spmm_outer_product_sparse_b_omp
_spmm_outer_product_sparse_b_omp.argtypes = _TripletArgs
_spmm_outer_product_sparse_b_omp.restype = None

_spmm_blocked_inner_product_sparse_b_omp = _lib.spmm_blocked_inner_product_sparse_b_omp
_spmm_blocked_inner_product_sparse_b_omp.argtypes = [
    ctypes.c_int,                            # m
    ctypes.c_int,                            # n/shared dimension
    ctypes.c_int,                            # k
    ctypes.POINTER(ctypes.c_double),         # a_data
    ctypes.POINTER(ctypes.c_int),            # a_indices
    ctypes.POINTER(ctypes.c_int),            # a_indptr
    ctypes.POINTER(ctypes.c_double),         # b_data
    ctypes.POINTER(ctypes.c_int),            # b_indices
    ctypes.POINTER(ctypes.c_int),            # b_indptr
    ctypes.c_int,                            # block_k
    ctypes.POINTER(ctypes.POINTER(ctypes.c_int)),
    ctypes.POINTER(ctypes.POINTER(ctypes.c_int)),
    ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int,                            # num_threads
]
_spmm_blocked_inner_product_sparse_b_omp.restype = None

_free_spmm_buffer = _lib.free_spmm_buffer
_free_spmm_buffer.argtypes = [ctypes.c_void_p]
_free_spmm_buffer.restype = None

_get_num_threads = _lib.get_num_threads
_get_num_threads.argtypes = []
_get_num_threads.restype = ctypes.c_int

_set_num_threads = _lib.set_num_threads
_set_num_threads.argtypes = [ctypes.c_int]
_set_num_threads.restype = None


def get_num_threads() -> int:
    """Return the maximum OpenMP thread count reported by the shared library."""
    return _get_num_threads()


def set_num_threads(num_threads: int) -> None:
    """Set the OpenMP thread count for subsequent kernel launches."""
    if num_threads <= 0:
        raise ValueError("num_threads must be > 0")
    _set_num_threads(num_threads)


def _as_csr(matrix: spmatrix) -> csr_matrix:
    if isinstance(matrix, csr_matrix):
        return matrix
    return matrix.tocsr()


def _as_csc(matrix: spmatrix) -> csc_matrix:
    if isinstance(matrix, csc_matrix):
        return matrix
    return matrix.tocsc()


def _validate_sparse_inputs(A: spmatrix, B: spmatrix, num_threads: int) -> tuple[csr_matrix, csr_matrix]:
    if not isinstance(B, spmatrix):
        raise TypeError("B must be a scipy sparse matrix")
    if num_threads <= 0:
        raise ValueError("num_threads must be > 0")

    A_csr = _as_csr(A)
    B_csr = _as_csr(B)

    if A_csr.shape[1] != B_csr.shape[0]:
        raise ValueError(
            f"Dimension mismatch: A is {A_csr.shape}, B is {B_csr.shape}"
        )

    return A_csr, B_csr


def _copy_and_free_triplets(
    rows_ptr: ctypes.POINTER(ctypes.c_int),
    cols_ptr: ctypes.POINTER(ctypes.c_int),
    vals_ptr: ctypes.POINTER(ctypes.c_double),
    nnz: int,
    shape: tuple[int, int],
) -> csr_matrix:
    if nnz == 0:
        return csr_matrix(shape, dtype=np.float64)

    try:
        rows = np.array(np.ctypeslib.as_array(rows_ptr, shape=(nnz,)), copy=True)
        cols = np.array(np.ctypeslib.as_array(cols_ptr, shape=(nnz,)), copy=True)
        vals = np.array(np.ctypeslib.as_array(vals_ptr, shape=(nnz,)), copy=True)
    finally:
        if rows_ptr:
            _free_spmm_buffer(ctypes.cast(rows_ptr, ctypes.c_void_p))
        if cols_ptr:
            _free_spmm_buffer(ctypes.cast(cols_ptr, ctypes.c_void_p))
        if vals_ptr:
            _free_spmm_buffer(ctypes.cast(vals_ptr, ctypes.c_void_p))

    return coo_matrix((vals, (rows, cols)), shape=shape).tocsr()


def _coo_from_kernel_call(
    kernel: Callable[..., None],
    m: int,
    shared_dim: int,
    k: int,
    A_data: np.ndarray,
    A_indices: np.ndarray,
    A_indptr: np.ndarray,
    B_data: np.ndarray,
    B_indices: np.ndarray,
    B_indptr: np.ndarray,
    shape: tuple[int, int],
    num_threads: int,
    block_k: int | None = None,
) -> csr_matrix:
    rows_ptr = ctypes.POINTER(ctypes.c_int)()
    cols_ptr = ctypes.POINTER(ctypes.c_int)()
    vals_ptr = ctypes.POINTER(ctypes.c_double)()
    out_nnz = ctypes.c_int()

    args = [
        m,
        shared_dim,
        k,
        A_data.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        A_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        A_indptr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        B_data.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        B_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        B_indptr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
    ]

    if block_k is not None:
        args.append(block_k)

    args.extend([
        ctypes.byref(rows_ptr),
        ctypes.byref(cols_ptr),
        ctypes.byref(vals_ptr),
        ctypes.byref(out_nnz),
        num_threads,
    ])

    kernel(*args)
    return _copy_and_free_triplets(rows_ptr, cols_ptr, vals_ptr, out_nnz.value, shape)


def spmm_row_wise_sparse_b_omp(
    A: spmatrix, B: spmatrix, num_threads: int = 1
) -> csr_matrix:
    """OpenMP row-wise sparse-sparse SpMM using CSR inputs."""
    A_csr, B_csr = _validate_sparse_inputs(A, B, num_threads)

    a_data = np.asarray(A_csr.data, dtype=np.float64, order="C")
    a_indices = np.asarray(A_csr.indices, dtype=np.int32, order="C")
    a_indptr = np.asarray(A_csr.indptr, dtype=np.int32, order="C")
    b_data = np.asarray(B_csr.data, dtype=np.float64, order="C")
    b_indices = np.asarray(B_csr.indices, dtype=np.int32, order="C")
    b_indptr = np.asarray(B_csr.indptr, dtype=np.int32, order="C")

    return _coo_from_kernel_call(
        _spmm_row_wise_sparse_b_omp,
        A_csr.shape[0],
        A_csr.shape[1],
        B_csr.shape[1],
        a_data,
        a_indices,
        a_indptr,
        b_data,
        b_indices,
        b_indptr,
        (A_csr.shape[0], B_csr.shape[1]),
        num_threads,
    )


def spmm_outer_product_sparse_b_omp(
    A: spmatrix, B: spmatrix, num_threads: int = 1
) -> csr_matrix:
    """OpenMP outer-product sparse-sparse SpMM using A in CSC and B in CSR."""
    A_csr, B_csr = _validate_sparse_inputs(A, B, num_threads)
    A_csc = _as_csc(A_csr)

    a_data = np.asarray(A_csc.data, dtype=np.float64, order="C")
    a_indices = np.asarray(A_csc.indices, dtype=np.int32, order="C")
    a_indptr = np.asarray(A_csc.indptr, dtype=np.int32, order="C")
    b_data = np.asarray(B_csr.data, dtype=np.float64, order="C")
    b_indices = np.asarray(B_csr.indices, dtype=np.int32, order="C")
    b_indptr = np.asarray(B_csr.indptr, dtype=np.int32, order="C")

    return _coo_from_kernel_call(
        _spmm_outer_product_sparse_b_omp,
        A_csr.shape[0],
        A_csr.shape[1],
        B_csr.shape[1],
        a_data,
        a_indices,
        a_indptr,
        b_data,
        b_indices,
        b_indptr,
        (A_csr.shape[0], B_csr.shape[1]),
        num_threads,
    )


def spmm_blocked_inner_product_sparse_b_omp(
    A: spmatrix, B: spmatrix, num_threads: int = 1, block_k: int = 32
) -> csr_matrix:
    """OpenMP blocked sparse-sparse SpMM using CSR inputs."""
    if block_k <= 0:
        raise ValueError("block_k must be > 0")

    A_csr, B_csr = _validate_sparse_inputs(A, B, num_threads)

    a_data = np.asarray(A_csr.data, dtype=np.float64, order="C")
    a_indices = np.asarray(A_csr.indices, dtype=np.int32, order="C")
    a_indptr = np.asarray(A_csr.indptr, dtype=np.int32, order="C")
    b_data = np.asarray(B_csr.data, dtype=np.float64, order="C")
    b_indices = np.asarray(B_csr.indices, dtype=np.int32, order="C")
    b_indptr = np.asarray(B_csr.indptr, dtype=np.int32, order="C")

    return _coo_from_kernel_call(
        _spmm_blocked_inner_product_sparse_b_omp,
        A_csr.shape[0],
        A_csr.shape[1],
        B_csr.shape[1],
        a_data,
        a_indices,
        a_indptr,
        b_data,
        b_indices,
        b_indptr,
        (A_csr.shape[0], B_csr.shape[1]),
        num_threads,
        block_k=block_k,
    )


def spmm_sparse_b_omp(
    A: spmatrix,
    B: spmatrix,
    algorithm: str = "row-wise",
    num_threads: int = 1,
    block_k: int = 32,
) -> csr_matrix:
    """Dispatch to one of the OpenMP sparse-B SpMM kernels."""
    if algorithm == "row-wise":
        return spmm_row_wise_sparse_b_omp(A, B, num_threads=num_threads)
    if algorithm == "outer-product":
        return spmm_outer_product_sparse_b_omp(A, B, num_threads=num_threads)
    if algorithm == "blocked":
        return spmm_blocked_inner_product_sparse_b_omp(
            A, B, num_threads=num_threads, block_k=block_k
        )

    raise ValueError(f"Unknown algorithm: {algorithm}")


def benchmark_spmm_algorithm_sparse_b_openmp(
    A: spmatrix,
    B: spmatrix,
    algorithm: str = "row-wise",
    repeats: int = 5,
    num_threads: int = 1,
    block_k: int = 32,
) -> Dict[str, float]:
    """Benchmark one sparse-B OpenMP SpMM algorithm."""
    A_csr, B_csr = _validate_sparse_inputs(A, B, num_threads)

    if algorithm == "scipy":
        warmup = lambda: (A_csr @ B_csr).tocsr()
    else:
        warmup = lambda: spmm_sparse_b_omp(
            A_csr, B_csr, algorithm=algorithm, num_threads=num_threads, block_k=block_k
        )

    _ = warmup()

    times = []
    for _ in range(repeats):
        gc.collect()
        start = time.perf_counter()
        C = warmup()
        end = time.perf_counter()
        times.append(end - start)

    mean_time = float(np.mean(times))
    std_time = float(np.std(times))
    min_time = float(np.min(times))

    nnz_a = int(A_csr.nnz)
    nnz_b = int(B_csr.nnz)
    nnz_c = int(C.nnz)
    flops = 2 * nnz_a * nnz_b / A_csr.shape[1] if A_csr.shape[1] > 0 else 0.0
    gflops = flops / (mean_time * 1e9) if mean_time > 0 else 0.0

    memory_a_mb = (
        A_csr.data.nbytes + A_csr.indices.nbytes + A_csr.indptr.nbytes
    ) / (1024 * 1024)
    memory_b_mb = (
        B_csr.data.nbytes + B_csr.indices.nbytes + B_csr.indptr.nbytes
    ) / (1024 * 1024)
    memory_c_mb = (
        C.data.nbytes + C.indices.nbytes + C.indptr.nbytes
    ) / (1024 * 1024)

    return {
        "algorithm": algorithm,
        "mean_time": mean_time,
        "std_time": std_time,
        "min_time": min_time,
        "gflops": gflops,
        "nnz_a": nnz_a,
        "nnz_b": nnz_b,
        "nnz_c": nnz_c,
        "memory_a_mb": memory_a_mb,
        "memory_b_mb": memory_b_mb,
        "memory_c_mb": memory_c_mb,
        "num_threads": num_threads,
    }
