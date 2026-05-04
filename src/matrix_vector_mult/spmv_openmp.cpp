#include <omp.h>
#include <cstring>
#include <cmath>

/*
 * Sparse Matrix-Vector Multiplication (SpMV) using OpenMP parallelization.
 * 
 * This implementation focuses on CSR (Compressed Sparse Row) format,
 * parallelizing the row loop with OpenMP.
 * 
 * CSR format:
 *   - data: Non-zero values
 *   - indices: Column indices for each non-zero
 *   - indptr: Row pointers (indptr[i] to indptr[i+1] is row i's range)
 * 
 * To compile:
 *   gcc -O3 -fopenmp -fPIC -shared -o spmv_openmp.so spmv_openmp.cpp
 * or:
 *   gcc -O3 -fopenmp -fPIC -shared -o spmv_openmp.so spmv_openmp.c
 *
 * See build.sh for automated compilation.
 */

extern "C" {
    
    /*
     * SpMV with OpenMP parallelization over rows.
     * 
     * Parameters:
     *   - m: Number of rows
     *   - n: Number of columns
     *   - nnz: Number of non-zeros
     *   - data: Non-zero values (nnz,)
     *   - indices: Column indices (nnz,)
     *   - indptr: Row pointers (m+1,)
     *   - x: Input vector (n,)
     *   - y: Output vector (m,) - preallocated
     *   - num_threads: Number of OpenMP threads to use (0 = use default)
     */
    void spmv_csr_omp(
        int m,
        int n,
        int nnz,
        const double *data,
        const int *indices,
        const int *indptr,
        const double *x,
        double *y,
        int num_threads
    ) {
        // Set number of threads if specified (num_threads > 0)
        if (num_threads > 0) {
            omp_set_num_threads(num_threads);
        }
        
        // Initialize output vector to zero
        #pragma omp parallel for
        for (int i = 0; i < m; i++) {
            y[i] = 0.0;
        }
        
        // Main SpMV computation: parallelize over rows
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < m; i++) {
            double sum = 0.0;
            
            // Iterate through non-zeros in row i
            for (int j_idx = indptr[i]; j_idx < indptr[i + 1]; j_idx++) {
                int j = indices[j_idx];
                sum += data[j_idx] * x[j];
            }
            
            y[i] = sum;
        }
    }
    
    
    /*
     * Get the number of OpenMP threads available.
     * 
     * Returns:
     *   - Number of available threads
     */
    int get_num_threads() {
        return omp_get_max_threads();
    }
    
    
    /*
     * Set the number of OpenMP threads for subsequent operations.
     * 
     * Parameters:
     *   - num_threads: Number of threads to use (must be > 0)
     */
    void set_num_threads(int num_threads) {
        if (num_threads > 0) {
            omp_set_num_threads(num_threads);
        }
    }
    
}  // extern "C"
