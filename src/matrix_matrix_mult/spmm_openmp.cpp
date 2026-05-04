#include <omp.h>

#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

struct TripletBuffer {
    std::vector<int> rows;
    std::vector<int> cols;
    std::vector<double> vals;
};

void copy_triplets_to_c_arrays(
    const std::vector<TripletBuffer>& buffers,
    int** out_rows,
    int** out_cols,
    double** out_vals,
    int* out_nnz
) {
    std::size_t total = 0;
    for (const auto& buffer : buffers) {
        total += buffer.rows.size();
    }

    *out_nnz = static_cast<int>(total);
    if (total == 0) {
        *out_rows = nullptr;
        *out_cols = nullptr;
        *out_vals = nullptr;
        return;
    }

    *out_rows = static_cast<int*>(std::malloc(total * sizeof(int)));
    *out_cols = static_cast<int*>(std::malloc(total * sizeof(int)));
    *out_vals = static_cast<double*>(std::malloc(total * sizeof(double)));

    std::size_t offset = 0;
    for (const auto& buffer : buffers) {
        const std::size_t count = buffer.rows.size();
        if (count == 0) {
            continue;
        }

        std::memcpy(*out_rows + offset, buffer.rows.data(), count * sizeof(int));
        std::memcpy(*out_cols + offset, buffer.cols.data(), count * sizeof(int));
        std::memcpy(*out_vals + offset, buffer.vals.data(), count * sizeof(double));
        offset += count;
    }
}

int resolve_thread_count(int requested_threads) {
    if (requested_threads > 0) {
        omp_set_num_threads(requested_threads);
        return requested_threads;
    }
    return omp_get_max_threads();
}

}  // namespace

extern "C" {

void spmm_row_wise_sparse_b_omp(
    int m,
    int n,
    int k,
    const double* a_data,
    const int* a_indices,
    const int* a_indptr,
    const double* b_data,
    const int* b_indices,
    const int* b_indptr,
    int** out_rows,
    int** out_cols,
    double** out_vals,
    int* out_nnz,
    int num_threads
) {
    const int thread_count = resolve_thread_count(num_threads);
    std::vector<TripletBuffer> buffers(thread_count);

    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        auto& buffer = buffers[tid];

        #pragma omp for schedule(dynamic)
        for (int i = 0; i < m; ++i) {
            const int row_start = a_indptr[i];
            const int row_end = a_indptr[i + 1];
            if (row_start == row_end) {
                continue;
            }

            std::vector<double> accum(k, 0.0);
            std::vector<unsigned char> touched(k, 0);
            std::vector<int> active_cols;

            for (int idx_a = row_start; idx_a < row_end; ++idx_a) {
                const int l = a_indices[idx_a];
                const double a_val = a_data[idx_a];

                for (int idx_b = b_indptr[l]; idx_b < b_indptr[l + 1]; ++idx_b) {
                    const int col = b_indices[idx_b];
                    if (!touched[col]) {
                        touched[col] = 1;
                        active_cols.push_back(col);
                    }
                    accum[col] += a_val * b_data[idx_b];
                }
            }

            for (int col : active_cols) {
                const double value = accum[col];
                if (value != 0.0) {
                    buffer.rows.push_back(i);
                    buffer.cols.push_back(col);
                    buffer.vals.push_back(value);
                }
            }
        }
    }

    copy_triplets_to_c_arrays(buffers, out_rows, out_cols, out_vals, out_nnz);
}

void spmm_outer_product_sparse_b_omp(
    int m,
    int n,
    int k,
    const double* a_data,
    const int* a_indices,
    const int* a_indptr,
    const double* b_data,
    const int* b_indices,
    const int* b_indptr,
    int** out_rows,
    int** out_cols,
    double** out_vals,
    int* out_nnz,
    int num_threads
) {
    const int thread_count = resolve_thread_count(num_threads);
    std::vector<TripletBuffer> buffers(thread_count);

    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        auto& buffer = buffers[tid];

        #pragma omp for schedule(dynamic)
        for (int shared_idx = 0; shared_idx < n; ++shared_idx) {
            const int col_start_a = a_indptr[shared_idx];
            const int col_end_a = a_indptr[shared_idx + 1];
            const int row_start_b = b_indptr[shared_idx];
            const int row_end_b = b_indptr[shared_idx + 1];

            if (col_start_a == col_end_a || row_start_b == row_end_b) {
                continue;
            }

            for (int idx_a = col_start_a; idx_a < col_end_a; ++idx_a) {
                const int row = a_indices[idx_a];
                const double a_val = a_data[idx_a];

                for (int idx_b = row_start_b; idx_b < row_end_b; ++idx_b) {
                    const int col = b_indices[idx_b];
                    const double value = a_val * b_data[idx_b];
                    if (value != 0.0) {
                        buffer.rows.push_back(row);
                        buffer.cols.push_back(col);
                        buffer.vals.push_back(value);
                    }
                }
            }
        }
    }

    copy_triplets_to_c_arrays(buffers, out_rows, out_cols, out_vals, out_nnz);
}

void spmm_blocked_inner_product_sparse_b_omp(
    int m,
    int n,
    int k,
    const double* a_data,
    const int* a_indices,
    const int* a_indptr,
    const double* b_data,
    const int* b_indices,
    const int* b_indptr,
    int block_k,
    int** out_rows,
    int** out_cols,
    double** out_vals,
    int* out_nnz,
    int num_threads
) {
    const int thread_count = resolve_thread_count(num_threads);
    const int block_size = block_k > 0 ? block_k : 32;
    std::vector<TripletBuffer> buffers(thread_count);

    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        auto& buffer = buffers[tid];

        #pragma omp for schedule(dynamic)
        for (int i = 0; i < m; ++i) {
            const int row_start = a_indptr[i];
            const int row_end = a_indptr[i + 1];
            if (row_start == row_end) {
                continue;
            }

            for (int block_start = 0; block_start < k; block_start += block_size) {
                const int block_end = block_start + block_size < k ? block_start + block_size : k;
                const int block_width = block_end - block_start;

                std::vector<double> accum(block_width, 0.0);
                std::vector<unsigned char> touched(block_width, 0);
                std::vector<int> active_offsets;

                for (int idx_a = row_start; idx_a < row_end; ++idx_a) {
                    const int l = a_indices[idx_a];
                    const double a_val = a_data[idx_a];

                    for (int idx_b = b_indptr[l]; idx_b < b_indptr[l + 1]; ++idx_b) {
                        const int col = b_indices[idx_b];
                        if (col < block_start || col >= block_end) {
                            continue;
                        }

                        const int offset = col - block_start;
                        if (!touched[offset]) {
                            touched[offset] = 1;
                            active_offsets.push_back(offset);
                        }
                        accum[offset] += a_val * b_data[idx_b];
                    }
                }

                for (int offset : active_offsets) {
                    const double value = accum[offset];
                    if (value != 0.0) {
                        buffer.rows.push_back(i);
                        buffer.cols.push_back(block_start + offset);
                        buffer.vals.push_back(value);
                    }
                }
            }
        }
    }

    copy_triplets_to_c_arrays(buffers, out_rows, out_cols, out_vals, out_nnz);
}

void free_spmm_buffer(void* ptr) {
    std::free(ptr);
}

int get_num_threads() {
    return omp_get_max_threads();
}

void set_num_threads(int num_threads) {
    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    }
}

}  // extern "C"
