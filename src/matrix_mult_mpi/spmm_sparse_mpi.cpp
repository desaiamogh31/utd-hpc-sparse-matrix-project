#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr int BLOCK_K_DEFAULT = 32;
const std::vector<std::string> ALGORITHMS{
    "row-wise",
    "outer-product",
    "blocked",
    "scipy",
};

struct CSRMatrix {
    int rows = 0;
    int cols = 0;
    std::vector<int> indptr;
    std::vector<int> indices;
    std::vector<double> data;

    int nnz() const {
        return static_cast<int>(data.size());
    }
};

struct CSCMatrix {
    int rows = 0;
    int cols = 0;
    std::vector<int> indptr;
    std::vector<int> indices;
    std::vector<double> data;
};

struct ParsedArgs {
    std::string output = "benchmark_spmm_sparse_mpi.csv";
    std::string outdir = "results_hpc_spmm_mpi";
    std::string cache_dir = "../matrix_matrix_mult/matrices";
    int repeats = 3;
    std::vector<int> b_cols{4, 8, 16};
    std::vector<double> sparsity{0.10};
    std::vector<std::string> matrices{
        "1138_bus",
        "abb313",
        "delaunay_n15",
        "bcsstk30",
        "delaunay_n19",
        "pkustk14",
    };
    bool validate = false;
};

struct MatrixLoadResult {
    bool ok = false;
    std::string source;
    CSRMatrix matrix;
};

struct AlgorithmMetrics {
    double mean_time = 0.0;
    double std_time = 0.0;
    double min_time = 0.0;
    double gflops = 0.0;
    int nnz_c = 0;
    double memory_a_mb_total = 0.0;
    double memory_b_mb_per_rank = 0.0;
    double memory_c_mb_total = 0.0;
    int validation_ok = -1;
};

std::string csv_escape(const std::string& value) {
    if (value.find_first_of(",\"\n") == std::string::npos) {
        return value;
    }
    std::string escaped = "\"";
    for (char ch : value) {
        if (ch == '"') {
            escaped += "\"\"";
        } else {
            escaped += ch;
        }
    }
    escaped += "\"";
    return escaped;
}

double sparse_memory_mb(const CSRMatrix& matrix) {
    const double bytes =
        static_cast<double>(matrix.data.size() * sizeof(double) +
                            matrix.indices.size() * sizeof(int) +
                            matrix.indptr.size() * sizeof(int));
    return bytes / (1024.0 * 1024.0);
}

CSCMatrix csr_to_csc(const CSRMatrix& csr) {
    CSCMatrix csc;
    csc.rows = csr.rows;
    csc.cols = csr.cols;
    csc.indptr.assign(csr.cols + 1, 0);
    csc.indices.resize(csr.indices.size());
    csc.data.resize(csr.data.size());

    for (int col : csr.indices) {
        csc.indptr[col + 1] += 1;
    }
    for (int col = 0; col < csr.cols; ++col) {
        csc.indptr[col + 1] += csc.indptr[col];
    }

    std::vector<int> next = csc.indptr;
    for (int row = 0; row < csr.rows; ++row) {
        for (int idx = csr.indptr[row]; idx < csr.indptr[row + 1]; ++idx) {
            const int col = csr.indices[idx];
            const int dest = next[col]++;
            csc.indices[dest] = row;
            csc.data[dest] = csr.data[idx];
        }
    }

    return csc;
}

CSRMatrix coo_to_csr(
    int rows,
    int cols,
    const std::vector<int>& row_idx,
    const std::vector<int>& col_idx,
    const std::vector<double>& vals
) {
    CSRMatrix csr;
    csr.rows = rows;
    csr.cols = cols;
    csr.indptr.assign(rows + 1, 0);

    const std::size_t nnz = vals.size();
    std::vector<int> order(nnz);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int lhs, int rhs) {
        if (row_idx[lhs] != row_idx[rhs]) {
            return row_idx[lhs] < row_idx[rhs];
        }
        return col_idx[lhs] < col_idx[rhs];
    });

    int current_row = 0;
    bool have_pending = false;
    int pending_row = 0;
    int pending_col = 0;
    double pending_val = 0.0;

    auto flush_pending = [&]() {
        if (!have_pending || pending_val == 0.0) {
            have_pending = false;
            return;
        }
        while (current_row <= pending_row) {
            csr.indptr[current_row] = static_cast<int>(csr.indices.size());
            current_row += 1;
        }
        csr.indices.push_back(pending_col);
        csr.data.push_back(pending_val);
        have_pending = false;
    };

    for (int pos : order) {
        const int row = row_idx[pos];
        const int col = col_idx[pos];
        const double val = vals[pos];
        if (!have_pending) {
            pending_row = row;
            pending_col = col;
            pending_val = val;
            have_pending = true;
        } else if (row == pending_row && col == pending_col) {
            pending_val += val;
        } else {
            flush_pending();
            pending_row = row;
            pending_col = col;
            pending_val = val;
            have_pending = true;
        }
    }
    flush_pending();

    while (current_row <= rows) {
        csr.indptr[current_row] = static_cast<int>(csr.indices.size());
        current_row += 1;
    }

    return csr;
}

bool csr_equal(const CSRMatrix& lhs, const CSRMatrix& rhs, double tol = 1e-10) {
    if (lhs.rows != rhs.rows || lhs.cols != rhs.cols) {
        return false;
    }
    if (lhs.indptr != rhs.indptr || lhs.indices != rhs.indices) {
        return false;
    }
    if (lhs.data.size() != rhs.data.size()) {
        return false;
    }
    for (std::size_t i = 0; i < lhs.data.size(); ++i) {
        if (std::fabs(lhs.data[i] - rhs.data[i]) > tol) {
            return false;
        }
    }
    return true;
}

std::vector<std::pair<int, int>> compute_row_partitions(int num_rows, int num_ranks) {
    std::vector<std::pair<int, int>> partitions;
    partitions.reserve(num_ranks);
    const int base = num_rows / num_ranks;
    const int remainder = num_rows % num_ranks;
    int start = 0;
    for (int rank = 0; rank < num_ranks; ++rank) {
        const int rows = base + (rank < remainder ? 1 : 0);
        const int end = start + rows;
        partitions.emplace_back(start, end);
        start = end;
    }
    return partitions;
}

CSRMatrix slice_csr_rows(const CSRMatrix& full, int row_start, int row_end) {
    CSRMatrix local;
    local.rows = row_end - row_start;
    local.cols = full.cols;
    local.indptr.resize(local.rows + 1, 0);

    const int nnz_start = full.indptr[row_start];
    const int nnz_end = full.indptr[row_end];
    local.indices.assign(
        full.indices.begin() + nnz_start,
        full.indices.begin() + nnz_end
    );
    local.data.assign(
        full.data.begin() + nnz_start,
        full.data.begin() + nnz_end
    );

    for (int i = 0; i < local.rows + 1; ++i) {
        local.indptr[i] = full.indptr[row_start + i] - nnz_start;
    }
    return local;
}

void send_csr_block(const CSRMatrix& matrix, int dest_rank) {
    const int meta[3] = {
        matrix.rows,
        matrix.cols,
        static_cast<int>(matrix.data.size()),
    };
    MPI_Send(meta, 3, MPI_INT, dest_rank, 0, MPI_COMM_WORLD);
    MPI_Send(matrix.indptr.data(), static_cast<int>(matrix.indptr.size()), MPI_INT, dest_rank, 1, MPI_COMM_WORLD);
    if (!matrix.indices.empty()) {
        MPI_Send(matrix.indices.data(), static_cast<int>(matrix.indices.size()), MPI_INT, dest_rank, 2, MPI_COMM_WORLD);
        MPI_Send(matrix.data.data(), static_cast<int>(matrix.data.size()), MPI_DOUBLE, dest_rank, 3, MPI_COMM_WORLD);
    }
}

CSRMatrix recv_csr_block(int source_rank) {
    int meta[3] = {0, 0, 0};
    MPI_Recv(meta, 3, MPI_INT, source_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    CSRMatrix matrix;
    matrix.rows = meta[0];
    matrix.cols = meta[1];
    const int nnz = meta[2];
    matrix.indptr.resize(matrix.rows + 1);
    matrix.indices.resize(nnz);
    matrix.data.resize(nnz);

    MPI_Recv(matrix.indptr.data(), static_cast<int>(matrix.indptr.size()), MPI_INT, source_rank, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    if (nnz > 0) {
        MPI_Recv(matrix.indices.data(), nnz, MPI_INT, source_rank, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(matrix.data.data(), nnz, MPI_DOUBLE, source_rank, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    return matrix;
}

CSRMatrix distribute_matrix_rows(const std::optional<CSRMatrix>& root_matrix) {
    int rank = 0;
    int world = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);

    if (rank == 0) {
        const auto& full = *root_matrix;
        const auto partitions = compute_row_partitions(full.rows, world);
        CSRMatrix local = slice_csr_rows(full, partitions[0].first, partitions[0].second);
        for (int dest = 1; dest < world; ++dest) {
            const CSRMatrix block = slice_csr_rows(full, partitions[dest].first, partitions[dest].second);
            send_csr_block(block, dest);
        }
        return local;
    }
    return recv_csr_block(0);
}

void broadcast_csr_matrix(CSRMatrix& matrix, int root = 0) {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int meta[3];
    if (rank == root) {
        meta[0] = matrix.rows;
        meta[1] = matrix.cols;
        meta[2] = static_cast<int>(matrix.data.size());
    }
    MPI_Bcast(meta, 3, MPI_INT, root, MPI_COMM_WORLD);

    if (rank != root) {
        matrix.rows = meta[0];
        matrix.cols = meta[1];
        matrix.indptr.resize(matrix.rows + 1);
        matrix.indices.resize(meta[2]);
        matrix.data.resize(meta[2]);
    }

    MPI_Bcast(matrix.indptr.data(), static_cast<int>(matrix.indptr.size()), MPI_INT, root, MPI_COMM_WORLD);
    if (meta[2] > 0) {
        MPI_Bcast(matrix.indices.data(), meta[2], MPI_INT, root, MPI_COMM_WORLD);
        MPI_Bcast(matrix.data.data(), meta[2], MPI_DOUBLE, root, MPI_COMM_WORLD);
    }
}

std::optional<fs::path> resolve_matrix_path(const std::string& matrix_name, const std::string& cache_dir) {
    const fs::path cache = fs::path(cache_dir);
    const fs::path local = cache / (matrix_name + ".mtx");
    if (fs::exists(local)) {
        return local;
    }

    const fs::path phase2 = cache / ".." / ".." / "matrix_vector_mult" / "matrices" / (matrix_name + ".mtx");
    if (fs::exists(phase2)) {
        return phase2.lexically_normal();
    }

    return std::nullopt;
}

MatrixLoadResult load_matrix_market(const std::string& matrix_name, const std::string& cache_dir) {
    MatrixLoadResult result;
    const auto resolved = resolve_matrix_path(matrix_name, cache_dir);
    if (!resolved.has_value()) {
        return result;
    }

    std::ifstream file(resolved->string());
    if (!file) {
        return result;
    }

    std::string header;
    std::getline(file, header);
    std::istringstream header_stream(header);
    std::string banner;
    std::string object;
    std::string format;
    std::string field;
    std::string symmetry;
    header_stream >> banner >> object >> format >> field >> symmetry;
    if (banner != "%%MatrixMarket" || object != "matrix" || format != "coordinate") {
        return result;
    }

    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty() && line[0] != '%') {
            break;
        }
    }
    if (line.empty()) {
        return result;
    }

    std::istringstream size_stream(line);
    int rows = 0;
    int cols = 0;
    int nnz = 0;
    size_stream >> rows >> cols >> nnz;

    std::vector<int> row_idx;
    std::vector<int> col_idx;
    std::vector<double> vals;
    row_idx.reserve(symmetry == "symmetric" ? nnz * 2 : nnz);
    col_idx.reserve(symmetry == "symmetric" ? nnz * 2 : nnz);
    vals.reserve(symmetry == "symmetric" ? nnz * 2 : nnz);

    for (int entry = 0; entry < nnz; ++entry) {
        if (!std::getline(file, line)) {
            break;
        }
        if (line.empty() || line[0] == '%') {
            --entry;
            continue;
        }

        std::istringstream entry_stream(line);
        int row = 0;
        int col = 0;
        double value = 1.0;
        entry_stream >> row >> col;
        if (field != "pattern") {
            entry_stream >> value;
        }
        row -= 1;
        col -= 1;

        row_idx.push_back(row);
        col_idx.push_back(col);
        vals.push_back(value);

        if (symmetry == "symmetric" && row != col) {
            row_idx.push_back(col);
            col_idx.push_back(row);
            vals.push_back(value);
        }
    }

    result.ok = true;
    result.source = resolved->string();
    result.matrix = coo_to_csr(rows, cols, row_idx, col_idx, vals);
    return result;
}

CSRMatrix generate_sparse_b(int rows, int cols, double density, std::uint64_t seed = 42) {
    if (rows <= 0 || cols <= 0 || density <= 0.0) {
        return CSRMatrix{rows, cols, std::vector<int>(rows + 1, 0), {}, {}};
    }

    const std::uint64_t total_slots = static_cast<std::uint64_t>(rows) * static_cast<std::uint64_t>(cols);
    const std::uint64_t target_nnz = std::min<std::uint64_t>(
        total_slots,
        static_cast<std::uint64_t>(std::llround(static_cast<long double>(total_slots) * density))
    );

    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int> row_dist(0, rows - 1);
    std::uniform_int_distribution<int> col_dist(0, cols - 1);
    std::uniform_real_distribution<double> val_dist(0.0, 1.0);

    std::unordered_set<std::uint64_t> used;
    used.reserve(static_cast<std::size_t>(target_nnz * 2 + 1));

    std::vector<int> row_idx;
    std::vector<int> col_idx;
    std::vector<double> vals;
    row_idx.reserve(static_cast<std::size_t>(target_nnz));
    col_idx.reserve(static_cast<std::size_t>(target_nnz));
    vals.reserve(static_cast<std::size_t>(target_nnz));

    while (used.size() < target_nnz) {
        const int row = row_dist(rng);
        const int col = col_dist(rng);
        const std::uint64_t key =
            static_cast<std::uint64_t>(row) * static_cast<std::uint64_t>(cols) + static_cast<std::uint64_t>(col);
        if (!used.insert(key).second) {
            continue;
        }
        row_idx.push_back(row);
        col_idx.push_back(col);
        vals.push_back(val_dist(rng));
    }

    return coo_to_csr(rows, cols, row_idx, col_idx, vals);
}

CSRMatrix spmm_reference_sparse_b(const CSRMatrix& a_csr, const CSRMatrix& b_csr) {
    std::vector<int> rows;
    std::vector<int> cols;
    std::vector<double> vals;

    for (int i = 0; i < a_csr.rows; ++i) {
        std::unordered_map<int, double> accum;
        for (int idx_a = a_csr.indptr[i]; idx_a < a_csr.indptr[i + 1]; ++idx_a) {
            const int shared = a_csr.indices[idx_a];
            const double a_val = a_csr.data[idx_a];
            for (int idx_b = b_csr.indptr[shared]; idx_b < b_csr.indptr[shared + 1]; ++idx_b) {
                accum[b_csr.indices[idx_b]] += a_val * b_csr.data[idx_b];
            }
        }

        std::vector<std::pair<int, double>> ordered(accum.begin(), accum.end());
        std::sort(ordered.begin(), ordered.end(), [](const auto& lhs, const auto& rhs) {
            return lhs.first < rhs.first;
        });
        for (const auto& [col, value] : ordered) {
            if (value != 0.0) {
                rows.push_back(i);
                cols.push_back(col);
                vals.push_back(value);
            }
        }
    }

    return coo_to_csr(a_csr.rows, b_csr.cols, rows, cols, vals);
}

CSRMatrix spmm_row_wise_sparse_b(const CSRMatrix& a_csr, const CSRMatrix& b_csr) {
    const int k = b_csr.cols;
    std::vector<int> rows;
    std::vector<int> cols;
    std::vector<double> vals;

    for (int i = 0; i < a_csr.rows; ++i) {
        std::vector<double> accum(k, 0.0);
        std::vector<unsigned char> touched(k, 0);
        std::vector<int> active_cols;

        for (int idx_a = a_csr.indptr[i]; idx_a < a_csr.indptr[i + 1]; ++idx_a) {
            const int shared = a_csr.indices[idx_a];
            const double a_val = a_csr.data[idx_a];
            for (int idx_b = b_csr.indptr[shared]; idx_b < b_csr.indptr[shared + 1]; ++idx_b) {
                const int col = b_csr.indices[idx_b];
                if (!touched[col]) {
                    touched[col] = 1;
                    active_cols.push_back(col);
                }
                accum[col] += a_val * b_csr.data[idx_b];
            }
        }

        std::sort(active_cols.begin(), active_cols.end());
        for (int col : active_cols) {
            const double value = accum[col];
            if (value != 0.0) {
                rows.push_back(i);
                cols.push_back(col);
                vals.push_back(value);
            }
        }
    }

    return coo_to_csr(a_csr.rows, b_csr.cols, rows, cols, vals);
}

CSRMatrix spmm_outer_product_sparse_b(const CSCMatrix& a_csc, const CSRMatrix& b_csr) {
    std::vector<int> rows;
    std::vector<int> cols;
    std::vector<double> vals;

    for (int shared = 0; shared < a_csc.cols; ++shared) {
        const int col_start_a = a_csc.indptr[shared];
        const int col_end_a = a_csc.indptr[shared + 1];
        const int row_start_b = b_csr.indptr[shared];
        const int row_end_b = b_csr.indptr[shared + 1];
        if (col_start_a == col_end_a || row_start_b == row_end_b) {
            continue;
        }

        for (int idx_a = col_start_a; idx_a < col_end_a; ++idx_a) {
            const int row = a_csc.indices[idx_a];
            const double a_val = a_csc.data[idx_a];
            for (int idx_b = row_start_b; idx_b < row_end_b; ++idx_b) {
                const double value = a_val * b_csr.data[idx_b];
                if (value != 0.0) {
                    rows.push_back(row);
                    cols.push_back(b_csr.indices[idx_b]);
                    vals.push_back(value);
                }
            }
        }
    }

    return coo_to_csr(a_csc.rows, b_csr.cols, rows, cols, vals);
}

CSRMatrix spmm_blocked_inner_product_sparse_b(
    const CSRMatrix& a_csr,
    const CSRMatrix& b_csr,
    int block_k = BLOCK_K_DEFAULT
) {
    std::vector<int> rows;
    std::vector<int> cols;
    std::vector<double> vals;
    const int k = b_csr.cols;

    for (int i = 0; i < a_csr.rows; ++i) {
        for (int block_start = 0; block_start < k; block_start += block_k) {
            const int block_end = std::min(block_start + block_k, k);
            const int block_width = block_end - block_start;
            std::vector<double> accum(block_width, 0.0);
            std::vector<unsigned char> touched(block_width, 0);
            std::vector<int> active_offsets;

            for (int idx_a = a_csr.indptr[i]; idx_a < a_csr.indptr[i + 1]; ++idx_a) {
                const int shared = a_csr.indices[idx_a];
                const double a_val = a_csr.data[idx_a];
                for (int idx_b = b_csr.indptr[shared]; idx_b < b_csr.indptr[shared + 1]; ++idx_b) {
                    const int col = b_csr.indices[idx_b];
                    if (col < block_start || col >= block_end) {
                        continue;
                    }
                    const int offset = col - block_start;
                    if (!touched[offset]) {
                        touched[offset] = 1;
                        active_offsets.push_back(offset);
                    }
                    accum[offset] += a_val * b_csr.data[idx_b];
                }
            }

            std::sort(active_offsets.begin(), active_offsets.end());
            for (int offset : active_offsets) {
                const double value = accum[offset];
                if (value != 0.0) {
                    rows.push_back(i);
                    cols.push_back(block_start + offset);
                    vals.push_back(value);
                }
            }
        }
    }

    return coo_to_csr(a_csr.rows, b_csr.cols, rows, cols, vals);
}

CSRMatrix run_algorithm(
    const std::string& algorithm,
    const CSRMatrix& a_csr,
    const CSCMatrix& a_csc,
    const CSRMatrix& b_csr
) {
    if (algorithm == "row-wise") {
        return spmm_row_wise_sparse_b(a_csr, b_csr);
    }
    if (algorithm == "outer-product") {
        return spmm_outer_product_sparse_b(a_csc, b_csr);
    }
    if (algorithm == "blocked") {
        return spmm_blocked_inner_product_sparse_b(a_csr, b_csr);
    }
    return spmm_reference_sparse_b(a_csr, b_csr);
}

AlgorithmMetrics benchmark_algorithm_mpi(
    const std::string& algorithm,
    const CSRMatrix& local_a_csr,
    const CSCMatrix& local_a_csc,
    const CSRMatrix& b_csr,
    int global_nnz_a,
    int repeats,
    bool validate
) {
    AlgorithmMetrics metrics;

    // Warmup
    CSRMatrix local_c = run_algorithm(algorithm, local_a_csr, local_a_csc, b_csr);

    std::vector<double> times;
    times.reserve(repeats);
    for (int rep = 0; rep < repeats; ++rep) {
        MPI_Barrier(MPI_COMM_WORLD);
        const double start = MPI_Wtime();
        local_c = run_algorithm(algorithm, local_a_csr, local_a_csc, b_csr);
        const double local_elapsed = MPI_Wtime() - start;
        double global_elapsed = 0.0;
        MPI_Allreduce(&local_elapsed, &global_elapsed, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        times.push_back(global_elapsed);
    }

    const double sum = std::accumulate(times.begin(), times.end(), 0.0);
    metrics.mean_time = sum / static_cast<double>(times.size());
    metrics.min_time = *std::min_element(times.begin(), times.end());
    double variance = 0.0;
    for (double t : times) {
        const double delta = t - metrics.mean_time;
        variance += delta * delta;
    }
    metrics.std_time = std::sqrt(variance / static_cast<double>(times.size()));

    const int local_nnz_c = local_c.nnz();
    MPI_Allreduce(&local_nnz_c, &metrics.nnz_c, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    const double local_mem_a = sparse_memory_mb(local_a_csr);
    const double local_mem_c = sparse_memory_mb(local_c);
    MPI_Allreduce(&local_mem_a, &metrics.memory_a_mb_total, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&local_mem_c, &metrics.memory_c_mb_total, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    metrics.memory_b_mb_per_rank = sparse_memory_mb(b_csr);

    const double flops =
        b_csr.rows > 0
            ? 2.0 * static_cast<double>(global_nnz_a) * static_cast<double>(b_csr.nnz()) / static_cast<double>(b_csr.rows)
            : 0.0;
    metrics.gflops = metrics.mean_time > 0.0 ? flops / (metrics.mean_time * 1e9) : 0.0;

    if (validate) {
        const CSRMatrix reference = spmm_reference_sparse_b(local_a_csr, b_csr);
        const int local_ok = csr_equal(local_c, reference) ? 1 : 0;
        MPI_Allreduce(&local_ok, &metrics.validation_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    }

    return metrics;
}

void write_csv_row(
    std::ofstream& out,
    const std::string& matrix_name,
    int m,
    int n,
    int nnz_a,
    int k,
    int nnz_b,
    double sparsity_b,
    int num_procs,
    const std::string& algorithm,
    const AlgorithmMetrics& metrics
) {
    const double density_a = (m > 0 && n > 0) ? static_cast<double>(nnz_a) / static_cast<double>(m) / static_cast<double>(n) : 0.0;
    out << csv_escape(matrix_name) << ','
        << m << ','
        << n << ','
        << nnz_a << ','
        << density_a << ','
        << k << ','
        << nnz_b << ','
        << sparsity_b << ','
        << num_procs << ','
        << csv_escape(algorithm) << ','
        << metrics.nnz_c << ','
        << metrics.mean_time << ','
        << metrics.std_time << ','
        << metrics.min_time << ','
        << metrics.gflops << ','
        << metrics.memory_a_mb_total << ','
        << metrics.memory_b_mb_per_rank << ','
        << metrics.memory_c_mb_total << ',';
    if (metrics.validation_ok < 0) {
        out << "";
    } else {
        out << (metrics.validation_ok == 1 ? "True" : "False");
    }
    out << '\n';
}

void ensure_csv_header(std::ofstream& out, const fs::path& output_path) {
    const bool needs_header = !fs::exists(output_path) || fs::file_size(output_path) == 0;
    if (needs_header) {
        out << "matrix_name,m,n,nnz_a,density_a,k,nnz_b,sparsity_b,num_procs,algorithm,nnz_c,"
               "mean_time_sec,std_time_sec,min_time_sec,gflops,memory_a_mb_total,"
               "memory_b_mb_per_rank,memory_c_mb_total,validation_ok\n";
    }
}

ParsedArgs parse_args(int argc, char** argv) {
    ParsedArgs args;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if ((arg == "--output" || arg == "-o") && i + 1 < argc) {
            args.output = argv[++i];
        } else if (arg == "--outdir" && i + 1 < argc) {
            args.outdir = argv[++i];
        } else if (arg == "--cache-dir" && i + 1 < argc) {
            args.cache_dir = argv[++i];
        } else if ((arg == "--repeats" || arg == "-r") && i + 1 < argc) {
            args.repeats = std::stoi(argv[++i]);
        } else if (arg == "--validate") {
            args.validate = true;
        } else if (arg == "--b-cols") {
            args.b_cols.clear();
            while (i + 1 < argc && argv[i + 1][0] != '-') {
                args.b_cols.push_back(std::stoi(argv[++i]));
            }
        } else if (arg == "--sparsity") {
            args.sparsity.clear();
            while (i + 1 < argc && argv[i + 1][0] != '-') {
                args.sparsity.push_back(std::stod(argv[++i]));
            }
        } else if (arg == "--matrices") {
            args.matrices.clear();
            while (i + 1 < argc && argv[i + 1][0] != '-') {
                args.matrices.push_back(argv[++i]);
            }
        } else if (arg == "--help" || arg == "-h") {
            int rank = 0;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            if (rank == 0) {
                std::cout
                    << "Sparse B SpMM MPI benchmark (native C++)\n\n"
                    << "Sample usage:\n"
                    << "  ./spmm_sparse_mpi --outdir results_hpc_spmm_mpi --b-cols 4 8 16 --sparsity 0.10\n";
            }
            MPI_Finalize();
            std::exit(0);
        }
    }

    return args;
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int world = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);

    const ParsedArgs args = parse_args(argc, argv);

    if (rank == 0) {
        std::cout << "==========================================================================================\n";
        std::cout << "Sparse B SpMM MPI Benchmark (Native C++)\n";
        std::cout << "==========================================================================================\n";
        std::cout << "Matrices: ";
        for (std::size_t i = 0; i < args.matrices.size(); ++i) {
            std::cout << args.matrices[i] << (i + 1 < args.matrices.size() ? ", " : "\n");
        }
        std::cout << "B columns: ";
        for (std::size_t i = 0; i < args.b_cols.size(); ++i) {
            std::cout << args.b_cols[i] << (i + 1 < args.b_cols.size() ? ", " : "\n");
        }
        std::cout << "B sparsity: ";
        for (std::size_t i = 0; i < args.sparsity.size(); ++i) {
            std::cout << args.sparsity[i] << (i + 1 < args.sparsity.size() ? ", " : "\n");
        }
        std::cout << "MPI processes: " << world << "\n";
        std::cout << "Repeats: " << args.repeats << "\n";
        std::cout << "Output directory: " << args.outdir << "\n";
        std::cout << "Cache directory: " << args.cache_dir << "\n";
        std::cout << "==========================================================================================\n";
    }

    if (rank == 0) {
        fs::create_directories(args.outdir);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    const fs::path output_path = fs::path(args.outdir) / args.output;
    std::ofstream output;
    if (rank == 0) {
        output.open(output_path, std::ios::app);
        ensure_csv_header(output, output_path);
        output << std::setprecision(12);
    }

    for (const std::string& matrix_name : args.matrices) {
        MatrixLoadResult load_result;
        if (rank == 0) {
            load_result = load_matrix_market(matrix_name, args.cache_dir);
            if (load_result.ok) {
                std::cout << "Loaded " << matrix_name << " from " << load_result.source << "\n";
            } else {
                std::cout << "Skipping " << matrix_name << " (matrix not found)\n";
            }
        }

        int load_ok = rank == 0 && load_result.ok ? 1 : 0;
        MPI_Bcast(&load_ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (!load_ok) {
            continue;
        }

        int matrix_meta[3];
        if (rank == 0) {
            matrix_meta[0] = load_result.matrix.rows;
            matrix_meta[1] = load_result.matrix.cols;
            matrix_meta[2] = load_result.matrix.nnz();
        }
        MPI_Bcast(matrix_meta, 3, MPI_INT, 0, MPI_COMM_WORLD);

        const int global_m = matrix_meta[0];
        const int global_n = matrix_meta[1];
        const int global_nnz_a = matrix_meta[2];

        CSRMatrix local_a = distribute_matrix_rows(
            rank == 0 ? std::optional<CSRMatrix>(load_result.matrix) : std::nullopt
        );
        CSCMatrix local_a_csc = csr_to_csc(local_a);

        if (rank == 0) {
            std::cout << "\nBenchmarking " << matrix_name
                      << " (" << global_m << "x" << global_n
                      << ", nnz=" << global_nnz_a << ")\n";
            std::cout << "  Density: "
                      << (global_m > 0 && global_n > 0
                              ? static_cast<double>(global_nnz_a) / static_cast<double>(global_m) / static_cast<double>(global_n)
                              : 0.0)
                      << "\n";
        }

        for (int k : args.b_cols) {
            for (double sparsity : args.sparsity) {
                CSRMatrix b_csr;
                if (rank == 0) {
                    b_csr = generate_sparse_b(global_n, k, sparsity, 42);
                    std::cout << "\n  B: " << global_n << "x" << k
                              << ", sparsity=" << sparsity
                              << " (nnz=" << b_csr.nnz() << ")\n";
                }
                broadcast_csr_matrix(b_csr, 0);

                for (const std::string& algorithm : ALGORITHMS) {
                    if (rank == 0) {
                        std::cout << "    Benchmarking " << algorithm << "... " << std::flush;
                    }
                    const AlgorithmMetrics metrics = benchmark_algorithm_mpi(
                        algorithm,
                        local_a,
                        local_a_csc,
                        b_csr,
                        global_nnz_a,
                        args.repeats,
                        args.validate
                    );
                    if (rank == 0) {
                        std::cout << "done (" << metrics.gflops << " GFlop/s)\n";
                        write_csv_row(
                            output,
                            matrix_name,
                            global_m,
                            global_n,
                            global_nnz_a,
                            k,
                            b_csr.nnz(),
                            sparsity,
                            world,
                            algorithm,
                            metrics
                        );
                        output.flush();
                    }
                }
            }
        }
    }

    if (rank == 0) {
        std::cout << "\nResults written to " << output_path << "\n";
    }

    MPI_Finalize();
    return 0;
}
