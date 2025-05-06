#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <omp.h>
#include <mpi.h>
#include <fstream>
#include <string>
#include <iomanip>
#include <sstream>

using namespace std;

// Toggle output generation
#define ENABLE_OUTPUT 1

// Precomputed factorials for n=0 to n=10
const vector<long long> factorial_list = {1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800};

// Shift value x to swap with its next element in vector v
void shift_adjacent(vector<int>& v, int x) {
    auto iter = find(v.begin(), v.end(), x);
    size_t pos = distance(v.begin(), iter);
    if (pos < v.size() - 1) {
        swap(v[pos], v[pos + 1]);
    }
}

// Identify the rightmost position where the value is not in sorted order
int locate_last_mismatch(const vector<int>& v) {
    int size = v.size();
    int idx = size - 1;
    while (idx >= 0) {
        if (v[idx] != idx + 1) {
            return idx + 1;
        }
        --idx;
    }
    return 0;
}

// Compute predecessor when last element is n and spanning_tree != n-1
vector<int> calculate_previous(const vector<int>& v, int spanning_tree, int size) {
    vector<int> sorted(size);
    for (int i = 0; i < size; ++i) sorted[i] = i + 1;
    vector<int> output = v;
    if (spanning_tree == 2) {
        shift_adjacent(output, spanning_tree);
        if (output == sorted) {
            output = v;
            shift_adjacent(output, spanning_tree - 1);
            return output;
        }
    } else if (v[size - 2] == spanning_tree || v[size - 2] == size - 1) {
        int mismatch = locate_last_mismatch(v);
        if (mismatch > 0 && mismatch <= size) {
            shift_adjacent(output, v[mismatch - 1]);
            return output;
        }
    }
    shift_adjacent(output, spanning_tree);
    return output;
}

// Determine the parent of vector v in the spanning_tree-th spanning tree
vector<int> determine_parent(const vector<int>& v, int spanning_tree, int size) {
    vector<int> sorted(size);
    for (int i = 0; i < size; ++i) sorted[i] = i + 1;
    vector<int> output = v;
    if (v[size - 1] == size) {
        if (spanning_tree != size - 1) {
            return calculate_previous(v, spanning_tree, size);
        } else {
            shift_adjacent(output, v[size - 2]);
            return output;
        }
    } else if (v[size - 1] == size - 1 && v[size - 2] == size) {
        shift_adjacent(output, size);
        if (output != sorted) {
            if (spanning_tree == 1) {
                return output;
            } else {
                output = v;
                shift_adjacent(output, spanning_tree - 1);
                return output;
            }
        }
    }
    if (v[size - 1] == spanning_tree) {
        shift_adjacent(output, size);
    } else {
        shift_adjacent(output, spanning_tree);
    }
    return output;
}

// Generate the k-th permutation in lexicographic order
vector<int> build_permutation(int size, long long k) {
    vector<int> perm(size);
    vector<int> numbers(size);
    for (int i = 0; i < size; ++i) numbers[i] = i + 1;
    k = k % factorial_list[size];
    int pos = 0;
    while (pos < size) {
        long long fact = factorial_list[size - 1 - pos];
        int idx = k / fact;
        k = k % fact;
        perm[pos] = numbers[idx];
        numbers.erase(numbers.begin() + idx);
        ++pos;
    }
    return perm;
}

int main(int argc, char** argv) {
    // Initialize MPI environment
    MPI_Init(&argc, &argv);
    int rank_id, total_ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_id);
    MPI_Comm_size(MPI_COMM_WORLD, &total_ranks);

    // Set OpenMP to single-threaded for MPI-only execution
    const int num_threads = 1;
    omp_set_num_threads(num_threads);

    // Timing variables
    double start_total, stop_total, start_compute, stop_compute, start_io, stop_io;
    start_total = MPI_Wtime();

    const int network_dim = 10; // Size of bubble-sort network
    const long long total_perms = factorial_list[network_dim];
    const long long output_max = 120; // Limit output to first 5! permutations

    // Distribute permutation workload across ranks
    long long block_size = total_perms / total_ranks;
    long long begin_idx = rank_id * block_size;
    long long end_idx = (rank_id == total_ranks - 1) ? total_perms : (rank_id + 1) * block_size;

    // Distribute output workload
    long long output_block = output_max / total_ranks;
    long long output_begin = rank_id * output_block;
    long long output_end = (rank_id == total_ranks - 1) ? output_max : (rank_id + 1) * output_block;

    // Initialize output file
    ofstream data_file("MPI_output_rank_" + to_string(rank_id) + ".txt");
    if (!data_file.is_open()) {
        cerr << "Rank " << rank_id << ": Cannot open output file" << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Initialize timing file
    ofstream metrics_file("MPI_timing_rank_" + to_string(rank_id) + ".txt");
    if (!metrics_file.is_open()) {
        cerr << "Rank " << rank_id << ": Cannot open timing file" << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Write aesthetically pleasing output header
    if (ENABLE_OUTPUT) {
        data_file << "╔══════════════════════════════════════════════════════════════╗\n";
        data_file << "║                Permutation Ancestor Analysis                 ║\n";
        data_file << "╚══════════════════════════════════════════════════════════════╝\n";
        data_file << "  Rank  |      Permutation      |  Tree  |      Parent       \n";
        data_file << "════════╪═══════════════════════╪════════╪═══════════════════\n";
    }

    // Start computation timing
    start_compute = MPI_Wtime();

    // Process assigned permutations
    long long idx = begin_idx;
    while (idx < end_idx) {
        vector<int> perm = build_permutation(network_dim, idx);
        int tree_num = 1;
        while (tree_num <= network_dim - 1) {
            vector<int> parent = determine_parent(perm, tree_num, network_dim);
            ++tree_num;
        }
        ++idx;
    }

    // End computation timing
    stop_compute = MPI_Wtime();

    // Start output timing
    start_io = MPI_Wtime();

    // Generate output for assigned permutations
    if (ENABLE_OUTPUT) {
        vector<stringstream> buffers(num_threads);
        idx = output_begin;
        while (idx < output_end) {
            vector<int> perm = build_permutation(network_dim, idx);
            int tree_num = 1;
            while (tree_num <= network_dim - 1) {
                vector<int> parent = determine_parent(perm, tree_num, network_dim);
                string perm_str, parent_str;
                for (int val : perm) perm_str += to_string(val) + " ";
                for (int val : parent) parent_str += to_string(val) + " ";
                buffers[0] << "  " << setw(4) << rank_id << "  │  "
                           << setw(19) << left << perm_str << "│  "
                           << setw(5) << tree_num << "  │  "
                           << setw(17) << parent_str << "\n";
                ++tree_num;
            }
            ++idx;
        }
        // Write buffer to file
        data_file << buffers[0].str();
        // Add footer
        data_file << "════════╧═══════════════════════╧════════╧═══════════════════\n";
    }

    // End output timing
    stop_io = MPI_Wtime();

    data_file.close();

    // End total timing
    stop_total = MPI_Wtime();

    // Write timing metrics in a tabular format
    metrics_file << fixed << setprecision(6);
    metrics_file << "╔════════════════════════════════════════════════════╗\n";
    metrics_file << "║               Performance Metrics                  ║\n";
    metrics_file << "╚════════════════════════════════════════════════════╝\n";
    metrics_file << "┌──────────────────────────────┬─────────────────────┐\n";
    metrics_file << "│ Metric                       │ Value               │\n";
    metrics_file << "├──────────────────────────────┼─────────────────────┤\n";
    metrics_file << "│ Total Duration               │ " << setw(12) << (stop_total - start_total) << " seconds │\n";
    metrics_file << "│ Computation Duration         │ " << setw(12) << (stop_compute - start_compute) << " seconds │\n";
    metrics_file << "│ I/O Duration                 │ " << setw(12) << (stop_io - start_io) << " seconds │\n";
    metrics_file << "│ MPI Ranks                    │ " << setw(12) << total_ranks << "         │\n";
    metrics_file << "│ OpenMP Threads per Rank      │ " << setw(12) << num_threads << "         │\n";
    metrics_file << "│ Total Threads                │ " << setw(12) << (total_ranks * num_threads) << "         │\n";
    metrics_file << "└──────────────────────────────┴─────────────────────┘\n";
    metrics_file.close();

    MPI_Finalize();
    return 0;
}
