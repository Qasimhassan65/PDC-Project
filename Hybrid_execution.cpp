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

// Toggle output
#define ENABLE_OUTPUT 1

// Precomputed factorials up to n=10
const vector<long long> precomputed_factorials = {1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800};

// Exchange element x with its successor in vector v
void exchange_successor(vector<int>& v, int x) {
    auto pos = find(v.begin(), v.end(), x) - v.begin();
    if (pos < v.size() - 1) {
        swap(v[pos], v[pos + 1]);
    }
}

// Identify the rightmost element out of order
int find_last_disordered(const vector<int>& v) {
    int len = v.size();
    int i = len - 1;
    while (i >= 0) {
        if (v[i] != i + 1) {
            return i + 1;
        }
        --i;
    }
    return 0;
}

// Compute predecessor for vector v when last element is n and spanning_tree ≠ n-1
vector<int> compute_predecessor(const vector<int>& v, int spanning_tree, int n) {
    vector<int> sorted(n);
    for (int i = 0; i < n; ++i) sorted[i] = i + 1;
    vector<int> result = v;
    if (spanning_tree == 2) {
        exchange_successor(result, spanning_tree);
        if (result == sorted) {
            result = v;
            exchange_successor(result, spanning_tree - 1);
            return result;
        }
    } else if (v[n - 2] == spanning_tree || v[n - 2] == n - 1) {
        int j = find_last_disordered(v);
        if (j > 0 && j <= n) {
            exchange_successor(result, v[j - 1]);
            return result;
        }
    }
    exchange_successor(result, spanning_tree);
    return result;
}

// Determine parent of vector v in the spanning_tree-th spanning tree
vector<int> locate_parent(const vector<int>& v, int spanning_tree, int n) {
    vector<int> sorted(n);
    for (int i = 0; i < n; ++i) sorted[i] = i + 1;
    vector<int> result = v;
    if (v[n - 1] == n) {
        if (spanning_tree != n - 1) {
            return compute_predecessor(v, spanning_tree, n);
        } else {
            exchange_successor(result, v[n - 2]);
            return result;
        }
    } else if (v[n - 1] == n - 1 && v[n - 2] == n) {
        exchange_successor(result, n);
        if (result != sorted) {
            if (spanning_tree == 1) {
                return result;
            } else {
                result = v;
                exchange_successor(result, spanning_tree - 1);
                return result;
            }
        }
    }
    if (v[n - 1] == spanning_tree) {
        exchange_successor(result, n);
    } else {
        exchange_successor(result, spanning_tree);
    }
    return result;
}

// Generate the k-th permutation in lexicographic order
vector<int> generate_permutation(int n, long long k) {
    vector<int> perm(n);
    vector<int> numbers(n);
    for (int i = 0; i < n; ++i) numbers[i] = i + 1;
    k = k % precomputed_factorials[n];
    int i = 0;
    while (i < n) {
        long long fact = precomputed_factorials[n - 1 - i];
        int idx = k / fact;
        k = k % fact;
        perm[i] = numbers[idx];
        numbers.erase(numbers.begin() + idx);
        ++i;
    }
    return perm;
}

int main(int argc, char** argv) {
    // MPI setup
    MPI_Init(&argc, &argv);
    int rank, total_ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &total_ranks);

    // Configure OpenMP threads
    int num_threads = (argc > 1) ? atoi(argv[1]) : 3;
    if (num_threads < 1) {
        if (rank == 0) {
            cerr << "Error: Number of threads must be positive" << endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    omp_set_num_threads(num_threads);

    // Timing metrics
    double start_total, stop_total, start_compute, stop_compute, start_io, stop_io;
    start_total = MPI_Wtime();

    int size = 10; // Network dimension
    long long node_count = precomputed_factorials[size];
    long long output_limit = 120; // Restrict output to first 5! permutations

    // Work distribution
    long long block_size = node_count / total_ranks;
    long long begin_idx = rank * block_size;
    long long end_idx = (rank == total_ranks - 1) ? node_count : (rank + 1) * block_size;

    // Output distribution
    long long output_block_size = output_limit / total_ranks;
    long long output_begin = rank * output_block_size;
    long long output_end = (rank == total_ranks - 1) ? output_limit : (rank + 1) * output_block_size;

    // Initialize output file
    ofstream output_file("Parallel_output_rank_" + to_string(rank) + ".txt");
    if (!output_file.is_open()) {
        cerr << "Rank " << rank << ": Unable to open output file" << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Initialize timing file
    ofstream timing_file("Parallel_timing_rank_" + to_string(rank) + ".txt");
    if (!timing_file.is_open()) {
        cerr << "Rank " << rank << ": Unable to open timing file" << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Output header with aesthetic design
    if (ENABLE_OUTPUT) {
        output_file << "╔══════════════════════════════════════════════════════════════╗\n";
        output_file << "║                Permutation Ancestor Analysis                 ║\n";
        output_file << "╚══════════════════════════════════════════════════════════════╝\n";
        output_file << "  Rank  |      Permutation      |  Tree  |      Parent       \n";
        output_file << "════════╪═══════════════════════╪════════╪═══════════════════\n";
    }

    // Computation timing start
    start_compute = MPI_Wtime();

    // Process assigned permutations
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        long long i = begin_idx;
        #pragma omp single
        while (i < end_idx) {
            vector<int> perm = generate_permutation(size, i);
            int t = 1;
            while (t <= size - 1) {
                vector<int> parent = locate_parent(perm, t, size);
                ++t;
            }
            ++i;
        }
    }

    // Computation timing end
    stop_compute = MPI_Wtime();

    // Output timing start
    start_io = MPI_Wtime();

    // Generate output for assigned permutations
    if (ENABLE_OUTPUT) {
        vector<stringstream> buffers(num_threads);
        #pragma omp parallel
        {
            int thread_id = omp_get_thread_num();
            long long i = output_begin;
            while (i < output_end) {
                vector<int> perm = generate_permutation(size, i);
                int t = 1;
                while (t <= size - 1) {
                    vector<int> parent = locate_parent(perm, t, size);
                    string perm_str, parent_str;
                    for (int val : perm) perm_str += to_string(val) + " ";
                    for (int val : parent) parent_str += to_string(val) + " ";
                    buffers[thread_id] << "  " << setw(4) << rank << "  │  "
                                       << setw(19) << left << perm_str << "│  "
                                       << setw(5) << t << "  │  "
                                       << setw(17) << parent_str << "\n";
                    ++t;
                }
                ++i;
            }
        }
        // Write buffers to file
        int tid = 0;
        while (tid < num_threads) {
            output_file << buffers[tid].str();
            ++tid;
        }
        // Add footer
        output_file << "════════╧═══════════════════════╧════════╧═══════════════════\n";
    }

    // Output timing end
    stop_io = MPI_Wtime();

    output_file.close();

    // Total timing end
    stop_total = MPI_Wtime();

    // Record timing metrics in a tabular format
    timing_file << fixed << setprecision(6);
    timing_file << "╔════════════════════════════════════════════════════╗\n";
    timing_file << "║               Performance Metrics                  ║\n";
    timing_file << "╚════════════════════════════════════════════════════╝\n";
    timing_file << "┌──────────────────────────────┬─────────────────────┐\n";
    timing_file << "│ Metric                       │ Value               │\n";
    timing_file << "├──────────────────────────────┼─────────────────────┤\n";
    timing_file << "│ Total Duration               │ " << setw(12) << (stop_total - start_total) << " seconds │\n";
    timing_file << "│ Computation Duration         │ " << setw(12) << (stop_compute - start_compute) << " seconds │\n";
    timing_file << "│ I/O Duration                 │ " << setw(12) << (stop_io - start_io) << " seconds │\n";
    timing_file << "│ MPI Ranks                    │ " << setw(12) << total_ranks << "         │\n";
    timing_file << "│ OpenMP Threads per Rank      │ " << setw(12) << num_threads << "         │\n";
    timing_file << "│ Total Threads                │ " << setw(12) << (total_ranks * num_threads) << "         │\n";
    timing_file << "└──────────────────────────────┴─────────────────────┘\n";
    timing_file.close();

    MPI_Finalize();
    return 0;
}
