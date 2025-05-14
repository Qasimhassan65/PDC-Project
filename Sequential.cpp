#include <iostream>
#include <vector>
#include <algorithm>
#include <fstream>
#include <string>
#include <iomanip>
#include <chrono>

using namespace std;

#define GENERATE_OUTPUT 1

const vector<long long> factorial_values = {1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800};



void swap_with_next(vector<int>& v, int x) {
    auto iter = find(v.begin(), v.end(), x);
    size_t idx = distance(v.begin(), iter);
    if (idx < v.size() - 1) {
        swap(v[idx], v[idx + 1]);
    }
}

int get_last_out_of_order(const vector<int>& v) {
    int n = v.size();
    int i = n - 1;
    while (i >= 0) {
        if (v[i] != i + 1) {
            return i + 1;
        }
        --i;
    }
    return 0;
}



vector<int> find_previous(const vector<int>& v, int tree_num, int n) {
    vector<int> ordered(n);
    for (int i = 0; i < n; ++i) ordered[i] = i + 1;
    vector<int> result = v;
    if (tree_num == 2) {
        swap_with_next(result, tree_num);
        if (result == ordered) {
            result = v;
            swap_with_next(result, tree_num - 1);
            return result;
        }
    } else if (v[n - 2] == tree_num || v[n - 2] == n - 1) {
        int j = get_last_out_of_order(v);
        if (j > 0 && j <= n) {
            swap_with_next(result, v[j - 1]);
            return result;
        }
    }
    swap_with_next(result, tree_num);
    return result;
}



vector<int> get_parent_node(const vector<int>& v, int tree_num, int n) {
    vector<int> ordered(n);
    for (int i = 0; i < n; ++i) ordered[i] = i + 1;
    vector<int> result = v;
    if (v[n - 1] == n) {
        if (tree_num != n - 1) {
            return find_previous(v, tree_num, n);
        } else {
            swap_with_next(result, v[n - 2]);
            return result;
        }
    } else if (v[n - 1] == n - 1 && v[n - 2] == n) {
        swap_with_next(result, n);
        if (result != ordered) {
            if (tree_num == 1) {
                return result;
            } else {
                result = v;
                swap_with_next(result, tree_num - 1);
                return result;
            }
        }
    }
    if (v[n - 1] == tree_num) {
        swap_with_next(result, n);
    } else {
        swap_with_next(result, tree_num);
    }
    return result;
}



vector<int> produce_permutation(int n, long long k) {
    vector<int> result(n);
    vector<int> available(n);
    for (int i = 0; i < n; ++i) available[i] = i + 1;
    k = k % factorial_values[n];
    int pos = 0;
    while (pos < n) {
        long long fact = factorial_values[n - 1 - pos];
        int idx = k / fact;
        k = k % fact;
        result[pos] = available[idx];
        available.erase(available.begin() + idx);
        ++pos;
    }
    return result;
}

int main() {

    auto start_overall = chrono::high_resolution_clock::now();
    auto start_processing = chrono::high_resolution_clock::now();

    const int network_size = 10;
    const long long total_perms = factorial_values[network_size];
    const long long output_limit = 120;



    ofstream data_output("Sequential_results.txt");
    if (!data_output.is_open()) {
        cerr << "Error: Failed to create output file" << endl;
        return 1;
    }


    ofstream metrics_output("Sequential_metrics.txt");
    if (!metrics_output.is_open()) {
        cerr << "Error: Failed to create timing file" << endl;
        return 1;
    }


    if (GENERATE_OUTPUT) {
        data_output << "╔══════════════════════════════════════════════════════════════╗\n";
        data_output << "║                Permutation Ancestor Analysis                 ║\n";
        data_output << "╚══════════════════════════════════════════════════════════════╝\n";
        data_output << "  Permutations         |  Tree  |  Parent              \n";
        data_output << "═══════════════════════╪════════╪══════════════════════\n";
    }



    long long current_idx = 0;
    while (current_idx < total_perms) {
        vector<int> perm = produce_permutation(network_size, current_idx);
        int tree_idx = 1;
        while (tree_idx <= network_size - 1) {
            vector<int> parent = get_parent_node(perm, tree_idx, network_size);
            if (GENERATE_OUTPUT && current_idx < output_limit) {
                string perm_str, parent_str;
                for (int val : perm) perm_str += to_string(val) + " ";
                for (int val : parent) parent_str += to_string(val) + " ";
                data_output << "  " << setw(19) << left << perm_str << "│  "
                            << setw(5) << tree_idx << "  │  "
                            << setw(18) << parent_str << "\n";
            }
            ++tree_idx;
        }
        ++current_idx;
    }

    auto end_processing = chrono::high_resolution_clock::now();
    auto start_writing = chrono::high_resolution_clock::now();

    data_output << "═══════════════════════╧════════╧══════════════════════\n";
    data_output.close();

    auto end_writing = chrono::high_resolution_clock::now();
    auto end_overall = chrono::high_resolution_clock::now();
    
    
    
    
    metrics_output << fixed << setprecision(6);
    metrics_output << "╔════════════════════════════════════════════════════╗\n";
    metrics_output << "║               Performance Metrics                  ║\n";
    metrics_output << "╚════════════════════════════════════════════════════╝\n";
    metrics_output << "┌──────────────────────────────┬─────────────────────┐\n";
    metrics_output << "│ Perf. Metric                 │ Perf. Value         │\n";
    metrics_output << "├──────────────────────────────┼─────────────────────┤\n";
    metrics_output << "│ Total Duration               │ " << setw(12) << chrono::duration<double>(end_overall - start_overall).count() << " seconds │\n";
    metrics_output << "│ Computation Duration         │ " << setw(12) << chrono::duration<double>(end_processing - start_processing).count() << " seconds │\n";
    metrics_output << "│ I/O Duration                 │ " << setw(12) << chrono::duration<double>(end_writing - start_writing).count() << " seconds │\n";
    metrics_output << "└──────────────────────────────┴─────────────────────┘\n";
    metrics_output.close();

    return 0;
}
