#include "financial_engine.h"

// Calculate the final score for each stock using a pure weighted dot product.
// Inputs in stock.indicators[] are already Min-Max normalized (0-100 scale)
// by the Python FFI layer before this function is called.
// No Min-Max scan or direction inversion is performed here.
void compute_scores(Stock* stocks, int num_stocks, const double* weights) {
    if (num_stocks <= 0) return;

    // Sum all 16 weights for normalization
    double total_weight = 0.0;
    for (int i = 0; i < NUM_INDICATORS; i++) {
        total_weight += weights[i];
    }

    // Edge case: if total weight is 0, assign neutral score to avoid division by zero
    if (total_weight == 0.0) {
        for (int i = 0; i < num_stocks; i++) {
            stocks[i].final_score = 0.0;
        }
        return;
    }

    // Pure dot product: sum of (normalized_indicator[j] * weight[j]) for j in 0..15
    // Cast float -> double for each indicator before accumulation to preserve precision.
    for (int i = 0; i < num_stocks; i++) {
        double weighted_sum = 0.0;
        for (int j = 0; j < NUM_INDICATORS; j++) {
            weighted_sum += (double)stocks[i].indicators[j] * weights[j];
        }
        stocks[i].final_score = (weighted_sum / total_weight);
    }
}

// Internal swap utility
static void swap_stocks(Stock* a, Stock* b) {
    Stock temp = *a;
    *a = *b;
    *b = temp;
}

// Internal Quick Sort partition logic (descending order by final_score)
static int partition_stocks(Stock* arr, int low, int high) {
    double pivot = arr[high].final_score;
    int i = (low - 1);

    for (int j = low; j <= high - 1; j++) {
        // Sort descending: if current final_score is greater than or equal to pivot
        if (arr[j].final_score >= pivot) {
            i++;
            swap_stocks(&arr[i], &arr[j]);
        }
    }
    swap_stocks(&arr[i + 1], &arr[high]);
    return (i + 1);
}

// Internal recursive Quick Sort logic
static void quick_sort_stocks(Stock* arr, int low, int high) {
    if (low < high) {
        // Fast-path: Check if all elements in the current subarray have identical final scores
        int all_equal = 1;
        for (int k = low + 1; k <= high; k++) {
            if (arr[k].final_score != arr[low].final_score) {
                all_equal = 0;
                break;
            }
        }
        if (all_equal) {
            return;
        }

        int pi = partition_stocks(arr, low, high);
        quick_sort_stocks(arr, low, pi - 1);
        quick_sort_stocks(arr, pi + 1, high);
    }
}

// Public API to sort the stocks in descending order by final_score
void sort_stocks_by_score(Stock* stocks, int num_stocks) {
    if (num_stocks > 1) {
        quick_sort_stocks(stocks, 0, num_stocks - 1);
    }
}
