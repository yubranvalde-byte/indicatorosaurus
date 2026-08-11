#ifndef FINANCIAL_ENGINE_H
#define FINANCIAL_ENGINE_H

#include <stddef.h>
#include <stdlib.h>
#include "stock.h"

// Computes the final_score for each stock using a pure weighted dot product.
// Inputs in stock.indicators[] must already be Min-Max normalized (0-100)
// by the Python FFI layer before this function is called.
void compute_scores(Stock* stocks, int num_stocks, const double* weights);

// Sorts the array of stocks in descending order based on their final_score using Quick Sort
void sort_stocks_by_score(Stock* stocks, int num_stocks);

#endif // FINANCIAL_ENGINE_H
