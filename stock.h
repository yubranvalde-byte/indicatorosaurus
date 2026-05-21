#ifndef STOCK_H
#define STOCK_H

#define MAX_TICKER_LEN  8
#define NUM_INDICATORS 16

// Generic fixed-size indicator array layout.
// indicators[0..15] hold pre-normalized (0-100) float values populated by
// the Python FFI layer. The C engine only performs a pure dot product.
//
// Index → Column mapping (matches financial_module.py INDICATOR_COLUMNS order):
//   [0]  ForwardPE        [1]  EV_EBITDA       [2]  FCFYield
//   [3]  RevenueGrowth    [4]  EPSGrowth        [5]  GrossMargin
//   [6]  OperatingMargin  [7]  ROIC             [8]  DebtToEquity
//   [9]  ShareholderYield [10] FiveYDivGrowth   [11] PayoutRatio
//   [12] Beta             [13] RSI              [14] PriceVs200SMA
//   [15] RelativeStrength
typedef struct {
    char   ticker[MAX_TICKER_LEN];     // e.g. "AAPL\0", fits up to 7-char tickers
    float  indicators[NUM_INDICATORS]; // pre-normalized 0-100 scores from Python
    double final_score;                // weighted dot-product result (0-100 scale)
} Stock;

#endif // STOCK_H
