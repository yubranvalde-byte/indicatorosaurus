# AlphaQuant V3.0 — Hybrid Quantitative Screening & Dynamic Rebalancing Platform

[![Python 3.x](https://img.shields.io/badge/Python-3.x-blue)](https://www.python.org/)
[![C](https://img.shields.io/badge/C-GCC%20FFI-lightgrey)](https://gcc.gnu.org/)
[![Flask](https://img.shields.io/badge/Flask-2.x-green)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-Academic-orange)]()

---

## Executive Summary

**AlphaQuant V3.0** is a high-performance, institutional-grade quantitative equity research and backtesting server. It is architected as a **hybrid Python/C application** that combines a real-time cross-sectional scoring engine with a historical rolling-window, point-in-time portfolio simulator.

The platform ingests live and quarterly-historical fundamental data across the full S&P 500 and Nasdaq universes (500+ tickers), normalizes the data using sector-neutral, outlier-resistant sigmoid Z-score transformations, and scores every asset via a compiled C-FFI scoring kernel. Users interact with a dynamic institutional-grade web dashboard to rank equities under customizable factor weight profiles (Momentum, Hyper-Growth, Quality, Deep Value, Income), apply GICS sector and market-cap filters, and trigger a trailing backtest simulation that dynamically rebalances the portfolio at strategy-appropriate frequencies, benchmarked against the SPY ETF.

---

## Core Architectural Pillars

### 1. The Performance Engine — C Layer

**Files:** `financial_engine.c`, `financial_engine.h`, `stock.h`, `Makefile`

The computational hot-path is implemented in C and compiled into a **shared object dynamic library** (`libsortbuslines.so`) that is loaded at runtime by Python via `ctypes`.

#### `Stock` Struct — Contiguous Memory Layout (`stock.h`)
```c
typedef struct {
    char   ticker[8];        //  8 bytes — max 7 meaningful chars + null terminator
    float  indicators[16];   // 64 bytes — normalized indicator values (0–100 scale)
    double final_score;      //  8 bytes
} Stock;
```
All 16 indicator values per asset are stored in a **contiguous fixed-length array** within the struct. This cache-line–friendly layout ensures that the vector dot-product loop iterates over sequential memory addresses, minimizing L1/L2 cache misses under large-universe batch scoring.

#### C FFI via `ctypes`
The Python orchestration layer (`financial_module.py`) uses `ctypes.CDLL` to load `libsortbuslines.so` and call the exported C functions with zero-copy pointer passing:
```python
lib = ctypes.CDLL('./libsortbuslines.so')
lib.score_and_sort_stocks(stock_array, n_stocks, weights_array)
```
The boundary between Python and C is strictly controlled: Python is responsible for I/O, normalization, and HTTP routing; C owns all compute-heavy sorting and scoring.

#### Vector Dot-Product Scoring
For each asset, the raw score is computed as the weighted inner product of its normalized indicator vector against the user-supplied weight vector:

$$\text{FinalScore}_i = \sum_{j=1}^{16} w_j \cdot \hat{x}_{i,j}$$

where $\hat{x}_{i,j}$ is the sector-neutral sigmoid Z-score of indicator $j$ for asset $i$, and $w_j \in [0, 10]$ is the user-configured weight slider value.

#### QuickSort with Duplicate-Value Early-Exit
After scoring, all `Stock` structs are sorted in descending order of `final_score` using an **in-place QuickSort** implemented in C. The partition routine includes an early-exit guard that detects uniform pivot values (all-equal sub-arrays) and terminates recursion immediately, protecting against $O(n^2)$ worst-case behaviour on ties.

---

### 2. The Orchestration Layer — Flask / Python Backend

**Files:** `main.py`, `financial_module.py`

#### Unified Data Ingestion Pipeline (`fetch_static_data.py`)
The ingestion pipeline fetches two datasets:
- **Real-time snapshot** (`market_data.csv`): Live fundamental metrics for all universe tickers via `yfinance`.
- **Historical panel** (`historical_fundamental_data.csv`): Quarterly-granularity fundamental snapshots across the trailing 7 quarters per ticker. This includes calculated indicators (EPS Growth, Free Cash Flow Yield, Shareholder Yield, etc.) computed at each quarter-end without look-ahead bias.

#### Cross-Sectional GICS-Neutral Sigmoid Z-Score Normalization
Raw fundamental indicators span incompatible magnitudes across sectors (e.g., a 3% FCF yield for a utility is excellent, while the same value for a tech hyperscaler is mediocre). To eliminate this sector bias, each indicator is normalized **within its GICS sector peer group** using a robust sigmoid Z-score:

$$Z_{i,j} = \frac{x_{i,j} - \tilde{x}_{j,\text{sector}}}{\text{MAD}_{j,\text{sector}} \cdot 1.4826}$$

$$\hat{x}_{i,j} = \text{sigmoid}(Z_{i,j}) = \frac{1}{1 + e^{-Z_{i,j}}}$$

where:
- $\tilde{x}_{j,\text{sector}}$ is the **sector-level median** of indicator $j$ (resistant to outliers)
- $\text{MAD}_{j,\text{sector}}$ is the **Median Absolute Deviation** (robust scale estimator)
- The $1.4826$ constant is the consistency factor making MAD a consistent estimator of the standard deviation under normality

The sigmoid squashing maps every score to $(0, 1)$, guaranteeing numerical stability regardless of how extreme an outlier is.

#### Dynamic Sector & Knock-Out Filter Registry
A strategy-specific knock-out filter registry applies hard binary gates **before** the scoring pass, eliminating assets that structurally violate the strategy mandate (e.g., EPS Growth below 0 for Momentum, or no dividend for Income). The registry is organized as a Python dictionary of lambdas keyed by strategy identifier:
```python
KNOCKOUT_FILTERS = {
    'momentum':    lambda s: s.get('EPSGrowthQoQ', 0) > 0.05 and ...,
    'income':      lambda s: s.get('NetShareholderYield', 0) > 0.02 and ...,
    ...
}
```
This two-phase architecture — hard filters first, soft scoring second — ensures that conceptually incompatible assets never reach the scoring kernel.

---

### 3. The Lean Simulation Layer — `backtest_engine.py`

#### Rolling Point-in-Time Rebalancing Loop
The backtest engine executes a discrete chronological time-loop over all historical quarter-end anchor dates $\{T_0, T_1, \ldots, T_n\}$ present in `historical_fundamental_data.csv`:

```
for each anchor T_k in selected_anchors:
    1. Slice the fundamental panel to the T_k snapshot (no future data)
    2. Apply knock-out filters and normalize the cross-section
    3. Score all assets via the C-engine and extract the Top 10
    4. Assign the Top 10 as the active portfolio for [T_k, T_{k+1})
    5. Compute daily portfolio returns using dynamic re-weighting
```

At each rebalance point, the engine uses only data available **at or before** $T_k$, eliminating look-ahead bias from the simulation.

#### Strategy-Based Dynamic Rebalancing Frequency Matrix

| Strategy     | Rebalancing Frequency | Step (quarters) |
|:------------|:---------------------:|:---------------:|
| `momentum`   | Every 3 months        | 1               |
| `hyper-growth` | Every 6 months      | 2               |
| `quality`    | Every 6 months        | 2               |
| `deep-value` | Every 12 months       | 4               |
| `income`     | Every 12 months       | 4               |
| `custom`     | Every 3 months        | 1               |

This alignment reflects factor half-lives: momentum signals decay rapidly and require frequent rebalancing, while deep-value and income signals are structural and rebalance annually.

#### Anti-Throttling Upfront Universe Price Download
Rather than dynamically querying `yfinance` per rebalance step (which causes rate-limit errors for high-turnover strategies with large ticker unions), the engine **pre-fetches the entire daily price history** for the full static universe in a **single bulk API call** at simulation start:
```python
price_data = yf.download(all_symbols, start=start_date, end=end_date)
```
All subsequent daily return lookups during the simulation loop are then pure in-memory Pandas index operations — zero additional API calls.

#### Terminal Window Boundary Patch ($T_n$ Fix)
The final holding interval has no $T_{n+1}$ anchor. Without correction, this silently drops the last live quarter of trading data from the performance chart. The engine explicitly forces the terminal boundary to the **maximum available date in the downloaded price index**:
```python
if k == len(selected_anchors) - 1:
    hold_end = trading_dates[-1]   # live trading day, not T_{n+1}
```

#### Dynamic Portfolio Re-Weighting for IPO Gaps
When a Top-10 asset has no price data for a given historical day (e.g., it IPO'd mid-period), its return is `NaN`. The engine uses `skipna=True` when averaging across the portfolio for that day, automatically redistributing its notional weight to the available assets — eliminating artificial timeline truncation.

---

### 4. The Analytical Frontend — `templates/index.html`

The single-page dashboard is built with **Tailwind CSS** and **Chart.js**, served by Flask's Jinja2 template engine.

#### Interactive Feature Highlights
- **16-Indicator Weight Sliders**: Live DOM-driven configuration of the scoring weight vector, streamed to the backend on every `Calculate Score` click.
- **Strategy Preset Profiles**: Momentum, Hyper-Growth, Quality, Deep Value, Income — each applying a pre-baked weight configuration and activating the corresponding knock-out filter.
- **GICS Sector Filter**: Multi-select checkbox grid allowing the user to restrict the universe to specific sectors before scoring.
- **Market Cap Range Gates**: Numeric input fields (Min / Max in billions) that apply absolute size filters to the pre-scoring universe.
- **Backtest Dashboard**: An institutional-grade card component with a time-window toggle (3Y / 5Y), triggering a `POST /api/backtest` call that returns a full performance timeseries and KPI block.

#### XSS-Safe Dynamic HTML Rendering
All ticker names, sector labels, and company names fetched from external data sources are sanitized via a custom `escapeHTML()` function before being injected into the DOM:
```javascript
function escapeHTML(str) {
  return String(str)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;')
    .replace(/>/g, '&gt;').replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}
```
This prevents injection of rogue HTML or JavaScript payloads from malicious ticker metadata.

#### Real-Time Chart.js Compounding Curve Rendering
The backtest result is rendered as a dual-line compounding growth chart (`Portfolio` vs `SPY`) using **Chart.js**, with smooth tension, gradient fill, and a zero-latency canvas teardown-rebuild cycle on each new simulation.

---

## Mathematical & Quantitative Models

### Dynamic Net Shareholder Yield

The Shareholder Yield metric captures the **total net cash flow returned to capital providers** (dividends + buybacks) while penalizing equity dilution and net debt accumulation:

$$\text{Net Stock Flow} = \text{Share Issuances} + \text{Share Repurchases}$$

$$\text{Net Debt Flow} = \text{Debt Issuances} + \text{Debt Repayments}$$

$$\text{Shareholder Yield} = \frac{-(\text{Dividends Paid} + \text{Net Stock Flow} + \text{Net Debt Flow})}{\text{Market Cap}}$$

The negative sign convention is applied because yfinance reports cash outflows from the company's perspective as negative; a company paying dividends and buying back stock has large negative cash flows to equity holders, which we invert to a positive yield signal.

### Sector-Neutral Sigmoid Z-Score (full derivation)

$$Z_{i,j}^{\text{sector}} = \frac{x_{i,j} - \text{median}(\{x_{k,j} : \text{sector}(k) = \text{sector}(i)\})}{\text{MAD}(\{x_{k,j}\}) \times 1.4826}$$

$$\hat{x}_{i,j} = \sigma(Z_{i,j}) = \frac{1}{1 + e^{-Z_{i,j}}} \in (0, 1)$$

### Composite Score

$$S_i = \sum_{j=1}^{16} w_j \cdot \hat{x}_{i,j}, \quad w_j \in [0, 10]$$

### Backtesting Performance Metrics

**Annualized Return:**
$$R_{\text{ann}} = \left(\prod_{t=1}^{T}(1 + r_t)\right)^{\frac{252}{T}} - 1$$

**Annualized Volatility:**
$$\sigma_{\text{ann}} = \sqrt{252} \cdot \text{std}(\{r_t\}_{t=1}^{T})$$

**Sharpe Ratio** (with 4% risk-free hurdle):
$$\text{Sharpe} = \frac{R_{\text{ann}} - r_f}{\sigma_{\text{ann}}}, \quad r_f = 0.04$$

**Maximum Drawdown:**
$$\text{MDD} = \min_{t \in [0,T]} \frac{\text{NAV}(t) - \max_{\tau \leq t} \text{NAV}(\tau)}{\max_{\tau \leq t} \text{NAV}(\tau)}$$

**Excess Return Alpha vs SPY:**
$$\alpha = R_{\text{ann}}^{\text{Portfolio}} - R_{\text{ann}}^{\text{SPY}}$$

---

## Installation & Build Instructions

### Prerequisites

| Dependency | Version | Purpose |
|:-----------|:--------|:--------|
| Python | ≥ 3.9 | Orchestration, Flask server |
| GCC | ≥ 11 | C shared library compilation |
| `make` | any | Build automation |
| `pandas` | ≥ 1.5 | Data manipulation |
| `yfinance` | ≥ 0.2 | Market data ingestion |
| `flask` | ≥ 2.0 | HTTP server |
| `numpy` | ≥ 1.23 | Numerical operations |

### Step 1 — Clone the Repository
```bash
git clone https://github.cs.huji.ac.il/cclab2-2025/ex2bonus-yuval.brande.git
cd ex2bonus-yuval.brande/ex2_sp
```

### Step 2 — Create and Activate a Python Virtual Environment (inside WSL/Linux)
```bash
python3 -m venv linux_env
source linux_env/bin/activate
pip install -r requirements.txt
```

### Step 3 — Compile the C Shared Library
This step compiles `financial_engine.c`, `sort_bus_lines.c`, and `memory_visualization.c` into the shared object `libsortbuslines.so`, which is loaded by the Python backend via `ctypes` at runtime.

```bash
make clean && make
```

Expected output:
```
gcc -c -fPIC -Wall -Wextra sort_bus_lines.c -o sort_bus_lines.o
gcc -c -fPIC -Wall -Wextra memory_visualization.c -o memory_visualization.o
gcc -c -fPIC -Wall -Wextra financial_engine.c -o financial_engine.o
gcc -shared -fPIC -Wall -Wextra -o libsortbuslines.so sort_bus_lines.o memory_visualization.o financial_engine.o
```

> ⚠️ **WSL Required on Windows**: The `Makefile` uses GCC. On Windows, all compilation and Python execution must be performed inside WSL (Windows Subsystem for Linux).

### Step 4 — Initialize the Data Cache (Real-Time Snapshot)
```bash
python fetch_static_data.py
```
This fetches live fundamental data for all universe tickers and writes `market_data.csv` (~500 tickers, ~100KB).

### Step 5 — Initialize the Historical Fundamental Panel (Optional, Required for Backtesting)
```bash
python fetch_static_data.py --historical
```
This fetches quarterly income statement, balance sheet, and cash flow data for each ticker across the trailing 7 quarters, computes all 16 indicators at each quarter-end (no look-ahead bias), and writes `historical_fundamental_data.csv` (~3,500 rows, ~500KB).

> ⚠️ This process takes approximately 30–90 minutes depending on API rate limits. Progress is printed per-ticker to stdout.

### Step 6 — Launch the Application
```bash
python main.py
```
The Flask development server will start on `http://127.0.0.1:5000`. Open this URL in your browser to access the AlphaQuant dashboard.

---

## REST API Reference

### `POST /api/calculate`
Triggers a full cross-sectional scoring pass on the live data snapshot.

**Request Body:**
```json
{
  "weights": [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
  "strategy": "momentum",
  "allowed_sectors": ["technology", "industrials"],
  "min_market_cap": 10000000000,
  "max_market_cap": null
}
```

**Response:** Ranked list of stocks with scores, sectors, and indicator breakdowns.

---

### `POST /api/backtest`
Triggers a rolling-window rebalancing simulation over the historical fundamental panel.

**Request Body:**
```json
{
  "tickers": ["AAPL", "MSFT"],
  "time_window": "3y",
  "weights": [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
  "strategy": "momentum",
  "allowed_sectors": null,
  "min_market_cap": null,
  "max_market_cap": null
}
```

**Response:**
```json
{
  "performance_series": [
    { "date": "2023-05-22", "portfolio_return": 1.0, "spy_return": 1.0 },
    ...
  ],
  "metrics": {
    "portfolio_sharpe": 1.24,
    "spy_sharpe": 1.21,
    "portfolio_max_drawdown": -0.24,
    "spy_max_drawdown": -0.19,
    "total_alpha": 0.07
  }
}
```

---

## Project File Manifest

```
ex2_sp/
│
├── main.py                          # Flask HTTP server & API routing layer
├── financial_module.py              # Python orchestration: normalization, filtering, C-FFI bridge
├── backtest_engine.py               # Rolling-window rebalancing simulation engine
├── fetch_static_data.py             # Data ingestion pipeline (live snapshot + historical panel)
│
├── financial_engine.c               # C scoring kernel: dot-product scoring, QuickSort
├── financial_engine.h               # C header: function prototypes for scoring & sorting
├── stock.h                          # C header: Stock struct definition (contiguous memory layout)
├── sort_bus_lines.c                 # C auxiliary sort utilities
├── sort_bus_lines.h                 # C header: sort utility prototypes
├── memory_visualization.c           # C memory diagnostics & visualization utilities
├── Makefile                         # Build automation: compiles all .c → libsortbuslines.so
│
├── templates/
│   └── index.html                   # Single-page dashboard: Tailwind CSS + Chart.js frontend
│
├── market_data.csv                  # [GENERATED] Live fundamental snapshot (real-time)
├── historical_fundamental_data.csv  # [GENERATED] Quarterly historical fundamental panel
├── requirements.txt                 # Python package dependencies
│
├── linux_env/                       # [GITIGNORED] Python virtual environment (WSL)
├── __pycache__/                     # [GITIGNORED] Python bytecode cache
├── *.o                              # [GITIGNORED] C object files
└── libsortbuslines.so               # [GITIGNORED] Compiled shared library (generated by make)
```

---

## Design Principles

1. **Hybrid Performance Architecture**: Python handles all I/O, HTTP routing, and data transformation. C owns all compute-heavy operations (scoring and sorting), called via a zero-overhead `ctypes` FFI bridge. This separation ensures the codebase remains readable and maintainable at the Python layer while retaining C-level performance for the inner scoring loop.

2. **Look-Ahead Bias Elimination**: Every point-in-time simulation step slices the historical fundamental panel strictly at or before the anchor date $T_k$. No future data ever contaminates a historical ranking decision.

3. **Outlier-Resistant Normalization**: The use of Median + MAD (rather than Mean + Std) for Z-score computation ensures that a single anomalous value (e.g., a one-time impairment charge) cannot distort the relative rankings of an entire sector peer group.

4. **Anti-Throttling Data Hygiene**: All yfinance downloads for the backtesting simulation are executed as a **single upfront bulk call** for the entire universe. This prevents the exponential API request fan-out that would occur if each rebalance step issued its own download, protecting against Yahoo Finance rate limiting.

---

*AlphaQuant V3.0 — Academic & Research Use Only*
*Department of Computer Science, The Hebrew University of Jerusalem*
