# AlphaQuant V3.3 — Hybrid Quantitative Equity Research & Rolling Backtesting Engine

[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![C GCC](https://img.shields.io/badge/C-GCC%2011%2B%20%7C%20Shared%20Lib-orange.svg)](https://gcc.gnu.org/)
[![Flask 3.x](https://img.shields.io/badge/Flask-3.x-green.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-Academic-lightgrey.svg)]()

---

## 1. Executive Summary & Objective

**AlphaQuant V3.3** is an institutional-grade hybrid C/Python quantitative equity research server and rolling-window portfolio backtester. It is designed to execute high-throughput, cross-sectional multi-factor equity scoring across 3,000+ US equities in sub-second latency while simulating point-in-time dynamic rebalancing strategies benchmarked against the SPY ETF.

### Core Business Purpose
The platform bridges raw corporate fundamental filings and daily market price data with a compiled high-performance calculation engine. Users construct custom factor weighting schemes across 16 financial indicators, select strategy profiles (e.g. *Momentum*, *Hyper-Growth*, *Quality*, *Deep Value*, *Income*), apply demographic filters (GICS sectors, market cap bounds), and simulate multi-year historical performance without look-ahead bias.

### Primary Capabilities
- **High-Speed C-FFI Scoring Engine**: Contiguous 80-byte memory layout and QuickSort algorithm in C linked via `ctypes` for zero-copy scoring.
- **Cross-Sectional Sector-Neutral Normalization**: Outlier-resistant Sigmoid Z-score normalizations using sector medians and Median Absolute Deviation (MAD), alongside piecewise Gaussian Goldilocks decay curves.
- **Zero Look-Ahead Bias Backtesting**: Point-in-time fundamental screening combined with PyArrow Parquet pushdown lazy price ingestion.
- **Two-Tier In-Memory TTL Cache**: Sub-millisecond fundamental data access and cached technical indicator calculation to bypass disk I/O.

---

## 2. System Architecture & Tech Stack

AlphaQuant uses a tiered architecture separating heavy computational math (C dynamic library) from data orchestration (Python/Flask) and interactive analytics (HTML5/Tailwind/Chart.js).

```
                      +---------------------------------------+
                      |         Web Browser Frontend          |
                      |  Tailwind CSS | Chart.js | DOM Async  |
                      +-------------------+-------------------+
                                          | JSON Payload
                                          v
                      +-------------------+-------------------+
                      |      Flask Application (main.py)      |
                      +-------------------+-------------------+
                                          |
        +---------------------------------+---------------------------------+
        |                                                                   |
        v                                                                   v
+-------+-----------------------+                         +-----------------+-----------------------+
|  In-Memory TTL Cache System   |                         |   Offline ETL & Data Lake Storage     |
| _LIVE_DATA_CACHE  (30m TTL)   |                         | historical_fundamental_data.csv       |
| _TECH_DATA_CACHE  (30m TTL)   |                         | us-shareprices-daily.parquet (PyArrow)|
+-------+-----------------------+                         +-----------------+-----------------------+
        |                                                                   |
        +---------------------------------+---------------------------------+
                                          |
                                          v
                      +-------------------+-------------------+
                      |  Financial Orchestration Module       |
                      |  (financial_module.py)                |
                      |  Normalization & Knockout Filters     |
                      +-------------------+-------------------+
                                          |
                                          v (ctypes zero-copy FFI)
                      +-------------------+-------------------+
                      |    Compiled C Engine (libsortbuslines)|
                      |  Stock Struct [80B] | QuickSort Kernel|
                      +---------------------------------------+
```

### Technology Stack Overview
| Layer | Technologies / Tools | Purpose |
| :--- | :--- | :--- |
| **Core Computational Kernel** | C (GCC `-O3 -shared -fPIC`), `ctypes` | Multi-factor weighted dot-product scoring and in-place QuickSort |
| **Backend Framework** | Python 3.9+, Flask 3.x, Gunicorn WSGI | REST API routing, query validation, JSON serialization |
| **Data Processing & Analytics** | Pandas, NumPy, PyArrow | Vectorized math, sector statistics, Parquet column/date pushdown I/O |
| **Data Storage / Lake** | Apache Parquet (Snappy), CSV | Compressed price time-series (`us-shareprices-daily.parquet`) |
| **Frontend UI** | HTML5, Tailwind CSS, Chart.js, Vanilla JS | Async single-page dashboard with real-time performance charting |

### Frontend Architecture
The single-page web client (`templates/index.html`) provides interactive analytics:
- **Async Polling & DOM Streaming**: Non-blocking `fetch()` requests submit 16-element weight vectors and strategy parameters to `/api/calculate` and `/api/backtest`.
- **Chart.js Compounding Growth Visualizer**: Dynamic canvas rendering comparing strategy portfolio NAV against the SPY benchmark over 3Y and 5Y windows.
- **16-Factor Weight Sliders & Filter Panels**: Dynamic UI control over factor weights, GICS sector filters, and market capitalization boundaries.

### Memory & Latency Optimizations
1. **Two-Tier In-Memory TTL Cache (`financial_module.py`)**:
   - `_LIVE_DATA_CACHE` (30-minute TTL): Retains baseline raw fundamental snapshots in RAM, returning `<1ms` cache hits.
   - `_TECH_DATA_CACHE` (30-minute TTL): Retains the global 370-day pivoted price matrix (`full_price_df`) and aligned SPY series (`full_spy_s`), eliminating 9.2s disk reads per request.
2. **Selective Live Quote Enrichment**: Inverts live data calls to query Tiingo's API *only* for the top 100 surviving stocks after scoring, reducing HTTP batches from 64 down to 2 and avoiding rate limits (HTTP 429).
3. **Lazy Price Ingestion in Backtest**: Screener identifies surviving tickers (~10-50 tickers) *first*, then loads daily prices from Parquet using PyArrow predicates (`Ticker` whitelist + `Date` bounds).
4. **Memory Footprint Control**: Explicit `float32` downcasting, pre-sorting DataFrames to use `drop_duplicates(subset=['Ticker'], keep='last')` instead of memory-heavy `groupby().last()`, and inner-loop `gc.collect()`.

---

## 3. Data Provenance & ETL Pipeline

AlphaQuant operates under a **Pure Offline Mandate** for simulation and fundamental screening. Raw market data is ingested, transformed, and cached via dedicated offline ETL scripts (`simfin_etl.py`).

```
+-----------------------------------------------------------------------------------+
|                            Raw Bulk SimFin CSV Datasets                           |
| us-income-ttm.csv | us-cashflow-ttm.csv | us-balance-quarterly.csv | us-companies.csv |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------+-----------------------------------------+
|                    simfin_etl.py Processing Pipeline                            |
| 1. Sector Alignment via IndustryId mapping                                        |
| 2. Point-in-Time Match: Fundamentals Publish Date <= Daily Share Price Date       |
| 3. Trailing PE Calculation: Price / EPS_TTM                                       |
| 4. Technical Indicator Generation (14-day RSI, 200SMA, RS, Beta)                  |
| 5. Zero Look-Ahead Bias Verification & Safe Float Coercion                        |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------+-----------------------------------------+
|                     Generated Offline Data Artifacts                              |
| historical_fundamental_data.csv   |   us-shareprices-daily.parquet                |
+-----------------------------------------------------------------------------------+
```

### ETL Key Rules (`simfin_etl.py`)
1. **Zero Look-Ahead Bias**: Fundamental statements are joined to price technicals strictly using the statement's **Publish Date** (`Publish Date <= Price Date`). Statements become visible to the engine only *after* public filing.
2. **Trailing PE Substitution**: `ForwardPE` is computed as $\text{Price} / \text{EPS}_{\text{TTM}}$ using reported trailing twelve-month earnings.
3. **No Data Spoofing**: Missing data fields are safely coerced to `0.0` or neutral defaults (`_safe_float`) without artificial backfilling.

---

## 4. Financial & Mathematical Engine

The core scoring engine maps 16 quantitative financial metrics onto normalized scales $[0, 100]$ using specific statistical transformations before executing C-FFI vector scoring.

### The 16 Quantitative Indicators
| Index | Indicator Key | Type | Description |
| :---: | :--- | :--- | :--- |
| `0` | `ForwardPE` | Valuation | Trailing Price-to-Earnings ($\text{Price} / \text{EPS}_{\text{TTM}}$) |
| `1` | `EV_EBITDA` | Valuation | Enterprise Value to EBITDA ratio |
| `2` | `FCFYield` | Valuation | Free Cash Flow Yield ($\text{FCF} / \text{Market Cap}$) |
| `3` | `RevenueGrowth` | Growth | Trailing 12-month Revenue Growth Rate |
| `4` | `EPSGrowth` | Growth | Trailing 12-month Diluted EPS Growth Rate |
| `5` | `GrossMargin` | Profitability | Gross Profit / Total Revenue |
| `6` | `OperatingMargin` | Profitability | Operating Income / Total Revenue |
| `7` | `ROIC` | Efficiency | Return on Invested Capital |
| `8` | `DebtToEquity` | Health | Total Debt / Total Shareholder Equity |
| `9` | `ShareholderYield` | Return | $(\text{Dividends} + \text{Share Repurchases} - \text{Debt Issued}) / \text{Market Cap}$ |
| `10` | `FiveYDivGrowth` | Return | 5-Year Compound Annual Dividend Growth Rate |
| `11` | `PayoutRatio` | Safety | Dividend Payout Ratio ($\text{Dividends} / \text{Net Income}$) |
| `12` | `Beta` | Risk | 1-Year Daily Return Beta vs SPY Benchmark |
| `13` | `RSI` | Momentum | 14-Day Wilder Relative Strength Index |
| `14` | `PriceVs200SMA` | Momentum | Ratio of Price to 200-Day Simple Moving Average |
| `15` | `RelativeStrength` | Momentum | 1-Year Excess Return relative to SPY Benchmark |

---

### Statistical Normalization Models

#### A. Asymmetric Piecewise Gaussian (Goldilocks Parameters)
For range-bound indicators where metrics that are too low *or* too high represent risks, scores are calculated using Gaussian decay on out-of-bound values:

$$\text{Score}(x) = \begin{cases} 
100 \cdot \exp\left(-\frac{(x - L)^2}{2 \sigma_L^2}\right) & \text{if } x < L \\ 
100.0 & \text{if } L \le x \le U \\ 
100 \cdot \exp\left(-\frac{(x - U)^2}{2 \sigma_R^2}\right) & \text{if } x > U 
\end{cases}$$

##### Configured Goldilocks Bounds (`GOLDILOCKS_PARAMS`)
- `PayoutRatio`: $[L=0.35, U=0.70, \sigma_L=0.25, \sigma_R=0.20]$
- `RSI`: $[L=45.0, U=60.0, \sigma_L=15.0, \sigma_R=10.0]$
- `ForwardPE`: $[L=12.0, U=28.0, \sigma_L=5.0, \sigma_R=15.0]$
- `EV_EBITDA`: $[L=8.0, U=16.0, \sigma_L=3.0, \sigma_R=8.0]$
- `DebtToEquity`: $[L=0.1, U=0.8, \sigma_L=0.4, \sigma_R=0.6]$
- `Beta`: $[L=0.8, U=1.2, \sigma_L=0.6, \sigma_R=0.4]$
- `PriceVs200SMA`: $[L=1.02, U=1.15, \sigma_L=0.10, \sigma_R=0.08]$
- `ShareholderYield`: $[L=0.04, U=0.09, \sigma_L=0.03, \sigma_R=0.04]$

---

#### B. Robust Sector-Neutral Sigmoid Z-Score Normalization
Unconstrained growth metrics (`RevenueGrowth`, `EPSGrowth`) are normalized **within their GICS sector peer group** using sector medians and Median Absolute Deviation (MAD):

$$Z_{i} = \frac{x_{i} - \text{Median}_{\text{sector}}}{\max\left(\text{MAD}_{\text{sector}} \cdot 1.4826, \, 10^{-4}\right)}$$

$$\text{Score}(x_{i}) = \frac{100}{1 + e^{-1.5 \cdot Z_{i}}}$$

Where $\text{MAD} = \text{Median}(|x_i - \text{Median}(x)|)$. If lower values are better (e.g. `ForwardPE`), the Z-score is inverted ($Z = -Z$).

---

### Strategy Knock-Out Filter Gates (`KNOCKOUT_FILTERS`)
Before scoring, assets failing strategy binary gates are removed:
- **`deep-value`**: `FCFYield > 0.0`, `OperatingMargin > 0.0`
- **`hyper-growth`**: `RevenueGrowth > 0.0`, `GrossMargin > 0.20`
- **`momentum`**: `PriceVs200SMA > 0.0` (verifies valid historical SMA data)
- **`income`**: `PayoutRatio > 0.0`, `PayoutRatio < 0.85`, `ShareholderYield > 0.015`
- **`quality`**: `ROIC > 0.08`, `CurrentRatio > 1.0`

---

### High-Performance C Engine Implementation (`financial_engine.c`)

#### Memory Struct Alignment (`stock.h`)
The C scoring kernel operates on 80-byte contiguous memory structures aligned with `ctypes.Structure`:
```c
typedef struct {
    char   ticker[8];       // 8 bytes — Ticker symbol string
    float  indicators[16];  // 64 bytes — Normalized indicator vector (0-100)
    double final_score;     // 8 bytes — Weighted inner product result
} Stock;                    // Total: 80 bytes (ctypes size asserted)
```

#### Vector Dot-Product Scoring
```c
void compute_scores(Stock* stocks, int num_stocks, const double* weights) {
    if (!stocks || num_stocks <= 0 || !weights) return;
    for (int i = 0; i < num_stocks; i++) {
        double score = 0.0;
        for (int j = 0; j < 16; j++) {
            score += (double)stocks[i].indicators[j] * weights[j];
        }
        stocks[i].final_score = score;
    }
}
```

#### In-Place QuickSort with Uniform Pivot Termination
`sort_stocks_by_score` executes an in-place QuickSort ordering stocks by `final_score` descending. The partition routine includes early termination when detecting uniform pivot values, protecting against $O(N^2)$ worst-case behavior on identical score ties.

---

## 5. Execution Pipeline & Data Flow

The complete step-by-step lifecycle of a live screener request via `/api/calculate`:

```
[HTTP POST /api/calculate]
       |
       v
1. Payload Validation -------> Verify weights[16], validate strategy & filters
       |
       v
2. Baseline Data Load -------> Fetch raw snapshot from _LIVE_DATA_CACHE (<1ms)
       |
       v
3. Static Demographics ------> Apply sector list & market cap filters (apply_dynamic_filters)
       |
       v
4. Technical Indicators ------> Fetch technicals from _TECH_DATA_CACHE (RelativeStrength, SMA, RSI, Beta)
                                - Non-momentum: Knockouts apply BEFORE technical calculation
                                - Momentum: Technicals compute BEFORE SMA knockout gate
       |
       v
5. Statistical Normalization -> Execute normalize_stocks (Goldilocks + Sector-Neutral Sigmoid Z-Scores)
       |
       v
6. C Engine Execution --------> Pack into Stock array, call compute_scores & sort_stocks_by_score via ctypes
       |
       v
7. Hard Truncation ----------> Truncate universe strictly to Top 100 assets
       |
       v
8. Live Price Enrichment ----> Fetch real-time quotes via fetch_tiingo_live_data for Top 100
       |
       v
9. JSON Serialization -------> Return JSON with raw metrics, normalized scores, and final scores
```

---

## 6. Backtest Simulation Engine (`backtest_engine.py`)

The backtest engine performs chronological rolling-window simulation over historical quarter-end anchor dates $\{T_0, T_1, \dots, T_n\}$.

```
Rebalancing Step (Quarter T_k):
1. Slice historical fundamental panel: df[Quarter_End_Date <= T_k]
2. Deduplicate: panel_slice.drop_duplicates(subset=['Ticker'], keep='last')
3. Normalize & Score via C Engine -> Select Top 10 assets for holding period [T_k, T_{k+1})
4. Lazy-load prices from us-shareprices-daily.parquet (PyArrow pushdown filter)
5. Calculate daily returns: R_portfolio(t) = Mean(R_i(t))
6. Compute performance metrics & scalar KPI extraction (_scalar_float)
```

### Strategy Rebalancing Frequency
- `momentum`, `custom`: Every 3 months (1 quarter step)
- `hyper-growth`, `quality`: Every 6 months (2 quarter steps)
- `deep-value`, `income`: Every 12 months (4 quarter steps)

### Robust Scalar Extraction & Performance Formulas
To avoid `TypeError: float() argument must be a string or a real number, not 'Series'`, all Pandas reductions are safely converted to scalars via `_scalar_float()`:

- **Annualized Return**: $R_{\text{ann}} = P_{\text{last}}^{252 / N} - 1$
- **Annualized Volatility**: $\sigma_{\text{ann}} = \text{std}(R_{\text{daily}}) \cdot \sqrt{252}$
- **Sharpe Ratio (4% risk-free rate)**: $\text{Sharpe} = \frac{R_{\text{ann}} - 0.04}{\sigma_{\text{ann}}}$
- **Maximum Drawdown**: $\text{MDD} = \min\left(\frac{\text{NAV}_t - \text{Peak}_t}{\text{Peak}_t}\right)$
- **Excess Return Alpha vs SPY**: $\alpha = R_{\text{ann}}^{\text{Portfolio}} - R_{\text{ann}}^{\text{SPY}}$

---

## 7. Developer Onboarding & Extensibility Guide

### System Prerequisites
- **Python**: $\ge 3.9$
- **C Compiler**: GCC $\ge 11$ with `make`
- **Operating System**: Linux / WSL (Windows Subsystem for Linux)

### Step 1 — Clone & Set Up Environment
```bash
# Clone the repository
git clone https://github.com/yubranvalde-byte/finance-app.git
cd finance-app

# Create virtual environment inside Linux / WSL
python3 -m venv venv
source venv/bin/activate

# Install python dependencies
pip install -r requirements.txt
```

### Step 2 — Compile the C Shared Library
```bash
make clean && make
```
*Compiles `financial_engine.c` into `libsortbuslines.so` using `-O3 -shared -fPIC`.*

### Step 3 — Run the Development Server
```bash
python main.py
```
*Access dashboard at `http://127.0.0.1:5000`.*

### Step 4 — Production Cloud Deployment (e.g. Render)
- **Build Command**: `make && pip install -r requirements.txt`
- **Start Command**: `gunicorn main:app`
- **Environment Variables**: `TIINGO_API_KEY` (Optional for live quote enrichment).

---

### Extensibility Guide: Adding a New Indicator

To add a 17th quantitative indicator (e.g. `PriceToBook`):

1. **Update C Struct Header (`stock.h`)**:
   ```c
   #define NUM_INDICATORS 17  // Increase from 16
   
   typedef struct {
       char   ticker[8];
       float  indicators[17]; // Increase size
       double final_score;
   } Stock;
   ```

2. **Update Python FFI Assertion & Column Registry (`financial_module.py`)**:
   ```python
   NUM_INDICATORS = 17
   INDICATOR_COLUMNS = [
       ...
       'PriceToBook',  # Add at position 16
   ]
   
   # Assert exact byte size matching C struct
   assert ctypes.sizeof(Stock) == 84, f"Stock size mismatch: {ctypes.sizeof(Stock)}"
   ```

3. **Configure Normalization Route (`financial_module.py`)**:
   Add `PriceToBook` to `GOLDILOCKS_PARAMS` or let it route to the Sigmoid Z-score logic.

4. **Update Frontend UI (`templates/index.html`)**:
   Add a 17th weight slider in the dashboard panel and pass the updated array to `/api/calculate`.

5. **Recompile C Library**:
   ```bash
   make clean && make
   ```

---

*AlphaQuant V3.3 — Quantitative Research & Portfolio Analytics Engine*
