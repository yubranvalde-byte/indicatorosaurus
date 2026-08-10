# AlphaQuant V3.3 — Hybrid Quantitative Screening & Dynamic Rebalancing Platform

[![Python 3.x](https://img.shields.io/badge/Python-3.x-blue)](https://www.python.org/)
[![C](https://img.shields.io/badge/C-GCC%20FFI-lightgrey)](https://gcc.gnu.org/)
[![Flask](https://img.shields.io/badge/Flask-2.x-green)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-Academic-orange)]()

---

## Executive Summary

**AlphaQuant V3.3** is a high-performance, institutional-grade quantitative equity research and backtesting server. It is architected as a **hybrid Python/C application** that combines a real-time cross-sectional scoring engine with a historical rolling-window, point-in-time portfolio simulator, now powered by an offline **SimFin Bulk ETL Pipeline** (`simfin_etl.py`).

The platform ingests local offline fundamental data across the full S&P 500 and Nasdaq universes (500+ tickers), normalizes the data using sector-neutral, outlier-resistant sigmoid Z-score transformations, and scores every asset via a compiled C-FFI scoring kernel. Users interact with a dynamic institutional-grade web dashboard to rank equities under customizable factor weight profiles (Momentum, Hyper-Growth, Quality, Deep Value, Income), apply GICS sector and market-cap filters, and trigger a trailing backtest simulation that dynamically rebalances the portfolio at strategy-appropriate frequencies, benchmarked against the SPY ETF.

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

**Files:** `main.py`, `financial_module.py`, `simfin_etl.py`

#### SimFin Bulk Offline ETL Pipeline & Ingestion (`simfin_etl.py` & `financial_module.py`) — V3.3
The standalone ETL module parses raw SimFin bulk CSVs (`data/raw_simfin/`) and outputs `historical_fundamental_data.csv`:
- Merges Income Statement TTM, Cash Flow TTM, and Balance Sheet Quarterly with metadata (`us-companies.csv` and `industries.csv`).
- Enforces **Zero Look-Ahead Bias** by using statement `Publish Date` to perform point-in-time merges with daily price technicals.
- Calculates all 16 proprietary quantitative indicators natively, substituting `ForwardPE` with Trailing PE (Price / EPS_TTM).
- `financial_module.py` features `load_latest_fundamental_snapshot()`, which dynamically extracts the most recent cross-sectional market snapshot per stock (3,189 tickers) to feed the C-FFI scoring engine.

#### Cross-Sectional GICS-Neutral Sigmoid Z-Score Normalization
Raw fundamental indicators span incompatible magnitudes across sectors. To eliminate this sector bias, each indicator is normalized **within its GICS sector peer group** using a robust sigmoid Z-score:

$$Z_{i,j} = \frac{x_{i,j} - \tilde{x}_{j,\text{sector}}}{\text{MAD}_{j,\text{sector}} \cdot 1.4826}$$

$$\hat{x}_{i,j} = \text{sigmoid}(Z_{i,j}) = \frac{1}{1 + e^{-Z_{i,j}}}$$

where:
- $\tilde{x}_{j,\text{sector}}$ is the **sector-level median** of indicator $j$ (resistant to outliers)
- $\text{MAD}_{j,\text{sector}}$ is the **Median Absolute Deviation** (robust scale estimator)
- The $1.4826$ constant is the consistency factor making MAD a consistent estimator of the standard deviation under normality

The sigmoid squashing maps every score to $(0, 1)$, guaranteeing numerical stability regardless of how extreme an outlier is. 

#### Asymmetric Gaussian Decay (Goldilocks Parameters)
For range-bound indicators (e.g., Payout Ratio, RSI) where values too high or too low are negative signals, the system enforces a strict optimal range $[L, U]$ using exponential decay on out-of-bounds metrics, governed by standard deviation variables $\sigma_L$ and $\sigma_R$.

#### Dynamic Sector & Knock-Out Filter Registry
A strategy-specific knock-out filter registry applies hard binary gates **before** the scoring pass, eliminating assets that structurally violate the strategy mandate (e.g., EPS Growth below 0 for Momentum, or no dividend for Income).

---

### 3. The Lean Simulation Layer — `backtest_engine.py` (V3.2.1 Update)

#### Pure Offline Mandate
In version 3.2.1, a **Pure Offline Mandate** was implemented. The system now strictly reads from the local `historical_prices_cache.pkl` and `historical_fundamental_data.csv` datasets. All run-time `yfinance` network calls have been eradicated. If local cache data is missing or corrupted, the system safely raises a local file error without violating the isolated network state.

#### Intelligent Fallback & The Empty Portfolio Fix
To comply with the "No Data Spoofing" principle, missing historical indicators (`NaN`) are gracefully coerced to `0.0` using `_safe_float` rather than artificially filled or aggressively dropped. This solves the "Empty Portfolio Bug" where strict knockouts previously decimated the asset universe, forcing the portfolio to drop down to the SPY benchmark.

If a stringent strategy knockout combined with sector filters eliminates all stocks, the engine employs an **Intelligent Fallback**—relaxing the knockouts and utilizing a dynamic market-cap filter alone to capture the top 10 ranked assets, ensuring alpha generation is preserved. If the overall market lacks data entirely, it falls back to a `last_valid_top_10` state to maintain holding continuity.

#### Rolling Point-in-Time Rebalancing Loop
The backtest engine executes a discrete chronological time-loop over all historical quarter-end anchor dates $\{T_0, T_1, \ldots, T_n\}$:
1. Slice the historical fundamental panel at or prior to $T_k$ ($df[\text{Quarter\_End\_Date} \le T_k]$) and extract the latest available fundamental report per stock as of $T_k$ (Zero Look-Ahead Bias).
2. Apply knock-out filters and normalize the cross-section.
3. Score all assets via the C-engine and extract the Top 10.
4. Assign the Top 10 as the active portfolio for $[T_k, T_{k+1})$.
5. Compute daily portfolio returns using dynamic re-weighting.

At each rebalance point, the engine uses only data available **at or before** $T_k$, strictly enforcing **Zero Look-Ahead Bias**.

#### Strategy-Based Dynamic Rebalancing Frequency Matrix
| Strategy     | Rebalancing Frequency | Step (quarters) |
|:------------|:---------------------:|:---------------:|
| `momentum`   | Every 3 months        | 1               |
| `hyper-growth` | Every 6 months      | 2               |
| `quality`    | Every 6 months        | 2               |
| `deep-value` | Every 12 months       | 4               |
| `income`     | Every 12 months       | 4               |
| `custom`     | Every 3 months        | 1               |

---

### 4. The Analytical Frontend — `templates/index.html`

The single-page dashboard is built with **Tailwind CSS** and **Chart.js**, served by Flask's Jinja2 template engine. The UI polls the backend asynchronously to maintain high responsiveness without blocking the client thread.

#### Interactive Feature Highlights
- **16-Indicator Weight Sliders**: Live DOM-driven configuration of the scoring weight vector, streamed to the backend on every `Calculate Score` click.
- **Strategy Preset Profiles**: Momentum, Hyper-Growth, Quality, Deep Value, Income.
- **Backtest Dashboard**: An institutional-grade card component with a time-window toggle (3Y / 5Y), returning a full performance timeseries and KPI block.
- **XSS-Safe Rendering**: Custom `escapeHTML()` sanitation is used before injecting any ticker metadata into the DOM.
- **Real-Time Charting**: Dual-line compounding growth chart (`Portfolio` vs `SPY`) rendered using Chart.js with dynamic canvas rebuilds.

---

## Mathematical & Quantitative Models

### Dynamic Net Shareholder Yield
$$\text{Shareholder Yield} = \frac{-(\text{Dividends Paid} + \text{Net Stock Flow} + \text{Net Debt Flow})}{\text{Market Cap}}$$

### Backtesting Performance Metrics
**Annualized Return:** $R_{\text{ann}} = \left(\prod_{t=1}^{T}(1 + r_t)\right)^{\frac{252}{T}} - 1$
**Annualized Volatility:** $\sigma_{\text{ann}} = \sqrt{252} \cdot \text{std}(\{r_t\}_{t=1}^{T})$
**Sharpe Ratio** (with 4% risk-free hurdle): $\text{Sharpe} = \frac{R_{\text{ann}} - r_f}{\sigma_{\text{ann}}}, \quad r_f = 0.04$
**Maximum Drawdown:** $\text{MDD} = \min_{t \in [0,T]} \frac{\text{NAV}(t) - \max_{\tau \leq t} \text{NAV}(\tau)}{\max_{\tau \leq t} \text{NAV}(\tau)}$
**Excess Return Alpha vs SPY:** $\alpha = R_{\text{ann}}^{\text{Portfolio}} - R_{\text{ann}}^{\text{SPY}}$

---

## V3.3 Roadmap (Next Steps)

1. **Offline Bulk Ingestion & Parquet Data Lake Compression**: Transitioning from on-demand scraping to a 100% offline data lake. High-density daily price history (6.2M+ rows, originally 415MB CSV) is compressed to <30MB using Snappy-compressed Apache Parquet (`us-shareprices-daily.parquet`), tracked directly in Git for zero-dependency cloud deployment.
2. **Pure Offline Backtesting Mandate**: All simulation runtime dependencies (fundamentals, stock price matrices, and benchmark SPY historical data) are stored statically within the repository (`data/spy_historical.csv` & `historical_fundamental_data.csv`). Zero network calls occur during backtest execution.
3. **Time-Series Forward-Fill Alignment**: Pinpointing exact corporate earnings reporting dates to align fundamentals directly to daily trading data without look-ahead bleed.
4. **Total Return Integration**: Injecting historic dividend payments directly into adjusted close prices to reflect true compounded total return vs SPY benchmark.

---

## Technical Notes & Memory Optimization

> [!IMPORTANT]
> **API Rate Limiting & Web Scraper Resilience:**
> - **Yahoo Finance & Wikipedia Scraper Throttling**: Auxiliary background scripts (e.g., `fetch_static_data.py`) that pull live market indicators from Yahoo Finance or Wikipedia may encounter HTTP 429 (*Too Many Requests*) rate limits or missing `lxml` warnings when executed on cloud server IP ranges (e.g., Render, AWS, Heroku).
> - **Zero-Downtime Fallback Architecture**: The core simulation engine (`backtest_engine.py`) is **completely decoupled** from external network scrapers. If web scraping triggers throttling during build/startup, `fetch_static_data.py` exits gracefully with status `0`, allowing the web application server (`gunicorn main:app`) to boot seamlessly using pre-built static datasets (`market_data.csv`, `spy_historical.csv`, and `us-shareprices-daily.parquet`).

> [!TIP]
> **C-FFI Boundary, In-Memory TTL Cache & RAM Safety Architecture:**
> - **Lazy Price Ingestion (>99% RAM Reduction)**: `backtest_engine.py` executes fundamental quarter-by-quarter portfolio screening *first* to identify the exact ~10–25 unique tickers present in the portfolio, then lazy-loads prices from `us-shareprices-daily.parquet` using PyArrow pushdown filters (`filters=[('Ticker', 'in', target_tickers)]`). This loads ~3,500 rows instead of 6,200,000 rows into RAM (< 5MB RAM vs 400MB RAM), completely eliminating Render OOM crashes.
> - **Dynamic Technical Indicator Engine**: `financial_module.py` (`compute_live_technical_indicators`) lazy-loads the trailing 260 trading days from `us-shareprices-daily.parquet` and `spy_historical.csv` to dynamically calculate real-time momentum indicators (`RelativeStrength`, `PriceVs200SMA`, `RSI`, `Beta`) for 3,000+ tickers in ~1 second. This guarantees that technical strategy knockouts (e.g., Momentum strategy) evaluate live market metrics without false-positive zero eliminations.
> - **Vectorized Anchor Loop & Inner-Loop GC**: In `backtest_engine.py` Phase 1, fundamental snapshots are converted via fast vectorized dictionary mapping (`to_dict(orient='records')`) rather than slow `iterrows()` Series allocations. All intermediate slice DataFrames (`panel_slice`, `snapshot_df`) are immediately deleted (`del`), and `gc.collect()` is triggered at the end of **every** quarter iteration, guaranteeing flat RAM usage across multi-year backtest windows.
> - **In-Memory TTL Cache & Sub-Second Response**: The live screener endpoint (`/api/calculate`) reads fundamental datasets from an in-memory TTL cache (`get_cached_fundamental_snapshot`) with a 30-minute expiration window. Cache hits return in < 1ms, eliminating network latency for the C-FFI scoring engine.
> - **Copy Isolation & Mutation Leak Prevention**: `get_cached_fundamental_snapshot` returns isolated copies of cached dictionary records (`[dict(r) for r in cached_data]`), preventing Flask requests from mutating or accumulating state on the global cache object.
> - **Batched Tiingo API Integration & Throttling**: When a `TIINGO_API_KEY` (or `API_KEY`) environment variable is configured, the system fetches real-time prices from the Tiingo IEX API in 50-ticker batches (`fetch_tiingo_live_data`) with a 150ms inter-batch throttle delay and safe JSON validation, preventing HTTP 414 URI errors and `JSONDecodeError` worker crashes while falling back seamlessly to disk CSV (`historical_fundamental_data.csv`) if the key is absent or rate-limited.
> - **C-FFI Null & Empty Universe Guards**: C exported functions (`compute_scores`, `sort_stocks_by_score`) and Python wrappers in `financial_module.py` explicitly validate pointers (`stocks != NULL`) and array lengths (`num_stocks > 0`). If strategy knockouts eliminate all stocks, the FFI call is safely bypassed and triggers intelligent fallback (`['SPY']`), preventing C-level Segmentation Faults.
> - **Selective Column Projection (`usecols` / `columns`)**: CSV and Parquet readers load strictly necessary data columns, avoiding loading unused data fields into memory.
> - **Float32 Downcasting**: All numeric time-series and financial indicator matrices are immediately downcasted from `float64` to `float32`, cutting RAM consumption by 50%.
> - **Explicit Garbage Collection (`import gc`)**: Large intermediate DataFrames, HTTP response buffers, and stale cache objects are explicitly deleted (`del`) and garbage-collected (`gc.collect()`) at key pipeline milestones to maintain peak memory usage well below free-tier cloud limits (e.g., 512MB RAM on Render).

---

## Installation & Build Instructions

### Prerequisites
| Dependency | Version | Purpose |
|:-----------|:--------|:--------|
| Python | ≥ 3.9 | Orchestration, Flask server |
| GCC | ≥ 11 | C shared library compilation |
| `make` | any | Build automation |
| `pandas` | ≥ 2.2 | Data manipulation |
| `pyarrow` | ≥ 15.0 | High-performance Parquet I/O |
| `gunicorn` | ≥ 22.0 | WSGI HTTP Server for UNIX / Render deployment |
| `lxml` | ≥ 5.0 | HTML parsing for Wikipedia scrapers |
| `flask` | ≥ 3.0 | Web application framework |
| `numpy` | ≥ 1.26 | Numerical array calculations |

### Step 1 — Clone the Repository
```bash
git clone https://github.com/yubranvalde-byte/finance-app.git
cd finance-app
```

### Step 2 — Create and Activate a Python Virtual Environment (inside WSL/Linux)
```bash
python3 -m venv linux_env
source linux_env/bin/activate
pip install -r requirements.txt
```

### Step 3 — Utility Offline Data Lake Scripts (Optional / Maintenance)
If rebuilding offline datasets from scratch:
```bash
# 1. Fetch SPY benchmark history into local data lake (data/spy_historical.csv)
python fetch_spy_offline.py

# 2. Compress raw SimFin CSV (415MB) into high-efficiency Parquet (30MB)
python compress_prices_to_parquet.py
```

### Step 4 — Compile the C Shared Library
```bash
make clean && make
```
> ⚠️ **WSL Required on Windows**: The `Makefile` uses GCC. On Windows, all compilation and Python execution must be performed inside WSL (Windows Subsystem for Linux).

### Step 5 — Cloud Deployment (e.g., Render)
For automated cloud deployment (e.g., Render), set the **Build Command** to:
```bash
make && pip install -r requirements.txt
```
and the **Start Command** to:
```bash
gunicorn main:app
```

### Step 6 — Launch the Application Locally
```bash
python main.py
```
The Flask development server will start on `http://127.0.0.1:5000`.

---

*AlphaQuant V3.3 — Academic & Research Use Only*
*Department of Computer Science, The Hebrew University of Jerusalem*

