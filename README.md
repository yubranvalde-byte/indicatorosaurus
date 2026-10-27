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
- **Vectorized NumPy Normalization Pipeline**: `_vectorized_goldilocks` and `_vectorized_sigmoid` utilizing NumPy array operations (`np.where`, `np.exp`, `np.clip`) to process column matrices in ~50ms, replacing element-wise Python loops.
- **Cross-Sectional Sector-Neutral Normalization**: Outlier-resistant Sigmoid Z-score normalizations using sector medians and Median Absolute Deviation (MAD), alongside piecewise Gaussian Goldilocks decay curves.
- **Zero Look-Ahead Bias Backtesting**: Point-in-time fundamental screening combined with PyArrow Parquet pushdown lazy price ingestion.
- **Two-Tier In-Memory TTL Cache & Pre-Flight State Management**: Sub-millisecond fundamental data access and explicit route pre-flight technical indicator cache hydration to prevent cold-start I/O tax during requests.
- **Portfolio Intelligence Phase 1 — Hybrid Multi-Currency Price & Macro Resolver (`core_logic/price_resolver.py`)**: 3-Tier historical price resolution engine combining the local SimFin Parquet lake (`us-shareprices-daily.parquet`), a Snappy-compressed portfolio cache (`portfolio_price_cache.parquet`), and external fallback (Tiingo / `yfinance`) with automatic Tel Aviv Stock Exchange (`.TA`) Agorot-to-ILS scaling ($\text{Price}_{\text{ILS}} = \text{Price}_{\text{Agorot}} / 100.0$) and a continuous daily calendar macro engine (`fx_cpi_cache.parquet`) for `USD/ILS` FX and Israeli CPI series.
- **Portfolio Intelligence Phase 2 — Dual-Currency FIFO Ledger & Israeli Tax Engine (`core_logic/portfolio_ledger.py`)**: Real-time transaction persistence (`data/portfolio_transactions.parquet`), fractional FIFO queue state machine, and exact Israeli capital gains tax decomposition (25% on real gains, inflationary exemptions, YTD tax shielding, and loss carry-forward).
- **Portfolio Intelligence Phase 3 — Vectorized Time-Series & Attribution Engine (`core_logic/portfolio_analytics.py`)**: Continuous daily holdings matrix ($N_{\text{days}} \times M_{\text{tickers}}$), cash-flow invariant Time-Weighted Return (TWR), exact Money-Weighted Return (MWR / XIRR via Newton-Raphson), ILS FX & tax attribution, and counterfactual benchmark opportunity simulation.
- **Portfolio Intelligence Phase 4 — Low-Latency REST API Engine (`api/main.py`)**: Low-latency endpoints (`/api/portfolio/summary`, `/api/portfolio/transactions`, `/api/portfolio/analytics`) with in-memory memoization (< 100ms warm SLA) and strict RFC-8259 NaN-safety.

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
                      |      Flask Application (api/main.py)  |
                      | Explicit Pre-Flight ensure_tech_cache()|
                      +-------------------+-------------------+
                                          |
        +---------------------------------+---------------------------------+
        |                                                                   |
        v                                                                   v
+-------+-----------------------+                         +-----------------+-----------------------+
|  In-Memory TTL Cache System   |                         |   Offline ETL & Data Lake Storage     |
| _LIVE_DATA_CACHE  (30m TTL)   |                         | data/historical_fundamental_data.csv  |
| _TECH_DATA_CACHE  (30m TTL)   |                         | data/raw_simfin/...                   |
+-------+-----------------------+                         +-----------------+-----------------------+
        |                                                                   |
        +---------------------------------+---------------------------------+
                                          |
                                          v
                      +-------------------+-------------------+
                      |  Financial Orchestration Module       |
                      |  (core_logic/financial_module.py)     |
                      |  Vectorized Normalization & Knockouts |
                      +-------------------+-------------------+
                                          |
                                          v (ctypes zero-copy FFI)
                      +-------------------+-------------------+
                      |    Compiled C Engine (libsortbuslines)|
                      |  Stock Struct [80B] | QuickSort Kernel|
                      |  (c_kernel/)                          |
                      +---------------------------------------+
```

### Technology Stack Overview
| Layer | Technologies / Tools | Purpose |
| :--- | :--- | :--- |
| **AI System Context** | `ALPHAQUANT_AI_RULES.md` | Centralized prompt & engineering constraints specification |
| **Deterministic Scripts** | Python (`/scripts/*.py`) | Automated testing & execution: `compile_c_engine.py`, `run_rolling_backtest.py`, `verify_price_resolver.py`, `verify_portfolio_ledger.py`, `verify_portfolio_analytics.py`, `verify_portfolio_api.py` |
| **Core Computational Kernel** | C (GCC `-O3 -shared -fPIC`), `ctypes` | Multi-factor weighted dot-product scoring and in-place QuickSort |
| **Backend Framework** | Python 3.9+, Flask 3.x, Gunicorn WSGI | REST API routing, query validation, JSON serialization |
| **Data Processing & Analytics** | Pandas, NumPy, PyArrow | Vectorized math, sector statistics, Parquet column/date pushdown I/O, FIFO tax lot engine, XIRR solver |
| **Data Storage / Lake** | Apache Parquet (Snappy), CSV | `us-shareprices-daily.parquet`, `portfolio_price_cache.parquet`, `fx_cpi_cache.parquet`, `portfolio_transactions.parquet` |
| **Frontend UI** | HTML5, Tailwind CSS, Chart.js, Vanilla JS | Async single-page dashboard with real-time performance charting |

### Frontend Architecture
The single-page web client (`frontend/templates/index.html`) provides interactive analytics:
- **Async Polling & DOM Streaming**: Non-blocking `fetch()` requests submit 16-element weight vectors and strategy parameters to `/api/calculate` and `/api/backtest`.
- **Chart.js Compounding Growth Visualizer**: Dynamic canvas rendering comparing strategy portfolio NAV against the SPY benchmark over 3Y and 5Y windows.
- **16-Factor Weight Sliders & Filter Panels**: Dynamic UI control over factor weights, GICS sector filters, and market capitalization boundaries.

### Memory & Latency Optimizations
1. **Vectorized Early Pruning & Filter Hoisting (`apply_early_pruning_vectorized`)**: Hoists fundamental strategy knockouts (`INTRINSIC_KNOCKOUTS`), GICS sector bounds, and Gatekeeper Matrix cutoffs (`MarketCap >= $1B`, `Price >= $5.00`, `20-day ADTV >= $5M`) to execute immediately on a `pandas.DataFrame` in-memory cache via boolean masking *before* technical indicator hydration and *before* Z-score/Goldilocks normalization. Shrinks dataset matrix ($N$) in ~1ms.
2. **DataFrame In-Memory TTL Cache (`core_logic/financial_module.py`)**:
   - `_LIVE_DATA_CACHE` (30-minute TTL): Retains baseline raw fundamental snapshots natively as a `pandas.DataFrame`, returning isolated deep copies (`df.copy()`) in `<1ms` for fast sub-second vectorized masking without cache mutation.
   - `_TECH_DATA_CACHE` (30-minute TTL): Retains global price/volume matrices. Hydrated explicitly via route pre-flight calls (`ensure_tech_cache()`), removing synchronous I/O. Optional boot-time hydration gated by `WARM_CACHE_ON_STARTUP`.
3. **Decoupled Strategy Knockouts (`INTRINSIC_KNOCKOUTS` vs `DERIVED_KNOCKOUTS`)**: Fundamental gates (`FCFYield`, `ROIC`, `RevenueGrowth`, `GrossMargin`, `PayoutRatio`, `CurrentRatio`) run during early pruning; technical gates (`DERIVED_KNOCKOUTS` like `PriceVs200SMA > 0`) run post-hydration on the surviving asset subset.
4. **Selective Live Quote Enrichment**: Inverts live data calls to query Tiingo's API *only* for the top 100 surviving stocks after scoring, reducing HTTP batches from 64 down to 2 and avoiding rate limits (HTTP 429).
5. **Lazy Price Ingestion in Backtest**: Screener identifies surviving tickers (~10-50 tickers) *first*, then loads daily prices from Parquet using PyArrow predicates (`Ticker` whitelist + `Date` bounds).
6. **Memory Footprint Control & NaN Safety**: Explicit `float32` downcasting, pre-sorting DataFrames to use `drop_duplicates(subset=['Ticker'], keep='last')` instead of memory-heavy `groupby().last()`, inner-loop `gc.collect()`, and strict `.fillna(0.0).replace([np.inf, -np.inf], 0.0)` coercion prior to C-FFI / JSON serialization.

---

## 3. Data Provenance & ETL Pipeline

AlphaQuant operates under a **Pure Offline Mandate** for simulation and fundamental screening. Raw market data is ingested, transformed, and cached via dedicated offline ETL scripts (`data_pipeline/simfin_etl.py`).

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

### ETL Key Rules (`data_pipeline/simfin_etl.py`)
1. **Zero Look-Ahead Bias**: Fundamental statements are joined to price technicals strictly using the statement's **Publish Date** (`Publish Date <= Price Date`). Statements become visible to the engine only *after* public filing.
2. **Trailing PE Substitution**: `TrailingPE` is computed as $\text{Price} / \text{EPS}_{\text{TTM}}$ using reported trailing twelve-month earnings.
3. **Upstream Gatekeeper Filtering**: Upstream point-in-time filtering permanently drops illiquid records (`MarketCap < $1B` or `Price < $5.00`), generating a lean `24,262` row fundamental dataset across `1,866` institutional-grade equities (`data/historical_fundamental_data.csv`).
4. **Parquet Price Compression (`data_pipeline/compress_prices_to_parquet.py`)**: Converts 6.2M daily share price rows into Snappy-compressed `us-shareprices-daily.parquet` (415.4MB CSV $\rightarrow$ 53.8MB Parquet, 87% reduction).
5. **No Data Spoofing**: Missing data fields are safely coerced to `0.0` or neutral defaults (`_safe_float`) without artificial backfilling.

---

## 4. Financial & Mathematical Engine

The core scoring engine maps 16 quantitative financial metrics onto normalized scales $[0, 100]$ using specific statistical transformations before executing C-FFI vector scoring.

### The 16 Quantitative Indicators
| Index | Indicator Key | Type | Description |
| :---: | :--- | :--- | :--- |
| `0` | `TrailingPE` | Valuation | Trailing Price-to-Earnings ($\text{Price} / \text{EPS}_{\text{TTM}}$) |
| `1` | `EV_EBITDA` | Valuation | Enterprise Value to EBITDA ratio |
| `2` | `FCFYield` | Valuation | Free Cash Flow Yield ($\text{FCF} / \text{Market Cap}$) |
| `3` | `RevenueGrowth` | Growth | Trailing 12-month Revenue Growth Rate |
| `4` | `EPSGrowth` | Growth | Trailing 12-month Diluted EPS Growth Rate |
| `5` | `GrossMargin` | Profitability | Gross Profit / Total Revenue |
| `6` | `OperatingMargin` | Profitability | Operating Income / Total Revenue |
| `7` | `ROIC` | Efficiency | Return on Invested Capital |
| `8` | `NetDebt_EBITDA` | Health | Net Debt / EBITDA ratio |
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
- `TrailingPE`: $[L=12.0, U=28.0, \sigma_L=5.0, \sigma_R=15.0]$
- `EV_EBITDA`: $[L=8.0, U=16.0, \sigma_L=3.0, \sigma_R=8.0]$
- `NetDebt_EBITDA`: $[L=0.0, U=3.0, \sigma_L=1.0, \sigma_R=2.0]$
- `Beta`: $[L=0.8, U=1.2, \sigma_L=0.6, \sigma_R=0.4]$
- `PriceVs200SMA`: $[L=1.02, U=1.15, \sigma_L=0.10, \sigma_R=0.08]$
- `ShareholderYield`: $[L=0.04, U=0.09, \sigma_L=0.03, \sigma_R=0.04]$

---

#### B. Robust Sector-Neutral Sigmoid Z-Score Normalization
Unconstrained growth metrics (`RevenueGrowth`, `EPSGrowth`) are normalized **within their GICS sector peer group** using sector medians and Median Absolute Deviation (MAD):

$$Z_{i} = \frac{x_{i} - \text{Median}_{\text{sector}}}{\max\left(\text{MAD}_{\text{sector}} \cdot 1.4826, \, 10^{-4}\right)}$$

$$\text{Score}(x_{i}) = \frac{100}{1 + e^{-1.5 \cdot Z_{i}}}$$

Where $\text{MAD} = \text{Median}(|x_i - \text{Median}(x)|)$. If lower values are better (e.g. `TrailingPE`), the Z-score is inverted ($Z = -Z$).

---

### Vectorized Gatekeeper Matrix & Decoupled Knock-Out Filter Gates (`apply_early_pruning_vectorized`, `apply_derived_knockouts_vectorized`)
Before scoring, assets must pass global Gatekeeper Matrix cut-offs and strategy-specific binary gates implemented via vectorized DataFrame boolean masking:
- **Global Gatekeeper Matrix**:
  - `MarketCap >= $1B` ($1,000,000,000)
  - `Price >= $5.00`
  - `20-day ADTV >= $5M` (vectorized 20-day mean of Price * Volume)
  - `NetDebt_EBITDA <= 15.0` insolvency safety net.
- **Intrinsic Knock-Out Gates (`INTRINSIC_KNOCKOUTS` — Applied early in data flow)**:
  - **`deep-value`**: `FCFYield > 0.0`, `OperatingMargin > 0.0`
  - **`hyper-growth`**: `RevenueGrowth > 0.0`, `GrossMargin > 0.20`
  - **`income`**: `PayoutRatio > 0.0`, `PayoutRatio < 0.85`, `ShareholderYield > 0.015`
  - **`quality`**: `ROIC > 0.08`, `CurrentRatio > 1.0`
- **Derived Knock-Out Gates (`DERIVED_KNOCKOUTS` — Applied post-technical indicator hydration)**:
  - **`momentum`**: `PriceVs200SMA > 0.0` (verifies valid historical SMA data)

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
2. Baseline Data Load -------> Fetch raw snapshot DataFrame from _LIVE_DATA_CACHE (df.copy(), <1ms)
       |
       v
3. Vectorized Early Pruning -> Apply apply_early_pruning_vectorized() using vectorized pandas boolean masking:
                                - Gatekeeper Matrix: MarketCap >= $1B, Price >= $5.00, 20-Day ADTV >= $5M
                                - Demographics: GICS sector whitelisting and user-specified cap bounds
                                - Intrinsic Knockouts: Fundamental strategy rules (e.g. FCFYield > 0)
       |
       v
4. Technical Indicators ------> Fetch technicals for surviving tickers via compute_live_technical_indicators()
                                - Merge RelativeStrength, SMA, RSI, Beta onto surviving DataFrame
                                - Execute apply_derived_knockouts_vectorized() (e.g. PriceVs200SMA > 0)
       |
       v
5. Serialization & Safety ----> Coerce np.nan -> 0.0, replace inf, and cast NumPy types to Python scalars
       |
       v
6. Statistical Normalization -> Execute normalize_stocks (Goldilocks + Sector-Neutral Sigmoid Z-Scores)
       |
       v
7. C Engine Execution --------> Pack into Stock array, call compute_scores & sort_stocks_by_score via ctypes
       |
       v
8. Hard Truncation ----------> Truncate universe strictly to Top 100 assets
       |
       v
9. Live Price Enrichment ----> Fetch real-time quotes via fetch_tiingo_live_data for Top 100
       |
       v
10. JSON Serialization ------> Return JSON with raw metrics, normalized scores, and final scores
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
- `momentum`, `deep-value`, `income`, `custom`: Every 3 months (1 quarter step)
- `hyper-growth`, `quality`: Every 6 months (2 quarter steps)

### Robust Scalar Extraction & Performance Formulas
To avoid `TypeError: float() argument must be a string or a real number, not 'Series'`, all Pandas reductions are safely converted to scalars via `_scalar_float()`:

- **Annualized Return**: $R_{\text{ann}} = P_{\text{last}}^{252 / N} - 1$
- **Annualized Volatility**: $\sigma_{\text{ann}} = \text{std}(R_{\text{daily}}) \cdot \sqrt{252}$
- **Sharpe Ratio (4% risk-free rate)**: $\text{Sharpe} = \frac{R_{\text{ann}} - 0.04}{\sigma_{\text{ann}}}$
- **Maximum Drawdown**: $\text{MDD} = \min\left(\frac{\text{NAV}_t - \text{Peak}_t}{\text{Peak}_t}\right)$
- **Excess Return Alpha vs SPY**: $\alpha = R_{\text{ann}}^{\text{Portfolio}} - R_{\text{ann}}^{\text{SPY}}$

---

## 6A. Portfolio Intelligence Phase 1: Hybrid Price & Macro Resolution Engine (`core_logic/price_resolver.py`)

To support multi-currency US/TASE portfolio tracking and Israeli real (CPI/FX-adjusted) capital gains tax calculations in sub-millisecond warm latency, `core_logic/price_resolver.py` implements a 3-Tier equity resolution architecture paired with a continuous daily macro index engine:

### 1. 3-Tier Hybrid Equity Price Resolution (`resolve_prices`)
- **Tier 1 (Primary Local Lake — `us-shareprices-daily.parquet`)**: Queries institutional US equities via PyArrow predicate pushdown (`Ticker` whitelist and `Date` bounds) matching `backtest_engine.py`.
- **Tier 2 (Local Portfolio Cache — `data/portfolio_price_cache.parquet`)**: Stores and serves ETFs (`QQQ`, `SMH`, `SPY`), TASE securities (`.TA`), and promoted Tier 1 slices in a compact Snappy-compressed Parquet artifact (`~22ms` disk read; `< 1ms` in-memory TTL hit).
- **Tier 3 (External Fallback & Cache Hydration)**: Automatically fetches missing tickers or uncached date ranges via Tiingo REST API (`TIINGO_API_KEY`) or `yfinance` fallback and atomically merges them into `data/portfolio_price_cache.parquet`.
- **TASE Agorot Normalization**: Detects Tel Aviv Stock Exchange tickers (`.TA`) quoted in Agorot (`ILA`) and normalizes prices to ILS ($\text{Price}_{\text{ILS}} = \text{Price}_{\text{Agorot}} / 100.0$) prior to caching, returning native currency metadata (`USD` vs `ILS`).
- **Zero Synthetic Data & Zero Look-Ahead**: Strictly forward-filled (`.ffill()`, no `.bfill()`); raises `RuntimeError` if a ticker cannot be resolved from local caches or authoritative external sources.

### 2. Macro FX (`USD/ILS`) & Israeli CPI Engine (`resolve_macro_series`)
- **`USD_ILS`**: Fetched directly from the **Bank of Israel SDMX API** (`RER_USD_ILS` official representative rate).
- **`IL_CPI`**: Fetched from the **Israel Central Bureau of Statistics (CBS) API** (index `120010`), chain-linked across base-year transitions to the latest CBS base, and keyed strictly to its **publication date** (the 15th of month $M+1$, "המדד הידוע" / the known index).
- Reindexed onto a continuous daily calendar grid (`2010-01-01` to present) using strict chronological forward-fill (`ffill`) in `data/fx_cpi_cache.parquet`.
- Verified via `python scripts/verify_price_resolver.py`.

---

## 6B. Portfolio Intelligence Phase 2: Dual-Currency FIFO Ledger & Israeli Tax Engine (`core_logic/portfolio_ledger.py`)

`core_logic/portfolio_ledger.py` implements a persistent dual-currency (`USD` / `ILS`) transaction ledger, FIFO tax-lot accounting state machine, and Israeli capital gains tax calculator (25% on real gains):

### 1. Transaction Ledger, Persistence & Fault-Tolerant Brokerage CSV Importer (`PortfolioLedger`)
- **Supported Operations**: `BUY`, `SELL`, `DIVIDEND`, `CASH_DEPOSIT`, `CASH_WITHDRAWAL`.
- **Schema**: `tx_id`, `date` (`YYYY-MM-DD`), `ticker`, `tx_type`, `quantity` (fractional shares supported), `price` (native currency), `fee` (native currency), `currency` (`USD` or `ILS`), and `notes`.
- **Atomic Persistence**: Stored in Snappy-compressed `data/portfolio_transactions.parquet` via atomic `.tmp` + `os.replace` commits (only persisted after full FIFO replay validation passes, preventing oversells or corrupt state), with deterministic CSV import/export (`import_csv`, `export_csv`).
- **Schema-Adaptive Brokerage CSV Importer (`import_csv`)**:
  - **Fuzzy Header Resolution**: Case-insensitively maps third-party brokerage exports (including Yahoo Finance `portfolio.csv`) using priority rules (`Trade Date` / `Execution Date` over live quote `Date`; `Purchase Price` / `Trade Price` over `Current Price`; `Symbol` / `Ticker`; `Commission` / `Fee`; `Comment` / `Notes`; auto-inferring `ILS` for `.TA` tickers and `USD` otherwise).
  - **Flexible Date & Same-Day Ordering**: Parses `YYYYMMDD` integers/floats (e.g., `20231205.0`), `YYYY-MM-DD`, `YYYY/MM/DD`, and `DD/MM/YYYY`, and sorts transactions chronologically by `(date, type_priority, source_line)` (`BUY` before `SELL` on identical dates).
  - **Fractional Rounding Tolerance & Row-by-Row Diagnostics**: Clamps small broker rounding discrepancies ($\le 0.01$ shares, such as selling `73.87` shares against `73.867` open shares) to the open lot balance, and skips incomplete watchlist rows, zero-price rows, or genuine oversells ($> 0.01$ shares) while returning a full diagnostic report (`imported_count`, `total_rows`, `skipped_count`, `skipped_rows` with exact 1-based line numbers and missing/invalid field details).

### 2. Israeli Real Capital Gains Tax Decomposition (`compute_israeli_lot_tax`)
On each `SELL`, open FIFO lots are consumed chronologically and decomposed in `ILS`:
- **Original Cost (`ILS`)**: $\text{OriginalCost}_{\text{ILS}} = (Q \cdot P_{\text{buy}} + \text{Fee}_{\text{buy}}) \cdot \text{FX}_{\text{buy}}$ (where $\text{FX} = 1.0$ for `ILS` securities).
- **Net Proceeds (`ILS`)**: $\text{Proceeds}_{\text{ILS}} = (Q \cdot P_{\text{sell}} - \text{Fee}_{\text{sell}}) \cdot \text{FX}_{\text{sell}}$.
- **Indexation Factor (`InflationaryRatio`)**:
  - `USD` securities: $\max\left(1.0, \frac{\text{FX}_{\text{sell}}}{\text{FX}_{\text{buy}}}\right)$
  - `ILS` (`.TA`) securities: $\max\left(1.0, \frac{\text{CPI}_{\text{sell}}}{\text{CPI}_{\text{buy}}}\right)$
- **Adjusted Cost (`ILS`)**: $\text{AdjustedCost}_{\text{ILS}} = \text{OriginalCost}_{\text{ILS}} \times \text{InflationaryRatio}$.
- **Real vs. Inflationary Split**:
  1. $\text{Proceeds}_{\text{ILS}} \ge \text{AdjustedCost}_{\text{ILS}}$: Real Taxable Gain $= \text{Proceeds}_{\text{ILS}} - \text{AdjustedCost}_{\text{ILS}}$; Inflationary Exempt Gain $= \text{AdjustedCost}_{\text{ILS}} - \text{OriginalCost}_{\text{ILS}}$.
  2. $\text{OriginalCost}_{\text{ILS}} \le \text{Proceeds}_{\text{ILS}} < \text{AdjustedCost}_{\text{ILS}}$: Real Taxable Gain $= 0.0$; Inflationary Exempt Gain $= \text{Proceeds}_{\text{ILS}} - \text{OriginalCost}_{\text{ILS}}$.
  3. $\text{Proceeds}_{\text{ILS}} < \text{OriginalCost}_{\text{ILS}}$: Real Capital Loss $= \text{Proceeds}_{\text{ILS}} - \text{OriginalCost}_{\text{ILS}}$ (creates offsettable tax shield).

### 3. YTD Tax Shield, Loss Carry-Forward & Portfolio Snapshot
- **Annual Tax Summary (`tax_summary`)**: Aggregates realized real gains and losses per tax year, applying current-year realized losses first and carrying forward unused real capital losses across subsequent tax years (`carry_forward_losses=True`) before applying the 25% tax rate. Tracks `DIVIDEND` distributions at a flat 25% withholding rate (`offset_dividends=False`).
- **Valuation & Tax-Loss Harvesting (`snapshot`)**: Computes open position valuations, FIFO lot breakdowns, average cost basis in `USD` and `ILS`, unrealized gross/net PnL after hypothetical liquidation tax, and unrealized Tax-Loss Harvesting (`tlh_candidates`) opportunities.
- Verified via `python scripts/verify_portfolio_ledger.py`.

---

## 6C. Portfolio Intelligence Phase 3: Vectorized Time-Series, TWR/XIRR & Benchmarking Engine (`core_logic/portfolio_analytics.py`)

`core_logic/portfolio_analytics.py` reconstructs continuous daily dual-currency portfolio trajectories, decomposes return attribution, and benchmarks performance against up to 3 reference assets:

### 1. Vectorized Daily Holdings Matrix & Dual-Currency Valuation (`build_daily_series`)
- Pivots signed share deltas onto a continuous daily calendar grid ($N_{\text{days}} \times M_{\text{tickers}}$) and computes daily market values in both `USD` and `ILS` via vectorized matrix multiplication against `resolve_prices` and `resolve_macro_series`.
- Supports both **Explicit Cash Mode** (when `CASH_DEPOSIT` / `CASH_WITHDRAWAL` records exist) and **Implicit Invested-Equity Mode** (default when only `BUY` / `SELL` / `DIVIDEND` records exist).

### 2. TWR vs. MWR (XIRR) & Attribution Breakdown
- **Time-Weighted Return (TWR)**: Computes daily sub-period returns invariant to external capital injections/withdrawals:
  $$r_t = \frac{V_t - V_{t-1} - CF_t}{V_{t-1} + \max(0.0, CF_t)}, \quad R_{\text{TWR}, t} = \prod_{k=1}^{t} (1 + r_k) - 1$$
- **Money-Weighted Return (MWR / XIRR)**: Solves $\sum_{k} \frac{CF_k}{(1 + \text{XIRR})^{(d_k - d_0)/365.0}} = 0$ via Newton-Raphson root-finding with bracketed bisection fallback, alongside Modified Dietz return.
- **ILS Attribution Breakdown**: Decomposes total `ILS` return into (a) Pure Asset Return at constant FX, (b) `USD/ILS` Exchange Rate Impact, and (c) Brokerage Fee + Realized & Unrealized Israeli Tax Drag.

### 3. Time-Slicing, Rebasing & Cash-Flow Matched What-If Simulator (`analyze_slice`)
- Supports window presets (`1M`, `3M`, `6M`, `YTD`, `1Y`, `3Y`, `5Y`, `ALL`) and custom `[start_date, end_date]` bounds, rebasing both the portfolio and up to 3 benchmark tickers (`SPY`, `QQQ`, `SMH`, `LUMI.TA`, etc.) to `0.0%` (and base `100.0`) at $t_0$ in the active `view_currency` (`USD` or `ILS`).
- **What-If Opportunity Simulator**: Simulates redirecting every historical net external cash flow into each benchmark asset on the exact same transaction dates, reporting hypothetical terminal NAV and excess value created.
- **Institutional Risk KPIs**: Annualized Return, Volatility, Sharpe Ratio ($R_f = 4\%$), Sortino Ratio, Portfolio Beta & Jensen's Alpha vs. primary benchmark, Underwater Drawdown series ($\text{DD}_t$), Maximum Drawdown (MDD), and realized Win-Rate / Payoff Ratio.
- Verified via `python scripts/verify_portfolio_analytics.py`.

---

## 6D. Portfolio Intelligence Phase 4 & Checkpoint 2: REST API Endpoints & Latency SLA (`api/main.py`)

`api/main.py` exposes institutional Portfolio Intelligence endpoints for frontend client applications, providing low-latency execution and strict RFC-8259 JSON serialization:

### 1. Portfolio Summary & Valuation (`GET /api/portfolio/summary`)
- **Query Parameters**:
  - `as_of` (optional): Effective valuation date (`YYYY-MM-DD`). Defaults to current date or latest transaction date.
  - `view_currency` (optional): Base display currency (`USD` or `ILS`, default `USD`).
- **Response Structure**:
  - `positions`: List of open equity positions (`ticker`, `currency`, `quantity`, `avg_cost_native`, `current_price_native`, `market_value_native`, `market_value_view`, `cost_basis_view`, `unrealized_pnl_native`, `unrealized_pnl_view`, `unrealized_pnl_pct`, `weight_pct`, `lots_count`).
  - `lots`: Granular FIFO open tax lot breakdown.
  - `cash`: `USD`, `ILS`, and consolidated `total_view` cash balances.
  - `totals`: Portfolio `nav`, `market_value`, `cost_basis`, `unrealized_pnl`, `unrealized_pnl_pct`, `realized_pnl_ytd`, `est_liquidation_tax_ils`, `unused_loss_shield_ils`, and current `usd_ils` rate.
  - `tax_summary`: Current calendar year Israeli real capital gains tax liabilities, inflationary exemptions, and loss carry-forwards.
  - `tlh_candidates`: Priority-ordered Tax-Loss Harvesting opportunities.

### 2. Transaction Ledger Management & CSV Streaming (`/api/portfolio/transactions`)
- **`GET /api/portfolio/transactions`**: Returns all recorded transactions in chronological order.
- **`POST /api/portfolio/transactions`**:
  - `action: "add"`: Adds a validated transaction (`BUY`, `SELL`, `DIVIDEND`, `CASH_DEPOSIT`, `CASH_WITHDRAWAL`), returns 201 Created with generated `tx_id`.
  - `action: "delete"`: Deletes transaction by `tx_id`, returns 200 OK or 404 Not Found.
  - `action: "clear"`: Clears entire ledger, returning empty state.
  - `action: "import_csv"`: Bulk ingests and validates internal or third-party brokerage CSV payloads (`merge: false` replaces ledger; `merge: true` appends), returning a structured diagnostic report (`status: "success" | "partial"`, `imported_count`, `total_rows`, `skipped_count`, `skipped_rows`) rendered in the frontend `#pf-csv-diagnostic-panel` inspector.
  - *Mutation Invalidation*: All mutations automatically invalidate downstream analytics time-series caches.
- **`GET /api/portfolio/transactions/export`**: Returns standard CSV payload as an attachment (`portfolio_transactions.csv`).

### 3. Vectorized Performance Analytics (`POST /api/portfolio/analytics`)
- **Request Payload**:
  ```json
  {
    "window": "1Y",
    "start_date": "2024-01-01",
    "end_date": "2024-12-31",
    "view_currency": "ILS",
    "benchmarks": ["SPY", "QQQ"],
    "cash_mode": "auto"
  }
  ```
- **Response Structure**:
  - `curves`: Daily time series containing rebased TWR percentages (`port_twr_rebased_pct`), portfolio NAV (`port_nav`), drawdown curve (`port_drawdown_pct`), rebased benchmark curves (`bench_<TICKER>_rebased_pct`), and What-If counterfactual equity accumulation curves (`what_if_<TICKER>_nav`).
  - `kpis`: Full suite of institutional risk & return metrics (`twr_total_pct`, `twr_annualized_pct`, `xirr_annualized_pct`, `annualized_volatility_pct`, `sharpe_ratio`, `sortino_ratio`, `max_drawdown_pct`, `beta_vs_benchmark`, `alpha_annualized_pct`, `win_rate_pct`, `payoff_ratio`).
  - `attribution`: Detailed `ILS` return decomposition (`gross_ils_twr_pct`, `pure_asset_return_pct`, `fx_impact_pct`, `fees_ils`, `realized_tax_ils`, `est_unrealized_tax_ils`, `tax_and_fee_drag_pct`).
  - `what_if`: Opportunity simulator metrics comparing actual terminal NAV against hypothetical benchmark accumulation.

### 4. Latency SLA & RFC-8259 Compliance
- **Warm Cache Latency**: In-memory memoization ensures warm-call latency well below the 100ms threshold (measured ~24ms on `/api/portfolio/summary` and ~59ms on `/api/portfolio/analytics`).
- **Strict NaN/Inf Coercion**: All output models pass through recursive `sanitize_for_json` and `_scalar_float`, eliminating non-standard floating point tokens (`NaN`, `Infinity`, `-Infinity`).
- **Regression Safety**: Existing core routes `/api/calculate` (C-FFI scoring) and `/api/backtest` (rolling backtest) remain fully functional and unregressed.
- **Verification**: Verified via `python scripts/verify_portfolio_api.py`.

---

## 6E. Portfolio Intelligence Phase 5: Interactive Frontend Portfolio Workspace (`frontend/templates/index.html`)

`frontend/templates/index.html` integrates an institutional Portfolio Intelligence workspace alongside the existing Screener & Backtester dashboard using Tailwind CSS, Chart.js, and asynchronous Vanilla JavaScript:

### 1. Top-Level Workspace Navigation & Global Currency Toggle
- **Zero-Reload View Switcher (`switchWorkspace`)**: Toggles seamlessly between **Screener & Backtester** (`#workspace-screener` + `#screener-sidebar`) and **Portfolio Intelligence** (`#workspace-portfolio`) without page reloads.
- **Global Currency Toggle (`setPortfolioCurrency`)**: Switches between `USD $` and `ILS ₪` across both the top header and workspace toolbar, dynamically re-evaluating all KPI cards, multi-asset charts, holdings valuations, and tax liabilities via `/api/portfolio/summary` and `/api/portfolio/analytics`, alongside a live `USD/ILS` exchange rate badge.

### 2. Executive KPI Ribbon & ILS Attribution Strip
- **Total Portfolio NAV Card**: Displays Gross Market Value, Estimated Net Liquidation NAV After Israeli Tax, Invested Holdings Value, and Cash Balance (`USD` + `ILS`).
- **TWR vs. MWR (XIRR) Card**: Side-by-side comparison of Cash-Flow Invariant Time-Weighted Return (Cumulative & Annualized) against Annualized Money-Weighted XIRR and Modified Dietz Return.
- **Unrealized & YTD PnL Card**: Displays Unrealized Gross PnL (`%`), Unrealized Net PnL (after 25% real tax on gains), YTD Realized PnL, and Estimated Liquidation Tax.
- **Israeli Tax Summary Card (`₪ ILS`)**: Displays statutory YTD Realized Tax Liability, Available YTD Tax Shield / Carry-Forward Loss Bucket, Dividend Withholding Tax paid, and Inflationary Exempt Gain (Part E Section 88).
- **ILS Return Attribution & TLH Strip**: Decomposes total `ILS` return into Pure Asset Return, `USD/ILS` Exchange Rate Drift, and Tax/Fee Drag, and highlights actionable Tax-Loss Harvesting (`tlh_candidates`) opportunities.

### 3. Interactive Multi-Benchmark & Time-Slice Charting (Chart.js)
- **Time-Slice Controls**: Preset buttons (`1M`, `3M`, `6M`, `YTD`, `1Y`, `3Y`, `5Y`, `ALL`) plus custom `start_date` and `end_date` pickers.
- **Chart Mode Switcher (`setPortfolioChartMode`)**:
  - **Rebased TWR (`%`)**: Plots portfolio and up to 3 benchmark curves rebased to `0.0%` at $t_0$.
  - **What-If Opportunity Simulator (`NAV`)**: Plots actual portfolio NAV against hypothetical cash-flow matched benchmark NAV trajectories (`what_if_<TICKER>_nav`), with an excess-value comparison banner.
  - **Underwater Drawdown (`%`)**: Plots daily peak-to-trough percentage drawdown across the active window.
- **Dynamic Benchmark Selector**: Toggle preset chips (`SPY`, `QQQ`, `SMH`, `LUMI.TA`) or add custom tickers (up to 3 simultaneous benchmarks). Calls `portfolioChartInstance.destroy()` prior to every re-render to prevent canvas memory leaks.
- **Slice Risk KPI Strip**: Displays Annualized TWR, Annualized Volatility, Sharpe Ratio, Sortino Ratio, Beta vs. Benchmark, Jensen's Alpha, and Maximum Drawdown for the active window.

### 4. Live Holdings Table, Expandable FIFO Tax Lots & Transaction Manager
- **Open Positions Table (`renderHoldingsTable`)**: Displays Ticker, Native Currency (`USD` / `ILS`), Quantity, Avg Cost Basis, Current Price, Market Value, Portfolio Weight (`%`), Unrealized Gross PnL, Estimated Tax (25% real), and Unrealized Net PnL.
- **Expandable FIFO Tax Lots (`togglePositionLots`)**: Clicking any position row expands an inline sub-table detailing each open FIFO lot (`lot_id`, `buy_date`, `quantity`, `buy_price`, base `USD/ILS` or Known CPI drift ratio, Inflationary Gain at 0% tax, Real Gain/Loss in `ILS`, and Estimated 25% Real Tax).
- **Transaction Ledger & CSV Controls**: Form for recording `BUY`, `SELL`, `DIVIDEND`, `CASH_DEPOSIT`, and `CASH_WITHDRAWAL` transactions, one-click `Seed Sample Ledger`, chronological Transaction History table with per-row delete (`deletePortfolioTransaction`), and batch CSV Import/Export/Clear controls.

---

## 6F. Portfolio Intelligence Phase 6: Portfolio Factor DNA (C-FFI), GICS Sector Attribution & Interactive TLH Simulator

Phase 6 bridges the personal multi-currency portfolio ledger with the 80-byte `Stock` C-FFI quantitative scoring kernel (`libsortbuslines.so` / `run_financial_engine`):

### 1. Portfolio Factor DNA & 5-Pillar Radar (`compute_portfolio_quant_dna` & `GET /api/portfolio/quant-dna`)
- **C-FFI Per-Holding Scoring**: Evaluates every open position against the 16-indicator normalized fundamental universe (`normalize_stocks` + `run_financial_engine`), assigning each stock a composite C-FFI `quant_score` $\in [0, 100]$ and health status badge (`STRONG` $\ge 70$, `NEUTRAL` $45\text{–}69$, `WARNING` $< 45$, or `ETF_TASE` for index ETFs such as `SPY`/`QQQ` and `.TA` TASE equities).
- **Strict Covered-Equity Weight Re-Normalization**: Computes `factor_coverage_pct` (share of total invested market value covered by the 16-factor dataset) and re-normalizes factor weights strictly across covered equities ($\sum_{i \in \text{cov}} w_i^{\text{cov}} = 1.0$) so ETFs and `.TA` holdings never dilute factor exposures toward zero.
- **5 Composite Factor Pillars vs. Market Baseline**:
  - **Valuation**: `TrailingPE`, `EV_EBITDA`, `FCFYield`
  - **Growth**: `RevenueGrowth`, `EPSGrowth`
  - **Quality**: `GrossMargin`, `OperatingMargin`, `ROIC`
  - **Safety & Income**: `NetDebt_EBITDA`, `ShareholderYield`, `FiveYDivGrowth`, `PayoutRatio`, `Beta`
  - **Momentum**: `RSI`, `PriceVs200SMA`, `RelativeStrength`
- **Frontend Radar & 16-Indicator Drawer**: Renders a Chart.js `radar` chart (`#portfolio-dna-radar-chart`) comparing the Portfolio's 5 Factor Pillars against the Market Baseline alongside an expandable 16-indicator comparison drawer.

### 2. GICS Sector Allocation & Return Attribution
- Aggregates portfolio weights (`weight_pct` summing to $100.0\%$), market values, unrealized PnL, sector return percentages, and value-weighted sector C-FFI quant scores across GICS sectors (grouping non-SimFin ETFs and TASE securities into `Unclassified / ETF / TASE`).
- Rendered via an interactive Chart.js horizontal bar chart (`#portfolio-sector-chart`) and breakdown table.

### 3. Interactive Israeli Tax-Loss Harvesting (TLH) Simulator (Part E Section 92)
- Identifies all open FIFO tax lots carrying an unrealized real ILS capital loss (`real_loss_ils < 0`).
- Provides per-lot interactive checkboxes allowing the user to simulate harvesting specific loss lots and dynamically computes:
  - **Selected Real Capital Loss (`₪ ILS`)**
  - **Immediate 25% Israeli Tax Shield Generated (`₪ ILS`)**: $0.25 \times |\text{real\_loss\_ils}|$
  - **YTD Capital Gains Tax Liability Wiped Out (`₪ ILS` & `%`)**: $\min(\text{Immediate Shield}, \text{YTD Realized Tax Due})$
  - **Carry-Forward Shield to Future Years (`₪ ILS`)**: $\max(0.0, \text{Immediate Shield} - \text{YTD Tax Wiped Out})$

---

## 6G. Bilingual (`EN` / `HE`) Localization, Contextual Info (`i`) Tooltips & Two-Screen Progressive Disclosure

To make institutional-grade quantitative analytics and Israeli tax mechanics accessible without overwhelming the user, `frontend/templates/index.html` implements:

### 1. Bilingual Localization (`EN` | `עב`) & Dynamic RTL/LTR Layout (`setAppLanguage`, `applyLayoutDirection`)
- **Top-Header Language Switcher (`#lang-btn-en` | `#lang-btn-he`)**: Persisted in `localStorage` (`aq_lang`), switching instantaneously between English (`en`) and Hebrew (`he`) via the structured `I18N = { en: {...}, he: {...} }` dictionary (`data-i18n` and `data-i18n-placeholder` bindings).
- **RTL Layout with Strict LTR Financial Figure Isolation**: When Hebrew (`he`) is active, the root document switches to `dir="rtl"` (`Heebo` / `Inter` typography) and docks the Screener sidebar to the right, while strictly enforcing `dir="ltr"` (`unicode-bidi: isolate; font-variant-numeric: tabular-nums;`) on all currency amounts, signed percentages, dates, ticker symbols, and Chart.js canvases so financial numbers never invert.

### 2. Contextual Info (`i`) Tooltips Dictionary (`TOOLTIPS`, `#global-info-popover`)
- Every technical, quantitative, and Israeli tax metric across both workspaces features an interactive `<span class="info-tooltip" data-term="...">i</span>` badge.
- Hovering, focusing, or tapping any `i` badge opens a viewport-aware popover (`#global-info-popover`) with a concise 2–3 line explanation in the active language (`EN` or `HE`), covering:
  - **Portfolio & Return Metrics**: `NAV`, `Net Liquidation NAV`, `TWR`, `MWR / XIRR`, `What-If Opportunity Simulator`, `Underwater Drawdown`, `MDD`, `Sharpe Ratio`, `Sortino Ratio`, `Beta`, `Jensen's Alpha`, `Annualized Volatility`.
  - **Israeli Tax & Dual-Currency Terms**: `FIFO Tax Lots`, `Original vs. Adjusted Cost Basis`, `Inflationary Exempt Gain`, `Real Taxable Gain`, `YTD Tax Shield`, `Tax-Loss Harvesting (TLH)`, `FX Attribution & Tax Drag`.
  - **Screener & Quant DNA Terms**: `C-FFI Quant Score`, `Factor Coverage %`, `Goldilocks Normalization`, `Sector-Neutral Sigmoid Z-Score`, the 5 Factor Pillars (`Valuation`, `Growth`, `Quality`, `Safety & Income`, `Momentum`), and all 16 quantitative indicators.

### 3. Two-Screen Progressive Disclosure Layout (`setPortfolioViewTier`)
- **Screen 1 — Standard Overview (`#pf-view-standard`, Default)**:
  - Executive KPI Ribbon (`NAV`, `Net Liquidation NAV`, `TWR` vs `XIRR`, `Unrealized Gross & Net PnL`, `Israeli Tax Summary`).
  - Primary Performance Chart (`TWR` curve, Benchmark comparison chips, Time-Slice presets).
  - Live Holdings Table with one-click expandable inline `FIFO` tax lot breakdowns.
  - Record Transaction & Fault-Tolerant Brokerage CSV Import/Export panel.
- **Screen 2 — Advanced Quant & Tax Lab (`#pf-view-advanced`)**:
  - Full 7-Metric Institutional Risk KPI Strip (`Annualized Return`, `Annualized Volatility`, `Sharpe Ratio`, `Sortino Ratio`, `Beta`, `Jensen's Alpha`, `Max Drawdown`) and `What-If Opportunity Simulator` cards.
  - `ILS (₪)` Return Attribution Bar (`Pure Asset Return` + `FX Impact` - `Tax & Fee Drag` = `Net After-Tax TWR`).
  - Portfolio Factor DNA Radar (`C-FFI` 5 Pillars + expandable 16-Indicator drawer) & GICS Sector Allocation/Attribution chart.
  - Interactive Israeli Tax-Loss Harvesting (`TLH`) Simulator (`Part E Section 92`) and full chronological Transaction History Ledger.

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

### Step 2 — Compile the C Shared Library & Validate Alignment
```bash
python scripts/compile_c_engine.py
```
*Compiles `financial_engine.c` into `libsortbuslines.so` (using GCC / WSL fallback) and asserts FFI `Stock` 80-byte memory alignment.*

### Step 3 — Run Strategy Rolling Backtests
```bash
python scripts/run_rolling_backtest.py --strategy momentum --window 3y
```
*Executes point-in-time dynamic portfolio rebalancing simulation without look-ahead bias.*

### Step 4 — Run the Development Server
```bash
python api/main.py
```
*Access dashboard at `http://127.0.0.1:5000`.*

### Step 5 — Production Cloud Deployment (e.g. Render)
- **Build Command**: `python scripts/compile_c_engine.py && pip install -r requirements.txt`
- **Start Command**: `gunicorn api.main:app`
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
