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
- **Portfolio Intelligence Phase 2 — Dual-Currency FIFO Ledger, 3-Bucket Capital Accounting & Israeli Tax Engine (`core_logic/portfolio_ledger.py`)**: Real-time transaction persistence (`data/portfolio_transactions.parquet`), fractional FIFO queue state machine, non-negative 3-Bucket Capital Accounting (`Available Cash`, `Total Invested Capital`, `Withdrawn Capital`, `Total Wealth Created`, `ROI %`), and exact Israeli capital gains tax decomposition (25% on real gains, inflationary exemptions, YTD tax shielding, and loss carry-forward).
- **Portfolio Intelligence Phase 3 — Vectorized Time-Series & Attribution Engine (`core_logic/portfolio_analytics.py`)**: Continuous daily holdings matrix ($N_{\text{days}} \times M_{\text{tickers}}$), cash-flow invariant Time-Weighted Return (TWR), exact Money-Weighted Return (MWR / XIRR via Newton-Raphson), ILS FX & tax attribution, and invested-capital driven counterfactual benchmark opportunity simulation.
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
- **Delisted / OTC Bankruptcy Ticker Fallback (`_is_otc_q_ticker`)**: Automatically detects 5-letter US OTC bankruptcy tickers ending in `'Q'` (e.g., `LAZRQ`) and queries the 4-letter base symbol (`LAZR`) across Tier 2, Tier 1, and Tier 3 when historical coverage is missing across the requested start date, splicing and forward-filling historical prices seamlessly into the portfolio cache.
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
  - **Zero-Price Corporate Action Split Folding (Pass A)**: Brokerage exports (e.g. Yahoo Finance `portfolio.csv`) frequently export stock splits on open positions as synthetic `BUY` rows with `Quantity > 0` and `Purchase Price = 0.0` (e.g. `XLK` on split date). The importer detects these rows chronologically, computes the exact split factor $S = (Q_{\text{open}} + Q_{\text{split}}) / Q_{\text{open}}$, and folds the corporate action into open `BUY` lots ($Q_{\text{adj}} = Q \cdot S$, $P_{\text{adj}} = P / S$), preserving total cost ($Q \cdot P$) and fee invariance without creating false `$0` rows or counting as skipped errors.
  - **Automatic Pre-Split Basis Alignment (Pass B, `align_pre_split_transactions`)**: When importing or analyzing closed historical trades that occurred prior to a stock split (e.g. `NVDA` bought in 2023 at `$402-$499` and sold in Jan 2024 at `$521.00`), transactions are compared against split-adjusted market prices from `resolve_prices`. Matching canonical split ratios ($\pm 22\%$ tolerance for $S \in \{2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 50\}$) scales lots to the split-adjusted basis ($Q \cdot S$, $P / S$), eliminating artificial -90% holding-period valuation dips and phantom TWR jumps on sale dates while preserving exact dollar cost, 3-Bucket cash balances, and Israeli Section 88 tax liabilities.
  - **Fractional Rounding Tolerance & Row-by-Row Diagnostics (Pass C)**: Clamps small broker rounding discrepancies ($\le 0.01$ shares, such as selling `73.87` shares against `73.867` open shares) to the open lot balance, and skips incomplete watchlist rows, zero-price rows, or genuine oversells ($> 0.01$ shares) while returning a full diagnostic report (`imported_count`, `total_rows`, `skipped_count`, `skipped_rows` with exact 1-based line numbers and missing/invalid field details).

### 2. 3-Bucket Capital Accounting Engine (`capital_buckets`)
To guarantee accurate return calculations and prevent negative cash artifacts when users record `BUY` transactions without prior manual `CASH_DEPOSIT` entries, the ledger maintains three non-negative, chronologically tracked capital buckets:
- **Available Cash (`available_cash`, יתרת מזומן פנויה להשקעה $\ge 0.0$)**: Tracks liquid funds inside the portfolio. Incremented by net `SELL` proceeds ($Q \cdot P_{\text{sell}} - \text{fee}$), net `DIVIDEND` distributions ($75\%$ after Israeli withholding), and explicit `CASH_DEPOSIT`. When a `BUY` order occurs (total cost $C = Q \cdot P_{\text{buy}} + \text{fee}$), the ledger draws from available cash only up to the available balance: $\text{used\_cash} = \min(\text{available\_cash}, C)$. The balance is clamped at zero and never turns negative.
- **Total Invested Capital (`total_invested_capital`, סה"כ כסף מושקע מהכיס)**: Tracks cumulative external out-of-pocket capital introduced into the account. Incremented on explicit `CASH_DEPOSIT` and whenever a `BUY` order requires external funds exceeding available cash (the uncovered shortfall $C - \text{used\_cash}$).
- **Withdrawn Capital (`withdrawn_capital`, כסף ממומש שנמשך לשימוש אישי)**: Tracks cash permanently extracted from the investment account via explicit `CASH_WITHDRAWAL` transactions, reducing `available_cash` (clamped at zero).
- **Consolidated Valuation & Return on Invested Capital (ROI %)**:
  $$\text{Total Portfolio NAV}_t = \text{Holdings Market Value}_t + \text{Available Cash}_t$$
  $$\text{Total Wealth Created} = (\text{Total Portfolio NAV}_t + \text{Withdrawn Capital}_t) - \text{Total Invested Capital}_t$$
  $$\text{ROI \%} = \frac{\text{Total Wealth Created}}{\text{Total Invested Capital}_t} \times 100$$

### 3. Israeli Real Capital Gains Tax Decomposition (`compute_israeli_lot_tax`)
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

### 4. YTD Tax Shield, Loss Carry-Forward & Portfolio Snapshot
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

### 3. Time-Slicing, Rebasing & Invested-Capital What-If Simulator (`analyze_slice`)
- Supports window presets (`1M`, `3M`, `6M`, `YTD`, `1Y`, `3Y`, `5Y`, `ALL`) and custom `[start_date, end_date]` bounds, rebasing both the portfolio and up to 3 benchmark tickers (`SPY`, `QQQ`, `SMH`, `LUMI.TA`, etc.) to `0.0%` (and base `100.0`) at $t_0$ in the active `view_currency` (`USD` or `ILS`).
- **Invested-Capital Driven What-If Opportunity Simulator**: Accurately tracks benchmark unit accumulation driven strictly by external out-of-pocket capital inflows (`external_inflow`) and explicit cash withdrawals (`withdrawal`):
  $$\Delta Q_{\text{bench}, t} = +\frac{\text{external\_inflow}_t}{P_{\text{bench}, t}} - \min\left(Q_{\text{bench}, t-1}, \, \frac{\text{withdrawal}_t}{P_{\text{bench}, t}}\right)$$
  Crucially, internal portfolio stock sales (`SELL`) do *not* trigger benchmark share liquidations; sales represent reallocations of existing invested equity, preserving counterfactual benchmark opportunity simulation across both deposits and withdrawals.
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

## 6G. Institutional Obsidian Fintech Design System, Bilingual (`EN` / `HE`) Logical CSS Symmetry & Two-Screen Progressive Disclosure

To make institutional-grade quantitative analytics and Israeli tax mechanics accessible without overwhelming the user, `frontend/templates/index.html` implements a single-file zero-build Obsidian Fintech design system:

### 1. Obsidian Dark & Institutional Light Mode Surface Hierarchy (`tailwind.config`, CSS Variables & `setTheme`)
- **Dual-Theme CSS Variable Architecture**: Maps all Tailwind semantic tokens (`surface`, `surface-container-*`, `on-surface`, `on-surface-variant`, `primary`, `secondary`, `tertiary`, `outline-variant`) to RGB CSS custom properties (`:root` for Obsidian Dark `#07090E` / `#131722` and `html.light` for Institutional Light `#F8FAFC` / `#FFFFFF`) supporting native Tailwind alpha modifiers (`bg-surface/80`, `bg-primary/15`).
- **Header Theme Switcher (`#theme-toggle-btn`, `toggleTheme`, `setTheme`)**: Persisted in `localStorage` (`aq_theme`) with `prefers-color-scheme` fallback. Toggling themes dynamically re-renders all active Chart.js instances (`getChartThemeColors()`) with theme-matched grid lines, tick labels, and tooltip cards.
- **Reusable `.aq-card` & `.aq-btn-primary` Utilities**: Every KPI card, chart panel, modal dialog, and table container uses `.aq-card` (adapting between dark obsidian glassmorphism and crisp light-mode elevated slate surfaces), while primary CTAs use `.aq-btn-primary` with gradient illumination and subtle hover elevation.
- **Expanded Viewport & Typography Scale**: Expands `#workspace-portfolio` to `max-w-[1560px]` and `#workspace-screener` to `max-w-[1360px]` inside an `overflow-x-hidden` viewport (`h-16` top header, `p-6 lg:p-8` main viewport), scaling main KPI figures to `text-2xl xl:text-3xl font-extrabold` and table cells to `py-3.5 px-4 text-sm`.
- **Authoritative Top-Header Currency Toggle**: Centralizes the `USD $` / `ILS ₪` switcher inside `#top-header` (`#portfolio-header-controls`) alongside the live `USD/ILS` exchange rate badge (`#header-fx-badge`), eliminating duplicate workspace currency controls.

### 2. Bilingual Localization (`EN` | `עב`) & Logical RTL/LTR Symmetry (`setAppLanguage`, `applyLayoutDirection`)
- **Top-Header Language Switcher (`#lang-btn-en` | `#lang-btn-he`)**: Persisted in `localStorage` (`aq_lang`), switching instantaneously between English (`en`) and Hebrew (`he`) via the structured `I18N = { en: {...}, he: {...} }` dictionary (`data-i18n` and `data-i18n-placeholder` bindings).
- **Logical CSS Utilities & Strict LTR Financial Figure Isolation**: Uses CSS logical properties (`text-start`, `text-end`, `ps-*`, `pe-*`, `ms-*`, `end-*`) across all tables, headers, modals, and cards so layout geometry mirrors automatically when `dir="rtl"` is active, while strictly enforcing `.num-ltr` (`direction: ltr; unicode-bidi: isolate; font-variant-numeric: tabular-nums;`) on all currency amounts, signed percentages, dates, ticker symbols, and Chart.js canvases.

### 3. Contextual Info (`i`) Tooltips Dictionary (`TOOLTIPS`, `#global-info-popover`)
- Every technical, quantitative, and Israeli tax metric across both workspaces features an interactive `<span class="info-tooltip" data-term="...">i</span>` badge.
- Hovering, focusing, or tapping any `i` badge opens a viewport-aware popover (`#global-info-popover`) with a concise 2–3 line explanation in the active language (`EN` or `HE`), covering:
  - **Portfolio & Return Metrics**: `NAV`, `Net Liquidation NAV`, `TWR`, `MWR / XIRR`, `What-If Opportunity Simulator`, `Underwater Drawdown`, `MDD`, `Sharpe Ratio`, `Sortino Ratio`, `Beta`, `Jensen's Alpha`, `Annualized Volatility`.
  - **Israeli Tax & Dual-Currency Terms**: `FIFO Tax Lots`, `Original vs. Adjusted Cost Basis`, `Inflationary Exempt Gain`, `Real Taxable Gain`, `YTD Tax Shield`, `Tax-Loss Harvesting (TLH)`, `FX Attribution & Tax Drag`.
  - **Screener & Quant DNA Terms**: `C-FFI Quant Score`, `Factor Coverage %`, `Goldilocks Normalization`, `Sector-Neutral Sigmoid Z-Score`, the 5 Factor Pillars (`Valuation`, `Growth`, `Quality`, `Safety & Income`, `Momentum`), and all 16 quantitative indicators.

### 4. Strict Two-Screen DOM Separation, Full-Width Holdings & Centered Command Modal (`setPortfolioViewTier`)
To provide both an accessible high-level overview and deep quant research capabilities without visual clutter, the workspace implements strict DOM separation between two mutually exclusive screens (`#pf-view-standard` and `#pf-view-advanced`):
- **Screen 1 — Standard Overview (`#pf-view-standard`, Default)**:
  - **Executive KPI Ribbon**: Card 1 (Total Portfolio NAV & Net Liquidation NAV), Card 2 (3-Bucket Capital Accounting: Total Invested Capital, Available Cash, Withdrawn Capital, Total Wealth Created, and ROI %), Card 3 (TWR vs XIRR / Modified Dietz), and Card 4 (Unrealized Gross/Net PnL & YTD Realized PnL).
  - **Primary Performance Chart (`#portfolio-main-chart`)**: Rebased TWR (%) performance canvas with dynamic benchmark comparison chips and time-slice selectors (`1M`, `3M`, `6M`, `YTD`, `1Y`, `3Y`, `5Y`, `ALL`).
  - **Full-Width Live Holdings Table (`w-full`)**: 11-column open positions table spanning 100% of the workspace width with one-click expandable inline FIFO tax lot breakdowns.
  - **Centered Transaction & Batch CSV Command Modal (`#pf-tx-modal` & `#pf-tx-dialog`)**: Triggered from the workspace header CTA (`+ Record Transaction / CSV`, `openTxDrawer()`), opening a centered glassmorphic modal (`z-[9950]`) that supports backdrop-click and `Escape` key dismissal without any off-screen RTL/LTR drawer overflow. Houses quick-action transaction entry (`Buy`, `Sell`, `Dividend`, `Deposit`, `Withdraw Capital`) and schema-adaptive CSV import/export/clear controls with row-by-row diagnostic inspection (`#pf-csv-diagnostic-panel`). Automatically closes and triggers a success toast upon transaction submission.
- **Screen 2 — Advanced Quant & Tax Lab (`#pf-view-advanced`)**:
  - **Deep-Dive Multi-Mode Chart (`#portfolio-adv-chart`)**: Interactive canvas toggling between Rebased TWR (%), What-If Simulator NAV, and Underwater Drawdown (%), accompanied by the What-If Opportunity comparison banner (`#pf-whatif-banner`) and 7-metric institutional risk KPI strip (`#pf-slice-risk-strip`).
  - **Statutory Israeli Tax Card & Return Attribution Bar**: Full Section 88 indexation, YTD tax shield, loss carry-forward, and visual ILS return attribution bar (`#pf-attr-bar`).
  - **Portfolio Factor DNA Radar (`C-FFI` Engine)**: 5-Pillar radar canvas (`#portfolio-dna-radar-chart`), factor pillar chips, expandable 16-indicator drawer, and GICS Sector Allocation/Attribution chart (`#portfolio-sector-chart`).
  - **Interactive Israeli TLH Simulator**: Section 92 loss lot selector calculating immediate tax shields and loss carry-forward.
  - **Chronological Transaction Ledger**: Full historical ledger table with per-row deletion.
- **Global Toast Stack, Confirmation Modal & Shimmer Skeletons**:
  - **Non-Blocking Toasts (`showToast`, `#global-toast-container`)**: Replaces blocking browser `alert()` dialogs with auto-dismissing (`4000ms`) color-coded notifications (`error`, `success`, `info`) anchored at `bottom-5 end-5 z-[9990]`.
  - **Destructive Action Confirmation Modal (`openConfirmModal`, `#global-confirm-modal`)**: Guards `clearPortfolioLedger()` and `seedDemoPortfolio()` behind an accessible confirmation dialog.
  - **Shimmer Skeleton Loaders (`renderTableSkeleton`) & Material Symbols**: Renders 5 animated shimmer skeleton rows (`animate-pulse bg-surface-container-high/60`) inside `#leaderboard-body` and `#portfolio-holdings-body` during async fetches, and uses `material-symbols-outlined` icons (`bolt`, `monitoring`, and CSS spinners) across action buttons.
- **Dual Chart.js Lifecycle Management**: Dedicated chart instances (`portfolioChartInstance` and `portfolioAdvChartInstance`) synchronized via `renderPortfolioChart` with automatic `requestAnimationFrame` resizing upon view toggling, completely eliminating canvas layout thrashing and ghost rendering.

---

## 6H. V3.3.1 Audit Remediation: Performance, Point-in-Time Backtest & Portfolio Math

### 1. Screener & C-FFI (`core_logic/financial_module.py`, `api/main.py`)
- **Tech cache anchoring**: `ensure_tech_cache()` falls back to `max(Date) - 370d` when the wall clock is beyond the offline Parquet horizon, so `_TECH_DATA_CACHE` hydrates once (TTL 30m). It also precomputes `latest_prices`, `adtv_series` and the full `tech_map`; `compute_live_technical_indicators()` is an O(1) lookup after the first call. The Gatekeeper ($5 price / $5M ADTV) is no longer silently bypassed.
- **No network in `/api/calculate`**: the synchronous Tiingo enrichment (whose output was never serialized) was removed (Pure Offline Mandate).
- **Vectorized FFI packing**: `STOCK_STRUCT_DTYPE = [('ticker','S8'),('indicators','<f4',(16,)),('final_score','<f8')]` (asserted `itemsize == 80`) is passed to C via `ndarray.ctypes.data_as`. The loader tries `libsortbuslines.dll` (Windows) and `.so` (Linux/WSL); without a library a vectorized NumPy fallback on the same buffer is used. `c_kernel/` sources are untouched.
- `normalize_stocks()` accepts DataFrames and builds each frame once; `_safe_float()` has a numeric fast path.

### 2. Point-in-Time Backtester (`core_logic/backtest_engine.py`, `data_pipeline/simfin_etl.py`)
- `simfin_etl.py` now persists `Publish_Date` (fallback `Report Date + 45d`). The backtester uses `Effective_Date = Publish_Date` (or the raw SimFin `Publish Date`, else `Quarter_End_Date + 45d`) so statements are only visible once published.
- **Technicals are recomputed at each anchor t_k** from prices `<= t_k` (`compute_vectorized_tech_series`), so no post-t_k prices leak into RSI / SMA / RelativeStrength / Beta / gatekeeper price.
- Anchors form a true calendar schedule every `freq_months` (3M / 6M) instead of every fiscal month-end. The window is anchored to `min(now, max price date, max SPY date)`, SPY is clipped to the price calendar, and holiday gaps are `ffill(limit=5)`-ed so the curve no longer flatlines. Windows `1y|3y|5y|10y` are supported; the start is clamped to the first date with >= 100 published filings.
- Fundamentals + price matrix are cached in-memory (invalidated by file mtime); the daily loop is a per-period vectorized NumPy slice; the per-anchor `gc.collect()` is removed.

### 3. Portfolio Ledger & Analytics (`core_logic/portfolio_ledger.py`, `core_logic/portfolio_analytics.py`)
- **Cross-currency cash sweep** (`sweep_available_cash`): shared by `PortfolioLedger._replay` and `build_daily_series`. A BUY / WITHDRAWAL first consumes same-currency cash, then the other currency at `fx_t`, and only the remainder is external capital. This removes phantom inflows and NAV / invested-capital double counting.
- **`cash_mode`**: `auto` -> `explicit` if any `CASH_DEPOSIT` / `CASH_WITHDRAWAL` exists, else `implicit`. `implicit` values the invested equity only; un-reinvested SELL proceeds and dividends are treated as immediate withdrawals. The resolved mode is returned as `cash_mode` and drives TWR, the What-If simulator, XIRR and Modified Dietz consistently (`mode_*` series).
- **Constant-FX attribution** is slice-invariant (`_compute_const_fx_twr`): daily FX revaluation of the USD bucket is stripped, so it no longer depends on the ledger-inception FX.
- **What-If** results now include `benchmark_xirr_pct` computed on the matched benchmark cash flows.
- **Risk KPIs** use trading-day returns (weekend zero padding excluded) scaled by sqrt(252); Beta uses active-market days.
- **UI**: FIFO lot table reads `fx_now` / `cpi_now` and no longer flips the sign of `real_loss_ils`.

---

## 6I. V3.3.2 Production Render 512MB RAM OOM & C-FFI Safety Hardening

To eliminate silent container terminations (`SIGKILL` cgroup OOM / `SIGSEGV`) on 512MB RAM cloud instances (e.g., Render free/starter tiers running `gunicorn --timeout 120 main:app` with `WEB_CONCURRENCY=1`), V3.3.2 establishes strict memory pool boundaries, re-layouts historical Parquet disk storage, decouples empty-portfolio execution paths, and hardens C-FFI memory alignment:

### 1. 32-Row-Group Date-Sorted Parquet Re-Layout (`data_pipeline/compress_prices_to_parquet.py`)
- **Root Cause**: `data/raw_simfin/us-shareprices-daily.parquet` (6,225,077 rows) previously existed as a single monolithic row group sorted by `Ticker`, with `Ticker` stored as a 5,115-category `CategoricalDtype`. Any PyArrow pushdown query with `Date >= min_date` was forced to decompress the entire 6.22M-row table into RAM simultaneously, triggering a ~300MB C++ decompression spike that exceeded the 512MB cgroup limit.
- **Architectural Solution**: Rebuilt the Parquet dataset sorted by `(Date, Ticker)` with plain `str` `Ticker` and `float32` prices/volumes (`Close`, `Adj. Close`, `Volume`), partitioned into **32 chronological row groups (`row_group_size=200_000`)**.
- **Impact**: PyArrow pushdown filters (`[('Date', '>=', min_date)]`) now evaluate row-group footer statistics on disk and skip ~90% of row groups without decompressing them into RAM, slashing C++ Arrow heap usage from >300MB down to ~30MB.

### 2. Zero-RAM Parquet Footer Horizon Discovery, Row-Group Streaming Matrix Builder & Linux OS Heap Arena Reclamation (`core_logic/financial_module.py`, `core_logic/backtest_engine.py`)
- **`discover_parquet_max_date()`**: Reads the terminal date of `us-shareprices-daily.parquet` directly from PyArrow `ParquetFile` row-group footer metadata in `< 1ms` and `0 MB` RAM, replacing legacy fallback routines that read the entire 6.22M-row `Date` column into Pandas when the wall clock exceeded the dataset horizon (`2025-08-14`).
- **Zero-Pivot Row-Group Matrix Streamer (`load_parquet_price_matrix`)**: Both `ensure_tech_cache()` and `_get_backtest_datasets()` stream `us-shareprices-daily.parquet` one 200k-row group at a time (~3 MB per batch), map `Ticker` symbols to integer column indices purely in C++ via `pyarrow.compute.index_in` without allocating millions of Python `str` objects, and populate pre-allocated `float32` `(Date x Ticker)` NumPy matrices directly without calling `DataFrame.pivot()`. This eliminates the ~340 MB transient `pd.read_parquet` + `pivot()` memory spike during `/api/backtest`.
- **Pre-Baked `Publish_Date` & Pure NumPy Rolling Technicals**: `data/historical_fundamental_data.csv` persists `Publish_Date` directly on disk so `_get_backtest_datasets()` never loads `us-income-ttm.csv` at runtime, while `compute_vectorized_tech_series()` computes daily returns, RSI, and Beta covariance directly on `float32` NumPy arrays (zero intermediate 2D DataFrame copies per rebalance anchor) and passes normalized `float32` DataFrames directly into `run_financial_engine()`.
- **System Memory Allocator & glibc Heap Trimming**: Configures `pa.set_memory_pool(pa.system_memory_pool())` and single-threaded Arrow I/O (`pa.set_cpu_count(1)`) on import. Added `_release_arrow_memory(run_gc=False)` which invokes `pa.default_memory_pool().release_unused()` and on Linux (`libc.so.6`) issues `libc.malloc_trim(0)`. This forces glibc to return freed heap arenas directly to the Linux OS kernel rather than hoarding virtual memory inside the worker process.
- **1D Tech Cache & Immediate 2D Pivot Matrix Disposal (`ensure_tech_cache`)**: Pushes down both the fundamental `Ticker` whitelist (~1,866 equities) and the 370-day `Date` filter to PyArrow, precomputes 1D `latest_prices`, `adtv_series`, and `tech_map`, and immediately discards the 2D `full_price_df` and `full_volume_df` matrices (`_TECH_DATA_CACHE['price_df'] = None`), cutting `_TECH_DATA_CACHE` retention to `< 1 MB`.

### 3. Cold Start Elimination: Lazy Empty-Portfolio Quant DNA & Fast Route Bypasses (`core_logic/portfolio_analytics.py`)
- **Empty-Portfolio Fast Path**: In `compute_portfolio_quant_dna()`, the engine now verifies `if tx_df.empty:` *before* hydrating `_get_universe_quant_baseline()` and `ensure_tech_cache()`. Initial page loads for empty portfolios return an empty factor DNA payload in `< 1ms` and `0 MB` additional RAM instead of hydrating all 1,866 universe stocks into memory.
- **Bounded FIFO Memoization**: Caps `_PRICE_RESOLVER_CACHE` (16 entries), `_cached_daily_series` (4 entries), `_cached_slices` (8 entries), and `_cached_quant_dna` (4 entries), preventing unbounded memory accumulation across long-lived Gunicorn worker sessions.

### 4. Isolated Local-Only CSV Upload Pass B & Offline Fallback Valuation (`core_logic/portfolio_ledger.py`)
- **Local-Only Pass B**: In `PortfolioLedger.import_csv()`, corporate action Pass B (`align_pre_split_transactions`) now calls `resolve_prices(..., local_only=True)` and invokes `_release_arrow_memory()`. External Yahoo Finance network requests are completely skipped during upload, preventing Gunicorn 120s worker timeout drops.
- **Fault-Tolerant Snapshot Valuation**: In `snapshot()`, `allow_partial=True` is passed to `resolve_prices` with an automatic fallback to the lot's purchase price (`state.open_lots[t][-1].price`) if external pricing is unreachable, guaranteeing zero 500 errors in offline or sandboxed execution environments.

### 5. Sequential Frontend Request Scheduling (`frontend/templates/index.html`)
- **Collision-Free Route Lifecycle**: In `refreshPortfolioWorkspace()`, concurrent background requests on initial page load have been restructured into a clean sequential pipeline:
  $$\text{fetchPortfolioTransactions()} \longrightarrow \text{fetchPortfolioSummary()} \longrightarrow \text{fetchPortfolioAnalytics()}$$
  This prevents multiple heavy endpoints from hydrating parallel caches concurrently on a single-worker Gunicorn process (`WEB_CONCURRENCY=1`).

### 6. C-FFI `Stock` Struct Null-Termination & Buffer Lifetime Safety (`run_financial_engine`)
- **Strict 7-Byte ASCII + `\0` Invariant**: Encodes tickers as ASCII and truncates to at most 7 bytes (`encode('ascii', errors='ignore')[:7]`) inside the zero-initialized 8-byte `S8` field (`char ticker[8]`). This guarantees that byte index 7 is always `\0` even for 13-character delisted symbols (e.g. `WMGI_DELISTED`), preventing out-of-bounds `strcmp` reads in `c_kernel/financial_engine.c`. Full ticker strings are restored via a dictionary map on output.
- **Contiguous Buffer & Lifetime Reference**: Enforces `np.ascontiguousarray(c_buf, dtype=STOCK_STRUCT_DTYPE)`, sanitizes `NaN`/`Inf` in indicator matrices and weights prior to C execution, and retains explicit Python object references to `c_buf` and `c_weights` across `lib.compute_scores` and `lib.sort_stocks_by_score` to prevent premature garbage collection.

### 7. Verified Single-Worker RSS Memory Profile (512MB RAM Budget)

The entire system was profiled under an isolated single-worker environment executing all production routes sequentially:

| Step / Action Executed | Process RSS | Tracemalloc Current / Peak | HTTP Status | Safety Margin vs 512MB Limit |
| :--- | :--- | :--- | :--- | :--- |
| **Gunicorn Worker Boot (`main:app`)** | **139.8 MB** | 19.1 MB / 19.1 MB | — | **+372.2 MB (73% Free)** |
| **Empty Portfolio Initial Load** (`tx` + `summary` + `analytics`) | **167.4 MB** | 42.3 MB / 42.3 MB | `200 / 200 / 200` | **+344.6 MB (67% Free)** |
| **CSV Ledger Import** (`POST /api/portfolio/transactions` `import_csv`) | **186.4 MB** | 51.0 MB / 51.0 MB | `200` | **+325.6 MB (64% Free)** |
| **Populated Portfolio Summary + Analytics + Quant DNA** | **240.9 MB** | 64.2 MB / 88.5 MB | `200 / 200` | **+271.1 MB (53% Free)** |
| **Stock Screener Execution** (`POST /api/calculate`, 1,866 equities) | **240.9 MB** | 67.6 MB / 90.6 MB | `200` | **+271.1 MB (53% Free)** |
| **Rolling 3-Year Backtest** (`POST /api/backtest`, 12 rebalance anchors) | **243.3 MB** | 86.2 MB / 103.6 MB | `200` | **+268.7 MB (52% Free)** |

All endpoints execute comfortably within the 512MB container constraint with zero memory leaks, zero SIGSEGV crashes, and sub-second warm-cache latency.

---

## 7. Developer Onboarding & Extensibility Guide

### System Prerequisites
- **Python**: $\ge 3.9$
- **C Compiler**: GCC $\ge 11$ with `make`
- **Operating System**: Linux / WSL (Windows Subsystem for Linux) / Windows Native

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

### Step 3 — Run Automated Verification Suites
```bash
python scripts/verify_price_resolver.py
python scripts/verify_portfolio_ledger.py
python scripts/verify_portfolio_analytics.py
python scripts/verify_portfolio_api.py
```
*Validates multi-currency pricing, FIFO lot tracking, Israeli tax calculations, TWR/MWR analytics, and REST API schemas.*

### Step 4 — Run Strategy Rolling Backtests
```bash
python scripts/run_rolling_backtest.py --strategy momentum --window 3y
```
*Executes point-in-time dynamic portfolio rebalancing simulation without look-ahead bias.*

### Step 5 — Run the Development Server
```bash
python api/main.py
```
*Access dashboard at `http://127.0.0.1:5000`.*

### Step 6 — Production Cloud Deployment (e.g. Render)
- **Build Command**: `python scripts/compile_c_engine.py && pip install -r requirements.txt`
- **Start Command**: `gunicorn --timeout 120 main:app`
- **Environment Variables**:
  - `WEB_CONCURRENCY=1` (Recommended for 512MB RAM containers)
  - `TIINGO_API_KEY` (Optional for external live quote enrichment)

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
