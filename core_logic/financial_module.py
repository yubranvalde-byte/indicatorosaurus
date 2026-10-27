import ctypes
import os
import math
import statistics
import csv
import gc
import time
import requests
import pandas as pd
import numpy as np

# ─── Constants ────────────────────────────────────────────────────────────────

NUM_INDICATORS = 16

# Strict column order matching fetch_static_data.py CSV_HEADER (positions 1-16)
INDICATOR_COLUMNS = [
    'TrailingPE',        # indicators[0]
    'EV_EBITDA',         # indicators[1] (replaced PEGRatio)
    'FCFYield',          # indicators[2]
    'RevenueGrowth',     # indicators[3]
    'EPSGrowth',         # indicators[4]
    'GrossMargin',       # indicators[5]
    'OperatingMargin',   # indicators[6]
    'ROIC',              # indicators[7]
    'NetDebt_EBITDA',    # indicators[8]
    'ShareholderYield',  # indicators[9] (replaced DividendYield)
    'FiveYDivGrowth',    # indicators[10]
    'PayoutRatio',       # indicators[11]
    'Beta',              # indicators[12]
    'RSI',               # indicators[13]
    'PriceVs200SMA',     # indicators[14]
    'RelativeStrength',  # indicators[15]
]

# Directionality flags: True = lower is better (inverted normalization)
# Lower is Better: TrailingPE, NetDebt_EBITDA, PayoutRatio
LOWER_IS_BETTER = {
    'TrailingPE':     True,
    'NetDebt_EBITDA': True,
    'PayoutRatio':    True,
}

# ─── Knock-out Filter Registry ────────────────────────────────────────────────
# Declarative Boolean gates applied PER STRATEGY before normalization.
# Each rule is a (column, operator, threshold) tuple evaluated against raw CSV values.
# Stocks failing ANY rule in their strategy's list are dropped entirely.
# 'CurrentRatio' is read from the CSV but is NOT in INDICATOR_COLUMNS —
# it acts as a pure filter gate and is never packed into the C struct.
INTRINSIC_KNOCKOUTS = {
    'deep-value': [
        ('FCFYield',         '>',  0.0   ),  # Must generate positive free cash flow
        ('OperatingMargin',  '>',  0.0   ),  # Must be profitable from core operations
    ],
    'hyper-growth': [
        ('RevenueGrowth',    '>',  0.0   ),  # Top-line must not be shrinking
        ('GrossMargin',      '>',  0.20  ),  # Baseline pricing power: >20%
    ],
    'income': [
        ('PayoutRatio',      '>',  0.0   ),  # Must have earnings (pays a dividend)
        ('PayoutRatio',      '<',  0.85  ),  # Sustainable payout: ceiling at 85%
        ('ShareholderYield', '>',  0.015 ),  # Minimum 1.5% annual shareholder yield
    ],
    'quality': [
        ('ROIC',             '>',  0.08  ),  # Return on assets proxy must exceed 8%
        ('CurrentRatio',     '>',  1.0   ),  # Short-term liquidity: assets > liabilities
    ],
}

DERIVED_KNOCKOUTS = {
    'momentum': [
        ('PriceVs200SMA',    '>',   0.0   ),  # Stock must have valid SMA data (>0 = exists in Parquet)
    ],
}

VALID_STRATEGIES = frozenset(list(INTRINSIC_KNOCKOUTS.keys()) + list(DERIVED_KNOCKOUTS.keys()))

# ─── Statistical Normalization Config (Phase 1) ────────────────────────────────
# Optimal bounds [L, U] and decay rates (sigma_L, sigma_R) for piecewise gaussian metrics.
GOLDILOCKS_PARAMS = {
    'PayoutRatio':      {'L': 0.35,  'U': 0.70,  'sigma_L': 0.25,  'sigma_R': 0.20},
    'RSI':              {'L': 45.0,  'U': 60.0,  'sigma_L': 15.0,  'sigma_R': 10.0},
    'TrailingPE':       {'L': 12.0,  'U': 28.0,  'sigma_L': 5.0,   'sigma_R': 15.0},
    'EV_EBITDA':        {'L': 8.0,   'U': 16.0,  'sigma_L': 3.0,   'sigma_R': 8.0},
    'NetDebt_EBITDA':   {'L': 0.0,   'U': 3.0,   'sigma_L': 1.0,   'sigma_R': 2.0},
    'Beta':             {'L': 0.8,   'U': 1.2,   'sigma_L': 0.6,   'sigma_R': 0.4},
    'PriceVs200SMA':    {'L': 1.02,  'U': 1.15,  'sigma_L': 0.10,  'sigma_R': 0.08},
    'ShareholderYield': {'L': 0.04,  'U': 0.09,  'sigma_L': 0.03,  'sigma_R': 0.04},
}

# ─── In-Memory TTL Cache & Tiingo API Ingestion ──────────────────────────────

_LIVE_DATA_CACHE = {
    'data': None,
    'timestamp': 0.0
}

_TECH_DATA_CACHE = {
    'price_df': None,
    'volume_df': None,
    'spy_s': None,
    'timestamp': 0.0
}
CACHE_TTL_SECONDS = 1800.0  # 30 minutes TTL


def fetch_tiingo_live_data(tiingo_key, tickers_list, batch_size=50):
    """
    Fetches real-time stock prices from Tiingo API in chunks/batches of `batch_size` (default 50)
    to prevent HTTP 414 URI Too Long errors when querying large universes.
    Returns dictionary mapping upper-case ticker -> {'price': float32, 'volume': float32}.
    Explicitly frees HTTP buffers to prevent RAM bloat on Render.
    """
    if not tiingo_key or not tickers_list:
        return {}

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Token {tiingo_key}'
    }

    price_map = {}
    total_tickers = len(tickers_list)

    for i in range(0, total_tickers, batch_size):
        batch = tickers_list[i:i + batch_size]
        tickers_str = ",".join(batch)
        url = f"https://api.tiingo.com/iex/?tickers={tickers_str}"
        
        resp = None
        try:
            resp = requests.get(url, headers=headers, timeout=8)
            if resp.status_code == 200:
                try:
                    data = resp.json()
                    if isinstance(data, list):
                        for item in data:
                            t_sym = str(item.get('ticker', '')).upper().strip()
                            p = item.get('last') or item.get('tngoLast') or item.get('prevClose')
                            v = item.get('volume', 0.0)
                            if t_sym and p is not None:
                                try:
                                    price_map[t_sym] = {
                                        'price': _safe_float(p),
                                        'volume': _safe_float(v)
                                    }
                                except (ValueError, TypeError):
                                    pass
                    del data
                except Exception as json_err:
                    print(f"[Tiingo API Warning] Batch {i//batch_size + 1} JSON decode failed: {json_err}")
            elif resp.status_code == 429:
                print("[Tiingo API Warning] Rate limit (HTTP 429) hit. Stopping API requests and returning accumulated quotes.")
                break
            else:
                print(f"[Tiingo API Warning] Batch {i//batch_size + 1} returned status code {resp.status_code}.")
        except Exception as e:
            print(f"[Tiingo API Error] Batch {i//batch_size + 1} failed: {e}")
        finally:
            if resp is not None:
                try:
                    resp.close()
                except Exception:
                    pass
                del resp
            time.sleep(0.15)  # Rate limit throttle delay (150ms) between batch HTTP requests

    gc.collect()
    if price_map:
        print(f"[Tiingo API] Successfully fetched real-time quotes for {len(price_map)} tickers in batches of {batch_size}.")
    return price_map


def ensure_tech_cache():
    """
    Hydrates _TECH_DATA_CACHE from us-shareprices-daily.parquet and spy_historical.csv if missing or expired (30m TTL).
    Returns tuple of (full_price_df, full_volume_df, full_spy_s).
    """
    global _TECH_DATA_CACHE
    now = time.time()

    if (_TECH_DATA_CACHE['price_df'] is not None and 
        _TECH_DATA_CACHE['volume_df'] is not None and 
        _TECH_DATA_CACHE['spy_s'] is not None and 
        (now - _TECH_DATA_CACHE['timestamp']) < CACHE_TTL_SECONDS):
        print(f"[CACHE HIT] Technical Data Cache valid ({int(now - _TECH_DATA_CACHE['timestamp'])}s old).")
        return _TECH_DATA_CACHE['price_df'], _TECH_DATA_CACHE['volume_df'], _TECH_DATA_CACHE['spy_s']

    print("[CACHE MISS] Technical Data Cache expired or uninitialized. Reading Parquet/CSV...")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    simfin_parquet_candidates = [
        os.path.join(base_dir, '..', 'data', 'us-shareprices-daily.parquet'),
        os.path.join(base_dir, '..', 'data', 'raw_simfin', 'us-shareprices-daily.parquet'),
    ]
    simfin_parquet_path = next((p for p in simfin_parquet_candidates if os.path.exists(p)), None)

    spy_candidates = [
        os.path.join(base_dir, '..', 'data', 'spy_historical.csv'),
    ]
    spy_path = next((p for p in spy_candidates if os.path.exists(p)), None)

    if not simfin_parquet_path or not spy_path:
        print(f"CRITICAL PATH WARNING: Data files missing for technical indicator computation. Parquet: '{simfin_parquet_path}', SPY: '{spy_path}'")
        return None, None, None

    try:
        t0_io = time.perf_counter()
        min_date = pd.Timestamp.now().normalize() - pd.Timedelta(days=370)

        # Read parquet with PyArrow date pushdown filter
        raw_p = pd.read_parquet(
            simfin_parquet_path,
            engine='pyarrow',
            columns=['Date', 'Ticker', 'Adj. Close', 'Close', 'Volume'],
            filters=[('Date', '>=', min_date)]
        )
        if raw_p.empty:
            return None, None, None

        raw_p['Date'] = pd.to_datetime(raw_p['Date'])
        raw_p['Price'] = raw_p['Adj. Close'].fillna(raw_p['Close']).astype('float32')
        if 'Volume' not in raw_p.columns:
            raw_p['Volume'] = 0.0
        else:
            raw_p['Volume'] = raw_p['Volume'].fillna(0.0).astype('float32')

        raw_p.drop(columns=['Adj. Close', 'Close'], inplace=True, errors='ignore')

        full_price_df = raw_p.pivot(index='Date', columns='Ticker', values='Price').sort_index().tail(260)
        full_volume_df = raw_p.pivot(index='Date', columns='Ticker', values='Volume').sort_index().tail(260)
        del raw_p
        gc.collect()

        # Load SPY benchmark history
        spy_df = pd.read_csv(spy_path, usecols=['Date', 'SPY_Adj_Close'])
        spy_df['Date'] = pd.to_datetime(spy_df['Date'])
        spy_full = spy_df.set_index('Date')['SPY_Adj_Close'].astype('float32').sort_index()
        del spy_df
        gc.collect()

        price_start_date = full_price_df.index[0]
        price_end_date = full_price_df.index[-1]
        full_spy_s = spy_full[(spy_full.index >= price_start_date) & (spy_full.index <= price_end_date)]
        if len(full_spy_s) < 2:
            full_spy_s = spy_full.tail(260)
        del spy_full

        _TECH_DATA_CACHE['price_df'] = full_price_df
        _TECH_DATA_CACHE['volume_df'] = full_volume_df
        _TECH_DATA_CACHE['spy_s'] = full_spy_s
        _TECH_DATA_CACHE['timestamp'] = now

        t1_io = time.perf_counter()
        print(f"[PROFILER] Technical Indicator Parquet/CSV I/O (Cache Miss): {t1_io - t0_io:.3f}s")
        print(f"[Technical Indicators] Cache hydrated: Parquet {price_start_date.date()} -> {price_end_date.date()}, "
              f"SPY aligned {full_spy_s.index[0].date()} -> {full_spy_s.index[-1].date()} ({len(full_spy_s)} rows)")

        return full_price_df, full_volume_df, full_spy_s
    except Exception as e:
        print(f"ERROR hydrating technical data cache: {e}")
        return None, None, None


def compute_live_technical_indicators(unique_tickers):
    """
    Dynamically calculates momentum & technical indicators (RelativeStrength, PriceVs200SMA, RSI, Beta)
    from offline us-shareprices-daily.parquet and spy_historical.csv for the active stock universe.
    Aggressively optimizes RAM lifecycle via date pre-filtering, 1D vectorization, and immediate matrix deletion.
    """
    if not unique_tickers:
        return {}

    tech_map = {}
    try:
        t0_io = time.perf_counter()
        full_price_df, full_volume_df, full_spy_s = ensure_tech_cache()
        if full_price_df is None or full_spy_s is None:
            return tech_map
        t1_io = time.perf_counter()

        # Subset for the currently requested unique universe
        t0_calc = time.perf_counter()
        valid_tickers = [t for t in unique_tickers if t in full_price_df.columns]
        if not valid_tickers:
            return tech_map
            
        price_df = full_price_df[valid_tickers]
        spy_s = full_spy_s
        # 1. Vectorized 1-Year Relative Strength vs SPY (same time window as stocks)
        spy_1y_ret = float((spy_s.iloc[-1] - spy_s.iloc[0]) / spy_s.iloc[0]) if len(spy_s) > 1 else 0.0
        stk_1y_ret = (price_df.iloc[-1] - price_df.iloc[0]) / price_df.iloc[0].replace(0, float('nan'))
        rel_strength = (stk_1y_ret - spy_1y_ret).astype('float32')

        # 2. Vectorized Price vs 200-day Simple Moving Average
        sma_200 = price_df.tail(200).mean()
        price_vs_sma = (price_df.iloc[-1] / sma_200.replace(0, float('nan'))).astype('float32')

        # Diagnostic log for PriceVs200SMA distribution
        valid_sma = price_vs_sma.dropna()
        pos_sma_c = int((valid_sma > 0.0).sum())
        zero_sma_c = len(price_vs_sma) - pos_sma_c
        sample_sma = [round(float(v), 4) for v in valid_sma.iloc[:5]] if not valid_sma.empty else []
        print(f"[DIAG] PriceVs200SMA statistics: Total tickers={len(price_vs_sma)}, > 0.0 count={pos_sma_c}, <= 0.0 / NaN count={zero_sma_c}, Sample={sample_sma}")

        # 3. Vectorized 14-day RSI
        delta = price_df.diff()
        gain = delta.where(delta > 0, 0.0).tail(14).mean()
        loss = (-delta.where(delta < 0, 0.0)).tail(14).mean()
        rs = gain / (loss + 1e-6)
        rsi_14 = (100.0 - (100.0 / (1.0 + rs))).astype('float32')

        # 4. Vectorized 1-Year Beta vs SPY (aligned daily returns via NumPy dot-product)
        stk_returns = price_df.pct_change()
        spy_returns = spy_s.pct_change().reindex(price_df.index)

        spy_valid = spy_returns.dropna()
        spy_var = float(spy_valid.var()) if len(spy_valid) > 1 else 0.0

        if spy_var > 0.0:
            spy_dm = (spy_returns - spy_returns.mean()).fillna(0.0).values
            stk_dm = (stk_returns - stk_returns.mean()).fillna(0.0).values
            
            # Explicitly sanitize arrays to prevent 'invalid value encountered in dot' RuntimeWarnings
            spy_dm = np.nan_to_num(spy_dm, nan=0.0, posinf=0.0, neginf=0.0)
            stk_dm = np.nan_to_num(stk_dm, nan=0.0, posinf=0.0, neginf=0.0)
            
            n_obs = max(1, len(spy_returns) - 1)
            cov_vals = np.dot(stk_dm.T, spy_dm) / n_obs
            cov_s = pd.Series(cov_vals, index=price_df.columns, dtype='float32')
            beta_s = (cov_s / spy_var).fillna(1.0)
        else:
            cov_s = pd.Series(0.0, index=price_df.columns, dtype='float32')
            beta_s = pd.Series(1.0, index=price_df.columns, dtype='float32')

        # IMMEDIATELY ERASE ALL HEAVY 2D MATRICES AND FREE RAM BEFORE DICT GENERATION
        del price_df, spy_s, delta, gain, loss, rs, stk_returns, spy_returns, cov_s
        gc.collect()

        # Construct dictionary mapping strictly from 1D Series vectors
        for ticker in rel_strength.index:
            rs_v = float(rel_strength[ticker]) if not math.isnan(rel_strength[ticker]) else 0.0
            sma_v = float(price_vs_sma[ticker]) if not math.isnan(price_vs_sma[ticker]) else 1.0
            rsi_v = float(rsi_14[ticker]) if not math.isnan(rsi_14[ticker]) else 50.0
            beta_v = float(beta_s[ticker]) if not math.isnan(beta_s[ticker]) else 1.0

            tech_map[ticker] = {
                'RelativeStrength': _safe_float(rs_v),
                'PriceVs200SMA': _safe_float(sma_v),
                'RSI': _safe_float(rsi_v),
                'Beta': _safe_float(beta_v)
            }

        del rel_strength, price_vs_sma, rsi_14, beta_s
        gc.collect()
        t1_calc = time.perf_counter()
        print(f"[PROFILER] Technical Indicator Computation Math: {t1_calc - t0_calc:.3f}s")
        print(f"[Technical Indicators] Ultra-fast computation completed for {len(tech_map)} tickers.")
    except Exception as e:
        import traceback
        print(f"⚠️ [Technical Indicators Error] Failed to compute dynamic indicators: {e}")
        traceback.print_exc()

    return tech_map


def get_cached_fundamental_snapshot(csv_path, ttl_seconds=CACHE_TTL_SECONDS):
    """
    Returns live fundamental dataset from In-Memory TTL Cache (1800s / 30 mins).
    Sub-second response time (< 1ms) for baseline dataset.
    Returns copy-isolated raw fundamental records.
    """
    global _LIVE_DATA_CACHE
    now = time.time()

    # 1. Cache Hit: Return an isolated copy of cached data instantly if fresh (< TTL)
    if _LIVE_DATA_CACHE['data'] is not None and (now - _LIVE_DATA_CACHE['timestamp']) < ttl_seconds:
        return _LIVE_DATA_CACHE['data'].copy(), None

    # Explicitly release old cache memory before reloading
    if _LIVE_DATA_CACHE['data'] is not None:
        _LIVE_DATA_CACHE['data'] = None
        gc.collect()

    # 2. Load baseline records from local disk CSV
    disk_records, err = load_latest_fundamental_snapshot(csv_path)
    if disk_records is None or disk_records.empty:
        return None, err or "Failed to load base dataset."

    # Update in-memory cache & timestamp
    _LIVE_DATA_CACHE['data'] = disk_records
    _LIVE_DATA_CACHE['timestamp'] = now
    print(f"[In-Memory Cache] Updated baseline cache with {len(disk_records)} raw records (TTL {int(ttl_seconds)}s).")

    # Return isolated copy to caller
    return _LIVE_DATA_CACHE['data'].copy(), None


# ─── Data Ingestion Helper ───────────────────────────────────────────────────

def load_latest_fundamental_snapshot(csv_path):
    """
    Ingests historical_fundamental_data.csv (or market_data.csv as fallback),
    extracts the latest cross-sectional snapshot per Ticker, and returns a pandas DataFrame.
    """
    if not os.path.exists(csv_path):
        return None, f"Dataset not found at {csv_path}"
    
    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            return pd.DataFrame(), None
            
        if "Quarter_End_Date" in df.columns:
            df["Quarter_End_Date"] = pd.to_datetime(df["Quarter_End_Date"])
            # Filter for the latest available quarter per ticker
            df = df.sort_values("Quarter_End_Date").groupby("Ticker").last().reset_index()
            df["Quarter_End_Date"] = df["Quarter_End_Date"].dt.strftime("%Y-%m-%d")
        
        # Ensure 'Sector' key exists (mapped from GICS_Sector if needed)
        if "GICS_Sector" in df.columns and "Sector" not in df.columns:
            df["Sector"] = df["GICS_Sector"]

        # Ensure 'LongName' key exists
        if "LongName" not in df.columns:
            df["LongName"] = df["Ticker"]
            
        float_cols = df.select_dtypes(include=['float64']).columns
        if not float_cols.empty:
            df[float_cols] = df[float_cols].astype('float32')

        return df, None
    except Exception as e:
        return None, f"Failed to parse fundamental snapshot dataset: {str(e)}"


# ─── 1. Data Structure Definition ─────────────────────────────────────────────

class Stock(ctypes.Structure):
    """
    Must be byte-for-byte identical to the C struct in stock.h:
        char   ticker[8];              //  8 bytes
        float  indicators[16];         // 64 bytes
        double final_score;            //  8 bytes
    Total: 80 bytes, no padding.
    """
    _fields_ = [
        ("ticker",      ctypes.c_char * 8),
        ("indicators",  ctypes.c_float * NUM_INDICATORS),
        ("final_score", ctypes.c_double),
    ]

# Assert exact alignment and 80-byte size matching C struct (stock.h)
assert ctypes.sizeof(Stock) == 80, f"Stock struct size mismatch: expected 80 bytes, got {ctypes.sizeof(Stock)}"


# ─── 2. Library Loading & Function Signatures ──────────────────────────────────

lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'c_kernel', 'libsortbuslines.so')

try:
    lib = ctypes.CDLL(lib_path)
    # void compute_scores(Stock* stocks, int num_stocks, const double* weights);
    lib.compute_scores.argtypes = [ctypes.POINTER(Stock), ctypes.c_int, ctypes.POINTER(ctypes.c_double)]
    lib.compute_scores.restype  = None

    # void sort_stocks_by_score(Stock* stocks, int num_stocks);
    lib.sort_stocks_by_score.argtypes = [ctypes.POINTER(Stock), ctypes.c_int]
    lib.sort_stocks_by_score.restype  = None
except OSError as e:
    print(f"Warning: Could not load C-library at {lib_path}. Execution will require WSL/Linux environment.")
    lib = None


def apply_early_pruning_vectorized(df, strategy, allowed_sectors=None, min_market_cap=None, max_market_cap=None, specific_tickers=None):
    """
    Vectorized early pruning of the DataFrame universe.
    Applies Intrinsic strategy knockouts, sector/demographic bounds, and the Gatekeeper matrix.
    """
    t0_dyn = time.perf_counter()
    if df.empty:
        return df

    before = len(df)

    if specific_tickers:
        ticker_set = {t.upper().strip() for t in specific_tickers}
        df = df[df['Ticker'].str.upper().str.strip().isin(ticker_set)]

    if allowed_sectors is not None:
        if isinstance(allowed_sectors, str):
            sector_set = {allowed_sectors.strip().lower()}
        else:
            sector_set = {str(s).strip().lower() for s in allowed_sectors if s}
        df = df[df['Sector'].str.strip().str.lower().isin(sector_set)]

    # Gatekeeper Cut-Off Constants
    GATEKEEPER_MIN_MARKET_CAP = 1_000_000_000.0  # $1B
    GATEKEEPER_MIN_PRICE = 5.00                  # $5.00
    GATEKEEPER_MIN_ADTV = 5_000_000.0            # $5M 20-day ADTV

    # Market Cap Filter
    m_cap_min = max(min_market_cap if min_market_cap is not None else 0.0, GATEKEEPER_MIN_MARKET_CAP)
    df = df[df['MarketCap'].fillna(0.0) >= m_cap_min]
    if max_market_cap is not None:
        df = df[df['MarketCap'].fillna(0.0) <= max_market_cap]

    # Price and ADTV from tech cache
    price_df = _TECH_DATA_CACHE.get('price_df')
    volume_df = _TECH_DATA_CACHE.get('volume_df')
    if price_df is not None and volume_df is not None:
        latest_prices = price_df.iloc[-1].rename('LatestPrice')
        
        tail_price = price_df.tail(20)
        tail_volume = volume_df.tail(20)
        adtv_series = (tail_price * tail_volume).mean(axis=0).rename('ADTV')

        df = df.merge(latest_prices, left_on='Ticker', right_index=True, how='left')
        df = df.merge(adtv_series, left_on='Ticker', right_index=True, how='left')

        df = df[df['LatestPrice'].fillna(0.0) > GATEKEEPER_MIN_PRICE]
        df = df[df['ADTV'].fillna(0.0) >= GATEKEEPER_MIN_ADTV]

        df = df.drop(columns=['LatestPrice', 'ADTV'])

    # Insolvency rule
    if 'NetDebt_EBITDA' in df.columns:
        df = df[~(df['NetDebt_EBITDA'].isna() | (df['NetDebt_EBITDA'] > 15.0))]

    # Intrinsic Knockouts
    rules = INTRINSIC_KNOCKOUTS.get(strategy)
    if rules:
        for col, op, threshold in rules:
            if col not in df.columns:
                continue
            val_col = df[col].fillna(0.0)
            if op == '>':   df = df[val_col > threshold]
            elif op == '<':  df = df[val_col < threshold]
            elif op == '>=': df = df[val_col >= threshold]
            elif op == '<=': df = df[val_col <= threshold]

    t1_dyn = time.perf_counter()
    print(f"[PROFILER] Vectorized Early Pruning ({strategy}): {t1_dyn - t0_dyn:.3f}s")
    print(f"[EarlyPruning] {before} -> {len(df)} stocks ({before - len(df)} eliminated).")
    return df


def apply_derived_knockouts_vectorized(df, strategy):
    """
    Vectorized pruning based on technical/derived metrics (e.g. PriceVs200SMA)
    that are populated AFTER tech hydration.
    """
    if df.empty:
        return df

    before = len(df)
    rules = DERIVED_KNOCKOUTS.get(strategy)
    if rules:
        for col, op, threshold in rules:
            if col not in df.columns:
                continue
            val_col = df[col].fillna(0.0)
            if op == '>':   df = df[val_col > threshold]
            elif op == '<':  df = df[val_col < threshold]
            elif op == '>=': df = df[val_col >= threshold]
            elif op == '<=': df = df[val_col <= threshold]

    after = len(df)
    if rules:
        print(f"[DerivedKnockouts] '{strategy}': {before} -> {after} stocks ({before - after} eliminated)")
    return df





# ─── 4. Min-Max Normalization Layer ───────────────────────────────────────────

def _safe_float(val, default=0.0):
    """Coerce a scalar/cell value to float, treating 'nan'/empty/None/inf as default."""
    if val is None:
        return default
    if hasattr(val, "squeeze"):
        val = val.squeeze()
    if hasattr(val, "item"):
        try:
            val = val.item()
        except (ValueError, AttributeError):
            if hasattr(val, "iloc"):
                val = val.iloc[0]
    elif hasattr(val, "iloc"):
        val = val.iloc[0]
    s = str(val).strip().lower()
    if s in ('nan', '', 'inf', '-inf', 'none'):
        return default
    try:
        result = float(s)
        if math.isnan(result) or math.isinf(result):
            return default
        return result
    except (ValueError, TypeError):
        return default

# Alias for cross-module consistency with portfolio engines
_scalar_float = _safe_float


def sanitize_for_json(obj):
    """
    Recursively sanitize any Python structure (DataFrame, Series, ndarray, np.generic, dict, list)
    into pure Python types guaranteed free of NaN, Infinity, or non-serializable objects.
    """
    if obj is None:
        return None
    if isinstance(obj, pd.DataFrame):
        df_clean = obj.fillna(0.0).replace([np.inf, -np.inf], 0.0)
        # Format DatetimeIndex if present
        if isinstance(df_clean.index, pd.DatetimeIndex):
            dates = df_clean.index.strftime('%Y-%m-%d').tolist()
            records = df_clean.to_dict(orient='records')
            for i in range(len(records)):
                records[i]['date'] = dates[i]
            return records
        return df_clean.to_dict(orient='records')
    if isinstance(obj, pd.Series):
        s_clean = obj.fillna(0.0).replace([np.inf, -np.inf], 0.0)
        if isinstance(s_clean.index, pd.DatetimeIndex):
            s_dict = {k.strftime('%Y-%m-%d'): v for k, v in s_clean.to_dict().items()}
        else:
            s_dict = s_clean.to_dict()
        return sanitize_for_json(s_dict)
    if isinstance(obj, (np.ndarray, list, tuple, set)):
        return [sanitize_for_json(item) for item in obj]
    if isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return 0.0 if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (pd.Timestamp, np.datetime64)):
        return str(obj)[:10]
    return str(obj)


def _clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Zero NaN/inf in every numeric column of a DataFrame."""
    if df is None or df.empty:
        return df if df is not None else pd.DataFrame()
    out = df.copy()
    num = out.select_dtypes(include=[np.number]).columns
    if len(num):
        out[num] = out[num].fillna(0.0).replace([np.inf, -np.inf], 0.0)
    return out


def _to_naive_ts(val) -> pd.Timestamp:
    """Normalize any date/timestamp representation to a timezone-naive midnight Timestamp."""
    ts = pd.Timestamp(val)
    if ts.tz is not None:
        ts = ts.tz_localize(None)
    return ts.normalize()

# Alias for cross-module compatibility
_to_naive_timestamp = _to_naive_ts


def _calculate_piecewise_gaussian(val, params):
    """Scalar piecewise Gaussian normalizer (retained for backward compatibility)."""
    if val is None:
        return 0.0
    try:
        val = float(val)
    except (ValueError, TypeError):
        return 0.0

    if math.isnan(val) or math.isinf(val):
        return 0.0

    L = params.get('L')
    U = params.get('U')
    sigma_L = params.get('sigma_L')
    sigma_R = params.get('sigma_R')

    if L is None or U is None or sigma_L is None or sigma_R is None:
        return 0.0

    if val < L:
        if sigma_L == 0.0:
            return 0.0
        exponent = -((val - L) ** 2) / (2 * (sigma_L ** 2))
        try:
            return 100.0 * math.exp(exponent)
        except OverflowError:
            return 0.0
    elif val > U:
        if sigma_R == 0.0:
            return 0.0
        exponent = -((val - U) ** 2) / (2 * (sigma_R ** 2))
        try:
            return 100.0 * math.exp(exponent)
        except OverflowError:
            return 0.0
    else:
        return 100.0


def _vectorized_goldilocks(arr, params):
    """
    Vectorized piecewise Gaussian normalizer for Goldilocks indicators using NumPy array operations.
    Returns a float32 ndarray of scores in [0.0, 100.0].
    """
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    L = float(params['L'])
    U = float(params['U'])
    sigma_L = float(params['sigma_L'])
    sigma_R = float(params['sigma_R'])

    scores = np.full(arr.shape, 100.0, dtype=np.float32)

    mask_left = arr < L
    if sigma_L > 0.0:
        exponent_l = -((arr - L) ** 2) / (2.0 * (sigma_L ** 2))
        scores = np.where(mask_left, 100.0 * np.exp(np.clip(exponent_l, -87.0, 87.0)), scores)
    else:
        scores = np.where(mask_left, 0.0, scores)

    mask_right = arr > U
    if sigma_R > 0.0:
        exponent_r = -((arr - U) ** 2) / (2.0 * (sigma_R ** 2))
        scores = np.where(mask_right, 100.0 * np.exp(np.clip(exponent_r, -87.0, 87.0)), scores)
    else:
        scores = np.where(mask_right, 0.0, scores)

    return scores.astype(np.float32)


def _calculate_sigmoid_zscore(val, median_val, mad_val, invert=False):
    """Scalar Sigmoid Z-score normalizer (retained for backward compatibility)."""
    if val is None or median_val is None or mad_val is None:
        return 50.0
    try:
        val = float(val)
        median_val = float(median_val)
        mad_val = float(mad_val)
    except (ValueError, TypeError):
        return 50.0

    if math.isnan(val) or math.isinf(val) or math.isnan(median_val) or math.isinf(median_val):
        return 50.0

    if math.isnan(mad_val) or math.isinf(mad_val):
        mad_val = 0.0

    denominator = max(mad_val, 1e-4)
    z = (val - median_val) / denominator
    if invert:
        z = -z

    try:
        exponent = -1.5 * z
        if exponent > 700.0:
            return 0.0
        if exponent < -700.0:
            return 100.0
        return 100.0 / (1.0 + math.exp(exponent))
    except (OverflowError, ZeroDivisionError):
        return 0.0 if z < 0.0 else 100.0


def _vectorized_sigmoid(arr, median_val, mad_val, invert=False):
    """
    Vectorized Monotonic Normalizer utilizing a Sigmoid over Robust Z-Scores.
    Returns a float32 ndarray of scores in [0.0, 100.0].
    """
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if median_val is None or mad_val is None or math.isnan(median_val) or math.isinf(median_val):
        return np.full(arr.shape, 50.0, dtype=np.float32)

    if math.isnan(mad_val) or math.isinf(mad_val):
        mad_val = 0.0

    denominator = max(float(mad_val), 1e-4)
    z = (arr - float(median_val)) / denominator
    if invert:
        z = -z

    exponent = np.clip(-1.5 * z, -87.0, 87.0)
    scores = 100.0 / (1.0 + np.exp(exponent))
    return scores.astype(np.float32)


OVERRIDE_COLUMNS = frozenset({'PayoutRatio', 'TrailingPE', 'NetDebt_EBITDA', 'EV_EBITDA'})


def normalize_stocks(stocks_list, strategy=None, allowed_sectors=None, min_market_cap=None, max_market_cap=None, global_stocks_list=None):
    """
    Apply dynamic filters, strategy knock-out filters, then normalize the surviving universe
    using vectorized piecewise Gaussian (Goldilocks) or robust Sigmoid Z-score curves.

    Args:
        stocks_list:     list of raw CSV dicts.
        strategy:        one of VALID_STRATEGIES, or None for no filtering.
        allowed_sectors: list of allowed sector names, or None.
        min_market_cap:  minimum market cap, or None.
        max_market_cap:  maximum market cap, or None.
        global_stocks_list: list of all raw CSV dicts (global baseline for sector neutrality).
    """
    if not stocks_list:
        return []

    # Sector-Neutral Baseline Stabilization: Use the entire global population prior to filtering
    baseline_list = global_stocks_list if global_stocks_list is not None else list(stocks_list)

    if strategy or allowed_sectors or min_market_cap or max_market_cap:
        df = pd.DataFrame(stocks_list)
        df = apply_early_pruning_vectorized(df, strategy, allowed_sectors, min_market_cap, max_market_cap)
        df = apply_derived_knockouts_vectorized(df, strategy)
        if df.empty:
            return []
        stocks_list = df.to_dict(orient='records')

    n = len(stocks_list)
    t0_norm = time.perf_counter()

    # Step 3: Vectorized extraction of raw indicator float matrix (N x 16)
    df_stocks = pd.DataFrame(stocks_list)
    for col in INDICATOR_COLUMNS:
        if col not in df_stocks.columns:
            df_stocks[col] = 0.0

    raw_matrix = df_stocks[INDICATOR_COLUMNS].apply(pd.to_numeric, errors='coerce').fillna(0.0).to_numpy(dtype=np.float32)
    raw_matrix = np.nan_to_num(raw_matrix, nan=0.0, posinf=0.0, neginf=0.0)

    norm_matrix = np.zeros((n, len(INDICATOR_COLUMNS)), dtype=np.float32)

    # Pre-extract sector array for baseline and current stocks_list for sector-neutral metrics
    base_sects = np.array([str(s.get('Sector', '')).strip().lower() for s in baseline_list])
    stock_sects = np.array([str(s.get('Sector', '')).strip().lower() for s in stocks_list])

    # Step 4: Normalize each column dynamically using vectorized math
    for j, col in enumerate(INDICATOR_COLUMNS):
        col_arr = raw_matrix[:, j]

        if col == 'EV_EBITDA':
            params = GOLDILOCKS_PARAMS[col]
            ebitda_arr = np.array([_safe_float(s.get('EBITDA', 0.0)) for s in stocks_list], dtype=np.float32)
            scores = _vectorized_goldilocks(col_arr, params)
            scores = np.where((ebitda_arr <= 0.0) | (col_arr < 0.0), 0.0, scores)
            norm_matrix[:, j] = scores

        elif col == 'ShareholderYield':
            params = GOLDILOCKS_PARAMS[col]
            scores = _vectorized_goldilocks(col_arr, params)
            scores = np.where(col_arr < -0.02, 0.0, scores)
            norm_matrix[:, j] = scores

        elif col in GOLDILOCKS_PARAMS:
            params = GOLDILOCKS_PARAMS[col]
            scores = _vectorized_goldilocks(col_arr, params)
            if col in OVERRIDE_COLUMNS:
                scores = np.where(col_arr < 0.0, 0.0, scores)
            norm_matrix[:, j] = scores

        else:
            invert = LOWER_IS_BETTER.get(col, False)
            if col in ('RevenueGrowth', 'EPSGrowth'):
                df_base = pd.DataFrame(baseline_list)
                base_col_vals = pd.to_numeric(df_base.get(col, pd.Series(0, index=df_base.index)), errors='coerce').fillna(0.0).to_numpy(dtype=np.float32)
                base_col_vals = np.nan_to_num(base_col_vals, nan=0.0, posinf=0.0, neginf=0.0)

                sector_stats = {}
                for sect in np.unique(base_sects):
                    mask = (base_sects == sect)
                    s_vals = base_col_vals[mask]
                    if len(s_vals) > 0:
                        med = float(np.median(s_vals))
                        mad = float(np.median(np.abs(s_vals - med)))
                    else:
                        med, mad = 0.0, 0.0
                    sector_stats[sect] = (med, mad)

                meds = np.zeros(n, dtype=np.float32)
                mads = np.zeros(n, dtype=np.float32)
                for sect, (med, mad) in sector_stats.items():
                    s_mask = (stock_sects == sect)
                    meds[s_mask] = med
                    mads[s_mask] = mad

                denominators = np.maximum(mads, 1e-4)
                z = (col_arr - meds) / denominators
                if invert:
                    z = -z
                exponent = np.clip(-1.5 * z, -87.0, 87.0)
                scores = (100.0 / (1.0 + np.exp(exponent))).astype(np.float32)
                norm_matrix[:, j] = scores
            else:
                if len(col_arr) > 0:
                    median_val = float(np.median(col_arr))
                    mad_val = float(np.median(np.abs(col_arr - median_val)))
                else:
                    median_val, mad_val = 0.0, 0.0

                scores = _vectorized_sigmoid(col_arr, median_val, mad_val, invert)
                if col in OVERRIDE_COLUMNS:
                    scores = np.where(col_arr < 0.0, 0.0, scores)
                norm_matrix[:, j] = scores

    # Step 5: Rebuild stock dicts with normalized values using vectorized DataFrame export
    norm_df = pd.DataFrame(norm_matrix, columns=INDICATOR_COLUMNS)
    tickers = [s.get('Ticker', '') for s in stocks_list]
    norm_df.insert(0, 'Ticker', tickers)
    result = norm_df.to_dict(orient='records')

    t1_norm = time.perf_counter()
    print(f"[PROFILER] Normalization Math (Vectorized NumPy): {t1_norm - t0_norm:.3f}s")

    return result


# ─── 4. Main FFI Entry Point ───────────────────────────────────────────────────

def run_financial_engine(normalized_stocks, weights_list):
    """
    Surgical Patch: Accepts pre-normalized stocks directly to bypass duplicate normalization.
    Populates ctypes array → C dot-product → QuickSort.
    """
    t0_ffi = time.perf_counter()
    num_stocks = len(normalized_stocks)
    if num_stocks == 0:
        return []

    if lib is None:
        # Fallback scoring for host OS without compiled C library
        results = []
        tot_w = sum(float(w) for w in weights_list)
        for stock_data in normalized_stocks:
            score = sum(float(stock_data.get(col, 0.0)) * float(w) for col, w in zip(INDICATOR_COLUMNS, weights_list))
            results.append({'Ticker': stock_data['Ticker'], 'FinalScore': round(score, 4)})
        results.sort(key=lambda x: x['FinalScore'], reverse=True)
        t1_ffi = time.perf_counter()
        print(f"[PROFILER] C-FFI Scoring Execution (Python Fallback): {t1_ffi - t0_ffi:.3f}s")
        return results

    # ── Step 1: Populate ctypes Stock array ──────────────────────────────────
    StockArray = Stock * num_stocks
    c_stocks   = StockArray()

    for i, stock_data in enumerate(normalized_stocks):
        # Encode ticker: max 7 meaningful chars + null terminator fits in char[8]
        encoded = stock_data['Ticker'].encode('utf-8')[:7]
        c_stocks[i].ticker = encoded

        # Map 16 normalized indicator values sequentially into indicators[]
        for j, col in enumerate(INDICATOR_COLUMNS):
            c_stocks[i].indicators[j] = float(stock_data.get(col, 0.0))

        c_stocks[i].final_score = 0.0

    # ── Step 2: Build weights array ──────────────────────────────────────────
    WeightsArray = ctypes.c_double * NUM_INDICATORS
    c_weights    = WeightsArray(*[float(w) for w in weights_list])

    # ── Step 3: Fire C engine (dot product + QuickSort) ──────────────────────
    start_ptr   = ctypes.cast(c_stocks,   ctypes.POINTER(Stock))
    weights_ptr = ctypes.cast(c_weights,  ctypes.POINTER(ctypes.c_double))

    lib.compute_scores(start_ptr, num_stocks, weights_ptr)
    lib.sort_stocks_by_score(start_ptr, num_stocks)

    res = [
        {
            'Ticker':     c_stocks[i].ticker.decode('utf-8'),
            'FinalScore': round(c_stocks[i].final_score, 4),
        }
        for i in range(num_stocks)
    ]
    t1_ffi = time.perf_counter()
    print(f"[PROFILER] C-FFI Scoring Execution: {t1_ffi - t0_ffi:.3f}s")
    return res
