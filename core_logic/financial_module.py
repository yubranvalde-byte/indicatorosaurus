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
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

# Configure PyArrow to use the system memory allocator and single-threaded I/O
# so temporary Parquet buffers are returned immediately to the OS instead of retained
# across multi-thread arenas.
try:
    if hasattr(pa, "set_memory_pool"):
        pa.set_memory_pool(pa.system_memory_pool())
    if hasattr(pa, "set_cpu_count"):
        pa.set_cpu_count(1)
    if hasattr(pa, "set_io_thread_count"):
        pa.set_io_thread_count(1)
except Exception:
    pass


_libc_malloc_trim = None
if os.name != "nt":
    try:
        _libc = ctypes.CDLL("libc.so.6")
        _libc.malloc_trim.argtypes = [ctypes.c_size_t]
        _libc.malloc_trim.restype = ctypes.c_int
        _libc_malloc_trim = _libc.malloc_trim
    except Exception:
        _libc_malloc_trim = None


def _release_arrow_memory(run_gc: bool = False):
    """Explicitly release unused PyArrow memory pool buffers and glibc heap arenas back to the OS."""
    if run_gc:
        gc.collect()
    try:
        pa.default_memory_pool().release_unused()
    except Exception:
        pass
    if _libc_malloc_trim is not None:
        try:
            _libc_malloc_trim(0)
        except Exception:
            pass


def discover_parquet_max_date(parquet_path: str) -> pd.Timestamp:
    """
    Discover the maximum Date in a Parquet file in O(1) memory (~0 MB RAM)
    by inspecting PyArrow ParquetFile row-group footer statistics without reading
    data pages into Pandas.
    """
    pf = pq.ParquetFile(parquet_path)
    md = pf.metadata
    max_dt = None
    for r in range(md.num_row_groups):
        rg = md.row_group(r)
        for c in range(rg.num_columns):
            col_meta = rg.column(c)
            if col_meta.path_in_schema == "Date" and col_meta.statistics and col_meta.statistics.has_min_max:
                val = pd.Timestamp(col_meta.statistics.max)
                if max_dt is None or val > max_dt:
                    max_dt = val
    if max_dt is not None:
        if max_dt.tz is not None:
            max_dt = max_dt.tz_localize(None)
        return max_dt.normalize()
    return pd.Timestamp.now().normalize()

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
    'latest_prices': None,
    'adtv_series': None,
    'tech_map': None,
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


def _get_universe_ticker_whitelist(base_dir: str) -> list[str]:
    """
    Return the deduplicated list of upper-case ticker symbols in the fundamental/market
    universe so PyArrow Parquet reads only decompress relevant equities (~1,866 tickers)
    instead of all 5,000+ tickers in the raw SimFin lake.
    """
    tickers: set[str] = set()
    cached_df = _LIVE_DATA_CACHE.get('data')
    if isinstance(cached_df, pd.DataFrame) and not cached_df.empty and 'Ticker' in cached_df.columns:
        tickers.update(cached_df['Ticker'].dropna().astype(str).str.upper().str.strip().tolist())

    for fname in ('historical_fundamental_data.csv', 'market_data.csv'):
        fpath = os.path.join(base_dir, '..', 'data', fname)
        if os.path.exists(fpath):
            try:
                t_df = pd.read_csv(fpath, usecols=['Ticker'])
                if not t_df.empty:
                    tickers.update(t_df['Ticker'].dropna().astype(str).str.upper().str.strip().tolist())
            except Exception:
                pass
    tickers.discard('')
    return sorted(tickers)


def load_parquet_price_matrix(
    parquet_path: str,
    tickers: list[str] | None = None,
    min_date: pd.Timestamp | None = None,
    include_volume: bool = False,
    tail_rows: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """
    Build a (Date x Ticker) float32 price matrix (and optional volume matrix) directly
    by streaming PyArrow Parquet row groups one at a time (~200k rows / ~3MB per batch).
    Maps Ticker symbols to integer column indices purely in C++ via `pyarrow.compute.index_in`
    without ever allocating millions of Python `str` objects or calling `DataFrame.pivot()`.
    """
    pf = pq.ParquetFile(parquet_path)
    md = pf.metadata

    if not tickers:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        tickers = _get_universe_ticker_whitelist(base_dir)
    sorted_tickers = sorted(set(str(t).strip().upper() for t in (tickers or []) if t))
    if not sorted_tickers:
        return pd.DataFrame(), (pd.DataFrame() if include_volume else None)

    n_tickers = len(sorted_tickers)
    pa_tickers = pa.array(sorted_tickers, type=pa.string())

    min_ts = pd.Timestamp(min_date).tz_localize(None) if min_date is not None else None
    min_dt64 = min_ts.to_datetime64() if min_ts is not None else None

    date_col_idx = None
    if md.num_row_groups > 0:
        rg0 = md.row_group(0)
        for c in range(rg0.num_columns):
            if rg0.column(c).path_in_schema == "Date":
                date_col_idx = c
                break

    cols_to_read = ['Date', 'Ticker', 'Adj. Close', 'Close']
    if include_volume:
        cols_to_read.append('Volume')

    date_blocks: list[np.ndarray] = []
    p_blocks: list[np.ndarray] = []
    v_blocks: list[np.ndarray] = []

    for r in range(md.num_row_groups):
        if min_ts is not None and date_col_idx is not None:
            col_meta = md.row_group(r).column(date_col_idx)
            if col_meta.statistics and col_meta.statistics.has_min_max:
                rg_max = pd.Timestamp(col_meta.statistics.max).tz_localize(None)
                if rg_max < min_ts:
                    continue

        tbl = pf.read_row_group(r, columns=cols_to_read)
        if tbl.num_rows == 0:
            del tbl
            continue

        col_idx_arr = pc.index_in(tbl['Ticker'], value_set=pa_tickers)
        valid_mask = pc.is_valid(col_idx_arr)
        tbl = tbl.filter(valid_mask)
        col_idx_arr = col_idx_arr.filter(valid_mask)
        del valid_mask

        if tbl.num_rows == 0:
            del tbl, col_idx_arr
            continue

        dates_np = tbl['Date'].to_numpy(zero_copy_only=False)
        col_idx_np = col_idx_arr.to_numpy(zero_copy_only=False).astype(np.int32, copy=False)
        adj_np = tbl['Adj. Close'].to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
        close_np = tbl['Close'].to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
        price_np = np.where(np.isnan(adj_np), close_np, adj_np)
        if include_volume:
            vol_np = np.nan_to_num(
                tbl['Volume'].to_numpy(zero_copy_only=False).astype(np.float32, copy=False),
                nan=0.0,
            )
        else:
            vol_np = None

        del tbl, col_idx_arr, adj_np, close_np

        if min_dt64 is not None and len(dates_np) > 0 and dates_np[0] < min_dt64:
            d_mask = dates_np >= min_dt64
            if not d_mask.any():
                del dates_np, col_idx_np, price_np, vol_np
                continue
            dates_np = dates_np[d_mask]
            col_idx_np = col_idx_np[d_mask]
            price_np = price_np[d_mask]
            if include_volume and vol_np is not None:
                vol_np = vol_np[d_mask]
            del d_mask

        rg_dates, inv_row = np.unique(dates_np, return_inverse=True)
        rg_p_mat = np.full((len(rg_dates), n_tickers), np.nan, dtype=np.float32)
        rg_p_mat[inv_row, col_idx_np] = price_np
        if include_volume and vol_np is not None:
            rg_v_mat = np.full((len(rg_dates), n_tickers), 0.0, dtype=np.float32)
            rg_v_mat[inv_row, col_idx_np] = vol_np
        else:
            rg_v_mat = None

        del dates_np, col_idx_np, price_np, vol_np, inv_row

        if date_blocks and rg_dates[0] == date_blocks[-1][-1]:
            valid_p = ~np.isnan(rg_p_mat[0])
            p_blocks[-1][-1, valid_p] = rg_p_mat[0, valid_p]
            if include_volume and rg_v_mat is not None:
                valid_v = rg_v_mat[0] > 0.0
                v_blocks[-1][-1, valid_v] = rg_v_mat[0, valid_v]
            rg_dates = rg_dates[1:]
            rg_p_mat = rg_p_mat[1:]
            if include_volume and rg_v_mat is not None:
                rg_v_mat = rg_v_mat[1:]

        if len(rg_dates) > 0:
            date_blocks.append(rg_dates)
            p_blocks.append(rg_p_mat)
            if include_volume and rg_v_mat is not None:
                v_blocks.append(rg_v_mat)

    del pf, md, pa_tickers
    if not date_blocks:
        _release_arrow_memory()
        return pd.DataFrame(), (pd.DataFrame() if include_volume else None)

    full_dates = pd.DatetimeIndex(np.concatenate(date_blocks)).tz_localize(None)
    full_p_mat = np.vstack(p_blocks)
    del date_blocks, p_blocks
    if include_volume:
        full_v_mat = np.vstack(v_blocks)
        del v_blocks
    else:
        full_v_mat = None

    if not full_dates.is_monotonic_increasing:
        order = np.argsort(full_dates.values)
        full_dates = full_dates[order]
        full_p_mat = full_p_mat[order]
        if full_v_mat is not None:
            full_v_mat = full_v_mat[order]

    if tail_rows is not None and len(full_dates) > tail_rows:
        full_dates = full_dates[-tail_rows:]
        full_p_mat = full_p_mat[-tail_rows:]
        if full_v_mat is not None:
            full_v_mat = full_v_mat[-tail_rows:]

    valid_cols = ~np.isnan(full_p_mat).all(axis=0)
    if not valid_cols.all():
        active_tickers = [t for t, keep in zip(sorted_tickers, valid_cols) if keep]
        full_p_mat = np.ascontiguousarray(full_p_mat[:, valid_cols], dtype=np.float32)
        if full_v_mat is not None:
            full_v_mat = np.ascontiguousarray(full_v_mat[:, valid_cols], dtype=np.float32)
    else:
        active_tickers = sorted_tickers

    price_df = pd.DataFrame(full_p_mat, index=full_dates, columns=active_tickers, copy=False)
    price_df.index.name = 'Date'
    price_df.columns.name = 'Ticker'

    if include_volume and full_v_mat is not None:
        volume_df = pd.DataFrame(full_v_mat, index=full_dates, columns=active_tickers, copy=False)
        volume_df.index.name = 'Date'
        volume_df.columns.name = 'Ticker'
    else:
        volume_df = None

    _release_arrow_memory()
    return price_df, volume_df


def ensure_tech_cache():
    """
    Hydrates _TECH_DATA_CACHE from us-shareprices-daily.parquet and spy_historical.csv if missing or expired (30m TTL).
    Strictly bounds memory (<512MB Render limit) by:
      1. Discovering the Parquet max date from footer statistics in O(1) memory (~0 MB).
      2. Streaming matching row groups via `load_parquet_price_matrix` (zero Python string explosion, zero DataFrame.pivot).
      3. Pre-computing 1D Series (latest_prices, adtv_series) and tech_map immediately,
         then freeing the 2D price/volume matrices and releasing Arrow pool memory.
    Returns tuple of (price_df, volume_df, spy_s).
    """
    global _TECH_DATA_CACHE
    now = time.time()

    if (_TECH_DATA_CACHE.get('tech_map') is not None and
        _TECH_DATA_CACHE.get('latest_prices') is not None and
        _TECH_DATA_CACHE.get('adtv_series') is not None and
        _TECH_DATA_CACHE.get('spy_s') is not None and
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
        # Discover terminal date from Parquet footer metadata in O(1) RAM without loading 6.2M rows
        max_ds_date = discover_parquet_max_date(simfin_parquet_path)
        wall_now = pd.Timestamp.now().normalize()
        effective_end = min(wall_now, max_ds_date)
        min_date = effective_end - pd.Timedelta(days=370)

        ticker_whitelist = _get_universe_ticker_whitelist(base_dir)
        full_price_df, full_volume_df = load_parquet_price_matrix(
            simfin_parquet_path,
            tickers=ticker_whitelist,
            min_date=min_date,
            include_volume=True,
            tail_rows=260,
        )
        if full_price_df.empty:
            _release_arrow_memory()
            return None, None, None

        # Load SPY benchmark history
        spy_df = pd.read_csv(spy_path, usecols=['Date', 'SPY_Adj_Close'])
        spy_df['Date'] = pd.to_datetime(spy_df['Date'])
        spy_full = spy_df.set_index('Date')['SPY_Adj_Close'].astype('float32').sort_index()
        del spy_df

        price_start_date = full_price_df.index[0]
        price_end_date = full_price_df.index[-1]
        full_spy_s = spy_full[(spy_full.index >= price_start_date) & (spy_full.index <= price_end_date)]
        if len(full_spy_s) < 2:
            full_spy_s = spy_full.tail(260)
        del spy_full

        latest_prices = full_price_df.ffill(limit=5).iloc[-1].astype('float32').rename('LatestPrice')
        adtv_series = (full_price_df.tail(20) * full_volume_df.tail(20)).mean(axis=0).astype('float32').rename('ADTV')

        # Pre-compute vectorized technical indicators immediately so 2D pivot matrices can be freed
        rel_strength, price_vs_sma, rsi_14, beta_s, _ = compute_vectorized_tech_series(
            full_price_df, full_spy_s, log_diag=True
        )
        rs_arr = np.nan_to_num(rel_strength.to_numpy(dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        sma_arr = np.nan_to_num(price_vs_sma.to_numpy(dtype=np.float64), nan=1.0, posinf=1.0, neginf=1.0)
        rsi_arr = np.nan_to_num(rsi_14.to_numpy(dtype=np.float64), nan=50.0, posinf=50.0, neginf=50.0)
        beta_arr = np.nan_to_num(beta_s.to_numpy(dtype=np.float64), nan=1.0, posinf=1.0, neginf=1.0)

        full_map = {}
        for idx, ticker in enumerate(rel_strength.index):
            full_map[ticker] = {
                'RelativeStrength': float(rs_arr[idx]),
                'PriceVs200SMA': float(sma_arr[idx]),
                'RSI': float(rsi_arr[idx]),
                'Beta': float(beta_arr[idx]),
            }

        del full_price_df, full_volume_df
        _TECH_DATA_CACHE['price_df'] = None
        _TECH_DATA_CACHE['volume_df'] = None
        _TECH_DATA_CACHE['spy_s'] = full_spy_s
        _TECH_DATA_CACHE['latest_prices'] = latest_prices
        _TECH_DATA_CACHE['adtv_series'] = adtv_series
        _TECH_DATA_CACHE['tech_map'] = full_map
        _TECH_DATA_CACHE['timestamp'] = now
        _release_arrow_memory(run_gc=True)

        t1_io = time.perf_counter()
        print(f"[PROFILER] Technical Indicator Parquet/CSV I/O + Precompute (Cache Miss): {t1_io - t0_io:.3f}s")
        print(f"[Technical Indicators] Cache hydrated: Parquet {price_start_date.date()} -> {price_end_date.date()}, "
              f"SPY aligned {full_spy_s.index[0].date()} -> {full_spy_s.index[-1].date()} ({len(full_spy_s)} rows, {len(full_map)} tickers)")

        return None, None, full_spy_s
    except Exception as e:
        print(f"ERROR hydrating technical data cache: {e}")
        _release_arrow_memory()
        return None, None, None


def compute_vectorized_tech_series(price_df, spy_s, log_diag=False):
    """
    Pure vectorized technical indicator computation on a point-in-time price matrix `price_df`
    and aligned benchmark `spy_s` (strictly <= anchor date t_k).
    Uses direct float32 NumPy array math for daily returns and Beta covariance to avoid
    allocating multiple intermediate 2D Pandas DataFrames per rebalance anchor.
    Returns (rel_strength, price_vs_sma, rsi_14, beta_s, latest_px) as 1D Series indexed by Ticker.
    """
    cols = price_df.columns
    px_arr = price_df.to_numpy(dtype=np.float32, copy=False)
    n_rows, n_cols = px_arr.shape

    # Forward-fill up to 5 rows at the tail for latest_px, and backward-fill up to 5 rows at the head for first_px
    tail_k = min(6, n_rows)
    last_vals = px_arr[-1].copy()
    for step in range(2, tail_k + 1):
        nan_m = np.isnan(last_vals)
        if not nan_m.any():
            break
        last_vals[nan_m] = px_arr[-step, nan_m]
    last_px = pd.Series(last_vals, index=cols, dtype='float32')

    head_k = min(6, n_rows)
    first_vals = px_arr[0].copy()
    for step in range(1, head_k):
        nan_m = np.isnan(first_vals) | (first_vals == 0.0)
        if not nan_m.any():
            break
        first_vals[nan_m] = px_arr[step, nan_m]
    first_vals = np.where(first_vals == 0.0, np.nan, first_vals)

    import warnings
    # 1. Vectorized 1-Year Relative Strength vs SPY (same time window as stocks)
    spy_1y_ret = float((spy_s.iloc[-1] - spy_s.iloc[0]) / spy_s.iloc[0]) if len(spy_s) > 1 and spy_s.iloc[0] != 0 else 0.0
    with warnings.catch_warnings(), np.errstate(divide='ignore', invalid='ignore'):
        warnings.simplefilter("ignore", category=RuntimeWarning)
        stk_1y_ret = (last_vals - first_vals) / first_vals
        rel_strength = pd.Series((stk_1y_ret - spy_1y_ret).astype(np.float32), index=cols, dtype='float32')

        # 2. Vectorized Price vs 200-day Simple Moving Average
        sma_200 = np.nanmean(px_arr[-200:], axis=0)
        sma_200 = np.where(sma_200 == 0.0, np.nan, sma_200)
        price_vs_sma = pd.Series((last_vals / sma_200).astype(np.float32), index=cols, dtype='float32')

    if log_diag:
        valid_sma = price_vs_sma.dropna()
        pos_sma_c = int((valid_sma > 0.0).sum())
        zero_sma_c = len(price_vs_sma) - pos_sma_c
        sample_sma = [round(float(v), 4) for v in valid_sma.iloc[:5]] if not valid_sma.empty else []
        print(f"[DIAG] PriceVs200SMA statistics: Total tickers={len(price_vs_sma)}, > 0.0 count={pos_sma_c}, <= 0.0 / NaN count={zero_sma_c}, Sample={sample_sma}")

    # 3. Vectorized 14-day RSI
    tail_15 = px_arr[-15:]
    if len(tail_15) >= 2:
        with warnings.catch_warnings(), np.errstate(divide='ignore', invalid='ignore'):
            warnings.simplefilter("ignore", category=RuntimeWarning)
            delta = np.diff(tail_15, axis=0)[-14:]
            gain = np.mean(np.where(delta > 0.0, delta, np.float32(0.0)), axis=0)
            loss = np.mean(np.where(delta < 0.0, -delta, np.float32(0.0)), axis=0)
            rs = gain / (loss + np.float32(1e-6))
            rsi_vals = (100.0 - (100.0 / (1.0 + rs))).astype(np.float32)
    else:
        rsi_vals = np.full(n_cols, 50.0, dtype=np.float32)
    rsi_14 = pd.Series(rsi_vals, index=cols, dtype='float32')

    # 4. Vectorized 1-Year Beta vs SPY (aligned daily returns via NumPy dot-product, zero 2D DataFrame copies)
    spy_returns = spy_s.pct_change(fill_method=None).reindex(price_df.index)
    spy_valid = spy_returns.dropna()
    spy_var = float(spy_valid.var()) if len(spy_valid) > 1 else 0.0

    if spy_var > 0.0 and n_rows >= 2:
        spy_dm = (spy_returns - spy_returns.mean()).fillna(0.0).to_numpy(dtype=np.float32)
        spy_dm = np.nan_to_num(spy_dm[1:], nan=0.0, posinf=0.0, neginf=0.0)

        with warnings.catch_warnings(), np.errstate(divide='ignore', invalid='ignore'):
            warnings.simplefilter("ignore", category=RuntimeWarning)
            prev_px = np.where(px_arr[:-1] == 0.0, np.nan, px_arr[:-1])
            stk_ret = (px_arr[1:] - px_arr[:-1]) / prev_px
            stk_mean = np.nanmean(stk_ret, axis=0, keepdims=True)
            stk_dm = np.nan_to_num(stk_ret - stk_mean, nan=0.0, posinf=0.0, neginf=0.0)

        n_obs = max(1, len(spy_returns) - 1)
        cov_vals = np.dot(stk_dm.T, spy_dm) / n_obs
        cov_s = pd.Series(cov_vals, index=cols, dtype='float32')
        beta_s = (cov_s / spy_var).fillna(1.0)
        del stk_ret, stk_dm
    else:
        beta_s = pd.Series(1.0, index=cols, dtype='float32')

    return (
        rel_strength.fillna(0.0),
        price_vs_sma.fillna(1.0),
        rsi_14.fillna(50.0),
        beta_s.fillna(1.0),
        last_px,
    )


def compute_live_technical_indicators(unique_tickers):
    """
    Dynamically calculates momentum & technical indicators (RelativeStrength, PriceVs200SMA, RSI, Beta)
    from offline us-shareprices-daily.parquet and spy_historical.csv for the active stock universe.
    Uses in-memory cached indicator map after initial hydration for O(1) sub-millisecond retrieval.
    """
    if not unique_tickers:
        return {}

    tech_map = {}
    try:
        ensure_tech_cache()
        cached_map = _TECH_DATA_CACHE.get('tech_map')
        if cached_map is not None:
            return {t: cached_map[t] for t in unique_tickers if t in cached_map}
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

_COMPANY_NAMES_CACHE = None

def _get_company_names_map() -> dict:
    global _COMPANY_NAMES_CACHE
    if _COMPANY_NAMES_CACHE is not None:
        return _COMPANY_NAMES_CACHE
    names = {}
    base_dir = os.path.dirname(os.path.abspath(__file__))
    mkt_path = os.path.join(base_dir, "..", "data", "market_data.csv")
    if os.path.exists(mkt_path):
        try:
            m_df = pd.read_csv(mkt_path, usecols=["Ticker", "LongName"])
            for t, n in zip(m_df["Ticker"], m_df["LongName"]):
                if pd.notna(n) and str(n).strip():
                    names[str(t).strip().upper()] = str(n).strip()
        except Exception:
            pass
    comp_path = os.path.join(base_dir, "..", "data", "raw_simfin", "us-companies.csv")
    if os.path.exists(comp_path):
        try:
            c_df = pd.read_csv(comp_path, sep=";", usecols=["Ticker", "Company Name"])
            c_df = c_df.dropna(subset=["Ticker"])
            for t, n in zip(c_df["Ticker"], c_df["Company Name"]):
                t_str = str(t).strip().upper()
                if t_str not in names and pd.notna(n) and str(n).strip():
                    names[t_str] = str(n).strip()
        except Exception:
            pass
    _COMPANY_NAMES_CACHE = names
    return _COMPANY_NAMES_CACHE

def load_latest_fundamental_snapshot(csv_path):
    """
    Ingests historical_fundamental_data.csv (or market_data.csv as fallback),
    extracts the latest cross-sectional snapshot per Ticker, enriches with LongName,
    and returns a pandas DataFrame.
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
            # Drop severely outdated/delisted corporate filings (older than 2023-01-01)
            df = df[df["Quarter_End_Date"] >= "2023-01-01"].copy()
            df["Quarter_End_Date"] = df["Quarter_End_Date"].dt.strftime("%Y-%m-%d")
        
        # Ensure 'Sector' key exists (mapped from GICS_Sector if needed)
        if "GICS_Sector" in df.columns and "Sector" not in df.columns:
            df["Sector"] = df["GICS_Sector"]

        # Ensure 'LongName' key exists and is enriched from company reference data
        names_map = _get_company_names_map()
        t_clean = df["Ticker"].astype(str).str.strip().str.upper()
        if "LongName" not in df.columns:
            df["LongName"] = t_clean.map(names_map).fillna(df["Ticker"])
        else:
            # Overwrite placeholder LongNames (where LongName == Ticker) with real names
            mask_placeholder = df["LongName"].isna() | (df["LongName"] == df["Ticker"])
            if mask_placeholder.any():
                df.loc[mask_placeholder, "LongName"] = t_clean[mask_placeholder].map(names_map).fillna(df.loc[mask_placeholder, "Ticker"])
            
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

# Vectorized NumPy structured dtype matching the exact 80-byte C Stock memory layout
STOCK_STRUCT_DTYPE = np.dtype([
    ('ticker', 'S8'),
    ('indicators', '<f4', (NUM_INDICATORS,)),
    ('final_score', '<f8'),
])
assert STOCK_STRUCT_DTYPE.itemsize == 80, f"STOCK_STRUCT_DTYPE size mismatch: expected 80 bytes, got {STOCK_STRUCT_DTYPE.itemsize}"


# ─── 2. Library Loading & Function Signatures ──────────────────────────────────

_c_kernel_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'c_kernel')
_lib_candidates = (
    [os.path.join(_c_kernel_dir, 'libsortbuslines.dll'), os.path.join(_c_kernel_dir, 'libsortbuslines.so')]
    if os.name == 'nt'
    else [os.path.join(_c_kernel_dir, 'libsortbuslines.so'), os.path.join(_c_kernel_dir, 'libsortbuslines.dll')]
)
lib_path = next((p for p in _lib_candidates if os.path.exists(p)), _lib_candidates[0])
lib = None
for _cand_path in _lib_candidates:
    if not os.path.exists(_cand_path):
        continue
    try:
        lib = ctypes.CDLL(_cand_path)
        lib_path = _cand_path
        # void compute_scores(Stock* stocks, int num_stocks, const double* weights);
        lib.compute_scores.argtypes = [ctypes.POINTER(Stock), ctypes.c_int, ctypes.POINTER(ctypes.c_double)]
        lib.compute_scores.restype  = None

        # void sort_stocks_by_score(Stock* stocks, int num_stocks);
        lib.sort_stocks_by_score.argtypes = [ctypes.POINTER(Stock), ctypes.c_int]
        lib.sort_stocks_by_score.restype  = None
        break
    except OSError:
        lib = None


def apply_early_pruning_vectorized(df, strategy, allowed_sectors=None, min_market_cap=None, max_market_cap=None, specific_tickers=None, use_live_tech_cache=True):
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

    # Market Cap Filter: respect explicit user min_market_cap, or default to structural $1B floor
    if min_market_cap is not None:
        m_cap_min = float(min_market_cap)
    elif max_market_cap is not None and float(max_market_cap) < GATEKEEPER_MIN_MARKET_CAP:
        m_cap_min = 0.0
    else:
        m_cap_min = GATEKEEPER_MIN_MARKET_CAP
    df = df[df['MarketCap'].fillna(0.0) >= m_cap_min]
    if max_market_cap is not None:
        df = df[df['MarketCap'].fillna(0.0) <= float(max_market_cap)]

    # Point-in-Time vs Live Price and ADTV Gatekeeper
    if 'LatestPrice' in df.columns:
        df = df[df['LatestPrice'].fillna(0.0) > GATEKEEPER_MIN_PRICE]
        if 'ADTV' in df.columns:
            df = df[df['ADTV'].fillna(0.0) >= GATEKEEPER_MIN_ADTV]
        df = df.drop(columns=[c for c in ('LatestPrice', 'ADTV') if c in df.columns])
    elif use_live_tech_cache:
        if _TECH_DATA_CACHE.get('latest_prices') is None or _TECH_DATA_CACHE.get('adtv_series') is None:
            ensure_tech_cache()
        latest_prices = _TECH_DATA_CACHE.get('latest_prices')
        adtv_series = _TECH_DATA_CACHE.get('adtv_series')

        if latest_prices is not None and adtv_series is not None:
            lp_mapped = df['Ticker'].map(latest_prices).fillna(0.0)
            adtv_mapped = df['Ticker'].map(adtv_series).fillna(0.0)
            df = df[(lp_mapped > GATEKEEPER_MIN_PRICE) & (adtv_mapped >= GATEKEEPER_MIN_ADTV)]

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
    if isinstance(val, (int, float, np.number)):
        f = float(val)
        return default if (math.isnan(f) or math.isinf(f)) else f
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
    if isinstance(val, (int, float, np.number)):
        f = float(val)
        return default if (math.isnan(f) or math.isinf(f)) else f
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


def normalize_stocks(stocks_list, strategy=None, allowed_sectors=None, min_market_cap=None, max_market_cap=None, global_stocks_list=None, use_live_tech_cache=True, return_df=False):
    """
    Apply dynamic filters, strategy knock-out filters, then normalize the surviving universe
    using vectorized piecewise Gaussian (Goldilocks) or robust Sigmoid Z-score curves.

    Args:
        stocks_list:     list of raw CSV dicts (or a pandas DataFrame).
        strategy:        one of VALID_STRATEGIES, or None for no filtering.
        allowed_sectors: list of allowed sector names, or None.
        min_market_cap:  minimum market cap, or None.
        max_market_cap:  maximum market cap, or None.
        global_stocks_list: list of all raw CSV dicts (or DataFrame) for sector neutrality baseline.
        use_live_tech_cache: whether to apply live _TECH_DATA_CACHE Gatekeeper when LatestPrice is absent.
        return_df:       if True, returns the normalized pandas DataFrame directly instead of list[dict].
    """
    if isinstance(stocks_list, pd.DataFrame):
        if stocks_list.empty:
            return pd.DataFrame() if return_df else []
        df_initial = stocks_list.copy()
    else:
        if not stocks_list:
            return pd.DataFrame() if return_df else []
        df_initial = pd.DataFrame(stocks_list)

    if 'Sector' not in df_initial.columns:
        df_initial['Sector'] = df_initial['GICS_Sector'] if 'GICS_Sector' in df_initial.columns else ''

    # Sector-Neutral Baseline Stabilization: Use the entire global population prior to filtering
    if global_stocks_list is None or (global_stocks_list is stocks_list):
        df_base = df_initial
    elif isinstance(global_stocks_list, pd.DataFrame):
        df_base = global_stocks_list
    else:
        df_base = pd.DataFrame(global_stocks_list)
        if 'Sector' not in df_base.columns:
            df_base['Sector'] = df_base['GICS_Sector'] if 'GICS_Sector' in df_base.columns else ''

    df_stocks = df_initial
    if strategy or allowed_sectors or (min_market_cap is not None) or (max_market_cap is not None) or ('LatestPrice' in df_stocks.columns):
        df_stocks = apply_early_pruning_vectorized(
            df_stocks,
            strategy,
            allowed_sectors,
            min_market_cap,
            max_market_cap,
            use_live_tech_cache=use_live_tech_cache,
        )
        df_stocks = apply_derived_knockouts_vectorized(df_stocks, strategy)
        if df_stocks.empty:
            return pd.DataFrame() if return_df else []

    n = len(df_stocks)
    t0_norm = time.perf_counter()

    # Step 3: Vectorized extraction of raw indicator float matrix (N x 16)
    for col in INDICATOR_COLUMNS:
        if col not in df_stocks.columns:
            df_stocks[col] = 0.0
        elif not pd.api.types.is_numeric_dtype(df_stocks[col]):
            df_stocks[col] = pd.to_numeric(df_stocks[col], errors='coerce')

    raw_matrix = df_stocks[INDICATOR_COLUMNS].fillna(0.0).to_numpy(dtype=np.float32)
    raw_matrix = np.nan_to_num(raw_matrix, nan=0.0, posinf=0.0, neginf=0.0)

    norm_matrix = np.zeros((n, len(INDICATOR_COLUMNS)), dtype=np.float32)

    # Pre-extract sector array for baseline and current df_stocks for sector-neutral metrics
    base_sects = df_base['Sector'].fillna('').astype(str).str.strip().str.lower().to_numpy()
    stock_sects = df_stocks['Sector'].fillna('').astype(str).str.strip().str.lower().to_numpy()
    unique_base_sects = np.unique(base_sects)

    if 'EBITDA' in df_stocks.columns:
        ebitda_s = df_stocks['EBITDA'] if pd.api.types.is_numeric_dtype(df_stocks['EBITDA']) else pd.to_numeric(df_stocks['EBITDA'], errors='coerce')
        ebitda_arr = np.nan_to_num(ebitda_s.fillna(0.0).to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    else:
        ebitda_arr = np.zeros(n, dtype=np.float32)

    # Step 4: Normalize each column dynamically using vectorized math
    for j, col in enumerate(INDICATOR_COLUMNS):
        col_arr = raw_matrix[:, j]

        if col == 'EV_EBITDA':
            params = GOLDILOCKS_PARAMS[col]
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
                if col in df_base.columns:
                    b_s = df_base[col] if pd.api.types.is_numeric_dtype(df_base[col]) else pd.to_numeric(df_base[col], errors='coerce')
                    base_col_vals = np.nan_to_num(b_s.fillna(0.0).to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
                else:
                    base_col_vals = np.zeros(len(df_base), dtype=np.float32)

                meds = np.zeros(n, dtype=np.float32)
                mads = np.zeros(n, dtype=np.float32)
                for sect in unique_base_sects:
                    mask = (base_sects == sect)
                    s_vals = base_col_vals[mask]
                    if len(s_vals) > 0:
                        med = float(np.median(s_vals))
                        mad = float(np.median(np.abs(s_vals - med)))
                    else:
                        med, mad = 0.0, 0.0
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

    # Step 5: Rebuild stock dicts (or return DataFrame directly) with normalized values
    norm_df = pd.DataFrame(norm_matrix, columns=INDICATOR_COLUMNS, copy=False)
    norm_df.insert(0, 'Ticker', df_stocks['Ticker'].astype(str).values)

    t1_norm = time.perf_counter()
    print(f"[PROFILER] Normalization Math (Vectorized NumPy): {t1_norm - t0_norm:.3f}s")

    if return_df:
        return norm_df
    return norm_df.to_dict(orient='records')


# ─── 4. Main FFI Entry Point ───────────────────────────────────────────────────

def run_financial_engine(normalized_stocks, weights_list):
    """
    Accepts pre-normalized stocks (either a pandas DataFrame or list[dict]) directly to bypass duplicate normalization.
    Packs contiguous 80-byte STOCK_STRUCT_DTYPE buffer → C dot-product → QuickSort.
    Strictly enforces 7-byte ASCII + null terminator ('\\0' at byte 7) for C `char ticker[8]`
    safety so `strcmp` in C never reads out-of-bounds, while restoring full ticker names on output.
    """
    t0_ffi = time.perf_counter()
    if isinstance(normalized_stocks, pd.DataFrame):
        if normalized_stocks.empty:
            return []
        num_stocks = len(normalized_stocks)
        ticker_seq = normalized_stocks['Ticker'].astype(str).tolist()
        ind_mat = normalized_stocks[INDICATOR_COLUMNS].to_numpy(dtype='<f4', copy=False)
        ind_mat = np.nan_to_num(ind_mat, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        num_stocks = len(normalized_stocks)
        if num_stocks == 0:
            return []
        ticker_seq = [str(s.get('Ticker', '')).strip() for s in normalized_stocks]
        ind_mat = np.array(
            [[_safe_float(s.get(col, 0.0), 0.0) for col in INDICATOR_COLUMNS] for s in normalized_stocks],
            dtype='<f4',
        )
        ind_mat = np.nan_to_num(ind_mat, nan=0.0, posinf=0.0, neginf=0.0)

    # ── Step 1: Pack contiguous 80-byte Stock buffer via NumPy structured dtype ──
    c_buf = np.zeros(num_stocks, dtype=STOCK_STRUCT_DTYPE)
    c_buf = np.ascontiguousarray(c_buf, dtype=STOCK_STRUCT_DTYPE)

    trunc_bytes_list = []
    full_ticker_lookup = {}
    for full_t in ticker_seq:
        # Strictly encode as ASCII and truncate to 7 bytes max so byte 7 in S8 is guaranteed '\x00'
        b7 = full_t.encode('ascii', errors='ignore')[:7]
        trunc_bytes_list.append(b7)
        t_key = b7.decode('ascii', errors='ignore')
        if t_key not in full_ticker_lookup:
            full_ticker_lookup[t_key] = full_t

    c_buf['ticker'] = trunc_bytes_list
    c_buf['indicators'] = ind_mat

    clean_weights = [_safe_float(w, 0.0) for w in weights_list][:NUM_INDICATORS]
    if len(clean_weights) < NUM_INDICATORS:
        clean_weights.extend([0.0] * (NUM_INDICATORS - len(clean_weights)))

    if lib is None:
        # Vectorized NumPy fallback on the exact 80-byte structured buffer when host OS lacks C shared library
        w_arr = np.asarray(clean_weights, dtype='<f8')
        c_buf['final_score'] = np.dot(c_buf['indicators'].astype('<f8'), w_arr)
        order = np.argsort(-c_buf['final_score'], kind='mergesort')
        c_buf = c_buf[order]
        res = []
        for i in range(num_stocks):
            t_key = bytes(c_buf['ticker'][i]).split(b'\x00', 1)[0].decode('ascii', errors='ignore')
            res.append({
                'Ticker':     full_ticker_lookup.get(t_key, t_key),
                'FinalScore': _safe_float(round(float(c_buf['final_score'][i]), 4), 0.0),
            })
        t1_ffi = time.perf_counter()
        print(f"[PROFILER] C-FFI Scoring Execution (Vectorized Fallback): {t1_ffi - t0_ffi:.3f}s")
        return res

    # ── Step 2: Build weights array & pass contiguous 80-byte buffer to C kernel ──
    WeightsArray = ctypes.c_double * NUM_INDICATORS
    c_weights    = WeightsArray(*clean_weights)

    start_ptr   = ctypes.cast(c_buf.ctypes.data, ctypes.POINTER(Stock))
    weights_ptr = ctypes.cast(c_weights, ctypes.POINTER(ctypes.c_double))
    n_c         = ctypes.c_int(num_stocks)

    # Hold explicit references to c_buf and c_weights for the entire duration of the C FFI calls
    lib.compute_scores(start_ptr, n_c, weights_ptr)
    lib.sort_stocks_by_score(start_ptr, n_c)

    res = []
    for i in range(num_stocks):
        t_key = bytes(c_buf['ticker'][i]).split(b'\x00', 1)[0].decode('ascii', errors='ignore')
        res.append({
            'Ticker':     full_ticker_lookup.get(t_key, t_key),
            'FinalScore': _safe_float(round(float(c_buf['final_score'][i]), 4), 0.0),
        })
    t1_ffi = time.perf_counter()
    print(f"[PROFILER] C-FFI Scoring Execution: {t1_ffi - t0_ffi:.3f}s")
    return res
