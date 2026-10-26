"""
price_resolver.py
=================
Portfolio Intelligence — Phase 1:
Tiered Hybrid Historical Price & Macro-Index (USD/ILS FX & Israeli CPI) Resolution Engine
with local Snappy-compressed Parquet caching and an in-memory TTL cache.

Architecture:
1. Tier 1 (Primary Local Lake): `us-shareprices-daily.parquet` (SimFin) via PyArrow predicate pushdown
   (`Ticker` whitelist + `Date` bounds), matching `backtest_engine.py`.
2. Tier 2 (Local Portfolio Cache): `data/portfolio_price_cache.parquet` for ETFs (QQQ, SMH, SPY),
   TASE securities (`.TA`), promoted Tier 1 slices, and date ranges missing from Tier 1.
3. Tier 3 (External Fallback & Cache Hydration): Tiingo REST API (if `TIINGO_API_KEY` is set) -> `yfinance`.
   TASE Agorot quotes (`ILA`) are normalized to ILS (Price_ILS = Price_Agorot / 100.0) before caching.
4. Macro Engine (`data/fx_cpi_cache.parquet`):
   - `USD_ILS`: Bank of Israel official representative rate (SDMX series `RER_USD_ILS`).
   - `IL_CPI`: CBS general CPI (index id 120010), chain-linked to the latest CBS base and keyed to its
     PUBLICATION date (15th of month M+1, "the known index" / המדד הידוע) before chronological `.ffill()`.
   - Continuous daily calendar grid from 2010-01-01 to today. Zero look-ahead bias.

Data integrity: no synthetic or hardcoded price/FX/CPI data. If data cannot be resolved from the
local caches or the authoritative sources, a RuntimeError is raised.
"""

import gc
import io
import os
import re
import time
import requests
import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    yf = None

# ─── Constants & Paths ────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

TIER1_PARQUET_CANDIDATES = [
    os.path.join(DATA_DIR, "us-shareprices-daily.parquet"),
    os.path.join(DATA_DIR, "raw_simfin", "us-shareprices-daily.parquet"),
]
PORTFOLIO_CACHE_PATH = os.path.join(DATA_DIR, "portfolio_price_cache.parquet")
FX_CPI_CACHE_PATH = os.path.join(DATA_DIR, "fx_cpi_cache.parquet")

MACRO_ORIGIN_DATE = pd.Timestamp("2010-01-01")
CACHE_TTL_SECONDS = 1800.0      # 30-minute TTL aligned with core_logic/financial_module.py
MAX_MACRO_STALE_DAYS = 10       # Max days a stale macro cache may be forward-extended if sources are unreachable
COVERAGE_TOLERANCE_DAYS = 7     # Weekend/holiday tolerance when checking cached window coverage
HTTP_TIMEOUT = 30
HTTP_RETRIES = 3
HTTP_HEADERS = {"User-Agent": "Mozilla/5.0 (AlphaQuant price_resolver)", "Accept": "application/json,text/csv,*/*"}

BOI_USD_ILS_URL = (
    "https://edge.boi.gov.il/FusionEdgeServer/sdmx/v2/data/dataflow/BOI.STATISTICS/EXR/1.0/RER_USD_ILS"
    "?startperiod={start}&endperiod={end}&format=csv"
)
CBS_CPI_URL = (
    "https://api.cbs.gov.il/index/data/price?id=120010&format=json&download=false"
    "&startPeriod={start}&endPeriod={end}&PageSize=1000&lang=en"
)

LONG_COLUMNS = ["Date", "Ticker", "Close", "Currency"]

# ─── In-Memory TTL Caches ─────────────────────────────────────────────────────

_PRICE_RESOLVER_CACHE = {
    "entries": {},  # key: (tuple(sorted_tickers), start_str, end_str, dtype) -> (df, currency_map, timestamp)
}

_MACRO_RESOLVER_CACHE = {
    "full_df": None,
    "timestamp": 0.0,
}


# ─── Helper Utilities ─────────────────────────────────────────────────────────

def _to_naive_timestamp(dt_val) -> pd.Timestamp:
    """Convert any date input to a timezone-naive normalized pd.Timestamp."""
    ts = pd.Timestamp(dt_val)
    if ts.tz is not None:
        ts = ts.tz_localize(None)
    return ts.normalize()


def _today() -> pd.Timestamp:
    return pd.Timestamp.now().normalize()


def _get_tier1_parquet_path():
    """Return the first existing Tier 1 SimFin Parquet file path."""
    return next((p for p in TIER1_PARQUET_CANDIDATES if os.path.exists(p)), None)


def _infer_native_currency(ticker: str) -> str:
    """Return native currency ('ILS' for .TA tickers, 'USD' otherwise)."""
    return "ILS" if str(ticker).upper().strip().endswith(".TA") else "USD"


def _http_get(url: str) -> requests.Response:
    """GET with bounded retries. Raises RuntimeError after the final failed attempt."""
    last_err = None
    for attempt in range(1, HTTP_RETRIES + 1):
        resp = None
        try:
            resp = requests.get(url, headers=HTTP_HEADERS, timeout=HTTP_TIMEOUT)
            if resp.status_code == 200:
                return resp
            last_err = f"HTTP {resp.status_code}"
        except Exception as e:
            last_err = str(e)
        if resp is not None:
            resp.close()
        time.sleep(0.5 * attempt)
    raise RuntimeError(f"GET failed after {HTTP_RETRIES} attempts ({last_err}): {url}")


def _normalize_tase_agorot_series(series: pd.Series, ticker: str, reported_currency=None) -> pd.Series:
    """
    Normalize Tel Aviv Stock Exchange (.TA) quotes from Agorot (ILA) to ILS (1 ILS = 100 Agorot).
    - reported 'ILA'  -> divide by 100.0
    - reported 'ILS'  -> unchanged
    - unknown         -> heuristic (median > 200 => Agorot) with an explicit warning
    """
    s = pd.to_numeric(series, errors="coerce").astype("float64")
    if s.empty or not str(ticker).upper().strip().endswith(".TA"):
        return s

    curr_flag = str(reported_currency or "").strip().upper()
    if curr_flag == "ILA":
        return s / 100.0
    if curr_flag == "ILS":
        return s

    valid_vals = s.dropna()
    if valid_vals.empty:
        return s
    median_price = float(valid_vals.median())
    is_agorot = median_price > 200.0
    print(f"[PriceResolver WARNING] {ticker}: quote currency not reported; heuristic "
          f"(median={median_price:.2f}) -> {'Agorot, scaling /100' if is_agorot else 'ILS, unchanged'}.")
    return s / 100.0 if is_agorot else s


def _covered_tickers(long_df: pd.DataFrame, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> set:
    """
    Tickers whose cached rows span the requested window
    (first obs <= start + tolerance and last obs >= min(end, today) - tolerance).
    """
    if long_df is None or long_df.empty:
        return set()
    eff_end = min(end_ts, _today())
    tol = pd.Timedelta(days=COVERAGE_TOLERANCE_DAYS)
    bounds = long_df.groupby("Ticker")["Date"].agg(["min", "max"])
    ok = (bounds["min"] <= start_ts + tol) & (bounds["max"] >= eff_end - tol)
    return set(bounds.index[ok])


def clear_resolver_caches():
    """Clear in-memory TTL caches for both equity and macro resolvers."""
    _PRICE_RESOLVER_CACHE["entries"].clear()
    _MACRO_RESOLVER_CACHE["full_df"] = None
    _MACRO_RESOLVER_CACHE["timestamp"] = 0.0
    gc.collect()


# ─── Tier 1: Primary Local Lake (us-shareprices-daily.parquet) ────────────────

def _query_tier1_lake(us_tickers: list, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame:
    """
    Query `us-shareprices-daily.parquet` using PyArrow predicate pushdown filters (matching `backtest_engine.py`).
    Returns long-format DataFrame ['Date', 'Ticker', 'Close', 'Currency'].
    """
    parquet_path = _get_tier1_parquet_path()
    if not parquet_path or not us_tickers:
        return pd.DataFrame(columns=LONG_COLUMNS)

    try:
        raw_df = pd.read_parquet(
            parquet_path,
            engine="pyarrow",
            columns=["Date", "Ticker", "Adj. Close", "Close"],
            filters=[
                ("Ticker", "in", list(us_tickers)),
                ("Date", ">=", start_ts - pd.Timedelta(days=COVERAGE_TOLERANCE_DAYS)),
                ("Date", "<=", end_ts),
            ],
        )
    except Exception as e:
        print(f"[PriceResolver Tier 1 WARNING] Failed reading primary lake: {e}")
        return pd.DataFrame(columns=LONG_COLUMNS)

    if raw_df.empty:
        return pd.DataFrame(columns=LONG_COLUMNS)

    raw_df["Date"] = pd.to_datetime(raw_df["Date"]).dt.tz_localize(None).dt.normalize()
    raw_df["Ticker"] = raw_df["Ticker"].astype(str).str.upper().str.strip()
    raw_df["Close"] = raw_df["Adj. Close"].fillna(raw_df["Close"]).astype("float64")
    raw_df["Currency"] = "USD"
    out = raw_df[LONG_COLUMNS].dropna(subset=["Date", "Ticker", "Close"])
    del raw_df
    return out


# ─── Tier 2: Local Portfolio Cache (portfolio_price_cache.parquet) ────────────

def _query_tier2_cache(tickers: list, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame:
    """Query `data/portfolio_price_cache.parquet` via PyArrow pushdown filters."""
    if not os.path.exists(PORTFOLIO_CACHE_PATH) or not tickers:
        return pd.DataFrame(columns=LONG_COLUMNS)

    clean_tickers = list(dict.fromkeys(tickers))
    try:
        cached_df = pd.read_parquet(
            PORTFOLIO_CACHE_PATH,
            engine="pyarrow",
            columns=LONG_COLUMNS,
            filters=[
                ("Ticker", "in", clean_tickers),
                ("Date", ">=", start_ts - pd.Timedelta(days=COVERAGE_TOLERANCE_DAYS)),
                ("Date", "<=", end_ts),
            ],
        )
    except Exception as e:
        print(f"[PriceResolver Tier 2 WARNING] Failed reading portfolio price cache: {e}")
        return pd.DataFrame(columns=LONG_COLUMNS)

    # If any tickers are missing from the requested window because their observations are
    # older than start_ts (e.g. offline execution beyond cache max date), fetch their latest known observation
    found_tickers = set(cached_df["Ticker"]) if not cached_df.empty else set()
    missing_tickers = [t for t in clean_tickers if t not in found_tickers]
    if missing_tickers:
        try:
            prior_df = pd.read_parquet(
                PORTFOLIO_CACHE_PATH,
                engine="pyarrow",
                columns=LONG_COLUMNS,
                filters=[
                    ("Ticker", "in", missing_tickers),
                    ("Date", "<=", end_ts),
                ],
            )
            if not prior_df.empty:
                latest_prior = prior_df.sort_values("Date").groupby("Ticker").last().reset_index()
                cached_df = pd.concat([cached_df, latest_prior], ignore_index=True) if not cached_df.empty else latest_prior
        except Exception as e:
            print(f"[PriceResolver Tier 2 WARNING] Fallback prior read failed: {e}")

    if cached_df.empty:
        return pd.DataFrame(columns=LONG_COLUMNS)

    cached_df["Date"] = pd.to_datetime(cached_df["Date"]).dt.tz_localize(None).dt.normalize()
    cached_df["Ticker"] = cached_df["Ticker"].astype(str).str.upper().str.strip()
    cached_df["Close"] = pd.to_numeric(cached_df["Close"], errors="coerce").astype("float64")
    return cached_df.dropna(subset=["Date", "Ticker", "Close"])


def _append_to_tier2_cache(new_rows_df: pd.DataFrame) -> None:
    """
    Atomically merge price rows into `data/portfolio_price_cache.parquet` (Snappy),
    deduplicated on (Ticker, Date), newest write wins.
    """
    if new_rows_df is None or new_rows_df.empty:
        return

    os.makedirs(DATA_DIR, exist_ok=True)
    clean_new = new_rows_df[LONG_COLUMNS].copy()
    clean_new["Date"] = pd.to_datetime(clean_new["Date"]).dt.tz_localize(None).dt.normalize()
    clean_new["Ticker"] = clean_new["Ticker"].astype(str).str.upper().str.strip()
    clean_new["Close"] = pd.to_numeric(clean_new["Close"], errors="coerce").astype("float64")
    clean_new["Currency"] = clean_new["Currency"].astype(str).str.upper().str.strip()
    clean_new = clean_new.dropna(subset=["Date", "Ticker", "Close"])

    combined = clean_new
    if os.path.exists(PORTFOLIO_CACHE_PATH):
        try:
            existing_df = pd.read_parquet(PORTFOLIO_CACHE_PATH, engine="pyarrow")
            existing_df["Date"] = pd.to_datetime(existing_df["Date"]).dt.tz_localize(None).dt.normalize()
            combined = pd.concat([existing_df[LONG_COLUMNS], clean_new], ignore_index=True)
            del existing_df
        except Exception as e:
            print(f"[PriceResolver Tier 2 WARNING] Existing cache unreadable, rewriting: {e}")

    combined.sort_values(["Ticker", "Date"], inplace=True, kind="mergesort")
    combined.drop_duplicates(subset=["Ticker", "Date"], keep="last", inplace=True)
    combined.reset_index(drop=True, inplace=True)

    tmp_path = PORTFOLIO_CACHE_PATH + ".tmp"
    combined.to_parquet(tmp_path, engine="pyarrow", compression="snappy", index=False)
    os.replace(tmp_path, PORTFOLIO_CACHE_PATH)
    del combined
    gc.collect()


# ─── Tier 3: External Fallback (Tiingo API -> yfinance) ───────────────────────

def _fetch_tiingo_daily_history(ticker: str, start_ts: pd.Timestamp, end_ts: pd.Timestamp, api_key: str) -> pd.Series:
    """Fetch historical daily adjusted close prices from Tiingo REST API for US tickers."""
    if not api_key or ticker.endswith(".TA"):
        return pd.Series(dtype="float64")

    s_str = (start_ts - pd.Timedelta(days=COVERAGE_TOLERANCE_DAYS)).strftime("%Y-%m-%d")
    e_str = end_ts.strftime("%Y-%m-%d")
    url = f"https://api.tiingo.com/tiingo/daily/{ticker}/prices?startDate={s_str}&endDate={e_str}"
    headers = {"Content-Type": "application/json", "Authorization": f"Token {api_key}"}

    resp = None
    try:
        resp = requests.get(url, headers=headers, timeout=HTTP_TIMEOUT)
        if resp.status_code == 200:
            payload = resp.json()
            if isinstance(payload, list) and payload:
                df = pd.DataFrame(payload)
                price_col = "adjClose" if "adjClose" in df.columns else "close"
                if "date" in df.columns and price_col in df.columns:
                    df["Date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
                    s = df.set_index("Date")[price_col].astype("float64").sort_index()
                    return s[~s.index.duplicated(keep="last")].dropna()
        else:
            print(f"[PriceResolver Tiingo WARNING] {ticker}: HTTP {resp.status_code}")
    except Exception as e:
        print(f"[PriceResolver Tiingo WARNING] Could not fetch {ticker}: {e}")
    finally:
        if resp is not None:
            resp.close()
    return pd.Series(dtype="float64")


def _fetch_yfinance_daily_history(ticker: str, start_ts: pd.Timestamp, end_ts: pd.Timestamp):
    """Fetch historical daily close prices via yfinance plus the reported quote currency."""
    if yf is None:
        print("[PriceResolver yfinance WARNING] yfinance is not installed.")
        return pd.Series(dtype="float64"), None

    s_str = (start_ts - pd.Timedelta(days=COVERAGE_TOLERANCE_DAYS + 3)).strftime("%Y-%m-%d")
    e_str = (end_ts + pd.Timedelta(days=1)).strftime("%Y-%m-%d")  # yfinance `end` is exclusive

    try:
        tk = yf.Ticker(ticker)
        hist = tk.history(start=s_str, end=e_str, auto_adjust=True)
        if hist is None or hist.empty or "Close" not in hist.columns:
            return pd.Series(dtype="float64"), None

        reported_curr = None
        try:
            reported_curr = (getattr(tk, "history_metadata", None) or {}).get("currency")
        except Exception:
            reported_curr = None
        if not reported_curr:
            try:
                reported_curr = tk.fast_info["currency"]
            except Exception:
                reported_curr = None

        idx = pd.to_datetime(hist.index)
        if idx.tz is not None:
            idx = idx.tz_localize(None)
        s = pd.Series(hist["Close"].to_numpy(dtype="float64"), index=idx.normalize()).sort_index()
        s = s[~s.index.duplicated(keep="last")].dropna()
        return s, reported_curr
    except Exception as e:
        print(f"[PriceResolver yfinance WARNING] Could not fetch {ticker}: {e}")
        return pd.Series(dtype="float64"), None


def _hydrate_missing_tickers_tier3(missing_tickers: list, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame:
    """
    Fetch missing tickers via Tier 3 (Tiingo -> yfinance), normalize TASE Agorot to ILS,
    persist into `data/portfolio_price_cache.parquet`, and return the long DataFrame.
    """
    if not missing_tickers:
        return pd.DataFrame(columns=LONG_COLUMNS)

    tiingo_key = os.environ.get("TIINGO_API_KEY", "").strip()
    fetched_frames = []

    for t_up in missing_tickers:
        s = _fetch_tiingo_daily_history(t_up, start_ts, end_ts, tiingo_key)
        reported_curr = "USD" if not s.empty else None
        if s.empty:
            s, reported_curr = _fetch_yfinance_daily_history(t_up, start_ts, end_ts)
        if s.empty:
            print(f"[PriceResolver Tier 3 WARNING] No external data returned for {t_up}.")
            continue

        if t_up.endswith(".TA"):
            s = _normalize_tase_agorot_series(s, t_up, reported_currency=reported_curr)

        fetched_frames.append(pd.DataFrame({
            "Date": s.index,
            "Ticker": t_up,
            "Close": s.to_numpy(dtype="float64"),
            "Currency": _infer_native_currency(t_up),
        }))

    if not fetched_frames:
        return pd.DataFrame(columns=LONG_COLUMNS)

    all_new_df = pd.concat(fetched_frames, ignore_index=True)
    _append_to_tier2_cache(all_new_df)
    return all_new_df


# ─── Unified Equity Price Resolution Interface ────────────────────────────────

def resolve_prices(tickers: list, start_date, end_date, dtype: str = "float64",
                   ttl_seconds: float = CACHE_TTL_SECONDS):
    """
    Unified 3-Tier Hybrid Price Resolution Interface.

    Returns:
      (prices_df, currency_map)
        - prices_df: daily calendar grid [start_date, end_date] x tickers, strictly forward-filled
          (no backfill => zero look-ahead). Values are native-currency closes (USD or ILS).
        - currency_map: ticker -> 'USD' | 'ILS'

    Raises:
      RuntimeError if any ticker cannot be resolved from Tier 1, Tier 2, or Tier 3.
    """
    if not tickers:
        return pd.DataFrame(), {}

    clean_tickers = list(dict.fromkeys(str(t).upper().strip() for t in tickers if str(t).strip()))

    start_ts = _to_naive_timestamp(start_date)
    end_ts = _to_naive_timestamp(end_date)
    if end_ts < start_ts:
        start_ts, end_ts = end_ts, start_ts

    out_dtype = "float32" if str(dtype).lower() == "float32" else "float64"
    cache_key = (tuple(sorted(clean_tickers)), start_ts.strftime("%Y-%m-%d"), end_ts.strftime("%Y-%m-%d"), out_dtype)

    now = time.time()
    cached_entry = _PRICE_RESOLVER_CACHE["entries"].get(cache_key)
    if cached_entry is not None and (now - cached_entry[2]) < ttl_seconds:
        return cached_entry[0][clean_tickers].copy(), dict(cached_entry[1])

    currency_map = {t: _infer_native_currency(t) for t in clean_tickers}
    us_candidates = [t for t in clean_tickers if not t.endswith(".TA")]

    # Tier 2 first: compact cache (fast). Tier 1 (6.2M-row lake) only for US tickers not warm in Tier 2.
    t2_df = _query_tier2_cache(clean_tickers, start_ts, end_ts)
    covered_t2 = _covered_tickers(t2_df, start_ts, end_ts)

    t1_df = _query_tier1_lake([t for t in us_candidates if t not in covered_t2], start_ts, end_ts)
    covered_t1 = _covered_tickers(t1_df, start_ts, end_ts)
    if not t1_df.empty:
        # Promote Tier 1 slices into Tier 2 so warm reads avoid scanning the full SimFin lake
        _append_to_tier2_cache(t1_df[t1_df["Ticker"].isin(covered_t1)])

    tier3_candidates = [t for t in clean_tickers if t not in covered_t2 and t not in covered_t1]
    t3_df = _hydrate_missing_tickers_tier3(tier3_candidates, start_ts, end_ts)

    # Precedence on overlapping (Date, Ticker): Tier 3 (freshest) > Tier 1 > Tier 2
    frames = [f[["Date", "Ticker", "Close"]] for f in (t2_df, t1_df, t3_df) if not f.empty]
    if not frames:
        raise RuntimeError(f"[PriceResolver] Unable to resolve prices for any of {clean_tickers} "
                           f"in [{start_ts.date()}, {end_ts.date()}].")

    merged_long = pd.concat(frames, ignore_index=True)
    merged_long.drop_duplicates(subset=["Date", "Ticker"], keep="last", inplace=True)
    pivoted = merged_long.pivot(index="Date", columns="Ticker", values="Close").sort_index()
    del merged_long

    missing = [t for t in clean_tickers if t not in pivoted.columns or pivoted[t].dropna().empty]
    if missing:
        raise RuntimeError(f"[PriceResolver] Unable to resolve prices for {missing} in "
                           f"[{start_ts.date()}, {end_ts.date()}] from local caches or external sources.")

    daily_grid = pd.date_range(start_ts, end_ts, freq="D")
    union_index = pivoted.index.union(daily_grid).sort_values()
    aligned = pivoted.reindex(union_index).ffill().reindex(daily_grid)[clean_tickers]

    leading_gaps = aligned.isna().sum()
    for t, n_gap in leading_gaps[leading_gaps > 0].items():
        print(f"[PriceResolver WARNING] {t}: {int(n_gap)} leading day(s) with no prior observation "
              f"(listing after window start); coerced to 0.0.")

    aligned = aligned.fillna(0.0).replace([np.inf, -np.inf], 0.0).astype(out_dtype)
    aligned.index.name = "Date"

    _PRICE_RESOLVER_CACHE["entries"][cache_key] = (aligned.copy(), dict(currency_map), now)
    return aligned, currency_map


# ─── Macro FX (USD/ILS) & Israeli CPI Engine ──────────────────────────────────

def _fetch_boi_usd_ils(start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.Series:
    """Bank of Israel official representative USD/ILS rate (daily, business days)."""
    url = BOI_USD_ILS_URL.format(start=start_ts.strftime("%Y-%m-%d"), end=end_ts.strftime("%Y-%m-%d"))
    resp = _http_get(url)
    try:
        df = pd.read_csv(io.StringIO(resp.text), usecols=["TIME_PERIOD", "OBS_VALUE"])
    finally:
        resp.close()
    df["Date"] = pd.to_datetime(df["TIME_PERIOD"]).dt.normalize()
    s = pd.to_numeric(df["OBS_VALUE"], errors="coerce")
    s.index = df["Date"]
    s = s.dropna().sort_index()
    s = s[~s.index.duplicated(keep="last")]
    if s.empty:
        raise RuntimeError("[MacroResolver] Bank of Israel returned an empty USD/ILS series.")
    return s.astype("float64")


def _chain_link_cpi(raw: pd.DataFrame) -> pd.Series:
    """
    Chain-link CBS CPI readings published on rotating bases ("Average YYYY" = 100) into the latest base.
    CBS definition: the new base equals the annual average of base-year YYYY expressed in the previous base,
    so link_factor = mean(previous-base values for year YYYY) / 100.
    `raw` columns: ['RefMonth', 'BaseYear', 'Value'].
    """
    raw = raw.sort_values("RefMonth").reset_index(drop=True)
    base_years = sorted(raw["BaseYear"].unique())
    factors = {base_years[-1]: 1.0}

    for newer, older in zip(reversed(base_years[1:]), reversed(base_years[:-1])):
        older_seg = raw[(raw["BaseYear"] == older) & (raw["RefMonth"].dt.year == newer)]
        if len(older_seg) != 12:
            raise RuntimeError(f"[MacroResolver] Cannot chain-link CPI base {older} -> {newer}: "
                               f"expected 12 monthly readings for {newer} in base {older}, got {len(older_seg)}.")
        link = float(older_seg["Value"].mean()) / 100.0
        factors[older] = factors[newer] / link

    linked = raw["Value"].to_numpy(dtype="float64") * raw["BaseYear"].map(factors).to_numpy(dtype="float64")
    return pd.Series(linked, index=pd.DatetimeIndex(raw["RefMonth"]), dtype="float64")


def _fetch_cbs_cpi(end_ts: pd.Timestamp) -> pd.Series:
    """
    CBS general CPI (id 120010) chain-linked to the latest base, keyed by PUBLICATION date
    (15th of month M+1 — 'the known index').
    Fetch starts 01-2009 so the index known on 2010-01-01 (Nov-2009, published 2009-12-15) is available.
    """
    url = CBS_CPI_URL.format(start="01-2009", end=end_ts.strftime("%m-%Y"))
    resp = _http_get(url)
    try:
        payload = resp.json()
    finally:
        resp.close()

    try:
        rows = payload["month"][0]["date"]
    except (KeyError, IndexError, TypeError) as e:
        raise RuntimeError(f"[MacroResolver] Unexpected CBS CPI payload structure: {e}")

    records = []
    for r in rows:
        base_desc = str((r.get("currBase") or {}).get("baseDesc", ""))
        value = (r.get("currBase") or {}).get("value")
        m = re.search(r"(\d{4})", base_desc)
        if value is None or not m:
            continue
        records.append({
            "RefMonth": pd.Timestamp(year=int(r["year"]), month=int(r["month"]), day=1),
            "BaseYear": int(m.group(1)),
            "Value": float(value),
        })
    if not records:
        raise RuntimeError("[MacroResolver] CBS returned no CPI readings.")

    linked = _chain_link_cpi(pd.DataFrame(records))
    pub_index = linked.index + pd.offsets.MonthBegin(1) + pd.Timedelta(days=14)  # 15th of month M+1
    known_cpi = pd.Series(linked.to_numpy(), index=pub_index, dtype="float64").sort_index()
    return known_cpi[known_cpi.index <= _today()]


def _build_macro_calendar_dataset(grid_end: pd.Timestamp) -> pd.DataFrame:
    """
    Continuous daily calendar grid [2010-01-01, grid_end] for USD_ILS and IL_CPI
    with strict chronological forward-fill (no backfill).
    """
    calendar_grid = pd.date_range(MACRO_ORIGIN_DATE, grid_end, freq="D")

    fx = _fetch_boi_usd_ils(MACRO_ORIGIN_DATE - pd.Timedelta(days=31), grid_end)
    cpi = _fetch_cbs_cpi(grid_end)

    usd_ils = fx.reindex(fx.index.union(calendar_grid)).ffill().reindex(calendar_grid)
    il_cpi = cpi.reindex(cpi.index.union(calendar_grid)).ffill().reindex(calendar_grid)

    macro_df = pd.DataFrame({"USD_ILS": usd_ils.to_numpy(), "IL_CPI": il_cpi.to_numpy()},
                            index=calendar_grid, dtype="float64")
    macro_df.index.name = "Date"

    n_missing = int(macro_df.isna().sum().sum())
    if n_missing:
        raise RuntimeError(f"[MacroResolver] {n_missing} unresolved macro cells after forward-fill "
                           f"(source series do not reach back to {MACRO_ORIGIN_DATE.date()}).")
    return macro_df


def _read_macro_disk_cache():
    if not os.path.exists(FX_CPI_CACHE_PATH):
        return None
    try:
        disk_df = pd.read_parquet(FX_CPI_CACHE_PATH, engine="pyarrow")
        disk_df["Date"] = pd.to_datetime(disk_df["Date"]).dt.tz_localize(None).dt.normalize()
        disk_df = disk_df.set_index("Date").sort_index()[["USD_ILS", "IL_CPI"]].astype("float64")
        if disk_df.index.min() > MACRO_ORIGIN_DATE or disk_df.isna().any().any():
            print("[MacroResolver WARNING] fx_cpi_cache.parquet is incomplete; ignoring it.")
            return None
        return disk_df
    except Exception as e:
        print(f"[MacroResolver WARNING] fx_cpi_cache.parquet unreadable; ignoring it: {e}")
        return None


def ensure_macro_cache(end_date=None, ttl_seconds: float = CACHE_TTL_SECONDS) -> pd.DataFrame:
    """
    Ensure `_MACRO_RESOLVER_CACHE` and `data/fx_cpi_cache.parquet` cover [2010-01-01, end_date].
    Order: in-memory TTL -> disk Parquet -> rebuild from Bank of Israel + CBS.
    If sources are unreachable, a stale disk cache is forward-extended by at most MAX_MACRO_STALE_DAYS
    (with a warning); otherwise a RuntimeError is raised.
    """
    now = time.time()
    req_end = _to_naive_timestamp(end_date) if end_date is not None else _today()
    if req_end > _today():
        raise ValueError(f"[MacroResolver] Requested end date {req_end.date()} is in the future.")

    mem_df = _MACRO_RESOLVER_CACHE["full_df"]
    if mem_df is not None and (now - _MACRO_RESOLVER_CACHE["timestamp"]) < ttl_seconds and mem_df.index.max() >= req_end:
        return mem_df

    disk_df = _read_macro_disk_cache()
    if disk_df is not None and disk_df.index.max() >= req_end:
        _MACRO_RESOLVER_CACHE["full_df"] = disk_df
        _MACRO_RESOLVER_CACHE["timestamp"] = now
        return disk_df

    try:
        macro_df = _build_macro_calendar_dataset(_today())
    except Exception as e:
        if disk_df is not None and (req_end - disk_df.index.max()).days <= MAX_MACRO_STALE_DAYS:
            print(f"[MacroResolver WARNING] Sources unreachable ({e}). Forward-extending cache from "
                  f"{disk_df.index.max().date()} to {req_end.date()} using last known values.")
            ext_grid = pd.date_range(disk_df.index.min(), req_end, freq="D")
            macro_df = disk_df.reindex(ext_grid).ffill()
            macro_df.index.name = "Date"
            _MACRO_RESOLVER_CACHE["full_df"] = macro_df
            _MACRO_RESOLVER_CACHE["timestamp"] = now
            return macro_df
        raise RuntimeError(f"[MacroResolver] Cannot resolve USD/ILS & CPI through {req_end.date()}: {e}")

    os.makedirs(DATA_DIR, exist_ok=True)
    tmp_path = FX_CPI_CACHE_PATH + ".tmp"
    macro_df.reset_index().to_parquet(tmp_path, engine="pyarrow", compression="snappy", index=False)
    os.replace(tmp_path, FX_CPI_CACHE_PATH)

    _MACRO_RESOLVER_CACHE["full_df"] = macro_df
    _MACRO_RESOLVER_CACHE["timestamp"] = now
    return macro_df


def resolve_macro_series(start_date, end_date, dtype: str = "float64",
                         ttl_seconds: float = CACHE_TTL_SECONDS) -> pd.DataFrame:
    """
    Resolve daily `USD_ILS` (BoI representative rate) and `IL_CPI` (CBS known index, chain-linked)
    over [start_date, end_date] on a continuous daily calendar grid.
    """
    start_ts = _to_naive_timestamp(start_date)
    end_ts = _to_naive_timestamp(end_date)
    if end_ts < start_ts:
        start_ts, end_ts = end_ts, start_ts
    if start_ts < MACRO_ORIGIN_DATE:
        raise ValueError(f"[MacroResolver] Macro series start at {MACRO_ORIGIN_DATE.date()}; got {start_ts.date()}.")

    out_dtype = "float32" if str(dtype).lower() == "float32" else "float64"
    full_macro = ensure_macro_cache(end_date=end_ts, ttl_seconds=ttl_seconds)

    sliced = full_macro.loc[start_ts:end_ts]
    sliced = sliced.fillna(0.0).replace([np.inf, -np.inf], 0.0).astype(out_dtype)
    sliced.index.name = "Date"
    return sliced.copy()
