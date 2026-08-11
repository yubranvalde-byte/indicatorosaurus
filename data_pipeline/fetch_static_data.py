import yfinance as yf
import pandas as pd
import csv
import time
import os
import requests
import io
import sys
import math

def sanitize_ticker(ticker):
    """Sanitize ticker string for yfinance compatibility (uppercase, dot -> hyphen)."""
    return str(ticker).strip().upper().replace('.', '-')

def fetch_sp500_tickers(session):
    """Fetch S&P 500 constituents from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    resp = session.get(url, timeout=15)
    resp.raise_for_status()
    tables = pd.read_html(io.StringIO(resp.text))
    df = tables[0]
    col = [c for c in df.columns if str(c).lower() in ('symbol', 'ticker')][0]
    return df[col].dropna().astype(str).tolist()

def fetch_nasdaq100_tickers(session):
    """Fetch Nasdaq 100 constituents from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/Nasdaq-100"
    resp = session.get(url, timeout=15)
    resp.raise_for_status()
    tables = pd.read_html(io.StringIO(resp.text))
    # Search for the constituents table (usually columns: Ticker/Symbol, Company)
    for df in tables:
        cols = [str(c).lower() for c in df.columns]
        if any(x in cols for x in ('ticker', 'symbol')) and len(df) < 200:
            col = [c for c in df.columns if str(c).lower() in ('ticker', 'symbol')][0]
            return df[col].dropna().astype(str).tolist()
    raise ValueError("Nasdaq 100 table not found in Wikipedia page")

def get_dynamic_universe():
    """Merge, deduplicate, and sanitize constituents of S&P 500 and Nasdaq 100, prepending SPY and QQQ."""
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"})
    
    try:
        sp500 = fetch_sp500_tickers(session)
        nasdaq100 = fetch_nasdaq100_tickers(session)
        combined = set(sp500 + nasdaq100)
        sanitized = {sanitize_ticker(t) for t in combined if t}
        # Prepend benchmarks and sort the rest
        other_tickers = sorted(list(sanitized - {'SPY', 'QQQ'}))
        return ['SPY', 'QQQ'] + other_tickers
    except Exception as e:
        print(f"⚠️ Warning: Wikipedia scraper failed ({e}). Falling back to static mega-cap list.")
        return ['SPY', 'QQQ', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'BRK-B', 'NVDA', 'META', 'TSLA']

# Deduplicated, dynamically fetched full universe (S&P 500 + Nasdaq 100)
ALL_TICKERS = get_dynamic_universe()

# ─── 22-column schema (+ Ticker = 23 fields total) ──────────────────────────
# CurrentRatio is filter-only (Quality gate). EBITDA is gating-only for EV/EBITDA
# override (prevents negative EBITDA inversion). Neither is packed into the C struct.
CSV_HEADER = [
    'Ticker', 'TrailingPE', 'EV_EBITDA', 'FCFYield', 'RevenueGrowth', 'EPSGrowth',
    'GrossMargin', 'OperatingMargin', 'ROIC', 'NetDebt_EBITDA', 'ShareholderYield',
    'FiveYDivGrowth', 'PayoutRatio', 'Beta', 'RSI', 'PriceVs200SMA', 'RelativeStrength',
    'CurrentRatio',   # Filter-only — not scored by C engine
    'LongName',       # Metadata — not scored by C engine
    'Sector',         # Metadata — not scored by C engine
    'MarketCap',      # Metadata/Filter — not scored by C engine
    'EBITDA',         # Gating-only — EV_EBITDA override guard, placed at the absolute end!
    'Total_Equity', 'FCF', 'EBIT', 'Interest_Expense'
]

import os
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), '..', 'data', 'market_data.csv')
MAX_RETRIES = 3
INTER_TICKER_DELAY = 0.5   # seconds between successful tickers


# ─── Helper: safe float coercion ──────────────────────────────────────────────
def safe_float(val):
    """Return float(val) or 0.0 on any failure / None."""
    if val is None:
        return 0.0
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0


# ─── Helper: Wilder's RSI (EWM smoothing, Wall Street standard) ───────────────
def compute_rsi(close_series: pd.Series, period: int = 14) -> float:
    """
    Compute 14-day Wilder's RSI using Exponential Weighted Means (EWM).
    alpha = 1/period replicates Wilder's original smoothing factor.
    Requires at least (period + 1) rows; returns 0.0 if series is too short.
    """
    if len(close_series) < period + 1:
        return 0.0
    delta = close_series.diff().dropna()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    # Wilder's smoothing: EWM with alpha = 1/period, adjust=False
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean().iloc[-1]
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean().iloc[-1]
    if math.isnan(avg_gain) or math.isnan(avg_loss):
        return 50.0
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100.0 - (100.0 / (1.0 + rs)), 4)


# ─── Helper: Price vs 200-day SMA ─────────────────────────────────────────────
def compute_price_vs_200sma(close_series: pd.Series) -> float:
    """
    Current close / 200-day SMA.
    > 1.0 = trading above 200-SMA (bullish); < 1.0 = below (bearish).
    Uses all available history if fewer than 200 trading days exist.
    """
    if len(close_series) < 2:
        return 0.0
    sma = close_series.iloc[-200:].mean() if len(close_series) >= 200 else close_series.mean()
    current = close_series.iloc[-1]
    return round(current / sma, 6) if sma > 0 else 0.0


# ─── Helper: Relative Strength vs SPY ─────────────────────────────────────────
def compute_relative_strength(close_series: pd.Series, spy_return: float | None) -> float:
    """
    Stock 1-year total return minus SPY 1-year total return.
    Positive = outperformed SPY; Negative = underperformed.
    Returns 0.0 if SPY fetch failed or history is too short.
    """
    if spy_return is None or len(close_series) < 2:
        return 0.0
    if close_series.iloc[0] <= 0:
        return 0.0
    stock_return = (close_series.iloc[-1] / close_series.iloc[0]) - 1.0
    return round(stock_return - spy_return, 6)


# ─── Helper: Shareholder Yield from cashflow statement ───────────────────────
def build_shareholder_yield(ticker_obj: 'yf.Ticker', mktcap: float) -> float:
    """
    Compute Net Cash Flow to Shareholders / MarketCap.
    Outflows (Dividends, Repurchases, Repayments) are negative in yfinance.
    Inflows (Issuances) are positive.
    Total Returned = - (DividendsPaid + NetStockFlow + NetDebtFlow)
    """
    if mktcap <= 0:
        return 0.0
    try:
        cf = ticker_obj.cashflow   # Annual cashflow — most recent column (index 0)
        if cf is None or cf.empty:
            return 0.0

        def _get_cf_row(*keys):
            """Try multiple key aliases; return first match or 0.0."""
            for key in keys:
                if key in cf.index:
                    val = cf.loc[key].iloc[0]
                    try:
                        v = float(val)
                        return 0.0 if (v != v) else v  # NaN guard
                    except (TypeError, ValueError):
                        pass
            return 0.0

        # 1. Cash Dividends Paid (outflow, negative)
        dividends = _get_cf_row(
            'DividendsPaid', 'Cash Dividends Paid',
            'CommonStockDividendsPaid', 'PaymentOfDividends'
        )
        if dividends > 0:
            dividends = -dividends

        # 2. Net Stock Flow = Issuance (positive) + Repurchases (negative)
        repurchases = _get_cf_row(
            'RepurchaseOfCapitalStock', 'RepurchaseOfStock',
            'CommonStockRepurchased', 'PurchaseOfBusiness', 'RepurchaseOfCommonStock'
        )
        if repurchases > 0:
            repurchases = -repurchases

        issuances = _get_cf_row(
            'CommonStockIssuance', 'IssuanceOfStock',
            'ProceedsFromIssuanceOfCommonStock', 'IssuanceOfCapitalStock'
        )
        if issuances < 0:
            issuances = -issuances

        net_stock_flow = issuances + repurchases

        # 3. Net Debt Flow = Issuance (positive) + Repayments (negative)
        net_debt_combined = _get_cf_row('NetIssuancePaymentsOfDebt', 'Net Issuance Payments Of Debt')
        if net_debt_combined != 0.0:
            net_debt_flow = net_debt_combined
        else:
            debt_issuances = _get_cf_row(
                'ProceedsFromIssuanceOfLongTermDebt', 'LongTermDebtIssuance', 'IssuanceOfDebt'
            )
            if debt_issuances < 0:
                debt_issuances = -debt_issuances

            debt_repayments = _get_cf_row(
                'RepaymentOfLongTermDebt', 'LongTermDebtPayment', 'RepaymentsOfLongTermDebt', 'RepaymentOfDebt'
            )
            if debt_repayments > 0:
                debt_repayments = -debt_repayments

            net_debt_flow = debt_issuances + debt_repayments

        total_returned = - (dividends + net_stock_flow + net_debt_flow)
        shareholder_yield = total_returned / mktcap
        return round(shareholder_yield, 6)
    except Exception:
        return 0.0


# ─── Helper: assemble a single data row ───────────────────────────────────────
def build_row(symbol: str, info: dict, hist: pd.DataFrame, spy_return: float | None,
              ticker_obj=None) -> list:
    """
    Compute all 16 scored metrics and auxiliary columns aligned to CSV_HEADER.

    ROIC proxy: returnOnAssets (ROA) — more stable than ROE for mega-caps.
    FiveYDivGrowth: fiveYearAvgDividendYield — true 5-year dividend yield anchor
      (eliminates prior payoutRatio mapping collision that caused identical raw/score
      output for both 'Div Safety' and 'Payout Ratio' columns).
    EV_EBITDA: enterpriseToEbitda from yfinance info dict.
    ShareholderYield: computed from cashflow statement (dividends + buybacks + debt reduction).
    EBITDA: stored as filter-only metadata for pre-scoring override gate.
    CurrentRatio: filter-only (Quality knockout gate).
    """
    close = hist['Close'] if not hist.empty else pd.Series(dtype=float)

    # ── Computed metrics ────────────────────────────────────────────────────
    fcf    = safe_float(info.get('freeCashflow'))
    mktcap = safe_float(info.get('marketCap'))
    fcf_yield = round(fcf / mktcap, 6) if mktcap > 0 else 0.0

    rsi              = compute_rsi(close)
    price_vs_200sma  = compute_price_vs_200sma(close)
    rel_strength     = compute_relative_strength(close, spy_return)

    ev_ebitda        = safe_float(info.get('enterpriseToEbitda'))   # EV/EBITDA multiple
    ebitda_raw       = safe_float(info.get('ebitda'))               # Gate-only: sign check

    # Shareholder Yield requires the cashflow statement — use ticker_obj if provided
    shareholder_yield = build_shareholder_yield(ticker_obj, mktcap) if ticker_obj else 0.0

    # FiveYDivGrowth: true 5-year avg dividend yield (NOT payoutRatio)
    five_y_div = safe_float(info.get('fiveYearAvgDividendYield'))  # % form from yfinance → keep as-is
    # yfinance returns fiveYearAvgDividendYield as a percent (e.g. 1.45 for 1.45%) — normalise to decimal
    if five_y_div > 1.0:
        five_y_div = round(five_y_div / 100.0, 6)

    return [
        symbol,
        safe_float(info.get('trailingPE')),          # TrailingPE
        ev_ebitda,                                    # EV_EBITDA  (enterpriseToEbitda)
        fcf_yield,                                    # FCFYield  (freeCashflow / marketCap)
        safe_float(info.get('revenueGrowth')),        # RevenueGrowth
        safe_float(info.get('earningsGrowth')),       # EPSGrowth
        safe_float(info.get('grossMargins')),         # GrossMargin
        safe_float(info.get('operatingMargins')),     # OperatingMargin
        safe_float(info.get('returnOnAssets')),       # ROIC  [proxy: ROA — avoids ROE distortion]
        safe_float(info.get('debtToEquity')),         # NetDebt_EBITDA
        shareholder_yield,                            # ShareholderYield  (cashflow-derived)
        five_y_div,                                   # FiveYDivGrowth  [5yr avg yield — not payoutRatio]
        safe_float(info.get('payoutRatio')),          # PayoutRatio
        safe_float(info.get('beta')),                 # Beta
        rsi,                                          # RSI  (Wilder's 14-day EWM)
        price_vs_200sma,                              # PriceVs200SMA
        rel_strength,                                 # RelativeStrength  (vs SPY 1yr)
        safe_float(info.get('currentRatio')),         # CurrentRatio  [filter-only: Quality gate]
        str(info.get('longName', '')),                # LongName
        str(info.get('sector', '')),                  # Sector
        safe_float(info.get('marketCap')),            # MarketCap
        ebitda_raw,                                   # EBITDA  [gating-only: EV_EBITDA override, placed at absolute end!]
    ]


# ─── Per-ticker fetch with 3-attempt incremental retry ────────────────────────
def fetch_ticker_data(symbol: str, spy_return: float | None) -> tuple[list, bool]:
    """
    Fetch info + 1-year history for a single ticker with up to MAX_RETRIES attempts.
    Validation guard (DUAL): info dict must be non-empty with 'symbol' key,
    AND price history must contain at least 15 rows (enough for Wilder's 14-period RSI).
    Backoff is incremental: attempt 1 → 1.5s, attempt 2 → 3.0s, attempt 3 → 4.5s.
    Returns (row_list, was_exception).
    """
    # 16 scored + CurrentRatio + LongName + Sector + MarketCap + EBITDA(gate)
    zeroed_row = [symbol] + [0.0] * 17 + ['', '', 0.0, 0.0]
    last_exception = None

    for attempt in range(1, MAX_RETRIES + 1):
        backoff = 1.5 * attempt
        try:
            t = yf.Ticker(symbol)
            info = t.info
            hist = t.history(period='1y')

            # ── DUAL validation guard ──────────────────────────────────────
            info_valid = bool(info) and 'symbol' in info
            hist_valid = len(hist) >= 15

            if info_valid and hist_valid:
                row = build_row(symbol, info, hist, spy_return, ticker_obj=t)
                price = hist['Close'].iloc[-1] if not hist.empty else 0.0
                print(f"  ✓ [{attempt}/{MAX_RETRIES}] {symbol:<8} "
                      f"RSI={row[14]:.1f}  P/200SMA={row[15]:.3f}  "
                      f"RelStr={row[16]:+.3f}  CurrRatio={row[17]:.2f}  Close=${price:.2f}")
                return row, False

            # Partial failure — report which guard failed
            if not info_valid:
                reason = "info dict empty or missing 'symbol'"
            else:
                reason = f"history only {len(hist)} rows (<15)"

            if attempt < MAX_RETRIES:
                print(f"  ↻ [{attempt}/{MAX_RETRIES}] {symbol} — {reason}. "
                      f"Retrying in {backoff:.1f}s...")
                time.sleep(backoff)
            else:
                print(f"  ✗ [{attempt}/{MAX_RETRIES}] {symbol} — {reason}. Writing zeroed row.")

        except Exception as exc:
            last_exception = exc
            print(f"  ✗ [{attempt}/{MAX_RETRIES}] {symbol} — Exception: {exc}")
            if attempt < MAX_RETRIES:
                time.sleep(backoff)

    return zeroed_row, (last_exception is not None)


# ─── Main driver ──────────────────────────────────────────────────────────────
def fetch_and_save_market_data():
    # ── Step 1: State Recovery & Metadata Repair Safeguard ────────────────
    fetched_tickers = set()
    valid_rows = []
    file_exists = os.path.exists(OUTPUT_FILE)
    file_empty = False

    if file_exists:
        file_empty = (os.path.getsize(OUTPUT_FILE) == 0)
        if not file_empty:
            try:
                with open(OUTPUT_FILE, 'r', newline='') as f:
                    reader = csv.reader(f)
                    header = next(reader, None)
                    if header:
                        try:
                            ticker_idx = header.index('Ticker')
                            longname_idx = header.index('LongName')
                            sector_idx = header.index('Sector')
                        except ValueError:
                            ticker_idx = 0
                            longname_idx = -1
                            sector_idx = -1

                        for r in reader:
                            if not r or len(r) != len(CSV_HEADER):
                                continue
                            ticker = r[ticker_idx].upper().strip()
                            
                            has_longname = bool(r[longname_idx].strip()) if longname_idx != -1 and longname_idx < len(r) else False
                            has_sector = bool(r[sector_idx].strip()) if sector_idx != -1 and sector_idx < len(r) else False
                            
                            if ticker and has_longname and has_sector:
                                fetched_tickers.add(ticker)
                                valid_rows.append(r)

                print(f"  ✓ State Recovery: Found {len(valid_rows)} valid, fully-enriched records.")
                
                # Overwrite/repair output file to contain only the fully-enriched valid rows
                with open(OUTPUT_FILE, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(CSV_HEADER)
                    if valid_rows:
                        writer.writerows(valid_rows)
                print(f"  ✓ Repaired {OUTPUT_FILE} (cleared {len(valid_rows)} complete rows, stripped incomplete ones)")
            except Exception as e:
                print(f"  Warning: Failed to parse and repair existing output file: {e}")

    # Determine remaining delta
    target_tickers = [t for t in ALL_TICKERS if t not in fetched_tickers]

    print("=" * 60)
    print(f"  23-Column Master Matrix (16 scored + 7 filter/metadata)")
    print(f"  Total Universe: {len(ALL_TICKERS)} tickers | Already Fetched: {len(fetched_tickers)} | Remaining: {len(target_tickers)}")
    print("=" * 60)

    if not target_tickers:
        print("  ✓ All tickers in the universe have already been processed and repaired. Nothing to do!")
        return

    # Ensure CSV file is initialized with headers if it doesn't exist/is empty
    if not file_exists or file_empty:
        try:
            with open(OUTPUT_FILE, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(CSV_HEADER)
            print(f"  ✓ Created new {OUTPUT_FILE} with headers")
        except Exception as e:
            print(f"  Error creating file: {e}")
            return

    # ── Step 2: Fetch SPY benchmark return (once) ────────────────────────
    spy_return = None
    print("\n📡 Downloading SPY 1-year history as benchmark...")
    try:
        spy_hist = yf.Ticker('SPY').history(period='1y')
        if len(spy_hist) >= 2:
            spy_return = (spy_hist['Close'].iloc[-1] / spy_hist['Close'].iloc[0]) - 1.0
            print(f"  ✓ SPY 1yr return: {spy_return:+.4f} ({spy_return * 100:+.2f}%)")
        else:
            print("  ✗ SPY history too short — RelativeStrength will be 0.0 for all tickers.")
    except Exception as exc:
        print(f"  ✗ SPY fetch failed: {exc} — RelativeStrength will be 0.0 for all tickers.")

    # ── Step 3: Fetch remaining tickers with batching and safeguards ──────────
    print(f"\n🔍 Processing {len(target_tickers)} remaining tickers (0.5s inter-ticker delay)...\n")
    
    buffer_list = []
    consecutive_failures = 0

    def write_batch(rows):
        if not rows:
            return
        try:
            with open(OUTPUT_FILE, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(rows)
        except Exception as e:
            print(f"  Error writing batch to CSV: {e}")

    for i, symbol in enumerate(target_tickers, start=1):
        print(f"→ [{i:03d}/{len(target_tickers)}] {symbol}")
        row_list, is_exception = fetch_ticker_data(symbol, spy_return)

        if is_exception:
            consecutive_failures += 1
        else:
            consecutive_failures = 0

        buffer_list.append(row_list)

        # Active Rate-Limit Safeguard (Early Exit)
        if consecutive_failures >= 3:
            write_batch(buffer_list)
            buffer_list.clear()
            print("\n❌ Throttling detected. System state saved safely. Please resume later.")
            sys.exit(0)

        # Chunk-Based Flushing (Size = 10)
        if len(buffer_list) >= 10:
            write_batch(buffer_list)
            buffer_list.clear()
            print(f"  💾 Flushed batch of 10 rows to disk")

        time.sleep(INTER_TICKER_DELAY)

    # Flush any remaining rows
    if buffer_list:
        write_batch(buffer_list)
        print(f"  💾 Flushed final {len(buffer_list)} rows to disk")
        buffer_list.clear()

    # ── Step 4: Summary ───────────────────────────────────────────────────
    final_count = 0
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, 'r', newline='') as f:
                reader = csv.reader(f)
                next(reader, None)
                for r in reader:
                    if r:
                        final_count += 1
        except Exception:
            pass

    print("\n" + "=" * 60)
    print(f"  ✅ {OUTPUT_FILE} written successfully.")
    print(f"     Total records in file: {final_count}")
    print("=" * 60)


# ─── Helper: find closest date column in statement DataFrame ─────────────────
def find_closest_date_col(df, target_date_str):
    if df is None or df.empty:
        return None
    try:
        target_dt = pd.to_datetime(target_date_str)
        best_col = None
        min_diff = pd.Timedelta(days=7) # maximum difference of 7 days
        for col in df.columns:
            col_dt = pd.to_datetime(col)
            diff = abs(col_dt - target_dt)
            if diff <= min_diff:
                min_diff = diff
                best_col = col
        return best_col
    except Exception:
        return None


# ─── Helper: get statement row value at closest date ────────────────────────
def get_statement_val(df, keys, target_date_str, default=0.0):
    col = find_closest_date_col(df, target_date_str)
    if col is None:
        return default
    for key in keys:
        if key in df.index:
            try:
                val = df.loc[key, col]
                if isinstance(val, pd.Series):
                    val = val.iloc[0]
                v = float(val)
                return default if (v != v) else v
            except (TypeError, ValueError):
                pass
    return default


# ─── Helper: YoY growth calculation ──────────────────────────────────────────
def get_yoy_growth(df, keys, q_date_str):
    current_val = get_statement_val(df, keys, q_date_str, default=None)
    if current_val is None or current_val == 0.0:
        return 0.0
    try:
        target_dt = pd.to_datetime(q_date_str) - pd.DateOffset(years=1)
        target_col = find_closest_date_col(df, target_dt.strftime('%Y-%m-%d'))
        if target_col is None:
            return 0.0
        prior_val = get_statement_val(df, keys, target_col.strftime('%Y-%m-%d'), default=None)
        if prior_val is None or prior_val <= 0.0:
            return 0.0
        return round((current_val - prior_val) / prior_val, 6)
    except Exception:
        return 0.0


# ─── Helper: get historical price ─────────────────────────────────────────────
def get_historical_price_on(price_series, date_str):
    if price_series.empty:
        return 0.0
    try:
        dt = pd.to_datetime(date_str)
        valid_dates = price_series.index[price_series.index <= dt]
        if len(valid_dates) == 0:
            return float(price_series.iloc[0])
        return float(price_series.loc[valid_dates[-1]])
    except Exception:
        return 0.0


# ─── Helper: get shares outstanding at date with multiple fallbacks ─────────
def get_shares_at(shares_series, balance_sheet, date_str, default_shares):
    # Fallback 1: get_shares_outstanding timeline
    if shares_series is not None and not shares_series.empty:
        try:
            dt = pd.to_datetime(date_str)
            valid_dates = shares_series.index[shares_series.index <= dt]
            if len(valid_dates) > 0:
                v = float(shares_series.loc[valid_dates[-1]])
                if v > 0:
                    return v
        except Exception:
            pass

    # Fallback 2: quarterly_balance_sheet rows
    if balance_sheet is not None and not balance_sheet.empty:
        v = get_statement_val(balance_sheet, ['Share Issued', 'Ordinary Shares Number', 'OrdinarySharesNumber', 'SharesIssued'], date_str, default=0.0)
        if v > 0:
            return v

    # Fallback 3: Ternary default
    return default_shares


# ─── Helper: compute point-in-time quarterly row ───────────────────────────
def compute_quarterly_row(symbol, info, date_str, financials, cashflow, balance_sheet, price_hist, spy_hist, shares_series):
    sector = str(info.get('sector', ''))
    
    # 1. TrailingPE
    net_income = get_statement_val(financials, ['Net Income', 'NetIncome', 'Net Income Common Stockholders'], date_str, 0.0)
    shares = get_shares_at(shares_series, balance_sheet, date_str, safe_float(info.get('sharesOutstanding')))
    close_price = get_historical_price_on(price_hist['Close'], date_str) if 'Close' in price_hist else 0.0
    
    if net_income > 0 and shares > 0:
        annualized_eps = (net_income * 4.0) / shares
        forward_pe = close_price / annualized_eps if annualized_eps > 0 else 0.0
    else:
        forward_pe = 0.0

    # 2. EV_EBITDA
    ebitda = get_statement_val(financials, ['EBITDA'], date_str, 0.0)
    if ebitda <= 0.0:
        op_inc = get_statement_val(financials, ['Operating Income', 'OperatingIncome'], date_str, 0.0)
        depr = get_statement_val(financials, ['DepreciationAndAmortization', 'Depreciation And Amortization', 'Reconciled Depreciation'], date_str, 0.0)
        ebitda = op_inc + depr
        
    market_cap = close_price * shares
    total_debt = get_statement_val(balance_sheet, ['Total Debt', 'TotalDebt'], date_str, 0.0)
    cash = get_statement_val(balance_sheet, ['Cash And Cash Equivalents', 'CashAndCashEquivalents', 'Cash Cash Equivalents And Short Term Investments', 'CashCashEquivalentsAndShortTermInvestments'], date_str, 0.0)
    ev = market_cap + total_debt - cash
    
    if ebitda > 0 and ev > 0:
        ev_ebitda = ev / (ebitda * 4.0)
    else:
        ev_ebitda = 0.0

    # 3. FCFYield
    fcf = get_statement_val(cashflow, ['Free Cash Flow', 'FreeCashFlow'], date_str, 0.0)
    if fcf == 0.0:
        ops_cf = get_statement_val(cashflow, ['Operating Cash Flow', 'OperatingCashFlow', 'CashFlowFromContinuingOperatingActivities'], date_str, 0.0)
        capex = get_statement_val(cashflow, ['Capital Expenditure', 'CapitalExpenditure', 'Capital Expenditures'], date_str, 0.0)
        if capex > 0:
            capex = -capex
        fcf = ops_cf + capex
    
    fcf_yield = (fcf * 4.0) / market_cap if market_cap > 0 else 0.0

    # 4. RevenueGrowth (YoY)
    rev_growth = get_yoy_growth(financials, ['Total Revenue', 'Revenue'], date_str)

    # 5. EPSGrowth (YoY)
    eps_growth = get_yoy_growth(financials, ['Net Income', 'NetIncome', 'Net Income Common Stockholders'], date_str)

    # 6. GrossMargin
    gross_profit = get_statement_val(financials, ['Gross Profit', 'GrossProfit'], date_str, 0.0)
    total_rev = get_statement_val(financials, ['Total Revenue', 'Revenue'], date_str, 0.0)
    gross_margin = gross_profit / total_rev if total_rev > 0 else 0.0

    # 7. OperatingMargin
    op_income = get_statement_val(financials, ['Operating Income', 'OperatingIncome'], date_str, 0.0)
    op_margin = op_income / total_rev if total_rev > 0 else 0.0

    # 8. ROIC (Proxy: ROA)
    total_assets = get_statement_val(balance_sheet, ['Total Assets', 'TotalAssets'], date_str, 0.0)
    roic = net_income / total_assets if total_assets > 0 else 0.0

    # 9. NetDebt_EBITDA
    equity = get_statement_val(balance_sheet, ['Stockholders Equity', 'Total Stockholder Equity', 'StockholdersEquity', 'TotalStockholdersEquity'], date_str, 0.0)
    net_debt = total_debt - cash
    net_debt_ebitda = net_debt / ebitda if ebitda > 0 else 0.0
    interest_exp = abs(get_statement_val(financials, ['Interest Expense', 'InterestExpense', 'Interest Expense Net'], date_str, 0.0))

    # 10. ShareholderYield
    divs = get_statement_val(cashflow, ['DividendsPaid', 'Cash Dividends Paid', 'CommonStockDividendsPaid', 'PaymentOfDividends'], date_str, 0.0)
    if divs > 0: divs = -divs
    
    repurchases = get_statement_val(cashflow, ['RepurchaseOfCapitalStock', 'RepurchaseOfStock', 'CommonStockRepurchased', 'PurchaseOfBusiness', 'RepurchaseOfCommonStock'], date_str, 0.0)
    if repurchases > 0: repurchases = -repurchases
    
    issuances = get_statement_val(cashflow, ['CommonStockIssuance', 'IssuanceOfStock', 'ProceedsFromIssuanceOfCommonStock', 'IssuanceOfCapitalStock'], date_str, 0.0)
    if issuances < 0: issuances = -issuances
    
    net_stock_flow = issuances + repurchases
    
    net_debt_flow = get_statement_val(cashflow, ['NetIssuancePaymentsOfDebt', 'Net Issuance Payments Of Debt'], date_str, 0.0)
    if net_debt_flow == 0.0:
        debt_iss = get_statement_val(cashflow, ['ProceedsFromIssuanceOfLongTermDebt', 'LongTermDebtIssuance', 'IssuanceOfDebt'], date_str, 0.0)
        if debt_iss < 0: debt_iss = -debt_iss
        debt_rep = get_statement_val(cashflow, ['RepaymentOfLongTermDebt', 'LongTermDebtPayment', 'RepaymentsOfLongTermDebt', 'RepaymentOfDebt'], date_str, 0.0)
        if debt_rep > 0: debt_rep = -debt_rep
        net_debt_flow = debt_iss + debt_rep

    total_returned = - (divs + net_stock_flow + net_debt_flow)
    shareholder_yield = (total_returned * 4.0) / market_cap if market_cap > 0 else 0.0

    # 11. FiveYDivGrowth
    five_y_div = safe_float(info.get('fiveYearAvgDividendYield'))
    if five_y_div > 1.0:
        five_y_div = five_y_div / 100.0

    # 12. PayoutRatio
    payout_ratio = abs(divs) / net_income if net_income > 0 else 0.0

    # 13. Beta
    beta = safe_float(info.get('beta'))

    # Technical series up to date_str
    dt = pd.to_datetime(date_str)
    close_series_hist = price_hist['Close'].loc[price_hist.index <= dt] if 'Close' in price_hist else pd.Series(dtype=float)

    # 14. RSI (Wilder's 14-day)
    rsi = compute_rsi(close_series_hist)

    # 15. PriceVs200SMA
    price_vs_200sma = compute_price_vs_200sma(close_series_hist)

    # 16. RelativeStrength (vs SPY)
    rel_strength = 0.0
    if not close_series_hist.empty:
        start_dt = dt - pd.DateOffset(years=1)
        close_at_start = get_historical_price_on(price_hist['Close'], start_dt.strftime('%Y-%m-%d')) if 'Close' in price_hist else 0.0
        close_at_q = float(close_series_hist.iloc[-1])
        if close_at_start > 0:
            stock_1yr_return = (close_at_q / close_at_start) - 1.0
            
            spy_series_hist = spy_hist['Close'].loc[spy_hist.index <= dt] if 'Close' in spy_hist else pd.Series(dtype=float)
            if not spy_series_hist.empty:
                spy_at_start = get_historical_price_on(spy_hist['Close'], start_dt.strftime('%Y-%m-%d')) if 'Close' in spy_hist else 0.0
                spy_at_q = float(spy_series_hist.iloc[-1])
                if spy_at_start > 0:
                    spy_1yr_return = (spy_at_q / spy_at_start) - 1.0
                    rel_strength = stock_1yr_return - spy_1yr_return

    # 17. CurrentRatio
    curr_assets = get_statement_val(balance_sheet, ['Current Assets', 'Total Current Assets', 'TotalCurrentAssets'], date_str, 0.0)
    curr_liabs = get_statement_val(balance_sheet, ['Current Liabilities', 'Total Current Liabilities', 'TotalCurrentLiabilities'], date_str, 0.0)
    current_ratio = curr_assets / curr_liabs if curr_liabs > 0 else 0.0

    return [
        symbol,
        sector,
        date_str,
        round(forward_pe, 4),
        round(ev_ebitda, 4),
        round(fcf_yield, 6),
        round(rev_growth, 6),
        round(eps_growth, 6),
        round(gross_margin, 6),
        round(op_margin, 6),
        round(roic, 6),
        round(net_debt_ebitda, 4),
        round(shareholder_yield, 6),
        round(five_y_div, 6),
        round(payout_ratio, 6),
        round(beta, 4),
        round(rsi, 4),
        round(price_vs_200sma, 6),
        round(rel_strength, 6),
        round(current_ratio, 4),
        round(market_cap, 2),
        round(ebitda, 2),
        round(equity, 2),
        round(fcf, 2),
        round(op_income, 2),
        round(interest_exp, 2)
    ]


# ─── Main driver for historical panel dataset ingestion ────────────────────
def fetch_and_save_historical_fundamental_data():
    HIST_OUTPUT_FILE = os.path.join(os.path.dirname(__file__), '..', 'data', 'historical_fundamental_data.csv')
    HIST_HEADER = [
        'Ticker', 'GICS_Sector', 'Quarter_End_Date', 'TrailingPE', 'EV_EBITDA', 'FCFYield',
        'RevenueGrowth', 'EPSGrowth', 'GrossMargin', 'OperatingMargin', 'ROIC', 'NetDebt_EBITDA',
        'ShareholderYield', 'FiveYDivGrowth', 'PayoutRatio', 'Beta', 'RSI', 'PriceVs200SMA',
        'RelativeStrength', 'CurrentRatio', 'MarketCap', 'EBITDA',
        'Total_Equity', 'FCF', 'EBIT', 'Interest_Expense'
    ]

    fetched_records = set()
    valid_rows = []
    file_exists = os.path.exists(HIST_OUTPUT_FILE)
    file_empty = False

    if file_exists:
        file_empty = (os.path.getsize(HIST_OUTPUT_FILE) == 0)
        if not file_empty:
            try:
                with open(HIST_OUTPUT_FILE, 'r', newline='') as f:
                    reader = csv.reader(f)
                    header = next(reader, None)
                    if header:
                        ticker_idx = header.index('Ticker')
                        date_idx = header.index('Quarter_End_Date')
                        for r in reader:
                            if not r or len(r) != len(HIST_HEADER):
                                continue
                            ticker = r[ticker_idx].upper().strip()
                            q_date = r[date_idx].strip()
                            if ticker and q_date:
                                fetched_records.add((ticker, q_date))
                                valid_rows.append(r)
                print(f"  ✓ State Recovery: Found {len(valid_rows)} valid historical records.")
            except Exception as e:
                print(f"  Warning: Failed to parse historical output file: {e}")

    if not file_exists or file_empty:
        try:
            with open(HIST_OUTPUT_FILE, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(HIST_HEADER)
            print(f"  ✓ Created new {HIST_OUTPUT_FILE} with headers")
        except Exception as e:
            print(f"  Error creating file: {e}")
            return

    print("\n📡 Downloading SPY 5-year history as benchmark...")
    try:
        spy_hist = yf.Ticker('SPY').history(period='5y')
        if spy_hist.empty:
            raise ValueError("SPY history fetch returned empty DataFrame.")
        if hasattr(spy_hist.index, 'tz') and spy_hist.index.tz is not None:
            spy_hist = spy_hist.tz_localize(None)
        print(f"  ✓ Downloaded {len(spy_hist)} rows of SPY history.")
    except Exception as e:
        print(f"  ✗ SPY 5y download failed: {e}. Cannot compute Relative Strength properly.")
        spy_hist = pd.DataFrame()

    consecutive_failures = 0
    buffer_list = []

    def write_hist_batch(rows):
        if not rows:
            return
        try:
            with open(HIST_OUTPUT_FILE, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(rows)
        except Exception as e:
            print(f"  Error writing batch to historical CSV: {e}")

    print(f"\n🔍 Processing {len(ALL_TICKERS)} tickers for historical quarterly panel dataset...\n")

    for idx, symbol in enumerate(ALL_TICKERS, start=1):
        print(f"→ [{idx:03d}/{len(ALL_TICKERS)}] {symbol} (quarterly historical)")
        
        try:
            t = yf.Ticker(symbol)
            financials = t.quarterly_financials
            cashflow = t.quarterly_cashflow
            balance_sheet = t.quarterly_balance_sheet
            
            if financials is None or financials.empty:
                print(f"  ✗ {symbol} has no quarterly financials data. Skipping.")
                continue

            price_hist = t.history(period='5y')
            if price_hist.empty:
                print(f"  ✗ {symbol} has no historical price data. Skipping.")
                continue
            if hasattr(price_hist.index, 'tz') and price_hist.index.tz is not None:
                price_hist = price_hist.tz_localize(None)
                
            try:
                shares_series = t.get_shares_outstanding(
                    start=(pd.Timestamp.now() - pd.DateOffset(years=5)).strftime('%Y-%m-%d'),
                    end=pd.Timestamp.now().strftime('%Y-%m-%d')
                )
                if shares_series is not None and not shares_series.empty:
                    if hasattr(shares_series.index, 'tz') and shares_series.index.tz is not None:
                        shares_series = shares_series.tz_localize(None)
            except Exception:
                shares_series = None

            quarter_dates = sorted([col.strftime('%Y-%m-%d') for col in financials.columns])
            info = t.info or {}
            
            ticker_rows_added = 0
            for q_date in quarter_dates:
                if (symbol, q_date) in fetched_records:
                    continue

                try:
                    row = compute_quarterly_row(symbol, info, q_date, financials, cashflow, balance_sheet, price_hist, spy_hist, shares_series)
                    buffer_list.append(row)
                    ticker_rows_added += 1
                except Exception as row_e:
                    print(f"  ⚠️ Error processing {symbol} at {q_date}: {row_e}")
                    
            if ticker_rows_added > 0:
                print(f"  ✓ Added {ticker_rows_added} historical quarters for {symbol}")
                consecutive_failures = 0
            else:
                print(f"  • No new quarters added for {symbol}")

            if buffer_list:
                write_hist_batch(buffer_list)
                buffer_list.clear()

        except Exception as ticker_e:
            consecutive_failures += 1
            print(f"  ✗ Failed to fetch quarterly historical data for {symbol}: {ticker_e}")
            
            if consecutive_failures >= 5:
                print("\n⚠️ Throttling / HTTP 429 detected. Saving state and exiting gracefully (0) to allow offline deployment boot.")
                if buffer_list:
                    write_hist_batch(buffer_list)
                sys.exit(0)

        time.sleep(INTER_TICKER_DELAY)

    print("\n" + "=" * 60)
    print(f"  ✅ {HIST_OUTPUT_FILE} written successfully.")
    print("=" * 60)


if __name__ == '__main__':
    if '--historical' in sys.argv:
        fetch_and_save_historical_fundamental_data()
    else:
        fetch_and_save_market_data()

