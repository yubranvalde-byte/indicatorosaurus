import os
import sys
import time
import math
import requests
import io
import pandas as pd
import numpy as np
import csv
from dotenv import load_dotenv
import simfin as sf

# ─── Load Environment & Initialize SimFin ─────────────────────────────────────
load_dotenv()
SIMFIN_API_KEY = os.environ.get('SIMFIN_API_KEY')
if not SIMFIN_API_KEY:
    print("Error: SIMFIN_API_KEY environment variable not found in .env.")
    sys.exit(1)

sf.set_api_key(SIMFIN_API_KEY)
sf.set_data_dir(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'simfin_data_cache'))

# ─── Helper Functions for Universe Selection ──────────────────────────────────
def sanitize_ticker(ticker):
    """Sanitize ticker string for SimFin compatibility."""
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
    for df in tables:
        cols = [str(c).lower() for c in df.columns]
        if any(x in cols for x in ('ticker', 'symbol')) and len(df) < 200:
            col = [c for c in df.columns if str(c).lower() in ('ticker', 'symbol')][0]
            return df[col].dropna().astype(str).tolist()
    raise ValueError("Nasdaq 100 table not found in Wikipedia page")

def get_pit_universe():
    """Union S&P/Nasdaq constituents and local delisted tickers passing $2B market cap proxy pre-filter."""
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"})
    
    # 1. Fetch current wiki constituents
    try:
        sp500 = fetch_sp500_tickers(session)
        nasdaq100 = fetch_nasdaq100_tickers(session)
        wiki_tickers = {sanitize_ticker(t) for t in (sp500 + nasdaq100) if t}
    except Exception as e:
        print(f"[WARN] Wikipedia scraper failed ({e}). Falling back to static constituent list.")
        wiki_tickers = {'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'BRK-B', 'NVDA', 'META', 'TSLA'}

    # Ensure benchmarks are in wiki_tickers
    wiki_tickers.update({'SPY', 'QQQ'})

    # 2. Mine local files for historical / delisted tickers
    s_tickers = set()
    if os.path.exists('securities.csv'):
        try:
            s_df = pd.read_csv('securities.csv')
            s_tickers = set(s_df.iloc[:, 0].dropna().apply(sanitize_ticker))
        except Exception as e:
            print(f"[WARN] Error reading securities.csv: {e}")

    f_tickers = set()
    if os.path.exists('fundamentals.csv'):
        try:
            f_df = pd.read_csv('fundamentals.csv')
            f_tickers = set(f_df['Ticker Symbol'].dropna().apply(sanitize_ticker))
        except Exception as e:
            print(f"[WARN] Error reading fundamentals.csv: {e}")

    local_tickers = s_tickers.union(f_tickers)
    historical_candidates = local_tickers - wiki_tickers

    passed_historical = set()
    if historical_candidates:
        print(f"[INFO] Evaluating {len(historical_candidates)} historical candidates with $2 Billion market cap pre-filter...")
        try:
            # Load daily share prices to check historical market capitalization
            prices = sf.load_shareprices(variant='daily', market='us')
            # Pre-filter for candidates to optimize memory/speed
            prices_cand = prices[prices.index.get_level_values('Ticker').isin(historical_candidates)].copy()
            if not prices_cand.empty:
                # Calculate MarketCap = Close * Shares Outstanding
                prices_cand['MarketCap'] = prices_cand['Close'] * prices_cand['Shares Outstanding']
                max_mc = prices_cand.groupby(level='Ticker')['MarketCap'].max()
                passed = max_mc[max_mc >= 2e9].index.tolist()
                passed_historical = set(passed)
                print(f"[OK] {len(passed_historical)} historical tickers passed $2B threshold.")
        except Exception as e:
            print(f"[WARN] Error applying $2B filter: {e}. Falling back to including all candidates.")
            passed_historical = historical_candidates

    # 3. Combine and sort
    all_sanitized = wiki_tickers.union(passed_historical)
    other_tickers = sorted(list(all_sanitized - {'SPY', 'QQQ'}))
    return ['SPY', 'QQQ'] + other_tickers

# Deduplicated, dynamically fetched full PIT universe
ALL_TICKERS = get_pit_universe()

# ─── Retained Custom Technical Helper Functions (Not used in bulk but preserved) ─────────────────
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
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean().iloc[-1]
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean().iloc[-1]
    if math.isnan(avg_gain) or math.isnan(avg_loss):
        return 50.0
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100.0 - (100.0 / (1.0 + rs)), 4)

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

# ─── Vectorized Technical Calculation Helpers (Mathematically Identical) ───────────
def compute_rsi_series(close_series: pd.Series, period: int = 14) -> pd.Series:
    """Vectorized calculation of Wilder's RSI over a series."""
    delta = close_series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.fillna(50.0)
    rsi.loc[avg_loss == 0] = 100.0
    rsi.iloc[:period] = 0.0
    return rsi

# ─── Master CSV Headers ────────────────────────────────────────────────────────
CSV_HEADER = [
    'Ticker', 'ForwardPE', 'EV_EBITDA', 'FCFYield', 'RevenueGrowth', 'EPSGrowth',
    'GrossMargin', 'OperatingMargin', 'ROIC', 'DebtToEquity', 'ShareholderYield',
    'FiveYDivGrowth', 'PayoutRatio', 'Beta', 'RSI', 'PriceVs200SMA', 'RelativeStrength',
    'CurrentRatio',   # Filter-only
    'LongName',       # Metadata
    'Sector',         # Metadata
    'MarketCap',      # Metadata/Filter
    'EBITDA',         # Gating-only EV_EBITDA override guard
]

HIST_HEADER = [
    'Ticker', 'GICS_Sector', 'Quarter_End_Date', 'ForwardPE', 'EV_EBITDA', 'FCFYield',
    'RevenueGrowth', 'EPSGrowth', 'GrossMargin', 'OperatingMargin', 'ROIC', 'DebtToEquity',
    'ShareholderYield', 'FiveYDivGrowth', 'PayoutRatio', 'Beta', 'RSI', 'PriceVs200SMA',
    'RelativeStrength', 'CurrentRatio', 'MarketCap', 'EBITDA'
]

OUTPUT_FILE = 'market_data.csv'
HIST_OUTPUT_FILE = 'historical_fundamental_data.csv'

# ─── Core Data Pipeline Execution ─────────────────────────────────────────────
def run_pipeline():
    print("[INFO] Downloading SimFin bulk datasets (US market)...")
    
    # 1. Load companies and industries
    companies = sf.load_companies(market='us').reset_index()
    industries = sf.load_industries().reset_index()
    metadata = pd.merge(companies, industries, on='IndustryId', how='left')
    metadata = metadata.rename(columns={'Company Name': 'LongName'})
    
    # 2. Load financial statements
    target_tickers = [t for t in ALL_TICKERS if t not in ('SPY', 'QQQ')]
    
    income = sf.load_income(variant='quarterly', market='us').reset_index()
    balance = sf.load_balance(variant='quarterly', market='us').reset_index()
    cashflow = sf.load_cashflow(variant='quarterly', market='us').reset_index()
    
    # Filter statements for target universe to speed up calculations
    income = income[income['Ticker'].isin(target_tickers)]
    balance = balance[balance['Ticker'].isin(target_tickers)]
    cashflow = cashflow[cashflow['Ticker'].isin(target_tickers)]
    
    # 3. Merge statements on Ticker and Report Date (Period End Date)
    q_df = pd.merge(income, balance, on=['Ticker', 'Report Date'], how='inner', suffixes=('', '_bal'))
    q_df = pd.merge(q_df, cashflow, on=['Ticker', 'Report Date'], how='inner', suffixes=('', '_cf'))
    
    # 4. YoY growth calculations
    q_df = q_df.sort_values(by=['Ticker', 'Report Date'])
    shifted = q_df[['Ticker', 'Fiscal Year', 'Fiscal Period', 'Revenue', 'Net Income']].copy()
    shifted['Fiscal Year'] = shifted['Fiscal Year'] + 1
    shifted = shifted.rename(columns={'Revenue': 'Revenue_Prior', 'Net Income': 'Net_Income_Prior'})
    
    q_df = pd.merge(q_df, shifted, on=['Ticker', 'Fiscal Year', 'Fiscal Period'], how='left')
    
    q_df['RevenueGrowth'] = (q_df['Revenue'] - q_df['Revenue_Prior']) / q_df['Revenue_Prior']
    q_df.loc[q_df['Revenue_Prior'] <= 0, 'RevenueGrowth'] = 0.0
    q_df['RevenueGrowth'] = q_df['RevenueGrowth'].fillna(0.0)
    
    q_df['EPSGrowth'] = (q_df['Net Income'] - q_df['Net_Income_Prior']) / q_df['Net_Income_Prior']
    q_df.loc[q_df['Net_Income_Prior'] <= 0, 'EPSGrowth'] = 0.0
    q_df['EPSGrowth'] = q_df['EPSGrowth'].fillna(0.0)
    
    # 5. Load share prices and perform weekly downsampling
    prices = sf.load_shareprices(variant='daily', market='us')
    prices = prices[prices.index.get_level_values('Ticker').isin(ALL_TICKERS)]
    
    # Resample daily to Friday weekly interval ('W-FRI')
    prices_weekly = prices.groupby(level='Ticker').resample('W-FRI', level='Date').last()
    prices_weekly = prices_weekly.groupby(level='Ticker').ffill().reset_index()
    
    # 6. Zero Look-Ahead Bias Weekly Alignment (pd.merge_asof on Date and Publish Date)
    prices_weekly['Date'] = pd.to_datetime(prices_weekly['Date'])
    q_df['Publish Date'] = pd.to_datetime(q_df['Publish Date'])
    
    prices_weekly = prices_weekly.sort_values(by='Date')
    q_df = q_df.sort_values(by='Publish Date')
    
    aligned = pd.merge_asof(
        prices_weekly,
        q_df,
        left_on='Date',
        right_on='Publish Date',
        by='Ticker',
        direction='backward'
    )
    
    # 7. Core Indicators & Features Calculation
    shares = aligned['Shares Outstanding'].fillna(aligned['Shares (Basic)']).fillna(0.0)
    aligned['MarketCap'] = aligned['Close'] * shares
    
    # ForwardPE
    annualized_eps = (aligned['Net Income'].fillna(0.0) * 4.0) / aligned['Shares (Basic)'].fillna(1.0)
    aligned['ForwardPE'] = np.where(annualized_eps > 0, aligned['Close'] / annualized_eps, 0.0)
    
    # EV_EBITDA
    total_debt = aligned['Short Term Debt'].fillna(0.0) + aligned['Long Term Debt'].fillna(0.0)
    cash_col = 'Cash, Cash Equivalents & Short Term Investments'
    if cash_col not in aligned.columns:
        cash_col = 'Cash & Cash Equivalents'
    cash = aligned[cash_col].fillna(0.0)
    EV = aligned['MarketCap'] + total_debt - cash
    
    EBITDA = aligned['Operating Income (Loss)'].fillna(0.0) + aligned['Depreciation & Amortization'].fillna(0.0)
    annualized_ebitda = EBITDA * 4.0
    aligned['EV_EBITDA'] = np.where(annualized_ebitda > 0, EV / annualized_ebitda, 0.0)
    aligned['EBITDA'] = annualized_ebitda
    
    # FCFYield
    capex = aligned['Change in Fixed Assets & Intangibles'].fillna(0.0)
    capex = -capex.abs()
    fcf = aligned['Net Cash from Operating Activities'].fillna(0.0) + capex
    annualized_fcf = fcf * 4.0
    aligned['FCFYield'] = np.where(aligned['MarketCap'] > 0, annualized_fcf / aligned['MarketCap'], 0.0)
    
    # Margins & Returns
    aligned['GrossMargin'] = np.where(aligned['Revenue'] > 0, aligned['Gross Profit'] / aligned['Revenue'], 0.0)
    aligned['OperatingMargin'] = np.where(aligned['Revenue'] > 0, aligned['Operating Income (Loss)'] / aligned['Revenue'], 0.0)
    aligned['ROIC'] = np.where(aligned['Total Assets'] > 0, aligned['Net Income'] / aligned['Total Assets'], 0.0)
    aligned['DebtToEquity'] = np.where(aligned['Total Equity'] > 0, aligned['Total Liabilities'] / aligned['Total Equity'], 0.0)
    
    # ShareholderYield
    divs = aligned['Dividends Paid'].fillna(0.0)
    rep = aligned['Cash from (Repurchase of) Equity'].fillna(0.0)
    debt_cf = aligned['Cash from (Repayment of) Debt'].fillna(0.0)
    total_returned = - (divs + rep + debt_cf)
    annualized_returned = total_returned * 4.0
    aligned['ShareholderYield'] = np.where(aligned['MarketCap'] > 0, annualized_returned / aligned['MarketCap'], 0.0)
    
    # FiveYDivGrowth (rolling 5yr avg yield)
    weekly_div = aligned['Dividends Paid'].fillna(0.0).abs() * 4.0
    div_yield = np.where(aligned['MarketCap'] > 0, weekly_div / aligned['MarketCap'], 0.0)
    aligned['DivYield'] = div_yield
    aligned['FiveYDivGrowth'] = aligned.groupby('Ticker')['DivYield'].transform(lambda x: x.rolling(260, min_periods=1).mean()).fillna(0.0)
    
    # PayoutRatio
    aligned['PayoutRatio'] = np.where(aligned['Net Income'] > 0, aligned['Dividends Paid'].fillna(0.0).abs() / aligned['Net Income'], 0.0)
    
    # CurrentRatio
    aligned['CurrentRatio'] = np.where(aligned['Total Current Liabilities'] > 0, aligned['Total Current Assets'] / aligned['Total Current Liabilities'], 0.0)
    
    # Technical Indicators (Wilder's RSI, 200 SMA)
    aligned = aligned.sort_values(by=['Ticker', 'Date'])
    aligned['RSI'] = aligned.groupby('Ticker')['Close'].transform(lambda x: compute_rsi_series(x)).fillna(0.0)
    aligned['PriceVs200SMA'] = aligned.groupby('Ticker')['Close'].transform(lambda x: x / x.rolling(200, min_periods=1).mean()).fillna(0.0)
    
    # Relative Strength vs SPY
    aligned['Close_1yr_ago'] = aligned.groupby('Ticker')['Close'].shift(52)
    aligned['Stock_Return_1yr'] = (aligned['Close'] / aligned['Close_1yr_ago']) - 1.0
    
    spy_prices = aligned[aligned['Ticker'] == 'SPY'][['Date', 'Close']].rename(columns={'Close': 'SPY_Close'})
    spy_prices['SPY_Close_1yr_ago'] = spy_prices['SPY_Close'].shift(52)
    spy_prices['SPY_Return_1yr'] = (spy_prices['SPY_Close'] / spy_prices['SPY_Close_1yr_ago']) - 1.0
    
    aligned = pd.merge(aligned, spy_prices[['Date', 'SPY_Return_1yr']], on='Date', how='left')
    aligned['RelativeStrength'] = aligned['Stock_Return_1yr'] - aligned['SPY_Return_1yr']
    aligned['RelativeStrength'] = aligned['RelativeStrength'].fillna(0.0)
    
    # Rolling Beta
    print("[INFO] Calculating rolling Beta...")
    prices_pivot = aligned.pivot(index='Date', columns='Ticker', values='Close')
    prices_pivot = prices_pivot.ffill().bfill()
    returns_pivot = prices_pivot.pct_change()
    
    spy_ret = returns_pivot['SPY']
    roll_cov = returns_pivot.rolling(window=156, min_periods=12).cov(spy_ret)
    roll_var = spy_ret.rolling(window=156, min_periods=12).var()
    beta_pivot = roll_cov.div(np.where(roll_var > 0, roll_var, np.nan), axis=0)
    beta_pivot = beta_pivot.fillna(1.0)
    
    beta_long = beta_pivot.reset_index().melt(id_vars='Date', value_name='Beta')
    aligned = pd.merge(aligned, beta_long, on=['Date', 'Ticker'], how='left')
    aligned['Beta'] = aligned['Beta'].fillna(1.0)
    
    # Merge Metadata (LongName, Sector)
    aligned = pd.merge(aligned, metadata[['Ticker', 'LongName', 'Sector']], on='Ticker', how='left')
    
    # 8. Save Prices Cache (for Backtest Engine offline mode)
    print("[INFO] Saving prices cache to historical_prices_cache.pkl...")
    cache_prices = aligned.pivot(index='Date', columns='Ticker', values='Close')
    cache_prices.index = pd.to_datetime(cache_prices.index)
    if cache_prices.index.tz is not None:
        cache_prices.index = cache_prices.index.tz_localize(None)
    for t in ALL_TICKERS:
        if t not in cache_prices.columns:
            cache_prices[t] = np.nan
    cache_prices.to_pickle('historical_prices_cache.pkl')
    
    return aligned

# ─── State Recovery, Flushing, and File Exporters ────────────────────────────
def write_rows_with_flushing(file_path, header, rows, append=False):
    """Write rows in chunks of 10 with active flushing to disk."""
    mode = 'a' if append else 'w'
    file_exists = os.path.exists(file_path)
    write_header = not append or not file_exists
    
    with open(file_path, mode, newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        
        chunk = []
        for r in rows:
            chunk.append(r)
            if len(chunk) >= 10:
                writer.writerows(chunk)
                f.flush()
                chunk.clear()
        if chunk:
            writer.writerows(chunk)
            f.flush()

def fetch_and_save_market_data():
    """Build and write the current/latest static dataset (market_data.csv)."""
    print("=" * 60)
    print("[START] Starting Current Market Data Ingestion Pipeline...")
    print("=" * 60)
    
    # Tickers to skip (already completed in recovery)
    fetched_tickers = set()
    if os.path.exists(OUTPUT_FILE) and os.path.getsize(OUTPUT_FILE) > 0:
        try:
            with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if header and 'Ticker' in header:
                    ticker_idx = header.index('Ticker')
                    for row in reader:
                        if row:
                            fetched_tickers.add(row[ticker_idx])
            print(f"[OK] State Recovery: Found {len(fetched_tickers)} already written tickers.")
        except Exception as e:
            print(f"Warning parsing existing market_data.csv: {e}")
            
    # Filter for target tickers
    target_tickers = [t for t in ALL_TICKERS if t not in ('SPY', 'QQQ')]
    remaining_tickers = [t for t in target_tickers if t not in fetched_tickers]
    
    if not remaining_tickers:
        print("[OK] All tickers already fetched and verified.")
        return
        
    print(f"Ingesting {len(remaining_tickers)} remaining tickers...")
    
    aligned = run_pipeline()
    
    # Get latest row for each remaining ticker
    latest_rows = []
    for ticker in remaining_tickers:
        stock_data = aligned[aligned['Ticker'] == ticker]
        if stock_data.empty:
            continue
        
        # Take the absolute last row (latest date)
        latest = stock_data.iloc[-1]
        
        # Drop Missing Policy: skip if critical parameters are missing/invalid
        if pd.isna(latest['MarketCap']) or latest['MarketCap'] <= 0 or \
           pd.isna(latest['EBITDA']) or latest['EBITDA'] == 0 or \
           pd.isna(latest['CurrentRatio']) or latest['CurrentRatio'] <= 0:
            print(f"[WARN] Skipping {ticker}: missing or invalid critical values (MarketCap, EBITDA, or CurrentRatio).")
            continue
            
        row = [
            ticker,
            round(float(latest['ForwardPE']), 4) if not pd.isna(latest['ForwardPE']) else 0.0,
            round(float(latest['EV_EBITDA']), 4) if not pd.isna(latest['EV_EBITDA']) else 0.0,
            round(float(latest['FCFYield']), 6) if not pd.isna(latest['FCFYield']) else 0.0,
            round(float(latest['RevenueGrowth']), 6) if not pd.isna(latest['RevenueGrowth']) else 0.0,
            round(float(latest['EPSGrowth']), 6) if not pd.isna(latest['EPSGrowth']) else 0.0,
            round(float(latest['GrossMargin']), 6) if not pd.isna(latest['GrossMargin']) else 0.0,
            round(float(latest['OperatingMargin']), 6) if not pd.isna(latest['OperatingMargin']) else 0.0,
            round(float(latest['ROIC']), 6) if not pd.isna(latest['ROIC']) else 0.0,
            round(float(latest['DebtToEquity']), 4) if not pd.isna(latest['DebtToEquity']) else 0.0,
            round(float(latest['ShareholderYield']), 6) if not pd.isna(latest['ShareholderYield']) else 0.0,
            round(float(latest['FiveYDivGrowth']), 6) if not pd.isna(latest['FiveYDivGrowth']) else 0.0,
            round(float(latest['PayoutRatio']), 6) if not pd.isna(latest['PayoutRatio']) else 0.0,
            round(float(latest['Beta']), 4) if not pd.isna(latest['Beta']) else 1.0,
            round(float(latest['RSI']), 4) if not pd.isna(latest['RSI']) else 0.0,
            round(float(latest['PriceVs200SMA']), 6) if not pd.isna(latest['PriceVs200SMA']) else 0.0,
            round(float(latest['RelativeStrength']), 6) if not pd.isna(latest['RelativeStrength']) else 0.0,
            round(float(latest['CurrentRatio']), 4) if not pd.isna(latest['CurrentRatio']) else 0.0,
            str(latest['LongName']) if not pd.isna(latest['LongName']) else '',
            str(latest['Sector']) if not pd.isna(latest['Sector']) else '',
            round(float(latest['MarketCap']), 2) if not pd.isna(latest['MarketCap']) else 0.0,
            round(float(latest['EBITDA']), 2) if not pd.isna(latest['EBITDA']) else 0.0,
        ]
        latest_rows.append(row)
        print(f"[OK] Formatted {ticker}: MarketCap={row[20]}, CurrentRatio={row[17]}, EBITDA={row[21]}")
        
    write_rows_with_flushing(OUTPUT_FILE, CSV_HEADER, latest_rows, append=True)
    print(f"[SUCCESS] Successfully wrote {len(latest_rows)} records to {OUTPUT_FILE}.")

def fetch_and_save_historical_fundamental_data():
    """Build and write the entire historical weekly panel dataset (historical_fundamental_data.csv)."""
    print("=" * 60)
    print("[START] Starting Historical Panel Ingestion Pipeline...")
    print("=" * 60)
    
    aligned = run_pipeline()
    
    # State Recovery Check
    fetched_records = set()
    if os.path.exists(HIST_OUTPUT_FILE) and os.path.getsize(HIST_OUTPUT_FILE) > 0:
        try:
            with open(HIST_OUTPUT_FILE, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if header and 'Ticker' in header and 'Quarter_End_Date' in header:
                    ticker_idx = header.index('Ticker')
                    date_idx = header.index('Quarter_End_Date')
                    for row in reader:
                        if row:
                            fetched_records.add((row[ticker_idx], row[date_idx]))
            print(f"[OK] State Recovery: Found {len(fetched_records)} already written historical rows.")
        except Exception as e:
            print(f"Warning parsing existing historical_fundamental_data.csv: {e}")
            
    target_tickers = [t for t in ALL_TICKERS if t not in ('SPY', 'QQQ')]
    
    historical_rows = []
    skipped_count = 0
    
    for ticker in target_tickers:
        stock_data = aligned[aligned['Ticker'] == ticker]
        if stock_data.empty:
            continue
        
        print(f"Processing {ticker} ({len(stock_data)} weeks)...")
        
        for _, row_data in stock_data.iterrows():
            date_str = row_data['Date'].strftime('%Y-%m-%d')
            if (ticker, date_str) in fetched_records:
                continue
                
            # Drop Missing Policy: skip if critical parameters are missing/invalid
            if pd.isna(row_data['MarketCap']) or row_data['MarketCap'] <= 0 or \
               pd.isna(row_data['EBITDA']) or row_data['EBITDA'] == 0 or \
               pd.isna(row_data['CurrentRatio']) or row_data['CurrentRatio'] <= 0:
                skipped_count += 1
                continue
                
            row = [
                ticker,
                str(row_data['Sector']) if not pd.isna(row_data['Sector']) else '',
                date_str,
                round(float(row_data['ForwardPE']), 4) if not pd.isna(row_data['ForwardPE']) else 0.0,
                round(float(row_data['EV_EBITDA']), 4) if not pd.isna(row_data['EV_EBITDA']) else 0.0,
                round(float(row_data['FCFYield']), 6) if not pd.isna(row_data['FCFYield']) else 0.0,
                round(float(row_data['RevenueGrowth']), 6) if not pd.isna(row_data['RevenueGrowth']) else 0.0,
                round(float(row_data['EPSGrowth']), 6) if not pd.isna(row_data['EPSGrowth']) else 0.0,
                round(float(row_data['GrossMargin']), 6) if not pd.isna(row_data['GrossMargin']) else 0.0,
                round(float(row_data['OperatingMargin']), 6) if not pd.isna(row_data['OperatingMargin']) else 0.0,
                round(float(row_data['ROIC']), 6) if not pd.isna(row_data['ROIC']) else 0.0,
                round(float(row_data['DebtToEquity']), 4) if not pd.isna(row_data['DebtToEquity']) else 0.0,
                round(float(row_data['ShareholderYield']), 6) if not pd.isna(row_data['ShareholderYield']) else 0.0,
                round(float(row_data['FiveYDivGrowth']), 6) if not pd.isna(row_data['FiveYDivGrowth']) else 0.0,
                round(float(row_data['PayoutRatio']), 6) if not pd.isna(row_data['PayoutRatio']) else 0.0,
                round(float(row_data['Beta']), 4) if not pd.isna(row_data['Beta']) else 1.0,
                round(float(row_data['RSI']), 4) if not pd.isna(row_data['RSI']) else 0.0,
                round(float(row_data['PriceVs200SMA']), 6) if not pd.isna(row_data['PriceVs200SMA']) else 0.0,
                round(float(row_data['RelativeStrength']), 6) if not pd.isna(row_data['RelativeStrength']) else 0.0,
                round(float(row_data['CurrentRatio']), 4) if not pd.isna(row_data['CurrentRatio']) else 0.0,
                round(float(row_data['MarketCap']), 2) if not pd.isna(row_data['MarketCap']) else 0.0,
                round(float(row_data['EBITDA']), 2) if not pd.isna(row_data['EBITDA']) else 0.0,
            ]
            historical_rows.append(row)
            
    if not historical_rows:
        print("[OK] No new historical records to write.")
        return
        
    write_rows_with_flushing(HIST_OUTPUT_FILE, HIST_HEADER, historical_rows, append=True)
    print(f"[SUCCESS] Successfully wrote {len(historical_rows)} records to {HIST_OUTPUT_FILE} (skipped {skipped_count} invalid records).")

if __name__ == '__main__':
    if '--historical' in sys.argv:
        fetch_and_save_historical_fundamental_data()
    else:
        fetch_and_save_market_data()
