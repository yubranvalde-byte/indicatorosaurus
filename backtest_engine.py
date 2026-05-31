import bisect
import gc
import math
import os
import pandas as pd

def run_backtest(tickers, time_window='3y', weights=None, strategy=None, allowed_sectors=None, min_market_cap=None, max_market_cap=None):
    """
    Perform a rolling-window rebalancing simulation using historical fundamentals and prices.
    
    Parameters:
      tickers (list): Dynamic current leaderboard tickers (ignored as we rank dynamically).
      time_window (str): Trailing window ('3y' or '5y').
      weights (list): 16-element indicator weights list.
      strategy (str): Active strategy preset identifier.
      allowed_sectors (list): Selected sector filters.
      min_market_cap (float): Minimum market cap threshold.
      max_market_cap (float): Maximum market cap threshold.
      
    Returns:
      dict: Performance series and metrics.
    """
    # 1. Clean and validate inputs
    if time_window not in ('3y', '5y'):
        time_window = '3y'

    if not weights or len(weights) != 16:
        weights = [1.0] * 16

    csv_path = 'historical_fundamental_data.csv'
    if not os.path.exists(csv_path):
        raise ValueError("Historical fundamental data file not found. Please run the ingestion process first.")
        
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("Historical fundamental data file is empty.")

    # Parse dates and sort chronologically, ensuring timezone-naive
    df['Quarter_End_Date'] = pd.to_datetime(df['Quarter_End_Date']).dt.tz_localize(None)
    unique_dates = sorted(df['Quarter_End_Date'].unique())
    if not unique_dates:
        raise ValueError("No historical quarter dates found in data.")

    # Determine rebalancing frequency based on strategy profile
    rebalance_freqs = {
        'momentum': 3,
        'hyper-growth': 6,
        'quality': 6,
        'deep-value': 12,
        'income': 12,
        'custom': 3
    }
    strategy_clean = str(strategy).strip().lower() if strategy else 'custom'
    if strategy_clean not in rebalance_freqs:
        strategy_clean = strategy_clean.replace('_', '-')
    freq_months = rebalance_freqs.get(strategy_clean, 3)
    step = freq_months // 3  # step size in terms of quarters

    # Determine start and end dates for the simulation window, ensuring timezone-naive
    end_date = pd.Timestamp.now().normalize().tz_localize(None)
    if time_window == '5y':
        start_date = (end_date - pd.DateOffset(years=5)).tz_localize(None)
    else:
        start_date = (end_date - pd.DateOffset(years=3)).tz_localize(None)

    # Establish window anchors, including a one-quarter buffer before start_date
    first_idx = 0
    for i, d in enumerate(unique_dates):
        if d >= start_date:
            first_idx = max(0, i - 1)
            break
    window_anchors = [d for d in unique_dates[first_idx:] if d <= end_date]
    if not window_anchors:
        raise ValueError("No matching historical quarter anchors in the selected time window.")

    # Select anchor dates with freq_months spacing to handle weekly frequency correctly
    selected_anchors = []
    if window_anchors:
        current_anchor = window_anchors[0]
        selected_anchors.append(current_anchor)
        for anchor in window_anchors:
            if anchor >= current_anchor + pd.DateOffset(months=freq_months):
                current_anchor = anchor
                selected_anchors.append(current_anchor)

    # Fetch historical price series baseline for the entire static universe upfront (weekly interval)
    unique_tickers = list(df['Ticker'].dropna().unique())
    unique_tickers = [t.strip().upper() for t in unique_tickers if t]
    all_symbols = list(set(unique_tickers + ['SPY']))

    # Load prices using a pure offline strategy
    cache_path = 'historical_prices_cache.pkl'
    price_data = None
    
    if os.path.exists(cache_path):
        try:
            price_data = pd.read_pickle(cache_path)
            print("[Backtest Engine] Loaded prices from local pickle cache.")
        except Exception as e:
            print(f"[Backtest Engine] Error reading local pickle cache: {e}.")
            price_data = None

    if price_data is None:
        csv_cache_path = os.path.join('simfin_data_cache', 'us-shareprices-daily.csv')
        if os.path.exists(csv_cache_path):
            print(f"[Backtest Engine] Local pickle cache '{cache_path}' not found or failed. "
                  f"Reading prices from offline CSV '{csv_cache_path}' in chunks...")
            try:
                chunks = []
                # To prevent memory bloat, read only specific columns and filter each chunk
                for chunk in pd.read_csv(csv_cache_path, sep=';', usecols=['Ticker', 'Date', 'Adj. Close', 'Close'], chunksize=100000):
                    chunk['Ticker'] = chunk['Ticker'].str.strip().str.upper()
                    # Filter chunk to keep only symbols that are part of our universe
                    chunk_filtered = chunk[chunk['Ticker'].isin(all_symbols)].copy()
                    chunks.append(chunk_filtered)
                
                raw_prices = pd.concat(chunks)
                del chunks
                
                # Check column mapping
                price_col = 'Adj. Close' if 'Adj. Close' in raw_prices.columns else 'Close'
                raw_prices['Date'] = pd.to_datetime(raw_prices['Date'])
                
                # Pivot
                price_data = raw_prices.pivot(index='Date', columns='Ticker', values=price_col)
                del raw_prices
                print("[Backtest Engine] Loaded and pivoted prices from offline CSV.")
            except Exception as e:
                print(f"[Backtest Engine] Error reading offline CSV: {e}")
                price_data = None

    if price_data is None:
        raise FileNotFoundError(
            f"Local price cache '{cache_path}' and offline CSV are missing or incomplete. "
            "Please run 'fetch_static_data.py' to download and cache the offline price dataset."
        )

    if price_data.empty:
        raise ValueError(f"Local price cache '{cache_path}' is empty.")

    # Downcast and handle MultiIndex
    price_data = price_data.astype('float32', errors='ignore')

    if isinstance(price_data.columns, pd.MultiIndex):
        lvl0 = price_data.columns.get_level_values(0).unique().tolist()
        price_col = 'Close' if 'Close' in lvl0 else ('Adj Close' if 'Adj Close' in lvl0 else lvl0[0])
        close_data = price_data[price_col].copy()
    else:
        close_data = price_data.copy()

    del price_data
    gc.collect()

    # Ensure close_data index is timezone-naive
    if hasattr(close_data.index, 'tz') and close_data.index.tz is not None:
        close_data.index = close_data.index.tz_localize(None)
    close_data.index = pd.to_datetime(close_data.index)
    close_data = close_data.astype('float32', errors='ignore')

    # Strict Drop-Missing Policy
    if 'SPY' not in close_data.columns:
        raise ValueError("Benchmark SPY is missing from the local price cache.")

    valid_symbols = []
    dropped_symbols = []

    for ticker in unique_tickers:
        if ticker not in close_data.columns:
            dropped_symbols.append((ticker, "missing from cache columns"))
            continue
        
        # Check if ticker has any valid (non-NaN) prices within the requested backtest window
        ticker_series = close_data.loc[start_date:end_date, ticker]
        if ticker_series.isna().all():
            dropped_symbols.append((ticker, "no valid prices in backtest window"))
            continue
            
        valid_symbols.append(ticker)

    if dropped_symbols:
        for ticker, reason in dropped_symbols:
            print(f"[Backtest Engine] Warning: Ticker '{ticker}' is dropped from active universe. Reason: {reason}.")
        
        dropped_set = {t for t, _ in dropped_symbols}
        df = df[~df['Ticker'].isin(dropped_set)]
        unique_tickers = [t for t in unique_tickers if t not in dropped_set]
        all_symbols = list(set(unique_tickers + ['SPY']))

    # Keep only active symbols in close_data and free unused columns to optimize memory
    close_data = close_data[all_symbols].copy()
    gc.collect()

    # Reindex close_data to include selected anchors and prevent day-of-week KeyError
    anchor_dates = pd.DatetimeIndex([pd.Timestamp(a).tz_localize(None) for a in selected_anchors])
    combined_index = close_data.index.union(anchor_dates).sort_values()
    close_data = close_data.reindex(combined_index).ffill().bfill()

    trading_dates = close_data.index

    # Clamp start_date to the first actual trading day present in price data.
    # This prevents the compounding chain from being zeroed-out on weekend /
    # holiday days that fall before the first real close price row.
    effective_start = trading_dates[trading_dates >= start_date][0] if any(trading_dates >= start_date) else trading_dates[0]

    # Establish interval holding periods (ensuring all dates are timezone-naive)
    periods = []
    for k, t_k in enumerate(selected_anchors):
        t_k = pd.Timestamp(t_k).tz_localize(None)
        hold_start = (t_k + pd.DateOffset(days=1)).tz_localize(None)
        if hold_start < start_date:
            hold_start = start_date
            
        if k < len(selected_anchors) - 1:
            hold_end = pd.Timestamp(selected_anchors[k+1]).tz_localize(None)
        else:
            hold_end = pd.Timestamp(trading_dates[-1]).tz_localize(None)
            
        if hold_start <= hold_end:
            periods.append({
                'anchor': t_k,
                'start': hold_start,
                'end': hold_end
            })

    # Rank and select portfolios for each interval
    from financial_module import normalize_stocks, run_financial_engine
    last_valid_top_10 = []

    for p in periods:
        t_k = p['anchor']
        snapshot_df = df[df['Quarter_End_Date'] == t_k]
        if snapshot_df.empty:
            p['tickers'] = []
            continue

        raw_snapshot = []
        for _, row in snapshot_df.iterrows():
            ticker = str(row.get('Ticker', '')).strip().upper()
            if not ticker or ticker not in close_data.columns:
                continue
                
            # Dynamic price availability check at anchor date t_k
            try:
                price_val = close_data.loc[t_k, ticker]
                if pd.isna(price_val) or price_val <= 0.0:
                    continue
            except KeyError:
                continue

            # Check if any critical parameters (MarketCap, EBITDA, CurrentRatio, or any of the 16 indicators) are missing
            critical_cols = [
                'MarketCap', 'EBITDA', 'CurrentRatio', 'ForwardPE', 'EV_EBITDA',
                'FCFYield', 'RevenueGrowth', 'EPSGrowth', 'GrossMargin',
                'OperatingMargin', 'ROIC', 'DebtToEquity', 'ShareholderYield',
                'FiveYDivGrowth', 'PayoutRatio', 'Beta', 'RSI', 'PriceVs200SMA',
                'RelativeStrength'
            ]
            
            is_missing = False
            for col in critical_cols:
                val = row.get(col)
                if pd.isna(val):
                    is_missing = True
                    break
                # Special cases: MarketCap, EBITDA, and CurrentRatio must be strictly positive (>0)
                if col in ('MarketCap', 'EBITDA', 'CurrentRatio') and float(val) <= 0.0:
                    is_missing = True
                    break
            
            if is_missing or pd.isna(row.get('GICS_Sector')) or str(row.get('GICS_Sector')).strip() == '':
                continue

            d = {}
            for col in snapshot_df.columns:
                d[col] = row[col]
            d['Sector'] = d.get('GICS_Sector', '')
            d['MarketCap'] = float(d.get('MarketCap', 0.0))
            d['EBITDA'] = float(d.get('EBITDA', 0.0))
            raw_snapshot.append(d)

        try:
            normalized_snapshot = normalize_stocks(
                raw_snapshot,
                strategy=strategy,
                allowed_sectors=allowed_sectors,
                min_market_cap=min_market_cap,
                max_market_cap=max_market_cap,
                global_stocks_list=raw_snapshot
            )
            if not normalized_snapshot:
                top_10 = []
            else:
                scored_stocks = run_financial_engine(normalized_snapshot, weights)
                top_10 = [s['Ticker'] for s in scored_stocks[:10]]
        except Exception as e:
            print(f"Error scoring portfolio at {t_k}: {e}")
            top_10 = []

        # Filter tickers present in price data
        valid_tickers = [t for t in top_10 if t in close_data.columns]
        if valid_tickers:
            p['tickers'] = valid_tickers
            last_valid_top_10 = valid_tickers
        else:
            if last_valid_top_10:
                p['tickers'] = last_valid_top_10
                print(f"[Backtest Engine] Warning: Empty portfolio at {t_k}. Falling back to previous quarter portfolio.")
            else:
                p['tickers'] = ['SPY']
                print(f"[Backtest Engine] Warning: Empty portfolio at {t_k} with no previous history. Falling back to SPY.")

    # Daily simulation loop
    daily_returns = close_data.pct_change()
    portfolio_daily_returns = []
    spy_daily_returns = []

    # Clamp effective_start forward to the first actual holding period start.
    # Without this, days between effective_start and periods[0]['start'] have
    # no active period and return 0.0 every day, locking the equity curve at 1.0
    # for the entire early portion of the chart.
    if periods:
        first_period_start = periods[0]['start']
        # Find the first real trading day >= first_period_start
        trading_on_or_after = trading_dates[trading_dates >= first_period_start]
        if len(trading_on_or_after) > 0:
            effective_start = max(effective_start, trading_on_or_after[0])

    simulation_dates = [d for d in trading_dates if d >= effective_start]

    # Build a sorted list of (start, end) boundaries for O(log n) lookup.
    # For each day d we binary-search to find which holding period it belongs to
    # rather than doing an O(n) scan through `periods`.
    period_starts = [p['start'] for p in periods]

    for d in simulation_dates:
        # Binary-search: find the rightmost period whose start <= d
        idx = bisect.bisect_right(period_starts, d) - 1
        active_period = periods[idx] if idx >= 0 and d <= periods[idx]['end'] else None

        if active_period and active_period['tickers']:
            tickers_for_day = active_period['tickers']
            day_slice = daily_returns.loc[d, tickers_for_day]

            # Guard: if EVERY ticker in the portfolio has NaN for this day
            # (e.g. market holiday, all assets pre-IPO), preserve capital at 0.0.
            valid_returns = day_slice.dropna()
            if valid_returns.empty:
                port_ret = 0.0
            else:
                port_ret = float(valid_returns.mean())
                if math.isnan(port_ret) or math.isinf(port_ret):
                    port_ret = 0.0
        else:
            port_ret = 0.0

        spy_ret_raw = daily_returns.loc[d, 'SPY']
        spy_ret = float(spy_ret_raw) if not (math.isnan(float(spy_ret_raw)) or math.isinf(float(spy_ret_raw))) else 0.0

        portfolio_daily_returns.append(port_ret)
        spy_daily_returns.append(spy_ret)

    del daily_returns
    gc.collect()

    port_series = pd.Series(portfolio_daily_returns, index=simulation_dates)
    spy_series = pd.Series(spy_daily_returns, index=simulation_dates)

    if len(port_series) > 0:
        port_series.iloc[0] = 0.0
        spy_series.iloc[0] = 0.0

    portfolio_cumulative = (1.0 + port_series).cumprod().fillna(1.0)
    spy_cumulative = (1.0 + spy_series).cumprod().fillna(1.0)

    # Compute final metrics
    num_weeks = len(port_series)
    if num_weeks > 1:
        annual_factor = 52.0 / num_weeks
        p_last = max(0.0001, float(portfolio_cumulative.iloc[-1]))
        s_last = max(0.0001, float(spy_cumulative.iloc[-1]))
        port_ann_return = float(p_last ** annual_factor - 1.0)
        spy_ann_return = float(s_last ** annual_factor - 1.0)

        port_vol = float(port_series.std() * math.sqrt(52))
        spy_vol = float(spy_series.std() * math.sqrt(52))

        # 4% risk free hurdle
        port_sharpe = (port_ann_return - 0.04) / port_vol if port_vol > 0 else 0.0
        spy_sharpe = (spy_ann_return - 0.04) / spy_vol if spy_vol > 0 else 0.0

        if math.isnan(port_sharpe) or math.isinf(port_sharpe):
            port_sharpe = 0.0
        if math.isnan(spy_sharpe) or math.isinf(spy_sharpe):
            spy_sharpe = 0.0

        port_peak = portfolio_cumulative.cummax()
        port_dd = (portfolio_cumulative - port_peak) / port_peak
        port_max_dd = float(port_dd.min())

        spy_peak = spy_cumulative.cummax()
        spy_dd = (spy_cumulative - spy_peak) / spy_peak
        spy_max_dd = float(spy_dd.min())
    else:
        port_ann_return, spy_ann_return = 0.0, 0.0
        port_vol, spy_vol = 0.0, 0.0
        port_sharpe, spy_sharpe = 0.0, 0.0
        port_max_dd, spy_max_dd = 0.0, 0.0

    alpha = port_ann_return - spy_ann_return
    if math.isnan(alpha) or math.isinf(alpha):
        alpha = 0.0
    if math.isnan(port_max_dd) or math.isinf(port_max_dd):
        port_max_dd = 0.0
    if math.isnan(spy_max_dd) or math.isinf(spy_max_dd):
        spy_max_dd = 0.0

    # Format output series
    performance_series = []
    for idx, date in enumerate(portfolio_cumulative.index):
        date_str = date.strftime('%Y-%m-%d')
        performance_series.append({
            "date": date_str,
            "portfolio_return": round(float(portfolio_cumulative.iloc[idx]), 4),
            "spy_return": round(float(spy_cumulative.iloc[idx]), 4)
        })

    return {
        "performance_series": performance_series,
        "metrics": {
            "portfolio_sharpe": round(port_sharpe, 4),
            "spy_sharpe": round(spy_sharpe, 4),
            "portfolio_max_drawdown": round(port_max_dd, 4),
            "spy_max_drawdown": round(spy_max_dd, 4),
            "total_alpha": round(alpha, 4)
        }
    }
