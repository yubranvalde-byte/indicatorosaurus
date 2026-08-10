import bisect
import math
import os
import pandas as pd
from financial_module import normalize_stocks, run_financial_engine, _safe_float

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

    # Select anchor dates with step size
    selected_anchors = []
    for idx, anchor in enumerate(window_anchors):
        if idx % step == 0:
            selected_anchors.append(anchor)

    # Load historical price series from local cache or raw SimFin prices (Pure Offline Mode)
    cache_path = 'historical_prices_cache.pkl'
    simfin_prices_path = os.path.join('data', 'raw_simfin', 'us-shareprices-daily.csv')
    
    price_data = None
    if os.path.exists(cache_path):
        try:
            price_data = pd.read_pickle(cache_path)
            print("[Backtest Engine] Loaded prices from local cache.")
        except Exception as e:
            print(f"[Backtest Engine] Notice: Could not load pickle cache ({e}). Loading from SimFin shareprices...")

    if price_data is None and os.path.exists(simfin_prices_path):
        raw_p = pd.read_csv(simfin_prices_path, sep=";")
        raw_p['Date'] = pd.to_datetime(raw_p['Date'])
        raw_p['Price'] = raw_p['Adj. Close'].fillna(raw_p['Close'])
        price_data = raw_p.pivot(index='Date', columns='Ticker', values='Price')
        print("[Backtest Engine] Loaded prices from SimFin daily shareprices CSV.")

    if price_data is None or price_data.empty:
        raise ValueError(f"Historical price data is missing or empty. Please ensure '{cache_path}' or '{simfin_prices_path}' exists.")

    # Extract adjusted close prices.
    if isinstance(price_data.columns, pd.MultiIndex):
        lvl0 = price_data.columns.get_level_values(0).unique().tolist()
        price_col = 'Close' if 'Close' in lvl0 else ('Adj Close' if 'Adj Close' in lvl0 else lvl0[0])
        close_data = price_data[price_col]
    else:
        close_data = price_data

    # Ensure close_data's index is timezone-naive
    if hasattr(close_data.index, 'tz') and close_data.index.tz is not None:
        close_data.index = close_data.index.tz_localize(None)

    # Drop columns that are entirely NaN across the whole window
    # (delisted or bad tickers). Keep SPY unconditionally.
    all_nan_cols = [c for c in close_data.columns if close_data[c].isna().all() and c != 'SPY']
    if all_nan_cols:
        close_data = close_data.drop(columns=all_nan_cols)

    if 'SPY' not in close_data.columns:
        raise ValueError("Benchmark SPY is missing from local historical price cache.")

    trading_dates = close_data.index

    # Clamp start_date to the first actual trading day present in price data.
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
    last_valid_top_10 = []

    for p in periods:
        t_k = p['anchor']
        # Slice historical fundamental panel at or prior to T_k (Zero Look-Ahead Bias)
        panel_slice = df[df['Quarter_End_Date'] <= t_k]
        if panel_slice.empty:
            p['tickers'] = []
            continue

        # Extract the latest fundamental record per ticker available as of T_k
        snapshot_df = panel_slice.sort_values('Quarter_End_Date').groupby('Ticker').last().reset_index()
        if snapshot_df.empty:
            p['tickers'] = []
            continue

        raw_snapshot = []
        for _, row in snapshot_df.iterrows():
            ticker = str(row.get('Ticker', '')).strip().upper()
            if not ticker or pd.isna(row.get('Ticker')):
                continue

            # Ensure valid non-zero market cap
            mkt_cap = _safe_float(row.get('MarketCap', 0.0))
            if mkt_cap <= 0.0:
                continue

            d = {}
            for col in snapshot_df.columns:
                d[col] = row[col]
            d['Ticker'] = ticker
            d['Sector'] = str(d.get('GICS_Sector') or d.get('Sector') or '').strip()
            d['MarketCap'] = mkt_cap
            d['EBITDA'] = _safe_float(d.get('EBITDA', 0.0))
            raw_snapshot.append(d)

        try:
            # 1. Primary scoring attempt with requested strategy & filters
            normalized_snapshot = normalize_stocks(
                raw_snapshot,
                strategy=strategy,
                allowed_sectors=allowed_sectors,
                min_market_cap=min_market_cap,
                max_market_cap=max_market_cap,
                global_stocks_list=raw_snapshot
            )

            # 2. If strategy knockout filter dropped all stocks, fallback to relaxed strategy gates
            if not normalized_snapshot and strategy is not None:
                print(f"[Backtest Engine] Strategy '{strategy}' knocked out all stocks at {t_k}. Relaxing strategy knockout filters.")
                normalized_snapshot = normalize_stocks(
                    raw_snapshot,
                    strategy=None,
                    allowed_sectors=allowed_sectors,
                    min_market_cap=min_market_cap,
                    max_market_cap=max_market_cap,
                    global_stocks_list=raw_snapshot
                )

            # 3. If dynamic filters (sector/cap) also resulted in 0 stocks, fallback to raw snapshot
            if not normalized_snapshot:
                print(f"[Backtest Engine] Dynamic filters knocked out all stocks at {t_k}. Falling back to baseline snapshot.")
                normalized_snapshot = normalize_stocks(
                    raw_snapshot,
                    strategy=None,
                    allowed_sectors=None,
                    min_market_cap=None,
                    max_market_cap=None,
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

    port_series = pd.Series(portfolio_daily_returns, index=simulation_dates)
    spy_series = pd.Series(spy_daily_returns, index=simulation_dates)

    if len(port_series) > 0:
        port_series.iloc[0] = 0.0
        spy_series.iloc[0] = 0.0

    portfolio_cumulative = (1.0 + port_series).cumprod().fillna(1.0)
    spy_cumulative = (1.0 + spy_series).cumprod().fillna(1.0)

    # Compute final metrics
    num_days = len(port_series)
    if num_days > 1:
        annual_factor = 252.0 / num_days
        p_last = max(0.0001, float(portfolio_cumulative.iloc[-1]))
        s_last = max(0.0001, float(spy_cumulative.iloc[-1]))
        port_ann_return = float(p_last ** annual_factor - 1.0)
        spy_ann_return = float(s_last ** annual_factor - 1.0)

        port_vol = float(port_series.std() * math.sqrt(252))
        spy_vol = float(spy_series.std() * math.sqrt(252))

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
