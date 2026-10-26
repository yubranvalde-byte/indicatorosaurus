import bisect
import gc
import math
import os
import time
import pandas as pd
from financial_module import normalize_stocks, run_financial_engine, _safe_float, _scalar_float

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

    csv_path = os.path.join('data', 'historical_fundamental_data.csv')
    if not os.path.exists(csv_path):
        raise ValueError("Historical fundamental data file not found. Please run the ingestion process first.")
        
    needed_fund_cols = [
        'Ticker', 'GICS_Sector', 'Sector', 'Quarter_End_Date', 'MarketCap', 'EBITDA',
        'TrailingPE', 'EV_EBITDA', 'FCFYield', 'RevenueGrowth', 'EPSGrowth',
        'GrossMargin', 'OperatingMargin', 'ROIC', 'NetDebt_EBITDA', 'ShareholderYield',
        'FiveYDivGrowth', 'PayoutRatio', 'Beta', 'RSI', 'PriceVs200SMA',
        'RelativeStrength', 'CurrentRatio', 'Total_Equity', 'FCF', 'EBIT', 'Interest_Expense'
    ]

    t0_csv = time.perf_counter()
    df = pd.read_csv(csv_path, usecols=lambda c: c in needed_fund_cols)
    t1_csv = time.perf_counter()
    print(f"[PROFILER] Backtest Fundamental CSV Loading: {t1_csv - t0_csv:.3f}s")
    if df.empty:
        raise ValueError("Historical fundamental data file is empty.")

    # Downcast float64 to float32 immediately to cut memory footprint
    float_cols = df.select_dtypes(include=['float64']).columns
    if not float_cols.empty:
        df[float_cols] = df[float_cols].astype('float32')

    # Parse dates and sort chronologically, ensuring timezone-naive
    df['Quarter_End_Date'] = pd.to_datetime(df['Quarter_End_Date']).dt.tz_localize(None)
    df.sort_values('Quarter_End_Date', inplace=True)
    unique_dates = sorted(df['Quarter_End_Date'].unique())
    if not unique_dates:
        raise ValueError("No historical quarter dates found in data.")

    # Determine rebalancing frequency based on strategy profile
    rebalance_freqs = {
        'momentum': 3,
        'hyper-growth': 6,
        'quality': 6,
        'deep-value': 3,
        'income': 3,
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

    # ── PHASE 1: Fundamental Screening & Portfolio Selection (FIRST) ─────────
    prelim_periods = []
    for k, t_k in enumerate(selected_anchors):
        t_k_dt = pd.Timestamp(t_k).tz_localize(None)
        hold_start = (t_k_dt + pd.DateOffset(days=1)).tz_localize(None)
        if hold_start < start_date:
            hold_start = start_date
            
        hold_end = pd.Timestamp(selected_anchors[k+1]).tz_localize(None) if k < len(selected_anchors) - 1 else pd.Timestamp(end_date).tz_localize(None)
        if hold_start <= hold_end:
            prelim_periods.append({
                'anchor': t_k_dt,
                'start': hold_start,
                'end': hold_end,
                'tickers': []
            })

    last_valid_top_10 = []
    needed_tickers = set()

    t0_phase1 = time.perf_counter()
    for p in prelim_periods:
        t0_q = time.perf_counter()
        t_k = p['anchor']
        panel_slice = df[df['Quarter_End_Date'] <= t_k]
        if panel_slice.empty:
            p['tickers'] = []
            del panel_slice
            continue

        # Memory Optimization: df is pre-sorted, so drop_duplicates('last') is identical to groupby('last') but O(N) faster and avoids RAM copies
        snapshot_df = panel_slice.drop_duplicates(subset=['Ticker'], keep='last')
        del panel_slice  # Free panel slice immediately

        if snapshot_df.empty:
            p['tickers'] = []
            del snapshot_df
            continue

        raw_records = snapshot_df.to_dict(orient='records')
        del snapshot_df  # Free snapshot DataFrame immediately

        raw_snapshot = []
        for r in raw_records:
            ticker = str(r.get('Ticker', '')).strip().upper()
            if not ticker or pd.isna(r.get('Ticker')):
                continue

            mkt_cap = _safe_float(r.get('MarketCap', 0.0))
            if mkt_cap <= 0.0:
                continue

            r['Ticker'] = ticker
            r['Sector'] = str(r.get('GICS_Sector') or r.get('Sector') or '').strip()
            r['MarketCap'] = mkt_cap
            r['EBITDA'] = _safe_float(r.get('EBITDA', 0.0))
            raw_snapshot.append(r)
        del raw_records

        try:
            normalized_snapshot = normalize_stocks(
                raw_snapshot,
                strategy=strategy,
                allowed_sectors=allowed_sectors,
                min_market_cap=min_market_cap,
                max_market_cap=max_market_cap,
                global_stocks_list=raw_snapshot
            )

            # Max 2 attempts: if empty, fall back directly to baseline unfiltered snapshot
            if not normalized_snapshot:
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
                del scored_stocks
            del normalized_snapshot
        except Exception as e:
            print(f"Error scoring portfolio at {t_k}: {e}")
            top_10 = []
        finally:
            del raw_snapshot

        if top_10:
            p['tickers'] = top_10
            last_valid_top_10 = top_10
        elif last_valid_top_10:
            p['tickers'] = last_valid_top_10
        else:
            p['tickers'] = ['SPY']

        needed_tickers.update(p['tickers'])
        t1_q = time.perf_counter()
        print(f"[PROFILER] Quarter {t_k.date()} Processing: {t1_q - t0_q:.3f}s")
        gc.collect()  # Explicit inner-loop garbage collection per quarter iteration

    t1_phase1 = time.perf_counter()
    print(f"[PROFILER] Phase 1 Total Fundamental Screening: {t1_phase1 - t0_phase1:.3f}s")
    needed_tickers.add('SPY')
    print(f"[Backtest Engine] Identified {len(needed_tickers)} unique target tickers for lazy price ingestion: {sorted(needed_tickers)}")

    # Reclaim fundamental DataFrame memory before price loading
    del df
    gc.collect()

    # ── PHASE 2: Lazy Price Ingestion (Filtered SECOND - >99% RAM Saving) ──────
    t0_p2 = time.perf_counter()
    simfin_parquet_path = os.path.join('data', 'raw_simfin', 'us-shareprices-daily.parquet')
    simfin_csv_path = os.path.join('data', 'raw_simfin', 'us-shareprices-daily.csv')

    price_data = None
    if os.path.exists(simfin_parquet_path):
        min_date_filter = pd.Timestamp(start_date) - pd.Timedelta(days=5)
        max_date_filter = pd.Timestamp(end_date) + pd.Timedelta(days=5)
        raw_p = pd.read_parquet(
            simfin_parquet_path,
            engine='pyarrow',
            columns=['Date', 'Ticker', 'Adj. Close', 'Close'],
            filters=[
                ('Ticker', 'in', list(needed_tickers)),
                ('Date', '>=', min_date_filter),
                ('Date', '<=', max_date_filter)
            ]
        )
        raw_p['Date'] = pd.to_datetime(raw_p['Date'])
        raw_p['Price'] = raw_p['Adj. Close'].fillna(raw_p['Close']).astype('float32')
        raw_p.drop(columns=['Adj. Close', 'Close'], inplace=True, errors='ignore')
        price_data = raw_p.pivot(index='Date', columns='Ticker', values='Price').astype('float32')
        del raw_p
        gc.collect()
        t1_p2 = time.perf_counter()
        print(f"[PROFILER] Phase 2 Price Parquet I/O: {t1_p2 - t0_p2:.3f}s")
        print(f"[Backtest Engine] Lazy-loaded prices for {len(price_data.columns)} tickers from SimFin Parquet (date filtered {start_date.date()} to {end_date.date()}).")
    elif os.path.exists(simfin_csv_path):
        raw_p = pd.read_csv(
            simfin_csv_path,
            sep=";",
            usecols=['Date', 'Ticker', 'Adj. Close', 'Close'],
            dtype={'Close': 'float32', 'Adj. Close': 'float32'}
        )
        raw_p = raw_p[raw_p['Ticker'].isin(needed_tickers)]
        raw_p['Date'] = pd.to_datetime(raw_p['Date'])
        raw_p['Price'] = raw_p['Adj. Close'].fillna(raw_p['Close']).astype('float32')
        raw_p.drop(columns=['Adj. Close', 'Close'], inplace=True, errors='ignore')
        price_data = raw_p.pivot(index='Date', columns='Ticker', values='Price').astype('float32')
        del raw_p
        gc.collect()
        t1_p2 = time.perf_counter()
        print(f"[PROFILER] Phase 2 Price CSV I/O: {t1_p2 - t0_p2:.3f}s")
        print(f"[Backtest Engine] Lazy-loaded prices for {len(price_data.columns)} tickers from SimFin CSV.")

    if price_data is None or price_data.empty:
        raise ValueError("Historical price data is missing for selected target universe.")

    # Extract adjusted close prices
    close_data = price_data

    # Ensure close_data's index is timezone-naive
    if hasattr(close_data.index, 'tz') and close_data.index.tz is not None:
        close_data.index = close_data.index.tz_localize(None)

    # Load SPY offline history and merge into close_data
    spy_path = os.path.join('data', 'spy_historical.csv')
    if not os.path.exists(spy_path):
        raise ValueError(f"Benchmark SPY historical data is missing at {spy_path}.")
    
    spy_df = pd.read_csv(spy_path, usecols=['Date', 'SPY_Adj_Close'], dtype={'SPY_Adj_Close': 'float32'})
    spy_df['Date'] = pd.to_datetime(spy_df['Date'])
    spy_df = spy_df.set_index('Date')
    if hasattr(spy_df.index, 'tz') and spy_df.index.tz is not None:
        spy_df.index = spy_df.index.tz_localize(None)
    
    # Memory Optimization: Use pd.concat instead of in-place assignment to avoid memory block reallocation spikes
    close_data = pd.concat([close_data, spy_df['SPY_Adj_Close'].rename('SPY').astype('float32')], axis=1)
    close_data['SPY'] = close_data['SPY'].ffill()
    del spy_df
    gc.collect()

    trading_dates = close_data.index
    effective_start = trading_dates[trading_dates >= start_date][0] if any(trading_dates >= start_date) else trading_dates[0]

    # Finalize periods with exact trading_dates bounds and filter valid_tickers
    periods = []
    for k, p_pre in enumerate(prelim_periods):
        t_k = p_pre['anchor']
        hold_start = p_pre['start']
        hold_end = pd.Timestamp(selected_anchors[k+1]).tz_localize(None) if k < len(selected_anchors) - 1 else pd.Timestamp(trading_dates[-1]).tz_localize(None)
        
        valid_tickers = [t for t in p_pre['tickers'] if t in close_data.columns]
        if not valid_tickers:
            valid_tickers = ['SPY']

        if hold_start <= hold_end:
            periods.append({
                'anchor': t_k,
                'start': hold_start,
                'end': hold_end,
                'tickers': valid_tickers
            })

    # Daily simulation loop: compute percentage changes downcasted to float32
    daily_returns = close_data.pct_change(fill_method=None).astype('float32', copy=False)
    del close_data
    if price_data is not None:
        del price_data
    gc.collect()

    portfolio_daily_returns = []
    spy_daily_returns = []

    # Clamp effective_start forward to the first actual holding period start.
    if periods:
        first_period_start = periods[0]['start']
        trading_on_or_after = trading_dates[trading_dates >= first_period_start]
        if len(trading_on_or_after) > 0:
            effective_start = max(effective_start, trading_on_or_after[0])

    simulation_dates = [d for d in trading_dates if d >= effective_start]
    period_starts = [p['start'] for p in periods]

    for d in simulation_dates:
        # Binary-search: find the rightmost period whose start <= d
        idx = bisect.bisect_right(period_starts, d) - 1
        active_period = periods[idx] if idx >= 0 and d <= periods[idx]['end'] else None

        if active_period and active_period['tickers']:
            tickers_for_day = active_period['tickers']
            day_slice = daily_returns.loc[d, tickers_for_day]

            valid_returns = day_slice.dropna()
            if valid_returns.empty:
                port_ret = 0.0
            else:
                port_ret = _scalar_float(valid_returns.mean())
        else:
            port_ret = 0.0

        spy_ret_raw = daily_returns.loc[d, 'SPY']
        spy_ret = _scalar_float(spy_ret_raw)

        portfolio_daily_returns.append(port_ret)
        spy_daily_returns.append(spy_ret)

    port_series = pd.Series(portfolio_daily_returns, index=simulation_dates, dtype='float32')
    spy_series = pd.Series(spy_daily_returns, index=simulation_dates, dtype='float32')
    del daily_returns, portfolio_daily_returns, spy_daily_returns
    gc.collect()

    if len(port_series) > 0:
        port_series.iloc[0] = 0.0
        spy_series.iloc[0] = 0.0

    portfolio_cumulative = (1.0 + port_series).cumprod().fillna(1.0)
    spy_cumulative = (1.0 + spy_series).cumprod().fillna(1.0)

    # Compute final metrics
    num_days = len(port_series)
    if num_days > 1:
        annual_factor = 252.0 / num_days
        p_last = max(0.0001, _scalar_float(portfolio_cumulative.iloc[-1]))
        s_last = max(0.0001, _scalar_float(spy_cumulative.iloc[-1]))
        port_ann_return = float(p_last ** annual_factor - 1.0)
        spy_ann_return = float(s_last ** annual_factor - 1.0)

        valid_port = port_series.dropna()
        valid_spy = spy_series.dropna()

        port_vol = _scalar_float(valid_port.std()) * math.sqrt(252) if len(valid_port) > 1 else 0.0
        spy_vol = _scalar_float(valid_spy.std()) * math.sqrt(252) if len(valid_spy) > 1 else 0.0

        if math.isnan(port_vol) or math.isinf(port_vol):
            port_vol = 0.0
        if math.isnan(spy_vol) or math.isinf(spy_vol):
            spy_vol = 0.0

        # 4% risk free hurdle
        port_sharpe = (port_ann_return - 0.04) / port_vol if port_vol > 0 else 0.0
        spy_sharpe = (spy_ann_return - 0.04) / spy_vol if spy_vol > 0 else 0.0

        if math.isnan(port_sharpe) or math.isinf(port_sharpe):
            port_sharpe = 0.0
        if math.isnan(spy_sharpe) or math.isinf(spy_sharpe):
            spy_sharpe = 0.0

        port_peak = portfolio_cumulative.cummax()
        port_dd = (portfolio_cumulative - port_peak) / port_peak
        port_max_dd = _scalar_float(port_dd.min())

        spy_peak = spy_cumulative.cummax()
        spy_dd = (spy_cumulative - spy_peak) / spy_peak
        spy_max_dd = _scalar_float(spy_dd.min())
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
            "portfolio_return": round(_scalar_float(portfolio_cumulative.iloc[idx]), 4),
            "spy_return": round(_scalar_float(spy_cumulative.iloc[idx]), 4)
        })

    # Cleanup temporary calculation objects before returning
    del port_series, spy_series, portfolio_cumulative, spy_cumulative
    gc.collect()

    return {
        "performance_series": performance_series,
        "metrics": {
            "portfolio_sharpe": round(port_sharpe, 4),
            "spy_sharpe": round(spy_sharpe, 4),
            "portfolio_max_drawdown": round(port_max_dd, 4),
            "spy_max_drawdown": round(spy_max_dd, 4),
            "total_alpha": round(alpha, 4),
            # Canonical unified metric keys for CLI & cross-module reporting
            "portfolio_annualized_return": round(port_ann_return, 4),
            "spy_annualized_return": round(spy_ann_return, 4),
            "portfolio_volatility": round(port_vol, 4),
            "spy_volatility": round(spy_vol, 4),
        }
    }

