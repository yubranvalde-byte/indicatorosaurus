import math
import os
import time
import numpy as np
import pandas as pd
from financial_module import (
    normalize_stocks,
    run_financial_engine,
    compute_vectorized_tech_series,
    discover_parquet_max_date,
    load_parquet_price_matrix,
    _release_arrow_memory,
    _safe_float,
    _scalar_float,
)

# In-memory cache for historical fundamentals and daily price matrices across backtest runs
_BACKTEST_DATA_CACHE = {
    'fund_df': None,
    'fund_mtime': 0.0,
    'price_df': None,
    'spy_s': None,
    'price_mtime': 0.0,
}


def _get_backtest_datasets():
    """
    Load and cache historical fundamentals (with point-in-time Effective_Date),
    daily SimFin price matrix (Date x Ticker, float32), and SPY benchmark series.
    """
    global _BACKTEST_DATA_CACHE
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)

    csv_candidates = [
        os.path.join(project_root, 'data', 'historical_fundamental_data.csv'),
        os.path.join('data', 'historical_fundamental_data.csv'),
    ]
    csv_path = next((p for p in csv_candidates if os.path.exists(p)), None)
    if not csv_path:
        raise ValueError("Historical fundamental data file not found. Please run the ingestion process first.")

    needed_fund_cols = {
        'Ticker', 'GICS_Sector', 'Sector', 'Quarter_End_Date', 'Publish_Date', 'MarketCap', 'EBITDA',
        'TrailingPE', 'EV_EBITDA', 'FCFYield', 'RevenueGrowth', 'EPSGrowth',
        'GrossMargin', 'OperatingMargin', 'ROIC', 'NetDebt_EBITDA', 'ShareholderYield',
        'FiveYDivGrowth', 'PayoutRatio', 'Beta', 'RSI', 'PriceVs200SMA',
        'RelativeStrength', 'CurrentRatio', 'Total_Equity', 'FCF', 'EBIT', 'Interest_Expense'
    }

    fund_mtime = os.path.getmtime(csv_path)
    if _BACKTEST_DATA_CACHE['fund_df'] is None or _BACKTEST_DATA_CACHE['fund_mtime'] != fund_mtime:
        t0_csv = time.perf_counter()
        df = pd.read_csv(csv_path, usecols=lambda c: c in needed_fund_cols)
        if df.empty:
            raise ValueError("Historical fundamental data file is empty.")

        float_cols = df.select_dtypes(include=['float64']).columns
        if not float_cols.empty:
            df[float_cols] = df[float_cols].astype('float32')

        df['Ticker'] = df['Ticker'].astype(str).str.strip().str.upper()
        if 'Sector' not in df.columns:
            df['Sector'] = df['GICS_Sector'].fillna('').astype(str).str.strip() if 'GICS_Sector' in df.columns else ''
        else:
            df['Sector'] = df['Sector'].fillna(df.get('GICS_Sector', '')).astype(str).str.strip()

        df['Quarter_End_Date'] = pd.to_datetime(df['Quarter_End_Date']).dt.tz_localize(None)

        # Point-in-Time Effective_Date resolution (Publish_Date -> raw SimFin Publish Date -> Quarter_End_Date + 45d lag)
        if 'Publish_Date' in df.columns:
            df['Effective_Date'] = pd.to_datetime(df['Publish_Date'], errors='coerce').dt.tz_localize(None)
            df['Effective_Date'] = df['Effective_Date'].fillna(df['Quarter_End_Date'] + pd.Timedelta(days=45))
        else:
            inc_candidates = [
                os.path.join(project_root, 'data', 'raw_simfin', 'us-income-ttm.csv'),
                os.path.join('data', 'raw_simfin', 'us-income-ttm.csv'),
            ]
            inc_path = next((p for p in inc_candidates if os.path.exists(p)), None)
            if inc_path:
                try:
                    inc_df = pd.read_csv(inc_path, sep=';', usecols=['Ticker', 'Report Date', 'Publish Date'])
                    inc_df['Ticker'] = inc_df['Ticker'].astype(str).str.strip().str.upper()
                    inc_df['Quarter_End_Date'] = pd.to_datetime(inc_df['Report Date'], errors='coerce').dt.tz_localize(None)
                    inc_df['Pub_DT'] = pd.to_datetime(inc_df['Publish Date'], errors='coerce').dt.tz_localize(None)
                    inc_df = inc_df.dropna(subset=['Ticker', 'Quarter_End_Date']).drop_duplicates(
                        subset=['Ticker', 'Quarter_End_Date'], keep='last'
                    )
                    df = df.merge(inc_df[['Ticker', 'Quarter_End_Date', 'Pub_DT']], on=['Ticker', 'Quarter_End_Date'], how='left')
                    df['Effective_Date'] = df['Pub_DT'].fillna(df['Quarter_End_Date'] + pd.Timedelta(days=45))
                    df.drop(columns=['Pub_DT'], inplace=True, errors='ignore')
                except Exception:
                    df['Effective_Date'] = df['Quarter_End_Date'] + pd.Timedelta(days=45)
            else:
                df['Effective_Date'] = df['Quarter_End_Date'] + pd.Timedelta(days=45)

        # Ensure Effective_Date is never earlier than Quarter_End_Date
        df['Effective_Date'] = df[['Quarter_End_Date', 'Effective_Date']].max(axis=1)
        df = df[df['MarketCap'].fillna(0.0) > 0.0].copy()
        df.sort_values(['Quarter_End_Date', 'Effective_Date'], inplace=True)
        df.reset_index(drop=True, inplace=True)

        _BACKTEST_DATA_CACHE['fund_df'] = df
        _BACKTEST_DATA_CACHE['fund_mtime'] = fund_mtime
        t1_csv = time.perf_counter()
        print(f"[PROFILER] Backtest Fundamental CSV Loading & PIT Alignment: {t1_csv - t0_csv:.3f}s")
    else:
        df = _BACKTEST_DATA_CACHE['fund_df']

    parquet_candidates = [
        os.path.join(project_root, 'data', 'us-shareprices-daily.parquet'),
        os.path.join(project_root, 'data', 'raw_simfin', 'us-shareprices-daily.parquet'),
        os.path.join('data', 'us-shareprices-daily.parquet'),
        os.path.join('data', 'raw_simfin', 'us-shareprices-daily.parquet'),
    ]
    simfin_parquet_path = next((p for p in parquet_candidates if os.path.exists(p)), None)
    simfin_csv_candidates = [
        os.path.join(project_root, 'data', 'raw_simfin', 'us-shareprices-daily.csv'),
        os.path.join('data', 'raw_simfin', 'us-shareprices-daily.csv'),
    ]
    simfin_csv_path = next((p for p in simfin_csv_candidates if os.path.exists(p)), None)

    spy_candidates = [
        os.path.join(project_root, 'data', 'spy_historical.csv'),
        os.path.join('data', 'spy_historical.csv'),
    ]
    spy_path = next((p for p in spy_candidates if os.path.exists(p)), None)
    if not spy_path:
        raise ValueError("Benchmark SPY historical data is missing.")

    price_source_path = simfin_parquet_path or simfin_csv_path
    if not price_source_path:
        raise ValueError("Historical price data is missing for selected target universe.")

    price_mtime = os.path.getmtime(price_source_path)
    if _BACKTEST_DATA_CACHE['price_df'] is None or _BACKTEST_DATA_CACHE['price_mtime'] != price_mtime:
        t0_p2 = time.perf_counter()
        universe_tickers = df['Ticker'].unique().tolist()
        if simfin_parquet_path:
            max_ds_date = discover_parquet_max_date(simfin_parquet_path)
            min_fund_dt = df['Quarter_End_Date'].min() if not df.empty else (max_ds_date - pd.DateOffset(years=5))
            min_parquet_date = min(min_fund_dt, max_ds_date - pd.DateOffset(years=6)) - pd.Timedelta(days=370)
            price_df, _ = load_parquet_price_matrix(
                simfin_parquet_path,
                tickers=universe_tickers,
                min_date=min_parquet_date,
                include_volume=False,
            )
        else:
            raw_p = pd.read_csv(
                simfin_csv_path,
                sep=';',
                usecols=['Date', 'Ticker', 'Adj. Close', 'Close'],
                dtype={'Close': 'float32', 'Adj. Close': 'float32'},
            )
            raw_p = raw_p[raw_p['Ticker'].isin(universe_tickers)]
            raw_p['Date'] = pd.to_datetime(raw_p['Date']).dt.tz_localize(None)
            raw_p['Price'] = raw_p['Adj. Close'].fillna(raw_p['Close']).astype('float32')
            raw_p.drop(columns=['Adj. Close', 'Close'], inplace=True, errors='ignore')
            price_df = raw_p.pivot(index='Date', columns='Ticker', values='Price').sort_index().astype('float32')
            del raw_p
        _release_arrow_memory()

        spy_df = pd.read_csv(spy_path, usecols=['Date', 'SPY_Adj_Close'], dtype={'SPY_Adj_Close': 'float32'})
        spy_df['Date'] = pd.to_datetime(spy_df['Date']).dt.tz_localize(None)
        spy_s = spy_df.set_index('Date')['SPY_Adj_Close'].sort_index().astype('float32')
        del spy_df

        _BACKTEST_DATA_CACHE['price_df'] = price_df
        _BACKTEST_DATA_CACHE['spy_s'] = spy_s
        _BACKTEST_DATA_CACHE['price_mtime'] = price_mtime
        t1_p2 = time.perf_counter()
        print(f"[PROFILER] Backtest Price Matrix Hydration ({len(price_df.columns)} tickers): {t1_p2 - t0_p2:.3f}s")
    else:
        price_df = _BACKTEST_DATA_CACHE['price_df']
        spy_s = _BACKTEST_DATA_CACHE['spy_s']

    return df, price_df, spy_s


def run_backtest(tickers, time_window='3y', weights=None, strategy=None, allowed_sectors=None, min_market_cap=None, max_market_cap=None):
    """
    Perform a rolling-window rebalancing simulation using point-in-time historical fundamentals and prices.
    """
    # 1. Clean and validate inputs
    window_clean = str(time_window or '3y').strip().lower()
    if window_clean not in ('1y', '3y', '5y', '10y'):
        window_clean = '3y'

    if not weights or len(weights) != 16:
        weights = [1.0] * 16

    df, full_price_df, spy_full = _get_backtest_datasets()
    if df.empty or full_price_df.empty:
        raise ValueError("Historical fundamental or price dataset is empty.")

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

    # Anchor simulation end_date to the actual offline price & benchmark coverage horizon
    now_ts = pd.Timestamp.now().normalize().tz_localize(None)
    max_price_date = pd.Timestamp(full_price_df.index[-1]).tz_localize(None)
    max_spy_date = pd.Timestamp(spy_full.index[-1]).tz_localize(None)
    end_date = min(now_ts, max_price_date, max_spy_date)

    years_map = {'1y': 1, '3y': 3, '5y': 5, '10y': 10}
    n_years = years_map[window_clean]
    target_start = (end_date - pd.DateOffset(years=n_years)).tz_localize(None)

    # Earliest date where a meaningful universe of point-in-time filings (>= 100 tickers) is published
    min_pit_date = df['Effective_Date'].nsmallest(min(100, len(df))).iloc[-1]
    min_price_date = pd.Timestamp(full_price_df.index[0]).tz_localize(None)
    start_date = max(target_start, min_pit_date, min_price_date)
    if start_date >= end_date:
        raise ValueError("No matching historical quarter anchors in the selected time window.")

    # Build true calendar rebalancing schedule spaced by `freq_months` months across [start_date, end_date]
    selected_anchors = []
    curr_anchor = start_date
    while curr_anchor < end_date:
        selected_anchors.append(curr_anchor)
        curr_anchor = (curr_anchor + pd.DateOffset(months=freq_months)).tz_localize(None)

    if not selected_anchors:
        selected_anchors = [start_date]

    # ── PHASE 1: Point-in-Time Fundamental & Technical Screening ─────────────
    prelim_periods = []
    for k, t_k in enumerate(selected_anchors):
        t_k_dt = pd.Timestamp(t_k).tz_localize(None)
        hold_start = start_date if k == 0 else (t_k_dt + pd.DateOffset(days=1)).tz_localize(None)
        hold_end = pd.Timestamp(selected_anchors[k + 1]).tz_localize(None) if k < len(selected_anchors) - 1 else end_date
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
        # Strictly filter by Effective_Date <= t_k (zero look-ahead bias on financial statements)
        panel_slice = df[df['Effective_Date'] <= t_k]
        if panel_slice.empty:
            p['tickers'] = last_valid_top_10 if last_valid_top_10 else ['SPY']
            needed_tickers.update(p['tickers'])
            continue

        snapshot_df = panel_slice.drop_duplicates(subset=['Ticker'], keep='last').copy()

        # Recompute Point-in-Time Technical Indicators & Price strictly from prices <= t_k
        price_slice = full_price_df.loc[:t_k].tail(260)
        spy_slice = spy_full.loc[:t_k].tail(260)
        if len(price_slice) >= 15 and len(spy_slice) >= 2:
            valid_cols = [t for t in snapshot_df['Ticker'] if t in price_slice.columns]
            if valid_cols:
                rs_s, sma_s, rsi_s, beta_s, last_px_s = compute_vectorized_tech_series(
                    price_slice[valid_cols], spy_slice, log_diag=False
                )
                snapshot_df['RelativeStrength'] = snapshot_df['Ticker'].map(rs_s).fillna(snapshot_df['RelativeStrength']).astype('float32')
                snapshot_df['PriceVs200SMA'] = snapshot_df['Ticker'].map(sma_s).fillna(snapshot_df['PriceVs200SMA']).astype('float32')
                snapshot_df['RSI'] = snapshot_df['Ticker'].map(rsi_s).fillna(snapshot_df['RSI']).astype('float32')
                snapshot_df['Beta'] = snapshot_df['Ticker'].map(beta_s).fillna(snapshot_df['Beta']).astype('float32')
                snapshot_df['LatestPrice'] = snapshot_df['Ticker'].map(last_px_s).fillna(5.01).astype('float32')

        try:
            normalized_snapshot = normalize_stocks(
                snapshot_df,
                strategy=strategy,
                allowed_sectors=allowed_sectors,
                min_market_cap=min_market_cap,
                max_market_cap=max_market_cap,
                global_stocks_list=snapshot_df,
                use_live_tech_cache=False,
                return_df=True,
            )

            # Max 2 attempts: if empty, fall back directly to baseline unfiltered snapshot
            if normalized_snapshot.empty:
                normalized_snapshot = normalize_stocks(
                    snapshot_df,
                    strategy=None,
                    allowed_sectors=None,
                    min_market_cap=None,
                    max_market_cap=None,
                    global_stocks_list=snapshot_df,
                    use_live_tech_cache=False,
                    return_df=True,
                )

            if normalized_snapshot.empty:
                top_10 = []
            else:
                scored_stocks = run_financial_engine(normalized_snapshot, weights)
                top_10 = [s['Ticker'] for s in scored_stocks[:10]]
                del scored_stocks
            del normalized_snapshot
        except Exception as e:
            print(f"Error scoring portfolio at {t_k}: {e}")
            top_10 = []
        del snapshot_df

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

    t1_phase1 = time.perf_counter()
    print(f"[PROFILER] Phase 1 Total Fundamental Screening: {t1_phase1 - t0_phase1:.3f}s")
    needed_tickers.add('SPY')
    print(f"[Backtest Engine] Identified {len(needed_tickers)} unique target tickers for lazy price ingestion: {sorted(needed_tickers)}")

    # ── PHASE 2: Vectorized Daily Return Simulation ──────────────────────────
    t0_p2 = time.perf_counter()
    target_stock_cols = [t for t in sorted(needed_tickers) if t in full_price_df.columns]
    close_data = full_price_df.loc[start_date:end_date, target_stock_cols].copy()
    if close_data.empty:
        raise ValueError("Historical price data is missing for selected target universe.")

    # Align SPY strictly to the active stock price trading calendar [start_date, end_date]
    close_data['SPY'] = spy_full.reindex(close_data.index).ffill().bfill().astype('float32')
    # Forward-fill short holiday/missing quote gaps (up to 5 trading days) without extending dead/delisted series
    close_data = close_data.ffill(limit=5)

    trading_dates = close_data.index
    effective_start = trading_dates[0]

    periods = []
    for k, p_pre in enumerate(prelim_periods):
        t_k = p_pre['anchor']
        hold_start = p_pre['start']
        hold_end = pd.Timestamp(selected_anchors[k + 1]).tz_localize(None) if k < len(selected_anchors) - 1 else pd.Timestamp(trading_dates[-1]).tz_localize(None)

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

    daily_returns = close_data.pct_change(fill_method=None).astype('float32', copy=False)
    if periods:
        first_period_start = periods[0]['start']
        trading_on_or_after = trading_dates[trading_dates >= first_period_start]
        if len(trading_on_or_after) > 0:
            effective_start = max(effective_start, trading_on_or_after[0])

    sim_returns = daily_returns.loc[daily_returns.index >= effective_start]
    simulation_dates = sim_returns.index
    n_sim = len(simulation_dates)

    port_arr = np.zeros(n_sim, dtype=np.float32)
    spy_arr = np.nan_to_num(sim_returns['SPY'].to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    for p in periods:
        mask = (simulation_dates >= p['start']) & (simulation_dates <= p['end'])
        if not mask.any():
            continue
        sub_mat = sim_returns.loc[mask, p['tickers']].to_numpy(dtype=np.float32)
        valid_counts = np.sum(~np.isnan(sub_mat), axis=1)
        row_sums = np.nansum(sub_mat, axis=1)
        mean_rets = np.where(valid_counts > 0, row_sums / np.maximum(valid_counts, 1), 0.0)
        port_arr[mask] = np.nan_to_num(mean_rets, nan=0.0, posinf=0.0, neginf=0.0)

    if n_sim > 0:
        port_arr[0] = 0.0
        spy_arr[0] = 0.0

    port_series = pd.Series(port_arr, index=simulation_dates, dtype='float32')
    spy_series = pd.Series(spy_arr, index=simulation_dates, dtype='float32')
    t1_p2 = time.perf_counter()
    print(f"[PROFILER] Phase 2 Vectorized Return Simulation ({n_sim} trading days): {t1_p2 - t0_p2:.3f}s")

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

    # Format output series vectorized
    date_strs = portfolio_cumulative.index.strftime('%Y-%m-%d')
    p_vals = np.round(portfolio_cumulative.to_numpy(dtype=np.float64), 4)
    s_vals = np.round(spy_cumulative.to_numpy(dtype=np.float64), 4)
    performance_series = [
        {"date": d_s, "portfolio_return": float(p_v), "spy_return": float(s_v)}
        for d_s, p_v, s_v in zip(date_strs, p_vals, s_vals)
    ]

    del close_data, daily_returns, sim_returns
    _release_arrow_memory()

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

