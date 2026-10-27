"""
portfolio_analytics.py
======================
Portfolio Intelligence — Phase 3:
Vectorized daily time-series reconstruction, TWR vs. MWR (XIRR) mathematical engine,
FX & tax attribution breakdown, time-slicing/rebasing, multi-asset benchmarking,
and cash-flow matched What-If opportunity simulator.

Key Capabilities:
1. Vectorized Daily Holdings Matrix (N_days x M_tickers) & Dual-Currency Valuation (USD & ILS):
   - Supports both Explicit Cash Mode (when CASH_DEPOSIT/CASH_WITHDRAWAL exist) and
     Implicit Invested-Equity Mode (default when only BUY/SELL/DIVIDEND exist).
2. TWR vs. MWR (XIRR) Mathematical Engine:
   - Exact daily sub-period Time-Weighted Return (TWR):
       r_t = (V_t - V_{t-1} - CF_t) / (V_{t-1} + max(0.0, CF_t))
       R_TWR,t = prod_{k=1..t}(1 + r_k) - 1
   - Exact Money-Weighted Return (MWR / XIRR) via Newton-Raphson + Brent/bisection fallback.
   - FX & Tax Attribution Breakdown (in ILS view): Pure Asset Return, FX Impact, and Tax & Fee Drag.
3. Time-Slicing, Rebasing & Multi-Benchmark Comparison:
   - Window presets ('1M', '3M', '6M', 'YTD', '1Y', '3Y', '5Y', 'ALL') and custom [start_date, end_date].
   - Rebases portfolio and up to 3 benchmark tickers to 0.0% (and index 100.0) at slice start t_0.
   - What-If Opportunity Simulator: Simulates investing every historical net external cash flow into
     each benchmark ticker on the exact same dates.
   - Institutional Risk KPIs: Annualized Return, Volatility, Sharpe (Rf=4%), Sortino, Beta, Alpha,
     Underwater Drawdown series, Maximum Drawdown (MDD), and realized Win-Rate / Payoff Ratio.
"""

import math
import os
import time
import numpy as np
import pandas as pd

from financial_module import (
    INDICATOR_COLUMNS,
    NUM_INDICATORS,
    _scalar_float,
    _clean_frame,
    _to_naive_ts,
    compute_live_technical_indicators,
    ensure_tech_cache,
    get_cached_fundamental_snapshot,
    normalize_stocks,
    run_financial_engine,
    sanitize_for_json,
)
from portfolio_ledger import (
    CAPITAL_GAINS_TAX_RATE,
    CASH_TX_TYPES,
    DIVIDEND_WITHHOLDING_RATE,
    PortfolioLedger,
    TX_TYPE_PRIORITY,
    align_pre_split_dataframe,
)

RISK_FREE_ANNUAL = 0.04
TRADING_DAYS_PER_YEAR = 252.0
CALENDAR_DAYS_PER_YEAR = 365.0
WINDOW_PRESETS = ("1M", "3M", "6M", "YTD", "1Y", "3Y", "5Y", "ALL")

# 5 Composite Factor Pillars mapping the 16 C-FFI indicators
FACTOR_PILLARS = {
    "Valuation": ["TrailingPE", "EV_EBITDA", "FCFYield"],
    "Growth": ["RevenueGrowth", "EPSGrowth"],
    "Quality": ["GrossMargin", "OperatingMargin", "ROIC"],
    "Safety & Income": ["NetDebt_EBITDA", "ShareholderYield", "FiveYDivGrowth", "PayoutRatio", "Beta"],
    "Momentum": ["RSI", "PriceVs200SMA", "RelativeStrength"],
}

KNOWN_ETF_TICKERS = frozenset({
    "SPY", "QQQ", "SMH", "IWM", "DIA", "VOO", "VTI", "SOXX", "ARKK",
    "XLK", "XLF", "XLV", "XLE", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC",
    "GLD", "SLV", "TLT", "IEF", "SHY", "HYG", "LQD", "EEM", "EFA", "VEA", "VWO",
})

_UNIVERSE_DNA_CACHE = {
    "timestamp": 0.0,
    "norm_by_ticker": {},
    "raw_by_ticker": {},
    "meta_by_ticker": {},
    "baseline_indicators": {},
    "baseline_pillars": {},
    "scores_by_weights": {},
}
_DNA_CACHE_TTL = 1800.0


# ─── Date & Window Helpers ────────────────────────────────────────────────────

def resolve_window_bounds(
    inception_date: pd.Timestamp,
    end_date: pd.Timestamp,
    window: str = "ALL",
    custom_start=None,
    custom_end=None,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Resolve (t0, t1) slice bounds for a preset window or custom [start, end] range."""
    inc = _to_naive_ts(inception_date)
    t1 = _to_naive_ts(custom_end) if custom_end is not None else _to_naive_ts(end_date)
    if t1 < inc:
        t1 = inc

    if custom_start is not None:
        t0 = max(inc, _to_naive_ts(custom_start))
        return (min(t0, t1), t1)

    w = str(window or "ALL").upper().strip()
    if w == "1M":
        t0 = t1 - pd.DateOffset(months=1)
    elif w == "3M":
        t0 = t1 - pd.DateOffset(months=3)
    elif w == "6M":
        t0 = t1 - pd.DateOffset(months=6)
    elif w == "YTD":
        t0 = pd.Timestamp(year=t1.year, month=1, day=1)
    elif w == "1Y":
        t0 = t1 - pd.DateOffset(years=1)
    elif w == "3Y":
        t0 = t1 - pd.DateOffset(years=3)
    elif w == "5Y":
        t0 = t1 - pd.DateOffset(years=5)
    else:
        t0 = inc

    t0 = max(inc, _to_naive_ts(t0))
    return (min(t0, t1), t1)


# ─── MWR / XIRR Root-Finding Engine ───────────────────────────────────────────

def compute_xirr(cash_flows: list[tuple[pd.Timestamp, float]], guess: float = 0.10) -> float:
    """
    Compute annualized Internal Rate of Return (XIRR) over irregularly spaced dated cash flows.
    Convention:
      - Contributions into the portfolio are NEGATIVE (CF < 0).
      - Withdrawals and terminal portfolio valuation are POSITIVE (CF > 0).
    Solves: sum_k [ CF_k / (1 + r)^((d_k - d_0) / 365.0) ] = 0
    Uses Newton-Raphson with analytical derivative and bracketed bisection fallback.
    """
    if not cash_flows or len(cash_flows) < 2:
        return 0.0

    # Aggregate cash flows occurring on the exact same calendar date
    by_date: dict[pd.Timestamp, float] = {}
    for dt, amt in cash_flows:
        ts = _to_naive_ts(dt)
        by_date[ts] = by_date.get(ts, 0.0) + _scalar_float(amt)

    dates = sorted(by_date.keys())
    amounts = np.array([by_date[d] for d in dates], dtype=np.float64)

    # Filter out zero entries
    nonzero = np.abs(amounts) > 1e-9
    if nonzero.sum() < 2:
        return 0.0
    amounts = amounts[nonzero]
    dates = [d for d, keep in zip(dates, nonzero) if keep]

    # XIRR requires at least one negative and one positive cash flow
    if not (np.any(amounts < 0.0) and np.any(amounts > 0.0)):
        return 0.0

    d0 = dates[0]
    years = np.array([(d - d0).days / CALENDAR_DAYS_PER_YEAR for d in dates], dtype=np.float64)
    if np.max(years) <= 0.0:
        return 0.0

    def _npv_and_deriv(r: float) -> tuple[float, float]:
        base = 1.0 + r
        if base <= 1e-6:
            return float("inf"), 0.0
        disc = np.power(base, -years)
        f_val = float(np.dot(amounts, disc))
        df_val = float(np.dot(amounts, -years * disc / base))
        return f_val, df_val

    # 1. Newton-Raphson iterations
    r = float(guess)
    for _ in range(80):
        f_val, df_val = _npv_and_deriv(r)
        if abs(f_val) < 1e-8:
            return _scalar_float(r)
        if abs(df_val) < 1e-12:
            break
        step = f_val / df_val
        r_next = r - step
        if r_next <= -0.9999:
            r_next = max(-0.9999, (r - 0.9999) * 0.5)
        elif r_next > 100.0:
            r_next = min(100.0, r + 5.0)
        if abs(r_next - r) < 1e-9:
            return _scalar_float(r_next)
        r = r_next

    # 2. Bracketed Bisection / Secant Fallback over [-0.999, 50.0]
    grid = [-0.99, -0.90, -0.75, -0.50, -0.25, 0.0, 0.10, 0.25, 0.50, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0]
    vals = [_npv_and_deriv(g)[0] for g in grid]
    lo, hi = None, None
    for i in range(len(grid) - 1):
        if vals[i] == 0.0:
            return _scalar_float(grid[i])
        if (vals[i] < 0.0 < vals[i + 1]) or (vals[i] > 0.0 > vals[i + 1]):
            lo, hi = grid[i], grid[i + 1]
            f_lo = vals[i]
            break

    if lo is None or hi is None:
        return 0.0

    for _ in range(120):
        mid = 0.5 * (lo + hi)
        f_mid, _ = _npv_and_deriv(mid)
        if abs(f_mid) < 1e-8 or (hi - lo) < 1e-9:
            return _scalar_float(mid)
        if (f_lo < 0.0 and f_mid < 0.0) or (f_lo > 0.0 and f_mid > 0.0):
            lo, f_lo = mid, f_mid
        else:
            hi = mid

    return _scalar_float(0.5 * (lo + hi))


def compute_modified_dietz_return(
    v_start: float,
    v_end: float,
    dated_external_flows: pd.Series,
    t0: pd.Timestamp,
    t1: pd.Timestamp,
) -> float:
    """
    Compute total Money-Weighted Return (Modified Dietz) over [t0, t1]:
      R_Dietz = (V_end - V_start - sum(CF)) / (V_start + sum(w_t * CF_t))
    where w_t = (t1 - t) / (t1 - t0), and on the inception day (when V_start == 0) w_{t0} = 1.0.
    """
    total_days = max(1.0, float((t1 - t0).days))
    cf_s = dated_external_flows.loc[t0:t1]
    if cf_s.empty:
        return _scalar_float((v_end - v_start) / v_start) if v_start > 0.0 else 0.0

    cf_vals = cf_s.to_numpy(dtype=np.float64)
    total_cf = float(np.sum(cf_vals))
    day_offsets = np.array([(t1 - d).days / total_days for d in cf_s.index], dtype=np.float64)
    weighted_cf = float(np.dot(cf_vals, day_offsets))

    denom = v_start + weighted_cf
    if denom <= 1e-9:
        denom = max(v_start + float(np.sum(np.maximum(0.0, cf_vals))), 1e-9)
    return _scalar_float((v_end - v_start - total_cf) / denom)


# ─── Vectorized Daily Time-Series Reconstruction Engine ──────────────────────

class PortfolioAnalytics:
    """
    Vectorized daily time-series reconstruction, TWR/XIRR performance attribution,
    and multi-asset benchmark comparison engine built on top of `PortfolioLedger`.
    """

    def __init__(
        self,
        ledger: PortfolioLedger,
        price_provider=None,
        macro_provider=None,
    ):
        self.ledger = ledger
        if price_provider is None:
            from price_resolver import resolve_prices as price_provider
        self.price_provider = price_provider
        self.macro_provider = macro_provider or ledger.macro_provider
        self._cached_daily_series: dict = {}
        self._cached_slices: dict = {}
        self._cached_quant_dna: dict = {}

    def invalidate_cache(self):
        """Invalidate memoized daily time series when the underlying ledger mutates."""
        self._cached_daily_series.clear()
        self._cached_slices.clear()
        self._cached_quant_dna.clear()

    def build_daily_series(self, end_date=None, cash_mode: str = "auto") -> pd.DataFrame:
        """
        Reconstruct a continuous daily portfolio time-series DataFrame from the first
        transaction date through `end_date` in both `USD` and `ILS`.

        cash_mode:
          - 'auto': explicit cash if any CASH_DEPOSIT/WITHDRAWAL exists, else implicit
          - 'explicit': forces explicit cash accounting
          - 'implicit': forces implicit invested-equity accounting

        Columns returned:
          - holdings_value_usd, holdings_value_ils
          - cash_balance_usd, cash_balance_ils
          - cumulative_dividends_usd, cumulative_dividends_ils
          - cost_basis_usd, cost_basis_ils
          - nav_usd, nav_ils
          - external_cf_usd, external_cf_ils
          - external_cf_const_fx_ils
          - nav_const_fx_ils
          - twr_daily_usd, twr_daily_ils, twr_daily_const_fx
          - twr_cum_usd, twr_cum_ils, twr_cum_const_fx
          - cumulative_fees_ils, cumulative_realized_tax_ils
          - usd_ils, il_cpi
        """
        end_ts_key = str(end_date)[:10] if end_date is not None else "latest"
        cache_key = (end_ts_key, str(cash_mode).lower())
        if cache_key in self._cached_daily_series:
            return self._cached_daily_series[cache_key].copy()
        tx_df = self.ledger.transactions()
        if tx_df.empty:
            return pd.DataFrame()

        t_start = _to_naive_ts(tx_df["date"].min())
        t_end = _to_naive_ts(end_date) if end_date is not None else max(_to_naive_ts(tx_df["date"].max()), pd.Timestamp.now().normalize())
        if t_end < t_start:
            t_end = t_start

        tx_in_range = tx_df[tx_df["date"] <= t_end.strftime("%Y-%m-%d")].copy()
        if tx_in_range.empty:
            return pd.DataFrame()

        daily_grid = pd.date_range(t_start, t_end, freq="D")
        n_days = len(daily_grid)

        # 1. Resolve Macro Series (USD_ILS, IL_CPI) on continuous daily grid
        macro_df = self.macro_provider(t_start, t_end)
        macro_aligned = macro_df.reindex(daily_grid).ffill().fillna(0.0)
        fx_vec = macro_aligned["USD_ILS"].to_numpy(dtype=np.float64)
        cpi_vec = macro_aligned["IL_CPI"].to_numpy(dtype=np.float64)

        # 2. Vectorized Daily Holdings Matrix (N_days x M_tickers)
        sec_tx = tx_in_range[tx_in_range["tx_type"].isin(["BUY", "SELL"])].copy()
        sec_tickers = sorted(sec_tx["ticker"].unique().tolist()) if not sec_tx.empty else []

        if sec_tickers:
            try:
                px_df, curr_map = self.price_provider(sec_tickers, t_start, t_end)
            except RuntimeError:
                from price_resolver import resolve_prices
                px_df, curr_map = resolve_prices(sec_tickers, t_start, t_end, allow_partial=True)

            # Align pre-split BUY/SELL quantities and prices to the split-adjusted market price basis
            sec_tx = align_pre_split_dataframe(sec_tx, px_df)
            sec_tx["tx_date"] = pd.to_datetime(sec_tx["date"]).dt.normalize()

            # Build daily price grid; seed any leading NaNs on/after first BUY date from split-aligned execution price
            px_raw = px_df.reindex(daily_grid).reindex(columns=sec_tickers).astype(np.float64)
            px_raw = px_raw.where(px_raw > 0.0, np.nan)
            px_ffilled = px_raw.ffill()

            for t in sec_tickers:
                t_tx = sec_tx[sec_tx["ticker"] == t]
                if t_tx.empty:
                    continue
                t_buys = t_tx[t_tx["tx_type"] == "BUY"]
                first_buy_ts = t_buys["tx_date"].min() if not t_buys.empty else t_tx["tx_date"].min()
                if px_ffilled.loc[first_buy_ts:, t].isna().any():
                    col_s = px_raw[t].copy()
                    for tx_r in t_tx.itertuples(index=False):
                        tx_dt = tx_r.tx_date
                        tx_px = float(tx_r.price)
                        if tx_dt in col_s.index and tx_px > 0.0 and pd.isna(col_s.at[tx_dt]):
                            col_s.at[tx_dt] = tx_px
                    px_ffilled[t] = col_s.ffill()

            px_aligned = px_ffilled.fillna(0.0)[sec_tickers].astype(np.float64)

            sec_tx["signed_qty"] = np.where(sec_tx["tx_type"] == "BUY", sec_tx["quantity"], -sec_tx["quantity"])

            qty_deltas = (
                sec_tx.pivot_table(index="tx_date", columns="ticker", values="signed_qty", aggfunc="sum", fill_value=0.0)
                .reindex(index=daily_grid, columns=sec_tickers, fill_value=0.0)
            )
            holdings_matrix = qty_deltas.cumsum().clip(lower=0.0)

            # Native-currency market value per ticker: H_{t, i} * P_{t, i}
            native_val_matrix = holdings_matrix.to_numpy(dtype=np.float64) * px_aligned.to_numpy(dtype=np.float64)
            is_usd_mask = np.array([curr_map.get(t, "USD") == "USD" for t in sec_tickers], dtype=bool)

            usd_tickers_val_usd = native_val_matrix[:, is_usd_mask].sum(axis=1) if is_usd_mask.any() else np.zeros(n_days)
            ils_tickers_val_ils = native_val_matrix[:, ~is_usd_mask].sum(axis=1) if (~is_usd_mask).any() else np.zeros(n_days)

            holdings_val_usd = usd_tickers_val_usd + (ils_tickers_val_ils / fx_vec)
            holdings_val_ils = (usd_tickers_val_usd * fx_vec) + ils_tickers_val_ils
            # Constant-FX ILS holdings value (evaluates USD holdings at inception FX_0 to isolate pure asset return)
            fx0 = float(fx_vec[0])
            holdings_val_const_fx_ils = (usd_tickers_val_usd * fx0) + ils_tickers_val_ils
        else:
            holdings_matrix = pd.DataFrame(index=daily_grid)
            holdings_val_usd = np.zeros(n_days, dtype=np.float64)
            holdings_val_ils = np.zeros(n_days, dtype=np.float64)
            holdings_val_const_fx_ils = np.zeros(n_days, dtype=np.float64)
            fx0 = float(fx_vec[0])

        # 3. Chronological 3-Bucket Capital Engine Replay
        tx_in_range["tx_date"] = pd.to_datetime(tx_in_range["date"]).dt.normalize()
        tx_in_range["priority"] = tx_in_range["tx_type"].map(lambda t: TX_TYPE_PRIORITY.get(t, 9))
        tx_sorted = tx_in_range.sort_values(["tx_date", "priority", "tx_id"], kind="mergesort")

        tx_by_day = {}
        for r in tx_sorted.itertuples(index=False):
            tx_by_day.setdefault(r.tx_date, []).append(r)

        avail_cash = {"USD": 0.0, "ILS": 0.0}
        cum_invested = {"USD": 0.0, "ILS": 0.0}
        cum_withdrawn = {"USD": 0.0, "ILS": 0.0}

        cash_bal_usd = np.zeros(n_days, dtype=np.float64)
        cash_bal_ils = np.zeros(n_days, dtype=np.float64)
        cash_bal_const_fx_ils = np.zeros(n_days, dtype=np.float64)

        ext_inflow_usd = np.zeros(n_days, dtype=np.float64)
        ext_inflow_ils = np.zeros(n_days, dtype=np.float64)
        ext_inflow_const_fx_ils = np.zeros(n_days, dtype=np.float64)

        withdrawal_usd = np.zeros(n_days, dtype=np.float64)
        withdrawal_ils = np.zeros(n_days, dtype=np.float64)
        withdrawal_const_fx_ils = np.zeros(n_days, dtype=np.float64)

        cum_div_usd = np.zeros(n_days, dtype=np.float64)
        cum_div_ils = np.zeros(n_days, dtype=np.float64)
        cum_div_const_fx_ils = np.zeros(n_days, dtype=np.float64)

        fee_day_ils = np.zeros(n_days, dtype=np.float64)

        tot_div_usd = 0.0
        tot_div_ils = 0.0
        tot_div_const_fx_ils = 0.0

        for day_idx, d_ts in enumerate(daily_grid):
            fx_t = float(fx_vec[day_idx])
            day_txs = tx_by_day.get(d_ts, [])

            day_inflow_usd = 0.0
            day_inflow_ils = 0.0
            day_wdr_usd = 0.0
            day_wdr_ils = 0.0

            for r in day_txs:
                cur = r.currency
                q = float(r.quantity)
                p = float(r.price)
                fee = float(r.fee)
                tt = r.tx_type
                fee_ils = fee * (fx_t if cur == "USD" else 1.0)
                fee_day_ils[day_idx] += fee_ils

                if tt == "CASH_DEPOSIT":
                    net_dep = max(0.0, q - fee)
                    avail_cash[cur] += net_dep
                    cum_invested[cur] += net_dep
                    if cur == "USD":
                        day_inflow_usd += net_dep
                    else:
                        day_inflow_ils += net_dep

                elif tt == "CASH_WITHDRAWAL":
                    amt = q + fee
                    avail_cash[cur] = max(0.0, avail_cash[cur] - amt)
                    cum_withdrawn[cur] += amt
                    if cur == "USD":
                        day_wdr_usd += amt
                    else:
                        day_wdr_ils += amt

                elif tt == "SELL":
                    net_proceeds = max(0.0, q * p - fee)
                    avail_cash[cur] += net_proceeds

                elif tt == "DIVIDEND":
                    gross_native = q * p
                    net_div = max(0.0, gross_native * (1.0 - DIVIDEND_WITHHOLDING_RATE) - fee)
                    avail_cash[cur] += net_div
                    if cur == "USD":
                        tot_div_usd += net_div
                        tot_div_ils += net_div * fx_t
                        tot_div_const_fx_ils += net_div * fx0
                    else:
                        tot_div_usd += net_div / fx_t
                        tot_div_ils += net_div
                        tot_div_const_fx_ils += net_div

                elif tt == "BUY":
                    cost = q * p + fee
                    used = min(avail_cash[cur], cost)
                    avail_cash[cur] -= used
                    shortfall = cost - used
                    if shortfall > 1e-9:
                        cum_invested[cur] += shortfall
                        if cur == "USD":
                            day_inflow_usd += shortfall
                        else:
                            day_inflow_ils += shortfall

            # End of day t balances (available cash is strictly >= 0)
            cash_bal_usd[day_idx] = max(0.0, avail_cash["USD"] + (avail_cash["ILS"] / fx_t))
            cash_bal_ils[day_idx] = max(0.0, (avail_cash["USD"] * fx_t) + avail_cash["ILS"])
            cash_bal_const_fx_ils[day_idx] = max(0.0, (avail_cash["USD"] * fx0) + avail_cash["ILS"])

            ext_inflow_usd[day_idx] = day_inflow_usd + (day_inflow_ils / fx_t)
            ext_inflow_ils[day_idx] = (day_inflow_usd * fx_t) + day_inflow_ils
            ext_inflow_const_fx_ils[day_idx] = (day_inflow_usd * fx0) + day_inflow_ils

            withdrawal_usd[day_idx] = day_wdr_usd + (day_wdr_ils / fx_t)
            withdrawal_ils[day_idx] = (day_wdr_usd * fx_t) + day_wdr_ils
            withdrawal_const_fx_ils[day_idx] = (day_wdr_usd * fx0) + day_wdr_ils

            cum_div_usd[day_idx] = tot_div_usd
            cum_div_ils[day_idx] = tot_div_ils
            cum_div_const_fx_ils[day_idx] = tot_div_const_fx_ils

        nav_usd = holdings_val_usd + cash_bal_usd
        nav_ils = holdings_val_ils + cash_bal_ils
        nav_const_fx_ils = holdings_val_const_fx_ils + cash_bal_const_fx_ils

        ext_cf_usd = ext_inflow_usd - withdrawal_usd
        ext_cf_ils = ext_inflow_ils - withdrawal_ils
        ext_cf_const_fx_ils = ext_inflow_const_fx_ils - withdrawal_const_fx_ils

        invested_cap_usd = np.cumsum(ext_inflow_usd)
        invested_cap_ils = np.cumsum(ext_inflow_ils)
        withdrawn_cap_usd = np.cumsum(withdrawal_usd)
        withdrawn_cap_ils = np.cumsum(withdrawal_ils)
        wealth_created_usd = (nav_usd + withdrawn_cap_usd) - invested_cap_usd
        wealth_created_ils = (nav_ils + withdrawn_cap_ils) - invested_cap_ils

        # 4. Vectorized Time-Weighted Return (TWR) Calculation
        twr_daily_usd, twr_cum_usd = self._compute_vectorized_twr(nav_usd, ext_cf_usd, ext_inflow_usd)
        twr_daily_ils, twr_cum_ils = self._compute_vectorized_twr(nav_ils, ext_cf_ils, ext_inflow_ils)
        twr_daily_const_fx, twr_cum_const_fx = self._compute_vectorized_twr(nav_const_fx_ils, ext_cf_const_fx_ils, ext_inflow_const_fx_ils)

        # 5. Cumulative Fees & Realized Israeli Capital Gains Tax Series
        cum_fees_ils = np.cumsum(fee_day_ils)
        rl_df = self.ledger.realized_lots()
        if not rl_df.empty:
            rl_df = rl_df.copy()
            rl_df["sell_ts"] = pd.to_datetime(rl_df["sell_date"]).dt.normalize()
            daily_realized_tax = (
                rl_df.groupby("sell_ts")["tax_ils"].sum().reindex(daily_grid, fill_value=0.0).cumsum().to_numpy(dtype=np.float64)
            )
        else:
            daily_realized_tax = np.zeros(n_days, dtype=np.float64)

        out = pd.DataFrame(
            {
                "holdings_value_usd": holdings_val_usd,
                "holdings_value_ils": holdings_val_ils,
                "cash_balance_usd": cash_bal_usd,
                "cash_balance_ils": cash_bal_ils,
                "available_cash_usd": cash_bal_usd,
                "available_cash_ils": cash_bal_ils,
                "cumulative_dividends_usd": cum_div_usd,
                "cumulative_dividends_ils": cum_div_ils,
                "invested_capital_usd": invested_cap_usd,
                "invested_capital_ils": invested_cap_ils,
                "withdrawn_capital_usd": withdrawn_cap_usd,
                "withdrawn_capital_ils": withdrawn_cap_ils,
                "total_wealth_created_usd": wealth_created_usd,
                "total_wealth_created_ils": wealth_created_ils,
                "cost_basis_usd": invested_cap_usd,
                "cost_basis_ils": invested_cap_ils,
                "nav_usd": nav_usd,
                "nav_ils": nav_ils,
                "nav_const_fx_ils": nav_const_fx_ils,
                "external_inflow_usd": ext_inflow_usd,
                "external_inflow_ils": ext_inflow_ils,
                "withdrawal_usd": withdrawal_usd,
                "withdrawal_ils": withdrawal_ils,
                "external_cf_usd": ext_cf_usd,
                "external_cf_ils": ext_cf_ils,
                "external_cf_const_fx_ils": ext_cf_const_fx_ils,
                "twr_daily_usd": twr_daily_usd,
                "twr_daily_ils": twr_daily_ils,
                "twr_daily_const_fx": twr_daily_const_fx,
                "twr_cum_usd": twr_cum_usd,
                "twr_cum_ils": twr_cum_ils,
                "twr_cum_const_fx": twr_cum_const_fx,
                "cumulative_fees_ils": cum_fees_ils,
                "cumulative_realized_tax_ils": daily_realized_tax,
                "usd_ils": fx_vec,
                "il_cpi": cpi_vec,
            },
            index=daily_grid,
        )
        out.index.name = "Date"
        self._last_holdings_matrix = holdings_matrix
        cleaned_out = _clean_frame(out)
        self._cached_daily_series[cache_key] = cleaned_out.copy()
        return cleaned_out

    @staticmethod
    def _compute_vectorized_twr(nav: np.ndarray, ext_cf: np.ndarray, ext_inflow: np.ndarray = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Vectorized daily sub-period Time-Weighted Return (TWR):
          r_t = (V_t - V_{t-1} - CF_t) / (V_{t-1} + max(0.0, inflow_t))
          R_TWR,t = prod_{k=0..t}(1 + r_k) - 1
        """
        n = len(nav)
        if n == 0:
            return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

        v_prev = np.empty(n, dtype=np.float64)
        v_prev[0] = 0.0
        if n > 1:
            v_prev[1:] = nav[:-1]

        inflow = ext_inflow if ext_inflow is not None else np.maximum(0.0, ext_cf)
        denom = v_prev + inflow
        numer = nav - v_prev - ext_cf

        r_daily = np.where(denom > 1e-9, numer / denom, 0.0)
        r_daily = np.nan_to_num(r_daily, nan=0.0, posinf=0.0, neginf=0.0)
        # Clamp extreme pathological single-day values to keep geometric product finite
        r_daily = np.clip(r_daily, -0.9999, 100.0)

        cum_growth = np.cumprod(1.0 + r_daily)
        twr_cum = cum_growth - 1.0
        return r_daily, twr_cum

    # ── Slicing, Rebasing, Attribution & Multi-Benchmark Engine ──────────────

    def analyze_slice(
        self,
        window: str = "ALL",
        start_date=None,
        end_date=None,
        view_currency: str = "ILS",
        benchmarks: list[str] = None,
        cash_mode: str = "auto",
    ) -> dict:
        """
        Slice the portfolio time-series to `[t0, t1]` (via preset `window` or custom `[start_date, end_date]`),
        rebase portfolio TWR and up to 3 benchmark tickers to `0.0%` (and base `100.0`) at `t0` in `view_currency`,
        run the What-If Cash-Flow Matched Opportunity Simulator, and compute full institutional risk KPIs.
        """
        view_cur = str(view_currency or "ILS").upper().strip()
        if view_cur not in ("USD", "ILS"):
            raise ValueError(f"view_currency must be 'USD' or 'ILS' (got {view_currency}).")

        bench_tuple = tuple(str(b).upper().strip() for b in (benchmarks or []) if str(b).strip())[:3]
        tx_count = len(self.ledger._tx) if hasattr(self.ledger, "_tx") else -1
        slice_cache_key = (tx_count, str(window), str(start_date), str(end_date), view_cur, bench_tuple, str(cash_mode))
        if slice_cache_key in self._cached_slices:
            return self._cached_slices[slice_cache_key]

        daily_df = self.build_daily_series(end_date=end_date, cash_mode=cash_mode)
        if daily_df.empty:
            return {"series": pd.DataFrame(), "kpis": {}, "attribution": {}, "what_if": {}}

        inception_ts = daily_df.index[0]
        full_end_ts = daily_df.index[-1]
        t0, t1 = resolve_window_bounds(inception_ts, full_end_ts, window=window, custom_start=start_date, custom_end=end_date)

        slice_df = daily_df.loc[t0:t1].copy()
        slice_grid = slice_df.index
        cur_lower = view_cur.lower()

        nav_col = f"nav_{cur_lower}"
        cf_col = f"external_cf_{cur_lower}"
        twr_daily_col = f"twr_daily_{cur_lower}"
        twr_cum_col = f"twr_cum_{cur_lower}"

        # Rebase portfolio TWR to 0.0% (and index 100.0) at t0
        # At t0, the slice baseline is the cumulative TWR at the end of t0 (so t0 starts at 0.0%)
        base_growth = 1.0 + float(slice_df[twr_cum_col].iloc[0])
        if abs(base_growth) < 1e-9:
            base_growth = 1.0
        rebased_port_twr = (1.0 + slice_df[twr_cum_col].to_numpy(dtype=np.float64)) / base_growth - 1.0
        rebased_port_index_100 = 100.0 * (1.0 + rebased_port_twr)

        # Underwater drawdown series of the TWR growth curve inside the slice
        growth_curve = 1.0 + rebased_port_twr
        running_peak = np.maximum.accumulate(growth_curve)
        drawdown_series = np.where(running_peak > 1e-9, (growth_curve - running_peak) / running_peak, 0.0)

        out_series = pd.DataFrame(
            {
                "nav": slice_df[nav_col].to_numpy(dtype=np.float64),
                "port_nav": slice_df[nav_col].to_numpy(dtype=np.float64),
                "holdings_value": slice_df[f"holdings_value_{cur_lower}"].to_numpy(dtype=np.float64),
                "cash_balance": slice_df[f"cash_balance_{cur_lower}"].to_numpy(dtype=np.float64),
                "available_cash": slice_df[f"cash_balance_{cur_lower}"].to_numpy(dtype=np.float64),
                "invested_capital": slice_df[f"invested_capital_{cur_lower}"].to_numpy(dtype=np.float64),
                "withdrawn_capital": slice_df[f"withdrawn_capital_{cur_lower}"].to_numpy(dtype=np.float64),
                "total_wealth_created": slice_df[f"total_wealth_created_{cur_lower}"].to_numpy(dtype=np.float64),
                "cost_basis": slice_df[f"invested_capital_{cur_lower}"].to_numpy(dtype=np.float64),
                "external_cf": slice_df[cf_col].to_numpy(dtype=np.float64),
                "external_inflow": slice_df[f"external_inflow_{cur_lower}"].to_numpy(dtype=np.float64),
                "withdrawal": slice_df[f"withdrawal_{cur_lower}"].to_numpy(dtype=np.float64),
                "portfolio_daily_return": slice_df[twr_daily_col].to_numpy(dtype=np.float64),
                "portfolio_twr_pct": rebased_port_twr * 100.0,
                "port_twr_rebased_pct": rebased_port_twr * 100.0,
                "portfolio_index_100": rebased_port_index_100,
                "portfolio_drawdown_pct": drawdown_series * 100.0,
                "port_drawdown_pct": drawdown_series * 100.0,
                "usd_ils": slice_df["usd_ils"].to_numpy(dtype=np.float64),
                "il_cpi": slice_df["il_cpi"].to_numpy(dtype=np.float64),
            },
            index=slice_grid,
        )
        out_series.index.name = "Date"

        # ── Multi-Benchmark Rebasing & Cash-Flow Matched What-If Simulator ───
        bench_list = [str(b).upper().strip() for b in (benchmarks or []) if str(b).strip()][:3]
        what_if_results = {}
        primary_bench_daily_ret = None
        primary_bench_total_ret = 0.0
        primary_bench_ann_ret = 0.0

        if bench_list:
            b_px_df, b_curr_map = self.price_provider(bench_list, t0, t1)
            b_px_aligned = b_px_df.reindex(slice_grid).ffill().fillna(0.0)
            fx_slice = slice_df["usd_ils"].to_numpy(dtype=np.float64)

            for idx_b, b_sym in enumerate(bench_list):
                if b_sym not in b_px_aligned.columns:
                    continue
                b_native = b_px_aligned[b_sym].to_numpy(dtype=np.float64)
                b_cur = b_curr_map.get(b_sym, "USD")

                # Convert benchmark price into view_currency
                if b_cur == view_cur:
                    b_view_px = b_native
                elif b_cur == "USD" and view_cur == "ILS":
                    b_view_px = b_native * fx_slice
                else:
                    b_view_px = b_native / fx_slice

                p0 = float(b_view_px[0]) if len(b_view_px) > 0 and b_view_px[0] > 0 else 1.0
                b_rebased_ret = (b_view_px / p0) - 1.0
                b_daily_ret = np.zeros_like(b_view_px)
                if len(b_view_px) > 1:
                    b_daily_ret[1:] = np.where(b_view_px[:-1] > 1e-9, (b_view_px[1:] - b_view_px[:-1]) / b_view_px[:-1], 0.0)

                out_series[f"bench_{b_sym}_twr_pct"] = b_rebased_ret * 100.0
                out_series[f"bench_{b_sym}_rebased_pct"] = b_rebased_ret * 100.0
                out_series[f"bench_{b_sym}_index_100"] = 100.0 * (1.0 + b_rebased_ret)

                if idx_b == 0:
                    primary_bench_daily_ret = b_daily_ret
                    primary_bench_total_ret = float(b_rebased_ret[-1]) if len(b_rebased_ret) else 0.0
                    n_years_slice = max((t1 - t0).days / CALENDAR_DAYS_PER_YEAR, 1.0 / CALENDAR_DAYS_PER_YEAR)
                    primary_bench_ann_ret = float((1.0 + max(-0.9999, primary_bench_total_ret)) ** (1.0 / n_years_slice) - 1.0)

                # What-If Opportunity Simulator:
                # Benchmark units purchased ONLY on new external capital entering total_invested_capital (external_inflow > 0).
                # Benchmark units sold ONLY on explicit CASH_WITHDRAWAL (withdrawal > 0).
                # Internal stock SELLs leave benchmark units completely untouched!
                inflow_vec = slice_df[f"external_inflow_{cur_lower}"].to_numpy(dtype=np.float64).copy()
                wdr_vec = slice_df[f"withdrawal_{cur_lower}"].to_numpy(dtype=np.float64).copy()

                if t0 > inception_ts:
                    v_before_t0 = float(daily_df.loc[: t0 - pd.Timedelta(days=1), nav_col].iloc[-1])
                    inflow_vec[0] += v_before_t0

                n_pts = len(b_view_px)
                cum_shares = np.zeros(n_pts, dtype=np.float64)
                curr_shares = 0.0

                for i in range(n_pts):
                    p_i = b_view_px[i]
                    if p_i > 1e-9:
                        if inflow_vec[i] > 1e-9:
                            curr_shares += inflow_vec[i] / p_i
                        if wdr_vec[i] > 1e-9:
                            delta_sell = min(curr_shares, wdr_vec[i] / p_i)
                            curr_shares = max(0.0, curr_shares - delta_sell)
                    cum_shares[i] = curr_shares

                what_if_nav = cum_shares * b_view_px
                out_series[f"what_if_{b_sym}_nav"] = what_if_nav

                term_port_nav = float(slice_df[nav_col].iloc[-1])
                term_bench_nav = float(what_if_nav[-1]) if len(what_if_nav) else 0.0
                what_if_results[b_sym] = {
                    "benchmark": b_sym,
                    "currency": view_cur,
                    "terminal_benchmark_nav": _scalar_float(term_bench_nav),
                    "terminal_portfolio_nav": _scalar_float(term_port_nav),
                    "excess_value_created": _scalar_float(term_port_nav - term_bench_nav),
                    "benchmark_twr_pct": _scalar_float(b_rebased_ret[-1] * 100.0) if len(b_rebased_ret) else 0.0,
                }

        # ── MWR (XIRR & Modified Dietz) on the Slice ─────────────────────────
        dated_cfs: list[tuple[pd.Timestamp, float]] = []
        if t0 > inception_ts:
            v_start = float(daily_df.loc[: t0 - pd.Timedelta(days=1), nav_col].iloc[-1])
            if v_start > 0.0:
                dated_cfs.append((t0, -v_start))
        else:
            v_start = 0.0

        inflow_s = slice_df[f"external_inflow_{cur_lower}"]
        wdr_s = slice_df[f"withdrawal_{cur_lower}"]

        for dt in slice_grid:
            inf = float(inflow_s.loc[dt])
            wdr = float(wdr_s.loc[dt])
            net_ext = inf - wdr
            if abs(net_ext) > 1e-9:
                dated_cfs.append((dt, -net_ext))

        v_end = float(slice_df[nav_col].iloc[-1])
        dated_cfs.append((t1, v_end))

        xirr_annual = compute_xirr(dated_cfs)
        cf_series = slice_df[cf_col]
        mwr_dietz = compute_modified_dietz_return(v_start, v_end, cf_series, t0, t1)

        # ── Institutional Risk & Return KPIs ─────────────────────────────────
        kpis = self._compute_slice_kpis(
            slice_df=slice_df,
            rebased_port_twr=rebased_port_twr,
            drawdown_series=drawdown_series,
            twr_daily_col=twr_daily_col,
            xirr_annual=xirr_annual,
            mwr_dietz=mwr_dietz,
            primary_bench_daily_ret=primary_bench_daily_ret,
            primary_bench_ann_ret=primary_bench_ann_ret,
            primary_bench_total_ret=primary_bench_total_ret,
            t0=t0,
            t1=t1,
            view_currency=view_cur,
        )

        # ── FX & Tax Attribution Breakdown (ILS View) ────────────────────────
        attribution = self._compute_ils_attribution(daily_df, slice_df, t0, t1)

        cleaned_series = _clean_frame(out_series)
        res_dict = {
            "window": window,
            "start_date": t0.strftime("%Y-%m-%d"),
            "end_date": t1.strftime("%Y-%m-%d"),
            "view_currency": view_cur,
            "series": cleaned_series,
            "series_clean": sanitize_for_json(cleaned_series),
            "kpis": kpis,
            "attribution": attribution,
            "what_if": what_if_results,
        }
        self._cached_slices[slice_cache_key] = res_dict
        return res_dict

    # Alias for API/consumer consistency
    slice_and_rebase = analyze_slice

    def _compute_slice_kpis(
        self,
        slice_df: pd.DataFrame,
        rebased_port_twr: np.ndarray,
        drawdown_series: np.ndarray,
        twr_daily_col: str,
        xirr_annual: float,
        mwr_dietz: float,
        primary_bench_daily_ret: np.ndarray | None,
        primary_bench_ann_ret: float,
        primary_bench_total_ret: float,
        t0: pd.Timestamp,
        t1: pd.Timestamp,
        view_currency: str,
    ) -> dict:
        total_twr = float(rebased_port_twr[-1]) if len(rebased_port_twr) else 0.0
        n_cal_days = max(1, (t1 - t0).days)
        years = max(n_cal_days / CALENDAR_DAYS_PER_YEAR, 1.0 / CALENDAR_DAYS_PER_YEAR)
        ann_twr = float((1.0 + max(-0.9999, total_twr)) ** (1.0 / years) - 1.0) if n_cal_days >= 1 else total_twr

        # Exclude day-0 (rebase anchor) when measuring daily return volatility
        r_daily = slice_df[twr_daily_col].to_numpy(dtype=np.float64)
        r_active = r_daily[1:] if len(r_daily) > 1 else r_daily
        # Restrict volatility calculation to active market days or full calendar grid scaled appropriately
        ann_factor = math.sqrt(CALENDAR_DAYS_PER_YEAR)
        vol_ann = float(np.std(r_active, ddof=1) * ann_factor) if len(r_active) > 1 else 0.0

        sharpe = (ann_twr - RISK_FREE_ANNUAL) / vol_ann if vol_ann > 1e-9 else 0.0

        daily_rf = RISK_FREE_ANNUAL / CALENDAR_DAYS_PER_YEAR
        downside_diff = np.minimum(0.0, r_active - daily_rf)
        downside_dev = float(math.sqrt(np.mean(downside_diff ** 2)) * ann_factor) if len(r_active) > 0 else 0.0
        sortino = (ann_twr - RISK_FREE_ANNUAL) / downside_dev if downside_dev > 1e-9 else 0.0

        max_dd = float(np.min(drawdown_series)) if len(drawdown_series) else 0.0

        # Beta & Alpha vs. primary benchmark
        beta = 0.0
        alpha = 0.0
        if primary_bench_daily_ret is not None and len(primary_bench_daily_ret) == len(r_daily) and len(r_active) > 1:
            b_active = primary_bench_daily_ret[1:]
            b_var = float(np.var(b_active, ddof=1))
            if b_var > 1e-12:
                cov = float(np.cov(r_active, b_active, ddof=1)[0, 1])
                beta = cov / b_var
            alpha = ann_twr - (RISK_FREE_ANNUAL + beta * (primary_bench_ann_ret - RISK_FREE_ANNUAL))

        # Realized Trade Win-Rate & Payoff Ratio inside [t0, t1]
        rl_df = self.ledger.realized_lots()
        win_rate = 0.0
        payoff_ratio = 0.0
        n_closed_lots = 0
        if not rl_df.empty:
            in_win = rl_df[(rl_df["sell_date"] >= t0.strftime("%Y-%m-%d")) & (rl_df["sell_date"] <= t1.strftime("%Y-%m-%d"))]
            n_closed_lots = len(in_win)
            if n_closed_lots > 0:
                net_pnl = in_win["nominal_gain_ils"].to_numpy(dtype=np.float64)
                wins = net_pnl[net_pnl > 0.0]
                losses = -net_pnl[net_pnl < 0.0]
                win_rate = float(len(wins) / n_closed_lots)
                avg_win = float(np.mean(wins)) if len(wins) > 0 else 0.0
                avg_loss = float(np.mean(losses)) if len(losses) > 0 else 0.0
                payoff_ratio = (avg_win / avg_loss) if avg_loss > 1e-9 else (float(avg_win > 0.0))

        return {
            "view_currency": view_currency,
            "twr_total_pct": _scalar_float(total_twr * 100.0),
            "twr_annualized_pct": _scalar_float(ann_twr * 100.0),
            "xirr_annualized_pct": _scalar_float(xirr_annual * 100.0),
            "mwr_dietz_total_pct": _scalar_float(mwr_dietz * 100.0),
            "annualized_volatility_pct": _scalar_float(vol_ann * 100.0),
            "sharpe_ratio": _scalar_float(sharpe),
            "sortino_ratio": _scalar_float(sortino),
            "max_drawdown_pct": _scalar_float(max_dd * 100.0),
            "beta_vs_benchmark": _scalar_float(beta),
            "alpha_annualized_pct": _scalar_float(alpha * 100.0),
            "benchmark_total_return_pct": _scalar_float(primary_bench_total_ret * 100.0),
            "closed_lots_count": int(n_closed_lots),
            "win_rate_pct": _scalar_float(win_rate * 100.0),
            "payoff_ratio": _scalar_float(payoff_ratio),
        }

    def _compute_ils_attribution(
        self,
        daily_df: pd.DataFrame,
        slice_df: pd.DataFrame,
        t0: pd.Timestamp,
        t1: pd.Timestamp,
    ) -> dict:
        """
        Decompose total ILS return over [t0, t1] into:
          1. Pure Asset Return (constant-FX TWR in native prices).
          2. FX Exchange Rate Impact (difference between actual ILS gross TWR and constant-FX TWR).
          3. Tax & Fee Drag (brokerage fees + realized capital gains tax + estimated unrealized liquidation tax).
        """
        base_ils = 1.0 + float(slice_df["twr_cum_ils"].iloc[0])
        base_const = 1.0 + float(slice_df["twr_cum_const_fx"].iloc[0])
        if abs(base_ils) < 1e-9:
            base_ils = 1.0
        if abs(base_const) < 1e-9:
            base_const = 1.0

        gross_ils_twr = (1.0 + float(slice_df["twr_cum_ils"].iloc[-1])) / base_ils - 1.0
        pure_asset_twr = (1.0 + float(slice_df["twr_cum_const_fx"].iloc[-1])) / base_const - 1.0
        fx_impact_twr = gross_ils_twr - pure_asset_twr

        # Fees and taxes incurred inside [t0, t1]
        fees_before = float(daily_df.loc[: t0 - pd.Timedelta(days=1), "cumulative_fees_ils"].iloc[-1]) if t0 > daily_df.index[0] else 0.0
        tax_before = float(daily_df.loc[: t0 - pd.Timedelta(days=1), "cumulative_realized_tax_ils"].iloc[-1]) if t0 > daily_df.index[0] else 0.0
        fees_in_slice = float(slice_df["cumulative_fees_ils"].iloc[-1]) - fees_before
        realized_tax_in_slice = float(slice_df["cumulative_realized_tax_ils"].iloc[-1]) - tax_before

        # Estimated unrealized liquidation tax at t1
        snap_t1 = self.ledger.snapshot(as_of=t1, price_provider=self.price_provider)
        est_unrealized_tax_ils = _scalar_float(snap_t1["totals"].get("est_liquidation_tax_ils", 0.0))

        denom_cap = max(
            float(slice_df["cost_basis_ils"].max()),
            float(slice_df["nav_ils"].max()),
            1.0,
        )
        total_drag_ils = fees_in_slice + realized_tax_in_slice + est_unrealized_tax_ils
        tax_and_fee_drag_pct = -(total_drag_ils / denom_cap) * 100.0
        net_after_tax_twr_pct = (gross_ils_twr * 100.0) + tax_and_fee_drag_pct

        return {
            "gross_ils_twr_pct": _scalar_float(gross_ils_twr * 100.0),
            "pure_asset_return_pct": _scalar_float(pure_asset_twr * 100.0),
            "fx_impact_pct": _scalar_float(fx_impact_twr * 100.0),
            "fees_ils": _scalar_float(fees_in_slice),
            "realized_tax_ils": _scalar_float(realized_tax_in_slice),
            "est_unrealized_tax_ils": _scalar_float(est_unrealized_tax_ils),
            "tax_and_fee_drag_pct": _scalar_float(tax_and_fee_drag_pct),
            "net_after_tax_twr_pct": _scalar_float(net_after_tax_twr_pct),
        }

    def compute_quant_dna(
        self,
        as_of=None,
        weights_16: list[float] | None = None,
        view_currency: str = "ILS",
    ) -> dict:
        """Memoized wrapper around `compute_portfolio_quant_dna` tied to ledger state."""
        view_cur = str(view_currency or "ILS").upper().strip()
        w_tuple = (
            tuple(round(float(w), 6) for w in weights_16)
            if weights_16 is not None and len(weights_16) == NUM_INDICATORS
            else tuple([round(1.0 / NUM_INDICATORS, 6)] * NUM_INDICATORS)
        )
        tx_count = len(self.ledger._tx) if hasattr(self.ledger, "_tx") else -1
        cache_key = (tx_count, str(as_of), view_cur, w_tuple)
        if cache_key in self._cached_quant_dna:
            return self._cached_quant_dna[cache_key]

        res = compute_portfolio_quant_dna(
            ledger=self.ledger,
            as_of=as_of,
            weights_16=list(w_tuple),
            view_currency=view_cur,
            price_provider=self.price_provider,
        )
        self._cached_quant_dna[cache_key] = res
        return res


# ─── Phase 6: Portfolio Factor DNA (C-FFI) & GICS Sector Attribution ──────────

def _get_universe_quant_baseline(weights_16: list[float]) -> dict:
    """
    Hydrate and cache the normalized 16-indicator fundamental universe and execute
    the 80-byte C-FFI `Stock` scoring kernel (`run_financial_engine`) across all
    universe equities. Returns O(1) lookup tables and universe baseline factor profiles.
    """
    global _UNIVERSE_DNA_CACHE
    now = time.time()

    # Normalize weights vector to sum to 1.0 for standardized [0, 100] C-FFI composite scores
    w_arr = np.asarray(weights_16, dtype=np.float64)
    w_sum = float(np.sum(w_arr))
    if w_sum > 1e-9:
        w_norm = (w_arr / w_sum).tolist()
    else:
        w_norm = [1.0 / NUM_INDICATORS] * NUM_INDICATORS
    w_key = tuple(round(float(x), 6) for x in w_norm)

    if (
        _UNIVERSE_DNA_CACHE["norm_by_ticker"]
        and (now - _UNIVERSE_DNA_CACHE["timestamp"]) < _DNA_CACHE_TTL
    ):
        if w_key not in _UNIVERSE_DNA_CACHE["scores_by_weights"]:
            norm_list = list(_UNIVERSE_DNA_CACHE["norm_by_ticker"].values())
            scored = run_financial_engine(norm_list, w_norm)
            _UNIVERSE_DNA_CACHE["scores_by_weights"][w_key] = {
                s["Ticker"]: _scalar_float(s["FinalScore"]) for s in scored
            }
        return {
            "norm_by_ticker": _UNIVERSE_DNA_CACHE["norm_by_ticker"],
            "raw_by_ticker": _UNIVERSE_DNA_CACHE["raw_by_ticker"],
            "meta_by_ticker": _UNIVERSE_DNA_CACHE["meta_by_ticker"],
            "baseline_indicators": _UNIVERSE_DNA_CACHE["baseline_indicators"],
            "baseline_pillars": _UNIVERSE_DNA_CACHE["baseline_pillars"],
            "scores_map": _UNIVERSE_DNA_CACHE["scores_by_weights"][w_key],
            "weights_normalized": w_norm,
        }

    base_dir = os.path.dirname(os.path.abspath(__file__))
    primary_csv = os.path.abspath(os.path.join(base_dir, "..", "data", "historical_fundamental_data.csv"))
    fallback_csv = os.path.abspath(os.path.join(base_dir, "..", "data", "market_data.csv"))
    csv_path = primary_csv if os.path.exists(primary_csv) else fallback_csv

    stocks_df, err = get_cached_fundamental_snapshot(csv_path)
    if err or stocks_df is None or stocks_df.empty:
        return {
            "norm_by_ticker": {},
            "raw_by_ticker": {},
            "meta_by_ticker": {},
            "baseline_indicators": {c: 50.0 for c in INDICATOR_COLUMNS},
            "baseline_pillars": {p: 50.0 for p in FACTOR_PILLARS},
            "scores_map": {},
            "weights_normalized": w_norm,
        }

    # Enrich GICS Sector & LongName metadata and backfill missing S&P 500 equities from market_data.csv
    if os.path.exists(fallback_csv) and csv_path != fallback_csv:
        try:
            mkt_df = pd.read_csv(fallback_csv)
            if not mkt_df.empty and "Ticker" in mkt_df.columns:
                mkt_df["Ticker"] = mkt_df["Ticker"].astype(str).str.upper().str.strip()
                if "ForwardPE" in mkt_df.columns and "TrailingPE" not in mkt_df.columns:
                    mkt_df["TrailingPE"] = mkt_df["ForwardPE"]
                if "DebtToEquity" in mkt_df.columns and "NetDebt_EBITDA" not in mkt_df.columns:
                    mkt_df["NetDebt_EBITDA"] = pd.to_numeric(mkt_df["DebtToEquity"], errors="coerce").fillna(0.0) / 25.0

                sec_map = dict(zip(mkt_df["Ticker"], mkt_df.get("Sector", pd.Series(dtype=str)).fillna("")))
                name_map = dict(zip(mkt_df["Ticker"], mkt_df.get("LongName", mkt_df["Ticker"]).fillna("")))

                stocks_df["Ticker"] = stocks_df["Ticker"].astype(str).str.upper().str.strip()
                mapped_sec = stocks_df["Ticker"].map(sec_map)
                stocks_df["Sector"] = np.where(
                    mapped_sec.notna() & (mapped_sec != ""),
                    mapped_sec,
                    stocks_df.get("Sector", "Other"),
                )
                mapped_name = stocks_df["Ticker"].map(name_map)
                stocks_df["LongName"] = np.where(
                    mapped_name.notna() & (mapped_name != ""),
                    mapped_name,
                    stocks_df.get("LongName", stocks_df["Ticker"]),
                )

                existing_set = set(stocks_df["Ticker"])
                missing_mkt = mkt_df[~mkt_df["Ticker"].isin(existing_set)].copy()
                if not missing_mkt.empty:
                    stocks_df = pd.concat([stocks_df, missing_mkt], ignore_index=True)
        except Exception:
            pass

    ensure_tech_cache()
    surviving_tickers = stocks_df["Ticker"].astype(str).tolist()
    tech_map = compute_live_technical_indicators(surviving_tickers)
    if tech_map:
        tech_df = pd.DataFrame.from_dict(tech_map, orient="index")
        tech_df.index.name = "Ticker"
        stocks_df = stocks_df.set_index("Ticker")
        for col in ("RelativeStrength", "PriceVs200SMA", "RSI", "Beta"):
            if col in tech_df.columns:
                stocks_df[col] = tech_df[col]
        stocks_df = stocks_df.reset_index()

    stocks_df = _clean_frame(stocks_df)
    stocks_data = stocks_df.to_dict(orient="records")

    normalized_stocks = normalize_stocks(
        stocks_data,
        strategy=None,
        allowed_sectors=None,
        min_market_cap=None,
        max_market_cap=None,
        global_stocks_list=stocks_data,
    )

    norm_by_ticker = {str(s["Ticker"]).upper(): s for s in normalized_stocks}
    raw_by_ticker = {str(s["Ticker"]).upper(): s for s in stocks_data}
    meta_by_ticker = {
        str(s["Ticker"]).upper(): {
            "longName": str(s.get("LongName", "") or ""),
            "sector": str(s.get("Sector", "") or "Unclassified").strip() or "Unclassified",
            "marketCap": _scalar_float(s.get("MarketCap", 0.0)),
        }
        for s in stocks_data
    }

    # Vectorized baseline calculation across the normalized universe matrix (N x 16)
    norm_df = pd.DataFrame(normalized_stocks)
    if not norm_df.empty:
        mat_16 = norm_df[INDICATOR_COLUMNS].to_numpy(dtype=np.float64)
        col_means = np.nanmean(mat_16, axis=0)
        baseline_indicators = {
            col: _scalar_float(round(float(col_means[idx]), 2), 50.0)
            for idx, col in enumerate(INDICATOR_COLUMNS)
        }
    else:
        baseline_indicators = {c: 50.0 for c in INDICATOR_COLUMNS}

    baseline_pillars = {}
    for pillar_name, cols in FACTOR_PILLARS.items():
        vals = np.array([baseline_indicators.get(c, 50.0) for c in cols], dtype=np.float64)
        baseline_pillars[pillar_name] = _scalar_float(round(float(np.mean(vals)), 2), 50.0)

    scored = run_financial_engine(normalized_stocks, w_norm)
    scores_map = {str(s["Ticker"]).upper(): _scalar_float(s["FinalScore"]) for s in scored}

    _UNIVERSE_DNA_CACHE["timestamp"] = now
    _UNIVERSE_DNA_CACHE["norm_by_ticker"] = norm_by_ticker
    _UNIVERSE_DNA_CACHE["raw_by_ticker"] = raw_by_ticker
    _UNIVERSE_DNA_CACHE["meta_by_ticker"] = meta_by_ticker
    _UNIVERSE_DNA_CACHE["baseline_indicators"] = baseline_indicators
    _UNIVERSE_DNA_CACHE["baseline_pillars"] = baseline_pillars
    _UNIVERSE_DNA_CACHE["scores_by_weights"] = {w_key: scores_map}

    return {
        "norm_by_ticker": norm_by_ticker,
        "raw_by_ticker": raw_by_ticker,
        "meta_by_ticker": meta_by_ticker,
        "baseline_indicators": baseline_indicators,
        "baseline_pillars": baseline_pillars,
        "scores_map": scores_map,
        "weights_normalized": w_norm,
    }


def compute_portfolio_quant_dna(
    ledger: PortfolioLedger,
    as_of=None,
    weights_16: list[float] | None = None,
    view_currency: str = "ILS",
    price_provider=None,
) -> dict:
    """
    Compute Phase 6 Portfolio Factor DNA, Per-Holding C-FFI Quant Scores,
    5-Pillar Radar Profile, and GICS Sector Attribution.

    Strictly normalizes factor weights across covered fundamental equities so ETFs
    (e.g., SPY, QQQ) or TASE (.TA) securities do not bias factor scores toward zero,
    and reports `factor_coverage_pct` (percentage of invested equity covered by the
    16-indicator SimFin fundamental dataset).
    """
    view_cur = str(view_currency or "ILS").upper().strip()
    if view_cur not in ("USD", "ILS"):
        raise ValueError(f"view_currency must be 'USD' or 'ILS' (got {view_currency}).")

    if weights_16 is None or len(weights_16) != NUM_INDICATORS:
        weights_16 = [1.0 / NUM_INDICATORS] * NUM_INDICATORS

    uni = _get_universe_quant_baseline(weights_16)
    norm_by_ticker = uni["norm_by_ticker"]
    meta_by_ticker = uni["meta_by_ticker"]
    baseline_indicators = uni["baseline_indicators"]
    baseline_pillars = uni["baseline_pillars"]
    scores_map = uni["scores_map"]
    w_norm = np.asarray(uni["weights_normalized"], dtype=np.float64)

    tx_df = ledger.transactions()
    if tx_df.empty:
        return sanitize_for_json({
            "as_of": as_of or pd.Timestamp.now().strftime("%Y-%m-%d"),
            "view_currency": view_cur,
            "portfolio_quant_score": 0.0,
            "benchmark_quant_score": _scalar_float(float(np.dot(
                np.array([baseline_indicators[c] for c in INDICATOR_COLUMNS], dtype=np.float64),
                w_norm
            ))),
            "factor_coverage_pct": 0.0,
            "covered_holdings_count": 0,
            "total_holdings_count": 0,
            "pillars": {p: 0.0 for p in FACTOR_PILLARS},
            "benchmark_pillars": baseline_pillars,
            "pillar_tilts": {p: 0.0 for p in FACTOR_PILLARS},
            "indicators": {c: 0.0 for c in INDICATOR_COLUMNS},
            "benchmark_indicators": baseline_indicators,
            "holdings_dna": [],
            "sector_attribution": [],
        })

    snap = ledger.snapshot(as_of=as_of, price_provider=price_provider)
    pos_df = snap["positions"]
    snap_as_of = snap["as_of"]

    bench_score = _scalar_float(round(float(np.dot(
        np.array([baseline_indicators[c] for c in INDICATOR_COLUMNS], dtype=np.float64),
        w_norm,
    )), 2))

    if pos_df.empty:
        return sanitize_for_json({
            "as_of": snap_as_of,
            "view_currency": view_cur,
            "portfolio_quant_score": 0.0,
            "benchmark_quant_score": bench_score,
            "factor_coverage_pct": 0.0,
            "covered_holdings_count": 0,
            "total_holdings_count": 0,
            "pillars": {p: 0.0 for p in FACTOR_PILLARS},
            "benchmark_pillars": baseline_pillars,
            "pillar_tilts": {p: 0.0 for p in FACTOR_PILLARS},
            "indicators": {c: 0.0 for c in INDICATOR_COLUMNS},
            "benchmark_indicators": baseline_indicators,
            "holdings_dna": [],
            "sector_attribution": [],
        })

    mv_col = "market_value_ils" if view_cur == "ILS" else "market_value_usd"
    cost_col = "cost_ils" if view_cur == "ILS" else "cost_usd"

    total_mv = _scalar_float(pos_df[mv_col].sum())
    total_cost = _scalar_float(pos_df[cost_col].sum())

    holdings_dna = []
    covered_mvs = []
    covered_scores = []
    covered_ind_rows = []
    sector_buckets: dict[str, dict] = {}

    for _, row in pos_df.iterrows():
        ticker = str(row["ticker"]).upper().strip()
        mv_view = _scalar_float(row[mv_col])
        cost_view = _scalar_float(row[cost_col])
        unr_pnl_view = mv_view - cost_view
        weight_pct = (mv_view / total_mv * 100.0) if total_mv > 1e-9 else 0.0

        is_etf_or_tase = (ticker in KNOWN_ETF_TICKERS) or ticker.endswith(".TA")
        is_covered = (ticker in norm_by_ticker) and (not is_etf_or_tase)
        if is_covered:
            norm_entry = norm_by_ticker[ticker]
            meta_entry = meta_by_ticker.get(ticker, {})
            sector = str(meta_entry.get("sector") or "Unclassified").strip() or "Unclassified"
            long_name = str(meta_entry.get("longName") or ticker)
            q_score = _scalar_float(round(scores_map.get(ticker, 0.0), 2))
            if q_score >= 70.0:
                q_status = "STRONG"
            elif q_score >= 45.0:
                q_status = "NEUTRAL"
            else:
                q_status = "WARNING"

            ind_vec = np.array([_scalar_float(norm_entry.get(c, 0.0)) for c in INDICATOR_COLUMNS], dtype=np.float64)
            ind_dict = {c: _scalar_float(round(float(ind_vec[idx]), 1)) for idx, c in enumerate(INDICATOR_COLUMNS)}
            h_pillars = {}
            for p_name, p_cols in FACTOR_PILLARS.items():
                p_vals = np.array([ind_dict[c] for c in p_cols], dtype=np.float64)
                h_pillars[p_name] = _scalar_float(round(float(np.mean(p_vals)), 1))

            covered_mvs.append(mv_view)
            covered_scores.append(q_score)
            covered_ind_rows.append(ind_vec)
        else:
            sector = "Unclassified / ETF / TASE"
            long_name = ticker
            q_score = None
            q_status = "ETF_TASE"
            ind_dict = {}
            h_pillars = {}

        holdings_dna.append({
            "ticker": ticker,
            "long_name": long_name,
            "sector": sector,
            "is_covered": bool(is_covered),
            "quant_score": q_score,
            "quant_status": q_status,
            "weight_pct": _scalar_float(round(weight_pct, 2)),
            "market_value_view": _scalar_float(round(mv_view, 2)),
            "unrealized_pnl_view": _scalar_float(round(unr_pnl_view, 2)),
            "pillars": h_pillars,
            "indicators": ind_dict,
        })

        # Accumulate GICS Sector bucket
        if sector not in sector_buckets:
            sector_buckets[sector] = {
                "sector": sector,
                "market_value_view": 0.0,
                "cost_basis_view": 0.0,
                "unrealized_pnl_view": 0.0,
                "covered_mv": 0.0,
                "weighted_score_sum": 0.0,
                "holdings_count": 0,
                "tickers": [],
            }
        b = sector_buckets[sector]
        b["market_value_view"] += mv_view
        b["cost_basis_view"] += cost_view
        b["unrealized_pnl_view"] += unr_pnl_view
        b["holdings_count"] += 1
        b["tickers"].append(ticker)
        if is_covered and q_score is not None:
            b["covered_mv"] += mv_view
            b["weighted_score_sum"] += mv_view * q_score

    # Vectorized Portfolio Factor DNA across covered equities
    covered_mv_total = float(np.sum(covered_mvs)) if covered_mvs else 0.0
    factor_coverage_pct = (covered_mv_total / total_mv * 100.0) if total_mv > 1e-9 else 0.0

    if covered_mvs and covered_mv_total > 1e-9:
        w_cov = np.asarray(covered_mvs, dtype=np.float64) / covered_mv_total
        scores_vec = np.asarray(covered_scores, dtype=np.float64)
        ind_mat = np.vstack(covered_ind_rows)  # shape: (K_cov, 16)

        port_quant_score = _scalar_float(round(float(np.dot(w_cov, scores_vec)), 2))
        port_ind_vec = np.dot(w_cov, ind_mat)  # shape: (16,)
        port_indicators = {
            c: _scalar_float(round(float(port_ind_vec[idx]), 2))
            for idx, c in enumerate(INDICATOR_COLUMNS)
        }
        port_pillars = {}
        pillar_tilts = {}
        for p_name, p_cols in FACTOR_PILLARS.items():
            p_vals = np.array([port_indicators[c] for c in p_cols], dtype=np.float64)
            p_score = _scalar_float(round(float(np.mean(p_vals)), 2))
            port_pillars[p_name] = p_score
            pillar_tilts[p_name] = _scalar_float(round(p_score - baseline_pillars.get(p_name, 50.0), 2))
    else:
        port_quant_score = 0.0
        port_indicators = {c: 0.0 for c in INDICATOR_COLUMNS}
        port_pillars = {p: 0.0 for p in FACTOR_PILLARS}
        pillar_tilts = {p: 0.0 for p in FACTOR_PILLARS}

    # Finalize GICS Sector Attribution array (sorted by weight descending, exact 100% sum)
    sector_list = []
    for sec_name, b in sector_buckets.items():
        s_mv = b["market_value_view"]
        s_cost = b["cost_basis_view"]
        s_pnl = b["unrealized_pnl_view"]
        s_weight = (s_mv / total_mv * 100.0) if total_mv > 1e-9 else 0.0
        s_ret_pct = (s_pnl / s_cost * 100.0) if s_cost > 1e-9 else 0.0
        s_contrib_pct = (s_pnl / total_cost * 100.0) if total_cost > 1e-9 else 0.0
        s_avg_score = (b["weighted_score_sum"] / b["covered_mv"]) if b["covered_mv"] > 1e-9 else None

        sector_list.append({
            "sector": sec_name,
            "weight_pct": _scalar_float(round(s_weight, 4)),
            "market_value_view": _scalar_float(round(s_mv, 2)),
            "cost_basis_view": _scalar_float(round(s_cost, 2)),
            "unrealized_pnl_view": _scalar_float(round(s_pnl, 2)),
            "sector_return_pct": _scalar_float(round(s_ret_pct, 2)),
            "pnl_contribution_pct": _scalar_float(round(s_contrib_pct, 2)),
            "avg_quant_score": _scalar_float(round(s_avg_score, 2)) if s_avg_score is not None else None,
            "holdings_count": int(b["holdings_count"]),
            "tickers": b["tickers"],
        })

    sector_list.sort(key=lambda x: x["weight_pct"], reverse=True)
    if sector_list and total_mv > 1e-9:
        # Ensure exact 100.0% sum after rounding on the largest sector
        sum_other = sum(s["weight_pct"] for s in sector_list[1:])
        sector_list[0]["weight_pct"] = _scalar_float(round(max(0.0, 100.0 - sum_other), 4))

    return sanitize_for_json({
        "as_of": snap_as_of,
        "view_currency": view_cur,
        "portfolio_quant_score": port_quant_score,
        "benchmark_quant_score": bench_score,
        "factor_coverage_pct": _scalar_float(round(factor_coverage_pct, 2)),
        "covered_holdings_count": len(covered_mvs),
        "total_holdings_count": len(holdings_dna),
        "pillars": port_pillars,
        "benchmark_pillars": baseline_pillars,
        "pillar_tilts": pillar_tilts,
        "indicators": port_indicators,
        "benchmark_indicators": baseline_indicators,
        "holdings_dna": holdings_dna,
        "sector_attribution": sector_list,
    })
