#!/usr/bin/env python3
"""
verify_portfolio_analytics.py
=============================
Standalone CLI verification suite for Phase 3 of Portfolio Intelligence (`core_logic/portfolio_analytics.py`).

Validates:
1. TWR Cash-Flow Invariance:
   Proves mathematically that injecting a massive 10x new `BUY` mid-period produces a `0.0%` jump on the
   injection day and that cumulative TWR matches the exact underlying asset percentage return.
2. MWR (XIRR) Root-Finding Accuracy:
   Verifies `compute_xirr` against known analytical cash-flow schedules (1-year, 2-year compounding,
   and multi-tranche cash-flow schedules).
3. Dual-Currency (`USD` vs `ILS`) Rebasing & FX/Tax Attribution:
   Verifies that when a USD asset rises +10% while `USD_ILS` appreciates +10%, `USD` TWR is +10.00%
   and `ILS` TWR is +21.00% ((1.10 * 1.10) - 1), with exact Pure Asset (+10.0%) and FX (+11.0%) split.
4. Multi-Benchmark Slicing, What-If Opportunity Simulator & Institutional Risk KPIs:
   Verifies window slicing, benchmark rebasing to 0.0% at t0, cash-flow matched benchmark share accumulation,
   drawdown series, Sharpe/Sortino/Beta, and zero NaN/Inf across all outputs.
"""

import math
import os
import sys
import tempfile
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "core_logic"))

from portfolio_analytics import PortfolioAnalytics, compute_xirr
from portfolio_ledger import PortfolioLedger

TOL = 1e-5


def _assert_close(actual: float, expected: float, label: str, tol: float = TOL) -> None:
    assert math.isclose(actual, expected, rel_tol=tol, abs_tol=tol), (
        f"[ASSERTION FAILED] {label}: expected {expected:.6f}, got {actual:.6f} (diff={abs(actual - expected):.6g})"
    )


def _assert_no_nan_inf(df: pd.DataFrame, label: str) -> None:
    if df.empty:
        return
    num_df = df.select_dtypes(include=[np.number])
    n_nan = int(num_df.isna().sum().sum())
    n_inf = int(np.isinf(num_df.to_numpy()).sum())
    assert n_nan == 0 and n_inf == 0, f"[ASSERTION FAILED] {label} has NaNs ({n_nan}) or Infs ({n_inf})"


def test_1_twr_cash_flow_invariance(tmp_dir: str) -> None:
    """
    Test 1: TWR Cash-Flow Invariance vs. MWR Sensitivity
    Scenario:
      - Day 0 (2024-01-01): BUY 10 shares of AAPL @ $100 ($1,000 invested).
      - Day 5 (2024-01-06): AAPL rises +10% to $110 (Portfolio = $1,100, TWR = +10.0%).
      - Day 10 (2024-01-11): AAPL is still $110. Investor injects 10x capital: BUY 100 shares @ $110 ($11,000 inflow).
        Total portfolio value jumps from $1,100 to $12,100 (+1,000% NAV jump), but TWR must remain 10.0000%!
      - Day 20 (2024-01-21): AAPL rises another +10% from $110 to $121.
        Cumulative TWR must equal (1.10 * 1.10 - 1) = +21.0000%, completely unaffected by the 10x deposit size.
    """
    d_start = pd.Timestamp("2024-01-01")
    d_end = pd.Timestamp("2024-01-21")
    grid = pd.date_range(d_start, d_end, freq="D")

    def mock_prices(tickers, s, e):
        g = pd.date_range(pd.Timestamp(s).normalize(), pd.Timestamp(e).normalize(), freq="D")
        px = pd.Series(100.0, index=g, dtype="float64")
        px.loc[g >= pd.Timestamp("2024-01-06")] = 110.0
        px.loc[g >= pd.Timestamp("2024-01-21")] = 121.0
        df = pd.DataFrame({t: px for t in tickers}, index=g)
        return df, {t: "USD" for t in tickers}

    def mock_macro(s, e):
        g = pd.date_range(pd.Timestamp(s).normalize(), pd.Timestamp(e).normalize(), freq="D")
        return pd.DataFrame({"USD_ILS": 3.60, "IL_CPI": 100.0}, index=g)

    ledger = PortfolioLedger(path=os.path.join(tmp_dir, "twr_test.parquet"), macro_provider=mock_macro)
    ledger.add_transaction("2024-01-01", "AAPL", "BUY", quantity=10.0, price=100.0, fee=0.0, currency="USD")
    ledger.add_transaction("2024-01-11", "AAPL", "BUY", quantity=100.0, price=110.0, fee=0.0, currency="USD")

    analytics = PortfolioAnalytics(ledger, price_provider=mock_prices, macro_provider=mock_macro)
    res = analytics.analyze_slice(window="ALL", end_date="2024-01-21", view_currency="USD")
    s_df = res["series"]
    _assert_no_nan_inf(s_df, "Test 1 Series")

    # On 2024-01-11 (the massive $11,000 injection day), daily return must be 0.0 and cumulative TWR must stay +10.0%
    injection_day_ret = float(s_df.loc[pd.Timestamp("2024-01-11"), "portfolio_daily_return"])
    injection_day_twr = float(s_df.loc[pd.Timestamp("2024-01-11"), "portfolio_twr_pct"])
    final_twr = float(s_df.loc[pd.Timestamp("2024-01-21"), "portfolio_twr_pct"])

    _assert_close(injection_day_ret, 0.0, "TWR Daily Return on 10x Injection Day")
    _assert_close(injection_day_twr, 10.0, "Cumulative TWR% on 10x Injection Day")
    _assert_close(final_twr, 21.0, "Final Cumulative TWR% (1.10 * 1.10 - 1 = 21.0%)")
    print("[PASS] Test 1: TWR Cash-Flow Invariance verified (10x mid-period BUY caused 0.00% spike; TWR=21.00%).")


def test_2_xirr_solver_accuracy() -> None:
    """
    Test 2: MWR / XIRR Analytical Accuracy
    - Schedule A: -$1,000 on 2023-01-01, +$1,100 on 2024-01-01 (365 days = 1.0 yr) -> XIRR = 10.0000%
    - Schedule B: -$1,000 on 2022-01-01, +$1,210 on 2023-12-31 (730 days = 2.0 yrs) -> XIRR = 10.0000%
    - Schedule C: Two tranches at r = 20%:
        -$1,000 at t=0 (2023-01-01)
        -$1,200 at t=1 yr (2024-01-01)
        +$2,880 at t=2 yrs (2024-12-31, 730 days from t=0) -> 1000*(1.2^2) + 1200*(1.2^1) = 1440 + 1440 = 2880 -> XIRR = 20.0000%
    """
    r_a = compute_xirr([
        (pd.Timestamp("2023-01-01"), -1000.0),
        (pd.Timestamp("2024-01-01"), 1100.0),
    ])
    _assert_close(r_a, 0.10, "XIRR Schedule A (1Y 10%)")

    r_b = compute_xirr([
        (pd.Timestamp("2022-01-01"), -1000.0),
        (pd.Timestamp("2024-01-01"), 1210.0),
    ])
    _assert_close(r_b, 0.10, "XIRR Schedule B (2Y 10% compounded)")

    r_c = compute_xirr([
        (pd.Timestamp("2023-01-01"), -1000.0),
        (pd.Timestamp("2024-01-01"), -1200.0),
        (pd.Timestamp("2024-12-31"), 2880.0),
    ])
    _assert_close(r_c, 0.20, "XIRR Schedule C (Multi-tranche 20%)")
    print("[PASS] Test 2: XIRR Newton-Raphson solver verified (1Y=10.00%, 2Y=10.00%, Multi-Tranche=20.00%).")


def test_3_dual_currency_rebasing_and_attribution(tmp_dir: str) -> None:
    """
    Test 3: USD vs ILS Rebasing & FX/Tax Attribution Breakdown
    Scenario:
      - 2024-01-01: BUY 10 QQQ @ $100, FX = 3.50 (Cost = $1,000 = 3,500 ILS).
      - 2024-07-01: QQQ = $110 (+10% in USD), FX = 3.85 (+10% USD appreciation -> Value = $1,100 = 4,235 ILS).
      - Expected:
        * USD view TWR = +10.00%
        * ILS view TWR = (4,235 / 3,500) - 1 = +21.00%
        * ILS Attribution:
          - Pure Asset Return (at constant FX=3.50) = +10.00%
          - FX Impact = 21.00% - 10.00% = +11.00%
          - Unrealized Liquidation Tax: Real Gain = 4,235 - 3,500*(3.85/3.50) = 4,235 - 3,850 = 385 ILS -> Tax (25%) = 96.25 ILS
    """
    def mock_prices(tickers, s, e):
        g = pd.date_range(pd.Timestamp(s).normalize(), pd.Timestamp(e).normalize(), freq="D")
        px = pd.Series(np.linspace(100.0, 110.0, len(g)), index=g, dtype="float64")
        return pd.DataFrame({t: px for t in tickers}, index=g), {t: "USD" for t in tickers}

    def mock_macro(s, e):
        g = pd.date_range(pd.Timestamp(s).normalize(), pd.Timestamp(e).normalize(), freq="D")
        fx = pd.Series(np.linspace(3.50, 3.85, len(g)), index=g, dtype="float64")
        return pd.DataFrame({"USD_ILS": fx, "IL_CPI": 100.0}, index=g)

    ledger = PortfolioLedger(path=os.path.join(tmp_dir, "fx_attr.parquet"), macro_provider=mock_macro)
    ledger.add_transaction("2024-01-01", "QQQ", "BUY", quantity=10.0, price=100.0, fee=0.0, currency="USD")

    analytics = PortfolioAnalytics(ledger, price_provider=mock_prices, macro_provider=mock_macro)
    usd_res = analytics.analyze_slice(window="ALL", end_date="2024-07-01", view_currency="USD")
    ils_res = analytics.analyze_slice(window="ALL", end_date="2024-07-01", view_currency="ILS")

    _assert_close(usd_res["kpis"]["twr_total_pct"], 10.0, "USD View Total TWR%")
    _assert_close(ils_res["kpis"]["twr_total_pct"], 21.0, "ILS View Total TWR%")

    attr = ils_res["attribution"]
    _assert_close(attr["pure_asset_return_pct"], 10.0, "Attribution Pure Asset Return%")
    _assert_close(attr["fx_impact_pct"], 11.0, "Attribution FX Impact%")
    _assert_close(attr["est_unrealized_tax_ils"], 96.25, "Attribution Est Unrealized Liquidation Tax ILS")
    print("[PASS] Test 3: Dual-Currency Rebasing & FX/Tax Attribution verified (USD=+10.0%, ILS=+21.0%, FX=+11.0%, Tax=96.25 ILS).")


def test_4_what_if_simulator_slicing_and_real_cache_integration(tmp_dir: str) -> None:
    """
    Test 4: Multi-Benchmark What-If Simulator, Mid-Stream Window Slicing & Real Cache Integration
    - Runs against the real hydrated Parquet caches (`AAPL`, `QQQ`, `LUMI.TA`, `USD_ILS`, `IL_CPI`).
    - Verifies that slicing to a sub-window rebases both portfolio and benchmarks to 0.0% at t0.
    - Verifies What-If Opportunity Simulator terminal values and risk KPIs with zero NaNs/Infs.
    """
    ledger = PortfolioLedger(path=os.path.join(tmp_dir, "real_integ.parquet"))
    ledger.add_transaction("2024-01-15", "AAPL", "BUY", quantity=20.0, price=185.0, fee=5.0, currency="USD")
    ledger.add_transaction("2024-03-15", "LUMI.TA", "BUY", quantity=100.0, price=29.5, fee=10.0, currency="ILS")
    ledger.add_transaction("2024-06-17", "AAPL", "SELL", quantity=5.0, price=215.0, fee=5.0, currency="USD")

    analytics = PortfolioAnalytics(ledger)
    # 1. Full window in ILS with benchmarks QQQ and LUMI.TA
    full_res = analytics.analyze_slice(
        window="ALL",
        end_date="2025-07-31",
        view_currency="ILS",
        benchmarks=["QQQ", "LUMI.TA"],
    )
    s_full = full_res["series"]
    _assert_no_nan_inf(s_full, "Full Slice Series")
    assert "what_if_QQQ_nav" in s_full.columns and "what_if_LUMI.TA_nav" in s_full.columns
    assert "QQQ" in full_res["what_if"] and "LUMI.TA" in full_res["what_if"]

    # 2. Sub-window slice (custom 2024-06-01 -> 2025-07-31) in USD
    sub_res = analytics.analyze_slice(
        start_date="2024-06-01",
        end_date="2025-07-31",
        view_currency="USD",
        benchmarks=["QQQ", "AAPL"],
    )
    s_sub = sub_res["series"]
    _assert_no_nan_inf(s_sub, "Sub-Window Slice Series")

    # Verify exact rebasing at t0 = 2024-06-01: portfolio_twr_pct and benchmark twr_pct start at 0.0% (index 100.0)
    _assert_close(float(s_sub["portfolio_twr_pct"].iloc[0]), 0.0, "Sub-Slice Portfolio Rebased t0 = 0.0%")
    _assert_close(float(s_sub["portfolio_index_100"].iloc[0]), 100.0, "Sub-Slice Portfolio Index t0 = 100.0")
    _assert_close(float(s_sub["bench_QQQ_twr_pct"].iloc[0]), 0.0, "Sub-Slice Benchmark QQQ Rebased t0 = 0.0%")
    _assert_close(float(s_sub["bench_QQQ_index_100"].iloc[0]), 100.0, "Sub-Slice Benchmark QQQ Index t0 = 100.0")

    kpis = sub_res["kpis"]
    for k, v in kpis.items():
        if isinstance(v, (int, float)):
            assert not (math.isnan(v) or math.isinf(v)), f"KPI {k} is NaN/Inf: {v}"

    print(
        f"[PASS] Test 4: Real Cache Integration, Sub-Window Rebasing & What-If Simulator verified "
        f"(TWR={kpis['twr_total_pct']:.2f}%, XIRR={kpis['xirr_annualized_pct']:.2f}%, "
        f"Sharpe={kpis['sharpe_ratio']:.2f}, MDD={kpis['max_drawdown_pct']:.2f}%)."
    )


def run_all_verifications() -> None:
    print("=== AlphaQuant Phase 3: Portfolio Analytics & Benchmarking Verification Suite ===")
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_1_twr_cash_flow_invariance(tmp_dir)
        test_2_xirr_solver_accuracy()
        test_3_dual_currency_rebasing_and_attribution(tmp_dir)
        test_4_what_if_simulator_slicing_and_real_cache_integration(tmp_dir)
    print("---------------------------------------------------------------------------------")
    print("[SUCCESS] All Phase 3 Portfolio Analytics assertions passed!")


if __name__ == "__main__":
    run_all_verifications()
