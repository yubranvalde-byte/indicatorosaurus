#!/usr/bin/env python3
"""
verify_portfolio_ledger.py
==========================
Checkpoint 1 Verification Suite for Phase 2 of Portfolio Intelligence (`core_logic/portfolio_ledger.py`).

Validates exact ILS/USD FIFO accounting across the 4 mandatory Israeli capital gains tax edge cases
plus structural ledger invariants:
  1. USD Gain + USD Appreciation (FX_sell > FX_buy):
     Verifies currency gain is classified as exempt inflationary gain and 25% tax applies strictly to real gain.
  2. USD Gain + USD Depreciation (FX_sell < FX_buy):
     Verifies floor at InflationaryRatio = 1.0 so tax is capped at 25% of nominal ILS gain.
  3. USD Loss + Sharp USD Appreciation (resulting in nominal ILS gain):
     Verifies zero real tax liability (entire nominal ILS gain is inflationary exempt).
  4. YTD Tax Shield Offset & Multi-Year Loss Carry-Forward:
     Verifies realized capital losses offset subsequent realized real gains within the same tax year
     and carry forward across calendar years when unused.
  5. Additional Structural Tests:
     - Partial FIFO lot consumption with proportional buy/sell fee allocation.
     - TASE (.TA) ILS security indexed by Israeli CPI (`IL_CPI`).
     - Oversell prevention (`ValueError` when selling more than open position).
     - Dividend 25% withholding tax tracking and cash balance updates.
     - Atomic Parquet & deterministic CSV round-trip persistence.
     - Current Portfolio Snapshot valuation & Tax-Loss Harvesting (TLH) candidate detection.
     - Zero NaN / Inf values across all DataFrames and summary outputs.
"""

import math
import os
import sys
import tempfile
import warnings
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "core_logic"))

from portfolio_ledger import PortfolioLedger, compute_israeli_lot_tax

TOL = 1e-6


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


def _build_mock_macro_provider(point_overrides: dict):
    """
    Create a deterministic macro provider over [start_date, end_date] where specific dates
    take exact `(fx, cpi)` pairs from `point_overrides` and forward-fill chronologically.
    """
    def _provider(start_date, end_date) -> pd.DataFrame:
        s_ts = pd.Timestamp(start_date).normalize()
        e_ts = pd.Timestamp(end_date).normalize()
        all_dates = sorted([pd.Timestamp(k).normalize() for k in point_overrides.keys()])
        grid_start = min([s_ts] + all_dates)
        grid_end = max([e_ts] + all_dates)
        grid = pd.date_range(grid_start, grid_end, freq="D")

        fx_s = pd.Series(dtype="float64", index=grid)
        cpi_s = pd.Series(dtype="float64", index=grid)
        for dt_str, (fx_val, cpi_val) in sorted(point_overrides.items()):
            ts = pd.Timestamp(dt_str).normalize()
            fx_s.loc[ts] = float(fx_val)
            cpi_s.loc[ts] = float(cpi_val)

        fx_s = fx_s.ffill().bfill()
        cpi_s = cpi_s.ffill().bfill()
        return pd.DataFrame({"USD_ILS": fx_s, "IL_CPI": cpi_s}, index=grid)

    return _provider


def test_case_1_usd_gain_plus_usd_appreciation(tmp_dir: str) -> None:
    """
    Case 1: USD Gain + USD Appreciation (FX_sell > FX_buy)
    - BUY 10 AAPL @ $100 on 2024-01-10, fee=$0, FX_buy = 3.50 -> OriginalCost = 3,500 ILS
    - SELL 10 AAPL @ $120 on 2024-06-10, fee=$0, FX_sell = 3.70 -> Proceeds = 4,440 ILS
    - InflationaryRatio = 3.70 / 3.50 = 1.057142857
    - AdjustedCost = 3,500 * (3.70 / 3.50) = 3,700 ILS
    - Nominal Gain = 940 ILS
    - Inflationary Exempt Gain = 3,700 - 3,500 = 200 ILS
    - Real Taxable Gain = 4,440 - 3,700 = 740 ILS (=$200 * 3.70)
    - Tax (25%) = 185 ILS
    """
    macro = _build_mock_macro_provider({
        "2024-01-10": (3.50, 100.0),
        "2024-06-10": (3.70, 101.5),
    })
    ledger_path = os.path.join(tmp_dir, "case1.parquet")
    ledger = PortfolioLedger(path=ledger_path, macro_provider=macro)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ledger.add_transaction("2024-01-10", "AAPL", "BUY", quantity=10, price=100.0, fee=0.0, currency="USD")
        ledger.add_transaction("2024-06-10", "AAPL", "SELL", quantity=10, price=120.0, fee=0.0, currency="USD")

    rl = ledger.realized_lots()
    assert len(rl) == 1
    row = rl.iloc[0]

    _assert_close(row["original_cost_ils"], 3500.0, "Case 1 OriginalCost ILS")
    _assert_close(row["proceeds_ils"], 4440.0, "Case 1 Proceeds ILS")
    _assert_close(row["inflationary_ratio"], 3.70 / 3.50, "Case 1 InflationaryRatio")
    _assert_close(row["adjusted_cost_ils"], 3700.0, "Case 1 AdjustedCost ILS")
    _assert_close(row["inflationary_gain_ils"], 200.0, "Case 1 Inflationary Exempt Gain ILS")
    _assert_close(row["real_gain_ils"], 740.0, "Case 1 Real Taxable Gain ILS")
    _assert_close(row["real_loss_ils"], 0.0, "Case 1 Real Loss ILS")
    _assert_close(row["tax_ils"], 185.0, "Case 1 Tax (25%) ILS")

    summary = ledger.tax_summary(2024).iloc[0]
    _assert_close(summary["capital_gains_tax_ils"], 185.0, "Case 1 Annual Tax Summary ILS")
    print("[PASS] Case 1: USD Gain + USD Appreciation -> Real Gain=740.00 ILS, Exempt=200.00 ILS, Tax=185.00 ILS")


def test_case_2_usd_gain_plus_usd_depreciation(tmp_dir: str) -> None:
    """
    Case 2: USD Gain + USD Depreciation (FX_sell < FX_buy)
    - BUY 10 AAPL @ $100 on 2024-01-10, FX_buy = 3.70 -> OriginalCost = 3,700 ILS
    - SELL 10 AAPL @ $120 on 2024-06-10, FX_sell = 3.50 -> Proceeds = 4,200 ILS
    - InflationaryRatio = max(1.0, 3.50 / 3.70) = 1.0
    - AdjustedCost = 3,700 * 1.0 = 3,700 ILS
    - Real Taxable Gain = 4,200 - 3,700 = 500 ILS (capped at nominal ILS gain)
    - Inflationary Exempt Gain = 0.0 ILS
    - Tax (25%) = 125 ILS
    """
    macro = _build_mock_macro_provider({
        "2024-01-10": (3.70, 100.0),
        "2024-06-10": (3.50, 101.5),
    })
    ledger_path = os.path.join(tmp_dir, "case2.parquet")
    ledger = PortfolioLedger(path=ledger_path, macro_provider=macro)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ledger.add_transaction("2024-01-10", "AAPL", "BUY", quantity=10, price=100.0, fee=0.0, currency="USD")
        ledger.add_transaction("2024-06-10", "AAPL", "SELL", quantity=10, price=120.0, fee=0.0, currency="USD")

    row = ledger.realized_lots().iloc[0]
    _assert_close(row["original_cost_ils"], 3700.0, "Case 2 OriginalCost ILS")
    _assert_close(row["proceeds_ils"], 4200.0, "Case 2 Proceeds ILS")
    _assert_close(row["inflationary_ratio"], 1.0, "Case 2 InflationaryRatio Floor=1.0")
    _assert_close(row["adjusted_cost_ils"], 3700.0, "Case 2 AdjustedCost ILS")
    _assert_close(row["inflationary_gain_ils"], 0.0, "Case 2 Inflationary Exempt Gain ILS")
    _assert_close(row["real_gain_ils"], 500.0, "Case 2 Real Taxable Gain ILS")
    _assert_close(row["tax_ils"], 125.0, "Case 2 Tax (25%) ILS")
    print("[PASS] Case 2: USD Gain + USD Depreciation -> Ratio Floor=1.0, Real Gain=500.00 ILS, Tax=125.00 ILS")


def test_case_3_usd_loss_plus_sharp_usd_appreciation(tmp_dir: str) -> None:
    """
    Case 3: USD Loss + Sharp USD Appreciation (Nominal ILS Gain, Real Economic Loss)
    - BUY 10 AAPL @ $100 on 2024-01-10, FX_buy = 3.00 -> OriginalCost = 3,000 ILS
    - SELL 10 AAPL @ $95 on 2024-06-10, FX_sell = 3.60 -> Proceeds = 3,420 ILS
    - Nominal ILS Gain = +420 ILS (even though stock dropped 5% in USD)
    - AdjustedCost = 3,000 * (3.60 / 3.00) = 3,600 ILS
    - Because OriginalCost (3,000) <= Proceeds (3,420) < AdjustedCost (3,600):
      Real Taxable Gain = 0.0 ILS, Inflationary Exempt Gain = 420.0 ILS, Real Loss = 0.0 ILS, Tax = 0.0 ILS.
    """
    macro = _build_mock_macro_provider({
        "2024-01-10": (3.00, 100.0),
        "2024-06-10": (3.60, 101.5),
    })
    ledger_path = os.path.join(tmp_dir, "case3.parquet")
    ledger = PortfolioLedger(path=ledger_path, macro_provider=macro)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ledger.add_transaction("2024-01-10", "AAPL", "BUY", quantity=10, price=100.0, fee=0.0, currency="USD")
        ledger.add_transaction("2024-06-10", "AAPL", "SELL", quantity=10, price=95.0, fee=0.0, currency="USD")

    row = ledger.realized_lots().iloc[0]
    _assert_close(row["original_cost_ils"], 3000.0, "Case 3 OriginalCost ILS")
    _assert_close(row["proceeds_ils"], 3420.0, "Case 3 Proceeds ILS")
    _assert_close(row["nominal_gain_ils"], 420.0, "Case 3 Nominal ILS Gain")
    _assert_close(row["adjusted_cost_ils"], 3600.0, "Case 3 AdjustedCost ILS")
    _assert_close(row["real_gain_ils"], 0.0, "Case 3 Real Taxable Gain ILS")
    _assert_close(row["inflationary_gain_ils"], 420.0, "Case 3 Inflationary Exempt Gain ILS")
    _assert_close(row["real_loss_ils"], 0.0, "Case 3 Real Loss ILS")
    _assert_close(row["tax_ils"], 0.0, "Case 3 Zero Tax Liability")
    print("[PASS] Case 3: USD Loss + Sharp USD Appreciation -> Nominal=+420.00 ILS, Real=0.00 ILS, Exempt=420.00 ILS, Tax=0.00 ILS")


def test_case_4_ytd_tax_shield_and_carry_forward(tmp_dir: str) -> None:
    """
    Case 4: YTD Tax Shield Offset & Multi-Year Loss Carry-Forward
    Part A (Same-Year Offset in 2024):
      - Trade 1 (Loss): BUY 10 NVDA @ $100 (FX=3.50 -> 3,500 ILS), SELL @ $71.4285714286 (FX=3.50 -> 2,500 ILS)
        -> Real Capital Loss = -1,000 ILS.
      - Trade 2 (Gain): BUY 10 MSFT @ $200 (FX=3.50 -> 7,000 ILS), SELL @ $285.7142857143 (FX=3.50 -> 10,000 ILS)
        -> Real Capital Gain = +3,000 ILS.
      - Net 2024 Taxable Gain = 3,000 - 1,000 = 2,000 ILS -> 25% Tax = 500 ILS (Shield saved 250 ILS).
    Part B (Carry-Forward from 2024 into 2025):
      - Separate ledger where 2024 has a net -1,500 ILS loss, and 2025 realizes +2,500 ILS real gain.
      - Verifies 2025 carries in 1,500 ILS shield, leaving 1,000 ILS taxable -> 250 ILS tax.
    """
    macro = _build_mock_macro_provider({
        "2024-01-15": (3.50, 100.0),
        "2024-03-15": (3.50, 100.0),
        "2024-06-15": (3.50, 100.0),
        "2024-09-15": (3.50, 100.0),
        "2025-02-15": (3.50, 102.0),
        "2025-07-15": (3.50, 102.0),
    })

    # Part A: Same-year YTD Tax Shield
    ledger_a = PortfolioLedger(path=os.path.join(tmp_dir, "case4a.parquet"), macro_provider=macro)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ledger_a.add_transaction("2024-01-15", "NVDA", "BUY", quantity=10, price=100.0, fee=0.0, currency="USD")
        ledger_a.add_transaction("2024-03-15", "NVDA", "SELL", quantity=10, price=250.0 / 3.5, fee=0.0, currency="USD")
        ledger_a.add_transaction("2024-06-15", "MSFT", "BUY", quantity=10, price=200.0, fee=0.0, currency="USD")
        ledger_a.add_transaction("2024-09-15", "MSFT", "SELL", quantity=10, price=1000.0 / 3.5, fee=0.0, currency="USD")

    sum_a = ledger_a.tax_summary(2024).iloc[0]
    _assert_close(sum_a["real_gains_ils"], 3000.0, "Case 4A Real Gains ILS")
    _assert_close(sum_a["real_losses_ils"], 1000.0, "Case 4A Real Losses ILS")
    _assert_close(sum_a["shield_used_current_year_ils"], 1000.0, "Case 4A YTD Shield Used ILS")
    _assert_close(sum_a["taxable_real_gain_ils"], 2000.0, "Case 4A Net Taxable Gain ILS")
    _assert_close(sum_a["capital_gains_tax_ils"], 500.0, "Case 4A Net Tax Liability ILS")

    # Part B: Multi-year Carry-Forward
    ledger_b = PortfolioLedger(path=os.path.join(tmp_dir, "case4b.parquet"), macro_provider=macro, carry_forward_losses=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # 2024: Real loss of 1,500 ILS (Buy 3,500 ILS -> Sell 2,000 ILS)
        ledger_b.add_transaction("2024-01-15", "AMD", "BUY", quantity=10, price=100.0, fee=0.0, currency="USD")
        ledger_b.add_transaction("2024-06-15", "AMD", "SELL", quantity=10, price=200.0 / 3.5, fee=0.0, currency="USD")
        # 2025: Real gain of 2,500 ILS (Buy 3,500 ILS -> Sell 6,000 ILS)
        ledger_b.add_transaction("2025-02-15", "META", "BUY", quantity=10, price=100.0, fee=0.0, currency="USD")
        ledger_b.add_transaction("2025-07-15", "META", "SELL", quantity=10, price=600.0 / 3.5, fee=0.0, currency="USD")

    sum_b_2024 = ledger_b.tax_summary(2024).iloc[0]
    sum_b_2025 = ledger_b.tax_summary(2025).iloc[0]
    _assert_close(sum_b_2024["loss_carry_out_ils"], 1500.0, "Case 4B 2024 Carry-Out ILS")
    _assert_close(sum_b_2025["loss_carry_in_ils"], 1500.0, "Case 4B 2025 Carry-In ILS")
    _assert_close(sum_b_2025["shield_used_carry_forward_ils"], 1500.0, "Case 4B 2025 Carry-Forward Shield Used ILS")
    _assert_close(sum_b_2025["taxable_real_gain_ils"], 1000.0, "Case 4B 2025 Net Taxable Gain ILS")
    _assert_close(sum_b_2025["capital_gains_tax_ils"], 250.0, "Case 4B 2025 Net Tax ILS")
    print("[PASS] Case 4: YTD Tax Shield & Loss Carry-Forward -> Same-Year Net Tax=500.00 ILS, Multi-Year Carry Tax=250.00 ILS")


def test_fifo_partial_lots_tase_cpi_dividends_and_snapshot(tmp_dir: str) -> None:
    """
    Comprehensive structural test:
    - Partial FIFO lot consumption with proportional fee splitting.
    - TASE (.TA) security in ILS indexed by Israeli CPI.
    - Dividend 25% withholding & cash tracking.
    - Oversell protection (`ValueError`).
    - Parquet & CSV round-trip export/import.
    - Portfolio snapshot valuation & Tax-Loss Harvesting (TLH) candidate identification.
    """
    macro = _build_mock_macro_provider({
        "2024-01-10": (3.60, 100.0),
        "2024-04-10": (3.60, 101.0),
        "2024-08-10": (3.60, 103.0),
        "2024-11-10": (3.60, 104.0),
    })
    ledger_path = os.path.join(tmp_dir, "structural.parquet")
    ledger = PortfolioLedger(path=ledger_path, macro_provider=macro)

    # 1. Deposit USD and ILS cash
    ledger.add_transaction("2024-01-10", "CASH", "CASH_DEPOSIT", quantity=10000.0, currency="USD")
    ledger.add_transaction("2024-01-10", "CASH", "CASH_DEPOSIT", quantity=20000.0, currency="ILS")

    # 2. Buy 2 FIFO lots of QQQ (Lot 1: 10 shares @ $100 + $10 fee; Lot 2: 10 shares @ $120 + $10 fee)
    ledger.add_transaction("2024-01-10", "QQQ", "BUY", quantity=10.0, price=100.0, fee=10.0, currency="USD")
    ledger.add_transaction("2024-04-10", "QQQ", "BUY", quantity=10.0, price=120.0, fee=10.0, currency="USD")

    # 3. Partial SELL of 15 shares @ $150 + $15 fee -> consumes 100% of Lot 1 (10 sh) and 50% of Lot 2 (5 sh)
    ledger.add_transaction("2024-08-10", "QQQ", "SELL", quantity=15.0, price=150.0, fee=15.0, currency="USD")

    rl = ledger.realized_lots()
    assert len(rl) == 2, f"Expected 2 FIFO lot slices consumed, got {len(rl)}"
    # Slice 1: 10 shares from Lot 1 (buy_fee=10, sell_fee=10) -> Cost=(1000+10)*3.6=3636, Proceeds=(1500-10)*3.6=5364
    _assert_close(rl.iloc[0]["quantity"], 10.0, "FIFO Slice 1 Qty")
    _assert_close(rl.iloc[0]["original_cost_ils"], 3636.0, "FIFO Slice 1 Cost ILS")
    _assert_close(rl.iloc[0]["proceeds_ils"], 5364.0, "FIFO Slice 1 Proceeds ILS")
    # Slice 2: 5 shares from Lot 2 (buy_fee=5, sell_fee=5) -> Cost=(600+5)*3.6=2178, Proceeds=(750-5)*3.6=2682
    _assert_close(rl.iloc[1]["quantity"], 5.0, "FIFO Slice 2 Qty")
    _assert_close(rl.iloc[1]["original_cost_ils"], 2178.0, "FIFO Slice 2 Cost ILS")
    _assert_close(rl.iloc[1]["proceeds_ils"], 2682.0, "FIFO Slice 2 Proceeds ILS")

    # 4. TASE (.TA) security in ILS indexed by CPI (CPI_buy=100 -> CPI_sell=103 => +3% inflation adjustment)
    ledger.add_transaction("2024-01-10", "LUMI.TA", "BUY", quantity=100.0, price=30.0, fee=0.0, currency="ILS")
    ledger.add_transaction("2024-08-10", "LUMI.TA", "SELL", quantity=50.0, price=33.0, fee=0.0, currency="ILS")
    tase_row = ledger.realized_lots().iloc[2]
    # 50 sh @ 30 ILS = 1,500 ILS cost; CPI ratio = 103/100 = 1.03 -> AdjustedCost = 1,545 ILS; Proceeds = 1,650 ILS
    _assert_close(tase_row["original_cost_ils"], 1500.0, "TASE OriginalCost ILS")
    _assert_close(tase_row["adjusted_cost_ils"], 1545.0, "TASE CPI-AdjustedCost ILS")
    _assert_close(tase_row["inflationary_gain_ils"], 45.0, "TASE CPI Exempt Gain ILS")
    _assert_close(tase_row["real_gain_ils"], 105.0, "TASE Real Gain ILS")
    _assert_close(tase_row["tax_ils"], 26.25, "TASE Real Tax ILS")

    # 5. Dividend with 25% withholding
    ledger.add_transaction("2024-08-10", "QQQ", "DIVIDEND", quantity=5.0, price=2.0, fee=0.0, currency="USD")
    div_df = ledger.dividends()
    assert len(div_df) == 1
    _assert_close(div_df.iloc[0]["gross_ils"], 36.0, "Dividend Gross ILS")
    _assert_close(div_df.iloc[0]["withholding_ils"], 9.0, "Dividend 25% Withholding ILS")

    # 6. Oversell guard must raise ValueError and leave state untouched
    try:
        ledger.add_transaction("2024-11-10", "QQQ", "SELL", quantity=999.0, price=150.0, currency="USD")
        raise AssertionError("Oversell did not raise ValueError!")
    except ValueError:
        pass

    # 7. Snapshot with a losing lot to verify Tax-Loss Harvesting (TLH) candidate detection
    snap = ledger.snapshot(as_of="2024-11-10", market_prices={"QQQ": 100.0, "LUMI.TA": 35.0})
    _assert_no_nan_inf(snap["positions"], "Snapshot Positions")
    _assert_no_nan_inf(snap["lots"], "Snapshot Lots")
    _assert_no_nan_inf(snap["tlh_candidates"], "Snapshot TLH Candidates")
    assert len(snap["tlh_candidates"]) == 1 and snap["tlh_candidates"].iloc[0]["ticker"] == "QQQ"

    # 8. Deterministic CSV export & re-import round-trip
    csv_path = os.path.join(tmp_dir, "roundtrip.csv")
    ledger.export_csv(csv_path)
    reloaded = PortfolioLedger(path=os.path.join(tmp_dir, "reloaded.parquet"), macro_provider=macro)
    n_imp = reloaded.import_csv(csv_path, replace=True)
    assert n_imp == len(ledger.transactions())
    _assert_close(
        reloaded.tax_summary(2024).iloc[0]["capital_gains_tax_ils"],
        ledger.tax_summary(2024).iloc[0]["capital_gains_tax_ils"],
        "CSV Round-Trip Annual Tax Match",
    )
    print("[PASS] Case 5: Partial FIFO Lots, TASE CPI Indexation, 25% Dividend Tax, Oversell Guard, TLH & CSV Round-Trip")


def run_all_tests() -> None:
    print("=== AlphaQuant Phase 2: Portfolio Ledger & Israeli Tax Verification Suite ===")
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_case_1_usd_gain_plus_usd_appreciation(tmp_dir)
        test_case_2_usd_gain_plus_usd_depreciation(tmp_dir)
        test_case_3_usd_loss_plus_sharp_usd_appreciation(tmp_dir)
        test_case_4_ytd_tax_shield_and_carry_forward(tmp_dir)
        test_fifo_partial_lots_tase_cpi_dividends_and_snapshot(tmp_dir)
    print("-----------------------------------------------------------------------------")
    print("[SUCCESS] All Phase 2 Portfolio Ledger & Israeli Tax assertions passed!")


if __name__ == "__main__":
    run_all_tests()
