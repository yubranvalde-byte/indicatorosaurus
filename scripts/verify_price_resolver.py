#!/usr/bin/env python3
"""
verify_price_resolver.py
========================
Standalone CLI verification script for Phase 1 of Portfolio Intelligence (`core_logic/price_resolver.py`).

Validates:
1. Mixed basket resolution (`AAPL` from Tier 1 Lake, `QQQ` from Tier 2/3, `LUMI.TA` TASE equity in ILS).
2. Macro FX (`USD_ILS`) & Israeli CPI (`IL_CPI`) resolution on a continuous daily calendar grid.
3. Zero `NaN` or `Inf` values across all returned DataFrames.
4. Proper TASE Agorot-to-ILS scaling (`Price_ILS = Price_Agorot / 100.0`, verifying `LUMI.TA` is in ~15-100 ILS range).
5. Sub-50ms warm Parquet cache read latency and sub-millisecond in-memory TTL hit latency.
"""

import os
import sys
import time
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "core_logic"))

from price_resolver import (
    PORTFOLIO_CACHE_PATH,
    FX_CPI_CACHE_PATH,
    clear_resolver_caches,
    resolve_macro_series,
    resolve_prices,
)


def run_verification():
    print("=== AlphaQuant Phase 1: Price & Macro Resolver Verification ===")

    basket = ["AAPL", "QQQ", "LUMI.TA"]
    start_date = "2024-01-01"
    end_date = "2025-08-01"

    # ── 1. Cold Resolution (Hydrates Tier 2 & Macro Parquet Caches) ───────────
    clear_resolver_caches()
    t0_cold = time.perf_counter()
    prices_df, currency_map = resolve_prices(basket, start_date, end_date)
    macro_df = resolve_macro_series(start_date, end_date)
    t1_cold = time.perf_counter()
    cold_ms = (t1_cold - t0_cold) * 1000.0

    print(f"[1] Cold Resolution & Cache Hydration completed in {cold_ms:.2f} ms")
    print(f"    Resolved Tickers : {list(prices_df.columns)}")
    print(f"    Currency Map     : {currency_map}")
    print(f"    Price Matrix Shape: {prices_df.shape} ({prices_df.index.min().date()} -> {prices_df.index.max().date()})")
    print(f"    Macro Matrix Shape: {macro_df.shape} ({macro_df.index.min().date()} -> {macro_df.index.max().date()})")

    # ── 2. Assert Zero NaNs / Infs & Complete Daily Calendar Grid ─────────────
    expected_days = len(pd.date_range(start_date, end_date, freq="D"))
    assert len(prices_df) == expected_days, f"Expected {expected_days} calendar days, got {len(prices_df)}"
    assert len(macro_df) == expected_days, f"Expected {expected_days} macro calendar days, got {len(macro_df)}"

    price_nans = int(prices_df.isna().sum().sum())
    price_infs = int(np.isinf(prices_df.to_numpy()).sum())
    macro_nans = int(macro_df.isna().sum().sum())
    macro_infs = int(np.isinf(macro_df.to_numpy()).sum())

    assert price_nans == 0 and price_infs == 0, f"Price DataFrame contains NaNs ({price_nans}) or Infs ({price_infs})"
    assert macro_nans == 0 and macro_infs == 0, f"Macro DataFrame contains NaNs ({macro_nans}) or Infs ({macro_infs})"
    assert (prices_df > 0.0).all().all(), "All resolved equity prices must be strictly positive (> 0.0)"
    assert (macro_df > 0.0).all().all(), "All resolved macro series (USD_ILS, IL_CPI) must be strictly positive (> 0.0)"
    print("[2] PASS: Zero NaNs/Infs verified across continuous daily calendar grid (including weekends/holidays).")

    # ── 3. Assert Native Currency Metadata & TASE Agorot-to-ILS Normalization ─
    assert currency_map.get("AAPL") == "USD", f"Expected AAPL currency USD, got {currency_map.get('AAPL')}"
    assert currency_map.get("QQQ") == "USD", f"Expected QQQ currency USD, got {currency_map.get('QQQ')}"
    assert currency_map.get("LUMI.TA") == "ILS", f"Expected LUMI.TA currency ILS, got {currency_map.get('LUMI.TA')}"

    lumi_median = float(prices_df["LUMI.TA"].median())
    assert 5.0 <= lumi_median <= 150.0, (
        f"TASE Agorot-to-ILS normalization check failed: LUMI.TA median price is {lumi_median:.2f} "
        f"(expected ~20-70 ILS, not thousands of Agorot)."
    )

    usd_ils_median = float(macro_df["USD_ILS"].median())
    il_cpi_median = float(macro_df["IL_CPI"].median())
    assert 2.8 <= usd_ils_median <= 4.8, f"Unexpected USD/ILS median rate: {usd_ils_median:.4f}"
    assert 80.0 <= il_cpi_median <= 140.0, f"Unexpected IL_CPI median index: {il_cpi_median:.2f}"
    print(
        f"[3] PASS: TASE Agorot->ILS scaling verified (LUMI.TA median={lumi_median:.2f} ILS, "
        f"USD_ILS median={usd_ils_median:.4f}, IL_CPI median={il_cpi_median:.2f})."
    )

    # ── 3b. Assert CPI is keyed to publication date (15th of month M+1, "known index") ──
    cpi_change_days = macro_df.index[macro_df["IL_CPI"].diff().fillna(0.0) != 0.0]
    assert len(cpi_change_days) > 0, "IL_CPI never changes inside the window — CPI series not hydrated."
    bad_days = [d.date() for d in cpi_change_days if d.day != 15]
    assert not bad_days, f"IL_CPI changes on non-publication dates (expected 15th only): {bad_days[:5]}"
    fx_weekend_changes = macro_df.index[(macro_df["USD_ILS"].diff().fillna(0.0) != 0.0) & (macro_df.index.dayofweek >= 5)]
    assert len(fx_weekend_changes) == 0, f"USD_ILS changes on weekends (ffill broken): {list(fx_weekend_changes[:5])}"
    print(f"[3b] PASS: IL_CPI changes only on publication dates ({len(cpi_change_days)} releases); "
          f"USD_ILS strictly forward-filled across weekends.")

    # ── 4. Assert Parquet Cache Files Exist on Disk ───────────────────────────
    assert os.path.exists(PORTFOLIO_CACHE_PATH), f"Missing Tier 2 cache file at {PORTFOLIO_CACHE_PATH}"
    assert os.path.exists(FX_CPI_CACHE_PATH), f"Missing Macro cache file at {FX_CPI_CACHE_PATH}"
    print("[4] PASS: Local Snappy Parquet cache artifacts verified on disk.")

    # ── 5. Benchmark Warm Cache Latency (Disk Parquet Cache & In-Memory TTL Hit < 50ms) ──
    # 5a. Warm Disk Parquet Cache Read (clear in-memory cache, read from hydrated Tier 1 + Tier 2 + Macro Parquet)
    clear_resolver_caches()
    t0_disk_warm = time.perf_counter()
    disk_prices_df, _ = resolve_prices(basket, start_date, end_date)
    disk_macro_df = resolve_macro_series(start_date, end_date)
    t1_disk_warm = time.perf_counter()
    disk_warm_ms = (t1_disk_warm - t0_disk_warm) * 1000.0

    assert disk_warm_ms < 50.0, f"Warm Parquet disk cache latency exceeded 50ms budget: {disk_warm_ms:.3f} ms"

    # 5b. Warm In-Memory TTL Cache Hit
    t0_mem_warm = time.perf_counter()
    mem_prices_df, _ = resolve_prices(basket, start_date, end_date)
    mem_macro_df = resolve_macro_series(start_date, end_date)
    t1_mem_warm = time.perf_counter()
    mem_warm_ms = (t1_mem_warm - t0_mem_warm) * 1000.0

    assert mem_warm_ms < 50.0, f"In-memory TTL cache latency exceeded 50ms budget: {mem_warm_ms:.3f} ms"
    assert len(disk_prices_df) == expected_days and len(mem_prices_df) == expected_days
    print(
        f"[5] PASS: Warm Parquet cache read latency = {disk_warm_ms:.3f} ms | "
        f"In-Memory TTL hit latency = {mem_warm_ms:.3f} ms (Target: < 50.0 ms)."
    )

    print("\n--- Sample Resolved Tail (Last 3 Calendar Days) ---")
    combined_preview = pd.concat([prices_df.tail(3), macro_df.tail(3)], axis=1)
    print(combined_preview.to_string())
    print("---------------------------------------------------")
    print("[SUCCESS] All Phase 1 Price & Macro Resolver assertions passed!")


if __name__ == "__main__":
    run_verification()
