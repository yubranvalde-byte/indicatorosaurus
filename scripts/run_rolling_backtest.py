#!/usr/bin/env python3
"""
run_rolling_backtest.py
=======================
Deterministic execution wrapper for running multi-factor rolling-window strategy backtests.
Supports selecting strategy preset, trailing window ('3y' or '5y'), and customizable weight vectors.
"""

import sys
import os
import argparse
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "core_logic"))

from backtest_engine import run_backtest


def main():
    parser = argparse.ArgumentParser(description="AlphaQuant Rolling Backtest Runner")
    parser.add_argument("--strategy", type=str, default="momentum",
                        choices=["deep-value", "hyper-growth", "momentum", "income", "quality"],
                        help="Strategy preset identifier")
    parser.add_argument("--window", type=str, default="3y", choices=["3y", "5y"],
                        help="Historical backtest window ('3y' or '5y')")
    parser.add_argument("--min-cap", type=float, default=1e9,
                        help="Minimum market cap filter (default: $1B)")
    parser.add_argument("--max-cap", type=float, default=None,
                        help="Maximum market cap filter (optional)")

    args = parser.parse_args()

    print(f"=== AlphaQuant Rolling Backtest Execution ===")
    print(f"Strategy: {args.strategy}")
    print(f"Window:   {args.window}")
    print(f"Min Cap:  ${args.min_cap:,.0f}")
    print("=============================================")

    # Uniform weights across 16 indicators if not specified
    weights = [1.0] * 16

    try:
        results = run_backtest(
            tickers=[],
            time_window=args.window,
            weights=weights,
            strategy=args.strategy,
            allowed_sectors=None,
            min_market_cap=args.min_cap,
            max_market_cap=args.max_cap
        )

        metrics = results.get("metrics", {})
        port_ann = metrics.get("portfolio_annualized_return", metrics.get("annualized_return", 0.0))
        spy_ann = metrics.get("spy_annualized_return", metrics.get("benchmark_return", 0.0))
        total_alpha = metrics.get("total_alpha", metrics.get("alpha", 0.0))
        port_sharpe = metrics.get("portfolio_sharpe", metrics.get("sharpe_ratio", 0.0))
        spy_sharpe = metrics.get("spy_sharpe", 0.0)
        port_mdd = metrics.get("portfolio_max_drawdown", metrics.get("max_drawdown", 0.0))
        spy_mdd = metrics.get("spy_max_drawdown", 0.0)

        print("\n--- Backtest Performance Summary ---")
        print(f"Strategy Annualized Return: {port_ann * 100:.2f}%")
        print(f"Benchmark SPY Return:       {spy_ann * 100:.2f}%")
        print(f"Excess Return Alpha:        {total_alpha * 100:.2f}%")
        print(f"Strategy Sharpe Ratio:      {port_sharpe:.2f} (SPY: {spy_sharpe:.2f})")
        print(f"Strategy Max Drawdown:      {port_mdd * 100:.2f}% (SPY: {spy_mdd * 100:.2f}%)")
        print("------------------------------------")
        print("[SUCCESS] Rolling backtest execution completed successfully.")

    except Exception as e:
        print(f"[ERROR] Backtest execution failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
