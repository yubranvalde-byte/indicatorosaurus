#!/usr/bin/env python3
"""
verify_portfolio_api.py
=======================
Checkpoint 2 & Phase 4 Verification Suite for Portfolio Intelligence REST APIs (`api/main.py`).

Validates:
1. Empty Ledger State:
   - GET /api/portfolio/summary returns 200 with clean zero-state JSON and zero NaNs/Infs.
   - GET /api/portfolio/transactions returns empty array [].
   - POST /api/portfolio/analytics returns empty curves array [].
2. Transaction Ledger CRUD & Invalidation:
   - POST /api/portfolio/transactions (actions: add, delete, clear, import_csv).
   - GET /api/portfolio/transactions/export (standard CSV attachment).
   - Cache invalidation upon state mutation.
3. Portfolio Summary Schema & Dual-Currency Valuation:
   - GET /api/portfolio/summary in USD & ILS.
   - Verifies positions, FIFO lots, cash balances, Israeli tax summary, TLH candidates.
4. Vectorized Analytics & Multi-Benchmark Slicing:
   - POST /api/portfolio/analytics across windows ("ALL", "YTD", "3M").
   - Rebased TWR curves, drawdown series, XIRR/MWR, What-If opportunity simulator, ILS attribution.
5. Latency Verification (< 100ms on warm calls):
   - Measures warm-cache latency across summary and analytics endpoints to guarantee sub-100ms SLA.
6. Screener & Engine Regression:
   - POST /api/calculate executes without regression and returns top 100 scored stocks.
7. Strict RFC-8259 JSON Compliance:
   - parse_constant rejection of NaN, Infinity, -Infinity across all responses.
"""

import json
import math
import os
import sys
import tempfile
import time
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "core_logic"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "api"))

from main import app, reset_portfolio_runtime


def _strict_json_parse(response, expected_status=200):
    """Parse JSON with strict RFC-8259 assertion rejecting NaN, Infinity, and -Infinity."""
    assert response.status_code == expected_status, (
        f"[HTTP ERROR] Expected status {expected_status}, got {response.status_code}. "
        f"Response body: {response.data.decode('utf-8')[:500]}"
    )

    def _constant_rejector(constant_str):
        raise ValueError(f"[RFC-8259 VIOLATION] Invalid JSON constant found: '{constant_str}'")

    raw_text = response.data.decode("utf-8")
    return json.loads(raw_text, parse_constant=_constant_rejector)


def run_api_verification():
    print("=== AlphaQuant Phase 4: Portfolio Intelligence API Verification Suite ===")

    tmp_dir = tempfile.TemporaryDirectory()
    test_parquet_path = os.path.join(tmp_dir.name, "test_portfolio.parquet")

    # Configure Flask test client with isolated storage path
    app.config["TESTING"] = True
    app.config["PORTFOLIO_LEDGER_PATH"] = test_parquet_path
    reset_portfolio_runtime()
    client = app.test_client()

    try:
        # ───────────────────────────────────────────────────────────────────────
        # 1. Test Empty Portfolio State Contract
        # ───────────────────────────────────────────────────────────────────────
        print("\n[Step 1] Verifying empty portfolio state contract...")
        res_summary = client.get("/api/portfolio/summary?view_currency=USD")
        data_summary = _strict_json_parse(res_summary, expected_status=200)

        assert data_summary["positions"] == [], "Empty ledger must return empty positions list"
        assert data_summary["lots"] == [], "Empty ledger must return empty lots list"
        assert data_summary["cash"]["USD"] == 0.0, "Empty ledger cash must be 0.0"
        assert data_summary["totals"]["nav"] == 0.0, "Empty ledger NAV must be 0.0"
        assert data_summary["tlh_candidates"] == [], "Empty ledger must return empty TLH list"

        res_txs = client.get("/api/portfolio/transactions")
        data_txs = _strict_json_parse(res_txs, expected_status=200)
        assert data_txs == [], "Empty transactions endpoint must return []"

        res_analytics = client.post("/api/portfolio/analytics", json={"window": "ALL", "view_currency": "ILS"})
        data_analytics = _strict_json_parse(res_analytics, expected_status=200)
        assert data_analytics["curves"] == [], "Empty analytics curves must return []"
        assert data_analytics["attribution"] == {}, "Empty analytics attribution must return {}"
        print(" -> [PASS] Empty portfolio state contract verified.")

        # ───────────────────────────────────────────────────────────────────────
        # 2. Test Transaction Ledger CRUD & Invalidation
        # ───────────────────────────────────────────────────────────────────────
        print("\n[Step 2] Testing Transaction Ledger CRUD & CSV Ingestion...")

        # 2a: Add CASH_DEPOSIT (USD and ILS)
        r1_usd = client.post("/api/portfolio/transactions", json={
            "action": "add",
            "transaction": {
                "date": "2024-01-01",
                "ticker": "CASH",
                "tx_type": "CASH_DEPOSIT",
                "quantity": 100000.0,
                "price": 1.0,
                "fee": 0.0,
                "currency": "USD",
                "notes": "Initial USD deposit"
            }
        })
        d1_usd = _strict_json_parse(r1_usd, expected_status=201)
        assert d1_usd["status"] == "success" and d1_usd["tx_id"] == 1

        r1_ils = client.post("/api/portfolio/transactions", json={
            "action": "add",
            "transaction": {
                "date": "2024-01-01",
                "ticker": "CASH",
                "tx_type": "CASH_DEPOSIT",
                "quantity": 50000.0,
                "price": 1.0,
                "fee": 0.0,
                "currency": "ILS",
                "notes": "Initial ILS deposit"
            }
        })
        d1_ils = _strict_json_parse(r1_ils, expected_status=201)
        assert d1_ils["tx_id"] == 2

        # 2b: Add BUY AAPL (50 shares @ $150, fee $5, USD)
        r2 = client.post("/api/portfolio/transactions", json={
            "action": "add",
            "transaction": {
                "date": "2024-01-05",
                "ticker": "AAPL",
                "tx_type": "BUY",
                "quantity": 50.0,
                "price": 150.0,
                "fee": 5.0,
                "currency": "USD",
                "notes": "Tranche 1 AAPL"
            }
        })
        d2 = _strict_json_parse(r2, expected_status=201)
        assert d2["tx_id"] == 3

        # 2c: Add BUY LUMI.TA (200 shares @ 30.0 ILS, fee 20.0 ILS)
        r3 = client.post("/api/portfolio/transactions", json={
            "action": "add",
            "transaction": {
                "date": "2024-01-10",
                "ticker": "LUMI.TA",
                "tx_type": "BUY",
                "quantity": 200.0,
                "price": 30.0,
                "fee": 20.0,
                "currency": "ILS",
                "notes": "Bank Leumi entry"
            }
        })
        d3 = _strict_json_parse(r3, expected_status=201)
        assert d3["tx_id"] == 4

        # 2d: Add DIVIDEND AAPL ($100 USD)
        r4 = client.post("/api/portfolio/transactions", json={
            "action": "add",
            "transaction": {
                "date": "2024-04-01",
                "ticker": "AAPL",
                "tx_type": "DIVIDEND",
                "quantity": 50.0,
                "price": 2.0,
                "fee": 0.0,
                "currency": "USD",
                "notes": "Q1 AAPL Dividend"
            }
        })
        d4 = _strict_json_parse(r4, expected_status=201)
        assert d4["tx_id"] == 5

        # 2e: Add SELL AAPL (20 shares @ $180, fee $5, USD)
        r5 = client.post("/api/portfolio/transactions", json={
            "action": "add",
            "transaction": {
                "date": "2024-06-01",
                "ticker": "AAPL",
                "tx_type": "SELL",
                "quantity": 20.0,
                "price": 180.0,
                "fee": 5.0,
                "currency": "USD",
                "notes": "Partial profit taking"
            }
        })
        d5 = _strict_json_parse(r5, expected_status=201)
        assert d5["tx_id"] == 6

        # Verify all 6 transactions listed
        all_txs_res = client.get("/api/portfolio/transactions")
        all_txs_data = _strict_json_parse(all_txs_res, expected_status=200)
        assert len(all_txs_data) == 6, f"Expected 6 transactions, got {len(all_txs_data)}"

        # 2f: Test temporary transaction deletion
        rtmp = client.post("/api/portfolio/transactions", json={
            "action": "add",
            "transaction": {
                "date": "2024-06-05",
                "ticker": "SPY",
                "tx_type": "BUY",
                "quantity": 1.0,
                "price": 500.0,
                "fee": 0.0,
                "currency": "USD"
            }
        })
        dtmp = _strict_json_parse(rtmp, expected_status=201)
        del_res = client.post("/api/portfolio/transactions", json={"action": "delete", "tx_id": dtmp["tx_id"]})
        del_data = _strict_json_parse(del_res, expected_status=200)
        assert del_data["status"] == "success" and del_data["deleted"] == dtmp["tx_id"]

        del_404 = client.post("/api/portfolio/transactions", json={"action": "delete", "tx_id": 99999})
        assert del_404.status_code == 404, f"Expected 404 on deleting non-existent tx, got {del_404.status_code}"

        # 2g: Test CSV Export & Round-Trip Re-import
        export_res = client.get("/api/portfolio/transactions/export")
        assert export_res.status_code == 200
        assert export_res.headers.get("Content-Type", "").startswith("text/csv")
        csv_text = export_res.data.decode("utf-8")
        assert "LUMI.TA" in csv_text and "AAPL" in csv_text and "CASH_DEPOSIT" in csv_text

        # Clear ledger
        clear_res = client.post("/api/portfolio/transactions", json={"action": "clear"})
        clear_data = _strict_json_parse(clear_res, expected_status=200)
        assert clear_data["status"] == "success"

        # Verify empty
        tx_check = _strict_json_parse(client.get("/api/portfolio/transactions"), expected_status=200)
        assert len(tx_check) == 0, "Ledger should be empty after clear"

        # Re-import via CSV
        import_res = client.post("/api/portfolio/transactions", json={"action": "import_csv", "csv_content": csv_text})
        import_data = _strict_json_parse(import_res, expected_status=200)
        assert import_data["status"] == "success"
        assert import_data["imported_count"] == 6, f"Expected 6 imported transactions, got {import_data}"

        re_txs = _strict_json_parse(client.get("/api/portfolio/transactions"), expected_status=200)
        assert len(re_txs) == 6
        print(" -> [PASS] Transaction Ledger CRUD & CSV Ingestion verified.")

        # ───────────────────────────────────────────────────────────────────────
        # 3. Test Portfolio Summary Schema & Dual-Currency Valuation
        # ───────────────────────────────────────────────────────────────────────
        print("\n[Step 3] Testing Portfolio Summary Schema & Dual-Currency Valuation...")

        # 3a: USD View
        res_sum_usd = client.get("/api/portfolio/summary?view_currency=USD&as_of=2024-06-15")
        sum_usd = _strict_json_parse(res_sum_usd, expected_status=200)

        assert sum_usd["view_currency"] == "USD"
        assert sum_usd["as_of"] == "2024-06-15"

        positions = sum_usd["positions"]
        pos_by_ticker = {p["ticker"]: p for p in positions}
        assert "AAPL" in pos_by_ticker, "AAPL must be present in open positions"
        assert "LUMI.TA" in pos_by_ticker, "LUMI.TA must be present in open positions"

        # Check AAPL remaining quantity: 50 bought - 20 sold = 30
        aapl_pos = pos_by_ticker["AAPL"]
        assert math.isclose(aapl_pos["quantity"], 30.0, rel_tol=1e-5), f"Expected 30 AAPL, got {aapl_pos['quantity']}"
        assert aapl_pos["currency"] == "USD"
        assert aapl_pos["lots_count"] == 1
        assert aapl_pos["current_price_native"] > 0.0
        assert aapl_pos["market_value_native"] > 0.0
        assert "est_tax_view" in aapl_pos and "unrealized_net_pnl_view" in aapl_pos
        assert math.isclose(
            aapl_pos["unrealized_net_pnl_view"],
            aapl_pos["unrealized_pnl_view"] - aapl_pos["est_tax_view"],
            rel_tol=1e-5,
            abs_tol=1e-5,
        )

        # Check LUMI.TA: 200 shares, native currency ILS
        lumi_pos = pos_by_ticker["LUMI.TA"]
        assert math.isclose(lumi_pos["quantity"], 200.0, rel_tol=1e-5)
        assert lumi_pos["currency"] == "ILS"
        assert lumi_pos["lots_count"] == 1

        # Check weights sum: sum of position weights + cash weight must approximate 100%
        pos_weight_sum = sum(p["weight_pct"] for p in positions)
        cash_weight = (sum_usd["cash"]["total_view"] / sum_usd["totals"]["nav"]) * 100.0
        total_alloc = pos_weight_sum + cash_weight
        assert math.isclose(total_alloc, 100.0, rel_tol=1e-3), f"Total allocation must equal 100%, got {total_alloc:.2f}%"

        # Check Totals & Tax Summary
        totals_usd = sum_usd["totals"]
        assert totals_usd["nav"] > 0.0
        assert "net_liquidation_nav" in totals_usd and "unrealized_net_pnl" in totals_usd
        assert math.isclose(
            totals_usd["net_liquidation_nav"],
            totals_usd["nav"] - totals_usd["est_liquidation_tax_view"],
            rel_tol=1e-5,
            abs_tol=1e-5,
        )
        assert totals_usd["market_value"] > 0.0
        assert totals_usd["usd_ils"] > 0.0

        tax_sum = sum_usd["tax_summary"]
        assert tax_sum["tax_year"] == 2024
        assert tax_sum["real_gains_ils"] > 0.0, "Realized gain on AAPL sell should be positive"
        assert tax_sum["capital_gains_tax_ils"] > 0.0, "Capital gains tax liability should be positive"

        # 3b: ILS View
        res_sum_ils = client.get("/api/portfolio/summary?view_currency=ILS&as_of=2024-06-15")
        sum_ils = _strict_json_parse(res_sum_ils, expected_status=200)

        assert sum_ils["view_currency"] == "ILS"
        fx_rate = sum_ils["totals"]["usd_ils"]
        # NAV in ILS should match NAV in USD * FX rate within rounding
        expected_nav_ils = totals_usd["nav"] * fx_rate
        actual_nav_ils = sum_ils["totals"]["nav"]
        assert math.isclose(actual_nav_ils, expected_nav_ils, rel_tol=1e-3), (
            f"Dual-currency conversion mismatch: USD {totals_usd['nav']} * FX {fx_rate} = {expected_nav_ils}, got {actual_nav_ils}"
        )

        # 3c: Invalid currency guard
        bad_cur_res = client.get("/api/portfolio/summary?view_currency=EUR")
        assert bad_cur_res.status_code == 400
        print(" -> [PASS] Portfolio Summary Schema & Dual-Currency Valuation verified.")

        # ───────────────────────────────────────────────────────────────────────
        # 4. Test Analytics Time-Series & Benchmark Slicing
        # ───────────────────────────────────────────────────────────────────────
        print("\n[Step 4] Testing Vectorized Analytics & Multi-Benchmark Slicing...")

        req_payload = {
            "window": "ALL",
            "view_currency": "ILS",
            "benchmarks": ["SPY"],
            "cash_mode": "auto"
        }
        res_analytics = client.post("/api/portfolio/analytics", json=req_payload)
        data_an = _strict_json_parse(res_analytics, expected_status=200)

        assert data_an["window"] == "ALL"
        assert data_an["view_currency"] == "ILS"
        curves = data_an["curves"]
        assert len(curves) > 0, "Analytics curves must not be empty"

        # Assert curve schema
        sample_pt = curves[0]
        assert "date" in sample_pt, "Curve points must contain 'date'"
        assert "port_twr_rebased_pct" in sample_pt, "Curve must contain 'port_twr_rebased_pct'"
        assert "port_nav" in sample_pt, "Curve must contain 'port_nav'"
        assert "port_drawdown_pct" in sample_pt, "Curve must contain 'port_drawdown_pct'"
        assert "bench_SPY_rebased_pct" in sample_pt, "Curve must contain 'bench_SPY_rebased_pct'"
        assert "what_if_SPY_nav" in sample_pt, "Curve must contain 'what_if_SPY_nav'"

        # Verify base anchor at t0: rebased curves must start at 0.0%
        assert math.isclose(curves[0]["port_twr_rebased_pct"], 0.0, abs_tol=1e-5), "TWR rebase anchor must be 0.0%"
        assert math.isclose(curves[0]["bench_SPY_rebased_pct"], 0.0, abs_tol=1e-5), "Benchmark rebase anchor must be 0.0%"

        # Verify institutional KPIs
        kpis = data_an["kpis"]
        for required_kpi in ("twr_total_pct", "twr_annualized_pct", "xirr_annualized_pct",
                             "annualized_volatility_pct", "sharpe_ratio", "max_drawdown_pct"):
            assert required_kpi in kpis, f"Missing KPI: {required_kpi}"
            val = kpis[required_kpi]
            assert isinstance(val, (int, float)) and not math.isnan(val) and not math.isinf(val), (
                f"KPI {required_kpi} must be a finite float, got {val}"
            )

        # Verify ILS Attribution Breakdown
        attrib = data_an["attribution"]
        for attr_k in ("gross_ils_twr_pct", "pure_asset_return_pct", "fx_impact_pct", "fees_ils", "realized_tax_ils"):
            assert attr_k in attrib, f"Missing attribution metric: {attr_k}"
            assert isinstance(attrib[attr_k], (int, float))

        # Verify What-If simulator
        what_if = data_an["what_if"]
        assert "SPY" in what_if, "What-If simulator must include SPY"
        spy_sim = what_if["SPY"]
        assert spy_sim["terminal_benchmark_nav"] > 0.0
        assert spy_sim["terminal_portfolio_nav"] > 0.0
        assert math.isclose(spy_sim["excess_value_created"],
                            spy_sim["terminal_portfolio_nav"] - spy_sim["terminal_benchmark_nav"], rel_tol=1e-4)

        # Slicing test: 3M window
        res_3m = client.post("/api/portfolio/analytics", json={"window": "3M", "view_currency": "USD", "benchmarks": ["SPY"]})
        data_3m = _strict_json_parse(res_3m, expected_status=200)
        assert data_3m["window"] == "3M"
        assert len(data_3m["curves"]) > 0
        print(" -> [PASS] Vectorized Analytics & Multi-Benchmark Slicing verified.")

        # ───────────────────────────────────────────────────────────────────────
        # 5. Latency Verification (< 100ms on warm calls)
        # ───────────────────────────────────────────────────────────────────────
        print("\n[Step 5] Benchmarking Warm-Cache API Latency SLA (< 100ms)...")

        # Prime the caches
        client.post("/api/portfolio/analytics", json={"window": "ALL", "view_currency": "ILS", "benchmarks": ["SPY"]})
        client.get("/api/portfolio/summary?view_currency=USD")

        # Benchmark 10 consecutive warm analytics requests
        analytics_latencies = []
        for _ in range(10):
            t_start = time.perf_counter()
            r = client.post("/api/portfolio/analytics", json={"window": "ALL", "view_currency": "ILS", "benchmarks": ["SPY"]})
            assert r.status_code == 200
            elapsed_ms = (time.perf_counter() - t_start) * 1000.0
            analytics_latencies.append(elapsed_ms)

        avg_an_latency = np.mean(analytics_latencies)
        max_an_latency = np.max(analytics_latencies)
        print(f" -> POST /api/portfolio/analytics warm latency: avg={avg_an_latency:.2f}ms, max={max_an_latency:.2f}ms")
        assert avg_an_latency < 100.0, f"[SLA BREACH] Warm analytics latency {avg_an_latency:.2f}ms >= 100ms"

        # Benchmark 10 consecutive warm summary requests
        summary_latencies = []
        for _ in range(10):
            t_start = time.perf_counter()
            r = client.get("/api/portfolio/summary?view_currency=USD")
            assert r.status_code == 200
            elapsed_ms = (time.perf_counter() - t_start) * 1000.0
            summary_latencies.append(elapsed_ms)

        avg_sum_latency = np.mean(summary_latencies)
        max_sum_latency = np.max(summary_latencies)
        print(f" -> GET /api/portfolio/summary warm latency: avg={avg_sum_latency:.2f}ms, max={max_sum_latency:.2f}ms")
        assert avg_sum_latency < 100.0, f"[SLA BREACH] Warm summary latency {avg_sum_latency:.2f}ms >= 100ms"
        print(" -> [PASS] Warm latency SLA passed (< 100ms).")

        # ───────────────────────────────────────────────────────────────────────
        # 6. Screener & Engine Regression Check
        # ───────────────────────────────────────────────────────────────────────
        print("\n[Step 6] Running Screener Engine (/api/calculate) Regression Check...")
        screener_payload = {"weights": [1.0 / 16.0] * 16}
        res_screener = client.post("/api/calculate", json=screener_payload)
        data_screener = _strict_json_parse(res_screener, expected_status=200)

        results = data_screener.get("results", [])
        assert len(results) > 0, "Screener results must not be empty"
        assert len(results) <= 100, f"Screener must return at most 100 results, got {len(results)}"

        top_stock = results[0]
        assert "ticker" in top_stock and "score" in top_stock and "indicators" in top_stock
        assert len(top_stock["indicators"]) == 16, f"Expected 16 indicator scores, got {len(top_stock['indicators'])}"
        print(f" -> [PASS] Screener regression passed: scored {len(results)} stocks; Top ticker={top_stock['ticker']}, Score={top_stock['score']}.")

        # ───────────────────────────────────────────────────────────────────────
        # 7. Phase 5 & Phase 6 Frontend Workspace HTML & Component Contract Check
        # ───────────────────────────────────────────────────────────────────────
        print("\n[Step 7] Verifying Phase 5 & Phase 6 Frontend Portfolio Workspace (GET /)...")
        res_html = client.get("/")
        assert res_html.status_code == 200, f"Expected HTTP 200 from GET /, got {res_html.status_code}"
        html_body = res_html.data.decode("utf-8")
        for token in (
            "workspace-screener",
            "workspace-portfolio",
            "portfolio-main-chart",
            "portfolio-dna-radar-chart",
            "portfolio-sector-chart",
            "portfolio-tlh-body",
            "portfolio-holdings-body",
            "portfolio-tx-body",
            "switchWorkspace",
            "setPortfolioCurrency",
            "portfolioChartInstance.destroy()",
            "portfolioRadarChartInstance.destroy()",
            "portfolioSectorChartInstance.destroy()",
        ):
            assert token in html_body, f"Missing required Phase 5/6 UI token in index.html: {token}"
        print(" -> [PASS] Phase 5 & Phase 6 Frontend Portfolio Workspace HTML & controller verified.")

        # ───────────────────────────────────────────────────────────────────────
        # 8. Phase 6 Quant Factor DNA (C-FFI), Sector Attribution & TLH Simulator
        # ───────────────────────────────────────────────────────────────────────
        print("\n[Step 8] Testing Phase 6 Quant Factor DNA, Sector Attribution & TLH Simulator...")
        # Add MSFT (high cost lot to trigger a real capital loss for TLH) and SPY (ETF, non-SimFin)
        r_msft = client.post("/api/portfolio/transactions", json={
            "action": "add",
            "transaction": {
                "date": "2024-03-01",
                "ticker": "MSFT",
                "tx_type": "BUY",
                "quantity": 10.0,
                "price": 600.0,
                "fee": 5.0,
                "currency": "USD",
                "notes": "High entry lot for TLH verification"
            }
        })
        _strict_json_parse(r_msft, expected_status=201)

        r_spy = client.post("/api/portfolio/transactions", json={
            "action": "add",
            "transaction": {
                "date": "2024-03-05",
                "ticker": "SPY",
                "tx_type": "BUY",
                "quantity": 10.0,
                "price": 500.0,
                "fee": 2.0,
                "currency": "USD",
                "notes": "Benchmark ETF holding"
            }
        })
        _strict_json_parse(r_spy, expected_status=201)

        res_dna = client.get("/api/portfolio/quant-dna?view_currency=ILS&as_of=2024-06-15")
        dna = _strict_json_parse(res_dna, expected_status=200)

        assert 0.0 < dna["portfolio_quant_score"] <= 100.0, f"Invalid portfolio_quant_score: {dna['portfolio_quant_score']}"
        assert 0.0 < dna["benchmark_quant_score"] <= 100.0
        assert dna["covered_holdings_count"] == 2, f"Expected 2 covered holdings (AAPL, MSFT), got {dna['covered_holdings_count']}"
        assert dna["total_holdings_count"] == 4, f"Expected 4 total holdings (AAPL, MSFT, SPY, LUMI.TA), got {dna['total_holdings_count']}"

        # Verify factor_coverage_pct equals exact share of covered MV over total MV
        holdings_by_sym = {h["ticker"]: h for h in dna["holdings_dna"]}
        assert holdings_by_sym["AAPL"]["is_covered"] is True and 0.0 < holdings_by_sym["AAPL"]["quant_score"] <= 100.0
        assert holdings_by_sym["MSFT"]["is_covered"] is True and 0.0 < holdings_by_sym["MSFT"]["quant_score"] <= 100.0
        assert holdings_by_sym["SPY"]["is_covered"] is False and holdings_by_sym["SPY"]["quant_status"] == "ETF_TASE"
        assert holdings_by_sym["LUMI.TA"]["is_covered"] is False and holdings_by_sym["LUMI.TA"]["quant_status"] == "ETF_TASE"

        cov_mv = holdings_by_sym["AAPL"]["market_value_view"] + holdings_by_sym["MSFT"]["market_value_view"]
        tot_mv = sum(h["market_value_view"] for h in dna["holdings_dna"])
        expected_cov_pct = (cov_mv / tot_mv) * 100.0
        assert math.isclose(dna["factor_coverage_pct"], expected_cov_pct, abs_tol=0.05), (
            f"factor_coverage_pct mismatch: expected {expected_cov_pct:.2f}%, got {dna['factor_coverage_pct']:.2f}%"
        )

        # Verify 5-Pillar Radar scores in [0, 100]
        for pillar in ("Valuation", "Growth", "Quality", "Safety & Income", "Momentum"):
            assert pillar in dna["pillars"], f"Missing pillar {pillar}"
            assert 0.0 <= dna["pillars"][pillar] <= 100.0
            assert 0.0 <= dna["benchmark_pillars"][pillar] <= 100.0

        # Verify GICS sector weights sum to 100.0%
        sectors = dna["sector_attribution"]
        assert len(sectors) >= 2
        sector_weight_sum = sum(s["weight_pct"] for s in sectors)
        assert math.isclose(sector_weight_sum, 100.0, abs_tol=1e-2), f"Sector weights must sum to 100%, got {sector_weight_sum}"
        sec_by_name = {s["sector"]: s for s in sectors}
        assert "Unclassified / ETF / TASE" in sec_by_name
        assert set(sec_by_name["Unclassified / ETF / TASE"]["tickers"]) == {"SPY", "LUMI.TA"}

        # Verify TLH Simulator candidates on summary endpoint
        res_sum_p6 = client.get("/api/portfolio/summary?view_currency=ILS&as_of=2024-06-15")
        sum_p6 = _strict_json_parse(res_sum_p6, expected_status=200)
        tlh_cands = sum_p6["tlh_candidates"]
        assert len(tlh_cands) >= 1, "Expected at least 1 TLH candidate from high-cost MSFT lot"
        msft_tlh = next((c for c in tlh_cands if c["ticker"] == "MSFT"), None)
        assert msft_tlh is not None and msft_tlh["real_loss_ils"] < 0.0
        assert math.isclose(msft_tlh["immediate_tax_shield_ils"], round(0.25 * abs(msft_tlh["real_loss_ils"]), 2), abs_tol=0.02)

        # Verify warm-cache latency on GET /api/portfolio/quant-dna (< 50ms)
        dna_latencies = []
        for _ in range(10):
            t0_dna = time.perf_counter()
            r_dna = client.get("/api/portfolio/quant-dna?view_currency=ILS&as_of=2024-06-15")
            assert r_dna.status_code == 200
            dna_latencies.append((time.perf_counter() - t0_dna) * 1000.0)
        avg_dna_ms = float(np.mean(dna_latencies))
        print(f" -> GET /api/portfolio/quant-dna warm latency: avg={avg_dna_ms:.2f}ms")
        assert avg_dna_ms < 50.0, f"[SLA BREACH] Warm Quant DNA latency {avg_dna_ms:.2f}ms >= 50ms"
        print(" -> [PASS] Phase 6 Quant Factor DNA, Sector Attribution & TLH Simulator verified.")

        print("\n---------------------------------------------------------------------------------")
        print("[SUCCESS] All Phase 4, Phase 5, Phase 6 & Checkpoint 2 assertions passed!")
        print("---------------------------------------------------------------------------------")

    finally:
        tmp_dir.cleanup()


if __name__ == "__main__":
    run_api_verification()
