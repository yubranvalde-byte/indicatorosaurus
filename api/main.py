from flask import Flask, request, jsonify, render_template, Response
import csv
import os
import gc
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'core_logic')))

import pandas as pd
import numpy as np

from financial_module import (
    run_financial_engine, normalize_stocks, apply_early_pruning_vectorized, apply_derived_knockouts_vectorized,
    compute_live_technical_indicators, ensure_tech_cache, INDICATOR_COLUMNS, VALID_STRATEGIES, _safe_float,
    _scalar_float, load_latest_fundamental_snapshot, get_cached_fundamental_snapshot,
    fetch_tiingo_live_data, sanitize_for_json
)
from backtest_engine import run_backtest
from portfolio_ledger import PortfolioLedger, DEFAULT_LEDGER_PATH
from portfolio_analytics import PortfolioAnalytics

app = Flask(__name__, template_folder='../frontend/templates', static_folder='../frontend/static')

PRIMARY_CSV = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "historical_fundamental_data.csv"))
FALLBACK_CSV = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "market_data.csv"))
CSV_PATH = PRIMARY_CSV if os.path.exists(PRIMARY_CSV) else FALLBACK_CSV

# Optional Startup Cache Warmup gated strictly behind WARM_CACHE_ON_STARTUP env var
if os.getenv('WARM_CACHE_ON_STARTUP', '').strip().lower() in ('1', 'true', 'yes'):
    print("[Startup Warmup] WARM_CACHE_ON_STARTUP set. Hydrating technical data cache on boot...")
    try:
        ensure_tech_cache()
    except Exception as _e:
        print(f"[Startup Warmup] Warning: Boot cache hydration failed: {_e}")


def _load_csv():
    """Read fundamental dataset from In-Memory TTL Cache for sub-second screener C-FFI execution."""
    return get_cached_fundamental_snapshot(CSV_PATH)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/calculate', methods=['POST'])
def calculate():
    """Endpoint A: Handles immediate C Engine stock scoring and returns ranked stock list."""
    payload = request.get_json()
    if not payload:
        return jsonify({"error": "No JSON payload provided"}), 400

    # ── Extract the 16-element weights array ─────────────────────────────────
    raw_weights = payload.get('weights', [])
    if len(raw_weights) != 16:
        return jsonify({"error": f"weights must be a 16-element array, got {len(raw_weights)}"}), 400

    weights_list = [float(w) for w in raw_weights]

    # ── Extract and validate strategy string ───────────────────────────────
    strategy = payload.get('strategy', None)
    if strategy is not None and strategy not in VALID_STRATEGIES:
        return jsonify({"error": f"Unknown strategy: '{strategy}'. "
                                 f"Valid values: {sorted(VALID_STRATEGIES)}"}), 400

    # ── Extract and validate dynamic filtering parameters ───────────────────
    specific_tickers = payload.get('specific_tickers', None)
    allowed_sectors = payload.get('allowed_sectors', None)
    min_market_cap = payload.get('min_market_cap', None)
    max_market_cap = payload.get('max_market_cap', None)

    if specific_tickers is not None:
        if not isinstance(specific_tickers, list) or not all(isinstance(t, str) for t in specific_tickers):
            return jsonify({"error": "specific_tickers must be a list of strings"}), 400

    if allowed_sectors is not None:
        if not isinstance(allowed_sectors, list) or not all(isinstance(s, str) for s in allowed_sectors):
            return jsonify({"error": "allowed_sectors must be a list of strings"}), 400

    if min_market_cap is not None:
        try:
            min_market_cap = float(min_market_cap)
        except (ValueError, TypeError):
            return jsonify({"error": "min_market_cap must be a numeric value"}), 400

    if max_market_cap is not None:
        try:
            max_market_cap = float(max_market_cap)
        except (ValueError, TypeError):
            return jsonify({"error": "max_market_cap must be a numeric value"}), 400

    if min_market_cap is not None and max_market_cap is not None:
        if min_market_cap > max_market_cap:
            return jsonify({"error": "min_market_cap cannot be greater than max_market_cap"}), 400

    # ── Load CSV baseline dataset ─────────────────────────────────────────────
    stocks_df, err = _load_csv()
    if err:
        return jsonify({"error": err}), (404 if "not found" in err else 500)

    total_csv_count = len(stocks_df)

    # ── Explicit Pre-Flight Cache Management ──────────────────────────────────
    ensure_tech_cache()

    # ── Step 1: Pre-Filtering — Apply static filters FIRST to reduce universe ─
    stocks_df = apply_early_pruning_vectorized(stocks_df, strategy, allowed_sectors, min_market_cap, max_market_cap, specific_tickers)

    if stocks_df.empty:
        return jsonify({
            "strategy": strategy,
            "stocks_before": total_csv_count,
            "stocks_after": 0,
            "filters_applied": {
                "specific_tickers": specific_tickers,
                "allowed_sectors": allowed_sectors,
                "min_market_cap": min_market_cap,
                "max_market_cap": max_market_cap,
            },
            "results": [],
        })

    try:
        # ── Step 2: Strategy-Conditional Technical Indicator Computation ─────
        surviving_tickers = stocks_df['Ticker'].tolist()
        tech_map = compute_live_technical_indicators(surviving_tickers)

        if tech_map:
            tech_df = pd.DataFrame.from_dict(tech_map, orient='index')
            tech_df.index.name = 'Ticker'
            
            stocks_df.set_index('Ticker', inplace=True)
            for col in ['RelativeStrength', 'PriceVs200SMA', 'RSI', 'Beta']:
                if col in tech_df.columns:
                    stocks_df[col] = tech_df[col]
            stocks_df.reset_index(inplace=True)
            
            del tech_map
            del tech_df
            gc.collect()

        # ── Step 3: Derived Knockouts ────────────────────────────────────────
        stocks_df = apply_derived_knockouts_vectorized(stocks_df, strategy)

        if stocks_df.empty:
            return jsonify({
                "strategy": strategy,
                "stocks_before": total_csv_count,
                "stocks_after": 0,
                "filters_applied": {
                    "specific_tickers": specific_tickers,
                    "allowed_sectors": allowed_sectors,
                    "min_market_cap": min_market_cap,
                    "max_market_cap": max_market_cap,
                },
                "results": [],
            })

        # Build metadata lookup for surviving universe
        meta_by_ticker = {
            s['Ticker']: {
                'longName': s.get('LongName', ''),
                'sector': s.get('Sector', ''),
                'marketCap': _safe_float(s.get('MarketCap', 0.0)),
            }
            for s in stocks_df.to_dict(orient='records')
        }

        # ── Type Serialization & NaN Safety ──────────────────────────────────
        stocks_df = stocks_df.fillna(0.0).replace([np.inf, -np.inf], 0.0)
        
        for col in stocks_df.select_dtypes(include=['float32', 'float64']).columns:
            stocks_df[col] = stocks_df[col].astype(float)
        for col in stocks_df.select_dtypes(include=['int32', 'int64']).columns:
            stocks_df[col] = stocks_df[col].astype(int)

        stocks_data = stocks_df.to_dict(orient='records')

        # ── Step 4: Normalization (Python) ────────────────────────────────────
        normalized_stocks = normalize_stocks(
            stocks_data,
            strategy=None,               # Already applied conditionally above
            allowed_sectors=None,       # Already filtered in Step 1
            min_market_cap=None,        # Already filtered in Step 1
            max_market_cap=None,        # Already filtered in Step 1
            global_stocks_list=stocks_data
        )

        if not normalized_stocks:
            return jsonify({
                "strategy": strategy,
                "stocks_before": total_csv_count,
                "stocks_after": 0,
                "filters_applied": {
                    "specific_tickers": specific_tickers,
                    "allowed_sectors": allowed_sectors,
                    "min_market_cap": min_market_cap,
                    "max_market_cap": max_market_cap,
                },
                "results": [],
            })

        # Build O(1) lookup tables keyed by Ticker from the normalized universe
        filtered_tickers = {s['Ticker'] for s in normalized_stocks}
        raw_by_ticker    = {s["Ticker"]: s for s in stocks_data if s["Ticker"] in filtered_tickers}
        norm_by_ticker   = {s["Ticker"]: s for s in normalized_stocks}

        # ── Step 4: Run C Scoring Engine ───────────────────────────────────────
        sorted_stocks = run_financial_engine(normalized_stocks, weights_list)

        # ── Step 5: Strict Hard Truncation to Top 100 Stocks ────────────────────
        total_survivors_count = len(sorted_stocks)
        sorted_stocks = sorted_stocks[:100]

        # ── Step 6: Selective Tiingo Live Price Enrichment (Top 100 max) ───────
        tiingo_key = os.getenv('TIINGO_API_KEY') or os.getenv('API_KEY')
        if tiingo_key and sorted_stocks:
            top_tickers = [s['Ticker'] for s in sorted_stocks]
            tiingo_prices = fetch_tiingo_live_data(tiingo_key, top_tickers, batch_size=50)
            if tiingo_prices:
                for ticker, quote in tiingo_prices.items():
                    if ticker in raw_by_ticker and quote.get('price', 0.0) > 0:
                        raw_by_ticker[ticker]['Price'] = quote['price']

        # ── Step 7: Build enriched JSON response for Top 100 ──────────────────
        frontend_response = []
        for s in sorted_stocks:
            ticker     = s["Ticker"]
            raw_stock  = raw_by_ticker.get(ticker, {})
            norm_stock = norm_by_ticker.get(ticker, {})

            indicators = {}
            for col in INDICATOR_COLUMNS:
                try:
                    raw_val = float(raw_stock.get(col, 0.0) or 0.0)
                    if raw_val != raw_val:
                        raw_val = 0.0
                except (ValueError, TypeError):
                    raw_val = 0.0

                try:
                    norm_score = float(norm_stock.get(col, 0.0) or 0.0)
                except (ValueError, TypeError):
                    norm_score = 0.0

                indicators[col] = {
                    "raw":   round(raw_val,   6),
                    "score": round(norm_score, 1),
                }

            frontend_response.append({
                "ticker":     ticker,
                "longName":   meta_by_ticker.get(ticker, {}).get("longName", ""),
                "sector":     meta_by_ticker.get(ticker, {}).get("sector", ""),
                "marketCap":  meta_by_ticker.get(ticker, {}).get("marketCap", 0.0),
                "score":      f"{s['FinalScore']:.2f}",
                "indicators": indicators,
            })

        res_json = jsonify({
            "strategy":      strategy,
            "stocks_before": total_csv_count,
            "stocks_after":  total_survivors_count,
            "filters_applied": {
                "specific_tickers": specific_tickers,
                "allowed_sectors":  allowed_sectors,
                "min_market_cap":   min_market_cap,
                "max_market_cap":   max_market_cap,
            },
            "results":       frontend_response,
        })
        del stocks_data, sorted_stocks, normalized_stocks, raw_by_ticker, norm_by_ticker, frontend_response
        gc.collect()
        return res_json

    except Exception as e:
        return jsonify({"error": f"Engine execution failed: {str(e)}"}), 500


@app.route('/api/backtest', methods=['POST'])
def backtest():
    """Endpoint B: Triggers asynchronous trailing backtest simulation for portfolio tickers."""
    payload = request.get_json()
    if not payload:
        return jsonify({"error": "No JSON payload provided"}), 400

    tickers = payload.get('tickers', [])
    if not tickers or not isinstance(tickers, list):
        return jsonify({"error": "tickers parameter must be a non-empty list of strings"}), 400

    time_window = payload.get('time_window', '3y')
    if time_window not in ('3y', '5y'):
        return jsonify({"error": "time_window parameter must be either '3y' or '5y'"}), 400

    weights = payload.get('weights', None)
    strategy = payload.get('strategy', None)
    allowed_sectors = payload.get('allowed_sectors', None)
    min_market_cap = payload.get('min_market_cap', None)
    max_market_cap = payload.get('max_market_cap', None)

    try:
        results = run_backtest(
            tickers,
            time_window,
            weights=weights,
            strategy=strategy,
            allowed_sectors=allowed_sectors,
            min_market_cap=min_market_cap,
            max_market_cap=max_market_cap
        )
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": f"Backtest failed: {str(e)}"}), 500


# ─── Portfolio Intelligence Engine Runtime & REST Endpoints ──────────────────

_portfolio_instances = {}


def get_portfolio_runtime():
    """Retrieve or initialize singleton PortfolioLedger & PortfolioAnalytics per ledger path."""
    ledger_path = app.config.get("PORTFOLIO_LEDGER_PATH", DEFAULT_LEDGER_PATH)
    if ledger_path not in _portfolio_instances:
        ledger = PortfolioLedger(path=ledger_path)
        analytics = PortfolioAnalytics(ledger=ledger)
        _portfolio_instances[ledger_path] = (ledger, analytics)
    return _portfolio_instances[ledger_path]


def reset_portfolio_runtime():
    """Clear memoized portfolio instances (used across test suites or runtime resets)."""
    _portfolio_instances.clear()


@app.route('/api/portfolio/summary', methods=['GET'])
def portfolio_summary():
    """
    Endpoint C: Returns current open positions, FIFO tax-lot breakdown, cash balances,
    realized PnL, YTD Israeli tax summary, and Tax-Loss Harvesting (TLH) candidates.
    """
    ledger, _ = get_portfolio_runtime()
    as_of = request.args.get('as_of', None)
    view_currency = request.args.get('view_currency', 'USD').upper().strip()
    if view_currency not in ('USD', 'ILS'):
        return jsonify({"error": "view_currency must be either 'USD' or 'ILS'"}), 400

    tx_df = ledger.transactions()
    if tx_df.empty:
        empty_payload = {
            "as_of": as_of or pd.Timestamp.now().strftime('%Y-%m-%d'),
            "view_currency": view_currency,
            "positions": [],
            "lots": [],
            "cash": {"USD": 0.0, "ILS": 0.0, "total_view": 0.0},
            "totals": {
                "nav": 0.0,
                "net_liquidation_nav": 0.0,
                "market_value": 0.0,
                "cost_basis": 0.0,
                "unrealized_pnl": 0.0,
                "unrealized_net_pnl": 0.0,
                "unrealized_pnl_pct": 0.0,
                "realized_pnl_ytd": 0.0,
                "est_liquidation_tax_ils": 0.0,
                "est_liquidation_tax_view": 0.0,
                "unused_loss_shield_ils": 0.0,
                "usd_ils": 1.0,
            },
            "tax_summary": {},
            "tlh_candidates": [],
        }
        return jsonify(sanitize_for_json(empty_payload))

    try:
        snap = ledger.snapshot(as_of=as_of)
    except Exception as e:
        return jsonify({"error": f"Failed to compute portfolio snapshot: {str(e)}"}), 500

    snap_as_of = snap["as_of"]
    as_of_dt = pd.Timestamp(snap_as_of)
    totals_raw = snap["totals"]
    fx_rate = _scalar_float(totals_raw.get("fx_usd_ils", 1.0), default=1.0)
    if fx_rate <= 0.0:
        fx_rate = 1.0

    # Cash balances
    cash_raw = totals_raw.get("cash", {})
    cash_usd = _scalar_float(cash_raw.get("USD", 0.0))
    cash_ils = _scalar_float(cash_raw.get("ILS", 0.0))
    if view_currency == "ILS":
        total_cash_view = cash_ils + (cash_usd * fx_rate)
    else:
        total_cash_view = cash_usd + (cash_ils / fx_rate)

    cash_dict = {
        "USD": cash_usd,
        "ILS": cash_ils,
        "total_view": total_cash_view,
    }

    # Positions breakdown
    pos_df = snap["positions"]
    positions_list = []

    if not pos_df.empty:
        mv_view_col = "market_value_ils" if view_currency == "ILS" else "market_value_usd"
        cost_view_col = "cost_ils" if view_currency == "ILS" else "cost_usd"
        total_mv_view = _scalar_float(pos_df[mv_view_col].sum())
        total_nav_view = total_mv_view + total_cash_view

        for _, row in pos_df.iterrows():
            ticker = str(row["ticker"])
            currency = str(row["currency"])
            qty = _scalar_float(row["quantity"])
            cost_native = _scalar_float(row["cost_native"])
            avg_cost_native = _scalar_float(row.get("avg_cost_native", cost_native / qty if qty > 0 else 0.0))
            cur_price_native = _scalar_float(row["market_price"])
            mv_native = _scalar_float(row["market_value_native"])
            mv_view = _scalar_float(row[mv_view_col])
            cost_view = _scalar_float(row[cost_view_col])
            unrealized_pnl_native = mv_native - cost_native
            unrealized_pnl_view = mv_view - cost_view
            unrealized_pnl_pct = (unrealized_pnl_view / cost_view * 100.0) if cost_view > 1e-9 else 0.0
            weight_pct = (mv_view / total_nav_view * 100.0) if total_nav_view > 1e-9 else 0.0

            est_tax_ils = _scalar_float(row.get("est_tax_ils", 0.0))
            est_tax_view = est_tax_ils if view_currency == "ILS" else (est_tax_ils / fx_rate)
            unrealized_net_pnl_view = unrealized_pnl_view - est_tax_view
            inflationary_gain_ils = _scalar_float(row.get("unrealized_inflationary_ils", 0.0))
            real_gain_ils = _scalar_float(row.get("unrealized_real_gain_ils", 0.0))
            real_loss_ils = _scalar_float(row.get("unrealized_real_loss_ils", 0.0))

            lots_for_ticker = snap["lots"][snap["lots"]["ticker"] == ticker] if not snap["lots"].empty else pd.DataFrame()
            lots_count = len(lots_for_ticker)

            positions_list.append({
                "ticker": ticker,
                "currency": currency,
                "quantity": qty,
                "avg_cost_native": avg_cost_native,
                "current_price_native": cur_price_native,
                "market_value_native": mv_native,
                "market_value_view": mv_view,
                "cost_basis_view": cost_view,
                "unrealized_pnl_native": unrealized_pnl_native,
                "unrealized_pnl_view": unrealized_pnl_view,
                "unrealized_pnl_pct": unrealized_pnl_pct,
                "est_tax_ils": est_tax_ils,
                "est_tax_view": est_tax_view,
                "unrealized_net_pnl_view": unrealized_net_pnl_view,
                "inflationary_gain_ils": inflationary_gain_ils,
                "real_gain_ils": real_gain_ils,
                "real_loss_ils": real_loss_ils,
                "weight_pct": weight_pct,
                "lots_count": lots_count,
            })
    else:
        total_mv_view = 0.0
        total_nav_view = total_cash_view

    # Aggregated totals
    if view_currency == "ILS":
        cost_basis_view = _scalar_float(totals_raw.get("cost_ils", 0.0))
        mv_view = _scalar_float(totals_raw.get("market_value_ils", 0.0))
    else:
        cost_basis_view = _scalar_float(totals_raw.get("cost_ils", 0.0)) / fx_rate
        mv_view = _scalar_float(totals_raw.get("market_value_usd", 0.0))

    unrealized_pnl_view = mv_view - cost_basis_view
    unrealized_pnl_pct = (unrealized_pnl_view / cost_basis_view * 100.0) if cost_basis_view > 1e-9 else 0.0

    # Israeli tax summary YTD
    tax_df = ledger.tax_summary(as_of_dt.year)
    tax_summary_dict = {}
    realized_pnl_ytd = 0.0
    if not tax_df.empty:
        tax_row = tax_df.iloc[0].to_dict()
        tax_summary_dict = {
            "tax_year": int(tax_row.get("tax_year", as_of_dt.year)),
            "real_gains_ils": _scalar_float(tax_row.get("real_gains_ils", 0.0)),
            "real_losses_ils": _scalar_float(tax_row.get("real_losses_ils", 0.0)),
            "inflationary_exempt_ils": _scalar_float(tax_row.get("inflationary_exempt_ils", 0.0)),
            "taxable_real_gain_ils": _scalar_float(tax_row.get("taxable_real_gain_ils", 0.0)),
            "capital_gains_tax_ils": _scalar_float(tax_row.get("capital_gains_tax_ils", 0.0)),
            "loss_carry_in_ils": _scalar_float(tax_row.get("loss_carry_in_ils", 0.0)),
            "loss_carry_out_ils": _scalar_float(tax_row.get("loss_carry_out_ils", 0.0)),
            "shield_used_current_year_ils": _scalar_float(tax_row.get("shield_used_current_year_ils", 0.0)),
            "shield_used_carry_forward_ils": _scalar_float(tax_row.get("shield_used_carry_forward_ils", 0.0)),
            "dividend_gross_ils": _scalar_float(tax_row.get("dividend_gross_ils", 0.0)),
            "dividend_withholding_ils": _scalar_float(tax_row.get("dividend_withholding_ils", 0.0)),
        }
        net_realized_ils = tax_summary_dict["real_gains_ils"] - tax_summary_dict["real_losses_ils"]
        realized_pnl_ytd = net_realized_ils if view_currency == "ILS" else (net_realized_ils / fx_rate)

    est_liq_tax_ils = _scalar_float(totals_raw.get("est_liquidation_tax_ils", 0.0))
    est_liq_tax_view = est_liq_tax_ils if view_currency == "ILS" else (est_liq_tax_ils / fx_rate)

    totals_dict = {
        "nav": total_nav_view,
        "net_liquidation_nav": total_nav_view - est_liq_tax_view,
        "market_value": mv_view,
        "cost_basis": cost_basis_view,
        "unrealized_pnl": unrealized_pnl_view,
        "unrealized_net_pnl": unrealized_pnl_view - est_liq_tax_view,
        "unrealized_pnl_pct": unrealized_pnl_pct,
        "realized_pnl_ytd": realized_pnl_ytd,
        "est_liquidation_tax_ils": est_liq_tax_ils,
        "est_liquidation_tax_view": est_liq_tax_view,
        "unused_loss_shield_ils": _scalar_float(totals_raw.get("unused_loss_shield_ils", 0.0)),
        "usd_ils": fx_rate,
    }

    response_payload = {
        "as_of": snap_as_of,
        "view_currency": view_currency,
        "positions": positions_list,
        "lots": sanitize_for_json(snap["lots"]),
        "cash": cash_dict,
        "totals": totals_dict,
        "tax_summary": tax_summary_dict,
        "tlh_candidates": sanitize_for_json(snap["tlh_candidates"]),
    }
    return jsonify(sanitize_for_json(response_payload))


@app.route('/api/portfolio/transactions', methods=['GET', 'POST'])
def portfolio_transactions():
    """
    Endpoint D: Transaction Ledger CRUD & Bulk CSV Ingestion.
    Supports GET (list transactions) and POST (add, delete, clear, import_csv).
    """
    ledger, analytics = get_portfolio_runtime()

    if request.method == 'GET':
        tx_df = ledger.transactions()
        return jsonify(sanitize_for_json(tx_df))

    payload = request.get_json()
    if not payload:
        return jsonify({"error": "No JSON payload provided"}), 400

    action = str(payload.get("action", "add")).lower().strip()

    if action == "add":
        tx = payload.get("transaction", payload)
        date = tx.get("date")
        ticker = tx.get("ticker")
        tx_type = tx.get("tx_type")
        quantity = tx.get("quantity")
        price = tx.get("price")
        fee = tx.get("fee", 0.0)
        currency = tx.get("currency", "USD")
        notes = tx.get("notes", "")

        if not date or not ticker or not tx_type:
            return jsonify({"error": "Missing required fields: date, ticker, tx_type"}), 400

        try:
            quantity = float(quantity)
            price = float(price)
            fee = float(fee or 0.0)
        except (ValueError, TypeError):
            return jsonify({"error": "quantity, price, and fee must be numeric"}), 400

        try:
            tx_id = ledger.add_transaction(
                date=str(date),
                ticker=str(ticker),
                tx_type=str(tx_type),
                quantity=quantity,
                price=price,
                fee=fee,
                currency=str(currency),
                notes=str(notes or ""),
            )
            analytics.invalidate_cache()
            return jsonify({
                "status": "success",
                "tx_id": int(tx_id),
                "message": f"Transaction {tx_id} added successfully",
            }), 201
        except Exception as e:
            return jsonify({"error": f"Failed to add transaction: {str(e)}"}), 400

    elif action == "delete":
        tx_id = payload.get("tx_id")
        if tx_id is None:
            return jsonify({"error": "tx_id is required for delete action"}), 400
        try:
            tx_id_int = int(tx_id)
            deleted = ledger.delete_transaction(tx_id_int)
            if not deleted:
                return jsonify({"error": f"Transaction {tx_id_int} not found"}), 404
            analytics.invalidate_cache()
            return jsonify({
                "status": "success",
                "deleted": tx_id_int,
                "message": f"Transaction {tx_id_int} deleted successfully",
            })
        except Exception as e:
            return jsonify({"error": f"Failed to delete transaction: {str(e)}"}), 500

    elif action == "clear":
        try:
            ledger.clear()
            analytics.invalidate_cache()
            return jsonify({"status": "success", "message": "Portfolio ledger cleared successfully"})
        except Exception as e:
            return jsonify({"error": f"Failed to clear ledger: {str(e)}"}), 500

    elif action == "import_csv":
        csv_content = payload.get("csv_content")
        if not csv_content:
            return jsonify({"error": "csv_content is required for import_csv"}), 400
        merge = bool(payload.get("merge", False))
        try:
            imported = ledger.import_csv(csv_content, merge=merge)
            analytics.invalidate_cache()
            return jsonify({
                "status": "success",
                "imported_count": int(imported),
                "message": f"Imported {imported} transactions",
            })
        except Exception as e:
            return jsonify({"error": f"CSV import failed: {str(e)}"}), 400

    else:
        return jsonify({"error": f"Unknown action: '{action}'. Supported: 'add', 'delete', 'clear', 'import_csv'"}), 400


@app.route('/api/portfolio/transactions/export', methods=['GET'])
def export_transactions():
    """Endpoint D (Export): Exports entire portfolio transaction ledger as standard CSV."""
    ledger, _ = get_portfolio_runtime()
    try:
        csv_str = ledger.export_csv()
        return Response(
            csv_str,
            mimetype="text/csv",
            headers={"Content-Disposition": "attachment;filename=portfolio_transactions.csv"},
        )
    except Exception as e:
        return jsonify({"error": f"Export failed: {str(e)}"}), 500


@app.route('/api/portfolio/analytics', methods=['POST'])
def portfolio_analytics():
    """
    Endpoint E: Reconstructs daily dual-currency holdings matrix, TWR/MWR curves,
    ILS FX & tax attribution, and What-If benchmark simulations across time-slices.
    """
    ledger, analytics = get_portfolio_runtime()
    payload = request.get_json() or {}

    window = str(payload.get("window", "ALL")).strip()
    start_date = payload.get("start_date", None)
    end_date = payload.get("end_date", None)
    view_currency = str(payload.get("view_currency", "ILS")).upper().strip()
    benchmarks = payload.get("benchmarks", ["SPY"])
    cash_mode = str(payload.get("cash_mode", "auto")).lower().strip()

    if view_currency not in ("USD", "ILS"):
        return jsonify({"error": "view_currency must be either 'USD' or 'ILS'"}), 400

    if not isinstance(benchmarks, list) or not all(isinstance(b, str) for b in benchmarks):
        return jsonify({"error": "benchmarks must be a list of ticker strings"}), 400

    tx_df = ledger.transactions()
    if tx_df.empty:
        empty_res = {
            "window": window,
            "start_date": None,
            "end_date": None,
            "view_currency": view_currency,
            "curves": [],
            "series": [],
            "attribution": {},
            "kpis": {},
            "what_if": {},
        }
        return jsonify(sanitize_for_json(empty_res))

    try:
        res = analytics.slice_and_rebase(
            window=window,
            start_date=start_date,
            end_date=end_date,
            view_currency=view_currency,
            benchmarks=benchmarks,
            cash_mode=cash_mode,
        )

        series_clean = sanitize_for_json(res["series"])
        out_payload = {
            "window": res["window"],
            "start_date": res["start_date"],
            "end_date": res["end_date"],
            "view_currency": res["view_currency"],
            "curves": series_clean,
            "series": series_clean,
            "kpis": sanitize_for_json(res["kpis"]),
            "attribution": sanitize_for_json(res["attribution"]),
            "what_if": sanitize_for_json(res["what_if"]),
        }
        return jsonify(out_payload)
    except Exception as e:
        return jsonify({"error": f"Analytics calculation failed: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
