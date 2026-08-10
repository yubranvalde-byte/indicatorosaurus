from flask import Flask, request, jsonify, render_template
import csv
import os
from financial_module import (
    run_financial_engine, normalize_stocks,
    INDICATOR_COLUMNS, VALID_STRATEGIES, _safe_float,
    load_latest_fundamental_snapshot, get_cached_fundamental_snapshot
)
from backtest_engine import run_backtest

app = Flask(__name__)

PRIMARY_CSV = os.path.join(os.path.dirname(__file__), "historical_fundamental_data.csv")
FALLBACK_CSV = os.path.join(os.path.dirname(__file__), "market_data.csv")
CSV_PATH = PRIMARY_CSV if os.path.exists(PRIMARY_CSV) else FALLBACK_CSV


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
    # Frontend sends: { "weights": [int × 16] } in INDICATOR_COLUMNS order.
    raw_weights = payload.get('weights', [])
    if len(raw_weights) != 16:
        return jsonify({"error": f"weights must be a 16-element array, got {len(raw_weights)}"}), 400

    weights_list = [float(w) for w in raw_weights]

    # ── Extract and validate strategy string ───────────────────────────────
    # Optional: if not provided (or null), all 140 stocks pass through unfiltered.
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

    # ── Load CSV ──────────────────────────────────────────────────────────────
    stocks_data, err = _load_csv()
    if err:
        return jsonify({"error": err}), (404 if "not found" in err else 500)

    total_csv_count = len(stocks_data)
    all_raw_stocks = list(stocks_data)  # Preserve the entire global population baseline

    # ── Step 1: Specific Tickers Whitelist Gate ──────────────────────────────
    if specific_tickers:
        ticker_set = {t.upper().strip() for t in specific_tickers}
        stocks_data = [s for s in stocks_data if s['Ticker'].upper().strip() in ticker_set]
        if not stocks_data:
            return jsonify({"error": "No stocks matched the specific_tickers whitelist"}), 400

    try:
        # Build metadata lookup for the post-whitelisted universe
        meta_by_ticker = {
            s['Ticker']: {
                'longName': s.get('LongName', ''),
                'sector': s.get('Sector', ''),
                'marketCap': _safe_float(s.get('MarketCap', 0.0)),
            }
            for s in stocks_data
        }

        # ── Step 2: Normalize filtered universe (one call — filter + normalize) ───
        normalized_stocks = normalize_stocks(
            stocks_data,
            strategy=strategy,
            allowed_sectors=allowed_sectors,
            min_market_cap=min_market_cap,
            max_market_cap=max_market_cap,
            global_stocks_list=all_raw_stocks
        )

        # Build O(1) lookup tables keyed by Ticker — from the FILTERED universe.
        filtered_tickers  = {s['Ticker'] for s in normalized_stocks}
        raw_by_ticker     = {s["Ticker"]: s for s in stocks_data
                             if s["Ticker"] in filtered_tickers}
        norm_by_ticker    = {s["Ticker"]: s for s in normalized_stocks}

        # ── Step 3: Fire C engine on the filtered list ───────────────────────────
        sorted_stocks     = run_financial_engine(normalized_stocks, weights_list)

        # ── Build enriched JSON response ──────────────────────────────────────
        frontend_response = []
        for s in sorted_stocks:
            ticker     = s["Ticker"]
            raw_stock  = raw_by_ticker.get(ticker, {})
            norm_stock = norm_by_ticker.get(ticker, {})

            indicators = {}
            for col in INDICATOR_COLUMNS:
                # Raw value: safe-coerce from CSV string
                try:
                    raw_val = float(raw_stock.get(col, 0.0) or 0.0)
                    if raw_val != raw_val:  # NaN check
                        raw_val = 0.0
                except (ValueError, TypeError):
                    raw_val = 0.0

                # Normalized 0-100 score produced by Python normalization layer
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

        return jsonify({
            "strategy":      strategy,
            "stocks_before": total_csv_count,
            "stocks_after":  len(sorted_stocks),
            "filters_applied": {
                "specific_tickers": specific_tickers,
                "allowed_sectors":  allowed_sectors,
                "min_market_cap":   min_market_cap,
                "max_market_cap":   max_market_cap,
            },
            "results":       frontend_response,
        })

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


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
