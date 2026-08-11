from flask import Flask, request, jsonify, render_template
import csv
import os
import gc
from financial_module import (
    run_financial_engine, normalize_stocks, apply_dynamic_filters,
    compute_live_technical_indicators, INDICATOR_COLUMNS, VALID_STRATEGIES, _safe_float,
    load_latest_fundamental_snapshot, get_cached_fundamental_snapshot,
    fetch_tiingo_live_data
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
    stocks_data, err = _load_csv()
    if err:
        return jsonify({"error": err}), (404 if "not found" in err else 500)

    total_csv_count = len(stocks_data)

    # ── Step 1: Pre-Filtering — Apply static filters FIRST to reduce universe ─
    stocks_data = apply_dynamic_filters(stocks_data, allowed_sectors, min_market_cap, max_market_cap)

    if specific_tickers:
        ticker_set = {t.upper().strip() for t in specific_tickers}
        stocks_data = [s for s in stocks_data if s['Ticker'].upper().strip() in ticker_set]

    if not stocks_data:
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
        # ── Step 2: Delayed Computation — Compute Heavy Technical Indicators ON SURVIVORS ─
        surviving_tickers = [s['Ticker'] for s in stocks_data if 'Ticker' in s]
        tech_map = compute_live_technical_indicators(surviving_tickers)

        if tech_map:
            for s in stocks_data:
                sym = s.get('Ticker', '').upper().strip()
                if sym in tech_map:
                    t_data = tech_map[sym]
                    for field in ['RelativeStrength', 'PriceVs200SMA', 'RSI', 'Beta']:
                        cur_val = _safe_float(s.get(field, 0.0))
                        if cur_val == 0.0 or field in ['RelativeStrength', 'PriceVs200SMA']:
                            s[field] = t_data[field]
            del tech_map
            gc.collect()

        # Build metadata lookup for surviving universe
        meta_by_ticker = {
            s['Ticker']: {
                'longName': s.get('LongName', ''),
                'sector': s.get('Sector', ''),
                'marketCap': _safe_float(s.get('MarketCap', 0.0)),
            }
            for s in stocks_data
        }

        # ── Step 3: Normalize filtered & enriched universe ─────────────────────
        normalized_stocks = normalize_stocks(
            stocks_data,
            strategy=strategy,
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


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
