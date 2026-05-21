import ctypes
import os
import math
import statistics

# ─── Constants ────────────────────────────────────────────────────────────────

NUM_INDICATORS = 16

# Strict column order matching fetch_static_data.py CSV_HEADER (positions 1-16)
INDICATOR_COLUMNS = [
    'ForwardPE',         # indicators[0]
    'EV_EBITDA',         # indicators[1] (replaced PEGRatio)
    'FCFYield',          # indicators[2]
    'RevenueGrowth',     # indicators[3]
    'EPSGrowth',         # indicators[4]
    'GrossMargin',       # indicators[5]
    'OperatingMargin',   # indicators[6]
    'ROIC',              # indicators[7]
    'DebtToEquity',      # indicators[8]
    'ShareholderYield',  # indicators[9] (replaced DividendYield)
    'FiveYDivGrowth',    # indicators[10]
    'PayoutRatio',       # indicators[11]
    'Beta',              # indicators[12]
    'RSI',               # indicators[13]
    'PriceVs200SMA',     # indicators[14]
    'RelativeStrength',  # indicators[15]
]

# Directionality flags: True = lower is better (inverted normalization)
# Lower is Better: ForwardPE, DebtToEquity, PayoutRatio
LOWER_IS_BETTER = {
    'ForwardPE':    True,
    'DebtToEquity': True,
    'PayoutRatio':  True,
}

# ─── Knock-out Filter Registry ────────────────────────────────────────────────
# Declarative Boolean gates applied PER STRATEGY before normalization.
# Each rule is a (column, operator, threshold) tuple evaluated against raw CSV values.
# Stocks failing ANY rule in their strategy's list are dropped entirely.
# 'CurrentRatio' is read from the CSV but is NOT in INDICATOR_COLUMNS —
# it acts as a pure filter gate and is never packed into the C struct.
KNOCKOUT_FILTERS = {
    'deep-value': [
        ('FCFYield',         '>',  0.0   ),  # Must generate positive free cash flow
        ('OperatingMargin',  '>',  0.0   ),  # Must be profitable from core operations
    ],
    'hyper-growth': [
        ('RevenueGrowth',    '>',  0.0   ),  # Top-line must not be shrinking
        ('GrossMargin',      '>',  0.20  ),  # Baseline pricing power: >20%
    ],
    'momentum': [
        ('PriceVs200SMA',    '>',  0.0   ),  # Price must be above 200-day SMA
        ('RelativeStrength', '>',  0.0   ),  # Must outperform SPY benchmark over 52w
    ],
    'income': [
        ('PayoutRatio',      '>',  0.0   ),  # Must have earnings (pays a dividend)
        ('PayoutRatio',      '<',  0.85  ),  # Sustainable payout: ceiling at 85%
        ('ShareholderYield', '>',  0.015 ),  # Minimum 1.5% annual shareholder yield
    ],
    'quality': [
        ('ROIC',             '>',  0.08  ),  # Return on assets proxy must exceed 8%
        ('CurrentRatio',     '>',  1.0   ),  # Short-term liquidity: assets > liabilities
    ],
}

VALID_STRATEGIES = frozenset(KNOCKOUT_FILTERS.keys())

# ─── Statistical Normalization Config (Phase 1) ────────────────────────────────
# Optimal bounds [L, U] and decay rates (sigma_L, sigma_R) for piecewise gaussian metrics.
GOLDILOCKS_PARAMS = {
    'PayoutRatio':      {'L': 0.35,  'U': 0.70,  'sigma_L': 0.25,  'sigma_R': 0.20},
    'RSI':              {'L': 45.0,  'U': 60.0,  'sigma_L': 15.0,  'sigma_R': 10.0},
    'ForwardPE':        {'L': 12.0,  'U': 28.0,  'sigma_L': 5.0,   'sigma_R': 15.0},
    'EV_EBITDA':        {'L': 8.0,   'U': 16.0,  'sigma_L': 3.0,   'sigma_R': 8.0},
    'DebtToEquity':     {'L': 0.1,   'U': 0.8,   'sigma_L': 0.4,   'sigma_R': 0.6},
    'Beta':             {'L': 0.8,   'U': 1.2,   'sigma_L': 0.6,   'sigma_R': 0.4},
    'PriceVs200SMA':    {'L': 1.02,  'U': 1.15,  'sigma_L': 0.10,  'sigma_R': 0.08},
    'ShareholderYield': {'L': 0.04,  'U': 0.09,  'sigma_L': 0.03,  'sigma_R': 0.04},
}

# ─── 1. Data Structure Definition ─────────────────────────────────────────────

class Stock(ctypes.Structure):
    """
    Must be byte-for-byte identical to the C struct in stock.h:
        char   ticker[8];              //  8 bytes
        float  indicators[16];         // 64 bytes
        double final_score;            //  8 bytes
    Total: 80 bytes, no padding.
    """
    _fields_ = [
        ("ticker",      ctypes.c_char * 8),
        ("indicators",  ctypes.c_float * NUM_INDICATORS),
        ("final_score", ctypes.c_double),
    ]


# ─── 2. Library Loading & Function Signatures ──────────────────────────────────

lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libsortbuslines.so")

try:
    lib = ctypes.CDLL(lib_path)
except OSError as e:
    print(f"Error loading {lib_path}. Did you compile it?")
    raise e

# void compute_scores(Stock* stocks, int num_stocks, const double* weights);
lib.compute_scores.argtypes = [ctypes.POINTER(Stock), ctypes.c_int, ctypes.POINTER(ctypes.c_double)]
lib.compute_scores.restype  = None

# void sort_stocks_by_score(Stock* stocks, int num_stocks);
lib.sort_stocks_by_score.argtypes = [ctypes.POINTER(Stock), ctypes.c_int]
lib.sort_stocks_by_score.restype  = None


# ─── 3. Knock-out Filter Engine ──────────────────────────────────────────────

def apply_knockout_filters(stocks_list, strategy):
    """
    Apply strategy-specific Boolean gates to drop ineligible stocks.

    Runs BEFORE Min-Max normalization so that extreme outlier values from
    knocked-out stocks do not anchor the column min/max and compress scores
    for the surviving, strategy-eligible universe.

    Args:
        stocks_list: list of raw CSV dicts (pre-normalization).
        strategy:    one of VALID_STRATEGIES, or None for no filtering.

    Returns:
        Filtered list (original is not mutated). If strategy is None or
        unrecognised, returns the original list unchanged.
    """
    rules = KNOCKOUT_FILTERS.get(strategy)
    if not rules:
        return stocks_list   # no-op: 'custom' profile or unknown strategy

    survivors = []
    for stock in stocks_list:
        passed = True
        for col, op, threshold in rules:
            val = _safe_float(stock.get(col, 0.0))
            if   op == '>'  and not (val >  threshold): passed = False; break
            elif op == '<'  and not (val <  threshold): passed = False; break
            elif op == '>=' and not (val >= threshold): passed = False; break
            elif op == '<=' and not (val <= threshold): passed = False; break
        if passed:
            survivors.append(stock)

    before = len(stocks_list)
    after  = len(survivors)
    print(f"[KnockoutFilter] '{strategy}': {before} → {after} stocks "
          f"({before - after} eliminated)")
    return survivors


def apply_dynamic_filters(stocks_list, allowed_sectors=None, min_market_cap=None, max_market_cap=None):
    """
    Apply optional user-defined demographic gates (sectors and market cap range).
    Returns a new list of dicts. If no filters are provided, returns the list unchanged.
    """
    if not stocks_list:
        return []

    # Process allowed_sectors into a set of lowercased stripped strings for robust matching
    sector_set = None
    if allowed_sectors is not None:
        if isinstance(allowed_sectors, str):
            sector_set = {allowed_sectors.strip().lower()}
        elif hasattr(allowed_sectors, '__iter__'):
            sector_set = {str(s).strip().lower() for s in allowed_sectors if s}

    survivors = []
    for stock in stocks_list:
        # 1. Sector filter
        if sector_set is not None:
            sector = str(stock.get('Sector', '')).strip().lower()
            if sector not in sector_set:
                continue

        # Coerce market cap to safe float
        mkt_cap = _safe_float(stock.get('MarketCap', 0.0))

        # 2. Min market cap filter
        if min_market_cap is not None:
            try:
                min_val = float(min_market_cap)
                if mkt_cap < min_val:
                    continue
            except (ValueError, TypeError):
                pass

        # 3. Max market cap filter
        if max_market_cap is not None:
            try:
                max_val = float(max_market_cap)
                if mkt_cap > max_val:
                    continue
            except (ValueError, TypeError):
                pass

        survivors.append(stock)

    before = len(stocks_list)
    after = len(survivors)
    print(f"[DynamicFilter] Sector/Cap: {before} → {after} stocks ({before - after} eliminated)")
    return survivors


# ─── 4. Min-Max Normalization Layer ───────────────────────────────────────────

def _safe_float(val):
    """Coerce a CSV cell value to float, treating 'nan'/empty/None as 0.0."""
    if val is None:
        return 0.0
    s = str(val).strip().lower()
    if s in ('nan', '', 'inf', '-inf'):
        return 0.0
    try:
        result = float(s)
        # Guard against IEEE nan/inf that slipped through
        if math.isnan(result) or math.isinf(result):
            return 0.0
        return result
    except (ValueError, TypeError):
        return 0.0


def _calculate_piecewise_gaussian(val, params):
    """
    Institutional-grade piecewise Gaussian normalizer for Goldilocks indicators.
    Scoring:
      - If val < L: Score = 100 * exp( -((val - L)^2) / (2 * sigma_L^2) )
      - If L <= val <= U: Score = 100
      - If val > U: Score = 100 * exp( -((val - U)^2) / (2 * sigma_R^2) )
    Returns a score in [0.0, 100.0]. Gracefully handles bad inputs and math overflows.
    """
    if val is None:
        return 0.0
    try:
        val = float(val)
    except (ValueError, TypeError):
        return 0.0

    if math.isnan(val) or math.isinf(val):
        return 0.0

    L = params.get('L')
    U = params.get('U')
    sigma_L = params.get('sigma_L')
    sigma_R = params.get('sigma_R')

    if L is None or U is None or sigma_L is None or sigma_R is None:
        return 0.0

    if val < L:
        if sigma_L == 0.0:
            return 0.0
        exponent = -((val - L) ** 2) / (2 * (sigma_L ** 2))
        try:
            return 100.0 * math.exp(exponent)
        except OverflowError:
            return 0.0
    elif val > U:
        if sigma_R == 0.0:
            return 0.0
        exponent = -((val - U) ** 2) / (2 * (sigma_R ** 2))
        try:
            return 100.0 * math.exp(exponent)
        except OverflowError:
            return 0.0
    else:
        return 100.0


def _calculate_sigmoid_zscore(val, median_val, mad_val, invert=False):
    """
    Institutional-grade Monotonic Normalizer utilizing a Sigmoid over Robust Z-Scores.
    Scoring:
      - z = (val - median_val) / max(mad_val, 10^-4)
      - If invert is True (Lower is Better): z = -z
      - Score = 100 / (1 + exp(-1.5 * z))
    Gracefully handles division by zero and outliers using a strict floor.
    """
    if val is None or median_val is None or mad_val is None:
        return 50.0
    try:
        val = float(val)
        median_val = float(median_val)
        mad_val = float(mad_val)
    except (ValueError, TypeError):
        return 50.0

    if math.isnan(val) or math.isinf(val) or math.isnan(median_val) or math.isinf(median_val):
        return 50.0

    if math.isnan(mad_val) or math.isinf(mad_val):
        mad_val = 0.0

    denominator = max(mad_val, 1e-4)
    z = (val - median_val) / denominator
    if invert:
        z = -z

    try:
        exponent = -1.5 * z
        # Bound exponent to prevent OverflowError in math.exp
        if exponent > 700.0:
            return 0.0
        if exponent < -700.0:
            return 100.0
        return 100.0 / (1.0 + math.exp(exponent))
    except (OverflowError, ZeroDivisionError):
        return 0.0 if z < 0.0 else 100.0


OVERRIDE_COLUMNS = frozenset({'PayoutRatio', 'ForwardPE', 'DebtToEquity', 'EV_EBITDA'})


def normalize_stocks(stocks_list, strategy=None, allowed_sectors=None, min_market_cap=None, max_market_cap=None, global_stocks_list=None):
    """
    Apply dynamic filters, strategy knock-out filters, then normalize the surviving universe
    using piecewise Gaussian (Goldilocks) or robust Sigmoid Z-score curves.

    Args:
        stocks_list:     list of raw CSV dicts.
        strategy:        one of VALID_STRATEGIES, or None for no filtering.
        allowed_sectors: list of allowed sector names, or None.
        min_market_cap:  minimum market cap, or None.
        max_market_cap:  maximum market cap, or None.
        global_stocks_list: list of all raw CSV dicts (global baseline for sector neutrality).
    """
    if not stocks_list:
        return []

    # Sector-Neutral Baseline Stabilization: Use the entire global population prior to filtering
    baseline_list = global_stocks_list if global_stocks_list is not None else list(stocks_list)

    # Step 1: Apply dynamic user gates (sectors, cap ranges)
    stocks_list = apply_dynamic_filters(stocks_list, allowed_sectors, min_market_cap, max_market_cap)
    if not stocks_list:
        return []

    # Step 2: Apply knock-out gates (no-op when strategy is None)
    stocks_list = apply_knockout_filters(stocks_list, strategy)
    if not stocks_list:
        return []   # all stocks knocked out — guard against empty universe

    n = len(stocks_list)

    # Step 3: Extract raw float columns (only scored indicators)
    raw = {}
    for col in INDICATOR_COLUMNS:
        raw[col] = [_safe_float(stocks_list[i].get(col, 0.0)) for i in range(n)]

    # Step 4: Normalize each column dynamically using the configuration router
    normalized = {}
    for col in INDICATOR_COLUMNS:
        col_vals = raw[col]
        
        # Check if this metric uses Goldilocks config
        if col == 'EV_EBITDA':
            params = GOLDILOCKS_PARAMS[col]
            scores = []
            for i, val in enumerate(col_vals):
                ebitda = _safe_float(stocks_list[i].get('EBITDA', 0.0))
                # Strict Pre-Scoring Gatekeeper Overrides:
                # EBITDA <= 0 -> 0.0, or negative EV/EBITDA multiple -> 0.0
                if ebitda <= 0.0 or val < 0.0:
                    scores.append(0.0)
                else:
                    scores.append(_calculate_piecewise_gaussian(val, params))
            normalized[col] = scores
        elif col == 'ShareholderYield':
            params = GOLDILOCKS_PARAMS[col]
            scores = []
            for val in col_vals:
                # Strict Pre-Scoring Gatekeeper Override:
                # Shareholder Yield < -0.02 -> 0.0
                if val < -0.02:
                    scores.append(0.0)
                else:
                    scores.append(_calculate_piecewise_gaussian(val, params))
            normalized[col] = scores
        elif col in GOLDILOCKS_PARAMS:
            params = GOLDILOCKS_PARAMS[col]
            scores = []
            for val in col_vals:
                # Apply Institutional Overrides
                if col in OVERRIDE_COLUMNS and val < 0.0:
                    scores.append(0.0)
                else:
                    scores.append(_calculate_piecewise_gaussian(val, params))
            normalized[col] = scores
        else:
            # Monotonic Sigmoid Z-score route
            invert = LOWER_IS_BETTER.get(col, False)
            if col in ('RevenueGrowth', 'EPSGrowth'):
                # Sector-Neutral Sigmoid Z-score
                # 1. Group values from baseline_list by sector
                sector_vals = {}
                for stock in baseline_list:
                    sect = str(stock.get('Sector', '')).strip().lower()
                    val = _safe_float(stock.get(col, 0.0))
                    if sect not in sector_vals:
                        sector_vals[sect] = []
                    sector_vals[sect].append(val)
                
                # 2. Compute median and MAD per sector using baseline
                sector_stats = {}
                for sect, vals in sector_vals.items():
                    med = statistics.median(vals) if vals else 0.0
                    mad = statistics.median([abs(x - med) for x in vals]) if vals else 0.0
                    sector_stats[sect] = (med, mad)
                
                # 3. Calculate scores for surviving stocks
                scores = []
                for i, stock in enumerate(stocks_list):
                    sect = str(stock.get('Sector', '')).strip().lower()
                    med, mad = sector_stats.get(sect, (0.0, 0.0))
                    val = col_vals[i]
                    scores.append(_calculate_sigmoid_zscore(val, med, mad, invert))
                normalized[col] = scores
            else:
                # Global Sigmoid Z-score
                median_val = statistics.median(col_vals) if col_vals else 0.0
                mad_val = statistics.median([abs(x - median_val) for x in col_vals]) if col_vals else 0.0
                scores = []
                for val in col_vals:
                    if col in OVERRIDE_COLUMNS and val < 0.0:
                        scores.append(0.0)
                    else:
                        scores.append(_calculate_sigmoid_zscore(val, median_val, mad_val, invert))
                normalized[col] = scores

    # Step 5: Rebuild stock dicts with normalized values
    result = []
    for i in range(n):
        norm_stock = {'Ticker': stocks_list[i].get('Ticker', '')}
        for col in INDICATOR_COLUMNS:
            norm_stock[col] = normalized[col][i]
        result.append(norm_stock)

    return result


# ─── 4. Main FFI Entry Point ───────────────────────────────────────────────────

def run_financial_engine(normalized_stocks, weights_list):
    """
    Surgical Patch: Accepts pre-normalized stocks directly to bypass duplicate normalization.
    Populates ctypes array → C dot-product → QuickSort.
    """
    num_stocks = len(normalized_stocks)
    if num_stocks == 0:
        return []

    # ── Step 1: Populate ctypes Stock array ──────────────────────────────────
    StockArray = Stock * num_stocks
    c_stocks   = StockArray()

    for i, stock_data in enumerate(normalized_stocks):
        # Encode ticker: max 7 meaningful chars + null terminator fits in char[8]
        encoded = stock_data['Ticker'].encode('utf-8')[:7]
        c_stocks[i].ticker = encoded

        # Map 16 normalized indicator values sequentially into indicators[]
        for j, col in enumerate(INDICATOR_COLUMNS):
            c_stocks[i].indicators[j] = float(stock_data.get(col, 0.0))

        c_stocks[i].final_score = 0.0

    # ── Step 2: Build weights array ──────────────────────────────────────────
    WeightsArray = ctypes.c_double * NUM_INDICATORS
    c_weights    = WeightsArray(*[float(w) for w in weights_list])

    # ── Step 3: Fire C engine (dot product + QuickSort) ──────────────────────
    start_ptr   = ctypes.cast(c_stocks,   ctypes.POINTER(Stock))
    weights_ptr = ctypes.cast(c_weights,  ctypes.POINTER(ctypes.c_double))

    lib.compute_scores(start_ptr, num_stocks, weights_ptr)
    lib.sort_stocks_by_score(start_ptr, num_stocks)

    # ── Step 4: Reconstruct sorted Python list ───────────────────────────────
    return [
        {
            'Ticker':     c_stocks[i].ticker.decode('utf-8'),
            'FinalScore': round(c_stocks[i].final_score, 4),
        }
        for i in range(num_stocks)
    ]
