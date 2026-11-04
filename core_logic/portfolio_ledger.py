"""
portfolio_ledger.py
===================
Portfolio Intelligence — Phase 2:
Persistent dual-currency (USD / ILS) transaction ledger, FIFO tax-lot engine, and Israeli
capital-gains tax calculator (25% on real gains).

Key rules (per lot, all amounts in ILS):
  OriginalCost      = Q * P_buy  * FX_buy  + Fee_buy  * FX_buy        (FX = 1.0 for ILS assets)
  Proceeds          = Q * P_sell * FX_sell - Fee_sell * FX_sell
  InflationaryRatio = max(1, FX_sell / FX_buy)    for USD assets
                      max(1, CPI_sell / CPI_buy)  for ILS assets
  AdjustedCost      = OriginalCost * InflationaryRatio
  Proceeds >= AdjustedCost                 -> real gain = Proceeds - AdjustedCost, exempt = AdjustedCost - OriginalCost
  OriginalCost <= Proceeds < AdjustedCost  -> real gain = 0,                       exempt = Proceeds - OriginalCost
  Proceeds < OriginalCost                  -> real loss = Proceeds - OriginalCost (offsettable)

Annual tax: 25% * max(0, real gains - (current-year real losses + carried-forward losses)).
Dividends: flat 25% withholding on the gross ILS amount.

FX (`USD_ILS`) and CPI (`IL_CPI`, the known index on that date) come from
`price_resolver.resolve_macro_series`. A custom `macro_provider` can be injected (used by the test suite).
The FIFO replay is an inherently sequential state machine, so it is a plain Python loop over
transactions; the NumPy-vectorization mandate covers factor normalization, not lot accounting.
"""

import math
import os
import warnings
from collections import deque
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd
from financial_module import _scalar_float, sanitize_for_json, _clean_frame, _to_naive_ts, _release_arrow_memory

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DEFAULT_LEDGER_PATH = os.path.join(PROJECT_ROOT, "data", "portfolio_transactions.parquet")

TX_TYPES = ("BUY", "SELL", "DIVIDEND", "CASH_DEPOSIT", "CASH_WITHDRAWAL")
CASH_TX_TYPES = ("CASH_DEPOSIT", "CASH_WITHDRAWAL")
CURRENCIES = ("USD", "ILS")
CAPITAL_GAINS_TAX_RATE = 0.25
DIVIDEND_WITHHOLDING_RATE = 0.25
CASH_TICKER = "CASH"
QTY_EPSILON = 1e-9
ROUNDING_TOLERANCE = 0.01
CANONICAL_SPLIT_FACTORS = (2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 50)
SPLIT_RATIO_TOLERANCE = 0.22
TX_TYPE_PRIORITY = {
    "CASH_DEPOSIT": 0,
    "BUY": 1,
    "DIVIDEND": 2,
    "SELL": 3,
    "CASH_WITHDRAWAL": 4,
}

LEDGER_COLUMNS = ["tx_id", "date", "ticker", "tx_type", "quantity", "price", "fee", "currency", "notes"]

_CSV_COLUMN_PRIORITIES = {
    "date": ["trade date", "execution date", "transaction date", "date"],
    "ticker": ["ticker", "symbol", "security", "instrument"],
    "tx_type": ["tx_type", "transaction type", "type", "action", "side"],
    "quantity": ["quantity", "qty", "shares", "units"],
    "price": ["purchase price", "trade price", "execution price", "avg price", "price"],
    "fee": ["fee", "commission", "commissions", "expenses"],
    "currency": ["currency", "ccy"],
    "notes": ["notes", "comment", "comments", "description"],
    "tx_id": ["tx_id", "transaction id"],
}

_TX_TYPE_ALIASES = {
    "BUY": "BUY",
    "B": "BUY",
    "BOT": "BUY",
    "PURCHASE": "BUY",
    "BUY_TO_OPEN": "BUY",
    "LONG": "BUY",
    "SELL": "SELL",
    "S": "SELL",
    "SLD": "SELL",
    "SALE": "SELL",
    "SELL_TO_CLOSE": "SELL",
    "SHORT": "SELL",
    "DIVIDEND": "DIVIDEND",
    "DIV": "DIVIDEND",
    "DIVIDENDS": "DIVIDEND",
    "CASH_DIVIDEND": "DIVIDEND",
    "CASH_DEPOSIT": "CASH_DEPOSIT",
    "DEPOSIT": "CASH_DEPOSIT",
    "CASH_IN": "CASH_DEPOSIT",
    "DEP": "CASH_DEPOSIT",
    "CASH_WITHDRAWAL": "CASH_WITHDRAWAL",
    "WITHDRAWAL": "CASH_WITHDRAWAL",
    "WITHDRAW": "CASH_WITHDRAWAL",
    "CASH_OUT": "CASH_WITHDRAWAL",
    "WDR": "CASH_WITHDRAWAL",
}


def _is_blank_csv_val(val) -> bool:
    if val is None:
        return True
    if isinstance(val, float) and math.isnan(val):
        return True
    s = str(val).strip().lower()
    return s in ("", "nan", "none", "null", "n/a", "na", "-")


def _parse_flexible_date(val) -> str:
    """
    Parse brokerage date strings/numbers into canonical 'YYYY-MM-DD'.
    Supports YYYYMMDD ints/floats (e.g. 20231205 or 20231205.0), YYYY-MM-DD, YYYY/MM/DD, DD/MM/YYYY.
    Returns '' if missing or unparseable.
    """
    if _is_blank_csv_val(val):
        return ""
    s = str(val).strip()
    if s.endswith(".0") and s[:-2].isdigit():
        s = s[:-2]
    try:
        if len(s) == 8 and s.isdigit():
            ts = pd.to_datetime(s, format="%Y%m%d", errors="raise")
            return ts.strftime("%Y-%m-%d")
        if "/" in s:
            parts = s.split("/")
            if len(parts[0]) == 4:
                ts = pd.to_datetime(s, format="%Y/%m/%d", errors="raise")
            else:
                ts = pd.to_datetime(s, dayfirst=True, errors="raise")
            return ts.strftime("%Y-%m-%d")
        ts = pd.Timestamp(s)
        if pd.isna(ts):
            return ""
        return ts.strftime("%Y-%m-%d")
    except Exception:
        return ""


def _normalize_tx_type(val) -> str:
    if _is_blank_csv_val(val):
        return ""
    norm = str(val).strip().upper().replace("-", "_").replace(" ", "_")
    return _TX_TYPE_ALIASES.get(norm, "")


def _resolve_csv_columns(raw_columns) -> tuple:
    """
    Case-insensitively resolve CSV header columns using priority rules.
    Returns (col_map, label_map) where col_map maps canonical name -> actual CSV header name.
    """
    lower_to_actual = {str(c).strip().lower(): c for c in raw_columns}
    col_map = {}
    for canonical, candidates in _CSV_COLUMN_PRIORITIES.items():
        for cand in candidates:
            if cand in lower_to_actual:
                col_map[canonical] = lower_to_actual[cand]
                break

    required_canonical = ("date", "ticker", "tx_type", "quantity", "price")
    missing_structural = [k for k in required_canonical if k not in col_map]
    if missing_structural:
        raise ValueError(
            f"CSV header is missing required column(s) for {missing_structural}. "
            f"Detected columns: {list(raw_columns)}"
        )

    date_col_lower = str(col_map["date"]).strip().lower()
    price_col_lower = str(col_map["price"]).strip().lower()
    label_map = {
        "date": "Trade Date" if "trade" in date_col_lower else str(col_map["date"]).strip().title(),
        "ticker": "Symbol" if "symbol" in str(col_map["ticker"]).strip().lower() else "Ticker",
        "tx_type": "Transaction Type",
        "quantity": "Quantity",
        "price": "Purchase Price" if "purchase" in price_col_lower else str(col_map["price"]).strip().title(),
    }
    return col_map, label_map


def _match_canonical_split_factor(p_tx: float, p_mkt: float, allow_reverse: bool = True) -> float:
    """
    Compare execution price p_tx against split-adjusted market price p_mkt on the transaction date.
    Returns scale factor S such that Q_adj = Q_tx * S and P_adj = P_tx / S:
      - Forward split: r = p_tx / p_mkt within +/- 22% of S in CANONICAL_SPLIT_FACTORS -> float(S)
      - Reverse split: r_rev = p_mkt / p_tx within +/- 22% of S in CANONICAL_SPLIT_FACTORS -> 1.0 / float(S)
      - Otherwise returns 1.0.
    """
    if not (p_tx > 0.0 and p_mkt > 0.0) or math.isnan(p_tx) or math.isnan(p_mkt):
        return 1.0
    r = p_tx / p_mkt
    if abs(r - 1.0) <= SPLIT_RATIO_TOLERANCE:
        return 1.0
    if r > 1.0:
        best_s = min(CANONICAL_SPLIT_FACTORS, key=lambda s: abs(r - s) / s)
        if abs(r - best_s) / best_s <= SPLIT_RATIO_TOLERANCE:
            return float(best_s)
    elif allow_reverse and r < 1.0:
        r_rev = p_mkt / p_tx
        best_s = min(CANONICAL_SPLIT_FACTORS, key=lambda s: abs(r_rev - s) / s)
        if abs(r_rev - best_s) / best_s <= SPLIT_RATIO_TOLERANCE:
            return 1.0 / float(best_s)
    return 1.0


def _lookup_market_price_on_date(px_df: pd.DataFrame, ticker: str, date_str: str) -> float:
    """Lookup resolved split-adjusted market price for `ticker` on or immediately before `date_str`."""
    if px_df is None or px_df.empty or ticker not in px_df.columns:
        return 0.0
    col = px_df[ticker]
    valid = col[col > 0.0].dropna()
    if valid.empty:
        return 0.0
    ts = pd.Timestamp(date_str).normalize()
    if ts in valid.index:
        return float(valid.loc[ts])
    prior = valid.loc[:ts]
    if not prior.empty:
        return float(prior.iloc[-1])
    after = valid.loc[ts : ts + pd.Timedelta(days=7)]
    if not after.empty:
        return float(after.iloc[0])
    return 0.0


def align_pre_split_transactions(tx_items: list, px_df: pd.DataFrame, allow_reverse: bool = True) -> list:
    """
    Deterministic split-basis alignment pass across chronological BUY/SELL transaction dicts.
    Compares each transaction's execution price P_tx against P_{mkt, t} from `px_df`.
    When r = P_tx / P_{mkt, t} (or P_{mkt, t} / P_tx for reverse splits) falls within +/- 22%
    of a canonical split integer S in {2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 50} and normalizing
    the lot sequence preserves non-negative FIFO share balances, normalizes:
      Q_adj = Q_tx * S,  P_adj = P_tx / S  (preserving Q_adj * P_adj == Q_tx * P_tx and fees).
    """
    if not tx_items or px_df is None or px_df.empty:
        return tx_items

    by_ticker = {}
    for it in tx_items:
        if it.get("tx_type") in ("BUY", "SELL") and not it.get("is_zero_price_buy", False):
            by_ticker.setdefault(str(it["ticker"]), []).append(it)

    for ticker, t_items in by_ticker.items():
        if ticker not in px_df.columns:
            continue

        item_factors = []
        item_mkt_prices = []
        for it in t_items:
            p_mkt = _lookup_market_price_on_date(px_df, ticker, str(it["date"]))
            p_tx = float(it["price"])
            s_i = _match_canonical_split_factor(p_tx, p_mkt, allow_reverse=allow_reverse) if p_mkt > 0.0 else None
            item_factors.append(s_i)
            item_mkt_prices.append(p_mkt)

        if all(s is None or s == 1.0 for s in item_factors):
            continue

        # Partition into FIFO cycles if unscaled shares are non-negative at every step
        cycles = []
        curr_cycle = []
        rem_open = 0.0
        unscaled_valid = True

        for it, s_i, p_mkt in zip(t_items, item_factors, item_mkt_prices):
            q = float(it["quantity"])
            curr_cycle.append((it, s_i, p_mkt))
            if it["tx_type"] == "BUY":
                rem_open += q
            else:
                if q > rem_open + ROUNDING_TOLERANCE:
                    unscaled_valid = False
                    break
                rem_open = max(0.0, rem_open - q)
                if rem_open <= ROUNDING_TOLERANCE:
                    rem_open = 0.0
                    cycles.append(("closed", curr_cycle))
                    curr_cycle = []

        if unscaled_valid and curr_cycle:
            cycles.append(("open", curr_cycle))

        if unscaled_valid:
            for cyc_type, cyc in cycles:
                valid_s = [s for _, s, _ in cyc if s is not None]
                if not valid_s or all(s == 1.0 for s in valid_s):
                    continue

                if cyc_type == "closed":
                    # In a balanced closed cycle (sum(BUY) ~= sum(SELL)), all lots in the pre-split
                    # regime must be scaled by the same factor S to preserve FIFO balance closure.
                    non_unit = [s for s in valid_s if s != 1.0]
                    if not non_unit:
                        continue
                    best_s = max(set(non_unit), key=lambda x: non_unit.count(x))
                    # Verify all priced items in the cycle align with best_s within tolerance
                    all_consistent = True
                    for it_c, _, p_mkt_c in cyc:
                        if p_mkt_c > 0.0:
                            aligned_p = float(it_c["price"]) / best_s
                            rel_diff = abs(aligned_p - p_mkt_c) / p_mkt_c
                            if rel_diff > SPLIT_RATIO_TOLERANCE + 0.03:
                                all_consistent = False
                                break
                    if all_consistent:
                        for it_c, _, _ in cyc:
                            it_c["quantity"] = float(it_c["quantity"]) * best_s
                            it_c["price"] = float(it_c["price"]) / best_s
                else:
                    # Open cycle
                    has_sells = any(it_c["tx_type"] == "SELL" for it_c, _, _ in cyc)
                    if not has_sells:
                        for it_c, s_c, _ in cyc:
                            if s_c is not None and s_c != 1.0:
                                it_c["quantity"] = float(it_c["quantity"]) * s_c
                                it_c["price"] = float(it_c["price"]) / s_c
                    else:
                        non_unit = [s for s in valid_s if s != 1.0]
                        best_s = max(set(non_unit), key=lambda x: non_unit.count(x)) if non_unit else 1.0
                        all_consistent = best_s != 1.0 and all(
                            abs((float(it_c["price"]) / best_s) - p_mkt_c) / p_mkt_c <= SPLIT_RATIO_TOLERANCE + 0.03
                            for it_c, _, p_mkt_c in cyc if p_mkt_c > 0.0
                        )
                        if all_consistent:
                            for it_c, _, _ in cyc:
                                it_c["quantity"] = float(it_c["quantity"]) * best_s
                                it_c["price"] = float(it_c["price"]) / best_s
                        else:
                            # Test individual scaling for FIFO non-negativity
                            test_open = 0.0
                            ind_ok = True
                            for it_c, s_c, _ in cyc:
                                s_eff = s_c if s_c is not None else 1.0
                                q_scaled = float(it_c["quantity"]) * s_eff
                                if it_c["tx_type"] == "BUY":
                                    test_open += q_scaled
                                else:
                                    if q_scaled > test_open + ROUNDING_TOLERANCE:
                                        ind_ok = False
                                        break
                                    test_open = max(0.0, test_open - q_scaled)
                            if ind_ok:
                                for it_c, s_c, _ in cyc:
                                    s_eff = s_c if s_c is not None else 1.0
                                    if s_eff != 1.0:
                                        it_c["quantity"] = float(it_c["quantity"]) * s_eff
                                        it_c["price"] = float(it_c["price"]) / s_eff
        else:
            # Unscaled sequence had an apparent oversell (e.g. pre-split BUY followed by post-split SELL).
            # Check if individual split factors restore non-negative FIFO balances.
            test_open = 0.0
            ind_ok = True
            for it, s_i in zip(t_items, item_factors):
                s_eff = s_i if s_i is not None else 1.0
                q_scaled = float(it["quantity"]) * s_eff
                if it["tx_type"] == "BUY":
                    test_open += q_scaled
                else:
                    if q_scaled > test_open + ROUNDING_TOLERANCE:
                        ind_ok = False
                        break
                    test_open = max(0.0, test_open - q_scaled)
            if ind_ok:
                for it, s_i in zip(t_items, item_factors):
                    s_eff = s_i if s_i is not None else 1.0
                    if s_eff != 1.0:
                        it["quantity"] = float(it["quantity"]) * s_eff
                        it["price"] = float(it["price"]) / s_eff

    return tx_items


def align_pre_split_dataframe(tx_df: pd.DataFrame, px_df: pd.DataFrame, allow_reverse: bool = True) -> pd.DataFrame:
    """Apply `align_pre_split_transactions` to a transaction DataFrame copy and return it."""
    if tx_df is None or tx_df.empty or px_df is None or px_df.empty:
        return tx_df
    df_out = tx_df.copy()
    records = df_out.to_dict(orient="records")
    align_pre_split_transactions(records, px_df, allow_reverse=allow_reverse)
    df_out["quantity"] = [float(r["quantity"]) for r in records]
    df_out["price"] = [float(r["price"]) for r in records]
    return df_out


# ─── Numerical Safety ─────────────────────────────────────────────────────────

def _clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Zero NaN/inf in every numeric column."""
    if df.empty:
        return df
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols):
        df[num_cols] = df[num_cols].fillna(0.0).replace([np.inf, -np.inf], 0.0)
    return df


# ─── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class Transaction:
    tx_id: int
    date: str
    ticker: str
    tx_type: str
    quantity: float
    price: float
    fee: float
    currency: str
    notes: str = ""

    @classmethod
    def build(cls, tx_id, date, ticker, tx_type, quantity, price=1.0, fee=0.0, currency="USD", notes=""):
        """Validate and normalize raw input. Raises ValueError on invalid fields."""
        tx_type = str(tx_type).upper().strip()
        if tx_type not in TX_TYPES:
            raise ValueError(f"Invalid tx_type '{tx_type}'. Expected one of {TX_TYPES}.")

        currency = str(currency).upper().strip()
        if currency not in CURRENCIES:
            raise ValueError(f"Invalid currency '{currency}'. Expected one of {CURRENCIES}.")

        try:
            ts = pd.Timestamp(str(date).strip())
        except Exception:
            raise ValueError(f"Invalid date '{date}'. Expected YYYY-MM-DD.")
        if ts.normalize() > pd.Timestamp.now().normalize():
            raise ValueError(f"Transaction date {ts.date()} is in the future.")

        ticker = str(ticker or "").upper().strip()
        if tx_type in CASH_TX_TYPES:
            ticker = CASH_TICKER
            price = 1.0
        elif not ticker or ticker == CASH_TICKER:
            raise ValueError(f"{tx_type} requires a security ticker.")
        if ticker.endswith(".TA") and currency != "ILS":
            raise ValueError(f"TASE security {ticker} must be recorded in ILS (got {currency}).")

        quantity, price, fee = (_scalar_float(quantity, float("nan")), _scalar_float(price, float("nan")),
                                _scalar_float(fee, float("nan")))
        if not quantity > 0.0:
            raise ValueError(f"quantity must be > 0 (got {quantity}).")
        if not price > 0.0:
            raise ValueError(f"price must be > 0 (got {price}).")
        if not fee >= 0.0:
            raise ValueError(f"fee must be >= 0 (got {fee}).")

        notes = "" if notes is None or (isinstance(notes, float) and math.isnan(notes)) else str(notes)
        return cls(int(tx_id), ts.strftime("%Y-%m-%d"), ticker, tx_type, quantity, price, fee, currency, notes)


@dataclass
class TaxLot:
    lot_id: int              # tx_id of the originating BUY
    ticker: str
    currency: str
    buy_date: str
    qty_original: float
    qty_open: float
    price: float             # native currency per share
    fee_total: float         # native currency, whole original lot
    fx_buy: float
    cpi_buy: float

    @property
    def fee_per_share(self) -> float:
        return self.fee_total / self.qty_original if self.qty_original > 0 else 0.0


@dataclass
class LedgerState:
    open_lots: dict = field(default_factory=dict)        # ticker -> deque[TaxLot]
    realized: list = field(default_factory=list)         # per consumed lot slice
    dividends: list = field(default_factory=list)
    cash: dict = field(default_factory=lambda: {c: 0.0 for c in CURRENCIES})
    available_cash: dict = field(default_factory=lambda: {c: 0.0 for c in CURRENCIES})
    total_invested_capital: dict = field(default_factory=lambda: {c: 0.0 for c in CURRENCIES})
    withdrawn_capital: dict = field(default_factory=lambda: {c: 0.0 for c in CURRENCIES})
    warnings: list = field(default_factory=list)


def sweep_available_cash(avail_cash: dict, cur: str, amount: float, fx: float) -> float:
    """
    Deduct `amount` (in currency `cur`) from `avail_cash[cur]` first, then sweep any
    remaining requirement from `avail_cash[other_cur]` converted at `fx` (USD_ILS).
    Returns the uncovered external capital shortfall in `cur` (>= 0.0).
    """
    if amount <= 0.0:
        return 0.0
    used_same = min(avail_cash.get(cur, 0.0), amount)
    avail_cash[cur] = max(0.0, avail_cash.get(cur, 0.0) - used_same)
    rem = max(0.0, amount - used_same)
    if rem > 1e-9 and fx > 0.0:
        other_cur = "ILS" if cur == "USD" else "USD"
        avail_other = avail_cash.get(other_cur, 0.0)
        if avail_other > 1e-9:
            if cur == "USD":
                needed_other = rem * fx
                used_other = min(avail_other, needed_other)
                avail_cash[other_cur] = max(0.0, avail_other - used_other)
                rem = max(0.0, rem - (used_other / fx))
            else:
                needed_other = rem / fx
                used_other = min(avail_other, needed_other)
                avail_cash[other_cur] = max(0.0, avail_other - used_other)
                rem = max(0.0, rem - (used_other * fx))
    return rem if rem > 1e-9 else 0.0


# ─── Israeli Lot Tax Math ─────────────────────────────────────────────────────

def compute_israeli_lot_tax(quantity, buy_price, buy_fee, sell_price, sell_fee, currency,
                            fx_buy, fx_sell, cpi_buy, cpi_sell) -> dict:
    """
    Israeli real capital-gain decomposition for one lot slice. Fees are the native-currency
    amounts already allocated to this slice. Returns ILS amounts.
    """
    q = _scalar_float(quantity)
    fx_b, fx_s = _scalar_float(fx_buy), _scalar_float(fx_sell)
    cpi_b, cpi_s = _scalar_float(cpi_buy), _scalar_float(cpi_sell)
    is_usd = str(currency).upper() == "USD"

    conv_buy = fx_b if is_usd else 1.0
    conv_sell = fx_s if is_usd else 1.0
    if is_usd and (fx_b <= 0.0 or fx_s <= 0.0):
        raise ValueError(f"Non-positive USD/ILS rate (buy={fx_b}, sell={fx_s}).")
    if not is_usd and (cpi_b <= 0.0 or cpi_s <= 0.0):
        raise ValueError(f"Non-positive CPI (buy={cpi_b}, sell={cpi_s}).")

    original_cost = (q * _scalar_float(buy_price) + _scalar_float(buy_fee)) * conv_buy
    proceeds = (q * _scalar_float(sell_price) - _scalar_float(sell_fee)) * conv_sell
    ratio = max(1.0, fx_s / fx_b) if is_usd else max(1.0, cpi_s / cpi_b)
    adjusted_cost = original_cost * ratio

    if proceeds >= adjusted_cost:
        real_gain, inflationary_gain, real_loss = proceeds - adjusted_cost, adjusted_cost - original_cost, 0.0
    elif proceeds >= original_cost:
        real_gain, inflationary_gain, real_loss = 0.0, proceeds - original_cost, 0.0
    else:
        real_gain, inflationary_gain, real_loss = 0.0, 0.0, proceeds - original_cost

    return {
        "original_cost_ils": _scalar_float(original_cost),
        "proceeds_ils": _scalar_float(proceeds),
        "nominal_gain_ils": _scalar_float(proceeds - original_cost),
        "inflationary_ratio": _scalar_float(ratio, 1.0),
        "adjusted_cost_ils": _scalar_float(adjusted_cost),
        "real_gain_ils": _scalar_float(real_gain),
        "inflationary_gain_ils": _scalar_float(inflationary_gain),
        "real_loss_ils": _scalar_float(real_loss),
        "tax_ils": _scalar_float(CAPITAL_GAINS_TAX_RATE * real_gain),
    }


# ─── Ledger ───────────────────────────────────────────────────────────────────

def _default_macro_provider(start_date, end_date) -> pd.DataFrame:
    from price_resolver import resolve_macro_series
    return resolve_macro_series(start_date, end_date)


class PortfolioLedger:
    """
    Persistent transaction ledger with FIFO tax lots and Israeli capital-gains tax reporting.

    Parameters:
      path: Parquet file for transaction persistence (atomic tmp + os.replace writes).
      macro_provider: callable(start, end) -> DataFrame[USD_ILS, IL_CPI] on a daily DatetimeIndex.
      carry_forward_losses: carry unused real capital losses into later tax years (Israeli law: yes).
      offset_dividends: offset leftover current-year capital losses against dividend income.
    """

    def __init__(self, path: str = DEFAULT_LEDGER_PATH, macro_provider=None,
                 carry_forward_losses: bool = True, offset_dividends: bool = False,
                 price_provider=None):
        self.path = path
        self.macro_provider = macro_provider or _default_macro_provider
        self.price_provider = price_provider
        self.carry_forward_losses = bool(carry_forward_losses)
        self.offset_dividends = bool(offset_dividends)
        self._tx = self._load()
        self._state = None
        self._macro = None
        self._snapshot_cache = {}

    # ── Persistence ──────────────────────────────────────────────────────────

    @staticmethod
    def _empty_frame() -> pd.DataFrame:
        return pd.DataFrame({
            "tx_id": pd.Series(dtype="int64"), "date": pd.Series(dtype="object"),
            "ticker": pd.Series(dtype="object"), "tx_type": pd.Series(dtype="object"),
            "quantity": pd.Series(dtype="float64"), "price": pd.Series(dtype="float64"),
            "fee": pd.Series(dtype="float64"), "currency": pd.Series(dtype="object"),
            "notes": pd.Series(dtype="object"),
        })

    @staticmethod
    def _normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
        df = df[LEDGER_COLUMNS].copy()
        df["tx_id"] = df["tx_id"].astype("int64")
        for c in ("quantity", "price", "fee"):
            df[c] = df[c].astype("float64")
        for c in ("date", "ticker", "tx_type", "currency", "notes"):
            df[c] = df[c].fillna("").astype(str)
        return df.sort_values(["date", "tx_id"], kind="mergesort").reset_index(drop=True)

    def _load(self) -> pd.DataFrame:
        if not os.path.exists(self.path):
            return self._empty_frame()
        df = pd.read_parquet(self.path, engine="pyarrow")
        missing = set(LEDGER_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"Ledger file {self.path} is missing columns {sorted(missing)}.")
        return self._normalize_frame(df)

    def _persist(self) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        tmp_path = self.path + ".tmp"
        self._tx.to_parquet(tmp_path, engine="pyarrow", compression="snappy", index=False)
        os.replace(tmp_path, self.path)

    def _commit(self, candidate: pd.DataFrame) -> None:
        """Validate a candidate ledger by full FIFO replay; persist only if replay succeeds."""
        candidate = self._normalize_frame(candidate)
        previous = (self._tx, self._state, self._macro)
        self._tx, self._state, self._macro = candidate, None, None
        self._snapshot_cache.clear()
        try:
            self.rebuild()
        except Exception:
            self._tx, self._state, self._macro = previous
            raise
        self._persist()

    # ── Mutations ────────────────────────────────────────────────────────────

    def _next_tx_id(self) -> int:
        return int(self._tx["tx_id"].max()) + 1 if not self._tx.empty else 1

    def add_transaction(self, date, ticker, tx_type, quantity, price=1.0, fee=0.0, currency="USD", notes="") -> int:
        """Validate, FIFO-check (e.g. no overselling), persist. Returns the new tx_id."""
        tx = Transaction.build(self._next_tx_id(), date, ticker, tx_type, quantity, price, fee, currency, notes)
        self._commit(pd.concat([self._tx, pd.DataFrame([asdict(tx)])], ignore_index=True))
        return tx.tx_id

    def remove_transaction(self, tx_id: int) -> None:
        if int(tx_id) not in set(self._tx["tx_id"]):
            raise KeyError(f"tx_id {tx_id} not found.")
        self._commit(self._tx[self._tx["tx_id"] != int(tx_id)])

    def delete_transaction(self, tx_id: int) -> bool:
        """Safe transaction deletion returning True on success, False if tx_id not found."""
        if int(tx_id) not in set(self._tx["tx_id"]):
            return False
        self.remove_transaction(tx_id)
        return True

    def clear(self) -> None:
        """Clear all transactions from the ledger and persist empty state."""
        self._commit(self._empty_frame())

    def import_csv(self, csv_source: str, replace: bool = False, merge: bool = False, return_report: bool = False):
        """
        Fault-tolerant, schema-adaptive CSV transaction importer supporting both internal AlphaQuant CSVs
        and third-party brokerage exports (e.g., Yahoo Finance `portfolio.csv`).

        Features:
          - Fuzzy case-insensitive column resolution with priority rules (`Trade Date` > `Date`, `Purchase Price` > `Price`).
          - Flexible date parsing (`YYYYMMDD` integers/floats, `YYYY-MM-DD`, `YYYY/MM/DD`, `DD/MM/YYYY`).
          - Chronological & same-day ordering (`BUY` before `SELL` on identical dates).
          - Fractional broker rounding tolerance (`<= 0.01` shares clamped to open lot balance).
          - Row-by-row diagnostic reporting with 1-based CSV line numbers and missing/invalid field details.

        Returns `int` (imported count) when `return_report=False`, or a diagnostic dictionary when `return_report=True`.
        """
        import io
        if isinstance(csv_source, str) and ("\n" in csv_source or "," in csv_source) and not os.path.exists(csv_source):
            src = io.StringIO(csv_source)
        else:
            src = csv_source

        raw = pd.read_csv(src, dtype=str, keep_default_na=False)
        col_map, label_map = _resolve_csv_columns(raw.columns)

        should_replace = replace and not merge
        keep_ids = should_replace and ("tx_id" in col_map)
        next_id = 1 if should_replace else self._next_tx_id()

        total_rows = 0
        candidates = []
        skipped_rows = []
        today_norm = pd.Timestamp.now().normalize()

        for idx, r in enumerate(raw.to_dict(orient="records")):
            line_num = idx + 2  # 1-based line number (header is line 1)
            if all(_is_blank_csv_val(v) for v in r.values()):
                continue
            total_rows += 1

            raw_date = r.get(col_map["date"], "")
            raw_ticker = r.get(col_map["ticker"], "")
            raw_tx_type = r.get(col_map["tx_type"], "")
            raw_qty = r.get(col_map["quantity"], "")
            raw_price = r.get(col_map["price"], "")
            raw_fee = r.get(col_map["fee"], "") if "fee" in col_map else ""
            raw_ccy = r.get(col_map["currency"], "") if "currency" in col_map else ""
            raw_notes = r.get(col_map["notes"], "") if "notes" in col_map else ""
            raw_tx_id = r.get(col_map["tx_id"], "") if "tx_id" in col_map else ""

            tx_type_norm = _normalize_tx_type(raw_tx_type)
            symbol_str = "" if _is_blank_csv_val(raw_ticker) else str(raw_ticker).strip().upper()
            if tx_type_norm in CASH_TX_TYPES and not symbol_str:
                symbol_str = CASH_TICKER

            missing_fields = []

            # 1. Validate Date
            date_str = _parse_flexible_date(raw_date)
            if not date_str:
                missing_fields.append(label_map["date"])
            elif pd.Timestamp(date_str).normalize() > today_norm:
                missing_fields.append(f"{label_map['date']} (future: {date_str})")

            # 2. Validate Transaction Type
            if not tx_type_norm:
                if _is_blank_csv_val(raw_tx_type):
                    missing_fields.append(label_map["tx_type"])
                else:
                    missing_fields.append(f"{label_map['tx_type']} ('{str(raw_tx_type).strip()}')")

            # 3. Validate Ticker / Symbol
            if tx_type_norm not in CASH_TX_TYPES and (not symbol_str or symbol_str == CASH_TICKER):
                missing_fields.append(label_map["ticker"])

            # 4. Validate Quantity
            qty_val = float("nan")
            if _is_blank_csv_val(raw_qty):
                missing_fields.append(label_map["quantity"])
            else:
                qty_val = _scalar_float(raw_qty, float("nan"))
                if math.isnan(qty_val) or qty_val <= 0.0:
                    missing_fields.append(f"{label_map['quantity']} ({str(raw_qty).strip()})")

            # 5. Validate Price (allow zero/blank price on valid BUY candidate for Pass A split-row check)
            price_val = 1.0
            is_zero_price_buy = False
            zero_price_missing_fields = []
            zero_price_reason = ""
            if tx_type_norm not in CASH_TX_TYPES:
                if _is_blank_csv_val(raw_price):
                    if tx_type_norm == "BUY" and not missing_fields:
                        is_zero_price_buy = True
                        price_val = 0.0
                        zero_price_missing_fields = [label_map["price"]]
                        zero_price_reason = f"Missing {label_map['price']}"
                    else:
                        missing_fields.append(label_map["price"])
                else:
                    price_val = _scalar_float(raw_price, float("nan"))
                    if math.isnan(price_val) or price_val <= 0.0:
                        if tx_type_norm == "BUY" and not missing_fields and not math.isnan(price_val) and price_val == 0.0:
                            is_zero_price_buy = True
                            price_val = 0.0
                            mf_str = f"{label_map['price']} ({str(raw_price).strip()})"
                            zero_price_missing_fields = [mf_str]
                            zero_price_reason = f"Missing/Invalid {mf_str}"
                        else:
                            missing_fields.append(f"{label_map['price']} ({str(raw_price).strip()})")

            if missing_fields:
                is_watchlist_row = (
                    _is_blank_csv_val(raw_date)
                    and _is_blank_csv_val(raw_qty)
                    and _is_blank_csv_val(raw_price)
                )
                if is_watchlist_row:
                    reason = f"Ignored Watchlist Row (Missing {', '.join(missing_fields)})"
                else:
                    has_invalid_val = any("(" in mf for mf in missing_fields)
                    prefix = "Missing/Invalid" if (len(missing_fields) == 1 and has_invalid_val) else "Missing"
                    reason = f"{prefix} {', '.join(missing_fields)}"
                skipped_rows.append({
                    "line": int(line_num),
                    "symbol": symbol_str or "UNKNOWN",
                    "missing_fields": list(missing_fields),
                    "reason": reason,
                })
                continue

            fee_val = 0.0 if _is_blank_csv_val(raw_fee) else max(0.0, _scalar_float(raw_fee, 0.0))
            if _is_blank_csv_val(raw_ccy):
                ccy_val = "ILS" if symbol_str.endswith(".TA") else "USD"
            else:
                ccy_val = str(raw_ccy).strip().upper()
                if ccy_val not in CURRENCIES:
                    ccy_val = "ILS" if symbol_str.endswith(".TA") else "USD"
            if symbol_str.endswith(".TA") and ccy_val != "ILS":
                ccy_val = "ILS"

            notes_val = "" if _is_blank_csv_val(raw_notes) else str(raw_notes).strip()

            candidates.append({
                "line": int(line_num),
                "date": date_str,
                "ticker": CASH_TICKER if tx_type_norm in CASH_TX_TYPES else symbol_str,
                "tx_type": tx_type_norm,
                "quantity": float(qty_val),
                "price": float(price_val),
                "fee": float(fee_val),
                "currency": ccy_val,
                "notes": notes_val,
                "raw_tx_id": raw_tx_id,
                "is_zero_price_buy": is_zero_price_buy,
                "zero_price_missing_fields": zero_price_missing_fields,
                "zero_price_reason": zero_price_reason,
            })

        # Sort chronologically with same-day BUY before SELL priority
        candidates.sort(key=lambda c: (c["date"], TX_TYPE_PRIORITY.get(c["tx_type"], 9), c["line"]))

        base = self._empty_frame() if should_replace else self._tx.copy()

        # Combine existing base rows + sorted candidate rows for chronological corporate-action & FIFO passes
        combined_events = []
        if not base.empty:
            for b_row in base.to_dict(orient="records"):
                combined_events.append({
                    "is_existing": True,
                    "order_key": (str(b_row["date"]), TX_TYPE_PRIORITY.get(str(b_row["tx_type"]), 9), 0, int(b_row["tx_id"])),
                    "item": b_row,
                })
        for cand in candidates:
            combined_events.append({
                "is_existing": False,
                "order_key": (cand["date"], TX_TYPE_PRIORITY.get(cand["tx_type"], 9), 1, cand["line"]),
                "item": cand,
            })
        combined_events.sort(key=lambda ev: ev["order_key"])

        # Pass A: Zero-Price Split Row Folding (e.g. Yahoo Finance `XLK` BUY 12 @ 0.0 on split date)
        open_lots_pass_a = {}  # ticker -> list of {"item": dict, "qty_open": float}
        post_fold_events = []
        folded_split_count = 0

        for ev in combined_events:
            it = ev["item"]
            t = str(it["ticker"])
            tt = str(it["tx_type"])
            q = float(it["quantity"])

            if not ev["is_existing"] and it.get("is_zero_price_buy", False):
                lots_for_t = open_lots_pass_a.get(t, [])
                held = sum(l["qty_open"] for l in lots_for_t)
                if held > QTY_EPSILON and q > 0.0:
                    split_ratio = (held + q) / held
                    for l_entry in lots_for_t:
                        lot_item = l_entry["item"]
                        orig_lot_q = float(lot_item["quantity"])
                        open_lot_q = float(l_entry["qty_open"])
                        if abs(open_lot_q - orig_lot_q) <= QTY_EPSILON:
                            lot_item["quantity"] = orig_lot_q * split_ratio
                            lot_item["price"] = float(lot_item["price"]) / split_ratio
                            l_entry["qty_open"] = open_lot_q * split_ratio
                        else:
                            # Only scale the unconsumed open portion if partially sold earlier
                            sold_q = max(0.0, orig_lot_q - open_lot_q)
                            new_total_q = sold_q + (open_lot_q * split_ratio)
                            if new_total_q > 0.0:
                                lot_item["price"] = (orig_lot_q * float(lot_item["price"])) / new_total_q
                                lot_item["quantity"] = new_total_q
                            l_entry["qty_open"] = open_lot_q * split_ratio
                    folded_split_count += 1
                    continue
                else:
                    skipped_rows.append({
                        "line": int(it["line"]),
                        "symbol": t,
                        "missing_fields": list(it.get("zero_price_missing_fields") or [f"{label_map['price']} (0.0)"]),
                        "reason": str(it.get("zero_price_reason") or f"Missing/Invalid {label_map['price']} (0.0)"),
                    })
                    continue

            if tt == "BUY":
                open_lots_pass_a.setdefault(t, []).append({"item": it, "qty_open": q})
            elif tt == "SELL":
                lots_for_t = open_lots_pass_a.get(t, [])
                rem = q
                while rem > QTY_EPSILON and lots_for_t:
                    head = lots_for_t[0]
                    take = min(head["qty_open"], rem)
                    head["qty_open"] -= take
                    rem -= take
                    if head["qty_open"] <= QTY_EPSILON:
                        lots_for_t.pop(0)
            post_fold_events.append(ev)

        # Pass B: Automatic Pre-Split Transaction Basis Alignment (e.g. closed pre-split NVDA / TQQQ / NVDL lots)
        # Skip automatic market-price lookup only on internal tx_id round-trips when no custom price_provider is set
        if not keep_ids or self.price_provider is not None:
            sec_items = [
                ev["item"] for ev in post_fold_events
                if str(ev["item"].get("tx_type")) in ("BUY", "SELL")
                and str(ev["item"].get("ticker", "")) not in ("", CASH_TICKER)
            ]
            if sec_items:
                sec_tickers = sorted({str(it["ticker"]) for it in sec_items})
                min_dt = min(str(it["date"]) for it in sec_items)
                max_dt = max(str(it["date"]) for it in sec_items)
                px_df = pd.DataFrame()
                try:
                    if self.price_provider is not None:
                        px_df, _ = self.price_provider(sec_tickers, min_dt, max_dt)
                    else:
                        from price_resolver import resolve_prices
                        px_df, _ = resolve_prices(
                            sec_tickers, min_dt, max_dt,
                            allow_partial=True,
                            local_only=True,
                        )
                except Exception:
                    px_df = pd.DataFrame()
                if px_df is not None and not px_df.empty:
                    align_pre_split_transactions(sec_items, px_df)
                    del px_df
                _release_arrow_memory()

        # Pass C: Pre-validate FIFO open lot quantities across base + candidates so >0.01 oversells are skipped gracefully
        open_shares = {}
        accepted_candidates = []
        updated_existing = []
        for ev in post_fold_events:
            it = ev["item"]
            t = str(it["ticker"])
            tt = str(it["tx_type"])
            q = float(it["quantity"])
            if tt == "BUY":
                open_shares[t] = open_shares.get(t, 0.0) + q
                if ev["is_existing"]:
                    updated_existing.append(it)
                else:
                    accepted_candidates.append(it)
            elif tt == "SELL":
                held = open_shares.get(t, 0.0)
                if held > QTY_EPSILON and held < q <= held + ROUNDING_TOLERANCE:
                    it["quantity"] = held
                    open_shares[t] = 0.0
                    if ev["is_existing"]:
                        updated_existing.append(it)
                    else:
                        accepted_candidates.append(it)
                elif q <= held + QTY_EPSILON:
                    open_shares[t] = max(0.0, held - q)
                    if ev["is_existing"]:
                        updated_existing.append(it)
                    else:
                        accepted_candidates.append(it)
                else:
                    if not ev["is_existing"]:
                        skipped_rows.append({
                            "line": int(it["line"]),
                            "symbol": t,
                            "missing_fields": ["Insufficient Open Shares"],
                            "reason": f"Oversell: attempted to sell {q:g} shares on {it['date']} but only {held:g} available",
                        })
                    else:
                        open_shares[t] = 0.0
                        updated_existing.append(it)
            else:
                if ev["is_existing"]:
                    updated_existing.append(it)
                else:
                    accepted_candidates.append(it)

        if not base.empty and updated_existing:
            base = pd.DataFrame(updated_existing, columns=LEDGER_COLUMNS)

        rows = []
        for i, c in enumerate(accepted_candidates):
            if keep_ids and not _is_blank_csv_val(c["raw_tx_id"]):
                tx_id = int(float(c["raw_tx_id"]))
            else:
                tx_id = next_id + i
            rows.append(asdict(Transaction.build(
                tx_id, c["date"], c["ticker"], c["tx_type"], c["quantity"],
                c["price"], c["fee"], c["currency"], c["notes"]
            )))

        new_df = pd.DataFrame(rows, columns=LEDGER_COLUMNS) if rows else self._empty_frame()
        if keep_ids and not new_df.empty and new_df["tx_id"].duplicated().any():
            raise ValueError("CSV contains duplicate tx_id values.")
        if should_replace or not new_df.empty:
            self._commit(pd.concat([base, new_df], ignore_index=True))

        skipped_rows.sort(key=lambda x: x["line"])
        if return_report:
            return {
                "status": "success" if len(skipped_rows) == 0 else "partial",
                "imported_count": len(rows),
                "total_rows": total_rows,
                "skipped_count": len(skipped_rows),
                "skipped_rows": skipped_rows,
            }
        return len(rows)

    def export_csv(self, csv_path: str = None) -> str:
        """
        Deterministic CSV export: fixed column order, sorted by (date, tx_id), '\\n' line endings.
        If csv_path is provided, writes to disk; otherwise returns CSV string.
        """
        if csv_path:
            self._tx.to_csv(csv_path, index=False, columns=LEDGER_COLUMNS, lineterminator="\n", float_format="%.10g")
            return ""
        return self._tx.to_csv(index=False, columns=LEDGER_COLUMNS, lineterminator="\n", float_format="%.10g")

    def transactions(self) -> pd.DataFrame:
        return self._tx.copy()

    # ── Macro Lookup ─────────────────────────────────────────────────────────

    def _ensure_macro(self, extra_date=None) -> None:
        dates = list(self._tx["date"]) + ([extra_date] if extra_date is not None else [])
        if not dates:
            return
        start = pd.Timestamp(min(dates)).normalize()
        end = pd.Timestamp(max(dates)).normalize()
        if self._macro is not None and self._macro.index.min() <= start and self._macro.index.max() >= end:
            return
        macro = self.macro_provider(start, end)
        if macro is None or macro.empty or not {"USD_ILS", "IL_CPI"}.issubset(macro.columns):
            raise RuntimeError("macro_provider returned no USD_ILS / IL_CPI data.")
        self._macro = macro[["USD_ILS", "IL_CPI"]].astype("float64").sort_index()

    def _macro_on(self, date) -> tuple:
        ts = pd.Timestamp(date).normalize()
        row = self._macro.loc[:ts].tail(1)
        if row.empty:
            raise RuntimeError(f"No USD_ILS / IL_CPI value known on or before {ts.date()}.")
        fx, cpi = _scalar_float(row["USD_ILS"]), _scalar_float(row["IL_CPI"])
        if fx <= 0.0 or cpi <= 0.0:
            raise RuntimeError(f"Invalid macro values on {ts.date()}: USD_ILS={fx}, IL_CPI={cpi}.")
        return fx, cpi

    # ── FIFO Replay ──────────────────────────────────────────────────────────

    def rebuild(self) -> LedgerState:
        """Replay all transactions chronologically (date, tx_id) and rebuild FIFO lots, realized PnL, cash."""
        self._state = self._replay(self._tx)
        return self._state

    def _state_as_of(self, as_of_ts: pd.Timestamp) -> LedgerState:
        """Ledger state including only transactions dated <= as_of (full cached state if nothing is excluded)."""
        as_of_str = pd.Timestamp(as_of_ts).strftime("%Y-%m-%d")
        if self._tx.empty or self._tx["date"].max() <= as_of_str:
            return self._get_state()
        return self._replay(self._tx[self._tx["date"] <= as_of_str])

    def _replay(self, tx_frame: pd.DataFrame) -> LedgerState:
        """FIFO replay of `tx_frame` (already sorted by date, tx_id). Does not mutate cached state."""
        state = LedgerState()
        self._ensure_macro()

        for r in tx_frame.itertuples(index=False):
            fx, cpi = self._macro_on(r.date)
            q, p, fee, cur = float(r.quantity), float(r.price), float(r.fee), r.currency

            if r.tx_type == "CASH_DEPOSIT":
                net_dep = max(0.0, q - fee)
                state.available_cash[cur] += net_dep
                state.total_invested_capital[cur] += net_dep
            elif r.tx_type == "CASH_WITHDRAWAL":
                amt = q + fee
                sweep_available_cash(state.available_cash, cur, amt, fx)
                state.withdrawn_capital[cur] += amt
            elif r.tx_type == "BUY":
                state.open_lots.setdefault(r.ticker, deque()).append(
                    TaxLot(int(r.tx_id), r.ticker, cur, r.date, q, q, p, fee, fx, cpi))
                cost = q * p + fee
                shortfall = sweep_available_cash(state.available_cash, cur, cost, fx)
                state.total_invested_capital[cur] += shortfall
            elif r.tx_type == "SELL":
                actual_sell_qty = self._consume_fifo(state, r, fx, cpi)
                net_proceeds = max(0.0, actual_sell_qty * p - fee)
                state.available_cash[cur] += net_proceeds
            elif r.tx_type == "DIVIDEND":
                gross_native = q * p
                gross_ils = gross_native * (fx if cur == "USD" else 1.0)
                withholding_native = DIVIDEND_WITHHOLDING_RATE * gross_native
                net_native = gross_native - withholding_native - fee
                state.dividends.append({
                    "tx_id": int(r.tx_id), "date": r.date, "tax_year": int(r.date[:4]), "ticker": r.ticker,
                    "currency": cur, "fx": fx, "gross_native": gross_native, "gross_ils": gross_ils,
                    "withholding_ils": DIVIDEND_WITHHOLDING_RATE * gross_ils,
                    "fee_native": fee, "net_native": net_native,
                })
                state.available_cash[cur] += max(0.0, net_native)

            # Ensure cash balance reflects non-negative available cash across both currency buckets
            state.cash["USD"] = max(0.0, state.available_cash["USD"])
            state.cash["ILS"] = max(0.0, state.available_cash["ILS"])

        for w in state.warnings:
            warnings.warn(w)
        return state

    @staticmethod
    def _consume_fifo(state: LedgerState, r, fx_sell: float, cpi_sell: float) -> float:
        lots = state.open_lots.get(r.ticker)
        held = sum(l.qty_open for l in lots) if lots else 0.0
        sell_qty = float(r.quantity)
        if held > QTY_EPSILON and held < sell_qty <= held + ROUNDING_TOLERANCE:
            sell_qty = held
        if sell_qty > held + QTY_EPSILON:
            raise ValueError(f"{r.date} tx {r.tx_id}: SELL {sell_qty} {r.ticker} exceeds open position {held}.")

        remaining = sell_qty
        sell_fee_per_share = float(r.fee) / sell_qty
        while remaining > QTY_EPSILON:
            lot = lots[0]
            if lot.currency != r.currency:
                raise ValueError(f"tx {r.tx_id}: SELL currency {r.currency} != lot {lot.lot_id} currency {lot.currency}.")
            take = min(lot.qty_open, remaining)
            metrics = compute_israeli_lot_tax(
                take, lot.price, lot.fee_per_share * take, float(r.price), sell_fee_per_share * take,
                lot.currency, lot.fx_buy, fx_sell, lot.cpi_buy, cpi_sell)
            state.realized.append({
                "sell_tx_id": int(r.tx_id), "buy_tx_id": lot.lot_id, "ticker": r.ticker, "currency": lot.currency,
                "buy_date": lot.buy_date, "sell_date": r.date, "tax_year": int(r.date[:4]), "quantity": take,
                "buy_price": lot.price, "sell_price": float(r.price), "fx_buy": lot.fx_buy, "fx_sell": fx_sell,
                "cpi_buy": lot.cpi_buy, "cpi_sell": cpi_sell, **metrics,
            })
            lot.qty_open -= take
            remaining -= take
            if lot.qty_open <= QTY_EPSILON:
                lots.popleft()
        if not lots:
            state.open_lots.pop(r.ticker, None)
        return sell_qty

    def _get_state(self) -> LedgerState:
        return self._state if self._state is not None else self.rebuild()

    # ── Reports ──────────────────────────────────────────────────────────────

    def realized_lots(self) -> pd.DataFrame:
        return _clean_frame(pd.DataFrame(self._get_state().realized))

    def dividends(self) -> pd.DataFrame:
        return _clean_frame(pd.DataFrame(self._get_state().dividends))

    def cash_balances(self, state: LedgerState = None) -> dict:
        state = state if state is not None else self._get_state()
        return {c: max(0.0, _scalar_float(v)) for c, v in state.available_cash.items()}

    def capital_buckets(self, state: LedgerState = None) -> dict:
        state = state if state is not None else self._get_state()
        return {
            "available_cash": {c: max(0.0, _scalar_float(v)) for c, v in state.available_cash.items()},
            "total_invested_capital": {c: max(0.0, _scalar_float(v)) for c, v in state.total_invested_capital.items()},
            "withdrawn_capital": {c: max(0.0, _scalar_float(v)) for c, v in state.withdrawn_capital.items()},
        }

    def tax_summary(self, year: int = None, state: LedgerState = None) -> pd.DataFrame:
        """
        Per-tax-year Israeli capital gains summary (ILS). Current-year losses offset gains first,
        then carried-forward losses (if enabled). Optionally leftover current-year losses offset dividends.
        `state` defaults to the full-ledger state (used internally for as-of snapshots).
        """
        state = state if state is not None else self._get_state()
        realized = pd.DataFrame(state.realized)
        divs = pd.DataFrame(state.dividends)
        years = sorted((set(realized["tax_year"]) if not realized.empty else set())
                       | (set(divs["tax_year"]) if not divs.empty else set()))

        rows, carry = [], 0.0
        for y in years:
            ry = realized[realized["tax_year"] == y] if not realized.empty else realized
            dy = divs[divs["tax_year"] == y] if not divs.empty else divs
            gains = _scalar_float(ry["real_gain_ils"].sum()) if not ry.empty else 0.0
            losses = -_scalar_float(ry["real_loss_ils"].sum()) if not ry.empty else 0.0
            exempt = _scalar_float(ry["inflationary_gain_ils"].sum()) if not ry.empty else 0.0
            div_gross = _scalar_float(dy["gross_ils"].sum()) if not dy.empty else 0.0
            div_withheld = _scalar_float(dy["withholding_ils"].sum()) if not dy.empty else 0.0

            carry_in = carry if self.carry_forward_losses else 0.0
            shield_current = min(gains, losses)
            shield_carry = min(gains - shield_current, carry_in)
            taxable = gains - shield_current - shield_carry
            leftover_current = losses - shield_current

            div_offset = min(leftover_current, div_gross) if self.offset_dividends else 0.0
            leftover_current -= div_offset
            carry = (carry_in - shield_carry + leftover_current) if self.carry_forward_losses else 0.0

            rows.append({
                "tax_year": int(y),
                "real_gains_ils": gains,
                "real_losses_ils": losses,
                "inflationary_exempt_ils": exempt,
                "loss_carry_in_ils": carry_in,
                "shield_used_current_year_ils": shield_current,
                "shield_used_carry_forward_ils": shield_carry,
                "taxable_real_gain_ils": taxable,
                "capital_gains_tax_ils": CAPITAL_GAINS_TAX_RATE * taxable,
                "dividend_gross_ils": div_gross,
                "dividend_withholding_ils": div_withheld,
                "dividend_loss_offset_ils": div_offset,
                "dividend_tax_refund_ils": DIVIDEND_WITHHOLDING_RATE * div_offset,
                "loss_carry_out_ils": carry,
            })

        out = _clean_frame(pd.DataFrame(rows))
        if year is not None and not out.empty:
            out = out[out["tax_year"] == int(year)].reset_index(drop=True)
        return out

    def snapshot(self, as_of=None, market_prices: dict = None, price_provider=None) -> dict:
        """
        Portfolio valuation as of `as_of` (default: today), using only transactions dated <= as_of.

        market_prices: optional {ticker: native price} override; otherwise prices come from
        `price_provider(tickers, start, end)` (default: price_resolver.resolve_prices, last row).
        Returns dict with 'positions' (DataFrame), 'lots' (DataFrame), 'tlh_candidates' (DataFrame), 'totals' (dict).
        """
        as_of_ts = pd.Timestamp(as_of).normalize() if as_of is not None else pd.Timestamp.now().normalize()
        cache_key = (len(self._tx), as_of_ts.strftime("%Y-%m-%d"))
        if market_prices is None and cache_key in self._snapshot_cache:
            return self._snapshot_cache[cache_key]
        state = self._state_as_of(as_of_ts)
        self._ensure_macro(extra_date=as_of_ts.strftime("%Y-%m-%d"))
        fx_now, cpi_now = self._macro_on(as_of_ts)

        tickers = sorted(state.open_lots.keys())
        prices = {t: _scalar_float((market_prices or {}).get(t), float("nan")) for t in tickers}
        need = [t for t in tickers if not prices[t] > 0.0]
        if need:
            if price_provider is None:
                from price_resolver import resolve_prices
                px_df, _ = resolve_prices(need, as_of_ts - pd.Timedelta(days=10), as_of_ts, allow_partial=True)
            else:
                px_df, _ = price_provider(need, as_of_ts - pd.Timedelta(days=10), as_of_ts)
            for t in need:
                prices[t] = _scalar_float(px_df[t].iloc[-1]) if t in px_df.columns else 0.0
                if not prices[t] > 0.0 and state.open_lots.get(t):
                    prices[t] = _scalar_float(state.open_lots[t][-1].price, 0.0)
            bad = [t for t in need if not prices[t] > 0.0]
            if bad:
                raise RuntimeError(f"No market price available for {bad} as of {as_of_ts.date()}.")

        lot_rows = []
        for t in tickers:
            for lot in state.open_lots[t]:
                m = compute_israeli_lot_tax(lot.qty_open, lot.price, lot.fee_per_share * lot.qty_open, prices[t], 0.0,
                                            lot.currency, lot.fx_buy, fx_now, lot.cpi_buy, cpi_now)
                cost_native = lot.qty_open * lot.price + lot.fee_per_share * lot.qty_open
                cost_usd = cost_native if lot.currency == "USD" else cost_native / lot.fx_buy
                lot_rows.append({
                    "ticker": t, "lot_id": lot.lot_id, "currency": lot.currency, "buy_date": lot.buy_date,
                    "quantity": lot.qty_open, "buy_price": lot.price, "market_price": prices[t],
                    "fx_buy": lot.fx_buy, "fx_now": fx_now, "cpi_buy": lot.cpi_buy, "cpi_now": cpi_now,
                    "cost_native": cost_native, "cost_usd": cost_usd,
                    "market_value_native": lot.qty_open * prices[t], **m,
                })
        lots_df = _clean_frame(pd.DataFrame(lot_rows))

        ytd = self.tax_summary(as_of_ts.year, state=state)
        ytd_taxable = _scalar_float(ytd["taxable_real_gain_ils"].iloc[0]) if not ytd.empty else 0.0
        if not ytd.empty:
            unused_shield = _scalar_float(ytd["loss_carry_out_ils"].iloc[0])
        else:
            prior = self.tax_summary(state=state)
            prior = prior[prior["tax_year"] < as_of_ts.year] if not prior.empty else prior
            unused_shield = _scalar_float(prior["loss_carry_out_ils"].iloc[-1]) if not prior.empty else 0.0

        avail_usd = state.available_cash["USD"] + (state.available_cash["ILS"] / fx_now)
        avail_ils = (state.available_cash["USD"] * fx_now) + state.available_cash["ILS"]
        inv_usd = state.total_invested_capital["USD"] + (state.total_invested_capital["ILS"] / fx_now)
        inv_ils = (state.total_invested_capital["USD"] * fx_now) + state.total_invested_capital["ILS"]
        wdr_usd = state.withdrawn_capital["USD"] + (state.withdrawn_capital["ILS"] / fx_now)
        wdr_ils = (state.withdrawn_capital["USD"] * fx_now) + state.withdrawn_capital["ILS"]

        if lots_df.empty:
            return {"as_of": as_of_ts.strftime("%Y-%m-%d"), "positions": pd.DataFrame(), "lots": lots_df,
                    "tlh_candidates": pd.DataFrame(), "totals": {"cash": self.cash_balances(state),
                    "unused_loss_shield_ils": unused_shield, "est_liquidation_tax_ils": 0.0,
                    "available_cash_usd": _scalar_float(avail_usd), "available_cash_ils": _scalar_float(avail_ils),
                    "invested_capital_usd": _scalar_float(inv_usd), "invested_capital_ils": _scalar_float(inv_ils),
                    "withdrawn_capital_usd": _scalar_float(wdr_usd), "withdrawn_capital_ils": _scalar_float(wdr_ils),
                    "market_value_ils": 0.0, "market_value_usd": 0.0, "cost_ils": 0.0}}

        lots_df["market_value_ils"] = lots_df["proceeds_ils"]
        lots_df["market_value_usd"] = lots_df["market_value_ils"] / fx_now

        pos = lots_df.groupby(["ticker", "currency"], as_index=False).agg(
            quantity=("quantity", "sum"), cost_native=("cost_native", "sum"), cost_usd=("cost_usd", "sum"),
            cost_ils=("original_cost_ils", "sum"), market_price=("market_price", "last"),
            market_value_native=("market_value_native", "sum"), market_value_ils=("market_value_ils", "sum"),
            market_value_usd=("market_value_usd", "sum"), unrealized_nominal_ils=("nominal_gain_ils", "sum"),
            unrealized_real_gain_ils=("real_gain_ils", "sum"), unrealized_real_loss_ils=("real_loss_ils", "sum"),
            unrealized_inflationary_ils=("inflationary_gain_ils", "sum"), est_tax_ils=("tax_ils", "sum"))
        pos["avg_cost_native"] = pos["cost_native"] / pos["quantity"]
        pos["avg_cost_usd"] = pos["cost_usd"] / pos["quantity"]
        pos["avg_cost_ils"] = pos["cost_ils"] / pos["quantity"]
        pos["unrealized_net_after_tax_ils"] = pos["unrealized_nominal_ils"] - pos["est_tax_ils"]
        pos = _clean_frame(pos)

        u_gains = _scalar_float(lots_df["real_gain_ils"].sum())
        u_losses = -_scalar_float(lots_df["real_loss_ils"].sum())
        liq_taxable = max(0.0, u_gains - u_losses - unused_shield)
        liq_tax = CAPITAL_GAINS_TAX_RATE * liq_taxable
        nominal = _scalar_float(lots_df["nominal_gain_ils"].sum())

        tlh = lots_df[lots_df["real_loss_ils"] < 0.0][
            ["ticker", "lot_id", "buy_date", "quantity", "original_cost_ils", "proceeds_ils", "real_loss_ils"]].copy()
        if not tlh.empty:
            tlh["potential_tax_saving_ils"] = CAPITAL_GAINS_TAX_RATE * np.minimum(-tlh["real_loss_ils"], ytd_taxable)
            tlh = _clean_frame(tlh.sort_values("real_loss_ils").reset_index(drop=True))

        totals = {
            "fx_usd_ils": fx_now, "cpi": cpi_now,
            "market_value_ils": _scalar_float(lots_df["market_value_ils"].sum()),
            "market_value_usd": _scalar_float(lots_df["market_value_usd"].sum()),
            "cost_ils": _scalar_float(lots_df["original_cost_ils"].sum()),
            "unrealized_nominal_ils": nominal,
            "unrealized_real_gains_ils": u_gains, "unrealized_real_losses_ils": u_losses,
            "unused_loss_shield_ils": unused_shield,
            "est_liquidation_tax_ils": liq_tax,
            "unrealized_net_after_tax_ils": nominal - liq_tax,
            "ytd_taxable_real_gain_ils": ytd_taxable,
            "cash": self.cash_balances(state),
            "available_cash_usd": _scalar_float(avail_usd),
            "available_cash_ils": _scalar_float(avail_ils),
            "invested_capital_usd": _scalar_float(inv_usd),
            "invested_capital_ils": _scalar_float(inv_ils),
            "withdrawn_capital_usd": _scalar_float(wdr_usd),
            "withdrawn_capital_ils": _scalar_float(wdr_ils),
        }
        res_snap = {"as_of": as_of_ts.strftime("%Y-%m-%d"), "positions": pos, "lots": lots_df,
                    "tlh_candidates": tlh, "totals": totals}
        if market_prices is None:
            self._snapshot_cache[cache_key] = res_snap
        return res_snap
