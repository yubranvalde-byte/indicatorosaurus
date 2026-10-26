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

LEDGER_COLUMNS = ["tx_id", "date", "ticker", "tx_type", "quantity", "price", "fee", "currency", "notes"]


# ─── Numerical Safety ─────────────────────────────────────────────────────────

def _scalar_float(val, default: float = 0.0) -> float:
    """Coerce any scalar / 0-d / 1-element pandas or NumPy value to a finite Python float."""
    if val is None:
        return default
    if hasattr(val, "squeeze"):
        val = val.squeeze()
    if hasattr(val, "item"):
        try:
            val = val.item()
        except (ValueError, AttributeError):
            if hasattr(val, "iloc"):
                val = val.iloc[0]
    try:
        f = float(val)
    except (ValueError, TypeError):
        return default
    return default if (math.isnan(f) or math.isinf(f)) else f


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
    warnings: list = field(default_factory=list)


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
                 carry_forward_losses: bool = True, offset_dividends: bool = False):
        self.path = path
        self.macro_provider = macro_provider or _default_macro_provider
        self.carry_forward_losses = bool(carry_forward_losses)
        self.offset_dividends = bool(offset_dividends)
        self._tx = self._load()
        self._state = None
        self._macro = None

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

    def import_csv(self, csv_path: str, replace: bool = False) -> int:
        """
        Import transactions from CSV (columns: date, ticker, tx_type, quantity, price, fee, currency[, notes, tx_id]).
        replace=True keeps CSV tx_ids (if present) and replaces the ledger; otherwise rows are appended
        with new sequential tx_ids. The whole import is atomic. Returns number of imported rows.
        """
        raw = pd.read_csv(csv_path, dtype={"ticker": str, "notes": str, "currency": str, "tx_type": str, "date": str})
        required = {"date", "ticker", "tx_type", "quantity", "price", "fee", "currency"}
        missing = required - set(raw.columns)
        if missing:
            raise ValueError(f"CSV {csv_path} missing columns {sorted(missing)}.")

        keep_ids = replace and "tx_id" in raw.columns
        next_id = 1 if replace else self._next_tx_id()
        rows = []
        for i, r in enumerate(raw.to_dict(orient="records")):
            tx_id = int(r["tx_id"]) if keep_ids else next_id + i
            try:
                rows.append(asdict(Transaction.build(tx_id, r["date"], r.get("ticker"), r["tx_type"], r["quantity"],
                                                     r["price"], r["fee"], r["currency"], r.get("notes", ""))))
            except ValueError as e:
                raise ValueError(f"CSV row {i + 2}: {e}")

        new_df = pd.DataFrame(rows, columns=LEDGER_COLUMNS) if rows else self._empty_frame()
        if keep_ids and new_df["tx_id"].duplicated().any():
            raise ValueError("CSV contains duplicate tx_id values.")
        base = self._empty_frame() if replace else self._tx
        self._commit(pd.concat([base, new_df], ignore_index=True))
        return len(rows)

    def export_csv(self, csv_path: str) -> None:
        """Deterministic CSV export: fixed column order, sorted by (date, tx_id), '\\n' line endings."""
        self._tx.to_csv(csv_path, index=False, columns=LEDGER_COLUMNS, lineterminator="\n", float_format="%.10g")

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
        state = LedgerState()
        self._ensure_macro()

        for r in self._tx.itertuples(index=False):
            fx, cpi = self._macro_on(r.date)
            q, p, fee, cur = float(r.quantity), float(r.price), float(r.fee), r.currency

            if r.tx_type == "CASH_DEPOSIT":
                state.cash[cur] += q - fee
            elif r.tx_type == "CASH_WITHDRAWAL":
                state.cash[cur] -= q + fee
            elif r.tx_type == "BUY":
                state.open_lots.setdefault(r.ticker, deque()).append(
                    TaxLot(int(r.tx_id), r.ticker, cur, r.date, q, q, p, fee, fx, cpi))
                state.cash[cur] -= q * p + fee
            elif r.tx_type == "SELL":
                self._consume_fifo(state, r, fx, cpi)
                state.cash[cur] += q * p - fee
            elif r.tx_type == "DIVIDEND":
                gross_native = q * p
                gross_ils = gross_native * (fx if cur == "USD" else 1.0)
                withholding_native = DIVIDEND_WITHHOLDING_RATE * gross_native
                state.dividends.append({
                    "tx_id": int(r.tx_id), "date": r.date, "tax_year": int(r.date[:4]), "ticker": r.ticker,
                    "currency": cur, "fx": fx, "gross_native": gross_native, "gross_ils": gross_ils,
                    "withholding_ils": DIVIDEND_WITHHOLDING_RATE * gross_ils,
                    "fee_native": fee, "net_native": gross_native - withholding_native - fee,
                })
                state.cash[cur] += gross_native - withholding_native - fee

            if state.cash[cur] < -QTY_EPSILON:
                state.warnings.append(f"{r.date} tx {r.tx_id}: {cur} cash balance negative ({state.cash[cur]:.2f}).")

        for w in state.warnings:
            warnings.warn(w)
        self._state = state
        return state

    @staticmethod
    def _consume_fifo(state: LedgerState, r, fx_sell: float, cpi_sell: float) -> None:
        lots = state.open_lots.get(r.ticker)
        held = sum(l.qty_open for l in lots) if lots else 0.0
        sell_qty = float(r.quantity)
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

    def _get_state(self) -> LedgerState:
        return self._state if self._state is not None else self.rebuild()

    # ── Reports ──────────────────────────────────────────────────────────────

    def realized_lots(self) -> pd.DataFrame:
        return _clean_frame(pd.DataFrame(self._get_state().realized))

    def dividends(self) -> pd.DataFrame:
        return _clean_frame(pd.DataFrame(self._get_state().dividends))

    def cash_balances(self) -> dict:
        return {c: _scalar_float(v) for c, v in self._get_state().cash.items()}

    def tax_summary(self, year: int = None) -> pd.DataFrame:
        """
        Per-tax-year Israeli capital gains summary (ILS). Current-year losses offset gains first,
        then carried-forward losses (if enabled). Optionally leftover current-year losses offset dividends.
        """
        state = self._get_state()
        realized = pd.DataFrame(state.realized)
        divs = pd.DataFrame(state.dividends)
        years = sorted(set(realized["tax_year"]) if not realized.empty else set()
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
        Current portfolio valuation as of `as_of` (default: today).

        market_prices: optional {ticker: native price} override; otherwise prices come from
        `price_provider(tickers, start, end)` (default: price_resolver.resolve_prices, last row).
        Returns dict with 'positions' (DataFrame), 'lots' (DataFrame), 'tlh_candidates' (DataFrame), 'totals' (dict).
        """
        as_of_ts = pd.Timestamp(as_of).normalize() if as_of is not None else pd.Timestamp.now().normalize()
        state = self._get_state()
        self._ensure_macro(extra_date=as_of_ts.strftime("%Y-%m-%d"))
        fx_now, cpi_now = self._macro_on(as_of_ts)

        tickers = sorted(state.open_lots.keys())
        prices = {t: _scalar_float((market_prices or {}).get(t), float("nan")) for t in tickers}
        need = [t for t in tickers if not prices[t] > 0.0]
        if need:
            if price_provider is None:
                from price_resolver import resolve_prices as price_provider
            px_df, _ = price_provider(need, as_of_ts - pd.Timedelta(days=10), as_of_ts)
            for t in need:
                prices[t] = _scalar_float(px_df[t].iloc[-1]) if t in px_df.columns else 0.0
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

        ytd = self.tax_summary(as_of_ts.year)
        ytd_taxable = _scalar_float(ytd["taxable_real_gain_ils"].iloc[0]) if not ytd.empty else 0.0
        if not ytd.empty:
            unused_shield = _scalar_float(ytd["loss_carry_out_ils"].iloc[0])
        else:
            prior = self.tax_summary()
            prior = prior[prior["tax_year"] < as_of_ts.year] if not prior.empty else prior
            unused_shield = _scalar_float(prior["loss_carry_out_ils"].iloc[-1]) if not prior.empty else 0.0

        if lots_df.empty:
            return {"as_of": as_of_ts.strftime("%Y-%m-%d"), "positions": pd.DataFrame(), "lots": lots_df,
                    "tlh_candidates": pd.DataFrame(), "totals": {"cash": self.cash_balances(),
                    "unused_loss_shield_ils": unused_shield}}

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
            "cash": self.cash_balances(),
        }
        return {"as_of": as_of_ts.strftime("%Y-%m-%d"), "positions": pos, "lots": lots_df,
                "tlh_candidates": tlh, "totals": totals}
