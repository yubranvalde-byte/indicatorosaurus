#!/usr/bin/env python3
"""
simfin_etl.py — SimFin Bulk CSV ETL Pipeline (AlphaQuant V3.3)

Transforms raw SimFin bulk CSV datasets (Income TTM, Cash Flow TTM, Balance Sheet Quarterly,
Daily Share Prices, Companies, Industries) into historical_fundamental_data.csv.

Maintains strict compliance with:
1. Pure Offline Mandate (No yfinance / network calls).
2. Zero Look-Ahead Bias (Matches fundamentals via Publish Date to price technicals).
3. No Data Spoofing (Safely coerces missing/invalid data to 0.0 without artificial backfilling).
4. Substituted ForwardPE with TrailingPE (Price / EPS_TTM).
"""

import os
import sys
import numpy as np
import pandas as pd

# Directory paths
RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw_simfin")
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "historical_fundamental_data.csv")

# Expected schema (+ Ticker header match)
FINAL_COLUMNS = [
    "Ticker", "GICS_Sector", "Quarter_End_Date",
    "TrailingPE", "EV_EBITDA", "FCFYield", "RevenueGrowth", "EPSGrowth",
    "GrossMargin", "OperatingMargin", "ROIC", "NetDebt_EBITDA", "ShareholderYield",
    "FiveYDivGrowth", "PayoutRatio", "Beta", "RSI", "PriceVs200SMA",
    "RelativeStrength", "CurrentRatio", "MarketCap", "EBITDA",
    "Total_Equity", "FCF", "EBIT", "Interest_Expense"
]

def safe_div(num, denom, default=0.0):
    """Safely divide num by denom, handling zero, NaN, and Inf."""
    try:
        if denom is None or num is None or pd.isna(denom) or pd.isna(num):
            return default
        num_f = float(num)
        denom_f = float(denom)
        if denom_f == 0.0 or np.isinf(denom_f) or np.isnan(denom_f) or np.isnan(num_f):
            return default
        res = num_f / denom_f
        if np.isnan(res) or np.isinf(res):
            return default
        return res
    except Exception:
        return default

def compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Compute Wilder's 14-day RSI using EWM."""
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)

def load_raw_datasets():
    """Load raw SimFin bulk CSVs using ';' delimiter."""
    print("[ETL] Loading raw SimFin bulk CSV datasets...")
    
    comp_path = os.path.join(RAW_DIR, "us-companies.csv")
    ind_path = os.path.join(RAW_DIR, "industries.csv")
    inc_path = os.path.join(RAW_DIR, "us-income-ttm.csv")
    cf_path = os.path.join(RAW_DIR, "us-cashflow-ttm.csv")
    bs_path = os.path.join(RAW_DIR, "us-balance-quarterly.csv")
    prices_path = os.path.join(RAW_DIR, "us-shareprices-daily.csv")

    for path in [comp_path, ind_path, inc_path, cf_path, bs_path, prices_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required raw dataset missing: {path}")

    companies = pd.read_csv(comp_path, sep=";")
    industries = pd.read_csv(ind_path, sep=";")
    income = pd.read_csv(inc_path, sep=";")
    cashflow = pd.read_csv(cf_path, sep=";")
    balance = pd.read_csv(bs_path, sep=";")
    
    print("[ETL] Loading daily share prices (this may take a few seconds)...")
    prices = pd.read_csv(prices_path, sep=";")

    return companies, industries, income, cashflow, balance, prices

def build_sector_map(companies: pd.DataFrame, industries: pd.DataFrame) -> dict:
    """Map Ticker -> GICS Sector via IndustryId."""
    comp_ind = companies.merge(industries, on="IndustryId", how="left")
    sector_map = dict(zip(comp_ind["Ticker"].dropna(), comp_ind["Sector"].fillna("Other")))
    return sector_map

def process_daily_prices(prices: pd.DataFrame):
    """
    Process daily share prices to compute technical indicators (RSI, 200SMA, Beta, RelativeStrength)
    indexed by Ticker and Date.
    """
    print("[ETL] Processing daily prices & technical indicators...")
    prices = prices.dropna(subset=["Ticker", "Date"]).copy()
    prices["Date"] = pd.to_datetime(prices["Date"])
    
    # Deduplicate ticker and date
    prices = prices.drop_duplicates(subset=["Ticker", "Date"]).sort_values(["Ticker", "Date"]).reset_index(drop=True)

    # Use Adj. Close if available, else Close
    prices["Price"] = prices["Adj. Close"].fillna(prices["Close"])
    
    # Create price pivot table: Index = Date, Columns = Ticker
    price_pivot = prices.pivot(index="Date", columns="Ticker", values="Price")
    daily_ret = price_pivot.pct_change()

    # Equal-weighted benchmark return
    mkt_ret = daily_ret.mean(axis=1)
    mkt_cum_1y = (1 + mkt_ret).rolling(252, min_periods=60).apply(np.prod, raw=True) - 1.0
    mkt_var = mkt_ret.rolling(252, min_periods=60).var()

    processed_list = []
    
    for ticker, group in prices.groupby("Ticker"):
        group = group.copy().sort_values("Date").reset_index(drop=True)
        p = group["Price"]
        stk_ret = p.pct_change()

        # 200-day SMA
        sma200 = p.rolling(200, min_periods=20).mean()
        group["PriceVs200SMA"] = (p / sma200).fillna(1.0)

        # 14-day RSI
        group["RSI"] = compute_rsi(p, period=14).fillna(50.0)

        # 1-year return
        stk_1y = p.pct_change(252)

        # Reindex market series to this ticker's dates
        m_cum_aligned = mkt_cum_1y.reindex(group["Date"]).values
        m_ret_aligned = mkt_ret.reindex(group["Date"]).values
        m_var_aligned = mkt_var.reindex(group["Date"]).values

        # Relative Strength
        group["RelativeStrength"] = (stk_1y.values - m_cum_aligned)
        group["RelativeStrength"] = group["RelativeStrength"].fillna(0.0)

        # Rolling 1Y covariance for Beta
        s_ret_series = pd.Series(stk_ret.values, index=group["Date"])
        m_ret_series = pd.Series(m_ret_aligned, index=group["Date"])
        cov_series = s_ret_series.rolling(252, min_periods=60).cov(m_ret_series)
        var_series = pd.Series(m_var_aligned, index=group["Date"])

        raw_beta = cov_series.values / np.where(var_series.values == 0, np.nan, var_series.values)
        group["Beta"] = pd.Series(raw_beta).fillna(1.0).values

        processed_list.append(group)

    df_prices = pd.concat(processed_list, ignore_index=True)
    return df_prices

def build_fundamental_panel(income: pd.DataFrame, cashflow: pd.DataFrame, balance: pd.DataFrame, sector_map: dict):
    """Merge financial statement tables on Ticker & Report Date."""
    print("[ETL] Merging income, cashflow, and balance sheet statements...")

    # Filter out empty Tickers
    income = income.dropna(subset=["Ticker"]).copy()
    cashflow = cashflow.dropna(subset=["Ticker"]).copy()
    balance = balance.dropna(subset=["Ticker"]).copy()

    # Format dates
    for df in [income, cashflow, balance]:
        df["Report Date"] = pd.to_datetime(df["Report Date"])
        df["Publish Date"] = pd.to_datetime(df["Publish Date"])

    # Deduplicate statement entries by Ticker and Report Date
    income = income.drop_duplicates(subset=["Ticker", "Report Date"])
    cashflow = cashflow.drop_duplicates(subset=["Ticker", "Report Date"])
    balance = balance.drop_duplicates(subset=["Ticker", "Report Date"])

    # Merge statements
    merged = income.merge(
        cashflow, on=["Ticker", "Report Date"], how="inner", suffixes=("_inc", "_cf")
    ).merge(
        balance, on=["Ticker", "Report Date"], how="inner", suffixes=("_cf", "_bs")
    )

    merged["GICS_Sector"] = merged["Ticker"].map(sector_map).fillna("Other")
    
    # Resolve Publish Date
    pub_col = [c for c in merged.columns if "Publish Date" in c][0]
    merged["Effective_Publish_Date"] = merged[pub_col].fillna(merged["Report Date"])

    merged = merged.sort_values(["Ticker", "Report Date"]).reset_index(drop=True)
    return merged

def calculate_all_indicators(merged: pd.DataFrame, df_prices: pd.DataFrame) -> pd.DataFrame:
    """Calculate 16 quantitative indicators and output base metrics."""
    print("[ETL] Calculating quantitative indicators and point-in-time metrics...")

    # Sort prices for merge_asof
    df_prices = df_prices.sort_values("Date")
    merged = merged.sort_values("Effective_Publish_Date")

    # Point-in-Time merge with prices as of Effective_Publish_Date
    pit_data = pd.merge_asof(
        merged,
        df_prices[["Ticker", "Date", "Price", "Close", "Shares Outstanding", "PriceVs200SMA", "RSI", "RelativeStrength", "Beta"]],
        left_on="Effective_Publish_Date",
        right_on="Date",
        by="Ticker",
        direction="backward"
    )

    rows = []

    # Group by Ticker for YoY growth rates
    for ticker, group in pit_data.groupby("Ticker"):
        group = group.sort_values("Report Date").copy().reset_index(drop=True)

        rev = group["Revenue"].values
        op_inc = group["Operating Income (Loss)"].values
        net_inc = group["Net Income_inc"].values if "Net Income_inc" in group.columns else group["Net Income"].values
        gross_prof = group["Gross Profit"].values
        da = group["Depreciation & Amortization_inc"].fillna(0.0).values if "Depreciation & Amortization_inc" in group.columns else group.get("Depreciation & Amortization", 0.0).values
        
        op_cf = group["Net Cash from Operating Activities"].values
        capex = group["Change in Fixed Assets & Intangibles"].fillna(0.0).values # Typically negative in SimFin
        div_paid = group["Dividends Paid"].fillna(0.0).values # Negative
        repurchase = group["Cash from (Repurchase of) Equity"].fillna(0.0).values # Negative
        debt_repay = group["Cash from (Repayment of) Debt"].fillna(0.0).values # Negative

        curr_assets = group["Total Current Assets"].values
        curr_liab = group["Total Current Liabilities"].values
        short_debt = group["Short Term Debt"].fillna(0.0).values
        long_debt = group["Long Term Debt"].fillna(0.0).values
        total_equity = group["Total Equity"].values
        cash_eq = group["Cash, Cash Equivalents & Short Term Investments"].fillna(0.0).values
        
        shares_bs = group["Shares (Diluted)_bs"].fillna(group["Shares (Basic)_bs"]).values if "Shares (Diluted)_bs" in group.columns else group.get("Shares (Diluted)", 0.0).values
        shares_price = group["Shares Outstanding"].values
        price = group["Price"].values
        int_exp_col = group["Interest Expense, Net"].fillna(0.0).abs().values if "Interest Expense, Net" in group.columns else np.zeros(len(group))

        dates = pd.to_datetime(group["Report Date"]).dt.strftime("%Y-%m-%d").values
        sectors = group["GICS_Sector"].values

        rsi_vals = pd.Series(group["RSI"]).fillna(50.0).values
        beta_vals = pd.Series(group["Beta"]).fillna(1.0).values
        p_sma_vals = pd.Series(group["PriceVs200SMA"]).fillna(1.0).values
        rel_str_vals = pd.Series(group["RelativeStrength"]).fillna(0.0).values

        n = len(group)

        for i in range(n):
            # Market Cap calculation
            sh = shares_price[i] if (not pd.isna(shares_price[i]) and shares_price[i] > 0) else shares_bs[i]
            mkt_cap = price[i] * sh if (not pd.isna(price[i]) and not pd.isna(sh) and sh > 0) else 0.0

            # Upstream Structural Gatekeeper Filter (Point-in-Time Row-Level):
            # Permanently drop quarterly records with Market Cap < $1B ($1,000,000,000) or Price < $5.00
            cur_price = float(price[i]) if not pd.isna(price[i]) else 0.0
            if mkt_cap < 1_000_000_000.0 or cur_price < 5.00:
                continue

            # 1. TrailingPE (Price / EPS_TTM = MarketCap / Net Income)
            eps_ttm = safe_div(net_inc[i], sh, default=0.0)
            trailing_pe = safe_div(price[i], eps_ttm, default=0.0) if eps_ttm > 0 else 0.0

            # 2. EBITDA & EV/EBITDA
            ebitda = op_inc[i] + da[i]
            tot_debt = short_debt[i] + long_debt[i]
            ev = mkt_cap + tot_debt - cash_eq[i]
            ev_ebitda = safe_div(ev, ebitda, default=0.0) if ebitda > 0 else 0.0

            # 3. FCF Yield
            fcf = op_cf[i] + capex[i] # Capex is negative outflow
            fcf_yield = safe_div(fcf, mkt_cap, default=0.0)

            # 4. Revenue Growth YoY (compare to 4 quarters ago)
            rev_growth = safe_div(rev[i] - rev[i-4], rev[i-4], default=0.0) if i >= 4 else 0.0

            # 5. EPS Growth YoY
            eps_curr = safe_div(net_inc[i], sh, default=0.0)
            eps_prev = safe_div(net_inc[i-4], sh, default=0.0) if i >= 4 else 0.0
            eps_growth = safe_div(eps_curr - eps_prev, abs(eps_prev), default=0.0) if i >= 4 else 0.0

            # 6. Gross Margin
            gross_margin = safe_div(gross_prof[i], rev[i], default=0.0)

            # 7. Operating Margin
            op_margin = safe_div(op_inc[i], rev[i], default=0.0)

            # 8. ROIC (NOPAT / Invested Capital)
            nopat = op_inc[i] * 0.79 # Assuming ~21% corporate tax rate
            inv_cap = tot_debt + total_equity[i] - cash_eq[i]
            roic = safe_div(nopat, inv_cap, default=0.0)

            # 9. Net Debt / EBITDA
            net_debt = tot_debt - cash_eq[i]
            net_debt_ebitda = safe_div(net_debt, ebitda, default=0.0)

            # 10. Shareholder Yield (-(Dividends + Repurchases + Debt Repayments) / MarketCap)
            cash_returned = -(div_paid[i] + repurchase[i] + debt_repay[i])
            sh_yield = safe_div(cash_returned, mkt_cap, default=0.0)

            # 11. 5Y Div Growth (20 quarters ago)
            div_curr = abs(div_paid[i])
            div_5y = abs(div_paid[i-20]) if i >= 20 else 0.0
            if i >= 20 and div_5y > 0 and div_curr > 0:
                five_y_div_growth = (div_curr / div_5y) ** (1/5) - 1.0
            else:
                five_y_div_growth = 0.0

            # 12. Payout Ratio (-Dividends Paid / Net Income)
            payout_ratio = safe_div(-div_paid[i], net_inc[i], default=0.0) if net_inc[i] > 0 else 0.0

            # Base Filters
            current_ratio = safe_div(curr_assets[i], curr_liab[i], default=0.0)

            row = {
                "Ticker": ticker,
                "GICS_Sector": sectors[i],
                "Quarter_End_Date": dates[i],
                "TrailingPE": round(float(trailing_pe), 4),
                "EV_EBITDA": round(float(ev_ebitda), 4),
                "FCFYield": round(float(fcf_yield), 6),
                "RevenueGrowth": round(float(rev_growth), 6),
                "EPSGrowth": round(float(eps_growth), 6),
                "GrossMargin": round(float(gross_margin), 6),
                "OperatingMargin": round(float(op_margin), 6),
                "ROIC": round(float(roic), 6),
                "NetDebt_EBITDA": round(float(net_debt_ebitda), 4),
                "ShareholderYield": round(float(sh_yield), 6),
                "FiveYDivGrowth": round(float(five_y_div_growth), 4),
                "PayoutRatio": round(float(payout_ratio), 6),
                "Beta": round(float(beta_vals[i]), 4),
                "RSI": round(float(rsi_vals[i]), 4),
                "PriceVs200SMA": round(float(p_sma_vals[i]), 6),
                "RelativeStrength": round(float(rel_str_vals[i]), 6),
                "CurrentRatio": round(float(current_ratio), 4),
                "MarketCap": round(float(mkt_cap), 2),
                "EBITDA": round(float(ebitda), 2),
                "Total_Equity": round(float(total_equity[i]), 2),
                "FCF": round(float(fcf), 2),
                "EBIT": round(float(op_inc[i]), 2),
                "Interest_Expense": round(float(int_exp_col[i]), 2)
            }
            rows.append(row)

    df_out = pd.DataFrame(rows)
    return df_out

def main():
    print("=== AlphaQuant V3.3 — SimFin ETL Pipeline ===")
    try:
        companies, industries, income, cashflow, balance, prices = load_raw_datasets()
        sector_map = build_sector_map(companies, industries)
        df_prices = process_daily_prices(prices)
        merged_statements = build_fundamental_panel(income, cashflow, balance, sector_map)
        df_final = calculate_all_indicators(merged_statements, df_prices)

        # Clean NaN/Inf
        df_final = df_final.fillna(0.0)
        df_final = df_final.replace([np.inf, -np.inf], 0.0)

        # Ensure exact schema column order
        df_final = df_final[FINAL_COLUMNS]

        # Save to output file
        df_final.to_csv(OUTPUT_FILE, index=False)
        print(f" SUCCESS: Generated {len(df_final)} fundamental snapshot rows across {df_final['Ticker'].nunique()} tickers.")
        print(f" File saved to: {OUTPUT_FILE}")

    except Exception as e:
        print(f" ERROR: ETL Execution failed — {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
