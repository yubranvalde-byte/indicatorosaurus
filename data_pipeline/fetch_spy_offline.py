import yfinance as yf
import pandas as pd
import os

def fetch_spy_offline():
    """
    Utility script to download the SPY benchmark history once and save it
    to the offline data lake.
    """
    output_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'spy_historical.csv')
    print(f"Downloading SPY history to {output_path}...")
    
    # Download SPY data from 2012 to today to match the longest possible window
    spy_data = yf.download('SPY', start='2012-01-01', progress=True)
    
    if spy_data.empty:
        print("Failed to download SPY data.")
        return

    # Handle MultiIndex and SingleIndex returned by yfinance versions
    if isinstance(spy_data.columns, pd.MultiIndex):
        lvl0 = spy_data.columns.get_level_values(0).unique().tolist()
        col_name = 'Close' if 'Close' in lvl0 else ('Adj Close' if 'Adj Close' in lvl0 else lvl0[0])
        spy_close = spy_data[col_name]
        if isinstance(spy_close, pd.DataFrame):
            spy_close = spy_close.iloc[:, 0]
    else:
        col_name = 'Close' if 'Close' in spy_data.columns else ('Adj Close' if 'Adj Close' in spy_data.columns else spy_data.columns[0])
        spy_close = spy_data[col_name]

    # Remove timezone information to match the offline fundamentals format
    spy_close.index = spy_close.index.tz_localize(None)

    # Convert to DataFrame with 'Date' and 'SPY_Adj_Close'
    spy_df = spy_close.reset_index()
    spy_df.columns = ['Date', 'SPY_Adj_Close']

    spy_df.to_csv(output_path, index=False)
    print(f"Successfully saved {len(spy_df)} trading days of SPY data to {output_path}.")

if __name__ == "__main__":
    fetch_spy_offline()
