"""
Utility script: compress_prices_to_parquet.py
Run ONCE locally to convert the 415MB SimFin CSV into a compact Parquet file
that can be tracked in Git and deployed to cloud servers.
"""
import pandas as pd
import os

def compress_prices_to_parquet():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, '..', 'data', 'raw_simfin', 'us-shareprices-daily.csv')
    parquet_path = os.path.join(base_dir, '..', 'data', 'raw_simfin', 'us-shareprices-daily.parquet')

    if os.path.exists(csv_path):
        print(f"Reading {csv_path} ...")
        df = pd.read_csv(
            csv_path,
            sep=';',
            usecols=['Ticker', 'Date', 'Close', 'Adj. Close', 'Volume'],
            dtype={'Ticker': 'str', 'Close': 'float32', 'Adj. Close': 'float32', 'Volume': 'float32'},
            parse_dates=['Date']
        )
    elif os.path.exists(parquet_path):
        print(f"CSV not found; re-layouting existing {parquet_path} into Date-sorted 200k-row groups...")
        df = pd.read_parquet(parquet_path, engine='pyarrow')
        df['Ticker'] = df['Ticker'].astype(str)
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
        for col in ('Close', 'Adj. Close', 'Volume'):
            if col in df.columns:
                df[col] = df[col].astype('float32')
    else:
        raise FileNotFoundError(f"Neither {csv_path} nor {parquet_path} exists.")

    print(f"Sorting {len(df):,} rows chronologically by (Date, Ticker) for row-group predicate pushdown...")
    df.sort_values(['Date', 'Ticker'], inplace=True, kind='mergesort')
    df.reset_index(drop=True, inplace=True)

    tmp_path = parquet_path + '.tmp'
    print("Writing Parquet with Snappy compression and row_group_size=200,000...")
    df.to_parquet(
        tmp_path,
        index=False,
        compression='snappy',
        engine='pyarrow',
        row_group_size=200_000,
    )
    os.replace(tmp_path, parquet_path)

    size_mb = os.path.getsize(parquet_path) / (1024 * 1024)
    print(f"Done! {parquet_path} written ({size_mb:.1f} MB, 200k-row Date-sorted row groups).")

if __name__ == '__main__':
    compress_prices_to_parquet()
