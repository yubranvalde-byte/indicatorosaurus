"""
Utility script: compress_prices_to_parquet.py
Run ONCE locally to convert the 415MB SimFin CSV into a compact Parquet file
that can be tracked in Git and deployed to cloud servers.
"""
import pandas as pd
import os

def compress_prices_to_parquet():
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw_simfin', 'us-shareprices-daily.csv')
    parquet_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw_simfin', 'us-shareprices-daily.parquet')

    print(f"Reading {csv_path} ...")
    df = pd.read_csv(
        csv_path,
        sep=';',
        usecols=['Ticker', 'Date', 'Close', 'Adj. Close', 'Volume'],
        dtype={'Ticker': 'category', 'Close': 'float32', 'Adj. Close': 'float32', 'Volume': 'float32'},
        parse_dates=['Date']
    )

    print(f"Loaded {len(df):,} rows. Writing Parquet with Snappy compression...")
    df.to_parquet(parquet_path, index=False, compression='snappy', engine='pyarrow')

    size_mb = os.path.getsize(parquet_path) / (1024 * 1024)
    original_mb = os.path.getsize(csv_path) / (1024 * 1024)
    print(f"Done! {original_mb:.1f}MB CSV -> {size_mb:.1f}MB Parquet ({100*(1-size_mb/original_mb):.0f}% smaller)")

if __name__ == '__main__':
    compress_prices_to_parquet()
