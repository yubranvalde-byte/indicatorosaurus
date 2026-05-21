import requests
import pandas as pd
import io

headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

def test_sp500():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    response = requests.get(url, headers=headers)
    tables = pd.read_html(io.StringIO(response.text))
    print(f"S&P 500 Tables found: {len(tables)}")
    print("S&P 500 Table 0 columns:", tables[0].columns.tolist())
    print("S&P 500 Table 0 sample tickers:", tables[0].iloc[:, 0].head().tolist())

def test_nasdaq100():
    url = "https://en.wikipedia.org/wiki/Nasdaq-100"
    response = requests.get(url, headers=headers)
    tables = pd.read_html(io.StringIO(response.text))
    print(f"Nasdaq 100 Tables found: {len(tables)}")
    for i, t in enumerate(tables):
        cols = [str(c).lower() for c in t.columns.tolist()]
        if any('ticker' in c or 'symbol' in c for c in cols):
            print(f"Nasdaq 100 Table {i} columns:", t.columns.tolist())
            print(f"Nasdaq 100 Table {i} sample tickers:", t.iloc[:, 0].head().tolist())

if __name__ == "__main__":
    test_sp500()
    test_nasdaq100()
