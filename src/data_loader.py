import os

import pandas as pd
import yfinance as yf


def download_prices(tickers, start, end, cache_path="data/raw/prices.csv", use_cache=True):
    if use_cache and os.path.exists(cache_path):
        prices = pd.read_csv(cache_path, index_col="Date", parse_dates=True)
        return prices

    try:
        data = yf.download(
            tickers,
            start=start,
            end=end,
            auto_adjust=False,
            progress=False,
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to download price data from yfinance: {exc}") from exc

    if data.empty:
        raise ValueError(
            "yfinance returned no data - check the tickers, date range, "
            "or your network access."
        )

    prices = data["Adj Close"].dropna()

    missing_tickers = set(tickers) - set(prices.columns)
    if missing_tickers:
        raise ValueError(f"No price data returned for: {sorted(missing_tickers)}")

    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    prices.to_csv(cache_path)

    return prices