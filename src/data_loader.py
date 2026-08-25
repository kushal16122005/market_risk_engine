import yfinance as yf


def download_prices(tickers, start, end):

    data = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=False
    )

    prices = data["Adj Close"]

    return prices.dropna()