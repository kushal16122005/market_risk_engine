import numpy as np


def calculate_portfolio_statistics(portfolio_returns):

    statistics = {
        "mean_return": portfolio_returns.mean(),
        "volatility": portfolio_returns.std(),
        "minimum_return": portfolio_returns.min(),
        "maximum_return": portfolio_returns.max(),
        "skewness": portfolio_returns.skew(),
        "kurtosis": portfolio_returns.kurtosis()
    }

    return statistics