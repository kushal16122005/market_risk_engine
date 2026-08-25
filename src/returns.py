import numpy as np


def calculate_simple_returns(prices):
    returns = prices.pct_change()
    return returns.dropna()


def calculate_log_returns(prices):
    log_returns = np.log(prices / prices.shift(1))
    return log_returns.dropna()


def validate_returns(returns):

    validation = {
        "rows": len(returns),
        "columns": len(returns.columns),
        "missing_values": returns.isna().sum().sum(),
        "infinite_values": np.isinf(returns).sum().sum()
    }

    return validation