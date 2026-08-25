import numpy as np


def historical_var(
    portfolio_returns,
    portfolio_value,
    confidence_level=0.95
):
    """
    Calculate Historical Simulation VaR.
    """

    percentile = (1 - confidence_level) * 100

    return_threshold = np.percentile(
        portfolio_returns,
        percentile
    )

    var = -return_threshold * portfolio_value

    return var