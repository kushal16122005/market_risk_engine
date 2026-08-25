import numpy as np

def historical_expected_shortfall(
    portfolio_returns,
    portfolio_value,
    confidence_level=0.95
):
    """
    Calculate Historical Expected Shortfall.
    """

    percentile = (1 - confidence_level) * 100

    var_threshold = np.percentile(
        portfolio_returns,
        percentile
    )

    tail_returns = portfolio_returns[
        portfolio_returns <= var_threshold
    ]

    expected_shortfall = (
        -tail_returns.mean() * portfolio_value
    )

    return expected_shortfall