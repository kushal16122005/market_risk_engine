import numpy as np
import pandas as pd
from scipy.stats import norm


def calculate_variance_covariance_es(
    returns: pd.DataFrame,
    weights: np.ndarray,
    portfolio_value: float,
    confidence_level: float = 0.95
) -> float:
    """
    Calculate Variance-Covariance Expected Shortfall (ES)
    assuming normally distributed portfolio returns.

    Returns ES as a positive monetary loss.
    """

    # Mean return of each asset
    mean_returns = returns.mean()

    # Portfolio mean return
    portfolio_mean = np.dot(
        weights,
        mean_returns
    )

    # Covariance matrix
    covariance_matrix = returns.cov()

    # Portfolio variance
    portfolio_variance = np.dot(
        weights,
        np.dot(covariance_matrix, weights)
    )

    # Portfolio volatility
    portfolio_volatility = np.sqrt(
        portfolio_variance
    )

    # Standard normal PDF and quantile
    z_score = norm.ppf(confidence_level)
    pdf_value = norm.pdf(z_score)

    # Expected Shortfall
    es = (
        portfolio_volatility
        * pdf_value
        / (1 - confidence_level)
        - portfolio_mean
    ) * portfolio_value

    return es