import numpy as np
import pandas as pd
from scipy.stats import norm


def calculate_portfolio_mean_return(
    returns: pd.DataFrame,
    weights: np.ndarray
) -> float:
    """
    Calculate the mean daily return of the portfolio.
    """

    mean_returns = returns.mean()

    portfolio_mean = np.dot(
        weights,
        mean_returns
    )

    return portfolio_mean


def calculate_portfolio_volatility(
    returns: pd.DataFrame,
    weights: np.ndarray
) -> float:
    """
    Calculate portfolio daily volatility
    using the covariance matrix.
    """

    covariance_matrix = returns.cov()

    portfolio_variance = np.dot(
        weights,
        np.dot(covariance_matrix, weights)
    )

    portfolio_volatility = np.sqrt(
        portfolio_variance
    )

    return portfolio_volatility


def calculate_variance_covariance_var(
    returns: pd.DataFrame,
    weights: np.ndarray,
    portfolio_value: float,
    confidence_level: float = 0.95
) -> float:
    """
    Calculate Variance-Covariance Historical VaR
    assuming normally distributed portfolio returns.

    Returns VaR as a positive monetary loss.
    """

    portfolio_mean = calculate_portfolio_mean_return(
        returns,
        weights
    )

    portfolio_volatility = calculate_portfolio_volatility(
        returns,
        weights
    )

    z_score = norm.ppf(confidence_level)

    var = (
        z_score * portfolio_volatility
        - portfolio_mean
    ) * portfolio_value

    return var