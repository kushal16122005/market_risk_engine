import numpy as np
from scipy.stats import norm

from src.portfolio import align_weights


def calculate_portfolio_mean_return(returns, weights):
    aligned_weights = align_weights(returns, weights)
    returns_ordered = returns[aligned_weights.index]
    mean_returns = returns_ordered.mean()
    return float(np.dot(aligned_weights.values, mean_returns.values))


def calculate_portfolio_volatility(returns, weights):
    aligned_weights = align_weights(returns, weights)
    returns_ordered = returns[aligned_weights.index]
    covariance_matrix = returns_ordered.cov()

    portfolio_variance = np.dot(
        aligned_weights.values,
        np.dot(covariance_matrix.values, aligned_weights.values),
    )
    return float(np.sqrt(portfolio_variance))


def calculate_variance_covariance_var(returns, weights, portfolio_value, confidence_level=0.95):
    portfolio_mean = calculate_portfolio_mean_return(returns, weights)
    portfolio_volatility = calculate_portfolio_volatility(returns, weights)

    z_score = norm.ppf(confidence_level)
    var = (z_score * portfolio_volatility - portfolio_mean) * portfolio_value

    return var