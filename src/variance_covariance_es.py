import numpy as np
from scipy.stats import norm

from src.portfolio import align_weights


def calculate_variance_covariance_es(returns, weights, portfolio_value, confidence_level=0.95):
    aligned_weights = align_weights(returns, weights)
    returns_ordered = returns[aligned_weights.index]
    w = aligned_weights.values

    mean_returns = returns_ordered.mean()
    portfolio_mean = float(np.dot(w, mean_returns.values))

    covariance_matrix = returns_ordered.cov()
    portfolio_variance = float(np.dot(w, np.dot(covariance_matrix.values, w)))
    portfolio_volatility = np.sqrt(portfolio_variance)

    z_score = norm.ppf(confidence_level)
    pdf_value = norm.pdf(z_score)

    es = (
        portfolio_volatility * pdf_value / (1 - confidence_level) - portfolio_mean
    ) * portfolio_value

    return es