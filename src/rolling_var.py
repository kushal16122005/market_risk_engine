import numpy as np
import pandas as pd
from scipy.stats import norm

from src.portfolio import align_weights


def calculate_rolling_historical_var_es(portfolio_returns, portfolio_value, window=250, confidence_level=0.95):
    rolling_var = []
    rolling_es = []
    dates = []

    for i in range(window, len(portfolio_returns) + 1):
        window_returns = portfolio_returns.iloc[i - window:i]
        window_pnl = window_returns * portfolio_value

        threshold = np.quantile(window_pnl, 1 - confidence_level)
        var = -threshold

        tail_losses = window_pnl[window_pnl <= threshold]
        es = -tail_losses.mean()

        rolling_var.append(var)
        rolling_es.append(es)
        dates.append(window_returns.index[-1])

    return pd.DataFrame({"VaR": rolling_var, "ES": rolling_es}, index=dates)


def calculate_rolling_variance_covariance_var_es(returns, weights, portfolio_value, window=250, confidence_level=0.95):
    aligned_weights = align_weights(returns, weights)
    returns_ordered = returns[aligned_weights.index]
    w = aligned_weights.values

    z_score = norm.ppf(confidence_level)
    pdf_value = norm.pdf(z_score)

    rolling_var = []
    rolling_es = []
    dates = []

    for i in range(window, len(returns_ordered) + 1):
        window_returns = returns_ordered.iloc[i - window:i]

        mean_returns = window_returns.mean().values
        covariance_matrix = window_returns.cov().values

        portfolio_mean = float(np.dot(w, mean_returns))
        portfolio_variance = float(np.dot(w, np.dot(covariance_matrix, w)))
        portfolio_std = np.sqrt(portfolio_variance)

        var = (z_score * portfolio_std - portfolio_mean) * portfolio_value
        es = (
            portfolio_std * pdf_value / (1 - confidence_level) - portfolio_mean
        ) * portfolio_value

        rolling_var.append(var)
        rolling_es.append(es)
        dates.append(window_returns.index[-1])

    return pd.DataFrame({"VaR": rolling_var, "ES": rolling_es}, index=dates)


def calculate_rolling_monte_carlo_var_es(returns, weights, portfolio_value, window=250, confidence_level=0.95,
                                          n_simulations=10000, random_seed=42):
    aligned_weights = align_weights(returns, weights)
    returns_ordered = returns[aligned_weights.index]
    w = aligned_weights.values

    rolling_var = []
    rolling_es = []
    dates = []

    for i in range(window, len(returns_ordered) + 1):
        window_returns = returns_ordered.iloc[i - window:i]

        mean_returns = window_returns.mean().values
        covariance_matrix = window_returns.cov().values

        rng = np.random.default_rng(random_seed + i)
        simulated_asset_returns = rng.multivariate_normal(
            mean_returns, covariance_matrix, size=n_simulations
        )

        simulated_portfolio_returns = simulated_asset_returns @ w
        simulated_pnl = simulated_portfolio_returns * portfolio_value

        threshold = np.quantile(simulated_pnl, 1 - confidence_level)
        var = -threshold

        tail_losses = simulated_pnl[simulated_pnl <= threshold]
        es = -tail_losses.mean()

        rolling_var.append(var)
        rolling_es.append(es)
        dates.append(window_returns.index[-1])

    return pd.DataFrame({"VaR": rolling_var, "ES": rolling_es}, index=dates)