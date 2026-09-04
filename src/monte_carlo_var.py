import numpy as np

from src.portfolio import align_weights


def simulate_portfolio_returns(returns, weights, n_simulations=10000, random_seed=42):
    aligned_weights = align_weights(returns, weights)
    returns_ordered = returns[aligned_weights.index]
    w = aligned_weights.values

    mean_returns = returns_ordered.mean().values
    covariance_matrix = returns_ordered.cov().values

    rng = np.random.default_rng(random_seed)

    simulated_asset_returns = rng.multivariate_normal(
        mean_returns, covariance_matrix, size=n_simulations
    )

    simulated_portfolio_returns = simulated_asset_returns @ w
    return simulated_portfolio_returns


def calculate_monte_carlo_var(returns, weights, portfolio_value, confidence_level=0.95,
                               n_simulations=10000, random_seed=42):
    simulated_portfolio_returns = simulate_portfolio_returns(
        returns=returns,
        weights=weights,
        n_simulations=n_simulations,
        random_seed=random_seed,
    )

    simulated_pnl = simulated_portfolio_returns * portfolio_value
    var = -np.quantile(simulated_pnl, 1 - confidence_level)

    return var