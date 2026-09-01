import numpy as np
import pandas as pd

def simulate_portfolio_returns(
    returns: pd.DataFrame,
    weights: np.ndarray,
    n_simulations: int = 10000,
    random_seed: int = 42
) -> np.ndarray:
    """
    Simulate portfolio returns using a multivariate normal distribution.

    Parameters
    ----------
    returns : pd.DataFrame
        Historical asset returns.
    weights : np.ndarray
        Portfolio weights.
    n_simulations : int
        Number of Monte Carlo scenarios.
    random_seed : int
        Seed for reproducibility.

    Returns
    -------
    np.ndarray
        Simulated portfolio returns.
    """

    # Set random seed for reproducibility
    np.random.seed(random_seed)

    # Historical mean returns
    mean_returns = returns.mean().values

    # Historical covariance matrix
    covariance_matrix = returns.cov().values

    # Generate correlated asset returns
    simulated_returns = np.random.multivariate_normal(
        mean_returns,
        covariance_matrix,
        size=n_simulations
    )

    # Convert asset returns into portfolio returns
    simulated_portfolio_returns = simulated_returns @ weights

    return simulated_portfolio_returns


def calculate_monte_carlo_var(
    returns: pd.DataFrame,
    weights: np.ndarray,
    portfolio_value: float,
    confidence_level: float = 0.95,
    n_simulations: int = 10000,
    random_seed: int = 42
) -> float:
    """
    Calculate Monte Carlo VaR.

    Returns VaR as a positive monetary loss.
    """

    simulated_portfolio_returns = simulate_portfolio_returns(
        returns=returns,
        weights=weights,
        n_simulations=n_simulations,
        random_seed=random_seed
    )

    # Convert returns into P&L
    simulated_pnl = (
        simulated_portfolio_returns
        * portfolio_value
    )

    # VaR is the negative of the return quantile
    var = -np.quantile(
        simulated_pnl,
        1 - confidence_level
    )

    return var