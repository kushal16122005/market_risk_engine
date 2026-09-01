import numpy as np
import pandas as pd

from src.monte_carlo_var import simulate_portfolio_returns


def calculate_monte_carlo_es(
    returns: pd.DataFrame,
    weights: np.ndarray,
    portfolio_value: float,
    confidence_level: float = 0.95,
    n_simulations: int = 10000,
    random_seed: int = 42
) -> float:
    """
    Calculate Monte Carlo Expected Shortfall.

    Expected Shortfall is the average portfolio loss
    beyond the VaR threshold.

    Parameters
    ----------
    returns : pd.DataFrame
        Historical asset returns.

    weights : np.ndarray
        Portfolio weights.

    portfolio_value : float
        Total portfolio market value.

    confidence_level : float
        Confidence level for ES calculation.

    n_simulations : int
        Number of Monte Carlo scenarios.

    random_seed : int
        Random seed for reproducibility.

    Returns
    -------
    float
        Expected Shortfall as a positive monetary loss.
    """

    # Generate Monte Carlo portfolio returns
    simulated_portfolio_returns = simulate_portfolio_returns(
        returns=returns,
        weights=weights,
        n_simulations=n_simulations,
        random_seed=random_seed
    )

    # Convert simulated returns into P&L
    simulated_pnl = (
        simulated_portfolio_returns
        * portfolio_value
    )

    # Calculate VaR threshold
    var_threshold = np.quantile(
        simulated_pnl,
        1 - confidence_level
    )

    # Select losses beyond the VaR threshold
    tail_losses = simulated_pnl[
        simulated_pnl <= var_threshold
    ]

    # Calculate average tail loss
    expected_shortfall = -np.mean(tail_losses)

    return expected_shortfall