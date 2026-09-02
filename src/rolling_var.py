import numpy as np
import pandas as pd

# Historical Simulation rolling VaR & ES

def calculate_rolling_historical_var_es(
    portfolio_returns: pd.Series,
    portfolio_value: float,
    window: int = 250,
    confidence_level: float = 0.95
) -> pd.DataFrame:
    """
    Calculate rolling Historical Simulation VaR and Expected Shortfall.

    Parameters
    ----------
    portfolio_returns : pd.Series
        Historical portfolio returns.

    portfolio_value : float
        Current portfolio value.

    window : int
        Number of observations used in each rolling window.

    confidence_level : float
        Confidence level for VaR and ES.

    Returns
    -------
    pd.DataFrame
        DataFrame containing rolling VaR and ES.
    """

    rolling_var = []
    rolling_es = []
    dates = []

    for i in range(window, len(portfolio_returns) + 1):

        # Select rolling window
        window_returns = portfolio_returns.iloc[
            i - window:i
        ]

        # Convert returns to P&L
        window_pnl = (
            window_returns * portfolio_value
        )

        # Calculate VaR threshold
        var_threshold = np.quantile(
            window_pnl,
            1 - confidence_level
        )

        # VaR as positive loss
        var = -var_threshold

        # Tail losses beyond VaR
        tail_losses = window_pnl[
            window_pnl <= var_threshold
        ]

        # Expected Shortfall
        es = -tail_losses.mean()

        rolling_var.append(var)
        rolling_es.append(es)

        dates.append(window_returns.index[-1])

    result = pd.DataFrame(
        {
            "VaR": rolling_var,
            "ES": rolling_es
        },
        index=dates
    )

    return result

# Variance-Covariance rolling VaR & ES

def calculate_rolling_variance_covariance_var_es(
    returns: pd.DataFrame,
    weights: np.ndarray,
    portfolio_value: float,
    window: int = 250,
    confidence_level: float = 0.95
) -> pd.DataFrame:
    """
    Calculate rolling Variance-Covariance VaR and Expected Shortfall.

    Parameters
    ----------
    returns : pd.DataFrame
        Historical asset returns.

    weights : np.ndarray
        Portfolio weights.

    portfolio_value : float
        Current portfolio value.

    window : int
        Number of observations used in each rolling window.

    confidence_level : float
        Confidence level for VaR and ES.

    Returns
    -------
    pd.DataFrame
        DataFrame containing rolling VaR and ES.
    """

    rolling_var = []
    rolling_es = []
    dates = []

    for i in range(window, len(returns) + 1):

        window_returns = returns.iloc[
            i - window:i
        ]

        mean_returns = window_returns.mean().values
        covariance_matrix = window_returns.cov().values

        portfolio_mean = (
            weights @ mean_returns
        )

        portfolio_variance = (
            weights
            @ covariance_matrix
            @ weights
        )

        portfolio_std = np.sqrt(
            portfolio_variance
        )

        z_score = {
            0.95: 1.645,
            0.99: 2.326
        }[confidence_level]

        var = (
            -(
                portfolio_mean
                - z_score * portfolio_std
            )
            * portfolio_value
        )

        phi_z = (
            np.exp(-0.5 * z_score ** 2)
            / np.sqrt(2 * np.pi)
        )

        es_return = (
            -portfolio_mean
            + portfolio_std
            * phi_z
            / (1 - confidence_level)
        )

        es = (
            es_return
            * portfolio_value
        )

        rolling_var.append(var)
        rolling_es.append(es)

        dates.append(
            window_returns.index[-1]
        )

    result = pd.DataFrame(
        {
            "VaR": rolling_var,
            "ES": rolling_es
        },
        index=dates
    )

    return result

# Monte Carlo rolling VaR & ES

def calculate_rolling_monte_carlo_var_es(
    returns: pd.DataFrame,
    weights: np.ndarray,
    portfolio_value: float,
    window: int = 250,
    confidence_level: float = 0.95,
    n_simulations: int = 10000,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Calculate rolling Monte Carlo VaR and Expected Shortfall.

    Each rolling window estimates the mean vector and covariance
    matrix from historical returns, then simulates portfolio
    returns using a multivariate normal distribution.

    Parameters
    ----------
    returns : pd.DataFrame
        Historical asset returns.

    weights : np.ndarray
        Portfolio weights.

    portfolio_value : float
        Current portfolio value.

    window : int
        Number of observations used in each rolling window.

    confidence_level : float
        Confidence level for VaR and ES.

    n_simulations : int
        Number of Monte Carlo scenarios per rolling window.

    random_seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame containing rolling VaR and ES.
    """

    rolling_var = []
    rolling_es = []
    dates = []

    for i in range(window, len(returns) + 1):

        window_returns = returns.iloc[
            i - window:i
        ]

        mean_returns = window_returns.mean().values

        covariance_matrix = window_returns.cov().values

        simulated_returns = np.random.default_rng(
            random_seed + i
        ).multivariate_normal(
            mean_returns,
            covariance_matrix,
            size=n_simulations
        )

        simulated_portfolio_returns = (
            simulated_returns @ weights
        )

        simulated_pnl = (
            simulated_portfolio_returns
            * portfolio_value
        )

        var_threshold = np.quantile(
            simulated_pnl,
            1 - confidence_level
        )

        var = -var_threshold

        tail_losses = simulated_pnl[
            simulated_pnl <= var_threshold
        ]

        es = -tail_losses.mean()

        rolling_var.append(var)
        rolling_es.append(es)

        dates.append(
            window_returns.index[-1]
        )

    result = pd.DataFrame(
        {
            "VaR": rolling_var,
            "ES": rolling_es
        },
        index=dates
    )

    return result