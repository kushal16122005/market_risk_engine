import numpy as np
import pandas as pd


def validate_weights(weights):

    weights = np.asarray(weights, dtype=float)

    if np.isnan(weights).any():
        raise ValueError("Portfolio weights contain missing values.")

    if not np.isclose(weights.sum(), 1.0):
        raise ValueError(
            f"Portfolio weights must sum to 1. Current sum = {weights.sum():.4f}"
        )

    return True


def calculate_portfolio_returns(returns, weights):
    
    validate_weights(weights)

    if len(returns.columns) != len(weights):
        raise ValueError(
            "Number of portfolio weights must match number of assets."
        )

    portfolio_returns = returns @ np.asarray(weights)

    portfolio_returns.name = "Portfolio Return"

    return portfolio_returns


def calculate_portfolio_pnl(
    portfolio_returns,
    portfolio_value
):
    portfolio_pnl = portfolio_returns * portfolio_value

    portfolio_pnl.name = "Portfolio P&L"

    return portfolio_pnl