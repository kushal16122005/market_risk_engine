import numpy as np
import pandas as pd


def validate_weights(weights):
    weights_arr = np.asarray(weights, dtype=float)

    if np.isnan(weights_arr).any():
        raise ValueError("Portfolio weights contain missing values.")

    if not np.isclose(weights_arr.sum(), 1.0):
        raise ValueError(
            f"Portfolio weights must sum to 1. Current sum = {weights_arr.sum():.4f}"
        )

    return True


def align_weights(returns, weights):
    if isinstance(weights, dict):
        weights = pd.Series(weights)

    if isinstance(weights, pd.Series):
        missing = set(returns.columns) - set(weights.index)
        extra = set(weights.index) - set(returns.columns)

        if missing:
            raise ValueError(f"Missing weights for assets: {sorted(missing)}")
        if extra:
            raise ValueError(f"Weights given for unknown/unmatched assets: {sorted(extra)}")

        aligned = weights.reindex(returns.columns).astype(float)
        return aligned

    weights_arr = np.asarray(weights, dtype=float)

    if len(weights_arr) != len(returns.columns):
        raise ValueError(
            "Number of portfolio weights must match number of assets "
            f"({len(weights_arr)} weights vs {len(returns.columns)} assets)."
        )

    return pd.Series(weights_arr, index=returns.columns)


def calculate_portfolio_returns(returns, weights):
    aligned_weights = align_weights(returns, weights)
    validate_weights(aligned_weights.values)

    returns_ordered = returns[aligned_weights.index]
    portfolio_returns = returns_ordered @ aligned_weights.values

    portfolio_returns.name = "Portfolio Return"
    return portfolio_returns


def calculate_portfolio_pnl(portfolio_returns, portfolio_value):
    portfolio_pnl = portfolio_returns * portfolio_value
    portfolio_pnl.name = "Portfolio P&L"
    return portfolio_pnl