import numpy as np
import pandas as pd
from scipy.stats import norm

from src.portfolio import align_weights


# ---------------------------------------------------------------------------
# 1. Variance-Covariance (parametric) decomposition
# ---------------------------------------------------------------------------
#
# Because portfolio volatility is homogeneous of degree 1 in the weights,
# Euler's theorem lets it be split exactly across assets:
#
#     vol_p   = w' (Sigma w) / vol_p            (sum of asset terms = vol_p)
#     VaR_p   = (z * vol_p - mean_p) * value
#     ES_p    = (vol_p * phi(z) / (1 - cl) - mean_p) * value
#
# Applying the same split to each asset's contribution to vol_p gives
# Component VaR / Component ES that sum EXACTLY to the portfolio totals.

def calculate_marginal_contributions(returns, weights):
    """
    Marginal contribution to portfolio volatility for each asset:
    MCTR_i = (Sigma w)_i / portfolio_volatility
    """
    aligned_weights = align_weights(returns, weights)
    returns_ordered = returns[aligned_weights.index]
    w = aligned_weights.values

    covariance_matrix = returns_ordered.cov().values
    sigma_w = covariance_matrix @ w
    portfolio_variance = float(w @ sigma_w)
    portfolio_volatility = np.sqrt(portfolio_variance)

    marginal_contributions = sigma_w / portfolio_volatility

    return pd.Series(marginal_contributions, index=aligned_weights.index, name="Marginal_Vol")


def calculate_variance_covariance_decomposition(returns, weights, portfolio_value, confidence_level=0.95):
    """
    Per-asset Component VaR and Component ES under the Variance-Covariance model.

    Returns a DataFrame indexed by asset with columns:
    Weight, Component_VaR, Pct_of_VaR, Component_ES, Pct_of_ES
    """
    aligned_weights = align_weights(returns, weights)
    returns_ordered = returns[aligned_weights.index]
    w = aligned_weights.values

    mean_returns = returns_ordered.mean().values
    marginal_vol = calculate_marginal_contributions(returns, weights).values

    component_vol = w * marginal_vol
    component_mean = w * mean_returns

    z_score = norm.ppf(confidence_level)
    pdf_value = norm.pdf(z_score)

    component_var = (z_score * component_vol - component_mean) * portfolio_value
    component_es = (
        component_vol * pdf_value / (1 - confidence_level) - component_mean
    ) * portfolio_value

    total_var = component_var.sum()
    total_es = component_es.sum()

    decomposition = pd.DataFrame({
        "Weight": aligned_weights.values,
        "Component_VaR": component_var,
        "Pct_of_VaR": component_var / total_var,
        "Component_ES": component_es,
        "Pct_of_ES": component_es / total_es,
    }, index=aligned_weights.index)

    return decomposition


# ---------------------------------------------------------------------------
# 2. Historical Simulation decomposition
# ---------------------------------------------------------------------------
#
# Component ES: average each asset's dollar P&L over exactly the same tail
# days used by historical_expected_shortfall.py. Since portfolio P&L on any
# day is the sum of asset P&Ls, this sums EXACTLY to the portfolio ES.
#
# Component VaR: VaR is a single quantile, not an average, so it is
# attributed using the single historical scenario at that quantile
# (nearest-rank convention). This will sum close to, but not always
# exactly equal to, the interpolated headline VaR from historical_var.py.

def calculate_historical_decomposition(returns, portfolio_returns, weights, portfolio_value, confidence_level=0.95):
    """
    Per-asset Component VaR and Component ES under Historical Simulation.
    """
    aligned_weights = align_weights(returns, weights)
    returns_ordered = returns[aligned_weights.index]
    w = aligned_weights.values

    asset_pnl = returns_ordered * w * portfolio_value  # dollar P&L per asset per day

    # --- Component ES: average over the tail beyond the VaR threshold ---
    threshold = np.percentile(portfolio_returns, (1 - confidence_level) * 100)
    tail_mask = portfolio_returns <= threshold

    component_es = -asset_pnl.loc[tail_mask].mean()
    total_es = component_es.sum()

    # --- Component VaR: single nearest-rank scenario at the VaR quantile ---
    sorted_returns = portfolio_returns.sort_values()
    rank = max(int(np.ceil((1 - confidence_level) * len(sorted_returns))), 1)
    var_scenario_date = sorted_returns.index[rank - 1]

    component_var = -asset_pnl.loc[var_scenario_date]
    total_var = component_var.sum()

    decomposition = pd.DataFrame({
        "Weight": aligned_weights.values,
        "Component_VaR": component_var.values,
        "Pct_of_VaR": component_var.values / total_var,
        "Component_ES": component_es.values,
        "Pct_of_ES": component_es.values / total_es,
    }, index=aligned_weights.index)

    return decomposition


# ---------------------------------------------------------------------------
# 3. Monte Carlo decomposition
# ---------------------------------------------------------------------------
#
# Same tail-average / nearest-rank logic as the historical model, but
# applied to simulated multivariate-normal asset returns instead of the
# realised return history. Uses the same random_seed as monte_carlo_var.py
# so the decomposition is reproducible alongside the headline MC VaR/ES.

def simulate_asset_returns(returns, weights, n_simulations=10000, random_seed=42):
    """
    Simulate correlated asset returns from a multivariate normal fitted on
    historical mean/covariance. Mirrors monte_carlo_var.simulate_portfolio_returns
    but keeps the per-asset simulated matrix needed for decomposition.
    """
    aligned_weights = align_weights(returns, weights)
    returns_ordered = returns[aligned_weights.index]

    mean_returns = returns_ordered.mean().values
    covariance_matrix = returns_ordered.cov().values

    rng = np.random.default_rng(random_seed)

    simulated_asset_returns = rng.multivariate_normal(
        mean_returns, covariance_matrix, size=n_simulations
    )

    return pd.DataFrame(simulated_asset_returns, columns=aligned_weights.index)


def calculate_monte_carlo_decomposition(returns, weights, portfolio_value, confidence_level=0.95,
                                         n_simulations=10000, random_seed=42):
    """
    Per-asset Component VaR and Component ES under Monte Carlo simulation.
    """
    aligned_weights = align_weights(returns, weights)
    w = aligned_weights.values

    simulated_asset_returns = simulate_asset_returns(
        returns, weights, n_simulations=n_simulations, random_seed=random_seed
    )

    simulated_asset_pnl = simulated_asset_returns * w * portfolio_value  # dollar P&L per asset per scenario
    simulated_portfolio_pnl = simulated_asset_pnl.sum(axis=1)

    # --- Component ES: average over the simulated tail ---
    threshold = np.quantile(simulated_portfolio_pnl, 1 - confidence_level)
    tail_mask = simulated_portfolio_pnl <= threshold

    component_es = -simulated_asset_pnl.loc[tail_mask].mean()
    total_es = component_es.sum()

    # --- Component VaR: single nearest-rank scenario at the VaR quantile ---
    sorted_pnl = simulated_portfolio_pnl.sort_values()
    rank = max(int(np.ceil((1 - confidence_level) * len(sorted_pnl))), 1)
    var_scenario_idx = sorted_pnl.index[rank - 1]

    component_var = -simulated_asset_pnl.loc[var_scenario_idx]
    total_var = component_var.sum()

    decomposition = pd.DataFrame({
        "Weight": aligned_weights.values,
        "Component_VaR": component_var.values,
        "Pct_of_VaR": component_var.values / total_var,
        "Component_ES": component_es.values,
        "Pct_of_ES": component_es.values / total_es,
    }, index=aligned_weights.index)

    return decomposition


# ---------------------------------------------------------------------------
# 4. Combined orchestrator
# ---------------------------------------------------------------------------

def calculate_risk_decomposition(returns, portfolio_returns, weights, portfolio_value, confidence_level=0.95,
                                  n_simulations=10000, random_seed=42):
    """
    Run all three decompositions and combine into one per-asset comparison table.

    Returns
    -------
    dict with keys "variance_covariance", "historical", "monte_carlo", each a
    DataFrame (see individual functions above), plus "summary": a DataFrame
    comparing Component_VaR and Component_ES side by side across models,
    with a totals row for reconciliation against the headline VaR/ES figures.
    """
    vc_decomp = calculate_variance_covariance_decomposition(
        returns, weights, portfolio_value, confidence_level
    )
    hist_decomp = calculate_historical_decomposition(
        returns, portfolio_returns, weights, portfolio_value, confidence_level
    )
    mc_decomp = calculate_monte_carlo_decomposition(
        returns, weights, portfolio_value, confidence_level, n_simulations, random_seed
    )

    summary = pd.DataFrame({
        "VC_Component_VaR": vc_decomp["Component_VaR"],
        "Hist_Component_VaR": hist_decomp["Component_VaR"],
        "MC_Component_VaR": mc_decomp["Component_VaR"],
        "VC_Component_ES": vc_decomp["Component_ES"],
        "Hist_Component_ES": hist_decomp["Component_ES"],
        "MC_Component_ES": mc_decomp["Component_ES"],
    })
    summary.loc["Total"] = summary.sum()

    return {
        "variance_covariance": vc_decomp,
        "historical": hist_decomp,
        "monte_carlo": mc_decomp,
        "summary": summary,
    }