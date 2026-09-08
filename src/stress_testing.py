import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.portfolio import align_weights


# ---------------------------------------------------------------------------
# 1. Historical scenario stress testing (replay worst historical moves)
# ---------------------------------------------------------------------------

def historical_scenario_pnl(portfolio_returns, portfolio_value, windows=(1, 5, 10, 20)):
    """
    For each window length (in trading days), finds the worst historical
    CUMULATIVE portfolio return of that length actually observed in the
    sample, and applies it to today's portfolio value - i.e. "if the
    worst N-day move we've already lived through happened again starting
    today, what would we lose."
    """
    rows = []
    for window in windows:
        cumulative_returns = (1 + portfolio_returns).rolling(window).apply(np.prod, raw=True) - 1
        cumulative_returns = cumulative_returns.dropna()

        end_date = cumulative_returns.idxmin()
        worst_return = cumulative_returns.min()

        end_pos = portfolio_returns.index.get_loc(end_date)
        start_pos = max(end_pos - window + 1, 0)
        start_date = portfolio_returns.index[start_pos]

        rows.append({
            "Window (days)": window,
            "Start Date": start_date,
            "End Date": end_date,
            "Cumulative Return": worst_return,
            "Stress P&L": worst_return * portfolio_value,
        })

    return pd.DataFrame(rows)


def replay_specific_period(portfolio_returns, portfolio_value, start_date, end_date, label=None):
    """
    Replays an actual, named historical period (e.g. a specific crash
    window you want to call out explicitly) against today's portfolio value.
    """
    window_returns = portfolio_returns.loc[start_date:end_date]
    if window_returns.empty:
        raise ValueError(f"No data found between {start_date} and {end_date}.")

    cumulative_return = (1 + window_returns).prod() - 1

    return {
        "label": label or f"{start_date} to {end_date}",
        "start_date": window_returns.index[0],
        "end_date": window_returns.index[-1],
        "n_days": len(window_returns),
        "cumulative_return": cumulative_return,
        "stress_pnl": cumulative_return * portfolio_value,
    }


# ---------------------------------------------------------------------------
# 2. Hypothetical shock scenarios (direct, user-defined)
# ---------------------------------------------------------------------------

def apply_shock_scenario(weights, shocks, portfolio_value, scenario_name="Scenario"):
    """
    Applies a hypothetical shock (a return, e.g. -0.10 for -10%) to one or
    more assets and computes the resulting portfolio P&L.

    `shocks` is a dict/Series of {ticker: shock_return}. Any ticker in
    `weights` NOT present in `shocks` is treated as unshocked (0% move) -
    this lets you define single-asset or partial/sector scenarios without
    specifying every ticker every time.
    """
    if isinstance(shocks, dict):
        shocks = pd.Series(shocks)

    aligned_shocks = shocks.reindex(weights.index).fillna(0.0)

    contribution = weights * aligned_shocks
    portfolio_shock_return = contribution.sum()

    breakdown = pd.DataFrame({
        "Weight": weights,
        "Shock": aligned_shocks,
        "Contribution to Return": contribution,
        "P&L Contribution": contribution * portfolio_value,
    })

    return {
        "scenario_name": scenario_name,
        "portfolio_shock_return": portfolio_shock_return,
        "stress_pnl": portfolio_shock_return * portfolio_value,
        "breakdown": breakdown,
    }


def run_shock_scenarios(weights, scenarios, portfolio_value):
    """
    scenarios: dict of {scenario_name: {ticker: shock_return, ...}}
    Runs apply_shock_scenario for each and returns a summary table plus
    the full per-scenario detail (breakdown, etc.) keyed by scenario name.
    """
    rows = []
    details = {}
    for name, shocks in scenarios.items():
        result = apply_shock_scenario(weights, shocks, portfolio_value, scenario_name=name)
        rows.append({
            "Scenario": name,
            "Portfolio Return": result["portfolio_shock_return"],
            "Stress P&L": result["stress_pnl"],
        })
        details[name] = result

    return pd.DataFrame(rows), details


# ---------------------------------------------------------------------------
# 3. Correlated shock propagation (single-factor / beta-based)
# ---------------------------------------------------------------------------

def propagate_correlated_shock(returns, weights, driver_ticker, driver_shock, portfolio_value,
                                scenario_name=None):
    """
    Shocks ONE asset directly (the "driver") and propagates an IMPLIED
    shock to every other asset using its historical beta to the driver:

        beta_i = Cov(r_i, r_driver) / Var(r_driver)
        implied_shock_i = beta_i * driver_shock

    This approximates how a shock to one name ripples through a
    correlated portfolio, using the same historical covariance structure
    the Variance-Covariance VaR model already relies on - more realistic
    than assuming every other asset stays flat while one crashes.
    """
    aligned_weights = align_weights(returns, weights)
    returns_ordered = returns[aligned_weights.index]

    if driver_ticker not in returns_ordered.columns:
        raise ValueError(f"'{driver_ticker}' not found in portfolio assets.")

    driver_returns = returns_ordered[driver_ticker]
    driver_variance = driver_returns.var()

    betas = returns_ordered.apply(lambda col: col.cov(driver_returns) / driver_variance)
    implied_shocks = betas * driver_shock
    implied_shocks[driver_ticker] = driver_shock  # driver gets exactly the specified shock

    result = apply_shock_scenario(
        aligned_weights, implied_shocks, portfolio_value,
        scenario_name=scenario_name or f"{driver_ticker} {driver_shock:+.0%} (correlated propagation)",
    )
    result["betas"] = betas
    return result


# ---------------------------------------------------------------------------
# 4. Compare stress losses against existing VaR estimates
# ---------------------------------------------------------------------------

def compare_to_var(stress_pnl, var_estimates):
    """
    var_estimates: dict of {label: var_value}, e.g.
        {"Historical 95%": 167647, "Historical 99%": 269178, ...}

    Expresses a stress scenario's loss as a multiple of each VaR estimate
    ("how many VaRs deep" the scenario is), and flags whether the
    scenario loss exceeds that VaR outright.
    """
    stress_loss = abs(min(stress_pnl, 0))

    rows = []
    for label, var_value in var_estimates.items():
        rows.append({
            "VaR Estimate": label,
            "VaR Value": var_value,
            "Multiple of VaR": stress_loss / var_value if var_value else np.nan,
            "Exceeds VaR": stress_loss > var_value,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 5. Visualization
# ---------------------------------------------------------------------------

def plot_stress_vs_var(scenario_summary, var_estimates):
    """
    Bar chart of stress scenario P&L with horizontal reference lines for
    each VaR estimate - the standard "how bad is bad" stress-test chart.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ["tab:red" if p < 0 else "tab:green" for p in scenario_summary["Stress P&L"]]
    ax.bar(scenario_summary["Scenario"], scenario_summary["Stress P&L"], color=colors, alpha=0.85)

    for label, var_value in var_estimates.items():
        ax.axhline(-var_value, linestyle="--", linewidth=1, label=f"-{label}")

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("₹ P&L")
    ax.set_title("Stress Scenario Losses vs. VaR Estimates")
    ax.legend(loc="lower right", fontsize=8)
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()

    return fig