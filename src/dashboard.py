import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ---------------------------------------------------------------------------
# Individual panels - each is also usable standalone, not just inside the
# combined dashboard below.
# ---------------------------------------------------------------------------

def plot_portfolio_composition(weights, ax=None):
    """Pie chart of capital allocation across the portfolio's assets."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(weights.values, labels=weights.index, autopct="%1.1f%%",
           startangle=90, colors=plt.cm.tab10.colors[:len(weights)])
    ax.set_title("Portfolio Composition")
    return ax


def plot_var_es_comparison(model_comparison, ax=None):
    """
    Grouped bar chart of 95%/99% VaR & ES across all three models, from
    the `model_comparison` DataFrame already built in the main notebook.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 5))

    models = model_comparison["Model"]
    x = np.arange(len(models))
    width = 0.2
    metrics = ["95% VaR", "99% VaR", "95% ES", "99% ES"]

    for i, metric in enumerate(metrics):
        ax.bar(x + (i - 1.5) * width, model_comparison[metric], width, label=metric)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=10)
    ax.set_ylabel("₹")
    ax.set_title("VaR & ES Across Models")
    ax.legend(fontsize=8)
    return ax


def plot_rolling_var_trend(rolling_dict, confidence_level=0.99, ax=None):
    """
    Line chart of rolling VaR across all three models for one confidence
    level. `rolling_dict` = {"Historical Simulation": df, "Variance-Covariance": df, "Monte Carlo": df}
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 5))

    for label, df in rolling_dict.items():
        ax.plot(df.index, df["VaR"], label=label, linewidth=1)

    ax.set_title(f"Rolling {confidence_level:.0%} VaR — Model Comparison")
    ax.set_ylabel("₹")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    return ax


def plot_backtest_verdict(backtest_results, ax=None):
    """
    Compact visual verdict of backtesting results at 99% VaR: one row per
    model, colored by Basel traffic light zone, annotated with Kupiec
    pass/fail - the "did these models actually work" panel.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 3.5))

    zone_colors = {"Green": "#2ecc71", "Yellow": "#f1c40f", "Red": "#e74c3c", "N/A": "#bdc3c7"}

    rows_99 = [r for r in backtest_results if np.isclose(r["confidence_level"], 0.99)]

    for i, r in enumerate(rows_99):
        zone = r["traffic_light"]["zone"]
        kupiec = "FAIL" if r["kupiec_pof"]["reject_null"] else "Pass"
        color = zone_colors.get(zone, "#bdc3c7")

        ax.barh(i, 1, color=color, alpha=0.85)
        ax.text(0.02, i, f"{r['model']} — 99% VaR", va="center", fontweight="bold", fontsize=9)
        ax.text(0.98, i, f"{zone} zone | Kupiec: {kupiec}", va="center", ha="right", fontsize=9)

    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, len(rows_99) - 0.5)
    ax.set_title("Backtesting Verdict (99% VaR, Basel Traffic Light)")
    for spine in ax.spines.values():
        spine.set_visible(False)
    return ax


def plot_decomposition_summary(decomp, ax=None):
    """
    Grouped bar chart of each asset's % contribution to total VaR, one
    bar per model, built from decomp["variance_covariance"]/["historical"]/
    ["monte_carlo"]["Pct_of_VaR"] (as returned by calculate_risk_decomposition).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    contribution = pd.DataFrame({
        "Variance-Covariance": decomp["variance_covariance"]["Pct_of_VaR"],
        "Historical Simulation": decomp["historical"]["Pct_of_VaR"],
        "Monte Carlo": decomp["monte_carlo"]["Pct_of_VaR"],
    })

    assets = contribution.index
    models = contribution.columns
    x = np.arange(len(assets))
    width = 0.8 / len(models)

    for i, model in enumerate(models):
        ax.bar(x + i * width - 0.4 + width / 2, contribution[model], width, label=model)

    ax.set_xticks(x)
    ax.set_xticklabels(assets, rotation=25, ha="right")
    ax.axhline(0, color="grey", linewidth=0.6)
    ax.set_ylabel("% of Total VaR")
    ax.set_title("Risk Contribution by Asset")
    ax.legend(fontsize=8)
    return ax


def plot_stress_summary(shock_summary, var_estimates, ax=None):
    """
    Bar chart of stress scenario losses against existing VaR estimates -
    the "how bad is bad" panel.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 5))

    colors = ["tab:red" if p < 0 else "tab:green" for p in shock_summary["Stress P&L"]]
    ax.bar(shock_summary["Scenario"], shock_summary["Stress P&L"], color=colors, alpha=0.85)

    for label, var_value in var_estimates.items():
        ax.axhline(-var_value, linestyle="--", linewidth=1, label=f"-{label}")

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("₹ P&L")
    ax.set_title("Stress Scenarios vs. VaR")
    ax.legend(fontsize=7, loc="lower right")
    ax.tick_params(axis="x", rotation=20, labelsize=8)
    return ax


def render_executive_summary(tickers, weights, portfolio_value, model_comparison,
                              backtest_results, decomp, shock_summary):
    """
    Plain-text "report card" summarizing the whole project's findings in
    a few sentences - the kind of paragraph that would open a risk
    report, generated directly from the objects already computed in the
    notebook (no hardcoded numbers).
    """
    n_assets = len(tickers)

    hist_99 = model_comparison.loc[model_comparison["Model"] == "Historical Simulation", "99% VaR"].iloc[0]
    vc_99 = model_comparison.loc[model_comparison["Model"] == "Variance-Covariance", "99% VaR"].iloc[0]
    mc_99 = model_comparison.loc[model_comparison["Model"] == "Monte Carlo", "99% VaR"].iloc[0]

    failed_99 = [
        r["model"] for r in backtest_results
        if np.isclose(r["confidence_level"], 0.99) and r["kupiec_pof"]["reject_null"]
    ]
    passed_99 = [
        r["model"] for r in backtest_results
        if np.isclose(r["confidence_level"], 0.99) and not r["kupiec_pof"]["reject_null"]
    ]

    top_risk_contributor = decomp["historical"]["Pct_of_VaR"].idxmax()
    top_risk_pct = decomp["historical"]["Pct_of_VaR"].max()

    worst_scenario = shock_summary.loc[shock_summary["Stress P&L"].idxmin()]

    lines = [
        f"Portfolio: {n_assets} NSE-listed equities, ₹{portfolio_value:,.0f} notional, "
        f"{'equal-weighted' if weights.nunique() == 1 else 'weighted as specified'}.",
        "",
        f"99% 1-day VaR ranges from ₹{min(hist_99, vc_99, mc_99):,.0f} to ₹{max(hist_99, vc_99, mc_99):,.0f} "
        f"across the three models (Historical ₹{hist_99:,.0f}, Variance-Covariance ₹{vc_99:,.0f}, "
        f"Monte Carlo ₹{mc_99:,.0f}).",
        "",
        f"Backtesting at 99%: {', '.join(passed_99) if passed_99 else 'no models'} passed the Kupiec "
        f"coverage test; {', '.join(failed_99) if failed_99 else 'no models'} "
        f"{'failed it' if failed_99 else ''}.",
        "",
        f"{top_risk_contributor} is the largest single source of portfolio risk, contributing "
        f"{top_risk_pct:.1%} of total historical VaR.",
        "",
        f"Worst stress scenario tested: \"{worst_scenario['Scenario']}\", "
        f"resulting in a loss of ₹{abs(worst_scenario['Stress P&L']):,.0f}.",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Combined dashboard
# ---------------------------------------------------------------------------

def build_risk_dashboard(tickers, weights, portfolio_value, model_comparison,
                          rolling_historical_99, rolling_vc_99, rolling_mc_99,
                          backtest_results, decomp, shock_summary, var_estimates,
                          confidence_level=0.99):
    """
    Assembles every stage of the project (portfolio, VaR/ES, rolling risk,
    backtesting, decomposition, stress testing) into one figure.

    Every argument is an object already produced by earlier cells in
    main.ipynb - this function computes nothing new, it only visualizes
    what has already been built.
    """
    fig = plt.figure(figsize=(18, 20))
    gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.55, wspace=0.3)

    fig.suptitle("Market Risk Engine — Portfolio Risk Dashboard", fontsize=18, fontweight="bold", y=0.995)

    ax1 = fig.add_subplot(gs[0, 0])
    plot_portfolio_composition(weights, ax=ax1)

    ax2 = fig.add_subplot(gs[0, 1])
    plot_var_es_comparison(model_comparison, ax=ax2)

    ax3 = fig.add_subplot(gs[1, :])
    rolling_dict = {
        "Historical Simulation": rolling_historical_99,
        "Variance-Covariance": rolling_vc_99,
        "Monte Carlo": rolling_mc_99,
    }
    plot_rolling_var_trend(rolling_dict, confidence_level=confidence_level, ax=ax3)

    ax4 = fig.add_subplot(gs[2, 0])
    plot_backtest_verdict(backtest_results, ax=ax4)

    ax5 = fig.add_subplot(gs[2, 1])
    plot_decomposition_summary(decomp, ax=ax5)

    ax6 = fig.add_subplot(gs[3, :])
    plot_stress_summary(shock_summary, var_estimates, ax=ax6)

    summary_text = render_executive_summary(
        tickers, weights, portfolio_value, model_comparison,
        backtest_results, decomp, shock_summary
    )

    return fig, summary_text