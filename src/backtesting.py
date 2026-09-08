import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2


# ---------------------------------------------------------------------------
# 1. Align each rolling VaR/ES forecast to the P&L realized on the NEXT day
# ---------------------------------------------------------------------------

def align_forecast_to_realized(rolling_risk_df, portfolio_pnl):
    """
    rolling_var.py labels each VaR/ES estimate with the LAST date inside
    its 250-day estimation window - i.e. "as of the close of this date,
    using the trailing 250 days ending here, the VaR is X". That estimate
    is meant to be used as the risk limit for the NEXT trading session,
    not tested against its own day's P&L (that day's return was part of
    the data used to build the estimate, so comparing it to itself would
    be in-sample / look-ahead biased).

    This function shifts every forecast forward by one trading day and
    pairs it with the P&L that was actually realized on that next day,
    producing a genuine out-of-sample backtest. The final window's
    forecast is dropped since no next-day P&L exists yet to test it against.
    """
    full_dates = portfolio_pnl.index

    forecast_dates = []
    realized_pnl = []
    var_values = []
    es_values = []

    for asof_date, row in rolling_risk_df.iterrows():
        pos = full_dates.get_loc(asof_date)
        next_pos = pos + 1
        if next_pos >= len(full_dates):
            continue  # last window has no next-day P&L yet - drop it

        target_date = full_dates[next_pos]

        forecast_dates.append(target_date)
        realized_pnl.append(portfolio_pnl.loc[target_date])
        var_values.append(row["VaR"])
        es_values.append(row["ES"])

    aligned = pd.DataFrame(
        {"VaR": var_values, "ES": es_values, "realized_pnl": realized_pnl},
        index=pd.Index(forecast_dates, name="Date"),
    )
    return aligned


# ---------------------------------------------------------------------------
# 2. Exceptions
# ---------------------------------------------------------------------------

def identify_exceptions(aligned_df):
    """
    An exception (VaR breach) occurs when the realized loss is worse than
    the VaR forecast, i.e. realized_pnl <= -VaR.
    """
    return aligned_df["realized_pnl"] <= -aligned_df["VaR"]


def calculate_exception_statistics(aligned_df, confidence_level):
    exceptions = identify_exceptions(aligned_df)

    n_obs = len(aligned_df)
    n_exceptions = int(exceptions.sum())
    expected_rate = 1 - confidence_level

    return {
        "n_obs": n_obs,
        "n_exceptions": n_exceptions,
        "exception_rate": n_exceptions / n_obs if n_obs > 0 else np.nan,
        "expected_rate": expected_rate,
        "expected_exceptions": expected_rate * n_obs,
    }


# ---------------------------------------------------------------------------
# 3. Kupiec Proportion-of-Failures (POF) test
# ---------------------------------------------------------------------------

def kupiec_pof_test(n_obs, n_exceptions, confidence_level, significance_level=0.05):
    """
    Kupiec's (1995) unconditional coverage test.

    H0: the true exception probability equals p = 1 - confidence_level
    (i.e. the model is correctly calibrated on average).

        LR_pof = -2 * ln[ (1-p)^(n-x) p^x / (1-x/n)^(n-x) (x/n)^x ]

    which is asymptotically chi-square distributed with 1 degree of freedom
    under H0.
    """
    p = 1 - confidence_level
    n, x = n_obs, n_exceptions

    if n == 0:
        return {
            "LR_stat": np.nan, "p_value": np.nan, "critical_value": np.nan,
            "reject_null": None, "conclusion": "No observations to test.",
        }

    pi_hat = x / n

    log_L_null = (n - x) * np.log(1 - p) + (x * np.log(p) if x > 0 else 0)

    if pi_hat in (0, 1):
        log_L_alt = 0.0  # perfectly matches its own observed rate -> LR contribution is 0
    else:
        log_L_alt = (n - x) * np.log(1 - pi_hat) + x * np.log(pi_hat)

    LR_stat = max(-2 * (log_L_null - log_L_alt), 0.0)  # guard tiny float noise
    critical_value = chi2.ppf(1 - significance_level, df=1)
    p_value = 1 - chi2.cdf(LR_stat, df=1)
    reject_null = LR_stat > critical_value

    conclusion = (
        f"Reject H0 at {significance_level:.0%} significance - exception rate "
        f"({pi_hat:.2%}) is statistically different from the expected {p:.2%}."
        if reject_null else
        f"Fail to reject H0 at {significance_level:.0%} significance - exception "
        f"rate ({pi_hat:.2%}) is statistically consistent with the expected {p:.2%}."
    )

    return {
        "LR_stat": LR_stat, "p_value": p_value, "critical_value": critical_value,
        "reject_null": reject_null, "conclusion": conclusion,
    }


# ---------------------------------------------------------------------------
# 4. Christoffersen independence test + combined conditional coverage test
# ---------------------------------------------------------------------------

def christoffersen_independence_test(aligned_df, significance_level=0.05):
    """
    Kupiec's test only checks the NUMBER of exceptions - a model can have
    exactly the right count but have them all clustered in one crisis week
    (which is far more dangerous than the same count spread evenly) and
    still pass Kupiec. This test checks whether exceptions are independent
    over time using a first-order Markov chain of the exception indicator.
    """
    exceptions = identify_exceptions(aligned_df).astype(int).values

    n00 = n01 = n10 = n11 = 0
    for t in range(1, len(exceptions)):
        prev, curr = exceptions[t - 1], exceptions[t]
        if prev == 0 and curr == 0:
            n00 += 1
        elif prev == 0 and curr == 1:
            n01 += 1
        elif prev == 1 and curr == 0:
            n10 += 1
        else:
            n11 += 1

    n0, n1 = n00 + n01, n10 + n11
    n = n0 + n1

    if n0 == 0 or n1 == 0:
        return {
            "LR_stat": np.nan, "p_value": np.nan, "critical_value": np.nan,
            "reject_null": None,
            "conclusion": "Not enough exceptions to test independence.",
        }

    pi01 = n01 / n0
    pi11 = n11 / n1
    pi = (n01 + n11) / n

    def _term(count, prob):
        if count == 0:
            return 0.0
        return count * np.log(prob) if prob > 0 else -np.inf

    log_L_null = _term(n00 + n10, 1 - pi) + _term(n01 + n11, pi)
    log_L_alt = (
        _term(n00, 1 - pi01) + _term(n01, pi01) + _term(n10, 1 - pi11) + _term(n11, pi11)
    )

    LR_stat = max(-2 * (log_L_null - log_L_alt), 0.0)
    critical_value = chi2.ppf(1 - significance_level, df=1)
    p_value = 1 - chi2.cdf(LR_stat, df=1)
    reject_null = LR_stat > critical_value

    conclusion = (
        "Reject H0 - exceptions cluster together (not independent over time)."
        if reject_null else
        "Fail to reject H0 - no significant clustering of exceptions detected."
    )

    return {
        "LR_stat": LR_stat, "p_value": p_value, "critical_value": critical_value,
        "reject_null": reject_null, "conclusion": conclusion,
    }


def christoffersen_combined_test(pof_result, independence_result, significance_level=0.05):
    """
    Combined conditional coverage test: LR_cc = LR_pof + LR_ind ~ chi2(2).
    Tests correct coverage AND independence at once - the strongest single
    pass/fail signal for a VaR model.
    """
    if np.isnan(pof_result["LR_stat"]) or np.isnan(independence_result["LR_stat"]):
        return {
            "LR_stat": np.nan, "p_value": np.nan, "critical_value": np.nan,
            "reject_null": None, "conclusion": "Could not compute (missing component test).",
        }

    LR_cc = pof_result["LR_stat"] + independence_result["LR_stat"]
    critical_value = chi2.ppf(1 - significance_level, df=2)
    p_value = 1 - chi2.cdf(LR_cc, df=2)
    reject_null = LR_cc > critical_value

    conclusion = (
        "Reject H0 - model fails the combined coverage + independence test."
        if reject_null else
        "Fail to reject H0 - model passes the combined conditional coverage test."
    )

    return {
        "LR_stat": LR_cc, "p_value": p_value, "critical_value": critical_value,
        "reject_null": reject_null, "conclusion": conclusion,
    }


# ---------------------------------------------------------------------------
# 5. Basel Traffic Light approach (99% VaR / 250-day window, by design)
# ---------------------------------------------------------------------------

def basel_traffic_light(n_exceptions, window=250):
    """
    Basel Committee traffic-light backtesting framework. The 4 / 9
    exception cutoffs below were calibrated SPECIFICALLY for 99% VaR over
    a 250-observation window (expected exceptions ~2.5) - they are not
    meaningful at other confidence levels or window sizes, so this is
    only called for 99% VaR elsewhere in this module.
    """
    if n_exceptions <= 4:
        zone, k_addon = "Green", 0.00
        comment = "Model appears accurate; no supervisory action required."
    elif n_exceptions <= 9:
        zone = "Yellow"
        k_addon = {5: 0.40, 6: 0.50, 7: 0.65, 8: 0.75, 9: 0.85}[n_exceptions]
        comment = "Inconclusive - model may be inaccurate; increased capital multiplier and monitoring."
    else:
        zone, k_addon = "Red", 1.00
        comment = "Model is inaccurate; immediate review and increase in capital multiplier required."

    return {
        "n_exceptions": n_exceptions,
        "window": window,
        "zone": zone,
        "scaling_factor_addon": k_addon,
        "capital_multiplier": 3.00 + k_addon,
        "comment": comment,
    }


# ---------------------------------------------------------------------------
# 6. Orchestration
# ---------------------------------------------------------------------------

def backtest_var_model(rolling_risk_df, portfolio_pnl, confidence_level, model_name,
                        significance_level=0.05, basel_window=250):
    """
    Runs the full backtesting suite for one (model, confidence_level) pair:
    alignment, exception stats, Kupiec POF, Christoffersen independence +
    combined test, and (99% VaR only) the Basel traffic light zone based
    on the most recent `basel_window` observations.
    """
    aligned_df = align_forecast_to_realized(rolling_risk_df, portfolio_pnl)

    exception_stats = calculate_exception_statistics(aligned_df, confidence_level)
    pof_result = kupiec_pof_test(
        exception_stats["n_obs"], exception_stats["n_exceptions"],
        confidence_level, significance_level,
    )
    independence_result = christoffersen_independence_test(aligned_df, significance_level)
    combined_result = christoffersen_combined_test(pof_result, independence_result, significance_level)

    if np.isclose(confidence_level, 0.99):
        recent = aligned_df.tail(basel_window)
        recent_exceptions = int(identify_exceptions(recent).sum())
        traffic_light_result = basel_traffic_light(recent_exceptions, window=len(recent))
        traffic_light_result["n_obs_used"] = len(recent)
        if len(recent) < basel_window:
            traffic_light_result["comment"] += (
                f" (Note: only {len(recent)} observations available, "
                f"fewer than the standard {basel_window}-day window.)"
            )
    else:
        traffic_light_result = {
            "zone": "N/A",
            "comment": "Basel traffic light applies to 99% VaR only.",
        }

    return {
        "model": model_name,
        "confidence_level": confidence_level,
        "aligned_df": aligned_df,
        "exception_stats": exception_stats,
        "kupiec_pof": pof_result,
        "christoffersen_independence": independence_result,
        "christoffersen_combined": combined_result,
        "traffic_light": traffic_light_result,
    }


def summarize_backtest_results(results_list):
    """
    results_list: list of dicts returned by backtest_var_model (one per
    model/confidence-level combination). Returns one comparison table.
    """
    rows = []
    for r in results_list:
        # traffic light is computed on the most recent `basel_window` obs
        # only (99% VaR); "Exceptions (full sample)" below is the total
        # over the ENTIRE backtest - these are two different windows on
        # purpose, don't compare them directly against each other
        tl = r["traffic_light"]
        recent_exceptions = tl.get("n_exceptions", "N/A")
        recent_obs = tl.get("n_obs_used", "N/A")

        rows.append({
            "Model": r["model"],
            "Confidence": f"{r['confidence_level']:.0%}",
            "Obs (full sample)": r["exception_stats"]["n_obs"],
            "Exceptions (full sample)": r["exception_stats"]["n_exceptions"],
            "Exception Rate": f"{r['exception_stats']['exception_rate']:.2%}",
            "Expected Rate": f"{r['exception_stats']['expected_rate']:.2%}",
            "Exceptions (last 250d, for Traffic Light)": recent_exceptions,
            "Kupiec LR": (
                round(r["kupiec_pof"]["LR_stat"], 3)
                if not np.isnan(r["kupiec_pof"]["LR_stat"]) else np.nan
            ),
            "Kupiec Reject H0": r["kupiec_pof"]["reject_null"],
            "Independence Reject H0": r["christoffersen_independence"]["reject_null"],
            "Combined Reject H0": r["christoffersen_combined"]["reject_null"],
            "Traffic Light Zone": r["traffic_light"]["zone"],
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 7. Visualization
# ---------------------------------------------------------------------------

def plot_var_backtest(aligned_df, model_name, confidence_level):
    """
    Plots realized P&L against the (negative) VaR threshold, marking each
    exception - the standard chart a risk desk uses to sanity-check a
    VaR model visually alongside the statistical tests above.
    """
    exceptions = identify_exceptions(aligned_df)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(aligned_df.index, aligned_df["realized_pnl"], label="Realized P&L",
            color="tab:blue", linewidth=0.8, alpha=0.8)
    ax.plot(aligned_df.index, -aligned_df["VaR"], label=f"{confidence_level:.0%} VaR threshold",
            color="tab:red", linewidth=1.2)
    ax.scatter(aligned_df.index[exceptions], aligned_df["realized_pnl"][exceptions],
               color="black", marker="x", s=45, label="Exception (breach)", zorder=5)
    ax.axhline(0, color="grey", linewidth=0.6)

    ax.set_title(f"{model_name} - {confidence_level:.0%} VaR Backtest ({int(exceptions.sum())} exceptions)")
    ax.set_xlabel("Date")
    ax.set_ylabel("₹ P&L")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    return fig