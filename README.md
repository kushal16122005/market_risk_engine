# Market Risk Engine

An end-to-end market risk engine for a multi-asset equity portfolio, built in Python. Computes Value at Risk (VaR) and Expected Shortfall (ES) under three methodologies, generates rolling risk forecasts, statistically backtests each model, decomposes portfolio risk by asset, runs stress scenarios, and consolidates everything into a single risk dashboard.

## Portfolio

5 NSE-listed equities, equal-weighted, ₹1,00,00,000 (₹1 crore) notional:

`RELIANCE.NS`, `ESCORTS.NS`, `INFY.NS`, `ABB.NS`, `HDFCBANK.NS`

## What this project does

| Stage | What it produces |
|---|---|
| **VaR & ES (3 models)** | Historical Simulation, Variance-Covariance (Delta-Normal), and Monte Carlo — each at 95% and 99% confidence |
| **Rolling forecasts** | 250-day rolling VaR/ES for all three models, tracking risk over time |
| **Backtesting** | Kupiec Proportion-of-Failures test, Christoffersen independence + combined test, Basel Traffic Light zones (99% VaR) — validates whether each model's exception rate matches theory |
| **Risk decomposition** | Component VaR/ES per asset, per model — shows exactly which holding drives portfolio risk, with contributions that sum exactly to the total |
| **Stress testing** | Historical scenario replay, hypothetical shock scenarios, and correlated shock propagation via historical betas |
| **Dashboard** | One consolidated figure + an auto-generated plain-English executive summary, built from the live notebook state |

## Project structure

```
Risk_Engine/
├── data/
│   └── raw/
│       └── prices.csv              # cached price data (tracked in git for reproducibility)
├── src/
│   ├── data_loader.py               # price download, with local caching + error handling
│   ├── returns.py                   # simple / log returns
│   ├── portfolio.py                 # name-based weight alignment, portfolio returns & P&L
│   ├── risk_metrics.py              # descriptive statistics (mean, vol, skew, kurtosis)
│   ├── historical_simulation_var.py
│   ├── historical_simulation_es.py
│   ├── variance_covariance_var.py   # parametric (Delta-Normal) VaR
│   ├── variance_covariance_es.py    # closed-form normal Expected Shortfall
│   ├── monte_carlo_var.py
│   ├── monte_carlo_es.py
│   ├── rolling_var.py               # 250-day rolling VaR/ES, all 3 models
│   ├── backtesting.py               # Kupiec POF, Christoffersen tests, Basel traffic light
│   ├── risk_decomposition.py        # Component VaR/ES, per asset, per model
│   ├── stress_testing.py            # historical & hypothetical scenario stress testing
│   └── dashboard.py                 # final consolidated dashboard
├── main.ipynb                       # orchestration notebook — run this top to bottom
├── requirements.txt
└── .gitignore
```

## Methodology

**Historical Simulation VaR/ES** — non-parametric; VaR is the empirical quantile of historical portfolio returns, ES is the average loss beyond that quantile. No distributional assumption, captures real fat tails.

**Variance-Covariance (Delta-Normal) VaR/ES** — assumes portfolio returns are normally distributed:
```
VaR = (z_α · σ_p − μ_p) · V
ES  = (σ_p · φ(z_α) / (1 − α) − μ_p) · V
```
where `σ_p = √(w'Σw)` is portfolio volatility from the covariance matrix, `z_α = Φ⁻¹(α)`.

**Monte Carlo VaR/ES** — simulates 10,000 portfolio return paths from a multivariate normal fitted to historical mean/covariance, then applies the same empirical-quantile logic as Historical Simulation.

**Backtesting** — every rolling VaR forecast is aligned to the **next trading day's** realized P&L (not the same day, to avoid look-ahead bias), then tested with:
- *Kupiec POF*: does the exception rate match the expected `1 − confidence_level`?
- *Christoffersen independence*: do exceptions cluster in time rather than spreading out?
- *Basel Traffic Light*: Green / Yellow / Red zone based on exceptions in the most recent 250 days (99% VaR only, per Basel's original calibration).

**Risk decomposition** — Component VaR/ES per asset, additive by construction (components sum exactly to the portfolio total), computed three ways: analytically for Variance-Covariance, via historical scenario/tail-averaging for Historical Simulation, and via simulated scenario/tail-averaging for Monte Carlo.

**Stress testing** — replays the worst actual 1/5/10/20-day moves the portfolio has lived through, applies hypothetical shocks (uniform, sector, single-stock), and propagates a shock to one asset through the rest of the portfolio using historical betas.

## Setup

```powershell
git clone <repo-url>
cd Risk_Engine

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

pip install -r requirements.txt

jupyter notebook main.ipynb
```
Run all cells top to bottom. `data/raw/prices.csv` is already cached in the repo, so the notebook runs fully offline without hitting the `yfinance` API — delete that file (or pass `use_cache=False` in `data_loader.download_prices`) to force a fresh download.

## Sample findings

*(Generated automatically by `src/dashboard.py` from the live notebook state — rerun the notebook and this updates itself.)*

> Portfolio: 5 NSE-listed equities, ₹10,000,000 notional, equal-weighted. 99% 1-day VaR ranges from ₹228,689 to ₹269,178 across the three models (Historical ₹269,178, Variance-Covariance ₹232,045, Monte Carlo ₹228,689). Backtesting at 99%: Historical Simulation passed the Kupiec coverage test; Variance-Covariance, Monte Carlo failed it. RELIANCE.NS is the largest single source of portfolio risk, contributing 38.7% of total historical VaR. Worst stress scenario tested: "Uniform market crash -10%", resulting in a loss of ₹1,000,000.

The portfolio's excess kurtosis (fat tails vs. a normal distribution) explains why Variance-Covariance and Monte Carlo — both of which assume normality — underestimate 99% tail risk and fail backtesting, while Historical Simulation, which uses the empirical distribution directly, passes.

## Limitations & possible extensions

- Single asset class (equities only), 5 names — no cross-asset or factor-model risk
- Monte Carlo currently draws from the same fitted normal distribution as Variance-Covariance; a Student-t or bootstrap-resampled simulation would better capture fat tails and differentiate it from the parametric model
- No multi-day VaR scaling or liquidity-adjusted VaR
- Stress scenarios are equity-shock-based; no rates/FX/credit factor shocks

## Tech stack

Python · pandas · NumPy · SciPy · Matplotlib · yfinance · Jupyter