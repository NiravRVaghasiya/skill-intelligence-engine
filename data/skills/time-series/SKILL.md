---
name: time-series
display_name: Time Series Forecasting (Seasonality, ARIMA, Prophet)
description: >
  Use when the user wants to forecast a time series, decompose trend and
  seasonality, check stationarity, or build ARIMA/SARIMA/Prophet forecasting
  models. Trigger phrases: "forecast next quarter's sales", "detect
  seasonality in this data", "build an ARIMA model", "forecast this time
  series with Prophet", "is this series stationary". NOT for generic tabular
  regression/classification without temporal ordering (see
  supervised-learning) or general feature construction (see
  feature-engineering).
type: workflow
domain: specialized
level: intermediate
lifecycle: stable
risk_level: low
evidence_level: established-practice
last_verified: 2026-09-21
capabilities:
  - seasonality-trend-decomposition
  - stationarity-testing-adf
  - sarima-order-selection
  - prophet-seasonality-holiday-modeling
  - walk-forward-backtesting
requires:
conflicts:
related:
  - supervised-learning
  - feature-engineering
  - model-evaluation
inputs: A timestamped numeric series (e.g. a pandas DataFrame with a date column) to be forecast.
outputs: A fitted SARIMA and/or Prophet forecaster, decomposition components, and walk-forward backtest error metrics.
version_constraints:
  - "statsmodels: `statsmodels.tsa.seasonal.STL` (robust STL decomposition) added around statsmodels 0.11 (2019); earlier versions only had the classical additive/multiplicative `seasonal_decompose` — model-knowledge estimate, not live-verified this session."
---

## Overview
Covers the forecasting-specific pipeline: getting a series onto a clean
fixed-frequency index, decomposing trend/seasonality, testing stationarity,
and fitting classical (SARIMA) and modern (Prophet) forecasters with
walk-forward backtesting. Generic regression mechanics (train/test split,
metric definitions) live in `supervised-learning`/`model-evaluation` — this
skill covers what's specifically different about sequential, autocorrelated
data.

## Workflow
1. **Load, index, and regularize frequency.** Forecasting models assume a
   fixed time step; gaps or irregular timestamps break autocorrelation.
   ```python
   import pandas as pd

   df = pd.read_csv("sales.csv", parse_dates=["date"])
   df = df.set_index("date").sort_index()
   df = df.asfreq("D")                      # enforce daily frequency
   df["sales"] = df["sales"].interpolate(limit=3)   # fill small gaps, flag large ones
   ```
2. **Decompose trend and seasonality.** STL is more robust than classical
   decomposition to outliers and non-constant seasonal amplitude.
   ```python
   from statsmodels.tsa.seasonal import STL

   stl = STL(df["sales"], period=7, robust=True)
   res = stl.fit()
   trend, seasonal, resid = res.trend, res.seasonal, res.resid
   ```
3. **Test stationarity before fitting ARIMA.** ARIMA assumes stationarity;
   fitting on a trending series without differencing biases forecasts. The
   ADF test's `p < 0.05` cutoff is a convention, not a proof — treat the
   resulting `d` as a starting point, not a final answer: some series need
   `d=2`, and a single ADF read can disagree with a KPSS test, so let
   `auto_arima`'s own order search (step 4) confirm or override it rather
   than hard-coding `d` from this test alone.
   ```python
   from statsmodels.tsa.stattools import adfuller

   stat, pvalue, *_ = adfuller(df["sales"].dropna())
   print(f"ADF stat={stat:.3f}, p-value={pvalue:.4f}")
   d = 0 if pvalue < 0.05 else 1   # starting guess; let auto_arima confirm/override d
   ```
4. **Fit SARIMA with an auto-selected order.** Use `pmdarima` to search
   (p,d,q)(P,D,Q,s) rather than hand-tuning by ACF/PACF plots alone.
   ```python
   import pmdarima as pm

   auto_model = pm.auto_arima(
       df["sales"], seasonal=True, m=7,
       d=None, D=None, trace=False, suppress_warnings=True,
       stepwise=True,
   )
   forecast, conf_int = auto_model.predict(n_periods=14, return_conf_int=True)
   ```
5. **Fit Prophet as a complementary model.** Prophet handles multiple
   seasonalities and holiday effects with less manual differencing.
   ```python
   from prophet import Prophet

   prophet_df = df.reset_index().rename(columns={"date": "ds", "sales": "y"})
   m = Prophet(yearly_seasonality=True, weekly_seasonality=True)
   m.add_country_holidays(country_name="US")
   m.fit(prophet_df)

   future = m.make_future_dataframe(periods=14)
   fcst = m.predict(future)[["ds", "yhat", "yhat_lower", "yhat_upper"]]
   ```
6. **Backtest with walk-forward validation.** Random splits leak future
   information; forecasts must only ever be evaluated against strictly
   later, unseen periods.
   ```python
   import numpy as np
   from sklearn.metrics import mean_absolute_percentage_error

   errors = []
   window = 90
   for cutoff in range(window, len(df) - 14, 14):
       train, test = df["sales"].iloc[:cutoff], df["sales"].iloc[cutoff:cutoff + 14]
       model = pm.auto_arima(train, seasonal=True, m=7, suppress_warnings=True)
       preds = model.predict(n_periods=len(test))
       errors.append(mean_absolute_percentage_error(test, preds))
   print(f"mean MAPE over folds: {np.mean(errors):.3f}")
   ```

## Gotchas
- **Random train/test splits leak the future.** Unlike i.i.d. tabular data,
  time series must be split chronologically (or walk-forward) — a random
  k-fold split lets the model "see" future values via autocorrelation with
  neighboring points.
- **Fitting ARIMA on a non-stationary series.** Skipping the ADF test and
  fitting on a trending/seasonal-non-differenced series produces confident
  but biased forecasts; always check stationarity and set `d`/`D` deliberately.
- **Prophet needs enough history for the seasonalities you enable.**
  Estimating a seasonal component reliably requires multiple full cycles of
  it in the training data, not a fixed calendar length — as a rough guide,
  yearly seasonality with under roughly 2 years of daily/weekly data is
  often mostly fitting noise, but the actual amount needed depends on data
  frequency and noise level. Disable seasonal components you don't have
  enough cycles to estimate.
- **Lookahead in engineered lag/rolling features.** A rolling mean or lag
  feature computed with `center=True` or using the full series (rather than
  only past values) leaks future information into the "past" feature.
- **Irregular timestamps silently break ACF/PACF and STL.** Always
  `asfreq()` (or resample) to a fixed cadence and explicitly handle
  resulting NaNs before decomposition — don't assume the raw timestamps are
  evenly spaced.

## References
- [statsmodels: Time Series Analysis](https://www.statsmodels.org/stable/tsa.html) — STL, ADF test, SARIMAX reference implementation.
- [Prophet documentation](https://facebook.github.io/prophet/docs/quick_start.html) — official quick-start and seasonality/holiday configuration.
- [pmdarima documentation](https://alkaline-ml.com/pmdarima/) — `auto_arima` order search, the practical alternative to manual ACF/PACF tuning.
