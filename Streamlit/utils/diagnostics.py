import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, acf
from statsmodels.stats.diagnostic import acorr_ljungbox


def perform_adf_test(series):
    """
    Performs the Augmented Dickey-Fuller test for stationarity.
    Returns: (p_value, is_stationary)
    """
    if series.empty or series.nunique() <= 1:
        return 1.0, False
    result = adfuller(series.dropna())
    p_value = result[1]
    return p_value, p_value < 0.05


def perform_ljung_box_test(series):
    """
    Performs the Ljung-Box test for white noise.
    Returns: (p_value, is_not_white_noise)
    """
    if series.empty or series.nunique() <= 1:
        return 1.0, False
    # lags=10 is a common default for white noise check
    result = acorr_ljungbox(series.dropna(), lags=[10], return_df=True)
    p_value = result['lb_pvalue'].iloc[0]
    # Small p-value means we reject the null hypothesis of white noise
    return p_value, p_value < 0.05


# ---------------------------------------------------------------------------
# 3-State Gatekeeper Classification
# ---------------------------------------------------------------------------

# Status constants — used for routing logic in app.py
STATUS_STATIONARY     = 'stationary'      # ✅ ADF pass + LB pass
STATUS_NON_STATIONARY = 'non_stationary'  # ⚠️ ADF fail + LB pass (has structure, needs differencing)
STATUS_WHITE_NOISE    = 'white_noise'     # ❌ LB fail (no learnable pattern)


def classify_vm_suitability(adf_p: float, ljung_p: float) -> dict:
    """
    Maps ADF and Ljung-Box p-values to one of three Gatekeeper states.

    Decision table:
      LB p < 0.05  AND ADF p < 0.05  →  STATUS_STATIONARY      (fully eligible)
      LB p < 0.05  AND ADF p >= 0.05 →  STATUS_NON_STATIONARY  (eligible; needs differencing)
      LB p >= 0.05 (any ADF result)  →  STATUS_WHITE_NOISE      (not eligible)

    The Ljung-Box test is the primary gate: a series that is white noise has no
    exploitable autocorrelation structure regardless of stationarity.

    Returns a dict with keys:
      status      (str)   — one of the STATUS_* constants above
      icon        (str)   — emoji shorthand for Fleet table
      label       (str)   — short human-readable label
      message     (str)   — actionable guidance for the analyst
      is_eligible (bool)  — whether Phase 3 modelling is permitted
    """
    is_not_noise    = ljung_p < 0.05
    is_stationary   = adf_p   < 0.05

    if is_not_noise and is_stationary:
        return {
            'status':      STATUS_STATIONARY,
            'icon':        '✅',
            'label':       'Stationary',
            'message':     'Series is stationary and structured. All baseline and advanced models may be applied.',
            'is_eligible': True,
        }
    elif is_not_noise and not is_stationary:
        return {
            'status':      STATUS_NON_STATIONARY,
            'icon':        '⚠️',
            'label':       'Non-Stationary',
            'message':     (
                'Series has drift or trend (ADF p ≥ 0.05) but retains autocorrelation structure. '
                'Advanced differencing models (e.g., ARIMA with d > 0) are recommended in Phase 3.'
            ),
            'is_eligible': True,
        }
    else:
        return {
            'status':      STATUS_WHITE_NOISE,
            'icon':        '❌',
            'label':       'White Noise',
            'message':     (
                'Data is purely random (Ljung-Box p ≥ 0.05). '
                'No autocorrelation structure was detected — forecasting is not statistically viable.'
            ),
            'is_eligible': False,
        }


def calculate_refined_seasonality(series, granularity):
    """
    Detects seasonality using ACF with adaptive search ranges and thresholds.
    
    Ranges:
    - Hourly (1-hour): Lags 4 to 168 (1 week)
    - 5-min (Native): Lags 48 (4h) to 2016 (1 week)
    
    Threshold: ACF peak > 0.2
    """
    if series.empty or series.nunique() <= 1:
        return 1

    if granularity == '1-hour':
        min_lag, max_lag = 4, 168
    else:  # '5-min'
        min_lag, max_lag = 48, 2016

    if len(series) <= max_lag:
        max_lag = len(series) - 1

    if max_lag < min_lag:
        return 1

    acf_values = acf(series.dropna(), nlags=max_lag)

    search_range_acf = acf_values[min_lag : max_lag + 1]

    if len(search_range_acf) == 0:
        return 1

    max_acf_idx = np.argmax(search_range_acf)
    max_acf_val = search_range_acf[max_acf_idx]
    detected_lag = min_lag + max_acf_idx

    if max_acf_val > 0.2:
        return int(detected_lag)
    else:
        return 1
