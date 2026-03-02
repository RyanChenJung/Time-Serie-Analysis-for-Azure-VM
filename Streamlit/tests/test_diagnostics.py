import pytest
import pandas as pd
import numpy as np
from utils.diagnostics import (
    perform_adf_test, perform_ljung_box_test,
    calculate_refined_seasonality, classify_vm_suitability,
    calculate_pacf,
    STATUS_STATIONARY, STATUS_NON_STATIONARY, STATUS_WHITE_NOISE
)


def test_perform_adf_test():
    np.random.seed(42)
    stationary = pd.Series(np.random.normal(0, 1, 100))
    p, is_stationary = perform_adf_test(stationary)
    assert p < 0.05
    assert is_stationary == True

    non_stationary = pd.Series(np.cumsum(np.random.normal(0, 1, 100)))
    p, is_stationary = perform_adf_test(non_stationary)
    assert p >= 0.05
    assert is_stationary == False


def test_perform_ljung_box_test():
    np.random.seed(42)
    wn = pd.Series(np.random.normal(0, 1, 100))
    p, is_not_wn = perform_ljung_box_test(wn)
    assert p > 0.05
    assert is_not_wn == False

    x = np.linspace(0, 10, 100)
    not_wn = pd.Series(np.sin(x))
    p, is_not_wn = perform_ljung_box_test(not_wn)
    assert p < 0.05
    assert is_not_wn == True


def test_calculate_refined_seasonality():
    np.random.seed(0)
    x = np.arange(200)
    period = 24
    seasonal = pd.Series(10 * np.sin(2 * np.pi * x / period) + np.random.normal(0, 1, 200))
    lag = calculate_refined_seasonality(seasonal, '1-hour')
    assert lag == 24

    wn = pd.Series(np.random.normal(0, 1, 200))
    lag = calculate_refined_seasonality(wn, '1-hour')
    assert lag == 1


# ---------------------------------------------------------------------------
# 3-State Gatekeeper Tests
# ---------------------------------------------------------------------------

def test_classify_stationary():
    """ADF p < 0.05 AND LB p < 0.05 → STATUS_STATIONARY, eligible."""
    result = classify_vm_suitability(adf_p=0.01, ljung_p=0.001)
    assert result['status']      == STATUS_STATIONARY
    assert result['icon']        == '✅'
    assert result['is_eligible'] == True
    for key in ('status', 'icon', 'label', 'message', 'is_eligible'):
        assert key in result, f"Missing key: {key}"


def test_classify_non_stationary():
    """ADF p >= 0.05 AND LB p < 0.05 → STATUS_NON_STATIONARY, still eligible."""
    result = classify_vm_suitability(adf_p=0.20, ljung_p=0.001)
    assert result['status']      == STATUS_NON_STATIONARY
    assert result['icon']        == '⚠️'
    assert result['is_eligible'] == True
    assert 'differencing' in result['message'].lower()


def test_classify_white_noise():
    """LB p >= 0.05 (any ADF) → STATUS_WHITE_NOISE, not eligible."""
    result = classify_vm_suitability(adf_p=0.50, ljung_p=0.30)
    assert result['status']      == STATUS_WHITE_NOISE
    assert result['icon']        == '❌'
    assert result['is_eligible'] == False

    # ADF passes but LB fails — LB is the primary gate, so still white noise
    result_b = classify_vm_suitability(adf_p=0.01, ljung_p=0.20)
    assert result_b['status']      == STATUS_WHITE_NOISE
    assert result_b['is_eligible'] == False


def test_classify_boundary_conditions():
    """Exact threshold values sit on the correct side of p < 0.05."""
    # p = 0.05 is NOT < 0.05 → LB fails → white noise
    result = classify_vm_suitability(adf_p=0.05, ljung_p=0.05)
    assert result['status'] == STATUS_WHITE_NOISE

    # p = 0.04999 IS < 0.05 → both pass → stationary
    result2 = classify_vm_suitability(adf_p=0.04999, ljung_p=0.04999)
    assert result2['status'] == STATUS_STATIONARY


# ---------------------------------------------------------------------------
# PACF Tests
# ---------------------------------------------------------------------------

def test_calculate_pacf_ar1_spike():
    """
    An AR(1) process with φ=0.9 should have a dominant PACF spike at lag 1
    that is much larger than at lag 5 (the 'cut-off' property of PACF for AR).
    """
    np.random.seed(7)
    n = 500
    phi = 0.9
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + np.random.normal(0, 0.5)
    series = pd.Series(x)

    pacf_vals = calculate_pacf(series, nlags=20)

    # lag-0 must be 1.0
    assert abs(pacf_vals[0] - 1.0) < 1e-6, "PACF lag-0 must equal 1.0"

    # lag-1 should be close to φ=0.9 and strongly dominant
    assert pacf_vals[1] > 0.7, f"Expected PACF[1] > 0.7 for AR(1) φ=0.9, got {pacf_vals[1]:.3f}"

    # lag-5 onwards should be much smaller (PACF of AR(1) cuts off after lag 1)
    assert abs(pacf_vals[5]) < 0.2, f"Expected PACF[5] ≈ 0 for AR(1), got {pacf_vals[5]:.3f}"


def test_calculate_pacf_white_noise_bounded():
    """
    PACF values for white noise should all sit near zero — no lag beyond 0
    should exceed ±0.35 for a reasonable sample size (n=200, 95 % CI ≈ ±0.14).
    We use a generous bound to avoid flaky test failures.
    """
    np.random.seed(99)
    wn = pd.Series(np.random.normal(0, 1, 200))
    pacf_vals = calculate_pacf(wn, nlags=30)

    # Skip lag-0 (always 1.0); all others should be small
    for lag, v in enumerate(pacf_vals[1:], start=1):
        assert abs(v) < 0.35, f"PACF[{lag}] = {v:.3f} exceeds ±0.35 for white noise"


def test_calculate_pacf_short_series_guard():
    """Very short series should not raise — function returns [1.0] gracefully."""
    short = pd.Series([1.0, 2.0, 3.0])
    result = calculate_pacf(short, nlags=20)
    assert len(result) >= 1
    assert result[0] == pytest.approx(1.0)
