import pytest
import pandas as pd
import numpy as np
from utils.diagnostics import (
    perform_adf_test, perform_ljung_box_test,
    calculate_refined_seasonality, classify_vm_suitability,
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
