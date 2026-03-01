"""
tests/test_forecasting.py
──────────────────────────────────────────────────────────────────────────────
Unit tests for utils/forecasting.py (Phase 3.1).

All tests use a deterministic 200-point synthetic series with a clear
sinusoidal pattern — fast to fit, reproducible, no random seeds needed
for the assertions to hold.
"""
import pytest
import numpy as np
import pandas as pd

from utils.forecasting import run_sarima_forecast, auto_optimize_sarima


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synthetic_series():
    """200-point sinusoidal + trend series with a proper DatetimeIndex."""
    np.random.seed(0)
    t = np.arange(200)
    values = 30 + 0.05 * t + 8 * np.sin(2 * np.pi * t / 12) + np.random.normal(0, 0.5, 200)
    idx = pd.date_range("2023-01-01", periods=200, freq="5min")
    return pd.Series(values, index=idx, name="avg_cpu")


@pytest.fixture(scope="module")
def sarima_result(synthetic_series):
    """Run one SARIMA forecast, shared across multiple tests."""
    return run_sarima_forecast(
        series=synthetic_series,
        order=(1, 0, 1),
        seasonal_order=(1, 0, 1, 12),
        forecast_steps=24,
    )


# ---------------------------------------------------------------------------
# run_sarima_forecast tests
# ---------------------------------------------------------------------------

class TestRunSarimaForecast:

    def test_returns_required_keys(self, sarima_result):
        """Output dict must contain all documented keys."""
        required = {'mean', 'upper', 'lower', 'aic', 'val_metrics', 'val_predictions'}
        assert required.issubset(sarima_result.keys())

    def test_forecast_length(self, sarima_result):
        """'mean' series must have exactly forecast_steps=24 entries."""
        assert len(sarima_result['mean']) == 24

    def test_forecast_index_is_datetimeindex(self, sarima_result):
        """Future forecast index must be a proper DatetimeIndex."""
        assert isinstance(sarima_result['mean'].index, pd.DatetimeIndex)

    def test_ci_bounds_ordering(self, sarima_result):
        """Upper bound must be >= mean, lower bound must be <= mean everywhere."""
        mean  = sarima_result['mean'].values
        upper = sarima_result['upper'].values
        lower = sarima_result['lower'].values
        assert np.all(upper >= mean - 1e-6), "upper CI should be >= mean"
        assert np.all(lower <= mean + 1e-6), "lower CI should be <= mean"

    def test_val_metrics_are_finite(self, sarima_result):
        """Validation metrics (from hold-out) must be finite floats."""
        vm = sarima_result['val_metrics']
        for metric in ('RMSE', 'MAE', 'MAPE'):
            assert np.isfinite(vm[metric]), f"{metric} should be finite"

    def test_val_metrics_are_non_negative(self, sarima_result):
        """RMSE / MAE / MAPE must be non-negative."""
        vm = sarima_result['val_metrics']
        assert vm['RMSE'] >= 0
        assert vm['MAE']  >= 0
        assert vm['MAPE'] >= 0

    def test_val_predictions_aligned_to_test_window(self, synthetic_series, sarima_result):
        """val_predictions index should lie within the series' DatetimeIndex."""
        vp = sarima_result['val_predictions']
        if not vp.empty:
            assert vp.index[0] >= synthetic_series.index[0]
            assert vp.index[-1] <= synthetic_series.index[-1]

    def test_aic_is_finite_float(self, sarima_result):
        """AIC must be a finite float (guards against convergence failure)."""
        assert isinstance(sarima_result['aic'], float)
        assert np.isfinite(sarima_result['aic'])

    def test_short_series_raises_valueerror(self):
        """Series shorter than 15 observations should raise ValueError."""
        tiny = pd.Series(range(10), index=pd.date_range("2023-01-01", periods=10, freq="5min"))
        with pytest.raises(ValueError, match="too short"):
            run_sarima_forecast(tiny, order=(1, 0, 1), seasonal_order=(0, 0, 0, 0), forecast_steps=5)


# ---------------------------------------------------------------------------
# auto_optimize_sarima tests
# ---------------------------------------------------------------------------

class TestAutoOptimizeSarima:

    def test_returns_order_and_seasonal_order(self, synthetic_series):
        """Return dict must have 'order' and 'seasonal_order' keys."""
        result = auto_optimize_sarima(synthetic_series, seasonal_period=12)
        assert 'order' in result
        assert 'seasonal_order' in result

    def test_order_is_length_3_tuple(self, synthetic_series):
        """ARIMA order must be a 3-element collection (p, d, q)."""
        result = auto_optimize_sarima(synthetic_series, seasonal_period=12)
        assert len(result['order']) == 3

    def test_seasonal_order_is_length_4_tuple(self, synthetic_series):
        """Seasonal order must be a 4-element collection (P, D, Q, s)."""
        result = auto_optimize_sarima(synthetic_series, seasonal_period=12)
        assert len(result['seasonal_order']) == 4

    def test_non_seasonal_period_gives_zero_s(self, synthetic_series):
        """When seasonal_period=1, the 's' component should be 0 or 1 (no seasonality)."""
        result = auto_optimize_sarima(synthetic_series, seasonal_period=1)
        # pmdarima with seasonal=False returns s=0; with m=1 it may return s=1
        assert result['seasonal_order'][3] in (0, 1)

    def test_short_series_raises_valueerror(self):
        """Series shorter than 15 observations should raise ValueError."""
        tiny = pd.Series(range(10), index=pd.date_range("2023-01-01", periods=10, freq="5min"))
        with pytest.raises(ValueError, match="too short"):
            auto_optimize_sarima(tiny, seasonal_period=12)
