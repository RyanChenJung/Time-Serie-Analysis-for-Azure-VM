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

from utils.forecasting import (
    run_sarima_forecast, auto_optimize_sarima, run_holt_winters_forecast,
    auto_optimize_holt_winters, build_exog_df,
)


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


# ---------------------------------------------------------------------------
# run_holt_winters_forecast tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def hw_result(synthetic_series):
    """Run one HW forecast, shared across multiple tests."""
    return run_holt_winters_forecast(
        series=synthetic_series,
        trend='add',
        seasonal='add',
        seasonal_periods=12,
        damped=False,
        forecast_steps=24,
    )


class TestRunHoltWinters:

    def test_returns_required_keys(self, hw_result):
        """Output dict must contain all documented keys."""
        required = {'mean', 'upper', 'lower', 'aic', 'val_metrics',
                    'val_predictions', 'val_residuals', 'model_label'}
        assert required.issubset(hw_result.keys())

    def test_forecast_length(self, hw_result):
        """'mean' series must have exactly forecast_steps=24 entries."""
        assert len(hw_result['mean']) == 24

    def test_forecast_index_is_datetimeindex(self, hw_result):
        """Future forecast index must be a proper DatetimeIndex."""
        assert isinstance(hw_result['mean'].index, pd.DatetimeIndex)

    def test_pi_bounds_ordering(self, hw_result):
        """upper >= mean and lower <= mean at every step."""
        mean  = hw_result['mean'].values
        upper = hw_result['upper'].values
        lower = hw_result['lower'].values
        assert np.all(upper >= mean - 1e-6), "upper PI should be >= mean"
        assert np.all(lower <= mean + 1e-6), "lower PI should be <= mean"

    def test_val_metrics_are_finite(self, hw_result):
        """Hold-out metrics must be finite floats."""
        vm = hw_result['val_metrics']
        for metric in ('RMSE', 'MAE', 'MAPE'):
            assert np.isfinite(vm[metric]), f"{metric} should be finite"

    def test_val_metrics_are_non_negative(self, hw_result):
        """RMSE / MAE / MAPE must be non-negative."""
        vm = hw_result['val_metrics']
        assert vm['RMSE'] >= 0
        assert vm['MAE']  >= 0
        assert vm['MAPE'] >= 0

    def test_val_residuals_aligned(self, hw_result):
        """val_residuals must have same length as val_predictions."""
        vp = hw_result['val_predictions']
        vr = hw_result['val_residuals']
        if not vp.empty and not vr.empty:
            assert len(vr) == len(vp), "val_residuals length mismatch with val_predictions"

    def test_aperiodic_guard_no_raise(self, synthetic_series):
        """Passing seasonal_periods=1 must not raise (seasonal forced to None)."""
        result = run_holt_winters_forecast(
            series=synthetic_series,
            trend='add',
            seasonal='add',   # should be overridden to None internally
            seasonal_periods=1,
            damped=False,
            forecast_steps=12,
        )
        assert 'mean' in result
        assert len(result['mean']) == 12


# ---------------------------------------------------------------------------
# auto_optimize_holt_winters tests
# ---------------------------------------------------------------------------

class TestAutoOptimizeHoltWinters:

    def test_returns_required_keys(self, synthetic_series):
        """Return dict must contain all documented keys."""
        result = auto_optimize_holt_winters(synthetic_series, seasonal_periods=12)
        assert {'best_config', 'best_rmse', 'best_label', 'trace'}.issubset(result.keys())

    def test_best_rmse_is_finite_and_non_negative(self, synthetic_series):
        """best_rmse must be a non-negative finite float."""
        result = auto_optimize_holt_winters(synthetic_series, seasonal_periods=12)
        assert np.isfinite(result['best_rmse'])
        assert result['best_rmse'] >= 0.0

    def test_trace_covers_all_combos(self, synthetic_series):
        """Trace must have one entry per evaluated combo (skipped or passed)."""
        import itertools
        seasonal_periods = 12
        expected_count = sum(
            1 for t, s, d in itertools.product(
                ['add', 'mul', None], ['add', 'mul', None], [True, False]
            )
            if not (d and t is None)
        )
        result = auto_optimize_holt_winters(synthetic_series, seasonal_periods=seasonal_periods)
        assert len(result['trace']) == expected_count

    def test_aperiodic_mode_only_non_seasonal_combos(self, synthetic_series):
        """When seasonal_periods=1, all trace entries must have seasonal=None."""
        result = auto_optimize_holt_winters(synthetic_series, seasonal_periods=1)
        for entry in result['trace']:
            assert entry['seasonal'] is None, (
                f"Expected seasonal=None in aperiodic mode, got {entry['seasonal']}"
            )


# ---------------------------------------------------------------------------
# build_exog_df + ARIMAX integration tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synthetic_vm_df():
    """Minimal vm_df with timestamp_dt, avg_cpu, max_cpu, min_cpu columns."""
    np.random.seed(1)
    n = 200
    idx = pd.date_range("2023-01-01", periods=n, freq="5min")
    avg = 30 + 5 * np.sin(2 * np.pi * np.arange(n) / 12) + np.random.normal(0, 0.3, n)
    mx  = avg + np.abs(np.random.normal(2, 0.5, n))
    mn  = avg - np.abs(np.random.normal(2, 0.5, n))
    return pd.DataFrame({'timestamp_dt': idx, 'avg_cpu': avg, 'max_cpu': mx, 'min_cpu': mn})


class TestBuildExogDf:

    def test_output_columns(self, synthetic_vm_df):
        """build_exog_df must return max_cpu and min_cpu columns."""
        result = build_exog_df(synthetic_vm_df)
        assert list(result.columns) == ['max_cpu', 'min_cpu']

    def test_length_is_n_minus_1(self, synthetic_vm_df):
        """After shift(1).iloc[1:], length must be len(vm_df) - 1."""
        result = build_exog_df(synthetic_vm_df)
        assert len(result) == len(synthetic_vm_df) - 1

    def test_no_nans(self, synthetic_vm_df):
        """Output must contain no NaN values after the lag."""
        result = build_exog_df(synthetic_vm_df)
        assert not result.isna().any().any()

    def test_causal_lag(self, synthetic_vm_df):
        """After shift(1).iloc[1:], exog.iloc[0] must equal vm_df.iloc[0].
        
        Mechanics: shift(1) moves row[0] → row[1]; iloc[1:] then exposes row[1]
        as exog.iloc[0], which holds the original row[0] value.
        """
        result = build_exog_df(synthetic_vm_df)
        expected_max = synthetic_vm_df['max_cpu'].iloc[0]  # original row 0 value
        assert abs(result['max_cpu'].iloc[0] - expected_max) < 1e-10


class TestArimax:

    def test_model_name_is_arimax_with_exog(self, synthetic_vm_df):
        """run_sarima_forecast with exog_df must return model_name='ARIMAX'."""
        series   = synthetic_vm_df.set_index('timestamp_dt')['avg_cpu']
        exog_df  = build_exog_df(synthetic_vm_df)
        result   = run_sarima_forecast(series, order=(1, 0, 0), seasonal_order=(0, 0, 0, 0),
                                       forecast_steps=12, exog_df=exog_df)
        assert result['model_name'] == 'ARIMAX'

    def test_model_name_is_sarima_without_exog(self, synthetic_vm_df):
        """run_sarima_forecast without exog must return model_name='SARIMA'."""
        series = synthetic_vm_df.set_index('timestamp_dt')['avg_cpu']
        result = run_sarima_forecast(series, order=(1, 0, 0), seasonal_order=(0, 0, 0, 0),
                                     forecast_steps=12)
        assert result['model_name'] == 'SARIMA'

    def test_arimax_returns_all_required_keys(self, synthetic_vm_df):
        """ARIMAX result dict must contain all same keys as SARIMA."""
        series  = synthetic_vm_df.set_index('timestamp_dt')['avg_cpu']
        exog_df = build_exog_df(synthetic_vm_df)
        result  = run_sarima_forecast(series, order=(1, 0, 0), seasonal_order=(0, 0, 0, 0),
                                      forecast_steps=12, exog_df=exog_df)
        for key in ('mean', 'upper', 'lower', 'aic', 'val_metrics', 'val_predictions', 'model_name'):
            assert key in result, f"Missing key: {key}"

    def test_arimax_forecast_length(self, synthetic_vm_df):
        """ARIMAX forecast horizon must match forecast_steps."""
        series  = synthetic_vm_df.set_index('timestamp_dt')['avg_cpu']
        exog_df = build_exog_df(synthetic_vm_df)
        result  = run_sarima_forecast(series, order=(1, 0, 0), seasonal_order=(0, 0, 0, 0),
                                      forecast_steps=24, exog_df=exog_df)
        assert len(result['mean']) == 24

    def test_arimax_val_rmse_finite(self, synthetic_vm_df):
        """ARIMAX validation RMSE must be a finite positive number."""
        series  = synthetic_vm_df.set_index('timestamp_dt')['avg_cpu']
        exog_df = build_exog_df(synthetic_vm_df)
        result  = run_sarima_forecast(series, order=(1, 0, 0), seasonal_order=(0, 0, 0, 0),
                                      forecast_steps=12, exog_df=exog_df)
        rmse = result['val_metrics']['RMSE']
        assert np.isfinite(rmse) and rmse >= 0
