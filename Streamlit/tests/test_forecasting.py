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
    auto_optimize_holt_winters, build_exog_df, run_var_forecast,
    _BoxCoxLayer,
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


# ---------------------------------------------------------------------------
# run_var_forecast tests  (Phase 3.3b)
# ---------------------------------------------------------------------------

VAR_COLS = ['avg_cpu', 'max_cpu', 'min_cpu']
FORECAST_STEPS = 12


@pytest.fixture(scope="module")
def synthetic_vm_df_for_var():
    """200-row vm_df with timestamp_dt + avg/max/min columns, stationary-ish."""
    np.random.seed(42)
    n   = 200
    idx = pd.date_range("2023-01-01", periods=n, freq="5min")
    avg = 30 + 5 * np.sin(2 * np.pi * np.arange(n) / 12) + np.random.normal(0, 0.5, n)
    mx  = avg + np.abs(np.random.normal(2, 0.4, n))
    mn  = avg - np.abs(np.random.normal(2, 0.4, n))
    return pd.DataFrame({'timestamp_dt': idx, 'avg_cpu': avg, 'max_cpu': mx, 'min_cpu': mn})


@pytest.fixture(scope="module")
def var_result(synthetic_vm_df_for_var):
    """Run one VAR forecast shared across all TestRunVarForecast tests."""
    return run_var_forecast(synthetic_vm_df_for_var, forecast_steps=FORECAST_STEPS)


class TestRunVarForecast:

    def test_returns_required_keys(self, var_result):
        """Output dict must contain all documented keys."""
        required = {
            'forecasts', 'val_metrics', 'val_predictions',
            'lag_order', 'diff_order', 'adf_pvalues',
            'granger', 'var_residuals', 'aic',
        }
        assert required.issubset(var_result.keys())

    def test_forecasts_has_all_variables(self, var_result):
        """'forecasts' must contain a Series for each of the 3 variables."""
        for col in VAR_COLS:
            assert col in var_result['forecasts'], f"Missing forecast for {col}"
            fc = var_result['forecasts'][col]
            assert len(fc) == FORECAST_STEPS, (
                f"{col} forecast length {len(fc)} != {FORECAST_STEPS}"
            )

    def test_forecast_index_is_datetimeindex(self, var_result):
        """All three forecast Series must have a DatetimeIndex in the future."""
        for col in VAR_COLS:
            assert isinstance(var_result['forecasts'][col].index, pd.DatetimeIndex), (
                f"{col} forecast index is not a DatetimeIndex"
            )

    def test_val_metrics_finite_and_non_negative(self, var_result):
        """Validation metrics for every variable must be finite and ≥ 0."""
        for col in VAR_COLS:
            vm = var_result['val_metrics'][col]
            for k in ('RMSE', 'MAE', 'MAPE'):
                assert np.isfinite(vm[k]), f"{col}.{k} is not finite"
                assert vm[k] >= 0,        f"{col}.{k} is negative"

    def test_granger_matrix_exhaustive(self, var_result):
        """Granger dict must contain exactly n*(n-1) = 6 ordered pairs."""
        expected_pairs = {
            f"{c}→{e}"
            for c in VAR_COLS for e in VAR_COLS if c != e
        }
        assert set(var_result['granger'].keys()) == expected_pairs

    def test_granger_entries_have_required_keys(self, var_result):
        """Every Granger entry must have p_value, significant, max_lag."""
        for key, entry in var_result['granger'].items():
            for field in ('p_value', 'significant', 'max_lag'):
                assert field in entry, f"Granger entry '{key}' missing '{field}'"

    def test_var_residuals_shape(self, var_result):
        """var_residuals must be a DataFrame with the 3 variable columns."""
        resid = var_result['var_residuals']
        assert isinstance(resid, pd.DataFrame)
        for col in VAR_COLS:
            assert col in resid.columns, f"Residuals missing column {col}"
        # Residuals lose the first p rows (lag_order); must be > 0
        assert len(resid) > 0

    def test_diff_order_is_valid(self, var_result):
        """diff_order must be 0, 1, or 2 (iterative ADF supports up to d=2)."""
        assert var_result['diff_order'] in (0, 1, 2)

    def test_auto_differencing_for_nonstationary_data(self, synthetic_vm_df_for_var):
        """Force a non-stationary series by adding a strong trend;
        diff_order must become 1 on the resulting data."""
        df = synthetic_vm_df_for_var.copy()
        # Add a steep trend so avg_cpu / max_cpu / min_cpu all fail ADF
        trend = np.linspace(0, 200, len(df))
        df['avg_cpu'] += trend
        df['max_cpu'] += trend
        df['min_cpu'] += trend
        result = run_var_forecast(df, forecast_steps=6)
        assert result['diff_order'] == 1, (
            "Expected diff_order=1 for strongly trended data"
        )

    def test_short_series_raises_valueerror(self):
        """DataFrames with fewer than 30 rows must raise ValueError."""
        idx = pd.date_range("2023-01-01", periods=20, freq="5min")
        df  = pd.DataFrame({
            'timestamp_dt': idx,
            'avg_cpu': np.random.rand(20),
            'max_cpu': np.random.rand(20),
            'min_cpu': np.random.rand(20),
        })
        with pytest.raises(ValueError, match="30"):
            run_var_forecast(df, forecast_steps=5)

    def test_auto_differencing_v2(self, synthetic_vm_df_for_var):
        """
        test_auto_differencing_v2
        --------------------------
        Build an I(2) system (cumsum of cumsum of noise) so that a single
        difference is insufficient for stationarity; the engine must apply d=2.

        Assertions
        ----------
        1. diff_order == 2  (two differences were required)
        2. All three forecast Series are finite  (no NaN/Inf after d=2 re-integration)
        3. Forecast index is strictly after the last historical timestamp.
        """
        rng = np.random.default_rng(7)
        n   = 150
        idx = pd.date_range("2023-01-01", periods=n, freq="5min")

        # I(2) process: cumsum of cumsum of noise (mean-corrected to avoid sign flip)
        noise  = rng.normal(0, 0.3, n)
        level1 = np.cumsum(noise)                    # I(1)
        level2 = np.cumsum(level1 - level1.mean())   # I(2), zero-centred drift

        # Scale to plausible CPU range [10, 90]
        def _scale(x):
            lo, hi = x.min(), x.max()
            return 10.0 + 80.0 * (x - lo) / (hi - lo + 1e-9)

        avg = _scale(level2 + rng.normal(0, 0.1, n))
        mx  = np.clip(avg + np.abs(rng.normal(2, 0.3, n)), 0, 100)
        mn  = np.clip(avg - np.abs(rng.normal(2, 0.3, n)), 0, 100)

        df_i2 = pd.DataFrame({'timestamp_dt': idx, 'avg_cpu': avg,
                               'max_cpu': mx, 'min_cpu': mn})

        result = run_var_forecast(df_i2, forecast_steps=8)

        # 1. Differencing order should be 2
        assert result['diff_order'] == 2, (
            f"Expected diff_order=2 for I(2) data, got {result['diff_order']}"
        )

        # 2. All forecasts finite (no NaN/Inf after d=2 re-integration)
        for col in VAR_COLS:
            fc = result['forecasts'][col].values
            assert np.all(np.isfinite(fc)), (
                f"{col} forecast contains NaN/Inf after d=2 re-integration"
            )

        # 3. Forecast index strictly after last historical timestamp
        last_hist_ts = idx[-1]
        for col in VAR_COLS:
            assert result['forecasts'][col].index[0] > last_hist_ts, (
                f"{col} forecast starts before or at last historical timestamp"
            )


# ---------------------------------------------------------------------------
# _BoxCoxLayer tests
# ---------------------------------------------------------------------------

class TestBoxCoxLayer:

    @pytest.fixture(scope="class")
    def cpu_series(self):
        """Plausible CPU % series with a few near-zero values."""
        np.random.seed(99)
        n   = 120
        idx = pd.date_range("2023-01-01", periods=n, freq="5min")
        vals = np.clip(30 + 8 * np.sin(2 * np.pi * np.arange(n) / 12)
                       + np.random.normal(0, 1, n), 0.0, 100.0)
        # Insert a couple of genuine-zero values
        vals[[5, 42]] = 0.0
        return pd.Series(vals, index=idx, name="avg_cpu")

    def test_lambda_is_finite_float(self, cpu_series):
        """Estimated lambda must be a finite float."""
        layer = _BoxCoxLayer(cpu_series)
        assert isinstance(layer.lam, float)
        assert np.isfinite(layer.lam)

    def test_transform_returns_same_index(self, cpu_series):
        """transform() output index must match input index."""
        layer = _BoxCoxLayer(cpu_series)
        t     = layer.transform(cpu_series)
        assert t.index.equals(cpu_series.index)

    def test_round_trip_within_tolerance(self, cpu_series):
        """
        transform → inverse must recover the original series within ±0.01 %
        (numerical tolerance of scipy's inv_boxcox).
        """
        layer   = _BoxCoxLayer(cpu_series)
        # Use clipped series as the 'true' input (zeros become floor)
        clipped = cpu_series.clip(lower=layer.floor)
        t       = layer.transform(clipped)
        recovered = layer.inverse(t)
        np.testing.assert_allclose(
            recovered.values, clipped.values,
            rtol=1e-4, atol=0.01,
            err_msg="Box-Cox round-trip failed: max error exceeds 0.01 %",
        )

    def test_zero_values_handled(self, cpu_series):
        """transform() must NOT raise when zeros are present in the input."""
        layer = _BoxCoxLayer(cpu_series)
        t = layer.transform(cpu_series)   # should not raise
        assert np.all(np.isfinite(t.values)), "transform produced non-finite values on zero-containing input"

    def test_inverse_clips_to_cpu_range(self, cpu_series):
        """inverse() must clip output to [0, 100]."""
        layer  = _BoxCoxLayer(cpu_series)
        # Feed extreme transformed values — inverse must not exceed [0, 100]
        extreme = pd.Series([1e6, -1e6, 0.0], index=[0, 1, 2])
        result  = layer.inverse(extreme)
        assert result.min() >= 0.0
        assert result.max() <= 100.0
