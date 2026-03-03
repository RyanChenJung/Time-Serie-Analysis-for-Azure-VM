"""
utils/forecasting.py
────────────────────────────────────────────────────────────────────────────
Phase 3.2 — Advanced Statistical Forecasting Engine

Pure-functional backend (no Streamlit imports).  All functions accept a
pd.Series with a pd.DatetimeIndex and return plain Python dicts / DataFrames.
"""
import warnings
import sys
import io
import itertools
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _infer_freq_td(series: pd.Series) -> pd.Timedelta:
    """Infer the most common step size in a DatetimeIndex series."""
    if not isinstance(series.index, pd.DatetimeIndex) or len(series) < 2:
        return pd.Timedelta(minutes=5)  # safe fallback
    diffs = series.index.to_series().diff().dropna()
    return diffs.mode().iloc[0] if not diffs.empty else pd.Timedelta(minutes=5)


class _StdoutCapture:
    """
    Context manager: temporarily redirects sys.stdout to an in-memory buffer
    so that third-party trace output (e.g. pmdarima auto_arima trace=True)
    can be captured and returned rather than printed to the terminal.

    Usage::

        with _StdoutCapture() as buf:
            some_library_function_that_prints()
        captured_text = buf.getvalue()
    """
    def __enter__(self) -> io.StringIO:
        self._buf  = io.StringIO()
        self._orig = sys.stdout
        sys.stdout = self._buf
        return self._buf

    def __exit__(self, *_):
        sys.stdout = self._orig


class _BoxCoxLayer:
    """
    Optional Box-Cox variance-stabilisation preprocessing layer.

    Estimates lambda by MLE (``scipy.stats.boxcox``) from the provided series,
    applies the transform before model fitting, and inverse-transforms
    forecasts / CI bounds back to the original scale.

    Zero handling
    -------------
    Values are clipped to ``floor`` (default 0.01 %) before transforming.
    This prevents log/power-domain blow-up on zero CPU readings.  The inverse
    transform clips the result to [0, 100] for valid CPU-percentage range.

    Lambda ≈ 0  → log transform (natural stabilisation for skewed data).
    Lambda ≈ 1  → near-identity; transform has negligible practical effect.
    """

    def __init__(self, series: pd.Series, floor: float = 0.01):
        from scipy.stats import boxcox as _fit_boxcox
        self.floor = float(floor)
        s_pos = series.clip(lower=self.floor).dropna().values
        _, self.lam = _fit_boxcox(s_pos)
        self.lam = float(self.lam)

    def transform(self, series: pd.Series) -> pd.Series:
        """Apply Box-Cox; returns a Series with the same index."""
        s_pos = series.clip(lower=self.floor).values
        if abs(self.lam) < 1e-8:
            transformed = np.log(s_pos)
        else:
            transformed = (s_pos ** self.lam - 1.0) / self.lam
        return pd.Series(transformed, index=series.index)

    def inverse(self, series: pd.Series) -> pd.Series:
        """Inverse Box-Cox; clips result to [0, 100] for CPU %."""
        from scipy.special import inv_boxcox as _inv
        vals = _inv(series.values, self.lam)
        return pd.Series(np.clip(vals, 0.0, 100.0), index=series.index)

    def inverse_ci(
        self,
        lower: pd.Series,
        upper: pd.Series,
    ) -> "tuple[pd.Series, pd.Series]":
        """Inverse-transform a CI pair; preserves ordering."""
        return self.inverse(lower), self.inverse(upper)


def _future_index(series: pd.Series, steps: int) -> pd.DatetimeIndex:
    """Build a DatetimeIndex for `steps` future periods after the series end."""
    freq_td = _infer_freq_td(series)
    last_ts = series.index[-1]
    return pd.date_range(start=last_ts + freq_td, periods=steps, freq=freq_td)


def _validate_series(series: pd.Series, min_len: int = 15) -> pd.Series:
    """Drop NaNs, ensure minimum length, raise clearly on violations."""
    series = series.dropna()
    if len(series) < min_len:
        raise ValueError(
            f"Series too short for SARIMA fitting: {len(series)} observations "
            f"(minimum required: {min_len})."
        )
    return series


def build_exog_df(vm_df: "pd.DataFrame") -> pd.DataFrame:
    """Build a causally-lagged exogenous DataFrame for ARIMAX.

    Applies a 1-step lag to `max_cpu` and `min_cpu` so that at time t the
    model sees Max(t-1) and Min(t-1) only, preserving strict causality:

        Target(t) ~ SARIMA(p,d,q) + β₁·Max(t-1) + β₂·Min(t-1)

    The first row (which becomes NaN after shifting) is dropped.  The caller
    must consequently also drop the first row of the `avg_cpu` series so that
    both share the same DatetimeIndex.

    Parameters
    ----------
    vm_df : DataFrame with columns ['timestamp_dt', 'max_cpu', 'min_cpu'].

    Returns
    -------
    pd.DataFrame  (DatetimeIndex, columns=['max_cpu', 'min_cpu'])
    with len == len(vm_df) - 1.
    """
    df = vm_df.set_index('timestamp_dt')[['max_cpu', 'min_cpu']].copy()
    df = df.shift(1).iloc[1:]   # lag by 1, drop NaN row
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_sarima_forecast(
    series: pd.Series,
    order: tuple,
    seasonal_order: tuple,
    forecast_steps: int,
    exog_df: "pd.DataFrame | None" = None,
    use_boxcox: bool = False,
) -> dict:
    """
    Dual-execution SARIMA / ARIMAX: validation fit (80 / 20) + final fit (100%).

    Parameters
    ----------
    series         : DatetimeIndex pd.Series of avg_cpu values.
    order          : ARIMA (p, d, q) tuple.
    seasonal_order : Seasonal (P, D, Q, s) tuple.
    forecast_steps : Number of future steps to forecast.
    exog_df        : Optional DataFrame with lagged exogenous features
                     (max_cpu, min_cpu, same index as series).
                     When provided the model becomes ARIMAX / SARIMAX.

    Returns
    -------
    dict with keys:
        'mean'            – pd.Series  (future forecast, DatetimeIndex)
        'upper'           – pd.Series  (95 % CI upper bound)
        'lower'           – pd.Series  (95 % CI lower bound)
        'aic'             – float      (AIC of the final model)
        'val_metrics'     – dict       {RMSE, MAE, MAPE}  (from hold-out)
        'val_predictions' – pd.Series  (model predictions on the test window)
        'model_name'      – str        ('ARIMAX' if exog_df provided else 'SARIMA')
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from sklearn.metrics import mean_squared_error, mean_absolute_error

    series = _validate_series(series)
    model_name = 'ARIMAX' if exog_df is not None else 'SARIMA'

    # ── Optional Box-Cox pre-processing ─────────────────────────────────────
    _bc_layer: '_BoxCoxLayer | None' = None
    _series_orig = series.copy()   # original levels — always kept for metric fix-up
    if use_boxcox:
        _bc_layer = _BoxCoxLayer(series)
        series    = _bc_layer.transform(series)

    # ── Align exog to series index (handles cases where timestamps differ) ──────
    if exog_df is not None:
        exog_df      = exog_df.reindex(series.index).dropna()
        series       = series.loc[exog_df.index]
        _series_orig = _series_orig.loc[exog_df.index]

    p, d, q = order
    P, D, Q, s = seasonal_order

    # ── Step 1: Validation fit (80 / 20 hold-out) ───────────────────────────
    split = int(len(series) * 0.8)
    train_val = series.iloc[:split]
    test_val  = series.iloc[split:]

    exog_train = exog_df.iloc[:split]  if exog_df is not None else None
    exog_test  = exog_df.iloc[split:]  if exog_df is not None else None

    val_predictions = pd.Series(dtype=float)
    val_metrics: dict = {'RMSE': np.nan, 'MAE': np.nan, 'MAPE': np.nan}

    if not test_val.empty:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                mod_val = SARIMAX(
                    train_val,
                    exog=exog_train,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                res_val = mod_val.fit(disp=False)
                fc_val  = res_val.get_forecast(steps=len(test_val), exog=exog_test)
                val_pred_values = fc_val.predicted_mean.values

                val_predictions = pd.Series(
                    val_pred_values[:len(test_val)],
                    index=test_val.index,
                )

                y_true = test_val.values
                y_pred = val_predictions.values

                rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
                mae  = float(mean_absolute_error(y_true, y_pred))

                mask = y_true != 0
                mape = (float(np.mean(
                    np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])
                )) * 100) if mask.any() else 0.0

                val_metrics = {'RMSE': rmse, 'MAE': mae, 'MAPE': mape}

            except Exception:
                pass

    # ── Step 2: Final fit (100 % of data) for future forecast ─────────────────
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mod_final = SARIMAX(
            series,
            exog=exog_df,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        res_final = mod_final.fit(disp=False)

    # ── Step 3: Future exog — repeat the last known observation for all steps ──
    # At time T we know Max(T) and Min(T); these become the t-1 values for T+1.
    # For T+2 … T+h we have no future max/min so we repeat Max(T), Min(T).
    future_exog = None
    if exog_df is not None:
        last_row = exog_df.iloc[-1]
        future_exog = pd.DataFrame(
            np.tile(last_row.values, (forecast_steps, 1)),
            columns=exog_df.columns,
        )

    fc = res_final.get_forecast(steps=forecast_steps, exog=future_exog)
    fc_mean = fc.predicted_mean
    fc_ci   = fc.conf_int(alpha=0.05)

    future_idx = _future_index(series, forecast_steps)
    fc_mean.index = future_idx
    fc_ci.index   = future_idx

    # ── Box-Cox inverse (if enabled) ────────────────────────────────────────
    boxcox_lambda = None
    _ci_lo = fc_ci.iloc[:, 0].copy()
    _ci_hi = fc_ci.iloc[:, 1].copy()
    if _bc_layer is not None:
        boxcox_lambda   = _bc_layer.lam
        fc_mean         = _bc_layer.inverse(fc_mean)
        _ci_lo, _ci_hi  = _bc_layer.inverse_ci(_ci_lo, _ci_hi)
        if not val_predictions.empty:
            val_predictions = _bc_layer.inverse(val_predictions)
            from sklearn.metrics import mean_squared_error as _mse, mean_absolute_error as _mae
            _yt  = _series_orig.loc[val_predictions.index].values
            _yp  = val_predictions.values
            if np.all(np.isfinite(_yp)):
                _rm  = float(np.sqrt(_mse(_yt, _yp)))
                _ma  = float(_mae(_yt, _yp))
                _mk  = _yt != 0
                _mp  = (float(np.mean(np.abs((_yt[_mk] - _yp[_mk]) / _yt[_mk]))) * 100) \
                       if _mk.any() else 0.0
                val_metrics = {'RMSE': _rm, 'MAE': _ma, 'MAPE': _mp}

    return {
        'mean':            fc_mean,
        'upper':           _ci_hi,
        'lower':           _ci_lo,
        'aic':             float(res_final.aic),
        'val_metrics':     val_metrics,
        'val_predictions': val_predictions,
        'model_name':      model_name,
        'boxcox_lambda':   boxcox_lambda,
    }


# ---------------------------------------------------------------------------
# Holt-Winters (Exponential Smoothing)
# ---------------------------------------------------------------------------

def run_holt_winters_forecast(
    series: pd.Series,
    trend: str | None,
    seasonal: str | None,
    seasonal_periods: int,
    damped: bool,
    forecast_steps: int,
    use_boxcox: bool = False,
) -> dict:
    """
    Dual-execution Holt-Winters: validation fit (80 / 20) + final fit (100%).

    Parameters
    ----------
    series           : DatetimeIndex pd.Series of avg_cpu values.
    trend            : 'add', 'mul', or None.
    seasonal         : 'add', 'mul', or None.
    seasonal_periods : Seasonal period length (pass 1 or 0 for aperiodic VMs
                       — seasonal is then forced to None automatically).
    damped           : If True, applies a damped trend (only when trend != None).
    forecast_steps   : Number of future steps to forecast.

    Returns
    -------
    dict with keys:
        'mean'            – pd.Series  (future forecast, DatetimeIndex)
        'upper'           – pd.Series  (95 % prediction interval upper bound)
        'lower'           – pd.Series  (95 % prediction interval lower bound)
        'aic'             – float      (AIC of the final model)
        'val_metrics'     – dict       {RMSE, MAE, MAPE}  (from hold-out)
        'val_predictions' – pd.Series  (predictions on the test window)
        'val_residuals'   – pd.Series  (actuals − predictions, test window)
        'model_label'     – str        (human-readable config string)
    """
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from sklearn.metrics import mean_squared_error, mean_absolute_error

    series = _validate_series(series)

    # ── Optional Box-Cox pre-processing ─────────────────────────────────────
    _bc_layer_hw: '_BoxCoxLayer | None' = None
    _series_orig_hw = series.copy()
    if use_boxcox:
        _bc_layer_hw = _BoxCoxLayer(series)
        series       = _bc_layer_hw.transform(series)

    # Aperiodic guard: if no seasonal period, disable seasonal component
    if seasonal_periods <= 1:
        seasonal = None

    # damped_trend is only valid when trend is set
    damped_trend = damped if trend is not None else False

    def _make_model(s: pd.Series) -> ExponentialSmoothing:
        return ExponentialSmoothing(
            s,
            trend=trend,
            seasonal=seasonal,
            seasonal_periods=seasonal_periods if seasonal is not None else None,
            damped_trend=damped_trend,
            initialization_method='estimated',
        )

    # ── Step 1: Validation fit (80 / 20 hold-out) ───────────────────────────
    split = int(len(series) * 0.8)
    train_val = series.iloc[:split]
    test_val  = series.iloc[split:]

    val_predictions = pd.Series(dtype=float)
    val_residuals   = pd.Series(dtype=float)
    val_metrics: dict = {'RMSE': np.nan, 'MAE': np.nan, 'MAPE': np.nan}

    if not test_val.empty:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            try:
                res_val = _make_model(train_val).fit(optimized=True, remove_bias=False)
                pred_vals = res_val.forecast(len(test_val))
                val_predictions = pd.Series(pred_vals.values, index=test_val.index)
                val_residuals   = test_val - val_predictions

                y_true, y_pred = test_val.values, val_predictions.values
                rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
                mae  = float(mean_absolute_error(y_true, y_pred))
                mask = y_true != 0
                mape = (float(np.mean(
                    np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])
                )) * 100) if mask.any() else 0.0
                val_metrics = {'RMSE': rmse, 'MAE': mae, 'MAPE': mape}
            except Exception:
                pass

    # ── Step 2: Final fit (100 % of data) for future forecast ───────────────
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res_final = _make_model(series).fit(optimized=True, remove_bias=False)

    # ── Step 3: Future forecast + prediction intervals ───────────────────────
    future_idx = _future_index(series, forecast_steps)
    fc_mean_vals = res_final.forecast(forecast_steps)
    fc_mean = pd.Series(fc_mean_vals.values, index=future_idx)

    # Simulation-based 95 % prediction interval (500 paths)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            sim = res_final.simulate(
                nsimulations=forecast_steps,
                repetitions=500,
                error='add',
            )
        fc_upper = pd.Series(np.quantile(sim, 0.975, axis=1), index=future_idx)
        fc_lower = pd.Series(np.quantile(sim, 0.025, axis=1), index=future_idx)
    except Exception:
        # Fallback: ±1.96 × training residual std
        resid_std = float(np.std(res_final.resid.dropna()))
        fc_upper = fc_mean + 1.96 * resid_std
        fc_lower = fc_mean - 1.96 * resid_std

    # Model label for UI display
    t_str = {'add': 'Additive', 'mul': 'Multiplicative'}.get(trend or '', 'None')
    s_str = {'add': 'Additive', 'mul': 'Multiplicative'}.get(seasonal or '', 'None')
    d_str = ' (Damped)' if damped_trend else ''
    label = f'HW · T={t_str}{d_str} · S={s_str} · m={seasonal_periods}'

    # ── Box-Cox inverse (if enabled) ────────────────────────────────────────
    boxcox_lambda_hw = None
    if _bc_layer_hw is not None:
        boxcox_lambda_hw = _bc_layer_hw.lam
        fc_mean          = _bc_layer_hw.inverse(fc_mean)
        fc_lower, fc_upper = _bc_layer_hw.inverse_ci(fc_lower, fc_upper)
        if not val_predictions.empty:
            val_predictions = _bc_layer_hw.inverse(val_predictions)
            _test_orig = _series_orig_hw.iloc[split:]
            val_residuals = _test_orig - val_predictions
            from sklearn.metrics import mean_squared_error as _mse, mean_absolute_error as _mae
            _yt = _test_orig.values
            _yp = val_predictions.values[:len(_yt)]
            if np.all(np.isfinite(_yp)):
                _rm = float(np.sqrt(_mse(_yt, _yp)))
                _ma = float(_mae(_yt, _yp))
                _mk = _yt != 0
                _mp = (float(np.mean(np.abs((_yt[_mk] - _yp[_mk]) / _yt[_mk]))) * 100) \
                      if _mk.any() else 0.0
                val_metrics = {'RMSE': _rm, 'MAE': _ma, 'MAPE': _mp}

    return {
        'mean':            fc_mean,
        'upper':           fc_upper,
        'lower':           fc_lower,
        'aic':             float(res_final.aic),
        'val_metrics':     val_metrics,
        'val_predictions': val_predictions,
        'val_residuals':   val_residuals,
        'model_label':     label,
        'boxcox_lambda':   boxcox_lambda_hw,
    }


def auto_optimize_holt_winters(
    series: pd.Series,
    seasonal_periods: int,
    progress_callback=None,
) -> dict:
    """
    Grid search over Holt-Winters configurations; returns the combo with the
    lowest validation RMSE on an 80 / 20 hold-out split.

    Parameters
    ----------
    series            : DatetimeIndex pd.Series of avg_cpu values.
    seasonal_periods  : Detected or manually-set lag count.  If <= 1, the
                        seasonal dimension is forced to [None] automatically.
    progress_callback : Optional callable(str) — called with a one-line
                        progress string after each combo is evaluated.
                        Allows the Streamlit UI to stream results via
                        st.status / st.write without any coupling to
                        Streamlit inside this pure-backend function.

    Returns
    -------
    dict:
        'best_config'  – dict {trend, seasonal, damped}  (statsmodels values)
        'best_rmse'    – float  (validation RMSE of winning config)
        'best_label'   – str    (human-readable label for the winner)
        'trace'        – list[dict]  (one entry per evaluated combo)
    Raises ValueError if the series is too short.
    """
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    series = _validate_series(series)

    # ── Search grid ────────────────────────────────────────────────────────────────
    trend_opts    = ['add', 'mul', None]
    seasonal_opts = ['add', 'mul', None] if seasonal_periods > 1 else [None]
    damped_opts   = [True, False]

    # Filter invalid combos before entering the loop:
    #   damped_trend=True is only valid when trend is not None
    combos = [
        (t, s, d)
        for t, s, d in itertools.product(trend_opts, seasonal_opts, damped_opts)
        if not (d is True and t is None)
    ]

    # ── 80 / 20 split (done once; same slice for all combos) ──────────────────
    split     = int(len(series) * 0.8)
    train_val = series.iloc[:split]
    test_val  = series.iloc[split:]

    # ── Human-readable label helper ───────────────────────────────────────────
    _lbl = {'add': 'Additive', 'mul': 'Multiplicative', None: 'None'}

    def _combo_label(t, s, d):
        d_str = ' (Damped)' if (d and t is not None) else ''
        return f"T={_lbl[t]}{d_str} · S={_lbl[s]} · m={seasonal_periods}"

    # ── Grid search ────────────────────────────────────────────────────────────────
    best_rmse   = float('inf')
    best_config = None
    best_label  = ''
    trace       = []

    for trend, seasonal, damped in combos:
        lbl          = _combo_label(trend, seasonal, damped)
        damped_trend = damped if trend is not None else False
        entry        = {'label': lbl, 'trend': trend, 'seasonal': seasonal,
                        'damped': damped, 'rmse': None, 'error': None}
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                mod = ExponentialSmoothing(
                    train_val,
                    trend=trend,
                    seasonal=seasonal,
                    seasonal_periods=seasonal_periods if seasonal is not None else None,
                    damped_trend=damped_trend,
                    initialization_method='estimated',
                )
                fit   = mod.fit(optimized=True, remove_bias=False)
                preds = fit.forecast(len(test_val))

            y_true = test_val.values
            y_pred = preds.values[:len(y_true)]
            rmse   = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

            entry['rmse'] = rmse
            if rmse < best_rmse:
                best_rmse   = rmse
                best_config = {'trend': trend, 'seasonal': seasonal, 'damped': damped}
                best_label  = lbl

            if progress_callback:
                progress_callback(f"✓ {lbl:<55}  RMSE = {rmse:.4f}")

        except Exception as exc:
            entry['error'] = str(exc)
            if progress_callback:
                progress_callback(f"⃟ {lbl:<55}  SKIPPED ({exc!s:.60})")

        trace.append(entry)

    if best_config is None:
        raise RuntimeError(
            "All Holt-Winters configurations failed. "
            "Check that the series has no non-positive values for multiplicative components."
        )

    return {
        'best_config': best_config,
        'best_rmse':   best_rmse,
        'best_label':  best_label,
        'trace':       trace,
    }


def auto_optimize_sarima(
    series: pd.Series,
    seasonal_period: int,
    exog_df: "pd.DataFrame | None" = None,
) -> dict:
    """
    Use pmdarima.auto_arima to find the best (p,d,q)(P,D,Q,s) orders.

    When exog_df is provided, the search is run as ARIMAX (pmdarima `X=` arg).

    ``seasonal_period`` is treated as a **hard constraint** for ``m`` (the
    seasonal cycle length) — p, d, q, P, D, Q are searched freely.
    The internal search log (one line per model tried) is captured from
    stdout and returned in the ``log`` key so callers can display it.

    Parameters
    ----------
    series          : DatetimeIndex pd.Series of avg_cpu values.
    seasonal_period : Detected or manually-set seasonality lag count.
    exog_df         : Optional lagged exogenous DataFrame (same index as series).

    Returns
    -------
    dict:
        'order'          – (p, d, q)
        'seasonal_order' – (P, D, Q, s)  where s == seasonal_period
        'log'            – str  (all trace lines joined by newlines)
    Raises ValueError if the series is too short.
    """
    import pmdarima as pm

    series = _validate_series(series)
    use_seasonal = seasonal_period > 1

    # Align and prepare exog array for pmdarima (expects numpy, not DataFrame)
    X = None
    if exog_df is not None:
        exog_aligned = exog_df.reindex(series.index).dropna()
        series       = series.loc[exog_aligned.index]
        X            = exog_aligned.values

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Capture pmdarima's trace=True stdout output
        with _StdoutCapture() as buf:
            model = pm.auto_arima(
                series.values,
                X=X,
                seasonal=use_seasonal,
                m=seasonal_period if use_seasonal else 1,   # hard constraint
                stepwise=True,
                approximation=False,    # full MLE — valid AIC comparisons across all orders
                information_criterion='aic',
                error_action='ignore',
                suppress_warnings=True,
                max_p=4,
                max_q=4,
                max_P=2,
                max_Q=2,
                max_d=2,
                max_D=1,
                trace=True,    # write iteration lines to stdout
            )
        log_text = buf.getvalue().strip()

    p, d, q = model.order
    seasonal = model.seasonal_order          # (P, D, Q, s)
    P, D, Q, s = seasonal if len(seasonal) == 4 else (0, 0, 0, 0)

    # Enforce s == seasonal_period (auto_arima may output s=0 for non-seasonal)
    s = seasonal_period if use_seasonal else 0

    return {
        'order':          (p, d, q),
        'seasonal_order': (P, D, Q, s),
        'log':            log_text or "(no trace output captured)",
    }


def stream_auto_arima(
    series: pd.Series,
    seasonal_period: int,
    log_queue: "queue.Queue",
) -> dict:
    """
    Thread-safe variant of auto_optimize_sarima for real-time UI streaming.

    Designed to run in a daemon ``threading.Thread``.  Each trace line emitted
    by ``pmdarima.auto_arima(trace=True)`` is put onto ``log_queue`` as it
    arrives, so the calling (Streamlit UI) thread can poll and display lines
    one-by-one.  A ``None`` sentinel is always placed on the queue last to
    signal completion (or failure).

    Parameters
    ----------
    series          : DatetimeIndex pd.Series of avg_cpu values.
    seasonal_period : Hard-constrained seasonal cycle length.
    log_queue       : ``queue.Queue`` instance shared with the UI thread.

    Returns  (in the holder dict the caller must pass)
    -------
    Caller should pass a mutable dict as ``result_holder``; this function is
    called via ``threading.Thread(target=stream_auto_arima, args=(...))`` so
    return values must be passed via the shared queue or a mutable container.
    See usage pattern in ``app.py``.

    The function puts each trace line into ``log_queue`` as a string, and
    puts ``None`` as a final sentinel when done (or on error).  The caller
    must read the result from its own ``result_holder`` dict.
    """
    import pmdarima as pm

    class _QueueWriter:
        """sys.stdout replacement: puts non-empty lines onto a Queue."""
        def __init__(self, q: "queue.Queue"):
            self._q = q

        def write(self, text: str) -> None:
            line = text.rstrip()
            if line:
                self._q.put(line)

        def flush(self) -> None:
            pass  # required by sys.stdout interface

    series = _validate_series(series)
    use_seasonal = seasonal_period > 1

    orig_stdout = sys.stdout
    sys.stdout = _QueueWriter(log_queue)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = pm.auto_arima(
                series.values,
                seasonal=use_seasonal,
                m=seasonal_period if use_seasonal else 1,
                stepwise=True,
                approximation=False,    # full MLE — valid AIC comparisons across all orders
                information_criterion='aic',
                error_action='ignore',
                suppress_warnings=True,
                max_p=4,
                max_q=4,
                max_P=2,
                max_Q=2,
                max_d=2,
                max_D=1,
                trace=True,
            )
    finally:
        # Always restore stdout before putting the sentinel
        sys.stdout = orig_stdout
        log_queue.put(None)   # sentinel — signals the UI thread to stop polling

    p, d, q = model.order
    seasonal = model.seasonal_order
    P, D, Q, s = seasonal if len(seasonal) == 4 else (0, 0, 0, 0)
    s = seasonal_period if use_seasonal else 0

    return {
        'order':          (p, d, q),
        'seasonal_order': (P, D, Q, s),
    }


# ---------------------------------------------------------------------------
# Vector Autoregression (VAR) — Multivariate System (Phase 3.3b)
# ---------------------------------------------------------------------------

def run_var_forecast(
    vm_df: pd.DataFrame,
    forecast_steps: int,
    maxlags: int = 15,
    alpha: float = 0.05,
    use_boxcox: bool = False,
) -> dict:
    """
    Multivariate VAR forecast for the system [avg_cpu, max_cpu, min_cpu].

    Auto-Stationarity Pipeline (iterative)
    ----------------------------------------
    1. Run ADF on all three series at the current differencing order d.
    2. If all pass (p < alpha) → fit VAR at this d.
    3. Otherwise increment d and repeat up to d_max = 2.
    4. Before each differencing step the "tail seeds" (last d rows of the
       original and first-differenced series) are stored.  These seeds are
       used for the nested cumsum re-integration described below.

    Re-integration (inverting d differences)
    -----------------------------------------
    The VAR forecasts are produced in the Δ^d domain.  Recovery to levels:

    d = 1:
        Δ̂y_{T+h} = VAR forecast
        ŷ_{T+h} = y_T + cumsum(Δ̂y)
        Seed: y_T  (last row of original levels)

    d = 2:
        Δ²̂y_{T+h} = VAR forecast
        Step 1 — recover Δy:
            Δ̂y_{T+h} = Δy_T + cumsum(Δ²̂y)
            Seed: Δy_T = y_T − y_{T-1}  (last row of first-diff series)
        Step 2 — recover y:
            ŷ_{T+h} = y_T + cumsum(Δ̂y)
            Seed: y_T  (last row of original levels)

    Both steps are identical in code — they are cumsum() + anchor.

    Validation
    ----------
    80 / 20 hold-out.  RMSE / MAE / MAPE always compared against **original
    level** data, regardless of d, using the same re-integration logic.

    Granger Causality
    -----------------
    Run on the stationary (fully-differenced) data for statistical validity.

    NaN / Inf guard
    ---------------
    If any forecast value is non-finite after re-integration, each affected
    column falls back to a simple drift forecast (mean change × h).

    Parameters
    ----------
    vm_df          : DataFrame with columns ['timestamp_dt', 'avg_cpu',
                     'max_cpu', 'min_cpu'].
    forecast_steps : Number of future steps to produce.
    maxlags        : Maximum lag order for select_order().
    alpha          : ADF significance level (default 0.05).

    Returns
    -------
    dict with keys:
        forecasts       – {col: pd.Series (DatetimeIndex)}
        val_metrics     – {col: {RMSE, MAE, MAPE}}
        val_predictions – {col: pd.Series (hold-out predictions, level domain)}
        lag_order       – int
        diff_order      – int  (0, 1, or 2)
        adf_pvalues     – {col: float}  (p-values at chosen d)
        granger         – {"cause→effect": {p_value, significant, max_lag}}
        var_residuals   – pd.DataFrame (fitted residuals in transformed domain)
        aic             – float

    Raises
    ------
    ValueError   if vm_df has < 30 rows or missing columns.
    RuntimeError if the final VAR fit fails at every configuration.
    """
    from statsmodels.tsa.api import VAR
    from statsmodels.tsa.stattools import adfuller
    from sklearn.metrics import mean_squared_error, mean_absolute_error

    # ── 0. Input validation ───────────────────────────────────────────────────
    required_cols = {'timestamp_dt', 'avg_cpu', 'max_cpu', 'min_cpu'}
    missing = required_cols - set(vm_df.columns)
    if missing:
        raise ValueError(f"vm_df is missing columns: {missing}")

    VAR_COLS = ['avg_cpu', 'max_cpu', 'min_cpu']
    df_orig  = vm_df.set_index('timestamp_dt')[VAR_COLS].copy().dropna()

    if len(df_orig) < 30:
        raise ValueError(
            f"VAR requires at least 30 observations; got {len(df_orig)}."
        )

    # ── Optional Box-Cox pre-processing (per column, individual lambdas) ─────
    # df_levels = truly original CPU %, used for metric comparison & drift fallback.
    # df_orig   = working copy (Box-Cox transformed if use_boxcox, else identical).
    df_levels = df_orig.copy()   # always original CPU scale
    var_bc_layers: "dict[str, _BoxCoxLayer]" = {}
    if use_boxcox:
        for _col in VAR_COLS:
            var_bc_layers[_col] = _BoxCoxLayer(df_orig[_col])
        for _col in VAR_COLS:
            df_orig[_col] = var_bc_layers[_col].transform(df_orig[_col])

    # ── 1. Iterative ADF loop — find minimum d ∈ {0, 1, 2} ──────────────────
    # At each iteration we test the *currently transformed* df_work.
    # We store the tail seeds needed for re-integration before each diff step.

    D_MAX = 2
    df_work      = df_orig.copy()
    diff_order   = 0
    adf_pvalues: dict[str, float] = {}

    # Seeds: seed_level[d_i] = last row of df after applying d_i differences
    # We need seeds[0] = last of levels, seeds[1] = last of first-diff if d=2.
    seed_level:  pd.Series | None = None  # y_T          (always needed if d≥1)
    seed_delta1: pd.Series | None = None  # Δy_T          (needed only if d=2)

    for _d in range(D_MAX + 1):
        # ADF test on current df_work
        _pvals: dict[str, float] = {}
        for col in VAR_COLS:
            try:
                _pvals[col] = float(adfuller(df_work[col].dropna().values)[1])
            except Exception:
                _pvals[col] = 1.0

        if all(p < alpha for p in _pvals.values()):
            # All stationary at this d — use it
            diff_order  = _d
            adf_pvalues = _pvals
            break

        if _d < D_MAX:
            # Store seed for re-integration before differencing
            if _d == 0:
                seed_level  = df_work.iloc[-1].copy()   # y_T
            elif _d == 1:
                seed_delta1 = df_work.iloc[-1].copy()   # Δy_T  (last of Δy)

            # Difference and advance
            df_work   = df_work.diff().dropna()
            diff_order = _d + 1
        else:
            # d=2 failed ADF too — still proceed at d=2, record p-values
            adf_pvalues = _pvals

    # If we differentiated at least once, ensure seeds are set
    if diff_order >= 1 and seed_level is None:
        seed_level = df_orig.iloc[-1].copy()
    if diff_order == 2 and seed_delta1 is None:
        seed_delta1 = df_orig.diff().dropna().iloc[-1].copy()

    # ── Helper: re-integrate Δ^d forecasts back to levels ────────────────────
    def _recover_levels(
        fc_diff_d: pd.DataFrame,
        d: int,
    ) -> pd.DataFrame:
        """
        Convert a DataFrame of Δ^d forecasts to level forecasts.

        Parameters
        ----------
        fc_diff_d : DataFrame in the Δ^d domain (rows = future steps,
                    columns = VAR_COLS).
        d         : differencing order actually applied (0, 1, or 2).

        Returns
        -------
        DataFrame in the original level domain.
        """
        if d == 0:
            return fc_diff_d.copy()

        recovered = fc_diff_d.copy()

        if d == 1:
            # ŷ = y_T + cumsum(Δ̂y)
            for col in VAR_COLS:
                anchor = float(seed_level[col])  # type: ignore[index]
                recovered[col] = float(anchor) + fc_diff_d[col].cumsum().values

        elif d == 2:
            # Step 1: recover Δy from Δ²y  →  Δ̂y = Δy_T + cumsum(Δ²̂y)
            delta1_fc = fc_diff_d.copy()
            for col in VAR_COLS:
                anchor_d1 = float(seed_delta1[col])  # type: ignore[index]
                delta1_fc[col] = anchor_d1 + fc_diff_d[col].cumsum().values

            # Step 2: recover y from Δy  →  ŷ = y_T + cumsum(Δ̂y)
            for col in VAR_COLS:
                anchor_d0 = float(seed_level[col])  # type: ignore[index]
                recovered[col] = anchor_d0 + delta1_fc[col].cumsum().values

        return recovered

    def _recover_levels_val(
        fc_diff_d: pd.DataFrame,
        d: int,
        val_seed_level: pd.Series,
        val_seed_delta1: pd.Series | None,
    ) -> pd.DataFrame:
        """Same as _recover_levels but uses validation-window seeds."""
        if d == 0:
            return fc_diff_d.copy()

        recovered = fc_diff_d.copy()

        if d == 1:
            for col in VAR_COLS:
                recovered[col] = (float(val_seed_level[col])
                                  + fc_diff_d[col].cumsum().values)
        elif d == 2:
            delta1_fc = fc_diff_d.copy()
            for col in VAR_COLS:
                anchor_d1 = float(val_seed_delta1[col])  # type: ignore[index]
                delta1_fc[col] = anchor_d1 + fc_diff_d[col].cumsum().values
            for col in VAR_COLS:
                anchor_d0 = float(val_seed_level[col])
                recovered[col] = anchor_d0 + delta1_fc[col].cumsum().values

        return recovered

    # ── 2. Lag-cap helper ────────────────────────────────────────────────────
    def _safe_lag_order(model: "VAR", n_obs: int, fallback: int = 1) -> int:  # type: ignore[name-defined]
        _cap = max(1, min(maxlags, n_obs // 4 - 1, 15))
        try:
            sel = model.select_order(maxlags=_cap)
            return max(int(sel.aic), 1)
        except Exception:
            return fallback

    # ── 3. 80 / 20 Validation split ──────────────────────────────────────────
    # Split is applied on df_work (the stationary transformed series).
    split    = int(len(df_work) * 0.8)
    train_wk = df_work.iloc[:split]
    test_wk  = df_work.iloc[split:]

    # Validation-window seeds for re-integration.
    # "split - 1" in df_work corresponds to a different row in df_orig
    # depending on d (because diff().dropna() drops the first d rows).
    # df_orig index offset: df_work.index[split-1] == df_orig.loc[that timestamp]
    _val_split_ts = df_work.index[split - 1]

    # Note: val_seed_level is in the Box-Cox domain (if use_boxcox=True) because
    # df_orig was already transformed above.  Re-integration happens in that domain;
    # the Box-Cox inverse applied later brings predictions back to original scale.
    val_seed_level:  pd.Series = df_orig.loc[_val_split_ts].copy()
    val_seed_delta1: pd.Series | None = None
    if diff_order == 2:
        # df_orig.diff() at the split boundary
        _delta1_at_split = df_orig.diff().dropna()
        val_seed_delta1  = _delta1_at_split.loc[_val_split_ts].copy() \
            if _val_split_ts in _delta1_at_split.index \
            else df_orig.diff().dropna().iloc[split - 1].copy()

    val_predictions: dict[str, pd.Series] = {c: pd.Series(dtype=float) for c in VAR_COLS}
    val_metrics: dict[str, dict]          = {
        c: {'RMSE': np.nan, 'MAE': np.nan, 'MAPE': np.nan} for c in VAR_COLS
    }
    val_lag_order = 1

    if len(test_wk) > 0 and len(train_wk) >= 10:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            try:
                mod_val       = VAR(train_wk)
                val_lag_order = _safe_lag_order(mod_val, len(train_wk), 1)
                res_val       = mod_val.fit(maxlags=val_lag_order, ic=None, verbose=False)

                fc_val_raw = res_val.forecast(
                    train_wk.values[-val_lag_order:], steps=len(test_wk)
                )
                fc_val_df = pd.DataFrame(fc_val_raw, columns=VAR_COLS, index=test_wk.index)

                # Re-integrate to levels (in Box-Cox domain if use_boxcox)
                fc_val_lvl = _recover_levels_val(
                    fc_val_df, diff_order, val_seed_level, val_seed_delta1
                )

                # Inverse Box-Cox → original CPU scale
                if use_boxcox:
                    for _c in VAR_COLS:
                        fc_val_lvl[_c] = var_bc_layers[_c].inverse(fc_val_lvl[_c])

                # Always compare against truly original levels (no Box-Cox)
                test_lvl = df_levels.loc[test_wk.index]

                for col in VAR_COLS:
                    y_true = test_lvl[col].values
                    y_pred = fc_val_lvl[col].values[:len(y_true)]

                    # NaN guard
                    if not np.all(np.isfinite(y_pred)):
                        continue

                    val_predictions[col] = pd.Series(
                        y_pred, index=test_lvl.index[:len(y_pred)]
                    )
                    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
                    mae  = float(mean_absolute_error(y_true, y_pred))
                    mask = y_true != 0
                    mape = (float(np.mean(
                        np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])
                    )) * 100) if mask.any() else 0.0
                    val_metrics[col] = {'RMSE': rmse, 'MAE': mae, 'MAPE': mape}

            except Exception:
                pass   # val_metrics stays as NaN dict — non-fatal

    # ── 4. Final fit on 100 % of df_work ─────────────────────────────────────
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            mod_final = VAR(df_work)
            lag_order = _safe_lag_order(mod_final, len(df_work), val_lag_order)
            res_final = mod_final.fit(maxlags=lag_order, ic=None, verbose=False)
        except Exception as exc:
            raise RuntimeError(f"VAR final fit failed: {exc}") from exc

    aic_val = float(res_final.aic)

    # ── 5. Multi-step forecast ────────────────────────────────────────────────
    freq_td = _infer_freq_td(df_orig['avg_cpu'])
    last_ts = df_orig.index[-1]
    fut_idx = pd.date_range(start=last_ts + freq_td, periods=forecast_steps, freq=freq_td)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fc_raw = res_final.forecast(df_work.values[-lag_order:], steps=forecast_steps)

    fc_df_raw = pd.DataFrame(fc_raw, columns=VAR_COLS, index=fut_idx)

    # Re-integrate to levels (in Box-Cox domain if use_boxcox)
    fc_df = _recover_levels(fc_df_raw, diff_order)

    # Inverse Box-Cox → original CPU scale
    if use_boxcox:
        for col in VAR_COLS:
            fc_df[col] = var_bc_layers[col].inverse(fc_df[col])

    # ── NaN / Inf fallback: drift from original CPU levels ───────────────────
    for col in VAR_COLS:
        if not np.all(np.isfinite(fc_df[col].values)):
            # Drift = mean of last 10 first-differences of ORIGINAL levels
            _delta   = df_levels[col].diff().dropna().iloc[-10:].mean()
            _last_v  = float(df_levels[col].iloc[-1])
            _drift_fc = pd.Series(
                [_last_v + _delta * h for h in range(1, forecast_steps + 1)],
                index=fut_idx,
            )
            fc_df[col] = _drift_fc

    forecasts = {col: fc_df[col] for col in VAR_COLS}

    # ── 6. Residuals (in transformed/stationary domain) ───────────────────────
    resid_df = pd.DataFrame(
        res_final.resid,
        columns=VAR_COLS,
        index=df_work.index[lag_order:],
    )

    # ── 7. Granger causality (on stationary data for validity) ───────────────
    granger: dict[str, dict] = {}
    for cause in VAR_COLS:
        for effect in VAR_COLS:
            if cause == effect:
                continue
            key = f"{cause}→{effect}"
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    gc_res = res_final.test_causality(
                        caused=effect, causing=cause, kind='f', signif=alpha,
                    )
                    gc_p = float(gc_res.pvalue)
                    granger[key] = {
                        'p_value':    gc_p,
                        'significant': gc_p < alpha,
                        'max_lag':     lag_order,
                    }
            except Exception:
                granger[key] = {
                    'p_value':    np.nan,
                    'significant': False,
                    'max_lag':     lag_order,
                }

    return {
        'forecasts':        forecasts,
        'val_metrics':      val_metrics,
        'val_predictions':  val_predictions,
        'lag_order':        int(lag_order),
        'diff_order':       int(diff_order),
        'adf_pvalues':      adf_pvalues,
        'granger':          granger,
        'var_residuals':    resid_df,
        'aic':              aic_val,
        'boxcox_lambdas':   {c: var_bc_layers[c].lam for c in VAR_COLS}
                            if use_boxcox else None,
    }


