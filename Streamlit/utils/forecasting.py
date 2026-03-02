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

    # ── Align exog to series index (handles cases where timestamps differ) ──────
    if exog_df is not None:
        exog_df = exog_df.reindex(series.index).dropna()
        # Trim series to the common index after reindex/dropna
        series = series.loc[exog_df.index]

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

    return {
        'mean':            fc_mean,
        'upper':           fc_ci.iloc[:, 1],
        'lower':           fc_ci.iloc[:, 0],
        'aic':             float(res_final.aic),
        'val_metrics':     val_metrics,
        'val_predictions': val_predictions,
        'model_name':      model_name,
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

    return {
        'mean':            fc_mean,
        'upper':           fc_upper,
        'lower':           fc_lower,
        'aic':             float(res_final.aic),
        'val_metrics':     val_metrics,
        'val_predictions': val_predictions,
        'val_residuals':   val_residuals,
        'model_label':     label,
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
