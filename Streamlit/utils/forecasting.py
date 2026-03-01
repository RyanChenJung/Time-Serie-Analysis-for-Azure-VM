"""
utils/forecasting.py
────────────────────────────────────────────────────────────────────────────
Phase 3.1 — Advanced Statistical Forecasting Engine

Pure-functional backend (no Streamlit imports).  All functions accept a
pd.Series with a pd.DatetimeIndex and return plain Python dicts / DataFrames.
"""
import warnings
import sys
import io
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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_sarima_forecast(
    series: pd.Series,
    order: tuple,
    seasonal_order: tuple,
    forecast_steps: int,
) -> dict:
    """
    Dual-execution SARIMA: validation fit (80 / 20) + final fit (100%).

    Parameters
    ----------
    series         : DatetimeIndex pd.Series of avg_cpu values.
    order          : ARIMA (p, d, q) tuple.
    seasonal_order : Seasonal (P, D, Q, s) tuple.
    forecast_steps : Number of future steps to forecast.

    Returns
    -------
    dict with keys:
        'mean'            – pd.Series  (future forecast, DatetimeIndex)
        'upper'           – pd.Series  (95 % CI upper bound)
        'lower'           – pd.Series  (95 % CI lower bound)
        'aic'             – float      (AIC of the final model)
        'val_metrics'     – dict       {RMSE, MAE, MAPE}  (from hold-out)
        'val_predictions' – pd.Series  (model predictions on the test window,
                                        same index as test slice of series)
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from sklearn.metrics import mean_squared_error, mean_absolute_error

    series = _validate_series(series)

    p, d, q = order
    P, D, Q, s = seasonal_order

    # ── Step 1: Validation fit (80 / 20 hold-out) ───────────────────────────
    split = int(len(series) * 0.8)
    train_val = series.iloc[:split]
    test_val  = series.iloc[split:]

    val_predictions = pd.Series(dtype=float)
    val_metrics: dict = {'RMSE': np.nan, 'MAE': np.nan, 'MAPE': np.nan}

    if not test_val.empty:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                mod_val = SARIMAX(
                    train_val,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                res_val = mod_val.fit(disp=False)
                fc_val  = res_val.get_forecast(steps=len(test_val))
                val_pred_values = fc_val.predicted_mean.values

                # Keep numeric-indexed predictions aligned to the test window's
                # DatetimeIndex for later chart overlay
                val_predictions = pd.Series(
                    val_pred_values[:len(test_val)],
                    index=test_val.index,
                )

                y_true = test_val.values
                y_pred = val_predictions.values

                rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
                mae  = float(mean_absolute_error(y_true, y_pred))

                # MAPE — guard against zero actuals
                mask = y_true != 0
                mape = (float(np.mean(
                    np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])
                )) * 100) if mask.any() else 0.0

                val_metrics = {'RMSE': rmse, 'MAE': mae, 'MAPE': mape}

            except Exception:
                # Validation fit failed — metrics stay NaN, continue to final
                pass

    # ── Step 2: Final fit (100 % of data) for future forecast ───────────────
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mod_final = SARIMAX(
            series,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        res_final = mod_final.fit(disp=False)

    fc = res_final.get_forecast(steps=forecast_steps)
    fc_mean  = fc.predicted_mean
    fc_ci    = fc.conf_int(alpha=0.05)   # columns: lower avg_cpu, upper avg_cpu

    future_idx = _future_index(series, forecast_steps)
    fc_mean.index  = future_idx
    fc_ci.index    = future_idx

    return {
        'mean':            fc_mean,
        'upper':           fc_ci.iloc[:, 1],   # upper 95 % bound
        'lower':           fc_ci.iloc[:, 0],   # lower 95 % bound
        'aic':             float(res_final.aic),
        'val_metrics':     val_metrics,
        'val_predictions': val_predictions,    # DatetimeIndex, test window only
    }


def auto_optimize_sarima(series: pd.Series, seasonal_period: int) -> dict:
    """
    Use pmdarima.auto_arima to find the best (p,d,q)(P,D,Q,s) orders.

    ``seasonal_period`` is treated as a **hard constraint** for ``m`` (the
    seasonal cycle length) — p, d, q, P, D, Q are searched freely.
    The internal search log (one line per model tried) is captured from
    stdout and returned in the ``log`` key so callers can display it.

    Parameters
    ----------
    series          : DatetimeIndex pd.Series of avg_cpu values.
    seasonal_period : Detected or manually-set seasonality lag count.

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

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Capture pmdarima's trace=True stdout output
        with _StdoutCapture() as buf:
            model = pm.auto_arima(
                series.values,
                seasonal=use_seasonal,
                m=seasonal_period if use_seasonal else 1,   # hard constraint
                stepwise=True,
                information_criterion='aic',
                error_action='ignore',
                suppress_warnings=True,
                max_p=2,
                max_q=2,
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
                information_criterion='aic',
                error_action='ignore',
                suppress_warnings=True,
                max_p=2,
                max_q=2,
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
