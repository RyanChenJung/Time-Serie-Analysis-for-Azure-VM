
import warnings
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import acf as sm_acf
from statsmodels.tsa.filters.hp_filter import hpfilter
from datetime import date, time as dtime
from concurrent.futures import ThreadPoolExecutor  # kept for future async tasks

from utils.data_engine import (
    load_data, apply_temporal_anchor, aggregate_and_interpolate,
    load_demo_cache, save_demo_cache, is_demo_data
)
from utils.diagnostics import (
    perform_adf_test, perform_ljung_box_test,
    calculate_refined_seasonality, classify_vm_suitability,
    calculate_pacf
)
from utils.baselines import expanding_window_cv, simple_holdout_evaluate
from utils.forecasting import (
    run_sarima_forecast, auto_optimize_sarima, stream_auto_arima,
    run_holt_winters_forecast, auto_optimize_holt_winters,
    build_exog_df, run_var_forecast,
)
import threading
import queue
import time as _time

# ---------------------------------------------------------------------------
# Page Config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Azure VM Workload Dashboard",
    page_icon="📊",
    layout="wide"
)

# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------
if 'fleet_diagnostics' not in st.session_state:
    st.session_state.fleet_diagnostics = {}
if 'manual_seasonality' not in st.session_state:
    st.session_state.manual_seasonality = {}
if 'diagnostics_run_time' not in st.session_state:
    st.session_state.diagnostics_run_time = None
if 'sarima_results' not in st.session_state:
    st.session_state.sarima_results = {}   # vm_id -> result dict from run_sarima_forecast
if 'sarima_params' not in st.session_state:
    st.session_state.sarima_params = {}    # vm_id -> {order, seasonal_order, steps}
if 'sarima_logs' not in st.session_state:
    st.session_state.sarima_logs = {}      # vm_id -> full log string from last Auto-Optimize
if 'hw_results' not in st.session_state:
    st.session_state.hw_results = {}       # vm_id -> result dict from run_holt_winters_forecast
if 'hw_params' not in st.session_state:
    st.session_state.hw_params = {}        # vm_id -> {trend_label, season_label, damped, steps, last_opt_summary, last_opt_error}
if 'hw_logs' not in st.session_state:
    st.session_state.hw_logs = {}          # vm_id -> full trace string from last Auto-Optimize
if 'var_results' not in st.session_state:
    st.session_state.var_results = {}      # vm_id -> result dict from run_var_forecast


def _invalidate_cv_if_seasonality_changed(vm_id, new_seasonality):
    """Invalidates cached CV and hold-out results when seasonality changes."""
    vm_diag = st.session_state.fleet_diagnostics.get(vm_id)
    if vm_diag:
        old_val = vm_diag.get('detected_seasonality')
        if old_val != new_seasonality:
            vm_diag['cv_results'] = None
            vm_diag['holdout_metrics'] = None
            vm_diag['detected_seasonality'] = new_seasonality
            st.session_state.fleet_diagnostics[vm_id] = vm_diag
            return True
    return False





# ---------------------------------------------------------------------------
# Sidebar — Global Controls
# ---------------------------------------------------------------------------
st.sidebar.header("⚙️ Controls")
start_date = st.sidebar.date_input("Start Date", date(2023, 1, 1))
start_time = st.sidebar.time_input("Start Time", dtime(0, 0))

granularity = st.sidebar.radio(
    "Select Granularity",
    ('5-min (Native)', '1-hour (Aggregated)'),
    index=0
)
granularity_arg = '5-min' if '5-min' in granularity else '1-hour'

uploaded_file = st.sidebar.file_uploader("Upload your own CSV", type=['csv'])

# ---------------------------------------------------------------------------
# Data Processing  (cached — only re-runs when inputs change)
# ---------------------------------------------------------------------------
@st.cache_data
def process_data(start_date, start_time, granularity_arg, uploaded_file_content):
    file_obj = None
    if uploaded_file_content:
        from io import StringIO
        class UploadedFile:
            def __init__(self, content): self._content = content
            def getvalue(self): return self._content
        file_obj = UploadedFile(uploaded_file_content)
    try:
        df = load_data(file_obj)
        df_anchored = apply_temporal_anchor(df, start_date, start_time)
        return aggregate_and_interpolate(df_anchored, granularity=granularity_arg)
    except Exception as e:
        st.error(f"Error processing data: {e}")
        return pd.DataFrame()


uploaded_file_content = uploaded_file.getvalue() if uploaded_file is not None else None
full_df = process_data(start_date, start_time, granularity_arg, uploaded_file_content)

# ---------------------------------------------------------------------------
# Fleet Diagnostics Logic
# ---------------------------------------------------------------------------
def run_vm_analysis_light(vm_id, vm_data, granularity_arg):
    """Lightweight Level-1 diagnostics: ADF + Ljung-Box + ACF seasonality.

    Stores the full 3-state suitability dict so the UI can render
    ✅ Stationary / ⚠️ Non-Stationary / ❌ White Noise without re-computing.
    """
    adf_p,   _  = perform_adf_test(vm_data)
    ljung_p, _  = perform_ljung_box_test(vm_data)
    detected_seasonality = calculate_refined_seasonality(vm_data, granularity_arg)
    suitability = classify_vm_suitability(adf_p, ljung_p)
    return vm_id, {
        'is_eligible':        suitability['is_eligible'],
        'suitability':        suitability,          # full 3-state dict
        'adf_p':              adf_p,
        'ljung_p':            ljung_p,
        'detected_seasonality': detected_seasonality,
        'cv_results':         None,
        'holdout_metrics':    None,
    }


def run_fleet_diagnostics(vm_list, full_df, granularity_arg):
    """Sequential fleet diagnostics with per-VM progress bar."""
    start_t = time.time()
    results = {}
    total = len(vm_list)
    progress_bar = st.progress(0, text="Starting fleet analysis...")
    status_box = st.empty()
    for i, vm_id in enumerate(vm_list):
        status_box.info(f"🔍 Processing VM {i+1}/{total}: `{vm_id[:24]}...`")
        vm_series = full_df[full_df['vm_id'] == vm_id]['avg_cpu']
        _, vm_result = run_vm_analysis_light(vm_id, vm_series, granularity_arg)
        results[vm_id] = vm_result
        progress_bar.progress((i + 1) / total, text=f"✅ Done {i+1}/{total} VMs")
    progress_bar.empty()
    status_box.empty()
    st.session_state.diagnostics_run_time = time.time() - start_t
    return results


# Auto-load demo cache on first visit
if not uploaded_file and not st.session_state.fleet_diagnostics:
    cached_res = load_demo_cache()
    if cached_res:
        st.session_state.fleet_diagnostics = cached_res
        st.session_state.diagnostics_run_time = 0.001

# ---------------------------------------------------------------------------
# Chart Helpers
# ---------------------------------------------------------------------------
PALETTE = {
    'actual':        '#94A3B8',
    'Naive':         '#3B82F6',
    'SNaive':        '#8B5CF6',
    'Drift':         '#F59E0B',
    'MovingAverage': '#10B981',
}

def _y_range_robust(series_values, spike_threshold=1.2, p99_pad=1.2):
    """
    Return a [0, ceiling] tuple for the *initial* Y-axis view.

    Strategy:
      - ceiling = p99 * p99_pad   (default: p99 × 1.2)
      - If true max > ceiling * spike_threshold (i.e., >44% above p99),
        we clip the initial view to ceiling so normal workload patterns
        stay visible.  Users can still zoom out to see the spike.
      - Otherwise (no significant spike), use true max × 1.1 as usual.

    Returns (y_range, clipped) where clipped=True means a spike was detected.
    """
    if len(series_values) == 0:
        return [0, 100], False
    arr = series_values.dropna().values if hasattr(series_values, 'dropna') else series_values
    if len(arr) == 0:
        return [0, 100], False
    p99     = float(np.percentile(arr, 99))
    true_max = float(arr.max())
    ceiling = max(p99 * p99_pad, 1.0)
    if true_max > ceiling * spike_threshold:
        return [0, ceiling], True          # initial view clips spike
    return [0, true_max * 1.1], False      # no spike — show everything


HW_COLOR = '#7C3AED'   # violet-700
SARIMA_COLOR = '#F43F5E'   # rose-500


def build_prediction_chart(vm_df, preds, vm_id, sarima_result=None, hw_result=None):
    """Full-width line chart: raw CPU + baseline forecast overlay.

    Overlays SARIMA and/or HW validation lines and future forecast ribbons
    when results are provided.  Y-axis uses p99-based robust scaling.
    """
    y_range, spike_clipped = _y_range_robust(vm_df['avg_cpu'])
    fig = go.Figure()
    # Actual CPU — always visible
    fig.add_trace(go.Scatter(
        x=vm_df['timestamp_dt'], y=vm_df['avg_cpu'],
        name='Actual', line=dict(color=PALETTE['actual'], width=1),
        visible=True,
        hovertemplate='%{x|%b %d %H:%M}<br>CPU: %{y:.1f}%<extra>Actual</extra>'
    ))

    # ── SARIMA validation-window overlay ────────────────────────────────────
    _vp = sarima_result.get('val_predictions') if sarima_result else None
    if _vp is not None and not _vp.empty:
        fig.add_trace(go.Scatter(
            x=_vp.index, y=_vp.values,
            name='SARIMA (validation)',
            line=dict(color=SARIMA_COLOR, width=1.5, dash='dot'),
            visible=True,
            hovertemplate='%{x|%b %d %H:%M}<br>CPU: %{y:.1f}%<extra>SARIMA val</extra>'
        ))

    # ── HW validation-window overlay ────────────────────────────────────────
    _hwvp = hw_result.get('val_predictions') if hw_result else None
    if _hwvp is not None and not _hwvp.empty:
        fig.add_trace(go.Scatter(
            x=_hwvp.index, y=_hwvp.values,
            name='HW (validation)',
            line=dict(color=HW_COLOR, width=1.5, dash='dot'),
            visible=True,
            hovertemplate='%{x|%b %d %H:%M}<br>CPU: %{y:.1f}%<extra>HW val</extra>'
        ))

    if not preds.empty:
        model_cols = [c for c in preds.columns if c != 'Actual']
        if model_cols:
            actuals = preds['Actual'].values
            rmses = {c: float(np.sqrt(((actuals - preds[c].values) ** 2).mean()))
                     for c in model_cols}
            best_model = min(rmses, key=rmses.get)
        else:
            best_model = None

        for col in model_cols:
            color = PALETTE.get(col, '#6B7280')
            is_best = (col == best_model)
            fig.add_trace(go.Scatter(
                x=preds.index, y=preds[col],
                name=f'🏆 {col}' if is_best else col,
                line=dict(color=color, width=2 if is_best else 1.5,
                          dash='solid' if is_best else 'dot'),
                visible=True if is_best else 'legendonly',
                hovertemplate=f'%{{x|%b %d %H:%M}}<br>CPU: %{{y:.1f}}%<extra>{col}</extra>'
            ))

    # ── HW future forecast + PI shading ─────────────────────────────────────
    if hw_result:
        hfc_mean  = hw_result['mean']
        hfc_upper = hw_result['upper']
        hfc_lower = hw_result['lower']
        fig.add_trace(go.Scatter(
            x=pd.concat([pd.Series(hfc_upper.index), pd.Series(hfc_lower.index[::-1])]),
            y=np.concatenate([hfc_upper.values, hfc_lower.values[::-1]]),
            fill='toself', fillcolor='rgba(124,58,237,0.12)',
            line=dict(width=0), showlegend=False, hoverinfo='skip', name='_hw_ci',
        ))
        fig.add_trace(go.Scatter(
            x=hfc_mean.index, y=hfc_mean.values,
            name='HW (95 % PI)',
            line=dict(color=HW_COLOR, width=2.5, dash='solid'),
            visible=True,
            hovertemplate='%{x|%b %d %H:%M}<br>Forecast: %{y:.1f}%<extra>HW</extra>'
        ))
        ci_max = float(hfc_upper.max())
        if ci_max > y_range[1]:
            y_range = [0, ci_max * 1.05]

    # ── SARIMA future forecast + CI shading ─────────────────────────────────
    if sarima_result:
        fc_mean  = sarima_result['mean']
        fc_upper = sarima_result['upper']
        fc_lower = sarima_result['lower']
        fig.add_trace(go.Scatter(
            x=pd.concat([pd.Series(fc_upper.index), pd.Series(fc_lower.index[::-1])]),
            y=np.concatenate([fc_upper.values, fc_lower.values[::-1]]),
            fill='toself', fillcolor='rgba(244,63,94,0.12)',
            line=dict(width=0), showlegend=False, hoverinfo='skip', name='_ci_band',
        ))
        fig.add_trace(go.Scatter(
            x=fc_mean.index, y=fc_mean.values,
            name='SARIMA (95 % CI)',
            line=dict(color=SARIMA_COLOR, width=2.5, dash='solid'),
            visible=True,
            hovertemplate='%{x|%b %d %H:%M}<br>Forecast: %{y:.1f}%<extra>SARIMA</extra>'
        ))
        ci_max = float(fc_upper.max())
        if ci_max > y_range[1]:
            y_range = [0, ci_max * 1.05]

    models_active = []
    if sarima_result: models_active.append('SARIMA')
    if hw_result:     models_active.append('HW')
    title_suffix = (' · ' + ' + '.join(models_active) + ' Active') if models_active else ''
    fig.update_layout(
        title=dict(
            text=f"CPU Usage & Forecasts — {vm_id[:28]}{title_suffix}",
            font=dict(size=15)
        ),
        template='plotly_white',
        height=440 if models_active else 380,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation='h', yanchor='bottom', y=1.01, xanchor='right', x=1),
        yaxis=dict(title='avg_cpu (%)', range=y_range, fixedrange=False),
        xaxis=dict(title='', fixedrange=False),
        hovermode='x unified',
    )
    return fig, spike_clipped


def build_decomposition_chart(series, period):
    """Full-width 3-panel STL decomposition: Trend / Seasonal / Residual."""
    res = seasonal_decompose(series.reset_index(drop=True), period=period, extrapolate_trend='freq')
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        subplot_titles=('Trend', 'Seasonality', 'Residual'),
                        vertical_spacing=0.08)
    fig.add_trace(go.Scatter(y=res.trend,    name='Trend',    line=dict(color='#3B82F6')), row=1, col=1)
    fig.add_trace(go.Scatter(y=res.seasonal, name='Seasonal', line=dict(color='#8B5CF6')), row=2, col=1)
    fig.add_trace(go.Scatter(y=res.resid,    name='Residual', line=dict(color='#EF4444', width=1)), row=3, col=1)
    fig.update_layout(
        template='plotly_white', height=420, showlegend=False,
        margin=dict(l=10, r=10, t=40, b=10),
        yaxis=dict(fixedrange=True),
        yaxis2=dict(fixedrange=True),
        yaxis3=dict(fixedrange=True),
    )
    return fig


def build_acf_chart(series, nlags):
    """Full-width ACF bar chart."""
    acf_vals = sm_acf(series.dropna(), nlags=nlags)
    lags = list(range(len(acf_vals)))
    colors = ['#EF4444' if abs(v) > 0.2 else '#94A3B8' for v in acf_vals]
    fig = go.Figure(go.Bar(x=lags, y=acf_vals, marker_color=colors, name='ACF'))
    fig.add_hline(y=0.2,  line_dash='dash', line_color='#EF4444', opacity=0.5)
    fig.add_hline(y=-0.2, line_dash='dash', line_color='#EF4444', opacity=0.5)
    fig.update_layout(
        title='Autocorrelation Function (ACF)',
        template='plotly_white', height=300,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(title='Lag', fixedrange=True),
        yaxis=dict(title='ACF', fixedrange=True),
    )
    return fig


def build_trend_residual_chart(series):
    """Two-panel Trend + Residual chart for Aperiodic VMs (s=1).

    Uses the Hodrick-Prescott (HP) filter to extract a smooth trend component.
    Lambda is auto-scaled by granularity:
      - 1-hour data  -> λ = 1600   (standard macro setting)
      - 5-min data   -> λ = 129600 (1600 × 81, scaled for higher-freq data)
    The residual is the raw series minus the extracted trend.
    """
    arr = series.dropna().reset_index(drop=True)
    n = len(arr)
    # HP-filter lambda: scaled so the trend is equally smooth across granularities
    lam = 129600 if n > 1000 else 1600
    try:
        # SparseEfficiencyWarning is a benign scipy hint about matrix format;
        # statsmodels' hp_filter triggers it internally and cannot be fixed here.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore',
                message='spsolve requires A be CSC or CSR matrix format',
                category=Warning,
            )
            cycle, trend = hpfilter(arr, lamb=lam)
    except Exception:
        # Fallback: 12-point centred moving average
        trend = arr.rolling(window=12, center=True, min_periods=1).mean()
        cycle = arr - trend

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        subplot_titles=('Trend (HP-Filter)', 'Residual'),
        vertical_spacing=0.10,
    )
    fig.add_trace(
        go.Scatter(y=trend.values, name='Trend',
                   line=dict(color='#3B82F6', width=2)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(y=arr.values, name='Raw',
                   line=dict(color='#94A3B8', width=1), opacity=0.45),
        row=1, col=1
    )
    resid_colors = ['#EF4444' if v > 0 else '#6366F1' for v in cycle.values]
    fig.add_trace(
        go.Bar(y=cycle.values, name='Residual',
               marker_color=resid_colors, opacity=0.7),
        row=2, col=1
    )
    fig.update_layout(
        template='plotly_white', height=420, showlegend=True,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation='h', yanchor='bottom', y=1.01, xanchor='right', x=1),
        yaxis=dict(fixedrange=True),
        yaxis2=dict(fixedrange=True, zeroline=True, zerolinecolor='#94A3B8'),
    )
    return fig


def build_pacf_chart(series, nlags):
    """Full-width PACF bar chart, mirroring build_acf_chart style."""
    pacf_vals = calculate_pacf(series, nlags)
    lags = list(range(len(pacf_vals)))
    colors = ['#EF4444' if abs(v) > 0.2 else '#94A3B8' for v in pacf_vals]
    fig = go.Figure(go.Bar(x=lags, y=pacf_vals, marker_color=colors, name='PACF'))
    fig.add_hline(y=0.2,  line_dash='dash', line_color='#EF4444', opacity=0.5)
    fig.add_hline(y=-0.2, line_dash='dash', line_color='#EF4444', opacity=0.5)
    fig.update_layout(
        title='Partial Autocorrelation Function (PACF)',
        template='plotly_white', height=300,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(title='Lag', fixedrange=True),
        yaxis=dict(title='PACF', fixedrange=True),
    )
    return fig


# ---------------------------------------------------------------------------
# Main App
# ---------------------------------------------------------------------------
st.title("📊 Azure VM Workload Dashboard")
st.caption("Phase 2.9.5 · Aperiodic Mode · SARIMA ARIMA-only · pmdarima restored")

if not full_df.empty:
    vm_list = sorted(full_df['vm_id'].unique())

    tab1, tab2 = st.tabs(["🚀 Fleet Analytics", "🔍 VM Deep Dive"])

    # -----------------------------------------------------------------------
    # TAB 1 — FLEET ANALYTICS
    # -----------------------------------------------------------------------
    with tab1:
        st.header("Fleet Suitability Overview")

        # --- Explainability box ---
        with st.expander("💡 How the Gatekeeper Works", expanded=False):
            st.markdown("""
The **Gatekeeper** is a two-stage statistical filter that determines whether a VM's CPU
time-series contains *learnable patterns* worth modelling.

#### Stage 1 & 2 — Stationarity & White Noise Tests

| Test | What it checks | Threshold | Outcome if failed |
|---|---|---|---|
| **ADF (Augmented Dickey-Fuller)** | Stationarity — does the series' mean/variance drift over time (unit root)? | p < 0.05 | Marked ⚠️ — drift models would extrapolate incorrectly; differencing (d > 0) required |
| **Ljung-Box** | White noise — is there *any* autocorrelation structure at lag 10? | p < 0.05 | Marked ❌ — the signal is pure noise; no model can beat a flat mean |

> A VM must **pass the Ljung-Box test** (primary gate) to be eligible for forecasting.
> A series that is white noise has no exploitable autocorrelation structure regardless of
> stationarity. Non-stationary-but-structured VMs are still eligible — ARIMA handles them
> via differencing.

---

#### Stage 3 — Auto-Seasonality Detection

Once a VM passes the Gatekeeper, its seasonal period is detected automatically using the
**Autocorrelation Function (ACF)**:

1. **Compute ACF** over a granularity-specific search window:
   - `1-hour` mode → lags **4 to 168** (4 hours to 1 week)
   - `5-min` mode  → lags **48 to 2016** (4 hours to 1 week)
2. **Find the dominant peak:** The lag with the highest ACF value inside the search window is identified.
3. **Apply the 0.2 significance threshold:** If that peak ACF value exceeds **0.2**, the corresponding
   lag is declared the seasonal period `s`. If no lag clears the threshold, `s = 1` — the VM is
   classified as **Aperiodic** (trend-driven or random, with no repeating seasonal cycle).

| Result | Meaning | Deep Dive behaviour |
|---|---|---|
| `s > 1` | **Periodic** — clear repeating cycle detected | Full STL decomposition shown |
| `s = 1` | **Aperiodic** — no dominant cycle | Trend & Residual analysis shown instead |

```
Example: 5-min data with a 24-hour workday cycle
  → dominant ACF peak at lag 288 (288 × 5 min = 24 hr)
  → ACF[288] = 0.63  >  0.2  →  s = 288  (Periodic)

Example: Bursty / trend-driven VM
  → ACF peak below 0.2 across all search lags
  → s = 1  (Aperiodic — Trend & Residual mode)
```

> **Manual Override:** If the auto-detected period seems wrong (e.g., weekend patterns not
> captured), use the **Seasonality Override** control in the sidebar to set `s` manually.
> Changing the override invalidates cached CV results to ensure re-evaluation.
""")

        st.divider()

        if st.button("🔄 Run / Refresh Fleet Diagnostics", type="primary"):
            results = run_fleet_diagnostics(vm_list, full_df, granularity_arg)
            st.session_state.fleet_diagnostics = results
            if not uploaded_file and is_demo_data(full_df):
                save_demo_cache(results)

        if st.session_state.fleet_diagnostics:
            run_time = st.session_state.diagnostics_run_time or 0
            time_label = (f"⚡ Fleet Processing Time: {run_time*1000:.0f}ms"
                          if run_time < 1.0 else f"⚡ Fleet Processing Time: {run_time:.2f}s")
            st.info(time_label)

            diag_vals = list(st.session_state.fleet_diagnostics.values())
            eligible_count = sum(1 for d in diag_vals if d['is_eligible'])
            total_vms = len(st.session_state.fleet_diagnostics)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total VMs", total_vms)
            c2.metric("Eligible for Forecasting", eligible_count)
            c3.metric("Eligible %", f"{(eligible_count/total_vms)*100:.1f}%" if total_vms else "N/A")
            c4.metric("Non-Forecastable", total_vms - eligible_count)

            st.divider()

            table_data = []
            for vm_id, diag in st.session_state.fleet_diagnostics.items():
                suit = diag.get('suitability') or classify_vm_suitability(diag['adf_p'], diag['ljung_p'])
                table_data.append({
                    "VM ID":                      vm_id,
                    "Status":                     f"{suit['icon']} {suit['label']}",
                    "ADF p-value":                f"{diag['adf_p']:.4f}",
                    "LB p-value":                 f"{diag['ljung_p']:.4f}",
                    "Detected Seasonality (lags)": diag['detected_seasonality'],
                    "Guidance":                   suit['message'],
                })
            st.dataframe(pd.DataFrame(table_data), width='stretch', hide_index=True)
        else:
            st.warning("No diagnostic data found. Click **Run / Refresh Fleet Diagnostics** above.")

    # -----------------------------------------------------------------------
    # TAB 2 — VM DEEP DIVE
    # -----------------------------------------------------------------------
    with tab2:
        selected_vm = st.selectbox("Select VM for Deep Dive", vm_list, key="deep_dive_vm")

        if selected_vm:
            vm_df  = full_df[full_df['vm_id'] == selected_vm].copy()
            series = vm_df.set_index('timestamp_dt')['avg_cpu']
            # Build causally-lagged exog (max_cpu, min_cpu shifted by 1 step)
            # Always computed here so ARIMAX toggle can use it on demand.
            exog_df = build_exog_df(vm_df)
            diag = st.session_state.fleet_diagnostics.get(selected_vm)

            if diag:
                # --- Sidebar: seasonality override ---
                st.sidebar.markdown("---")
                st.sidebar.subheader(f"Manual Override: {selected_vm[:18]}…")
                use_override = st.sidebar.checkbox(
                    "Enable Seasonality Override",
                    value=selected_vm in st.session_state.manual_seasonality
                )
                effective_seasonality = diag['detected_seasonality']
                if use_override:
                    new_val = st.sidebar.number_input(
                        "Set Seasonality (lags)", min_value=1,
                        value=st.session_state.manual_seasonality.get(
                            selected_vm, diag['detected_seasonality']
                        )
                    )
                    if new_val != st.session_state.manual_seasonality.get(selected_vm):
                        st.session_state.manual_seasonality[selected_vm] = new_val
                        if _invalidate_cv_if_seasonality_changed(selected_vm, new_val):
                            st.rerun()
                    effective_seasonality = new_val
                else:
                    if selected_vm in st.session_state.manual_seasonality:
                        st.session_state.manual_seasonality.pop(selected_vm)
                        if _invalidate_cv_if_seasonality_changed(selected_vm, diag['detected_seasonality']):
                            st.rerun()

                # --- Level 1: hold-out evaluation (auto-runs if cache empty) ---
                if diag.get('holdout_metrics') is None:
                    with st.spinner("⏳ Calculating baseline forecasts on hold-out window…"):
                        metrics, preds = simple_holdout_evaluate(series.reset_index(drop=True), effective_seasonality)
                        diag['holdout_metrics'] = metrics
                        diag['holdout_preds'] = preds
                        st.session_state.fleet_diagnostics[selected_vm] = diag

                # --- 3-state status banner ---
                suit = diag.get('suitability') or classify_vm_suitability(diag['adf_p'], diag['ljung_p'])
                _stats = f"ADF p={diag['adf_p']:.4f} · LB p={diag['ljung_p']:.4f} · Seasonality: **{effective_seasonality}** lags"
                if suit['status'] == 'stationary':
                    st.success(f"**{suit['icon']} {suit['label']}** — {suit['message']}  \n{_stats}")
                elif suit['status'] == 'non_stationary':
                    st.warning(f"**{suit['icon']} {suit['label']}** — {suit['message']}  \n{_stats}")
                else:
                    st.error(f"**{suit['icon']} {suit['label']}** — {suit['message']}  \n{_stats}")

                # ═══════════════════════════════════════════════════════════
                # SECTION 1 — CPU Usage & Baseline Forecasts
                # ═══════════════════════════════════════════════════════════
                st.divider()
                st.subheader("📈 CPU Usage & Baseline Forecasts")
                st.caption(
                    "The **last 20% of the series** is the hold-out test window. "
                    "Baseline models are fit on the training window and projected forward — "
                    "their performance here sets the hurdle any advanced model must clear."
                )

                preds_df = diag.get('holdout_preds', pd.DataFrame())
                # Re-attach datetime index to preds for overlay chart
                if not preds_df.empty and not isinstance(preds_df.index, pd.DatetimeIndex):
                    train_len = len(series) - len(preds_df)
                    preds_df.index = series.index[train_len:]

                sarima_res = st.session_state.sarima_results.get(selected_vm)
                hw_res     = st.session_state.hw_results.get(selected_vm)
                cpu_fig, spike_clipped = build_prediction_chart(
                    vm_df, preds_df, selected_vm,
                    sarima_result=sarima_res,
                    hw_result=hw_res,
                )
                st.plotly_chart(cpu_fig, width='stretch')
                if spike_clipped:
                    st.caption(
                        "📊 *Y-axis scaled to 99th percentile for trend clarity. "
                        "Spikes may be clipped in the default view — "
                        "scroll or pinch on the chart to zoom out and see peak values.*"
                    )
                # Extra margin before leaderboard — separates Observation from Evaluation
                st.markdown("&nbsp;", unsafe_allow_html=True)

                # ═══════════════════════════════════════════════════════════
                # SECTION 2 — Model Leaderboard
                # ═══════════════════════════════════════════════════════════
                st.divider()
                st.subheader("🏆 Model Leaderboard")
                cv_res = diag.get('cv_results')
                curr_metrics = cv_res if cv_res is not None else diag.get('holdout_metrics', pd.DataFrame())
                metric_type = "Expanding-Window CV · RMSE" if cv_res is not None else "Hold-out · RMSE"
                st.caption(f"Ranked by **{metric_type}** (lower is better). ⭐ = current best.")

                if curr_metrics is not None and not curr_metrics.empty:
                    display_df = curr_metrics.copy()

                    # ── Inject HW row into leaderboard if available ──────────
                    hw_res_lb = st.session_state.hw_results.get(selected_vm)
                    if hw_res_lb:
                        vm_hw = hw_res_lb.get('val_metrics', {})
                        if all(np.isfinite(vm_hw.get(k, np.nan)) for k in ('RMSE', 'MAE', 'MAPE')):
                            hw_row = pd.DataFrame([{
                                'Model':     'Holt-Winters',
                                'RMSE':      vm_hw['RMSE'],
                                'MAE':       vm_hw['MAE'],
                                'MAPE':      vm_hw['MAPE'],
                                'is_winner': False,
                            }])
                            display_df = pd.concat([display_df, hw_row], ignore_index=True)

                    # ── Inject SARIMA / ARIMAX row into leaderboard if available ──
                    sarima_res = st.session_state.sarima_results.get(selected_vm)
                    if sarima_res:
                        vm = sarima_res.get('val_metrics', {})
                        if all(np.isfinite(vm.get(k, np.nan)) for k in ('RMSE', 'MAE', 'MAPE')):
                            sarima_row = pd.DataFrame([{
                                'Model':     sarima_res.get('model_name', 'SARIMA'),
                                'RMSE':      vm['RMSE'],
                                'MAE':       vm['MAE'],
                                'MAPE':      vm['MAPE'],
                                'is_winner': False,
                            }])
                            display_df = pd.concat([display_df, sarima_row], ignore_index=True)

                    # ── Inject VAR row into leaderboard if available ─────────
                    var_res_lb = st.session_state.var_results.get(selected_vm)
                    if var_res_lb:
                        _var_avg_vm = var_res_lb.get('val_metrics', {}).get('avg_cpu', {})
                        if all(np.isfinite(_var_avg_vm.get(k, np.nan)) for k in ('RMSE', 'MAE', 'MAPE')):
                            var_row = pd.DataFrame([{
                                'Model':     'VAR (Systemic)',
                                'RMSE':      _var_avg_vm['RMSE'],
                                'MAE':       _var_avg_vm['MAE'],
                                'MAPE':      _var_avg_vm['MAPE'],
                                'is_winner': False,
                            }])
                            display_df = pd.concat([display_df, var_row], ignore_index=True)

                    # Recalculate winner across all models (including VAR)
                    min_rmse = display_df['RMSE'].min()
                    display_df['is_winner'] = display_df['RMSE'] == min_rmse
                    display_df = display_df.sort_values('RMSE').reset_index(drop=True)

                    winner_row  = display_df[display_df['is_winner']]
                    winner_name = winner_row['Model'].iloc[0] if not winner_row.empty else '—'
                    winner_rmse = winner_row['RMSE'].iloc[0]  if not winner_row.empty else float('nan')

                    display_df['Model'] = display_df.apply(
                        lambda r: f"🏆 Best · {r['Model']}" if r['is_winner'] else r['Model'], axis=1
                    )
                    display_df['RMSE'] = display_df['RMSE'].map('{:.3f}'.format)
                    display_df['MAE']  = display_df['MAE'].map('{:.3f}'.format)
                    display_df['MAPE'] = display_df['MAPE'].map('{:.2f}%'.format)

                    def _style_winner(row):
                        """
                        Highlight the winner row with a light green background
                        and dark green text for maximum readability.
                        """
                        if row['is_winner']:
                            return ['background-color: #F0FDF4; color: #15803D; font-weight: 700'] * len(row)
                        return [''] * len(row)

                    styled = (
                        display_df[['Model', 'RMSE', 'MAE', 'MAPE', 'is_winner']]
                        .style.apply(_style_winner, axis=1)
                        .hide(axis='index')
                    )
                    st.dataframe(styled, width='stretch', column_config={'is_winner': None})

                    # Winner summary box
                    st.success(
                        f"🏆 **Winner: {winner_name}** — RMSE: **{winner_rmse:.3f}** "
                        f"({'CV' if cv_res is not None else 'hold-out'})"
                    )

                # Level 2 — on-demand CV
                st.markdown("")
                if cv_res is None:
                    if st.button("🔬 Deep Stability Check — Run Expanding Window CV", type="secondary"):
                        with st.spinner("Running expanding-window cross-validation…"):
                            step_size = 288 if granularity_arg == '5-min' else 24
                            cv_results = expanding_window_cv(
                                series.reset_index(drop=True), effective_seasonality, step_size=step_size
                            )
                            diag['cv_results'] = cv_results
                            st.session_state.fleet_diagnostics[selected_vm] = diag
                            st.rerun()
                else:
                    col_a, col_b = st.columns([3, 1])
                    col_a.success("✅ Stability verified via Expanding-Window Cross-Validation.")
                    if col_b.button("↺ Reset CV Cache"):
                        diag['cv_results'] = None
                        st.session_state.fleet_diagnostics[selected_vm] = diag
                        st.rerun()

                # ═══════════════════════════════════════════════════════════
                # SECTION 3 — Workload Decomposition (STL or Aperiodic mode)
                # ═══════════════════════════════════════════════════════════
                st.divider()

                _is_aperiodic = (effective_seasonality <= 1)

                if _is_aperiodic:
                    # ── Aperiodic mode: show Trend & Residual via HP-Filter ──
                    st.subheader("🔬 Workload Decomposition (Aperiodic Mode)")
                    st.caption(
                        "No dominant seasonal cycle was detected (ACF peak < 0.2 across the full "
                        f"search window). This VM is **Aperiodic** — its workload is trend-driven "
                        "or random rather than periodic. The chart below separates the **Trend** "
                        "(via HP-Filter, blue line) from the **Residual** (unexplained fluctuations). "
                        "Use the **Manual Override** in the sidebar if you believe a cycle exists."
                    )
                    try:
                        st.plotly_chart(
                            build_trend_residual_chart(series),
                            width='stretch'
                        )
                    except Exception as e:
                        st.warning(f"Trend & Residual analysis unavailable: {e}")

                elif len(series) <= 2 * effective_seasonality:
                    # ── Periodic but insufficient data for STL ───────────────
                    st.subheader("🔬 Seasonal Decomposition (STL)")
                    st.info(
                        f"Not enough data for STL decomposition: need at least "
                        f"**{2 * effective_seasonality}** observations "
                        f"(2 × seasonality period = {effective_seasonality} lags), "
                        f"have **{len(series)}**."
                    )

                else:
                    # ── Periodic + sufficient data: full STL ─────────────────
                    st.subheader("🔬 Seasonal Decomposition (STL)")
                    st.caption(
                        "Decomposition splits the raw signal into three independent components: "
                        "**Trend** (long-run direction), **Seasonality** (repeating cycle of "
                        f"**{effective_seasonality}** lags), and **Residual** (unexplained noise). "
                        "A dominant seasonal component confirms the detected period is meaningful."
                    )
                    try:
                        st.plotly_chart(
                            build_decomposition_chart(series, effective_seasonality),
                            width='stretch'
                        )
                    except Exception as e:
                        st.warning(f"STL Decomposition unavailable: {e}")

                # ═══════════════════════════════════════════════════════════
                # SECTION 4 — ACF + PACF (side-by-side)
                # ═══════════════════════════════════════════════════════════
                st.divider()
                st.subheader("📊 Autocorrelation (ACF) & Partial Autocorrelation (PACF)")
                st.caption(
                    "**ACF** measures the correlation of the series with its own lagged values "
                    "(including indirect correlations). **PACF** isolates the *direct* effect of each "
                    "lag after removing the influence of intervening lags — useful for identifying "
                    "the AR order `p` for SARIMA. Bars highlighted in **red** exceed the ±0.2 "
                    "significance threshold."
                )
                nlags = min(effective_seasonality * 3 if effective_seasonality > 1 else 72, len(series) // 2 - 1)
                nlags = max(nlags, 10)
                _acf_col, _pacf_col = st.columns(2)
                with _acf_col:
                    st.plotly_chart(build_acf_chart(series, nlags), width='stretch')
                with _pacf_col:
                    st.plotly_chart(build_pacf_chart(series, nlags), width='stretch')

                # ═══════════════════════════════════════════════════════════
                # SECTION 5 — Holt-Winters Exponential Smoothing (Phase 3.2)
                # ═══════════════════════════════════════════════════════════
                st.divider()
                st.subheader("🌊 Advanced Forecasting (Holt-Winters)")

                _hw_aperiodic = (effective_seasonality <= 1)
                if _hw_aperiodic:
                    st.info(
                        "ℹ️ **Aperiodic Mode** — no seasonal cycle detected (`s = 1`). "
                        "Seasonal component is automatically set to **None**. "
                        "Use the sidebar **Manual Override** to enable seasonality."
                    )

                _hw_saved = st.session_state.hw_params.get(selected_vm, {})

                # ── Persistent log expander (SARIMA pattern) ──────────────────
                # Rendered every pass from session state; never wiped by st.rerun().
                _saved_hw_logs = st.session_state.hw_logs.get(selected_vm)
                if _saved_hw_logs:
                    with st.expander("📝 Last HW Optimization Logs", expanded=False):
                        st.code(_saved_hw_logs, language=None)

                # ── Last optimization result banner ───────────────────────────
                _hw_last_sum = _hw_saved.get('last_opt_summary')
                _hw_last_err = _hw_saved.get('last_opt_error')
                if _hw_last_sum:
                    st.success(f"✅ **HW Optimization Complete** — {_hw_last_sum}")
                if _hw_last_err:
                    st.error(f"❌ HW Optimization failed: {_hw_last_err}")

                with st.expander("🌊 Holt-Winters Control Panel", expanded=True):
                    st.caption(
                        "**Holt-Winters Triple Exponential Smoothing** does not require stationarity "
                        "and is effective for workloads with clear **level + trend + seasonal** structure. "
                        "Parameters are fitted automatically using AIC minimisation. "
                        "Results compete directly in the **Leaderboard**."
                    )

                    _hw_horizon_max = min(288 if granularity_arg == '5-min' else 24 * 7, len(series) // 4)
                    _hw_steps = st.slider(
                        "Forecast Horizon (steps)", min_value=6,
                        max_value=max(_hw_horizon_max, 12),
                        value=_hw_saved.get('steps', min(48, len(series) // 5)),
                        key=f"hw_horizon_{selected_vm}",
                    )

                    _hw_c1, _hw_c2, _hw_c3 = st.columns(3)

                    _trend_opts  = ['None', 'Additive', 'Multiplicative']
                    _season_opts = ['None', 'Additive', 'Multiplicative']

                    # Widget indices derive directly from hw_params so they update
                    # correctly after Auto-Optimize writes new values and st.rerun() fires.
                    _hw_trend_label = _hw_saved.get('trend_label', 'Additive')
                    _hw_trend_idx   = _trend_opts.index(_hw_trend_label) if _hw_trend_label in _trend_opts else 1
                    _hw_trend_sel   = _hw_c1.selectbox(
                        "Trend", _trend_opts, index=_hw_trend_idx,
                        key=f"hw_trend_{selected_vm}",
                    )

                    if _hw_aperiodic:
                        _hw_season_sel = 'None'
                        _hw_c2.selectbox("Seasonal", _season_opts, index=0,
                                         disabled=True, key=f"hw_season_{selected_vm}")
                    else:
                        _hw_season_label = _hw_saved.get('season_label', 'Additive')
                        _hw_season_idx   = _season_opts.index(_hw_season_label) if _hw_season_label in _season_opts else 1
                        _hw_season_sel   = _hw_c2.selectbox(
                            "Seasonal", _season_opts, index=_hw_season_idx,
                            key=f"hw_season_{selected_vm}",
                        )

                    _hw_damped = _hw_c3.checkbox(
                        "Damped Trend", value=_hw_saved.get('damped', False),
                        key=f"hw_damped_{selected_vm}",
                        help="Dampens the trend so it converges to a flat line in the long run.",
                    )

                    # Map UI labels → statsmodels args
                    _label_map    = {'None': None, 'Additive': 'add', 'Multiplicative': 'mul'}
                    _hw_trend_sm  = _label_map[_hw_trend_sel]
                    _hw_season_sm = _label_map[_hw_season_sel]

                    st.markdown("")
                    # ── Box-Cox toggle ────────────────────────────────────────────
                    _hw_boxcox = st.checkbox(
                        "🔬 Enable Box-Cox Transformation",
                        value=False,
                        key=f"hw_boxcox_{selected_vm}",
                        help=(
                            "Applies Box-Cox variance-stabilisation before fitting. "
                            "Helps when CPU residuals fan out (multiplicative error). "
                            "Forecasts and metrics are back-transformed to original CPU % scale."
                        ),
                    )
                    _hw_btn1, _hw_btn2 = st.columns(2)

                    _do_hw_opt = _hw_btn1.button(
                        "✨ Auto-Optimize Parameters",
                        key=f"hw_auto_opt_{selected_vm}",
                        help="Grid-search all Trend × Seasonal × Damping combos; picks lowest Val RMSE.",
                    )
                    _do_hw_run = _hw_btn2.button(
                        "🚀 Run Holt-Winters Forecast", type="primary",
                        key=f"hw_exec_{selected_vm}",
                    )

                    # ── Auto-Optimize (SARIMA-pattern double persistence) ─────────
                    # CRITICAL: st.rerun() must fire AFTER the st.status context
                    # manager exits, otherwise Streamlit tears down the status widget
                    # mid-execution and the trace lines vanish from the DOM.
                    # Solution: collect the rerun flag, complete the status normally,
                    # exit its context, THEN call st.rerun().
                    if _do_hw_opt:
                        _hw_do_rerun = False
                        _n_combos = sum(
                            1 for t, s, d in
                            __import__('itertools').product(
                                ['add', 'mul', None],
                                ['add', 'mul', None] if effective_seasonality > 1 else [None],
                                [True, False],
                            )
                            if not (d and t is None)
                        )
                        with st.status(
                            f"🔍 Scanning {_n_combos} Holt-Winters configurations…",
                            expanded=True,
                        ) as _hw_status:
                            _hw_opt_lines: list[str] = []

                            def _hw_cb(line: str) -> None:
                                _hw_opt_lines.append(line)
                                st.write(line)

                            try:
                                _opt_res = auto_optimize_holt_winters(
                                    series,
                                    seasonal_periods=effective_seasonality,
                                    progress_callback=_hw_cb,
                                )
                                _best     = _opt_res['best_config']
                                _best_lbl = _opt_res['best_label']
                                _best_rmse= _opt_res['best_rmse']

                                _sm_to_lbl = {None: 'None', 'add': 'Additive', 'mul': 'Multiplicative'}

                                # ── Double persistence (params + logs) ──────────────────────
                                _prev_hw = st.session_state.hw_params.get(selected_vm, {})
                                st.session_state.hw_params[selected_vm] = {
                                    **_prev_hw,
                                    'trend_label':      _sm_to_lbl[_best['trend']],
                                    'season_label':     _sm_to_lbl[_best['seasonal']],
                                    'damped':           bool(_best['damped']),
                                    'last_opt_summary': (
                                        f"Best: **{_best_lbl}** — "
                                        f"Val RMSE = **{_best_rmse:.4f}**"
                                    ),
                                    'last_opt_error':   None,
                                }
                                # Persist trace text to dedicated hw_logs dict
                                st.session_state.hw_logs[selected_vm] = '\n'.join(_hw_opt_lines)

                                _hw_status.update(
                                    label=(
                                        f"✅ Best config: {_best_lbl} — "
                                        f"Val RMSE = {_best_rmse:.4f}"
                                    ),
                                    state='complete', expanded=False,
                                )
                                _hw_do_rerun = True

                            except Exception as _e:
                                _err_str = str(_e)
                                _prev_hw2 = st.session_state.hw_params.get(selected_vm, {})
                                st.session_state.hw_params[selected_vm] = {
                                    **_prev_hw2,
                                    'last_opt_summary': None,
                                    'last_opt_error':   _err_str,
                                }
                                st.session_state.hw_logs[selected_vm] = '\n'.join(_hw_opt_lines)
                                _hw_status.update(
                                    label=f"❌ Auto-Optimize failed: {_e}",
                                    state='error', expanded=True,
                                )

                        # st.status context has exited cleanly — safe to rerun now
                        if _hw_do_rerun:
                            st.rerun()

                    # ── Run Forecast ──────────────────────────────────────────────
                    if _do_hw_run:
                        with st.spinner("Fitting Holt-Winters model (validation + final fit)…"):
                            try:
                                _hw_boxcox = st.session_state.get(f"hw_boxcox_{selected_vm}", False)
                                hw_result = run_holt_winters_forecast(
                                    series,
                                    trend=_hw_trend_sm,
                                    seasonal=_hw_season_sm,
                                    seasonal_periods=effective_seasonality,
                                    damped=_hw_damped,
                                    forecast_steps=int(_hw_steps),
                                    use_boxcox=_hw_boxcox,
                                )
                                st.session_state.hw_results[selected_vm] = hw_result
                                _prev_hw3 = st.session_state.hw_params.get(selected_vm, {})
                                st.session_state.hw_params[selected_vm] = {
                                    **_prev_hw3,  # preserve last_opt_summary/logs keys
                                    'trend_label':  _hw_trend_sel,
                                    'season_label': _hw_season_sel,
                                    'damped':       _hw_damped,
                                    'steps':        int(_hw_steps),
                                }
                                st.rerun()
                            except Exception as e:
                                st.error(f"Holt-Winters failed: {e}")

                # ── Active forecast summary + clear button ────────────────────────
                _cur_hw = st.session_state.hw_results.get(selected_vm)
                if _cur_hw:
                    _hw_vm  = _cur_hw.get('val_metrics', {})
                    _hw_aic = _cur_hw.get('aic', float('nan'))
                    _hw_bc_lam = _cur_hw.get('boxcox_lambda')
                    _hw_bc_str = (
                        f" · Box-Cox λ: **{_hw_bc_lam:.3f}**"
                        + (" *(≈ linear — minimal effect)*" if _hw_bc_lam and abs(_hw_bc_lam - 1.0) < 0.15 else "")
                        if _hw_bc_lam is not None else ""
                    )
                    st.success(
                        f"✅ **{_cur_hw.get('model_label', 'HW')}** active · "
                        f"AIC: **{_hw_aic:.1f}** · "
                        f"Val RMSE: **{_hw_vm.get('RMSE', float('nan')):.3f}**"
                        + _hw_bc_str
                    )
                    if st.button("🗑️ Clear HW Forecast", key=f"clear_hw_{selected_vm}"):
                        st.session_state.hw_results.pop(selected_vm, None)
                        st.session_state.hw_params.pop(selected_vm, None)
                        st.session_state.hw_logs.pop(selected_vm, None)
                        st.rerun()

                # ═══════════════════════════════════════════════════════════
                # SECTION 6 — Expert Forecasting Panel (SARIMA)
                # ═══════════════════════════════════════════════════════════
                st.divider()
                st.subheader("🔮 Advanced Statistical Forecasting (ARIMA / SARIMA / ARIMAX)")

                # Gatekeeper guard — allow expert override
                _sarima_eligible = diag['is_eligible']
                if not _sarima_eligible:
                    col_gate, col_force = st.columns([3, 1])
                    col_gate.error(
                        "🔒 **Gatekeeper Blocked** — This VM fails the white-noise check. "
                        "SARIMA results may be statistically meaningless."
                    )
                    _force_sarima = col_force.checkbox("Expert Override (bypass gate)",
                                                       key=f"force_sarima_{selected_vm}")
                    if not _force_sarima:
                        st.info("Enable 'Expert Override' above to unlock SARIMA on this VM.")
                    else:
                        _sarima_eligible = True

                if _sarima_eligible:
                    # Restore cached params for this VM (populated by Auto-Optimize)
                    _cached_p = st.session_state.sarima_params.get(selected_vm, {})

                    # Smart `d` default: 1 for Non-Stationary, 0 otherwise
                    _d_default = 1 if suit.get('status') == 'non_stationary' else 0

                    # ── Persistent log expander (Phase 3.1.6) ───────────────
                    # Read from dedicated sarima_logs dict (never from sarima_results)
                    # so a partial dict from Auto-Optimize never bleeds into chart code.
                    _saved_logs = st.session_state.sarima_logs.get(selected_vm)
                    if _saved_logs:
                        with st.expander("📝 Last Auto-Optimize Logs", expanded=False):
                            st.code(_saved_logs, language=None)

                    with st.expander("🔮 Advanced Forecasting Control Panel", expanded=True):
                        st.caption(
                            "SARIMA / ARIMAX is an advanced statistical model. "
                            "It captures trend (via differencing **d**) and seasonality (via **s**). "
                            "Enable **Multi-Metric Regression** to add Max/Min as exogenous inputs (ARIMAX). "
                            "Results compete directly in the Leaderboard above."
                        )

                        # ── ARIMAX toggle ──────────────────────────────────────────────────
                        _use_exog = st.toggle(
                            "🚀 Enable Multi-Metric Regression — ARIMAX (Max/Min as exogenous)",
                            value=_cached_p.get('use_exog', False),
                            key=f"use_exog_{selected_vm}",
                            help=(
                                "Uses `max_cpu(t−1)` and `min_cpu(t−1)` as causal regressors. "
                                "The result appears as **ARIMAX** in the Leaderboard."
                            ),
                        )
                        if _use_exog:
                            st.caption(
                                "⚠️ Auto-Optimizing with exogenous variables (ARIMAX) takes "
                                "slightly longer than pure SARIMA due to the additional regressor fitting."
                            )

                        # ── Forecast horizon slider ──────────────────────────
                        _horizon_default = min(48, len(series) // 5)
                        _max_horizon = min(288 if granularity_arg == '5-min' else 24 * 7, len(series) // 4)
                        forecast_steps = st.slider(
                            "Forecast Horizon (steps)",
                            min_value=6,
                            max_value=max(_max_horizon, 12),
                            value=_cached_p.get('steps', _horizon_default),
                            key=f"sarima_horizon_{selected_vm}",
                            help="Number of future steps to forecast after the end of the series."
                        )

                        # ── Persistent banner: last optimization result ──────
                        _last_sum = _cached_p.get('last_opt_summary')
                        _last_err = _cached_p.get('last_opt_error')
                        if _last_sum:
                            st.success(
                                f"🎯 **Auto-Optimize found:** {_last_sum}"
                            )
                        if _last_err:
                            st.error(f"❌ Auto-Optimize failed: {_last_err}")

                        st.markdown("**Manual Parameter Tuning**")

                        # ── Aperiodic VMs get ARIMA-only mode (no seasonal block) ──
                        _aperiodic_arima = (effective_seasonality <= 1)

                        if _aperiodic_arima:
                            # Full-width pdq controls; P/D/Q/s zeroed out silently
                            st.info(
                                "ℹ️ **Aperiodic Mode — ARIMA only** (P, D, Q, s hidden). "
                                "No seasonal cycle was detected (`s = 1`), so the seasonal "
                                "order is set to **(0, 0, 0, 0)** automatically. "
                                "Use the sidebar **Manual Override** to re-enable seasonal terms "
                                "if you believe a cycle is present."
                            )
                            st.markdown("*Non-seasonal (p, d, q)*")
                            _c1, _c2, _c3 = st.columns(3)
                            _p = _c1.number_input(
                                "p", 0, 5,
                                st.session_state.sarima_params.get(selected_vm, {}).get('p', 1),
                                key=f"sp_{selected_vm}",
                            )
                            _d = _c2.number_input(
                                "d", 0, 2,
                                st.session_state.sarima_params.get(selected_vm, {}).get('d', _d_default),
                                key=f"sd_{selected_vm}",
                            )
                            _q = _c3.number_input(
                                "q", 0, 5,
                                st.session_state.sarima_params.get(selected_vm, {}).get('q', 1),
                                key=f"sq_{selected_vm}",
                            )
                            # Hard-zero seasonal components for aperiodic execution
                            _P, _D, _Q, _s = 0, 0, 0, 0

                        else:
                            # ── Periodic: show full SARIMA p,d,q | P,D,Q,s grid ──
                            _col_pdq, _col_PDQs = st.columns(2)

                            with _col_pdq:
                                st.markdown("*Non-seasonal (p, d, q)*")
                                _c1, _c2, _c3 = st.columns(3)
                                _p = _c1.number_input(
                                    "p", 0, 5,
                                    st.session_state.sarima_params.get(selected_vm, {}).get('p', 1),
                                    key=f"sp_{selected_vm}",
                                )
                                _d = _c2.number_input(
                                    "d", 0, 2,
                                    st.session_state.sarima_params.get(selected_vm, {}).get('d', _d_default),
                                    key=f"sd_{selected_vm}",
                                )
                                _q = _c3.number_input(
                                    "q", 0, 5,
                                    st.session_state.sarima_params.get(selected_vm, {}).get('q', 1),
                                    key=f"sq_{selected_vm}",
                                )

                            with _col_PDQs:
                                st.markdown("*Seasonal (P, D, Q, s)*")
                                _c4, _c5, _c6, _c7 = st.columns(4)
                                _P = _c4.number_input(
                                    "P", 0, 2,
                                    st.session_state.sarima_params.get(selected_vm, {}).get('P', 1),
                                    key=f"sP_{selected_vm}",
                                )
                                _D = _c5.number_input(
                                    "D", 0, 1,
                                    st.session_state.sarima_params.get(selected_vm, {}).get('D', 0),
                                    key=f"sD_{selected_vm}",
                                )
                                _Q = _c6.number_input(
                                    "Q", 0, 2,
                                    st.session_state.sarima_params.get(selected_vm, {}).get('Q', 1),
                                    key=f"sQ_{selected_vm}",
                                )
                                _s = _c7.number_input(
                                    "s", 0, 500,
                                    st.session_state.sarima_params.get(selected_vm, {}).get('s', effective_seasonality),
                                    key=f"ss_{selected_vm}",
                                    help="Seasonal period. Defaults to detected seasonality.",
                                )

                        st.markdown("")
                        # ── Box-Cox toggle ────────────────────────────────────────
                        _sarima_boxcox = st.checkbox(
                            "🔬 Enable Box-Cox Transformation",
                            value=False,
                            key=f"sarima_boxcox_{selected_vm}",
                            help=(
                                "Applies a Box-Cox variance-stabilisation transform before fitting. "
                                "Helps when residuals show multiplicative heteroscedasticity. "
                                "Forecasts and metrics are automatically back-transformed to original CPU % scale."
                            ),
                        )
                        # ── Buttons: side-by-side; all output renders full-width ──
                        _btn_col1, _btn_col2 = st.columns([1, 1])
                        _do_auto_opt = _btn_col1.button(
                            "✨ Auto-Optimize Parameters",
                            key=f"auto_opt_{selected_vm}",
                            help="Run stepwise AIC search to find the best SARIMA order.",
                        )
                        _do_execute = _btn_col2.button(
                            "🚀 Execute Forecast",
                            type="primary",
                            key=f"exec_sarima_{selected_vm}",
                        )

                        # ── Auto-Optimize: real-time streaming + double persistence ──
                        # Output is full-width (declared outside columns above).
                        if _do_auto_opt:
                            _log_lines: list = []
                            _log_q: queue.Queue = queue.Queue()
                            _result_holder: dict = {}

                            def _arima_worker():
                                # Capture _use_exog from closure so the thread
                                # picks the right mode at click-time.
                                _worker_exog = exog_df if _use_exog else None
                                try:
                                    res = stream_auto_arima(
                                        series, effective_seasonality, _log_q
                                    )
                                    _result_holder['data'] = res
                                except Exception as exc:
                                    _result_holder['error'] = exc
                                    _log_q.put(None)

                            _t = threading.Thread(target=_arima_worker, daemon=True)
                            _t.start()

                            with st.status(
                                f"🔍 Rigorous Optimization — "
                                f"{'ARIMAX/SARIMAX' if _use_exog else 'ARIMA/SARIMA'} — "
                                "MLE, max p/q=4…",
                                expanded=True,
                            ) as _status:
                                st.write(
                                    f"Stepwise AIC search · `s={effective_seasonality}` fixed · "
                                    f"max p/q=**4**, P/Q=**2** · full MLE (`approximation=False`) · "
                                    + (
                                        f"**ARIMAX** mode: exog `max_cpu(t−1)`, `min_cpu(t−1)` active · "
                                        if _use_exog else ""
                                    )
                                    + "output streams in real-time:"
                                )
                                _log_box = st.empty()

                                # ── Live streaming loop ──────────────────────
                                while True:
                                    line = _log_q.get()
                                    if line is None:
                                        break
                                    _log_lines.append(line)
                                    _log_box.code(
                                        "\n".join(_log_lines[-20:]),
                                        language=None,
                                    )

                                _t.join()

                                # ── Handle result ────────────────────────────
                                if 'error' in _result_holder:
                                    _status.update(
                                        label="❌ Auto-Optimize failed",
                                        state="error", expanded=True,
                                    )
                                    st.error(str(_result_holder['error']))
                                    # Persist error into params banner
                                    _ep = st.session_state.sarima_params.get(selected_vm, {})
                                    _ep['last_opt_error']   = str(_result_holder['error'])
                                    _ep['last_opt_summary'] = None
                                    st.session_state.sarima_params[selected_vm] = _ep
                                else:
                                    opt = _result_holder['data']
                                    op, od, oq = opt['order']
                                    oP, oD, oQ, os = opt['seasonal_order']
                                    _full_log = "\n".join(_log_lines)

                                    # ── PERSISTENCE A: save full log to dedicated dict ─
                                    # Use sarima_logs (not sarima_results) so the chart
                                    # function never sees a partial dict without
                                    # 'val_predictions' and crashes with a KeyError.
                                    st.session_state.sarima_logs[selected_vm] = _full_log

                                    # ── PERSISTENCE B: save params + summary ──────
                                    # Delete widget keys so value= re-initialises
                                    # from sarima_params on the NEXT rerun.
                                    for _wk in [
                                        f"sp_{selected_vm}", f"sd_{selected_vm}",
                                        f"sq_{selected_vm}", f"sP_{selected_vm}",
                                        f"sD_{selected_vm}", f"sQ_{selected_vm}",
                                        f"ss_{selected_vm}",
                                    ]:
                                        st.session_state.pop(_wk, None)
                                    _curr_use_exog = _cached_p.get('use_exog', False)
                                    _curr_s        = int(os)
                                    _opt_has_seas  = _curr_s > 1
                                    if _curr_use_exog:
                                        _opt_label = 'SARIMAX' if _opt_has_seas else 'ARIMAX'
                                    else:
                                        _opt_label = 'SARIMA'  if _opt_has_seas else 'ARIMA'

                                    st.session_state.sarima_params[selected_vm] = {
                                        'p': int(op), 'd': int(od), 'q': int(oq),
                                        'P': int(oP), 'D': int(oD), 'Q': int(oQ),
                                        's': int(os), 'steps': forecast_steps,
                                        'use_exog':    _curr_use_exog,
                                        'model_label': _opt_label,
                                        'last_opt_summary': (
                                            f"{_opt_label}({op},{od},{oq})"
                                            f"({oP},{oD},{oQ},{os})"
                                        ),
                                        'last_opt_error': None,
                                    }

                                    _status.update(
                                        label=(
                                            f"✅ Best model found: "
                                            f"{_opt_label}({op},{od},{oq})"
                                            f"({oP},{oD},{oQ},{os})"
                                        ),
                                        state="complete",
                                        expanded=True,
                                    )
                                    st.success(
                                        f"✅ **Optimization Complete. Parameters Injected.** "
                                        f"p={op}, d={od}, q={oq} | "
                                        f"P={oP}, D={oD}, Q={oQ}, s={os}  "
                                        f"→ Refreshing inputs…"
                                    )
                                    # Rerun NOW: logs are in sarima_logs so the
                                    # expander renders them, and widget keys are
                                    # deleted so number_inputs refresh from sarima_params.
                                    st.rerun()

                        # ── Execute Forecast output — full-width ──────────────
                        if _do_execute:
                            _order          = (int(_p), int(_d), int(_q))
                            _seasonal_order = (int(_P), int(_D), int(_Q), int(_s))
                            _exog_input     = exog_df if _use_exog else None
                            # 4-way model label: exog × seasonal
                            _has_seasonal   = int(_s) > 1
                            if _use_exog:
                                _model_label = 'SARIMAX' if _has_seasonal else 'ARIMAX'
                            else:
                                _model_label = 'SARIMA'  if _has_seasonal else 'ARIMA'
                            with st.spinner(
                                f"Fitting {_model_label}… running validation + final fit"
                            ):
                                try:
                                    result = run_sarima_forecast(
                                        series,
                                        order=_order,
                                        seasonal_order=_seasonal_order,
                                        forecast_steps=int(forecast_steps),
                                        exog_df=_exog_input,
                                        use_boxcox=_sarima_boxcox,
                                    )
                                    st.session_state.sarima_results[selected_vm] = result
                                    _prev_p = st.session_state.sarima_params.get(selected_vm, {})
                                    st.session_state.sarima_params[selected_vm] = {
                                        **_prev_p,
                                        'p': int(_p), 'd': int(_d), 'q': int(_q),
                                        'P': int(_P), 'D': int(_D), 'Q': int(_Q),
                                        's': int(_s), 'steps': int(forecast_steps),
                                        'use_exog':   _use_exog,
                                        'model_label': _model_label,
                                    }
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"{_model_label} failed: {e}")


                    # ── Active forecast summary outside expander ─────────────
                    _cur_sarima = st.session_state.sarima_results.get(selected_vm)
                    if _cur_sarima:
                        _cp  = st.session_state.sarima_params.get(selected_vm, {})
                        _o   = (_cp.get('p',1), _cp.get('d',0), _cp.get('q',1))
                        _so  = (_cp.get('P',0), _cp.get('D',0), _cp.get('Q',0), _cp.get('s',0))
                        _vm  = _cur_sarima.get('val_metrics', {})
                        _aic = _cur_sarima.get('aic', float('nan'))
                        # Use model_name from result if available ('ARIMAX' vs 'SARIMA')
                        _mn  = _cur_sarima.get('model_name', 'SARIMA')
                        _bc_lam = _cur_sarima.get('boxcox_lambda')
                        _bc_str = (
                            f" · Box-Cox λ: **{_bc_lam:.3f}**"
                            + (" *(≈ linear — minimal effect)*" if _bc_lam and abs(_bc_lam - 1.0) < 0.15 else "")
                            if _bc_lam is not None else ""
                        )
                        st.success(
                            f"✅ **{_mn}{_o}{_so}** active · "
                            f"AIC: **{_aic:.1f}** · "
                            f"Val RMSE: **{_vm.get('RMSE', float('nan')):.3f}** · "
                            f"Forecast: **{_cp.get('steps', '?')} steps** ahead"
                            + _bc_str
                        )
                        st.caption(
                            "Shaded area represents the **95 % confidence interval**. "
                            + (
                                "Exogenous regressors: `max_cpu(t−1)`, `min_cpu(t−1)`. "
                                "Future exog uses last-known values repeated across the horizon."
                                if _mn == 'ARIMAX' else
                                "The validation line (dotted) shows model accuracy on the hold-out window."
                            )
                        )
                        if st.button("🗑️ Clear SARIMA Forecast",
                                     key=f"clear_sarima_{selected_vm}"):
                            st.session_state.sarima_results.pop(selected_vm, None)
                            st.session_state.sarima_params.pop(selected_vm, None)
                            st.rerun()

                # ═══════════════════════════════════════════════════════════
                # SECTION 7 — Multivariate System (VAR)
                # ═══════════════════════════════════════════════════════════
                st.divider()
                VAR_COLOR_MAP = {
                    'avg_cpu': '#3B82F6',    # blue-500
                    'max_cpu': '#F97316',    # orange-500
                    'min_cpu': '#10B981',    # emerald-500
                }

                st.subheader("🚀 Advanced Multivariate Statistical Forecasting (VAR)")

                with st.expander("🚀 Advanced Multivariate Statistical Forecasting (VAR)", expanded=False):
                    st.caption(
                        "**Vector Autoregression (VAR)** treats `avg_cpu`, `max_cpu`, and `min_cpu` as a "
                        "jointly-evolving system. Each variable is modelled as a linear function of *all* "
                        "variables' past values. This captures cross-metric dynamics that no univariate "
                        "model can detect — e.g., whether a spike in `max_cpu` at t−1 predicts "
                        "`avg_cpu` at t. Lag order **p** is selected automatically by AIC."
                    )

                    # ── Stationarity info box ─────────────────────────────────────
                    st.info(
                        "🔬 **Auto-Stationarity:** Before fitting, an iterative ADF loop tests d ∈ {0, 1, 2}. "
                        "The minimum d where all three series are stationary is chosen; "
                        "forecasts are re-integrated back to original levels automatically."
                    )

                    # ── Box-Cox toggle ────────────────────────────────────────────
                    _var_boxcox = st.checkbox(
                        "🔬 Enable Box-Cox Transformation",
                        value=False,
                        key=f"var_boxcox_{selected_vm}",
                        help=(
                            "Applies per-column Box-Cox transforms (individual λ per variable) "
                            "before fitting to stabilise variance. "
                            "Applied BEFORE differencing; inverse-transformed AFTER re-integration. "
                            "Forecasts and metrics are returned on the original CPU % scale."
                        ),
                    )

                    # ── Horizon slider + execute button ───────────────────────────
                    _var_horizon_max = min(288 if granularity_arg == '5-min' else 24 * 7, len(series) // 4)
                    _var_steps = st.slider(
                        "Forecast Horizon (steps)", min_value=6,
                        max_value=max(_var_horizon_max, 12),
                        value=min(48, len(series) // 5),
                        key=f"var_horizon_{selected_vm}",
                    )

                    _do_var = st.button(
                        "⚡ Execute System-wide VAR Forecast",
                        type="primary",
                        key=f"var_exec_{selected_vm}",
                    )

                    if _do_var:
                        with st.spinner("Fitting VAR: ADF pre-check → lag selection → validation → final fit…"):
                            try:
                                _var_res = run_var_forecast(
                                    vm_df,
                                    forecast_steps=int(_var_steps),
                                    use_boxcox=_var_boxcox,
                                )
                                st.session_state.var_results[selected_vm] = _var_res
                                st.rerun()
                            except Exception as _ve:
                                st.error(f"VAR failed: {_ve}")

                # ── VAR results — rendered outside the expander widget ────────────
                # (so the charts are always visible when a result is cached)
                _cur_var = st.session_state.var_results.get(selected_vm)
                if _cur_var:
                    _var_lag   = _cur_var['lag_order']
                    _var_diff  = _cur_var['diff_order']
                    _var_aic   = _cur_var['aic']
                    _var_adfs  = _cur_var['adf_pvalues']

                    # ── Status banner — Model Identity style (mirrors SARIMA banner) ──────
                    _var_bc_lams = _cur_var.get('boxcox_lambdas')
                    _var_bc_str  = (
                        f" · Box-Cox λ: **{_var_bc_lams['avg_cpu']:.2f}**"
                        f"/**{_var_bc_lams['max_cpu']:.2f}**"
                        f"/**{_var_bc_lams['min_cpu']:.2f}**"
                        if _var_bc_lams else " · Box-Cox λ: —"
                    )
                    st.success(
                        f"✅ **VAR(p={_var_lag})** active · "
                        f"System stationary at d=**{_var_diff}** · "
                        + _var_bc_str
                        + f" · AIC: **{_var_aic:.2f}**"
                    )

                    # ── VAR Master Chart ────────────────────────────────────────────
                    st.subheader("📈 VAR System Forecast")
                    st.caption(
                        "Historical (`avg_cpu` shown as reference) + simultaneous 3-variable forecast. "
                        "Bold lines = forecast horizon; thin lines = historical. "
                        "Toggle any variable via the legend."
                    )
                    _fig_var = go.Figure()

                    # Historical avg_cpu as anchor reference
                    _fig_var.add_trace(go.Scatter(
                        x=vm_df['timestamp_dt'], y=vm_df['avg_cpu'],
                        name='avg_cpu (hist)', mode='lines',
                        line=dict(color=VAR_COLOR_MAP['avg_cpu'], width=1, dash='dot'),
                        opacity=0.45,
                        hovertemplate='%{x|%b %d %H:%M}<br>avg_cpu: %{y:.1f}%<extra>Historical</extra>'
                    ))

                    # Validation window overlays (hold-out actual vs predicted)
                    for _vcol, _vc_color in VAR_COLOR_MAP.items():
                        _vp_var = _cur_var['val_predictions'].get(_vcol, pd.Series(dtype=float))
                        if not _vp_var.empty:
                            _fig_var.add_trace(go.Scatter(
                                x=_vp_var.index, y=_vp_var.values,
                                name=f'{_vcol} (val)',
                                mode='lines',
                                line=dict(color=_vc_color, width=1.5, dash='dashdot'),
                                visible='legendonly',
                                hovertemplate=f'%{{x|%b %d %H:%M}}<br>{_vcol}: %{{y:.1f}}%<extra>{_vcol} val</extra>'
                            ))

                    # Future forecasts — 3 bold lines
                    for _vcol, _vc_color in VAR_COLOR_MAP.items():
                        _fc_s = _cur_var['forecasts'][_vcol]
                        _fig_var.add_trace(go.Scatter(
                            x=_fc_s.index, y=_fc_s.values,
                            name=f'{_vcol} (forecast)',
                            mode='lines',
                            line=dict(color=_vc_color, width=2.5),
                            visible=True,
                            hovertemplate=f'%{{x|%b %d %H:%M}}<br>{_vcol}: %{{y:.1f}}%<extra>{_vcol} fc</extra>'
                        ))

                    _var_y_all = np.concatenate([
                        vm_df['avg_cpu'].dropna().values,
                        *[_cur_var['forecasts'][c].values for c in VAR_COLOR_MAP]
                    ])
                    _var_y_range, _ = _y_range_robust(pd.Series(_var_y_all))
                    _fig_var.update_layout(
                        title=dict(text=f"VAR({_var_lag}) System Forecast — {selected_vm[:32]}", font=dict(size=14)),
                        template='plotly_white',
                        height=420,
                        margin=dict(l=10, r=10, t=50, b=10),
                        legend=dict(orientation='h', yanchor='bottom', y=1.01, xanchor='right', x=1),
                        yaxis=dict(title='CPU (%)', range=_var_y_range, fixedrange=False),
                        xaxis=dict(fixedrange=False),
                        hovermode='x unified',
                    )
                    st.plotly_chart(_fig_var, width='stretch')

                    # ── Per-variable metric row (RMSE / MAE) ────────────────────────────
                    st.markdown("**Validation Metrics (80/20 hold-out)**")
                    _mc1, _mc2, _mc3 = st.columns(3)
                    for _mc_col, _mc_container, _mc_emoji, _mc_color_label in [
                        ('avg_cpu', _mc1, '🔵', 'Avg CPU'),
                        ('max_cpu', _mc2, '🟠', 'Max CPU'),
                        ('min_cpu', _mc3, '🟢', 'Min CPU'),
                    ]:
                        _mc_m = _cur_var['val_metrics'][_mc_col]
                        _mc_rmse = _mc_m.get('RMSE', float('nan'))
                        _mc_mae  = _mc_m.get('MAE',  float('nan'))
                        _mc_mape = _mc_m.get('MAPE', float('nan'))
                        with _mc_container:
                            st.metric(
                                label=f"{_mc_emoji} {_mc_color_label} — RMSE",
                                value=f"{_mc_rmse:.3f}",
                                help=f"MAE: {_mc_mae:.3f} · MAPE: {_mc_mape:.1f}%",
                            )

                    # ── ADF pre-check table ──────────────────────────────────────────
                    st.caption(
                        f"**ADF pre-check** — iterative loop tested d ∈ {{0, 1, 2}}; "
                        f"chose d = **{_var_diff}** (minimum d where all series are stationary, p < 0.05). "
                        "P-values shown below are from the **final chosen d**."
                    )
                    _adf_rows = []
                    for _col, _pv in _var_adfs.items():
                        _stat = '✅ Stationary' if _pv < 0.05 else '⚠️ Non-Stationary'
                        _adf_rows.append({'Variable': _col, 'ADF p-value': f'{_pv:.4f}', 'Status': _stat})
                    st.dataframe(pd.DataFrame(_adf_rows), hide_index=True, width='stretch')

                    # ── Granger Causality toggle + matrix ────────────────────────────
                    st.divider()
                    _show_granger = st.toggle(
                        "🔍 Show Granger Causality Matrix",
                        value=False,
                        key=f"granger_toggle_{selected_vm}",
                        help=(
                            "F-test for Granger causality: does knowing variable X at lags 1…p "
                            "improve the prediction of Y beyond Y's own history? "
                            "p < 0.05 → significant lead-lag relationship."
                        ),
                    )

                    if _show_granger:
                        st.subheader("🔗 Granger Causality Matrix")
                        st.caption(
                            f"F-test at α = 0.05 · VAR({_var_lag}) fitted on "
                            + ("first-differenced" if _var_diff == 1 else "level") +
                            " data. **Significant** = p < 0.05 → the cause variable's "
                            "past values provide statistically useful information for "
                            "predicting the effect variable."
                        )

                        _gc = _cur_var['granger']
                        _gc_rows = []
                        for _pair_key, _gc_val in _gc.items():
                            _cause, _effect = _pair_key.split('→')
                            _gc_p = _gc_val['p_value']
                            _gc_sig = _gc_val['significant']
                            _gc_rows.append({
                                'Cause (X)':   _cause,
                                'Effect (Y)':  _effect,
                                'F-test p':    f"{_gc_p:.4f}" if np.isfinite(_gc_p) else 'N/A',
                                'Significant': '✅ Yes' if _gc_sig else '❌ No',
                                'Interpretation': (
                                    f"`{_cause}` lags Granger-cause `{_effect}`"
                                    if _gc_sig else
                                    f"No evidence `{_cause}` leads `{_effect}`"
                                ),
                            })

                        _gc_df = pd.DataFrame(_gc_rows)

                        def _style_granger(row):
                            if row['Significant'] == '✅ Yes':
                                return ['background-color: #F0FDF4; color: #15803D'] * len(row)
                            return [''] * len(row)

                        _gc_styled = (
                            _gc_df.style
                            .apply(_style_granger, axis=1)
                            .hide(axis='index')
                        )
                        st.dataframe(_gc_styled, width='stretch')

                    # ── Clear VAR Forecast — column-aligned (mirrors SARIMA/HW pattern) ──
                    st.divider()
                    _var_clear_col, _ = st.columns([1, 3])
                    if _var_clear_col.button(
                        "🗑️ Clear VAR Forecast",
                        key=f"clear_var_{selected_vm}",
                    ):
                        st.session_state.var_results.pop(selected_vm, None)
                        st.rerun()

                # ═══════════════════════════════════════════════════════════
                # SECTION 8 — Unified Residual Diagnostics
                # ═══════════════════════════════════════════════════════════
                _cur_hw_rd     = st.session_state.hw_results.get(selected_vm)
                _cur_sarima_rd = st.session_state.sarima_results.get(selected_vm)
                _cur_var_rd    = st.session_state.var_results.get(selected_vm)

                if _cur_hw_rd or _cur_sarima_rd or _cur_var_rd:
                    st.divider()
                    st.subheader("🧪 Residual Diagnostics")
                    st.caption(
                        "Unified post-model residual analysis across all active models. "
                        "**Univariate (HW / SARIMA / ARIMAX):** source = out-of-sample hold-out errors "
                        "(actual − ŷ on the 20 % test window) — a strict, honest measure of adequacy. "
                        "**VAR (Multivariate):** source = in-sample fit residuals from the VAR equation "
                        "system — select a variable sub-tab to inspect each equation independently. "
                        "A well-specified model leaves residuals as white noise: "
                        "no autocorrelation, near-zero mean, approximately normal distribution."
                    )

                    # Build typed model list: (display_name, result_dict, model_class)
                    # model_class ∈ {'univariate', 'var'}
                    # Session-state presence is the sole dispatch mechanism —
                    # clearing any forecast automatically removes it from this section.
                    _active_models = []
                    if _cur_hw_rd:
                        _active_models.append(('HW', _cur_hw_rd, 'univariate'))
                    if _cur_sarima_rd:
                        _s_tab_name = _cur_sarima_rd.get('model_name', 'SARIMA')
                        _active_models.append((_s_tab_name, _cur_sarima_rd, 'univariate'))
                    if _cur_var_rd:
                        _var_p_label = _cur_var_rd.get('lag_order', '?')
                        _active_models.append((f'VAR({_var_p_label})', _cur_var_rd, 'var'))

                    _rd_tabs = st.tabs([f"📊 {name}" for name, _, _ in _active_models])


                    for _rd_tab, (_model_name, _model_res, _model_class) in zip(_rd_tabs, _active_models):
                        with _rd_tab:

                            # ══════════════════════════════════════════════
                            # VAR path — in-sample equation residuals
                            # ══════════════════════════════════════════════
                            if _model_class == 'var':
                                from statsmodels.stats.diagnostic import acorr_ljungbox as _lb_fn_var
                                _vd_resid_df   = _model_res['var_residuals']
                                _vd_lag        = _model_res.get('lag_order', '?')
                                _vd_diff       = _model_res.get('diff_order', 0)
                                st.info(
                                    f"🔎 **VAR({_vd_lag}) in-sample equation residuals** "
                                    + ("— fitted on **first-differences (Δ)**, displayed in diff-domain. "
                                       if _vd_diff == 1 else "— fitted in **levels**. ")
                                    + "Select a variable equation below."
                                )

                                # ── Variable sub-tabs ──────────────────────
                                _vd_tabs = st.tabs([
                                    f"{'🔵' if c == 'avg_cpu' else '🟠' if c == 'max_cpu' else '🟢'} {c}"
                                    for c in VAR_COLOR_MAP
                                ])
                                for _vd_tab, (_vd_col, _vd_color) in zip(_vd_tabs, VAR_COLOR_MAP.items()):
                                    with _vd_tab:
                                        _vd_resid = _vd_resid_df[_vd_col].dropna()
                                        if len(_vd_resid) < 4:
                                            st.info("Not enough residual observations.")
                                            continue

                                        # LB slider
                                        _vd_lb_max = max(5, min(len(_vd_resid) // 5, 50))
                                        _vd_lb_def = min(10, _vd_lb_max)
                                        _vd_lb_sel = st.slider(
                                            "Ljung-Box Test — Number of Lags",
                                            min_value=1, max_value=_vd_lb_max,
                                            value=_vd_lb_def,
                                            key=f"lb_lags_{selected_vm}_{_model_name}_{_vd_col}",
                                            help=(
                                                f"Tests residual autocorrelation for the `{_vd_col}` "
                                                "equation. p < 0.05 → consider a higher VAR lag order."
                                            ),
                                        )
                                        _vd_lb_res = _lb_fn_var(_vd_resid, lags=[_vd_lb_sel], return_df=True)
                                        _vd_lb_p   = float(_vd_lb_res['lb_pvalue'].iloc[0])
                                        _vd_p_str  = f"{_vd_lb_p:.4f}" if _vd_lb_p >= 0.0001 else f"{_vd_lb_p:.2e}"

                                        if _vd_lb_p > 0.05:
                                            st.success(
                                                f"✅ **`{_vd_col}` equation residuals are white noise** — "
                                                f"Ljung-Box p = {_vd_p_str} > 0.05 (at lag **{_vd_lb_sel}**). "
                                                "VAR equation is well-specified for this variable."
                                            )
                                        else:
                                            st.warning(
                                                f"⚠️ **`{_vd_col}` residuals NOT white noise** — "
                                                f"Ljung-Box p = {_vd_p_str} < 0.05 (at lag **{_vd_lb_sel}**). "
                                                "Consider a higher lag order `p` for the VAR system."
                                            )

                                        # ── 3-panel chart ──────────────────
                                        _vd_c1, _vd_c2, _vd_c3 = st.columns(3)

                                        with _vd_c1:
                                            _fig_vd = go.Figure(go.Scatter(
                                                y=_vd_resid.values, mode='lines',
                                                line=dict(color=_vd_color, width=1),
                                                name='Residual',
                                            ))
                                            _fig_vd.add_hline(y=0, line_dash='dash',
                                                              line_color='#EF4444', opacity=0.6)
                                            _fig_vd.update_layout(
                                                title=f'{_vd_col} — Residuals vs Time',
                                                template='plotly_white', height=280,
                                                margin=dict(l=5, r=5, t=36, b=5),
                                                yaxis=dict(fixedrange=True),
                                                xaxis=dict(fixedrange=True, title='Fitted step'),
                                                showlegend=False,
                                            )
                                            st.plotly_chart(_fig_vd, width='stretch')

                                        with _vd_c2:
                                            _vd_nlags  = min(40, max(len(_vd_resid) // 2 - 1, 4))
                                            _vd_acf    = sm_acf(_vd_resid, nlags=_vd_nlags)
                                            _vd_colors = ['#EF4444' if abs(v) > 0.2 else '#94A3B8'
                                                          for v in _vd_acf]
                                            _fig_vd_acf = go.Figure(go.Bar(
                                                x=list(range(len(_vd_acf))), y=_vd_acf,
                                                marker_color=_vd_colors, name='Residual ACF',
                                            ))
                                            _fig_vd_acf.add_vrect(
                                                x0=0.5, x1=min(_vd_lb_sel, _vd_nlags) + 0.5,
                                                fillcolor='rgba(99,102,241,0.08)',
                                                layer='below', line_width=0,
                                                annotation_text=f"LB window (lag {_vd_lb_sel})",
                                                annotation_position='top left',
                                                annotation=dict(font=dict(size=9, color='#6366F1'),
                                                                showarrow=False),
                                            )
                                            _fig_vd_acf.add_vline(
                                                x=min(_vd_lb_sel, _vd_nlags) + 0.5,
                                                line_dash='dot', line_color='#6366F1',
                                                line_width=1.5, opacity=0.7,
                                            )
                                            _fig_vd_acf.add_hline(y=0.2,  line_dash='dash',
                                                                   line_color='#EF4444', opacity=0.5)
                                            _fig_vd_acf.add_hline(y=-0.2, line_dash='dash',
                                                                   line_color='#EF4444', opacity=0.5)
                                            _fig_vd_acf.update_layout(
                                                title=f'{_vd_col} — Residual ACF (LB up to lag {_vd_lb_sel})',
                                                template='plotly_white', height=280,
                                                margin=dict(l=5, r=5, t=36, b=5),
                                                xaxis=dict(title='Lag', fixedrange=True),
                                                yaxis=dict(title='ACF', fixedrange=True),
                                                showlegend=False,
                                            )
                                            st.plotly_chart(_fig_vd_acf, width='stretch')

                                        with _vd_c3:
                                            _vd_vals = _vd_resid.values
                                            _vd_std  = float(np.std(_vd_vals))
                                            _vd_bw   = 1.06 * _vd_std * len(_vd_vals) ** (-0.2) if _vd_std > 0 else 1.0
                                            _vd_grid = np.linspace(_vd_vals.min(), _vd_vals.max(), 200)
                                            _vd_kde  = np.sum(
                                                np.exp(-0.5 * ((_vd_grid[:, None] - _vd_vals[None, :]) / _vd_bw) ** 2),
                                                axis=1,
                                            ) / (len(_vd_vals) * _vd_bw * np.sqrt(2 * np.pi))
                                            _vd_kde_sc = _vd_kde * len(_vd_vals) * (
                                                (_vd_vals.max() - _vd_vals.min()) / 20
                                            )
                                            _fig_vd_hist = go.Figure()
                                            _fig_vd_hist.add_trace(go.Histogram(
                                                x=_vd_vals, nbinsx=20,
                                                marker_color=_vd_color, opacity=0.7, name='Count',
                                            ))
                                            _fig_vd_hist.add_trace(go.Scatter(
                                                x=_vd_grid, y=_vd_kde_sc,
                                                mode='lines', line=dict(color='#EF4444', width=2),
                                                name='KDE',
                                            ))
                                            _fig_vd_hist.update_layout(
                                                title=f'{_vd_col} — Residual Distribution',
                                                template='plotly_white', height=280,
                                                margin=dict(l=5, r=5, t=36, b=5),
                                                xaxis=dict(title='Residual', fixedrange=True),
                                                yaxis=dict(title='Count', fixedrange=True),
                                                showlegend=False, barmode='overlay',
                                            )
                                            st.plotly_chart(_fig_vd_hist, width='stretch')

                            # ══════════════════════════════════════════════
                            # Univariate path (HW / SARIMA / ARIMA / ARIMAX)
                            # Source: hold-out OOS errors (actual − ŷ)
                            # ══════════════════════════════════════════════
                            else:
                                _vp_rd  = _model_res.get('val_predictions', pd.Series(dtype=float))
                                _vr_raw = _model_res.get('val_residuals',   None)

                                if _vr_raw is not None and not _vr_raw.empty:
                                    _resid = _vr_raw.dropna()
                                elif not _vp_rd.empty:
                                    _test_start = _vp_rd.index[0]
                                    _actuals_rd = series.loc[_test_start:_vp_rd.index[-1]]
                                    _resid = (_actuals_rd - _vp_rd).dropna()
                                else:
                                    st.info("Run the model to compute residuals.")
                                    continue

                                if len(_resid) < 4:
                                    st.info("Not enough hold-out data to compute residuals.")
                                    continue

                                # LB slider
                                _lb_max = max(5, min(len(_resid) // 5, 50))
                                _lb_def = (
                                    min(2 * effective_seasonality, _lb_max)
                                    if effective_seasonality > 1
                                    else min(10, _lb_max)
                                )
                                _lb_sel = st.slider(
                                    "Ljung-Box Test — Number of Lags",
                                    min_value=1, max_value=_lb_max, value=_lb_def,
                                    key=f"lb_lags_{selected_vm}_{_model_name}",
                                    help=(
                                        "Tests whether residual autocorrelations up to the chosen lag "
                                        "are jointly zero (white noise). "
                                        f"Default = **{_lb_def}** "
                                        f"({'2 × s = ' + str(2 * effective_seasonality) if effective_seasonality > 1 else 'aperiodic default'}). "
                                        "Higher lags catch long-range structure but reduce statistical power."
                                    ),
                                )

                                from statsmodels.stats.diagnostic import acorr_ljungbox
                                _lb_res = acorr_ljungbox(_resid, lags=[_lb_sel], return_df=True)
                                _lb_p   = float(_lb_res['lb_pvalue'].iloc[0])
                                _p_str  = f"{_lb_p:.4f}" if _lb_p >= 0.0001 else f"{_lb_p:.2e}"

                                st.info(
                                    f"🔎 **Currently analyzing residuals for: `{_model_name}`** — "
                                    f"{len(_resid)} out-of-sample observations (actual − predicted)."
                                )

                                if _lb_p > 0.05:
                                    st.success(
                                        f"✅ **Residuals are white noise** — "
                                        f"Ljung-Box p = {_p_str} > 0.05 (at lag **{_lb_sel}**). "
                                        f"No autocorrelation structure remains in `{_model_name}` residuals. "
                                        "Model is well-specified."
                                    )
                                else:
                                    _lb_note = (
                                        f" For high-seasonality VMs (s = {effective_seasonality}), "
                                        "p ≈ 0 is expected — even a well-fitted SARIMA rarely captures "
                                        "all seasonal harmonics in a short hold-out window. "
                                        "Try reducing the lag count to focus on short-range structure."
                                        if _lb_p < 0.001 else ""
                                    )
                                    st.warning(
                                        f"⚠️ **Residuals are NOT white noise** — "
                                        f"Ljung-Box p = {_p_str} < 0.05 (at lag **{_lb_sel}**). "
                                        f"The `{_model_name}` model has not captured all structure. "
                                        f"Consider adjusting parameters.{_lb_note}"
                                    )

                                # ── 3-panel chart ──────────────────────────
                                _rd_c1, _rd_c2, _rd_c3 = st.columns(3)

                                with _rd_c1:
                                    _fig_r = go.Figure(go.Scatter(
                                        y=_resid.values, mode='lines',
                                        line=dict(color='#64748B', width=1),
                                        name='Residual',
                                    ))
                                    _fig_r.add_hline(y=0, line_dash='dash',
                                                     line_color='#EF4444', opacity=0.6)
                                    _fig_r.update_layout(
                                        title='Residuals vs Time',
                                        template='plotly_white', height=280,
                                        margin=dict(l=5, r=5, t=36, b=5),
                                        yaxis=dict(fixedrange=True),
                                        xaxis=dict(fixedrange=True, title='Hold-out step'),
                                        showlegend=False,
                                    )
                                    st.plotly_chart(_fig_r, width='stretch')

                                with _rd_c2:
                                    _r_nlags  = min(40, max(len(_resid) // 2 - 1, 4))
                                    _r_acf    = sm_acf(_resid, nlags=_r_nlags)
                                    _r_lags   = list(range(len(_r_acf)))
                                    _r_colors = ['#EF4444' if abs(v) > 0.2 else '#94A3B8'
                                                 for v in _r_acf]
                                    _fig_acf  = go.Figure(go.Bar(
                                        x=_r_lags, y=_r_acf,
                                        marker_color=_r_colors, name='Residual ACF',
                                    ))
                                    _fig_acf.add_vrect(
                                        x0=0.5, x1=min(_lb_sel, _r_nlags) + 0.5,
                                        fillcolor='rgba(99,102,241,0.08)',
                                        layer='below', line_width=0,
                                        annotation_text=f"LB window (lag {_lb_sel})",
                                        annotation_position='top left',
                                        annotation=dict(
                                            font=dict(size=9, color='#6366F1'),
                                            showarrow=False,
                                        ),
                                    )
                                    _fig_acf.add_vline(
                                        x=min(_lb_sel, _r_nlags) + 0.5,
                                        line_dash='dot', line_color='#6366F1',
                                        line_width=1.5, opacity=0.7,
                                    )
                                    _fig_acf.add_hline(y=0.2,  line_dash='dash',
                                                       line_color='#EF4444', opacity=0.5)
                                    _fig_acf.add_hline(y=-0.2, line_dash='dash',
                                                       line_color='#EF4444', opacity=0.5)
                                    _fig_acf.update_layout(
                                        title=f'Residual ACF (LB up to lag {_lb_sel})',
                                        template='plotly_white', height=280,
                                        margin=dict(l=5, r=5, t=36, b=5),
                                        xaxis=dict(title='Lag', fixedrange=True),
                                        yaxis=dict(title='ACF', fixedrange=True),
                                        showlegend=False,
                                    )
                                    st.plotly_chart(_fig_acf, width='stretch')

                                with _rd_c3:
                                    _r_vals = _resid.values
                                    _r_std  = float(np.std(_r_vals))
                                    _grid   = np.linspace(_r_vals.min(), _r_vals.max(), 200)
                                    _bw     = 1.06 * _r_std * len(_r_vals) ** (-0.2) if _r_std > 0 else 1.0
                                    _kde    = np.sum(
                                        np.exp(-0.5 * ((_grid[:, None] - _r_vals[None, :]) / _bw) ** 2),
                                        axis=1,
                                    ) / (len(_r_vals) * _bw * np.sqrt(2 * np.pi))
                                    _kde_scaled = _kde * len(_r_vals) * (
                                        (_r_vals.max() - _r_vals.min()) / 20
                                    )
                                    _fig_hist = go.Figure()
                                    _fig_hist.add_trace(go.Histogram(
                                        x=_r_vals, nbinsx=20,
                                        marker_color='#94A3B8', opacity=0.7, name='Count',
                                    ))
                                    _fig_hist.add_trace(go.Scatter(
                                        x=_grid, y=_kde_scaled,
                                        mode='lines', line=dict(color='#EF4444', width=2),
                                        name='KDE',
                                    ))
                                    _fig_hist.update_layout(
                                        title='Residual Distribution',
                                        template='plotly_white', height=280,
                                        margin=dict(l=5, r=5, t=36, b=5),
                                        xaxis=dict(title='Residual', fixedrange=True),
                                        yaxis=dict(title='Count', fixedrange=True),
                                        showlegend=False, barmode='overlay',
                                    )
                                    st.plotly_chart(_fig_hist, width='stretch')




            else:
                st.info("ℹ️ Run **Fleet Diagnostics** in the Fleet Analytics tab first to unlock the Deep Dive.")

else:
    st.info("⏳ Awaiting data — upload a CSV or allow the demo dataset to load.")

st.sidebar.divider()
st.sidebar.info("v2.3 · Phase 3.3.5 · Unified Residual Diagnostics")
