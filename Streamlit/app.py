
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import acf as sm_acf
from datetime import date, time as dtime
from concurrent.futures import ThreadPoolExecutor  # kept for future async tasks

from utils.data_engine import (
    load_data, apply_temporal_anchor, aggregate_and_interpolate,
    load_demo_cache, save_demo_cache, is_demo_data
)
from utils.diagnostics import (
    perform_adf_test, perform_ljung_box_test,
    calculate_refined_seasonality, classify_vm_suitability
)
from utils.baselines import expanding_window_cv, simple_holdout_evaluate
from utils.forecasting import run_sarima_forecast, auto_optimize_sarima, stream_auto_arima
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


SARIMA_COLOR = '#F43F5E'   # rose-500


def build_prediction_chart(vm_df, preds, vm_id, sarima_result=None):
    """Full-width line chart: raw CPU + baseline forecast overlay.

    If sarima_result is provided, overlays the SARIMA future forecast line
    and its 95 % CI shading ribbon.  The CI uses two invisible bound traces
    (showlegend=False) plus a single 'SARIMA (95 % CI)' legend entry.

    Y-axis uses p99-based robust scaling so occasional spikes don't
    flatten the visible trend in the default view.  Y-axis is NOT
    fixed-range so users can zoom out to reveal clipped spikes.
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

    # ── SARIMA validation-window overlay (hold-out period) ──────────────────
    _vp = sarima_result.get('val_predictions') if sarima_result else None
    if _vp is not None and not _vp.empty:
        fig.add_trace(go.Scatter(
            x=_vp.index, y=_vp.values,
            name='SARIMA (validation)',
            line=dict(color=SARIMA_COLOR, width=1.5, dash='dot'),
            visible=True,
            hovertemplate='%{x|%b %d %H:%M}<br>CPU: %{y:.1f}%<extra>SARIMA val</extra>'
        ))

    if not preds.empty:
        # Identify the best-performing baseline (lowest RMSE)
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

    # ── SARIMA future forecast + CI shading ─────────────────────────────────
    if sarima_result:
        fc_mean  = sarima_result['mean']
        fc_upper = sarima_result['upper']
        fc_lower = sarima_result['lower']

        # CI ribbon — upper bound trace (invisible, no legend entry)
        fig.add_trace(go.Scatter(
            x=pd.concat([pd.Series(fc_upper.index), pd.Series(fc_lower.index[::-1])]),
            y=np.concatenate([fc_upper.values, fc_lower.values[::-1]]),
            fill='toself',
            fillcolor='rgba(244,63,94,0.12)',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip',
            name='_ci_band',
        ))

        # SARIMA future mean line — one legend entry for the whole feature
        fig.add_trace(go.Scatter(
            x=fc_mean.index, y=fc_mean.values,
            name='SARIMA (95 % CI)',
            line=dict(color=SARIMA_COLOR, width=2.5, dash='solid'),
            visible=True,
            hovertemplate='%{x|%b %d %H:%M}<br>Forecast: %{y:.1f}%<extra>SARIMA</extra>'
        ))

        # Extend Y-axis ceiling to include CI upper if needed
        ci_max = float(fc_upper.max())
        if ci_max > y_range[1]:
            y_range = [0, ci_max * 1.05]

    title_suffix = " · SARIMA Active" if sarima_result else ""
    fig.update_layout(
        title=dict(
            text=f"CPU Usage & Forecasts — {vm_id[:28]}{title_suffix}",
            font=dict(size=15)
        ),
        template='plotly_white',
        height=420 if sarima_result else 380,
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


# ---------------------------------------------------------------------------
# Main App
# ---------------------------------------------------------------------------
st.title("📊 Azure VM Workload Dashboard")
st.caption("Phase 2.8 · On-Demand Diagnostics · p99 Y-Axis Scaling · Full-Width Charts")

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

| Test | What it checks | Threshold | Outcome if failed |
|---|---|---|---|
| **ADF (Augmented Dickey-Fuller)** | Stationarity — does the series' mean/variance drift over time (unit root)? | p < 0.05 | Marked ❌ — drift models would extrapolate incorrectly |
| **Ljung-Box** | White noise — is there *any* autocorrelation structure at lag 10? | p < 0.05 | Marked ❌ — the signal is pure noise; no model can beat a flat mean |

> A VM must **pass both tests** to be eligible for forecasting. A series that is
> non-stationary *or* white noise has no exploitable pattern — fitting a model would
> produce statistically meaningless predictions.
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
            vm_df = full_df[full_df['vm_id'] == selected_vm].copy()
            series = vm_df.set_index('timestamp_dt')['avg_cpu']
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
                cpu_fig, spike_clipped = build_prediction_chart(
                    vm_df, preds_df, selected_vm, sarima_result=sarima_res
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

                    # ── Inject SARIMA row into leaderboard if available ──────
                    sarima_res = st.session_state.sarima_results.get(selected_vm)
                    if sarima_res:
                        vm = sarima_res.get('val_metrics', {})
                        if all(np.isfinite(vm.get(k, np.nan)) for k in ('RMSE', 'MAE', 'MAPE')):
                            sarima_row = pd.DataFrame([{
                                'Model':     'SARIMA',
                                'RMSE':      vm['RMSE'],
                                'MAE':       vm['MAE'],
                                'MAPE':      vm['MAPE'],
                                'is_winner': False,
                            }])
                            display_df = pd.concat([display_df, sarima_row], ignore_index=True)

                    # Recalculate winner across all models (including SARIMA)
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
                # SECTION 3 — STL Decomposition
                # ═══════════════════════════════════════════════════════════
                st.divider()
                st.subheader("🔬 Seasonal Decomposition (STL)")
                st.caption(
                    "Decomposition splits the raw signal into three independent components: "
                    "**Trend** (long-run direction), **Seasonality** (repeating cycle of "
                    f"{effective_seasonality} lags), and **Residual** (unexplained noise). "
                    "A dominant seasonal component confirms the detected period is meaningful."
                )
                if effective_seasonality <= 1:
                    st.info(
                        "**Notice:** STL Decomposition requires a seasonal period > 1. "
                        "No detectable repeating cycle was found in this VM's workload "
                        f"(ACF peak below 0.2 threshold in the search range). "
                        "The signal cannot be separated into seasonal components — "
                        "consider using the Manual Override in the sidebar to set a period manually."
                    )
                elif len(series) <= 2 * effective_seasonality:
                    st.info(
                        f"Not enough data for decomposition: need at least "
                        f"**{2 * effective_seasonality}** observations "
                        f"(2 × seasonality period), have **{len(series)}**."
                    )
                elif effective_seasonality > 1 and len(series) > 2 * effective_seasonality:
                    try:
                        st.plotly_chart(
                            build_decomposition_chart(series, effective_seasonality),
                            width='stretch'
                        )
                    except Exception as e:
                        st.warning(f"Decomposition unavailable: {e}")
                else:
                    st.info(
                        "Decomposition requires seasonality > 1 and at least "
                        f"2× the seasonal period ({2*effective_seasonality} lags) of data."
                    )

                # ═══════════════════════════════════════════════════════════
                # SECTION 4 — ACF
                # ═══════════════════════════════════════════════════════════
                st.divider()
                st.subheader("📊 Autocorrelation Function (ACF)")
                st.caption(
                    "ACF measures the correlation of the series with its own lagged values. "
                    "Bars highlighted in **red** exceed the 0.2 significance threshold — "
                    "strong peaks confirm the detected seasonality period and imply the "
                    "series is *not* white noise (Gatekeeper eligible)."
                )
                nlags = min(effective_seasonality * 3 if effective_seasonality > 1 else 72, len(series) // 2 - 1)
                st.plotly_chart(build_acf_chart(series, max(nlags, 10)), width='stretch')

                # ═══════════════════════════════════════════════════════════
                # SECTION 5 — Expert Forecasting Panel (Phase 3.1)
                # ═══════════════════════════════════════════════════════════
                st.divider()
                st.subheader("🔮 Advanced Forecasting (SARIMA)")

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
                            "SARIMA (Seasonal ARIMA) is an advanced statistical model. "
                            "It captures trend (via differencing **d**) and seasonality (via **s**). "
                            "Results compete directly in the Leaderboard above."
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
                                "🔍 Auto-Optimizing Model Parameters…",
                                expanded=True,
                            ) as _status:
                                st.write(
                                    f"Stepwise AIC search · `s={effective_seasonality}` fixed · "
                                    f"max p/q=2, P/Q=2 · output streams in real-time:"
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
                                    st.session_state.sarima_params[selected_vm] = {
                                        'p': int(op), 'd': int(od), 'q': int(oq),
                                        'P': int(oP), 'D': int(oD), 'Q': int(oQ),
                                        's': int(os), 'steps': forecast_steps,
                                        'last_opt_summary': (
                                            f"SARIMA({op},{od},{oq})"
                                            f"({oP},{oD},{oQ},{os})"
                                        ),
                                        'last_opt_error': None,
                                    }

                                    _status.update(
                                        label=(
                                            f"✅ Best model found: "
                                            f"SARIMA({op},{od},{oq})"
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
                            with st.spinner(
                                f"Fitting SARIMA{_order}{_seasonal_order}… "
                                "running validation + final fit"
                            ):
                                try:
                                    result = run_sarima_forecast(
                                        series,
                                        order=_order,
                                        seasonal_order=_seasonal_order,
                                        forecast_steps=int(forecast_steps),
                                    )
                                    # sarima_results must only contain proper forecast results
                                    st.session_state.sarima_results[selected_vm] = result
                                    # Preserve last_opt_summary in params
                                    _prev_p = st.session_state.sarima_params.get(selected_vm, {})
                                    st.session_state.sarima_params[selected_vm] = {
                                        **_prev_p,
                                        'p': int(_p), 'd': int(_d), 'q': int(_q),
                                        'P': int(_P), 'D': int(_D), 'Q': int(_Q),
                                        's': int(_s), 'steps': int(forecast_steps),
                                    }
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"SARIMA failed: {e}")


                    # ── Active forecast summary outside expander ─────────────
                    _cur_sarima = st.session_state.sarima_results.get(selected_vm)
                    if _cur_sarima:
                        _cp = st.session_state.sarima_params.get(selected_vm, {})
                        _o  = (_cp.get('p',1), _cp.get('d',0), _cp.get('q',1))
                        _so = (_cp.get('P',0), _cp.get('D',0), _cp.get('Q',0), _cp.get('s',0))
                        _vm = _cur_sarima.get('val_metrics', {})
                        _aic = _cur_sarima.get('aic', float('nan'))
                        st.success(
                            f"✅ **SARIMA{_o}{_so}** active · "
                            f"AIC: **{_aic:.1f}** · "
                            f"Val RMSE: **{_vm.get('RMSE', float('nan')):.3f}** · "
                            f"Forecast: **{_cp.get('steps', '?')} steps** ahead"
                        )
                        st.caption(
                            "Shaded area on the chart represents the **95 % confidence interval** "
                            "(potential load range). The validation line (dotted) shows the model's "
                            "accuracy on the hold-out window before projecting into the future."
                        )
                        if st.button("🗑️ Clear SARIMA Forecast",
                                     key=f"clear_sarima_{selected_vm}"):
                            st.session_state.sarima_results.pop(selected_vm, None)
                            st.session_state.sarima_params.pop(selected_vm, None)
                            st.rerun()

            else:
                st.info("ℹ️ Run **Fleet Diagnostics** in the Fleet Analytics tab first to unlock the Deep Dive.")

else:
    st.info("⏳ Awaiting data — upload a CSV or allow the demo dataset to load.")

st.sidebar.divider()
st.sidebar.info("v1.5 · Phase 3.1 · SARIMA Expert Forecasting")
