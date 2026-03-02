# docs/ROADMAP.md

## Project Status: 🚀 Phase 3 — Advanced Forecasting (3.1 DONE · 2.9 DONE · 2.9.5 DONE)

---

## Phase 1: Foundation & Ingestion [DONE]
**Goal:** Establish a robust data pipeline with a "Demo-First" approach and dual-granularity support.

- [x] **1.1 Hybrid Data Loading (Demo & Upload)**
    - **Default Mode:** Automatically load `final_deep_readings_20vms.csv` as the demonstration dataset.
    - **Upload Mode:** Provide `st.file_uploader`. If a new file is provided, override the demo data.
    - Validate schema (timestamp, vm_id, avg_cpu, max_cpu, min_cpu).
- [x] **1.2 Dynamic Temporal Anchoring**
    - UI inputs for `Start Date` and `Start Time`.
    - Map integer offsets to `datetime` objects relative to the user's input.
- [x] **1.3 Dual-Granularity Engine**
    - Toggle for "5-min (Native)" vs. "1-hour (Aggregated)".
    - **Aggregation Logic:** `avg_cpu` -> Mean, `max_cpu` -> Max, `min_cpu` -> Min.
- [x] **1.4 Data Cleaning & Interpolation**
    - Detect index gaps and apply **Linear Interpolation**.
- [x] **1.5 Basic Descriptive UI**
    - VM Selector & Metric Summary Cards.
    - Interactive **Plotly** Line Chart with standardized "Enterprise" theme.

---

## Phase 2: The Gatekeeper & Benchmarking [DONE]
- [x] **2.1 The Suitability Gate:** ADF Test & Ljung-Box White Noise Test.
- [x] **2.2 Baseline Matrix:** Naive, Seasonal Naive, Drift, and Mean models.
- [x] **2.3 Leaderboard UI:** Metric comparison (RMSE/MAE/MAPE).

## Phase 2.5: Performance Optimization & State Synchronization [DONE]
- [x] Parallelized Diagnostics Engine (ThreadPoolExecutor).
- [x] Persistent caching for Demo data (`cache/demo_results.pkl`).
- [x] Step-based CV optimization.
- [x] State synchronization and UI fixes.

## Phase 2.6: On-Demand Deep Dive Workflow [DONE]
- [x] Lightweight Fleet Analytics (ADF/LB/Seasonality only).
- [x] Level 1: Instant baseline predictions for test hold-out.
- [x] Level 2: Triggered Expanding Window CV with session caching.
- [x] Smart Cache Invalidation on seasonality change.
- [x] Dynamic Leaderboard winner logic.

## Phase 2.7: UI/UX Refinement & Polish [DONE]
- [x] **Gatekeeper Explainability:** `💡 How the Gatekeeper Works` expander with ADF/LB comparison table.
- [x] **Vertical Chart Layout:** CPU+Forecasts → Decomposition → ACF, all full-width with captions.
- [x] **Deprecation Fix:** All `use_container_width=True` replaced with `width='stretch'` (Streamlit 1.54+).
- [x] **Smart Y-Axis:** All charts start at 0, scaled to `[0, max × 1.1]`; axes are fixed-range (no slider drag).
- [x] **Visual Hierarchy:** `st.divider()` between every major Deep Dive section (7 total).
- [x] **Winner Badge:** Best model row highlighted with green `#D1FAE5` background.
- [x] **ACF Signal Coloring:** Bars exceeding |0.2| significance threshold rendered in red.

## Phase 3: Advanced Forecasting & Interactive Tuning
- [x] **3.1 SARIMA Expert Forecasting Panel** ✅
    - `utils/forecasting.py`: `run_sarima_forecast()` (dual-execution: 80/20 validation + 100% final fit) and `auto_optimize_sarima()` (pmdarima).
    - Single Master Chart: SARIMA forecast line + 95% CI ribbon overlaid on the Section 1 CPU chart.
    - Leaderboard Integration: SARIMA row injected with hold-out RMSE/MAE/MAPE; winner recalculated across all models.
    - Expert Control Panel: p,d,q | P,D,Q,s inputs; smart `d` default (1 for ⚠️ Non-Stationary); Auto-Optimize + Execute buttons.
    - 14 new pytest tests → total 26/26 passing.
- [x] **2.9 Explainability Boost & PACF Integration** ✅
    - `utils/diagnostics.py`: `calculate_pacf()` with Yule-Walker method, safe nlags clamping.
    - `app.py`: `build_pacf_chart()` helper; Section 4 upgraded to side-by-side ACF + PACF columns.
    - Tab 1 Gatekeeper expander: new **Stage 3 — Auto-Seasonality Detection** section with full algorithm explanation and worked example.
    - Holt-Winters placeholder expander added to Tab 2 (Section 6) as Phase 3.2 lead-in.
    - 3 new pytest tests (AR1 spike, white noise bound, short-series guard) → 24/24 non-pmdarima tests passing.
- [x] **2.9.5 Environment Fix & Aperiodic Logic Refinement** ✅
    - `pmdarima 2.1.1` installed; **29/29 pytest tests now passing** (31.30s).
    - `utils/diagnostics.py`: no change (HP-filter is a chart-helper concern).
    - `app.py`: `hpfilter` imported from `statsmodels.tsa.filters.hp_filter`.
    - `build_trend_residual_chart(series)`: HP-Filter with auto-lambda (1600 / 129600), fallback to 12-pt MA, 2-panel Trend+Residual chart (Plotly).
    - Section 3 rewritten: `_is_aperiodic = (s <= 1)` branches to Aperiodic chart vs STL vs insufficient-data notice.
    - Section 5 SARIMA panel: `_aperiodic_arima = (s <= 1)` hides P/D/Q/s block; `_P,_D,_Q,_s = 0,0,0,0` injected silently.
    - Tab 1 Gatekeeper text: added Periodic/Aperiodic decision table and second example.
- [ ] **3.2 Holt-Winters Exponential Smoothing:** Additive/multiplicative modes, AIC auto-fit, Leaderboard integration.
- [ ] **3.3 Residual Diagnostics:** Post-modeling Ljung-Box and error distribution.
- [ ] **3.4 Optimization Engine:** Capacity planning recommendations based on forecast peaks.