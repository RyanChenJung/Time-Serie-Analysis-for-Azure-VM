# docs/ROADMAP.md

## Project Status: 🚀 Phase 3 — Advanced Forecasting (3.1 DONE)

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
- [ ] **3.2 Additional Advanced Models:** Random Forest or Prophet.
- [ ] **3.3 Residual Diagnostics:** Post-modeling Ljung-Box and error distribution.
- [ ] **3.4 Optimization Engine:** Capacity planning recommendations based on forecast peaks.