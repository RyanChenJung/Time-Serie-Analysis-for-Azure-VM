# docs/PROJECT_MANIFESTO.md

## 1. PROJECT VISION
**Project Name:** Azure VM Workload Patterns Forecasting Dashboard (v1.0.0)
**Objective:** A generalized analysis tool for Azure VM telemetry that supports dynamic temporal anchoring, multi-granularity processing, and rigorous statistical benchmarking.

## 2. DATA INGESTION & TEMPORAL LOGIC
- **Dynamic Anchoring:** Users must define the start date/time in the UI. Integer offsets in the CSV will be mapped relative to this user-defined T0.
- **Dual-Granularity Support:** - **Original (5-min):** For high-resolution diagnostic analysis.
    - **Aggregated (1-hr):** For trend forecasting and efficiency.
- **Aggregation Logic:** - `avg_cpu`: Mean | `max_cpu`: Max | `min_cpu`: Min.
- **Completeness:** Linear interpolation for any missing steps in the time index.

## 3. THE "GATEKEEPER" FILTRATION (STRICT)
Before modeling, data must pass:
1. **Stationarity (ADF Test):** To detect unit roots and determine differencing needs.
2. **White Noise Check (Ljung-Box Test):** If the p-value is high (data is white noise), the VM is flagged as "Non-Forecastable."
3. **Variance Threshold:** To filter out VMs with near-constant usage.

## 4. BENCHMARKING HIERARCHY (THE MATRIX)
- **Baseline Models:** Naive, Seasonal Naive, Drift, and Moving Average.
- **Performance Hurdle:** Advanced models (ARIMA, RF, etc.) must outperform the *best-performing* baseline model in terms of RMSE/MAE to be considered valid.
- **Residual Integrity:** Post-modeling Ljung-Box test to ensure no information is left in the residuals.

## 5. ENGINEERING & UI STANDARDS
- **Maintenance-First Architecture:** Pure-functional backend logic (stateless functions) + Streamlit UI layer.
- **Aesthetic Consistency:** Use Plotly with a unified theme. Graphs must include interactive tooltips, confidence interval shading (for forecasts), and standardized color palettes for CPU metrics.