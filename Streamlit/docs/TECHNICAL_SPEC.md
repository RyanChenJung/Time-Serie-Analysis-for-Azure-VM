# docs/TECHNICAL_SPEC.md

## 1. DEVELOPMENT ENVIRONMENT
- **Python Version:** 3.9+
- **Environment Manager:** `venv` (Virtual Environment)
- **Primary Framework:** Streamlit (Latest stable)
- **Dependency Management:** All libraries must be recorded in `requirements.txt`.

## 2. CORE STACK & LIBRARIES
- **Data Manipulation:** `pandas`, `numpy`
- **Visualization:** `plotly` (Enterprise theme: `plotly_white`)
- **Statistical Testing:** `statsmodels` (for ADF & Ljung-Box)
- **Machine Learning:** `scikit-learn` (for Benchmarking & Advanced models)

## 3. CODE ARCHITECTURE STANDARDS
- **Logic-View Separation:**
    - All mathematical computations, statistical tests, and data aggregations must reside in standalone functions (e.g., `utils/analytics.py`).
    - The Streamlit script (`app.py`) should only handle UI rendering and function calls.
- **Data Pipeline:**
    - Avoid global state where possible; use Streamlit's `@st.cache_data` for heavy computations.
    - Standardized column naming convention: `timestamp_dt`, `vm_id`, `avg_cpu`, `max_cpu`, `min_cpu`.

## 4. UI/UX STANDARDS
- **Responsive Layout:** Use `st.columns` for metric cards.
- **Consistency:** All Plotly charts must share the same color palette and interactive configuration (e.g., display mode bar).