# ☁️ Azure VM Time Series Forecasting & Cloud Optimization

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](這裡貼上你的_Streamlit_Deploy_網址)
[![Demo Video](https://img.shields.io/badge/YouTube-Watch_Demo-FF0000?logo=youtube&logoColor=white)](這裡貼上你的_影片_網址)
![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Pandas](https://img.shields.io/badge/Pandas-Data_Manipulation-150458?logo=pandas)
![Statsmodels](https://img.shields.io/badge/Statsmodels-Time_Series-3f51b5)

> **🚀 Try it live:** [Interactive Streamlit Dashboard](https://time-serie-analysis-for-azure-vm-2ge4g8z2fw6erg46yj3shb.streamlit.app/) | **📹 Watch:** [Project Demo Video](https://youtu.be/39ACc8nZ1_4)

## 📌 Project Overview
Predicting CPU utilization of Virtual Machines (VMs) is a critical challenge in modern Cloud Computing. This project analyzes the **Azure Public Dataset** to forecast VM workloads using advanced Time Series and Machine Learning models. 

By accurately predicting CPU spikes and idle windows, this project provides a data-driven foundation for **Proactive Auto-scaling** and **Cloud FinOps**, enabling enterprises to optimize resource allocation and significantly reduce cloud infrastructure costs.

## 💼 Business Value (FinOps & SRE)
* **Cost Optimization (FinOps):** Identifies stable, predictable VMs that can be safely transitioned from expensive Pay-As-You-Go pricing to Reserved Instances (RIs), saving up to 80% in costs.
* **Proactive Auto-Scaling (SRE):** Anticipates heavy workloads and CPU spikes before they occur, allowing systems to scale up resources proactively to prevent performance degradation.
* **Zombie VM Detection:** Identifies machines with pure random noise (White Noise) and consistently low loads for safe deallocation.

## 📊 Dataset & Feature Engineering
* **Source:** Azure VM Public Dataset (telemetry data).
* **Features Used:** `avg_cpu`, `min_cpu`, `max_cpu`.
* **Multi-scale Aggregation:** Filtered noisy 5-minute raw data into **Hourly** and **Daily** frequencies to extract robust macroeconomic seasonal patterns.
* **Transformation:** Applied **Box-Cox Transformation** (with offset) to stabilize variance and normalize highly skewed CPU distributions.

## 🔬 Methodology & Modeling Pipeline
1. **Statistical Verification:** Conducted **Ljung-Box Tests** across 20 candidate VMs to confirm strong non-randomness and high predictability.
2. **Diagnostics:** Utilized ACF/PACF plots and Spectral Analysis to identify strong 24-hour (daily) seasonalities.
3. **Sliding Window Validation:** Implemented cross-validation to ensure model robustness over time, avoiding the pitfalls of single-shot forecasting.
4. **Residual Diagnostics:** Validated model completeness using Standardized Residuals, KDE, Q-Q Plots, and Correlograms to ensure residuals resemble White Noise.

## 🏆 Key Results & Model Evaluation
We pitched our advanced models (ARIMA, SARIMA, SARIMAX, VAR, TBATS, Holt-Winters, Gradient Boosting) against robust baselines (Mean, Naive, Seasonal Naive, Drift).

* **The Winner:** The **SARIMAX** model using `min_cpu` as an external variable performed best, cutting the error of our baseline (Seasonal Naive) by more than half.
* **Exogenous Boost:** Adding extra context (like minimum CPU values) significantly improved the mathematical models compared to looking at the timeline alone.
* **Machine Learning vs. Stats:** Traditional statistical models (SARIMAX) actually outperformed the ML approach (Gradient Boosting) for this specific dataset.
* **Baseline Comparison:** Both Seasonal Naive and Moving Average served as the "floor"—any model ranked above them provided genuine predictive value.

## 🚀 Interactive Streamlit Dashboard
We built an interactive **Streamlit Dashboard** to make our forecasting workflow accessible and actionable. 

👉 **[Access the Live Dashboard Here](https://time-serie-analysis-for-azure-vm-2ge4g8z2fw6erg46yj3shb.streamlit.app/)**
👉 **[Watch the Demo Video](https://youtu.be/39ACc8nZ1_4)**

* **Key Features:**
  * Automated execution of Baseline Benchmarks.
  * Interactive toggling between Univariate (ARIMA, Holt-Winters) and Multivariate (VAR, ARIMAX) models.
  * Real-time visualized residual diagnostics for model reliability checks.

## 👥 Team 2 Members & Contributions
* **Ryan Chen** 
* **Lawrence**
* **Sami** 
* **Jack** 
