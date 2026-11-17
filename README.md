# Hyderabad Air Quality Forecasting and Predictive Hotspot Mapping

[![Python 3.13](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![Streamlit 1.52](https://img.shields.io/badge/Streamlit-1.52.2-FF4B4B.svg)](https://streamlit.io/)
[![Machine Learning](https://img.shields.io/badge/ML-LightGBM%20%7C%20XGBoost%20%7C%20Stacking-green.svg)](https://scikit-learn.org/)
[![Interpretability](https://img.shields.io/badge/Explainability-TreeSHAP-orange.svg)](https://shap.readthedocs.io/)
[![Testing](https://img.shields.io/badge/Tests-64%2F64%20Passing-brightgreen.svg)](implementation/tests/)

> **Academic Capstone & Environmental Research Prototype**  
> Spatio-Temporal Air Quality Modeling, Transferability Auditing, and Explainable Machine Learning across the Greater Hyderabad Metropolitan Area (1,600 km²).

---

## Table of Contents
- [Abstract](#abstract)
- [Problem Statement](#problem-statement)
- [Formal Research Objectives](#formal-research-objectives)
- [Proposed Methodology](#proposed-methodology)
  - [Chronological Partitioning & Zero-Leakage Buffering](#chronological-partitioning--zero-leakage-buffering)
  - [Feature Engineering Architecture (84 Primary Predictors)](#feature-engineering-architecture-84-primary-predictors)
  - [Scientific Quarantine of Unverified Road Proxies](#scientific-quarantine-of-unverified-road-proxies)
- [Machine Learning Models & Algorithms](#machine-learning-models--algorithms)
- [Verified Empirical Results](#verified-empirical-results)
  - [Point Forecasting Performance (+1h to +48h)](#point-forecasting-performance-1h-to-48h)
  - [The 388% Spatial Generalizability Penalty (LOSOCV)](#the-388-spatial-generalizability-penalty-losocv)
  - [TreeSHAP Explainability & Temporal Feature Transition](#treeshap-explainability--temporal-feature-transition)
- [Dataset Provenance & Quality Accounting](#dataset-provenance--quality-accounting)
- [Project Architecture & Directory Structure](#project-architecture--directory-structure)
- [Installation & Environment Setup](#installation--environment-setup)
- [Running the Streamlit Application](#running-the-streamlit-application)
- [Running the Automated Test Suite](#running-the-automated-test-suite)
- [Research Ethics & Regulatory Disclaimers](#research-ethics--regulatory-disclaimers)

---

## Abstract

Ambient particulate matter ($\text{PM}_{2.5}$) presents a chronic public health crisis in rapidly developing global South metropolises. In the Greater Hyderabad Metropolitan Area (~10 million residents across 1,600 km²), ambient regulatory monitoring is constrained to just four Continuous Ambient Air Quality Monitoring Stations (CAAQMS). This work formulates, audits, and open-sources a leak-free multi-source machine learning pipeline for multi-horizon $\text{PM}_{2.5}$ forecasting (+1h, +6h, +24h, +48h) and continuous spatial hotspot mapping. 

By integrating continuous in-situ sensor telemetry with Open-Meteo ERA5 atmospheric thermodynamics, we benchmark Gradient Boosted Decision Trees (LightGBM, XGBoost, Random Forest) and Stacking Ensembles against Zero-Order Persistence and Seasonal-Naive baselines. Evaluating on an out-of-time test partition (Jan–Sep 2026; 18,970 primary samples), stacking achieves **$3.01\,\mu\text{g/m}^3$ MAE** at +1h and **$5.76\,\mu\text{g/m}^3$ MAE** at +6h ($18.0\%$ skill over persistence). However, rigorous Leave-One-Station-Out Cross-Validation (LOSOCV) reveals a **$+388\%$ spatial error penalty** ($11.703\,\mu\text{g/m}^3$ macro MAE; $11.429\,\mu\text{g/m}^3$ pooled), highlighting the severe limitations of naive spatial interpolation across sparse urban geometries. Axiomatic TreeSHAP attributions reveal that aerosol inertia dominates short horizons ($79.6\%$ share at +1h), while synoptic thermodynamics govern medium-range horizons ($16.8\%$ at +48h).

---

## Problem Statement

Continuous ambient air quality monitoring in Greater Hyderabad is severely constrained:
- **Spatial Sparsity:** Only 4 continuous reference stations exist across a $1,600\,\text{km}^2$ urban expanse with heterogeneous micro-climates (industrial bulk-drug manufacturing, forested enclaves, densely packed commercial corridors, and agricultural research hubs).
- **Spatial Transfer Blindspot:** Most published literature evaluates models exclusively at the sensors where they were trained ($R^2 > 0.90$), failing to measure performance degradation in unmonitored sectors.
- **Temporal Boundary Leakage:** Naive rolling statistics or random cross-validation leak prospective target values into past feature sets.

---

## Formal Research Objectives

1. **Zero-Leakage Multi-Source Pipeline:** Build an audited ingestion, imputation, and feature store framework enforcing strict chronological boundary buffers.
2. **Multi-Horizon Temporal Benchmarking:** Quantify predictive skill against Zero-Order Persistence across short-range (+1h, +6h) and synoptic (+24h, +48h) forecast horizons.
3. **Spatial Generalizability Audit:** Empirically quantify the spatial degradation penalty via Leave-One-Station-Out Cross-Validation (LOSOCV) comparing Inverse Distance Weighting (IDW), donor station means, and held-out historical baselines.
4. **Axiomatic Interpretability:** Apply TreeSHAP to decouple physical atmospheric associations from spurious statistical artifacts across lead horizons.

---

## Proposed Methodology

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       MULTI-SOURCE DATA INGESTION                           │
│   CPCB CAAQMS Telemetry (4 Stations)  │  Open-Meteo ERA5 Atmospheric Data   │
│   [PM2.5, PM10, NO2, SO2, CO, Ozone]  │  [Temp, RH, Surface Pres, Wind U/V] │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     CHRONOLOGICAL ZERO-LEAKAGE PURGING                      │
│   Train: 2023-01-01 to 2024-12-31  (48h Boundary Buffer: 192 samples purged) │
│   Val:   2025-01-01 to 2025-12-31  (48h Boundary Buffer: 192 samples purged) │
│   Test:  2026-01-01 to 2026-09-30  (Locked Out-of-Time Test Set)            │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       84 APPROVED PRIMARY PREDICTORS                        │
│   • Historical Dynamics (23) : Lags (1-24h), Past-Only Rolling Stats, Accel │
│   • Atmospheric Physics (18) : Temp, Pres Tendency, Dewpoint, Blh           │
│   • Wind Dynamics (14)       : U/V Orthogonal Vectors, Direction Harmonics   │
│   • Co-Pollutants (18)       : PM10, NO, NO2, SO2, CO, Ozone, PM2.5/PM10    │
│   • Cyclical Harmonics (11)  : Diurnal Hour, Day-of-Week, Solar Zenith      │
│   [19 Static Road Metrics Quarantined — Excluded From Primary Models]       │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      PREDICTIVE MODELING & INFERENCE                        │
│   • Point Forecasting: Stacking Ensembles (+1h, +6h), LightGBM (+24h, +48h) │
│   • Spatial Surface:   IDW Spatial Interpolation & LOSOCV Spatial Audit     │
│   • Interpretability:  Axiomatic TreeSHAP Attribution & Natural Narratives  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Chronological Partitioning & Zero-Leakage Buffering
To eliminate lookahead bias:
- **Training Set:** `2023-01-01 00:00:00` to `2024-12-31 23:00:00` ($70,176$ station-hours).
- **Validation Set:** `2025-01-01 00:00:00` to `2025-12-31 23:00:00` ($35,040$ station-hours).
- **Locked Test Set:** `2026-01-01 00:00:00` to `2026-09-30 23:00:00` ($26,208$ station-hours).
- **48-Hour Boundary Purge:** Exactly 192 samples at the end of Train and 192 samples at the end of Validation ($48\,\text{h} \times 4\,\text{stations}$) were deleted to ensure no forward-looking target overlap crossed partition boundaries.

### Feature Engineering Architecture (84 Primary Predictors)
Total column identity:
$$\text{Total Schema (173)} = \text{Identifiers (3)} + \text{Metadata/Flags (51)} + \text{Approved Features (84)} + \text{Quarantined Road Proxies (19)} + \text{Targets (16)}$$

### Scientific Quarantine of Unverified Road Proxies
An independent audit revealed that road-distance and intersection metrics derived from static buffer assumptions lacked ground-truth vehicular volume telemetry. Consequently, **all 19 derived road proxies were quarantined** and excluded from all operational models to preserve scientific integrity.

---

## Machine Learning Models & Algorithms

1. **Stacking Meta-Regressor:** Combines cross-validated base predictions from LightGBM, XGBoost, and Random Forest using Ridge Regression meta-learners. Serves as primary point forecast at $+1\text{h}$ and $+6\text{h}$.
2. **LightGBM (Light Gradient Boosting Machine):** Fast, leaf-wise gradient boosting optimized with early stopping. Serves as primary model at $+24\text{h}$ and $+48\text{h}$.
3. **XGBoost:** Scalable tree-boosting with depth-wise tree growth.
4. **Random Forest:** Ensemble of decorrelated decision trees with bagging.
5. **Baselines:** Zero-Order Persistence ($y_{t+h} = y_t$) and Seasonal Naive ($y_{t+h} = y_{t+h-24}$).

---

## Verified Empirical Results

All evaluations strictly enforce the **persistence-valid sample mask** (samples where target and persistence prediction are both valid non-NaN observations).

### Point Forecasting Performance (+1h to +48h)

| Horizon | Primary Model | Test Samples ($n$) | MAE ($\mu\text{g/m}^3$) | RMSE ($\mu\text{g/m}^3$) | $R^2$ Score | Persistence MAE | Skill vs Persistence |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **+1h** | **Stacking Ensemble** | 18,970 | **3.01** | **4.97** | **0.963** | 3.01 | **0.0%** (Equal) |
| **+6h** | **Stacking Ensemble** | 18,940 | **5.76** | **8.86** | **0.865** | 7.02 | **+18.0%** |
| **+24h** | **LightGBM** | 18,610 | **6.84** | **10.51** | **0.803** | 7.74 | **+11.6%** |
| **+48h** | **LightGBM** | 18,170 | **8.44** | **12.78** | **0.697** | 9.07 | **+6.9%** |

### The 388% Spatial Generalizability Penalty (LOSOCV)

Leave-One-Station-Out Cross-Validation (LOSOCV) across the 2026 test partition reveals that point forecasting accuracy does not generalize naively to unmonitored sectors:

| Held-Out Station | Valid Test Hours | Observed Mean ($\mu\text{g/m}^3$) | IDW Estimate ($\mu\text{g/m}^3$) | LOSOCV MAE ($\mu\text{g/m}^3$) | Spatial $R^2$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Bollaram Industrial Area** | 1,875 | 41.85 | 30.34 | 13.14 | +0.507 |
| **Central University** | 5,743 | 19.22 | 31.22 | 12.84 | +0.025 |
| **ECIL Kapra** | 5,849 | 30.02 | 27.60 | 11.67 | -1.527 |
| **ICRISAT Patancheru** | 5,606 | 27.10 | 26.88 | 9.16 | +0.535 |
| **Network Macro-Average** | **19,073 Total** | **29.55** | **29.01** | **11.703** | **-0.115** |
| **Network Pooled (Weighted)** | **19,073 Total** | **29.55** | **29.01** | **11.429** | **-0.088** |

> [!WARNING]
> While point nowcasting precision is **$3.01\,\mu\text{g/m}^3$ MAE**, spatial transfer error rises to **$11.703\,\mu\text{g/m}^3$** (a **$+388\%$ degradation**). Naive spatial interpolation cannot substitute for in-situ reference sensors across complex urban typology.

### TreeSHAP Explainability & Temporal Feature Transition

TreeSHAP attribution decomposes feature category importance across forecast horizons:
- **At +1h:** Historical $\text{PM}_{2.5}$ dynamics (autoregressive lags and rolling momentum) govern **$79.6\%$** of predictive share.
- **At +48h:** Aerosol inertia decays significantly, while synoptic thermodynamics and boundary layer dynamics expand to govern **$16.8\%$** of predictive share.

---

## Dataset Provenance & Quality Accounting

- **CPCB Ground Telemetry:** 4 Continuous Ambient Air Quality Monitoring Stations (CAAQMS) from Central Pollution Control Board India (2023-01-01 to 2026-09-30; 131,424 total station-hours).
- **Atmospheric Reanalysis:** Open-Meteo ERA5 atmospheric thermodynamics, 10m wind orthogonal dynamics, and surface pressure.
- **Station Hardware Accounting:** Bollaram Industrial Area experienced hardware outages resulting in valid data restricted to Jan–Feb and Aug–Sep 2026 ($36\%$ coverage); Central University ($n=5,743$), ECIL Kapra ($n=5,849$), and ICRISAT ($n=5,606$) maintained continuous uptime.

---

## Project Architecture & Directory Structure

```
Hyderabad-Air-Quality-Forecasting/
├── app.py                             # Root application launcher
├── requirements.txt                   # Verified production environment dependencies
├── README.md                          # Comprehensive project documentation
├── .gitignore                         # Git exclusion rules
│
├── dashboard/                         # Streamlit Interactive Web Application
│   ├── app.py                         # Application routing and page orchestrator
│   ├── backend/                       # Data loader, forecasting, and spatial services
│   ├── components/                    # Sidebar, topbar, disclaimers, and html_utils
│   ├── pages/                         # 7 Production Dashboard Pages
│   │   ├── page_overview.py           # 1. Project Overview & Domain Metrics
│   │   ├── page_observations.py       # 2. In-Situ Station Telemetry
│   │   ├── page_forecasting.py        # 3. Multi-Horizon Forecasting (+1h to +48h)
│   │   ├── page_spatial.py            # 4. Spatial Hotspot Mapping & LOSOCV
│   │   ├── page_explainability.py     # 5. TreeSHAP Attribution & Narratives
│   │   ├── page_methodology.py        # 6. Zero-Leakage Chronology & Audit
│   │   └── page_about.py              # 7. About the Research & Capstone Details
│   └── styles/                        # Custom CSS, Plotly theme, and render_html helper
│
├── demo_site/                         # Standalone Forecasting Prototype
│   └── demo_app.py                    # Self-contained Streamlit demo
│
├── implementation/                    # Core Research Engineering Codebase
│   ├── configs/                       # Configuration, constants, and station metadata
│   ├── data_ingestion/                # CPCB telemetry and ERA5 weather parsers
│   ├── preprocessing/                 # Prospective imputer, gap analyzer, and scaler
│   ├── feature_engineering/           # 84 approved features, lags, thermodynamics
│   ├── target_generation/             # Multi-step horizons and classification targets
│   ├── modeling/                      # LightGBM, XGBoost, Random Forest, Stacking
│   ├── spatial/                       # IDW interpolation and LOSOCV evaluation
│   ├── explainability/                # TreeSHAP computation and station narratives
│   ├── validation/                    # Chronological split and boundary buffer audits
│   └── tests/                         # 64 Automated Pytest Verification Tests
│
├── analysis/                          # Paper Robustness & Significance Evaluation
│   ├── step1_bollaram_safe.py         # Bollaram-safe multi-station evaluation
│   ├── step2_seasonal_robustness.py   # Seasonal quarterly skill evaluation
│   ├── step3_significance_tests.py    # Diebold-Mariano & Bootstrap significance
│   ├── step4_spatial_baselines.py     # LOSOCV comparison against donor baselines
│   ├── step5_shap_scale.py            # Large-sample TreeSHAP attribution
│   └── outputs/                       # Analysis JSONs, CSVs, PNGs, and Summary
│
├── data/                              # Verified Data Stores
│   ├── raw/                           # Raw CPCB and ERA5 CSV datasets
│   ├── interim/                       # Cleaned prospective features
│   ├── processed/                     # Feature stores in Parquet and JSON catalogs
│   └── predictions/                   # Model prediction Parquets and metric JSONs
│
├── models/                            # Trained Machine Learning Model Artifacts
│   ├── lgb_regressor_*.joblib         # LightGBM models (+1h, +6h, +24h, +48h)
│   ├── xgb_regressor_*.joblib         # XGBoost models (+1h, +6h, +24h, +48h)
│   ├── stacking_meta_regressor_*.joblib # Stacking Meta-models
│   ├── rf_regressor_*.joblib          # Random Forest models (Tracked via Git LFS)
│   └── regression_feature_imputer.joblib # Feature imputation transformers
│
└── reports/                           # Academic Milestone Audits & Visualizations
    ├── figures/                       # 40 High-Resolution PNG Analytical Plots
    └── *.md                           # Milestone audits, leakage checks, reconciliation
```

---

## Installation & Environment Setup

### 1. Prerequisites
- Python 3.11, 3.12, or 3.13
- Git and Git LFS:
  ```bash
  git lfs install
  ```

### 2. Clone the Repository
```bash
git clone https://github.com/<your-username>/Hyderabad-Air-Quality-Forecasting.git
cd Hyderabad-Air-Quality-Forecasting
```

### 3. Create and Activate a Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Running the Streamlit Application

Launch the research dashboard:
```bash
streamlit run app.py
```
Or directly launch via the dashboard package:
```bash
streamlit run dashboard/app.py
```
The application will open in your web browser at:
`http://localhost:8501`

---

## Running the Automated Test Suite

To verify mathematical integrity, zero target leakage, and numerical reproducibility:
```bash
python -m pytest implementation/tests -v
```
**Expected Outcome:**
```text
============================== 64 passed in ~16s ==============================
```



