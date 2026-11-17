# MILESTONE 5 COMPLETION REPORT
## Research-Grade Streamlit Dashboard and System Integration

**Project Title:** Traffic-Aware Spatio-Temporal $\text{PM}_{2.5}$ Concentration and Hotspot Forecasting in Hyderabad Using Multi-Source Environmental Data and Explainable Machine Learning  
**Milestone:** Milestone 5 — Dashboard Development & Integration  
**Date of Completion:** November 15, 2026  
**Status:** **APPROVED & FULLY VERIFIED (64/64 Tests Passing)**  
**Local Port:** `http://localhost:8501`

---

## 1. Executive Summary

Milestone 5 has successfully integrated the complete, leakage-free data science pipeline into a responsive, modular, academic capstone Streamlit application. The dashboard decouples presentation logic from backend data loaders, spatial interpolation services, and TreeSHAP explainability routines while maintaining all mandatory scientific safeguards (network sparsity disclosures, 388% spatial transfer penalty, retrospective weather limitations, and concentration vs. AQI clarity).

### Key Accomplishments
1. **7-Section Modular Navigation:** Full implementation of Project Overview, Air Quality Observations, PM2.5 Forecasting, Spatial Hotspot Mapping, Explainability, Methodology & Data Quality, and About the Research.
2. **Strict Architectural Separation:** Presentation layer (`pages/`, `components/`) isolated from backend services (`backend/data_loader.py`, `backend/forecasting_service.py`, `backend/spatial_service.py`, `backend/explainability_service.py`).
3. **Automated Test Suite Expansion:** Added 8 integration tests in `test_dashboard.py`, bringing the full automated test suite to **64/64 passing tests** in 50.99s.
4. **End-to-End Visual Verification:** Verified all 7 pages via headless browser rendering, capturing 7 publication-grade full-compositor screenshots with zero runtime errors.

---

## 2. Dashboard Page Structure

The application features a clean sidebar navigation with the following 7 sections:

1. **1. Project Overview:** Executive summary, research motivation, 6-stage architecture overview, continuous monitoring network table (4 CPCB stations), model performance scorecard, and core scientific limitations.
2. **2. Air Quality Observations:** Historical continuous physical observations (2023–2026), station-wise interactive time series (PM2.5, PM10, NO2, SO2, CO, Ozone), diurnal hourly cycle profiles (highlighting rush-hour peaks 08–11 and 18–22), and sensor uptime/downtime accounting.
3. **3. PM2.5 Forecasting:** Multi-horizon forecast evaluation (+1h, +6h, +24h, +48h), station selectors, date window presets, designated winning model cards (Stacking Bayesian Ridge vs LightGBM), actual vs. predicted Plotly timeseries, parity plots, residual histograms, and live KPI metric rows.
4. **4. Spatial Hotspot Mapping:** Interactive geographic scatter map of Greater Hyderabad with station markers, IDW continuous field grid, 3-tier uncertainty overlays (Low $\le 5\,\text{km}$, Moderate $5\text{--}15\,\text{km}$, High Extrapolation $> 15\,\text{km}$), acute excursion ($\ge 60\,\mu\text{g/m}^3$) and severe spike ($\ge 120\,\mu\text{g/m}^3$) metric cards, high-resolution publication maps, and forensic hotspot explanations.
5. **5. Explainability (TreeSHAP):** Global feature importance bar chart (Top 15 predictors), relative group share pie charts (demonstrating the 10-fold expansion of atmospheric thermodynamics from +1h to +48h), high-resolution beeswarm and partial dependence plots, local station waterfall attributions, and natural language diagnostic narratives.
6. **6. Methodology & Data Quality:** Preprocessing protocols, 84 approved primary predictors taxonomy, full disclosure of the 19 quarantined unverified road proxies, temporal boundary 48-hour buffer purging, and Leave-One-Station-Out Cross-Validation (LOSOCV) metric accounting (explaining the 388% spatial transfer penalty and Kriging variogram underdetermination).
7. **7. About the Research:** Problem statement, literature gap analysis, formal primary and secondary research objectives, model architecture details, and future roadmap (dense low-cost sensor networks, operational NWP integration).

---

## 3. Mandatory Scientific Safeguards Enforced in UI

| Scientific Requirement | Implementation in Streamlit Dashboard | Visual Safeguard Location |
| :--- | :--- | :--- |
| **Sparse 4-Station Network** | Prominently stated across sidebar, overview, and spatial pages. Station table lists all 4 official CPCB coordinates. | Persistent Sidebar & Section 1 / Section 4 |
| **Spatial Transfer Penalty** | LOSOCV MAE ($11.703\,\mu\text{g/m}^3$) displayed alongside Point ML error ($3.01\,\mu\text{g/m}^3$). Highlights the $+388\%$ degradation. | KPI Metric Scorecards & Section 4 / Section 6 |
| **Retrospective Reanalysis** | Explicit notice: "Retrospective Historical Data Archive (2023–2026) — Not an active operational sensor feed." | Section 2 Header & Sidebar Data Freshness Badge |
| **Quarantine of Road Proxies** | Full documentation of why the 19 unverified OSM proxies were quarantined from the 84 primary predictors. | Section 6 Feature Store Table |
| **Experimental Ordinary Kriging** | Ordinary Kriging labeled as an experimental diagnostic due to underdetermined variograms (3–6 point pairs) and condition numbers $>10^{12}$. | Section 4 & Section 6 |
| **Concentration vs. AQI** | All predictions, targets, and plots explicitly labeled in $\mu\text{g/m}^3$. $60\,\mu\text{g/m}^3$ and $120\,\mu\text{g/m}^3$ demarcated as concentration thresholds. | Chart Reference Lines & Section 3 Warnings |
| **Three-Tier Uncertainty Zoning** | Spatial map displays 3 explicit tiers: Low ($\le 5\,\text{km}$, 16.8%), Moderate ($5\text{--}15\,\text{km}$, 48.3%), and High Extrapolation ($> 15\,\text{km}$, 34.9%). | Section 4 Map Overlay & Metric Counters |
| **Non-Causal TreeSHAP** | Prominent warning that SHAP attributions represent statistical predictive associations, not direct physical or traffic causality. | Section 5 Callout Banner |

---

## 4. Verification & Testing Evidence

```
$env:PYTHONPATH="."; python -m pytest new_research_project/implementation/tests -v
============================= test session starts =============================
platform win32 -- Python 3.13.11, pytest-9.1.1, pluggy-1.5.0
collected 64 items

new_research_project/implementation/tests/test_dashboard.py (8 tests) ........... PASSED
new_research_project/implementation/tests/test_explainability.py (3 tests) ...... PASSED
new_research_project/implementation/tests/test_feature_engineering.py (18 tests)  PASSED
new_research_project/implementation/tests/test_ingestion.py (7 tests) ........... PASSED
new_research_project/implementation/tests/test_leakage_safety.py (2 tests) ...... PASSED
new_research_project/implementation/tests/test_missing_values.py (5 tests) ...... PASSED
new_research_project/implementation/tests/test_modeling.py (6 tests) ............ PASSED
new_research_project/implementation/tests/test_partitioning.py (2 tests) ........ PASSED
new_research_project/implementation/tests/test_reproducibility.py (1 test) ...... PASSED
new_research_project/implementation/tests/test_spatial.py (8 tests) ............. PASSED
new_research_project/implementation/tests/test_target_generation.py (4 tests) ... PASSED

============================= 64 passed in 50.99s =============================
```

---

## 5. Execution Instructions

To launch the research dashboard locally:
```powershell
# From project directory
streamlit run new_research_project/dashboard/app.py

# Alternatively, from root launcher
streamlit run app.py
```
Open `http://localhost:8501` in any modern web browser.
