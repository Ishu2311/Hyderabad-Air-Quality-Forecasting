# Milestone 4 — Spatial Modeling, Interpolation and Explainability Report

**Document Code:** `REPORT-M4-SPATIAL-MODELING`  
**Project Title:** Traffic-Aware Spatio-Temporal PM2.5 Concentration and Hotspot Forecasting in Hyderabad Using Multi-Source Environmental Data and Explainable Machine Learning  
**Milestone:** Milestone 4 – Spatial Hotspot Mapping, Interpolation and SHAP Explainability  
**Status:** **COMPLETED AND VERIFIED (56/56 Tests Passing)**  

---

## 1. Executive Summary

Milestone 4 establishes the spatial interpolation, hotspot identification, and explainable machine learning architecture for the Hyderabad air quality forecasting framework. Building upon the verified temporal forecasting models from Milestone 3, Milestone 4 implements:
1. **Dual Spatial Interpolation:** Inverse Distance Weighting (IDW with configurable decay power $p \in \{1, 2, 3\}$) and Ordinary Kriging (Best Linear Unbiased Estimator under Spherical, Exponential, and Gaussian theoretical variograms).
2. **Leave-One-Station-Out Spatial Cross-Validation (LOSOCV):** Rigorous spatial transfer benchmarking across all 4 continuous ambient air quality monitoring stations (CAAQMS) and 4 forecasting horizons (+1h, +6h, +24h, +48h) over the 26,016 station-hours of the 2026 test partition.
3. **Statistical Identifiability Forensic Audit:** Quantitative proof that Ordinary Kriging is statistically under-determined on a 4-station network ($\binom{4}{2} = 6$ distance pairs; $\binom{3}{2} = 3$ pairs under cross-validation), causing IDW ($p=2$) to systematically outperform Kriging in spatial generalization across all 16 station-horizon evaluations.
4. **Multi-Tier Spatial Uncertainty Delineation:** Geometric distance zoning (Proximate $\le 5\,\text{km}$, Intermediate $5–15\,\text{km}$, Extrapolation $> 15\,\text{km}$) combined with theoretical Kriging estimation variance $\sigma_k^2(\mathbf{s}_0)$ surfaces.
5. **TreeSHAP Explainability & Composite Stacking Attribution:** Complete feature attribution across all 84 approved primary predictors. Proves that autoregressive persistence dominates short-term nowcasting (79.6% SHAP share at +1h), while atmospheric thermodynamics and boundary layer dynamics expand tenfold (from 1.7% at +1h to 16.8% at +48h).
6. **Dynamic Human-Readable Hotspot Narratives:** Dynamic natural-language generation attributing model forecasts to specific meteorological and persistence drivers while strictly preserving the boundary between statistical association and physical causation.

---

## 2. Spatial Infrastructure & Monitoring Geometry

### 2.1 Physical Station Coordinates (UTM Zone 44N & WGS84)
The spatial interpolation grid operates within projected Universal Transverse Mercator (UTM Zone 44N, EPSG:32644) coordinates, preserving metric Euclidean distances across the Hyderabad metropolitan region:

| Station Name | Station ID | Latitude (°N) | Longitude (°E) | Easting UTM44N (m) | Northing UTM44N (m) | Typology & Micro-Environment |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Bollaram Industrial Area** | `HYD_BOL` | 17.5400 | 78.3500 | 218,650.0 | 1,941,800.0 | Heavy Industrial / Pharmaceutical Corridor |
| **Central University** | `HYD_UOH` | 17.4600 | 78.3300 | 216,420.0 | 1,932,950.0 | Institutional Campus / Vegetative Background |
| **ECIL Kapra** | `HYD_ECL` | 17.4700 | 78.5700 | 241,880.0 | 1,933,850.0 | Dense Commercial / Mixed Urban Residential |
| **ICRISAT Patancheru** | `HYD_ICR` | 17.5100 | 78.2700 | 210,180.0 | 1,938,550.0 | Peri-Urban / Agricultural-Industrial Boundary |

### 2.2 Network Geometry & Inter-Station Distances
The 4 monitoring stations yield exactly $\binom{4}{2} = 6$ distinct pairwise spatial distances:
- **ICRISAT Patancheru $\longleftrightarrow$ Central University:** $8.38\,\text{km}$
- **Bollaram Industrial $\longleftrightarrow$ ICRISAT Patancheru:** $9.07\,\text{km}$
- **Bollaram Industrial $\longleftrightarrow$ Central University:** $9.13\,\text{km}$
- **Bollaram Industrial $\longleftrightarrow$ ECIL Kapra:** $24.55\,\text{km}$
- **Central University $\longleftrightarrow$ ECIL Kapra:** $25.48\,\text{km}$
- **ECIL Kapra $\longleftrightarrow$ ICRISAT Patancheru:** $32.05\,\text{km}$

> [!IMPORTANT]
> **Spatial Network Asymmetry:** Three stations (Bollaram, Central University, ICRISAT Patancheru) form a tightly clustered triangular cluster in the north-western sector of Hyderabad (inter-station distances $< 9.2\,\text{km}$). Conversely, ECIL Kapra is isolated $25$ to $32\,\text{km}$ to the east. The southern and central urban cores contain zero active CPCB CAAQMS stations during the study period.

---

## 3. Spatial Interpolation Formulations

### 3.1 Inverse Distance Weighting (IDW)
For target spatial cell $\mathbf{s}_0 = (x_0, y_0)$ and $K$ reporting monitoring stations with values $Z(\mathbf{s}_k)$:
$$\hat{Z}_{\text{IDW}}(\mathbf{s}_0) = \begin{cases} Z(\mathbf{s}_k) & \text{if } \|\mathbf{s}_0 - \mathbf{s}_k\|_2 < \epsilon \\ \frac{\sum_{k=1}^K w_k(\mathbf{s}_0) Z(\mathbf{s}_k)}{\sum_{k=1}^K w_k(\mathbf{s}_0)} & \text{otherwise} \end{cases}$$
where:
$$w_k(\mathbf{s}_0) = \frac{1}{\|\mathbf{s}_0 - \mathbf{s}_k\|_2^p}, \quad p \in \{1.0, 2.0, 3.0\}, \quad \epsilon = 10^{-5}\,\text{m}$$

### 3.2 Ordinary Kriging (BLUE)
Ordinary Kriging models the spatial concentration field as an intrinsically stationary random process $Z(\mathbf{s}) = \mu + \varepsilon(\mathbf{s})$ with semivariogram $\gamma(h) = \frac{1}{2} E[(Z(\mathbf{s}) - Z(\mathbf{s}+h))^2]$.
The linear unbiased estimator $\hat{Z}(\mathbf{s}_0) = \sum_{k=1}^K w_k Z(\mathbf{s}_k)$ with $\sum w_k = 1$ is obtained by solving the Kriging system:
$$\begin{bmatrix} \boldsymbol{\Gamma} & \mathbf{1} \\ \mathbf{1}^T & 0 \end{bmatrix} \begin{bmatrix} \mathbf{w} \\ \lambda \end{bmatrix} = \begin{bmatrix} \boldsymbol{\gamma}_0 \\ 1 \end{bmatrix}$$
where $\boldsymbol{\Gamma}_{jk} = \gamma(\|\mathbf{s}_j - \mathbf{s}_k\|)$, $\boldsymbol{\gamma}_{0, k} = \gamma(\|\mathbf{s}_k - \mathbf{s}_0\|)$, and $\lambda$ is the Lagrange multiplier.
The theoretical Kriging estimation variance is:
$$\sigma_k^2(\mathbf{s}_0) = \mathbf{w}^T \boldsymbol{\gamma}_0 + \lambda$$

---

## 4. Key Scientific Results

### 4.1 Leave-One-Station-Out Spatial Cross-Validation (2026 Test Partition)
Benchmarked across 26,016 station-hours using the winning Milestone 3 models (Stacking at +1h/+6h; LightGBM at +24h/+48h):

| Forecasting Horizon | Held-Out Station | Observed Mean ($\mu\text{g/m}^3$) | IDW ($p=2$) MAE | IDW ($p=2$) RMSE | IDW ($p=2$) Bias | Kriging (Exp) MAE | Kriging (Exp) RMSE | Spatial Superiority |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **+1h** | Bollaram Industrial | 59.45 | **13.14** | **16.80** | -11.52 | 14.10 | 18.02 | **IDW (+6.8% lower MAE)** |
| **+1h** | Central University | 28.77 | **12.84** | 14.85 | +12.00 | 12.99 | **14.82** | **IDW (+1.2% lower MAE)** |
| **+1h** | ECIL Kapra | 33.36 | **11.67** | **15.20** | -2.42 | 11.87 | 15.25 | **IDW (+1.7% lower MAE)** |
| **+1h** | ICRISAT Patancheru | 41.97 | **9.16** | **13.50** | -0.22 | 9.92 | 13.82 | **IDW (+7.7% lower MAE)** |
| **+6h** | Bollaram Industrial | 59.44 | **12.36** | **15.93** | -9.31 | 13.51 | 17.20 | **IDW (+8.5% lower MAE)** |
| **+6h** | Central University | 28.77 | **14.19** | 16.24 | +13.50 | 14.27 | **16.13** | **IDW (+0.6% lower MAE)** |
| **+6h** | ECIL Kapra | 33.37 | **10.83** | **14.65** | -0.64 | 11.02 | 14.68 | **IDW (+1.7% lower MAE)** |
| **+6h** | ICRISAT Patancheru | 41.95 | **9.80** | **13.63** | +1.42 | 10.58 | 14.03 | **IDW (+7.4% lower MAE)** |
| **+24h** | Bollaram Industrial | 59.42 | **12.85** | **16.64** | -9.48 | 13.83 | 17.73 | **IDW (+7.1% lower MAE)** |
| **+24h** | Central University | 28.76 | **14.69** | 16.64 | +13.87 | 14.81 | **16.62** | **IDW (+0.8% lower MAE)** |
| **+24h** | ECIL Kapra | 33.37 | **9.88** | **13.67** | -0.30 | 10.09 | 13.76 | **IDW (+2.1% lower MAE)** |
| **+24h** | ICRISAT Patancheru | 41.97 | **10.43** | **14.08** | +1.90 | 11.24 | 14.56 | **IDW (+7.2% lower MAE)** |
| **+48h** | Bollaram Industrial | 59.40 | **12.83** | **16.42** | -8.45 | 13.74 | 17.45 | **IDW (+6.6% lower MAE)** |
| **+48h** | Central University | 28.76 | **15.32** | 17.37 | +14.49 | 15.42 | **17.30** | **IDW (+0.6% lower MAE)** |
| **+48h** | ECIL Kapra | 33.38 | **9.26** | **13.15** | +0.39 | 9.48 | 13.27 | **IDW (+2.3% lower MAE)** |
| **+48h** | ICRISAT Patancheru | 41.97 | **11.13** | **14.59** | +2.54 | 11.86 | 15.03 | **IDW (+6.2% lower MAE)** |

### 4.2 Core Spatial Insights
1. **IDW Superiority Over Kriging:** Across all 16 evaluations, IDW ($p=2$) achieves lower MAE than Ordinary Kriging. Because the remaining network under leave-one-out has only $K=3$ stations, empirical variogram fitting has only 3 spatial distance points. Kriging matrices suffer from ill-conditioning and over-extrapolation, whereas IDW's deterministic power law provides smooth, bounded distance regularization.
2. **Micro-Climatic Spatial Transfer Biases:**
   - **Central University:** Experiences severe positive interpolation bias ($+12.0$ to $+14.5\,\mu\text{g/m}^3$) because its clean vegetative micro-environment is interpolated from surrounding commercial and industrial stations.
   - **Bollaram Industrial Area:** Experiences severe negative interpolation bias ($-8.5$ to $-11.5\,\mu\text{g/m}^3$) because its localized heavy industrial emissions are smoothed out when interpolated from cleaner peripheral stations.
3. **Temporal Forecast vs. Spatial Transfer Accuracy:** While temporal point forecasts achieve MAE of $3.01\,\mu\text{g/m}^3$ at +1h and $6.67\,\mu\text{g/m}^3$ at +24h, spatial interpolation transfer error is $9.2$ to $15.3\,\mu\text{g/m}^3$. This proves that a 4-station network cannot resolve intra-urban neighborhood micro-climates without dense local physical monitoring.

---

## 5. SHAP Explainability & Physical Attribution

### 5.1 Domain Importance Shift Across Horizons
Analysis of mean absolute SHAP values across the 84 approved primary predictors reveals how predictive drivers evolve as forecast horizon expands:

| Feature Domain | +1h Attribution Share | +6h Attribution Share | +24h Attribution Share | +48h Attribution Share | Temporal Trend |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Historical PM2.5 Dynamics** | **79.6%** ($13.69\,\mu\text{g/m}^3$) | **62.3%** ($13.08\,\mu\text{g/m}^3$) | **62.5%** ($11.44\,\mu\text{g/m}^3$) | **49.3%** ($10.12\,\mu\text{g/m}^3$) | Decays steadily as persistence weakens |
| **Thermodynamics & Boundary Layer** | **1.7%** ($0.29\,\mu\text{g/m}^3$) | **10.0%** ($2.10\,\mu\text{g/m}^3$) | **12.2%** ($2.24\,\mu\text{g/m}^3$) | **16.8%** ($3.44\,\mu\text{g/m}^3$) | **Expands 10-fold** as synoptic weather dominates |
| **Co-Pollutants / Chemical Ratios** | **11.4%** ($1.96\,\mu\text{g/m}^3$) | **6.1%** ($1.28\,\mu\text{g/m}^3$) | **7.1%** ($1.30\,\mu\text{g/m}^3$) | **9.3%** ($1.91\,\mu\text{g/m}^3$) | Stable secondary chemical precursor signal |
| **Wind & Dispersion Dynamics** | **2.0%** ($0.35\,\mu\text{g/m}^3$) | **6.7%** ($1.40\,\mu\text{g/m}^3$) | **7.0%** ($1.29\,\mu\text{g/m}^3$) | **5.5%** ($1.13\,\mu\text{g/m}^3$) | Triples from nowcasting to daily forecasting |
| **Cyclical Calendar & Solar Timing** | **2.9%** ($0.50\,\mu\text{g/m}^3$) | **9.2%** ($1.93\,\mu\text{g/m}^3$) | **4.2%** ($0.77\,\mu\text{g/m}^3$) | **5.4%** ($1.10\,\mu\text{g/m}^3$) | Peaks at +6h capturing diurnal morning/evening cycle |
| **Other Environmental Predictors** | **2.4%** ($0.41\,\mu\text{g/m}^3$) | **5.8%** ($1.21\,\mu\text{g/m}^3$) | **6.9%** ($1.26\,\mu\text{g/m}^3$) | **13.8%** ($2.84\,\mu\text{g/m}^3$) | Synoptic pressure and multi-day trends |

### 5.2 Top 5 Global Predictors Across Horizons
1. **Horizon +1h (Stacking Ensemble):**
   - 1. `PM2.5` (Lag 0 origin observation): $9.589\,\mu\text{g/m}^3$ (Autoregressive persistence)
   - 2. `pm25_roll_mean_3h`: $1.533\,\mu\text{g/m}^3$ (Short-term accumulation)
   - 3. `pm25_roll_mean_24h`: $0.960\,\mu\text{g/m}^3$ (Diurnal baseline loading)
   - 4. `PM10`: $0.752\,\mu\text{g/m}^3$ (Coarse particulate fraction)
   - 5. `pm25_trend_diff_1h`: $0.445\,\mu\text{g/m}^3$ (Instantaneous acceleration)
2. **Horizon +6h (Stacking Ensemble):**
   - 1. `pm25_roll_mean_24h`: $6.976\,\mu\text{g/m}^3$
   - 2. `PM2.5`: $2.626\,\mu\text{g/m}^3$
   - 3. `pressure_tendency_3h`: $1.072\,\mu\text{g/m}^3$ (Barometric synoptic change)
   - 4. `hour_sin`: $0.955\,\mu\text{g/m}^3$ (Diurnal solar timing)
   - 5. `pm25_roll_mean_12h`: $0.687\,\mu\text{g/m}^3$
3. **Horizon +24h (LightGBM):**
   - 1. `pm25_roll_mean_3h`: $3.456\,\mu\text{g/m}^3$
   - 2. `PM2.5`: $3.126\,\mu\text{g/m}^3$
   - 3. `pm25_roll_mean_24h`: $1.828\,\mu\text{g/m}^3$
   - 4. `pm25_lag_24`: $1.094\,\mu\text{g/m}^3$ (Exact 24h seasonal lag)
   - 5. `surface_pressure`: $0.600\,\mu\text{g/m}^3$
4. **Horizon +48h (LightGBM):**
   - 1. `PM2.5`: $2.619\,\mu\text{g/m}^3$
   - 2. `pm25_roll_mean_3h`: $1.946\,\mu\text{g/m}^3$
   - 3. `pm25_roll_mean_24h`: $1.294\,\mu\text{g/m}^3$
   - 4. `surface_pressure`: $1.163\,\mu\text{g/m}^3$ (Synoptic air mass stagnation)
   - 5. `pm25_roll_min_24h`: $1.021\,\mu\text{g/m}^3$

---

## 6. Generated Cartographic & Explainability Figures Inventory

All 31 figures were generated at 300 DPI and saved in [reports/figures/](file:///C:/Users/harsha%20madala/Documents/Hyderabad_Air_Research/new_research_project/reports/figures/):

### 6.1 Spatial Prediction & Hotspot Maps
- **+1h Spatial Surface (IDW):** [spatial_forecast_h1_idw.png](file:///C:/Users/harsha%20madala/Documents/Hyderabad_Air_Research/new_research_project/reports/figures/spatial_forecast_h1_idw.png)
- **+1h Spatial Surface (Kriging):** [spatial_forecast_h1_kriging.png](file:///C:/Users/harsha%20madala/Documents/Hyderabad_Air_Research/new_research_project/reports/figures/spatial_forecast_h1_kriging.png)
- **+6h Spatial Surface (IDW):** [spatial_forecast_h6_idw.png](file:///C:/Users/harsha%20madala/Documents/Hyderabad_Air_Research/new_research_project/reports/figures/spatial_forecast_h6_idw.png)
- **+24h Spatial Surface (IDW):** [spatial_forecast_h24_idw.png](file:///C:/Users/harsha%20madala/Documents/Hyderabad_Air_Research/new_research_project/reports/figures/spatial_forecast_h24_idw.png)
- **+48h Spatial Surface (IDW):** [spatial_forecast_h48_idw.png](file:///C:/Users/harsha%20madala/Documents/Hyderabad_Air_Research/new_research_project/reports/figures/spatial_forecast_h48_idw.png)
- **IDW vs Kriging Comparison (+1h):** [idw_vs_kriging_comparison_h1.png](file:///C:/Users/harsha%20madala/Documents/Hyderabad_Air_Research/new_research_project/reports/figures/idw_vs_kriging_comparison_h1.png)
- **IDW vs Kriging Comparison (+24h):** [idw_vs_kriging_comparison_h24.png](file:///C:/Users/harsha%20madala/Documents/Hyderabad_Air_Research/new_research_project/reports/figures/idw_vs_kriging_comparison_h24.png)
- **Spatial Uncertainty Surface (+1h):** [spatial_uncertainty_h1.png](file:///C:/Users/harsha%20madala/Documents/Hyderabad_Air_Research/new_research_project/reports/figures/spatial_uncertainty_h1.png)
- **Spatial Uncertainty Surface (+24h):** [spatial_uncertainty_h24.png](file:///C:/Users/harsha%20madala/Documents/Hyderabad_Air_Research/new_research_project/reports/figures/spatial_uncertainty_h24.png)
- **Hotspot Delineation Map (+1h):** [hotspot_delineation_h1.png](file:///C:/Users/harsha%20madala/Documents/Hyderabad_Air_Research/new_research_project/reports/figures/hotspot_delineation_h1.png)
- **Hotspot Delineation Map (+24h):** [hotspot_delineation_h24.png](file:///C:/Users/harsha%20madala/Documents/Hyderabad_Air_Research/new_research_project/reports/figures/hotspot_delineation_h24.png)

### 6.2 SHAP Attribution Figures
- **Global Feature Importance (+1h):** [shap_global_importance_h1.png](file:///C:/Users/harsha%20madala/Documents/Hyderabad_Air_Research/new_research_project/reports/figures/shap_global_importance_h1.png)
- **Global Feature Importance (+24h):** [shap_global_importance_h24.png](file:///C:/Users/harsha%20madala/Documents/Hyderabad_Air_Research/new_research_project/reports/figures/shap_global_importance_h24.png)
- **SHAP Beeswarm Summary (+1h):** [shap_beeswarm_h1.png](file:///C:/Users/harsha%20madala/Documents/Hyderabad_Air_Research/new_research_project/reports/figures/shap_beeswarm_h1.png)
- **SHAP Beeswarm Summary (+24h):** [shap_beeswarm_h24.png](file:///C:/Users/harsha%20madala/Documents/Hyderabad_Air_Research/new_research_project/reports/figures/shap_beeswarm_h24.png)
- **SHAP Dependence Plot (PM2.5 vs Temp):** [shap_dependence_h1_PM2.5.png](file:///C:/Users/harsha%20madala/Documents/Hyderabad_Air_Research/new_research_project/reports/figures/shap_dependence_h1_PM2.5.png)
- **SHAP Dependence Plot (Rolling Mean vs RH):** [shap_dependence_h24_pm25_roll_mean_24h.png](file:///C:/Users/harsha%20madala/Documents/Hyderabad_Air_Research/new_research_project/reports/figures/shap_dependence_h24_pm25_roll_mean_24h.png)
- **Local Waterfall Plot (ECIL Kapra +1h):** [shap_waterfall_ECIL_Kapra_h1_sample10.png](file:///C:/Users/harsha%20madala/Documents/Hyderabad_Air_Research/new_research_project/reports/figures/shap_waterfall_ECIL_Kapra_h1_sample10.png)
- **Local Waterfall Plot (Bollaram +24h):** [shap_waterfall_Bollaram_Industrial_h24_sample25.png](file:///C:/Users/harsha%20madala/Documents/Hyderabad_Air_Research/new_research_project/reports/figures/shap_waterfall_Bollaram_Industrial_h24_sample25.png)

---

## 7. Dynamic Human-Readable Hotspot Narratives

Dynamic narratives synthesized from model predictions, TreeSHAP attributions, and spatial uncertainty diagnostics:

### Case Study 1: ECIL Kapra (+1h Horizon)
> "ECIL Kapra is predicted to remain within the statutory 24-hour NAAQS benchmark (60 µg/m³) over the next 1 to 6 hours (forecast: 30.9 µg/m³, compared to a recent baseline of 31.1 µg/m³).
> 
> The upward elevation in predicted PM2.5 is primarily driven by high recent persistence from the preceding hour (+8.4 µg/m³), accumulated diurnal pollutant loading across prior hours (+1.8 µg/m³), and local thermodynamic inversion conditions (+0.6 µg/m³).
> 
> Downward mitigation is exerted by active atmospheric ventilation mitigating concentration build-up (-2.1 µg/m³) and convective vertical dilution into a deeper boundary layer (-1.4 µg/m³).
> 
> Methodological Note: Feature contributions are quantified via TreeSHAP and reflect statistical predictive associations rather than proven direct physical causation. Diurnal window variables capture historical cyclical correlations but do not constitute direct vehicle telemetry measurements.
> 
> Spatial Interpretation: This prediction reflects Low Uncertainty (Proximate <= 5km). Spatial interpolation beyond physical monitoring locations carries inherent interpolation variance and should be interpreted alongside the 4-station network geometry limitations."

### Case Study 2: Bollaram Industrial Area (+24h Horizon)
> "Bollaram Industrial Area is predicted to experience elevated PM2.5 concentrations approaching the statutory 24-hour NAAQS benchmark (forecast: 59.5 µg/m³, compared to a recent baseline of 61.0 µg/m³).
> 
> The upward elevation in predicted PM2.5 is primarily driven by accumulated diurnal pollutant loading across prior hours (+4.2 µg/m³), high recent persistence (+3.1 µg/m³), and reduced horizontal ventilation under boundary-layer wind stagnation (+1.8 µg/m³).
> 
> Downward mitigation is exerted by negative historical trajectory in recent telemetry (-1.9 µg/m³) and daytime solar convective boundary layer growth (-1.2 µg/m³).
> 
> Methodological Note: Feature contributions represent TreeSHAP attributions from LightGBM. Industrial emissions from pharmaceutical operations in Bollaram are reflected via physical station persistence and co-pollutant ratios, not direct stack emissions telemetry.
> 
> Spatial Interpretation: This prediction reflects Low Uncertainty (Proximate <= 5km) at the sensor centroid, but spatial interpolation eastward toward central Hyderabad incurs high extrapolation uncertainty."

---

## 8. Milestone 4 Verification & Stop Condition

### Status: 100% Complete
- **Modular Code:** Implemented in `implementation/spatial/` and `implementation/explainability/`.
- **Automated Test Suite:** **56/56 tests passing** (`pytest implementation/tests -v`).
- **Reports Generated:** All 6 mandated reports completed and synchronized.
- **Strict Stop Condition:** No dashboard development, web deployment, or Milestone 5 tasks have been started. Awaiting explicit user direction.
