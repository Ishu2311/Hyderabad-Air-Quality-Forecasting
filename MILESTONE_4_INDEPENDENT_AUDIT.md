# MILESTONE 4 — INDEPENDENT SCIENTIFIC AUDIT
## Spatial Hotspot Mapping, Interpolation, and SHAP Explainability

**Project Title:** Traffic-Aware Spatio-Temporal $\text{PM}_{2.5}$ Concentration and Hotspot Forecasting in Hyderabad Using Multi-Source Environmental Data and Explainable Machine Learning  
**Audit Stage:** Milestone 4 Independent Scientific Audit (Prior to Milestone 5 Authorization)  
**Execution Mode:** Read-Only Forensic Verification (Zero Code, Dataset, or Model Modifications)  
**Date of Audit:** November 15, 2026  
**Auditor:** Independent AI Research Systems Auditor (Google DeepMind Antigravity)

---

## Executive Summary & Audit Classification Matrix

An exhaustive, evidence-based independent scientific audit was performed on the code, artifacts, data predictions, and reports delivered for **Milestone 4 (Spatial Hotspot Mapping, Interpolation, and SHAP Explainability)**. In accordance with strict auditing mandates, **no source code, model parameters, prediction tables, thresholds, or reports were modified** during this audit. Every check was validated against actual Parquet datasets, JSON prediction files, model weights, serialized arrays, and test logs on disk.

The overall status of Milestone 4 is classified as **PASS WITH LIMITATIONS / APPROVED WITH CONDITIONS**. The spatial mapping framework, Leave-One-Station-Out Cross-Validation (LOSOCV) protocol, TreeSHAP explainability pipeline, and automated test suite (56/56 passing tests) are methodologically rigorous and fully reproducible. Two documentation-level discrepancies were identified (an inaccurate uniform sample count of 1,875 in Table 2.1 of `SPATIAL_VALIDATION_REPORT.md` and an undocumented 25-tree subsampling optimization in the Stacking Random Forest SHAP routine), both of which are documented herein alongside concrete, non-breaking remediation recommendations.

### Summary Classification Matrix

| Audit Dimension | Core Verification Check | Audit Status | Forensic Evidence & File References |
| :--- | :--- | :--- | :--- |
| **1. Spatial Validation Counts** | Reconciliation of 1,875 vs. 7,500 vs. 26,016 test hours and per-station sample accounting | **PASS WITH LIMITATIONS** | `new_research_project/data/predictions/spatial_validation_results.json`<br>`new_research_project/reports/SPATIAL_VALIDATION_REPORT.md` |
| **2. IDW Interpolation** | Distance metric (UTM Zone 44N), formula ($p=2.0$), zero-distance handling, and 390% spatial transfer penalty | **PASS** | `new_research_project/implementation/spatial/idw.py`<br>`new_research_project/reports/IDW_VS_KRIGING_COMPARISON.md` |
| **3. Kriging & Variogram** | Variogram identifiability ($N=4$), condition numbers ($>10^{12}$), Moore-Penrose pseudo-inversion, variance interpretation | **PASS WITH LIMITATIONS** | `new_research_project/implementation/spatial/kriging.py`<br>`new_research_project/reports/SPATIAL_UNCERTAINTY_REPORT.md` |
| **4. Spatial Uncertainty** | Distance-to-sensor zoning (16.8% Low, 48.3% Moderate, 34.9% High extrapolation) and unmonitored urban sectors | **PASS** | `new_research_project/reports/SPATIAL_UNCERTAINTY_REPORT.md`<br>`new_research_project/reports/figures/spatial_uncertainty_zones.png` |
| **5. Hotspot Claims** | Regulatory ($\ge 60\,\mu\text{g/m}^3$) vs exploratory ($\ge 120\,\mu\text{g/m}^3$) thresholds and 0% extrapolation-risk hotspot proof | **PASS** | `new_research_project/data/predictions/hotspot_detection_summary.json`<br>`new_research_project/reports/HOTSPOT_INTERPRETATION_REPORT.md` |
| **6. TreeSHAP Explainability** | Stacking composite attribution, additive reconstruction fidelity, feature groups, and exclusion of unverified road proxies | **PASS WITH LIMITATIONS** | `new_research_project/data/predictions/shap_values_h*.npz`<br>`new_research_project/reports/SHAP_EXPLAINABILITY_REPORT.md` |
| **7. Software Integrity** | Full test suite execution across all modules (56 passing unit and integration tests) | **PASS** | `pytest new_research_project/implementation/tests -v`<br>56 passed in 24.96 seconds |

---

## 1. Reconciliation of Spatial Validation Counts

### 1.1 Forensic Analysis of Reported Figures

Three numbers appeared in previous Milestone 4 documentation: **26,016 test hours**, **1,875 test hours**, and **7,500 evaluations**. The audit traced the exact genesis and mathematical meaning of each figure:

1. **26,016 Test Hours (The Domain Test Partition Size):**
   - The test partition spans `2026-01-01 00:00:00` to `2026-09-28 23:00:00`, totaling exactly **6,504 hourly chronological timestamps**.
   - With 4 CAAQMS monitoring stations in the network, the total theoretical cross-product of station-hours is:
     $$\text{Total Test Station-Hours} = 6,504 \text{ timestamps} \times 4 \text{ stations} = 26,016 \text{ station-hours}$$
   - This represents the complete spatio-temporal volume of the test evaluation window.

2. **1,875 Test Hours (Bollaram Sensor Data Availability):**
   - Due to extended physical sensor telemetry downtime and quality control flagging at the Bollaram Industrial Area CAAQMS station during the 2026 test period, exactly **1,875 valid hourly observations** exist where Bollaram had non-null, unflagged ground-truth physical target measurements at the $+1\text{h}$ horizon.
   - The remaining 4,629 hours at Bollaram were null or flagged and properly excluded from loss calculation in accordance with zero-target-imputation protocols.

3. **7,500 Evaluations per Horizon (Reporting Artifact in Table 2.1):**
   - A theoretical figure obtained by multiplying Bollaram's 1,875 valid hours across all four stations ($1,875 \times 4 = 7,500$).
   - **Audit Finding:** In `SPATIAL_VALIDATION_REPORT.md` (Table 2.1), the `Samples (N)` column mistakenly listed `1,875` for *all four stations*. However, physical inspection of `feature_store_test_purged.parquet` and `spatial_validation_results.json` reveals that the other three stations operated with far higher sensor uptime.

### 1.2 Actual Station-Wise and Horizon-Wise Evaluation Counts

Direct query of `spatial_validation_results.json` and the test prediction Parquet archives established the true empirical sample accounting:

| Forecast Horizon | Bollaram Industrial ($N$) | Central University ($N$) | ECIL Kapra ($N$) | ICRISAT Patancheru ($N$) | Total Station Predictions Evaluated | Synchronous 4-Station Timestamps |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **+1 Hour** | 1,875 | 5,743 | 5,849 | 5,606 | **19,073** | 1,479 |
| **+6 Hours** | 1,875 | 5,743 | 5,849 | 5,606 | **19,073** | 1,479 |
| **+24 Hours** | 1,875 | 5,743 | 5,849 | 5,606 | **19,073** | 1,479 |
| **+48 Hours** | 1,875 | 5,743 | 5,849 | 5,606 | **19,073** | 1,479 |

- **Synchronous Timestamps ($N = 1,479$):** There are exactly 1,479 hourly timestamps where *all four stations simultaneously recorded valid physical targets*. These timestamps represent complete 4-point spatial snapshots.
- **Asynchronous Evaluations ($N = 19,073$):** In Leave-One-Station-Out CV, whenever held-out station $s_k$ has a valid target, its prediction is evaluated against donors $\mathcal{S}_{-k}$ having valid values at that timestamp. If a donor station is offline, IDW and Kriging dynamically interpolate from the remaining available donors ($K \ge 2$).

### 1.3 LOSOCV Fold Isolation & Donor Separation

The audit inspected `new_research_project/implementation/spatial/spatial_validation.py` (lines 101–147). The code enforces:
$$\mathcal{S}_{\text{donor}} = \mathcal{S} \setminus \{s_k\}$$
- In every cross-validation fold, the held-out station's coordinates $(x_k, y_k)$ and target value $y_k(t)$ are strictly purged from the donor coordinate matrix and donor value vector prior to calling `IDWInterpolator.fit()` or `OrdinaryKriging.fit()`.
- **Zero spatial leakage confirmed:** Under no circumstances was the held-out station used to construct its own spatial estimate.

### 1.4 Macro-Average vs. Sample-Weighted MAE

In `SPATIAL_VALIDATION_REPORT.md`, the reported network mean IDW MAE at $+1\text{h}$ is **$11.703\,\mu\text{g/m}^3$**.
- This is the **unweighted macro-average** across the 4 station folds:
  $$\text{Macro MAE} = \frac{13.14 + 12.84 + 11.67 + 9.16}{4} = 11.7025 \approx 11.703\,\mu\text{g/m}^3$$
- If weighted by the actual station sample sizes ($N_k$):
  $$\text{Weighted MAE} = \frac{13.14(1875) + 12.84(5743) + 11.67(5849) + 9.16(5606)}{19073} = 11.429\,\mu\text{g/m}^3$$
- Both metrics are valid, but the report must explicitly clarify that $11.703\,\mu\text{g/m}^3$ is an unweighted macro-average across station typologies.

---

## 2. IDW Results & Spatial Transfer Degradation Audit

### 2.1 Coordinate Reference System & Mathematical Formulation

Inspection of `new_research_project/implementation/spatial/idw.py` confirmed:
- **Projection:** Projected metric coordinates in **UTM Zone 44N (EPSG:32644)** are used for all spatial calculations. Distances are computed as true Euclidean meters:
  $$d_i = \sqrt{(x_{\text{target}} - x_i)^2 + (y_{\text{target}} - y_i)^2}$$
- **Power Parameter:** Inverse distance power parameter $p = 2.0$, following the approved methodology:
  $$w_i = \frac{1}{(d_i + \epsilon)^p}, \quad \hat{z} = \frac{\sum_{i=1}^K w_i z_i}{\sum_{i=1}^K w_i}$$
- **Zero-Distance Handling:** Tested and confirmed in `test_spatial.py` (lines 45–62). If $d_i \le \epsilon$ ($10^{-5}\,\text{m}$), the exact coincident station value is returned directly ($w_i = 1, w_{j \ne i} = 0$), avoiding division-by-zero singularities.

### 2.2 Independent Verification of Station-Level IDW Metrics

The audit independently queried `spatial_validation_results.json` and verified all reported metrics across all four horizons:

| Horizon | Station | Actual Mean ($\mu\text{g/m}^3$) | Interp Mean ($\mu\text{g/m}^3$) | MAE ($\mu\text{g/m}^3$) | RMSE ($\mu\text{g/m}^3$) | Bias ($\mu\text{g/m}^3$) | $R^2$ | NRMSE (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **+1h** | Bollaram Industrial | 41.85 | 30.34 | **13.14** | 16.80 | **-11.52** | +0.507 | 40.13% |
| **+1h** | Central University | 17.51 | 29.51 | **12.84** | 14.85 | **+12.00** | +0.025 | 84.81% |
| **+1h** | ECIL Kapra | 28.46 | 26.04 | **11.67** | 15.20 | -2.42 | **-1.527** | 53.41% |
| **+1h** | ICRISAT Patancheru | 29.80 | 29.58 | **9.16** | 13.50 | -0.22 | +0.535 | 45.31% |
| **+1h Mean** | **Network Macro-Average** | **29.41** | **28.87** | **11.70** | **15.09** | **-0.54** | **-0.115** | **55.92%** |
| **+6h Mean** | **Network Macro-Average** | **29.41** | **28.87** | **12.39** | **16.03** | **-0.54** | **-0.246** | **59.41%** |
| **+24h Mean** | **Network Macro-Average** | **29.41** | **28.87** | **13.91** | **17.98** | **-0.54** | **-0.538** | **66.63%** |
| **+48h Mean** | **Network Macro-Average** | **29.41** | **28.87** | **14.82** | **19.12** | **-0.54** | **-0.722** | **70.85%** |

### 2.3 Physical Interpretation of the Spatial Transfer Penalty

The audit confirmed the core finding: **Point ML forecasting error is $\approx 3.01\,\mu\text{g/m}^3$ MAE, but spatial transfer error jumps to $11.70\,\mu\text{g/m}^3$ MAE—a $390\%$ error penalty.**
Forensic inspection confirms that this degradation is driven by two physical and geographic factors rather than algorithmic failure:
1. **Typological Mismatch:**
   - Central University is located in a heavily forested, clean academic enclave. When held out, its donors are heavy industrial zones (Bollaram, Patancheru), yielding a persistent positive bias of **$+12.00\,\mu\text{g/m}^3$**.
   - Bollaram is a dense pharmaceutical and chemical industrial corridor. When held out, its donors are cleaner suburban stations, yielding a persistent negative bias of **$-11.52\,\mu\text{g/m}^3$**.
2. **Geographic Isolation of Eastern Sector:**
   - ECIL Kapra is situated $24.6\,\text{km}$ east of Central University and $>33\,\text{km}$ east of Patancheru/Bollaram.
   - When ECIL is held out, the remaining donors are all clustered in the western half of the metropolitan area. The resulting $R^2$ is **$-1.527$**, proving that western industrial/university sensors cannot capture localized emission dynamics in eastern Hyderabad.

---

## 3. Kriging Variogram & Spatial Uncertainty Reliability Audit

### 3.1 Variogram Identifiability with $N=4$ Stations

The audit analyzed `OrdinaryKriging.fit_variogram()` in `new_research_project/implementation/spatial/kriging.py` (lines 75–140):
- With $N = 4$ monitoring stations, the number of unique pairwise Euclidean distances is:
  $$\frac{K(K - 1)}{2} = \frac{4(3)}{2} = 6 \text{ distance pairs}$$
- In Leave-One-Station-Out cross-validation, with $N = 3$ donor stations:
  $$\frac{3(2)}{2} = 3 \text{ distance pairs}$$
- **Mathematical Fact:** Fitting a 3-parameter non-linear theoretical variogram (nugget $c_0$, partial sill $c$, effective range $a$) to 3 or 6 empirical semivariance points is mathematically underdetermined.
- The code handles this via a robust heuristic fallback (setting nugget to $0.05 \times \text{sample variance}$, sill to $0.95 \times \text{sample variance}$, and range to $0.6 \times \text{max distance}$).
- **Verdict:** The audit confirms that the code explicitly logs this identifiability limitation (`identifiability_limitation` key in `fit_diagnostics`).

### 3.2 Matrix Ill-Conditioning and Pseudo-Inversion

In Ordinary Kriging, the $(K+1) \times (K+1)$ Kriging system matrix is:
$$\mathbf{K} = \begin{bmatrix} \gamma(d_{11}) & \dots & \gamma(d_{1K}) & 1 \\ \vdots & \ddots & \vdots & \vdots \\ \gamma(d_{K1}) & \dots & \gamma(d_{KK}) & 1 \\ 1 & \dots & 1 & 0 \end{bmatrix}$$
- When donor stations have disparate inter-station distances (e.g., Bollaram to Patancheru is $10.1\,\text{km}$, while Bollaram to ECIL is $33.4\,\text{km}$), the condition number of $\mathbf{K}$ frequently exceeds **$10^{12}$**.
- The audit verified that `OrdinaryKriging.fit()` detects condition numbers $> 10^{12}$ and automatically substitutes the **Moore-Penrose pseudo-inverse** (`scipy.linalg.pinv(K_mat, rcond=1e-10)`).
- **Physical Consequence:** Kriging theoretical estimation variance $\sigma_k^2(\mathbf{s}_0) = \mathbf{k}^T \mathbf{K}^{-1} \mathbf{k}$ in this sparse 4-station regime reflects the geometric configuration of the network and numerical regularization parameters rather than genuine micro-meteorological turbulence.
- **Audit Verdict:** Retaining Kriging as an **experimental diagnostic** while designating IDW as the primary reporting surface in `IDW_VS_KRIGING_COMPARISON.md` is scientifically sound and methodologically honest.

### 3.3 Verification of Metropolitan Uncertainty Tiers

Inspection of `SPATIAL_UNCERTAINTY_REPORT.md` and the 6,400-cell metropolitan grid ($80 \times 80$, $40\,\text{km} \times 40\,\text{km}$ domain) confirmed the exact cell and area distributions:

| Uncertainty Tier | Euclidean Distance Threshold ($d_{\min}$) | Grid Cell Count ($N_{\text{cells}}$) | Domain Area Share (%) | Representative Urban Localities |
| :--- | :---: | :---: | :---: | :--- |
| **Tier 1: Low Uncertainty** | $d_{\min} \le 5.0\,\text{km}$ | **1,075** | **16.80%** | Patancheru, Bollaram, Central Univ / Gachibowli, ECIL / Kushaiguda |
| **Tier 2: Moderate Uncertainty** | $5.0 < d_{\min} \le 15.0\,\text{km}$ | **3,091** | **48.30%** | Miyapur, Kukatpally, Madhapur, Hitec City, Jubilee Hills, Secunderabad |
| **Tier 3: High Extrapolation** | $d_{\min} > 15.0\,\text{km}$ | **2,234** | **34.90%** | Old City / Charminar, Falaknuma, Shamshabad Airport, LB Nagar, Uppal |
| **Total Domain Grid** | Full $40 \times 40\,\text{km}$ Bounding Box | **6,400** | **100.00%** | Greater Hyderabad Metropolitan Area |

The audit verified that over **one-third (34.9%)** of the metropolitan area falls outside the 15 km observation radius of any CAAQMS sensor. The documentation appropriately highlights that applying model predictions to unmonitored sectors (such as Charminar/Old City or Shamshabad Airport) represents high-variance spatial extrapolation.

---

## 4. Hotspot Concentration Thresholds & Risk Verification

### 4.1 Regulatory vs. Exploratory Thresholds

The audit verified the scientific justification for the two concentration thresholds applied in `hotspot_detection.py`:
1. **Operational Acute Excursion Threshold ($\ge 60\,\mu\text{g/m}^3$):**
   - Corresponds to the official **CPCB 24-hour National Ambient Air Quality Standard (NAAQS)** for $\text{PM}_{2.5}$ in India.
   - Concentrations above $60\,\mu\text{g/m}^3$ represent a statutory air quality violation requiring municipal alert.
2. **Exploratory Severe Spike Threshold ($\ge 120\,\mu\text{g/m}^3$):**
   - Exactly **$2 \times$ the statutory NAAQS limit**, demarcating emergency hazardous pollution episodes.
   - Identified in the research methodology as an exploratory benchmark for acute intervention triggers.

### 4.2 Verification of the "0% Extrapolation-Risk Hotspots" Claim

In `HOTSPOT_INTERPRETATION_REPORT.md` and `hotspot_detection_summary.json`, it was reported that **0.0% of detected hotspots fell into the high-uncertainty extrapolation zone ($> 15\,\text{km}$)**. The audit performed an independent spatial coordinate check to determine whether this claim is mathematically true:

- **Empirical Hotspot Delineation Results (from `hotspot_detection_summary.json`):**
  - **$+1\text{h}$ Forecast:** 2 acute excursion cells (0.03% domain area). Distance to nearest sensor: $0.15\,\text{km}$ (immediately adjacent to Patancheru).
  - **$+6\text{h}$ Forecast:** 0 acute excursion cells (0.0% domain area).
  - **$+24\text{h}$ Forecast:** 1,435 acute excursion cells (22.42% domain area). All 1,435 cells are located in the northwest corridor around Bollaram and Patancheru (734 proximate $\le 5\,\text{km}$, 701 intermediate $5\text{--}15\,\text{km}$, **0 extrapolation $> 15\,\text{km}$**).
  - **$+48\text{h}$ Forecast:** 1,116 acute excursion cells (17.44% domain area). All 1,116 cells are located in the northwest corridor (571 proximate $\le 5\,\text{km}$, 545 intermediate $5\text{--}15\,\text{km}$, **0 extrapolation $> 15\,\text{km}$**).
  - **Severe Spikes ($\ge 120\,\mu\text{g/m}^3$):** 0 cells across all four horizons.

- **Mathematical Proof of the 0% Extrapolation Claim:**
  1. Acute excursions ($\ge 60\,\mu\text{g/m}^3$) only occur when the northwest industrial stations (Bollaram predicted $62.4\,\mu\text{g/m}^3$, ICRISAT predicted $67.3\,\mu\text{g/m}^3$) are elevated.
  2. The remaining stations to the south and east (Central Univ at $43.9\,\mu\text{g/m}^3$, ECIL at $37.5\,\mu\text{g/m}^3$) are compliant ($< 60\,\mu\text{g/m}^3$).
  3. Under inverse distance weighting ($p=2.0$), interpolated values decay rapidly toward the compliant regional background as distance from Bollaram/Patancheru increases.
  4. By the time a grid cell is $> 15\,\text{km}$ away from Bollaram/Patancheru, the weighting from the clean/compliant stations pulls the interpolated concentration well below the $60\,\mu\text{g/m}^3$ threshold.
  5. Therefore, no cell with $\hat{z} \ge 60\,\mu\text{g/m}^3$ can exist $> 15\,\text{km}$ from a sensor.
- **Audit Verdict:** The claim is **mathematically verified** and physically consistent with continuous inverse distance attenuation.

### 4.3 Continuous Interpolation vs. Discrete Classification

The audit verified whether hotspot maps were derived from continuous regression interpolation or classification models:
- Hotspot delineations are constructed by performing **continuous 2D spatial interpolation** of the ensemble point regression forecasts across the $80 \times 80$ grid, and subsequently applying the $\ge 60\,\mu\text{g/m}^3$ threshold to contour the continuous concentration field.
- This approach avoids artificial spatial step-discontinuities and provides smooth concentration gradients.

---

## 5. SHAP Explainability & Additive Reconstruction Audit

### 5.1 TreeSHAP Formulation & Winning Model Attribution

The explainability pipeline in `new_research_project/implementation/explainability/shap_analyzer.py` handles model attribution across horizons:
- **Horizons $+24\text{h}$ and $+48\text{h}$:** LightGBM was the definitive winning regressor. Direct `shap.TreeExplainer(lgb_model)` was used.
- **Horizons $+1\text{h}$ and $+6\text{h}$:** Stacking Meta-Regression (Bayesian Ridge combining Random Forest, XGBoost, and LightGBM) was the winning model.
- Because the meta-regressor is an affine combination $\hat{y}_{\text{stack}} = \beta_{\text{rf}} \hat{y}_{\text{rf}} + \beta_{\text{xgb}} \hat{y}_{\text{xgb}} + \beta_{\text{lgb}} \hat{y}_{\text{lgb}} + \beta_0$, the composite SHAP values are formed as:
  $$\phi_{i, \text{stack}} = \beta_{\text{rf}} \phi_{i, \text{rf}} + \beta_{\text{xgb}} \phi_{i, \text{xgb}} + \beta_{\text{lgb}} \phi_{i, \text{lgb}}$$
  $$\mathbb{E}[\hat{y}_{\text{stack}}] = \beta_{\text{rf}} \mathbb{E}[\hat{y}_{\text{rf}}] + \beta_{\text{xgb}} \mathbb{E}[\hat{y}_{\text{xgb}}] + \beta_{\text{lgb}} \mathbb{E}[\hat{y}_{\text{lgb}}] + \beta_0$$

### 5.2 Independent Additive Reconstruction Verification

To audit numerical fidelity, the audit executed an independent test script comparing $\sum_{i=1}^{84} \phi_i + \mathbb{E}[\hat{y}]$ against actual model predictions on the 100 test explanation samples:

| Forecast Horizon | Primary Model | Mean Absolute Residual ($\mu\text{g/m}^3$) | Maximum Residual ($\mu\text{g/m}^3$) | Additive Reconstruction Verdict |
| :--- | :--- | :---: | :---: | :--- |
| **+1h (Stacking vs 25-Tree RF)** | Stacking (Bayesian Ridge) | $4.72 \times 10^{-6}$ | $1.63 \times 10^{-5}$ | **Exact to Machine Precision** |
| **+1h (Stacking vs 100-Tree RF)** | Stacking (Bayesian Ridge) | $0.0749$ | $0.6003$ | **Approximation Delta (Subsampled RF)** |
| **+6h (Stacking vs 25-Tree RF)** | Stacking (Bayesian Ridge) | $2.29 \times 10^{-6}$ | $1.14 \times 10^{-5}$ | **Exact to Machine Precision** |
| **+6h (Stacking vs 100-Tree RF)** | Stacking (Bayesian Ridge) | $0.0818$ | $0.8728$ | **Approximation Delta (Subsampled RF)** |
| **+24h (LightGBM)** | LightGBM Regressor | $2.32 \times 10^{-14}$ | $7.11 \times 10^{-14}$ | **Exact to Floating Point Epsilon** |
| **+48h (LightGBM)** | LightGBM Regressor | $4.66 \times 10^{-14}$ | $1.28 \times 10^{-13}$ | **Exact to Floating Point Epsilon** |

**Forensic Finding on Random Forest Subsampling:**
- In `shap_analyzer.py` (lines 92–97), the Random Forest model was sub-sampled to 25 trees (`rf_sub.estimators_ = rf_model.estimators_[:25]`) to accelerate TreeSHAP computation.
- Against this 25-tree surrogate ensemble, additive reconstruction is exact ($10^{-6}\,\mu\text{g/m}^3$).
- Against the full 100-tree production ensemble, there is a minor attribution delta of $\approx 0.075\,\mu\text{g/m}^3$ (max $0.60\,\mu\text{g/m}^3$), which is well below sensor measurement precision ($1.0\,\mu\text{g/m}^3$).
- **Verdict: PASS WITH LIMITATIONS.** The approximation is scientifically defensible for computational tractability, but it should be explicitly documented in `SHAP_EXPLAINABILITY_REPORT.md`.

### 5.3 Verification of Approved Predictor Matrix (No Unverified Road Features)

The audit verified the feature names serialized in `shap_values_h1.npz`, `shap_summary_h*.json`, and `global_feature_importance.json`:
- Exactly **84 predictor features** are present.
- Every single feature corresponds to the approved `PRIMARY_MODEL_PREDICTORS` list.
- **Zero unverified road-density, highway-distance, or road-length proxies are present.**
- Categorical importance shares evolve logically across forecast horizons:
  - **$+1\text{h}$:** Historical $\text{PM}_{2.5}$ dynamics dominate with **$79.6\%$** share (top features: `PM2.5` nowcast, `pm25_roll_mean_3h`, `pm25_roll_mean_24h`).
  - **$+6\text{h}$:** Rolling trends and diurnal solar cycles expand (Historical $\text{PM}_{2.5}$ at $62.3\%$, Solar/Calendar at $9.2\%$, Boundary Layer at $10.0\%$).
  - **$+24\text{h}$:** Autoregressive lags balance with thermodynamics (Historical $\text{PM}_{2.5}$ at $62.5\%$, Thermodynamics at $12.2\%$, Wind/Dispersion at $7.0\%$).
  - **$+48\text{h}$:** Atmospheric thermodynamics and boundary layer dynamics expand tenfold to **$16.8\%$** share (`surface_pressure`, `pm25_roll_min_24h`, `relative_humidity_2m`).

---

## 6. Software Integrity, Artifacts & Test Suite Verification

### 6.1 Execution of Automated Test Suite

The complete pytest test suite was executed in PowerShell across the implementation repository:
```
pytest new_research_project/implementation/tests -v
```
**Results:**
- **56 passed in 24.96 seconds** (Exit code: 0).
- Breakdown of passing test suites:
  - `test_spatial.py` (15 tests passed): Coordinate projections, IDW interpolation, zero-distance coincidence, LOO cross-validation, Kriging variogram fitting, Kriging singularity handling, spatial uncertainty surfaces, hotspot contouring.
  - `test_explainability.py` (10 tests passed): TreeSHAP computation, stacking composite attribution, additive reconstruction tolerance, human-readable narrative synthesis.
  - `test_modeling.py` (14 tests passed): Baselines, Random Forest, XGBoost, LightGBM, hyperparameter tuning, stacking meta-regressor, ablation experiments.
  - `test_features.py` (8 tests passed): Lag generation, rolling metrics, solar transformations, exclusion of road proxies.
  - `test_pipeline.py` (9 tests passed): End-to-end ingestion, quality flags, temporal boundary leakage buffer enforcement.

### 6.2 Complete Inventory of Reports and High-Resolution Artifacts

The audit confirmed the presence and structural integrity of all required Milestone 4 deliverables:

1. **Required Technical Reports (in `new_research_project/reports/`):**
   - `MILESTONE_4_SPATIAL_MODELING_REPORT.md` (Comprehensive executive milestone synthesis)
   - `SPATIAL_VALIDATION_REPORT.md` (LOSOCV results, degradation analysis, baseline comparison)
   - `IDW_VS_KRIGING_COMPARISON.md` (Rigorous spatial model comparison, justification of IDW)
   - `SPATIAL_UNCERTAINTY_REPORT.md` (Metropolitan zoning, cell accounting, unmonitored sector risk)
   - `HOTSPOT_INTERPRETATION_REPORT.md` (Operational excursion delineation, 0% extrapolation proof)
   - `SHAP_EXPLAINABILITY_REPORT.md` (TreeSHAP attribution, category shifts, narratives)

2. **Required High-Resolution Figures (33 PNG files in `new_research_project/reports/figures/`):**
   - IDW continuous surface maps: `idw_interpolation_surface_h1.png`, `h6.png`, `h24.png`, `h48.png`
   - Kriging surface maps: `kriging_interpolation_surface_h1.png`, `h6.png`, `h24.png`, `h48.png`
   - Method comparison maps: `idw_vs_kriging_h1.png`, `h6.png`, `h24.png`, `h48.png`
   - Spatial uncertainty maps: `spatial_uncertainty_zones.png`, `spatial_uncertainty_buffer_zones.png`, `spatial_uncertainty_kriging_variance_h1.png`
   - Hotspot delineation maps: `hotspot_delineation_h1.png`, `h6.png`, `h24.png`, `h48.png`
   - SHAP summary & beeswarm plots: `shap_beeswarm_h1.png`, `h6.png`, `h24.png`, `h48.png`
   - SHAP global feature importance: `shap_global_importance_h1.png`, `h6.png`, `h24.png`, `h48.png`
   - SHAP dependence plots: `shap_dependence_h1_PM2.5.png`, `shap_dependence_h6_wind_speed.png`, `shap_dependence_h24_pm25_roll_mean_24h.png`
   - SHAP station waterfalls: `shap_waterfall_Bollaram_Industrial_Area_h1.png`, `shap_waterfall_ECIL_Kapra_h1.png`

---

## 7. Audit Discrepancies, Recommended Corrections, and Final Recommendation

### 7.1 Discrepancies Uncovered During Audit

In accordance with strict audit instructions ("If a check fails, report it and recommend a correction separately. Do not silently fix it. Do not modify source code, models, predictions, datasets, thresholds, or reports during this audit"), the following two discrepancies are documented:

1. **Discrepancy 1: Sample Count Inaccuracy in Table 2.1 of `SPATIAL_VALIDATION_REPORT.md`**
   - *Observation:* Table 2.1 reported $N = 1,875$ samples for all four monitoring stations, implying a total of 7,500 evaluations.
   - *Forensic Fact:* $N = 1,875$ applies only to Bollaram Industrial Area due to sensor downtime. Central University ($N = 5,743$), ECIL Kapra ($N = 5,849$), and ICRISAT Patancheru ($N = 5,606$) have far higher sample sizes, yielding **19,073 total station evaluations** and **1,479 synchronous 4-station snapshots**.
   - *Severity:* Low (Documentation display error; underlying calculation code in `spatial_validation.py` and JSON metrics in `spatial_validation_results.json` are correct).
   - *Recommended Correction:* Update Table 2.1 in `SPATIAL_VALIDATION_REPORT.md` during post-audit documentation cleanup to reflect actual per-station sample counts and explicitly note that the network mean MAE of $11.703\,\mu\text{g/m}^3$ is an unweighted macro-average across station typologies.

2. **Discrepancy 2: Undocumented 25-Tree Subsampling in Stacking Random Forest SHAP**
   - *Observation:* In `shap_analyzer.py`, the Random Forest estimator was truncated to 25 trees for TreeSHAP tractability, creating a small numerical attribution delta ($\approx 0.075\,\mu\text{g/m}^3$) relative to the 100-tree production ensemble.
   - *Severity:* Very Low (Pragmatic computational optimization; residual is far below sensor noise).
   - *Recommended Correction:* Add a methodological clarification in `SHAP_EXPLAINABILITY_REPORT.md` documenting this 25-tree surrogate formulation for TreeSHAP.

### 7.2 Formal Recommendation

| Recommendation Option | Determination |
| :--- | :--- |
| **APPROVE MILESTONE 5** | Not Recommended (Pending formal acknowledgment of documentation discrepancies) |
| **APPROVE WITH CONDITIONS** | **RECOMMENDED DETERMINATION** |
| **DO NOT APPROVE** | Not Justified (All empirical results, pipelines, and tests are sound) |

### Conditions for Proceeding to Milestone 5:
1. **Condition 1 (Documentation Alignment):** Amend Table 2.1 of `SPATIAL_VALIDATION_REPORT.md` to display the verified sample sizes ($1,875$, $5,743$, $5,849$, $5,606$) and clarify the macro-average vs. sample-weighted metric definitions.
2. **Condition 2 (Explainability Note):** Add a concise explanatory footnote in `SHAP_EXPLAINABILITY_REPORT.md` documenting the 25-tree TreeSHAP subsampling for Random Forest within the Stacking ensemble.
3. **Condition 3 (Spatial Map Disclosures):** Ensure all future public or academic dissemination maps retain the explicit 3-tier uncertainty shading, explicitly marking the 34.9% high-uncertainty extrapolation zone ($> 15\,\text{km}$) to prevent misinterpretation of regional forecasts as confirmed ground violations.
4. **Condition 4 (Milestone 5 Authorization):** In accordance with the project instructions, **stop and await explicit user approval** before commencing Milestone 5 (Deployment, System Integration, API, or Dashboard development).

---
*Audit completed by Google DeepMind Antigravity on November 15, 2026. All source files, test logs, and Parquet tables verified independently.*
