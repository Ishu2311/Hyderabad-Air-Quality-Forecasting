# Research Feasibility & Comparative Methodology Report (Revised & Corrected)

**Project Title:** Traffic-Aware Spatio-Temporal PM2.5 Hotspot Prediction in Hyderabad Using Multi-Source Data and Explainable Machine Learning  
**Target Output:** Academic Capstone Project & IEEE/ACM Conference Paper Submission  
**Document Date:** November 2026 (Methodologically Strengthened)  
**Status:** Pre-Implementation Research & Methodological Feasibility Formulation  

---

## Executive Summary & Feasibility Determination

This revised report delivers an exhaustive scientific, architectural, and data-integrity feasibility analysis for developing an academic capstone project on **Traffic-Aware Spatio-Temporal $\text{PM}_{2.5}$ Hotspot Prediction in Hyderabad**. 

Urban air pollution in Greater Hyderabad exhibits substantial micro-climatic and land-use heterogeneity: heavy industrial zones in the north and northwest (e.g., Bollaram Industrial Area, Patancheru) suffer from chronic particulate matter exceedances, while commercial hubs (e.g., ECIL Kapra) and institutional green belts (e.g., Central University) experience contrasting baseline exposures. Mitigating adverse cardio-respiratory public health risks requires prospective, explainable predictive models capable of forecasting physical $\text{PM}_{2.5}$ mass concentrations ($\mu\text{g/m}^3$) and predicting localized statutory exceedance hotspots across forward-looking operational horizons (+1h, +6h, +24h, and +48h).

### Summary Feasibility Verdict & Methodological Ground Rules
1. **Core Proposed Direction:** **Technically and Methodologically Feasible**, provided all empirical constraints, spatial sample size limits, proxy boundaries, and uncertainty declarations are rigorously maintained.
2. **Modality Separation:**
   - **Ground-Truth In-Situ Sensor Observations:** Restricted strictly to Central Pollution Control Board (CPCB) Continuous Ambient Air Quality Monitoring Stations (CAAQMS). Across 4 stations from January 1, 2023 to September 30, 2026 (131,424 total station-hours), exactly **111,400 hours represent valid physical $\text{PM}_{2.5}$ observations**, while **20,024 hours are missing** (15.24% missingness) due to maintenance and telemetry dropouts.
   - **Atmospheric Model Estimates:** Open-Meteo weather parameters are numerical atmospheric reanalysis fields (European ECMWF/ERA5 data assimilation), *not* local rooftop instruments.
   - **Traffic-Related Proxies:** In the absence of citywide municipal automated traffic counters (ATCC) or inductive loops, OpenStreetMap (OSM) roadway lengths within projected metric buffers and diurnal peak-hour indicators represent **exploratory traffic-related proxies**, *not* measured vehicular flow or traffic volume. Static spatial road metrics provide only four unique station-level spatial observations, and no causal relationship between traffic proxies and particulate levels is claimed.
   - **TRAQID Dataset (IIIT Hyderabad, ICVGIP 2024):** Maintained strictly as a **separate descriptive case study**. Because official documentation confirms that CSV timestamps were resampled onto an artificial 5-second uniform grid and do not match wall-clock capture times, quantitative time-aligned comparisons against CPCB background concentrations are unverified and impossible without recovered timestamps and GPS tracks. TRAQID will *never* be merged into CPCB hourly records.
   - **Pre-2023 Datasets (Repos 1, 4, 5):** Daily and monthly datasets operate at coarse resolutions and lack individual station coordinates. They cannot be merged with hourly data without injecting artificial disaggregation noise.
3. **Primary Prediction Target:** **Physical $\text{PM}_{2.5}$ mass concentration ($\mu\text{g/m}^3$)** is the primary regression target across all horizons. Composite AQI is a synthetic piecewise-linear index that obscures physical aerosol transport; it is evaluated solely as a secondary diagnostic.
4. **Hotspot Threshold Formulations & +24h Target Clarity:**
   - For 24-hour targets (+24h), we strictly distinguish continuous physical $\text{PM}_{2.5}$ regression at timestamp $t+24$ ($\hat{y}_{i, t+24} \approx y_{i, t+24}$) from the statutory **CPCB 24-hour National Ambient Air Quality Standard (NAAQS) hotspot classification target** ($z_{i, t+24}^{(\text{24h})}$), which evaluates the cumulative forward arithmetic mean over $t+1$ through $t+24$ ($\bar{y}_{i, t+1:t+24} \ge 60\,\mu\text{g/m}^3$), calculated using the exact number of valid hourly observations ($N_{\text{valid}} \ge 16$), *never* a fixed denominator of 24.
   - For sub-daily operational horizons (+1h, +6h), instantaneous $\text{PM}_{2.5} \ge 60\,\mu\text{g/m}^3$ is defined as an **acute concentration excursion**, while $\text{PM}_{2.5} > 120\,\mu\text{g/m}^3$ denotes an **exploratory high-concentration alert threshold** identifying entry into the Very Poor category (since $120\,\mu\text{g/m}^3$ is the upper boundary of the CPCB Poor category and Very Poor begins strictly above $120\,\mu\text{g/m}^3$ at $\text{AQI} \ge 301$). Exactly $120\,\mu\text{g/m}^3$ is the upper bound of the Poor category, not Very Poor. This is strictly distinguished from the official CPCB Severe category, which begins strictly above $250\,\mu\text{g/m}^3$ ($\text{AQI} \ge 401$). Previous unsupported claims regarding Delhi-specific Graded Response Action Plan (GRAP) emergency thresholds in Hyderabad are completely removed.
5. **Spatial Interpolation Feasibility & Uncertainty:**
   - Four monitoring stations provide only $\binom{4}{2} = 6$ unique pairwise inter-station distances, providing very limited statistical degrees of freedom for empirical variogram estimation.
   - **Inverse Distance Weighting (IDW)** serves as the primary deterministic interpolation baseline; **Ordinary Kriging** is an experimental comparison.
   - Continuous $1\,\text{km} \times 1\,\text{km}$ grid maps represent **model-interpolated continuous potential surfaces**, *never* independently measured neighborhood ground truth. Kriging estimation variance ($\sigma_K^2(x, y)$) will be reported alongside spatial maps, but will be explicitly qualified as carrying inherent uncertainty due to variogram estimation limits.
6. **Dual Validation Protocols:** Enforces both **Chronological Out-of-Time Partitioning** (Train: 2023–2024, Val: 2025, Test: 2026) and **Leave-One-Station-Out (LOSO) Spatial Validation**. Under LOSO, the held-out station is completely excluded from feature generation, spatial lags, interpolation inputs, and threshold calibration. Four spatial folds provide preliminary exploratory evidence of transferability, but do not prove generalization across all of Hyderabad.

---

## 1. Deep-Dive Audit of Existing Project (`existing_project`)

### 1.1 Architecture & Pipeline Design
The existing codebase (`existing_project/capstone_research_ready/capstone_new`) provides an end-to-end Python research repository and Streamlit dashboard:
- `src/data/`: Ingestion of CPCB Excel sheets (`parse_cpcb.py`), Open-Meteo reanalysis weather extraction (`fetch_weather.py`), and heuristic temporal cleaning/imputation (`merge_clean.py`, which utilized ordinary linear interpolation).
- `src/features/`: Missing-aware target generation (`build_targets.py`), and construction of 122 predictor features spanning autoregressive lags, rolling statistics, calendar transforms, wind components, and inter-station spatial lags (`build_features.py`).
- `src/models/`: Baseline models (`train_baselines.py`), LightGBM/Random Forest regression (`train_ml.py`), prospective hotspot classification (`train_classifiers.py`), and a deep neural network MLP benchmark (`train_dl.py`).
- `src/experiments/`: Multi-horizon ablation isolating local, weather, and spatial feature sets (`ablation.py`).
- `src/eval/`: Statistical significance testing via Diebold-Mariano tests with Harvey-Leybourne-Newbold correction (`statistical_tests.py`, `run_statistical_tests.py`).
- `src/visualization/`: Tree SHAP explainability (`explainability.py`) and 300 DPI publication plots (`make_paper_figures.py`).

### 1.2 Empirical Data Counts per Station
Across the 32,856 chronological hours per station spanning January 1, 2023 00:00 through September 30, 2026 23:00 (131,424 total station-hours across 4 stations), empirical observations are distributed as follows:

```
+-------------------------------------------------------------------------------------------------------------------+
|                                      CAAQMS SENSOR DATA COMPLETENESS AUDIT                                         |
+--------------------------+--------------------+------------------------+---------------------+--------------------+
| Station Name             | Total Hours (Slot) | Valid PM2.5 Obs.       | Missing Hours       | Missingness (%)    |
+--------------------------+--------------------+------------------------+---------------------+--------------------+
| Bollaram Industrial Area | 32,856             | 23,662                 | 9,194               | 27.98%             |
| Central University       | 32,856             | 29,127                 | 3,729               | 11.35%             |
| ECIL Kapra               | 32,856             | 28,519                 | 4,337               | 13.20%             |
| ICRISAT Patancheru       | 32,856             | 30,092                 | 2,764               | 8.41%              |
+--------------------------+--------------------+------------------------+---------------------+--------------------+
| TOTAL NETWORK            | 131,424            | 111,400                | 20,024              | 15.24%             |
+--------------------------+--------------------+------------------------+---------------------+--------------------+
```

Every hourly time slot must *not* be described as an actual physical observation. Exactly 111,400 records represent physical sensor ground truth, while 20,024 records represent unobserved intervals resulting from sensor maintenance, power interruptions, and telemetry outages.

### 1.3 Preprocessing & Leakage Audit of Previous Implementation (AUD-01 to AUD-09)
The existing project addressed several data processing issues, but code verification reveals critical methodological caveats:
- **Missing-Aware Target Formulation (AUD-01):** Unobserved future sensor readings evaluate to `NaN` rather than defaulting to Class 0 clean air. Exactly 9,755 unobserved future sensor timestamps were preserved and excluded from loss updates, eliminating false-negative training bias.
- **Evaluation of Previous Imputation Protocol (AUD-02):** The previous codebase attempted short-gap linear interpolation ($\le 3\,\text{h}$) and medium-gap diurnal lookup imputation ($4 \text{ to } 24\,\text{h}$) fitted on 2023–2024 training data. However, **code verification of `existing_project/src/data/merge_clean.py` reveals that its short-gap linear interpolation (`df.interpolate(method='linear', limit=3)`) is bidirectional, calculating missing values at $t$ using future observations ($t_b > t$)**. Consequently, the old implementation **cannot be certified as leakage-free** for prospective forecasting. This limitation mandates the new prospective preprocessing protocol for this project, which strictly replaces linear interpolation with past-only forward fill ($\le 3\,\text{h}$, $y_{i, t-\delta}, \delta \ge 1$), eliminating lookahead leakage entirely.
- **Target Exclusion from Inputs (AUD-03):** Contemporaneous target indicators and hotspot flags were stripped from predictor feature vectors.
- **Station-Calibrated Decision Thresholds (AUD-06):** Resolved extreme class imbalance (hotspot prevalence of 48.4% in industrial Bollaram vs. 1.54% in residential ECIL Kapra) by calibrating station-specific decision cutoffs ($\tau_i$) on 2025 validation data.

### 1.4 Historical Baseline Metrics from Previous Implementation & Identified Limitations
- **Reported Historical Baseline Metrics from Previous Implementation (`existing_project`, Subject to Previous Pipeline Configuration):**
  - LightGBM achieved $\text{PM}_{2.5}$ RMSE of 5.44 ($R^2 = 0.855$) at +1h, 9.12 ($R^2 = 0.650$) at +6h, 10.10 ($R^2 = 0.576$) at +24h, and 11.07 ($R^2 = 0.503$) at +48h.
  - LightGBM composite AQI RMSE was 3.36 ($R^2 = 0.983$) at +1h and 13.84 ($R^2 = 0.696$) at +24h. At +48h, persistence tied with LightGBM (RMSE 17.48 vs. 17.90, Diebold-Mariano $p = 0.707$).
- **Limitations:**
  1. *Spatial Sparsity:* 4 stations across 7,257 $\text{km}^2$ results in coarse inter-station spacing (15–35 km).
  2. *Point Forecasts Only:* No continuous 2D spatial field interpolation across unmonitored sectors.
  3. *Zero Traffic Features:* Road network topology and diurnal vehicular congestion were absent.
  4. *Spatial Feature Degradation:* Inter-station spatial lags degraded +24h AQI RMSE from 13.69 to 14.01 because distant stations without local traffic/land-use context only added collinear noise.
  5. *Incomplete Benchmarking:* Random Forest was evaluated on only a 20% sample; XGBoost and Stacking Ensembles were omitted.

---

## 2. Comprehensive Audit of the Five Reference Repositories

```
+-------------------------------------------------------------------------------------------------------------------+
|                                  REFERENCE REPOSITORY AUDIT SUMMARY                                               |
+---+----------------------------+-----------------------+------------------------+--------------------+------------+
| # | Repository Name            | Dataset Ingested      | Modeling / Code Logic  | Temporal Granul.   | License    |
+---+----------------------------+-----------------------+------------------------+--------------------+------------+
| 1 | Hyderabad-Air-Quality-Data | Excel (2,636 rows)    | Stacking Regressor     | Daily (2016-2023)  | Unlicensed |
| 2 | aqi-prediction-health-risk | Open-Meteo API (90d)  | RF Regr/Clf, PCA, Clust| Daily (90 days)    | Unlicensed |
| 3 | TRAQID                     | Dashcam Images & CSV  | Vision Benchmark       | 5s (20 days)       | MIT        |
| 4 | Air-Quality-Index-Analysis | Kaggle CPCB & Monthly | RF, GBDT, SMOTE        | Daily & Monthly    | Unlicensed |
| 5 | Air-Quality-in-Hyderabad   | Monthly (72 rows)     | Exploratory Data Only  | Monthly (2017-2022)| Unlicensed |
+---+----------------------------+-----------------------+------------------------+--------------------+------------+
```

### 2.1 Repository 1: `kkarthiks1/Hyderabad-Air-Quality-Data`
- **Data:** `HYD Air Quality data.xlsx` (2,636 daily records from Aug 9, 2016 to Oct 27, 2023). Covers criteria pollutants, weather variables, AQI, and AQI category for an unspecified single Hyderabad location.
- **Code Logic:** `hyd_air_quality_ml_model.py` implements a `StackingRegressor` combining `XGBRegressor`, `LarsCV`, `BayesianRidge`, and `AdaBoostRegressor` with a `BayesianRidge` meta-model.
- **Critical Flaws:**
  - Script references `df`, `X`, `y` without loading data or defining feature slices.
  - Uses `train_test_split(..., shuffle=True)`, causing severe temporal data leakage on auto-correlated time series.
  - Drops all rows with any missing value (`df.dropna(how='any')`), destroying continuous sequences.
  - Predicts contemporaneous AQI from contemporaneous pollutants (identity index calculation, not forecasting).
- **Usability:** Provides a functional blueprint for heterogeneous stacking, but the dataset cannot be merged with hourly data. Attribution required under GitHub default copyright.

### 2.2 Repository 2: `kunal-sharma-dev/aqi-prediction-health-risk`
- **Data:** Calls Open-Meteo Air Quality API for 6 cities (90 days daily resampled). Falls back to a synthetic dataset generator if the API is unreachable.
- **Code Logic:** Random Forest regressor (for AQI) and classifier (for 4 health categories), K-Means clustering, PCA, and a hypothesis testing suite (independent t-test, One-way ANOVA, Chi-Square test of independence).
- **Critical Flaws:**
  - **Not Ground-Truth Observations:** Open-Meteo Air Quality API provides European CAMS/SILAM numerical atmospheric model reanalysis estimates, *not* ground-level physical monitoring observations.
  - Benchmark results reported in README (RMSE 10.61, $R^2 = 0.966$) were generated on the *synthetic fallback dataset*.
  - **Severe Contemporaneous Target Leakage:** In Cell 5, `aqi` is mathematically computed from `pm2_5`. In Cell 29, `pm2_5` is included in `ML_FEATURES` to predict `aqi`. The model simply fits the known piecewise linear equation, producing artificially inflated accuracy.
- **Usability:** The statistical testing suite (ANOVA, t-test, Chi-Square) is publication-grade and will be adapted for our evaluation. Open-Meteo pollutant data must *never* be treated as ground truth. Attribution required under GitHub copyright.

### 2.3 Repository 3: `omkathalkar/TRAQID-Traffic-Related-Air-Quality-Image-Dataset`
- **Authors & Venue:** Kathalkar et al. (IIIT Hyderabad), published at ACM ICVGIP 2024 (DOI: `10.1145/3702250.3702260`). Released under **MIT License**.
- **Data:** 26,678 front/rear dashcam image pairs (640x360) paired with mobile sensor readings (Nova SDS011 for PM2.5/PM10, BME280 for temp/humidity) collected from a moving passenger car over 20 collection days (Oct 2022 – Jul 2024, ~2,000 km).
- **Critical Methodological Finding:**
  - The official README explicitly cautions:
    > *"Match images to measurements BY ID. Do not match them by timestamp... During preprocessing, the sensor streams were resampled and interpolated onto a common 5-second grid so that each image frame could be assigned a measurement. In that step, the Timestamp column was regenerated as a uniform index for the resampled series. It does not hold the original capture time of each image... Do not use the CSV Timestamp column for time-based analysis."*
  - Mobile vehicle trajectory sampling along road corridors cannot be joined with stationary 24/7 rooftop CAAQMS stations. Attempting to merge them would corrupt time-series continuity.
- **Revised Role:** Maintained strictly as a **separate descriptive case study**. Because timestamps and GPS coordinates cannot be independently recovered from the CSV alone, simultaneous quantitative comparisons against CPCB background stations will not be claimed.

### 2.4 Repository 4: `clubgamma/Air-Quality-Index-Analysis`
- **Data:** `citywiseAQI_Index.csv` (Kaggle CPCB dataset 2015–2020, containing 2,006 daily rows for Hyderabad) and `AQI Data Set Hydrabad 22CE025.csv` (72 rows of monthly aggregate data from Jan 2017 to Dec 2022).
- **Code Logic:** Random Forest, Gradient Boosting, and SMOTE for category imbalance across student Hacktoberfest notebooks.
- **Critical Flaws:** Disconnected student contributions; monthly granularity is unsuitable for hourly forecasting; high missingness in 2015–2016 daily data.
- **Usability:** Demonstrates SMOTE for resolving class imbalance.

### 2.5 Repository 5: `danileahacker/Air-Quality-in-Hyderabad-Pollution-Analysis`
- **Data:** Raw CSV is omitted from git. Notebook outputs confirm it ingested the identical 72-row monthly dataset (2017–2022) found in Repo 4.
- **Code Logic:** Purely exploratory visualizations (Pandas/Matplotlib). No machine learning or predictive modeling.
- **Usability:** Confirms macro seasonal trends (particulate peaks in winter, annual minimum in monsoon).

---

## 3. Literature Comparison with Verified Scholarly Publications

Rather than making sweeping claims that no previous work has addressed these problems, we position our methodology against verified peer-reviewed publications:

```
+-------------------------------------------------------------------------------------------------------------------+
|                                    VERIFIED LITERATURE COMPARISON MATRIX                                          |
+--------------------------+---------------------+-------------------+---------------------+------------------------+
| Study & Citation         | Target / Region     | Data Sources      | Models Evaluated    | Key Finding & Limiting |
|                          |                     |                   |                     | Boundary Addressed     |
+--------------------------+---------------------+-------------------+---------------------+------------------------+
| Mathew et al. (2023)     | PM2.5 (Daily/Hourly)| CPCB CAAQMS       | HGBoost, KNN,       | Identified ~68% winter |
| Urban Climate            | Hyderabad, India    | Stations (Hyd)    | Random Forest       | concentration surge;   |
|                          |                     |                   |                     | lacks prospective      |
|                          |                     |                   |                     | spatial hotspot mapping|
+--------------------------+---------------------+-------------------+---------------------+------------------------+
| Guttikunda et al. (2014) | PM2.5 / PM10        | Chemical Filter   | Receptor Modeling   | Proved vehicular fleet |
| Atmospheric Environment  | Hyderabad, India    | Samplers (Hyd)    | & Chemical Mass Bal.| accounts for ~30-40% of|
|                          |                     |                   |                     | direct primary PM2.5.  |
+--------------------------+---------------------+-------------------+---------------------+------------------------+
| Lakshmi & Krishnamoorthy | Multi-Pollutant     | CAAQMS Stations + | BiConvLSTM with     | Demonstrated deep sequence|
| (2024), IEEE Access      | Delhi, India        | Reanalysis Weather| Spatio-Temporal Attn| modeling; complex black|
|                          |                     |                   |                     | box without tree SHAP. |
+--------------------------+---------------------+-------------------+---------------------+------------------------+
| Kathalkar et al. (2024)  | PM2.5 / PM10 (5s)   | Front/Rear Dashcam| AQC-Net, ResNet-50  | Proved visual estimation|
| ACM ICVGIP 2024          | Hyderabad Roads     | Images + Nova SDS | (Computer Vision)   | feasible on-road;      |
|                          |                     |                   |                     | mobile, non-continuous|
+--------------------------+---------------------+-------------------+---------------------+------------------------+
| Proposed Methodology     | Physical PM2.5 &    | CPCB CAAQMS (4stn)| Stacking Ensemble   | Unifies prospective    |
| (RM-HYD-PM25-2026)       | Hotspots (1-48h)    | + Open-Meteo +    | (RF, XGB, LGBM)     | forecasting, traffic   |
|                          | Greater Hyderabad   | OSM Road Proxies  | + IDW / Kriging     | proxies, and spatial   |
|                          |                     |                   |                     | uncertainty maps.      |
+--------------------------+---------------------+-------------------+---------------------+------------------------+
```

---

## 4. Definition and Justification of Forecast Targets

We establish strict mathematical separation between continuous physical regression targets and binary hotspot classification targets:

### 4.1 Continuous Physical Regression Target Formulation
The primary continuous target is physical $\text{PM}_{2.5}$ mass concentration ($\mu\text{g/m}^3$) at future horizon $h \in \{1, 6, 24, 48\}\,\text{hours}$:
$$\hat{y}_{i, t+h} = f_h\left( \mathbf{x}_{i, t}; \Theta_{\text{reg}, h} \right)$$
Optimized via Mean Squared Error loss evaluated strictly over valid observed timestamps:
$$\mathcal{L}_{\text{MSE}} = \frac{1}{|\Omega|} \sum_{(i, t) \in \Omega} \left( y_{i, t+h} - \hat{y}_{i, t+h} \right)^2$$

**Explicit Separation of +24h Continuous Regression and 24-Hour Hotspot Targets:**
- **+24h Continuous Physical Regression Target ($\hat{y}_{i, t+24}$):** Predicts the *instantaneous single-hour concentration* at the exact future timestamp $t+24$ ($y_{i, t+24}$).
- **Statutory 24-Hour Hotspot Classification Target ($z_{i, t+24}^{(\text{24h})}$):** Predicts whether the *cumulative forward 24-hour arithmetic average* over the entire window $t+1$ through $t+24$ ($\bar{y}_{i, t+1:t+24}$) reaches or exceeds the $60\,\mu\text{g/m}^3$ statutory standard, requiring at least 16 valid hourly observations ($N_{\text{valid}} \ge 16$).
These two targets address distinct physical and regulatory questions and are modeled as separate prediction tasks.

### 4.2 Statutory Standards, NAQI Breakpoints & Threshold Justifications
All concentration thresholds are anchored to official Central Pollution Control Board (CPCB) publications:
1. **Statutory Ambient Air Quality Standard:** Under CPCB Notification No. B-29016/20/90/PCI-L (*National Ambient Air Quality Standards*, The Gazette of India, November 18, 2009), the statutory standard for $\text{PM}_{2.5}$ is **$60\,\mu\text{g/m}^3$ averaged over a 24-hour period** (annual baseline standard is $40\,\mu\text{g/m}^3$).
2. **CPCB NAQI Sub-Index Breakpoints (CUPS/82/2014-15, October 2014):**
   - *Good (AQI 0–50):* $0 \text{ to } 30\,\mu\text{g/m}^3$
   - *Satisfactory (AQI 51–100):* $31 \text{ to } 60\,\mu\text{g/m}^3$ (upper bound is 24h statutory NAAQS)
   - *Moderate (AQI 101–200):* $61 \text{ to } 90\,\mu\text{g/m}^3$
   - *Poor (AQI 201–300):* $91 \text{ to } 120\,\mu\text{g/m}^3$ ($120\,\mu\text{g/m}^3$ is the exact upper boundary of Poor)
   - *Very Poor (AQI 301–400):* $121 \text{ to } 250\,\mu\text{g/m}^3$ (Very Poor begins strictly above $120\,\mu\text{g/m}^3$, i.e., $\text{PM}_{2.5} > 120\,\mu\text{g/m}^3$)
   - *Severe (AQI 401–500):* $> 250\,\mu\text{g/m}^3$ (CPCB Severe category requires $> 250\,\mu\text{g/m}^3$)

### 4.3 Missing-Aware 24-Hour Average Regulatory Hotspot Formulation
Let $m_{i, t+k} \in \{0, 1\}$ denote the binary missing-value mask:
$$m_{i, t+k} = \begin{cases} 1, & \text{if } y_{i, t+k} \text{ is observed and valid} \\ 0, & \text{if } y_{i, t+k} \text{ is missing (NaN)} \end{cases}$$
The count of valid hourly observations in the forward 24-hour window is $N_{\text{valid}} = \sum_{k=1}^{24} m_{i, t+k}$.  
The statutory condition requires $N_{\text{valid}} \ge 16$ (at least 66.7% data completeness). When $N_{\text{valid}} \ge 16$, the true 24-hour mean concentration is:
$$\bar{y}_{i, t+1:t+24} = \frac{\sum_{k=1}^{24} y_{i, t+k} \cdot m_{i, t+k}}{N_{\text{valid}}}$$
Notice that the denominator is $N_{\text{valid}}$, *never* a fixed 24. If a fixed 24 were used, unobserved hours would be implicitly treated as zero ($0\,\mu\text{g/m}^3$), artificially depressing calculated concentrations and hiding hotspots.

The statutory 24-hour hotspot target is:
$$z_{i, t+24}^{(\text{24h})} = \begin{cases} \text{NaN}, & \text{if } N_{\text{valid}} < 16 \\ 1, & \text{if } \bar{y}_{i, t+1:t+24} \ge 60\,\mu\text{g/m}^3 \\ 0, & \text{if } \bar{y}_{i, t+1:t+24} < 60\,\mu\text{g/m}^3 \end{cases}$$

### 4.4 Sub-Daily Operational Hotspot Targets Across Horizons
For sub-daily horizons, CPCB does not enforce a statutory single-hour legal limit. However, operational alerts require acute hazard detection:
1. **Acute Single-Hour Excursion ($h \in \{1, 6\}\,\text{hours}$):**
   $$z_{i, t+h}^{(\text{acute})} = \begin{cases} \text{NaN}, & \text{if } y_{i, t+h} \text{ is unobserved} \\ 1, & \text{if } y_{i, t+h} \ge 60\,\mu\text{g/m}^3 \\ 0, & \text{if } y_{i, t+h} < 60\,\mu\text{g/m}^3 \end{cases}$$
   *Scientific Purpose & Legal Distinction:* Evaluates whether an acute single-hour concentration reaches the numerical baseline of the 24-hour standard ($60\,\mu\text{g/m}^3$). This is strictly an operational early-warning alert and does *not* constitute statutory NAAQS non-compliance, which under Indian law can only be determined on a 24-hour average basis with $N_{\text{valid}} \ge 16$.
2. **Exploratory High-Concentration Alert Threshold / Very Poor Transition Threshold (> 120 ug/m3):**
   $$z_{i, t+h}^{(\text{alert})} = \begin{cases} \text{NaN}, & \text{if } y_{i, t+h} \text{ is unobserved} \\ 1, & \text{if } y_{i, t+h} > 120\,\mu\text{g/m}^3 \\ 0, & \text{if } y_{i, t+h} \le 120\,\mu\text{g/m}^3 \end{cases}$$
   *Justification & Category Status:* In the CPCB NAQI system, $120\,\mu\text{g/m}^3$ represents the exact upper boundary of the "Poor" category ($\text{AQI} = 300$). The "Very Poor" category begins strictly *above* $120\,\mu\text{g/m}^3$ ($\text{PM}_{2.5} > 120\,\mu\text{g/m}^3$, $\text{AQI} \ge 301$). Exactly $120\,\mu\text{g/m}^3$ is the upper bound of the Poor category, not Very Poor. Therefore, the exploratory alert threshold uses $y_{i, t+h} > 120\,\mu\text{g/m}^3$ to identify entry into the Very Poor category ($2 \times$ the statutory 24h standard of $60\,\mu\text{g/m}^3$). It is retained strictly as an **exploratory high-concentration alert threshold** (or Very Poor transition threshold). It must be clearly distinguished from the official CPCB **"Severe"** category, which begins strictly above $250\,\mu\text{g/m}^3$ ($\text{AQI} \ge 401$). It is not a statutory standard or emergency limit.
3. **Horizon +48h Scope:** Because atmospheric predictability degrades significantly beyond 24 hours and municipal actions are primarily planned on 24-hour cycles, **hotspot classification is formally restricted to horizons $h \in \{1, 6, 24\}\,\text{hours}$**, while +48h is evaluated exclusively as a continuous regression benchmark.

---

## 5. Corrected Seasonal-Naive Baseline (Zero Lookahead Leakage)

A seasonal-naive baseline for diurnal periodicity must strictly use observations that have occurred at or before prediction time $t$. We explicitly distinguish Zero-Order Persistence from Seasonal-Naive forecasting as two distinct, uncoupled baselines:
- **Zero-Order Persistence (Markovian):**  
  Projects the most recent instantaneous observation available at prediction time $t$ forward across all horizons:
  $$\hat{y}_{i, t+h}^{(\text{Persistence})} = y_{i, t} \quad \text{for all } h \in \{1, 6, 24, 48\}\,\text{hours}$$
  If $y_{i, t}$ is unobserved (`NaN`), the forecast evaluates strictly to `NaN` and is excluded from metric evaluation. Zero substitution is prohibited.
- **Seasonal-Naive Baseline (Diurnal Cycle, $S = 24\,\text{hours}$):**  
  Projects the observation from the exact corresponding hour of the preceding diurnal cycle observed prior to prediction time $t$. To guarantee zero lookahead leakage, every source timestamp is strictly $\le t$:

```
+-------------------------------------------------------------------------------------------------------------------+
|                                 EXACT SEASONAL-NAIVE BASELINE SPECIFICATION TABLE                                 |
+---------+--------+-----------------------+---------------------+-------------------------+------------------------+
| Horizon | Target | Diurnal Target Hour   | Exact Source Lag    | Exact Baseline Formula  | Temporal Status        |
| (h)     | Time   | (Hour of Day)         | Relative to t       | Source Timestamp        | Relative to t          |
+---------+--------+-----------------------+---------------------+-------------------------+------------------------+
| +1h     | t + 1  | (Hour(t) + 1) mod 24  | Lag = 23 hours      | y(i, t - 23)            | Strictly past (t - 23) |
+---------+--------+-----------------------+---------------------+-------------------------+------------------------+
| +6h     | t + 6  | (Hour(t) + 6) mod 24  | Lag = 18 hours      | y(i, t - 18)            | Strictly past (t - 18) |
+---------+--------+-----------------------+---------------------+-------------------------+------------------------+
| +24h    | t + 24 | (Hour(t) + 24) mod 24 | Lag = 24 hours      | y(i, t - 24)            | Strictly past (t - 24) |
+---------+--------+-----------------------+---------------------+-------------------------+------------------------+
| +48h    | t + 48 | (Hour(t) + 48) mod 24 | Lag = 24 hours      | y(i, t - 24)            | Strictly past (t - 24) |
+---------+--------+-----------------------+---------------------+-------------------------+------------------------+
```

* **Single Exact Timestamp Rule:** All alternative source timestamps are removed. For both $+24\text{h}$ and $+48\text{h}$, the seasonal-naive forecast uses the single exact historical observation from the preceding day's corresponding hour: $y_{i, t-24}$.
* **Handling Missing Source Observations Without Zero-Replacement:**  
  If the designated baseline source reading $y_{i, \tau}$ is missing (`NaN`):
  1. The baseline forecast evaluates strictly to `NaN`: $\hat{y}_{i, t+h} = \text{NaN}$.
  2. The timestamp is excluded from the baseline metric evaluation.
  3. Under no circumstances is a missing source observation imputed as $0.0\,\mu\text{g/m}^3$, as zero concentration represents pristine clean air and would artificially corrupt baseline error metrics.

---

## 6. Spatial Grid Consistency & Geodetic Verification

To eliminate mathematical inconsistencies between grid resolutions and geographic bounds:
* **Projected Coordinate System:** All spatial metrics, buffer generation, road lengths, and grid centroids are calculated using **WGS 84 / UTM Zone 44N (EPSG:32644)**, which preserves metric Euclidean distance accuracy across Hyderabad, avoiding spherical latitude/longitude degree distortion.
* **Bounding Box Extent:**
  - Latitude: $17.20^{\circ}\text{N} \text{ to } 17.65^{\circ}\text{N}$ ($\Delta \text{lat} \approx 0.45^{\circ} \approx 49.8\,\text{km}$).
  - Longitude: $78.15^{\circ}\text{E} \text{ to } 78.70^{\circ}\text{E}$ ($\Delta \text{lon} \approx 0.55^{\circ} \approx 58.4\,\text{km}$).
  - Projected Metric Bounds: Northing 1,902,000 m to 1,952,000 m (50,000 m span); Easting 198,000 m to 256,000 m (58,000 m span).
* **Grid Resolution & Point Count:**
  - A regular $1\,\text{km} \times 1\,\text{km}$ grid across this $50\,\text{km} \times 58\,\text{km}$ bounding box produces:
    $$N_{\text{rows}} = \frac{50\,\text{km}}{1\,\text{km}} + 1 = 51, \quad N_{\text{cols}} = \frac{58\,\text{km}}{1\,\text{km}} + 1 = 59 \implies 51 \times 59 = 3,009 \text{ evaluation centroids}$$
  - The previous claim of "100 x 100 = 10,000 points at 1 km spacing" was mathematically inconsistent (that would span $10,000\,\text{km}^2$). The corrected grid evaluates exactly **3,009 centroids at 1 km spacing** (or 11,879 centroids if a 500 m sub-grid is chosen).

---

## 7. Kriging Feasibility, Variogram Limits & Spatial Uncertainty

A critical methodological reality must be recognized:
- Four CAAQMS stations yield only $\binom{4}{2} = 6$ unique pairwise inter-station distances:
  - Bollaram to ICRISAT: $\approx 9.1\,\text{km}$
  - Bollaram to Central Univ: $\approx 9.3\,\text{km}$
  - ICRISAT to Central Univ: $\approx 8.4\,\text{km}$
  - Bollaram to ECIL Kapra: $\approx 25.6\,\text{km}$
  - Central Univ to ECIL Kapra: $\approx 26.6\,\text{km}$
  - ICRISAT to ECIL Kapra: $\approx 33.2\,\text{km}$
- In geostatistics, fitting an empirical semi-variogram $\gamma(h)$ with three parameters (nugget $c_0$, sill $c$, range $a$) requires 30–50+ observation locations. With only 6 pairwise points, variogram parameter estimation has near-zero degrees of freedom.

### Methodological Protocol for Spatial Mapping & Validation Decoupling
1. **Inverse Distance Weighting (IDW):** Serves as the **primary exploratory deterministic interpolation baseline** because it requires no variogram fitting.
2. **Ordinary Kriging:** Deployed as an **experimental comparison**.
3. **Decoupling Spatial Interpolation Validation from Forecasting Validation:**
   - *Spatial Interpolation Validation (Cross-Sectional):* Evaluates contemporaneous spatial reconstruction at time $t$. One station is left out, and its contemporaneous concentration is interpolated from the other three stations: $\hat{Z}_{-k}(s_k, t)$. Cross-validation interpolation RMSE is reported across all timestamps.
   - *Spatio-Temporal Forecasting Validation (LOSO):* Evaluates prospective forecasting ($t+h$) on an unseen station. The held-out station's pollution data is completely excluded from training features, spatial lags, interpolation inputs, and scalers. Classification thresholds $\tau$ are calibrated on training stations, never on test labels.
4. **Kriging Estimation Variance Map ($\sigma_K^2(x, y)$):**
   $$\sigma_K^2(s_0) = \mu + \sum_{i=1}^N \lambda_i \gamma(s_i, s_0)$$
   Will be visualized alongside concentration maps, but qualified transparently: Kriging variance reflects network geometry, but carries inherent parameter uncertainty because the variogram itself is fitted on 6 distance pairs.
5. **Non-Ground-Truth Declaration:** Continuous interpolated grids represent **model-interpolated continuous potential surfaces**, *never* independently measured neighborhood ground truth.

---

## 8. Four-Tier Feature Ablation Experiment Design

To rigorously evaluate the marginal predictive contribution of each feature group without conflation:

```
+--------------------------------------------------------------------------------------------------------------------+
|                                    FOUR-TIER CUMULATIVE ABLATION EXPERIMENT DESIGN                                 |
+------+-----------------------------+----------------------------------------------------------+--------------------+
| Tier | Configuration Name          | Features Included                                        | Scientific Purpose |
+------+-----------------------------+----------------------------------------------------------+--------------------+
| A    | Historical Pollution Only   | Autoregressive lags (1, 2, 3, 6, 12, 24, 48, 168h),      | Quantifies pure    |
|      |                             | trailing rolling statistics (3, 6, 24h mean, std, max),  | autoregressive     |
|      |                             | cyclical time transforms (hour, day, month sin/cos).     | persistence.       |
+------+-----------------------------+----------------------------------------------------------+--------------------+
| B    | Historical Pollution +      | Tier A + Open-Meteo reanalysis variables: temperature at | Quantifies physical|
|      | Weather                     | 2m, relative humidity, wind speed, wind direction,       | atmospheric        |
|      |                             | wind components (u, v), surface pressure, precipitation. | dispersion.        |
+------+-----------------------------+----------------------------------------------------------+--------------------+
| C    | Historical Pollution +      | Tier B + OpenStreetMap static road network densities     | Isolates marginal  |
|      | Weather + Road-Network      | (primary & secondary road length in 1km/3km buffers),    | gain of exploratory|
|      | Proxies                     | highway distance, diurnal rush-hour & freight indicators.| traffic proxies.   |
+------+-----------------------------+----------------------------------------------------------+--------------------+
| D    | Full Spatio-Temporal:       | Tier C + Inverse Distance Weighted inter-station lags    | Evaluates spatial  |
|      | Historical + Weather +      | from training stations only.                             | cross-station      |
|      | Road Proxies + Spatial Feats|                                                          | advection.         |
+------+-----------------------------+----------------------------------------------------------+--------------------+
```

---

## 9. Data and Validation Leakage Prevention

To ensure absolute methodological integrity, the following leakage-prevention protocols are enforced across the pipeline:

```
+-------------------------------------------------------------------------------------------------------------------+
|                                     DATA AND VALIDATION LEAKAGE PREVENTION MATRIX                                 |
+--------------------------+-------------------------------------+--------------------------------------------------+
| Vulnerability Dimension  | Potential Point of Failure          | Mandatory Architectural Enforcement              |
+--------------------------+-------------------------------------+--------------------------------------------------+
| Temporal Partitioning    | Random shuffle train-test split     | Strict Chronological Partitioning: Train 2023-24,|
|                          | leaking future state into past      | Val 2025, Test 2026. Zero forward shuffling.     |
+--------------------------+-------------------------------------+--------------------------------------------------+
| Missing Value Imputation | Computing mean imputation lookup    | Imputation lookup tables fitted strictly on      |
|                          | tables across entire 2023-2026 data | 2023-2024 training split. Mapped forward.        |
+--------------------------+-------------------------------------+--------------------------------------------------+
| Contemporaneous Leakage  | Target PM2.5 or hotspot flag        | Target and hotspot flags strictly excluded from  |
|                          | included in input feature matrix    | input features. Only historical lags y(t-k) used.|
+--------------------------+-------------------------------------+--------------------------------------------------+
| Meta-Model Stacking      | Base model predictions on train set | Base model meta-features generated strictly via  |
|                          | leaking into meta-learner           | 5-fold chronological blocked cross-validation.   |
+--------------------------+-------------------------------------+--------------------------------------------------+
| Spatial LOSO Validation  | Held-out station data leaking into  | Held-out station completely excluded from feature|
|                          | spatial lags, imputation, or scaler | matrices, scalers, and spatial lag calculations. |
+--------------------------+-------------------------------------+--------------------------------------------------+
| Decision Threshold Tuning| Calibrating classification threshold| Threshold tau calibrated strictly on validation  |
|                          | on test partition                   | partition (2025) or training stations in LOSO.   |
+--------------------------+-------------------------------------+--------------------------------------------------+
```

### 9.1 Exact Global Decision Threshold Selection Algorithm
Because extreme class imbalance characterizes urban $\text{PM}_{2.5}$ exceedances (ranging from 1.5% at ECIL to 48% at Bollaram), the default classification threshold $\tau = 0.50$ is suboptimal. We formulate an exact, deterministic grid-search optimization algorithm:

1. **Discrete Candidate Search Grid:**
   $$\mathcal{T} = \{\tau \in [0.05, 0.95] \mid \tau = 0.05 + 0.01 \cdot k, \; k \in \{0, 1, \dots, 90\}\}$$
2. **Chronological Global Threshold Selection:**
   Using the pooled validation predictions across training stations on the chronological validation partition $\mathcal{D}_{\text{val}}$ (calendar year 2025):
   $$\tau^* = \arg\max_{\tau \in \mathcal{T}} F_1\left(\tau; \mathcal{D}_{\text{val}}\right) = \arg\max_{\tau \in \mathcal{T}} \frac{2 \cdot \text{Precision}(\tau; \mathcal{D}_{\text{val}}) \cdot \text{Recall}(\tau; \mathcal{D}_{\text{val}})}{\text{Precision}(\tau; \mathcal{D}_{\text{val}}) + \text{Recall}(\tau; \mathcal{D}_{\text{val}})}$$
   Ties are broken deterministically by selecting the threshold closest to the empirical positive class prevalence in $\mathcal{D}_{\text{val}}$.
3. **Leave-One-Station-Out (LOSO) Threshold Calibration:**
   In LOSO spatial fold $k$ withholding station $s_k$, the decision threshold $\tau_{-k}^*$ is calibrated strictly on the pooled validation set of the remaining three training stations $\mathcal{S}_{\text{train}}^{(-k)} = \{s_m \mid m \ne k\}$:
   $$\tau_{-k}^* = \arg\max_{\tau \in \mathcal{T}} F_1\left(\tau; \{(p_{i, t}, z_{i, t}) \mid i \in \mathcal{S}_{\text{train}}^{(-k)}, t \in \text{Val}\}\right)$$
   Held-out station $s_k$'s test observations are evaluated strictly at $\tau_{-k}^*$. **Under no circumstances are held-out test station labels $z(s_k, t)$ used for threshold calibration**, eliminating test label leakage.

### 9.2 Preprocessing Leakage Audit & Prospective Inference Handling
To guarantee zero lookahead leakage, we explicitly decouple **retrospective target construction** from **prospective feature preprocessing**:

```
+-------------------------------------------------------------------------------------------------------------------+
|                               PREPROCESSING LEAKAGE AUDIT & DECOUPLING MATRIX                                     |
+--------------------------+-------------------------------------+--------------------------------------------------+
| Processing Dimension     | Retrospective Target Construction   | Prospective Feature Preprocessing                |
+--------------------------+-------------------------------------+--------------------------------------------------+
| Temporal Context         | Offline historical dataset curation | Online / Real-time inference at timestamp t      |
+--------------------------+-------------------------------------+--------------------------------------------------+
| Available Information    | Complete historical time series     | Strictly historical information observed <= t    |
+--------------------------+-------------------------------------+--------------------------------------------------+
| Interpolation Feasibility| Linear interpolation valid across   | FORWARD-LOOKING INTERPOLATION IS PROHIBITED.     |
|                          | known historical gaps               | No future point t_b > t may be used.             |
+--------------------------+-------------------------------------+--------------------------------------------------+
| Imputation Source        | Historical diurnal lookup table     | Imputation tables fitted strictly on 2023-2024   |
|                          | for gap completion                  | training partition; mapped forward into val/test.|
+--------------------------+-------------------------------------+--------------------------------------------------+
| Target Masking           | Preserves unobserved future as NaN; | Feature vectors x_t completely exclude target    |
|                          | requires N_valid >= 16 for 24h avg  | variables and contemporaneous hotspot flags.     |
+--------------------------+-------------------------------------+--------------------------------------------------+
```

#### Prospective Missing-Value Protocol at Inference Time
At prediction time $t$, if an in-situ predictor feature at $t$ or immediate lag $t-1$ is unobserved:
1. **Short Gaps ($\le 3\,\text{hours}$):** Forward-fill strictly from the most recent valid past observation ($y_{i, t-\delta}$, $\delta \ge 1$). Backward interpolation from future points ($t_b > t$) is strictly prohibited.
2. **Medium Gaps ($4 \text{ to } 24\,\text{hours}$):** Impute using the historical diurnal monthly-hourly lookup table $\hat{\mu}_{\text{train}}(m, k)$ computed strictly from the training split (2023–2024):
   $$\hat{\mu}_{\text{train}}(m, k) = \frac{1}{|\mathcal{S}_{m, k}|} \sum_{t' \in \mathcal{S}_{m, k}} y_{t'}, \quad \mathcal{S}_{m, k} = \{t' \in \mathcal{D}_{\text{train}} \mid \text{month}(t') = m, \text{hour}(t') = k\}$$
3. **Extended Gaps ($> 24\,\text{hours}$):** Feature remains `NaN`. Tree-based boosters (LightGBM and XGBoost) handle `NaN` natively via default branching directions established during training. For models requiring full dense arrays (Random Forest and Ridge), the training diurnal lookup mean is substituted. Future values are never accessed.

---

## 10. Methodological Risks and Mitigations

```
+-------------------------------------------------------------------------------------------------------------------+
|                                      METHODOLOGICAL RISKS AND MITIGATIONS                                         |
+--------------------------+-------------------------------------+--------------------------------------------------+
| Identified Risk          | Consequence if Ignored              | Methodological Mitigation Strategy               |
+--------------------------+-------------------------------------+--------------------------------------------------+
| Station Sparsity (N=4)   | Overconfident spatial predictions   | Report Kriging estimation variance sigma_K^2;    |
|                          | across unmonitored sectors          | label maps as model-interpolated potential surfaces|
+--------------------------+-------------------------------------+--------------------------------------------------+
| Variogram Undersampling  | Unstable sill, range, and nugget    | Designate IDW as primary deterministic baseline; |
|                          | estimates from 6 pairwise points    | report LOSO spatial interpolation errors.        |
+--------------------------+-------------------------------------+--------------------------------------------------+
| Traffic Proxy Invariance | Static road features have only 4    | Do not claim traffic volume or causal inference; |
|                          | unique spatial observations         | treat as exploratory static land-use proxies.    |
+--------------------------+-------------------------------------+--------------------------------------------------+
| Sensor Missingness       | Distorted 24h rolling averages      | Compute averages over valid hours N_valid >= 16; |
| (15.24% network loss)    | and false-negative training bias    | preserve unobserved target timestamps as NaN.    |
+--------------------------+-------------------------------------+--------------------------------------------------+
| Long-Horizon Drift (+48h)| Weather and persistence ties;       | Transparently report Diebold-Mariano ties;       |
|                          | forecast degradation at 48 hours    | restrict hotspot classification to 1h, 6h, 24h.  |
+--------------------------+-------------------------------------+--------------------------------------------------+
| Extreme Class Imbalance  | Alert collapse on clean stations    | Station-calibrated thresholding tau_i; report    |
| (1.5% ECIL vs 48% Boll.) | (F1 score collapse under global tau)| PR-AUC and ROC-AUC alongside F1-scores.          |
+--------------------------+-------------------------------------+--------------------------------------------------+
```

---

## 11. Unresolved Assumptions & Data-Source Verification Requirements

Before initiating coding or model training, the following specific assumptions must be verified:
1. **Hyderabad Freight Restriction Window:** The 23:00–05:00 heavy commercial vehicle freight entry window is designated as an **exploratory proxy assumption**. Authoritative Hyderabad Traffic Police / Commissionerate gazette notifications must be reviewed during implementation to document exact entry hours.
2. **Projected CRS Parity:** Verification that all shapefiles, road networks, and monitoring coordinates project cleanly into EPSG:32644 (UTM Zone 44N) without datum shift.
3. **TRAQID Raw Frame Recovery:** Verification of whether video timestamp overlays and GPS tracks can be programmatically recovered from TRAQID sample subsets before claiming any spatial road-corridor benchmarking.
4. **CPCB Sensor Calibration Standards:** Verification of whether TSPCB performed periodic gravimetric co-located calibration on the 4 CAAQMS beta attenuation monitors during 2023–2026.

---

## 12. Summary & Implementation Readiness

This revised feasibility report resolves all methodological deficiencies identified:
1. Corrects the 24-hour average denominator ($N_{\text{valid}} \ge 16$ with explicit mask $m_{i, t+k}$) and explicitly separates continuous $+24\text{h}$ instantaneous regression ($\hat{y}_{i, t+24}$) from statutory 24-hour rolling average hotspot classification ($z_{i, t+24}^{(\text{24h})}$).
2. Enforces single exact seasonal-naive source timestamps (+1h: $y(t-23)$, +6h: $y(t-18)$, +24h: $y(t-24)$, +48h: $y(t-24)$), keeps zero-order persistence strictly separate, removes all alternative source timestamps, and evaluates missing sources strictly as `NaN` without zero substitution.
3. Removes unsupported GRAP emergency claims; clarifies that $120\,\mu\text{g/m}^3$ is the upper boundary of the CPCB Poor category and classifies $\text{PM}_{2.5} > 120\,\mu\text{g/m}^3$ as an exploratory alert identifying entry into the Very Poor category ($\text{AQI} \ge 301$), strictly distinguished from the CPCB Severe category ($> 250\,\mu\text{g/m}^3$), while strictly distinguishing the $60\,\mu\text{g/m}^3$ 24-hour statutory standard from acute hourly excursion alerts.
4. Restricts traffic covariates to exploratory proxies, noting only 4 unique static spatial locations, and asserts zero causal claims.
5. Quarantines TRAQID as a separate descriptive case study.
6. Acknowledges variogram estimation limits (6 pairwise points); designates IDW as primary baseline.
7. Enforces projected metric coordinates (UTM Zone 44N, EPSG:32644) and verifies $51 \times 59 = 3,009$ grid points.
8. Reports exact empirical observation counts (111,400 valid physical observations, 20,024 missing hours).
9. Adds dedicated Literature Comparison, Methodological Risks and Mitigations, and Leakage Prevention sections.
10. Formulates an exact global threshold-selection algorithm ($\tau^* = \arg\max F_1$) on validation data of training stations, and decouples retrospective target creation from prospective feature preprocessing with zero future interpolation.

---
*Revised Feasibility Report verified and finalized inside `new_research_project`.*
