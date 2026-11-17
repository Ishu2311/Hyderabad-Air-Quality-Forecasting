# Final Mathematical and Scientific Audit Report

**Audit Code:** `AUDIT-HYD-PM25-FINAL`  
**Document Title:** Final Mathematical, Scientific, and Data Feasibility Audit of Methodology Specifications  
**Auditor:** Technical & Research Systems Review  
**Subject Files:** `RESEARCH_METHODOLOGY.md` and `RESEARCH_FEASIBILITY_REPORT.md`  
**Target Project:** Traffic-Aware Spatio-Temporal $\text{PM}_{2.5}$ Concentration and Hotspot Forecasting in Hyderabad  
**Date:** November 2026  
**Status:** Audit Completed — Methodological Guardrails Enforced  

---

## Executive Summary

This document presents the final mathematical, empirical, and scientific audit of the research methodology and feasibility specifications for the proposed Hyderabad air quality forecasting project. The audit verifies the mathematical formulation of baselines, clarifies statutory versus exploratory particulate thresholds, verifies empirical dataset observation counts, decouples spatial interpolation validation from temporal forecasting validation, and enforces strict non-leakage protocols.

> [!IMPORTANT]
> **Audit Status & Boundaries:**
> - This audit constitutes a **pre-implementation mathematical and methodological verification**.
> - In accordance with research standards, **no claim of publication readiness** is made prior to the successful, peer-reviewed execution of empirical experiments.
> - **Zero experimental results are fabricated or claimed** in this document. All empirical numbers reported herein represent descriptive statistics directly computed from the raw CPCB monitoring records.

---

## 1. Audit Dimension 1: Mathematical Verification of Baselines

### 1.1 Mathematical Distinction: Persistence vs. Seasonal-Naive
A fundamental error in environmental time-series benchmarking is conflating **Zero-Order Persistence** with **Seasonal-Naive** forecasting:
- **Zero-Order Persistence (Markovian Baseline):**  
  Projects the most recent instantaneous observation available at prediction time $t$ forward across all horizons $h \ge 1$:
  $$\hat{y}_{i, t+h}^{(\text{Persistence})} = y_{i, t}$$
  It assumes that the future state is best approximated by the current state, ignoring all diurnal, synoptic, or seasonal periodicity.
- **Seasonal-Naive (Diurnal Cyclical Baseline, Period $S = 24\,\text{hours}$):**  
  Projects the observation from the *same hour of the diurnal cycle* observed in the prior period:
  $$\hat{y}_{i, t+h}^{(\text{Seasonal Naive})} = y_{i, \tau(h)}$$
  where $\tau(h) \le t$ is a valid historical timestamp such that $\text{Hour}(\tau(h)) = \text{Hour}(t + h)$.

### 1.2 Mathematical Derivation & Exact Verification Across Horizons
To guarantee zero lookahead leakage ($\tau \le t$) and eliminate ambiguity, each horizon is mapped to exactly one historical source timestamp. Zero-Order Persistence is maintained as a separate, uncoupled baseline:

- **Zero-Order Persistence Baseline:**
  $$\hat{y}_{i, t+h}^{(\text{Persistence})} = y_{i, t} \quad \text{for all } h \in \{1, 6, 24, 48\}\,\text{hours}$$
- **Seasonal-Naive Baseline (Diurnal Periodicity, $S = 24\,\text{hours}$):**

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

### 1.3 Audit Verification on Single Exact Timestamps and Horizon +48h
1. **Purging of Alternative Timestamps:** All dual-option formulations (`[OptA]` / `[OptB]`) have been completely purged from both `RESEARCH_METHODOLOGY.md` and `RESEARCH_FEASIBILITY_REPORT.md`.
2. **Horizon +48h Exact Definition:** The seasonal-naive forecast for $+48\text{h}$ uses the single exact historical observation from the preceding day's corresponding hour: $y_{i, t-24}$ (which is lag 24 relative to prediction time $t$ and lag 72 relative to target $t+48$). Conflating the contemporaneous reading $y_{i, t}$ as a "seasonal-naive fallback" has been eliminated; $y_{i, t}$ is evaluated strictly under the independent Zero-Order Persistence baseline.
3. **Temporal Verification:** Every seasonal-naive source timestamp satisfies $\tau \le t$, strictly preventing lookahead leakage.

### 1.4 Handling Missing Source Observations Without Zero-Replacement
In physical atmospheric measurements, a particulate concentration of $0.0\,\mu\text{g/m}^3$ is non-physical under urban ambient conditions. Replacing missing source readings with zero corrupts baseline error metrics (MAE/RMSE) and fabricates artificial clean-air baselines.
- **Mandatory Audit Rule:**
  If the designated baseline source observation $y_{i, \tau}$ is missing (i.e., `np.isnan(y[i, tau])` evaluates to `True`):
  1. The baseline forecast evaluates strictly to `NaN`: $\hat{y}_{i, t+h} = \text{NaN}$.
  2. The timestamp $(i, t+h)$ is excluded from the baseline loss calculation.
  3. Under no circumstances is a missing reading imputed as zero ($0.0\,\mu\text{g/m}^3$). Zero substitution is strictly prohibited.

---

## 2. Audit Dimension 2: Particulate Standards, Breakpoints & Terminology

### 2.1 Authoritative CPCB Regulatory Foundations
To eliminate legal and scientific inaccuracies, all air quality thresholds are mapped directly to official statutory publications of the Central Pollution Control Board (CPCB), Ministry of Environment, Forest and Climate Change (MoEFCC), Government of India:

1. **Statutory Ambient Air Quality Standard:**
   - *Source:* CPCB Notification No. B-29016/20/90/PCI-L, *National Ambient Air Quality Standards (NAAQS)*, published in The Gazette of India, Extraordinary, Part III, Section 4, November 18, 2009.
   - *Statutory $\text{PM}_{2.5}$ 24-Hour Standard:* **$60\,\mu\text{g/m}^3$** (applicable to residential, rural, and industrial areas alike; annual standard is $40\,\mu\text{g/m}^3$).
   - *Statutory Mandate:* Requires an arithmetic mean of at least 16 hourly measurements monitored over a 24-hour period ($N_{\text{valid}} \ge 16$).
   - *Legal Distinction:* An instantaneous hourly concentration $\ge 60\,\mu\text{g/m}^3$ evaluated at $+1\text{h}$ or $+6\text{h}$ represents strictly an **acute operational excursion alert**. Under Indian environmental law, single-hour exceedances do *not* constitute statutory NAAQS non-compliance.
2. **National Air Quality Index (NAQI) Sub-Index Breakpoints:**
   - *Source:* CPCB Technical Report, *National Air Quality Index (NAQI)*, Control of Urban Pollution Series (CUPS/82/2014-15), October 2014.
   - The 24-hour pollutant concentration breakpoints for $\text{PM}_{2.5}$ ($\mu\text{g/m}^3$) are defined as:
     - **Good (AQI 0–50):** $0 \text{ to } 30\,\mu\text{g/m}^3$
     - **Satisfactory (AQI 51–100):** $31 \text{ to } 60\,\mu\text{g/m}^3$
     - **Moderate (AQI 101–200):** $61 \text{ to } 90\,\mu\text{g/m}^3$
     - **Poor (AQI 201–300):** $91 \text{ to } 120\,\mu\text{g/m}^3$ ($120\,\mu\text{g/m}^3$ is the exact upper boundary of Poor)
     - **Very Poor (AQI 301–400):** $121 \text{ to } 250\,\mu\text{g/m}^3$ (Very Poor begins strictly above $120\,\mu\text{g/m}^3$, i.e., $\text{PM}_{2.5} > 120\,\mu\text{g/m}^3$)
     - **Severe (AQI 401–500):** $> 250\,\mu\text{g/m}^3$ (CPCB Severe category requires $> 250\,\mu\text{g/m}^3$)

```
+-------------------------------------------------------------------------------------------------------------------+
|                                  CPCB THRESHOLD TAXONOMY & CLASSIFICATION AUDIT                                   |
+-------------------+--------------------+------------------------+-------------------------------------------------+
| Concentration     | Exact CPCB NAQI    | Regulatory / Legal     | Formal Methodology Classification               |
| Level (ug/m3)     | Sub-Index Bracket  | Category Status        | In Proposed Research Framework                  |
+-------------------+--------------------+------------------------+-------------------------------------------------+
| 60 ug/m3          | Upper limit of     | Statutory 24-Hour      | Statutory Hotspot Target (at +24h horizon)      |
|                   | "Satisfactory"     | NAAQS Legal Standard   | Acute Hourly Excursion Target (at +1h, +6h)     |
+-------------------+--------------------+------------------------+-------------------------------------------------+
| 90 ug/m3          | Upper limit of     | CPCB Sub-Index         | Intermediate CPCB Category Boundary             |
|                   | "Moderate" (AQI=200| Category Transition    | (Not evaluated as an independent binary target) |
+-------------------+--------------------+------------------------+-------------------------------------------------+
| > 120 ug/m3        | Exceeding upper    | CPCB NAQI Boundary     | Exploratory Alert for Entry into Very Poor       |
|                    | bound of "Poor"    | entering "Very Poor"   | (Poor upper bound is 120; Severe is > 250 ug/m3) |
+-------------------+--------------------+------------------------+-------------------------------------------------+
| 250 ug/m3         | Upper limit of     | CPCB NAQI Boundary     | Statutory Emergency Severe Threshold            |
|                   | "Very Poor" (AQI400| entering "Severe"      | (Official CPCB Severe category: AQI >= 401)     |
+-------------------+--------------------+------------------------+-------------------------------------------------+
```

### 2.2 Clarification of the $120\,\mu\text{g/m}^3$ Threshold
- **Correction & Category Boundary:** In the official CPCB NAQI system, $120\,\mu\text{g/m}^3$ represents the exact upper bound of the "Poor" category ($\text{AQI} = 300$). Exactly $120\,\mu\text{g/m}^3$ is the upper boundary of Poor, not Very Poor. Air quality enters the **"Very Poor"** category strictly above $120\,\mu\text{g/m}^3$ ($\text{PM}_{2.5} > 120\,\mu\text{g/m}^3$, $\text{AQI} \ge 301$).
- **Strict Distinction from CPCB Severe Category:** The concentration $120\,\mu\text{g/m}^3$ must **not** be confused with the official CPCB **"Severe"** category, which begins strictly above **$250\,\mu\text{g/m}^3$** ($\text{AQI} \ge 401$). It is also not a statutory standard or Graded Response Action Plan (GRAP) emergency standard in Hyderabad. (GRAP is a regulatory mechanism specific to the National Capital Region of Delhi under CAQM).
- In this project, $\text{PM}_{2.5} > 120\,\mu\text{g/m}^3$ is formally defined as an **exploratory high-concentration alert threshold** (or Very Poor transition threshold), representing $2 \times$ the 24-hour statutory baseline of $60\,\mu\text{g/m}^3$, providing early operational awareness of atmospheric stagnation.

---

## 3. Audit Dimension 3: Empirical Dataset Feasibility & Observation Audit

### 3.1 Verification of Station Observation Counts
Descriptive verification was conducted directly on the raw continuous CPCB dataset (`data/interim/combined_cpcb.csv`) spanning 32,856 chronological hours from January 1, 2023 00:00 through September 30, 2026 23:00.

```
+-------------------------------------------------------------------------------------------------------------------+
|                                  EMPIRICAL CAAQMS OBSERVATION & MISSINGNESS AUDIT                                 |
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

### 3.2 Audit Findings on Data Feasibility
1. **Consistency:** The stated count of **131,424 station-hours** accurately represents the theoretical full-rank time matrix (4 stations $\times$ 32,856 chronological hours, including the 2024 leap year).
2. **Observation Integrity:** Exactly **111,400 station-hours represent valid physical $\text{PM}_{2.5}$ sensor measurements** (84.76% network completeness).
3. **Missingness Distribution:** Exactly **20,024 station-hours are missing** (15.24% overall network loss). Missingness varies significantly across micro-environments:
   - Industrial Bollaram has the highest telemetry loss (**27.98%**), reflecting industrial power interruptions and sensor maintenance.
   - Peri-urban ICRISAT maintains the highest data continuity (**91.59%** valid data).
4. **Post-Cleaning Data State & Implementation Distinction:** In the previous codebase (`existing_project/.../cleaned.csv`), short-gap linear interpolation ($\le 3\,\text{h}$) was applied bidirectionally, which introduced future lookahead leakage ($t_b > t$). Code verification confirms that the old implementation **cannot be certified as leakage-free**. In contrast, the new prospective preprocessing protocol eliminates lookahead leakage entirely: short gaps ($\le 3\,\text{h}$) are forward-filled strictly using past observations only ($y_{i, t-\delta}, \delta \ge 1$), medium gaps ($4 \text{ to } 24\,\text{h}$) are imputed with diurnal means fitted strictly on the 2023–2024 training split, and the **9,506 extended outage hours** ($> 24\,\text{h}$, 7.23% of total network hours) remain `NaN`, preserved from loss updates.

---

## 4. Audit Dimension 4: Spatial Validation & Decoupling

A rigorous distinction is enforced between **Spatial Interpolation Validation** and **Spatio-Temporal Forecasting Validation**:

```
+-------------------------------------------------------------------------------------------------------------------+
|                                  SPATIAL VALIDATION DECOUPLING SPECIFICATION                                      |
+--------------------------+-----------------------------------------+----------------------------------------------+
| Dimension                | Task 1: Spatial Interpolation Validation| Task 2: Spatio-Temporal Forecasting (LOSO)   |
+--------------------------+-----------------------------------------+----------------------------------------------+
| Primary Objective        | Evaluate cross-sectional spatial field  | Evaluate temporal forecasting generalizability|
|                          | reconstruction across unmonitored space | to an unmonitored geographical station       |
+--------------------------+-----------------------------------------+----------------------------------------------+
| Temporal Dimension       | Static contemporaneous slice (time t)   | Multi-horizon prospective forecast (t + h)   |
+--------------------------+-----------------------------------------+----------------------------------------------+
| Held-Out Station Status  | Target station s_test is withheld;      | s_test is withheld from training feature     |
|                          | its value at time t is estimated from   | matrices, scalers, imputations, and spatial  |
|                          | {s1, s2, s3} at time t                  | lag feature calculations                     |
+--------------------------+-----------------------------------------+----------------------------------------------+
| Model Evaluated          | Spatial algorithms only: IDW (p=2)      | Full ML Model Zoo: Random Forest, XGBoost,   |
|                          | and Ordinary Kriging (semi-variogram)   | LightGBM, and Stacking Ensemble              |
+--------------------------+-----------------------------------------+----------------------------------------------+
| Spatial Lag Features     | Not applicable (pure spatial geometry)  | Spatial lags of training stations must ONLY  |
|                          |                                         | average the other 2 training stations        |
+--------------------------+-----------------------------------------+----------------------------------------------+
| Threshold Calibration    | Not applicable (continuous field)       | Decision cutoff tau is derived from training |
|                          |                                         | stations; NEVER calibrated on test station   |
+--------------------------+-----------------------------------------+----------------------------------------------+
| Metrics Reported         | Spatial Cross-Validation RMSE (ug/m3),  | Temporal RMSE, MAE, R2, ROC-AUC, PR-AUC,     |
|                          | MAE, and Kriging Variance sigma_K^2     | and F1-score evaluated at training cutoff    |
+--------------------------+-----------------------------------------+----------------------------------------------+
```

### 4.1 Strict Exclusion Protocol for Held-Out Station
Under Leave-One-Station-Out (LOSO) spatial validation for station $s_k$:
1. **Feature Matrix Isolation:** All rows where $\text{station} == s_k$ are removed from the training matrix.
2. **Spatial Lag Contamination Elimination:** For training stations $s_j \ne s_k$, any inter-station spatial lag feature (e.g., `neighbor_mean_PM25_lag1`) must be computed strictly over $\{s_m \mid m \ne k, m \ne j\}$. Station $s_k$ is never included in the spatial lag pool of training models.
3. **Preprocessing Isolation:** Imputation lookup tables, standardizers, and Min-Max scalers are fitted strictly on the three training stations.
4. **Decision Threshold Non-Leakage:** The classification decision threshold $\tau_{-k}^*$ is calibrated strictly on the pooled validation observations of the three training stations ($\mathcal{S}_{\text{train}}^{(-k)}$). Tuning $\tau$ on held-out test station $s_k$'s test labels constitutes test label leakage and is strictly prohibited.
5. **Spatial Evidence Boundary:** Four spatial folds provide preliminary exploratory evidence of cross-station transferability. They do *not* statistically prove generalizability across all unmonitored sectors of Greater Hyderabad.

### 4.2 Audit Dimension 5: Exact Global Decision Threshold Selection Algorithm
To prevent arbitrary heuristic cutoffs on heavily imbalanced classes (positive class prevalence ranging from 1.5% at ECIL Kapra to 48% at Bollaram Industrial Area), the documents were audited to verify the presence of an exact, deterministic mathematical algorithm:
1. **Discrete Grid Search Space:**
   $$\mathcal{T} = \{\tau \in [0.05, 0.95] \mid \tau = 0.05 + 0.01 \cdot k, \; k \in \{0, 1, \dots, 90\}\}$$
2. **Chronological Global Threshold Selection:**
   $$\tau^* = \arg\max_{\tau \in \mathcal{T}} F_1\left(\tau; \mathcal{D}_{\text{val}}\right) = \arg\max_{\tau \in \mathcal{T}} \frac{2 \cdot \text{Precision}(\tau; \mathcal{D}_{\text{val}}) \cdot \text{Recall}(\tau; \mathcal{D}_{\text{val}})}{\text{Precision}(\tau; \mathcal{D}_{\text{val}}) + \text{Recall}(\tau; \mathcal{D}_{\text{val}})}$$
   Evaluated strictly on the pooled validation partition $\mathcal{D}_{\text{val}}$ (calendar year 2025 across training stations). Ties are broken deterministically by selecting the threshold closest to empirical validation prevalence.
3. **LOSO Fold Threshold Calibration:**
   In LOSO spatial fold $k$ withholding station $s_k$, the decision threshold $\tau_{-k}^*$ is calibrated strictly on the pooled validation set of the remaining three training stations $\mathcal{S}_{\text{train}}^{(-k)} = \{s_m \mid m \ne k\}$:
   $$\tau_{-k}^* = \arg\max_{\tau \in \mathcal{T}} F_1\left(\tau; \{(p_{i, t}, z_{i, t}) \mid i \in \mathcal{S}_{\text{train}}^{(-k)}, t \in \text{Val}\}\right)$$
   Held-out station $s_k$'s test observations are evaluated strictly at $\tau_{-k}^*$. **Under no circumstances are held-out test station labels $z(s_k, t)$ accessed during calibration.**
4. **Verification Finding:** Formally specified in `RESEARCH_METHODOLOGY.md` (Section 11.2.1) and `RESEARCH_FEASIBILITY_REPORT.md` (Section 9.1).

### 4.3 Audit Dimension 6: Preprocessing Leakage Audit & Prospective Inference Protocol
Environmental time-series systems frequently suffer from lookahead leakage during missing-value imputation (e.g., bidirectional linear interpolation across gaps using future points). The audit verified the strict decoupling of retrospective target construction from prospective feature preprocessing:
1. **Retrospective Target Construction (Offline Historical Curation):**
   - Target physical continuous values $y_{i, t+h}$ and 24-hour averages $\bar{y}_{i, t+1:t+24}$ are constructed offline for model training and evaluation.
   - Retrospective linear interpolation across gaps $\le 3\,\text{hours}$ is permissible only for constructing historical target series.
   - Gaps $> 3\,\text{hours}$ and rolling windows with $N_{\text{valid}} < 16$ evaluate strictly to `NaN`, preserving unobserved future periods.
2. **Prospective Feature Preprocessing (Online Real-Time Inference at Timestamp $t$):**
   - Forward-looking linear interpolation between $t_a < t$ and $t_b > t$ is **strictly prohibited**. At inference time $t$, observations at $t_b > t$ do not exist.
   - *Short Gaps ($\le 3\,\text{hours}$):* Forward-filled strictly from the most recent valid past observation ($y_{i, t-\delta}$, $\delta \ge 1$).
   - *Medium Gaps ($4 \text{ to } 24\,\text{hours}$):* Imputed using the diurnal monthly-hourly lookup table $\hat{\mu}_{\text{train}}(m, k)$ computed strictly on the 2023–2024 training split.
   - *Extended Gaps ($> 24\,\text{hours}$):* Features remain `NaN`, processed natively by gradient boosting tree algorithms (LightGBM/XGBoost). Ridge and Random Forest models substitute the training-fitted diurnal mean. Future values are never accessed.
3. **Verification Finding:** Formally specified in `RESEARCH_METHODOLOGY.md` (Section 8.6) and `RESEARCH_FEASIBILITY_REPORT.md` (Section 9.2).

---

## 5. Comprehensive Audit Matrix: Issues, Evidence, Corrections & Limitations

```
+---------------------------------------------------------------------------------------------------------------------------------------------------+
|                                                 COMPREHENSIVE SCIENTIFIC AUDIT MATRIX                                                             |
+----+-----------------------+-------------------------+-----------------------------------------+--------------------------------------------------+
| ID | Audit Finding / Issue | Empirical Evidence      | Mandatory Correction Implemented        | Document Evidence & Cross-References             |
+----+-----------------------+-------------------------+-----------------------------------------+--------------------------------------------------+
| 01 | 24-Hour PM2.5 Average | Fixed denominator 24    | Denominator replaced with N_valid >= 16 | RESEARCH_METHODOLOGY.md (Sec 8.2, 8.3)           |
|    | Denominator Distortion| depresses average and   | via explicit binary mask; unobserved    | RESEARCH_FEASIBILITY_REPORT.md (Sec 4.3)         |
|    |                       | fabricates 0 ug/m3.     | evaluate strictly to NaN.               | Equation: \bar{y} = \sum y*m / N_valid           |
+----+-----------------------+-------------------------+-----------------------------------------+--------------------------------------------------+
| 02 | Seasonal-Naive Single | Dual OptA/OptB options  | Single exact past timestamps enforced:  | RESEARCH_METHODOLOGY.md (Sec 8.5)                |
|    | Source Timestamps &   | and +48h persistence    | +1h: y(t-23), +6h: y(t-18),             | RESEARCH_FEASIBILITY_REPORT.md (Sec 5)           |
|    | Zero Lookahead        | fallback ambiguity.     | +24h: y(t-24), +48h: y(t-24).           | Persistence y(t) separated; all OptA/OptB purged.|
+----+-----------------------+-------------------------+-----------------------------------------+--------------------------------------------------+
| 03 | Non-Physical Zero     | Imputing missing source | Missing source readings evaluate strictly| RESEARCH_METHODOLOGY.md (Sec 8.5)               |
|    | Baseline Imputation   | baseline readings as 0  | to NaN and are excluded from metrics.   | RESEARCH_FEASIBILITY_REPORT.md (Sec 5)           |
|    |                       | fabricates clean air.   | Zero substitution is strictly prohibited| Zero substitution explicitly banned.             |
+----+-----------------------+-------------------------+-----------------------------------------+--------------------------------------------------+
| 04 | Unsupported GRAP      | Claimed 120 ug/m3 is a  | Purged GRAP claims; 120 is Poor upper   | RESEARCH_METHODOLOGY.md (Sec 8.2, 8.4)           |
|    | Emergency Standard    | statutory GRAP emergency| bound; PM2.5 > 120 ug/m3 is alert for   | RESEARCH_FEASIBILITY_REPORT.md (Sec 4.2, 4.4)    |
|    | Claim for Hyderabad   | threshold in Hyderabad. | Very Poor entry; CPCB Severe is > 250   | CPCB CUPS/82/2014-15 citation verified.          |
+----+-----------------------+-------------------------+-----------------------------------------+--------------------------------------------------+
| 05 | Spatial Degree of     | 131,424 hourly rows     | Clarified that static road features have| RESEARCH_METHODOLOGY.md (Sec 4.3, 7.2, 13)       |
|    | Freedom Overstatement | conflated with 131,424  | only 4 spatial points; labeled as       | RESEARCH_FEASIBILITY_REPORT.md (Sec 2.3, 8)      |
|    | on Traffic Proxies    | spatial observations.   | exploratory proxies; no causal claims.  | Static land-use proxy limitation documented.     |
+----+-----------------------+-------------------------+-----------------------------------------+--------------------------------------------------+
| 06 | Variogram Fitting     | 4 stations produce only | Designated IDW as primary baseline;     | RESEARCH_METHODOLOGY.md (Sec 8.7, 10.2)          |
|    | Feasibility in Sparse | 6 pairwise distances;   | Ordinary Kriging as experimental        | RESEARCH_FEASIBILITY_REPORT.md (Sec 7)           |
|    | Network (N=4)         | undersampled for Kriging| comparison; Kriging variance qualified. | Variogram degrees-of-freedom limit stated.       |
+----+-----------------------+-------------------------+-----------------------------------------+--------------------------------------------------+
| 07 | TRAQID Timestamp      | Timestamps are index    | Quarantined TRAQID as separate          | RESEARCH_METHODOLOGY.md (Sec 3.3, 4.4)           |
|    | Resampling & Mobile   | counters; moving vehicle| descriptive case study; zero merge with | RESEARCH_FEASIBILITY_REPORT.md (Sec 2.4, 3)      |
|    | Alignment Incompatible| mobile sensor.          | CPCB hourly stationary dataset.         | Auxiliary case study role enforced.              |
+----+-----------------------+-------------------------+-----------------------------------------+--------------------------------------------------+
| 08 | Mathematical Grid     | Previous text claimed   | Project UTM Zone 44N (EPSG:32644);      | RESEARCH_METHODOLOGY.md (Sec 8.7)                |
|    | Inconsistency         | 100x100 = 10,000 points | verified 51x59 = 3,009 centroids across | RESEARCH_FEASIBILITY_REPORT.md (Sec 6)           |
|    | (100x100 at 1 km)     | at 1 km across 50x58 km | 50 km x 58 km bounding box.             | Geodetic EPSG:32644 bounds mathematically verified|
+----+-----------------------+-------------------------+-----------------------------------------+--------------------------------------------------+
| 09 | LOSO Spatial          | Potential leakage of    | Complete exclusion of test station from | RESEARCH_METHODOLOGY.md (Sec 10.2)               |
|    | Contamination &       | test station in lags    | lags, scalers, imputations, and         | RESEARCH_FEASIBILITY_REPORT.md (Sec 7, 9)        |
|    | Held-Out Station Isolation and threshold tuning.| thresholds; test labels never accessed. | LOSO sub-protocol 2B strict isolation enforced.  |
+----+-----------------------+-------------------------+-----------------------------------------+--------------------------------------------------+
| 10 | Exact Classification  | Arbitrary default 0.50  | Formulated exact grid search:           | RESEARCH_METHODOLOGY.md (Sec 11.2.1)             |
|    | Threshold Selection   | threshold causes alert  | tau* = argmax F1(tau; D_val) on training| RESEARCH_FEASIBILITY_REPORT.md (Sec 9.1)         |
|    | Algorithm             | collapse on ECIL (1.5%) | validation data; LOSO on 3 train stns.  | Optimization space T = [0.05, 0.95], step 0.01.  |
+----+-----------------------+-------------------------+-----------------------------------------+--------------------------------------------------+
| 11 | Preprocessing Leakage | Previous implementation | Decoupled retrospective target creation | RESEARCH_METHODOLOGY.md (Sec 8.6)                |
|    | in Old Codebase vs    | ordinary linear inter-  | from prospective feature preprocessing; | RESEARCH_FEASIBILITY_REPORT.md (Sec 1.3, 9.2)    |
|    | New Prospective Protocol| polation was bidirect.| old code uncertified; new uses ffill<=3h| Zero future interpolation at inference time t.   |
+----+-----------------------+-------------------------+-----------------------------------------+--------------------------------------------------+
| 12 | +24h Target           | Conflation of +24h      | Explicitly separated: +24h continuous   | RESEARCH_METHODOLOGY.md (Sec 8.1, 8.3)           |
|    | Formulation Clarity   | regression with 24h     | physical regression y(t+24) vs statutory| RESEARCH_FEASIBILITY_REPORT.md (Sec 4.1, 4.3)    |
|    |                       | rolling average hotspot.| 24h window average z(t+24) (N_valid>=16)| Two distinct tasks evaluated separately.         |
+----+-----------------------+-------------------------+-----------------------------------------+--------------------------------------------------+
```

---

## 6. Unresolved Assumptions & Implementation Guardrails

Before any implementation script is executed, the following four items must be respected:
1. **Authoritative Freight Regulation Notification:** The diurnal heavy commercial vehicle freight entry window (23:00–05:00) is recognized as an **exploratory assumption**. If formal gazette orders from the Hyderabad City Police Commissionerate are inaccessible, this feature must remain explicitly designated as an experimental proxy.
2. **Projected Geodetic System:** All distance, road length, and buffer analyses must execute in **EPSG:32644 (UTM Zone 44N)**. Under no circumstances should Euclidean distance formulas be applied directly to unprojected spherical latitude/longitude degrees.
3. **No Claim of Causality:** Tree SHAP attributions must be reported as non-linear statistical associations between inputs and outputs. The project does not assert physical or chemical causality.
4. **No Claim of Publication Acceptance:** All materials produced must be described as pre-submission academic manuscripts. Acceptance or publication readiness can only be determined through independent peer review.

---
*Audit completed, verified, and certified for methodological alignment in `new_research_project`.*
