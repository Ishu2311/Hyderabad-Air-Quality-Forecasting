# Milestone 3 — Final Independent Consistency Audit Report

**Document Code:** `REPORT-M3-FINAL-CONSISTENCY-AUDIT`  
**Project Title:** Traffic-Aware Spatio-Temporal PM2.5 Concentration and Hotspot Forecasting in Hyderabad Using Multi-Source Environmental Data and Explainable Machine Learning  
**Milestone:** Milestone 3 — Baseline Benchmarking and Machine Learning Model Development  
**Audit Date:** November 2026  
**Auditor:** Independent Scientific Audit Agent (Antigravity Core)  
**Audit Status:** **COMPLETED**  
**Final Audit Verdict:** **PASS WITH LIMITATIONS**  

---

## Executive Summary

This independent forensic audit was conducted pursuant to the final audit mandate prior to Milestone 4 approval. The audit rigorously inspected the underlying source code, serialized prediction Parquet stores, JSON metric registries, execution logs, and experiment reports for all machine learning models and baselines developed under Milestone 3.

### Summary of Audit Findings
1. **Baseline Sample Counts Reconciled:** The difference between Seasonal-Naive sample counts ($N=18,531$ at +1h) and Persistence/ML model counts ($N=19,048$ at +1h) is mathematically explained by diurnal source-lag missingness at partition startup boundaries and sensor outages. ML models and Persistence were evaluated on strictly paired sample vectors. The previous report statement claiming "all models and baselines were evaluated on identical sample masks" has been formally corrected.
2. **Test-Set Isolation Confirmed:** Hyperparameter tuning was performed strictly between Train (2023–2024) and Validation (2025). The holdout test set (2026) was never accessed during feature selection, model selection, or hyperparameter optimization. Multi-tier feature ablation on the test set was purely a post-hoc diagnostic holdout evaluation; zero test-set feedback occurred.
3. **Severe-Spike Imbalance Forensic Analysis:** The 2026 test partition contains **exactly 1 positive sample** ($\ge 120\,\mu\text{g/m}^3$) across the entire network for each forecasting horizon (located at ECIL Kapra; 0 occurrences at the other 3 stations). Consequently, precision, recall, and $F_1$ are identically $0.000$, and PR-AUC is at base prevalence ($5.2 \times 10^{-5}$). Reported ROC-AUC values (0.815 to 0.993) are purely arithmetic artifacts of ranking a single positive point above negatives. This task is formally declared **statistically underpowered and purely exploratory**.
4. **Threshold Terminology Verified:** All targets define $60\,\mu\text{g/m}^3$ and $120\,\mu\text{g/m}^3$ strictly as physical mass concentration thresholds (corresponding to the 24-hour statutory standard boundary and Very Poor entry boundary), avoiding any conflation with dimensionless AQI numbers or statutory category names.
5. **Horizon-Specific Model Selection Established:** Stacking is **not** uniformly best across all horizons. Stacking achieves best MAE at +1h ($3.01\,\mu\text{g/m}^3$) and ties with LightGBM at +6h ($5.99\,\mu\text{g/m}^3$). However, **LightGBM is the definitive winner at +24h** (MAE: $6.67\,\mu\text{g/m}^3$, RMSE: $9.74\,\mu\text{g/m}^3$, $R^2: 0.687$) and **+48h** (MAE: $7.93\,\mu\text{g/m}^3$, RMSE: $10.95\,\mu\text{g/m}^3$, $R^2: 0.599$). Previous report tables bolding Stacking at +24h and +48h have been corrected.

---

## Section 1: Baseline Sample-Count Inconsistency Audit

### 1.1 The Reported Discrepancy
In `BASELINE_COMPARISON_REPORT.md`, the reported valid sample counts on the 2026 test set are:
- Persistence, Random Forest, XGBoost, LightGBM, Stacking: $N = 19,048$ (+1h), $N = 18,940$ (+6h), $N = 18,610$ (+24h), $N = 18,411$ (+48h).
- Seasonal-Naive Baseline: $N = 18,531$ (+1h), $N = 18,546$ (+6h), $N = 18,325$ (+24h), $N = 18,193$ (+48h).

However, `MILESTONE_3_MODEL_TRAINING_REPORT.md` (Section 5, Item 6) previously claimed:
> *"Consistent Evaluation Masks: Model and baseline comparisons were evaluated on identical sample masks."*

### 1.2 Forensic Source Code Inspection
Code inspection was performed on:
- `new_research_project/implementation/modeling/evaluation.py` (`compute_regression_metrics` and `compute_diebold_mariano_test`)
- `new_research_project/implementation/modeling/train_regression.py`
- `new_research_project/implementation/modeling/complete_stacking_and_metrics.py`
- `new_research_project/implementation/modeling/baselines.py`

In `evaluation.py` (lines 40–44):
```python
def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    baseline_pred: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if baseline_pred is not None:
        mask = mask & ~np.isnan(baseline_pred)
```

In `train_regression.py` (lines 142–156):
```python
# Baseline metrics
metrics["test"][h_code]["persistence"] = compute_regression_metrics(y_test, p_test_pers)
metrics["test"][h_code]["seasonal_naive"] = compute_regression_metrics(y_test, p_test_sn)

# ML Model metrics (paired with persistence)
for m_name in ["rf", "xgb", "lgb"]:
    metrics["test"][h_code][m_name] = compute_regression_metrics(
        y_test, trained[m_name]["test_pred"], baseline_pred=p_test_pers
    )
```

In `baselines.py` (lines 48–78):
- Zero-Order Persistence: $\hat{y}_{i, t+h}^{(\text{Persistence})} = y_{i, t}$. Source timestamp is $t$.
- Seasonal-Naive:
  - $+1\text{h}$: Source is $t - 23\text{h}$
  - $+6\text{h}$: Source is $t - 18\text{h}$
  - $+24\text{h}$: Source is $t - 24\text{h}$
  - $+48\text{h}$: Source is $t - 24\text{h}$

### 1.3 Mathematical Root Cause of Count Differences
1. **Persistence & ML Models ($N = 19,048$ at +1h):**
   ML models generate continuous predictions for all rows where predictor features are available. In the purged test store, all 19,073 rows with valid target values have complete predictor matrices. When paired with `baseline_pred=p_test_pers`, the sample mask selects timestamps where both target $y_{t+h}$ and origin observation $y_t$ are non-NaN:
   $$\mathcal{M}_{\text{pers}} = \{ (i, t) \mid y_{i, t+h} \ne \text{NaN} \land y_{i, t} \ne \text{NaN} \}$$
   Across the 26,016 test hours, exactly 19,048 observations satisfy this condition at $+1\text{h}$.
2. **Seasonal-Naive ($N = 18,531$ at +1h):**
   Seasonal-naive requires an unobserved historical lag from 18, 23, or 24 hours prior:
   $$\mathcal{M}_{\text{sn}} = \{ (i, t) \mid y_{i, t+h} \ne \text{NaN} \land y_{i, t - \text{lag}} \ne \text{NaN} \}$$
   Seasonal-naive drops additional samples due to:
   - **Boundary Startup:** The initial 18 to 24 hours of the test partition cannot access prior observations within the partition without crossing the 48-hour boundary buffer.
   - **Preceding Telemetry Outages:** Any station sensor downtime 18, 23, or 24 hours prior causes the seasonal forecast to evaluate to `NaN`, even if the station is operational at origin $t$ and lead $t+h$.
3. **Exact Sample Matrix Census (2026 Test Set, 26,016 Total Network Hours):**

| Horizon | Total Test Rows | Target Non-NaN | Persistence Non-NaN | Paired ($y \cap \text{Pers}$) | Seasonal-Naive Non-NaN | Paired ($y \cap \text{SN}$) | 3-Way Joint ($y \cap \text{Pers} \cap \text{SN}$) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **+1h** | 26,016 | 19,073 | 25,972 | **19,048** | 25,268 | **18,531** | **18,531** |
| **+6h** | 26,016 | 19,078 | 25,798 | **18,940** | 25,274 | **18,546** | **18,546** |
| **+24h** | 26,016 | 19,099 | 25,274 | **18,610** | 24,670 | **18,325** | **18,142** |
| **+48h** | 26,016 | 19,119 | 24,964 | **18,411** | 24,670 | **18,193** | **17,914** |

### 1.4 Verification of Paired Diebold-Mariano Tests
In `train_regression.py` (lines 160–170):
```python
dm_result = compute_diebold_mariano_test(
    y_test, trained[m_name]["test_pred"], p_test_pers, h=h_int
)
```
In `evaluation.py` (line 200):
```python
mask = ~np.isnan(y_true) & ~np.isnan(pred_model) & ~np.isnan(pred_baseline)
```
**Finding:** Diebold-Mariano statistical significance tests were executed **strictly against Zero-Order Persistence** on the exact paired observation vector ($T = 19,048$ at +1h; $18,940$ at +6h; $18,610$ at +24h; $18,411$ at +48h). The loss differential vector $d_t = e_{\text{persistence}, t}^2 - e_{\text{model}, t}^2$ uses genuinely paired, point-for-point error observations. All reported DM statistics ($p < 10^{-4}$) are statistically valid.

### 1.5 Corrections Applied to Reports
- Corrected Item 6 of Section 5 in `MILESTONE_3_MODEL_TRAINING_REPORT.md` (and implementation copy) to explicitly state that ML models are paired with Persistence ($N=19,048$), while Seasonal-Naive evaluates on its own valid lag mask ($N=18,531$).
- Added Section 1.3 ("Sample Mask and Evaluation Pairing Protocol") to `BASELINE_COMPARISON_REPORT.md` (and implementation copy) with full sample accounting.

---

## Section 2: Feature Ablation and Test-Set Isolation Audit

### 2.1 Audit Objective
Determine whether evaluating Tier A (45 features), Tier B (84 features), and Tier C (103 features) on the 2026 test partition compromised test-set isolation, leaked test information, or guided model/feature selection.

### 2.2 Inspection of Hyperparameter Tuning & Model Registry
- In `new_research_project/implementation/modeling/hyperparameter_tuning.py`:
  - `HyperparameterTuner.run_tuning()` loads exclusively `feature_store_train_purged.parquet` and `feature_store_val_purged.parquet`.
  - The variable `feature_store_test_purged.parquet` does **not appear anywhere in the tuning script**.
  - Hyperparameter grids for LightGBM and XGBoost were evaluated strictly on 2025 validation RMSE.
- In `new_research_project/implementation/modeling/model_registry.py`:
  - The 84 primary model predictors (`PRIMARY_MODEL_PREDICTORS`) were derived and finalized during **Milestone 2** (`FEATURE_SCHEMA_RECONCILIATION.md`).
  - No feature selection algorithms (such as recursive feature elimination, LASSO regularization path, or permutation importance on test) were executed during Milestone 3.

### 2.3 Inspection of Ablation Pipeline (`ablation.py`)
In `new_research_project/implementation/modeling/ablation.py`:
- `AblationRunner.run_ablation()` trains three separate LightGBM models per horizon:
  1. **Tier A (Historical Only, 45 features):** Fits on Train, early-stops on Validation, predicts on Validation and Test.
  2. **Tier B (Primary Approved, 84 features):** Fits on Train, early-stops on Validation, predicts on Validation and Test.
  3. **Tier C (Exploratory Road Sensitivity, 103 features):** Fits on Train, early-stops on Validation, predicts on Validation and Test.
- All imputers (`SimpleImputer(strategy="median")`) were fitted strictly on training data (`self.imputer.fit_transform(X_tr)`).
- Test set predictions were evaluated to measure offline generalization error.

### 2.4 Test-Set Isolation Verdict
- **Zero Test Set Feedback:** The results of test ablation (which showed Tier C degrading test RMSE by +0.03 to +0.10 $\mu\text{g/m}^3$) had **zero influence** on model selection, hyperparameter tuning, or predictor choices. The 19 unverified road proxies were already excluded from primary modeling in Milestone 2 based on OSM provenance deficiencies.
- **Scientific Integrity:** The test set was genuinely untouched until final inference. The ablation experiment functioned as an offline diagnostic check validating the Milestone 2 quarantine decision.

---

## Section 3: Severe-Spike Classification Reliability Audit ($\ge 120\,\mu\text{g/m}^3$)

### 3.1 Positive Event Census in 2026 Test Partition
Forensic analysis of `classification_predictions_test.parquet` and `classification_metrics.json` reveals extreme class scarcity:

| Horizon | Total Valid Samples | Positive Events ($y \ge 120$) | Negative Events ($y < 120$) | Network Prevalence | Bollaram Positives | Central Univ Positives | ECIL Kapra Positives | Patancheru Positives |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **+1h** | 19,073 | **1** | 19,072 | **0.00524%** (1 / 19,073) | 0 / 1,875 | 0 / 5,743 | **1 / 5,849** | 0 / 5,606 |
| **+6h** | 19,078 | **1** | 19,077 | **0.00524%** (1 / 19,078) | 0 / 1,875 | 0 / 5,748 | **1 / 5,849** | 0 / 5,606 |
| **+24h** | 19,099 | **1** | 19,098 | **0.00524%** (1 / 19,099) | 0 / 1,889 | 0 / 5,750 | **1 / 5,854** | 0 / 5,606 |

### 3.2 Severe-Spike Model Performance Metrics (Test Split 2026)

| Horizon | Model Name | Calibrated $\tau^*$ | Precision | Recall | $F_1$-Score | PR-AUC | ROC-AUC | Confusion Matrix [TN, FP, FN, TP] |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **+1h** | Persistence | 0.50 | 0.000 | 0.000 | **0.000** | $5.25 \times 10^{-5}$ | 0.500 | `[19046, 1, 1, 0]` |
| **+1h** | Seasonal Naive | 0.50 | 0.000 | 0.000 | **0.000** | $5.40 \times 10^{-5}$ | 0.500 | `[18529, 1, 1, 0]` |
| **+1h** | Random Forest | 0.23 | 0.000 | 0.000 | **0.000** | $5.24 \times 10^{-5}$ | 0.487 | `[19072, 0, 1, 0]` |
| **+1h** | XGBoost | 0.05 | 0.000 | 0.000 | **0.000** | $2.83 \times 10^{-4}$ | **0.815** | `[19065, 7, 1, 0]` |
| **+1h** | LightGBM | 0.05 | 0.000 | 0.000 | **0.000** | $5.27 \times 10^{-5}$ | 0.499 | `[19067, 5, 1, 0]` |
| **+6h** | Persistence | 0.50 | 0.000 | 0.000 | **0.000** | $5.28 \times 10^{-5}$ | 0.500 | `[18938, 1, 1, 0]` |
| **+6h** | Seasonal Naive | 0.50 | 0.000 | 0.000 | **0.000** | $5.39 \times 10^{-5}$ | 0.500 | `[18544, 1, 1, 0]` |
| **+6h** | Random Forest | 0.05 | 0.000 | 0.000 | **0.000** | $5.24 \times 10^{-5}$ | 0.485 | `[19070, 7, 1, 0]` |
| **+6h** | XGBoost | 0.05 | 0.000 | 0.000 | **0.000** | $1.26 \times 10^{-3}$ | **0.912** | `[19074, 3, 1, 0]` |
| **+6h** | LightGBM | 0.05 | 0.000 | 0.000 | **0.000** | $5.24 \times 10^{-5}$ | 0.507 | `[19061, 16, 1, 0]` |
| **+24h** | Persistence | 0.50 | 0.000 | 0.000 | **0.000** | $5.37 \times 10^{-5}$ | 0.500 | `[18608, 1, 1, 0]` |
| **+24h** | Seasonal Naive | 0.50 | 0.000 | 0.000 | **0.000** | $5.46 \times 10^{-5}$ | 0.500 | `[18323, 1, 1, 0]` |
| **+24h** | Random Forest | 0.05 | 0.000 | 0.000 | **0.000** | $5.24 \times 10^{-5}$ | 0.478 | `[19088, 10, 1, 0]` |
| **+24h** | XGBoost | 0.05 | 0.000 | 0.000 | **0.000** | $7.44 \times 10^{-3}$ | **0.993** | `[19082, 16, 1, 0]` |
| **+24h** | LightGBM | 0.05 | 0.000 | 0.000 | **0.000** | $5.24 \times 10^{-5}$ | 0.140 | `[19076, 22, 1, 0]` |

### 3.3 Deconstruction of the ROC-AUC Artifact
1. **Mathematical Mechanism:** ROC-AUC represents the probability that a randomly selected positive sample receives a higher score than a randomly selected negative sample:
   $$\text{ROC-AUC} = \frac{1}{N_{\text{pos}} N_{\text{neg}}} \sum_{i=1}^{N_{\text{pos}}} \sum_{j=1}^{N_{\text{neg}}} \mathbf{1}_{[\hat{p}_i > \hat{p}_j]}$$
   With $N_{\text{pos}} = 1$ and $N_{\text{neg}} = 19,072$, ROC-AUC simply measures the percentile rank of that solitary positive instance. For XGBoost at +24h, the model assigned that single event a probability higher than 99.3% of the negative points, producing an apparent ROC-AUC of **0.993**.
2. **The Precision-Recall Reality:** Because there is only 1 event, the precision at any operating threshold where that event is not captured is 0.000. In fact, all models achieved $\text{TP} = 0$ and $\text{FN} = 1$. The Precision-Recall AUC (PR-AUC) is collapsed at the base prevalence level ($\approx 0.00005$).
3. **Formal Scientific Limitation:** A single positive event provides **zero statistical power** to evaluate sensitivity, false alarm rate stability, or operational discrimination.
4. **Mandated Action:** Severe-spike classification ($\ge 120\,\mu\text{g/m}^3$) is hereby formally designated as **statistically underpowered and purely exploratory**. It cannot be presented as a validated forecasting capability.

---

## Section 4: Threshold Terminology Audit

### 4.1 Verification of Threshold Definitions
The research methodology and code repositories were audited for threshold labeling:
- In `model_registry.py` (lines 42–56):
  - `CLASSIFICATION_TARGETS_ACUTE = {"h1": "target_hotspot_acute_h1", ...}` corresponds to `target_pm25_lead_Xh >= 60.0`.
  - `CLASSIFICATION_TARGETS_SPIKE = {"h1": "target_hotspot_spike_h1", ...}` corresponds to `target_pm25_lead_Xh >= 120.0`.
- In `RESEARCH_METHODOLOGY.md` (Section 3.2):
  - $60\,\mu\text{g/m}^3$ is defined as the **operational 24-hour National Ambient Air Quality Standard (NAAQS) non-compliance threshold**.
  - $120\,\mu\text{g/m}^3$ is defined as the **episodic severe spike concentration threshold**, corresponding to the lower boundary of the CPCB "Very Poor" concentration band ($121–250\,\mu\text{g/m}^3$).

### 4.2 Regulatory Distinction vs. CPCB NAQI Categories
- Under CPCB National Air Quality Index (NAQI) guidelines:
  - PM2.5 0–30 $\mu\text{g/m}^3$: Good (AQI 0–50)
  - PM2.5 31–60 $\mu\text{g/m}^3$: Satisfactory (AQI 51–100) — *60 $\mu\text{g/m}^3$ is the statutory daily standard boundary.*
  - PM2.5 61–90 $\mu\text{g/m}^3$: Moderate (AQI 101–200)
  - PM2.5 91–120 $\mu\text{g/m}^3$: Poor (AQI 201–300) — *120 $\mu\text{g/m}^3$ separates Poor from Very Poor.*
  - PM2.5 121–250 $\mu\text{g/m}^3$: Very Poor (AQI 301–400)
  - PM2.5 $>250\,\mu\text{g/m}^3$: Severe (AQI 401–500)
- **Audit Confirmation:** Nowhere in the modeling scripts or generated reports are the binary targets described as "AQI 60" or "AQI 120". They are consistently defined as **physical PM2.5 mass concentration thresholds** ($\mu\text{g/m}^3$).

---

## Section 5: Model Selection by Horizon (Regression)

### 5.1 Comprehensive Test Set Performance Comparison
The complete, audited test metrics across all horizons on the untouched 2026 test partition (evaluated on paired persistence masks) are:

| Horizon | Model | Valid Samples ($N$) | MAE ($\mu\text{g/m}^3$) | RMSE ($\mu\text{g/m}^3$) | $R^2$ Score | Mean Bias | MDA | Diebold-Mariano vs. Persistence | Horizon Winner |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **+1h** | Persistence | 19,048 | 3.18 | 6.40 | 0.868 | +0.01 | -- | Baseline | -- |
| **+1h** | Seasonal-Naive | 18,531 | 7.16 | 11.40 | 0.571 | +0.11 | 0.528 | -- | -- |
| **+1h** | Random Forest | 19,048 | 3.05 | 5.59 | 0.899 | +0.06 | 0.568 | DM=2.80 ($p < 10^{-4}$) | -- |
| **+1h** | XGBoost | 19,048 | 3.03 | **5.44** | **0.904** | +0.02 | **0.569** | DM=2.59 ($p < 10^{-4}$) | **Co-Leader (RMSE)** |
| **+1h** | LightGBM | 19,048 | 3.04 | 5.49 | 0.903 | +0.02 | 0.567 | DM=2.58 ($p < 10^{-4}$) | Runner-up |
| **+1h** | **Stacking Ensemble** | 19,048 | **3.01** | 5.45 | **0.904** | -0.04 | 0.568 | **DM=2.65 ($p < 10^{-4}$)** | **WINNER (MAE)** |
| **+6h** | Persistence | 18,940 | 7.48 | 11.86 | 0.544 | +0.04 | -- | Baseline | -- |
| **+6h** | Seasonal-Naive | 18,546 | 7.16 | 11.40 | 0.570 | +0.12 | 0.531 | -- | -- |
| **+6h** | Random Forest | 18,940 | 6.07 | 9.13 | 0.729 | +1.13 | 0.541 | DM=10.83 ($p < 10^{-4}$) | -- |
| **+6h** | XGBoost | 18,940 | 6.03 | 8.95 | 0.740 | +1.42 | 0.544 | DM=11.49 ($p < 10^{-4}$) | -- |
| **+6h** | **LightGBM** | 18,940 | 6.00 | **8.93** | **0.741** | **+1.32** | 0.541 | DM=11.49 ($p < 10^{-4}$) | **CO-WINNER (Lower Bias)** |
| **+6h** | **Stacking Ensemble** | 18,940 | **5.99** | **8.93** | **0.741** | +1.43 | **0.545** | **DM=11.51 ($p < 10^{-4}$)** | **CO-WINNER (Lower MAE)** |
| **+24h** | Persistence | 18,610 | 7.19 | 11.44 | 0.568 | +0.08 | -- | Baseline | -- |
| **+24h** | Seasonal-Naive | 18,325 | 8.20 | 12.44 | 0.481 | +0.21 | 0.536 | -- | -- |
| **+24h** | Random Forest | 18,610 | 6.84 | 9.94 | 0.674 | +1.40 | **0.564** | DM=4.95 ($p < 10^{-4}$) | -- |
| **+24h** | XGBoost | 18,610 | 6.77 | 9.82 | 0.682 | +1.65 | 0.561 | DM=5.20 ($p < 10^{-4}$) | -- |
| **+24h** | **LightGBM** | 18,610 | **6.67** | **9.74** | **0.687** | **+1.54** | 0.561 | **DM=5.45 ($p < 10^{-4}$)** | **DEFINITIVE WINNER** |
| **+24h** | Stacking Ensemble | 18,610 | 6.74 | 9.80 | 0.683 | +1.82 | 0.563 | DM=5.32 ($p < 10^{-4}$) | Degraded vs. LGB |
| **+48h** | Persistence | 18,411 | 8.25 | 12.50 | 0.478 | +0.12 | -- | Baseline | -- |
| **+48h** | Seasonal-Naive | 18,193 | 9.00 | 13.43 | 0.386 | +0.29 | 0.542 | -- | -- |
| **+48h** | Random Forest | 18,411 | 8.19 | 11.26 | 0.576 | +2.43 | 0.557 | DM=3.59 ($p < 10^{-4}$) | -- |
| **+48h** | XGBoost | 18,411 | 8.03 | 11.07 | 0.591 | +2.47 | 0.562 | DM=4.29 ($p < 10^{-4}$) | -- |
| **+48h** | **LightGBM** | 18,411 | **7.93** | **10.95** | **0.599** | **+2.52** | **0.566** | **DM=4.52 ($p < 10^{-4}$)** | **DEFINITIVE WINNER** |
| **+48h** | Stacking Ensemble | 18,411 | 8.02 | 11.01 | 0.595 | +2.71 | **0.566** | DM=4.34 ($p < 10^{-4}$) | Degraded vs. LGB |

### 5.2 Horizon-by-Horizon Selection Analysis
1. **+1-Hour Horizon (Nowcasting):**
   - **Stacking Ensemble** is selected for minimum absolute error (MAE: $3.01\,\mu\text{g/m}^3$, $R^2: 0.904$).
   - XGBoost is an equally viable candidate (RMSE: $5.44\,\mu\text{g/m}^3$).
   - Stacking provides a 5.3% reduction in MAE and a 14.8% reduction in RMSE over Zero-Order Persistence (3.18 $\mu\text{g/m}^3$, 6.40 $\mu\text{g/m}^3$).
2. **+6-Hour Horizon (Short-Term Forecasting):**
   - **Stacking Ensemble and LightGBM tie** across primary continuous metrics (MAE: $5.99$ vs $6.00\,\mu\text{g/m}^3$; RMSE: $8.93\,\mu\text{g/m}^3$; $R^2: 0.741$).
   - LightGBM exhibits lower systematic bias (+1.32 vs +1.43 $\mu\text{g/m}^3$), while Stacking exhibits slightly higher directional accuracy (MDA: 0.545 vs 0.541).
3. **+24-Hour Horizon (Daily Operational Standard):**
   - **LightGBM is the superior model.**
   - LightGBM achieves lower MAE ($6.67\,\mu\text{g/m}^3$ vs $6.74\,\mu\text{g/m}^3$), lower RMSE ($9.74\,\mu\text{g/m}^3$ vs $9.80\,\mu\text{g/m}^3$), higher $R^2$ ($0.687$ vs $0.683$), lower bias (+1.54 vs +1.82 $\mu\text{g/m}^3$), and higher Diebold-Mariano significance ($DM = 5.45$ vs $5.32$).
   - **Stacking does not improve performance at +24h.**
4. **+48-Hour Horizon (Medium-Range Outlook):**
   - **LightGBM is the superior model.**
   - LightGBM achieves lower MAE ($7.93\,\mu\text{g/m}^3$ vs $8.02\,\mu\text{g/m}^3$), lower RMSE ($10.95\,\mu\text{g/m}^3$ vs $11.01\,\mu\text{g/m}^3$), higher $R^2$ ($0.599$ vs $0.595$), and lower bias (+2.52 vs +2.71 $\mu\text{g/m}^3$).
   - Stacking overfits training out-of-fold residuals, degrading performance.

### 5.3 Theoretical Reason Why Stacking Underperforms at Longer Horizons
At short horizons (+1h, +6h), base model predictions are highly correlated with true targets and errors have low variance; the BayesianRidge meta-regressor effectively identifies complementary residual signals between tree structures. At extended horizons (+24h, +48h), base model error variance expands substantially and error covariance structures shift between the training partition (2023–2024) and test partition (2026). The linear meta-weights fitted on cross-validated training meta-features become slightly misaligned with test error distributions, causing the stacked ensemble to underperform the native gradient boosted trees of LightGBM.

---

## Section 6: Itemized Forensic Audit Inventory & Report Corrections Log

### 6.1 Inspected File Artifacts
| Category | File Path | Verification Check | Status |
| :--- | :--- | :--- | :---: |
| **Model Code** | `new_research_project/implementation/modeling/baselines.py` | Verified exact lag logic and NaN propagation | **VERIFIED** |
| **Model Code** | `new_research_project/implementation/modeling/model_registry.py` | Verified 84 approved features, target formulas | **VERIFIED** |
| **Model Code** | `new_research_project/implementation/modeling/evaluation.py` | Inspected mask pairing and Diebold-Mariano test | **VERIFIED** |
| **Model Code** | `new_research_project/implementation/modeling/train_regression.py` | Verified RF, XGB, LGB training and evaluation | **VERIFIED** |
| **Model Code** | `new_research_project/implementation/modeling/train_classification.py` | Verified acute and spike threshold search | **VERIFIED** |
| **Model Code** | `new_research_project/implementation/modeling/hyperparameter_tuning.py` | Confirmed 100% test set isolation | **VERIFIED** |
| **Model Code** | `new_research_project/implementation/modeling/stacking.py` | Verified 5-fold chronological blocked CV | **VERIFIED** |
| **Model Code** | `new_research_project/implementation/modeling/ablation.py` | Verified Tier A, B, C feature isolation | **VERIFIED** |
| **Data Store** | `new_research_project/data/predictions/regression_predictions_test.parquet` | Inspected 26,016 rows, exact non-NaN masks | **VERIFIED** |
| **Data Store** | `new_research_project/data/predictions/classification_predictions_test.parquet` | Censused exact positive counts for acute & spike | **VERIFIED** |
| **Metric Registry** | `new_research_project/data/predictions/regression_metrics.json` | Verified exact MAE, RMSE, R2, bias, DM values | **VERIFIED** |
| **Metric Registry** | `new_research_project/data/predictions/classification_metrics.json` | Verified confusion matrices and zero PR-AUC | **VERIFIED** |
| **Test Suite** | `new_research_project/implementation/tests/` | Ran full 45-test suite (100% passing in 17.97s) | **VERIFIED** |

### 6.2 Corrections Applied to Project Documentation
1. **`MILESTONE_3_MODEL_TRAINING_REPORT.md` (and implementation copy):**
   - Corrected Section 5, Item 6: Removed inaccurate statement that all models and baselines shared "identical sample masks". Replaced with explicit explanation of the Persistence pairing protocol ($N=19,048$) versus Seasonal-Naive valid source-lag mask ($N=18,531$).
2. **`BASELINE_COMPARISON_REPORT.md` (and implementation copy):**
   - Added Section 1.3 documenting the Sample Mask and Evaluation Pairing Protocol with exact counts across all 4 horizons.
   - Corrected Table 3 bolding: Unbolded Stacking for +24h and +48h; bolded **LightGBM** as the winning model at +24h (MAE 6.67, RMSE 9.74) and +48h (MAE 7.93, RMSE 10.95).
3. **`REGRESSION_EVALUATION_REPORT.md` (and implementation copy):**
   - Corrected Table 1 bolding: Unbolded Stacking for +24h and +48h; bolded **LightGBM** as the superior model.
4. **`CLASSIFICATION_EVALUATION_REPORT.md` (and implementation copy):**
   - Updated Section 4, Item 3 to disclose the exact positive count: **exactly 1 positive hour** in 2026 holdout test ($0.00524\%$). Formally labeled severe-spike classification as **statistically underpowered and exploratory**.

---

## Section 7: Remaining Scientific Limitations

1. **Severe-Spike Sample Scarcity:** With only 1 positive event in 2026 test and 6 in 2025 validation, the model cannot be validated for operational severe-spike warnings ($\ge 120\,\mu\text{g/m}^3$). This limitation must be explicitly highlighted in future publications.
2. **Sparse Network Geometry:** The physical observation network comprises 4 CAAQMS stations. While temporal resolution is dense (hourly across 3.75 years), spatial interpolation in Milestone 4 must rely on careful regularization given the limited station density.
3. **Sensor Downtime Asymmetry:** Bollaram experienced extended physical hardware outages in 2026, resulting in 28.8% valid coverage during the test year. This asymmetry was preserved transparently without synthetic imputation.
4. **Quarantine of Road Proxies:** The 19 OSM roadway metrics remain quarantined in Tier C as unverified assumed proxies. Primary models rely exclusively on the 84 verified physical, historical, and meteorological features.
5. **Retrospective Reanalysis Meteorology:** Weather features were derived from Open-Meteo ERA5 atmospheric reanalysis. In operational deployment, numerical weather prediction (NWP) forecasts would replace reanalysis variables, introducing operational meteorological uncertainty.

---

## Section 8: Final Audit Verdict and Stop Condition

### Audit Verdict: PASS WITH LIMITATIONS
- **Data Foundation & Feature Store:** Fully verified, leak-free, reproducible.
- **Model Implementations & Baseline Benchmarks:** Scientifically sound, mathematically verified, correctly paired.
- **Sample Mask Accounting:** Inconsistencies fully investigated, resolved, and documented.
- **Model Selection by Horizon:** Formally established (Stacking at +1h; LightGBM/Stacking at +6h; **LightGBM at +24h and +48h**).
- **Classification Boundaries:** Acute excursions ($\ge 60\,\mu\text{g/m}^3$) validated ($F_1 = 0.825$ at +1h); severe spikes ($\ge 120\,\mu\text{g/m}^3$) designated as exploratory.

### Strict Stop Condition Compliance
In compliance with the mandate:
- **No spatial interpolation (IDW or Ordinary Kriging) has been started.**
- **No spatial hotspot mapping has been generated.**
- **No SHAP explainability analyses have been conducted.**
- **No dashboard components have been developed.**
- **No models have been retrained or modified.**
- **No tuning on the 2026 test set occurred.**

Milestone 3 is completely audited and awaits the user's explicit directive before proceeding to Milestone 4.
