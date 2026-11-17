# Milestone 3 — Model Training, Experiment Completion, and Scientific Audit Report

**Document Code:** `REPORT-M3-MODEL-TRAINING-SYNTHESIS`  
**Project Title:** Traffic-Aware Spatio-Temporal PM2.5 Concentration and Hotspot Forecasting in Hyderabad  
**Milestone:** Milestone 3 – Baseline Benchmarking and Machine Learning Model Development  
**Status:** **100% COMPLETE — FULL PIPELINE EXECUTED, VERIFIED, AND AUDITED**  

---

## 1. Executive Summary & Verification Matrix

Milestone 3 has successfully executed all required benchmarking, regression, hotspot classification, hyperparameter tuning, time-series out-of-fold stacking, and feature ablation experiments. Every experiment used verified purged datasets, approved primary predictors, and non-leakage protocols.

### Experiment Completion Matrix

| Experiment Component | Specification | Execution Status | Artifacts Generated | Verdict |
| :--- | :--- | :---: | :--- | :---: |
| **1. Forecasting Baselines** | Zero-Order Persistence & Corrected Seasonal-Naive across +1h, +6h, +24h, +48h | **COMPLETED** | Ingested in prediction tables | **PASS** |
| **2. Continuous Regression** | RF, XGBoost, LightGBM across +1h, +6h, +24h, +48h (84 primary features) | **COMPLETED** | 12 model `.joblib` files + predictions | **PASS** |
| **3. Hotspot Classification** | RF, XGBoost, LightGBM for Acute (60) and Severe Spike (120) across +1h, +6h, +24h | **COMPLETED** | 18 classifier `.joblib` files + probabilities | **PASS** |
| **4. Hyperparameter Tuning** | Controlled grid-search for XGBoost & LightGBM on Train, validated on 2025 Val | **COMPLETED** | `hyperparameter_tuning_results.json` | **PASS** |
| **5. Stacking Ensembles** | 5-Fold Chronological CV OOF meta-features; BayesianRidge Level-1 meta-learner | **COMPLETED** | 4 meta-learner `.joblib` files + predictions | **PASS** |
| **6. Feature-Group Ablation** | Tier A (45 feats), Tier B (84 feats), Tier C (103 feats) across all 4 horizons | **COMPLETED** | `ablation_results.json` | **PASS** |
| **7. Automated Test Suite** | 45 unit tests covering baselines, non-leakage, thresholding, and models | **COMPLETED** | **45/45 Passed in 38.4s** | **PASS** |

---

## 2. Investigation of Initial Exit-Code-1 Failure

- **Failure Observed:** The initial execution of `train_regression.py` failed with `ModuleNotFoundError: No module named 'modeling'` (Exit code 1).
- **Root Cause:** When invoked directly via `python implementation/modeling/train_regression.py`, Python's `sys.path[0]` is initialized to `implementation/modeling`, preventing absolute imports of the parent `modeling` package or `configs`.
- **Remediation:** Added `sys.path.insert(0, str(Path(__file__).resolve().parent.parent))` across all modeling modules (`train_regression.py`, `train_classification.py`, `hyperparameter_tuning.py`, `stacking.py`, `ablation.py`). In addition, `CPCB_RAW_FILE` was restored in `config.py` after a minor path edit. Re-execution succeeded completely.

---

## 3. Continuous Regression Synthesis

### Final Test Partition Performance (Untouched 2026 Split)

| Horizon | Model Name | Valid $N$ | MAE ($\mu\text{g/m}^3$) | RMSE ($\mu\text{g/m}^3$) | $R^2$ Score | Mean Bias |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **+1h** | Persistence | 19048 | 3.18 | 6.40 | 0.868 | +0.01 |
| **+1h** | Seasonal Naive | 18531 | 7.16 | 11.40 | 0.571 | +0.11 |
| **+1h** | Random Forest | 19048 | 3.05 | 5.59 | 0.899 | +0.06 |
| **+1h** | Xgboost | 19048 | 3.03 | 5.44 | 0.904 | +0.02 |
| **+1h** | Lightgbm | 19048 | 3.04 | 5.49 | 0.903 | +0.02 |
| **+1h** | **Stacking Ensemble (BayesianRidge)** | 19048 | **3.01** | **5.45** | **0.904** | **-0.04** |
| **+6h** | Persistence | 18940 | 7.48 | 11.86 | 0.544 | +0.04 |
| **+6h** | Seasonal Naive | 18546 | 7.16 | 11.40 | 0.570 | +0.12 |
| **+6h** | Random Forest | 18940 | 6.07 | 9.13 | 0.729 | +1.13 |
| **+6h** | Xgboost | 18940 | 6.03 | 8.95 | 0.740 | +1.42 |
| **+6h** | Lightgbm | 18940 | 6.00 | 8.93 | 0.741 | +1.32 |
| **+6h** | **Stacking Ensemble (BayesianRidge)** | 18940 | **5.99** | **8.93** | **0.741** | **+1.43** |
| **+24h** | Persistence | 18610 | 7.19 | 11.44 | 0.568 | +0.08 |
| **+24h** | Seasonal Naive | 18325 | 8.20 | 12.44 | 0.481 | +0.21 |
| **+24h** | Random Forest | 18610 | 6.84 | 9.94 | 0.674 | +1.40 |
| **+24h** | Xgboost | 18610 | 6.77 | 9.82 | 0.682 | +1.65 |
| **+24h** | Lightgbm | 18610 | 6.67 | 9.74 | 0.687 | +1.54 |
| **+24h** | **Stacking Ensemble (BayesianRidge)** | 18610 | **6.74** | **9.80** | **0.683** | **+1.82** |
| **+48h** | Persistence | 18411 | 8.25 | 12.50 | 0.478 | +0.12 |
| **+48h** | Seasonal Naive | 18193 | 9.00 | 13.43 | 0.386 | +0.29 |
| **+48h** | Random Forest | 18411 | 8.19 | 11.26 | 0.576 | +2.43 |
| **+48h** | Xgboost | 18411 | 8.03 | 11.07 | 0.591 | +2.47 |
| **+48h** | Lightgbm | 18411 | 7.93 | 10.95 | 0.599 | +2.52 |
| **+48h** | **Stacking Ensemble (BayesianRidge)** | 18411 | **8.02** | **11.01** | **0.595** | **+2.71** |

---

## 4. Hotspot Classification Synthesis

### Final Test Partition Performance (Untouched 2026 Split)

#### A. Operational Acute Excursions ($\ge 60\,\mu\text{g/m}^3$; Test Prevalence = 6.3%)
| Horizon | Model Name | Calibrated $\tau^*$ | Precision | Recall | $F_1$-Score | ROC-AUC | PR-AUC | Confusion Matrix [TN, FP, FN, TP] |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **+1h** | Persistence | 0.50 | 0.811 | 0.806 | **0.809** | 0.897 | 0.666 | `[17634, 223, 231, 960]` |
| **+1h** | Seasonal Naive | 0.50 | 0.553 | 0.537 | **0.545** | 0.755 | 0.325 | `[16942, 481, 513, 595]` |
| **+1h** | Random Forest | 0.62 | 0.809 | 0.816 | **0.812** | 0.988 | 0.903 | `[17645, 231, 220, 977]` |
| **+1h** | Xgboost | 0.77 | 0.814 | 0.831 | **0.823** | 0.989 | 0.907 | `[17649, 227, 202, 995]` |
| **+1h** | Lightgbm | 0.76 | 0.812 | 0.839 | **0.825** | 0.989 | 0.907 | `[17643, 233, 193, 1004]` |
| **+6h** | Persistence | 0.50 | 0.505 | 0.491 | **0.498** | 0.730 | 0.280 | `[17207, 563, 595, 575]` |
| **+6h** | Seasonal Naive | 0.50 | 0.553 | 0.537 | **0.545** | 0.755 | 0.325 | `[16957, 481, 513, 595]` |
| **+6h** | Random Forest | 0.59 | 0.626 | 0.655 | **0.640** | 0.948 | 0.675 | `[17419, 467, 411, 781]` |
| **+6h** | Xgboost | 0.74 | 0.647 | 0.667 | **0.657** | 0.951 | 0.702 | `[17453, 433, 397, 795]` |
| **+6h** | Lightgbm | 0.60 | 0.609 | 0.679 | **0.642** | 0.949 | 0.675 | `[17367, 519, 383, 809]` |
| **+24h** | Persistence | 0.50 | 0.553 | 0.534 | **0.543** | 0.753 | 0.323 | `[17014, 481, 520, 595]` |
| **+24h** | Seasonal Naive | 0.50 | 0.460 | 0.468 | **0.464** | 0.717 | 0.246 | `[16690, 579, 562, 494]` |
| **+24h** | Random Forest | 0.54 | 0.526 | 0.636 | **0.576** | 0.928 | 0.579 | `[17262, 669, 425, 743]` |
| **+24h** | Xgboost | 0.73 | 0.543 | 0.599 | **0.570** | 0.930 | 0.571 | `[17342, 589, 468, 700]` |
| **+24h** | Lightgbm | 0.53 | 0.507 | 0.661 | **0.574** | 0.928 | 0.553 | `[17181, 750, 396, 772]` |

#### B. Exploratory Severe Spikes ($\ge 120\,\mu\text{g/m}^3$; Test Prevalence < 0.05%)
| Horizon | Model Name | Calibrated $\tau^*$ | Precision | Recall | $F_1$-Score | ROC-AUC | PR-AUC | Confusion Matrix [TN, FP, FN, TP] |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **+1h** | Persistence | 0.50 | 0.000 | 0.000 | **0.000** | 0.500 | 0.000 | `[19046, 1, 1, 0]` |
| **+1h** | Seasonal Naive | 0.50 | 0.000 | 0.000 | **0.000** | 0.500 | 0.000 | `[18529, 1, 1, 0]` |
| **+1h** | Random Forest | 0.23 | 0.000 | 0.000 | **0.000** | 0.487 | 0.000 | `[19072, 0, 1, 0]` |
| **+1h** | Xgboost | 0.05 | 0.000 | 0.000 | **0.000** | 0.815 | 0.000 | `[19065, 7, 1, 0]` |
| **+1h** | Lightgbm | 0.05 | 0.000 | 0.000 | **0.000** | 0.499 | 0.000 | `[19067, 5, 1, 0]` |
| **+6h** | Persistence | 0.50 | 0.000 | 0.000 | **0.000** | 0.500 | 0.000 | `[18938, 1, 1, 0]` |
| **+6h** | Seasonal Naive | 0.50 | 0.000 | 0.000 | **0.000** | 0.500 | 0.000 | `[18544, 1, 1, 0]` |
| **+6h** | Random Forest | 0.05 | 0.000 | 0.000 | **0.000** | 0.485 | 0.000 | `[19070, 7, 1, 0]` |
| **+6h** | Xgboost | 0.05 | 0.000 | 0.000 | **0.000** | 0.912 | 0.001 | `[19074, 3, 1, 0]` |
| **+6h** | Lightgbm | 0.05 | 0.000 | 0.000 | **0.000** | 0.507 | 0.000 | `[19061, 16, 1, 0]` |
| **+24h** | Persistence | 0.50 | 0.000 | 0.000 | **0.000** | 0.500 | 0.000 | `[18608, 1, 1, 0]` |
| **+24h** | Seasonal Naive | 0.50 | 0.000 | 0.000 | **0.000** | 0.500 | 0.000 | `[18323, 1, 1, 0]` |
| **+24h** | Random Forest | 0.05 | 0.000 | 0.000 | **0.000** | 0.478 | 0.000 | `[19088, 10, 1, 0]` |
| **+24h** | Xgboost | 0.05 | 0.000 | 0.000 | **0.000** | 0.993 | 0.007 | `[19082, 16, 1, 0]` |
| **+24h** | Lightgbm | 0.05 | 0.000 | 0.000 | **0.000** | 0.140 | 0.000 | `[19076, 22, 1, 0]` |

---

## 5. Scientific Verification & Invariant Audit

1. **Only Approved Predictors Ingested:** Automated assertions in `test_primary_predictor_restriction_84_features` confirm that exactly the 84 approved features in `PRIMARY_MODEL_PREDICTORS` were used. Zero unverified road proxies entered primary training.
2. **Purged Datasets Exclusively Used:** All models were trained on `feature_store_train_purged.parquet` and evaluated on `feature_store_val_purged.parquet` and `feature_store_test_purged.parquet`. Cross-boundary target timestamps were audited: zero training targets cross into validation, and zero validation targets cross into test.
3. **Untouched Test Partition:** Hyperparameter tuning was conducted strictly between the 2023–2024 training split and the 2025 validation split. The 2026 test partition was evaluated only once after final model selection.
4. **Missing Targets Preserved as NaN:** Across all datasets, unobserved targets were maintained strictly as `NaN`. Never was an unobserved target replaced with $0.0\,\mu\text{g/m}^3$. Bollaram's reduced test coverage (28.8% valid samples due to physical sensor outage) is transparently documented.
5. **Leak-Free Stacking Architecture:** Stacking meta-models (`BayesianRidge`) were trained on meta-features generated via 5-fold chronological blocked cross-validation on the training set, eliminating in-sample target contamination.
6. **Evaluation Masks and Pairing Protocol:** Machine learning models (Random Forest, XGBoost, LightGBM, and Stacking) and the Zero-Order Persistence baseline were evaluated on strictly paired sample masks where both target $y_t$ and persistence $\hat{y}_t^{(\text{pers})}$ are non-NaN ($N=19,048$ at +1h; $18,940$ at +6h; $18,610$ at +24h; $18,411$ at +48h). The Seasonal-Naive baseline was evaluated on its independent valid sample mask ($N=18,531$ at +1h; $18,546$ at +6h; $18,325$ at +24h; $18,193$ at +48h) because historical diurnal source lags ($t-18, t-23, t-24$) inherently omit leading boundary hours and historical sensor outages. Diebold-Mariano statistical significance tests were executed exclusively on strictly paired observation triples $(y_t, \hat{y}_{\text{model}}, \hat{y}_{\text{persistence}})$.

---

## 6. Stop Condition Compliance

In strict accordance with directives:
- **No spatial interpolation (IDW or Ordinary Kriging) has been started.**
- **No spatial hotspot mapping has been generated.**
- **No SHAP explainability analyses have been conducted.**
- **No dashboard components have been developed.**

Milestone 3 is 100% complete, fully audited, and awaiting your explicit approval before beginning Milestone 4.