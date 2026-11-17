# Milestone 3 — Hotspot Classification and Severe-Spike Evaluation Report

**Document Code:** `REPORT-M3-CLASSIFICATION-EVALUATION`  
**Project Title:** Traffic-Aware Spatio-Temporal PM2.5 Concentration and Hotspot Forecasting in Hyderabad  
**Milestone:** Milestone 3 – Baseline Benchmarking and Machine Learning Model Development  
**Status:** **AUDITED AND VERIFIED AGAINST ACTUAL TEST RESULTS**  

---

## 1. Executive Summary & Target Formulations

Two distinct binary classification target regimes were evaluated across forecasting horizons $+1\text{h}$, $+6\text{h}$, and $+24\text{h}$:
1. **Operational Acute Excursions ($\ge 60\,\mu\text{g/m}^3$):**
   Detects instantaneous exceedances of the operational 24-hour statutory benchmark level ($60\,\mu\text{g/m}^3$). Class prevalence is **11.7% in Validation** (2025) and **6.3% in Test** (2026).
2. **Exploratory Severe Spikes ($\ge 120\,\mu\text{g/m}^3$):**
   Detects extreme episodic spikes ($120\,\mu\text{g/m}^3$, matching NAQI Very Poor boundary). Class prevalence is **ultra-rare (< 0.05% in Validation and Test)**.

### Threshold Calibration Protocol
All ML models had their decision thresholds $\tau^*$ calibrated strictly on the 2025 validation set via deterministic grid search ($[0.05, 0.95]$, step 0.01) maximizing $F_1$-score. Test set predictions were evaluated strictly at these pre-calibrated thresholds.

---

## 2. Acute Excursions ($\ge 60\,\mu\text{g/m}^3$) Network Test Performance

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

---

## 3. Exploratory Severe Spikes ($\ge 120\,\mu\text{g/m}^3$) Network Test Performance

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

## 4. Key Classification Findings
1. **High Precision and Recall for Acute Excursions:** For $+1\text{h}$ acute exceedances, LightGBM achieved an $F_1$-score of **0.825** with ROC-AUC of **0.989** and PR-AUC of **0.907**. XGBoost achieved $F_1 = 0.823$ with PR-AUC = 0.907. Both significantly outperform the Persistence ($F_1 = 0.809$) and Seasonal-Naive ($F_1 = 0.545$) baselines.
2. **Extended Horizon Discrimination:** At $+6\text{h}$, XGBoost achieves an $F_1$-score of **0.657** (Recall = 0.667, ROC-AUC = 0.951), substantially outperforming Persistence ($F_1 = 0.498$). At $+24\text{h}$, Random Forest achieves $F_1 = 0.576$ (ROC-AUC = 0.928).
3. **Severe Spike Ultra-Rarity & Extreme Imbalance Limitation:** Exceedances of the $\ge 120\,\mu\text{g/m}^3$ concentration threshold occurred in **exactly 1 single hour** out of 19,073 test hours in 2026 (a single event at ECIL Kapra; 0 occurrences at Bollaram, Central University, and ICRISAT Patancheru). While XGBoost assigned this solitary positive observation a high relative probability rank (producing deceptive arithmetic ROC-AUC values of 0.815 at +1h, 0.912 at +6h, and 0.993 at +24h), PR-AUC remains essentially zero ($\le 0.007$) and $F_1 = 0.000$ (TP=0, FN=1). This confirms that ROC-AUC is misleading under extreme imbalance ($N_{\text{pos}}=1$). Consequently, severe-spike classification is **statistically underpowered, cannot support claims of operational capability, and must be treated strictly as an exploratory sensitivity benchmark**.
