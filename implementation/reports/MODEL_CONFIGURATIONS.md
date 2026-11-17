# Milestone 3 Model Configurations, Hyperparameter Search Grids, and Architectural Registry

**Document Code:** `SPEC-M3-MODEL-CONFIGURATIONS`  
**Project Title:** Traffic-Aware Spatio-Temporal PM2.5 Concentration and Hotspot Forecasting in Hyderabad Using Multi-Source Environmental Data and Explainable Machine Learning  
**Milestone:** Milestone 3 – Baseline Benchmarking and Machine Learning Model Development  
**Status:** **ACTIVE SPECIFICATION & EXPERIMENTAL REGISTRY**  

---

## 1. Overview & Reproducibility Mandate

This document specifies the exact inductive biases, hyperparameters, random seeds, training protocols, and feature spaces for all machine learning models, ensembling architectures, and forecasting baselines evaluated in Milestone 3.

### Core Governance Rules
1. **Zero Future Lookahead:** All models condition strictly on historical and contemporaneous observations observed up to forecast origin $t$ ($s \le t$).
2. **Purged Partitions Exclusively:** Models are trained on `feature_store_train_purged.parquet` (69,984 rows), validated on `feature_store_val_purged.parquet` (34,848 rows), and tested on `feature_store_test_purged.parquet` (26,016 rows).
3. **Approved Primary Feature Restriction:** All primary models ingest strictly the **84 approved Primary Model Predictors** (`PRIMARY_MODEL_PREDICTORS`). All 19 unverified OpenStreetMap road proxies and interaction features are barred from primary configurations.
4. **Target Decoupling & No Imputation:** Prospective targets (+1h, +6h, +24h, +48h) are evaluated on physical raw sensor ground truth. Missing targets remain strictly `NaN` and are never substituted with $0.0\,\mu\text{g/m}^3$.
5. **Class Imbalance Handled on Training Only:** Positive class re-weighting (`scale_pos_weight` and `class_weight='balanced'`) is calculated strictly from the 2023–2024 training partition.

---

## 2. Baseline Model Specifications

### 2.1 Zero-Order Persistence Baseline (Markovian)
- **Mathematical Formulation:**
  $$\hat{y}_{i, t+h}^{(\text{Persistence})} = y_{i, t} \quad \forall h \in \{1, 6, 24, 48\}\,\text{hours}$$
- **Inductive Bias:** Assumes atmospheric inertia and persistence dominate short-term ambient pollutant concentrations.
- **Missing Source Protocol:** If $y_{i, t}$ is unobserved (`NaN`), $\hat{y}_{i, t+h} = \text{NaN}$. Excluded from metric computation; zero substitution is prohibited.
- **Hotspot Formulation:**
  $$\hat{z}_{i, t+h}^{(\text{Persistence})} = \mathbb{I}\left(y_{i, t} \ge \tau\right) \quad \text{for } \tau \in \{60, 120\}\,\mu\text{g/m}^3$$

### 2.2 Corrected Seasonal-Naive Baseline (Diurnal Cycle $S=24\,\text{hours}$)
- **Mathematical Formulation:**
  Maps strictly to the single exact corresponding hour of the preceding diurnal cycle observed prior to $t$:
  - **+1h Horizon:** $\hat{y}_{i, t+1} = y_{i, t-23}$ (strictly past, lag 23h)
  - **+6h Horizon:** $\hat{y}_{i, t+6} = y_{i, t-18}$ (strictly past, lag 18h)
  - **+24h Horizon:** $\hat{y}_{i, t+24} = y_{i, t-24}$ (strictly past, lag 24h)
  - **+48h Horizon:** $\hat{y}_{i, t+48} = y_{i, t-24}$ (single exact source timestamp rule per Section 8.5)
- **Inductive Bias:** Assumes diurnal solar insolation, planetary boundary layer dynamics, and morning/evening traffic rush cycles repeat periodically every 24 hours.
- **Missing Source Protocol:** If $y_{i, \tau}$ is unobserved, $\hat{y}_{i, t+h} = \text{NaN}$.

---

## 3. Tree Ensemble Model Configurations

All tree ensemble models utilize fixed random seed `RANDOM_SEED = 42` and full CPU parallelization (`n_jobs = -1`).

```
+-------------------------------------------------------------------------------------------------------------------+
|                                     TREE ENSEMBLE HYPERPARAMETER SPECIFICATIONS                                   |
+----------------------+----------------------------+-----------------------------------+---------------------------+
| Model Family         | Parameter Name             | Configured Value / Default        | Operational Function      |
+----------------------+----------------------------+-----------------------------------+---------------------------+
| Random Forest        | n_estimators               | 300                               | Number of bagged trees    |
|                      | max_depth                  | 16                                | Maximum tree depth        |
|                      | min_samples_split          | 5                                 | Min samples for split     |
|                      | min_samples_leaf           | 2                                 | Min samples per leaf      |
|                      | max_features               | 'sqrt' (classification) / 1.0 (reg)| Random feature subsampling|
|                      | class_weight               | 'balanced' (classification only)  | Inverse frequency weights |
+----------------------+----------------------------+-----------------------------------+---------------------------+
| XGBoost              | n_estimators               | 300                               | Gradient boosting rounds  |
|                      | learning_rate (eta)        | 0.03                              | Shrinkage rate            |
|                      | max_depth                  | 6                                 | Exact greedy tree depth   |
|                      | subsample                  | 0.80                              | Row subsampling ratio     |
|                      | colsample_bytree           | 0.80                              | Column subsampling ratio  |
|                      | reg_alpha (L1)             | 0.10                              | L1 regularization weight  |
|                      | reg_lambda (L2)            | 1.00                              | L2 regularization weight  |
|                      | tree_method                | 'hist'                            | Fast histogram binning    |
|                      | scale_pos_weight           | N_neg / N_pos (train only)        | Positive class multiplier |
+----------------------+----------------------------+-----------------------------------+---------------------------+
| LightGBM             | n_estimators               | 300                               | Gradient boosting rounds  |
|                      | learning_rate              | 0.03                              | Shrinkage rate            |
|                      | num_leaves                 | 31                                | Leaf-wise split complexity|
|                      | max_depth                  | -1 (unconstrained by depth)       | Leaf-wise growth strategy |
|                      | subsample                  | 0.80                              | Bagging fraction          |
|                      | colsample_bytree           | 0.80                              | Feature fraction          |
|                      | reg_alpha (L1)             | 0.10                              | L1 regularization weight  |
|                      | reg_lambda (L2)            | 1.00                              | L2 regularization weight  |
|                      | early_stopping_rounds      | 50 (on validation loss)           | Overfitting prevention    |
|                      | scale_pos_weight           | N_neg / N_pos (train only)        | Positive class multiplier |
+----------------------+----------------------------+-----------------------------------+---------------------------+
```

---

## 4. Time-Series Out-of-Fold Stacking Architecture

### Level-0 Base Estimators
- `RandomForestRegressor` / `RandomForestClassifier`
- `XGBRegressor` / `XGBClassifier`
- `LGBMRegressor` / `LGBMClassifier`

### Time-Series Out-of-Fold Cross-Validation Protocol
- **Cross-Validation Scheme:** 5-fold chronological blocked split on the 2023–2024 training partition.
- **Blocking Mechanism:** Timestamps across all four monitoring stations are partitioned into 5 contiguous chronological blocks. In each fold $k \in \{1, \dots, 5\}$, base models are trained on past/available training blocks and predict exclusively on the $k$-th validation block.
- **Guarantee:** Zero future-to-past target information is exposed when generating out-of-fold meta-features.

### Level-1 Meta-Learners
- **Regression Meta-Learner:** `BayesianRidge(compute_score=True)`  
  Estimates posterior distributions over base model blend weights with automatic regularization parameter estimation.
- **Classification Meta-Learner:** `LogisticRegression(penalty='l2', C=1.0, random_state=42)`  
  Blends base model probability estimates using L2-regularized logistic regression.

---

## 5. Decision Threshold Optimization Algorithm for Hotspot Alerts

Because urban $\text{PM}_{2.5}$ exceedances exhibit substantial class imbalance, the standard threshold $\tau = 0.50$ produces poor recall for extreme events.

### The Deterministic Grid-Search Optimization Algorithm
1. **Search Space:**
   $$\mathcal{T} = \{\tau \in [0.05, 0.95] \mid \tau = 0.05 + 0.01 \cdot k, \; k \in \{0, 1, \dots, 90\}\}$$
2. **Optimization Criterion (on 2025 Validation Partition):**
   $$\tau^* = \arg\max_{\tau \in \mathcal{T}} F_1\left(\tau; \mathcal{D}_{\text{val}}\right) = \arg\max_{\tau \in \mathcal{T}} \frac{2 \cdot \text{Precision}(\tau) \cdot \text{Recall}(\tau)}{\text{Precision}(\tau) + \text{Recall}(\tau)}$$
3. **Deterministic Tie-Breaking:**
   If multiple thresholds achieve identical maximum $F_1$-score (within $10^{-5}$), select the threshold $\tau^*$ that minimizes absolute distance to the empirical validation positive prevalence:
   $$\tau^* = \arg\min_{\tau \in \arg\max F_1} |\tau - \text{Prevalence}(\mathcal{D}_{\text{val}})|$$
4. **Test Set Evaluation Protocol:**
   The 2026 test partition is evaluated **strictly at the pre-calibrated threshold $\tau^*$**. No threshold search or tuning is performed on test labels.

---

## 6. Feature-Tier Ablation Architecture

```
+-------------------------------------------------------------------------------------------------------------------+
|                                      CUMULATIVE FEATURE-TIER ABLATION HIERARCHY                                   |
+--------+---------------------------------------+-------------+----------------------------------------------------+
| Tier   | Tier Nomenclature                     | Feature Qty | Component Modalities                               |
+--------+---------------------------------------+-------------+----------------------------------------------------+
| Tier A | Historical Pollution & Calendar Only  | 45 features | In-situ PM2.5, AR lags, rolling stats, dynamic rate|
|        |                                       |             | of change, co-pollutant ratios, cyclical calendar  |
+--------+---------------------------------------+-------------+----------------------------------------------------+
| Tier B | Primary Physical & Meteorological     | 84 features | Tier A + Open-Meteo ERA5 atmospheric thermodynamics|
|        | (Approved Primary Model Configuration)|             | wind vectors, stability + temporal traffic windows |
+--------+---------------------------------------+-------------+----------------------------------------------------+
| Tier C | Exploratory Road Proxy Sensitivity    | 103 features| Tier B + 19 unverified OpenStreetMap static roadway|
|        | (Quarantined Sensitivity Ablation)    |             | metrics and traffic pressure interaction terms     |
+--------+---------------------------------------+-------------+----------------------------------------------------+
```

---

*Specification certified for Milestone 3 execution.*
