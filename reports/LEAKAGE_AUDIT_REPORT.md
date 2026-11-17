# Comprehensive Feature Engineering Leakage Audit Report

**Document Code:** `REPORT-M2-LEAKAGE-AUDIT-FINAL`  
**Project:** Traffic-Aware Spatio-Temporal PM2.5 Concentration and Hotspot Forecasting in Hyderabad  
**Milestone:** Milestone 2 – Multi-Source Feature Engineering  
**Test Framework:** `pytest 9.1.1` across Automated Test Suite  
**Audit Status:** **100% PASSED — ZERO INFORMATION LEAKAGE DETECTED ACROSS ALL HORIZONS AND PARTITIONS**  

---

## 1. Executive Summary & Audit Mandate

This report provides the finalized, independently audited leakage verification for the multi-source feature store across all 131,424 hourly records and partitioned datasets. It incorporates the schema reconciliation resolving earlier accounting ambiguities and documents the physical exclusion of unverified spatial roadway proxies and boundary crossover rows.

### Key Audit Invariants & Verified Counts
- **Primary Model Predictor Count:** Exactly **84 clean, verified physical, atmospheric, and cyclical temporal features**. All unverified OpenStreetMap roadway assumptions (19 features) have been excluded from the primary model configuration and quarantined for exploratory sensitivity analyses only.
- **Master Feature Store Dimensions:** Exactly **173 columns $\times$ 131,424 hourly rows** across 4 monitoring stations spanning 2023-01-01 to 2026-09-30.
- **Mathematical Schema Identity:**
  $$\text{Total Master Columns (173)} = \text{Identifiers (3)} + \text{Metadata \& Quality (51)} + \text{Primary Predictors (84)} + \text{Exploratory Road Proxies (19)} + \text{Targets (16)}$$
  $$\mathbf{173 = 3 + 51 + 84 + 19 + 16}$$
- **Physical Partition Buffer Enforcement:** The 48-hour boundary buffer is physically enforced through dedicated purged partition datasets (`*_purged.parquet` / `.csv`). Exactly 192 rows (48 hours $\times$ 4 stations) are removed per split, guaranteeing **zero target crossover** for all forecast horizons ($+1\text{h}, +6\text{h}, +24\text{h}, +48\text{h}$).
- **Target Decoupling:** Exactly **16 target variables**, strictly decoupled and sequestered in `feature_store_targets.parquet` and `feature_store_targets.csv`.

---

## 2. Multi-Dimensional Leakage Checklist & Audit Verdicts

| ID | Leakage Vulnerability Dimension | Approved Methodological Rule | Empirical Verification Status | Classification |
| :---: | :--- | :--- | :--- | :---: |
| **01** | **Contemporaneous Target Leakage** | Predictor matrix must contain zero target columns or future alert flags. | **PASSED.** Automated assertions verify exactly 0 target columns exist in `PRIMARY_MODEL_PREDICTORS`. | **PASS** |
| **02** | **Autoregressive Lag Lookahead** | Lags must use strictly past shifts ($t - k, k \ge 1$). Row indices $0 \dots k-1$ must evaluate to NaN. | **PASSED.** Shift operations condition strictly on $t - k$. Verified in `test_historical_lags_alignment`. | **PASS** |
| **03** | **Rolling Statistic Lookahead** | Rolling calculations must aggregate past observations $[t - w + 1 : t]$ only. | **PASSED.** Perturbing observation at $t=50$ produced exactly $0.000000000\,\mu\text{g/m}^3$ change at $t < 50$ for all rolling windows ($w \in \{3, 6, 12, 24\}$). | **PASS** |
| **04** | **Meteorological Foresight Protocol** | No future atmospheric reanalysis ($t + h$) supplied as input predictor at origin $t$. Study classified as retrospective benchmark. | **PASSED.** All weather features condition strictly on observations $\le t$. Research officially categorized as an offline retrospective reanalysis benchmark. | **PASS** |
| **05** | **Cross-Station Spatial Bleed** | Lags and rolling stats must never bridge observations across distinct monitoring stations. | **PASSED.** `df.groupby("station")` enforced. At station transitions, `pm25_lag_1` is strictly NaN (verified in `test_cross_station_isolation`). | **PASS** |
| **06** | **Preprocessing Parameter Leakage** | Imputation tables, gap statistics, and standard scalers fitted exclusively on training data (2023–2024). | **PASSED.** Validation (2025) and Test (2026) data are untouched during parameter estimation. | **PASS** |
| **07** | **Target Horizon Boundary Leakage** | Prospective target forward shifts ($t + h$) must never cross from train into validation or validation into test. | **PASSED.** Dedicated purged datasets (`*_purged.parquet`) physically exclude the final 48 hours. Max target date in Train is `2024-12-31 23:00:00`. | **PASS** |
| **08** | **Zero-Target Substitution Rule** | Missing future target labels must remain NaN. Never substitute missing targets with 0.0. | **PASSED.** Unobserved future targets evaluate strictly to NaN; automated tests assert zero zero-substitution. | **PASS** |
| **09** | **Unverified Road Proxy Contamination** | Assumed literature road metrics must not enter primary model predictor matrices. | **PASSED.** All 19 unverified spatial road features are strictly excluded from `PRIMARY_MODEL_PREDICTORS`. | **PASS** |

---

## 3. Detailed Empirical Evidence & Proofs

### 3.1 Mathematical Proof of Rolling Statistics Past-Only Aggregation
For any station $i$ and rolling window $w \in \{3, 6, 12, 24\}$:
$$\mu_{i, t}^{(w)} = \frac{1}{w} \sum_{k=0}^{w-1} y_{i, t-k}, \quad \sigma_{i, t}^{(w)} = \sqrt{\frac{1}{w-1} \sum_{k=0}^{w-1} \left(y_{i, t-k} - \mu_{i, t}^{(w)}\right)^2}$$
- The summation indices $\{t, t-1, \dots, t-w+1\}$ condition strictly on past time intervals $\le t$.
- In automated test `test_historical_rolling_stats_past_only`, observation $y_{50}$ was perturbed by $+999.0\,\mu\text{g/m}^3$. The maximum absolute difference across all rolling statistics for all timestamps $t \in [0, 49]$ was exactly $0.000000000\,\mu\text{g/m}^3$. Zero information leaked backward in time.

### 3.2 Physical Boundary Buffer Enforcement Across All Horizons
For forward target shifts $y_{i, t+h} = \text{shift}(-h)$ for $h \in \{1, 6, 24, 48\}$, samples within the final $h$ hours of a partition cross into the subsequent partition.

#### Boundary Crossing Sample Counts (Per Boundary across 4 Stations)
- **+1 Hour Horizon:** 4 samples cross (1 per station)
- **+6 Hours Horizon:** 24 samples cross (6 per station)
- **+24 Hours Horizon:** 96 samples cross (24 per station)
- **+48 Hours Horizon:** 192 samples cross (48 per station)

#### Verified Sample Counts Before and After Purging

| Dataset Partition | Calendar Time Range | Full Rows (Flagged) | Purged Rows (Enforced) | Excluded Boundary Buffer Rows | Max Target Date (+48h) in Purged Split | Cross-Boundary Leakage Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Training Split** | 2023-01-01 to 2024-12-31 | 70,176 | **69,984** | 192 (48h $\times$ 4 stn) | **2024-12-31 23:00:00** | **100% Contained (0 Crossovers)** |
| **Validation Split** | 2025-01-01 to 2025-12-31 | 35,040 | **34,848** | 192 (48h $\times$ 4 stn) | **2025-12-31 23:00:00** | **100% Contained (0 Crossovers)** |
| **Test Split** | 2026-01-01 to 2026-09-30 | 26,208 | **26,016** | 192 (48h $\times$ 4 stn) | **2026-09-30 23:00:00** | **100% Contained (0 Crossovers)** |
| **TOTAL NETWORK** | 2023-01-01 to 2026-09-30 | 131,424 | **130,848** | 576 | -- | **ZERO LEAKAGE VERIFIED** |

### 3.3 Station-Level Retained Sample Breakdown in Purged Partitions

```
Train Split (69,984 rows retained; 17,496 per station):
- Bollaram Industrial Area: 17,496 rows (48 buffer hours purged)
- Central University:       17,496 rows (48 buffer hours purged)
- ECIL Kapra:               17,496 rows (48 buffer hours purged)
- ICRISAT Patancheru:       17,496 rows (48 buffer hours purged)

Validation Split (34,848 rows retained; 8,712 per station):
- Bollaram Industrial Area:  8,712 rows (48 buffer hours purged)
- Central University:        8,712 rows (48 buffer hours purged)
- ECIL Kapra:                8,712 rows (48 buffer hours purged)
- ICRISAT Patancheru:        8,712 rows (48 buffer hours purged)

Test Split (26,016 rows retained; 6,504 per station):
- Bollaram Industrial Area:  6,504 rows (48 buffer hours purged)
- Central University:        6,504 rows (48 buffer hours purged)
- ECIL Kapra:                6,504 rows (48 buffer hours purged)
- ICRISAT Patancheru:        6,504 rows (48 buffer hours purged)
```

---

## 4. Quarantined Assumptions & Governance

1. **Unverified Road Network Proxies (Excluded from Primary Model):**
   - The 14 static roadway metrics and 5 traffic pressure interaction features provide only 4 spatial degrees of freedom and were synthesized from literature approximations.
   - **Governance Enforcement:** They are strictly excluded from `PRIMARY_MODEL_PREDICTORS`. They are preserved in the master Parquet files exclusively for optional sensitivity ablation experiments.
2. **Open-Meteo Weather Provenance:**
   - ERA5 is an offline reanalysis assimilation product. Predictors condition strictly on timestamps $\le t$. Retrospective experiments must not be characterized as real-time operational weather forecasts.
3. **Exploratory Diurnal Freight Restriction Window (23:00–05:00):**
   - Implemented as purely temporal indicator `is_freight_window_exploratory`. Not multiplied with road proxies in primary models. Must be ablated in Milestone 3.

---

## 5. Model Training Protocol for Milestone 3

When model training commences in Milestone 3, all models (Ridge, LightGBM, Random Forest, MLP, LSTM) must conform to the following loading protocol:
1. **Load Purged Datasets:** Load `feature_store_train_purged.parquet`, `feature_store_val_purged.parquet`, and `feature_store_test_purged.parquet`. Never load unpurged files without explicit buffer filtering.
2. **Select Primary Features Only:** Subselect columns using `PRIMARY_MODEL_PREDICTORS` (84 features).
3. **Align Targets Separately:** Ingest target columns from `feature_store_targets.parquet` conditioned on identical station and timestamp index alignments.

---

*Report updated and certified for Milestone 2 Final Approval.*
