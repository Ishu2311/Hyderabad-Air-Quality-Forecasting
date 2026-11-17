# Milestone 2 — Final Scientific Corrections and Pre-Approval Audit

**Document Code:** `AUDIT-M2-FINAL-APPROVAL`  
**Project Title:** Traffic-Aware Spatio-Temporal PM2.5 Concentration and Hotspot Forecasting in Hyderabad Using Multi-Source Environmental Data and Explainable Machine Learning  
**Milestone:** Milestone 2 – Multi-Source Feature Engineering  
**Scope:** Final Audit of Road Assumption Exclusions, Schema Reconciliation Identity, Purged Partition Integrity, and Automated Test Reproduction  
**Audit Status:** **100% COMPLIANT — ALL DIRECTIVES REMEDIATED, EMPIRICALLY AUDITED, AND READY FOR MILESTONE 3 APPROVAL**  

---

## Executive Summary & Audit Classification Matrix

In response to the user review on Milestone 2, four critical scientific corrections were implemented and independently verified against the actual source code and generated datasets:

1. **Unverified Road Assumptions:** All 19 features derived from assumed roadway metrics (14 static spatial proxies + 5 traffic pressure interaction terms) have been **completely excluded from the primary model predictor configuration** (`PRIMARY_MODEL_PREDICTORS`). They are preserved strictly for an optional exploratory sensitivity ablation.
2. **Feature Store Column Reconciliation:** The previous reporting ambiguity ($153 + 14 + 16 = 183 \neq 173$) was diagnosed and mathematically resolved. The true physical schema of `feature_store_primary.parquet` contains exactly **173 columns**, satisfying the exact identity:
   $$\text{Total (173)} = \text{Identifiers (3)} + \text{Metadata \& Quality (51)} + \text{Primary Predictors (84)} + \text{Exploratory Road Proxies (19)} + \text{Targets (16)}$$
3. **Purged Partition Verification:** Every retained sample in `feature_store_train_purged.parquet` (69,984 rows), `feature_store_val_purged.parquet` (34,848 rows), and `feature_store_test_purged.parquet` (26,016 rows) was audited across all forecast horizons ($+1\text{h}, +6\text{h}, +24\text{h}, +48\text{h}$). Exactly 192 boundary buffer rows were excluded per partition, guaranteeing **zero target crossover** into subsequent splits.
4. **Automated Test Reproduction:** Four new regression-prevention unit tests were added to `test_feature_engineering.py`. The entire test suite was executed: **39/39 tests passed cleanly in 11.2s**.

### Comprehensive Audit Matrix

| Audit Item | Governing Scientific Rule | Empirical Finding / Evidence | Remediated State & Code Enforcement | Final Classification |
| :--- | :--- | :--- | :--- | :---: |
| **1. Road Proxy Exclusion** | Assumed road metrics must not enter primary model predictor configuration. | 14 static proxies (`osm_*`) and 5 interaction terms were previously mixed in candidate pool. | Excluded all 19 road-derived features from `PRIMARY_MODEL_PREDICTORS`. Quarantined in `EXPLORATORY_ROAD_PROXY_FEATURES`. | **PASS** |
| **2. Schema Reconciliation Identity** | Reconcile all 173 columns into exact, non-forced, mutually disjoint categories. | Identified that earlier code labeled 153 columns as predictors by subtracting only 4 keys and 16 targets. | Proved identity: $3 + 51 + 84 + 19 + 16 = 173$. Cataloged all 173 columns in `FEATURE_SCHEMA_RECONCILIATION.md`. | **PASS** |
| **3. Purged Partitions: Train Boundary** | No train sample targets may cross into validation (after 2024-12-31 23:00). | In unpurged data, 192 samples crossed. In purged data (69,984 rows), max target timestamp for +48h is `2024-12-31 23:00:00`. | Purged dataset physically excludes buffer. Zero training targets enter validation. | **PASS** |
| **4. Purged Partitions: Val Boundary** | No validation sample targets may cross into test (after 2025-12-31 23:00). | In unpurged data, 192 samples crossed. In purged data (34,848 rows), max target timestamp for +48h is `2025-12-31 23:00:00`. | Purged dataset physically excludes buffer. Zero validation targets enter test. | **PASS** |
| **5. Purged Partitions: Test Boundary** | Test sample targets must remain observed within available timeline ($\le$ 2026-09-30 23:00). | In purged test data (26,016 rows), max target timestamp for +48h is `2026-09-30 23:00:00`. | Zero test targets exceed the dataset boundary. | **PASS** |
| **6. Predictor Matrix Hygiene** | Primary predictor matrix must contain zero targets, future variables, quality flags, or identifiers. | Verified in code and automated tests. Exactly 84 clean physical, atmospheric, and cyclical features. | `PRIMARY_MODEL_PREDICTORS` has 0 target, 0 raw, 0 mask, 0 gap, and 0 identifier columns. | **PASS** |
| **7. Test Suite Reproduction** | Entire project test suite must pass with new verification tests added. | 39 automated tests executed from project root via `pytest`. | **39 passed in 11.2s** (100% pass rate). | **PASS** |

---

## 1. Removal of Unverified Road Assumptions from Primary Model Inputs

### 1.1 Identification of Road-Derived Features
Because no raw OpenStreetMap vector files (`.osm`, `.pbf`, `.geojson`) were present and Overpass API queries failed with HTTP 406 / timeouts, the road metrics in `config.py` represent **assumed literature-derived spatial proxies** with 4 degrees of freedom across the monitoring network.

An exhaustive audit identified **19 features** that incorporate or derive from these assumed road proxies:

#### A. Static Spatial Roadway Proxies (14 features)
1. `osm_primary_length_1km` (Assumed primary road length in 1km buffer)
2. `osm_secondary_length_1km` (Assumed secondary road length in 1km buffer)
3. `osm_total_length_1km` (Sum of assumed primary + secondary length in 1km buffer)
4. `osm_primary_length_3km` (Assumed primary road length in 3km buffer)
5. `osm_secondary_length_3km` (Assumed secondary road length in 3km buffer)
6. `osm_total_length_3km` (Sum of assumed primary + secondary length in 3km buffer)
7. `osm_road_density_1km` (Assumed total road density in 1km buffer: $\text{length} / \pi$)
8. `osm_road_density_3km` (Assumed total road density in 3km buffer: $\text{length} / 9\pi$)
9. `osm_primary_density_1km` (Assumed primary road density in 1km buffer)
10. `osm_primary_density_3km` (Assumed primary road density in 3km buffer)
11. `osm_primary_concentration_1km` (Ratio of assumed primary to total road length in 1km buffer)
12. `osm_primary_concentration_3km` (Ratio of assumed primary to total road length in 3km buffer)
13. `osm_distance_to_highway_km` (Assumed Euclidean distance to nearest national highway corridor)
14. `osm_intersection_density_km2` (Assumed multi-leg intersection density)

#### B. Spatio-Temporal Traffic Pressure Interactions (5 features)
15. `traffic_pressure_total_1km` ($= \text{osm\_road\_density\_1km} \times \text{is\_traffic\_peak}$)
16. `traffic_pressure_weekday_peak` ($= \text{osm\_road\_density\_1km} \times \text{is\_weekday\_peak}$)
17. `traffic_pressure_primary_1km` ($= \text{osm\_primary\_length\_1km} \times \text{is\_traffic\_peak}$)
18. `freight_pressure_proxy` ($= \text{osm\_road\_density\_3km} \times \text{is\_freight\_window\_exploratory}$)
19. `highway_rush_pressure` ($= \text{is\_traffic\_peak} / (\text{osm\_distance\_to\_highway\_km} + 0.1)$)

### 1.2 Governance & Code Isolation Enforced
- **Primary Model Configuration:** In [configs/config.py](file:///c:/Users/harsha%20madala/Documents/Hyderabad_Air_Research/new_research_project/implementation/configs/config.py#L225-L270), `PRIMARY_MODEL_PREDICTORS` has been explicitly defined with **84 clean features**, strictly excluding all 19 road-derived features.
- **Exploratory Sensitivity Group:** All 19 features are quarantined under `EXPLORATORY_ROAD_PROXY_FEATURES` in `config.py`. They are preserved in the master Parquet feature store for optional exploratory sensitivity ablation experiments only.
- **No Claims of Measured Traffic:** All reports, docstrings, and feature dictionaries explicitly declare that these features are **unverified literature spatial proxies**, not empirical OSM geometric measurements or live vehicular traffic volume.

---

## 2. Exact Reconciliation of Feature-Store Column Counts

### 2.1 Diagnosis of the Discrepancy
Earlier documentation cited "173 columns, 153 predictors, 14 administrative columns, and 16 targets" ($153 + 14 + 16 = 183 \neq 173$). 

Inspection of the codebase revealed that `feature_cols` was previously calculated simply as:
$$\text{Total Columns (173)} - \text{Targets (16)} - 4\text{ Excluded Keys} (\text{station}, \text{datetime}, \text{partition}, \text{is\_boundary\_buffer}) = 153$$
This calculation inadvertently conflated 40 data-quality flags, 10 discrete calendar and spatial metadata fields, and 19 unverified road proxies into a single "predictor" bucket.

### 2.2 Mathematical Schema Reconciliation Identity
Every column in `feature_store_primary.parquet` has been accounted for exactly once:

$$\mathbf{Total\;Columns\;(173) = Identifiers\;(3) + Metadata\;\&\;Quality\;(51) + Candidate\;Predictors\;(103) + Targets\;(16)}$$
$$\mathbf{173 = 3 + 51 + (84 + 19) + 16}$$

#### Operational 5-Category Breakdown

| Category | Column Count | Allowed in Primary Model? | Description & Contents |
| :--- | :---: | :---: | :--- |
| **1. Identifiers** | **3** | **NO** | Administrative non-numeric keys: `station`, `station_id`, `datetime`. |
| **2. Metadata & Quality Fields** | **51** | **NO** | Geodetic coords (7), discrete calendar (4), gap metadata (16), raw telemetry (8), missing masks (8), imputation flags (8). |
| **3. Primary Model Predictors** | **84** | **YES** | **Verified clean features:** Ambient pollutants (8), AR lags (6), rolling stats (8), dynamic trends (5), co-pollutant chemical ratios (8), cyclical calendar (10), weather & atmospheric stability (32), diurnal traffic windows (7). |
| **4. Exploratory Road Proxies** | **19** | **NO** | **Unverified spatial proxies:** Static road metrics (14) + traffic pressure interactions (5). Quarantined for sensitivity experiments. |
| **5. Prospective Targets** | **16** | **NO** | Ground-truth labels: Single-horizon continuous (+1, +6, +24, +48h), 24h rolling mean, count, and hotspot classification alerts. |
| **TOTAL** | **173** | **84 YES / 89 NO** | **Exact match with physical Parquet schema.** Zero missing, zero orphaned, zero forced counts. |

---

## 3. Independent Verification of Purged Partitions

### 3.1 Crossover Sample Accounting per Forecast Horizon
In multi-horizon forecasting, target observation $y_{i, t+h}$ crosses partition boundaries if $t + h > \text{partition\_end}$:

| Forecast Horizon | Train Boundary Crossovers (into 2025) | Val Boundary Crossovers (into 2026) | Test Horizon Overhang (beyond dataset) | Crossover Formula |
| :---: | :---: | :---: | :---: | :--- |
| **+1 Hour** | **4 samples** (1 / station) | **4 samples** (1 / station) | 0 samples | $t + 1\text{h} > \text{end}$ |
| **+6 Hours** | **24 samples** (6 / station) | **24 samples** (6 / station) | 0 samples | $t + 6\text{h} > \text{end}$ |
| **+24 Hours** | **96 samples** (24 / station) | **96 samples** (24 / station) | 0 samples | $t + 24\text{h} > \text{end}$ |
| **+48 Hours** | **192 samples** (48 / station) | **192 samples** (48 / station) | 0 samples | $t + 48\text{h} > \text{end}$ |

### 3.2 Purged Partitions Empirical Verification
By physically purging the final 48 hours (192 rows) from each partition, all prospective target timestamps are guaranteed to remain strictly within the partition bounds:

| Evaluation Partition | Calendar Bounds | Unpurged Rows (Flagged) | Purged Rows (Enforced) | Excluded Buffer Rows | Max Timestamp ($t$) in Purged Split | Max Target Timestamp ($t+48\text{h}$) | Cross-Boundary Leakage Verdict |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Training Split** | 2023-01-01 to 2024-12-31 | 70,176 | **69,984** | 192 (48h $\times$ 4 stn) | 2024-12-29 23:00:00 | **2024-12-31 23:00:00** | **0 samples cross into Val (PASS)** |
| **Validation Split** | 2025-01-01 to 2025-12-31 | 35,040 | **34,848** | 192 (48h $\times$ 4 stn) | 2025-12-29 23:00:00 | **2025-12-31 23:00:00** | **0 samples cross into Test (PASS)** |
| **Test Split** | 2026-01-01 to 2026-09-30 | 26,208 | **26,016** | 192 (48h $\times$ 4 stn) | 2026-09-28 23:00:00 | **2026-09-30 23:00:00** | **0 samples exceed dataset (PASS)** |
| **TOTAL NETWORK** | 2023-01-01 to 2026-09-30 | 131,424 | **130,848** | 576 | -- | -- | **ZERO LEAKAGE CONFIRMED** |

### 3.3 Station-Level Retained Sample and Valid Target Breakdown

#### Training Partition (69,984 retained rows; 17,496 rows / station)
- **Bollaram Industrial Area:** 17,496 rows. Valid observed targets: +1h: 14,233 (81.35%), +6h: 14,228 (81.32%), +24h: 14,230 (81.33%), +48h: 14,225 (81.30%). Missing: 3,271.
- **Central University:** 17,496 rows. Valid observed targets: +1h: 15,915 (90.96%), +6h: 15,910 (90.94%), +24h: 15,903 (90.90%), +48h: 15,903 (90.90%). Missing: 1,593.
- **ECIL Kapra:** 17,496 rows. Valid observed targets: +1h: 14,777 (84.46%), +6h: 14,777 (84.46%), +24h: 14,777 (84.46%), +48h: 14,761 (84.37%). Missing: 2,735.
- **ICRISAT Patancheru:** 17,496 rows. Valid observed targets: +1h: 15,938 (91.10%), +6h: 15,938 (91.10%), +24h: 15,938 (91.10%), +48h: 15,939 (91.10%). Missing: 1,557.

#### Validation Partition (34,848 retained rows; 8,712 rows / station)
- **Bollaram Industrial Area:** 8,712 rows. Valid observed targets: +1h: 7,460 (85.63%), +6h: 7,455 (85.57%), +24h: 7,448 (85.49%), +48h: 7,431 (85.30%). Missing: 1,281.
- **Central University:** 8,712 rows. Valid observed targets: +1h: 7,343 (84.29%), +6h: 7,343 (84.29%), +24h: 7,341 (84.26%), +48h: 7,339 (84.24%). Missing: 1,373.
- **ECIL Kapra:** 8,712 rows. Valid observed targets: +1h: 7,780 (89.30%), +6h: 7,780 (89.30%), +24h: 7,780 (89.30%), +48h: 7,778 (89.28%). Missing: 934.
- **ICRISAT Patancheru:** 8,712 rows. Valid observed targets: +1h: 8,405 (96.48%), +6h: 8,405 (96.48%), +24h: 8,405 (96.48%), +48h: 8,405 (96.48%). Missing: 307.

#### Test Partition (26,016 retained rows; 6,504 rows / station)
- **Bollaram Industrial Area:** 6,504 rows. Valid observed targets: +1h: 1,875 (28.83%), +6h: 1,875 (28.83%), +24h: 1,889 (29.04%), +48h: 1,908 (29.34%). Missing: 4,596 (known 2026 sensor outage preserved as NaN).
- **Central University:** 6,504 rows. Valid observed targets: +1h: 5,743 (88.30%), +6h: 5,748 (88.38%), +24h: 5,750 (88.41%), +48h: 5,750 (88.41%). Missing: 754.
- **ECIL Kapra:** 6,504 rows. Valid observed targets: +1h: 5,849 (89.93%), +6h: 5,849 (89.93%), +24h: 5,854 (90.01%), +48h: 5,855 (90.02%). Missing: 649.
- **ICRISAT Patancheru:** 6,504 rows. Valid observed targets: +1h: 5,606 (86.19%), +6h: 5,606 (86.19%), +24h: 5,606 (86.19%), +48h: 5,606 (86.19%). Missing: 898.

*No missing target labels have been imputed or replaced with 0.0; unobserved targets evaluate strictly to NaN in full compliance with research specifications.*

---

## 4. Final Automated Test Suite Execution

The complete automated test suite was executed from the repository root:

```powershell
python -m pytest new_research_project/implementation/tests -v
```

### Test Session Output (Excerpt)
```
============================= test session starts =============================
platform win32 -- Python 3.13.11, pytest-9.1.1, pluggy-1.5.0
rootdir: C:\Users\harsha madala\Documents\Hyderabad_Air_Research
collected 39 items

test_feature_engineering.py::test_historical_lags_alignment PASSED              [  2%]
test_feature_engineering.py::test_historical_rolling_stats_past_only PASSED      [  5%]
test_feature_engineering.py::test_historical_trends_and_acceleration PASSED      [  7%]
test_feature_engineering.py::test_weather_wind_vector_identity PASSED            [ 10%]
test_feature_engineering.py::test_weather_psychrometric_consistency PASSED       [ 12%]
test_feature_engineering.py::test_atmospheric_stagnation_logic PASSED            [ 15%]
test_feature_engineering.py::test_osm_road_metrics_consistency PASSED             [ 17%]
test_feature_engineering.py::test_congestion_windows_logic PASSED                 [ 20%]
test_feature_engineering.py::test_feature_store_target_decoupling PASSED         [ 23%]
test_feature_engineering.py::test_cross_station_isolation PASSED                 [ 25%]
test_feature_engineering.py::test_osm_data_provenance_explicit_assumed_label PASSED [ 28%]
test_feature_engineering.py::test_temporal_boundary_purging_zero_target_crossover PASSED [ 30%]
test_feature_engineering.py::test_forecast_time_feature_availability_point_in_time PASSED [ 33%]
test_feature_engineering.py::test_feature_dictionary_exhaustive_173_reconciliation PASSED [ 35%]
test_feature_engineering.py::test_assumed_road_proxies_excluded_from_primary_predictors PASSED [ 38%]
test_feature_engineering.py::test_feature_schema_exact_identity_and_disjointness PASSED [ 41%]
test_feature_engineering.py::test_purged_partitions_zero_boundary_crossover_all_horizons PASSED [ 43%]
test_feature_engineering.py::test_primary_model_input_hygiene PASSED             [ 46%]
test_ingestion.py::test_cpcb_duplicate_timestamps PASSED                         [ 48%]
test_ingestion.py::test_weather_duplicate_timestamps PASSED                      [ 51%]
test_ingestion.py::test_negative_readings_cleaning PASSED                         [ 53%]
test_ingestion.py::test_station_identifier_standardization PASSED                 [ 56%]
test_ingestion.py::test_hourly_continuity PASSED                                  [ 58%]
test_ingestion.py::test_exact_dimensions_and_valid_counts PASSED                  [ 61%]
test_ingestion.py::test_weather_completeness PASSED                              [ 64%]
test_leakage_safety.py::test_scaler_zero_leakage_from_future PASSED               [ 66%]
test_leakage_safety.py::test_diurnal_imputer_zero_future_leakage PASSED          [ 69%]
test_missing_values.py::test_gap_analyzer_categorization PASSED                   [ 71%]
test_missing_values.py::test_missing_mask_preservation PASSED                    [ 74%]
test_missing_values.py::test_short_gap_ffill_no_lookahead PASSED                 [ 76%]
test_missing_values.py::test_extended_outages_remain_nan_in_primary PASSED      [ 79%]
test_missing_values.py::test_zero_target_substitution_absence PASSED             [ 82%]
test_partitioning.py::test_partition_dimensions_and_chronology PASSED            [ 84%]
test_partitioning.py::test_boundary_buffer_leakage_guard PASSED                   [ 87%]
test_reproducibility.py::test_pipeline_reproducibility PASSED                     [ 89%]
test_target_generation.py::test_multi_horizon_shift_alignment PASSED             [ 92%]
test_target_generation.py::test_missing_target_preservation_nan PASSED           [ 94%]
test_target_generation.py::test_forward_24h_mean_completeness_and_denominator PASSED [ 97%]
test_target_generation.py::test_threshold_boundaries PASSED                     [100%]

============================= 39 passed in 11.24s =============================
```

---

## 5. Deliverables Generated & Updated

All deliverables are generated and synchronized:
1. [MILESTONE_2_FINAL_APPROVAL_AUDIT.md](file:///c:/Users/harsha%20madala/Documents/Hyderabad_Air_Research/new_research_project/reports/MILESTONE_2_FINAL_APPROVAL_AUDIT.md): This overarching final pre-approval audit synthesis.
2. [FEATURE_SCHEMA_RECONCILIATION.md](file:///c:/Users/harsha%20madala/Documents/Hyderabad_Air_Research/new_research_project/reports/FEATURE_SCHEMA_RECONCILIATION.md): Complete accounting table of all 173 columns with individual categories, permissions, sources, and transformations.
3. [LEAKAGE_AUDIT_REPORT.md](file:///c:/Users/harsha%20madala/Documents/Hyderabad_Air_Research/new_research_project/reports/LEAKAGE_AUDIT_REPORT.md): Updated audit report with corrected feature counts, road proxy exclusions, and purged boundary proofs.
4. [FEATURE_DICTIONARY.md](file:///c:/Users/harsha%20madala/Documents/Hyderabad_Air_Research/new_research_project/reports/FEATURE_DICTIONARY.md) and [feature_dictionary.json](file:///c:/Users/harsha%20madala/Documents/Hyderabad_Air_Research/new_research_project/data/processed/feature_dictionary.json): Complete 173-entry data dictionary with `role`, `category`, and `allowed_in_primary_model`.
5. [feature_store_summary.json](file:///c:/Users/harsha%20madala/Documents/Hyderabad_Air_Research/new_research_project/data/processed/feature_store_summary.json): Synchronized metadata summary validating the exact $3 + 51 + 84 + 19 + 16 = 173$ identity.
6. [test_feature_engineering.py](file:///c:/Users/harsha%20madala/Documents/Hyderabad_Air_Research/new_research_project/implementation/tests/test_feature_engineering.py): Automated tests covering all 4 final verification requirements (39 passing tests total).

---

## 6. Compliance with Stop Condition

In strict accordance with the stop condition:
- **No machine-learning models have been trained.**
- **No spatial interpolation (IDW / Ordinary Kriging) has been implemented.**
- **No SHAP explainability analyses have been conducted.**
- **No dashboard or web application components have been developed.**

All final corrections for Milestone 2 have been completed, audited, and verified. The project is at a clean stop condition awaiting your explicit approval before beginning **Milestone 3 (Baseline and Spatio-Temporal Model Training)**.
