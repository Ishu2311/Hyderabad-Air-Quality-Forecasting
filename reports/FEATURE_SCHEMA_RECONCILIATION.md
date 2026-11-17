# Feature Store Schema Reconciliation and Column Accounting Report

**Document Code:** `SPEC-M2-SCHEMA-RECONCILIATION`  
**Project Title:** Traffic-Aware Spatio-Temporal PM2.5 Concentration and Hotspot Forecasting in Hyderabad  
**Milestone:** Milestone 2 – Multi-Source Feature Engineering  
**Audit Mandate:** Complete and Exact Accounting of All 173 Columns in `feature_store_primary.parquet`  
**Audit Status:** **100% RECONCILED — EXACT MATHEMATICAL IDENTITY VERIFIED**  

---

## 1. Executive Summary & Root-Cause Analysis

The Milestone 2 independent verification audit identified an accounting discrepancy in previous completion materials: earlier summaries vaguely cited *"153 predictors, 14 administrative columns, and 16 targets"* against a 173-column Parquet dataset. Mathematically, $153 + 14 + 16 = 183 
eq 173$.

### Root Cause of the Discrepancy
Direct inspection of `feature_store_builder.py` confirms that earlier code computed `predictor_feature_count` by taking `total_columns (173)` and subtracting strictly **4 columns** (`station`, `datetime`, `partition`, `is_boundary_buffer`) along with the **16 target columns**:
$$173 - 16 - 4 = 153$$
That calculation erroneously labeled all remaining 153 columns as "predictors", failing to recognize that:
1. **40 columns** are raw sensor telemetry, missing masks, gap lengths, and imputation flags.
2. **10 columns** are discrete calendar indices and spatial land-use metadata (`latitude`, `longitude`, `hour`, `month`, etc.).
3. **19 columns** are unverified OpenStreetMap road proxies and traffic pressure interaction terms derived from assumed literature approximations.
4. **Only 84 columns** are actual, verified physical ambient, autoregressive, rolling, meteorological, and temporal predictors.

---

## 2. Mathematical Identity and Schema Breakdown

Every single column in `feature_store_primary.parquet` has been individually cataloged into mutually exclusive and collectively exhaustive categories.

### 2.1 The Four-Bucket Identity
$$\text{Total Columns} = \text{Identifiers} + \text{Metadata \& Quality Fields} + \text{Candidate Predictors} + \text{Targets}$$
$$\mathbf{173 = 3 + 51 + 103 + 16}$$

### 2.2 The Five-Category Operational Model Configuration Breakdown
Following user instructions, the 103 candidate predictors are strictly bifurcated into primary verified features and excluded exploratory proxies:

| Category ID | Category Name | Exact Column Count | Allowed in Primary Model Predictor Matrix? | Operational Definition & Governance |
| :---: | :--- | :---: | :---: | :--- |
| **1** | **Identifiers** | **3** | **NO** | Administrative keys (`station`, `station_id`, `datetime`). Strictly excluded from numeric training tensors. |
| **2** | **Metadata & Quality Fields** | **51** | **NO** | Geodetic coords (7), discrete calendar (4), gap metadata (16), raw telemetry (8), missing masks (8), imputation flags (8). |
| **3** | **Primary Model Predictors** | **84** | **YES** | **Clean physical, atmospheric, and cyclical predictors.** Prospective ambient gases (8), AR lags (6), rolling stats (8), trends (5), co-pollutant ratios (8), cyclical temporal (10), weather & atmospheric stability (32), diurnal traffic windows (7). |
| **4** | **Exploratory Road Proxies** | **19** | **NO (Exploratory Sensitivity Only)** | **Unverified literature spatial proxies.** 14 static roadway metrics + 5 traffic pressure interactions. Excluded from primary models; preserved for optional sensitivity ablation. |
| **5** | **Prospective Targets** | **16** | **NO (Evaluation Labels Only)** | Continuous regression targets (+1h, +6h, +24h, +48h, 24h rolling mean, count) and binary hotspot classification targets. |
| **TOTAL** | **Master Parquet Columns** | **173** | **84 YES / 89 NO** | **Sum: $3 + 51 + 84 + 19 + 16 = 173$ (100% Exact Identity)** |

---

## 3. Predictor Matrix Hygiene & Non-Leakage Guarantees

When model training commences in Milestone 3, model pipelines will ingest the **84 Primary Model Predictors** from the purged partition files:
1. **Zero Target Leakage:** Exactly 0 target variables (`target_*`) exist in `PRIMARY_MODEL_PREDICTORS` (verified by automated assertion).
2. **Zero Future Lookahead:** All rolling features $[t-w+1 : t]$ and historical lags condition strictly on past observations $\le t$. Perturbing future timestamps produces zero variation in historical features.
3. **Zero Quality Mask Exposure:** All 40 missing masks, gap lengths, and imputation flags are strictly quarantined in metadata fields.
4. **Zero Unverified Road Contamination:** All 19 unverified road proxy features are barred from the primary feature configuration.
5. **Zero Identifier Cross-Talk:** Administrative string columns (`station`, `station_id`, `datetime`) are excluded from model tensors.

---

## 4. Complete Accounting Table of All 173 Columns

| # | Column Name | Data Type | Category | Allowed in Primary Predictor Matrix? | Source | Transformation & Operational Definition |
| :---: | :--- | :---: | :---: | :---: | :--- | :--- |
| 1 | `datetime` | `datetime64[ns]` | identifier | **NO (Administrative Key)** | CPCB & Open-Meteo Hourly Timeline | Continuous ISO 8601 hourly timestamp in Indian Standard Time (UTC+5:30) |
| 2 | `PM2.5` | `float64` | predictor | **YES (Primary Model Predictor)** | CPCB In-Situ CAAQMS Telemetry | Preprocessed ambient concentration at prediction origin t (short gaps filled) |
| 3 | `PM10` | `float64` | predictor | **YES (Primary Model Predictor)** | CPCB In-Situ CAAQMS Telemetry | Preprocessed ambient concentration at prediction origin t (short gaps filled) |
| 4 | `NO` | `float64` | predictor | **YES (Primary Model Predictor)** | CPCB In-Situ CAAQMS Telemetry | Preprocessed ambient concentration at prediction origin t (short gaps filled) |
| 5 | `NO2` | `float64` | predictor | **YES (Primary Model Predictor)** | CPCB In-Situ CAAQMS Telemetry | Preprocessed ambient concentration at prediction origin t (short gaps filled) |
| 6 | `NH3` | `float64` | predictor | **YES (Primary Model Predictor)** | CPCB In-Situ CAAQMS Telemetry | Preprocessed ambient concentration at prediction origin t (short gaps filled) |
| 7 | `SO2` | `float64` | predictor | **YES (Primary Model Predictor)** | CPCB In-Situ CAAQMS Telemetry | Preprocessed ambient concentration at prediction origin t (short gaps filled) |
| 8 | `CO` | `float64` | predictor | **YES (Primary Model Predictor)** | CPCB In-Situ CAAQMS Telemetry | Preprocessed ambient concentration at prediction origin t (short gaps filled) |
| 9 | `Ozone` | `float64` | predictor | **YES (Primary Model Predictor)** | CPCB In-Situ CAAQMS Telemetry | Preprocessed ambient concentration at prediction origin t (short gaps filled) |
| 10 | `station` | `object` | identifier | **NO (Administrative Key)** | CPCB CAAQMS Station Registry | Official station name string (Bollaram, Central Univ, ECIL, ICRISAT) |
| 11 | `station_id` | `object` | identifier | **NO (Administrative Key)** | CPCB Station Identifier Standard | Unique standardized station abbreviation code |
| 12 | `temperature_2m` | `float64` | predictor | **YES (Primary Model Predictor)** | Open-Meteo ERA5 Reanalysis (Retrospective Benchmark) | Atmospheric thermodynamic/vector parameter conditioned on observations <= t |
| 13 | `relative_humidity_2m` | `int64` | predictor | **YES (Primary Model Predictor)** | Open-Meteo ERA5 Reanalysis (Retrospective Benchmark) | Atmospheric thermodynamic/vector parameter conditioned on observations <= t |
| 14 | `wind_speed_10m` | `float64` | predictor | **YES (Primary Model Predictor)** | Open-Meteo ERA5 Reanalysis (Retrospective Benchmark) | Atmospheric thermodynamic/vector parameter conditioned on observations <= t |
| 15 | `wind_direction_10m` | `int64` | predictor | **YES (Primary Model Predictor)** | Open-Meteo ERA5 Reanalysis (Retrospective Benchmark) | Atmospheric thermodynamic/vector parameter conditioned on observations <= t |
| 16 | `precipitation` | `float64` | predictor | **YES (Primary Model Predictor)** | Open-Meteo ERA5 Reanalysis (Retrospective Benchmark) | Atmospheric thermodynamic/vector parameter conditioned on observations <= t |
| 17 | `surface_pressure` | `float64` | predictor | **YES (Primary Model Predictor)** | Open-Meteo ERA5 Reanalysis (Retrospective Benchmark) | Atmospheric thermodynamic/vector parameter conditioned on observations <= t |
| 18 | `latitude` | `float64` | metadata | **NO (Spatial Land-Use Metadata)** | CPCB Station Registry / Land-Use Survey | Static geodetic coordinate or urban typology classification |
| 19 | `longitude` | `float64` | metadata | **NO (Spatial Land-Use Metadata)** | CPCB Station Registry / Land-Use Survey | Static geodetic coordinate or urban typology classification |
| 20 | `easting_utm44n` | `float64` | metadata | **NO (Spatial Land-Use Metadata)** | CPCB Station Registry / Land-Use Survey | Static geodetic coordinate or urban typology classification |
| 21 | `northing_utm44n` | `float64` | metadata | **NO (Spatial Land-Use Metadata)** | CPCB Station Registry / Land-Use Survey | Static geodetic coordinate or urban typology classification |
| 22 | `typology` | `object` | metadata | **NO (Spatial Land-Use Metadata)** | CPCB Station Registry / Land-Use Survey | Static geodetic coordinate or urban typology classification |
| 23 | `PM2.5_gap_length` | `float64` | quality flag | **NO (Missingness Tracking Flag)** | Preprocessing Missingness Module | Consecutive missing-hour outage run-length and gap severity category |
| 24 | `PM2.5_gap_category` | `object` | quality flag | **NO (Missingness Tracking Flag)** | Preprocessing Missingness Module | Consecutive missing-hour outage run-length and gap severity category |
| 25 | `PM10_gap_length` | `float64` | quality flag | **NO (Missingness Tracking Flag)** | Preprocessing Missingness Module | Consecutive missing-hour outage run-length and gap severity category |
| 26 | `PM10_gap_category` | `object` | quality flag | **NO (Missingness Tracking Flag)** | Preprocessing Missingness Module | Consecutive missing-hour outage run-length and gap severity category |
| 27 | `NO_gap_length` | `float64` | quality flag | **NO (Missingness Tracking Flag)** | Preprocessing Missingness Module | Consecutive missing-hour outage run-length and gap severity category |
| 28 | `NO_gap_category` | `object` | quality flag | **NO (Missingness Tracking Flag)** | Preprocessing Missingness Module | Consecutive missing-hour outage run-length and gap severity category |
| 29 | `NO2_gap_length` | `float64` | quality flag | **NO (Missingness Tracking Flag)** | Preprocessing Missingness Module | Consecutive missing-hour outage run-length and gap severity category |
| 30 | `NO2_gap_category` | `object` | quality flag | **NO (Missingness Tracking Flag)** | Preprocessing Missingness Module | Consecutive missing-hour outage run-length and gap severity category |
| 31 | `NH3_gap_length` | `float64` | quality flag | **NO (Missingness Tracking Flag)** | Preprocessing Missingness Module | Consecutive missing-hour outage run-length and gap severity category |
| 32 | `NH3_gap_category` | `object` | quality flag | **NO (Missingness Tracking Flag)** | Preprocessing Missingness Module | Consecutive missing-hour outage run-length and gap severity category |
| 33 | `SO2_gap_length` | `float64` | quality flag | **NO (Missingness Tracking Flag)** | Preprocessing Missingness Module | Consecutive missing-hour outage run-length and gap severity category |
| 34 | `SO2_gap_category` | `object` | quality flag | **NO (Missingness Tracking Flag)** | Preprocessing Missingness Module | Consecutive missing-hour outage run-length and gap severity category |
| 35 | `CO_gap_length` | `float64` | quality flag | **NO (Missingness Tracking Flag)** | Preprocessing Missingness Module | Consecutive missing-hour outage run-length and gap severity category |
| 36 | `CO_gap_category` | `object` | quality flag | **NO (Missingness Tracking Flag)** | Preprocessing Missingness Module | Consecutive missing-hour outage run-length and gap severity category |
| 37 | `Ozone_gap_length` | `float64` | quality flag | **NO (Missingness Tracking Flag)** | Preprocessing Missingness Module | Consecutive missing-hour outage run-length and gap severity category |
| 38 | `Ozone_gap_category` | `object` | quality flag | **NO (Missingness Tracking Flag)** | Preprocessing Missingness Module | Consecutive missing-hour outage run-length and gap severity category |
| 39 | `PM2.5_raw` | `float64` | quality flag | **NO (Unimputed Telemetry)** | CPCB CAAQMS Physical Sensor | Raw uncleaned sensor observation prior to imputation |
| 40 | `PM2.5_missing_mask` | `float64` | quality flag | **NO (Sensor Mask Flag)** | Ingestion Quality Filter | Original missingness mask (1.0 = missing/unphysical, 0.0 = valid) |
| 41 | `PM2.5_imputed_flag` | `float64` | quality flag | **NO (Imputation Indicator)** | Preprocessing Imputation Engine | Flag indicating observation was filled via approved short-gap interpolation |
| 42 | `PM10_raw` | `float64` | quality flag | **NO (Unimputed Telemetry)** | CPCB CAAQMS Physical Sensor | Raw uncleaned sensor observation prior to imputation |
| 43 | `PM10_missing_mask` | `float64` | quality flag | **NO (Sensor Mask Flag)** | Ingestion Quality Filter | Original missingness mask (1.0 = missing/unphysical, 0.0 = valid) |
| 44 | `PM10_imputed_flag` | `float64` | quality flag | **NO (Imputation Indicator)** | Preprocessing Imputation Engine | Flag indicating observation was filled via approved short-gap interpolation |
| 45 | `NO_raw` | `float64` | quality flag | **NO (Unimputed Telemetry)** | CPCB CAAQMS Physical Sensor | Raw uncleaned sensor observation prior to imputation |
| 46 | `NO_missing_mask` | `float64` | quality flag | **NO (Sensor Mask Flag)** | Ingestion Quality Filter | Original missingness mask (1.0 = missing/unphysical, 0.0 = valid) |
| 47 | `NO_imputed_flag` | `float64` | quality flag | **NO (Imputation Indicator)** | Preprocessing Imputation Engine | Flag indicating observation was filled via approved short-gap interpolation |
| 48 | `NO2_raw` | `float64` | quality flag | **NO (Unimputed Telemetry)** | CPCB CAAQMS Physical Sensor | Raw uncleaned sensor observation prior to imputation |
| 49 | `NO2_missing_mask` | `float64` | quality flag | **NO (Sensor Mask Flag)** | Ingestion Quality Filter | Original missingness mask (1.0 = missing/unphysical, 0.0 = valid) |
| 50 | `NO2_imputed_flag` | `float64` | quality flag | **NO (Imputation Indicator)** | Preprocessing Imputation Engine | Flag indicating observation was filled via approved short-gap interpolation |
| 51 | `NH3_raw` | `float64` | quality flag | **NO (Unimputed Telemetry)** | CPCB CAAQMS Physical Sensor | Raw uncleaned sensor observation prior to imputation |
| 52 | `NH3_missing_mask` | `float64` | quality flag | **NO (Sensor Mask Flag)** | Ingestion Quality Filter | Original missingness mask (1.0 = missing/unphysical, 0.0 = valid) |
| 53 | `NH3_imputed_flag` | `float64` | quality flag | **NO (Imputation Indicator)** | Preprocessing Imputation Engine | Flag indicating observation was filled via approved short-gap interpolation |
| 54 | `SO2_raw` | `float64` | quality flag | **NO (Unimputed Telemetry)** | CPCB CAAQMS Physical Sensor | Raw uncleaned sensor observation prior to imputation |
| 55 | `SO2_missing_mask` | `float64` | quality flag | **NO (Sensor Mask Flag)** | Ingestion Quality Filter | Original missingness mask (1.0 = missing/unphysical, 0.0 = valid) |
| 56 | `SO2_imputed_flag` | `float64` | quality flag | **NO (Imputation Indicator)** | Preprocessing Imputation Engine | Flag indicating observation was filled via approved short-gap interpolation |
| 57 | `CO_raw` | `float64` | quality flag | **NO (Unimputed Telemetry)** | CPCB CAAQMS Physical Sensor | Raw uncleaned sensor observation prior to imputation |
| 58 | `CO_missing_mask` | `float64` | quality flag | **NO (Sensor Mask Flag)** | Ingestion Quality Filter | Original missingness mask (1.0 = missing/unphysical, 0.0 = valid) |
| 59 | `CO_imputed_flag` | `float64` | quality flag | **NO (Imputation Indicator)** | Preprocessing Imputation Engine | Flag indicating observation was filled via approved short-gap interpolation |
| 60 | `Ozone_raw` | `float64` | quality flag | **NO (Unimputed Telemetry)** | CPCB CAAQMS Physical Sensor | Raw uncleaned sensor observation prior to imputation |
| 61 | `Ozone_missing_mask` | `float64` | quality flag | **NO (Sensor Mask Flag)** | Ingestion Quality Filter | Original missingness mask (1.0 = missing/unphysical, 0.0 = valid) |
| 62 | `Ozone_imputed_flag` | `float64` | quality flag | **NO (Imputation Indicator)** | Preprocessing Imputation Engine | Flag indicating observation was filled via approved short-gap interpolation |
| 63 | `hour` | `int32` | metadata | **NO (Discrete Calendar Metadata)** | Timestamp Extraction | Discrete temporal calendar index (replaced in models by cyclical sin/cos) |
| 64 | `hour_sin` | `float64` | predictor | **YES (Primary Model Predictor)** | Timestamp (Astronomical Calendar) | Trigonometric sine/cosine transformation or one-hot seasonal indicator |
| 65 | `hour_cos` | `float64` | predictor | **YES (Primary Model Predictor)** | Timestamp (Astronomical Calendar) | Trigonometric sine/cosine transformation or one-hot seasonal indicator |
| 66 | `dayofweek` | `int32` | metadata | **NO (Discrete Calendar Metadata)** | Timestamp Extraction | Discrete temporal calendar index (replaced in models by cyclical sin/cos) |
| 67 | `dow_sin` | `float64` | predictor | **YES (Primary Model Predictor)** | Timestamp (Astronomical Calendar) | Trigonometric sine/cosine transformation or one-hot seasonal indicator |
| 68 | `dow_cos` | `float64` | predictor | **YES (Primary Model Predictor)** | Timestamp (Astronomical Calendar) | Trigonometric sine/cosine transformation or one-hot seasonal indicator |
| 69 | `month` | `int32` | metadata | **NO (Discrete Calendar Metadata)** | Timestamp Extraction | Discrete temporal calendar index (replaced in models by cyclical sin/cos) |
| 70 | `month_sin` | `float64` | predictor | **YES (Primary Model Predictor)** | Timestamp (Astronomical Calendar) | Trigonometric sine/cosine transformation or one-hot seasonal indicator |
| 71 | `month_cos` | `float64` | predictor | **YES (Primary Model Predictor)** | Timestamp (Astronomical Calendar) | Trigonometric sine/cosine transformation or one-hot seasonal indicator |
| 72 | `season` | `object` | metadata | **NO (Discrete Calendar Metadata)** | Timestamp Extraction | Discrete temporal calendar index (replaced in models by cyclical sin/cos) |
| 73 | `is_winter` | `float64` | predictor | **YES (Primary Model Predictor)** | Timestamp (Astronomical Calendar) | Trigonometric sine/cosine transformation or one-hot seasonal indicator |
| 74 | `is_summer` | `float64` | predictor | **YES (Primary Model Predictor)** | Timestamp (Astronomical Calendar) | Trigonometric sine/cosine transformation or one-hot seasonal indicator |
| 75 | `is_monsoon` | `float64` | predictor | **YES (Primary Model Predictor)** | Timestamp (Astronomical Calendar) | Trigonometric sine/cosine transformation or one-hot seasonal indicator |
| 76 | `is_post_monsoon` | `float64` | predictor | **YES (Primary Model Predictor)** | Timestamp (Astronomical Calendar) | Trigonometric sine/cosine transformation or one-hot seasonal indicator |
| 77 | `pm25_lag_1` | `float64` | predictor | **YES (Primary Model Predictor)** | Historical PM2.5 Telemetry | Station-wise historical autoregressive lag at t - k hours |
| 78 | `pm25_lag_2` | `float64` | predictor | **YES (Primary Model Predictor)** | Historical PM2.5 Telemetry | Station-wise historical autoregressive lag at t - k hours |
| 79 | `pm25_lag_3` | `float64` | predictor | **YES (Primary Model Predictor)** | Historical PM2.5 Telemetry | Station-wise historical autoregressive lag at t - k hours |
| 80 | `pm25_lag_6` | `float64` | predictor | **YES (Primary Model Predictor)** | Historical PM2.5 Telemetry | Station-wise historical autoregressive lag at t - k hours |
| 81 | `pm25_lag_12` | `float64` | predictor | **YES (Primary Model Predictor)** | Historical PM2.5 Telemetry | Station-wise historical autoregressive lag at t - k hours |
| 82 | `pm25_lag_24` | `float64` | predictor | **YES (Primary Model Predictor)** | Historical PM2.5 Telemetry | Station-wise historical autoregressive lag at t - k hours |
| 83 | `pm25_roll_mean_3h` | `float64` | predictor | **YES (Primary Model Predictor)** | Historical PM2.5 Telemetry | Past-only station-wise rolling window [t-w+1 : t] aggregation |
| 84 | `pm25_roll_mean_6h` | `float64` | predictor | **YES (Primary Model Predictor)** | Historical PM2.5 Telemetry | Past-only station-wise rolling window [t-w+1 : t] aggregation |
| 85 | `pm25_roll_std_6h` | `float64` | predictor | **YES (Primary Model Predictor)** | Historical PM2.5 Telemetry | Past-only station-wise rolling window [t-w+1 : t] aggregation |
| 86 | `pm25_roll_mean_12h` | `float64` | predictor | **YES (Primary Model Predictor)** | Historical PM2.5 Telemetry | Past-only station-wise rolling window [t-w+1 : t] aggregation |
| 87 | `pm25_roll_mean_24h` | `float64` | predictor | **YES (Primary Model Predictor)** | Historical PM2.5 Telemetry | Past-only station-wise rolling window [t-w+1 : t] aggregation |
| 88 | `pm25_roll_std_24h` | `float64` | predictor | **YES (Primary Model Predictor)** | Historical PM2.5 Telemetry | Past-only station-wise rolling window [t-w+1 : t] aggregation |
| 89 | `pm25_roll_min_24h` | `float64` | predictor | **YES (Primary Model Predictor)** | Historical PM2.5 Telemetry | Past-only station-wise rolling window [t-w+1 : t] aggregation |
| 90 | `pm25_roll_max_24h` | `float64` | predictor | **YES (Primary Model Predictor)** | Historical PM2.5 Telemetry | Past-only station-wise rolling window [t-w+1 : t] aggregation |
| 91 | `pm25_trend_diff_1h` | `float64` | predictor | **YES (Primary Model Predictor)** | Historical PM2.5 Telemetry | First/second order backward difference or ratio to 24h rolling baseline |
| 92 | `pm25_trend_diff_3h` | `float64` | predictor | **YES (Primary Model Predictor)** | Historical PM2.5 Telemetry | First/second order backward difference or ratio to 24h rolling baseline |
| 93 | `pm25_trend_diff_24h` | `float64` | predictor | **YES (Primary Model Predictor)** | Historical PM2.5 Telemetry | First/second order backward difference or ratio to 24h rolling baseline |
| 94 | `pm25_acceleration_1h` | `float64` | predictor | **YES (Primary Model Predictor)** | Historical PM2.5 Telemetry | First/second order backward difference or ratio to 24h rolling baseline |
| 95 | `pm25_ratio_to_24h_mean` | `float64` | predictor | **YES (Primary Model Predictor)** | Historical PM2.5 Telemetry | First/second order backward difference or ratio to 24h rolling baseline |
| 96 | `ratio_pm25_pm10` | `float64` | predictor | **YES (Primary Model Predictor)** | CPCB Multi-Pollutant Telemetry | Inter-pollutant photochemical ratio or co-pollutant historical lag |
| 97 | `ratio_no_no2` | `float64` | predictor | **YES (Primary Model Predictor)** | CPCB Multi-Pollutant Telemetry | Inter-pollutant photochemical ratio or co-pollutant historical lag |
| 98 | `pm10_lag_1` | `float64` | predictor | **YES (Primary Model Predictor)** | CPCB Multi-Pollutant Telemetry | Inter-pollutant photochemical ratio or co-pollutant historical lag |
| 99 | `pm10_lag_24` | `float64` | predictor | **YES (Primary Model Predictor)** | CPCB Multi-Pollutant Telemetry | Inter-pollutant photochemical ratio or co-pollutant historical lag |
| 100 | `no2_lag_1` | `float64` | predictor | **YES (Primary Model Predictor)** | CPCB Multi-Pollutant Telemetry | Inter-pollutant photochemical ratio or co-pollutant historical lag |
| 101 | `no2_lag_24` | `float64` | predictor | **YES (Primary Model Predictor)** | CPCB Multi-Pollutant Telemetry | Inter-pollutant photochemical ratio or co-pollutant historical lag |
| 102 | `so2_lag_1` | `float64` | predictor | **YES (Primary Model Predictor)** | CPCB Multi-Pollutant Telemetry | Inter-pollutant photochemical ratio or co-pollutant historical lag |
| 103 | `so2_lag_24` | `float64` | predictor | **YES (Primary Model Predictor)** | CPCB Multi-Pollutant Telemetry | Inter-pollutant photochemical ratio or co-pollutant historical lag |
| 104 | `wind_dir_sin` | `float64` | predictor | **YES (Primary Model Predictor)** | Timestamp (Astronomical Calendar) | Trigonometric sine/cosine transformation or one-hot seasonal indicator |
| 105 | `wind_dir_cos` | `float64` | predictor | **YES (Primary Model Predictor)** | Timestamp (Astronomical Calendar) | Trigonometric sine/cosine transformation or one-hot seasonal indicator |
| 106 | `wind_u` | `float64` | predictor | **YES (Primary Model Predictor)** | Open-Meteo ERA5 Reanalysis (Retrospective Benchmark) | Atmospheric thermodynamic/vector parameter conditioned on observations <= t |
| 107 | `wind_v` | `float64` | predictor | **YES (Primary Model Predictor)** | Open-Meteo ERA5 Reanalysis (Retrospective Benchmark) | Atmospheric thermodynamic/vector parameter conditioned on observations <= t |
| 108 | `sat_vapor_pressure_es` | `float64` | predictor | **YES (Primary Model Predictor)** | Open-Meteo ERA5 Reanalysis (Retrospective Benchmark) | Atmospheric thermodynamic/vector parameter conditioned on observations <= t |
| 109 | `act_vapor_pressure_ea` | `float64` | predictor | **YES (Primary Model Predictor)** | Open-Meteo ERA5 Reanalysis (Retrospective Benchmark) | Atmospheric thermodynamic/vector parameter conditioned on observations <= t |
| 110 | `vapor_pressure_deficit_vpd` | `float64` | predictor | **YES (Primary Model Predictor)** | Open-Meteo ERA5 Reanalysis (Retrospective Benchmark) | Atmospheric thermodynamic/vector parameter conditioned on observations <= t |
| 111 | `dew_point_temp` | `float64` | predictor | **YES (Primary Model Predictor)** | Open-Meteo ERA5 Reanalysis (Retrospective Benchmark) | Atmospheric thermodynamic/vector parameter conditioned on observations <= t |
| 112 | `dew_point_depression` | `float64` | predictor | **YES (Primary Model Predictor)** | Open-Meteo ERA5 Reanalysis (Retrospective Benchmark) | Atmospheric thermodynamic/vector parameter conditioned on observations <= t |
| 113 | `is_atmospheric_stagnation` | `float64` | predictor | **YES (Primary Model Predictor)** | Open-Meteo ERA5 Reanalysis (Retrospective Benchmark) | Atmospheric thermodynamic/vector parameter conditioned on observations <= t |
| 114 | `is_wind_dispersion` | `float64` | predictor | **YES (Primary Model Predictor)** | Open-Meteo ERA5 Reanalysis (Retrospective Benchmark) | Atmospheric thermodynamic/vector parameter conditioned on observations <= t |
| 115 | `ventilation_index_proxy` | `float64` | predictor | **YES (Primary Model Predictor)** | Open-Meteo ERA5 Reanalysis (Retrospective Benchmark) | Atmospheric thermodynamic/vector parameter conditioned on observations <= t |
| 116 | `precip_roll_sum_6h` | `float64` | predictor | **YES (Primary Model Predictor)** | Open-Meteo ERA5 Reanalysis (Retrospective Benchmark) | Atmospheric thermodynamic/vector parameter conditioned on observations <= t |
| 117 | `precip_roll_sum_24h` | `float64` | predictor | **YES (Primary Model Predictor)** | Open-Meteo ERA5 Reanalysis (Retrospective Benchmark) | Atmospheric thermodynamic/vector parameter conditioned on observations <= t |
| 118 | `has_recent_rain_24h` | `float64` | predictor | **YES (Primary Model Predictor)** | Open-Meteo ERA5 Reanalysis (Retrospective Benchmark) | Atmospheric thermodynamic/vector parameter conditioned on observations <= t |
| 119 | `temp_lag_1` | `float64` | predictor | **YES (Primary Model Predictor)** | Open-Meteo ERA5 Reanalysis (Retrospective Benchmark) | Atmospheric thermodynamic/vector parameter conditioned on observations <= t |
| 120 | `temp_lag_24` | `float64` | predictor | **YES (Primary Model Predictor)** | Open-Meteo ERA5 Reanalysis (Retrospective Benchmark) | Atmospheric thermodynamic/vector parameter conditioned on observations <= t |
| 121 | `temp_diff_24h` | `float64` | predictor | **YES (Primary Model Predictor)** | Open-Meteo ERA5 Reanalysis (Retrospective Benchmark) | Atmospheric thermodynamic/vector parameter conditioned on observations <= t |
| 122 | `rh_lag_1` | `float64` | predictor | **YES (Primary Model Predictor)** | Open-Meteo ERA5 Reanalysis (Retrospective Benchmark) | Atmospheric thermodynamic/vector parameter conditioned on observations <= t |
| 123 | `rh_lag_24` | `float64` | predictor | **YES (Primary Model Predictor)** | Open-Meteo ERA5 Reanalysis (Retrospective Benchmark) | Atmospheric thermodynamic/vector parameter conditioned on observations <= t |
| 124 | `rh_diff_24h` | `float64` | predictor | **YES (Primary Model Predictor)** | Open-Meteo ERA5 Reanalysis (Retrospective Benchmark) | Atmospheric thermodynamic/vector parameter conditioned on observations <= t |
| 125 | `wind_speed_lag_1` | `float64` | predictor | **YES (Primary Model Predictor)** | Open-Meteo ERA5 Reanalysis (Retrospective Benchmark) | Atmospheric thermodynamic/vector parameter conditioned on observations <= t |
| 126 | `wind_speed_roll_mean_6h` | `float64` | predictor | **YES (Primary Model Predictor)** | Open-Meteo ERA5 Reanalysis (Retrospective Benchmark) | Atmospheric thermodynamic/vector parameter conditioned on observations <= t |
| 127 | `wind_speed_roll_mean_24h` | `float64` | predictor | **YES (Primary Model Predictor)** | Open-Meteo ERA5 Reanalysis (Retrospective Benchmark) | Atmospheric thermodynamic/vector parameter conditioned on observations <= t |
| 128 | `pressure_lag_1` | `float64` | predictor | **YES (Primary Model Predictor)** | Open-Meteo ERA5 Reanalysis (Retrospective Benchmark) | Atmospheric thermodynamic/vector parameter conditioned on observations <= t |
| 129 | `pressure_tendency_3h` | `float64` | predictor | **YES (Primary Model Predictor)** | Open-Meteo ERA5 Reanalysis (Retrospective Benchmark) | Atmospheric thermodynamic/vector parameter conditioned on observations <= t |
| 130 | `osm_primary_length_1km` | `float64` | predictor (exploratory proxy) | **NO (Exploratory Sensitivity Only)** | Literature Spatial Proxy (Assumed, Unverified) | Static spatial road length / density / highway distance (4 spatial DOF across network) |
| 131 | `osm_secondary_length_1km` | `float64` | predictor (exploratory proxy) | **NO (Exploratory Sensitivity Only)** | Literature Spatial Proxy (Assumed, Unverified) | Static spatial road length / density / highway distance (4 spatial DOF across network) |
| 132 | `osm_total_length_1km` | `float64` | predictor (exploratory proxy) | **NO (Exploratory Sensitivity Only)** | Literature Spatial Proxy (Assumed, Unverified) | Static spatial road length / density / highway distance (4 spatial DOF across network) |
| 133 | `osm_primary_length_3km` | `float64` | predictor (exploratory proxy) | **NO (Exploratory Sensitivity Only)** | Literature Spatial Proxy (Assumed, Unverified) | Static spatial road length / density / highway distance (4 spatial DOF across network) |
| 134 | `osm_secondary_length_3km` | `float64` | predictor (exploratory proxy) | **NO (Exploratory Sensitivity Only)** | Literature Spatial Proxy (Assumed, Unverified) | Static spatial road length / density / highway distance (4 spatial DOF across network) |
| 135 | `osm_total_length_3km` | `float64` | predictor (exploratory proxy) | **NO (Exploratory Sensitivity Only)** | Literature Spatial Proxy (Assumed, Unverified) | Static spatial road length / density / highway distance (4 spatial DOF across network) |
| 136 | `osm_road_density_1km` | `float64` | predictor (exploratory proxy) | **NO (Exploratory Sensitivity Only)** | Literature Spatial Proxy (Assumed, Unverified) | Static spatial road length / density / highway distance (4 spatial DOF across network) |
| 137 | `osm_road_density_3km` | `float64` | predictor (exploratory proxy) | **NO (Exploratory Sensitivity Only)** | Literature Spatial Proxy (Assumed, Unverified) | Static spatial road length / density / highway distance (4 spatial DOF across network) |
| 138 | `osm_primary_density_1km` | `float64` | predictor (exploratory proxy) | **NO (Exploratory Sensitivity Only)** | Literature Spatial Proxy (Assumed, Unverified) | Static spatial road length / density / highway distance (4 spatial DOF across network) |
| 139 | `osm_primary_density_3km` | `float64` | predictor (exploratory proxy) | **NO (Exploratory Sensitivity Only)** | Literature Spatial Proxy (Assumed, Unverified) | Static spatial road length / density / highway distance (4 spatial DOF across network) |
| 140 | `osm_primary_concentration_1km` | `float64` | predictor (exploratory proxy) | **NO (Exploratory Sensitivity Only)** | Literature Spatial Proxy (Assumed, Unverified) | Static spatial road length / density / highway distance (4 spatial DOF across network) |
| 141 | `osm_primary_concentration_3km` | `float64` | predictor (exploratory proxy) | **NO (Exploratory Sensitivity Only)** | Literature Spatial Proxy (Assumed, Unverified) | Static spatial road length / density / highway distance (4 spatial DOF across network) |
| 142 | `osm_distance_to_highway_km` | `float64` | predictor (exploratory proxy) | **NO (Exploratory Sensitivity Only)** | Literature Spatial Proxy (Assumed, Unverified) | Static spatial road length / density / highway distance (4 spatial DOF across network) |
| 143 | `osm_intersection_density_km2` | `float64` | predictor (exploratory proxy) | **NO (Exploratory Sensitivity Only)** | Literature Spatial Proxy (Assumed, Unverified) | Static spatial road length / density / highway distance (4 spatial DOF across network) |
| 144 | `is_morning_peak` | `float64` | predictor | **YES (Primary Model Predictor)** | Calendar & Clock Hour (Purely Temporal) | Diurnal traffic congestion window (NO road metric multiplication) |
| 145 | `is_evening_peak` | `float64` | predictor | **YES (Primary Model Predictor)** | Calendar & Clock Hour (Purely Temporal) | Diurnal traffic congestion window (NO road metric multiplication) |
| 146 | `is_traffic_peak` | `float64` | predictor | **YES (Primary Model Predictor)** | Calendar & Clock Hour (Purely Temporal) | Diurnal traffic congestion window (NO road metric multiplication) |
| 147 | `is_weekend` | `float64` | predictor | **YES (Primary Model Predictor)** | Calendar & Clock Hour (Purely Temporal) | Diurnal traffic congestion window (NO road metric multiplication) |
| 148 | `is_weekday_peak` | `float64` | predictor | **YES (Primary Model Predictor)** | Calendar & Clock Hour (Purely Temporal) | Diurnal traffic congestion window (NO road metric multiplication) |
| 149 | `is_freight_window_exploratory` | `float64` | predictor | **YES (Primary Model Predictor)** | Calendar & Clock Hour (Purely Temporal) | Diurnal traffic congestion window (NO road metric multiplication) |
| 150 | `is_daytime_offpeak` | `float64` | predictor | **YES (Primary Model Predictor)** | Calendar & Clock Hour (Purely Temporal) | Diurnal traffic congestion window (NO road metric multiplication) |
| 151 | `traffic_pressure_total_1km` | `float64` | predictor (exploratory proxy) | **NO (Exploratory Sensitivity Only)** | Assumed Road Proxy x Temporal Window | Interaction product of assumed static road proxy and temporal congestion flag |
| 152 | `traffic_pressure_weekday_peak` | `float64` | predictor (exploratory proxy) | **NO (Exploratory Sensitivity Only)** | Assumed Road Proxy x Temporal Window | Interaction product of assumed static road proxy and temporal congestion flag |
| 153 | `traffic_pressure_primary_1km` | `float64` | predictor (exploratory proxy) | **NO (Exploratory Sensitivity Only)** | Assumed Road Proxy x Temporal Window | Interaction product of assumed static road proxy and temporal congestion flag |
| 154 | `freight_pressure_proxy` | `float64` | predictor (exploratory proxy) | **NO (Exploratory Sensitivity Only)** | Assumed Road Proxy x Temporal Window | Interaction product of assumed static road proxy and temporal congestion flag |
| 155 | `highway_rush_pressure` | `float64` | predictor (exploratory proxy) | **NO (Exploratory Sensitivity Only)** | Assumed Road Proxy x Temporal Window | Interaction product of assumed static road proxy and temporal congestion flag |
| 156 | `partition` | `object` | metadata | **NO (Evaluation Partition Guard)** | Chronological Splitter Protocol | Chronological split assignment or 48-hour boundary purge flag |
| 157 | `is_boundary_buffer` | `bool` | metadata | **NO (Evaluation Partition Guard)** | Chronological Splitter Protocol | Chronological split assignment or 48-hour boundary purge flag |
| 158 | `target_pm25_h1` | `float64` | target | **NO (Target Ground Truth)** | CPCB Physical Raw PM2.5 Sensor Telemetry | Forward prospective shift t+h / forward 24h rolling mean / threshold alert |
| 159 | `target_pm25_h6` | `float64` | target | **NO (Target Ground Truth)** | CPCB Physical Raw PM2.5 Sensor Telemetry | Forward prospective shift t+h / forward 24h rolling mean / threshold alert |
| 160 | `target_pm25_h24` | `float64` | target | **NO (Target Ground Truth)** | CPCB Physical Raw PM2.5 Sensor Telemetry | Forward prospective shift t+h / forward 24h rolling mean / threshold alert |
| 161 | `target_pm25_h48` | `float64` | target | **NO (Target Ground Truth)** | CPCB Physical Raw PM2.5 Sensor Telemetry | Forward prospective shift t+h / forward 24h rolling mean / threshold alert |
| 162 | `target_pm25_24h_mean` | `float64` | target | **NO (Target Ground Truth)** | CPCB Physical Raw PM2.5 Sensor Telemetry | Forward prospective shift t+h / forward 24h rolling mean / threshold alert |
| 163 | `target_pm25_24h_valid_count` | `float64` | target | **NO (Target Ground Truth)** | CPCB Physical Raw PM2.5 Sensor Telemetry | Forward prospective shift t+h / forward 24h rolling mean / threshold alert |
| 164 | `target_hotspot_acute_h1` | `float64` | target | **NO (Target Ground Truth)** | CPCB Physical Raw PM2.5 Sensor Telemetry | Forward prospective shift t+h / forward 24h rolling mean / threshold alert |
| 165 | `target_hotspot_spike_h1` | `float64` | target | **NO (Target Ground Truth)** | CPCB Physical Raw PM2.5 Sensor Telemetry | Forward prospective shift t+h / forward 24h rolling mean / threshold alert |
| 166 | `target_hotspot_very_poor_h1` | `float64` | target | **NO (Target Ground Truth)** | CPCB Physical Raw PM2.5 Sensor Telemetry | Forward prospective shift t+h / forward 24h rolling mean / threshold alert |
| 167 | `target_hotspot_acute_h6` | `float64` | target | **NO (Target Ground Truth)** | CPCB Physical Raw PM2.5 Sensor Telemetry | Forward prospective shift t+h / forward 24h rolling mean / threshold alert |
| 168 | `target_hotspot_spike_h6` | `float64` | target | **NO (Target Ground Truth)** | CPCB Physical Raw PM2.5 Sensor Telemetry | Forward prospective shift t+h / forward 24h rolling mean / threshold alert |
| 169 | `target_hotspot_very_poor_h6` | `float64` | target | **NO (Target Ground Truth)** | CPCB Physical Raw PM2.5 Sensor Telemetry | Forward prospective shift t+h / forward 24h rolling mean / threshold alert |
| 170 | `target_hotspot_acute_h24` | `float64` | target | **NO (Target Ground Truth)** | CPCB Physical Raw PM2.5 Sensor Telemetry | Forward prospective shift t+h / forward 24h rolling mean / threshold alert |
| 171 | `target_hotspot_spike_h24` | `float64` | target | **NO (Target Ground Truth)** | CPCB Physical Raw PM2.5 Sensor Telemetry | Forward prospective shift t+h / forward 24h rolling mean / threshold alert |
| 172 | `target_hotspot_very_poor_h24` | `float64` | target | **NO (Target Ground Truth)** | CPCB Physical Raw PM2.5 Sensor Telemetry | Forward prospective shift t+h / forward 24h rolling mean / threshold alert |
| 173 | `target_hotspot_24h_statutory` | `float64` | target | **NO (Target Ground Truth)** | CPCB Physical Raw PM2.5 Sensor Telemetry | Forward prospective shift t+h / forward 24h rolling mean / threshold alert |

---

*Report generated and mathematically verified against master Parquet schema.*