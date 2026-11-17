# Exhaustive Multi-Source Feature Dictionary

**Document Code:** `SPEC-M2-FEATURE-DICTIONARY`  
**Project:** Traffic-Aware Spatio-Temporal PM2.5 Concentration and Hotspot Forecasting in Hyderabad  
**Milestone:** Milestone 2 – Multi-Source Feature Engineering  
**Scope:** 173 Reconciled Columns across Administrative, Telemetry, Predictor, Road Proxy, and Target Modalities  

---

## 1. Schema Reconciliation & Governance Architecture

The master feature store (`feature_store_primary.parquet`) contains exactly **173 columns**. In strict accordance with research governance:

- **Primary Model Predictors (84 columns):** Verified physical sensor telemetry, historical autoregressive lags, past-only rolling statistics, dynamic trends, co-pollutant photochemical ratios, cyclical calendar features, Open-Meteo ERA5 atmospheric variables, and purely temporal diurnal traffic windows. **Approved for model training.**
- **Exploratory Road Proxies (19 columns):** 14 static spatial roadway proxies and 5 traffic pressure interaction features derived from assumed literature proxies. **EXCLUDED from primary model inputs.** Reserved strictly for optional exploratory sensitivity experiments.
- **Data Quality & Telemetry Masks (40 columns):** 16 gap length/category trackers, 8 raw sensor readings, 8 missingness masks, and 8 imputation flags. Quarantined as metadata; **prohibited from model inputs.**
- **Administrative Metadata (11 columns):** 7 geodetic/partition attributes and 4 discrete calendar fields. **Prohibited from model inputs.**
- **Administrative Identifiers (3 columns):** `station`, `station_id`, `datetime`. Non-numeric keys.
- **Prospective Targets (16 columns):** Ground-truth prediction targets (+1h, +6h, +24h, +48h regression and hotspot classification). Sequestered in `feature_store_targets.parquet`.

### Mathematical Identity Verification
$$\text{Total Columns (173)} = \text{Identifiers (3)} + \text{Metadata/Quality (51)} + \text{Primary Predictors (84)} + \text{Exploratory Proxies (19)} + \text{Targets (16)}$$
$$\mathbf{173 = 3 + 51 + 84 + 19 + 16}$$

---

## 2. Complete Catalog of Reconciled Feature Store Columns

| # | Variable Name | Role | Category | Unit | Allowed in Primary Model? | Source | Transformation & Operational Definition |
| :---: | :--- | :--- | :--- | :--- | :---: | :--- | :--- |
| 1 | `datetime` | Identifier | identifier | ISO 8601 | NO | CPCB & Open-Meteo Hourly Timeline | Continuous ISO 8601 hourly timestamp in Indian Standard Time (UTC+5:30) |
| 2 | `PM2.5` | Primary Predictor | predictor | ug/m3 (CO: mg/m3) | **YES** | CPCB In-Situ CAAQMS Telemetry | Preprocessed ambient concentration at prediction origin t (short gaps filled) |
| 3 | `PM10` | Primary Predictor | predictor | ug/m3 (CO: mg/m3) | **YES** | CPCB In-Situ CAAQMS Telemetry | Preprocessed ambient concentration at prediction origin t (short gaps filled) |
| 4 | `NO` | Primary Predictor | predictor | ug/m3 (CO: mg/m3) | **YES** | CPCB In-Situ CAAQMS Telemetry | Preprocessed ambient concentration at prediction origin t (short gaps filled) |
| 5 | `NO2` | Primary Predictor | predictor | ug/m3 (CO: mg/m3) | **YES** | CPCB In-Situ CAAQMS Telemetry | Preprocessed ambient concentration at prediction origin t (short gaps filled) |
| 6 | `NH3` | Primary Predictor | predictor | ug/m3 (CO: mg/m3) | **YES** | CPCB In-Situ CAAQMS Telemetry | Preprocessed ambient concentration at prediction origin t (short gaps filled) |
| 7 | `SO2` | Primary Predictor | predictor | ug/m3 (CO: mg/m3) | **YES** | CPCB In-Situ CAAQMS Telemetry | Preprocessed ambient concentration at prediction origin t (short gaps filled) |
| 8 | `CO` | Primary Predictor | predictor | ug/m3 (CO: mg/m3) | **YES** | CPCB In-Situ CAAQMS Telemetry | Preprocessed ambient concentration at prediction origin t (short gaps filled) |
| 9 | `Ozone` | Primary Predictor | predictor | ug/m3 (CO: mg/m3) | **YES** | CPCB In-Situ CAAQMS Telemetry | Preprocessed ambient concentration at prediction origin t (short gaps filled) |
| 10 | `station` | Identifier | identifier | Text | NO | CPCB CAAQMS Station Registry | Official station name string (Bollaram, Central Univ, ECIL, ICRISAT) |
| 11 | `station_id` | Identifier | identifier | Text | NO | CPCB Station Identifier Standard | Unique standardized station abbreviation code |
| 12 | `temperature_2m` | Primary Predictor | predictor | Standard Meteorological Units | **YES** | Open-Meteo ERA5 Reanalysis (Retrospective Benchmark) | Atmospheric thermodynamic/vector parameter conditioned on observations <= t |
| 13 | `relative_humidity_2m` | Primary Predictor | predictor | Standard Meteorological Units | **YES** | Open-Meteo ERA5 Reanalysis (Retrospective Benchmark) | Atmospheric thermodynamic/vector parameter conditioned on observations <= t |
| 14 | `wind_speed_10m` | Primary Predictor | predictor | Standard Meteorological Units | **YES** | Open-Meteo ERA5 Reanalysis (Retrospective Benchmark) | Atmospheric thermodynamic/vector parameter conditioned on observations <= t |
| 15 | `wind_direction_10m` | Primary Predictor | predictor | Standard Meteorological Units | **YES** | Open-Meteo ERA5 Reanalysis (Retrospective Benchmark) | Atmospheric thermodynamic/vector parameter conditioned on observations <= t |
| 16 | `precipitation` | Primary Predictor | predictor | Standard Meteorological Units | **YES** | Open-Meteo ERA5 Reanalysis (Retrospective Benchmark) | Atmospheric thermodynamic/vector parameter conditioned on observations <= t |
| 17 | `surface_pressure` | Primary Predictor | predictor | Standard Meteorological Units | **YES** | Open-Meteo ERA5 Reanalysis (Retrospective Benchmark) | Atmospheric thermodynamic/vector parameter conditioned on observations <= t |
| 18 | `latitude` | Metadata / Quality | metadata | Degrees / Metres / Category | NO | CPCB Station Registry / Land-Use Survey | Static geodetic coordinate or urban typology classification |
| 19 | `longitude` | Metadata / Quality | metadata | Degrees / Metres / Category | NO | CPCB Station Registry / Land-Use Survey | Static geodetic coordinate or urban typology classification |
| 20 | `easting_utm44n` | Metadata / Quality | metadata | Degrees / Metres / Category | NO | CPCB Station Registry / Land-Use Survey | Static geodetic coordinate or urban typology classification |
| 21 | `northing_utm44n` | Metadata / Quality | metadata | Degrees / Metres / Category | NO | CPCB Station Registry / Land-Use Survey | Static geodetic coordinate or urban typology classification |
| 22 | `typology` | Metadata / Quality | metadata | Degrees / Metres / Category | NO | CPCB Station Registry / Land-Use Survey | Static geodetic coordinate or urban typology classification |
| 23 | `PM2.5_gap_length` | Metadata / Quality | quality flag | Hours / Category | NO | Preprocessing Missingness Module | Consecutive missing-hour outage run-length and gap severity category |
| 24 | `PM2.5_gap_category` | Metadata / Quality | quality flag | Hours / Category | NO | Preprocessing Missingness Module | Consecutive missing-hour outage run-length and gap severity category |
| 25 | `PM10_gap_length` | Metadata / Quality | quality flag | Hours / Category | NO | Preprocessing Missingness Module | Consecutive missing-hour outage run-length and gap severity category |
| 26 | `PM10_gap_category` | Metadata / Quality | quality flag | Hours / Category | NO | Preprocessing Missingness Module | Consecutive missing-hour outage run-length and gap severity category |
| 27 | `NO_gap_length` | Metadata / Quality | quality flag | Hours / Category | NO | Preprocessing Missingness Module | Consecutive missing-hour outage run-length and gap severity category |
| 28 | `NO_gap_category` | Metadata / Quality | quality flag | Hours / Category | NO | Preprocessing Missingness Module | Consecutive missing-hour outage run-length and gap severity category |
| 29 | `NO2_gap_length` | Metadata / Quality | quality flag | Hours / Category | NO | Preprocessing Missingness Module | Consecutive missing-hour outage run-length and gap severity category |
| 30 | `NO2_gap_category` | Metadata / Quality | quality flag | Hours / Category | NO | Preprocessing Missingness Module | Consecutive missing-hour outage run-length and gap severity category |
| 31 | `NH3_gap_length` | Metadata / Quality | quality flag | Hours / Category | NO | Preprocessing Missingness Module | Consecutive missing-hour outage run-length and gap severity category |
| 32 | `NH3_gap_category` | Metadata / Quality | quality flag | Hours / Category | NO | Preprocessing Missingness Module | Consecutive missing-hour outage run-length and gap severity category |
| 33 | `SO2_gap_length` | Metadata / Quality | quality flag | Hours / Category | NO | Preprocessing Missingness Module | Consecutive missing-hour outage run-length and gap severity category |
| 34 | `SO2_gap_category` | Metadata / Quality | quality flag | Hours / Category | NO | Preprocessing Missingness Module | Consecutive missing-hour outage run-length and gap severity category |
| 35 | `CO_gap_length` | Metadata / Quality | quality flag | Hours / Category | NO | Preprocessing Missingness Module | Consecutive missing-hour outage run-length and gap severity category |
| 36 | `CO_gap_category` | Metadata / Quality | quality flag | Hours / Category | NO | Preprocessing Missingness Module | Consecutive missing-hour outage run-length and gap severity category |
| 37 | `Ozone_gap_length` | Metadata / Quality | quality flag | Hours / Category | NO | Preprocessing Missingness Module | Consecutive missing-hour outage run-length and gap severity category |
| 38 | `Ozone_gap_category` | Metadata / Quality | quality flag | Hours / Category | NO | Preprocessing Missingness Module | Consecutive missing-hour outage run-length and gap severity category |
| 39 | `PM2.5_raw` | Metadata / Quality | quality flag | ug/m3 or mg/m3 | NO | CPCB CAAQMS Physical Sensor | Raw uncleaned sensor observation prior to imputation |
| 40 | `PM2.5_missing_mask` | Metadata / Quality | quality flag | Binary (0/1) | NO | Ingestion Quality Filter | Original missingness mask (1.0 = missing/unphysical, 0.0 = valid) |
| 41 | `PM2.5_imputed_flag` | Metadata / Quality | quality flag | Binary (0/1) | NO | Preprocessing Imputation Engine | Flag indicating observation was filled via approved short-gap interpolation |
| 42 | `PM10_raw` | Metadata / Quality | quality flag | ug/m3 or mg/m3 | NO | CPCB CAAQMS Physical Sensor | Raw uncleaned sensor observation prior to imputation |
| 43 | `PM10_missing_mask` | Metadata / Quality | quality flag | Binary (0/1) | NO | Ingestion Quality Filter | Original missingness mask (1.0 = missing/unphysical, 0.0 = valid) |
| 44 | `PM10_imputed_flag` | Metadata / Quality | quality flag | Binary (0/1) | NO | Preprocessing Imputation Engine | Flag indicating observation was filled via approved short-gap interpolation |
| 45 | `NO_raw` | Metadata / Quality | quality flag | ug/m3 or mg/m3 | NO | CPCB CAAQMS Physical Sensor | Raw uncleaned sensor observation prior to imputation |
| 46 | `NO_missing_mask` | Metadata / Quality | quality flag | Binary (0/1) | NO | Ingestion Quality Filter | Original missingness mask (1.0 = missing/unphysical, 0.0 = valid) |
| 47 | `NO_imputed_flag` | Metadata / Quality | quality flag | Binary (0/1) | NO | Preprocessing Imputation Engine | Flag indicating observation was filled via approved short-gap interpolation |
| 48 | `NO2_raw` | Metadata / Quality | quality flag | ug/m3 or mg/m3 | NO | CPCB CAAQMS Physical Sensor | Raw uncleaned sensor observation prior to imputation |
| 49 | `NO2_missing_mask` | Metadata / Quality | quality flag | Binary (0/1) | NO | Ingestion Quality Filter | Original missingness mask (1.0 = missing/unphysical, 0.0 = valid) |
| 50 | `NO2_imputed_flag` | Metadata / Quality | quality flag | Binary (0/1) | NO | Preprocessing Imputation Engine | Flag indicating observation was filled via approved short-gap interpolation |
| 51 | `NH3_raw` | Metadata / Quality | quality flag | ug/m3 or mg/m3 | NO | CPCB CAAQMS Physical Sensor | Raw uncleaned sensor observation prior to imputation |
| 52 | `NH3_missing_mask` | Metadata / Quality | quality flag | Binary (0/1) | NO | Ingestion Quality Filter | Original missingness mask (1.0 = missing/unphysical, 0.0 = valid) |
| 53 | `NH3_imputed_flag` | Metadata / Quality | quality flag | Binary (0/1) | NO | Preprocessing Imputation Engine | Flag indicating observation was filled via approved short-gap interpolation |
| 54 | `SO2_raw` | Metadata / Quality | quality flag | ug/m3 or mg/m3 | NO | CPCB CAAQMS Physical Sensor | Raw uncleaned sensor observation prior to imputation |
| 55 | `SO2_missing_mask` | Metadata / Quality | quality flag | Binary (0/1) | NO | Ingestion Quality Filter | Original missingness mask (1.0 = missing/unphysical, 0.0 = valid) |
| 56 | `SO2_imputed_flag` | Metadata / Quality | quality flag | Binary (0/1) | NO | Preprocessing Imputation Engine | Flag indicating observation was filled via approved short-gap interpolation |
| 57 | `CO_raw` | Metadata / Quality | quality flag | ug/m3 or mg/m3 | NO | CPCB CAAQMS Physical Sensor | Raw uncleaned sensor observation prior to imputation |
| 58 | `CO_missing_mask` | Metadata / Quality | quality flag | Binary (0/1) | NO | Ingestion Quality Filter | Original missingness mask (1.0 = missing/unphysical, 0.0 = valid) |
| 59 | `CO_imputed_flag` | Metadata / Quality | quality flag | Binary (0/1) | NO | Preprocessing Imputation Engine | Flag indicating observation was filled via approved short-gap interpolation |
| 60 | `Ozone_raw` | Metadata / Quality | quality flag | ug/m3 or mg/m3 | NO | CPCB CAAQMS Physical Sensor | Raw uncleaned sensor observation prior to imputation |
| 61 | `Ozone_missing_mask` | Metadata / Quality | quality flag | Binary (0/1) | NO | Ingestion Quality Filter | Original missingness mask (1.0 = missing/unphysical, 0.0 = valid) |
| 62 | `Ozone_imputed_flag` | Metadata / Quality | quality flag | Binary (0/1) | NO | Preprocessing Imputation Engine | Flag indicating observation was filled via approved short-gap interpolation |
| 63 | `hour` | Metadata / Quality | metadata | Integer / Category | NO | Timestamp Extraction | Discrete temporal calendar index (replaced in models by cyclical sin/cos) |
| 64 | `hour_sin` | Primary Predictor | predictor | Continuous [-1, 1] or Binary | **YES** | Timestamp (Astronomical Calendar) | Trigonometric sine/cosine transformation or one-hot seasonal indicator |
| 65 | `hour_cos` | Primary Predictor | predictor | Continuous [-1, 1] or Binary | **YES** | Timestamp (Astronomical Calendar) | Trigonometric sine/cosine transformation or one-hot seasonal indicator |
| 66 | `dayofweek` | Metadata / Quality | metadata | Integer / Category | NO | Timestamp Extraction | Discrete temporal calendar index (replaced in models by cyclical sin/cos) |
| 67 | `dow_sin` | Primary Predictor | predictor | Continuous [-1, 1] or Binary | **YES** | Timestamp (Astronomical Calendar) | Trigonometric sine/cosine transformation or one-hot seasonal indicator |
| 68 | `dow_cos` | Primary Predictor | predictor | Continuous [-1, 1] or Binary | **YES** | Timestamp (Astronomical Calendar) | Trigonometric sine/cosine transformation or one-hot seasonal indicator |
| 69 | `month` | Metadata / Quality | metadata | Integer / Category | NO | Timestamp Extraction | Discrete temporal calendar index (replaced in models by cyclical sin/cos) |
| 70 | `month_sin` | Primary Predictor | predictor | Continuous [-1, 1] or Binary | **YES** | Timestamp (Astronomical Calendar) | Trigonometric sine/cosine transformation or one-hot seasonal indicator |
| 71 | `month_cos` | Primary Predictor | predictor | Continuous [-1, 1] or Binary | **YES** | Timestamp (Astronomical Calendar) | Trigonometric sine/cosine transformation or one-hot seasonal indicator |
| 72 | `season` | Metadata / Quality | metadata | Integer / Category | NO | Timestamp Extraction | Discrete temporal calendar index (replaced in models by cyclical sin/cos) |
| 73 | `is_winter` | Primary Predictor | predictor | Continuous [-1, 1] or Binary | **YES** | Timestamp (Astronomical Calendar) | Trigonometric sine/cosine transformation or one-hot seasonal indicator |
| 74 | `is_summer` | Primary Predictor | predictor | Continuous [-1, 1] or Binary | **YES** | Timestamp (Astronomical Calendar) | Trigonometric sine/cosine transformation or one-hot seasonal indicator |
| 75 | `is_monsoon` | Primary Predictor | predictor | Continuous [-1, 1] or Binary | **YES** | Timestamp (Astronomical Calendar) | Trigonometric sine/cosine transformation or one-hot seasonal indicator |
| 76 | `is_post_monsoon` | Primary Predictor | predictor | Continuous [-1, 1] or Binary | **YES** | Timestamp (Astronomical Calendar) | Trigonometric sine/cosine transformation or one-hot seasonal indicator |
| 77 | `pm25_lag_1` | Primary Predictor | predictor | ug/m3 | **YES** | Historical PM2.5 Telemetry | Station-wise historical autoregressive lag at t - k hours |
| 78 | `pm25_lag_2` | Primary Predictor | predictor | ug/m3 | **YES** | Historical PM2.5 Telemetry | Station-wise historical autoregressive lag at t - k hours |
| 79 | `pm25_lag_3` | Primary Predictor | predictor | ug/m3 | **YES** | Historical PM2.5 Telemetry | Station-wise historical autoregressive lag at t - k hours |
| 80 | `pm25_lag_6` | Primary Predictor | predictor | ug/m3 | **YES** | Historical PM2.5 Telemetry | Station-wise historical autoregressive lag at t - k hours |
| 81 | `pm25_lag_12` | Primary Predictor | predictor | ug/m3 | **YES** | Historical PM2.5 Telemetry | Station-wise historical autoregressive lag at t - k hours |
| 82 | `pm25_lag_24` | Primary Predictor | predictor | ug/m3 | **YES** | Historical PM2.5 Telemetry | Station-wise historical autoregressive lag at t - k hours |
| 83 | `pm25_roll_mean_3h` | Primary Predictor | predictor | ug/m3 | **YES** | Historical PM2.5 Telemetry | Past-only station-wise rolling window [t-w+1 : t] aggregation |
| 84 | `pm25_roll_mean_6h` | Primary Predictor | predictor | ug/m3 | **YES** | Historical PM2.5 Telemetry | Past-only station-wise rolling window [t-w+1 : t] aggregation |
| 85 | `pm25_roll_std_6h` | Primary Predictor | predictor | ug/m3 | **YES** | Historical PM2.5 Telemetry | Past-only station-wise rolling window [t-w+1 : t] aggregation |
| 86 | `pm25_roll_mean_12h` | Primary Predictor | predictor | ug/m3 | **YES** | Historical PM2.5 Telemetry | Past-only station-wise rolling window [t-w+1 : t] aggregation |
| 87 | `pm25_roll_mean_24h` | Primary Predictor | predictor | ug/m3 | **YES** | Historical PM2.5 Telemetry | Past-only station-wise rolling window [t-w+1 : t] aggregation |
| 88 | `pm25_roll_std_24h` | Primary Predictor | predictor | ug/m3 | **YES** | Historical PM2.5 Telemetry | Past-only station-wise rolling window [t-w+1 : t] aggregation |
| 89 | `pm25_roll_min_24h` | Primary Predictor | predictor | ug/m3 | **YES** | Historical PM2.5 Telemetry | Past-only station-wise rolling window [t-w+1 : t] aggregation |
| 90 | `pm25_roll_max_24h` | Primary Predictor | predictor | ug/m3 | **YES** | Historical PM2.5 Telemetry | Past-only station-wise rolling window [t-w+1 : t] aggregation |
| 91 | `pm25_trend_diff_1h` | Primary Predictor | predictor | ug/m3 or Ratio | **YES** | Historical PM2.5 Telemetry | First/second order backward difference or ratio to 24h rolling baseline |
| 92 | `pm25_trend_diff_3h` | Primary Predictor | predictor | ug/m3 or Ratio | **YES** | Historical PM2.5 Telemetry | First/second order backward difference or ratio to 24h rolling baseline |
| 93 | `pm25_trend_diff_24h` | Primary Predictor | predictor | ug/m3 or Ratio | **YES** | Historical PM2.5 Telemetry | First/second order backward difference or ratio to 24h rolling baseline |
| 94 | `pm25_acceleration_1h` | Primary Predictor | predictor | ug/m3 or Ratio | **YES** | Historical PM2.5 Telemetry | First/second order backward difference or ratio to 24h rolling baseline |
| 95 | `pm25_ratio_to_24h_mean` | Primary Predictor | predictor | ug/m3 or Ratio | **YES** | Historical PM2.5 Telemetry | First/second order backward difference or ratio to 24h rolling baseline |
| 96 | `ratio_pm25_pm10` | Primary Predictor | predictor | Ratio or ug/m3 | **YES** | CPCB Multi-Pollutant Telemetry | Inter-pollutant photochemical ratio or co-pollutant historical lag |
| 97 | `ratio_no_no2` | Primary Predictor | predictor | Ratio or ug/m3 | **YES** | CPCB Multi-Pollutant Telemetry | Inter-pollutant photochemical ratio or co-pollutant historical lag |
| 98 | `pm10_lag_1` | Primary Predictor | predictor | Ratio or ug/m3 | **YES** | CPCB Multi-Pollutant Telemetry | Inter-pollutant photochemical ratio or co-pollutant historical lag |
| 99 | `pm10_lag_24` | Primary Predictor | predictor | Ratio or ug/m3 | **YES** | CPCB Multi-Pollutant Telemetry | Inter-pollutant photochemical ratio or co-pollutant historical lag |
| 100 | `no2_lag_1` | Primary Predictor | predictor | Ratio or ug/m3 | **YES** | CPCB Multi-Pollutant Telemetry | Inter-pollutant photochemical ratio or co-pollutant historical lag |
| 101 | `no2_lag_24` | Primary Predictor | predictor | Ratio or ug/m3 | **YES** | CPCB Multi-Pollutant Telemetry | Inter-pollutant photochemical ratio or co-pollutant historical lag |
| 102 | `so2_lag_1` | Primary Predictor | predictor | Ratio or ug/m3 | **YES** | CPCB Multi-Pollutant Telemetry | Inter-pollutant photochemical ratio or co-pollutant historical lag |
| 103 | `so2_lag_24` | Primary Predictor | predictor | Ratio or ug/m3 | **YES** | CPCB Multi-Pollutant Telemetry | Inter-pollutant photochemical ratio or co-pollutant historical lag |
| 104 | `wind_dir_sin` | Primary Predictor | predictor | Continuous [-1, 1] or Binary | **YES** | Timestamp (Astronomical Calendar) | Trigonometric sine/cosine transformation or one-hot seasonal indicator |
| 105 | `wind_dir_cos` | Primary Predictor | predictor | Continuous [-1, 1] or Binary | **YES** | Timestamp (Astronomical Calendar) | Trigonometric sine/cosine transformation or one-hot seasonal indicator |
| 106 | `wind_u` | Primary Predictor | predictor | Standard Meteorological Units | **YES** | Open-Meteo ERA5 Reanalysis (Retrospective Benchmark) | Atmospheric thermodynamic/vector parameter conditioned on observations <= t |
| 107 | `wind_v` | Primary Predictor | predictor | Standard Meteorological Units | **YES** | Open-Meteo ERA5 Reanalysis (Retrospective Benchmark) | Atmospheric thermodynamic/vector parameter conditioned on observations <= t |
| 108 | `sat_vapor_pressure_es` | Primary Predictor | predictor | Standard Meteorological Units | **YES** | Open-Meteo ERA5 Reanalysis (Retrospective Benchmark) | Atmospheric thermodynamic/vector parameter conditioned on observations <= t |
| 109 | `act_vapor_pressure_ea` | Primary Predictor | predictor | Standard Meteorological Units | **YES** | Open-Meteo ERA5 Reanalysis (Retrospective Benchmark) | Atmospheric thermodynamic/vector parameter conditioned on observations <= t |
| 110 | `vapor_pressure_deficit_vpd` | Primary Predictor | predictor | Standard Meteorological Units | **YES** | Open-Meteo ERA5 Reanalysis (Retrospective Benchmark) | Atmospheric thermodynamic/vector parameter conditioned on observations <= t |
| 111 | `dew_point_temp` | Primary Predictor | predictor | Standard Meteorological Units | **YES** | Open-Meteo ERA5 Reanalysis (Retrospective Benchmark) | Atmospheric thermodynamic/vector parameter conditioned on observations <= t |
| 112 | `dew_point_depression` | Primary Predictor | predictor | Standard Meteorological Units | **YES** | Open-Meteo ERA5 Reanalysis (Retrospective Benchmark) | Atmospheric thermodynamic/vector parameter conditioned on observations <= t |
| 113 | `is_atmospheric_stagnation` | Primary Predictor | predictor | Standard Meteorological Units | **YES** | Open-Meteo ERA5 Reanalysis (Retrospective Benchmark) | Atmospheric thermodynamic/vector parameter conditioned on observations <= t |
| 114 | `is_wind_dispersion` | Primary Predictor | predictor | Standard Meteorological Units | **YES** | Open-Meteo ERA5 Reanalysis (Retrospective Benchmark) | Atmospheric thermodynamic/vector parameter conditioned on observations <= t |
| 115 | `ventilation_index_proxy` | Primary Predictor | predictor | Standard Meteorological Units | **YES** | Open-Meteo ERA5 Reanalysis (Retrospective Benchmark) | Atmospheric thermodynamic/vector parameter conditioned on observations <= t |
| 116 | `precip_roll_sum_6h` | Primary Predictor | predictor | Standard Meteorological Units | **YES** | Open-Meteo ERA5 Reanalysis (Retrospective Benchmark) | Atmospheric thermodynamic/vector parameter conditioned on observations <= t |
| 117 | `precip_roll_sum_24h` | Primary Predictor | predictor | Standard Meteorological Units | **YES** | Open-Meteo ERA5 Reanalysis (Retrospective Benchmark) | Atmospheric thermodynamic/vector parameter conditioned on observations <= t |
| 118 | `has_recent_rain_24h` | Primary Predictor | predictor | Standard Meteorological Units | **YES** | Open-Meteo ERA5 Reanalysis (Retrospective Benchmark) | Atmospheric thermodynamic/vector parameter conditioned on observations <= t |
| 119 | `temp_lag_1` | Primary Predictor | predictor | Standard Meteorological Units | **YES** | Open-Meteo ERA5 Reanalysis (Retrospective Benchmark) | Atmospheric thermodynamic/vector parameter conditioned on observations <= t |
| 120 | `temp_lag_24` | Primary Predictor | predictor | Standard Meteorological Units | **YES** | Open-Meteo ERA5 Reanalysis (Retrospective Benchmark) | Atmospheric thermodynamic/vector parameter conditioned on observations <= t |
| 121 | `temp_diff_24h` | Primary Predictor | predictor | Standard Meteorological Units | **YES** | Open-Meteo ERA5 Reanalysis (Retrospective Benchmark) | Atmospheric thermodynamic/vector parameter conditioned on observations <= t |
| 122 | `rh_lag_1` | Primary Predictor | predictor | Standard Meteorological Units | **YES** | Open-Meteo ERA5 Reanalysis (Retrospective Benchmark) | Atmospheric thermodynamic/vector parameter conditioned on observations <= t |
| 123 | `rh_lag_24` | Primary Predictor | predictor | Standard Meteorological Units | **YES** | Open-Meteo ERA5 Reanalysis (Retrospective Benchmark) | Atmospheric thermodynamic/vector parameter conditioned on observations <= t |
| 124 | `rh_diff_24h` | Primary Predictor | predictor | Standard Meteorological Units | **YES** | Open-Meteo ERA5 Reanalysis (Retrospective Benchmark) | Atmospheric thermodynamic/vector parameter conditioned on observations <= t |
| 125 | `wind_speed_lag_1` | Primary Predictor | predictor | Standard Meteorological Units | **YES** | Open-Meteo ERA5 Reanalysis (Retrospective Benchmark) | Atmospheric thermodynamic/vector parameter conditioned on observations <= t |
| 126 | `wind_speed_roll_mean_6h` | Primary Predictor | predictor | Standard Meteorological Units | **YES** | Open-Meteo ERA5 Reanalysis (Retrospective Benchmark) | Atmospheric thermodynamic/vector parameter conditioned on observations <= t |
| 127 | `wind_speed_roll_mean_24h` | Primary Predictor | predictor | Standard Meteorological Units | **YES** | Open-Meteo ERA5 Reanalysis (Retrospective Benchmark) | Atmospheric thermodynamic/vector parameter conditioned on observations <= t |
| 128 | `pressure_lag_1` | Primary Predictor | predictor | Standard Meteorological Units | **YES** | Open-Meteo ERA5 Reanalysis (Retrospective Benchmark) | Atmospheric thermodynamic/vector parameter conditioned on observations <= t |
| 129 | `pressure_tendency_3h` | Primary Predictor | predictor | Standard Meteorological Units | **YES** | Open-Meteo ERA5 Reanalysis (Retrospective Benchmark) | Atmospheric thermodynamic/vector parameter conditioned on observations <= t |
| 130 | `osm_primary_length_1km` | Exploratory Road Proxy | predictor (exploratory proxy) | km, km/km2, or Ratio | NO | Literature Spatial Proxy (Assumed, Unverified) | Static spatial road length / density / highway distance (4 spatial DOF across network) |
| 131 | `osm_secondary_length_1km` | Exploratory Road Proxy | predictor (exploratory proxy) | km, km/km2, or Ratio | NO | Literature Spatial Proxy (Assumed, Unverified) | Static spatial road length / density / highway distance (4 spatial DOF across network) |
| 132 | `osm_total_length_1km` | Exploratory Road Proxy | predictor (exploratory proxy) | km, km/km2, or Ratio | NO | Literature Spatial Proxy (Assumed, Unverified) | Static spatial road length / density / highway distance (4 spatial DOF across network) |
| 133 | `osm_primary_length_3km` | Exploratory Road Proxy | predictor (exploratory proxy) | km, km/km2, or Ratio | NO | Literature Spatial Proxy (Assumed, Unverified) | Static spatial road length / density / highway distance (4 spatial DOF across network) |
| 134 | `osm_secondary_length_3km` | Exploratory Road Proxy | predictor (exploratory proxy) | km, km/km2, or Ratio | NO | Literature Spatial Proxy (Assumed, Unverified) | Static spatial road length / density / highway distance (4 spatial DOF across network) |
| 135 | `osm_total_length_3km` | Exploratory Road Proxy | predictor (exploratory proxy) | km, km/km2, or Ratio | NO | Literature Spatial Proxy (Assumed, Unverified) | Static spatial road length / density / highway distance (4 spatial DOF across network) |
| 136 | `osm_road_density_1km` | Exploratory Road Proxy | predictor (exploratory proxy) | km, km/km2, or Ratio | NO | Literature Spatial Proxy (Assumed, Unverified) | Static spatial road length / density / highway distance (4 spatial DOF across network) |
| 137 | `osm_road_density_3km` | Exploratory Road Proxy | predictor (exploratory proxy) | km, km/km2, or Ratio | NO | Literature Spatial Proxy (Assumed, Unverified) | Static spatial road length / density / highway distance (4 spatial DOF across network) |
| 138 | `osm_primary_density_1km` | Exploratory Road Proxy | predictor (exploratory proxy) | km, km/km2, or Ratio | NO | Literature Spatial Proxy (Assumed, Unverified) | Static spatial road length / density / highway distance (4 spatial DOF across network) |
| 139 | `osm_primary_density_3km` | Exploratory Road Proxy | predictor (exploratory proxy) | km, km/km2, or Ratio | NO | Literature Spatial Proxy (Assumed, Unverified) | Static spatial road length / density / highway distance (4 spatial DOF across network) |
| 140 | `osm_primary_concentration_1km` | Exploratory Road Proxy | predictor (exploratory proxy) | km, km/km2, or Ratio | NO | Literature Spatial Proxy (Assumed, Unverified) | Static spatial road length / density / highway distance (4 spatial DOF across network) |
| 141 | `osm_primary_concentration_3km` | Exploratory Road Proxy | predictor (exploratory proxy) | km, km/km2, or Ratio | NO | Literature Spatial Proxy (Assumed, Unverified) | Static spatial road length / density / highway distance (4 spatial DOF across network) |
| 142 | `osm_distance_to_highway_km` | Exploratory Road Proxy | predictor (exploratory proxy) | km, km/km2, or Ratio | NO | Literature Spatial Proxy (Assumed, Unverified) | Static spatial road length / density / highway distance (4 spatial DOF across network) |
| 143 | `osm_intersection_density_km2` | Exploratory Road Proxy | predictor (exploratory proxy) | km, km/km2, or Ratio | NO | Literature Spatial Proxy (Assumed, Unverified) | Static spatial road length / density / highway distance (4 spatial DOF across network) |
| 144 | `is_morning_peak` | Primary Predictor | predictor | Binary (0/1) | **YES** | Calendar & Clock Hour (Purely Temporal) | Diurnal traffic congestion window (NO road metric multiplication) |
| 145 | `is_evening_peak` | Primary Predictor | predictor | Binary (0/1) | **YES** | Calendar & Clock Hour (Purely Temporal) | Diurnal traffic congestion window (NO road metric multiplication) |
| 146 | `is_traffic_peak` | Primary Predictor | predictor | Binary (0/1) | **YES** | Calendar & Clock Hour (Purely Temporal) | Diurnal traffic congestion window (NO road metric multiplication) |
| 147 | `is_weekend` | Primary Predictor | predictor | Binary (0/1) | **YES** | Calendar & Clock Hour (Purely Temporal) | Diurnal traffic congestion window (NO road metric multiplication) |
| 148 | `is_weekday_peak` | Primary Predictor | predictor | Binary (0/1) | **YES** | Calendar & Clock Hour (Purely Temporal) | Diurnal traffic congestion window (NO road metric multiplication) |
| 149 | `is_freight_window_exploratory` | Primary Predictor | predictor | Binary (0/1) | **YES** | Calendar & Clock Hour (Purely Temporal) | Diurnal traffic congestion window (NO road metric multiplication) |
| 150 | `is_daytime_offpeak` | Primary Predictor | predictor | Binary (0/1) | **YES** | Calendar & Clock Hour (Purely Temporal) | Diurnal traffic congestion window (NO road metric multiplication) |
| 151 | `traffic_pressure_total_1km` | Exploratory Road Proxy | predictor (exploratory proxy) | Proxy Intensity Index | NO | Assumed Road Proxy x Temporal Window | Interaction product of assumed static road proxy and temporal congestion flag |
| 152 | `traffic_pressure_weekday_peak` | Exploratory Road Proxy | predictor (exploratory proxy) | Proxy Intensity Index | NO | Assumed Road Proxy x Temporal Window | Interaction product of assumed static road proxy and temporal congestion flag |
| 153 | `traffic_pressure_primary_1km` | Exploratory Road Proxy | predictor (exploratory proxy) | Proxy Intensity Index | NO | Assumed Road Proxy x Temporal Window | Interaction product of assumed static road proxy and temporal congestion flag |
| 154 | `freight_pressure_proxy` | Exploratory Road Proxy | predictor (exploratory proxy) | Proxy Intensity Index | NO | Assumed Road Proxy x Temporal Window | Interaction product of assumed static road proxy and temporal congestion flag |
| 155 | `highway_rush_pressure` | Exploratory Road Proxy | predictor (exploratory proxy) | Proxy Intensity Index | NO | Assumed Road Proxy x Temporal Window | Interaction product of assumed static road proxy and temporal congestion flag |
| 156 | `partition` | Metadata / Quality | metadata | Category / Binary | NO | Chronological Splitter Protocol | Chronological split assignment or 48-hour boundary purge flag |
| 157 | `is_boundary_buffer` | Metadata / Quality | metadata | Category / Binary | NO | Chronological Splitter Protocol | Chronological split assignment or 48-hour boundary purge flag |
| 158 | `target_pm25_h1` | Prospective Target | target | ug/m3 or Binary (0/1) | NO | CPCB Physical Raw PM2.5 Sensor Telemetry | Forward prospective shift t+h / forward 24h rolling mean / threshold alert |
| 159 | `target_pm25_h6` | Prospective Target | target | ug/m3 or Binary (0/1) | NO | CPCB Physical Raw PM2.5 Sensor Telemetry | Forward prospective shift t+h / forward 24h rolling mean / threshold alert |
| 160 | `target_pm25_h24` | Prospective Target | target | ug/m3 or Binary (0/1) | NO | CPCB Physical Raw PM2.5 Sensor Telemetry | Forward prospective shift t+h / forward 24h rolling mean / threshold alert |
| 161 | `target_pm25_h48` | Prospective Target | target | ug/m3 or Binary (0/1) | NO | CPCB Physical Raw PM2.5 Sensor Telemetry | Forward prospective shift t+h / forward 24h rolling mean / threshold alert |
| 162 | `target_pm25_24h_mean` | Prospective Target | target | ug/m3 or Binary (0/1) | NO | CPCB Physical Raw PM2.5 Sensor Telemetry | Forward prospective shift t+h / forward 24h rolling mean / threshold alert |
| 163 | `target_pm25_24h_valid_count` | Prospective Target | target | ug/m3 or Binary (0/1) | NO | CPCB Physical Raw PM2.5 Sensor Telemetry | Forward prospective shift t+h / forward 24h rolling mean / threshold alert |
| 164 | `target_hotspot_acute_h1` | Prospective Target | target | ug/m3 or Binary (0/1) | NO | CPCB Physical Raw PM2.5 Sensor Telemetry | Forward prospective shift t+h / forward 24h rolling mean / threshold alert |
| 165 | `target_hotspot_spike_h1` | Prospective Target | target | ug/m3 or Binary (0/1) | NO | CPCB Physical Raw PM2.5 Sensor Telemetry | Forward prospective shift t+h / forward 24h rolling mean / threshold alert |
| 166 | `target_hotspot_very_poor_h1` | Prospective Target | target | ug/m3 or Binary (0/1) | NO | CPCB Physical Raw PM2.5 Sensor Telemetry | Forward prospective shift t+h / forward 24h rolling mean / threshold alert |
| 167 | `target_hotspot_acute_h6` | Prospective Target | target | ug/m3 or Binary (0/1) | NO | CPCB Physical Raw PM2.5 Sensor Telemetry | Forward prospective shift t+h / forward 24h rolling mean / threshold alert |
| 168 | `target_hotspot_spike_h6` | Prospective Target | target | ug/m3 or Binary (0/1) | NO | CPCB Physical Raw PM2.5 Sensor Telemetry | Forward prospective shift t+h / forward 24h rolling mean / threshold alert |
| 169 | `target_hotspot_very_poor_h6` | Prospective Target | target | ug/m3 or Binary (0/1) | NO | CPCB Physical Raw PM2.5 Sensor Telemetry | Forward prospective shift t+h / forward 24h rolling mean / threshold alert |
| 170 | `target_hotspot_acute_h24` | Prospective Target | target | ug/m3 or Binary (0/1) | NO | CPCB Physical Raw PM2.5 Sensor Telemetry | Forward prospective shift t+h / forward 24h rolling mean / threshold alert |
| 171 | `target_hotspot_spike_h24` | Prospective Target | target | ug/m3 or Binary (0/1) | NO | CPCB Physical Raw PM2.5 Sensor Telemetry | Forward prospective shift t+h / forward 24h rolling mean / threshold alert |
| 172 | `target_hotspot_very_poor_h24` | Prospective Target | target | ug/m3 or Binary (0/1) | NO | CPCB Physical Raw PM2.5 Sensor Telemetry | Forward prospective shift t+h / forward 24h rolling mean / threshold alert |
| 173 | `target_hotspot_24h_statutory` | Prospective Target | target | ug/m3 or Binary (0/1) | NO | CPCB Physical Raw PM2.5 Sensor Telemetry | Forward prospective shift t+h / forward 24h rolling mean / threshold alert |

---

*Feature dictionary generated automatically and synchronized with `feature_store_summary.json`.*