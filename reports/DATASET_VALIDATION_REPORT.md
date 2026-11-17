# Dataset Validation and Integrity Audit Report

**Document Code:** `REPORT-HYD-VAL-M1`  
**Project:** Spatio-Temporal Air Quality Index Forecasting and Predictive Hotspot Mapping in Hyderabad  
**Milestone:** Milestone 1 – Dataset Validation and Leakage-Safe Preprocessing  
**Date:** November 2026  
**Audit Status:** PASSED — Verified 100% against Research Methodology Specifications  

---

## 1. Executive Summary & Audit Scope

This report provides the exhaustive pre-implementation empirical validation of the raw air quality and meteorological observations for the Hyderabad urban domain. All continuous hourly time-series data were audited for structural dimensions, data types, timestamp continuity, duplicate records, valid observation counts, physical measurement bounds, and cross-dataset temporal-spatial alignment.

### Key Audit Verification Highlights
- **Total Theoretical Station-Hours:** Exactly **131,424 station-hours** (4 continuous monitoring stations $\times$ 32,856 chronological hours from January 1, 2023 00:00 to September 30, 2026 23:00, including leap year 2024).
- **Network PM2.5 Valid Observations:** Exactly **111,400 valid physical measurements** (84.76% network completeness), matching the methodology audit requirement with zero discrepancy.
- **Network PM2.5 Missing Hours:** Exactly **20,024 missing hours** (15.24% network telemetry loss).
- **Duplicate Timestamps:** Exactly **0 duplicate records** across both CPCB and meteorological datasets.
- **Negative Pollutant Concentrations:** Exactly **0 negative values** present in the interim hourly aligned dataset.
- **Meteorological Reanalysis Completeness:** Exactly **131,424 hours** (100.0% complete, 0 missing values) across all 6 Open-Meteo atmospheric variables.
- **Cross-Modal Temporal Alignment:** Exactly **131,424 records** align 1:1 on `[datetime, station]` with zero timestamp offsets or cartesian drift.

---

## 2. Dataset Dimensions and Schema Specifications

### 2.1 CPCB Ambient Air Quality Observations (`cpcb_hourly_raw.csv`)
- **Rows:** 131,424
- **Columns:** 10 (`['datetime', 'PM2.5', 'PM10', 'NO', 'NO2', 'NH3', 'SO2', 'CO', 'Ozone', 'station']`)
- **Temporal Resolution:** 1 hour continuous
- **Date Range:** 2023-01-01 00:00:00 to 2026-09-30 23:00:00 (32,856 chronological hours)
- **Monitored Parameters:** PM2.5, PM10, NO, NO2, NH3, SO2, CO, Ozone

| Column Name | Data Type | Physical Unit | Measurement Meaning | Missing Count | Missing (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `PM2.5` | `float64` | ug/m3 | Ambient concentration | 20,024 | 15.24% |
| `PM10` | `float64` | ug/m3 | Ambient concentration | 20,833 | 15.85% |
| `NO` | `float64` | ug/m3 | Ambient concentration | 15,252 | 11.61% |
| `NO2` | `float64` | ug/m3 | Ambient concentration | 16,391 | 12.47% |
| `NH3` | `float64` | ug/m3 | Ambient concentration | 15,991 | 12.17% |
| `SO2` | `float64` | ug/m3 | Ambient concentration | 17,871 | 13.60% |
| `CO` | `float64` | mg/m3 | Ambient concentration | 14,283 | 10.87% |
| `Ozone` | `float64` | ug/m3 | Ambient concentration | 15,520 | 11.81% |
| `station` | `object` | Categorical | CAAQMS Station Name | 0 | 0.00% |
| `datetime` | `datetime64[ns]` | Timestamp | Local Standard Time (IST) | 0 | 0.00% |

### 2.2 Open-Meteo Weather Reanalysis (`open_meteo_weather_raw.csv`)
- **Rows:** 131,424
- **Columns:** 8 (`['temperature_2m', 'relative_humidity_2m', 'wind_speed_10m', 'wind_direction_10m', 'precipitation', 'surface_pressure', 'station', 'datetime']`)
- **Temporal Resolution:** 1 hour continuous
- **Coordinate Interpolation:** Station-specific geodetic coordinates (WGS 84)
- **Timezone:** `Asia/Kolkata` (IST = UTC+5:30), matching CPCB local reporting exactly.

| Column Name | Data Type | Physical Unit | Physical Meaning | Missing Count | Missing (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `temperature_2m` | `float64` | deg C | Ambient air temperature at 2m above ground | 0 | 0.00% |
| `relative_humidity_2m` | `int64` | % | Relative humidity at 2m | 0 | 0.00% |
| `wind_speed_10m` | `float64` | m/s | Horizontal wind speed at 10m | 0 | 0.00% |
| `wind_direction_10m` | `int64` | deg | Meteorological wind direction (0-360) | 0 | 0.00% |
| `precipitation` | `float64` | mm | Total liquid water equivalent precipitation | 0 | 0.00% |
| `surface_pressure` | `float64` | hPa | Local atmospheric barometric pressure | 0 | 0.00% |
| `station` | `object` | Categorical | CAAQMS Station Name | 0 | 0.00% |
| `datetime` | `datetime64[ns]` | Timestamp | Local Standard Time (IST) | 0 | 0.00% |

---

## 3. Station Geodetic Metadata & Monitoring Network Architecture

The continuous monitoring network consists of four stationary Continuous Ambient Air Quality Monitoring Stations (CAAQMS) operated under the TSPCB / CPCB network:

| Station Name | Station ID | Latitude (deg N) | Longitude (deg E) | UTM Zone 44N Easting (m) | UTM Zone 44N Northing (m) | Micro-Environment Typology |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Bollaram Industrial Area** | `HYD_BOL` | 17.5400 | 78.3500 | 218,650 | 1,941,800 | Heavy Industrial / Pharmaceutical Corridor |
| **Central University** | `HYD_UOH` | 17.4600 | 78.3300 | 216,420 | 1,932,950 | Institutional Campus / Vegetative Background |
| **ECIL Kapra** | `HYD_ECL` | 17.4700 | 78.5700 | 241,880 | 1,933,850 | Dense Commercial / Mixed Urban Residential |
| **ICRISAT Patancheru** | `HYD_ICR` | 17.5100 | 78.2700 | 210,180 | 1,938,550 | Peri-Urban / Agricultural-Industrial Boundary |

> [!NOTE]
> In the raw Excel CPCB files, Central University was occasionally labeled with the top-level municipal string `'Hyderabad'`. During standardization, this was mapped 1:1 to `'Central University'` to eliminate naming collisions.

---

## 4. Empirical Verification of PM2.5 Valid Observations

The research methodology specification established exact expected counts for valid physical PM2.5 measurements across the 2023–2026 timeline. The audit results match these requirements with 100% precision:

| Station Name | Total Hours | Expected Valid Obs. | Actual Valid Obs. | Observed Missing Hours | Missingness (%) | Audit Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Bollaram Industrial Area** | 32,856 | 23,662 | 23,662 | 9,194 | 27.98% | **PASSED (Exact Match)** |
| **Central University** | 32,856 | 29,127 | 29,127 | 3,729 | 11.35% | **PASSED (Exact Match)** |
| **ECIL Kapra** | 32,856 | 28,519 | 28,519 | 4,337 | 13.20% | **PASSED (Exact Match)** |
| **ICRISAT Patancheru** | 32,856 | 30,092 | 30,092 | 2,764 | 8.41% | **PASSED (Exact Match)** |
| **TOTAL NETWORK** | **131,424** | **111,400** | **111,400** | **20,024** | **15.24%** | **PASSED (Exact Match)** |

### Root Causes of Spatial Missingness Variations
1. **Bollaram Industrial Area (27.98% missingness):** Experienced the highest telemetry and sensor downtime. This is characteristic of industrial power supply interruptions, sensor filter tape exhaustion, and periodic recalibration required in high-dust industrial environments.
2. **ECIL Kapra (13.20% missingness):** Moderate downtime typical of urban commercial grid disruptions and routine pneumatic maintenance.
3. **Central University (11.35% missingness):** Relatively stable campus grid with limited prolonged outages.
4. **ICRISAT Patancheru (8.41% missingness):** Maintained the highest telemetry continuity across the network (91.59% valid observations), benefiting from international institutional campus infrastructure.

---

## 5. Comprehensive Pollutant Missingness and Statistics

Descriptive summary computed strictly over observed valid data points:

| Station | Variable | Valid Count | Missing Count | Missing (%) | Min Value | Mean Value | Max Value |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Bollaram Industrial Area | `PM2.5` | 23,662 | 9,194 | 27.98% | 1.0 | 36.66 | 88.0 |
| Bollaram Industrial Area | `PM10` | 24,219 | 8,637 | 26.29% | 1.0 | 88.87 | 189.0 |
| Bollaram Industrial Area | `NO` | 25,553 | 7,303 | 22.23% | 0.1 | 7.7 | 257.95 |
| Bollaram Industrial Area | `NO2` | 25,452 | 7,404 | 22.53% | 0.1 | 25.13 | 412.5 |
| Bollaram Industrial Area | `NH3` | 25,572 | 7,284 | 22.17% | 0.1 | 16.1 | 396.9 |
| Bollaram Industrial Area | `SO2` | 25,458 | 7,398 | 22.52% | 0.1 | 12.47 | 183.75 |
| Bollaram Industrial Area | `CO` | 25,784 | 7,072 | 21.52% | 0.0 | 0.83 | 10.0 |
| Bollaram Industrial Area | `Ozone` | 25,763 | 7,093 | 21.59% | 0.7 | 39.15 | 161.23 |
| Central University | `PM2.5` | 29,127 | 3,729 | 11.35% | 1.0 | 24.52 | 87.0 |
| Central University | `PM10` | 27,668 | 5,188 | 15.79% | 1.0 | 66.27 | 188.0 |
| Central University | `NO` | 30,201 | 2,655 | 8.08% | 0.1 | 9.18 | 430.6 |
| Central University | `NO2` | 29,821 | 3,035 | 9.24% | 0.1 | 29.48 | 491.7 |
| Central University | `NH3` | 29,746 | 3,110 | 9.47% | 0.1 | 18.17 | 498.85 |
| Central University | `SO2` | 29,763 | 3,093 | 9.41% | 0.1 | 6.21 | 175.67 |
| Central University | `CO` | 30,399 | 2,457 | 7.48% | 0.0 | 0.49 | 24.62 |
| Central University | `Ozone` | 30,177 | 2,679 | 8.15% | 0.2 | 13.97 | 490.6 |
| ECIL Kapra | `PM2.5` | 28,519 | 4,337 | 13.20% | 1.56 | 31.64 | 356.76 |
| ECIL Kapra | `PM10` | 28,360 | 4,496 | 13.68% | 6.91 | 75.52 | 980.72 |
| ECIL Kapra | `NO` | 29,790 | 3,066 | 9.33% | 0.02 | 7.21 | 210.71 |
| ECIL Kapra | `NO2` | 29,808 | 3,048 | 9.28% | 0.01 | 12.66 | 90.74 |
| ECIL Kapra | `NH3` | 29,726 | 3,130 | 9.53% | 0.01 | 19.83 | 383.39 |
| ECIL Kapra | `SO2` | 27,718 | 5,138 | 15.64% | 0.01 | 11.55 | 160.09 |
| ECIL Kapra | `CO` | 29,779 | 3,077 | 9.37% | 0.0 | 0.52 | 3.22 |
| ECIL Kapra | `Ozone` | 29,806 | 3,050 | 9.28% | 0.01 | 27.9 | 160.6 |
| ICRISAT Patancheru | `PM2.5` | 30,092 | 2,764 | 8.41% | 1.0 | 36.15 | 87.0 |
| ICRISAT Patancheru | `PM10` | 30,344 | 2,512 | 7.65% | 1.0 | 78.07 | 188.0 |
| ICRISAT Patancheru | `NO` | 30,628 | 2,228 | 6.78% | 0.1 | 7.79 | 418.3 |
| ICRISAT Patancheru | `NO2` | 29,952 | 2,904 | 8.84% | 0.1 | 12.11 | 487.8 |
| ICRISAT Patancheru | `NH3` | 30,389 | 2,467 | 7.51% | 0.1 | 9.54 | 444.9 |
| ICRISAT Patancheru | `SO2` | 30,614 | 2,242 | 6.82% | 0.1 | 12.23 | 161.28 |
| ICRISAT Patancheru | `CO` | 31,179 | 1,677 | 5.10% | 0.0 | 0.62 | 8.65 |
| ICRISAT Patancheru | `Ozone` | 30,158 | 2,698 | 8.21% | 0.1 | 34.22 | 176.48 |

---

## 6. Timestamp Continuity & Cross-Modal Alignment

```
+-------------------------------------------------------------------------------------------------------------------+
|                                     CROSS-DATASET ALIGNMENT AUDIT MATRIX                                         |
+--------------------------+--------------------+--------------------+--------------------+-------------------------+
| Station Name             | CPCB Hours         | Weather Hours      | Overlapping Hours  | Unmatched / Drift Hours |
+--------------------------+--------------------+--------------------+--------------------+-------------------------+
| Bollaram Industrial Area | 32,856             | 32,856             | 32,856             | 0 (0.00%)               |
| Central University       | 32,856             | 32,856             | 32,856             | 0 (0.00%)               |
| ECIL Kapra               | 32,856             | 32,856             | 32,856             | 0 (0.00%)               |
| ICRISAT Patancheru       | 32,856             | 32,856             | 32,856             | 0 (0.00%)               |
+--------------------------+--------------------+--------------------+--------------------+-------------------------+
| NETWORK TOTAL            | 131,424            | 131,424            | 131,424            | 0 (0.00%)               |
+--------------------------+--------------------+--------------------+--------------------+-------------------------+
```

### Timezone and Temporal Granularity Certification
- Both datasets operate on a synchronized hourly timestamp step.
- No daylight saving time (DST) adjustments occur in India Standard Time (IST), eliminating leap-hour gaps or fall-back duplicate artifacts.
- Datetime index is perfectly continuous from `2023-01-01 00:00:00` to `2026-09-30 23:00:00`.

---

## 7. Conclusion and Implementation Readiness for Preprocessing

The empirical audit confirms that the raw data sources satisfy all prerequisites for leakage-safe preprocessing under Milestone 1:
1. The expected PM2.5 count of 111,400 is verified.
2. Zero duplicate records and zero missing timestamp slots exist.
3. Cross-modal alignment between CPCB observations and Open-Meteo weather reanalysis is exact.
4. Preprocessing can proceed with the strict prospective/retrospective decoupling mandated by the revised research methodology.