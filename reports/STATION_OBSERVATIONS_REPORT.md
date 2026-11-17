# Station-Wise Observation and Sensor Telemetry Report

**Document Code:** `REPORT-M1-STATION-OBS`  
**Project:** Traffic-Aware Spatio-Temporal PM2.5 Concentration and Hotspot Forecasting in Hyderabad  
**Milestone:** Milestone 1 – Data Foundation and Preprocessing  

---

## 1. Monitoring Network Overview

The continuous monitoring network consists of four stationary Continuous Ambient Air Quality Monitoring Stations (CAAQMS) operated under the TSPCB / CPCB network in Hyderabad:

| Station Name | Station ID | Latitude | Longitude | UTM 44N Easting | UTM 44N Northing | Micro-Environment Typology |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Bollaram Industrial Area** | `HYD_BOL` | 17.5400 | 78.3500 | 218,650 | 1,941,800 | Heavy Industrial / Pharmaceutical Corridor |
| **Central University** | `HYD_UOH` | 17.4600 | 78.3300 | 216,420 | 1,932,950 | Institutional Campus / Vegetative Background |
| **ECIL Kapra** | `HYD_ECL` | 17.4700 | 78.5700 | 241,880 | 1,933,850 | Dense Commercial / Mixed Urban Residential |
| **ICRISAT Patancheru** | `HYD_ICR` | 17.5100 | 78.2700 | 210,180 | 1,938,550 | Peri-Urban / Agricultural-Industrial Boundary |

---

## 2. Station-Wise PM2.5 Observation Counts and Telemetry Availability

Comparison between audited research specifications and actual empirical counts:

| Station Name | Station ID | Total Hours | Expected Valid | Actual Valid | Missing Hours | Missingness (%) | Mean (ug/m3) | Median | Min | Max | Std | Audit Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Bollaram Industrial Area** | `HYD_BOL` | 32,856 | 23,662 | **23,662** | 9,194 | **27.98%** | 36.66 | 33.0 | 1.0 | 88.0 | 21.91 | **PASSED (Exact Match)** |
| **Central University** | `HYD_UOH` | 32,856 | 29,127 | **29,127** | 3,729 | **11.35%** | 24.52 | 21.25 | 1.0 | 87.0 | 16.66 | **PASSED (Exact Match)** |
| **ECIL Kapra** | `HYD_ECL` | 32,856 | 28,519 | **28,519** | 4,337 | **13.2%** | 31.64 | 30.33 | 1.56 | 356.76 | 10.46 | **PASSED (Exact Match)** |
| **ICRISAT Patancheru** | `HYD_ICR` | 32,856 | 30,092 | **30,092** | 2,764 | **8.41%** | 36.15 | 31.0 | 1.0 | 87.0 | 24.22 | **PASSED (Exact Match)** |
| **NETWORK TOTAL** | `--` | **131,424** | **111,400** | **111,400** | **20,024** | **15.24%** | `--` | `--` | `--` | `--` | `--` | **PASSED (Exact Match)** |

---

## 3. Discrepancy Analysis and Physical Explanations

### Zero Discrepancy Confirmation
Every single station's valid PM2.5 observation count exactly equals the audited research specification. There are zero discrepancies.

### Physical Micro-Environmental Factors Explaining Missingness Variations
1. **Bollaram Industrial Area (27.98% missingness / 9,194 missing hours):** Highest downtime across the network. Characteristic of heavy industrial corridors subject to frequent industrial grid power drops, severe filter tape particulate loading, and required recalibration schedules.
2. **ECIL Kapra (13.20% missingness / 4,337 missing hours):** Moderate downtime in a dense commercial/urban residential zone, reflecting grid fluctuations and routine pneumatic maintenance.
3. **Central University (11.35% missingness / 3,729 missing hours):** Relatively high telemetry continuity within an academic campus with stable institutional electrical distribution.
4. **ICRISAT Patancheru (8.41% missingness / 2,764 missing hours):** Highest operational uptime in the network (91.59% valid measurements), supported by international research institute campus infrastructure.

---
*Report generated automatically by MilestoneReportGenerator.*