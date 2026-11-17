# Missing-Value and Gap Analysis Report

**Document Code:** `REPORT-M1-MISSINGNESS`  
**Project:** Traffic-Aware Spatio-Temporal PM2.5 Concentration and Hotspot Forecasting in Hyderabad  
**Milestone:** Milestone 1 – Data Foundation and Preprocessing  

---

## 1. Missingness Protocol & Architectural Principles

In compliance with the approved research methodology, missing values in features are strictly decoupled from targets:
1. **Zero Backward Interpolation:** Ordinary bidirectional linear interpolation accesses future values ($t_b > t$), introducing fatal forward-looking data leakage. In prospective features, this has been completely banned.
2. **Short Gaps (<= 3 hours):** Forward-filled strictly from the latest past valid observation ($y_{i, t-\delta}, \delta \ge 1$, limit=3).
3. **Medium Gaps (4 to 24 hours):** Imputed via diurnal monthly/hourly lookup tables $\hat{\mu}_{\text{train}}(m, h)$ calculated strictly from the 2023–2024 training partition.
4. **Extended Outages (> 24 hours):** Retained as `NaN` in the primary preprocessed feature dataset for native gradient boosting tree handling (LightGBM/XGBoost). In the dense dataset, diurnal table values are used.
5. **Target Protection Rule:** Target labels are NEVER imputed. Missing targets evaluate strictly to `NaN` and are NEVER substituted with zero.

---

## 2. Station-Wise Imputation and Gap Distribution

| Station | Pollutant | Total Hours | Original Missing | Short Gaps (<=3h FFill) | Medium Gaps (4-24h Diurnal) | Extended Outages (>24h Retained NaN) | Final Retained NaN (Primary) | Training Mean |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Bollaram Industrial Area | `PM2.5` | 32,856 | 9,194 | 629 | 2,904 | 5,661 | **5,661** | 40.06 |
| Bollaram Industrial Area | `PM10` | 32,856 | 8,637 | 811 | 2,305 | 5,521 | **5,521** | 94.75 |
| Bollaram Industrial Area | `NO` | 32,856 | 7,303 | 302 | 1,736 | 5,265 | **5,265** | 4.87 |
| Bollaram Industrial Area | `NO2` | 32,856 | 7,404 | 334 | 1,810 | 5,260 | **5,260** | 24.45 |
| Bollaram Industrial Area | `NH3` | 32,856 | 7,284 | 246 | 1,751 | 5,287 | **5,287** | 14.91 |
| Bollaram Industrial Area | `SO2` | 32,856 | 7,398 | 326 | 1,782 | 5,290 | **5,290** | 16.59 |
| Bollaram Industrial Area | `CO` | 32,856 | 7,072 | 184 | 1,752 | 5,136 | **5,136** | 0.88 |
| Bollaram Industrial Area | `Ozone` | 32,856 | 7,093 | 185 | 1,698 | 5,210 | **5,210** | 31.05 |
| Central University | `PM2.5` | 32,856 | 3,729 | 645 | 1,789 | 1,295 | **1,295** | 26.48 |
| Central University | `PM10` | 32,856 | 5,188 | 477 | 2,207 | 2,504 | **2,504** | 73.85 |
| Central University | `NO` | 32,856 | 2,655 | 241 | 1,115 | 1,299 | **1,299** | 10.33 |
| Central University | `NO2` | 32,856 | 3,035 | 615 | 1,120 | 1,300 | **1,300** | 31.94 |
| Central University | `NH3` | 32,856 | 3,110 | 615 | 1,127 | 1,368 | **1,368** | 16.65 |
| Central University | `SO2` | 32,856 | 3,093 | 518 | 1,413 | 1,162 | **1,162** | 5.9 |
| Central University | `CO` | 32,856 | 2,457 | 139 | 1,127 | 1,191 | **1,191** | 0.51 |
| Central University | `Ozone` | 32,856 | 2,679 | 225 | 1,316 | 1,138 | **1,138** | 13.22 |
| ECIL Kapra | `PM2.5` | 32,856 | 4,337 | 494 | 2,119 | 1,724 | **1,724** | 32.41 |
| ECIL Kapra | `PM10` | 32,856 | 4,496 | 514 | 2,258 | 1,724 | **1,724** | 75.34 |
| ECIL Kapra | `NO` | 32,856 | 3,066 | 268 | 1,243 | 1,555 | **1,555** | 6.11 |
| ECIL Kapra | `NO2` | 32,856 | 3,048 | 265 | 1,228 | 1,555 | **1,555** | 9.65 |
| ECIL Kapra | `NH3` | 32,856 | 3,130 | 278 | 1,222 | 1,630 | **1,630** | 20.0 |
| ECIL Kapra | `SO2` | 32,856 | 5,138 | 1,244 | 2,180 | 1,714 | **1,714** | 9.92 |
| ECIL Kapra | `CO` | 32,856 | 3,077 | 254 | 1,257 | 1,566 | **1,566** | 0.52 |
| ECIL Kapra | `Ozone` | 32,856 | 3,050 | 266 | 1,287 | 1,497 | **1,497** | 29.31 |
| ICRISAT Patancheru | `PM2.5` | 32,856 | 2,764 | 553 | 791 | 1,420 | **1,420** | 36.53 |
| ICRISAT Patancheru | `PM10` | 32,856 | 2,512 | 436 | 630 | 1,446 | **1,446** | 80.39 |
| ICRISAT Patancheru | `NO` | 32,856 | 2,228 | 518 | 476 | 1,234 | **1,234** | 8.68 |
| ICRISAT Patancheru | `NO2` | 32,856 | 2,904 | 694 | 754 | 1,456 | **1,456** | 13.63 |
| ICRISAT Patancheru | `NH3` | 32,856 | 2,467 | 646 | 548 | 1,273 | **1,273** | 9.51 |
| ICRISAT Patancheru | `SO2` | 32,856 | 2,242 | 375 | 439 | 1,428 | **1,428** | 14.63 |
| ICRISAT Patancheru | `CO` | 32,856 | 1,677 | 137 | 321 | 1,219 | **1,219** | 0.7 |
| ICRISAT Patancheru | `Ozone` | 32,856 | 2,698 | 433 | 698 | 1,567 | **1,567** | 35.42 |

---

## 3. Network PM2.5 Telemetry Recovery Summary

- **Network Total PM2.5 Gaps:** 20,024 missing hours out of 131,424 total station-hours (15.24%).
- **Short Gaps Restored (<= 3h):** 2,752 hours restored prospective-only forward fill.
- **Medium Gaps Diurnal-Imputed (4-24h):** 7,766 hours imputed with training-fitted diurnal means.
- **Extended Sensor Outages Preserved (> 24h):** Exactly **9,506 hours** preserved as `NaN` in primary features to protect model training from structural synthetic bias.

---
*Report generated automatically by MilestoneReportGenerator.*