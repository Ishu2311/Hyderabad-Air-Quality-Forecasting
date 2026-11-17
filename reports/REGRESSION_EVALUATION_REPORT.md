# Milestone 3 — Continuous PM2.5 Regression Evaluation Report

**Document Code:** `REPORT-M3-REGRESSION-EVALUATION`  
**Project Title:** Traffic-Aware Spatio-Temporal PM2.5 Concentration and Hotspot Forecasting in Hyderabad  
**Milestone:** Milestone 3 – Baseline Benchmarking and Machine Learning Model Development  
**Status:** **AUDITED AND VERIFIED AGAINST ACTUAL TEST RESULTS**  

---

## 1. Network-Level Performance Overview

Models evaluated on the untouched 2026 test partition (130,848 total network station-hours; 26,016 test hours across 4 stations):

| Horizon | Model Name | Valid Samples ($N$) | MAE ($\mu\text{g/m}^3$) | RMSE ($\mu\text{g/m}^3$) | $R^2$ Score | Mean Bias | Directional Accuracy (MDA) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **+1h** | Random Forest | 19048 | 3.05 | 5.59 | 0.899 | +0.06 | 0.568 |
| **+1h** | Xgboost | 19048 | 3.03 | 5.44 | 0.904 | +0.02 | 0.569 |
| **+1h** | Lightgbm | 19048 | 3.04 | 5.49 | 0.903 | +0.02 | 0.567 |
| **+1h** | **Stacking Ensemble** | 19048 | **3.01** | **5.45** | **0.904** | **-0.04** | **0.568** |
| **+6h** | Random Forest | 18940 | 6.07 | 9.13 | 0.729 | +1.13 | 0.541 |
| **+6h** | Xgboost | 18940 | 6.03 | 8.95 | 0.740 | +1.42 | 0.544 |
| **+6h** | Lightgbm | 18940 | 6.00 | 8.93 | 0.741 | +1.32 | 0.541 |
| **+6h** | **Stacking Ensemble** | 18940 | **5.99** | **8.93** | **0.741** | **+1.43** | **0.545** |
| **+24h** | Random Forest | 18610 | 6.84 | 9.94 | 0.674 | +1.40 | 0.564 |
| **+24h** | Xgboost | 18610 | 6.77 | 9.82 | 0.682 | +1.65 | 0.561 |
| **+24h** | **Lightgbm** | 18610 | **6.67** | **9.74** | **0.687** | **+1.54** | **0.561** |
| **+24h** | Stacking Ensemble | 18610 | 6.74 | 9.80 | 0.683 | +1.82 | 0.563 |
| **+48h** | Random Forest | 18411 | 8.19 | 11.26 | 0.576 | +2.43 | 0.557 |
| **+48h** | Xgboost | 18411 | 8.03 | 11.07 | 0.591 | +2.47 | 0.562 |
| **+48h** | **Lightgbm** | 18411 | **7.93** | **10.95** | **0.599** | **+2.52** | **0.566** |
| **+48h** | Stacking Ensemble | 18411 | 8.02 | 11.01 | 0.595 | +2.71 | 0.566 |

---

## 2. Station-by-Station Breakdown on Test Split (2026)

Evaluation broken down across all four Continuous Ambient Air Quality Monitoring Stations (CAAQMS):

### Bollaram Industrial Area
| Horizon | Model | Valid Samples ($N$) | MAE ($\mu\text{g/m}^3$) | RMSE ($\mu\text{g/m}^3$) | $R^2$ Score | Bias |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **+1h** | Persistence | 1866 | 4.45 | 6.35 | 0.929 | -0.15 |
| **+1h** | Random Forest | 1875 | 4.16 | 5.91 | 0.939 | -0.06 |
| **+1h** | Xgboost | 1875 | 4.22 | 5.94 | 0.938 | +0.01 |
| **+1h** | Lightgbm | 1875 | 4.24 | 5.98 | 0.938 | +0.01 |
| **+1h** | **Stacking Ensemble** | 1875 | **4.19** | **5.92** | **0.939** | **+0.12** |
| **+6h** | Persistence | 1823 | 12.13 | 15.54 | 0.572 | -0.52 |
| **+6h** | Random Forest | 1875 | 8.17 | 10.57 | 0.805 | +0.16 |
| **+6h** | Xgboost | 1875 | 8.06 | 10.46 | 0.809 | +0.24 |
| **+6h** | Lightgbm | 1875 | 8.09 | 10.52 | 0.807 | +0.15 |
| **+6h** | **Stacking Ensemble** | 1875 | **8.02** | **10.45** | **0.809** | **+0.80** |
| **+24h** | Persistence | 1715 | 9.48 | 12.46 | 0.720 | -0.06 |
| **+24h** | Random Forest | 1889 | 9.29 | 11.79 | 0.756 | -0.36 |
| **+24h** | Xgboost | 1889 | 9.35 | 11.89 | 0.752 | -0.12 |
| **+24h** | Lightgbm | 1889 | 9.26 | 11.83 | 0.755 | -0.07 |
| **+24h** | **Stacking Ensemble** | 1889 | **9.16** | **11.71** | **0.760** | **+0.54** |
| **+48h** | Persistence | 1701 | 11.56 | 14.59 | 0.616 | -0.21 |
| **+48h** | Random Forest | 1908 | 10.84 | 13.43 | 0.679 | -1.15 |
| **+48h** | Xgboost | 1908 | 10.80 | 13.40 | 0.681 | -1.03 |
| **+48h** | Lightgbm | 1908 | 10.64 | 13.21 | 0.690 | -0.94 |
| **+48h** | **Stacking Ensemble** | 1908 | **10.63** | **13.19** | **0.691** | **-0.75** |

### Central University
| Horizon | Model | Valid Samples ($N$) | MAE ($\mu\text{g/m}^3$) | RMSE ($\mu\text{g/m}^3$) | $R^2$ Score | Bias |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **+1h** | Persistence | 5735 | 2.70 | 3.94 | 0.932 | +0.03 |
| **+1h** | Random Forest | 5743 | 2.61 | 3.73 | 0.939 | +0.03 |
| **+1h** | Xgboost | 5743 | 2.56 | 3.63 | 0.942 | -0.09 |
| **+1h** | Lightgbm | 5743 | 2.55 | 3.64 | 0.941 | -0.04 |
| **+1h** | **Stacking Ensemble** | 5743 | **2.56** | **3.64** | **0.941** | **-0.22** |
| **+6h** | Persistence | 5705 | 6.22 | 8.75 | 0.663 | +0.09 |
| **+6h** | Random Forest | 5748 | 5.30 | 7.05 | 0.780 | +1.84 |
| **+6h** | Xgboost | 5748 | 5.39 | 7.04 | 0.781 | +2.43 |
| **+6h** | Lightgbm | 5748 | 5.31 | 6.98 | 0.785 | +2.34 |
| **+6h** | **Stacking Ensemble** | 5748 | **5.25** | **6.94** | **0.787** | **+2.16** |
| **+24h** | Persistence | 5585 | 5.65 | 7.87 | 0.722 | +0.19 |
| **+24h** | Random Forest | 5750 | 6.12 | 8.20 | 0.692 | +2.43 |
| **+24h** | Xgboost | 5750 | 6.11 | 8.18 | 0.693 | +2.60 |
| **+24h** | Lightgbm | 5750 | 5.98 | 8.07 | 0.701 | +2.62 |
| **+24h** | **Stacking Ensemble** | 5750 | **6.02** | **8.13** | **0.697** | **+2.73** |
| **+48h** | Persistence | 5484 | 6.76 | 9.35 | 0.592 | +0.28 |
| **+48h** | Random Forest | 5750 | 8.20 | 10.51 | 0.468 | +4.61 |
| **+48h** | Xgboost | 5750 | 7.65 | 9.87 | 0.531 | +4.00 |
| **+48h** | Lightgbm | 5750 | 7.60 | 9.80 | 0.538 | +4.24 |
| **+48h** | **Stacking Ensemble** | 5750 | **7.81** | **10.01** | **0.518** | **+4.54** |

### ECIL Kapra
| Horizon | Model | Valid Samples ($N$) | MAE ($\mu\text{g/m}^3$) | RMSE ($\mu\text{g/m}^3$) | $R^2$ Score | Bias |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **+1h** | Persistence | 5849 | 2.07 | 6.74 | 0.504 | +0.03 |
| **+1h** | Random Forest | 5849 | 2.17 | 5.77 | 0.636 | +0.19 |
| **+1h** | Xgboost | 5849 | 2.21 | 5.52 | 0.667 | +0.16 |
| **+1h** | Lightgbm | 5849 | 2.23 | 5.62 | 0.654 | +0.20 |
| **+1h** | **Stacking Ensemble** | 5849 | **2.18** | **5.56** | **0.661** | **+0.15** |
| **+6h** | Persistence | 5849 | 4.79 | 9.56 | -0.001 | +0.21 |
| **+6h** | Random Forest | 5849 | 4.64 | 8.22 | 0.262 | +1.04 |
| **+6h** | Xgboost | 5849 | 4.65 | 8.07 | 0.287 | +1.15 |
| **+6h** | Lightgbm | 5849 | 4.67 | 8.09 | 0.284 | +1.13 |
| **+6h** | **Stacking Ensemble** | 5849 | **4.72** | **8.15** | **0.274** | **+1.34** |
| **+24h** | Persistence | 5854 | 5.92 | 10.45 | -0.195 | +0.16 |
| **+24h** | Random Forest | 5854 | 5.49 | 8.74 | 0.164 | +1.60 |
| **+24h** | Xgboost | 5854 | 5.42 | 8.56 | 0.198 | +1.85 |
| **+24h** | Lightgbm | 5854 | 5.31 | 8.48 | 0.213 | +1.63 |
| **+24h** | **Stacking Ensemble** | 5854 | **5.46** | **8.61** | **0.190** | **+2.00** |
| **+48h** | Persistence | 5855 | 6.42 | 10.82 | -0.280 | +0.13 |
| **+48h** | Random Forest | 5855 | 6.21 | 9.26 | 0.062 | +2.32 |
| **+48h** | Xgboost | 5855 | 6.13 | 9.16 | 0.082 | +2.32 |
| **+48h** | Lightgbm | 5855 | 6.05 | 9.05 | 0.103 | +2.46 |
| **+48h** | **Stacking Ensemble** | 5855 | **6.13** | **9.10** | **0.093** | **+2.64** |

### ICRISAT Patancheru
| Horizon | Model | Valid Samples ($N$) | MAE ($\mu\text{g/m}^3$) | RMSE ($\mu\text{g/m}^3$) | $R^2$ Score | Bias |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **+1h** | Persistence | 5598 | 4.39 | 7.90 | 0.840 | +0.02 |
| **+1h** | Random Forest | 5606 | 4.14 | 6.95 | 0.877 | +0.00 |
| **+1h** | Xgboost | 5606 | 4.06 | 6.82 | 0.881 | -0.02 |
| **+1h** | Lightgbm | 5606 | 4.06 | 6.83 | 0.881 | -0.10 |
| **+1h** | **Stacking Ensemble** | 5606 | **4.03** | **6.81** | **0.882** | **-0.12** |
| **+6h** | Persistence | 5563 | 10.09 | 15.00 | 0.415 | -0.00 |
| **+6h** | Random Forest | 5606 | 7.96 | 11.62 | 0.654 | +0.95 |
| **+6h** | Xgboost | 5606 | 7.79 | 11.31 | 0.672 | +1.21 |
| **+6h** | Lightgbm | 5606 | 7.73 | 11.26 | 0.675 | +1.01 |
| **+6h** | **Stacking Ensemble** | 5606 | **7.73** | **11.28** | **0.673** | **+1.14** |
| **+24h** | Persistence | 5456 | 9.40 | 14.73 | 0.413 | -0.09 |
| **+24h** | Random Forest | 5606 | 8.99 | 12.80 | 0.573 | +1.24 |
| **+24h** | Xgboost | 5606 | 8.90 | 12.71 | 0.579 | +1.60 |
| **+24h** | Lightgbm | 5606 | 8.84 | 12.70 | 0.580 | +1.41 |
| **+24h** | **Stacking Ensemble** | 5606 | **8.92** | **12.73** | **0.578** | **+1.66** |
| **+48h** | Persistence | 5371 | 10.71 | 15.85 | 0.313 | +0.06 |
| **+48h** | Random Forest | 5606 | 10.30 | 14.03 | 0.483 | +2.19 |
| **+48h** | Xgboost | 5606 | 10.47 | 14.12 | 0.477 | +2.97 |
| **+48h** | Lightgbm | 5606 | 10.36 | 13.98 | 0.487 | +2.72 |
| **+48h** | **Stacking Ensemble** | 5606 | **10.35** | **13.97** | **0.488** | **+2.79** |

---

## 3. Bollaram 2026 Sensor Outage Audit
- **Audit Finding:** In the 2026 test partition, Bollaram Industrial Area experienced a severe hardware sensor outage, yielding only **1,875 to 1,908 valid target observations (28.8% to 29.3% coverage)** compared to 5,600–5,850 observations (>86% coverage) at Central University, ECIL Kapra, and ICRISAT Patancheru.
- **Methodological Compliance:** In strict adherence to research directives, **unobserved observations were NOT imputed or replaced with 0.0**. Error metrics at Bollaram were computed strictly on the 1,875 observed hours. The reduced coverage is transparently declared.
