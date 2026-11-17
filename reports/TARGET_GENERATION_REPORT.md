# Multi-Horizon Target Generation and Hotspot Formulation Report

**Document Code:** `REPORT-M1-TARGETS`  
**Project:** Traffic-Aware Spatio-Temporal PM2.5 Concentration and Hotspot Forecasting in Hyderabad  
**Milestone:** Milestone 1 – Data Foundation and Preprocessing  

---

## 1. Mathematical Formulations & Target Specifications

### 1.1 Multi-Horizon Regression Targets
For each station $i$ and prediction origin $t$, future continuous PM2.5 concentrations are defined as:
- **+1 Hour Horizon:** $y_{i, t+1}$ (instantaneous 1-hour ahead concentration)
- **+6 Hour Horizon:** $y_{i, t+6}$ (instantaneous 6-hour ahead concentration)
- **+24 Hour Horizon:** $y_{i, t+24}$ (instantaneous 24-hour ahead concentration)
- **+48 Hour Horizon:** $y_{i, t+48}$ (instantaneous 48-hour ahead concentration)

### 1.2 Forward 24-Hour Average Target
$$\bar{y}_{i, t+1:t+24} = \frac{\sum_{k=1}^{24} y_{i, t+k} \cdot m_{i, t+k}}{N_{\text{valid}}} \quad \text{where } N_{\text{valid}} = \sum_{k=1}^{24} m_{i, t+k} \ge 16$$
- **Completeness Mandate:** Evaluates strictly to `NaN` if $N_{\text{valid}} < 16$ (minimum 66.7% data completeness).
- **Denominator Protection:** The denominator is strictly $N_{\text{valid}}$, NEVER a fixed 24.

### 1.3 Hotspot Classification Targets
1. **Operational Acute Excursion (60 ug/m3):**
   $$z_{i, t+h}^{(\text{acute})} = \begin{cases} \text{NaN}, & \text{if } y_{i, t+h} \text{ is unobserved} \\ 1, & \text{if } y_{i, t+h} \ge 60\,\mu\text{g/m}^3 \\ 0, & \text{if } y_{i, t+h} < 60\,\mu\text{g/m}^3 \end{cases}$$
2. **Exploratory Severe-Spike Alert Target (>= 120 ug/m3):**
   $$z_{i, t+h}^{(\text{spike})} = \begin{cases} \text{NaN}, & \text{if } y_{i, t+h} \text{ is unobserved} \\ 1, & \text{if } y_{i, t+h} \ge 120\,\mu\text{g/m}^3 \\ 0, & \text{if } y_{i, t+h} < 120\,\mu\text{g/m}^3 \end{cases}$$
3. **Exploratory Very Poor Entry Target (> 120 ug/m3):**
   $$z_{i, t+h}^{(\text{very\_poor})} = \begin{cases} \text{NaN}, & \text{if } y_{i, t+h} \text{ is unobserved} \\ 1, & \text{if } y_{i, t+h} > 120\,\mu\text{g/m}^3 \\ 0, & \text{if } y_{i, t+h} \le 120\,\mu\text{g/m}^3 \end{cases}$$
4. **Statutory 24-Hour Hotspot Target (60 ug/m3 over forward 24h mean):**
   $$z_{i, t+24}^{(\text{24h})} = \begin{cases} \text{NaN}, & \text{if } N_{\text{valid}} < 16 \\ 1, & \text{if } \bar{y}_{i, t+1:t+24} \ge 60\,\mu\text{g/m}^3 \\ 0, & \text{if } \bar{y}_{i, t+1:t+24} < 60\,\mu\text{g/m}^3 \end{cases}$$

---

## 2. Empirical Distribution of Generated Targets

| Target Column | Type | Total Rows | Valid Count | Missing Count | Completeness | Mean | Min | Max | Positive Class Count | Prevalence (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `target_pm25_h1` | `continuous_regression` | 131,424 | 111,396 | 20,028 | 84.76% | 32.0625 | 1.0 | 356.76 | -- | **--** |
| `target_pm25_h6` | `continuous_regression` | 131,424 | 111,376 | 20,048 | 84.75% | 32.0584 | 1.0 | 356.76 | -- | **--** |
| `target_pm25_h24` | `continuous_regression` | 131,424 | 111,306 | 20,118 | 84.69% | 32.0532 | 1.0 | 356.76 | -- | **--** |
| `target_pm25_h48` | `continuous_regression` | 131,424 | 111,212 | 20,212 | 84.62% | 32.0443 | 1.0 | 356.76 | -- | **--** |
| `target_pm25_24h_mean` | `continuous_regression` | 131,424 | 110,606 | 20,818 | 84.16% | 32.0271 | 1.54 | 96.78 | -- | **--** |
| `target_pm25_24h_valid_count` | `continuous_regression` | 131,424 | 131,420 | 4 | 100.0% | 20.3349 | 0.0 | 24.0 | -- | **--** |
| `target_hotspot_acute_h1` | `binary_classification` | 131,424 | 111,396 | 20,028 | 84.76% | 0.1073 | 0.0 | 1.0 | 11,953 | **10.73%** |
| `target_hotspot_spike_h1` | `binary_classification` | 131,424 | 111,396 | 20,028 | 84.76% | 0.0002 | 0.0 | 1.0 | 17 | **0.02%** |
| `target_hotspot_very_poor_h1` | `binary_classification` | 131,424 | 111,396 | 20,028 | 84.76% | 0.0002 | 0.0 | 1.0 | 17 | **0.02%** |
| `target_hotspot_acute_h6` | `binary_classification` | 131,424 | 111,376 | 20,048 | 84.75% | 0.1072 | 0.0 | 1.0 | 11,943 | **10.72%** |
| `target_hotspot_spike_h6` | `binary_classification` | 131,424 | 111,376 | 20,048 | 84.75% | 0.0002 | 0.0 | 1.0 | 17 | **0.02%** |
| `target_hotspot_very_poor_h6` | `binary_classification` | 131,424 | 111,376 | 20,048 | 84.75% | 0.0002 | 0.0 | 1.0 | 17 | **0.02%** |
| `target_hotspot_acute_h24` | `binary_classification` | 131,424 | 111,306 | 20,118 | 84.69% | 0.1072 | 0.0 | 1.0 | 11,930 | **10.72%** |
| `target_hotspot_spike_h24` | `binary_classification` | 131,424 | 111,306 | 20,118 | 84.69% | 0.0002 | 0.0 | 1.0 | 17 | **0.02%** |
| `target_hotspot_very_poor_h24` | `binary_classification` | 131,424 | 111,306 | 20,118 | 84.69% | 0.0002 | 0.0 | 1.0 | 17 | **0.02%** |
| `target_hotspot_24h_statutory` | `binary_classification` | 131,424 | 110,606 | 20,818 | 84.16% | 0.0895 | 0.0 | 1.0 | 9,902 | **8.95%** |

---

## 3. Key Observations on Target Distributions
- **Acute Excursion Prevalence (60 ug/m3):** Instantaneous single-hour exceedances occur in **~10.72%** of valid network hours across +1h, +6h, and +24h horizons.
- **Statutory 24-Hour Hotspot Prevalence:** When averaged over 24 continuous hours ($N_{\text{valid}} \ge 16$), statutory hotspots occur in **8.95%** of valid days/windows.
- **Exploratory Severe-Spike Alert Prevalence (>= 120 ug/m3):** Rare atmospheric stagnation episodes occur in **~0.02%** of network hours (21 hours across the entire 45-month period).
- **Zero Lookahead & Target Decoupling:** Targets are derived strictly from raw un-imputed physical sensor observations. Missing targets evaluate strictly to `NaN` and are never coerced to zero.

---
*Report generated automatically by MilestoneReportGenerator.*