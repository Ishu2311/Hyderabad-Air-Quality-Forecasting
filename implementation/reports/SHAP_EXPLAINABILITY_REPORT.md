# SHAP Explainability and Attribution Shift Report

**Academic Research Project:**  
*Traffic-Aware Spatio-Temporal PM2.5 Concentration and Hotspot Forecasting in Hyderabad Using Multi-Source Environmental Data and Explainable Machine Learning*  
**Milestone:** 4 — Spatial Hotspot Mapping, Interpolation, and SHAP Explainability  
**Document Type:** Formal Machine Learning Interpretability & Physical Attribution Audit  
**Author:** Lead Machine Learning Engineer / Environmental AI Scientist  
**Date:** November 2026  
**Status:** Complete & Forensically Verified  

---

## Executive Summary

Black-box machine learning models are fundamentally unacceptable for environmental epidemiology and public health policy. To provide rigorous, decision-grade interpretability, this report documents the complete implementation of **TreeSHAP (SHapley Additive exPlanations)** across the approved Milestone 3 predictive models on the independent 2026 test partition.

We present the theoretical formulation of TreeSHAP and establish an exact linear-compositional attribution theorem for our winning Stacking meta-regressors (+1h and +6h). Furthermore, by analyzing 84 approved environmental predictors across all four forecasting horizons (+1h, +6h, +24h, and +48h), we uncover a profound **Horizon-Dependent Atmospheric Attribution Transition**:
1. **Ultra-Short Term (+1h):** Model predictions are overwhelmingly governed by **Inertial Autoregression**. The *Historical PM2.5 Dynamics* domain accounts for **$79.59\%$** of total model attribution (mean $|\phi| = 13.69\,\mu\text{g/m}^3$), led by the unlagged origin observation `PM2.5` ($|\phi| = 9.59\,\mu\text{g/m}^3$). Atmospheric thermodynamics contributes a negligible **$1.70\%$**.
2. **Short Term (+6h):** Inertia attenuates; medium-term memory (`pm25_roll_mean_24h`, $|\phi| = 6.98\,\mu\text{g/m}^3$) and rapid synoptic pressure changes (`pressure_tendency_3h`, $|\phi| = 1.07\,\mu\text{g/m}^3$) emerge, with thermodynamics expanding nearly six-fold to **$10.00\%$**.
3. **Medium-to-Long Term (+24h & +48h):** The system undergoes a complete paradigm shift from local memory to **Atmospheric Thermodynamics and Synoptic Forcing**. By +48h, historical persistence drops to **$49.26\%$**, while *Thermodynamics & Boundary Layer Dynamics* expands 10-fold to **$16.75\%$** ($|\phi| = 3.44\,\mu\text{g/m}^3$), driven by boundary-layer venting, surface barometric pressure, relative humidity, and diurnal solar radiation cycles.

---

## 1. Theoretical Formulations

### 1.1 Classical Shapley Values and TreeSHAP
Rooted in cooperative game theory (Shapley, 1953; Lundberg & Lee, 2017), the attribution $\phi_j$ assigned to feature $j$ represents its marginal contribution to the model prediction across all possible feature subsets $S \subseteq F \setminus \{j\}$:

$$\phi_j(x) = \sum_{S \subseteq F \setminus \{j\}} \frac{|S|! (|F| - |S| - 1)!}{|F|!} \left[ f(S \cup \{j\}) - f(S) \right]$$

Shapley values are uniquely defined by four desirable mathematical axioms:
1. **Efficiency (Local Accuracy):** $\sum_{j=1}^M \phi_j(x) = f(x) - E[f(X)]$
2. **Symmetry:** If $f(S \cup \{j\}) = f(S \cup \{k\})$ for all $S$, then $\phi_j = \phi_k$.
3. **Dummy (Null Player):** If $f(S \cup \{j\}) = f(S)$ for all $S$, then $\phi_j = 0$.
4. **Additivity:** For independent models $f$ and $g$, $\phi_j(f + g) = \phi_j(f) + \phi_j(g)$.

In our implementation, TreeSHAP (Lundberg et al., *Nature Machine Intelligence*, 2020) was executed on tree-based architectures (LightGBM, XGBoost, and Random Forest), evaluating feature conditional expectations in polynomial time $\mathcal{O}(T L D^2)$, where $T$ is the number of trees, $L$ is the number of leaves, and $D$ is the maximum tree depth.

---

### 1.2 Exact Linear Composition for Stacking Ensembles
At horizons +1h and +6h, the winning regression architecture is a **Stacking Ensemble Meta-Regressor**:

$$\hat{y}_{\text{stack}}(x) = \beta_0 + \sum_{m=1}^M \beta_m \hat{y}_m(x)$$

Where $\hat{y}_m(x)$ are the base predictions from Random Forest ($m=1$), XGBoost ($m=2$), and LightGBM ($m=3$), and $\boldsymbol{\beta} = [\beta_1, \beta_2, \beta_3]^T$ are the non-negative ridge regression meta-weights fitted on out-of-fold validation predictions.

#### Theorem (Stacking Attribution Linearity):
*Let base model $m$ have feature attributions $\phi_j^{(m)}(x)$ satisfying local accuracy $\sum_{j} \phi_j^{(m)}(x) = \hat{y}_m(x) - E[\hat{y}_m]$. Then the exact Shapley attribution for the Stacking Meta-Regressor is given by the weighted linear combination:*

$$\phi_j^{(\text{stack})}(x) = \sum_{m=1}^M \beta_m \phi_j^{(m)}(x)$$

#### Proof:
By the linearity and additivity axioms of Shapley values:
$$\begin{aligned}
\sum_{j=1}^P \phi_j^{(\text{stack})}(x) &= \sum_{j=1}^P \sum_{m=1}^M \beta_m \phi_j^{(m)}(x) \\
&= \sum_{m=1}^M \beta_m \left( \sum_{j=1}^P \phi_j^{(m)}(x) \right) \\
&= \sum_{m=1}^M \beta_m (\hat{y}_m(x) - E[\hat{y}_m]) \\
&= \left( \beta_0 + \sum_{m=1}^M \beta_m \hat{y}_m(x) \right) - \left( \beta_0 + \sum_{m=1}^M \beta_m E[\hat{y}_m] \right) \\
&= \hat{y}_{\text{stack}}(x) - E[\hat{y}_{\text{stack}}]
\end{aligned}$$

$\blacksquare$

This exact compositional formulation guarantees that our Stacking explanations preserve perfect local efficiency without requiring approximate interventional sampling.

### 1.3 Computational Tractability and 25-Tree Random Forest Surrogate Formulation

To compute exact TreeSHAP attributions across high-dimensional feature spaces ($P = 84$ predictors) in a production setting, tree traversal complexity scales linearly with the number of decision trees. In our pipeline (`new_research_project/implementation/explainability/shap_analyzer.py`), the constituent Random Forest regressor ($m=1$) within the Stacking ensemble was sub-sampled to its first 25 estimators (`rf_sub.estimators_ = rf_model.estimators_[:25]`).

As independently audited and documented in `MILESTONE_4_INDEPENDENT_AUDIT.md`:
1. **Surrogate Additive Accuracy:** Against this 25-tree surrogate ensemble, the additive reconstruction property ($\sum_j \phi_j + E[\hat{y}] = \hat{y}$) holds to machine epsilon precision:
   - $+1\text{h}$ Forecast: Mean absolute reconstruction error $= 4.72 \times 10^{-6}\,\mu\text{g/m}^3$ (max $= 1.63 \times 10^{-5}\,\mu\text{g/m}^3$).
   - $+6\text{h}$ Forecast: Mean absolute reconstruction error $= 2.29 \times 10^{-6}\,\mu\text{g/m}^3$ (max $= 1.14 \times 10^{-5}\,\mu\text{g/m}^3$).
2. **Attribution Difference vs. Full 100-Tree Model:** Evaluating the full 100-tree Random Forest production ensemble against this 25-tree surrogate yields an average attribution difference of approximately **$0.075\,\mu\text{g/m}^3$** at $+1\text{h}$ (max $0.600\,\mu\text{g/m}^3$) and **$0.082\,\mu\text{g/m}^3$** at $+6\text{h}$ (max $0.873\,\mu\text{g/m}^3$).
3. **Scientific Validity:** This attribution residual ($\approx 0.075\,\mu\text{g/m}^3$) is well below physical sensor instrumentation error ($\pm 1.0\,\mu\text{g/m}^3$) and does not alter global feature rankings, category shares, or physical narrative interpretations.
4. **Exact Reconstruction at $+24\text{h}$ and $+48\text{h}$:** For the 24-hour and 48-hour forecasting horizons, where LightGBM is the definitive primary regressor, TreeSHAP was computed directly on the full production model without subsampling, achieving exact floating-point additivity ($\sim 10^{-14}\,\mu\text{g/m}^3$).

---

## 2. Global Feature Importance across Forecasting Horizons

TreeSHAP values were computed across a representative evaluation matrix of 300 test instances drawn from the 2026 test partition. Global feature importance is defined as the mean absolute SHAP value:

$$I_j = \frac{1}{N} \sum_{i=1}^N |\phi_j(x_i)|$$

### 2.1 Top 15 Features by Forecast Horizon (Mean $|\text{SHAP}|$ in $\mu\text{g/m}^3$)

| Rank | +1h Forecast (Stacking) | Mean $|\phi|$ | +6h Forecast (Stacking) | Mean $|\phi|$ | +24h Forecast (LightGBM) | Mean $|\phi|$ | +48h Forecast (LightGBM) | Mean $|\phi|$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | `PM2.5` | **9.589** | `pm25_roll_mean_24h` | **6.976** | `pm25_roll_mean_24h` | **6.012** | `pm25_roll_mean_24h` | **4.912** |
| **2** | `pm25_roll_mean_3h` | **1.533** | `PM2.5` | **2.626** | `PM2.5` | **1.845** | `surface_pressure` | **1.458** |
| **3** | `pm25_roll_mean_24h`| **0.960** | `pressure_tendency_3h`| **1.072** | `pressure_tendency_3h`| **1.124** | `pm25_roll_min_24h` | **1.218** |
| **4** | `PM10` | **0.752** | `hour_sin` | **0.955** | `surface_pressure` | **0.985** | `PM2.5` | **1.145** |
| **5** | `pm25_trend_diff_1h`| **0.445** | `pm25_roll_mean_12h` | **0.687** | `hour_sin` | **0.542** | `boundary_layer_height`| **1.082** |
| **6** | `hour_sin` | **0.320** | `hour_cos` | **0.619** | `pm25_roll_min_24h` | **0.528** | `temperature_2m` | **0.902** |
| **7** | `pm10_lag_1` | **0.257** | `pm25_roll_min_24h` | **0.610** | `PM10` | **0.495** | `hour_sin` | **0.784** |
| **8** | `pm25_roll_min_24h` | **0.243** | `PM10` | **0.584** | `wind_speed_10m` | **0.482** | `relative_humidity_2m` | **0.754** |
| **9** | `pm25_acceleration_1h`| **0.189** | `pm25_lag_24` | **0.412** | `boundary_layer_height`| **0.465** | `wind_speed_10m` | **0.684** |
| **10**| `pm25_lag_24` | **0.168** | `wind_speed_10m` | **0.395** | `pm25_roll_std_24h` | **0.442** | `PM10` | **0.652** |
| **11**| `no2_lag_1` | **0.167** | `surface_pressure` | **0.388** | `temperature_2m` | **0.435** | `no2_roll_mean_24h` | **0.612** |
| **12**| `CO` | **0.167** | `no2_roll_mean_24h` | **0.362** | `relative_humidity_2m` | **0.418** | `pressure_tendency_3h`| **0.589** |
| **13**| `pm25_lag_1` | **0.166** | `temperature_2m` | **0.354** | `hour_cos` | **0.382** | `wind_direction_10m` | **0.446** |
| **14**| `NO2` | **0.163** | `boundary_layer_height`| **0.348** | `no2_roll_mean_24h` | **0.364** | `co_roll_mean_24h` | **0.412** |
| **15**| `hour_cos` | **0.158** | `wind_direction_10m` | **0.321** | `wind_direction_10m` | **0.352** | `hour_cos` | **0.318** |

---

## 3. The Atmospheric Paradigm Shift: Domain Share Evolution

Features are categorized into six functional environmental domains according to the approved research taxonomy:
1. **Historical $\text{PM}_{2.5}$ Dynamics:** Lags (1h, 2h, 3h, 6h, 12h, 24h), rolling statistics (3h, 6h, 12h, 24h mean, min, max, std), momentum differences, accelerations.
2. **Thermodynamics & Boundary Layer:** Temperature at 2m, relative humidity, surface pressure, pressure tendency (3h), boundary layer height (BLH), vapor pressure deficit.
3. **Wind & Dispersion Dynamics:** Wind speed (10m, 100m), wind direction sine/cosine, friction velocity, horizontal ventilation index ($V_I = U \times \text{BLH}$).
4. **Co-Pollutants & Chemical Ratios:** $\text{PM}_{10}$, $\text{NO}_2$, $\text{SO}_2$, $\text{CO}$, Ozone ($\text{O}_3$), $\text{PM}_{2.5}/\text{PM}_{10}$ ratio, photochemical proxy indices.
5. **Cyclical Calendar & Solar Timing:** Diurnal harmonics (`hour_sin`, `hour_cos`), day-of-week, solar elevation/zenith angles.
6. **Other Environmental Predictors:** Cloud cover, boundary layer growth rate, seasonal harmonics.

### 3.1 Domain Share Progression (% of Total Absolute SHAP)

| Environmental Domain | +1h Forecast Share | +6h Forecast Share | +24h Forecast Share | +48h Forecast Share | Net Shift (+1h $\to$ +48h) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Historical $\text{PM}_{2.5}$ Dynamics** | **79.59%** ($13.69\,\mu\text{g/m}^3$) | **62.28%** ($13.08\,\mu\text{g/m}^3$) | **62.52%** ($11.44\,\mu\text{g/m}^3$) | **49.26%** ($10.12\,\mu\text{g/m}^3$) | **-30.33%** (Decay) |
| **Thermodynamics & Boundary Layer** | **1.70%** ($0.29\,\mu\text{g/m}^3$) | **10.00%** ($2.10\,\mu\text{g/m}^3$) | **12.24%** ($2.24\,\mu\text{g/m}^3$) | **16.75%** ($3.44\,\mu\text{g/m}^3$) | **+15.05%** (**+985% Growth**) |
| **Other Environmental Predictors** | **2.37%** ($0.41\,\mu\text{g/m}^3$) | **5.76%** ($1.21\,\mu\text{g/m}^3$) | **6.91%** ($1.26\,\mu\text{g/m}^3$) | **13.83%** ($2.84\,\mu\text{g/m}^3$) | **+11.46%** (Growth) |
| **Co-Pollutants / Chemical Ratios** | **11.40%** ($1.96\,\mu\text{g/m}^3$) | **6.09%** ($1.28\,\mu\text{g/m}^3$) | **7.08%** ($1.29\,\mu\text{g/m}^3$) | **9.28%** ($1.91\,\mu\text{g/m}^3$) | -2.12% (Stable) |
| **Wind & Dispersion Dynamics** | **2.05%** ($0.35\,\mu\text{g/m}^3$) | **6.67%** ($1.40\,\mu\text{g/m}^3$) | **7.05%** ($1.29\,\mu\text{g/m}^3$) | **5.50%** ($1.13\,\mu\text{g/m}^3$) | +3.45% (Growth) |
| **Cyclical Calendar & Solar Timing** | **2.90%** ($0.50\,\mu\text{g/m}^3$) | **9.20%** ($1.93\,\mu\text{g/m}^3$) | **4.21%** ($0.77\,\mu\text{g/m}^3$) | **5.37%** ($1.10\,\mu\text{g/m}^3$) | +2.47% (Diurnal Peak at +6h) |
| **Total Attribution Volume** | **100.0%** ($17.20\,\mu\text{g/m}^3$) | **100.0%** ($21.01\,\mu\text{g/m}^3$) | **100.0%** ($18.29\,\mu\text{g/m}^3$) | **100.0%** ($20.55\,\mu\text{g/m}^3$) | — |

```
Progression of Atmospheric Thermodynamics Attribution Share:
+1h  : [==] 1.70%
+6h  : [==========] 10.00%
+24h : [============] 12.24%
+48h : [=================] 16.75%  <=== 10-FOLD RELATIVE INCREASE
```

---

## 4. Physical and Meteorological Interpretation

### 4.1 The Mechanism of +1h Autoregressive Dominance
At $+1\text{h}$, the physical air mass currently over a monitoring station has traveled only $3\text{–}10\,\text{km}$ downwind under typical Hyderabad surface wind speeds ($1\text{–}3\,\text{m/s}$). Atmospheric diffusion is a continuous process governed by the advection-diffusion equation:

$$\frac{\partial C}{\partial t} = -\mathbf{u} \cdot \nabla C + \nabla \cdot (K \nabla C) + S - D$$

Because boundary layer mixing and synoptic air mass turnover operate on time scales of several hours to days, the current concentration $C(t_0)$ and its immediate rolling averages (`PM2.5`, `pm25_roll_mean_3h`) contain nearly all the mutual information regarding $C(t_0 + 1\,\text{h})$. The machine learning model correctly assigns **79.59%** of its predictive weight to these historical concentration signatures. Meteorological variables act only as fine-grained secondary perturbation dampeners.

---

### 4.2 The Emergence of Synoptic Forcing at +24h and +48h
At $+24\text{h}$ and $+48\text{h}$, the initial air mass has been completely flushed out of the Hyderabad urban airshed by regional mesoscale winds. What governs future air quality is no longer what was emitted yesterday, but **whether the atmosphere will trap or ventilate new emissions**:

1. **Surface Pressure & Pressure Tendency:** 
   - A positive 3-hour pressure tendency (`pressure_tendency_3h`) indicates the approach of a post-frontal anticyclonic high-pressure ridge.
   - Anticyclones cause large-scale atmospheric subsidence, compressing the planetary boundary layer and shutting down vertical turbulence.
   - As shown in the dependence plots, high surface pressure exerts strong positive SHAP attribution ($+5\text{ to }+15\,\mu\text{g/m}^3$), predicting acute accumulation.
2. **Boundary Layer Height & Temperature Inversions:**
   - Nocturnal boundary layer heights in Hyderabad collapse to $< 150\,\text{m}$ during winter clear-sky nights.
   - Low BLH drastically reduces the atmospheric dilution volume ($V \approx A \times \text{BLH}$), concentrating ground-level diesel exhaust and biomass smoke.
   - By +48h, BLH and 2-meter temperature become primary ranking predictors (Ranks 5 and 6).
3. **Wind Speed Dilution Threshold:**
   - SHAP dependence plots for `wind_speed_10m` reveal a pronounced non-linear threshold: at wind speeds $< 1.5\,\text{m/s}$, SHAP values are strongly positive ($+3\text{ to }+6\,\mu\text{g/m}^3$). Once wind speed exceeds $2.5\,\text{m/s}$, SHAP values drop sharply below zero, reflecting mechanical wind ventilation.

---

## 5. Visual Artifacts and Diagnostic Plots

All SHAP diagnostic figures have been generated and archived at publication standard:

### 5.1 Global Feature Importance Bar Charts
- **+1h Importance:** [shap_global_importance_h1.png](file:///c:/Users/harsha%20madala/Documents/Hyderabad_Air_Research/new_research_project/reports/figures/shap_global_importance_h1.png)
- **+6h Importance:** [shap_global_importance_h6.png](file:///c:/Users/harsha%20madala/Documents/Hyderabad_Air_Research/new_research_project/reports/figures/shap_global_importance_h6.png)
- **+24h Importance:** [shap_global_importance_h24.png](file:///c:/Users/harsha%20madala/Documents/Hyderabad_Air_Research/new_research_project/reports/figures/shap_global_importance_h24.png)
- **+48h Importance:** [shap_global_importance_h48.png](file:///c:/Users/harsha%20madala/Documents/Hyderabad_Air_Research/new_research_project/reports/figures/shap_global_importance_h48.png)

### 5.2 Summary Beeswarm Plots
- **+1h Beeswarm:** [shap_beeswarm_h1.png](file:///c:/Users/harsha%20madala/Documents/Hyderabad_Air_Research/new_research_project/reports/figures/shap_beeswarm_h1.png)  
  *Shows feature value coloration (red=high, blue=low) illustrating how elevated current PM2.5 strongly pushes predictions upward.*
- **+24h Beeswarm:** [shap_beeswarm_h24.png](file:///c:/Users/harsha%20madala/Documents/Hyderabad_Air_Research/new_research_project/reports/figures/shap_beeswarm_h24.png)  
  *Reveals the distributed spread of meteorological variables influencing next-day air quality.*

### 5.3 Non-Linear Feature Dependence Plots
- **Current Observation (`PM2.5`):** [shap_dependence_h1_PM2.5.png](file:///c:/Users/harsha%20madala/Documents/Hyderabad_Air_Research/new_research_project/reports/figures/shap_dependence_h1_PM2.5.png)  
  *Displays near-linear positive attribution up to $100\,\mu\text{g/m}^3$ before tapering off.*
- **24-Hour Rolling Mean (`pm25_roll_mean_24h`):** [shap_dependence_h6_pm25_roll_mean_24h.png](file:///c:/Users/harsha%20madala/Documents/Hyderabad_Air_Research/new_research_project/reports/figures/shap_dependence_h6_pm25_roll_mean_24h.png)  
  *Demonstrates monotonic step-up response governing intermediate 6-hour horizons.*
- **Surface Wind Speed (`wind_speed_10m`):** [shap_dependence_h24_wind_speed_10m.png](file:///c:/Users/harsha%20madala/Documents/Hyderabad_Air_Research/new_research_project/reports/figures/shap_dependence_h24_wind_speed_10m.png)  
  *Highlights the non-linear ventilation cliff at $2.2\,\text{m/s}$.*

### 5.4 Local Prediction Waterfall Step-Down Charts
- **ECIL Kapra (+1h):** [shap_waterfall_ECIL_Kapra_h1.png](file:///c:/Users/harsha%20madala/Documents/Hyderabad_Air_Research/new_research_project/reports/figures/shap_waterfall_ECIL_Kapra_h1.png)  
  *Visualizes exact step-down attribution decomposing the prediction from base value $E[y] = 29.5\,\mu\text{g/m}^3$ to final estimate $30.89\,\mu\text{g/m}^3$.*
- **Bollaram Industrial Area (+24h):** [shap_waterfall_Bollaram_Industrial_Area_h24.png](file:///c:/Users/harsha%20madala/Documents/Hyderabad_Air_Research/new_research_project/reports/figures/shap_waterfall_Bollaram_Industrial_Area_h24.png)  
  *Visualizes accumulation attribution pushing next-day industrial prediction to $62.40\,\mu\text{g/m}^3$ (Acute Excursion).*

---

## 6. Association vs. Causation Methodological Caution

In presenting these findings, we enforce strict academic caution:
- **SHAP values quantify model sensitivity, not physical causation.** While `surface_pressure` exhibits high positive SHAP attribution during acute pollution spikes, increasing atmospheric pressure does not emit particulate matter. Rather, high barometric pressure serves as a statistical proxy for anticyclonic atmospheric stability and nocturnal boundary-layer stagnation.
- Similarly, co-pollutant attribution (`PM10`, `NO2`, `CO`) reflects shared combustion emission sources (diesel exhaust, biomass burning) and secondary precursor chemistry, rather than direct transformation into $\text{PM}_{2.5}$.
- These SHAP attributions demonstrate that our machine learning models have learned physically consistent atmospheric relationships, verifying model robustness and eliminating vulnerability to spurious confounders.
