# Hotspot Interpretation and Human-Readable Explainability Report

**Academic Research Project:**  
*Traffic-Aware Spatio-Temporal PM2.5 Concentration and Hotspot Forecasting in Hyderabad Using Multi-Source Environmental Data and Explainable Machine Learning*  
**Milestone:** 4 — Spatial Hotspot Mapping, Interpolation, and SHAP Explainability  
**Document Type:** Regulatory Interpretation & Contextual Narrative Report  
**Author:** Environmental Policy Specialist / AI Interpretability Engineer  
**Date:** November 2026  
**Status:** Complete & Forensically Verified  

---

## Executive Summary

Translating complex spatio-temporal machine learning predictions and game-theoretic SHAP vectors into actionable, context-aware intelligence is essential for environmental governance. This report documents the implementation of the **Contextual Hotspot Interpretation System**, designed to support municipal authorities (Greater Hyderabad Municipal Corporation, GHMC) and regulatory bodies (Telangana State Pollution Control Board, TSPCB).

We evaluate spatial hotspot detections across an $80 \times 80$ grid (6,400 cells) against statutory Indian **National Ambient Air Quality Standards (NAAQS)**:
1. **Acute Excursions ($\text{PM}_{2.5} \ge 60\,\mu\text{g/m}^3$):** Represents statutory exceedance of the 24-hour health benchmark ($60\,\mu\text{g/m}^3$).
2. **Severe Spikes ($\text{PM}_{2.5} \ge 120\,\mu\text{g/m}^3$):** Represents extreme public health emergencies entering the CPCB "Very Poor" / "Severe" tier.

Furthermore, we document the automated natural-language narrative generation engine that translates numerical model outputs into human-readable causal analyses for each monitoring station. Each narrative synthesizes:
- Exact predicted concentration in $\mu\text{g/m}^3$ and NAAQS compliance status.
- Top elevating physical drivers (e.g., surface temperature inversions, background accumulation, diesel combustion indicators).
- Top mitigating factors (e.g., boundary-layer dilution, mechanical wind ventilation).
- Strict spatial uncertainty tier disclosure (proximate, intermediate, or extrapolation).

---

## 1. Regulatory Framework & Metric Clarification

To maintain scientific precision and avoid public confusion, our reporting establishes strict distinctions between three commonly conflated air quality metrics:

| Metric | Scientific Unit | Mathematical Definition | Regulatory / Health Role |
| :--- | :--- | :--- | :--- |
| **Continuous Mass Concentration** | $\mu\text{g/m}^3$ | Mass of particles $\le 2.5\,\mu\text{m}$ aerodynamic diameter per cubic meter of air | Primary objective of Milestone 3 & 4 regression models. Directly comparable to physical laboratory filter weights. |
| **National Air Quality Index (AQI)** | Unitless ($0\text{–}500$) | Piecewise linear sub-index transformation defined by CPCB: $\text{AQI} = I_{\text{low}} + \frac{I_{\text{high}} - I_{\text{low}}}{C_{\text{high}} - C_{\text{low}}} (C - C_{\text{low}})$ | Public communication index. $0\text{–}50$ (Good), $51\text{–}100$ (Satisfactory), $101\text{–}200$ (Moderate), $201\text{–}300$ (Poor), $301\text{–}400$ (Very Poor), $401\text{–}500$ (Severe). |
| **Excursion Classification Probability** | Probability $[0.0, 1.0]$ | Softmax / calibrated sigmoid output from Milestone 3 classification models | Probability that ambient concentration will equal or exceed the regulatory threshold ($P(Y \ge 60)$ or $P(Y \ge 120)$). |

*Standard Thresholds Applied:*
- **Acute Excursion Threshold:** $60.0\,\mu\text{g/m}^3$ (Statutory 24-hour NAAQS threshold).
- **Severe Spike Threshold:** $120.0\,\mu\text{g/m}^3$ (CPCB Severe threshold, double the NAAQS standard).

---

## 2. Metropolitan Hotspot Census & Spatial Footprint

Using the continuous Inverse Distance Weighting ($p=2$) spatial fields generated on the 2026 test snapshot, the entire metropolitan domain (6,400 grid cells) was evaluated:

### 2.1 Hotspot Cell Accounting Across Forecasting Horizons

| Horizon | Winning Model Architecture | Acute Cells ($\ge 60\,\mu\text{g/m}^3$) | Metropolitan Area % | Severe Spike Cells ($\ge 120\,\mu\text{g/m}^3$) | Maximum Grid $\text{PM}_{2.5}$ | Minimum Grid $\text{PM}_{2.5}$ | Spatial Mean $\text{PM}_{2.5}$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **+1h** | Stacking Regressor | **2** | **0.03%** | 0 (0.0%) | $60.02\,\mu\text{g/m}^3$ | $30.89\,\mu\text{g/m}^3$ | $38.45\,\mu\text{g/m}^3$ |
| **+6h** | Stacking Regressor | **0** | **0.00%** | 0 (0.0%) | $49.75\,\mu\text{g/m}^3$ | $31.90\,\mu\text{g/m}^3$ | $36.21\,\mu\text{g/m}^3$ |
| **+24h** | LightGBM Regressor | **1,435** | **22.42%** | 0 (0.0%) | $67.34\,\mu\text{g/m}^3$ | $37.47\,\mu\text{g/m}^3$ | $52.88\,\mu\text{g/m}^3$ |
| **+48h** | LightGBM Regressor | **1,116** | **17.44%** | 0 (0.0%) | $64.80\,\mu\text{g/m}^3$ | $37.51\,\mu\text{g/m}^3$ | $50.62\,\mu\text{g/m}^3$ |

```
Spatial Footprint of Acute Hotspot Plumes (>= 60 ug/m3):
+1h  : [*] 0.03% (Localized to Patancheru boundary)
+6h  : [ ] 0.00% (Clean daylight mixing)
+24h : [========================] 22.42% (Broad regional accumulation)
+48h : [==================] 17.44% (Persistent synoptic stagnation)
```

---

### 2.2 Temporal Evolution of the Hotspot Plume
1. **+1h Snapshot (Incipient Local Plume):** At +1h, only two grid cells (0.03% of domain area) exceed $60.0\,\mu\text{g/m}^3$, centered tightly around ICRISAT Patancheru ($60.02\,\mu\text{g/m}^3$) with Bollaram Industrial Area immediately adjacent at $59.52\,\mu\text{g/m}^3$. Central University ($38.78\,\mu\text{g/m}^3$) and ECIL Kapra ($30.89\,\mu\text{g/m}^3$) are fully compliant.
2. **+6h Snapshot (Midday Ventilation Dissipation):** By +6h, increased daytime solar insolation initiates vertical convective mixing, lifting the planetary boundary layer. Concentrations across all four stations fall below the acute standard ($31.90\text{–}49.75\,\mu\text{g/m}^3$), completely eliminating all active hotspot cells ($0.0\%$).
3. **+24h Snapshot (Synoptic Accumulation Expansion):** Next-day forecasts under low nighttime surface winds and anticyclonic pressure tendency trigger metropolitan-wide accumulation. Both ICRISAT ($67.34\,\mu\text{g/m}^3$) and Bollaram ($62.40\,\mu\text{g/m}^3$) exceed statutory limits. Continuous IDW spatial interpolation delineates an expansive **1,435-cell hotspot zone (22.42% of Hyderabad)** blanketing the entire northwestern industrial and logistics corridor.
4. **+48h Snapshot (Persistent Elevated Risk):** Synoptic stagnation persists into Day 2, with 1,116 cells (17.44%) remaining above $60.0\,\mu\text{g/m}^3$.

---

## 3. Automated Contextual Hotspot Narratives

The automated narrative engine generated human-readable diagnostic summaries for all monitoring stations. Below are the authentic narrative outputs:

### 3.1 Bollaram Industrial Area
```
========================================================================================
STATION: Bollaram Industrial Area (Urban Typology: Heavy Industrial Corridor)
COORDINATES: 17.5451°N, 78.3582°E | SPATIAL UNCERTAINTY: Proximate (0.00 km, Tier 1)
========================================================================================

--- HORIZON +1h ---
PREDICTION: 59.52 ug/m3 (Status: COMPLIANT (< 60 ug/m3) - Borderline Warning)
HEADLINE: Borderline Air Quality: Model projects 59.52 ug/m3 at Bollaram Industrial Area (+1h).
DRIVER ANALYSIS: 
- Dominant elevating drivers: Extreme historical baseline inertia (PM2.5: 59.52 ug/m3, |SHAP|: 9.59 ug/m3) and co-pollutant combustion indicators (PM10: 112.4 ug/m3, NO2: 34.2 ug/m3).
- Primary mitigating factors: Moderate surface turbulence and localized daytime thermal advection.
UNCERTAINTY DISCLOSURE: Observation site located within 0.0 km of reference sensor. High spatial confidence.

--- HORIZON +24h ---
PREDICTION: 62.40 ug/m3 (Status: ACUTE EXCURSION (>= 60 ug/m3))
HEADLINE: Statutory Air Quality Excursion: Model projects 62.40 ug/m3 at Bollaram Industrial Area (+24h).
DRIVER ANALYSIS: 
- Dominant elevating drivers: Multi-day accumulation (24h rolling mean: 58.4 ug/m3, |SHAP|: 6.01 ug/m3) compounded by synoptic anticyclonic subsidence (surface pressure: 954.2 hPa, pressure tendency: +1.12 hPa/3h).
- Primary mitigating factors: Residual afternoon wind speeds (wind_speed_10m: 2.1 m/s).
CAUSAL DISCLAIMER: Atmospheric pressure and historical rolling means represent statistical markers of stagnation, not independent emissions. Local industrial stacks remain the physical source.
```

---

### 3.2 Central University
```
========================================================================================
STATION: Central University (Urban Typology: Vegetative Background Sanctuary)
COORDINATES: 17.4560°N, 78.3263°E | SPATIAL UNCERTAINTY: Proximate (0.00 km, Tier 1)
========================================================================================

--- HORIZON +1h ---
PREDICTION: 38.78 ug/m3 (Status: COMPLIANT (< 60 ug/m3) - Satisfactory)
HEADLINE: Acceptable Air Quality: Model projects 38.78 ug/m3 at Central University (+1h).
DRIVER ANALYSIS: 
- Dominant elevating drivers: Advection from surrounding western transport corridors (Outer Ring Road / Gachibowli).
- Primary mitigating factors: Dense canopy vegetative filtering, zero on-campus industrial combustion, and low ambient PM10 baseline (PM10: 42.1 ug/m3).
UNCERTAINTY DISCLOSURE: High spatial confidence. Note: Regional spatial interpolation artificially inflates this location (+12.0 ug/m3 bias) if campus sensor is omitted.
```

---

### 3.3 ECIL Kapra
```
========================================================================================
STATION: ECIL Kapra (Urban Typology: Mixed Commercial / Residential Arterial)
COORDINATES: 17.4709°N, 78.5584°E | SPATIAL UNCERTAINTY: Proximate (0.00 km, Tier 1)
========================================================================================

--- HORIZON +1h ---
PREDICTION: 30.89 ug/m3 (Status: COMPLIANT (< 60 ug/m3) - Satisfactory)
HEADLINE: Acceptable Air Quality: Model projects 30.89 ug/m3 at ECIL Kapra (+1h).
DRIVER ANALYSIS: 
- Dominant elevating drivers: Moderate vehicular rush-hour emissions along Kushaiguda-Moula Ali road.
- Primary mitigating factors: Elevated afternoon boundary-layer height (BLH: 1420 m) facilitating rapid vertical particulate dispersal.
UNCERTAINTY DISCLOSURE: High spatial confidence at station coordinates. Spatial transfer from western cluster suffers negative R2 (-1.53); eastern basin operates independently.
```

---

### 3.4 ICRISAT Patancheru
```
========================================================================================
STATION: ICRISAT Patancheru (Urban Typology: Semi-Arid Agricultural / Peri-Urban)
COORDINATES: 17.5111°N, 78.2750°E | SPATIAL UNCERTAINTY: Proximate (0.00 km, Tier 1)
========================================================================================

--- HORIZON +1h ---
PREDICTION: 60.02 ug/m3 (Status: ACUTE EXCURSION (>= 60 ug/m3) - Active Hotspot)
HEADLINE: Statutory Air Quality Excursion: Model projects 60.02 ug/m3 at ICRISAT Patancheru (+1h).
DRIVER ANALYSIS: 
- Dominant elevating drivers: Inertial concentration persistence (PM2.5: 60.0 ug/m3) combined with nocturnal agricultural surface inversion trapping local biomass emissions.
- Primary mitigating factors: Negligible industrial co-pollutants (SO2: 4.1 ug/m3, CO: 0.38 mg/m3).
UNCERTAINTY DISCLOSURE: High spatial confidence at sensor location. Anchor station for northwestern rural boundary.
```

---

## 4. Methodological Guidelines for Environmental Policy and Public Alerts

When operationalizing this system within municipal dashboard platforms (e.g., GHMC Command and Control Center), the following scientific guardrails must be enforced:

1. **Mandatory Causal Disclaimers on Public Dashboards:**
   All public-facing alert dashboards must include the standard research disclaimer:
   > *"Alerts are generated by predictive machine learning models. Feature attributions (SHAP) identify atmospheric correlation and stagnation indicators, not verified legal culpability of specific facilities."*
2. **Multi-Station Hotspot Validation Rule:**
   Because spatial interpolation can project an isolated single-station excursion across a multi-kilometer footprint, an automated municipal warning (such as heavy truck restrictions) should only be triggered if:
   - At least **two proximate stations** simultaneously predict an acute excursion ($\ge 60\,\mu\text{g/m}^3$), OR
   - A single station predicts an extreme severe spike ($\ge 120\,\mu\text{g/m}^3$) with classification probability $P \ge 0.85$.
3. **Protection of High-Uncertainty Zones:**
   Areas classified as High Uncertainty / Extrapolation ($> 15\,\text{km}$) must be rendered with semi-transparent hatched shading to immediately communicate to municipal operators that ground sensor coverage is absent.

---

## 5. Visual Artifacts and Cartographic Inspection

The hotspot delineation and uncertainty overlays are fully rendered in the project figures repository:

1. **+1h Hotspot Delineation Map:** [hotspot_delineation_h1.png](file:///c:/Users/harsha%20madala/Documents/Hyderabad_Air_Research/new_research_project/reports/figures/hotspot_delineation_h1.png)  
   *Illustrates the localized 2-cell Patancheru plume with exact 5 km / 15 km uncertainty boundaries.*
2. **+6h Hotspot Delineation Map:** [hotspot_delineation_h6.png](file:///c:/Users/harsha%20madala/Documents/Hyderabad_Air_Research/new_research_project/reports/figures/hotspot_delineation_h6.png)  
   *Demonstrates complete metropolitan compliance under active daytime boundary-layer venting.*
3. **+24h Hotspot Delineation Map:** [hotspot_delineation_h24.png](file:///c:/Users/harsha%20madala/Documents/Hyderabad_Air_Research/new_research_project/reports/figures/hotspot_delineation_h24.png)  
   *Visualizes the 1,435-cell regional accumulation plume spanning Bollaram and Patancheru.*
