# Meteorological Data Provenance and Point-in-Time Availability Report

**Document Code:** `REPORT-M2-WEATHER-PROVENANCE`  
**Project:** Traffic-Aware Spatio-Temporal PM2.5 Concentration and Hotspot Forecasting in Hyderabad  
**Milestone:** Milestone 2 – Multi-Source Feature Engineering  
**Status:** COMPLETE & VERIFIED  

---

## 1. Executive Summary & Atmospheric Provenance

This report formalizes the meteorological data provenance, mathematical transformations, and strict point-in-time availability protocols governing the integration of atmospheric covariates in the Hyderabad air quality forecasting framework.

### Atmospheric Data Source & License
- **Provider:** Open-Meteo Historical Weather API.
- **Underlying Assimilation Model:** European Centre for Medium-Range Weather Forecasts (ECMWF) ERA5 atmospheric reanalysis and high-resolution land models.
- **Spatial Resolution:** Reanalysis grid cell dynamically bilinearly interpolated to the exact geodetic coordinates (WGS 84) of each of the 4 Continuous Ambient Air Quality Monitoring Stations (CAAQMS).
- **Temporal Resolution:** Hourly continuous observations (1-hour time step) aligned to Indian Standard Time (IST = UTC+5:30).
- **Total Ingested Volume:** Exactly **131,424 station-hours** (4 stations $\times$ 32,856 hours from 2023-01-01 00:00 to 2026-09-30 23:00).
- **Missingness & Continuity:** **0 missing values (100.0% continuous coverage)** across all raw atmospheric variables.
- **License & Attribution:** Open-Meteo is provided under Creative Commons Attribution 4.0 International (CC BY 4.0). ECMWF ERA5 reanalysis is provided under the Copernicus Climate Change Service (C3S) Open Data terms.

---

## 2. Ingested Atmospheric Variables and Physical Valid Ranges

| Atmospheric Parameter | Symbol | Native Unit | Physical Range | Ingested Network Mean | Std Dev | Physical Role in Aerosol Dynamics |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **2m Temperature** | $T_{2\text{m}}$ | $^\circ\text{C}$ | $[-5.0, 60.0]$ | 25.90 | 4.64 | Governs thermal boundary-layer height and photochemical reaction rates. |
| **2m Relative Humidity** | $\text{RH}_{2\text{m}}$ | % | $[0.0, 100.0]$ | 62.56 | 21.89 | Drives hygroscopic particle growth, secondary aerosol formation, and fog. |
| **10m Wind Speed** | $U_{10\text{m}}$ | $\text{m/s}$ | $[0.0, 60.0]$ | 2.90 | 1.46 | Dictates mechanical ventilation and advective particulate dispersion. |
| **10m Wind Direction** | $\theta_{10\text{m}}$ | Degrees ($^\circ$) | $[0.0, 360.0]$ | 187.21 | 98.01 | Identifies downwind pollution transport corridors across the urban canopy. |
| **Precipitation** | $\text{Rain}$ | $\text{mm}$ | $[0.0, 500.0]$ | 0.11 | 0.57 | Drives wet scavenging and atmospheric washout of particulate mass. |
| **Surface Pressure** | $P_{\text{sfc}}$ | $\text{hPa}$ | $[800.0, 1100.0]$ | 946.62 | 4.57 | Barometric indicator of synoptic anticyclonic subsidence vs. cyclonic lifting. |

---

## 3. Mathematical Transformations and Meteorological Vector Formulations

### 3.1 Wind Direction Trigonometry and Velocity Components ($u, v$)
In meteorological convention, wind direction $\theta \in [0, 360^\circ)$ represents the direction **from which the wind is blowing**:
- $\theta = 0^\circ / 360^\circ$: Wind blowing from True North towards South.
- $\theta = 90^\circ$: Wind blowing from True East towards West.
- $\theta = 180^\circ$: Wind blowing from True South towards North.
- $\theta = 270^\circ$: Wind blowing from True West towards East.

To avoid artificial 0°/360° discontinuity during machine learning optimization, scalar direction is decomposed onto the unit circle:
$$\text{wind\_dir\_sin} = \sin\left(\frac{\pi \cdot \theta}{180}\right), \quad \text{wind\_dir\_cos} = \cos\left(\frac{\pi \cdot \theta}{180}\right)$$

Physical orthogonal wind velocity vectors ($u$, Eastward zonal velocity; $v$, Northward meridional velocity) are calculated as:
$$u = -U_{10\text{m}} \cdot \sin\left(\frac{\pi \cdot \theta}{180}\right)$$
$$v = -U_{10\text{m}} \cdot \cos\left(\frac{\pi \cdot \theta}{180}\right)$$

**Trigonometric Identity Verification:**
$$\sqrt{u^2 + v^2} = \sqrt{U_{10\text{m}}^2 \left( \sin^2\theta + \cos^2\theta \right)} = U_{10\text{m}}$$
Automated unit test `test_weather_wind_vector_identity` passed with maximum residual difference $< 10^{-6}\,\text{m/s}$.

### 3.2 Psychrometric Formulations & Vapor Pressure Deficit (VPD)
Atmospheric moisture is modeled using the Magnus-Tetens formulation:
1. **Saturation Vapor Pressure ($e_s$, $\text{hPa}$):**
   $$e_s(T) = 6.112 \cdot \exp\left( \frac{17.67 \cdot T_{2\text{m}}}{T_{2\text{m}} + 243.5} \right)$$
2. **Actual Vapor Pressure ($e_a$, $\text{hPa}$):**
   $$e_a = e_s(T) \cdot \left(\frac{\text{RH}_{2\text{m}}}{100.0}\right)$$
3. **Vapor Pressure Deficit ($\text{VPD}$, $\text{hPa}$):**
   $$\text{VPD} = \max\left(0.0, \; e_s(T) - e_a\right)$$
   *Scientific Purpose:* High VPD indicates dry air driving dust re-suspension; low VPD coupled with high humidity triggers secondary inorganic aerosol formation (sulfate/nitrate hygroscopic swelling).
4. **Dew Point Temperature ($T_{\text{dew}}$, $^\circ\text{C}$):**
   $$\alpha(T, \text{RH}) = \frac{17.67 \cdot T_{2\text{m}}}{243.5 + T_{2\text{m}}} + \ln\left(\frac{\text{RH}_{2\text{m}}}{100.0}\right)$$
   $$T_{\text{dew}} = \frac{243.5 \cdot \alpha(T, \text{RH})}{17.67 - \alpha(T, \text{RH})}$$
   The temperature-dewpoint depression $(T_{2\text{m}} - T_{\text{dew}})$ acts as a sensitive micro-climatic indicator of radiation inversion and surface condensation.

### 3.3 Boundary-Layer Stagnation & Ventilation Proxies
- **Atmospheric Stagnation Flag (`is_atmospheric_stagnation`):**
  $$\text{Stagnation} = \mathbb{I}\left( U_{10\text{m}} < 1.5\,\text{m/s} \; \land \; \text{Precipitation} \le 0.01\,\text{mm} \right)$$
  Identifies calms that trap vehicular and industrial primary emissions within the surface layer.
- **Ventilation Index Proxy (`ventilation_index_proxy`):**
  $$V_I = \frac{U_{10\text{m}} \cdot (T_{2\text{m}} + 273.15)}{100.0} \quad [\text{m}^2/\text{s}]$$

---

## 4. Point-in-Time Availability Protocol & Operational Forecast Boundaries

> [!IMPORTANT]
> **Strict Operational Distinction: Reanalysis vs. Numerical Weather Prediction (NWP)**
> Reanalysis datasets (ERA5) assimilate worldwide satellite and station observations *retrospectively* and are issued days or weeks after the valid timestamp. They represent the best available scientific historical state of the atmosphere, **not real-time operational weather forecasts**.

### 4.1 Non-Leakage Enforcement in Feature Engineering
To ensure the feature store is 100% operationally deployable and leak-free:
1. **Zero Future Weather Conditioning:** For a prediction made at origin timestamp $t$, the input predictor feature vector $\mathbf{x}_{i, t}$ conditions **strictly on meteorological observations up to and including time $t$ ($\le t$)**.
2. **Exclusion of Future Reanalysis ($t+h$):** Under no circumstances is the reanalysis weather at target timestamp $t+h$ ($h \in \{1, 6, 24, 48\}$) supplied as a feature in the primary feature store. Feeding future ERA5 values would simulate "perfect weather foresight" (oracle knowledge), which is impossible in real-world municipal deployment.
3. **Historical Weather Trends Only:** Weather features include past-only lags ($T_{t-1}, T_{t-24}, \text{RH}_{t-1}, U_{t-1}$) and rolling aggregations (rolling 6h/24h mean wind speed, rolling 6h/24h precipitation sums), all ending at $t$.

### 4.2 Retrospective vs. Prospective Experimental Protocol
- **Primary Operational Benchmark (This Implementation):** Features use strictly $t \le \text{issue\_time}$. Models learn to forecast PM2.5 based on current weather state, historical aerosol inertia, and localized road dynamics.
- **Retrospective Upper-Bound Benchmark (Future Milestone Ablation):** Any experiment evaluating model performance under "perfect weather foresight" must be explicitly labeled as an idealized offline retrospective ablation, segregated from operational comparisons against persistent and seasonal-naive baselines.

---

*Report generated automatically for Milestone 2: Multi-Source Feature Engineering.*
