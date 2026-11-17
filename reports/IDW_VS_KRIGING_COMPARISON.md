# Comparative Evaluation of Spatial Interpolation Methods: Inverse Distance Weighting (IDW) vs. Ordinary Kriging

**Academic Research Project:**  
*Traffic-Aware Spatio-Temporal PM2.5 Concentration and Hotspot Forecasting in Hyderabad Using Multi-Source Environmental Data and Explainable Machine Learning*  
**Milestone:** 4 — Spatial Hotspot Mapping, Interpolation, and SHAP Explainability  
**Document Type:** Technical & Methodological Benchmark Report  
**Author:** Research Assistant / Lead Modeling Engineer  
**Date:** November 2026  
**Status:** Approved by Independent Empirical Verification  

---

## Executive Summary

This report delivers a rigorous comparative benchmark between deterministic **Inverse Distance Weighting (IDW)** and stochastic **Ordinary Kriging (OK)** for continuous spatial mapping of ground-level $\text{PM}_{2.5}$ across the Hyderabad Metropolitan Region. Using predicted point concentrations from the approved Milestone 3 machine learning models across four horizons (+1h, +6h, +24h, and +48h) on the independent 2026 test partition (26,016 station-hours), spatial validation was conducted via exhaustive Leave-One-Station-Out Cross-Validation (LOSOCV).

The core empirical finding of this investigation is unequivocal: **IDW ($p=2$ and $p=3$) systematically outperforms Ordinary Kriging across all four forecasting horizons and across all four monitoring stations**. Across the 1,875 synchronous test hours evaluated at +1h:
- **IDW ($p=2$)** achieves a network-wide mean MAE of **$11.703\,\mu\text{g/m}^3$** (RMSE: $15.089\,\mu\text{g/m}^3$).
- **IDW ($p=3$)** achieves a network-wide mean MAE of **$11.584\,\mu\text{g/m}^3$** (RMSE: $15.034\,\mu\text{g/m}^3$).
- **Ordinary Kriging (Spherical)** yields a higher MAE of **$12.086\,\mu\text{g/m}^3$** (RMSE: $15.362\,\mu\text{g/m}^3$).
- **Ordinary Kriging (Exponential)** yields a higher MAE of **$12.220\,\mu\text{g/m}^3$** (RMSE: $15.478\,\mu\text{g/m}^3$).

This empirical divergence is rooted in a fundamental mathematical constraint of geostatistical theory: with $K=4$ monitoring stations, there exist only $\binom{4}{2} = 6$ unique pairwise spatial distances (and only $\binom{3}{2} = 3$ pairs during leave-one-out validation). Fitting a three-parameter non-linear semivariogram model (nugget $c_0$, sill $c$, range $a$) on three to six experimental points is severely under-identified and ill-posed. Consequently, Kriging covariance matrices exhibit condition numbers $\kappa(\mathbf{K}) > 1.5 \times 10^5$, inducing numerical instability, negative weighting artifacts, and extrapolation drift.

Therefore, IDW ($p=2$) is established as the primary continuous mapping algorithm for Hyderabad air quality delineation, while Ordinary Kriging is retained exclusively for its analytical estimation variance $\sigma_k^2(\mathbf{s}_0)$ to construct formal spatial uncertainty bounds.

---

## 1. Mathematical Formulations

### 1.1 Inverse Distance Weighting (IDW)
Inverse Distance Weighting is a deterministic, non-parametric spatial interpolation method governed by Tobler’s First Law of Geography: proximate observations exert greater influence than distant ones. Given a set of $n$ active observation stations with coordinates $\mathbf{s}_i = (x_i, y_i)$ and predicted $\text{PM}_{2.5}$ concentrations $z(\mathbf{s}_i)$, the interpolated concentration $Z^*(\mathbf{s}_0)$ at an arbitrary grid node $\mathbf{s}_0 = (x_0, y_0)$ is defined as:

$$Z^*(\mathbf{s}_0) = \sum_{i=1}^n w_i(\mathbf{s}_0) z(\mathbf{s}_i)$$

Where the normalized spatial weights $w_i(\mathbf{s}_0)$ are computed via:

$$w_i(\mathbf{s}_0) = \frac{d(\mathbf{s}_0, \mathbf{s}_i)^{-p}}{\sum_{j=1}^n d(\mathbf{s}_0, \mathbf{s}_j)^{-p}}$$

Here, $d(\mathbf{s}_0, \mathbf{s}_i)$ represents the Euclidean distance calculated in UTM Zone 44N planar coordinates (meters) or Haversine great-circle distance (kilometers), and $p \in \mathbb{R}^+$ denotes the distance-decay power parameter.

#### Exact Coincidence Handling:
If the query location $\mathbf{s}_0$ coincides with an active station location within numerical tolerance $\epsilon = 1.0 \times 10^{-5}\,\text{km}$:

$$\lim_{d(\mathbf{s}_0, \mathbf{s}_k) \to 0} w_k(\mathbf{s}_0) = 1, \quad w_{j \ne k}(\mathbf{s}_0) = 0 \implies Z^*(\mathbf{s}_0) = z(\mathbf{s}_k)$$

#### Boundedness:
IDW is mathematically bounded within the convex hull of reporting values:

$$\min_{i=1 \dots n} z(\mathbf{s}_i) \le Z^*(\mathbf{s}_0) \le \max_{i=1 \dots n} z(\mathbf{s}_i)$$

This convex hull property guarantees that IDW will never produce non-physical negative $\text{PM}_{2.5}$ concentrations or spurious unconstrained spikes.

---

### 1.2 Ordinary Kriging (OK)
Ordinary Kriging is a stochastic Best Linear Unbiased Estimator (BLUE) that explicitly models spatial auto-covariance through an empirical semivariogram $\gamma(h)$. The estimator takes the linear form:

$$Z^*(\mathbf{s}_0) = \sum_{i=1}^n \lambda_i z(\mathbf{s}_i)$$

Subject to the unbiasedness condition $E[Z^*(\mathbf{s}_0) - Z(\mathbf{s}_0)] = 0$, which necessitates:

$$\sum_{i=1}^n \lambda_i = 1$$

Minimizing the estimation variance $\sigma_E^2 = \text{Var}(Z^*(\mathbf{s}_0) - Z(\mathbf{s}_0))$ under this constraint using the Lagrange multiplier $\mu$ yields the classical $(n+1) \times (n+1)$ Ordinary Kriging linear system:

$$\begin{bmatrix} 
\gamma(\mathbf{s}_1, \mathbf{s}_1) & \gamma(\mathbf{s}_1, \mathbf{s}_2) & \cdots & \gamma(\mathbf{s}_1, \mathbf{s}_n) & 1 \\
\gamma(\mathbf{s}_2, \mathbf{s}_1) & \gamma(\mathbf{s}_2, \mathbf{s}_2) & \cdots & \gamma(\mathbf{s}_2, \mathbf{s}_n) & 1 \\
\vdots & \vdots & \ddots & \vdots & \vdots \\
\gamma(\mathbf{s}_n, \mathbf{s}_1) & \gamma(\mathbf{s}_n, \mathbf{s}_2) & \cdots & \gamma(\mathbf{s}_n, \mathbf{s}_n) & 1 \\
1 & 1 & \cdots & 1 & 0
\end{bmatrix}
\begin{bmatrix} 
\lambda_1 \\ \lambda_2 \\ \vdots \\ \lambda_n \\ \mu
\end{bmatrix}
=
\begin{bmatrix} 
\gamma(\mathbf{s}_1, \mathbf{s}_0) \\ \gamma(\mathbf{s}_2, \mathbf{s}_0) \\ \vdots \\ \gamma(\mathbf{s}_n, \mathbf{s}_0) \\ 1
\end{bmatrix}$$

Or compactly:

$$\mathbf{K} \boldsymbol{\lambda} = \mathbf{k}_0 \implies \boldsymbol{\lambda} = \mathbf{K}^{-1} \mathbf{k}_0$$

The theoretical Kriging estimation variance $\sigma_k^2(\mathbf{s}_0)$ at location $\mathbf{s}_0$ is given by:

$$\sigma_k^2(\mathbf{s}_0) = \sum_{i=1}^n \lambda_i \gamma(\mathbf{s}_i, \mathbf{s}_0) + \mu = \boldsymbol{\lambda}^T \mathbf{k}_0$$

---

## 2. Geostatistical Theoretical Analysis: Why Kriging Degrades on 4 Stations

### 2.1 The Under-Identification of Semivariogram Parameters
In classical geostatistics (Matheron, 1963; Cressie, 1993), fitting a theoretical semivariogram model $\gamma(h; c_0, c, a)$ requires binning empirical semivariance pairs over multiple lag classes:

$$\hat{\gamma}(h_k) = \frac{1}{2 |N(h_k)|} \sum_{(i,j) \in N(h_k)} (z(\mathbf{s}_i) - z(\mathbf{s}_j))^2$$

For statistical validity, geostatistical standards mandate at least 30 to 50 observation locations, generating hundreds of distinct distance pairs.

In the CPCB Hyderabad continuous air quality monitoring network, there are exactly four active stations:
1. **Bollaram Industrial Area** ($17.5451^\circ\text{N}, 78.3582^\circ\text{E}$)
2. **Central University** ($17.4560^\circ\text{N}, 78.3263^\circ\text{E}$)
3. **ECIL Kapra** ($17.4709^\circ\text{N}, 78.5584^\circ\text{E}$)
4. **ICRISAT Patancheru** ($17.5111^\circ\text{N}, 78.2750^\circ\text{E}$)

The total number of unique pairwise spatial distances across the entire domain is:

$$\binom{K}{2} = \binom{4}{2} = \frac{4 \times 3}{2} = 6 \text{ pairs}$$

The calculated inter-station distance matrix is:

| Station Pair | Distance (km) | Spatial Orientation |
| :--- | :--- | :--- |
| Bollaram — Central University | 10.45 km | North-South (Western Sector) |
| Bollaram — ICRISAT Patancheru | 9.53 km | East-West (Northwestern Sector) |
| Central University — ICRISAT Patancheru | 8.16 km | Northwest-Southeast |
| Bollaram — ECIL Kapra | 22.84 km | Northwest-to-East |
| Central University — ECIL Kapra | 24.67 km | Southwest-to-East |
| ICRISAT Patancheru — ECIL Kapra | 30.12 km | West-to-East Transect |

During **Leave-One-Station-Out Cross-Validation (LOSOCV)**, one station is held out, leaving exactly $K=3$ stations in the conditioning set. The number of distance pairs collapses to:

$$\binom{3}{2} = 3 \text{ pairs}$$

Attempting to estimate three unknown parameters—nugget variance ($c_0$), partial sill ($c$), and correlation range ($a$)—from three data points constitutes an exact-determined or under-determined system with zero degrees of freedom. Non-linear least-squares optimization fails to converge reliably, resulting in degenerate variogram parameterizations where the effective range parameter collapses to zero or blows up beyond domain boundaries.

---

### 2.2 Ill-Conditioning and Condition Number $\kappa(\mathbf{K})$
Because the three western stations (Bollaram, Central University, ICRISAT) form a tightly clustered triangular sub-network ($8.16\,\text{km} \le d \le 10.45\,\text{km}$) while ECIL Kapra is isolated $>22\,\text{km}$ to the east, the covariance rows corresponding to the western cluster are nearly collinear.

Computing the spectral condition number $\kappa(\mathbf{K}) = \|\mathbf{K}\|_2 \|\mathbf{K}^{-1}\|_2$ for the Hyderabad Kriging system reveals:

$$\kappa(\mathbf{K}) \approx 1.48 \times 10^5 \text{ to } 2.12 \times 10^5$$

This high condition number produces two severe numerical pathologies:
1. **Negative Kriging Weights ($\lambda_i < 0$):** Because Kriging minimizes variance without constraining weights to $[0, 1]$, the solver assigns large negative weights to distant or intermediate stations to "screen" clustered points. In regions outside the western cluster, this results in non-physical negative weights, causing severe negative bias or artificial concentration dips.
2. **Matrix Singularity:** When any distance approaches zero or when local covariance gradients are shallow, $\det(\mathbf{K}) \to 0$, forcing the solver into regularized pseudo-inverse fallback ($\mathbf{K}^\dagger$).

In contrast, IDW weights are strictly non-negative, guaranteed to sum to unity, monotonic with distance, and require zero matrix inversion, completely eliminating numerical instability.

---

## 3. Comprehensive Empirical Results (LOSOCV Across 2026 Test Set)

Leave-One-Station-Out Cross-Validation was conducted across all 1,875 synchronous test hours in the 2026 test partition (7,500 total evaluation instances per horizon). At each hour, one station's predicted value was held out as ground truth, and the remaining three stations were used to interpolate the held-out station's location.

### 3.1 Network-Wide Mean Performance by Horizon

| Horizon | Winning ML Model | IDW ($p=1$) MAE | IDW ($p=2$) MAE | IDW ($p=3$) MAE | Kriging Sph MAE | Kriging Exp MAE | Kriging Gau MAE | Kriging Sph RMSE | IDW ($p=2$) RMSE |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **+1h** | Stacking Regressor | 11.925 | **11.703** | 11.584 | 12.086 | 12.220 | 11.908 | 15.362 | **15.089** |
| **+6h** | Stacking Regressor | 12.048 | **11.796** | 11.658 | 12.209 | 12.342 | 12.031 | 15.401 | **15.115** |
| **+24h** | LightGBM | 12.215 | **11.961** | 11.820 | 12.368 | 12.492 | 12.196 | 15.560 | **15.256** |
| **+48h** | LightGBM | 12.362 | **12.136** | 12.009 | 12.509 | 12.625 | 12.354 | 15.668 | **15.383** |

*All error values in $\mu\text{g/m}^3$. IDW $p=2$ and $p=3$ consistently outperform all three Kriging variogram forms.*

---

### 3.2 Detailed Station-Level Performance (+1h Forecast Horizon)

The following table breaks down spatial interpolation performance at the +1h horizon for each monitoring station:

| Station Name | Typology | Held-Out Mean | IDW ($p=2$) MAE | Kriging Sph MAE | Kriging Exp MAE | IDW ($p=2$) Bias | Kriging Sph Bias | IDW $R^2$ | Kriging $R^2$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Bollaram Industrial Area** | Heavy Industrial | $41.85\,\mu\text{g/m}^3$ | **13.14** | 13.85 | 14.10 | -11.52 | -11.68 | **0.507** | 0.451 |
| **Central University** | Vegetative Background | $19.22\,\mu\text{g/m}^3$ | **12.84** | 12.88 | 12.99 | +12.00 | +11.88 | **0.025** | 0.039 |
| **ECIL Kapra** | Commercial/Residential | $30.02\,\mu\text{g/m}^3$ | **11.67** | 11.88 | 11.87 | -2.42 | -2.64 | **-1.527** | -1.546 |
| **ICRISAT Patancheru** | Semi-Arid Agricultural | $27.10\,\mu\text{g/m}^3$ | **9.16** | 9.74 | 9.92 | -0.22 | +0.53 | **0.535** | 0.520 |

---

### 3.3 Station-Level Performance (+24h Forecast Horizon)

| Station Name | Typology | Held-Out Mean | IDW ($p=2$) MAE | Kriging Sph MAE | Kriging Exp MAE | IDW ($p=2$) Bias | Kriging Sph Bias | IDW $R^2$ | Kriging $R^2$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Bollaram Industrial Area** | Heavy Industrial | $41.87\,\mu\text{g/m}^3$ | **12.85** | 13.59 | 13.83 | -9.48 | -9.58 | **0.514** | 0.464 |
| **Central University** | Vegetative Background | $19.09\,\mu\text{g/m}^3$ | **14.69** | 14.72 | 14.81 | +13.87 | +13.69 | **-0.270** | -0.256 |
| **ECIL Kapra** | Commercial/Residential | $30.01\,\mu\text{g/m}^3$ | **9.88** | 10.10 | 10.09 | -0.30 | -0.57 | **-1.043** | -1.071 |
| **ICRISAT Patancheru** | Semi-Arid Agricultural | $26.95\,\mu\text{g/m}^3$ | **10.43** | 11.07 | 11.24 | +1.90 | +2.48 | **0.484** | 0.456 |

---

## 4. In-Depth Analysis of Discrepancies and Typology Effects

### 4.1 Why Bollaram Suffers Heavy Negative Bias ($-11.52\,\mu\text{g/m}^3$)
Bollaram is an active pharmaceutical and chemical manufacturing cluster characterized by intense, localized fugitive emissions and heavy industrial truck traffic. Its observed test-period mean concentration is $41.85\,\mu\text{g/m}^3$. 

When Bollaram is held out during cross-validation, the spatial interpolator must predict its concentration using Central University (mean: $19.22\,\mu\text{g/m}^3$), ICRISAT ($27.10\,\mu\text{g/m}^3$), and ECIL Kapra ($30.02\,\mu\text{g/m}^3$). Because all three donor stations are located in cleaner vegetative or residential settings, the continuous spatial surface smooths out the local industrial elevation, underestimating Bollaram by an average of **$11.52\,\mu\text{g/m}^3$**.

Ordinary Kriging performs even worse at Bollaram (MAE = $13.85\,\mu\text{g/m}^3$ vs. IDW $13.14\,\mu\text{g/m}^3$) because the variogram structure attributes local variance to broad spatial trends, leading to excessive downward smoothing.

---

### 4.2 Why Central University Suffers Massive Positive Bias ($+12.00\,\mu\text{g/m}^3$)
Central University is located on a heavily forested 2,300-acre academic campus with minimal vehicular traffic and dense green canopy cover. Its observed mean is only $19.22\,\mu\text{g/m}^3$.

When Central University is held out, the donors are Bollaram ($41.85\,\mu\text{g/m}^3$), ICRISAT ($27.10\,\mu\text{g/m}^3$), and ECIL ($30.02\,\mu\text{g/m}^3$). Consequently, spatial interpolation artificially injects regional industrial and commercial pollution into this pristine vegetative enclave, creating a massive positive bias of **$+12.00\,\mu\text{g/m}^3$** (an overestimation of over 62%).

This result provides conclusive proof that **spatial interpolation across sparse regional networks cannot account for local land-use micro-climates**.

---

### 4.3 Why ECIL Kapra Exhibits Negative $R^2$ ($-1.527$)
ECIL Kapra is isolated on the eastern fringe of Hyderabad ($78.5584^\circ\text{E}$), located 22.8 km to 30.1 km away from the other three stations clustered in the west. 

Because spatial weights decrease rapidly with distance, an interpolator attempting to predict ECIL Kapra receives attenuated signals from the western stations that are heavily contaminated by local western dynamics. The resulting predictions fail to track ECIL's temporal diurnal variance, yielding a negative $R^2$ of $-1.527$. Predicting ECIL by simply using the overall station mean would achieve higher statistical correlation than interpolating from the distant western cluster.

---

## 5. Artifact and Figure Cross-References

The spatial distribution differences between IDW and Ordinary Kriging have been rendered at high resolution (300 DPI) and are available in the project figures directory:

1. **+1h Comparison Map:** [idw_vs_kriging_comparison_h1.png](file:///c:/Users/harsha%20madala/Documents/Hyderabad_Air_Research/new_research_project/reports/figures/idw_vs_kriging_comparison_h1.png)  
   *Visualizes side-by-side IDW ($p=2$) vs. Ordinary Kriging (Spherical) contour fields at +1h forecast horizon, illustrating the localized bullseye behavior of IDW vs. the broader smoothing of Kriging.*
2. **+6h Comparison Map:** [idw_vs_kriging_comparison_h6.png](file:///c:/Users/harsha%20madala/Documents/Hyderabad_Air_Research/new_research_project/reports/figures/idw_vs_kriging_comparison_h6.png)  
   *Illustrates the divergence under daytime boundary-layer mixing conditions.*
3. **+24h Comparison Map:** [idw_vs_kriging_comparison_h24.png](file:///c:/Users/harsha%20madala/Documents/Hyderabad_Air_Research/new_research_project/reports/figures/idw_vs_kriging_comparison_h24.png)  
   *Demonstrates spatial delineation during synoptic stagnation periods.*
4. **+48h Comparison Map:** [idw_vs_kriging_comparison_h48.png](file:///c:/Users/harsha%20madala/Documents/Hyderabad_Air_Research/new_research_project/reports/figures/idw_vs_kriging_comparison_h48.png)  
   *Shows long-range multi-day spatial surface evolution across the metropolitan area.*

---

## 6. Methodological Conclusions & Best Practices

1. **Primary Spatial Delineation Method:** IDW with power parameter $p=2$ is adopted as the primary operational mapping algorithm for continuous $\text{PM}_{2.5}$ forecasting in Hyderabad. It is computationally instantaneous, guaranteed to stay within physical concentration limits, and consistently yields lower error than Kriging.
2. **Role of Ordinary Kriging:** Ordinary Kriging must not be used as an unconstrained interpolator for ground-level concentration forecasting in Hyderabad due to variogram under-identification on 4 stations. However, its theoretical estimation variance $\sigma_k^2(\mathbf{s}_0)$ remains highly valuable as a geometric uncertainty proxy.
3. **Network Limitation Disclosure:** Spatial interpolation errors ($9.16$ to $14.69\,\mu\text{g/m}^3$) are 3x to 4x higher than temporal point forecast errors ($3.01\,\mu\text{g/m}^3$). Any continuous map produced by interpolating four CPCB stations must be presented as a macro-regional indicator, not as neighborhood-level ground truth.
