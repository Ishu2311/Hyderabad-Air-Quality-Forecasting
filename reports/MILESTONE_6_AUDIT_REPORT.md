# MILESTONE 6 — COMPREHENSIVE SCIENTIFIC AUDIT REPORT
## Scientific Validation, Technical Audit, and Research Paper Readiness

**Project Title:** Traffic-Aware Spatio-Temporal $\text{PM}_{2.5}$ Concentration and Hotspot Forecasting in Hyderabad Using Multi-Source Environmental Data and Explainable Machine Learning  
**Milestone:** 6 — Scientific Validation & Research Paper Readiness Audit  
**Audit Stage:** Pre-Publication Final Forensic Verification  
**Execution Mode:** Read-Only Forensic Audit (Strict Zero-Code-Modification Policy Enforced)  
**Date of Audit:** November 15, 2026  
**Auditor:** Independent AI Research Systems Auditor (Google DeepMind Antigravity)  
**Status:** **APPROVED WITH MINOR REMEDIATIONS (64/64 Automated Tests Passing)**  

---

## Executive Summary

This independent scientific audit performs a forensic, evidence-based verification of the entire research pipeline across Milestones 1 through 5, assessing methodological rigor, code implementation, numerical precision, and research paper readiness. In strict accordance with the audit mandate, **no source code was modified, no models were retrained, and no experimental datasets were overwritten**. Every claim, metric, formula, and file link was audited directly against the physical Parquet stores, JSON prediction artifacts, Python source scripts, and automated test logs on disk.

### Core Audit Verdicts

1. **Milestone 5 Implementation Status: VERIFIED & COMPLETE.**  
   The Streamlit research dashboard (`new_research_project/dashboard/`) is fully implemented with strict decoupling between UI components (`components/`, `pages/`) and backend services (`backend/data_loader.py`, `backend/forecasting_service.py`, `backend/spatial_service.py`, `backend/explainability_service.py`). All mandatory scientific safeguards (sparse network warnings, retrospective reanalysis labels, 288.7% spatial transfer penalty, quarantine of unverified road proxies, and unit distinctions) are actively enforced.
2. **Test Suite Reconciliation: RECONCILED (64/64 Tests Passing).**  
   The apparent divergence between 56 tests (reported in Milestone 4) and 64 tests (reported in Milestone 5) is completely reconciled. The Milestone 4 suite comprised 56 unit and integration tests across 10 core modules. Milestone 5 introduced `test_dashboard.py`, contributing exactly 8 end-to-end integration tests ($56 + 8 = 64$ tests). The entire 64-test suite executes and passes in **23.18 seconds**.
3. **Sample Size Discrepancy (1,875 vs. 26,016 vs. 7,500): FORENSICALLY RESOLVED.**  
   - **26,016 test hours:** Represents the full domain cross-product ($6,504\text{ chronological test hours} \times 4\text{ monitoring stations} = 26,016\text{ station-hours}$ in `regression_predictions_test.parquet`).
   - **1,875 test hours:** Represents the subset of valid, unflagged physical target measurements at **Bollaram Industrial Area** at $+1\text{h}$ due to sensor downtime.
   - **7,500 evaluations:** A documentation artifact resulting from multiplying Bollaram's 1,875 count by 4. In reality, Central University had 5,743 valid test hours, ECIL Kapra had 5,849, and ICRISAT Patancheru had 5,606, yielding **19,073 actual evaluated station-hour predictions per horizon** (and 1,479 synchronous 4-station snapshots).
4. **Spatial Transfer Penalty Calculation: MATHEMATICALLY CORRECTED.**  
   The Milestone 4 report incorrectly stated that spatial error "increases by 388.7% (3.89x)". Mathematically, an increase from $\text{MAE}_{\text{point}} = 3.0106\,\mu\text{g/m}^3$ to $\text{MAE}_{\text{spatial}} = 11.7025\,\mu\text{g/m}^3$ represents an absolute increase of $8.6919\,\mu\text{g/m}^3$, which is a **$288.71\%$ relative increase** ($\approx 289\%$), or a ratio of **$3.887\times$** ($3.89\times$) the baseline error.
5. **Numerical Metric Verification: 100% MATCH WITH SOURCE FILES.**  
   Every regression metric, classification metric, SHAP attribution, and spatial uncertainty cell count reported across Milestones 3, 4, and 5 matches the underlying raw JSON and Parquet artifacts to within floating-point precision.
6. **Overall Scientific Soundness: HIGH.**  
   All empirical conclusions (IDW outperforming Kriging due to variogram under-identification on $N=4$; the 289% spatial transfer penalty; the 10-fold expansion of thermodynamic SHAP attribution; the quarantine of unverified road proxies) are firmly grounded in experimental evidence.

---

## 1. Forensic Audit Dimension Checks

### Dimension 1: Milestone 4 Report as Historical Evidence vs. Current Implementation
The Milestone 4 completion report was treated strictly as historical documentation. Forensic checks verified that the actual source tree in `new_research_project/implementation/` contains:
- `spatial/`: `idw.py`, `kriging.py`, `spatial_validation.py`, `hotspot_detection.py`, `spatial_metrics.py`, `spatial_visualization.py`.
- `explainability/`: `shap_analyzer.py`, `feature_importance.py`, `prediction_explainer.py`, `hotspot_narrative.py`.
- `data/predictions/`: All 25 Parquet, CSV, NPZ, and JSON output files generated by the master script `generate_milestone_4_artifacts.py`.
- `reports/figures/`: All 31 cartographic, diagnostic, and SHAP figures (300 DPI PNGs).
- **Finding:** The physical implementation strictly mirrors the described algorithms, with no phantom files or unexecuted stubs.

---

### Dimension 2: Milestone 5 Implementation Verification
Physical inspection of `new_research_project/dashboard/` confirmed complete implementation:
- **Presentation Layer:**
  - `pages/page_overview.py`: Executive summary, architecture diagram, station scorecard.
  - `pages/page_observations.py`: Historical multi-pollutant timeseries (2023–2026), diurnal profiles, uptime/downtime.
  - `pages/page_forecasting.py`: Actual vs. predicted Plotly timeseries, parity plots, residual histograms, dynamic KPI metric rows across horizons $h \in \{1, 6, 24, 48\}$.
  - `pages/page_spatial.py`: Interactive Mapbox scatter map, IDW continuous field grid, 3-tier uncertainty overlays, acute ($\ge 60$) and severe ($\ge 120$) hotspot KPI cards.
  - `pages/page_explainability.py`: TreeSHAP global importance bar charts, domain share donut charts, beeswarm/dependence plots, waterfall charts.
  - `pages/page_methodology.py`: Preprocessing protocols, 84 approved predictors table, 19 quarantined road proxies disclosure, 48-hour buffer purging, LOSOCV accounting.
  - `pages/page_about.py`: Problem statement, literature gap, research objectives, future roadmap.
- **Backend Service Layer:**
  - `backend/data_loader.py` (374 lines): Caching decorators (`@st.cache_data`), schema validation, date sorting.
  - `backend/forecasting_service.py` (203 lines): Multi-horizon slicing, real-time KPI metrics calculation (MAE, RMSE, $R^2$, Bias, MDA).
  - `backend/spatial_service.py` (223 lines): Bounding coordinates (`HYD_BOUNDS`), uncertainty tiers (`UNCERTAINTY_TIERS`), dynamic IDW continuous field computation.
  - `backend/explainability_service.py` (197 lines): Top feature rankings, category domain shares, contextual station narrative retrieval.
- **Components & Safeguards:**
  - `components/disclaimers.py`: Enforces 5 persistent scientific warning banners across the application.
  - `components/metrics_cards.py`: HTML metric cards with delta indicators.
  - `components/sidebar.py`: Navigation controller and dataset freshness badge.
- **Launchers:** Root launcher `new_research_project/app.py` successfully references `new_research_project/dashboard/app.py`.

---

### Dimension 3: Reconciliation of Test Suite Results (56 vs. 64 Tests)
The audit inspected all test files in `new_research_project/implementation/tests/`:

```
========================================================================================
TEST SUITE RECONCILIATION AUDIT
========================================================================================
Test Module                                      Milestone 4     Milestone 5     Current
----------------------------------------------------------------------------------------
test_explainability.py                                3               3             3
test_feature_engineering.py                         18              18            18
test_ingestion.py                                     7               7             7
test_leakage_safety.py                                2               2             2
test_missing_values.py                                5               5             5
test_modeling.py                                      6               6             6
test_partitioning.py                                  2               2             2
test_reproducibility.py                               1               1             1
test_spatial.py                                       8               8             8
test_target_generation.py                             4               4             4
----------------------------------------------------------------------------------------
Subtotal (Core Data Science Pipeline):               56              56            56
----------------------------------------------------------------------------------------
test_dashboard.py (Milestone 5 Integration):         --               8             8
----------------------------------------------------------------------------------------
TOTAL ACTIVE AUTOMATED TESTS:                        56              64            64
========================================================================================
```

- **Reconciliation Proof:** Running `python -m pytest new_research_project/implementation/tests -v` collects exactly 64 items and executes with **64 passed in 23.18 seconds**.
- **Verdict:** There is zero inconsistency. 56 tests validated the core ML and spatial pipeline in Milestone 4; 8 integration tests were added in Milestone 5 to validate dashboard data loading, caching, service slicing, and spatial uncertainty rendering.

---

### Dimension 4: Forensic Investigation of Sample Counts (1,875 vs. 26,016 vs. 7,500)
A deep query of `regression_predictions_test.parquet`, `feature_store_test_purged.parquet`, and `spatial_validation_results.json` resolved the exact mathematical origins of all three figures:

#### 1. Total Domain Test Partition Size: 26,016 Station-Hours
- Test date range: `2026-01-01 00:00:00` to `2026-09-28 23:00:00`.
- Number of hourly timestamps = **6,504 hours**.
- Number of CPCB monitoring stations = **4 stations**.
- Domain volume:
  $$\text{Domain Total} = 6,504 \times 4 = 26,016 \text{ station-hours}$$
- This matches the exact row count of `regression_predictions_test.parquet` ($N = 26,016$).

#### 2. Bollaram Physical Sensor Uptime: 1,875 Hours
Due to sensor maintenance outages and quality control flagging at Bollaram Industrial Area during 2026:
- Valid physical target measurements at $+1\text{h}$ = **1,875 hours**.
- Missing / flagged hours = **4,629 hours** ($71.2\%$ missingness).
- Under strict zero-target-imputation protocols, missing targets were preserved as `NaN` and excluded from loss evaluation, leaving exactly 1,875 evaluated instances at Bollaram.

#### 3. Actual Sample Accounting across All Four Stations
The other three stations operated with far higher uptime during 2026:

| Station Name | Total Chronological Hours | Valid Physical Targets (+1h) | Missing / Flagged Hours | Effective Sensor Uptime (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Bollaram Industrial Area** | 6,504 | **1,875** | 4,629 | 28.83% |
| **Central University** | 6,504 | **5,743** | 761 | 88.30% |
| **ECIL Kapra** | 6,504 | **5,849** | 655 | 89.93% |
| **ICRISAT Patancheru** | 6,504 | **5,606** | 898 | 86.19% |
| **Total Evaluated Predictions** | **26,016** | **19,073** | **6,943** | **73.31%** |

- **Synchronous 4-Station Snapshots ($N = 1,479$):** Exactly 1,479 timestamps had non-null targets across all four stations simultaneously.
- **The "7,500 Evaluations" Error:** In Table 2.1 of `SPATIAL_VALIDATION_REPORT.md`, the author erroneously copied `1,875` into all four station rows ($1,875 \times 4 = 7,500$). The actual number of evaluated station predictions in LOSOCV is **19,073**.

---

### Dimension 5: Verification of LOSOCV Evaluation Execution
Inspection of `new_research_project/implementation/spatial/spatial_validation.py` (lines 101–155) confirmed:
1. **Dynamic Donor Handling:** When holding out station $s_k$, donor stations $\mathcal{S}_{-k}$ are selected from those having valid physical targets at timestamp $t$. If an auxiliary donor is offline, IDW interpolates from the remaining $K \ge 2$ available stations.
2. **Complete Fold Isolation:** The held-out station's coordinates $(x_k, y_k)$ and target value $y_k(t)$ are strictly excluded from donor matrices before fitting the interpolator.
3. **Data Verification:** In `spatial_validation_results.json`, each station's record contains its exact empirical sample count ($N_{\text{bollaram}} = 1,875$, $N_{\text{central\_univ}} = 5,743$, $N_{\text{ecil}} = 5,849$, $N_{\text{icrisat}} = 5,606$), proving that the code executed on all 19,073 valid instances, and that the "7,500" figure was purely a report formatting mistake.

---

### Dimension 6: Mathematical Correction of the Spatial Error Increase
The Milestone 4 report asserted:
> *"Spatial transfer increases error by nearly 400% (388.7% / 3.89x)."*

This statement conflates the **error ratio** with the **relative percentage increase**:
- Point ML Forecast Error (+1h Stacking): $\text{MAE}_{\text{point}} = 3.0106\,\mu\text{g/m}^3$
- Spatial Transfer Error (+1h LOSOCV IDW $p=2$): $\text{MAE}_{\text{spatial}} = 11.7025\,\mu\text{g/m}^3$

#### Exact Mathematical Relations:
1. **Absolute Error Increase:**
   $$\Delta \text{MAE} = \text{MAE}_{\text{spatial}} - \text{MAE}_{\text{point}} = 11.7025 - 3.0106 = 8.6919\,\mu\text{g/m}^3$$
2. **Relative Percentage Increase:**
   $$\text{Relative Increase} = \frac{\text{MAE}_{\text{spatial}} - \text{MAE}_{\text{point}}}{\text{MAE}_{\text{point}}} \times 100\% = \frac{8.6919}{3.0106} \times 100\% = \mathbf{288.71\%}$$
3. **Ratio of Spatial Error to Baseline Error:**
   $$\text{Ratio} = \frac{\text{MAE}_{\text{spatial}}}{\text{MAE}_{\text{point}}} = \frac{11.7025}{3.0106} = \mathbf{3.887} \approx \mathbf{3.89\times}$$

- **Audit Correction:** The spatial transfer error is **$3.89\times$ the point forecast error**, representing an **increase of $288.7\%$** (not 388.7%). All research paper manuscripts and presentation slides must adopt this corrected wording.

---

### Dimension 7: Direct Verification of All Reported Metrics Against Source Outputs

#### A. Regression Metrics on 2026 Test Partition
Verified against `stacking_regression_metrics.json` and `regression_metrics.json`:

| Horizon | Model | Evaluated $N$ | MAE ($\mu\text{g/m}^3$) | RMSE ($\mu\text{g/m}^3$) | $R^2$ | Bias ($\mu\text{g/m}^3$) | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **+1h** | **Stacking Ensemble** | 19,048 | **3.0106** | **5.4546** | **0.9040** | **-0.0429** | **MATCH** |
| +1h | XGBoost | 19,048 | 3.0305 | 5.4444 | 0.9044 | +0.0185 | MATCH |
| +1h | LightGBM | 19,048 | 3.0386 | 5.4861 | 0.9029 | +0.0224 | MATCH |
| +1h | Random Forest | 19,048 | 3.0544 | 5.5892 | 0.8992 | +0.0620 | MATCH |
| +1h | Persistence Baseline | 19,048 | 3.1763 | 6.3965 | 0.8680 | +0.0088 | MATCH |
| +1h | Seasonal-Naive Baseline | 18,531 | 7.1605 | 11.3995 | 0.5710 | +0.1145 | MATCH |
| **+6h** | **Stacking Ensemble** | 18,940 | **5.9874** | **8.9348** | **0.7409** | **+1.4321** | **MATCH** |
| +6h | LightGBM | 18,940 | 6.0010 | 8.9301 | 0.7412 | +1.3234 | MATCH |
| +6h | Persistence Baseline | 18,940 | 7.4834 | 11.8604 | 0.5435 | +0.0402 | MATCH |
| **+24h** | **LightGBM** | 18,610 | **6.6682** | **9.7409** | **0.6867** | **+1.5405** | **MATCH** |
| +24h | Stacking Ensemble | 18,610 | 6.7405 | 9.7985 | 0.6830 | +1.8183 | MATCH |
| +24h | Persistence Baseline | 18,610 | 7.1895 | 11.4418 | 0.5677 | +0.0778 | MATCH |
| **+48h** | **LightGBM** | 18,411 | **7.9304** | **10.9466** | **0.5995** | **+2.5152** | **MATCH** |
| +48h | Stacking Ensemble | 18,411 | 8.0184 | 11.0087 | 0.5949 | +2.7055 | MATCH |
| +48h | Persistence Baseline | 18,411 | 8.2497 | 12.4991 | 0.4778 | +0.1248 | MATCH |

#### B. Classification Metrics on 2026 Test Partition
Verified against `classification_metrics.json`:
- **Acute Excursions ($\ge 60\,\mu\text{g/m}^3$) at +1h:**
  - LightGBM: $F_1 = 0.8250$, ROC-AUC = $0.9889$, PR-AUC = $0.9073$, Recall = $0.8388$, Precision = $0.8116$.
  - XGBoost: $F_1 = 0.8227$, ROC-AUC = $0.9892$, PR-AUC = $0.9071$, Recall = $0.8312$, Precision = $0.8142$.
  - Random Forest: $F_1 = 0.8125$, ROC-AUC = $0.9882$, PR-AUC = $0.9032$.
  - Persistence: $F_1 = 0.8088$, ROC-AUC = $0.8968$, PR-AUC = $0.6662$.
- **Acute Excursions ($\ge 60\,\mu\text{g/m}^3$) at +24h:**
  - Random Forest: $F_1 = 0.5760$, ROC-AUC = $0.9280$, PR-AUC = $0.5790$.
  - LightGBM: $F_1 = 0.5740$, ROC-AUC = $0.9280$, PR-AUC = $0.5533$.
  - XGBoost: $F_1 = 0.5698$, ROC-AUC = $0.9297$, PR-AUC = $0.5710$.
  - Persistence: $F_1 = 0.5431$, ROC-AUC = $0.7531$, PR-AUC = $0.3230$.
- **Severe Spikes ($\ge 120\,\mu\text{g/m}^3$):**
  - Extremely rare in 2026 test partition (prevalence $< 0.05\%$). $F_1 = 0.0000$ across all models, but XGBoost demonstrates strong probability ranking: ROC-AUC = $0.8149$ (+1h), $0.9116$ (+6h), and $0.9927$ (+24h).

#### C. TreeSHAP Attribution Shift
Verified against `global_feature_importance.json`:
- **+1h Forecast Horizon:** Total absolute SHAP = $17.20\,\mu\text{g/m}^3$.
  - Historical $\text{PM}_{2.5}$ Dynamics: **$79.59\%$** ($13.69\,\mu\text{g/m}^3$), led by unlagged `PM2.5` ($9.589\,\mu\text{g/m}^3$).
  - Atmospheric Thermodynamics: **$1.70\%$** ($0.29\,\mu\text{g/m}^3$).
  - Chemical Ratios: **$11.40\%$** ($1.96\,\mu\text{g/m}^3$).
- **+48h Forecast Horizon:** Total absolute SHAP = $20.55\,\mu\text{g/m}^3$.
  - Historical $\text{PM}_{2.5}$ Dynamics: **$49.26\%$** ($10.12\,\mu\text{g/m}^3$).
  - Atmospheric Thermodynamics: **$16.75\%$** ($3.44\,\mu\text{g/m}^3$) — **$985\%$ relative expansion ($9.85\times$)**.
  - Other Environmental (BLH, Solar): **$13.83\%$** ($2.84\,\mu\text{g/m}^3$).

#### D. Metropolitan Spatial Uncertainty Census (6,400 Grid Cells)
Verified against `hotspot_detection_summary.json`:
- Low Uncertainty ($d_{\min} \le 5\,\text{km}$): **1,075 cells (16.80%)**.
- Moderate Uncertainty ($5 < d_{\min} \le 15\,\text{km}$): **3,091 cells (48.30%)**.
- High Uncertainty / Extrapolation ($d_{\min} > 15\,\text{km}$): **2,234 cells (34.90%)**.
- Detected Hotspot Cells ($\ge 60\,\mu\text{g/m}^3$):
  - +1h: 2 cells ($0.03\%$), all within $\le 5\,\text{km}$ ($0.0\%$ extrapolation risk).
  - +6h: 0 cells ($0.00\%$), daylight boundary layer venting.
  - +24h: 1,435 cells ($22.42\%$), $51.2\%$ Low, $48.8\%$ Moderate, **$0.0\%$ High Extrapolation**.
  - +48h: 1,116 cells ($17.44\%$), $51.2\%$ Low, $48.8\%$ Moderate, **$0.0\%$ High Extrapolation**.

---

### Dimension 8: Experimental Evidence Supporting Scientific Conclusions

```
========================================================================================
EVIDENCE SUPPORT MATRIX FOR CORE SCIENTIFIC CONCLUSIONS
========================================================================================
Scientific Conclusion                    Claimed in Reports    Empirical Evidence on Disk    Verdict
----------------------------------------------------------------------------------------
1. IDW outperforms Ordinary Kriging      Yes (IDW MAE 11.70    Verified in spatial_           SUPPORTED
   on 4-station sparse network           vs Kriging 12.09)     validation_results.json
2. Spatial transfer error is ~3.9x       Yes (+288.7% / 3.89x  Verified (3.01 -> 11.70 ug/m3; SUPPORTED
   higher than point ML forecasting      increase over point)  LOSOCV on 19,073 instances)
3. Central University has massive        Yes (+12.0 ug/m3      Verified in spatial_           SUPPORTED
   positive bias under spatial transfer  positive bias)        validation_results.json
4. Bollaram Industrial Area suffers      Yes (-11.5 ug/m3      Verified in spatial_           SUPPORTED
   severe negative smoothing bias        negative bias)        validation_results.json
5. Eastern Hyderabad (ECIL) is           Yes (Negative R2 =    Verified (R2 = -1.527; 24 km   SUPPORTED
   spatially uncoupled from West         -1.527 at ECIL)       inter-station separation)
6. Atmospheric thermodynamics expands    Yes (1.7% -> 16.8%    Verified in global_feature_    SUPPORTED
   10-fold from +1h to +48h              attribution share)    importance.json (9.85x growth)
7. Stacking Ensemble wins at +1h/+6h;    Yes (Stacking wins    Verified in stacking_regres-   SUPPORTED
   LightGBM wins at +24h/+48h            h1/h6; LGB h24/h48)   sion_metrics.json & reg_m.json
8. High Extrapolation covers >30%        Yes (34.9% of grid    Verified in hotspot_           SUPPORTED
   of Greater Hyderabad domain           area > 15 km)         detection_summary.json
9. Quarantined road proxies are not      Yes (Excluded from    Verified in test_feature_      SUPPORTED
   measured OpenStreetMap data           primary 84 features)  engineering.py (18/18 pass)
========================================================================================
```

---

## 2. Forensic Issue Log and Recommended Corrections

| Issue ID | File Path & Approximate Location | Nature of Issue | Severity | Actual Evidence & Impact | Recommended Correction |
| :--- | :--- | :--- | :---: | :--- | :--- |
| **ISS-01** | `reports/SPATIAL_VALIDATION_REPORT.md`<br>(Table 2.1, lines 44–80) | Clerical Sample Count Duplication | **MAJOR** | Table lists `Samples (N): 1,875` for all 4 stations ($1,875 \times 4 = 7,500$). Actual counts: Bollaram (1,875), Central Univ (5,743), ECIL (5,849), ICRISAT (5,606), Total = 19,073. | Update Table 2.1 in manuscript to present station-specific sample counts and explain Bollaram's 28.8% sensor uptime. |
| **ISS-02** | `reports/SPATIAL_VALIDATION_REPORT.md`<br>`reports/MILESTONE_5_COMPLETION_REPORT.md` | Mathematical Percentage Wording Error | **MINOR** | Text states "error increases by 388.7% (3.89x)". Ratio is $3.887\times$, but the relative increase is $(11.7025 - 3.0106)/3.0106 = 288.71\%$. | Correct text to state: "spatial error increases by 288.7% (reaching 3.89x the point forecast error)". |
| **ISS-03** | `reports/IDW_VS_KRIGING_COMPARISON.md`<br>(Section 2.2, line 78) | Floating-Point Condition Number Precision | **COSMETIC** | Text cites condition numbers $\approx 1.5 \times 10^5$, but singular configurations with ill-conditioned variograms exceed $10^{12}$ (triggering `pinv`). | Clarify that well-conditioned distance matrices have $\kappa \approx 10^5$, but empirical variogram inversion frequently exceeds $10^{12}$. |
| **ISS-04** | `reports/SHAP_EXPLAINABILITY_REPORT.md`<br>(Section 1.2, line 58) | Undocumented RF Subsampling in Stacking SHAP | **MINOR** | For Stacking TreeSHAP at +1h/+6h, Random Forest base model was subsampled to 25 trees for polynomial runtime efficiency without explicit textual note in Section 1.2. | Add explanatory footnote in research paper methodology documenting that 25-tree subsampling preserves attribution rank correlation ($r_s > 0.99$). |
| **ISS-05** | `reports/HOTSPOT_INTERPRETATION_REPORT.md`<br>(Section 2.1, line 35) | Severe Spike Zero-Prevalence Disclosure | **MINOR** | In 2026 test partition, zero cells exceeded $120\,\mu\text{g/m}^3$ during evaluated snapshots. Text correctly notes 0 cells, but could more strongly emphasize season-dependent prevalence. | Add note clarifying that severe spikes ($\ge 120\,\mu\text{g/m}^3$) occur predominantly during winter inversion months (November–January). |

---

## 3. Summary Audit Verdict

- **Milestones 1 through 5 Audit Status:** **APPROVED WITH MINOR REMEDIATIONS**
- **Automated Test Suite Status:** **64/64 PASSING (100% SUCCESS RATE)**
- **Mathematical Integrity:** All core algorithms, data splits, and model evaluations are strictly leakage-free and reproducible.
- **Publication Readiness:** The empirical findings are of high scholarly quality and ready for academic conference/journal submission, subject to incorporating the minor documentation corrections identified above.
